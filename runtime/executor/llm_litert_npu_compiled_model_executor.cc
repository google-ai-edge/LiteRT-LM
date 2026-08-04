// Copyright 2025 The ODML Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "runtime/executor/llm_litert_npu_compiled_model_executor.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/log/log.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/numbers.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/str_split.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"  // from @litert
#include "litert/c/litert_model_types.h"  // from @litert
#include "litert/c/litert_op_code.h"  // from @litert
#include "litert/cc/internal/litert_extended_model.h"  // from @litert
#include "litert/cc/litert_common.h"  // from @litert
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_expected.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_model_types.h"  // from @litert
#include "litert/cc/litert_options.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/cc/litert_tensor_buffer_types.h"  // from @litert
#include "support/util/status_macros.h"  // from @litert
#include "tflite/types/half.h"  // from @litert
#if defined(__ANDROID__)
#include "litert/cc/options/litert_google_tensor_options.h"  // from @litert
#include "litert/cc/options/litert_qualcomm_options.h"  // from @litert
#endif

#include "runtime/components/embedding_lookup/embedding_lookup_manager.h"
#include "runtime/components/logits_processor/logits_processor.h"
#include "runtime/components/model_resources.h"
#include "runtime/executor/litert/legacy_map_state.h"
#include "runtime/executor/litert_compiled_model_executor_utils.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/executor/llm_executor_processed_tokens.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/executor/llm_litert_npu_compiled_model_executor_utils.h"
#include "runtime/executor/llm_processed_context.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/status_macros.h"  // NOLINT
#include "runtime/util/tensor_buffer_util.h"

namespace litert::lm {

namespace {
using ::litert::CompiledModel;
using ::litert::Environment;
using ::litert::Model;
using ::litert::TensorBuffer;

constexpr int kInvalidTokenId = -1;

absl::Status FillKVCacheBuffer(TensorBuffer& buffer, int64_t init_value) {
  LITERT_ASSIGN_OR_RETURN(RankedTensorType tensor_type, buffer.TensorType());
  LITERT_ASSIGN_OR_RETURN(auto size, buffer.PackedSize());
  LITERT_ASSIGN_OR_RETURN(
      auto lock, ::litert::TensorBufferScopedLock::Create(
                     buffer, ::litert::TensorBuffer::LockMode::kWrite));

  auto element_type = tensor_type.ElementType();
  if (element_type == ::litert::ElementType::Int16) {
    auto* ptr = static_cast<int16_t*>(lock.second);
    std::fill(ptr, ptr + size / sizeof(int16_t),
              static_cast<int16_t>(init_value));
  } else if (element_type == ::litert::ElementType::UInt16) {
    auto* ptr = static_cast<uint16_t*>(lock.second);
    std::fill(ptr, ptr + size / sizeof(uint16_t),
              static_cast<uint16_t>(init_value));
  } else if (element_type == ::litert::ElementType::Int8) {
    auto* ptr = static_cast<int8_t*>(lock.second);
    std::fill(ptr, ptr + size / sizeof(int8_t),
              static_cast<int8_t>(init_value));
  } else if (element_type == ::litert::ElementType::UInt8) {
    auto* ptr = static_cast<uint8_t*>(lock.second);
    std::fill(ptr, ptr + size / sizeof(uint8_t),
              static_cast<uint8_t>(init_value));
  } else {
    std::memset(lock.second, 0, size);
  }
  return absl::OkStatus();
}

constexpr absl::string_view kPrefillSignatureBase = "prefill";
constexpr absl::string_view kPrefillEmbedderBase = "prefill_embedder";
constexpr absl::string_view kPrefillEmbedderPerLayerBase =
    "prefill_per_layer_embedder";
constexpr absl::string_view kPrefillMaskBase = "prefill_mask";
constexpr absl::string_view kPrefillRopeBase = "prefill_rope";
constexpr absl::string_view kPrefillCacheUpdateBase = "prefill_cache_update";
constexpr char kDecodeSignature[] = "decode";
constexpr char kDecodeEmbedder[] = "decode_embedder";
constexpr char kDecodeEmbedderPerLayer[] = "decode_per_layer_embedder";
constexpr char kDecodeMask[] = "decode_mask";
constexpr char kDecodeRope[] = "decode_rope";
constexpr char kDecodeCacheUpdate[] = "decode_cache_update";

constexpr char kVerifySignature[] = "verify";
constexpr char kVerifyEmbedder[] = "verify_embedder";
constexpr char kVerifyEmbedderPerLayer[] = "verify_per_layer_embedder";
constexpr char kVerifyMask[] = "verify_mask";
constexpr char kVerifyRope[] = "verify_rope";
constexpr char kVerifyCacheUpdate[] = "verify_cache_update";
constexpr char cache_k31[] = "kv_cache_k_31";
constexpr char cache_k25[] = "kv_cache_k_25";
constexpr char cache_v25[] = "kv_cache_v_25";
constexpr char cache_k19[] = "kv_cache_k_19";
constexpr char cache_v19[] = "kv_cache_v_19";
constexpr char cache_k23[] = "kv_cache_k_23";
constexpr char cache_v23[] = "kv_cache_v_23";
constexpr char cache_k17[] = "kv_cache_k_17";
constexpr char cache_v17[] = "kv_cache_v_17";

constexpr absl::string_view kv_cache_k_root_name = "kv_cache_k_";
constexpr absl::string_view kv_cache_v_root_name = "kv_cache_v_";
constexpr absl::string_view kv_cache_c_root_name = "kv_cache_c_";

constexpr absl::string_view kv_cache_slice_k_root_name = "kv_slice_k_";
constexpr absl::string_view kv_cache_slice_v_root_name = "kv_slice_v_";
constexpr absl::string_view kv_cache_slice_c_root_name = "kv_slice_c_";

// Detect if the model uses Sliding Window Attention (SWA) by checking if
// there are different KV cache sizes (mixed local/global attention).
//
// TODO(b/532090618): Remove this heuristic once the engine provides this via
// metadata to the executor. I.e. new litertlm models will have this
// information embedded and the engine passes it down.
bool DetectIsSwa(const absl::flat_hash_map<absl::string_view, TensorBuffer>&
                     input_kv_cache_buffers) {
  absl::flat_hash_set<int64_t> cache_seqs;
  for (const auto& [name, buffer] : input_kv_cache_buffers) {
    if (name.starts_with(kv_cache_k_root_name) ||
        name.starts_with(kv_cache_v_root_name) ||
        name.starts_with(kv_cache_c_root_name)) {
      auto tensor_type_expected = buffer.TensorType();
      if (tensor_type_expected.HasValue()) {
        auto dims = tensor_type_expected->Layout().Dimensions();
        int rank = dims.size();
        if (rank >= 2) {
          int last_dim = dims[rank - 1];
          int second_last_dim = dims[rank - 2];
          int64_t cache_seq = std::max(last_dim, second_last_dim);
          cache_seqs.insert(cache_seq);
        }
      }
    }
  }
  bool is_swa = cache_seqs.size() > 1;
  return is_swa;
}

namespace {

using LogitsQuantizationParams =
    LlmLiteRtNpuCompiledModelExecutor::LogitsQuantizationParams;
using TensorBufferMap = LlmLiteRtNpuCompiledModelExecutor::TensorBufferMap;
using TensorBufferMapBySize =
    LlmLiteRtNpuCompiledModelExecutor::TensorBufferMapBySize;
using ResolvedSignatures =
    LlmLiteRtNpuCompiledModelExecutor::ResolvedSignatures;

// Builds a prefill signature name from its base and the prefill length, e.g.
// PrefillSig("prefill_mask", 256) -> "prefill_mask_256".
std::string PrefillSig(absl::string_view base, int prefill_size) {
  return absl::StrCat(base, "_", prefill_size);
}

bool DetectHasSuffix(const litert::Model& llm_model) {
  auto signatures = llm_model.GetSignatures();
  if (signatures) {
    for (const auto& signature : *signatures) {
      absl::string_view key = signature.Key();
      if (absl::StartsWith(key, "decode_cache_")) {
        absl::string_view suffix = key.substr(sizeof("decode_cache_") - 1);
        int size = 0;
        if (absl::SimpleAtoi(suffix, &size) && size > 0) {
          return true;
        }
      }
    }
  }
  return false;
}

bool DetectHasRopeSuffix(ModelResources& resources) {
  auto aux_model_or = resources.GetTFLiteModel(ModelType::kTfLiteAux);
  if (aux_model_or.ok()) {
    auto aux_signatures = (*aux_model_or)->GetSignatures();
    if (aux_signatures) {
      for (const auto& signature : *aux_signatures) {
        absl::string_view key = signature.Key();
        if (absl::StartsWith(key, "prefill_rope_") &&
            absl::StrContains(key, "_cache_")) {
          return true;
        }
      }
    }
  }
  return false;
}

// Detects all prefill lengths supported by the transformer model by
// scanning its signatures for prefill signatures named "prefill_<N>..."
// and returns a sorted list of unique sizes. Falls back to probing a list
// of common sizes if the model does not expose enumerable signatures.
absl::StatusOr<std::vector<int>> DetectPrefillSizes(
    const litert::Model& transformer_model) {
  const std::string prefix = absl::StrCat(kPrefillSignatureBase, "_");
  auto signatures = transformer_model.GetSignatures();
  absl::flat_hash_set<int> unique_sizes;
  if (signatures) {
    for (const auto& signature : *signatures) {
      absl::string_view key = signature.Key();
      if (!absl::StartsWith(key, prefix)) {
        continue;
      }
      absl::string_view suffix = key.substr(prefix.size());
      std::vector<absl::string_view> parts = absl::StrSplit(suffix, '_');
      if (parts.empty()) {
        continue;
      }
      // Only the bare LLM prefill signature has a numeric first segment after
      // the "prefill_" prefix (e.g., "prefill_128_cache_640"). Auxiliary
      // signatures (prefill_mask_128_..., etc.) start with a non-numeric
      // segment.
      int prefill_size = 0;
      if (absl::SimpleAtoi(parts[0], &prefill_size) && prefill_size > 0) {
        unique_sizes.insert(prefill_size);
      }
    }
  }
  // Fallback: probe a list of common prefill sizes.
  if (unique_sizes.empty()) {
    for (int candidate : {256, 128, 512, 1024, 64}) {
      if (transformer_model
              .FindSignature(PrefillSig(kPrefillSignatureBase, candidate))
              .HasValue()) {
        unique_sizes.insert(candidate);
      }
    }
  }
  if (unique_sizes.empty()) {
    return absl::NotFoundError(
        "Could not detect a prefill signature (e.g. \"prefill_128\") in the "
        "transformer model.");
  }
  std::vector<int> sorted_sizes(unique_sizes.begin(), unique_sizes.end());
  absl::c_sort(sorted_sizes);
  return sorted_sizes;
}

void DumpModelSignatures(absl::string_view tag, const litert::Model& model) {
  auto signatures_opt = model.GetSignatures();
  if (signatures_opt) {
    ABSL_LOG(INFO) << "--- DUMPING SIGNATURES FOR: " << tag << " ---";
    for (const auto& signature : *signatures_opt) {
      ABSL_LOG(INFO) << "  Signature Key: " << signature.Key();
    }
    ABSL_LOG(INFO) << "-----------------------------------------";
  } else {
    ABSL_LOG(WARNING) << "Model " << tag << " has no signatures!";
  }
}

bool IsKVCacheTensor(absl::string_view name) {
  return absl::StartsWith(name, kv_cache_k_root_name) ||
         absl::StartsWith(name, kv_cache_v_root_name) ||
         absl::StartsWith(name, kv_cache_c_root_name);
}

absl::StatusOr<std::vector<int>> DetectSupportedContextSizes(
    const litert::Model& model, int prefill_size, int max_sequence_length) {
  std::vector<int> sizes;
  auto signatures = model.GetSignatures();
  if (signatures) {
    for (const auto& signature : *signatures) {
      absl::string_view key = signature.Key();
      if (absl::StartsWith(key, "decode_cache_")) {
        absl::string_view suffix = key.substr(sizeof("decode_cache_") - 1);
        int size = 0;
        if (absl::SimpleAtoi(suffix, &size) && size > 0) {
          sizes.push_back(size);
        }
      }
    }
  }
  if (sizes.empty()) {
    sizes.push_back(max_sequence_length);
  } else {
    absl::c_sort(sizes);
  }
  std::vector<int> filtered_sizes;
  auto it = absl::c_lower_bound(sizes, max_sequence_length);
  if (it != sizes.end()) {
    filtered_sizes.assign(sizes.begin(), it + 1);
  } else {
    filtered_sizes = std::move(sizes);
  }
  return filtered_sizes;
}

LlmLiteRtNpuCompiledModelExecutor::ResolvedSignatures BuildResolvedSignatures(
    int prefill_size, int context_size, bool has_suffix, bool has_rope_suffix) {
  // Helper for main/mask/cache_update signatures which get context size suffix
  auto context_suffixed_prefill_name = [](absl::string_view base, int p, int c,
                                          bool suffix) {
    if (suffix) {
      return absl::StrCat(base, "_", p, "_cache_", c);
    } else {
      return absl::StrCat(base, "_", p);
    }
  };

  auto context_suffixed_decode_name = [](absl::string_view base, int c,
                                         bool suffix) {
    if (suffix) {
      return absl::StrCat(base, "_cache_", c);
    } else {
      return std::string(base);
    }
  };

  // Helper for embedder signatures which do NOT get context size suffix
  auto static_prefill_name = [](absl::string_view base, int p) {
    return absl::StrCat(base, "_", p);
  };

  auto static_decode_name = [](absl::string_view base) {
    return std::string(base);
  };

  return LlmLiteRtNpuCompiledModelExecutor::ResolvedSignatures{
      .size = prefill_size,
      .context_size = context_size,
      .prefill = context_suffixed_prefill_name(
          kPrefillSignatureBase, prefill_size, context_size, has_suffix),
      .embedder = static_prefill_name(kPrefillEmbedderBase, prefill_size),
      .embedder_per_layer =
          static_prefill_name(kPrefillEmbedderPerLayerBase, prefill_size),
      .mask = context_suffixed_prefill_name(kPrefillMaskBase, prefill_size,
                                            context_size, has_suffix),
      .rope = context_suffixed_prefill_name(kPrefillRopeBase, prefill_size,
                                            context_size, has_rope_suffix),
      .cache_update = context_suffixed_prefill_name(
          kPrefillCacheUpdateBase, prefill_size, context_size, has_suffix),
      .decode = context_suffixed_decode_name(kDecodeSignature, context_size,
                                             has_suffix),
      .decode_embedder = static_decode_name(kDecodeEmbedder),
      .decode_embedder_per_layer = static_decode_name(kDecodeEmbedderPerLayer),
      .decode_mask =
          context_suffixed_decode_name(kDecodeMask, context_size, has_suffix),
      .decode_rope = context_suffixed_decode_name(kDecodeRope, context_size,
                                                  has_rope_suffix),
      .decode_cache_update = context_suffixed_decode_name(
          kDecodeCacheUpdate, context_size, has_suffix),
      .verify = context_suffixed_decode_name(kVerifySignature, context_size,
                                             has_suffix),
      .verify_embedder = static_decode_name(kVerifyEmbedder),
      .verify_embedder_per_layer = static_decode_name(kVerifyEmbedderPerLayer),
      .verify_mask =
          context_suffixed_decode_name(kVerifyMask, context_size, has_suffix),
      .verify_rope = context_suffixed_decode_name(kVerifyRope, context_size,
                                                  has_rope_suffix),
      .verify_cache_update = context_suffixed_decode_name(
          kVerifyCacheUpdate, context_size, has_suffix)};
}

absl::StatusOr<::litert::TensorBuffer> CreateAliasBuffer(
    const ::litert::Environment& env,
    const ::litert::TensorBuffer& large_buffer,
    const ::litert::RankedTensorType& target_type) {
  LITERT_ASSIGN_OR_RETURN(auto buffer_type, large_buffer.BufferType());
  if (buffer_type == ::litert::TensorBufferType::kAhwb) {
    auto ahwb_res = large_buffer.GetAhwb();
    if (ahwb_res.HasValue()) {
      LITERT_ASSIGN_OR_RETURN(auto buf,
                              ::litert::TensorBuffer::CreateFromAhwb(
                                  env, target_type, ahwb_res.Value(), 0));
      return std::move(buf);
    }
  } else if (buffer_type == ::litert::TensorBufferType::kHostMemory) {
    auto env_holder = env.GetHolder();
    void* host_mem_addr;
    if (env_holder.runtime->GetTensorBufferHostMemory(
            large_buffer.Get(), &host_mem_addr) == kLiteRtStatusOk) {
      LITERT_ASSIGN_OR_RETURN(auto target_bytes, target_type.Bytes());
      LITERT_ASSIGN_OR_RETURN(
          auto buf, ::litert::TensorBuffer::CreateFromHostMemory(
                        env, target_type, host_mem_addr, target_bytes));
      return std::move(buf);
    }
  }
  return absl::InternalError(
      absl::StrCat("Unsupported buffer type for context size aliasing: ",
                   static_cast<int>(buffer_type)));
}

}  // namespace

// On Windows, `ERROR` is defined as a macro, which can cause issues if it is
// expanded prematurely where the literal token `ERROR` is expected.
//
// To work around this, we use token concatenation (`##`) to construct the
// underlying macro name. Because `severity` is pasted (##), it is NOT expanded
// to its macro value first. For example, `NPU_EXECUTOR_LOG(ERROR)` simply
// pastes `NPU_EXECUTOR_LOG_` and `ERROR` to form `NPU_EXECUTOR_LOG_ERROR`,
// which then safely expands to `ABSL_LOG_IF(ERROR, ...)`.
#define NPU_EXECUTOR_LOG_INFO \
  ABSL_LOG_IF(INFO, npu_config_.enable_npu_debug_logging)
#define NPU_EXECUTOR_LOG_ERROR \
  ABSL_LOG_IF(ERROR, npu_config_.enable_npu_debug_logging)
#define NPU_EXECUTOR_LOG_WARNING \
  ABSL_LOG_IF(WARNING, npu_config_.enable_npu_debug_logging)
#define NPU_EXECUTOR_LOG(severity) NPU_EXECUTOR_LOG_##severity

// Signature names for the embedder.
struct EmbedderSignatures {
  static constexpr absl::string_view kDecodeEmbedder = "decode_embedder";
  static constexpr absl::string_view kVerifyEmbedder = "verify_embedder";
  // Prefill and decode use identical tensor signature names.
  static constexpr absl::string_view kEmbedderInput = "token_ids";
  static constexpr absl::string_view kEmbedderOutput = "embeddings";
};

static constexpr absl::string_view kPerLayerEmbedderTensor =
    "per_layer_embeddings";

struct EmbedderPerLayerSignatures {
  static constexpr absl::string_view kDecodeEmbedderPerLayer =
      "decode_per_layer_embedder";
  static constexpr absl::string_view kVerifyEmbedderPerLayer =
      "verify_per_layer_embedder";
  // Prefill and decode use identical tensor signature names.
  static constexpr absl::string_view kEmbedderInput = "token_ids";
  static constexpr absl::string_view kEmbedderOutput = "embeddings";
};

// Signature names for the mask signatures.
struct MaskSignatures {
  static constexpr absl::string_view kDecodeMask = "decode_mask";
  static constexpr absl::string_view kVerifyMask = "verify_mask";
  // Prefill and decode use identical tensor signature names.
  static constexpr absl::string_view kMaskInputTimeStep = "time_step";
  static constexpr absl::string_view kMaskInputTokens = "input_tokens";
  static constexpr absl::string_view kMaskInputValidMask = "valid_mask";
};

// Signature names for the rope signatures.
struct RopeSignatures {
  static constexpr absl::string_view kDecodeRope = "decode_rope";
  static constexpr absl::string_view kVerifyRope = "verify_rope";
  // Prefill and decode use identical tensor signature names.
  static constexpr absl::string_view kInputPos = "input_pos";
  static constexpr absl::string_view kOutputPosEmbeddingLocalLow =
      "pos_emb_local_cos";
  static constexpr absl::string_view kOutputPosEmbeddingHigh = "pos_emb_sin";
  static constexpr absl::string_view kOutputPosEmbeddingLocalHigh =
      "pos_emb_local_sin";
  static constexpr absl::string_view kOutputPosEmbeddingLow = "pos_emb_cos";
};

// Signature names for the LLM signatures.
struct LlmSignatures {
  static constexpr absl::string_view kDecodeLlm = "decode";
  static constexpr absl::string_view kVerifyLlm = "verify";
  static constexpr absl::string_view kInputEmbeddings = "embeddings";
  static constexpr absl::string_view kDecodeLogitsOutput = "logits";
  static constexpr absl::string_view kVerifyLogitsOutput = "logits";
  static constexpr absl::string_view kLastLayerActivationsOutput =
      "activations";
};

// Signature names for the cache update signatures.
struct CacheUpdateSignatures {
  static constexpr absl::string_view kDecodeCacheUpdate = "decode_cache_update";
  static constexpr absl::string_view kVerifyCacheUpdate = "verify_cache_update";
  static constexpr absl::string_view kInputPos = "input_pos";
  static constexpr absl::string_view kInputValidMask = "valid_mask";
};

struct MtpSignatures {
  static constexpr absl::string_view kMtpDrafter = "mtp_drafter";
  static constexpr absl::string_view kMtpRope = "rope";
  static constexpr absl::string_view kMtpMask = "mask";
  static constexpr absl::string_view kInputActivations = "activations";
  static constexpr absl::string_view kInputPos = "input_pos";
  static constexpr absl::string_view kInputTokens = "input_tokens";
  static constexpr absl::string_view kInputTimeStep = "time_step";
  static constexpr absl::string_view kOutputLogits = "logits";
  static constexpr absl::string_view kOutputActivations =
      "projected_activations";
};

absl::Status Fill(TensorBuffer& tensor_buffer, uint16_t value) {
  LITERT_ASSIGN_OR_RETURN(RankedTensorType tensor_buffer_type,
                          tensor_buffer.TensorType());
  LITERT_ASSIGN_OR_RETURN(
      auto lock_and_addr,
      ::litert::TensorBufferScopedLock::Create(
          tensor_buffer, ::litert::TensorBuffer::LockMode::kWrite));
  LITERT_ASSIGN_OR_RETURN(size_t num_elements,
                          tensor_buffer_type.Layout().NumElements());
  if (tensor_buffer_type.ElementType() == ::litert::ElementType::Float32) {
    float* ptr = static_cast<float*>(lock_and_addr.second);
    float float_value = static_cast<float>(value);
    for (int i = 0; i < num_elements; ++i) {
      ptr[i] = float_value;
    }

  } else {
    if (tensor_buffer_type.ElementType() == ::litert::ElementType::Int16) {
      int16_t* ptr = static_cast<int16_t*>(lock_and_addr.second);
      int16_t int16_value = static_cast<int16_t>(value);
      for (int i = 0; i < num_elements; ++i) {
        ptr[i] = int16_value;
      }

    } else if (tensor_buffer_type.ElementType() ==
               ::litert::ElementType::UInt16) {
      uint16_t* ptr = static_cast<uint16_t*>(lock_and_addr.second);
      for (int i = 0; i < num_elements; ++i) {
        ptr[i] = value;
      }
    } else {
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported tensor element type for Fill: ",
                       tensor_buffer_type.ElementType()));
    }
  }
  return absl::OkStatus();
}

absl::Status SetFirstElement(::litert::TensorBuffer& buffer, int32_t value) {
  LITERT_ASSIGN_OR_RETURN(
      auto lock_and_addr,
      ::litert::TensorBufferScopedLock::Create(
          buffer, ::litert::TensorBuffer::LockMode::kWrite));
  static_cast<int32_t*>(lock_and_addr.second)[0] = value;
  return absl::OkStatus();
}

// Copies the raw bytes from the tensor buffer.
absl::StatusOr<std::vector<uint8_t>> CopyRawBytesFromTensorBuffer(
    const TensorBuffer& buffer) {
  LITERT_ASSIGN_OR_RETURN(RankedTensorType tensor_type, buffer.TensorType());
  LITERT_ASSIGN_OR_RETURN(size_t num_bytes, buffer.Size());
  if (tensor_type.ElementType() == ::litert::ElementType::Float32) {
    LITERT_ASSIGN_OR_RETURN(auto buf, CopyFromTensorBuffer<float>(buffer));
    std::vector<uint8_t> res(num_bytes);
    std::memcpy(res.data(), buf.data(), num_bytes);
    return res;
  } else if (tensor_type.ElementType() == ::litert::ElementType::Int16) {
    LITERT_ASSIGN_OR_RETURN(auto buf, CopyFromTensorBuffer<int16_t>(buffer));
    std::vector<uint8_t> res(num_bytes);
    std::memcpy(res.data(), buf.data(), num_bytes);
    return res;
  } else if (tensor_type.ElementType() == ::litert::ElementType::Int8) {
    LITERT_ASSIGN_OR_RETURN(auto buf, CopyFromTensorBuffer<int8_t>(buffer));
    std::vector<uint8_t> res(num_bytes);
    std::memcpy(res.data(), buf.data(), num_bytes);
    return res;
  } else {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported tensor element type for copying: ",
                     tensor_type.ElementType()));
  }
}

// Returns true if the transformer model has a per layer embedder input buffer.
litert::Expected<bool> HasPerLayerEmbedder(
    const litert::Model& transformer_model,
    absl::string_view prefill_signature) {
  LITERT_ASSIGN_OR_RETURN(
      auto input_names,
      transformer_model.GetSignatureInputNames(prefill_signature));
  for (auto input_name : input_names) {
    if (kPerLayerEmbedderTensor == input_name) {
      return true;
    }
  }
  return false;
}

int64_t GetKvCacheInitValue(ModelResources& resources) {
  int64_t kv_cache_init_value = 0;
  if (auto metadata_status = resources.GetLlmMetadata(); metadata_status.ok()) {
    const proto::LlmMetadata* metadata = *metadata_status;
    if (metadata && metadata->has_kv_cache_init_value()) {
      kv_cache_init_value = metadata->kv_cache_init_value();
    }
  }
  return kv_cache_init_value;
}

absl::Status ClearKVCacheToZero(
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>& buffers) {
  for (auto& [buffer_name, buffer] : buffers) {
    if (buffer_name.starts_with(kv_cache_k_root_name) ||
        buffer_name.starts_with(kv_cache_v_root_name) ||
        buffer_name.starts_with(kv_cache_c_root_name)) {
      auto status = buffer.Clear();
      if (!status) {
        LITERT_ASSIGN_OR_RETURN(
            auto lock_and_addr,
            ::litert::TensorBufferScopedLock::Create(
                buffer, ::litert::TensorBuffer::LockMode::kWrite));
        LITERT_ASSIGN_OR_RETURN(size_t size, buffer.Size());
        std::memset(lock_and_addr.second, 0, size);
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace

std::ostream& operator<<(
    std::ostream& os,
    const LlmLiteRtNpuCompiledModelExecutor::LatencyStats& stats) {
  auto safe_tokens_per_sec = [](uint32_t num_tokens,
                                uint64_t latency_us) -> float {
    if (latency_us == 0) return 0.0f;
    return (static_cast<float>(num_tokens) * 1000000.0f) /
           static_cast<float>(latency_us);
  };
  auto safe_percentage = [](uint64_t part_us, uint64_t total_us) -> float {
    if (total_us == 0) return 0.0f;
    return (static_cast<float>(part_us) * 100.0f) /
           static_cast<float>(total_us);
  };

  os << "\n" << "====== PREFILL STATS ======";
  os << "\n" << "Total prefill latency [us]: " << stats.prefill_e2e_latency_us;
  os << "\n" << "(e2e) Prefill num tokens: " << stats.prefill_num_tokens;
  os << "\n"
     << "(e2e) Prefill tokens per second: "
     << safe_tokens_per_sec(stats.prefill_num_tokens,
                            stats.prefill_e2e_latency_us);
  os << "\n"
     << "(TransformerStackOnly) Prefill tokens per second: "
     << safe_tokens_per_sec(stats.prefill_num_tokens,
                            stats.prefill_llm_inference_latency_us);

  os << "\n" << "------ Prefill breakdown ------";
  os << "\n"
     << "Total prefill prepare input tensors latency [us]: "
     << stats.prefill_prepare_input_latency_us << " ("
     << safe_percentage(stats.prefill_prepare_input_latency_us,
                        stats.prefill_e2e_latency_us)
     << "%)";
  os << "\n"
     << "Total prefill embedder inference latency [us]: "
     << stats.prefill_embedder_inference_latency_us << " ("
     << safe_percentage(stats.prefill_embedder_inference_latency_us,
                        stats.prefill_e2e_latency_us)
     << "%)";
  if (stats.prefill_embedder_per_layer_inference_latency_us > 0) {
    os << "\n"
       << "Total prefill embedder per layer inference latency [us]: "
       << stats.prefill_embedder_per_layer_inference_latency_us << " ("
       << safe_percentage(stats.prefill_embedder_per_layer_inference_latency_us,
                          stats.prefill_e2e_latency_us)
       << "%)";
  }
  os << "\n"
     << "Total prefill rope inference latency [us]: "
     << stats.prefill_rope_inference_latency_us << " ("
     << safe_percentage(stats.prefill_rope_inference_latency_us,
                        stats.prefill_e2e_latency_us)
     << "%)";
  os << "\n"
     << "Total prefill mask inference latency [us]: "
     << stats.prefill_mask_inference_latency_us << " ("
     << safe_percentage(stats.prefill_mask_inference_latency_us,
                        stats.prefill_e2e_latency_us)
     << "%)";
  os << "\n"
     << "Total prefill llm inference latency [us]: "
     << stats.prefill_llm_inference_latency_us << " ("
     << safe_percentage(stats.prefill_llm_inference_latency_us,
                        stats.prefill_e2e_latency_us)
     << "%)";
  os << "\n"
     << "Total prefill cache update inference latency [us]: "
     << stats.prefill_cache_update_inference_latency_us << " ("
     << safe_percentage(stats.prefill_cache_update_inference_latency_us,
                        stats.prefill_e2e_latency_us)
     << "%)";

  os << "\n\n" << "====== DECODE STATS ======";
  os << "\n" << "Total decode latency [us]: " << stats.decode_e2e_latency_us;
  os << "\n" << "(e2e) Decode num tokens: " << stats.decode_num_tokens;
  os << "\n"
     << "(e2e) Decode tokens per second (avg): "
     << safe_tokens_per_sec(stats.decode_num_tokens,
                            stats.decode_e2e_latency_us);
  if (stats.mtp_num_draft_tokens > 0) {
    os << "\n"
       << "Speculative decoding acceptance rate [%]: "
       << (float)stats.mtp_num_accepted_tokens / stats.mtp_num_draft_tokens *
              100;
  }
  os << "\n"
     << "(TransformerStackOnly) Decode tokens per second: "
     << safe_tokens_per_sec(stats.decode_num_tokens,
                            stats.decode_llm_inference_latency_us);

  os << "\n" << "------ Decode breakdown ------";
  os << "\n"
     << "Total decode prepare input tensors latency [us]: "
     << stats.decode_prepare_input_latency_us << " ("
     << safe_percentage(stats.decode_prepare_input_latency_us,
                        stats.decode_e2e_latency_us)
     << "%)";
  os << "\n"
     << "Total decode embedder inference latency [us]: "
     << stats.decode_embedder_inference_latency_us << " ("
     << safe_percentage(stats.decode_embedder_inference_latency_us,
                        stats.decode_e2e_latency_us)
     << "%)";
  if (stats.decode_embedder_per_layer_inference_latency_us > 0) {
    os << "\n"
       << "Total decode embedder per layer inference latency [us]: "
       << stats.decode_embedder_per_layer_inference_latency_us << " ("
       << safe_percentage(stats.decode_embedder_per_layer_inference_latency_us,
                          stats.decode_e2e_latency_us)
       << "%)";
  }
  os << "\n"
     << "Total decode rope inference latency [us]: "
     << stats.decode_rope_inference_latency_us << " ("
     << safe_percentage(stats.decode_rope_inference_latency_us,
                        stats.decode_e2e_latency_us)
     << "%)";
  os << "\n"
     << "Total decode mask inference latency [us]: "
     << stats.decode_mask_inference_latency_us << " ("
     << safe_percentage(stats.decode_mask_inference_latency_us,
                        stats.decode_e2e_latency_us)
     << "%)";
  os << "\n"
     << "Total decode llm inference latency [us]: "
     << stats.decode_llm_inference_latency_us << " ("
     << safe_percentage(stats.decode_llm_inference_latency_us,
                        stats.decode_e2e_latency_us)
     << "%)";
  os << "\n"
     << "Total decode cache update inference latency [us]: "
     << stats.decode_cache_update_inference_latency_us << " ("
     << safe_percentage(stats.decode_cache_update_inference_latency_us,
                        stats.decode_e2e_latency_us)
     << "%)";
  os << "\n"
     << "Total decode sampling latency [us]: "
     << stats.decode_sampling_latency_us << " ("
     << safe_percentage(stats.decode_sampling_latency_us,
                        stats.decode_e2e_latency_us)
     << "%)";
  if (stats.decode_mtp_rejection_sampling_latency_us > 0) {
    os << "\n"
       << "Total decode MTP rejection sampling latency [us]: "
       << stats.decode_mtp_rejection_sampling_latency_us << " ("
       << safe_percentage(stats.decode_mtp_rejection_sampling_latency_us,
                          stats.decode_e2e_latency_us)
       << "%)";
  }
  if (stats.decode_mtp_activation_copy_latency_us > 0) {
    os << "\n"
       << "Total decode MTP activation copy latency [us]: "
       << stats.decode_mtp_activation_copy_latency_us << " ("
       << safe_percentage(stats.decode_mtp_activation_copy_latency_us,
                          stats.decode_e2e_latency_us)
       << "%)";
  }
  os << "\n"
     << "Total decode token queue latency [us]: "
     << stats.decode_token_queue_latency_us << " ("
     << safe_percentage(stats.decode_token_queue_latency_us,
                        stats.decode_e2e_latency_us)
     << "%)";

  return os;
}

// Creates LiteRT options for NPU accelerator.
litert::Expected<litert::Options>
LlmLiteRtNpuCompiledModelExecutor::CreateLiteRtNpuOptions(
    const LlmExecutorSettings& settings) {
  LITERT_ASSIGN_OR_RETURN(auto options, ::litert::Options::Create());
  options.SetHardwareAccelerators(litert::HwAccelerators::kNpu |
                                  litert::HwAccelerators::kCpu);
  // TODO: saliltambe - Bug: 498622107
#if defined(__ANDROID__)
  LITERT_ASSIGN_OR_RETURN(::litert::qualcomm::QualcommOptions & qnn_opts,
                          options.GetQualcommOptions());
  qnn_opts.SetLogLevel(::litert::qualcomm::QualcommOptions::LogLevel::kOff);
  qnn_opts.SetHtpPerformanceMode(
      ::litert::qualcomm::QualcommOptions::HtpPerformanceMode::kBurst);
  LITERT_ASSIGN_OR_RETURN(auto& google_tensor_opts,
                          options.GetGoogleTensorOptions());
  google_tensor_opts.SetPerformanceMode(
      ::litert::google_tensor::GoogleTensorOptions::PerformanceMode::kBurst);
#endif
  return options;
}

litert::Expected<litert::Options>
LlmLiteRtNpuCompiledModelExecutor::CreateLiteRtCpuOptions(
    const LlmExecutorSettings& settings) {
  LITERT_ASSIGN_OR_RETURN(auto options, ::litert::Options::Create());
  options.SetHardwareAccelerators(litert::HwAccelerators::kCpu);
  return options;
}

LlmLiteRtNpuCompiledModelExecutor::~LlmLiteRtNpuCompiledModelExecutor() {
  ABSL_VLOG(1) << "LatencyStats: " << GetLatencyStats();
}

absl::StatusOr<LlmLiteRtNpuCompiledModelExecutor::EmbedderContext>
LlmLiteRtNpuCompiledModelExecutor::CreateEmbedderContextWithBufferSharing(
    ::litert::Environment& env, const litert::Model& embedder_model,
    const std::vector<ResolvedSignatures>& prefill_signatures_list,
    TensorBufferMapBySize& gemma_prefill_input_buffers_by_size,
    TensorBufferMap& gemma_decode_input_buffers,
    TensorBufferMap& gemma_verify_input_buffers,
    const LlmExecutorSettings& settings) {
  LITERT_ASSIGN_OR_RETURN(auto options, CreateLiteRtCpuOptions(settings));
  LITERT_ASSIGN_OR_RETURN(
      CompiledModel embedder_compiled_model,
      CompiledModel::Create(env, embedder_model.Get(), options));

  TensorBufferMapBySize prefill_input_buffers_by_size;
  TensorBufferMapBySize prefill_output_buffers_by_size;

  for (const auto& sigs : prefill_signatures_list) {
    int prefill_size = sigs.size;
    auto& prefill_input_buffers = prefill_input_buffers_by_size[prefill_size];
    auto& prefill_output_buffers = prefill_output_buffers_by_size[prefill_size];

    LITERT_ASSIGN_OR_RETURN(
        prefill_input_buffers[EmbedderSignatures::kEmbedderInput],
        embedder_compiled_model.CreateInputBuffer(
            sigs.embedder, EmbedderSignatures::kEmbedderInput));
    prefill_input_buffers[EmbedderSignatures::kEmbedderInput].Clear();

    LITERT_ASSIGN_OR_RETURN(
        prefill_output_buffers[EmbedderSignatures::kEmbedderOutput],
        gemma_prefill_input_buffers_by_size.at(prefill_size)
            .at(LlmSignatures::kInputEmbeddings)
            .Duplicate());
  }

  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      decode_input_buffers;
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      decode_output_buffers;
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      verify_input_buffers;
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      verify_output_buffers;

  LITERT_ASSIGN_OR_RETURN(
      decode_input_buffers[EmbedderSignatures::kEmbedderInput],
      embedder_compiled_model.CreateInputBuffer(
          EmbedderSignatures::kDecodeEmbedder,
          EmbedderSignatures::kEmbedderInput));
  decode_input_buffers[EmbedderSignatures::kEmbedderInput].Clear();

  LITERT_ASSIGN_OR_RETURN(
      decode_output_buffers[EmbedderSignatures::kEmbedderOutput],
      gemma_decode_input_buffers[LlmSignatures::kInputEmbeddings].Duplicate());

  if (embedder_compiled_model.FindSignature(
          EmbedderSignatures::kVerifyEmbedder)) {
    LITERT_ASSIGN_OR_RETURN(
        verify_input_buffers[EmbedderSignatures::kEmbedderInput],
        embedder_compiled_model.CreateInputBuffer(
            EmbedderSignatures::kVerifyEmbedder,
            EmbedderSignatures::kEmbedderInput));
    verify_input_buffers[EmbedderSignatures::kEmbedderInput].Clear();

    LITERT_ASSIGN_OR_RETURN(
        verify_output_buffers[EmbedderSignatures::kEmbedderOutput],
        gemma_verify_input_buffers[LlmSignatures::kInputEmbeddings]
            .Duplicate());
  }

  return EmbedderContext(
      std::move(embedder_compiled_model),
      std::move(prefill_input_buffers_by_size),
      std::move(prefill_output_buffers_by_size),
      std::move(decode_input_buffers), std::move(decode_output_buffers),
      std::move(verify_input_buffers), std::move(verify_output_buffers));
}

absl::StatusOr<LlmLiteRtNpuCompiledModelExecutor::EmbedderPerLayerContext>
LlmLiteRtNpuCompiledModelExecutor::
    CreateEmbedderPerLayerContextWithBufferSharing(
        ::litert::Environment& env, const litert::Model& embedder_model,
        const absl::flat_hash_map<int, ::litert::TensorBuffer>&
            prefill_input_tokens_by_size,
        const ::litert::TensorBuffer& decode_input_tokens,
        const ::litert::TensorBuffer& verify_input_tokens,
        TensorBufferMapBySize& gemma_prefill_input_buffers_by_size,
        TensorBufferMap& gemma_decode_input_buffers,
        TensorBufferMap& gemma_verify_input_buffers,
        const LlmExecutorSettings& settings) {
  LITERT_ASSIGN_OR_RETURN(auto options, CreateLiteRtCpuOptions(settings));
  LITERT_ASSIGN_OR_RETURN(
      CompiledModel embedder_compiled_model,
      CompiledModel::Create(env, embedder_model.Get(), options));

  TensorBufferMapBySize prefill_input_buffers_by_size;
  TensorBufferMapBySize prefill_output_buffers_by_size;

  for (const auto& [size, prefill_input_tokens] :
       prefill_input_tokens_by_size) {
    auto& prefill_input_buffers = prefill_input_buffers_by_size[size];
    auto& prefill_output_buffers = prefill_output_buffers_by_size[size];

    LITERT_ASSIGN_OR_RETURN(
        prefill_input_buffers[EmbedderPerLayerSignatures::kEmbedderInput],
        prefill_input_tokens.Duplicate());

    LITERT_ASSIGN_OR_RETURN(
        prefill_output_buffers[EmbedderPerLayerSignatures::kEmbedderOutput],
        gemma_prefill_input_buffers_by_size.at(size)
            .at(kPerLayerEmbedderTensor)
            .Duplicate());
  }

  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      decode_input_buffers;
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      decode_output_buffers;
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      verify_input_buffers;
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      verify_output_buffers;

  LITERT_ASSIGN_OR_RETURN(
      decode_input_buffers[EmbedderPerLayerSignatures::kEmbedderInput],
      decode_input_tokens.Duplicate());

  LITERT_ASSIGN_OR_RETURN(
      decode_output_buffers[EmbedderPerLayerSignatures::kEmbedderOutput],
      gemma_decode_input_buffers[kPerLayerEmbedderTensor].Duplicate());

  if (embedder_compiled_model.FindSignature(
          EmbedderPerLayerSignatures::kVerifyEmbedderPerLayer)) {
    LITERT_ASSIGN_OR_RETURN(
        verify_input_buffers[EmbedderPerLayerSignatures::kEmbedderInput],
        verify_input_tokens.Duplicate());

    LITERT_ASSIGN_OR_RETURN(
        verify_output_buffers[EmbedderPerLayerSignatures::kEmbedderOutput],
        gemma_verify_input_buffers[kPerLayerEmbedderTensor].Duplicate());
  }

  return EmbedderPerLayerContext(
      std::move(embedder_compiled_model),
      std::move(prefill_input_buffers_by_size),
      std::move(prefill_output_buffers_by_size),
      std::move(decode_input_buffers), std::move(decode_output_buffers),
      std::move(verify_input_buffers), std::move(verify_output_buffers));
}

absl::StatusOr<LlmLiteRtNpuCompiledModelExecutor::NpuAuxiliaryContext>
LlmLiteRtNpuCompiledModelExecutor::CreateNpuAuxiliaryContext(
    ::litert::Environment& env, const litert::Model& npu_auxiliary_model,
    const LlmExecutorSettings& settings) {
  LITERT_ASSIGN_OR_RETURN(auto options, CreateLiteRtNpuOptions(settings));
  LITERT_ASSIGN_OR_RETURN(
      CompiledModel npu_auxiliary_compiled_model,
      CompiledModel::Create(env, npu_auxiliary_model.Get(), options));
  NpuAuxiliaryContext npu_auxiliary_context(
      std::move(npu_auxiliary_compiled_model));
  return npu_auxiliary_context;
}

absl::StatusOr<LlmLiteRtNpuCompiledModelExecutor::InferenceContext>
LlmLiteRtNpuCompiledModelExecutor::CreateMaskContextWithBufferSharing(
    const NpuAuxiliaryContext& npu_auxiliary_context,
    const std::vector<ResolvedSignatures>& prefill_signatures_list,
    TensorBufferMapBySize& gemma_prefill_input_buffers_by_size,
    TensorBufferMap& gemma_decode_input_buffers,
    TensorBufferMap& gemma_verify_input_buffers) {
  TensorBufferMapBySize prefill_input_buffers_by_size;
  TensorBufferMapBySize prefill_output_buffers_by_size;

  for (const auto& sigs : prefill_signatures_list) {
    int size = sigs.size;
    auto& prefill_input_buffers = prefill_input_buffers_by_size[size];
    auto& prefill_output_buffers = prefill_output_buffers_by_size[size];
    auto gemma_buffers_it = gemma_prefill_input_buffers_by_size.find(size);

    LITERT_ASSIGN_OR_RETURN(
        prefill_input_buffers[MaskSignatures::kMaskInputTimeStep],
        npu_auxiliary_context.npu_auxiliary_compiled_model.CreateInputBuffer(
            sigs.mask, MaskSignatures::kMaskInputTimeStep));
    prefill_input_buffers[MaskSignatures::kMaskInputTimeStep].Clear();
    LITERT_ASSIGN_OR_RETURN(
        prefill_input_buffers[MaskSignatures::kMaskInputTokens],
        npu_auxiliary_context.npu_auxiliary_compiled_model.CreateInputBuffer(
            sigs.mask, MaskSignatures::kMaskInputTokens));
    prefill_input_buffers[MaskSignatures::kMaskInputTokens].Clear();

    LITERT_ASSIGN_OR_RETURN(auto prefill_mask_input_names,
                            npu_auxiliary_context.npu_auxiliary_compiled_model
                                .GetSignatureInputNames(sigs.mask));
    if (absl::c_find(prefill_mask_input_names,
                     MaskSignatures::kMaskInputValidMask) !=
        prefill_mask_input_names.end()) {
      LITERT_ASSIGN_OR_RETURN(
          prefill_input_buffers[MaskSignatures::kMaskInputValidMask],
          npu_auxiliary_context.npu_auxiliary_compiled_model.CreateInputBuffer(
              sigs.mask, MaskSignatures::kMaskInputValidMask));
      prefill_input_buffers[MaskSignatures::kMaskInputValidMask].Clear();
    }

    LITERT_ASSIGN_OR_RETURN(auto prefill_mask_output_names,
                            npu_auxiliary_context.npu_auxiliary_compiled_model
                                .GetSignatureOutputNames(sigs.mask));
    for (const auto& mask_output_name : prefill_mask_output_names) {
      bool found = false;
      if (gemma_buffers_it != gemma_prefill_input_buffers_by_size.end()) {
        const auto& gemma_buffers = gemma_buffers_it->second;
        if (auto it = gemma_buffers.find(mask_output_name);
            it != gemma_buffers.end()) {
          LITERT_ASSIGN_OR_RETURN(prefill_output_buffers[mask_output_name],
                                  it->second.Duplicate());
          found = true;
        }
      }
      if (!found) {
        LITERT_ASSIGN_OR_RETURN(
            prefill_output_buffers[mask_output_name],
            npu_auxiliary_context.npu_auxiliary_compiled_model
                .CreateOutputBuffer(sigs.mask, mask_output_name));
      }
    }
  }

  const auto& representative_signatures = prefill_signatures_list.back();

  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      decode_input_buffers;
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      decode_output_buffers;
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      verify_input_buffers;
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      verify_output_buffers;

  LITERT_ASSIGN_OR_RETURN(
      decode_input_buffers[MaskSignatures::kMaskInputTimeStep],
      npu_auxiliary_context.npu_auxiliary_compiled_model.CreateInputBuffer(
          representative_signatures.decode_mask,
          MaskSignatures::kMaskInputTimeStep));
  decode_input_buffers[MaskSignatures::kMaskInputTimeStep].Clear();
  LITERT_ASSIGN_OR_RETURN(
      decode_input_buffers[MaskSignatures::kMaskInputTokens],
      npu_auxiliary_context.npu_auxiliary_compiled_model.CreateInputBuffer(
          representative_signatures.decode_mask,
          MaskSignatures::kMaskInputTokens));
  decode_input_buffers[MaskSignatures::kMaskInputTokens].Clear();

  LITERT_ASSIGN_OR_RETURN(
      auto decode_mask_input_names,
      npu_auxiliary_context.npu_auxiliary_compiled_model.GetSignatureInputNames(
          representative_signatures.decode_mask));
  if (absl::c_find(decode_mask_input_names,
                   MaskSignatures::kMaskInputValidMask) !=
      decode_mask_input_names.end()) {
    LITERT_ASSIGN_OR_RETURN(
        decode_input_buffers[MaskSignatures::kMaskInputValidMask],
        npu_auxiliary_context.npu_auxiliary_compiled_model.CreateInputBuffer(
            representative_signatures.decode_mask,
            MaskSignatures::kMaskInputValidMask));
    decode_input_buffers[MaskSignatures::kMaskInputValidMask].Clear();
  }

  LITERT_ASSIGN_OR_RETURN(
      auto decode_mask_output_names,
      npu_auxiliary_context.npu_auxiliary_compiled_model
          .GetSignatureOutputNames(representative_signatures.decode_mask));
  for (const auto& mask_output_name : decode_mask_output_names) {
    if (gemma_decode_input_buffers.contains(mask_output_name)) {
      LITERT_ASSIGN_OR_RETURN(
          decode_output_buffers[mask_output_name],
          gemma_decode_input_buffers[mask_output_name].Duplicate());
    } else {
      LITERT_ASSIGN_OR_RETURN(
          decode_output_buffers[mask_output_name],
          npu_auxiliary_context.npu_auxiliary_compiled_model.CreateOutputBuffer(
              representative_signatures.decode_mask, mask_output_name));
    }
  }

  if (npu_auxiliary_context.npu_auxiliary_compiled_model.FindSignature(
          representative_signatures.verify_mask)) {
    LITERT_ASSIGN_OR_RETURN(
        verify_input_buffers[MaskSignatures::kMaskInputTimeStep],
        npu_auxiliary_context.npu_auxiliary_compiled_model.CreateInputBuffer(
            representative_signatures.verify_mask,
            MaskSignatures::kMaskInputTimeStep));
    verify_input_buffers[MaskSignatures::kMaskInputTimeStep].Clear();
    LITERT_ASSIGN_OR_RETURN(
        verify_input_buffers[MaskSignatures::kMaskInputTokens],
        npu_auxiliary_context.npu_auxiliary_compiled_model.CreateInputBuffer(
            representative_signatures.verify_mask,
            MaskSignatures::kMaskInputTokens));
    verify_input_buffers[MaskSignatures::kMaskInputTokens].Clear();

    LITERT_ASSIGN_OR_RETURN(
        auto verify_mask_input_names,
        npu_auxiliary_context.npu_auxiliary_compiled_model
            .GetSignatureInputNames(representative_signatures.verify_mask));
    if (absl::c_find(verify_mask_input_names,
                     MaskSignatures::kMaskInputValidMask) !=
        verify_mask_input_names.end()) {
      LITERT_ASSIGN_OR_RETURN(
          verify_input_buffers[MaskSignatures::kMaskInputValidMask],
          npu_auxiliary_context.npu_auxiliary_compiled_model.CreateInputBuffer(
              representative_signatures.verify_mask,
              MaskSignatures::kMaskInputValidMask));
      verify_input_buffers[MaskSignatures::kMaskInputValidMask].Clear();
    }

    LITERT_ASSIGN_OR_RETURN(
        auto verify_mask_output_names,
        npu_auxiliary_context.npu_auxiliary_compiled_model
            .GetSignatureOutputNames(representative_signatures.verify_mask));
    for (const auto& mask_output_name : verify_mask_output_names) {
      if (gemma_verify_input_buffers.contains(mask_output_name)) {
        LITERT_ASSIGN_OR_RETURN(
            verify_output_buffers[mask_output_name],
            gemma_verify_input_buffers[mask_output_name].Duplicate());
      } else {
        LITERT_ASSIGN_OR_RETURN(
            verify_output_buffers[mask_output_name],
            npu_auxiliary_context.npu_auxiliary_compiled_model
                .CreateOutputBuffer(representative_signatures.verify_mask,
                                    mask_output_name));
      }
    }
  }
  return InferenceContext(
      std::move(prefill_input_buffers_by_size),
      std::move(prefill_output_buffers_by_size),
      std::move(decode_input_buffers), std::move(decode_output_buffers),
      std::move(verify_input_buffers), std::move(verify_output_buffers));
}

absl::StatusOr<LlmLiteRtNpuCompiledModelExecutor::InferenceContext>
LlmLiteRtNpuCompiledModelExecutor::CreateRopeContextWithBufferSharing(
    const NpuAuxiliaryContext& npu_auxiliary_context,
    const std::vector<ResolvedSignatures>& prefill_signatures_list,
    TensorBufferMapBySize& gemma_prefill_input_buffers_by_size,
    TensorBufferMap& gemma_decode_input_buffers,
    TensorBufferMap& gemma_verify_input_buffers) {
  TensorBufferMapBySize prefill_input_buffers_by_size;
  TensorBufferMapBySize prefill_output_buffers_by_size;

  constexpr absl::string_view rope_output_names[] = {
      RopeSignatures::kOutputPosEmbeddingLocalLow,
      RopeSignatures::kOutputPosEmbeddingHigh,
      RopeSignatures::kOutputPosEmbeddingLocalHigh,
      RopeSignatures::kOutputPosEmbeddingLow};

  for (const auto& sigs : prefill_signatures_list) {
    int size = sigs.size;
    auto& prefill_input_buffers = prefill_input_buffers_by_size[size];
    auto& prefill_output_buffers = prefill_output_buffers_by_size[size];
    auto gemma_buffers_it = gemma_prefill_input_buffers_by_size.find(size);

    LITERT_ASSIGN_OR_RETURN(
        prefill_input_buffers[RopeSignatures::kInputPos],
        npu_auxiliary_context.npu_auxiliary_compiled_model.CreateInputBuffer(
            sigs.rope, RopeSignatures::kInputPos));
    prefill_input_buffers[RopeSignatures::kInputPos].Clear();

    for (const auto& rope_output_name : rope_output_names) {
      if (gemma_buffers_it != gemma_prefill_input_buffers_by_size.end()) {
        const auto& gemma_buffers = gemma_buffers_it->second;
        if (auto it = gemma_buffers.find(rope_output_name);
            it != gemma_buffers.end()) {
          LITERT_ASSIGN_OR_RETURN(prefill_output_buffers[rope_output_name],
                                  it->second.Duplicate());
        }
      }
    }
  }

  const auto& representative_signatures = prefill_signatures_list.back();

  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      decode_input_buffers;
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      decode_output_buffers;
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      verify_input_buffers;
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      verify_output_buffers;

  LITERT_ASSIGN_OR_RETURN(
      decode_input_buffers[RopeSignatures::kInputPos],
      npu_auxiliary_context.npu_auxiliary_compiled_model.CreateInputBuffer(
          representative_signatures.decode_rope, RopeSignatures::kInputPos));
  decode_input_buffers[RopeSignatures::kInputPos].Clear();

  for (const auto& rope_output_name : rope_output_names) {
    if (gemma_decode_input_buffers.contains(rope_output_name)) {
      LITERT_ASSIGN_OR_RETURN(
          decode_output_buffers[rope_output_name],
          gemma_decode_input_buffers[rope_output_name].Duplicate());
    }
  }

  if (npu_auxiliary_context.npu_auxiliary_compiled_model.FindSignature(
          representative_signatures.verify_rope)) {
    LITERT_ASSIGN_OR_RETURN(
        verify_input_buffers[RopeSignatures::kInputPos],
        npu_auxiliary_context.npu_auxiliary_compiled_model.CreateInputBuffer(
            representative_signatures.verify_rope, RopeSignatures::kInputPos));
    verify_input_buffers[RopeSignatures::kInputPos].Clear();

    for (const auto& rope_output_name : rope_output_names) {
      if (gemma_verify_input_buffers.contains(rope_output_name)) {
        LITERT_ASSIGN_OR_RETURN(
            verify_output_buffers[rope_output_name],
            gemma_verify_input_buffers[rope_output_name].Duplicate());
      }
    }
  }

  return InferenceContext(
      std::move(prefill_input_buffers_by_size),
      std::move(prefill_output_buffers_by_size),
      std::move(decode_input_buffers), std::move(decode_output_buffers),
      std::move(verify_input_buffers), std::move(verify_output_buffers));
}

absl::Status LlmLiteRtNpuCompiledModelExecutor::AllocateTransformerBuffers(
    litert::Environment& env, const litert::Model* transformer_model,
    CompiledModel& llm_compiled_model,
    const std::vector<ResolvedSignatures>& prefill_signatures_list,
    TensorBufferMapBySize& gemma_prefill_input_buffers_by_size,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        gemma_decode_input_buffers,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        gemma_verify_input_buffers,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        input_kv_cache_buffers,
    TensorBufferMapBySize& prefill_output_kv_cache_slice_buffers_by_size,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        decode_output_kv_cache_slice_buffers,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        verify_output_kv_cache_slice_buffers,
    absl::flat_hash_map<absl::string_view, HWQuantParams>& kv_quant_params,
    int64_t kv_cache_init_value) {
  const auto& representative_signatures = prefill_signatures_list.back();

  auto prefill_signature =
      transformer_model->FindSignature(representative_signatures.prefill);

  if (prefill_signature.HasValue()) {
    for (auto output_name : prefill_signature->OutputNames()) {
      if (absl::StartsWith(output_name, kv_cache_slice_k_root_name) ||
          absl::StartsWith(output_name, kv_cache_slice_v_root_name)) {
        auto tensor_expected = prefill_signature->OutputTensor(output_name);
        if (tensor_expected.HasValue()) {
          HWQuantParams q_params;
          if (tensor_expected->HasQuantization()) {
            auto pq = tensor_expected->PerTensorQuantization();
            q_params.scale = pq.scale;
            q_params.zero_point = pq.zero_point;
          }
          kv_quant_params[output_name] = q_params;
        }
      }
    }
  }

  // Create input buffers for prefill signatures (KV cache only, once).
  for (auto input_name : prefill_signature->InputNames()) {
    if (absl::StartsWith(input_name, kv_cache_k_root_name) ||
        absl::StartsWith(input_name, kv_cache_v_root_name) ||
        absl::StartsWith(input_name, kv_cache_c_root_name)) {
      auto [it, inserted] = input_kv_cache_buffers.try_emplace(input_name);
      if (inserted) {
        LITERT_ASSIGN_OR_RETURN(
            it->second, llm_compiled_model.CreateInputBuffer(
                            representative_signatures.prefill, input_name));
        LITERT_RETURN_IF_ERROR(
            FillKVCacheBuffer(it->second, kv_cache_init_value));
      }
    }
  }

  // Allocate prefill-specific buffers for each prefill size.
  for (const auto& sigs : prefill_signatures_list) {
    int size = sigs.size;
    auto& gemma_prefill_input_buffers =
        gemma_prefill_input_buffers_by_size[size];
    auto& prefill_output_kv_cache_slice_buffers =
        prefill_output_kv_cache_slice_buffers_by_size[size];

    LITERT_ASSIGN_OR_RETURN(auto sig,
                            transformer_model->FindSignature(sigs.prefill));
    for (auto input_name : sig.InputNames()) {
      if (absl::StartsWith(input_name, kv_cache_k_root_name) ||
          absl::StartsWith(input_name, kv_cache_v_root_name) ||
          absl::StartsWith(input_name, kv_cache_c_root_name)) {
        // Skip KV cache buffers.
      } else {
        LITERT_ASSIGN_OR_RETURN(
            gemma_prefill_input_buffers[input_name],
            llm_compiled_model.CreateInputBuffer(sigs.prefill, input_name));
        gemma_prefill_input_buffers[input_name].Clear();
      }
    }

    // Create output buffers for prefill signatures.
    for (auto output_name : sig.OutputNames()) {
      if (absl::StartsWith(output_name, kv_cache_slice_k_root_name) ||
          absl::StartsWith(output_name, kv_cache_slice_v_root_name) ||
          absl::StartsWith(output_name, kv_cache_slice_c_root_name)) {
        LITERT_ASSIGN_OR_RETURN(
            prefill_output_kv_cache_slice_buffers[output_name],
            llm_compiled_model.CreateOutputBuffer(sigs.prefill, output_name));
      }
    }
  }

  // Create input buffers for decode signature. Skip kv cache input buffers as
  // they are already created in the prefill signature.
  auto decode_signature =
      transformer_model->FindSignature(representative_signatures.decode);
  for (auto input_name : decode_signature->InputNames()) {
    if (absl::StartsWith(input_name, kv_cache_k_root_name) ||
        absl::StartsWith(input_name, kv_cache_v_root_name) ||
        absl::StartsWith(input_name, kv_cache_c_root_name)) {
      // Create the input kv cache buffer for the decode signature if it is not
      // created in the prefill signature.
      if (!input_kv_cache_buffers.contains(input_name)) {
        LITERT_ASSIGN_OR_RETURN(
            input_kv_cache_buffers[input_name],
            llm_compiled_model.CreateInputBuffer(
                representative_signatures.decode, input_name));
        LITERT_RETURN_IF_ERROR(FillKVCacheBuffer(
            input_kv_cache_buffers[input_name], kv_cache_init_value));
      }
      continue;
    }
    LITERT_ASSIGN_OR_RETURN(gemma_decode_input_buffers[input_name],
                            llm_compiled_model.CreateInputBuffer(
                                representative_signatures.decode, input_name));
    gemma_decode_input_buffers[input_name].Clear();
  }

  // Create output buffers for decode signature.
  for (auto output_name : decode_signature->OutputNames()) {
    if (absl::StartsWith(output_name, kv_cache_slice_k_root_name) ||
        absl::StartsWith(output_name, kv_cache_slice_v_root_name) ||
        absl::StartsWith(output_name, kv_cache_slice_c_root_name)) {
      LITERT_ASSIGN_OR_RETURN(
          decode_output_kv_cache_slice_buffers[output_name],
          llm_compiled_model.CreateOutputBuffer(
              representative_signatures.decode, output_name));
    }
  }

  // Create input/output buffers for verify signature if it exists.
  auto verify_signature =
      transformer_model->FindSignature(representative_signatures.verify);
  if (verify_signature) {
    for (auto input_name : verify_signature->InputNames()) {
      LITERT_ASSIGN_OR_RETURN(
          gemma_verify_input_buffers[input_name],
          llm_compiled_model.CreateInputBuffer(representative_signatures.verify,
                                               input_name));
      gemma_verify_input_buffers[input_name].Clear();
    }
    for (auto output_name : verify_signature->OutputNames()) {
      if (absl::StartsWith(output_name, kv_cache_slice_k_root_name) ||
          absl::StartsWith(output_name, kv_cache_slice_v_root_name) ||
          absl::StartsWith(output_name, kv_cache_slice_c_root_name)) {
        LITERT_ASSIGN_OR_RETURN(
            verify_output_kv_cache_slice_buffers[output_name],
            llm_compiled_model.CreateOutputBuffer(
                representative_signatures.verify, output_name));
      }
    }
  }

  return absl::OkStatus();
}

absl::StatusOr<LlmLiteRtNpuCompiledModelExecutor::InferenceContext>
LlmLiteRtNpuCompiledModelExecutor::CreateLlmInferenceContextWithBufferSharing(
    ::litert::Environment& env, ::litert::CompiledModel& llm_compiled_model,
    const std::vector<ResolvedSignatures>& prefill_signatures_list,
    TensorBufferMap& input_kv_cache_buffers,
    TensorBufferMapBySize& prefill_output_kv_cache_slice_buffers_by_size,
    TensorBufferMap& decode_output_kv_cache_slice_buffers,
    TensorBufferMap& verify_output_kv_cache_slice_buffers,
    TensorBufferMapBySize& gemma_prefill_input_buffers_by_size,
    TensorBufferMap& gemma_decode_input_buffers,
    TensorBufferMap& gemma_verify_input_buffers) {
  TensorBufferMapBySize prefill_input_buffers_by_size;
  TensorBufferMapBySize prefill_output_buffers_by_size;

  for (const auto& sigs : prefill_signatures_list) {
    int size = sigs.size;
    auto& prefill_input_buffers = prefill_input_buffers_by_size[size];
    auto& prefill_output_buffers = prefill_output_buffers_by_size[size];
    const auto& gemma_prefill_input_buffers =
        gemma_prefill_input_buffers_by_size.at(size);
    const auto& prefill_output_kv_cache_slice_buffers =
        prefill_output_kv_cache_slice_buffers_by_size.at(size);

    for (const auto& [key, value] : gemma_prefill_input_buffers) {
      LITERT_ASSIGN_OR_RETURN(prefill_input_buffers[key], value.Duplicate());
    }

    // Duplicate all kv cache buffers to prefill inputs.
    LITERT_ASSIGN_OR_RETURN(
        auto prefill_input_names,
        llm_compiled_model.GetSignatureInputNames(sigs.prefill));
    for (const auto& [key, value] : input_kv_cache_buffers) {
      // Check if the kv cache buffer is used in the prefill signature.
      if (absl::c_find(prefill_input_names, std::string(key)) ==
          prefill_input_names.end()) {
        continue;
      }

      // The last layer kv cache in the prefill signature has float32 elements,
      // although it's not used in the model, CompiledModel will complain about
      // the mismatched buffer size. So we need to correct the buffer size here,
      // by creating a new buffer with the correct size.
      LITERT_ASSIGN_OR_RETURN(
          auto input_tensor_type,
          llm_compiled_model.GetInputTensorType(sigs.prefill, key));
      LITERT_ASSIGN_OR_RETURN(auto input_tensor_size,
                              input_tensor_type.Bytes());
      LITERT_ASSIGN_OR_RETURN(auto input_buffer_size, value.Size());
      if (input_tensor_size > input_buffer_size) {
        LITERT_ASSIGN_OR_RETURN(
            auto corrected_input_buffer,
            llm_compiled_model.CreateInputBuffer(sigs.prefill, key));
        corrected_input_buffer.Clear();
        LITERT_ASSIGN_OR_RETURN(prefill_input_buffers[key],
                                corrected_input_buffer.Duplicate());
      } else {
        LITERT_ASSIGN_OR_RETURN(prefill_input_buffers[key], value.Duplicate());
      }
    }

    // Duplicate all output kv cache slice buffers to prefill output buffers.
    for (const auto& [key, value] : prefill_output_kv_cache_slice_buffers) {
      LITERT_ASSIGN_OR_RETURN(prefill_output_buffers[key], value.Duplicate());
    }
  }

  const auto& representative_signatures = prefill_signatures_list.back();

  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      decode_input_buffers;
  for (const auto& [key, value] : gemma_decode_input_buffers) {
    LITERT_ASSIGN_OR_RETURN(decode_input_buffers[key], value.Duplicate());
  }
  // Duplicate all kv cache buffers to decode inputs.
  for (const auto& [key, value] : input_kv_cache_buffers) {
    LITERT_ASSIGN_OR_RETURN(decode_input_buffers[key], value.Duplicate());
  }

  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      decode_output_buffers;

  // Duplicate all output kv cache slice buffers to decode output
  // buffers.
  for (const auto& [key, value] : decode_output_kv_cache_slice_buffers) {
    LITERT_ASSIGN_OR_RETURN(decode_output_buffers[key], value.Duplicate());
  }

  LITERT_ASSIGN_OR_RETURN(auto decode_output_names,
                          llm_compiled_model.GetSignatureOutputNames(
                              representative_signatures.decode));

  for (const auto& name : decode_output_names) {
    if (name == LlmSignatures::kDecodeLogitsOutput) {
      LITERT_ASSIGN_OR_RETURN(
          decode_output_buffers[LlmSignatures::kDecodeLogitsOutput],
          llm_compiled_model.CreateOutputBuffer(
              representative_signatures.decode,
              LlmSignatures::kDecodeLogitsOutput));
    } else if (name == LlmSignatures::kLastLayerActivationsOutput) {
      LITERT_ASSIGN_OR_RETURN(
          decode_output_buffers[LlmSignatures::kLastLayerActivationsOutput],
          llm_compiled_model.CreateOutputBuffer(
              representative_signatures.decode,
              LlmSignatures::kLastLayerActivationsOutput));
    }
  }

  auto verify_signature =
      llm_compiled_model.FindSignature(representative_signatures.verify);
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      verify_input_buffers;
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      verify_output_buffers;
  if (verify_signature) {
    for (const auto& [key, value] : gemma_verify_input_buffers) {
      LITERT_ASSIGN_OR_RETURN(verify_input_buffers[key], value.Duplicate());
    }
    // Duplicate all kv cache buffers to verify inputs.
    LITERT_ASSIGN_OR_RETURN(auto verify_input_names,
                            llm_compiled_model.GetSignatureInputNames(
                                representative_signatures.verify));
    for (const auto& [key, value] : input_kv_cache_buffers) {
      if (absl::c_find(verify_input_names, std::string(key)) !=
          verify_input_names.end()) {
        LITERT_ASSIGN_OR_RETURN(verify_input_buffers[key], value.Duplicate());
      }
    }

    for (const auto& [key, value] : verify_output_kv_cache_slice_buffers) {
      LITERT_ASSIGN_OR_RETURN(verify_output_buffers[key], value.Duplicate());
    }

    LITERT_ASSIGN_OR_RETURN(auto verify_output_names,
                            llm_compiled_model.GetSignatureOutputNames(
                                representative_signatures.verify));

    for (const auto& name : verify_output_names) {
      if (name == LlmSignatures::kDecodeLogitsOutput) {
        LITERT_ASSIGN_OR_RETURN(
            verify_output_buffers[LlmSignatures::kDecodeLogitsOutput],
            llm_compiled_model.CreateOutputBuffer(
                representative_signatures.verify,
                LlmSignatures::kDecodeLogitsOutput));
      } else if (name == LlmSignatures::kLastLayerActivationsOutput) {
        LITERT_ASSIGN_OR_RETURN(
            verify_output_buffers[LlmSignatures::kLastLayerActivationsOutput],
            llm_compiled_model.CreateOutputBuffer(
                representative_signatures.verify,
                LlmSignatures::kLastLayerActivationsOutput));
      }
    }
  }

  return InferenceContext(
      std::move(prefill_input_buffers_by_size),
      std::move(prefill_output_buffers_by_size),
      std::move(decode_input_buffers), std::move(decode_output_buffers),
      std::move(verify_input_buffers), std::move(verify_output_buffers));
}

absl::StatusOr<LlmLiteRtNpuCompiledModelExecutor::InferenceContext>
LlmLiteRtNpuCompiledModelExecutor::
    CreateCacheUpdateInferenceContextWithBufferSharing(
        TensorBufferMap& input_kv_cache_buffers,
        TensorBufferMapBySize& prefill_output_kv_cache_slice_buffers_by_size,
        TensorBufferMap& decode_output_kv_cache_slice_buffers,
        TensorBufferMap& verify_output_kv_cache_slice_buffers,
        absl::flat_hash_map<int, ::litert::TensorBuffer>&
            prefill_input_pos_by_size,
        ::litert::TensorBuffer decode_input_pos,
        ::litert::TensorBuffer verify_input_pos,
        absl::flat_hash_map<int, ::litert::TensorBuffer>&
            prefill_valid_mask_by_size,
        ::litert::TensorBuffer decode_valid_mask,
        ::litert::TensorBuffer verify_valid_mask) {
  absl::flat_hash_map<
      int, absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>>
      prefill_input_buffers_by_size;
  absl::flat_hash_map<
      int, absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>>
      prefill_output_buffers_by_size;

  for (const auto& [size, prefill_output_kv_cache_slice_buffers] :
       prefill_output_kv_cache_slice_buffers_by_size) {
    auto& prefill_input_buffers = prefill_input_buffers_by_size[size];
    auto& prefill_output_buffers = prefill_output_buffers_by_size[size];

    for (const auto& [key, value] : input_kv_cache_buffers) {
      LITERT_ASSIGN_OR_RETURN(prefill_input_buffers[key], value.Duplicate());
    }
    for (const auto& [key, value] : prefill_output_kv_cache_slice_buffers) {
      LITERT_ASSIGN_OR_RETURN(prefill_input_buffers[key], value.Duplicate());
    }
    LITERT_ASSIGN_OR_RETURN(
        prefill_input_buffers[CacheUpdateSignatures::kInputPos],
        prefill_input_pos_by_size.at(size).Duplicate());

    if (auto it = prefill_valid_mask_by_size.find(size);
        it != prefill_valid_mask_by_size.end() && it->second) {
      LITERT_ASSIGN_OR_RETURN(
          prefill_input_buffers[CacheUpdateSignatures::kInputValidMask],
          it->second.Duplicate());
    }

    for (const auto& [key, value] : input_kv_cache_buffers) {
      LITERT_ASSIGN_OR_RETURN(prefill_output_buffers[key], value.Duplicate());
    }
  }

  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      decode_input_buffers;
  {
    for (const auto& [key, value] : input_kv_cache_buffers) {
      LITERT_ASSIGN_OR_RETURN(decode_input_buffers[key], value.Duplicate());
    }
    for (const auto& [key, value] : decode_output_kv_cache_slice_buffers) {
      LITERT_ASSIGN_OR_RETURN(decode_input_buffers[key], value.Duplicate());
    }
    decode_input_buffers[CacheUpdateSignatures::kInputPos] =
        std::move(decode_input_pos);
    if (decode_valid_mask) {
      decode_input_buffers[CacheUpdateSignatures::kInputValidMask] =
          std::move(decode_valid_mask);
    }
  }
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      decode_output_buffers;
  {
    for (const auto& [key, value] : input_kv_cache_buffers) {
      LITERT_ASSIGN_OR_RETURN(decode_output_buffers[key], value.Duplicate());
    }
  }

  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      verify_input_buffers;
  {
    for (const auto& [key, value] : input_kv_cache_buffers) {
      LITERT_ASSIGN_OR_RETURN(verify_input_buffers[key], value.Duplicate());
    }
    for (const auto& [key, value] : verify_output_kv_cache_slice_buffers) {
      LITERT_ASSIGN_OR_RETURN(verify_input_buffers[key], value.Duplicate());
    }
    verify_input_buffers[CacheUpdateSignatures::kInputPos] =
        std::move(verify_input_pos);
    if (verify_valid_mask) {
      verify_input_buffers[CacheUpdateSignatures::kInputValidMask] =
          std::move(verify_valid_mask);
    }
  }
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      verify_output_buffers;
  {
    for (const auto& [key, value] : input_kv_cache_buffers) {
      LITERT_ASSIGN_OR_RETURN(verify_output_buffers[key], value.Duplicate());
    }
  }

  return InferenceContext(
      std::move(prefill_input_buffers_by_size),
      std::move(prefill_output_buffers_by_size),
      std::move(decode_input_buffers), std::move(decode_output_buffers),
      std::move(verify_input_buffers), std::move(verify_output_buffers));
}

absl::StatusOr<LlmLiteRtNpuCompiledModelExecutor::DrafterContext>
LlmLiteRtNpuCompiledModelExecutor::
    CreateDrafterInferenceContextWithBufferSharing(
        ::litert::Environment& env, const litert::Model& mtp_drafter_model,
        absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
            drafter_input_kv_cache_buffers,
        ::litert::TensorBuffer& output_activations_buffers) {
  LITERT_ASSIGN_OR_RETURN(auto mtp_compiled_model,
                          CompiledModel::Create(env, mtp_drafter_model.Get(),
                                                litert::HwAccelerators::kCpu));
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      mtp_input_buffers;
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      mtp_output_buffers;

  // Create input and output buffers for the MTP drafter.
  auto mtp_signature =
      mtp_compiled_model.FindSignature(MtpSignatures::kMtpDrafter);
  for (const auto& input_name : mtp_signature->InputNames()) {
    // Reuse kv cache buffers from main model
    if (absl::StartsWith(input_name, kv_cache_k_root_name) ||
        absl::StartsWith(input_name, kv_cache_v_root_name)) {
      LITERT_ASSIGN_OR_RETURN(
          mtp_input_buffers[input_name],
          drafter_input_kv_cache_buffers[input_name].Duplicate());
    } else {
      LITERT_ASSIGN_OR_RETURN(mtp_input_buffers[input_name],
                              mtp_compiled_model.CreateInputBuffer(
                                  MtpSignatures::kMtpDrafter, input_name));
      mtp_input_buffers[input_name].Clear();
    }
  }
  for (const auto& output_name : mtp_signature->OutputNames()) {
    {
      LITERT_ASSIGN_OR_RETURN(mtp_output_buffers[output_name],
                              mtp_compiled_model.CreateOutputBuffer(
                                  MtpSignatures::kMtpDrafter, output_name));
    }
  }
  return DrafterContext(std::move(mtp_compiled_model),
                        std::move(mtp_input_buffers),
                        std::move(mtp_output_buffers));
}

absl::StatusOr<LlmLiteRtNpuCompiledModelExecutor::DrafterAuxContext>
LlmLiteRtNpuCompiledModelExecutor::
    CreateDrafterInferenceAuxContextWithBufferSharing(
        ::litert::Environment& env, const litert::Model& mtp_aux_model,
        const absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
            drafter_aux_output_buffers) {
  LITERT_ASSIGN_OR_RETURN(auto mtp_aux_compiled_model,
                          CompiledModel::Create(env, mtp_aux_model.Get(),
                                                litert::HwAccelerators::kCpu));
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      rope_input_buffers;
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      rope_output_buffers;
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      mask_input_buffers;
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      mask_output_buffers;

  // Drafter aux rope signature.
  LITERT_ASSIGN_OR_RETURN(
      rope_input_buffers[MtpSignatures::kInputPos],
      mtp_aux_compiled_model.CreateInputBuffer(MtpSignatures::kMtpRope,
                                               MtpSignatures::kInputPos));
  rope_input_buffers[MtpSignatures::kInputPos].Clear();

  LITERT_ASSIGN_OR_RETURN(
      auto rope_output_names,
      mtp_aux_compiled_model.GetSignatureOutputNames(MtpSignatures::kMtpRope));
  // All drafter aux output buffers are shared with the drafter model.
  for (const auto& name : rope_output_names) {
    LITERT_ASSIGN_OR_RETURN(rope_output_buffers[name],
                            drafter_aux_output_buffers.at(name).Duplicate());
  }
  // Drafter aux mask signature.
  LITERT_ASSIGN_OR_RETURN(
      mask_input_buffers[MtpSignatures::kInputTimeStep],
      mtp_aux_compiled_model.CreateInputBuffer(MtpSignatures::kMtpMask,
                                               MtpSignatures::kInputTimeStep));
  mask_input_buffers[MtpSignatures::kInputTimeStep].Clear();

  LITERT_ASSIGN_OR_RETURN(
      mask_input_buffers[MtpSignatures::kInputTokens],
      mtp_aux_compiled_model.CreateInputBuffer(MtpSignatures::kMtpMask,
                                               MtpSignatures::kInputTokens));

  LITERT_ASSIGN_OR_RETURN(
      auto mask_output_names,
      mtp_aux_compiled_model.GetSignatureOutputNames(MtpSignatures::kMtpMask));
  for (const auto& name : mask_output_names) {
    LITERT_ASSIGN_OR_RETURN(mask_output_buffers[name],
                            drafter_aux_output_buffers.at(name).Duplicate());
  }

  return DrafterAuxContext(
      std::move(mtp_aux_compiled_model), std::move(rope_input_buffers),
      std::move(rope_output_buffers), std::move(mask_input_buffers),
      std::move(mask_output_buffers));
}
absl::Status LlmLiteRtNpuCompiledModelExecutor::WarmupInference(
    ::litert::CompiledModel& compiled_model_llm,
    InferenceContext& llm_inference_context,
    ::litert::CompiledModel& compiled_model_auxiliary,
    const std::vector<ResolvedSignatures>& prefill_signatures_list,
    const InferenceContext& rope_inference_context,
    const InferenceContext& mask_inference_context,
    const InferenceContext& cache_update_inference_context) {
  // We need to fill the embedding input buffers with non-zero values because
  // some of the Gemma3 models contain embedding lookup preprocessing that
  // quantize a float embedding tensor into a quantized embedding tensor and use
  // 'DIV' operations in the process. Without this we risk running into: ERROR:
  // third_party/tensorflow/lite/kernels/div.cc:242 data[i] != 0 was not true.
  // ERROR: Node number 21 (DIV) failed to invoke.

  auto run_prefill_warmup = [](CompiledModel& model,
                               absl::string_view signature,
                               const InferenceContext& context, int size,
                               absl::string_view name) -> absl::Status {
    auto result = model.Run(signature, context.GetPrefillInputBuffers(size),
                            context.GetPrefillOutputBuffers(size));
    RET_CHECK(result) << "Inference warmup run for " << name
                      << " (prefill) failed for size " << size << ": "
                      << result.Error().Message();
    return absl::OkStatus();
  };

  for (const auto& sigs : prefill_signatures_list) {
    int size = sigs.size;
    if (llm_inference_context.HasPrefillSize(size) &&
        llm_inference_context.GetPrefillInputBuffers(size).contains(
            LlmSignatures::kInputEmbeddings)) {
      ABSL_RETURN_IF_ERROR(Fill(llm_inference_context.GetPrefillInputBuffers(
                                    size)[LlmSignatures::kInputEmbeddings],
                                1));
    }
    ABSL_RETURN_IF_ERROR(run_prefill_warmup(
        compiled_model_llm, sigs.prefill, llm_inference_context, size, "LLM"));
    ABSL_RETURN_IF_ERROR(run_prefill_warmup(compiled_model_auxiliary, sigs.rope,
                                            rope_inference_context, size,
                                            "RoPE"));
    ABSL_RETURN_IF_ERROR(run_prefill_warmup(compiled_model_auxiliary, sigs.mask,
                                            mask_inference_context, size,
                                            "mask"));
    ABSL_RETURN_IF_ERROR(run_prefill_warmup(
        compiled_model_auxiliary, sigs.cache_update,
        cache_update_inference_context, size, "cache update"));
  }

  const auto& representative_signatures = prefill_signatures_list.back();

  if (llm_inference_context.decode_input_buffers.contains(
          LlmSignatures::kInputEmbeddings)) {
    ABSL_RETURN_IF_ERROR(
        Fill(llm_inference_context
                 .decode_input_buffers[LlmSignatures::kInputEmbeddings],
             1));
  }

  auto run_decode_warmup = [](CompiledModel& model, absl::string_view signature,
                              const InferenceContext& context,
                              absl::string_view name) -> absl::Status {
    auto result = model.Run(signature, context.decode_input_buffers,
                            context.decode_output_buffers);
    RET_CHECK(result) << "Inference warmup run for " << name
                      << " (decode) failed: " << result.Error().Message();
    return absl::OkStatus();
  };

  ABSL_RETURN_IF_ERROR(run_decode_warmup(compiled_model_llm,
                                         representative_signatures.decode,
                                         llm_inference_context, "LLM"));
  ABSL_RETURN_IF_ERROR(run_decode_warmup(compiled_model_auxiliary,
                                         representative_signatures.decode_rope,
                                         rope_inference_context, "RoPE"));
  ABSL_RETURN_IF_ERROR(run_decode_warmup(compiled_model_auxiliary,
                                         representative_signatures.decode_mask,
                                         mask_inference_context, "mask"));
  ABSL_RETURN_IF_ERROR(run_decode_warmup(
      compiled_model_auxiliary, representative_signatures.decode_cache_update,
      cache_update_inference_context, "cache update"));

  auto run_verify_warmup = [](CompiledModel& model, absl::string_view signature,
                              const InferenceContext& context,
                              absl::string_view name) -> absl::Status {
    if (model.FindSignature(signature)) {
      auto result = model.Run(signature, context.verify_input_buffers,
                              context.verify_output_buffers);
      RET_CHECK(result) << "Inference warmup run for " << name
                        << " (verify) failed: " << result.Error().Message();
    }
    return absl::OkStatus();
  };

  // Warmup verify signatures if they exist.
  ABSL_RETURN_IF_ERROR(run_verify_warmup(compiled_model_llm,
                                         representative_signatures.verify,
                                         llm_inference_context, "MTP verify"));
  ABSL_RETURN_IF_ERROR(run_verify_warmup(compiled_model_auxiliary,
                                         representative_signatures.verify_rope,
                                         rope_inference_context, "RoPE"));
  ABSL_RETURN_IF_ERROR(run_verify_warmup(compiled_model_auxiliary,
                                         representative_signatures.verify_mask,
                                         mask_inference_context, "mask"));
  ABSL_RETURN_IF_ERROR(run_verify_warmup(
      compiled_model_auxiliary, representative_signatures.verify_cache_update,
      cache_update_inference_context, "cache update"));

  // Clear the KV cache buffers after warmup.
  for (auto& [size, prefill_input_buffers] :
       llm_inference_context.prefill_input_buffers_by_size) {
    ABSL_RETURN_IF_ERROR(ClearKVCacheToZero(prefill_input_buffers));
  }
  return absl::OkStatus();
}

absl::Status LlmLiteRtNpuCompiledModelExecutor::WarmupDrafterInference(
    const DrafterContext& drafter_context,
    const DrafterAuxContext& drafter_aux_context) {
  auto result = drafter_context.mtp_compiled_model.Run(
      MtpSignatures::kMtpDrafter, drafter_context.mtp_input_buffers,
      drafter_context.mtp_output_buffers);
  RET_CHECK(result) << "Inference warmup run for MTP failed."
                    << result.Error().Message();

  result = drafter_aux_context.mtp_aux_compiled_model.Run(
      MtpSignatures::kMtpMask, drafter_aux_context.mask_input_buffers,
      drafter_aux_context.mask_output_buffers);
  RET_CHECK(result) << "Inference warmup run for MTP mask failed."
                    << result.Error().Message();

  result = drafter_aux_context.mtp_aux_compiled_model.Run(
      MtpSignatures::kMtpRope, drafter_aux_context.rope_input_buffers,
      drafter_aux_context.rope_output_buffers);
  RET_CHECK(result) << "Inference warmup run for MTP rope failed."
                    << result.Error().Message();
  return absl::OkStatus();
}

LlmLiteRtNpuCompiledModelExecutor::InferenceContext::InferenceContext(
    TensorBufferMap prefill_input_buffers,
    TensorBufferMap prefill_output_buffers,
    TensorBufferMap decode_input_buffers, TensorBufferMap decode_output_buffers,
    TensorBufferMap verify_input_buffers, TensorBufferMap verify_output_buffers)
    : prefill_input_buffers(std::move(prefill_input_buffers)),
      prefill_output_buffers(std::move(prefill_output_buffers)),
      decode_input_buffers(std::move(decode_input_buffers)),
      decode_output_buffers(std::move(decode_output_buffers)),
      verify_input_buffers(std::move(verify_input_buffers)),
      verify_output_buffers(std::move(verify_output_buffers)) {}

LlmLiteRtNpuCompiledModelExecutor::InferenceContext::InferenceContext(
    TensorBufferMapBySize prefill_input_buffers_by_size,
    TensorBufferMapBySize prefill_output_buffers_by_size,
    TensorBufferMap decode_input_buffers, TensorBufferMap decode_output_buffers,
    TensorBufferMap verify_input_buffers, TensorBufferMap verify_output_buffers)
    : prefill_input_buffers_by_size(std::move(prefill_input_buffers_by_size)),
      prefill_output_buffers_by_size(std::move(prefill_output_buffers_by_size)),
      decode_input_buffers(std::move(decode_input_buffers)),
      decode_output_buffers(std::move(decode_output_buffers)),
      verify_input_buffers(std::move(verify_input_buffers)),
      verify_output_buffers(std::move(verify_output_buffers)) {}

LlmLiteRtNpuCompiledModelExecutor::EmbedderContext::EmbedderContext(
    CompiledModel embedder_compiled_model,
    TensorBufferMap prefill_input_buffers,
    TensorBufferMap prefill_output_buffers,
    TensorBufferMap decode_input_buffers, TensorBufferMap decode_output_buffers,
    TensorBufferMap verify_input_buffers, TensorBufferMap verify_output_buffers)
    : embedder_compiled_model(std::move(embedder_compiled_model)),
      inference_context(
          std::move(prefill_input_buffers), std::move(prefill_output_buffers),
          std::move(decode_input_buffers), std::move(decode_output_buffers),
          std::move(verify_input_buffers), std::move(verify_output_buffers)) {}

LlmLiteRtNpuCompiledModelExecutor::EmbedderContext::EmbedderContext(
    CompiledModel embedder_compiled_model,
    TensorBufferMapBySize prefill_input_buffers_by_size,
    TensorBufferMapBySize prefill_output_buffers_by_size,
    TensorBufferMap decode_input_buffers, TensorBufferMap decode_output_buffers,
    TensorBufferMap verify_input_buffers, TensorBufferMap verify_output_buffers)
    : embedder_compiled_model(std::move(embedder_compiled_model)),
      inference_context(
          std::move(prefill_input_buffers_by_size),
          std::move(prefill_output_buffers_by_size),
          std::move(decode_input_buffers), std::move(decode_output_buffers),
          std::move(verify_input_buffers), std::move(verify_output_buffers)) {}

LlmLiteRtNpuCompiledModelExecutor::NpuAuxiliaryContext::NpuAuxiliaryContext(
    CompiledModel npu_auxiliary_compiled_model)
    : npu_auxiliary_compiled_model(std::move(npu_auxiliary_compiled_model)) {}

absl::Status LlmLiteRtNpuCompiledModelExecutor::Prefill(
    const ExecutorInputs& inputs) {
  return Prefill(inputs, ExecutorPrefillParams());
}

absl::Status LlmLiteRtNpuCompiledModelExecutor::Prefill(
    const ExecutorInputs& inputs, const ExecutorPrefillParams& params) {
  ran_decode_ = false;
  auto start = absl::Now();
  LITERT_ASSIGN_OR_RETURN(const auto* text_token_ids,
                          inputs.GetTextTokenIdsPtr());
  LITERT_ASSIGN_OR_RETURN(auto tensor_type, text_token_ids->TensorType());
  // Only accept batch size 1 for now.
  RET_CHECK_EQ(tensor_type.Layout().Dimensions()[0], 1);
  RET_CHECK_GT(tensor_type.Layout().Dimensions()[1], 0)
      << "Prefill token ids must be non-empty.";
  if (UseEmbeddingLookupManager()) {
    ABSL_RETURN_IF_ERROR(
        embedding_lookup_manager_->UpdateMultiModalEmbeddings(inputs));
  }
  LITERT_ASSIGN_OR_RETURN(auto ids,
                          ReferTensorBufferAsSpan<int32_t>(*text_token_ids));

  const int largest_context_size = sorted_supported_context_sizes_.back();
  const SortedPrefillSignatureMap largest_map =
      (active_context_size_ == largest_context_size)
          ? prefill_signature_map_
          : context_groups_.at(largest_context_size).prefill_signature_map;
  LITERT_ASSIGN_OR_RETURN(
      auto work_groups, GetOptimizedPrefillWorkGroups(largest_map, ids.size()));

  int current_processed_tokens = 0;
  for (const auto& [dummy_signature, prefill_length] : work_groups) {
    std::optional<int> copy_len =
        current_processed_tokens > 0
            ? std::make_optional(current_processed_tokens)
            : std::nullopt;
    ABSL_RETURN_IF_ERROR(SwitchContextSizeIfRequired(
        current_processed_tokens + prefill_length, copy_len));

    int runner_size = 0;
    for (const auto& [size, sig] : largest_map) {
      if (sig == dummy_signature) {
        runner_size = size;
        break;
      }
    }
    RET_CHECK_GT(runner_size, 0)
        << "Could not find runner size for signature " << dummy_signature;

    auto sig_it = prefill_signature_map_.find(runner_size);
    RET_CHECK(sig_it != prefill_signature_map_.end())
        << "Prefill runner size " << runner_size
        << " not supported in current context size " << active_context_size_;
    const std::string& resolved_signature = sig_it->second;

    ABSL_RETURN_IF_ERROR(PrefillInternal(
        resolved_signature, ids.subspan(/*pos=*/0, prefill_length)));
    ids = ids.subspan(/*pos=*/prefill_length);
    current_processed_tokens += prefill_length;
    latency_stats_.prefill_num_tokens += prefill_length;
  }
  RET_CHECK_EQ(ids.size(), 0).SetCode(absl::StatusCode::kInternal)
      << "Work groups not covering the entire prefill input.";

  if (UseEmbeddingLookupManager()) {
    ABSL_RETURN_IF_ERROR(
        embedding_lookup_manager_->CleanupMultiModalEmbeddings());
  }
  auto end = absl::Now();
  latency_stats_.prefill_e2e_latency_us +=
      absl::ToInt64Microseconds(end - start);

  return absl::OkStatus();
}

absl::StatusOr<::litert::TensorBuffer>
LlmLiteRtNpuCompiledModelExecutor::DecodeLogits(const ExecutorInputs& inputs) {
  return DecodeLogits(inputs, ExecutorDecodeParams());
}

absl::StatusOr<::litert::TensorBuffer>
LlmLiteRtNpuCompiledModelExecutor::DecodeLogits(
    const ExecutorInputs& inputs, const ExecutorDecodeParams& decode_params) {

  if (current_step_ >= executor_settings_.GetMaxNumTokens()) {
    return absl::ResourceExhaustedError("Reached maximum number of tokens.");
  }

  if (processed_tokens_.TokenCount() != current_step_) {
    LITERT_RETURN_IF_ERROR(processed_tokens_.RollBackToStep(current_step_));
  }

  if (inputs.GetTextDataPtr().ok()) {
    auto token_ids_buffer = inputs.GetTextTokenIdsPtr();
    if (token_ids_buffer.ok()) {
      auto input_tensor_size = (*token_ids_buffer)->PackedSize();
      if (input_tensor_size && *input_tensor_size != 0) {
        RET_CHECK_EQ(*input_tensor_size, sizeof(int32_t));
        LITERT_ASSIGN_OR_RETURN(
            auto ids, ReferTensorBufferAsSpan<int32_t>(**token_ids_buffer));
        if (ids[0] >= 0) {
          processed_tokens_.InvalidatePendingInputToken();
          std::shared_ptr<TokenData> token =
              std::make_shared<TokenData>(ids[0]);
          ABSL_RETURN_IF_ERROR(processed_tokens_.AddPendingInputToken({token}));
        }
      }
    }
  }

  auto [internal_start_step, pending_input_token] =
      processed_tokens_.GetNextUnprocessedToken();
  if (pending_input_token.empty()) {
    return absl::InvalidArgumentError("No id available to be decoded.");
  }

  bool last_run_is_decode = ran_decode_;

  std::shared_ptr<TokenData> token = pending_input_token[0];
  if (UseEmbeddingLookupManager() && token->embedding().empty()) {
    ABSL_RETURN_IF_ERROR(embedding_lookup_manager_->LookupDecode(
        token->id(), token->mutable_embedding()));
  }

  ABSL_RETURN_IF_ERROR(DecodeInternal(internal_start_step, token));
  ABSL_RETURN_IF_ERROR(processed_tokens_.MarkPendingInputTokenAsProcessed());

  const auto& src_buffer =
      llm_inference_context_
          .decode_output_buffers[LlmSignatures::kDecodeLogitsOutput];

  LITERT_ASSIGN_OR_RETURN(auto vocab_size, GetVocabSize());
  LITERT_ASSIGN_OR_RETURN(auto output_logits,
                          CreateTensorBuffer<float>({1, 1, vocab_size}));

  ABSL_RETURN_IF_ERROR(DequantizeLogits(src_buffer, output_logits,
                                        per_tensor_logits_scale_,
                                        per_tensor_logits_zero_point_, false));

  if (!decode_params.GetLogitsProcessorList().empty()) {
    std::vector<int> current_token_ids = {token->id()};
    if (last_run_is_decode) {
      for (LogitsProcessor* logits_processor :
           decode_params.GetLogitsProcessorList()) {
        ABSL_RETURN_IF_ERROR(
            logits_processor->UpdateState(absl::MakeSpan(current_token_ids)));
      }
    }

    for (LogitsProcessor* logits_processor :
         decode_params.GetLogitsProcessorList()) {
      ABSL_RETURN_IF_ERROR(logits_processor->ProcessLogits(output_logits));
    }
  }

  current_step_++;
  ran_decode_ = true;

  return output_logits;
}

absl::StatusOr<std::vector<std::vector<int>>>
LlmLiteRtNpuCompiledModelExecutor::Decode() {
  return Decode(ExecutorDecodeParams());
}

absl::StatusOr<std::vector<std::vector<int>>>
LlmLiteRtNpuCompiledModelExecutor::Decode(
    const ExecutorDecodeParams& decode_params) {
  if (!decode_params.GetLogitsProcessorList().empty()) {
    auto start = absl::Now();

    LITERT_ASSIGN_OR_RETURN(auto masked_logits,
                            DecodeLogits(ExecutorInputs(), decode_params));

    LITERT_ASSIGN_OR_RETURN(
        const int max_index,
        ApplyGreedySampling(masked_logits,
                            npu_config_.enable_neon_for_npu_greedy_sampling));

    std::shared_ptr<TokenData> last_output_token =
        std::make_shared<TokenData>(max_index);

    if (UseEmbeddingLookupManager()) {
      auto start_lookup = absl::Now();
      ABSL_RETURN_IF_ERROR(embedding_lookup_manager_->LookupDecode(
          last_output_token->id(), last_output_token->mutable_embedding()));
      latency_stats_.decode_embedder_inference_latency_us +=
          absl::ToInt64Microseconds(absl::Now() - start_lookup);
    }

    auto start_add = absl::Now();
    ABSL_RETURN_IF_ERROR(
        processed_tokens_.AddPendingInputToken({std::move(last_output_token)}));
    latency_stats_.decode_token_queue_latency_us +=
        absl::ToInt64Microseconds(absl::Now() - start_add);

    latency_stats_.decode_e2e_latency_us +=
        absl::ToInt64Microseconds(absl::Now() - start);
    latency_stats_.decode_num_tokens++;
    return std::vector<std::vector<int>>{{max_index}};
  }
  auto start = absl::Now();

  if (current_step_ >= executor_settings_.GetMaxNumTokens()) {
    return absl::ResourceExhaustedError("Reached maximum number of tokens.");
  }

  if (processed_tokens_.TokenCount() != current_step_) {
    auto start_queue = absl::Now();
    LITERT_RETURN_IF_ERROR(processed_tokens_.RollBackToStep(current_step_));
    latency_stats_.decode_token_queue_latency_us +=
        absl::ToInt64Microseconds(absl::Now() - start_queue);
  }

  if (!pending_accepted_tokens_.empty()) {
    auto start_queue = absl::Now();
    int next_token_id = pending_accepted_tokens_.front();
    pending_accepted_tokens_.erase(pending_accepted_tokens_.begin());

    NPU_EXECUTOR_LOG(INFO) << "Decode returning token from queue: "
                           << next_token_id << " (Remaining in queue: "
                           << pending_accepted_tokens_.size() << ")";

    std::shared_ptr<TokenData> next_token =
        std::make_shared<TokenData>(next_token_id);
    if (UseEmbeddingLookupManager()) {
      auto start_lookup = absl::Now();
      ABSL_RETURN_IF_ERROR(embedding_lookup_manager_->LookupDecode(
          next_token->id(), next_token->mutable_embedding()));
      latency_stats_.decode_embedder_inference_latency_us +=
          absl::ToInt64Microseconds(absl::Now() - start_lookup);
    }
    // We must add it as a pending input token so that the NEXT Decode call
    // can find it via GetNextUnprocessedToken if the queue is empty.
    auto mark_status = processed_tokens_.MarkPendingInputTokenAsProcessed();
    if (!mark_status.ok() && !absl::IsNotFound(mark_status)) {
      return mark_status;
    }
    ABSL_RETURN_IF_ERROR(
        processed_tokens_.AddPendingInputToken({std::move(next_token)}));
    current_step_++;
    latency_stats_.decode_token_queue_latency_us +=
        absl::ToInt64Microseconds(absl::Now() - start_queue);

    latency_stats_.decode_e2e_latency_us +=
        absl::ToInt64Microseconds(absl::Now() - start);
    latency_stats_.decode_num_tokens++;
    return std::vector<std::vector<int>>{{next_token_id}};
  }
  // No tokens in queue, run a full Speculative or Normal Decode cycle.
  auto start_get_token = absl::Now();
  auto [internal_start_step, pending_input_token] =
      processed_tokens_.GetNextUnprocessedToken();
  latency_stats_.decode_token_queue_latency_us +=
      absl::ToInt64Microseconds(absl::Now() - start_get_token);

  if (pending_input_token.empty()) {
    return absl::InvalidArgumentError("No id available to be decoded.");
  }

  int mtp_start_step = internal_start_step;
  int mtp_start_token_id = pending_input_token[0]->id();

  if (!has_valid_verify_activations_) {
    NPU_EXECUTOR_LOG(INFO) << "Step " << internal_start_step
                           << ": Running Main Decode Signature";
    ABSL_RETURN_IF_ERROR(
        DecodeInternal(internal_start_step, pending_input_token[0]));

    auto start_mark = absl::Now();
    ABSL_RETURN_IF_ERROR(processed_tokens_.MarkPendingInputTokenAsProcessed());
    latency_stats_.decode_token_queue_latency_us +=
        absl::ToInt64Microseconds(absl::Now() - start_mark);

    // Sample the output of the main decode to get the 'good' token for MTP.
    auto start_sample = absl::Now();
    LITERT_ASSIGN_OR_RETURN(
        mtp_start_token_id,
        ApplyGreedySampling(
            llm_inference_context_
                .decode_output_buffers[LlmSignatures::kDecodeLogitsOutput],
            npu_config_.enable_neon_for_npu_greedy_sampling));
    latency_stats_.decode_sampling_latency_us +=
        absl::ToInt64Microseconds(absl::Now() - start_sample);

    // The MTP drafter starts from the position of the token we just
    // generated.
    mtp_start_step = internal_start_step + 1;
  } else {
    NPU_EXECUTOR_LOG(INFO)
        << "Step " << internal_start_step
        << ": Skipping Main Decode (Using verify activations)";
    if (speculative_decoding_type_ == SpeculativeDecodingType::kMTP) {
      auto& ctx = drafter_context_.value();
      LITERT_ASSIGN_OR_RETURN(
          auto drafter_activations_lock_and_addr,
          ::litert::TensorBufferScopedLock::Create(
              ctx.mtp_input_buffers[MtpSignatures::kInputActivations],
              ::litert::TensorBuffer::LockMode::kWrite));
      memcpy(drafter_activations_lock_and_addr.second,
             last_verify_activations_.data(), last_verify_activations_.size());
    }
    ABSL_RETURN_IF_ERROR(processed_tokens_.MarkPendingInputTokenAsProcessed());
  }

  if (speculative_decoding_type_ == SpeculativeDecodingType::kMTP) {
    NPU_EXECUTOR_LOG(INFO) << "Step " << mtp_start_step
                           << ": Starting MTP Speculative Cycle";
    NPU_EXECUTOR_LOG(INFO) << "    [Verify] Step -1 at pos "
                           << mtp_start_step - 1 << ": Good token ID "
                           << mtp_start_token_id;

    LITERT_ASSIGN_OR_RETURN(std::vector<int> draft_tokens,
                            RunDrafterLoop(mtp_start_step, mtp_start_token_id));
    auto start_verify = absl::Now();
    ABSL_RETURN_IF_ERROR(
        RunVerifierBatch(mtp_start_step, mtp_start_token_id, draft_tokens));
    latency_stats_.decode_llm_inference_latency_us +=
        absl::ToInt64Microseconds(absl::Now() - start_verify);

    auto start_rs = absl::Now();
    LITERT_ASSIGN_OR_RETURN(
        auto rs_result,
        PerformRejectionSampling(
            draft_tokens,
            llm_inference_context_
                .verify_output_buffers[LlmSignatures::kVerifyLogitsOutput]));
    latency_stats_.decode_mtp_rejection_sampling_latency_us +=
        absl::ToInt64Microseconds(absl::Now() - start_rs);

    NPU_EXECUTOR_LOG(INFO) << "  MTP Accepted " << rs_result.num_accepted
                           << " draft tokens. Bonus: "
                           << rs_result.bonus_token_id;

    auto start_commit = absl::Now();
    ABSL_RETURN_IF_ERROR(CommitVerifiedKVCache(mtp_start_step));
    latency_stats_.decode_cache_update_inference_latency_us +=
        absl::ToInt64Microseconds(absl::Now() - start_commit);

    // Prepare tokens to be returned.
    std::vector<int> all_accepted;
    if (!has_valid_verify_activations_) {
      all_accepted.push_back(mtp_start_token_id);
    }

    for (int i = 0; i < rs_result.num_accepted; ++i) {
      all_accepted.push_back(draft_tokens[i]);
    }
    all_accepted.push_back(rs_result.bonus_token_id);

    // Prepare next activation slice.
    {
      auto start_act_copy = absl::Now();
      const auto& verify_activations_buffer =
          llm_inference_context_.verify_output_buffers
              [LlmSignatures::kLastLayerActivationsOutput];
      LITERT_ASSIGN_OR_RETURN(
          auto full_activations,
          CopyRawBytesFromTensorBuffer(verify_activations_buffer));
      // Divide total bytes by the sequence length (1 current token + N draft
      // tokens) to get the number of bytes per token activation.
      size_t dratfer_seq_len = draft_tokens.size() + 1;
      size_t hidden_size_in_bytes = full_activations.size() / dratfer_seq_len;
      last_verify_activations_.resize(hidden_size_in_bytes);
      memcpy(last_verify_activations_.data(),
             full_activations.data() +
                 rs_result.num_accepted * hidden_size_in_bytes,
             hidden_size_in_bytes);
      has_valid_verify_activations_ = true;
      latency_stats_.decode_mtp_activation_copy_latency_us +=
          absl::ToInt64Microseconds(absl::Now() - start_act_copy);
    }

    // Return the first token now, queue the rest for future Decode() calls.
    int first_token_id = all_accepted[0];
    for (size_t i = 1; i < all_accepted.size(); ++i) {
      pending_accepted_tokens_.push_back(all_accepted[i]);
    }

    NPU_EXECUTOR_LOG(INFO) << "MTP cycle returning first token: "
                           << first_token_id
                           << " (Queued: " << pending_accepted_tokens_.size()
                           << ")";

    std::shared_ptr<TokenData> first_token =
        std::make_shared<TokenData>(first_token_id);
    if (UseEmbeddingLookupManager()) {
      auto start_lookup = absl::Now();
      ABSL_RETURN_IF_ERROR(embedding_lookup_manager_->LookupDecode(
          first_token->id(), first_token->mutable_embedding()));
      latency_stats_.decode_embedder_inference_latency_us +=
          absl::ToInt64Microseconds(absl::Now() - start_lookup);
    }
    // For MTP, we need to mark them as processed so the next step's
    // GetNextUnprocessedToken works correctly.
    ABSL_RETURN_IF_ERROR(
        processed_tokens_.AddPendingInputToken({std::move(first_token)}));
    current_step_++;

    latency_stats_.decode_num_tokens++;
    latency_stats_.decode_e2e_latency_us +=
        absl::ToInt64Microseconds(absl::Now() - start);
    return std::vector<std::vector<int>>{{first_token_id}};

  } else {
    // Standard Non-Speculative path.
    const int max_index = mtp_start_token_id;

    // Store the sampled id as the pending input token for next Decode.

    std::shared_ptr<TokenData> last_output_token =
        std::make_shared<TokenData>(max_index);

    if (UseEmbeddingLookupManager()) {
      auto start_lookup = absl::Now();
      ABSL_RETURN_IF_ERROR(embedding_lookup_manager_->LookupDecode(
          last_output_token->id(), last_output_token->mutable_embedding()));
      latency_stats_.decode_embedder_inference_latency_us +=
          absl::ToInt64Microseconds(absl::Now() - start_lookup);
    }
    // For Gemma3 we don't need to do anything here because we invoke
    // the Embedder before invoking the transformer during prefill/decode. All
    // we need to do is keep the token id around (which is stored as the pending
    // token).

    auto start_add = absl::Now();
    ABSL_RETURN_IF_ERROR(
        processed_tokens_.AddPendingInputToken({std::move(last_output_token)}));
    latency_stats_.decode_token_queue_latency_us +=
        absl::ToInt64Microseconds(absl::Now() - start_add);

    ++current_step_;

    latency_stats_.decode_e2e_latency_us +=
        absl::ToInt64Microseconds(absl::Now() - start);
    latency_stats_.decode_num_tokens++;
    return std::vector<std::vector<int>>{{max_index}};
  }
}

// Prefill internal implementation, for one prefill call to the compiled model
// with a certain length.
absl::Status LlmLiteRtNpuCompiledModelExecutor::PrefillInternal(
    absl::string_view prefill_signature, absl::Span<const int> ids) {
  int runner_size = 0;
  for (const auto& [size, sig] : prefill_signature_map_) {
    if (sig == prefill_signature) {
      runner_size = size;
      break;
    }
  }
  if (runner_size == 0) {
    return absl::InternalError(absl::StrCat(
        "Could not find runner size for signature ", prefill_signature));
  }

  llm_inference_context_.ActivatePrefillSize(runner_size);
  rope_context_.ActivatePrefillSize(runner_size);
  mask_context_.ActivatePrefillSize(runner_size);
  cache_update_inference_context_.ActivatePrefillSize(runner_size);
  embedder_context_.inference_context.ActivatePrefillSize(runner_size);
  if (embedder_per_layer_context_.has_value()) {
    embedder_per_layer_context_->inference_context.ActivatePrefillSize(
        runner_size);
  }

  const auto& active_signatures = prefill_signatures_by_size_.at(runner_size);
  const int prefill_size = active_signatures.size;
  if (current_step_ + ids.size() > executor_settings_.GetMaxNumTokens()) {
    ABSL_LOG(ERROR) << "Prefill length exceeds user capacity. current_step_="
                    << current_step_ << ", ids.size()=" << ids.size()
                    << ", max_sequence_length_="
                    << executor_settings_.GetMaxNumTokens();
    return absl::InvalidArgumentError(
        absl::StrCat("Prefill length (", ids.size(), ") plus current step (",
                     current_step_, ") exceeds max sequence length (",
                     executor_settings_.GetMaxNumTokens(), ")."));
  }
  // We use `prefill_size` (the size of the prefill chunk signature) instead of
  // `ids.size()` because the NPU prefill execution always processes a fixed
  // block of `prefill_size` tokens (padding with zeros if necessary). The NPU
  // hardware cache update will attempt to write `prefill_size` tokens, which
  // can cause out-of-bounds writes if we are close to the end of the context
  // size.
  if (current_step_ + prefill_size > active_signatures.context_size) {
    ABSL_LOG(ERROR) << "Prefill length exceeds NPU hardware context size. "
                    << "current_step_=" << current_step_
                    << ", kPrefillSize=" << prefill_size
                    << ", context_size_=" << active_signatures.context_size;
    return absl::InvalidArgumentError(absl::StrCat(
        "Padded prefill length (", prefill_size, ") plus current step (",
        current_step_, ") exceeds NPU context size (",
        active_signatures.context_size, ")."));
  }
  auto start_prepare_inputs = absl::Now();
  std::vector<int> tokens_to_embed;
  tokens_to_embed.reserve(ids.size());
  {
    // Prefill input tokens.
    auto& tb = embedder_context_.inference_context
                   .prefill_input_buffers[EmbedderSignatures::kEmbedderInput];
    LITERT_ASSIGN_OR_RETURN(auto prefill_input_size, tb.Size());
    LITERT_ASSIGN_OR_RETURN(
        auto prefill_input_lock_and_addr,
        ::litert::TensorBufferScopedLock::Create(
            embedder_context_.inference_context
                .prefill_input_buffers[EmbedderSignatures::kEmbedderInput],
            ::litert::TensorBuffer::LockMode::kWrite));
    auto* prefill_input_ptr =
        static_cast<int32_t*>(prefill_input_lock_and_addr.second);

    // Prefill input position.
    LITERT_ASSIGN_OR_RETURN(
        auto prefill_input_pos_size,
        rope_context_.prefill_input_buffers[RopeSignatures::kInputPos].Size());
    LITERT_ASSIGN_OR_RETURN(
        auto prefill_input_pos_lock_and_addr,
        ::litert::TensorBufferScopedLock::Create(
            rope_context_.prefill_input_buffers[RopeSignatures::kInputPos],
            ::litert::TensorBuffer::LockMode::kWrite));
    auto* prefill_input_pos_ptr =
        static_cast<int32_t*>(prefill_input_pos_lock_and_addr.second);

    // Timestep input.
    LITERT_ASSIGN_OR_RETURN(
        auto prefill_timestep_size,
        mask_context_.prefill_input_buffers[MaskSignatures::kMaskInputTimeStep]
            .Size());
    LITERT_ASSIGN_OR_RETURN(
        auto prefill_timestep_lock_and_addr,
        ::litert::TensorBufferScopedLock::Create(
            mask_context_
                .prefill_input_buffers[MaskSignatures::kMaskInputTimeStep],
            ::litert::TensorBuffer::LockMode::kWrite));
    auto* prefill_timestep_ptr =
        static_cast<int32_t*>(prefill_timestep_lock_and_addr.second);

    // Mask input tokens.
    int32_t* mask_input_tokens_ptr = nullptr;
    std::optional<::litert::TensorBufferScopedLock> mask_input_tokens_lock;
    auto it = mask_context_.prefill_input_buffers.find(
        MaskSignatures::kMaskInputTokens);
    if (it != mask_context_.prefill_input_buffers.end()) {
      auto& buf = it->second;
      LITERT_ASSIGN_OR_RETURN(auto type, buf.TensorType());
      LITERT_ASSIGN_OR_RETURN(auto mask_input_tokens_size, buf.Size());
      LITERT_ASSIGN_OR_RETURN(auto num_elements, type.Layout().NumElements());
      LITERT_ASSIGN_OR_RETURN(auto buffer_type, buf.BufferType());
      NPU_EXECUTOR_LOG(INFO)
          << "mask_input_tokens type: " << (int)type.ElementType()
          << " size: " << mask_input_tokens_size
          << " num_elements: " << num_elements
          << " buffer_type: " << (int)buffer_type;
      auto lock_res = ::litert::TensorBufferScopedLock::Create(
          buf, ::litert::TensorBuffer::LockMode::kWrite);
      if (!lock_res) {
        NPU_EXECUTOR_LOG(ERROR) << "Failed to lock mask_input_tokens: "
                                << lock_res.Error().Message();
        return ::litert::ErrorStatusBuilder(lock_res.Error());
      }
      auto lock = std::move(*lock_res);
      mask_input_tokens_ptr = static_cast<int32_t*>(lock.second);
      mask_input_tokens_lock.emplace(std::move(lock.first));
      memset(mask_input_tokens_ptr, 0, mask_input_tokens_size);
    }

    bool* valid_mask_ptr = nullptr;
    std::optional<::litert::TensorBufferScopedLock> valid_mask_lock;
    auto valid_mask_it = mask_context_.prefill_input_buffers.find(
        MaskSignatures::kMaskInputValidMask);
    if (valid_mask_it != mask_context_.prefill_input_buffers.end()) {
      auto& buf = valid_mask_it->second;
      LITERT_ASSIGN_OR_RETURN(auto type, buf.TensorType());
      RET_CHECK(type.ElementType() == ::litert::ElementType::Bool)
          << "valid_mask tensor must have Bool element type.";
      LITERT_ASSIGN_OR_RETURN(
          auto lock, ::litert::TensorBufferScopedLock::Create(
                         buf, ::litert::TensorBuffer::LockMode::kWrite));
      valid_mask_ptr = static_cast<bool*>(lock.second);
      valid_mask_lock.emplace(std::move(lock.first));
    }

    memset(prefill_input_ptr, 0, prefill_input_size);
    memset(prefill_input_pos_ptr, 0, prefill_input_pos_size);
    memset(prefill_timestep_ptr, 0, prefill_timestep_size);

    if (processed_tokens_.TokenCount() != current_step_) {
      ABSL_RETURN_IF_ERROR(processed_tokens_.RollBackToStep(current_step_));
    }
    // Check if have a pending input token. Note that 'internal_start_step' is
    // always equal to the number of processed tokens plus 1.
    auto [internal_start_step, pending_input_token] =
        processed_tokens_.GetNextUnprocessedToken();
    int input_idx = 0;
    if (!pending_input_token.empty()) {
      tokens_to_embed.push_back(pending_input_token[0]->id());
      // We'll write any pending embedding directly into the transformer
      // embedding buffer.
      if (UseEmbeddingLookupManager()) {
        LITERT_ASSIGN_OR_RETURN(
            auto transformer_embedding_buffer_lock_and_addr,
            ::litert::TensorBufferScopedLock::Create(
                llm_inference_context_
                    .prefill_input_buffers[LlmSignatures::kInputEmbeddings],
                ::litert::TensorBuffer::LockMode::kWrite));
        float* transformer_embedding_buffer_ptr = static_cast<float*>(
            transformer_embedding_buffer_lock_and_addr.second);
        memcpy(transformer_embedding_buffer_ptr,
               pending_input_token[0]->embedding().data(),
               pending_input_token[0]->embedding().size() * sizeof(float));
      }

      prefill_input_ptr[input_idx] = pending_input_token[0]->id();
      prefill_input_pos_ptr[input_idx] = internal_start_step;
      if (mask_input_tokens_ptr) {
        mask_input_tokens_ptr[input_idx] = pending_input_token[0]->id();
      }
      ABSL_RETURN_IF_ERROR(
          processed_tokens_.MarkPendingInputTokenAsProcessed());
      ++input_idx;
    }

    prefill_timestep_ptr[0] = internal_start_step;
    std::vector<int> processed_input_tokens;
    // We will not fill the last token of the current input into the compiled
    // model input buffers just yet. It will be stored in the
    // 'processed_tokens_' and used in the next prefill or decode.
    processed_input_tokens.reserve(ids.size() - 1);
    for (int i = 0; i < ids.size() - 1; input_idx++, current_step_++, i++) {
      tokens_to_embed.push_back(ids[i]);
      prefill_input_ptr[input_idx] = ids[i];
      prefill_input_pos_ptr[input_idx] = current_step_;
      if (mask_input_tokens_ptr) {
        mask_input_tokens_ptr[input_idx] = ids[i];
      }
      processed_input_tokens.push_back(ids[i]);
    }
    processed_tokens_.AddProcessedTokens(processed_input_tokens);
    if (!processed_input_tokens.empty()) {
      NPU_EXECUTOR_LOG(INFO)
          << "Prefill tokens: " << FormatFirstN<int>(processed_input_tokens);
    }

    if (valid_mask_ptr) {
      auto& buf = valid_mask_it->second;
      LITERT_ASSIGN_OR_RETURN(auto type, buf.TensorType());
      LITERT_ASSIGN_OR_RETURN(auto num_elements, type.Layout().NumElements());
      for (size_t i = 0; i < num_elements; ++i) {
        valid_mask_ptr[i] = (i < input_idx);
      }
    }

    auto end_prepare_inputs = absl::Now();
    latency_stats_.prefill_prepare_input_latency_us +=
        absl::ToInt64Microseconds(end_prepare_inputs - start_prepare_inputs);

    if (UseEmbeddingLookupManager()) {
      auto start = absl::Now();
      // We use the embedding lookup manager to populate the embedding buffer.
      // If we already placed a pending input token into the embedding buffer
      // before, we'll flag that as an offset to the embedding lookup manager.
      litert::TensorBuffer& embedding_buffer =
          llm_inference_context_
              .prefill_input_buffers[LlmSignatures::kInputEmbeddings];
      ABSL_RETURN_IF_ERROR(embedding_lookup_manager_->LookupPrefill(
          processed_input_tokens, &embedding_buffer,
          pending_input_token.empty() ? 0 : 1));
      latency_stats_.prefill_embedder_inference_latency_us +=
          absl::ToInt64Microseconds(absl::Now() - start);
    }
  }

  // Add the last token of the current input as a pending input token, to be
  // used in the next prefill or decode.
  std::shared_ptr<TokenData> last_input_token =
      std::make_shared<TokenData>(ids.back());

  if (UseEmbeddingLookupManager()) {
    auto start = absl::Now();
    // Look up the embeddings for the last token so they can be used in the next
    // prefill or decode. This has to be done now in the case of multi-modal
    // prefill so the embeddings are used in the correct order.
    ABSL_RETURN_IF_ERROR(embedding_lookup_manager_->LookupPrefill(
        last_input_token->id(), last_input_token->mutable_embedding()));
    latency_stats_.prefill_embedder_inference_latency_us +=
        absl::ToInt64Microseconds(absl::Now() - start);
  }

  // Add the last input token to the pending input token list.
  ABSL_RETURN_IF_ERROR(
      processed_tokens_.AddPendingInputToken({std::move(last_input_token)}));
  ++current_step_;

  if (!UseEmbeddingLookupManager()) {
    // Invoke embedder signature for Gemma3, because we don't have the
    // embedding lookup manager to do it for us.
    auto start = absl::Now();
    auto res = embedder_context_.embedder_compiled_model.Run(
        active_signatures.embedder,
        embedder_context_.inference_context.prefill_input_buffers,
        embedder_context_.inference_context.prefill_output_buffers);
    RET_CHECK(res) << "Failed to run embedder model." << res.Error().Message();
    auto end = absl::Now();
    latency_stats_.prefill_embedder_inference_latency_us +=
        absl::ToInt64Microseconds(end - start);
  }

  // Use tokens_to_embed for PLE lookup since we need the lookup for the
  // pending token (if available) and ids[0, last_token-1].
  if (use_hw_ple_for_npu_ && !ple_table_ptrs_.empty()) {
    auto start = absl::Now();
    auto& ple_output_buffer =
        llm_inference_context_.prefill_input_buffers[kPerLayerEmbedderTensor];
    LITERT_ASSIGN_OR_RETURN(
        auto lock,
        ::litert::TensorBufferScopedLock::Create(
            ple_output_buffer, ::litert::TensorBuffer::LockMode::kWrite));
    void* output_ptr = lock.second;

    ABSL_RETURN_IF_ERROR(HWPerLayerEmbeddingLookup(
        tokens_to_embed.data(), tokens_to_embed.size(), ple_table_ptrs_.data(),
        ple_quant_params_.data(), num_tables_, ple_embedding_dim_, output_ptr,
        output_type_, ple_table_element_type_, mul_scale_, output_scale_,
        final_zero_point_));

    latency_stats_.prefill_embedder_per_layer_inference_latency_us +=
        absl::ToInt64Microseconds(absl::Now() - start);
  } else if (embedder_per_layer_context_.has_value()) {
    // Invoke embedder per layer signature if it exists.
    auto start = absl::Now();
    auto res =
        embedder_per_layer_context_->embedder_per_layer_compiled_model.Run(
            active_signatures.embedder_per_layer,
            embedder_per_layer_context_->inference_context
                .prefill_input_buffers,
            embedder_per_layer_context_->inference_context
                .prefill_output_buffers);
    RET_CHECK(res) << "Failed to run embedder per layer model."
                   << res.Error().Message();
    latency_stats_.prefill_embedder_per_layer_inference_latency_us +=
        absl::ToInt64Microseconds(absl::Now() - start);
  }

  return PrefillCommonPipeline(active_signatures);
}

absl::Status LlmLiteRtNpuCompiledModelExecutor::PrefillInternalFromEmbeddings(
    absl::string_view prefill_signature,
    absl::Span<const int32_t> sliced_tokens, absl::Span<const float> embeddings,
    absl::Span<const float> ple_embeddings,
    absl::Span<const int32_t> seq_positions) {
  int runner_size = 0;
  for (const auto& [size, sig] : prefill_signature_map_) {
    if (sig == prefill_signature) {
      runner_size = size;
      break;
    }
  }
  if (runner_size == 0) {
    return absl::InternalError(absl::StrCat(
        "Could not find runner size for signature ", prefill_signature));
  }

  llm_inference_context_.ActivatePrefillSize(runner_size);
  rope_context_.ActivatePrefillSize(runner_size);
  mask_context_.ActivatePrefillSize(runner_size);
  cache_update_inference_context_.ActivatePrefillSize(runner_size);
  embedder_context_.inference_context.ActivatePrefillSize(runner_size);
  if (embedder_per_layer_context_.has_value()) {
    embedder_per_layer_context_->inference_context.ActivatePrefillSize(
        runner_size);
  }

  const auto& active_signatures = prefill_signatures_by_size_.at(runner_size);
  NPU_EXECUTOR_LOG(INFO) << "PrefillInternalFromEmbeddings called";
  NPU_EXECUTOR_LOG(INFO) << "PLE config: type="
                         << static_cast<int>(output_type_)
                         << ", mul_scale=" << mul_scale_
                         << ", output_scale=" << output_scale_
                         << ", zero_point=" << final_zero_point_;
  if (!sliced_tokens.empty()) {
    NPU_EXECUTOR_LOG(INFO) << "Prefill tokens: " << FormatFirstN(sliced_tokens);
  }
  if (!embeddings.empty()) {
    NPU_EXECUTOR_LOG(INFO) << "Prefill embeddings: "
                           << FormatFirstN(embeddings);
  }
  if (!ple_embeddings.empty()) {
    NPU_EXECUTOR_LOG(INFO) << "Prefill PLE embeddings: "
                           << FormatFirstN(ple_embeddings);
  }
  // Set prefill input embeddings.
  {
    auto& buffer = llm_inference_context_
                       .prefill_input_buffers[LlmSignatures::kInputEmbeddings];
    auto tensor_type = buffer.TensorType();
    NPU_EXECUTOR_LOG(INFO) << "Embeddings buffer element type: "
                           << (tensor_type.HasValue()
                                   ? static_cast<int>(
                                         tensor_type->ElementType())
                                   : -1);
    if (!tensor_type.HasValue()) {
      return absl::InternalError(
          "Failed to get prefill input embeddings tensor type.");
    }
    const auto elem_type = tensor_type->ElementType();
    const size_t elem_size = elem_type == litert::ElementType::Float16
                                 ? sizeof(tflite::half)
                                 : sizeof(float);

    LITERT_ASSIGN_OR_RETURN(size_t buffer_size, buffer.PackedSize());
    RET_CHECK_GE(buffer_size, embeddings.size() * elem_size);

    LITERT_ASSIGN_OR_RETURN(
        auto lock_and_addr,
        ::litert::TensorBufferScopedLock::Create(
            buffer, ::litert::TensorBuffer::LockMode::kWrite));

    std::vector<float> default_emb;
    if (UseEmbeddingLookupManager()) {
      auto* text_lookup = embedding_lookup_manager_->GetTextEmbeddingLookup();
      if (text_lookup != nullptr) {
        default_emb = text_lookup->GetDefaultEmbeddingVector();
      }
    }

    const auto& dims = tensor_type->Layout().Dimensions();
    if (dims.size() < 3) {
      return absl::InternalError(
          "Prefill input embeddings tensor has unexpected shape.");
    }
    const size_t embedding_dim =
        default_emb.empty() ? dims[2] : default_emb.size();
    size_t starting_token = embeddings.size() / embedding_dim;
    size_t num_tokens_to_fill = buffer_size / (embedding_dim * elem_size);

    if (elem_type == litert::ElementType::Float16) {
      tflite::half* buffer_ptr =
          static_cast<tflite::half*>(lock_and_addr.second);
      for (size_t i = 0; i < embeddings.size(); ++i) {
        buffer_ptr[i] = tflite::half(embeddings[i]);
      }
      tflite::half* padding_ptr = buffer_ptr + starting_token * embedding_dim;
      std::vector<tflite::half> fp16_default_emb(embedding_dim,
                                                 tflite::half(0.0f));
      if (default_emb.size() == embedding_dim) {
        for (size_t i = 0; i < embedding_dim; ++i) {
          fp16_default_emb[i] = tflite::half(default_emb[i]);
        }
      }
      for (size_t i = starting_token; i < num_tokens_to_fill; ++i) {
        std::memcpy(padding_ptr, fp16_default_emb.data(),
                    embedding_dim * sizeof(tflite::half));
        padding_ptr += embedding_dim;
      }
    } else {
      float* buffer_ptr = static_cast<float*>(lock_and_addr.second);
      std::memcpy(buffer_ptr, embeddings.data(),
                  embeddings.size() * sizeof(float));
      float* padding_ptr = buffer_ptr + starting_token * embedding_dim;
      if (default_emb.size() == embedding_dim) {
        for (size_t i = starting_token; i < num_tokens_to_fill; ++i) {
          std::memcpy(padding_ptr, default_emb.data(),
                      embedding_dim * sizeof(float));
          padding_ptr += embedding_dim;
        }
      } else {
        std::memset(padding_ptr, 0,
                    (num_tokens_to_fill - starting_token) * embedding_dim *
                        sizeof(float));
      }
    }
  }

  // Set prefill positions.
  {
    auto& buffer =
        rope_context_.prefill_input_buffers[RopeSignatures::kInputPos];
    LITERT_ASSIGN_OR_RETURN(size_t buffer_size, buffer.PackedSize());
    RET_CHECK_GE(buffer_size, seq_positions.size() * sizeof(int32_t));

    LITERT_ASSIGN_OR_RETURN(
        auto lock_and_addr,
        ::litert::TensorBufferScopedLock::Create(
            buffer, ::litert::TensorBuffer::LockMode::kWrite));
    int32_t* buffer_ptr = static_cast<int32_t*>(lock_and_addr.second);

    std::memcpy(buffer_ptr, seq_positions.data(),
                seq_positions.size() * sizeof(int32_t));

    size_t starting_token = seq_positions.size();
    size_t num_tokens_to_fill = buffer_size / sizeof(int32_t);
    std::memset(buffer_ptr + starting_token, 0,
                (num_tokens_to_fill - starting_token) * sizeof(int32_t));
  }

  // Set prefill per-layer embeddings if provided.
  if (!ple_embeddings.empty()) {
    auto& buffer =
        llm_inference_context_.prefill_input_buffers[kPerLayerEmbedderTensor];

    std::vector<float> default_ple_emb;
    if (per_layer_embedding_lookup_manager_ == nullptr &&
        embedder_per_layer_model_ != nullptr) {
      LITERT_ASSIGN_OR_RETURN(per_layer_embedding_lookup_manager_,
                              EmbeddingLookupManager::Create(
                                  env_, embedder_per_layer_model_, false));
    }
    if (per_layer_embedding_lookup_manager_ != nullptr) {
      auto* ple_lookup =
          per_layer_embedding_lookup_manager_->GetTextEmbeddingLookup();
      if (ple_lookup != nullptr) {
        default_ple_emb = ple_lookup->GetDefaultEmbeddingVector();
      }
    }

    LITERT_ASSIGN_OR_RETURN(RankedTensorType tensor_type, buffer.TensorType());
    const auto& dims = tensor_type.Layout().Dimensions();
    if (dims.size() < 3) {
      return absl::InternalError(
          "Prefill per-layer embeddings tensor has unexpected shape.");
    }
    const size_t ple_dim =
        default_ple_emb.empty() ? dims[2] : default_ple_emb.size();
    size_t starting_token = ple_embeddings.size() / ple_dim;

    ABSL_RETURN_IF_ERROR(WriteAndPadPleEmbeddings(
        buffer, ple_embeddings, ple_dim, starting_token, default_ple_emb,
        output_type_, output_scale_, final_zero_point_));
  }

  // Set prefill mask timestep.
  {
    auto& buffer =
        mask_context_.prefill_input_buffers[MaskSignatures::kMaskInputTimeStep];
    LITERT_ASSIGN_OR_RETURN(size_t buffer_size, buffer.PackedSize());
    RET_CHECK_GE(buffer_size, sizeof(int32_t));
    int32_t step = seq_positions[0];
    LITERT_RETURN_IF_ERROR(buffer.Write(absl::MakeConstSpan(&step, 1)));
  }

  // Set temporary token IDs to prefill mask context to enable hardware masking.
  if (!sliced_tokens.empty()) {
    auto it = mask_context_.prefill_input_buffers.find(
        MaskSignatures::kMaskInputTokens);
    if (it != mask_context_.prefill_input_buffers.end()) {
      auto& input_tokens_buffer = it->second;

      LITERT_ASSIGN_OR_RETURN(size_t buffer_size,
                              input_tokens_buffer.PackedSize());
      size_t num_tokens_to_fill = buffer_size / sizeof(int32_t);
      RET_CHECK_LE(sliced_tokens.size(), num_tokens_to_fill)
          << "sliced_tokens size exceeds input tokens buffer capacity.";

      std::vector<int32_t> temp_tokens(num_tokens_to_fill, 0);
      absl::c_copy(sliced_tokens, temp_tokens.begin());
      for (auto& tid : temp_tokens) {
        if (tid < 0) {
          tid = 0;
        }
      }
      LITERT_RETURN_IF_ERROR(
          input_tokens_buffer.Write(absl::MakeConstSpan(temp_tokens)));
    }
  }

  return PrefillCommonPipeline(active_signatures);
}

absl::Status LlmLiteRtNpuCompiledModelExecutor::PrefillCommonPipeline(
    const ResolvedSignatures& active_signatures) {
  // Invoke RoPE signature.
  {
    auto start = absl::Now();
    auto res = npu_auxiliary_context_.npu_auxiliary_compiled_model.Run(
        active_signatures.rope, rope_context_.prefill_input_buffers,
        rope_context_.prefill_output_buffers);
    RET_CHECK(res) << "Failed to run RoPE model." << res.Error().Message();
    auto end = absl::Now();
    latency_stats_.prefill_rope_inference_latency_us +=
        absl::ToInt64Microseconds(end - start);
  }

  // Invoke mask signature.
  {
    auto start = absl::Now();
    if (prefill_mask_update_method_ == MaskUpdateMethod::kWH) {
      ABSL_RETURN_IF_ERROR(HWMaskUpdate(mask_context_.prefill_input_buffers,
                                        mask_context_.prefill_output_buffers));
    } else {
      auto res = npu_auxiliary_context_.npu_auxiliary_compiled_model.Run(
          active_signatures.mask, mask_context_.prefill_input_buffers,
          mask_context_.prefill_output_buffers);
      RET_CHECK(res) << "Failed to run compiled model."
                     << res.Error().Message();
    }
    auto end = absl::Now();
    latency_stats_.prefill_mask_inference_latency_us +=
        absl::ToInt64Microseconds(end - start);
  }

  // Invoke LLM signature.
  {
    auto start = absl::Now();
    auto res = llm_compiled_model_.Run(
        active_signatures.prefill, llm_inference_context_.prefill_input_buffers,
        llm_inference_context_.prefill_output_buffers);
    RET_CHECK(res) << "Failed to run LLM model." << res.Error().Message();
    auto end = absl::Now();
    latency_stats_.prefill_llm_inference_latency_us +=
        absl::ToInt64Microseconds(end - start);
  }

  // Cache update.
  {
    auto start = absl::Now();
    if (prefill_kv_cache_update_method_ == KVCacheUpdateMethod::kWH) {
      ABSL_RETURN_IF_ERROR(HWKVCacheUpdate(
          cache_update_inference_context_.prefill_input_buffers,
          cache_update_inference_context_.prefill_output_buffers,
          kv_quant_params_, has_sliding_window_attention_));
    } else {
      auto res = npu_auxiliary_context_.npu_auxiliary_compiled_model.Run(
          active_signatures.cache_update,
          cache_update_inference_context_.prefill_input_buffers,
          cache_update_inference_context_.prefill_output_buffers);
      RET_CHECK(res) << "Failed to run cache update model."
                     << res.Error().Message();
    }
    auto end = absl::Now();
    latency_stats_.prefill_cache_update_inference_latency_us +=
        absl::ToInt64Microseconds(end - start);
  }

  return absl::OkStatus();
}

absl::Status LlmLiteRtNpuCompiledModelExecutor::DecodeInternal(
    int step, std::shared_ptr<TokenData> token) {
  ABSL_RETURN_IF_ERROR(SwitchContextSizeIfRequired(step + 1, step));

  if (step >= executor_settings_.GetMaxNumTokens()) {
    return absl::ResourceExhaustedError("Reached maximum number of tokens.");
  }
  int id = token->id();
  auto start_prepare_inputs = absl::Now();

  {
    if (id == kInvalidTokenId && token->embedding().empty()) {
      return absl::InvalidArgumentError("No id available to be decoded.");
    }

    if (id != kInvalidTokenId) {
      // Decode input tokens.
      ABSL_RETURN_IF_ERROR(SetFirstElement(
          embedder_context_.inference_context
              .decode_input_buffers[EmbedderSignatures::kEmbedderInput],
          id));
    }

    // Always update decode input position and timestep, even if
    // run_rope_and_mask is false. The LLM and Cache Update models still need to
    // know the current step.

    // 1. RoPE position
    ABSL_RETURN_IF_ERROR(SetFirstElement(
        rope_context_.decode_input_buffers[RopeSignatures::kInputPos], step));

    // 2. Mask timestep
    ABSL_RETURN_IF_ERROR(SetFirstElement(
        mask_context_.decode_input_buffers[MaskSignatures::kMaskInputTimeStep],
        step));

    if (mask_context_.decode_input_buffers.contains(
            MaskSignatures::kMaskInputTokens)) {
      ABSL_RETURN_IF_ERROR(SetFirstElement(
          mask_context_.decode_input_buffers[MaskSignatures::kMaskInputTokens],
          id));
    }
    if (mask_context_.decode_input_buffers.contains(
            MaskSignatures::kMaskInputValidMask)) {
      auto& buf =
          mask_context_
              .decode_input_buffers[MaskSignatures::kMaskInputValidMask];
      LITERT_ASSIGN_OR_RETURN(
          auto lock, ::litert::TensorBufferScopedLock::Create(
                         buf, ::litert::TensorBuffer::LockMode::kWrite));
      static_cast<bool*>(lock.second)[0] = true;
    }

    // 3. Cache update position
    ABSL_RETURN_IF_ERROR(SetFirstElement(
        cache_update_inference_context_
            .decode_input_buffers[CacheUpdateSignatures::kInputPos],
        step));
  }

  auto end_prepare_inputs = absl::Now();
  latency_stats_.decode_prepare_input_latency_us +=
      absl::ToInt64Microseconds(end_prepare_inputs - start_prepare_inputs);

  if (id != kInvalidTokenId && !UseEmbeddingLookupManager()) {
    // Invoke embedder signature for Gemma3, because we don't have the embedding
    // lookup manager to do it for us.
    {
      auto start = absl::Now();
      auto res = embedder_context_.embedder_compiled_model.Run(
          active_signatures_.decode_embedder,
          embedder_context_.inference_context.decode_input_buffers,
          embedder_context_.inference_context.decode_output_buffers);
      RET_CHECK(res) << "Failed to run embedder model."
                     << res.Error().Message();
      auto end = absl::Now();
      latency_stats_.decode_embedder_inference_latency_us +=
          absl::ToInt64Microseconds(end - start);
    }
  }

  if (UseEmbeddingLookupManager() || id == kInvalidTokenId) {
    // We'll write any pending embedding directly into the transformer
    // embedding buffer.
    auto start = absl::Now();
    LITERT_ASSIGN_OR_RETURN(
        auto transformer_embedding_buffer_lock_and_addr,
        ::litert::TensorBufferScopedLock::Create(
            llm_inference_context_
                .decode_input_buffers[LlmSignatures::kInputEmbeddings],
            ::litert::TensorBuffer::LockMode::kWrite));
    float* transformer_embedding_buffer_ptr =
        static_cast<float*>(transformer_embedding_buffer_lock_and_addr.second);
    LITERT_RETURN_IF_ERROR(!token->embedding().empty())
        << "Token embedding is empty.";
    memcpy(transformer_embedding_buffer_ptr, token->embedding().data(),
           token->embedding().size() * sizeof(float));

    // Log the input embedding for comparison

    latency_stats_.decode_embedder_inference_latency_us +=
        absl::ToInt64Microseconds(absl::Now() - start);
  }

  {
    if (!token->per_layer_embedding().empty()) {
      auto& buffer =
          llm_inference_context_.decode_input_buffers[kPerLayerEmbedderTensor];
      ABSL_RETURN_IF_ERROR(
          WritePleEmbeddings(buffer, token->per_layer_embedding(), output_type_,
                             output_scale_, final_zero_point_));
    } else if (use_hw_ple_for_npu_ && !ple_table_ptrs_.empty()) {
      auto start = absl::Now();
      auto& ple_output_buffer =
          llm_inference_context_.decode_input_buffers[kPerLayerEmbedderTensor];
      LITERT_ASSIGN_OR_RETURN(
          auto lock,
          ::litert::TensorBufferScopedLock::Create(
              ple_output_buffer, ::litert::TensorBuffer::LockMode::kWrite));
      void* output_ptr = lock.second;

      int id = token->id();
      ABSL_RETURN_IF_ERROR(HWPerLayerEmbeddingLookup(
          &id, 1, ple_table_ptrs_.data(), ple_quant_params_.data(), num_tables_,
          ple_embedding_dim_, output_ptr, output_type_, ple_table_element_type_,
          mul_scale_, output_scale_, final_zero_point_));

      latency_stats_.decode_embedder_per_layer_inference_latency_us +=
          absl::ToInt64Microseconds(absl::Now() - start);

    } else if (embedder_per_layer_context_.has_value()) {
      auto start = absl::Now();
      auto res =
          embedder_per_layer_context_->embedder_per_layer_compiled_model.Run(
              active_signatures_.decode_embedder_per_layer,
              embedder_per_layer_context_->inference_context
                  .decode_input_buffers,
              embedder_per_layer_context_->inference_context
                  .decode_output_buffers);
      RET_CHECK(res) << "Failed to run embedder per layer model."
                     << res.Error().Message();
      latency_stats_.decode_embedder_per_layer_inference_latency_us +=
          absl::ToInt64Microseconds(absl::Now() - start);

      // Log device-computed PLE embeddings.
      auto& ple_output_buffer =
          embedder_per_layer_context_->inference_context.decode_output_buffers
              [EmbedderPerLayerSignatures::kEmbedderOutput];
      LITERT_ASSIGN_OR_RETURN(size_t ple_buffer_size,
                              ple_output_buffer.PackedSize());
      std::vector<float> ple_host(ple_buffer_size / sizeof(float));
      auto read_status = ple_output_buffer.Read(absl::MakeSpan(ple_host));
      if (read_status.HasValue()) {
        NPU_EXECUTOR_LOG(INFO)
            << "Device-computed PLE embeddings (first few): " << ple_host[0]
            << ", " << ple_host[1] << ", " << ple_host[2];
      } else {
        NPU_EXECUTOR_LOG(WARNING) << "Failed to read PLE buffer from device: "
                                  << read_status.Error().Message();
      }
    }
  }

  // Invoke RoPE signature.
  {
    auto start = absl::Now();
    auto res = npu_auxiliary_context_.npu_auxiliary_compiled_model.Run(
        active_signatures_.decode_rope, rope_context_.decode_input_buffers,
        rope_context_.decode_output_buffers);
    RET_CHECK(res) << "Failed to run RoPE model." << res.Error().Message();
    auto end = absl::Now();
    latency_stats_.decode_rope_inference_latency_us +=
        absl::ToInt64Microseconds(end - start);
  }

  // Invoke mask signature.
  {
    auto start = absl::Now();
    if (decode_mask_update_method_ == MaskUpdateMethod::kWH) {
      ABSL_RETURN_IF_ERROR(HWMaskUpdate(mask_context_.decode_input_buffers,
                                        mask_context_.decode_output_buffers));
    } else {
      auto res = npu_auxiliary_context_.npu_auxiliary_compiled_model.Run(
          active_signatures_.decode_mask, mask_context_.decode_input_buffers,
          mask_context_.decode_output_buffers);
      RET_CHECK(res) << "Failed to run compiled model."
                     << res.Error().Message();
    }
    auto end = absl::Now();
    latency_stats_.decode_mask_inference_latency_us +=
        absl::ToInt64Microseconds(end - start);
  }

  // Invoke LLM signature.
  {
    auto start = absl::Now();
    auto res = llm_compiled_model_.Run(
        active_signatures_.decode, llm_inference_context_.decode_input_buffers,
        llm_inference_context_.decode_output_buffers);
    auto end = absl::Now();
    latency_stats_.decode_llm_inference_latency_us +=
        absl::ToInt64Microseconds(end - start);
    RET_CHECK(res) << "Failed to run LLM model." << res.Error().Message();
  }

  // Cache update.
  {
    auto start = absl::Now();
    if (decode_kv_cache_update_method_ == KVCacheUpdateMethod::kWH) {
      ABSL_RETURN_IF_ERROR(
          HWKVCacheUpdate(cache_update_inference_context_.decode_input_buffers,
                          cache_update_inference_context_.decode_output_buffers,
                          kv_quant_params_, has_sliding_window_attention_));
    } else {
      auto res = npu_auxiliary_context_.npu_auxiliary_compiled_model.Run(
          active_signatures_.decode_cache_update,
          cache_update_inference_context_.decode_input_buffers,
          cache_update_inference_context_.decode_output_buffers);
      RET_CHECK(res) << "Failed to run cache update model."
                     << res.Error().Message();
    }
    auto end = absl::Now();
    latency_stats_.decode_cache_update_inference_latency_us +=
        absl::ToInt64Microseconds(end - start);
  }
  return absl::OkStatus();
}

absl::Status LlmLiteRtNpuCompiledModelExecutor::DecodeSingleToken(
    size_t idx, absl::Span<const int32_t> seq_pos_span,
    absl::Span<const int32_t> tokens_span, absl::Span<const float> embeddings,
    size_t embedding_dim, absl::Span<const float> ple_embeddings,
    size_t ple_dim) {
  int step = seq_pos_span[idx];
  int token_id = tokens_span.empty() ? -1 : tokens_span[idx];
  auto token = std::make_shared<TokenData>(token_id);
  if (!embeddings.empty()) {
    token->mutable_embedding() =
        std::vector<float>(embeddings.begin() + idx * embedding_dim,
                           embeddings.begin() + (idx + 1) * embedding_dim);
  }
  if (!ple_embeddings.empty()) {
    token->mutable_per_layer_embedding() =
        std::vector<float>(ple_embeddings.begin() + idx * ple_dim,
                           ple_embeddings.begin() + (idx + 1) * ple_dim);
  }

  ABSL_RETURN_IF_ERROR(DecodeInternal(step, token));
  current_step_ = step + 1;
  return absl::OkStatus();
}

absl::StatusOr<std::vector<int>>
LlmLiteRtNpuCompiledModelExecutor::RunDrafterLoop(int start_step,
                                                  int current_token_id) {
  if (!drafter_context_.has_value() || !drafter_aux_context_.has_value()) {
    return absl::InternalError("Drafter contexts not initialized.");
  }
  // Get model drafter sequence length.
  LITERT_ASSIGN_OR_RETURN(
      RankedTensorType verify_output_tensor_type,
      llm_inference_context_
          .verify_output_buffers[MtpSignatures::kInputActivations]
          .TensorType());
  const int drafter_sequence_length =
      verify_output_tensor_type.Layout().Dimensions()[1] - 1;
  auto& ctx = drafter_context_.value();
  auto& aux_ctx = drafter_aux_context_.value();

  std::vector<int> draft_tokens;

  // Initial activations from the last layer of the main LLM (or previous
  // cycle).
  std::vector<uint8_t> current_activations;
  if (!has_valid_verify_activations_) {
    LITERT_ASSIGN_OR_RETURN(
        current_activations,
        CopyRawBytesFromTensorBuffer(
            llm_inference_context_.decode_output_buffers
                [LlmSignatures::kLastLayerActivationsOutput]));
  } else {
    current_activations = last_verify_activations_;
  }

  int input_token_id = current_token_id;

  {
    // Lock the rope input pos buffer for writing.
    LITERT_ASSIGN_OR_RETURN(
        auto pos_lock, ::litert::TensorBufferScopedLock::Create(
                           aux_ctx.rope_input_buffers[MtpSignatures::kInputPos],
                           ::litert::TensorBuffer::LockMode::kWrite));
    static_cast<int32_t*>(pos_lock.second)[0] = start_step;

    // Lock the mask input time step buffer for writing.
    LITERT_ASSIGN_OR_RETURN(
        auto mask_lock,
        ::litert::TensorBufferScopedLock::Create(
            aux_ctx.mask_input_buffers[MtpSignatures::kInputTimeStep],
            ::litert::TensorBuffer::LockMode::kWrite));
    static_cast<int32_t*>(mask_lock.second)[0] = start_step;

    LITERT_ASSIGN_OR_RETURN(
        auto token_lock,
        ::litert::TensorBufferScopedLock::Create(
            aux_ctx.mask_input_buffers[MtpSignatures::kInputTokens],
            ::litert::TensorBuffer::LockMode::kWrite));
    static_cast<int32_t*>(token_lock.second)[0] = input_token_id;

    // Lock the drafter input pos buffer for writing.
    LITERT_ASSIGN_OR_RETURN(auto drafter_pos_lock,
                            ::litert::TensorBufferScopedLock::Create(
                                ctx.mtp_input_buffers[MtpSignatures::kInputPos],
                                ::litert::TensorBuffer::LockMode::kWrite));
    static_cast<int32_t*>(drafter_pos_lock.second)[0] = start_step;
  }

  // Run Rope/Mask for drafter.
  auto start = absl::Now();
  LITERT_RETURN_IF_ERROR(aux_ctx.mtp_aux_compiled_model.Run(
      MtpSignatures::kMtpRope, aux_ctx.rope_input_buffers,
      aux_ctx.rope_output_buffers));
  auto end = absl::Now();
  latency_stats_.decode_rope_inference_latency_us +=
      absl::ToInt64Microseconds(end - start);

  start = absl::Now();
  if (mtp_mask_update_method_ == MaskUpdateMethod::kWH) {
    ABSL_RETURN_IF_ERROR(
        HWMaskUpdate(aux_ctx.mask_input_buffers, aux_ctx.mask_output_buffers));
  } else {
    LITERT_RETURN_IF_ERROR(aux_ctx.mtp_aux_compiled_model.Run(
        MtpSignatures::kMtpMask, aux_ctx.mask_input_buffers,
        aux_ctx.mask_output_buffers));
  }
  end = absl::Now();
  latency_stats_.decode_mask_inference_latency_us +=
      absl::ToInt64Microseconds(end - start);

  for (int i = 0; i < drafter_sequence_length; ++i) {
    std::vector<float> draft_embedding;
    if (UseEmbeddingLookupManager()) {
      auto start = absl::Now();
      auto* text_lookup = embedding_lookup_manager_->GetTextEmbeddingLookup();
      if (text_lookup == nullptr) {
        return absl::InternalError("Text embedding lookup not available.");
      }
      draft_embedding.resize(text_lookup->GetFloatsPerToken());
      LITERT_RETURN_IF_ERROR(embedding_lookup_manager_->LookupDecode(
          input_token_id, draft_embedding));
      latency_stats_.decode_embedder_inference_latency_us +=
          absl::ToInt64Microseconds(absl::Now() - start);
    } else {
      return absl::InternalError(
          "EmbeddingLookupManager not available for MTP.");
    }

    {
      // Lock the drafter activations buffer for writing.
      LITERT_ASSIGN_OR_RETURN(
          auto drafter_activations_lock_and_addr,
          ::litert::TensorBufferScopedLock::Create(
              ctx.mtp_input_buffers[MtpSignatures::kInputActivations],
              ::litert::TensorBuffer::LockMode::kWrite));

      uint8_t* base_ptr =
          static_cast<uint8_t*>(drafter_activations_lock_and_addr.second);

      // Copy embedding FIRST.
      memcpy(base_ptr, draft_embedding.data(),
             draft_embedding.size() * sizeof(float));
      // Copy activations SECOND.
      memcpy(base_ptr + draft_embedding.size() * sizeof(float),
             current_activations.data(), current_activations.size());
    }
    auto start = absl::Now();
    LITERT_RETURN_IF_ERROR(ctx.mtp_compiled_model.Run(
        MtpSignatures::kMtpDrafter, ctx.mtp_input_buffers,
        ctx.mtp_output_buffers));
    auto end = absl::Now();
    latency_stats_.decode_drafter_inference_latency_us +=
        absl::ToInt64Microseconds(end - start);

    start = absl::Now();
    LITERT_ASSIGN_OR_RETURN(
        int draft_id, ApplyGreedySampling(
                          ctx.mtp_output_buffers[MtpSignatures::kOutputLogits],
                          npu_config_.enable_neon_for_npu_greedy_sampling));
    end = absl::Now();
    latency_stats_.decode_sampling_latency_us +=
        absl::ToInt64Microseconds(end - start);
    draft_tokens.push_back(draft_id);

    NPU_EXECUTOR_LOG(INFO) << "    [Drafter] Step " << i << " at pos "
                           << start_step << ": Generated Token ID " << draft_id;
    if (i < drafter_sequence_length - 1) {
      LITERT_ASSIGN_OR_RETURN(
          current_activations,
          CopyRawBytesFromTensorBuffer(
              ctx.mtp_output_buffers[MtpSignatures::kOutputActivations]));
      input_token_id = draft_id;
    }
  }

  return draft_tokens;
}

namespace {
// Helper to sample from a slice of logits pointer directly without
// acquiring/releasing locks.
absl::StatusOr<int> SampleLogitsSliceFromLockedPtr(
    ::litert::ElementType element_type, const uint8_t* base_ptr, int batch_idx,
    int vocab_size, size_t element_size, bool enable_neon_sampling) {
  const uint8_t* logits_ptr =
      base_ptr + (batch_idx * vocab_size * element_size);

  auto find_max_index_plain = [&](auto ptr) {
    int max_idx = 0;
    auto max_val = ptr[0];
    for (int i = 1; i < vocab_size; ++i) {
      if (ptr[i] > max_val) {
        max_val = ptr[i];
        max_idx = i;
      }
    }
    return max_idx;
  };

  if (element_type == ::litert::ElementType::Float32) {
#if defined(__ANDROID__) && defined(__ARM_NEON)
    if (enable_neon_sampling) {
      return FindMaxIndexFloatNeon(reinterpret_cast<const float*>(logits_ptr),
                                   vocab_size);
    }
#endif
#if defined(__x86_64__) || defined(_M_X64)
    if (enable_neon_sampling) {
      return FindMaxIndexSse2Float(reinterpret_cast<const float*>(logits_ptr),
                                   vocab_size);
    }
#endif
    return find_max_index_plain(reinterpret_cast<const float*>(logits_ptr));
  } else if (element_type == ::litert::ElementType::Int16) {
#if defined(__ANDROID__) && defined(__ARM_NEON)
    if (enable_neon_sampling) {
      return FindMaxIndexInt16Neon(reinterpret_cast<const int16_t*>(logits_ptr),
                                   vocab_size);
    }
#endif
#if defined(__x86_64__) || defined(_M_X64)
    if (enable_neon_sampling) {
      return FindMaxIndexSse2Int16(reinterpret_cast<const int16_t*>(logits_ptr),
                                   vocab_size);
    }
#endif
    return find_max_index_plain(reinterpret_cast<const int16_t*>(logits_ptr));
  } else if (element_type == ::litert::ElementType::Int8) {
#if defined(__ANDROID__) && defined(__ARM_NEON)
    if (enable_neon_sampling) {
      return FindMaxIndexInt8Neon(reinterpret_cast<const int8_t*>(logits_ptr),
                                  vocab_size);
    }
#endif
#if defined(__x86_64__) || defined(_M_X64)
    if (enable_neon_sampling) {
      return FindMaxIndexSse2Int8(reinterpret_cast<const int8_t*>(logits_ptr),
                                  vocab_size);
    }
#endif
    return find_max_index_plain(reinterpret_cast<const int8_t*>(logits_ptr));
  }

  return absl::UnimplementedError("Unsupported logit type for batch sampling.");
}
}  // namespace

absl::Status LlmLiteRtNpuCompiledModelExecutor::RunVerifierBatch(
    int start_step, int current_token_id,
    const std::vector<int>& draft_tokens) {
  std::vector<int32_t> verify_ids;
  verify_ids.push_back(current_token_id);
  for (int id : draft_tokens) {
    verify_ids.push_back(id);
  }

  auto& embedder_verify_inputs =
      embedder_context_.inference_context.verify_input_buffers;

  {
    LITERT_ASSIGN_OR_RETURN(
        auto input_lock,
        ::litert::TensorBufferScopedLock::Create(
            embedder_verify_inputs[EmbedderSignatures::kEmbedderInput],
            ::litert::TensorBuffer::LockMode::kWrite));
    auto* input_ptr = static_cast<int32_t*>(input_lock.second);
    for (int i = 0; i < verify_ids.size(); ++i) {
      input_ptr[i] = verify_ids[i];
    }
  }

  if (UseEmbeddingLookupManager()) {
    litert::TensorBuffer& verify_embedding_buffer =
        embedder_context_.inference_context
            .verify_output_buffers[EmbedderSignatures::kEmbedderOutput];
    auto start = absl::Now();
    LITERT_RETURN_IF_ERROR(embedding_lookup_manager_->LookupPrefill(
        verify_ids, &verify_embedding_buffer, 0));
    auto end = absl::Now();
    latency_stats_.decode_embedder_inference_latency_us +=
        absl::ToInt64Microseconds(end - start);
  } else {
    return absl::InternalError("EmbeddingLookupManager not available for MTP.");
  }

  if (use_hw_ple_for_npu_ && !ple_table_ptrs_.empty()) {
    auto start = absl::Now();
    auto& ple_output_buffer =
        llm_inference_context_.verify_input_buffers[kPerLayerEmbedderTensor];
    LITERT_ASSIGN_OR_RETURN(
        auto lock,
        ::litert::TensorBufferScopedLock::Create(
            ple_output_buffer, ::litert::TensorBuffer::LockMode::kWrite));
    void* output_ptr = lock.second;

    ABSL_RETURN_IF_ERROR(HWPerLayerEmbeddingLookup(
        verify_ids.data(), verify_ids.size(), ple_table_ptrs_.data(),
        ple_quant_params_.data(), num_tables_, ple_embedding_dim_, output_ptr,
        output_type_, ple_table_element_type_, mul_scale_, output_scale_,
        final_zero_point_));

    latency_stats_.decode_embedder_per_layer_inference_latency_us +=
        absl::ToInt64Microseconds(absl::Now() - start);
  } else if (embedder_per_layer_context_.has_value()) {
    {
      LITERT_ASSIGN_OR_RETURN(
          auto verify_ple_input_lock,
          ::litert::TensorBufferScopedLock::Create(
              embedder_per_layer_context_->inference_context
                  .verify_input_buffers
                      [EmbedderPerLayerSignatures::kEmbedderInput],
              ::litert::TensorBuffer::LockMode::kWrite));
      auto* input_ptr = static_cast<int32_t*>(verify_ple_input_lock.second);
      for (int i = 0; i < verify_ids.size(); ++i) {
        input_ptr[i] = verify_ids[i];
      }
    }
    auto start = absl::Now();
    auto res =
        embedder_per_layer_context_->embedder_per_layer_compiled_model.Run(
            EmbedderPerLayerSignatures::kVerifyEmbedderPerLayer,
            embedder_per_layer_context_->inference_context.verify_input_buffers,
            embedder_per_layer_context_->inference_context
                .verify_output_buffers);
    auto end = absl::Now();
    latency_stats_.decode_embedder_per_layer_inference_latency_us +=
        absl::ToInt64Microseconds(end - start);
    RET_CHECK(res) << "Failed to run embedder per layer model."
                   << res.Error().Message();
  }

  {
    LITERT_ASSIGN_OR_RETURN(
        auto pos_lock,
        ::litert::TensorBufferScopedLock::Create(
            rope_context_.verify_input_buffers[RopeSignatures::kInputPos],
            ::litert::TensorBuffer::LockMode::kWrite));
    auto* pos_ptr = static_cast<int32_t*>(pos_lock.second);
    std::vector<int> verify_pos_ids;
    for (int i = 0; i < verify_ids.size(); ++i) {
      pos_ptr[i] = start_step + i;
      verify_pos_ids.push_back(start_step + i);
    }
    NPU_EXECUTOR_LOG(INFO) << "    [Verify] Input Token IDs: ["
                           << absl::StrJoin(verify_ids, ", ") << "]";
    NPU_EXECUTOR_LOG(INFO) << "    [Verify] Position IDs: ["
                           << absl::StrJoin(verify_pos_ids, ", ") << "]";

    LITERT_ASSIGN_OR_RETURN(
        auto mask_step_lock,
        ::litert::TensorBufferScopedLock::Create(
            mask_context_
                .verify_input_buffers[MaskSignatures::kMaskInputTimeStep],
            ::litert::TensorBuffer::LockMode::kWrite));
    static_cast<int32_t*>(mask_step_lock.second)[0] = start_step;

    LITERT_ASSIGN_OR_RETURN(
        auto mask_tokens_lock,
        ::litert::TensorBufferScopedLock::Create(
            mask_context_
                .verify_input_buffers[MaskSignatures::kMaskInputTokens],
            ::litert::TensorBuffer::LockMode::kWrite));
    auto* mask_tokens_ptr = static_cast<int32_t*>(mask_tokens_lock.second);
    for (int i = 0; i < verify_ids.size(); ++i) {
      mask_tokens_ptr[i] = verify_ids[i];
    }
  }

  LITERT_RETURN_IF_ERROR(
      npu_auxiliary_context_.npu_auxiliary_compiled_model.Run(
          RopeSignatures::kVerifyRope, rope_context_.verify_input_buffers,
          rope_context_.verify_output_buffers));
  if (verify_mask_update_method_ == MaskUpdateMethod::kWH) {
    LITERT_RETURN_IF_ERROR(HWMaskUpdate(mask_context_.verify_input_buffers,
                                        mask_context_.verify_output_buffers));
  } else {
    LITERT_RETURN_IF_ERROR(
        npu_auxiliary_context_.npu_auxiliary_compiled_model.Run(
            MaskSignatures::kVerifyMask, mask_context_.verify_input_buffers,
            mask_context_.verify_output_buffers));
  }

  LITERT_RETURN_IF_ERROR(llm_compiled_model_.Run(
      LlmSignatures::kVerifyLlm, llm_inference_context_.verify_input_buffers,
      llm_inference_context_.verify_output_buffers));

  return absl::OkStatus();
}

absl::StatusOr<LlmLiteRtNpuCompiledModelExecutor::RejectionSamplingResult>
LlmLiteRtNpuCompiledModelExecutor::PerformRejectionSampling(
    const std::vector<int>& draft_tokens,
    const ::litert::TensorBuffer& verifier_logits_buffer) {
  int num_accepted = 0;
  int bonus_token_id = kInvalidTokenId;

  LITERT_ASSIGN_OR_RETURN(RankedTensorType tensor_type,
                          verifier_logits_buffer.TensorType());
  if (tensor_type.Layout().Dimensions().size() < 3) {
    return absl::InvalidArgumentError(
        "Logits tensor must have at least 3 dimensions.");
  }
  const int vocab_size = tensor_type.Layout().Dimensions()[2];
  if (vocab_size == 0) {
    return absl::InvalidArgumentError("Vocab size cannot be 0.");
  }
  size_t element_size = 0;
  switch (tensor_type.ElementType()) {
    case ::litert::ElementType::Float32:
      element_size = sizeof(float);
      break;
    case ::litert::ElementType::Int16:
      element_size = sizeof(int16_t);
      break;
    case ::litert::ElementType::Int8:
      element_size = sizeof(int8_t);
      break;
    default:
      return absl::UnimplementedError(
          "Unsupported logit type for element size.");
  }

  LITERT_ASSIGN_OR_RETURN(
      auto lock_and_addr,
      ::litert::TensorBufferScopedLock::Create(
          verifier_logits_buffer, ::litert::TensorBuffer::LockMode::kRead));
  const uint8_t* base_ptr = static_cast<const uint8_t*>(lock_and_addr.second);

  // Log all sampled tokens from the verifier for transparency.
  std::vector<int> all_verifier_sampled;
  all_verifier_sampled.reserve(draft_tokens.size() + 1);
  for (int i = 0; i < draft_tokens.size() + 1; ++i) {
    LITERT_ASSIGN_OR_RETURN(
        int sampled_token,
        SampleLogitsSliceFromLockedPtr(
            tensor_type.ElementType(), base_ptr, i, vocab_size, element_size,
            npu_config_.enable_neon_for_npu_greedy_sampling));
    all_verifier_sampled.push_back(sampled_token);
  }
  NPU_EXECUTOR_LOG(INFO) << "    [RS] Verifier Sampled Tokens: ["
                         << absl::StrJoin(all_verifier_sampled, ", ") << "]";

  for (int i = 0; i < draft_tokens.size(); ++i) {
    int sampled_verifier_token = all_verifier_sampled[i];

    NPU_EXECUTOR_LOG(INFO) << "    [RS] Step " << i << ": Drafter Token "
                           << draft_tokens[i] << " vs Verifier Sampled Token "
                           << sampled_verifier_token;

    if (sampled_verifier_token == draft_tokens[i]) {
      num_accepted++;
    } else {
      bonus_token_id = sampled_verifier_token;
      break;
    }
  }

  if (num_accepted == draft_tokens.size()) {
    bonus_token_id = all_verifier_sampled[num_accepted];
  }
  latency_stats_.mtp_num_draft_tokens += draft_tokens.size();
  latency_stats_.mtp_num_accepted_tokens += num_accepted;

  return RejectionSamplingResult{num_accepted, bonus_token_id};
}

absl::Status LlmLiteRtNpuCompiledModelExecutor::CommitVerifiedKVCache(
    int start_step) {
  {
    LITERT_ASSIGN_OR_RETURN(
        auto pos_lock, ::litert::TensorBufferScopedLock::Create(
                           cache_update_inference_context_
                               .verify_input_buffers[RopeSignatures::kInputPos],
                           ::litert::TensorBuffer::LockMode::kWrite));
    auto* pos_ptr = static_cast<int32_t*>(pos_lock.second);
    // Use the size of the pos tensor to determine the number of tokens to
    // commit. Input pos is 1D tensor, where dimension 0 represents the number
    // of steps that draft model processed.
    LITERT_ASSIGN_OR_RETURN(RankedTensorType tensor_type,
                            cache_update_inference_context_
                                .verify_input_buffers[RopeSignatures::kInputPos]
                                .TensorType());
    int tensor_size = tensor_type.Layout().Dimensions()[0];
    for (int i = 0; i < tensor_size; ++i) {
      pos_ptr[i] = start_step + i;
    }
  }
  if (prefill_kv_cache_update_method_ == KVCacheUpdateMethod::kWH) {
    ABSL_RETURN_IF_ERROR(
        HWKVCacheUpdate(cache_update_inference_context_.verify_input_buffers,
                        cache_update_inference_context_.verify_output_buffers,
                        kv_quant_params_, has_sliding_window_attention_));
  } else {
    LITERT_RETURN_IF_ERROR(
        npu_auxiliary_context_.npu_auxiliary_compiled_model.Run(
            CacheUpdateSignatures::kVerifyCacheUpdate,
            cache_update_inference_context_.verify_input_buffers,
            cache_update_inference_context_.verify_output_buffers));
  }
  return absl::OkStatus();
}

absl::Status LlmLiteRtNpuCompiledModelExecutor::SetCurrentStep(int new_step) {
  const int max_step = processed_tokens_.TokenCount();
  if (new_step != (max_step - 1)) {
    return absl::InvalidArgumentError(
        "NPU executor's SetCurrentStep only supports rolling back one token at "
        "the end of decode.");
  }

  for (int i = new_step; i < current_step_; ++i) {
    if (processed_tokens_.GetTokenAtStep(i).empty()) {
      return absl::InvalidArgumentError(
          "SetCurrentStep does not currently support rolling back vision or "
          "audio tokens.");
    }
  }

  current_step_ = new_step;
  return absl::OkStatus();
};

absl::StatusOr<const ProcessedTokens*>
LlmLiteRtNpuCompiledModelExecutor::GetProcessedTokens() const {
  return &processed_tokens_;
}

absl::StatusOr<int> LlmLiteRtNpuCompiledModelExecutor::GetVocabSize() {
  LITERT_ASSIGN_OR_RETURN(
      auto logits_tensor_type,
      llm_inference_context_
          .decode_output_buffers[LlmSignatures::kDecodeLogitsOutput]
          .TensorType());
  const auto rank = logits_tensor_type.Layout().Dimensions().size();
  RET_CHECK(rank == 2 || rank == 3) << "Logits must be a 2D or 3D tensor.";
  return logits_tensor_type.Layout().Dimensions()[rank - 1];
}

const LlmLiteRtNpuCompiledModelExecutor::LatencyStats&
LlmLiteRtNpuCompiledModelExecutor::GetLatencyStats() const {
  return latency_stats_;
}

absl::Status LlmLiteRtNpuCompiledModelExecutor::Reset() {
  NPU_EXECUTOR_LOG(INFO) << "Custom NPU execution latency stats:\n"
                         << latency_stats_;
  current_step_ = 0;
  ran_decode_ = false;
  ABSL_RETURN_IF_ERROR(processed_tokens_.RollBackToStep(0));
  sampled_ids_.clear();
  latency_stats_ = {};
  last_verify_activations_.clear();
  pending_accepted_tokens_.clear();

  ABSL_RETURN_IF_ERROR(
      ClearKVCache(llm_inference_context_.prefill_input_buffers));
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<LlmContext>>
LlmLiteRtNpuCompiledModelExecutor::CreateNewContext(
    std::optional<uint32_t> lora_id, RuntimeConfig runtime_config) const {
  std::unique_ptr<ProcessedContext> processed_context =
      std::make_unique<LlmProcessedContext>(
          lora_id,
          std::make_unique<LegacyMapState>(
              absl::flat_hash_map<std::string, ::litert::TensorBuffer>()));

  return std::make_unique<LlmContext>(
      std::move(processed_context),
      std::make_unique<RuntimeConfig>(std::move(runtime_config)),
      std::make_unique<RuntimeState>());
}

absl::StatusOr<std::unique_ptr<LlmContext>>
LlmLiteRtNpuCompiledModelExecutor::CloneContext() const {
  absl::flat_hash_map<std::string, ::litert::TensorBuffer> kv_cache_buffers;
  for (const auto& [name, buffer] :
       llm_inference_context_.prefill_input_buffers) {
    if (absl::StartsWith(name, kv_cache_k_root_name) ||
        absl::StartsWith(name, kv_cache_v_root_name) ||
        absl::StartsWith(name, kv_cache_c_root_name)) {
      LITERT_ASSIGN_OR_RETURN(auto buffer_copy, CopyTensorBuffer(env_, buffer));
      kv_cache_buffers[name] = std::move(buffer_copy);
    }
  }

  std::unique_ptr<ProcessedContext> processed_context =
      std::make_unique<LlmProcessedContext>(
          /*lora_id=*/std::nullopt,
          std::make_unique<LegacyMapState>(std::move(kv_cache_buffers)),
          processed_tokens_);

  RuntimeConfig runtime_config;
  runtime_config.sampler_params = sampler_params_;

  RuntimeState runtime_state;
  runtime_state.current_step = current_step_;

  return std::make_unique<LlmContext>(
      std::move(processed_context),
      std::make_unique<RuntimeConfig>(std::move(runtime_config)),
      std::make_unique<RuntimeState>(std::move(runtime_state)));
}

absl::Status LlmLiteRtNpuCompiledModelExecutor::RestoreContext(
    std::unique_ptr<LlmContext> context_data) {
  if (context_data->runtime_state().current_step > 0) {
    auto& processed_ctx =
        static_cast<LlmProcessedContext&>(context_data->processed_context());
    auto* map_state =
        dynamic_cast<LegacyMapState*>(processed_ctx.state().get());
    RET_CHECK(map_state != nullptr)
        << "Expected LegacyMapState in RestoreContext";
    const auto& saved_kv_buffers = map_state->buffers();
    for (const auto& [name, saved_buffer] : saved_kv_buffers) {
      if (llm_inference_context_.prefill_input_buffers.contains(name)) {
        auto& target_buffer =
            llm_inference_context_.prefill_input_buffers[name];

        LITERT_ASSIGN_OR_RETURN(
            auto src_lock_and_addr,
            ::litert::TensorBufferScopedLock::Create(
                saved_buffer, ::litert::TensorBuffer::LockMode::kRead));

        LITERT_ASSIGN_OR_RETURN(
            auto dst_lock_and_addr,
            ::litert::TensorBufferScopedLock::Create(
                target_buffer, ::litert::TensorBuffer::LockMode::kWrite));

        LITERT_ASSIGN_OR_RETURN(size_t src_size, saved_buffer.PackedSize());
        LITERT_ASSIGN_OR_RETURN(size_t dst_size, target_buffer.PackedSize());
        if (src_size != dst_size) {
          return absl::InternalError("Buffer size mismatch in RestoreContext");
        }

        std::memcpy(dst_lock_and_addr.second, src_lock_and_addr.second,
                    src_size);
      }
    }
  } else {
    ABSL_RETURN_IF_ERROR(
        ClearKVCache(llm_inference_context_.prefill_input_buffers));
  }

  processed_tokens_ = context_data->processed_context().processed_tokens();
  current_step_ = context_data->runtime_state().current_step;

  if (context_data->runtime_config().sampler_params.has_value()) {
    sampler_params_ = *context_data->runtime_config().sampler_params;
  }

  return absl::OkStatus();
}

absl::StatusOr<int>
LlmLiteRtNpuCompiledModelExecutor::DetermineMaxSequenceLength(
    const LlmExecutorSettings& executor_settings, ModelResources& resources,
    const litert::Model& llm_model) {
  int ans = 0;

  // (1) Check for the presence of the max_num_tokens in the LlmMetadata
  if (auto metadata_status = resources.GetLlmMetadata(); metadata_status.ok()) {
    const proto::LlmMetadata* metadata = *metadata_status;
    if (metadata && metadata->max_num_tokens() > 0) {
      ans = metadata->max_num_tokens();
    }
  }

  // (2) If not present fall back to iterate through all KV cache input buffers
  // of the llm_model and get the maximum number.
  if (ans <= 0) {
    // We need to go through all KV cache layers because in sliding window
    // attention models various layers will have a smaller (ringbuffer) cache,
    // and instead we need to find the true global KV cache.
    // Once the "Executor metadata" design is implemented the information can
    // instead be taken from there.
    LITERT_ASSIGN_OR_RETURN(const std::vector<int> prefill_sizes,
                            DetectPrefillSizes(llm_model));
    if (prefill_sizes.empty()) {
      return absl::InternalError("No prefill sizes detected.");
    }
    const int prefill_size = prefill_sizes.front();
    bool found_any = false;
    auto signatures = llm_model.GetSignatures();
    if (signatures) {
      for (const auto& signature : *signatures) {
        absl::string_view key = signature.Key();
        std::string target_prefix =
            absl::StrCat(kPrefillSignatureBase, "_", prefill_size);
        if (key == target_prefix ||
            absl::StartsWith(key, absl::StrCat(target_prefix, "_"))) {
          found_any = true;
          for (auto input_name : signature.InputNames()) {
            if (IsKVCacheTensor(input_name)) {
              LITERT_ASSIGN_OR_RETURN(const litert::SimpleTensor& tensor,
                                      signature.InputTensor(input_name));
              LITERT_ASSIGN_OR_RETURN(::litert::RankedTensorType type,
                                      tensor.RankedTensorType());
              for (auto dim : type.Layout().Dimensions()) {
                ans = std::max(ans, dim);
              }
            }
          }
        }
      }
    }
    if (!found_any) {
      return absl::NotFoundError(absl::StrCat(
          "Could not find any prefill signature for size ", prefill_size));
    }
  }

  // (3) Check the passed in executor setting max num tokens field.
  int settings_max_num_tokens = executor_settings.GetMaxNumTokens();
  if (settings_max_num_tokens > 0) {
    if (ans > 0) {
      if (settings_max_num_tokens > ans) {
        ABSL_LOG(WARNING) << "Passed in max_num_tokens ("
                          << settings_max_num_tokens
                          << ") is larger than what the model supports (" << ans
                          << "). Using model limit.";
      } else {
        ans = settings_max_num_tokens;
      }
    } else {
      ans = settings_max_num_tokens;
    }
  }

  if (ans <= 0) {
    return absl::InternalError("Failed to determine max sequence length.");
  }

  return ans;
}

// static
absl::StatusOr<std::unique_ptr<LlmLiteRtNpuCompiledModelExecutor>>
LlmLiteRtNpuCompiledModelExecutor::Create(
    const LlmExecutorSettings& executor_settings, ModelResources& resources,
    Environment& env) {
  bool enable_npu_debug_logging = false;
  auto npu_config_status = executor_settings.GetBackendConfig<NpuConfig>();
  if (npu_config_status.ok()) {
    enable_npu_debug_logging = npu_config_status->enable_npu_debug_logging;
  }

  LITERT_ASSIGN_OR_RETURN(
      const litert::Model* llm_model,
      resources.GetTFLiteModel(ModelType::kTfLitePrefillDecode));

  DumpModelSignatures("PrefillDecode", *llm_model);

  LITERT_ASSIGN_OR_RETURN(
      int max_sequence_length,
      DetermineMaxSequenceLength(executor_settings, resources, *llm_model));

  // `DetermineMaxSequenceLength` resolves the effective limit by taking the
  // minimum of the user-requested limit (if > 0) and the model-supported limit.
  // If they differ (e.g. user limit is 0 or larger than what the model
  // supports), we override the settings. If the user requested a smaller limit,
  // it is respected and not overridden.
  LlmExecutorSettings mutable_settings = executor_settings;
  if (mutable_settings.GetMaxNumTokens() != max_sequence_length) {
    ABSL_LOG(WARNING) << "Overriding executor settings max_num_tokens ("
                      << mutable_settings.GetMaxNumTokens()
                      << ") with NPU model max sequence length: ("
                      << max_sequence_length << ")";
    mutable_settings.SetMaxNumTokens(max_sequence_length);
  }

  // Detect the prefill lengths the model was compiled with (e.g. 128 or 256)
  // and resolve all prefill-family signature names from it.
  LITERT_ASSIGN_OR_RETURN(std::vector<int> sorted_supported_prefill_sizes,
                          DetectPrefillSizes(*llm_model));
  LITERT_ASSIGN_OR_RETURN(
      std::vector<int> sorted_supported_context_sizes,
      DetectSupportedContextSizes(*llm_model, sorted_supported_prefill_sizes[0],
                                  max_sequence_length));

  const bool has_suffix = DetectHasSuffix(*llm_model);
  const bool has_rope_suffix = DetectHasRopeSuffix(resources);

  std::vector<std::vector<ResolvedSignatures>> signatures_list_of_list;
  signatures_list_of_list.reserve(sorted_supported_context_sizes.size());
  for (int context_size : sorted_supported_context_sizes) {
    std::vector<ResolvedSignatures> context_signatures;
    context_signatures.reserve(sorted_supported_prefill_sizes.size());
    for (int prefill_size : sorted_supported_prefill_sizes) {
      context_signatures.push_back(BuildResolvedSignatures(
          prefill_size, context_size, has_suffix, has_rope_suffix));
    }
    signatures_list_of_list.push_back(std::move(context_signatures));
  }

  const auto& largest_signatures = signatures_list_of_list.back().back();
  ABSL_LOG(INFO) << "Detected NPU prefill sizes: "
                 << absl::StrJoin(sorted_supported_prefill_sizes, ", ")
                 << " (largest signature \"" << largest_signatures.prefill
                 << "\").";

  // Initialize logits quantization parameters using the 'decode' signature.
  LogitsQuantizationParams quantization_params = {.scale = 1.0f,
                                                  .zero_point = 0};
  const auto& representative_signatures =
      signatures_list_of_list.front().front();
  LITERT_ASSIGN_OR_RETURN(
      auto decode_signature,
      llm_model->FindSignature(representative_signatures.decode));
  LITERT_ASSIGN_OR_RETURN(
      auto logits_tensor,
      decode_signature.OutputTensor(LlmSignatures::kDecodeLogitsOutput));
  if (logits_tensor.HasQuantization()) {
    auto q_params = logits_tensor.PerTensorQuantization();
    quantization_params.scale = q_params.scale;
    quantization_params.zero_point = static_cast<int32_t>(q_params.zero_point);
    ABSL_LOG_IF(INFO, enable_npu_debug_logging)
        << "Logits quantization params from '"
        << representative_signatures.decode
        << "' signature: scale=" << quantization_params.scale
        << " zero_point=" << quantization_params.zero_point;
  } else {
    ABSL_LOG_IF(WARNING, enable_npu_debug_logging)
        << "No quantization for logits in '" << representative_signatures.decode
        << "' signature (using default scale= " << quantization_params.scale
        << ", zero_point= " << quantization_params.zero_point << ").";
  }

  // For the lack of a better way to identify the model variants, we use the
  // presence of per-layer embeddings as the signal for Gemma3n.
  LITERT_ASSIGN_OR_RETURN(
      const bool has_per_layer_embeddings,
      HasPerLayerEmbedder(*llm_model, largest_signatures.prefill));
  if (has_per_layer_embeddings) {
    return CreateForModelHasPerLayerEmbedding(
        mutable_settings, resources, env, llm_model, quantization_params,
        signatures_list_of_list, sorted_supported_context_sizes);
  } else {
    return CreateForModelWithoutPerLayerEmbedding(
        mutable_settings, resources, env, llm_model, quantization_params,
        signatures_list_of_list, sorted_supported_context_sizes);
  }
};

absl::StatusOr<std::unique_ptr<LlmLiteRtNpuCompiledModelExecutor>>
LlmLiteRtNpuCompiledModelExecutor::CreateForModelHasPerLayerEmbedding(
    const LlmExecutorSettings& executor_settings, ModelResources& resources,
    litert::Environment& env, const litert::Model* transformer_model,
    LogitsQuantizationParams quantization_params,
    const std::vector<std::vector<ResolvedSignatures>>& signatures_list_of_list,
    const std::vector<int>& sorted_supported_context_sizes) {
  const auto& prefill_signatures_list = signatures_list_of_list.back();
  const int largest_context_size = sorted_supported_context_sizes.back();
  int64_t kv_cache_init_value = GetKvCacheInitValue(resources);
  // If the model is fully AOT compiled for NPU, NPU accelerator is used
  // automatically.
  LITERT_ASSIGN_OR_RETURN(auto options,
                          CreateLiteRtNpuOptions(executor_settings));
  LITERT_ASSIGN_OR_RETURN(
      CompiledModel llm_compiled_model,
      CompiledModel::Create(env, transformer_model->Get(), options));

  LITERT_ASSIGN_OR_RETURN(auto npu_auxiliary_lrt_model,
                          resources.GetTFLiteModel(ModelType::kTfLiteAux));
  LITERT_ASSIGN_OR_RETURN(
      auto npu_auxiliary_context,
      CreateNpuAuxiliaryContext(env, *npu_auxiliary_lrt_model,
                                executor_settings));

  LITERT_ASSIGN_OR_RETURN(auto embedder_lrt_model,
                          resources.GetTFLiteModel(ModelType::kTfLiteEmbedder));
  LITERT_ASSIGN_OR_RETURN(
      const litert::Model* embedder_per_layer_model,
      resources.GetTFLiteModel(ModelType::kTfLitePerLayerEmbedder));

  DumpModelSignatures("Aux", *npu_auxiliary_lrt_model);
  if (embedder_lrt_model) {
    DumpModelSignatures("Embedder", *embedder_lrt_model);
  }
  if (embedder_per_layer_model) {
    DumpModelSignatures("EmbedderPerLayer", *embedder_per_layer_model);
  }

  bool use_hw_ple_for_npu = false;
  auto npu_config_status = executor_settings.GetBackendConfig<NpuConfig>();
  if (npu_config_status.ok()) {
    use_hw_ple_for_npu = npu_config_status->use_hw_ple_for_npu;
  }

  absl::flat_hash_map<absl::string_view, HWQuantParams> kv_quant_params;

  auto create_context_group =
      [&](const std::vector<ResolvedSignatures>& signatures_list,
          const ContextGroup* largest_group) -> absl::StatusOr<ContextGroup> {
    ContextGroup group;
    group.representative_signatures = signatures_list.back();
    for (const auto& sigs : signatures_list) {
      group.prefill_signatures[sigs.size] = sigs;
    }

    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
        input_kv_cache_buffers;

    if (largest_group != nullptr) {
      // Share KV cache buffers from the largest group's decode input buffers.
      // Decode buffers contain all KV cache tensors.
      for (const auto& [name, buffer] :
           largest_group->llm_inference_context.decode_input_buffers) {
        if (name.starts_with(kv_cache_k_root_name) ||
            name.starts_with(kv_cache_v_root_name) ||
            name.starts_with(kv_cache_c_root_name)) {
          // Verify that the tensor exists in the compiled model before trying
          // to alias it.
          auto sig_expected = transformer_model->FindSignature(
              group.representative_signatures.decode);
          if (!sig_expected.HasValue()) {
            return absl::NotFoundError(
                absl::StrCat("Signature not found: ",
                             group.representative_signatures.decode));
          }
          auto tensor_expected = sig_expected->InputTensor(name);
          if (!tensor_expected.HasValue()) {
            return absl::NotFoundError(
                absl::StrCat("Input tensor not found: ", name));
          }
          const auto& tensor = tensor_expected.Value();
          LITERT_ASSIGN_OR_RETURN(auto target_type, tensor.RankedTensorType());

          LITERT_ASSIGN_OR_RETURN(auto alias_buf,
                                  CreateAliasBuffer(env, buffer, target_type));
          input_kv_cache_buffers[name] = std::move(alias_buf);
        }
      }
    }

    absl::flat_hash_map<
        int, absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>>
        gemma_prefill_input_buffers_by_size;
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
        gemma_decode_input_buffers;
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
        gemma_verify_input_buffers;
    absl::flat_hash_map<
        int, absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>>
        prefill_output_kv_cache_slice_buffers_by_size;
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
        decode_output_kv_cache_slice_buffers;
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
        verify_output_kv_cache_slice_buffers;

    // Allocate all input and output buffers of the LLM model that are meant to
    // be used by the NPU chip first, so that we can later duplicate the buffers
    // into the output buffer maps of the embedder, mask, and rope signatures.
    LITERT_RETURN_IF_ERROR(AllocateTransformerBuffers(
        env, transformer_model, llm_compiled_model, signatures_list,
        gemma_prefill_input_buffers_by_size, gemma_decode_input_buffers,
        gemma_verify_input_buffers, input_kv_cache_buffers,
        prefill_output_kv_cache_slice_buffers_by_size,
        decode_output_kv_cache_slice_buffers,
        verify_output_kv_cache_slice_buffers, kv_quant_params,
        kv_cache_init_value));

    // Workaround for NPU model issues where some KV cache tensors (e.g. layer
    // 19 in Gemma3n) are not connected to any OPs in the prefill subgraph,
    // causing the runtime to allocate them as HostMemory. We dynamically detect
    // these and re-allocate them using the decode signature to ensure they are
    // on NPU.
    for (auto& [name, buf] : input_kv_cache_buffers) {
      if (IsKVCacheTensor(name)) {
        auto buffer_type = buf.BufferType();
        if (buffer_type.HasValue() &&
            buffer_type.Value() == ::litert::TensorBufferType::kHostMemory) {
          ABSL_LOG(WARNING) << "KV cache tensor '" << name
                            << "' was allocated as HostMemory in prefill. "
                               "Re-allocating using decode signature.";
          LITERT_ASSIGN_OR_RETURN(
              auto new_buf, llm_compiled_model.CreateInputBuffer(
                                group.representative_signatures.decode, name));
          LITERT_RETURN_IF_ERROR(
              FillKVCacheBuffer(new_buf, kv_cache_init_value));
          buf = std::move(new_buf);
        }
      }
    }

    LITERT_ASSIGN_OR_RETURN(
        group.llm_inference_context,
        CreateLlmInferenceContextWithBufferSharing(
            env, llm_compiled_model, signatures_list, input_kv_cache_buffers,
            prefill_output_kv_cache_slice_buffers_by_size,
            decode_output_kv_cache_slice_buffers,
            verify_output_kv_cache_slice_buffers,
            gemma_prefill_input_buffers_by_size, gemma_decode_input_buffers,
            gemma_verify_input_buffers));

    // Gemma3n / VLM / Tiny Gemma specific fixes:
    const auto& first_prefill_input_buffers =
        group.llm_inference_context.prefill_input_buffers_by_size.begin()
            ->second;
    if (first_prefill_input_buffers.contains(cache_k31)) {
      // For models with 32 layers. Do nothing.
    } else if (first_prefill_input_buffers.contains(cache_k25)) {
      // Gemma3 specific fix:
      LITERT_ASSIGN_OR_RETURN(
          auto buffer_k,
          llm_compiled_model.CreateInputBuffer(
              group.representative_signatures.decode, cache_k25));
      group.llm_inference_context.decode_input_buffers[cache_k25] =
          std::move(buffer_k);
      LITERT_ASSIGN_OR_RETURN(
          auto buffer_v,
          llm_compiled_model.CreateInputBuffer(
              group.representative_signatures.decode, cache_v25));
      group.llm_inference_context.decode_input_buffers[cache_v25] =
          std::move(buffer_v);
    } else if (first_prefill_input_buffers.contains(cache_k23)) {
      // Fast VLM model specific fix:
      LITERT_ASSIGN_OR_RETURN(
          auto buffer_k,
          llm_compiled_model.CreateInputBuffer(
              group.representative_signatures.decode, cache_k23));
      group.llm_inference_context.decode_input_buffers[cache_k23] =
          std::move(buffer_k);
      LITERT_ASSIGN_OR_RETURN(
          auto buffer_v,
          llm_compiled_model.CreateInputBuffer(
              group.representative_signatures.decode, cache_v23));
      group.llm_inference_context.decode_input_buffers[cache_v23] =
          std::move(buffer_v);
    } else if (first_prefill_input_buffers.contains(cache_k17)) {
      // Tiny Gemma 270M specific fix:
      LITERT_ASSIGN_OR_RETURN(
          auto buffer_k,
          llm_compiled_model.CreateInputBuffer(
              group.representative_signatures.decode, cache_k17));
      group.llm_inference_context.decode_input_buffers[cache_k17] =
          std::move(buffer_k);
      LITERT_ASSIGN_OR_RETURN(
          auto buffer_v,
          llm_compiled_model.CreateInputBuffer(
              group.representative_signatures.decode, cache_v17));
      group.llm_inference_context.decode_input_buffers[cache_v17] =
          std::move(buffer_v);
    }

    LITERT_ASSIGN_OR_RETURN(
        group.mask_context,
        CreateMaskContextWithBufferSharing(
            npu_auxiliary_context, signatures_list,
            gemma_prefill_input_buffers_by_size, gemma_decode_input_buffers,
            gemma_verify_input_buffers));

    LITERT_ASSIGN_OR_RETURN(
        group.rope_context,
        CreateRopeContextWithBufferSharing(
            npu_auxiliary_context, signatures_list,
            gemma_prefill_input_buffers_by_size, gemma_decode_input_buffers,
            gemma_verify_input_buffers));

    // Duplicate the rope's buffers that are used to store the prefill and
    // decode input position, because they will need to be passed to the
    // cache update inference context as well.
    absl::flat_hash_map<int, ::litert::TensorBuffer> prefill_input_pos_by_size;
    absl::flat_hash_map<int, ::litert::TensorBuffer> prefill_valid_mask_by_size;

    for (const auto& sigs : signatures_list) {
      int size = sigs.size;
      LITERT_ASSIGN_OR_RETURN(prefill_input_pos_by_size[size],
                              group.rope_context.GetPrefillInputBuffers(size)
                                  .at(RopeSignatures::kInputPos)
                                  .Duplicate());

      const auto& mask_prefill_buffers =
          group.mask_context.GetPrefillInputBuffers(size);
      auto prefill_valid_mask_it =
          mask_prefill_buffers.find(MaskSignatures::kMaskInputValidMask);
      if (prefill_valid_mask_it != mask_prefill_buffers.end()) {
        LITERT_ASSIGN_OR_RETURN(prefill_valid_mask_by_size[size],
                                prefill_valid_mask_it->second.Duplicate());
      }
    }

    auto verify_input_pos_it =
        group.rope_context.verify_input_buffers.find(RopeSignatures::kInputPos);
    ::litert::TensorBuffer verify_input_pos;
    if (verify_input_pos_it != group.rope_context.verify_input_buffers.end()) {
      LITERT_ASSIGN_OR_RETURN(verify_input_pos,
                              verify_input_pos_it->second.Duplicate());
    }

    LITERT_ASSIGN_OR_RETURN(
        ::litert::TensorBuffer decode_input_pos,
        group.rope_context.decode_input_buffers[RopeSignatures::kInputPos]
            .Duplicate());

    ::litert::TensorBuffer decode_valid_mask;
    auto decode_valid_mask_it = group.mask_context.decode_input_buffers.find(
        MaskSignatures::kMaskInputValidMask);
    if (decode_valid_mask_it != group.mask_context.decode_input_buffers.end()) {
      LITERT_ASSIGN_OR_RETURN(decode_valid_mask,
                              decode_valid_mask_it->second.Duplicate());
    }

    ::litert::TensorBuffer verify_valid_mask;
    auto verify_valid_mask_it = group.mask_context.verify_input_buffers.find(
        MaskSignatures::kMaskInputValidMask);
    if (verify_valid_mask_it != group.mask_context.verify_input_buffers.end()) {
      LITERT_ASSIGN_OR_RETURN(verify_valid_mask,
                              verify_valid_mask_it->second.Duplicate());
    }

    LITERT_ASSIGN_OR_RETURN(
        group.cache_update_inference_context,
        CreateCacheUpdateInferenceContextWithBufferSharing(
            input_kv_cache_buffers,
            prefill_output_kv_cache_slice_buffers_by_size,
            decode_output_kv_cache_slice_buffers,
            verify_output_kv_cache_slice_buffers, prefill_input_pos_by_size,
            std::move(decode_input_pos), std::move(verify_input_pos),
            prefill_valid_mask_by_size, std::move(decode_valid_mask),
            std::move(verify_valid_mask)));

    if (embedder_lrt_model) {
      LITERT_ASSIGN_OR_RETURN(
          group.embedder_context,
          CreateEmbedderContextWithBufferSharing(
              env, *embedder_lrt_model, signatures_list,
              gemma_prefill_input_buffers_by_size, gemma_decode_input_buffers,
              gemma_verify_input_buffers, executor_settings));
    }

    if (embedder_per_layer_model && !use_hw_ple_for_npu &&
        (largest_group == nullptr ||
         largest_group->embedder_per_layer_context.has_value())) {
      absl::flat_hash_map<int, ::litert::TensorBuffer>
          mask_prefill_input_tokens_by_size;
      for (const auto& sigs : signatures_list) {
        int size = sigs.size;
        LITERT_ASSIGN_OR_RETURN(mask_prefill_input_tokens_by_size[size],
                                group.mask_context.GetPrefillInputBuffers(size)
                                    .at(MaskSignatures::kMaskInputTokens)
                                    .Duplicate());
      }
      LITERT_ASSIGN_OR_RETURN(
          auto ple_ctx,
          CreateEmbedderPerLayerContextWithBufferSharing(
              env, *embedder_per_layer_model, mask_prefill_input_tokens_by_size,
              group.mask_context
                  .decode_input_buffers[MaskSignatures::kMaskInputTokens],
              group.mask_context
                  .verify_input_buffers[MaskSignatures::kMaskInputTokens],
              gemma_prefill_input_buffers_by_size, gemma_decode_input_buffers,
              gemma_verify_input_buffers, executor_settings));
      group.embedder_per_layer_context = std::move(ple_ctx);
    }

    for (const auto& sigs : signatures_list) {
      int size = sigs.size;
      ABSL_LOG(INFO)
          << "CreateContextGroup: inserting into prefill_signature_map: "
          << size << " -> " << sigs.prefill;
      group.prefill_signature_map[size] = sigs.prefill;
    }

    return std::move(group);
  };

  LITERT_ASSIGN_OR_RETURN(
      auto largest_group_created,
      create_context_group(prefill_signatures_list, /*largest_group=*/nullptr));

  absl::flat_hash_map<int, const Model*> end_of_multi_modal_embedding_models;
  auto add_multi_modal_end_model = [&](ModelType type, int token) {
    auto model_buffer = resources.GetTFLiteModelBuffer(type);
    if (model_buffer.ok() && !model_buffer->empty()) {
      auto model = resources.GetTFLiteModel(type);
      if (model.ok()) {
        end_of_multi_modal_embedding_models[token] = *model;
      }
    }
  };

  add_multi_modal_end_model(ModelType::kTfLiteEndOfAudio,
                            litert::lm::ExecutorAudioData::kEndToken);
  add_multi_modal_end_model(ModelType::kTfLiteEndOfVision,
                            litert::lm::ExecutorVisionData::kEndToken);

  LITERT_ASSIGN_OR_RETURN(
      std::unique_ptr<EmbeddingLookupManager> embedding_lookup_manager,
      EmbeddingLookupManager::Create(env, embedder_lrt_model,
                                     end_of_multi_modal_embedding_models, true,
                                     "decode_embedder"));

  std::vector<const uint8_t*> ple_table_ptrs;
  std::vector<HWQuantizationParams> ple_quant_params;
  std::vector<float> ple_per_tensor_scales;

  int table_count = 0;
  int ple_embedding_dim_val = 0;
  litert::ElementType output_type = litert::ElementType::None;
  litert::ElementType ple_table_element_type = litert::ElementType::None;
  float mul_scale = 1.0f;
  float output_scale = 1.0f;
  int32_t final_zero_point = 0;

  if (use_hw_ple_for_npu) {
    // We are bypassing the model based inference for the Per-Layer Embedding
    // (PLE) subgraph. Instead, we perform the embedding lookup manually in C++
    // on the CPU.
    //
    // To do this, we must parse the PLE subgraph to dynamically extract all
    // necessary parameters (table pointers, dimensions, quantization scales,
    // and mathematical scaling factors) so our C++ lookup code can mimic
    // the bypassed NPU operations perfectly.
    auto extended_model = ExtendedModel::CreateFromNonOwnedHandle(
        embedder_per_layer_model->Get());
    LITERT_ASSIGN_OR_RETURN(auto subgraph, extended_model.MainSubgraph());
    auto ops = subgraph.Ops();
    for (const auto& op : ops) {
      // =======================================================================
      // 1. Parse the EmbeddingLookup Op
      // =======================================================================
      // We need to extract the raw table pointers, the embedding dimension,
      // the quantization type (Int4 vs Int8), and the dequantization scales.
      if (op.Code() == kLiteRtOpCodeTflEmbeddingLookup) {
        LITERT_ASSIGN_OR_RETURN(auto table_tensor, op.Input(1));
        LITERT_ASSIGN_OR_RETURN(auto table_type_info,
                                table_tensor.RankedTensorType());
        auto table_dims = table_type_info.Layout().Dimensions();
        int col_size = table_dims[1];  // The embedding dimension (e.g., 8960)

        // Initialize table properties from the first table we encounter.
        if (table_count == 0) {
          ple_table_element_type = table_tensor.ElementType();
          ple_embedding_dim_val = col_size;
        } else {
          // Validate that all embedding tables in the model are consistent.
          RET_CHECK_EQ((int)ple_table_element_type,
                       (int)table_tensor.ElementType())
              << "All embedding tables must have the same element type";
          RET_CHECK_EQ(ple_embedding_dim_val, col_size)
              << "All embedding tables must have the same embedding dimension.";
        }

        // Extract the raw pointer to the table's weight data in memory.
        auto weights = table_tensor.Weights();
        ple_table_ptrs.push_back(weights.Bytes().data());

        // Extract quantization parameters (scales) for dequantization.
        HWQuantizationParams qp;
        qp.scales = nullptr;
        qp.is_per_channel = false;

        if (table_tensor.HasQuantization()) {
          auto q_type = table_tensor.QTypeId();
          if (q_type == kLiteRtQuantizationPerTensor) {
            // Single scale for the entire table.
            auto q_params = table_tensor.PerTensorQuantization();
            ple_per_tensor_scales.push_back(q_params.scale);
            qp.scales = &ple_per_tensor_scales.back();
          } else if (q_type == kLiteRtQuantizationPerChannel) {
            // Per-channel (per-row/per-token) quantization. Our model uses
            // this, meaning each token has its own dequantization scale.
            auto q_params = table_tensor.PerChannelQuantization();
            qp.scales = q_params.scales;
            qp.is_per_channel = true;
          }
        }
        ple_quant_params.push_back(qp);
        table_count++;
      }

      // =======================================================================
      // 2. Parse the Mul (Multiplication) Op
      // =======================================================================
      // Gemma models scale their embeddings by sqrt(d_model) before the
      // transformer. For this model (d_model = 256), the scaling factor is 16.0
      // (sqrt(256)).
      //
      // Since we bypassed the model based inference for the Mul op, we must
      // extract this constant 16.0 multiplier and apply it manually in our C++
      // lookup. Without this, our embeddings would be too small, causing the
      // model to diverge.
      if (op.Code() == kLiteRtOpCodeTflMul) {
        auto inputs = op.Inputs();
        for (const auto& input : inputs) {
          // Look for the constant input tensor containing the multiplier.
          if (input.HasWeights()) {
            auto type_info = input.RankedTensorType();
            if (type_info.HasValue() && type_info.Value().ElementType() ==
                                            litert::ElementType::Float32) {
              auto weights = input.Weights();
              const float* vals =
                  reinterpret_cast<const float*>(weights.Bytes().data());
              mul_scale = vals[0];  // Store the multiplier (e.g., 16.0)
            }
          }
        }
      }
    }

    auto outputs = subgraph.Outputs();
    RET_CHECK(!outputs.empty()) << "No outputs in subgraph";
    auto output_tensor = outputs[0];
    output_type = output_tensor.ElementType();

    // =======================================================================
    // 3. Handle Output Quantization (Only for Int16 models)
    // =======================================================================
    // If the model expects quantized Int16 outputs (for NPUs that don't support
    // float), we must extract the output tensor's quantization parameters and
    // use them to quantize our dequantized floats back to Int16 before writing
    // to NPU.
    if (output_type == litert::ElementType::Int16) {
      RET_CHECK(output_tensor.HasQuantization());
      auto q_params = output_tensor.PerTensorQuantization();
      output_scale = q_params.scale;
      final_zero_point = q_params.zero_point;
    }
  }

  SpeculativeDecodingType speculative_decoding_type =
      SpeculativeDecodingType::kNone;
  std::optional<DrafterContext> drafter_context = std::nullopt;
  std::optional<DrafterAuxContext> drafter_aux_context = std::nullopt;

  if (executor_settings.GetAdvancedSettings().has_value() &&
      executor_settings.GetAdvancedSettings()->enable_speculative_decoding) {
    auto mtp_drafter_model =
        resources.GetTFLiteModel(ModelType::kTfLiteMtpDrafter);
    auto mtp_aux_model = resources.GetTFLiteModel(ModelType::kTfLiteMtpAux);

    if (mtp_drafter_model.ok() && mtp_aux_model.ok()) {
      LITERT_ASSIGN_OR_RETURN(
          drafter_context,
          CreateDrafterInferenceContextWithBufferSharing(
              env, **mtp_drafter_model,
              largest_group_created.llm_inference_context.prefill_input_buffers,
              largest_group_created.llm_inference_context.decode_output_buffers
                  [LlmSignatures::kLastLayerActivationsOutput]));
      LITERT_ASSIGN_OR_RETURN(
          drafter_aux_context,
          CreateDrafterInferenceAuxContextWithBufferSharing(
              env, **mtp_aux_model, drafter_context->mtp_input_buffers));
      speculative_decoding_type = SpeculativeDecodingType::kMTP;

      ABSL_RETURN_IF_ERROR(WarmupDrafterInference(drafter_context.value(),
                                                  drafter_aux_context.value()));
    }
  }

  const bool has_sliding_window_attention = DetectIsSwa(
      largest_group_created.llm_inference_context.prefill_input_buffers_by_size
          .begin()
          ->second);

  ABSL_RETURN_IF_ERROR(WarmupInference(
      llm_compiled_model, largest_group_created.llm_inference_context,
      npu_auxiliary_context.npu_auxiliary_compiled_model,
      prefill_signatures_list, largest_group_created.rope_context,
      largest_group_created.mask_context,
      largest_group_created.cache_update_inference_context));

  if (drafter_context.has_value()) {
    largest_group_created.drafter_context = std::move(*drafter_context);
  }
  if (drafter_aux_context.has_value()) {
    largest_group_created.drafter_aux_context = std::move(*drafter_aux_context);
  }

  absl::flat_hash_map<int, ContextGroup> context_groups;

  for (int idx = 0; idx < sorted_supported_context_sizes.size() - 1; ++idx) {
    int c = sorted_supported_context_sizes[idx];
    LITERT_ASSIGN_OR_RETURN(auto group,
                            create_context_group(signatures_list_of_list[idx],
                                                 &largest_group_created));

    ABSL_RETURN_IF_ERROR(WarmupInference(
        llm_compiled_model, group.llm_inference_context,
        npu_auxiliary_context.npu_auxiliary_compiled_model,
        signatures_list_of_list[idx], group.rope_context, group.mask_context,
        group.cache_update_inference_context));

    context_groups.emplace(c, std::move(group));
  }

  context_groups.emplace(largest_context_size,
                         std::move(largest_group_created));

  auto executor = absl::WrapUnique(new LlmLiteRtNpuCompiledModelExecutor(
      executor_settings, env, EmbedderContext(),
      std::move(npu_auxiliary_context), std::move(llm_compiled_model),
      std::move(embedding_lookup_manager),
      /*per_layer_embedding_lookup_manager=*/nullptr, quantization_params,
      std::move(ple_table_ptrs), std::move(ple_quant_params),
      std::move(ple_per_tensor_scales), table_count, ple_embedding_dim_val,
      output_type, ple_table_element_type, mul_scale, output_scale,
      final_zero_point, std::move(kv_quant_params), kv_cache_init_value,
      embedder_per_layer_model, std::nullopt, std::move(context_groups),
      sorted_supported_context_sizes, speculative_decoding_type,
      has_sliding_window_attention));
  ABSL_RETURN_IF_ERROR(executor->SwitchContextSizeIfRequired(
      sorted_supported_context_sizes.front()));
  return executor;
}

absl::StatusOr<std::unique_ptr<LlmLiteRtNpuCompiledModelExecutor>>
LlmLiteRtNpuCompiledModelExecutor::CreateForModelWithoutPerLayerEmbedding(
    const LlmExecutorSettings& executor_settings, ModelResources& resources,
    litert::Environment& env, const litert::Model* transformer_model,
    LogitsQuantizationParams quantization_params,
    const std::vector<std::vector<ResolvedSignatures>>& signatures_list_of_list,
    const std::vector<int>& sorted_supported_context_sizes) {
  const auto& prefill_signatures_list = signatures_list_of_list.back();
  const int largest_context_size = sorted_supported_context_sizes.back();
  int64_t kv_cache_init_value = GetKvCacheInitValue(resources);
  // Set up LiteRt options.
  LITERT_ASSIGN_OR_RETURN(auto options,
                          CreateLiteRtNpuOptions(executor_settings));
  LITERT_ASSIGN_OR_RETURN(
      CompiledModel llm_compiled_model,
      CompiledModel::Create(env, transformer_model->Get(), options));

  LITERT_ASSIGN_OR_RETURN(auto npu_auxiliary_lrt_model,
                          resources.GetTFLiteModel(ModelType::kTfLiteAux));
  LITERT_ASSIGN_OR_RETURN(
      auto npu_auxiliary_context,
      CreateNpuAuxiliaryContext(env, *npu_auxiliary_lrt_model,
                                executor_settings));

  LITERT_ASSIGN_OR_RETURN(auto embedder_lrt_model,
                          resources.GetTFLiteModel(ModelType::kTfLiteEmbedder));

  DumpModelSignatures("Aux", *npu_auxiliary_lrt_model);
  if (embedder_lrt_model) {
    DumpModelSignatures("Embedder", *embedder_lrt_model);
  }

  absl::flat_hash_map<absl::string_view, HWQuantParams> kv_quant_params;

  auto create_context_group =
      [&](const std::vector<ResolvedSignatures>& signatures_list,
          const ContextGroup* largest_group) -> absl::StatusOr<ContextGroup> {
    ContextGroup group;
    group.representative_signatures = signatures_list.back();
    for (const auto& sigs : signatures_list) {
      group.prefill_signatures[sigs.size] = sigs;
    }

    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
        input_kv_cache_buffers;

    if (largest_group != nullptr) {
      // Share KV cache buffers from the largest group's decode input buffers.
      // Decode buffers contain all KV cache tensors.
      for (const auto& [name, buffer] :
           largest_group->llm_inference_context.decode_input_buffers) {
        if (name.starts_with(kv_cache_k_root_name) ||
            name.starts_with(kv_cache_v_root_name) ||
            name.starts_with(kv_cache_c_root_name)) {
          // Verify that the tensor exists in the compiled model before trying
          // to alias it.
          auto sig_expected = transformer_model->FindSignature(
              group.representative_signatures.decode);
          if (!sig_expected.HasValue()) {
            return absl::NotFoundError(
                absl::StrCat("Signature not found: ",
                             group.representative_signatures.decode));
          }
          auto tensor_expected = sig_expected->InputTensor(name);
          if (!tensor_expected.HasValue()) {
            return absl::NotFoundError(
                absl::StrCat("Input tensor not found: ", name));
          }
          const auto& tensor = tensor_expected.Value();
          LITERT_ASSIGN_OR_RETURN(auto target_type, tensor.RankedTensorType());

          LITERT_ASSIGN_OR_RETURN(auto alias_buf,
                                  CreateAliasBuffer(env, buffer, target_type));
          input_kv_cache_buffers[name] = std::move(alias_buf);
        }
      }
    }

    absl::flat_hash_map<
        int, absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>>
        gemma_prefill_input_buffers_by_size;
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
        gemma_decode_input_buffers;
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
        gemma_verify_input_buffers;
    absl::flat_hash_map<
        int, absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>>
        prefill_output_kv_cache_slice_buffers_by_size;
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
        decode_output_kv_cache_slice_buffers;
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
        verify_output_kv_cache_slice_buffers;

    // Allocate all input and output buffers of the LLM model that are meant to
    // be used by the NPU chip first, so that we can later duplicate the buffers
    // into the output buffer maps of the embedder, mask, and rope signatures.
    LITERT_RETURN_IF_ERROR(AllocateTransformerBuffers(
        env, transformer_model, llm_compiled_model, signatures_list,
        gemma_prefill_input_buffers_by_size, gemma_decode_input_buffers,
        gemma_verify_input_buffers, input_kv_cache_buffers,
        prefill_output_kv_cache_slice_buffers_by_size,
        decode_output_kv_cache_slice_buffers,
        verify_output_kv_cache_slice_buffers, kv_quant_params,
        kv_cache_init_value));

    // Workaround for NPU model issues where some KV cache tensors (e.g. layer
    // 19 in Gemma3n) are not connected to any OPs in the prefill subgraph,
    // causing the runtime to allocate them as HostMemory. We dynamically detect
    // these and re-allocate them using the decode signature to ensure they are
    // on NPU.
    for (auto& [name, buf] : input_kv_cache_buffers) {
      if (IsKVCacheTensor(name)) {
        auto buffer_type = buf.BufferType();
        if (buffer_type.HasValue() &&
            buffer_type.Value() == ::litert::TensorBufferType::kHostMemory) {
          ABSL_LOG(WARNING) << "KV cache tensor '" << name
                            << "' was allocated as HostMemory in prefill. "
                               "Re-allocating using decode signature.";
          LITERT_ASSIGN_OR_RETURN(
              auto new_buf, llm_compiled_model.CreateInputBuffer(
                                group.representative_signatures.decode, name));
          LITERT_RETURN_IF_ERROR(
              FillKVCacheBuffer(new_buf, kv_cache_init_value));
          buf = std::move(new_buf);
        }
      }
    }

    LITERT_ASSIGN_OR_RETURN(
        group.llm_inference_context,
        CreateLlmInferenceContextWithBufferSharing(
            env, llm_compiled_model, signatures_list, input_kv_cache_buffers,
            prefill_output_kv_cache_slice_buffers_by_size,
            decode_output_kv_cache_slice_buffers,
            verify_output_kv_cache_slice_buffers,
            gemma_prefill_input_buffers_by_size, gemma_decode_input_buffers,
            gemma_verify_input_buffers));

    // Gemma3 / VLM / Tiny Gemma specific fixes:
    const auto& first_prefill_input_buffers =
        group.llm_inference_context.prefill_input_buffers_by_size.begin()
            ->second;
    if (first_prefill_input_buffers.contains(cache_k31)) {
      // For models with 32 layers. Do nothing.
    } else if (first_prefill_input_buffers.contains(cache_k25)) {
      // Gemma3 specific fix:
      LITERT_ASSIGN_OR_RETURN(
          auto buffer_k,
          llm_compiled_model.CreateInputBuffer(
              group.representative_signatures.decode, cache_k25));
      group.llm_inference_context.decode_input_buffers[cache_k25] =
          std::move(buffer_k);
      LITERT_ASSIGN_OR_RETURN(
          auto buffer_v,
          llm_compiled_model.CreateInputBuffer(
              group.representative_signatures.decode, cache_v25));
      group.llm_inference_context.decode_input_buffers[cache_v25] =
          std::move(buffer_v);
    } else if (first_prefill_input_buffers.contains(cache_k23)) {
      // Fast VLM model specific fix:
      LITERT_ASSIGN_OR_RETURN(
          auto buffer_k,
          llm_compiled_model.CreateInputBuffer(
              group.representative_signatures.decode, cache_k23));
      group.llm_inference_context.decode_input_buffers[cache_k23] =
          std::move(buffer_k);
      LITERT_ASSIGN_OR_RETURN(
          auto buffer_v,
          llm_compiled_model.CreateInputBuffer(
              group.representative_signatures.decode, cache_v23));
      group.llm_inference_context.decode_input_buffers[cache_v23] =
          std::move(buffer_v);
    } else if (first_prefill_input_buffers.contains(cache_k17)) {
      // Tiny Gemma 270M specific fix:
      LITERT_ASSIGN_OR_RETURN(
          auto buffer_k,
          llm_compiled_model.CreateInputBuffer(
              group.representative_signatures.decode, cache_k17));
      group.llm_inference_context.decode_input_buffers[cache_k17] =
          std::move(buffer_k);
      LITERT_ASSIGN_OR_RETURN(
          auto buffer_v,
          llm_compiled_model.CreateInputBuffer(
              group.representative_signatures.decode, cache_v17));
      group.llm_inference_context.decode_input_buffers[cache_v17] =
          std::move(buffer_v);
    }

    LITERT_ASSIGN_OR_RETURN(
        group.mask_context,
        CreateMaskContextWithBufferSharing(
            npu_auxiliary_context, signatures_list,
            gemma_prefill_input_buffers_by_size, gemma_decode_input_buffers,
            gemma_verify_input_buffers));

    LITERT_ASSIGN_OR_RETURN(
        group.rope_context,
        CreateRopeContextWithBufferSharing(
            npu_auxiliary_context, signatures_list,
            gemma_prefill_input_buffers_by_size, gemma_decode_input_buffers,
            gemma_verify_input_buffers));

    // Duplicate the rope's buffers that are used to store the prefill and
    // decode input position, because they will need to be passed to the
    // cache update inference context as well.
    absl::flat_hash_map<int, ::litert::TensorBuffer> prefill_input_pos_by_size;
    absl::flat_hash_map<int, ::litert::TensorBuffer> prefill_valid_mask_by_size;

    for (const auto& sigs : signatures_list) {
      int size = sigs.size;
      LITERT_ASSIGN_OR_RETURN(prefill_input_pos_by_size[size],
                              group.rope_context.GetPrefillInputBuffers(size)
                                  .at(RopeSignatures::kInputPos)
                                  .Duplicate());

      const auto& mask_prefill_buffers =
          group.mask_context.GetPrefillInputBuffers(size);
      auto prefill_valid_mask_it =
          mask_prefill_buffers.find(MaskSignatures::kMaskInputValidMask);
      if (prefill_valid_mask_it != mask_prefill_buffers.end()) {
        LITERT_ASSIGN_OR_RETURN(prefill_valid_mask_by_size[size],
                                prefill_valid_mask_it->second.Duplicate());
      }
    }

    auto verify_input_pos_it =
        group.rope_context.verify_input_buffers.find(RopeSignatures::kInputPos);
    ::litert::TensorBuffer verify_input_pos;
    if (verify_input_pos_it != group.rope_context.verify_input_buffers.end()) {
      LITERT_ASSIGN_OR_RETURN(verify_input_pos,
                              verify_input_pos_it->second.Duplicate());
    }

    LITERT_ASSIGN_OR_RETURN(
        ::litert::TensorBuffer decode_input_pos,
        group.rope_context.decode_input_buffers[RopeSignatures::kInputPos]
            .Duplicate());

    ::litert::TensorBuffer decode_valid_mask;
    auto decode_valid_mask_it = group.mask_context.decode_input_buffers.find(
        MaskSignatures::kMaskInputValidMask);
    if (decode_valid_mask_it != group.mask_context.decode_input_buffers.end()) {
      LITERT_ASSIGN_OR_RETURN(decode_valid_mask,
                              decode_valid_mask_it->second.Duplicate());
    }

    ::litert::TensorBuffer verify_valid_mask;
    auto verify_valid_mask_it = group.mask_context.verify_input_buffers.find(
        MaskSignatures::kMaskInputValidMask);
    if (verify_valid_mask_it != group.mask_context.verify_input_buffers.end()) {
      LITERT_ASSIGN_OR_RETURN(verify_valid_mask,
                              verify_valid_mask_it->second.Duplicate());
    }

    LITERT_ASSIGN_OR_RETURN(
        group.cache_update_inference_context,
        CreateCacheUpdateInferenceContextWithBufferSharing(
            input_kv_cache_buffers,
            prefill_output_kv_cache_slice_buffers_by_size,
            decode_output_kv_cache_slice_buffers,
            verify_output_kv_cache_slice_buffers, prefill_input_pos_by_size,
            std::move(decode_input_pos), std::move(verify_input_pos),
            prefill_valid_mask_by_size, std::move(decode_valid_mask),
            std::move(verify_valid_mask)));

    if (embedder_lrt_model) {
      LITERT_ASSIGN_OR_RETURN(
          group.embedder_context,
          CreateEmbedderContextWithBufferSharing(
              env, *embedder_lrt_model, signatures_list,
              gemma_prefill_input_buffers_by_size, gemma_decode_input_buffers,
              gemma_verify_input_buffers, executor_settings));
    }

    for (const auto& sigs : signatures_list) {
      int size = sigs.size;
      ABSL_LOG(INFO)
          << "CreateContextGroup: inserting into prefill_signature_map: "
          << size << " -> " << sigs.prefill;
      group.prefill_signature_map[size] = sigs.prefill;
    }

    return std::move(group);
  };

  LITERT_ASSIGN_OR_RETURN(
      auto largest_group_created,
      create_context_group(prefill_signatures_list, /*largest_group=*/nullptr));

  std::unique_ptr<EmbeddingLookupManager> maybe_embedding_lookup_manager =
      nullptr;
  if (resources.GetTFLiteModel(ModelType::kTfLiteVisionEncoder).ok() ||
      resources.GetTFLiteModel(ModelType::kTfLiteAudioEncoderHw).ok()) {
    absl::flat_hash_map<int, const Model*> end_of_multi_modal_embedding_models;
    auto add_multi_modal_end_model = [&](ModelType type, int token) {
      auto model_buffer = resources.GetTFLiteModelBuffer(type);
      if (model_buffer.ok() && !model_buffer->empty()) {
        auto model = resources.GetTFLiteModel(type);
        if (model.ok()) {
          end_of_multi_modal_embedding_models[token] = *model;
        }
      }
    };

    add_multi_modal_end_model(ModelType::kTfLiteEndOfAudio,
                              litert::lm::ExecutorAudioData::kEndToken);
    add_multi_modal_end_model(ModelType::kTfLiteEndOfVision,
                              litert::lm::ExecutorVisionData::kEndToken);

    LITERT_ASSIGN_OR_RETURN(
        maybe_embedding_lookup_manager,
        EmbeddingLookupManager::Create(env, embedder_lrt_model,
                                       end_of_multi_modal_embedding_models,
                                       true, "decode_embedder"));
  }

  SpeculativeDecodingType speculative_decoding_type =
      SpeculativeDecodingType::kNone;
  std::optional<DrafterContext> drafter_context = std::nullopt;
  std::optional<DrafterAuxContext> drafter_aux_context = std::nullopt;

  if (executor_settings.GetAdvancedSettings().has_value() &&
      executor_settings.GetAdvancedSettings()->enable_speculative_decoding) {
    auto mtp_drafter_model =
        resources.GetTFLiteModel(ModelType::kTfLiteMtpDrafter);
    auto mtp_aux_model = resources.GetTFLiteModel(ModelType::kTfLiteMtpAux);

    if (mtp_drafter_model.ok() && mtp_aux_model.ok()) {
      return absl::InvalidArgumentError(
          "Speculative decoding is not supported for model without per layer "
          "embedding.");
    }
  }

  const bool has_sliding_window_attention = DetectIsSwa(
      largest_group_created.llm_inference_context.prefill_input_buffers_by_size
          .begin()
          ->second);

  ABSL_RETURN_IF_ERROR(WarmupInference(
      llm_compiled_model, largest_group_created.llm_inference_context,
      npu_auxiliary_context.npu_auxiliary_compiled_model,
      prefill_signatures_list, largest_group_created.rope_context,
      largest_group_created.mask_context,
      largest_group_created.cache_update_inference_context));

  absl::flat_hash_map<int, ContextGroup> context_groups;

  for (int idx = 0; idx < sorted_supported_context_sizes.size() - 1; ++idx) {
    int c = sorted_supported_context_sizes[idx];
    LITERT_ASSIGN_OR_RETURN(auto group,
                            create_context_group(signatures_list_of_list[idx],
                                                 &largest_group_created));

    ABSL_RETURN_IF_ERROR(WarmupInference(
        llm_compiled_model, group.llm_inference_context,
        npu_auxiliary_context.npu_auxiliary_compiled_model,
        signatures_list_of_list[idx], group.rope_context, group.mask_context,
        group.cache_update_inference_context));

    context_groups.emplace(c, std::move(group));
  }

  context_groups.emplace(largest_context_size,
                         std::move(largest_group_created));

  auto executor = absl::WrapUnique(new LlmLiteRtNpuCompiledModelExecutor(
      executor_settings, env, EmbedderContext(),
      std::move(npu_auxiliary_context), std::move(llm_compiled_model),
      std::move(maybe_embedding_lookup_manager),
      /*per_layer_embedding_lookup_manager=*/nullptr, quantization_params, {},
      {}, {}, 0, 0, litert::ElementType::None, litert::ElementType::None, 1.0f,
      1.0f, 0, std::move(kv_quant_params), kv_cache_init_value, nullptr,
      std::nullopt, std::move(context_groups), sorted_supported_context_sizes,
      speculative_decoding_type, has_sliding_window_attention));
  ABSL_RETURN_IF_ERROR(executor->SwitchContextSizeIfRequired(
      sorted_supported_context_sizes.front()));
  return executor;
}

absl::Status LlmLiteRtNpuCompiledModelExecutor::ClearKVCache(
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>& buffers)
    const {
  for (auto& [buffer_name, buffer] : buffers) {
    if (buffer_name.starts_with(kv_cache_k_root_name) ||
        buffer_name.starts_with(kv_cache_v_root_name) ||
        buffer_name.starts_with(kv_cache_c_root_name)) {
      LITERT_RETURN_IF_ERROR(FillKVCacheBuffer(buffer, kv_cache_init_value_));
    }
  }
  return absl::OkStatus();
}

}  // namespace litert::lm
