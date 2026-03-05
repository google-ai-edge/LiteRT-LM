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

#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_common.h"  // from @litert
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_expected.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_options.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/cc/options/litert_qualcomm_options.h"  // from @litert
#include "runtime/components/embedding_lookup/embedding_lookup_manager.h"
#include "runtime/components/model_resources.h"
#include "runtime/executor/litert_compiled_model_executor_utils.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/executor/llm_executor_processed_tokens.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/status_macros.h"  // NOLINT

namespace litert::lm {

namespace {
using ::litert::CompiledModel;
using ::litert::Environment;
using ::litert::TensorBuffer;

constexpr char kPrefillSignature[] = "prefill_128";
constexpr int kPrefillSize = 128;
constexpr char kDecodeSignature[] = "decode";
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

// Signature names for the embedder.
struct EmbedderSignatures {
  static constexpr absl::string_view kPrefillEmbedder = "prefill_embedder_128";
  static constexpr absl::string_view kDecodeEmbedder = "decode_embedder";
  // Prefill and decode use identical tensor signature names.
  static constexpr absl::string_view kEmbedderInput = "token_ids";
  static constexpr absl::string_view kEmbedderOutput = "embeddings";
};

static constexpr absl::string_view kPerLayerEmbedderTensor =
    "per_layer_embeddings";

struct EmbedderPerLayerSignatures {
  static constexpr absl::string_view kPrefillEmbedderPerLayer =
      "prefill_per_layer_embedder_128";
  static constexpr absl::string_view kDecodeEmbedderPerLayer =
      "decode_per_layer_embedder";
  // Prefill and decode use identical tensor signature names.
  static constexpr absl::string_view kEmbedderInput = "token_ids";
  static constexpr absl::string_view kEmbedderOutput = "embeddings";
};

// Signature names for the mask signatures.
struct MaskSignatures {
  static constexpr absl::string_view kPrefillMask = "prefill_mask_128";
  static constexpr absl::string_view kDecodeMask = "decode_mask";
  // Prefill and decode use identical tensor signature names.
  static constexpr absl::string_view kMaskInputTimeStep = "time_step";
  static constexpr absl::string_view kMaskInputTokens = "input_tokens";
  static constexpr absl::string_view kMaskOutputLocalMask = "mask_local";
  static constexpr absl::string_view kMaskOutputGlobalMask = "mask_global";
};

// Signature names for the rope signatures.
struct RopeSignatures {
  static constexpr absl::string_view kPrefillRope = "prefill_rope_128";
  static constexpr absl::string_view kDecodeRope = "decode_rope";
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
  static constexpr absl::string_view kPrefillLlm = "prefill_128";
  static constexpr absl::string_view kDecodeLlm = "decode";
  static constexpr absl::string_view kInputEmbeddings = "embeddings";
  static constexpr absl::string_view kDecodeLogitsOutput = "logits";
};

// Signature names for the cache update signatures.
struct CacheUpdateSignatures {
  static constexpr absl::string_view kPrefillCacheUpdate =
      "prefill_cache_update_128";
  static constexpr absl::string_view kDecodeCacheUpdate = "decode_cache_update";
  static constexpr absl::string_view kInputPos = "input_pos";
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

template <typename T>
absl::StatusOr<int> FindMaxIndex(const TensorBuffer& decoded_logits) {
  LITERT_ASSIGN_OR_RETURN(auto logits_buffer,
                          CopyFromTensorBuffer<T>(decoded_logits));
  if (logits_buffer.empty()) {
    return absl::InvalidArgumentError("Logits buffer is empty.");
  }
  int max_index = 0;
  T max_value = std::numeric_limits<T>::min();
  for (int i = 0; i < logits_buffer.size(); ++i) {
    if (logits_buffer[i] > max_value) {
      max_value = logits_buffer[i];
      max_index = i;
    }
  }
  return max_index;
}

// Applies greedy sampling to the decoded logits. TODO(b/416702864) this logic
// should be replaced by the LiteRT-LM sampler once it supports greedy sampling
// for quantized tensors.
absl::StatusOr<int> ApplyGreedySampling(const TensorBuffer& decoded_logits) {
  LITERT_ASSIGN_OR_RETURN(RankedTensorType logits_tensor_type,
                          decoded_logits.TensorType());
  if (logits_tensor_type.ElementType() == ::litert::ElementType::Float32) {
    return FindMaxIndex<float>(decoded_logits);
  } else if (logits_tensor_type.ElementType() == ::litert::ElementType::Int16) {
    return FindMaxIndex<int16_t>(decoded_logits);
  } else if (logits_tensor_type.ElementType() == ::litert::ElementType::Int8) {
    return FindMaxIndex<int8_t>(decoded_logits);
  } else {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported tensor element type for greedy sampling: ",
                     logits_tensor_type.ElementType()));
  }
}

// Returns true if the transformer model has a per layer embedder input buffer.
litert::Expected<bool> HasPerLayerEmbedder(
    const litert::Model& transformer_model) {
  LITERT_ASSIGN_OR_RETURN(
      auto input_names,
      transformer_model.GetSignatureInputNames(kPrefillSignature));
  for (auto input_name : input_names) {
    if (kPerLayerEmbedderTensor == input_name) {
      return true;
    }
  }
  return false;
}

float GetToksPrefill(
    const LlmLiteRtNpuCompiledModelExecutor::LatencyStats& latency_stats) {
  return ((latency_stats.prefill_num_tokens * 1000 * 1000) /
          (float)latency_stats.prefill_e2e_latency_us);
}

float GetToksDecode(
    const LlmLiteRtNpuCompiledModelExecutor::LatencyStats& latency_stats) {
  return ((latency_stats.decode_num_tokens * 1000 * 1000) /
          (float)latency_stats.decode_e2e_latency_us);
}

void PrintLatencyStats(
    const LlmLiteRtNpuCompiledModelExecutor::LatencyStats& latency_stats) {
  std::ostringstream formatted_stats;
  formatted_stats << "\n" << "====== PREFILL STATS ======";
  formatted_stats << "\n"
                  << "Total prefill latency [us]: "
                  << latency_stats.prefill_e2e_latency_us;
  formatted_stats << "\n"
                  << "(e2e) Prefill num tokens: "
                  << latency_stats.prefill_num_tokens;
  formatted_stats << "\n"
                  << "(e2e) Prefill tokens per second: "
                  << GetToksPrefill(latency_stats);
  formatted_stats << "\n"
                  << "(TransformerStackOnly) Prefill tokens per second: "
                  << ((latency_stats.prefill_num_tokens * 1000 * 1000) /
                      (float)latency_stats.prefill_llm_inference_latency_us);

  formatted_stats << "\n" << "------ Prefill breakdown ------";
  formatted_stats << "\n"
                  << "Total prefill prepare input tensors latency [us]: "
                  << latency_stats.prefill_prepare_input_latency_us << " ("
                  << ((latency_stats.prefill_prepare_input_latency_us * 100) /
                      (float)latency_stats.prefill_e2e_latency_us)
                  << "%)";
  formatted_stats << "\n"
                  << "Total prefill embedder inference latency [us]: "
                  << latency_stats.prefill_embedder_inference_latency_us << " ("
                  << ((latency_stats.prefill_embedder_inference_latency_us *
                       100) /
                      (float)latency_stats.prefill_e2e_latency_us)
                  << "%)";
  if (latency_stats.prefill_embedder_per_layer_inference_latency_us
          .has_value()) {
    formatted_stats
        << "\n"
        << "Total prefill embedder per layer inference latency [us]: "
        << latency_stats.prefill_embedder_per_layer_inference_latency_us.value()
        << " ("
        << ((latency_stats.prefill_embedder_per_layer_inference_latency_us
                 .value() *
             100) /
            (float)latency_stats.prefill_e2e_latency_us)
        << "%)";
  }
  formatted_stats << "\n"
                  << "Total prefill rope inference latency [us]: "
                  << latency_stats.prefill_rope_inference_latency_us << " ("
                  << ((latency_stats.prefill_rope_inference_latency_us * 100) /
                      (float)latency_stats.prefill_e2e_latency_us)
                  << "%)";
  formatted_stats << "\n"
                  << "Total prefill mask inference latency [us]: "
                  << latency_stats.prefill_mask_inference_latency_us << " ("
                  << ((latency_stats.prefill_mask_inference_latency_us * 100) /
                      (float)latency_stats.prefill_e2e_latency_us)
                  << "%)";
  formatted_stats << "\n"
                  << "Total prefill LLM inference latency [us]: "
                  << latency_stats.prefill_llm_inference_latency_us << " ("
                  << ((latency_stats.prefill_llm_inference_latency_us * 100) /
                      (float)latency_stats.prefill_e2e_latency_us)
                  << "%)";
  formatted_stats << "\n"
                  << "Total prefill cache update inference latency [us]: "
                  << latency_stats.prefill_cache_update_inference_latency_us
                  << " ("
                  << ((latency_stats.prefill_cache_update_inference_latency_us *
                       100) /
                      (float)latency_stats.prefill_e2e_latency_us)
                  << "%)";

  formatted_stats << "\n" << "\n====== DECODE STATS ======";
  formatted_stats << "\n"
                  << "Total decode latency [us]: "
                  << latency_stats.decode_e2e_latency_us;
  formatted_stats << "\n"
                  << "Decode num tokens: " << latency_stats.decode_num_tokens;
  formatted_stats << "\n"
                  << "Decode tokens per second: "
                  << GetToksDecode(latency_stats);
  formatted_stats << "\n"
                  << "(TransformerStackOnly) Decode tokens per second: "
                  << ((latency_stats.decode_num_tokens * 1000 * 1000) /
                      (float)latency_stats.decode_llm_inference_latency_us);

  formatted_stats << "\n" << "------ Decode breakdown ------";
  formatted_stats << "\n"
                  << "Total decode prepare input tensors latency [us]: "
                  << latency_stats.decode_prepare_input_latency_us << " ("
                  << ((latency_stats.decode_prepare_input_latency_us * 100) /
                      (float)latency_stats.decode_e2e_latency_us)
                  << "%)";
  formatted_stats << "\n"
                  << "Total decode embedder inference latency [us]: "
                  << latency_stats.decode_embedder_inference_latency_us << " ("
                  << ((latency_stats.decode_embedder_inference_latency_us *
                       100) /
                      (float)latency_stats.decode_e2e_latency_us)
                  << "%)";
  if (latency_stats.decode_embedder_per_layer_inference_latency_us
          .has_value()) {
    formatted_stats
        << "\n"
        << "Total decode embedder per layer inference latency [us]: "
        << latency_stats.decode_embedder_per_layer_inference_latency_us.value()
        << " ("
        << ((latency_stats.decode_embedder_per_layer_inference_latency_us
                 .value() *
             100) /
            (float)latency_stats.decode_e2e_latency_us)
        << "%)";
  }
  formatted_stats << "\n"
                  << "Total decode rope inference latency [us]: "
                  << latency_stats.decode_rope_inference_latency_us << " ("
                  << ((latency_stats.decode_rope_inference_latency_us * 100) /
                      (float)latency_stats.decode_e2e_latency_us)
                  << "%)";
  formatted_stats << "\n"
                  << "Total decode mask inference latency [us]: "
                  << latency_stats.decode_mask_inference_latency_us << " ("
                  << ((latency_stats.decode_mask_inference_latency_us * 100) /
                      (float)latency_stats.decode_e2e_latency_us)
                  << "%)";
  formatted_stats << "\n"
                  << "Total decode LLM inference latency [us]: "
                  << latency_stats.decode_llm_inference_latency_us << " ("
                  << ((latency_stats.decode_llm_inference_latency_us * 100) /
                      (float)latency_stats.decode_e2e_latency_us)
                  << "%)";
  formatted_stats << "\n"
                  << "Total decode cache update inference latency [us]: "
                  << latency_stats.decode_cache_update_inference_latency_us
                  << " ("
                  << ((latency_stats.decode_cache_update_inference_latency_us *
                       100) /
                      (float)latency_stats.decode_e2e_latency_us)
                  << "%)";
  formatted_stats << "\n"
                  << "Total decode sampling latency [us]: "
                  << latency_stats.decode_sampling_latency_us << " ("
                  << ((latency_stats.decode_sampling_latency_us * 100) /
                      (float)latency_stats.decode_e2e_latency_us)
                  << "%)\n";

  ABSL_LOG(INFO) << "Custom NPU execution latency stats:\n"
                 << formatted_stats.str();
}

}  // namespace

absl::StatusOr<LlmLiteRtNpuCompiledModelExecutor::EmbedderContext>
LlmLiteRtNpuCompiledModelExecutor::CreateEmbedderContextWithBufferSharing(
    ::litert::Environment& env, const litert::Model& embedder_model,
    const ::litert::TensorBuffer& prefill_input_tokens,
    const ::litert::TensorBuffer& decode_input_tokens,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        gemma_prefill_input_buffers,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        gemma_decode_input_buffers) {
  LITERT_ASSIGN_OR_RETURN(CompiledModel embedder_compiled_model,
                          CompiledModel::Create(env, embedder_model.Get(),
                                                litert::HwAccelerators::kCpu));

  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      prefill_input_buffers;
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      prefill_output_buffers;
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      decode_input_buffers;
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      decode_output_buffers;

  LITERT_ASSIGN_OR_RETURN(
      prefill_input_buffers[EmbedderSignatures::kEmbedderInput],
      prefill_input_tokens.Duplicate());

  LITERT_ASSIGN_OR_RETURN(
      prefill_output_buffers[EmbedderSignatures::kEmbedderOutput],
      gemma_prefill_input_buffers[LlmSignatures::kInputEmbeddings].Duplicate());

  LITERT_ASSIGN_OR_RETURN(
      decode_input_buffers[EmbedderSignatures::kEmbedderInput],
      decode_input_tokens.Duplicate());

  LITERT_ASSIGN_OR_RETURN(
      decode_output_buffers[EmbedderSignatures::kEmbedderOutput],
      gemma_decode_input_buffers[LlmSignatures::kInputEmbeddings].Duplicate());

  EmbedderContext embedder_context(
      std::move(embedder_compiled_model), std::move(prefill_input_buffers),
      std::move(prefill_output_buffers), std::move(decode_input_buffers),
      std::move(decode_output_buffers));
  return embedder_context;
}

absl::StatusOr<LlmLiteRtNpuCompiledModelExecutor::EmbedderPerLayerContext>
LlmLiteRtNpuCompiledModelExecutor::
    CreateEmbedderPerLayerContextWithBufferSharing(
        ::litert::Environment& env, const litert::Model& embedder_model,
        const ::litert::TensorBuffer& prefill_input_tokens,
        const ::litert::TensorBuffer& decode_input_tokens,
        absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
            gemma_prefill_input_buffers,
        absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
            gemma_decode_input_buffers) {
  LITERT_ASSIGN_OR_RETURN(CompiledModel embedder_compiled_model,
                          CompiledModel::Create(env, embedder_model.Get(),
                                                litert::HwAccelerators::kCpu));

  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      prefill_input_buffers;
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      prefill_output_buffers;
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      decode_input_buffers;
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      decode_output_buffers;

  LITERT_ASSIGN_OR_RETURN(
      prefill_input_buffers[EmbedderPerLayerSignatures::kEmbedderInput],
      prefill_input_tokens.Duplicate());

  LITERT_ASSIGN_OR_RETURN(
      prefill_output_buffers[EmbedderPerLayerSignatures::kEmbedderOutput],
      gemma_prefill_input_buffers[kPerLayerEmbedderTensor].Duplicate());

  LITERT_ASSIGN_OR_RETURN(
      decode_input_buffers[EmbedderPerLayerSignatures::kEmbedderInput],
      decode_input_tokens.Duplicate());

  LITERT_ASSIGN_OR_RETURN(
      decode_output_buffers[EmbedderPerLayerSignatures::kEmbedderOutput],
      gemma_decode_input_buffers[kPerLayerEmbedderTensor].Duplicate());

  EmbedderPerLayerContext embedder_per_layer_context(
      std::move(embedder_compiled_model), std::move(prefill_input_buffers),
      std::move(prefill_output_buffers), std::move(decode_input_buffers),
      std::move(decode_output_buffers));
  return embedder_per_layer_context;
}

absl::StatusOr<LlmLiteRtNpuCompiledModelExecutor::NpuAuxiliaryContext>
LlmLiteRtNpuCompiledModelExecutor::CreateNpuAuxiliaryContext(
    ::litert::Environment& env, const litert::Model& npu_auxiliary_model) {
  LITERT_ASSIGN_OR_RETURN(auto npu_auxiliary_compiled_model,
                          CompiledModel::Create(env, npu_auxiliary_model.Get(),
                                                litert::HwAccelerators::kCpu));
  NpuAuxiliaryContext npu_auxiliary_context(
      std::move(npu_auxiliary_compiled_model));
  return npu_auxiliary_context;
}

absl::StatusOr<LlmLiteRtNpuCompiledModelExecutor::InferenceContext>
LlmLiteRtNpuCompiledModelExecutor::CreateMaskContextWithBufferSharing(
    NpuAuxiliaryContext& npu_auxiliary_context,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        gemma_prefill_input_buffers,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        gemma_decode_input_buffers) {
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      prefill_input_buffers;
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      prefill_output_buffers;
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      decode_input_buffers;
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      decode_output_buffers;

  LITERT_ASSIGN_OR_RETURN(
      prefill_input_buffers[MaskSignatures::kMaskInputTimeStep],
      npu_auxiliary_context.npu_auxiliary_compiled_model.CreateInputBuffer(
          MaskSignatures::kPrefillMask, MaskSignatures::kMaskInputTimeStep));
  prefill_input_buffers[MaskSignatures::kMaskInputTimeStep].Clear();
  LITERT_ASSIGN_OR_RETURN(
      prefill_input_buffers[MaskSignatures::kMaskInputTokens],
      npu_auxiliary_context.npu_auxiliary_compiled_model.CreateInputBuffer(
          MaskSignatures::kPrefillMask, MaskSignatures::kMaskInputTokens));
  prefill_input_buffers[MaskSignatures::kMaskInputTokens].Clear();
  const std::set<absl::string_view> mask_output_names = {
      MaskSignatures::kMaskOutputLocalMask,
      MaskSignatures::kMaskOutputGlobalMask};
  for (const auto& mask_output_name : mask_output_names) {
    LITERT_RETURN_IF_ERROR(
        gemma_prefill_input_buffers.contains(mask_output_name))
        << "Missing mask output buffer: " << mask_output_name;
    LITERT_ASSIGN_OR_RETURN(
        prefill_output_buffers[mask_output_name],
        gemma_prefill_input_buffers[mask_output_name].Duplicate());
  }

  LITERT_ASSIGN_OR_RETURN(
      decode_input_buffers[MaskSignatures::kMaskInputTimeStep],
      npu_auxiliary_context.npu_auxiliary_compiled_model.CreateInputBuffer(
          MaskSignatures::kDecodeMask, MaskSignatures::kMaskInputTimeStep));
  decode_input_buffers[MaskSignatures::kMaskInputTimeStep].Clear();
  LITERT_ASSIGN_OR_RETURN(
      decode_input_buffers[MaskSignatures::kMaskInputTokens],
      npu_auxiliary_context.npu_auxiliary_compiled_model.CreateInputBuffer(
          MaskSignatures::kDecodeMask, MaskSignatures::kMaskInputTokens));
  decode_input_buffers[MaskSignatures::kMaskInputTokens].Clear();

  for (const auto& mask_output_name : mask_output_names) {
    LITERT_RETURN_IF_ERROR(
        gemma_decode_input_buffers.contains(mask_output_name))
        << "Missing mask output buffer: " << mask_output_name;
    LITERT_ASSIGN_OR_RETURN(
        decode_output_buffers[mask_output_name],
        gemma_decode_input_buffers[mask_output_name].Duplicate());
  }

  InferenceContext mask_context(
      std::move(prefill_input_buffers), std::move(prefill_output_buffers),
      std::move(decode_input_buffers), std::move(decode_output_buffers));
  return mask_context;
}

absl::StatusOr<LlmLiteRtNpuCompiledModelExecutor::InferenceContext>
LlmLiteRtNpuCompiledModelExecutor::CreateRopeContextWithBufferSharing(
    NpuAuxiliaryContext& npu_auxiliary_context,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        gemma_prefill_input_buffers,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        gemma_decode_input_buffers) {
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      prefill_input_buffers;
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      prefill_output_buffers;
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      decode_input_buffers;
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      decode_output_buffers;

  LITERT_ASSIGN_OR_RETURN(
      prefill_input_buffers[RopeSignatures::kInputPos],
      npu_auxiliary_context.npu_auxiliary_compiled_model.CreateInputBuffer(
          RopeSignatures::kPrefillRope, RopeSignatures::kInputPos));
  prefill_input_buffers[RopeSignatures::kInputPos].Clear();

  const std::set<absl::string_view> rope_output_names = {
      RopeSignatures::kOutputPosEmbeddingLocalLow,
      RopeSignatures::kOutputPosEmbeddingHigh,
      RopeSignatures::kOutputPosEmbeddingLocalHigh,
      RopeSignatures::kOutputPosEmbeddingLow};
  for (const auto& rope_output_name : rope_output_names) {
    LITERT_RETURN_IF_ERROR(
        gemma_prefill_input_buffers.contains(rope_output_name))
        << "Missing rope output buffer: " << rope_output_name;
    LITERT_ASSIGN_OR_RETURN(
        prefill_output_buffers[rope_output_name],
        gemma_prefill_input_buffers[rope_output_name].Duplicate());
  }

  LITERT_ASSIGN_OR_RETURN(
      decode_input_buffers[RopeSignatures::kInputPos],
      npu_auxiliary_context.npu_auxiliary_compiled_model.CreateInputBuffer(
          RopeSignatures::kDecodeRope, RopeSignatures::kInputPos));
  decode_input_buffers[RopeSignatures::kInputPos].Clear();

  for (const auto& rope_output_name : rope_output_names) {
    LITERT_RETURN_IF_ERROR(
        gemma_decode_input_buffers.contains(rope_output_name))
        << "Missing rope output buffer: " << rope_output_name;
    LITERT_ASSIGN_OR_RETURN(
        decode_output_buffers[rope_output_name],
        gemma_decode_input_buffers[rope_output_name].Duplicate());
  }

  InferenceContext rope_context(
      std::move(prefill_input_buffers), std::move(prefill_output_buffers),
      std::move(decode_input_buffers), std::move(decode_output_buffers));
  return rope_context;
}

absl::Status LlmLiteRtNpuCompiledModelExecutor::AllocateTransformerBuffers(
    litert::Environment& env, const litert::Model* transformer_model,
    CompiledModel& llm_compiled_model,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        gemma_prefill_input_buffers,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        gemma_decode_input_buffers,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        input_kv_cache_buffers,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        prefill_output_kv_cache_slice_buffers,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        decode_output_kv_cache_slice_buffers) {
  auto prefill_signature = transformer_model->FindSignature(kPrefillSignature);

  constexpr absl::string_view kv_cache_slice_k_root_name = "kv_slice_k_";
  constexpr absl::string_view kv_cache_slice_v_root_name = "kv_slice_v_";

  // Create input buffers for prefill signature.
  for (auto input_name : prefill_signature->InputNames()) {
    if (absl::StartsWith(input_name, kv_cache_k_root_name) ||
        absl::StartsWith(input_name, kv_cache_v_root_name)) {
      LITERT_ASSIGN_OR_RETURN(
          input_kv_cache_buffers[input_name],
          llm_compiled_model.CreateInputBuffer(kPrefillSignature, input_name));
      input_kv_cache_buffers[input_name].Clear();
    } else {
      LITERT_ASSIGN_OR_RETURN(
          gemma_prefill_input_buffers[input_name],
          llm_compiled_model.CreateInputBuffer(kPrefillSignature, input_name));
      gemma_prefill_input_buffers[input_name].Clear();
    }
  }
  // Create input buffers for decode signature. Skip kv cache input buffers as
  // they are already created in the prefill signature.
  auto decode_signature = transformer_model->FindSignature(kDecodeSignature);
  for (auto input_name : decode_signature->InputNames()) {
    if (absl::StartsWith(input_name, kv_cache_k_root_name) ||
        absl::StartsWith(input_name, kv_cache_v_root_name)) {
      // Create the input kv cache buffer for the decode signature if it is not
      // created in the prefill signature.
      if (!input_kv_cache_buffers.contains(input_name)) {
        LITERT_ASSIGN_OR_RETURN(
            input_kv_cache_buffers[input_name],
            llm_compiled_model.CreateInputBuffer(kDecodeSignature, input_name));
        input_kv_cache_buffers[input_name].Clear();
      }
      continue;
    }
    LITERT_ASSIGN_OR_RETURN(
        gemma_decode_input_buffers[input_name],
        llm_compiled_model.CreateInputBuffer(kDecodeSignature, input_name));
    gemma_decode_input_buffers[input_name].Clear();
  }

  // Create output buffers for prefill signature.
  for (auto output_name : prefill_signature->OutputNames()) {
    if (absl::StartsWith(output_name, kv_cache_slice_k_root_name) ||
        absl::StartsWith(output_name, kv_cache_slice_v_root_name)) {
      LITERT_ASSIGN_OR_RETURN(
          prefill_output_kv_cache_slice_buffers[output_name],
          llm_compiled_model.CreateOutputBuffer(kPrefillSignature,
                                                output_name));
    }
  }
  // Create output buffers for decode signature.
  for (auto output_name : decode_signature->OutputNames()) {
    if (absl::StartsWith(output_name, kv_cache_slice_k_root_name) ||
        absl::StartsWith(output_name, kv_cache_slice_v_root_name)) {
      LITERT_ASSIGN_OR_RETURN(
          decode_output_kv_cache_slice_buffers[output_name],
          llm_compiled_model.CreateOutputBuffer(kDecodeSignature, output_name));
    }
  }

  return absl::OkStatus();
}

absl::StatusOr<LlmLiteRtNpuCompiledModelExecutor::InferenceContext>
LlmLiteRtNpuCompiledModelExecutor::CreateLlmInferenceContextWithBufferSharing(
    ::litert::Environment& env, ::litert::CompiledModel& llm_compiled_model,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        input_kv_cache_buffers,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        prefill_output_kv_cache_slice_buffers,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        decode_output_kv_cache_slice_buffers,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        gemma_prefill_input_buffers,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        gemma_decode_input_buffers) {
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      prefill_input_buffers;
  {
    for (const auto& [key, value] : gemma_prefill_input_buffers) {
      LITERT_ASSIGN_OR_RETURN(prefill_input_buffers[key], value.Duplicate());
    }
    // Duplicate all kv cache buffers to prefill inputs.
    LITERT_ASSIGN_OR_RETURN(
        auto prefill_input_names,
        llm_compiled_model.GetSignatureInputNames(kPrefillSignature));
    for (const auto& [key, value] : input_kv_cache_buffers) {
      // Check if the kv cache buffer is used in the prefill signature.
      if (absl::c_find(prefill_input_names, std::string(key)) ==
          prefill_input_names.end()) {
        continue;
      }

      // The last layer kv cache in the prefill signature has float32 elements,
      // although its not used in the model, CompiledModel will complain about
      // the mismatched buffer size. So we need to correct the buffer size here,
      // by creating a new buffer with the correct size.
      LITERT_ASSIGN_OR_RETURN(
          auto input_tensor_type,
          llm_compiled_model.GetInputTensorType(kPrefillSignature, key));
      LITERT_ASSIGN_OR_RETURN(auto input_tensor_size,
                              input_tensor_type.Bytes());
      LITERT_ASSIGN_OR_RETURN(auto input_buffer_size, value.Size());
      if (input_tensor_size != input_buffer_size) {
        LITERT_ASSIGN_OR_RETURN(
            auto corrected_input_buffer,
            llm_compiled_model.CreateInputBuffer(kPrefillSignature, key));
        corrected_input_buffer.Clear();
        LITERT_ASSIGN_OR_RETURN(prefill_input_buffers[key],
                                corrected_input_buffer.Duplicate());
      } else {
        LITERT_ASSIGN_OR_RETURN(prefill_input_buffers[key], value.Duplicate());
      }
    }
  }
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      prefill_output_buffers;
  {
    // Duplicate all output kv cache slice buffers to prefill output
    // buffers.
    for (const auto& [key, value] : prefill_output_kv_cache_slice_buffers) {
      LITERT_ASSIGN_OR_RETURN(prefill_output_buffers[key], value.Duplicate());
    }
  }
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      decode_input_buffers;
  {
    for (const auto& [key, value] : gemma_decode_input_buffers) {
      LITERT_ASSIGN_OR_RETURN(decode_input_buffers[key], value.Duplicate());
    }
    // Duplicate all kv cache buffers to decode inputs.
    for (const auto& [key, value] : input_kv_cache_buffers) {
      LITERT_ASSIGN_OR_RETURN(decode_input_buffers[key], value.Duplicate());
    }
  }
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      decode_output_buffers;
  {
    // Duplicate all output kv cache slice buffers to decode output
    // buffers.
    for (const auto& [key, value] : decode_output_kv_cache_slice_buffers) {
      LITERT_ASSIGN_OR_RETURN(decode_output_buffers[key], value.Duplicate());
    }

    // The decode signature has an additional output buffer for logits.
    LITERT_ASSIGN_OR_RETURN(
        decode_output_buffers[LlmSignatures::kDecodeLogitsOutput],
        llm_compiled_model.CreateOutputBuffer(
            kDecodeSignature, LlmSignatures::kDecodeLogitsOutput));
  }
  return InferenceContext(
      std::move(prefill_input_buffers), std::move(prefill_output_buffers),
      std::move(decode_input_buffers), std::move(decode_output_buffers));
}

absl::StatusOr<LlmLiteRtNpuCompiledModelExecutor::InferenceContext>
LlmLiteRtNpuCompiledModelExecutor::
    CreateCacheUpdateInferenceContextWithBufferSharing(
        absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
            input_kv_cache_buffers,
        absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
            prefill_output_kv_cache_slice_buffers,
        absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
            decode_output_kv_cache_slice_buffers,
        ::litert::TensorBuffer prefill_input_pos,
        ::litert::TensorBuffer decode_input_pos)

{
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      prefill_input_buffers;
  {
    for (const auto& [key, value] : input_kv_cache_buffers) {
      LITERT_ASSIGN_OR_RETURN(prefill_input_buffers[key], value.Duplicate());
    }
    for (const auto& [key, value] : prefill_output_kv_cache_slice_buffers) {
      LITERT_ASSIGN_OR_RETURN(prefill_input_buffers[key], value.Duplicate());
    }
    prefill_input_buffers[CacheUpdateSignatures::kInputPos] =
        std::move(prefill_input_pos);
  }
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      prefill_output_buffers;
  {
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
  }
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      decode_output_buffers;
  {
    for (const auto& [key, value] : input_kv_cache_buffers) {
      LITERT_ASSIGN_OR_RETURN(decode_output_buffers[key], value.Duplicate());
    }
  }
  return InferenceContext(
      std::move(prefill_input_buffers), std::move(prefill_output_buffers),
      std::move(decode_input_buffers), std::move(decode_output_buffers));
}

absl::Status LlmLiteRtNpuCompiledModelExecutor::WarmupInference(
    ::litert::CompiledModel& compiled_model_llm,
    InferenceContext& llm_inference_context,
    ::litert::CompiledModel& compiled_model_auxiliary,
    const InferenceContext& rope_inference_context,
    const InferenceContext& mask_inference_context,
    const InferenceContext& cache_update_inference_context) {
  // We need to fill the embedding input buffers with non-zero values because
  // some of the Gemma3 models contain embedding lookup preprocessing that
  // quantize a float embedding tensor into a quantized embedding tensor and use
  // 'DIV' operations in the process. Without this we risk running into: ERROR:
  // third_party/tensorflow/lite/kernels/div.cc:242 data[i] != 0 was not true.
  // ERROR: Node number 21 (DIV) failed to invoke.

  if (llm_inference_context.decode_input_buffers.contains(
          LlmSignatures::kInputEmbeddings)) {
    RETURN_IF_ERROR(
        Fill(llm_inference_context
                 .decode_input_buffers[LlmSignatures::kInputEmbeddings],
             1));
  }
  if (llm_inference_context.prefill_input_buffers.contains(
          LlmSignatures::kInputEmbeddings)) {
    RETURN_IF_ERROR(
        Fill(llm_inference_context
                 .prefill_input_buffers[LlmSignatures::kInputEmbeddings],
             1));
  }
  auto result = compiled_model_llm.Run(
      LlmSignatures::kPrefillLlm, llm_inference_context.prefill_input_buffers,
      llm_inference_context.prefill_output_buffers);
  RET_CHECK(result) << "Inference warmup run for Gemma3 (prefill) failed."
                    << result.Error().Message();
  result = compiled_model_llm.Run(LlmSignatures::kDecodeLlm,
                                  llm_inference_context.decode_input_buffers,
                                  llm_inference_context.decode_output_buffers);
  RET_CHECK(result) << "Inference warmup run for Gemma3 (decode) failed."
                    << result.Error().Message();

  result = compiled_model_auxiliary.Run(
      RopeSignatures::kPrefillRope,
      rope_inference_context.prefill_input_buffers,
      rope_inference_context.prefill_output_buffers);
  RET_CHECK(result)
      << "Inference warmup run for RoPE signature (prefill) failed."
      << result.Error().Message();
  result = compiled_model_auxiliary.Run(
      RopeSignatures::kDecodeRope, rope_inference_context.decode_input_buffers,
      rope_inference_context.decode_output_buffers);
  RET_CHECK(result)
      << "Inference warmup run for RoPE signature (decode) failed."
      << result.Error().Message();

  result = compiled_model_auxiliary.Run(
      MaskSignatures::kPrefillMask,
      mask_inference_context.prefill_input_buffers,
      mask_inference_context.prefill_output_buffers);
  RET_CHECK(result)
      << "Inference warmup run for mask signature (prefill) failed."
      << result.Error().Message();
  result = compiled_model_auxiliary.Run(
      MaskSignatures::kDecodeMask, mask_inference_context.decode_input_buffers,
      mask_inference_context.decode_output_buffers);
  RET_CHECK(result)
      << "Inference warmup run for mask signature (decode) failed."
      << result.Error().Message();

  result = compiled_model_auxiliary.Run(
      CacheUpdateSignatures::kPrefillCacheUpdate,
      cache_update_inference_context.prefill_input_buffers,
      cache_update_inference_context.prefill_output_buffers);
  RET_CHECK(result)
      << "Inference warmup run for cache update signature (prefill) failed."
      << result.Error().Message();
  result = compiled_model_auxiliary.Run(
      CacheUpdateSignatures::kDecodeCacheUpdate,
      cache_update_inference_context.decode_input_buffers,
      cache_update_inference_context.decode_output_buffers);
  RET_CHECK(result)
      << "Inference warmup run for cache update signature (decode) failed."
      << result.Error().Message();

  // Clear the KV cache buffers after warmup.
  RETURN_IF_ERROR(ClearKVCache(llm_inference_context.prefill_input_buffers));
  return absl::OkStatus();
}

LlmLiteRtNpuCompiledModelExecutor::InferenceContext::InferenceContext(
    absl::flat_hash_map<absl::string_view, TensorBuffer> prefill_input_buffers,
    absl::flat_hash_map<absl::string_view, TensorBuffer> prefill_output_buffers,
    absl::flat_hash_map<absl::string_view, TensorBuffer> decode_input_buffers,
    absl::flat_hash_map<absl::string_view, TensorBuffer> decode_output_buffers)
    : prefill_input_buffers(std::move(prefill_input_buffers)),
      prefill_output_buffers(std::move(prefill_output_buffers)),
      decode_input_buffers(std::move(decode_input_buffers)),
      decode_output_buffers(std::move(decode_output_buffers)) {}

LlmLiteRtNpuCompiledModelExecutor::EmbedderContext::EmbedderContext(
    CompiledModel embedder_compiled_model,
    absl::flat_hash_map<absl::string_view, TensorBuffer> prefill_input_buffers,
    absl::flat_hash_map<absl::string_view, TensorBuffer> prefill_output_buffers,
    absl::flat_hash_map<absl::string_view, TensorBuffer> decode_input_buffers,
    absl::flat_hash_map<absl::string_view, TensorBuffer> decode_output_buffers)
    : embedder_compiled_model(std::move(embedder_compiled_model)),
      inference_context(
          std::move(prefill_input_buffers), std::move(prefill_output_buffers),
          std::move(decode_input_buffers), std::move(decode_output_buffers)) {}

LlmLiteRtNpuCompiledModelExecutor::NpuAuxiliaryContext::NpuAuxiliaryContext(
    CompiledModel npu_auxiliary_compiled_model)
    : npu_auxiliary_compiled_model(std::move(npu_auxiliary_compiled_model)) {}

absl::Status LlmLiteRtNpuCompiledModelExecutor::Prefill(
    const ExecutorInputs& inputs) {
  return Prefill(inputs, ExecutorPrefillParams());
}

absl::Status LlmLiteRtNpuCompiledModelExecutor::Prefill(
    const ExecutorInputs& inputs, const ExecutorPrefillParams& params) {
  auto start = absl::Now();
  LITERT_ASSIGN_OR_RETURN(auto tensor_type,
                          (*inputs.GetTextTokenIdsPtr())->TensorType());
  // Only accept batch size 1 for now.
  RET_CHECK_EQ(tensor_type.Layout().Dimensions()[0], 1);
  RET_CHECK_GT(tensor_type.Layout().Dimensions()[1], 0)
      << "Prefill token ids must be non-empty.";
  if (UseEmbeddingLookupManager()) {
    RETURN_IF_ERROR(
        embedding_lookup_manager_.value()->UpdateMultiModalEmbeddings(inputs));
  }
  LITERT_ASSIGN_OR_RETURN(auto ids, ReferTensorBufferAsSpan<int32_t>(
                                        *(*inputs.GetTextTokenIdsPtr())));

  ASSIGN_OR_RETURN(auto work_groups, GetOptimizedPrefillWorkGroups(
                                         prefill_signature_map_, ids.size()));
  for (const auto& [prefill_signature, prefill_length] : work_groups) {
    RETURN_IF_ERROR(PrefillInternal(prefill_signature,
                                    ids.subspan(/*pos=*/0, prefill_length)));
    ids = ids.subspan(/*pos=*/prefill_length);
    latency_stats_.prefill_num_tokens += kPrefillSize;
  }
  RET_CHECK_EQ(ids.size(), 0).SetCode(absl::StatusCode::kInternal)
      << "Work groups not covering the entire prefill input.";

  if (UseEmbeddingLookupManager()) {
    RETURN_IF_ERROR(
        embedding_lookup_manager_.value()->CleanupMultiModalEmbeddings());
  }
  auto end = absl::Now();
  latency_stats_.prefill_e2e_latency_us +=
      absl::ToInt64Microseconds(end - start);

  return absl::OkStatus();
}

absl::StatusOr<std::vector<std::vector<int>>>
LlmLiteRtNpuCompiledModelExecutor::Decode() {
  return Decode(ExecutorDecodeParams());
}

absl::StatusOr<std::vector<std::vector<int>>>
LlmLiteRtNpuCompiledModelExecutor::Decode(
    const ExecutorDecodeParams& decode_params) {
  if (decode_params.HasConstraintDecoder()) {
    return absl::UnimplementedError(
        "Constrained decoding is not supported on NPU.");
  }
  auto start = absl::Now();
  ::litert::TensorBuffer& decoded_logits =
      llm_inference_context_
          .decode_output_buffers[LlmSignatures::kDecodeLogitsOutput];

  if (processed_tokens_.TokenCount() != current_step_) {
    RETURN_IF_ERROR(processed_tokens_.RollBackToStep(current_step_));
  }

  // We must have a pending input token to decode that's either coming from
  // the previous prefill or decode.
  auto [internal_start_step, pending_input_token] =
      processed_tokens_.GetNextUnprocessedToken();
  if (pending_input_token.empty()) {
    return absl::InvalidArgumentError("No id available to be decoded.");
  }
  RETURN_IF_ERROR(DecodeInternal(internal_start_step, pending_input_token[0]));
  RETURN_IF_ERROR(processed_tokens_.MarkPendingInputTokenAsProcessed());

  auto start_sample = absl::Now();
  ASSIGN_OR_RETURN(const int max_index, ApplyGreedySampling(decoded_logits));

  latency_stats_.decode_sampling_latency_us +=
      absl::ToInt64Microseconds(absl::Now() - start_sample);

  // Store the sampled id as the pending input token for next Decode.

  std::shared_ptr<TokenData> last_output_token =
      std::make_shared<TokenData>(max_index);

  if (UseEmbeddingLookupManager()) {
    RETURN_IF_ERROR(embedding_lookup_manager_.value()->LookupDecode(
        last_output_token->id(), last_output_token->mutable_embedding()));
  }
  // For Gemma3 we don't need to do anything here because we invoke
  // the Embedder before invoking the transformer during prefill/decode. All
  // we need to do is keep the token id around (which is stored as the pending
  // token).

  RETURN_IF_ERROR(
      processed_tokens_.AddPendingInputToken({std::move(last_output_token)}));
  ++current_step_;

  auto end = absl::Now();
  latency_stats_.decode_e2e_latency_us +=
      absl::ToInt64Microseconds(end - start);
  latency_stats_.decode_num_tokens += 1;

  return std::vector<std::vector<int>>{{max_index}};
}

// Prefill internal implementation, for one prefill call to the compiled model
// with a certain length.
absl::Status LlmLiteRtNpuCompiledModelExecutor::PrefillInternal(
    absl::string_view prefill_signature, absl::Span<const int> ids) {
  auto start_prepare_inputs = absl::Now();
  {
    // Prefill input tokens.
    LITERT_ASSIGN_OR_RETURN(
        auto prefill_input_size,
        embedder_context_.inference_context
            .prefill_input_buffers[EmbedderSignatures::kEmbedderInput]
            .Size());
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

    memset(prefill_input_ptr, 0, prefill_input_size);
    memset(prefill_input_pos_ptr, 0, prefill_input_pos_size);
    memset(prefill_timestep_ptr, 0, prefill_timestep_size);

    if (processed_tokens_.TokenCount() != current_step_) {
      RETURN_IF_ERROR(processed_tokens_.RollBackToStep(current_step_));
    }
    // Check if have a pending input token. Note that 'internal_start_step' is
    // always equal to the number of processed tokens plus 1.
    auto [internal_start_step, pending_input_token] =
        processed_tokens_.GetNextUnprocessedToken();
    int input_idx = 0;
    if (!pending_input_token.empty()) {
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
      RETURN_IF_ERROR(processed_tokens_.MarkPendingInputTokenAsProcessed());
      ++input_idx;
    }

    prefill_timestep_ptr[0] = internal_start_step;
    std::vector<int> processed_input_tokens;
    // We will not fill the last token of the current input into the compiled
    // model input buffers just yet. It will be stored in the
    // 'processed_tokens_' and used in the next prefill or decode.
    processed_input_tokens.reserve(ids.size() - 1);
    for (int i = 0; i < ids.size() - 1; input_idx++, current_step_++, i++) {
      prefill_input_ptr[input_idx] = ids[i];
      prefill_input_pos_ptr[input_idx] = current_step_;
      processed_input_tokens.push_back(ids[i]);
    }
    processed_tokens_.AddProcessedTokens(processed_input_tokens);

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
      RETURN_IF_ERROR(embedding_lookup_manager_.value()->LookupPrefill(
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
    RETURN_IF_ERROR(embedding_lookup_manager_.value()->LookupPrefill(
        last_input_token->id(), last_input_token->mutable_embedding()));
    latency_stats_.prefill_embedder_inference_latency_us +=
        absl::ToInt64Microseconds(absl::Now() - start);
  }

  // Add the last input token to the pending input token list.
  RETURN_IF_ERROR(
      processed_tokens_.AddPendingInputToken({std::move(last_input_token)}));
  ++current_step_;

  if (!UseEmbeddingLookupManager()) {
    // Invoke embedder signature for Gemma3, because we don't have the
    // embedding lookup manager to do it for us.
    auto start = absl::Now();
    auto res = embedder_context_.embedder_compiled_model.Run(
        EmbedderSignatures::kPrefillEmbedder,
        embedder_context_.inference_context.prefill_input_buffers,
        embedder_context_.inference_context.prefill_output_buffers);
    RET_CHECK(res) << "Failed to run embedder model." << res.Error().Message();
    auto end = absl::Now();
    latency_stats_.prefill_embedder_inference_latency_us +=
        absl::ToInt64Microseconds(end - start);
  }

  // Invoke embedder per layer signature if it exists.
  if (embedder_per_layer_context_.has_value()) {
    auto start = absl::Now();
    auto res =
        embedder_per_layer_context_->embedder_per_layer_compiled_model.Run(
            EmbedderPerLayerSignatures::kPrefillEmbedderPerLayer,
            embedder_per_layer_context_->inference_context
                .prefill_input_buffers,
            embedder_per_layer_context_->inference_context
                .prefill_output_buffers);
    RET_CHECK(res) << "Failed to run embedder per layer model."
                   << res.Error().Message();
    latency_stats_.prefill_embedder_per_layer_inference_latency_us.value() +=
        absl::ToInt64Microseconds(absl::Now() - start);
  }

  // Invoke RoPE signature.
  {
    auto start = absl::Now();
    auto res = npu_auxiliary_context_.npu_auxiliary_compiled_model.Run(
        RopeSignatures::kPrefillRope, rope_context_.prefill_input_buffers,
        rope_context_.prefill_output_buffers);
    RET_CHECK(res) << "Failed to run RoPE model." << res.Error().Message();
    auto end = absl::Now();
    latency_stats_.prefill_rope_inference_latency_us +=
        absl::ToInt64Microseconds(end - start);
  }

  // Invoke mask signature.
  {
    auto start = absl::Now();
    auto res = npu_auxiliary_context_.npu_auxiliary_compiled_model.Run(
        MaskSignatures::kPrefillMask, mask_context_.prefill_input_buffers,
        mask_context_.prefill_output_buffers);
    RET_CHECK(res) << "Failed to run compiled model." << res.Error().Message();
    auto end = absl::Now();
    latency_stats_.prefill_mask_inference_latency_us +=
        absl::ToInt64Microseconds(end - start);
  }

  // Invoke LLM signature.
  {
    auto start = absl::Now();
    auto res =
        llm_compiled_model_.Run(LlmSignatures::kPrefillLlm,
                                llm_inference_context_.prefill_input_buffers,
                                llm_inference_context_.prefill_output_buffers);
    RET_CHECK(res) << "Failed to run LLM model." << res.Error().Message();
    auto end = absl::Now();
    latency_stats_.prefill_llm_inference_latency_us +=
        absl::ToInt64Microseconds(end - start);
  }

  // Cache update.
  {
    auto start = absl::Now();
    auto res = npu_auxiliary_context_.npu_auxiliary_compiled_model.Run(
        CacheUpdateSignatures::kPrefillCacheUpdate,
        cache_update_inference_context_.prefill_input_buffers,
        cache_update_inference_context_.prefill_output_buffers);
    auto end = absl::Now();
    latency_stats_.prefill_cache_update_inference_latency_us +=
        absl::ToInt64Microseconds(end - start);
    RET_CHECK(res) << "Failed to run cache update model."
                   << res.Error().Message();
  }
  return absl::OkStatus();
}

absl::Status LlmLiteRtNpuCompiledModelExecutor::DecodeInternal(
    int step, std::shared_ptr<TokenData> token) {
  auto start_prepare_inputs = absl::Now();

  int id = token->id();
  if (id == -1) {
    return absl::InvalidArgumentError("No id available to be decoded.");
  }

  {
    // Decode input tokens.
    LITERT_ASSIGN_OR_RETURN(
        auto decode_input_lock_and_addr,
        ::litert::TensorBufferScopedLock::Create(
            embedder_context_.inference_context
                .decode_input_buffers[EmbedderSignatures::kEmbedderInput],
            ::litert::TensorBuffer::LockMode::kWrite));
    auto* decode_input_ptr =
        static_cast<int32_t*>(decode_input_lock_and_addr.second);
    decode_input_ptr[0] = id;

    // Decode input position
    LITERT_ASSIGN_OR_RETURN(
        auto decode_input_pos_lock_and_addr,
        ::litert::TensorBufferScopedLock::Create(
            rope_context_.decode_input_buffers[RopeSignatures::kInputPos],
            ::litert::TensorBuffer::LockMode::kWrite));
    auto* decode_input_pos_ptr =
        static_cast<int32_t*>(decode_input_pos_lock_and_addr.second);
    decode_input_pos_ptr[0] = step;

    // Timestep input.
    LITERT_ASSIGN_OR_RETURN(
        auto decode_timestep_lock_and_addr,
        ::litert::TensorBufferScopedLock::Create(
            mask_context_
                .decode_input_buffers[MaskSignatures::kMaskInputTimeStep],
            ::litert::TensorBuffer::LockMode::kWrite));
    auto* decode_timestep_ptr =
        static_cast<int32_t*>(decode_timestep_lock_and_addr.second);
    decode_timestep_ptr[0] = step;
  }
  auto end_prepare_inputs = absl::Now();
  latency_stats_.decode_prepare_input_latency_us +=
      absl::ToInt64Microseconds(end_prepare_inputs - start_prepare_inputs);

  if (!UseEmbeddingLookupManager()) {
    // Invoke embedder signature for Gemma3, because we don't have the embedding
    // lookup manager to do it for us.
    {
      auto start = absl::Now();
      auto res = embedder_context_.embedder_compiled_model.Run(
          EmbedderSignatures::kDecodeEmbedder,
          embedder_context_.inference_context.decode_input_buffers,
          embedder_context_.inference_context.decode_output_buffers);
      RET_CHECK(res) << "Failed to run embedder model."
                     << res.Error().Message();
      auto end = absl::Now();
      latency_stats_.decode_embedder_inference_latency_us +=
          absl::ToInt64Microseconds(end - start);
    }
  }

  if (UseEmbeddingLookupManager()) {
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
    memcpy(transformer_embedding_buffer_ptr, token->embedding().data(),
           token->embedding().size() * sizeof(float));
    latency_stats_.decode_embedder_inference_latency_us +=
        absl::ToInt64Microseconds(absl::Now() - start);
  }

  {
    if (embedder_per_layer_context_.has_value()) {
      auto start = absl::Now();
      auto res =
          embedder_per_layer_context_->embedder_per_layer_compiled_model.Run(
              EmbedderPerLayerSignatures::kDecodeEmbedderPerLayer,
              embedder_per_layer_context_->inference_context
                  .decode_input_buffers,
              embedder_per_layer_context_->inference_context
                  .decode_output_buffers);
      RET_CHECK(res) << "Failed to run embedder per layer model."
                     << res.Error().Message();
      latency_stats_.decode_embedder_per_layer_inference_latency_us.value() +=
          absl::ToInt64Microseconds(absl::Now() - start);
    }
  }

  // Invoke RoPE signature.
  {
    auto start = absl::Now();
    auto res = npu_auxiliary_context_.npu_auxiliary_compiled_model.Run(
        RopeSignatures::kDecodeRope, rope_context_.decode_input_buffers,
        rope_context_.decode_output_buffers);
    RET_CHECK(res) << "Failed to run RoPE model." << res.Error().Message();
    auto end = absl::Now();
    latency_stats_.decode_rope_inference_latency_us +=
        absl::ToInt64Microseconds(end - start);
  }

  // Invoke mask signature.
  {
    auto start = absl::Now();
    auto res = npu_auxiliary_context_.npu_auxiliary_compiled_model.Run(
        MaskSignatures::kDecodeMask, mask_context_.decode_input_buffers,
        mask_context_.decode_output_buffers);
    RET_CHECK(res) << "Failed to run compiled model." << res.Error().Message();
    auto end = absl::Now();
    latency_stats_.decode_mask_inference_latency_us +=
        absl::ToInt64Microseconds(end - start);
  }

  // Invoke LLM signature.
  {
    auto start = absl::Now();
    auto res = llm_compiled_model_.Run(
        LlmSignatures::kDecodeLlm, llm_inference_context_.decode_input_buffers,
        llm_inference_context_.decode_output_buffers);
    auto end = absl::Now();
    latency_stats_.decode_llm_inference_latency_us +=
        absl::ToInt64Microseconds(end - start);
    RET_CHECK(res) << "Failed to run LLM model." << res.Error().Message();
  }

  // Cache update.
  {
    auto start = absl::Now();
    auto res = npu_auxiliary_context_.npu_auxiliary_compiled_model.Run(
        CacheUpdateSignatures::kDecodeCacheUpdate,
        cache_update_inference_context_.decode_input_buffers,
        cache_update_inference_context_.decode_output_buffers);
    RET_CHECK(res) << "Failed to run cache update model."
                   << res.Error().Message();
    auto end = absl::Now();
    latency_stats_.decode_cache_update_inference_latency_us +=
        absl::ToInt64Microseconds(end - start);
  }
  return absl::OkStatus();
}

absl::StatusOr<int> LlmLiteRtNpuCompiledModelExecutor::GetVocabSize() {
  LITERT_ASSIGN_OR_RETURN(
      auto logits_tensor_type,
      llm_inference_context_
          .decode_output_buffers[LlmSignatures::kDecodeLogitsOutput]
          .TensorType());
  return logits_tensor_type.Layout().Dimensions()[2];
}

LlmLiteRtNpuCompiledModelExecutor::LatencyStats
LlmLiteRtNpuCompiledModelExecutor::GetLatencyStats() const {
  return latency_stats_;
}

absl::Status LlmLiteRtNpuCompiledModelExecutor::Reset() {
  PrintLatencyStats(GetLatencyStats());
  current_step_ = 0;
  RETURN_IF_ERROR(processed_tokens_.RollBackToStep(0));
  sampled_ids_.clear();
  latency_stats_ = {};

  RETURN_IF_ERROR(ClearKVCache(llm_inference_context_.prefill_input_buffers));
  return absl::OkStatus();
}

// static
absl::StatusOr<std::unique_ptr<LlmLiteRtNpuCompiledModelExecutor>>
LlmLiteRtNpuCompiledModelExecutor::Create(
    const LlmExecutorSettings& executor_settings, ModelResources& resources,
    Environment& env) {
  ASSIGN_OR_RETURN(const litert::Model* llm_model,
                   resources.GetTFLiteModel(ModelType::kTfLitePrefillDecode));

  // For the lack of a better way to identify the model variants, we use the
  // presence of per-layer embeddings as the signal for Gemma3n.
  LITERT_ASSIGN_OR_RETURN(bool has_per_layer_embeddings,
                          HasPerLayerEmbedder(*llm_model));
  const bool model_has_per_layer_embedding = has_per_layer_embeddings;
  if (model_has_per_layer_embedding) {
    return CreateForModelHasPerLayerEmbedding(executor_settings, resources, env,
                                              llm_model);
  } else {
    return CreateForModelWithoutPerLayerEmbedding(executor_settings, resources,
                                                  env, llm_model);
  }
};

// Creates LiteRT options for NPU accelerator.
litert::Expected<litert::Options> CreateLiteRtOptions() {
  LITERT_ASSIGN_OR_RETURN(auto options, ::litert::Options::Create());
  options.SetHardwareAccelerators(litert::HwAccelerators::kCpu);
  LITERT_ASSIGN_OR_RETURN(auto& qnn_opts,
                          options.GetQualcommOptions());
  qnn_opts.SetLogLevel(::litert::qualcomm::QualcommOptions::LogLevel::kOff);
  qnn_opts.SetHtpPerformanceMode(
      ::litert::qualcomm::QualcommOptions::HtpPerformanceMode::kBurst);
  return options;
}

absl::StatusOr<std::unique_ptr<LlmLiteRtNpuCompiledModelExecutor>>
LlmLiteRtNpuCompiledModelExecutor::CreateForModelHasPerLayerEmbedding(
    const LlmExecutorSettings& executor_settings, ModelResources& resources,
    litert::Environment& env, const litert::Model* transformer_model) {
  // If the model is fully AOT compiled for NPU, NPU accelerator is used
  // automatically.
  // Set up LiteRt options.
  LITERT_ASSIGN_OR_RETURN(auto options, CreateLiteRtOptions());
  LITERT_ASSIGN_OR_RETURN(
      CompiledModel llm_compiled_model,
      CompiledModel::Create(env, transformer_model->Get(), options));

  // Allocate all input and output buffers of the LLM model that are meant to be
  // used by the NPU chip first, so that we can later duplicate the buffers into
  // the output buffer maps of the embedder, mask, and rope signatures.

  absl::flat_hash_map<absl::string_view, TensorBuffer>
      gemma_prefill_input_buffers;
  absl::flat_hash_map<absl::string_view, TensorBuffer>
      gemma_decode_input_buffers;
  absl::flat_hash_map<absl::string_view, TensorBuffer> input_kv_cache_buffers;
  absl::flat_hash_map<absl::string_view, TensorBuffer>
      prefill_output_kv_cache_slice_buffers;
  absl::flat_hash_map<absl::string_view, TensorBuffer>
      decode_output_kv_cache_slice_buffers;

  absl::Status allocate_status = AllocateTransformerBuffers(
      env, transformer_model, llm_compiled_model, gemma_prefill_input_buffers,
      gemma_decode_input_buffers, input_kv_cache_buffers,
      prefill_output_kv_cache_slice_buffers,
      decode_output_kv_cache_slice_buffers);
  if (!allocate_status.ok()) {
    return allocate_status;
  }

  // Gemma3n specific fix: KV cache buffer 19 of *prefill* is not connected
  // to any OPs in the model, making the LiteRT runtime allocate host memory
  // for it. This is incompatible when running the transformer model on the NPU.
  if (input_kv_cache_buffers.contains(cache_k19)) {
    LITERT_ASSIGN_OR_RETURN(
        input_kv_cache_buffers[cache_k19],
        llm_compiled_model.CreateInputBuffer(kDecodeSignature, cache_k19));
    input_kv_cache_buffers[cache_k19].Clear();
    LITERT_ASSIGN_OR_RETURN(
        input_kv_cache_buffers[cache_v19],
        llm_compiled_model.CreateInputBuffer(kDecodeSignature, cache_v19));
    input_kv_cache_buffers[cache_v19].Clear();
  }
  ASSIGN_OR_RETURN(
      auto llm_inference_context,
      CreateLlmInferenceContextWithBufferSharing(
          env, llm_compiled_model, input_kv_cache_buffers,
          prefill_output_kv_cache_slice_buffers,
          decode_output_kv_cache_slice_buffers, gemma_prefill_input_buffers,
          gemma_decode_input_buffers));

  ASSIGN_OR_RETURN(auto npu_auxiliary_lrt_model,
                   resources.GetTFLiteModel(ModelType::kTfLiteAux));

  ASSIGN_OR_RETURN(auto npu_auxiliary_context,
                   CreateNpuAuxiliaryContext(env, *npu_auxiliary_lrt_model));

  ASSIGN_OR_RETURN(auto mask_context,
                   CreateMaskContextWithBufferSharing(
                       npu_auxiliary_context, gemma_prefill_input_buffers,
                       gemma_decode_input_buffers));

  ASSIGN_OR_RETURN(auto embedder_lrt_model,
                   resources.GetTFLiteModel(ModelType::kTfLiteEmbedder));
  ASSIGN_OR_RETURN(
      auto embedder_context,
      CreateEmbedderContextWithBufferSharing(
          env, *embedder_lrt_model,
          mask_context.prefill_input_buffers[MaskSignatures::kMaskInputTokens],
          mask_context.decode_input_buffers[MaskSignatures::kMaskInputTokens],
          gemma_prefill_input_buffers, gemma_decode_input_buffers));

  ASSIGN_OR_RETURN(auto rope_context,
                   CreateRopeContextWithBufferSharing(
                       npu_auxiliary_context, gemma_prefill_input_buffers,
                       gemma_decode_input_buffers));

  // Duplicate the rope's buffers that are used to store the prefill and
  // decode input position, because they will need to be passed to the
  // cache update inference context as well.
  LITERT_ASSIGN_OR_RETURN(
      ::litert::TensorBuffer prefill_input_pos,
      rope_context.prefill_input_buffers[RopeSignatures::kInputPos]
          .Duplicate());
  LITERT_ASSIGN_OR_RETURN(
      ::litert::TensorBuffer decode_input_pos,
      rope_context.decode_input_buffers[RopeSignatures::kInputPos].Duplicate());
  ASSIGN_OR_RETURN(
      auto cache_update_inference_context,
      CreateCacheUpdateInferenceContextWithBufferSharing(
          input_kv_cache_buffers, prefill_output_kv_cache_slice_buffers,
          decode_output_kv_cache_slice_buffers, std::move(prefill_input_pos),
          std::move(decode_input_pos)));

  RETURN_IF_ERROR(WarmupInference(
      llm_compiled_model, llm_inference_context,
      npu_auxiliary_context.npu_auxiliary_compiled_model, rope_context,
      mask_context, cache_update_inference_context));

  // For now we only support one prefill length in the model.
  SortedPrefillSignatureMap prefill_runner_set;
  prefill_runner_set[kPrefillSize] = kPrefillSignature;

  absl::flat_hash_map<int, const Model*> end_of_multi_modal_embedding_models;
  absl::StatusOr<const litert::Model*> maybe_end_of_audio_model =
      resources.GetTFLiteModel(ModelType::kTfLiteEndOfAudio);
  if (maybe_end_of_audio_model.ok()) {
    end_of_multi_modal_embedding_models
        [litert::lm::ExecutorAudioData::kEndToken] =
            maybe_end_of_audio_model.value();
  }
  absl::StatusOr<const litert::Model*> maybe_end_of_vision_model =
      resources.GetTFLiteModel(ModelType::kTfLiteEndOfVision);
  if (maybe_end_of_vision_model.ok()) {
    end_of_multi_modal_embedding_models
        [litert::lm::ExecutorVisionData::kEndToken] =
            maybe_end_of_vision_model.value();
  }
  ASSIGN_OR_RETURN(
      std::unique_ptr<EmbeddingLookupManager> embedding_lookup_manager,
      EmbeddingLookupManager::Create(embedder_lrt_model,
                                     end_of_multi_modal_embedding_models, true,
                                     "decode_embedder"));

  std::optional<EmbedderPerLayerContext> embedder_per_layer_context =
      std::nullopt;

  ASSIGN_OR_RETURN(
      const litert::Model* embedder_per_layer_model,
      resources.GetTFLiteModel(ModelType::kTfLitePerLayerEmbedder));
  ASSIGN_OR_RETURN(
      embedder_per_layer_context,
      CreateEmbedderPerLayerContextWithBufferSharing(
          env, *embedder_per_layer_model,
          mask_context.prefill_input_buffers[MaskSignatures::kMaskInputTokens],
          mask_context.decode_input_buffers[MaskSignatures::kMaskInputTokens],
          gemma_prefill_input_buffers, gemma_decode_input_buffers));

  auto executor = absl::WrapUnique(new LlmLiteRtNpuCompiledModelExecutor(
      executor_settings, env, std::move(embedder_context),
      std::move(npu_auxiliary_context), std::move(mask_context),
      std::move(rope_context), std::move(llm_compiled_model),
      std::move(llm_inference_context),
      std::move(cache_update_inference_context), std::move(prefill_runner_set),
      std::move(embedding_lookup_manager),
      std::move(embedder_per_layer_context)));
  return executor;
}

absl::StatusOr<std::unique_ptr<LlmLiteRtNpuCompiledModelExecutor>>
LlmLiteRtNpuCompiledModelExecutor::CreateForModelWithoutPerLayerEmbedding(
    const LlmExecutorSettings& executor_settings, ModelResources& resources,
    litert::Environment& env, const litert::Model* transformer_model) {
  // If the model is fully AOT compiled for NPU, NPU accelerator is used
  // automatically.
  LITERT_ASSIGN_OR_RETURN(auto options, CreateLiteRtOptions());
  LITERT_ASSIGN_OR_RETURN(
      CompiledModel llm_compiled_model,
      CompiledModel::Create(env, transformer_model->Get(), options));

  // Allocate all input and output buffers of the LLM model that are meant to be
  // used by the NPU chip first, so that we can later duplicate the buffers into
  // the output buffer maps of the embedder, mask, and rope signatures.

  absl::flat_hash_map<absl::string_view, TensorBuffer>
      gemma_prefill_input_buffers;
  absl::flat_hash_map<absl::string_view, TensorBuffer>
      gemma_decode_input_buffers;
  absl::flat_hash_map<absl::string_view, TensorBuffer> input_kv_cache_buffers;
  absl::flat_hash_map<absl::string_view, TensorBuffer>
      prefill_output_kv_cache_slice_buffers;
  absl::flat_hash_map<absl::string_view, TensorBuffer>
      decode_output_kv_cache_slice_buffers;

  absl::Status allocate_status = AllocateTransformerBuffers(
      env, transformer_model, llm_compiled_model, gemma_prefill_input_buffers,
      gemma_decode_input_buffers, input_kv_cache_buffers,
      prefill_output_kv_cache_slice_buffers,
      decode_output_kv_cache_slice_buffers);
  if (!allocate_status.ok()) {
    return allocate_status;
  }
  ASSIGN_OR_RETURN(
      auto llm_inference_context,
      CreateLlmInferenceContextWithBufferSharing(
          env, llm_compiled_model, input_kv_cache_buffers,
          prefill_output_kv_cache_slice_buffers,
          decode_output_kv_cache_slice_buffers, gemma_prefill_input_buffers,
          gemma_decode_input_buffers));

  // Gemma3 specific fix:
  //
  // TODO(b/416702118): Buffers kv_cache_{k,v}_25 have float element type for
  // the prefill signature but int16_t for the decode signature. Therefore,
  // unlike for the other KV cache tensors, we can not re-use the same tensor
  // during prefill and decode (because trying to register a tensor of element
  // type float for the decode signature that expects it in int16_t will
  // fail). Luckily these buffers are not used, so we can simply create new
  // ones to satisfy the compiled model run API.  We can remove this
  // workaround once we have a model that removes these buffers.
  if (llm_inference_context.prefill_input_buffers.contains(cache_k25)) {
    LITERT_ASSIGN_OR_RETURN(
        llm_inference_context.decode_input_buffers[cache_k25],
        llm_compiled_model.CreateInputBuffer(kDecodeSignature, cache_k25));
    LITERT_ASSIGN_OR_RETURN(
        llm_inference_context.decode_input_buffers[cache_v25],
        llm_compiled_model.CreateInputBuffer(kDecodeSignature, cache_v25));
  } else if (llm_inference_context.prefill_input_buffers.contains(cache_k23)) {
    // Fast VLM model specific fix:
    LITERT_ASSIGN_OR_RETURN(
        llm_inference_context.decode_input_buffers[cache_k23],
        llm_compiled_model.CreateInputBuffer(kDecodeSignature, cache_k23));
    LITERT_ASSIGN_OR_RETURN(
        llm_inference_context.decode_input_buffers[cache_v23],
        llm_compiled_model.CreateInputBuffer(kDecodeSignature, cache_v23));
  } else {
    // Tiny Gemma 270M specific fix:
    LITERT_ASSIGN_OR_RETURN(
        llm_inference_context.decode_input_buffers[cache_k17],
        llm_compiled_model.CreateInputBuffer(kDecodeSignature, cache_k17));
    LITERT_ASSIGN_OR_RETURN(
        llm_inference_context.decode_input_buffers[cache_v17],
        llm_compiled_model.CreateInputBuffer(kDecodeSignature, cache_v17));
  }

  ASSIGN_OR_RETURN(auto npu_auxiliary_lrt_model,
                   resources.GetTFLiteModel(ModelType::kTfLiteAux));

  ASSIGN_OR_RETURN(auto npu_auxiliary_context,
                   CreateNpuAuxiliaryContext(env, *npu_auxiliary_lrt_model));

  ASSIGN_OR_RETURN(auto mask_context,
                   CreateMaskContextWithBufferSharing(
                       npu_auxiliary_context, gemma_prefill_input_buffers,
                       gemma_decode_input_buffers));

  ASSIGN_OR_RETURN(auto embedder_lrt_model,
                   resources.GetTFLiteModel(ModelType::kTfLiteEmbedder));
  ASSIGN_OR_RETURN(
      auto embedder_context,
      CreateEmbedderContextWithBufferSharing(
          env, *embedder_lrt_model,
          mask_context.prefill_input_buffers[MaskSignatures::kMaskInputTokens],
          mask_context.decode_input_buffers[MaskSignatures::kMaskInputTokens],
          gemma_prefill_input_buffers, gemma_decode_input_buffers));

  ASSIGN_OR_RETURN(auto rope_context,
                   CreateRopeContextWithBufferSharing(
                       npu_auxiliary_context, gemma_prefill_input_buffers,
                       gemma_decode_input_buffers));

  // Duplicate the rope's buffers that are used to store the prefill and
  // decode input position, because they will need to be passed to the
  // cache update inference context as well.
  LITERT_ASSIGN_OR_RETURN(
      ::litert::TensorBuffer prefill_input_pos,
      rope_context.prefill_input_buffers[RopeSignatures::kInputPos]
          .Duplicate());
  LITERT_ASSIGN_OR_RETURN(
      ::litert::TensorBuffer decode_input_pos,
      rope_context.decode_input_buffers[RopeSignatures::kInputPos].Duplicate());
  ASSIGN_OR_RETURN(
      auto cache_update_inference_context,
      CreateCacheUpdateInferenceContextWithBufferSharing(
          input_kv_cache_buffers, prefill_output_kv_cache_slice_buffers,
          decode_output_kv_cache_slice_buffers, std::move(prefill_input_pos),
          std::move(decode_input_pos)));

  RETURN_IF_ERROR(WarmupInference(
      llm_compiled_model, llm_inference_context,
      npu_auxiliary_context.npu_auxiliary_compiled_model, rope_context,
      mask_context, cache_update_inference_context));

  // For now we only support one prefill length in the model.
  SortedPrefillSignatureMap prefill_runner_set;
  prefill_runner_set[kPrefillSize] = kPrefillSignature;

  std::optional<EmbedderPerLayerContext> embedder_per_layer_context =
      std::nullopt;

  std::optional<std::unique_ptr<EmbeddingLookupManager>>
      maybe_embedding_lookup_manager = std::nullopt;
  // If the model has vision encoder, we need to create the embedding lookup
  // manager.
  if (resources.GetTFLiteModel(ModelType::kTfLiteVisionEncoder).ok()) {
    ASSIGN_OR_RETURN(maybe_embedding_lookup_manager,
                     EmbeddingLookupManager::Create(embedder_lrt_model, true,
                                                    "decode_embedder"));
  }

  auto executor = absl::WrapUnique(new LlmLiteRtNpuCompiledModelExecutor(
      executor_settings, env, std::move(embedder_context),
      std::move(npu_auxiliary_context), std::move(mask_context),
      std::move(rope_context), std::move(llm_compiled_model),
      std::move(llm_inference_context),
      std::move(cache_update_inference_context), std::move(prefill_runner_set),
      std::move(maybe_embedding_lookup_manager),
      /*embedder_per_layer_context=*/std::nullopt));
  return executor;
}

absl::Status LlmLiteRtNpuCompiledModelExecutor::ClearKVCache(
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>& buffers) {
  for (auto& [buffer_name, buffer] : buffers) {
    if (buffer_name.starts_with(kv_cache_k_root_name) ||
        buffer_name.starts_with(kv_cache_v_root_name)) {
      LITERT_RETURN_IF_ERROR(buffer.Clear());
    }
  }
  return absl::OkStatus();
}

}  // namespace litert::lm
