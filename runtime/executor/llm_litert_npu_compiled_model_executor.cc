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
#include <fstream>
#include <iomanip>
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
#include "absl/log/absl_vlog_is_on.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"  // from @litert
#include "litert/c/litert_model.h"  // from @litert
#include "litert/c/litert_model_types.h"  // from @litert
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
#include "tflite/types/half.h"  // from @litert
#if defined(__ANDROID__)
#include "litert/cc/options/litert_qualcomm_options.h"  // from @litert
#endif
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
using ::litert::Model;
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
absl::StatusOr<int> FindMaxIndex(const ::litert::TensorBuffer& decoded_logits) {
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

void PrintLatencyStats(
    const LlmLiteRtNpuCompiledModelExecutor::LatencyStats& latency_stats) {
  ABSL_LOG(INFO) << "LatencyStats: " << latency_stats.DebugString();
}

void DumpTensorBufferToFile(const std::string& label,
                            const TensorBuffer& buffer,
                            std::ofstream& log_file) {
  auto type_or = buffer.TensorType();
  if (!type_or.HasValue()) return;
  auto lock_or = ::litert::TensorBufferScopedLock::Create(
      const_cast<TensorBuffer&>(buffer),
      ::litert::TensorBuffer::LockMode::kRead);
  if (!lock_or.HasValue()) return;

  auto num_elements_or = type_or->Layout().NumElements();
  if (!num_elements_or.HasValue()) return;
  size_t num_elements = *num_elements_or;

  log_file << "LITERT_DUMP: " << label
           << " TYPE: " << (int)type_or->ElementType()
           << " SIZE: " << num_elements << " VALUES: [START] ";
  log_file << std::fixed << std::setprecision(6);

  if (type_or->ElementType() == ::litert::ElementType::Float32) {
    const float* ptr = static_cast<const float*>(lock_or->second);
    for (size_t i = 0; i < std::min(num_elements, (size_t)20); ++i)
      log_file << ptr[i] << " ";
  } else if (type_or->ElementType() == ::litert::ElementType::Int16) {
    const int16_t* ptr = static_cast<const int16_t*>(lock_or->second);
    for (size_t i = 0; i < std::min(num_elements, (size_t)20); ++i)
      log_file << (int)ptr[i] << " ";
  } else if (type_or->ElementType() == ::litert::ElementType::Int32) {
    const int32_t* ptr = static_cast<const int32_t*>(lock_or->second);
    for (size_t i = 0; i < std::min(num_elements, (size_t)20); ++i)
      log_file << ptr[i] << " ";
  } else if (type_or->ElementType() == ::litert::ElementType::UInt16) {
    const uint16_t* ptr = static_cast<const uint16_t*>(lock_or->second);
    for (size_t i = 0; i < std::min(num_elements, (size_t)20); ++i)
      log_file << (int)ptr[i] << " ";
  } else if (type_or->ElementType() == ::litert::ElementType::Int8) {
    const int8_t* ptr = static_cast<const int8_t*>(lock_or->second);
    for (size_t i = 0; i < std::min(num_elements, (size_t)20); ++i)
      log_file << (int)ptr[i] << " ";
  } else if (type_or->ElementType() == ::litert::ElementType::Float16 ||
             type_or->ElementType() == ::litert::ElementType::BFloat16) {
    const uint16_t* ptr = static_cast<const uint16_t*>(lock_or->second);
    for (size_t i = 0; i < std::min(num_elements, (size_t)20); ++i)
      log_file << "hex:" << std::hex << ptr[i] << std::dec << " ";
  } else {
    log_file << "UNSUPPORTED_TYPE";
  }
  log_file << "[END]\n";
}

void DumpBuffersToFile(
    const std::string& prefix,
    const absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        buffers,
    std::ofstream& log_file) {
  for (const auto& [name, buffer] : buffers) {
    DumpTensorBufferToFile(prefix + ":" + std::string(name), buffer, log_file);
  }
}

void NormalizeMask(const std::string& label, ::litert::TensorBuffer& buffer,
                   bool should_dump) {
  auto type_or = buffer.TensorType();
  if (!type_or.HasValue() ||
      type_or->ElementType() != ::litert::ElementType::Int8)
    return;

  auto lock_or = ::litert::TensorBufferScopedLock::Create(
      buffer, ::litert::TensorBuffer::LockMode::kWrite);
  if (!lock_or.HasValue()) return;

  auto num_elements_or = type_or->Layout().NumElements();
  if (!num_elements_or.HasValue()) return;

  int8_t* ptr = static_cast<int8_t*>(lock_or->second);
  bool modified = false;
  size_t n = *num_elements_or;
  for (size_t i = 0; i < n; ++i) {
    if (ptr[i] > 0 && ptr[i] != 127) {
      ptr[i] = 127;
      modified = true;
    }
  }
  if (modified) {
    if (should_dump) {
      ABSL_LOG(INFO) << "LITERT_DUMP: Mask " << label
                     << " normalized to 127 scale (modified values)";
    }
  } else {
    if (should_dump) {
      ABSL_LOG(INFO) << "LITERT_DUMP: Mask " << label
                     << " checked (no modification needed)";
    }
  }
}

absl::Status DequantizeLogits(const ::litert::TensorBuffer& src,
                              ::litert::TensorBuffer& dst, float scale,
                              int32_t zero_point, bool should_dump) {
  auto src_type_or = src.TensorType();
  RET_CHECK(src_type_or.HasValue());
  auto dst_type_or = dst.TensorType();
  RET_CHECK(dst_type_or.HasValue());
  RET_CHECK_EQ((int)dst_type_or->ElementType(),
               (int)::litert::ElementType::Float32);

  auto src_lock_or = ::litert::TensorBufferScopedLock::Create(
      const_cast<::litert::TensorBuffer&>(src),
      ::litert::TensorBuffer::LockMode::kRead);
  RET_CHECK(src_lock_or.HasValue());
  auto dst_lock_or = ::litert::TensorBufferScopedLock::Create(
      dst, ::litert::TensorBuffer::LockMode::kWrite);
  RET_CHECK(dst_lock_or.HasValue());

  LITERT_ASSIGN_OR_RETURN(size_t num_elements,
                          src_type_or->Layout().NumElements());

  float* dst_ptr = static_cast<float*>(dst_lock_or->second);
  const void* src_raw_ptr = src_lock_or->second;

  const auto src_elem_type = src_type_or->ElementType();

  if (src_elem_type == ::litert::ElementType::Int16) {
    const int16_t* src_ptr = static_cast<const int16_t*>(src_raw_ptr);
    for (size_t i = 0; i < num_elements; ++i) {
      dst_ptr[i] = scale * (static_cast<float>(src_ptr[i]) -
                            static_cast<float>(zero_point));
    }
  } else if (src_elem_type == ::litert::ElementType::Int8) {
    const int8_t* src_ptr = static_cast<const int8_t*>(src_raw_ptr);
    for (size_t i = 0; i < num_elements; ++i) {
      dst_ptr[i] = scale * (static_cast<float>(src_ptr[i]) -
                            static_cast<float>(zero_point));
    }
  } else if (src_elem_type == ::litert::ElementType::Float32) {
    std::memcpy(dst_ptr, src_raw_ptr, num_elements * sizeof(float));
  } else if (src_elem_type == ::litert::ElementType::Float16) {
    const tflite::half* src_ptr = static_cast<const tflite::half*>(src_raw_ptr);
    for (size_t i = 0; i < num_elements; ++i) {
      dst_ptr[i] = static_cast<float>(src_ptr[i]);
    }
  } else {
    return absl::InvalidArgumentError(absl::StrCat(
        "Unsupported source type for dequantization: ", (int)src_elem_type));
  }

  // Log first few values and max logit for debugging.
  if (num_elements > 0 && should_dump) {
    float max_logit = dst_ptr[0];
    size_t max_index = 0;
    for (size_t i = 1; i < num_elements; ++i) {
      if (dst_ptr[i] > max_logit) {
        max_logit = dst_ptr[i];
        max_index = i;
      }
    }
    std::cerr << "LITERT_DUMP: Max logit: index=" << max_index
              << " value=" << max_logit << std::endl;
    std::cerr << "LITERT_DUMP: First 5 dequantized logits: " << dst_ptr[0]
              << " " << (num_elements > 1 ? dst_ptr[1] : 0) << " "
              << (num_elements > 2 ? dst_ptr[2] : 0) << " "
              << (num_elements > 3 ? dst_ptr[3] : 0) << " "
              << (num_elements > 4 ? dst_ptr[4] : 0) << std::endl;
  }

  return absl::OkStatus();
}

absl::Status CopyTensorBuffer(const ::litert::TensorBuffer& src,
                              ::litert::TensorBuffer& dst) {
  auto src_type_or = src.TensorType();
  RET_CHECK(src_type_or.HasValue());
  auto dst_type_or = dst.TensorType();
  RET_CHECK(dst_type_or.HasValue());

  auto src_lock_or = ::litert::TensorBufferScopedLock::Create(
      const_cast<::litert::TensorBuffer&>(src),
      ::litert::TensorBuffer::LockMode::kRead);
  RET_CHECK(src_lock_or.HasValue());
  auto dst_lock_or = ::litert::TensorBufferScopedLock::Create(
      dst, ::litert::TensorBuffer::LockMode::kWrite);
  RET_CHECK(dst_lock_or.HasValue());

  auto src_size_or = src.PackedSize();
  RET_CHECK(src_size_or.HasValue());
  auto dst_size_or = dst.PackedSize();
  RET_CHECK(dst_size_or.HasValue());

  RET_CHECK(*src_size_or == *dst_size_or)
      << "Buffer size mismatch during copy: " << *src_size_or << " vs "
      << *dst_size_or;

  std::memcpy(dst_lock_or->second, src_lock_or->second, *src_size_or);
  return absl::OkStatus();
}

}  // namespace

std::string LlmLiteRtNpuCompiledModelExecutor::LatencyStats::DebugString()
    const {
  std::ostringstream formatted_stats;
  formatted_stats << "\n" << "====== PREFILL STATS ======";
  formatted_stats << "\n"
                  << "Total prefill latency [us]: " << prefill_e2e_latency_us;
  formatted_stats << "\n"
                  << "(e2e) Prefill num tokens: " << prefill_num_tokens;
  formatted_stats << "\n"
                  << "(e2e) Prefill tokens per second: "
                  << ((prefill_num_tokens * 1000 * 1000) /
                      (float)prefill_e2e_latency_us);
  formatted_stats << "\n"
                  << "(TransformerStackOnly) Prefill tokens per second: "
                  << ((prefill_num_tokens * 1000 * 1000) /
                      (float)prefill_llm_inference_latency_us);

  formatted_stats << "\n" << "------ Prefill breakdown ------";
  formatted_stats << "\n"
                  << "Total prefill prepare input tensors latency [us]: "
                  << prefill_prepare_input_latency_us << " ("
                  << ((prefill_prepare_input_latency_us * 100) /
                      (float)prefill_e2e_latency_us)
                  << "%)";
  formatted_stats << "\n"
                  << "Total prefill embedder inference latency [us]: "
                  << prefill_embedder_inference_latency_us << " ("
                  << ((prefill_embedder_inference_latency_us * 100) /
                      (float)prefill_e2e_latency_us)
                  << "%)";
  if (prefill_embedder_per_layer_inference_latency_us.has_value()) {
    formatted_stats
        << "\n"
        << "Total prefill embedder per layer inference latency [us]: "
        << prefill_embedder_per_layer_inference_latency_us.value() << " ("
        << ((prefill_embedder_per_layer_inference_latency_us.value() * 100) /
            (float)prefill_e2e_latency_us)
        << "%)";
  }
  formatted_stats << "\n"
                  << "Total prefill rope inference latency [us]: "
                  << prefill_rope_inference_latency_us << " ("
                  << ((prefill_rope_inference_latency_us * 100) /
                      (float)prefill_e2e_latency_us)
                  << "%)";
  formatted_stats << "\n"
                  << "Total prefill mask inference latency [us]: "
                  << prefill_mask_inference_latency_us << " ("
                  << ((prefill_mask_inference_latency_us * 100) /
                      (float)prefill_e2e_latency_us)
                  << "%)";
  formatted_stats << "\n"
                  << "Total prefill llm inference latency [us]: "
                  << prefill_llm_inference_latency_us << " ("
                  << ((prefill_llm_inference_latency_us * 100) /
                      (float)prefill_e2e_latency_us)
                  << "%)";
  formatted_stats << "\n"
                  << "Total prefill cache update inference latency [us]: "
                  << prefill_cache_update_inference_latency_us << " ("
                  << ((prefill_cache_update_inference_latency_us * 100) /
                      (float)prefill_e2e_latency_us)
                  << "%)";

  formatted_stats << "\n\n" << "====== DECODE STATS ======";
  formatted_stats << "\n"
                  << "Total decode latency [us]: " << decode_e2e_latency_us;
  formatted_stats << "\n"
                  << "(e2e) Decode num tokens: " << decode_num_tokens;
  formatted_stats << "\n"
                  << "(e2e) Decode tokens per second (avg): "
                  << ((decode_num_tokens * 1000 * 1000) /
                      (float)decode_e2e_latency_us);
  formatted_stats << "\n"
                  << "(TransformerStackOnly) Decode tokens per second (avg): "
                  << ((decode_num_tokens * 1000 * 1000) /
                      (float)decode_llm_inference_latency_us);

  formatted_stats << "\n" << "------ Decode breakdown ------";
  formatted_stats << "\n"
                  << "Total decode prepare input tensors latency [us]: "
                  << decode_prepare_input_latency_us << " ("
                  << ((decode_prepare_input_latency_us * 100) /
                      (float)decode_e2e_latency_us)
                  << "%)";
  formatted_stats << "\n"
                  << "Total decode embedder inference latency [us]: "
                  << decode_embedder_inference_latency_us << " ("
                  << ((decode_embedder_inference_latency_us * 100) /
                      (float)decode_e2e_latency_us)
                  << "%)";
  if (decode_embedder_per_layer_inference_latency_us.has_value()) {
    formatted_stats
        << "\n"
        << "Total decode embedder per layer inference latency [us]: "
        << decode_embedder_per_layer_inference_latency_us.value() << " ("
        << ((decode_embedder_per_layer_inference_latency_us.value() * 100) /
            (float)decode_e2e_latency_us)
        << "%)";
  }
  formatted_stats << "\n"
                  << "Total decode rope inference latency [us]: "
                  << decode_rope_inference_latency_us << " ("
                  << ((decode_rope_inference_latency_us * 100) /
                      (float)decode_e2e_latency_us)
                  << "%)";
  formatted_stats << "\n"
                  << "Total decode mask inference latency [us]: "
                  << decode_mask_inference_latency_us << " ("
                  << ((decode_mask_inference_latency_us * 100) /
                      (float)decode_e2e_latency_us)
                  << "%)";
  formatted_stats << "\n"
                  << "Total decode llm inference latency [us]: "
                  << decode_llm_inference_latency_us << " ("
                  << ((decode_llm_inference_latency_us * 100) /
                      (float)decode_e2e_latency_us)
                  << "%)";
  formatted_stats << "\n"
                  << "Total decode cache update inference latency [us]: "
                  << decode_cache_update_inference_latency_us << " ("
                  << ((decode_cache_update_inference_latency_us * 100) /
                      (float)decode_e2e_latency_us)
                  << "%)";
  formatted_stats << "\n"
                  << "Total decode sampling latency [us]: "
                  << decode_sampling_latency_us << " ("
                  << ((decode_sampling_latency_us * 100) /
                      (float)decode_e2e_latency_us)
                  << "%)";

  return formatted_stats.str();
}

bool LlmLiteRtNpuCompiledModelExecutor::ShouldLogPerformance() const {
  return false;  // executor_settings_.GetAdvancedSettings().has_value() &&
  // executor_settings_.GetAdvancedSettings()
  //     ->enable_prefill_performance_logging;
}

// Creates LiteRT options for NPU accelerator.
litert::Expected<litert::Options>
LlmLiteRtNpuCompiledModelExecutor::CreateLiteRtOptions(
    const LlmExecutorSettings& settings) {
  LITERT_ASSIGN_OR_RETURN(auto options, ::litert::Options::Create());
  options.SetHardwareAccelerators(litert::HwAccelerators::kNpu |
                                  litert::HwAccelerators::kCpu);
#if defined(__ANDROID__)
  LITERT_ASSIGN_OR_RETURN(::litert::qualcomm::QualcommOptions & qnn_opts,
                          options.GetQualcommOptions());
  qnn_opts.SetLogLevel(::litert::qualcomm::QualcommOptions::LogLevel::kError);
  qnn_opts.SetHtpPerformanceMode(
      ::litert::qualcomm::QualcommOptions::HtpPerformanceMode::kBurst);
#endif
  return options;
}

LlmLiteRtNpuCompiledModelExecutor::LlmLiteRtNpuCompiledModelExecutor(
    LlmExecutorSettings executor_settings, Environment& llm_env,
    EmbedderContext embedder_context, NpuAuxiliaryContext npu_auxiliary_context,
    InferenceContext mask_context, InferenceContext rope_context,
    ::litert::CompiledModel llm_compiled_model,
    InferenceContext llm_inference_context,
    InferenceContext cache_update_inference_context,
    SortedPrefillSignatureMap prefill_signature_map,
    std::optional<std::unique_ptr<EmbeddingLookupManager>>
        embedding_lookup_manager,
    std::optional<std::unique_ptr<EmbeddingLookupManager>>
        per_layer_embedding_lookup_manager,
    std::optional<EmbedderPerLayerContext> embedder_per_layer_context)
    : executor_settings_(std::move(executor_settings)),
      env_(llm_env),
      embedder_context_(std::move(embedder_context)),
      npu_auxiliary_context_(std::move(npu_auxiliary_context)),
      mask_context_(std::move(mask_context)),
      rope_context_(std::move(rope_context)),
      llm_compiled_model_(std::move(llm_compiled_model)),
      embedding_lookup_manager_(std::move(embedding_lookup_manager)),
      per_layer_embedding_lookup_manager_(
          std::move(per_layer_embedding_lookup_manager)),
      embedder_per_layer_context_(std::move(embedder_per_layer_context)),
      llm_inference_context_(std::move(llm_inference_context)),
      cache_update_inference_context_(
          std::move(cache_update_inference_context)),
      prefill_signature_map_(std::move(prefill_signature_map)) {
  if (embedder_per_layer_context_.has_value()) {
    latency_stats_.prefill_embedder_per_layer_inference_latency_us = 0;
    latency_stats_.decode_embedder_per_layer_inference_latency_us = 0;
  }
  if (ShouldDump()) {
    ABSL_LOG(INFO)
        << "LITERT_DUMP: LlmLiteRtNpuCompiledModelExecutor Constructor called";
  }

  // Initialize logits quantization parameters using the 'decode' signature.
  auto decode_sig_idx_or =
      llm_compiled_model_.GetSignatureIndex(kDecodeSignature);
  if (decode_sig_idx_or.HasValue()) {
    LiteRtSignature sig;
    if (LiteRtGetModelSignature(llm_compiled_model_.model_.Get(),
                                *decode_sig_idx_or, &sig) == kLiteRtStatusOk) {
      LiteRtTensor logits_tensor;
      if (LiteRtGetSignatureOutputTensor(sig, "logits", &logits_tensor) ==
          kLiteRtStatusOk) {
        LiteRtQuantizationPerTensor q_params;
        if (LiteRtGetPerTensorQuantization(logits_tensor, &q_params) ==
            kLiteRtStatusOk) {
          logits_scale_ = q_params.scale;
          logits_zero_point_ = (int32_t)q_params.zero_point;
          std::cerr << "LITERT_DUMP: Logits quantization params from '"
                    << kDecodeSignature
                    << "' signature: scale=" << logits_scale_
                    << " zero_point=" << logits_zero_point_ << std::endl;
        } else {
          std::cerr << "LITERT_DUMP: No per-tensor quantization for logits in '"
                    << kDecodeSignature
                    << "' signature (using default scale=1.0, zp=0)."
                    << std::endl;
        }
      }
    }
  } else {
    std::cerr << "LITERT_DUMP: Could not find '" << kDecodeSignature
              << "' signature for logits quantization retrieval." << std::endl;
  }
}

LlmLiteRtNpuCompiledModelExecutor::~LlmLiteRtNpuCompiledModelExecutor() {
  if (ABSL_VLOG_IS_ON(1) || ShouldDump()) {
    PrintLatencyStats(GetLatencyStats());
  }
}

absl::StatusOr<LlmLiteRtNpuCompiledModelExecutor::EmbedderContext>
LlmLiteRtNpuCompiledModelExecutor::CreateEmbedderContextWithBufferSharing(
    ::litert::Environment& env, const litert::Model& embedder_model,
    const ::litert::TensorBuffer& prefill_input_tokens,
    const ::litert::TensorBuffer& decode_input_tokens,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        gemma_prefill_input_buffers,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        gemma_decode_input_buffers,
    const LlmExecutorSettings& settings) {
  LITERT_ASSIGN_OR_RETURN(auto options, CreateLiteRtOptions(settings));
  LITERT_ASSIGN_OR_RETURN(
      CompiledModel embedder_compiled_model,
      CompiledModel::Create(env, embedder_model.Get(), options));

  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      prefill_input_buffers;
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      prefill_output_buffers;
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      decode_input_buffers;
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      decode_output_buffers;

  LITERT_ASSIGN_OR_RETURN(auto prefill_input_tokens_dup,
                          prefill_input_tokens.Duplicate());
  prefill_input_buffers.insert_or_assign(EmbedderSignatures::kEmbedderInput,
                                         std::move(prefill_input_tokens_dup));

  LITERT_ASSIGN_OR_RETURN(
      auto gemma_prefill_input_buffer_dup,
      gemma_prefill_input_buffers[LlmSignatures::kInputEmbeddings].Duplicate());
  prefill_output_buffers.insert_or_assign(
      EmbedderSignatures::kEmbedderOutput,
      std::move(gemma_prefill_input_buffer_dup));

  LITERT_ASSIGN_OR_RETURN(auto decode_input_tokens_dup,
                          decode_input_tokens.Duplicate());
  decode_input_buffers.insert_or_assign(EmbedderSignatures::kEmbedderInput,
                                        std::move(decode_input_tokens_dup));

  LITERT_ASSIGN_OR_RETURN(
      auto gemma_decode_input_buffer_dup,
      gemma_decode_input_buffers[LlmSignatures::kInputEmbeddings].Duplicate());
  decode_output_buffers.insert_or_assign(
      EmbedderSignatures::kEmbedderOutput,
      std::move(gemma_decode_input_buffer_dup));

  EmbedderContext embedder_context(
      std::move(embedder_compiled_model), std::move(prefill_input_buffers),
      std::move(prefill_output_buffers), std::move(decode_input_buffers),
      std::move(decode_output_buffers));
  return std::move(embedder_context);
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
            gemma_decode_input_buffers,
        const LlmExecutorSettings& settings) {
  LITERT_ASSIGN_OR_RETURN(auto options, CreateLiteRtOptions(settings));
  LITERT_ASSIGN_OR_RETURN(
      CompiledModel embedder_compiled_model,
      CompiledModel::Create(env, embedder_model.Get(), options));

  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      prefill_input_buffers;
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      prefill_output_buffers;
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      decode_input_buffers;
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      decode_output_buffers;

  LITERT_ASSIGN_OR_RETURN(auto prefill_input_tokens_dup,
                          prefill_input_tokens.Duplicate());
  prefill_input_buffers.insert_or_assign(
      EmbedderPerLayerSignatures::kEmbedderInput,
      std::move(prefill_input_tokens_dup));

  LITERT_ASSIGN_OR_RETURN(
      auto gemma_prefill_input_buffer_dup,
      gemma_prefill_input_buffers[kPerLayerEmbedderTensor].Duplicate());
  prefill_output_buffers.insert_or_assign(
      EmbedderPerLayerSignatures::kEmbedderOutput,
      std::move(gemma_prefill_input_buffer_dup));

  LITERT_ASSIGN_OR_RETURN(auto decode_input_tokens_dup,
                          decode_input_tokens.Duplicate());
  decode_input_buffers.insert_or_assign(
      EmbedderPerLayerSignatures::kEmbedderInput,
      std::move(decode_input_tokens_dup));

  LITERT_ASSIGN_OR_RETURN(
      auto gemma_decode_input_buffer_dup,
      gemma_decode_input_buffers[kPerLayerEmbedderTensor].Duplicate());
  decode_output_buffers.insert_or_assign(
      EmbedderPerLayerSignatures::kEmbedderOutput,
      std::move(gemma_decode_input_buffer_dup));

  EmbedderPerLayerContext embedder_per_layer_context(
      std::move(embedder_compiled_model), std::move(prefill_input_buffers),
      std::move(prefill_output_buffers), std::move(decode_input_buffers),
      std::move(decode_output_buffers));
  return std::move(embedder_per_layer_context);
}

absl::StatusOr<LlmLiteRtNpuCompiledModelExecutor::NpuAuxiliaryContext>
LlmLiteRtNpuCompiledModelExecutor::CreateNpuAuxiliaryContext(
    ::litert::Environment& env, const litert::Model& npu_auxiliary_model,
    const LlmExecutorSettings& settings) {
  LITERT_ASSIGN_OR_RETURN(auto options, CreateLiteRtOptions(settings));
  LITERT_ASSIGN_OR_RETURN(
      CompiledModel npu_auxiliary_compiled_model,
      CompiledModel::Create(env, npu_auxiliary_model.Get(), options));
  NpuAuxiliaryContext npu_auxiliary_context(
      std::move(npu_auxiliary_compiled_model));
  return std::move(npu_auxiliary_context);
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
      auto prefill_mask_time_step_buffer,
      npu_auxiliary_context.npu_auxiliary_compiled_model.CreateInputBuffer(
          MaskSignatures::kPrefillMask, MaskSignatures::kMaskInputTimeStep));
  prefill_mask_time_step_buffer.Clear();
  prefill_input_buffers.insert_or_assign(
      MaskSignatures::kMaskInputTimeStep,
      std::move(prefill_mask_time_step_buffer));

  LITERT_ASSIGN_OR_RETURN(
      auto prefill_mask_tokens_buffer,
      npu_auxiliary_context.npu_auxiliary_compiled_model.CreateInputBuffer(
          MaskSignatures::kPrefillMask, MaskSignatures::kMaskInputTokens));
  prefill_mask_tokens_buffer.Clear();
  prefill_input_buffers.insert_or_assign(MaskSignatures::kMaskInputTokens,
                                         std::move(prefill_mask_tokens_buffer));

  const std::set<absl::string_view> mask_output_names = {
      MaskSignatures::kMaskOutputLocalMask,
      MaskSignatures::kMaskOutputGlobalMask};
  for (const auto& mask_output_name : mask_output_names) {
    LITERT_RETURN_IF_ERROR(
        gemma_prefill_input_buffers.contains(mask_output_name))
        << "Missing mask output buffer: " << mask_output_name;
    LITERT_ASSIGN_OR_RETURN(
        auto buffer, gemma_prefill_input_buffers[mask_output_name].Duplicate());
    prefill_output_buffers.insert_or_assign(mask_output_name,
                                            std::move(buffer));
  }

  LITERT_ASSIGN_OR_RETURN(
      auto decode_mask_time_step_buffer,
      npu_auxiliary_context.npu_auxiliary_compiled_model.CreateInputBuffer(
          MaskSignatures::kDecodeMask, MaskSignatures::kMaskInputTimeStep));
  decode_mask_time_step_buffer.Clear();
  decode_input_buffers.insert_or_assign(
      MaskSignatures::kMaskInputTimeStep,
      std::move(decode_mask_time_step_buffer));

  LITERT_ASSIGN_OR_RETURN(
      auto decode_mask_tokens_buffer,
      npu_auxiliary_context.npu_auxiliary_compiled_model.CreateInputBuffer(
          MaskSignatures::kDecodeMask, MaskSignatures::kMaskInputTokens));
  decode_mask_tokens_buffer.Clear();
  decode_input_buffers.insert_or_assign(MaskSignatures::kMaskInputTokens,
                                        std::move(decode_mask_tokens_buffer));

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
      auto prefill_rope_pos_buffer,
      npu_auxiliary_context.npu_auxiliary_compiled_model.CreateInputBuffer(
          RopeSignatures::kPrefillRope, RopeSignatures::kInputPos));
  prefill_rope_pos_buffer.Clear();
  prefill_input_buffers.insert_or_assign(RopeSignatures::kInputPos,
                                         std::move(prefill_rope_pos_buffer));

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
        auto buffer, gemma_prefill_input_buffers[rope_output_name].Duplicate());
    prefill_output_buffers.insert_or_assign(rope_output_name,
                                            std::move(buffer));
  }

  LITERT_ASSIGN_OR_RETURN(
      auto decode_rope_pos_buffer,
      npu_auxiliary_context.npu_auxiliary_compiled_model.CreateInputBuffer(
          RopeSignatures::kDecodeRope, RopeSignatures::kInputPos));
  decode_rope_pos_buffer.Clear();
  decode_input_buffers.insert_or_assign(RopeSignatures::kInputPos,
                                        std::move(decode_rope_pos_buffer));

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
      LITERT_ASSIGN_OR_RETURN(auto buffer, llm_compiled_model.CreateInputBuffer(
                                               kPrefillSignature, input_name));
      buffer.Clear();
      input_kv_cache_buffers.insert_or_assign(input_name, std::move(buffer));
    } else {
      LITERT_ASSIGN_OR_RETURN(auto buffer, llm_compiled_model.CreateInputBuffer(
                                               kPrefillSignature, input_name));
      buffer.Clear();
      gemma_prefill_input_buffers.insert_or_assign(input_name,
                                                   std::move(buffer));
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
            auto buffer,
            llm_compiled_model.CreateInputBuffer(kDecodeSignature, input_name));
        buffer.Clear();
        input_kv_cache_buffers.insert_or_assign(input_name, std::move(buffer));
      }
      continue;
    }
    LITERT_ASSIGN_OR_RETURN(auto buffer, llm_compiled_model.CreateInputBuffer(
                                             kDecodeSignature, input_name));
    buffer.Clear();
    gemma_decode_input_buffers.insert_or_assign(input_name, std::move(buffer));
  }

  // Create output buffers for prefill signature.
  for (auto output_name : prefill_signature->OutputNames()) {
    if (absl::StartsWith(output_name, kv_cache_slice_k_root_name) ||
        absl::StartsWith(output_name, kv_cache_slice_v_root_name)) {
      LITERT_ASSIGN_OR_RETURN(
          auto buffer, llm_compiled_model.CreateOutputBuffer(kPrefillSignature,
                                                             output_name));
      prefill_output_kv_cache_slice_buffers.insert_or_assign(output_name,
                                                             std::move(buffer));
    }
  }
  // Create output buffers for decode signature.
  for (auto output_name : decode_signature->OutputNames()) {
    if (absl::StartsWith(output_name, kv_cache_slice_k_root_name) ||
        absl::StartsWith(output_name, kv_cache_slice_v_root_name)) {
      LITERT_ASSIGN_OR_RETURN(
          auto buffer,
          llm_compiled_model.CreateOutputBuffer(kDecodeSignature, output_name));
      decode_output_kv_cache_slice_buffers.insert_or_assign(output_name,
                                                            std::move(buffer));
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
      LITERT_ASSIGN_OR_RETURN(auto buffer, value.Duplicate());
      prefill_input_buffers.insert_or_assign(key, std::move(buffer));
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
      // although it's not used in the model, CompiledModel will complain about
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
        LITERT_ASSIGN_OR_RETURN(auto buffer,
                                corrected_input_buffer.Duplicate());
        prefill_input_buffers.insert_or_assign(key, std::move(buffer));
      } else {
        LITERT_ASSIGN_OR_RETURN(auto buffer, value.Duplicate());
        prefill_input_buffers.insert_or_assign(key, std::move(buffer));
      }
    }
  }
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      prefill_output_buffers;
  {
    // Duplicate all output kv cache slice buffers to prefill output
    // buffers.
    for (const auto& [key, value] : prefill_output_kv_cache_slice_buffers) {
      LITERT_ASSIGN_OR_RETURN(auto buffer, value.Duplicate());
      prefill_output_buffers.insert_or_assign(key, std::move(buffer));
    }
  }
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      decode_input_buffers;
  {
    for (const auto& [key, value] : gemma_decode_input_buffers) {
      LITERT_ASSIGN_OR_RETURN(auto buffer, value.Duplicate());
      decode_input_buffers.insert_or_assign(key, std::move(buffer));
    }
    // Duplicate all kv cache buffers to decode inputs.
    for (const auto& [key, value] : input_kv_cache_buffers) {
      LITERT_ASSIGN_OR_RETURN(auto buffer, value.Duplicate());
      decode_input_buffers.insert_or_assign(key, std::move(buffer));
    }
  }
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      decode_output_buffers;
  {
    // Duplicate all output kv cache slice buffers to decode output
    // buffers.
    for (const auto& [key, value] : decode_output_kv_cache_slice_buffers) {
      LITERT_ASSIGN_OR_RETURN(auto buffer, value.Duplicate());
      decode_output_buffers.insert_or_assign(key, std::move(buffer));
    }

    // The decode signature has an additional output buffer for logits.
    LITERT_ASSIGN_OR_RETURN(
        auto buffer, llm_compiled_model.CreateOutputBuffer(
                         kDecodeSignature, LlmSignatures::kDecodeLogitsOutput));
    decode_output_buffers.insert_or_assign(LlmSignatures::kDecodeLogitsOutput,
                                           std::move(buffer));
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
      LITERT_ASSIGN_OR_RETURN(auto buffer, value.Duplicate());
      prefill_input_buffers.insert_or_assign(key, std::move(buffer));
    }
    for (const auto& [key, value] : prefill_output_kv_cache_slice_buffers) {
      LITERT_ASSIGN_OR_RETURN(auto buffer, value.Duplicate());
      prefill_input_buffers.insert_or_assign(key, std::move(buffer));
    }
    prefill_input_buffers.insert_or_assign(CacheUpdateSignatures::kInputPos,
                                           std::move(prefill_input_pos));
  }
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      prefill_output_buffers;
  {
    for (const auto& [key, value] : input_kv_cache_buffers) {
      LITERT_ASSIGN_OR_RETURN(auto buffer, value.Duplicate());
      prefill_output_buffers.insert_or_assign(key, std::move(buffer));
    }
  }

  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      decode_input_buffers;
  {
    for (const auto& [key, value] : input_kv_cache_buffers) {
      LITERT_ASSIGN_OR_RETURN(auto buffer, value.Duplicate());
      decode_input_buffers.insert_or_assign(key, std::move(buffer));
    }
    for (const auto& [key, value] : decode_output_kv_cache_slice_buffers) {
      LITERT_ASSIGN_OR_RETURN(auto buffer, value.Duplicate());
      decode_input_buffers.insert_or_assign(key, std::move(buffer));
    }
    decode_input_buffers.insert_or_assign(CacheUpdateSignatures::kInputPos,
                                          std::move(decode_input_pos));
  }
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      decode_output_buffers;
  {
    for (const auto& [key, value] : input_kv_cache_buffers) {
      LITERT_ASSIGN_OR_RETURN(auto buffer, value.Duplicate());
      decode_output_buffers.insert_or_assign(key, std::move(buffer));
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
    RETURN_IF_ERROR(Fill(llm_inference_context.decode_input_buffers.at(
                             LlmSignatures::kInputEmbeddings),
                         1));
  }
  if (llm_inference_context.prefill_input_buffers.contains(
          LlmSignatures::kInputEmbeddings)) {
    RETURN_IF_ERROR(Fill(llm_inference_context.prefill_input_buffers.at(
                             LlmSignatures::kInputEmbeddings),
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
  LITERT_ASSIGN_OR_RETURN(auto id_ptr, inputs.GetTextTokenIdsPtr());
  LITERT_ASSIGN_OR_RETURN(auto tensor_type, id_ptr->TensorType());
  // Only accept batch size 1 for now.
  RET_CHECK_EQ(tensor_type.Layout().Dimensions()[0], 1);
  RET_CHECK_GT(tensor_type.Layout().Dimensions()[1], 0)
      << "Prefill token ids must be non-empty.";
  if (embedding_lookup_manager_.has_value()) {
    RETURN_IF_ERROR(
        (*embedding_lookup_manager_)->UpdateMultiModalEmbeddings(inputs));
  }
  LITERT_ASSIGN_OR_RETURN(auto ids, ReferTensorBufferAsSpan<int32_t>(*id_ptr));
  std::stringstream shape_ss;
  for (int dim : tensor_type.Layout().Dimensions()) {
    shape_ss << dim << " ";
  }
  ABSL_VLOG(1)
      << "LlmLiteRtNpuCompiledModelExecutor::Prefill input tensor shape: "
      << shape_ss.str();
  ABSL_VLOG(1)
      << "LlmLiteRtNpuCompiledModelExecutor::Prefill input token count: "
      << ids.size();
  std::stringstream ss;
  for (int i = 0; i < ids.size(); ++i) {
    ss << ids[i] << " ";
  }
  if (ShouldDump()) {
    ABSL_LOG(INFO) << "LITERT_DUMP: Prefill called. num_tokens=" << ids.size()
                   << " ids=" << ss.str()
                   << " Current TokenCount=" << processed_tokens_.TokenCount();
  }

  ASSIGN_OR_RETURN(auto work_groups, GetOptimizedPrefillWorkGroups(
                                         prefill_signature_map_, ids.size()));
  for (const auto& [prefill_signature, prefill_length] : work_groups) {
    ABSL_LOG_IF(INFO, ShouldLogPerformance())
        << "Prefill calling PrefillInternal with signature="
        << prefill_signature << " length=" << prefill_length;
    RETURN_IF_ERROR(PrefillInternal(prefill_signature,
                                    ids.subspan(/*pos=*/0, prefill_length)));
    ids = ids.subspan(/*pos=*/prefill_length);
    latency_stats_.prefill_num_tokens += prefill_length;
  }
  RET_CHECK_EQ(ids.size(), 0).SetCode(absl::StatusCode::kInternal)
      << "Work groups not covering the entire prefill input.";

  if (UseEmbeddingLookupManager()) {
    RETURN_IF_ERROR(
        embedding_lookup_manager_.value()->CleanupMultiModalEmbeddings());
  }
  auto end = absl::Now();
  int64_t e2e_latency = absl::ToInt64Microseconds(end - start);
  latency_stats_.prefill_e2e_latency_us += e2e_latency;
  ABSL_LOG_IF(INFO, ShouldLogPerformance())
      << "Prefill E2E latency: " << e2e_latency << " us";

  if (ABSL_VLOG_IS_ON(1)) {
    PrintLatencyStats(latency_stats_);
  }

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
      llm_inference_context_.decode_output_buffers.at(
          LlmSignatures::kDecodeLogitsOutput);

  if (ShouldDump()) {
    ABSL_LOG(INFO) << "LITERT_DUMP: Decode entry. TokenCount="
                   << processed_tokens_.TokenCount();
  }
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
  std::ofstream log_file;
  if (ShouldDump()) {
    log_file.open("/data/local/tmp/prefill_only.txt", std::ios::app);
  }
  if (log_file.is_open()) {
    log_file << "PREFILL_START: signature=" << prefill_signature
             << " num_ids=" << ids.size() << " ids=[ ";
    for (size_t i = 0; i < std::min(ids.size(), (size_t)20); ++i) {
      log_file << ids[i] << " ";
    }
    log_file << "]\n";
  }
  auto start_prepare_inputs = absl::Now();
  {
    // Prefill input tokens.
    LITERT_ASSIGN_OR_RETURN(
        auto prefill_input_size,
        embedder_context_.inference_context.prefill_input_buffers
            .at(EmbedderSignatures::kEmbedderInput)
            .Size());
    LITERT_ASSIGN_OR_RETURN(
        auto prefill_input_lock_and_addr,
        ::litert::TensorBufferScopedLock::Create(
            embedder_context_.inference_context.prefill_input_buffers.at(
                EmbedderSignatures::kEmbedderInput),
            ::litert::TensorBuffer::LockMode::kWrite));
    auto* prefill_input_ptr =
        static_cast<int32_t*>(prefill_input_lock_and_addr.second);

    // Prefill input position.
    LITERT_ASSIGN_OR_RETURN(
        auto prefill_input_pos_size,
        rope_context_.prefill_input_buffers.at(RopeSignatures::kInputPos)
            .Size());
    LITERT_ASSIGN_OR_RETURN(
        auto prefill_input_pos_lock_and_addr,
        ::litert::TensorBufferScopedLock::Create(
            rope_context_.prefill_input_buffers.at(RopeSignatures::kInputPos),
            ::litert::TensorBuffer::LockMode::kWrite));
    auto* prefill_input_pos_ptr =
        static_cast<int32_t*>(prefill_input_pos_lock_and_addr.second);

    // Timestep input.
    LITERT_ASSIGN_OR_RETURN(auto prefill_timestep_size,
                            mask_context_.prefill_input_buffers
                                .at(MaskSignatures::kMaskInputTimeStep)
                                .Size());
    LITERT_ASSIGN_OR_RETURN(auto prefill_timestep_lock_and_addr,
                            ::litert::TensorBufferScopedLock::Create(
                                mask_context_.prefill_input_buffers.at(
                                    MaskSignatures::kMaskInputTimeStep),
                                ::litert::TensorBuffer::LockMode::kWrite));
    auto* prefill_timestep_ptr =
        static_cast<int32_t*>(prefill_timestep_lock_and_addr.second);

    if (log_file.is_open()) {
      DumpBuffersToFile("PREFILL_ROPE_INPUT",
                        rope_context_.prefill_input_buffers, log_file);
      DumpBuffersToFile("PREFILL_MASK_INPUT",
                        mask_context_.prefill_input_buffers, log_file);
    }

    if (processed_tokens_.TokenCount() != current_step_) {
      RETURN_IF_ERROR(processed_tokens_.RollBackToStep(current_step_));
    }
    // Check if have a pending input token. Note that 'internal_start_step' is
    // always equal to the number of processed tokens plus 1.
    auto [internal_start_step, pending_input_token] =
        processed_tokens_.GetNextUnprocessedToken();

    const size_t prefill_capacity = prefill_input_size / sizeof(int32_t);
    const size_t num_tokens_to_write =
        (pending_input_token.empty() ? 0 : 1) + (ids.size() - 1);
    if (num_tokens_to_write < prefill_capacity) {
      memset(prefill_input_ptr, 0, prefill_input_size);
      memset(prefill_input_pos_ptr, 0, prefill_input_pos_size);
      memset(prefill_timestep_ptr, 0, prefill_timestep_size);
    }

    int input_idx = 0;
    if (!pending_input_token.empty()) {
      // We'll write any pending embedding directly into the transformer
      // embedding buffer.
      if (UseEmbeddingLookupManager()) {
        LITERT_ASSIGN_OR_RETURN(
            auto transformer_embedding_buffer_lock_and_addr,
            ::litert::TensorBufferScopedLock::Create(
                llm_inference_context_.prefill_input_buffers.at(
                    LlmSignatures::kInputEmbeddings),
                ::litert::TensorBuffer::LockMode::kWrite));
        float* transformer_embedding_buffer_ptr = static_cast<float*>(
            transformer_embedding_buffer_lock_and_addr.second);
        memcpy(transformer_embedding_buffer_ptr,
               pending_input_token[0]->embedding().data(),
               pending_input_token[0]->embedding().size() * sizeof(float));

        // We'll also write any pending per-layer embedding directly into the
        // transformer per-layer embedding buffer.
        if (per_layer_embedding_lookup_manager_.has_value() &&
            embedder_per_layer_context_.has_value()) {
          LITERT_ASSIGN_OR_RETURN(
              auto ple_buffer_lock_and_addr,
              ::litert::TensorBufferScopedLock::Create(
                  embedder_per_layer_context_->inference_context
                      .prefill_output_buffers.at(
                          EmbedderPerLayerSignatures::kEmbedderOutput),
                  ::litert::TensorBuffer::LockMode::kWrite));
          float* ple_ptr = static_cast<float*>(ple_buffer_lock_and_addr.second);
          memcpy(ple_ptr, pending_input_token[0]->per_layer_embedding().data(),
                 pending_input_token[0]->per_layer_embedding().size() *
                     sizeof(float));
        }
      }

      prefill_input_ptr[input_idx] = pending_input_token[0]->id();
      prefill_input_pos_ptr[input_idx] = internal_start_step;
      RETURN_IF_ERROR(processed_tokens_.MarkPendingInputTokenAsProcessed());
      ++input_idx;
    }

    prefill_timestep_ptr[0] = internal_start_step;
    // We will not fill the last token of the current input into the compiled
    // model input buffers just yet. It will be stored in the
    // 'processed_tokens_' and used in the next prefill or decode.
    const auto processed_input_tokens = ids.subspan(0, ids.size() - 1);
    for (int i = 0; i < processed_input_tokens.size();
         input_idx++, current_step_++, i++) {
      prefill_input_ptr[input_idx] = processed_input_tokens[i];
      prefill_input_pos_ptr[input_idx] = current_step_;
    }
    processed_tokens_.AddProcessedTokens(processed_input_tokens);

    auto end_prepare_inputs = absl::Now();
    int64_t prepare_input_latency =
        absl::ToInt64Microseconds(end_prepare_inputs - start_prepare_inputs);
    latency_stats_.prefill_prepare_input_latency_us += prepare_input_latency;
    ABSL_LOG_IF(INFO, ShouldLogPerformance())
        << "PrefillInternal: Prepare inputs took " << prepare_input_latency
        << " us";

    if (UseEmbeddingLookupManager()) {
      auto start = absl::Now();
      // We use the embedding lookup manager to populate the embedding buffer.
      // If we already placed a pending input token into the embedding buffer
      // before, we'll flag that as an offset to the embedding lookup manager.
      litert::TensorBuffer& embedding_buffer =
          llm_inference_context_.prefill_input_buffers.at(
              LlmSignatures::kInputEmbeddings);
      RETURN_IF_ERROR(embedding_lookup_manager_.value()->LookupPrefill(
          processed_input_tokens, &embedding_buffer,
          pending_input_token.empty() ? 0 : 1));
      auto end = absl::Now();
      int64_t embedder_latency = absl::ToInt64Microseconds(end - start);
      latency_stats_.prefill_embedder_inference_latency_us += embedder_latency;
      ABSL_LOG_IF(INFO, ShouldLogPerformance())
          << "PrefillInternal: Embedder lookup took " << embedder_latency
          << " us";
    }

    if (log_file.is_open()) {
      DumpBuffersToFile(
          "PREFILL_INPUT",
          embedder_context_.inference_context.prefill_input_buffers, log_file);
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
    if (per_layer_embedding_lookup_manager_.has_value()) {
      RETURN_IF_ERROR(
          per_layer_embedding_lookup_manager_.value()->LookupPrefill(
              last_input_token->id(),
              last_input_token->mutable_per_layer_embedding()));
    }
    if (ShouldDump()) {
      ABSL_LOG(INFO) << "Prefill looked up embedding for last token: id="
                     << last_input_token->id()
                     << ", size=" << last_input_token->embedding().size()
                     << ", PLE size="
                     << last_input_token->per_layer_embedding().size();
    }
    auto end = absl::Now();
    int64_t embedder_latency = absl::ToInt64Microseconds(end - start);
    latency_stats_.prefill_embedder_inference_latency_us += embedder_latency;
    ABSL_LOG_IF(INFO, ShouldLogPerformance())
        << "PrefillInternal: Embedder lookup took " << embedder_latency
        << " us";
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
    if (log_file.is_open()) {
      DumpBuffersToFile(
          "PREFILL_EMBEDDER_OUT",
          embedder_context_.inference_context.prefill_output_buffers, log_file);
    }
    auto end = absl::Now();
    latency_stats_.prefill_embedder_inference_latency_us +=
        absl::ToInt64Microseconds(end - start);

    if (!UseEmbeddingLookupManager()) {
      RETURN_IF_ERROR(CopyTensorBuffer(
          embedder_context_.inference_context.prefill_output_buffers.at(
              EmbedderSignatures::kEmbedderOutput),
          llm_inference_context_.prefill_input_buffers.at(
              LlmSignatures::kInputEmbeddings)));
    }
  }
  // Invoke embedder per layer signature if it exists.
  if (embedder_per_layer_context_.has_value()) {
    auto start = absl::Now();
    // For Gemma4 models with a separate PLE sub-model, we need to fill the
    // input token ID buffer before running the model.
    if (UseEmbeddingLookupManager()) {
      // Note: Currently PLE still uses the prefill input tokens buffer.
      // We'll need to make sure this is correctly synchronized if we use NPU.
    }

    auto res =
        embedder_per_layer_context_->embedder_per_layer_compiled_model.Run(
            EmbedderPerLayerSignatures::kPrefillEmbedderPerLayer,
            embedder_per_layer_context_->inference_context
                .prefill_input_buffers,
            embedder_per_layer_context_->inference_context
                .prefill_output_buffers);
    RET_CHECK(res) << "Failed to run embedder per layer model."
                   << res.Error().Message();
    auto end = absl::Now();
    int64_t embedder_per_layer_latency = absl::ToInt64Microseconds(end - start);
    latency_stats_.prefill_embedder_per_layer_inference_latency_us.value() +=
        embedder_per_layer_latency;
    ABSL_LOG_IF(INFO, ShouldLogPerformance())
        << "PrefillInternal: Embedder per layer took "
        << embedder_per_layer_latency << " us";
  }

  // Invoke RoPE signature.
  {
    auto start = absl::Now();
    auto res = npu_auxiliary_context_.npu_auxiliary_compiled_model.Run(
        RopeSignatures::kPrefillRope, rope_context_.prefill_input_buffers,
        rope_context_.prefill_output_buffers);
    RET_CHECK(res) << "Failed to run RoPE model." << res.Error().Message();
    auto end = absl::Now();
    int64_t rope_latency = absl::ToInt64Microseconds(end - start);
    latency_stats_.prefill_rope_inference_latency_us += rope_latency;
    ABSL_LOG_IF(INFO, ShouldLogPerformance())
        << "PrefillInternal: RoPE took " << rope_latency << " us";
    if (log_file.is_open()) {
      DumpBuffersToFile("PREFILL_ROPE_OUT",
                        rope_context_.prefill_output_buffers, log_file);
    }
  }

  // Invoke mask signature.
  {
    auto start = absl::Now();
    auto res = npu_auxiliary_context_.npu_auxiliary_compiled_model.Run(
        MaskSignatures::kPrefillMask, mask_context_.prefill_input_buffers,
        mask_context_.prefill_output_buffers);
    RET_CHECK(res) << "Failed to run compiled model." << res.Error().Message();
    auto end = absl::Now();
    int64_t mask_latency = absl::ToInt64Microseconds(end - start);
    latency_stats_.prefill_mask_inference_latency_us += mask_latency;
    ABSL_LOG_IF(INFO, ShouldLogPerformance())
        << "PrefillInternal: Mask took " << mask_latency << " us";

    for (auto& [name, buffer] : mask_context_.prefill_output_buffers) {
      NormalizeMask(std::string(name), buffer, ShouldDump());
    }

    if (log_file.is_open()) {
      DumpBuffersToFile("PREFILL_MASK_OUT",
                        mask_context_.prefill_output_buffers, log_file);
    }
  }

  // Invoke LLM signature.
  {
    if (log_file.is_open()) {
      DumpBuffersToFile("PREFILL_LLM_IN",
                        llm_inference_context_.prefill_input_buffers, log_file);
    }
    auto start = absl::Now();
    auto res =
        llm_compiled_model_.Run(LlmSignatures::kPrefillLlm,
                                llm_inference_context_.prefill_input_buffers,
                                llm_inference_context_.prefill_output_buffers);
    RET_CHECK(res) << "Failed to run LLM model." << res.Error().Message();
    auto end = absl::Now();
    int64_t llm_inference_latency = absl::ToInt64Microseconds(end - start);
    latency_stats_.prefill_llm_inference_latency_us += llm_inference_latency;
    ABSL_LOG_IF(INFO, ShouldLogPerformance())
        << "PrefillInternal: NPU LLM Inference took " << llm_inference_latency
        << " us";
    if (log_file.is_open()) {
      DumpBuffersToFile("PREFILL_LLM_OUT",
                        llm_inference_context_.prefill_output_buffers,
                        log_file);
      // Log logits if they exist in output buffers
      if (llm_inference_context_.prefill_output_buffers.contains(
              LlmSignatures::kDecodeLogitsOutput)) {
        DumpTensorBufferToFile("PREFILL_LOGITS",
                               llm_inference_context_.prefill_output_buffers.at(
                                   LlmSignatures::kDecodeLogitsOutput),
                               log_file);
      }
    }
  }

  // Cache update.
  {
    auto start = absl::Now();
    auto res = npu_auxiliary_context_.npu_auxiliary_compiled_model.Run(
        CacheUpdateSignatures::kPrefillCacheUpdate,
        cache_update_inference_context_.prefill_input_buffers,
        cache_update_inference_context_.prefill_output_buffers);
    auto end = absl::Now();
    int64_t cache_update_latency = absl::ToInt64Microseconds(end - start);
    latency_stats_.prefill_cache_update_inference_latency_us +=
        cache_update_latency;
    ABSL_LOG_IF(INFO, ShouldLogPerformance())
        << "PrefillInternal: Cache update took " << cache_update_latency
        << " us";
    RET_CHECK(res) << "Failed to run cache update model."
                   << res.Error().Message();
    if (log_file.is_open()) {
      DumpBuffersToFile("PREFILL_CACHE_UPDATE_OUT",
                        cache_update_inference_context_.prefill_output_buffers,
                        log_file);
    }
  }
  return absl::OkStatus();
}

absl::Status LlmLiteRtNpuCompiledModelExecutor::DecodeInternal(
    int step, std::shared_ptr<TokenData> token, bool run_rope_and_mask) {
  int id = token->id();
  std::ofstream log_file;
  if (ShouldDump()) {
    log_file.open("/data/local/tmp/decode_only.txt", std::ios::app);
  }
  if (log_file.is_open()) {
    log_file << "DECODE_START: step=" << step << " id=" << id
             << (id == -1 ? " (EMBEDDING_ONLY)" : "") << "\n";
  }
  auto start_prepare_inputs = absl::Now();

  // int id = token->id();
  {
    if (id != -1) {
      // Decode input tokens.
      LITERT_ASSIGN_OR_RETURN(
          auto decode_input_lock_and_addr,
          ::litert::TensorBufferScopedLock::Create(
              embedder_context_.inference_context.decode_input_buffers.at(
                  EmbedderSignatures::kEmbedderInput),
              ::litert::TensorBuffer::LockMode::kWrite));
      auto* decode_input_ptr =
          static_cast<int32_t*>(decode_input_lock_and_addr.second);
      decode_input_ptr[0] = id;
    }

    // Always update decode input position and timestep, even if
    // run_rope_and_mask is false. The LLM and Cache Update models still need to
    // know the current step.

    // 1. RoPE position
    {
      LITERT_ASSIGN_OR_RETURN(
          auto lock_and_addr,
          ::litert::TensorBufferScopedLock::Create(
              rope_context_.decode_input_buffers.at(RopeSignatures::kInputPos),
              ::litert::TensorBuffer::LockMode::kWrite));
      static_cast<int32_t*>(lock_and_addr.second)[0] = step;
    }

    // 2. Mask timestep
    {
      LITERT_ASSIGN_OR_RETURN(auto lock_and_addr,
                              ::litert::TensorBufferScopedLock::Create(
                                  mask_context_.decode_input_buffers.at(
                                      MaskSignatures::kMaskInputTimeStep),
                                  ::litert::TensorBuffer::LockMode::kWrite));
      static_cast<int32_t*>(lock_and_addr.second)[0] = step;
    }

    // 3. Cache update position
    {
      LITERT_ASSIGN_OR_RETURN(
          auto lock_and_addr,
          ::litert::TensorBufferScopedLock::Create(
              cache_update_inference_context_.decode_input_buffers.at(
                  CacheUpdateSignatures::kInputPos),
              ::litert::TensorBuffer::LockMode::kWrite));
      static_cast<int32_t*>(lock_and_addr.second)[0] = step;
    }

    if (run_rope_and_mask) {
      // RoPE and Mask models are run in the caller.
    }
    if (log_file.is_open()) {
      DumpBuffersToFile(
          "DECODE_INPUT",
          embedder_context_.inference_context.decode_input_buffers, log_file);
    }
  }

  auto end_prepare_inputs = absl::Now();
  latency_stats_.decode_prepare_input_latency_us +=
      absl::ToInt64Microseconds(end_prepare_inputs - start_prepare_inputs);

  if (!UseEmbeddingLookupManager() && id != -1) {
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

      if (!UseEmbeddingLookupManager()) {
        RETURN_IF_ERROR(CopyTensorBuffer(
            embedder_context_.inference_context.decode_output_buffers.at(
                EmbedderSignatures::kEmbedderOutput),
            llm_inference_context_.decode_input_buffers.at(
                LlmSignatures::kInputEmbeddings)));
      }

      if (log_file.is_open()) {
        DumpBuffersToFile(
            "DECODE_EMBEDDER_OUT",
            embedder_context_.inference_context.decode_output_buffers,
            log_file);
      }
    }
  }

  if (UseEmbeddingLookupManager() || id == -1) {
    // We'll write any pending embedding directly into the transformer
    // embedding buffer.
    auto start = absl::Now();
    LITERT_ASSIGN_OR_RETURN(auto transformer_embedding_buffer_lock_and_addr,
                            ::litert::TensorBufferScopedLock::Create(
                                llm_inference_context_.decode_input_buffers.at(
                                    LlmSignatures::kInputEmbeddings),
                                ::litert::TensorBuffer::LockMode::kWrite));
    float* transformer_embedding_buffer_ptr =
        static_cast<float*>(transformer_embedding_buffer_lock_and_addr.second);
    if (!token->embedding().empty()) {
      memcpy(transformer_embedding_buffer_ptr, token->embedding().data(),
             token->embedding().size() * sizeof(float));
    }

    if (llm_inference_context_.decode_input_buffers.contains(
            kPerLayerEmbedderTensor) &&
        !token->per_layer_embedding().empty()) {
      LITERT_ASSIGN_OR_RETURN(
          auto transformer_ple_embedding_buffer_lock_and_addr,
          ::litert::TensorBufferScopedLock::Create(
              llm_inference_context_.decode_input_buffers.at(
                  kPerLayerEmbedderTensor),
              ::litert::TensorBuffer::LockMode::kWrite));
      float* transformer_ple_embedding_buffer_ptr = static_cast<float*>(
          transformer_ple_embedding_buffer_lock_and_addr.second);
      memcpy(transformer_ple_embedding_buffer_ptr,
             token->per_layer_embedding().data(),
             token->per_layer_embedding().size() * sizeof(float));
    }

    // Log the input embedding for comparison
    if (log_file.is_open()) {
      log_file << "LITERT_DUMP: DECODE_EMBEDDING: size="
               << token->embedding().size()
               << " PLE_size=" << token->per_layer_embedding().size()
               << " VALUES: [START] ";
      log_file << std::fixed << std::setprecision(6);
      for (size_t i = 0; i < std::min(token->embedding().size(), (size_t)20);
           ++i) {
        log_file << token->embedding()[i] << " ";
      }
      log_file << "[END]\n";
    }

    latency_stats_.decode_embedder_inference_latency_us +=
        absl::ToInt64Microseconds(absl::Now() - start);
  }

  {
    if (embedder_per_layer_context_.has_value() && id != -1) {
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

  if (log_file.is_open()) {
    DumpBuffersToFile("DECODE_ROPE_INPUT", rope_context_.decode_input_buffers,
                      log_file);
    DumpBuffersToFile("DECODE_MASK_INPUT", mask_context_.decode_input_buffers,
                      log_file);
  }

  if (run_rope_and_mask) {
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
      if (log_file.is_open()) {
        DumpBuffersToFile("DECODE_ROPE_OUT",
                          rope_context_.decode_output_buffers, log_file);
      }
    }

    // Invoke mask signature.
    {
      auto start = absl::Now();
      auto res = npu_auxiliary_context_.npu_auxiliary_compiled_model.Run(
          MaskSignatures::kDecodeMask, mask_context_.decode_input_buffers,
          mask_context_.decode_output_buffers);
      RET_CHECK(res) << "Failed to run compiled model."
                     << res.Error().Message();
      auto end = absl::Now();
      latency_stats_.decode_mask_inference_latency_us +=
          absl::ToInt64Microseconds(end - start);

      for (auto& [name, buffer] : mask_context_.decode_output_buffers) {
        NormalizeMask(std::string(name), buffer, ShouldDump());
      }

      if (log_file.is_open()) {
        DumpBuffersToFile("DECODE_MASK_OUT",
                          mask_context_.decode_output_buffers, log_file);
      }
    }
  }

  // Invoke LLM signature.
  {
    if (log_file.is_open()) {
      DumpBuffersToFile("DECODE_LLM_IN",
                        llm_inference_context_.decode_input_buffers, log_file);
    }
    auto start = absl::Now();
    auto res = llm_compiled_model_.Run(
        LlmSignatures::kDecodeLlm, llm_inference_context_.decode_input_buffers,
        llm_inference_context_.decode_output_buffers);
    auto end = absl::Now();
    latency_stats_.decode_llm_inference_latency_us +=
        absl::ToInt64Microseconds(end - start);
    RET_CHECK(res) << "Failed to run LLM model." << res.Error().Message();
    if (log_file.is_open()) {
      DumpBuffersToFile("DECODE_LLM_OUT",
                        llm_inference_context_.decode_output_buffers, log_file);
    }
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
    if (log_file.is_open()) {
      DumpBuffersToFile("DECODE_CACHE_UPDATE_OUT",
                        cache_update_inference_context_.decode_output_buffers,
                        log_file);
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<int> LlmLiteRtNpuCompiledModelExecutor::GetVocabSize() {
  LITERT_ASSIGN_OR_RETURN(
      auto logits_tensor_type,
      llm_inference_context_
          .decode_output_buffers[LlmSignatures::kDecodeLogitsOutput]
          .TensorType());
  RET_CHECK_EQ(logits_tensor_type.Layout().Dimensions().size(), 2);
  return logits_tensor_type.Layout().Dimensions()[1];
}

absl::Status LlmLiteRtNpuCompiledModelExecutor::DecodeToLogits(
    const uint32_t model_size, const uint32_t cache_size,
    const ::litert::TensorBuffer& input_tokens,
    const ::litert::TensorBuffer& local_attention_mask,
    const ::litert::TensorBuffer& global_attention_mask,
    const ::litert::TensorBuffer& seq_position,
    absl::Span<const uint32_t> logits_indices, const std::atomic_bool* cancel,
    ::litert::TensorBuffer& output_logits) {
  LITERT_ASSIGN_OR_RETURN(auto input_tokens_span,
                          ReferTensorBufferAsSpan<int32_t>(input_tokens));
  if (ShouldDump()) {
    ABSL_LOG(INFO) << "LITERT_DUMP: DecodeToLogits (tokens) model_size: "
                   << model_size;
  }
  if (model_size > 1) {
    RETURN_IF_ERROR(PrefillInternal(kPrefillSignature,
                                    input_tokens_span.subspan(0, model_size)));
  }

  if (logits_indices.empty()) {
    return absl::OkStatus();
  }

  RET_CHECK_EQ(logits_indices.size(), 1);
  RET_CHECK_EQ(logits_indices[0], model_size - 1);

  // Use the actual last token ID from the input tokens.
  std::shared_ptr<TokenData> last_token =
      std::make_shared<TokenData>(input_tokens_span[model_size - 1]);

  if (UseEmbeddingLookupManager()) {
    RETURN_IF_ERROR(embedding_lookup_manager_.value()->LookupPrefill(
        last_token->id(), last_token->mutable_embedding()));
    if (per_layer_embedding_lookup_manager_.has_value()) {
      RETURN_IF_ERROR(
          per_layer_embedding_lookup_manager_.value()->LookupPrefill(
              last_token->id(), last_token->mutable_per_layer_embedding()));
    }
  }

  LITERT_ASSIGN_OR_RETURN(auto seq_pos_span,
                          ReferTensorBufferAsSpan<int32_t>(seq_position));
  int step = seq_pos_span[logits_indices[0]];
  if (ShouldDump()) {
    ABSL_LOG(INFO) << "LITERT_DUMP: DecodeToLogits (tokens) step: " << step;
  }
  RETURN_IF_ERROR(DecodeInternal(step, last_token));
  current_step_ = step + 1;

  const auto& src_buffer = llm_inference_context_.decode_output_buffers.at(
      LlmSignatures::kDecodeLogitsOutput);
  return DequantizeLogits(src_buffer, output_logits, logits_scale_,
                          logits_zero_point_, ShouldDump());
}

absl::Status LlmLiteRtNpuCompiledModelExecutor::DecodeToLogits(
    const uint32_t model_size, const uint32_t cache_size,
    absl::Span<const float> embeddings, absl::Span<const float> ple_embeddings,
    const ::litert::TensorBuffer& local_attention_mask,
    const ::litert::TensorBuffer& global_attention_mask,
    const ::litert::TensorBuffer& seq_position,
    absl::Span<const uint32_t> logits_indices, const std::atomic_bool* cancel,
    ::litert::TensorBuffer& output_logits) {
  LITERT_ASSIGN_OR_RETURN(auto seq_pos_span,
                          ReferTensorBufferAsSpan<int32_t>(seq_position));
  RET_CHECK_GE(seq_pos_span.size(), model_size);

  const size_t embedding_dim = embeddings.size() / model_size;
  const size_t ple_embedding_dim =
      ple_embeddings.empty() ? 0 : ple_embeddings.size() / model_size;

  if (is_bulk_embedding_ && model_size > 1) {
    int prefill_num = model_size - 1;
    LITERT_ASSIGN_OR_RETURN(
        auto prefill_wgs,
        GetOptimizedPrefillWorkGroups(prefill_signature_map_, prefill_num));
    int offset = 0;
    for (const auto& [prefill_signature, prefill_length] : prefill_wgs) {
      RETURN_IF_ERROR(PrefillInternalFromEmbeddings(
          prefill_signature,
          embeddings.subspan(offset * embedding_dim,
                             prefill_length * embedding_dim),
          ple_embeddings.empty()
              ? absl::Span<const float>()
              : ple_embeddings.subspan(offset * ple_embedding_dim,
                                       prefill_length * ple_embedding_dim),
          seq_pos_span.subspan(offset, prefill_length)));
      offset += prefill_length;
      latency_stats_.prefill_num_tokens += prefill_length;
    }
    RET_CHECK_EQ(offset, prefill_num).SetCode(absl::StatusCode::kInternal)
        << "Work groups not covering the entire prefill input.";

    // Process the last token with DecodeInternal to generate logits
    std::shared_ptr<TokenData> token = std::make_shared<TokenData>(-1);
    if (!embeddings.empty()) {
      token->mutable_embedding() = std::vector<float>(
          embeddings.begin() + offset * embedding_dim, embeddings.end());
    }
    if (!ple_embeddings.empty()) {
      token->mutable_per_layer_embedding() = std::vector<float>(
          ple_embeddings.begin() + offset * ple_embedding_dim,
          ple_embeddings.end());
    }
    int step = seq_pos_span[offset];
    RETURN_IF_ERROR(DecodeInternal(step, token, /*run_rope_and_mask=*/true));
    current_step_ = step + 1;
  } else {
    for (int i = 0; i < model_size; ++i) {
      std::shared_ptr<TokenData> token = std::make_shared<TokenData>(-1);
      if (!embeddings.empty()) {
        token->mutable_embedding() =
            std::vector<float>(embeddings.begin() + i * embedding_dim,
                               embeddings.begin() + (i + 1) * embedding_dim);
      }
      if (!ple_embeddings.empty()) {
        token->mutable_per_layer_embedding() = std::vector<float>(
            ple_embeddings.begin() + i * ple_embedding_dim,
            ple_embeddings.begin() + (i + 1) * ple_embedding_dim);
      }
      int step = seq_pos_span[i];
      RETURN_IF_ERROR(DecodeInternal(step, token, /*run_rope_and_mask=*/true));
      current_step_ = step + 1;
    }
  }

  if (logits_indices.empty()) {
    return absl::OkStatus();
  }
  RET_CHECK_EQ(logits_indices.size(), 1);
  RET_CHECK_EQ(logits_indices[0], model_size - 1);

  const auto& src_buffer = llm_inference_context_.decode_output_buffers.at(
      LlmSignatures::kDecodeLogitsOutput);
  return DequantizeLogits(src_buffer, output_logits, logits_scale_,
                          logits_zero_point_, ShouldDump());
}

absl::StatusOr<std::vector<std::pair<uint32_t, uint32_t>>>
LlmLiteRtNpuCompiledModelExecutor::GetVariants() {
  // Context size from KV cache buffer.
  LITERT_ASSIGN_OR_RETURN(auto kvc_type,
                          llm_inference_context_.prefill_input_buffers.begin()
                              ->second.TensorType());
  int context_size = 0;
  for (auto dim : kvc_type.Layout().Dimensions()) {
    context_size = std::max(context_size, dim);
  }
  // TODO: b/398321095 - Context size deduction might be incorrect for some
  // models, leading to cache full errors. Forcing 4096 for now.
  context_size = 4096;

  std::vector<std::pair<uint32_t, uint32_t>> variants;
  variants.push_back({1, context_size});
  variants.push_back({kPrefillSize, context_size});
  return variants;
}

LlmLiteRtNpuCompiledModelExecutor::LatencyStats
LlmLiteRtNpuCompiledModelExecutor::GetLatencyStats() const {
  return latency_stats_;
}

absl::Status LlmLiteRtNpuCompiledModelExecutor::Reset() {
  if (ABSL_VLOG_IS_ON(1) || ShouldDump()) {
    PrintLatencyStats(GetLatencyStats());
  }
  current_step_ = 0;
  RETURN_IF_ERROR(processed_tokens_.RollBackToStep(0));
  sampled_ids_.clear();
  latency_stats_ = {};
  if (embedder_per_layer_context_.has_value()) {
    latency_stats_.prefill_embedder_per_layer_inference_latency_us = 0;
    latency_stats_.decode_embedder_per_layer_inference_latency_us = 0;
  }

  if (embedder_per_layer_context_.has_value()) {
    latency_stats_.prefill_embedder_per_layer_inference_latency_us = 0;
    latency_stats_.decode_embedder_per_layer_inference_latency_us = 0;
  }

  RETURN_IF_ERROR(ClearKVCache(llm_inference_context_.prefill_input_buffers));
  return absl::OkStatus();
}

absl::Status LlmLiteRtNpuCompiledModelExecutor::UseLoRA(
    std::optional<uint32_t> lora_id) {
  if (lora_id.has_value()) {
    return absl::UnimplementedError("LoRA is not supported by NPU executor.");
  }
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
  LITERT_ASSIGN_OR_RETURN(auto& qnn_opts, options.GetQualcommOptions());
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
  LITERT_ASSIGN_OR_RETURN(auto options, CreateLiteRtOptions(executor_settings));
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

  RETURN_IF_ERROR(AllocateTransformerBuffers(
      env, transformer_model, llm_compiled_model, gemma_prefill_input_buffers,
      gemma_decode_input_buffers, input_kv_cache_buffers,
      prefill_output_kv_cache_slice_buffers,
      decode_output_kv_cache_slice_buffers));

  // Gemma3n specific fix: KV cache buffer 19 of *prefill* is not connected
  // to any OPs in the model, making the LiteRT runtime allocate host memory
  // for it. This is incompatible when running the transformer model on the NPU.
  if (input_kv_cache_buffers.contains(cache_k19)) {
    LITERT_ASSIGN_OR_RETURN(auto buffer_k, llm_compiled_model.CreateInputBuffer(
                                               kDecodeSignature, cache_k19));
    buffer_k.Clear();
    input_kv_cache_buffers.insert_or_assign(cache_k19, std::move(buffer_k));

    LITERT_ASSIGN_OR_RETURN(auto buffer_v, llm_compiled_model.CreateInputBuffer(
                                               kDecodeSignature, cache_v19));
    buffer_v.Clear();
    input_kv_cache_buffers.insert_or_assign(cache_v19, std::move(buffer_v));
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
                   CreateNpuAuxiliaryContext(env, *npu_auxiliary_lrt_model,
                                             executor_settings));

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
          gemma_prefill_input_buffers, gemma_decode_input_buffers,
          executor_settings));

  ASSIGN_OR_RETURN(auto rope_context,
                   CreateRopeContextWithBufferSharing(
                       npu_auxiliary_context, gemma_prefill_input_buffers,
                       gemma_decode_input_buffers));

  // Duplicate the rope's buffers that are used to store the prefill and
  // decode input position, because they will need to be passed to the
  // cache update inference context as well.
  LITERT_ASSIGN_OR_RETURN(
      ::litert::TensorBuffer prefill_input_pos,
      rope_context.prefill_input_buffers.at(RopeSignatures::kInputPos)
          .Duplicate());
  LITERT_ASSIGN_OR_RETURN(
      ::litert::TensorBuffer decode_input_pos,
      rope_context.decode_input_buffers.at(RopeSignatures::kInputPos)
          .Duplicate());
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
                                     "decode_embedder", &env));

  std::optional<EmbedderPerLayerContext> embedder_per_layer_context =
      std::nullopt;

  ASSIGN_OR_RETURN(
      const litert::Model* embedder_per_layer_model,
      resources.GetTFLiteModel(ModelType::kTfLitePerLayerEmbedder));
  ASSIGN_OR_RETURN(embedder_per_layer_context,
                   CreateEmbedderPerLayerContextWithBufferSharing(
                       env, *embedder_per_layer_model,
                       mask_context.prefill_input_buffers.at(
                           MaskSignatures::kMaskInputTokens),
                       mask_context.decode_input_buffers.at(
                           MaskSignatures::kMaskInputTokens),
                       gemma_prefill_input_buffers, gemma_decode_input_buffers,
                       executor_settings));

  ASSIGN_OR_RETURN(
      std::unique_ptr<EmbeddingLookupManager>
          per_layer_embedding_lookup_manager,
      EmbeddingLookupManager::Create(
          embedder_per_layer_model, end_of_multi_modal_embedding_models, true,
          std::string(EmbedderPerLayerSignatures::kDecodeEmbedderPerLayer),
          &env));

  auto executor = absl::WrapUnique(new LlmLiteRtNpuCompiledModelExecutor(
      executor_settings, env, std::move(embedder_context),
      std::move(npu_auxiliary_context), std::move(mask_context),
      std::move(rope_context), std::move(llm_compiled_model),
      std::move(llm_inference_context),
      std::move(cache_update_inference_context), std::move(prefill_runner_set),
      std::move(embedding_lookup_manager),
      std::move(per_layer_embedding_lookup_manager),
      std::move(embedder_per_layer_context)));
  return executor;
}

absl::StatusOr<std::unique_ptr<LlmLiteRtNpuCompiledModelExecutor>>
LlmLiteRtNpuCompiledModelExecutor::CreateForModelWithoutPerLayerEmbedding(
    const LlmExecutorSettings& executor_settings, ModelResources& resources,
    litert::Environment& env, const litert::Model* transformer_model) {
  // Set up LiteRt options.
  LITERT_ASSIGN_OR_RETURN(auto options, CreateLiteRtOptions(executor_settings));
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

  RETURN_IF_ERROR(AllocateTransformerBuffers(
      env, transformer_model, llm_compiled_model, gemma_prefill_input_buffers,
      gemma_decode_input_buffers, input_kv_cache_buffers,
      prefill_output_kv_cache_slice_buffers,
      decode_output_kv_cache_slice_buffers));
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
    LITERT_ASSIGN_OR_RETURN(auto buffer_k, llm_compiled_model.CreateInputBuffer(
                                               kDecodeSignature, cache_k25));
    llm_inference_context.decode_input_buffers.insert_or_assign(
        cache_k25, std::move(buffer_k));
    LITERT_ASSIGN_OR_RETURN(auto buffer_v, llm_compiled_model.CreateInputBuffer(
                                               kDecodeSignature, cache_v25));
    llm_inference_context.decode_input_buffers.insert_or_assign(
        cache_v25, std::move(buffer_v));
  } else if (llm_inference_context.prefill_input_buffers.contains(cache_k23)) {
    // Fast VLM model specific fix:
    LITERT_ASSIGN_OR_RETURN(auto buffer_k, llm_compiled_model.CreateInputBuffer(
                                               kDecodeSignature, cache_k23));
    llm_inference_context.decode_input_buffers.insert_or_assign(
        cache_k23, std::move(buffer_k));
    LITERT_ASSIGN_OR_RETURN(auto buffer_v, llm_compiled_model.CreateInputBuffer(
                                               kDecodeSignature, cache_v23));
    llm_inference_context.decode_input_buffers.insert_or_assign(
        cache_v23, std::move(buffer_v));
  } else {
    // Tiny Gemma 270M specific fix:
    LITERT_ASSIGN_OR_RETURN(auto buffer_k, llm_compiled_model.CreateInputBuffer(
                                               kDecodeSignature, cache_k17));
    llm_inference_context.decode_input_buffers.insert_or_assign(
        cache_k17, std::move(buffer_k));
    LITERT_ASSIGN_OR_RETURN(auto buffer_v, llm_compiled_model.CreateInputBuffer(
                                               kDecodeSignature, cache_v17));
    llm_inference_context.decode_input_buffers.insert_or_assign(
        cache_v17, std::move(buffer_v));
  }

  ASSIGN_OR_RETURN(auto npu_auxiliary_lrt_model,
                   resources.GetTFLiteModel(ModelType::kTfLiteAux));

  ASSIGN_OR_RETURN(auto npu_auxiliary_context,
                   CreateNpuAuxiliaryContext(env, *npu_auxiliary_lrt_model,
                                             executor_settings));

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
          gemma_prefill_input_buffers, gemma_decode_input_buffers,
          executor_settings));

  ASSIGN_OR_RETURN(auto rope_context,
                   CreateRopeContextWithBufferSharing(
                       npu_auxiliary_context, gemma_prefill_input_buffers,
                       gemma_decode_input_buffers));

  // Duplicate the rope's buffers that are used to store the prefill and
  // decode input position, because they will need to be passed to the
  // cache update inference context as well.
  LITERT_ASSIGN_OR_RETURN(
      ::litert::TensorBuffer prefill_input_pos,
      rope_context.prefill_input_buffers.at(RopeSignatures::kInputPos)
          .Duplicate());
  LITERT_ASSIGN_OR_RETURN(
      ::litert::TensorBuffer decode_input_pos,
      rope_context.decode_input_buffers.at(RopeSignatures::kInputPos)
          .Duplicate());
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
  // If the model has vision or audio encoder, we need to create the embedding
  // lookup manager.
  if (resources.GetTFLiteModel(ModelType::kTfLiteVisionEncoder).ok() ||
      resources.GetTFLiteModel(ModelType::kTfLiteAudioEncoderHw).ok()) {
    ASSIGN_OR_RETURN(maybe_embedding_lookup_manager,
                     EmbeddingLookupManager::Create(embedder_lrt_model, true,
                                                    "decode_embedder", &env));
  }

  auto executor = absl::WrapUnique(new LlmLiteRtNpuCompiledModelExecutor(
      executor_settings, env, std::move(embedder_context),
      std::move(npu_auxiliary_context), std::move(mask_context),
      std::move(rope_context), std::move(llm_compiled_model),
      std::move(llm_inference_context),
      std::move(cache_update_inference_context), std::move(prefill_runner_set),
      std::move(maybe_embedding_lookup_manager), std::nullopt, std::nullopt));
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

absl::Status LlmLiteRtNpuCompiledModelExecutor::ComputeTokenEmbeddings(
    const ::litert::TensorBuffer& input_tokens,
    absl::Span<float> output_embeddings,
    absl::Span<float> output_ple_embeddings) {
  return GenericComputeTokenEmbeddings(
      input_tokens, output_embeddings, output_ple_embeddings,
      embedding_lookup_manager_.has_value() ? embedding_lookup_manager_->get()
                                            : nullptr,
      per_layer_embedding_lookup_manager_.has_value()
          ? per_layer_embedding_lookup_manager_->get()
          : nullptr);
}

absl::Status LlmLiteRtNpuCompiledModelExecutor::PrefillInternalFromEmbeddings(
    absl::string_view prefill_signature, absl::Span<const float> embeddings,
    absl::Span<const float> ple_embeddings,
    absl::Span<const int32_t> seq_positions) {
  LITERT_LM_PERFETTO_TRACE_EVENT("PrefillInternalFromEmbeddings");
  auto start = absl::Now();

  // RoPE and Mask inputs
  {
    LITERT_ASSIGN_OR_RETURN(
        auto prefill_input_pos_lock_and_addr,
        ::litert::TensorBufferScopedLock::Create(
            rope_context_.prefill_input_buffers.at(RopeSignatures::kInputPos),
            ::litert::TensorBuffer::LockMode::kWrite));
    auto* prefill_input_pos_ptr =
        static_cast<int32_t*>(prefill_input_pos_lock_and_addr.second);

    LITERT_ASSIGN_OR_RETURN(auto prefill_timestep_size,
                            mask_context_.prefill_input_buffers
                                .at(MaskSignatures::kMaskInputTimeStep)
                                .Size());
    LITERT_ASSIGN_OR_RETURN(auto prefill_timestep_lock_and_addr,
                            ::litert::TensorBufferScopedLock::Create(
                                mask_context_.prefill_input_buffers.at(
                                    MaskSignatures::kMaskInputTimeStep),
                                ::litert::TensorBuffer::LockMode::kWrite));
    auto* prefill_timestep_ptr =
        static_cast<int32_t*>(prefill_timestep_lock_and_addr.second);

    const size_t num_tokens_to_write = seq_positions.size();
    LITERT_ASSIGN_OR_RETURN(
        auto prefill_input_pos_size,
        rope_context_.prefill_input_buffers.at(RopeSignatures::kInputPos)
            .Size());
    const size_t prefill_capacity = prefill_input_pos_size / sizeof(int32_t);

    if (num_tokens_to_write < prefill_capacity) {
      memset(prefill_input_pos_ptr, 0, prefill_input_pos_size);
      memset(prefill_timestep_ptr, 0, prefill_timestep_size);
    }

    for (size_t i = 0; i < num_tokens_to_write; ++i) {
      prefill_input_pos_ptr[i] = seq_positions[i];
    }
    prefill_timestep_ptr[0] = seq_positions[0];

    if (num_tokens_to_write > 0) {
      std::vector<int> fake_ids(num_tokens_to_write, -1);
      processed_tokens_.AddProcessedTokens(fake_ids);
    }

    current_step_ = seq_positions.back() + 1;

    // Copy embeddings verbatim to the transformer graphs kInputEmbeddings
    // buffer.
    LITERT_ASSIGN_OR_RETURN(auto transformer_embedding_buffer_lock_and_addr,
                            ::litert::TensorBufferScopedLock::Create(
                                llm_inference_context_.prefill_input_buffers.at(
                                    LlmSignatures::kInputEmbeddings),
                                ::litert::TensorBuffer::LockMode::kWrite));
    float* transformer_embedding_buffer_ptr =
        static_cast<float*>(transformer_embedding_buffer_lock_and_addr.second);

    LITERT_ASSIGN_OR_RETURN(auto prefill_emb_size,
                            llm_inference_context_.prefill_input_buffers
                                .at(LlmSignatures::kInputEmbeddings)
                                .Size());
    if (embeddings.size() * sizeof(float) < prefill_emb_size) {
      memset(transformer_embedding_buffer_ptr, 0, prefill_emb_size);
    }
    memcpy(transformer_embedding_buffer_ptr, embeddings.data(),
           std::min(embeddings.size() * sizeof(float),
                    static_cast<size_t>(prefill_emb_size)));

    if (!ple_embeddings.empty() && embedder_per_layer_context_.has_value()) {
      LITERT_ASSIGN_OR_RETURN(
          auto transformer_ple_buffer_lock_and_addr,
          ::litert::TensorBufferScopedLock::Create(
              llm_inference_context_.prefill_input_buffers.at(
                  kPerLayerEmbedderTensor),
              ::litert::TensorBuffer::LockMode::kWrite));
      float* transformer_ple_buffer_ptr =
          static_cast<float*>(transformer_ple_buffer_lock_and_addr.second);

      LITERT_ASSIGN_OR_RETURN(auto prefill_ple_size,
                              llm_inference_context_.prefill_input_buffers
                                  .at(kPerLayerEmbedderTensor)
                                  .Size());
      if (ple_embeddings.size() * sizeof(float) < prefill_ple_size) {
        memset(transformer_ple_buffer_ptr, 0, prefill_ple_size);
      }
      memcpy(transformer_ple_buffer_ptr, ple_embeddings.data(),
             std::min(ple_embeddings.size() * sizeof(float),
                      static_cast<size_t>(prefill_ple_size)));
    }
  }

  {
    auto res = npu_auxiliary_context_.npu_auxiliary_compiled_model.Run(
        RopeSignatures::kPrefillRope, rope_context_.prefill_input_buffers,
        rope_context_.prefill_output_buffers);
    RET_CHECK(res) << "Failed to run RoPE model: " << res.Error().Message();
  }
  {
    auto res = npu_auxiliary_context_.npu_auxiliary_compiled_model.Run(
        MaskSignatures::kPrefillMask, mask_context_.prefill_input_buffers,
        mask_context_.prefill_output_buffers);
    RET_CHECK(res) << "Failed to run Mask model: " << res.Error().Message();
  }
  {
    auto res = llm_compiled_model_.Run(
        prefill_signature, llm_inference_context_.prefill_input_buffers,
        llm_inference_context_.prefill_output_buffers);
    RET_CHECK(res) << "Failed to run LLM compiled model: "
                   << res.Error().Message();
  }
  {
    auto res = npu_auxiliary_context_.npu_auxiliary_compiled_model.Run(
        CacheUpdateSignatures::kPrefillCacheUpdate,
        cache_update_inference_context_.prefill_input_buffers,
        cache_update_inference_context_.prefill_output_buffers);
    RET_CHECK(res) << "Failed to run NPU auxiliary model: KVCache"
                   << res.Error().Message();
  }

  latency_stats_.prefill_e2e_latency_us +=
      absl::ToInt64Microseconds(absl::Now() - start);
  return absl::OkStatus();
}

bool LlmLiteRtNpuCompiledModelExecutor::ShouldDump() const {
  if (ABSL_VLOG_IS_ON(1)) return true;
  const auto& settings = executor_settings_.GetAdvancedSettings();
  return settings.has_value() && settings->enable_litert_dump;
}
}  // namespace litert::lm
