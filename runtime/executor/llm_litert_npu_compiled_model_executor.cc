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
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/log/log.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
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
#include "litert/cc/litert_options.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#if defined(__ANDROID__)
#include "litert/cc/options/litert_google_tensor_options.h"  // from @litert
#include "litert/cc/options/litert_qualcomm_options.h"  // from @litert
#endif

#include "runtime/components/embedding_lookup/embedding_lookup_manager.h"
#include "runtime/components/model_resources.h"
#include "runtime/executor/litert_compiled_model_executor_utils.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/executor/llm_executor_processed_tokens.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/executor/llm_litert_npu_compiled_model_executor_utils.h"
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

constexpr absl::string_view kv_cache_slice_k_root_name = "kv_slice_k_";
constexpr absl::string_view kv_cache_slice_v_root_name = "kv_slice_v_";

namespace {

using LogitsQuantizationParams =
    LlmLiteRtNpuCompiledModelExecutor::LogitsQuantizationParams;

}  // namespace

#define NPU_EXECUTOR_LOG(X) ABSL_LOG_IF(X, npu_config_.enable_npu_debug_logging)

// Signature names for the embedder.
struct EmbedderSignatures {
  static constexpr absl::string_view kPrefillEmbedder = "prefill_embedder_128";
  static constexpr absl::string_view kDecodeEmbedder = "decode_embedder";
  static constexpr absl::string_view kVerifyEmbedder = "verify_embedder";
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
  static constexpr absl::string_view kVerifyEmbedderPerLayer =
      "verify_per_layer_embedder";
  // Prefill and decode use identical tensor signature names.
  static constexpr absl::string_view kEmbedderInput = "token_ids";
  static constexpr absl::string_view kEmbedderOutput = "embeddings";
};

// Signature names for the mask signatures.
struct MaskSignatures {
  static constexpr absl::string_view kPrefillMask = "prefill_mask_128";
  static constexpr absl::string_view kDecodeMask = "decode_mask";
  static constexpr absl::string_view kVerifyMask = "verify_mask";
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
  static constexpr absl::string_view kPrefillLlm = "prefill_128";
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
  static constexpr absl::string_view kPrefillCacheUpdate =
      "prefill_cache_update_128";
  static constexpr absl::string_view kDecodeCacheUpdate = "decode_cache_update";
  static constexpr absl::string_view kVerifyCacheUpdate = "verify_cache_update";
  static constexpr absl::string_view kInputPos = "input_pos";
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

// Dequantize logits to float32.
// This is required because `DecodeToLogits` must return float logits to the
// caller, but NPU models may return quantized logits.
absl::Status DequantizeLogits(const ::litert::TensorBuffer& src,
                              ::litert::TensorBuffer& dst, float scale,
                              int32_t zero_point, bool should_dump) {
  LITERT_ASSIGN_OR_RETURN(auto src_type, src.TensorType());
  LITERT_ASSIGN_OR_RETURN(auto dst_type, dst.TensorType());
  RET_CHECK_EQ((int)dst_type.ElementType(),
               (int)::litert::ElementType::Float32);

  LITERT_ASSIGN_OR_RETURN(size_t num_elements, src_type.Layout().NumElements());

  const auto src_elem_type = src_type.ElementType();

  LITERT_ASSIGN_OR_RETURN(auto src_lock,
                          ::litert::TensorBufferScopedLock::Create(
                              const_cast<::litert::TensorBuffer&>(src),
                              ::litert::TensorBuffer::LockMode::kRead));
  LITERT_ASSIGN_OR_RETURN(auto dst_lock,
                          ::litert::TensorBufferScopedLock::Create(
                              dst, ::litert::TensorBuffer::LockMode::kWrite));

  float* dst_ptr = static_cast<float*>(dst_lock.second);
  const void* src_raw_ptr = src_lock.second;

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
  } else {
    return absl::InvalidArgumentError(absl::StrCat(
        "Unsupported source type for dequantization: ", (int)src_elem_type));
  }

  return absl::OkStatus();
}

}  // namespace

std::ostream& operator<<(
    std::ostream& os,
    const LlmLiteRtNpuCompiledModelExecutor::LatencyStats& stats) {
  os << "\n" << "====== PREFILL STATS ======";
  os << "\n" << "Total prefill latency [us]: " << stats.prefill_e2e_latency_us;
  os << "\n" << "(e2e) Prefill num tokens: " << stats.prefill_num_tokens;
  os << "\n"
     << "(e2e) Prefill tokens per second: "
     << ((stats.prefill_num_tokens * 1000 * 1000) /
         (float)stats.prefill_e2e_latency_us);
  os << "\n"
     << "(TransformerStackOnly) Prefill tokens per second: "
     << ((stats.prefill_num_tokens * 1000 * 1000) /
         (float)stats.prefill_llm_inference_latency_us);

  os << "\n" << "------ Prefill breakdown ------";
  os << "\n"
     << "Total prefill prepare input tensors latency [us]: "
     << stats.prefill_prepare_input_latency_us << " ("
     << ((stats.prefill_prepare_input_latency_us * 100) /
         (float)stats.prefill_e2e_latency_us)
     << "%)";
  os << "\n"
     << "Total prefill embedder inference latency [us]: "
     << stats.prefill_embedder_inference_latency_us << " ("
     << ((stats.prefill_embedder_inference_latency_us * 100) /
         (float)stats.prefill_e2e_latency_us)
     << "%)";
  if (stats.prefill_embedder_per_layer_inference_latency_us > 0) {
    os << "\n"
       << "Total prefill embedder per layer inference latency [us]: "
       << stats.prefill_embedder_per_layer_inference_latency_us << " ("
       << ((stats.prefill_embedder_per_layer_inference_latency_us * 100) /
           (float)stats.prefill_e2e_latency_us)
       << "%)";
  }
  os << "\n"
     << "Total prefill rope inference latency [us]: "
     << stats.prefill_rope_inference_latency_us << " ("
     << ((stats.prefill_rope_inference_latency_us * 100) /
         (float)stats.prefill_e2e_latency_us)
     << "%)";
  os << "\n"
     << "Total prefill mask inference latency [us]: "
     << stats.prefill_mask_inference_latency_us << " ("
     << ((stats.prefill_mask_inference_latency_us * 100) /
         (float)stats.prefill_e2e_latency_us)
     << "%)";
  os << "\n"
     << "Total prefill llm inference latency [us]: "
     << stats.prefill_llm_inference_latency_us << " ("
     << ((stats.prefill_llm_inference_latency_us * 100) /
         (float)stats.prefill_e2e_latency_us)
     << "%)";
  os << "\n"
     << "Total prefill cache update inference latency [us]: "
     << stats.prefill_cache_update_inference_latency_us << " ("
     << ((stats.prefill_cache_update_inference_latency_us * 100) /
         (float)stats.prefill_e2e_latency_us)
     << "%)";

  os << "\n\n" << "====== DECODE STATS ======";
  os << "\n" << "Total decode latency [us]: " << stats.decode_e2e_latency_us;
  os << "\n" << "(e2e) Decode num tokens: " << stats.decode_num_tokens;
  os << "\n"
     << "(e2e) Decode tokens per second (avg): "
     << ((stats.decode_num_tokens * 1000 * 1000) /
         (float)stats.decode_e2e_latency_us);
  if (stats.mtp_num_draft_tokens > 0) {
    os << "\n"
       << "Speculative decoding acceptance rate [%]: "
       << (float)stats.mtp_num_accepted_tokens / stats.mtp_num_draft_tokens *
              100;
  }
  os << "\n"
     << "(TransformerStackOnly) Decode tokens per second: "
     << ((stats.decode_num_tokens * 1000 * 1000) /
         (float)stats.decode_llm_inference_latency_us);

  os << "\n" << "------ Decode breakdown ------";
  os << "\n"
     << "Total decode prepare input tensors latency [us]: "
     << stats.decode_prepare_input_latency_us << " ("
     << ((stats.decode_prepare_input_latency_us * 100) /
         (float)stats.decode_e2e_latency_us)
     << "%)";
  os << "\n"
     << "Total decode embedder inference latency [us]: "
     << stats.decode_embedder_inference_latency_us << " ("
     << ((stats.decode_embedder_inference_latency_us * 100) /
         (float)stats.decode_e2e_latency_us)
     << "%)";
  if (stats.decode_embedder_per_layer_inference_latency_us > 0) {
    os << "\n"
       << "Total decode embedder per layer inference latency [us]: "
       << stats.decode_embedder_per_layer_inference_latency_us << " ("
       << ((stats.decode_embedder_per_layer_inference_latency_us * 100) /
           (float)stats.decode_e2e_latency_us)
       << "%)";
  }
  os << "\n"
     << "Total decode rope inference latency [us]: "
     << stats.decode_rope_inference_latency_us << " ("
     << ((stats.decode_rope_inference_latency_us * 100) /
         (float)stats.decode_e2e_latency_us)
     << "%)";
  os << "\n"
     << "Total decode mask inference latency [us]: "
     << stats.decode_mask_inference_latency_us << " ("
     << ((stats.decode_mask_inference_latency_us * 100) /
         (float)stats.decode_e2e_latency_us)
     << "%)";
  os << "\n"
     << "Total decode llm inference latency [us]: "
     << stats.decode_llm_inference_latency_us << " ("
     << ((stats.decode_llm_inference_latency_us * 100) /
         (float)stats.decode_e2e_latency_us)
     << "%)";
  os << "\n"
     << "Total decode cache update inference latency [us]: "
     << stats.decode_cache_update_inference_latency_us << " ("
     << ((stats.decode_cache_update_inference_latency_us * 100) /
         (float)stats.decode_e2e_latency_us)
     << "%)";
  os << "\n"
     << "Total decode sampling latency [us]: "
     << stats.decode_sampling_latency_us << " ("
     << ((stats.decode_sampling_latency_us * 100) /
         (float)stats.decode_e2e_latency_us)
     << "%)";
  if (stats.decode_mtp_rejection_sampling_latency_us > 0) {
    os << "\n"
       << "Total decode MTP rejection sampling latency [us]: "
       << stats.decode_mtp_rejection_sampling_latency_us << " ("
       << ((stats.decode_mtp_rejection_sampling_latency_us * 100) /
           (float)stats.decode_e2e_latency_us)
       << "%)";
  }
  if (stats.decode_mtp_activation_copy_latency_us > 0) {
    os << "\n"
       << "Total decode MTP activation copy latency [us]: "
       << stats.decode_mtp_activation_copy_latency_us << " ("
       << ((stats.decode_mtp_activation_copy_latency_us * 100) /
           (float)stats.decode_e2e_latency_us)
       << "%)";
  }
  os << "\n"
     << "Total decode token queue latency [us]: "
     << stats.decode_token_queue_latency_us << " ("
     << ((stats.decode_token_queue_latency_us * 100) /
         (float)stats.decode_e2e_latency_us)
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
    const ::litert::TensorBuffer& prefill_input_tokens,
    const ::litert::TensorBuffer& decode_input_tokens,
    const ::litert::TensorBuffer& verify_input_tokens,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        gemma_prefill_input_buffers,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        gemma_decode_input_buffers,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        gemma_verify_input_buffers,
    const LlmExecutorSettings& settings) {
  LITERT_ASSIGN_OR_RETURN(auto options, CreateLiteRtCpuOptions(settings));
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
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      verify_input_buffers;
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      verify_output_buffers;

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

  if (embedder_compiled_model.FindSignature(
          EmbedderSignatures::kVerifyEmbedder)) {
    LITERT_ASSIGN_OR_RETURN(
        verify_input_buffers[EmbedderSignatures::kEmbedderInput],
        verify_input_tokens.Duplicate());

    LITERT_ASSIGN_OR_RETURN(
        verify_output_buffers[EmbedderSignatures::kEmbedderOutput],
        gemma_verify_input_buffers[LlmSignatures::kInputEmbeddings]
            .Duplicate());
  }

  return EmbedderContext(
      std::move(embedder_compiled_model), std::move(prefill_input_buffers),
      std::move(prefill_output_buffers), std::move(decode_input_buffers),
      std::move(decode_output_buffers), std::move(verify_input_buffers),
      std::move(verify_output_buffers));
}

absl::StatusOr<LlmLiteRtNpuCompiledModelExecutor::EmbedderPerLayerContext>
LlmLiteRtNpuCompiledModelExecutor::
    CreateEmbedderPerLayerContextWithBufferSharing(
        ::litert::Environment& env, const litert::Model& embedder_model,
        const ::litert::TensorBuffer& prefill_input_tokens,
        const ::litert::TensorBuffer& decode_input_tokens,
        const ::litert::TensorBuffer& verify_input_tokens,
        absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
            gemma_prefill_input_buffers,
        absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
            gemma_decode_input_buffers,
        absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
            gemma_verify_input_buffers,
        const LlmExecutorSettings& settings) {
  LITERT_ASSIGN_OR_RETURN(auto options, CreateLiteRtCpuOptions(settings));
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
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      verify_input_buffers;
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      verify_output_buffers;

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
      std::move(embedder_compiled_model), std::move(prefill_input_buffers),
      std::move(prefill_output_buffers), std::move(decode_input_buffers),
      std::move(decode_output_buffers), std::move(verify_input_buffers),
      std::move(verify_output_buffers));
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
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        gemma_prefill_input_buffers,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        gemma_decode_input_buffers,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        gemma_verify_input_buffers) {
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      prefill_input_buffers;
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      prefill_output_buffers;
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      decode_input_buffers;
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      decode_output_buffers;
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      verify_input_buffers;
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      verify_output_buffers;

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
    if (gemma_prefill_input_buffers.contains(mask_output_name)) {
      LITERT_ASSIGN_OR_RETURN(
          prefill_output_buffers[mask_output_name],
          gemma_prefill_input_buffers[mask_output_name].Duplicate());
    }
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
    if (gemma_decode_input_buffers.contains(mask_output_name)) {
      LITERT_ASSIGN_OR_RETURN(
          decode_output_buffers[mask_output_name],
          gemma_decode_input_buffers[mask_output_name].Duplicate());
    }
  }

  if (npu_auxiliary_context.npu_auxiliary_compiled_model.FindSignature(
          MaskSignatures::kVerifyMask)) {
    LITERT_ASSIGN_OR_RETURN(
        verify_input_buffers[MaskSignatures::kMaskInputTimeStep],
        npu_auxiliary_context.npu_auxiliary_compiled_model.CreateInputBuffer(
            MaskSignatures::kVerifyMask, MaskSignatures::kMaskInputTimeStep));
    verify_input_buffers[MaskSignatures::kMaskInputTimeStep].Clear();
    LITERT_ASSIGN_OR_RETURN(
        verify_input_buffers[MaskSignatures::kMaskInputTokens],
        npu_auxiliary_context.npu_auxiliary_compiled_model.CreateInputBuffer(
            MaskSignatures::kVerifyMask, MaskSignatures::kMaskInputTokens));
    verify_input_buffers[MaskSignatures::kMaskInputTokens].Clear();

    for (const auto& mask_output_name : mask_output_names) {
      if (gemma_verify_input_buffers.contains(mask_output_name)) {
        LITERT_ASSIGN_OR_RETURN(
            verify_output_buffers[mask_output_name],
            gemma_verify_input_buffers[mask_output_name].Duplicate());
      }
    }
  }
  return InferenceContext(
      std::move(prefill_input_buffers), std::move(prefill_output_buffers),
      std::move(decode_input_buffers), std::move(decode_output_buffers),
      std::move(verify_input_buffers), std::move(verify_output_buffers));
}

absl::StatusOr<LlmLiteRtNpuCompiledModelExecutor::InferenceContext>
LlmLiteRtNpuCompiledModelExecutor::CreateRopeContextWithBufferSharing(
    const NpuAuxiliaryContext& npu_auxiliary_context,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        gemma_prefill_input_buffers,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        gemma_decode_input_buffers,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        gemma_verify_input_buffers) {
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      prefill_input_buffers;
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      prefill_output_buffers;
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      decode_input_buffers;
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      decode_output_buffers;
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      verify_input_buffers;
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      verify_output_buffers;

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
    if (gemma_prefill_input_buffers.contains(rope_output_name)) {
      LITERT_ASSIGN_OR_RETURN(
          prefill_output_buffers[rope_output_name],
          gemma_prefill_input_buffers[rope_output_name].Duplicate());
    }
  }

  LITERT_ASSIGN_OR_RETURN(
      decode_input_buffers[RopeSignatures::kInputPos],
      npu_auxiliary_context.npu_auxiliary_compiled_model.CreateInputBuffer(
          RopeSignatures::kDecodeRope, RopeSignatures::kInputPos));
  decode_input_buffers[RopeSignatures::kInputPos].Clear();

  for (const auto& rope_output_name : rope_output_names) {
    if (gemma_decode_input_buffers.contains(rope_output_name)) {
      LITERT_ASSIGN_OR_RETURN(
          decode_output_buffers[rope_output_name],
          gemma_decode_input_buffers[rope_output_name].Duplicate());
    }
  }

  if (npu_auxiliary_context.npu_auxiliary_compiled_model.FindSignature(
          RopeSignatures::kVerifyRope)) {
    LITERT_ASSIGN_OR_RETURN(
        verify_input_buffers[RopeSignatures::kInputPos],
        npu_auxiliary_context.npu_auxiliary_compiled_model.CreateInputBuffer(
            RopeSignatures::kVerifyRope, RopeSignatures::kInputPos));
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
      std::move(prefill_input_buffers), std::move(prefill_output_buffers),
      std::move(decode_input_buffers), std::move(decode_output_buffers),
      std::move(verify_input_buffers), std::move(verify_output_buffers));
}

absl::Status LlmLiteRtNpuCompiledModelExecutor::AllocateTransformerBuffers(
    litert::Environment& env, const litert::Model* transformer_model,
    CompiledModel& llm_compiled_model,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        gemma_prefill_input_buffers,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        gemma_decode_input_buffers,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        gemma_verify_input_buffers,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        input_kv_cache_buffers,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        prefill_output_kv_cache_slice_buffers,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        decode_output_kv_cache_slice_buffers,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        verify_output_kv_cache_slice_buffers) {
  auto prefill_signature = transformer_model->FindSignature(kPrefillSignature);

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

  // Create input/output buffers for verify signature if it exists.
  auto verify_signature =
      transformer_model->FindSignature(LlmSignatures::kVerifyLlm);
  if (verify_signature) {
    for (auto input_name : verify_signature->InputNames()) {
      LITERT_ASSIGN_OR_RETURN(gemma_verify_input_buffers[input_name],
                              llm_compiled_model.CreateInputBuffer(
                                  LlmSignatures::kVerifyLlm, input_name));
      gemma_verify_input_buffers[input_name].Clear();
    }
    for (auto output_name : verify_signature->OutputNames()) {
      if (absl::StartsWith(output_name, kv_cache_slice_k_root_name) ||
          absl::StartsWith(output_name, kv_cache_slice_v_root_name)) {
        LITERT_ASSIGN_OR_RETURN(
            verify_output_kv_cache_slice_buffers[output_name],
            llm_compiled_model.CreateOutputBuffer(LlmSignatures::kVerifyLlm,
                                                  output_name));
      }
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
        verify_output_kv_cache_slice_buffers,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        gemma_prefill_input_buffers,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        gemma_decode_input_buffers,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        gemma_verify_input_buffers) {
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

    LITERT_ASSIGN_OR_RETURN(
        auto decode_output_names,
        llm_compiled_model.GetSignatureOutputNames(kDecodeSignature));

    for (const auto& name : decode_output_names) {
      if (name == LlmSignatures::kDecodeLogitsOutput) {
        LITERT_ASSIGN_OR_RETURN(
            decode_output_buffers[LlmSignatures::kDecodeLogitsOutput],
            llm_compiled_model.CreateOutputBuffer(
                kDecodeSignature, LlmSignatures::kDecodeLogitsOutput));
      } else if (name == LlmSignatures::kLastLayerActivationsOutput) {
        LITERT_ASSIGN_OR_RETURN(
            decode_output_buffers[LlmSignatures::kLastLayerActivationsOutput],
            llm_compiled_model.CreateOutputBuffer(
                kDecodeSignature, LlmSignatures::kLastLayerActivationsOutput));
      }
    }
  }

  auto verify_signature =
      llm_compiled_model.FindSignature(LlmSignatures::kVerifyLlm);
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      verify_input_buffers;
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      verify_output_buffers;
  if (verify_signature) {
    for (const auto& [key, value] : gemma_verify_input_buffers) {
      LITERT_ASSIGN_OR_RETURN(verify_input_buffers[key], value.Duplicate());
    }
    // Duplicate all kv cache buffers to verify inputs.
    LITERT_ASSIGN_OR_RETURN(
        auto verify_input_names,
        llm_compiled_model.GetSignatureInputNames(LlmSignatures::kVerifyLlm));
    for (const auto& [key, value] : input_kv_cache_buffers) {
      if (absl::c_find(verify_input_names, std::string(key)) !=
          verify_input_names.end()) {
        LITERT_ASSIGN_OR_RETURN(verify_input_buffers[key], value.Duplicate());
      }
    }

    for (const auto& [key, value] : verify_output_kv_cache_slice_buffers) {
      LITERT_ASSIGN_OR_RETURN(verify_output_buffers[key], value.Duplicate());
    }

    LITERT_ASSIGN_OR_RETURN(
        auto verify_output_names,
        llm_compiled_model.GetSignatureOutputNames(LlmSignatures::kVerifyLlm));

    for (const auto& name : verify_output_names) {
      if (name == LlmSignatures::kDecodeLogitsOutput) {
        LITERT_ASSIGN_OR_RETURN(
            verify_output_buffers[LlmSignatures::kDecodeLogitsOutput],
            llm_compiled_model.CreateOutputBuffer(
                LlmSignatures::kVerifyLlm, LlmSignatures::kDecodeLogitsOutput));
      } else if (name == LlmSignatures::kLastLayerActivationsOutput) {
        LITERT_ASSIGN_OR_RETURN(
            verify_output_buffers[LlmSignatures::kLastLayerActivationsOutput],
            llm_compiled_model.CreateOutputBuffer(
                LlmSignatures::kVerifyLlm,
                LlmSignatures::kLastLayerActivationsOutput));
      }
    }
  }

  return InferenceContext(
      std::move(prefill_input_buffers), std::move(prefill_output_buffers),
      std::move(decode_input_buffers), std::move(decode_output_buffers),
      std::move(verify_input_buffers), std::move(verify_output_buffers));
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
        absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
            verify_output_kv_cache_slice_buffers,
        ::litert::TensorBuffer prefill_input_pos,
        ::litert::TensorBuffer decode_input_pos,
        ::litert::TensorBuffer verify_input_pos)

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
  }
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      verify_output_buffers;
  {
    for (const auto& [key, value] : input_kv_cache_buffers) {
      LITERT_ASSIGN_OR_RETURN(verify_output_buffers[key], value.Duplicate());
    }
  }

  return InferenceContext(
      std::move(prefill_input_buffers), std::move(prefill_output_buffers),
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

  // Warmup verify signatures if they exist.
  if (compiled_model_llm.FindSignature(LlmSignatures::kVerifyLlm)) {
    result = compiled_model_llm.Run(
        LlmSignatures::kVerifyLlm, llm_inference_context.verify_input_buffers,
        llm_inference_context.verify_output_buffers);
    RET_CHECK(result) << "Inference warmup run for MTP verify failed."
                      << result.Error().Message();
  }

  if (compiled_model_auxiliary.FindSignature(RopeSignatures::kVerifyRope)) {
    result = compiled_model_auxiliary.Run(
        RopeSignatures::kVerifyRope,
        rope_inference_context.verify_input_buffers,
        rope_inference_context.verify_output_buffers);
    RET_CHECK(result)
        << "Inference warmup run for RoPE signature (verify) failed."
        << result.Error().Message();
  }

  if (compiled_model_auxiliary.FindSignature(MaskSignatures::kVerifyMask)) {
    result = compiled_model_auxiliary.Run(
        MaskSignatures::kVerifyMask,
        mask_inference_context.verify_input_buffers,
        mask_inference_context.verify_output_buffers);
    RET_CHECK(result)
        << "Inference warmup run for mask signature (verify) failed."
        << result.Error().Message();
  }

  if (compiled_model_auxiliary.FindSignature(
          CacheUpdateSignatures::kVerifyCacheUpdate)) {
    result = compiled_model_auxiliary.Run(
        CacheUpdateSignatures::kVerifyCacheUpdate,
        cache_update_inference_context.verify_input_buffers,
        cache_update_inference_context.verify_output_buffers);
    RET_CHECK(result)
        << "Inference warmup run for cache update signature (verify) failed."
        << result.Error().Message();
  }

  // Clear the KV cache buffers after warmup.
  RETURN_IF_ERROR(ClearKVCache(llm_inference_context.prefill_input_buffers));
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
    absl::flat_hash_map<absl::string_view, TensorBuffer> prefill_input_buffers,
    absl::flat_hash_map<absl::string_view, TensorBuffer> prefill_output_buffers,
    absl::flat_hash_map<absl::string_view, TensorBuffer> decode_input_buffers,
    absl::flat_hash_map<absl::string_view, TensorBuffer> decode_output_buffers,
    absl::flat_hash_map<absl::string_view, TensorBuffer> verify_input_buffers,
    absl::flat_hash_map<absl::string_view, TensorBuffer> verify_output_buffers)
    : prefill_input_buffers(std::move(prefill_input_buffers)),
      prefill_output_buffers(std::move(prefill_output_buffers)),
      decode_input_buffers(std::move(decode_input_buffers)),
      decode_output_buffers(std::move(decode_output_buffers)),
      verify_input_buffers(std::move(verify_input_buffers)),
      verify_output_buffers(std::move(verify_output_buffers)) {}

LlmLiteRtNpuCompiledModelExecutor::EmbedderContext::EmbedderContext(
    CompiledModel embedder_compiled_model,
    absl::flat_hash_map<absl::string_view, TensorBuffer> prefill_input_buffers,
    absl::flat_hash_map<absl::string_view, TensorBuffer> prefill_output_buffers,
    absl::flat_hash_map<absl::string_view, TensorBuffer> decode_input_buffers,
    absl::flat_hash_map<absl::string_view, TensorBuffer> decode_output_buffers,
    absl::flat_hash_map<absl::string_view, TensorBuffer> verify_input_buffers,
    absl::flat_hash_map<absl::string_view, TensorBuffer> verify_output_buffers)
    : embedder_compiled_model(std::move(embedder_compiled_model)),
      inference_context(
          std::move(prefill_input_buffers), std::move(prefill_output_buffers),
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
  auto start = absl::Now();
  LITERT_ASSIGN_OR_RETURN(const auto* text_token_ids,
                          inputs.GetTextTokenIdsPtr());
  LITERT_ASSIGN_OR_RETURN(auto tensor_type, text_token_ids->TensorType());
  // Only accept batch size 1 for now.
  RET_CHECK_EQ(tensor_type.Layout().Dimensions()[0], 1);
  RET_CHECK_GT(tensor_type.Layout().Dimensions()[1], 0)
      << "Prefill token ids must be non-empty.";
  if (UseEmbeddingLookupManager()) {
    RETURN_IF_ERROR(
        embedding_lookup_manager_->UpdateMultiModalEmbeddings(inputs));
  }
  LITERT_ASSIGN_OR_RETURN(auto ids,
                          ReferTensorBufferAsSpan<int32_t>(*text_token_ids));

  LITERT_ASSIGN_OR_RETURN(
      auto work_groups,
      GetOptimizedPrefillWorkGroups(prefill_signature_map_, ids.size()));
  for (const auto& [prefill_signature, prefill_length] : work_groups) {
    RETURN_IF_ERROR(PrefillInternal(prefill_signature,
                                    ids.subspan(/*pos=*/0, prefill_length)));
    ids = ids.subspan(/*pos=*/prefill_length);
    latency_stats_.prefill_num_tokens += prefill_length;
  }
  RET_CHECK_EQ(ids.size(), 0).SetCode(absl::StatusCode::kInternal)
      << "Work groups not covering the entire prefill input.";

  if (UseEmbeddingLookupManager()) {
    RETURN_IF_ERROR(embedding_lookup_manager_->CleanupMultiModalEmbeddings());
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
      RETURN_IF_ERROR(embedding_lookup_manager_->LookupDecode(
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
    RETURN_IF_ERROR(
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
    RETURN_IF_ERROR(
        DecodeInternal(internal_start_step, pending_input_token[0]));

    auto start_mark = absl::Now();
    RETURN_IF_ERROR(processed_tokens_.MarkPendingInputTokenAsProcessed());
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
    RETURN_IF_ERROR(processed_tokens_.MarkPendingInputTokenAsProcessed());
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
    RETURN_IF_ERROR(
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
    RETURN_IF_ERROR(CommitVerifiedKVCache(mtp_start_step));
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
      RETURN_IF_ERROR(embedding_lookup_manager_->LookupDecode(
          first_token->id(), first_token->mutable_embedding()));
      latency_stats_.decode_embedder_inference_latency_us +=
          absl::ToInt64Microseconds(absl::Now() - start_lookup);
    }
    // For MTP, we need to mark them as processed so the next step's
    // GetNextUnprocessedToken works correctly.
    RETURN_IF_ERROR(
        processed_tokens_.AddPendingInputToken({std::move(first_token)}));
    current_step_++;

    latency_stats_.decode_num_tokens++;
    latency_stats_.decode_e2e_latency_us +=
        absl::ToInt64Microseconds(absl::Now() - start);
    return std::vector<std::vector<int>>{{first_token_id}};

  } else {
    // Standard Non-Speculative path.
    ::litert::TensorBuffer& decoded_logits =
        llm_inference_context_
            .decode_output_buffers[LlmSignatures::kDecodeLogitsOutput];
    auto start_sample = absl::Now();
    LITERT_ASSIGN_OR_RETURN(
        const int max_index,
        ApplyGreedySampling(decoded_logits,
                            npu_config_.enable_neon_for_npu_greedy_sampling));
    latency_stats_.decode_sampling_latency_us +=
        absl::ToInt64Microseconds(absl::Now() - start_sample);

    // Store the sampled id as the pending input token for next Decode.

    std::shared_ptr<TokenData> last_output_token =
        std::make_shared<TokenData>(max_index);

    if (UseEmbeddingLookupManager()) {
      auto start_lookup = absl::Now();
      RETURN_IF_ERROR(embedding_lookup_manager_->LookupDecode(
          last_output_token->id(), last_output_token->mutable_embedding()));
      latency_stats_.decode_embedder_inference_latency_us +=
          absl::ToInt64Microseconds(absl::Now() - start_lookup);
    }
    // For Gemma3 we don't need to do anything here because we invoke
    // the Embedder before invoking the transformer during prefill/decode. All
    // we need to do is keep the token id around (which is stored as the pending
    // token).

    auto start_add = absl::Now();
    RETURN_IF_ERROR(
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
      RETURN_IF_ERROR(embedding_lookup_manager_->LookupPrefill(
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
    RETURN_IF_ERROR(embedding_lookup_manager_->LookupPrefill(
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
  if (use_hw_ple_for_npu_ && !ple_table_ptrs_.empty()) {
    auto start = absl::Now();
    auto& ple_output_buffer =
        llm_inference_context_.prefill_input_buffers[kPerLayerEmbedderTensor];
    LITERT_ASSIGN_OR_RETURN(
        auto lock,
        ::litert::TensorBufferScopedLock::Create(
            ple_output_buffer, ::litert::TensorBuffer::LockMode::kWrite));
    void* output_ptr = lock.second;

    RETURN_IF_ERROR(HWPerLayerEmbeddingLookup(
        ids.data(), ids.size(), ple_table_ptrs_.data(),
        ple_quant_params_.data(), num_tables_, 256, output_ptr, output_type_,
        final_scale_, final_zero_point_));

    latency_stats_.prefill_embedder_per_layer_inference_latency_us +=
        absl::ToInt64Microseconds(absl::Now() - start);
  } else if (embedder_per_layer_context_.has_value()) {
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
    latency_stats_.prefill_embedder_per_layer_inference_latency_us +=
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
    if (prefill_mask_update_method_ == MaskUpdateMethod::kWH) {
      RETURN_IF_ERROR(HWMaskUpdate(mask_context_.prefill_input_buffers,
                                   mask_context_.prefill_output_buffers));
    } else {
      auto res = npu_auxiliary_context_.npu_auxiliary_compiled_model.Run(
          MaskSignatures::kPrefillMask, mask_context_.prefill_input_buffers,
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
    if (prefill_kv_cache_update_method_ == KVCacheUpdateMethod::kWH) {
      RETURN_IF_ERROR(HWKVCacheUpdate(
          cache_update_inference_context_.prefill_input_buffers,
          cache_update_inference_context_.prefill_output_buffers));
    } else {
      auto res = npu_auxiliary_context_.npu_auxiliary_compiled_model.Run(
          CacheUpdateSignatures::kPrefillCacheUpdate,
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
  int id = token->id();
  auto start_prepare_inputs = absl::Now();

  {
    if (id == -1) {
      return absl::InvalidArgumentError("No id available to be decoded.");
    }

    // Decode input tokens.
    RETURN_IF_ERROR(SetFirstElement(
        embedder_context_.inference_context
            .decode_input_buffers[EmbedderSignatures::kEmbedderInput],
        id));

    // Always update decode input position and timestep, even if
    // run_rope_and_mask is false. The LLM and Cache Update models still need to
    // know the current step.

    // 1. RoPE position
    RETURN_IF_ERROR(SetFirstElement(
        rope_context_.decode_input_buffers[RopeSignatures::kInputPos], step));

    // 2. Mask timestep
    RETURN_IF_ERROR(SetFirstElement(
        mask_context_.decode_input_buffers[MaskSignatures::kMaskInputTimeStep],
        step));

    // 3. Cache update position
    RETURN_IF_ERROR(SetFirstElement(
        cache_update_inference_context_
            .decode_input_buffers[CacheUpdateSignatures::kInputPos],
        step));
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
    LITERT_RETURN_IF_ERROR(!token->embedding().empty())
        << "Token embedding is empty.";
    memcpy(transformer_embedding_buffer_ptr, token->embedding().data(),
           token->embedding().size() * sizeof(float));

    // Log the input embedding for comparison

    latency_stats_.decode_embedder_inference_latency_us +=
        absl::ToInt64Microseconds(absl::Now() - start);
  }

  {
    if (use_hw_ple_for_npu_ && !ple_table_ptrs_.empty()) {
      auto start = absl::Now();
      auto& ple_output_buffer =
          llm_inference_context_.decode_input_buffers[kPerLayerEmbedderTensor];
      LITERT_ASSIGN_OR_RETURN(
          auto lock,
          ::litert::TensorBufferScopedLock::Create(
              ple_output_buffer, ::litert::TensorBuffer::LockMode::kWrite));
      void* output_ptr = lock.second;

      int id = token->id();
      RETURN_IF_ERROR(HWPerLayerEmbeddingLookup(
          &id, 1, ple_table_ptrs_.data(), ple_quant_params_.data(), num_tables_,
          256, output_ptr, output_type_, final_scale_, final_zero_point_));

      latency_stats_.decode_embedder_per_layer_inference_latency_us +=
          absl::ToInt64Microseconds(absl::Now() - start);
    } else if (embedder_per_layer_context_.has_value()) {
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
      latency_stats_.decode_embedder_per_layer_inference_latency_us +=
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
    if (decode_mask_update_method_ == MaskUpdateMethod::kWH) {
      RETURN_IF_ERROR(HWMaskUpdate(mask_context_.decode_input_buffers,
                                   mask_context_.decode_output_buffers));
    } else {
      auto res = npu_auxiliary_context_.npu_auxiliary_compiled_model.Run(
          MaskSignatures::kDecodeMask, mask_context_.decode_input_buffers,
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
    if (decode_kv_cache_update_method_ == KVCacheUpdateMethod::kWH) {
      RETURN_IF_ERROR(HWKVCacheUpdate(
          cache_update_inference_context_.decode_input_buffers,
          cache_update_inference_context_.decode_output_buffers));
    } else {
      auto res = npu_auxiliary_context_.npu_auxiliary_compiled_model.Run(
          CacheUpdateSignatures::kDecodeCacheUpdate,
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
    RETURN_IF_ERROR(
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
// Helper to sample from a batch of logits at a specific index.
absl::StatusOr<int> GetLogitsAtBatchIndex(const TensorBuffer& logits_buffer,
                                          int batch_idx,
                                          bool enable_neon_sampling) {
  LITERT_ASSIGN_OR_RETURN(RankedTensorType tensor_type,
                          logits_buffer.TensorType());
  LITERT_ASSIGN_OR_RETURN(
      auto lock_and_addr,
      ::litert::TensorBufferScopedLock::Create(
          logits_buffer, ::litert::TensorBuffer::LockMode::kRead));

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

  const uint8_t* base_ptr = static_cast<const uint8_t*>(lock_and_addr.second);
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

  if (tensor_type.ElementType() == ::litert::ElementType::Float32) {
#if defined(__ANDROID__) && defined(__ARM_NEON)
    if (enable_neon_sampling) {
      return FindMaxIndexFloatNeon(reinterpret_cast<const float*>(logits_ptr),
                                   vocab_size);
    }
#endif
    return find_max_index_plain(reinterpret_cast<const float*>(logits_ptr));
  } else if (tensor_type.ElementType() == ::litert::ElementType::Int16) {
#if defined(__ANDROID__) && defined(__ARM_NEON)
    if (enable_neon_sampling) {
      return FindMaxIndexInt16Neon(reinterpret_cast<const int16_t*>(logits_ptr),
                                   vocab_size);
    }
#endif
    return find_max_index_plain(reinterpret_cast<const int16_t*>(logits_ptr));
  } else if (tensor_type.ElementType() == ::litert::ElementType::Int8) {
#if defined(__ANDROID__) && defined(__ARM_NEON)
    if (enable_neon_sampling) {
      return FindMaxIndexInt8Neon(reinterpret_cast<const int8_t*>(logits_ptr),
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

    RETURN_IF_ERROR(HWPerLayerEmbeddingLookup(
        verify_ids.data(), verify_ids.size(), ple_table_ptrs_.data(),
        ple_quant_params_.data(), num_tables_, 256, output_ptr, output_type_,
        final_scale_, final_zero_point_));

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
  int bonus_token_id = -1;

  // Log all sampled tokens from the verifier for transparency.
  std::vector<int> all_verifier_sampled;
  for (int i = 0; i < draft_tokens.size() + 1; ++i) {
    LITERT_ASSIGN_OR_RETURN(
        int sampled_token,
        GetLogitsAtBatchIndex(verifier_logits_buffer, i,
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
    LITERT_ASSIGN_OR_RETURN(
        bonus_token_id,
        GetLogitsAtBatchIndex(verifier_logits_buffer, num_accepted,
                              npu_config_.enable_neon_for_npu_greedy_sampling));
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
    RETURN_IF_ERROR(
        HWKVCacheUpdate(cache_update_inference_context_.verify_input_buffers,
                        cache_update_inference_context_.verify_output_buffers));
  } else {
    LITERT_RETURN_IF_ERROR(
        npu_auxiliary_context_.npu_auxiliary_compiled_model.Run(
            CacheUpdateSignatures::kVerifyCacheUpdate,
            cache_update_inference_context_.verify_input_buffers,
            cache_update_inference_context_.verify_output_buffers));
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

const LlmLiteRtNpuCompiledModelExecutor::LatencyStats&
LlmLiteRtNpuCompiledModelExecutor::GetLatencyStats() const {
  return latency_stats_;
}

absl::Status LlmLiteRtNpuCompiledModelExecutor::Reset() {
  ABSL_LOG(INFO) << "Custom NPU execution latency stats:\n" << latency_stats_;
  current_step_ = 0;
  RETURN_IF_ERROR(processed_tokens_.RollBackToStep(0));
  sampled_ids_.clear();
  latency_stats_ = {};
  last_verify_activations_.clear();
  pending_accepted_tokens_.clear();

  RETURN_IF_ERROR(ClearKVCache(llm_inference_context_.prefill_input_buffers));
  return absl::OkStatus();
}

// static
absl::StatusOr<std::unique_ptr<LlmLiteRtNpuCompiledModelExecutor>>
LlmLiteRtNpuCompiledModelExecutor::Create(
    const LlmExecutorSettings& executor_settings, ModelResources& resources,
    Environment& env) {
  LITERT_ASSIGN_OR_RETURN(
      const litert::Model* llm_model,
      resources.GetTFLiteModel(ModelType::kTfLitePrefillDecode));

  // Initialize logits quantization parameters using the 'decode' signature.
  LogitsQuantizationParams quantization_params = {.scale = 1.0f,
                                                  .zero_point = 0};
  LITERT_ASSIGN_OR_RETURN(auto decode_signature,
                          llm_model->FindSignature(kDecodeSignature));
  LITERT_ASSIGN_OR_RETURN(
      auto logits_tensor,
      decode_signature.OutputTensor(LlmSignatures::kDecodeLogitsOutput));
  if (logits_tensor.HasQuantization()) {
    auto q_params = logits_tensor.PerTensorQuantization();
    quantization_params.scale = q_params.scale;
    quantization_params.zero_point = static_cast<int32_t>(q_params.zero_point);
    ABSL_LOG(INFO) << "Logits quantization params from '" << kDecodeSignature
                   << "' signature: scale=" << quantization_params.scale
                   << " zero_point=" << quantization_params.zero_point;
  } else {
    ABSL_LOG(WARNING) << "No quantization for logits in '" << kDecodeSignature
                      << "' signature (using default scale= "
                      << quantization_params.scale
                      << ", zero_point= " << quantization_params.zero_point
                      << ").";
  }
  // For the lack of a better way to identify the model variants, we use the
  // presence of per-layer embeddings as the signal for Gemma3n.
  LITERT_ASSIGN_OR_RETURN(const bool has_per_layer_embeddings,
                          HasPerLayerEmbedder(*llm_model));
  if (has_per_layer_embeddings) {
    return CreateForModelHasPerLayerEmbedding(executor_settings, resources, env,
                                              llm_model, quantization_params);
  } else {
    return CreateForModelWithoutPerLayerEmbedding(
        executor_settings, resources, env, llm_model, quantization_params);
  }
};

absl::StatusOr<std::unique_ptr<LlmLiteRtNpuCompiledModelExecutor>>
LlmLiteRtNpuCompiledModelExecutor::CreateForModelHasPerLayerEmbedding(
    const LlmExecutorSettings& executor_settings, ModelResources& resources,
    litert::Environment& env, const litert::Model* transformer_model,
    LogitsQuantizationParams quantization_params) {
  // If the model is fully AOT compiled for NPU, NPU accelerator is used
  // automatically.
  LITERT_ASSIGN_OR_RETURN(auto options,
                          CreateLiteRtNpuOptions(executor_settings));
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
  absl::flat_hash_map<absl::string_view, TensorBuffer>
      gemma_verify_input_buffers;
  absl::flat_hash_map<absl::string_view, TensorBuffer> input_kv_cache_buffers;
  absl::flat_hash_map<absl::string_view, TensorBuffer>
      prefill_output_kv_cache_slice_buffers;
  absl::flat_hash_map<absl::string_view, TensorBuffer>
      decode_output_kv_cache_slice_buffers;
  absl::flat_hash_map<absl::string_view, TensorBuffer>
      verify_output_kv_cache_slice_buffers;

  RETURN_IF_ERROR(AllocateTransformerBuffers(
      env, transformer_model, llm_compiled_model, gemma_prefill_input_buffers,
      gemma_decode_input_buffers, gemma_verify_input_buffers,
      input_kv_cache_buffers, prefill_output_kv_cache_slice_buffers,
      decode_output_kv_cache_slice_buffers,
      verify_output_kv_cache_slice_buffers));

  // Gemma3n specific fix: KV cache buffer 19 of *prefill* is not connected
  // to any OPs in the model, making the LiteRT runtime allocate host memory
  // for it. This is incompatible when running the transformer model on the NPU.
  if (input_kv_cache_buffers.contains(cache_k19)) {
    LITERT_ASSIGN_OR_RETURN(auto buffer_k, llm_compiled_model.CreateInputBuffer(
                                               kDecodeSignature, cache_k19));
    buffer_k.Clear();
    input_kv_cache_buffers[cache_k19] = std::move(buffer_k);

    LITERT_ASSIGN_OR_RETURN(auto buffer_v, llm_compiled_model.CreateInputBuffer(
                                               kDecodeSignature, cache_v19));
    buffer_v.Clear();
    input_kv_cache_buffers[cache_v19] = std::move(buffer_v);
  }
  LITERT_ASSIGN_OR_RETURN(
      auto llm_inference_context,
      CreateLlmInferenceContextWithBufferSharing(
          env, llm_compiled_model, input_kv_cache_buffers,
          prefill_output_kv_cache_slice_buffers,
          decode_output_kv_cache_slice_buffers,
          verify_output_kv_cache_slice_buffers, gemma_prefill_input_buffers,
          gemma_decode_input_buffers, gemma_verify_input_buffers));

  LITERT_ASSIGN_OR_RETURN(auto npu_auxiliary_lrt_model,
                          resources.GetTFLiteModel(ModelType::kTfLiteAux));

  LITERT_ASSIGN_OR_RETURN(
      auto npu_auxiliary_context,
      CreateNpuAuxiliaryContext(env, *npu_auxiliary_lrt_model,
                                executor_settings));

  LITERT_ASSIGN_OR_RETURN(
      auto mask_context,
      CreateMaskContextWithBufferSharing(
          npu_auxiliary_context, gemma_prefill_input_buffers,
          gemma_decode_input_buffers, gemma_verify_input_buffers));

  LITERT_ASSIGN_OR_RETURN(auto embedder_lrt_model,
                          resources.GetTFLiteModel(ModelType::kTfLiteEmbedder));
  LITERT_ASSIGN_OR_RETURN(
      auto embedder_context,
      CreateEmbedderContextWithBufferSharing(
          env, *embedder_lrt_model,
          mask_context.prefill_input_buffers[MaskSignatures::kMaskInputTokens],
          mask_context.decode_input_buffers[MaskSignatures::kMaskInputTokens],
          mask_context.verify_input_buffers[MaskSignatures::kMaskInputTokens],
          gemma_prefill_input_buffers, gemma_decode_input_buffers,
          gemma_verify_input_buffers, executor_settings));

  LITERT_ASSIGN_OR_RETURN(
      auto rope_context,
      CreateRopeContextWithBufferSharing(
          npu_auxiliary_context, gemma_prefill_input_buffers,
          gemma_decode_input_buffers, gemma_verify_input_buffers));

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
  ::litert::TensorBuffer verify_input_pos;
  if (rope_context.verify_input_buffers.contains(RopeSignatures::kInputPos)) {
    LITERT_ASSIGN_OR_RETURN(
        verify_input_pos,
        rope_context.verify_input_buffers[RopeSignatures::kInputPos]
            .Duplicate());
  }

  LITERT_ASSIGN_OR_RETURN(
      auto cache_update_inference_context,
      CreateCacheUpdateInferenceContextWithBufferSharing(
          input_kv_cache_buffers, prefill_output_kv_cache_slice_buffers,
          decode_output_kv_cache_slice_buffers,
          verify_output_kv_cache_slice_buffers, std::move(prefill_input_pos),
          std::move(decode_input_pos), std::move(verify_input_pos)));

  RETURN_IF_ERROR(WarmupInference(
      llm_compiled_model, llm_inference_context,
      npu_auxiliary_context.npu_auxiliary_compiled_model, rope_context,
      mask_context, cache_update_inference_context));

  // For now we only support one prefill length in the model.
  SortedPrefillSignatureMap prefill_runner_set;
  prefill_runner_set[kPrefillSize] = kPrefillSignature;

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

  bool use_hw_ple_for_npu = false;
  auto npu_config_status = executor_settings.GetBackendConfig<NpuConfig>();
  if (npu_config_status.ok()) {
    use_hw_ple_for_npu = npu_config_status->use_hw_ple_for_npu;
  }

  std::optional<EmbedderPerLayerContext> embedder_per_layer_context =
      std::nullopt;

  LITERT_ASSIGN_OR_RETURN(
      const litert::Model* embedder_per_layer_model,
      resources.GetTFLiteModel(ModelType::kTfLitePerLayerEmbedder));

  std::vector<const uint8_t*> ple_table_ptrs;
  std::vector<HWQuantizationParams> ple_quant_params;
  std::vector<float> ple_per_tensor_scales;

  int table_count = 0;
  litert::ElementType output_type = litert::ElementType::None;
  float final_scale = 1.0f;
  int32_t final_zero_point = 0;

  if (use_hw_ple_for_npu) {
    auto extended_model = ExtendedModel::CreateFromNonOwnedHandle(
        embedder_per_layer_model->Get());
    LITERT_ASSIGN_OR_RETURN(auto subgraph, extended_model.MainSubgraph());
    auto ops = subgraph.Ops();
    for (const auto& op : ops) {
      if (op.Code() == kLiteRtOpCodeTflEmbeddingLookup) {
        LITERT_ASSIGN_OR_RETURN(auto table_tensor, op.Input(1));
        auto weights = table_tensor.Weights();
        ple_table_ptrs.push_back(weights.Bytes().data());

        HWQuantizationParams qp;
        qp.scales = nullptr;
        qp.is_per_channel = false;

        if (table_tensor.HasQuantization()) {
          auto q_type = table_tensor.QTypeId();
          if (q_type == kLiteRtQuantizationPerTensor) {
            auto q_params = table_tensor.PerTensorQuantization();
            ple_per_tensor_scales.push_back(q_params.scale);
            qp.scales = &ple_per_tensor_scales.back();
          } else if (q_type == kLiteRtQuantizationPerChannel) {
            auto q_params = table_tensor.PerChannelQuantization();
            qp.scales = q_params.scales;
            qp.is_per_channel = true;
          }
        }
        ple_quant_params.push_back(qp);
        table_count++;
      }
    }

    auto outputs = subgraph.Outputs();
    RET_CHECK(!outputs.empty()) << "No outputs in subgraph";
    auto output_tensor = outputs[0];
    output_type = output_tensor.ElementType();

    if (output_type == litert::ElementType::Int16) {
      RET_CHECK(output_tensor.HasQuantization());
      auto q_params = output_tensor.PerTensorQuantization();
      final_scale = q_params.scale;
      final_zero_point = q_params.zero_point;
    }
  } else {
    LITERT_ASSIGN_OR_RETURN(
        embedder_per_layer_context,
        CreateEmbedderPerLayerContextWithBufferSharing(
            env, *embedder_per_layer_model,
            mask_context
                .prefill_input_buffers[MaskSignatures::kMaskInputTokens],
            mask_context.decode_input_buffers[MaskSignatures::kMaskInputTokens],
            mask_context.verify_input_buffers[MaskSignatures::kMaskInputTokens],
            gemma_prefill_input_buffers, gemma_decode_input_buffers,
            gemma_verify_input_buffers, executor_settings));
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
              env, **mtp_drafter_model, input_kv_cache_buffers,
              llm_inference_context.decode_output_buffers
                  [LlmSignatures::kLastLayerActivationsOutput]));
      LITERT_ASSIGN_OR_RETURN(
          drafter_aux_context,
          CreateDrafterInferenceAuxContextWithBufferSharing(
              env, **mtp_aux_model, drafter_context->mtp_input_buffers));
      speculative_decoding_type = SpeculativeDecodingType::kMTP;

      RETURN_IF_ERROR(WarmupDrafterInference(drafter_context.value(),
                                             drafter_aux_context.value()));
    }
  }

  auto executor = absl::WrapUnique(new LlmLiteRtNpuCompiledModelExecutor(
      executor_settings, env, std::move(embedder_context),
      std::move(npu_auxiliary_context), std::move(mask_context),
      std::move(rope_context), std::move(llm_compiled_model),
      std::move(llm_inference_context),
      std::move(cache_update_inference_context), std::move(prefill_runner_set),
      std::move(embedding_lookup_manager),
      std::move(embedder_per_layer_context), quantization_params,
      std::move(ple_table_ptrs), std::move(ple_quant_params),
      std::move(ple_per_tensor_scales), table_count, output_type, final_scale,
      final_zero_point, speculative_decoding_type, std::move(drafter_context),
      std::move(drafter_aux_context)));
  return executor;
}

absl::StatusOr<std::unique_ptr<LlmLiteRtNpuCompiledModelExecutor>>
LlmLiteRtNpuCompiledModelExecutor::CreateForModelWithoutPerLayerEmbedding(
    const LlmExecutorSettings& executor_settings, ModelResources& resources,
    litert::Environment& env, const litert::Model* transformer_model,
    LogitsQuantizationParams quantization_params) {
  // Set up LiteRt options.
  LITERT_ASSIGN_OR_RETURN(auto options,
                          CreateLiteRtNpuOptions(executor_settings));
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
  absl::flat_hash_map<absl::string_view, TensorBuffer>
      gemma_verify_input_buffers;
  absl::flat_hash_map<absl::string_view, TensorBuffer> input_kv_cache_buffers;
  absl::flat_hash_map<absl::string_view, TensorBuffer>
      prefill_output_kv_cache_slice_buffers;
  absl::flat_hash_map<absl::string_view, TensorBuffer>
      decode_output_kv_cache_slice_buffers;
  absl::flat_hash_map<absl::string_view, TensorBuffer>
      verify_output_kv_cache_slice_buffers;

  RETURN_IF_ERROR(AllocateTransformerBuffers(
      env, transformer_model, llm_compiled_model, gemma_prefill_input_buffers,
      gemma_decode_input_buffers, gemma_verify_input_buffers,
      input_kv_cache_buffers, prefill_output_kv_cache_slice_buffers,
      decode_output_kv_cache_slice_buffers,
      verify_output_kv_cache_slice_buffers));
  LITERT_ASSIGN_OR_RETURN(
      auto llm_inference_context,
      CreateLlmInferenceContextWithBufferSharing(
          env, llm_compiled_model, input_kv_cache_buffers,
          prefill_output_kv_cache_slice_buffers,
          decode_output_kv_cache_slice_buffers,
          verify_output_kv_cache_slice_buffers, gemma_prefill_input_buffers,
          gemma_decode_input_buffers, gemma_verify_input_buffers));

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
    llm_inference_context.decode_input_buffers[cache_k25] = std::move(buffer_k);
    LITERT_ASSIGN_OR_RETURN(auto buffer_v, llm_compiled_model.CreateInputBuffer(
                                               kDecodeSignature, cache_v25));
    llm_inference_context.decode_input_buffers[cache_v25] = std::move(buffer_v);
  } else if (llm_inference_context.prefill_input_buffers.contains(cache_k23)) {
    // Fast VLM model specific fix:
    LITERT_ASSIGN_OR_RETURN(auto buffer_k, llm_compiled_model.CreateInputBuffer(
                                               kDecodeSignature, cache_k23));
    llm_inference_context.decode_input_buffers[cache_k23] = std::move(buffer_k);
    LITERT_ASSIGN_OR_RETURN(auto buffer_v, llm_compiled_model.CreateInputBuffer(
                                               kDecodeSignature, cache_v23));
    llm_inference_context.decode_input_buffers[cache_v23] = std::move(buffer_v);
  } else {
    // Tiny Gemma 270M specific fix:
    LITERT_ASSIGN_OR_RETURN(auto buffer_k, llm_compiled_model.CreateInputBuffer(
                                               kDecodeSignature, cache_k17));
    llm_inference_context.decode_input_buffers[cache_k17] = std::move(buffer_k);
    LITERT_ASSIGN_OR_RETURN(auto buffer_v, llm_compiled_model.CreateInputBuffer(
                                               kDecodeSignature, cache_v17));
    llm_inference_context.decode_input_buffers[cache_v17] = std::move(buffer_v);
  }

  LITERT_ASSIGN_OR_RETURN(auto npu_auxiliary_lrt_model,
                          resources.GetTFLiteModel(ModelType::kTfLiteAux));

  LITERT_ASSIGN_OR_RETURN(
      auto npu_auxiliary_context,
      CreateNpuAuxiliaryContext(env, *npu_auxiliary_lrt_model,
                                executor_settings));

  LITERT_ASSIGN_OR_RETURN(
      auto mask_context,
      CreateMaskContextWithBufferSharing(
          npu_auxiliary_context, gemma_prefill_input_buffers,
          gemma_decode_input_buffers, gemma_verify_input_buffers));

  LITERT_ASSIGN_OR_RETURN(auto embedder_lrt_model,
                          resources.GetTFLiteModel(ModelType::kTfLiteEmbedder));
  LITERT_ASSIGN_OR_RETURN(
      auto embedder_context,
      CreateEmbedderContextWithBufferSharing(
          env, *embedder_lrt_model,
          mask_context.prefill_input_buffers[MaskSignatures::kMaskInputTokens],
          mask_context.decode_input_buffers[MaskSignatures::kMaskInputTokens],
          llm_inference_context
              .verify_input_buffers[LlmSignatures::kInputEmbeddings],
          gemma_prefill_input_buffers, gemma_decode_input_buffers,
          gemma_verify_input_buffers, executor_settings));

  LITERT_ASSIGN_OR_RETURN(
      auto rope_context,
      CreateRopeContextWithBufferSharing(
          npu_auxiliary_context, gemma_prefill_input_buffers,
          gemma_decode_input_buffers, gemma_verify_input_buffers));

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
  ::litert::TensorBuffer verify_input_pos;
  LITERT_ASSIGN_OR_RETURN(
      auto cache_update_inference_context,
      CreateCacheUpdateInferenceContextWithBufferSharing(
          input_kv_cache_buffers, prefill_output_kv_cache_slice_buffers,
          decode_output_kv_cache_slice_buffers,
          verify_output_kv_cache_slice_buffers, std::move(prefill_input_pos),
          std::move(decode_input_pos), std::move(verify_input_pos)));

  RETURN_IF_ERROR(WarmupInference(
      llm_compiled_model, llm_inference_context,
      npu_auxiliary_context.npu_auxiliary_compiled_model, rope_context,
      mask_context, cache_update_inference_context));

  // For now we only support one prefill length in the model.
  SortedPrefillSignatureMap prefill_runner_set;
  prefill_runner_set[kPrefillSize] = kPrefillSignature;

  std::optional<EmbedderPerLayerContext> embedder_per_layer_context =
      std::nullopt;

  std::unique_ptr<EmbeddingLookupManager> maybe_embedding_lookup_manager =
      nullptr;
  // If the model has vision or audio encoder, we need to create the embedding
  // lookup manager.
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

  auto executor = absl::WrapUnique(new LlmLiteRtNpuCompiledModelExecutor(
      executor_settings, env, std::move(embedder_context),
      std::move(npu_auxiliary_context), std::move(mask_context),
      std::move(rope_context), std::move(llm_compiled_model),
      std::move(llm_inference_context),
      std::move(cache_update_inference_context), std::move(prefill_runner_set),
      std::move(maybe_embedding_lookup_manager),
      /*embedder_per_layer_context=*/std::nullopt, quantization_params, {}, {},
      {}, 0, litert::ElementType::None, 1.0f, 0, speculative_decoding_type,
      std::move(drafter_context), std::move(drafter_aux_context)));
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
