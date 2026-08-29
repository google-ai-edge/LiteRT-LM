// Copyright 2026 The ODML Authors.
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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_NPU_LLM_LITERT_NPU_COMPILED_MODEL_EXECUTOR_UTILS_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_NPU_LLM_LITERT_NPU_COMPILED_MODEL_EXECUTOR_UTILS_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/log/log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_expected.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_options.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/embedding_lookup/embedding_lookup_manager.h"
#include "runtime/components/model_resources.h"
#include "runtime/executor/litert_compiled_model_executor_utils.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/executor/llm_executor_processed_tokens.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/util/status_macros.h"

namespace litert::lm {

// Holds the latency breakdown stats for the executor.
// TODO: b/405424188 - Use 'litert::lm::BenchmarkInfo' instead.
struct LatencyStats {
  // Prefill latency stats.
  uint64_t prefill_e2e_latency_us = 0;
  int prefill_num_tokens = 0;
  uint64_t prefill_prepare_input_latency_us = 0;
  uint64_t prefill_embedder_inference_latency_us = 0;
  uint64_t prefill_embedder_per_layer_inference_latency_us = 0;
  uint64_t prefill_mask_inference_latency_us = 0;
  uint64_t prefill_rope_inference_latency_us = 0;
  uint64_t prefill_llm_inference_latency_us = 0;
  uint64_t prefill_cache_update_inference_latency_us = 0;

  // Decode latency stats.
  uint64_t decode_e2e_latency_us = 0;
  int decode_num_tokens = 0;
  uint64_t decode_prepare_input_latency_us = 0;
  uint64_t decode_embedder_inference_latency_us = 0;
  uint64_t decode_embedder_per_layer_inference_latency_us = 0;
  uint64_t decode_mask_inference_latency_us = 0;
  uint64_t decode_rope_inference_latency_us = 0;
  uint64_t decode_llm_inference_latency_us = 0;
  uint64_t decode_drafter_inference_latency_us = 0;
  uint64_t decode_cache_update_inference_latency_us = 0;
  uint64_t decode_sampling_latency_us = 0;
  uint64_t decode_mtp_rejection_sampling_latency_us = 0;
  uint64_t decode_mtp_activation_copy_latency_us = 0;
  uint64_t decode_token_queue_latency_us = 0;

  // MTP / Speculative Decoding latency stats.
  int mtp_num_draft_tokens = 0;
  int mtp_num_accepted_tokens = 0;
};

std::ostream& operator<<(std::ostream& os, const LatencyStats& stats);

// The prefill family of signature names resolved for the prefill length the
// model was compiled with (e.g. 128 or 256) and optional context size (e.g.
// 640).
struct ResolvedPrefillSignatures {
  int size = 0;
  int context_size = 0;
  std::string prefill;
  std::string embedder;
  std::string embedder_per_layer;
  std::string mask;
  std::string rope;
  std::string cache_update;
};

// Signature names for the Text Decoder signatures.
struct TextDecoderSignatures {
  static constexpr absl::string_view kDecode = "decode";
  static constexpr absl::string_view kVerify = "verify";
  static constexpr absl::string_view kInputEmbeddings = "embeddings";
  static constexpr absl::string_view kDecodeLogitsOutput = "logits";
  static constexpr absl::string_view kVerifyLogitsOutput = "logits";
  static constexpr absl::string_view kLastLayerActivationsOutput =
      "activations";
};
using LlmSignatures = TextDecoderSignatures;

// Signature names for MTP speculative decoding signatures.
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

inline constexpr int kInvalidTokenId = -1;

inline constexpr absl::string_view kPrefillSignatureBase = "prefill";
inline constexpr absl::string_view kPrefillEmbedderBase = "prefill_embedder";
inline constexpr absl::string_view kPrefillEmbedderPerLayerBase =
    "prefill_per_layer_embedder";
inline constexpr absl::string_view kPrefillMaskBase = "prefill_mask";
inline constexpr absl::string_view kPrefillRopeBase = "prefill_rope";
inline constexpr absl::string_view kPrefillCacheUpdateBase =
    "prefill_cache_update";
inline constexpr char kDecodeSignature[] = "decode";
inline constexpr absl::string_view kDecodeMaskBase = "decode_mask";
inline constexpr absl::string_view kDecodeRopeBase = "decode_rope";
inline constexpr absl::string_view kDecodeCacheUpdateBase =
    "decode_cache_update";
inline constexpr absl::string_view kVerifyMaskBase = "verify_mask";
inline constexpr absl::string_view kVerifyRopeBase = "verify_rope";
inline constexpr absl::string_view kVerifyCacheUpdateBase =
    "verify_cache_update";

inline constexpr absl::string_view kKvCacheKRootName = "kv_cache_k_";
inline constexpr absl::string_view kKvCacheVRootName = "kv_cache_v_";
inline constexpr absl::string_view kKvCacheCRootName = "kv_cache_c_";

inline constexpr absl::string_view kKvCacheSliceKRootName = "kv_slice_k_";
inline constexpr absl::string_view kKvCacheSliceVRootName = "kv_slice_v_";
inline constexpr absl::string_view kKvCacheSliceCRootName = "kv_slice_c_";

bool IsNpuSyncWorkaroundEnabled();

// Fills a TensorBuffer with a specific uint16 value across all supported types.
absl::Status Fill(::litert::TensorBuffer& tensor_buffer, uint16_t value);

// Copies raw bytes from the tensor buffer.
absl::StatusOr<std::vector<uint8_t>> CopyRawBytesFromTensorBuffer(
    const ::litert::TensorBuffer& buffer);

// Detect if the model uses Sliding Window Attention (SWA) by checking if
// there are different KV cache sizes (mixed local/global attention).
bool DetectIsSwa(
    const absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        input_kv_cache_buffers);

// Builds a prefill signature name from its base and the prefill length, e.g.
// PrefillSig("prefill_mask", 256) -> "prefill_mask_256".
std::string PrefillSig(absl::string_view base, int prefill_size);

// Detects the prefill length the transformer model was compiled with by
// scanning its signatures for the bare LLM prefill signature named
// "prefill_<N>" (excluding the prefill_mask/rope/embedder/cache_update family),
// and returns <N>.
absl::StatusOr<int> DetectPrefillSize(const litert::Model& transformer_model);

// Detects all supported context sizes by scanning transformer signatures for
// "_cache_<N>" pattern. Returns sorted vector of context sizes in ascending
// order. Returns empty vector if no dynamic context sizes are found.
absl::StatusOr<std::vector<int>> DetectSupportedContextSizes(
    const litert::Model& transformer_model);

// Builds the prefill family of signature names for a given prefill length and
// optional context length.
ResolvedPrefillSignatures BuildResolvedPrefillSignatures(int prefill_size,
                                                         int context_size = 0);

static constexpr absl::string_view kPerLayerEmbedderTensor =
    "per_layer_embeddings";

// Returns true if the transformer model has a per layer embedder input buffer.
litert::Expected<bool> HasPerLayerEmbedder(
    const litert::Model& transformer_model,
    absl::string_view prefill_signature);


// Creates CPU options for LiteRT compiled models.
litert::Expected<litert::Options> CreateLiteRtCpuOptions(
    const LlmExecutorSettings& settings);

// Creates LiteRT options for NPU accelerator.
litert::Expected<litert::Options> CreateLiteRtNpuOptions(
    const LlmExecutorSettings& settings);

// Formats the first N elements of a span as a comma-separated list inside
// brackets, e.g., "[1, 2, 3, ...]".
template <typename T>
std::string FormatFirstN(absl::Span<const T> span, size_t n = 10) {
  return absl::StrFormat("[%s%s]", absl::StrJoin(span.subspan(0, n), ", "),
                         span.size() > n ? ", ..." : "");
}

// Quantizes a float value to a quantized type T.
template <typename T>
T Quantize(float value, float scale, int32_t zero_point) {
  static_assert(std::is_same_v<T, int16_t> || std::is_same_v<T, int8_t>,
                "Unsupported quantization type.");
  int32_t qval = std::round(value / scale) + zero_point;
  return static_cast<T>(
      std::clamp(qval, static_cast<int32_t>(std::numeric_limits<T>::min()),
                 static_cast<int32_t>(std::numeric_limits<T>::max())));
}
#if defined(__ANDROID__) && defined(__ARM_NEON)
int FindMaxIndexFloatNeon(const float* data, int size);
int FindMaxIndexInt16Neon(const int16_t* data, int size);
int FindMaxIndexInt8Neon(const int8_t* data, int size);
#endif
#if defined(__x86_64__) || defined(_M_X64)
int FindMaxIndexSse2Float(const float* data, int size);
int FindMaxIndexSse2Int16(const int16_t* data, int size);
int FindMaxIndexSse2Int8(const int8_t* data, int size);
#endif

// Generic function to find the index of the maximum value in a TensorBuffer.
// Uses NEON optimizations if available.
template <typename T>
absl::StatusOr<int> FindMaxIndex(::litert::TensorBuffer& decoded_logits,
                                 bool use_neon_sampling) {
  LITERT_ASSIGN_OR_RETURN(auto tensor_type, decoded_logits.TensorType());
  LITERT_ASSIGN_OR_RETURN(size_t num_elements,
                          tensor_type.Layout().NumElements());
  if (num_elements == 0) {
    return absl::InvalidArgumentError("Logits buffer is empty.");
  }
  LITERT_ASSIGN_OR_RETURN(
      auto lock_and_addr,
      ::litert::TensorBufferScopedLock::Create(
          const_cast<::litert::TensorBuffer&>(decoded_logits),
          ::litert::TensorBuffer::LockMode::kRead));
  const T* data = static_cast<const T*>(lock_and_addr.second);

  LITERT_ASSIGN_OR_RETURN(size_t size, tensor_type.Layout().NumElements());

  if (size == 0) {
    return absl::InvalidArgumentError("Logits buffer is empty.");
  }

#if defined(__ANDROID__) && defined(__ARM_NEON)
  if (use_neon_sampling) {
    if constexpr (std::is_same_v<T, float>) {
      return FindMaxIndexFloatNeon(data, static_cast<int>(size));
    } else if constexpr (std::is_same_v<T, int16_t>) {
      return FindMaxIndexInt16Neon(data, static_cast<int>(size));
    } else if constexpr (std::is_same_v<T, int8_t>) {
      return FindMaxIndexInt8Neon(data, static_cast<int>(size));
    }
  }
#endif
#if defined(__x86_64__) || defined(_M_X64)
  if (use_neon_sampling) {
    if constexpr (std::is_same_v<T, float>) {
      return FindMaxIndexSse2Float(data, static_cast<int>(size));
    } else if constexpr (std::is_same_v<T, int16_t>) {
      return FindMaxIndexSse2Int16(data, static_cast<int>(size));
    } else if constexpr (std::is_same_v<T, int8_t>) {
      return FindMaxIndexSse2Int8(data, static_cast<int>(size));
    }
  }
#endif

  int max_index = 0;
  T max_value = data[0];
  for (size_t i = 1; i < num_elements; ++i) {
    if (data[i] > max_value) {
      max_value = data[i];
      max_index = static_cast<int>(i);
    }
  }
  return max_index;
}

// Applies greedy sampling (argmax) to the decoded logits.
absl::StatusOr<int> ApplyGreedySampling(::litert::TensorBuffer& decoded_logits,
                                        bool use_neon_sampling);

struct HWQuantParams {
  float scale = 1.0f;
  int64_t zero_point = 0;
};


// Context holding input and output tensor buffers for an inference phase.
struct InferenceContext {
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

  InferenceContext() = default;
  InferenceContext(const InferenceContext&) = delete;
  InferenceContext& operator=(const InferenceContext&) = delete;
  InferenceContext(InferenceContext&&) = default;
  InferenceContext& operator=(InferenceContext&&) = default;

  InferenceContext(
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
          prefill_input_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
          prefill_output_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
          decode_input_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
          decode_output_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
          verify_input_buffers = {},
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
          verify_output_buffers = {})
      : prefill_input_buffers(std::move(prefill_input_buffers)),
        prefill_output_buffers(std::move(prefill_output_buffers)),
        decode_input_buffers(std::move(decode_input_buffers)),
        decode_output_buffers(std::move(decode_output_buffers)),
        verify_input_buffers(std::move(verify_input_buffers)),
        verify_output_buffers(std::move(verify_output_buffers)) {}
};

// Holds pre-resolved signatures for auxiliary models (Mask, RoPE, Cache
// Update).
struct ResolvedAuxiliarySignatures {
  std::string mask;
  std::string rope;
  std::string cache_update;
};

// Holds model signatures and tensor buffer bindings for a specific context
// size.
struct ContextGroup {
  int context_size = 0;
  ResolvedPrefillSignatures prefill_signatures;
  std::string decode_signature;
  ResolvedAuxiliarySignatures decode_aux_signatures;
  std::string verify_signature;
  ResolvedAuxiliarySignatures verify_aux_signatures;
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      input_kv_cache_buffers;
  InferenceContext text_decoder_inference_context;
};

ResolvedAuxiliarySignatures BuildResolvedDecodeAuxiliarySignatures(
    const ::litert::CompiledModel& aux_model, int context_size);

ResolvedAuxiliarySignatures BuildResolvedVerifyAuxiliarySignatures(
    const ::litert::CompiledModel& aux_model, int context_size);

inline absl::Status SetFirstElement(::litert::TensorBuffer& buffer,
                                    int32_t value) {
  LITERT_ASSIGN_OR_RETURN(
      auto lock_and_addr,
      ::litert::TensorBufferScopedLock::Create(
          buffer, ::litert::TensorBuffer::LockMode::kWrite));
  static_cast<int32_t*>(lock_and_addr.second)[0] = value;
  return absl::OkStatus();
}




// Holds the context for the NPU auxiliary model, which contains several
// signatures for Mask, RoPE and KV cache update computation.
struct NpuAuxiliaryContext {
  ::litert::CompiledModel npu_auxiliary_compiled_model;
  explicit NpuAuxiliaryContext(
      ::litert::CompiledModel npu_auxiliary_compiled_model)
      : npu_auxiliary_compiled_model(std::move(npu_auxiliary_compiled_model)) {}

  static absl::StatusOr<NpuAuxiliaryContext> Create(
      ::litert::Environment& env, const litert::Model& npu_auxiliary_model,
      const LlmExecutorSettings& settings);
};

// Creates the context for the NPU auxiliary model.
absl::StatusOr<NpuAuxiliaryContext> CreateNpuAuxiliaryContext(
    ::litert::Environment& env, const litert::Model& npu_auxiliary_model,
    const LlmExecutorSettings& settings);

// Holds the context for the drafter model.
struct DrafterContext {
  ::litert::CompiledModel mtp_compiled_model;
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      mtp_input_buffers;
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      mtp_output_buffers;
  DrafterContext(::litert::CompiledModel mtp_compiled_model,
                 absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
                     mtp_input_buffers,
                 absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
                     mtp_output_buffers)
      : mtp_compiled_model(std::move(mtp_compiled_model)),
        mtp_input_buffers(std::move(mtp_input_buffers)),
        mtp_output_buffers(std::move(mtp_output_buffers)) {}

  // Sets the drafter step position in
  // mtp_input_buffers[MtpSignatures::kInputPos].
  absl::Status SetInputPos(int32_t pos);

  // Re-binds the drafter's KV cache input buffers to the new active context
  // group.
  absl::Status UpdateKVCacheBuffers(
      const absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          new_input_kv_cache_buffers);

  // Allocates the MTP drafter buffers and creates the DrafterContext.
  static absl::StatusOr<DrafterContext> Create(
      ::litert::Environment& env, const litert::Model& mtp_drafter_model,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          drafter_input_kv_cache_buffers,
      ::litert::TensorBuffer& output_activations_buffers);
};

// Dequantize logits to float32.
absl::Status DequantizeLogits(const ::litert::TensorBuffer& src,
                              ::litert::TensorBuffer& dst, float scale,
                              int32_t zero_point, bool should_dump);

// Creates a zero-copy alias of `source_buffer` with a smaller `target_type`
// layout. Supports LiteRT platform buffer backends (AHWB, DMA-BUF, ION,
// FastRPC, Host).
absl::StatusOr<::litert::TensorBuffer> CreateAliasBuffer(
    const ::litert::Environment& env,
    const ::litert::TensorBuffer& source_buffer,
    const ::litert::RankedTensorType& target_type);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_NPU_LLM_LITERT_NPU_COMPILED_MODEL_EXECUTOR_UTILS_H_
