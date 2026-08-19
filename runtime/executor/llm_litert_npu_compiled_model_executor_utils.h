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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LLM_LITERT_NPU_COMPILED_MODEL_EXECUTOR_UTILS_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LLM_LITERT_NPU_COMPILED_MODEL_EXECUTOR_UTILS_H_

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
// model was compiled with (e.g. 128 or 256).
struct ResolvedPrefillSignatures {
  int size = 0;
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

// Builds the prefill family of signature names for a given prefill length.
ResolvedPrefillSignatures BuildResolvedPrefillSignatures(int prefill_size);

// Returns true if the transformer model has a per layer embedder input buffer.
litert::Expected<bool> HasPerLayerEmbedder(
    const litert::Model& transformer_model,
    absl::string_view prefill_signature);

// Extracts the KV cache init value from ModelResources metadata if available.
int64_t GetKvCacheInitValue(ModelResources& resources);

// Fills a KV cache TensorBuffer with the specified initialization value.
absl::Status FillKVCacheBuffer(::litert::TensorBuffer& buffer,
                               int64_t init_value);

// Clears all KV cache buffers in the map with the specified initialization
// value.
absl::Status ClearKVCacheBuffers(
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>& buffers,
    int64_t init_value = 0);

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

// Performs manual KV cache update.
absl::Status HWKVCacheUpdate(
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>& in_buffers,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>& out_buffers,
    const absl::flat_hash_map<absl::string_view, HWQuantParams>& quant_params =
        {},
    bool enable_swa = false);

enum class MaskUpdateMethod {
  kModel,
  kWH,
};

// Signature names for the mask signatures.
struct MaskSignatures {
  static constexpr absl::string_view kDecodeMask = "decode_mask";
  static constexpr absl::string_view kVerifyMask = "verify_mask";
  static constexpr absl::string_view kMtpMask = "mask";
  // Prefill and decode use identical tensor signature names.
  static constexpr absl::string_view kMaskInputTimeStep = "time_step";
  static constexpr absl::string_view kMaskInputTokens = "input_tokens";
  static constexpr absl::string_view kMaskInputValidMask = "valid_mask";
  static constexpr absl::string_view kMaskLocal = "mask_local";
  static constexpr absl::string_view kMaskGlobal = "mask_global";
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

inline absl::Status SetFirstElement(::litert::TensorBuffer& buffer,
                                    int32_t value) {
  LITERT_ASSIGN_OR_RETURN(
      auto lock_and_addr,
      ::litert::TensorBufferScopedLock::Create(
          buffer, ::litert::TensorBuffer::LockMode::kWrite));
  static_cast<int32_t*>(lock_and_addr.second)[0] = value;
  return absl::OkStatus();
}

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

struct EmbedderContext {
  ::litert::Model embedder_model;
  ::litert::CompiledModel embedder_compiled_model;
  InferenceContext inference_context;
  EmbedderContext(::litert::CompiledModel embedder_compiled_model,
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
      : embedder_compiled_model(std::move(embedder_compiled_model)),
        inference_context(
            std::move(prefill_input_buffers), std::move(prefill_output_buffers),
            std::move(decode_input_buffers), std::move(decode_output_buffers),
            std::move(verify_input_buffers), std::move(verify_output_buffers)) {
  }
  EmbedderContext(EmbedderContext&&) = default;
  EmbedderContext& operator=(EmbedderContext&&) = default;
};

// Holds the context for the embedder per layer model.
struct EmbedderPerLayerContext {
  ::litert::CompiledModel embedder_per_layer_compiled_model;
  InferenceContext inference_context;
  EmbedderPerLayerContext(
      ::litert::CompiledModel embedder_per_layer_compiled_model,
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
      : embedder_per_layer_compiled_model(
            std::move(embedder_per_layer_compiled_model)),
        inference_context(
            std::move(prefill_input_buffers), std::move(prefill_output_buffers),
            std::move(decode_input_buffers), std::move(decode_output_buffers),
            std::move(verify_input_buffers), std::move(verify_output_buffers)) {
  }
  EmbedderPerLayerContext(EmbedderPerLayerContext&&) = default;
  EmbedderPerLayerContext& operator=(EmbedderPerLayerContext&&) = default;
};

// Hardware quantization parameters for Per-Layer Embeddings.
struct HWQuantizationParams {
  const float* scales;
  bool is_per_channel;
};

// Holds hardware quantization parameters and lookup tables for Per-Layer
// Embeddings (PLE).
struct HWPleParams {
  bool use_hw_ple = false;
  std::vector<const uint8_t*> ple_table_ptrs;
  std::vector<HWQuantizationParams> ple_quant_params;
  std::vector<float> ple_per_tensor_scales;
  int num_tables = 0;
  int ple_embedding_dim = 0;
  litert::ElementType output_type = litert::ElementType::None;
  litert::ElementType ple_table_element_type = litert::ElementType::None;
  float mul_scale = 1.0f;
  float output_scale = 1.0f;
  int32_t final_zero_point = 0;
};

// =============================================================================
// NpuEmbedder Usage Guide:
//
// 1. Multimodal Lifecycle (Optional/No-op for text-only):
//    embedder_.UpdateMultiModalEmbeddings(inputs);
//    ... prefill ...
//    embedder_.CleanupMultiModalEmbeddings();
//
// 2. Regular Prefill:
//    embedder_.RunPrefill(signature, pending_token, processed_tokens,
//    last_input_token); if (embedder_.HasPerLayerEmbeddings()) {
//      embedder_.RunPrefillPerLayer(ple_signature, tokens_to_embed);
//    }
//    * Note on Prefill: All 3 embedding steps (copy previous pending token to
//      slot 0, lookup processed input tokens, and lookup/cache the new holdback
//      token) are encapsulated in a single `RunPrefill` call because all tokens
//      in the prefill chunk are known together at the time of execution.
//
// 3. Regular Single-Token Decode:
//    // Step A (End of Step T): Look up embedding for newly sampled token and
//    // attach it to TokenData before enqueuing into the pending token queue:
//    embedder_.LookupDecode(token);
//
//    // Step B (Start of Step T+1): When popped from queue, copy embedding to
//    // decode buffer (or run compiled embedder model) before text decoder
//    runs: embedder_.RunDecode(*token); if (embedder_.HasPerLayerEmbeddings())
//    {
//      embedder_.RunDecodePerLayer(token->id());
//    }
//    * Note on Decode: Decode separates `LookupDecode` and `RunDecode` across
//      step boundaries because token generation is autoregressive: the newly
//      sampled token's embedding is looked up immediately upon generation at
//      the end of step T, whereas hardware buffer loading (`RunDecode`) occurs
//      at the start of step T+1 when preparing decoder inputs.
//
// 4. MTP Speculative Decoding - Draft Token Embedding:
//    embedder_.LookupDecode(draft_token_id, draft_embedding);
//
// 5. MTP Speculative Decoding - Verification:
//    embedder_.RunVerify(verify_ids);
//    if (embedder_.HasPerLayerEmbeddings()) {
//      embedder_.RunVerifyPerLayer(verify_ids);
//    }
// =============================================================================
class NpuEmbedder {
 public:
  NpuEmbedder() = default;
  NpuEmbedder(const NpuEmbedder&) = delete;
  NpuEmbedder& operator=(const NpuEmbedder&) = delete;
  NpuEmbedder(NpuEmbedder&&) = default;
  NpuEmbedder& operator=(NpuEmbedder&&) = default;

  // --- Lifecycle & Creation ---
  static absl::StatusOr<NpuEmbedder> Create(
      ::litert::Environment& env, ModelResources& resources,
      const LlmExecutorSettings& executor_settings,
      const ResolvedPrefillSignatures& prefill_signatures,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          text_decoder_prefill_input_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          text_decoder_decode_input_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          text_decoder_verify_input_buffers,
      bool has_per_layer_embeddings);

  absl::Status UpdateMultiModalEmbeddings(const ExecutorInputs& inputs);
  absl::Status CleanupMultiModalEmbeddings();
  std::vector<float> GetDefaultEmbeddingVector() const;

  // --- Stage 1: Prefill ---
  absl::Status RunPrefill(absl::string_view embedder_signature,
                          const TokenData* pending_token,
                          absl::Span<const int> processed_input_tokens,
                          TokenData* last_input_token);
  absl::Status RunPrefillPerLayer(absl::string_view signature,
                                  absl::Span<const int> tokens_to_embed);
  absl::Status WriteAndPadPleEmbeddings(::litert::Environment& env,
                                        absl::Span<const float> ple_embeddings);

  // --- Stage 2: Decode ---
  absl::Status LookupDecode(TokenData* token);
  absl::Status LookupDecode(int32_t token_id,
                            std::vector<float>& out_embedding);
  absl::Status RunDecode(const TokenData& token);
  absl::Status RunDecodePerLayer(int32_t token_id);
  absl::Status WriteDecodePleEmbeddings(absl::Span<const float> ple_embeddings);

  // --- Stage 3: Speculative Decoding (Verification) ---
  absl::Status RunVerify(absl::Span<const int> verify_ids);
  absl::Status RunVerifyPerLayer(absl::Span<const int> verify_ids);

  // --- Query & Introspection ---
  bool HasPerLayerEmbeddings() const {
    return (ple_params_.use_hw_ple && !ple_params_.ple_table_ptrs.empty()) ||
           embedder_per_layer_context_.has_value();
  }
  bool UseHwPle() const {
    return ple_params_.use_hw_ple && !ple_params_.ple_table_ptrs.empty();
  }
  const HWPleParams& PleParams() const { return ple_params_; }

  // --- Warmup & Context Accessors ---
  absl::Status RunPrefillEmbedder(absl::string_view signature);
  absl::Status RunDecodeEmbedder();

  EmbedderContext* MutableEmbedderContext() {
    return embedder_context_.has_value() ? &*embedder_context_ : nullptr;
  }
  EmbedderPerLayerContext* MutableEmbedderPerLayerContext() {
    return embedder_per_layer_context_.has_value()
               ? &*embedder_per_layer_context_
               : nullptr;
  }
  const EmbedderPerLayerContext* GetEmbedderPerLayerContext() const {
    return embedder_per_layer_context_.has_value()
               ? &*embedder_per_layer_context_
               : nullptr;
  }

 private:
  absl::Status LookupHwPle(const int* token_ids, int num_tokens,
                           void* output_buffer) const;

  static absl::StatusOr<NpuEmbedder> Create(
      std::unique_ptr<EmbeddingLookupManager> embedding_lookup_manager,
      std::optional<EmbedderContext> embedder_context,
      std::optional<EmbedderPerLayerContext> embedder_per_layer_context,
      std::unique_ptr<EmbeddingLookupManager>
          per_layer_embedding_lookup_manager,
      const litert::Model* embedder_per_layer_model, HWPleParams ple_params,
      std::optional<::litert::TensorBuffer> prefill_embeddings_buffer =
          std::nullopt,
      std::optional<::litert::TensorBuffer> decode_embeddings_buffer =
          std::nullopt,
      std::optional<::litert::TensorBuffer> verify_embeddings_buffer =
          std::nullopt,
      std::optional<::litert::TensorBuffer> prefill_ple_buffer = std::nullopt,
      std::optional<::litert::TensorBuffer> decode_ple_buffer = std::nullopt,
      std::optional<::litert::TensorBuffer> verify_ple_buffer = std::nullopt);

  absl::Status RunVerifyEmbedder(absl::Span<const int> verify_ids);
  absl::Status WriteAndPadPleEmbeddings(::litert::Environment& env,
                                        ::litert::TensorBuffer& buffer,
                                        absl::Span<const float> ple_embeddings);

  static absl::StatusOr<EmbedderContext> CreateEmbedderContext(
      ::litert::Environment& env, const litert::Model& embedder_model,
      const ResolvedPrefillSignatures& prefill_signatures,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          text_decoder_prefill_input_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          text_decoder_decode_input_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          text_decoder_verify_input_buffers,
      const LlmExecutorSettings& settings);

  static absl::StatusOr<EmbedderPerLayerContext> CreateEmbedderPerLayerContext(
      ::litert::Environment& env, const litert::Model& embedder_model,
      const ResolvedPrefillSignatures& prefill_signatures,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          text_decoder_prefill_input_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          text_decoder_decode_input_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          text_decoder_verify_input_buffers,
      const LlmExecutorSettings& settings);

  bool HasEmbeddingLookupManager() const {
    return embedding_lookup_manager_ != nullptr;
  }

  absl::Status CopyEmbeddingToBuffer(
      absl::Span<const float> embedding,
      ::litert::TensorBuffer& destination_buffer);

  std::unique_ptr<EmbeddingLookupManager> embedding_lookup_manager_;
  std::optional<EmbedderContext> embedder_context_;
  std::optional<EmbedderPerLayerContext> embedder_per_layer_context_;
  std::unique_ptr<EmbeddingLookupManager> per_layer_embedding_lookup_manager_;
  const litert::Model* embedder_per_layer_model_ = nullptr;
  HWPleParams ple_params_;
  std::optional<::litert::TensorBuffer> prefill_embeddings_buffer_;
  std::optional<::litert::TensorBuffer> decode_embeddings_buffer_;
  std::optional<::litert::TensorBuffer> verify_embeddings_buffer_;
  std::optional<::litert::TensorBuffer> prefill_ple_buffer_;
  std::optional<::litert::TensorBuffer> decode_ple_buffer_;
  std::optional<::litert::TensorBuffer> verify_ple_buffer_;
};

// Signature names for the rope signatures.
struct RopeSignatures {
  static constexpr absl::string_view kDecodeRope = "decode_rope";
  static constexpr absl::string_view kVerifyRope = "verify_rope";
  static constexpr absl::string_view kMtpRope = "rope";
  // Prefill and decode use identical tensor signature names.
  static constexpr absl::string_view kInputPos = "input_pos";
  static constexpr absl::string_view kOutputPosEmbeddingLocalLow =
      "pos_emb_local_cos";
  static constexpr absl::string_view kOutputPosEmbeddingHigh = "pos_emb_sin";
  static constexpr absl::string_view kOutputPosEmbeddingLocalHigh =
      "pos_emb_local_sin";
  static constexpr absl::string_view kOutputPosEmbeddingLow = "pos_emb_cos";
};

// =============================================================================
// NpuRope Usage Guide:
//
// 1. Regular Prefill:
//    rope_.SetPrefillPositions(positions_span);
//    rope_.RunPrefill(prefill_signature);
//
// 2. Regular Single-Token Decode:
//    rope_.SetDecodePosition(current_step);
//    rope_.RunDecode();
//
// 3. MTP Speculative Decoding - Draft Generation (on drafter_rope):
//    drafter_rope.SetDecodePosition(draft_step);
//    drafter_rope.RunDrafter();
//
// 4. MTP Speculative Decoding - Verification (on main_rope):
//    main_rope_.SetVerifyPositions(start_step, num_tokens);
//    main_rope_.RunVerify();
// =============================================================================
class NpuRope {
 public:
  NpuRope() = default;
  NpuRope(const NpuRope&) = delete;
  NpuRope& operator=(const NpuRope&) = delete;
  NpuRope(NpuRope&&) = default;
  NpuRope& operator=(NpuRope&&) = default;

  // --- Lifecycle & Creation ---
  static absl::StatusOr<NpuRope> Create(
      const ::litert::CompiledModel* npu_auxiliary_compiled_model,
      const ResolvedPrefillSignatures& prefill_signatures,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          text_decoder_prefill_input_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          text_decoder_decode_input_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          text_decoder_verify_input_buffers);

  static absl::StatusOr<NpuRope> CreateForDrafter(
      const ::litert::CompiledModel* compiled_model,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
          rope_input_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
          rope_output_buffers);

  static absl::StatusOr<NpuRope> CreateForTest(
      const ::litert::CompiledModel* compiled_model,
      InferenceContext rope_context);

  void SetCompiledModel(const ::litert::CompiledModel* compiled_model) {
    compiled_model_ = compiled_model;
  }

  // --- Stage 1: Prefill ---
  absl::Status SetPrefillPositions(absl::Span<const int32_t> positions);
  absl::Status RunPrefill(absl::string_view signature) const;

  // --- Stage 2: Decode (Main & Drafter) ---
  absl::Status SetDecodePosition(int32_t step);
  absl::Status RunDecode() const;

  // --- Stage 3: Speculative Decoding (MTP Draft & Verify) ---
  absl::Status RunDrafter() const;
  absl::Status SetVerifyPositions(int32_t start_step, size_t num_tokens);
  absl::Status RunVerify() const;

  // --- Accessors ---
  const InferenceContext& Context() const { return rope_context_; }
  InferenceContext ReleaseContext() { return std::move(rope_context_); }

 private:
  explicit NpuRope(const ::litert::CompiledModel* compiled_model,
                   InferenceContext rope_context)
      : compiled_model_(compiled_model),
        rope_context_(std::move(rope_context)) {}

  const ::litert::CompiledModel* compiled_model_ = nullptr;
  InferenceContext rope_context_;
};

// Performs manual attention mask update.
absl::Status HWMaskUpdate(
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>& in_buffers,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        out_buffers);

// =============================================================================
// NpuMask Usage Guide:
//
// 1. Regular Prefill:
//    mask_.SetPrefillInput(internal_start_step, tokens_to_embed);
//    mask_.RunPrefill(prefill_signature);
//
// 2. Regular Single-Token Decode:
//    mask_.SetDecodeInput(current_step, token_id);
//    mask_.RunDecode();
//
// 3. MTP Speculative Decoding - Draft Generation (on drafter_mask):
//    drafter_mask.SetDecodeInput(draft_step, draft_token_id);
//    drafter_mask.RunDrafter();
//
// 4. MTP Speculative Decoding - Verification (on main_mask):
//    main_mask_.SetVerifyInput(start_step, verify_ids);
//    main_mask_.RunVerify();
// =============================================================================
class NpuMask {
 public:
  NpuMask() = default;
  NpuMask(const NpuMask&) = delete;
  NpuMask& operator=(const NpuMask&) = delete;
  NpuMask(NpuMask&&) = default;
  NpuMask& operator=(NpuMask&&) = default;

  // --- Lifecycle & Creation ---
  static absl::StatusOr<NpuMask> Create(
      MaskUpdateMethod method,
      const ::litert::CompiledModel* npu_auxiliary_compiled_model,
      const ResolvedPrefillSignatures& prefill_signatures,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          text_decoder_prefill_input_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          text_decoder_decode_input_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          text_decoder_verify_input_buffers);

  static absl::StatusOr<NpuMask> CreateForDrafter(
      MaskUpdateMethod method, const ::litert::CompiledModel* compiled_model,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
          mask_input_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
          mask_output_buffers);

  static absl::StatusOr<NpuMask> CreateForTest(
      MaskUpdateMethod method, const ::litert::CompiledModel* compiled_model,
      InferenceContext mask_context);

  void SetCompiledModel(const ::litert::CompiledModel* compiled_model) {
    compiled_model_ = compiled_model;
  }

  // --- Stage 1: Prefill ---
  absl::Status SetPrefillInput(int32_t start_step,
                               absl::Span<const int> token_ids);
  absl::Status RunPrefill(absl::string_view signature) const;

  // --- Stage 2: Decode (Main & Drafter) ---
  absl::Status SetDecodeInput(int32_t step, int32_t token_id);
  absl::Status RunDecode() const;

  // --- Stage 3: Speculative Decoding (MTP Draft & Verify) ---
  absl::Status RunDrafter() const;
  absl::Status SetVerifyInput(int32_t start_step,
                              absl::Span<const int> verify_ids);
  absl::Status RunVerify() const;

  // --- Accessors ---
  MaskUpdateMethod GetMethod() const { return method_; }
  const InferenceContext& Context() const { return mask_context_; }
  InferenceContext ReleaseContext() { return std::move(mask_context_); }

 private:
  explicit NpuMask(MaskUpdateMethod method,
                   const ::litert::CompiledModel* compiled_model,
                   InferenceContext mask_context)
      : method_(method),
        compiled_model_(compiled_model),
        mask_context_(std::move(mask_context)) {}

  MaskUpdateMethod method_ = MaskUpdateMethod::kModel;
  const ::litert::CompiledModel* compiled_model_ = nullptr;
  InferenceContext mask_context_;
};

enum class KVCacheUpdateMethod {
  kModel,
  kWH,
};

struct CacheUpdateSignatures {
  static constexpr absl::string_view kDecodeCacheUpdate = "decode_cache_update";
  static constexpr absl::string_view kVerifyCacheUpdate = "verify_cache_update";
  static constexpr absl::string_view kInputPos = "input_pos";
  static constexpr absl::string_view kInputValidMask = "valid_mask";
};

// =============================================================================
// NpuKVCache Usage Guide:
//
// 1. Regular Prefill:
//    cache_.SetPrefillPositions(seq_positions);
//    cache_.RunPrefill(prefill_signature);
//
// 2. Regular Single-Token Decode:
//    cache_.SetDecodePosition(current_step);
//    cache_.RunDecode();
//
// 3. MTP Speculative Decoding - Verification:
//    cache_.CommitVerifiedKVCache(start_step);
// =============================================================================
class NpuKVCache {
 public:
  NpuKVCache() = default;
  NpuKVCache(const NpuKVCache&) = delete;
  NpuKVCache& operator=(const NpuKVCache&) = delete;
  NpuKVCache(NpuKVCache&&) = default;
  NpuKVCache& operator=(NpuKVCache&&) = default;

  // --- Lifecycle & Creation ---
  static absl::StatusOr<NpuKVCache> Create(
      KVCacheUpdateMethod method,
      const ::litert::CompiledModel* npu_auxiliary_compiled_model,
      const ResolvedPrefillSignatures& prefill_signatures,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          input_kv_cache_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          prefill_output_kv_cache_slice_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          decode_output_kv_cache_slice_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          verify_output_kv_cache_slice_buffers,
      absl::flat_hash_map<absl::string_view, HWQuantParams> kv_quant_params =
          {},
      bool has_sliding_window_attention = false);

  static absl::StatusOr<NpuKVCache> CreateForTest(
      KVCacheUpdateMethod method, const ::litert::CompiledModel* compiled_model,
      InferenceContext cache_update_context,
      absl::flat_hash_map<absl::string_view, HWQuantParams> kv_quant_params =
          {},
      bool has_sliding_window_attention = false);

  void SetCompiledModel(const ::litert::CompiledModel* compiled_model) {
    compiled_model_ = compiled_model;
  }

  // --- Stage 1: Prefill ---
  absl::Status SetPrefillPositions(absl::Span<const int32_t> seq_positions);
  absl::Status RunPrefill(absl::string_view signature);

  // --- Stage 2: Decode ---
  absl::Status SetDecodePosition(int32_t step);
  absl::Status RunDecode();

  // --- Stage 3: Speculative Decoding (Verify Commit) ---
  absl::Status SetVerifyPos(int start_step);
  absl::Status CommitVerifiedKVCache(int start_step);

  // --- Accessors ---
  KVCacheUpdateMethod GetMethod() const { return method_; }
  const InferenceContext& Context() const { return cache_update_context_; }
  InferenceContext ReleaseContext() { return std::move(cache_update_context_); }

 private:
  explicit NpuKVCache(
      KVCacheUpdateMethod method, const ::litert::CompiledModel* compiled_model,
      InferenceContext cache_update_context,
      absl::flat_hash_map<absl::string_view, HWQuantParams> kv_quant_params,
      bool has_sliding_window_attention)
      : method_(method),
        compiled_model_(compiled_model),
        cache_update_context_(std::move(cache_update_context)),
        kv_quant_params_(std::move(kv_quant_params)),
        has_sliding_window_attention_(has_sliding_window_attention) {}

  KVCacheUpdateMethod method_ = KVCacheUpdateMethod::kModel;
  const ::litert::CompiledModel* compiled_model_ = nullptr;
  InferenceContext cache_update_context_;
  absl::flat_hash_map<absl::string_view, HWQuantParams> kv_quant_params_;
  bool has_sliding_window_attention_ = false;
};

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

  // Allocates the MTP drafter buffers and creates the DrafterContext.
  static absl::StatusOr<DrafterContext> Create(
      ::litert::Environment& env, const litert::Model& mtp_drafter_model,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          drafter_input_kv_cache_buffers,
      ::litert::TensorBuffer& output_activations_buffers);
};

// Holds the context for the drafter auxiliary model (MTP RoPE & Mask).
struct DrafterAuxContext {
  ::litert::CompiledModel mtp_aux_compiled_model;
  NpuRope drafter_rope;
  NpuMask drafter_mask;

  DrafterAuxContext(::litert::CompiledModel mtp_aux_compiled_model,
                    NpuRope drafter_rope, NpuMask drafter_mask)
      : mtp_aux_compiled_model(std::move(mtp_aux_compiled_model)),
        drafter_rope(std::move(drafter_rope)),
        drafter_mask(std::move(drafter_mask)) {
    this->drafter_rope.SetCompiledModel(&this->mtp_aux_compiled_model);
    this->drafter_mask.SetCompiledModel(&this->mtp_aux_compiled_model);
  }
  DrafterAuxContext(DrafterAuxContext&& other) noexcept
      : mtp_aux_compiled_model(std::move(other.mtp_aux_compiled_model)),
        drafter_rope(std::move(other.drafter_rope)),
        drafter_mask(std::move(other.drafter_mask)) {
    drafter_rope.SetCompiledModel(&mtp_aux_compiled_model);
    drafter_mask.SetCompiledModel(&mtp_aux_compiled_model);
  }
  DrafterAuxContext& operator=(DrafterAuxContext&& other) noexcept {
    if (this != &other) {
      mtp_aux_compiled_model = std::move(other.mtp_aux_compiled_model);
      drafter_rope = std::move(other.drafter_rope);
      drafter_mask = std::move(other.drafter_mask);
      drafter_rope.SetCompiledModel(&mtp_aux_compiled_model);
      drafter_mask.SetCompiledModel(&mtp_aux_compiled_model);
    }
    return *this;
  }

  // Compiles the MTP auxiliary model and initializes drafter_rope and
  // drafter_mask.
  static absl::StatusOr<DrafterAuxContext> Create(
      ::litert::Environment& env, const litert::Model& mtp_aux_model,
      const absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          drafter_aux_output_buffers,
      MaskUpdateMethod mtp_mask_update_method);
};

// Performs manual per-layer embedding lookup.
absl::Status HWPerLayerEmbeddingLookup(
    const int* token_ids, int num_tokens, const uint8_t* const* table_ptrs,
    const HWQuantizationParams* quant_params, int num_tables,
    int ple_embedding_dim, void* output_buffer, litert::ElementType output_type,
    litert::ElementType ple_table_element_type, float mul_scale = 1.0f,
    float output_scale = 1.0f, int32_t final_zero_point = 0);

// Dequantize logits to float32.
absl::Status DequantizeLogits(const ::litert::TensorBuffer& src,
                              ::litert::TensorBuffer& dst, float scale,
                              int32_t zero_point, bool should_dump);

// Writes PLE embeddings to the buffer, quantizing them to Int16 if needed.
absl::Status WritePleEmbeddings(::litert::TensorBuffer& buffer,
                                absl::Span<const float> ple_embeddings,
                                litert::ElementType output_type,
                                float final_scale, int32_t final_zero_point);

// Writes PLE embeddings to the buffer, quantizing them to Int16 if needed,
// and pads the remaining buffer space with a default embedding.
absl::Status WriteAndPadPleEmbeddings(::litert::TensorBuffer& buffer,
                                      absl::Span<const float> ple_embeddings,
                                      size_t ple_dim, size_t seq_pos_size,
                                      const std::vector<float>& default_ple_emb,
                                      litert::ElementType output_type,
                                      float final_scale,
                                      int32_t final_zero_point);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LLM_LITERT_NPU_COMPILED_MODEL_EXECUTOR_UTILS_H_
