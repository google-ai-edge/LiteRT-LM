#include "runtime/executor/llm_litert_npu_compiled_model_executor.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"  // from @litert
#include "litert/c/litert_model.h"  // from @litert
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/litert_compiled_model_executor_utils.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/litert_status_util.h"
#include "runtime/util/status_macros.h"  // NOLINT

namespace odml::infra {

namespace {
using ::litert::CompiledModel;
using ::litert::Environment;
using ::litert::Model;
using ::litert::TensorBuffer;
using ::litert::lm::CopyFromTensorBuffer;
using ::litert::lm::ExecutorInputs;
using ::litert::lm::ExecutorPrefillParams;
using ::litert::lm::GetOptimizedPrefillWorkGroups;
using ::litert::lm::ReferTensorBufferAsSpan;
using ::litert::lm::SortedPrefillSignatureMap;

// Names of the signature runners, used to get the signature runners from the
// interpreter.
constexpr char kPrefillSignatureRunner[] = "prefill_128";
constexpr int kPrefillSize = 128;
constexpr char kDecodeSignatureRunner[] = "decode";
constexpr char cache_k25[] = "kv_cache_k_25";
constexpr char cache_v25[] = "kv_cache_v_25";

// Signature names for the embedder.
struct EmbedderSignatures {
  static constexpr absl::string_view kPrefillEmbedder = "prefill_embedder_128";
  static constexpr absl::string_view kDecodeEmbedder = "decode_embedder";
  // Prefill and decode use identical tensor signature names.
  static constexpr absl::string_view kEmbedderInput = "tokens";
  static constexpr absl::string_view kEmbedderOutput = "embeds";
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
  static constexpr absl::string_view kInputEmbeddings = "input_embeds";
  static constexpr absl::string_view kDecodeLogitsOutput = "logits";
};

// Signature names for the cache update signatures.
struct CacheUpdateSignatures {
  static constexpr absl::string_view kPrefillCacheUpdate =
      "prefill_cache_update_128";
  static constexpr absl::string_view kDecodeCacheUpdate = "decode_cache_update";
  static constexpr absl::string_view kInputPos = "input_pos";
};

// Iterates through the given 'unquantized_buffer' quantizes each value and
// copies the result into the 'quantized_buffer'. Quantization is applied
// according to the 'quantization_info'.  This function assumes symmetric
// quantization, i.e. the offset is 0.
absl::Status QuantizeThenCopyValues(
    const ::litert::TensorBuffer& unquantized_buffer,
    ::litert::TensorBuffer& quantized_buffer,
    const LiteRtQuantizationPerTensor quantization_info) {
  LITERT_ASSIGN_OR_RETURN_ABSL(
      auto float_values, ReferTensorBufferAsSpan<float>(unquantized_buffer));
  LITERT_ASSIGN_OR_RETURN(
      auto quantized_buffer_lock_and_addr,
      ::litert::TensorBufferScopedLock::Create(quantized_buffer));
  int16_t* quantized_buffer_ptr =
      static_cast<int16_t*>(quantized_buffer_lock_and_addr.second);
  if (quantization_info.scale == 0.0f) {
    return absl::InvalidArgumentError("Quantization scale must be non-zero.");
  }
  float inversed_scale = 1.0f / quantization_info.scale;

  for (int i = 0; i < float_values.size(); ++i) {
    float tmp = float_values[i] * inversed_scale;
    // Compute the nearest integer value to num (in floating-point format).
    float rounded_float = std::round(tmp);
    int16_t rounded_int = -1;
    // Clip the value to the range of int16_t.
    if (rounded_float > std::numeric_limits<int16_t>::max()) {
      rounded_int = std::numeric_limits<int16_t>::max();
    } else if (rounded_float < std::numeric_limits<int16_t>::min()) {
      rounded_int = std::numeric_limits<int16_t>::min();
    } else {
      rounded_int = static_cast<int16_t>(rounded_float);
    }
    quantized_buffer_ptr[i] = rounded_int;
  }

  return absl::OkStatus();
}

// Iterates through the given 'quantized_buffer' de-quantizes each value and
// copies the result into the 'unquantized_buffer'. De-quantization is applied
// according to the 'quantization_info'.  This function assumes symmetric
// quantization, i.e. the offset is 0.
absl::Status DequantizeThenCopyValues(
    const ::litert::TensorBuffer& quantized_buffer,
    ::litert::TensorBuffer& unquantized_buffer,
    const LiteRtQuantizationPerTensor quantization_info) {
  LITERT_ASSIGN_OR_RETURN_ABSL(std::vector<int16_t> int16_values,
                               CopyFromTensorBuffer<int16_t>(quantized_buffer));
  LITERT_ASSIGN_OR_RETURN(
      auto unquantized_buffer_lock_and_addr,
      ::litert::TensorBufferScopedLock::Create(unquantized_buffer));
  float* unquantized_buffer_ptr =
      static_cast<float*>(unquantized_buffer_lock_and_addr.second);
  for (int i = 0; i < int16_values.size(); ++i) {
    unquantized_buffer_ptr[i] =
        (float)int16_values[i] * quantization_info.scale;
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<LlmLiteRtNpuCompiledModelExecutor::EmbedderContext>
LlmLiteRtNpuCompiledModelExecutor::CreateEmbedderContextWithoutBufferSharing(
    Environment& env, const std::string& embedder_model) {
  LITERT_ASSIGN_OR_RETURN(Model embedder_lrt_model,
                          Model::CreateFromFile(embedder_model));
  LITERT_ASSIGN_OR_RETURN(
      CompiledModel embedder_compiled_model,
      CompiledModel::Create(env, embedder_lrt_model, kLiteRtHwAcceleratorCpu));

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
      embedder_compiled_model.CreateInputBuffer(
          EmbedderSignatures::kPrefillEmbedder,
          EmbedderSignatures::kEmbedderInput));

  LITERT_ASSIGN_OR_RETURN(
      prefill_output_buffers[EmbedderSignatures::kEmbedderOutput],
      embedder_compiled_model.CreateOutputBuffer(
          EmbedderSignatures::kPrefillEmbedder,
          EmbedderSignatures::kEmbedderOutput));

  LITERT_ASSIGN_OR_RETURN(
      decode_input_buffers[EmbedderSignatures::kEmbedderInput],
      embedder_compiled_model.CreateInputBuffer(
          EmbedderSignatures::kDecodeEmbedder,
          EmbedderSignatures::kEmbedderInput));

  LITERT_ASSIGN_OR_RETURN(
      decode_output_buffers[EmbedderSignatures::kEmbedderOutput],
      embedder_compiled_model.CreateOutputBuffer(
          EmbedderSignatures::kDecodeEmbedder,
          EmbedderSignatures::kEmbedderOutput));

  EmbedderContext embedder_context(
      std::move(embedder_lrt_model), std::move(embedder_compiled_model),
      std::move(prefill_input_buffers), std::move(prefill_output_buffers),
      std::move(decode_input_buffers), std::move(decode_output_buffers));
  return embedder_context;
}

absl::StatusOr<LlmLiteRtNpuCompiledModelExecutor::EmbedderContext>
LlmLiteRtNpuCompiledModelExecutor::CreateEmbedderContextWithBufferSharing(
    ::litert::Environment& env, litert::Model& embedder_lrt_model,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        gemma_prefill_input_buffers,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        gemma_decode_input_buffers) {
  LITERT_ASSIGN_OR_RETURN(
      CompiledModel embedder_compiled_model,
      CompiledModel::Create(env, embedder_lrt_model, kLiteRtHwAcceleratorCpu));

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
      embedder_compiled_model.CreateInputBuffer(
          EmbedderSignatures::kPrefillEmbedder,
          EmbedderSignatures::kEmbedderInput));

  LITERT_ASSIGN_OR_RETURN(
      prefill_output_buffers[EmbedderSignatures::kEmbedderOutput],
      gemma_prefill_input_buffers[LlmSignatures::kInputEmbeddings].Duplicate());

  LITERT_ASSIGN_OR_RETURN(
      decode_input_buffers[EmbedderSignatures::kEmbedderInput],
      embedder_compiled_model.CreateInputBuffer(
          EmbedderSignatures::kDecodeEmbedder,
          EmbedderSignatures::kEmbedderInput));

  LITERT_ASSIGN_OR_RETURN(
      decode_output_buffers[EmbedderSignatures::kEmbedderOutput],
      gemma_decode_input_buffers[LlmSignatures::kInputEmbeddings].Duplicate());

  EmbedderContext embedder_context(
      std::move(embedder_lrt_model), std::move(embedder_compiled_model),
      std::move(prefill_input_buffers), std::move(prefill_output_buffers),
      std::move(decode_input_buffers), std::move(decode_output_buffers));
  return embedder_context;
}

absl::StatusOr<LlmLiteRtNpuCompiledModelExecutor::NpuAuxiliaryContext>
LlmLiteRtNpuCompiledModelExecutor::CreateNpuAuxiliaryContext(
    ::litert::Environment& env, litert::Model& npu_auxiliary_lrt_model) {
  LITERT_ASSIGN_OR_RETURN(auto npu_auxiliary_compiled_model,
                          CompiledModel::Create(env, npu_auxiliary_lrt_model,
                                                kLiteRtHwAcceleratorCpu));
  NpuAuxiliaryContext npu_auxiliary_context(
      std::move(npu_auxiliary_lrt_model),
      std::move(npu_auxiliary_compiled_model));
  return npu_auxiliary_context;
}

absl::StatusOr<LlmLiteRtNpuCompiledModelExecutor::InferenceContext>
LlmLiteRtNpuCompiledModelExecutor::CreateMaskContextWithoutBufferSharing(
    NpuAuxiliaryContext& npu_auxiliary_context, const std::string& mask_model,
    ::litert::TensorBuffer prefill_input_tokens,
    ::litert::TensorBuffer decode_input_tokens) {
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
  prefill_input_buffers[MaskSignatures::kMaskInputTokens] =
      std::move(prefill_input_tokens);

  LITERT_ASSIGN_OR_RETURN(
      prefill_output_buffers[MaskSignatures::kMaskOutputLocalMask],
      npu_auxiliary_context.npu_auxiliary_compiled_model.CreateOutputBuffer(
          MaskSignatures::kPrefillMask, MaskSignatures::kMaskOutputLocalMask));
  LITERT_ASSIGN_OR_RETURN(
      prefill_output_buffers[MaskSignatures::kMaskOutputGlobalMask],
      npu_auxiliary_context.npu_auxiliary_compiled_model.CreateOutputBuffer(
          MaskSignatures::kPrefillMask, MaskSignatures::kMaskOutputGlobalMask));
  LITERT_ASSIGN_OR_RETURN(
      decode_input_buffers[MaskSignatures::kMaskInputTimeStep],
      npu_auxiliary_context.npu_auxiliary_compiled_model.CreateInputBuffer(
          MaskSignatures::kDecodeMask, MaskSignatures::kMaskInputTimeStep));
  decode_input_buffers[MaskSignatures::kMaskInputTokens] =
      std::move(decode_input_tokens);

  LITERT_ASSIGN_OR_RETURN(
      decode_output_buffers[MaskSignatures::kMaskOutputLocalMask],
      npu_auxiliary_context.npu_auxiliary_compiled_model.CreateOutputBuffer(
          MaskSignatures::kDecodeMask, MaskSignatures::kMaskOutputLocalMask));

  LITERT_ASSIGN_OR_RETURN(
      decode_output_buffers[MaskSignatures::kMaskOutputGlobalMask],
      npu_auxiliary_context.npu_auxiliary_compiled_model.CreateOutputBuffer(
          MaskSignatures::kDecodeMask, MaskSignatures::kMaskOutputGlobalMask));

  InferenceContext mask_context(
      std::move(prefill_input_buffers), std::move(prefill_output_buffers),
      std::move(decode_input_buffers), std::move(decode_output_buffers));
  return mask_context;
}

absl::StatusOr<LlmLiteRtNpuCompiledModelExecutor::InferenceContext>
LlmLiteRtNpuCompiledModelExecutor::CreateMaskContextWithBufferSharing(
    NpuAuxiliaryContext& npu_auxiliary_context,
    ::litert::TensorBuffer prefill_input_tokens,
    ::litert::TensorBuffer decode_input_tokens,
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
  prefill_input_buffers[MaskSignatures::kMaskInputTokens] =
      std::move(prefill_input_tokens);

  const std::set<absl::string_view> mask_output_names = {
      MaskSignatures::kMaskOutputLocalMask,
      MaskSignatures::kMaskOutputGlobalMask};
  for (const auto& mask_output_name : mask_output_names) {
    LITERT_ASSIGN_OR_RETURN(
        prefill_output_buffers[mask_output_name],
        gemma_prefill_input_buffers[mask_output_name].Duplicate());
  }

  LITERT_ASSIGN_OR_RETURN(
      decode_input_buffers[MaskSignatures::kMaskInputTimeStep],
      npu_auxiliary_context.npu_auxiliary_compiled_model.CreateInputBuffer(
          MaskSignatures::kDecodeMask, MaskSignatures::kMaskInputTimeStep));
  decode_input_buffers[MaskSignatures::kMaskInputTokens] =
      std::move(decode_input_tokens);

  for (const auto& mask_output_name : mask_output_names) {
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
LlmLiteRtNpuCompiledModelExecutor::CreateRopeContextWithoutBufferSharing(
    LlmLiteRtNpuCompiledModelExecutor::NpuAuxiliaryContext&
        npu_auxiliary_context,
    const std::string& rope_model) {
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

  LITERT_ASSIGN_OR_RETURN(
      prefill_output_buffers[RopeSignatures::kOutputPosEmbeddingLocalLow],
      npu_auxiliary_context.npu_auxiliary_compiled_model.CreateOutputBuffer(
          RopeSignatures::kPrefillRope,
          RopeSignatures::kOutputPosEmbeddingLocalLow));

  LITERT_ASSIGN_OR_RETURN(
      prefill_output_buffers[RopeSignatures::kOutputPosEmbeddingHigh],
      npu_auxiliary_context.npu_auxiliary_compiled_model.CreateOutputBuffer(
          RopeSignatures::kPrefillRope,
          RopeSignatures::kOutputPosEmbeddingHigh));

  LITERT_ASSIGN_OR_RETURN(
      prefill_output_buffers[RopeSignatures::kOutputPosEmbeddingLocalHigh],
      npu_auxiliary_context.npu_auxiliary_compiled_model.CreateOutputBuffer(
          RopeSignatures::kPrefillRope,
          RopeSignatures::kOutputPosEmbeddingLocalHigh));

  LITERT_ASSIGN_OR_RETURN(
      prefill_output_buffers[RopeSignatures::kOutputPosEmbeddingLow],
      npu_auxiliary_context.npu_auxiliary_compiled_model.CreateOutputBuffer(
          RopeSignatures::kPrefillRope,
          RopeSignatures::kOutputPosEmbeddingLow));

  LITERT_ASSIGN_OR_RETURN(
      decode_input_buffers[RopeSignatures::kInputPos],
      npu_auxiliary_context.npu_auxiliary_compiled_model.CreateInputBuffer(
          RopeSignatures::kDecodeRope, RopeSignatures::kInputPos));

  LITERT_ASSIGN_OR_RETURN(
      decode_output_buffers[RopeSignatures::kOutputPosEmbeddingLocalLow],
      npu_auxiliary_context.npu_auxiliary_compiled_model.CreateOutputBuffer(
          RopeSignatures::kDecodeRope,
          RopeSignatures::kOutputPosEmbeddingLocalLow));
  LITERT_ASSIGN_OR_RETURN(
      decode_output_buffers[RopeSignatures::kOutputPosEmbeddingHigh],
      npu_auxiliary_context.npu_auxiliary_compiled_model.CreateOutputBuffer(
          RopeSignatures::kDecodeRope,
          RopeSignatures::kOutputPosEmbeddingHigh));
  LITERT_ASSIGN_OR_RETURN(
      decode_output_buffers[RopeSignatures::kOutputPosEmbeddingLocalHigh],
      npu_auxiliary_context.npu_auxiliary_compiled_model.CreateOutputBuffer(
          RopeSignatures::kDecodeRope,
          RopeSignatures::kOutputPosEmbeddingLocalHigh));
  LITERT_ASSIGN_OR_RETURN(
      decode_output_buffers[RopeSignatures::kOutputPosEmbeddingLow],
      npu_auxiliary_context.npu_auxiliary_compiled_model.CreateOutputBuffer(
          RopeSignatures::kDecodeRope, RopeSignatures::kOutputPosEmbeddingLow));

  InferenceContext rope_context(
      std::move(prefill_input_buffers), std::move(prefill_output_buffers),
      std::move(decode_input_buffers), std::move(decode_output_buffers));
  return rope_context;
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

  const std::set<absl::string_view> rope_output_names = {
      RopeSignatures::kOutputPosEmbeddingLocalLow,
      RopeSignatures::kOutputPosEmbeddingHigh,
      RopeSignatures::kOutputPosEmbeddingLocalHigh,
      RopeSignatures::kOutputPosEmbeddingLow};
  for (const auto& rope_output_name : rope_output_names) {
    LITERT_ASSIGN_OR_RETURN(
        prefill_output_buffers[rope_output_name],
        gemma_prefill_input_buffers[rope_output_name].Duplicate());
  }

  LITERT_ASSIGN_OR_RETURN(
      decode_input_buffers[RopeSignatures::kInputPos],
      npu_auxiliary_context.npu_auxiliary_compiled_model.CreateInputBuffer(
          RopeSignatures::kDecodeRope, RopeSignatures::kInputPos));

  for (const auto& rope_output_name : rope_output_names) {
    LITERT_ASSIGN_OR_RETURN(
        decode_output_buffers[rope_output_name],
        gemma_decode_input_buffers[rope_output_name].Duplicate());
  }

  InferenceContext rope_context(
      std::move(prefill_input_buffers), std::move(prefill_output_buffers),
      std::move(decode_input_buffers), std::move(decode_output_buffers));
  return rope_context;
}

absl::StatusOr<LlmLiteRtNpuCompiledModelExecutor::InferenceContext>
LlmLiteRtNpuCompiledModelExecutor::
    CreateLlmInferenceContextWithoutBufferSharing(
        ::litert::CompiledModel& llm_compiled_model,
        const absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
            input_kv_cache_buffers,
        const absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
            prefill_output_kv_cache_slice_buffers,
        const absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
            decode_output_kv_cache_slice_buffers,
        const absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
            prefill_input_buffers_ext,
        const absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
            decode_input_buffers_ext) {
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      prefill_input_buffers;
  {
    // Duplicate all provided inputs to prefill inputs.
    for (const auto& [key, value] : prefill_input_buffers_ext) {
      LITERT_ASSIGN_OR_RETURN(prefill_input_buffers[key], value.Duplicate());
    }
    // Duplicate all kv cache buffers to prefill inputs.
    for (const auto& [key, value] : input_kv_cache_buffers) {
      LITERT_ASSIGN_OR_RETURN(prefill_input_buffers[key], value.Duplicate());
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
    // Duplicate all provided outputs to prefill inputs.
    for (const auto& [key, value] : decode_input_buffers_ext) {
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

    // Create the decode logits output buffer.
    LITERT_ASSIGN_OR_RETURN(
        decode_output_buffers[LlmSignatures::kDecodeLogitsOutput],
        llm_compiled_model.CreateOutputBuffer(
            kDecodeSignatureRunner, LlmSignatures::kDecodeLogitsOutput));
  }
  return InferenceContext(
      std::move(prefill_input_buffers), std::move(prefill_output_buffers),
      std::move(decode_input_buffers), std::move(decode_output_buffers));
}

absl::StatusOr<LlmLiteRtNpuCompiledModelExecutor::InferenceContext>
LlmLiteRtNpuCompiledModelExecutor::CreateLlmInferenceContextWithBufferSharing(
    ::litert::Environment& env, ::litert::Model& llm_model,
    ::litert::CompiledModel& llm_compiled_model,
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
    for (const auto& [key, value] : input_kv_cache_buffers) {
      LITERT_ASSIGN_OR_RETURN(prefill_input_buffers[key], value.Duplicate());
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

    // TODO(b/405424188): Buffers kv_cache_{k,v}_25 have float element type for
    // the prefill signature but int16_t for the decode signature. Therefore,
    // unlike for the other KV cache tensors, we can not re-use the same tensor
    // during prefill and decode (because trying to register a tensor of element
    // type float for the decode signature that expects it in int16_t will
    // fail). Luckily these buffers are not used, so we can simply create new
    // ones to satisfy the compiled model run API.  We can remove this
    // workaround once we have a model that removes these buffers.
    LITERT_ASSIGN_OR_RETURN(decode_input_buffers[cache_k25],
                            llm_compiled_model.CreateInputBuffer(
                                kDecodeSignatureRunner, cache_k25));
    LITERT_ASSIGN_OR_RETURN(decode_input_buffers[cache_v25],
                            llm_compiled_model.CreateInputBuffer(
                                kDecodeSignatureRunner, cache_v25));
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
            kDecodeSignatureRunner, LlmSignatures::kDecodeLogitsOutput));
  }
  return InferenceContext(
      std::move(prefill_input_buffers), std::move(prefill_output_buffers),
      std::move(decode_input_buffers), std::move(decode_output_buffers));
}

absl::StatusOr<LlmLiteRtNpuCompiledModelExecutor::InferenceContext>
LlmLiteRtNpuCompiledModelExecutor::
    CreateCacheUpdateInferenceContextWithoutBufferSharing(
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

absl::StatusOr<LlmLiteRtNpuCompiledModelExecutor::InferenceContext>
LlmLiteRtNpuCompiledModelExecutor::
    CreateCacheUpdateInferenceContextWithoutBufferSharing(
        ::litert::Model& auxiliary_model,
        ::litert::CompiledModel& compiled_auxiliary_model,
        ::litert::TensorBuffer prefill_input_pos,
        ::litert::TensorBuffer decode_input_pos)

{
  auto prefill_signature =
      auxiliary_model.FindSignature(CacheUpdateSignatures::kPrefillCacheUpdate);
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      prefill_input_buffers;
  {
    // Move the provided (duplicated) input pos buffer but create other inputs.
    prefill_input_buffers[CacheUpdateSignatures::kInputPos] =
        std::move(prefill_input_pos);
    for (auto input_name : prefill_signature->InputNames()) {
      if (input_name == CacheUpdateSignatures::kInputPos) {
        continue;
      }
      LITERT_ASSIGN_OR_RETURN(
          prefill_input_buffers[input_name],
          compiled_auxiliary_model.CreateInputBuffer(
              CacheUpdateSignatures::kPrefillCacheUpdate, input_name));
    }
  }
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      prefill_output_buffers;
  {
    // The cache update model output should re-use the KV cache that is already
    // available in the input buffers.
    for (auto output_name : prefill_signature->OutputNames()) {
      ::litert::TensorBuffer& output_buffer =
          prefill_input_buffers[output_name];
      LITERT_ASSIGN_OR_RETURN(prefill_output_buffers[output_name],
                              output_buffer.Duplicate());
    }
  }

  auto decode_signature =
      auxiliary_model.FindSignature(CacheUpdateSignatures::kDecodeCacheUpdate);
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      decode_input_buffers;
  {
    decode_input_buffers[CacheUpdateSignatures::kInputPos] =
        std::move(decode_input_pos);
    for (auto input_name : decode_signature->InputNames()) {
      if (input_name == CacheUpdateSignatures::kInputPos) {
        continue;
      }
      ::litert::TensorBuffer& buffer = prefill_input_buffers[input_name];
      LITERT_ASSIGN_OR_RETURN(decode_input_buffers[input_name],
                              buffer.Duplicate());
    }
  }
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      decode_output_buffers;
  {
    // The cache update model should re-use the KV cache that is already
    // available in the input buffers.
    for (auto output_name : decode_signature->OutputNames()) {
      ::litert::TensorBuffer& output_buffer = decode_input_buffers[output_name];
      LITERT_ASSIGN_OR_RETURN(decode_output_buffers[output_name],
                              output_buffer.Duplicate());
    }
  }
  return InferenceContext(
      std::move(prefill_input_buffers), std::move(prefill_output_buffers),
      std::move(decode_input_buffers), std::move(decode_output_buffers));
}

absl::StatusOr<std::unique_ptr<LlmLiteRtNpuCompiledModelExecutor>>
LlmLiteRtNpuCompiledModelExecutor::CreateInternalAllQuantized(
    const std::string& llm_model, const std::string& embedder_model,
    const std::string& npu_auxiliary_model,
    const std::optional<std::string>& dispatch_library_path) {
  // TODO(b/405424188): Remove 'Create' functions that require the caller
  // to provide tflite files, we should only be using the litertlm file
  // format.
  return absl::UnimplementedError("This method is not implemented.");
}

absl::Status LlmLiteRtNpuCompiledModelExecutor::WarmupInference(
    ::litert::CompiledModel& compiled_model_llm,
    const InferenceContext& llm_inference_context,
    ::litert::CompiledModel& compiled_model_auxiliary,
    const InferenceContext& rope_inference_context,
    const InferenceContext& mask_inference_context,
    const InferenceContext& cache_update_inference_context) {
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
    Model embedder_model, CompiledModel embedder_compiled_model,
    absl::flat_hash_map<absl::string_view, TensorBuffer> prefill_input_buffers,
    absl::flat_hash_map<absl::string_view, TensorBuffer> prefill_output_buffers,
    absl::flat_hash_map<absl::string_view, TensorBuffer> decode_input_buffers,
    absl::flat_hash_map<absl::string_view, TensorBuffer> decode_output_buffers)
    : embedder_model(std::move(embedder_model)),
      embedder_compiled_model(std::move(embedder_compiled_model)),
      inference_context(
          std::move(prefill_input_buffers), std::move(prefill_output_buffers),
          std::move(decode_input_buffers), std::move(decode_output_buffers)) {}

LlmLiteRtNpuCompiledModelExecutor::NpuAuxiliaryContext::NpuAuxiliaryContext(
    Model npu_auxiliary_model, CompiledModel npu_auxiliary_compiled_model)
    : npu_auxiliary_model(std::move(npu_auxiliary_model)),
      npu_auxiliary_compiled_model(std::move(npu_auxiliary_compiled_model)) {}

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

  auto end = absl::Now();
  latency_stats_.prefill_e2e_latency_us +=
      absl::ToInt64Microseconds(end - start);

  return absl::OkStatus();
}

absl::Status LlmLiteRtNpuCompiledModelExecutor::Decode(
    TensorBuffer& output_tokens) {
  auto start = absl::Now();
  ::litert::TensorBuffer& decoded_logits =
      llm_inference_context_
          .decode_output_buffers[LlmSignatures::kDecodeLogitsOutput];
  RETURN_IF_ERROR(Decode(ExecutorInputs(), decoded_logits));
  auto start_sample = absl::Now();
  LITERT_ASSIGN_OR_RETURN(auto logits_buffer_int16,
                          CopyFromTensorBuffer<int16_t>(decoded_logits));
  int max_index = 0;
  int16_t max_value = logits_buffer_int16[0];
  for (int i = 1; i < logits_buffer_int16.size(); ++i) {
    if (logits_buffer_int16[i] > max_value) {
      max_value = logits_buffer_int16[i];
      max_index = i;
    }
  }

  latency_stats_.decode_sampling_latency_us +=
      absl::ToInt64Microseconds(absl::Now() - start_sample);

  next_input_token_id_ = max_index;
  output_tokens.Write(absl::MakeConstSpan({max_index}));
  auto end = absl::Now();
  latency_stats_.decode_e2e_latency_us +=
      absl::ToInt64Microseconds(end - start);
  latency_stats_.decode_num_tokens += 1;
  return absl::OkStatus();
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
                .prefill_input_buffers[EmbedderSignatures::kEmbedderInput]));
    auto* prefill_input_ptr =
        static_cast<int32_t*>(prefill_input_lock_and_addr.second);

    // Prefill input position.
    LITERT_ASSIGN_OR_RETURN(
        auto prefill_input_pos_size,
        rope_context_.prefill_input_buffers[RopeSignatures::kInputPos].Size());
    LITERT_ASSIGN_OR_RETURN(
        auto prefill_input_pos_lock_and_addr,
        ::litert::TensorBufferScopedLock::Create(
            rope_context_.prefill_input_buffers[RopeSignatures::kInputPos]));
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
                .prefill_input_buffers[MaskSignatures::kMaskInputTimeStep]));
    auto* prefill_timestep_ptr =
        static_cast<int32_t*>(prefill_timestep_lock_and_addr.second);

    memset(prefill_input_ptr, 0, prefill_input_size);
    memset(prefill_input_pos_ptr, 0, prefill_input_pos_size);
    memset(prefill_timestep_ptr, 0, prefill_timestep_size);

    // We will not fill the last token of the current input into the interpreter
    // now. It will be stored in next_input_token_id_ and used in the next
    // prefill or decode.
    int start_step = current_step_;
    prefill_timestep_ptr[0] = start_step;
    for (int i = 0, input_idx = 0; i < ids.size() - 1;
         input_idx++, current_step_++) {
      if (next_input_token_id_ != -1) {
        // Use next_input_token_id_ if it is valid.
        // Currently we use -1 to indicate that next_input_token_id_ is invalid.
        prefill_input_ptr[input_idx] = next_input_token_id_;
        // next_input_token_id_ should only be used once at the beginning of the
        // loop.
        next_input_token_id_ = -1;
      } else {
        prefill_input_ptr[input_idx] = ids[i];
        // Only increase i if we used the token inside ids.
        i++;
      }
      prefill_input_pos_ptr[input_idx] = current_step_;
    }
  }
  next_input_token_id_ = ids[ids.size() - 1];
  auto end_prepare_inputs = absl::Now();
  latency_stats_.prefill_prepare_input_latency_us +=
      absl::ToInt64Microseconds(end_prepare_inputs - start_prepare_inputs);

  // Invoke embedder signature.
  {
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
    // The Gemma model expects quantized inputs, but the buffers of all
    // auxiliary models are not quantized. So we need to quantize them here on
    // the fly.
    auto start = absl::Now();
    if (model_quantization_ ==
        ModelQuantization::kTransformerStackOnlyQuantized) {
      for (auto& [input_name, quantized_input_buffer] :
           llm_inference_context_.prefill_input_buffers) {
        if (input_name == cache_v25 || input_name == cache_k25) {
          continue;
        }
        // Get tensor and check it has quantization.
        LITERT_ASSIGN_OR_RETURN(
            auto prefill_subgraph,
            llm_model_.Subgraph(LlmSignatures::kPrefillLlm));
        auto tensor = prefill_subgraph.Input(input_name);
        RET_CHECK(tensor->HasQuantization());
        auto quantization_info = tensor->PerTensorQuantization();
        if (input_name == LlmSignatures::kInputEmbeddings) {
          ::litert::TensorBuffer& input_embeds =
              embedder_context_.inference_context
                  .prefill_output_buffers[EmbedderSignatures::kEmbedderOutput];
          RETURN_IF_ERROR(QuantizeThenCopyValues(
              input_embeds, quantized_input_buffer, quantization_info));
        } else if (absl::StartsWith(input_name, "kv_cache_")) {
          ::litert::TensorBuffer& input_kv_cache =
              cache_update_inference_context_
                  .prefill_output_buffers[input_name];
          RETURN_IF_ERROR(QuantizeThenCopyValues(
              input_kv_cache, quantized_input_buffer, quantization_info));
        } else if (absl::StartsWith(input_name, "mask_")) {
          ::litert::TensorBuffer& input_mask =
              mask_context_.prefill_output_buffers[input_name];
          RETURN_IF_ERROR(QuantizeThenCopyValues(
              input_mask, quantized_input_buffer, quantization_info));
        } else if (absl::StartsWith(input_name, "pos_emb_")) {
          ::litert::TensorBuffer& input_pos_emb =
              rope_context_.prefill_output_buffers[input_name];
          RETURN_IF_ERROR(QuantizeThenCopyValues(
              input_pos_emb, quantized_input_buffer, quantization_info));
        }
      }
    }

    auto end = absl::Now();
    latency_stats_.prefill_quantization_latency_us +=
        absl::ToInt64Microseconds(end - start);

    start = absl::Now();
    auto res =
        llm_compiled_model_.Run(LlmSignatures::kPrefillLlm,
                                llm_inference_context_.prefill_input_buffers,
                                llm_inference_context_.prefill_output_buffers);
    RET_CHECK(res) << "Failed to run LLM model." << res.Error().Message();
    end = absl::Now();
    latency_stats_.prefill_llm_inference_latency_us +=
        absl::ToInt64Microseconds(end - start);
  }

  // Cache update.
  {
    auto start = absl::Now();
    // The cache model expects non-quantized inputs, but the buffers of the
    // Gemma model are quantized. So we need to de-quantize them here on
    // the fly.
    if (model_quantization_ ==
        ModelQuantization::kTransformerStackOnlyQuantized) {
      for (auto& [input_name, input_buffer] :
           cache_update_inference_context_.prefill_input_buffers) {
        if (absl::StartsWith(input_name, "kv_slice_")) {
          LITERT_ASSIGN_OR_RETURN(
              auto prefill_subgraph,
              llm_model_.Subgraph(LlmSignatures::kPrefillLlm));
          auto tensor = prefill_subgraph.Output(input_name);
          RET_CHECK(tensor->HasQuantization());
          auto quantization_info = tensor->PerTensorQuantization();
          ::litert::TensorBuffer& quantized_kv_slice =
              llm_inference_context_.prefill_output_buffers[input_name];
          RETURN_IF_ERROR(DequantizeThenCopyValues(
              quantized_kv_slice, input_buffer, quantization_info));
        }
      }
    }

    auto end = absl::Now();
    latency_stats_.prefill_quantization_latency_us +=
        absl::ToInt64Microseconds(end - start);

    start = absl::Now();
    auto res = npu_auxiliary_context_.npu_auxiliary_compiled_model.Run(
        CacheUpdateSignatures::kPrefillCacheUpdate,
        cache_update_inference_context_.prefill_input_buffers,
        cache_update_inference_context_.prefill_output_buffers);
    end = absl::Now();
    latency_stats_.prefill_cache_update_inference_latency_us +=
        absl::ToInt64Microseconds(end - start);
    RET_CHECK(res) << "Failed to run cache update model."
                   << res.Error().Message();
  }
  return absl::OkStatus();
}

absl::Status LlmLiteRtNpuCompiledModelExecutor::Decode(
    const ExecutorInputs& inputs, TensorBuffer& output_logits) {
  auto start_prepare_inputs = absl::Now();
  int id = next_input_token_id_;
  if (inputs.GetTextDataPtr().ok()) {
    auto input_tensor_size = (*inputs.GetTextTokenIdsPtr())->Size();
    if (input_tensor_size && *input_tensor_size != 0) {
      // Input token ids provided, so use it regardless of whether next input
      // token id is set. Only accept batch size 1 and a single token for now.
      RET_CHECK_EQ(*input_tensor_size, 1 * sizeof(int32_t));
      LITERT_ASSIGN_OR_RETURN_ABSL(
          auto ids,
          ReferTensorBufferAsSpan<int32_t>(*(*inputs.GetTextTokenIdsPtr())));
      id = ids[0];
    }
  }
  if (id == -1) {
    return absl::InvalidArgumentError("No id available to be decoded.");
  }

  // Invalidate the previous next_input_token_id_, regardless of whether it is
  // used.
  next_input_token_id_ = -1;

  {
    // Decode input tokens.
    LITERT_ASSIGN_OR_RETURN(
        auto decode_input_lock_and_addr,
        ::litert::TensorBufferScopedLock::Create(
            embedder_context_.inference_context
                .decode_input_buffers[EmbedderSignatures::kEmbedderInput]));
    auto* decode_input_ptr =
        static_cast<int32_t*>(decode_input_lock_and_addr.second);
    decode_input_ptr[0] = id;

    // Decode input position
    LITERT_ASSIGN_OR_RETURN(
        auto decode_input_pos_lock_and_addr,
        ::litert::TensorBufferScopedLock::Create(
            rope_context_.decode_input_buffers[RopeSignatures::kInputPos]));
    auto* decode_input_pos_ptr =
        static_cast<int32_t*>(decode_input_pos_lock_and_addr.second);
    decode_input_pos_ptr[0] = current_step_;

    // Timestep input.
    LITERT_ASSIGN_OR_RETURN(
        auto decode_timestep_lock_and_addr,
        ::litert::TensorBufferScopedLock::Create(
            mask_context_
                .decode_input_buffers[MaskSignatures::kMaskInputTimeStep]));
    auto* decode_timestep_ptr =
        static_cast<int32_t*>(decode_timestep_lock_and_addr.second);
    decode_timestep_ptr[0] = current_step_;
  }
  auto end_prepare_inputs = absl::Now();
  latency_stats_.decode_prepare_input_latency_us +=
      absl::ToInt64Microseconds(end_prepare_inputs - start_prepare_inputs);

  // Invoke embedder signature.
  {
    auto start = absl::Now();
    auto res = embedder_context_.embedder_compiled_model.Run(
        EmbedderSignatures::kDecodeEmbedder,
        embedder_context_.inference_context.decode_input_buffers,
        embedder_context_.inference_context.decode_output_buffers);
    RET_CHECK(res) << "Failed to run embedder model." << res.Error().Message();
    auto end = absl::Now();
    latency_stats_.decode_embedder_inference_latency_us +=
        absl::ToInt64Microseconds(end - start);
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
    // The Gemma model expects quantized inputs, but the buffers of all
    // auxiliary models are not quantized. So we need to quantize them here on
    // the fly.
    auto start = absl::Now();
    if (model_quantization_ ==
        ModelQuantization::kTransformerStackOnlyQuantized) {
      for (auto& [input_name, quantized_input_buffer] :
           llm_inference_context_.decode_input_buffers) {
        // Get tensor and check it has quantization.
        LITERT_ASSIGN_OR_RETURN(auto decode_subgraph,
                                llm_model_.Subgraph(LlmSignatures::kDecodeLlm));
        auto tensor = decode_subgraph.Input(input_name);
        RET_CHECK(tensor->HasQuantization());
        auto quantization_info = tensor->PerTensorQuantization();
        if (input_name == LlmSignatures::kInputEmbeddings) {
          ::litert::TensorBuffer& input_embeds =
              embedder_context_.inference_context
                  .decode_output_buffers[EmbedderSignatures::kEmbedderOutput];
          RETURN_IF_ERROR(QuantizeThenCopyValues(
              input_embeds, quantized_input_buffer, quantization_info));
        } else if (absl::StartsWith(input_name, "kv_cache_")) {
          ::litert::TensorBuffer& input_kv_cache =
              cache_update_inference_context_.decode_output_buffers[input_name];
          RETURN_IF_ERROR(QuantizeThenCopyValues(
              input_kv_cache, quantized_input_buffer, quantization_info));
        } else if (absl::StartsWith(input_name, "mask_")) {
          ::litert::TensorBuffer& input_mask =
              mask_context_.decode_output_buffers[input_name];
          RETURN_IF_ERROR(QuantizeThenCopyValues(
              input_mask, quantized_input_buffer, quantization_info));
        } else if (absl::StartsWith(input_name, "pos_emb_")) {
          ::litert::TensorBuffer& input_pos_emb =
              rope_context_.decode_output_buffers[input_name];
          RETURN_IF_ERROR(QuantizeThenCopyValues(
              input_pos_emb, quantized_input_buffer, quantization_info));
        }
      }
    }
    auto end = absl::Now();
    latency_stats_.decode_quantization_latency_us +=
        absl::ToInt64Microseconds(end - start);

    start = absl::Now();
    auto res = llm_compiled_model_.Run(
        LlmSignatures::kDecodeLlm, llm_inference_context_.decode_input_buffers,
        llm_inference_context_.decode_output_buffers);
    end = absl::Now();
    latency_stats_.decode_llm_inference_latency_us +=
        absl::ToInt64Microseconds(end - start);
    RET_CHECK(res) << "Failed to run LLM model." << res.Error().Message();
  }

  // Cache update.
  {
    auto start = absl::Now();
    // The cache model expects non-quantized inputs, but the buffers of the
    // Gemma model are quantized. So we need to de-quantize them here on
    // the fly.
    if (model_quantization_ ==
        ModelQuantization::kTransformerStackOnlyQuantized) {
      for (auto& [input_name, input_buffer] :
           cache_update_inference_context_.decode_input_buffers) {
        if (!absl::StartsWith(input_name, "kv_slice_")) {
          continue;
        }
        // If it is a kv_slice input, we need to dequantize it and we know
        // the gemma model has it as an output tensor.
        // Get tensor and check it has quantization.
        LITERT_ASSIGN_OR_RETURN(auto decode_subgraph,
                                llm_model_.Subgraph(LlmSignatures::kDecodeLlm));
        auto tensor = decode_subgraph.Output(input_name);
        RET_CHECK(tensor->HasQuantization());
        auto quantization_info = tensor->PerTensorQuantization();
        ::litert::TensorBuffer& quantized_kv_slice =
            llm_inference_context_.decode_output_buffers[input_name];
        RETURN_IF_ERROR(DequantizeThenCopyValues(
            quantized_kv_slice, input_buffer, quantization_info));
      }
    }
    auto end = absl::Now();
    latency_stats_.decode_quantization_latency_us +=
        absl::ToInt64Microseconds(end - start);

    start = absl::Now();
    auto res = npu_auxiliary_context_.npu_auxiliary_compiled_model.Run(
        CacheUpdateSignatures::kDecodeCacheUpdate,
        cache_update_inference_context_.decode_input_buffers,
        cache_update_inference_context_.decode_output_buffers);
    RET_CHECK(res) << "Failed to run cache update model."
                   << res.Error().Message();
    end = absl::Now();
    latency_stats_.decode_cache_update_inference_latency_us +=
        absl::ToInt64Microseconds(end - start);
  }
  ++current_step_;
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

// static
absl::StatusOr<std::unique_ptr<LlmLiteRtNpuCompiledModelExecutor>>
LlmLiteRtNpuCompiledModelExecutor::Create(
    ModelQuantization model_quantization, const std::string& llm_model,
    const std::string& embedder_model, const std::string& npu_auxiliary_model,
    const std::optional<std::string>& dispatch_library_path) {
  if (model_quantization == ModelQuantization::kTransformerStackOnlyQuantized) {
    return CreateInternalGemmaOnlyQuantized(
        llm_model, embedder_model, npu_auxiliary_model, dispatch_library_path);
  } else {
    return CreateInternalAllQuantized(
        llm_model, embedder_model, npu_auxiliary_model, dispatch_library_path);
  }
};

// static
absl::StatusOr<std::unique_ptr<LlmLiteRtNpuCompiledModelExecutor>>
LlmLiteRtNpuCompiledModelExecutor::Create(
    const litert::lm::LlmExecutorSettings& executor_settings,
    const std::unique_ptr<litert::lm::ModelResources>& resources,
    const std::optional<std::string>& dispatch_library_path) {
  std::vector<::litert::Environment::Option> environment_options = {};
  if (dispatch_library_path.has_value()) {
    ABSL_LOG(INFO) << "Setting dispatch library path: "
                   << dispatch_library_path.value();
    environment_options.push_back(::litert::Environment::Option{
        ::litert::Environment::OptionTag::DispatchLibraryDir,
        absl::string_view(dispatch_library_path.value())});
  } else {
    ABSL_LOG(INFO) << "No dispatch library path provided.";
  }
  LITERT_ASSIGN_OR_RETURN(
      Environment env,
      ::litert::Environment::Create(absl::MakeConstSpan(environment_options)));
  ASSIGN_OR_RETURN(
      auto model_shared_ptr,
      resources->GetTFLiteModel(litert::lm::ModelType::kTfLitePrefillDecode));

  // If the model is fully AOT compiled for NPU, NPU accelerator is used
  // automatically.
  LITERT_ASSIGN_OR_RETURN(CompiledModel compiled_model_llm,
                          CompiledModel::Create(env, *model_shared_ptr.get(),
                                                kLiteRtHwAcceleratorCpu));

  // Allocate all input and output buffers for the LLM model that is meant
  // to run on the QC NPU chip. The buffers will be using 'FastRPC'. Later on,
  // the buffers will be duplicated into the output buffer maps of the embedder,
  // mask, and rope signatures.

  absl::flat_hash_map<absl::string_view, TensorBuffer>
      gemma_prefill_input_buffers;
  absl::flat_hash_map<absl::string_view, TensorBuffer>
      gemma_decode_input_buffers;
  absl::flat_hash_map<absl::string_view, TensorBuffer> input_kv_cache_buffers;
  absl::flat_hash_map<absl::string_view, TensorBuffer>
      prefill_output_kv_cache_slice_buffers;
  absl::flat_hash_map<absl::string_view, TensorBuffer>
      decode_output_kv_cache_slice_buffers;

  auto prefill_signature =
      model_shared_ptr->FindSignature(kPrefillSignatureRunner);
  constexpr absl::string_view kv_cache_k_root_name = "kv_cache_k_";
  constexpr absl::string_view kv_cache_v_root_name = "kv_cache_v_";
  constexpr absl::string_view kv_cache_slice_k_root_name = "kv_slice_k_";
  constexpr absl::string_view kv_cache_slice_v_root_name = "kv_slice_v_";

  for (auto input_name : prefill_signature->InputNames()) {
    if (absl::StartsWith(input_name, kv_cache_k_root_name) ||
        absl::StartsWith(input_name, kv_cache_v_root_name)) {
      LITERT_ASSIGN_OR_RETURN(input_kv_cache_buffers[input_name],
                              compiled_model_llm.CreateInputBuffer(
                                  kPrefillSignatureRunner, input_name));
    } else {
      LITERT_ASSIGN_OR_RETURN(gemma_prefill_input_buffers[input_name],
                              compiled_model_llm.CreateInputBuffer(
                                  kPrefillSignatureRunner, input_name));
    }
  }
  auto decode_signature =
      model_shared_ptr->FindSignature(kDecodeSignatureRunner);
  for (auto input_name : decode_signature->InputNames()) {
    if (absl::StartsWith(input_name, kv_cache_k_root_name) ||
        absl::StartsWith(input_name, kv_cache_v_root_name)) {
      continue;
    }
    LITERT_ASSIGN_OR_RETURN(gemma_decode_input_buffers[input_name],
                            compiled_model_llm.CreateInputBuffer(
                                kDecodeSignatureRunner, input_name));
  }
  for (auto output_name : prefill_signature->OutputNames()) {
    if (absl::StartsWith(output_name, kv_cache_slice_k_root_name) ||
        absl::StartsWith(output_name, kv_cache_slice_v_root_name)) {
      LITERT_ASSIGN_OR_RETURN(
          prefill_output_kv_cache_slice_buffers[output_name],
          compiled_model_llm.CreateOutputBuffer(kPrefillSignatureRunner,
                                                output_name));
    }
  }
  for (auto output_name : decode_signature->OutputNames()) {
    if (absl::StartsWith(output_name, kv_cache_slice_k_root_name) ||
        absl::StartsWith(output_name, kv_cache_slice_v_root_name)) {
      LITERT_ASSIGN_OR_RETURN(decode_output_kv_cache_slice_buffers[output_name],
                              compiled_model_llm.CreateOutputBuffer(
                                  kDecodeSignatureRunner, output_name));
    }
  }

  ASSIGN_OR_RETURN(
      auto llm_inference_context,
      CreateLlmInferenceContextWithBufferSharing(
          env, *model_shared_ptr.get(), compiled_model_llm,
          input_kv_cache_buffers, prefill_output_kv_cache_slice_buffers,
          decode_output_kv_cache_slice_buffers, gemma_prefill_input_buffers,
          gemma_decode_input_buffers));

  ASSIGN_OR_RETURN(
      auto embedder_lrt_model_shared_ptr,
      resources->GetTFLiteModel(litert::lm::ModelType::kTfLiteEmbedder));

  ASSIGN_OR_RETURN(
      auto embedder_context,
      CreateEmbedderContextWithBufferSharing(
          env, *embedder_lrt_model_shared_ptr.get(),
          gemma_prefill_input_buffers, gemma_decode_input_buffers));

  ASSIGN_OR_RETURN(
      auto npu_auxiliary_lrt_model_shared_ptr,
      resources->GetTFLiteModel(litert::lm::ModelType::kTfLiteAux));

  ASSIGN_OR_RETURN(auto npu_auxiliary_context,
                   CreateNpuAuxiliaryContext(
                       env, *npu_auxiliary_lrt_model_shared_ptr.get()));

  // Duplicate the embedder's buffers that are used to store the prefill and
  // decode input tokens, because they will need to be passed to the mask
  // inference context as well.
  LITERT_ASSIGN_OR_RETURN(
      ::litert::TensorBuffer prefill_input_tokens,
      embedder_context.inference_context
          .prefill_input_buffers[EmbedderSignatures::kEmbedderInput]
          .Duplicate());
  LITERT_ASSIGN_OR_RETURN(
      ::litert::TensorBuffer decode_input_tokens,
      embedder_context.inference_context
          .decode_input_buffers[EmbedderSignatures::kEmbedderInput]
          .Duplicate());

  ASSIGN_OR_RETURN(
      auto mask_context,
      CreateMaskContextWithBufferSharing(
          npu_auxiliary_context, std::move(prefill_input_tokens),
          std::move(decode_input_tokens), gemma_prefill_input_buffers,
          gemma_decode_input_buffers));

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
      CreateCacheUpdateInferenceContextWithoutBufferSharing(
          input_kv_cache_buffers, prefill_output_kv_cache_slice_buffers,
          decode_output_kv_cache_slice_buffers, std::move(prefill_input_pos),
          std::move(decode_input_pos)));

  RETURN_IF_ERROR(WarmupInference(
      compiled_model_llm, llm_inference_context,
      npu_auxiliary_context.npu_auxiliary_compiled_model, rope_context,
      mask_context, cache_update_inference_context));

  // For now we only support one prefill length in the model.
  SortedPrefillSignatureMap prefill_runner_set;
  prefill_runner_set[kPrefillSize] = kPrefillSignatureRunner;
  // TODO(b/423997573): Support litertlm file format for NPU. Then we can
  // remove the dummy model path.
  auto executor = absl::WrapUnique(new LlmLiteRtNpuCompiledModelExecutor(
      executor_settings, ModelQuantization::kAllQuantized,
      std::move(embedder_context), std::move(npu_auxiliary_context),
      std::move(mask_context), std::move(rope_context), std::move(env),
      std::move(*model_shared_ptr), std::move(compiled_model_llm),
      std::move(llm_inference_context),
      std::move(cache_update_inference_context),
      std::move(prefill_runner_set)));
  ABSL_LOG(INFO) << "Executor created.";
  return executor;
};

// Creates an executor from the given models.
absl::StatusOr<std::unique_ptr<LlmLiteRtNpuCompiledModelExecutor>>
LlmLiteRtNpuCompiledModelExecutor::CreateInternalGemmaOnlyQuantized(
    const std::string& llm_model, const std::string& embedder_model,
    const std::string& npu_auxiliary_model,
    const std::optional<std::string>& dispatch_library_path) {
  // TODO(b405424188): Remove the support for the 'gemma_only_quantized' variant
  // of the executor.  This was a temporary solution to allow us to use the
  // executor with the Gemma3 model, which was quantized, but the embedder and
  // auxiliary models were not quantized. Now that we have the full generality
  // of the executor, we should remove this path.
  return absl::Status(absl::StatusCode::kUnimplemented,
                      "This method is not implemented.");
}

}  // namespace odml::infra
