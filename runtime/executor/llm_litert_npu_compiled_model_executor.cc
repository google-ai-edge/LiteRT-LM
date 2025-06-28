#include "runtime/executor/llm_litert_npu_compiled_model_executor.h"

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
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/executor/litert_compiled_model_executor_utils.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/litert_status_util.h"
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

// Signature names for the embedder.
struct EmbedderSignatures {
  static constexpr absl::string_view kPrefillEmbedder = "prefill_embedder_128";
  static constexpr absl::string_view kDecodeEmbedder = "decode_embedder";
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

}  // namespace

absl::StatusOr<LlmLiteRtNpuCompiledModelExecutor::EmbedderContext>
LlmLiteRtNpuCompiledModelExecutor::CreateEmbedderContextWithBufferSharing(
    ::litert::Environment& env, const litert::Model& embedder_model,
    ::litert::TensorBuffer prefill_input_tokens,
    ::litert::TensorBuffer decode_input_tokens,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        gemma_prefill_input_buffers,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        gemma_decode_input_buffers) {
  LITERT_ASSIGN_OR_RETURN(
      CompiledModel embedder_compiled_model,
      CompiledModel::Create(env, embedder_model, kLiteRtHwAcceleratorCpu));

  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      prefill_input_buffers;
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      prefill_output_buffers;
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      decode_input_buffers;
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      decode_output_buffers;

  prefill_input_buffers[EmbedderSignatures::kEmbedderInput] =
      std::move(prefill_input_tokens);

  LITERT_ASSIGN_OR_RETURN(
      prefill_output_buffers[EmbedderSignatures::kEmbedderOutput],
      gemma_prefill_input_buffers[LlmSignatures::kInputEmbeddings].Duplicate());

  decode_input_buffers[EmbedderSignatures::kEmbedderInput] =
      std::move(decode_input_tokens);

  LITERT_ASSIGN_OR_RETURN(
      decode_output_buffers[EmbedderSignatures::kEmbedderOutput],
      gemma_decode_input_buffers[LlmSignatures::kInputEmbeddings].Duplicate());

  EmbedderContext embedder_context(
      std::move(embedder_compiled_model), std::move(prefill_input_buffers),
      std::move(prefill_output_buffers), std::move(decode_input_buffers),
      std::move(decode_output_buffers));
  return embedder_context;
}

absl::StatusOr<LlmLiteRtNpuCompiledModelExecutor::NpuAuxiliaryContext>
LlmLiteRtNpuCompiledModelExecutor::CreateNpuAuxiliaryContext(
    ::litert::Environment& env, const litert::Model& npu_auxiliary_model) {
  LITERT_ASSIGN_OR_RETURN(
      auto npu_auxiliary_compiled_model,
      CompiledModel::Create(env, npu_auxiliary_model, kLiteRtHwAcceleratorCpu));
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
  LITERT_ASSIGN_OR_RETURN(
      prefill_input_buffers[MaskSignatures::kMaskInputTokens],
      npu_auxiliary_context.npu_auxiliary_compiled_model.CreateInputBuffer(
          MaskSignatures::kPrefillMask, MaskSignatures::kMaskInputTokens));

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
  LITERT_ASSIGN_OR_RETURN(
      decode_input_buffers[MaskSignatures::kMaskInputTokens],
      npu_auxiliary_context.npu_auxiliary_compiled_model.CreateInputBuffer(
          MaskSignatures::kDecodeMask, MaskSignatures::kMaskInputTokens));

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
    LITERT_ASSIGN_OR_RETURN(
        decode_input_buffers[cache_k25],
        llm_compiled_model.CreateInputBuffer(kDecodeSignature, cache_k25));
    LITERT_ASSIGN_OR_RETURN(
        decode_input_buffers[cache_v25],
        llm_compiled_model.CreateInputBuffer(kDecodeSignature, cache_v25));
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
  int16_t max_value = std::numeric_limits<int16_t>::min();
  for (int i = 0; i < logits_buffer_int16.size(); ++i) {
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
    for (std::size_t i = 0, input_idx = 0; i < ids.size() - 1;
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

absl::Status LlmLiteRtNpuCompiledModelExecutor::Reset() {
  current_step_ = 0;
  next_input_token_id_ = -1;
  processed_tokens_.clear();
  sampled_ids_.clear();
  latency_stats_ = {};
  return absl::OkStatus();
}

// static
absl::StatusOr<std::unique_ptr<LlmLiteRtNpuCompiledModelExecutor>>
LlmLiteRtNpuCompiledModelExecutor::Create(
    const litert::lm::LlmExecutorSettings& executor_settings,
    litert::lm::ModelResources& resources,
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
      const litert::Model* llm_model,
      resources.GetTFLiteModel(litert::lm::ModelType::kTfLitePrefillDecode));
  // If the model is fully AOT compiled for NPU, NPU accelerator is used
  // automatically.
  LITERT_ASSIGN_OR_RETURN(
      CompiledModel llm_compiled_model,
      CompiledModel::Create(env, *llm_model, kLiteRtHwAcceleratorNpu));

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

  auto prefill_signature = llm_model->FindSignature(kPrefillSignature);
  constexpr absl::string_view kv_cache_k_root_name = "kv_cache_k_";
  constexpr absl::string_view kv_cache_v_root_name = "kv_cache_v_";
  constexpr absl::string_view kv_cache_slice_k_root_name = "kv_slice_k_";
  constexpr absl::string_view kv_cache_slice_v_root_name = "kv_slice_v_";

  for (auto input_name : prefill_signature->InputNames()) {
    if (absl::StartsWith(input_name, kv_cache_k_root_name) ||
        absl::StartsWith(input_name, kv_cache_v_root_name)) {
      LITERT_ASSIGN_OR_RETURN(
          input_kv_cache_buffers[input_name],
          llm_compiled_model.CreateInputBuffer(kPrefillSignature, input_name));
    } else {
      LITERT_ASSIGN_OR_RETURN(
          gemma_prefill_input_buffers[input_name],
          llm_compiled_model.CreateInputBuffer(kPrefillSignature, input_name));
    }
  }
  auto decode_signature = llm_model->FindSignature(kDecodeSignature);
  for (auto input_name : decode_signature->InputNames()) {
    if (absl::StartsWith(input_name, kv_cache_k_root_name) ||
        absl::StartsWith(input_name, kv_cache_v_root_name)) {
      continue;
    }
    LITERT_ASSIGN_OR_RETURN(
        gemma_decode_input_buffers[input_name],
        llm_compiled_model.CreateInputBuffer(kDecodeSignature, input_name));
  }
  for (auto output_name : prefill_signature->OutputNames()) {
    if (absl::StartsWith(output_name, kv_cache_slice_k_root_name) ||
        absl::StartsWith(output_name, kv_cache_slice_v_root_name)) {
      LITERT_ASSIGN_OR_RETURN(
          prefill_output_kv_cache_slice_buffers[output_name],
          llm_compiled_model.CreateOutputBuffer(kPrefillSignature,
                                                output_name));
    }
  }
  for (auto output_name : decode_signature->OutputNames()) {
    if (absl::StartsWith(output_name, kv_cache_slice_k_root_name) ||
        absl::StartsWith(output_name, kv_cache_slice_v_root_name)) {
      LITERT_ASSIGN_OR_RETURN(
          decode_output_kv_cache_slice_buffers[output_name],
          llm_compiled_model.CreateOutputBuffer(kDecodeSignature, output_name));
    }
  }

  ASSIGN_OR_RETURN(
      auto llm_inference_context,
      CreateLlmInferenceContextWithBufferSharing(
          env, llm_compiled_model, input_kv_cache_buffers,
          prefill_output_kv_cache_slice_buffers,
          decode_output_kv_cache_slice_buffers, gemma_prefill_input_buffers,
          gemma_decode_input_buffers));

  ASSIGN_OR_RETURN(auto npu_auxiliary_lrt_model,
                   resources.GetTFLiteModel(litert::lm::ModelType::kTfLiteAux));

  ASSIGN_OR_RETURN(auto npu_auxiliary_context,
                   CreateNpuAuxiliaryContext(env, *npu_auxiliary_lrt_model));

  ASSIGN_OR_RETURN(auto mask_context,
                   CreateMaskContextWithBufferSharing(
                       npu_auxiliary_context, gemma_prefill_input_buffers,
                       gemma_decode_input_buffers));

  // Duplicate the mask buffers that are used to store the prefill and
  // decode input tokens, because they will need to be passed to the embedder
  // inference context as well so that they can be shared.
  LITERT_ASSIGN_OR_RETURN(
      ::litert::TensorBuffer prefill_input_tokens,
      mask_context.prefill_input_buffers[MaskSignatures::kMaskInputTokens]
          .Duplicate());
  LITERT_ASSIGN_OR_RETURN(
      ::litert::TensorBuffer decode_input_tokens,
      mask_context.decode_input_buffers[MaskSignatures::kMaskInputTokens]
          .Duplicate());

  ASSIGN_OR_RETURN(
      auto embedder_lrt_model,
      resources.GetTFLiteModel(litert::lm::ModelType::kTfLiteEmbedder));
  ASSIGN_OR_RETURN(
      auto embedder_context,
      CreateEmbedderContextWithBufferSharing(
          env, *embedder_lrt_model, std::move(prefill_input_tokens),
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
  auto executor = absl::WrapUnique(new LlmLiteRtNpuCompiledModelExecutor(
      executor_settings, std::move(embedder_context),
      std::move(npu_auxiliary_context), std::move(mask_context),
      std::move(rope_context), std::move(env), std::move(llm_compiled_model),
      std::move(llm_inference_context),
      std::move(cache_update_inference_context),
      std::move(prefill_runner_set)));
  return executor;
};

}  // namespace litert::lm
