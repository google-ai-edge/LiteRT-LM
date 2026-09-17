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

#include "runtime/executor/llm_litert_compiled_model_executor.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"  // from @litert
#include "litert/cc/internal/litert_handle.h"  // from @litert
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_environment_options.h"  // from @litert
#include "litert/cc/litert_expected.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_model_types.h"  // from @litert
#include "litert/cc/litert_options.h"  // from @litert
#include "litert/cc/litert_profiler.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/cc/litert_tensor_buffer_types.h"  // from @litert
#if LITERT_HAS_WEBGPU_SUPPORT
#include "third_party/ml_drift/webgpu/spatial_tensor.h"
#include "third_party/ml_drift/webgpu/webgpu_headers.h"
#endif  // LITERT_HAS_WEBGPU_SUPPORT
#if defined(__APPLE__)
#include "litert/cc/options/litert_gpu_options.h"  // from @litert
#endif  // defined(__APPLE__)
#include "runtime/components/constrained_decoding/constrained_decoder.h"
#include "runtime/components/constrained_decoding/constraint.h"
#include "runtime/components/constrained_decoding/logit_mask.h"
#include "runtime/components/constrained_decoding/repetition_penalty_constraint.h"
#include "runtime/components/embedding_lookup/embedding_lookup_manager.h"
#include "runtime/components/model_resources.h"
#include "runtime/components/sampler_factory.h"
#include "runtime/executor/common_utils.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/litert/state.h"
#include "runtime/executor/litert_compiled_model_executor_utils.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/executor/llm_executor_processed_tokens.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/executor/llm_executor_settings_utils.h"
#include "runtime/executor/llm_litert_compiled_model_cache_utils.h"
#include "runtime/executor/llm_litert_mtp_drafter.h"
#include "runtime/executor/state_interface.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/log_tensor_buffer.h"
#include "runtime/util/lora_util.h"
#include "runtime/util/status_macros.h"  // IWYU pragma: keep
#include "runtime/util/tensor_buffer_util.h"
#include "tflite/types/half.h"  // from @litert

namespace litert::lm {
namespace {

using ::absl::Span;

// Names of the signature runners, used to get the signature runners from the
// interpreter.
constexpr absl::string_view kPrefillSignatureRunner = "prefill";
constexpr absl::string_view kDecodeSignatureRunner = "decode";
constexpr int kDynamicDimValue = -1;

void ClampMaxNumTokens(LlmExecutorSettings& executor_settings,
                       int model_max_num_tokens) {
  if (model_max_num_tokens > 0 &&
      model_max_num_tokens != std::numeric_limits<int>::max()) {
    if (executor_settings.GetMaxNumTokens() >
        static_cast<uint32_t>(model_max_num_tokens)) {
      ABSL_LOG(WARNING) << "Passed in max_num_tokens ("
                        << executor_settings.GetMaxNumTokens()
                        << ") is larger than what the static model supports ("
                        << model_max_num_tokens << "). Using model limit.";
      executor_settings.SetMaxNumTokens(model_max_num_tokens);
    } else if (executor_settings.GetMaxNumTokens() == 0) {
      executor_settings.SetMaxNumTokens(model_max_num_tokens);
    }
  }
}

absl::StatusOr<bool> HasDynamicDim(const CompiledModel& compiled_model,
                                   absl::string_view signature,
                                   absl::string_view tensor_name) {
  LITERT_ASSIGN_OR_RETURN(
      const RankedTensorType ranked_tensor_type,
      compiled_model.GetInputTensorType(signature, tensor_name));
  for (int dim : ranked_tensor_type.Layout().Dimensions()) {
    if (dim == kDynamicDimValue) {
      return true;
    }
  }
  return false;
}

absl::Status ResolveDynamicShape(CompiledModel& compiled_model,
                                 absl::string_view signature,
                                 absl::string_view tensor_name, int new_value) {
  LITERT_ASSIGN_OR_RETURN(
      const RankedTensorType ranked_tensor_type,
      compiled_model.GetInputTensorType(signature, tensor_name));
  auto dimensions = ranked_tensor_type.Layout().Dimensions();

  bool has_dynamic_dim = false;
  std::vector<int> new_shape;
  new_shape.reserve(dimensions.size());
  for (int i = 0; i < dimensions.size(); ++i) {
    if (dimensions[i] == kDynamicDimValue) {
      has_dynamic_dim = true;
      new_shape.push_back(new_value);
    } else {
      new_shape.push_back(dimensions[i]);
    }
  }

  if (has_dynamic_dim) {
    LITERT_RETURN_IF_ERROR(
        compiled_model.ResizeInputTensor(signature, tensor_name, new_shape));
  }

  return absl::OkStatus();
}

absl::Status CopyOrAssignTensorBuffer(const TensorBuffer& src,
                                      TensorBuffer& dst) {
  LITERT_ASSIGN_OR_RETURN(auto src_type, src.BufferType());
  LITERT_ASSIGN_OR_RETURN(auto dst_type, dst.BufferType());
  LITERT_ASSIGN_OR_RETURN(auto in_size, src.PackedSize());
  LITERT_ASSIGN_OR_RETURN(auto out_size, dst.PackedSize());
  if (src_type == dst_type && in_size == out_size) {
    auto src_tensor_type = src.TensorType();
    auto dst_tensor_type = dst.TensorType();
    if (src_tensor_type.HasValue() && dst_tensor_type.HasValue() &&
        *src_tensor_type == *dst_tensor_type) {
      LITERT_ASSIGN_OR_RETURN(dst, src.Duplicate());
      return absl::OkStatus();
    }
  }
  LITERT_ASSIGN_OR_RETURN(auto in_lock,
                          TensorBufferScopedLock::Create<const char>(
                              src, TensorBuffer::LockMode::kRead));
  LITERT_ASSIGN_OR_RETURN(auto out_lock,
                          TensorBufferScopedLock::Create<char>(
                              dst, TensorBuffer::LockMode::kWrite));
  size_t bytes_to_copy = std::min(in_size, out_size);
  std::memcpy(out_lock.second, in_lock.second, bytes_to_copy);
  return absl::OkStatus();
}

// Builds the output tensor type for the embedding lookup. The output tensor
// type is the same as the input tensor type, except the first dimension is the
// number of tokens.
absl::StatusOr<RankedTensorType> GetEmbeddingLookupOutputTensorType(
    int num_tokens, const RankedTensorType& output_element_type) {
  if (num_tokens == 1) {
    return output_element_type;
  } else if (num_tokens == 0) {
    return absl::InvalidArgumentError(
        "Number of tokens must be greater than 0.");
  }

  const auto& dims = output_element_type.Layout().Dimensions();
  if (dims.size() < 3) {
    return absl::InvalidArgumentError("Tensor type must have rank 3 or more.");
  }
  if (dims[0] != 1 || dims[1] != 1) {
    return absl::InvalidArgumentError(
        "Element type must have first two dimensions as 1.");
  }
  Dimensions embedding_dims(dims.begin(), dims.end());
  embedding_dims[1] = num_tokens;
  return RankedTensorType(output_element_type.ElementType(),
                          Layout(std::move(embedding_dims)));
}

// Returns a subspan of the given span for a chunk at the given index.
template <typename T>
absl::Span<const T> GetSpanForChunk(absl::Span<T> span, int num_chunks,
                                    int chunk_index) {
  size_t total_size = span.size();
  size_t chunk_size = total_size / num_chunks;
  return span.subspan(chunk_size * chunk_index, chunk_size);
}

absl::StatusOr<TensorBuffer> CreateFP16OutputBuffer(
    Environment& env, CompiledModel& compiled_model, size_t signature_index,
    absl::string_view output_name, size_t output_index) {
  LITERT_ASSIGN_OR_RETURN(
      std::vector<Layout> runtime_layouts,
      compiled_model.GetOutputTensorLayouts(signature_index,
                                            /*update_allocation=*/true));
  // Use runtime layout.
  Layout runtime_layout = runtime_layouts[output_index];
  LITERT_ASSIGN_OR_RETURN(
      auto requirements,
      compiled_model.GetOutputBufferRequirements(signature_index, output_name));
  LITERT_ASSIGN_OR_RETURN(auto strides, requirements.Strides());
  if (!strides.empty()) {
    auto dims = runtime_layout.Dimensions();
    runtime_layout = Layout(litert::Dimensions(dims.begin(), dims.end()),
                            litert::Strides(strides.begin(), strides.end()));
  }
  RankedTensorType new_tensor_type(litert::ElementType::Float16,
                                   std::move(runtime_layout));
  LITERT_ASSIGN_OR_RETURN(size_t size, requirements.BufferSize());
  LITERT_ASSIGN_OR_RETURN(auto buffer_types, requirements.SupportedTypes());
  if (buffer_types.empty()) {
    return absl::InternalError("No supported buffer types found.");
  }
  auto buffer_type = buffer_types[0];
  LITERT_ASSIGN_OR_RETURN(
      auto buffer, TensorBuffer::CreateManaged(
                       env, buffer_type, std::move(new_tensor_type), size));
  return buffer;
}

absl::StatusOr<TensorBuffer> CreateHostOutputBuffer(
    Environment& env, CompiledModel& compiled_model, size_t signature_index,
    size_t output_index, RankedTensorType tensor_type) {
  LITERT_ASSIGN_OR_RETURN(
      std::vector<Layout> runtime_layouts,
      compiled_model.GetOutputTensorLayouts(signature_index,
                                            /*update_allocation=*/true));
  Layout runtime_layout = runtime_layouts[output_index];
  LITERT_ASSIGN_OR_RETURN(auto requirements,
                          compiled_model.GetOutputBufferRequirements(
                              signature_index, output_index));
  LITERT_ASSIGN_OR_RETURN(auto strides, requirements.Strides());
  if (!strides.empty()) {
    auto dims = runtime_layout.Dimensions();
    runtime_layout = Layout(litert::Dimensions(dims.begin(), dims.end()),
                            litert::Strides(strides.begin(), strides.end()));
  }
  RankedTensorType host_tensor_type(tensor_type.ElementType(),
                                    std::move(runtime_layout));
  LITERT_ASSIGN_OR_RETURN(size_t size, requirements.BufferSize());
  LITERT_ASSIGN_OR_RETURN(auto buffer, TensorBuffer::CreateManaged(
                                           env, TensorBufferType::kHostMemory,
                                           std::move(host_tensor_type), size));
  return buffer;
}

absl::StatusOr<std::pair<const TensorBuffer*, int /*placeholder_token_id*/>>
GetEmbeddingsFromInputs(const ExecutorInputs& inputs) {
  if (auto vision_emb = inputs.GetVisionEmbeddingsPtr(); vision_emb.ok()) {
    return std::make_pair(*vision_emb, ExecutorVisionData::kSpecialToken);
  }
  if (auto proj_audio = inputs.GetProjectedAudioEmbeddingsPtr();
      proj_audio.ok()) {
    return std::make_pair(*proj_audio, ExecutorAudioData::kSpecialToken);
  }
  if (auto audio_emb = inputs.GetAudioEmbeddingsPtr(); audio_emb.ok()) {
    return std::make_pair(*audio_emb, ExecutorAudioData::kSpecialToken);
  }
  return absl::NotFoundError("No embeddings found in inputs.");
}

absl::StatusOr<std::pair<const TensorBuffer*, int /*placeholder_token_id*/>>
GetPerLayerEmbeddingsFromInputs(const ExecutorInputs& inputs) {
  if (auto vision_per_layer = inputs.GetVisionPerLayerEmbeddingsPtr();
      vision_per_layer.ok()) {
    return std::make_pair(*vision_per_layer, ExecutorVisionData::kSpecialToken);
  }
  if (auto audio_per_layer = inputs.GetAudioPerLayerEmbeddingsPtr();
      audio_per_layer.ok()) {
    return std::make_pair(*audio_per_layer, ExecutorAudioData::kSpecialToken);
  }
  return absl::NotFoundError("No per-layer embeddings found in inputs.");
}

}  // namespace

absl::Status LlmLiteRtCompiledModelExecutorBase::CreatePrefillInputBuffers(
    absl::string_view prefill_signature, int sequence_length,
    int context_length,
    absl::flat_hash_map<absl::string_view, TensorBuffer>&
        prefill_input_buffers) {
  auto dyn_shape_resolver = [&](absl::string_view tensor_name) -> absl::Status {
    return ResolveDynamicShape(*compiled_model_, prefill_signature, tensor_name,
                               sequence_length);
  };
  // Create input_token, positions and attn_mask buffers after determining
  // the prefill length.
  if (!signatures_.input_tokens.empty()) {
    ABSL_RETURN_IF_ERROR(dyn_shape_resolver(signatures_.input_tokens));
    LITERT_ASSIGN_OR_RETURN(auto tokens_buffer,
                            compiled_model_->CreateInputBuffer(
                                prefill_signature, signatures_.input_tokens));
    prefill_input_buffers[signatures_.input_tokens] = std::move(tokens_buffer);
  } else {
    // If input_tokens is empty, we must have input_embeddings.
    if (!signatures_.input_embeddings.has_value()) {
      return absl::FailedPreconditionError(
          "Input tokens or embeddings must be provided.");
    }
    ABSL_RETURN_IF_ERROR(
        dyn_shape_resolver(signatures_.input_embeddings.value()));
    LITERT_ASSIGN_OR_RETURN(
        auto embeddings_buffer,
        compiled_model_->CreateInputBuffer(
            prefill_signature, signatures_.input_embeddings.value()));
    prefill_input_buffers[signatures_.input_embeddings.value()] =
        std::move(embeddings_buffer);

    // We may have per layer embedding as well.
    if (signatures_.input_per_layer_embeddings.has_value()) {
      ABSL_RETURN_IF_ERROR(
          dyn_shape_resolver(signatures_.input_per_layer_embeddings.value()));
      LITERT_ASSIGN_OR_RETURN(
          auto per_layer_embeddings_buffer,
          compiled_model_->CreateInputBuffer(
              prefill_signature,
              signatures_.input_per_layer_embeddings.value()));
      prefill_input_buffers[signatures_.input_per_layer_embeddings.value()] =
          std::move(per_layer_embeddings_buffer);
    }
  }
  ABSL_RETURN_IF_ERROR(dyn_shape_resolver(signatures_.input_positions));
  LITERT_ASSIGN_OR_RETURN(auto positions_buffer,
                          compiled_model_->CreateInputBuffer(
                              prefill_signature, signatures_.input_positions));
  prefill_input_buffers[signatures_.input_positions] =
      std::move(positions_buffer);

  if (signatures_.input_attn_mask.has_value()) {
    ABSL_ASSIGN_OR_RETURN(bool is_attn_dyn,
                          HasDynamicDim(*compiled_model_, prefill_signature,
                                        signatures_.input_attn_mask.value()));
    if (is_attn_dyn) {
      std::vector<int> new_shape = {1, 1, sequence_length, context_length};
      LITERT_RETURN_IF_ERROR(compiled_model_->ResizeInputTensor(
          prefill_signature, signatures_.input_attn_mask.value(), new_shape));
    }

    LITERT_ASSIGN_OR_RETURN(
        auto attn_mask_buffer,
        compiled_model_->CreateInputBuffer(
            prefill_signature, signatures_.input_attn_mask.value()));
    prefill_input_buffers[signatures_.input_attn_mask.value()] =
        std::move(attn_mask_buffer);
    if (signatures_.input_attn_mask_local.has_value()) {
      auto attn_mask_local_buffer = compiled_model_->CreateInputBuffer(
          prefill_signature, signatures_.input_attn_mask_local.value());
      prefill_input_buffers[signatures_.input_attn_mask_local.value()] =
          std::move(*attn_mask_local_buffer);
    }
  }
  if (signatures_.input_int32_param.has_value()) {
    gpu_optimized_single_buffer_cache_ = true;
    LITERT_ASSIGN_OR_RETURN(
        auto param_tensor_buffer,
        compiled_model_->CreateInputBuffer(
            prefill_signature, signatures_.input_int32_param.value()));
    prefill_input_buffers[signatures_.input_int32_param.value()] =
        std::move(param_tensor_buffer);
  }
  return absl::OkStatus();
}

// Allocates and initializes non-KV-cache output buffers for a given prefill
// signature. KV-cache buffers are skipped as they are managed independently
// by LitertState.
absl::Status LlmLiteRtCompiledModelExecutorBase::CreatePrefillOutputBuffers(
    absl::string_view prefill_signature, int sequence_length,
    absl::flat_hash_map<absl::string_view, TensorBuffer>&
        prefill_output_buffers) {
  LITERT_ASSIGN_OR_RETURN(auto signature,
                          compiled_model_->FindSignature(prefill_signature));

  for (auto output_name : signature.OutputNames()) {
    // Skip KV-cache state tensors; their lifecycle and memory allocation are
    // owned and maintained entirely by LitertState.
    if (IsKVCacheTensor(output_name)) {
      continue;
    }
    LITERT_ASSIGN_OR_RETURN(
        auto output_buffer,
        compiled_model_->CreateOutputBuffer(prefill_signature, output_name));
    prefill_output_buffers[output_name] = std::move(output_buffer);
  }
  return absl::OkStatus();
}

absl::Status LlmLiteRtCompiledModelExecutorBase::FillInputBufferWithToken(
    const std::vector<std::shared_ptr<TokenData>>& unprocessed_token,
    TensorBuffer& input_buffer, bool is_per_layer_embedding) {
  if (unprocessed_token.empty()) {
    return absl::InvalidArgumentError("Unprocessed token is null.");
  }

  LITERT_ASSIGN_OR_RETURN(auto input_buffer_lock_and_addr,
                          TensorBufferScopedLock::Create(
                              input_buffer, TensorBuffer::LockMode::kWrite));
  LITERT_ASSIGN_OR_RETURN(size_t packed_size, input_buffer.PackedSize());
  size_t stride = packed_size / unprocessed_token.size();
  char* input_buffer_ptr =
      static_cast<char*>(input_buffer_lock_and_addr.second);
  for (const auto& token : unprocessed_token) {
    size_t size_to_fill = 0;
    if (token->embedding().empty()) {
      size_to_fill = sizeof(int32_t);
      RET_CHECK_GE(stride, size_to_fill);
      // If the token has no embedding, the input_buffer should takes token id.
      *reinterpret_cast<int32_t*>(input_buffer_ptr) = token->id();
    } else if (is_per_layer_embedding) {
      size_to_fill = token->per_layer_embedding().size() * sizeof(float);
      RET_CHECK_GE(stride, size_to_fill);
      memcpy(input_buffer_ptr, token->per_layer_embedding().data(),
             size_to_fill);
    } else {
      size_to_fill = token->embedding().size() * sizeof(float);
      RET_CHECK_GE(stride, size_to_fill);
      memcpy(input_buffer_ptr, token->embedding().data(), size_to_fill);
    }

    if (stride > size_to_fill) {
      memset(input_buffer_ptr + size_to_fill, 0, stride - size_to_fill);
    }
    input_buffer_ptr += stride;
  }
  return absl::OkStatus();
}

absl::Status LlmLiteRtCompiledModelExecutorBase::RollBackProcessedTokens() {
  int current_step = llm_context_->runtime_state().current_step;
  ProcessedTokens& processed_tokens =
      llm_context_->processed_context().processed_tokens();
  if (current_step == processed_tokens.TokenCount()) {
    return absl::OkStatus();
  }
  if (current_step == 0) {
    ABSL_RETURN_IF_ERROR(processed_tokens.RollBackToStep(0));
  } else {
    auto token_at_step = processed_tokens.GetTokenAtStep(current_step - 1);
    ABSL_RETURN_IF_ERROR(processed_tokens.RollBackToStep(current_step - 1));
    if (!token_at_step.empty()) {
      RET_CHECK_EQ(token_at_step.size(), 1);
      // Multimodal input cannot become a pending input token.
      if (token_at_step.at(0) > 0) {
        ABSL_RETURN_IF_ERROR(processed_tokens.AddPendingInputToken(
            {std::make_shared<TokenData>(token_at_step.at(0))}));
      } else {
        processed_tokens.AddProcessedTokens({token_at_step.at(0)});
      }
    }
  }

  // Reset sampler input handling as the step is rolled back.
  if (sampler_ != nullptr && sampler_->HandlesInput()) {
    ABSL_RETURN_IF_ERROR(SetSamplerInputHandling(/*reset=*/true));
  }

  return absl::OkStatus();
}

absl::Status LlmLiteRtCompiledModelExecutorBase::PrepareFirstPrefillAfterDecode(
    int token_index_to_reduce) {
  if (!llm_context_->runtime_state().ran_decode && !force_prepare_needed_) {
    return absl::OkStatus();
  }

  force_prepare_needed_ = false;
  llm_context_->runtime_state().ran_decode = false;

  int output_heads = 1;
  if (llm_context_->runtime_config().output_heads.has_value()) {
    output_heads = llm_context_->runtime_config().output_heads.value();
  }

  if (output_heads > 1) {
    LITERT_RETURN_IF_ERROR(llm_context_->processed_context()
                               .processed_tokens()
                               .ReduceTokenCandidates(token_index_to_reduce));
    RET_CHECK(state_ != nullptr);
    RET_CHECK(decode_state_ != nullptr);
    LITERT_RETURN_IF_ERROR(
        state_->SelectAndCopyFrom(*decode_state_, token_index_to_reduce));
  }

  // Reset sampler input handling if it handles input for next decode.
  if (sampler_ != nullptr && sampler_->HandlesInput()) {
    ABSL_RETURN_IF_ERROR(SetSamplerInputHandling(/*reset=*/true));
  }

  return absl::OkStatus();
}

absl::Status LlmLiteRtCompiledModelExecutorBase::PrefillInternal(
    absl::string_view prefill_signature,
    absl::flat_hash_map<absl::string_view, TensorBuffer>& prefill_input_buffers,
    absl::flat_hash_map<absl::string_view, TensorBuffer>&
        prefill_output_buffers,
    Span<const int> ids, bool async, const ExecutorInputs* inputs) {
  ABSL_RETURN_IF_ERROR(RollBackProcessedTokens());

  auto [internal_start_step_initial, pending_input_token_initial] =
      llm_context_->processed_context()
          .processed_tokens()
          .GetNextUnprocessedToken();

  {
    // Fill the input buffers with scoped locks.
    auto& prefill_input_pos =
        prefill_input_buffers[signatures_.input_positions];
    LITERT_ASSIGN_OR_RETURN(auto prefill_input_pos_size,
                            prefill_input_pos.PackedSize());
    LITERT_ASSIGN_OR_RETURN(
        auto prefill_input_pos_lock_and_addr,
        TensorBufferScopedLock::Create(prefill_input_pos,
                                       TensorBuffer::LockMode::kWrite));
    auto* prefill_input_pos_ptr =
        static_cast<int32_t*>(prefill_input_pos_lock_and_addr.second);

    memset(prefill_input_pos_ptr, 0, prefill_input_pos_size);
    if (signatures_.input_attn_mask.has_value()) {
      ABSL_RETURN_IF_ERROR(InitializeAttentionMask(
          prefill_input_buffers[signatures_.input_attn_mask.value()],
          use_fp16_precision_));
      if (signatures_.input_attn_mask_local.has_value()) {
        ABSL_RETURN_IF_ERROR(InitializeAttentionMask(
            prefill_input_buffers[signatures_.input_attn_mask_local.value()],
            use_fp16_precision_));
      }
    }
    // TODO(b/425396146): Add the unit tests for checking the prefill length.
    // We always hold one pending token in the input ids for the next
    // prefill or decode step.
    int prefill_length = ids.size() - 1;

    // Check if have a pending input token. Note that 'internal_start_step' is
    // always equal to the number of processed tokens plus 1.
    auto [internal_start_step, pending_input_token] =
        llm_context_->processed_context()
            .processed_tokens()
            .GetNextUnprocessedToken();
    RET_CHECK_LE(pending_input_token.size(), 1);
    const int start_step = internal_start_step;
    const bool has_pending_input_token = !pending_input_token.empty();
    const bool use_token_as_lookup = !signatures_.input_tokens.empty();
    const bool use_per_layer_embedding =
        signatures_.input_per_layer_embeddings.has_value();
    // If there is no pending input token and no input token to prefill, we can
    // skip the prefill by storing the token as a pending input token.
    bool skip_prefill = !has_pending_input_token && prefill_length == 0;
    if (!skip_prefill) {
      int input_idx = 0;
      if (has_pending_input_token) {
        if (use_token_as_lookup) {
          ABSL_RETURN_IF_ERROR(FillInputBufferWithToken(
              pending_input_token,
              prefill_input_buffers[signatures_.input_tokens]));
        } else {
          ABSL_RETURN_IF_ERROR(FillInputBufferWithToken(
              pending_input_token,
              prefill_input_buffers[signatures_.input_embeddings.value()]));
          if (use_per_layer_embedding) {
            ABSL_RETURN_IF_ERROR(FillInputBufferWithToken(
                pending_input_token,
                prefill_input_buffers[signatures_.input_per_layer_embeddings
                                          .value()],
                /*is_per_layer_embedding=*/true));
          }
        }
        prefill_input_pos_ptr[input_idx] = internal_start_step;
        ABSL_RETURN_IF_ERROR(llm_context_->processed_context()
                                 .processed_tokens()
                                 .MarkPendingInputTokenAsProcessed());
        llm_context_->runtime_state().current_step = internal_start_step + 1;

        ++prefill_input_pos_ptr;
        ++input_idx;
      }
      std::transform(prefill_input_pos_ptr,
                     prefill_input_pos_ptr + prefill_length,
                     prefill_input_pos_ptr, [&](int token) mutable {
                       return llm_context_->runtime_state().current_step++;
                     });
      std::vector<int> processed_input_tokens(ids.begin(),
                                              ids.begin() + prefill_length);
      llm_context_->processed_context().processed_tokens().AddProcessedTokens(
          processed_input_tokens);

      if (use_token_as_lookup) {
        auto& prefill_input_buffer =
            prefill_input_buffers[signatures_.input_tokens];
        LITERT_ASSIGN_OR_RETURN(
            auto prefill_input_lock_and_addr,
            TensorBufferScopedLock::Create(prefill_input_buffer,
                                           TensorBuffer::LockMode::kWrite));
        int32_t* prefill_input_ptr =
            static_cast<int32_t*>(prefill_input_lock_and_addr.second);
        LITERT_ASSIGN_OR_RETURN(auto prefill_input_size,
                                prefill_input_buffer.PackedSize());
        memcpy(prefill_input_ptr + input_idx, processed_input_tokens.data(),
               processed_input_tokens.size() * sizeof(int32_t));
        int pad_token_id = executor_settings_.GetPadTokenId();
        if (pad_token_id == -1) {
          pad_token_id = 0;
        }
        int active_tokens = input_idx + processed_input_tokens.size();
        int total_elements = prefill_input_size / sizeof(int32_t);
        ABSL_VLOG(1) << "Prefill padding with token " << pad_token_id
                     << " from index " << active_tokens << " to "
                     << total_elements;
        std::fill(prefill_input_ptr + active_tokens,
                  prefill_input_ptr + total_elements, pad_token_id);
      } else {
        // If not using token as lookup, we must have input_embeddings.
        TensorBuffer* prefill_input_embeddings_buffer =
            &(prefill_input_buffers[signatures_.input_embeddings.value()]);
        if (embedding_lookup_ != nullptr) {
          ABSL_RETURN_IF_ERROR(embedding_lookup_->LookupPrefill(
              processed_input_tokens, prefill_input_embeddings_buffer,
              /*offset=*/input_idx));
        } else if (inputs == nullptr) {
          return absl::InvalidArgumentError(
              "Prefill requires inputs when embedding_lookup_ is null.");
        } else {
          LITERT_ASSIGN_OR_RETURN(auto embeddings,
                                  GetEmbeddingsFromInputs(*inputs));
          if (embeddings.first == nullptr) {
            return absl::InvalidArgumentError(
                "Prefill requires embeddings in inputs when embedding_lookup_ "
                "is null.");
          }
          ABSL_RETURN_IF_ERROR(CopyOrAssignTensorBuffer(
              *embeddings.first, *prefill_input_embeddings_buffer));
        }

        // We may have per layer embedding as well.
        if (signatures_.input_per_layer_embeddings) {
          TensorBuffer* prefill_input_per_layer_embeddings_buffer =
              &(prefill_input_buffers[signatures_.input_per_layer_embeddings
                                          .value()]);
          if (per_layer_embedding_lookup_ != nullptr) {
            ABSL_RETURN_IF_ERROR(per_layer_embedding_lookup_->LookupPrefill(
                processed_input_tokens,
                prefill_input_per_layer_embeddings_buffer,
                /*offset=*/input_idx));
          } else if (inputs == nullptr) {
            return absl::InvalidArgumentError(
                "Prefill requires inputs when per_layer_embedding_lookup_ is "
                "null.");
          } else {
            LITERT_ASSIGN_OR_RETURN(auto per_layer,
                                    GetPerLayerEmbeddingsFromInputs(*inputs));
            if (per_layer.first == nullptr) {
              return absl::InvalidArgumentError(
                  "Prefill requires per_layer_embeddings in inputs when "
                  "per_layer_embedding_lookup_ is null.");
            }
            ABSL_RETURN_IF_ERROR(CopyOrAssignTensorBuffer(
                *per_layer.first, *prefill_input_per_layer_embeddings_buffer));
          }
        }
      }
      if (signatures_.input_attn_mask.has_value()) {
        const AttentionMaskParams attn_params =
            GetAttentionMaskParams(executor_metadata_);
        auto tokens_copy = llm_context_->processed_context()
                               .processed_tokens()
                               .GetCopyOfTokens();
        absl::Span<const int> token_ids_span =
            tokens_copy.empty() ? absl::Span<const int>()
                                : absl::MakeConstSpan(tokens_copy[0]);

        ABSL_RETURN_IF_ERROR(FillAttentionMask(
            prefill_input_buffers[signatures_.input_attn_mask.value()],
            start_step,
            /*steps=*/prefill_length + input_idx, attn_params.global_type,
            token_ids_span));
        if (signatures_.input_attn_mask_local.has_value()) {
          ABSL_RETURN_IF_ERROR(FillAttentionMask(
              prefill_input_buffers[signatures_.input_attn_mask_local.value()],
              start_step,
              /*steps=*/prefill_length + input_idx, attn_params.local_type,
              token_ids_span, attn_params.sliding_window_size,
              RingBufferAttentionMaskMode::kPrefill));
        }
      }
      if (gpu_optimized_single_buffer_cache_) {
        LITERT_RETURN_IF_ERROR(signatures_.input_int32_param.has_value());
        ABSL_RETURN_IF_ERROR(FillSingleBufferCacheParamTensor(
            prefill_input_buffers[signatures_.input_int32_param.value()],
            start_step, ids.size()));
      }
    }

    // Add the last token of the current input as a pending input token, to be
    // used in the next prefill or decode.
    auto last_input_token = std::make_shared<TokenData>(ids.back());
    if (!use_token_as_lookup) {
      if (embedding_lookup_ != nullptr) {
        // Look up the embeddings for the last token so they can be used in the
        // next prefill or decode. This has to be done now in the case of
        // multi-modal prefill so the embeddings are used in the correct order.
        ABSL_RETURN_IF_ERROR(embedding_lookup_->LookupPrefill(
            last_input_token->id(), last_input_token->mutable_embedding()));
        if (use_per_layer_embedding) {
          if (per_layer_embedding_lookup_ != nullptr) {
            ABSL_RETURN_IF_ERROR(per_layer_embedding_lookup_->LookupPrefill(
                last_input_token->id(),
                last_input_token->mutable_per_layer_embedding()));
          }
        }
      } else if (inputs == nullptr) {
        return absl::InvalidArgumentError(
            "Prefill requires inputs when embedding_lookup_ is null.");
      } else {
        LITERT_ASSIGN_OR_RETURN(auto embeddings,
                                GetEmbeddingsFromInputs(*inputs));
        if (embeddings.first == nullptr) {
          return absl::InvalidArgumentError(
              "Prefill requires embeddings in inputs when embedding_lookup_ "
              "is null.");
        }
        LITERT_ASSIGN_OR_RETURN(
            auto in_lock,
            TensorBufferScopedLock::Create<const float>(
                *embeddings.first, TensorBuffer::LockMode::kRead));
        LITERT_ASSIGN_OR_RETURN(auto in_size, embeddings.first->Size());
        size_t num_floats = in_size / sizeof(float);
        size_t num_tokens = ids.size();
        if (num_tokens > 0) {
          size_t hidden_dim = num_floats / num_tokens;
          const float* last_tok_emb =
              in_lock.second + (num_tokens - 1) * hidden_dim;
          last_input_token->mutable_embedding().assign(
              last_tok_emb, last_tok_emb + hidden_dim);
        }
      }
    }
    // Add the last input token to the pending input token list.
    ABSL_RETURN_IF_ERROR(
        llm_context_->processed_context()
            .processed_tokens()
            .AddPendingInputToken({std::move(last_input_token)}));
    ++llm_context_->runtime_state().current_step;
    if (skip_prefill) {
      return absl::OkStatus();
    }
  }
  return BindTensorsAndRunPrefill(prefill_signature, prefill_input_buffers,
                                  prefill_output_buffers, async);
}

absl::Status LlmLiteRtCompiledModelExecutorBase::BindTensorsAndRunPrefill(
    absl::string_view prefill_signature,
    absl::flat_hash_map<absl::string_view, TensorBuffer>& prefill_input_buffers,
    absl::flat_hash_map<absl::string_view, TensorBuffer>&
        prefill_output_buffers,
    bool async) {
  absl::flat_hash_map<absl::string_view, TensorBuffer> input_buffers;
  for (const auto& [input_name, input_buffer] : prefill_input_buffers) {
    LITERT_ASSIGN_OR_RETURN(auto input_buffer_dup, input_buffer.Duplicate());
    input_buffers[input_name] = std::move(input_buffer_dup);
  }

  LitertState* litert_state = nullptr;
  if (state_ != nullptr) {
    litert_state = dynamic_cast<LitertState*>(state_.get());
    RET_CHECK(litert_state != nullptr);
  }

  absl::flat_hash_map<absl::string_view, TensorBuffer> output_buffers;

  if (litert_state != nullptr) {
    LITERT_ASSIGN_OR_RETURN(
        auto state_buffers,
        litert_state->GetStateBuffers(*compiled_model_, prefill_signature));
    for (auto& [name, buffer] : state_buffers.input_buffers) {
      input_buffers[name] = std::move(buffer);
    }
    for (auto& [name, buffer] : state_buffers.output_buffers) {
      buffer.ClearEvent();
      output_buffers[name] = std::move(buffer);
    }
  }
  // Bind non-KV-cache output buffers to the final output buffers map.
  // Duplicate buffer handles and clear completion events so they are ready
  // for the upcoming graph execution.
  for (const auto& [output_name, output_buffer] : prefill_output_buffers) {
    LITERT_ASSIGN_OR_RETURN(auto output_buffer_dup, output_buffer.Duplicate());
    output_buffer_dup.ClearEvent();
    output_buffers[output_name] = std::move(output_buffer_dup);
  }

  if (pre_graph_run_callback_) {
    ABSL_ASSIGN_OR_RETURN(auto current_step, GetCurrentStep());
    pre_graph_run_callback_(prefill_signature, current_step, input_buffers);
  }

  litert::Options run_options = GetRunOptions();
  if (async) {
    LITERT_RETURN_IF_ERROR(compiled_model_->RunAsync(
        prefill_signature, input_buffers, output_buffers, async, &run_options));
  } else {
    LITERT_RETURN_IF_ERROR(compiled_model_->Run(
        prefill_signature, input_buffers, output_buffers, &run_options));
  }

  if (post_graph_run_callback_) {
    ABSL_ASSIGN_OR_RETURN(auto current_step, GetCurrentStep());
    post_graph_run_callback_(prefill_signature, current_step, output_buffers);
  }

  return absl::OkStatus();
}

absl::StatusOr<ProcessedTokens::StepAndToken>
LlmLiteRtCompiledModelExecutorBase::GetTokenToDecode(
    const ExecutorInputs& inputs) {
  ABSL_RETURN_IF_ERROR(RollBackProcessedTokens());

  if (inputs.GetTextDataPtr().ok()) {
    LITERT_ASSIGN_OR_RETURN(auto token_ids_buffer, inputs.GetTextTokenIdsPtr());
    auto input_tensor_size = token_ids_buffer->PackedSize();
    if (input_tensor_size && *input_tensor_size != 0) {
      int output_heads = 1;
      if (llm_context_->runtime_config().output_heads.has_value()) {
        output_heads = llm_context_->runtime_config().output_heads.value();
      }
      // Input token ids provided, so use it regardless of whether next input
      // token id is set.
      RET_CHECK_EQ(*input_tensor_size, output_heads * sizeof(int32_t));
      LITERT_ASSIGN_OR_RETURN(
          auto ids, ReferTensorBufferAsSpan<int32_t>(*token_ids_buffer));
      if (ids[0] >= 0) {
        // If the input token id is >= 0, it means the input token is provided
        // by the user. In this case, we should invalidate the pending input
        // token and add the input token as a pending input token.
        llm_context_->processed_context()
            .processed_tokens()
            .InvalidatePendingInputToken();
        std::vector<std::shared_ptr<TokenData>> token;
        token.reserve(output_heads);
        for (int i = 0; i < output_heads; ++i) {
          token.push_back(std::make_shared<TokenData>(ids[i]));
        }
        ABSL_RETURN_IF_ERROR(llm_context_->processed_context()
                                 .processed_tokens()
                                 .AddPendingInputToken(token));
      }
    }
  }

  // If multimodal embeddings (such as vision or audio embeddings) are provided
  // directly in the input, extract them and populate a pending input token to
  // be consumed during decoding.
  auto embeddings = GetEmbeddingsFromInputs(inputs);
  if (embeddings.ok() && embeddings->first != nullptr) {
    auto& emb_buffer = *embeddings->first;
    LITERT_ASSIGN_OR_RETURN(auto lock,
                            TensorBufferScopedLock::Create<const float>(
                                emb_buffer, TensorBuffer::LockMode::kRead));
    LITERT_ASSIGN_OR_RETURN(auto size, emb_buffer.Size());
    int num_floats = size / sizeof(float);
    std::vector<float> emb_vec(lock.second, lock.second + num_floats);
    auto token = std::make_shared<TokenData>(
        embeddings->second, std::move(emb_vec), std::vector<float>{});
    llm_context_->processed_context()
        .processed_tokens()
        .InvalidatePendingInputToken();
    ABSL_RETURN_IF_ERROR(llm_context_->processed_context()
                             .processed_tokens()
                             .AddPendingInputToken({token}));
  }

  // Here we must have a pending input token to decode that's either coming from
  // the previous prefill or decode, or we just added one from the inputs.
  for (const auto& token : llm_context_->processed_context()
                               .processed_tokens()
                               .GetNextUnprocessedToken()
                               .token) {
    // If the token has no embedding, we will look up the embedding for the
    // token here. This reduces the complexity for internal or external
    // sampling.
    if (signatures_.input_embeddings.has_value() &&
        token->mutable_embedding().empty()) {
      if (embedding_lookup_ == nullptr) {
        return absl::FailedPreconditionError(
            "Decode requires embedding_lookup_ when input_embeddings are used, "
            "but embedding_lookup_ is null.");
      }
      ABSL_RETURN_IF_ERROR(embedding_lookup_->LookupDecode(
          token->id(), token->mutable_embedding()));
      if (signatures_.input_per_layer_embeddings.has_value()) {
        if (per_layer_embedding_lookup_ == nullptr) {
          return absl::FailedPreconditionError(
              "Decode requires per_layer_embedding_lookup_ when required by "
              "signature, but per_layer_embedding_lookup_ is null.");
        }
        ABSL_RETURN_IF_ERROR(per_layer_embedding_lookup_->LookupDecode(
            token->id(), token->mutable_per_layer_embedding()));
      }
    }
  }
  return llm_context_->processed_context()
      .processed_tokens()
      .GetNextUnprocessedToken();
}

absl::Status
LlmLiteRtCompiledModelExecutorBase::ConsumePendingOrAddProcessedToken(
    const std::vector<std::shared_ptr<TokenData>>& token) {
  auto status = llm_context_->processed_context()
                    .processed_tokens()
                    .MarkPendingInputTokenAsProcessed();
  if (status.ok() || status.code() != absl::StatusCode::kNotFound) {
    return status;
  }

  // If the pending input token was not used, we should add the token to the
  // processed tokens.
  std::vector<int> processed_tokens;
  int output_heads = 1;
  if (llm_context_->runtime_config().output_heads.has_value()) {
    output_heads = llm_context_->runtime_config().output_heads.value();
  }
  processed_tokens.reserve(output_heads);
  for (const auto& t : token) {
    processed_tokens.push_back(t->id());
  }
  llm_context_->processed_context().processed_tokens().AddProcessedTokens(
      processed_tokens);
  ++llm_context_->runtime_state().current_step;
  return absl::OkStatus();
}

absl::Status LlmLiteRtCompiledModelExecutorBase::DecodeInternal(
    const std::vector<std::shared_ptr<TokenData>>& token,
    TensorBuffer& output_logits) {
  int step = llm_context_->runtime_state().current_step - 1;
  if (sampler_ && sampler_->HandlesInput()) {
    // The sampler has already been running decode for this step. Check if
    // output_logits is the one used last time, i.e. by
    // BindTensorsAndRunDecodeStatic().
    LITERT_RETURN_IF_ERROR(
        output_logits.Get() ==
        decode_output_buffers_[signatures_.output_logits].Get());
    return absl::OkStatus();
  }

  const bool use_token_as_lookup = !signatures_.input_tokens.empty();
  const bool use_per_layer_embedding =
      signatures_.input_per_layer_embeddings.has_value();

  // Fill the input buffers with scoped locks.
  if (use_token_as_lookup) {
    ABSL_RETURN_IF_ERROR(FillInputBufferWithToken(
        token, decode_input_buffers_[signatures_.input_tokens]));
  } else {
    if (!signatures_.input_embeddings.has_value()) {
      return absl::InvalidArgumentError(
          "Input tokens or embeddings must be provided.");
    }
    ABSL_RETURN_IF_ERROR(FillInputBufferWithToken(
        token, decode_input_buffers_[signatures_.input_embeddings.value()]));
    if (use_per_layer_embedding) {
      ABSL_RETURN_IF_ERROR(FillInputBufferWithToken(
          token,
          decode_input_buffers_[signatures_.input_per_layer_embeddings.value()],
          /*is_per_layer_embedding=*/true));
    }
  }

  {
    LITERT_ASSIGN_OR_RETURN(
        auto input_pos_type,
        decode_input_buffers_[signatures_.input_positions].TensorType());
    LITERT_ASSIGN_OR_RETURN(
        auto input_pos_lock_and_addr,
        TensorBufferScopedLock::Create(
            decode_input_buffers_[signatures_.input_positions],
            TensorBuffer::LockMode::kWrite));
    auto* input_pos_ptr = static_cast<int32_t*>(input_pos_lock_and_addr.second);
    if (input_pos_type.Layout().Dimensions()[0] == 1) {
      *input_pos_ptr = step;
    } else {
      int output_heads = 1;
      if (llm_context_->runtime_config().output_heads.has_value()) {
        output_heads = llm_context_->runtime_config().output_heads.value();
      }
      RET_CHECK_EQ(input_pos_type.Layout().Dimensions()[0], output_heads);
      LITERT_ASSIGN_OR_RETURN(
          auto input_pos_size,
          decode_input_buffers_[signatures_.input_positions].PackedSize());
      size_t offset = input_pos_size / output_heads / sizeof(int32_t);
      for (int i = 0; i < output_heads; ++i) {
        input_pos_ptr[i * offset] = step;
      }
    }
  }

  if (signatures_.input_attn_mask.has_value()) {
    ABSL_RETURN_IF_ERROR(InitializeAttentionMask(
        decode_input_buffers_[signatures_.input_attn_mask.value()],
        use_fp16_precision_));
    if (signatures_.input_attn_mask_local.has_value()) {
      ABSL_RETURN_IF_ERROR(InitializeAttentionMask(
          decode_input_buffers_[signatures_.input_attn_mask_local.value()],
          use_fp16_precision_));
    }
    const AttentionMaskParams attn_params =
        GetAttentionMaskParams(executor_metadata_);
    auto tokens_copy =
        llm_context_->processed_context().processed_tokens().GetCopyOfTokens();
    absl::Span<const int> token_ids_span =
        tokens_copy.empty() ? absl::Span<const int>()
                            : absl::MakeConstSpan(tokens_copy[0]);

    ABSL_RETURN_IF_ERROR(FillAttentionMask(
        decode_input_buffers_[signatures_.input_attn_mask.value()], step,
        /*steps=*/1, attn_params.global_type, token_ids_span));
    if (signatures_.input_attn_mask_local.has_value()) {
      ABSL_RETURN_IF_ERROR(FillAttentionMask(
          decode_input_buffers_[signatures_.input_attn_mask_local.value()],
          step,
          /*steps=*/1, attn_params.local_type, token_ids_span,
          attn_params.sliding_window_size,
          RingBufferAttentionMaskMode::kDecode));
    }
  }
  if (gpu_optimized_single_buffer_cache_) {
    LITERT_RETURN_IF_ERROR(signatures_.input_int32_param.has_value());
    ABSL_RETURN_IF_ERROR(FillSingleBufferCacheParamTensor(
        decode_input_buffers_[signatures_.input_int32_param.value()], step, 1));
  }

  return BindTensorsAndRunDecode(&output_logits);
}

absl::Status LlmLiteRtCompiledModelExecutorBase::BindTensorsAndRunDecode(
    TensorBuffer* output_logits) {
  absl::flat_hash_map<absl::string_view, TensorBuffer> decode_input_buffers;
  for (const auto& [input_name, input_buffer] : decode_input_buffers_) {
    LITERT_ASSIGN_OR_RETURN(auto input_buffer_dup, input_buffer.Duplicate());
    decode_input_buffers[input_name] = std::move(input_buffer_dup);
  }

  int output_heads = 1;
  if (llm_context_->runtime_config().output_heads.has_value()) {
    output_heads = llm_context_->runtime_config().output_heads.value();
  }
  StateInterface* active_state =
      (output_heads > 1) ? decode_state_.get() : state_.get();
  RET_CHECK(active_state != nullptr);

  auto* litert_state = dynamic_cast<LitertState*>(active_state);
  RET_CHECK(litert_state != nullptr);

  absl::flat_hash_map<absl::string_view, TensorBuffer> decode_output_buffers;
  for (const auto& [output_name, output_buffer] : decode_output_buffers_) {
    // LITERT_ASSIGN_OR_RETURN() causes a compilation error on windows.
    auto output_buffer_dup =
        output_logits && output_name == signatures_.output_logits
            ? output_logits->Duplicate()
            : output_buffer.Duplicate();
    RET_CHECK(output_buffer_dup) << "Failed to duplicate output buffer.";
    output_buffer_dup->ClearEvent();
    decode_output_buffers[output_name] = std::move(*output_buffer_dup);
  }

  LITERT_ASSIGN_OR_RETURN(
      auto state_buffers,
      litert_state->GetStateBuffers(*compiled_model_, kDecodeSignatureRunner));
  for (auto& [name, buffer] : state_buffers.input_buffers) {
    decode_input_buffers[name] = std::move(buffer);
  }
  for (auto& [name, buffer] : state_buffers.output_buffers) {
    buffer.ClearEvent();
    decode_output_buffers[name] = std::move(buffer);
  }

  if (pre_graph_run_callback_) {
    ABSL_ASSIGN_OR_RETURN(auto current_step, GetCurrentStep());
    pre_graph_run_callback_(kDecodeSignatureRunner, current_step,
                            decode_input_buffers);
  }

  litert::Options run_options = GetRunOptions();
  bool async = true;
  LITERT_RETURN_IF_ERROR(
      compiled_model_->RunAsync(kDecodeSignatureRunner, decode_input_buffers,
                                decode_output_buffers, async, &run_options));

  if (post_graph_run_callback_) {
    ABSL_ASSIGN_OR_RETURN(auto current_step, GetCurrentStep());
    post_graph_run_callback_(kDecodeSignatureRunner, current_step,
                             decode_output_buffers);
  }

  return absl::OkStatus();
}

int LlmLiteRtCompiledModelExecutorBase::BindTensorsAndRunDecodeStatic(
    void* arg) {
  auto self = static_cast<LlmLiteRtCompiledModelExecutorBase*>(arg);
  // Run decode with default output_logits.
  auto status = self->BindTensorsAndRunDecode(/*output_logits=*/nullptr);
  if (!status.ok()) {
    ABSL_LOG(ERROR) << "Failed to bind tensors and run decode: " << status;
  }
  return status.raw_code();
}

absl::Status LlmLiteRtCompiledModelExecutorBase::PrepareFirstDecode() {
  if (llm_context_->runtime_state().ran_decode && !force_prepare_needed_) {
    return absl::OkStatus();
  }
  force_prepare_needed_ = false;
  // Mark that we have run decode at least once.
  llm_context_->runtime_state().ran_decode = true;

  int output_heads = 1;
  if (llm_context_->runtime_config().output_heads.has_value()) {
    output_heads = llm_context_->runtime_config().output_heads.value();
  }

  if (output_heads <= 1) {
    return absl::OkStatus();
  }

  LITERT_RETURN_IF_ERROR(llm_context_->processed_context()
                             .processed_tokens()
                             .BroadcastTokenCandidates(output_heads));

  RET_CHECK(state_ != nullptr);
  RET_CHECK(decode_state_ != nullptr);
  LITERT_RETURN_IF_ERROR(decode_state_->BroadcastAndCopyFrom(*state_));

  return absl::OkStatus();
}

absl::Status LlmLiteRtCompiledModelExecutorBase::EnsureMtpDrafterLoaded() {
  if (mtp_drafter_ != nullptr) {
    return absl::OkStatus();
  }
  if (resources_ == nullptr) {
    return absl::FailedPreconditionError(
        "Cannot lazily load MTP drafter: ModelResources is not available.");
  }
  int output_heads = 1;
  if (llm_context_->runtime_config().output_heads.has_value()) {
    output_heads = llm_context_->runtime_config().output_heads.value();
  }
  RET_CHECK_EQ(output_heads, 1)
      << "Speculative decoding (MTP) only supports a single output head.";
  RET_CHECK_NE(embedding_lookup_, nullptr)
      << "Speculative decoding requires embedding lookup.";
  std::optional<std::reference_wrapper<EmbeddingLookupManager>> ple_manager_opt;
  if (per_layer_embedding_lookup_) {
    ple_manager_opt = std::ref(*per_layer_embedding_lookup_);
  }
  ABSL_ASSIGN_OR_RETURN(mtp_drafter_, LlmLiteRtMtpDrafter::Create(
                                          env_, *resources_, executor_settings_,
                                          *compiled_model_, *embedding_lookup_,
                                          ple_manager_opt, executor_metadata_));
  return absl::OkStatus();
}

absl::StatusOr<std::vector<std::vector<int>>>
LlmLiteRtCompiledModelExecutorBase::Decode() {
  return Decode(ExecutorDecodeParams());
}

absl::StatusOr<std::vector<std::vector<int>>>
LlmLiteRtCompiledModelExecutorBase::Decode(
    const ExecutorDecodeParams& decode_params) {

  bool enable_mtp_drafter = false;
  if (decode_params.GetEnableSpeculativeDecoding().has_value()) {
    enable_mtp_drafter = *decode_params.GetEnableSpeculativeDecoding();
    if (enable_mtp_drafter && mtp_drafter_ == nullptr) {
      ABSL_RETURN_IF_ERROR(EnsureMtpDrafterLoaded());
    }
  } else {
    enable_mtp_drafter = (mtp_drafter_ != nullptr);
  }

  std::vector<std::vector<int>> output_tokens_vector;
  if (!enable_mtp_drafter) {
    ABSL_ASSIGN_OR_RETURN(auto decoded_logits,
                          DecodeLogits(ExecutorInputs(), decode_params));
    std::optional<TensorBuffer> output_tokens;
    {
      LITERT_ASSIGN_OR_RETURN(auto decoded_logits_type,
                              decoded_logits.TensorType());
      auto dimensions = decoded_logits_type.Layout().Dimensions();
      // Shape of decoded_logits is [batch_size, Token_length, vocab_size].
      RET_CHECK_EQ(dimensions.size(), 3);
      LITERT_ASSIGN_OR_RETURN(
          output_tokens,
          CreateTensorBuffer<int>({dimensions[0], dimensions[1]}));
    }
    ABSL_RETURN_IF_ERROR(SampleLogits(decoded_logits, *output_tokens));
    LITERT_ASSIGN_OR_RETURN(output_tokens_vector,
                            CopyFromTensorBuffer2D<int>(*output_tokens));
  } else {
    // MTP keeps an internal state of the last time it was called and will
    // use those projected activations to kick off the next draft steps. As
    // such, we need to do a single decode step on the first decode call after
    // prefill and provide the projected activations to the MTP drafted only
    // once.
    StateInterface* active_state = state_.get();
    RET_CHECK(active_state != nullptr);

    ConstrainedDecoder* constrained_decoder =
        decode_params.GetConstrainedDecoder();
    Constraint* constraint = constrained_decoder != nullptr
                                 ? constrained_decoder->GetConstraint()
                                 : nullptr;

    bool last_run_is_decode = llm_context_->runtime_state().ran_decode;
    if (last_run_is_decode) {
      ABSL_ASSIGN_OR_RETURN(auto step_and_token,
                            GetTokenToDecode(ExecutorInputs()));
      ABSL_RETURN_IF_ERROR(
          ConsumePendingOrAddProcessedToken(step_and_token.token));
      // Output: [Batch, drafted and verified tokens]
      LITERT_ASSIGN_OR_RETURN(output_tokens_vector,
                              mtp_drafter_->Draft(step_and_token.step,
                                                  step_and_token.token[0]->id(),
                                                  /*activations=*/std::nullopt,
                                                  *active_state, constraint));
      RET_CHECK_EQ(output_tokens_vector.size(), 1);
      llm_context_->runtime_state().current_step +=
          output_tokens_vector[0].size();
    } else {
      int token_id = -1;
      {
        ABSL_ASSIGN_OR_RETURN(auto decoded_logits,
                              DecodeLogits(ExecutorInputs(), decode_params));
        LITERT_ASSIGN_OR_RETURN(auto decoded_logits_type,
                                decoded_logits.TensorType());
        auto dimensions = decoded_logits_type.Layout().Dimensions();
        // Shape of decoded_logits is [batch_size, Token_length, vocab_size].
        RET_CHECK_EQ(dimensions.size(), 3);
        LITERT_ASSIGN_OR_RETURN(
            auto output_tokens,
            CreateTensorBuffer<int>({dimensions[0], dimensions[1]}));
        ABSL_RETURN_IF_ERROR(SampleLogits(decoded_logits, output_tokens));
        LITERT_ASSIGN_OR_RETURN(output_tokens_vector,
                                CopyFromTensorBuffer2D<int>(output_tokens));
        RET_CHECK_EQ(output_tokens_vector.size(), 1);
        RET_CHECK_EQ(output_tokens_vector[0].size(), 1);
        token_id = output_tokens_vector[0][0];
      }

      RET_CHECK(decode_output_buffers_.contains("activations"));
      LITERT_ASSIGN_OR_RETURN(
          auto activations, decode_output_buffers_["activations"].Duplicate());
      // Note: Position remains the same as the prefill step. However,
      // current_step is incremented in DecodeLogits and as such needs to be
      // decremented.
      LITERT_ASSIGN_OR_RETURN(
          output_tokens_vector,
          mtp_drafter_->Draft(llm_context_->runtime_state().current_step - 1,
                              token_id, std::move(activations), *active_state,
                              constraint));
      llm_context_->runtime_state().current_step +=
          output_tokens_vector[0].size();
      output_tokens_vector[0].insert(output_tokens_vector[0].begin(), token_id);
    }
  }

  // Check for any invalid token ids and set them to zero, if any.
  bool has_invalid_output_token = false;
  for (int batch = 0; batch < output_tokens_vector.size(); ++batch) {
    for (int token_idx = 0; token_idx < output_tokens_vector[batch].size();
         ++token_idx) {
      if (output_tokens_vector[batch][token_idx] < 0) {
        has_invalid_output_token = true;
        output_tokens_vector[batch][token_idx] = 0;
      }
    }
  }
  if (has_invalid_output_token) {
    const auto& advanced_settings = executor_settings_.GetAdvancedSettings();
    if (advanced_settings.has_value() &&
        advanced_settings->error_on_invalid_sampled_token_id) {
      return absl::InternalError(
          "Invalid decode and sample result. The sampled token is negative. "
          "This is caused by invalid sampling or sampling from an invalid "
          "logits tensor, usually an overflowed logits tensor.");
    }
    ABSL_LOG(WARNING) << "Invalid decode and sample result. The sampled token "
                         "is casted to 0 to avoid crash.";
  }

  // Update context with the assumption that there is one output per head.
  // We must change this when doing drafter based decoding.
  std::vector<int> processed_tokens;
  std::vector<std::shared_ptr<TokenData>> pending_tokens;
  for (auto& output_head_tokens : output_tokens_vector) {
    for (int i = 0; i < output_head_tokens.size(); ++i) {
      // Last token is reserved as pending input token.
      if (i == output_head_tokens.size() - 1) {
        pending_tokens.push_back(
            std::make_shared<TokenData>(output_head_tokens[i]));
      } else {
        processed_tokens.push_back(output_head_tokens[i]);
      }
    }
  }
  if (!processed_tokens.empty()) {
    llm_context_->processed_context().processed_tokens().AddProcessedTokens(
        processed_tokens);
  }
  ABSL_RETURN_IF_ERROR(
      llm_context_->processed_context().processed_tokens().AddPendingInputToken(
          pending_tokens));

  return output_tokens_vector;
}

absl::Status LlmLiteRtCompiledModelExecutorBase::Decode(
    const ExecutorInputs& inputs, TensorBuffer& output_logits) {
  ABSL_RETURN_IF_ERROR(PrepareFirstDecode());
  ABSL_ASSIGN_OR_RETURN(auto step_and_token, GetTokenToDecode(inputs));
  ABSL_RETURN_IF_ERROR(DecodeInternal(step_and_token.token, output_logits));
  ABSL_RETURN_IF_ERROR(ConsumePendingOrAddProcessedToken(step_and_token.token));
  ++llm_context_->runtime_state().current_step;
  return absl::OkStatus();
}

#if LITERT_HAS_WEBGPU_SUPPORT
namespace {

inline uint32_t FloatToBits(float v) {
  uint32_t u;
  std::memcpy(&u, &v, sizeof(u));
  return u;
}

struct GpuLogitMaskParams {
  uint32_t tensor_vocab_size;
  uint32_t mask_vocab_size;
  uint32_t num_bitmap_pairs;
  uint32_t num_sparse_pairs;
};

struct GpuVec4U32 {
  uint32_t x;
  uint32_t y;
  uint32_t z;
  uint32_t w;
};

struct RawSparseOp {
  int token_id;
  // 0 = RepetitionPenalty, 1 = Sparse, 2 = SparseSignDependent
  uint32_t op_type;
  float param0;
  float param1;
};

bool ExtractMasksForWebGpu(const LogitMask* mask,
                           std::vector<const BitmapLogitMask*>& bitmap_masks,
                           std::vector<RawSparseOp>& raw_ops) {
  if (mask == nullptr) {
    return true;
  }
  switch (mask->GetType()) {
    case MaskType::kBitmap: {
      bitmap_masks.push_back(static_cast<const BitmapLogitMask*>(mask));
      return true;
    }
    case MaskType::kSparse: {
      const auto* sparse_mask = static_cast<const SparseLogitMask*>(mask);
      for (const auto& entry : sparse_mask->entries()) {
        raw_ops.push_back({
            .token_id = entry.token_id,
            .op_type = entry.sign_dependent_weight ? 2u : 1u,
            .param0 = entry.weight,
            .param1 = entry.bias,
        });
      }
      return true;
    }
    case MaskType::kComposite: {
      const auto* comp_mask = static_cast<const CompositeLogitMask*>(mask);
      std::vector<const LogitMask*> other_masks;
      std::vector<const LogitMask*> sparse_masks;
      for (const auto& child : comp_mask->masks()) {
        if (!child) continue;
        if (child->GetType() == MaskType::kBitmap) {
          bitmap_masks.push_back(
              static_cast<const BitmapLogitMask*>(child.get()));
        } else if (child->GetType() == MaskType::kSparse) {
          sparse_masks.push_back(child.get());
        } else {
          other_masks.push_back(child.get());
        }
      }
      for (const auto* child : other_masks) {
        if (!ExtractMasksForWebGpu(child, bitmap_masks, raw_ops)) {
          return false;
        }
      }
      for (const auto* child : sparse_masks) {
        if (!ExtractMasksForWebGpu(child, bitmap_masks, raw_ops)) {
          return false;
        }
      }
      return true;
    }
    case MaskType::kCustom: {
      const auto* rep_mask = dynamic_cast<const RepetitionPenaltyMask*>(mask);
      if (rep_mask == nullptr) {
        return false;
      }
      for (const auto& entry : rep_mask->entries()) {
        raw_ops.push_back({
            .token_id = entry.token_id,
            .op_type = 0u,
            .param0 = entry.repetition_penalty,
            .param1 = entry.bias,
        });
      }
      return true;
    }
  }
  return false;
}

constexpr char kWgslBitmapFp16Source[] = R"(
struct Params {
  tensor_vocab_size: u32,
  mask_vocab_size: u32,
  num_bitmap_pairs: u32,
  num_sparse_pairs: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> bitmap: array<u32>;
@group(0) @binding(2) var<storage, read_write> logits: array<u32>;

fn is_token_allowed(token_id: u32) -> bool {
  if (token_id >= params.mask_vocab_size) {
    return false;
  }
  let word_idx = token_id >> 5u;
  let bit_idx = token_id & 31u;
  return ((bitmap[word_idx] >> bit_idx) & 1u) != 0u;
}

@compute @workgroup_size(64)
fn main_bitmap(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  if (idx >= params.num_bitmap_pairs) {
    return;
  }
  let token0 = idx * 2u;
  let token1 = token0 + 1u;
  let raw = logits[idx];
  var out_raw = raw;
  if (!is_token_allowed(token0)) {
    out_raw = (out_raw & 0xffff0000u) | 0x0000fbffu;
  }
  if (!is_token_allowed(token1)) {
    out_raw = (out_raw & 0x0000ffffu) | 0xfbff0000u;
  }
  if (out_raw != raw) {
    logits[idx] = out_raw;
  }
}
)";

constexpr char kWgslSparseFp16Source[] = R"(
struct Params {
  tensor_vocab_size: u32,
  mask_vocab_size: u32,
  num_bitmap_pairs: u32,
  num_sparse_pairs: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> sparse_data: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> logits: array<u32>;

fn ieee754_f16_to_f32(h: u32) -> u32 {
  let sign = (h & 0x8000u) << 16u;
  let exp = (h >> 10u) & 0x1fu;
  let mant = h & 0x03ffu;
  if (exp == 0u) {
    if (mant == 0u) {
      return sign;
    }
    var m = mant;
    var e = 113u;
    while ((m & 0x0400u) == 0u) {
      m = m << 1u;
      e = e - 1u;
    }
    return sign | (e << 23u) | ((m & 0x03ffu) << 13u);
  }
  if (exp == 31u) {
    return sign | 0x7f800000u | (mant << 13u);
  }
  return sign | ((exp + 112u) << 23u) | (mant << 13u);
}

fn ieee754_f32_to_f16(x: u32) -> u32 {
  let sign = (x >> 16u) & 0x8000u;
  let abs_x = x & 0x7fffffffu;
  if (abs_x >= 0x47800000u) {
    if (abs_x > 0x7f800000u) {
      return sign | 0x7e00u | ((abs_x >> 13u) & 0x03ffu);
    }
    return sign | 0x7c00u;
  }
  if (abs_x < 0x33000000u) {
    return sign;
  }
  if (abs_x < 0x38800000u) {
    let exp32 = i32(abs_x >> 23u);
    let mant32 = (abs_x & 0x7fffffu) | 0x800000u;
    let shift = u32(126 - exp32);
    var mant16 = mant32 >> shift;
    let round_bit = (mant32 >> (shift - 1u)) & 1u;
    let sticky_mask = (1u << (shift - 1u)) - 1u;
    let sticky_bit = select(0u, 1u, (mant32 & sticky_mask) != 0u);
    if (round_bit == 1u && (sticky_bit == 1u || (mant16 & 1u) == 1u)) {
      mant16 = mant16 + 1u;
    }
    return sign | mant16;
  }
  let exp16 = ((abs_x >> 23u) - 112u) << 10u;
  let mant16 = (abs_x >> 13u) & 0x03ffu;
  let round_bit = (abs_x >> 12u) & 1u;
  let sticky_bit = select(0u, 1u, (abs_x & 0x0fffu) != 0u);
  var res = exp16 | mant16;
  if (round_bit == 1u && (sticky_bit == 1u || (res & 1u) == 1u)) {
    res = res + 1u;
  }
  return sign | res;
}

fn ieee754_div(ua: u32, ub: u32) -> u32 {
  let sign = (ua ^ ub) & 0x80000000u;
  let abs_a = ua & 0x7fffffffu;
  let abs_b = ub & 0x7fffffffu;
  if (abs_a == 0u) {
    return sign;
  }
  if (abs_b == 0u) {
    return sign | 0x7f800000u;
  }
  var exp_a = i32(abs_a >> 23u);
  var mant_a = abs_a & 0x7fffffu;
  if (exp_a == 0) {
    exp_a = 1;
    while ((mant_a & 0x800000u) == 0u) {
      mant_a = mant_a << 1u;
      exp_a = exp_a - 1;
    }
  } else {
    mant_a = mant_a | 0x800000u;
  }
  var exp_b = i32(abs_b >> 23u);
  var mant_b = abs_b & 0x7fffffu;
  if (exp_b == 0) {
    exp_b = 1;
    while ((mant_b & 0x800000u) == 0u) {
      mant_b = mant_b << 1u;
      exp_b = exp_b - 1;
    }
  } else {
    mant_b = mant_b | 0x800000u;
  }
  var exp_out = exp_a - exp_b + 127;
  var rem = mant_a;
  if (rem < mant_b) {
    rem = rem << 1u;
    exp_out = exp_out - 1;
  }
  var q = 0u;
  for (var bit = 0u; bit < 25u; bit = bit + 1u) {
    q = q << 1u;
    if (rem >= mant_b) {
      rem = rem - mant_b;
      q = q | 1u;
    }
    rem = rem << 1u;
  }
  var sig = q >> 1u;
  let round_bit = q & 1u;
  let sticky_bit = select(0u, 1u, rem != 0u);
  if (round_bit == 1u && (sticky_bit == 1u || (sig & 1u) == 1u)) {
    sig = sig + 1u;
    if (sig == 0x1000000u) {
      sig = 0x800000u;
      exp_out = exp_out + 1;
    }
  }
  if (exp_out >= 255) {
    return sign | 0x7f800000u;
  }
  if (exp_out <= 0) {
    let shift = u32(1 - exp_out);
    if (shift > 24u) {
      return sign;
    }
    let full_rem_sticky = select(0u, 1u, (rem != 0u) || (round_bit != 0u));
    let sub_sig = sig >> shift;
    let sub_round = (sig >> (shift - 1u)) & 1u;
    let sub_mask = (1u << (shift - 1u)) - 1u;
    let sub_sticky = select(0u, 1u, ((sig & sub_mask) != 0u) || (full_rem_sticky != 0u));
    var final_sub = sub_sig;
    if (sub_round == 1u && (sub_sticky == 1u || (final_sub & 1u) == 1u)) {
      final_sub = final_sub + 1u;
    }
    return sign | final_sub;
  }
  return sign | (u32(exp_out) << 23u) | (sig & 0x7fffffu);
}

fn ieee754_mul(ua: u32, ub: u32) -> u32 {
  let sign = (ua ^ ub) & 0x80000000u;
  let abs_a = ua & 0x7fffffffu;
  let abs_b = ub & 0x7fffffffu;
  if (abs_a == 0u || abs_b == 0u) {
    return sign;
  }
  var exp_a = i32(abs_a >> 23u);
  var mant_a = abs_a & 0x7fffffu;
  if (exp_a == 0) {
    exp_a = 1;
    while ((mant_a & 0x800000u) == 0u) {
      mant_a = mant_a << 1u;
      exp_a = exp_a - 1;
    }
  } else {
    mant_a = mant_a | 0x800000u;
  }
  var exp_b = i32(abs_b >> 23u);
  var mant_b = abs_b & 0x7fffffu;
  if (exp_b == 0) {
    exp_b = 1;
    while ((mant_b & 0x800000u) == 0u) {
      mant_b = mant_b << 1u;
      exp_b = exp_b - 1;
    }
  } else {
    mant_b = mant_b | 0x800000u;
  }
  let a_lo = mant_a & 0xfffu;
  let a_hi = mant_a >> 12u;
  let b_lo = mant_b & 0xfffu;
  let b_hi = mant_b >> 12u;
  let p0 = a_lo * b_lo;
  let p1 = a_hi * b_lo + a_lo * b_hi;
  let p2 = a_hi * b_hi;
  let lo_part = p0 + ((p1 & 0xfffu) << 12u);
  let exact_lo24 = lo_part & 0xffffffu;
  let exact_hi24 = p2 + (p1 >> 12u) + (lo_part >> 24u);

  var exp_out = exp_a + exp_b - 127;
  var sig = 0u;
  var round_bit = 0u;
  var sticky_bit = 0u;
  if ((exact_hi24 & 0x800000u) != 0u) {
    exp_out = exp_out + 1;
    sig = exact_hi24;
    round_bit = (exact_lo24 >> 23u) & 1u;
    sticky_bit = select(0u, 1u, (exact_lo24 & 0x7fffffu) != 0u);
  } else {
    sig = (exact_hi24 << 1u) | (exact_lo24 >> 23u);
    round_bit = (exact_lo24 >> 22u) & 1u;
    sticky_bit = select(0u, 1u, (exact_lo24 & 0x3fffffu) != 0u);
  }
  if (round_bit == 1u && (sticky_bit == 1u || (sig & 1u) == 1u)) {
    sig = sig + 1u;
    if (sig == 0x1000000u) {
      sig = 0x800000u;
      exp_out = exp_out + 1;
    }
  }
  if (exp_out >= 255) {
    return sign | 0x7f800000u;
  }
  if (exp_out <= 0) {
    let shift = u32(1 - exp_out);
    if (shift > 24u) {
      return sign;
    }
    let full_sticky = select(0u, 1u, (sticky_bit != 0u) || (round_bit != 0u));
    let sub_sig = sig >> shift;
    let sub_round = (sig >> (shift - 1u)) & 1u;
    let sub_mask = (1u << (shift - 1u)) - 1u;
    let sub_sticky = select(0u, 1u, ((sig & sub_mask) != 0u) || (full_sticky != 0u));
    var final_sub = sub_sig;
    if (sub_round == 1u && (sub_sticky == 1u || (final_sub & 1u) == 1u)) {
      final_sub = final_sub + 1u;
    }
    return sign | final_sub;
  }
  return sign | (u32(exp_out) << 23u) | (sig & 0x7fffffu);
}

fn ieee754_add(ua: u32, ub: u32) -> u32 {
  let abs_a = ua & 0x7fffffffu;
  let abs_b = ub & 0x7fffffffu;
  if (abs_a == 0u) {
    if (abs_b == 0u) {
      return ua & ub & 0x80000000u;
    }
    return ub;
  }
  if (abs_b == 0u) {
    return ua;
  }
  var a_u = ua;
  var b_u = ub;
  if (abs_b > abs_a) {
    a_u = ub;
    b_u = ua;
  }
  let sign_a = a_u & 0x80000000u;
  let sign_b = b_u & 0x80000000u;
  var exp_a = i32((a_u >> 23u) & 0xffu);
  var mant_a = a_u & 0x7fffffu;
  if (exp_a == 0) {
    exp_a = 1;
  } else {
    mant_a = mant_a | 0x800000u;
  }
  var exp_b = i32((b_u >> 23u) & 0xffu);
  var mant_b = b_u & 0x7fffffu;
  if (exp_b == 0) {
    exp_b = 1;
  } else {
    mant_b = mant_b | 0x800000u;
  }
  let ma = mant_a << 3u;
  var mb = mant_b << 3u;
  let diff = u32(exp_a - exp_b);
  if (diff > 0u) {
    if (diff >= 27u) {
      mb = select(0u, 1u, mb != 0u);
    } else {
      let sticky = select(0u, 1u, (mb & ((1u << diff) - 1u)) != 0u);
      mb = (mb >> diff) | sticky;
    }
  }
  var res_exp = exp_a;
  var res_sig = 0u;
  if (sign_a == sign_b) {
    res_sig = ma + mb;
    if ((res_sig & 0x8000000u) != 0u) {
      let sticky = res_sig & 1u;
      res_sig = (res_sig >> 1u) | sticky;
      res_exp = res_exp + 1;
    }
  } else {
    res_sig = ma - mb;
    if (res_sig == 0u) {
      return 0u;
    }
    while ((res_sig & 0x4000000u) == 0u && res_exp > 1) {
      res_sig = res_sig << 1u;
      res_exp = res_exp - 1;
    }
  }
  var sig24 = res_sig >> 3u;
  let round_bit = (res_sig >> 2u) & 1u;
  let sticky_bit = select(0u, 1u, (res_sig & 3u) != 0u);
  if (round_bit == 1u && (sticky_bit == 1u || (sig24 & 1u) == 1u)) {
    sig24 = sig24 + 1u;
    if (sig24 == 0x1000000u) {
      sig24 = 0x800000u;
      res_exp = res_exp + 1;
    }
  }
  if (res_exp >= 255) {
    return sign_a | 0x7f800000u;
  }
  if ((sig24 & 0x800000u) == 0u) {
    return sign_a | (sig24 & 0x7fffffu);
  }
  return sign_a | (u32(res_exp) << 23u) | (sig24 & 0x7fffffu);
}

@compute @workgroup_size(64)
fn main_sparse(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  if (idx >= params.num_sparse_pairs) {
    return;
  }
  let pair = sparse_data[idx];
  let pair_idx = pair.x;
  let op_start = pair.y;
  let op_count = pair.z;

  let raw = logits[pair_idx];
  var bits0 = raw & 0xffffu;
  var bits1 = (raw >> 16u) & 0xffffu;
  var val0_bits = ieee754_f16_to_f32(bits0);
  var val1_bits = ieee754_f16_to_f32(bits1);

  for (var i = 0u; i < op_count; i = i + 1u) {
    let op = sparse_data[op_start + i];
    let op_type = op.x;
    let token_id = op.y;
    let param0_bits = op.z;
    let param1_bits = op.w;
    let param0 = bitcast<f32>(param0_bits);

    let is_odd = (token_id & 1u) != 0u;
    let cur_bits = select(bits0, bits1, is_odd);
    var val_bits = select(val0_bits, val1_bits, is_odd);
    let val = bitcast<f32>(val_bits);

    if (cur_bits == 0xfbffu || cur_bits == 0xfc00u || val <= -65504.0) {
      continue;
    }

    if (op_type == 0u) {
      if (param0 > 1.0) {
        if (val > 0.0) {
          val_bits = ieee754_div(val_bits, param0_bits);
        } else {
          val_bits = ieee754_mul(val_bits, param0_bits);
        }
      }
      val_bits = ieee754_add(val_bits, param1_bits);
    } else if (op_type == 1u) {
      let prod_bits = ieee754_mul(val_bits, param0_bits);
      val_bits = ieee754_add(prod_bits, param1_bits);
    } else if (op_type == 2u) {
      if (val > 0.0) {
        let prod_bits = ieee754_mul(val_bits, param0_bits);
        val_bits = ieee754_add(prod_bits, param1_bits);
      } else {
        if (param0 != 0.0) {
          let div_bits = ieee754_div(val_bits, param0_bits);
          val_bits = ieee754_add(div_bits, param1_bits);
        } else {
          val_bits = ieee754_add(val_bits, param1_bits);
        }
      }
    }

    let new_bits = ieee754_f32_to_f16(val_bits);
    let rounded_val_bits = ieee754_f16_to_f32(new_bits);

    if (is_odd) {
      bits1 = new_bits;
      val1_bits = rounded_val_bits;
    } else {
      bits0 = new_bits;
      val0_bits = rounded_val_bits;
    }
  }

  logits[pair_idx] = bits0 | (bits1 << 16u);
}
)";

constexpr char kWgslBitmapFp32Source[] = R"(
struct Params {
  tensor_vocab_size: u32,
  mask_vocab_size: u32,
  num_bitmap_pairs: u32,
  num_sparse_pairs: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> bitmap: array<u32>;
@group(0) @binding(2) var<storage, read_write> logits_f32: array<f32>;

fn is_token_allowed(token_id: u32) -> bool {
  if (token_id >= params.mask_vocab_size) {
    return false;
  }
  let word_idx = token_id >> 5u;
  let bit_idx = token_id & 31u;
  return ((bitmap[word_idx] >> bit_idx) & 1u) != 0u;
}

@compute @workgroup_size(64)
fn main_bitmap_f32(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  if (idx >= params.num_bitmap_pairs) {
    return;
  }
  let token0 = idx * 2u;
  let token1 = token0 + 1u;
  let min_val = -0.7 * 3.402823466e+38;
  if (!is_token_allowed(token0)) {
    logits_f32[token0] = min_val;
  }
  if (!is_token_allowed(token1)) {
    logits_f32[token1] = min_val;
  }
}
)";

constexpr char kWgslSparseFp32Source[] = R"(
struct Params {
  tensor_vocab_size: u32,
  mask_vocab_size: u32,
  num_bitmap_pairs: u32,
  num_sparse_pairs: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> sparse_data: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> logits_f32: array<f32>;

fn ieee754_div(ua: u32, ub: u32) -> u32 {
  let sign = (ua ^ ub) & 0x80000000u;
  let abs_a = ua & 0x7fffffffu;
  let abs_b = ub & 0x7fffffffu;
  if (abs_a == 0u) {
    return sign;
  }
  if (abs_b == 0u) {
    return sign | 0x7f800000u;
  }
  var exp_a = i32(abs_a >> 23u);
  var mant_a = abs_a & 0x7fffffu;
  if (exp_a == 0) {
    exp_a = 1;
    while ((mant_a & 0x800000u) == 0u) {
      mant_a = mant_a << 1u;
      exp_a = exp_a - 1;
    }
  } else {
    mant_a = mant_a | 0x800000u;
  }
  var exp_b = i32(abs_b >> 23u);
  var mant_b = abs_b & 0x7fffffu;
  if (exp_b == 0) {
    exp_b = 1;
    while ((mant_b & 0x800000u) == 0u) {
      mant_b = mant_b << 1u;
      exp_b = exp_b - 1;
    }
  } else {
    mant_b = mant_b | 0x800000u;
  }
  var exp_out = exp_a - exp_b + 127;
  var rem = mant_a;
  if (rem < mant_b) {
    rem = rem << 1u;
    exp_out = exp_out - 1;
  }
  var q = 0u;
  for (var bit = 0u; bit < 25u; bit = bit + 1u) {
    q = q << 1u;
    if (rem >= mant_b) {
      rem = rem - mant_b;
      q = q | 1u;
    }
    rem = rem << 1u;
  }
  var sig = q >> 1u;
  let round_bit = q & 1u;
  let sticky_bit = select(0u, 1u, rem != 0u);
  if (round_bit == 1u && (sticky_bit == 1u || (sig & 1u) == 1u)) {
    sig = sig + 1u;
    if (sig == 0x1000000u) {
      sig = 0x800000u;
      exp_out = exp_out + 1;
    }
  }
  if (exp_out >= 255) {
    return sign | 0x7f800000u;
  }
  if (exp_out <= 0) {
    let shift = u32(1 - exp_out);
    if (shift > 24u) {
      return sign;
    }
    let full_rem_sticky = select(0u, 1u, (rem != 0u) || (round_bit != 0u));
    let sub_sig = sig >> shift;
    let sub_round = (sig >> (shift - 1u)) & 1u;
    let sub_mask = (1u << (shift - 1u)) - 1u;
    let sub_sticky = select(0u, 1u, ((sig & sub_mask) != 0u) || (full_rem_sticky != 0u));
    var final_sub = sub_sig;
    if (sub_round == 1u && (sub_sticky == 1u || (final_sub & 1u) == 1u)) {
      final_sub = final_sub + 1u;
    }
    return sign | final_sub;
  }
  return sign | (u32(exp_out) << 23u) | (sig & 0x7fffffu);
}

fn ieee754_mul(ua: u32, ub: u32) -> u32 {
  let sign = (ua ^ ub) & 0x80000000u;
  let abs_a = ua & 0x7fffffffu;
  let abs_b = ub & 0x7fffffffu;
  if (abs_a == 0u || abs_b == 0u) {
    return sign;
  }
  var exp_a = i32(abs_a >> 23u);
  var mant_a = abs_a & 0x7fffffu;
  if (exp_a == 0) {
    exp_a = 1;
    while ((mant_a & 0x800000u) == 0u) {
      mant_a = mant_a << 1u;
      exp_a = exp_a - 1;
    }
  } else {
    mant_a = mant_a | 0x800000u;
  }
  var exp_b = i32(abs_b >> 23u);
  var mant_b = abs_b & 0x7fffffu;
  if (exp_b == 0) {
    exp_b = 1;
    while ((mant_b & 0x800000u) == 0u) {
      mant_b = mant_b << 1u;
      exp_b = exp_b - 1;
    }
  } else {
    mant_b = mant_b | 0x800000u;
  }
  let a_lo = mant_a & 0xfffu;
  let a_hi = mant_a >> 12u;
  let b_lo = mant_b & 0xfffu;
  let b_hi = mant_b >> 12u;
  let p0 = a_lo * b_lo;
  let p1 = a_hi * b_lo + a_lo * b_hi;
  let p2 = a_hi * b_hi;
  let lo_part = p0 + ((p1 & 0xfffu) << 12u);
  let exact_lo24 = lo_part & 0xffffffu;
  let exact_hi24 = p2 + (p1 >> 12u) + (lo_part >> 24u);

  var exp_out = exp_a + exp_b - 127;
  var sig = 0u;
  var round_bit = 0u;
  var sticky_bit = 0u;
  if ((exact_hi24 & 0x800000u) != 0u) {
    exp_out = exp_out + 1;
    sig = exact_hi24;
    round_bit = (exact_lo24 >> 23u) & 1u;
    sticky_bit = select(0u, 1u, (exact_lo24 & 0x7fffffu) != 0u);
  } else {
    sig = (exact_hi24 << 1u) | (exact_lo24 >> 23u);
    round_bit = (exact_lo24 >> 22u) & 1u;
    sticky_bit = select(0u, 1u, (exact_lo24 & 0x3fffffu) != 0u);
  }
  if (round_bit == 1u && (sticky_bit == 1u || (sig & 1u) == 1u)) {
    sig = sig + 1u;
    if (sig == 0x1000000u) {
      sig = 0x800000u;
      exp_out = exp_out + 1;
    }
  }
  if (exp_out >= 255) {
    return sign | 0x7f800000u;
  }
  if (exp_out <= 0) {
    let shift = u32(1 - exp_out);
    if (shift > 24u) {
      return sign;
    }
    let full_sticky = select(0u, 1u, (sticky_bit != 0u) || (round_bit != 0u));
    let sub_sig = sig >> shift;
    let sub_round = (sig >> (shift - 1u)) & 1u;
    let sub_mask = (1u << (shift - 1u)) - 1u;
    let sub_sticky = select(0u, 1u, ((sig & sub_mask) != 0u) || (full_sticky != 0u));
    var final_sub = sub_sig;
    if (sub_round == 1u && (sub_sticky == 1u || (final_sub & 1u) == 1u)) {
      final_sub = final_sub + 1u;
    }
    return sign | final_sub;
  }
  return sign | (u32(exp_out) << 23u) | (sig & 0x7fffffu);
}

fn ieee754_add(ua: u32, ub: u32) -> u32 {
  let abs_a = ua & 0x7fffffffu;
  let abs_b = ub & 0x7fffffffu;
  if (abs_a == 0u) {
    if (abs_b == 0u) {
      return ua & ub & 0x80000000u;
    }
    return ub;
  }
  if (abs_b == 0u) {
    return ua;
  }
  var a_u = ua;
  var b_u = ub;
  if (abs_b > abs_a) {
    a_u = ub;
    b_u = ua;
  }
  let sign_a = a_u & 0x80000000u;
  let sign_b = b_u & 0x80000000u;
  var exp_a = i32((a_u >> 23u) & 0xffu);
  var mant_a = a_u & 0x7fffffu;
  if (exp_a == 0) {
    exp_a = 1;
  } else {
    mant_a = mant_a | 0x800000u;
  }
  var exp_b = i32((b_u >> 23u) & 0xffu);
  var mant_b = b_u & 0x7fffffu;
  if (exp_b == 0) {
    exp_b = 1;
  } else {
    mant_b = mant_b | 0x800000u;
  }
  let ma = mant_a << 3u;
  var mb = mant_b << 3u;
  let diff = u32(exp_a - exp_b);
  if (diff > 0u) {
    if (diff >= 27u) {
      mb = select(0u, 1u, mb != 0u);
    } else {
      let sticky = select(0u, 1u, (mb & ((1u << diff) - 1u)) != 0u);
      mb = (mb >> diff) | sticky;
    }
  }
  var res_exp = exp_a;
  var res_sig = 0u;
  if (sign_a == sign_b) {
    res_sig = ma + mb;
    if ((res_sig & 0x8000000u) != 0u) {
      let sticky = res_sig & 1u;
      res_sig = (res_sig >> 1u) | sticky;
      res_exp = res_exp + 1;
    }
  } else {
    res_sig = ma - mb;
    if (res_sig == 0u) {
      return 0u;
    }
    while ((res_sig & 0x4000000u) == 0u && res_exp > 1) {
      res_sig = res_sig << 1u;
      res_exp = res_exp - 1;
    }
  }
  var sig24 = res_sig >> 3u;
  let round_bit = (res_sig >> 2u) & 1u;
  let sticky_bit = select(0u, 1u, (res_sig & 3u) != 0u);
  if (round_bit == 1u && (sticky_bit == 1u || (sig24 & 1u) == 1u)) {
    sig24 = sig24 + 1u;
    if (sig24 == 0x1000000u) {
      sig24 = 0x800000u;
      res_exp = res_exp + 1;
    }
  }
  if (res_exp >= 255) {
    return sign_a | 0x7f800000u;
  }
  if ((sig24 & 0x800000u) == 0u) {
    return sign_a | (sig24 & 0x7fffffu);
  }
  return sign_a | (u32(res_exp) << 23u) | (sig24 & 0x7fffffu);
}

@compute @workgroup_size(64)
fn main_sparse_f32(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  if (idx >= params.num_sparse_pairs) {
    return;
  }
  let pair = sparse_data[idx];
  let op_start = pair.y;
  let op_count = pair.z;

  for (var i = 0u; i < op_count; i = i + 1u) {
    let op = sparse_data[op_start + i];
    let op_type = op.x;
    let token_id = op.y;
    let param0_bits = op.z;
    let param1_bits = op.w;
    let param0 = bitcast<f32>(param0_bits);

    var val_bits = bitcast<u32>(logits_f32[token_id]);
    let val = bitcast<f32>(val_bits);

    if (val_bits == 0xff800000u || val <= -0.7 * 3.402823466e+38) {
      continue;
    }

    if (op_type == 0u) {
      if (param0 > 1.0) {
        if (val > 0.0) {
          val_bits = ieee754_div(val_bits, param0_bits);
        } else {
          val_bits = ieee754_mul(val_bits, param0_bits);
        }
      }
      val_bits = ieee754_add(val_bits, param1_bits);
    } else if (op_type == 1u) {
      let prod_bits = ieee754_mul(val_bits, param0_bits);
      val_bits = ieee754_add(prod_bits, param1_bits);
    } else if (op_type == 2u) {
      if (val > 0.0) {
        let prod_bits = ieee754_mul(val_bits, param0_bits);
        val_bits = ieee754_add(prod_bits, param1_bits);
      } else {
        if (param0 != 0.0) {
          let div_bits = ieee754_div(val_bits, param0_bits);
          val_bits = ieee754_add(div_bits, param1_bits);
        } else {
          val_bits = ieee754_add(val_bits, param1_bits);
        }
      }
    }

    logits_f32[token_id] = bitcast<f32>(val_bits);
  }
}
)";

wgpu::ComputePipeline CompileComputePipeline(wgpu::Device& device,
                                             wgpu::PipelineLayout& layout,
                                             const char* wgsl_source,
                                             const char* entry_point) {
  wgpu::ShaderSourceWGSL wgsl_desc;
  wgsl_desc.code = wgsl_source;
  wgpu::ShaderModuleDescriptor sm_desc;
  sm_desc.nextInChain = &wgsl_desc;
  wgpu::ShaderModule module = device.CreateShaderModule(&sm_desc);
  if (!module) return nullptr;

  wgpu::ComputePipelineDescriptor cp_desc;
  cp_desc.layout = layout;
  cp_desc.compute.module = module;
  cp_desc.compute.entryPoint = entry_point;
  return device.CreateComputePipeline(&cp_desc);
}

}  // namespace
#endif  // LITERT_HAS_WEBGPU_SUPPORT

bool LlmLiteRtCompiledModelExecutorBase::TryProcessLogitsWebGpu(
    Environment& env, ConstrainedDecoder* constrained_decoder,
    TensorBuffer& output_logits) {
#if !LITERT_HAS_WEBGPU_SUPPORT
  return false;
#else
  if (constrained_decoder == nullptr ||
      constrained_decoder->GetBatchSize() != 1) {
    return false;
  }
  if (!output_logits.IsWebGpuMemory()) {
    return false;
  }
  auto tensor_type_or = output_logits.TensorType();
  if (!tensor_type_or.HasValue()) {
    return false;
  }
  const auto element_type = tensor_type_or->ElementType();
  if (element_type != ElementType::Float16 &&
      element_type != ElementType::Float32) {
    return false;
  }
  const auto dims = tensor_type_or->Layout().Dimensions();
  if (dims.size() != 3 || dims[0] != 1 || dims[1] != 1) {
    return false;
  }
  const int vocab_size = dims[2];
  if (vocab_size <= 0 || vocab_size > 262144 || (vocab_size % 2) != 0) {
    return false;
  }

  auto env_options_or = env.GetOptions();
  if (!env_options_or.HasValue()) {
    return false;
  }
  auto wgpu_device_res =
      env_options_or->GetOption(EnvironmentOptions::Tag::kWebGpuDevice);
  if (!wgpu_device_res.HasValue()) {
    return false;
  }
  const void* device_handle = nullptr;
  if (std::holds_alternative<int64_t>(*wgpu_device_res)) {
    device_handle =
        reinterpret_cast<const void*>(std::get<int64_t>(*wgpu_device_res));
  } else if (std::holds_alternative<const void*>(*wgpu_device_res)) {
    device_handle = std::get<const void*>(*wgpu_device_res);
  }
  if (device_handle == nullptr) {
    return false;
  }

  auto wgpu_buf_res = output_logits.GetWebGpuBuffer();
  if (!wgpu_buf_res.HasValue()) {
    return false;
  }
  const auto* spatial_tensor =
      reinterpret_cast<const ::ml_drift::webgpu::SpatialTensor*>(
          wgpu_buf_res.Value());
  if (spatial_tensor == nullptr) {
    return false;
  }
  wgpu::Buffer logits_buf = spatial_tensor->GetBufferHandle();
  if (!logits_buf) {
    return false;
  }
  const uint64_t logits_byte_size = spatial_tensor->GetMemorySizeInBytes();

  Constraint* constraint = constrained_decoder->GetConstraint();
  if (constraint == nullptr) {
    return true;
  }
  auto mask_or = constraint->ComputeMask(constrained_decoder->GetState(0));
  if (!mask_or.ok()) {
    return false;
  }
  const std::unique_ptr<LogitMask>& mask = *mask_or;
  if (mask == nullptr) {
    return true;
  }

  std::vector<const BitmapLogitMask*> bitmap_masks;
  std::vector<RawSparseOp> raw_ops;
  if (!ExtractMasksForWebGpu(mask.get(), bitmap_masks, raw_ops)) {
    return false;
  }

  const bool has_bitmap = !bitmap_masks.empty();
  std::vector<uint32_t> bitmap_u32;
  int mask_vocab_size = vocab_size;
  if (has_bitmap) {
    mask_vocab_size = bitmap_masks[0]->vocab_size();
    for (size_t i = 1; i < bitmap_masks.size(); ++i) {
      mask_vocab_size =
          std::min(mask_vocab_size, bitmap_masks[i]->vocab_size());
    }
    mask_vocab_size = std::max(0, std::min(mask_vocab_size, vocab_size));
    const int num_words64 =
        (mask_vocab_size > 0) ? (mask_vocab_size + 63) / 64 : 0;
    bitmap_u32.resize(std::max(1, num_words64 * 2), 0u);
    for (int w = 0; w < num_words64; ++w) {
      uint64_t fused_word = ~uint64_t{0};
      for (const auto* bm : bitmap_masks) {
        fused_word &= bm->words()[w];
        if (fused_word == 0) break;
      }
      if (w == num_words64 - 1 && (mask_vocab_size % 64) != 0) {
        const int valid_bits = mask_vocab_size % 64;
        fused_word &= (uint64_t{1} << valid_bits) - 1;
      }
      bitmap_u32[w * 2] = static_cast<uint32_t>(fused_word & 0xFFFFFFFFu);
      bitmap_u32[w * 2 + 1] = static_cast<uint32_t>(fused_word >> 32);
    }
  }

  struct IndexedOp {
    uint32_t pair_idx;
    uint32_t original_order;
    uint32_t op_type;
    uint32_t token_id;
    float param0;
    float param1;
  };
  std::vector<IndexedOp> indexed_ops;
  indexed_ops.reserve(raw_ops.size());
  for (size_t i = 0; i < raw_ops.size(); ++i) {
    const auto& op = raw_ops[i];
    if (op.token_id < 0 || op.token_id >= vocab_size) {
      continue;
    }
    indexed_ops.push_back({
        .pair_idx = static_cast<uint32_t>(op.token_id / 2),
        .original_order = static_cast<uint32_t>(i),
        .op_type = op.op_type,
        .token_id = static_cast<uint32_t>(op.token_id),
        .param0 = op.param0,
        .param1 = op.param1,
    });
  }

  const bool has_sparse = !indexed_ops.empty();
  if (!has_bitmap && !has_sparse) {
    return true;
  }

  std::vector<GpuVec4U32> sparse_buffer_data;
  uint32_t num_sparse_pairs = 0;
  if (has_sparse) {
    std::sort(indexed_ops.begin(), indexed_ops.end(),
              [](const IndexedOp& a, const IndexedOp& b) {
                if (a.pair_idx != b.pair_idx) return a.pair_idx < b.pair_idx;
                return a.original_order < b.original_order;
              });

    std::vector<GpuVec4U32> pairs;
    std::vector<GpuVec4U32> ops;
    ops.reserve(indexed_ops.size());

    size_t i = 0;
    while (i < indexed_ops.size()) {
      const uint32_t pair_idx = indexed_ops[i].pair_idx;
      const uint32_t op_start_offset = static_cast<uint32_t>(ops.size());
      uint32_t op_count = 0;
      while (i < indexed_ops.size() && indexed_ops[i].pair_idx == pair_idx) {
        ops.push_back({
            .x = indexed_ops[i].op_type,
            .y = indexed_ops[i].token_id,
            .z = FloatToBits(indexed_ops[i].param0),
            .w = FloatToBits(indexed_ops[i].param1),
        });
        ++op_count;
        ++i;
      }
      pairs.push_back({
          .x = pair_idx,
          .y = op_start_offset,
          .z = op_count,
          .w = 0u,
      });
    }

    num_sparse_pairs = static_cast<uint32_t>(pairs.size());
    for (auto& pair : pairs) {
      pair.y += num_sparse_pairs;
    }

    sparse_buffer_data.reserve(pairs.size() + ops.size());
    sparse_buffer_data.insert(sparse_buffer_data.end(), pairs.begin(),
                              pairs.end());
    sparse_buffer_data.insert(sparse_buffer_data.end(), ops.begin(), ops.end());
  }

  wgpu::Device device(reinterpret_cast<WGPUDevice>(  // NOLINT
      const_cast<void*>(device_handle)));
  wgpu::Queue queue = device.GetQueue();
  if (!queue) {
    return false;
  }

  if (!webgpu_logit_mask_state_ ||
      webgpu_logit_mask_state_->device_handle != device_handle) {
    auto new_state = std::make_unique<WebGpuLogitMaskState>();
    new_state->device_handle = device_handle;

    wgpu::BindGroupLayoutEntry bgl_entries[3];
    bgl_entries[0].binding = 0;
    bgl_entries[0].visibility = wgpu::ShaderStage::Compute;
    bgl_entries[0].buffer.type = wgpu::BufferBindingType::Uniform;
    bgl_entries[0].buffer.minBindingSize = sizeof(GpuLogitMaskParams);

    bgl_entries[1].binding = 1;
    bgl_entries[1].visibility = wgpu::ShaderStage::Compute;
    bgl_entries[1].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
    bgl_entries[1].buffer.minBindingSize = 0;

    bgl_entries[2].binding = 2;
    bgl_entries[2].visibility = wgpu::ShaderStage::Compute;
    bgl_entries[2].buffer.type = wgpu::BufferBindingType::Storage;
    bgl_entries[2].buffer.minBindingSize = 0;

    wgpu::BindGroupLayoutDescriptor bgl_desc;
    bgl_desc.entryCount = 3;
    bgl_desc.entries = bgl_entries;

    new_state->bitmap_bgl = device.CreateBindGroupLayout(&bgl_desc);
    new_state->sparse_bgl = device.CreateBindGroupLayout(&bgl_desc);
    if (!new_state->bitmap_bgl || !new_state->sparse_bgl) {
      return false;
    }

    wgpu::PipelineLayoutDescriptor bitmap_pl_desc;
    bitmap_pl_desc.bindGroupLayoutCount = 1;
    bitmap_pl_desc.bindGroupLayouts = &new_state->bitmap_bgl;
    wgpu::PipelineLayout bitmap_pl =
        device.CreatePipelineLayout(&bitmap_pl_desc);

    wgpu::PipelineLayoutDescriptor sparse_pl_desc;
    sparse_pl_desc.bindGroupLayoutCount = 1;
    sparse_pl_desc.bindGroupLayouts = &new_state->sparse_bgl;
    wgpu::PipelineLayout sparse_pl =
        device.CreatePipelineLayout(&sparse_pl_desc);

    new_state->bitmap_pipeline_f16 = CompileComputePipeline(
        device, bitmap_pl, kWgslBitmapFp16Source, "main_bitmap");
    new_state->sparse_pipeline_f16 = CompileComputePipeline(
        device, sparse_pl, kWgslSparseFp16Source, "main_sparse");
    new_state->bitmap_pipeline_f32 = CompileComputePipeline(
        device, bitmap_pl, kWgslBitmapFp32Source, "main_bitmap_f32");
    new_state->sparse_pipeline_f32 = CompileComputePipeline(
        device, sparse_pl, kWgslSparseFp32Source, "main_sparse_f32");

    if (!new_state->bitmap_pipeline_f16 || !new_state->sparse_pipeline_f16 ||
        !new_state->bitmap_pipeline_f32 || !new_state->sparse_pipeline_f32) {
      return false;
    }

    for (int i = 0; i < 2; ++i) {
      wgpu::BufferDescriptor params_desc;
      params_desc.size = sizeof(GpuLogitMaskParams);
      params_desc.usage =
          wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
      new_state->params_buf[i] = device.CreateBuffer(&params_desc);

      wgpu::BufferDescriptor bm_desc;
      bm_desc.size = 32768;
      bm_desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
      new_state->bitmap_buf[i] = device.CreateBuffer(&bm_desc);
      new_state->bitmap_buf_size[i] = bm_desc.size;

      wgpu::BufferDescriptor sp_desc;
      sp_desc.size = 4096;
      sp_desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
      new_state->sparse_buf[i] = device.CreateBuffer(&sp_desc);
      new_state->sparse_buf_size[i] = sp_desc.size;
    }

    webgpu_logit_mask_state_ = std::move(new_state);
  }

  auto* state = webgpu_logit_mask_state_.get();

  const int slot = state->buf_slot;
  state->buf_slot = (slot + 1) % 2;

  const uint32_t num_bitmap_pairs = static_cast<uint32_t>(vocab_size / 2);
  GpuLogitMaskParams params = {
      .tensor_vocab_size = static_cast<uint32_t>(vocab_size),
      .mask_vocab_size = static_cast<uint32_t>(mask_vocab_size),
      .num_bitmap_pairs = num_bitmap_pairs,
      .num_sparse_pairs = num_sparse_pairs,
  };
  queue.WriteBuffer(state->params_buf[slot], 0, &params, sizeof(params));

  if (has_bitmap) {
    const uint64_t required_bm_bytes =
        static_cast<uint64_t>(bitmap_u32.size() * sizeof(uint32_t));
    if (required_bm_bytes > state->bitmap_buf_size[slot]) {
      wgpu::BufferDescriptor bm_desc;
      bm_desc.size = (required_bm_bytes + 255) & ~uint64_t{255};
      bm_desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
      state->bitmap_buf[slot] = device.CreateBuffer(&bm_desc);
      state->bitmap_buf_size[slot] = bm_desc.size;
    }
    queue.WriteBuffer(state->bitmap_buf[slot], 0, bitmap_u32.data(),
                      required_bm_bytes);
  }

  if (has_sparse) {
    const uint64_t required_sp_bytes =
        static_cast<uint64_t>(sparse_buffer_data.size() * sizeof(GpuVec4U32));
    if (required_sp_bytes > state->sparse_buf_size[slot]) {
      wgpu::BufferDescriptor sp_desc;
      sp_desc.size = (required_sp_bytes + 255) & ~uint64_t{255};
      sp_desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
      state->sparse_buf[slot] = device.CreateBuffer(&sp_desc);
      state->sparse_buf_size[slot] = sp_desc.size;
    }
    queue.WriteBuffer(state->sparse_buf[slot], 0, sparse_buffer_data.data(),
                      required_sp_bytes);
  }

  const bool is_f16 = (element_type == ElementType::Float16);
  wgpu::CommandEncoder encoder = device.CreateCommandEncoder();

  if (has_bitmap) {
    wgpu::BindGroupEntry entries[3];
    entries[0].binding = 0;
    entries[0].buffer = state->params_buf[slot];
    entries[0].offset = 0;
    entries[0].size = sizeof(GpuLogitMaskParams);

    entries[1].binding = 1;
    entries[1].buffer = state->bitmap_buf[slot];
    entries[1].offset = 0;
    entries[1].size = state->bitmap_buf_size[slot];

    entries[2].binding = 2;
    entries[2].buffer = logits_buf;
    entries[2].offset = 0;
    entries[2].size = logits_byte_size;

    wgpu::BindGroupDescriptor bg_desc;
    bg_desc.layout = state->bitmap_bgl;
    bg_desc.entryCount = 3;
    bg_desc.entries = entries;
    wgpu::BindGroup bg = device.CreateBindGroup(&bg_desc);

    wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
    pass.SetPipeline(is_f16 ? state->bitmap_pipeline_f16
                            : state->bitmap_pipeline_f32);
    pass.SetBindGroup(0, bg);
    pass.DispatchWorkgroups((num_bitmap_pairs + 63) / 64);
    pass.End();
  }

  if (has_sparse) {
    wgpu::BindGroupEntry entries[3];
    entries[0].binding = 0;
    entries[0].buffer = state->params_buf[slot];
    entries[0].offset = 0;
    entries[0].size = sizeof(GpuLogitMaskParams);

    entries[1].binding = 1;
    entries[1].buffer = state->sparse_buf[slot];
    entries[1].offset = 0;
    entries[1].size = state->sparse_buf_size[slot];

    entries[2].binding = 2;
    entries[2].buffer = logits_buf;
    entries[2].offset = 0;
    entries[2].size = logits_byte_size;

    wgpu::BindGroupDescriptor bg_desc;
    bg_desc.layout = state->sparse_bgl;
    bg_desc.entryCount = 3;
    bg_desc.entries = entries;
    wgpu::BindGroup bg = device.CreateBindGroup(&bg_desc);

    wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
    pass.SetPipeline(is_f16 ? state->sparse_pipeline_f16
                            : state->sparse_pipeline_f32);
    pass.SetBindGroup(0, bg);
    pass.DispatchWorkgroups((num_sparse_pairs + 63) / 64);
    pass.End();
  }

  wgpu::CommandBuffer cb = encoder.Finish();
  queue.Submit(1, &cb);
  return true;
#endif  // LITERT_HAS_WEBGPU_SUPPORT
}

absl::StatusOr<TensorBuffer> LlmLiteRtCompiledModelExecutorBase::DecodeLogits(
    const ExecutorInputs& inputs) {
  return DecodeLogits(inputs, ExecutorDecodeParams());
}

absl::StatusOr<TensorBuffer> LlmLiteRtCompiledModelExecutorBase::DecodeLogits(
    const ExecutorInputs& inputs, const ExecutorDecodeParams& decode_params) {
  LITERT_ASSIGN_OR_RETURN(
      auto output_logits,
      decode_output_buffers_[signatures_.output_logits].Duplicate());

  bool last_run_is_decode = llm_context_->runtime_state().ran_decode;
  ABSL_RETURN_IF_ERROR(PrepareFirstDecode());
  ABSL_ASSIGN_OR_RETURN(auto step_and_token, GetTokenToDecode(inputs));
  ABSL_RETURN_IF_ERROR(DecodeInternal(step_and_token.token, output_logits));
  ABSL_RETURN_IF_ERROR(ConsumePendingOrAddProcessedToken(step_and_token.token));

  if (ConstrainedDecoder* constrained_decoder =
          decode_params.GetConstrainedDecoder();
      constrained_decoder != nullptr && !step_and_token.token.empty()) {
    std::vector<int> current_token_ids;
    current_token_ids.reserve(step_and_token.token.size());
    for (const auto& token : step_and_token.token) {
      current_token_ids.push_back(token->id());
    }
    // Update constraint state only with decode ids.
    if (last_run_is_decode) {
      ABSL_RETURN_IF_ERROR(
          constrained_decoder->UpdateState(absl::MakeSpan(current_token_ids)));
    }
    // Process logits based on the current constraint state.
    if (!TryProcessLogitsWebGpu(env_, constrained_decoder, output_logits)) {
      ABSL_RETURN_IF_ERROR(constrained_decoder->ProcessLogits(output_logits));
    }
  }

  ++llm_context_->runtime_state().current_step;

  const auto& advanced_settings = executor_settings_.GetAdvancedSettings();
  if (advanced_settings &&
      advanced_settings->num_logits_to_print_after_decode > 0) {
    LogTensor(output_logits,
              advanced_settings->num_logits_to_print_after_decode, "Logits")
        .IgnoreError();
  }
  return output_logits;
}

absl::StatusOr<std::string>
LlmLiteRtCompiledModelExecutorBase::GetPrefillSignatureKey() const {
  std::string prefill_signature_key;
  for (int i = 0; i < model_.GetNumSignatures(); ++i) {
    LITERT_ASSIGN_OR_RETURN(auto sig, model_.GetSignature(i));
    absl::string_view key = sig.Key();
    if (absl::StartsWith(key, kPrefillSignatureRunner)) {
      prefill_signature_key = key;
      break;
    }
  }
  RET_CHECK(!prefill_signature_key.empty());
  return prefill_signature_key;
}

absl::StatusOr<std::unique_ptr<StateInterface>>
LlmLiteRtCompiledModelExecutorBase::CloneState() const {
  int output_heads = 1;
  if (llm_context_->runtime_config().output_heads.has_value()) {
    output_heads = llm_context_->runtime_config().output_heads.value();
  }
  StateInterface* active_state =
      (output_heads > 1) ? decode_state_.get() : state_.get();
  if (active_state == nullptr) {
    return nullptr;
  }
  return active_state->DeepCopy();
}

absl::Status LlmLiteRtCompiledModelExecutorBase::RestoreState(
    std::unique_ptr<StateInterface> state) {
  if (state == nullptr) {
    return absl::OkStatus();
  }
  if (state->GetBatchSize() > 1) {
    decode_state_ = std::move(state);
  } else {
    state_ = std::move(state);
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<LlmContext>>
LlmLiteRtCompiledModelExecutorBase::CreateNewContext(
    std::optional<uint32_t> lora_id, RuntimeConfig runtime_config) const {
  std::unique_ptr<ProcessedContext> processed_context =
      std::make_unique<LlmProcessedContext>(lora_id, nullptr);

  auto runtime_state = std::make_unique<RuntimeState>();
  if (runtime_config.sampler_params.has_value()) {
    runtime_state->rand_gen = std::make_shared<std::default_random_engine>(
        runtime_config.sampler_params->seed());
  } else {
    runtime_state->rand_gen = std::make_shared<std::default_random_engine>(0);
  }

  return std::make_unique<LlmContext>(
      std::move(processed_context),
      std::make_unique<RuntimeConfig>(std::move(runtime_config)),
      std::move(runtime_state));
}

absl::StatusOr<std::unique_ptr<LlmContext>>
LlmLiteRtCompiledModelExecutorBase::CloneContext() const {
  std::optional<uint32_t> lora_id;
  ABSL_ASSIGN_OR_RETURN(auto state, CloneState());
  ProcessedTokens new_processed_tokens =
      llm_context_->processed_context().processed_tokens();
  auto new_processed_context = std::make_unique<LlmProcessedContext>(
      std::move(lora_id), std::move(state), std::move(new_processed_tokens));
  auto new_runtime_config =
      std::make_unique<RuntimeConfig>(llm_context_->runtime_config());
  auto new_runtime_state =
      std::make_unique<RuntimeState>(llm_context_->runtime_state());
  return std::make_unique<LlmContext>(std::move(new_processed_context),
                                      std::move(new_runtime_config),
                                      std::move(new_runtime_state));
}

absl::Status LlmLiteRtCompiledModelExecutorBase::RestoreContext(
    std::unique_ptr<LlmContext> context_data) {
  llm_context_ = std::move(context_data);

  // We can keep our kv cache buffers if this is the first step. This lets us
  // restore from LlmContexts at step 0 with an empty kv cache.
  if (llm_context_->runtime_state().current_step > 0) {
    auto restored_state = std::move(
        static_cast<LlmProcessedContext&>(llm_context_->processed_context())
            .state());
    ABSL_RETURN_IF_ERROR(RestoreState(std::move(restored_state)));
  }

  force_prepare_needed_ = true;

  return absl::OkStatus();
}

absl::Status LlmLiteRtCompiledModelExecutorBase::InitializeSampler(
    std::optional<ActivationDataType> logits_data_type) {
  if (sampler_ != nullptr) {
    return absl::OkStatus();
  }

  // Use the provided activation data type if available, otherwise fallback to
  // the member variable.
  auto data_type = logits_data_type.value_or(logits_data_type_);

  ABSL_ASSIGN_OR_RETURN(auto vocab_size, GetVocabSize());
  ABSL_ASSIGN_OR_RETURN(auto sampler_backend,
                        GetSamplerBackend(executor_settings_));
  int output_heads = 1;
  if (llm_context_->runtime_config().output_heads.has_value()) {
    output_heads = llm_context_->runtime_config().output_heads.value();
  }
  proto::SamplerParameters sampler_params;
  if (llm_context_->runtime_config().sampler_params.has_value()) {
    sampler_params = llm_context_->runtime_config().sampler_params.value();
  }
  if (sampler_params.type() == proto::SamplerParameters::TYPE_UNSPECIFIED) {
    sampler_params.set_type(proto::SamplerParameters::TOP_P);
    sampler_params.set_k(1);
    sampler_params.set_p(0.0f);
    sampler_params.set_temperature(1.0f);
    sampler_params.set_seed(0);
  }

  gpu_sampler_max_top_k_ = sampler_params.k();

  ABSL_ASSIGN_OR_RETURN(
      sampler_,
      CreateSampler(sampler_backend, output_heads, std::move(sampler_params),
                    env_, /*sequence_size=*/1, vocab_size, data_type));

  // Disable GPU token copy for models that run embedding on the GPU.
  const bool runs_embedding_on_gpu = (embedding_lookup_ == nullptr);

  // If the sampler can handle input, prepare the input tensors for it.
  bool sampler_handles_input = true;
  if (executor_settings_.GetAdvancedSettings().has_value()) {
    sampler_handles_input =
        executor_settings_.GetAdvancedSettings()->sampler_handles_input;
  }
  sampler_handles_input_ =
      sampler_handles_input && sampler_->CanHandleInput() &&
      runs_embedding_on_gpu && !signatures_.input_tokens.empty() &&
      !signatures_.input_attn_mask_local.has_value();
  if (sampler_handles_input_) {
    ABSL_LOG(INFO) << "Sampler will handle decode input tensors.";
    if (!decode_prev_input_pos_) {
      LITERT_ASSIGN_OR_RETURN(
          decode_prev_input_pos_,
          compiled_model_->CreateInputBuffer(kDecodeSignatureRunner,
                                             signatures_.input_positions));
    }
    if (!decode_prev_mask_ && signatures_.input_attn_mask.has_value()) {
      LITERT_ASSIGN_OR_RETURN(
          decode_prev_mask_,
          compiled_model_->CreateInputBuffer(kDecodeSignatureRunner,
                                             *signatures_.input_attn_mask));
    }
    if (!decode_prev_param_ && signatures_.input_int32_param.has_value()) {
      LITERT_ASSIGN_OR_RETURN(
          decode_prev_param_,
          compiled_model_->CreateInputBuffer(kDecodeSignatureRunner,
                                             *signatures_.input_int32_param));
    }
    // Set, then reset the input handling to get the underlying model ready, but
    // not to bind the input tensors.
    ABSL_RETURN_IF_ERROR(SetSamplerInputHandling(/*reset=*/false));
    ABSL_RETURN_IF_ERROR(SetSamplerInputHandling(/*reset=*/true));
  }

  return absl::OkStatus();
}

absl::Status LlmLiteRtCompiledModelExecutorBase::SwapSamplerInputTensors() {
  // Move the input_pos and mask to previous ones.
  std::swap(decode_prev_input_pos_,
            decode_input_buffers_[signatures_.input_positions]);
  if (signatures_.input_attn_mask.has_value()) {
    std::swap(decode_prev_mask_,
              decode_input_buffers_[*signatures_.input_attn_mask]);
  }
  if (signatures_.input_int32_param.has_value()) {
    std::swap(decode_prev_param_,
              decode_input_buffers_[*signatures_.input_int32_param]);
  }
  return SetSamplerInputHandling(/*reset=*/false);
}

absl::Status LlmLiteRtCompiledModelExecutorBase::SetSamplerInputHandling(
    bool reset) {
  if (reset) {
    return sampler_->SetInferenceFuncAndInputTensors(nullptr, nullptr, nullptr,
                                                     nullptr, nullptr, nullptr,
                                                     nullptr, nullptr, nullptr);
  }

  bool has_input_attn_mask = signatures_.input_attn_mask.has_value();
  bool has_input_int32_param = signatures_.input_int32_param.has_value();
  return sampler_->SetInferenceFuncAndInputTensors(
      BindTensorsAndRunDecodeStatic, this,
      &decode_input_buffers_[signatures_.input_tokens], &decode_prev_input_pos_,
      &decode_input_buffers_[signatures_.input_positions],
      has_input_attn_mask ? &decode_prev_mask_ : nullptr,
      has_input_attn_mask ? &decode_input_buffers_[*signatures_.input_attn_mask]
                          : nullptr,
      has_input_int32_param ? &decode_prev_param_ : nullptr,
      has_input_int32_param
          ? &decode_input_buffers_[*signatures_.input_int32_param]
          : nullptr);
}

absl::Status LlmLiteRtCompiledModelExecutorBase::SampleLogits(
    const TensorBuffer& logits, TensorBuffer& ids_tensor) {
  if (sampler_ == nullptr) {
    LITERT_ASSIGN_OR_RETURN(auto logits_tensor_type, logits.TensorType());
    ActivationDataType logits_data_type;
    if (logits_tensor_type.ElementType() == ElementType::Float16) {
      logits_data_type = ActivationDataType::FLOAT16;
    } else if (logits_tensor_type.ElementType() == ElementType::Float32) {
      logits_data_type = ActivationDataType::FLOAT32;
    } else {
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported logits data type for sampler: ",
                       static_cast<int>(logits_tensor_type.ElementType())));
    }

    ABSL_RETURN_IF_ERROR(InitializeSampler(logits_data_type));
  }

  if (sampler_handles_input_) {
    ABSL_RETURN_IF_ERROR(SwapSamplerInputTensors());
  }

  ABSL_RETURN_IF_ERROR(sampler_->SampleToIdAndScoreBuffer(
      logits, ids_tensor, /*scores_tensor=*/nullptr));
  return absl::OkStatus();
}

absl::Status LlmLiteRtCompiledModelExecutorBase::UpdateExecutorSettings(
    const LlmExecutorSettings& executor_settings) {
  executor_settings_ = executor_settings;
  if (executor_settings_.GetAdvancedSettings().has_value()) {
    gpu_enable_metal_residency_set_ = executor_settings_.GetAdvancedSettings()
                                          ->gpu_enable_metal_residency_set;
  }
  return absl::OkStatus();
}

litert::Options LlmLiteRtCompiledModelExecutorBase::GetRunOptions() const {
  litert::Options run_options;
#if defined(__APPLE__)
  auto gpu_options = run_options.GetOptions<::litert::GpuOptions>();
  if (gpu_options.HasValue()) {
    (void)gpu_options->EnableMetalResidencySet(gpu_enable_metal_residency_set_);
  }
#endif
  return run_options;
}

absl::Status LlmLiteRtCompiledModelExecutorBase::SetCurrentStep(int new_step) {
  ABSL_ASSIGN_OR_RETURN(auto old_step, GetCurrentStep());
  if (old_step == new_step) {
    return absl::OkStatus();
  }

  int max_step = old_step;
  ABSL_ASSIGN_OR_RETURN(auto processed_tokens, GetProcessedTokens());
  max_step = processed_tokens->TokenCount();
  RET_CHECK_LE(new_step, max_step).SetCode(absl::StatusCode::kInvalidArgument)
      << "New step cannot be greater than the max step: " << max_step;
  RET_CHECK_GE(new_step, 0).SetCode(absl::StatusCode::kInvalidArgument)
      << "New step cannot be negative.";
  if (new_step == max_step) {
    llm_context_->runtime_state().current_step = new_step;
    return absl::OkStatus();
  }
  RET_CHECK_LE(new_step, max_step).SetCode(absl::StatusCode::kInvalidArgument)
      << "New step cannot be greater than the max step: " << max_step;
  if (new_step < 0) {
    // Current step is negative after rolling back. This can only happen when
    // the user wants to set the step to 0 while there is a pending input token.
    // Thus we can roll back executor state to step 0.
    return Reset();
  }
  llm_context_->runtime_state().current_step = new_step;

  return absl::OkStatus();
}

absl::Status LlmLiteRtCompiledModelExecutorBase::Reset() {
  llm_context_->runtime_state().current_step = 0;
  return absl::OkStatus();
}

absl::StatusOr<int> LlmLiteRtCompiledModelExecutorBase::GetVocabSize() {
  if (!decode_output_buffers_.contains(signatures_.output_logits)) {
    return absl::NotFoundError("Output logits info not found.");
  }

  LITERT_ASSIGN_OR_RETURN(
      auto logits_tensor_type,
      decode_output_buffers_[signatures_.output_logits].TensorType());
  RET_CHECK_EQ(logits_tensor_type.Layout().Dimensions().size(), 3);
  return logits_tensor_type.Layout().Dimensions()[2];
}

absl::StatusOr<litert::Profiler>
LlmLiteRtCompiledModelExecutorBase::GetProfiler() const {
  if (compiled_model_ == nullptr) {
    return absl::FailedPreconditionError("Compiled model is null.");
  }
  auto holder = env_.GetHolder();
  if (holder.runtime == nullptr) {
    return absl::FailedPreconditionError(
        "LiteRT runtime proxy is null in environment.");
  }
  if (holder.handle == nullptr) {
    return absl::FailedPreconditionError("LiteRT environment handle is null.");
  }
  LiteRtProfiler profiler = nullptr;
  LITERT_RETURN_IF_ERROR(holder.runtime->CompiledModelGetProfiler(
      compiled_model_->Get(), &profiler));
  return litert::Profiler(profiler, litert::OwnHandle::kNo);
}

absl::Status LlmLiteRtCompiledModelExecutorBase::StartProfiling() {
  ABSL_ASSIGN_OR_RETURN(auto profiler, GetProfiler());
  if (!profiler || profiler.Get() == nullptr) {
    return absl::FailedPreconditionError("Profiling is not enabled.");
  }
  LITERT_RETURN_IF_ERROR(profiler.StartProfiling());
  return absl::OkStatus();
}

absl::Status LlmLiteRtCompiledModelExecutorBase::StopProfiling() {
  ABSL_ASSIGN_OR_RETURN(auto profiler, GetProfiler());
  if (!profiler || profiler.Get() == nullptr) {
    return absl::FailedPreconditionError("Profiling is not enabled.");
  }
  LITERT_RETURN_IF_ERROR(profiler.StopProfiling());
  return absl::OkStatus();
}

absl::StatusOr<std::string>
LlmLiteRtCompiledModelExecutorBase::GetProfileSummary() {
  ABSL_ASSIGN_OR_RETURN(auto profiler, GetProfiler());
  if (!profiler || profiler.Get() == nullptr) {
    return absl::FailedPreconditionError("Profiling is not enabled.");
  }
  LITERT_ASSIGN_OR_RETURN(auto summary,
                          profiler.GetProfileSummary(compiled_model_->Get()));
  return summary;
}

/* ===========================================================================*/
/* LlmLiteRtCompiledModelExecutorStatic */
/* ===========================================================================*/

absl::Status LlmLiteRtCompiledModelExecutorStatic::Prefill(
    const ExecutorInputs& inputs, const ExecutorPrefillParams& params) {

  int output_heads = 1;
  if (llm_context_->runtime_config().output_heads.has_value()) {
    output_heads = llm_context_->runtime_config().output_heads.value();
  }

  // For now, we reduce the input and processed tokens for prefill only with
  // the first input and processed tokens. This should be updated if user select
  // the decode output candidate.
  constexpr int kTokenIndexToReduce = 0;
  LITERT_RETURN_IF_ERROR(PrepareFirstPrefillAfterDecode(kTokenIndexToReduce));

  LITERT_ASSIGN_OR_RETURN(auto token_ids_buffer, inputs.GetTextTokenIdsPtr());
  LITERT_ASSIGN_OR_RETURN(auto tensor_type, token_ids_buffer->TensorType());
  // Accept batch size 1 or output_heads though prefill handles only the
  // first batch element.
  int32_t input_batch_size = tensor_type.Layout().Dimensions()[0];
  if (input_batch_size != 1) {
    RET_CHECK_EQ(input_batch_size, output_heads);
  }
  RET_CHECK_GT(tensor_type.Layout().Dimensions()[1], 0)
      << "Prefill token ids must be non-empty.";

  if (embedding_lookup_ != nullptr) {
    ABSL_RETURN_IF_ERROR(embedding_lookup_->UpdateMultiModalEmbeddings(inputs));
  }

  LITERT_ASSIGN_OR_RETURN(auto ids,
                          ReferTensorBufferAsSpan<int32_t>(*token_ids_buffer));
  // Reduce the input ids only with one user selected.
  auto input_length = ids.size() / input_batch_size;
  ids = ids.subspan(kTokenIndexToReduce * input_length, input_length);
  int remaining_capacity =
      state_->GetNumEntries() - llm_context_->runtime_state().current_step;

  const bool is_cpu = executor_settings_.GetBackend() == Backend::CPU;
  ABSL_ASSIGN_OR_RETURN(auto work_groups, GetOptimizedPrefillWorkGroups(
                                              prefill_signature_map_,
                                              ids.size(), remaining_capacity,
                                              /*use_greedy_chunking=*/is_cpu));
  for (int i = 0; i < work_groups.size(); ++i) {
    const auto& prefill_signature = work_groups[i].first;
    int prefill_length = work_groups[i].second;
    // Keep track of the signatures that have already had their buffers
    // created only create them once.
    if (!prefill_input_buffers_.contains(prefill_signature)) {
      prefill_input_buffers_[prefill_signature] = {};
      ABSL_RETURN_IF_ERROR(CreatePrefillInputBuffers(
          prefill_signature, prefill_length, prefill_length,
          prefill_input_buffers_[prefill_signature]));
    }
    if (!prefill_output_buffers_.contains(prefill_signature)) {
      prefill_output_buffers_[prefill_signature] = {};
      ABSL_RETURN_IF_ERROR(CreatePrefillOutputBuffers(
          prefill_signature, prefill_length,
          prefill_output_buffers_[prefill_signature]));
    }

    // TODO: b/494284915 - Switch to use async prefill for Metal backend.
    if (!do_prefill_sync_.has_value()) {
      do_prefill_sync_ = std::any_of(
          prefill_input_buffers_[prefill_signature].begin(),
          prefill_input_buffers_[prefill_signature].end(),
          [](const auto& pair) { return pair.second.IsMetalMemory(); });
    }
    bool async = !*do_prefill_sync_ &&
                 (i < work_groups.size() - 1 || !params.GetWaitForCompletion());
    ABSL_RETURN_IF_ERROR(PrefillInternal(
        prefill_signature, prefill_input_buffers_[prefill_signature],
        prefill_output_buffers_[prefill_signature],
        ids.subspan(/*pos=*/0, prefill_length), async, &inputs));
    ids = ids.subspan(/*pos=*/prefill_length);
  }
  RET_CHECK_EQ(ids.size(), 0).SetCode(absl::StatusCode::kInternal)
      << "Work groups not covering the entire prefill input.";

  if (embedding_lookup_ != nullptr) {
    ABSL_RETURN_IF_ERROR(embedding_lookup_->CleanupMultiModalEmbeddings());
  }

  return absl::OkStatus();
}

// static
// Creates a LlmLiteRtCompiledModelExecutorStatic from a LiteRt model.
absl::StatusOr<std::unique_ptr<LlmLiteRtCompiledModelExecutorStatic>>
LlmLiteRtCompiledModelExecutorStatic::Create(
    LlmExecutorSettings executor_settings, Environment& lrt_env,
    ModelResources& resources) {
  ABSL_ASSIGN_OR_RETURN(
      auto litert_model,
      resources.GetTFLiteModel(ModelType::kTfLitePrefillDecode));
  std::string cache_path = executor_settings.GetCacheDir();
  auto activation_data_type = ActivationDataType::FLOAT16;
  // TODO: b/433590109 - Some GPUs do not support FP16, so we need to check the
  // capabilities of the GPU and set the activation data type accordingly.
  if (executor_settings.GetActivationDataType().has_value()) {
    activation_data_type = executor_settings.GetActivationDataType().value();
  }
  const Backend backend = executor_settings.GetBackend();
  bool use_generic_npu_compiler_plugin = false;
  if (backend == Backend::NPU) {
    auto npu_config = executor_settings.GetBackendConfig<NpuConfig>();
    use_generic_npu_compiler_plugin =
        npu_config.ok() && npu_config->use_generic_litert_compiler_plugin;
  }
  bool use_fp16_precision =
      activation_data_type == ActivationDataType::FLOAT16 &&
      backend == Backend::GPU;

  if (!litert_model || !*litert_model) {
    return absl::InternalError("Failed to build LiteRt model");
  }

  const proto::ExecutorMetadata* executor_metadata = nullptr;
  auto executor_metadata_or = resources.GetExecutorMetadata();
  if (executor_metadata_or.ok()) {
    executor_metadata = *executor_metadata_or;
  }

  absl::string_view prefill_signature_key = "";
  for (int i = 0; i < litert_model->GetNumSignatures(); ++i) {
    LITERT_ASSIGN_OR_RETURN(auto sig, litert_model->GetSignature(i));
    absl::string_view key = sig.Key();
    if (absl::StartsWith(key, kPrefillSignatureRunner)) {
      prefill_signature_key = key;
      break;
    }
  }

  LITERT_ASSIGN_OR_RETURN(auto decode_signature,
                          litert_model->FindSignature(kDecodeSignatureRunner));
  ABSL_ASSIGN_OR_RETURN(
      ModelSignatures signatures,
      GetModelSignaturesFromInputOutputNames(decode_signature.InputNames(),
                                             decode_signature.OutputNames()));

  LITERT_ASSIGN_OR_RETURN(
      auto compilation_options,
      CreateCompilationOptions(executor_settings, activation_data_type,
                               &signatures));

  ABSL_RETURN_IF_ERROR(SetExternalWeightOptions(
      resources, ModelType::kTfLitePrefillDecode, compilation_options));

  std::unique_ptr<CompiledModel> compiled_model;
  {
    LITERT_ASSIGN_OR_RETURN(auto compiled_model_tmp,
                            CompiledModel::Create(lrt_env, litert_model->Get(),
                                                  compilation_options));
    compiled_model =
        std::make_unique<CompiledModel>(std::move(compiled_model_tmp));
  }

  ABSL_ASSIGN_OR_RETURN(
      auto prefill_runner_set,
      GetPrefillRunnerSetFromModel(
          *litert_model, kPrefillSignatureRunner,
          /*input_positions_name=*/signatures.input_positions));
  RET_CHECK(!prefill_runner_set.empty()) << "No prefill runner available.";

  LitertState::AllocationPolicy allocation_policy =
      LitertState::AllocationPolicy::kInplace;
  if (backend == Backend::GPU) {
    if (signatures.input_int32_param.has_value()) {
      allocation_policy = LitertState::AllocationPolicy::kGpuOptimizedInplace;
    } else {
      allocation_policy = LitertState::AllocationPolicy::kPingPong;
    }
  }

  bool clear_kv_cache_before_prefill =
      !executor_settings.GetAdvancedSettings() ||
      executor_settings.GetAdvancedSettings()->clear_kv_cache_before_prefill;

  LITERT_ASSIGN_OR_RETURN(
      auto state,
      LitertState::Create(lrt_env, *compiled_model, prefill_signature_key,
                          executor_metadata, allocation_policy,
                          /*batch_size=*/1, clear_kv_cache_before_prefill));
  ClampMaxNumTokens(executor_settings, state->GetNumEntries());

  absl::flat_hash_map<absl::string_view, TensorBuffer> decode_input_buffers;
  absl::flat_hash_map<absl::string_view, TensorBuffer> decode_output_buffers;
  for (auto input_name : decode_signature.InputNames()) {
    if (IsLoRAInputName(input_name)) {
      // We let LoraManager handle LoRA inputs.
      continue;
    }
    if (state->Contains(input_name)) {
      continue;
    }
    LITERT_ASSIGN_OR_RETURN(
        auto input_buffer,
        compiled_model->CreateInputBuffer(kDecodeSignatureRunner, input_name));
    decode_input_buffers[input_name] = std::move(input_buffer);
  }
  LITERT_ASSIGN_OR_RETURN(
      size_t decode_signature_index,
      compiled_model->GetSignatureIndex(kDecodeSignatureRunner));
  for (size_t i = 0; i < decode_signature.OutputNames().size(); ++i) {
    auto output_name = decode_signature.OutputNames()[i];
    if (state->Contains(output_name)) {
      continue;
    }
    // If we are using the GPU sampler and the model is compiled with FP16
    // precision, we force the output logits to be FP16 as the
    // GPU sampler supports FP16 inputs.
    // If we use CPU sampler or the model is executed with FP32 / mixed
    // precision, we will keep the logits in FP32
    auto sampler_backend = GetSamplerBackend(executor_settings);

    if (output_name == signatures.output_logits && use_fp16_precision &&
        sampler_backend.ok() && *sampler_backend == Backend::GPU) {
      LITERT_ASSIGN_OR_RETURN(
          size_t signature_index,
          compiled_model->GetSignatureIndex(kDecodeSignatureRunner));
      LITERT_ASSIGN_OR_RETURN(
          auto output_buffer,
          CreateFP16OutputBuffer(lrt_env, *compiled_model, signature_index,
                                 output_name, i));
      decode_output_buffers[output_name] = std::move(output_buffer);
    } else {
      auto output_buffer_or = compiled_model->CreateOutputBuffer(
          kDecodeSignatureRunner, output_name);
      if (output_buffer_or) {
        decode_output_buffers[output_name] = std::move(*output_buffer_or);
        continue;
      }
      if (!use_generic_npu_compiler_plugin) {
        LITERT_ASSIGN_OR_RETURN(auto output_buffer,
                                std::move(output_buffer_or));
        decode_output_buffers[output_name] = std::move(output_buffer);
        continue;
      }
      ABSL_LOG(WARNING) << "Falling back to host memory for NPU decode output '"
                        << output_name
                        << "' after compiled-model output buffer allocation "
                        << "failed: " << output_buffer_or.Error().Message();
      LITERT_ASSIGN_OR_RETURN(auto output_tensor_type,
                              decode_signature.OutputTensorType(i));
      ABSL_ASSIGN_OR_RETURN(
          auto output_buffer,
          CreateHostOutputBuffer(lrt_env, *compiled_model,
                                 decode_signature_index, i,
                                 std::move(output_tensor_type)));
      decode_output_buffers[output_name] = std::move(output_buffer);
    }
  }

  LITERT_ASSIGN_OR_RETURN(
      auto output_logits_buffer,
      decode_output_buffers[signatures.output_logits].Duplicate());
  LITERT_ASSIGN_OR_RETURN(auto output_logits_buffer_tensor_type,
                          output_logits_buffer.TensorType());
  RET_CHECK(output_logits_buffer_tensor_type.Layout().Dimensions().size() == 3)
      << "Output logits must be (batch, seq, vocab)";
  int batch_size = output_logits_buffer_tensor_type.Layout().Dimensions()[0];

  std::unique_ptr<LitertState> decode_state;
  if (batch_size > 1) {
    ABSL_VLOG(1) << "Decode batch size is larger than 1. Allocate decode "
                 << "only KV cache buffers.";
    LITERT_ASSIGN_OR_RETURN(
        decode_state,
        LitertState::Create(lrt_env, *compiled_model, kDecodeSignatureRunner,
                            executor_metadata, allocation_policy, batch_size,
                            clear_kv_cache_before_prefill));
  }

  std::unique_ptr<EmbeddingLookupManager> embedding_lookup;
  std::unique_ptr<EmbeddingLookupManager> per_layer_embedding_lookup;
  ABSL_RETURN_IF_ERROR(InitializeEmbeddingLookups(
      lrt_env, resources, embedding_lookup, per_layer_embedding_lookup));
  std::unique_ptr<LlmLiteRtMtpDrafter> mtp_drafter;
  {
    const auto& advanced_settings = executor_settings.GetAdvancedSettings();
    if (advanced_settings.has_value() &&
        advanced_settings->enable_speculative_decoding) {
      RET_CHECK_EQ(batch_size, 1)
          << "Speculative decoding (MTP) only supports a single output head.";
      RET_CHECK_NE(embedding_lookup, nullptr);
      std::optional<std::reference_wrapper<EmbeddingLookupManager>>
          ple_manager_opt;
      if (per_layer_embedding_lookup) {
        ple_manager_opt = std::ref(*per_layer_embedding_lookup);
      }
      ABSL_ASSIGN_OR_RETURN(
          mtp_drafter,
          LlmLiteRtMtpDrafter::Create(lrt_env, resources, executor_settings,
                                      *compiled_model, *embedding_lookup,
                                      ple_manager_opt, executor_metadata));
    }
  }

  bool enable_profiling =
      executor_settings.GetAdvancedSettings() &&
      executor_settings.GetAdvancedSettings()->enable_profiling;
  auto executor = absl::WrapUnique(new LlmLiteRtCompiledModelExecutorStatic(
      std::move(executor_settings), lrt_env, litert_model,
      std::move(compiled_model), std::move(decode_input_buffers),
      std::move(decode_output_buffers), std::move(state),
      std::move(decode_state), std::move(prefill_runner_set), signatures,
      batch_size, std::move(cache_path), std::move(embedding_lookup),
      std::move(per_layer_embedding_lookup), use_fp16_precision,
      activation_data_type, std::move(mtp_drafter), executor_metadata,
      &resources));

  if (enable_profiling) {
    auto status = executor->StartProfiling();
    if (!status.ok()) {
      ABSL_LOG(WARNING) << "Failed to start profiling: " << status;
    }
  }
  return executor;
}

absl::Status LlmLiteRtCompiledModelExecutorStatic::UpdateExecutorSettings(
    const LlmExecutorSettings& executor_settings) {
  ABSL_RETURN_IF_ERROR(
      LlmLiteRtCompiledModelExecutorBase::UpdateExecutorSettings(
          executor_settings));
  if (state_ != nullptr) {
    ClampMaxNumTokens(executor_settings_, state_->GetNumEntries());
  }
  return absl::OkStatus();
}

/* ===========================================================================*/
/* LlmLiteRtCompiledModelExecutorDynamic */
/* ===========================================================================*/

absl::Status LlmLiteRtCompiledModelExecutorDynamic::Prefill(
    const ExecutorInputs& inputs, const ExecutorPrefillParams& params) {

  // Only accept batch size 1 for now.
  LITERT_RETURN_IF_ERROR(PrepareFirstPrefillAfterDecode(0));

  if (embedding_lookup_ != nullptr) {
    ABSL_RETURN_IF_ERROR(embedding_lookup_->UpdateMultiModalEmbeddings(inputs));
  }
  auto cleanup = absl::MakeCleanup([this]() {
    if (embedding_lookup_ != nullptr) {
      embedding_lookup_->CleanupMultiModalEmbeddings().IgnoreError();
    }
  });

  LITERT_ASSIGN_OR_RETURN(auto token_ids_buffer, inputs.GetTextTokenIdsPtr());
  LITERT_ASSIGN_OR_RETURN(auto tensor_type, token_ids_buffer->TensorType());
  RET_CHECK_EQ(tensor_type.Layout().Dimensions()[0], 1);
  RET_CHECK_GT(tensor_type.Layout().Dimensions()[1], 0)
      << "Prefill token ids must be non-empty.";
  LITERT_ASSIGN_OR_RETURN(absl::Span<int> ids,
                          ReferTensorBufferAsSpan<int32_t>(*token_ids_buffer));

  if (prefill_chunk_size_ <= 0) {
    return PrefillInternal(ids, params);
  }

  while (!ids.empty()) {
    int chunk_size =
        std::min(static_cast<int>(ids.size()), prefill_chunk_size_);
    absl::Span<int> chunk_ids = ids.first(chunk_size);
    ids = ids.subspan(chunk_size);
    ABSL_RETURN_IF_ERROR(PrefillInternal(chunk_ids, params));
  }
  return absl::OkStatus();
}

absl::Status LlmLiteRtCompiledModelExecutorDynamic::PrefillInternal(
    absl::Span<int> ids, const ExecutorPrefillParams& params) {
  ABSL_RETURN_IF_ERROR(RollBackProcessedTokens());
  // Check if have a pending input token. Note that 'internal_start_step' is
  // always equal to the number of processed tokens plus 1.
  ProcessedTokens::StepAndToken step_and_token =
      llm_context_->processed_context()
          .processed_tokens()
          .GetNextUnprocessedToken();
  bool has_pending_input_token = !step_and_token.token.empty();
  int prefill_length = has_pending_input_token ? ids.size() : ids.size() - 1;
  // If there is no pending input token and no input token to prefill, we can
  // return early by storing the token as a pending input token.
  if (!has_pending_input_token && prefill_length == 0) {
    auto pending_token = std::make_shared<TokenData>(ids[0]);
    if (embedding_lookup_ != nullptr) {
      ABSL_RETURN_IF_ERROR(embedding_lookup_->LookupPrefill(
          pending_token->id(), pending_token->mutable_embedding()));
      if (per_layer_embedding_lookup_ != nullptr) {
        ABSL_RETURN_IF_ERROR(per_layer_embedding_lookup_->LookupPrefill(
            pending_token->id(), pending_token->mutable_per_layer_embedding()));
      }
    }
    ABSL_RETURN_IF_ERROR(llm_context_->processed_context()
                             .processed_tokens()
                             .AddPendingInputToken({std::move(pending_token)}));
    ++llm_context_->runtime_state().current_step;
    return absl::OkStatus();
  }

  auto* litert_state = dynamic_cast<LitertState*>(state_.get());
  RET_CHECK(litert_state != nullptr);

  int kv_length = litert_state->GetNumEntries();
  if (kv_length == 1 && step_and_token.step == 0) {
    LITERT_RETURN_IF_ERROR(litert_state->Resize(
        *compiled_model_, kPrefillSignatureRunner, prefill_length));
    kv_length = prefill_length;
  } else {
    int free_kv_entries = kv_length - step_and_token.step;
    if (prefill_length > free_kv_entries) {
      int new_kv_seq_len = kv_length + prefill_length;
      LITERT_RETURN_IF_ERROR(litert_state->Resize(
          *compiled_model_, kPrefillSignatureRunner, new_kv_seq_len));
      kv_length = new_kv_seq_len;
    }
  }

  absl::flat_hash_map<absl::string_view, TensorBuffer> prefill_input_buffers;
  ABSL_RETURN_IF_ERROR(CreatePrefillInputBuffers(
      "prefill", prefill_length, kv_length, prefill_input_buffers));
  absl::flat_hash_map<absl::string_view, TensorBuffer> prefill_output_buffers;
  ABSL_RETURN_IF_ERROR(CreatePrefillOutputBuffers("prefill", prefill_length,
                                                  prefill_output_buffers));

  bool async = !params.GetWaitForCompletion();
  return LlmLiteRtCompiledModelExecutorBase::PrefillInternal(
      "prefill", prefill_input_buffers, prefill_output_buffers, ids, async);
}

absl::Status LlmLiteRtCompiledModelExecutorDynamic::DecodeInternal(
    const std::vector<std::shared_ptr<TokenData>>& token,
    TensorBuffer& output_logits) {
  auto* litert_state = dynamic_cast<LitertState*>(state_.get());
  RET_CHECK(litert_state != nullptr);

  int current_kv_len = litert_state->GetNumEntries();

  if (current_kv_len <= llm_context_->runtime_state().current_step - 1) {
    int entries_to_add = kv_increament_size_;
    int new_kv_len = current_kv_len + entries_to_add;
    LITERT_RETURN_IF_ERROR(litert_state->Resize(
        *compiled_model_, kDecodeSignatureRunner, new_kv_len));
    current_kv_len = new_kv_len;
  }

  ABSL_RETURN_IF_ERROR(ResolveDynamicShape(*compiled_model_, "decode",
                                           signatures_.input_attn_mask.value(),
                                           current_kv_len));
  LITERT_ASSIGN_OR_RETURN(
      decode_input_buffers_[signatures_.input_attn_mask.value()],
      compiled_model_->CreateInputBuffer("decode",
                                         signatures_.input_attn_mask.value()));

  return LlmLiteRtCompiledModelExecutorBase::DecodeInternal(token,
                                                            output_logits);
}

// static
// Creates a LlmLiteRtCompiledModelExecutorDynamic from a LiteRt model.
absl::StatusOr<std::unique_ptr<LlmLiteRtCompiledModelExecutorDynamic>>
LlmLiteRtCompiledModelExecutorDynamic::Create(
    LlmExecutorSettings executor_settings, Environment& lrt_env,
    ModelResources& resources) {
  ABSL_ASSIGN_OR_RETURN(
      auto litert_model,
      resources.GetTFLiteModel(ModelType::kTfLitePrefillDecode));

  const proto::ExecutorMetadata* executor_metadata = nullptr;
  auto executor_metadata_or = resources.GetExecutorMetadata();
  if (executor_metadata_or.ok()) {
    executor_metadata = *executor_metadata_or;
  }
  ABSL_ASSIGN_OR_RETURN(
      auto compilation_options,
      CreateCompilationOptions(executor_settings, ActivationDataType::FLOAT32,
                               /*signatures=*/std::nullopt));
  std::string weight_cache_path = executor_settings.GetCacheDir();

  const Backend backend = executor_settings.GetBackend();
  RET_CHECK_EQ(backend, Backend::CPU)
      << "LlmLiteRtCompiledModelExecutorDynamic only supports CPU backend.";
  uint32_t kv_increament_size = 0;
  int prefill_chunk_size = -1;
  {
    ABSL_ASSIGN_OR_RETURN(const auto& cpu_config,
                          executor_settings.GetBackendConfig<CpuConfig>());
    kv_increament_size = cpu_config.kv_increment_size;
    prefill_chunk_size = cpu_config.prefill_chunk_size;
    RET_CHECK_GT(kv_increament_size, 0)
        << "KV increment size must be greater than 0.";
  }

  std::unique_ptr<CompiledModel> compiled_model;
  {
    LITERT_ASSIGN_OR_RETURN(auto compiled_model_tmp,
                            CompiledModel::Create(lrt_env, litert_model->Get(),
                                                  compilation_options));
    compiled_model =
        std::make_unique<CompiledModel>(std::move(compiled_model_tmp));
  }

  LITERT_ASSIGN_OR_RETURN(auto decode_signature,
                          litert_model->FindSignature(kDecodeSignatureRunner));
  ABSL_ASSIGN_OR_RETURN(
      ModelSignatures signatures,
      GetModelSignaturesFromInputOutputNames(decode_signature.InputNames(),
                                             decode_signature.OutputNames()));

  LITERT_ASSIGN_OR_RETURN(
      const SimpleTensor& output_logits_tensor,
      decode_signature.OutputTensor(signatures.output_logits));
  LITERT_ASSIGN_OR_RETURN(const RankedTensorType output_logits_tensor_type,
                          output_logits_tensor.RankedTensorType());
  RET_CHECK(output_logits_tensor_type.Layout().Dimensions().size() == 3)
      << "Output logits must be (batch, seq, vocab)";
  int batch_size = output_logits_tensor_type.Layout().Dimensions()[0];
  RET_CHECK_EQ(batch_size, 1) << "Only support batch size 1 for now.";

  bool clear_kv_cache_before_prefill =
      !executor_settings.GetAdvancedSettings() ||
      executor_settings.GetAdvancedSettings()->clear_kv_cache_before_prefill;

  LITERT_ASSIGN_OR_RETURN(
      auto state, LitertState::Create(
                      lrt_env, *compiled_model, "prefill", executor_metadata,
                      LitertState::AllocationPolicy::kInplace, batch_size,
                      clear_kv_cache_before_prefill));

  absl::flat_hash_map<absl::string_view, TensorBuffer> decode_input_buffers;
  absl::flat_hash_map<absl::string_view, TensorBuffer> decode_output_buffers;

  for (auto input_name : decode_signature.InputNames()) {
    if (state->Contains(input_name)) {
      continue;
    }
    bool is_attn_mask_input =
        signatures.input_attn_mask.has_value() &&
        absl::StartsWith(input_name, signatures.input_attn_mask.value());
    if (!is_attn_mask_input) {
      LITERT_ASSIGN_OR_RETURN(auto input_buffer,
                              compiled_model->CreateInputBuffer(
                                  kDecodeSignatureRunner, input_name));
      decode_input_buffers[input_name] = std::move(input_buffer);
    }
  }
  for (auto output_name : decode_signature.OutputNames()) {
    if (state->Contains(output_name)) {
      continue;
    }
    LITERT_ASSIGN_OR_RETURN(auto output_buffer,
                            compiled_model->CreateOutputBuffer(
                                kDecodeSignatureRunner, output_name));
    decode_output_buffers[output_name] = std::move(output_buffer);
  }

  std::unique_ptr<EmbeddingLookupManager> embedding_lookup;
  std::unique_ptr<EmbeddingLookupManager> per_layer_embedding_lookup;
  ABSL_RETURN_IF_ERROR(InitializeEmbeddingLookups(
      lrt_env, resources, embedding_lookup, per_layer_embedding_lookup));

  bool enable_profiling =
      executor_settings.GetAdvancedSettings() &&
      executor_settings.GetAdvancedSettings()->enable_profiling;
  auto executor = absl::WrapUnique(new LlmLiteRtCompiledModelExecutorDynamic(
      std::move(executor_settings), lrt_env, litert_model,
      std::move(compiled_model), std::move(decode_input_buffers),
      std::move(decode_output_buffers), std::move(state), prefill_chunk_size,
      kv_increament_size, signatures, batch_size, std::move(weight_cache_path),
      std::move(embedding_lookup), std::move(per_layer_embedding_lookup),
      /*use_fp16_precision=*/false,
      /*logits_data_type=*/LogitsDataType::FLOAT32,
      /*mtp_drafter=*/nullptr, executor_metadata, &resources));
  if (enable_profiling) {
    auto status = executor->StartProfiling();
    if (!status.ok()) {
      ABSL_LOG(WARNING) << "Failed to start profiling: " << status;
    }
  }
  return executor;
}

}  // namespace litert::lm
