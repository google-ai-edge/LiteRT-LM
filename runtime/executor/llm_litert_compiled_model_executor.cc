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
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <utility>
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
#include "runtime/components/constrained_decoding/constrained_decoder.h"
#include "runtime/components/constrained_decoding/constraint.h"
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
    if (embedding_lookup_ == nullptr) {
      return absl::FailedPreconditionError(
          "Input embeddings required by signature but embedding lookup "
          "model is not initialized.");
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
      if (embedding_lookup_ == nullptr) {
        return absl::FailedPreconditionError(
            "Input per layer embeddings required by signature but "
            "embedding lookup model is not initialized.");
      }
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
    Span<const int> ids, bool async) {
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
        // If not using token as lookup, we must have input_embeddings. There is
        // no need to create input_embeddings_ptr because TensorBuffer locking
        // and filling is handled by the embedding lookup.
        if (embedding_lookup_ == nullptr) {
          return absl::FailedPreconditionError(
              "Prefill requires embedding_lookup_ when use_token_as_lookup is "
              "false, but embedding_lookup_ is null.");
        }
        TensorBuffer* prefill_input_embeddings_buffer =
            &(prefill_input_buffers[signatures_.input_embeddings.value()]);
        ABSL_RETURN_IF_ERROR(embedding_lookup_->LookupPrefill(
            processed_input_tokens, prefill_input_embeddings_buffer,
            /*offset=*/input_idx));

        // We may have per layer embedding as well.
        if (signatures_.input_per_layer_embeddings) {
          if (per_layer_embedding_lookup_ == nullptr) {
            return absl::FailedPreconditionError(
                "Prefill requires per_layer_embedding_lookup_ when signature "
                "has input_per_layer_embeddings, but per_layer_embedding_"
                "lookup_ is null.");
          }
          TensorBuffer* prefill_input_per_layer_embeddings_buffer =
              &(prefill_input_buffers[signatures_.input_per_layer_embeddings
                                          .value()]);
          ABSL_RETURN_IF_ERROR(per_layer_embedding_lookup_->LookupPrefill(
              processed_input_tokens, prefill_input_per_layer_embeddings_buffer,
              /*offset=*/input_idx));
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
      // Look up the embeddings for the last token so they can be used in the
      // next prefill or decode. This has to be done now in the case of
      // multi-modal prefill so the embeddings are used in the correct order.
      if (embedding_lookup_ == nullptr) {
        return absl::FailedPreconditionError(
            "Prefill requires embedding_lookup_ for the last pending token "
            "when use_token_as_lookup is false, but embedding_lookup_ is "
            "null.");
      }
      ABSL_RETURN_IF_ERROR(embedding_lookup_->LookupPrefill(
          last_input_token->id(), last_input_token->mutable_embedding()));
      if (use_per_layer_embedding) {
        if (per_layer_embedding_lookup_ == nullptr) {
          return absl::FailedPreconditionError(
              "Prefill requires per_layer_embedding_lookup_ for the last "
              "pending token, but per_layer_embedding_lookup_ is null.");
        }
        ABSL_RETURN_IF_ERROR(per_layer_embedding_lookup_->LookupPrefill(
            last_input_token->id(),
            last_input_token->mutable_per_layer_embedding()));
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

absl::StatusOr<std::vector<std::vector<int>>>
LlmLiteRtCompiledModelExecutorBase::Decode() {
  return Decode(ExecutorDecodeParams());
}

absl::StatusOr<std::vector<std::vector<int>>>
LlmLiteRtCompiledModelExecutorBase::Decode(
    const ExecutorDecodeParams& decode_params) {

  std::vector<std::vector<int>> output_tokens_vector;
  if (mtp_drafter_ == nullptr) {
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
    ABSL_RETURN_IF_ERROR(constrained_decoder->ProcessLogits(output_logits));
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
    gpu_enable_metal_residency_set_ =
        executor_settings_.GetAdvancedSettings()
            ->gpu_enable_metal_residency_set;
  }
  return absl::OkStatus();
}

litert::Options LlmLiteRtCompiledModelExecutorBase::GetRunOptions() const {
  litert::Options run_options;
#if defined(__APPLE__)
  auto gpu_options = run_options.GetGpuOptions();
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
        ids.subspan(/*pos=*/0, prefill_length), async));
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
      activation_data_type, std::move(mtp_drafter), executor_metadata));

  if (enable_profiling) {
    auto status = executor->StartProfiling();
    if (!status.ok()) {
      ABSL_LOG(WARNING) << "Failed to start profiling: " << status;
    }
  }
  return executor;
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
      /*mtp_drafter=*/nullptr, executor_metadata));
  if (enable_profiling) {
    auto status = executor->StartProfiling();
    if (!status.ok()) {
      ABSL_LOG(WARNING) << "Failed to start profiling: " << status;
    }
  }
  return executor;
}

}  // namespace litert::lm
