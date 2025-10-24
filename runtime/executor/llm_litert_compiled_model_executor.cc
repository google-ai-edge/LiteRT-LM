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
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
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
#include "litert/cc/options/litert_cpu_options.h"  // from @litert
#include "litert/cc/options/litert_gpu_options.h"  // from @litert
#include "litert/cc/options/litert_runtime_options.h"  // from @litert
#include "runtime/components/embedding_lookup/embedding_lookup_manager.h"
#include "runtime/components/model_resources.h"
#include "runtime/components/sampler_factory.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/litert_compiled_model_executor_utils.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/executor/llm_executor_processed_tokens.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/file_util.h"
#include "runtime/util/lora_util.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/status_macros.h"  // IWYU pragma: keep
#include "tflite/delegates/xnnpack/xnnpack_delegate.h"  // from @litert

namespace litert::lm {
namespace {

using ::absl::Span;
using ::litert::Expected;
using ::litert::GpuOptions;
using ::litert::TensorBuffer;

// Names of the signature runners, used to get the signature runners from the
// interpreter.
constexpr char kPrefillSignatureRunner[] = "prefill";
constexpr char kDecodeSignatureRunner[] = "decode";

absl::Status GetCacheRootNames(std::vector<absl::string_view> input_names,
                               std::string& k_root_name,
                               std::string& v_root_name) {
  for (auto input_name : input_names) {
    if (input_name == "kv_cache_k_0") {
      k_root_name = "kv_cache_k_";
      v_root_name = "kv_cache_v_";
      return absl::OkStatus();
    } else if (input_name == "k_cache_0") {
      k_root_name = "k_cache_";
      v_root_name = "v_cache_";
      return absl::OkStatus();
    }
  }
  return absl::FailedPreconditionError("No KV cache inputs found.");
}

bool IsCalculationPrecisionF16() { return true; }

absl::Status InitializeEmbeddingLookups(
    ModelResources& resources,
    std::unique_ptr<EmbeddingLookupManager>& embedding_lookup,
    std::unique_ptr<EmbeddingLookupManager>& per_layer_embedding_lookup) {
  auto end_of_audio_model =
      resources.GetTFLiteModel(ModelType::kTfLiteEndOfAudio);
  absl::flat_hash_map<int, const litert::Model*>
      end_of_multi_modal_embedding_models;
  if (end_of_audio_model.ok()) {
    end_of_multi_modal_embedding_models.insert(
        {ExecutorAudioData::kEndToken, end_of_audio_model.value()});
  }

  auto text_embedder_model =
      resources.GetTFLiteModel(ModelType::kTfLiteEmbedder);
  if (text_embedder_model.ok()) {
    ASSIGN_OR_RETURN(
        embedding_lookup,
        EmbeddingLookupManager::Create(*text_embedder_model,
                                       end_of_multi_modal_embedding_models));
  }

  // Create per layer embedding lookups from the resources.
  auto per_layer_embedder_model =
      resources.GetTFLiteModel(ModelType::kTfLitePerLayerEmbedder);
  if (per_layer_embedder_model.ok()) {
    ASSIGN_OR_RETURN(
        per_layer_embedding_lookup,
        EmbeddingLookupManager::Create(*per_layer_embedder_model,
                                       /*fully_supports_multi_modal=*/false));
  }
  return absl::OkStatus();
}

// Returns the backend to be used for sampling.
absl::StatusOr<Backend> GetSamplerBackend(
    const LlmExecutorSettings& executor_settings) {
  Backend backend = executor_settings.GetBackend();
  Backend sampler_backend = executor_settings.GetSamplerBackend();

  if (sampler_backend == Backend::UNSPECIFIED) {
    sampler_backend = backend;
  }

  if (sampler_backend != Backend::CPU && sampler_backend != Backend::GPU) {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported sampler backend: ", sampler_backend,
                     " for backend: ", backend));
  }

  return sampler_backend;
}

void LogValues(absl::Span<const float> values, size_t num_values_to_log,
               absl::string_view debug) {
  constexpr size_t kNumExtraValuesToLog = 10;
  if (num_values_to_log * 3 + kNumExtraValuesToLog >= values.size()) {
    ABSL_LOG(INFO) << debug << "(size=" << values.size()
                   << "): " << absl::StrJoin(values, ", ");
    return;
  }

  size_t end_offset = values.size() - num_values_to_log;
  size_t mid_offset = end_offset / 2;
  ABSL_LOG(INFO) << debug << "(size=" << values.size() << "): "
                 << absl::StrJoin(values.subspan(0, num_values_to_log), ", ")
                 << " ... "
                 << absl::StrJoin(values.subspan(mid_offset, num_values_to_log),
                                  ", ")
                 << " ... " << absl::StrJoin(values.subspan(end_offset), ", ");
}

void LogTensor(TensorBuffer& tensor, size_t num_values_to_log,
               absl::string_view debug) {
  // Try to get the reference if tensor is in CPU memory.
  auto values_span = ReferTensorBufferAsSpan<float>(tensor);
  if (values_span) {
    LogValues(*values_span, num_values_to_log, debug);
    return;
  }

  // Otherwise, copy the logits from the tensor buffer to a vector.
  auto values_vector = CopyFromTensorBuffer<float>(tensor);
  if (values_vector) {
    LogValues(*values_vector, num_values_to_log, debug);
    return;
  }

  ABSL_LOG(ERROR) << debug << ": Failed to log logits.";
}

}  // namespace

absl::Status LlmLiteRtCompiledModelExecutor::Prefill(
    const ExecutorInputs& inputs, const ExecutorPrefillParams& params) {
  LITERT_ASSIGN_OR_RETURN(auto tensor_type,
                          (*inputs.GetTextTokenIdsPtr())->TensorType());
  // Only accept batch size 1 for now.
  RET_CHECK_EQ(tensor_type.Layout().Dimensions()[0], 1);
  RET_CHECK_GT(tensor_type.Layout().Dimensions()[1], 0)
      << "Prefill token ids must be non-empty.";
  LITERT_ASSIGN_OR_RETURN(auto ids, ReferTensorBufferAsSpan<int32_t>(
                                        *(*inputs.GetTextTokenIdsPtr())));
  if (embedding_lookup_ != nullptr) {
    RETURN_IF_ERROR(embedding_lookup_->UpdateMultiModalEmbeddings(inputs));
  }

  ASSIGN_OR_RETURN(auto work_groups, GetOptimizedPrefillWorkGroups(
                                         prefill_signature_map_, ids.size()));
  for (const auto& [prefill_signature, prefill_length] : work_groups) {
    // Keep track of the signatures that have already had their buffers
    // created only create them once.
    if (!prefill_input_buffers_.contains(prefill_signature)) {
      RETURN_IF_ERROR(CreatePrefillInputBuffers(prefill_signature));
    }
    RETURN_IF_ERROR(PrefillInternal(prefill_signature,
                                    ids.subspan(/*pos=*/0, prefill_length)));
    ids = ids.subspan(/*pos=*/prefill_length);
  }
  RET_CHECK_EQ(ids.size(), 0).SetCode(absl::StatusCode::kInternal)
      << "Work groups not covering the entire prefill input.";

  // If requested, wait for prefill to complete, for example, by benchmark.
  if (params.GetWaitForCompletion()) {
    // A workaround to sync with backend especially for GPU backends is to do
    // read-lock a small decode buffer, input_positions which most likely
    // consists only of one value.
    if (!signatures_.input_positions.empty() &&
        decode_input_buffers_.contains(signatures_.input_positions)) {
      ABSL_LOG(INFO) << "Waiting for prefill to complete.";
      auto lock = ::litert::TensorBufferScopedLock::Create(
          decode_input_buffers_[signatures_.input_positions],
          TensorBuffer::LockMode::kRead);
      if (!lock) {
        ABSL_LOG(ERROR) << "Failed to lock decode input_positions as a "
                        << "workaround to sync with backend.";
      }
    } else {
      ABSL_LOG(WARNING) << "Ignore waiting for prefill to complete.";
    }
  }

  if (embedding_lookup_ != nullptr) {
    RETURN_IF_ERROR(embedding_lookup_->CleanupMultiModalEmbeddings());
  }

  return absl::OkStatus();
}

absl::Status LlmLiteRtCompiledModelExecutor::CreatePrefillInputBuffers(
    absl::string_view prefill_signature) {
  auto& curr_prefill_input_buffers = prefill_input_buffers_[prefill_signature];
  // Create input_token, positions and attn_mask buffers after determining
  // the prefill length.
  if (!signatures_.input_tokens.empty()) {
    auto tokens_buffer = compiled_model_.CreateInputBuffer(
        prefill_signature, signatures_.input_tokens);
    curr_prefill_input_buffers[signatures_.input_tokens] =
        std::move(*tokens_buffer);
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
    auto embeddings_buffer = compiled_model_.CreateInputBuffer(
        prefill_signature, signatures_.input_embeddings.value());
    curr_prefill_input_buffers[signatures_.input_embeddings.value()] =
        std::move(*embeddings_buffer);

    // We may have per layer embedding as well.
    if (signatures_.input_per_layer_embeddings.has_value()) {
      if (embedding_lookup_ == nullptr) {
        return absl::FailedPreconditionError(
            "Input per layer embeddings required by signature but "
            "embedding lookup model is not initialized.");
      }
      auto per_layer_embeddings_buffer = compiled_model_.CreateInputBuffer(
          prefill_signature, signatures_.input_per_layer_embeddings.value());
      curr_prefill_input_buffers[signatures_.input_per_layer_embeddings
                                     .value()] =
          std::move(*per_layer_embeddings_buffer);
    }
  }
  auto positions_buffer = compiled_model_.CreateInputBuffer(
      prefill_signature, signatures_.input_positions);
  curr_prefill_input_buffers[signatures_.input_positions] =
      std::move(*positions_buffer);

  if (signatures_.input_attn_mask.has_value()) {
    auto attn_mask_buffer = compiled_model_.CreateInputBuffer(
        prefill_signature, signatures_.input_attn_mask.value());
    curr_prefill_input_buffers[signatures_.input_attn_mask.value()] =
        std::move(*attn_mask_buffer);
  }
  return absl::OkStatus();
}

absl::Status LlmLiteRtCompiledModelExecutor::FillInputBufferWithToken(
    std::shared_ptr<TokenData> unprocessed_token,
    ::litert::TensorBuffer& input_buffer, bool is_per_layer_embedding) {
  if (unprocessed_token == nullptr) {
    return absl::InvalidArgumentError("Unprocessed token is null.");
  }
  LITERT_ASSIGN_OR_RETURN(auto input_buffer_lock_and_addr,
                          ::litert::TensorBufferScopedLock::Create(
                              input_buffer, TensorBuffer::LockMode::kWrite));
  if (unprocessed_token->embedding().empty()) {
    // If the token has no embedding, the input_buffer should takes
    // token id.
    int32_t* input_buffer_ptr =
        static_cast<int32_t*>(input_buffer_lock_and_addr.second);
    LITERT_ASSIGN_OR_RETURN(size_t packed_size, input_buffer.PackedSize());
    memset(input_buffer_ptr, 0, packed_size);
    input_buffer_ptr[0] = unprocessed_token->id();
  } else {
    float* input_buffer_ptr =
        static_cast<float*>(input_buffer_lock_and_addr.second);
    if (is_per_layer_embedding) {
      memcpy(input_buffer_ptr, unprocessed_token->per_layer_embedding().data(),
             unprocessed_token->per_layer_embedding().size() * sizeof(float));
    } else {
      memcpy(input_buffer_ptr, unprocessed_token->embedding().data(),
             unprocessed_token->embedding().size() * sizeof(float));
    }
  }
  return absl::OkStatus();
}

absl::Status LlmLiteRtCompiledModelExecutor::PrefillInternal(
    absl::string_view prefill_signature, Span<const int> ids) {

  auto& curr_prefill_input_buffers = prefill_input_buffers_[prefill_signature];
  {
    // Fill the input buffers with scoped locks.
    auto& prefill_input_pos =
        curr_prefill_input_buffers[signatures_.input_positions];
    LITERT_ASSIGN_OR_RETURN(auto prefill_input_pos_size,
                            prefill_input_pos.PackedSize());
    LITERT_ASSIGN_OR_RETURN(
        auto prefill_input_pos_lock_and_addr,
        ::litert::TensorBufferScopedLock::Create(
            prefill_input_pos, TensorBuffer::LockMode::kWrite));
    auto* prefill_input_pos_ptr =
        static_cast<int32_t*>(prefill_input_pos_lock_and_addr.second);

    memset(prefill_input_pos_ptr, 0, prefill_input_pos_size);
    if (signatures_.input_attn_mask.has_value()) {
      RETURN_IF_ERROR(InitializeAttentionMask(
          curr_prefill_input_buffers[signatures_.input_attn_mask.value()],
          IsCalculationPrecisionF16()));
    }
    // TODO(b/425396146): Add the unit tests for checking the prefill length.
    // We always hold one pending token in the input ids for the next
    // prefill or decode step.
    int prefill_length = ids.size() - 1;

    // Check if have a pending input token. Note that 'internal_start_step' is
    // always equal to the number of processed tokens plus 1.
    auto [internal_start_step, pending_input_token] =
        processed_tokens_.GetNextUnprocessedToken();
    const int start_step = internal_start_step;
    const bool has_pending_input_token = pending_input_token != nullptr;
    const bool use_token_as_lookup = !signatures_.input_tokens.empty();
    const bool use_per_layer_embedding =
        signatures_.input_per_layer_embeddings.has_value();
    // If there is no pending input token and no input token to prefill, we can
    // return early by storing the token as a pending input token.
    if (!has_pending_input_token && prefill_length == 0) {
      RETURN_IF_ERROR(processed_tokens_.AddPendingInputToken(
          std::make_shared<TokenData>(ids[0])));
      return absl::OkStatus();
    }
    int input_idx = 0;
    if (has_pending_input_token) {
      if (use_token_as_lookup) {
        RETURN_IF_ERROR(FillInputBufferWithToken(
            pending_input_token,
            curr_prefill_input_buffers[signatures_.input_tokens]));
      } else {
        RETURN_IF_ERROR(FillInputBufferWithToken(
            pending_input_token,
            curr_prefill_input_buffers[signatures_.input_embeddings.value()]));
        if (use_per_layer_embedding) {
          RETURN_IF_ERROR(FillInputBufferWithToken(
              pending_input_token,
              curr_prefill_input_buffers[signatures_.input_per_layer_embeddings
                                             .value()],
              /*is_per_layer_embedding=*/true));
        }
      }
      prefill_input_pos_ptr[input_idx] = internal_start_step;
      RETURN_IF_ERROR(processed_tokens_.MarkPendingInputTokenAsProcessed());
      ++prefill_input_pos_ptr;
      ++input_idx;
    }
    std::transform(prefill_input_pos_ptr,
                   prefill_input_pos_ptr + prefill_length,
                   prefill_input_pos_ptr,
                   [&](int token) mutable { return current_step_++; });
    std::vector<int> processed_input_tokens(ids.begin(),
                                            ids.begin() + prefill_length);
    processed_tokens_.AddProcessedTokens(processed_input_tokens);

    if (use_token_as_lookup) {
      auto& prefill_input_buffer =
          curr_prefill_input_buffers[signatures_.input_tokens];
      LITERT_ASSIGN_OR_RETURN(
          auto prefill_input_lock_and_addr,
          ::litert::TensorBufferScopedLock::Create(
              prefill_input_buffer, TensorBuffer::LockMode::kWrite));
      int32_t* prefill_input_ptr =
          static_cast<int32_t*>(prefill_input_lock_and_addr.second);
      if (!has_pending_input_token) {
        LITERT_ASSIGN_OR_RETURN(auto prefill_input_size,
                                prefill_input_buffer.PackedSize());
        // If there is a pending input token, the zeros and the pending input
        // token id are already filled in the above
        // FillInputBufferWithToken() function, so we cannot zero out the
        // whole prefill input buffer here.
        //
        // If there is no pending input token, we need to zero out the whole
        // prefill input buffer.
        memset(prefill_input_ptr, 0, prefill_input_size);
      }
      memcpy(prefill_input_ptr + input_idx, processed_input_tokens.data(),
             processed_input_tokens.size() * sizeof(int32_t));
    } else {
      // If not using token as lookup, we must have input_embeddings. There is
      // no need to create input_embeddings_ptr because TensorBuffer locking and
      // filling is handled by the embedding lookup.
      TensorBuffer* prefill_input_embeddings_buffer =
          &(curr_prefill_input_buffers[signatures_.input_embeddings.value()]);
      RETURN_IF_ERROR(embedding_lookup_->LookupPrefill(
          processed_input_tokens, prefill_input_embeddings_buffer,
          /*offset=*/input_idx));

      // We may have per layer embedding as well.
      if (signatures_.input_per_layer_embeddings) {
        TensorBuffer* prefill_input_per_layer_embeddings_buffer =
            &(curr_prefill_input_buffers[signatures_.input_per_layer_embeddings
                                             .value()]);
        RETURN_IF_ERROR(per_layer_embedding_lookup_->LookupPrefill(
            processed_input_tokens, prefill_input_per_layer_embeddings_buffer,
            /*offset=*/input_idx));
      }
    }
    if (signatures_.input_attn_mask.has_value()) {
      RETURN_IF_ERROR(FillAttentionMask(
          curr_prefill_input_buffers[signatures_.input_attn_mask.value()],
          start_step,
          /*steps=*/prefill_length + input_idx));
    }

    // Add the last token of the current input as a pending input token, to be
    // used in the next prefill or decode.
    std::shared_ptr<TokenData> last_input_token =
        std::make_shared<TokenData>(ids.back());
    if (!use_token_as_lookup) {
      // Look up the embeddings for the last token so they can be used in the
      // next prefill or decode. This has to be done now in the case of
      // multi-modal prefill so the embeddings are used in the correct order.
      RETURN_IF_ERROR(embedding_lookup_->LookupPrefill(
          last_input_token->id(), last_input_token->mutable_embedding()));
      if (use_per_layer_embedding) {
        RETURN_IF_ERROR(per_layer_embedding_lookup_->LookupPrefill(
            last_input_token->id(),
            last_input_token->mutable_per_layer_embedding()));
      }
    }
    // Add the last input token to the pending input token list.
    RETURN_IF_ERROR(
        processed_tokens_.AddPendingInputToken(std::move(last_input_token)));
    current_step_++;
  }

  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      prefill_input_buffers;
  for (const auto& [input_name, input_buffer] : curr_prefill_input_buffers) {
    LITERT_ASSIGN_OR_RETURN(auto input_buffer_dup, input_buffer.Duplicate());
    prefill_input_buffers[input_name] = std::move(input_buffer_dup);
  }
  for (const auto& [input_name, input_buffer] : *input_kv_cache_buffers_) {
    LITERT_ASSIGN_OR_RETURN(auto input_buffer_dup, input_buffer.Duplicate());
    prefill_input_buffers[input_name] = std::move(input_buffer_dup);
  }
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      prefill_output_buffers;
  for (const auto& [output_name, output_buffer] : *output_kv_cache_buffers_) {
    LITERT_ASSIGN_OR_RETURN(auto output_buffer_dup, output_buffer.Duplicate());
    prefill_output_buffers[output_name] = std::move(output_buffer_dup);
  }

  LITERT_RETURN_IF_ERROR(compiled_model_.Run(
      prefill_signature, prefill_input_buffers, prefill_output_buffers));

  std::swap(input_kv_cache_buffers_, output_kv_cache_buffers_);
  return absl::OkStatus();
}

absl::StatusOr<ProcessedTokens::StepAndToken>
LlmLiteRtCompiledModelExecutor::GetTokenToDecode(const ExecutorInputs& inputs) {
  if (inputs.GetTextDataPtr().ok()) {
    auto input_tensor_size = (*inputs.GetTextTokenIdsPtr())->PackedSize();
    if (input_tensor_size && *input_tensor_size != 0) {
      // Input token ids provided, so use it regardless of whether next input
      // token id is set. Only accept batch size 1 and a single token for now.
      RET_CHECK_EQ(*input_tensor_size, 1 * sizeof(int32_t));
      LITERT_ASSIGN_OR_RETURN(auto ids, ReferTensorBufferAsSpan<int32_t>(
                                            *(*inputs.GetTextTokenIdsPtr())));
      const int id = ids[0];
      if (id >= 0) {
        // If the input token id is >= 0, it means the input token is provided
        // by the user. In this case, we should invalidate the pending input
        // token and add the input token as a pending input token.
        processed_tokens_.InvalidatePendingInputToken();
        TokenData token_data(ids[0]);
        RETURN_IF_ERROR(processed_tokens_.AddPendingInputToken(
            std::make_shared<TokenData>(token_data)));
      }
    }
  }
  // Here we must have a pending input token to decode that's either coming from
  // the previous prefill or decode, or we just added one from the inputs.
  auto token = processed_tokens_.GetNextUnprocessedToken().token;
  // If the token has no embedding, we will look up the embedding for the token
  // here. This reduces the complexity for internal or external sampling.
  if (signatures_.input_embeddings.has_value() &&
      token->mutable_embedding().empty()) {
    RETURN_IF_ERROR(embedding_lookup_->LookupDecode(
        token->id(), token->mutable_embedding()));
    if (signatures_.input_per_layer_embeddings.has_value()) {
      RETURN_IF_ERROR(per_layer_embedding_lookup_->LookupDecode(
          token->id(), token->mutable_per_layer_embedding()));
    }
  }
  return processed_tokens_.GetNextUnprocessedToken();
}

absl::Status LlmLiteRtCompiledModelExecutor::ConsumePendingOrAddProcessedToken(
    std::shared_ptr<TokenData> token) {
  auto status = processed_tokens_.MarkPendingInputTokenAsProcessed();
  if (!status.ok()) {
    if (status.message() ==
        "MarkPendingInputTokenAsProcessed called with no pending token.") {
      // If the pending input token was not used, we should add the token to the
      // processed tokens.
      processed_tokens_.AddProcessedTokens({token->id()});
    } else {
      // Return on error other than the expected "no pending token" error.
      return status;
    }
  }
  return absl::OkStatus();
}

absl::Status LlmLiteRtCompiledModelExecutor::DecodeInternal(
    const int step, const std::shared_ptr<TokenData> token,
    TensorBuffer& output_logits) {
  const bool use_token_as_lookup = !signatures_.input_tokens.empty();
  const bool use_per_layer_embedding =
      signatures_.input_per_layer_embeddings.has_value();

  // Fill the input buffers with scoped locks.
  if (use_token_as_lookup) {
    RETURN_IF_ERROR(FillInputBufferWithToken(
        token, decode_input_buffers_[signatures_.input_tokens]));
  } else {
    if (!signatures_.input_embeddings.has_value()) {
      return absl::InvalidArgumentError(
          "Input tokens or embeddings must be provided.");
    }
    RETURN_IF_ERROR(FillInputBufferWithToken(
        token, decode_input_buffers_[signatures_.input_embeddings.value()]));
    if (use_per_layer_embedding) {
      RETURN_IF_ERROR(FillInputBufferWithToken(
          token,
          decode_input_buffers_[signatures_.input_per_layer_embeddings.value()],
          /*is_per_layer_embedding=*/true));
    }
  }

  {
    LITERT_ASSIGN_OR_RETURN(
        auto decode_input_pos_lock_and_addr,
        TensorBufferScopedLock::Create(
            decode_input_buffers_[signatures_.input_positions],
            TensorBuffer::LockMode::kWrite));
    auto* decode_input_pos_ptr =
        static_cast<int32_t*>(decode_input_pos_lock_and_addr.second);
    decode_input_pos_ptr[0] = step;
  }

  if (signatures_.input_attn_mask.has_value()) {
    RETURN_IF_ERROR(InitializeAttentionMask(
        decode_input_buffers_[signatures_.input_attn_mask.value()],
        IsCalculationPrecisionF16()));
    RETURN_IF_ERROR(FillAttentionMask(
        decode_input_buffers_[signatures_.input_attn_mask.value()], step,
        /*steps=*/1));
  }

  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      decode_input_buffers;
  for (const auto& [input_name, input_buffer] : decode_input_buffers_) {
    LITERT_ASSIGN_OR_RETURN(auto input_buffer_dup, input_buffer.Duplicate());
    decode_input_buffers[input_name] = std::move(input_buffer_dup);
  }
  for (const auto& [input_name, input_buffer] : *input_kv_cache_buffers_) {
    LITERT_ASSIGN_OR_RETURN(auto input_buffer_dup, input_buffer.Duplicate());
    decode_input_buffers[input_name] = std::move(input_buffer_dup);
  }
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      decode_output_buffers;
  for (const auto& [output_name, output_buffer] : decode_output_buffers_) {
    // LITERT_ASSIGN_OR_RETURN() causes a compilation error on windows.
    auto output_buffer_dup = output_name == signatures_.output_logits
                                 ? output_logits.Duplicate()
                                 : output_buffer.Duplicate();
    RET_CHECK(output_buffer_dup) << "Failed to duplicate output buffer.";
    decode_output_buffers[output_name] = std::move(*output_buffer_dup);
  }
  for (const auto& [output_name, output_buffer] : *output_kv_cache_buffers_) {
    LITERT_ASSIGN_OR_RETURN(auto output_buffer_dup, output_buffer.Duplicate());
    decode_output_buffers[output_name] = std::move(output_buffer_dup);
  }

  LITERT_RETURN_IF_ERROR(compiled_model_.Run(
      kDecodeSignatureRunner, decode_input_buffers, decode_output_buffers));

  std::swap(input_kv_cache_buffers_, output_kv_cache_buffers_);
  return absl::OkStatus();
}

absl::Status LlmLiteRtCompiledModelExecutor::Decode(
    ::litert::TensorBuffer& output_tokens) {
  return Decode(output_tokens, ExecutorDecodeParams());
}

absl::Status LlmLiteRtCompiledModelExecutor::Decode(
    ::litert::TensorBuffer& output_tokens,
    const ExecutorDecodeParams& decode_params) {

  ASSIGN_OR_RETURN(auto decoded_logits,
                   DecodeLogits(ExecutorInputs(), decode_params));
  RETURN_IF_ERROR(SampleLogits(decoded_logits, output_tokens));

  // Read the first output token for the next input token id.
  bool reset_output_token = false;
  {
    LITERT_ASSIGN_OR_RETURN(auto lock_and_addr,
                            ::litert::TensorBufferScopedLock::Create(
                                output_tokens, TensorBuffer::LockMode::kRead));
    auto output_tokens_ptr = static_cast<int32_t*>(lock_and_addr.second);
    reset_output_token = output_tokens_ptr[0] < 0;
    RETURN_IF_ERROR(
        processed_tokens_.AddPendingInputToken(std::make_shared<TokenData>(
            reset_output_token ? 0 : output_tokens_ptr[0])));
  }

  // If the first output token is invalid, reset it to 0 to avoid crash.
  if (reset_output_token) {
    LITERT_ASSIGN_OR_RETURN(auto lock_and_addr,
                            ::litert::TensorBufferScopedLock::Create(
                                output_tokens, TensorBuffer::LockMode::kWrite));
    ABSL_LOG(WARNING) << "Invalid decode and sample result. The sampled token "
                         "is casted to 0 to avoid crash.";
    auto output_tokens_ptr = static_cast<int32_t*>(lock_and_addr.second);
    output_tokens_ptr[0] = 0;
  }
  return absl::OkStatus();
}

absl::Status LlmLiteRtCompiledModelExecutor::Decode(
    const ExecutorInputs& inputs, ::litert::TensorBuffer& output_logits) {
  ASSIGN_OR_RETURN(auto step_and_token, GetTokenToDecode(inputs));
  RETURN_IF_ERROR(
      DecodeInternal(step_and_token.step, step_and_token.token, output_logits));
  RETURN_IF_ERROR(ConsumePendingOrAddProcessedToken(step_and_token.token));
  current_step_ = step_and_token.step + 1;
  return absl::OkStatus();
}

absl::StatusOr<::litert::TensorBuffer>
LlmLiteRtCompiledModelExecutor::DecodeLogits(const ExecutorInputs& inputs) {
  return DecodeLogits(inputs, ExecutorDecodeParams());
}

absl::StatusOr<::litert::TensorBuffer>
LlmLiteRtCompiledModelExecutor::DecodeLogits(
    const ExecutorInputs& inputs, const ExecutorDecodeParams& decode_params) {
  LITERT_ASSIGN_OR_RETURN(
      auto output_logits,
      decode_output_buffers_[signatures_.output_logits].Duplicate());

  ASSIGN_OR_RETURN(auto step_and_token, GetTokenToDecode(inputs));
  RETURN_IF_ERROR(
      DecodeInternal(step_and_token.step, step_and_token.token, output_logits));
  RETURN_IF_ERROR(ConsumePendingOrAddProcessedToken(step_and_token.token));

  if (decode_params.HasConstraintDecoder()) {
    int current_token_id = step_and_token.token->id();
    // Update constraint state based on the current token id.
    RETURN_IF_ERROR(decode_params.GetConstraintDecoder()->UpdateConstraintState(
        absl::MakeSpan(&current_token_id, 1)));

    LITERT_ASSIGN_OR_RETURN(auto output_logits_buffer_type,
                            output_logits.BufferType());
    // If the output logits are already on the host memory, use the buffer
    // directly.
    if (output_logits_buffer_type == kLiteRtTensorBufferTypeHostMemory) {
      // Mask logits based on the current constraint state.
      RETURN_IF_ERROR(
          decode_params.GetConstraintDecoder()->MaskLogits(output_logits));
    } else {
      // For GPU, we always copy the logits to CPU and mask them, then write
      // them back to GPU.
      LITERT_ASSIGN_OR_RETURN(RankedTensorType logits_tensor_type,
                              output_logits.TensorType());
      if (logits_tensor_type.ElementType() == ::litert::ElementType::Float32) {
        // Copy the logits from the tensor buffer to a vector.
        LITERT_ASSIGN_OR_RETURN(auto logits_vector,
                                CopyFromTensorBuffer<float>(output_logits));
        // Mask logits based on the current constraint state.
        RETURN_IF_ERROR(decode_params.GetConstraintDecoder()->MaskLogits(
            absl::MakeSpan(logits_vector.data(), logits_vector.size()),
            logits_tensor_type.Layout().Dimensions()));
        // Write the masked logits back to the tensor buffer.
        output_logits.Write(
            absl::MakeConstSpan(logits_vector.data(), logits_vector.size()));
      } else {
        return absl::InvalidArgumentError("Output logits are not in float32.");
      }
    }
  }

  current_step_ = step_and_token.step + 1;

  const auto& settings = executor_settings_.GetAdvancedSettings();
  if (settings && settings->num_logits_to_print_after_decode > 0) {
    LogTensor(output_logits, settings->num_logits_to_print_after_decode,
              "Logits");
  }
  return output_logits;
}

absl::Status LlmLiteRtCompiledModelExecutor::InitializeSampler(int batch_size) {
  if (sampler_ != nullptr) {
    return absl::OkStatus();
  }

  ASSIGN_OR_RETURN(auto vocab_size, GetVocabSize());
  ASSIGN_OR_RETURN(auto sampler_backend, GetSamplerBackend(executor_settings_));
  proto::SamplerParameters sampler_params;
  sampler_params.set_type(proto::SamplerParameters::TOP_P);
  sampler_params.set_k(1);
  sampler_params.set_p(0.0f);
  sampler_params.set_temperature(1.0f);
  sampler_params.set_seed(0);
  ASSIGN_OR_RETURN(
      sampler_,
      CreateSampler(sampler_backend, batch_size, std::move(sampler_params),
                    env_.Get(), vocab_size, logits_data_type_));
  return absl::OkStatus();
}

absl::Status LlmLiteRtCompiledModelExecutor::SampleLogits(
    const TensorBuffer& logits, TensorBuffer& ids_tensor) {
  if (sampler_ == nullptr) {
    LITERT_ASSIGN_OR_RETURN(auto decoded_logits_tensor_type,
                            logits.TensorType());
    RETURN_IF_ERROR(InitializeSampler(
        /*batch_size=*/decoded_logits_tensor_type.Layout().Dimensions()[0]));
  }

  RETURN_IF_ERROR(sampler_->SampleToIdAndScoreBuffer(
      logits, ids_tensor, /*scores_tensor=*/nullptr));
  return absl::OkStatus();
}

absl::Status LlmLiteRtCompiledModelExecutor::Reset() {
  current_step_ = 0;
  RETURN_IF_ERROR(processed_tokens_.RollBackToStep(0));
  sampler_.reset();
  return absl::OkStatus();
}

absl::StatusOr<int> LlmLiteRtCompiledModelExecutor::GetVocabSize() {
  if (!decode_output_buffers_.contains(signatures_.output_logits)) {
    return absl::NotFoundError("Output logits info not found.");
  }

  LITERT_ASSIGN_OR_RETURN(
      auto logits_tensor_type,
      decode_output_buffers_[signatures_.output_logits].TensorType());
  RET_CHECK_EQ(logits_tensor_type.Layout().Dimensions().size(), 3);
  return logits_tensor_type.Layout().Dimensions()[2];
}

// static
// Creates a LlmLiteRtCompiledModelExecutor from a LiteRt model.
absl::StatusOr<std::unique_ptr<LlmLiteRtCompiledModelExecutor>>
LlmLiteRtCompiledModelExecutor::Create(LlmExecutorSettings executor_settings,
                                       Environment& lrt_env,
                                       ModelResources& resources) {
  ASSIGN_OR_RETURN(auto litert_model,
                   resources.GetTFLiteModel(ModelType::kTfLitePrefillDecode));
  // For the LlmLiteRtCompiledModelExecutor, ML_DRIFT backend is used by
  // default.
  // TODO(b/405424188): - Add support for NPU backends.
  auto compilation_options = ::litert::Options::Create();
  std::string weight_cache_path = executor_settings.GetCacheDir();
  auto activation_data_type = ActivationDataType::FLOAT16;
  // TODO(b/433590109): Some GPUs do not support FP16, so we need to check the
  // capabilities of the GPU and set the activation data type accordingly.
  if (executor_settings.GetActivationDataType().has_value()) {
    activation_data_type = executor_settings.GetActivationDataType().value();
  }
  const Backend backend = executor_settings.GetBackend();
  switch (backend) {
    case Backend::GPU: {
      // TODO: b/403132820 - Add accelerator compilation options for ML_DRIFT.
      LITERT_ASSIGN_OR_RETURN(auto gpu_compilation_options,
                              GpuOptions::Create());
      gpu_compilation_options.EnableConstantTensorSharing(true);
      gpu_compilation_options.EnableInfiniteFloatCapping(true);
      gpu_compilation_options.EnableAllowSrcQuantizedFcConvOps(true);
      if (activation_data_type == ActivationDataType::FLOAT32) {
        gpu_compilation_options.SetDelegatePrecision(
            LiteRtDelegatePrecision::kLiteRtDelegatePrecisionFp32);
      } else {
        gpu_compilation_options.SetDelegatePrecision(
            LiteRtDelegatePrecision::kLiteRtDelegatePrecisionFp16);
      }
      gpu_compilation_options.SetPreferTextureWeights(true);
      if (weight_cache_path != ":nocache") {
        ASSIGN_OR_RETURN(auto model_path,
                         executor_settings.GetModelAssets().GetPath());
        if (weight_cache_path.empty()) {
          weight_cache_path = Dirname(model_path);
        }
        gpu_compilation_options.SetSerializationDir(weight_cache_path.c_str());
        absl::string_view model_name = Basename(model_path);
        gpu_compilation_options.SetModelCacheKey(model_name.data());
        gpu_compilation_options.SetSerializeProgramCache(true);
        gpu_compilation_options.SetSerializeExternalTensors(true);
      }
      // Use NoExternalTensorsMode to get better performance.
      bool external_tensor_mode =
          executor_settings.GetBackendConfig<GpuConfig>()->external_tensor_mode;
      gpu_compilation_options.EnableExternalTensorsMode(external_tensor_mode);
      if (!external_tensor_mode) {
        // This option prevents KVCache handling from being affected by
        // BHWC conversion in NoExternalTensorsMode.
        gpu_compilation_options.AddExternalTensorPattern("kv_cache_");
        ASSIGN_OR_RETURN(auto sampler_backend,
                         GetSamplerBackend(executor_settings));
        if (sampler_backend == Backend::GPU) {
          // GPU Sampler requires logits to be external tensors (PHWC4 format).
          gpu_compilation_options.AddExternalTensorPattern("logits");
        }
      }
      auto advanced_settings = executor_settings.GetAdvancedSettings();
      if (advanced_settings) {
        gpu_compilation_options.SetMadviseOriginalSharedTensors(
            advanced_settings->gpu_madvise_original_shared_tensors);
      }
      // TODO b/441627719 - Select backend by runtime options.
#if defined(LITERT_USE_WEBGPU_ACCELERATOR)
      gpu_compilation_options.SetGpuBackend(kLiteRtGpuBackendWebGpu);
#endif  // defined(LITERT_USE_WEBGPU_ACCELERATOR)
      compilation_options->AddOpaqueOptions(std::move(gpu_compilation_options));
      compilation_options->SetHardwareAccelerators(kLiteRtHwAcceleratorGpu);
      break;
    }
    case Backend::CPU: {
      Expected<CpuOptions> cpu_compilation_options = CpuOptions::Create();
      const uint32_t num_threads =
          executor_settings.GetBackendConfig<CpuConfig>()->number_of_threads;
      cpu_compilation_options->SetNumThreads(num_threads);
      auto weight_cache_file =
          executor_settings.GetWeightCacheFile(".xnnpack_cache");
      if (weight_cache_file.ok()) {
        if (std::holds_alternative<std::string>(*weight_cache_file)) {
          weight_cache_path = std::get<std::string>(*weight_cache_file);
          cpu_compilation_options->SetXNNPackWeightCachePath(
              weight_cache_path.c_str());
        } else {
          auto scoped_cache_file =
              std::get<std::shared_ptr<ScopedFile>>(*weight_cache_file);
          ASSIGN_OR_RETURN(auto duplicated, scoped_cache_file->Duplicate());
          ASSIGN_OR_RETURN(int fd, duplicated.Release());
          cpu_compilation_options->SetXNNPackWeightCacheFileDescriptor(fd);
        }
      }
      LITERT_ASSIGN_OR_RETURN(const uint32_t default_xnnpack_flags,
                              cpu_compilation_options->GetXNNPackFlags());
      cpu_compilation_options->SetXNNPackFlags(
          default_xnnpack_flags |
          TFLITE_XNNPACK_DELEGATE_FLAG_ENABLE_LATEST_OPERATORS |
          TFLITE_XNNPACK_DELEGATE_FLAG_ENABLE_SUBGRAPH_RESHAPING);
      LITERT_ASSIGN_OR_RETURN(auto runtime_options, RuntimeOptions::Create());
      runtime_options.SetShloCompositeInlining(true);
      compilation_options->AddOpaqueOptions(std::move(runtime_options));
      compilation_options->AddOpaqueOptions(
          std::move(*cpu_compilation_options));
      compilation_options->SetHardwareAccelerators(kLiteRtHwAcceleratorCpu);
      break;
    }
    default:
      return absl::InvalidArgumentError(absl::StrCat(
          "Unsupported backend: ", executor_settings.GetBackend()));
  }

  if (!litert_model || !*litert_model) {
    return absl::InternalError("Failed to build LiteRt model");
  }

  LITERT_ASSIGN_OR_RETURN(
      auto compiled_model,
      CompiledModel::Create(lrt_env, *litert_model,
                            std::move(*compilation_options)));

  absl::flat_hash_map<absl::string_view, TensorBuffer> decode_input_buffers;
  absl::flat_hash_map<absl::string_view, TensorBuffer> decode_output_buffers;
  absl::flat_hash_map<absl::string_view, TensorBuffer> input_kv_cache_buffers;
  absl::flat_hash_map<absl::string_view, TensorBuffer> output_kv_cache_buffers;

  absl::string_view prefill_signature_key = "";
  for (int i = 0; i < litert_model->GetNumSignatures(); ++i) {
    LITERT_ASSIGN_OR_RETURN(auto sig, litert_model->GetSignature(i));
    absl::string_view key = sig.Key();
    if (absl::StartsWith(key, kPrefillSignatureRunner)) {
      prefill_signature_key = key;
      break;
    }
  }
  LITERT_ASSIGN_OR_RETURN(auto prefill_signature,
                          litert_model->FindSignature(prefill_signature_key));
  std::string kv_cache_k_root_name;
  std::string kv_cache_v_root_name;
  RETURN_IF_ERROR(GetCacheRootNames(prefill_signature.InputNames(),
                                    kv_cache_k_root_name,
                                    kv_cache_v_root_name));
  LITERT_ASSIGN_OR_RETURN(auto decode_signature,
                          litert_model->FindSignature(kDecodeSignatureRunner));
  ASSIGN_OR_RETURN(
      ModelSignatures signatures,
      GetModelSignaturesFromInputOutputNames(decode_signature.InputNames(),
                                             decode_signature.OutputNames()));

  for (auto input_name : prefill_signature.InputNames()) {
    // Skip creating buffers for the input tokens, positions and attn mask. Move
    // into prefill function to create them based on the ids size.
    if (!absl::StartsWith(input_name, kv_cache_k_root_name) &&
        !absl::StartsWith(input_name, kv_cache_v_root_name)) {
      continue;
    }
    LITERT_ASSIGN_OR_RETURN(
        auto input_buffer,
        compiled_model.CreateInputBuffer(prefill_signature_key, input_name));
    if (backend == Backend::CPU) {
      LITERT_ASSIGN_OR_RETURN(auto output_buffer, input_buffer.Duplicate());
      output_kv_cache_buffers[input_name] = std::move(output_buffer);
    }
    input_kv_cache_buffers[input_name] = std::move(input_buffer);
    const auto& settings = executor_settings.GetAdvancedSettings();
    if (settings && settings->clear_kv_cache_before_prefill) {
      auto kv_cache_span =
          ReferTensorBufferAsSpan<float>(input_kv_cache_buffers[input_name]);
      if (kv_cache_span) {
        ABSL_LOG(INFO) << "Clearing kv cache: " << input_name;
        for (float& v : *kv_cache_span) v = 0.0f;
      }
    }
  }
  for (auto output_name : prefill_signature.OutputNames()) {
    LITERT_ASSIGN_OR_RETURN(
        auto output_buffer,
        compiled_model.CreateOutputBuffer(prefill_signature_key, output_name));
    if (absl::StartsWith(output_name, kv_cache_k_root_name) ||
        absl::StartsWith(output_name, kv_cache_v_root_name)) {
      if (backend == Backend::GPU) {
        output_kv_cache_buffers[output_name] = std::move(output_buffer);
      }
      // For CPU, we will use single buffer for kv cache input and output to
      // improve performance and memory usage.
    } else {
      // TODO b/444063139 - Support non-kv_cache tensors as prefill outputs.
      // This should be done once we have a model that has non-kv_cache tensors
      // as prefill outputs. It should be done in the same place as the prefill
      // inputs are created.
      return absl::UnimplementedError(absl::StrCat(
          "Failed to create prefill output buffer for '", output_name,
          "'. Only kv_cache tensors are supported as outputs to "
          "prefill at the moment."));
    }
  }

  for (auto input_name : decode_signature.InputNames()) {
    if (IsLoRAInputName(input_name)) {
      // We let LoraManager handle LoRA inputs.
      continue;
    }
    if (!absl::StartsWith(input_name, kv_cache_k_root_name) &&
        !absl::StartsWith(input_name, kv_cache_v_root_name)) {
      LITERT_ASSIGN_OR_RETURN(
          auto input_buffer,
          compiled_model.CreateInputBuffer(kDecodeSignatureRunner, input_name));
      decode_input_buffers[input_name] = std::move(input_buffer);
    }
  }
  for (auto output_name : decode_signature.OutputNames()) {
    if (!absl::StartsWith(output_name, kv_cache_k_root_name) &&
        !absl::StartsWith(output_name, kv_cache_v_root_name)) {
      LITERT_ASSIGN_OR_RETURN(auto output_buffer,
                              compiled_model.CreateOutputBuffer(
                                  kDecodeSignatureRunner, output_name));
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
  RET_CHECK(output_logits_buffer_tensor_type.Layout().Dimensions()[0] == 1)
      << "Only support batch size 1 for now.";

  ASSIGN_OR_RETURN(auto prefill_runner_set,
                   GetPrefillRunnerSetFromModel(
                       *litert_model, kPrefillSignatureRunner,
                       /*input_positions_name=*/signatures.input_positions));
  RET_CHECK(!prefill_runner_set.empty()) << "No prefill runner available.";

  std::unique_ptr<EmbeddingLookupManager> embedding_lookup;
  std::unique_ptr<EmbeddingLookupManager> per_layer_embedding_lookup;
  RETURN_IF_ERROR(InitializeEmbeddingLookups(resources, embedding_lookup,
                                             per_layer_embedding_lookup));
  return absl::WrapUnique(new LlmLiteRtCompiledModelExecutor(
      std::move(executor_settings), lrt_env, litert_model,
      std::move(compiled_model), std::move(decode_input_buffers),
      std::move(decode_output_buffers), std::move(input_kv_cache_buffers),
      std::move(output_kv_cache_buffers), std::move(prefill_runner_set),
      signatures, std::move(weight_cache_path),
      std::move(embedding_lookup), std::move(per_layer_embedding_lookup),
      activation_data_type));
}

}  // namespace litert::lm
