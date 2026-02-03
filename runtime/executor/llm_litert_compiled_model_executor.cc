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
#include <filesystem>  // NOLINT(build/c++17) for std::filesystem::path
#include <memory>
#include <optional>
#include <random>
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
#include "litert/cc/litert_tensor_buffer_types.h"  // from @litert
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
#include "runtime/executor/llm_litert_compiled_model_cache_utils.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/file_util.h"
#include "runtime/util/lora_util.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/status_macros.h"  // IWYU pragma: keep
#include "tflite/delegates/xnnpack/xnnpack_delegate.h"  // from @litert

namespace litert::lm {
namespace {

using ::absl::Span;
using ::litert::GpuOptions;
using ::litert::TensorBuffer;

// Names of the signature runners, used to get the signature runners from the
// interpreter.
constexpr absl::string_view kPrefillSignatureRunner = "prefill";
constexpr absl::string_view kDecodeSignatureRunner = "decode";
constexpr int kDynamicDimValue = -1;

// Default number of threads for WebGPU weight upload and kernel compilation.
constexpr int kDefaultNumThreadsToUpload = 2;
constexpr int kDefaultNumThreadsToCompile = 1;

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

absl::Status CopyKvCacheBuffers(
    size_t decode_batch_size, int src_index_to_copy_on_prefill,
    const absl::flat_hash_map<absl::string_view, TensorBuffer>&
        src_kv_cache_buffers,
    const absl::flat_hash_map<absl::string_view, TensorBuffer>&
        dst_kv_cache_buffers) {
  for (const auto& [name, src_buffer] : src_kv_cache_buffers) {
    if (!dst_kv_cache_buffers.contains(name)) {
      return absl::FailedPreconditionError(
          absl::StrCat("KV cache buffer ", name, " not found."));
    }
    const auto& dst_buffer = dst_kv_cache_buffers.at(name);
    LITERT_ASSIGN_OR_RETURN(auto src_buffer_lock_and_addr,
                            TensorBufferScopedLock::Create(
                                src_buffer, TensorBuffer::LockMode::kRead));
    LITERT_ASSIGN_OR_RETURN(size_t src_buffer_size, src_buffer.PackedSize());
    const char* src_buffer_ptr =
        static_cast<const char*>(src_buffer_lock_and_addr.second);

    LITERT_ASSIGN_OR_RETURN(auto dst_buffer_lock_and_addr,
                            TensorBufferScopedLock::Create(
                                dst_buffer, TensorBuffer::LockMode::kWrite));
    LITERT_ASSIGN_OR_RETURN(size_t dst_buffer_size, dst_buffer.PackedSize());
    char* dst_buffer_ptr =
        static_cast<char*>(const_cast<void*>(dst_buffer_lock_and_addr.second));
    // This copy is based on the assumption that the KV cache buffers are in the
    // layout of [batch * X, ...] or [1, batch * X, ...] where X could be 1 or
    // more and X doesn't make values interleaved across batches which is true
    // for the current LLM models of all backends.
    if (src_index_to_copy_on_prefill >= 0) {
      // This is the case of the first prefill after decode. It reduces the KV
      // cache size to one by copying only the cache content of the given index.
      RET_CHECK_EQ(src_buffer_size, dst_buffer_size * decode_batch_size);
      RET_CHECK_LT(src_index_to_copy_on_prefill, decode_batch_size);
      src_buffer_ptr += src_index_to_copy_on_prefill * dst_buffer_size;
      memcpy(dst_buffer_ptr, src_buffer_ptr, dst_buffer_size);
    } else {
      // This is the case of the first decode after prefill. It broadcasts the
      // KV cache contents to all the batches.
      RET_CHECK_EQ(src_buffer_size * decode_batch_size, dst_buffer_size);
      for (int i = 0; i < decode_batch_size; ++i) {
        memcpy(dst_buffer_ptr, src_buffer_ptr, src_buffer_size);
        dst_buffer_ptr += src_buffer_size;
      }
    }
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

absl::StatusOr<int> GetDynamicDimIndex(const Model& model,
                                       absl::string_view signature,
                                       absl::string_view tensor_name) {
  LITERT_ASSIGN_OR_RETURN(const SimpleSignature& sig,
                          model.FindSignature(signature));
  LITERT_ASSIGN_OR_RETURN(const SimpleTensor& tensor,
                          sig.InputTensor(tensor_name));
  LITERT_ASSIGN_OR_RETURN(const RankedTensorType ranked_tensor_type,
                          tensor.RankedTensorType());
  auto dimensions = ranked_tensor_type.Layout().Dimensions();
  for (int i = 0; i < dimensions.size(); ++i) {
    if (dimensions[i] == kDynamicDimValue) {
      return i;
    }
  }
  return absl::InvalidArgumentError("No dynamic dimension found.");
}

absl::StatusOr<bool> HasDynamicDim(const Model& model,
                                   absl::string_view signature,
                                   absl::string_view tensor_name) {
  LITERT_ASSIGN_OR_RETURN(const SimpleSignature& sig,
                          model.FindSignature(signature));
  LITERT_ASSIGN_OR_RETURN(const SimpleTensor& tensor,
                          sig.InputTensor(tensor_name));
  LITERT_ASSIGN_OR_RETURN(const RankedTensorType ranked_tensor_type,
                          tensor.RankedTensorType());
  auto dimensions = ranked_tensor_type.Layout().Dimensions();
  for (int i = 0; i < dimensions.size(); ++i) {
    if (dimensions[i] == kDynamicDimValue) {
      return true;
    }
  }
  return false;
}

absl::Status ResolveDynamicShape(const Model& model,
                                 CompiledModel& compiled_model,
                                 absl::string_view signature,
                                 absl::string_view tensor_name, int new_value) {
  LITERT_ASSIGN_OR_RETURN(const SimpleSignature& sig,
                          model.FindSignature(signature));
  LITERT_ASSIGN_OR_RETURN(const SimpleTensor& tensor,
                          sig.InputTensor(tensor_name));
  LITERT_ASSIGN_OR_RETURN(const RankedTensorType ranked_tensor_type,
                          tensor.RankedTensorType());
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

absl::StatusOr<TensorBuffer> ResizeKVCacheTensorBuffer(
    Environment& env, TensorBuffer& tensor_buffer, int dynamic_dim_index,
    int num_entries_to_insert) {
  LITERT_ASSIGN_OR_RETURN(const RankedTensorType& tensor_type,
                          tensor_buffer.TensorType());
  RET_CHECK(!tensor_type.Layout().HasStrides());
  auto dimensions = tensor_type.Layout().Dimensions();
  std::vector<int> new_dimensions;
  new_dimensions.reserve(dimensions.size());
  for (int i = 0; i < dimensions.size(); ++i) {
    if (i == dynamic_dim_index) {
      new_dimensions.push_back(dimensions[i] + num_entries_to_insert);
    } else {
      new_dimensions.push_back(dimensions[i]);
    }
  }

  LITERT_ASSIGN_OR_RETURN(litert::TensorBufferType buffer_type,
                          tensor_buffer.BufferType());
  Layout new_layout(Dimensions(new_dimensions.begin(), new_dimensions.end()));
  auto new_out_type =
      RankedTensorType(tensor_type.ElementType(), std::move(new_layout));
  LITERT_ASSIGN_OR_RETURN(size_t new_size, new_out_type.Bytes());

  LITERT_ASSIGN_OR_RETURN(
      TensorBuffer new_tensor_buffer,
      TensorBuffer::CreateManaged(env, buffer_type, new_out_type, new_size));

  LITERT_ASSIGN_OR_RETURN(auto tensor_buffer_lock_and_addr,
                          TensorBufferScopedLock::Create(
                              tensor_buffer, TensorBuffer::LockMode::kRead));
  auto* tensor_buffer_ptr =
      static_cast<uint8_t*>(tensor_buffer_lock_and_addr.second);
  LITERT_ASSIGN_OR_RETURN(
      auto new_tensor_buffer_lock_and_addr,
      TensorBufferScopedLock::Create(new_tensor_buffer,
                                     TensorBuffer::LockMode::kWrite));
  auto* new_tensor_buffer_ptr =
      static_cast<uint8_t*>(new_tensor_buffer_lock_and_addr.second);
  std::optional<size_t> element_size = GetByteWidth(tensor_type.ElementType());
  RET_CHECK(element_size.has_value());

  RETURN_IF_ERROR(ExpandBuffer(tensor_buffer_ptr, dimensions,
                               new_tensor_buffer_ptr, new_dimensions,
                               element_size.value()));

  return new_tensor_buffer;
}

absl::Status CopyBuffer(const TensorBuffer& buffers_from,
                        TensorBuffer& buffers_to) {
  // TODO: b/452977992: For GPU, we could use a shader to copy the buffer. If we
  // were to do it this way for GPU, then it might make more sense just to keep
  // the copy on the host. Also for GPU, consider optionally keeping its buffer
  // copies in CPU memory to save on GPU memory.
  LITERT_ASSIGN_OR_RETURN(auto read_lock,
                          ::litert::TensorBufferScopedLock::Create(
                              buffers_from, TensorBuffer::LockMode::kRead));
  LITERT_ASSIGN_OR_RETURN(auto write_lock,
                          ::litert::TensorBufferScopedLock::Create(
                              buffers_to, TensorBuffer::LockMode::kWrite));

  LITERT_ASSIGN_OR_RETURN(auto buffer_size, buffers_from.PackedSize());
  memcpy(write_lock.second, read_lock.second, buffer_size);
  return absl::OkStatus();
}

}  // namespace

absl::Status LlmLiteRtCompiledModelExecutorBase::CreatePrefillInputBuffers(
    absl::string_view prefill_signature, int sequence_length,
    int context_length,
    absl::flat_hash_map<absl::string_view, TensorBuffer>&
        prefill_input_buffers) {
  auto dyn_shape_resolver = [&](absl::string_view tensor_name) -> absl::Status {
    return ResolveDynamicShape(model_, compiled_model_, prefill_signature,
                               tensor_name, sequence_length);
  };
  // Create input_token, positions and attn_mask buffers after determining
  // the prefill length.
  if (!signatures_.input_tokens.empty()) {
    RETURN_IF_ERROR(dyn_shape_resolver(signatures_.input_tokens));
    auto tokens_buffer = compiled_model_.CreateInputBuffer(
        prefill_signature, signatures_.input_tokens);
    prefill_input_buffers[signatures_.input_tokens] = std::move(*tokens_buffer);
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
    RETURN_IF_ERROR(dyn_shape_resolver(signatures_.input_embeddings.value()));
    auto embeddings_buffer = compiled_model_.CreateInputBuffer(
        prefill_signature, signatures_.input_embeddings.value());
    prefill_input_buffers[signatures_.input_embeddings.value()] =
        std::move(*embeddings_buffer);

    // We may have per layer embedding as well.
    if (signatures_.input_per_layer_embeddings.has_value()) {
      if (embedding_lookup_ == nullptr) {
        return absl::FailedPreconditionError(
            "Input per layer embeddings required by signature but "
            "embedding lookup model is not initialized.");
      }
      RETURN_IF_ERROR(
          dyn_shape_resolver(signatures_.input_per_layer_embeddings.value()));
      auto per_layer_embeddings_buffer = compiled_model_.CreateInputBuffer(
          prefill_signature, signatures_.input_per_layer_embeddings.value());
      prefill_input_buffers[signatures_.input_per_layer_embeddings.value()] =
          std::move(*per_layer_embeddings_buffer);
    }
  }
  RETURN_IF_ERROR(dyn_shape_resolver(signatures_.input_positions));
  auto positions_buffer = compiled_model_.CreateInputBuffer(
      prefill_signature, signatures_.input_positions);
  prefill_input_buffers[signatures_.input_positions] =
      std::move(*positions_buffer);

  if (signatures_.input_attn_mask.has_value()) {
    ASSIGN_OR_RETURN(bool is_attn_dyn,
                     HasDynamicDim(model_, prefill_signature,
                                   signatures_.input_attn_mask.value()));
    if (is_attn_dyn) {
      std::vector<int> new_shape = {1, 1, sequence_length, context_length};
      LITERT_RETURN_IF_ERROR(compiled_model_.ResizeInputTensor(
          prefill_signature, signatures_.input_attn_mask.value(), new_shape));
    }

    auto attn_mask_buffer = compiled_model_.CreateInputBuffer(
        prefill_signature, signatures_.input_attn_mask.value());
    prefill_input_buffers[signatures_.input_attn_mask.value()] =
        std::move(*attn_mask_buffer);
  }
  return absl::OkStatus();
}

absl::Status LlmLiteRtCompiledModelExecutorBase::FillInputBufferWithToken(
    const std::vector<std::shared_ptr<TokenData>>& unprocessed_token,
    ::litert::TensorBuffer& input_buffer, bool is_per_layer_embedding) {
  if (unprocessed_token.empty()) {
    return absl::InvalidArgumentError("Unprocessed token is null.");
  }

  LITERT_ASSIGN_OR_RETURN(auto input_buffer_lock_and_addr,
                          ::litert::TensorBufferScopedLock::Create(
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
  if (llm_context_->runtime_state().current_step ==
      llm_context_->processed_context().processed_tokens().TokenCount()) {
    return absl::OkStatus();
  }
  if (llm_context_->runtime_state().current_step == 0) {
    RETURN_IF_ERROR(
        llm_context_->processed_context().processed_tokens().RollBackToStep(0));
  } else {
    auto step_and_token =
        llm_context_->processed_context().processed_tokens().GetTokenAtStep(
            llm_context_->runtime_state().current_step - 1);
    RETURN_IF_ERROR(
        llm_context_->processed_context().processed_tokens().RollBackToStep(
            llm_context_->runtime_state().current_step - 1));
    llm_context_->processed_context().processed_tokens().AddProcessedTokens(
        {step_and_token});
  }

  // Reset sampler input handling as the step is rolled back.
  if (sampler_ != nullptr && sampler_->HandlesInput()) {
    RETURN_IF_ERROR(SetSamplerInputHandling(/*reset=*/true));
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
    LITERT_RETURN_IF_ERROR(
        CopyKvCacheBuffers(output_heads, token_index_to_reduce,
                           *input_kv_cache_buffers_, kv_cache_buffers_1_));
    input_kv_cache_buffers_ = &kv_cache_buffers_1_;
    output_kv_cache_buffers_ = &kv_cache_buffers_2_;
  }

  // Reset sampler input handling if it handles input for next decode.
  if (sampler_ != nullptr && sampler_->HandlesInput()) {
    RETURN_IF_ERROR(SetSamplerInputHandling(/*reset=*/true));
  }

  return absl::OkStatus();
}

absl::Status LlmLiteRtCompiledModelExecutorBase::PrefillInternal(
    absl::string_view prefill_signature,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        prefill_input_buffers,
    Span<const int> ids, bool async) {
  RETURN_IF_ERROR(RollBackProcessedTokens());

  {
    // Fill the input buffers with scoped locks.
    auto& prefill_input_pos =
        prefill_input_buffers[signatures_.input_positions];
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
          prefill_input_buffers[signatures_.input_attn_mask.value()],
          use_fp16_precision_));
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
    // return early by storing the token as a pending input token.
    if (!has_pending_input_token && prefill_length == 0) {
      RETURN_IF_ERROR(
          llm_context_->processed_context()
              .processed_tokens()
              .AddPendingInputToken({std::make_shared<TokenData>(ids[0])}));
      ++llm_context_->runtime_state().current_step;
      return absl::OkStatus();
    }
    int input_idx = 0;
    if (has_pending_input_token) {
      if (use_token_as_lookup) {
        RETURN_IF_ERROR(FillInputBufferWithToken(
            pending_input_token,
            prefill_input_buffers[signatures_.input_tokens]));
      } else {
        RETURN_IF_ERROR(FillInputBufferWithToken(
            pending_input_token,
            prefill_input_buffers[signatures_.input_embeddings.value()]));
        if (use_per_layer_embedding) {
          RETURN_IF_ERROR(FillInputBufferWithToken(
              pending_input_token,
              prefill_input_buffers[signatures_.input_per_layer_embeddings
                                        .value()],
              /*is_per_layer_embedding=*/true));
        }
      }
      prefill_input_pos_ptr[input_idx] = internal_start_step;
      RETURN_IF_ERROR(llm_context_->processed_context()
                          .processed_tokens()
                          .MarkPendingInputTokenAsProcessed());
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
          &(prefill_input_buffers[signatures_.input_embeddings.value()]);
      RETURN_IF_ERROR(embedding_lookup_->LookupPrefill(
          processed_input_tokens, prefill_input_embeddings_buffer,
          /*offset=*/input_idx));

      // We may have per layer embedding as well.
      if (signatures_.input_per_layer_embeddings) {
        TensorBuffer* prefill_input_per_layer_embeddings_buffer =
            &(prefill_input_buffers[signatures_.input_per_layer_embeddings
                                        .value()]);
        RETURN_IF_ERROR(per_layer_embedding_lookup_->LookupPrefill(
            processed_input_tokens, prefill_input_per_layer_embeddings_buffer,
            /*offset=*/input_idx));
      }
    }
    if (signatures_.input_attn_mask.has_value()) {
      RETURN_IF_ERROR(FillAttentionMask(
          prefill_input_buffers[signatures_.input_attn_mask.value()],
          start_step,
          /*steps=*/prefill_length + input_idx));
    }

    // Add the last token of the current input as a pending input token, to be
    // used in the next prefill or decode.
    auto last_input_token = std::make_shared<TokenData>(ids.back());
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
    RETURN_IF_ERROR(llm_context_->processed_context()
                        .processed_tokens()
                        .AddPendingInputToken({std::move(last_input_token)}));
    ++llm_context_->runtime_state().current_step;
  }

  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer> input_buffers;
  for (const auto& [input_name, input_buffer] : prefill_input_buffers) {
    LITERT_ASSIGN_OR_RETURN(auto input_buffer_dup, input_buffer.Duplicate());
    input_buffers[input_name] = std::move(input_buffer_dup);
  }
  for (const auto& [input_name, input_buffer] : *input_kv_cache_buffers_) {
    LITERT_ASSIGN_OR_RETURN(auto input_buffer_dup, input_buffer.Duplicate());
    input_buffers[input_name] = std::move(input_buffer_dup);
  }
  // Inject LoRA buffers for the active adapter, if any.
  if (lora_manager_ != nullptr &&
      lora_manager_->GetCurrentLoRAId().has_value()) {
    auto lora_buffers = lora_manager_->GetLoRABuffers();
    if (lora_buffers.ok()) {
      for (auto& [name, buffer] : *lora_buffers) {
        input_buffers[name] = std::move(buffer);
      }
    } else {
      ABSL_LOG(WARNING) << "Failed to get LoRA buffers for prefill: "
                        << lora_buffers.status();
    }
  }
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer> output_buffers;
  for (const auto& [output_name, output_buffer] : *output_kv_cache_buffers_) {
    LITERT_ASSIGN_OR_RETURN(auto output_buffer_dup, output_buffer.Duplicate());
    output_buffer_dup.ClearEvent();
    output_buffers[output_name] = std::move(output_buffer_dup);
  }

  if (async) {
    LITERT_RETURN_IF_ERROR(compiled_model_.RunAsync(
        prefill_signature, input_buffers, output_buffers, async));
  } else {
    LITERT_RETURN_IF_ERROR(
        compiled_model_.Run(prefill_signature, input_buffers, output_buffers));
  }

  if (!gpu_optimized_single_buffer_cache_) {
    std::swap(input_kv_cache_buffers_, output_kv_cache_buffers_);
  }
  return absl::OkStatus();
}

absl::StatusOr<ProcessedTokens::StepAndToken>
LlmLiteRtCompiledModelExecutorBase::GetTokenToDecode(
    const ExecutorInputs& inputs) {
  RETURN_IF_ERROR(RollBackProcessedTokens());

  if (inputs.GetTextDataPtr().ok()) {
    auto input_tensor_size = (*inputs.GetTextTokenIdsPtr())->PackedSize();
    if (input_tensor_size && *input_tensor_size != 0) {
      int output_heads = 1;
      if (llm_context_->runtime_config().output_heads.has_value()) {
        output_heads = llm_context_->runtime_config().output_heads.value();
      }
      // Input token ids provided, so use it regardless of whether next input
      // token id is set.
      RET_CHECK_EQ(*input_tensor_size, output_heads * sizeof(int32_t));
      LITERT_ASSIGN_OR_RETURN(auto ids, ReferTensorBufferAsSpan<int32_t>(
                                            *(*inputs.GetTextTokenIdsPtr())));
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
        RETURN_IF_ERROR(llm_context_->processed_context()
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
      RETURN_IF_ERROR(embedding_lookup_->LookupDecode(
          token->id(), token->mutable_embedding()));
      if (signatures_.input_per_layer_embeddings.has_value()) {
        RETURN_IF_ERROR(per_layer_embedding_lookup_->LookupDecode(
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
    RETURN_IF_ERROR(InitializeAttentionMask(
        decode_input_buffers_[signatures_.input_attn_mask.value()],
        use_fp16_precision_));
    RETURN_IF_ERROR(FillAttentionMask(
        decode_input_buffers_[signatures_.input_attn_mask.value()], step,
        /*steps=*/1));
  }

  return BindTensorsAndRunDecode(decode_input_buffers_, &output_logits);
}

absl::Status LlmLiteRtCompiledModelExecutorBase::BindTensorsAndRunDecode(
    const absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        input_buffers,
    TensorBuffer* output_logits) {
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      decode_input_buffers;
  for (const auto& [input_name, input_buffer] : input_buffers) {
    LITERT_ASSIGN_OR_RETURN(auto input_buffer_dup, input_buffer.Duplicate());
    decode_input_buffers[input_name] = std::move(input_buffer_dup);
  }
  for (const auto& [input_name, input_buffer] : *input_kv_cache_buffers_) {
    LITERT_ASSIGN_OR_RETURN(auto input_buffer_dup, input_buffer.Duplicate());
    decode_input_buffers[input_name] = std::move(input_buffer_dup);
  }
  // Inject LoRA buffers for the active adapter, if any.
  if (lora_manager_ != nullptr &&
      lora_manager_->GetCurrentLoRAId().has_value()) {
    auto lora_buffers = lora_manager_->GetLoRABuffers();
    if (lora_buffers.ok()) {
      for (auto& [name, buffer] : *lora_buffers) {
        decode_input_buffers[name] = std::move(buffer);
      }
    } else {
      ABSL_LOG(WARNING) << "Failed to get LoRA buffers for decode: "
                        << lora_buffers.status();
    }
  }
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      decode_output_buffers;
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
  for (const auto& [output_name, output_buffer] : *output_kv_cache_buffers_) {
    LITERT_ASSIGN_OR_RETURN(auto output_buffer_dup, output_buffer.Duplicate());
    output_buffer_dup.ClearEvent();
    decode_output_buffers[output_name] = std::move(output_buffer_dup);
  }

  bool async = true;
  LITERT_RETURN_IF_ERROR(
      compiled_model_.RunAsync(kDecodeSignatureRunner, decode_input_buffers,
                               decode_output_buffers, async));

  if (!gpu_optimized_single_buffer_cache_) {
    std::swap(input_kv_cache_buffers_, output_kv_cache_buffers_);
  }
  return absl::OkStatus();
}

int LlmLiteRtCompiledModelExecutorBase::BindTensorsAndRunDecodeStatic(
    void* arg) {
  auto self = static_cast<LlmLiteRtCompiledModelExecutorBase*>(arg);
  // Run decode with default output_logits.
  auto status = self->BindTensorsAndRunDecode(self->decode_input_buffers_,
                                              /*output_logits=*/nullptr);
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

  LITERT_RETURN_IF_ERROR(decode_kv_cache_buffers_1_.has_value());
  LITERT_RETURN_IF_ERROR(decode_kv_cache_buffers_2_.has_value());
  // Broadcast the prefill kv cache buffers to the decode kv cache buffers.
  // This is only needed when decode batch size > 1.
  LITERT_RETURN_IF_ERROR(CopyKvCacheBuffers(
      output_heads, /*src_index_to_copy_on_prefill=*/-1,
      *input_kv_cache_buffers_, *decode_kv_cache_buffers_1_));
  input_kv_cache_buffers_ = &decode_kv_cache_buffers_1_.value();
  output_kv_cache_buffers_ = &decode_kv_cache_buffers_2_.value();

  return absl::OkStatus();
}

absl::Status LlmLiteRtCompiledModelExecutorBase::Decode(
    ::litert::TensorBuffer& output_tokens) {
  return Decode(output_tokens, ExecutorDecodeParams());
}

absl::Status LlmLiteRtCompiledModelExecutorBase::Decode(
    ::litert::TensorBuffer& output_tokens,
    const ExecutorDecodeParams& decode_params) {

  ASSIGN_OR_RETURN(auto decoded_logits,
                   DecodeLogits(ExecutorInputs(), decode_params));
  RETURN_IF_ERROR(SampleLogits(decoded_logits, output_tokens));

  LITERT_ASSIGN_OR_RETURN(auto output_tokens_size, output_tokens.PackedSize());
  int output_heads = 1;
  if (llm_context_->runtime_config().output_heads.has_value()) {
    output_heads = llm_context_->runtime_config().output_heads.value();
  }
  RET_CHECK_EQ(output_tokens_size, output_heads * sizeof(int32_t));

  bool has_invalid_output_token = false;
  {
    std::vector<std::shared_ptr<TokenData>> tokens;
    tokens.reserve(output_heads);
    LITERT_ASSIGN_OR_RETURN(auto lock_and_addr,
                            ::litert::TensorBufferScopedLock::Create(
                                output_tokens, TensorBuffer::LockMode::kRead));
    auto output_token_span = absl::MakeConstSpan(
        static_cast<int32_t*>(lock_and_addr.second), output_heads);
    for (auto tid : output_token_span) {
      has_invalid_output_token |= tid < 0;
      tokens.push_back(std::make_shared<TokenData>(tid < 0 ? 0 : tid));
    }
    RETURN_IF_ERROR(llm_context_->processed_context()
                        .processed_tokens()
                        .AddPendingInputToken(tokens));
  }

  // Reset invalid token IDs if any.
  if (has_invalid_output_token) {
    ABSL_LOG(WARNING) << "Invalid decode and sample result. The sampled token "
                         "is casted to 0 to avoid crash.";
    LITERT_ASSIGN_OR_RETURN(
        auto lock_and_addr,
        ::litert::TensorBufferScopedLock::Create(
            output_tokens, TensorBuffer::LockMode::kReadWrite));
    auto output_token_span = absl::MakeSpan(
        static_cast<int32_t*>(lock_and_addr.second), output_heads);
    for (auto& tid : output_token_span) {
      if (tid < 0) {
        tid = 0;
      }
    }
  }
  return absl::OkStatus();
}

absl::Status LlmLiteRtCompiledModelExecutorBase::Decode(
    const ExecutorInputs& inputs, ::litert::TensorBuffer& output_logits) {
  RETURN_IF_ERROR(PrepareFirstDecode());
  ASSIGN_OR_RETURN(auto step_and_token, GetTokenToDecode(inputs));
  RETURN_IF_ERROR(DecodeInternal(step_and_token.token, output_logits));
  RETURN_IF_ERROR(ConsumePendingOrAddProcessedToken(step_and_token.token));
  ++llm_context_->runtime_state().current_step;
  return absl::OkStatus();
}

absl::StatusOr<::litert::TensorBuffer>
LlmLiteRtCompiledModelExecutorBase::DecodeLogits(const ExecutorInputs& inputs) {
  return DecodeLogits(inputs, ExecutorDecodeParams());
}

absl::StatusOr<::litert::TensorBuffer>
LlmLiteRtCompiledModelExecutorBase::DecodeLogits(
    const ExecutorInputs& inputs, const ExecutorDecodeParams& decode_params) {
  LITERT_ASSIGN_OR_RETURN(
      auto output_logits,
      decode_output_buffers_[signatures_.output_logits].Duplicate());

  bool last_run_is_decode = llm_context_->runtime_state().ran_decode;
  RETURN_IF_ERROR(PrepareFirstDecode());
  ASSIGN_OR_RETURN(auto step_and_token, GetTokenToDecode(inputs));
  RETURN_IF_ERROR(DecodeInternal(step_and_token.token, output_logits));
  RETURN_IF_ERROR(ConsumePendingOrAddProcessedToken(step_and_token.token));

  if (decode_params.HasConstraintDecoder() && !step_and_token.token.empty()) {
    int output_heads = 1;
    if (llm_context_->runtime_config().output_heads.has_value()) {
      output_heads = llm_context_->runtime_config().output_heads.value();
    }

    RET_CHECK_EQ(step_and_token.token.size(), output_heads);
    std::vector<int> current_token_ids;
    current_token_ids.reserve(output_heads);
    for (const auto& token : step_and_token.token) {
      current_token_ids.push_back(token->id());
    }
    // Update constraint state only with decode ids.
    if (last_run_is_decode) {
      RETURN_IF_ERROR(
          decode_params.GetConstraintDecoder()->UpdateConstraintState(
              absl::MakeSpan(current_token_ids)));
    }

    LITERT_ASSIGN_OR_RETURN(auto output_logits_buffer_type,
                            output_logits.BufferType());
    // If the output logits are already on the host memory, use the buffer
    // directly.
    if (output_logits_buffer_type == ::litert::TensorBufferType::kHostMemory) {
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

  ++llm_context_->runtime_state().current_step;

  const auto& settings = executor_settings_.GetAdvancedSettings();
  if (settings && settings->num_logits_to_print_after_decode > 0) {
    LogTensor(output_logits, settings->num_logits_to_print_after_decode,
              "Logits");
  }
  return output_logits;
}

absl::Status LlmLiteRtCompiledModelExecutorBase::InitializeSampler(
    std::optional<ActivationDataType> logits_data_type) {
  if (sampler_ != nullptr) {
    return absl::OkStatus();
  }

  // Use the provided activation data type if available, otherwise fallback to
  // the member variable.
  auto data_type = logits_data_type.value_or(logits_data_type_);

  ASSIGN_OR_RETURN(auto vocab_size, GetVocabSize());
  ASSIGN_OR_RETURN(auto sampler_backend, GetSamplerBackend(executor_settings_));
  int output_heads = 1;
  if (llm_context_->runtime_config().output_heads.has_value()) {
    output_heads = llm_context_->runtime_config().output_heads.value();
  }
  proto::SamplerParameters sampler_params;
  sampler_params.set_type(proto::SamplerParameters::TOP_P);
  sampler_params.set_k(1);
  sampler_params.set_p(0.0f);
  sampler_params.set_temperature(1.0f);
  sampler_params.set_seed(0);
  ASSIGN_OR_RETURN(sampler_, CreateSampler(sampler_backend, output_heads,
                                           std::move(sampler_params),
                                           env_.Get(), vocab_size, data_type));

  // If the sampler can handle input, prepare the input tensors for it.
  sampler_handles_input_ =
      (!executor_settings_.GetAdvancedSettings().has_value() ||
       executor_settings_.GetAdvancedSettings()->sampler_handles_input) &&
      sampler_->CanHandleInput() && !signatures_.input_tokens.empty();
  if (sampler_handles_input_) {
    ABSL_LOG(INFO) << "Sampler will handle decode input tensors.";
    if (!decode_prev_input_pos_) {
      LITERT_ASSIGN_OR_RETURN(
          decode_prev_input_pos_,
          compiled_model_.CreateInputBuffer(kDecodeSignatureRunner,
                                            signatures_.input_positions));
    }
    if (!decode_prev_mask_ && signatures_.input_attn_mask.has_value()) {
      LITERT_ASSIGN_OR_RETURN(
          decode_prev_mask_,
          compiled_model_.CreateInputBuffer(kDecodeSignatureRunner,
                                            *signatures_.input_attn_mask));
    }
    // Set, then reset the input handling to get the underlying model ready, but
    // not to bind the input tensors.
    RETURN_IF_ERROR(SetSamplerInputHandling(/*reset=*/false));
    RETURN_IF_ERROR(SetSamplerInputHandling(/*reset=*/true));
  }

  return absl::OkStatus();
}

absl::Status LlmLiteRtCompiledModelExecutorBase::SwapSamplerInputTensors() {
  bool has_input_attn_mask = signatures_.input_attn_mask.has_value();
  // Move the input_pos and mask to previous ones.
  std::swap(decode_prev_input_pos_,
            decode_input_buffers_[signatures_.input_positions]);
  if (has_input_attn_mask) {
    std::swap(decode_prev_mask_,
              decode_input_buffers_[*signatures_.input_attn_mask]);
  }
  return SetSamplerInputHandling(/*reset=*/false);
}

absl::Status LlmLiteRtCompiledModelExecutorBase::SetSamplerInputHandling(
    bool reset) {
  if (reset) {
    return sampler_->SetInputTensorsAndInferenceFunc(
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
  }

  bool has_input_attn_mask = signatures_.input_attn_mask.has_value();
  return sampler_->SetInputTensorsAndInferenceFunc(
      &decode_input_buffers_[signatures_.input_tokens], &decode_prev_input_pos_,
      &decode_input_buffers_[signatures_.input_positions],
      has_input_attn_mask ? &decode_prev_mask_ : nullptr,
      has_input_attn_mask ? &decode_input_buffers_[*signatures_.input_attn_mask]
                          : nullptr,
      BindTensorsAndRunDecodeStatic, this);
}

absl::Status LlmLiteRtCompiledModelExecutorBase::SampleLogits(
    const TensorBuffer& logits, TensorBuffer& ids_tensor) {
  if (sampler_ == nullptr) {
    RETURN_IF_ERROR(InitializeSampler(logits_data_type_));
  }

  if (sampler_handles_input_) {
    RETURN_IF_ERROR(SwapSamplerInputTensors());
  }

  RETURN_IF_ERROR(sampler_->SampleToIdAndScoreBuffer(
      logits, ids_tensor, /*scores_tensor=*/nullptr));
  return absl::OkStatus();
}

absl::Status LlmLiteRtCompiledModelExecutorBase::SetCurrentStep(int new_step) {
  ASSIGN_OR_RETURN(auto old_step, GetCurrentStep());
  if (old_step == new_step) {
    return absl::OkStatus();
  }

  int max_step = old_step;
  RET_CHECK_LE(new_step, max_step)
          .SetCode(absl::StatusCode::kInvalidArgument)
      << "New step cannot be greater than the max step: "
      << max_step;
  RET_CHECK_GE(new_step, 0).SetCode(absl::StatusCode::kInvalidArgument)
      << "New step cannot be negative.";
  if (new_step == max_step) {
    llm_context_->runtime_state().current_step = new_step;
    return absl::OkStatus();
  }
  RET_CHECK_LE(new_step, max_step)
          .SetCode(absl::StatusCode::kInvalidArgument)
      << "New step cannot be greater than the max step: "
      << max_step;
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

  LITERT_ASSIGN_OR_RETURN(auto tensor_type,
                          (*inputs.GetTextTokenIdsPtr())->TensorType());
  // Accept batch size 1 or output_heads though prefill handles only the
  // first batch element.
  int32_t input_batch_size = tensor_type.Layout().Dimensions()[0];
  if (input_batch_size != 1) {
    RET_CHECK_EQ(input_batch_size, output_heads);
  }
  RET_CHECK_GT(tensor_type.Layout().Dimensions()[1], 0)
      << "Prefill token ids must be non-empty.";

  if (embedding_lookup_ != nullptr) {
    RETURN_IF_ERROR(embedding_lookup_->UpdateMultiModalEmbeddings(inputs));
  }

  LITERT_ASSIGN_OR_RETURN(auto ids, ReferTensorBufferAsSpan<int32_t>(
                                        *(*inputs.GetTextTokenIdsPtr())));
  // Reduce the input ids only with one user selected.
  auto input_length = ids.size() / input_batch_size;
  ids = ids.subspan(kTokenIndexToReduce * input_length, input_length);
  ASSIGN_OR_RETURN(auto work_groups, GetOptimizedPrefillWorkGroups(
                                         prefill_signature_map_, ids.size()));
  for (int i = 0; i < work_groups.size(); ++i) {
    const auto& prefill_signature = work_groups[i].first;
    int prefill_length = work_groups[i].second;
    // Keep track of the signatures that have already had their buffers
    // created only create them once.
    if (!prefill_input_buffers_.contains(prefill_signature)) {
      prefill_input_buffers_[prefill_signature] = {};
      RETURN_IF_ERROR(CreatePrefillInputBuffers(
          prefill_signature, prefill_length, prefill_length,
          prefill_input_buffers_[prefill_signature]));
    }
    bool async = i < work_groups.size() - 1 || !params.GetWaitForCompletion();
    RETURN_IF_ERROR(PrefillInternal(
        prefill_signature, prefill_input_buffers_[prefill_signature],
        ids.subspan(/*pos=*/0, prefill_length), async));
    ids = ids.subspan(/*pos=*/prefill_length);
  }
  RET_CHECK_EQ(ids.size(), 0).SetCode(absl::StatusCode::kInternal)
      << "Work groups not covering the entire prefill input.";

  if (embedding_lookup_ != nullptr) {
    RETURN_IF_ERROR(embedding_lookup_->CleanupMultiModalEmbeddings());
  }

  return absl::OkStatus();
}

// static
// Creates a LlmLiteRtCompiledModelExecutorStatic from a LiteRt model.
absl::StatusOr<std::unique_ptr<LlmLiteRtCompiledModelExecutorStatic>>
LlmLiteRtCompiledModelExecutorStatic::Create(
    LlmExecutorSettings executor_settings, Environment& lrt_env,
    ModelResources& resources) {
  ASSIGN_OR_RETURN(auto litert_model,
                   resources.GetTFLiteModel(ModelType::kTfLitePrefillDecode));
  // For the LlmLiteRtCompiledModelExecutorStatic, ML_DRIFT backend is used by
  // default.
  // TODO(b/405424188): - Add support for NPU backends.
  LITERT_ASSIGN_OR_RETURN(auto compilation_options,
                          ::litert::Options::Create());
  std::string cache_path = executor_settings.GetCacheDir();
  auto activation_data_type = ActivationDataType::FLOAT16;
  // TODO(b/433590109): Some GPUs do not support FP16, so we need to check the
  // capabilities of the GPU and set the activation data type accordingly.
  if (executor_settings.GetActivationDataType().has_value()) {
    activation_data_type = executor_settings.GetActivationDataType().value();
  }
  const Backend backend = executor_settings.GetBackend();
  bool use_fp16_precision = true;
  bool gpu_optimized_single_buffer_cache = false;

  if (!litert_model || !*litert_model) {
    return absl::InternalError("Failed to build LiteRt model");
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
  LITERT_ASSIGN_OR_RETURN(auto prefill_signature,
                          litert_model->FindSignature(prefill_signature_key));
  std::string kv_cache_k_root_name;
  std::string kv_cache_v_root_name;
  RETURN_IF_ERROR(GetKVCacheRootNames(
      prefill_signature.InputNames(), prefill_signature.OutputNames(),
      kv_cache_k_root_name, kv_cache_v_root_name));
  LITERT_ASSIGN_OR_RETURN(auto decode_signature,
                          litert_model->FindSignature(kDecodeSignatureRunner));
  ASSIGN_OR_RETURN(
      ModelSignatures signatures,
      GetModelSignaturesFromInputOutputNames(decode_signature.InputNames(),
                                             decode_signature.OutputNames()));

  switch (backend) {
    case Backend::GPU: {
      // TODO: b/403132820 - Add accelerator compilation options for ML_DRIFT.
      LITERT_ASSIGN_OR_RETURN(auto& gpu_compilation_options,
                              compilation_options.GetGpuOptions());
      gpu_compilation_options.EnableInfiniteFloatCapping(true);
      // Setting this to true could lead to improved latency but lower quality
      // on high end GPUs.
      gpu_compilation_options.EnableAllowSrcQuantizedFcConvOps(false);
      if (activation_data_type == ActivationDataType::FLOAT32) {
        gpu_compilation_options.SetPrecision(GpuOptions::Precision::kFp32);
      } else {
        gpu_compilation_options.SetPrecision(GpuOptions::Precision::kFp16);
      }
#if defined(__APPLE__)
      gpu_compilation_options.SetPreferTextureWeights(false);
      gpu_compilation_options.SetUseMetalArgumentBuffers(true);
#else   // !__APPLE__
      gpu_compilation_options.SetPreferTextureWeights(true);
#endif  // !__APPLE__
      ASSIGN_OR_RETURN(auto model_path,
                       executor_settings.GetModelAssets().GetPath());
      absl::string_view model_name = Basename(model_path);
      gpu_compilation_options.SetModelCacheKey(model_name.data());

      bool serialization_dir_set = false;
      if (cache_path != ":nocache") {
        if (cache_path.empty()) {
          cache_path = std::filesystem::path(std::string(model_path))
                           .parent_path()
                           .string();
          if (cache_path.empty()) {
            cache_path = std::filesystem::current_path().string();
          }
        }
        ABSL_LOG(INFO) << "Setting serialization dir: " << cache_path;
        gpu_compilation_options.SetSerializationDir(cache_path.c_str());
        serialization_dir_set = true;
        gpu_compilation_options.SetSerializeExternalTensors(true);
      } else {
        gpu_compilation_options.SetSerializeExternalTensors(false);
      }

      auto program_cache_file =
          executor_settings.GetProgramCacheFile(".mldrift_program_cache.bin");
      if (program_cache_file.ok()) {
        if (std::holds_alternative<std::string>(*program_cache_file)) {
          if (!serialization_dir_set) {
            cache_path = std::filesystem::path(
                             std::get<std::string>(*program_cache_file))
                             .parent_path()
                             .string();
            ABSL_LOG(INFO) << "Setting program cache dir: " << cache_path;
            gpu_compilation_options.SetSerializationDir(cache_path.c_str());
          }
        } else {
          auto scoped_cache_file =
              std::get<std::shared_ptr<litert::lm::ScopedFile>>(
                  *program_cache_file);
          ASSIGN_OR_RETURN(auto duplicated, scoped_cache_file->Duplicate());
          ASSIGN_OR_RETURN(int fd, duplicated.Release());
          gpu_compilation_options.SetProgramCacheFd(fd);
        }
        gpu_compilation_options.SetSerializeProgramCache(true);
      } else {
        gpu_compilation_options.SetSerializeProgramCache(false);
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
      // Prefill and decode are always fully delegated to single delegate.
      gpu_compilation_options.SetHintFullyDelegatedToSingleDelegate(true);
      auto advanced_settings = executor_settings.GetAdvancedSettings();
      int num_threads_to_upload = kDefaultNumThreadsToUpload;
      int num_threads_to_compile = kDefaultNumThreadsToCompile;
      bool enable_constant_tensor_sharing = true;
      if (advanced_settings) {
        gpu_compilation_options.SetMadviseOriginalSharedTensors(
            advanced_settings->gpu_madvise_original_shared_tensors);
        if (advanced_settings->is_benchmark) {
          gpu_compilation_options.SetSyncExecutionModeWaitType(
              GpuOptions::SyncExecutionModeWaitType::kActive);
        }
        if (!advanced_settings->preferred_device_substr.empty()) {
          gpu_compilation_options.SetPreferredDeviceSubstr(
              advanced_settings->preferred_device_substr.c_str());
        }
        if (advanced_settings->num_threads_to_upload >= 0) {
          num_threads_to_upload = advanced_settings->num_threads_to_upload;
        }
        if (advanced_settings->num_threads_to_compile >= 0) {
          num_threads_to_compile = advanced_settings->num_threads_to_compile;
        }
        if (advanced_settings->convert_weights_on_gpu) {
          gpu_compilation_options.SetConvertWeightsOnGpu(true);
        }
        if (!advanced_settings->optimize_shader_compilation) {
          gpu_compilation_options.DisableShaderOptimization(true);
        }
        enable_constant_tensor_sharing =
            advanced_settings->share_constant_tensors;
      }
      gpu_compilation_options.EnableConstantTensorSharing(
          enable_constant_tensor_sharing);
      // TODO b/441627719 - Select backend by runtime options.
#if defined(LITERT_USE_WEBGPU_ACCELERATOR)
      gpu_compilation_options.SetBackend(GpuOptions::Backend::kWebGpu);
#endif  // defined(LITERT_USE_WEBGPU_ACCELERATOR)
      // Prepare WebGPU or Vulkan command buffers ahead to reduce the overhead
      // of command buffer preparation. 2 steps ahead because KV cache is
      // swapped and the GPU resource bindings are the same as the previous
      // previous step.
      gpu_compilation_options.SetNumStepsOfCommandBufferPreparations(2);
      gpu_compilation_options.SetNumThreadsToUpload(num_threads_to_upload);
      gpu_compilation_options.SetNumThreadsToCompile(num_threads_to_compile);
      compilation_options.SetHardwareAccelerators(litert::HwAccelerators::kGpu);
      break;
    }
    case Backend::CPU: {
      use_fp16_precision = false;
      LITERT_ASSIGN_OR_RETURN(auto& cpu_compilation_options,
                              compilation_options.GetCpuOptions());
      const uint32_t num_threads =
          executor_settings.GetBackendConfig<CpuConfig>()->number_of_threads;
      cpu_compilation_options.SetNumThreads(num_threads);
      auto weight_cache_file =
          executor_settings.GetWeightCacheFile(".xnnpack_cache");
      if (weight_cache_file.ok()) {
        if (std::holds_alternative<std::string>(*weight_cache_file)) {
          cache_path = std::get<std::string>(*weight_cache_file);
          cpu_compilation_options.SetXNNPackWeightCachePath(cache_path.c_str());
        } else {
          auto scoped_cache_file =
              std::get<std::shared_ptr<ScopedFile>>(*weight_cache_file);
          ASSIGN_OR_RETURN(auto duplicated, scoped_cache_file->Duplicate());
          ASSIGN_OR_RETURN(int fd, duplicated.Release());
          cpu_compilation_options.SetXNNPackWeightCacheFileDescriptor(fd);
        }
      }
      LITERT_ASSIGN_OR_RETURN(const uint32_t default_xnnpack_flags,
                              cpu_compilation_options.GetXNNPackFlags());
      cpu_compilation_options.SetXNNPackFlags(
          default_xnnpack_flags |
          TFLITE_XNNPACK_DELEGATE_FLAG_ENABLE_LATEST_OPERATORS);
      LITERT_ASSIGN_OR_RETURN(auto& runtime_options,
                             compilation_options.GetRuntimeOptions());
      runtime_options.SetCompressQuantizationZeroPoints(true);
      compilation_options.SetHardwareAccelerators(litert::HwAccelerators::kCpu);
      break;
    }
    default:
      return absl::InvalidArgumentError(absl::StrCat(
          "Unsupported backend: ", executor_settings.GetBackend()));
  }

  auto section_offset =
      resources.GetWeightsSectionOffset(ModelType::kTfLitePrefillDecode);
  if (section_offset.ok()) {
    if (backend != Backend::GPU) {
      return absl::InvalidArgumentError(
          "Weights section offset is only "
          "supported for GPU backend.");
    }
    Options::ScopedWeightSectionMap section_map;
    section_map["tflite_weights"] = {
        section_offset.value().first,
        section_offset.value().second - section_offset.value().first};
    ABSL_LOG(INFO) << "section_map: " << section_map["tflite_weights"].offset
                   << " " << section_map["tflite_weights"].length;
    LITERT_ASSIGN_OR_RETURN(auto scoped_file, resources.GetScopedFile());
    compilation_options.SetExternalWeightScopedFile(scoped_file.get(),
                                                    section_map);
  };

  LITERT_ASSIGN_OR_RETURN(
      auto compiled_model,
      CompiledModel::Create(lrt_env, *litert_model, compilation_options));

  absl::flat_hash_map<absl::string_view, TensorBuffer> decode_input_buffers;
  absl::flat_hash_map<absl::string_view, TensorBuffer> decode_output_buffers;
  absl::flat_hash_map<absl::string_view, TensorBuffer> input_kv_cache_buffers;
  absl::flat_hash_map<absl::string_view, TensorBuffer> output_kv_cache_buffers;

  bool clear_kv_cache_before_prefill =
      !executor_settings.GetAdvancedSettings() ||
      executor_settings.GetAdvancedSettings()->clear_kv_cache_before_prefill;
  for (auto input_name : prefill_signature.InputNames()) {
    // Skip creating buffers for the input tokens, positions and attn mask. Move
    // into prefill function to create them based on the ids size.
    if ((!absl::StartsWith(input_name, kv_cache_k_root_name) &&
         !absl::StartsWith(input_name, kv_cache_v_root_name)) ||
        gpu_optimized_single_buffer_cache) {
      continue;
    }
    LITERT_ASSIGN_OR_RETURN(
        auto input_buffer,
        compiled_model.CreateInputBuffer(prefill_signature_key, input_name));
    if (clear_kv_cache_before_prefill) {
      LITERT_RETURN_IF_ERROR(input_buffer.Clear());
    }
    if (backend == Backend::CPU) {
      LITERT_ASSIGN_OR_RETURN(auto output_buffer, input_buffer.Duplicate());
      output_kv_cache_buffers[input_name] = std::move(output_buffer);
    }
    input_kv_cache_buffers[input_name] = std::move(input_buffer);
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
  int batch_size = output_logits_buffer_tensor_type.Layout().Dimensions()[0];

  std::optional<absl::flat_hash_map<absl::string_view, TensorBuffer>>
      decode_input_kv_cache_buffers;
  std::optional<absl::flat_hash_map<absl::string_view, TensorBuffer>>
      decode_output_kv_cache_buffers;
  if (batch_size > 1) {
    ABSL_LOG(INFO) << "Decode batch size is larger than 1. Allocate decode "
                   << "only KV cache buffers.";
    decode_input_kv_cache_buffers =
        absl::flat_hash_map<absl::string_view, TensorBuffer>();
    decode_output_kv_cache_buffers =
        absl::flat_hash_map<absl::string_view, TensorBuffer>();
    for (auto input_name : decode_signature.InputNames()) {
      if (absl::StartsWith(input_name, kv_cache_k_root_name) ||
          absl::StartsWith(input_name, kv_cache_v_root_name)) {
        LITERT_ASSIGN_OR_RETURN(auto input_buffer,
                                compiled_model.CreateInputBuffer(
                                    kDecodeSignatureRunner, input_name));
        if (clear_kv_cache_before_prefill) {
          LITERT_RETURN_IF_ERROR(input_buffer.Clear());
        }
        (*decode_input_kv_cache_buffers)[input_name] = std::move(input_buffer);
      }
    }
    for (auto output_name : decode_signature.OutputNames()) {
      if (absl::StartsWith(output_name, kv_cache_k_root_name) ||
          absl::StartsWith(output_name, kv_cache_v_root_name)) {
        LITERT_ASSIGN_OR_RETURN(auto output_buffer,
                                compiled_model.CreateOutputBuffer(
                                    kDecodeSignatureRunner, output_name));
        (*decode_output_kv_cache_buffers)[output_name] =
            std::move(output_buffer);
      }
    }
  }

  ASSIGN_OR_RETURN(auto prefill_runner_set,
                   GetPrefillRunnerSetFromModel(
                       *litert_model, kPrefillSignatureRunner,
                       /*input_positions_name=*/signatures.input_positions));
  RET_CHECK(!prefill_runner_set.empty()) << "No prefill runner available.";

  std::unique_ptr<EmbeddingLookupManager> embedding_lookup;
  std::unique_ptr<EmbeddingLookupManager> per_layer_embedding_lookup;
  RETURN_IF_ERROR(InitializeEmbeddingLookups(resources, embedding_lookup,
                                             per_layer_embedding_lookup));
  return absl::WrapUnique(new LlmLiteRtCompiledModelExecutorStatic(
      std::move(executor_settings), lrt_env, litert_model,
      std::move(compiled_model), std::move(decode_input_buffers),
      std::move(decode_output_buffers), std::move(input_kv_cache_buffers),
      std::move(output_kv_cache_buffers),
      std::move(decode_input_kv_cache_buffers),
      std::move(decode_output_kv_cache_buffers), std::move(prefill_runner_set),
      signatures, batch_size, std::move(cache_path),
      std::move(embedding_lookup), std::move(per_layer_embedding_lookup),
      use_fp16_precision, activation_data_type));
}

/* ===========================================================================*/
/* LlmLiteRtCompiledModelExecutorDynamic */
/* ===========================================================================*/

absl::Status LlmLiteRtCompiledModelExecutorDynamic::Prefill(
    const ExecutorInputs& inputs, const ExecutorPrefillParams& params) {

  // Only accept batch size 1 for now.
  LITERT_RETURN_IF_ERROR(PrepareFirstPrefillAfterDecode(0));

  LITERT_ASSIGN_OR_RETURN(auto tensor_type,
                          (*inputs.GetTextTokenIdsPtr())->TensorType());
  RET_CHECK_EQ(tensor_type.Layout().Dimensions()[0], 1);
  RET_CHECK_GT(tensor_type.Layout().Dimensions()[1], 0)
      << "Prefill token ids must be non-empty.";
  LITERT_ASSIGN_OR_RETURN(
      absl::Span<int> ids,
      ReferTensorBufferAsSpan<int32_t>(*(*inputs.GetTextTokenIdsPtr())));

  if (prefill_chunk_size_ <= 0) {
    return PrefillInternal(ids, params);
  }

  while (!ids.empty()) {
    int chunk_size =
        std::min(static_cast<int>(ids.size()), prefill_chunk_size_);
    absl::Span<int> chunk_ids = ids.first(chunk_size);
    ids = ids.subspan(chunk_size);
    RETURN_IF_ERROR(PrefillInternal(chunk_ids, params));
  }
  return absl::OkStatus();
}

absl::Status LlmLiteRtCompiledModelExecutorDynamic::PrefillInternal(
    absl::Span<int> ids, const ExecutorPrefillParams& params) {
  RETURN_IF_ERROR(RollBackProcessedTokens());
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
    RETURN_IF_ERROR(
        llm_context_->processed_context()
            .processed_tokens()
            .AddPendingInputToken({std::make_shared<TokenData>(ids[0])}));
    return absl::OkStatus();
  }

  int kv_length = 0;
  if (kv_cache_buffers_1_.empty()) {
    kv_length = prefill_length;
    // First time prefilling, allocate KV cache buffers.
    bool clear_kv_cache_before_prefill =
        !executor_settings_.GetAdvancedSettings() ||
        executor_settings_.GetAdvancedSettings()->clear_kv_cache_before_prefill;
    for (const auto& k_cache_input_name : key_cache_input_names_) {
      RETURN_IF_ERROR(ResolveDynamicShape(model_, compiled_model_, "prefill",
                                          k_cache_input_name, prefill_length));
      LITERT_ASSIGN_OR_RETURN(
          auto input_buffer,
          compiled_model_.CreateInputBuffer("prefill", k_cache_input_name));
      if (clear_kv_cache_before_prefill) {
        LITERT_RETURN_IF_ERROR(input_buffer.Clear());
      }
      kv_cache_buffers_1_[k_cache_input_name] = std::move(input_buffer);
    }
    for (const auto& v_cache_input_name : value_cache_input_names_) {
      RETURN_IF_ERROR(ResolveDynamicShape(model_, compiled_model_, "prefill",
                                          v_cache_input_name, prefill_length));
      LITERT_ASSIGN_OR_RETURN(
          auto input_buffer,
          compiled_model_.CreateInputBuffer("prefill", v_cache_input_name));
      if (clear_kv_cache_before_prefill) {
        LITERT_RETURN_IF_ERROR(input_buffer.Clear());
      }
      kv_cache_buffers_1_[v_cache_input_name] = std::move(input_buffer);
    }
  } else {
    {
      RET_CHECK(!kv_cache_buffers_1_.empty());
      const TensorBuffer& key_buffer =
          kv_cache_buffers_1_[key_cache_input_names_[0]];
      LITERT_ASSIGN_OR_RETURN(const RankedTensorType& key_buffer_tensor_type,
                              key_buffer.TensorType());
      kv_length =
          key_buffer_tensor_type.Layout().Dimensions()[key_dynamic_dim_index_];
    }

    int free_kv_entries = kv_length - step_and_token.step;
    if (prefill_length > free_kv_entries) {
      int new_kv_seq_len = kv_length + prefill_length;
      int entries_to_add = new_kv_seq_len - kv_length;
      for (const auto& k_cache_input_name : key_cache_input_names_) {
        RETURN_IF_ERROR(ResolveDynamicShape(model_, compiled_model_, "prefill",
                                            k_cache_input_name,
                                            new_kv_seq_len));
        ASSIGN_OR_RETURN(kv_cache_buffers_1_[k_cache_input_name],
                         ResizeKVCacheTensorBuffer(
                             env_, kv_cache_buffers_1_[k_cache_input_name],
                             key_dynamic_dim_index_, entries_to_add));
      }
      for (const auto& v_cache_input_name : value_cache_input_names_) {
        RETURN_IF_ERROR(ResolveDynamicShape(model_, compiled_model_, "prefill",
                                            v_cache_input_name,
                                            new_kv_seq_len));
        ASSIGN_OR_RETURN(kv_cache_buffers_1_[v_cache_input_name],
                         ResizeKVCacheTensorBuffer(
                             env_, kv_cache_buffers_1_[v_cache_input_name],
                             value_dynamic_dim_index_, entries_to_add));
      }
      kv_length = new_kv_seq_len;
    }
  }

  absl::flat_hash_map<absl::string_view, TensorBuffer> prefill_input_buffers;
  RETURN_IF_ERROR(CreatePrefillInputBuffers("prefill", prefill_length,
                                            kv_length, prefill_input_buffers));

  input_kv_cache_buffers_ = &kv_cache_buffers_1_;
  output_kv_cache_buffers_ = &kv_cache_buffers_1_;

  bool async = !params.GetWaitForCompletion();
  return LlmLiteRtCompiledModelExecutorBase::PrefillInternal(
      "prefill", prefill_input_buffers, ids, async);
}

absl::Status LlmLiteRtCompiledModelExecutorDynamic::DecodeInternal(
    const std::vector<std::shared_ptr<TokenData>>& token,
    TensorBuffer& output_logits) {
  int current_kv_len = 0;
  {
    RET_CHECK(!kv_cache_buffers_1_.empty());
    const TensorBuffer& key_buffer =
        kv_cache_buffers_1_[key_cache_input_names_[0]];
    LITERT_ASSIGN_OR_RETURN(const RankedTensorType& key_buffer_tensor_type,
                            key_buffer.TensorType());
    current_kv_len =
        key_buffer_tensor_type.Layout().Dimensions()[key_dynamic_dim_index_];
  }

  if (current_kv_len <= llm_context_->runtime_state().current_step - 1) {
    int entries_to_add = kv_increament_size_;
    int new_kv_len = current_kv_len + entries_to_add;
    for (const auto& k_cache_input_name : key_cache_input_names_) {
      RETURN_IF_ERROR(ResolveDynamicShape(model_, compiled_model_, "decode",
                                          k_cache_input_name, new_kv_len));
      ASSIGN_OR_RETURN(kv_cache_buffers_1_[k_cache_input_name],
                       ResizeKVCacheTensorBuffer(
                           env_, kv_cache_buffers_1_[k_cache_input_name],
                           key_dynamic_dim_index_, entries_to_add));
    }
    for (const auto& v_cache_input_name : value_cache_input_names_) {
      RETURN_IF_ERROR(ResolveDynamicShape(model_, compiled_model_, "decode",
                                          v_cache_input_name, new_kv_len));
      ASSIGN_OR_RETURN(kv_cache_buffers_1_[v_cache_input_name],
                       ResizeKVCacheTensorBuffer(
                           env_, kv_cache_buffers_1_[v_cache_input_name],
                           value_dynamic_dim_index_, entries_to_add));
    }
    current_kv_len = new_kv_len;
  }

  RETURN_IF_ERROR(ResolveDynamicShape(model_, compiled_model_, "decode",
                                      signatures_.input_attn_mask.value(),
                                      current_kv_len));
  LITERT_ASSIGN_OR_RETURN(
      decode_input_buffers_[signatures_.input_attn_mask.value()],
      compiled_model_.CreateInputBuffer("decode",
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
  ASSIGN_OR_RETURN(auto litert_model,
                   resources.GetTFLiteModel(ModelType::kTfLitePrefillDecode));
  LITERT_ASSIGN_OR_RETURN(auto compilation_options,
                          ::litert::Options::Create());
  std::string weight_cache_path = executor_settings.GetCacheDir();
  const Backend backend = executor_settings.GetBackend();
  RET_CHECK_EQ(backend, Backend::CPU)
      << "LlmLiteRtCompiledModelExecutorDynamic only supports CPU backend.";
  uint32_t kv_increament_size = 0;
  int prefill_chunk_size = -1;
  {
    LITERT_ASSIGN_OR_RETURN(auto& cpu_compilation_options,
                            compilation_options.GetCpuOptions());
    ASSIGN_OR_RETURN(const auto& cpu_config,
                     executor_settings.GetBackendConfig<CpuConfig>());
    kv_increament_size = cpu_config.kv_increment_size;
    prefill_chunk_size = cpu_config.prefill_chunk_size;
    cpu_compilation_options.SetNumThreads(cpu_config.number_of_threads);
    auto weight_cache_file =
        executor_settings.GetWeightCacheFile(".xnnpack_cache");
    if (weight_cache_file.ok()) {
      if (std::holds_alternative<std::string>(*weight_cache_file)) {
        weight_cache_path = std::get<std::string>(*weight_cache_file);
        cpu_compilation_options.SetXNNPackWeightCachePath(
            weight_cache_path.c_str());
      } else {
        auto scoped_cache_file =
            std::get<std::shared_ptr<ScopedFile>>(*weight_cache_file);
        ASSIGN_OR_RETURN(auto duplicated, scoped_cache_file->Duplicate());
        ASSIGN_OR_RETURN(int fd, duplicated.Release());
        cpu_compilation_options.SetXNNPackWeightCacheFileDescriptor(fd);
      }
    }
    RET_CHECK_GT(kv_increament_size, 0)
        << "KV increment size must be greater than 0.";
    LITERT_ASSIGN_OR_RETURN(const uint32_t default_xnnpack_flags,
                            cpu_compilation_options.GetXNNPackFlags());
    cpu_compilation_options.SetXNNPackFlags(
        default_xnnpack_flags |
        TFLITE_XNNPACK_DELEGATE_FLAG_ENABLE_LATEST_OPERATORS);
    LITERT_ASSIGN_OR_RETURN(auto& runtime_options,
                            compilation_options.GetRuntimeOptions());
    runtime_options.SetCompressQuantizationZeroPoints(true);
    compilation_options.SetHardwareAccelerators(litert::HwAccelerators::kCpu);
  }

  LITERT_ASSIGN_OR_RETURN(
      auto compiled_model,
      CompiledModel::Create(lrt_env, *litert_model, compilation_options));

  absl::flat_hash_map<absl::string_view, TensorBuffer> decode_input_buffers;
  absl::flat_hash_map<absl::string_view, TensorBuffer> decode_output_buffers;

  LITERT_ASSIGN_OR_RETURN(auto decode_signature,
                          litert_model->FindSignature(kDecodeSignatureRunner));
  std::string kv_cache_k_root_name;
  std::string kv_cache_v_root_name;
  RETURN_IF_ERROR(GetKVCacheRootNames(
      decode_signature.InputNames(), decode_signature.OutputNames(),
      kv_cache_k_root_name, kv_cache_v_root_name));
  ASSIGN_OR_RETURN(
      ModelSignatures signatures,
      GetModelSignaturesFromInputOutputNames(decode_signature.InputNames(),
                                             decode_signature.OutputNames()));

  std::vector<std::string> key_cache_input_names;
  std::vector<std::string> value_cache_input_names;
  for (auto input_name : decode_signature.InputNames()) {
    bool is_key_cache_input =
        absl::StartsWith(input_name, kv_cache_k_root_name);
    if (is_key_cache_input) {
      key_cache_input_names.push_back(std::string(input_name));
    }

    bool is_value_cache_input =
        absl::StartsWith(input_name, kv_cache_v_root_name);
    if (is_value_cache_input) {
      value_cache_input_names.push_back(std::string(input_name));
    }

    bool is_kv_cache_input = is_key_cache_input || is_value_cache_input;
    bool is_attn_mask_input =
        signatures.input_attn_mask.has_value() &&
        absl::StartsWith(input_name, signatures.input_attn_mask.value());
    if (!is_kv_cache_input && !is_attn_mask_input) {
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

  ASSIGN_OR_RETURN(
      int k_dynamic_dim,
      GetDynamicDimIndex(*litert_model, "prefill", key_cache_input_names[0]));
  ASSIGN_OR_RETURN(
      int v_dynamic_dim,
      GetDynamicDimIndex(*litert_model, "prefill", value_cache_input_names[0]));

  LITERT_ASSIGN_OR_RETURN(
      auto output_logits_buffer,
      decode_output_buffers[signatures.output_logits].Duplicate());
  LITERT_ASSIGN_OR_RETURN(auto output_logits_buffer_tensor_type,
                          output_logits_buffer.TensorType());
  RET_CHECK(output_logits_buffer_tensor_type.Layout().Dimensions().size() == 3)
      << "Output logits must be (batch, seq, vocab)";
  int batch_size = output_logits_buffer_tensor_type.Layout().Dimensions()[0];
  RET_CHECK_EQ(batch_size, 1) << "Only support batch size 1 for now.";
  std::unique_ptr<EmbeddingLookupManager> embedding_lookup;
  std::unique_ptr<EmbeddingLookupManager> per_layer_embedding_lookup;
  RETURN_IF_ERROR(InitializeEmbeddingLookups(resources, embedding_lookup,
                                             per_layer_embedding_lookup));

  return absl::WrapUnique(new LlmLiteRtCompiledModelExecutorDynamic(
      std::move(executor_settings), lrt_env, litert_model,
      std::move(compiled_model), std::move(decode_input_buffers),
      std::move(decode_output_buffers), prefill_chunk_size, k_dynamic_dim,
      v_dynamic_dim, kv_increament_size, std::move(key_cache_input_names),
      std::move(value_cache_input_names), signatures, batch_size,
      std::move(weight_cache_path), std::move(embedding_lookup),
      std::move(per_layer_embedding_lookup), /*use_fp16_precision=*/false));
}

}  // namespace litert::lm
