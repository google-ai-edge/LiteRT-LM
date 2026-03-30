#include "runtime/executor/litert/llm_executor.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>  // NOLINT
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
#include "litert/cc/litert_model_types.h"  // from @litert
#include "litert/cc/litert_options.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/cc/litert_tensor_buffer_types.h"  // from @litert
#include "litert/cc/options/litert_cpu_options.h"  // from @litert
#include "litert/cc/options/litert_gpu_options.h"  // from @litert
#include "litert/cc/options/litert_runtime_options.h"  // from @litert
#include "runtime/components/embedding_lookup/embedding_lookup_manager.h"
#include "runtime/components/model_resources.h"
#include "runtime/components/sampler.h"
#include "runtime/components/sampler_factory.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/kv_cache_interface.h"
#include "runtime/executor/litert/kv_cache.h"
#include "runtime/executor/litert_compiled_model_executor_utils.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/executor/llm_litert_compiled_model_cache_utils.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/file_util.h"
#include "runtime/util/lora_util.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/status_macros.h"
#include "tflite/delegates/xnnpack/xnnpack_delegate.h"  // from @litert

namespace litert::lm {

// Names of the signature runners, used to get the signature runners from the
// interpreter.
constexpr absl::string_view kPrefillSignatureRunner = "prefill";
constexpr absl::string_view kDecodeSignatureRunner = "decode";
constexpr int kDynamicDimValue = -1;

// Default number of threads for WebGPU weight upload and kernel compilation.
constexpr int kDefaultNumThreadsToUpload = 2;
constexpr int kDefaultNumThreadsToCompile = 1;

namespace {

class WrappedCompiledModel : public CompiledModel {
 public:
  using CompiledModel::Create;
};

absl::Status InitializeEmbeddingLookups(
    ModelResources& resources,
    std::unique_ptr<EmbeddingLookupManager>& embedding_lookup,
    std::unique_ptr<EmbeddingLookupManager>& per_layer_embedding_lookup) {
  auto end_of_audio_model =
      resources.GetTFLiteModel(ModelType::kTfLiteEndOfAudio);
  absl::flat_hash_map<int, const Model*> end_of_multi_modal_embedding_models;
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

}  // namespace

/* ===========================================================================*/
/* LlmExecutorCommonCore */
/* ===========================================================================*/

absl::StatusOr<std::unique_ptr<KVCacheInterface>>
LlmExecutorCommonCore::CreateKVCache() {
  bool inplace_update = true;
  if (executor_settings_.GetBackend() == Backend::GPU) {
    inplace_update = false;
  }
  return LitertKVCache::Create(env_, model_, kDecodeSignatureRunner,
                               compiled_model_, inplace_update);
}

absl::StatusOr<int> LlmExecutorCommonCore::LoadLoRA(
    const ModelAssets& model_assets) {
  // create a random lora id
  return lora_manager_->LoadLoRA(next_lora_id_++, model_assets);
}

absl::Status LlmExecutorCommonCore::UnloadLoRA(int lora_id) {
  return absl::UnimplementedError("UnloadLoRA is not implemented.");
}

absl::Status LlmExecutorCommonCore::Cancel() {
  return absl::UnimplementedError("Cancel is not implemented.");
}

absl::Status LlmExecutorCommonCore::Prefill(
    absl::string_view prefill_signature,
    absl::flat_hash_map<absl::string_view, TensorBuffer>& prefill_input_buffers,
    absl::Span<const int> ids, int next_position, LitertKVCache& kv_cache,
    std::optional<int> lora_id) {
  int prefill_length = ids.size();
  const bool use_token_as_lookup = !signatures_.input_tokens.empty();
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
    for (int i = 0; i < prefill_length; ++i) {
      prefill_input_pos_ptr[i] = next_position + i;
    }

    if (signatures_.input_attn_mask.has_value()) {
      RETURN_IF_ERROR(InitializeAttentionMask(
          prefill_input_buffers[signatures_.input_attn_mask.value()],
          use_fp16_precision_));
    }

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
      memset(prefill_input_ptr, 0, prefill_input_size);
      memcpy(prefill_input_ptr, ids.data(), ids.size() * sizeof(int32_t));
    } else {
      // If not using token as lookup, we must have input_embeddings. There is
      // no need to create input_embeddings_ptr because TensorBuffer locking and
      // filling is handled by the embedding lookup.
      TensorBuffer* prefill_input_embeddings_buffer =
          &(prefill_input_buffers[signatures_.input_embeddings.value()]);
      RETURN_IF_ERROR(
          embedding_lookup_->LookupPrefill(ids, prefill_input_embeddings_buffer,
                                           /*offset=*/0));

      // We may have per layer embedding as well.
      if (signatures_.input_per_layer_embeddings) {
        TensorBuffer* prefill_input_per_layer_embeddings_buffer =
            &(prefill_input_buffers[signatures_.input_per_layer_embeddings
                                        .value()]);
        RETURN_IF_ERROR(per_layer_embedding_lookup_->LookupPrefill(
            ids, prefill_input_per_layer_embeddings_buffer,
            /*offset=*/0));
      }
    }
    if (signatures_.input_attn_mask.has_value()) {
      RETURN_IF_ERROR(FillAttentionMask(
          prefill_input_buffers[signatures_.input_attn_mask.value()],
          next_position, prefill_length));
    }
    if (signatures_.input_int32_param.has_value()) {
      RETURN_IF_ERROR(FillSingleBufferCacheParamTensor(
          prefill_input_buffers[signatures_.input_int32_param.value()],
          next_position, prefill_length));
    }
  }

  return BindTensorsAndRunPrefill(prefill_signature, prefill_input_buffers,
                                  kv_cache, lora_id);
}

absl::StatusOr<absl::flat_hash_map<absl::string_view, TensorBuffer>>
LlmExecutorCommonCore::CreatePrefillInputBuffers(
    absl::string_view prefill_signature, int sequence_length,
    int context_length) {
  absl::flat_hash_map<absl::string_view, TensorBuffer> input_buffers;
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
    input_buffers[signatures_.input_tokens] = std::move(*tokens_buffer);
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
    input_buffers[signatures_.input_embeddings.value()] =
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
      input_buffers[signatures_.input_per_layer_embeddings.value()] =
          std::move(*per_layer_embeddings_buffer);
    }
  }
  RETURN_IF_ERROR(dyn_shape_resolver(signatures_.input_positions));
  auto positions_buffer = compiled_model_.CreateInputBuffer(
      prefill_signature, signatures_.input_positions);
  input_buffers[signatures_.input_positions] = std::move(*positions_buffer);

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
    input_buffers[signatures_.input_attn_mask.value()] =
        std::move(*attn_mask_buffer);
  }
  return input_buffers;
}

absl::Status LlmExecutorCommonCore::BindTensorsAndRunPrefill(
    absl::string_view signature,
    absl::flat_hash_map<absl::string_view, TensorBuffer>& input_buffers,
    LitertKVCache& kv_cache, std::optional<int> lora_id) {
  absl::flat_hash_map<absl::string_view, TensorBuffer> all_input_buffers;
  for (const auto& [input_name, input_buffer] : input_buffers) {
    LITERT_ASSIGN_OR_RETURN(auto input_buffer_dup, input_buffer.Duplicate());
    all_input_buffers[input_name] = std::move(input_buffer_dup);
  }
  ASSIGN_OR_RETURN(auto kv_cache_buffers, kv_cache.GetKVCacheBuffers());
  for (const auto& [input_name, input_buffer] :
       kv_cache_buffers.input_buffers) {
    LITERT_ASSIGN_OR_RETURN(auto input_buffer_dup, input_buffer.Duplicate());
    all_input_buffers[input_name] = std::move(input_buffer_dup);
  }

  absl::flat_hash_map<absl::string_view, TensorBuffer> output_buffers;
  for (const auto& [output_name, output_buffer] :
       kv_cache_buffers.output_buffers) {
    LITERT_ASSIGN_OR_RETURN(auto output_buffer_dup, output_buffer.Duplicate());
    output_buffer_dup.ClearEvent();
    output_buffers[output_name] = std::move(output_buffer_dup);
  }

  LITERT_RETURN_IF_ERROR(
      compiled_model_.Run(signature, all_input_buffers, output_buffers));

  return absl::OkStatus();
}

absl::StatusOr<TensorBuffer> LlmExecutorCommonCore::Step(
    int token, int next_position, LitertKVCache& kv_cache,
    std::optional<int> lora_id) {
  const bool use_token_as_lookup = !signatures_.input_tokens.empty();
  if (use_token_as_lookup) {
    auto& decode_input_buffer = decode_input_buffers_[signatures_.input_tokens];
    LITERT_ASSIGN_OR_RETURN(
        auto decode_input_lock_and_addr,
        TensorBufferScopedLock::Create(decode_input_buffer,
                                       TensorBuffer::LockMode::kWrite));
    int32_t* decode_input_ptr =
        static_cast<int32_t*>(decode_input_lock_and_addr.second);
    *decode_input_ptr = token;
  } else {
    // If not using token as lookup, we must have input_embeddings. There is
    // no need to create input_embeddings_ptr because TensorBuffer locking and
    // filling is handled by the embedding lookup.
    TensorBuffer* decode_input_embeddings_buffer =
        &(decode_input_buffers_[signatures_.input_embeddings.value()]);
    RETURN_IF_ERROR(
        embedding_lookup_->LookupDecode(token, decode_input_embeddings_buffer));

    // We may have per layer embedding as well.
    if (signatures_.input_per_layer_embeddings) {
      TensorBuffer* decode_input_per_layer_embeddings_buffer =
          &(decode_input_buffers_[signatures_.input_per_layer_embeddings
                                      .value()]);
      RETURN_IF_ERROR(per_layer_embedding_lookup_->LookupDecode(
          token, decode_input_per_layer_embeddings_buffer));
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
      *input_pos_ptr = next_position;
    } else {
      int output_heads = input_pos_type.Layout().Dimensions()[0];
      LITERT_ASSIGN_OR_RETURN(
          auto input_pos_size,
          decode_input_buffers_[signatures_.input_positions].PackedSize());
      size_t offset = input_pos_size / output_heads / sizeof(int32_t);
      for (int i = 0; i < output_heads; ++i) {
        input_pos_ptr[i * offset] = next_position;
      }
    }
  }

  if (signatures_.input_attn_mask.has_value()) {
    RETURN_IF_ERROR(InitializeAttentionMask(
        decode_input_buffers_[signatures_.input_attn_mask.value()],
        use_fp16_precision_));
    RETURN_IF_ERROR(FillAttentionMask(
        decode_input_buffers_[signatures_.input_attn_mask.value()],
        next_position,
        /*steps=*/1));
  }
  if (signatures_.input_int32_param.has_value()) {
    RETURN_IF_ERROR(FillSingleBufferCacheParamTensor(
        decode_input_buffers_[signatures_.input_int32_param.value()],
        next_position, 1));
  }

  return BindTensorsAndRunStep(kv_cache, lora_id);
}

absl::StatusOr<TensorBuffer> LlmExecutorCommonCore::BindTensorsAndRunStep(
    LitertKVCache& kv_cache, std::optional<int> lora_id) {
  absl::flat_hash_map<absl::string_view, TensorBuffer> decode_input_buffers;
  for (const auto& [input_name, input_buffer] : decode_input_buffers_) {
    LITERT_ASSIGN_OR_RETURN(auto input_buffer_dup, input_buffer.Duplicate());
    decode_input_buffers[input_name] = std::move(input_buffer_dup);
  }

  ASSIGN_OR_RETURN(auto kv_cache_buffers, kv_cache.GetKVCacheBuffers());
  for (const auto& [input_name, input_buffer] :
       kv_cache_buffers.input_buffers) {
    LITERT_ASSIGN_OR_RETURN(auto input_buffer_dup, input_buffer.Duplicate());
    decode_input_buffers[input_name] = std::move(input_buffer_dup);
  }

  absl::flat_hash_map<absl::string_view, TensorBuffer> decode_output_buffers;
  for (const auto& [output_name, output_buffer] : decode_output_buffers_) {
    LITERT_ASSIGN_OR_RETURN(auto output_buffer_dup, output_buffer.Duplicate());
    decode_output_buffers[output_name] = std::move(output_buffer_dup);
  }
  for (const auto& [output_name, output_buffer] :
       kv_cache_buffers.output_buffers) {
    LITERT_ASSIGN_OR_RETURN(auto output_buffer_dup, output_buffer.Duplicate());
    output_buffer_dup.ClearEvent();
    decode_output_buffers[output_name] = std::move(output_buffer_dup);
  }

  LITERT_RETURN_IF_ERROR(compiled_model_.Run(
      kDecodeSignatureRunner, decode_input_buffers, decode_output_buffers));

  return std::move(decode_output_buffers[signatures_.output_logits]);
};

absl::StatusOr<int> LlmExecutorCommonCore::InferVocabSize() {
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
/* LlmExecutorExternalSamplerStatic */
/* ===========================================================================*/

absl::StatusOr<std::unique_ptr<LlmExecutorExternalSamplerStatic>>
LlmExecutorExternalSamplerStatic::Create(LlmExecutorSettings executor_settings,
                                         Environment& env,
                                         ModelResources& resources) {
  ASSIGN_OR_RETURN(auto litert_model,
                   resources.GetTFLiteModel(ModelType::kTfLitePrefillDecode));
  LITERT_ASSIGN_OR_RETURN(auto compilation_options, Options::Create());
  std::string cache_path = executor_settings.GetCacheDir();
  auto activation_data_type = ActivationDataType::FLOAT16;
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
  LITERT_ASSIGN_OR_RETURN(auto decode_signature,
                          litert_model->FindSignature(kDecodeSignatureRunner));
  ASSIGN_OR_RETURN(
      ModelSignatures signatures,
      GetModelSignaturesFromInputOutputNames(decode_signature.InputNames(),
                                             decode_signature.OutputNames()));

  ASSIGN_OR_RETURN(auto sampler_backend, GetSamplerBackend(executor_settings));

  switch (backend) {
    case Backend::GPU: {
      // TODO: b/403132820 - Add accelerator compilation options for ML_DRIFT.
      LITERT_ASSIGN_OR_RETURN(auto& gpu_compilation_options,
                              compilation_options.GetGpuOptions());
      gpu_compilation_options.EnableInfiniteFloatCapping(true);
      if (activation_data_type == ActivationDataType::FLOAT32) {
        use_fp16_precision = false;
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

      bool has_valid_model_fd =
          executor_settings.GetModelAssets().GetScopedFile().ok() &&
          executor_settings.GetModelAssets().GetScopedFile().value()->IsValid();

      auto program_cache_file =
          executor_settings.GetProgramCacheFile(".mldrift_program_cache.bin");
      bool has_valid_program_cache_fd =
          program_cache_file.ok() &&
          !std::holds_alternative<std::string>(*program_cache_file);

      auto model_path_or_status = executor_settings.GetModelAssets().GetPath();
      if (model_path_or_status.ok()) {
        // If the model path is available, use the model name as the cache key.
        absl::string_view model_path = *model_path_or_status;
        absl::string_view model_name = Basename(model_path);
        gpu_compilation_options.SetModelCacheKey(model_name.data());
      } else if (has_valid_model_fd && has_valid_program_cache_fd) {
        // If the model is loaded from an fd, there is no way to automatically
        // generate a cache key. But if we are loading a model from an fd, it is
        // likely that our program cache is also loaded from an fd which does
        // not require a cache key to prevent collisions. The GPU delegate will
        // still expect a cache key, so we set it to a constant value.
        gpu_compilation_options.SetModelCacheKey("fd_token");
      }

      AdvancedSettings advanced_settings;
      if (executor_settings.GetAdvancedSettings()) {
        advanced_settings = *executor_settings.GetAdvancedSettings();
      }

      bool serialization_dir_set = false;
      if (cache_path != ":nocache") {
        if (cache_path.empty()) {
          ASSIGN_OR_RETURN(auto model_path,
                           executor_settings.GetModelAssets().GetPath());
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
        gpu_compilation_options.CacheCompiledProgramsOnly(
            advanced_settings.cache_compiled_shaders_only);
      } else {
        gpu_compilation_options.SetSerializeExternalTensors(false);
      }

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
              std::get<std::shared_ptr<lm::ScopedFile>>(*program_cache_file);
          ASSIGN_OR_RETURN(auto duplicated, scoped_cache_file->Duplicate());
          ASSIGN_OR_RETURN(int fd, duplicated.Release());
          gpu_compilation_options.SetProgramCacheFd(fd);
        }
        gpu_compilation_options.SetSerializeProgramCache(true);
      } else {
        gpu_compilation_options.SetSerializeProgramCache(false);
      }

      // Use NoExternalTensorsMode to get better performance.
      ASSIGN_OR_RETURN(auto backend_config,
                       executor_settings.GetBackendConfig<GpuConfig>());
      bool external_tensor_mode = backend_config.external_tensor_mode;
      gpu_compilation_options.EnableExternalTensorsMode(external_tensor_mode);
      if (!external_tensor_mode) {
        // This option prevents KVCache handling from being affected by
        // BHWC conversion in NoExternalTensorsMode.
        gpu_compilation_options.AddExternalTensorPattern("kv_cache_");
        if (signatures.input_int32_param.has_value()) {
          gpu_optimized_single_buffer_cache = true;
          gpu_compilation_options.AddBufferStorageTensorPattern("kv_cache_");
          gpu_compilation_options.AddExternalTensorPattern("param_tensor");
          gpu_compilation_options.AddBufferStorageTensorPattern("param_tensor");
        }
        if (sampler_backend == Backend::GPU) {
          // GPU Sampler requires logits to be external tensors (PHWC4 format).
          gpu_compilation_options.AddExternalTensorPattern("logits");
        }
      }
      // Prefill and decode are always fully delegated to single delegate.
      gpu_compilation_options.SetHintFullyDelegatedToSingleDelegate(true);

      gpu_compilation_options.SetMadviseOriginalSharedTensors(
          advanced_settings.gpu_madvise_original_shared_tensors);
      gpu_compilation_options.SetConvertWeightsOnGpu(
          advanced_settings.convert_weights_on_gpu);
      gpu_compilation_options.EnableConstantTensorSharing(
          advanced_settings.share_constant_tensors);
      gpu_compilation_options.EnableAllowSrcQuantizedFcConvOps(
          !advanced_settings.allow_src_quantized_fc_conv_ops.has_value() ||
          advanced_settings.allow_src_quantized_fc_conv_ops.value());
      gpu_compilation_options.HintWaitingForCompletion(
          advanced_settings.hint_waiting_for_completion.has_value() &&
          advanced_settings.hint_waiting_for_completion.value());
      if (advanced_settings.is_benchmark) {
        gpu_compilation_options.SetSyncExecutionModeWaitType(
            GpuOptions::SyncExecutionModeWaitType::kActive);
        gpu_compilation_options.WaitForWeightsConversionComplete(
            advanced_settings
                .wait_for_weights_conversion_complete_in_benchmark);
      }
      if (!advanced_settings.preferred_device_substr.empty()) {
        gpu_compilation_options.SetPreferredDeviceSubstr(
            advanced_settings.preferred_device_substr.c_str());
      }
      gpu_compilation_options.DisableShaderOptimization(
          !advanced_settings.optimize_shader_compilation);
      // TODO b/441627719 - Select backend by runtime options.
#if defined(LITERT_USE_WEBGPU_ACCELERATOR)
      gpu_compilation_options.SetBackend(GpuOptions::Backend::kWebGpu);
#endif  // defined(LITERT_USE_WEBGPU_ACCELERATOR)
      // Prepare WebGPU or Vulkan command buffers ahead to reduce the overhead
      // of command buffer preparation. 2 steps ahead because KV cache is
      // swapped and the GPU resource bindings are the same as the previous
      // previous step.
      gpu_compilation_options.SetNumStepsOfCommandBufferPreparations(2);
      gpu_compilation_options.SetNumThreadsToUpload(
          advanced_settings.num_threads_to_upload >= 0
              ? advanced_settings.num_threads_to_upload
              : kDefaultNumThreadsToUpload);
      gpu_compilation_options.SetNumThreadsToCompile(
          advanced_settings.num_threads_to_compile >= 0
              ? advanced_settings.num_threads_to_compile
              : kDefaultNumThreadsToCompile);
      compilation_options.SetHardwareAccelerators(HwAccelerators::kGpu);
      break;
    }
    case Backend::CPU: {
      use_fp16_precision = false;
      LITERT_ASSIGN_OR_RETURN(auto& cpu_compilation_options,
                              compilation_options.GetCpuOptions());
      ASSIGN_OR_RETURN(const auto cpu_config,
                       executor_settings.GetBackendConfig<CpuConfig>());
      const uint32_t num_threads = cpu_config.number_of_threads;
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
      } else {
        ABSL_LOG(WARNING) << "Can't use cache: " << weight_cache_file.status();
      }
      auto default_xnn_options = TfLiteXNNPackDelegateOptionsDefault();
      cpu_compilation_options.SetXNNPackFlags(
          default_xnn_options.flags |
          TFLITE_XNNPACK_DELEGATE_FLAG_ENABLE_LATEST_OPERATORS);
      LITERT_ASSIGN_OR_RETURN(auto& runtime_options,
                              compilation_options.GetRuntimeOptions());
      runtime_options.SetCompressQuantizationZeroPoints(true);
      compilation_options.SetHardwareAccelerators(HwAccelerators::kCpu);
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

  LITERT_ASSIGN_OR_RETURN(auto compiled_model,
                          WrappedCompiledModel::Create(env, litert_model->Get(),
                                                       compilation_options));

  absl::flat_hash_map<absl::string_view, TensorBuffer> decode_input_buffers;
  absl::flat_hash_map<absl::string_view, TensorBuffer> decode_output_buffers;
  for (auto input_name : decode_signature.InputNames()) {
    if (IsLoRAInputName(input_name)) {
      // We let LoraManager handle LoRA inputs.
      continue;
    }
    if (IsKVCacheTensor(input_name)) {
      continue;
    }
    LITERT_ASSIGN_OR_RETURN(
        auto input_buffer,
        compiled_model.CreateInputBuffer(kDecodeSignatureRunner, input_name));
    decode_input_buffers[input_name] = std::move(input_buffer);
  }
  auto output_names = decode_signature.OutputNames();
  for (int i = 0; i < output_names.size(); ++i) {
    auto output_name = output_names[i];
    if (IsKVCacheTensor(output_name)) {
      continue;
    }
    // If we are using the GPU sampler and the model is compiled with FP16
    // precision, we force the output logits to be FP16 as the
    // GPU sampler supports FP16 inputs.
    // If we use CPU sampler or the model is executed with FP32 / mixed
    // precision, we will keep the logits in FP32
    if (output_name == signatures.output_logits && use_fp16_precision &&
        sampler_backend == Backend::GPU) {
      LITERT_ASSIGN_OR_RETURN(
          size_t signature_index,
          compiled_model.GetSignatureIndex(kDecodeSignatureRunner));
      LITERT_ASSIGN_OR_RETURN(
          auto output_buffer,
          CreateFP16OutputBuffer(env, compiled_model, signature_index,
                                 output_name, i));
      decode_output_buffers[output_name] = std::move(output_buffer);
    } else {
      LITERT_ASSIGN_OR_RETURN(auto output_buffer,
                              compiled_model.CreateOutputBuffer(
                                  kDecodeSignatureRunner, output_name));

      decode_output_buffers[output_name] = std::move(output_buffer);
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

  auto common_core = std::make_unique<LlmExecutorCommonCore>(
      std::move(executor_settings), env, *litert_model,
      std::move(compiled_model), std::move(decode_input_buffers),
      std::move(decode_output_buffers), signatures, std::move(embedding_lookup),
      std::move(per_layer_embedding_lookup), use_fp16_precision);

  return absl::WrapUnique(new LlmExecutorExternalSamplerStatic(
      std::move(common_core), activation_data_type,
      std::move(prefill_runner_set)));
}

absl::StatusOr<std::unique_ptr<KVCacheInterface>>
LlmExecutorExternalSamplerStatic::CreateKVCache() {
  return common_core_->CreateKVCache();
}

absl::StatusOr<int> LlmExecutorExternalSamplerStatic::LoadLoRA(
    const ModelAssets& model_assets) {
  return common_core_->LoadLoRA(model_assets);
}

absl::Status LlmExecutorExternalSamplerStatic::UnloadLoRA(int lora_id) {
  return common_core_->UnloadLoRA(lora_id);
}

absl::Status LlmExecutorExternalSamplerStatic::Cancel() {
  return common_core_->Cancel();
}

absl::Status LlmExecutorExternalSamplerStatic::Prefill(
    ExecutorInputs&& input_data, KVCacheInterface& kv_cache,
    std::optional<int> lora_id) {

  ASSIGN_OR_RETURN(const TensorBuffer* text_token_ids_ptr,
                   input_data.GetTextTokenIdsPtr());
  LITERT_ASSIGN_OR_RETURN(auto tensor_type, text_token_ids_ptr->TensorType());
  RET_CHECK_GT(tensor_type.Layout().Dimensions()[1], 0)
      << "Prefill token ids must be non-empty.";

  if (common_core_->embedding_lookup_ != nullptr) {
    RETURN_IF_ERROR(common_core_->embedding_lookup_->UpdateMultiModalEmbeddings(
        input_data));
  }

  auto litert_kv_cache = dynamic_cast<LitertKVCache*>(&kv_cache);
  RET_CHECK(litert_kv_cache != nullptr) << "Only support LitertKVCache.";

  LITERT_ASSIGN_OR_RETURN(
      auto ids, ReferTensorBufferAsSpan<int32_t>(*(text_token_ids_ptr)));
  // Reduce the input ids only with one user selected.
  ASSIGN_OR_RETURN(auto work_groups, GetOptimizedPrefillWorkGroups(
                                         prefill_signature_map_, ids.size()));
  int next_position = input_data.GetNextPosition();
  for (int i = 0; i < work_groups.size(); ++i) {
    const std::string& prefill_signature = work_groups[i].first;
    int prefill_length = work_groups[i].second;
    // Keep track of the signatures that have already had their buffers
    // created only create them once.
    if (!prefill_input_buffers_.contains(prefill_signature)) {
      prefill_input_buffers_[prefill_signature] = {};
      ASSIGN_OR_RETURN(prefill_input_buffers_[prefill_signature],
                       common_core_->CreatePrefillInputBuffers(
                           prefill_signature, prefill_length, prefill_length));
    }
    RETURN_IF_ERROR(common_core_->Prefill(
        prefill_signature, prefill_input_buffers_[prefill_signature],
        ids.subspan(/*pos=*/0, prefill_length), next_position, *litert_kv_cache,
        lora_id));
    ids = ids.subspan(/*pos=*/prefill_length);
    next_position += prefill_length;
  }
  RET_CHECK_EQ(ids.size(), 0).SetCode(absl::StatusCode::kInternal)
      << "Work groups not covering the entire prefill input.";

  if (common_core_->embedding_lookup_ != nullptr) {
    RETURN_IF_ERROR(
        common_core_->embedding_lookup_->CleanupMultiModalEmbeddings());
  }

  return absl::OkStatus();
}

absl::StatusOr<TensorBuffer> LlmExecutorExternalSamplerStatic::Step(
    ExecutorInputs&& input_data, KVCacheInterface& kv_cache,
    std::optional<int> lora_id) {

  auto litert_kv_cache = dynamic_cast<LitertKVCache*>(&kv_cache);
  RET_CHECK(litert_kv_cache != nullptr) << "Only support LitertKVCache.";

  int next_position = input_data.GetNextPosition();

  ASSIGN_OR_RETURN(const TensorBuffer* text_token_ids_ptr,
                   input_data.GetTextTokenIdsPtr());
  LITERT_ASSIGN_OR_RETURN(absl::Span<int> ids, ReferTensorBufferAsSpan<int32_t>(
                                                   *text_token_ids_ptr));
  RET_CHECK_EQ(ids.size(), 1) << "Expect 1 token id for decode.";

  ASSIGN_OR_RETURN(
      TensorBuffer output_logits,
      common_core_->Step(ids[0], next_position, *litert_kv_cache, lora_id));

  return output_logits;
}

/* ===========================================================================*/
/* LlmExecutorExternalSamplerDynamic */
/* ===========================================================================*/

absl::StatusOr<std::unique_ptr<LlmExecutorExternalSamplerDynamic>>
LlmExecutorExternalSamplerDynamic::Create(LlmExecutorSettings executor_settings,
                                          Environment& lrt_env,
                                          ModelResources& resources) {
  ASSIGN_OR_RETURN(auto litert_model,
                   resources.GetTFLiteModel(ModelType::kTfLitePrefillDecode));
  LITERT_ASSIGN_OR_RETURN(auto compilation_options, Options::Create());
  std::string weight_cache_path = executor_settings.GetCacheDir();
  const Backend backend = executor_settings.GetBackend();
  RET_CHECK_EQ(backend, Backend::CPU)
      << "LlmExecutorExternalSamplerDynamic only supports CPU backend.";
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
    auto default_xnn_options = TfLiteXNNPackDelegateOptionsDefault();
    cpu_compilation_options.SetXNNPackFlags(
        default_xnn_options.flags |
        TFLITE_XNNPACK_DELEGATE_FLAG_ENABLE_LATEST_OPERATORS);
    LITERT_ASSIGN_OR_RETURN(auto& runtime_options,
                            compilation_options.GetRuntimeOptions());
    runtime_options.SetCompressQuantizationZeroPoints(true);
    compilation_options.SetHardwareAccelerators(HwAccelerators::kCpu);
  }

  LITERT_ASSIGN_OR_RETURN(auto compiled_model, WrappedCompiledModel::Create(
                                                   lrt_env, litert_model->Get(),
                                                   compilation_options));

  absl::flat_hash_map<absl::string_view, TensorBuffer> decode_input_buffers;
  absl::flat_hash_map<absl::string_view, TensorBuffer> decode_output_buffers;

  LITERT_ASSIGN_OR_RETURN(auto decode_signature,
                          litert_model->FindSignature(kDecodeSignatureRunner));
  ASSIGN_OR_RETURN(
      ModelSignatures signatures,
      GetModelSignaturesFromInputOutputNames(decode_signature.InputNames(),
                                             decode_signature.OutputNames()));

  for (auto input_name : decode_signature.InputNames()) {
    bool is_attn_mask_input =
        signatures.input_attn_mask.has_value() &&
        absl::StartsWith(input_name, signatures.input_attn_mask.value());
    if (!IsKVCacheTensor(input_name) && !is_attn_mask_input) {
      LITERT_ASSIGN_OR_RETURN(
          auto input_buffer,
          compiled_model.CreateInputBuffer(kDecodeSignatureRunner, input_name));
      decode_input_buffers[input_name] = std::move(input_buffer);
    }
  }
  for (auto output_name : decode_signature.OutputNames()) {
    if (!IsKVCacheTensor(output_name)) {
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
  RET_CHECK_EQ(batch_size, 1) << "Only support batch size 1 for now.";
  std::unique_ptr<EmbeddingLookupManager> embedding_lookup;
  std::unique_ptr<EmbeddingLookupManager> per_layer_embedding_lookup;
  RETURN_IF_ERROR(InitializeEmbeddingLookups(resources, embedding_lookup,
                                             per_layer_embedding_lookup));

  auto common_core = std::make_unique<LlmExecutorCommonCore>(
      std::move(executor_settings), lrt_env, *litert_model,
      std::move(compiled_model), std::move(decode_input_buffers),
      std::move(decode_output_buffers), signatures, std::move(embedding_lookup),
      std::move(per_layer_embedding_lookup),
      /*use_fp16_precision=*/false);

  return absl::WrapUnique(new LlmExecutorExternalSamplerDynamic(
      std::move(common_core), prefill_chunk_size, kv_increament_size));
}

absl::StatusOr<std::unique_ptr<KVCacheInterface>>
LlmExecutorExternalSamplerDynamic::CreateKVCache() {
  return common_core_->CreateKVCache();
}

absl::StatusOr<int> LlmExecutorExternalSamplerDynamic::LoadLoRA(
    const ModelAssets& model_assets) {
  return common_core_->LoadLoRA(model_assets);
}

absl::Status LlmExecutorExternalSamplerDynamic::UnloadLoRA(int lora_id) {
  return common_core_->UnloadLoRA(lora_id);
}

absl::Status LlmExecutorExternalSamplerDynamic::Cancel() {
  return common_core_->Cancel();
}

absl::Status LlmExecutorExternalSamplerDynamic::Prefill(
    ExecutorInputs&& input_data, KVCacheInterface& kv_cache,
    std::optional<int> lora_id) {

  ASSIGN_OR_RETURN(const TensorBuffer* text_token_ids_ptr,
                   input_data.GetTextTokenIdsPtr());
  LITERT_ASSIGN_OR_RETURN(auto tensor_type, text_token_ids_ptr->TensorType());
  RET_CHECK_EQ(tensor_type.Layout().Dimensions()[0], 1);
  RET_CHECK_GT(tensor_type.Layout().Dimensions()[1], 0)
      << "Prefill token ids must be non-empty.";
  LITERT_ASSIGN_OR_RETURN(absl::Span<int> ids, ReferTensorBufferAsSpan<int32_t>(
                                                   *text_token_ids_ptr));
  int next_position = input_data.GetNextPosition();

  auto litert_kv_cache = dynamic_cast<LitertKVCache*>(&kv_cache);
  RET_CHECK(litert_kv_cache != nullptr) << "Only support LitertKVCache.";

  while (!ids.empty()) {
    int chunk_size =
        prefill_chunk_size_ > 0
            ? std::min(static_cast<int>(ids.size()), prefill_chunk_size_)
            : static_cast<int>(ids.size());
    absl::Span<int> chunk_ids = ids.first(chunk_size);
    ids = ids.subspan(chunk_size);

    int kv_length = kv_cache.GetNumEntries();
    int free_kv_entries = kv_length - next_position;
    int prefill_length = chunk_ids.size();
    if (prefill_length > free_kv_entries) {
      int new_kv_seq_len = kv_length + prefill_length - free_kv_entries;
      RETURN_IF_ERROR(litert_kv_cache->Resize(new_kv_seq_len));
      kv_length = new_kv_seq_len;
    }

    LITERT_ASSIGN_OR_RETURN(auto prefill_input_buffers,
                            common_core_->CreatePrefillInputBuffers(
                                "prefill", prefill_length, kv_length));

    RETURN_IF_ERROR(common_core_->Prefill("prefill", prefill_input_buffers,
                                          chunk_ids, next_position,
                                          *litert_kv_cache, lora_id));
    next_position += chunk_size;
  }

  return absl::OkStatus();
}

absl::StatusOr<TensorBuffer> LlmExecutorExternalSamplerDynamic::Step(
    ExecutorInputs&& input_data, KVCacheInterface& kv_cache,
    std::optional<int> lora_id) {

  auto litert_kv_cache = dynamic_cast<LitertKVCache*>(&kv_cache);
  RET_CHECK(litert_kv_cache != nullptr) << "Only support LitertKVCache.";

  int current_kv_len = kv_cache.GetNumEntries();
  int next_position = input_data.GetNextPosition();
  if (current_kv_len <= next_position) {
    RETURN_IF_ERROR(
        litert_kv_cache->Resize(current_kv_len + kv_increament_size_));
    current_kv_len = kv_cache.GetNumEntries();
  }

  RETURN_IF_ERROR(ResolveDynamicShape(
      common_core_->model_, common_core_->compiled_model_, "decode",
      common_core_->signatures_.input_attn_mask.value(), current_kv_len));
  LITERT_ASSIGN_OR_RETURN(
      common_core_->decode_input_buffers_[common_core_->signatures_
                                              .input_attn_mask.value()],
      common_core_->compiled_model_.CreateInputBuffer(
          "decode", common_core_->signatures_.input_attn_mask.value()));

  ASSIGN_OR_RETURN(const TensorBuffer* text_token_ids_ptr,
                   input_data.GetTextTokenIdsPtr());
  LITERT_ASSIGN_OR_RETURN(absl::Span<int> ids, ReferTensorBufferAsSpan<int32_t>(
                                                   *text_token_ids_ptr));
  RET_CHECK_EQ(ids.size(), 1) << "Expect 1 token id for decode.";

  return common_core_->Step(ids[0], next_position, *litert_kv_cache, lora_id);
}

/* ===========================================================================*/
/* LlmExecutorInternalSamplerStatic */
/* ===========================================================================*/

absl::StatusOr<std::unique_ptr<LlmExecutorInternalSamplerStatic>>
LlmExecutorInternalSamplerStatic::Create(LlmExecutorSettings executor_settings,
                                         Environment& env,
                                         ModelResources& resources) {
  ASSIGN_OR_RETURN(auto external_sampler_static_executor,
                   LlmExecutorExternalSamplerStatic::Create(
                       std::move(executor_settings), env, resources));

  std::unique_ptr<Sampler> sampler;
  {
    ASSIGN_OR_RETURN(
        int vocab_size,
        external_sampler_static_executor->common_core_->InferVocabSize());
    ASSIGN_OR_RETURN(auto sampler_backend,
                     GetSamplerBackend(external_sampler_static_executor
                                           ->common_core_->executor_settings_));
    proto::SamplerParameters sampler_params;
    sampler_params.set_type(proto::SamplerParameters::TOP_P);
    sampler_params.set_k(1);
    sampler_params.set_p(0.0f);
    sampler_params.set_temperature(1.0f);
    sampler_params.set_seed(0);
    ASSIGN_OR_RETURN(
        sampler,
        CreateSampler(sampler_backend, /*batch_size=*/1,
                      std::move(sampler_params), env.Get(), vocab_size,
                      external_sampler_static_executor->logits_data_type_));
  }

  return absl::WrapUnique(new LlmExecutorInternalSamplerStatic(
      std::move(external_sampler_static_executor), std::move(sampler)));
}

absl::StatusOr<std::unique_ptr<KVCacheInterface>>
LlmExecutorInternalSamplerStatic::CreateKVCache() {
  return external_sampler_static_executor_->CreateKVCache();
}

absl::StatusOr<int> LlmExecutorInternalSamplerStatic::LoadLoRA(
    const ModelAssets& model_assets) {
  return external_sampler_static_executor_->LoadLoRA(model_assets);
}

absl::Status LlmExecutorInternalSamplerStatic::UnloadLoRA(int lora_id) {
  return external_sampler_static_executor_->UnloadLoRA(lora_id);
}

absl::Status LlmExecutorInternalSamplerStatic::Cancel() {
  return external_sampler_static_executor_->Cancel();
}

absl::Status LlmExecutorInternalSamplerStatic::Prefill(
    ExecutorInputs&& input_data, KVCacheInterface& kv_cache,
    std::optional<int> lora_id) {
  return external_sampler_static_executor_->Prefill(std::move(input_data),
                                                    kv_cache, lora_id);
}

absl::StatusOr<std::vector<std::vector<int>>>
LlmExecutorInternalSamplerStatic::SampleTokens(int num_steps,
                                               ExecutorInputs&& input_data,
                                               KVCacheInterface& kv_cache,
                                               std::optional<int> lora_id) {
  std::vector<std::vector<int>> sampled_tokens = {{}};
  sampled_tokens[0].reserve(num_steps);
  int starting_position = input_data.GetNextPosition();
  std::optional<ExecutorInputs> step_inputs = std::move(input_data);
  for (int i = 0; i < num_steps; ++i) {
    step_inputs->SetNextPosition(starting_position + i);
    ASSIGN_OR_RETURN(auto logits,
                     external_sampler_static_executor_->Step(
                         std::move(step_inputs.value()), kv_cache, lora_id));
    LITERT_ASSIGN_OR_RETURN(
        TensorBuffer ids_tensor,
        litert::TensorBuffer::CreateManaged(
            external_sampler_static_executor_->common_core_->env_,
            litert::TensorBufferType::kHostMemory,
            litert::RankedTensorType(litert::ElementType::Int32,
                                     litert::Layout(litert::Dimensions({1}))),
            /*buffer_size=*/sizeof(int32_t)));
    RETURN_IF_ERROR(sampler_->SampleToIdAndScoreBuffer(
        logits, ids_tensor, /*scores_tensor=*/nullptr));
    LITERT_ASSIGN_OR_RETURN(absl::Span<int> sampled_token,
                            ReferTensorBufferAsSpan<int32_t>(ids_tensor));
    sampled_tokens[0].push_back(sampled_token[0]);

    step_inputs = ExecutorInputs();
    step_inputs->SetTextData(ExecutorTextData(std::move(ids_tensor)));
  }
  return sampled_tokens;
}

}  // namespace litert::lm
