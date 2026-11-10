// Copyright 2024 The ODML Authors.
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

#include "runtime/executor/gpu_artisan/llm_gpu_artisan_executor.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/constrained_decoding/constrained_decoder.h"
#include "runtime/components/embedding_lookup/embedding_lookup_end_of_multi_modal.h"
#include "runtime/components/model_data_provider.h"
#include "runtime/components/model_resources.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/gpu_artisan/gpu_artisan_executor_utils.h"
#include "runtime/executor/gpu_artisan/llm_gpu_artisan_processed_context.h"
#include "runtime/executor/gpu_artisan/ml_drift/gpu_config_converters.h"
#include "runtime/executor/gpu_artisan/ml_drift/gpu_model_info_generator.h"
#include "runtime/executor/gpu_artisan/ml_drift/litert_thread_pool_executor.h"
#include "runtime/executor/gpu_artisan/ml_drift/llm.h"
#include "runtime/executor/gpu_artisan/ml_drift/llm_gpu_runner.h"
#include "runtime/executor/gpu_artisan/ml_drift/speculative_utils.h"
#include "runtime/executor/gpu_artisan/ml_drift/tensor_loaders/gpu_optimized_tensor_loader.h"
#include "runtime/executor/gpu_artisan/ml_drift/tensor_loaders/llm_tensor_loader.h"
#include "runtime/executor/gpu_artisan/ml_drift/transformer.h"
#include "runtime/executor/gpu_artisan/proto/llm_inference_engine.pb.h"
#include "runtime/executor/gpu_artisan/proto/llm_params.pb.h"
#include "runtime/executor/gpu_artisan/utils/bitmap.h"
#include "runtime/executor/gpu_artisan/utils/constraint_adapter.h"
#include "runtime/executor/gpu_artisan/utils/model_data.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/executor/llm_executor_processed_tokens.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/status_macros.h"

namespace odml::infra {
namespace {

using ::litert::lm::ExecutorAudioData;
using ::litert::lm::ExecutorInputs;
using ::litert::lm::ExecutorPrefillParams;
using ::litert::lm::ExecutorTextData;
using ::litert::lm::ExecutorVisionData;
using ::litert::lm::GpuArtisanConfig;
using ::litert::lm::LlmContext;
using ::litert::lm::LlmExecutorSettings;
using ::litert::lm::ModelAssets;
using ::litert::lm::ModelDataProvider;
using ::litert::lm::ReferTensorBufferAsSpan;
using ::litert::lm::RuntimeConfig;
using ::litert::lm::RuntimeState;
namespace oi = ::odml::infra;

bool SupportsPeekNextToken(gpu::LlmGpuRunnerType runner_type) {
  return runner_type == gpu::LlmGpuRunnerType::kWebGpu ||
         runner_type == gpu::LlmGpuRunnerType::kWebGpu_Emscripten ||
         runner_type == gpu::LlmGpuRunnerType::kMetal;
}

// Converts GpuArtisanConfig to MlDriftOptions with default values.
proto::SessionConfig::MlDriftOptions GetMlDriftOptionsFromArtisanConfig(
    GpuArtisanConfig gpu_artisan_config) {
  proto::SessionConfig::MlDriftOptions mldrift_options;
  mldrift_options.set_num_output_candidates(
      gpu_artisan_config.num_output_candidates);
  mldrift_options.set_num_decode_steps_per_sync(
      gpu_artisan_config.num_decode_steps_per_sync);
  mldrift_options.set_sequence_batch_size(
      gpu_artisan_config.sequence_batch_size);
  mldrift_options.mutable_supported_lora_ranks()->Assign(
      gpu_artisan_config.supported_lora_ranks.begin(),
      gpu_artisan_config.supported_lora_ranks.end());
  mldrift_options.set_max_top_k(gpu_artisan_config.max_top_k);
  mldrift_options.set_wait_for_weight_uploads(
      gpu_artisan_config.wait_for_weight_uploads);
  mldrift_options.set_enable_fast_tuning(true);
  mldrift_options.set_prefer_texture_weights(
      gpu_artisan_config.prefer_texture_weights);
  mldrift_options.set_enable_host_mapped_pointer(
      gpu_artisan_config.set_enable_host_mapped_pointer);
  mldrift_options.set_disallow_8bit_convs(
      gpu_artisan_config.disallow_8bit_convs);
  return mldrift_options;
}

// Computes the embedding for `end_token` (e.g. end-of-vision or end-of-audio)
// from the given end-of-multimodal embedding model.
absl::StatusOr<std::vector<float>> ComputeEndOfMultiModalEmbedding(
    const ::litert::Model* absl_nonnull end_of_multi_modal_model,
    int end_token, int embedding_dimension) {
  LITERT_ASSIGN_OR_RETURN(auto env, ::litert::Environment::Create({}));
  LITERT_ASSIGN_OR_RETURN(
      auto end_of_multi_modal_lookup,
      ::litert::lm::EndOfMultiModalEmbedding::Create(
          env, end_of_multi_modal_model, end_token));
  // LookupPrefill requires the embedding vector to be pre-allocated to the
  // correct size.
  std::vector<float> embedding(embedding_dimension);
  LITERT_RETURN_IF_ERROR(
      end_of_multi_modal_lookup->LookupPrefill(end_token, embedding));
  return embedding;
}

// The vision embeddings of a single image, padded to the size of one image
// slot in the media embedding cache.
struct SingleImageEmbeddings {
  // Shape [1, 1, max_num_vision_tokens_per_image, embedding_dim].
  ml_drift::TensorFloat32 embedding;
  // Each with shape [1, 1, max_num_vision_tokens_per_image, ple_dim].
  std::vector<ml_drift::TensorFloat32> per_layer_embeddings;
};

// Creates the embeddings of a single image from the embeddings of all images.
// `token_offset` is the index of the first token of the image in `embedding`
// and `per_layer_embeddings`. If the image has an end-of-image token,
// `end_of_vision_embedding` is placed right after the image's active tokens.
absl::StatusOr<SingleImageEmbeddings> CreateSingleImageEmbeddings(
    const ml_drift::TensorFloat32& embedding,
    absl::Span<const ml_drift::TensorFloat32> per_layer_embeddings,
    const ImageTokenInfo& image_info, int token_offset,
    int max_num_vision_tokens_per_image,
    absl::Span<const float> end_of_vision_embedding) {
  const int active_tokens = image_info.active_tokens;
  SingleImageEmbeddings result;

  // Create a single embedding tensor for the current image, with the shape
  // [1, 1, max_num_vision_tokens_per_image, embedding_dim], and copy the
  // corresponding tokens from the input embedding tensor to the single
  // embedding tensor. The input embedding tensor has the shape [1, 1,
  // num_active_tokens, embedding_dim].
  ml_drift::TensorFloat32& single_embedding = result.embedding;
  single_embedding.shape = embedding.shape;
  single_embedding.shape.w = max_num_vision_tokens_per_image;
  single_embedding.data.assign(single_embedding.shape.DimensionsProduct(),
                               0.0f);

  const int embedding_channels = embedding.shape.c;
  if (embedding.Data() != nullptr) {
    std::copy(
        embedding.Data() + token_offset * embedding_channels,
        embedding.Data() + (token_offset + active_tokens) * embedding_channels,
        single_embedding.data.begin());
  }
  if (image_info.has_eoi) {
    RET_CHECK_LT(active_tokens, max_num_vision_tokens_per_image)
        << "EndOfVision token index exceeds maximum allowed per image.";
    RET_CHECK_EQ(end_of_vision_embedding.size(), embedding_channels)
        << "EndOfVision embedding size mismatch.";
    absl::c_copy(
        end_of_vision_embedding,
        single_embedding.data.begin() + active_tokens * embedding_channels);
  }

  result.per_layer_embeddings.reserve(per_layer_embeddings.size());
  for (const auto& ple : per_layer_embeddings) {
    // Create a single PLE tensor for the current image, with the shape
    // [1, 1, max_num_vision_tokens_per_image, ple_dim], and copy the
    // corresponding tokens from the input PLE tensor to the single PLE
    // tensor.
    ml_drift::TensorFloat32& single_ple =
        result.per_layer_embeddings.emplace_back();
    single_ple.shape = ple.shape;
    single_ple.shape.w = max_num_vision_tokens_per_image;
    single_ple.data.assign(single_ple.shape.DimensionsProduct(), 0.0f);

    const int ple_channels = ple.shape.c;
    if (ple.Data() != nullptr) {
      std::copy(ple.Data() + token_offset * ple_channels,
                ple.Data() + (token_offset + active_tokens) * ple_channels,
                single_ple.data.begin());
    }
  }
  return result;
}

}  // namespace

absl::StatusOr<std::unique_ptr<LlmGpuArtisanExecutor>>
LlmGpuArtisanExecutor::Create(
    const ::litert::lm::LlmExecutorSettings& executor_settings,
    ::litert::lm::ModelResources& model_resources) {
  auto& model_data_provider =
      dynamic_cast<::litert::lm::ModelDataProvider&>(model_resources);
  ASSIGN_OR_RETURN(auto model_data,
                   model_data_provider.GetSharedArtisanModeldata());
  const int embedding_dimension =
      model_data->GetLlmParameters().transformer_parameters().embedding_dim();

  std::vector<float> end_of_vision_embedding;
  auto end_of_vision_model = model_resources.GetTFLiteModel(
      ::litert::lm::proto::LlmMetadata::TF_LITE_END_OF_VISION);
  if (end_of_vision_model.ok()) {
    ASSIGN_OR_RETURN(end_of_vision_embedding,
                     ComputeEndOfVisionEmbedding(end_of_vision_model.value(),
                                                 embedding_dimension));
  }

  std::vector<float> end_of_audio_embedding;
  auto end_of_audio_model = model_resources.GetTFLiteModel(
      ::litert::lm::proto::LlmMetadata::TF_LITE_END_OF_AUDIO);
  if (end_of_audio_model.ok()) {
    ASSIGN_OR_RETURN(end_of_audio_embedding,
                     ComputeEndOfAudioEmbedding(end_of_audio_model.value(),
                                                embedding_dimension));
  }

  // If the audio encoder is not part of the artisan model (e.g. a LiteRT audio
  // encoder in a separate section), the artisan model doesn't know how many
  // audio embeddings to reserve room for, so reserve a default amount.
  int max_num_audio_tokens = 0;
  if (!model_data->GetLlmParameters().has_audio_encoder_parameters() &&
      model_resources
          .GetTFLiteModel(
              ::litert::lm::proto::LlmMetadata::TF_LITE_AUDIO_ENCODER_HW)
          .ok()) {
    const int max_num_tokens = executor_settings.GetMaxNumTokens();
    max_num_audio_tokens =
        max_num_tokens > 0
            ? std::min<int>(kDefaultMaxNumExternalAudioTokens, max_num_tokens)
            : kDefaultMaxNumExternalAudioTokens;
  }

  return Create(executor_settings, std::move(model_data),
                std::move(end_of_vision_embedding),
                std::move(end_of_audio_embedding), max_num_audio_tokens);
}

absl::StatusOr<std::vector<float>>
LlmGpuArtisanExecutor::ComputeEndOfVisionEmbedding(
    const ::litert::Model* absl_nonnull end_of_vision_model,
    int embedding_dimension) {
  return ComputeEndOfMultiModalEmbedding(
      end_of_vision_model, ::litert::lm::ExecutorVisionData::kEndToken,
      embedding_dimension);
}

absl::StatusOr<std::vector<float>>
LlmGpuArtisanExecutor::ComputeEndOfAudioEmbedding(
    const ::litert::Model* absl_nonnull end_of_audio_model,
    int embedding_dimension) {
  return ComputeEndOfMultiModalEmbedding(
      end_of_audio_model, ::litert::lm::ExecutorAudioData::kEndToken,
      embedding_dimension);
}

absl::StatusOr<std::unique_ptr<LlmGpuArtisanExecutor>>
LlmGpuArtisanExecutor::Create(
    const ::litert::lm::LlmExecutorSettings& executor_settings,
    std::shared_ptr<odml::infra::llm_utils::ModelData> model_data,
    std::vector<float> end_of_vision_embedding,
    std::vector<float> end_of_audio_embedding, int max_num_audio_tokens) {
  ABSL_LOG(INFO) << "LlmGpuArtisanExecutor::Create with the following config: ";
  ABSL_LOG(INFO) << "  LlmExecutorSettings: " << executor_settings;

  ASSIGN_OR_RETURN(const GpuArtisanConfig gpu_artisan_config,
                   executor_settings.GetBackendConfig<GpuArtisanConfig>());

  // Create ml_drift::LlmConfig from the LlmParameters (stored in model_data).
  ASSIGN_OR_RETURN(ml_drift::LlmConfig gpu_llm_config,
                   oi::gpu::CreateGpuConfigFromLlmParametersProto(
                       model_data->GetLlmParameters(),
                       /*use_submodel=*/gpu_artisan_config.use_submodel));
  if (auto model_type = model_data->GetModelType(); model_type.has_value()) {
    gpu_llm_config.model_type = *model_type;
  }
  gpu_llm_config.text_decoder->cache_size = executor_settings.GetMaxNumTokens();
  gpu_llm_config.text_decoder->max_num_images =
      executor_settings.GetMaxNumImages();
  gpu_llm_config.text_decoder->max_num_audio_tokens = max_num_audio_tokens;
  for (auto lora_rank : gpu_artisan_config.supported_lora_ranks) {
    gpu_llm_config.text_decoder->lora_ranks.push_back(
        static_cast<int>(lora_rank));
  }
  gpu_llm_config.text_decoder->max_top_k = gpu_artisan_config.max_top_k;
  gpu_llm_config.text_decoder->use_autosized_ringbuffers =
      gpu_artisan_config.use_autosized_ringbuffers;

  // Create tensor_loader from model_data.
  auto tensor_loader = std::make_unique<oi::gpu::GpuOptimizedTensorLoader>(
      std::move(model_data), /*lora_data=*/nullptr, /*own_model_data=*/false);

  // Convert GpuArtisanConfig to MlDriftOptions with default values.
  proto::SessionConfig::MlDriftOptions mldrift_options =
      GetMlDriftOptionsFromArtisanConfig(gpu_artisan_config);

  auto activation_data_type = proto::SessionConfig::ACTIVATION_DATA_TYPE_F16;
  if (executor_settings.GetActivationDataType().has_value()) {
    activation_data_type =
        static_cast<proto::SessionConfig::ActivationDataType>(
            executor_settings.GetActivationDataType().value());
  }
  ml_drift::CreateGpuModelInfo create_info =
      oi::gpu::GenerateGpuModelInfo(mldrift_options, activation_data_type);

  // Create llm_gpu_runner from the above assets/configs.
#ifdef __EMSCRIPTEN__
  return absl::UnimplementedError(
      "GPU Artisan Emscripten WebGPU device initialization is not yet wired "
      "in OSS.");
#else
  ASSIGN_OR_RETURN(
      auto llm_gpu_runner,
      gpu::LlmGpuRunner::Create(
          gpu_llm_config,
          gpu::LlmTensorLoader::MakeCaching(std::move(tensor_loader)),
          std::move(create_info), /*num_output_heads=*/1,
          gpu_artisan_config.wait_for_weight_uploads,
          std::make_shared<gpu::LitertThreadPoolExecutorFactory>()));
#endif  // __EMSCRIPTEN__
  return std::make_unique<LlmGpuArtisanExecutor>(
      executor_settings, std::move(llm_gpu_runner),
      gpu_llm_config.text_decoder.value(), std::move(end_of_vision_embedding),
      std::move(end_of_audio_embedding));
}

absl::Status LlmGpuArtisanExecutor::Prefill(const ExecutorInputs& inputs) {
  return LlmGpuArtisanExecutor::Prefill(inputs, ExecutorPrefillParams());
}

absl::Status LlmGpuArtisanExecutor::Prefill(
    const ExecutorInputs& inputs, const ExecutorPrefillParams& prefill_params) {
  // If the cancel flag is set and the cancel flag is true, immediately return.
  if (prefill_params.GetCancelFlag() != nullptr &&
      *prefill_params.GetCancelFlag()) {
    return absl::OkStatus();
  }
  ABSL_VLOG(2)
      << "LlmGpuArtisanExecutor::Prefill with the following arguments: ";
  ABSL_VLOG(2) << "  inputs: " << inputs;
  ABSL_VLOG(2) << "  prefill_params: " << prefill_params;
  RET_CHECK_NE(llm_gpu_runner_, nullptr)
      << "The LlmGpuArtisanExecutor's llm_gpu_runner_ is null, not created "
         "correctly.";
  RET_CHECK_NE(llm_context_, nullptr)
      << "The llm_context_ should not be nullptr.";

  LITERT_ASSIGN_OR_RETURN(auto input_ids_type,
                          (*inputs.GetTextTokenIdsPtr())->TensorType());
  // Check if the input token is 1 batch. Currently only support 1 batch per
  // prefill.
  RET_CHECK_EQ(input_ids_type.Layout().Dimensions()[0], 1)
      << "Currently only support 1 batch per prefill.";

  if (prefill_params.GetCurrentStep() != -1) {
    llm_context_->runtime_state().current_step =
        prefill_params.GetCurrentStep();
  }
  RET_CHECK_LE(
      llm_context_->runtime_state().current_step,
      llm_context_->processed_context().processed_tokens().TokenCount())
      << "The current_step should not be larger than the number of "
         "processed_tokens.";
  if (llm_context_->processed_context().processed_tokens().TokenCount() !=
      llm_context_->runtime_state().current_step) {
    RETURN_IF_ERROR(llm_gpu_runner_->CleanCache(
        llm_context_->runtime_state().current_step));
    RETURN_IF_ERROR(
        llm_context_->processed_context().processed_tokens().RollBackToStep(
            llm_context_->runtime_state().current_step));
  }

  if (llm_context_->runtime_config().constraint) {
    const auto& constraint = *llm_context_->runtime_config().constraint;
    llm_context_->runtime_state().constraint_state = constraint->Start();
    ASSIGN_OR_RETURN(auto bitmap,
                     constraint->ComputeBitmap(
                         *llm_context_->runtime_state().constraint_state));
    RETURN_IF_ERROR(llm_gpu_runner_->UpdateConstraintMask(*bitmap));
  } else {
    // An unconstrained prefill starts a new turn: drop any constraint state
    // left over from a previous constrained decode so that the next
    // constrained Decode() re-initializes it from the current constraint and
    // re-samples the lookahead token under the new constraint mask.
    llm_context_->runtime_state().constraint_state = nullptr;
  }

  LITERT_ASSIGN_OR_RETURN(
      auto input_ids_span,
      ReferTensorBufferAsSpan<int32_t>(*(*inputs.GetTextTokenIdsPtr())));
  const std::vector<int> input_ids_vec(input_ids_span.begin(),
                                       input_ids_span.end());

  // The audio embeddings can come from either an artisan audio encoder (with
  // shape [1, 1, num_audio_embeddings, 1, embedding_dim]) or a LiteRT audio
  // encoder (with shape [1, num_audio_embeddings, embedding_dim]). The audio
  // embeddings of all the audio clips of the input are concatenated in order.
  absl::Span<const float> audio_embeddings;
  int audio_embedding_dimension = 0;
  if (inputs.GetAudioDataPtr().ok()) {
    ASSIGN_OR_RETURN(auto projected_audio_embeddings_tensor,
                     inputs.GetProjectedAudioEmbeddingsPtr());
    LITERT_ASSIGN_OR_RETURN(auto audio_tensor_type,
                            projected_audio_embeddings_tensor->TensorType());
    const auto audio_dimensions = audio_tensor_type.Layout().Dimensions();
    RET_CHECK(!audio_dimensions.empty() && audio_dimensions.back() > 0)
        << "Invalid audio embeddings shape.";
    audio_embedding_dimension = audio_dimensions.back();
    LITERT_ASSIGN_OR_RETURN(
        audio_embeddings,
        ReferTensorBufferAsSpan<float>(*projected_audio_embeddings_tensor));
  }

  const int max_num_vision_tokens_per_image = text_config_.vision_tokens_num;
  std::optional<ml_drift::TensorFloat32> embedding;
  std::vector<ml_drift::TensorFloat32> ple_vec;
  std::vector<ImageTokenInfo> image_infos;
  if (inputs.GetVisionDataPtr().ok()) {
    ASSIGN_OR_RETURN(auto vision_embeddings_tensor,
                     inputs.GetVisionEmbeddingsPtr());

    ASSIGN_OR_RETURN(embedding,
                     CreateMLDriftTensorEmbeddingFromLitertTensorBuffer(
                         *vision_embeddings_tensor));

    const int num_vision_tokens =
        std::count(input_ids_vec.begin(), input_ids_vec.end(),
                   ExecutorVisionData::kSpecialToken);

    RET_CHECK_EQ(num_vision_tokens, embedding->shape.w)
        << "The number of vision tokens should be the same as the number of "
           "vision embeddings, but got "
        << num_vision_tokens << " vs " << embedding->shape.w;

    if (inputs.GetVisionPerLayerEmbeddingsPtr().ok()) {
      ASSIGN_OR_RETURN(ple_vec, CreateMLDriftTensorPLEFromLitertTensorBuffer(*(
                                    *inputs.GetVisionPerLayerEmbeddingsPtr())));
    }

    image_infos =
        GetImageTokenInfos(input_ids_vec, ExecutorVisionData::kSpecialToken,
                           ExecutorVisionData::kEndToken);
    const int num_images = image_infos.size();

    RET_CHECK_LE(num_vision_tokens,
                 num_images * max_num_vision_tokens_per_image)
        << "The number of vision tokens should be less than or equal to the "
           "number of images times the number of vision tokens per image.";
    for (int i = 0; i < num_images; ++i) {
      RET_CHECK_LE(image_infos[i].total_tokens(),
                   max_num_vision_tokens_per_image)
          << "Image " << i << " has " << image_infos[i].total_tokens()
          << " total vision tokens (including end), which exceeds the maximum "
             "allowed "
          << max_num_vision_tokens_per_image;
    }
  }

  // The media embedding cache only holds `max_num_images` images, and the audio
  // embeddings of a prefill are stored after its images. So split the input
  // into chunks with at most `max_num_images` images and at most one audio
  // clip each, so that the media embeddings of each chunk fit in the cache, and
  // prefill them one after another. Images and audio clips are never split.
  const std::vector<PrefillChunkRange> chunks = SplitTokensForPrefill(
      input_ids_vec, ExecutorVisionData::kSpecialToken,
      ExecutorAudioData::kSpecialToken, ExecutorAudioData::kEndToken,
      text_config_.max_num_images, max_num_vision_tokens_per_image,
      text_config_.max_num_audio_tokens);

  int vision_token_offset = 0;
  for (int i = 0; i < chunks.size(); ++i) {
    // If the cancel flag is set and the cancel flag is true, stop prefilling
    // the remaining chunks.
    if (prefill_params.GetCancelFlag() != nullptr &&
        *prefill_params.GetCancelFlag()) {
      return absl::OkStatus();
    }
    const PrefillChunkRange& chunk = chunks[i];
    const bool is_last_chunk = i + 1 == chunks.size();
    const absl::Span<const ImageTokenInfo> chunk_image_infos =
        absl::MakeConstSpan(image_infos)
            .subspan(chunk.image_begin, chunk.image_end - chunk.image_begin);
    const PrefillChunkVisionInputs vision_inputs = {
        .embeddings = embedding.has_value() ? &*embedding : nullptr,
        .per_layer_embeddings = ple_vec,
        .image_infos = chunk_image_infos,
        .token_offset = vision_token_offset,
    };

    // Pass the audio embeddings of the chunk's audio clip, if any, starting
    // from the embedding of its first audio token.
    PrefillChunkAudioInputs audio_inputs;
    if (!audio_embeddings.empty() && chunk.HasAudio()) {
      const size_t audio_begin = static_cast<size_t>(chunk.audio_token_begin) *
                                 audio_embedding_dimension;
      RET_CHECK_LT(audio_begin, audio_embeddings.size())
          << "There are more audio special tokens than audio embeddings.";
      audio_inputs = {
          .embeddings = audio_embeddings.subspan(audio_begin),
          .embedding_dimension = audio_embedding_dimension,
      };
    }

    // The next chunk overwrites the media embedding cache, so wait for every
    // chunk but the last to complete before filling the next one.
    RETURN_IF_ERROR(PrefillChunk(
        absl::MakeConstSpan(input_ids_vec)
            .subspan(chunk.token_begin, chunk.token_end - chunk.token_begin),
        vision_inputs, audio_inputs,
        /*wait_for_completion=*/
        is_last_chunk ? prefill_params.GetWaitForCompletion() : true,
        prefill_params.GetCancelFlag()));
    for (const ImageTokenInfo& image_info : chunk_image_infos) {
      vision_token_offset += image_info.active_tokens;
    }
  }
  return absl::OkStatus();
}

absl::Status LlmGpuArtisanExecutor::PrefillChunk(
    absl::Span<const int> input_ids,
    const PrefillChunkVisionInputs& vision_inputs,
    const PrefillChunkAudioInputs& audio_inputs, bool wait_for_completion,
    const std::atomic_bool* absl_nullable cancel) {
  std::vector<int> input_ids_vec(input_ids.begin(), input_ids.end());
  std::vector<int> input_ids_vec_with_special_tokens_replaced = input_ids_vec;

  // If there is no end-of-audio embedding (from an end-of-audio model), the
  // end-of-audio token is looked up in the text embedding table.
  // TODO - b/424634660: Remove the following tokens once the parameter is
  // bundled with the model.
  if (end_of_audio_embedding_.empty()) {
    std::replace(input_ids_vec_with_special_tokens_replaced.begin(),
                 input_ids_vec_with_special_tokens_replaced.end(),
                 ExecutorAudioData::kEndToken, 518272);
  }

  const int max_num_vision_tokens_per_image = text_config_.vision_tokens_num;
  const int num_images = vision_inputs.image_infos.size();
  if (vision_inputs.embeddings != nullptr) {
    // The media embedding cache only needs to hold the embeddings until they
    // are consumed by this prefill, so the images of every chunk are stored
    // starting from slot 0.
    int token_offset = vision_inputs.token_offset;
    for (int i = 0; i < num_images; ++i) {
      const ImageTokenInfo& image_info = vision_inputs.image_infos[i];
      ASSIGN_OR_RETURN(
          SingleImageEmbeddings single_image,
          CreateSingleImageEmbeddings(
              *vision_inputs.embeddings, vision_inputs.per_layer_embeddings,
              image_info, token_offset, max_num_vision_tokens_per_image,
              end_of_vision_embedding_));
      token_offset += image_info.active_tokens;

      RETURN_IF_ERROR(llm_gpu_runner_->FillVisionEmbeddings(
          single_image.embedding, single_image.per_layer_embeddings,
          /*image_index=*/i));
    }

    // Suppose max_num_vision_tokens_per_image = 280, and there are 2 images
    // in this chunk, one with 100 tokens and one with 180 tokens. Then the
    // input ids for the corresponding image tokens are
    // Case 1 without kEndToken: [-1, -2, ..., -100, -281, -282, ..., -460]
    // Case 2 with kEndToken: [-1, -2, ..., -101, -281, -282, ..., -461].
    // The replacement of the input ids is because the media embedding cache has
    // shape [1, 1, max_num_vision_tokens_per_image, embedding_dim]. The indices
    // in input_ids_vec_with_special_tokens_replaced corresponding to the image
    // tokens will be used to gather the corresponding embeddings from the media
    // embedding cache.
    int current_image_index = 0;
    int current_token_within_image = 0;
    bool in_image = false;
    for (int i = 0; i < input_ids_vec.size(); ++i) {
      if (input_ids_vec[i] == ExecutorVisionData::kSpecialToken ||
          input_ids_vec[i] == ExecutorVisionData::kEndToken) {
        in_image = true;
        const int index_in_cached_embeddings =
            -current_image_index * max_num_vision_tokens_per_image - 1 -
            current_token_within_image;
        input_ids_vec_with_special_tokens_replaced[i] =
            index_in_cached_embeddings;
        current_token_within_image++;
      } else {
        if (in_image) {
          current_image_index++;
          current_token_within_image = 0;
          in_image = false;
        }
      }
    }
  }

  // Audio embeddings, and end-of-audio embeddings if available, are written to
  // the media embedding cache after the slots of the images in this chunk, in
  // the order their tokens appear in the chunk.
  const bool has_audio_data = !audio_inputs.embeddings.empty();
  const bool has_end_of_audio_embeddings =
      !end_of_audio_embedding_.empty() &&
      absl::c_linear_search(input_ids_vec, ExecutorAudioData::kEndToken);
  if (has_audio_data || has_end_of_audio_embeddings) {
    // Each audio special token corresponds to one audio embedding, in order.
    // There may be more audio embeddings than audio special tokens, e.g. due to
    // padding.
    const absl::Span<const float> audio_embeddings = audio_inputs.embeddings;
    const int embedding_dimension = has_audio_data
                                        ? audio_inputs.embedding_dimension
                                        : text_config_.model_dimension;
    const int num_audio_embeddings =
        has_audio_data
            ? static_cast<int>(audio_embeddings.size() / embedding_dimension)
            : 0;

    const int audio_cache_offset = num_images * max_num_vision_tokens_per_image;
    std::vector<float> media_embeddings;
    int num_media_embeddings = 0;
    int num_audio_embeddings_used = 0;
    for (int i = 0; i < input_ids_vec.size(); ++i) {
      absl::Span<const float> embedding;
      if (input_ids_vec[i] == ExecutorAudioData::kSpecialToken) {
        RET_CHECK_LT(num_audio_embeddings_used, num_audio_embeddings)
            << "There are more audio special tokens than audio embeddings.";
        embedding = audio_embeddings.subspan(
            num_audio_embeddings_used * embedding_dimension,
            embedding_dimension);
        ++num_audio_embeddings_used;
      } else if (input_ids_vec[i] == ExecutorAudioData::kEndToken &&
                 !end_of_audio_embedding_.empty()) {
        RET_CHECK_EQ(static_cast<int>(end_of_audio_embedding_.size()),
                     embedding_dimension)
            << "EndOfAudio embedding size mismatch.";
        embedding = end_of_audio_embedding_;
      } else {
        continue;
      }
      media_embeddings.insert(media_embeddings.end(), embedding.begin(),
                              embedding.end());
      // The token id -k - 1 refers to the k-th slot of the media embedding
      // cache.
      input_ids_vec_with_special_tokens_replaced[i] =
          -(audio_cache_offset + num_media_embeddings) - 1;
      ++num_media_embeddings;
    }

    if (num_media_embeddings > 0) {
      LITERT_ASSIGN_OR_RETURN(
          auto media_embeddings_tensor,
          ::litert::lm::CopyToTensorBuffer<float>(
              absl::MakeConstSpan(media_embeddings),
              {1, 1, num_media_embeddings, 1, embedding_dimension}));
      RETURN_IF_ERROR(llm_gpu_runner_->FillAudioEmbeddingsWithTensorBuffer(
          media_embeddings_tensor, audio_cache_offset));
    }
  }

  // If the cancel flag is set and the cancel flag is true, do not trigger
  // prefill, and just return early.
  if (cancel != nullptr && *cancel) {
    return absl::OkStatus();
  }

#ifdef __EMSCRIPTEN__
  RETURN_IF_ERROR(llm_gpu_runner_->InitInputTokens(
      input_ids_vec_with_special_tokens_replaced,
      llm_context_->runtime_state().current_step, wait_for_completion));
#else
  RETURN_IF_ERROR(llm_gpu_runner_->InitInputTokens(
      input_ids_vec_with_special_tokens_replaced,
      llm_context_->runtime_state().current_step, wait_for_completion, cancel));
  // If the cancel flag is set and the cancel flag is true, return early.
  if (cancel != nullptr && *cancel) {
    return absl::OkStatus();
  }
#endif  // __EMSCRIPTEN__
  const size_t num_input_tokens = input_ids_vec.size();
  llm_context_->processed_context().processed_tokens().AddProcessedTokens(
      std::move(input_ids_vec));
  llm_context_->runtime_state().current_step += num_input_tokens;

  ready_for_decode_ = true;

  return absl::OkStatus();
}

absl::StatusOr<std::vector<std::vector<int>>> LlmGpuArtisanExecutor::Decode() {
  return Decode(::litert::lm::ExecutorDecodeParams());
}

absl::StatusOr<std::vector<std::vector<int>>> LlmGpuArtisanExecutor::Decode(
    const ::litert::lm::ExecutorDecodeParams& decode_params) {
  ::litert::lm::ConstrainedDecoder* constrained_decoder =
      decode_params.GetConstrainedDecoder();
  if (constrained_decoder != nullptr) {
    RuntimeConfig updated_config = llm_context_->runtime_config();
    updated_config.constraint = std::make_shared<llm_utils::ConstraintAdapter>(
        constrained_decoder->GetConstraint());
    RETURN_IF_ERROR(UpdateRuntimeConfig(updated_config));
  }

  absl::Cleanup cleanup_constraint = [this, constrained_decoder] {
    if (constrained_decoder != nullptr) {
      RuntimeConfig updated_config = llm_context_->runtime_config();
      updated_config.constraint = nullptr;
      auto status = UpdateRuntimeConfig(updated_config);
      if (!status.ok()) {
        ABSL_LOG(ERROR) << "Failed to clean up constraint: " << status;
      }
    }
  };

  RET_CHECK_NE(llm_gpu_runner_, nullptr)
      << "The LlmGpuArtisanExecutor's llm_gpu_runner_ is null, not created "
         "correctly.";
  RET_CHECK_NE(llm_context_, nullptr)
      << "The llm_context_ should not be nullptr.";
  RET_CHECK_LE(
      llm_context_->runtime_state().current_step,
      llm_context_->processed_context().processed_tokens().TokenCount())
      << "The current_step should not be larger than the processed_tokens "
         "size.";
  RET_CHECK_GE(llm_context_->runtime_state().current_step, 0)
      << "The current_step should not be negative.";
  RET_CHECK_NE(llm_context_->runtime_state().current_step, 0)
      << "Decode should not be called without any processed context.";

  // If the constraint is set and the constraint state is null, initialize the
  // constraint state and update the constraint mask.
  if (llm_context_->runtime_config().constraint &&
      llm_context_->runtime_state().constraint_state == nullptr) {
    const auto& constraint = **llm_context_->runtime_config().constraint;
    llm_context_->runtime_state().constraint_state = constraint.Start();
    ASSIGN_OR_RETURN(auto bitmap,
                     constraint.ComputeBitmap(
                         *llm_context_->runtime_state().constraint_state));
    RETURN_IF_ERROR(llm_gpu_runner_->UpdateConstraintMask(*bitmap));

    // Set ready_for_decode_ to false to resample the next token using the
    // updated constraint mask.
    ready_for_decode_ = false;
  }

  // If ready_for_decode_ is false, re-prefill the last token is needed.
  if (!ready_for_decode_) {
    llm_context_->runtime_state().current_step--;
    std::vector<int> last_input_token_id =
        llm_context_->processed_context().processed_tokens().GetTokenAtStep(
            llm_context_->runtime_state().current_step);
    ABSL_CHECK(!last_input_token_id.empty());
    LITERT_ASSIGN_OR_RETURN(
        auto new_inputs_token_ids,
        ::litert::lm::CopyToTensorBuffer(
            absl::MakeConstSpan(last_input_token_id),
            {1, static_cast<int>(last_input_token_id.size())}));
    auto new_inputs =
        ExecutorInputs(ExecutorTextData(std::move(new_inputs_token_ids)),
                       std::nullopt, std::nullopt);

    RETURN_IF_ERROR(Prefill(new_inputs));
  }

  // Since GPU artisan executor stores the next token in the runner, we need to
  // update the constraint state here using PeekNextToken.
  if (llm_context_->runtime_config().constraint &&
      SupportsPeekNextToken(llm_gpu_runner_->GetLlmGpuRunnerType())) {
    ASSIGN_OR_RETURN(auto next_token, llm_gpu_runner_->PeekNextToken());
    RETURN_IF_ERROR(UpdateConstraintState(next_token.token()));
  }

  std::vector<std::vector<int>> output_tokens_vec(
      *llm_context_->runtime_config().output_heads,
      std::vector<int>(*llm_context_->runtime_config().tokens_per_decode));
  RETURN_IF_ERROR(llm_gpu_runner_->GetNextTokens(&output_tokens_vec,
                                                 /*update_constraint_fn=*/{}));
  // Currently only support output head = 1.
  // TODO: b/367340580 - Support multiple output heads.
  RET_CHECK_EQ(output_tokens_vec.size(), 1)
      << "Currently only support output head = 1.";
  llm_context_->processed_context().processed_tokens().AddProcessedTokens(
      output_tokens_vec[0]);
  llm_context_->runtime_state().current_step += output_tokens_vec[0].size();

  return output_tokens_vec;
};

absl::StatusOr<std::unique_ptr<LlmContext>>
LlmGpuArtisanExecutor::CreateNewContext(std::optional<uint32_t> lora_id,
                                        RuntimeConfig runtime_config) const {
  auto context_data = ml_drift::ContextData();
  context_data.lora_id = std::move(lora_id);

  auto processed_context =
      std::make_unique<odml::infra::LlmGpuArtisanProcessedContext>(
          std::move(context_data));

  auto runtime_state = std::make_unique<RuntimeState>();
  runtime_state->rand_gen = std::make_shared<std::default_random_engine>(
      runtime_config.sampler_params->seed());
  return std::make_unique<LlmContext>(
      std::move(processed_context),
      std::make_unique<RuntimeConfig>(std::move(runtime_config)),
      std::move(runtime_state));
}

absl::StatusOr<std::unique_ptr<LlmContext>>
LlmGpuArtisanExecutor::CloneContext() const {
  RET_CHECK_NE(llm_gpu_runner_, nullptr)
      << "The LlmGpuArtisanExecutor's llm_gpu_runner_ is null, not created "
         "correctly.";
  RET_CHECK_NE(llm_context_, nullptr)
      << "The llm_context_ should not be nullptr.";

  ASSIGN_OR_RETURN(auto context_data, llm_gpu_runner_->SaveContext());
  auto new_processed_context =
      std::make_unique<odml::infra::LlmGpuArtisanProcessedContext>(
          std::move(context_data),
          llm_context_->processed_context().processed_tokens());
  auto new_runtime_config =
      std::make_unique<RuntimeConfig>(llm_context_->runtime_config());
  auto new_runtime_state =
      std::make_unique<RuntimeState>(llm_context_->runtime_state());
  return std::make_unique<LlmContext>(std::move(new_processed_context),
                                      std::move(new_runtime_config),
                                      std::move(new_runtime_state));
}

absl::Status LlmGpuArtisanExecutor::RestoreContext(
    std::unique_ptr<LlmContext> llm_context) {
  ready_for_decode_ = false;
  RET_CHECK_NE(llm_gpu_runner_, nullptr)
      << "The LlmGpuArtisanExecutor's llm_gpu_runner_ is null, not created "
         "correctly.";
  auto gpu_processed_context = dynamic_cast<LlmGpuArtisanProcessedContext*>(
      &llm_context->processed_context());
  RET_CHECK_NE(gpu_processed_context, nullptr)
      << "The gpu_processed_context should not be nullptr.";
  // If the processed_tokens is not empty, the context_data_ should not be
  // nullptr.
  if (llm_context->processed_context().processed_tokens().TokenCount() > 0) {
    ASSIGN_OR_RETURN(bool has_kv_cache, gpu_processed_context->HasKVCache());
    RET_CHECK(has_kv_cache) << "The context_data_ should have kv cache.";
  }
  if (llm_context_ == nullptr ||
      llm_context_->processed_context().lora_id() !=
          llm_context->processed_context().lora_id()) {
    RETURN_IF_ERROR(
        llm_gpu_runner_->UseLoRA(llm_context->processed_context().lora_id()));
  }

  gpu_processed_context->mutable_context_data().tokens =
      llm_context->processed_context().processed_tokens().GetCopyOfTokens()[0];
  RETURN_IF_ERROR(llm_gpu_runner_->RestoreContextWithCacheCleaning(
      gpu_processed_context->mutable_context_data()));
  RETURN_IF_ERROR(llm_gpu_runner_->UpdateSamplerParams(
      ConvertToOdmlSamplerParams(*llm_context->runtime_config().sampler_params),
      llm_context->runtime_state().rand_gen));
  llm_context_ = std::move(llm_context);
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<::litert::lm::ProcessedContext>>
LlmGpuArtisanExecutor::SaveContext() const {
  RET_CHECK_NE(llm_gpu_runner_, nullptr)
      << "The LlmGpuArtisanExecutor's llm_gpu_runner_ is null, not created "
         "correctly.";
  ASSIGN_OR_RETURN(auto context_data, llm_gpu_runner_->SaveLocalContext());
  return std::make_unique<odml::infra::LlmGpuArtisanProcessedContext>(
      std::move(context_data));
}

absl::Status LlmGpuArtisanExecutor::RestoreContext(
    std::unique_ptr<::litert::lm::ProcessedContext> processed_context) {
  ready_for_decode_ = false;
  RET_CHECK_NE(llm_gpu_runner_, nullptr)
      << "The LlmGpuArtisanExecutor's llm_gpu_runner_ is null, not created "
         "correctly.";
  RET_CHECK_NE(llm_context_, nullptr)
      << "The llm_context_ should not be nullptr.";
  RET_CHECK_NE(processed_context, nullptr)
      << "The processed_context should not be nullptr.";
  auto gpu_processed_context =
      dynamic_cast<LlmGpuArtisanProcessedContext*>(processed_context.get());
  RET_CHECK_NE(gpu_processed_context, nullptr)
      << "The gpu_processed_context should not be nullptr.";
  if (llm_context_->processed_context().processed_tokens().TokenCount() !=
      llm_context_->runtime_state().current_step) {
    RETURN_IF_ERROR(
        llm_gpu_runner_->CleanCache(llm_context_->runtime_state().current_step,
                                    /*skip_ringbuffer_delta_check=*/true));
    RETURN_IF_ERROR(
        llm_context_->processed_context().processed_tokens().RollBackToStep(
            llm_context_->runtime_state().current_step));
  }
  return llm_gpu_runner_->RestoreLocalContext(
      std::move(gpu_processed_context->mutable_context_data()));
}

absl::Status LlmGpuArtisanExecutor::UpdateRuntimeConfig(
    const RuntimeConfig& runtime_config) {
  if (runtime_config.sampler_params.has_value()) {
    llm_context_->runtime_config().sampler_params =
        runtime_config.sampler_params;
    RETURN_IF_ERROR(llm_gpu_runner_->UpdateSamplerParams(
        ConvertToOdmlSamplerParams(
            *llm_context_->runtime_config().sampler_params),
        llm_context_->runtime_state().rand_gen));
  }
  if (runtime_config.output_heads.has_value()) {
    llm_context_->runtime_config().output_heads = *runtime_config.output_heads;
  }
  if (runtime_config.tokens_per_decode.has_value()) {
    llm_context_->runtime_config().tokens_per_decode =
        *runtime_config.tokens_per_decode;
  }
  if (runtime_config.constraint.has_value()) {
    if (runtime_config.constraint.value() == nullptr) {
      // Remove the constraint.
      llm_context_->runtime_config().constraint = std::nullopt;
      RETURN_IF_ERROR(
          llm_gpu_runner_->UpdateConstraintMask(llm_utils::AllAllowedBitmap()));
    } else {
      // Update the constraint to the new one.
      llm_context_->runtime_config().constraint = runtime_config.constraint;
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<RuntimeConfig> LlmGpuArtisanExecutor::GetRuntimeConfig() const {
  return llm_context_->runtime_config();
}

absl::Status LlmGpuArtisanExecutor::UpdateRuntimeState(
    const RuntimeState& runtime_state) {
  // If the current step is different from the new step, reset the
  // ready_for_decode_ flag.
  if (llm_context_->runtime_state().current_step !=
      runtime_state.current_step) {
    ready_for_decode_ = false;
  }
  llm_context_->runtime_state() = runtime_state;
  RETURN_IF_ERROR(llm_gpu_runner_->UpdateSamplerParams(
      ConvertToOdmlSamplerParams(
          *llm_context_->runtime_config().sampler_params),
      llm_context_->runtime_state().rand_gen));
  return absl::OkStatus();
}

absl::StatusOr<RuntimeState> LlmGpuArtisanExecutor::GetRuntimeState() const {
  return llm_context_->runtime_state();
}

absl::StatusOr<int> LlmGpuArtisanExecutor::GetCurrentStep() const {
  return llm_context_->runtime_state().current_step;
}

absl::Status LlmGpuArtisanExecutor::SetCurrentStep(int new_step) {
  RET_CHECK_LE(
      new_step,
      llm_context_->processed_context().processed_tokens().TokenCount())
      << "Time step is too large.";
  RET_CHECK_GE(new_step, 0) << "Time step is negative.";
  if (llm_context_->runtime_state().current_step != new_step) {
    llm_context_->runtime_state().current_step = new_step;
    ready_for_decode_ = false;
  }
  return absl::OkStatus();
}

absl::StatusOr<const ::litert::lm::ProcessedTokens*>
LlmGpuArtisanExecutor::GetProcessedTokens() const {
  return &llm_context_->processed_context().processed_tokens();
}

absl::Status LlmGpuArtisanExecutor::UpdateConstraintState(int next_token) {
  RuntimeState& runtime_state = llm_context_->runtime_state();
  const auto& constraint = **llm_context_->runtime_config().constraint;
  ASSIGN_OR_RETURN(
      auto next_state,
      constraint.ComputeNext(*runtime_state.constraint_state, next_token));
  // If the next state is the ended state, drop the constraint state and lift
  // the mask. The next constrained Decode() (or the re-prefill it triggers)
  // will start again from `constraint.Start()`.
  if (constraint.IsEnded(*next_state)) {
    runtime_state.constraint_state = nullptr;
    RETURN_IF_ERROR(
        llm_gpu_runner_->UpdateConstraintMask(llm_utils::AllAllowedBitmap()));
    return absl::OkStatus();
  }
  runtime_state.constraint_state = std::move(next_state);
  ASSIGN_OR_RETURN(auto bitmap,
                   constraint.ComputeBitmap(*runtime_state.constraint_state));
  RETURN_IF_ERROR(llm_gpu_runner_->UpdateConstraintMask(*bitmap));
  return absl::OkStatus();
}

absl::Status LlmGpuArtisanExecutor::Reset() {
  RETURN_IF_ERROR(llm_gpu_runner_->CleanCache(0));
  RETURN_IF_ERROR(llm_gpu_runner_->UseLoRA(std::nullopt));

  // Reset the LlmContext to clear the history and state.
  // We reset the runtime config to default and clear lora id.
  ::litert::lm::proto::SamplerParameters sampling_params;
  sampling_params.set_type(::litert::lm::proto::SamplerParameters::GREEDY);
  sampling_params.set_k(1);
  sampling_params.set_seed(0);

  ASSIGN_OR_RETURN(auto new_context,
                   CreateNewContext(/*lora_id=*/std::nullopt,
                                    ::litert::lm::RuntimeConfig({
                                        .sampler_params = sampling_params,
                                        .output_heads = 1,
                                        .tokens_per_decode = 1,
                                    })));
  llm_context_ = std::move(new_context);

  ready_for_decode_ = false;

  return absl::OkStatus();
}

absl::Status LlmGpuArtisanExecutor::FillVisionEmbeddings(
    const ExecutorVisionData& vision_input, int image_index) {
  RET_CHECK(text_config_.vision_tokens_num > 0)
      << "Model does not support vision embeddings.";

  const int max_vision_tokens_num = text_config_.vision_tokens_num;
  const int model_dimension = text_config_.model_dimension;

  ml_drift::TensorFloat32 vision_embedding_tensor;
  std::vector<ml_drift::TensorFloat32> vision_ple_embeddings_tensors;

  int active_tokens = -1;

  // Copy flattened vision embeddings to vision_embedding_tensor.
  // vision_embedding_tensor has shape [1, 1, max_vision_tokens_num,
  // model_dimension]. The raw vision embeddings from vision_input has shape
  // [1, 1, active_tokens, model_dimension] where active_tokens <=
  // max_vision_tokens_num. We will copy the active_tokens to the beginning of
  // the vision_embedding_tensor and fill the rest with zeros.
  if (vision_input.GetEmbeddingsPtr().ok()) {
    // vision_embedding_tensor is a 4-D tensor with shape
    // [1, 1, vision_tokens_num, model_dimension]
    vision_embedding_tensor.shape = ml_drift::BHWC{
        /*batch=*/1, /*height=*/1, /*width=*/max_vision_tokens_num,
        /*channels=*/model_dimension};
    LITERT_ASSIGN_OR_RETURN(auto vision_embedding_size_in_bytes,
                            (*vision_input.GetEmbeddingsPtr())->Size());
    auto vision_embedding_size = vision_embedding_size_in_bytes / sizeof(float);
    RET_CHECK_LE(vision_embedding_size,
                 vision_embedding_tensor.shape.DimensionsProduct())
        << "Vision embedding size exceeds expected capacity.";
    RET_CHECK_EQ(vision_embedding_size % model_dimension, 0)
        << "Invalid embedding size.";
    active_tokens = vision_embedding_size / model_dimension;
    vision_embedding_tensor.data.assign(
        vision_embedding_tensor.shape.DimensionsProduct(), 0.0f);

    // Read the data directly from the tensor buffer instead of
    // ReferToTensorBufferAsSpan to support non-host memory buffers.
    LITERT_ASSIGN_OR_RETURN(auto mutable_vision_embeddings,
                            (*vision_input.GetEmbeddingsPtr())->Duplicate());
    LITERT_RETURN_IF_ERROR(mutable_vision_embeddings.Read(
        absl::MakeSpan(vision_embedding_tensor.data)
            .first(vision_embedding_size)));

    if (!end_of_vision_embedding_.empty()) {
      RET_CHECK_LT(active_tokens, max_vision_tokens_num)
          << "EndOfVision token index exceeds maximum allowed per image.";
      RET_CHECK_EQ(end_of_vision_embedding_.size(), model_dimension)
          << "EndOfVision embedding size mismatch.";
      std::copy(end_of_vision_embedding_.begin(),
                end_of_vision_embedding_.end(),
                vision_embedding_tensor.data.begin() + vision_embedding_size);
    }
  }

  // Copy flattened vision per layer embeddings to
  // vision_ple_embeddings_tensors. vision_ple_embeddings_tensors is a vector of
  // 4-D tensors with shape [1, 1, max_vision_tokens_num,
  // per_layer_embedding_dimension]. The raw vision per layer embeddings from
  // vision_input has shape [1, 1, active_tokens, per_layer_embedding_dimension]
  // where active_tokens <= max_vision_tokens_num. We will copy the
  // active_tokens to the beginning of the vision_ple_embeddings_tensors and
  // fill the rest with zeros.
  if (vision_input.GetPerLayerEmbeddingsPtr().ok()) {
    RET_CHECK(text_config_.per_layer_embedding_dimension.has_value());

    const int per_layer_embedding_dimension =
        *text_config_.per_layer_embedding_dimension;
    const int stack_size = text_config_.stack_size;  // Aka number of layers

    LITERT_ASSIGN_OR_RETURN(auto per_layer_embeddings_size_in_bytes,
                            (*vision_input.GetPerLayerEmbeddingsPtr())->Size());
    auto per_layer_embeddings_size =
        per_layer_embeddings_size_in_bytes / sizeof(float);
    const int expected_elements_per_token =
        stack_size * per_layer_embedding_dimension;
    RET_CHECK_EQ(per_layer_embeddings_size % expected_elements_per_token, 0)
        << "Invalid per-layer embedding size.";

    RET_CHECK_LE(active_tokens, max_vision_tokens_num)
        << "Number of per-layer embeddings tokens exceeds maximum allowed "
        << max_vision_tokens_num;
    RET_CHECK_EQ(active_tokens * expected_elements_per_token,
                 per_layer_embeddings_size)
        << "Mismatch in vision tokens count for per-layer embeddings.";

    // Each vision_ple_embeddings_tensor is a 4-D tensor with shape
    // [1, 1, max_vision_tokens_num, per_layer_embedding_dimension]
    vision_ple_embeddings_tensors.resize(stack_size);
    ml_drift::BHWC expected_vision_ple_shape{
        /*batch=*/1, /*height=*/1,
        /*width=*/max_vision_tokens_num,
        /*channels=*/per_layer_embedding_dimension};

    LITERT_ASSIGN_OR_RETURN(auto vision_per_layer_embeddings,
                            ::litert::lm::ReferTensorBufferAsSpan<float>(
                                *(*vision_input.GetPerLayerEmbeddingsPtr())));

    int offset = 0;
    for (auto& tensor : vision_ple_embeddings_tensors) {
      tensor.shape = expected_vision_ple_shape;
      tensor.data.assign(max_vision_tokens_num * per_layer_embedding_dimension,
                         0.0f);

      absl::Span<const float> ple_embedding_span =
          vision_per_layer_embeddings.subspan(
              offset, active_tokens * per_layer_embedding_dimension);

      std::memcpy(tensor.data.data(), ple_embedding_span.data(),
                  ple_embedding_span.size() * sizeof(float));
      offset += active_tokens * per_layer_embedding_dimension;
    }
  }

  return llm_gpu_runner_->FillVisionEmbeddings(
      vision_embedding_tensor, vision_ple_embeddings_tensors, image_index);
}

absl::StatusOr<::litert::lm::LlmExecutorSettings>
LlmGpuArtisanExecutor::GetExecutorSettings() const {
  return executor_settings_;
}

}  // namespace odml::infra
