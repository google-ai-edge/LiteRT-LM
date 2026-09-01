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

#include "runtime/core/embedding_engine_impl.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/types/optional.h"  // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/engine/embedding_engine.h"
#include "runtime/engine/embedding_engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/audio/audio_executor.h"
#include "runtime/executor/audio_litert_compiled_model_executor.h"
#include "runtime/executor/embedding/embedding_executor_base.h"
#include "runtime/executor/embedding/embedding_executor_settings.h"
#include "runtime/executor/embedding_litert_compiled_model_executor.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/litert_compiled_model_executor_utils.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/executor/model_signature_utils.h"
#include "runtime/executor/vision/vision_executor.h"
#include "runtime/executor/vision_litert_compiled_model_executor.h"
#include "runtime/proto/embedding_metadata.pb.h"
#include "runtime/proto/embedding_model_type.pb.h"
#include "runtime/proto/engine.pb.h"
#include "runtime/proto/token.pb.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/executor_data_util.h"
#include "runtime/util/litert_util.h"
#include "runtime/util/status_macros.h"
#include "runtime/util/tensor_buffer_util.h"
#include "support/preprocessor/audio_preprocessor.h"
#include "support/preprocessor/audio_preprocessor_miniaudio.h"
#include "support/preprocessor/image_preprocessor.h"
#include "support/tokenizer/tokenizer.h"
#include "support/util/io_types.h"

namespace litert::lm {

namespace {

absl::StatusOr<std::vector<int>> TokenUnionToTokenIds(
    const proto::TokenUnion& token_union,
    ::litert::support::Tokenizer& tokenizer) {
  if (token_union.has_token_ids()) {
    return std::vector<int>(token_union.token_ids().ids().begin(),
                            token_union.token_ids().ids().end());
  } else if (token_union.has_token_str()) {
    return tokenizer.TextToTokenIds(token_union.token_str());
  }
  return absl::InvalidArgumentError(
      "Neither token_str nor token_ids is set in TokenUnion.");
}

absl::StatusOr<SpecialTokens> ExtractSpecialTokens(
    const proto::EmbeddingMetadata& metadata,
    ::litert::support::Tokenizer& tokenizer) {
  SpecialTokens special_tokens;
  if (metadata.has_bos_token()) {
    LITERT_ASSIGN_OR_RETURN(
        special_tokens.bos_token_ids,
        TokenUnionToTokenIds(metadata.bos_token(), tokenizer));
  }
  if (metadata.has_eos_token()) {
    LITERT_ASSIGN_OR_RETURN(
        special_tokens.eos_token_ids,
        TokenUnionToTokenIds(metadata.eos_token(), tokenizer));
  }
  if (!metadata.has_embedding_model_type()) {
    return special_tokens;
  }
  const auto& model_type = metadata.embedding_model_type();
  return special_tokens;
}

std::optional<::litert::support::ImagePreprocessParameter>
ExtractImagePreprocessParameter(const proto::EmbeddingMetadata& metadata) {
  if (!metadata.has_embedding_model_type()) {
    return std::nullopt;
  }
  const auto& model_type = metadata.embedding_model_type();
  return std::nullopt;
}

absl::StatusOr<std::unique_ptr<::litert::support::AudioPreprocessor>>
ExtractAudioPreprocessor(const proto::EmbeddingMetadata& metadata) {
  if (metadata.has_audio_preprocessor()) {
    const auto& audio_preprocessor_proto = metadata.audio_preprocessor();
    switch (audio_preprocessor_proto.preprocessor_case()) {
      case proto::AudioPreprocessorConfig::kMiniaudio: {
        const auto& miniaudio = audio_preprocessor_proto.miniaudio();
        using FftPaddingType =
            ::litert::support::AudioPreprocessorConfig::FftPaddingType;
        FftPaddingType padding_type = FftPaddingType::kRight;
        if (miniaudio.fft_padding_type() ==
            proto::MiniAudioPreprocessorConfig::FFT_PADDING_TYPE_CENTER) {
          padding_type = FftPaddingType::kCenter;
        }
        auto config = ::litert::support::AudioPreprocessorConfig::Create(
            miniaudio.sample_rate_hz(), miniaudio.num_channels(),
            miniaudio.frame_length(), miniaudio.hop_length(),
            miniaudio.fft_length(), miniaudio.input_scale(),
            miniaudio.pre_emphasis_factor(), miniaudio.num_mel_bins(),
            miniaudio.mel_low_hz(), miniaudio.mel_high_hz(),
            miniaudio.mel_floor(), miniaudio.normalize_mel(),
            miniaudio.add_floor_to_mel_before_log(),
            miniaudio.semicausal_padding(), miniaudio.non_zero_hanning(),
            miniaudio.periodic_hanning(), padding_type,
            miniaudio.skip_mel_spectrogram_extraction());
        return ::litert::support::AudioPreprocessorMiniAudio::Create(config);
      }
      case proto::AudioPreprocessorConfig::PREPROCESSOR_NOT_SET:
        break;
    }
  }

  return nullptr;
}

std::vector<float> L2Norm(const std::vector<float>& vec) {
  float sum_sq = 0.0f;
  for (float val : vec) {
    sum_sq += val * val;
  }

  if (sum_sq <= 0.0) {
    return vec;
  }

  float norm = std::sqrt(sum_sq);
  std::vector<float> result = vec;
  for (float& val : result) {
    val /= norm;
  }

  return result;
};

absl::StatusOr<uint64_t> GetNumTokens(const ExecutorInputs& executor_inputs) {
  ABSL_ASSIGN_OR_RETURN(auto text_data, executor_inputs.GetTextDataPtr());
  if (text_data == nullptr) {
    return 0;
  }

  const auto& token_ids = text_data->GetTokenIds();
  LITERT_ASSIGN_OR_RETURN(auto span, ReferTensorBufferAsSpan<int>(token_ids));
  return span.size();
}

}  // namespace

// static
absl::StatusOr<std::unique_ptr<EmbeddingEngine>> EmbeddingEngineImpl::Create(
    std::unique_ptr<ModelResources> resources,
    std::unique_ptr<OwnedEnvironment> env,
    std::unique_ptr<::litert::support::Tokenizer> tokenizer,
    EmbeddingEngineSettings settings,
    std::optional<BenchmarkInfo> benchmark_info) {
  if (resources == nullptr) {
    return absl::InvalidArgumentError("ModelResources cannot be null.");
  }
  if (env == nullptr) {
    return absl::InvalidArgumentError("OwnedEnvironment cannot be null.");
  }
  if (tokenizer == nullptr) {
    return absl::InvalidArgumentError("Tokenizer cannot be null.");
  }

  // Initialize metadata.
  std::optional<proto::EmbeddingMetadata> metadata =
      settings.GetEmbeddingMetadata();
  if (!metadata.has_value()) {
    auto resources_metadata = resources->GetEmbeddingMetadata();
    if (resources_metadata.ok() && *resources_metadata != nullptr) {
      metadata = **resources_metadata;
    }
  }

  // Auto-select text encoder signatures if max_input_length is set.
  std::optional<SelectedTextSignaturesInfo> selected_text_signatures_info =
      std::nullopt;
  if (settings.GetMaxInputLength().has_value()) {
    if (*settings.GetMaxInputLength() <= 0) {
      return absl::InvalidArgumentError(
          absl::StrCat("max_input_length must be positive, got: ",
                       *settings.GetMaxInputLength()));
    }
    LITERT_ASSIGN_OR_RETURN(
        auto text_sig_info,
        SelectTextEncoderSignatures(*resources, *settings.GetMaxInputLength()));
    settings.GetMutableMainExecutorSettings().SetSelectedSignatures(
        text_sig_info.signature_names);
    selected_text_signatures_info = std::move(text_sig_info);
  }

  // Auto-select vision encoder and adapter signatures if
  // vision_tokens_per_image is set.
  std::optional<SelectedVisionSignatureInfo> selected_vision_signature_info =
      std::nullopt;
  if (settings.GetVisionTokensPerImage().has_value()) {
    if (!settings.GetVisionExecutorSettings().has_value()) {
      return absl::FailedPreconditionError(
          "Vision executor settings are not configured.");
    }
    const int vision_tokens_per_image = *settings.GetVisionTokensPerImage();
    if (vision_tokens_per_image <= 0) {
      return absl::InvalidArgumentError(
          absl::StrCat("vision_tokens_per_image must be positive, got: ",
                       vision_tokens_per_image));
    }

    // Configure vision patch metadata.
    int pooling_kernel_size = 1;

    const int patch_num_shrink_factor =
        pooling_kernel_size * pooling_kernel_size;
    const int max_num_patches =
        vision_tokens_per_image * patch_num_shrink_factor;

    LITERT_ASSIGN_OR_RETURN(
        auto vision_sig_info,
        SelectVisionEncoderSignatures(*resources, vision_tokens_per_image));
    settings.GetMutableVisionExecutorSettings()->SetEncoderSelectedSignatures(
        vision_sig_info.signature_names);

    LITERT_ASSIGN_OR_RETURN(
        auto adapter_sig_info,
        SelectVisionAdapterSignatures(*resources, vision_tokens_per_image));
    if (adapter_sig_info.has_value()) {
      settings.GetMutableVisionExecutorSettings()->SetAdapterSelectedSignatures(
          adapter_sig_info->signature_names);
    }

    SelectedVisionSignatureInfo selected_vision_info;
    selected_vision_info.signature_names = vision_sig_info.signature_names;
    selected_vision_info.signature_lengths = vision_sig_info.signature_lengths;
    selected_vision_info.max_signature_length =
        vision_sig_info.max_signature_length;
    if (adapter_sig_info.has_value()) {
      selected_vision_info.adapter_signature_names =
          adapter_sig_info->signature_names;
      selected_vision_info.adapter_signature_lengths =
          adapter_sig_info->signature_lengths;
    }
    selected_vision_info.max_num_patches = max_num_patches;
    selected_vision_signature_info = std::move(selected_vision_info);
  }

  SpecialTokens special_tokens;
  std::optional<::litert::support::ImagePreprocessParameter>
      image_preprocess_parameter = std::nullopt;
  std::unique_ptr<::litert::support::AudioPreprocessor> audio_preprocessor =
      nullptr;
  if (metadata.has_value()) {
    LITERT_ASSIGN_OR_RETURN(special_tokens,
                            ExtractSpecialTokens(*metadata, *tokenizer));
    image_preprocess_parameter =
        ExtractImagePreprocessParameter(*metadata);
    LITERT_ASSIGN_OR_RETURN(audio_preprocessor,
                            ExtractAudioPreprocessor(*metadata));
  }

  // Initialize BenchmarkInfo if benchmarking is enabled and it wasn't passed
  // in as an argument.
  if (settings.IsBenchmarkEnabled() && !benchmark_info.has_value()) {
    benchmark_info = BenchmarkInfo(*settings.GetBenchmarkParams());
    ABSL_RETURN_IF_ERROR(
        benchmark_info->TimeInitPhaseStart(BenchmarkInfo::InitPhase::kTotal));
  }

  if (benchmark_info.has_value()) {
    ABSL_RETURN_IF_ERROR(benchmark_info->TimeInitPhaseStart(
        BenchmarkInfo::InitPhase::kExecutor));
  }

  // Initialize the vision executor.
  std::unique_ptr<VisionExecutor> vision_executor = nullptr;
  if (resources->GetTFLiteModel(ModelType::kTfLiteVisionEncoder).ok() &&
      settings.GetVisionExecutorSettings().has_value()) {
    LITERT_ASSIGN_OR_RETURN(
        vision_executor, VisionLiteRtCompiledModelExecutor::Create(
                             *settings.GetVisionExecutorSettings(), env->env));
  }

  // Initialize the audio executor.
  std::unique_ptr<AudioExecutor> audio_executor = nullptr;
  if ((resources->GetTFLiteModel(ModelType::kTfLiteAudioEncoderHw).ok()) &&
      settings.GetAudioExecutorSettings().has_value()) {
    LITERT_ASSIGN_OR_RETURN(
        audio_executor, AudioLiteRtCompiledModelExecutor::Create(
                            *settings.GetAudioExecutorSettings(), env->env));
  }

  special_tokens.has_end_of_vision_model =
      resources->GetTFLiteModel(ModelType::kTfLiteEndOfVision).ok();
  special_tokens.has_end_of_audio_model =
      resources->GetTFLiteModel(ModelType::kTfLiteEndOfAudio).ok();

  // Initialize the image preprocessor if requisite parameters are available in
  // metadata.
  std::unique_ptr<::litert::support::ImagePreprocessor> image_preprocessor =
      nullptr;
  if (image_preprocess_parameter.has_value()) {
    image_preprocessor = ::litert::support::ImagePreprocessor::Create();
  }

  // Initialize the embedding model executor.
  LITERT_ASSIGN_OR_RETURN(
      auto embedding_executor,
      EmbeddingLiteRtCompiledModelExecutor::Create(
          std::move(settings.GetMutableMainExecutorSettings()), env->env,
          std::move(resources)));

  if (benchmark_info.has_value()) {
    ABSL_RETURN_IF_ERROR(
        benchmark_info->TimeInitPhaseEnd(BenchmarkInfo::InitPhase::kExecutor));
    ABSL_RETURN_IF_ERROR(
        benchmark_info->TimeInitPhaseEnd(BenchmarkInfo::InitPhase::kTotal));
  }

  return std::make_unique<EmbeddingEngineImpl>(
      std::move(env), std::move(tokenizer), std::move(embedding_executor),
      std::move(vision_executor), std::move(audio_executor),
      std::move(benchmark_info), std::move(special_tokens),
      std::move(image_preprocessor), std::move(image_preprocess_parameter),
      std::move(audio_preprocessor), std::move(metadata),
      std::move(selected_text_signatures_info),
      std::move(selected_vision_signature_info));
}

// static
absl::StatusOr<std::unique_ptr<EmbeddingEngine>> EmbeddingEngineImpl::Create(
    EmbeddingEngineSettings settings) {
  const auto& model_assets =
      settings.GetMainExecutorSettings().GetModelAssets();
  const bool enable_file_backed_model_loading =
      settings.GetMainExecutorSettings().GetBackend() == Backend::NPU;

  // Build model resources.
  LITERT_ASSIGN_OR_RETURN(auto resources,
                          BuildLiteRtCompiledModelResources(
                              model_assets, enable_file_backed_model_loading));

  if (resources == nullptr) {
    return absl::InvalidArgumentError("ModelResources cannot be null.");
  }

  // Initialize BenchmarkInfo if benchmarking is enabled.
  std::optional<BenchmarkInfo> benchmark_info = std::nullopt;
  if (settings.IsBenchmarkEnabled()) {
    benchmark_info = BenchmarkInfo(*settings.GetBenchmarkParams());
    ABSL_RETURN_IF_ERROR(
        benchmark_info->TimeInitPhaseStart(BenchmarkInfo::InitPhase::kTotal));
  }

  // Load tokenizer.
  if (benchmark_info.has_value()) {
    ABSL_RETURN_IF_ERROR(benchmark_info->TimeInitPhaseStart(
        BenchmarkInfo::InitPhase::kTokenizer));
  }

  LITERT_ASSIGN_OR_RETURN(auto tokenizer, resources->GetTokenizer());

  if (benchmark_info.has_value()) {
    ABSL_RETURN_IF_ERROR(
        benchmark_info->TimeInitPhaseEnd(BenchmarkInfo::InitPhase::kTokenizer));
  }

  if (tokenizer == nullptr) {
    return absl::InvalidArgumentError("Tokenizer cannot be null.");
  }

  // Create LiteRT environment.
  std::unique_ptr<OwnedEnvironment> owned_env;
  {
    LITERT_ASSIGN_OR_RETURN(auto env,
                            CreateEnvironment(settings, resources.get()));
    owned_env = std::make_unique<OwnedEnvironment>(std::move(env));
  }

  return Create(std::move(resources), std::move(owned_env),
                std::move(tokenizer), std::move(settings),
                std::move(benchmark_info));
}

EmbeddingEngineImpl::EmbeddingEngineImpl(
    std::unique_ptr<OwnedEnvironment> env,
    std::unique_ptr<::litert::support::Tokenizer> tokenizer,
    std::unique_ptr<EmbeddingExecutorBase> embedding_executor,
    std::unique_ptr<VisionExecutor> vision_executor,
    std::unique_ptr<AudioExecutor> audio_executor,
    std::optional<BenchmarkInfo> benchmark_info, SpecialTokens special_tokens,
    std::unique_ptr<::litert::support::ImagePreprocessor> image_preprocessor,
    std::optional<::litert::support::ImagePreprocessParameter>
        image_preprocess_parameter,
    std::unique_ptr<::litert::support::AudioPreprocessor> audio_preprocessor,
    std::optional<proto::EmbeddingMetadata> metadata,
    std::optional<SelectedTextSignaturesInfo> selected_text_signatures_info,
    std::optional<SelectedVisionSignatureInfo> selected_vision_signature_info)
    : env_(std::move(env)),
      tokenizer_(std::move(tokenizer)),
      embedding_executor_(std::move(embedding_executor)),
      vision_executor_(std::move(vision_executor)),
      audio_executor_(std::move(audio_executor)),
      benchmark_info_(std::move(benchmark_info)),
      special_tokens_(std::move(special_tokens)),
      image_preprocessor_(std::move(image_preprocessor)),
      image_preprocess_parameter_(std::move(image_preprocess_parameter)),
      audio_preprocessor_(std::move(audio_preprocessor)),
      metadata_(std::move(metadata)),
      selected_text_signatures_info_(std::move(selected_text_signatures_info)),
      selected_vision_signature_info_(
          std::move(selected_vision_signature_info)) {}

absl::StatusOr<std::vector<InputData>> EmbeddingEngineImpl::InsertSpecialTokens(
    const std::vector<InputData>& contents) const {
  std::vector<InputData> new_contents;
  if (!special_tokens_.bos_token_ids.empty()) {
    LITERT_ASSIGN_OR_RETURN(
        auto bos_tensor,
        litert::support::Tokenizer::TokenIdsToTensorBuffer(
            special_tokens_.bos_token_ids));
    new_contents.push_back(InputText(std::move(bos_tensor)));
  }
  for (const auto& item : contents) {
    if (const auto* input_image = std::get_if<InputImage>(&item)) {
      if (!special_tokens_.start_of_image_token_ids.empty()) {
        LITERT_ASSIGN_OR_RETURN(
            auto start_tensor,
            litert::support::Tokenizer::TokenIdsToTensorBuffer(
                special_tokens_.start_of_image_token_ids));
        new_contents.push_back(InputText(std::move(start_tensor)));
      }
      LITERT_ASSIGN_OR_RETURN(auto image_copy, input_image->CreateCopy());
      new_contents.push_back(std::move(image_copy));
      if (special_tokens_.has_end_of_vision_model) {
        new_contents.push_back(InputImageEnd());
      } else if (!special_tokens_.end_of_image_token_ids.empty()) {
        LITERT_ASSIGN_OR_RETURN(
            auto end_tensor, litert::support::Tokenizer::TokenIdsToTensorBuffer(
                                 special_tokens_.end_of_image_token_ids));
        new_contents.push_back(InputText(std::move(end_tensor)));
      }
    } else if (const auto* input_audio = std::get_if<InputAudio>(&item)) {
      if (!special_tokens_.start_of_audio_token_ids.empty()) {
        LITERT_ASSIGN_OR_RETURN(
            auto start_tensor,
            litert::support::Tokenizer::TokenIdsToTensorBuffer(
                special_tokens_.start_of_audio_token_ids));
        new_contents.push_back(InputText(std::move(start_tensor)));
      }
      LITERT_ASSIGN_OR_RETURN(auto audio_copy, input_audio->CreateCopy());
      new_contents.push_back(std::move(audio_copy));
      if (special_tokens_.has_end_of_audio_model) {
        new_contents.push_back(InputAudioEnd());
      } else if (!special_tokens_.end_of_audio_token_ids.empty()) {
        LITERT_ASSIGN_OR_RETURN(
            auto end_tensor, litert::support::Tokenizer::TokenIdsToTensorBuffer(
                                 special_tokens_.end_of_audio_token_ids));
        new_contents.push_back(InputText(std::move(end_tensor)));
      }
    } else {
      LITERT_ASSIGN_OR_RETURN(auto item_copy, CreateInputDataCopy(item));
      new_contents.push_back(std::move(item_copy));
    }
  }
  if (!special_tokens_.eos_token_ids.empty()) {
    LITERT_ASSIGN_OR_RETURN(
        auto eos_tensor,
        litert::support::Tokenizer::TokenIdsToTensorBuffer(
            special_tokens_.eos_token_ids));
    new_contents.push_back(InputText(std::move(eos_tensor)));
  }
  return new_contents;
}

absl::StatusOr<ExecutorInputs> EmbeddingEngineImpl::ProcessAndCombineContents(
    const std::vector<InputData>& contents, const EmbeddingOptions& options) {
  if (options.vision_tokens_per_image.has_value() &&
      *options.vision_tokens_per_image <= 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("vision_tokens_per_image must be positive, got: ",
                     *options.vision_tokens_per_image));
  }

  std::vector<int> combined_token_ids;
  std::vector<ExecutorVisionData> all_image_data;
  std::vector<ExecutorAudioData> all_audio_data;

  for (const auto& content : contents) {
    if (const auto* input_text = std::get_if<InputText>(&content)) {
      if (input_text->IsTensorBuffer()) {
        LITERT_ASSIGN_OR_RETURN(const auto* token_ids,
                                input_text->GetPreprocessedTextTensor());
        if (token_ids == nullptr) {
          return absl::InvalidArgumentError("Token IDs is null in contents.");
        }
        LITERT_ASSIGN_OR_RETURN(auto ids_buffer_span,
                                ReferTensorBufferAsSpan<int>(*token_ids));
        combined_token_ids.insert(combined_token_ids.end(),
                                  ids_buffer_span.begin(),
                                  ids_buffer_span.end());
      } else {
        if (benchmark_info_.has_value()) {
          ABSL_RETURN_IF_ERROR(benchmark_info_->TimeTextToTokenIdsStart());
        }
        LITERT_ASSIGN_OR_RETURN(auto raw_text, input_text->GetRawTextString());
        LITERT_ASSIGN_OR_RETURN(auto token_ids,
                                tokenizer_->TextToTokenIds(raw_text));
        combined_token_ids.insert(combined_token_ids.end(), token_ids.begin(),
                                  token_ids.end());
        if (benchmark_info_.has_value()) {
          ABSL_RETURN_IF_ERROR(
              benchmark_info_->TimeTextToTokenIdsEnd(token_ids.size()));
        }
      }
    } else if (const auto* input_image = std::get_if<InputImage>(&content)) {
      if (vision_executor_ == nullptr) {
        return absl::FailedPreconditionError(
            "Vision executor is not available for image input.");
      }
      ExecutorVisionData single_image_data;
      if (input_image->IsTensorBuffer()) {
        LITERT_ASSIGN_OR_RETURN(auto tensor_buffer,
                                input_image->GetPreprocessedImageTensor());
        LITERT_ASSIGN_OR_RETURN(single_image_data,
                                vision_executor_->Encode(*tensor_buffer));
      } else if (input_image->IsTensorBufferMap()) {
        LITERT_ASSIGN_OR_RETURN(auto tensor_buffer_map,
                                input_image->GetPreprocessedImageTensorMap());
        LITERT_ASSIGN_OR_RETURN(single_image_data,
                                vision_executor_->Encode(*tensor_buffer_map));
      } else {
        if (image_preprocessor_ == nullptr) {
          return absl::FailedPreconditionError(
              "Image preprocessor is not available for raw image input.");
        }
        if (!image_preprocess_parameter_.has_value()) {
          return absl::FailedPreconditionError(
              "Image preprocess parameter is not available for raw image "
              "input.");
        }
        ::litert::support::ImagePreprocessParameter
            per_call_image_preprocess_parameter = *image_preprocess_parameter_;
        if (options.vision_tokens_per_image.has_value() &&
            per_call_image_preprocess_parameter.GetPatchifyConfig()
                .has_value()) {
          auto patchify_config =
              *per_call_image_preprocess_parameter.GetPatchifyConfig();
          const int pooling_kernel_size =
              std::max(patchify_config.pooling_kernel_size, 1);
          const int patches_per_vision_token =
              pooling_kernel_size * pooling_kernel_size;
          patchify_config.max_num_patches =
              *options.vision_tokens_per_image * patches_per_vision_token;
          per_call_image_preprocess_parameter.SetPatchifyConfig(
              patchify_config);
        }
        LITERT_ASSIGN_OR_RETURN(
            auto preprocessed_image,
            image_preprocessor_->Preprocess(
                *input_image, per_call_image_preprocess_parameter));
        if (preprocessed_image.IsTensorBuffer()) {
          LITERT_ASSIGN_OR_RETURN(
              auto tensor_buffer,
              preprocessed_image.GetPreprocessedImageTensor());
          LITERT_ASSIGN_OR_RETURN(single_image_data,
                                  vision_executor_->Encode(*tensor_buffer));
        } else if (preprocessed_image.IsTensorBufferMap()) {
          LITERT_ASSIGN_OR_RETURN(
              auto tensor_buffer_map,
              preprocessed_image.GetPreprocessedImageTensorMap());
          LITERT_ASSIGN_OR_RETURN(
              single_image_data,
              vision_executor_->Encode(*tensor_buffer_map));
        } else {
          return absl::InternalError(
              "Failed to get tensor buffer from preprocessed image.");
        }
      }
      LITERT_ASSIGN_OR_RETURN(auto embeddings_ptr,
                              single_image_data.GetEmbeddingsPtr());
      LITERT_ASSIGN_OR_RETURN(const auto& dimensions,
                              TensorBufferDims(*embeddings_ptr));
      const int image_token_num = dimensions.at(dimensions.size() - 2);
      combined_token_ids.insert(combined_token_ids.end(), image_token_num,
                                ExecutorVisionData::kSpecialToken);
      all_image_data.push_back(std::move(single_image_data));
    } else if (const auto* input_image_end =
                   std::get_if<InputImageEnd>(&content)) {
      combined_token_ids.push_back(ExecutorVisionData::kEndToken);
    } else if (const auto* input_audio = std::get_if<InputAudio>(&content)) {
      if (audio_executor_ == nullptr) {
        return absl::FailedPreconditionError(
            "Audio executor is not available for audio input.");
      }
      const ::litert::TensorBuffer* spectrogram_tensor = nullptr;
      std::optional<InputAudio> preprocessed_audio;
      if (input_audio->IsTensorBuffer()) {
        LITERT_ASSIGN_OR_RETURN(spectrogram_tensor,
                                input_audio->GetPreprocessedAudioTensor());
      } else {
        if (audio_preprocessor_ == nullptr) {
          return absl::FailedPreconditionError(
              "Audio preprocessor is not available for unprocessed audio "
              "input.");
        }
        LITERT_ASSIGN_OR_RETURN(InputAudio temp_audio,
                                audio_preprocessor_->Preprocess(*input_audio));
        preprocessed_audio.emplace(std::move(temp_audio));
        LITERT_ASSIGN_OR_RETURN(
            spectrogram_tensor,
            preprocessed_audio->GetPreprocessedAudioTensor());
      }
      LITERT_ASSIGN_OR_RETURN(auto single_audio_data,
                              audio_executor_->Encode(*spectrogram_tensor));
      const int num_audio_tokens = single_audio_data.GetValidTokens();
      if (num_audio_tokens > 0) {
        all_audio_data.push_back(std::move(single_audio_data));
        combined_token_ids.insert(combined_token_ids.end(), num_audio_tokens,
                                  ExecutorAudioData::kSpecialToken);
      }
    } else if (const auto* input_audio_end =
                   std::get_if<InputAudioEnd>(&content)) {
      if (audio_executor_ != nullptr) {
        auto flushed_audio_data = audio_executor_->Flush();
        if (flushed_audio_data.ok()) {
          const int flushed_tokens = flushed_audio_data->GetValidTokens();
          if (flushed_tokens > 0) {
            all_audio_data.push_back(std::move(*flushed_audio_data));
            combined_token_ids.insert(combined_token_ids.end(), flushed_tokens,
                                      ExecutorAudioData::kSpecialToken);
          }
        } else if (!absl::IsUnimplemented(flushed_audio_data.status())) {
          return flushed_audio_data.status();
        }
      }
      combined_token_ids.push_back(ExecutorAudioData::kEndToken);
    } else {
      return absl::InvalidArgumentError("Unsupported input type in contents.");
    }
  }

  if (combined_token_ids.empty()) {
    return absl::InvalidArgumentError("No token IDs found in contents.");
  }

  std::optional<ExecutorVisionData> combined_image_data = std::nullopt;
  if (!all_image_data.empty()) {
    LITERT_ASSIGN_OR_RETURN(combined_image_data,
                            CombineExecutorVisionData(all_image_data));
  }
  std::optional<ExecutorAudioData> combined_audio_data = std::nullopt;
  if (!all_audio_data.empty()) {
    LITERT_ASSIGN_OR_RETURN(combined_audio_data,
                            CombineExecutorAudioData(all_audio_data));
  }

  if (benchmark_info_.has_value() &&
      benchmark_info_->GetBenchmarkParams().num_prefill_tokens() > 0) {
    combined_token_ids.resize(
        benchmark_info_->GetBenchmarkParams().num_prefill_tokens());
  }

  LITERT_ASSIGN_OR_RETURN(
      auto token_ids_buffer,
      litert::support::Tokenizer::TokenIdsToTensorBuffer(combined_token_ids));

  return ExecutorInputs(ExecutorTextData(std::move(token_ids_buffer)),
                        std::move(combined_image_data),
                        std::move(combined_audio_data));
}

absl::StatusOr<EmbeddingResponse> EmbeddingEngineImpl::ComputeEmbeddingInternal(
    const ExecutorInputs& inputs, const EmbeddingOptions& options) {
  ComputeEmbeddingOptions compute_options{
      .input_overflow_strategy = options.input_overflow_strategy,
  };
  if (options.insert_special_tokens && !special_tokens_.eos_token_ids.empty()) {
    compute_options.eos_token_ids = special_tokens_.eos_token_ids;
  }
  LITERT_ASSIGN_OR_RETURN(
      auto embedding_output,
      embedding_executor_->ComputeEmbedding(inputs, compute_options));

  EmbeddingResponse response;
  response.embedding = std::move(embedding_output.embedding);
  response.input_length = embedding_output.input_length;
  response.truncated_length = embedding_output.truncated_length;
  response.num_chunks = embedding_output.num_chunks;

  if (options.output_size.has_value()) {
    if (*options.output_size < 0) {
      return absl::InvalidArgumentError(absl::StrCat(
          "output_size cannot be negative, got: ", *options.output_size));
    }
    if (*options.output_size > response.embedding.size()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "output_size cannot be larger than default output embedding size (",
          response.embedding.size(), "), got: ", *options.output_size));
    }
    response.embedding.resize(*options.output_size);
  }

  if (options.normalize) {
    response.embedding = L2Norm(response.embedding);
  }

  return response;
}

absl::StatusOr<EmbeddingResponse> EmbeddingEngineImpl::ComputeEmbedding(
    const std::vector<InputData>& contents, const EmbeddingOptions& options) {
  if (benchmark_info_.has_value()) {
    ABSL_RETURN_IF_ERROR(benchmark_info_->TimePrefillTurnStart());
  }

  const std::vector<InputData>* contents_to_process = &contents;
  std::vector<InputData> expanded_contents;
  if (options.insert_special_tokens) {
    LITERT_ASSIGN_OR_RETURN(expanded_contents, InsertSpecialTokens(contents));
    contents_to_process = &expanded_contents;
  }

  LITERT_ASSIGN_OR_RETURN(
      auto executor_inputs,
      ProcessAndCombineContents(*contents_to_process, options));
  ABSL_ASSIGN_OR_RETURN(auto response,
                        ComputeEmbeddingInternal(executor_inputs, options));

  if (benchmark_info_.has_value()) {
    ABSL_ASSIGN_OR_RETURN(uint64_t num_tokens, GetNumTokens(executor_inputs));
    ABSL_RETURN_IF_ERROR(benchmark_info_->TimePrefillTurnEnd(num_tokens));
  }

  return response;
}

absl::StatusOr<std::vector<EmbeddingResponse>>
EmbeddingEngineImpl::ComputeEmbeddingBatch(
    const std::vector<std::vector<InputData>>& contents,
    const EmbeddingOptions& options) {
  if (benchmark_info_.has_value()) {
    ABSL_RETURN_IF_ERROR(benchmark_info_->TimePrefillTurnStart());
  }

  std::vector<EmbeddingResponse> batch_responses;
  batch_responses.reserve(contents.size());
  uint64_t total_tokens = 0;

  for (const auto& single_contents : contents) {
    const std::vector<InputData>* contents_to_process = &single_contents;
    std::vector<InputData> expanded_contents;
    if (options.insert_special_tokens) {
      LITERT_ASSIGN_OR_RETURN(expanded_contents,
                              InsertSpecialTokens(single_contents));
      contents_to_process = &expanded_contents;
    }
    LITERT_ASSIGN_OR_RETURN(
        auto executor_inputs,
        ProcessAndCombineContents(*contents_to_process, options));
    if (benchmark_info_.has_value()) {
      ABSL_ASSIGN_OR_RETURN(uint64_t num_tokens, GetNumTokens(executor_inputs));
      total_tokens += num_tokens;
    }
    ABSL_ASSIGN_OR_RETURN(auto response,
                          ComputeEmbeddingInternal(executor_inputs, options));
    batch_responses.push_back(std::move(response));
  }

  if (benchmark_info_.has_value()) {
    ABSL_RETURN_IF_ERROR(benchmark_info_->TimePrefillTurnEnd(total_tokens));
  }

  return batch_responses;
}

absl::optional<BenchmarkInfo> EmbeddingEngineImpl::GetBenchmarkInfo() {
  return benchmark_info_;
}

BenchmarkInfo* EmbeddingEngineImpl::GetMutableBenchmarkInfo() {
  if (!benchmark_info_.has_value()) {
    benchmark_info_ = BenchmarkInfo(proto::BenchmarkParams());
  }
  return &(*benchmark_info_);
}

const std::optional<proto::EmbeddingMetadata>&
EmbeddingEngineImpl::GetEmbeddingMetadata() const {
  return metadata_;
}

}  // namespace litert::lm
