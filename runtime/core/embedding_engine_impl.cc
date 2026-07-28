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

#include <cmath>
#include <memory>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "support/tokenizer/tokenizer.h"  // from @litert
#include "support/util/io_types.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/engine/embedding_engine.h"
#include "runtime/engine/embedding_engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/audio_executor.h"
#include "runtime/executor/audio_litert_compiled_model_executor.h"
#include "runtime/executor/embedding_executor_base.h"
#include "runtime/executor/embedding_executor_settings.h"
#include "runtime/executor/embedding_litert_compiled_model_executor.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/executor/vision_executor.h"
#include "runtime/executor/vision_litert_compiled_model_executor.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/executor_data_util.h"
#include "runtime/util/litert_util.h"
#include "runtime/util/status_macros.h"
#include "runtime/util/tensor_buffer_util.h"

namespace litert::lm {

namespace {
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

}  // namespace

// static
absl::StatusOr<std::unique_ptr<EmbeddingEngine>> EmbeddingEngineImpl::Create(
    std::unique_ptr<ModelResources> resources,
    std::unique_ptr<OwnedEnvironment> env,
    std::unique_ptr<::litert::support::Tokenizer> tokenizer,
    EmbeddingEngineSettings settings) {
  if (resources == nullptr) {
    return absl::InvalidArgumentError("ModelResources cannot be null.");
  }
  if (env == nullptr) {
    return absl::InvalidArgumentError("OwnedEnvironment cannot be null.");
  }
  if (tokenizer == nullptr) {
    return absl::InvalidArgumentError("Tokenizer cannot be null.");
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

  // Initialize the embedding model executor.
  LITERT_ASSIGN_OR_RETURN(
      auto embedding_executor,
      EmbeddingLiteRtCompiledModelExecutor::Create(
          std::move(settings.GetMutableMainExecutorSettings()), env->env,
          std::move(resources)));

  return std::make_unique<EmbeddingEngineImpl>(
      std::move(env), std::move(tokenizer), std::move(embedding_executor),
      std::move(vision_executor), std::move(audio_executor));
}

EmbeddingEngineImpl::EmbeddingEngineImpl(
    std::unique_ptr<OwnedEnvironment> env,
    std::unique_ptr<::litert::support::Tokenizer> tokenizer,
    std::unique_ptr<EmbeddingExecutorBase> embedding_executor,
    std::unique_ptr<VisionExecutor> vision_executor,
    std::unique_ptr<AudioExecutor> audio_executor)
    : env_(std::move(env)),
      tokenizer_(std::move(tokenizer)),
      embedding_executor_(std::move(embedding_executor)),
      vision_executor_(std::move(vision_executor)),
      audio_executor_(std::move(audio_executor)) {}

absl::StatusOr<ExecutorInputs> EmbeddingEngineImpl::ProcessAndCombineContents(
    const std::vector<InputData>& contents) {
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
        LITERT_ASSIGN_OR_RETURN(auto raw_text, input_text->GetRawTextString());
        LITERT_ASSIGN_OR_RETURN(auto token_ids,
                                tokenizer_->TextToTokenIds(raw_text));
        combined_token_ids.insert(combined_token_ids.end(), token_ids.begin(),
                                  token_ids.end());
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
        return absl::FailedPreconditionError(
            "Image tensor or tensor map is null in contents.");
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
      LITERT_ASSIGN_OR_RETURN(const auto* spectrogram_tensor,
                              input_audio->GetPreprocessedAudioTensor());
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

  LITERT_ASSIGN_OR_RETURN(
      auto token_ids_buffer,
      litert::support::Tokenizer::TokenIdsToTensorBuffer(combined_token_ids));

  return ExecutorInputs(ExecutorTextData(std::move(token_ids_buffer)),
                        std::move(combined_image_data),
                        std::move(combined_audio_data));
}

absl::StatusOr<EmbeddingResponse> EmbeddingEngineImpl::ComputeEmbedding(
    const std::vector<InputData>& contents, const EmbeddingOptions& options) {
  LITERT_ASSIGN_OR_RETURN(auto inputs, ProcessAndCombineContents(contents));
  LITERT_ASSIGN_OR_RETURN(auto embedding,
                          embedding_executor_->ComputeEmbedding(inputs));

  EmbeddingResponse response;
  response.embedding = embedding;

  if (options.normalize) {
    response.embedding = L2Norm(response.embedding);
  }

  return response;
}

absl::StatusOr<std::vector<EmbeddingResponse>>
EmbeddingEngineImpl::ComputeEmbeddingBatch(
    const std::vector<std::vector<InputData>>& contents,
    const EmbeddingOptions& options) {
  std::vector<EmbeddingResponse> batch_responses;
  batch_responses.reserve(contents.size());
  for (const auto& single_contents : contents) {
    LITERT_ASSIGN_OR_RETURN(auto response,
                            ComputeEmbedding(single_contents, options));
    batch_responses.push_back(std::move(response));
  }
  return batch_responses;
}

}  // namespace litert::lm
