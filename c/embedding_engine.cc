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

#include "c/embedding_engine.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "c/embedding_engine_internal.h"  // IWYU pragma: keep
#include "c/engine.h"
#include "c/engine_internal.h"  // IWYU pragma: keep
#include "runtime/components/model_resources.h"
#include "runtime/core/embedding_engine_impl.h"
#include "runtime/engine/embedding_engine.h"
#include "runtime/engine/embedding_engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/embedding/embedding_executor_base.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/litert_compiled_model_executor_utils.h"
#include "support/preprocessor/audio_preprocessor.h"
#include "support/preprocessor/audio_preprocessor_miniaudio.h"

namespace {

absl::StatusOr<std::vector<litert::lm::InputData>> ToEngineInputData(
    const LiteRtLmInputData* const* inputs, size_t num_inputs) {
  std::vector<litert::lm::InputData> engine_inputs;
  if (inputs == nullptr || num_inputs == 0) {
    return engine_inputs;
  }
  engine_inputs.reserve(num_inputs);
  for (size_t i = 0; i < num_inputs; ++i) {
    if (inputs[i] != nullptr) {
      if (const auto* raw_audio =
              std::get_if<litert::lm::InputAudio>(&inputs[i]->data)) {
        if (!raw_audio->GetPreprocessedAudioTensor().ok()) {
          auto preprocessor =
              ::litert::support::AudioPreprocessorMiniAudio::Create(
                  ::litert::support::AudioPreprocessorConfig::
                      CreateDefaultUsmConfig());
          if (!preprocessor.ok()) {
            return preprocessor.status();
          }
          auto processed_audio = (*preprocessor)->Preprocess(*raw_audio);
          if (!processed_audio.ok()) {
            return processed_audio.status();
          }
          engine_inputs.push_back(std::move(*processed_audio));
          continue;
        }
      }
      auto copy_status = litert::lm::CreateInputDataCopy(inputs[i]->data);
      if (!copy_status.ok()) {
        return copy_status.status();
      }
      engine_inputs.push_back(std::move(*copy_status));
    }
  }
  return engine_inputs;
}

}  // namespace

LiteRtLmEmbeddingEngineSettings* litert_lm_embedding_engine_settings_create(
    const char* model_path, const char* backend_str,
    const char* vision_backend_str, const char* audio_backend_str) {
  if (model_path == nullptr || backend_str == nullptr) {
    ABSL_LOG(ERROR) << "model_path and backend_str must not be null.";
    return nullptr;
  }

  auto model_assets = litert::lm::ModelAssets::Create(model_path);
  if (!model_assets.ok()) {
    ABSL_LOG(ERROR) << "Failed to create model assets: "
                    << model_assets.status();
    return nullptr;
  }

  auto backend = litert::lm::GetBackendFromString(backend_str);
  if (!backend.ok()) {
    ABSL_LOG(ERROR) << "Failed to parse backend: " << backend.status();
    return nullptr;
  }

  std::optional<litert::lm::Backend> vision_backend = std::nullopt;
  if (vision_backend_str != nullptr && vision_backend_str[0] != '\0') {
    auto v_backend = litert::lm::GetBackendFromString(vision_backend_str);
    if (!v_backend.ok()) {
      ABSL_LOG(ERROR) << "Failed to parse vision backend: "
                      << v_backend.status();
      return nullptr;
    }
    vision_backend = *v_backend;
  }

  std::optional<litert::lm::Backend> audio_backend = std::nullopt;
  if (audio_backend_str != nullptr && audio_backend_str[0] != '\0') {
    auto a_backend = litert::lm::GetBackendFromString(audio_backend_str);
    if (!a_backend.ok()) {
      ABSL_LOG(ERROR) << "Failed to parse audio backend: "
                      << a_backend.status();
      return nullptr;
    }
    audio_backend = *a_backend;
  }

  if (!vision_backend.has_value() || !audio_backend.has_value()) {
    auto resources = litert::lm::BuildLiteRtCompiledModelResources(
        *model_assets, /*enable_file_backed_model_loading=*/false);
    if (resources.ok() && *resources != nullptr) {
      if (!vision_backend.has_value() &&
          (*resources)
              ->GetTFLiteModel(litert::lm::ModelType::kTfLiteVisionEncoder)
              .ok()) {
        vision_backend = *backend;
      }
      if (!audio_backend.has_value() &&
          ((*resources)
               ->GetTFLiteModel(litert::lm::ModelType::kTfLiteAudioEncoderHw)
               .ok() ||
           (*resources)
               ->GetTFLiteModel(litert::lm::ModelType::kTfLiteAudioFrontend)
               .ok())) {
        audio_backend = *backend;
      }
    }
  }

  auto settings = litert::lm::EmbeddingEngineSettings::CreateDefault(
      std::move(*model_assets), *backend, vision_backend, audio_backend);
  if (!settings.ok()) {
    ABSL_LOG(ERROR) << "Failed to create embedding engine settings: "
                    << settings.status();
    return nullptr;
  }

  return new LiteRtLmEmbeddingEngineSettings{
      std::make_unique<litert::lm::EmbeddingEngineSettings>(
          std::move(*settings))};
}

void litert_lm_embedding_engine_settings_delete(
    LiteRtLmEmbeddingEngineSettings* settings) {
  delete settings;
}

void litert_lm_embedding_engine_settings_set_audio_num_threads(
    LiteRtLmEmbeddingEngineSettings* settings, int num_threads) {
  if (settings && settings->settings && num_threads > 0 &&
      settings->settings->GetAudioExecutorSettings().has_value()) {
    settings->settings->GetMutableAudioExecutorSettings()->SetNumThreads(
        num_threads);
  }
}

void litert_lm_embedding_engine_settings_set_cache_dir(
    LiteRtLmEmbeddingEngineSettings* settings, const char* cache_dir) {
  if (settings && settings->settings && cache_dir != nullptr) {
    settings->settings->GetMutableMainExecutorSettings().SetCacheDir(cache_dir);
    if (settings->settings->GetVisionExecutorSettings().has_value()) {
      settings->settings->GetMutableVisionExecutorSettings()->SetCacheDir(
          cache_dir);
    }
    if (settings->settings->GetAudioExecutorSettings().has_value()) {
      settings->settings->GetMutableAudioExecutorSettings()->SetCacheDir(
          cache_dir);
    }
  }
}

void litert_lm_embedding_engine_settings_set_litert_dispatch_lib_dir(
    LiteRtLmEmbeddingEngineSettings* settings, const char* lib_dir) {
  if (settings && settings->settings && lib_dir != nullptr) {
    settings->settings->GetMutableMainExecutorSettings()
        .SetLitertDispatchLibDir(lib_dir);
  }
}

void litert_lm_embedding_engine_settings_set_vision_litert_dispatch_lib_dir(
    LiteRtLmEmbeddingEngineSettings* settings, const char* lib_dir) {
  if (settings && settings->settings && lib_dir != nullptr &&
      settings->settings->GetVisionExecutorSettings().has_value()) {
    settings->settings->GetMutableVisionExecutorSettings()
        ->SetLitertDispatchLibDir(lib_dir);
  }
}

void litert_lm_embedding_engine_settings_set_audio_litert_dispatch_lib_dir(
    LiteRtLmEmbeddingEngineSettings* settings, const char* lib_dir) {
  if (settings && settings->settings && lib_dir != nullptr &&
      settings->settings->GetAudioExecutorSettings().has_value()) {
    settings->settings->GetMutableAudioExecutorSettings()
        ->SetLitertDispatchLibDir(lib_dir);
  }
}

void litert_lm_embedding_engine_settings_set_max_input_length(
    LiteRtLmEmbeddingEngineSettings* settings, int max_input_length) {
  if (settings && settings->settings) {
    if (max_input_length > 0) {
      settings->settings->SetMaxInputLength(max_input_length);
    } else {
      settings->settings->SetMaxInputLength(std::nullopt);
    }
  }
}

void litert_lm_embedding_engine_settings_set_vision_tokens_per_image(
    LiteRtLmEmbeddingEngineSettings* settings, int vision_tokens_per_image) {
  if (settings && settings->settings) {
    if (vision_tokens_per_image > 0) {
      settings->settings->SetVisionTokensPerImage(vision_tokens_per_image);
    } else {
      settings->settings->SetVisionTokensPerImage(std::nullopt);
    }
  }
}

LiteRtLmEmbeddingOptions* litert_lm_embedding_options_create(void) {
  return new LiteRtLmEmbeddingOptions{litert::lm::EmbeddingOptions{}};
}

void litert_lm_embedding_options_delete(LiteRtLmEmbeddingOptions* options) {
  delete options;
}

void litert_lm_embedding_options_set_normalize(
    LiteRtLmEmbeddingOptions* options, bool normalize) {
  if (options) {
    options->options.normalize = normalize;
  }
}

bool litert_lm_embedding_options_get_normalize(
    const LiteRtLmEmbeddingOptions* options) {
  if (!options) {
    return false;
  }
  return options->options.normalize;
}

void litert_lm_embedding_options_set_insert_special_tokens(
    LiteRtLmEmbeddingOptions* options, bool insert_special_tokens) {
  if (options) {
    options->options.insert_special_tokens = insert_special_tokens;
  }
}

bool litert_lm_embedding_options_get_insert_special_tokens(
    const LiteRtLmEmbeddingOptions* options) {
  if (!options) {
    return false;
  }
  return options->options.insert_special_tokens;
}

void litert_lm_embedding_options_set_input_overflow_strategy(
    LiteRtLmEmbeddingOptions* options, LiteRtLmInputOverflowStrategy strategy) {
  if (options) {
    options->options.input_overflow_strategy =
        static_cast<litert::lm::InputOverflowStrategy>(strategy);
  }
}

LiteRtLmInputOverflowStrategy
litert_lm_embedding_options_get_input_overflow_strategy(
    const LiteRtLmEmbeddingOptions* options) {
  if (!options) {
    return kLiteRtLmInputOverflowStrategyError;
  }
  return static_cast<LiteRtLmInputOverflowStrategy>(
      options->options.input_overflow_strategy);
}

void litert_lm_embedding_options_set_output_size(
    LiteRtLmEmbeddingOptions* options, int output_size) {
  if (options) {
    if (output_size <= 0) {
      options->options.output_size = std::nullopt;
    } else {
      options->options.output_size = output_size;
    }
  }
}

int litert_lm_embedding_options_get_output_size(
    const LiteRtLmEmbeddingOptions* options) {
  if (!options || !options->options.output_size.has_value()) {
    return -1;
  }
  return *options->options.output_size;
}

void litert_lm_embedding_options_set_vision_tokens_per_image(
    LiteRtLmEmbeddingOptions* options, int vision_tokens_per_image) {
  if (options) {
    if (vision_tokens_per_image > 0) {
      options->options.vision_tokens_per_image = vision_tokens_per_image;
    } else {
      options->options.vision_tokens_per_image = std::nullopt;
    }
  }
}

int litert_lm_embedding_options_get_vision_tokens_per_image(
    const LiteRtLmEmbeddingOptions* options) {
  if (!options || !options->options.vision_tokens_per_image.has_value()) {
    return 0;
  }
  return *options->options.vision_tokens_per_image;
}
void litert_lm_embedding_response_delete(LiteRtLmEmbeddingResponse* response) {
  delete response;
}

size_t litert_lm_embedding_response_get_size(
    const LiteRtLmEmbeddingResponse* response) {
  if (!response) {
    return 0;
  }
  return response->response.embedding.size();
}

const float* litert_lm_embedding_response_get_values(
    const LiteRtLmEmbeddingResponse* response) {
  if (!response || response->response.embedding.empty()) {
    return nullptr;
  }
  return response->response.embedding.data();
}

void litert_lm_embedding_responses_delete(
    LiteRtLmEmbeddingResponses* responses) {
  delete responses;
}

size_t litert_lm_embedding_responses_get_size(
    const LiteRtLmEmbeddingResponses* responses) {
  if (!responses) {
    return 0;
  }
  return responses->responses.size();
}

const LiteRtLmEmbeddingResponse* litert_lm_embedding_responses_get_at(
    const LiteRtLmEmbeddingResponses* responses, size_t index) {
  if (!responses || index >= responses->responses.size()) {
    return nullptr;
  }
  return reinterpret_cast<const LiteRtLmEmbeddingResponse*>(
      &responses->responses[index]);
}

LiteRtLmEmbeddingEngine* litert_lm_embedding_engine_create(
    const LiteRtLmEmbeddingEngineSettings* settings) {
  if (settings == nullptr || settings->settings == nullptr) {
    ABSL_LOG(ERROR) << "Settings must not be null.";
    return nullptr;
  }

  auto engine = litert::lm::EmbeddingEngineImpl::Create(*settings->settings);
  if (!engine.ok()) {
    ABSL_LOG(ERROR) << "Failed to create EmbeddingEngine: " << engine.status();
    return nullptr;
  }

  return new LiteRtLmEmbeddingEngine{std::move(*engine)};
}

void litert_lm_embedding_engine_delete(LiteRtLmEmbeddingEngine* engine) {
  delete engine;
}

LiteRtLmEmbeddingResponse* litert_lm_embedding_engine_compute_embedding(
    LiteRtLmEmbeddingEngine* engine, const LiteRtLmInputData* const* inputs,
    size_t num_inputs, const LiteRtLmEmbeddingOptions* options) {
  if (!engine || !engine->engine) {
    ABSL_LOG(ERROR) << "EmbeddingEngine is null.";
    return nullptr;
  }

  auto engine_inputs = ToEngineInputData(inputs, num_inputs);
  if (!engine_inputs.ok()) {
    ABSL_LOG(ERROR) << "Failed to convert input data: "
                    << engine_inputs.status();
    return nullptr;
  }

  litert::lm::EmbeddingOptions opts =
      options ? options->options : litert::lm::EmbeddingOptions{};

  auto response = engine->engine->ComputeEmbedding(*engine_inputs, opts);
  if (!response.ok()) {
    ABSL_LOG(ERROR) << "ComputeEmbedding failed: " << response.status();
    return nullptr;
  }

  return new LiteRtLmEmbeddingResponse{std::move(*response)};
}

LiteRtLmEmbeddingResponses* litert_lm_embedding_engine_compute_embedding_batch(
    LiteRtLmEmbeddingEngine* engine,
    const LiteRtLmInputData* const* const* inputs_batch,
    const size_t* num_inputs_per_batch, size_t batch_size,
    const LiteRtLmEmbeddingOptions* options) {
  if (!engine || !engine->engine) {
    ABSL_LOG(ERROR) << "EmbeddingEngine is null.";
    return nullptr;
  }

  std::vector<std::vector<litert::lm::InputData>> contents_batch;
  contents_batch.reserve(batch_size);

  for (size_t i = 0; i < batch_size; ++i) {
    size_t num_inputs = num_inputs_per_batch ? num_inputs_per_batch[i] : 0;
    const LiteRtLmInputData* const* inputs =
        inputs_batch ? inputs_batch[i] : nullptr;
    auto engine_inputs = ToEngineInputData(inputs, num_inputs);
    if (!engine_inputs.ok()) {
      ABSL_LOG(ERROR) << "Failed to convert input data for batch index " << i
                      << ": " << engine_inputs.status();
      return nullptr;
    }
    contents_batch.push_back(std::move(*engine_inputs));
  }

  litert::lm::EmbeddingOptions opts =
      options ? options->options : litert::lm::EmbeddingOptions{};

  auto responses = engine->engine->ComputeEmbeddingBatch(contents_batch, opts);
  if (!responses.ok()) {
    ABSL_LOG(ERROR) << "ComputeEmbeddingBatch failed: " << responses.status();
    return nullptr;
  }

  return new LiteRtLmEmbeddingResponses{std::move(*responses)};
}
