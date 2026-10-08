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
#include <vector>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "c/embedding_engine_internal.h"  // IWYU pragma: keep
#include "c/engine.h"
#include "c/engine_internal.h"  // IWYU pragma: keep
#include "c/error_reporter.h"
#include "c/error_reporter_internal.h"
#include "runtime/components/model_resources.h"
#include "runtime/core/embedding_engine_impl.h"
#include "runtime/engine/embedding_engine.h"
#include "runtime/engine/embedding_engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/embedding/embedding_executor_base.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/litert_compiled_model_executor_utils.h"

namespace {

absl::StatusOr<std::vector<litert::lm::InputData>> ToEngineInputData(
    const LiteRtLmInputData* const* inputs, size_t num_inputs) {
  if (inputs == nullptr && num_inputs > 0) {
    return absl::InvalidArgumentError(
        "inputs must not be NULL when num_inputs is non-zero.");
  }
  std::vector<litert::lm::InputData> engine_inputs;
  if (inputs == nullptr || num_inputs == 0) {
    return engine_inputs;
  }
  engine_inputs.reserve(num_inputs);
  for (size_t i = 0; i < num_inputs; ++i) {
    if (inputs[i] != nullptr) {
      auto copy_status = litert::lm::CreateInputDataCopy(inputs[i]->data);
      if (!copy_status.ok()) {
        return copy_status.status();
      }
      engine_inputs.push_back(std::move(*copy_status));
    }
  }
  return engine_inputs;
}

// Returns true if `settings` is a usable embedding engine settings handle.
// Otherwise records a kInvalidArgument last error and returns false.
bool IsValidEmbeddingEngineSettings(
    const LiteRtLmEmbeddingEngineSettings* settings) {
  if (settings != nullptr && settings->settings != nullptr) {
    return true;
  }
  litert::lm::c::SetLastError(absl::StatusCode::kInvalidArgument,
                              "Invalid embedding engine settings.");
  return false;
}

bool IsValidActivationDataType(LiteRtLmActivationDataType type) {
  switch (type) {
    case kLiteRtLmActivationDataTypeFloat32:
    case kLiteRtLmActivationDataTypeFloat16:
    case kLiteRtLmActivationDataTypeInt16:
    case kLiteRtLmActivationDataTypeInt8:
      return true;
  }
  return false;
}

bool IsValidInputOverflowStrategy(LiteRtLmInputOverflowStrategy strategy) {
  switch (strategy) {
    case kLiteRtLmInputOverflowStrategyChunkAndAverage:
    case kLiteRtLmInputOverflowStrategyTruncate:
    case kLiteRtLmInputOverflowStrategyError:
      return true;
  }
  return false;
}

}  // namespace

LiteRtLmStatusCode litert_lm_embedding_engine_settings_create(
    const char* model_path, const char* backend_str,
    const char* vision_backend_str, const char* audio_backend_str,
    LiteRtLmEmbeddingEngineSettings** out_settings) {
  LITERT_LM_C_RETURN_IF_NULL(out_settings);
  *out_settings = nullptr;
  if (model_path == nullptr || backend_str == nullptr) {
    ABSL_LOG(ERROR) << "model_path and backend_str must not be null.";
    return litert::lm::c::ReturnError(
        absl::StatusCode::kInvalidArgument,
        "model_path and backend_str must not be null.");
  }

  auto model_assets = litert::lm::ModelAssets::Create(model_path);
  if (!model_assets.ok()) {
    ABSL_LOG(ERROR) << "Failed to create model assets: "
                    << model_assets.status();
    return litert::lm::c::ToCStatus(model_assets.status());
  }

  auto backend = litert::lm::GetBackendFromString(backend_str);
  if (!backend.ok()) {
    ABSL_LOG(ERROR) << "Failed to parse backend: " << backend.status();
    return litert::lm::c::ToCStatus(backend.status());
  }

  std::optional<litert::lm::Backend> vision_backend = std::nullopt;
  if (vision_backend_str != nullptr && vision_backend_str[0] != '\0') {
    auto v_backend = litert::lm::GetBackendFromString(vision_backend_str);
    if (!v_backend.ok()) {
      ABSL_LOG(ERROR) << "Failed to parse vision backend: "
                      << v_backend.status();
      return litert::lm::c::ToCStatus(v_backend.status());
    }
    vision_backend = *v_backend;
  }

  std::optional<litert::lm::Backend> audio_backend = std::nullopt;
  if (audio_backend_str != nullptr && audio_backend_str[0] != '\0') {
    auto a_backend = litert::lm::GetBackendFromString(audio_backend_str);
    if (!a_backend.ok()) {
      ABSL_LOG(ERROR) << "Failed to parse audio backend: "
                      << a_backend.status();
      return litert::lm::c::ToCStatus(a_backend.status());
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
    return litert::lm::c::ToCStatus(settings.status());
  }

  *out_settings = new LiteRtLmEmbeddingEngineSettings{
      std::make_unique<litert::lm::EmbeddingEngineSettings>(
          std::move(*settings))};
  return kLiteRtLmStatusOk;
}

void litert_lm_embedding_engine_settings_delete(
    LiteRtLmEmbeddingEngineSettings* settings) {
  delete settings;
}

LiteRtLmStatusCode litert_lm_embedding_engine_settings_set_num_threads(
    LiteRtLmEmbeddingEngineSettings* settings, int num_threads) {
  if (!IsValidEmbeddingEngineSettings(settings)) {
    return kLiteRtLmStatusInvalidArgument;
  }
  // A non-positive value is ignored.
  if (num_threads > 0) {
    settings->settings->GetMutableMainExecutorSettings().SetNumThreads(
        num_threads);
  }
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_embedding_engine_settings_set_audio_num_threads(
    LiteRtLmEmbeddingEngineSettings* settings, int num_threads) {
  if (!IsValidEmbeddingEngineSettings(settings)) {
    return kLiteRtLmStatusInvalidArgument;
  }
  // A non-positive value, or no audio executor, makes this a no-op.
  if (num_threads > 0 &&
      settings->settings->GetAudioExecutorSettings().has_value()) {
    settings->settings->GetMutableAudioExecutorSettings()->SetNumThreads(
        num_threads);
  }
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_embedding_engine_settings_set_cache_dir(
    LiteRtLmEmbeddingEngineSettings* settings, const char* cache_dir) {
  if (!IsValidEmbeddingEngineSettings(settings)) {
    return kLiteRtLmStatusInvalidArgument;
  }
  LITERT_LM_C_RETURN_IF_NULL(cache_dir);
  settings->settings->GetMutableMainExecutorSettings().SetCacheDir(cache_dir);
  if (settings->settings->GetVisionExecutorSettings().has_value()) {
    settings->settings->GetMutableVisionExecutorSettings()->SetCacheDir(
        cache_dir);
  }
  if (settings->settings->GetAudioExecutorSettings().has_value()) {
    settings->settings->GetMutableAudioExecutorSettings()->SetCacheDir(
        cache_dir);
  }
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode
litert_lm_embedding_engine_settings_set_litert_dispatch_lib_dir(
    LiteRtLmEmbeddingEngineSettings* settings, const char* lib_dir) {
  if (!IsValidEmbeddingEngineSettings(settings)) {
    return kLiteRtLmStatusInvalidArgument;
  }
  LITERT_LM_C_RETURN_IF_NULL(lib_dir);
  settings->settings->GetMutableMainExecutorSettings().SetLitertDispatchLibDir(
      lib_dir);
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode
litert_lm_embedding_engine_settings_set_vision_litert_dispatch_lib_dir(
    LiteRtLmEmbeddingEngineSettings* settings, const char* lib_dir) {
  if (!IsValidEmbeddingEngineSettings(settings)) {
    return kLiteRtLmStatusInvalidArgument;
  }
  LITERT_LM_C_RETURN_IF_NULL(lib_dir);
  // No-op if no vision executor is configured.
  if (settings->settings->GetVisionExecutorSettings().has_value()) {
    settings->settings->GetMutableVisionExecutorSettings()
        ->SetLitertDispatchLibDir(lib_dir);
  }
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode
litert_lm_embedding_engine_settings_set_audio_litert_dispatch_lib_dir(
    LiteRtLmEmbeddingEngineSettings* settings, const char* lib_dir) {
  if (!IsValidEmbeddingEngineSettings(settings)) {
    return kLiteRtLmStatusInvalidArgument;
  }
  LITERT_LM_C_RETURN_IF_NULL(lib_dir);
  // No-op if no audio executor is configured.
  if (settings->settings->GetAudioExecutorSettings().has_value()) {
    settings->settings->GetMutableAudioExecutorSettings()
        ->SetLitertDispatchLibDir(lib_dir);
  }
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_embedding_engine_settings_set_max_input_length(
    LiteRtLmEmbeddingEngineSettings* settings, int max_input_length) {
  if (!IsValidEmbeddingEngineSettings(settings)) {
    return kLiteRtLmStatusInvalidArgument;
  }
  if (max_input_length > 0) {
    settings->settings->SetMaxInputLength(max_input_length);
  } else {
    settings->settings->SetMaxInputLength(std::nullopt);
  }
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_embedding_engine_settings_set_min_input_length(
    LiteRtLmEmbeddingEngineSettings* settings, int min_input_length) {
  if (!IsValidEmbeddingEngineSettings(settings)) {
    return kLiteRtLmStatusInvalidArgument;
  }
  if (min_input_length >= 0) {
    settings->settings->SetMinInputLength(min_input_length);
  } else {
    settings->settings->SetMinInputLength(std::nullopt);
  }
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode
litert_lm_embedding_engine_settings_set_vision_tokens_per_image(
    LiteRtLmEmbeddingEngineSettings* settings, int vision_tokens_per_image) {
  if (!IsValidEmbeddingEngineSettings(settings)) {
    return kLiteRtLmStatusInvalidArgument;
  }
  if (vision_tokens_per_image > 0) {
    settings->settings->SetVisionTokensPerImage(vision_tokens_per_image);
  } else {
    settings->settings->SetVisionTokensPerImage(std::nullopt);
  }
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_embedding_engine_settings_set_activation_data_type(
    LiteRtLmEmbeddingEngineSettings* settings,
    LiteRtLmActivationDataType activation_data_type) {
  if (!IsValidEmbeddingEngineSettings(settings)) {
    return kLiteRtLmStatusInvalidArgument;
  }
  if (!IsValidActivationDataType(activation_data_type)) {
    return litert::lm::c::ReturnError(absl::StatusCode::kInvalidArgument,
                                      "Unknown LiteRtLmActivationDataType.");
  }
  settings->settings->GetMutableMainExecutorSettings().SetActivationDataType(
      static_cast<litert::lm::ActivationDataType>(activation_data_type));
  if (settings->settings->GetVisionExecutorSettings().has_value()) {
    settings->settings->GetMutableVisionExecutorSettings()
        ->SetActivationDataType(
            static_cast<litert::lm::ActivationDataType>(activation_data_type));
  }
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_embedding_options_create(
    LiteRtLmEmbeddingOptions** out_options) {
  LITERT_LM_C_RETURN_IF_NULL(out_options);
  *out_options = new LiteRtLmEmbeddingOptions{litert::lm::EmbeddingOptions{}};
  return kLiteRtLmStatusOk;
}

void litert_lm_embedding_options_delete(LiteRtLmEmbeddingOptions* options) {
  delete options;
}

LiteRtLmStatusCode litert_lm_embedding_options_set_normalize(
    LiteRtLmEmbeddingOptions* options, bool normalize) {
  LITERT_LM_C_RETURN_IF_NULL(options);
  options->options.normalize = normalize;
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_embedding_options_get_normalize(
    const LiteRtLmEmbeddingOptions* options, bool* out_normalize) {
  LITERT_LM_C_RETURN_IF_NULL(out_normalize);
  LITERT_LM_C_RETURN_IF_NULL(options);
  *out_normalize = options->options.normalize;
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_embedding_options_set_insert_special_tokens(
    LiteRtLmEmbeddingOptions* options, bool insert_special_tokens) {
  LITERT_LM_C_RETURN_IF_NULL(options);
  options->options.insert_special_tokens = insert_special_tokens;
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_embedding_options_get_insert_special_tokens(
    const LiteRtLmEmbeddingOptions* options, bool* out_insert_special_tokens) {
  LITERT_LM_C_RETURN_IF_NULL(out_insert_special_tokens);
  LITERT_LM_C_RETURN_IF_NULL(options);
  *out_insert_special_tokens = options->options.insert_special_tokens;
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_embedding_options_set_input_overflow_strategy(
    LiteRtLmEmbeddingOptions* options, LiteRtLmInputOverflowStrategy strategy) {
  LITERT_LM_C_RETURN_IF_NULL(options);
  if (!IsValidInputOverflowStrategy(strategy)) {
    return litert::lm::c::ReturnError(absl::StatusCode::kInvalidArgument,
                                      "Unknown LiteRtLmInputOverflowStrategy.");
  }
  options->options.input_overflow_strategy =
      static_cast<litert::lm::InputOverflowStrategy>(strategy);
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_embedding_options_get_input_overflow_strategy(
    const LiteRtLmEmbeddingOptions* options,
    LiteRtLmInputOverflowStrategy* out_strategy) {
  LITERT_LM_C_RETURN_IF_NULL(out_strategy);
  LITERT_LM_C_RETURN_IF_NULL(options);
  *out_strategy = static_cast<LiteRtLmInputOverflowStrategy>(
      options->options.input_overflow_strategy);
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_embedding_options_set_output_size(
    LiteRtLmEmbeddingOptions* options, int output_size) {
  LITERT_LM_C_RETURN_IF_NULL(options);
  if (output_size <= 0) {
    options->options.output_size = std::nullopt;
  } else {
    options->options.output_size = output_size;
  }
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_embedding_options_get_output_size(
    const LiteRtLmEmbeddingOptions* options, int* out_output_size) {
  LITERT_LM_C_RETURN_IF_NULL(out_output_size);
  LITERT_LM_C_RETURN_IF_NULL(options);
  if (!options->options.output_size.has_value()) {
    return litert::lm::c::ReturnError(absl::StatusCode::kNotFound,
                                      "output_size is not set.");
  }
  *out_output_size = *options->options.output_size;
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_embedding_options_set_vision_tokens_per_image(
    LiteRtLmEmbeddingOptions* options, int vision_tokens_per_image) {
  LITERT_LM_C_RETURN_IF_NULL(options);
  if (vision_tokens_per_image > 0) {
    options->options.vision_tokens_per_image = vision_tokens_per_image;
  } else {
    options->options.vision_tokens_per_image = std::nullopt;
  }
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_embedding_options_get_vision_tokens_per_image(
    const LiteRtLmEmbeddingOptions* options, int* out_vision_tokens_per_image) {
  LITERT_LM_C_RETURN_IF_NULL(out_vision_tokens_per_image);
  LITERT_LM_C_RETURN_IF_NULL(options);
  if (!options->options.vision_tokens_per_image.has_value()) {
    return litert::lm::c::ReturnError(absl::StatusCode::kNotFound,
                                      "vision_tokens_per_image is not set.");
  }
  *out_vision_tokens_per_image = *options->options.vision_tokens_per_image;
  return kLiteRtLmStatusOk;
}

void litert_lm_embedding_response_delete(LiteRtLmEmbeddingResponse* response) {
  delete response;
}

LiteRtLmStatusCode litert_lm_embedding_response_get_size(
    const LiteRtLmEmbeddingResponse* response, size_t* out_size) {
  LITERT_LM_C_RETURN_IF_NULL(out_size);
  LITERT_LM_C_RETURN_IF_NULL(response);
  *out_size = response->response.embedding.size();
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_embedding_response_get_values(
    const LiteRtLmEmbeddingResponse* response, const float** out_values) {
  LITERT_LM_C_RETURN_IF_NULL(out_values);
  *out_values = nullptr;
  LITERT_LM_C_RETURN_IF_NULL(response);
  // An empty embedding is reported as success with a NULL array.
  if (!response->response.embedding.empty()) {
    *out_values = response->response.embedding.data();
  }
  return kLiteRtLmStatusOk;
}

void litert_lm_embedding_responses_delete(
    LiteRtLmEmbeddingResponses* responses) {
  delete responses;
}

LiteRtLmStatusCode litert_lm_embedding_responses_get_size(
    const LiteRtLmEmbeddingResponses* responses, size_t* out_size) {
  LITERT_LM_C_RETURN_IF_NULL(out_size);
  LITERT_LM_C_RETURN_IF_NULL(responses);
  *out_size = responses->responses.size();
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_embedding_responses_get_at(
    const LiteRtLmEmbeddingResponses* responses, size_t index,
    const LiteRtLmEmbeddingResponse** out_response) {
  LITERT_LM_C_RETURN_IF_NULL(out_response);
  *out_response = nullptr;
  LITERT_LM_C_RETURN_IF_NULL(responses);
  if (index >= responses->responses.size()) {
    return litert::lm::c::ReturnError(
        absl::StatusCode::kOutOfRange,
        absl::StrFormat("Embedding response index %u is out of range; the "
                        "batch has %u response(s).",
                        index, responses->responses.size()));
  }
  *out_response = reinterpret_cast<const LiteRtLmEmbeddingResponse*>(
      &responses->responses[index]);
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_embedding_engine_create(
    const LiteRtLmEmbeddingEngineSettings* settings,
    LiteRtLmEmbeddingEngine** out_engine) {
  LITERT_LM_C_RETURN_IF_NULL(out_engine);
  *out_engine = nullptr;
  if (settings == nullptr || settings->settings == nullptr) {
    ABSL_LOG(ERROR) << "Settings must not be null.";
    return litert::lm::c::ReturnError(absl::StatusCode::kInvalidArgument,
                                      "Settings must not be null.");
  }

  auto engine = litert::lm::EmbeddingEngineImpl::Create(*settings->settings);
  if (!engine.ok()) {
    ABSL_LOG(ERROR) << "Failed to create EmbeddingEngine: " << engine.status();
    return litert::lm::c::ToCStatus(engine.status());
  }

  *out_engine = new LiteRtLmEmbeddingEngine{std::move(*engine)};
  return kLiteRtLmStatusOk;
}

void litert_lm_embedding_engine_delete(LiteRtLmEmbeddingEngine* engine) {
  delete engine;
}

LiteRtLmStatusCode litert_lm_embedding_engine_compute_embedding(
    LiteRtLmEmbeddingEngine* engine, const LiteRtLmInputData* const* inputs,
    size_t num_inputs, const LiteRtLmEmbeddingOptions* options,
    LiteRtLmEmbeddingResponse** out_response) {
  LITERT_LM_C_RETURN_IF_NULL(out_response);
  *out_response = nullptr;
  if (!engine || !engine->engine) {
    ABSL_LOG(ERROR) << "EmbeddingEngine is null.";
    return litert::lm::c::ReturnError(absl::StatusCode::kInvalidArgument,
                                      "EmbeddingEngine is null.");
  }

  auto engine_inputs = ToEngineInputData(inputs, num_inputs);
  if (!engine_inputs.ok()) {
    ABSL_LOG(ERROR) << "Failed to convert input data: "
                    << engine_inputs.status();
    return litert::lm::c::ToCStatus(engine_inputs.status());
  }

  litert::lm::EmbeddingOptions opts =
      options ? options->options : litert::lm::EmbeddingOptions{};

  auto response = engine->engine->ComputeEmbedding(*engine_inputs, opts);
  if (!response.ok()) {
    ABSL_LOG(ERROR) << "ComputeEmbedding failed: " << response.status();
    return litert::lm::c::ToCStatus(response.status());
  }

  *out_response = new LiteRtLmEmbeddingResponse{std::move(*response)};
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_embedding_engine_compute_embedding_batch(
    LiteRtLmEmbeddingEngine* engine,
    const LiteRtLmInputData* const* const* inputs_batch,
    const size_t* num_inputs_per_batch, size_t batch_size,
    const LiteRtLmEmbeddingOptions* options,
    LiteRtLmEmbeddingResponses** out_responses) {
  LITERT_LM_C_RETURN_IF_NULL(out_responses);
  *out_responses = nullptr;
  if (!engine || !engine->engine) {
    ABSL_LOG(ERROR) << "EmbeddingEngine is null.";
    return litert::lm::c::ReturnError(absl::StatusCode::kInvalidArgument,
                                      "EmbeddingEngine is null.");
  }

  if (batch_size > 0 &&
      (inputs_batch == nullptr || num_inputs_per_batch == nullptr)) {
    return litert::lm::c::ReturnError(
        absl::StatusCode::kInvalidArgument,
        "inputs_batch and num_inputs_per_batch must not be NULL when "
        "batch_size is non-zero.");
  }

  std::vector<std::vector<litert::lm::InputData>> contents_batch;
  contents_batch.reserve(batch_size);

  for (size_t i = 0; i < batch_size; ++i) {
    auto engine_inputs =
        ToEngineInputData(inputs_batch[i], num_inputs_per_batch[i]);
    if (!engine_inputs.ok()) {
      ABSL_LOG(ERROR) << "Failed to convert input data for batch index " << i
                      << ": " << engine_inputs.status();
      return litert::lm::c::ToCStatus(engine_inputs.status());
    }
    contents_batch.push_back(std::move(*engine_inputs));
  }

  litert::lm::EmbeddingOptions opts =
      options ? options->options : litert::lm::EmbeddingOptions{};

  auto responses = engine->engine->ComputeEmbeddingBatch(contents_batch, opts);
  if (!responses.ok()) {
    ABSL_LOG(ERROR) << "ComputeEmbeddingBatch failed: " << responses.status();
    return litert::lm::c::ToCStatus(responses.status());
  }

  *out_responses = new LiteRtLmEmbeddingResponses{std::move(*responses)};
  return kLiteRtLmStatusOk;
}
