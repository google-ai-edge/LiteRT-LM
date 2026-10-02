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

#include "c/engine.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#if defined(_WIN32)
#include <io.h>
#endif
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "c/engine_internal.h"
#include "c/error_reporter.h"
#include "c/error_reporter_internal.h"
#include "runtime/components/constrained_decoding/no_repeat_ngram_config.h"
#include "runtime/components/constrained_decoding/repetition_penalty_config.h"
#include "runtime/components/constrained_decoding/suppress_tokens_config.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_factory.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/proto/sampler_params.pb.h"
#include "runtime/proto/token.pb.h"
#include "runtime/util/logging.h"
#include "runtime/util/scoped_file.h"
#include "support/tokenizer/tokenizer.h"

namespace {

absl::AnyInvocable<void(absl::StatusOr<litert::lm::Responses>)> CreateCallback(
    LiteRtLmStreamCallback callback, void* callback_data) {
  return [callback,
          callback_data](absl::StatusOr<litert::lm::Responses> responses) {
    if (!responses.ok()) {
      LiteRtLmStreamChunk chunk;
      chunk.text = nullptr;
      chunk.is_final = true;
      std::string error_str = responses.status().ToString();
      chunk.error_msg = error_str.c_str();
      callback(callback_data, &chunk);
      return;
    }
    if (responses->GetTaskState() == litert::lm::TaskState::kDone) {
      LiteRtLmStreamChunk chunk;
      chunk.text = nullptr;
      chunk.is_final = true;
      chunk.error_msg = nullptr;
      callback(callback_data, &chunk);
    } else if (responses->GetTaskState() ==
               litert::lm::TaskState::kMaxNumTokensReached) {
      LiteRtLmStreamChunk chunk;
      chunk.text = nullptr;
      chunk.is_final = true;
      chunk.error_msg = "Max number of tokens reached.";
      callback(callback_data, &chunk);
    } else if (responses->GetTaskState() == litert::lm::TaskState::kCancelled) {
      LiteRtLmStreamChunk chunk;
      chunk.text = nullptr;
      chunk.is_final = true;
      chunk.error_msg = "CANCELLED.";
      callback(callback_data, &chunk);
    } else {
      for (const auto& text : responses->GetTexts()) {
        LiteRtLmStreamChunk chunk;
        chunk.text = text.data();
        chunk.is_final = false;
        chunk.error_msg = nullptr;
        callback(callback_data, &chunk);
      }
    }
  };
}

absl::StatusOr<std::vector<litert::lm::InputData>> ToEngineInputData(
    const LiteRtLmInputData* const* inputs, size_t num_inputs) {
  if (inputs == nullptr && num_inputs > 0) {
    return absl::InvalidArgumentError(
        "inputs must not be NULL when num_inputs is non-zero.");
  }
  std::vector<litert::lm::InputData> engine_inputs;
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

// Returns true if `settings` is a usable engine settings handle. Otherwise
// records a kInvalidArgument last error and returns false.
bool IsValidEngineSettings(const LiteRtLmEngineSettings* settings) {
  if (settings != nullptr && settings->settings != nullptr) {
    return true;
  }
  litert::lm::c::SetLastError(absl::StatusCode::kInvalidArgument,
                              "Invalid engine settings.");
  return false;
}

// Returns true if `config` is a usable session config handle. Otherwise
// records a kInvalidArgument last error and returns false.
bool IsValidSessionConfig(const LiteRtLmSessionConfig* config) {
  if (config != nullptr && config->config != nullptr) {
    return true;
  }
  litert::lm::c::SetLastError(absl::StatusCode::kInvalidArgument,
                              "Invalid session config.");
  return false;
}

bool IsValidLogSeverity(LiteRtLmLogSeverity level) {
  switch (level) {
    case kLiteRtLmLogSeverityVerbose:
    case kLiteRtLmLogSeverityDebug:
    case kLiteRtLmLogSeverityInfo:
    case kLiteRtLmLogSeverityWarning:
    case kLiteRtLmLogSeverityError:
    case kLiteRtLmLogSeverityFatal:
    case kLiteRtLmLogSeveritySilent:
      return true;
  }
  return false;
}

bool IsValidSamplerType(LiteRtLmSamplerType type) {
  switch (type) {
    case kLiteRtLmSamplerTypeUnspecified:
    case kLiteRtLmSamplerTypeTopK:
    case kLiteRtLmSamplerTypeTopP:
    case kLiteRtLmSamplerTypeGreedy:
      return true;
  }
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

// Returns the number of candidates in `responses`: the number of texts or, if
// there are none, the number of scores or token lengths.
size_t NumCandidates(const litert::lm::Responses& responses) {
  size_t num_candidates = responses.GetTexts().size();
  if (num_candidates == 0) {
    num_candidates = responses.GetScores().size();
  }
  if (num_candidates == 0 && responses.GetTokenLengths().has_value()) {
    num_candidates = responses.GetTokenLengths()->size();
  }
  return num_candidates;
}

// Returns an OutOfRange error if `index` is not a valid candidate index of
// `responses`.
absl::Status CheckCandidateIndex(const litert::lm::Responses& responses,
                                 int index) {
  const size_t num_candidates = NumCandidates(responses);
  if (index < 0 || static_cast<size_t>(index) >= num_candidates) {
    return absl::OutOfRangeError(
        absl::StrFormat("Response index %d is out of range; the responses "
                        "have %d candidate(s).",
                        index, num_candidates));
  }
  return absl::OkStatus();
}

// Returns true if `values` holds an element at the already range-checked
// candidate `index`.
template <typename Container>
bool HasValueAt(const Container& values, int index) {
  return static_cast<size_t>(index) < values.size();
}

// Returns a NotFound error naming `what` at candidate `index`.
LiteRtLmStatusCode ReturnNotFoundAt(absl::string_view what, int index) {
  return litert::lm::c::ReturnError(
      absl::StatusCode::kNotFound,
      absl::StrFormat("No %s available at response index %d.", what, index));
}

}  // namespace

using ::litert::lm::Engine;
using ::litert::lm::EngineFactory;
using ::litert::lm::EngineSettings;
using ::litert::lm::ModelAssets;
using ::litert::lm::ScopedFile;
using ::litert::lm::SessionConfig;
using ::litert::lm::proto::SamplerParameters;

LiteRtLmStatusCode litert_lm_input_data_create(
    LiteRtLmInputDataType type, const void* data, size_t size,
    LiteRtLmInputData** out_input_data) {
  LITERT_LM_C_RETURN_IF_NULL(out_input_data);
  *out_input_data = nullptr;
  if (data == nullptr && size > 0) {
    return litert::lm::c::ReturnError(
        absl::StatusCode::kInvalidArgument,
        "data must not be NULL when size is non-zero.");
  }
  std::unique_ptr<LiteRtLmInputData> input_data;
  switch (type) {
    case kLiteRtLmInputDataTypeText:
      input_data = std::make_unique<LiteRtLmInputData>(litert::lm::InputText(
          std::string(static_cast<const char*>(data), size)));
      break;
    case kLiteRtLmInputDataTypeImage:
      input_data = std::make_unique<LiteRtLmInputData>(litert::lm::InputImage(
          std::string(static_cast<const char*>(data), size)));
      break;
    case kLiteRtLmInputDataTypeImageEnd:
      input_data =
          std::make_unique<LiteRtLmInputData>(litert::lm::InputImageEnd());
      break;
    case kLiteRtLmInputDataTypeAudio:
      input_data = std::make_unique<LiteRtLmInputData>(litert::lm::InputAudio(
          std::string(static_cast<const char*>(data), size)));
      break;
    case kLiteRtLmInputDataTypeAudioEnd:
      input_data =
          std::make_unique<LiteRtLmInputData>(litert::lm::InputAudioEnd());
      break;
    default:
      return litert::lm::c::ReturnError(absl::StatusCode::kInvalidArgument,
                                        "Unknown LiteRtLmInputDataType.");
  }
  *out_input_data = input_data.release();
  return kLiteRtLmStatusOk;
}

void litert_lm_input_data_delete(LiteRtLmInputData* input_data) {
  delete input_data;
}

static absl::StatusOr<std::unique_ptr<LiteRtLmEngineSettings>>
CreateEngineSettingsHelper(ModelAssets model_assets,
                           absl::string_view backend_str,
                           absl::string_view vision_backend_str,
                           absl::string_view audio_backend_str) {
  auto backend = litert::lm::GetBackendFromString(backend_str);
  if (!backend.ok()) {
    ABSL_LOG(ERROR) << "Failed to parse backend: " << backend.status();
    return backend.status();
  }

  std::optional<litert::lm::Backend> vision_backend;
  if (!vision_backend_str.empty()) {
    auto backend = litert::lm::GetBackendFromString(vision_backend_str);
    if (!backend.ok()) {
      ABSL_LOG(ERROR) << "Failed to parse vision backend: " << backend.status();
      return backend.status();
    }
    vision_backend = *backend;
  }

  std::optional<litert::lm::Backend> audio_backend;
  if (!audio_backend_str.empty()) {
    auto backend = litert::lm::GetBackendFromString(audio_backend_str);
    if (!backend.ok()) {
      ABSL_LOG(ERROR) << "Failed to parse audio backend: " << backend.status();
      return backend.status();
    }
    audio_backend = *backend;
  }

  auto engine_settings = EngineSettings::CreateDefault(
      std::move(model_assets), *backend, vision_backend, audio_backend);
  if (!engine_settings.ok()) {
    ABSL_LOG(ERROR) << "Failed to create engine settings: "
                    << engine_settings.status();
    return engine_settings.status();
  }

  auto c_settings = std::make_unique<LiteRtLmEngineSettings>();
  c_settings->settings =
      std::make_unique<EngineSettings>(std::move(*engine_settings));
  return c_settings;
}

extern "C" {

LiteRtLmStatusCode litert_lm_set_min_log_level(LiteRtLmLogSeverity level) {
  if (!IsValidLogSeverity(level)) {
    return litert::lm::c::ReturnError(absl::StatusCode::kInvalidArgument,
                                      "Unknown LiteRtLmLogSeverity.");
  }
  litert::lm::SetMinLogSeverity(static_cast<litert::lm::LogSeverity>(level));
  return kLiteRtLmStatusOk;
}

SamplerParameters::Type ToSamplerParametersType(LiteRtLmSamplerType type) {
  switch (type) {
    case kLiteRtLmSamplerTypeUnspecified:
      return SamplerParameters::TYPE_UNSPECIFIED;
    case kLiteRtLmSamplerTypeTopK:
      return SamplerParameters::TOP_K;
    case kLiteRtLmSamplerTypeTopP:
      return SamplerParameters::TOP_P;
    case kLiteRtLmSamplerTypeGreedy:
      return SamplerParameters::GREEDY;
  }
  return SamplerParameters::TYPE_UNSPECIFIED;
}

LiteRtLmStatusCode litert_lm_sampler_params_create(
    LiteRtLmSamplerType type, LiteRtLmSamplerParams** out_params) {
  LITERT_LM_C_RETURN_IF_NULL(out_params);
  *out_params = nullptr;
  if (!IsValidSamplerType(type)) {
    return litert::lm::c::ReturnError(absl::StatusCode::kInvalidArgument,
                                      "Unknown LiteRtLmSamplerType.");
  }
  auto params = std::make_unique<LiteRtLmSamplerParams>();
  params->type = type;
  params->top_k = 0;
  params->top_p = 0.0f;
  params->temperature = 0.0f;
  params->seed = 0;
  *out_params = params.release();
  return kLiteRtLmStatusOk;
}

void litert_lm_sampler_params_delete(LiteRtLmSamplerParams* params) {
  delete params;
}

LiteRtLmStatusCode litert_lm_sampler_params_set_top_k(
    LiteRtLmSamplerParams* params, int32_t top_k) {
  LITERT_LM_C_RETURN_IF_NULL(params);
  params->top_k = top_k;
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_sampler_params_set_top_p(
    LiteRtLmSamplerParams* params, float top_p) {
  LITERT_LM_C_RETURN_IF_NULL(params);
  params->top_p = top_p;
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_sampler_params_set_temperature(
    LiteRtLmSamplerParams* params, float temperature) {
  LITERT_LM_C_RETURN_IF_NULL(params);
  params->temperature = temperature;
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_sampler_params_set_seed(
    LiteRtLmSamplerParams* params, int32_t seed) {
  LITERT_LM_C_RETURN_IF_NULL(params);
  params->seed = seed;
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_session_config_create(
    LiteRtLmSessionConfig** out_config) {
  LITERT_LM_C_RETURN_IF_NULL(out_config);
  *out_config = nullptr;
  auto c_config = std::make_unique<LiteRtLmSessionConfig>();
  c_config->config =
      std::make_unique<SessionConfig>(SessionConfig::CreateDefault());
  *out_config = c_config.release();
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_session_config_set_max_output_tokens(
    LiteRtLmSessionConfig* config, int max_output_tokens) {
  if (!IsValidSessionConfig(config)) {
    return kLiteRtLmStatusInvalidArgument;
  }
  config->config->SetMaxOutputTokens(max_output_tokens);
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_session_config_set_apply_prompt_template(
    LiteRtLmSessionConfig* config, bool apply_prompt_template) {
  if (!IsValidSessionConfig(config)) {
    return kLiteRtLmStatusInvalidArgument;
  }
  config->config->SetApplyPromptTemplateInSession(apply_prompt_template);
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_session_config_set_enable_speculative_decoding(
    LiteRtLmSessionConfig* config, bool enable_speculative_decoding) {
  if (!IsValidSessionConfig(config)) {
    return kLiteRtLmStatusInvalidArgument;
  }
  config->config->SetEnableSpeculativeDecoding(enable_speculative_decoding);
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_session_config_set_sampler_params(
    LiteRtLmSessionConfig* config,
    const LiteRtLmSamplerParams* sampler_params) {
  if (!IsValidSessionConfig(config)) {
    return kLiteRtLmStatusInvalidArgument;
  }
  LITERT_LM_C_RETURN_IF_NULL(sampler_params);
  if (!IsValidSamplerType(sampler_params->type)) {
    return litert::lm::c::ReturnError(absl::StatusCode::kInvalidArgument,
                                      "Unknown LiteRtLmSamplerType.");
  }
  SamplerParameters& params = config->config->GetMutableSamplerParams();

  params.set_type(ToSamplerParametersType(sampler_params->type));

  params.set_k(sampler_params->top_k);
  params.set_p(sampler_params->top_p);
  params.set_temperature(sampler_params->temperature);
  params.set_seed(sampler_params->seed);
  return kLiteRtLmStatusOk;
}

void litert_lm_session_config_delete(LiteRtLmSessionConfig* config) {
  delete config;
}

LiteRtLmStatusCode litert_lm_session_config_set_lora_path(
    LiteRtLmSessionConfig* config, const char* lora_path) {
  if (!config || !config->config || !lora_path) {
    return litert::lm::c::ReturnError(absl::StatusCode::kInvalidArgument,
                                      "Invalid session config or LoRA path.");
  }
  absl::string_view path_view(lora_path);
  if (path_view.empty()) {
    return litert::lm::c::ReturnError(absl::StatusCode::kInvalidArgument,
                                      "LoRA path is empty.");
  }
  auto lora_file = ScopedFile::Open(lora_path);
  if (!lora_file.ok()) {
    ABSL_LOG(ERROR) << "Failed to open LoRA file: " << lora_file.status();
    return litert::lm::c::ToCStatus(lora_file.status());
  }
  config->config->SetScopedLoraFile(
      std::make_shared<litert::lm::ScopedFile>(std::move(*lora_file)));
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_session_config_set_audio_lora_path(
    LiteRtLmSessionConfig* config, const char* audio_lora_path) {
  if (!config || !config->config || !audio_lora_path) {
    return litert::lm::c::ReturnError(
        absl::StatusCode::kInvalidArgument,
        "Invalid session config or Audio LoRA path.");
  }
  absl::string_view path_view(audio_lora_path);
  if (path_view.empty()) {
    return litert::lm::c::ReturnError(absl::StatusCode::kInvalidArgument,
                                      "Audio LoRA path is empty.");
  }
  auto lora_file = ScopedFile::Open(path_view);
  if (!lora_file.ok()) {
    ABSL_LOG(ERROR) << "Failed to open Audio LoRA file: " << lora_file.status();
    return litert::lm::c::ToCStatus(lora_file.status());
  }
  config->config->SetAudioScopedLoraFile(
      std::make_shared<litert::lm::ScopedFile>(std::move(*lora_file)));
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_repetition_penalty_config_create(
    LiteRtLmRepetitionPenaltyConfig** out_config) {
  LITERT_LM_C_RETURN_IF_NULL(out_config);
  *out_config = new LiteRtLmRepetitionPenaltyConfig{
      .repetition_penalty_config =
          litert::lm::RepetitionPenaltyConfig::Default(),
  };
  return kLiteRtLmStatusOk;
}

void litert_lm_repetition_penalty_config_delete(
    LiteRtLmRepetitionPenaltyConfig* config) {
  delete config;
}

LiteRtLmStatusCode litert_lm_repetition_penalty_config_set_repetition_penalty(
    LiteRtLmRepetitionPenaltyConfig* config, float repetition_penalty) {
  LITERT_LM_C_RETURN_IF_NULL(config);
  config->repetition_penalty_config = litert::lm::RepetitionPenaltyConfig(
      repetition_penalty, config->repetition_penalty_config.presence_penalty(),
      config->repetition_penalty_config.frequency_penalty(),
      config->repetition_penalty_config.window_size());
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_repetition_penalty_config_set_presence_penalty(
    LiteRtLmRepetitionPenaltyConfig* config, float presence_penalty) {
  LITERT_LM_C_RETURN_IF_NULL(config);
  config->repetition_penalty_config = litert::lm::RepetitionPenaltyConfig(
      config->repetition_penalty_config.repetition_penalty(), presence_penalty,
      config->repetition_penalty_config.frequency_penalty(),
      config->repetition_penalty_config.window_size());
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_repetition_penalty_config_set_frequency_penalty(
    LiteRtLmRepetitionPenaltyConfig* config, float frequency_penalty) {
  LITERT_LM_C_RETURN_IF_NULL(config);
  config->repetition_penalty_config = litert::lm::RepetitionPenaltyConfig(
      config->repetition_penalty_config.repetition_penalty(),
      config->repetition_penalty_config.presence_penalty(), frequency_penalty,
      config->repetition_penalty_config.window_size());
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_repetition_penalty_config_set_window_size(
    LiteRtLmRepetitionPenaltyConfig* config, int window_size) {
  LITERT_LM_C_RETURN_IF_NULL(config);
  config->repetition_penalty_config = litert::lm::RepetitionPenaltyConfig(
      config->repetition_penalty_config.repetition_penalty(),
      config->repetition_penalty_config.presence_penalty(),
      config->repetition_penalty_config.frequency_penalty(), window_size);
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_no_repeat_ngram_config_create(
    LiteRtLmNoRepeatNgramConfig** out_config) {
  LITERT_LM_C_RETURN_IF_NULL(out_config);
  *out_config = new LiteRtLmNoRepeatNgramConfig{
      .no_repeat_ngram_config = litert::lm::NoRepeatNgramConfig::Default(),
  };
  return kLiteRtLmStatusOk;
}

void litert_lm_no_repeat_ngram_config_delete(
    LiteRtLmNoRepeatNgramConfig* config) {
  delete config;
}

LiteRtLmStatusCode litert_lm_no_repeat_ngram_config_set_no_repeat_ngram_size(
    LiteRtLmNoRepeatNgramConfig* config, int no_repeat_ngram_size) {
  LITERT_LM_C_RETURN_IF_NULL(config);
  config->no_repeat_ngram_config = litert::lm::NoRepeatNgramConfig(
      no_repeat_ngram_size, config->no_repeat_ngram_config.window_size());
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_no_repeat_ngram_config_set_window_size(
    LiteRtLmNoRepeatNgramConfig* config, int window_size) {
  LITERT_LM_C_RETURN_IF_NULL(config);
  config->no_repeat_ngram_config = litert::lm::NoRepeatNgramConfig(
      config->no_repeat_ngram_config.no_repeat_ngram_size(), window_size);
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_suppress_tokens_config_create(
    LiteRtLmSuppressTokensConfig** out_config) {
  LITERT_LM_C_RETURN_IF_NULL(out_config);
  *out_config = new LiteRtLmSuppressTokensConfig{
      .suppress_tokens_config = litert::lm::SuppressTokensConfig::Default(),
  };
  return kLiteRtLmStatusOk;
}

void litert_lm_suppress_tokens_config_delete(
    LiteRtLmSuppressTokensConfig* config) {
  delete config;
}

LiteRtLmStatusCode litert_lm_suppress_tokens_config_set_suppress_tokens(
    LiteRtLmSuppressTokensConfig* config, const int* suppress_tokens,
    size_t num_tokens) {
  LITERT_LM_C_RETURN_IF_NULL(config);
  if (num_tokens == 0) {
    config->suppress_tokens_config =
        litert::lm::SuppressTokensConfig::Default();
    return kLiteRtLmStatusOk;
  }

  if (suppress_tokens == nullptr) {
    ABSL_LOG(ERROR) << "Suppress tokens are null but num_tokens is not 0.";
    return litert::lm::c::ReturnError(
        absl::StatusCode::kInvalidArgument,
        "suppress_tokens must not be NULL when num_tokens is non-zero.");
  }

  config->suppress_tokens_config = litert::lm::SuppressTokensConfig(
      absl::flat_hash_set<int>(suppress_tokens, suppress_tokens + num_tokens));
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_engine_settings_create(
    const char* model_path, const char* backend_str,
    const char* vision_backend_str, const char* audio_backend_str,
    LiteRtLmEngineSettings** out_settings) {
  LITERT_LM_C_RETURN_IF_NULL(out_settings);
  *out_settings = nullptr;
  LITERT_LM_C_RETURN_IF_NULL(model_path);
  auto model_assets = ModelAssets::Create(model_path);
  if (!model_assets.ok()) {
    ABSL_LOG(ERROR) << "Failed to create model assets: "
                    << model_assets.status();
    return litert::lm::c::ToCStatus(model_assets.status());
  }
  LITERT_LM_C_ASSIGN_OR_RETURN(
      std::unique_ptr<LiteRtLmEngineSettings> settings,
      CreateEngineSettingsHelper(std::move(*model_assets),
                                 absl::NullSafeStringView(backend_str),
                                 absl::NullSafeStringView(vision_backend_str),
                                 absl::NullSafeStringView(audio_backend_str)));
  *out_settings = settings.release();
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_engine_settings_create_from_raw_file_descriptor(
    int fd, const char* backend_str, const char* vision_backend_str,
    const char* audio_backend_str, LiteRtLmEngineSettings** out_settings) {
  LITERT_LM_C_RETURN_IF_NULL(out_settings);
  *out_settings = nullptr;
  if (fd < 0) {
    ABSL_LOG(ERROR) << "Invalid file descriptor: " << fd;
    return litert::lm::c::ReturnError(absl::StatusCode::kInvalidArgument,
                                      "Invalid file descriptor.");
  }
  auto model_assets = ModelAssets::Create(
#if defined(_WIN32)
      std::make_shared<litert::lm::ScopedFile>(litert::lm::ScopedFile(
          reinterpret_cast<litert::lm::ScopedFile::PlatformFile>(
              _get_osfhandle(fd)))));
#else
      std::make_shared<litert::lm::ScopedFile>(litert::lm::ScopedFile(fd)));
#endif
  if (!model_assets.ok()) {
    ABSL_LOG(ERROR) << "Failed to create model assets from raw FD: "
                    << model_assets.status();
    return litert::lm::c::ToCStatus(model_assets.status());
  }
  ABSL_VLOG(1) << "LiteRT-LM successfully created EngineSettings directly "
                  "from raw File Descriptor: "
               << fd;
  LITERT_LM_C_ASSIGN_OR_RETURN(
      std::unique_ptr<LiteRtLmEngineSettings> settings,
      CreateEngineSettingsHelper(std::move(*model_assets),
                                 absl::NullSafeStringView(backend_str),
                                 absl::NullSafeStringView(vision_backend_str),
                                 absl::NullSafeStringView(audio_backend_str)));
  *out_settings = settings.release();
  return kLiteRtLmStatusOk;
}

void litert_lm_engine_settings_delete(LiteRtLmEngineSettings* settings) {
  delete settings;
}

LiteRtLmStatusCode litert_lm_engine_settings_set_max_num_tokens(
    LiteRtLmEngineSettings* settings, int max_num_tokens) {
  if (!IsValidEngineSettings(settings)) {
    return kLiteRtLmStatusInvalidArgument;
  }
  settings->settings->GetMutableMainExecutorSettings().SetMaxNumTokens(
      max_num_tokens);
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_engine_settings_set_num_threads(
    LiteRtLmEngineSettings* settings, int num_threads) {
  if (!IsValidEngineSettings(settings)) {
    return kLiteRtLmStatusInvalidArgument;
  }
  auto& main_settings = settings->settings->GetMutableMainExecutorSettings();
  auto config = main_settings.MutableBackendConfig<litert::lm::CpuConfig>();
  if (config.ok()) {
    litert::lm::CpuConfig cpu_config = *config;
    cpu_config.number_of_threads = num_threads;
    main_settings.SetBackendConfig(cpu_config);
  } else {
    ABSL_LOG(WARNING) << "Failed to get CpuConfig to set num threads: "
                      << config.status();
  }
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_engine_settings_set_audio_num_threads(
    LiteRtLmEngineSettings* settings, int num_threads) {
  if (!IsValidEngineSettings(settings)) {
    return kLiteRtLmStatusInvalidArgument;
  }
  auto& audio_settings = settings->settings->GetMutableAudioExecutorSettings();
  if (audio_settings.has_value()) {
    audio_settings->SetNumThreads(num_threads);
  }
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_engine_settings_set_parallel_file_section_loading(
    LiteRtLmEngineSettings* settings, bool parallel_file_section_loading) {
  if (!IsValidEngineSettings(settings)) {
    return kLiteRtLmStatusInvalidArgument;
  }
  settings->settings->SetParallelFileSectionLoading(
      parallel_file_section_loading);
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_engine_settings_set_single_threaded_execution(
    LiteRtLmEngineSettings* settings, bool single_threaded_execution) {
  if (!IsValidEngineSettings(settings)) {
    return kLiteRtLmStatusInvalidArgument;
  }
  settings->settings->SetSingleThreadedExecution(single_threaded_execution);
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_engine_settings_set_max_num_images(
    LiteRtLmEngineSettings* settings, int max_num_images) {
  if (!IsValidEngineSettings(settings)) {
    return kLiteRtLmStatusInvalidArgument;
  }
  settings->settings->GetMutableMainExecutorSettings().SetMaxNumImages(
      max_num_images);
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_engine_settings_set_max_vision_tokens_per_image(
    LiteRtLmEngineSettings* settings, int max_vision_tokens_per_image) {
  if (!IsValidEngineSettings(settings)) {
    return kLiteRtLmStatusInvalidArgument;
  }
  settings->settings->SetMaxVisionTokensPerImage(max_vision_tokens_per_image);
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_engine_settings_set_cache_dir(
    LiteRtLmEngineSettings* settings, const char* cache_dir) {
  if (!IsValidEngineSettings(settings)) {
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

LiteRtLmStatusCode litert_lm_engine_settings_set_litert_dispatch_lib_dir(
    LiteRtLmEngineSettings* settings, const char* lib_dir) {
  if (!IsValidEngineSettings(settings)) {
    return kLiteRtLmStatusInvalidArgument;
  }
  LITERT_LM_C_RETURN_IF_NULL(lib_dir);
  settings->settings->GetMutableMainExecutorSettings().SetLitertDispatchLibDir(
      lib_dir);
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_engine_settings_enable_benchmark(
    LiteRtLmEngineSettings* settings) {
  if (!IsValidEngineSettings(settings)) {
    return kLiteRtLmStatusInvalidArgument;
  }
  settings->settings->GetMutableBenchmarkParams();
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_engine_settings_set_num_prefill_tokens(
    LiteRtLmEngineSettings* settings, int num_prefill_tokens) {
  if (!IsValidEngineSettings(settings)) {
    return kLiteRtLmStatusInvalidArgument;
  }
  settings->settings->GetMutableBenchmarkParams().set_num_prefill_tokens(
      num_prefill_tokens);
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_engine_settings_set_num_decode_tokens(
    LiteRtLmEngineSettings* settings, int num_decode_tokens) {
  if (!IsValidEngineSettings(settings)) {
    return kLiteRtLmStatusInvalidArgument;
  }
  settings->settings->GetMutableBenchmarkParams().set_num_decode_tokens(
      num_decode_tokens);
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_engine_settings_set_enable_speculative_decoding(
    LiteRtLmEngineSettings* settings, bool enable_speculative_decoding) {
  if (!IsValidEngineSettings(settings)) {
    return kLiteRtLmStatusInvalidArgument;
  }
  auto& main_settings = settings->settings->GetMutableMainExecutorSettings();
  auto advanced_settings = main_settings.GetAdvancedSettings().value_or(
      litert::lm::AdvancedSettings());
  advanced_settings.enable_speculative_decoding = enable_speculative_decoding;
  main_settings.SetAdvancedSettings(advanced_settings);
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_engine_settings_set_gpu_decode_steps_per_sync(
    LiteRtLmEngineSettings* settings, int num_decode_steps_per_sync) {
  if (!IsValidEngineSettings(settings)) {
    return kLiteRtLmStatusInvalidArgument;
  }
  // Note: This setting is currently only supported for the Artisan GPU
  // backend.
  auto backend_config =
      settings->settings->GetMutableMainExecutorSettings()
          .MutableBackendConfig<litert::lm::GpuArtisanConfig>();
  if (backend_config.ok()) {
    auto config = backend_config.value();
    config.num_decode_steps_per_sync = num_decode_steps_per_sync;
    settings->settings->GetMutableMainExecutorSettings().SetBackendConfig(
        config);
  }
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_engine_settings_set_gpu_wait_for_weight_uploads(
    LiteRtLmEngineSettings* settings, bool wait_for_weight_uploads) {
  if (!IsValidEngineSettings(settings)) {
    return kLiteRtLmStatusInvalidArgument;
  }
  // Note: This setting is currently only supported for the Artisan GPU
  // backend.
  auto backend_config =
      settings->settings->GetMutableMainExecutorSettings()
          .MutableBackendConfig<litert::lm::GpuArtisanConfig>();
  if (backend_config.ok()) {
    auto config = backend_config.value();
    config.wait_for_weight_uploads = wait_for_weight_uploads;
    settings->settings->GetMutableMainExecutorSettings().SetBackendConfig(
        config);
  }
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode
litert_lm_engine_settings_set_use_ringbuffers_local_attention(
    LiteRtLmEngineSettings* settings, bool use_ringbuffers_local_attention) {
  if (!IsValidEngineSettings(settings)) {
    return kLiteRtLmStatusInvalidArgument;
  }
  auto& main_settings = settings->settings->GetMutableMainExecutorSettings();
  auto config =
      main_settings.MutableBackendConfig<litert::lm::GpuArtisanConfig>();
  if (config.ok()) {
    litert::lm::GpuArtisanConfig gpu_artisan_config = *config;
    gpu_artisan_config.use_autosized_ringbuffers =
        use_ringbuffers_local_attention;
    main_settings.SetBackendConfig(gpu_artisan_config);
  } else {
    ABSL_LOG(INFO) << "Failed to get GpuArtisanConfig to set "
                      "use_ringbuffers_local_attention: "
                   << config.status();
  }
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_engine_settings_set_lora_rank(
    LiteRtLmEngineSettings* settings, int lora_rank) {
  if (!IsValidEngineSettings(settings)) {
    return kLiteRtLmStatusInvalidArgument;
  }
  settings->settings->GetMutableMainExecutorSettings().SetLoraRank(lora_rank);
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_engine_settings_set_supported_lora_ranks(
    LiteRtLmEngineSettings* settings, const int* lora_ranks, size_t num_ranks) {
  if (!settings || !settings->settings || !lora_ranks || num_ranks == 0) {
    return litert::lm::c::ReturnError(absl::StatusCode::kInvalidArgument,
                                      "Invalid engine settings or LoRA ranks.");
  }
  std::vector<uint32_t> ranks;
  ranks.reserve(num_ranks);
  for (size_t i = 0; i < num_ranks; ++i) {
    ranks.push_back(static_cast<uint32_t>(lora_ranks[i]));
  }
  return litert::lm::c::ToCStatus(
      settings->settings->GetMutableMainExecutorSettings()
          .SetSupportedLoraRanks(ranks));
}

LiteRtLmStatusCode litert_lm_engine_settings_set_audio_lora_rank(
    LiteRtLmEngineSettings* settings, int lora_rank) {
  if (!IsValidEngineSettings(settings)) {
    return kLiteRtLmStatusInvalidArgument;
  }
  // No-op if no audio executor is configured.
  if (settings->settings->GetAudioExecutorSettings().has_value()) {
    settings->settings->GetMutableAudioExecutorSettings()->SetLoraRank(
        lora_rank);
  }
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_engine_settings_set_supported_audio_lora_ranks(
    LiteRtLmEngineSettings* settings, const int* lora_ranks, size_t num_ranks) {
  if (!settings || !settings->settings || !lora_ranks || num_ranks == 0) {
    return litert::lm::c::ReturnError(
        absl::StatusCode::kInvalidArgument,
        "Invalid engine settings or Audio LoRA ranks.");
  }
  if (!settings->settings->GetAudioExecutorSettings().has_value()) {
    return litert::lm::c::ReturnError(
        absl::StatusCode::kFailedPrecondition,
        "Audio executor settings not configured in engine settings.");
  }
  std::vector<uint32_t> ranks;
  ranks.reserve(num_ranks);
  for (size_t i = 0; i < num_ranks; ++i) {
    ranks.push_back(static_cast<uint32_t>(lora_ranks[i]));
  }
  return litert::lm::c::ToCStatus(
      settings->settings->GetMutableAudioExecutorSettings()
          ->SetSupportedLoraRanks(ranks));
}

LiteRtLmStatusCode litert_lm_engine_settings_set_activation_data_type(
    LiteRtLmEngineSettings* settings,
    LiteRtLmActivationDataType activation_data_type) {
  if (!IsValidEngineSettings(settings)) {
    return kLiteRtLmStatusInvalidArgument;
  }
  if (!IsValidActivationDataType(activation_data_type)) {
    return litert::lm::c::ReturnError(absl::StatusCode::kInvalidArgument,
                                      "Unknown LiteRtLmActivationDataType.");
  }
  settings->settings->GetMutableMainExecutorSettings().SetActivationDataType(
      static_cast<litert::lm::ActivationDataType>(activation_data_type));
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_engine_settings_set_prefill_chunk_size(
    LiteRtLmEngineSettings* settings, int prefill_chunk_size) {
  if (!IsValidEngineSettings(settings)) {
    return kLiteRtLmStatusInvalidArgument;
  }
  auto& main_settings = settings->settings->GetMutableMainExecutorSettings();
  auto config = main_settings.MutableBackendConfig<litert::lm::CpuConfig>();
  if (!config.ok()) {
    ABSL_LOG(WARNING) << "Failed to get CpuConfig to set prefill chunk size: "
                      << config.status();
    return kLiteRtLmStatusOk;
  }
  config->prefill_chunk_size = prefill_chunk_size;
  main_settings.SetBackendConfig(*config);
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_engine_settings_set_enable_ynnpack(
    LiteRtLmEngineSettings* settings, bool enable_ynnpack) {
  if (!IsValidEngineSettings(settings)) {
    return kLiteRtLmStatusInvalidArgument;
  }
  auto& main_settings = settings->settings->GetMutableMainExecutorSettings();
  auto config = main_settings.MutableBackendConfig<litert::lm::CpuConfig>();
  if (!config.ok()) {
    ABSL_LOG(WARNING) << "Failed to get CpuConfig to set enable ynnpack: "
                      << config.status();
    return kLiteRtLmStatusOk;
  }
  config->enable_ynnpack = enable_ynnpack;
  main_settings.SetBackendConfig(*config);
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_engine_settings_set_gpu_enable_metal_residency_set(
    LiteRtLmEngineSettings* settings, bool enable_metal_residency_set) {
  if (!IsValidEngineSettings(settings)) {
    return kLiteRtLmStatusInvalidArgument;
  }
  auto advanced_settings = settings->settings->GetMainExecutorSettings()
                               .GetAdvancedSettings()
                               .value_or(litert::lm::AdvancedSettings());
  advanced_settings.gpu_enable_metal_residency_set = enable_metal_residency_set;
  settings->settings->GetMutableMainExecutorSettings().SetAdvancedSettings(
      advanced_settings);
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_engine_create(
    const LiteRtLmEngineSettings* settings, LiteRtLmEngine** out_engine) {
  LITERT_LM_C_RETURN_IF_NULL(out_engine);
  *out_engine = nullptr;
  if (!settings || !settings->settings) {
    return litert::lm::c::ReturnError(absl::StatusCode::kInvalidArgument,
                                      "Invalid engine settings.");
  }

  absl::StatusOr<std::unique_ptr<Engine>> engine =
      EngineFactory::CreateDefault(*settings->settings);

  if (!engine.ok()) {
    ABSL_LOG(ERROR) << "Failed to create engine: " << engine.status();
    return litert::lm::c::ToCStatus(engine.status());
  }

  auto c_engine = std::make_unique<LiteRtLmEngine>();
  c_engine->engine = *std::move(engine);
  *out_engine = c_engine.release();
  return kLiteRtLmStatusOk;
}

void litert_lm_engine_delete(LiteRtLmEngine* engine) { delete engine; }

LiteRtLmStatusCode litert_lm_engine_create_session(
    LiteRtLmEngine* engine, LiteRtLmSessionConfig* config,
    LiteRtLmSession** out_session) {
  LITERT_LM_C_RETURN_IF_NULL(out_session);
  *out_session = nullptr;
  if (!engine || !engine->engine) {
    return litert::lm::c::ReturnError(absl::StatusCode::kInvalidArgument,
                                      "Invalid engine.");
  }

  SessionConfig session_config = config && config->config
                                     ? *config->config
                                     : SessionConfig::CreateDefault();
  if (engine->engine->GetEngineSettings()
          .GetAudioExecutorSettings()
          .has_value()) {
    session_config.SetAudioModalityEnabled(true);
  }
  if (engine->engine->GetEngineSettings()
          .GetVisionExecutorSettings()
          .has_value()) {
    session_config.SetVisionModalityEnabled(true);
  }

  absl::StatusOr<std::unique_ptr<Engine::Session>> session =
      engine->engine->CreateSession(session_config);
  if (!session.ok()) {
    ABSL_LOG(ERROR) << "Failed to create session: " << session.status();
    return litert::lm::c::ToCStatus(session.status());
  }

  auto c_session = std::make_unique<LiteRtLmSession>();
  c_session->session = *std::move(session);
  *out_session = c_session.release();
  return kLiteRtLmStatusOk;
}

void litert_lm_session_delete(LiteRtLmSession* session) { delete session; }

LiteRtLmStatusCode litert_lm_session_cancel_process(LiteRtLmSession* session) {
  if (!session || !session->session) {
    return litert::lm::c::ReturnError(absl::StatusCode::kInvalidArgument,
                                      "Invalid session.");
  }
  session->session->CancelProcess();
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_session_save_checkpoint(LiteRtLmSession* session,
                                                     const char* label) {
  if (!session || !session->session || !label) {
    return litert::lm::c::ReturnError(absl::StatusCode::kInvalidArgument,
                                      "Invalid session or checkpoint label.");
  }
  auto status = session->session->SaveCheckpoint(label);
  if (!status.ok()) {
    ABSL_LOG(ERROR) << "Failed to save checkpoint " << label << ": " << status;
    return litert::lm::c::ToCStatus(status);
  }
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_session_rewind_to_checkpoint(
    LiteRtLmSession* session, const char* label) {
  if (!session || !session->session || !label) {
    return litert::lm::c::ReturnError(absl::StatusCode::kInvalidArgument,
                                      "Invalid session or checkpoint label.");
  }
  auto status = session->session->RewindToCheckpoint(label);
  if (!status.ok()) {
    ABSL_LOG(ERROR) << "Failed to rewind to checkpoint " << label << ": "
                    << status;
    return litert::lm::c::ToCStatus(status);
  }
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_session_rewind_to_step(LiteRtLmSession* session,
                                                    int step) {
  if (!session || !session->session) {
    return litert::lm::c::ReturnError(absl::StatusCode::kInvalidArgument,
                                      "Invalid session.");
  }
  auto status = session->session->RewindToStep(step);
  if (!status.ok()) {
    ABSL_LOG(ERROR) << "Failed to rewind to step " << step << ": " << status;
    return litert::lm::c::ToCStatus(status);
  }
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_session_run_text_scoring(
    LiteRtLmSession* session, const char** target_text, size_t num_targets,
    bool store_token_lengths, LiteRtLmResponses** out_responses) {
  LITERT_LM_C_RETURN_IF_NULL(out_responses);
  *out_responses = nullptr;
  if (!session || !session->session || !target_text || num_targets <= 0) {
    return litert::lm::c::ReturnError(absl::StatusCode::kInvalidArgument,
                                      "Invalid session or target texts.");
  }
  std::vector<absl::string_view> target_text_views;
  target_text_views.reserve(num_targets);
  for (size_t i = 0; i < num_targets; ++i) {
    if (target_text[i] == nullptr) {
      return litert::lm::c::ReturnError(
          absl::StatusCode::kInvalidArgument,
          "target_text elements must not be NULL.");
    }
    target_text_views.push_back(target_text[i]);
  }
  auto responses =
      session->session->RunTextScoring(target_text_views, store_token_lengths);
  if (!responses.ok()) {
    ABSL_LOG(ERROR) << "Failed to run text scoring: " << responses.status();
    return litert::lm::c::ToCStatus(responses.status());
  }
  auto c_responses = std::make_unique<LiteRtLmResponses>(std::move(*responses));
  if (c_responses->responses.GetTexts().empty()) {
    auto& mutable_texts = c_responses->responses.GetMutableTexts();
    mutable_texts.reserve(num_targets);
    for (size_t i = 0; i < num_targets; ++i) {
      mutable_texts.emplace_back(target_text[i]);
    }
  }
  *out_responses = c_responses.release();
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_session_run_prefill(
    LiteRtLmSession* session, const LiteRtLmInputData* const* inputs,
    size_t num_inputs) {
  if (!session || !session->session || !inputs || num_inputs <= 0) {
    return litert::lm::c::ReturnError(absl::StatusCode::kInvalidArgument,
                                      "Invalid session or inputs.");
  }
  auto engine_inputs = ToEngineInputData(inputs, num_inputs);
  if (!engine_inputs.ok()) {
    ABSL_LOG(ERROR) << "Failed to copy inputs: " << engine_inputs.status();
    return litert::lm::c::ToCStatus(engine_inputs.status());
  }
  auto status = session->session->RunPrefill(*engine_inputs);
  if (!status.ok()) {
    ABSL_LOG(ERROR) << "Failed to run prefill: " << status;
    return litert::lm::c::ToCStatus(status);
  }
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_session_run_decode(
    LiteRtLmSession* session, LiteRtLmResponses** out_responses) {
  LITERT_LM_C_RETURN_IF_NULL(out_responses);
  *out_responses = nullptr;
  if (!session || !session->session) {
    return litert::lm::c::ReturnError(absl::StatusCode::kInvalidArgument,
                                      "Invalid session.");
  }
  auto responses = session->session->RunDecode();
  if (!responses.ok()) {
    ABSL_LOG(ERROR) << "Failed to run decode: " << responses.status();
    return litert::lm::c::ToCStatus(responses.status());
  }
  *out_responses = new LiteRtLmResponses{std::move(*responses)};
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_session_run_decode_async(
    LiteRtLmSession* session, LiteRtLmStreamCallback callback,
    void* callback_data) {
  if (!session || !session->session) {
    return litert::lm::c::ReturnError(absl::StatusCode::kInvalidArgument,
                                      "Invalid session.");
  }
  LITERT_LM_C_RETURN_IF_NULL(callback);
  auto status =
      session->session->RunDecodeAsync(CreateCallback(callback, callback_data));
  if (!status.ok()) {
    ABSL_LOG(ERROR) << "Failed to start decode stream: " << status.status();
    return litert::lm::c::ToCStatus(status.status());
  }
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_session_generate_content(
    LiteRtLmSession* session, const LiteRtLmInputData* const* inputs,
    size_t num_inputs, LiteRtLmResponses** out_responses) {
  LITERT_LM_C_RETURN_IF_NULL(out_responses);
  *out_responses = nullptr;
  if (!session || !session->session) {
    return litert::lm::c::ReturnError(absl::StatusCode::kInvalidArgument,
                                      "Invalid session.");
  }
  auto engine_inputs = ToEngineInputData(inputs, num_inputs);
  if (!engine_inputs.ok()) {
    ABSL_LOG(ERROR) << "Failed to copy inputs: " << engine_inputs.status();
    return litert::lm::c::ToCStatus(engine_inputs.status());
  }
  auto responses = session->session->GenerateContent(std::move(*engine_inputs));
  if (!responses.ok()) {
    ABSL_LOG(ERROR) << "Failed to generate content: " << responses.status();
    return litert::lm::c::ToCStatus(responses.status());
  }

  *out_responses = new LiteRtLmResponses{std::move(*responses)};
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_session_generate_content_stream(
    LiteRtLmSession* session, const LiteRtLmInputData* const* inputs,
    size_t num_inputs, LiteRtLmStreamCallback callback, void* callback_data) {
  if (!session || !session->session) {
    return litert::lm::c::ReturnError(absl::StatusCode::kInvalidArgument,
                                      "Invalid session.");
  }
  LITERT_LM_C_RETURN_IF_NULL(callback);
  auto engine_inputs = ToEngineInputData(inputs, num_inputs);
  if (!engine_inputs.ok()) {
    ABSL_LOG(ERROR) << "Failed to copy inputs: " << engine_inputs.status();
    return litert::lm::c::ToCStatus(engine_inputs.status());
  }

  absl::Status status = session->session->GenerateContentStream(
      std::move(*engine_inputs), CreateCallback(callback, callback_data));

  if (!status.ok()) {
    ABSL_LOG(ERROR) << "Failed to start content stream: " << status;
    // No need to delete callbacks, unique_ptr handles it if not moved.
    return litert::lm::c::ToCStatus(status);
  }
  // The call is non-blocking and returns immediately.
  return kLiteRtLmStatusOk;
}

void litert_lm_responses_delete(LiteRtLmResponses* responses) {
  delete responses;
}

LiteRtLmStatusCode litert_lm_responses_get_num_candidates(
    const LiteRtLmResponses* responses, int* out_num_candidates) {
  LITERT_LM_C_RETURN_IF_NULL(out_num_candidates);
  LITERT_LM_C_RETURN_IF_NULL(responses);
  *out_num_candidates = static_cast<int>(NumCandidates(responses->responses));
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_responses_get_response_text_at(
    const LiteRtLmResponses* responses, int index, const char** out_text) {
  LITERT_LM_C_RETURN_IF_NULL(out_text);
  *out_text = nullptr;
  LITERT_LM_C_RETURN_IF_NULL(responses);
  LITERT_LM_C_RETURN_IF_ERROR(CheckCandidateIndex(responses->responses, index));
  const auto& texts = responses->responses.GetTexts();
  if (!HasValueAt(texts, index)) {
    return ReturnNotFoundAt("response text", index);
  }
  // The string's data is valid as long as the responses object is alive.
  *out_text = texts[index].data();
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_responses_has_score_at(
    const LiteRtLmResponses* responses, int index, bool* out_has_score) {
  LITERT_LM_C_RETURN_IF_NULL(out_has_score);
  LITERT_LM_C_RETURN_IF_NULL(responses);
  LITERT_LM_C_RETURN_IF_ERROR(CheckCandidateIndex(responses->responses, index));
  *out_has_score = HasValueAt(responses->responses.GetScores(), index);
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_responses_get_score_at(
    const LiteRtLmResponses* responses, int index, float* out_score) {
  LITERT_LM_C_RETURN_IF_NULL(out_score);
  LITERT_LM_C_RETURN_IF_NULL(responses);
  LITERT_LM_C_RETURN_IF_ERROR(CheckCandidateIndex(responses->responses, index));
  const auto& scores = responses->responses.GetScores();
  if (!HasValueAt(scores, index)) {
    return ReturnNotFoundAt("score", index);
  }
  *out_score = scores[index];
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_responses_has_token_length_at(
    const LiteRtLmResponses* responses, int index, bool* out_has_token_length) {
  LITERT_LM_C_RETURN_IF_NULL(out_has_token_length);
  LITERT_LM_C_RETURN_IF_NULL(responses);
  LITERT_LM_C_RETURN_IF_ERROR(CheckCandidateIndex(responses->responses, index));
  const auto& token_lengths = responses->responses.GetTokenLengths();
  *out_has_token_length =
      token_lengths.has_value() && HasValueAt(*token_lengths, index);
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_responses_get_token_length_at(
    const LiteRtLmResponses* responses, int index, int* out_token_length) {
  LITERT_LM_C_RETURN_IF_NULL(out_token_length);
  LITERT_LM_C_RETURN_IF_NULL(responses);
  LITERT_LM_C_RETURN_IF_ERROR(CheckCandidateIndex(responses->responses, index));
  const auto& token_lengths = responses->responses.GetTokenLengths();
  if (!token_lengths.has_value() || !HasValueAt(*token_lengths, index)) {
    return ReturnNotFoundAt("token length", index);
  }
  *out_token_length = static_cast<int>((*token_lengths)[index]);
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_responses_has_token_scores_at(
    const LiteRtLmResponses* responses, int index, bool* out_has_token_scores) {
  LITERT_LM_C_RETURN_IF_NULL(out_has_token_scores);
  LITERT_LM_C_RETURN_IF_NULL(responses);
  LITERT_LM_C_RETURN_IF_ERROR(CheckCandidateIndex(responses->responses, index));
  const auto& token_scores = responses->responses.GetTokenScores();
  *out_has_token_scores =
      token_scores.has_value() && HasValueAt(*token_scores, index);
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_responses_get_num_token_scores_at(
    const LiteRtLmResponses* responses, int index, int* out_num_token_scores) {
  LITERT_LM_C_RETURN_IF_NULL(out_num_token_scores);
  LITERT_LM_C_RETURN_IF_NULL(responses);
  LITERT_LM_C_RETURN_IF_ERROR(CheckCandidateIndex(responses->responses, index));
  const auto& token_scores = responses->responses.GetTokenScores();
  if (!token_scores.has_value() || !HasValueAt(*token_scores, index)) {
    return ReturnNotFoundAt("token scores", index);
  }
  *out_num_token_scores = static_cast<int>((*token_scores)[index].size());
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_responses_get_token_scores_at(
    const LiteRtLmResponses* responses, int index,
    const float** out_token_scores) {
  LITERT_LM_C_RETURN_IF_NULL(out_token_scores);
  *out_token_scores = nullptr;
  LITERT_LM_C_RETURN_IF_NULL(responses);
  LITERT_LM_C_RETURN_IF_ERROR(CheckCandidateIndex(responses->responses, index));
  const auto& token_scores = responses->responses.GetTokenScores();
  if (!token_scores.has_value() || !HasValueAt(*token_scores, index)) {
    return ReturnNotFoundAt("token scores", index);
  }
  *out_token_scores = (*token_scores)[index].data();
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_session_get_benchmark_info(
    LiteRtLmSession* session, LiteRtLmBenchmarkInfo** out_benchmark_info) {
  LITERT_LM_C_RETURN_IF_NULL(out_benchmark_info);
  *out_benchmark_info = nullptr;
  if (!session || !session->session) {
    return litert::lm::c::ReturnError(absl::StatusCode::kInvalidArgument,
                                      "Invalid session.");
  }
  auto benchmark_info = session->session->GetBenchmarkInfo();
  if (!benchmark_info.ok()) {
    ABSL_LOG(ERROR) << "Failed to get benchmark info: "
                    << benchmark_info.status();
    return litert::lm::c::ToCStatus(benchmark_info.status());
  }
  *out_benchmark_info = new LiteRtLmBenchmarkInfo{std::move(*benchmark_info)};
  return kLiteRtLmStatusOk;
}

void litert_lm_benchmark_info_delete(LiteRtLmBenchmarkInfo* benchmark_info) {
  delete benchmark_info;
}

LiteRtLmStatusCode litert_lm_benchmark_info_get_time_to_first_token(
    const LiteRtLmBenchmarkInfo* benchmark_info, double* out_seconds) {
  LITERT_LM_C_RETURN_IF_NULL(out_seconds);
  LITERT_LM_C_RETURN_IF_NULL(benchmark_info);
  *out_seconds = benchmark_info->benchmark_info.GetTimeToFirstToken();
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_benchmark_info_get_total_init_time_in_second(
    const LiteRtLmBenchmarkInfo* benchmark_info, double* out_seconds) {
  LITERT_LM_C_RETURN_IF_NULL(out_seconds);
  LITERT_LM_C_RETURN_IF_NULL(benchmark_info);
  double total_init_time_ms = 0.0;
  for (const auto& phase : benchmark_info->benchmark_info.GetInitPhases()) {
    total_init_time_ms += absl::ToDoubleMilliseconds(phase.second);
  }
  *out_seconds = total_init_time_ms / 1000.0;
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_benchmark_info_get_num_prefill_turns(
    const LiteRtLmBenchmarkInfo* benchmark_info, int* out_num_turns) {
  LITERT_LM_C_RETURN_IF_NULL(out_num_turns);
  LITERT_LM_C_RETURN_IF_NULL(benchmark_info);
  *out_num_turns =
      static_cast<int>(benchmark_info->benchmark_info.GetTotalPrefillTurns());
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_benchmark_info_get_num_decode_turns(
    const LiteRtLmBenchmarkInfo* benchmark_info, int* out_num_turns) {
  LITERT_LM_C_RETURN_IF_NULL(out_num_turns);
  LITERT_LM_C_RETURN_IF_NULL(benchmark_info);
  *out_num_turns =
      static_cast<int>(benchmark_info->benchmark_info.GetTotalDecodeTurns());
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_benchmark_info_get_prefill_token_count_at(
    const LiteRtLmBenchmarkInfo* benchmark_info, int index,
    int* out_token_count) {
  LITERT_LM_C_RETURN_IF_NULL(out_token_count);
  LITERT_LM_C_RETURN_IF_NULL(benchmark_info);
  LITERT_LM_C_ASSIGN_OR_RETURN(
      const litert::lm::BenchmarkTurnData turn,
      benchmark_info->benchmark_info.GetPrefillTurn(index));
  *out_token_count = static_cast<int>(turn.num_tokens);
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_benchmark_info_get_decode_token_count_at(
    const LiteRtLmBenchmarkInfo* benchmark_info, int index,
    int* out_token_count) {
  LITERT_LM_C_RETURN_IF_NULL(out_token_count);
  LITERT_LM_C_RETURN_IF_NULL(benchmark_info);
  LITERT_LM_C_ASSIGN_OR_RETURN(
      const litert::lm::BenchmarkTurnData turn,
      benchmark_info->benchmark_info.GetDecodeTurn(index));
  *out_token_count = static_cast<int>(turn.num_tokens);
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_benchmark_info_get_prefill_tokens_per_sec_at(
    const LiteRtLmBenchmarkInfo* benchmark_info, int index,
    double* out_tokens_per_sec) {
  LITERT_LM_C_RETURN_IF_NULL(out_tokens_per_sec);
  LITERT_LM_C_RETURN_IF_NULL(benchmark_info);
  // Validates `index`; GetPrefillTokensPerSec() silently returns 0 otherwise.
  LITERT_LM_C_RETURN_IF_ERROR(
      benchmark_info->benchmark_info.GetPrefillTurn(index).status());
  *out_tokens_per_sec =
      benchmark_info->benchmark_info.GetPrefillTokensPerSec(index);
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_benchmark_info_get_decode_tokens_per_sec_at(
    const LiteRtLmBenchmarkInfo* benchmark_info, int index,
    double* out_tokens_per_sec) {
  LITERT_LM_C_RETURN_IF_NULL(out_tokens_per_sec);
  LITERT_LM_C_RETURN_IF_NULL(benchmark_info);
  // Validates `index`; GetDecodeTokensPerSec() silently returns 0 otherwise.
  LITERT_LM_C_RETURN_IF_ERROR(
      benchmark_info->benchmark_info.GetDecodeTurn(index).status());
  *out_tokens_per_sec =
      benchmark_info->benchmark_info.GetDecodeTokensPerSec(index);
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_engine_tokenize(
    LiteRtLmEngine* engine, const char* text,
    LiteRtLmTokenizeResult** out_result) {
  LITERT_LM_C_RETURN_IF_NULL(out_result);
  *out_result = nullptr;
  if (!engine || !engine->engine || !text) {
    return litert::lm::c::ReturnError(absl::StatusCode::kInvalidArgument,
                                      "Invalid engine or text.");
  }
  const auto& tokenizer = engine->engine->GetTokenizer();
  auto token_ids =
      const_cast<litert::support::Tokenizer&>(tokenizer).TextToTokenIds(text);
  if (!token_ids.ok()) {
    ABSL_LOG(ERROR) << "Failed to tokenize: " << token_ids.status();
    return litert::lm::c::ToCStatus(token_ids.status());
  }
  *out_result = new LiteRtLmTokenizeResult{std::move(*token_ids)};
  return kLiteRtLmStatusOk;
}

void litert_lm_tokenize_result_delete(LiteRtLmTokenizeResult* result) {
  delete result;
}

LiteRtLmStatusCode litert_lm_tokenize_result_get_tokens(
    const LiteRtLmTokenizeResult* result, const int** out_tokens) {
  LITERT_LM_C_RETURN_IF_NULL(out_tokens);
  *out_tokens = nullptr;
  LITERT_LM_C_RETURN_IF_NULL(result);
  *out_tokens = result->tokens.data();
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_tokenize_result_get_num_tokens(
    const LiteRtLmTokenizeResult* result, size_t* out_num_tokens) {
  LITERT_LM_C_RETURN_IF_NULL(out_num_tokens);
  LITERT_LM_C_RETURN_IF_NULL(result);
  *out_num_tokens = result->tokens.size();
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_engine_detokenize(
    LiteRtLmEngine* engine, const int* tokens, size_t num_tokens,
    LiteRtLmDetokenizeResult** out_result) {
  LITERT_LM_C_RETURN_IF_NULL(out_result);
  *out_result = nullptr;
  if (!engine || !engine->engine || !tokens) {
    return litert::lm::c::ReturnError(absl::StatusCode::kInvalidArgument,
                                      "Invalid engine or tokens.");
  }
  const auto& tokenizer = engine->engine->GetTokenizer();
  std::vector<int> token_ids(tokens, tokens + num_tokens);
  auto text = const_cast<litert::support::Tokenizer&>(tokenizer).TokenIdsToText(
      token_ids);
  if (!text.ok()) {
    ABSL_LOG(ERROR) << "Failed to detokenize: " << text.status();
    return litert::lm::c::ToCStatus(text.status());
  }
  *out_result = new LiteRtLmDetokenizeResult{std::move(*text)};
  return kLiteRtLmStatusOk;
}

void litert_lm_detokenize_result_delete(LiteRtLmDetokenizeResult* result) {
  delete result;
}

LiteRtLmStatusCode litert_lm_detokenize_result_get_string(
    const LiteRtLmDetokenizeResult* result, const char** out_text) {
  LITERT_LM_C_RETURN_IF_NULL(out_text);
  *out_text = nullptr;
  LITERT_LM_C_RETURN_IF_NULL(result);
  *out_text = result->text.c_str();
  return kLiteRtLmStatusOk;
}

void litert_lm_token_union_delete(LiteRtLmTokenUnion* token_union) {
  delete token_union;
}

LiteRtLmStatusCode litert_lm_token_union_get_type(
    const LiteRtLmTokenUnion* token_union, LiteRtLmTokenUnionType* out_type) {
  LITERT_LM_C_RETURN_IF_NULL(out_type);
  LITERT_LM_C_RETURN_IF_NULL(token_union);
  *out_type = token_union->token_union.has_token_str()
                  ? kLiteRtLmTokenUnionTypeString
                  : kLiteRtLmTokenUnionTypeIds;
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_token_union_get_string(
    const LiteRtLmTokenUnion* token_union, const char** out_string) {
  LITERT_LM_C_RETURN_IF_NULL(out_string);
  *out_string = nullptr;
  LITERT_LM_C_RETURN_IF_NULL(token_union);
  if (!token_union->token_union.has_token_str()) {
    return litert::lm::c::ReturnError(absl::StatusCode::kInvalidArgument,
                                      "Token union does not contain a string.");
  }
  *out_string = token_union->token_union.token_str().c_str();
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_token_union_get_ids(
    const LiteRtLmTokenUnion* token_union, const int** out_tokens,
    size_t* out_num_tokens) {
  LITERT_LM_C_RETURN_IF_NULL(out_tokens);
  *out_tokens = nullptr;
  LITERT_LM_C_RETURN_IF_NULL(out_num_tokens);
  LITERT_LM_C_RETURN_IF_NULL(token_union);
  if (!token_union->token_union.has_token_ids()) {
    return litert::lm::c::ReturnError(
        absl::StatusCode::kInvalidArgument,
        "Token union does not contain token ids.");
  }
  *out_tokens = token_union->token_union.token_ids().ids().data();
  *out_num_tokens = token_union->token_union.token_ids().ids_size();
  return kLiteRtLmStatusOk;
}

void litert_lm_token_unions_delete(LiteRtLmTokenUnions* tokens) {
  delete tokens;
}

LiteRtLmStatusCode litert_lm_token_unions_get_num_tokens(
    const LiteRtLmTokenUnions* tokens, size_t* out_num_tokens) {
  LITERT_LM_C_RETURN_IF_NULL(out_num_tokens);
  LITERT_LM_C_RETURN_IF_NULL(tokens);
  *out_num_tokens = tokens->tokens.size();
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_token_unions_get_token_at(
    const LiteRtLmTokenUnions* tokens, size_t index,
    LiteRtLmTokenUnion** out_token) {
  LITERT_LM_C_RETURN_IF_NULL(out_token);
  *out_token = nullptr;
  LITERT_LM_C_RETURN_IF_NULL(tokens);
  if (index >= tokens->tokens.size()) {
    return litert::lm::c::ReturnError(absl::StatusCode::kOutOfRange,
                                      "Token index out of range.");
  }
  auto result = std::make_unique<LiteRtLmTokenUnion>();
  result->token_union = tokens->tokens[index];
  *out_token = result.release();
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_engine_get_start_token(
    LiteRtLmEngine* engine, LiteRtLmTokenUnion** out_token) {
  LITERT_LM_C_RETURN_IF_NULL(out_token);
  *out_token = nullptr;
  if (!engine || !engine->engine) {
    return litert::lm::c::ReturnError(absl::StatusCode::kInvalidArgument,
                                      "Invalid engine.");
  }
  const auto& metadata = engine->engine->GetEngineSettings().GetLlmMetadata();
  if (!metadata.has_value() || !metadata->has_start_token()) {
    // No start token configured: success with a NULL result.
    return kLiteRtLmStatusOk;
  }
  *out_token = new LiteRtLmTokenUnion{metadata->start_token()};
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_engine_get_stop_tokens(
    LiteRtLmEngine* engine, LiteRtLmTokenUnions** out_tokens) {
  LITERT_LM_C_RETURN_IF_NULL(out_tokens);
  *out_tokens = nullptr;
  if (!engine || !engine->engine) {
    return litert::lm::c::ReturnError(absl::StatusCode::kInvalidArgument,
                                      "Invalid engine.");
  }
  const auto& metadata = engine->engine->GetEngineSettings().GetLlmMetadata();
  if (!metadata.has_value() || metadata->stop_tokens_size() == 0) {
    // No stop tokens configured: success with a NULL result.
    return kLiteRtLmStatusOk;
  }
  auto c_tokens = std::make_unique<LiteRtLmTokenUnions>();
  c_tokens->tokens.assign(metadata->stop_tokens().begin(),
                          metadata->stop_tokens().end());
  *out_tokens = c_tokens.release();
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_stream_chunk_get_text(
    const LiteRtLmStreamChunk* chunk, const char** out_text) {
  LITERT_LM_C_RETURN_IF_NULL(out_text);
  *out_text = nullptr;
  LITERT_LM_C_RETURN_IF_NULL(chunk);
  // A chunk without text content yields success with a NULL result.
  *out_text = chunk->text;
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_stream_chunk_is_final(
    const LiteRtLmStreamChunk* chunk, bool* out_is_final) {
  LITERT_LM_C_RETURN_IF_NULL(out_is_final);
  LITERT_LM_C_RETURN_IF_NULL(chunk);
  *out_is_final = chunk->is_final;
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_stream_chunk_get_error(
    const LiteRtLmStreamChunk* chunk, const char** out_error) {
  LITERT_LM_C_RETURN_IF_NULL(out_error);
  *out_error = nullptr;
  LITERT_LM_C_RETURN_IF_NULL(chunk);
  // A chunk without an error yields success with a NULL result.
  *out_error = chunk->error_msg;
  return kLiteRtLmStatusOk;
}

}  // extern "C"
