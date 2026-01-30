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

#include "runtime/engine/engine_settings.h"

#include <algorithm>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_split.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/internal/scoped_file.h"  // from @litert
#include "runtime/components/tokenizer.h"
#include "runtime/executor/audio_executor_settings.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/executor/vision_executor_settings.h"
#include "runtime/proto/engine.pb.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/proto/llm_model_type.pb.h"
#include "runtime/proto/sampler_params.pb.h"
#include "runtime/proto/token.pb.h"
#include "runtime/util/model_type_utils.h"
#include "runtime/util/status_macros.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

// Margin for the default prefill batch size assuming the tokens to indicate the
// start and end of the input prompt.
constexpr int kDefaultPrefillBatchSizeMargin = 2;

std::ostream& operator<<(std::ostream& os, const std::vector<int>& vec) {
  constexpr int newline_num = 10;
  os << "vector size: " << vec.size() << ": [";
  for (int i = 0; i < vec.size(); ++i) {
    os << vec[i];
    if (i < vec.size() - 1) {
      os << ", ";
    }
    if ((i + 1) % newline_num == 0) {
      os << "\n";
    }
  }
  os << "]";
  return os;
}

absl::Status ValidateBackendConstraint(
    ExecutorSettingsBase& executor_settings,  // Polymorphic executor settings.
    const std::optional<std::string>& backend_constraint,
    absl::string_view modality_name) {
  if (backend_constraint.has_value()) {
    // When both the executor settings and the backend constraint are set, we
    // check if the backend constraint contains the backend of the executor
    // settings.
    std::string backend_constraint_str = backend_constraint.value();
    std::string backend = GetBackendString(executor_settings.GetBackend());
    std::vector<std::string> constraints =
        absl::StrSplit(backend_constraint_str, ',');
    bool found =
        std::any_of(constraints.begin(), constraints.end(),
                    [&](absl::string_view constraint) {
                      return absl::EqualsIgnoreCase(constraint, backend);
                    });
    if (!found) {
      return absl::InvalidArgumentError(
          absl::StrCat(modality_name,
                       " backend constraint mismatch. Model requires one of [",
                       backend_constraint_str, "] but ", modality_name,
                       " backend is ", backend));
    }
    ABSL_LOG(INFO) << "The " << modality_name
                   << " backend constraint is matched: " << backend;
  } else {
    ABSL_LOG(INFO) << "The " << modality_name
                   << " backend constraint is not set.";
  }
  return absl::OkStatus();
}

}  // namespace

// static
absl::StatusOr<EngineSettings> EngineSettings::CreateDefault(
    ModelAssets model_assets, Backend backend,
    std::optional<Backend> vision_backend,
    std::optional<Backend> audio_backend) {
  ASSIGN_OR_RETURN(  // NOLINT
      auto executor_settings,
      LlmExecutorSettings::CreateDefault(model_assets, backend));
  std::optional<VisionExecutorSettings> vision_executor_settings;
  if (vision_backend.has_value()) {
    ASSIGN_OR_RETURN(
        vision_executor_settings,
        VisionExecutorSettings::CreateDefault(
            model_assets, /*encoder_backend=*/vision_backend.value(),
            // Vision adapter can only run on CPU.
            /*adapter_backend=*/Backend::CPU));
  }
  std::optional<AudioExecutorSettings> audio_executor_settings;
  if (audio_backend.has_value()) {
    ASSIGN_OR_RETURN(audio_executor_settings,
                     AudioExecutorSettings::CreateDefault(
                         model_assets, executor_settings.GetMaxNumTokens(),
                         audio_backend.value()));
  }
  return EngineSettings(std::move(executor_settings),
                        std::move(vision_executor_settings),
                        std::move(audio_executor_settings));
}

absl::Status EngineSettings::MaybeUpdateAndValidate(
    Tokenizer& tokenizer,
    const proto::LlmMetadata* absl_nullable metadata_from_file,
    absl::string_view input_prompt_as_hint,
    const std::optional<std::string>& text_backend_constraint,
    const std::optional<std::string>& vision_backend_constraint,
    const std::optional<std::string>& audio_backend_constraint) {
  proto::LlmMetadata& metadata = GetMutableLlmMetadata();
  // Copy the metadata from the file if it is provided.
  if (metadata_from_file != nullptr) {
    metadata = *metadata_from_file;
  }

  // Convert the start/stop tokens from string to token ids.
  for (auto& stop_token : *metadata.mutable_stop_tokens()) {
    if (stop_token.has_token_str()) {
      auto stop_token_id = tokenizer.TokenToId(stop_token.token_str());
      if (stop_token_id.ok()) {
        stop_token.mutable_token_ids()->mutable_ids()->Add(*stop_token_id);
      } else {
        auto stop_token_ids = tokenizer.TextToTokenIds(stop_token.token_str());
        if (stop_token_ids.ok()) {
          stop_token.mutable_token_ids()->mutable_ids()->Add(
              stop_token_ids->begin(), stop_token_ids->end());
        }
      }
    }
  }
  if (metadata.start_token().has_token_str()) {
    auto start_token_id =
        tokenizer.TokenToId(metadata.start_token().token_str());
    if (start_token_id.ok()) {
      metadata.mutable_start_token()->mutable_token_ids()->mutable_ids()->Add(
          *start_token_id);
    } else {
      auto start_token_ids =
          tokenizer.TextToTokenIds(metadata.start_token().token_str());
      if (start_token_ids.ok()) {
        metadata.mutable_start_token()->mutable_token_ids()->mutable_ids()->Add(
            start_token_ids->begin(), start_token_ids->end());
      }
    }
  }

  int num_prompt_tokens = 0;
  if (!input_prompt_as_hint.empty()) {
    num_prompt_tokens = tokenizer.TextToTokenIds(input_prompt_as_hint)
                            .value_or(std::vector<int>())
                            .size();
  }

  // Load the max num tokens from the model file.
  // If not set, we set the default value to one based on the number of tokens
  // in the prompt.
  if (main_executor_settings_.GetMaxNumTokens() == 0) {
    // The default maximum number of tokens is set to the smallest multiple of
    // 4096 greater than the number of tokens in the prompt plus the default
    // decode length, 1024.
    int max_num_tokens = ((num_prompt_tokens + 1023) / 4096 + 1) * 4096;
    if (metadata.max_num_tokens() > 0) {
      max_num_tokens = metadata.max_num_tokens();
    }
    main_executor_settings_.SetMaxNumTokens(max_num_tokens);
  }

  if (num_prompt_tokens > 0) {
    AdvancedSettings advanced_settings;
    if (main_executor_settings_.GetAdvancedSettings()) {
      advanced_settings = *main_executor_settings_.GetAdvancedSettings();
    }
    if (advanced_settings.prefill_batch_sizes.empty()) {
      // If the prefill batch size is not set, set it to the number of tokens
      // in the input prompt with some margin.
      advanced_settings.prefill_batch_sizes.insert(
          num_prompt_tokens + kDefaultPrefillBatchSizeMargin);
      main_executor_settings_.SetAdvancedSettings(advanced_settings);
    }
  }

  // Set the default values for the sampler params.
  if (!metadata.has_sampler_params()) {
    proto::SamplerParameters& sampler_params =
        *metadata.mutable_sampler_params();
    Backend backend = main_executor_settings_.GetBackend();
    if (backend == Backend::NPU || backend == Backend::GPU_ARTISAN) {
      sampler_params.set_type(proto::SamplerParameters::TYPE_UNSPECIFIED);
    } else if (backend == Backend::CPU || backend == Backend::GPU
    ) {
      sampler_params.set_type(proto::SamplerParameters::TOP_P);
      sampler_params.set_k(1);
      sampler_params.set_p(0.95f);
      sampler_params.set_temperature(1.0f);
      sampler_params.set_seed(0);
    } else {
      return absl::InvalidArgumentError(
          absl::StrCat("Not recognized backend: ", backend));
    }
  }

  if (!metadata.has_llm_model_type()) {
    const auto& model_assets = main_executor_settings_.GetModelAssets();
    auto model_path = model_assets.GetPath();
    ASSIGN_OR_RETURN(*metadata.mutable_llm_model_type(),
                     InferLlmModelType(metadata, tokenizer));
  }

  if (metadata.has_llm_model_type() &&
      metadata.llm_model_type().model_type_case() ==
          proto::LlmModelType::kGemma3N) {
    auto advanced_settings = AdvancedSettings();
    if (main_executor_settings_.GetAdvancedSettings()) {
      advanced_settings = *main_executor_settings_.GetAdvancedSettings();
    }
    advanced_settings.allow_src_quantized_fc_conv_ops = true;
    main_executor_settings_.SetAdvancedSettings(advanced_settings);
  }

  if (!metadata.has_jinja_prompt_template()) {
    ASSIGN_OR_RETURN(*metadata.mutable_jinja_prompt_template(),
                     GetDefaultJinjaPromptTemplate(metadata.prompt_templates(),
                                                   metadata.llm_model_type()));
  }

  // If the executor settings is set, then check if the input backend
  // constraint is compatible with the executor settings.
  RETURN_IF_ERROR(ValidateBackendConstraint(main_executor_settings_,
                                            text_backend_constraint, "Main"));

  if (vision_executor_settings_.has_value()) {
    RETURN_IF_ERROR(ValidateBackendConstraint(vision_executor_settings_.value(),
                                              vision_backend_constraint,
                                              "Vision"));
  }
  if (audio_executor_settings_.has_value()) {
    RETURN_IF_ERROR(ValidateBackendConstraint(
        audio_executor_settings_.value(), audio_backend_constraint, "Audio"));
  }

  ABSL_VLOG(5) << "The llm metadata: " << metadata.DebugString();
  ABSL_LOG(INFO) << "The validated engine settings: " << *this;
  return absl::OkStatus();
}

EngineSettings::EngineSettings(
    LlmExecutorSettings executor_settings,
    std::optional<VisionExecutorSettings> vision_executor_settings,
    std::optional<AudioExecutorSettings> audio_executor_settings,
    std::optional<proto::BenchmarkParams> benchmark_params)
    : main_executor_settings_(std::move(executor_settings)),
      vision_executor_settings_(std::move(vision_executor_settings)),
      audio_executor_settings_(std::move(audio_executor_settings)),
      benchmark_params_(benchmark_params) {}

const LlmExecutorSettings& EngineSettings::GetMainExecutorSettings() const {
  return main_executor_settings_;
}

LlmExecutorSettings& EngineSettings::GetMutableMainExecutorSettings() {
  return main_executor_settings_;
}

const std::optional<VisionExecutorSettings>&
EngineSettings::GetVisionExecutorSettings() const {
  return vision_executor_settings_;
}

std::optional<VisionExecutorSettings>&
EngineSettings::GetMutableVisionExecutorSettings() {
  return vision_executor_settings_;
}

const std::optional<AudioExecutorSettings>&
EngineSettings::GetAudioExecutorSettings() const {
  return audio_executor_settings_;
}

std::optional<AudioExecutorSettings>&
EngineSettings::GetMutableAudioExecutorSettings() {
  return audio_executor_settings_;
}

// Benchmark parameters:
// Returns true if the benchmark is enabled.
bool EngineSettings::IsBenchmarkEnabled() const {
  return benchmark_params_.has_value();
}
// Returns the benchmark parameters.
const std::optional<proto::BenchmarkParams>&
EngineSettings::GetBenchmarkParams() const {
  return benchmark_params_;
}
// Returns the mutable benchmark parameters.
proto::BenchmarkParams& EngineSettings::GetMutableBenchmarkParams() {
  if (!benchmark_params_.has_value()) {
    benchmark_params_ = proto::BenchmarkParams();
  }
  return benchmark_params_.value();
}

const std::optional<proto::LlmMetadata>& EngineSettings::GetLlmMetadata()
    const {
  return metadata_;
}

std::ostream& operator<<(std::ostream& os, const EngineSettings& settings) {
  os << "EngineSettings: " << std::endl;
  os << "  MainExecutorSettings: " << settings.GetMainExecutorSettings();
  if (settings.GetLlmMetadata().has_value()) {
    os << "  LlmMetadata: " << settings.GetLlmMetadata().value().DebugString();
  } else {
    os << "  LlmMetadata: Not set" << std::endl;
  }
  if (settings.GetBenchmarkParams().has_value()) {
    os << "  BenchmarkParams: "
       << settings.GetBenchmarkParams().value().DebugString();
  } else {
    os << "  BenchmarkParams: Not set" << std::endl;
  }
  if (settings.GetVisionExecutorSettings().has_value()) {
    os << "  VisionExecutorSettings: "
       << settings.GetVisionExecutorSettings().value();
  } else {
    os << "  VisionExecutorSettings: Not set" << std::endl;
  }
  if (settings.GetAudioExecutorSettings().has_value()) {
    os << "  AudioExecutorSettings: "
       << settings.GetAudioExecutorSettings().value();
  } else {
    os << "  AudioExecutorSettings: Not set" << std::endl;
  }
  return os;
}

proto::LlmMetadata& EngineSettings::GetMutableLlmMetadata() {
  if (!metadata_.has_value()) {
    metadata_ = proto::LlmMetadata();
  }
  return metadata_.value();
}

SessionConfig SessionConfig::CreateDefault() {
  proto::SamplerParameters sampler_params;
  sampler_params.set_type(proto::SamplerParameters::TYPE_UNSPECIFIED);
  auto config = SessionConfig(sampler_params);
  config.SetNumOutputCandidates(1);
  // Default to -1 to indicate the start token is not set. This is to be
  // overridden by the EngineSettings.
  config.SetStartTokenId(-1);
  return config;
}

absl::Status SessionConfig::MaybeUpdateAndValidate(
    const EngineSettings& engine_settings) {
  if ((stop_token_ids_.empty()) &&
      !engine_settings.GetLlmMetadata().has_value()) {
    return absl::InvalidArgumentError(
        "Required: set stop tokens, or provide LlmMetadata.");
  }

  // Update the parameters from the engine settings when the LlmMetadata is
  // present.
  if (engine_settings.GetLlmMetadata().has_value()) {
    const auto llm_metadata = engine_settings.GetLlmMetadata().value();
    proto::SamplerParameters& sampler_params = GetMutableSamplerParams();
    // Update the sampler params if the session config does not have a sampler
    // params and the engine settings has a sampler params (probably read from
    // the model file).
    if ((sampler_params.type() == proto::SamplerParameters::TYPE_UNSPECIFIED)) {
      if (llm_metadata.has_sampler_params()) {
        sampler_params = engine_settings.GetLlmMetadata()->sampler_params();
      }
    }

    // Set and validate the start token.
    if (start_token_id_ == -1) {
      if (llm_metadata.has_start_token()) {
        if (llm_metadata.start_token().token_ids().ids_size() > 1) {
          ABSL_LOG(WARNING) << "The start token has more than one token ids: ";
        }
        start_token_id_ = llm_metadata.start_token().token_ids().ids(0);
      }
    }

    // Set and validate the stop tokens.
    if (stop_token_ids_.empty()) {
      for (const auto& stop_token : llm_metadata.stop_tokens()) {
        if (stop_token.has_token_ids() &&
            stop_token.token_ids().ids_size() > 0) {
          std::vector<int> stop_token_ids(stop_token.token_ids().ids().begin(),
                                          stop_token.token_ids().ids().end());
          stop_token_ids_.push_back(stop_token_ids);
        }
      }
    }

    // Set the prompt template from LlmMetadata, if not provided in
    // SessionConfig.
    //
    // Hack: use the user field to check if the prompt template is being set.
    // To use the empty prompt_template, set the user field with empty prefix.
    //
    // TODO(b/439648399): Remove this logic when LiteRT-LM no longer use
    // template in Session level.
    if (!prompt_templates_.has_user() && llm_metadata.has_prompt_templates()) {
      prompt_templates_ = llm_metadata.prompt_templates();
    }

    if (llm_model_type_.model_type_case() ==
        proto::LlmModelType::MODEL_TYPE_NOT_SET) {
      llm_model_type_ = llm_metadata.llm_model_type();
    }
  }

  // Validating the required fields are set correctly.
  if (stop_token_ids_.empty()) {
    return absl::InvalidArgumentError(
        "Stop tokens are required. Either set the stop token ids or "
        "provide "
        "a valid stop token in the model file/engine settings.");
  }
  if (num_output_candidates_ < 1) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Number of output candidates need to be at least 1, but got: ",
        num_output_candidates_));
  }

  if (sampler_backend_ == Backend::UNSPECIFIED) {
    if (engine_settings.GetMainExecutorSettings().GetBackend() ==
        Backend::GPU) {
      sampler_backend_ = Backend::GPU;
    } else {
      sampler_backend_ = Backend::CPU;
    }
  }

  ABSL_VLOG(5) << "The validated session config: " << *this;
  return absl::OkStatus();
}

SessionConfig::SessionConfig(const proto::SamplerParameters& sampler_params)
    : sampler_params_(sampler_params) {}

const proto::SamplerParameters& SessionConfig::GetSamplerParams() const {
  return sampler_params_;
}

proto::SamplerParameters& SessionConfig::GetMutableSamplerParams() {
  return sampler_params_;
}

const std::vector<std::vector<int>>& SessionConfig::GetStopTokenIds() const {
  return stop_token_ids_;
}

std::vector<std::vector<int>>& SessionConfig::GetMutableStopTokenIds() {
  return stop_token_ids_;
}

int SessionConfig::GetStartTokenId() const { return start_token_id_; }

void SessionConfig::SetStartTokenId(int start_token_id) {
  start_token_id_ = start_token_id;
}

int SessionConfig::GetNumOutputCandidates() const {
  return num_output_candidates_;
}

void SessionConfig::SetNumOutputCandidates(int num_output_candidates) {
  num_output_candidates_ = num_output_candidates;
}

const proto::PromptTemplates& SessionConfig::GetPromptTemplates() const {
  return prompt_templates_;
}

proto::PromptTemplates& SessionConfig::GetMutablePromptTemplates() {
  return prompt_templates_;
}

const proto::LlmModelType& SessionConfig::GetLlmModelType() const {
  return llm_model_type_;
}

proto::LlmModelType& SessionConfig::GetMutableLlmModelType() {
  return llm_model_type_;
}

std::shared_ptr<::litert::ScopedFile> SessionConfig::GetScopedLoraFile()
    const {
  return scoped_lora_file_;
}

void SessionConfig::SetScopedLoraFile(
    std::shared_ptr<::litert::ScopedFile> scoped_lora_file) {
  scoped_lora_file_ = std::move(scoped_lora_file);
}

std::ostream& operator<<(std::ostream& os, const SessionConfig& config) {
  os << "SessionConfig: " << std::endl;
  os << "  AudioModalityEnabled: " << config.AudioModalityEnabled()
     << std::endl;
  os << "  VisionModalityEnabled: " << config.VisionModalityEnabled()
     << std::endl;
  os << "  SamplerParams: " << config.GetSamplerParams().DebugString()
     << std::endl;
  os << "  SamplerBackend: " << config.GetSamplerBackend() << std::endl;
  os << "  StartTokenId: " << config.GetStartTokenId() << std::endl;
  os << "  StopTokenIds: " << std::endl;
  for (const auto& stop_token_ids : config.GetStopTokenIds()) {
    os << "    " << stop_token_ids << std::endl;
  }
  os << "  NumOutputCandidates: " << config.GetNumOutputCandidates()
     << std::endl;
  os << "  LlmModelType: " << config.GetLlmModelType().DebugString()
     << std::endl;
  os << "  PromptTemplates: " << config.GetPromptTemplates().DebugString()
     << std::endl;
  os << "  ApplyPromptTemplatesInSession: "
     << config.GetApplyPromptTemplateInSession() << std::endl;
  os << "  ScopedLoraFile: "
     << (config.GetScopedLoraFile() != nullptr ? "Present" : "Not present")
     << std::endl;
  return os;
}

Backend SessionConfig::GetSamplerBackend() const { return sampler_backend_; }
void SessionConfig::SetSamplerBackend(Backend sampler_backend) {
  sampler_backend_ = sampler_backend;
}

}  // namespace litert::lm
