#include "runtime/engine/engine_settings.h"

#include <optional>
#include <ostream>
#include <utility>
#include <vector>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/components/tokenizer.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/proto/engine.pb.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/proto/sampler_params.pb.h"
#include "runtime/util/status_macros.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

std::ostream& operator<<(std::ostream& os,
                         const std::vector<int>& vec) {
  constexpr int newline_num = 10;
  os << "vector size: " << vec.size() << ": [";
  for (std::size_t i = 0; i < vec.size(); ++i) {
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

}  // namespace

// static
absl::StatusOr<EngineSettings> EngineSettings::CreateDefault(
    ModelAssets model_assets, Backend backend) {
  ASSIGN_OR_RETURN(  // NOLINT
      auto executor_settings,
      LlmExecutorSettings::CreateDefault(std::move(model_assets), backend));
  return EngineSettings(std::move(executor_settings), std::nullopt);
}

absl::Status EngineSettings::MaybeUpdateAndValidate(
    Tokenizer& tokenizer, const proto::LlmMetadata* metadata_from_file) {
  proto::LlmMetadata& metadata = GetMutableLlmMetadata();
  // Copy the metadata from the file if it is provided.
  if (metadata_from_file != nullptr) {
    metadata = *metadata_from_file;
  }

  // Convert the start/stop tokens from string to token ids.
  for (auto& stop_token : *metadata.mutable_stop_tokens()) {
    if (stop_token.has_token_str()) {
      auto stop_token_ids = tokenizer.TextToTokenIds(stop_token.token_str());
      if (stop_token_ids.ok()) {
        stop_token.mutable_token_ids()->mutable_ids()->Add(
            stop_token_ids->begin(), stop_token_ids->end());
      }
    }
  }
  if (metadata.start_token().has_token_str()) {
    auto start_token_ids =
        tokenizer.TextToTokenIds(metadata.start_token().token_str());
    if (start_token_ids.ok()) {
      metadata.mutable_start_token()
          ->mutable_token_ids()
          ->mutable_ids()
          ->Add(start_token_ids->begin(), start_token_ids->end());
    }
  }
  // Load the max num tokens from the model file.
  // If not set, we set the default value to 4096.
  if (main_executor_settings_.GetMaxNumTokens() == 0) {
    int max_num_tokens = 4096;
    if (metadata.max_num_tokens() > 0) {
      max_num_tokens = metadata.max_num_tokens();
    }
    main_executor_settings_.SetMaxNumTokens(max_num_tokens);
  }

  // Set the default values for the sampler params.
  if (!metadata.has_sampler_params()) {
    proto::SamplerParameters& sampler_params =
        *metadata.mutable_sampler_params();
    Backend backend = main_executor_settings_.GetBackend();
    if (backend == Backend::NPU) {
      sampler_params.set_type(proto::SamplerParameters::TYPE_UNSPECIFIED);
    } else if (backend == Backend::CPU || backend == Backend::GPU) {
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
  ABSL_LOG(INFO) << "The llm metadata: " << metadata.DebugString();
  ABSL_LOG(INFO) << "The validated engine settings: " << *this;
  return absl::OkStatus();
}

EngineSettings::EngineSettings(
    LlmExecutorSettings executor_settings,
    std::optional<proto::BenchmarkParams> benchmark_params)
    : main_executor_settings_(std::move(executor_settings)),
      benchmark_params_(benchmark_params) {}

const LlmExecutorSettings& EngineSettings::GetMainExecutorSettings() const {
  return main_executor_settings_;
}

LlmExecutorSettings& EngineSettings::GetMutableMainExecutorSettings() {
  return main_executor_settings_;
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
  ABSL_LOG(INFO)
      << "The GetLlmMetadata: "
      << (engine_settings.GetLlmMetadata().has_value()
              ? engine_settings.GetLlmMetadata().value().DebugString()
              : "Not set");
  if ((start_token_id_ == -1 || stop_token_ids_.empty()) &&
      !engine_settings.GetLlmMetadata().has_value()) {
    return absl::InvalidArgumentError(
        "Required: set start and stop tokens, or provide LlmMetadata.");
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
          std::vector<int> stop_token_ids(
              stop_token.token_ids().ids().begin(),
              stop_token.token_ids().ids().end());
          stop_token_ids_.push_back(stop_token_ids);
        }
      }
    }

    // Set the prompt template.
    if (llm_metadata.has_prompt_templates()) {
      prompt_templates_ = llm_metadata.prompt_templates();
    }
  }

  // Validating the required fields are set correctly.
  if (start_token_id_ == -1) {
    return absl::InvalidArgumentError(
        "Start token is required. Either set the start token id or provide "
        "a valid start token in the model file/engine settings.");
  }
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

  ABSL_LOG(INFO) << "The validated session config: " << *this;
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

std::ostream& operator<<(std::ostream& os, const SessionConfig& config) {
  os << "SessionConfig: " << std::endl;
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
  os << "  PromptTemplates: " << config.GetPromptTemplates().DebugString()
     << std::endl;
  return os;
}

Backend SessionConfig::GetSamplerBackend() const { return sampler_backend_; }
void SessionConfig::SetSamplerBackend(Backend sampler_backend) {
  sampler_backend_ = sampler_backend;
}

}  // namespace litert::lm
