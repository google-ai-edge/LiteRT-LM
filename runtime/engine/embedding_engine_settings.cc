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

#include "runtime/engine/embedding_engine_settings.h"

#include <algorithm>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_split.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/executor/audio/audio_executor_settings.h"
#include "runtime/executor/embedding/embedding_executor_settings.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/vision/vision_executor_settings.h"
#include "runtime/proto/embedding_metadata.pb.h"
#include "runtime/util/file_util.h"
#include "runtime/util/status_macros.h"

namespace litert::lm {

absl::StatusOr<EmbeddingEngineSettings> EmbeddingEngineSettings::CreateDefault(
    ModelAssets model_assets, Backend backend,
    std::optional<Backend> vision_backend,
    std::optional<Backend> audio_backend) {
  LITERT_ASSIGN_OR_RETURN(
      auto embedding_executor_settings,
      EmbeddingExecutorSettings::CreateDefault(model_assets, backend));
  std::optional<VisionExecutorSettings> vision_executor_settings;
  if (vision_backend.has_value()) {
    LITERT_ASSIGN_OR_RETURN(
        vision_executor_settings,
        VisionExecutorSettings::CreateDefault(
            model_assets, /*encoder_backend=*/*vision_backend,
            /*adapter_backend=*/Backend::CPU));
  }
  std::optional<AudioExecutorSettings> audio_executor_settings;
  if (audio_backend.has_value()) {
    LITERT_ASSIGN_OR_RETURN(
        audio_executor_settings,
        AudioExecutorSettings::CreateDefault(
            model_assets, /*max_sequence_length=*/30, *audio_backend));
  }
  return EmbeddingEngineSettings(std::move(embedding_executor_settings),
                                 std::move(vision_executor_settings),
                                 std::move(audio_executor_settings));
}

EmbeddingEngineSettings::EmbeddingEngineSettings(
    EmbeddingExecutorSettings embedding_executor_settings,
    std::optional<VisionExecutorSettings> vision_executor_settings,
    std::optional<AudioExecutorSettings> audio_executor_settings,
    std::optional<proto::BenchmarkParams> benchmark_params)
    : main_executor_settings_(std::move(embedding_executor_settings)),
      vision_executor_settings_(std::move(vision_executor_settings)),
      audio_executor_settings_(std::move(audio_executor_settings)),
      benchmark_params_(std::move(benchmark_params)) {}

std::optional<int> EmbeddingEngineSettings::GetMaxInputLength() const {
  return max_input_length_;
}

void EmbeddingEngineSettings::SetMaxInputLength(
    std::optional<int> max_input_length) {
  max_input_length_ = max_input_length;
}

std::optional<int> EmbeddingEngineSettings::GetVisionTokensPerImage() const {
  return vision_tokens_per_image_;
}

void EmbeddingEngineSettings::SetVisionTokensPerImage(
    std::optional<int> vision_tokens_per_image) {
  vision_tokens_per_image_ = vision_tokens_per_image;
}

const EmbeddingExecutorSettings&
EmbeddingEngineSettings::GetMainExecutorSettings() const {
  return main_executor_settings_;
}

EmbeddingExecutorSettings&
EmbeddingEngineSettings::GetMutableMainExecutorSettings() {
  return main_executor_settings_;
}

const std::optional<VisionExecutorSettings>&
EmbeddingEngineSettings::GetVisionExecutorSettings() const {
  return vision_executor_settings_;
}

std::optional<VisionExecutorSettings>&
EmbeddingEngineSettings::GetMutableVisionExecutorSettings() {
  return vision_executor_settings_;
}

const std::optional<AudioExecutorSettings>&
EmbeddingEngineSettings::GetAudioExecutorSettings() const {
  return audio_executor_settings_;
}

std::optional<AudioExecutorSettings>&
EmbeddingEngineSettings::GetMutableAudioExecutorSettings() {
  return audio_executor_settings_;
}

bool EmbeddingEngineSettings::IsBenchmarkEnabled() const {
  return benchmark_params_.has_value();
}

const std::optional<proto::BenchmarkParams>&
EmbeddingEngineSettings::GetBenchmarkParams() const {
  return benchmark_params_;
}

proto::BenchmarkParams& EmbeddingEngineSettings::GetMutableBenchmarkParams() {
  if (!benchmark_params_.has_value()) {
    benchmark_params_ = proto::BenchmarkParams();
  }
  return *benchmark_params_;
}

const std::optional<proto::EmbeddingMetadata>&
EmbeddingEngineSettings::GetEmbeddingMetadata() const {
  return metadata_;
}

proto::EmbeddingMetadata&
EmbeddingEngineSettings::GetMutableEmbeddingMetadata() {
  if (!metadata_.has_value()) {
    metadata_ = proto::EmbeddingMetadata();
  }
  return *metadata_;
}

namespace {

// Resolves the activation data type according to the precedence waterfall:
// 1. User settings in code (takes priority if already set).
// 2. prefer_activation_type from model metadata / TOML (if provided).
// 3. FLOAT16 fallback if the backend is GPU.
absl::Status ResolveActivationDataType(
    ExecutorSettingsBase& executor_settings,
    const std::optional<std::string>& prefer_activation_type) {
  // Precedence 1: User explicitly set it in code.
  if (executor_settings.GetActivationDataType().has_value()) {
    return absl::OkStatus();
  }

  // Precedence 2: prefer_activation_type from metadata / TOML.
  if (prefer_activation_type.has_value() && !prefer_activation_type->empty()) {
    LITERT_ASSIGN_OR_RETURN(
        ActivationDataType activation_data_type,
        GetActivationDataTypeFromString(*prefer_activation_type));
    executor_settings.SetActivationDataType(activation_data_type);
    if (*prefer_activation_type == "fp32_fp16") {
      executor_settings.SetEnableMixedPrecision(true);
    }
    return absl::OkStatus();
  }

  // Precedence 3: Fallback to FLOAT16 if the backend is GPU.
  if (executor_settings.GetBackend() == Backend::GPU) {
    executor_settings.SetActivationDataType(ActivationDataType::FLOAT16);
  }
  return absl::OkStatus();
}

absl::Status ValidateBackendConstraint(
    const ExecutorSettingsBase& executor_settings,
    const std::optional<std::string>& backend_constraint,
    absl::string_view modality_name) {
  if (!backend_constraint.has_value() || backend_constraint->empty()) {
    return absl::OkStatus();
  }
  std::string backend_str = GetBackendString(executor_settings.GetBackend());
  std::vector<std::string> constraints =
      absl::StrSplit(*backend_constraint, ',');
  bool found =
      std::any_of(constraints.begin(), constraints.end(),
                  [&](absl::string_view constraint) {
                    return absl::EqualsIgnoreCase(constraint, backend_str);
                  });
  if (!found) {
    return absl::InvalidArgumentError(absl::StrCat(
        modality_name, " backend constraint mismatch. Model requires one of [",
        *backend_constraint, "] but ", modality_name, " backend is ",
        backend_str));
  }
  return absl::OkStatus();
}

absl::Status ValidateCacheDir(absl::string_view cache_dir) {
  if (cache_dir.empty() || cache_dir == ":nocache" || cache_dir == ":memory") {
    return absl::OkStatus();
  }
  if (!IsDirectoryWritable(cache_dir)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Cache directory does not exist or is not writable: ", cache_dir));
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status EmbeddingEngineSettings::ResolveDefaults(
    const std::optional<std::string>& text_prefer_activation_type,
    const std::optional<std::string>& vision_prefer_activation_type,
    const std::optional<std::string>& audio_prefer_activation_type) {
  LITERT_RETURN_IF_ERROR(ResolveActivationDataType(
      main_executor_settings_, text_prefer_activation_type));
  if (vision_executor_settings_.has_value()) {
    LITERT_RETURN_IF_ERROR(ResolveActivationDataType(
        *vision_executor_settings_, vision_prefer_activation_type));
  }
  if (audio_executor_settings_.has_value()) {
    LITERT_RETURN_IF_ERROR(ResolveActivationDataType(
        *audio_executor_settings_, audio_prefer_activation_type));
  }
  return absl::OkStatus();
}

absl::Status EmbeddingEngineSettings::Validate(
    const std::optional<std::string>& text_backend_constraint,
    const std::optional<std::string>& vision_backend_constraint,
    const std::optional<std::string>& audio_backend_constraint) const {
  LITERT_RETURN_IF_ERROR(ValidateBackendConstraint(
      main_executor_settings_, text_backend_constraint, "Main"));
  if (vision_executor_settings_.has_value()) {
    LITERT_RETURN_IF_ERROR(ValidateBackendConstraint(
        *vision_executor_settings_, vision_backend_constraint, "Vision"));
  }
  if (audio_executor_settings_.has_value()) {
    LITERT_RETURN_IF_ERROR(ValidateBackendConstraint(
        *audio_executor_settings_, audio_backend_constraint, "Audio"));
  }

  LITERT_RETURN_IF_ERROR(
      ValidateCacheDir(main_executor_settings_.GetCacheDir()));
  if (vision_executor_settings_.has_value()) {
    LITERT_RETURN_IF_ERROR(
        ValidateCacheDir(vision_executor_settings_->GetCacheDir()));
  }
  if (audio_executor_settings_.has_value()) {
    LITERT_RETURN_IF_ERROR(
        ValidateCacheDir(audio_executor_settings_->GetCacheDir()));
  }

  return absl::OkStatus();
}

std::ostream& operator<<(std::ostream& os,
                         const EmbeddingEngineSettings& settings) {
  os << "EmbeddingEngineSettings: " << std::endl;
  os << settings.GetMainExecutorSettings() << std::endl;
  if (settings.GetVisionExecutorSettings().has_value()) {
    os << *settings.GetVisionExecutorSettings() << std::endl;
  }
  if (settings.GetAudioExecutorSettings().has_value()) {
    os << *settings.GetAudioExecutorSettings() << std::endl;
  }
  return os;
}

}  // namespace litert::lm
