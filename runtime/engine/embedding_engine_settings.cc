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

#include <optional>
#include <ostream>
#include <utility>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/executor/audio/audio_executor_settings.h"
#include "runtime/executor/embedding/embedding_executor_settings.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/vision/vision_executor_settings.h"
#include "runtime/proto/embedding_metadata.pb.h"
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
