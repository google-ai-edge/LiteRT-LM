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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_EMBEDDING_ENGINE_SETTINGS_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_EMBEDDING_ENGINE_SETTINGS_H_

#include <optional>
#include <ostream>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/executor/audio/audio_executor_settings.h"
#include "runtime/executor/embedding/embedding_executor_settings.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/vision/vision_executor_settings.h"
#include "runtime/proto/embedding_metadata.pb.h"
#include "runtime/proto/engine.pb.h"

namespace litert::lm {

// Settings used for initializing EmbeddingEngine.
// This class encapsulates the model-specific settings for the embedding text
// encoder, vision encoder, and audio encoder models.
class EmbeddingEngineSettings {
 public:
  // Creates a default EmbeddingEngineSettings with the given model assets and
  // specified backends.
  static absl::StatusOr<EmbeddingEngineSettings> CreateDefault(
      ModelAssets model_assets, Backend backend = Backend::CPU,
      std::optional<Backend> vision_backend = std::nullopt,
      std::optional<Backend> audio_backend = std::nullopt);

  // Maximum sequence length (in tokens) for text encoder signatures. If set,
  // EmbeddingEngine will automatically select signatures up to this length.
  std::optional<int> GetMaxInputLength() const;
  void SetMaxInputLength(std::optional<int> max_input_length);

  // Desired number of vision tokens generated per image. If set,
  // EmbeddingEngine will automatically select the smallest vision encoder
  // (and adapter) signatures and configure patch metadata accordingly.
  std::optional<int> GetVisionTokensPerImage() const;
  void SetVisionTokensPerImage(std::optional<int> vision_tokens_per_image);

  // Returns the EmbeddingExecutorSettings for the embedding model.
  const EmbeddingExecutorSettings& GetMainExecutorSettings() const;
  EmbeddingExecutorSettings& GetMutableMainExecutorSettings();

  // Returns the VisionExecutorSettings for the vision model.
  const std::optional<VisionExecutorSettings>& GetVisionExecutorSettings()
      const;
  std::optional<VisionExecutorSettings>& GetMutableVisionExecutorSettings();

  // Returns the AudioExecutorSettings for the audio model.
  const std::optional<AudioExecutorSettings>& GetAudioExecutorSettings() const;
  std::optional<AudioExecutorSettings>& GetMutableAudioExecutorSettings();

  // Benchmark parameters:
  // Returns true if the benchmark is enabled.
  bool IsBenchmarkEnabled() const;
  // Returns the benchmark parameters.
  const std::optional<proto::BenchmarkParams>& GetBenchmarkParams() const;
  // Returns the mutable benchmark parameters.
  proto::BenchmarkParams& GetMutableBenchmarkParams();

  // Returns the EmbeddingMetadata parameters if loaded.
  const std::optional<proto::EmbeddingMetadata>& GetEmbeddingMetadata() const;
  proto::EmbeddingMetadata& GetMutableEmbeddingMetadata();

 private:
  explicit EmbeddingEngineSettings(
      EmbeddingExecutorSettings embedding_executor_settings,
      std::optional<VisionExecutorSettings> vision_executor_settings,
      std::optional<AudioExecutorSettings> audio_executor_settings,
      std::optional<proto::BenchmarkParams> benchmark_params = std::nullopt);

  EmbeddingExecutorSettings main_executor_settings_;
  std::optional<VisionExecutorSettings> vision_executor_settings_;
  std::optional<AudioExecutorSettings> audio_executor_settings_;
  std::optional<proto::EmbeddingMetadata> metadata_;
  std::optional<proto::BenchmarkParams> benchmark_params_;
  std::optional<int> max_input_length_;
  std::optional<int> vision_tokens_per_image_;
};

std::ostream& operator<<(std::ostream& os,
                         const EmbeddingEngineSettings& settings);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_EMBEDDING_ENGINE_SETTINGS_H_
