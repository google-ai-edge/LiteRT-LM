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
#include <string>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
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

  // Updates the settings from the embedding metadata, resolves default values,
  // and validates the result. This is the single entry point that both the
  // resources-based and the streamed creation paths use, mirroring
  // EngineSettings::MaybeUpdateAndValidate on the LLM side.
  //
  // `metadata_from_file` is adopted only when the caller has not already
  // supplied metadata, so metadata set in code keeps taking precedence over the
  // model's. Passing nullptr is fine, and leaves any existing metadata alone.
  //
  // Safe to call more than once. The streamed path relies on that, calling it
  // again as each section arrives, because it has to resolve settings before
  // the text encoder is compiled mid-stream.
  absl::Status MaybeUpdateAndValidate(
      const proto::EmbeddingMetadata* absl_nullable metadata_from_file,
      const std::optional<std::string>& text_backend_constraint = std::nullopt,
      const std::optional<std::string>& vision_backend_constraint =
          std::nullopt,
      const std::optional<std::string>& audio_backend_constraint = std::nullopt,
      const std::optional<std::string>& text_prefer_activation_type =
          std::nullopt,
      const std::optional<std::string>& vision_prefer_activation_type =
          std::nullopt,
      const std::optional<std::string>& audio_prefer_activation_type =
          std::nullopt);

  // Maximum sequence length (in tokens) for text encoder signatures. If set,
  // EmbeddingEngine will automatically select signatures up to this length.
  std::optional<int> GetMaxInputLength() const;
  void SetMaxInputLength(std::optional<int> max_input_length);

  // Minimum sequence length (in tokens) for text encoder signatures. If set,
  // EmbeddingEngine will exclude signatures shorter than this length.
  std::optional<int> GetMinInputLength() const;
  void SetMinInputLength(std::optional<int> min_input_length);

  // Upper bound on how many text encoder signatures EmbeddingEngine prepares.
  // Unset loads every signature the input length bounds allow.
  //
  // Each signature is a full private copy of the graph, so a bundle shipping
  // one variant per supported sequence length costs that multiple in subgraphs
  // and tensors. On wasm32 that is enough to exhaust the 4GB address space
  // during compilation, which is why the web bindings cap this by default.
  //
  // The longest signature is always among those prepared, so capping never
  // reduces the input length the engine accepts. Inputs that would have fit a
  // dropped signature are padded up to the next one that survived.
  std::optional<int> GetMaxNumSignatures() const;
  void SetMaxNumSignatures(std::optional<int> max_num_signatures);

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

  // Resolves default values and metadata preferences across all executor
  // settings based on the precedence waterfall:
  // 1. Settings explicitly set by user in code (highest precedence).
  // 2. prefer_activation_type from model metadata / TOML (supports "fp32_fp16"
  //    for mixed precision).
  // 3. Fallback to FLOAT16 if the backend is GPU.
  absl::Status ResolveDefaults(
      const std::optional<std::string>& text_prefer_activation_type =
          std::nullopt,
      const std::optional<std::string>& vision_prefer_activation_type =
          std::nullopt,
      const std::optional<std::string>& audio_prefer_activation_type =
          std::nullopt);

  // Validates the engine settings to ensure cache directories and backend
  // constraints are valid. Returns an error if validation fails.
  absl::Status Validate(
      const std::optional<std::string>& text_backend_constraint = std::nullopt,
      const std::optional<std::string>& vision_backend_constraint =
          std::nullopt,
      const std::optional<std::string>& audio_backend_constraint =
          std::nullopt) const;

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
  std::optional<int> min_input_length_;
  std::optional<int> max_num_signatures_;
  std::optional<int> vision_tokens_per_image_;
};

std::ostream& operator<<(std::ostream& os,
                         const EmbeddingEngineSettings& settings);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_EMBEDDING_ENGINE_SETTINGS_H_
