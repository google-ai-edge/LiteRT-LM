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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_TTS_ENGINE_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_TTS_ENGINE_H_

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "omni/base/io_types.h"
#include "omni/base/model_resources.h"
#include "omni/tts/kokoro/kokoro_model_config.h"
#include "omni/tts/qwen3_tts/qwen3_tts_model_config.h"
#include "omni/tts/text_chunk_utils.h"
#include "omni/tts/tts_session.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/framework/threadpool.h"

namespace litert::omni::tts {

// Supported TTS model types.
enum class ModelType {
  UNSPECIFIED = 0,
  // Kokoro-82M.
  KOKORO = 1,
  // Qwen3-TTS.
  QWEN3_TTS = 2,
};

// Model-specific configuration variant.
using ModelConfig =
    std::variant<std::monostate, KokoroModelConfig, Qwen3TtsModelConfig>;

// Determines the TTS model architecture from the model files found under
// `model_folder` (e.g. "kokoro_acoustic.tflite" -> ModelType::KOKORO,
// "talker_int4.tflite" -> ModelType::QWEN3_TTS). Returns an error if the model
// folder cannot be read or holds no recognizable TTS model.
//
// TODO(b/538727793): Remove the empirical detection once the model type is
// embedded in the model file metadata.
absl::StatusOr<ModelType> DetectModelType(absl::string_view model_folder);

// Configuration settings for a TtsSession instance.
struct TtsSessionConfig {
  // Target language for speech synthesis in BCP-47 format (e.g., "en-US",
  // "en-GB", "es", "zh-CN", "hi", "ja", "fr", "it", "pt-BR"). Defaults to
  // American English ("en-US").
  // Available languages of the engine can be fetched via
  // `TtsEngine::GetAvailableLanguages()`.
  std::string language = "en-US";

  // Optional voice profile name (e.g., "af_heart", "ef_dora", "hf_alpha")
  // or voice file path. If unset, uses the engine's default voice for
  // `language`.
  // Available voices of the engine can be fetched via
  // `TtsEngine::GetAvailableVoices()`.
  std::optional<std::string> voice;

  // Text chunk configuration.
  TextChunkConfig text_chunk_config;
};

// Configuration settings for TtsEngine initialization.
struct TtsEngineSettings {
  // Folder containing the TTS model files.
  std::string model_folder;
  // Optional cache directory for model acceleration (e.g., XNNPack weight
  // cache).
  std::string cache_dir;
  // Backend to use for model execution.
  lm::Backend backend = lm::Backend::CPU;
  // Number of threads to use for model execution (CPU only).
  int num_threads = 4;

  // Model-specific configuration. If left unset (std::monostate), the model
  // type is detected from `model_folder` in TtsEngine::Create and the default
  // configuration of the detected model is used.
  ModelConfig model_config;

  ModelType GetModelType() const {
    if (std::holds_alternative<KokoroModelConfig>(model_config)) {
      return ModelType::KOKORO;
    }
    if (std::holds_alternative<Qwen3TtsModelConfig>(model_config)) {
      return ModelType::QWEN3_TTS;
    }
    return ModelType::UNSPECIFIED;
  }
};

// High-level TTS Engine owning heavy model resources and creating lightweight
// TtsSession instances for streaming text synthesis.
class TtsEngine {
 public:
  using AsyncCallback =
      absl::AnyInvocable<absl::Status(absl::StatusOr<AudioOutput>)>;

  // Factory method to create a TtsEngine instance from settings.
  static absl::StatusOr<std::unique_ptr<TtsEngine>> Create(
      const TtsEngineSettings& settings);

  ~TtsEngine() = default;

  // Resolves the `TextChunkConfig` for `session_config`, applying any
  // model-specific revisions (such as Kokoro delimiter/buffer size defaults).
  TextChunkConfig ResolveTextChunkConfig(
      const TtsSessionConfig& session_config = {}) const;

  // Creates a lightweight TtsSession for a synthesis stream.
  // If `text_source` is null, a `StreamTextSource` is created using
  // `ResolveTextChunkConfig(session_config)`. If a custom `text_source` is
  // provided, its `TextChunkConfig` is used as-is (callers can use
  // `ResolveTextChunkConfig()` when constructing `text_source`).
  absl::StatusOr<std::unique_ptr<TtsSession>> CreateSession(
      const TtsSessionConfig& session_config = {},
      std::unique_ptr<StreamTextSource> absl_nullable text_source = nullptr);

  // Returns the list of available languages in BCP-47 format (e.g., "en-US",
  // "es", "zh-CN") supported by the engine and its available voices.
  std::vector<std::string> GetAvailableLanguages() const;

  // Returns the list of available voice profile names (e.g., "af_heart",
  // "ef_dora"). If `language` is non-empty (e.g. "es", "en-US"), filters to
  // return only voices compatible with that language.
  std::vector<std::string> GetAvailableVoices(
      absl::string_view language = "") const;

  // Checks whether the specified voice is available in the engine.
  bool HasVoice(absl::string_view voice) const;

  // Returns the default voice profile name for the specified language (in
  // BCP-47 format, e.g. "es", "en-US"). Defaults to "en-US".
  std::string GetDefaultVoice(absl::string_view language = "en-US") const;

  const TtsEngineSettings& settings() const { return settings_; }
  std::shared_ptr<ModelResources> model_resources() const {
    return model_resources_;
  }

 private:
  // Friend class for testing.
  friend struct TtsEngineTestingPeer;

  TtsEngine(const TtsEngineSettings& settings,
            std::vector<std::string> available_voices,
            std::shared_ptr<ModelResources> resources,
            std::unique_ptr<lm::ThreadPool> thread_pool)
      : settings_(settings),
        available_voices_(std::move(available_voices)),
        model_resources_(std::move(resources)),
        thread_pool_(std::move(thread_pool)) {}

  TtsEngineSettings settings_;
  std::vector<std::string> available_voices_;
  std::shared_ptr<ModelResources> model_resources_;
  std::unique_ptr<lm::ThreadPool> thread_pool_;
};

}  // namespace litert::omni::tts

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_TTS_ENGINE_H_
