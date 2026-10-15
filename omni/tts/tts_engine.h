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
#include <string>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "omni/base/io_types.h"
#include "omni/base/model_resources.h"
#include "omni/multi_staged_session.h"
#include "omni/omni_engine.h"
#include "omni/tts/stream_text_source.h"
#include "omni/tts/text_chunk_utils.h"
#include "omni/tts/tts_session.h"
#include "runtime/proto/tts_metadata.pb.h"
#include "runtime/proto/tts_session_config.pb.h"

namespace litert::omni::tts {

// Supported TTS model types.
enum class ModelType {
  UNSPECIFIED = 0,
  // Kokoro-82M.
  KOKORO = 1,
  // Qwen3-TTS.
  QWEN3_TTS = 2,
};

// Determines the TTS model architecture from the model files found under
// `model_folder` (e.g. "kokoro_acoustic.tflite" -> ModelType::KOKORO,
// "talker_int4.tflite" -> ModelType::QWEN3_TTS). Returns an error if the model
// folder cannot be read or holds no recognizable TTS model.
//
// TODO(b/538727793): Remove the empirical detection once the model type is
// embedded in the model file metadata.
absl::StatusOr<ModelType> DetectModelType(absl::string_view model_folder);

// Thin Text-to-Speech (TTS) wrapper around `OmniEngine`.
// TODO(b/524681030): Remove TtsEngine once all callers migrate to OmniEngine.
class TtsEngine {
 public:
  using AsyncCallback =
      absl::AnyInvocable<absl::Status(absl::StatusOr<AudioOutput>)>;

  struct Settings {
    OmniEngine::Settings engine_settings;
    lm::proto::TtsSessionConfig session_config;
    lm::proto::TtsMetadata metadata;
  };

  // Factory method to create a TtsEngine instance from model name and settings.
  static absl::StatusOr<std::unique_ptr<TtsEngine>> Create(
      absl::string_view model_name, Settings settings = {});

  ~TtsEngine() = default;

  // Resolves the `TextChunkConfig` for `session_config`, applying any
  // model-specific revisions (such as Kokoro delimiter/buffer size defaults).
  TextChunkConfig ResolveTextChunkConfig(
      const lm::proto::TtsSessionConfig& session_config = {}) const;

  // Creates a lightweight MultiStagedSession for a synthesis stream.
  // The first stage (`stages()[0]`) is guaranteed to be a `StreamTextSource`.
  // If `text_source` is null, a `StreamTextSource` is created using
  // `ResolveTextChunkConfig(session_config)`. If a custom `text_source` is
  // provided, its `TextChunkConfig` is used as-is (callers can use
  // `ResolveTextChunkConfig()` when constructing `text_source`).
  absl::StatusOr<std::unique_ptr<MultiStagedSession>> CreateSession(
      const lm::proto::TtsSessionConfig& session_config = {},
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

  const Settings& settings() const { return settings_; }
  std::shared_ptr<ModelResources> model_resources() const {
    return tts_session_factory_->model_resources();
  }

 private:
  TtsEngine(Settings settings,
            std::unique_ptr<OmniEngine> absl_nonnull omni_engine,
            TtsSessionFactory* absl_nonnull tts_session_factory);

  Settings settings_;
  std::unique_ptr<OmniEngine> absl_nonnull omni_engine_;
  TtsSessionFactory* absl_nonnull tts_session_factory_;
};

}  // namespace litert::omni::tts

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_TTS_ENGINE_H_
