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

#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "omni/base/io_types.h"
#include "omni/tts/stream_text_source.h"
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

// Configuration settings for TtsEngine initialization.
struct TtsEngineSettings {
  ModelType model_type = ModelType::UNSPECIFIED;
  // Folder containing the TTS model files.
  std::string model_folder;
  ::litert::lm::Backend backend = ::litert::lm::Backend::CPU;
  TextChunkConfig text_chunk_config;
  int num_threads = 4;
  // TODO b/538727793 introduce more settings for TtsEngine, and need to add
  // model type specific settings.
};

// High-level TTS Engine wrapping TtsSession and managing streaming text
// synthesis.
class TtsEngine {
 public:
  using AsyncCallback =
      absl::AnyInvocable<absl::Status(absl::StatusOr<AudioOutput>)>;

  // Factory method to create a TtsEngine instance from settings.
  static absl::StatusOr<std::unique_ptr<TtsEngine>> Create(
      const TtsEngineSettings& settings);

  ~TtsEngine() = default;

  // Resets session and text source state for a new synthesis stream.
  void Reset();

  // Marks input stream finished and flushes remaining synthesized audio.
  absl::StatusOr<AudioOutput> Flush();

  // Synchronously synthesizes input text as a whole chunk and forcefully
  // flushes all AudioOutput.
  absl::StatusOr<AudioOutput> Synthesize(absl::string_view text);

  // Asynchronously synthesizes input text using internal thread pool,
  // relying on text_chunk_config for chunk scheduling.
  absl::Status SynthesizeAsync(absl::string_view text, AsyncCallback callback);

  // TODO b/538727793: add WaitUntilDone to wait for async synthesis to finish,
  // so users don't handle the notification themselves.

  const TtsEngineSettings& settings() const { return settings_; }

 private:
  // Friend class for testing.
  friend struct TtsEngineTestingPeer;

  static absl::StatusOr<std::unique_ptr<TtsEngine>> CreateWithComponents(
      const TtsEngineSettings& settings, TtsSession::Components components);

  TtsEngine(const TtsEngineSettings& settings,
            std::unique_ptr<TtsSession> session,
            StreamTextSource* stream_text_source,
            std::unique_ptr<::litert::lm::ThreadPool> thread_pool);

  TtsEngineSettings settings_;
  std::unique_ptr<TtsSession> session_;
  StreamTextSource* stream_text_source_;  // Non-owning pointer into session_
  std::unique_ptr<::litert::lm::ThreadPool> thread_pool_;
};

}  // namespace litert::omni::tts

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_TTS_ENGINE_H_
