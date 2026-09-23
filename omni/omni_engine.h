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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_OMNI_ENGINE_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_OMNI_ENGINE_H_

#include <memory>
#include <string>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "omni/omni_session.h"

namespace litert::omni {

// Unified engine creating `OmniSession` instances for the specified
// `model_name` with pre-defined configurations.
//
// Supported ASR model names (from `omni/asr/model_metadata.json`):
//   - "parakeet-tdt-0.6b-v3"
//   - "parakeet-ctc-0.6b"
//   - "moonshine-tiny"
//   - "whisper-tiny"
//   - "qwen3-asr-0.6b"
//   - "tinygemma-asr"
//
// Supported TTS model names / folders:
//   - "kokoro" / "kokoro-82m"
//   - "qwen3-tts" / "qwen3"
//   - Any directory path containing Kokoro (`kokoro_*.tflite`) or Qwen3-TTS
//     (`talker_*.tflite` / `codec_*.tflite`) model files.
class OmniEngine {
 public:
  // Common runtime options for `OmniEngine`.
  struct Options {
    enum class Backend {
      kCpu = 0,
      kGpu = 1,
      kNpu = 2,
    };

    Backend backend = Backend::kCpu;
    std::string cache_dir;
    int num_threads = 4;
  };

  // Creates an OmniEngine for the specified `model_name` using pre-defined
  // model and session configurations and optional runtime `options`.
  static absl::StatusOr<std::unique_ptr<OmniEngine>> Create(
      absl::string_view model_name, const Options& options);
  static absl::StatusOr<std::unique_ptr<OmniEngine>> Create(
      absl::string_view model_name) {
    return Create(model_name, Options{});
  }

  // Creates an OmniEngine for the specified `model_name` with a custom
  // `OmniSessionFactory`.
  static absl::StatusOr<std::unique_ptr<OmniEngine>> Create(
      absl::string_view model_name,
      std::unique_ptr<OmniSessionFactory> absl_nonnull session_factory);

  ~OmniEngine() = default;

  absl::string_view model_name() const { return model_name_; }

  // Creates a new OmniSession using the engine's `OmniSessionFactory`.
  absl::StatusOr<std::unique_ptr<OmniSession>> CreateSession();

 private:
  OmniEngine(std::string model_name,
             std::unique_ptr<OmniSessionFactory> absl_nonnull session_factory);

  std::string model_name_;
  std::unique_ptr<OmniSessionFactory> absl_nonnull session_factory_;
};

}  // namespace litert::omni

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_OMNI_ENGINE_H_
