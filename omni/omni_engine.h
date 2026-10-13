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

namespace internal {

// Common runtime settings for `OmniEngine`. Defined outside `OmniEngine` so
// `const Settings& settings = {}` can be used as a default argument in
// `OmniEngine::Create`.
struct OmniEngineSettings {
  enum class Backend {
    kCpu = 0,
    kGpu = 1,
    kNpu = 2,
  };

  Backend backend = Backend::kCpu;
  std::string cache_dir;
  int num_threads = 4;

  // Optional per-model asset paths and download URLs.
  std::string model_path;
  std::string model_remote_url;
  std::string npu_model_remote_url_pattern;
  std::string tokenizer_path;
  std::string tokenizer_remote_url;
};

}  // namespace internal

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
// Supported TTS model names / paths:
//   - "kokoro" / "kokoro-82m"
//   - "qwen3-tts" / "qwen3"
//   - Any directory of `.tflite` or `.litertlm` model files for Kokoro or
//     Qwen3-TTS.
//
// Supported Text2Image model names / paths:
//   - "bonsai-flux2" / "flux2-klein" / "flux.2-klein-4b"
//   - Any directory of `.tflite` or `.litertlm` model files (or `.litertlm`
//     file path) for Bonsai-FLUX.2.
class OmniEngine {
 public:
  using Settings = internal::OmniEngineSettings;

  // TODO: Remove Options alias once callers migrate to Settings.
  using Options = Settings;

  // Creates an OmniEngine for the specified `model_name` using pre-defined
  // model and session configurations and optional runtime `settings`.
  // TODO: Pass `session_config` and `metadata` when TTS and Text2Image move to
  // static registration.
  static absl::StatusOr<std::unique_ptr<OmniEngine>> Create(
      absl::string_view model_name, const Settings& settings = {});

  // Creates an OmniEngine for the specified `model_name` with a custom
  // `OmniSessionFactory`.
  // TODO: Remove when TTS and Text2Image move to static registration.
  static absl::StatusOr<std::unique_ptr<OmniEngine>> Create(
      absl::string_view model_name,
      std::unique_ptr<OmniSessionFactory> absl_nonnull factory);

  ~OmniEngine() = default;

  absl::string_view model_name() const { return model_name_; }

  // Creates a new OmniSession using the engine's `OmniSessionFactory`.
  absl::StatusOr<std::unique_ptr<OmniSession>> CreateSession(
      std::unique_ptr<OmniSession::InputSource> absl_nonnull input_source);

 private:
  OmniEngine(std::string model_name,
             std::unique_ptr<OmniSessionFactory> absl_nonnull session_factory);

  std::string model_name_;
  std::unique_ptr<OmniSessionFactory> absl_nonnull session_factory_;
};

}  // namespace litert::omni

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_OMNI_ENGINE_H_
