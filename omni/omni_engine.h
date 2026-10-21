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
  // Directory for downloaded model files and compiled-model caches. Caches are
  // keyed by `<model_name>_<model file id>` so a replaced model file never
  // reuses stale ones:
  //   - CPU: XNNPack packed weights in `<key>.xnnpack_cache` (mmap'ed instead
  //     of held in the native heap).
  //   - GPU: compiled programs and GPU-layout weights in
  //     `<key>_mldrift_{program,weight}_cache.bin` (skips shader compilation
  //     and weight conversion).
  // This lowers memory and load time at the cost of disk space.
  //   - Empty: as elsewhere in LiteRT-LM, caches are written next to the model
  //     file (skipped if that directory is not writable); model files are
  //     resolved and downloaded under `/tmp/omni_models` unless `model_path`
  //     is set.
  //   - ":nocache": no caches; model files are resolved as for empty.
  //   - Otherwise: model files and caches live here. The directory is created
  //     if missing. If it is not writable, caches are skipped when the model
  //     file is in this directory (e.g. read-only pre-installed models);
  //     otherwise model creation fails with `PermissionDeniedError`.
  std::string cache_dir;
  int num_threads = 4;
  // If true, forces FP32 precision on GPU instead of the default FP16.
  bool force_fp32_on_gpu = false;

  // Optional per-model asset paths and download URLs.
  std::string model_path;
  std::string model_remote_url;
  std::string npu_model_remote_url_pattern;
  std::string tokenizer_path;
  std::string tokenizer_remote_url;
};

}  // namespace internal

// Unified engine creating `OmniSession` instances for the specified
// `model_name` (or a path containing a registered model name) with registered
// model and session configurations.
//
// Supported ASR model names (registered in `omni/speech2text/`):
//   - "parakeet-tdt-0.6b-v3"
//   - "parakeet-ctc-0.6b"
//   - "moonshine-tiny"
//   - "whisper-tiny"
//   - "qwen3-asr-0.6b"
//   - "tinygemma-asr"
//
// Supported TTS model names (registered in `omni/text2speech/`):
//   - "kokoro" / "kokoro-82m"
//   - "qwen3-tts" / "qwen3"
//
// Supported Text2Image model names (registered in `omni/text2image/`):
//   - "bonsai-flux2" / "flux2" / "flux2-klein" / "flux.2-klein-4b"
class OmniEngine {
 public:
  using Settings = internal::OmniEngineSettings;

  // Creates an OmniEngine for the specified `model_name` using registered
  // model and session configurations and optional runtime `settings`,
  // `session_config`, and `metadata`.
  static absl::StatusOr<std::unique_ptr<OmniEngine>> Create(
      absl::string_view model_name, const Settings& settings = {},
      const MessageLite* absl_nullable session_config = nullptr,
      const MessageLite* absl_nullable metadata = nullptr);

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
