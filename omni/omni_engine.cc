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

#include "omni/omni_engine.h"

#include <filesystem>  // NOLINT
#include <memory>
#include <string>
#include <utility>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/ascii.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "omni/asr/asr_engine.h"
#include "omni/asr/asr_omni_session.h"
#include "omni/asr/model_metadata.h"
#include "omni/omni_session.h"
#include "omni/tts/kokoro/kokoro_model_config.h"
#include "omni/tts/qwen3_tts/qwen3_tts_model_config.h"
#include "omni/tts/tts_engine.h"
#include "omni/tts/tts_omni_session.h"
#include "runtime/executor/executor_settings_base.h"

namespace litert::omni {
namespace {

asr::AsrEngineConfig::Backend ToAsrBackend(
    OmniEngine::Options::Backend backend) {
  switch (backend) {
    case OmniEngine::Options::Backend::kGpu:
      return asr::AsrEngineConfig::Backend::kGpu;
    case OmniEngine::Options::Backend::kNpu:
      return asr::AsrEngineConfig::Backend::kNpu;
    case OmniEngine::Options::Backend::kCpu:
    default:
      return asr::AsrEngineConfig::Backend::kCpu;
  }
}

lm::Backend ToTtsBackend(OmniEngine::Options::Backend backend) {
  switch (backend) {
    case OmniEngine::Options::Backend::kGpu:
      return lm::Backend::GPU;
    case OmniEngine::Options::Backend::kNpu:
      return lm::Backend::NPU;
    case OmniEngine::Options::Backend::kCpu:
    default:
      return lm::Backend::CPU;
  }
}

void ApplyOptionsToAsrConfig(const OmniEngine::Options& options,
                             asr::AsrEngineConfig& asr_config) {
  asr_config.backend = ToAsrBackend(options.backend);
  if (!options.cache_dir.empty()) {
    asr_config.cache_dir = options.cache_dir;
  }
  if (options.num_threads > 0) {
    asr_config.num_threads = options.num_threads;
  }
  if (asr_config.model_path.empty() && !asr_config.model_url.empty()) {
    std::string ext =
        std::filesystem::path(asr_config.model_url).extension().string();
    if (ext.empty()) {
      ext = ".tflite";
    }
    asr_config.model_path = (std::filesystem::path(asr_config.cache_dir) /
                             absl::StrCat(asr_config.model_name, ext))
                                .string();
  }
  if (asr_config.tokenizer_path.empty() && !asr_config.tokenizer_url.empty()) {
    asr_config.tokenizer_path =
        (std::filesystem::path(asr_config.cache_dir) /
         absl::StrCat(asr_config.model_name, "_tokenizer.json"))
            .string();
  }
}

tts::TtsEngineSettings BuildTtsSettings(absl::string_view model_folder,
                                        tts::ModelConfig model_config,
                                        const OmniEngine::Options& options) {
  tts::TtsEngineSettings tts_settings;
  tts_settings.model_folder = std::string(model_folder);
  tts_settings.cache_dir = options.cache_dir;
  tts_settings.backend = ToTtsBackend(options.backend);
  if (options.num_threads > 0) {
    tts_settings.num_threads = options.num_threads;
  }
  tts_settings.model_config = std::move(model_config);
  return tts_settings;
}

// Instantiates an `OmniSessionFactory` for `model_name` and `options`.
// TODO(b/538727793): Determine whether static registry is the way to go or not.
absl::StatusOr<std::unique_ptr<OmniSessionFactory>> CreateSessionFactory(
    absl::string_view model_name, const OmniEngine::Options& options) {
  // 1. Check if `model_name` matches a known ASR model in
  // `model_metadata.json`.
  asr::AsrEngineConfig asr_config;
  if (asr::PopulateConfigFromMetadataJson(model_name, /*json_str=*/"",
                                          asr_config)
          .ok()) {
    ApplyOptionsToAsrConfig(options, asr_config);
    return asr::AsrOmniSessionFactory::CreateFactory(std::move(asr_config));
  }

  // 2. Check if `model_name` is a known TTS model name.
  const std::string lower_name = absl::AsciiStrToLower(model_name);
  const std::string model_folder =
      options.cache_dir.empty()
          ? std::string(model_name)
          : (std::filesystem::path(options.cache_dir) / std::string(model_name))
                .string();
  if (lower_name == "kokoro" || lower_name == "kokoro-82m") {
    return tts::TtsOmniSessionFactory::CreateFactory(
        BuildTtsSettings(model_folder, tts::KokoroModelConfig{}, options));
  }
  if (lower_name == "qwen3-tts" || lower_name == "qwen3") {
    return tts::TtsOmniSessionFactory::CreateFactory(
        BuildTtsSettings(model_folder, tts::Qwen3TtsModelConfig{}, options));
  }

  // 3. Check if `model_name` is a directory path containing recognizable TTS
  // model files.
  auto detected_tts = tts::DetectModelType(model_name);
  if (detected_tts.ok()) {
    tts::ModelConfig model_config;
    if (*detected_tts == tts::ModelType::KOKORO) {
      model_config = tts::KokoroModelConfig{};
    } else if (*detected_tts == tts::ModelType::QWEN3_TTS) {
      model_config = tts::Qwen3TtsModelConfig{};
    }
    return tts::TtsOmniSessionFactory::CreateFactory(
        BuildTtsSettings(model_name, std::move(model_config), options));
  }

  return absl::NotFoundError(
      absl::StrCat("Unsupported or unknown Omni model: ", model_name));
}

}  // namespace

OmniEngine::OmniEngine(
    std::string model_name,
    std::unique_ptr<OmniSessionFactory> absl_nonnull session_factory)
    : model_name_(std::move(model_name)),
      session_factory_(std::move(session_factory)) {}

absl::StatusOr<std::unique_ptr<OmniEngine>> OmniEngine::Create(
    absl::string_view model_name, const Options& options) {
  if (model_name.empty()) {
    return absl::InvalidArgumentError("model_name must not be empty.");
  }

  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<OmniSessionFactory> session_factory,
                        CreateSessionFactory(model_name, options));
  return Create(model_name, std::move(session_factory));
}

absl::StatusOr<std::unique_ptr<OmniEngine>> OmniEngine::Create(
    absl::string_view model_name,
    std::unique_ptr<OmniSessionFactory> absl_nonnull factory) {
  if (model_name.empty()) {
    return absl::InvalidArgumentError("model_name must not be empty.");
  }
  return std::unique_ptr<OmniEngine>(
      new OmniEngine(std::string(model_name), std::move(factory)));
}

absl::StatusOr<std::unique_ptr<OmniSession>> OmniEngine::CreateSession(
    std::unique_ptr<OmniSession::InputSource> absl_nonnull input_source) {
  return session_factory_->Create(std::move(input_source));
}

}  // namespace litert::omni
