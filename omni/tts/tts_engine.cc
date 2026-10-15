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

#include "omni/tts/tts_engine.h"

#include <filesystem>  // NOLINT
#include <memory>
#include <string>
#include <system_error>  // NOLINT: Required for std::error_code.
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_macros.h"  // from @litert
#include "omni/base/model_utils.h"
#include "omni/multi_staged_session.h"
#include "omni/omni_engine.h"
#include "omni/omni_registry.h"
#include "omni/omni_session.h"
#include "omni/tts/stream_text_source.h"
#include "omni/tts/text_chunk_utils.h"
#include "omni/tts/tts_session.h"
#include "runtime/components/model_resources.h"
#include "runtime/proto/tts_metadata.pb.h"
#include "runtime/proto/tts_model_type.pb.h"
#include "runtime/proto/tts_session_config.pb.h"

namespace litert::omni::tts {
namespace {

// Extension shared by all LiteRT model files.
constexpr absl::string_view kTfLiteExtension = ".tflite";
// Filename prefix of the Kokoro model files (e.g. "kokoro_acoustic.tflite").
constexpr absl::string_view kKokoroFilePrefix = "kokoro";
// Filename prefixes of the Qwen3-TTS model files (e.g. "talker_int4.tflite",
// "codec_decoder_fp32.tflite").
constexpr absl::string_view kQwen3TalkerFilePrefix = "talker";
constexpr absl::string_view kQwen3CodecFilePrefix = "codec_";

absl::StatusOr<ModelType> DetectModelTypeFromLitertLm(
    lm::ModelResources& lm_resources) {
  auto tts_metadata = lm_resources.GetTtsMetadata();
  if (tts_metadata.ok() && *tts_metadata != nullptr) {
    if ((*tts_metadata)->tts_model_type().has_kokoro()) {
      return ModelType::KOKORO;
    }
    if ((*tts_metadata)->tts_model_type().has_qwen3_tts()) {
      return ModelType::QWEN3_TTS;
    }
  }
  if (lm_resources
          .GetTFLiteModelBuffer(lm::proto::TtsMetadata::TF_LITE_ACOUSTIC)
          .ok()) {
    return ModelType::KOKORO;
  }
  if (lm_resources.GetTFLiteModelBuffer(lm::proto::TtsMetadata::TF_LITE_TALKER)
          .ok()) {
    return ModelType::QWEN3_TTS;
  }
  return ModelType::UNSPECIFIED;
}

}  // namespace

absl::StatusOr<ModelType> DetectModelType(absl::string_view model_folder) {
  if (model_folder.empty()) {
    return absl::InvalidArgumentError(
        "TTS model_path must not be empty when detecting model type.");
  }

  const bool is_litertlm_file =
      absl::EndsWithIgnoreCase(model_folder, ".litertlm");
  const std::string litertlm_path = is_litertlm_file
                                        ? std::string(model_folder)
                                        : ResolveLitertLmPath(model_folder);
  if (!litertlm_path.empty()) {
    LITERT_ASSIGN_OR_RETURN(auto lm_resources,
                            CreateLmModelResources(litertlm_path));
    LITERT_ASSIGN_OR_RETURN(const ModelType from_litertlm,
                            DetectModelTypeFromLitertLm(*lm_resources));
    if (from_litertlm != ModelType::UNSPECIFIED) {
      return from_litertlm;
    }
    if (is_litertlm_file) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Unable to determine the TTS model type from .litertlm file '",
          model_folder, "'."));
    }
  }

  std::error_code ec;
  std::filesystem::directory_iterator it(std::string(model_folder), ec);
  if (ec) {
    return absl::InvalidArgumentError(
        absl::StrCat("Failed to open TTS model folder '", model_folder,
                     "': ", ec.message()));
  }

  ModelType detected = ModelType::UNSPECIFIED;
  for (const std::filesystem::directory_iterator end; it != end;
       it.increment(ec)) {
    if (ec) {
      return absl::InvalidArgumentError(
          absl::StrCat("Failed to scan TTS model folder '", model_folder,
                       "': ", ec.message()));
    }
    if (it->path().extension().string() != kTfLiteExtension) {
      continue;
    }
    const std::string filename = it->path().filename().string();
    if (absl::StartsWith(filename, kKokoroFilePrefix)) {
      return ModelType::KOKORO;
    }
    if (absl::StartsWith(filename, kQwen3TalkerFilePrefix) ||
        absl::StartsWith(filename, kQwen3CodecFilePrefix)) {
      detected = ModelType::QWEN3_TTS;
    }
  }

  if (detected == ModelType::UNSPECIFIED) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Unable to determine the TTS model type from model folder '",
        model_folder,
        "': expected Kokoro (kokoro_*.tflite) or Qwen3-TTS (talker_*.tflite) "
        "model files."));
  }
  return detected;
}

absl::StatusOr<std::unique_ptr<TtsEngine>> TtsEngine::Create(
    absl::string_view model_name, Settings settings) {
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<OmniSessionFactory> session_factory,
                        OmniRegistry::Get().CreateSessionFactory(
                            model_name, settings.engine_settings,
                            &settings.session_config, &settings.metadata));
  if (session_factory == nullptr) {
    return absl::InternalError(
        "OmniRegistry returned null OmniSessionFactory.");
  }
  // NOTE: `static_cast<TtsSessionFactory*>` is acceptable here because all
  // built-in TTS factories inherit from `TtsSessionFactory`, and `TtsEngine`
  // is transitional and will be removed once all clients migrate to the
  // unified `OmniEngine` API.
  auto* tts_factory = static_cast<TtsSessionFactory*>(session_factory.get());
  settings.engine_settings = tts_factory->engine_settings();
  settings.metadata = tts_factory->metadata();
  settings.session_config = tts_factory->session_config();

  ABSL_ASSIGN_OR_RETURN(
      std::unique_ptr<OmniEngine> omni_engine,
      OmniEngine::Create(model_name, std::move(session_factory)));
  return std::unique_ptr<TtsEngine>(
      new TtsEngine(std::move(settings), std::move(omni_engine), tts_factory));
}

TtsEngine::TtsEngine(Settings settings,
                     std::unique_ptr<OmniEngine> absl_nonnull omni_engine,
                     TtsSessionFactory* absl_nonnull tts_session_factory)
    : settings_(std::move(settings)),
      omni_engine_(std::move(omni_engine)),
      tts_session_factory_(tts_session_factory) {}

TextChunkConfig TtsEngine::ResolveTextChunkConfig(
    const lm::proto::TtsSessionConfig& session_config) const {
  return tts_session_factory_->ResolveTextChunkConfig(session_config);
}

absl::StatusOr<std::unique_ptr<MultiStagedSession>> TtsEngine::CreateSession(
    const lm::proto::TtsSessionConfig& session_config,
    std::unique_ptr<StreamTextSource> absl_nullable text_source) {
  return tts_session_factory_->CreateWithTextSource(session_config,
                                                    std::move(text_source));
}

std::vector<std::string> TtsEngine::GetAvailableLanguages() const {
  return tts_session_factory_->GetAvailableLanguages();
}

std::vector<std::string> TtsEngine::GetAvailableVoices(
    absl::string_view language) const {
  return tts_session_factory_->GetAvailableVoices(language);
}

bool TtsEngine::HasVoice(absl::string_view voice) const {
  return tts_session_factory_->HasVoice(voice);
}

std::string TtsEngine::GetDefaultVoice(absl::string_view language) const {
  return tts_session_factory_->GetDefaultVoice(language);
}

}  // namespace litert::omni::tts
