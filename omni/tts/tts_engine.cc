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

#include <algorithm>
#include <cstddef>
#include <filesystem>  // NOLINT
#include <memory>
#include <string>
#include <system_error>  // NOLINT: Required for std::error_code.
#include <utility>
#include <variant>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "omni/base/model_resources.h"
#include "omni/tts/kokoro/kokoro_factory.h"
#include "omni/tts/kokoro/kokoro_model_config.h"
#include "omni/tts/qwen3_tts/qwen3_tts_factory.h"
#include "omni/tts/qwen3_tts/qwen3_tts_model_config.h"
#include "omni/tts/tts_session.h"
#include "runtime/framework/threadpool.h"

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

}  // namespace

absl::StatusOr<ModelType> DetectModelType(absl::string_view model_folder) {
  if (model_folder.empty()) {
    return absl::InvalidArgumentError(
        "TtsEngineSettings::model_folder must not be empty.");
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
    const TtsEngineSettings& settings) {
  TtsEngineSettings resolved_settings = settings;
  if (resolved_settings.GetModelType() == ModelType::UNSPECIFIED) {
    LITERT_ASSIGN_OR_RETURN(const ModelType model_type,
                            DetectModelType(resolved_settings.model_folder));
    switch (model_type) {
      case ModelType::KOKORO:
        resolved_settings.model_config = KokoroModelConfig{};
        break;
      case ModelType::QWEN3_TTS:
        resolved_settings.model_config = Qwen3TtsModelConfig{};
        break;
      case ModelType::UNSPECIFIED:
        return absl::InvalidArgumentError(
            absl::StrCat("Unable to determine the TTS model type from ",
                         resolved_settings.model_folder));
    }
  }

  LITERT_ASSIGN_OR_RETURN(auto env, Environment::Create({}));
  auto shared_env = std::make_shared<Environment>(std::move(env));
  auto resources = std::make_shared<ModelResources>(shared_env);

  std::vector<std::string> available_voices;
  if (auto* config =
          std::get_if<KokoroModelConfig>(&resolved_settings.model_config)) {
    LITERT_RETURN_IF_ERROR(InitKokoroResources(
        *config, resolved_settings.model_folder, resolved_settings.cache_dir,
        resolved_settings.backend, resolved_settings.num_threads, *shared_env,
        *resources));
    available_voices = GetAvailableKokoroVoices(resolved_settings.model_folder);
  } else if (auto* config = std::get_if<Qwen3TtsModelConfig>(
                 &resolved_settings.model_config)) {
    LITERT_RETURN_IF_ERROR(InitQwen3TtsResources(
        *config, resolved_settings.model_folder, resolved_settings.cache_dir,
        resolved_settings.backend, resolved_settings.num_threads, *shared_env,
        *resources));
    if (!config->speaker_file.empty()) {
      available_voices.push_back(config->speaker_file);
    }
  } else {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported model_config in TtsEngineSettings: ",
                     static_cast<int>(resolved_settings.GetModelType())));
  }

  auto thread_pool = std::make_unique<lm::ThreadPool>(
      "tts_engine_pool", resolved_settings.num_threads);

  return std::unique_ptr<TtsEngine>(
      new TtsEngine(resolved_settings, std::move(available_voices), resources,
                    std::move(thread_pool)));
}

std::vector<std::string> TtsEngine::GetAvailableLanguages() const {
  if (settings_.GetModelType() == ModelType::KOKORO) {
    std::vector<std::string> languages;
    for (const auto& voice : available_voices_) {
      std::string kokoro_lang = GetKokoroVoiceLanguage(voice);
      std::string bcp47 = KokoroCodeToBcp47(kokoro_lang);
      if (!bcp47.empty()) {
        languages.push_back(std::move(bcp47));
      }
    }
    std::sort(languages.begin(), languages.end());
    languages.erase(std::unique(languages.begin(), languages.end()),
                    languages.end());
    return languages;
  }
  if (settings_.GetModelType() == ModelType::QWEN3_TTS) {
    return {"en-US", "zh-CN"};
  }
  return {};
}

std::vector<std::string> TtsEngine::GetAvailableVoices(
    absl::string_view language) const {
  if (language.empty()) {
    return available_voices_;
  }
  std::string target_code = ToKokoroLanguageCode(language);
  if (target_code.empty()) {
    return {};
  }
  std::vector<std::string> filtered;
  for (const auto& voice : available_voices_) {
    std::string voice_lang = GetKokoroVoiceLanguage(voice);
    if (voice_lang == target_code) {
      filtered.push_back(voice);
    }
  }
  return filtered;
}

bool TtsEngine::HasVoice(absl::string_view voice) const {
  if (voice.empty()) return false;
  absl::string_view voice_name = voice;
  size_t last_slash = voice_name.find_last_of("/\\");
  if (last_slash != absl::string_view::npos) {
    voice_name = voice_name.substr(last_slash + 1);
  }
  if (absl::EndsWith(voice_name, ".bin")) {
    voice_name = voice_name.substr(0, voice_name.size() - 4);
  }
  for (const auto& v : available_voices_) {
    if (v == voice || v == voice_name) {
      return true;
    }
  }
  return false;
}

std::string TtsEngine::GetDefaultVoice(absl::string_view language) const {
  if (settings_.GetModelType() == ModelType::KOKORO) {
    std::string lang_code = ToKokoroLanguageCode(language);
    std::string default_voice = GetDefaultKokoroVoice(lang_code);
    if (HasVoice(default_voice)) {
      return default_voice;
    }
    std::vector<std::string> matching = GetAvailableVoices(language);
    if (!matching.empty()) {
      return matching.front();
    }
    // No voice matches the requested language. Fall back to an arbitrary
    // available voice only when no language was requested; otherwise return the
    // language-specific default so that the caller can report the language as
    // unsupported instead of silently switching to another language.
    if (language.empty() && !available_voices_.empty()) {
      return available_voices_.front();
    }
    return default_voice;
  }
  if (settings_.GetModelType() == ModelType::QWEN3_TTS) {
    if (!available_voices_.empty()) {
      return available_voices_.front();
    }
    if (auto* config =
            std::get_if<Qwen3TtsModelConfig>(&settings_.model_config)) {
      return config->speaker_file;
    }
  }
  return "";
}

absl::StatusOr<std::unique_ptr<TtsSession>> TtsEngine::CreateSession(
    const TtsSessionConfig& session_config) {
  if (session_config.language.empty()) {
    return absl::InvalidArgumentError(
        "TtsSessionConfig::language must not be empty.");
  }
  const std::string kokoro_lang = ToKokoroLanguageCode(session_config.language);
  if (kokoro_lang.empty()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported language: '", session_config.language, "'"));
  }

  TtsSession::Components components;
  if (auto* config = std::get_if<KokoroModelConfig>(&settings_.model_config)) {
    KokoroModelConfig session_model_config = *config;
    std::string resolved_voice = session_config.voice.value_or("");

    if (!resolved_voice.empty()) {
      if (!HasVoice(resolved_voice) &&
          !std::filesystem::exists(resolved_voice)) {
        return absl::InvalidArgumentError(
            absl::StrCat("Voice '", resolved_voice,
                         "' is not available in TtsEngine. Call "
                         "GetAvailableVoices() to inspect available voices."));
      }
    } else {
      resolved_voice = GetDefaultVoice(session_config.language);
      if (!available_voices_.empty() && !HasVoice(resolved_voice)) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "No voice is available for language '%s' (%s). Call "
            "GetAvailableVoices() to inspect available voices.",
            session_config.language, kokoro_lang));
      }
    }

    std::string voice_lang = GetKokoroVoiceLanguage(resolved_voice);
    if (!voice_lang.empty() && voice_lang != kokoro_lang) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Voice '%s' (language: %s) conflicts with requested language '%s' "
          "(%s)",
          resolved_voice, voice_lang, session_config.language, kokoro_lang));
    }

    session_model_config.voice_name = resolved_voice;
    session_model_config.language = kokoro_lang;

    LITERT_ASSIGN_OR_RETURN(
        components, CreateKokoroComponents(
                        session_model_config, settings_.model_folder,
                        session_config.text_chunk_config, model_resources_));
  } else if (auto* config =
                 std::get_if<Qwen3TtsModelConfig>(&settings_.model_config)) {
    Qwen3TtsModelConfig session_model_config = *config;
    if (kokoro_lang == "cmn") {
      // TODO(b538727793): Support non-English languages for Qwen3-TTS.
      // Qwen3 TTS language code:
      // https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base/blob/main/config.json#L115-L126
      session_model_config.language = "zh-CN";
    } else {
      session_model_config.language = "en-US";
    }
    const std::string resolved_voice = session_config.voice.value_or("");
    if (!resolved_voice.empty()) {
      session_model_config.speaker_file = resolved_voice;
    }
    LITERT_ASSIGN_OR_RETURN(
        components, CreateQwen3TtsComponents(
                        session_model_config, settings_.model_folder,
                        session_config.text_chunk_config, model_resources_));
  } else {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported model_config in TtsEngineSettings: ",
                     static_cast<int>(settings_.GetModelType())));
  }

  return TtsSession::Create(std::move(components), thread_pool_.get());
}

}  // namespace litert::omni::tts
