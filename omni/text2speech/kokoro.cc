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

#include "omni/text2speech/kokoro.h"

#include <algorithm>
#include <filesystem>  // NOLINT
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_options.h"  // from @litert
#include "omni/base/io_types.h"
#include "omni/base/model_resources.h"
#include "omni/base/model_utils.h"
#include "omni/base/stage.h"
#include "omni/multi_staged_session.h"
#include "omni/omni_engine.h"
#include "omni/omni_registry.h"
#include "omni/omni_session.h"
#include "omni/text2speech/kokoro/kokoro_factory.h"
#include "omni/text2speech/kokoro/kokoro_model_config.h"
#include "omni/text2speech/stream_text_source.h"
#include "omni/text2speech/text_chunk_utils.h"
#include "omni/text2speech/tts_session.h"
#include "runtime/components/model_resources.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/proto/text2speech_metadata.pb.h"
#include "runtime/proto/text2speech_model_type.pb.h"
#include "runtime/proto/text2speech_session_config.pb.h"

namespace litert::omni::tts {
namespace {

constexpr absl::string_view kModelNames[] = {
    "kokoro",
    "kokoro-82m",
};

lm::proto::Text2SpeechMetadata ResolveMetadata(
    const MessageLite* absl_nullable metadata) {
  lm::proto::Text2SpeechMetadata resolved;
  if (metadata != nullptr) {
    resolved.CheckTypeAndMergeFrom(*metadata);
  }
  resolved.mutable_tts_model_type()->mutable_kokoro();
  return resolved;
}

absl::Status PopulateKokoroConfigFromProto(const lm::proto::Kokoro& proto,
                                           KokoroModelConfig& config) {
  if (proto.has_voice_name() && !proto.voice_name().empty()) {
    config.voice_name = proto.voice_name();
  }
  if (proto.has_language() && !proto.language().empty()) {
    config.language = proto.language();
  }
  if (proto.has_target_bucket()) {
    config.target_bucket = proto.target_bucket();
  }
  if (proto.has_vocoder_backend() && !proto.vocoder_backend().empty()) {
    LITERT_ASSIGN_OR_RETURN(config.vocoder_backend,
                            lm::GetBackendFromString(proto.vocoder_backend()));
  }
  return absl::OkStatus();
}

}  // namespace

KokoroSessionFactory::KokoroSessionFactory(
    const OmniEngine::Settings& settings,
    const MessageLite* absl_nullable session_config,
    const lm::proto::Text2SpeechMetadata& metadata,
    std::vector<std::string> available_voices,
    std::shared_ptr<ModelResources> absl_nullable model_resources,
    KokoroModelConfig config)
    : TtsSessionFactory(settings, session_config, metadata,
                        std::move(available_voices),
                        std::move(model_resources)),
      config_(std::move(config)) {
  if (metadata.tts_model_type().has_kokoro()) {
    (void)PopulateKokoroConfigFromProto(metadata.tts_model_type().kokoro(),
                                        config_);
  }
}

std::vector<std::string> KokoroSessionFactory::GetAvailableLanguages() const {
  std::vector<std::string> languages;
  for (const auto& voice : available_voices()) {
    std::string kokoro_lang = GetKokoroVoiceLanguage(voice);
    std::string bcp47 = KokoroCodeToBcp47(kokoro_lang);
    if (!bcp47.empty()) {
      languages.push_back(std::move(bcp47));
    }
  }
  if (!languages.empty()) {
    std::sort(languages.begin(), languages.end());
    languages.erase(std::unique(languages.begin(), languages.end()),
                    languages.end());
    return languages;
  }
  return TtsSessionFactory::GetAvailableLanguages();
}

std::vector<std::string> KokoroSessionFactory::GetAvailableVoices(
    absl::string_view language) const {
  if (language.empty()) {
    return available_voices();
  }
  std::string target_code = ToKokoroLanguageCode(language);
  if (target_code.empty()) {
    return {};
  }
  std::vector<std::string> filtered;
  for (const auto& voice : available_voices()) {
    if (GetKokoroVoiceLanguage(voice) == target_code) {
      filtered.push_back(voice);
    }
  }
  return filtered;
}

std::string KokoroSessionFactory::GetDefaultVoice(
    absl::string_view language) const {
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
  if (language.empty() && !available_voices().empty()) {
    return available_voices().front();
  }
  return default_voice;
}

TextChunkConfig KokoroSessionFactory::ResolveTextChunkConfig(
    const lm::proto::Text2SpeechSessionConfig& session_config) const {
  lm::proto::Text2SpeechSessionConfig effective =
      MergeSessionConfig(session_config);
  KokoroModelConfig session_model_config = config_;
  const std::string kokoro_lang = ToKokoroLanguageCode(effective.language());
  if (!kokoro_lang.empty()) {
    session_model_config.language = kokoro_lang;
  }
  return ReviseTextChunkConfigForKokoro(
      session_model_config, ToTextChunkConfig(effective.text_chunk_config()));
}

absl::StatusOr<std::unique_ptr<MultiStagedSession>>
KokoroSessionFactory::CreateWithTextSource(
    const lm::proto::Text2SpeechSessionConfig& session_config,
    std::unique_ptr<StreamTextSource> absl_nullable text_source) {
  lm::proto::Text2SpeechSessionConfig effective =
      MergeSessionConfig(session_config);
  if (effective.language().empty()) {
    return absl::InvalidArgumentError(
        "Text2SpeechSessionConfig::language must not be empty.");
  }
  const std::string kokoro_lang = ToKokoroLanguageCode(effective.language());
  if (kokoro_lang.empty()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported language: '", effective.language(), "'"));
  }

  if (text_source == nullptr) {
    text_source =
        std::make_unique<StreamTextSource>(ResolveTextChunkConfig(effective));
  }

  KokoroModelConfig session_model_config = config_;
  std::string resolved_voice = effective.voice();
  if (!resolved_voice.empty()) {
    if (!HasVoice(resolved_voice) && !std::filesystem::exists(resolved_voice)) {
      return absl::InvalidArgumentError(
          absl::StrCat("Voice '", resolved_voice,
                       "' is not available in TtsEngine. Call "
                       "GetAvailableVoices() to inspect available voices."));
    }
  } else {
    resolved_voice = GetDefaultVoice(effective.language());
    if (!available_voices().empty() && !HasVoice(resolved_voice)) {
      return absl::InvalidArgumentError(
          absl::StrFormat("No voice is available for language '%s' (%s). Call "
                          "GetAvailableVoices() to inspect available voices.",
                          effective.language(), kokoro_lang));
    }
  }

  std::string voice_lang = GetKokoroVoiceLanguage(resolved_voice);
  if (!voice_lang.empty() && voice_lang != kokoro_lang) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Voice '%s' (language: %s) conflicts with requested language '%s' "
        "(%s)",
        resolved_voice, voice_lang, effective.language(), kokoro_lang));
  }

  if (model_resources() == nullptr) {
    return absl::FailedPreconditionError("ModelResources must not be null.");
  }

  session_model_config.voice_name = resolved_voice;
  session_model_config.language = kokoro_lang;

  std::vector<std::unique_ptr<internal::StageBase>> stages;
  Stage<Output>* output_stage = nullptr;
  LITERT_RETURN_IF_ERROR(CreateKokoroComponents(
      session_model_config, engine_settings().model_path,
      std::move(text_source), model_resources(), stages, &output_stage));
  if (output_stage == nullptr) {
    return absl::InternalError("Output stage must not be null.");
  }
  return MultiStagedSession::Create(std::move(stages), output_stage,
                                    thread_pool());
}

namespace {

absl::StatusOr<std::unique_ptr<OmniSessionFactory>> GetKokoroSessionFactory(
    absl::string_view model_name, const OmniEngine::Settings& settings,
    const MessageLite* absl_nullable session_config,
    const MessageLite* absl_nullable metadata, ::litert::Environment& env,
    ::litert::Options& /*options*/,
    std::shared_ptr<lm::ModelResources> absl_nullable lm_resources) {
  if (!absl::c_linear_search(kModelNames, model_name)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Unexpected model_name: ", model_name));
  }
  lm::proto::Text2SpeechMetadata resolved_metadata = ResolveMetadata(metadata);
  KokoroModelConfig config;
  config.force_fp32_on_gpu = settings.force_fp32_on_gpu;
  LITERT_RETURN_IF_ERROR(PopulateKokoroConfigFromProto(
      resolved_metadata.tts_model_type().kokoro(), config));

  auto shared_env = std::make_shared<Environment>(std::move(env));
  auto resources = std::make_shared<ModelResources>(shared_env);
  if (lm_resources == nullptr && !settings.model_path.empty()) {
    const std::string litertlm_path =
        absl::EndsWithIgnoreCase(settings.model_path, ".litertlm")
            ? settings.model_path
            : ResolveLitertLmPath(settings.model_path);
    if (!litertlm_path.empty()) {
      LITERT_ASSIGN_OR_RETURN(lm_resources,
                              CreateLmModelResources(litertlm_path));
    }
  }
  if (lm_resources != nullptr) {
    resources->SetLmModelResources(lm_resources);
  }

  const int num_threads = settings.num_threads > 0 ? settings.num_threads : 4;
  LITERT_RETURN_IF_ERROR(InitKokoroResources(
      config, settings.model_path, settings.cache_dir,
      ToLmBackend(settings.backend), num_threads, *shared_env, *resources));
  std::vector<std::string> available_voices =
      GetAvailableKokoroVoices(settings.model_path, lm_resources.get());

  return std::make_unique<KokoroSessionFactory>(
      settings, session_config, resolved_metadata, std::move(available_voices),
      std::move(resources), std::move(config));
}

[[maybe_unused]] const bool kKokoroRegistered = [] {
  ABSL_CHECK_OK(OmniRegistry::Get().RegisterSessionFactory(RegistrySpec{
      .model_names = {std::begin(kModelNames), std::end(kModelNames)},
      .get_session_factory = &GetKokoroSessionFactory,
      .session_config_type_id = "Text2Speech",
      .model_remote_url =
          "https://huggingface.co/litert-community/Kokoro-82M/resolve/main/"
          "kokoro_int8.litertlm",
  }));
  return true;
}();

}  // namespace
}  // namespace litert::omni::tts
