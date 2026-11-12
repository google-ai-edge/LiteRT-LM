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

#include "omni/speech2text/whisper.h"

#include <array>
#include <cstddef>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/ascii.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/strings/strip.h"  // from @com_google_absl
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_options.h"  // from @litert
#include "litert/cc/options/litert_gpu_options.h"  // from @litert
#include "omni/base/litert_runner.h"
#include "omni/base/model_utils.h"
#include "omni/multi_staged_session.h"
#include "omni/omni_engine.h"
#include "omni/omni_registry.h"
#include "omni/omni_session.h"
#include "omni/speech2text/asr_session.h"
#include "omni/speech2text/audio_source.h"
#include "omni/speech2text/whisper_decoder.h"
#include "runtime/components/model_resources.h"
#include "runtime/proto/speech2text_metadata.pb.h"
#include "runtime/proto/speech2text_model_type.pb.h"
#include "runtime/proto/speech2text_session_config.pb.h"
#include "support/tokenizer/tokenizer.h"

namespace litert::omni::asr {
namespace {

constexpr absl::string_view kWhisperTinyModelNames[] = {
    "whisper-tiny",
    "whisper-tiny-int8",
    "whisper",
    "whisper-int8",
};
constexpr absl::string_view kWhisperBaseModelNames[] = {
    "whisper-base",
    "whisper-base-int8",
};
constexpr absl::string_view kWhisperSmallModelNames[] = {
    "whisper-small",
    "whisper-small-int8",
};
constexpr absl::string_view kWhisperMediumModelNames[] = {
    "whisper-medium",
    "whisper-medium-int8",
};
constexpr absl::string_view kAllModelNames[] = {
    "whisper-tiny",      "whisper-tiny-int8",   "whisper-base",
    "whisper-base-int8", "whisper-small",       "whisper-small-int8",
    "whisper-medium",    "whisper-medium-int8", "whisper",
    "whisper-int8",
};

constexpr int kWhisperEndOfTextTokenId = 50257;
constexpr int kWhisperStartOfTranscriptTokenId = 50258;
constexpr int kWhisperDefaultLanguageTokenId = 50259;  // <|en|>
constexpr int kWhisperTranscribeTokenId = 50359;
constexpr int kWhisperNoTimestampsTokenId = 50363;

// Ordered 99 shared Whisper language codes corresponding to token IDs
// 50259..50357 across whisper-tiny/base/small/medium.
constexpr absl::string_view kWhisperLanguageCodes[] = {
    "en", "zh", "de", "es", "ru", "ko",  "fr", "ja", "pt", "tr", "pl",
    "ca", "nl", "ar", "sv", "it", "id",  "hi", "fi", "vi", "he", "uk",
    "el", "ms", "cs", "ro", "da", "hu",  "ta", "no", "th", "ur", "hr",
    "bg", "lt", "la", "mi", "ml", "cy",  "sk", "te", "fa", "lv", "bn",
    "sr", "az", "sl", "kn", "et", "mk",  "br", "eu", "is", "hy", "ne",
    "mn", "bs", "kk", "sq", "sw", "gl",  "mr", "pa", "si", "km", "sn",
    "yo", "so", "af", "oc", "ka", "be",  "tg", "sd", "gu", "am", "yi",
    "lo", "uz", "fo", "ht", "ps", "tk",  "nn", "mt", "sa", "lb", "my",
    "bo", "tl", "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su",
};

std::string NormalizeLanguageCode(absl::string_view language) {
  language = absl::StripAsciiWhitespace(language);
  absl::ConsumePrefix(&language, "<|");
  absl::ConsumeSuffix(&language, "|>");
  language = absl::StripAsciiWhitespace(language);
  std::string normalized = absl::AsciiStrToLower(language);
  const size_t sep = normalized.find_first_of("-_");
  if (sep != std::string::npos) {
    normalized.resize(sep);
  }
  if (normalized == "cmn" || normalized == "cn") {
    return "zh";
  }
  if (normalized == "jp") {
    return "ja";
  }
  if (normalized == "kr") {
    return "ko";
  }
  if (normalized == "iw") {
    return "he";
  }
  if (normalized == "in") {
    return "id";
  }
  if (normalized == "ji") {
    return "yi";
  }
  if (normalized == "nb") {
    return "no";
  }
  if (normalized == "jv") {
    return "jw";
  }
  if (normalized == "fil") {
    return "tl";
  }
  return normalized;
}

// Resolves the 4 Whisper decoder prefix token IDs
// `[<|startoftranscript|>, <|lang|>, <|transcribe|>, <|notimestamps|>]` for
// `language` (defaulting to `"en"` when `language` is empty). Queries
// `tokenizer` for all four tokens when `tokenizer` carries Whisper control
// tokens, and falls back to the 99 shared Whisper language table otherwise.
absl::StatusOr<std::array<int, 4>> ResolveWhisperStartTokenIds(
    absl::string_view language,
    ::litert::support::Tokenizer* absl_nullable tokenizer) {
  const std::string normalized =
      language.empty() ? "en" : NormalizeLanguageCode(language);
  if (normalized.empty()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported Whisper language: '", language, "'."));
  }

  if (tokenizer != nullptr) {
    const auto sot_id = tokenizer->TokenToId("<|startoftranscript|>");
    const auto translate_id = tokenizer->TokenToId("<|translate|>");
    const auto transcribe_id = tokenizer->TokenToId("<|transcribe|>");
    const auto notimestamps_id = tokenizer->TokenToId("<|notimestamps|>");
    if (sot_id.ok() && translate_id.ok() && transcribe_id.ok() &&
        notimestamps_id.ok() && *sot_id < *translate_id &&
        *translate_id < *transcribe_id && *transcribe_id < *notimestamps_id) {
      const auto lang_id =
          tokenizer->TokenToId(absl::StrCat("<|", normalized, "|>"));
      if (!lang_id.ok() || *lang_id <= *sot_id || *lang_id >= *translate_id) {
        return absl::InvalidArgumentError(
            absl::StrCat("Unsupported Whisper language: '", language, "'."));
      }
      return std::array<int, 4>{*sot_id, *lang_id, *transcribe_id,
                                *notimestamps_id};
    }
  }

  for (size_t i = 0; i < std::size(kWhisperLanguageCodes); ++i) {
    if (kWhisperLanguageCodes[i] == normalized) {
      return std::array<int, 4>{
          kWhisperStartOfTranscriptTokenId,
          kWhisperDefaultLanguageTokenId + static_cast<int>(i),
          kWhisperTranscribeTokenId, kWhisperNoTimestampsTokenId};
    }
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Unsupported Whisper language: '", language, "'."));
}

// Resolves configs and instantiates encoder, decoder, and sessions on demand.
class WhisperSessionFactory : public AsrSessionFactory {
 public:
  WhisperSessionFactory(
      const OmniEngine::Settings& settings,
      const MessageLite* absl_nullable session_config,
      const MessageLite* absl_nullable metadata,
      ::litert::Environment env,
      std::shared_ptr<lm::ModelResources> absl_nullable lm_resources,
      std::unique_ptr<::litert::support::Tokenizer> absl_nonnull tokenizer,
      std::unique_ptr<::litert::CompiledModel> absl_nonnull compiled_model)
      : AsrSessionFactory(settings, session_config, metadata, std::move(env),
                          std::move(lm_resources), std::move(tokenizer)),
        compiled_model_(std::move(compiled_model)) {}

 protected:
  absl::StatusOr<std::unique_ptr<MultiStagedSession>> CreateWithAudioSource(
      std::unique_ptr<AudioSource> absl_nonnull audio_source) override {
    auto encoder = std::make_unique<LiteRtRunnerImpl>(compiled_model_.get());
    std::vector<int> start_token_ids(
        metadata().decode_start_token_ids().begin(),
        metadata().decode_start_token_ids().end());
    WhisperDecoder::Options decoder_options;
    decoder_options.enable_timestamps = true;
    ABSL_ASSIGN_OR_RETURN(
        auto decoder,
        WhisperDecoder::Create(
            encoder.get(), "decode", /*num_non_state_inputs=*/2,
            /*num_non_state_outputs=*/1, std::move(start_token_ids),
            std::move(decoder_options)));
    return CreateMultiStagedSession(std::move(audio_source), std::move(encoder),
                                    std::move(decoder));
  }

 private:
  std::unique_ptr<::litert::CompiledModel> absl_nonnull compiled_model_;
};

// Instantiates common objects shared among sessions, then returns the session
// factory.
absl::StatusOr<std::unique_ptr<OmniSessionFactory>> GetWhisperSessionFactory(
    absl::string_view model_name, const OmniEngine::Settings& settings,
    const MessageLite* absl_nullable session_config,
    const MessageLite* absl_nullable metadata,
    ::litert::Environment& env, ::litert::Options& options,
    std::shared_ptr<lm::ModelResources> absl_nullable lm_resources) {
  if (!absl::c_linear_search(kAllModelNames, model_name)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Unexpected model_name: ", model_name));
  }
  if (lm_resources == nullptr) {
    return absl::InvalidArgumentError(absl::StrCat(
        "lm_resources must not be null (model_path: ", settings.model_path,
        ")."));
  }
  // Enforces mixed precision (FP16 with FP32 accumulator) on GPU for
  // whisper-medium for better quality, as medium model's deep layers easily
  // overflow FP16 dynamic range.
  if (settings.backend == OmniEngine::Settings::Backend::kGpu &&
      !settings.force_fp32_on_gpu &&
      absl::c_linear_search(kWhisperMediumModelNames, model_name)) {
    auto gpu_options = options.GetGpuOptions();
    if (gpu_options.HasValue()) {
      (void)gpu_options->SetPrecision(
          ::litert::GpuOptions::Precision::kFp16WithFp32Accum);
    }
  }
  ABSL_ASSIGN_OR_RETURN(
      auto compiled_model,
      CreateCompiledModel(
          settings, env, options, lm_resources.get(),
          lm::proto::Speech2TextMetadata::TF_LITE_ENCODER_DECODER));
  ABSL_ASSIGN_OR_RETURN(
      auto tokenizer,
      CreateTokenizer(lm_resources.get(), settings.tokenizer_path));
  ABSL_ASSIGN_OR_RETURN(
      lm::proto::Speech2TextMetadata resolved_metadata,
      ResolveWhisperMetadata(metadata, session_config, tokenizer.get()));
  return std::make_unique<WhisperSessionFactory>(
      settings, session_config, &resolved_metadata, std::move(env),
      std::move(lm_resources), std::move(tokenizer), std::move(compiled_model));
}

[[maybe_unused]] const bool kWhisperRegistered = [] {
  ABSL_CHECK_OK(OmniRegistry::Get().RegisterSessionFactory(RegistrySpec{
      .model_names = {std::begin(kWhisperTinyModelNames),
                      std::end(kWhisperTinyModelNames)},
      .get_session_factory = &GetWhisperSessionFactory,
      .session_config_type_id = "Speech2Text",
      .model_remote_url =
          "https://huggingface.co/litert-community/whisper-tiny/resolve/main/"
          "whisper_tiny_30s_i8_stateful.litertlm",
  }));
  ABSL_CHECK_OK(OmniRegistry::Get().RegisterSessionFactory(RegistrySpec{
      .model_names = {std::begin(kWhisperBaseModelNames),
                      std::end(kWhisperBaseModelNames)},
      .get_session_factory = &GetWhisperSessionFactory,
      .session_config_type_id = "Speech2Text",
      .model_remote_url =
          "https://huggingface.co/litert-community/whisper-base/resolve/main/"
          "whisper_base_30s_i8_stateful.litertlm",
  }));
  ABSL_CHECK_OK(OmniRegistry::Get().RegisterSessionFactory(RegistrySpec{
      .model_names = {std::begin(kWhisperSmallModelNames),
                      std::end(kWhisperSmallModelNames)},
      .get_session_factory = &GetWhisperSessionFactory,
      .session_config_type_id = "Speech2Text",
      .model_remote_url =
          "https://huggingface.co/litert-community/whisper-small/resolve/main/"
          "whisper_small_30s_i8_stateful.litertlm",
  }));
  ABSL_CHECK_OK(OmniRegistry::Get().RegisterSessionFactory(RegistrySpec{
      .model_names = {std::begin(kWhisperMediumModelNames),
                      std::end(kWhisperMediumModelNames)},
      .get_session_factory = &GetWhisperSessionFactory,
      .session_config_type_id = "Speech2Text",
      .model_remote_url =
          "https://huggingface.co/litert-community/whisper-medium/resolve/main/"
          "whisper_medium_30s_i8_stateful.litertlm",
  }));
  return true;
}();

}  // namespace

absl::StatusOr<lm::proto::Speech2TextMetadata> ResolveWhisperMetadata(
    const MessageLite* absl_nullable metadata,
    const MessageLite* absl_nullable session_config,
    ::litert::support::Tokenizer* absl_nullable tokenizer) {
  lm::proto::Speech2TextMetadata resolved;
  if (metadata != nullptr) {
    resolved.CheckTypeAndMergeFrom(*metadata);
  }
  if (!resolved.has_asr_model_type()) {
    resolved.mutable_asr_model_type()->mutable_whisper();
  }
  if (!resolved.has_input_milliseconds()) {
    resolved.set_input_milliseconds(30000);
  }

  std::string configured_language;
  if (session_config != nullptr) {
    lm::proto::Speech2TextSessionConfig resolved_session_config;
    resolved_session_config.CheckTypeAndMergeFrom(*session_config);
    if (resolved_session_config.has_language()) {
      configured_language = resolved_session_config.language();
    }
  }

  if (resolved.decode_start_token_ids_size() == 0 ||
      !configured_language.empty()) {
    ABSL_ASSIGN_OR_RETURN(
        const auto start_ids,
        ResolveWhisperStartTokenIds(configured_language, tokenizer));
    resolved.clear_decode_start_token_ids();
    for (int id : start_ids) {
      resolved.add_decode_start_token_ids(id);
    }
  }

  if (!resolved.has_decode_stop_token_id()) {
    int stop_id = kWhisperEndOfTextTokenId;
    if (tokenizer != nullptr) {
      if (const auto eot_id = tokenizer->TokenToId("<|endoftext|>");
          eot_id.ok() && *eot_id >= 0) {
        stop_id = *eot_id;
      }
    }
    resolved.set_decode_stop_token_id(stop_id);
  }
  auto* mel = resolved.mutable_log_mel_spectrogram_config();
  if (!mel->has_n_fft()) mel->set_n_fft(400);
  if (!mel->has_n_frames()) mel->set_n_frames(3000);
  if (!mel->has_norm_type()) {
    mel->set_norm_type(lm::proto::LogMelSpectrogramConfig::NORM_TYPE_WHISPER);
  }
  if (!mel->has_slaney_mel()) mel->set_slaney_mel(true);
  if (!mel->has_power_spectrum()) mel->set_power_spectrum(true);
  if (!mel->has_mel_floor()) mel->set_mel_floor(1e-10f);
  return resolved;
}

}  // namespace litert::omni::asr
