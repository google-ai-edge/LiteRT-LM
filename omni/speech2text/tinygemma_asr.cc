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
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_options.h"  // from @litert
#include "omni/base/litert_lm_engine_runner.h"
#include "omni/base/litert_runner.h"
#include "omni/base/model_utils.h"
#include "omni/multi_staged_session.h"
#include "omni/omni_engine.h"
#include "omni/omni_registry.h"
#include "omni/omni_session.h"
#include "omni/speech2text/asr_session.h"
#include "omni/speech2text/audio_source.h"
#include "omni/speech2text/lm_engine_decoder.h"
#include "runtime/components/model_resources.h"
#include "runtime/proto/speech2text_metadata.pb.h"
#include "runtime/proto/speech2text_model_type.pb.h"
#include "runtime/proto/speech2text_session_config.pb.h"
#include "support/tokenizer/tokenizer.h"

namespace litert::omni::asr {
namespace {

constexpr absl::string_view kModelNames[] = {
    "tinygemma-asr",
    "tinygemma-asr-int8",
};

constexpr absl::string_view kDefaultPrompt =
    "Transcribe the following speech segment: ";
constexpr int kDefaultMaxOutputTokens = 128;

// Resolves configs and instantiates encoder, decoder, and sessions on demand.
class TinyGemmaAsrSessionFactory : public AsrSessionFactory {
 public:
  static lm::proto::Speech2TextSessionConfig ResolveSessionConfig(
      const MessageLite* absl_nullable session_config) {
    lm::proto::Speech2TextSessionConfig resolved;
    if (session_config != nullptr) {
      resolved.CheckTypeAndMergeFrom(*session_config);
    }
    if (!resolved.has_text_merger_type() ||
        resolved.text_merger_type() ==
            lm::proto::Speech2TextSessionConfig::TEXT_MERGER_TYPE_UNSPECIFIED) {
      resolved.set_text_merger_type(
          lm::proto::Speech2TextSessionConfig::TEXT_MERGER_TYPE_LEVENSHTEIN);
    }
    return resolved;
  }

  static lm::proto::Speech2TextMetadata ResolveMetadata(
      const MessageLite* absl_nullable metadata) {
    lm::proto::Speech2TextMetadata resolved;
    if (metadata != nullptr) {
      resolved.CheckTypeAndMergeFrom(*metadata);
    }
    if (!resolved.has_decode_stop_token_id()) {
      resolved.set_decode_stop_token_id(1);
    }
    auto* mel = resolved.mutable_log_mel_spectrogram_config();
    if (!mel->has_n_fft()) mel->set_n_fft(512);
    if (!mel->has_n_mels()) mel->set_n_mels(128);
    if (!mel->has_n_frames()) mel->set_n_frames(500);
    if (!mel->has_fft_length()) mel->set_fft_length(1024);
    if (!mel->has_input_scale()) mel->set_input_scale(32768.0f);
    if (!mel->has_preemphasis()) mel->set_preemphasis(0.97f);
    if (!mel->has_mel_low_hz()) mel->set_mel_low_hz(125.0f);
    if (!mel->has_mel_high_hz()) mel->set_mel_high_hz(7500.0f);
    if (!mel->has_mel_floor()) mel->set_mel_floor(1e-6f);
    if (!mel->has_normalize_mel()) mel->set_normalize_mel(true);
    if (!mel->has_add_floor_to_mel_before_log()) {
      mel->set_add_floor_to_mel_before_log(false);
    }
    if (!mel->has_transpose()) mel->set_transpose(true);
    return resolved;
  }

  TinyGemmaAsrSessionFactory(
      const OmniEngine::Settings& settings,
      const MessageLite* absl_nullable session_config,
      const MessageLite* absl_nullable metadata,
      ::litert::Environment env,
      std::shared_ptr<lm::ModelResources> absl_nullable lm_resources,
      std::unique_ptr<::litert::support::Tokenizer> absl_nonnull tokenizer,
      std::unique_ptr<LiteRtLmEngineRunner> absl_nonnull lm_engine_runner)
      : AsrSessionFactory(settings, session_config, metadata, std::move(env),
                          std::move(lm_resources), std::move(tokenizer)),
        lm_engine_runner_(std::move(lm_engine_runner)) {}

 protected:
  absl::StatusOr<std::unique_ptr<MultiStagedSession>> CreateWithAudioSource(
      std::unique_ptr<AudioSource> absl_nonnull audio_source) override {
    const auto& mel = metadata().log_mel_spectrogram_config();
    const size_t n_frames = mel.n_frames() > 0 ? mel.n_frames() : 500;
    const size_t n_mels = mel.n_mels() > 0 ? mel.n_mels() : 128;
    const size_t buffer_size_bytes = n_frames * n_mels * sizeof(float);
    auto encoder = std::make_unique<PassthroughRunner>(
        std::vector<size_t>{buffer_size_bytes});
    std::vector<int> start_token_ids(
        metadata().decode_start_token_ids().begin(),
        metadata().decode_start_token_ids().end());
    ABSL_ASSIGN_OR_RETURN(
        auto decoder, LmEngineDecoder::Create(
                          lm_engine_runner_.get(), std::string(kDefaultPrompt),
                          kDefaultMaxOutputTokens, std::move(start_token_ids),
                          metadata().decode_stop_token_id(),
                          metadata().decode_skip_until_token_id()));
    return CreateMultiStagedSession(std::move(audio_source), std::move(encoder),
                                    std::move(decoder));
  }

 private:
  std::unique_ptr<LiteRtLmEngineRunner> absl_nonnull lm_engine_runner_;
};

// Instantiates common objects shared among sessions, then returns the session
// factory.
absl::StatusOr<std::unique_ptr<OmniSessionFactory>>
GetTinyGemmaAsrSessionFactory(
    absl::string_view model_name, const OmniEngine::Settings& settings,
    const MessageLite* absl_nullable session_config,
    const MessageLite* absl_nullable metadata,
    ::litert::Environment& env, ::litert::Options& /*options*/,
    std::shared_ptr<lm::ModelResources> absl_nullable lm_resources) {
  if (!absl::c_linear_search(kModelNames, model_name)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Unexpected model_name: ", model_name));
  }
  if (lm_resources == nullptr) {
    return absl::InvalidArgumentError(absl::StrCat(
        "lm_resources must not be null (model_path: ", settings.model_path,
        ")."));
  }
  ModelOptions lm_options = ToLmModelOptions(settings);
  // tinygemma-asr's USM Conformer audio encoder requires FP32 precision on GPU
  // to avoid FP16 numerical degradation.
  lm_options.force_fp32_on_gpu = true;
  ABSL_ASSIGN_OR_RETURN(
      auto lm_engine_runner,
      CreateLmEngineRunner(env, lm_options, settings.model_path));
  ABSL_ASSIGN_OR_RETURN(
      auto tokenizer,
      CreateTokenizer(lm_resources.get(), settings.tokenizer_path));
  lm::proto::Speech2TextSessionConfig resolved_session_config =
      TinyGemmaAsrSessionFactory::ResolveSessionConfig(session_config);
  lm::proto::Speech2TextMetadata resolved_metadata =
      TinyGemmaAsrSessionFactory::ResolveMetadata(metadata);
  return std::make_unique<TinyGemmaAsrSessionFactory>(
      settings, &resolved_session_config, &resolved_metadata, std::move(env),
      std::move(lm_resources), std::move(tokenizer),
      std::move(lm_engine_runner));
}

[[maybe_unused]] const bool kTinyGemmaAsrRegistered = [] {
  ABSL_CHECK_OK(OmniRegistry::Get().RegisterSessionFactory(RegistrySpec{
      .model_names = {std::begin(kModelNames), std::end(kModelNames)},
      .get_session_factory = &GetTinyGemmaAsrSessionFactory,
      .session_config_type_id = "Speech2Text",
      .model_remote_url =
          "https://huggingface.co/litert-community/TinyGemma-ASR/resolve/main/"
          "tiny_gemma_asr_5s_i8.litertlm",
  }));
  return true;
}();

}  // namespace
}  // namespace litert::omni::asr
