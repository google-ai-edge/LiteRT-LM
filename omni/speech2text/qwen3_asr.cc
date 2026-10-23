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

#include <iterator>
#include <memory>
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
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_options.h"  // from @litert
#include "omni/base/litert_lm_runner.h"
#include "omni/base/litert_runner.h"
#include "omni/base/model_utils.h"
#include "omni/multi_staged_session.h"
#include "omni/omni_engine.h"
#include "omni/omni_registry.h"
#include "omni/omni_session.h"
#include "omni/speech2text/asr_session.h"
#include "omni/speech2text/audio_source.h"
#include "omni/speech2text/lm_decoder.h"
#include "runtime/components/model_resources.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/proto/speech2text_metadata.pb.h"
#include "runtime/proto/speech2text_model_type.pb.h"
#include "support/tokenizer/tokenizer.h"

namespace litert::omni::asr {
namespace {

constexpr absl::string_view kModelNames[] = {
    "qwen3-asr-0.6b", "qwen3-asr-0.6b-int8", "qwen3-asr-0.6b-int4",
    "qwen3-asr-1.7b", "qwen3-asr-1.7b-int8", "qwen3-asr-1.7b-int4",
    "qwen3-asr",      "qwen3-asr-int8",
};

// Resolves configs and instantiates encoder, decoder, and sessions on demand.
class Qwen3AsrSessionFactory : public AsrSessionFactory {
 public:
  static lm::proto::Speech2TextMetadata ResolveMetadata(
      const MessageLite* absl_nullable metadata) {
    lm::proto::Speech2TextMetadata resolved;
    if (metadata != nullptr) {
      resolved.CheckTypeAndMergeFrom(*metadata);
    }
    if (!resolved.has_asr_model_type()) {
      resolved.mutable_asr_model_type()->mutable_qwen3_asr();
    }
    if (!resolved.has_decode_skip_until_token_id()) {
      resolved.set_decode_skip_until_token_id(151704);
    }
    auto* mel = resolved.mutable_log_mel_spectrogram_config();
    if (!mel->has_n_fft()) mel->set_n_fft(400);
    if (!mel->has_n_mels()) mel->set_n_mels(128);
    if (!mel->has_n_frames()) mel->set_n_frames(500);
    if (!mel->has_norm_type()) {
      mel->set_norm_type(lm::proto::LogMelSpectrogramConfig::NORM_TYPE_WHISPER);
    }
    if (!mel->has_slaney_mel()) mel->set_slaney_mel(true);
    if (!mel->has_power_spectrum()) mel->set_power_spectrum(true);
    if (!mel->has_mel_floor()) mel->set_mel_floor(1e-10f);
    return resolved;
  }

  Qwen3AsrSessionFactory(
      const OmniEngine::Settings& settings,
      const MessageLite* absl_nullable session_config,
      const MessageLite* absl_nullable metadata,
      ::litert::Environment env,
      std::shared_ptr<lm::ModelResources> absl_nullable lm_resources,
      std::unique_ptr<::litert::support::Tokenizer> absl_nonnull tokenizer,
      std::unique_ptr<::litert::CompiledModel> absl_nonnull compiled_model,
      std::unique_ptr<LiteRtLmRunner> absl_nonnull lm_runner)
      : AsrSessionFactory(settings, session_config, metadata, std::move(env),
                          std::move(lm_resources), std::move(tokenizer)),
        compiled_model_(std::move(compiled_model)),
        lm_runner_(std::move(lm_runner)) {}

 protected:
  bool ShouldEnableWarmStart() const override { return true; }

  absl::StatusOr<std::unique_ptr<MultiStagedSession>> CreateWithAudioSource(
      std::unique_ptr<AudioSource> absl_nonnull audio_source) override {
    auto encoder = std::make_unique<LiteRtRunnerImpl>(compiled_model_.get());
    std::vector<int> start_token_ids(
        metadata().decode_start_token_ids().begin(),
        metadata().decode_start_token_ids().end());
    ABSL_ASSIGN_OR_RETURN(
        auto decoder,
        LmDecoder::Create(lm_runner_.get(), std::move(start_token_ids),
                          metadata().decode_stop_token_id(),
                          metadata().decode_skip_until_token_id()));
    return CreateMultiStagedSession(std::move(audio_source), std::move(encoder),
                                    std::move(decoder));
  }

 private:
  std::unique_ptr<::litert::CompiledModel> absl_nonnull compiled_model_;
  std::unique_ptr<LiteRtLmRunner> absl_nonnull lm_runner_;
};

// Instantiates common objects shared among sessions, then returns the session
// factory.
absl::StatusOr<std::unique_ptr<OmniSessionFactory>> GetQwen3AsrSessionFactory(
    absl::string_view model_name, const OmniEngine::Settings& settings,
    const MessageLite* absl_nullable session_config,
    const MessageLite* absl_nullable metadata,
    ::litert::Environment& env, ::litert::Options& options,
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
  ABSL_ASSIGN_OR_RETURN(
      auto lm_runner,
      CreateLmRunner(env, ToLmModelOptions(settings), settings.model_path));
  ABSL_ASSIGN_OR_RETURN(
      auto tokenizer,
      CreateTokenizer(lm_resources.get(), settings.tokenizer_path));
  ABSL_ASSIGN_OR_RETURN(
      auto compiled_model,
      CreateCompiledModel(settings, env, options, lm_resources.get(),
                          lm::proto::LlmMetadata::TF_LITE_AUDIO_ENCODER_HW));
  lm::proto::Speech2TextMetadata resolved_metadata =
      Qwen3AsrSessionFactory::ResolveMetadata(metadata);
  return std::make_unique<Qwen3AsrSessionFactory>(
      settings, session_config, &resolved_metadata, std::move(env),
      std::move(lm_resources), std::move(tokenizer), std::move(compiled_model),
      std::move(lm_runner));
}

[[maybe_unused]] const bool kQwen3AsrRegistered = [] {
  ABSL_CHECK_OK(OmniRegistry::Get().RegisterSessionFactory(RegistrySpec{
      .model_names = {std::begin(kModelNames), std::end(kModelNames)},
      .get_session_factory = &GetQwen3AsrSessionFactory,
      .session_config_type_id = "Speech2Text",
      .model_remote_url =
          "https://huggingface.co/litert-community/Qwen3-ASR-0.6B/resolve/"
          "main/qwen3_asr_0.6b_5s_i8.litertlm",
  }));
  return true;
}();

}  // namespace
}  // namespace litert::omni::asr
