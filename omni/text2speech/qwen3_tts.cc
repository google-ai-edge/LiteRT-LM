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
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
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
#include "omni/text2speech/qwen3_tts/common.h"
#include "omni/text2speech/qwen3_tts/qwen3_tts_factory.h"
#include "omni/text2speech/qwen3_tts/qwen3_tts_model_config.h"
#include "omni/text2speech/stream_text_source.h"
#include "omni/text2speech/tts_session.h"
#include "runtime/components/model_resources.h"
#include "runtime/proto/text2speech_metadata.pb.h"
#include "runtime/proto/text2speech_model_type.pb.h"
#include "runtime/proto/text2speech_session_config.pb.h"

namespace litert::omni::tts {
namespace {

constexpr absl::string_view kModelNames[] = {
    "qwen3-tts",
};

lm::proto::Text2SpeechMetadata ResolveMetadata(
    const MessageLite* absl_nullable metadata) {
  lm::proto::Text2SpeechMetadata resolved;
  if (metadata != nullptr) {
    resolved.CheckTypeAndMergeFrom(*metadata);
  }
  resolved.mutable_tts_model_type()->mutable_qwen3_tts();
  if (resolved.supported_languages().empty()) {
    resolved.add_supported_languages("en-US");
    resolved.add_supported_languages("zh-CN");
  }
  return resolved;
}

void PopulateQwen3TtsConfigFromProto(const lm::proto::Qwen3Tts& proto,
                                     Qwen3TtsModelConfig& config) {
  if (proto.has_language() && !proto.language().empty()) {
    config.language = proto.language();
  }
  if (proto.has_speaker_file() && !proto.speaker_file().empty()) {
    config.speaker_file = proto.speaker_file();
  }
  if (proto.has_max_frames() && proto.max_frames() > 0) {
    config.max_frames = proto.max_frames();
  }
}

class Qwen3TtsSessionFactory : public TtsSessionFactory {
 public:
  Qwen3TtsSessionFactory(
      const OmniEngine::Settings& settings,
      const MessageLite* absl_nullable session_config,
      const lm::proto::Text2SpeechMetadata& metadata,
      std::vector<std::string> available_voices,
      std::shared_ptr<ModelResources> absl_nullable model_resources,
      Qwen3TtsModelConfig config)
      : TtsSessionFactory(settings, session_config, metadata,
                          std::move(available_voices),
                          std::move(model_resources)),
        config_(std::move(config)) {}

  std::string GetDefaultVoice(absl::string_view /*language*/) const override {
    if (!available_voices().empty()) {
      return available_voices().front();
    }
    return config_.speaker_file;
  }

 protected:
  absl::StatusOr<std::unique_ptr<MultiStagedSession>> CreateWithTextSource(
      const lm::proto::Text2SpeechSessionConfig& session_config,
      std::unique_ptr<StreamTextSource> absl_nullable text_source) override {
    lm::proto::Text2SpeechSessionConfig effective =
        MergeSessionConfig(session_config);
    if (effective.language().empty()) {
      return absl::InvalidArgumentError(
          "Text2SpeechSessionConfig::language must not be empty.");
    }

    if (text_source == nullptr) {
      text_source =
          std::make_unique<StreamTextSource>(ResolveTextChunkConfig(effective));
    }

    Qwen3TtsModelConfig session_model_config = config_;
    ABSL_ASSIGN_OR_RETURN(session_model_config.language,
                          ResolveQwen3TtsLanguage(effective.language()));
    if (!effective.voice().empty()) {
      session_model_config.speaker_file = effective.voice();
    }

    std::vector<std::unique_ptr<internal::StageBase>> stages;
    Stage<Output>* output_stage = nullptr;
    LITERT_RETURN_IF_ERROR(CreateQwen3TtsComponents(
        session_model_config, engine_settings().model_path,
        std::move(text_source), model_resources(), stages, &output_stage));
    if (output_stage == nullptr) {
      return absl::InternalError("Output stage must not be null.");
    }
    return MultiStagedSession::Create(std::move(stages), output_stage,
                                      thread_pool());
  }

 private:
  Qwen3TtsModelConfig config_;
};

absl::StatusOr<std::unique_ptr<OmniSessionFactory>> GetQwen3TtsSessionFactory(
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
  Qwen3TtsModelConfig config;
  config.force_fp32_on_gpu = settings.force_fp32_on_gpu;
  PopulateQwen3TtsConfigFromProto(
      resolved_metadata.tts_model_type().qwen3_tts(), config);

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
    resources->SetLmModelResources(std::move(lm_resources));
  }

  const int num_threads = settings.num_threads > 0 ? settings.num_threads : 4;
  LITERT_RETURN_IF_ERROR(InitQwen3TtsResources(
      config, settings.model_path, settings.cache_dir,
      ToLmBackend(settings.backend), num_threads, *shared_env, *resources));

  std::vector<std::string> available_voices;
  if (!config.speaker_file.empty()) {
    available_voices.push_back(config.speaker_file);
  }

  return std::make_unique<Qwen3TtsSessionFactory>(
      settings, session_config, resolved_metadata, std::move(available_voices),
      std::move(resources), std::move(config));
}

[[maybe_unused]] const bool kQwen3TtsRegistered = [] {
  ABSL_CHECK_OK(OmniRegistry::Get().RegisterSessionFactory(RegistrySpec{
      .model_names = {std::begin(kModelNames), std::end(kModelNames)},
      .get_session_factory = &GetQwen3TtsSessionFactory,
      .session_config_type_id = "Text2Speech",
      .model_remote_url =
          "https://huggingface.co/litert-community/Qwen3-TTS-12Hz-0.6B-Base/"
          "resolve/main/qwen3_tts_int8.litertlm",
  }));
  return true;
}();

}  // namespace
}  // namespace litert::omni::tts
