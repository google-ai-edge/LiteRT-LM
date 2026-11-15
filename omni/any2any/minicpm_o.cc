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

#include <atomic>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_options.h"  // from @litert
#include "omni/any2any/any2any_session.h"
#include "omni/any2any/minicpm_o/minicpm_o_factory.h"
#include "omni/any2any/minicpm_o/minicpm_o_model_config.h"
#include "omni/base/io_types.h"
#include "omni/base/model_resources.h"
#include "omni/base/model_utils.h"
#include "omni/base/stage.h"
#include "omni/multi_staged_session.h"
#include "omni/omni_engine.h"
#include "omni/omni_registry.h"
#include "omni/omni_session.h"
#include "runtime/components/model_resources.h"
#include "runtime/framework/threadpool.h"
#include "runtime/proto/any2any_metadata.pb.h"
#include "runtime/proto/any2any_model_type.pb.h"
#include "runtime/proto/any2any_session_config.pb.h"

namespace litert::omni::any2any {
namespace {

constexpr absl::string_view kModelNames[] = {
    "minicpm-o-4.5",
    "minicpm-o-4_5",
    "minicpm-o",
    "minicpm_o",
    "openbmb/minicpm-o-4.5",
    "openbmb/MiniCPM-o-4_5",
    "openbmb/MiniCPM-o-4.5",
};

lm::proto::Any2AnyMetadata ResolveMetadata(
    lm::ModelResources* absl_nullable lm_resources,
    const MessageLite* absl_nullable metadata) {
  lm::proto::Any2AnyMetadata resolved;
  if (lm_resources != nullptr) {
    absl::StatusOr<const lm::proto::Any2AnyMetadata*> embedded =
        lm_resources->GetAny2AnyMetadata();
    if (embedded.ok() && *embedded != nullptr) {
      resolved = **embedded;
    }
  }
  if (metadata != nullptr) {
    resolved.CheckTypeAndMergeFrom(*metadata);
  }
  resolved.mutable_any2any_model_type()->mutable_minicpm_o();
  if (resolved.supported_languages().empty()) {
    resolved.add_supported_languages("en-US");
    resolved.add_supported_languages("zh-CN");
  }
  return resolved;
}

void PopulateMiniCpmOConfigFromProto(const lm::proto::Any2AnyMetadata& metadata,
                                     MiniCpmOModelConfig& config) {
  if (metadata.has_input_sample_rate_hz() &&
      metadata.input_sample_rate_hz() > 0) {
    config.audio_input_sample_rate_hz = metadata.input_sample_rate_hz();
  }
  if (metadata.has_output_sample_rate_hz() &&
      metadata.output_sample_rate_hz() > 0) {
    config.audio_output_sample_rate_hz = metadata.output_sample_rate_hz();
  }
  if (metadata.has_system_prompt() && !metadata.system_prompt().empty()) {
    config.system_prompt = metadata.system_prompt();
  }
  if (!metadata.any2any_model_type().has_minicpm_o()) {
    return;
  }
  const lm::proto::MiniCpmO& proto = metadata.any2any_model_type().minicpm_o();
  if (proto.has_llm_hidden_size() && proto.llm_hidden_size() > 0) {
    config.llm_hidden_size = proto.llm_hidden_size();
  }
  if (proto.has_llm_vocab_size() && proto.llm_vocab_size() > 0) {
    config.llm_vocab_size = proto.llm_vocab_size();
  }
  if (proto.has_tts_hidden_size() && proto.tts_hidden_size() > 0) {
    config.tts_hidden_size = proto.tts_hidden_size();
  }
  if (proto.has_tts_vocab_size() && proto.tts_vocab_size() > 0) {
    config.tts_vocab_size = proto.tts_vocab_size();
  }
  if (proto.has_tts_num_mel_bins() && proto.tts_num_mel_bins() > 0) {
    config.tts_num_mel_bins = proto.tts_num_mel_bins();
  }
  if (proto.has_vision_scale_resolution() &&
      proto.vision_scale_resolution() > 0) {
    config.vision_scale_resolution = proto.vision_scale_resolution();
  }
  if (proto.has_vision_patch_size() && proto.vision_patch_size() > 0) {
    config.vision_patch_size = proto.vision_patch_size();
  }
  if (proto.has_vision_max_slice_nums() && proto.vision_max_slice_nums() > 0) {
    config.vision_max_slice_nums = proto.vision_max_slice_nums();
  }
  if (proto.has_num_image_queries() && proto.num_image_queries() > 0) {
    config.num_image_queries = proto.num_image_queries();
  }
  if (proto.has_audio_pool_step() && proto.audio_pool_step() > 0) {
    config.audio_pool_step = proto.audio_pool_step();
  }
  if (proto.has_generate_audio()) {
    config.generate_audio = proto.generate_audio();
  }
  if (proto.has_max_new_tokens() && proto.max_new_tokens() > 0) {
    config.max_new_tokens = proto.max_new_tokens();
  }
  if (proto.has_max_tts_tokens() && proto.max_tts_tokens() > 0) {
    config.max_tts_tokens = proto.max_tts_tokens();
  }
  if (proto.has_voice() && !proto.voice().empty()) {
    config.voice_name = proto.voice();
  }
  if (proto.has_language() && !proto.language().empty()) {
    config.language = proto.language();
  }
}

void ApplySessionConfigToModelConfig(
    const lm::proto::Any2AnySessionConfig& session_config,
    MiniCpmOModelConfig& config) {
  if (session_config.has_generate_audio()) {
    config.generate_audio = session_config.generate_audio();
  }
  if (session_config.has_max_new_tokens() &&
      session_config.max_new_tokens() > 0) {
    config.max_new_tokens = session_config.max_new_tokens();
  }
  if (session_config.has_max_tts_tokens() &&
      session_config.max_tts_tokens() > 0) {
    config.max_tts_tokens = session_config.max_tts_tokens();
  }
  if (session_config.has_temperature()) {
    config.temperature = session_config.temperature();
    config.do_sample = session_config.temperature() > 0.0f;
  }
  if (session_config.has_top_p() && session_config.top_p() > 0.0f) {
    config.top_p = session_config.top_p();
  }
  if (session_config.has_top_k() && session_config.top_k() > 0) {
    config.top_k = session_config.top_k();
  }
  if (session_config.has_tts_temperature()) {
    config.tts_temperature = session_config.tts_temperature();
  }
  if (session_config.has_tts_top_p() && session_config.tts_top_p() > 0.0f) {
    config.tts_top_p = session_config.tts_top_p();
  }
  if (session_config.has_tts_top_k() && session_config.tts_top_k() > 0) {
    config.tts_top_k = session_config.tts_top_k();
  }
  if (session_config.has_system_prompt()) {
    config.system_prompt = session_config.system_prompt();
  }
  if (session_config.has_voice() && !session_config.voice().empty()) {
    config.voice_name = session_config.voice();
  }
  if (session_config.has_language() && !session_config.language().empty()) {
    config.language = session_config.language();
  }
}

class MiniCpmOSession : public MultiStagedSession {
 public:
  MiniCpmOSession(std::vector<std::unique_ptr<internal::StageBase>> stages,
                  Stage<Output>* absl_nonnull output_stage,
                  ::litert::lm::ThreadPool* absl_nullable thread_pool,
                  std::shared_ptr<std::atomic<bool>> active_session)
      : MultiStagedSession(std::move(stages), output_stage, thread_pool),
        active_session_(std::move(active_session)) {}

  ~MiniCpmOSession() override {
    Reset();
    if (active_session_ != nullptr) {
      active_session_->store(false, std::memory_order_release);
    }
  }

 private:
  std::shared_ptr<std::atomic<bool>> active_session_;
};

class MiniCpmOSessionFactory : public Any2AnySessionFactory {
 public:
  MiniCpmOSessionFactory(
      const OmniEngine::Settings& settings,
      const MessageLite* absl_nullable session_config,
      const lm::proto::Any2AnyMetadata& metadata,
      std::vector<std::string> available_voices,
      std::shared_ptr<ModelResources> absl_nullable model_resources,
      MiniCpmOModelConfig config = {})
      : Any2AnySessionFactory(settings, session_config, metadata,
                              std::move(available_voices),
                              std::move(model_resources)),
        config_(std::move(config)) {}

  std::string GetDefaultVoice(absl::string_view /*language*/) const override {
    if (!available_voices().empty()) {
      return available_voices().front();
    }
    return config_.voice_name;
  }

 protected:
  absl::StatusOr<std::unique_ptr<MultiStagedSession>> CreateWithInputSource(
      const lm::proto::Any2AnySessionConfig& session_config,
      std::unique_ptr<OmniSession::InputSource> absl_nonnull input_source)
      override {
    if (model_resources() == nullptr) {
      return absl::FailedPreconditionError("ModelResources must not be null.");
    }

    bool expected = false;
    if (!active_session_->compare_exchange_strong(expected, true,
                                                  std::memory_order_acq_rel,
                                                  std::memory_order_relaxed)) {
      return absl::FailedPreconditionError(
          "Another MiniCPM-o session is already active on this engine.");
    }
    bool lease_transferred = false;
    absl::Cleanup release_lease = [&] {
      if (!lease_transferred) {
        active_session_->store(false, std::memory_order_release);
      }
    };

    lm::proto::Any2AnySessionConfig effective =
        MergeSessionConfig(session_config);
    MiniCpmOModelConfig session_model_config = config_;
    ApplySessionConfigToModelConfig(effective, session_model_config);

    if (input_source == nullptr) {
      input_source = std::make_unique<PushInputSource>();
    }

    std::vector<std::unique_ptr<internal::StageBase>> stages;
    ABSL_ASSIGN_OR_RETURN(
        Stage<Output> * output_stage,
        CreateMiniCpmOComponents(session_model_config, std::move(input_source),
                                 model_resources(), stages));
    if (output_stage == nullptr) {
      return absl::InternalError("Output stage must not be null.");
    }
    lease_transferred = true;
    return std::make_unique<MiniCpmOSession>(std::move(stages), output_stage,
                                             thread_pool(), active_session_);
  }

 private:
  MiniCpmOModelConfig config_;
  std::shared_ptr<std::atomic<bool>> active_session_ =
      std::make_shared<std::atomic<bool>>(false);
};

absl::StatusOr<std::unique_ptr<OmniSessionFactory>> GetMiniCpmOSessionFactory(
    absl::string_view model_name, const OmniEngine::Settings& settings,
    const MessageLite* absl_nullable session_config,
    const MessageLite* absl_nullable metadata, ::litert::Environment& env,
    ::litert::Options& /*options*/,
    std::shared_ptr<lm::ModelResources> absl_nullable lm_resources) {
  if (!absl::c_linear_search(kModelNames, model_name)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Unexpected model_name: ", model_name));
  }
  if (lm_resources == nullptr &&
      absl::EndsWithIgnoreCase(settings.model_path, ".litertlm")) {
    ABSL_ASSIGN_OR_RETURN(lm_resources,
                          CreateLmModelResources(settings.model_path));
  }

  lm::proto::Any2AnyMetadata resolved_metadata =
      ResolveMetadata(lm_resources.get(), metadata);
  MiniCpmOModelConfig config;
  config.force_fp32_on_gpu = settings.force_fp32_on_gpu;
  PopulateMiniCpmOConfigFromProto(resolved_metadata, config);
  if (session_config != nullptr) {
    lm::proto::Any2AnySessionConfig init_session_cfg;
    init_session_cfg.CheckTypeAndMergeFrom(*session_config);
    ApplySessionConfigToModelConfig(init_session_cfg, config);
  }

  auto shared_env = std::make_shared<Environment>(std::move(env));
  auto resources = std::make_shared<ModelResources>(shared_env);
  if (lm_resources != nullptr) {
    resources->SetLmModelResources(std::move(lm_resources));
  }

  const int num_threads = settings.num_threads > 0 ? settings.num_threads : 4;
  ABSL_RETURN_IF_ERROR(InitMiniCpmOResources(
      config, settings.model_path, settings.cache_dir,
      ToLmBackend(settings.backend), num_threads, *shared_env, *resources));

  std::vector<std::string> available_voices;
  if (!config.voice_name.empty()) {
    available_voices.push_back(config.voice_name);
  }

  return std::make_unique<MiniCpmOSessionFactory>(
      settings, session_config, resolved_metadata, std::move(available_voices),
      std::move(resources), std::move(config));
}

[[maybe_unused]] const bool kMiniCpmORegistered = [] {
  ABSL_CHECK_OK(OmniRegistry::Get().RegisterSessionFactory(RegistrySpec{
      .model_names = {std::begin(kModelNames), std::end(kModelNames)},
      .get_session_factory = &GetMiniCpmOSessionFactory,
      .session_config_type_id = "Any2Any",
      .model_remote_url =
          "https://huggingface.co/litert-community/MiniCPM-o-4_5/resolve/main/"
          "minicpm_o_4_5_int8.litertlm",
  }));
  return true;
}();

}  // namespace
}  // namespace litert::omni::any2any
