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

#include "omni/any2any/any2any_session.h"

#include <atomic>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "omni/base/io_types.h"
#include "omni/base/model_resources.h"
#include "omni/base/stage.h"
#include "omni/multi_staged_session.h"
#include "omni/omni_engine.h"
#include "omni/omni_session.h"
#include "runtime/framework/threadpool.h"
#include "runtime/proto/any2any_metadata.pb.h"
#include "runtime/proto/any2any_session_config.pb.h"

namespace litert::omni::any2any {
namespace {

lm::proto::Any2AnySessionConfig ResolveAny2AnySessionConfig(
    const MessageLite* absl_nullable session_config) {
  lm::proto::Any2AnySessionConfig resolved;
  if (session_config != nullptr) {
    resolved.CheckTypeAndMergeFrom(*session_config);
  }
  return resolved;
}

class LeasedAny2AnySession : public MultiStagedSession {
 public:
  LeasedAny2AnySession(std::vector<std::unique_ptr<internal::StageBase>> stages,
                       Stage<Output>* absl_nonnull output_stage,
                       ::litert::lm::ThreadPool* absl_nullable thread_pool,
                       std::shared_ptr<std::atomic<bool>> active_session)
      : MultiStagedSession(std::move(stages), output_stage, thread_pool),
        active_session_(std::move(active_session)) {}

  ~LeasedAny2AnySession() override {
    Reset();
    if (active_session_ != nullptr) {
      active_session_->store(false, std::memory_order_release);
    }
  }

 private:
  std::shared_ptr<std::atomic<bool>> active_session_;
};

}  // namespace

Any2AnySessionFactory::Any2AnySessionFactory(
    const OmniEngine::Settings& settings,
    const MessageLite* absl_nullable session_config,
    const lm::proto::Any2AnyMetadata& metadata,
    std::vector<std::string> available_voices,
    std::shared_ptr<ModelResources> absl_nullable model_resources)
    : engine_settings_(settings),
      session_config_(ResolveAny2AnySessionConfig(session_config)),
      metadata_(metadata),
      available_voices_(std::move(available_voices)),
      model_resources_(std::move(model_resources)),
      thread_pool_(std::make_unique<lm::ThreadPool>(
          "any2any_engine_pool",
          settings.num_threads > 0 ? settings.num_threads : 4)) {}

Any2AnySessionFactory::~Any2AnySessionFactory() = default;

std::vector<std::string> Any2AnySessionFactory::GetAvailableLanguages() const {
  return std::vector<std::string>(metadata_.supported_languages().begin(),
                                  metadata_.supported_languages().end());
}

bool Any2AnySessionFactory::HasVoice(absl::string_view voice) const {
  return absl::c_any_of(available_voices_, [voice](absl::string_view v) {
    return absl::EqualsIgnoreCase(v, voice);
  });
}

std::string Any2AnySessionFactory::GetDefaultVoice(
    absl::string_view /*language*/) const {
  if (!available_voices_.empty()) {
    return available_voices_.front();
  }
  return "";
}

lm::proto::Any2AnySessionConfig Any2AnySessionFactory::MergeSessionConfig(
    const lm::proto::Any2AnySessionConfig& override_config) const {
  lm::proto::Any2AnySessionConfig merged = session_config_;
  merged.MergeFrom(override_config);
  return merged;
}

int Any2AnySessionFactory::GetInputSampleRateHz() const {
  if (metadata_.input_sample_rate_hz() > 0) {
    return metadata_.input_sample_rate_hz();
  }
  return kDefaultInputSampleRateHz;
}

absl::StatusOr<std::unique_ptr<OmniSession>> Any2AnySessionFactory::Create(
    std::unique_ptr<OmniSession::InputSource> absl_nonnull input_source) {
  return CreateWithInputSource(session_config_, std::move(input_source));
}

lm::proto::Any2AnyMetadata ResolveAny2AnyMetadata(
    const MessageLite* absl_nullable metadata) {
  lm::proto::Any2AnyMetadata resolved;
  if (metadata != nullptr) {
    resolved.CheckTypeAndMergeFrom(*metadata);
  }
  if (resolved.supported_languages().empty()) {
    resolved.add_supported_languages("en-US");
    resolved.add_supported_languages("zh-CN");
  }
  return resolved;
}

LeasedAny2AnySessionFactory::LeasedAny2AnySessionFactory(
    const OmniEngine::Settings& settings,
    const MessageLite* absl_nullable session_config,
    const lm::proto::Any2AnyMetadata& metadata,
    std::vector<std::string> available_voices,
    std::shared_ptr<ModelResources> absl_nullable model_resources,
    ComponentBuilder component_builder, LeasedAny2AnySessionOptions options)
    : Any2AnySessionFactory(settings, session_config, metadata,
                            std::move(available_voices),
                            std::move(model_resources)),
      component_builder_(std::move(component_builder)),
      model_display_name_(std::move(options.model_display_name)),
      fallback_default_voice_(std::move(options.fallback_default_voice)),
      active_session_(std::make_shared<std::atomic<bool>>(false)) {}

std::string LeasedAny2AnySessionFactory::GetDefaultVoice(
    absl::string_view /*language*/) const {
  if (!available_voices().empty()) {
    return available_voices().front();
  }
  return fallback_default_voice_;
}

absl::StatusOr<std::unique_ptr<MultiStagedSession>>
LeasedAny2AnySessionFactory::CreateWithInputSource(
    const lm::proto::Any2AnySessionConfig& session_config,
    std::unique_ptr<OmniSession::InputSource> absl_nonnull input_source) {
  if (model_resources() == nullptr) {
    return absl::FailedPreconditionError("ModelResources must not be null.");
  }
  if (!component_builder_) {
    return absl::FailedPreconditionError("ComponentBuilder must not be null.");
  }

  bool expected = false;
  if (!active_session_->compare_exchange_strong(expected, true,
                                                std::memory_order_acq_rel,
                                                std::memory_order_relaxed)) {
    const absl::string_view display_name =
        model_display_name_.empty() ? absl::string_view("Any2Any")
                                    : absl::string_view(model_display_name_);
    return absl::FailedPreconditionError(
        absl::StrCat("Another ", display_name,
                     " session is already active on this engine."));
  }
  bool lease_transferred = false;
  absl::Cleanup release_lease = [&] {
    if (!lease_transferred) {
      active_session_->store(false, std::memory_order_release);
    }
  };

  lm::proto::Any2AnySessionConfig effective =
      MergeSessionConfig(session_config);
  if (input_source == nullptr) {
    input_source = std::make_unique<PushInputSource>();
  }

  std::vector<std::unique_ptr<internal::StageBase>> stages;
  ABSL_ASSIGN_OR_RETURN(Stage<Output> * output_stage,
                        component_builder_(effective, std::move(input_source),
                                           model_resources(), stages));
  if (output_stage == nullptr) {
    return absl::InternalError("Output stage must not be null.");
  }
  if (stages.empty() || stages.back().get() != output_stage) {
    return absl::InternalError(
        "Component builder must append output stage as the final stage.");
  }
  for (const std::unique_ptr<internal::StageBase>& stage : stages) {
    if (stage == nullptr) {
      return absl::InternalError("Pipeline stages must not contain null.");
    }
  }

  lease_transferred = true;
  return std::make_unique<LeasedAny2AnySession>(std::move(stages), output_stage,
                                                thread_pool(), active_session_);
}

}  // namespace litert::omni::any2any
