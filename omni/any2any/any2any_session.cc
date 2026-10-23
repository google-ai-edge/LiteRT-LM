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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "omni/base/io_types.h"
#include "omni/base/model_resources.h"
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
  return absl::c_linear_search(available_voices_, voice);
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

}  // namespace litert::omni::any2any
