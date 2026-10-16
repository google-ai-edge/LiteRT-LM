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

#include "omni/omni_engine.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "omni/omni_registry.h"
#include "omni/omni_session.h"

namespace litert::omni {

OmniEngine::OmniEngine(
    std::string model_name,
    std::unique_ptr<OmniSessionFactory> absl_nonnull session_factory)
    : model_name_(std::move(model_name)),
      session_factory_(std::move(session_factory)) {}

absl::StatusOr<std::unique_ptr<OmniEngine>> OmniEngine::Create(
    absl::string_view model_name, const Settings& settings,
    const MessageLite* absl_nullable session_config,
    const MessageLite* absl_nullable metadata) {
  if (model_name.empty()) {
    return absl::InvalidArgumentError("model_name must not be empty.");
  }

  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<OmniSessionFactory> session_factory,
                        OmniRegistry::Get().CreateSessionFactory(
                            model_name, settings, session_config, metadata));
  return std::unique_ptr<OmniEngine>(
      new OmniEngine(std::string(model_name), std::move(session_factory)));
}

absl::StatusOr<std::unique_ptr<OmniSession>> OmniEngine::CreateSession(
    std::unique_ptr<OmniSession::InputSource> absl_nonnull input_source) {
  return session_factory_->Create(std::move(input_source));
}

}  // namespace litert::omni
