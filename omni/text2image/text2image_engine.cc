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

#include "omni/text2image/text2image_engine.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "omni/base/io_types.h"
#include "omni/multi_staged_session.h"
#include "omni/omni_engine.h"
#include "omni/omni_registry.h"
#include "omni/omni_session.h"
#include "omni/text2image/prompt_source.h"
#include "omni/text2image/text2image_session.h"
#include "runtime/proto/text2image_session_config.pb.h"

namespace litert::omni::text2image {

absl::StatusOr<std::unique_ptr<Text2ImageEngine>> Text2ImageEngine::Create(
    absl::string_view model_name, Settings settings) {
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<OmniSessionFactory> session_factory,
                        OmniRegistry::Get().CreateSessionFactory(
                            model_name, settings.engine_settings,
                            &settings.session_config, &settings.metadata));
  if (session_factory == nullptr) {
    return absl::InternalError(
        "OmniRegistry returned null OmniSessionFactory.");
  }
  // NOTE: `static_cast<Text2ImageSessionFactory*>` is acceptable here because
  // all built-in Text2Image factories inherit from `Text2ImageSessionFactory`,
  // and `Text2ImageEngine` is transitional and will be removed once all clients
  // migrate to the unified `OmniEngine` API.
  auto* t2i_factory =
      static_cast<Text2ImageSessionFactory*>(session_factory.get());
  settings.engine_settings = t2i_factory->engine_settings();
  settings.metadata = t2i_factory->metadata();
  settings.session_config = t2i_factory->session_config();

  const std::string effective_model_name =
      model_name.empty() ? "text2image" : std::string(model_name);
  ABSL_ASSIGN_OR_RETURN(
      std::unique_ptr<OmniEngine> omni_engine,
      OmniEngine::Create(effective_model_name, std::move(session_factory)));
  return std::unique_ptr<Text2ImageEngine>(new Text2ImageEngine(
      std::move(settings), std::move(omni_engine), t2i_factory));
}

Text2ImageEngine::Text2ImageEngine(
    Settings settings, std::unique_ptr<OmniEngine> absl_nonnull omni_engine,
    Text2ImageSessionFactory* absl_nonnull text2image_session_factory)
    : settings_(std::move(settings)),
      omni_engine_(std::move(omni_engine)),
      text2image_session_factory_(text2image_session_factory) {}

ImageGenInputMetadata Text2ImageEngine::ResolveDefaultPromptParams(
    const lm::proto::Text2ImageSessionConfig& session_config) const {
  return text2image_session_factory_->ResolveDefaultPromptParams(
      session_config);
}

absl::StatusOr<std::unique_ptr<MultiStagedSession>>
Text2ImageEngine::CreateSession(
    const lm::proto::Text2ImageSessionConfig& session_config,
    std::unique_ptr<PromptSource> absl_nullable prompt_source) {
  return text2image_session_factory_->CreateWithPromptSource(
      session_config, std::move(prompt_source));
}

}  // namespace litert::omni::text2image
