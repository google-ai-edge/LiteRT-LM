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

#include "omni/asr/asr_engine.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "omni/asr/asr_session.h"
#include "omni/asr/audio_source.h"
#include "omni/multi_staged_session.h"
#include "omni/omni_engine.h"
#include "omni/omni_registry.h"
#include "omni/omni_session.h"

namespace litert::omni::asr {

absl::StatusOr<std::unique_ptr<AsrEngine>> AsrEngine::Create(
    absl::string_view model_name, Settings settings) {
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<OmniSessionFactory> session_factory,
                        OmniRegistry::Get().CreateSessionFactory(
                            model_name, settings.engine_settings,
                            &settings.session_config, &settings.metadata));
  if (session_factory == nullptr) {
    return absl::InternalError(
        "OmniRegistry returned null OmniSessionFactory.");
  }
  // NOTE: `static_cast<AsrSessionFactory*>` is acceptable here because all
  // built-in ASR factories inherit from `AsrSessionFactory`, and `AsrEngine`
  // is transitional and will be removed soon once all clients migrate to the
  // unified `OmniEngine` API.
  auto* asr_factory = static_cast<AsrSessionFactory*>(session_factory.get());
  settings.metadata = asr_factory->metadata();
  settings.session_config = asr_factory->session_config();

  const std::string effective_model_name =
      model_name.empty() ? "asr" : std::string(model_name);
  ABSL_ASSIGN_OR_RETURN(
      std::unique_ptr<OmniEngine> omni_engine,
      OmniEngine::Create(effective_model_name, std::move(session_factory)));
  return std::unique_ptr<AsrEngine>(
      new AsrEngine(std::move(settings), std::move(omni_engine), asr_factory));
}

AsrEngine::AsrEngine(Settings settings,
                     std::unique_ptr<OmniEngine> absl_nonnull omni_engine,
                     AsrSessionFactory* absl_nonnull asr_session_factory)
    : settings_(std::move(settings)),
      omni_engine_(std::move(omni_engine)),
      asr_session_factory_(asr_session_factory) {}

absl::StatusOr<std::unique_ptr<MultiStagedSession>> AsrEngine::CreateSession(
    std::unique_ptr<AudioSource> absl_nonnull audio_source) {
  return asr_session_factory_->CreateWithAudioSource(std::move(audio_source));
}

}  // namespace litert::omni::asr
