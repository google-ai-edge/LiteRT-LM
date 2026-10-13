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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_ASR_ENGINE_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_ASR_ENGINE_H_

#include <memory>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "omni/asr/asr_session.h"
#include "omni/asr/audio_source.h"
#include "omni/multi_staged_session.h"
#include "omni/omni_engine.h"
#include "runtime/proto/asr_metadata.pb.h"
#include "runtime/proto/asr_session_config.pb.h"

namespace litert::omni::asr {

// Thin ASR wrapper around `OmniEngine`.
// TODO: Remove AsrEngine once TTS and Text2Image move to static registry.
class AsrEngine {
 public:
  struct Settings {
    OmniEngine::Settings engine_settings;
    lm::proto::AsrSessionConfig session_config;
    lm::proto::AsrMetadata metadata;
  };

  static absl::StatusOr<std::unique_ptr<AsrEngine>> Create(
      absl::string_view model_name, Settings settings = {});

  ~AsrEngine() = default;

  absl::StatusOr<std::unique_ptr<MultiStagedSession>> CreateSession(
      std::unique_ptr<AudioSource> absl_nonnull audio_source);

  const Settings& settings() const { return settings_; }

 private:
  AsrEngine(Settings settings,
            std::unique_ptr<OmniEngine> absl_nonnull omni_engine,
            AsrSessionFactory* absl_nonnull asr_session_factory);

  Settings settings_;
  std::unique_ptr<OmniEngine> absl_nonnull omni_engine_;
  AsrSessionFactory* absl_nonnull asr_session_factory_;
};

}  // namespace litert::omni::asr

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_ASR_ENGINE_H_
