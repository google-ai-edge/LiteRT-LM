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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2IMAGE_TEXT2IMAGE_ENGINE_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2IMAGE_TEXT2IMAGE_ENGINE_H_

#include <memory>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "omni/base/io_types.h"
#include "omni/base/model_resources.h"
#include "omni/multi_staged_session.h"
#include "omni/omni_engine.h"
#include "omni/text2image/prompt_source.h"
#include "omni/text2image/text2image_session.h"
#include "runtime/proto/text2image_metadata.pb.h"
#include "runtime/proto/text2image_session_config.pb.h"

namespace litert::omni::text2image {

// Thin Text-to-Image wrapper around `OmniEngine`.
// TODO: Remove Text2ImageEngine once all callers migrate to OmniEngine.
class Text2ImageEngine {
 public:
  struct Settings {
    OmniEngine::Settings engine_settings;
    lm::proto::Text2ImageSessionConfig session_config;
    lm::proto::Text2ImageMetadata metadata;
  };

  static absl::StatusOr<std::unique_ptr<Text2ImageEngine>> Create(
      absl::string_view model_name, Settings settings = {});

  ~Text2ImageEngine() = default;

  // Resolves the default `ImageGenInputMetadata` parameters for
  // `session_config`, applying any model-specific defaults.
  ImageGenInputMetadata ResolveDefaultPromptParams(
      const lm::proto::Text2ImageSessionConfig& session_config = {}) const;

  // Creates a lightweight `MultiStagedSession` for image generation.
  // The first stage (`stages()[0]`) is guaranteed to be a `PromptSource`.
  absl::StatusOr<std::unique_ptr<MultiStagedSession>> CreateSession(
      const lm::proto::Text2ImageSessionConfig& session_config = {},
      std::unique_ptr<PromptSource> absl_nullable prompt_source = nullptr);

  const Settings& settings() const { return settings_; }
  std::shared_ptr<ModelResources> model_resources() const {
    return text2image_session_factory_->model_resources();
  }

 private:
  Text2ImageEngine(
      Settings settings, std::unique_ptr<OmniEngine> absl_nonnull omni_engine,
      Text2ImageSessionFactory* absl_nonnull text2image_session_factory);

  Settings settings_;
  std::unique_ptr<OmniEngine> absl_nonnull omni_engine_;
  Text2ImageSessionFactory* absl_nonnull text2image_session_factory_;
};

}  // namespace litert::omni::text2image

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2IMAGE_TEXT2IMAGE_ENGINE_H_
