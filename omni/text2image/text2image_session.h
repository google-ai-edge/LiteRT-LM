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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2IMAGE_TEXT2IMAGE_SESSION_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2IMAGE_TEXT2IMAGE_SESSION_H_

#include <memory>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "omni/base/io_types.h"
#include "omni/base/model_resources.h"
#include "omni/multi_staged_session.h"
#include "omni/omni_engine.h"
#include "omni/omni_session.h"
#include "omni/text2image/prompt_source.h"
#include "runtime/framework/threadpool.h"
#include "runtime/proto/text2image_metadata.pb.h"
#include "runtime/proto/text2image_session_config.pb.h"

namespace litert::omni {
class OmniSessionTest;
}  // namespace litert::omni

namespace litert::omni::text2image {

// Base `OmniSessionFactory` implementation for Text-to-Image models.
class Text2ImageSessionFactory : public OmniSessionFactory {
 public:
  Text2ImageSessionFactory(
      const OmniEngine::Settings& settings,
      const MessageLite* absl_nullable session_config,
      const lm::proto::Text2ImageMetadata& metadata,
      std::shared_ptr<ModelResources> absl_nullable model_resources);

  ~Text2ImageSessionFactory() override;

  absl::StatusOr<std::unique_ptr<OmniSession>> Create(
      std::unique_ptr<OmniSession::InputSource> absl_nonnull input_source)
      override;

  const lm::proto::Text2ImageMetadata& metadata() const { return metadata_; }
  const lm::proto::Text2ImageSessionConfig& session_config() const {
    return session_config_;
  }
  const OmniEngine::Settings& engine_settings() const {
    return engine_settings_;
  }
  std::shared_ptr<ModelResources> model_resources() const {
    return model_resources_;
  }

  // Resolves the default `ImageGenInputMetadata` parameters for
  // `session_config`, applying any model-specific defaults.
  virtual ImageGenInputMetadata ResolveDefaultPromptParams(
      const lm::proto::Text2ImageSessionConfig& session_config = {}) const = 0;

 protected:
  virtual absl::StatusOr<std::unique_ptr<MultiStagedSession>>
  CreateWithPromptSource(
      const lm::proto::Text2ImageSessionConfig& session_config,
      std::unique_ptr<PromptSource> absl_nullable prompt_source) = 0;

  lm::ThreadPool* thread_pool() const { return thread_pool_.get(); }

 private:
  friend class ::litert::omni::OmniSessionTest;
  friend class Text2ImageSessionTest;

  static std::unique_ptr<PromptSource> CreatePromptInputSource(
      std::unique_ptr<OmniSession::InputSource> absl_nonnull input_source,
      ImageGenInputMetadata default_params = {});

  OmniEngine::Settings engine_settings_;
  lm::proto::Text2ImageSessionConfig session_config_;
  lm::proto::Text2ImageMetadata metadata_;
  std::shared_ptr<ModelResources> absl_nullable model_resources_;
  std::unique_ptr<lm::ThreadPool> absl_nonnull thread_pool_;
};

}  // namespace litert::omni::text2image

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2IMAGE_TEXT2IMAGE_SESSION_H_
