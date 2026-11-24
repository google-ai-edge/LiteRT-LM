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

#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "omni/base/io_types.h"
#include "omni/base/model_resources.h"
#include "omni/base/stage.h"
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

  lm::proto::Text2ImageSessionConfig MergeSessionConfig(
      const lm::proto::Text2ImageSessionConfig& override_config) const;

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

// Resolves `ImageGenInputMetadata` prompt parameters from `override_config`,
// falling back to `base_config` and finally `model_defaults` when width,
// height, or `num_inference_steps` are unset/non-positive or when `seed` is
// unset.
ImageGenInputMetadata ResolveDefaultPromptParams(
    const lm::proto::Text2ImageSessionConfig& base_config,
    const lm::proto::Text2ImageSessionConfig& override_config,
    const ImageGenInputMetadata& model_defaults);

// Options configuring a `LeasedText2ImageSessionFactory`.
struct LeasedText2ImageSessionOptions {
  // Human-readable model name included in concurrent-session error messages.
  std::string model_display_name = "Text2Image";
  // Fallback prompt parameters applied when neither the engine config nor
  // session override sets positive width/height/steps or an explicit seed.
  ImageGenInputMetadata default_prompt_params;
};

// `Text2ImageSessionFactory` implementation that enforces a single active
// session lease per engine instance (protecting shared compiled model
// resources) and delegates stage graph construction to a model-specific
// `ComponentBuilder`.
class LeasedText2ImageSessionFactory : public Text2ImageSessionFactory {
 public:
  // Builds model-specific pipeline stages for a session from the merged
  // `session_config` (whose `seed` may be unset when neither the engine config
  // nor per-session override explicitly specifies one), non-null
  // `prompt_source`, and shared `model_resources`. Must append all stages
  // (with `prompt_source` first and the returned output stage last) to
  // `stages`.
  using ComponentBuilder =
      absl::AnyInvocable<absl::StatusOr<Stage<Output>* absl_nonnull>(
          const lm::proto::Text2ImageSessionConfig& session_config,
          std::unique_ptr<PromptSource> absl_nonnull prompt_source,
          std::shared_ptr<ModelResources> absl_nonnull model_resources,
          std::vector<std::unique_ptr<internal::StageBase>>& stages) const>;

  LeasedText2ImageSessionFactory(
      const OmniEngine::Settings& settings,
      const MessageLite* absl_nullable session_config,
      const lm::proto::Text2ImageMetadata& metadata,
      std::shared_ptr<ModelResources> absl_nullable model_resources,
      ComponentBuilder component_builder,
      LeasedText2ImageSessionOptions options = {});

  ImageGenInputMetadata ResolveDefaultPromptParams(
      const lm::proto::Text2ImageSessionConfig& session_config) const override;

 protected:
  absl::StatusOr<std::unique_ptr<MultiStagedSession>> CreateWithPromptSource(
      const lm::proto::Text2ImageSessionConfig& session_config,
      std::unique_ptr<PromptSource> absl_nullable prompt_source) override;

 private:
  ComponentBuilder component_builder_;
  std::string model_display_name_;
  ImageGenInputMetadata default_prompt_params_;
  bool engine_has_explicit_seed_ = false;
  std::shared_ptr<std::atomic<bool>> active_session_;
};

}  // namespace litert::omni::text2image

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2IMAGE_TEXT2IMAGE_SESSION_H_
