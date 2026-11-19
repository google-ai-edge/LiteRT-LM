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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_ANY2ANY_SESSION_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_ANY2ANY_SESSION_H_

#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
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

// Base `OmniSessionFactory` implementation for Any-to-Any multimodal models.
class Any2AnySessionFactory : public OmniSessionFactory {
 public:
  ~Any2AnySessionFactory() override;

  absl::StatusOr<std::unique_ptr<OmniSession>> Create(
      std::unique_ptr<OmniSession::InputSource> absl_nonnull input_source)
      override;

  const lm::proto::Any2AnyMetadata& metadata() const { return metadata_; }
  const lm::proto::Any2AnySessionConfig& session_config() const {
    return session_config_;
  }
  const OmniEngine::Settings& engine_settings() const {
    return engine_settings_;
  }
  std::shared_ptr<ModelResources> model_resources() const {
    return model_resources_;
  }
  const std::vector<std::string>& available_voices() const {
    return available_voices_;
  }

  std::vector<std::string> GetAvailableLanguages() const;
  // Returns true if `voice` matches an entry in `available_voices()`
  // (case-insensitive).
  bool HasVoice(absl::string_view voice) const;
  virtual std::string GetDefaultVoice(absl::string_view language) const;

  lm::proto::Any2AnySessionConfig MergeSessionConfig(
      const lm::proto::Any2AnySessionConfig& override_config) const;

  // Returns the target input audio sample rate in Hz (default 16000).
  virtual int GetInputSampleRateHz() const;

 protected:
  Any2AnySessionFactory(
      const OmniEngine::Settings& settings,
      const MessageLite* absl_nullable session_config,
      const lm::proto::Any2AnyMetadata& metadata,
      std::vector<std::string> available_voices = {},
      std::shared_ptr<ModelResources> absl_nullable model_resources = nullptr);

  // Must be implemented by subclasses (e.g. `LeasedAny2AnySessionFactory`) to
  // construct model-specific pipeline stages connected to `input_source` and
  // return a `MultiStagedSession`.
  virtual absl::StatusOr<std::unique_ptr<MultiStagedSession>>
  CreateWithInputSource(
      const lm::proto::Any2AnySessionConfig& session_config,
      std::unique_ptr<OmniSession::InputSource> absl_nonnull input_source) = 0;

  lm::ThreadPool* thread_pool() const { return thread_pool_.get(); }

 private:
  friend class Any2AnySessionTest;

  OmniEngine::Settings engine_settings_;
  lm::proto::Any2AnySessionConfig session_config_;
  lm::proto::Any2AnyMetadata metadata_;
  std::vector<std::string> available_voices_;
  std::shared_ptr<ModelResources> absl_nullable model_resources_;
  std::unique_ptr<lm::ThreadPool> absl_nonnull thread_pool_;
};

// Resolves `Any2AnyMetadata` from the `metadata` proto (already merged with
// bundle-embedded metadata by `OmniRegistry::CreateSessionFactory`), populating
// default supported languages (`"en-US"`, `"zh-CN"`) if none are specified.
lm::proto::Any2AnyMetadata ResolveAny2AnyMetadata(
    const MessageLite* absl_nullable metadata);

// Options configuring a `LeasedAny2AnySessionFactory`.
struct LeasedAny2AnySessionOptions {
  // Human-readable model name included in concurrent-session error messages.
  std::string model_display_name = "Any2Any";
  // Fallback voice returned by `GetDefaultVoice` when `available_voices()` is
  // empty.
  std::string fallback_default_voice;
};

// `Any2AnySessionFactory` implementation that enforces a single active session
// lease per engine instance (protecting shared stateful model runners) and
// delegates stage graph construction to a model-specific `ComponentBuilder`.
class LeasedAny2AnySessionFactory : public Any2AnySessionFactory {
 public:
  // Builds model-specific pipeline stages for a session from the merged
  // `session_config`, `input_source`, and shared `model_resources`. Must append
  // all stages (with `input_source` first and the returned output stage last)
  // to `stages`.
  using ComponentBuilder =
      absl::AnyInvocable<absl::StatusOr<Stage<Output>* absl_nonnull>(
          const lm::proto::Any2AnySessionConfig& session_config,
          std::unique_ptr<OmniSession::InputSource> absl_nonnull input_source,
          std::shared_ptr<ModelResources> absl_nonnull model_resources,
          std::vector<std::unique_ptr<internal::StageBase>>& stages) const>;

  LeasedAny2AnySessionFactory(
      const OmniEngine::Settings& settings,
      const MessageLite* absl_nullable session_config,
      const lm::proto::Any2AnyMetadata& metadata,
      std::vector<std::string> available_voices,
      std::shared_ptr<ModelResources> absl_nullable model_resources,
      ComponentBuilder component_builder,
      LeasedAny2AnySessionOptions options = {});

  std::string GetDefaultVoice(absl::string_view language) const override;

 protected:
  absl::StatusOr<std::unique_ptr<MultiStagedSession>> CreateWithInputSource(
      const lm::proto::Any2AnySessionConfig& session_config,
      std::unique_ptr<OmniSession::InputSource> absl_nonnull input_source)
      override;

 private:
  ComponentBuilder component_builder_;
  std::string model_display_name_;
  std::string fallback_default_voice_;
  std::shared_ptr<std::atomic<bool>> active_session_;
};

}  // namespace litert::omni::any2any

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_ANY2ANY_SESSION_H_
