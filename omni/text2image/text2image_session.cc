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

#include "omni/text2image/text2image_session.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
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

namespace litert::omni::text2image {
namespace {

lm::proto::Text2ImageSessionConfig ResolveText2ImageSessionConfig(
    const MessageLite* absl_nullable session_config) {
  lm::proto::Text2ImageSessionConfig resolved;
  if (session_config != nullptr) {
    resolved.CheckTypeAndMergeFrom(*session_config);
  }
  if (!resolved.has_seed()) {
    resolved.set_seed(42);
  }
  return resolved;
}

bool HasExplicitSeed(const MessageLite* absl_nullable session_config) {
  if (session_config == nullptr) {
    return false;
  }
  lm::proto::Text2ImageSessionConfig parsed;
  parsed.CheckTypeAndMergeFrom(*session_config);
  return parsed.has_seed();
}

// PromptSource implementation that pulls `ImageGenInputMetadata` and
// `TextInput` payloads from an `OmniSession::InputSource` and yields
// `Text2ImagePrompt` requests to a `MultiStagedSession`.
class PromptInputSource : public PromptSource {
 public:
  explicit PromptInputSource(
      std::unique_ptr<OmniSession::InputSource> absl_nonnull input_source,
      ImageGenInputMetadata default_params = {})
      : input_source_(std::move(input_source)),
        default_params_(std::move(default_params)),
        current_params_(default_params_) {}

  ~PromptInputSource() override = default;

  // TODO(b/568027544): Return absl::Status from Finish().
  void Finish() override { ABSL_CHECK_OK(Flush()); }

 protected:
  absl::Status FlushInternal() override {
    ABSL_RETURN_IF_ERROR(input_source_->Flush());
    ABSL_RETURN_IF_ERROR(DrainAvailableInputs());
    is_finished_ = true;
    return absl::OkStatus();
  }

  void ResetInternal() override {
    input_source_->Reset();
    current_params_ = default_params_;
    is_finished_ = false;
  }

  bool NeedScheduleInternal() const override {
    return !is_finished_ &&
           (input_source_->NeedSchedule() || input_source_->HasOutput());
  }

  absl::Status ScheduleInternal() override {
    SetState(State::kRunning);
    absl::Cleanup cleanup = [this] { SetState(State::kIdle); };

    ABSL_RETURN_IF_ERROR(DrainAvailableInputs());
    if (HasOutput()) {
      return absl::OkStatus();
    }
    if (is_finished_) {
      return absl::OutOfRangeError("End of prompt stream reached.");
    }
    return absl::OkStatus();
  }

 private:
  // Drains all currently available inputs from `input_source_` into output
  // prompts, updating `current_params_` on `ImageGenInputMetadata` and marking
  // `is_finished_` when `EndOfInput` or source `OutOfRangeError` is reached.
  absl::Status DrainAvailableInputs() {
    while (!is_finished_) {
      if (!input_source_->HasOutput()) {
        if (!input_source_->NeedSchedule()) {
          break;
        }
        absl::Status status = input_source_->Schedule();
        if (absl::IsOutOfRange(status)) {
          is_finished_ = true;
          break;
        }
        if (absl::IsNotFound(status)) {
          break;
        }
        ABSL_RETURN_IF_ERROR(status);
        if (!input_source_->HasOutput()) {
          break;
        }
      }
      ABSL_ASSIGN_OR_RETURN(OmniSession::Input input,
                            input_source_->GetOutput());
      if (std::holds_alternative<OmniSession::EndOfInput>(input)) {
        is_finished_ = true;
        break;
      }
      if (const auto* meta =
              std::get_if<OmniSession::ImageGenInputMetadata>(&input)) {
        current_params_ = *meta;
        continue;
      }
      const auto* text_input = std::get_if<OmniSession::TextInput>(&input);
      if (text_input == nullptr) {
        return absl::InvalidArgumentError(
            "Text2Image Session requires TextInput or ImageGenInputMetadata.");
      }
      if (text_input->text.empty()) {
        return absl::InvalidArgumentError("Prompt text must not be empty.");
      }
      PushOutput(Text2ImagePrompt{.metadata = current_params_,
                                  .text = text_input->text});
    }
    return absl::OkStatus();
  }

  std::unique_ptr<OmniSession::InputSource> absl_nonnull input_source_;
  const ImageGenInputMetadata default_params_;
  ImageGenInputMetadata current_params_;
  bool is_finished_ = false;
};

// RAII guard releasing the owning factory's active-session lease on
// destruction.
// TODO(b/568027544): Extract a shared leased `MultiStagedSession` helper in
// `omni/` and share it with `any2any/any2any_session.cc` (preserving
// post-stage-destruction lease release ordering).
class ActiveSessionLease {
 public:
  explicit ActiveSessionLease(std::shared_ptr<std::atomic<bool>> active_session)
      : active_session_(std::move(active_session)) {}

  ActiveSessionLease(const ActiveSessionLease&) = delete;
  ActiveSessionLease& operator=(const ActiveSessionLease&) = delete;

  ~ActiveSessionLease() {
    if (active_session_ != nullptr) {
      active_session_->store(false, std::memory_order_release);
    }
  }

 private:
  std::shared_ptr<std::atomic<bool>> active_session_;
};

// `ActiveSessionLease` precedes `MultiStagedSession` in the base-class list so
// base destructors run in reverse order: `~LeasedText2ImageSession()` resets
// stages, `~MultiStagedSession()` stops the async scheduler and destroys
// `stages_` (`TensorBuffer`s and `LiteRtRunner`s), and only then does
// `~ActiveSessionLease()` release `active_session_`.
class LeasedText2ImageSession : private ActiveSessionLease,
                                public MultiStagedSession {
 public:
  LeasedText2ImageSession(
      std::vector<std::unique_ptr<internal::StageBase>> stages,
      Stage<Output>* absl_nonnull output_stage,
      lm::ThreadPool* absl_nullable thread_pool,
      std::shared_ptr<std::atomic<bool>> active_session)
      : ActiveSessionLease(std::move(active_session)),
        MultiStagedSession(std::move(stages), output_stage, thread_pool) {}

  ~LeasedText2ImageSession() override { Reset(); }
};

}  // namespace

std::unique_ptr<PromptSource> Text2ImageSessionFactory::CreatePromptInputSource(
    std::unique_ptr<OmniSession::InputSource> absl_nonnull input_source,
    ImageGenInputMetadata default_params) {
  return std::make_unique<PromptInputSource>(std::move(input_source),
                                             std::move(default_params));
}

Text2ImageSessionFactory::Text2ImageSessionFactory(
    const OmniEngine::Settings& settings,
    const MessageLite* absl_nullable session_config,
    const lm::proto::Text2ImageMetadata& metadata,
    std::shared_ptr<ModelResources> absl_nullable model_resources)
    : engine_settings_(settings),
      session_config_(ResolveText2ImageSessionConfig(session_config)),
      metadata_(metadata),
      model_resources_(std::move(model_resources)),
      thread_pool_(std::make_unique<lm::ThreadPool>(
          "text2image_engine_pool",
          settings.num_threads > 0 ? settings.num_threads : 4)) {}

Text2ImageSessionFactory::~Text2ImageSessionFactory() = default;

lm::proto::Text2ImageSessionConfig Text2ImageSessionFactory::MergeSessionConfig(
    const lm::proto::Text2ImageSessionConfig& override_config) const {
  lm::proto::Text2ImageSessionConfig merged = session_config_;
  merged.MergeFrom(override_config);
  return merged;
}

absl::StatusOr<std::unique_ptr<OmniSession>> Text2ImageSessionFactory::Create(
    std::unique_ptr<OmniSession::InputSource> absl_nonnull input_source) {
  // Pass an empty per-session override so subclasses can distinguish explicit
  // per-session fields from base `session_config_` defaults (notably `seed`).
  auto prompt_source = CreatePromptInputSource(std::move(input_source),
                                               ResolveDefaultPromptParams({}));
  return CreateWithPromptSource({}, std::move(prompt_source));
}

ImageGenInputMetadata ResolveDefaultPromptParams(
    const lm::proto::Text2ImageSessionConfig& base_config,
    const lm::proto::Text2ImageSessionConfig& override_config,
    const ImageGenInputMetadata& model_defaults) {
  ImageGenInputMetadata params{
      .width = override_config.has_width() ? override_config.width()
                                           : base_config.width(),
      .height = override_config.has_height() ? override_config.height()
                                             : base_config.height(),
      .num_inference_steps = override_config.has_num_inference_steps()
                                 ? override_config.num_inference_steps()
                                 : base_config.num_inference_steps(),
      .seed = override_config.has_seed()
                  ? static_cast<uint64_t>(override_config.seed())
                  : (base_config.has_seed()
                         ? static_cast<uint64_t>(base_config.seed())
                         : model_defaults.seed),
  };
  if (params.width <= 0) {
    params.width = model_defaults.width;
  }
  if (params.height <= 0) {
    params.height = model_defaults.height;
  }
  if (params.num_inference_steps <= 0) {
    params.num_inference_steps = model_defaults.num_inference_steps;
  }
  return params;
}

LeasedText2ImageSessionFactory::LeasedText2ImageSessionFactory(
    const OmniEngine::Settings& settings,
    const MessageLite* absl_nullable session_config,
    const lm::proto::Text2ImageMetadata& metadata,
    std::shared_ptr<ModelResources> absl_nullable model_resources,
    ComponentBuilder component_builder, LeasedText2ImageSessionOptions options)
    : Text2ImageSessionFactory(settings, session_config, metadata,
                               std::move(model_resources)),
      component_builder_(std::move(component_builder)),
      model_display_name_(std::move(options.model_display_name)),
      default_prompt_params_(std::move(options.default_prompt_params)),
      engine_has_explicit_seed_(HasExplicitSeed(session_config)),
      active_session_(std::make_shared<std::atomic<bool>>(false)) {}

ImageGenInputMetadata
LeasedText2ImageSessionFactory::ResolveDefaultPromptParams(
    const lm::proto::Text2ImageSessionConfig& session_config) const {
  lm::proto::Text2ImageSessionConfig base_config = this->session_config();
  if (!engine_has_explicit_seed_) {
    base_config.clear_seed();
  }
  return litert::omni::text2image::ResolveDefaultPromptParams(
      base_config, session_config, default_prompt_params_);
}

absl::StatusOr<std::unique_ptr<MultiStagedSession>>
LeasedText2ImageSessionFactory::CreateWithPromptSource(
    const lm::proto::Text2ImageSessionConfig& session_config,
    std::unique_ptr<PromptSource> absl_nullable prompt_source) {
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
        model_display_name_.empty() ? absl::string_view("Text2Image")
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

  lm::proto::Text2ImageSessionConfig effective =
      MergeSessionConfig(session_config);
  if (!engine_has_explicit_seed_ && !session_config.has_seed()) {
    effective.clear_seed();
  }
  if (prompt_source == nullptr) {
    prompt_source = std::make_unique<PushPromptSource>(
        ResolveDefaultPromptParams(session_config));
  }

  std::vector<std::unique_ptr<internal::StageBase>> stages;
  ABSL_ASSIGN_OR_RETURN(Stage<Output> * output_stage,
                        component_builder_(effective, std::move(prompt_source),
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

  std::unique_ptr<MultiStagedSession> session =
      std::make_unique<LeasedText2ImageSession>(std::move(stages), output_stage,
                                                thread_pool(), active_session_);
  lease_transferred = true;
  return session;
}

}  // namespace litert::omni::text2image
