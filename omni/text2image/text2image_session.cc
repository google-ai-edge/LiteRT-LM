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

#include <memory>
#include <utility>
#include <variant>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "omni/base/io_types.h"
#include "omni/base/model_resources.h"
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

absl::StatusOr<std::unique_ptr<OmniSession>> Text2ImageSessionFactory::Create(
    std::unique_ptr<OmniSession::InputSource> absl_nonnull input_source) {
  auto prompt_source = CreatePromptInputSource(
      std::move(input_source), ResolveDefaultPromptParams(session_config_));
  return CreateWithPromptSource(session_config_, std::move(prompt_source));
}

}  // namespace litert::omni::text2image
