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

#include "omni/tts/tts_session.h"

#include <memory>
#include <utility>
#include <variant>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/base/thread_annotations.h"  // from @com_google_absl
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "omni/omni_session.h"
#include "omni/tts/stream_text_source.h"
#include "omni/tts/text_chunk_utils.h"
#include "omni/tts/tts_engine.h"

namespace litert::omni::tts {
namespace {

// StreamTextSource subclass that pulls `TextInput` payloads from an
// `OmniSession::InputSource` and yields text chunks to an `OmniSession`.
class TextInputSource : public StreamTextSource {
 public:
  explicit TextInputSource(
      std::unique_ptr<OmniSession::InputSource> absl_nonnull input_source,
      TextChunkConfig config = {})
      : StreamTextSource(std::move(config)),
        input_source_(std::move(input_source)) {}

  ~TextInputSource() override = default;

 protected:
  void ResetInternal() override {
    input_source_->Reset();
    StreamTextSource::ResetInternal();
  }

  bool NeedScheduleInternal() const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_) override {
    return (!IsFinishedLocked() &&
            (input_source_->NeedSchedule() || input_source_->HasOutput())) ||
           StreamTextSource::NeedScheduleInternal();
  }

  absl::Status ScheduleInternal() ABSL_NO_THREAD_SAFETY_ANALYSIS override {
    SetState(State::kRunning);
    absl::Cleanup cleanup = [this] { SetState(State::kIdle); };

    ABSL_RETURN_IF_ERROR(DrainAvailableInputs());
    return StreamTextSource::ScheduleInternal();
  }

  absl::Status FlushInternal() ABSL_NO_THREAD_SAFETY_ANALYSIS override {
    ABSL_RETURN_IF_ERROR(input_source_->Flush());
    ABSL_RETURN_IF_ERROR(DrainAvailableInputs());
    return StreamTextSource::FlushInternal();
  }

 private:
  // Drains all currently available inputs from `input_source_` into
  // `StreamTextSource`, coalescing consecutive `TextInput`s and calling
  // `Finish()` when `EndOfInput` or source `OutOfRangeError` is reached.
  absl::Status DrainAvailableInputs() ABSL_NO_THREAD_SAFETY_ANALYSIS {
    while (!IsFinished()) {
      if (!input_source_->HasOutput()) {
        if (!input_source_->NeedSchedule()) {
          break;
        }
        absl::Status status = input_source_->Schedule();
        if (absl::IsOutOfRange(status)) {
          Finish();
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
        Finish();
        break;
      }
      const auto* text_input = std::get_if<OmniSession::TextInput>(&input);
      if (text_input == nullptr) {
        return absl::InvalidArgumentError("TTS Session requires TextInput.");
      }
      AppendText(text_input->text);
    }
    return absl::OkStatus();
  }

  std::unique_ptr<OmniSession::InputSource> absl_nonnull input_source_;
};

}  // namespace

std::unique_ptr<StreamTextSource> TtsSessionFactory::CreateTextInputSource(
    std::unique_ptr<OmniSession::InputSource> absl_nonnull input_source,
    TextChunkConfig config) {
  return std::make_unique<TextInputSource>(std::move(input_source),
                                           std::move(config));
}

absl::StatusOr<std::unique_ptr<OmniSessionFactory>>
TtsSessionFactory::CreateFactory(TtsEngineSettings settings) {
  ABSL_ASSIGN_OR_RETURN(auto tts_engine,
                        TtsEngine::Create(std::move(settings)));
  return std::unique_ptr<OmniSessionFactory>(
      new TtsSessionFactory(std::move(tts_engine)));
}

TtsSessionFactory::TtsSessionFactory(
    std::unique_ptr<TtsEngine> absl_nonnull tts_engine)
    : tts_engine_(std::move(tts_engine)) {}

TtsSessionFactory::~TtsSessionFactory() = default;

absl::StatusOr<std::unique_ptr<OmniSession>> TtsSessionFactory::Create(
    std::unique_ptr<OmniSession::InputSource> absl_nonnull input_source) {
  TtsSessionConfig session_config;
  auto text_source = CreateTextInputSource(
      std::move(input_source),
      tts_engine_->ResolveTextChunkConfig(session_config));
  return tts_engine_->CreateSession(session_config, std::move(text_source));
}

}  // namespace litert::omni::tts
