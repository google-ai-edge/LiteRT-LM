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

#include "omni/asr/asr_session.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "omni/asr/text_merger.h"
#include "omni/base/async_stage_scheduler.h"
#include "omni/base/stage.h"
#include "runtime/framework/threadpool.h"

namespace litert::omni::asr {

absl::StatusOr<std::unique_ptr<AsrSession>> AsrSession::Create(
    Components components) {
  if (components.audio_source == nullptr) {
    return absl::InvalidArgumentError("AudioSource component is required.");
  }
  if (components.preprocessor == nullptr) {
    return absl::InvalidArgumentError(
        "AudioPreprocessor component is required.");
  }
  if (components.speech_recognizer == nullptr) {
    return absl::InvalidArgumentError(
        "SpeechRecognizer component is required.");
  }
  if (components.detokenizer == nullptr) {
    return absl::InvalidArgumentError("Detokenizer component is required.");
  }
  if (components.text_merger == nullptr) {
    return absl::InvalidArgumentError("TextMerger component is required.");
  }
  return std::unique_ptr<AsrSession>(new AsrSession(std::move(components)));
}

AsrSession::AsrSession(Components components)
    : components_(std::move(components)) {}

AsrSession::~AsrSession() { ResetAsyncScheduler(); }

void AsrSession::ResetAsyncScheduler() {
  absl::MutexLock lock(mutex_);
  if (async_scheduler_) {
    // 3 seconds is a arbitrary timeout to wait for the scheduler to finish, but
    // not too long. Better to crash than to wait too long.
    ABSL_CHECK_OK(async_scheduler_->Stop(absl::Seconds(3)));
  }
}

void AsrSession::Reset() {
  ResetAsyncScheduler();
  components_.audio_source->Reset();
  components_.preprocessor->Reset();
  components_.speech_recognizer->Reset();
  components_.detokenizer->Reset();
  components_.text_merger->Reset();
}

absl::StatusOr<TextMerger::MergeResult> AsrSession::ProcessNextChunk() {
  ABSL_RETURN_IF_ERROR(components_.audio_source->Schedule());

  if (components_.preprocessor->NeedSchedule()) {
    ABSL_RETURN_IF_ERROR(components_.preprocessor->Schedule());
  }

  if (components_.speech_recognizer->NeedSchedule()) {
    ABSL_RETURN_IF_ERROR(components_.speech_recognizer->Schedule());
  }

  if (components_.detokenizer->NeedSchedule()) {
    ABSL_RETURN_IF_ERROR(components_.detokenizer->Schedule());
  }

  if (components_.text_merger->NeedSchedule()) {
    ABSL_RETURN_IF_ERROR(components_.text_merger->Schedule());
  }

  return components_.text_merger->GetOutput();
}

absl::Status AsrSession::ProcessAsync(::litert::lm::ThreadPool& thread_pool,
                                      AsyncCallback callback) {
  absl::MutexLock lock(mutex_);
  if (async_scheduler_ != nullptr) {
    if (async_scheduler_->IsRunning()) {
      return absl::AlreadyExistsError("Async processing is already active.");
    }
    ABSL_RETURN_IF_ERROR(async_scheduler_->Stop(absl::Seconds(3)));
  }

  std::vector<internal::StageBase*> stages = {
      components_.audio_source.get(),      components_.preprocessor.get(),
      components_.speech_recognizer.get(), components_.detokenizer.get(),
      components_.text_merger.get(),
  };
  auto callback_with_flush_on_eos =
      [callback = std::move(callback),
       this](absl::StatusOr<TextMerger::MergeResult> result) mutable
      -> absl::Status {
    if (absl::IsOutOfRange(result.status())) {
      ABSL_RETURN_IF_ERROR(components_.text_merger->Flush());
      ABSL_RETURN_IF_ERROR(callback(components_.text_merger->GetOutput()));
    }
    return callback(std::move(result));
  };
  async_scheduler_ =
      std::make_unique<AsyncStageScheduler<TextMerger::MergeResult>>(
          std::move(stages), components_.text_merger.get(), &thread_pool,
          std::move(callback_with_flush_on_eos));
  return async_scheduler_->Start();
}

absl::StatusOr<TextMerger::MergeResult> AsrSession::Flush() {
  ABSL_RETURN_IF_ERROR(components_.text_merger->Flush());
  return components_.text_merger->GetOutput();
}

}  // namespace litert::omni::asr
