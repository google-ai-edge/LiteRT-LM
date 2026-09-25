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
#include <vector>

#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "absl/synchronization/notification.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "omni/base/async_stage_scheduler.h"
#include "omni/base/io_types.h"
#include "omni/base/stage.h"
#include "omni/omni_session.h"
#include "omni/tts/vocoder.h"
#include "runtime/framework/threadpool.h"

namespace litert::omni::tts {

absl::StatusOr<std::unique_ptr<TtsSession>> TtsSession::Create(
    Components components, ::litert::lm::ThreadPool* thread_pool) {
  if (thread_pool == nullptr) {
    return absl::InvalidArgumentError("ThreadPool is required.");
  }
  if (components.text_source == nullptr) {
    return absl::InvalidArgumentError("TextSource component is required.");
  }
  for (const auto& stage : components.intermediate_stages) {
    if (stage == nullptr) {
      return absl::InvalidArgumentError(
          "intermediate_stages contains null stage pointer.");
    }
  }
  if (components.vocoder == nullptr) {
    return absl::InvalidArgumentError("Vocoder component is required.");
  }
  return std::unique_ptr<TtsSession>(
      new TtsSession(std::move(components), thread_pool));
}

TtsSession::TtsSession(Components components,
                       ::litert::lm::ThreadPool* thread_pool)
    : components_(std::move(components)), thread_pool_(thread_pool) {}

TtsSession::~TtsSession() { ResetAsyncScheduler(); }

void TtsSession::ResetAsyncScheduler() {
  absl::MutexLock lock(mutex_);
  if (async_scheduler_) {
    absl::Status status = async_scheduler_->Stop(absl::Seconds(3));
    if (!status.ok()) {
      ABSL_LOG(ERROR) << "Failed to stop async scheduler: " << status;
    }
  }
}

void TtsSession::WaitForIdleOrStopped() {
  absl::MutexLock lock(mutex_);
  if (async_scheduler_) {
    absl::Status status =
        async_scheduler_->WaitForIdleOrStopped(absl::Seconds(3));
    if (!status.ok()) {
      ABSL_LOG(ERROR) << "Failed to wait for async scheduler: " << status;
    }
  }
}

void TtsSession::Reset() {
  ResetAsyncScheduler();
  components_.text_source->Reset();
  for (auto& stage : components_.intermediate_stages) {
    stage->Reset();
  }
  components_.vocoder->Reset();
}

absl::StatusOr<OmniSession::Output> TtsSession::ProcessNext() {
  // TODO(b/538727793): Remove Finish() and Reset() here and drive stages
  // inline.
  absl::Cleanup reset_cleanup = [this] { Reset(); };
  components_.text_source->Finish();

  AudioOutput result;
  absl::Notification done;
  absl::Status final_status;
  absl::Mutex mutex;

  ABSL_RETURN_IF_ERROR(
      ProcessAsync([&](absl::StatusOr<Output> output) -> absl::Status {
        if (absl::IsOutOfRange(output.status())) {
          if (result.pcm_samples.empty()) {
            final_status = output.status();
          }
          done.Notify();
          return output.status();
        }
        if (absl::IsNotFound(output.status())) {
          return absl::OkStatus();
        }
        if (!output.ok()) {
          final_status = output.status();
          done.Notify();
          return output.status();
        }
        const auto* audio = std::get_if<AudioOutput>(&*output);
        if (audio != nullptr) {
          absl::MutexLock lock(mutex);
          if (result.sample_rate_hz == 0) {
            result.sample_rate_hz = audio->sample_rate_hz;
          }
          result.pcm_samples.insert(result.pcm_samples.end(),
                                    audio->pcm_samples.begin(),
                                    audio->pcm_samples.end());
        }
        return absl::OkStatus();
      }));

  done.WaitForNotification();
  if (!final_status.ok()) {
    return final_status;
  }
  return result;
}

absl::Status TtsSession::ProcessAsync(OutputCallback callback) {
  if (thread_pool_ == nullptr) {
    return absl::FailedPreconditionError("ThreadPool is null.");
  }
  absl::MutexLock lock(mutex_);
  if (async_scheduler_ != nullptr) {
    if (async_scheduler_->IsRunning()) {
      return absl::AlreadyExistsError("Async processing is already active.");
    }
    ABSL_RETURN_IF_ERROR(async_scheduler_->Stop(absl::Seconds(3)));
  }

  std::vector<internal::StageBase*> stages;
  stages.reserve(2 + components_.intermediate_stages.size());
  stages.push_back(components_.text_source.get());
  for (const auto& stage : components_.intermediate_stages) {
    stages.push_back(stage.get());
  }
  stages.push_back(components_.vocoder.get());

  auto callback_with_flush_on_eos =
      [callback = std::move(callback),
       this](absl::StatusOr<AudioOutput> result) mutable -> absl::Status {
    if (absl::IsOutOfRange(result.status())) {
      ABSL_RETURN_IF_ERROR(components_.vocoder->Flush());
      while (components_.vocoder->HasOutput()) {
        auto out = components_.vocoder->GetOutput();
        if (out.ok()) {
          ABSL_RETURN_IF_ERROR(callback(std::move(out)));
        } else if (!absl::IsNotFound(out.status())) {
          return out.status();
        }
      }
    }
    return callback(std::move(result));
  };
  async_scheduler_ = std::make_unique<AsyncStageScheduler<AudioOutput>>(
      std::move(stages), components_.vocoder.get(), thread_pool_,
      std::move(callback_with_flush_on_eos));
  return async_scheduler_->Start();
}

absl::StatusOr<OmniSession::Output> TtsSession::Flush() {
  components_.text_source->Finish();
  WaitForIdleOrStopped();
  ABSL_RETURN_IF_ERROR(components_.vocoder->Flush());
  auto result = components_.vocoder->GetOutput();
  if (absl::IsNotFound(result.status())) {
    return AudioOutput();
  }
  return result;
}

}  // namespace litert::omni::tts
