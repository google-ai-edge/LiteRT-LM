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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_BASE_ASYNC_STAGE_SCHEDULER_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_BASE_ASYNC_STAGE_SCHEDULER_H_

#include <algorithm>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/base/thread_annotations.h"  // from @com_google_absl
#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "omni/base/stage.h"
#include "runtime/framework/threadpool.h"

namespace litert::omni {

// Schedules stages asynchronously using a thread pool and calls a callback on
// error or when the stream ends.
template <typename T>
class AsyncStageScheduler {
 public:
  using AsyncCallback = absl::AnyInvocable<absl::Status(absl::StatusOr<T>)>;

  // Constructs an AsyncStageScheduler.
  //
  // Args:
  //   stages: A vector of stages to schedule including the output stage.
  //   output_stage: The output stage to get outputs from.
  //   thread_pool: The thread pool to use for scheduling.
  //   callback: The callback to call on error or when the stream ends.
  AsyncStageScheduler(std::vector<internal::StageBase* absl_nonnull> stages,
                      Stage<T>* absl_nonnull output_stage,
                      ::litert::lm::ThreadPool* absl_nonnull thread_pool,
                      AsyncCallback callback)
      : stages_(std::move(stages)),
        output_stage_(*output_stage),
        thread_pool_(*thread_pool),
        callback_(std::move(callback)) {}

  // Starts the async scheduler.
  // Returns absl::AlreadyExistsError if the scheduler is already running.
  absl::Status Start() {
    absl::MutexLock lock(mutex_);
    if (state_ != State::kNotStarted) {
      return absl::AlreadyExistsError("AsyncStageScheduler already started.");
    }
    state_ = State::kRunning;
    return thread_pool_.Schedule(
        [this]() { CallCallbackOnError(ScheduleReadyStages()); });
  }

  // Stops the async scheduler and waits for all stages to finish.
  // Returns absl::DeadlineExceededError if the timeout is reached before all
  // stages finish.
  absl::Status Stop(absl::Duration timeout) {
    absl::MutexLock lock(mutex_);
    if (state_ != State::kRunning && state_ != State::kStopping) {
      state_ = State::kStopped;
      return absl::OkStatus();
    }
    state_ = State::kStopping;

    auto condition = [this] {
      mutex_.AssertHeld();
      return !IsAnyStageRunning();
    };
    if (!mutex_.AwaitWithTimeout(absl::Condition(&condition), timeout)) {
      return absl::DeadlineExceededError(
          "Timeout reached while waiting for stages to finish.");
    }
    state_ = State::kStopped;
    return absl::OkStatus();
  }

  bool IsRunning() const {
    absl::MutexLock lock(mutex_);
    return state_ == State::kRunning;
  }

 private:
  enum class State { kNotStarted, kRunning, kStopping, kStopped };

  void CallCallbackOnError(absl::Status status) {
    // Lock mutex_ even when status is ok to inform schedule the state change.
    absl::MutexLock lock(mutex_);
    if (!status.ok() && state_ == State::kRunning && !callback_(status).ok()) {
      state_ = State::kStopping;
    }
  }

  // Schedules stage if it has work to do and calls callback_ on error.
  absl::Status ScheduleStageIfReady(internal::StageBase& stage) {
    if (!stage.NeedSchedule()) {
      return absl::OkStatus();
    }
    return thread_pool_.Schedule([this, &stage]() {
      if (!IsRunning() || !stage.NeedSchedule()) {
        return;
      }
      absl::Status status = stage.Schedule();
      if (!status.ok() && !absl::IsNotFound(status)) {
        CallCallbackOnError(status);
      }
    });
  }

  bool IsAnyStageReady() const {
    return output_stage_.HasOutput() ||
           std::any_of(stages_.begin(), stages_.end(),
                       [](internal::StageBase* stage) {
                         return stage->NeedSchedule();
                       });
  }

  bool IsAnyStageRunning() const {
    return std::any_of(
        stages_.begin(), stages_.end(),
        [](internal::StageBase* stage) { return !stage->IsIdle(); });
  }

  void WaitForAnyStagesReadyOrStopped(absl::Duration timeout) {
    auto condition = [this] {
      mutex_.AssertHeld();
      return state_ != State::kRunning || IsAnyStageReady() ||
             !IsAnyStageRunning();
    };
    absl::MutexLock lock(mutex_);
    mutex_.AwaitWithTimeout(absl::Condition(&condition), timeout);
  }

  absl::Status ScheduleReadyStages() {
    if (!IsRunning()) {
      return absl::CancelledError("AsyncStageScheduler not running.");
    } else if (!IsAnyStageReady() && !IsAnyStageRunning()) {
      return absl::OutOfRangeError("End of stream reached.");
    }

    // Process any available outputs.
    while (output_stage_.HasOutput()) {
      auto result = output_stage_.GetOutput();
      ABSL_RETURN_IF_ERROR(callback_(result));
    }

    for (internal::StageBase* stage : stages_) {
      ABSL_RETURN_IF_ERROR(ScheduleStageIfReady(*stage));
    }

    // Schedule self to run again after a short timeout to check for more work.
    return thread_pool_.Schedule([this]() {
      WaitForAnyStagesReadyOrStopped(absl::Seconds(1));
      CallCallbackOnError(ScheduleReadyStages());
    });
  }

  std::vector<internal::StageBase*> stages_;
  Stage<T>& output_stage_;
  ::litert::lm::ThreadPool& thread_pool_;
  AsyncCallback callback_;

  mutable absl::Mutex mutex_;
  State state_ ABSL_GUARDED_BY(mutex_) = State::kNotStarted;
};

}  // namespace litert::omni

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_BASE_ASYNC_STAGE_SCHEDULER_H_
