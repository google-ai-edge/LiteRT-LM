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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_BASE_STAGE_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_BASE_STAGE_H_

#include <deque>
#include <utility>

#include "absl/base/thread_annotations.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl

namespace litert::omni {
namespace internal {

// Base class for Stage below with abstract methods not dependent on output
// template parameter.
// Implementation must be thread-safe.
class StageBase {
 public:
  virtual ~StageBase() = default;

  // Determines if the stage doesn't do any jobs for now.
  //
  // When a stage is idle, NeedSchedule() may still be false if the stage
  // doesn't have any jobs to do. Or, NeedSchedule() may be true if the stage
  // has some jobs to do but not yet scheduled yet.
  //
  // When a stage is not idle, NeedSchedule() would likely be false as the stage
  // is already scheduled or running to handle its jobs.
  // But, NeedSchedule() may still be true if the stage has more jobs to
  // schedule simultaneously even when it's already running.
  virtual bool IsIdle() const = 0;

  // Determines if Schedule() should be called soon to process jobs. If a stage
  // is multi-threaded, it may return true even if this stage is not idle.
  virtual bool NeedSchedule() const = 0;

  // Schedules and executes jobs of this stage on the current calling thread.
  // Returns absl::NotFoundError if there are no jobs to execute.
  virtual absl::Status Schedule() = 0;

  // Indicates whether the stage's output queue contains 1 or more outputs in a
  // thread-safe manner.
  virtual bool HasOutput() const = 0;

  // Resets internal cached state and clears outputs for a new stream/session.
  virtual void Reset() = 0;
};

}  // namespace internal

// Abstract base class template for all omni models' stages producing outputs
// of type T.
// Implementation must be thread-safe.
template <typename T>
class Stage : public internal::StageBase {
 public:
  ~Stage() override = default;

  // See StageBase above for more methods.

  // Returns the next available output object T in a thread-safe manner, or
  // absl::NotFoundError if no output is currently available in the queue.
  virtual absl::StatusOr<T> GetOutput() = 0;
};

// Base class template for stages that
// 1) can be scheduled on a single thread synchronously or asynchronously by
//    AsyncStageScheduler
// 2) manages its outputs using an internal thread-safe deque.
//
// Subclasses must implement the logic to populate the deque in
// ScheduleInternal() or jobs triggered by ScheduleInternal().
template <typename T>
class SingleThreadedStageWithDeque : public Stage<T> {
 public:
  ~SingleThreadedStageWithDeque() override = default;

  bool IsIdle() const override {
    absl::MutexLock lock(mutex_);
    return state_ == State::kIdle;
  }

  bool NeedSchedule() const override {
    absl::MutexLock lock(mutex_);
    return state_ == State::kIdle && NeedScheduleInternal();
  }

  absl::Status Schedule() override {
    {
      absl::MutexLock lock(mutex_);
      if (state_ != State::kIdle) {
        return absl::NotFoundError(
            "This stage is already running or being scheduled.");
      }
      state_ = State::kScheduling;
    }
    return ScheduleInternal();
  }

  bool HasOutput() const override {
    absl::MutexLock lock(mutex_);
    return !outputs_.empty();
  }

  absl::StatusOr<T> GetOutput() override {
    absl::MutexLock lock(mutex_);
    if (outputs_.empty()) {
      return absl::NotFoundError("Deque has no output available.");
    }
    T item = std::move(outputs_.front());
    outputs_.pop_front();
    return item;
  }

  void Reset() override {
    WaitForStateThenSetState(State::kIdle, State::kRunning);
    ResetInternal();
    ClearOutputsThenSetState(State::kIdle);
  }

 protected:
  enum class State { kIdle, kScheduling, kRunning };

  // Subclasses can override this method to reset subclass-specific state.
  virtual void ResetInternal() {};

  // Subclasses must implement this method to determine if Schedule() should be
  // called soon to process work.
  virtual bool NeedScheduleInternal() const = 0;

  // Subclasses must run jobs synchronously or schedule them asynchronously
  // which do:
  // 1) set the state to kRunning when the job is started.
  // 2) enqueue outputs to the deque.
  // 3) set the state to kIdle when the job is finished.
  //
  // This method is called by Schedule() on the calling thread.
  virtual absl::Status ScheduleInternal() = 0;

  void SetState(State state) {
    absl::MutexLock lock(mutex_);
    state_ = state;
  }

  void WaitForStateThenSetState(State expected_state, State new_state) {
    absl::MutexLock lock(mutex_);
    auto condition = [this, expected_state] {
      mutex_.AssertHeld();
      return state_ == expected_state;
    };
    mutex_.Await(absl::Condition(&condition));
    state_ = new_state;
  }

  void ClearOutputsThenSetState(State state) {
    absl::MutexLock lock(mutex_);
    outputs_.clear();
    state_ = state;
  }

  void PushOutput(T output) {
    absl::MutexLock lock(mutex_);
    outputs_.push_back(std::move(output));
  }

  mutable absl::Mutex mutex_;
  State state_ ABSL_GUARDED_BY(mutex_) = State::kIdle;
  std::deque<T> outputs_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace litert::omni

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_BASE_STAGE_H_
