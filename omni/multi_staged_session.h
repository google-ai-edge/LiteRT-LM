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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_MULTI_STAGED_SESSION_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_MULTI_STAGED_SESSION_H_

#include <memory>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/base/thread_annotations.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "omni/base/async_stage_scheduler.h"
#include "omni/base/stage.h"
#include "omni/omni_session.h"
#include "runtime/framework/threadpool.h"

namespace litert::omni {

// Generic `OmniSession` implementation that orchestrates a linear,
// upstream-to-downstream pipeline of stages ending with a
// `Stage<OmniSession::Output>`.
//
// Contract:
// - `stages` must be a non-empty, upstream-to-downstream ordered pipeline of
//   non-null stages. `stages[0]` is treated as the source stage: in
//   `ProcessNext()`, `stages[0]` is only scheduled when no downstream stage has
//   work, and `absl::OutOfRangeError` from `stages[0]` signals end-of-stream.
// - `output_stage` must equal `stages.back().get()`; it is passed alongside
//   `stages` to preserve its static `Stage<Output>*` type without RTTI or
//   unchecked downcasts.
// - `thread_pool` may be null for synchronous-only sessions; calling
//   `ProcessAsync()` when `thread_pool` is null fails with
//   `absl::FailedPreconditionError`.
// - `ProcessNext()` loops through the pipeline until an `Output` chunk is
//   produced or no stage has `NeedSchedule() == true`. It returns one `Output`
//   chunk per call, `absl::NotFoundError` when no output is ready yet, and
//   `absl::OutOfRangeError` when the source stage reaches end-of-stream.
// - `Flush()` stops any active `ProcessAsync()` scheduler first (callers must
//   call `ProcessAsync()` again afterward if they want to resume async
//   processing), flushes and drains all stages in order, and returns
//   `EndOfOutput` when no output is produced or a single combined `Output`
//   across all flushed chunks.
// - Thread-safety: `Reset()`, `ProcessNext()`, and `Flush()` stop any active
//   `ProcessAsync()` scheduler before accessing stages. In `ProcessAsync()`,
//   end-of-stream stage flushing runs on a worker thread before invoking
//   `callback` with the terminal status.
class MultiStagedSession : public OmniSession {
 public:
  static absl::StatusOr<std::unique_ptr<MultiStagedSession>> Create(
      std::vector<std::unique_ptr<internal::StageBase>> stages,
      Stage<Output>* absl_nonnull output_stage,
      ::litert::lm::ThreadPool* absl_nullable thread_pool = nullptr);

  ~MultiStagedSession() override;

  void Reset() override;

  absl::StatusOr<Output> ProcessNext() override;

  absl::Status ProcessAsync(OutputCallback callback) override;

  absl::StatusOr<Output> Flush() override;

  absl::Span<const std::unique_ptr<internal::StageBase>> stages() const {
    return stages_;
  }

 protected:
  MultiStagedSession(std::vector<std::unique_ptr<internal::StageBase>> stages,
                     Stage<Output>* absl_nonnull output_stage,
                     ::litert::lm::ThreadPool* absl_nullable thread_pool)
      : stages_(std::move(stages)),
        output_stage_(output_stage),
        thread_pool_(thread_pool) {}

 private:
  static absl::Status CombineOutput(Output& accumulated, Output&& next);
  static absl::Status DrainStageUntilIdle(internal::StageBase& stage,
                                          bool is_source_stage);

  bool AnyStageNeedsSchedule() const;
  absl::Status FlushAllStages();
  void ResetAsyncScheduler();

  std::vector<std::unique_ptr<internal::StageBase>> stages_;
  Stage<Output>* absl_nonnull const output_stage_;
  ::litert::lm::ThreadPool* absl_nullable const thread_pool_ = nullptr;

  mutable absl::Mutex mutex_;
  std::unique_ptr<AsyncStageScheduler<Output>> async_scheduler_
      ABSL_GUARDED_BY(mutex_);
};

}  // namespace litert::omni

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_MULTI_STAGED_SESSION_H_
