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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_BASE_FAN_IN_STAGE_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_BASE_FAN_IN_STAGE_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/base/thread_annotations.h"  // from @com_google_absl
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "omni/base/sequenced.h"
#include "omni/base/stage.h"

namespace litert::omni {

// Adapter around an upstream `Stage<Sequenced<T>>` that unwraps and returns
// output `T` only when the requested sequence number matches the front item's
// `seqnum`.
template <typename T>
class DesequencedStage {
 public:
  explicit DesequencedStage(Stage<Sequenced<T>>* absl_nonnull stage)
      : stage_(*stage) {}

  // Returns an output only when `seqnum` matches the next available output's
  // sequence number; otherwise returns `kNotFound`.
  absl::StatusOr<T> GetOutput(uint64_t seqnum) {
    absl::MutexLock lock(mutex_);
    if (!pending_output_.has_value()) {
      auto status_or_output = stage_.GetOutput();
      if (!status_or_output.ok()) {
        return status_or_output.status();
      }
      pending_output_ = std::move(*status_or_output);
    }
    if (pending_output_->seqnum != seqnum) {
      return absl::NotFoundError(
          "Requested seqnum does not match the available output seqnum.");
    }
    T item = std::move(pending_output_->item);
    pending_output_.reset();
    return item;
  }

  // Returns true if this stage has an output ready for `seqnum`.
  bool HasOutput(uint64_t seqnum) {
    absl::MutexLock lock(mutex_);
    if (!pending_output_.has_value()) {
      auto status_or_output = stage_.GetOutput();
      if (!status_or_output.ok()) {
        return false;
      }
      pending_output_ = std::move(*status_or_output);
    }
    return pending_output_->seqnum == seqnum;
  }

  // Returns true if this stage has any buffered or upstream output available.
  bool HasAnyOutput() const {
    absl::MutexLock lock(mutex_);
    return pending_output_.has_value() || stage_.HasOutput();
  }

  absl::Status Flush() { return stage_.Flush(); }

  void Reset() {
    absl::MutexLock lock(mutex_);
    pending_output_.reset();
    stage_.Reset();
  }

 private:
  mutable absl::Mutex mutex_;
  Stage<Sequenced<T>>& stage_;
  std::optional<Sequenced<T>> pending_output_ ABSL_GUARDED_BY(mutex_);
};

// Fan-in stage that collects `Sequenced<T>` from multiple upstream input
// stages via `DesequencedStage<T>` and emits `T` in contiguous sequence order
// (`0, 1, 2, ...`).
//
// Note: `FlushInternal()` and `ResetInternal()` cascade `Flush()` and `Reset()`
// to registered input stages, so branch `FlushInternal()` and `ResetInternal()`
// implementations must be idempotent when used with `MultiStagedSession`.
template <typename T>
class FanInStage : public SingleThreadedStageWithDeque<T> {
 public:
  FanInStage() = default;
  ~FanInStage() override = default;

  // Registers an upstream input stage whose output is `Sequenced<T>`.
  void AddInputStage(Stage<Sequenced<T>>* absl_nonnull input_stage) {
    input_stages_.push_back(std::make_unique<DesequencedStage<T>>(input_stage));
  }

  // Returns the next expected sequence number.
  uint64_t next_expected_seqnum() const { return next_expected_seqnum_; }

 protected:
  bool NeedScheduleInternal() const override {
    for (const auto& input_stage : input_stages_) {
      if (input_stage->HasOutput(next_expected_seqnum_)) {
        return true;
      }
    }
    return false;
  }

  absl::Status ScheduleInternal() override {
    absl::Cleanup cleanup = [this] {
      this->SetState(SingleThreadedStageWithDeque<T>::State::kIdle);
    };

    ABSL_RETURN_IF_ERROR(DrainReadyMatches());

    if (!this->HasOutput()) {
      return absl::NotFoundError("FanInStage has no ready sequence items.");
    }
    return absl::OkStatus();
  }

  void ResetInternal() override {
    SingleThreadedStageWithDeque<T>::ResetInternal();
    for (const auto& input_stage : input_stages_) {
      input_stage->Reset();
    }
    next_expected_seqnum_ = 0;
  }

  absl::Status FlushInternal() override {
    for (const auto& input_stage : input_stages_) {
      ABSL_RETURN_IF_ERROR(input_stage->Flush());
    }
    return DrainReadyMatches();
  }

 private:
  absl::Status DrainReadyMatches() {
    bool matched = true;
    while (matched && !input_stages_.empty()) {
      matched = false;
      bool all_have_output = true;
      for (const auto& input_stage : input_stages_) {
        if (input_stage->HasOutput(next_expected_seqnum_)) {
          ABSL_ASSIGN_OR_RETURN(T item,
                                input_stage->GetOutput(next_expected_seqnum_));
          this->PushOutput(std::move(item));
          ++next_expected_seqnum_;
          matched = true;
          break;
        }
        if (!input_stage->HasAnyOutput()) {
          all_have_output = false;
        }
      }
      if (!matched && all_have_output) {
        return absl::OutOfRangeError(
            "All input stages have outputs, but none matches the expected "
            "sequence number.");
      }
    }
    return absl::OkStatus();
  }

  std::vector<std::unique_ptr<DesequencedStage<T>>> input_stages_;
  uint64_t next_expected_seqnum_ = 0;
};

}  // namespace litert::omni

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_BASE_FAN_IN_STAGE_H_
