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
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "omni/base/sequenced.h"
#include "omni/base/stage.h"

namespace litert::omni {

// Adapter around an upstream `SingleThreadedStageWithDeque<Sequenced<T>>` that
// unwraps and returns output `T` only when the requested sequence number
// matches the front item's `seqnum`.
template <typename T>
class DesequencedStage {
 public:
  explicit DesequencedStage(
      SingleThreadedStageWithDeque<Sequenced<T>>* absl_nonnull stage)
      : stage_(*stage) {}

  // Returns an output only when `seqnum` matches the next available output's
  // sequence number; otherwise returns `kNotFound` without removing the front
  // item from `stage_`.
  absl::StatusOr<T> GetOutput(uint64_t seqnum) {
    ABSL_ASSIGN_OR_RETURN(const Sequenced<T>* peeked, stage_.PeekOutput());
    if (peeked->seqnum != seqnum) {
      return absl::NotFoundError(
          "Requested seqnum does not match the available output seqnum.");
    }
    ABSL_ASSIGN_OR_RETURN(Sequenced<T> output, stage_.GetOutput());
    return std::move(output.item);
  }

  // Returns true if this stage has an output ready for `seqnum`.
  bool HasOutput(uint64_t seqnum) const {
    absl::StatusOr<const Sequenced<T>*> peeked = stage_.PeekOutput();
    return peeked.ok() && (*peeked)->seqnum == seqnum;
  }

  // Returns true if this stage has any upstream output available.
  bool HasAnyOutput() const { return stage_.HasOutput(); }

  absl::Status Flush() { return stage_.Flush(); }

  void Reset() { stage_.Reset(); }

 private:
  SingleThreadedStageWithDeque<Sequenced<T>>& stage_;
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
  void AddInputStage(
      SingleThreadedStageWithDeque<Sequenced<T>>* absl_nonnull input_stage) {
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
