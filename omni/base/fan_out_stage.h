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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_BASE_FAN_OUT_STAGE_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_BASE_FAN_OUT_STAGE_H_

#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/base/thread_annotations.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "omni/base/sequenced.h"
#include "omni/base/stage.h"

namespace litert::omni {

// Fan-out / fan-in pipeline overview for parallel per-data-type processing
// with ordered reconstruction:
//
//   Upstream Stage<InputT>
//            |
//            v
//     FanOutStage<InputT>  (assigns contiguous seqnum = 0, 1, 2, ...)
//      /           \
//     v             v
//  Queue<TypeA>   Queue<TypeB>   (FanOutOutputQueue / SequencedVariantQueue)
//     |             |
//     v             v
//  Stage<TypeA>   Stage<TypeB>   (process each data type in parallel,
//     \             /             preserving Sequenced<ProcessedT>::seqnum)
//      v           v
//     FanInStage<ProcessedT>     (reconstructs ordered sequence by
//                                 seqnum = 0, 1, 2, ...)
//
// 1. `FanOutStage<InputT>` pulls heterogeneous inputs (such as a `std::variant`
//    over text, audio, and image inputs) from an upstream `Stage<InputT>`,
//    assigns a monotonically increasing sequence number (`seqnum`) to each item
//    in arrival order, and dispatches it to the matching output queue
//    (`FanOutOutputQueue` or `SequencedVariantQueue`) registered via
//    `AddOutputQueue()`.
// 2. Downstream worker stages for each data type independently pop
//    `Sequenced<ModalityT>` items from their respective output queues, process
//    them in parallel (potentially finishing out of order due to varying
//    modality latencies), and emit `Sequenced<ProcessedT>` with the original
//    `seqnum` intact.
// 3. `FanInStage<ProcessedT>` registers those parallel worker stages via
//    `AddInputStage()`, wraps each in a `DesequencedStage<ProcessedT>`, and
//    emits contiguous items starting from `0, 1, 2, ...` to reconstruct the
//    exact original input order.
//
// Invariant: every `seqnum` assigned by `FanOutStage` must reach `FanInStage`
// exactly once through one registered branch (control inputs that produce no
// downstream item should be filtered before `RouteInput()` assigns a `seqnum`).
// TODO(b/538727793): Support 0 or N outputs per seqnum when needed by
// downstream branches.

// Output queue for FanOutStage that accepts `InputT` items and stores
// `Sequenced<OutputT>` items for downstream consumption.
template <typename InputT, typename OutputT = InputT>
class FanOutOutputQueue : public SequencedChannel<InputT> {
 public:
  FanOutOutputQueue() = default;
  ~FanOutOutputQueue() override = default;

  bool Accepts(const InputT& input) const override {
    return std::is_constructible_v<OutputT, InputT>;
  }

  absl::Status PushBack(uint64_t seqnum, InputT input) override {
    if constexpr (std::is_constructible_v<OutputT, InputT>) {
      absl::MutexLock lock(mutex_);
      queue_.push_back(Sequenced<OutputT>{
          .seqnum = seqnum,
          .item = OutputT(std::move(input)),
      });
      return absl::OkStatus();
    } else {
      return absl::UnimplementedError(
          "PushBack(seqnum, InputT) is not implemented for this type pair.");
    }
  }

  bool Empty() const override {
    absl::MutexLock lock(mutex_);
    return queue_.empty();
  }

  void Reset() override {
    absl::MutexLock lock(mutex_);
    queue_.clear();
  }

  // Returns a pointer to the next available Sequenced<OutputT> at the front of
  // the queue without removing it, or absl::NotFoundError if the queue is
  // empty.
  virtual absl::StatusOr<const Sequenced<OutputT>* absl_nonnull> PeekFront()
      const {
    absl::MutexLock lock(mutex_);
    if (queue_.empty()) {
      return absl::NotFoundError("Queue is empty.");
    }
    return &queue_.front();
  }

  // Removes and returns the next available Sequenced<OutputT> from the front
  // of the queue.
  virtual absl::StatusOr<Sequenced<OutputT>> PopFront() {
    absl::MutexLock lock(mutex_);
    if (queue_.empty()) {
      return absl::NotFoundError("Queue is empty.");
    }
    auto item = std::move(queue_.front());
    queue_.pop_front();
    return item;
  }

  // Returns number of buffered items.
  size_t size() const {
    absl::MutexLock lock(mutex_);
    return queue_.size();
  }

 protected:
  mutable absl::Mutex mutex_;
  std::deque<Sequenced<OutputT>> queue_ ABSL_GUARDED_BY(mutex_);
};

// Output queue for variant inputs where `VariantInputT` is a std::variant
// containing `OutputT`. Automatically unwraps the variant and enqueues matching
// items with their sequence number.
template <typename VariantInputT, typename OutputT>
class SequencedVariantQueue : public FanOutOutputQueue<VariantInputT, OutputT> {
 public:
  SequencedVariantQueue() = default;
  ~SequencedVariantQueue() override = default;

  bool Accepts(const VariantInputT& input) const override {
    return std::holds_alternative<OutputT>(input);
  }

  absl::Status PushBack(uint64_t seqnum, VariantInputT input) override {
    if (auto* val = std::get_if<OutputT>(&input)) {
      absl::MutexLock lock(this->mutex_);
      this->queue_.push_back(Sequenced<OutputT>{
          .seqnum = seqnum,
          .item = std::move(*val),
      });
      return absl::OkStatus();
    }
    return absl::UnimplementedError(
        "VariantInput does not hold requested Output type.");
  }
};

// Abstract base class for fan-out stages that consume inputs of type
// `InputT` from an upstream stage, assign contiguous sequence numbers
// (`seqnum`), and dispatch modality items to registered `SequencedChannel`
// output queues.
template <typename InputT>
class FanOutStage : public internal::StageBase {
 public:
  explicit FanOutStage(Stage<InputT>* absl_nonnull upstream)
      : upstream_(*upstream) {}
  ~FanOutStage() override = default;

  // Registers an owned output queue with this fan-out stage during setup.
  template <typename ChannelT>
  ChannelT* AddOutputQueue(std::unique_ptr<ChannelT> queue) {
    auto* ptr = queue.get();
    output_queues_.push_back(std::move(queue));
    return ptr;
  }

  // StageBase interface implementation.
  bool IsIdle() const override {
    absl::MutexLock lock(mutex_);
    return state_ == State::kIdle;
  }

  bool NeedSchedule() const override {
    absl::MutexLock lock(mutex_);
    if (state_ != State::kIdle) {
      return false;
    }
    return upstream_.HasOutput();
  }

  absl::Status Schedule() override {
    {
      absl::MutexLock lock(mutex_);
      if (state_ != State::kIdle) {
        return absl::NotFoundError(
            "FanOutStage is already running or being scheduled.");
      }
      state_ = State::kScheduling;
    }

    absl::Status status = ScheduleInternal();

    absl::MutexLock lock(mutex_);
    state_ = State::kIdle;
    return status;
  }

  bool HasOutput() const override {
    for (const auto& queue : output_queues_) {
      if (queue != nullptr && !queue->Empty()) {
        return true;
      }
    }
    return false;
  }

  absl::Status Flush() override {
    {
      absl::MutexLock lock(mutex_);
      auto condition = [this] {
        mutex_.AssertHeld();
        return state_ == State::kIdle;
      };
      mutex_.Await(absl::Condition(&condition));
      state_ = State::kRunning;
    }

    absl::Status status = FlushInternal();

    absl::MutexLock lock(mutex_);
    state_ = State::kIdle;
    return status;
  }

  void Reset() override {
    {
      absl::MutexLock lock(mutex_);
      auto condition = [this] {
        mutex_.AssertHeld();
        return state_ == State::kIdle;
      };
      mutex_.Await(absl::Condition(&condition));
      state_ = State::kRunning;
      next_seqnum_ = 0;
    }

    ResetInternal();
    for (const auto& queue : output_queues_) {
      if (queue != nullptr) {
        queue->Reset();
      }
    }

    absl::MutexLock lock(mutex_);
    state_ = State::kIdle;
  }

 protected:
  enum class State { kIdle, kScheduling, kRunning };

  uint64_t GetNextSeqnum() {
    absl::MutexLock lock(mutex_);
    return next_seqnum_++;
  }

  // Subclasses can override RouteInput, or use the default implementation
  // which finds the first registered output queue accepting `input`, assigns
  // the next sequence number, and moves `input` into that queue.
  virtual absl::Status RouteInput(InputT input) {
    for (const auto& queue : output_queues_) {
      if (queue == nullptr || !queue->Accepts(input)) continue;
      return queue->PushBack(GetNextSeqnum(), std::move(input));
    }
    return absl::InvalidArgumentError("No output queue accepted the input.");
  }

  virtual absl::Status FlushInternal() { return absl::OkStatus(); }
  virtual void ResetInternal() {}

  absl::Status ScheduleInternal() {
    bool processed_any = false;
    while (upstream_.HasOutput()) {
      auto status_or_input = upstream_.GetOutput();
      if (!status_or_input.ok()) {
        if (absl::IsNotFound(status_or_input.status())) {
          break;
        }
        return status_or_input.status();
      }
      processed_any = true;
      auto status = RouteInput(std::move(*status_or_input));
      if (!status.ok()) {
        return status;
      }
    }

    if (!processed_any) {
      return absl::NotFoundError(
          "No input available from upstream to schedule.");
    }
    return absl::OkStatus();
  }

  mutable absl::Mutex mutex_;
  State state_ ABSL_GUARDED_BY(mutex_) = State::kIdle;
  uint64_t next_seqnum_ ABSL_GUARDED_BY(mutex_) = 0;
  Stage<InputT>& upstream_;
  std::vector<std::unique_ptr<SequencedChannel<InputT>>> output_queues_;
};

}  // namespace litert::omni

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_BASE_FAN_OUT_STAGE_H_
