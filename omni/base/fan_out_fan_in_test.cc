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

#include <cctype>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "omni/base/fan_in_stage.h"
#include "omni/base/fan_out_stage.h"
#include "omni/base/sequenced.h"
#include "omni/base/stage.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep for ASSERT_OK

namespace litert::omni {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;

using TestItem = std::variant<std::string, int>;

// Upstream source for pushing TestItem objects.
class PushTestSource : public SingleThreadedStageWithDeque<TestItem> {
 public:
  void Push(TestItem item) { this->PushOutput(std::move(item)); }

 protected:
  bool NeedScheduleInternal() const override { return false; }
  absl::Status ScheduleInternal() override { return absl::OkStatus(); }
};

// Upstream test stage that emits `Sequenced<T>` items.
template <typename T>
class SequencedTestStage : public SingleThreadedStageWithDeque<Sequenced<T>> {
 public:
  void Push(uint64_t seqnum, T item) {
    this->PushOutput(Sequenced<T>{.seqnum = seqnum, .item = std::move(item)});
  }

 protected:
  bool NeedScheduleInternal() const override { return false; }
  absl::Status ScheduleInternal() override { return absl::OkStatus(); }
};

// Generic intermediate worker stage that transforms Sequenced<InT> to
// Sequenced<OutT> while preserving sequence numbers.
template <typename InT, typename OutT, typename QueueInT = TestItem>
class TransformStage : public SingleThreadedStageWithDeque<Sequenced<OutT>> {
 public:
  using TransformFn = absl::AnyInvocable<OutT(const InT&)>;

  TransformStage(FanOutOutputQueue<QueueInT, InT>* upstream, TransformFn fn)
      : upstream_(upstream), fn_(std::move(fn)) {}

 protected:
  bool NeedScheduleInternal() const override {
    return upstream_ != nullptr && !upstream_->Empty();
  }

  absl::Status ScheduleInternal() override {
    absl::Cleanup cleanup = [this] {
      this->SetState(
          SingleThreadedStageWithDeque<Sequenced<OutT>>::State::kIdle);
    };
    if (upstream_ == nullptr || upstream_->Empty()) {
      return absl::OkStatus();
    }
    auto status_or_input = upstream_->PopFront();
    if (!status_or_input.ok()) {
      return status_or_input.status();
    }
    auto input = std::move(*status_or_input);
    OutT transformed = fn_(input.item);
    this->PushOutput(Sequenced<OutT>{
        .seqnum = input.seqnum,
        .item = std::move(transformed),
    });
    return absl::OkStatus();
  }

 private:
  FanOutOutputQueue<QueueInT, InT>* upstream_ = nullptr;
  TransformFn fn_;
};

TEST(FanOutFanInTest, SequencedBasic) {
  Sequenced<std::string> seq{.seqnum = 42, .item = "test"};
  EXPECT_EQ(seq.seqnum, 42);
  EXPECT_EQ(seq.item, "test");
}

TEST(FanOutFanInTest, FanOutOutputQueueBasics) {
  FanOutOutputQueue<std::string> queue;
  EXPECT_EQ(queue.size(), 0);
  EXPECT_TRUE(queue.Empty());
  EXPECT_THAT(queue.PeekFront(), StatusIs(absl::StatusCode::kNotFound));

  EXPECT_OK(queue.PushBack(0, "test"));
  EXPECT_EQ(queue.size(), 1);
  EXPECT_FALSE(queue.Empty());

  auto peeked = queue.PeekFront();
  ASSERT_OK(peeked);
  EXPECT_EQ((*peeked)->seqnum, 0);
  EXPECT_EQ((*peeked)->item, "test");
  EXPECT_EQ(queue.size(), 1);
  EXPECT_FALSE(queue.Empty());

  auto item = queue.PopFront();
  ASSERT_OK(item);
  EXPECT_EQ(item->seqnum, 0);
  EXPECT_EQ(item->item, "test");
  EXPECT_TRUE(queue.Empty());
  EXPECT_THAT(queue.PeekFront(), StatusIs(absl::StatusCode::kNotFound));
}

TEST(FanOutFanInTest, SequencedVariantQueueUnwrapsMatchingType) {
  SequencedVariantQueue<TestItem, std::string> str_queue;
  SequencedVariantQueue<TestItem, int> int_queue;

  // Matching type pushes output.
  EXPECT_OK(str_queue.PushBack(0, std::string("hello")));
  EXPECT_FALSE(str_queue.Empty());
  auto str_out = str_queue.PopFront();
  ASSERT_OK(str_out);
  EXPECT_EQ(str_out->seqnum, 0);
  EXPECT_EQ(str_out->item, "hello");

  // Non-matching type returns UnimplementedError and pushes nothing.
  EXPECT_TRUE(absl::IsUnimplemented(str_queue.PushBack(1, 42)));
  EXPECT_TRUE(str_queue.Empty());

  // Matching type for int_queue.
  EXPECT_OK(int_queue.PushBack(1, 42));
  EXPECT_FALSE(int_queue.Empty());
  auto int_out = int_queue.PopFront();
  ASSERT_OK(int_out);
  EXPECT_EQ(int_out->seqnum, 1);
  EXPECT_EQ(int_out->item, 42);
}

TEST(FanOutStageTest, AddOutputQueueAndAutomaticVariantRouting) {
  PushTestSource source;
  FanOutStage<TestItem> fan_out(&source);
  auto* str_queue = fan_out.AddOutputQueue(
      std::make_unique<SequencedVariantQueue<TestItem, std::string>>());
  auto* int_queue = fan_out.AddOutputQueue(
      std::make_unique<SequencedVariantQueue<TestItem, int>>());

  source.Push(std::string("first"));
  source.Push(123);
  source.Push(std::string("second"));

  EXPECT_TRUE(fan_out.NeedSchedule());
  EXPECT_OK(fan_out.Schedule());

  EXPECT_FALSE(str_queue->Empty());
  EXPECT_FALSE(int_queue->Empty());

  // String queue should have received item 0 and item 2.
  auto str0 = str_queue->PopFront();
  ASSERT_OK(str0);
  EXPECT_EQ(str0->seqnum, 0);
  EXPECT_EQ(str0->item, "first");

  auto str2 = str_queue->PopFront();
  ASSERT_OK(str2);
  EXPECT_EQ(str2->seqnum, 2);
  EXPECT_EQ(str2->item, "second");

  // Int queue should have received item 1.
  auto int1 = int_queue->PopFront();
  ASSERT_OK(int1);
  EXPECT_EQ(int1->seqnum, 1);
  EXPECT_EQ(int1->item, 123);
}

TEST(FanOutStageTest, UpstreamSourceRouting) {
  PushTestSource source;
  FanOutStage<TestItem> fan_out(&source);
  auto* str_queue = fan_out.AddOutputQueue(
      std::make_unique<SequencedVariantQueue<TestItem, std::string>>());

  EXPECT_FALSE(fan_out.NeedSchedule());

  source.Push(std::string("upstream_val"));
  EXPECT_TRUE(fan_out.NeedSchedule());
  EXPECT_OK(fan_out.Schedule());

  auto item = str_queue->PopFront();
  ASSERT_OK(item);
  EXPECT_EQ(item->seqnum, 0);
  EXPECT_EQ(item->item, "upstream_val");
}

TEST(DesequencedStageTest, ReturnsOutputOnlyWhenSeqnumMatches) {
  SequencedTestStage<std::string> input_stage;
  DesequencedStage<std::string> desequenced(&input_stage);

  // Empty stage returns kNotFound.
  EXPECT_FALSE(desequenced.HasAnyOutput());
  EXPECT_FALSE(desequenced.HasOutput(0));
  EXPECT_THAT(desequenced.GetOutput(0), StatusIs(absl::StatusCode::kNotFound));

  input_stage.Push(1, "one");
  input_stage.Push(2, "two");
  EXPECT_TRUE(desequenced.HasAnyOutput());

  // Requested seqnum (0) is smaller than what it has (1) -> kNotFound.
  EXPECT_FALSE(desequenced.HasOutput(0));
  EXPECT_THAT(desequenced.GetOutput(0), StatusIs(absl::StatusCode::kNotFound));

  // Requested seqnum (2) is larger than what it has (1) -> kNotFound.
  EXPECT_FALSE(desequenced.HasOutput(2));
  EXPECT_THAT(desequenced.GetOutput(2), StatusIs(absl::StatusCode::kNotFound));

  // Requested seqnum (1) matches what it has (1) -> returns "one".
  EXPECT_TRUE(desequenced.HasOutput(1));
  EXPECT_THAT(desequenced.GetOutput(1), IsOkAndHolds("one"));

  // Next item is 2; requesting 2 now matches -> returns "two".
  EXPECT_TRUE(desequenced.HasOutput(2));
  EXPECT_THAT(desequenced.GetOutput(2), IsOkAndHolds("two"));
  EXPECT_FALSE(desequenced.HasAnyOutput());
}

TEST(FanInStageTest, InOrderItemsEmittedDirectly) {
  SequencedTestStage<std::string> input_stage;
  FanInStage<std::string> fan_in;
  fan_in.AddInputStage(&input_stage);

  input_stage.Push(0, "first");
  input_stage.Push(1, "second");

  EXPECT_TRUE(fan_in.NeedSchedule());
  EXPECT_OK(fan_in.Schedule());

  EXPECT_TRUE(fan_in.HasOutput());
  EXPECT_THAT(fan_in.GetOutput(), IsOkAndHolds("first"));
  EXPECT_THAT(fan_in.GetOutput(), IsOkAndHolds("second"));
  EXPECT_FALSE(fan_in.HasOutput());
}

TEST(FanInStageTest, OutOfOrderArrivalReordersCorrectly) {
  SequencedTestStage<std::string> stage_a;
  SequencedTestStage<std::string> stage_b;
  FanInStage<std::string> fan_in;
  fan_in.AddInputStage(&stage_a);
  fan_in.AddInputStage(&stage_b);

  // Simulate out-of-order delivery: seqnum 0 and seqnum 2 arrive before
  // seqnum 1.
  stage_a.Push(0, "zero");
  stage_b.Push(2, "two");

  // FanIn emits seqnum 0 immediately and holds seqnum 2 waiting for seqnum 1.
  EXPECT_OK(fan_in.Schedule());
  EXPECT_TRUE(fan_in.HasOutput());
  EXPECT_THAT(fan_in.GetOutput(), IsOkAndHolds("zero"));
  EXPECT_FALSE(fan_in.HasOutput());
  EXPECT_EQ(fan_in.next_expected_seqnum(), 1);

  // Now seqnum 1 arrives.
  stage_a.Push(1, "one");

  EXPECT_TRUE(fan_in.NeedSchedule());
  EXPECT_OK(fan_in.Schedule());

  // Both seqnum 1 and buffered seqnum 2 are now emitted in contiguous order.
  EXPECT_TRUE(fan_in.HasOutput());
  EXPECT_EQ(fan_in.next_expected_seqnum(), 3);

  EXPECT_THAT(fan_in.GetOutput(), IsOkAndHolds("one"));
  EXPECT_THAT(fan_in.GetOutput(), IsOkAndHolds("two"));
  EXPECT_FALSE(fan_in.HasOutput());
}

TEST(FanInStageTest, OutOfRangeSeqnumReturnsError) {
  SequencedTestStage<std::string> stage_a;
  SequencedTestStage<std::string> stage_b;
  FanInStage<std::string> fan_in;
  fan_in.AddInputStage(&stage_a);
  fan_in.AddInputStage(&stage_b);

  stage_a.Push(0, "zero");
  stage_a.Push(2, "two");
  stage_b.Push(0, "duplicate_zero");

  // Schedule consumes seqnum 0 from stage_a and advances next_expected_seqnum
  // to 1. Both stage_a (seqnum 2) and stage_b (seqnum 0) still have outputs
  // (`HasAnyOutput()` is true), but neither has seqnum 1 (`HasOutput(1)` is
  // false), so Schedule returns kOutOfRange.
  EXPECT_THAT(fan_in.Schedule(), StatusIs(absl::StatusCode::kOutOfRange));
}

TEST(FanOutFanInTest, EndToEndPipeline) {
  PushTestSource source;
  FanOutStage<TestItem> fan_out(&source);

  auto* str_channel = fan_out.AddOutputQueue(
      std::make_unique<SequencedVariantQueue<TestItem, std::string>>());
  auto* int_channel = fan_out.AddOutputQueue(
      std::make_unique<SequencedVariantQueue<TestItem, int>>());

  // String branch: converts string to uppercase.
  TransformStage<std::string, std::string> str_worker(str_channel,
                                                      [](const std::string& s) {
                                                        std::string res = s;
                                                        for (char& c : res)
                                                          c = std::toupper(c);
                                                        return res;
                                                      });

  // Int branch: converts int to string representation.
  TransformStage<int, std::string> int_worker(
      int_channel, [](int val) { return absl::StrCat("num(", val, ")"); });

  // FanIn collects from all input stages and reconstructs the ordered stream.
  FanInStage<std::string> fan_in;
  fan_in.AddInputStage(&str_worker);
  fan_in.AddInputStage(&int_worker);

  // Push items: [str("alpha"), int(99), str("beta")]
  source.Push(std::string("alpha"));
  source.Push(99);
  source.Push(std::string("beta"));

  // Schedule fan-out.
  EXPECT_OK(fan_out.Schedule());

  // String worker runs: processes "alpha" (seqnum 0) and "beta" (seqnum 2).
  EXPECT_OK(str_worker.Schedule());
  EXPECT_OK(str_worker.Schedule());

  // FanIn inspects input stages: sees seqnum 0 ("ALPHA") and seqnum 2 ("BETA").
  // Emits seqnum 0 immediately and buffers seqnum 2 until seqnum 1 arrives.
  EXPECT_OK(fan_in.Schedule());
  EXPECT_TRUE(fan_in.HasOutput());
  EXPECT_THAT(fan_in.GetOutput(), IsOkAndHolds("ALPHA"));
  EXPECT_FALSE(fan_in.HasOutput());

  // Int worker runs: processes 99 (seqnum 1).
  EXPECT_OK(int_worker.Schedule());

  // FanIn runs again: receives seqnum 1 ("num(99)") and releases seqnum 2
  // ("BETA") in order.
  EXPECT_OK(fan_in.Schedule());
  EXPECT_TRUE(fan_in.HasOutput());
  EXPECT_THAT(fan_in.GetOutput(), IsOkAndHolds("num(99)"));
  EXPECT_THAT(fan_in.GetOutput(), IsOkAndHolds("BETA"));
  EXPECT_FALSE(fan_in.HasOutput());
}

TEST(FanOutFanInTest, ResetClearsStateCleanly) {
  SequencedTestStage<std::string> stage_a;
  SequencedTestStage<std::string> stage_b;
  FanInStage<std::string> fan_in;
  fan_in.AddInputStage(&stage_a);
  fan_in.AddInputStage(&stage_b);

  stage_a.Push(0, "zero");
  stage_b.Push(2, "orphan");
  EXPECT_OK(fan_in.Schedule());
  EXPECT_EQ(fan_in.next_expected_seqnum(), 1);
  EXPECT_TRUE(fan_in.HasOutput());

  stage_a.Push(3, "unconsumed_upstream");
  fan_in.Reset();
  EXPECT_EQ(fan_in.next_expected_seqnum(), 0);
  EXPECT_FALSE(fan_in.HasOutput());
  EXPECT_FALSE(stage_a.HasOutput());
  EXPECT_FALSE(stage_b.HasOutput());
}

class FlushTestStage : public SingleThreadedStageWithDeque<Sequenced<int>> {
 public:
  void SetFlushItem(uint64_t seqnum, int item) {
    flush_item_ = Sequenced<int>{.seqnum = seqnum, .item = item};
  }
  void SetFlushStatus(absl::Status status) {
    flush_status_ = std::move(status);
  }

 protected:
  bool NeedScheduleInternal() const override { return false; }
  absl::Status ScheduleInternal() override { return absl::OkStatus(); }
  absl::Status FlushInternal() override {
    if (!flush_status_.ok()) {
      return flush_status_;
    }
    if (flush_item_.has_value()) {
      this->PushOutput(*std::move(flush_item_));
      flush_item_.reset();
    }
    return absl::OkStatus();
  }

 private:
  std::optional<Sequenced<int>> flush_item_;
  absl::Status flush_status_ = absl::OkStatus();
};

TEST(FanInStageTest, FlushDrainsMatchesAndPropagatesUpstreamErrors) {
  FlushTestStage stage_a;
  FlushTestStage stage_b;
  FanInStage<int> fan_in;
  fan_in.AddInputStage(&stage_a);
  fan_in.AddInputStage(&stage_b);

  stage_a.SetFlushItem(0, 100);
  stage_b.SetFlushItem(1, 200);
  EXPECT_OK(fan_in.Flush());
  EXPECT_THAT(fan_in.GetOutput(), IsOkAndHolds(100));
  EXPECT_THAT(fan_in.GetOutput(), IsOkAndHolds(200));

  stage_b.SetFlushStatus(absl::InternalError("upstream flush failed"));
  EXPECT_THAT(fan_in.Flush(), StatusIs(absl::StatusCode::kInternal));
}

class CallbackChannel : public SequencedChannel<TestItem> {
 public:
  explicit CallbackChannel(FanOutStage<TestItem>* stage) : stage_(stage) {}

  bool Accepts(const TestItem& input) const override { return true; }

  absl::Status PushBack(uint64_t seqnum, TestItem input) override {
    // Querying stage state inside PushBack must not deadlock on mutex_.
    observed_idle_ = stage_->IsIdle();
    empty_ = false;
    observed_has_output_ = stage_->HasOutput();
    return absl::OkStatus();
  }

  bool Empty() const override { return empty_; }
  void Reset() override { empty_ = true; }

  bool observed_idle() const { return observed_idle_; }
  bool observed_has_output() const { return observed_has_output_; }

 private:
  FanOutStage<TestItem>* stage_;
  bool empty_ = true;
  bool observed_idle_ = true;
  bool observed_has_output_ = false;
};

TEST(FanOutStageTest, RouteInputDoesNotHoldMutexDuringPushBack) {
  PushTestSource source;
  FanOutStage<TestItem> fan_out(&source);
  auto* channel =
      fan_out.AddOutputQueue(std::make_unique<CallbackChannel>(&fan_out));

  source.Push(std::string("item"));
  EXPECT_OK(fan_out.Schedule());
  EXPECT_FALSE(channel->observed_idle());
  EXPECT_TRUE(channel->observed_has_output());
}

TEST(FanOutStageTest, UnacceptedInputDoesNotBurnSeqnum) {
  PushTestSource source;
  FanOutStage<TestItem> fan_out(&source);
  auto* str_queue = fan_out.AddOutputQueue(
      std::make_unique<SequencedVariantQueue<TestItem, std::string>>());

  // Push an int (no registered int queue), followed by a string.
  source.Push(42);
  source.Push(std::string("accepted"));

  EXPECT_THAT(fan_out.Schedule(), StatusIs(absl::StatusCode::kInvalidArgument));
  // Subsequent accepted item gets seqnum 0 because the rejected item did not
  // burn a sequence number.
  EXPECT_OK(fan_out.Schedule());
  ASSERT_OK_AND_ASSIGN(auto item, str_queue->PopFront());
  EXPECT_EQ(item.seqnum, 0);
  EXPECT_EQ(item.item, "accepted");
}

class FailingSource : public Stage<TestItem> {
 public:
  bool IsIdle() const override { return true; }
  bool NeedSchedule() const override { return false; }
  absl::Status Schedule() override { return absl::OkStatus(); }
  bool HasOutput() const override { return true; }
  absl::Status Flush() override { return absl::OkStatus(); }
  void Reset() override {}
  absl::StatusOr<TestItem> GetOutput() override {
    return absl::InternalError("upstream failure");
  }
};

TEST(FanOutStageTest, SchedulePropagatesNonNotFoundUpstreamError) {
  FailingSource source;
  FanOutStage<TestItem> fan_out(&source);
  fan_out.AddOutputQueue(
      std::make_unique<SequencedVariantQueue<TestItem, std::string>>());

  EXPECT_THAT(fan_out.Schedule(), StatusIs(absl::StatusCode::kInternal));
}

}  // namespace
}  // namespace litert::omni
