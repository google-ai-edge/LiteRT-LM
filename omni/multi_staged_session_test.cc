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

#include "omni/multi_staged_session.h"

#include <deque>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "absl/synchronization/notification.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "omni/base/io_types.h"
#include "omni/base/stage.h"
#include "omni/omni_session.h"
#include "runtime/framework/threadpool.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace litert::omni {
namespace {

using ::absl_testing::StatusIs;
using ::testing::ElementsAre;

// Source stage that emits queued strings and optionally flushes a final string.
class FakeStringSource : public SingleThreadedStageWithDeque<std::string> {
 public:
  void Push(std::string item) {
    absl::MutexLock lock(mu_);
    pending_.push_back(std::move(item));
  }

  void SetFlushItem(std::string item) {
    absl::MutexLock lock(mu_);
    flush_item_ = std::move(item);
  }

  void Finish() {
    absl::MutexLock lock(mu_);
    finished_ = true;
  }

 protected:
  void ResetInternal() override {
    absl::MutexLock lock(mu_);
    pending_.clear();
    flush_item_.clear();
    finished_ = false;
  }

  bool NeedScheduleInternal() const override {
    absl::MutexLock lock(mu_);
    return !pending_.empty();
  }

  absl::Status ScheduleInternal() override {
    SetState(State::kRunning);
    absl::Cleanup cleanup = [this] { SetState(State::kIdle); };
    std::string next;
    {
      absl::MutexLock lock(mu_);
      if (!pending_.empty()) {
        next = std::move(pending_.front());
        pending_.pop_front();
      } else if (finished_) {
        return absl::OutOfRangeError("Source exhausted");
      } else {
        return absl::OkStatus();
      }
    }
    PushOutput(std::move(next));
    return absl::OkStatus();
  }

  absl::Status FlushInternal() override {
    std::string item;
    {
      absl::MutexLock lock(mu_);
      finished_ = true;
      item = std::move(flush_item_);
      flush_item_.clear();
    }
    if (!item.empty()) {
      PushOutput(std::move(item));
    }
    return absl::OkStatus();
  }

 private:
  mutable absl::Mutex mu_;
  std::deque<std::string> pending_;
  std::string flush_item_;
  bool finished_ = false;
};

// Stage that requires `required_schedules` calls to `Schedule()` per input
// string before emitting a `TextOutput`, and can also emit a flushed item or
// fail on `Flush()`.
class MultiStepTextStage : public SingleThreadedStageWithDeque<Output> {
 public:
  explicit MultiStepTextStage(Stage<std::string>* input,
                              int required_schedules = 1)
      : input_(*input), required_schedules_(required_schedules) {}

  void SetFlushText(std::string text) { flush_text_ = std::move(text); }
  void SetFlushError(absl::Status status) { flush_error_ = std::move(status); }

 protected:
  void ResetInternal() override {
    current_text_.clear();
    remaining_steps_ = 0;
  }

  bool NeedScheduleInternal() const override {
    return remaining_steps_ > 0 || input_.HasOutput();
  }

  absl::Status ScheduleInternal() override {
    SetState(State::kRunning);
    absl::Cleanup cleanup = [this] { SetState(State::kIdle); };
    if (remaining_steps_ == 0) {
      auto in = input_.GetOutput();
      if (absl::IsNotFound(in.status())) {
        return absl::OkStatus();
      }
      if (!in.ok()) {
        return in.status();
      }
      current_text_ = *std::move(in);
      remaining_steps_ = required_schedules_;
    }
    --remaining_steps_;
    if (remaining_steps_ == 0) {
      PushOutput(TextOutput{.confirmed_text = std::move(current_text_),
                            .unconfirmed_text = ""});
      current_text_.clear();
    }
    return absl::OkStatus();
  }

  absl::Status FlushInternal() override {
    if (!flush_error_.ok()) {
      return flush_error_;
    }
    if (remaining_steps_ > 0 && !current_text_.empty()) {
      remaining_steps_ = 0;
      PushOutput(TextOutput{.confirmed_text = std::move(current_text_),
                            .unconfirmed_text = ""});
      current_text_.clear();
    }
    if (!flush_text_.empty()) {
      PushOutput(TextOutput{.confirmed_text = std::move(flush_text_),
                            .unconfirmed_text = ""});
      flush_text_.clear();
    }
    return absl::OkStatus();
  }

 private:
  Stage<std::string>& input_;
  const int required_schedules_;
  std::string current_text_;
  int remaining_steps_ = 0;
  std::string flush_text_;
  absl::Status flush_error_ = absl::OkStatus();
};

// Stage that emits pre-queued `Output` variants on `Flush()`.
class CannedOutputStage : public SingleThreadedStageWithDeque<Output> {
 public:
  explicit CannedOutputStage(std::vector<Output> flush_outputs)
      : flush_outputs_(std::move(flush_outputs)) {}

 protected:
  bool NeedScheduleInternal() const override { return false; }
  absl::Status ScheduleInternal() override {
    SetState(State::kIdle);
    return absl::OkStatus();
  }
  absl::Status FlushInternal() override {
    for (auto& out : flush_outputs_) {
      PushOutput(std::move(out));
    }
    flush_outputs_.clear();
    return absl::OkStatus();
  }

 private:
  std::vector<Output> flush_outputs_;
};

TEST(MultiStagedSessionTest, CreateValidation) {
  // 1. Empty stages vector fails.
  auto dummy = std::make_unique<CannedOutputStage>(std::vector<Output>{});
  EXPECT_THAT(
      MultiStagedSession::Create(
          std::vector<std::unique_ptr<internal::StageBase>>{}, dummy.get()),
      StatusIs(absl::StatusCode::kInvalidArgument));

  // 2. Null stage entry in stages fails.
  {
    auto out_stage = std::make_unique<CannedOutputStage>(std::vector<Output>{});
    Stage<Output>* raw_out = out_stage.get();
    std::vector<std::unique_ptr<internal::StageBase>> stages;
    stages.push_back(nullptr);
    stages.push_back(std::move(out_stage));
    EXPECT_THAT(MultiStagedSession::Create(std::move(stages), raw_out),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }

  // 3. Null output_stage fails.
  {
    auto out_stage = std::make_unique<CannedOutputStage>(std::vector<Output>{});
    std::vector<std::unique_ptr<internal::StageBase>> stages;
    stages.push_back(std::move(out_stage));
    Stage<Output>* null_out = nullptr;
    EXPECT_THAT(MultiStagedSession::Create(std::move(stages), null_out),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }

  // 4. output_stage != stages.back() fails.
  {
    auto stage1 = std::make_unique<CannedOutputStage>(std::vector<Output>{});
    auto stage2 = std::make_unique<CannedOutputStage>(std::vector<Output>{});
    Stage<Output>* wrong_out = stage1.get();
    std::vector<std::unique_ptr<internal::StageBase>> stages;
    stages.push_back(std::move(stage1));
    stages.push_back(std::move(stage2));
    EXPECT_THAT(MultiStagedSession::Create(std::move(stages), wrong_out),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }
}

TEST(MultiStagedSessionTest, ProcessNextLoopsForMultiScheduleStage) {
  auto source = std::make_unique<FakeStringSource>();
  FakeStringSource* raw_source = source.get();
  // Require 3 Schedule() calls before MultiStepTextStage produces an output.
  auto text_stage = std::make_unique<MultiStepTextStage>(
      raw_source, /*required_schedules=*/3);
  Stage<Output>* raw_text_stage = text_stage.get();

  std::vector<std::unique_ptr<internal::StageBase>> stages;
  stages.push_back(std::move(source));
  stages.push_back(std::move(text_stage));

  ASSERT_OK_AND_ASSIGN(auto session, MultiStagedSession::Create(
                                         std::move(stages), raw_text_stage));
  EXPECT_EQ(session->stages().size(), 2);

  // Without input, ProcessNext() returns NotFoundError.
  EXPECT_THAT(session->ProcessNext(), StatusIs(absl::StatusCode::kNotFound));

  // Push one item: ProcessNext() loops through all 3 steps and returns the
  // TextOutput in a single call.
  raw_source->Push("chunk_one");
  ASSERT_OK_AND_ASSIGN(Output out, session->ProcessNext());
  ASSERT_TRUE(std::holds_alternative<TextOutput>(out));
  EXPECT_EQ(std::get<TextOutput>(out).confirmed_text, "chunk_one");

  // Once finished and drained, ProcessNext() returns OutOfRangeError.
  raw_source->Finish();
  EXPECT_THAT(session->ProcessNext(), StatusIs(absl::StatusCode::kOutOfRange));
}

TEST(MultiStagedSessionTest, ProcessAsyncFlushesAllStagesInOrderOnEos) {
  auto source = std::make_unique<FakeStringSource>();
  FakeStringSource* raw_source = source.get();
  auto text_stage = std::make_unique<MultiStepTextStage>(raw_source, 1);
  MultiStepTextStage* raw_text_stage = text_stage.get();

  raw_source->Push("stream_item");
  raw_source->SetFlushItem("source_flushed");
  raw_source->Finish();
  raw_text_stage->SetFlushText("stage_flushed");

  std::vector<std::unique_ptr<internal::StageBase>> stages;
  stages.push_back(std::move(source));
  stages.push_back(std::move(text_stage));

  ::litert::lm::ThreadPool pool("async_eos_pool", 2);
  ASSERT_OK_AND_ASSIGN(
      auto session,
      MultiStagedSession::Create(std::move(stages), raw_text_stage, &pool));

  std::vector<std::string> received;
  absl::Status terminal_status;
  absl::Notification done;

  ASSERT_OK(
      session->ProcessAsync([&](absl::StatusOr<Output> result) -> absl::Status {
        if (!result.ok()) {
          terminal_status = result.status();
          done.Notify();
          return result.status();
        }
        if (const auto* text = std::get_if<TextOutput>(&*result)) {
          received.push_back(text->confirmed_text);
        }
        return absl::OkStatus();
      }));

  done.WaitForNotification();
  EXPECT_THAT(terminal_status, StatusIs(absl::StatusCode::kOutOfRange));
  EXPECT_THAT(received,
              ElementsAre("stream_item", "source_flushed", "stage_flushed"));
}

TEST(MultiStagedSessionTest, ProcessAsyncPropagatesStageFlushErrorToCallback) {
  auto source = std::make_unique<FakeStringSource>();
  FakeStringSource* raw_source = source.get();
  auto text_stage = std::make_unique<MultiStepTextStage>(raw_source, 1);
  MultiStepTextStage* raw_text_stage = text_stage.get();

  raw_source->Push("item");
  raw_source->Finish();
  raw_text_stage->SetFlushError(absl::InternalError("Vocoder flush failed"));

  std::vector<std::unique_ptr<internal::StageBase>> stages;
  stages.push_back(std::move(source));
  stages.push_back(std::move(text_stage));

  ::litert::lm::ThreadPool pool("async_flush_err_pool", 2);
  ASSERT_OK_AND_ASSIGN(
      auto session,
      MultiStagedSession::Create(std::move(stages), raw_text_stage, &pool));

  absl::Status terminal_status;
  absl::Notification done;

  ASSERT_OK(
      session->ProcessAsync([&](absl::StatusOr<Output> result) -> absl::Status {
        if (!result.ok()) {
          terminal_status = result.status();
          done.Notify();
          return result.status();
        }
        return absl::OkStatus();
      }));

  done.WaitForNotification();
  EXPECT_THAT(terminal_status, StatusIs(absl::StatusCode::kInternal));
  EXPECT_EQ(terminal_status.message(), "Vocoder flush failed");
}

TEST(MultiStagedSessionTest, FlushTwiceAndAfterResetAndSequentialSynthesize) {
  auto source = std::make_unique<FakeStringSource>();
  FakeStringSource* raw_source = source.get();
  auto text_stage = std::make_unique<MultiStepTextStage>(raw_source, 2);
  MultiStepTextStage* raw_text_stage = text_stage.get();

  std::vector<std::unique_ptr<internal::StageBase>> stages;
  stages.push_back(std::move(source));
  stages.push_back(std::move(text_stage));

  ASSERT_OK_AND_ASSIGN(auto session, MultiStagedSession::Create(
                                         std::move(stages), raw_text_stage));

  // First synthesize + flush cycle.
  raw_source->Push("first");
  raw_source->SetFlushItem("second");
  ASSERT_OK_AND_ASSIGN(Output flushed1, session->Flush());
  ASSERT_TRUE(std::holds_alternative<TextOutput>(flushed1));
  EXPECT_EQ(std::get<TextOutput>(flushed1).confirmed_text, "first second");

  // Calling Flush() a second time with no new input returns EndOfOutput.
  ASSERT_OK_AND_ASSIGN(Output flushed2, session->Flush());
  EXPECT_TRUE(std::holds_alternative<EndOfOutput>(flushed2));

  // Calling Reset() then Flush() on an empty session returns EndOfOutput.
  session->Reset();
  ASSERT_OK_AND_ASSIGN(Output flushed_after_reset, session->Flush());
  EXPECT_TRUE(std::holds_alternative<EndOfOutput>(flushed_after_reset));

  // Second sequential synthesize + flush cycle after Reset().
  session->Reset();
  raw_source->Push("third");
  ASSERT_OK_AND_ASSIGN(Output flushed3, session->Flush());
  ASSERT_TRUE(std::holds_alternative<TextOutput>(flushed3));
  EXPECT_EQ(std::get<TextOutput>(flushed3).confirmed_text, "third");
}

TEST(MultiStagedSessionTest, CombineOutputErrorsOnMismatchedTypesOrRates) {
  // 1. Mismatched variant types (TextOutput then AudioOutput).
  {
    auto stage = std::make_unique<CannedOutputStage>(std::vector<Output>{
        TextOutput{.confirmed_text = "hello"},
        AudioOutput{.pcm_samples = {0.1f}, .sample_rate_hz = 16000}});
    Stage<Output>* raw_stage = stage.get();
    std::vector<std::unique_ptr<internal::StageBase>> stages;
    stages.push_back(std::move(stage));
    ASSERT_OK_AND_ASSIGN(
        auto session, MultiStagedSession::Create(std::move(stages), raw_stage));
    EXPECT_THAT(session->Flush(), StatusIs(absl::StatusCode::kInternal));
  }

  // 2. Mismatched AudioOutput sample_rate_hz.
  {
    auto stage = std::make_unique<CannedOutputStage>(std::vector<Output>{
        AudioOutput{.pcm_samples = {0.1f}, .sample_rate_hz = 16000},
        AudioOutput{.pcm_samples = {0.2f}, .sample_rate_hz = 24000}});
    Stage<Output>* raw_stage = stage.get();
    std::vector<std::unique_ptr<internal::StageBase>> stages;
    stages.push_back(std::move(stage));
    ASSERT_OK_AND_ASSIGN(
        auto session, MultiStagedSession::Create(std::move(stages), raw_stage));
    EXPECT_THAT(session->Flush(), StatusIs(absl::StatusCode::kInternal));
  }
}

class BlockingForeverSource : public SingleThreadedStageWithDeque<std::string> {
 protected:
  bool NeedScheduleInternal() const override { return false; }
  absl::Status ScheduleInternal() override {
    SetState(State::kIdle);
    return absl::OkStatus();
  }
};

TEST(MultiStagedSessionTest, ResetAndFlushStopActiveAsyncScheduler) {
  auto source = std::make_unique<BlockingForeverSource>();
  auto text_stage = std::make_unique<MultiStepTextStage>(source.get(), 1);
  MultiStepTextStage* raw_text_stage = text_stage.get();

  std::vector<std::unique_ptr<internal::StageBase>> stages;
  stages.push_back(std::move(source));
  stages.push_back(std::move(text_stage));

  ::litert::lm::ThreadPool pool("active_async_pool", 2);
  ASSERT_OK_AND_ASSIGN(
      auto session,
      MultiStagedSession::Create(std::move(stages), raw_text_stage, &pool));

  ASSERT_OK(session->ProcessAsync(
      [](absl::StatusOr<Output>) { return absl::OkStatus(); }));
  absl::SleepFor(absl::Milliseconds(20));

  // Flush() stops the active async scheduler and drains stages safely.
  raw_text_stage->SetFlushText("flushed_while_async");
  ASSERT_OK_AND_ASSIGN(Output flushed, session->Flush());
  ASSERT_TRUE(std::holds_alternative<TextOutput>(flushed));
  EXPECT_EQ(std::get<TextOutput>(flushed).confirmed_text,
            "flushed_while_async");

  // ProcessAsync() can be started again after Flush(), and Reset() stops it.
  ASSERT_OK(session->ProcessAsync(
      [](absl::StatusOr<Output>) { return absl::OkStatus(); }));
  absl::SleepFor(absl::Milliseconds(20));
  session->Reset();
}

}  // namespace
}  // namespace litert::omni
