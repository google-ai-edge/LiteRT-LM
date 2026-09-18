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

#include "omni/base/async_stage_scheduler.h"

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/synchronization/notification.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "omni/base/stage.h"
#include "runtime/framework/threadpool.h"

namespace litert::omni {
namespace {

// Dummy source stage producing items from a list.
class TestSourceStage : public SingleThreadedStageWithDeque<int> {
 public:
  explicit TestSourceStage(std::vector<int> items) : items_(std::move(items)) {}

 protected:
  void ResetInternal() override { index_ = 0; }

  bool NeedScheduleInternal() const override { return index_ < items_.size(); }

  absl::Status ScheduleInternal() override {
    absl::Cleanup cleanup = [this] { SetState(State::kIdle); };
    if (index_ >= items_.size()) {
      return absl::OutOfRangeError("End of stream");
    }
    PushOutput(items_[index_++]);
    return absl::OkStatus();
  }

 private:
  std::vector<int> items_;
  size_t index_ = 0;
};

// Dummy transform stage taking int and outputting string.
class TestTransformStage : public SingleThreadedStageWithDeque<std::string> {
 public:
  explicit TestTransformStage(Stage<int>* input_stage)
      : input_stage_(*input_stage) {}

 protected:
  bool NeedScheduleInternal() const override {
    return input_stage_.HasOutput();
  }

  absl::Status ScheduleInternal() override {
    absl::Cleanup cleanup = [this] { SetState(State::kIdle); };
    auto item = input_stage_.GetOutput();
    if (absl::IsNotFound(item.status())) {
      return absl::OkStatus();
    } else if (!item.ok()) {
      return item.status();
    }
    PushOutput("item_" + std::to_string(*item));
    return absl::OkStatus();
  }

 private:
  Stage<int>& input_stage_;
};

TEST(AsyncStageSchedulerTest, FullPipelineAsyncFlow) {
  TestSourceStage stage1({10, 20, 30});
  TestTransformStage stage2(&stage1);

  ::litert::lm::ThreadPool pool("test_pool", 4);
  std::vector<std::string> results;
  absl::Notification done;

  std::vector<internal::StageBase*> stages = {&stage1, &stage2};
  AsyncStageScheduler<std::string> scheduler(
      stages, &stage2, &pool,
      [&results, &done](absl::StatusOr<std::string> res) -> absl::Status {
        if (!res.ok()) {
          done.Notify();
          return res.status();
        }
        results.push_back(*res);
        return absl::OkStatus();
      });

  ASSERT_TRUE(scheduler.Start().ok());
  done.WaitForNotification();

  ASSERT_EQ(results.size(), 3);
  EXPECT_EQ(results[0], "item_10");
  EXPECT_EQ(results[1], "item_20");
  EXPECT_EQ(results[2], "item_30");
}

TEST(AsyncStageSchedulerTest, RejectsDoubleStart) {
  TestSourceStage stage1({1});
  TestTransformStage stage2(&stage1);

  ::litert::lm::ThreadPool pool("test_pool", 4);
  std::vector<internal::StageBase*> stages = {&stage1, &stage2};
  absl::Notification done;

  AsyncStageScheduler<std::string> scheduler(
      stages, &stage2, &pool,
      [&done](absl::StatusOr<std::string> res) -> absl::Status {
        if (!res.ok()) {
          done.Notify();
        }
        return res.status();
      });

  ASSERT_TRUE(scheduler.Start().ok());
  EXPECT_FALSE(scheduler.Start().ok());

  done.WaitForNotification();
}

TEST(AsyncStageSchedulerTest, StopSafely) {
  TestSourceStage stage1({1, 2, 3});
  TestTransformStage stage2(&stage1);

  ::litert::lm::ThreadPool pool("test_pool", 4);
  std::vector<internal::StageBase*> stages = {&stage1, &stage2};

  AsyncStageScheduler<std::string> scheduler(
      stages, &stage2, &pool,
      [](absl::StatusOr<std::string> res) -> absl::Status {
        return absl::InternalError("Stop requested");
      });

  ASSERT_TRUE(scheduler.Start().ok());
  EXPECT_TRUE(scheduler.Stop(absl::Seconds(1)).ok());
}

class IdleSourceStage : public SingleThreadedStageWithDeque<int> {
 protected:
  bool NeedScheduleInternal() const override { return false; }
  absl::Status ScheduleInternal() override { return absl::OkStatus(); }
};

TEST(AsyncStageSchedulerTest, StopWhileWaitingForIdleStageDoesNotDeadlock) {
  IdleSourceStage stage1;
  TestTransformStage stage2(&stage1);

  ::litert::lm::ThreadPool pool("test_pool", 4);
  std::vector<internal::StageBase*> stages = {&stage1, &stage2};

  AsyncStageScheduler<std::string> scheduler(
      stages, &stage2, &pool,
      [](absl::StatusOr<std::string> res) -> absl::Status {
        return absl::OkStatus();
      });

  ASSERT_TRUE(scheduler.Start().ok());
  // Give worker thread time to enter WaitForAnyStagesReadyOrStopped().
  absl::SleepFor(absl::Milliseconds(20));

  absl::Time start_time = absl::Now();
  EXPECT_TRUE(scheduler.Stop(absl::Seconds(5)).ok());
  EXPECT_LT(absl::Now() - start_time, absl::Seconds(1));
}

class BlockingSourceStage : public SingleThreadedStageWithDeque<int> {
 public:
  void NotifyTaskStarted() { task_started_.Notify(); }
  void WaitForTaskStarted() { task_started_.WaitForNotification(); }
  void AllowTaskFinish() { allow_finish_.Notify(); }
  bool IsTaskFinished() const { return task_finished_.HasBeenNotified(); }

 protected:
  bool NeedScheduleInternal() const override { return !scheduled_; }

  absl::Status ScheduleInternal() override {
    absl::Cleanup cleanup = [this] {
      SetState(State::kIdle);
      task_finished_.Notify();
    };
    scheduled_ = true;
    task_started_.Notify();
    allow_finish_.WaitForNotification();
    PushOutput(42);
    return absl::OkStatus();
  }

 private:
  bool scheduled_ = false;
  absl::Notification task_started_;
  absl::Notification allow_finish_;
  absl::Notification task_finished_;
};

TEST(AsyncStageSchedulerTest, StopWaitsForInFlightTaskCompletion) {
  BlockingSourceStage stage1;
  TestTransformStage stage2(&stage1);

  ::litert::lm::ThreadPool pool("test_pool", 4);
  std::vector<internal::StageBase*> stages = {&stage1, &stage2};

  AsyncStageScheduler<std::string> scheduler(
      stages, &stage2, &pool,
      [](absl::StatusOr<std::string> res) -> absl::Status {
        return absl::OkStatus();
      });

  ASSERT_TRUE(scheduler.Start().ok());
  stage1.WaitForTaskStarted();

  // Unblock stage1 shortly after calling Stop() on another thread.
  ::litert::lm::ThreadPool helper_pool("helper_pool", 1);
  ASSERT_TRUE(helper_pool
                  .Schedule([&stage1]() {
                    absl::SleepFor(absl::Milliseconds(50));
                    stage1.AllowTaskFinish();
                  })
                  .ok());

  EXPECT_TRUE(scheduler.Stop(absl::Seconds(5)).ok());
  // By the time Stop() returns, the in-flight stage task must have completed.
  EXPECT_TRUE(stage1.IsTaskFinished());
}

}  // namespace
}  // namespace litert::omni
