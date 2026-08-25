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

#include "omni/base/stage.h"

#include <deque>
#include <string>
#include <utility>

#include <gtest/gtest.h>
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl

namespace litert::omni {
namespace {

template <typename T>
class TestStageWithDeque : public SingleThreadedStageWithDeque<T> {
 public:
  void AddInput(T input) { inputs_.push_front(std::move(input)); }

 protected:
  void ResetInternal() override {
    SingleThreadedStageWithDeque<T>::ResetInternal();
    inputs_.clear();
  }

  bool NeedScheduleInternal() const override { return !inputs_.empty(); }

  absl::Status ScheduleInternal() override {
    absl::Cleanup cleanup = [this] {
      this->SetState(SingleThreadedStageWithDeque<T>::State::kIdle);
    };
    if (inputs_.empty()) {
      return absl::OkStatus();
    }

    this->PushOutput(std::move(inputs_.front()));
    inputs_.pop_front();
    return absl::OkStatus();
  }

 private:
  std::deque<T> inputs_;
};

TEST(StageTest, StageWithDequeLifecycle) {
  TestStageWithDeque<std::string> stage;

  EXPECT_FALSE(stage.NeedSchedule());
  EXPECT_FALSE(stage.HasOutput());
  EXPECT_TRUE(absl::IsNotFound(stage.GetOutput().status()));

  stage.AddInput("item1");
  EXPECT_TRUE(stage.NeedSchedule());
  EXPECT_FALSE(stage.HasOutput());

  EXPECT_TRUE(stage.Schedule().ok());
  EXPECT_FALSE(stage.NeedSchedule());
  EXPECT_TRUE(stage.HasOutput());

  auto item = stage.GetOutput();
  EXPECT_TRUE(item.ok());
  EXPECT_EQ(*item, "item1");
  EXPECT_FALSE(stage.HasOutput());
  EXPECT_TRUE(absl::IsNotFound(stage.GetOutput().status()));
}

TEST(StageTest, ResetClearsOutputsAndResetsState) {
  TestStageWithDeque<std::string> stage;
  stage.AddInput("item1");
  EXPECT_TRUE(stage.Schedule().ok());
  EXPECT_TRUE(stage.HasOutput());

  stage.Reset();
  EXPECT_FALSE(stage.HasOutput());
  EXPECT_TRUE(stage.IsIdle());
  EXPECT_TRUE(absl::IsNotFound(stage.GetOutput().status()));
}

}  // namespace
}  // namespace litert::omni
