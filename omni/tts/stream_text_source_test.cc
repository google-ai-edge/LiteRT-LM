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

#include "omni/tts/stream_text_source.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl

namespace litert::omni::tts {
namespace {

using ::absl_testing::IsOk;

TEST(StreamTextSourceTest, PushTextAndSchedule) {
  StreamTextSource source;

  EXPECT_FALSE(source.NeedSchedule());
  ASSERT_THAT(source.PushText("Hello "), IsOk());
  EXPECT_FALSE(source.NeedSchedule());

  ASSERT_THAT(source.PushText("world. "), IsOk());
  EXPECT_TRUE(source.NeedSchedule());

  ASSERT_THAT(source.Schedule(), IsOk());
  auto out1 = source.GetOutput();
  ASSERT_THAT(out1, IsOk());
  EXPECT_EQ(*out1, "Hello world.");

  EXPECT_FALSE(source.NeedSchedule());
  ASSERT_THAT(source.PushText("How are you?"), IsOk());
  EXPECT_TRUE(source.NeedSchedule());

  ASSERT_THAT(source.Schedule(), IsOk());
  auto out2 = source.GetOutput();
  ASSERT_THAT(out2, IsOk());
  EXPECT_EQ(*out2, " How are you?");

  ASSERT_THAT(source.PushText(" Unfinished tail"), IsOk());
  EXPECT_FALSE(source.NeedSchedule());

  source.Finish();
  EXPECT_TRUE(source.IsFinished());
  EXPECT_TRUE(source.NeedSchedule());

  ASSERT_THAT(source.Schedule(), IsOk());
  auto out3 = source.GetOutput();
  ASSERT_THAT(out3, IsOk());
  EXPECT_EQ(*out3, " Unfinished tail");

  EXPECT_FALSE(source.NeedSchedule());
}

TEST(StreamTextSourceTest, PushTextAfterFinishFails) {
  StreamTextSource source;
  source.Finish();
  EXPECT_TRUE(source.IsFinished());

  auto status = source.PushText("more text");
  EXPECT_TRUE(absl::IsFailedPrecondition(status));
}

TEST(StreamTextSourceTest, ResetClearsState) {
  StreamTextSource source;
  ASSERT_THAT(source.PushText("First line.\n"), IsOk());
  ASSERT_THAT(source.Schedule(), IsOk());
  source.Finish();
  EXPECT_TRUE(source.IsFinished());

  source.Reset();
  EXPECT_FALSE(source.IsFinished());
  EXPECT_FALSE(source.NeedSchedule());

  ASSERT_THAT(source.PushText("New line.\n"), IsOk());
  EXPECT_TRUE(source.NeedSchedule());
  ASSERT_THAT(source.Schedule(), IsOk());
  auto out = source.GetOutput();
  ASSERT_THAT(out, IsOk());
  EXPECT_EQ(*out, "New line.");
}

}  // namespace
}  // namespace litert::omni::tts
