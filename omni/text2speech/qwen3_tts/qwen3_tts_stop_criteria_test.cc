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

#include "omni/text2speech/qwen3_tts/qwen3_tts_stop_criteria.h"

#include <gtest/gtest.h>

namespace litert::omni::tts {
namespace {

constexpr int kTextTokens = 20;
constexpr int kMaxFrames = 512;

TEST(Qwen3TtsStopCriteriaTest, FrameLimitScalesWithTextLength) {
  EXPECT_EQ(Qwen3TtsStopCriteria(kTextTokens, kMaxFrames).FrameLimit(),
            Qwen3TtsStopCriteria::kMaxFramesPerTextToken * kTextTokens +
                Qwen3TtsStopCriteria::kExtraFrames);
}

TEST(Qwen3TtsStopCriteriaTest, FrameLimitIsCappedByMaxFrames) {
  EXPECT_EQ(Qwen3TtsStopCriteria(/*text_tokens=*/1000, kMaxFrames).FrameLimit(),
            kMaxFrames);
}

TEST(Qwen3TtsStopCriteriaTest, DoesNotStopOnVaryingTokens) {
  Qwen3TtsStopCriteria criteria(kTextTokens, kMaxFrames);
  for (int frame = 0; frame < 100; ++frame) {
    EXPECT_FALSE(criteria.ShouldStop(frame, /*cb0=*/frame % 3)) << frame;
  }
}

TEST(Qwen3TtsStopCriteriaTest, StopsAtFrameLimit) {
  Qwen3TtsStopCriteria criteria(kTextTokens, kMaxFrames);
  const int limit = criteria.FrameLimit();
  for (int frame = 0; frame < limit - 1; ++frame) {
    ASSERT_FALSE(criteria.ShouldStop(frame, /*cb0=*/frame % 3)) << frame;
  }
  EXPECT_TRUE(criteria.ShouldStop(limit - 1, /*cb0=*/0));
}

TEST(Qwen3TtsStopCriteriaTest, StopsOnRepeatedTokenAfterText) {
  Qwen3TtsStopCriteria criteria(kTextTokens, kMaxFrames);
  int frame = 0;
  for (; frame < 50; ++frame) {
    ASSERT_FALSE(criteria.ShouldStop(frame, /*cb0=*/frame % 3)) << frame;
  }
  for (int i = 1; i < Qwen3TtsStopCriteria::kMaxRepeatedFrames; ++i, ++frame) {
    ASSERT_FALSE(criteria.ShouldStop(frame, /*cb0=*/1657)) << frame;
  }
  EXPECT_TRUE(criteria.ShouldStop(frame, /*cb0=*/1657));
}

TEST(Qwen3TtsStopCriteriaTest, ShortRepeatsDoNotStop) {
  Qwen3TtsStopCriteria criteria(kTextTokens, kMaxFrames);
  int frame = 0;
  for (int run = 0; run < 10; ++run) {
    for (int i = 0; i < Qwen3TtsStopCriteria::kMaxRepeatedFrames - 1;
         ++i, ++frame) {
      ASSERT_FALSE(criteria.ShouldStop(frame, /*cb0=*/run)) << frame;
    }
  }
}

TEST(Qwen3TtsStopCriteriaTest, RepeatsWhileTextIsFedDoNotStop) {
  Qwen3TtsStopCriteria criteria(kTextTokens, kMaxFrames);
  for (int frame = 0; frame < kTextTokens; ++frame) {
    EXPECT_FALSE(criteria.ShouldStop(frame, /*cb0=*/7)) << frame;
  }
}

TEST(Qwen3TtsStopCriteriaTest, SuppressesNothingInitially) {
  EXPECT_EQ(Qwen3TtsStopCriteria(kTextTokens, kMaxFrames).SuppressedToken(),
            -1);
}

TEST(Qwen3TtsStopCriteriaTest, SuppressesTokenRepeatedWhileTextIsFed) {
  Qwen3TtsStopCriteria criteria(kTextTokens, kMaxFrames);
  int frame = 0;
  for (; frame < 4; ++frame) {
    criteria.ShouldStop(frame, /*cb0=*/frame);
  }
  for (int i = 1; i < Qwen3TtsStopCriteria::kMaxRepeatedFramesWhileText;
       ++i, ++frame) {
    criteria.ShouldStop(frame, /*cb0=*/668);
    EXPECT_EQ(criteria.SuppressedToken(), -1) << frame;
  }
  criteria.ShouldStop(frame++, /*cb0=*/668);
  EXPECT_EQ(criteria.SuppressedToken(), 668);
  criteria.ShouldStop(frame, /*cb0=*/617);
  EXPECT_EQ(criteria.SuppressedToken(), -1);
}

TEST(Qwen3TtsStopCriteriaTest, DoesNotSuppressAfterText) {
  Qwen3TtsStopCriteria criteria(kTextTokens, kMaxFrames);
  for (int frame = 0; frame < kTextTokens - 1; ++frame) {
    criteria.ShouldStop(frame, /*cb0=*/frame);
  }
  // The next frames are fed the TTS pad embedding, where trailing silence and
  // the codec EOS are expected.
  for (int frame = kTextTokens - 1; frame < kTextTokens + 6; ++frame) {
    criteria.ShouldStop(frame, /*cb0=*/668);
    EXPECT_EQ(criteria.SuppressedToken(), -1) << frame;
  }
}

}  // namespace
}  // namespace litert::omni::tts
