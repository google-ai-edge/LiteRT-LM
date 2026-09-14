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

#include "omni/tts/kokoro/common.h"

#include <cstddef>
#include <vector>

#include <gtest/gtest.h>
#include "omni/tts/kokoro/kokoro_io_types.h"

namespace litert::omni::tts {
namespace {

TEST(CommonTest, ConstantsSanity) {
  EXPECT_EQ(kokoro::kMaxTokens, 256);
  EXPECT_EQ(kokoro::kSampleRate, 24000);
  EXPECT_EQ(kokoro::kAudioHop, 600);
  EXPECT_EQ(kokoro::kVoicePackSize, 510 * 256);
  EXPECT_EQ(kokoro::kBosTokenId, 0);
  EXPECT_EQ(kokoro::kEosTokenId, 0);
  EXPECT_EQ(kokoro::kSpaceTokenId, 16);
}

TEST(CommonTest, LoadVoiceEmbeddingNotFound) {
  auto embed =
      kokoro::LoadVoiceEmbedding("/non_existent_dir", "non_existent_voice");
  EXPECT_FALSE(embed.ok());
}

TEST(CommonTest, SliceTokenIdsUnderCapacityReturnsSingleSlice) {
  std::vector<int> tokens = {kokoro::kBosTokenId, 20, 21, 22,
                             kokoro::kEosTokenId};
  auto slices = kokoro::SliceTokenIds(tokens, /*max_capacity=*/10);
  ASSERT_EQ(slices.size(), 1);
  EXPECT_EQ(slices[0].join_before, SliceJoin::kChunkBoundary);
  EXPECT_EQ(slices[0].join_after, SliceJoin::kChunkBoundary);
  EXPECT_EQ(slices[0].token_ids, tokens);
}

TEST(CommonTest, SliceTokenIdsPrefersPunctuationOverSpace) {
  std::vector<int> tokens = {kokoro::kBosTokenId, 20, 21, 22, 23, 24,
                             kokoro::kSpaceTokenId, 3, 25, 26, 27, 28,
                             kokoro::kEosTokenId};
  auto slices = kokoro::SliceTokenIds(tokens, /*max_capacity=*/10);
  ASSERT_GE(slices.size(), 2);
  EXPECT_EQ(slices[0].join_after, SliceJoin::kPunctuation);
  EXPECT_EQ(slices[1].join_before, SliceJoin::kPunctuation);
  EXPECT_EQ(slices[0].token_ids.front(), kokoro::kBosTokenId);
  EXPECT_EQ(slices[0].token_ids.back(), kokoro::kEosTokenId);
  EXPECT_EQ(slices[0].token_ids[slices[0].token_ids.size() - 2], 3);
}

TEST(CommonTest, SliceTokenIdsFallsBackToSpaceBoundary) {
  std::vector<int> tokens = {kokoro::kBosTokenId, 20, 21, 22, 23, 24,
                             kokoro::kSpaceTokenId, 25, 26, 27, 28, 29,
                             kokoro::kEosTokenId};
  auto slices = kokoro::SliceTokenIds(tokens, /*max_capacity=*/10);
  ASSERT_GE(slices.size(), 2);
  EXPECT_EQ(slices[0].join_after, SliceJoin::kWordBoundary);
  EXPECT_EQ(slices[1].join_before, SliceJoin::kWordBoundary);
}

TEST(CommonTest, SliceTokenIdsMidWordCutWhenNoBoundary) {
  std::vector<int> tokens(25, 20);
  tokens.front() = kokoro::kBosTokenId;
  tokens.back() = kokoro::kEosTokenId;
  auto slices = kokoro::SliceTokenIds(tokens, /*max_capacity=*/10);
  ASSERT_GE(slices.size(), 3);
  for (size_t i = 0; i < slices.size() - 1; ++i) {
    EXPECT_EQ(slices[i].join_after, SliceJoin::kWordBoundary);
    EXPECT_EQ(slices[i + 1].join_before, SliceJoin::kWordBoundary);
  }
}

TEST(CommonTest, SliceTokenIdsAdjacentSlicesAgreeOnJoins) {
  std::vector<int> tokens;
  for (int i = 0; i < 60; ++i) {
    if (i % 7 == 0) {
      tokens.push_back(kokoro::kMinPunctuationTokenId);
    } else if (i % 4 == 0) {
      tokens.push_back(kokoro::kSpaceTokenId);
    } else {
      tokens.push_back(30 + (i % 10));
    }
  }
  auto slices = kokoro::SliceTokenIds(tokens, /*max_capacity=*/15);
  ASSERT_GT(slices.size(), 1);
  EXPECT_EQ(slices.front().join_before, SliceJoin::kChunkBoundary);
  EXPECT_EQ(slices.back().join_after, SliceJoin::kChunkBoundary);
  for (size_t i = 0; i < slices.size() - 1; ++i) {
    EXPECT_EQ(slices[i].join_after, slices[i + 1].join_before);
    EXPECT_EQ(slices[i].token_ids.front(), kokoro::kBosTokenId);
    EXPECT_EQ(slices[i].token_ids.back(), kokoro::kEosTokenId);
  }
}

TEST(CommonTest, TrimSliceJoinSilenceChunkBoundaryBothEndsUntouched) {
  std::vector<float> pcm = {0.0f, 0.0f, 0.5f, 0.8f, 0.0f, 0.0f};
  std::vector<float> expected = pcm;
  kokoro::TrimSliceJoinSilence(pcm, SliceJoin::kChunkBoundary,
                               SliceJoin::kChunkBoundary);
  EXPECT_EQ(pcm, expected);
}

TEST(CommonTest, TrimSliceJoinSilenceDigitalSilenceTruncated) {
  std::vector<float> pcm(5000, 0.0f);
  kokoro::TrimSliceJoinSilence(pcm, SliceJoin::kWordBoundary,
                               SliceJoin::kWordBoundary);
  EXPECT_EQ(pcm.size(),
            static_cast<size_t>(kokoro::kSliceJoinMarginSamples * 2));
}

TEST(CommonTest, TrimSliceJoinSilenceTrimsLeadingAndTrailingSilenceAtWordBoundary) {
  std::vector<float> pcm(500, 0.0f);
  pcm.push_back(1.0f);
  pcm.insert(pcm.end(), 500, 0.0f);

  kokoro::TrimSliceJoinSilence(pcm, SliceJoin::kWordBoundary,
                               SliceJoin::kWordBoundary);
  EXPECT_EQ(pcm.size(),
            static_cast<size_t>(kokoro::kSliceJoinMarginSamples * 2 + 1));
  EXPECT_EQ(pcm[kokoro::kSliceJoinMarginSamples], 1.0f);
}

TEST(CommonTest, TrimSliceJoinSilencePunctuationPausePreservedOnTail) {
  std::vector<float> pcm(500, 0.0f);
  pcm.push_back(1.0f);
  pcm.insert(pcm.end(), 20000, 0.0f);

  kokoro::TrimSliceJoinSilence(pcm, SliceJoin::kChunkBoundary,
                               SliceJoin::kPunctuation);
  const size_t expected_size =
      500 + 1 + static_cast<size_t>(kokoro::kSlicePunctuationPauseSamples);
  EXPECT_EQ(pcm.size(), expected_size);
  EXPECT_EQ(pcm[500], 1.0f);
}

}  // namespace
}  // namespace litert::omni::tts
