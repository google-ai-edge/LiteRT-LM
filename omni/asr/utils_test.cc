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

#include "omni/asr/utils.h"

#include <optional>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "omni/asr/speech_recognizer.h"

namespace litert::omni::asr {
namespace {

using ::testing::ElementsAre;
using ::testing::Field;
using ::testing::IsEmpty;

std::vector<SpeechRecognizer::DecodedToken> MakeTokens(
    const std::vector<int>& token_ids) {
  std::vector<SpeechRecognizer::DecodedToken> tokens;
  tokens.reserve(token_ids.size());
  for (int id : token_ids) {
    tokens.push_back(SpeechRecognizer::DecodedToken{
        .token_id = id, .timestamp_ms = std::nullopt});
  }
  return tokens;
}

TEST(UtilsTest, IsSilentAudioReturnsTrueForEmptyAndZeroAudio) {
  EXPECT_TRUE(IsSilentAudio({}));
  EXPECT_TRUE(IsSilentAudio(std::vector<float>(1600, 0.0f)));
}

TEST(UtilsTest, IsSilentAudioReturnsTrueBelowRmsAndPeakThresholds) {
  EXPECT_TRUE(IsSilentAudio(std::vector<float>(1600, 5e-4f)));
}

TEST(UtilsTest, IsSilentAudioReturnsFalseWhenRmsExceedsThreshold) {
  EXPECT_FALSE(IsSilentAudio(std::vector<float>(1600, 2e-3f)));
}

TEST(UtilsTest, IsSilentAudioReturnsFalseWhenPeakExceedsThreshold) {
  std::vector<float> samples(1600, 0.0f);
  samples[0] = 6e-3f;
  EXPECT_FALSE(IsSilentAudio(samples));
}

TEST(UtilsTest, EmptyTokensReturnsFalse) {
  std::vector<SpeechRecognizer::DecodedToken> tokens;
  EXPECT_FALSE(TruncateOnTrailingRepetition(tokens, 1));
  EXPECT_THAT(tokens, IsEmpty());
}

TEST(UtilsTest, FewerThanFourRepeatsReturnsFalseAndPreservesTokens) {
  auto tokens = MakeTokens({1, 4, 4});
  EXPECT_FALSE(TruncateOnTrailingRepetition(tokens, 4));
  EXPECT_THAT(tokens,
              ElementsAre(Field(&SpeechRecognizer::DecodedToken::token_id, 1),
                          Field(&SpeechRecognizer::DecodedToken::token_id, 4),
                          Field(&SpeechRecognizer::DecodedToken::token_id, 4)));
}

TEST(UtilsTest, SingleTokenRepeatedFourTimesTruncatesExtraCopies) {
  auto tokens = MakeTokens({1, 4, 4, 4});
  EXPECT_TRUE(TruncateOnTrailingRepetition(tokens, 4));
  EXPECT_THAT(tokens,
              ElementsAre(Field(&SpeechRecognizer::DecodedToken::token_id, 1),
                          Field(&SpeechRecognizer::DecodedToken::token_id, 4)));
}

TEST(UtilsTest, TwoGramRepeatedFourTimesTruncatesExtraCopies) {
  auto tokens = MakeTokens({1, 2, 3, 2, 3, 2, 3, 2});
  EXPECT_TRUE(TruncateOnTrailingRepetition(tokens, 3));
  EXPECT_THAT(tokens,
              ElementsAre(Field(&SpeechRecognizer::DecodedToken::token_id, 1),
                          Field(&SpeechRecognizer::DecodedToken::token_id, 2),
                          Field(&SpeechRecognizer::DecodedToken::token_id, 3)));
}

TEST(UtilsTest, ThreeGramRepeatedFourTimesTruncatesExtraCopies) {
  auto tokens = MakeTokens({9, 5, 6, 7, 5, 6, 7, 5, 6, 7, 5, 6});
  EXPECT_TRUE(TruncateOnTrailingRepetition(tokens, 7));
  EXPECT_THAT(tokens,
              ElementsAre(Field(&SpeechRecognizer::DecodedToken::token_id, 9),
                          Field(&SpeechRecognizer::DecodedToken::token_id, 5),
                          Field(&SpeechRecognizer::DecodedToken::token_id, 6),
                          Field(&SpeechRecognizer::DecodedToken::token_id, 7)));
}

TEST(UtilsTest, SixteenGramRepeatedFourTimesTruncatesExtraCopies) {
  std::vector<int> ids;
  for (int r = 0; r < 4; ++r) {
    for (int i = 0; i < 16; ++i) {
      if (r == 3 && i == 15) continue;
      ids.push_back(100 + i);
    }
  }
  auto tokens = MakeTokens(ids);
  EXPECT_TRUE(TruncateOnTrailingRepetition(tokens, 115));
  ASSERT_EQ(tokens.size(), 16);
  for (int i = 0; i < 16; ++i) {
    EXPECT_EQ(tokens[i].token_id, 100 + i);
  }
}

TEST(UtilsTest, SeventeenGramRepeatedFourTimesIsNotTruncated) {
  std::vector<int> ids;
  for (int r = 0; r < 4; ++r) {
    for (int i = 0; i < 17; ++i) {
      if (r == 3 && i == 16) continue;
      ids.push_back(100 + i);
    }
  }
  auto tokens = MakeTokens(ids);
  EXPECT_FALSE(TruncateOnTrailingRepetition(tokens, 116));
  EXPECT_EQ(tokens.size(), 4 * 17 - 1);
}

}  // namespace
}  // namespace litert::omni::asr
