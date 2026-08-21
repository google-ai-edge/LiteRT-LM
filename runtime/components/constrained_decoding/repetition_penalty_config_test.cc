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

#include "runtime/components/constrained_decoding/repetition_penalty_config.h"

#include <tuple>

#include <gtest/gtest.h>

namespace litert::lm {
namespace {

TEST(RepetitionPenaltyConfigTest, InitializesWithGivenValues) {
  RepetitionPenaltyConfig config(/*repetition_penalty=*/1.2f,
                                 /*presence_penalty=*/0.5f,
                                 /*frequency_penalty=*/0.1f,
                                 /*window_size=*/10);

  EXPECT_FLOAT_EQ(config.repetition_penalty(), 1.2f);
  EXPECT_FLOAT_EQ(config.presence_penalty(), 0.5f);
  EXPECT_FLOAT_EQ(config.frequency_penalty(), 0.1f);
  EXPECT_EQ(config.window_size(), 10);
  EXPECT_TRUE(config.enabled());
}

TEST(RepetitionPenaltyConfigTest, ClampsRepetitionPenaltyAndWindowSize) {
  RepetitionPenaltyConfig config(/*repetition_penalty=*/-1.0f,
                                 /*presence_penalty=*/-0.5f,
                                 /*frequency_penalty=*/-0.1f,
                                 /*window_size=*/-10);

  EXPECT_FLOAT_EQ(config.repetition_penalty(), 1.0f);
  EXPECT_EQ(config.window_size(), 0);

  // Presence and frequency penalties are not clamped.
  EXPECT_FLOAT_EQ(config.presence_penalty(), -0.5f);
  EXPECT_FLOAT_EQ(config.frequency_penalty(), -0.1f);
}

TEST(RepetitionPenaltyConfigTest, DefaultConfigIsDisabled) {
  EXPECT_FALSE(RepetitionPenaltyConfig::Default().enabled());
}

class RepetitionPenaltyConfigEnabledTest
    : public ::testing::TestWithParam<std::tuple<float, float, float, int>> {};

TEST_P(RepetitionPenaltyConfigEnabledTest, EvaluatesEnabledCorrectly) {
  float repetition_penalty = std::get<0>(GetParam());
  float presence_penalty = std::get<1>(GetParam());
  float frequency_penalty = std::get<2>(GetParam());
  int window_size = std::get<3>(GetParam());

  RepetitionPenaltyConfig config(repetition_penalty, presence_penalty,
                                 frequency_penalty, window_size);

  bool expected_enabled =
      (repetition_penalty > 1.0f || presence_penalty != 0.0f ||
       frequency_penalty != 0.0f);
  EXPECT_EQ(config.enabled(), expected_enabled);
}

INSTANTIATE_TEST_SUITE_P(
    PenaltyCombinations, RepetitionPenaltyConfigEnabledTest,
    ::testing::Combine(
        ::testing::Values(1.0f, 1.2f),   // repetition_penalty (1.0 = inactive)
        ::testing::Values(0.0f, -0.5f),  // presence_penalty (0.0 = inactive)
        ::testing::Values(0.0f, 0.1f),   // frequency_penalty (0.0 = inactive)
        ::testing::Values(10, 10)));     // window_size (irrelevant)

}  // namespace
}  // namespace litert::lm
