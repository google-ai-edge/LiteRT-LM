// Copyright 2024 The MediaPipe Authors.
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

#include "runtime/executor/gpu_artisan/ml_drift/heuristic_drafter.h"

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "runtime/executor/gpu_artisan/ml_drift/speculative_utils.h"
#include "runtime/util/test_utils.h"

namespace odml::infra::gpu {

namespace {

using ::ml_drift::speculative_utils::TokenAtTimeStep;

TEST(HeuristicDrafterTest, Basic) {
  struct {
    std::vector<int> input_tokens;
    int seed_token;
    std::vector<int> expected_draft_tokens;
  } test_cases[] = {
      // No historical match at all. Just returns the seed token.
      {{11, 12, 13, 14, 15, 16}, 17, {17}},
      // Really short input (shorter than gamma). Just returns the seed token.
      {{11}, 17, {17}},
      // Basic historical match.
      {{11, 12, 13, 14, 15, 16}, 12, {12, 13, 14, 15}},
      // Basic historical match, but too near the end of the input, so there
      // aren't at least gamma tokens to predict.
      {{11, 12, 13, 14, 15, 16}, 14, {14}},
      // Two equally relevant historical matches, pick the more recent one.
      {{11, 12, 13, 14, 15, 16, 17, 18, 19, 14, 141, 142, 143},
       14,
       {14, 141, 142, 143}},
      // Historical match length 2 is better than historical match length 1.
      {{11, 12, 13, 14, 15, 16, 17, 18, 19, 14, 141, 142, 143, 13},
       14,
       {14, 15, 16, 17}},
      // Robust match length 3 is better than historical match length 1.
      {{11, 12, 13, 14, 15, 16, 17, 18, 19, 14, 141, 142, 143, 12, 12},
       14,
       {14, 15, 16, 17}},
      // Given two robust matches, the higher weight is given to the one with
      // the more proximal exact match.
      {{
           // sequence that matches robustly, and will be chosen
           10,
           12,
           12,
           13,
           14,
           // tokens after first match (will be predicted)
           30,
           31,
           32,
           // some tokens in between the first and second match
           42,
           43,
           44,
           44,
           45,
           // sequence that matches robustly, but will NOT be chosen
           10,
           11,
           11,
           13,
           14,
           // tokens after second match (will NOT be predicted)
           50,
           51,
           52,
           // most recent tokens (to find matches for)
           10,
           11,
           12,
           13,
       },
       14,
       {14, 30, 31, 32}},
  };

  for (auto& test_case : test_cases) {
    HeuristicDrafter drafter(/*gamma=*/3);
    EXPECT_OK(
        drafter.InitInputTokens(test_case.input_tokens, 0, true, nullptr));

    ASSERT_OK_AND_ASSIGN(
        std::vector<int> tokens,
        drafter.DraftTokensAfter(TokenAtTimeStep(test_case.input_tokens.size(),
                                                 test_case.seed_token)));
    EXPECT_THAT(tokens,
                testing::ElementsAreArray(test_case.expected_draft_tokens));
  }
}

TEST(HeuristicDrafterTest, MultipleRounds) {
  HeuristicDrafter drafter(/*gamma=*/3);
  // This is the expected state of the world.
  std::vector<int> token_state = {11, 12, 13, 14, 15, 16, 17};
  EXPECT_OK(drafter.InitInputTokens(token_state, 0, true, nullptr));

  // Verify that seed tokens and resampled tokens are inserted into the
  // input/output stream as appropriate.
  const int kSeededToken = 13;
  const int kResampledToken = 42;
  int main_model_time_step = token_state.size();
  {
    ASSERT_OK_AND_ASSIGN(std::vector<int> tokens,
                         drafter.DraftTokensAfter(TokenAtTimeStep(
                             main_model_time_step, kSeededToken)));
    EXPECT_THAT(tokens, testing::ElementsAreArray({13, 14, 15, 16}));

    ASSERT_OK(drafter.RejectAfterResample(
        TokenAtTimeStep(++main_model_time_step, kResampledToken)));
  }
  token_state.push_back(kSeededToken);
  token_state.push_back(kResampledToken);
  EXPECT_THAT(drafter.GetTokenStateForTesting(),
              testing::ElementsAreArray(token_state));

  {
    ASSERT_OK_AND_ASSIGN(
        std::vector<int> tokens,
        drafter.DraftTokensAfter(TokenAtTimeStep(++main_model_time_step, 14)));
    EXPECT_THAT(tokens, testing::ElementsAreArray({14, 15, 16, 17}));
    ASSERT_OK(drafter.RejectAfterResample(
        TokenAtTimeStep(++main_model_time_step, 43)));
  }
  token_state.push_back(14);
  token_state.push_back(43);
  EXPECT_THAT(drafter.GetTokenStateForTesting(),
              testing::ElementsAreArray(token_state));

  {
    ASSERT_OK_AND_ASSIGN(
        std::vector<int> tokens,
        drafter.DraftTokensAfter(TokenAtTimeStep(++main_model_time_step, 16)));
    EXPECT_THAT(tokens, testing::ElementsAreArray(
                            {16, 17, kSeededToken, kResampledToken}));
    // accept drafted tokens by not calling RejectAfterResample.
  }
  token_state.push_back(16);
  token_state.push_back(17);
  token_state.push_back(kSeededToken);
  token_state.push_back(kResampledToken);
  EXPECT_THAT(drafter.GetTokenStateForTesting(),
              testing::ElementsAreArray(token_state));
}

TEST(HeuristicDrafterTest, MultipleInitInputTokens) {
  HeuristicDrafter drafter(/*gamma=*/3);

  {
    int current_step = 0;
    std::vector<int> input_tokens = {11, 12, 13, 14, 15, 16};
    int seed_token = 12;
    std::vector<int> expected_draft_tokens = {12, 13, 14, 15};

    EXPECT_OK(
        drafter.InitInputTokens(input_tokens, current_step, true, nullptr));
    ASSERT_OK_AND_ASSIGN(std::vector<int> tokens,
                         drafter.DraftTokensAfter(TokenAtTimeStep(
                             current_step + input_tokens.size(), seed_token)));
    EXPECT_THAT(tokens, testing::ElementsAreArray(expected_draft_tokens));
  }

  {
    int current_step = 3;
    std::vector<int> input_tokens = {97, 98, 99};
    int seed_token = 12;
    std::vector<int> expected_draft_tokens = {12, 13, 97, 98};

    EXPECT_OK(
        drafter.InitInputTokens(input_tokens, current_step, true, nullptr));
    ASSERT_OK_AND_ASSIGN(std::vector<int> tokens,
                         drafter.DraftTokensAfter(TokenAtTimeStep(
                             current_step + input_tokens.size(), seed_token)));
    EXPECT_THAT(tokens, testing::ElementsAreArray(expected_draft_tokens));
  }
}

}  // namespace
}  // namespace odml::infra::gpu
