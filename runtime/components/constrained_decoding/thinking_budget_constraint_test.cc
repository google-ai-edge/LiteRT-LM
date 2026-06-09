// Copyright 2025 The ODML Authors.
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

#include "runtime/components/constrained_decoding/thinking_budget_constraint.h"

#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "runtime/components/constrained_decoding/bitmap.h"
#include "runtime/components/constrained_decoding/fake_constraint.h"
#include "runtime/util/test_utils.h"  // NOLINT

namespace litert::lm {
namespace {

TEST(ThinkingBudgetConstraintTest, TestStart) {
  std::vector<int> start_tokens = {10, 11};
  std::vector<int> end_tokens = {12, 13};
  ThinkingBudgetConstraint constraint(nullptr, 5, start_tokens, end_tokens,
                                      100);

  auto state_ptr = constraint.Start();
  ASSERT_NE(state_ptr, nullptr);
  auto* state =
      static_cast<ThinkingBudgetConstraint::ThinkingState*>(state_ptr.get());

  EXPECT_TRUE(state->in_thinking);
  EXPECT_EQ(state->thinking_token_count, 0);
  EXPECT_EQ(state->forced_end_token_index, -1);
  EXPECT_EQ(state->matching_start_index, 0);
}

TEST(ThinkingBudgetConstraintTest, TestStartBudgetZero) {
  std::vector<int> start_tokens = {10, 11};
  std::vector<int> end_tokens = {12, 13};
  ThinkingBudgetConstraint constraint(nullptr, 0, start_tokens, end_tokens,
                                      100);

  auto state_ptr = constraint.Start();
  ASSERT_NE(state_ptr, nullptr);
  auto* state =
      static_cast<ThinkingBudgetConstraint::ThinkingState*>(state_ptr.get());

  EXPECT_TRUE(state->in_thinking);
  EXPECT_EQ(state->thinking_token_count, 0);
  EXPECT_EQ(state->forced_end_token_index, 0);  // Immediately forcing end.
}

TEST(ThinkingBudgetConstraintTest, TestStartBudgetZeroNoEndTokens) {
  std::vector<int> start_tokens = {10, 11};
  std::vector<int> end_tokens = {};
  ThinkingBudgetConstraint constraint(nullptr, 0, start_tokens, end_tokens,
                                      100);

  auto state_ptr = constraint.Start();
  ASSERT_NE(state_ptr, nullptr);
  auto* state =
      static_cast<ThinkingBudgetConstraint::ThinkingState*>(state_ptr.get());

  EXPECT_FALSE(
      state->in_thinking);  // Transitions out immediately if no end tokens.
  EXPECT_EQ(state->forced_end_token_index, -1);
}

TEST(ThinkingBudgetConstraintTest, TestThinkingOnlyCountTokens) {
  std::vector<int> start_tokens = {10, 11};
  std::vector<int> end_tokens = {12, 13};
  ThinkingBudgetConstraint constraint(nullptr, 5, start_tokens, end_tokens,
                                      100);

  auto state = constraint.Start();

  // Feed some normal tokens.
  for (int i = 0; i < 3; ++i) {
    ASSERT_OK_AND_ASSIGN(state, constraint.ComputeNext(*state, 50 + i));
    auto* s =
        static_cast<ThinkingBudgetConstraint::ThinkingState*>(state.get());
    EXPECT_TRUE(s->in_thinking);
    EXPECT_EQ(s->thinking_token_count, i + 1);
  }
}

TEST(ThinkingBudgetConstraintTest, TestThinkingSkipStartTokens) {
  std::vector<int> start_tokens = {10, 11};
  std::vector<int> end_tokens = {12, 13};
  ThinkingBudgetConstraint constraint(nullptr, 5, start_tokens, end_tokens,
                                      100);

  auto state = constraint.Start();

  // Feed start tokens: 10, 11
  ASSERT_OK_AND_ASSIGN(state, constraint.ComputeNext(*state, 10));
  auto* s = static_cast<ThinkingBudgetConstraint::ThinkingState*>(state.get());
  EXPECT_TRUE(s->in_thinking);
  EXPECT_EQ(s->thinking_token_count, 0);  // Start tokens shouldn't count.
  EXPECT_EQ(s->matching_start_index, 1);

  ASSERT_OK_AND_ASSIGN(state, constraint.ComputeNext(*state, 11));
  s = static_cast<ThinkingBudgetConstraint::ThinkingState*>(state.get());
  EXPECT_TRUE(s->in_thinking);
  EXPECT_EQ(s->thinking_token_count, 0);
  EXPECT_EQ(s->matching_start_index, -1);  // Finished matching start.

  // Feed a normal token.
  ASSERT_OK_AND_ASSIGN(state, constraint.ComputeNext(*state, 50));
  s = static_cast<ThinkingBudgetConstraint::ThinkingState*>(state.get());
  EXPECT_TRUE(s->in_thinking);
  EXPECT_EQ(s->thinking_token_count, 1);
}

TEST(ThinkingBudgetConstraintTest, TestThinkingBudgetExceeded) {
  std::vector<int> start_tokens = {10, 11};
  std::vector<int> end_tokens = {12, 13};
  ThinkingBudgetConstraint constraint(nullptr, 3, start_tokens, end_tokens,
                                      100);

  auto state = constraint.Start();

  // Feed 3 normal tokens.
  for (int i = 0; i < 3; ++i) {
    ASSERT_OK_AND_ASSIGN(state, constraint.ComputeNext(*state, 50 + i));
  }
  auto* s = static_cast<ThinkingBudgetConstraint::ThinkingState*>(state.get());
  EXPECT_TRUE(s->in_thinking);
  EXPECT_EQ(s->thinking_token_count, 3);
  EXPECT_EQ(s->forced_end_token_index,
            0);  // Budget exceeded, should start forcing.

  // Verify bitmap only allows first end token (12).
  ASSERT_OK_AND_ASSIGN(auto bitmap, constraint.ComputeBitmap(*state));
  EXPECT_TRUE(bitmap->Get(12));
  EXPECT_FALSE(bitmap->Get(50));

  // Feed the forced end tokens.
  ASSERT_OK_AND_ASSIGN(state, constraint.ComputeNext(*state, 12));
  s = static_cast<ThinkingBudgetConstraint::ThinkingState*>(state.get());
  EXPECT_TRUE(s->in_thinking);
  EXPECT_EQ(s->forced_end_token_index, 1);

  ASSERT_OK_AND_ASSIGN(state, constraint.ComputeNext(*state, 13));
  s = static_cast<ThinkingBudgetConstraint::ThinkingState*>(state.get());
  EXPECT_FALSE(s->in_thinking);  // Should be out of thinking now.
  EXPECT_EQ(s->forced_end_token_index, -1);
}

TEST(ThinkingBudgetConstraintTest, TestNaturalEnd) {
  std::vector<int> start_tokens = {10, 11};
  std::vector<int> end_tokens = {12, 13};
  ThinkingBudgetConstraint constraint(nullptr, 5, start_tokens, end_tokens,
                                      100);

  auto state = constraint.Start();

  // Feed 2 normal tokens.
  ASSERT_OK_AND_ASSIGN(state, constraint.ComputeNext(*state, 50));
  ASSERT_OK_AND_ASSIGN(state, constraint.ComputeNext(*state, 51));

  // Feed end tokens naturally: 12, 13
  ASSERT_OK_AND_ASSIGN(state, constraint.ComputeNext(*state, 12));
  auto* s = static_cast<ThinkingBudgetConstraint::ThinkingState*>(state.get());
  EXPECT_TRUE(s->in_thinking);
  EXPECT_EQ(s->natural_match_index, 1);

  ASSERT_OK_AND_ASSIGN(state, constraint.ComputeNext(*state, 13));
  s = static_cast<ThinkingBudgetConstraint::ThinkingState*>(state.get());
  EXPECT_FALSE(s->in_thinking);  // Transitions out naturally.
  EXPECT_EQ(s->natural_match_index, 0);
}

TEST(ThinkingBudgetConstraintTest, TestWithUserConstraint) {
  std::vector<int> start_tokens = {10, 11};
  std::vector<int> end_tokens = {12, 13};
  // User constraint forces [20, 21, 1] (1 is stop token).
  FakeConstraint user_constraint({20, 21, 1}, 100);

  ThinkingBudgetConstraint constraint(&user_constraint, 3, start_tokens,
                                      end_tokens, 100);

  auto state = constraint.Start();

  // During thinking, user constraint should not restrict anything (except when
  // forcing end, but here we are not).
  ASSERT_OK_AND_ASSIGN(auto bitmap, constraint.ComputeBitmap(*state));
  EXPECT_TRUE(bitmap->Get(20));
  EXPECT_TRUE(bitmap->Get(99));  // Any token allowed by default in thinking.

  // Feed 3 normal tokens to exceed budget.
  for (int i = 0; i < 3; ++i) {
    ASSERT_OK_AND_ASSIGN(state, constraint.ComputeNext(*state, 50 + i));
  }

  // Feed forced end tokens.
  ASSERT_OK_AND_ASSIGN(state, constraint.ComputeNext(*state, 12));
  ASSERT_OK_AND_ASSIGN(state, constraint.ComputeNext(*state, 13));

  auto* s = static_cast<ThinkingBudgetConstraint::ThinkingState*>(state.get());
  EXPECT_FALSE(s->in_thinking);
  ASSERT_NE(s->user_state, nullptr);

  // Now user constraint should be active.
  ASSERT_OK_AND_ASSIGN(bitmap, constraint.ComputeBitmap(*state));
  EXPECT_TRUE(bitmap->Get(20));
  EXPECT_FALSE(bitmap->Get(99));  // Only 20 allowed now by FakeConstraint.

  // Feed user tokens.
  ASSERT_OK_AND_ASSIGN(state, constraint.ComputeNext(*state, 20));
  ASSERT_OK_AND_ASSIGN(bitmap, constraint.ComputeBitmap(*state));
  EXPECT_TRUE(bitmap->Get(21));
  EXPECT_FALSE(bitmap->Get(20));

  ASSERT_OK_AND_ASSIGN(state, constraint.ComputeNext(*state, 21));
  EXPECT_FALSE(constraint.IsEnded(*state));

  ASSERT_OK_AND_ASSIGN(state, constraint.ComputeNext(*state, 1));
  EXPECT_TRUE(constraint.IsEnded(*state));
}

}  // namespace
}  // namespace litert::lm
