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

#include "runtime/components/constrained_decoding/repetition_penalty_constraint.h"

#include <limits>
#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "runtime/components/constrained_decoding/logit_mask.h"
#include "runtime/components/constrained_decoding/repetition_penalty_config.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep
#include "tflite/types/half.h"  // from @litert

namespace litert::lm {
namespace {

using ::testing::status::StatusIs;

enum class LogitsVariant {
  kFloat,
  kHalf,
};

class RepetitionPenaltyConstraintParamTest
    : public ::testing::TestWithParam<LogitsVariant> {
 protected:
  template <typename Func>
  void RunWithParam(Func&& func) {
    switch (GetParam()) {
      case LogitsVariant::kFloat:
        func(float{});
        break;
      case LogitsVariant::kHalf:
        func(tflite::half{});
        break;
    }
  }
};

TEST_P(RepetitionPenaltyConstraintParamTest, AppliesMultiplicativePenalty) {
  RepetitionPenaltyConfig config(
      /*repetition_penalty=*/1.2f, /*presence_penalty=*/0.0f,
      /*frequency_penalty=*/0.0f, /*window_size=*/0);
  constexpr int kVocabSize = 3;
  RepetitionPenaltyConstraint constraint(kVocabSize, config);

  EXPECT_EQ(constraint.GetVocabularySize(), kVocabSize);

  auto state = constraint.Start();
  ASSERT_NE(state, nullptr);
  EXPECT_FALSE(constraint.IsEnded(*state));

  // Before seeing any tokens, mask is empty/no-op
  {
    ASSERT_OK_AND_ASSIGN(auto mask, constraint.ComputeMask(*state));
    EXPECT_EQ(mask->GetType(), MaskType::kCustom);
    auto* rpm = static_cast<RepetitionPenaltyMask*>(mask.get());
    EXPECT_TRUE(rpm->entries().empty());
  }

  // Update with token 1
  ASSERT_OK_AND_ASSIGN(state, constraint.ComputeNext(*state, 1));

  ASSERT_OK_AND_ASSIGN(auto mask, constraint.ComputeMask(*state));
  EXPECT_EQ(mask->GetType(), MaskType::kCustom);

  RunWithParam([&](auto dummy_type) {
    using T = decltype(dummy_type);
    std::vector<T> logits = {static_cast<T>(2.0f), static_cast<T>(1.2f),
                             static_cast<T>(3.0f)};

    EXPECT_OK(mask->Apply(absl::MakeSpan(logits)));

    // Token 0 and 2 were not seen
    EXPECT_FLOAT_EQ(static_cast<float>(logits[0]), 2.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(logits[2]), 3.0f);

    // Token 1 was seen, scaled down: 1.2 / 1.2 = 1.0f
    EXPECT_FLOAT_EQ(static_cast<float>(logits[1]), 1.0f);
  });
}

TEST_P(RepetitionPenaltyConstraintParamTest,
       AppliesMultiplicativePenaltyToNegativeLogits) {
  RepetitionPenaltyConfig config(
      /*repetition_penalty=*/1.5f, /*presence_penalty=*/0.0f,
      /*frequency_penalty=*/0.0f, /*window_size=*/0);
  constexpr int kVocabSize = 2;
  RepetitionPenaltyConstraint constraint(kVocabSize, config);

  auto state = constraint.Start();
  ASSERT_OK_AND_ASSIGN(state, constraint.ComputeNext(*state, 0));

  ASSERT_OK_AND_ASSIGN(auto mask, constraint.ComputeMask(*state));

  RunWithParam([&](auto dummy_type) {
    using T = decltype(dummy_type);
    std::vector<T> logits = {static_cast<T>(-3.0f), static_cast<T>(-3.0f)};

    EXPECT_OK(mask->Apply(absl::MakeSpan(logits)));

    // Token 0 was seen, scaled to be more negative: -3.0 * 1.5 = -4.5f
    EXPECT_FLOAT_EQ(static_cast<float>(logits[0]), -4.5f);
    // Token 1 was not seen: unchanged -3.0f
    EXPECT_FLOAT_EQ(static_cast<float>(logits[1]), -3.0f);
  });
}

TEST_P(RepetitionPenaltyConstraintParamTest,
       AppliesPresenceAndFrequencyPenalties) {
  RepetitionPenaltyConfig config(
      /*repetition_penalty=*/1.0f, /*presence_penalty=*/2.0f,
      /*frequency_penalty=*/1.0f, /*window_size=*/0);
  constexpr int kVocabSize = 3;
  RepetitionPenaltyConstraint constraint(kVocabSize, config);

  auto state = constraint.Start();
  // Token 0 once, Token 1 twice
  ASSERT_OK_AND_ASSIGN(state, constraint.ComputeNext(*state, 0));
  ASSERT_OK_AND_ASSIGN(state, constraint.ComputeNext(*state, 1));
  ASSERT_OK_AND_ASSIGN(state, constraint.ComputeNext(*state, 1));

  ASSERT_OK_AND_ASSIGN(auto mask, constraint.ComputeMask(*state));

  RunWithParam([&](auto dummy_type) {
    using T = decltype(dummy_type);
    std::vector<T> logits = {static_cast<T>(10.0f), static_cast<T>(10.0f),
                             static_cast<T>(10.0f)};

    EXPECT_OK(mask->Apply(absl::MakeSpan(logits)));

    // Token 0 seen 1 time: Presence (2.0) + Freq (1 * 1.0) = -3.0 -> 7.0f
    EXPECT_FLOAT_EQ(static_cast<float>(logits[0]), 7.0f);

    // Token 1 seen 2 times: Presence (2.0) + Freq (2 * 1.0) = -4.0 -> 6.0f
    EXPECT_FLOAT_EQ(static_cast<float>(logits[1]), 6.0f);

    // Token 2 seen 0 times -> 10.0f
    EXPECT_FLOAT_EQ(static_cast<float>(logits[2]), 10.0f);
  });
}

TEST_P(RepetitionPenaltyConstraintParamTest,
       AppliesMultiplicativeThenSubtractive) {
  RepetitionPenaltyConfig config(
      /*repetition_penalty=*/1.5f, /*presence_penalty=*/1.0f,
      /*frequency_penalty=*/0.0f, /*window_size=*/0);
  constexpr int kVocabSize = 1;
  RepetitionPenaltyConstraint constraint(kVocabSize, config);

  auto state = constraint.Start();
  ASSERT_OK_AND_ASSIGN(state, constraint.ComputeNext(*state, 0));

  ASSERT_OK_AND_ASSIGN(auto mask, constraint.ComputeMask(*state));

  RunWithParam([&](auto dummy_type) {
    using T = decltype(dummy_type);
    std::vector<T> logits = {static_cast<T>(3.0f)};

    EXPECT_OK(mask->Apply(absl::MakeSpan(logits)));

    // (3.0 / 1.5) - 1.0 = 2.0 - 1.0 = 1.0
    EXPECT_FLOAT_EQ(static_cast<float>(logits[0]), 1.0f);
  });
}

TEST_P(RepetitionPenaltyConstraintParamTest,
       HandlesNegativePresenceAndFrequencyPenalties) {
  RepetitionPenaltyConfig config(
      /*repetition_penalty=*/3.0f, /*presence_penalty=*/-1.0f,
      /*frequency_penalty=*/-2.0f, /*window_size=*/0);
  constexpr int kVocabSize = 1;
  RepetitionPenaltyConstraint constraint(kVocabSize, config);

  auto state = constraint.Start();
  ASSERT_OK_AND_ASSIGN(state, constraint.ComputeNext(*state, 0));

  ASSERT_OK_AND_ASSIGN(auto mask, constraint.ComputeMask(*state));

  RunWithParam([&](auto dummy_type) {
    using T = decltype(dummy_type);
    std::vector<T> logits = {static_cast<T>(-3.0f)};

    EXPECT_OK(mask->Apply(absl::MakeSpan(logits)));

    // Repetition: -3.0 * 3.0 = -9.0
    // Presence: -1.0
    // Freq: 1 * -2.0 = -2.0.
    // Logit = -9.0 - (-1.0) - (-2.0) = -6.0f.
    EXPECT_FLOAT_EQ(static_cast<float>(logits[0]), -6.0f);
  });
}

TEST_P(RepetitionPenaltyConstraintParamTest, ObservesSlidingWindowLength) {
  RepetitionPenaltyConfig config(
      /*repetition_penalty=*/1.0f, /*presence_penalty=*/0.0f,
      /*frequency_penalty=*/1.0f, /*window_size=*/2);
  constexpr int kVocabSize = 3;
  RepetitionPenaltyConstraint constraint(kVocabSize, config);

  auto state = constraint.Start();
  ASSERT_OK_AND_ASSIGN(state, constraint.ComputeNext(*state, 0));
  ASSERT_OK_AND_ASSIGN(state, constraint.ComputeNext(*state, 1));
  ASSERT_OK_AND_ASSIGN(state, constraint.ComputeNext(*state, 2));

  ASSERT_OK_AND_ASSIGN(auto mask, constraint.ComputeMask(*state));

  RunWithParam([&](auto dummy_type) {
    using T = decltype(dummy_type);
    std::vector<T> logits = {static_cast<T>(10.0f), static_cast<T>(10.0f),
                             static_cast<T>(10.0f)};

    EXPECT_OK(mask->Apply(absl::MakeSpan(logits)));

    // Token 0 fell off window -> untouched (10.0f)
    EXPECT_FLOAT_EQ(static_cast<float>(logits[0]), 10.0f);

    // Token 1 in window (once) -> 9.0f
    EXPECT_FLOAT_EQ(static_cast<float>(logits[1]), 9.0f);

    // Token 2 in window (once) -> 9.0f
    EXPECT_FLOAT_EQ(static_cast<float>(logits[2]), 9.0f);
  });
}

TEST_P(RepetitionPenaltyConstraintParamTest, PreservesDisallowedTokens) {
  RepetitionPenaltyConfig config(
      /*repetition_penalty=*/1.5f, /*presence_penalty=*/1.0f,
      /*frequency_penalty=*/0.5f, /*window_size=*/0);
  constexpr int kVocabSize = 2;
  RepetitionPenaltyConstraint constraint(kVocabSize, config);

  auto state = constraint.Start();
  ASSERT_OK_AND_ASSIGN(state, constraint.ComputeNext(*state, 0));
  ASSERT_OK_AND_ASSIGN(state, constraint.ComputeNext(*state, 1));

  ASSERT_OK_AND_ASSIGN(auto mask, constraint.ComputeMask(*state));

  RunWithParam([&](auto dummy_type) {
    using T = decltype(dummy_type);
    T min_val;
    if constexpr (std::is_same_v<T, float>) {
      min_val = -std::numeric_limits<float>::infinity();
    } else {
      min_val = tflite::half::min();
    }

    std::vector<T> logits = {min_val, static_cast<T>(-2.0f)};

    EXPECT_OK(mask->Apply(absl::MakeSpan(logits)));

    // Token 0: Disallowed, must remain -inf / min
    if constexpr (std::is_same_v<T, float>) {
      EXPECT_TRUE(std::isinf(logits[0]) && logits[0] < 0.0f);
    } else {
      EXPECT_EQ(logits[0], tflite::half::min());
    }

    // Token 1: -2.0 * 1.5 - (1.0 + 1 * 0.5) = -3.0 - 1.5 = -4.5f
    EXPECT_FLOAT_EQ(static_cast<float>(logits[1]), -4.5f);
  });
}

TEST(RepetitionPenaltyConstraintTest, HandlesDisabledConfig) {
  RepetitionPenaltyConfig config = RepetitionPenaltyConfig::Default();
  constexpr int kVocabSize = 3;
  RepetitionPenaltyConstraint constraint(kVocabSize, config);

  auto state = constraint.Start();
  ASSERT_OK_AND_ASSIGN(state, constraint.ComputeNext(*state, 1));

  ASSERT_OK_AND_ASSIGN(auto mask, constraint.ComputeMask(*state));
  std::vector<float> logits = {2.0f, 1.2f, -1.2f};
  EXPECT_OK(mask->Apply(absl::MakeSpan(logits)));

  EXPECT_FLOAT_EQ(logits[0], 2.0f);
  EXPECT_FLOAT_EQ(logits[1], 1.2f);
  EXPECT_FLOAT_EQ(logits[2], -1.2f);
}

TEST(RepetitionPenaltyConstraintTest, ComputeNextValidatesTokens) {
  RepetitionPenaltyConfig config(1.2f, 0.0f, 0.0f, 0);
  constexpr int kVocabSize = 5;
  RepetitionPenaltyConstraint constraint(kVocabSize, config);

  auto state = constraint.Start();
  EXPECT_THAT(constraint.ComputeNext(*state, -1),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(constraint.ComputeNext(*state, 5),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

INSTANTIATE_TEST_SUITE_P(RepetitionPenaltyConstraintParamTests,
                         RepetitionPenaltyConstraintParamTest,
                         ::testing::Values(LogitsVariant::kFloat,
                                           LogitsVariant::kHalf));

}  // namespace
}  // namespace litert::lm
