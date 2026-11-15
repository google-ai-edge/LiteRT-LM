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

#include "runtime/components/constrained_decoding/suppress_tokens_constraint.h"

#include <cmath>
#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "runtime/components/constrained_decoding/logit_mask.h"
#include "runtime/components/constrained_decoding/suppress_tokens_config.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep
#include "tflite/types/half.h"  // from @litert

namespace litert::lm {
namespace {

using ::testing::status::StatusIs;

enum class LogitsVariant {
  kFloat,
  kHalf,
};

class SuppressTokensConstraintParamTest
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

TEST_P(SuppressTokensConstraintParamTest, SuppressesTokensProperly) {
  constexpr int kVocabSize = 10;
  SuppressTokensConfig config({2, 5, 8});
  SuppressTokensConstraint constraint(kVocabSize, config);

  EXPECT_EQ(constraint.GetVocabularySize(), kVocabSize);

  auto state = constraint.Start();
  ASSERT_NE(state, nullptr);
  EXPECT_FALSE(constraint.IsEnded(*state));

  ASSERT_OK_AND_ASSIGN(auto mask, constraint.ComputeMask(*state));
  ASSERT_NE(mask, nullptr);
  EXPECT_EQ(mask->GetType(), MaskType::kBitmap);

  RunWithParam([&](auto dummy_type) {
    using T = decltype(dummy_type);
    std::vector<T> logits(kVocabSize);
    for (int i = 0; i < kVocabSize; ++i) {
      logits[i] = static_cast<T>(1.0f + 0.5f * i);
    }

    EXPECT_OK(mask->Apply(absl::MakeSpan(logits)));

    for (int i = 0; i < kVocabSize; ++i) {
      if (i == 2 || i == 5 || i == 8) {
        if constexpr (std::is_same_v<T, float>) {
          EXPECT_TRUE(std::isinf(logits[i]) && logits[i] < 0.0f);
        } else {
          EXPECT_EQ(logits[i], tflite::half::min());
        }
      } else {
        EXPECT_FLOAT_EQ(static_cast<float>(logits[i]), 1.0f + 0.5f * i);
      }
    }
  });
}

TEST(SuppressTokensConstraintTest, HandlesDisabledConfig) {
  constexpr int kVocabSize = 5;
  SuppressTokensConfig config = SuppressTokensConfig::Default();
  SuppressTokensConstraint constraint(kVocabSize, config);

  auto state = constraint.Start();
  ASSERT_OK_AND_ASSIGN(auto mask, constraint.ComputeMask(*state));

  std::vector<float> logits = {2.0f, 1.2f, -1.2f, 0.5f, 0.0f};
  EXPECT_OK(mask->Apply(absl::MakeSpan(logits)));

  EXPECT_FLOAT_EQ(logits[0], 2.0f);
  EXPECT_FLOAT_EQ(logits[1], 1.2f);
  EXPECT_FLOAT_EQ(logits[2], -1.2f);
  EXPECT_FLOAT_EQ(logits[3], 0.5f);
  EXPECT_FLOAT_EQ(logits[4], 0.0f);
}

TEST(SuppressTokensConstraintTest, IgnoresOutOfRangeTokenIds) {
  constexpr int kVocabSize = 5;
  SuppressTokensConfig config({-1, 10, 5, 2});
  SuppressTokensConstraint constraint(kVocabSize, config);

  auto state = constraint.Start();
  ASSERT_OK_AND_ASSIGN(auto mask, constraint.ComputeMask(*state));

  std::vector<float> logits = {2.0f, 1.2f, -1.2f, 0.5f, 0.0f};
  EXPECT_OK(mask->Apply(absl::MakeSpan(logits)));

  EXPECT_FLOAT_EQ(logits[0], 2.0f);
  EXPECT_FLOAT_EQ(logits[1], 1.2f);
  EXPECT_TRUE(std::isinf(logits[2]) && logits[2] < 0.0f);
  EXPECT_FLOAT_EQ(logits[3], 0.5f);
  EXPECT_FLOAT_EQ(logits[4], 0.0f);
}

TEST(SuppressTokensConstraintTest, ComputeNextValidatesTokens) {
  constexpr int kVocabSize = 5;
  SuppressTokensConstraint constraint(kVocabSize, absl::flat_hash_set<int>{1});

  auto state = constraint.Start();
  ASSERT_OK_AND_ASSIGN(auto next_state, constraint.ComputeNext(*state, 2));
  EXPECT_NE(next_state, nullptr);

  EXPECT_THAT(constraint.ComputeNext(*state, -1),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(constraint.ComputeNext(*state, 5),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

INSTANTIATE_TEST_SUITE_P(SuppressTokensConstraintParamTests,
                         SuppressTokensConstraintParamTest,
                         ::testing::Values(LogitsVariant::kFloat,
                                           LogitsVariant::kHalf));

}  // namespace
}  // namespace litert::lm
