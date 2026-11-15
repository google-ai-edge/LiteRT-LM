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

#include "runtime/components/constrained_decoding/no_repeat_ngram_constraint.h"

#include <cmath>
#include <cstddef>
#include <memory>
#include <string>
#include <thread>  // NOLINT
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "runtime/components/constrained_decoding/logit_mask.h"
#include "runtime/components/constrained_decoding/no_repeat_ngram_config.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep
#include "tflite/types/half.h"  // from @litert

namespace litert::lm {
namespace {

using ::testing::TestParamInfo;
using ::testing::TestWithParam;
using ::testing::ValuesIn;
using ::testing::status::StatusIs;

enum class LogitsVariant {
  kFloat,
  kHalf,
};

struct TestParam {
  int ngram_size;
  int window_size;
  LogitsVariant logits_variant;
};

std::vector<TestParam> GetTestParams() {
  std::vector<TestParam> params;
  for (int ngram_size : {1, 2, 3}) {
    for (int window_size : {0, ngram_size, ngram_size + 1, ngram_size + 2}) {
      for (LogitsVariant variant :
           {LogitsVariant::kFloat, LogitsVariant::kHalf}) {
        params.push_back({ngram_size, window_size, variant});
      }
    }
  }
  return params;
}

class NoRepeatNgramConstraintParamTest : public TestWithParam<TestParam> {
 protected:
  template <typename Func>
  void RunWithParam(Func&& func) {
    switch (GetParam().logits_variant) {
      case LogitsVariant::kFloat:
        func(float{});
        break;
      case LogitsVariant::kHalf:
        func(tflite::half{});
        break;
    }
  }
};

TEST_P(NoRepeatNgramConstraintParamTest, BansSeenNgramsAccordingToWindowSize) {
  const int ngram_size = GetParam().ngram_size;
  const int window_size = GetParam().window_size;
  NoRepeatNgramConfig config(ngram_size, window_size);
  constexpr int kVocabSize = 10;
  NoRepeatNgramConstraint constraint(kVocabSize, config);

  EXPECT_EQ(constraint.GetVocabularySize(), kVocabSize);

  auto state = constraint.Start();
  ASSERT_NE(state, nullptr);
  EXPECT_FALSE(constraint.IsEnded(*state));

  std::vector<int> prefix;
  for (int i = 0; i < ngram_size - 1; ++i) {
    prefix.push_back(i);
  }
  const int target_token = ngram_size - 1;
  const int filler_token = ngram_size;

  for (int token_id : prefix) {
    ASSERT_OK_AND_ASSIGN(state, constraint.ComputeNext(*state, token_id));
  }
  ASSERT_OK_AND_ASSIGN(state, constraint.ComputeNext(*state, target_token));
  ASSERT_OK_AND_ASSIGN(state, constraint.ComputeNext(*state, filler_token));
  for (int token_id : prefix) {
    ASSERT_OK_AND_ASSIGN(state, constraint.ComputeNext(*state, token_id));
  }

  ASSERT_OK_AND_ASSIGN(auto mask, constraint.ComputeMask(*state));

  // The target token_id should be banned if the window size is large enough to
  // contain the prefix pattern twice: 2 * (ngram_size - 1) + 2.
  const bool should_ban =
      window_size == 0 || (ngram_size == 1 && window_size >= ngram_size + 1) ||
      (ngram_size > 1 && window_size >= 2 * (ngram_size - 1) +
                                            /*target_token*/ 1 +
                                            /*filler_token*/ 1);

  RunWithParam([&](auto dummy_type) {
    using T = decltype(dummy_type);
    std::vector<T> logits(kVocabSize);
    for (int i = 0; i < kVocabSize; ++i) {
      logits[i] = static_cast<T>(1.0f + 0.5f * i);
    }

    EXPECT_OK(mask->Apply(absl::MakeSpan(logits)));

    if (should_ban) {
      if constexpr (std::is_same_v<T, float>) {
        EXPECT_TRUE(std::isinf(logits[target_token]) &&
                    logits[target_token] < 0.0f);
      } else {
        EXPECT_EQ(logits[target_token], tflite::half::min());
      }
    } else {
      EXPECT_FLOAT_EQ(static_cast<float>(logits[target_token]),
                      1.0f + 0.5f * target_token);
    }

    for (int i = 0; i < kVocabSize; ++i) {
      if (i == target_token) continue;
      if (ngram_size == 1 && i == filler_token) {
        if constexpr (std::is_same_v<T, float>) {
          EXPECT_TRUE(std::isinf(logits[i]) && logits[i] < 0.0f);
        } else {
          EXPECT_EQ(logits[i], tflite::half::min());
        }
        continue;
      }
      EXPECT_FLOAT_EQ(static_cast<float>(logits[i]), 1.0f + 0.5f * i);
    }
  });

  EXPECT_EQ(constraint.prefix_hash_collision_count(), 0);
}

TEST_P(NoRepeatNgramConstraintParamTest, HandlesNotEnoughTokensGeneratedYet) {
  const int ngram_size = GetParam().ngram_size;
  const int window_size = GetParam().window_size;
  NoRepeatNgramConfig config(ngram_size, window_size);
  constexpr int kVocabSize = 10;
  NoRepeatNgramConstraint constraint(kVocabSize, config);
  const int fixed_token_id = 0;

  auto state = constraint.Start();

  for (int i = 0; i < ngram_size; ++i) {
    ASSERT_OK_AND_ASSIGN(state, constraint.ComputeNext(*state, fixed_token_id));
    ASSERT_OK_AND_ASSIGN(auto mask, constraint.ComputeMask(*state));

    RunWithParam([&](auto dummy_type) {
      using T = decltype(dummy_type);
      std::vector<T> logits(kVocabSize, static_cast<T>(2.0f));

      EXPECT_OK(mask->Apply(absl::MakeSpan(logits)));

      for (int token_id = 0; token_id < kVocabSize; ++token_id) {
        if (i >= ngram_size - 1 && token_id == fixed_token_id) {
          if constexpr (std::is_same_v<T, float>) {
            EXPECT_TRUE(std::isinf(logits[token_id]) &&
                        logits[token_id] < 0.0f);
          } else {
            EXPECT_EQ(logits[token_id], tflite::half::min());
          }
        } else {
          EXPECT_FLOAT_EQ(static_cast<float>(logits[token_id]), 2.0f);
        }
      }
    });
  }

  EXPECT_EQ(constraint.prefix_hash_collision_count(), 0);
}

INSTANTIATE_TEST_SUITE_P(
    NoRepeatNgramConstraintParamTests, NoRepeatNgramConstraintParamTest,
    ValuesIn(GetTestParams()), [](const TestParamInfo<TestParam>& info) {
      std::string variant_str =
          info.param.logits_variant == LogitsVariant::kFloat ? "Float" : "Half";
      return "Ngram" + std::to_string(info.param.ngram_size) + "_Window" +
             std::to_string(info.param.window_size) + "_" + variant_str;
    });

TEST(NoRepeatNgramConstraintTest, HandlesDisabledConfig) {
  NoRepeatNgramConfig config(/*no_repeat_ngram_size=*/0, /*window_size=*/0);
  constexpr int kVocabSize = 3;
  NoRepeatNgramConstraint constraint(kVocabSize, config);

  auto state = constraint.Start();
  for (int token_id : {1, 2, 1}) {
    ASSERT_OK_AND_ASSIGN(state, constraint.ComputeNext(*state, token_id));
  }

  ASSERT_OK_AND_ASSIGN(auto mask, constraint.ComputeMask(*state));

  std::vector<float> logits = {2.0f, 1.2f, -1.2f};
  EXPECT_OK(mask->Apply(absl::MakeSpan(logits)));

  EXPECT_FLOAT_EQ(logits[0], 2.0f);
  EXPECT_FLOAT_EQ(logits[1], 1.2f);
  EXPECT_FLOAT_EQ(logits[2], -1.2f);
}

TEST(NoRepeatNgramConstraintTest, HandlesMultiplePrefixOccurrencesInWindow) {
  NoRepeatNgramConfig config(/*no_repeat_ngram_size=*/3, /*window_size=*/6);
  constexpr int kVocabSize = 10;
  NoRepeatNgramConstraint constraint(kVocabSize, config);

  auto state = constraint.Start();
  for (int token_id : {1, 2, 1, 2, 3, 1, 2}) {
    ASSERT_OK_AND_ASSIGN(state, constraint.ComputeNext(*state, token_id));
  }

  ASSERT_OK_AND_ASSIGN(auto mask, constraint.ComputeMask(*state));

  std::vector<float> logits(kVocabSize, 1.0f);
  EXPECT_OK(mask->Apply(absl::MakeSpan(logits)));

  // (1) is no longer banned because the first (1, 2) has left the window.
  EXPECT_FLOAT_EQ(logits[1], 1.0f);
  // (3) MUST be banned because it was seen after the second (1, 2), which is
  // still in the window.
  EXPECT_TRUE(std::isinf(logits[3]) && logits[3] < 0.0f);
}

TEST(NoRepeatNgramConstraintTest, HandlesUnigramMultipleOccurrencesInWindow) {
  NoRepeatNgramConfig config(/*no_repeat_ngram_size=*/1, /*window_size=*/3);
  constexpr int kVocabSize = 5;
  NoRepeatNgramConstraint constraint(kVocabSize, config);

  auto state = constraint.Start();
  for (int token_id : {0, 1, 1}) {
    ASSERT_OK_AND_ASSIGN(state, constraint.ComputeNext(*state, token_id));
  }

  {
    ASSERT_OK_AND_ASSIGN(auto mask, constraint.ComputeMask(*state));
    std::vector<float> logits(5, 1.0f);
    EXPECT_OK(mask->Apply(absl::MakeSpan(logits)));
    EXPECT_TRUE(std::isinf(logits[0]) && logits[0] < 0.0f);
    EXPECT_TRUE(std::isinf(logits[1]) && logits[1] < 0.0f);
    EXPECT_FLOAT_EQ(logits[2], 1.0f);
  }

  // Update token 2 -> window is [1, 1, 2]
  ASSERT_OK_AND_ASSIGN(state, constraint.ComputeNext(*state, 2));
  {
    ASSERT_OK_AND_ASSIGN(auto mask, constraint.ComputeMask(*state));
    std::vector<float> logits(5, 1.0f);
    EXPECT_OK(mask->Apply(absl::MakeSpan(logits)));
    EXPECT_FLOAT_EQ(logits[0], 1.0f);
    EXPECT_TRUE(std::isinf(logits[1]) && logits[1] < 0.0f);
    EXPECT_TRUE(std::isinf(logits[2]) && logits[2] < 0.0f);
  }

  // Update token 0 -> window is [1, 2, 0]
  ASSERT_OK_AND_ASSIGN(state, constraint.ComputeNext(*state, 0));
  {
    ASSERT_OK_AND_ASSIGN(auto mask, constraint.ComputeMask(*state));
    std::vector<float> logits(5, 1.0f);
    EXPECT_OK(mask->Apply(absl::MakeSpan(logits)));
    EXPECT_TRUE(std::isinf(logits[0]) && logits[0] < 0.0f);
    EXPECT_TRUE(std::isinf(logits[1]) && logits[1] < 0.0f);
    EXPECT_TRUE(std::isinf(logits[2]) && logits[2] < 0.0f);
  }

  // Update token 0 -> window is [2, 0, 0]
  ASSERT_OK_AND_ASSIGN(state, constraint.ComputeNext(*state, 0));
  {
    ASSERT_OK_AND_ASSIGN(auto mask, constraint.ComputeMask(*state));
    std::vector<float> logits(5, 1.0f);
    EXPECT_OK(mask->Apply(absl::MakeSpan(logits)));
    EXPECT_TRUE(std::isinf(logits[0]) && logits[0] < 0.0f);
    EXPECT_FLOAT_EQ(logits[1], 1.0f);
    EXPECT_TRUE(std::isinf(logits[2]) && logits[2] < 0.0f);
  }
}

TEST(NoRepeatNgramConstraintTest, ComputeNextValidatesTokens) {
  NoRepeatNgramConfig config(/*no_repeat_ngram_size=*/2, /*window_size=*/0);
  constexpr int kVocabSize = 5;
  NoRepeatNgramConstraint constraint(kVocabSize, config);

  auto state = constraint.Start();
  EXPECT_THAT(constraint.ComputeNext(*state, -1),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(constraint.ComputeNext(*state, 5),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(NoRepeatNgramConstraintTest, HandlesHashCollisionWithPrefixesMatch) {
  NoRepeatNgramConfig config(/*no_repeat_ngram_size=*/3, /*window_size=*/0);
  constexpr int kVocabSize = 10;
  NoRepeatNgramConstraint constraint(kVocabSize, config);

  auto state = constraint.Start();
  // Sequence: [0, 1] followed by 3
  ASSERT_OK_AND_ASSIGN(state, constraint.ComputeNext(*state, 0));
  ASSERT_OK_AND_ASSIGN(state, constraint.ComputeNext(*state, 1));
  ASSERT_OK_AND_ASSIGN(state, constraint.ComputeNext(*state, 3));

  // Now next prefix: [1, 1]
  ASSERT_OK_AND_ASSIGN(state, constraint.ComputeNext(*state, 1));

  // Force hash collision on state
  auto& s = static_cast<NoRepeatNgramConstraint::NoRepeatNgramState&>(*state);
  // Look up hash of [0, 1]
  NoRepeatNgramConstraint::NoRepeatNgramState tmp_s;
  tmp_s.token_history = {0, 1};
  tmp_s.token_history_index = 1;
  size_t hash_01 = constraint.ComputePrefixHash(tmp_s);

  s.latest_prefix_hash = hash_01;

  // Compute mask: should NOT ban token 3 because [1, 1] != [0, 1]
  ASSERT_OK_AND_ASSIGN(auto mask, constraint.ComputeMask(*state));
  std::vector<float> logits(10, 1.0f);
  EXPECT_OK(mask->Apply(absl::MakeSpan(logits)));

  EXPECT_EQ(constraint.prefix_hash_collision_count(), 1);
  EXPECT_FLOAT_EQ(logits[3], 1.0f);
}

TEST(NoRepeatNgramConstraintTest, ConcurrentCallsAreThreadSafe) {
  NoRepeatNgramConfig config(/*no_repeat_ngram_size=*/3, /*window_size=*/6);
  constexpr int kVocabSize = 10;
  const NoRepeatNgramConstraint constraint(kVocabSize, config);

  constexpr int kNumThreads = 8;
  constexpr int kIterations = 100;
  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);

  for (int t = 0; t < kNumThreads; ++t) {
    threads.emplace_back([&constraint]() {
      for (int iter = 0; iter < kIterations; ++iter) {
        auto state = constraint.Start();
        for (int token_id : {1, 2, 1, 2, 3, 1, 2}) {
          ASSERT_OK_AND_ASSIGN(state, constraint.ComputeNext(*state, token_id));
        }

        ASSERT_OK_AND_ASSIGN(auto mask, constraint.ComputeMask(*state));
        std::vector<float> logits(kVocabSize, 1.0f);
        EXPECT_OK(mask->Apply(absl::MakeSpan(logits)));

        EXPECT_FLOAT_EQ(logits[1], 1.0f);
        EXPECT_TRUE(std::isinf(logits[3]) && logits[3] < 0.0f);
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }
}

}  // namespace
}  // namespace litert::lm
