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

#include "runtime/components/constrained_decoding/logit_mask.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"  // from @com_google_absl
#include "runtime/util/test_utils.h"  // IWYU pragma: keep
#include "tflite/types/half.h"  // from @litert

namespace litert::lm {
namespace {

using ::testing::ElementsAre;

enum class LogitsVariant {
  kFloat,
  kHalf,
};

class LogitMaskParamTest : public ::testing::TestWithParam<LogitsVariant> {
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

// =============================================================================
// BitmapLogitMask Tests
// =============================================================================

TEST(BitmapLogitMaskTest, CreateAllAllowed) {
  constexpr int kVocabSize = 100;
  auto mask = BitmapLogitMask::CreateAllAllowed(kVocabSize);
  EXPECT_EQ(mask->vocab_size(), kVocabSize);
  EXPECT_EQ(mask->GetType(), MaskType::kBitmap);

  for (int i = 0; i < kVocabSize; ++i) {
    EXPECT_TRUE(mask->IsAllowed(i));
  }
  EXPECT_FALSE(mask->IsAllowed(-1));
  EXPECT_FALSE(mask->IsAllowed(kVocabSize));
  EXPECT_FALSE(mask->IsAllowed(kVocabSize + 10));
}

TEST(BitmapLogitMaskTest, CreateAllDisallowed) {
  constexpr int kVocabSize = 70;
  auto mask = BitmapLogitMask::CreateAllDisallowed(kVocabSize);
  EXPECT_EQ(mask->vocab_size(), kVocabSize);

  for (int i = 0; i < kVocabSize; ++i) {
    EXPECT_FALSE(mask->IsAllowed(i));
  }
}

TEST(BitmapLogitMaskTest, CreateSingleAllowedToken) {
  constexpr int kVocabSize = 80;
  auto mask = BitmapLogitMask::CreateSingleAllowedToken(kVocabSize, 42);
  EXPECT_EQ(mask->vocab_size(), kVocabSize);

  for (int i = 0; i < kVocabSize; ++i) {
    if (i == 42) {
      EXPECT_TRUE(mask->IsAllowed(i));
    } else {
      EXPECT_FALSE(mask->IsAllowed(i));
    }
  }
}

TEST(BitmapLogitMaskTest, CreateFromDisallowedTokens) {
  constexpr int kVocabSize = 10;
  std::vector<int> disallowed = {1, 3, 7};
  auto mask = BitmapLogitMask::CreateFromDisallowedTokens(
      kVocabSize, absl::MakeConstSpan(disallowed));

  EXPECT_TRUE(mask->IsAllowed(0));
  EXPECT_FALSE(mask->IsAllowed(1));
  EXPECT_TRUE(mask->IsAllowed(2));
  EXPECT_FALSE(mask->IsAllowed(3));
  EXPECT_TRUE(mask->IsAllowed(4));
  EXPECT_TRUE(mask->IsAllowed(5));
  EXPECT_TRUE(mask->IsAllowed(6));
  EXPECT_FALSE(mask->IsAllowed(7));
  EXPECT_TRUE(mask->IsAllowed(8));
  EXPECT_TRUE(mask->IsAllowed(9));
}

TEST(BitmapLogitMaskTest, CreateFromAllowedTokens) {
  constexpr int kVocabSize = 10;
  std::vector<int> allowed = {2, 5, 8};
  auto mask = BitmapLogitMask::CreateFromAllowedTokens(
      kVocabSize, absl::MakeConstSpan(allowed));

  for (int i = 0; i < kVocabSize; ++i) {
    if (i == 2 || i == 5 || i == 8) {
      EXPECT_TRUE(mask->IsAllowed(i));
    } else {
      EXPECT_FALSE(mask->IsAllowed(i));
    }
  }
}

TEST(BitmapLogitMaskTest, ConstructFromUint32Words) {
  constexpr int kVocabSize = 70;
  // 32-bit words: word 0 has bit 1 set, word 1 has bit 2 set (token 34),
  // word 2 has bit 3 set (token 67).
  std::vector<uint32_t> words_u32 = {0x00000002, 0x00000004, 0x00000008};
  BitmapLogitMask mask(kVocabSize, absl::MakeConstSpan(words_u32));

  EXPECT_TRUE(mask.IsAllowed(1));
  EXPECT_TRUE(mask.IsAllowed(34));
  EXPECT_TRUE(mask.IsAllowed(67));
  EXPECT_FALSE(mask.IsAllowed(0));
  EXPECT_FALSE(mask.IsAllowed(2));
  EXPECT_FALSE(mask.IsAllowed(33));
  EXPECT_FALSE(mask.IsAllowed(35));
}

TEST(BitmapLogitMaskTest, ConstructFromBoolVector) {
  constexpr int kVocabSize = 5;
  std::vector<bool> allowed_mask = {true, false, true, false, true};
  BitmapLogitMask mask(kVocabSize, allowed_mask);

  EXPECT_TRUE(mask.IsAllowed(0));
  EXPECT_FALSE(mask.IsAllowed(1));
  EXPECT_TRUE(mask.IsAllowed(2));
  EXPECT_FALSE(mask.IsAllowed(3));
  EXPECT_TRUE(mask.IsAllowed(4));
}

TEST(BitmapLogitMaskTest, ConstructFromBoolSpan) {
  constexpr int kVocabSize = 4;
  const bool allowed_arr[] = {false, true, true, false};
  BitmapLogitMask mask(kVocabSize, absl::MakeConstSpan(allowed_arr));

  EXPECT_FALSE(mask.IsAllowed(0));
  EXPECT_TRUE(mask.IsAllowed(1));
  EXPECT_TRUE(mask.IsAllowed(2));
  EXPECT_FALSE(mask.IsAllowed(3));
}

TEST_P(LogitMaskParamTest, BitmapApplyMasksDisallowedAndPadding) {
  constexpr int kVocabSize = 5;
  std::vector<int> allowed = {1, 3};
  auto mask = BitmapLogitMask::CreateFromAllowedTokens(
      kVocabSize, absl::MakeConstSpan(allowed));

  RunWithParam([&](auto dummy_type) {
    using T = decltype(dummy_type);
    // Logits span of size 7 (5 vocab + 2 padding)
    std::vector<T> logits = {
        static_cast<T>(1.0f), static_cast<T>(2.0f), static_cast<T>(3.0f),
        static_cast<T>(4.0f), static_cast<T>(5.0f), static_cast<T>(6.0f),
        static_cast<T>(7.0f),
    };

    EXPECT_OK(mask->Apply(absl::MakeSpan(logits)));

    // Allowed tokens (1 and 3) remain intact
    EXPECT_FLOAT_EQ(static_cast<float>(logits[1]), 2.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(logits[3]), 4.0f);

    // Disallowed tokens (0, 2, 4) and padding (5, 6) must be -inf / min
    if constexpr (std::is_same_v<T, float>) {
      EXPECT_TRUE(std::isinf(logits[0]) && logits[0] < 0.0f);
      EXPECT_TRUE(std::isinf(logits[2]) && logits[2] < 0.0f);
      EXPECT_TRUE(std::isinf(logits[4]) && logits[4] < 0.0f);
      EXPECT_TRUE(std::isinf(logits[5]) && logits[5] < 0.0f);
      EXPECT_TRUE(std::isinf(logits[6]) && logits[6] < 0.0f);
    } else {
      EXPECT_EQ(logits[0], tflite::half::min());
      EXPECT_EQ(logits[2], tflite::half::min());
      EXPECT_EQ(logits[4], tflite::half::min());
      EXPECT_EQ(logits[5], tflite::half::min());
      EXPECT_EQ(logits[6], tflite::half::min());
    }
  });
}

TEST_P(LogitMaskParamTest, BitmapApplyMultiWordAllAllowedSkipped) {
  constexpr int kVocabSize = 130;
  auto mask = BitmapLogitMask::CreateAllAllowed(kVocabSize);

  RunWithParam([&](auto dummy_type) {
    using T = decltype(dummy_type);
    std::vector<T> logits(kVocabSize);
    for (int i = 0; i < kVocabSize; ++i) {
      logits[i] = static_cast<T>(static_cast<float>(i));
    }

    EXPECT_OK(mask->Apply(absl::MakeSpan(logits)));

    for (int i = 0; i < kVocabSize; ++i) {
      EXPECT_FLOAT_EQ(static_cast<float>(logits[i]), static_cast<float>(i));
    }
  });
}

// =============================================================================
// SparseLogitMask Tests
// =============================================================================

TEST(SparseLogitMaskTest, MetadataAndTypes) {
  std::vector<SparseLogitMask::Entry> entries = {
      {.token_id = 2, .weight = 0.5f, .bias = 1.0f},
      {.token_id = 5, .weight = 1.2f, .bias = -0.5f},
  };
  SparseLogitMask mask(entries);
  EXPECT_EQ(mask.GetType(), MaskType::kSparse);
  EXPECT_EQ(mask.entries().size(), 2);
  EXPECT_EQ(mask.entries()[0].token_id, 2);
  EXPECT_FLOAT_EQ(mask.entries()[0].weight, 0.5f);
  EXPECT_FLOAT_EQ(mask.entries()[0].bias, 1.0f);
}

TEST(SparseLogitMaskTest, CreateBiasesAndWeights) {
  std::vector<std::pair<int, float>> biases = {{1, 2.5f}, {4, -1.0f}};
  auto bias_mask = SparseLogitMask::CreateBiases(absl::MakeConstSpan(biases));
  EXPECT_EQ(bias_mask->entries().size(), 2);
  EXPECT_FLOAT_EQ(bias_mask->entries()[0].bias, 2.5f);
  EXPECT_FLOAT_EQ(bias_mask->entries()[0].weight, 1.0f);

  std::vector<std::pair<int, float>> weights = {{0, 0.8f}, {3, 1.5f}};
  auto weight_mask =
      SparseLogitMask::CreateWeights(absl::MakeConstSpan(weights));
  EXPECT_EQ(weight_mask->entries().size(), 2);
  EXPECT_FLOAT_EQ(weight_mask->entries()[0].weight, 0.8f);
  EXPECT_FLOAT_EQ(weight_mask->entries()[0].bias, 0.0f);
}

TEST_P(LogitMaskParamTest, SparseApplyWeightsAndBiases) {
  std::vector<SparseLogitMask::Entry> entries = {
      {.token_id = 1, .weight = 2.0f, .bias = 0.5f},   // 2.0 * 2.0 + 0.5 = 4.5
      {.token_id = 3, .weight = 0.5f, .bias = -1.0f},  // 4.0 * 0.5 - 1.0 = 1.0
      {.token_id = 10, .weight = 1.0f, .bias = 1.0f},  // Out of bounds, ignored
  };
  SparseLogitMask mask(entries);

  RunWithParam([&](auto dummy_type) {
    using T = decltype(dummy_type);
    std::vector<T> logits = {
        static_cast<T>(1.0f), static_cast<T>(2.0f), static_cast<T>(3.0f),
        static_cast<T>(4.0f), static_cast<T>(5.0f),
    };

    EXPECT_OK(mask.Apply(absl::MakeSpan(logits)));

    EXPECT_FLOAT_EQ(static_cast<float>(logits[0]), 1.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(logits[1]), 4.5f);
    EXPECT_FLOAT_EQ(static_cast<float>(logits[2]), 3.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(logits[3]), 1.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(logits[4]), 5.0f);
  });
}

TEST_P(LogitMaskParamTest, SparseApplySignDependentWeight) {
  // repetition_penalty = 2.0 -> weight = 0.5
  std::vector<SparseLogitMask::Entry> entries = {
      {.token_id = 0,
       .weight = 0.5f,
       .bias = 0.0f,
       .sign_dependent_weight = true},  // Positive: 4.0 * 0.5 = 2.0
      {.token_id = 1,
       .weight = 0.5f,
       .bias = 0.0f,
       .sign_dependent_weight = true},  // Negative: -4.0 / 0.5 = -8.0
  };
  SparseLogitMask mask(entries);

  RunWithParam([&](auto dummy_type) {
    using T = decltype(dummy_type);
    std::vector<T> logits = {
        static_cast<T>(4.0f),
        static_cast<T>(-4.0f),
    };

    EXPECT_OK(mask.Apply(absl::MakeSpan(logits)));

    EXPECT_FLOAT_EQ(static_cast<float>(logits[0]), 2.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(logits[1]), -8.0f);
  });
}

TEST_P(LogitMaskParamTest, SparseApplyPreservesDisallowedTokens) {
  std::vector<SparseLogitMask::Entry> entries = {
      {.token_id = 0, .weight = 2.0f, .bias = 10.0f},
      {.token_id = 1, .weight = 0.5f, .bias = -1.0f},
  };
  SparseLogitMask mask(entries);

  RunWithParam([&](auto dummy_type) {
    using T = decltype(dummy_type);
    T min_val;
    if constexpr (std::is_same_v<T, float>) {
      min_val = -std::numeric_limits<float>::infinity();
    } else {
      min_val = tflite::half::min();
    }

    std::vector<T> logits = {
        min_val,               // Disallowed token
        static_cast<T>(4.0f),  // Allowed token
    };

    EXPECT_OK(mask.Apply(absl::MakeSpan(logits)));

    // Token 0 must remain -inf / min and NOT be resurrected or NaN
    if constexpr (std::is_same_v<T, float>) {
      EXPECT_TRUE(std::isinf(logits[0]) && logits[0] < 0.0f);
    } else {
      EXPECT_EQ(logits[0], tflite::half::min());
    }

    // Token 1 is modified: 4.0 * 0.5 - 1.0 = 1.0
    EXPECT_FLOAT_EQ(static_cast<float>(logits[1]), 1.0f);
  });
}

// =============================================================================
// CompositeLogitMask Tests
// =============================================================================

TEST(CompositeLogitMaskTest, EmptyCompositeMaskIsNoOp) {
  CompositeLogitMask mask;
  EXPECT_TRUE(mask.empty());
  EXPECT_EQ(mask.size(), 0);
  EXPECT_EQ(mask.GetType(), MaskType::kComposite);

  std::vector<float> logits = {1.0f, 2.0f, 3.0f};
  EXPECT_OK(mask.Apply(absl::MakeSpan(logits)));
  EXPECT_THAT(logits, ElementsAre(1.0f, 2.0f, 3.0f));
}

TEST_P(LogitMaskParamTest, CompositeFusesBitmapMasksWithBitwiseAnd) {
  constexpr int kVocabSize = 6;
  // Mask 1 allows {0, 1, 2, 3}
  std::vector<int> allowed1 = {0, 1, 2, 3};
  auto bm1 = BitmapLogitMask::CreateFromAllowedTokens(
      kVocabSize, absl::MakeConstSpan(allowed1));

  // Mask 2 allows {2, 3, 4, 5}
  std::vector<int> allowed2 = {2, 3, 4, 5};
  auto bm2 = BitmapLogitMask::CreateFromAllowedTokens(
      kVocabSize, absl::MakeConstSpan(allowed2));

  CompositeLogitMask composite;
  composite.AddMask(std::move(bm1));
  composite.AddMask(std::move(bm2));
  EXPECT_EQ(composite.size(), 2);

  RunWithParam([&](auto dummy_type) {
    using T = decltype(dummy_type);
    std::vector<T> logits = {
        static_cast<T>(10.0f), static_cast<T>(20.0f), static_cast<T>(30.0f),
        static_cast<T>(40.0f), static_cast<T>(50.0f), static_cast<T>(60.0f),
    };

    EXPECT_OK(composite.Apply(absl::MakeSpan(logits)));

    // Only tokens 2 and 3 are allowed in BOTH masks
    EXPECT_FLOAT_EQ(static_cast<float>(logits[2]), 30.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(logits[3]), 40.0f);

    // Tokens 0, 1, 4, 5 are disallowed
    if constexpr (std::is_same_v<T, float>) {
      EXPECT_TRUE(std::isinf(logits[0]) && logits[0] < 0.0f);
      EXPECT_TRUE(std::isinf(logits[1]) && logits[1] < 0.0f);
      EXPECT_TRUE(std::isinf(logits[4]) && logits[4] < 0.0f);
      EXPECT_TRUE(std::isinf(logits[5]) && logits[5] < 0.0f);
    } else {
      EXPECT_EQ(logits[0], tflite::half::min());
      EXPECT_EQ(logits[1], tflite::half::min());
      EXPECT_EQ(logits[4], tflite::half::min());
      EXPECT_EQ(logits[5], tflite::half::min());
    }
  });
}

TEST_P(LogitMaskParamTest, CompositeChainsBitmapAndSparseMasks) {
  constexpr int kVocabSize = 5;
  // Hard constraint: allows tokens {1, 2, 3}
  std::vector<int> allowed = {1, 2, 3};
  auto hard_mask = BitmapLogitMask::CreateFromAllowedTokens(
      kVocabSize, absl::MakeConstSpan(allowed));

  // Soft constraint: applies weights and biases to tokens {0, 1, 2}
  std::vector<SparseLogitMask::Entry> entries = {
      {.token_id = 0, .weight = 2.0f, .bias = 100.0f},  // Disallowed by hard
      {.token_id = 1, .weight = 0.5f, .bias = 2.0f},  // 10.0 * 0.5 + 2.0 = 7.0
      {.token_id = 2,
       .weight = 1.0f,
       .bias = -5.0f},  // 20.0 * 1.0 - 5.0 = 15.0
  };
  auto soft_mask = std::make_unique<SparseLogitMask>(entries);

  CompositeLogitMask composite;
  composite.AddMask(std::move(hard_mask));
  composite.AddMask(std::move(soft_mask));

  RunWithParam([&](auto dummy_type) {
    using T = decltype(dummy_type);
    std::vector<T> logits = {
        static_cast<T>(5.0f),   // token 0 (disallowed)
        static_cast<T>(10.0f),  // token 1 (allowed, adjusted)
        static_cast<T>(20.0f),  // token 2 (allowed, adjusted)
        static_cast<T>(30.0f),  // token 3 (allowed, untouched)
        static_cast<T>(40.0f),  // token 4 (disallowed)
    };

    EXPECT_OK(composite.Apply(absl::MakeSpan(logits)));

    // Token 0: Disallowed, must NOT be altered by soft bias
    if constexpr (std::is_same_v<T, float>) {
      EXPECT_TRUE(std::isinf(logits[0]) && logits[0] < 0.0f);
      EXPECT_TRUE(std::isinf(logits[4]) && logits[4] < 0.0f);
    } else {
      EXPECT_EQ(logits[0], tflite::half::min());
      EXPECT_EQ(logits[4], tflite::half::min());
    }

    // Token 1: 10.0 * 0.5 + 2.0 = 7.0
    EXPECT_FLOAT_EQ(static_cast<float>(logits[1]), 7.0f);

    // Token 2: 20.0 * 1.0 - 5.0 = 15.0
    EXPECT_FLOAT_EQ(static_cast<float>(logits[2]), 15.0f);

    // Token 3: Allowed and untouched = 30.0
    EXPECT_FLOAT_EQ(static_cast<float>(logits[3]), 30.0f);
  });
}

TEST_P(LogitMaskParamTest, CompositeChainsMultipleSparseMasks) {
  // Soft mask 1: Add bias +2.0 to token 1
  std::vector<std::pair<int, float>> biases = {{1, 2.0f}};
  auto mask1 = SparseLogitMask::CreateBiases(absl::MakeConstSpan(biases));

  // Soft mask 2: Multiply token 1 by 0.5
  std::vector<std::pair<int, float>> weights = {{1, 0.5f}};
  auto mask2 = SparseLogitMask::CreateWeights(absl::MakeConstSpan(weights));

  CompositeLogitMask composite;
  composite.AddMask(std::move(mask1));
  composite.AddMask(std::move(mask2));

  RunWithParam([&](auto dummy_type) {
    using T = decltype(dummy_type);
    std::vector<T> logits = {static_cast<T>(10.0f), static_cast<T>(4.0f)};

    EXPECT_OK(composite.Apply(absl::MakeSpan(logits)));

    EXPECT_FLOAT_EQ(static_cast<float>(logits[0]), 10.0f);
    // (4.0 + 2.0) * 0.5 = 3.0
    EXPECT_FLOAT_EQ(static_cast<float>(logits[1]), 3.0f);
  });
}

TEST_P(LogitMaskParamTest, NestedCompositeMaskWorks) {
  constexpr int kVocabSize = 4;
  auto bm = BitmapLogitMask::CreateSingleAllowedToken(kVocabSize, 2);

  auto inner = std::make_unique<CompositeLogitMask>();
  inner->AddMask(std::move(bm));

  CompositeLogitMask outer;
  outer.AddMask(std::move(inner));

  RunWithParam([&](auto dummy_type) {
    using T = decltype(dummy_type);
    std::vector<T> logits = {static_cast<T>(1.0f), static_cast<T>(2.0f),
                             static_cast<T>(3.0f), static_cast<T>(4.0f)};

    EXPECT_OK(outer.Apply(absl::MakeSpan(logits)));

    EXPECT_FLOAT_EQ(static_cast<float>(logits[2]), 3.0f);

    if constexpr (std::is_same_v<T, float>) {
      EXPECT_TRUE(std::isinf(logits[0]) && logits[0] < 0.0f);
      EXPECT_TRUE(std::isinf(logits[1]) && logits[1] < 0.0f);
      EXPECT_TRUE(std::isinf(logits[3]) && logits[3] < 0.0f);
    } else {
      EXPECT_EQ(logits[0], tflite::half::min());
      EXPECT_EQ(logits[1], tflite::half::min());
      EXPECT_EQ(logits[3], tflite::half::min());
    }
  });
}

INSTANTIATE_TEST_SUITE_P(LogitMaskParamTests, LogitMaskParamTest,
                         ::testing::Values(LogitsVariant::kFloat,
                                           LogitsVariant::kHalf));

}  // namespace
}  // namespace litert::lm
