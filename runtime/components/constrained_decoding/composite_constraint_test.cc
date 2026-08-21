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

#include "runtime/components/constrained_decoding/composite_constraint.h"

#include <cmath>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "runtime/components/constrained_decoding/bitmap.h"
#include "runtime/components/constrained_decoding/constraint.h"
#include "runtime/components/constrained_decoding/constraint_provider.h"
#include "runtime/components/constrained_decoding/fake_constraint.h"
#include "runtime/components/constrained_decoding/llg_constraint_config.h"
#include "runtime/components/constrained_decoding/llg_constraint_provider.h"
#include "runtime/components/constrained_decoding/logit_mask.h"
#include "runtime/components/constrained_decoding/no_repeat_ngram_config.h"
#include "runtime/components/constrained_decoding/no_repeat_ngram_constraint.h"
#include "runtime/components/constrained_decoding/repetition_penalty_config.h"
#include "runtime/components/constrained_decoding/repetition_penalty_constraint.h"
#include "runtime/components/constrained_decoding/suppress_tokens_config.h"
#include "runtime/components/constrained_decoding/suppress_tokens_constraint.h"
#include "runtime/components/constrained_decoding/thinking_budget_constraint.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep
#include "support/tokenizer/tokenizer.h"
#include "tflite/types/half.h"  // from @litert

namespace litert::lm {
namespace {

using ::testing::Return;
using ::testing::status::StatusIs;

using Tokenizer = ::litert::support::Tokenizer;
using TokenizerType = ::litert::support::TokenizerType;
using TokenIds = ::litert::support::TokenIds;

template <typename T>
void ExpectMaskedOut(T val) {
  if constexpr (std::is_same_v<T, float>) {
    EXPECT_TRUE(std::isinf(val) && val < 0.0f);
  } else {
    EXPECT_EQ(val, tflite::half::min());
  }
}

class MockTokenizer : public Tokenizer {
 public:
  MOCK_METHOD(TokenizerType, GetTokenizerType, (), (const, override));
  MOCK_METHOD(absl::StatusOr<TokenIds>, TextToTokenIds, (absl::string_view),
              (override));
  MOCK_METHOD(absl::StatusOr<int>, TokenToId, (absl::string_view), (override));
  MOCK_METHOD(absl::StatusOr<std::string>, TokenIdsToText,
              (absl::Span<const int>, bool), (override));
  MOCK_METHOD(std::vector<std::string>, GetTokens, (), (const, override));
  MOCK_METHOD(int, GetVocabSize, (), (const, override));
};

enum class LogitsVariant {
  kFloat,
  kHalf,
};

class CompositeConstraintParamTest
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

TEST(CompositeConstraintTest, EmptyCompositeConstraint) {
  constexpr int kVocabSize = 10;
  ASSERT_OK_AND_ASSIGN(auto composite, CompositeConstraint::Create(kVocabSize));

  EXPECT_EQ(composite->GetVocabularySize(), kVocabSize);
  EXPECT_TRUE(composite->empty());
  EXPECT_EQ(composite->size(), 0);

  auto state = composite->Start();
  ASSERT_NE(state, nullptr);
  EXPECT_FALSE(composite->IsEnded(*state));

  ASSERT_OK_AND_ASSIGN(auto next_state, composite->ComputeNext(*state, 3));
  ASSERT_NE(next_state, nullptr);
  EXPECT_FALSE(composite->IsEnded(*next_state));

  ASSERT_OK_AND_ASSIGN(auto mask, composite->ComputeMask(*state));
  ASSERT_NE(mask, nullptr);
  EXPECT_EQ(mask->GetType(), MaskType::kComposite);

  std::vector<float> logits(kVocabSize, 1.0f);
  EXPECT_OK(mask->Apply(absl::MakeSpan(logits)));
  for (int i = 0; i < kVocabSize; ++i) {
    EXPECT_FLOAT_EQ(logits[i], 1.0f);
  }
}

TEST(CompositeConstraintTest, AddConstraintPopulatesVocabSize) {
  ASSERT_OK_AND_ASSIGN(auto composite, CompositeConstraint::Create());
  EXPECT_EQ(composite->GetVocabularySize(), 0);

  constexpr int kVocabSize = 50;
  EXPECT_OK(composite->AddConstraint(std::make_unique<SuppressTokensConstraint>(
      kVocabSize, SuppressTokensConfig(absl::flat_hash_set<int>{1, 2}))));

  EXPECT_EQ(composite->GetVocabularySize(), kVocabSize);
  EXPECT_EQ(composite->size(), 1);
  EXPECT_FALSE(composite->empty());
}

TEST_P(CompositeConstraintParamTest, SingleConstraintDelegation) {
  constexpr int kVocabSize = 5;
  std::vector<int> token_sequence = {1, 3, 0};
  auto fake = std::make_unique<FakeConstraint>(token_sequence, kVocabSize);

  ASSERT_OK_AND_ASSIGN(auto composite, CompositeConstraint::Create(kVocabSize));
  ASSERT_OK(composite->AddConstraint(std::move(fake)));

  auto state = composite->Start();
  ASSERT_NE(state, nullptr);
  EXPECT_FALSE(composite->IsEnded(*state));

  // Step 1: initial state expects token 1
  ASSERT_OK_AND_ASSIGN(auto mask1, composite->ComputeMask(*state));
  RunWithParam([&](auto dummy_type) {
    using T = decltype(dummy_type);
    std::vector<T> logits(kVocabSize, static_cast<T>(0.0f));
    EXPECT_OK(mask1->Apply(absl::MakeSpan(logits)));
    EXPECT_FLOAT_EQ(static_cast<float>(logits[1]), 0.0f);
    ExpectMaskedOut(logits[0]);
    ExpectMaskedOut(logits[2]);
    ExpectMaskedOut(logits[3]);
  });

  // Transition to token 1
  ASSERT_OK_AND_ASSIGN(state, composite->ComputeNext(*state, 1));
  EXPECT_FALSE(composite->IsEnded(*state));

  // Step 2: next expects token 3
  ASSERT_OK_AND_ASSIGN(auto mask2, composite->ComputeMask(*state));
  RunWithParam([&](auto dummy_type) {
    using T = decltype(dummy_type);
    std::vector<T> logits(kVocabSize, static_cast<T>(2.0f));
    EXPECT_OK(mask2->Apply(absl::MakeSpan(logits)));
    EXPECT_FLOAT_EQ(static_cast<float>(logits[3]), 2.0f);
    ExpectMaskedOut(logits[1]);
  });

  // Transition to token 3
  ASSERT_OK_AND_ASSIGN(state, composite->ComputeNext(*state, 3));
  EXPECT_FALSE(composite->IsEnded(*state));

  // Transition to token 0
  ASSERT_OK_AND_ASSIGN(state, composite->ComputeNext(*state, 0));
  EXPECT_TRUE(composite->IsEnded(*state));
}

TEST_P(CompositeConstraintParamTest, MultipleHardConstraintsChaining) {
  constexpr int kVocabSize = 6;
  // Constraint 1 suppresses {1, 2}
  auto c1 = std::make_unique<SuppressTokensConstraint>(
      kVocabSize, SuppressTokensConfig(absl::flat_hash_set<int>{1, 2}));
  // Constraint 2 suppresses {2, 4}
  auto c2 = std::make_unique<SuppressTokensConstraint>(
      kVocabSize, SuppressTokensConfig(absl::flat_hash_set<int>{2, 4}));

  std::vector<std::unique_ptr<Constraint>> list;
  list.push_back(std::move(c1));
  list.push_back(std::move(c2));
  ASSERT_OK_AND_ASSIGN(auto composite,
                       CompositeConstraint::Create(std::move(list)));

  auto state = composite->Start();
  ASSERT_OK_AND_ASSIGN(auto mask, composite->ComputeMask(*state));

  RunWithParam([&](auto dummy_type) {
    using T = decltype(dummy_type);
    std::vector<T> logits(kVocabSize, static_cast<T>(5.0f));
    EXPECT_OK(mask->Apply(absl::MakeSpan(logits)));

    // Allowed: 0, 3, 5 -> untouched
    EXPECT_FLOAT_EQ(static_cast<float>(logits[0]), 5.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(logits[3]), 5.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(logits[5]), 5.0f);

    // Disallowed: 1, 2, 4 -> masked out
    ExpectMaskedOut(logits[1]);
    ExpectMaskedOut(logits[2]);
    ExpectMaskedOut(logits[4]);
  });
}

TEST_P(CompositeConstraintParamTest, HardAndSoftConstraintsChaining) {
  constexpr int kVocabSize = 4;
  // Hard constraint suppresses token 1
  auto hard = std::make_unique<SuppressTokensConstraint>(
      kVocabSize, SuppressTokensConfig(absl::flat_hash_set<int>{1}));

  // Soft constraint applies repetition penalty
  RepetitionPenaltyConfig rep_config(
      /*repetition_penalty=*/2.0f, /*presence_penalty=*/1.0f,
      /*frequency_penalty=*/0.5f, /*window_size=*/0);
  auto soft =
      std::make_unique<RepetitionPenaltyConstraint>(kVocabSize, rep_config);

  std::vector<std::unique_ptr<Constraint>> list;
  list.push_back(std::move(hard));
  list.push_back(std::move(soft));
  ASSERT_OK_AND_ASSIGN(auto composite,
                       CompositeConstraint::Create(std::move(list)));

  auto state = composite->Start();

  // Feed tokens: 0 (once), 1 (once), 2 (twice)
  ASSERT_OK_AND_ASSIGN(state, composite->ComputeNext(*state, 0));
  ASSERT_OK_AND_ASSIGN(state, composite->ComputeNext(*state, 1));
  ASSERT_OK_AND_ASSIGN(state, composite->ComputeNext(*state, 2));
  ASSERT_OK_AND_ASSIGN(state, composite->ComputeNext(*state, 2));

  ASSERT_OK_AND_ASSIGN(auto mask, composite->ComputeMask(*state));

  RunWithParam([&](auto dummy_type) {
    using T = decltype(dummy_type);
    std::vector<T> logits = {static_cast<T>(10.0f), static_cast<T>(10.0f),
                             static_cast<T>(10.0f), static_cast<T>(10.0f)};
    EXPECT_OK(mask->Apply(absl::MakeSpan(logits)));

    // Token 0: seen 1 time.
    // Penalty: (10.0 / 2.0) - (1.0 + 1 * 0.5) = 5.0 - 1.5 = 3.5f
    EXPECT_FLOAT_EQ(static_cast<float>(logits[0]), 3.5f);

    // Token 1: suppressed by hard constraint -> MUST remain masked out!
    ExpectMaskedOut(logits[1]);

    // Token 2: seen 2 times.
    // Penalty: (10.0 / 2.0) - (1.0 + 2 * 0.5) = 5.0 - 2.0 = 3.0f
    EXPECT_FLOAT_EQ(static_cast<float>(logits[2]), 3.0f);

    // Token 3: never seen, not suppressed -> 10.0f
    EXPECT_FLOAT_EQ(static_cast<float>(logits[3]), 10.0f);
  });
}

TEST_P(CompositeConstraintParamTest,
       NoRepeatNgramAndRepetitionPenaltyChaining) {
  constexpr int kVocabSize = 5;
  // No repeat 2-gram constraint
  NoRepeatNgramConfig ngram_config(/*no_repeat_ngram_size=*/2,
                                   /*window_size=*/0);
  auto ngram_constraint =
      std::make_unique<NoRepeatNgramConstraint>(kVocabSize, ngram_config);

  // Soft repetition penalty
  RepetitionPenaltyConfig rep_config(
      /*repetition_penalty=*/1.5f, /*presence_penalty=*/0.0f,
      /*frequency_penalty=*/0.0f, /*window_size=*/0);
  auto rep_constraint =
      std::make_unique<RepetitionPenaltyConstraint>(kVocabSize, rep_config);

  ASSERT_OK_AND_ASSIGN(auto composite, CompositeConstraint::Create(kVocabSize));
  ASSERT_OK(composite->AddConstraint(std::move(ngram_constraint)));
  ASSERT_OK(composite->AddConstraint(std::move(rep_constraint)));

  auto state = composite->Start();

  // Generate bigram (1, 2)
  ASSERT_OK_AND_ASSIGN(state, composite->ComputeNext(*state, 1));
  ASSERT_OK_AND_ASSIGN(state, composite->ComputeNext(*state, 2));

  // Generate token 3, then token 1 again (prefix of previous bigram)
  ASSERT_OK_AND_ASSIGN(state, composite->ComputeNext(*state, 3));
  ASSERT_OK_AND_ASSIGN(state, composite->ComputeNext(*state, 1));

  // Now next token 2 would repeat bigram (1, 2) and should be banned (-inf)
  // Tokens 1, 2, 3 have repetition penalties. Token 0 and 4 are untouched.
  ASSERT_OK_AND_ASSIGN(auto mask, composite->ComputeMask(*state));

  RunWithParam([&](auto dummy_type) {
    using T = decltype(dummy_type);
    std::vector<T> logits = {static_cast<T>(6.0f), static_cast<T>(6.0f),
                             static_cast<T>(6.0f), static_cast<T>(6.0f),
                             static_cast<T>(6.0f)};
    EXPECT_OK(mask->Apply(absl::MakeSpan(logits)));

    // Token 0 and 4: untouched (6.0)
    EXPECT_FLOAT_EQ(static_cast<float>(logits[0]), 6.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(logits[4]), 6.0f);

    // Token 2: banned by no-repeat 2-gram -> masked out
    ExpectMaskedOut(logits[2]);

    // Token 1 and 3: scaled by repetition penalty (6.0 / 1.5 = 4.0)
    EXPECT_FLOAT_EQ(static_cast<float>(logits[1]), 4.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(logits[3]), 4.0f);
  });
}

// Comprehensive test verifying chained execution of:
// JSON Schema constraint + Repetition penalty + Thinking budget constraint
TEST_F(CompositeConstraintParamTest,
       ChainedExecution_JsonSchema_RepetitionPenalty_ThinkingBudget) {
  constexpr int kVocabSize = 7;
  // Token vocabulary:
  // 0: <pad>
  // 1: <eos>
  // 2: "a"
  // 3: "b"
  // 4: "\""
  // 5: <start_thought>
  // 6: <end_thought>

  MockTokenizer tokenizer;
  EXPECT_CALL(tokenizer, GetTokens())
      .WillOnce(
          Return(std::vector<std::string>{"<pad>", "<eos>", "a", "b", "\"",
                                          "<start_thought>", "<end_thought>"}));

  EXPECT_CALL(tokenizer, TextToTokenIds(::testing::_))
      .WillRepeatedly([](absl::string_view text) {
        if (text == "a") return TokenIds{2};
        if (text == "b") return TokenIds{3};
        if (text == "\"") return TokenIds{4};
        if (text == "<start_thought>") return TokenIds{5};
        if (text == "<end_thought>") return TokenIds{6};
        if (text == "ab") return TokenIds{2, 3};
        return TokenIds{};
      });

  LlGuidanceConfig config;
  config.eos_id = 1;
  ASSERT_OK_AND_ASSIGN(auto provider,
                       LlgConstraintProvider::Create(tokenizer, config));

  // 1. JSON Schema constraint: accepts a JSON string (e.g. "a")
  ASSERT_OK_AND_ASSIGN(auto json_constraint,
                       provider->CreateConstraint(LlGuidanceConstraintArg{
                           .constraint_type = LlgConstraintType::kJsonSchema,
                           .constraint_string = R"({"type": "string"})"}));

  // 2. ThinkingBudgetConstraint: budget = 2 thinking tokens,
  //    start = {5}, end = {6}, wrapping the JSON Schema constraint
  auto thinking_constraint = std::make_unique<ThinkingBudgetConstraint>(
      json_constraint.get(), /*budget=*/2,
      /*start_token_ids=*/std::vector<int>{5},
      /*end_token_ids=*/std::vector<int>{6}, kVocabSize);

  // 3. RepetitionPenaltyConstraint: penalty factor = 2.0
  RepetitionPenaltyConfig rep_config(
      /*repetition_penalty=*/2.0f, /*presence_penalty=*/0.0f,
      /*frequency_penalty=*/0.0f, /*window_size=*/0);
  auto repetition_constraint =
      std::make_unique<RepetitionPenaltyConstraint>(kVocabSize, rep_config);

  // Combine into CompositeConstraint
  ASSERT_OK_AND_ASSIGN(auto composite, CompositeConstraint::Create(kVocabSize));
  ASSERT_OK(composite->AddConstraint(std::move(thinking_constraint)));
  ASSERT_OK(composite->AddConstraint(std::move(repetition_constraint)));

  auto state = composite->Start();
  ASSERT_NE(state, nullptr);
  EXPECT_FALSE(composite->IsEnded(*state));

  // --- Step 0: Start-of-thought matching ---
  // Mask allows all tokens for start matching.
  ASSERT_OK_AND_ASSIGN(auto mask0, composite->ComputeMask(*state));
  std::vector<float> logits0(kVocabSize, 4.0f);
  EXPECT_OK(mask0->Apply(absl::MakeSpan(logits0)));
  EXPECT_FLOAT_EQ(logits0[5], 4.0f);

  // Generate <start_thought> (5)
  ASSERT_OK_AND_ASSIGN(state, composite->ComputeNext(*state, 5));
  EXPECT_FALSE(composite->IsEnded(*state));

  // --- Step 1: Thinking phase (token 1: "a" = 2) ---
  // In thinking, model generates "a". Repetition penalty sees token 5.
  ASSERT_OK_AND_ASSIGN(auto mask1, composite->ComputeMask(*state));
  std::vector<float> logits1(kVocabSize, 4.0f);
  EXPECT_OK(mask1->Apply(absl::MakeSpan(logits1)));
  EXPECT_FLOAT_EQ(logits1[2], 4.0f);  // "a" not seen yet -> 4.0f
  EXPECT_FLOAT_EQ(logits1[5],
                  2.0f);  // <start_thought> seen -> 4.0 / 2.0 = 2.0f

  // Generate "a" (2)
  ASSERT_OK_AND_ASSIGN(state, composite->ComputeNext(*state, 2));
  EXPECT_FALSE(composite->IsEnded(*state));

  // --- Step 2: Thinking phase (token 2: hits budget of 2) ---
  ASSERT_OK_AND_ASSIGN(auto mask2, composite->ComputeMask(*state));
  std::vector<float> logits2(kVocabSize, 4.0f);
  EXPECT_OK(mask2->Apply(absl::MakeSpan(logits2)));
  EXPECT_FLOAT_EQ(logits2[3], 4.0f);  // "b" not seen yet

  // Generate another token "b" (3) during thinking.
  ASSERT_OK_AND_ASSIGN(state, composite->ComputeNext(*state, 3));
  EXPECT_FALSE(composite->IsEnded(*state));

  // --- Step 3: Budget reached! Forced end token <end_thought> (6) ---
  // Thinking budget constraint MUST force token 6 only.
  ASSERT_OK_AND_ASSIGN(auto mask3, composite->ComputeMask(*state));
  std::vector<float> logits3(kVocabSize, 4.0f);
  EXPECT_OK(mask3->Apply(absl::MakeSpan(logits3)));
  EXPECT_FLOAT_EQ(logits3[6], 4.0f);  // <end_thought> is allowed
  ExpectMaskedOut(logits3[0]);
  ExpectMaskedOut(logits3[2]);
  ExpectMaskedOut(logits3[3]);
  ExpectMaskedOut(logits3[4]);

  // Generate forced end token <end_thought> (6)
  ASSERT_OK_AND_ASSIGN(state, composite->ComputeNext(*state, 6));
  EXPECT_FALSE(composite->IsEnded(*state));

  // --- Step 4: Content phase begins! JSON Schema constraint is now active! ---
  // JSON schema for string requires opening quote `"` (4).
  ASSERT_OK_AND_ASSIGN(auto mask4, composite->ComputeMask(*state));
  std::vector<float> logits4(kVocabSize, 4.0f);
  EXPECT_OK(mask4->Apply(absl::MakeSpan(logits4)));
  EXPECT_FLOAT_EQ(logits4[4], 4.0f);  // `"` is allowed
  ExpectMaskedOut(logits4[2]);        // "a" not allowed yet
  ExpectMaskedOut(logits4[3]);

  // Generate opening quote `"` (4)
  ASSERT_OK_AND_ASSIGN(state, composite->ComputeNext(*state, 4));
  EXPECT_FALSE(composite->IsEnded(*state));

  // --- Step 5: Inside JSON string ---
  // JSON schema allows string characters ("a" = 2) or closing quote (`"` = 4).
  // Repetition penalty has history: 5, 2, 3, 6, 4.
  // Both "a" (2) and `"` (4) have been seen, so both receive repetition
  // penalty!
  ASSERT_OK_AND_ASSIGN(auto mask5, composite->ComputeMask(*state));
  std::vector<float> logits5(kVocabSize, 4.0f);
  EXPECT_OK(mask5->Apply(absl::MakeSpan(logits5)));

  // "a" (2) was seen in thinking -> 4.0 / 2.0 = 2.0f
  EXPECT_FLOAT_EQ(logits5[2], 2.0f);
  // `"` (4) was seen in opening quote -> 4.0 / 2.0 = 2.0f
  EXPECT_FLOAT_EQ(logits5[4], 2.0f);

  // Generate "a" (2)
  ASSERT_OK_AND_ASSIGN(state, composite->ComputeNext(*state, 2));
  EXPECT_FALSE(composite->IsEnded(*state));

  // --- Step 6: Close JSON string with `"` (4) ---
  ASSERT_OK_AND_ASSIGN(auto mask6, composite->ComputeMask(*state));
  ASSERT_OK_AND_ASSIGN(state, composite->ComputeNext(*state, 4));

  // Compute mask on terminal state so the parser recognizes completion
  ASSERT_OK_AND_ASSIGN(auto final_mask, composite->ComputeMask(*state));

  // JSON string is complete and valid -> composite constraint is ended!
  EXPECT_TRUE(composite->IsEnded(*state));
}

TEST(CompositeConstraintTest, ErrorHandlingInvalidTokenAndMismatchedState) {
  constexpr int kVocabSize = 10;
  auto c1 = std::make_unique<SuppressTokensConstraint>(
      kVocabSize, SuppressTokensConfig(absl::flat_hash_set<int>{1}));
  ASSERT_OK_AND_ASSIGN(auto composite, CompositeConstraint::Create(kVocabSize));
  ASSERT_OK(composite->AddConstraint(std::move(c1)));

  auto state = composite->Start();

  // Invalid token bounds
  EXPECT_THAT(composite->ComputeNext(*state, -1),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(composite->ComputeNext(*state, 10),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // Mismatched sub-state count
  CompositeConstraint::CompositeState invalid_state;
  EXPECT_THAT(composite->ComputeNext(invalid_state, 0),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(composite->ComputeMask(invalid_state),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // Null sub-state in composite state
  CompositeConstraint::CompositeState null_sub_state;
  null_sub_state.sub_states.push_back(nullptr);
  EXPECT_THAT(composite->ComputeNext(null_sub_state, 0),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(composite->ComputeMask(null_sub_state),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(CompositeConstraintTest, AddConstraintNullAndMismatchErrors) {
  ASSERT_OK_AND_ASSIGN(auto composite, CompositeConstraint::Create(10));
  EXPECT_THAT(composite->AddConstraint(nullptr),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(composite->AddUnownedConstraint(nullptr),
              StatusIs(absl::StatusCode::kInvalidArgument));

  auto mismatched = std::make_unique<SuppressTokensConstraint>(
      20, SuppressTokensConfig(absl::flat_hash_set<int>{1}));
  EXPECT_THAT(composite->AddConstraint(std::move(mismatched)),
              StatusIs(absl::StatusCode::kInvalidArgument));

  auto mismatched2 = std::make_unique<SuppressTokensConstraint>(
      20, SuppressTokensConfig(absl::flat_hash_set<int>{1}));
  EXPECT_THAT(composite->AddUnownedConstraint(mismatched2.get()),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(CompositeConstraintTest, AddConstraintInferredVocabSizeMismatch) {
  ASSERT_OK_AND_ASSIGN(auto composite, CompositeConstraint::Create());
  auto c1 = std::make_unique<SuppressTokensConstraint>(
      10, SuppressTokensConfig(absl::flat_hash_set<int>{1}));
  EXPECT_OK(composite->AddConstraint(std::move(c1)));
  EXPECT_EQ(composite->GetVocabularySize(), 10);

  auto c2 = std::make_unique<SuppressTokensConstraint>(
      20, SuppressTokensConfig(absl::flat_hash_set<int>{1}));
  EXPECT_THAT(composite->AddConstraint(std::move(c2)),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(CompositeConstraintTest, AddUnownedConstraintExecution) {
  constexpr int kVocabSize = 6;
  auto c1 = std::make_unique<SuppressTokensConstraint>(
      kVocabSize, SuppressTokensConfig(absl::flat_hash_set<int>{1, 2}));
  SuppressTokensConstraint c2(
      kVocabSize, SuppressTokensConfig(absl::flat_hash_set<int>{2, 4}));

  ASSERT_OK_AND_ASSIGN(auto composite, CompositeConstraint::Create(kVocabSize));
  EXPECT_OK(composite->AddConstraint(std::move(c1)));
  EXPECT_OK(composite->AddUnownedConstraint(&c2));
  EXPECT_EQ(composite->size(), 2);

  auto state = composite->Start();
  ASSERT_OK_AND_ASSIGN(auto mask, composite->ComputeMask(*state));
  std::vector<float> logits(kVocabSize, 5.0f);
  EXPECT_OK(mask->Apply(absl::MakeSpan(logits)));
  EXPECT_FLOAT_EQ(logits[0], 5.0f);
  EXPECT_FLOAT_EQ(logits[3], 5.0f);
  EXPECT_FLOAT_EQ(logits[5], 5.0f);
  ExpectMaskedOut(logits[1]);
  ExpectMaskedOut(logits[2]);
  ExpectMaskedOut(logits[4]);
}

TEST(CompositeConstraintTest, CreateValidationFailsOnNullConstraint) {
  std::vector<std::unique_ptr<Constraint>> list;
  list.push_back(nullptr);
  EXPECT_THAT(CompositeConstraint::Create(std::move(list)),
              StatusIs(absl::StatusCode::kInvalidArgument));

  std::vector<std::unique_ptr<Constraint>> list2;
  list2.push_back(std::make_unique<SuppressTokensConstraint>(
      10, SuppressTokensConfig(absl::flat_hash_set<int>{1})));
  list2.push_back(nullptr);
  EXPECT_THAT(CompositeConstraint::Create(std::move(list2)),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(CompositeConstraintTest, CreateValidationFailsOnVocabSizeMismatch) {
  std::vector<std::unique_ptr<Constraint>> list;
  list.push_back(std::make_unique<SuppressTokensConstraint>(
      10, SuppressTokensConfig(absl::flat_hash_set<int>{1})));
  list.push_back(std::make_unique<SuppressTokensConstraint>(
      20, SuppressTokensConfig(absl::flat_hash_set<int>{1})));
  EXPECT_THAT(CompositeConstraint::Create(std::move(list)),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(CompositeConstraintTest, CreateValidationFailsOnNegativeVocabSize) {
  EXPECT_THAT(CompositeConstraint::Create(-5),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(CompositeConstraintTest, CreateInfersVocabSizeFromConstraints) {
  std::vector<std::unique_ptr<Constraint>> list;
  list.push_back(std::make_unique<SuppressTokensConstraint>(
      10, SuppressTokensConfig(absl::flat_hash_set<int>{1})));
  list.push_back(std::make_unique<SuppressTokensConstraint>(
      10, SuppressTokensConfig(absl::flat_hash_set<int>{2})));
  ASSERT_OK_AND_ASSIGN(auto composite,
                       CompositeConstraint::Create(std::move(list)));
  EXPECT_EQ(composite->GetVocabularySize(), 10);
  EXPECT_EQ(composite->size(), 2);
}

class LegacyBitmapTestConstraint : public Constraint {
 public:
  class SimpleState : public State {};

  explicit LegacyBitmapTestConstraint(int vocab_size, int allowed_token)
      : vocab_size_(vocab_size), allowed_token_(allowed_token) {}

  std::unique_ptr<State> Start() const override {
    return std::make_unique<SimpleState>();
  }

  bool IsEnded(const State& state) const override { return false; }

  int GetVocabularySize() const override { return vocab_size_; }

  absl::StatusOr<std::unique_ptr<State>> ComputeNext(const State& state,
                                                     int token) const override {
    return std::make_unique<SimpleState>();
  }

  absl::StatusOr<std::unique_ptr<Bitmap>> ComputeBitmap(
      const State& state) const override {
    return std::make_unique<SingleAllowedTokenBitmap>(allowed_token_);
  }

 private:
  int vocab_size_;
  int allowed_token_;
};

class ErrorBitmapTestConstraint : public Constraint {
 public:
  class SimpleState : public State {};

  std::unique_ptr<State> Start() const override {
    return std::make_unique<SimpleState>();
  }

  bool IsEnded(const State& state) const override { return false; }

  int GetVocabularySize() const override { return 10; }

  absl::StatusOr<std::unique_ptr<State>> ComputeNext(const State& state,
                                                     int token) const override {
    return std::make_unique<SimpleState>();
  }

  absl::StatusOr<std::unique_ptr<Bitmap>> ComputeBitmap(
      const State& state) const override {
    return absl::InternalError("Simulated ComputeBitmap failure");
  }
};

TEST(CompositeConstraintTest, ComputeBitmapEmptyReturnsAllAllowed) {
  constexpr int kVocabSize = 5;
  ASSERT_OK_AND_ASSIGN(auto composite, CompositeConstraint::Create(kVocabSize));
  auto state = composite->Start();

  ASSERT_OK_AND_ASSIGN(auto bitmap, composite->ComputeBitmap(*state));
  ASSERT_NE(bitmap, nullptr);
  for (int i = 0; i < kVocabSize; ++i) {
    EXPECT_TRUE(bitmap->Get(i));
  }
}

TEST(CompositeConstraintTest, ComputeBitmapLegacyConstraint) {
  constexpr int kVocabSize = 5;
  auto legacy = std::make_unique<LegacyBitmapTestConstraint>(kVocabSize, 3);

  ASSERT_OK_AND_ASSIGN(auto composite, CompositeConstraint::Create(kVocabSize));
  ASSERT_OK(composite->AddConstraint(std::move(legacy)));

  auto state = composite->Start();
  ASSERT_OK_AND_ASSIGN(auto bitmap, composite->ComputeBitmap(*state));
  ASSERT_NE(bitmap, nullptr);

  EXPECT_FALSE(bitmap->Get(0));
  EXPECT_FALSE(bitmap->Get(1));
  EXPECT_FALSE(bitmap->Get(2));
  EXPECT_TRUE(bitmap->Get(3));
  EXPECT_FALSE(bitmap->Get(4));
}

TEST(CompositeConstraintTest, ComputeBitmapUnownedLegacyConstraint) {
  constexpr int kVocabSize = 5;
  LegacyBitmapTestConstraint legacy(kVocabSize, 2);

  ASSERT_OK_AND_ASSIGN(auto composite, CompositeConstraint::Create(kVocabSize));
  ASSERT_OK(composite->AddUnownedConstraint(&legacy));

  auto state = composite->Start();
  ASSERT_OK_AND_ASSIGN(auto bitmap, composite->ComputeBitmap(*state));
  ASSERT_NE(bitmap, nullptr);

  EXPECT_FALSE(bitmap->Get(0));
  EXPECT_FALSE(bitmap->Get(1));
  EXPECT_TRUE(bitmap->Get(2));
  EXPECT_FALSE(bitmap->Get(3));
  EXPECT_FALSE(bitmap->Get(4));
}

TEST(CompositeConstraintTest, ComputeBitmapModernMaskConstraintFallback) {
  constexpr int kVocabSize = 5;
  auto suppress = std::make_unique<SuppressTokensConstraint>(
      kVocabSize, SuppressTokensConfig(absl::flat_hash_set<int>{1, 3}));

  ASSERT_OK_AND_ASSIGN(auto composite, CompositeConstraint::Create(kVocabSize));
  ASSERT_OK(composite->AddConstraint(std::move(suppress)));

  auto state = composite->Start();
  ASSERT_OK_AND_ASSIGN(auto bitmap, composite->ComputeBitmap(*state));
  ASSERT_NE(bitmap, nullptr);

  EXPECT_TRUE(bitmap->Get(0));
  EXPECT_FALSE(bitmap->Get(1));
  EXPECT_TRUE(bitmap->Get(2));
  EXPECT_FALSE(bitmap->Get(3));
  EXPECT_TRUE(bitmap->Get(4));
}

TEST(CompositeConstraintTest, ComputeBitmapMultipleConstraintsChaining) {
  constexpr int kVocabSize = 5;
  auto suppress1 = std::make_unique<SuppressTokensConstraint>(
      kVocabSize, SuppressTokensConfig(absl::flat_hash_set<int>{1}));
  auto legacy = std::make_unique<LegacyBitmapTestConstraint>(kVocabSize, 2);
  auto suppress2 = std::make_unique<SuppressTokensConstraint>(
      kVocabSize, SuppressTokensConfig(absl::flat_hash_set<int>{4}));

  ASSERT_OK_AND_ASSIGN(auto composite, CompositeConstraint::Create(kVocabSize));
  ASSERT_OK(composite->AddConstraint(std::move(suppress1)));
  ASSERT_OK(composite->AddConstraint(std::move(legacy)));
  ASSERT_OK(composite->AddConstraint(std::move(suppress2)));

  auto state = composite->Start();
  ASSERT_OK_AND_ASSIGN(auto bitmap, composite->ComputeBitmap(*state));
  ASSERT_NE(bitmap, nullptr);

  EXPECT_FALSE(bitmap->Get(0));
  EXPECT_FALSE(bitmap->Get(1));
  EXPECT_TRUE(bitmap->Get(2));
  EXPECT_FALSE(bitmap->Get(3));
  EXPECT_FALSE(bitmap->Get(4));
}

class SparseMaskTestConstraint : public Constraint {
 public:
  class SimpleState : public State {};

  std::unique_ptr<State> Start() const override {
    return std::make_unique<SimpleState>();
  }

  bool IsEnded(const State& state) const override { return false; }

  int GetVocabularySize() const override { return 5; }

  absl::StatusOr<std::unique_ptr<State>> ComputeNext(const State& state,
                                                     int token) const override {
    return std::make_unique<SimpleState>();
  }

  absl::StatusOr<std::unique_ptr<LogitMask>> ComputeMask(
      const State& state) const override {
    return SparseLogitMask::CreateBiases({{1, -1.0f}});
  }
};

TEST(CompositeConstraintTest, ComputeBitmapPropagatesError) {
  auto error_constraint = std::make_unique<ErrorBitmapTestConstraint>();
  ASSERT_OK_AND_ASSIGN(auto composite, CompositeConstraint::Create(10));
  ASSERT_OK(composite->AddConstraint(std::move(error_constraint)));

  auto state = composite->Start();
  EXPECT_THAT(
      composite->ComputeBitmap(*state),
      StatusIs(absl::StatusCode::kInternal, "Simulated ComputeBitmap failure"));
}

class CustomMaskTestConstraint : public Constraint {
 public:
  class SimpleState : public State {};

  class CustomLogitMask : public LogitMask {
   public:
    MaskType GetType() const override { return MaskType::kCustom; }
    absl::Status Apply(absl::Span<float> logits) const override {
      return absl::OkStatus();
    }
    absl::Status Apply(absl::Span<tflite::half> logits) const override {
      return absl::OkStatus();
    }
  };

  std::unique_ptr<State> Start() const override {
    return std::make_unique<SimpleState>();
  }

  bool IsEnded(const State& state) const override { return false; }

  int GetVocabularySize() const override { return 10; }

  absl::StatusOr<std::unique_ptr<State>> ComputeNext(const State& state,
                                                     int token) const override {
    return std::make_unique<SimpleState>();
  }

  absl::StatusOr<std::unique_ptr<LogitMask>> ComputeMask(
      const State& state) const override {
    return std::make_unique<CustomLogitMask>();
  }
};

TEST(CompositeConstraintTest, ComputeBitmapUnsupportedMaskTypeReturnsError) {
  auto custom_constraint = std::make_unique<CustomMaskTestConstraint>();
  ASSERT_OK_AND_ASSIGN(auto composite, CompositeConstraint::Create(10));
  ASSERT_OK(composite->AddConstraint(std::move(custom_constraint)));

  auto state = composite->Start();
  EXPECT_THAT(composite->ComputeBitmap(*state),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "Cannot convert non-bitmap LogitMask to Bitmap."));
}

TEST(CompositeConstraintTest, ComputeBitmapStateValidationErrors) {
  auto suppress = std::make_unique<SuppressTokensConstraint>(
      10, SuppressTokensConfig(absl::flat_hash_set<int>{1}));
  ASSERT_OK_AND_ASSIGN(auto composite, CompositeConstraint::Create(10));
  ASSERT_OK(composite->AddConstraint(std::move(suppress)));

  CompositeConstraint::CompositeState invalid_state;
  EXPECT_THAT(composite->ComputeBitmap(invalid_state),
              StatusIs(absl::StatusCode::kInvalidArgument));

  CompositeConstraint::CompositeState null_sub_state;
  null_sub_state.sub_states.push_back(nullptr);
  EXPECT_THAT(composite->ComputeBitmap(null_sub_state),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

INSTANTIATE_TEST_SUITE_P(CompositeConstraintParamTests,
                         CompositeConstraintParamTest,
                         ::testing::Values(LogitsVariant::kFloat,
                                           LogitsVariant::kHalf));

}  // namespace
}  // namespace litert::lm
