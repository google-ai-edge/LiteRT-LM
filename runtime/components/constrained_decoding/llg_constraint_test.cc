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

#include <cmath>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "runtime/components/constrained_decoding/bitmap.h"
#include "runtime/components/constrained_decoding/constraint.h"
#include "runtime/components/constrained_decoding/constraint_provider.h"
#include "runtime/components/constrained_decoding/llg_constraint_config.h"
#include "runtime/components/constrained_decoding/llg_constraint_provider.h"
#include "runtime/components/constrained_decoding/logit_mask.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep
#include "support/tokenizer/tokenizer.h"

namespace litert::lm {

using Tokenizer = ::litert::support::Tokenizer;
using TokenizerType = ::litert::support::TokenizerType;
using TokenIds = ::litert::support::TokenIds;

namespace {

using ::testing::Return;

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

class LlgConstraintTest : public ::testing::Test {
 protected:
  absl::StatusOr<std::unique_ptr<ConstraintProvider>> CreateProvider() {
    LlGuidanceConfig config;
    config.eos_ids = {1};

    EXPECT_CALL(tokenizer_, GetTokens())
        .WillOnce(
            Return(std::vector<std::string>{"<pad>", "<eos>", "a", "b", "\""}));

    EXPECT_CALL(tokenizer_, TextToTokenIds(::testing::_))
        .WillRepeatedly([](absl::string_view text) {
          if (text == "a") return TokenIds{2};
          if (text == "b") return TokenIds{3};
          if (text == "\"") return TokenIds{4};
          if (text == "ab") return TokenIds{2, 3};
          return TokenIds{};
        });

    return LlgConstraintProvider::Create(tokenizer_, config);
  }

  MockTokenizer tokenizer_;
};

TEST_F(LlgConstraintTest, GetVocabularySize) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<ConstraintProvider> provider,
                       CreateProvider());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Constraint> constraint,
                       provider->CreateConstraint(LlGuidanceConstraintArg{
                           .constraint_type = LlgConstraintType::kRegex,
                           .constraint_string = "a+"}));
  EXPECT_EQ(constraint->GetVocabularySize(), 5);
}

TEST_F(LlgConstraintTest, ComputeBitmap) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<ConstraintProvider> provider,
                       CreateProvider());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Constraint> constraint,
                       provider->CreateConstraint(LlGuidanceConstraintArg{
                           .constraint_type = LlgConstraintType::kRegex,
                           .constraint_string = "a+"}));

  std::unique_ptr<Constraint::State> state = constraint->Start();
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Bitmap> bitmap,
                       constraint->ComputeBitmap(*state));

  EXPECT_TRUE(bitmap->Get(2));   // a
  EXPECT_FALSE(bitmap->Get(3));  // b
}

TEST_F(LlgConstraintTest, ComputeMask) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<ConstraintProvider> provider,
                       CreateProvider());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Constraint> constraint,
                       provider->CreateConstraint(LlGuidanceConstraintArg{
                           .constraint_type = LlgConstraintType::kRegex,
                           .constraint_string = "a+"}));

  std::unique_ptr<Constraint::State> state = constraint->Start();
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<LogitMask> mask,
                       constraint->ComputeMask(*state));

  ASSERT_NE(mask, nullptr);
  EXPECT_EQ(mask->GetType(), MaskType::kBitmap);
  auto* bitmap_mask = static_cast<BitmapLogitMask*>(mask.get());
  EXPECT_TRUE(bitmap_mask->IsAllowed(2));   // a
  EXPECT_FALSE(bitmap_mask->IsAllowed(3));  // b

  // Verify Apply on logits
  std::vector<float> logits(5, 0.0f);
  EXPECT_OK(mask->Apply(absl::MakeSpan(logits)));
  EXPECT_EQ(logits[2], 0.0f);
  EXPECT_TRUE(std::isinf(logits[3]) && logits[3] < 0.0f);
}

TEST_F(LlgConstraintTest, TransitionAndEnd) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<ConstraintProvider> provider,
                       CreateProvider());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Constraint> constraint,
                       provider->CreateConstraint(LlGuidanceConstraintArg{
                           .constraint_type = LlgConstraintType::kRegex,
                           .constraint_string = "a"}));

  std::unique_ptr<Constraint::State> state = constraint->Start();
  EXPECT_FALSE(constraint->IsEnded(*state));
  EXPECT_OK(constraint->ComputeBitmap(*state));

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Constraint::State> next_state,
                       constraint->ComputeNext(*state, 2));
  state = std::move(next_state);
  EXPECT_OK(constraint->ComputeBitmap(*state));
  EXPECT_TRUE(constraint->IsEnded(*state));
}

TEST_F(LlgConstraintTest, InvalidTransition) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<ConstraintProvider> provider,
                       CreateProvider());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Constraint> constraint,
                       provider->CreateConstraint(LlGuidanceConstraintArg{
                           .constraint_type = LlgConstraintType::kRegex,
                           .constraint_string = "a"}));

  std::unique_ptr<Constraint::State> state = constraint->Start();
  auto next_state_or = constraint->ComputeNext(*state, 3);  // b
}

TEST_F(LlgConstraintTest, LarkConstraint) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<ConstraintProvider> provider,
                       CreateProvider());
  // Simple Lark grammar: matches "a".
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Constraint> constraint,
                       provider->CreateConstraint(LlGuidanceConstraintArg{
                           .constraint_type = LlgConstraintType::kLark,
                           .constraint_string = "start: \"a\""}));

  std::unique_ptr<Constraint::State> state = constraint->Start();
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Bitmap> bitmap,
                       constraint->ComputeBitmap(*state));
  EXPECT_TRUE(bitmap->Get(2));  // a allowed
}

TEST_F(LlgConstraintTest, JsonConstraint) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<ConstraintProvider> provider,
                       CreateProvider());
  // JSON schema accepting a string.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Constraint> constraint,
                       provider->CreateConstraint(LlGuidanceConstraintArg{
                           .constraint_type = LlgConstraintType::kJsonSchema,
                           .constraint_string = R"({"type": "string"})"}));

  std::unique_ptr<Constraint::State> state = constraint->Start();
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Bitmap> bitmap,
                       constraint->ComputeBitmap(*state));
  // Should allow quote for starting a string.
  EXPECT_TRUE(bitmap->Get(4));  // " allowed.
}

TEST_F(LlgConstraintTest, RegexSequenceTest) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<ConstraintProvider> provider,
                       CreateProvider());
  // Match exact sequence "ab".
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Constraint> constraint,
                       provider->CreateConstraint(LlGuidanceConstraintArg{
                           .constraint_type = LlgConstraintType::kRegex,
                           .constraint_string = "ab"}));

  std::unique_ptr<Constraint::State> state = constraint->Start();

  // 1. Check start: Expect 'a'.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Bitmap> bitmap,
                       constraint->ComputeBitmap(*state));
  EXPECT_TRUE(bitmap->Get(2));   // a
  EXPECT_FALSE(bitmap->Get(3));  // b

  // 2. Consume 'a'.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Constraint::State> next_state,
                       constraint->ComputeNext(*state, 2));
  state = std::move(next_state);
  EXPECT_FALSE(constraint->IsEnded(*state));

  // 3. Check middle: Expect 'b'.
  ASSERT_OK_AND_ASSIGN(bitmap, constraint->ComputeBitmap(*state));
  EXPECT_FALSE(bitmap->Get(2));  // a
  EXPECT_TRUE(bitmap->Get(3));   // b

  // 4. Consume 'b'.
  ASSERT_OK_AND_ASSIGN(next_state, constraint->ComputeNext(*state, 3));
  state = std::move(next_state);

  // 5. Check end.
  EXPECT_OK(constraint->ComputeBitmap(*state));
  EXPECT_TRUE(constraint->IsEnded(*state));
}

TEST_F(LlgConstraintTest, LarkSequenceTest) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<ConstraintProvider> provider,
                       CreateProvider());
  // Simple Lark grammar: matches "a" then "b".
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Constraint> constraint,
                       provider->CreateConstraint(LlGuidanceConstraintArg{
                           .constraint_type = LlgConstraintType::kLark,
                           .constraint_string = "start: \"a\" \"b\""}));

  std::unique_ptr<Constraint::State> state = constraint->Start();

  // 1. Check start: Expect 'a'.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Bitmap> bitmap,
                       constraint->ComputeBitmap(*state));
  EXPECT_TRUE(bitmap->Get(2));   // a
  EXPECT_FALSE(bitmap->Get(3));  // b

  // 2. Consume 'a'.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Constraint::State> next_state,
                       constraint->ComputeNext(*state, 2));
  state = std::move(next_state);
  EXPECT_FALSE(constraint->IsEnded(*state));

  // 3. Check middle: Expect 'b'.
  ASSERT_OK_AND_ASSIGN(bitmap, constraint->ComputeBitmap(*state));
  EXPECT_FALSE(bitmap->Get(2));  // a
  EXPECT_TRUE(bitmap->Get(3));   // b

  // 4. Consume 'b'.
  ASSERT_OK_AND_ASSIGN(next_state, constraint->ComputeNext(*state, 3));
  state = std::move(next_state);

  // 5. Check end.
  EXPECT_OK(constraint->ComputeBitmap(*state));
  EXPECT_TRUE(constraint->IsEnded(*state));
}

TEST_F(LlgConstraintTest, JsonSequenceTest) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<ConstraintProvider> provider,
                       CreateProvider());
  // JSON schema for a string.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Constraint> constraint,
                       provider->CreateConstraint(LlGuidanceConstraintArg{
                           .constraint_type = LlgConstraintType::kJsonSchema,
                           .constraint_string = R"({"type": "string"})"}));

  std::unique_ptr<Constraint::State> state = constraint->Start();

  // 1. Check start: Expect '"'.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Bitmap> bitmap,
                       constraint->ComputeBitmap(*state));
  EXPECT_TRUE(bitmap->Get(4));  // " allowed.

  // 2. Consume '"'.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Constraint::State> next_state,
                       constraint->ComputeNext(*state, 4));
  state = std::move(next_state);
  EXPECT_FALSE(constraint->IsEnded(*state));

  // 3. 'a' and '"' are allowed.
  ASSERT_OK_AND_ASSIGN(bitmap, constraint->ComputeBitmap(*state));
  EXPECT_TRUE(bitmap->Get(2));  // a
  EXPECT_TRUE(bitmap->Get(4));  // "

  // 4. Consume 'a'.
  ASSERT_OK_AND_ASSIGN(next_state, constraint->ComputeNext(*state, 2));
  state = std::move(next_state);
  EXPECT_FALSE(constraint->IsEnded(*state));

  // 5. 'a' and '"' are allowed.
  ASSERT_OK_AND_ASSIGN(bitmap, constraint->ComputeBitmap(*state));
  EXPECT_TRUE(bitmap->Get(2));  // a
  EXPECT_TRUE(bitmap->Get(4));  // "

  // 6. Consume '"'.
  ASSERT_OK_AND_ASSIGN(next_state, constraint->ComputeNext(*state, 4));
  state = std::move(next_state);

  // 7. Check end.
  EXPECT_OK(constraint->ComputeBitmap(*state));
  EXPECT_TRUE(constraint->IsEnded(*state));
}

TEST_F(LlgConstraintTest, ComputeNextDoesNotMutateInputState) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<ConstraintProvider> provider,
                       CreateProvider());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Constraint> constraint,
                       provider->CreateConstraint(LlGuidanceConstraintArg{
                           .constraint_type = LlgConstraintType::kRegex,
                           .constraint_string = "ab"}));

  std::unique_ptr<Constraint::State> state_0 = constraint->Start();
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<LogitMask> mask_0,
                       constraint->ComputeMask(*state_0));
  auto* bitmap_mask_0 = static_cast<BitmapLogitMask*>(mask_0.get());
  EXPECT_TRUE(bitmap_mask_0->IsAllowed(2));   // 'a'
  EXPECT_FALSE(bitmap_mask_0->IsAllowed(3));  // 'b'

  // Speculatively advance from state_0 -> state_1 ('a') -> state_2 ('b').
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Constraint::State> state_1,
                       constraint->ComputeNext(*state_0, 2));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<LogitMask> mask_1,
                       constraint->ComputeMask(*state_1));
  auto* bitmap_mask_1 = static_cast<BitmapLogitMask*>(mask_1.get());
  EXPECT_FALSE(bitmap_mask_1->IsAllowed(2));  // 'a'
  EXPECT_TRUE(bitmap_mask_1->IsAllowed(3));   // 'b'

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Constraint::State> state_2,
                       constraint->ComputeNext(*state_1, 3));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<LogitMask> mask_2,
                       constraint->ComputeMask(*state_2));
  auto* bitmap_mask_2 = static_cast<BitmapLogitMask*>(mask_2.get());
  EXPECT_TRUE(bitmap_mask_2->IsAllowed(1));   // '<eos>'
  EXPECT_FALSE(bitmap_mask_2->IsAllowed(2));  // 'a'
  EXPECT_FALSE(bitmap_mask_2->IsAllowed(3));  // 'b'
  EXPECT_TRUE(constraint->IsEnded(*state_2));

  // Verify that state_0 and state_1 were NOT mutated in place by ComputeNext.
  EXPECT_FALSE(constraint->IsEnded(*state_0));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<LogitMask> mask_0_recomputed,
                       constraint->ComputeMask(*state_0));
  auto* bitmap_mask_0_recomputed =
      static_cast<BitmapLogitMask*>(mask_0_recomputed.get());
  EXPECT_TRUE(bitmap_mask_0_recomputed->IsAllowed(2));   // 'a'
  EXPECT_FALSE(bitmap_mask_0_recomputed->IsAllowed(3));  // 'b'

  EXPECT_FALSE(constraint->IsEnded(*state_1));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<LogitMask> mask_1_recomputed,
                       constraint->ComputeMask(*state_1));
  auto* bitmap_mask_1_recomputed =
      static_cast<BitmapLogitMask*>(mask_1_recomputed.get());
  EXPECT_FALSE(bitmap_mask_1_recomputed->IsAllowed(2));  // 'a'
  EXPECT_TRUE(bitmap_mask_1_recomputed->IsAllowed(3));   // 'b'

  // Branching from state_0 again after rejecting speculative draft tokens must
  // succeed and produce a valid state_1_retry.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Constraint::State> state_1_retry,
                       constraint->ComputeNext(*state_0, 2));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Bitmap> bitmap_1_retry,
                       constraint->ComputeBitmap(*state_1_retry));
  EXPECT_FALSE(bitmap_1_retry->Get(2));  // 'a'
  EXPECT_TRUE(bitmap_1_retry->Get(3));   // 'b'
}

TEST_F(LlgConstraintTest, ComputeMaskAndBitmapAreIdempotentAfterStop) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<ConstraintProvider> provider,
                       CreateProvider());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Constraint> constraint,
                       provider->CreateConstraint(LlGuidanceConstraintArg{
                           .constraint_type = LlgConstraintType::kRegex,
                           .constraint_string = "a"}));

  std::unique_ptr<Constraint::State> state = constraint->Start();
  ASSERT_OK_AND_ASSIGN(state, constraint->ComputeNext(*state, 2));  // 'a'

  // First ComputeMask call transitions llguidance to its stopped state and
  // returns the EOS-only mask.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<LogitMask> mask_1,
                       constraint->ComputeMask(*state));
  EXPECT_TRUE(constraint->IsEnded(*state));
  auto* bitmap_mask_1 = static_cast<BitmapLogitMask*>(mask_1.get());
  EXPECT_TRUE(bitmap_mask_1->IsAllowed(1));   // '<eos>'
  EXPECT_FALSE(bitmap_mask_1->IsAllowed(2));  // 'a'

  // Subsequent ComputeMask and ComputeBitmap calls on the stopped state must
  // succeed idempotently (instead of failing with "compute_mask() called after
  // stop") and continue returning the EOS-only mask.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<LogitMask> mask_2,
                       constraint->ComputeMask(*state));
  auto* bitmap_mask_2 = static_cast<BitmapLogitMask*>(mask_2.get());
  EXPECT_TRUE(bitmap_mask_2->IsAllowed(1));   // '<eos>'
  EXPECT_FALSE(bitmap_mask_2->IsAllowed(2));  // 'a'

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Bitmap> bitmap,
                       constraint->ComputeBitmap(*state));
  EXPECT_TRUE(bitmap->Get(1));   // '<eos>'
  EXPECT_FALSE(bitmap->Get(2));  // 'a'
}

TEST_F(LlgConstraintTest,
       SpeculativeDecodingDraftRejectionOnJsonSchemaPreservesGrammar) {
  LlGuidanceConfig config;
  config.eos_ids = {1};

  // Token IDs:
  // 0: <pad>, 1: <eos>, 2: {, 3: }, 4: ", 5: :, 6: ,,
  // 7: "intent", 8: "slots", 9: UN, 10: SUPPORTED, 11: KNOWN
  const std::vector<std::string> vocab = {
      "<pad>", "<eos>",      "{",         "}",  "\"",        ":",
      ",",     "\"intent\"", "\"slots\"", "UN", "SUPPORTED", "KNOWN"};

  MockTokenizer json_tokenizer;
  EXPECT_CALL(json_tokenizer, GetTokens()).WillOnce(Return(vocab));
  EXPECT_CALL(json_tokenizer, TextToTokenIds(::testing::_))
      .WillRepeatedly([&vocab](absl::string_view text) {
        TokenIds ids;
        while (!text.empty()) {
          int best_id = -1;
          size_t best_len = 0;
          for (int id = 2; id < static_cast<int>(vocab.size()); ++id) {
            const std::string& tok = vocab[id];
            if (tok.size() > best_len && text.substr(0, tok.size()) == tok) {
              best_id = id;
              best_len = tok.size();
            }
          }
          if (best_id < 0) {
            return TokenIds{};
          }
          ids.push_back(best_id);
          text.remove_prefix(best_len);
        }
        return ids;
      });

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<ConstraintProvider> provider,
                       LlgConstraintProvider::Create(json_tokenizer, config));
  constexpr absl::string_view kSchema = R"json({
    "type": "object",
    "additionalProperties": false,
    "required": ["intent", "slots"],
    "properties": {
      "intent": {"type": "string", "enum": ["UNSUPPORTED", "UNKNOWN"]},
      "slots": {"type": "object", "additionalProperties": {"type": "string"}}
    }
  })json";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Constraint> constraint,
                       provider->CreateConstraint(LlGuidanceConstraintArg{
                           .constraint_type = LlgConstraintType::kJsonSchema,
                           .constraint_string = std::string(kSchema)}));

  // Advance verified state through `{"intent":"` (tokens 2, 7, 5, 4).
  std::unique_ptr<Constraint::State> verified_state = constraint->Start();
  for (int token : {2, 7, 5, 4}) {
    ASSERT_OK_AND_ASSIGN(verified_state,
                         constraint->ComputeNext(*verified_state, token));
  }

  // Simulate an MTP round (gamma = 3) starting at `{"intent":"`:
  // Drafter proposes d_0 = 9 ("UN"), d_1 = 11 ("KNOWN"), d_2 = 4 ("\"").
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<LogitMask> draft_mask_0,
                       constraint->ComputeMask(*verified_state));
  EXPECT_TRUE(static_cast<BitmapLogitMask*>(draft_mask_0.get())->IsAllowed(9));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Constraint::State> draft_state_0,
                       constraint->ComputeNext(*verified_state, 9));  // "UN"

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<LogitMask> draft_mask_1,
                       constraint->ComputeMask(*draft_state_0));
  auto* bitmap_draft_1 = static_cast<BitmapLogitMask*>(draft_mask_1.get());
  EXPECT_TRUE(bitmap_draft_1->IsAllowed(10));  // "SUPPORTED"
  EXPECT_TRUE(bitmap_draft_1->IsAllowed(11));  // "KNOWN"
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Constraint::State> draft_state_1,
                       constraint->ComputeNext(*draft_state_0, 11));  // "KNOWN"

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<LogitMask> draft_mask_2,
                       constraint->ComputeMask(*draft_state_1));
  EXPECT_TRUE(static_cast<BitmapLogitMask*>(draft_mask_2.get())->IsAllowed(4));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Constraint::State> draft_state_2,
                       constraint->ComputeNext(*draft_state_1, 4));  // "\""

  // Verifier accepts d_0 = 9 ("UN"), rejects d_1 = 11 ("KNOWN"), and samples
  // bonus_token = 10 ("SUPPORTED") at position 1. Recomputing draft_state_0's
  // mask must still reflect `{"intent":"UN`. Before the fix, draft_state_0 was
  // mutated in place to draft_state_2, which disallowed token 10 ("SUPPORTED")
  // and allowed token 6 (",") prematurely.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<LogitMask> draft_mask_1_recomputed,
                       constraint->ComputeMask(*draft_state_0));
  auto* bitmap_draft_1_recomputed =
      static_cast<BitmapLogitMask*>(draft_mask_1_recomputed.get());
  EXPECT_TRUE(bitmap_draft_1_recomputed->IsAllowed(10));  // "SUPPORTED"
  EXPECT_FALSE(
      bitmap_draft_1_recomputed->IsAllowed(6));  // "," before closing "

  verified_state = std::move(draft_state_0);
  ASSERT_OK_AND_ASSIGN(
      verified_state,
      constraint->ComputeNext(*verified_state, 10));  // bonus: "SUPPORTED"

  // Remaining sequence `","slots":{}}` (tokens 4, 6, 8, 5, 2, 3, 3) must all
  // be allowed and lead to a stopped state that allows only <eos> (token 1).
  for (int token : {4, 6, 8, 5, 2, 3, 3}) {
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<LogitMask> mask,
                         constraint->ComputeMask(*verified_state));
    EXPECT_TRUE(static_cast<BitmapLogitMask*>(mask.get())->IsAllowed(token));
    ASSERT_OK_AND_ASSIGN(verified_state,
                         constraint->ComputeNext(*verified_state, token));
  }

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<LogitMask> final_mask,
                       constraint->ComputeMask(*verified_state));
  EXPECT_TRUE(constraint->IsEnded(*verified_state));
  EXPECT_TRUE(static_cast<BitmapLogitMask*>(final_mask.get())->IsAllowed(1));
  EXPECT_FALSE(static_cast<BitmapLogitMask*>(final_mask.get())->IsAllowed(3));
}

// Creates a provider for a vocabulary with two EOS tokens, "<eos>" and "<end>",
// neither of which is listed as a special token.
absl::StatusOr<std::unique_ptr<ConstraintProvider>> CreateTwoEosProvider(
    MockTokenizer& tokenizer) {
  EXPECT_CALL(tokenizer, GetTokens())
      .WillOnce(Return(
          std::vector<std::string>{"<pad>", "<eos>", "a", "b", "<end>"}));
  EXPECT_CALL(tokenizer, TextToTokenIds(::testing::_))
      .WillRepeatedly([](absl::string_view text) {
        if (text == "a") return TokenIds{2};
        if (text == "b") return TokenIds{3};
        return TokenIds{};
      });
  LlGuidanceConfig config;
  config.eos_ids = {1, 4};
  return LlgConstraintProvider::Create(tokenizer, config);
}

TEST(LlgConstraintMultipleEosTest, AnyEosTokenEndsTheOutput) {
  MockTokenizer tokenizer;
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<ConstraintProvider> provider,
                       CreateTwoEosProvider(tokenizer));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Constraint> constraint,
                       provider->CreateConstraint(LlGuidanceConstraintArg{
                           .constraint_type = LlgConstraintType::kRegex,
                           .constraint_string = "a+"}));

  std::unique_ptr<Constraint::State> state = constraint->Start();
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Bitmap> start_bitmap,
                       constraint->ComputeBitmap(*state));
  EXPECT_FALSE(start_bitmap->Get(1));  // '<eos>'
  EXPECT_FALSE(start_bitmap->Get(4));  // '<end>'

  ASSERT_OK_AND_ASSIGN(state, constraint->ComputeNext(*state, 2));  // 'a'
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Bitmap> bitmap,
                       constraint->ComputeBitmap(*state));
  EXPECT_TRUE(bitmap->Get(1));  // '<eos>'
  EXPECT_TRUE(bitmap->Get(4));  // '<end>'

  ASSERT_OK_AND_ASSIGN(state, constraint->ComputeNext(*state, 4));  // '<end>'
  // llguidance applies the stop when it computes the next mask.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Bitmap> end_bitmap,
                       constraint->ComputeBitmap(*state));
  EXPECT_TRUE(constraint->IsEnded(*state));
  EXPECT_FALSE(end_bitmap->Get(2));  // 'a'
}

TEST(LlgConstraintMultipleEosTest, EosTokensAreNotText) {
  MockTokenizer tokenizer;
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<ConstraintProvider> provider,
                       CreateTwoEosProvider(tokenizer));
  // Text, then 'b'. As plain text, "<eos>" and "<end>" would match the text.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Constraint> constraint,
                       provider->CreateConstraint(LlGuidanceConstraintArg{
                           .constraint_type = LlgConstraintType::kRegex,
                           .constraint_string = "[^b]*b"}));

  std::unique_ptr<Constraint::State> state = constraint->Start();
  ASSERT_OK_AND_ASSIGN(state, constraint->ComputeNext(*state, 2));  // 'a'
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Bitmap> bitmap,
                       constraint->ComputeBitmap(*state));
  EXPECT_TRUE(bitmap->Get(2));   // 'a'
  EXPECT_TRUE(bitmap->Get(3));   // 'b'
  EXPECT_FALSE(bitmap->Get(1));  // '<eos>'
  EXPECT_FALSE(bitmap->Get(4));  // '<end>'
}

TEST(LlgConstraintMultipleEosTest, EndedConstraintAllowsEveryEosToken) {
  MockTokenizer tokenizer;
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<ConstraintProvider> provider,
                       CreateTwoEosProvider(tokenizer));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Constraint> constraint,
                       provider->CreateConstraint(LlGuidanceConstraintArg{
                           .constraint_type = LlgConstraintType::kRegex,
                           .constraint_string = "a"}));

  std::unique_ptr<Constraint::State> state = constraint->Start();
  ASSERT_OK_AND_ASSIGN(state, constraint->ComputeNext(*state, 2));  // 'a'

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<LogitMask> mask,
                       constraint->ComputeMask(*state));
  EXPECT_TRUE(constraint->IsEnded(*state));
  auto* bitmap_mask = static_cast<BitmapLogitMask*>(mask.get());
  EXPECT_TRUE(bitmap_mask->IsAllowed(1));   // '<eos>'
  EXPECT_TRUE(bitmap_mask->IsAllowed(4));   // '<end>'
  EXPECT_FALSE(bitmap_mask->IsAllowed(2));  // 'a'

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Bitmap> bitmap,
                       constraint->ComputeBitmap(*state));
  EXPECT_TRUE(bitmap->Get(1));   // '<eos>'
  EXPECT_TRUE(bitmap->Get(4));   // '<end>'
  EXPECT_FALSE(bitmap->Get(2));  // 'a'
}

}  // namespace
}  // namespace litert::lm
