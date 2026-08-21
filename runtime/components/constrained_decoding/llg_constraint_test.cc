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
    config.eos_id = 1;

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

}  // namespace
}  // namespace litert::lm
