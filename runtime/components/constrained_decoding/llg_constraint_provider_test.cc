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

#include "runtime/components/constrained_decoding/llg_constraint_provider.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "runtime/components/constrained_decoding/llg_constraint_config.h"
#include "runtime/util/test_utils.h"  // NOLINT
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

class LlgConstraintProviderTest : public ::testing::Test {
 protected:
  MockTokenizer tokenizer_;
};

TEST_F(LlgConstraintProviderTest, CreateSuccess) {
  LlGuidanceConfig config;
  config.eos_id = 1;

  EXPECT_CALL(tokenizer_, GetTokens())
      .WillOnce(Return(std::vector<std::string>{"<pad>", "<eos>", "a", "b"}));

  auto provider = LlgConstraintProvider::Create(tokenizer_, config);
  EXPECT_TRUE(provider.ok());
}

TEST_F(LlgConstraintProviderTest, CreateFailsWithoutEosId) {
  LlGuidanceConfig config;
  auto provider = LlgConstraintProvider::Create(tokenizer_, config);
  EXPECT_FALSE(provider.ok());
  EXPECT_EQ(provider.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST_F(LlgConstraintProviderTest, CreateConstraintProviderSuccess) {
  LlGuidanceConfig config;
  config.eos_id = 1;

  EXPECT_CALL(tokenizer_, GetTokens())
      .WillOnce(Return(std::vector<std::string>{"<pad>", "<eos>", "a", "b"}));

  auto constraint_provider = LlgConstraintProvider::Create(tokenizer_, config);
  ASSERT_TRUE(constraint_provider.ok());
}

TEST_F(LlgConstraintProviderTest, CreateConstraintSuccess) {
  LlGuidanceConfig config;
  config.eos_id = 1;

  EXPECT_CALL(tokenizer_, GetTokens())
      .WillOnce(Return(std::vector<std::string>{"<pad>", "<eos>", "a", "b"}));

  auto constraint_provider = LlgConstraintProvider::Create(tokenizer_, config);
  ASSERT_TRUE(constraint_provider.ok());
  auto provider = std::move(constraint_provider.value());

  auto constraint = provider->CreateConstraint(LlGuidanceConstraintArg{
      .constraint_type = LlgConstraintType::kRegex, .constraint_string = "a+"});
  EXPECT_TRUE(constraint.ok());
}

}  // namespace
}  // namespace litert::lm
