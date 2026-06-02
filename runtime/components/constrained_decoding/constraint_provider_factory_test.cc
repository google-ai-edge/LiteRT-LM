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

#include "runtime/components/constrained_decoding/constraint_provider_factory.h"

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/components/constrained_decoding/constraint_provider_config.h"
#include "runtime/components/constrained_decoding/external_constraint_config.h"
#include "runtime/components/constrained_decoding/llg_constraint_config.h"
#include "runtime/components/tokenizer.h"
#include "runtime/util/test_utils.h"  // NOLINT

namespace litert::lm {
namespace {

using ::testing::Return;

class MockTokenizer : public Tokenizer {
 public:
  MOCK_METHOD(TokenizerType, GetTokenizerType, (), (const, override));
  MOCK_METHOD(absl::StatusOr<TokenIds>, TextToTokenIds, (absl::string_view),
              (override));
  MOCK_METHOD(absl::StatusOr<int>, TokenToId, (absl::string_view), (override));
  MOCK_METHOD(absl::StatusOr<std::string>, TokenIdsToText, (const TokenIds&),
              (override));
  MOCK_METHOD(std::vector<std::string>, GetTokens, (), (const, override));
};

class ConstraintProviderFactoryTest : public ::testing::Test {
 protected:
  MockTokenizer tokenizer_;
};

TEST_F(ConstraintProviderFactoryTest, CreateExternalConstraintProvider) {
  ExternalConstraintConfig config;
  std::vector<std::vector<int>> stop_token_ids;
  auto provider = CreateConstraintProvider(config, tokenizer_, stop_token_ids);
  ASSERT_TRUE(provider.ok());
  EXPECT_NE(provider.value(), nullptr);
}

TEST_F(ConstraintProviderFactoryTest, CreateLlgConstraintProvider) {
  LlGuidanceConfig config;
  config.eos_id = 1;
  std::vector<std::vector<int>> stop_token_ids;

  EXPECT_CALL(tokenizer_, GetTokens())
      .WillOnce(Return(std::vector<std::string>{"<pad>", "<eos>", "a", "b"}));

  auto provider = CreateConstraintProvider(config, tokenizer_, stop_token_ids);
  ASSERT_TRUE(provider.ok());
  EXPECT_NE(provider.value(), nullptr);
}

TEST_F(ConstraintProviderFactoryTest, CreateLlgConstraintProviderInferEosId) {
  LlGuidanceConfig config;
  // eos_id is missing, but stop_token_ids has a valid token
  std::vector<std::vector<int>> stop_token_ids = {{2}};

  EXPECT_CALL(tokenizer_, GetTokens())
      .WillOnce(Return(std::vector<std::string>{"<pad>", "<eos>", "a", "b"}));

  auto provider = CreateConstraintProvider(config, tokenizer_, stop_token_ids);
  ASSERT_TRUE(provider.ok());
  EXPECT_NE(provider.value(), nullptr);
}

TEST_F(ConstraintProviderFactoryTest, CreateLlgConstraintProviderMissingEosId) {
  LlGuidanceConfig config;
  // eos_id is missing, and stop_token_ids is empty
  std::vector<std::vector<int>> stop_token_ids;

  auto provider = CreateConstraintProvider(config, tokenizer_, stop_token_ids);
  EXPECT_FALSE(provider.ok());
  EXPECT_EQ(provider.status().code(), absl::StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace litert::lm
