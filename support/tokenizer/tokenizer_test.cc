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

#include "support/tokenizer/tokenizer.h"

#include <fcntl.h>

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
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/test/matchers.h"  // from @litert
#include "support/util/convert_tensor_buffer.h"

namespace litert::support {
namespace {

class MockTokenizer : public Tokenizer {
 public:
  MOCK_METHOD(absl::StatusOr<std::vector<int>>, TextToTokenIds,
              (absl::string_view text), (override));
  MOCK_METHOD(absl::StatusOr<int>, TokenToId, (absl::string_view token),
              (override));
  using Tokenizer::TokenIdsToText;
  MOCK_METHOD(absl::StatusOr<std::string>, TokenIdsToText,
              (absl::Span<const int> token_ids, bool skip_special_tokens),
              (override));
  MOCK_METHOD(TokenizerType, GetTokenizerType, (), (const, override));
  MOCK_METHOD(std::vector<std::string>, GetTokens, (), (const, override));
  MOCK_METHOD(int, GetVocabSize, (), (const, override));
};

TEST(TokenizerTest, TextToTensorBuffer) {
  auto tokenizer = std::make_unique<MockTokenizer>();
  EXPECT_CALL(*tokenizer, TextToTokenIds("Hello World!"))
      .WillOnce(
          testing::Return(std::vector<int>{90, 547, 58, 735, 210, 466, 2294}));

  absl::string_view text = "Hello World!";
  auto ids_or = tokenizer->TextToTokenIds(text);
  EXPECT_TRUE(ids_or.ok());

  auto tensor_or = tokenizer->TokenIdsToTensorBuffer(ids_or.value());
  auto tensor = std::move(tensor_or.value());
  LITERT_ASSERT_OK_AND_ASSIGN(auto tensor_type, tensor.TensorType());
  EXPECT_EQ(tensor_type.Layout().Dimensions(), ::litert::Dimensions({1, 7}));

  auto copied_data = CopyFromTensorBuffer2D<int>(tensor);
  EXPECT_TRUE(copied_data.HasValue());
  EXPECT_THAT((*copied_data)[0],
              ::testing::ElementsAre(90, 547, 58, 735, 210, 466, 2294));
}

TEST(TokenizerTest, TensorBufferToTokenIds) {
  auto tokenizer = std::make_unique<MockTokenizer>();

  const std::vector<int> ids = {90,  547, 58, 735, 210, 466, 2294,
                                224, 24,  8,  66,  246, 18,  2295};
  LITERT_ASSERT_OK_AND_ASSIGN(TensorBuffer tensor_buffer,
                              CopyToTensorBuffer<int>(ids, {2, 7}));
  LITERT_ASSERT_OK_AND_ASSIGN(auto tensor_buffer_type,
                              tensor_buffer.TensorType());
  EXPECT_EQ(tensor_buffer_type.Layout().Dimensions(),
            ::litert::Dimensions({2, 7}));

  auto token_ids = Tokenizer::TensorBufferToTokenIds(tensor_buffer);
  EXPECT_TRUE(token_ids.ok());
  EXPECT_EQ(token_ids.value().size(), 2);
  EXPECT_EQ(token_ids.value()[0],
            std::vector<int>({90, 547, 58, 735, 210, 466, 2294}));
  EXPECT_EQ(token_ids.value()[1],
            std::vector<int>({224, 24, 8, 66, 246, 18, 2295}));
}

TEST(TokenizerTest, TokenIdsToTexts) {
  auto tokenizer = std::make_unique<MockTokenizer>();
  EXPECT_CALL(*tokenizer, TokenIdsToText(::testing::_, ::testing::_))
      .WillOnce(testing::Return("▁Hello▁World!"))
      .WillOnce(testing::Return("▁How's▁it▁going?"));

  const std::vector<std::vector<int>> ids = {{90, 547, 58, 735, 210, 466, 2294},
                                             {224, 24, 8, 66, 246, 18, 2295}};

  auto texts = tokenizer->TokenIdsToTexts(/*batch_size=*/2, ids);
  EXPECT_TRUE(texts.ok());
  EXPECT_EQ(texts.value().size(), 2);
  EXPECT_EQ(texts.value()[0].value(), "▁Hello▁World!");
  EXPECT_EQ(texts.value()[1].value(), "▁How's▁it▁going?");
}

TEST(TokenizerTest, TokenIdsToTextsWithIncompleteBPESequence) {
  auto tokenizer = std::make_unique<MockTokenizer>();
  EXPECT_CALL(*tokenizer, TokenIdsToText(::testing::_, ::testing::_))
      .WillOnce(testing::Return(absl::DataLossError("Incomplete BPE sequence")))
      .WillOnce(testing::Return("▁How's▁it▁going?"));

  const std::vector<std::vector<int>> ids = {{90, 547, 58, 735, 210, 466, 2294},
                                             {224, 24, 8, 66, 246, 18, 2295}};

  auto texts = tokenizer->TokenIdsToTexts(/*batch_size=*/2, ids);
  EXPECT_TRUE(texts.ok());
  EXPECT_EQ(texts.value().size(), 2);
  EXPECT_EQ(texts.value()[0].status().code(), absl::StatusCode::kDataLoss);
  EXPECT_EQ(texts.value()[1].value(), "▁How's▁it▁going?");
}

TEST(TokenizerTest, TokenToId) {
  auto tokenizer = std::make_unique<MockTokenizer>();
  EXPECT_CALL(*tokenizer, TokenToId("X")).WillOnce(testing::Return(123));
  EXPECT_EQ(tokenizer->TokenToId("X").value(), 123);
}

TEST(TokenizerTest, MergeTokenIds) {
  const std::vector<std::vector<int>> previous_ids = {{90, 547, 58, 735},
                                                      {224, 24}};
  const std::vector<std::vector<int>> current_ids = {{210, 466, 2294},
                                                     {8, 66, 246, 18, 2295}};
  auto merged = Tokenizer::MergeTokenIds(previous_ids, current_ids);
  EXPECT_TRUE(merged.ok());
  EXPECT_EQ(merged->size(), 2);
  EXPECT_EQ((*merged)[0], std::vector<int>({90, 547, 58, 735, 210, 466, 2294}));
  EXPECT_EQ((*merged)[1], std::vector<int>({224, 24, 8, 66, 246, 18, 2295}));
}

TEST(TokenizerTest, GetTokens) {
  auto tokenizer = std::make_unique<MockTokenizer>();
  EXPECT_CALL(*tokenizer, GetTokens())
      .WillRepeatedly(
          testing::Return(std::vector<std::string>{"Hello", "World!"}));
  EXPECT_EQ(tokenizer->GetTokens().size(), 2);
  EXPECT_EQ(tokenizer->GetTokens()[0], "Hello");
  EXPECT_EQ(tokenizer->GetTokens()[1], "World!");
}

TEST(TokenizerTest, GetVocabSize) {
  auto tokenizer = std::make_unique<MockTokenizer>();
  EXPECT_CALL(*tokenizer, GetVocabSize()).WillRepeatedly(testing::Return(100));
  EXPECT_EQ(tokenizer->GetVocabSize(), 100);
}

}  // namespace
}  // namespace litert::support
