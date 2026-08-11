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

#include "support/tokenizer/sentencepiece_tokenizer.h"

#include <fcntl.h>

#include <cstdint>
#include <cstdio>
#include <filesystem>  // NOLINT: Required for path manipulation.
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "support/tokenizer/tokenizer.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace litert::support {
namespace {

using ::testing::status::IsOkAndHolds;
using ::testing::status::StatusIs;

constexpr char kTestdataDir[] =
    "litert_lm/support/tokenizer/testdata/";

std::string GetSentencePieceModelPath() {
  return (std::filesystem::path(::testing::SrcDir()) / kTestdataDir /
          "sentencepiece.model")
      .string();
}

std::string GetGemma3TokenizerModelPath() {
  return (std::filesystem::path(::testing::SrcDir()) / kTestdataDir /
          "gemma3_sentencepiece.model")
      .string();
}

absl::StatusOr<std::string> GetContents(absl::string_view path) {
#ifdef _WIN32
  int fd = open(path.data(), O_RDONLY | O_BINARY);
#else
  int fd = open(path.data(), O_RDONLY);
#endif
  if (fd < 0) {
    return absl::NotFoundError(absl::StrCat("File not found: ", path));
  }

  absl::Cleanup fd_closer = [fd]() { close(fd); };  // Called on return.

  int64_t contents_length = lseek(fd, 0, SEEK_END);
  if (contents_length < 0) {
    return absl::InternalError(absl::StrCat("Failed to get length: ", path));
  }

  std::string contents(contents_length, '\0');
  lseek(fd, 0, SEEK_SET);
  char* contents_ptr = contents.data();
  while (contents_length > 0) {
    int read_bytes = read(fd, contents_ptr, contents_length);
    if (read_bytes < 0) {
      return absl::InternalError(absl::StrCat("Failed to read: ", path));
    } else if (read_bytes == 0) {
      return absl::InternalError(absl::StrCat("File is empty: ", path));
    }
    contents_ptr += read_bytes;
    contents_length -= read_bytes;
  }

  return std::move(contents);
}

TEST(SentencePieceTokenizerTest, CreateFromFile) {
  auto tokenizer_or =
      SentencePieceTokenizer::CreateFromFile(GetSentencePieceModelPath());
  EXPECT_TRUE(tokenizer_or.ok());
}

TEST(SentencePieceTokenizerTest, CreateFromBuffer) {
  ASSERT_OK_AND_ASSIGN(auto model_buffer,
                       GetContents(GetSentencePieceModelPath()));
  ASSERT_OK_AND_ASSIGN(auto tokenizer,
                       SentencePieceTokenizer::CreateFromBuffer(model_buffer));
}

TEST(SentencePieceTokenizerTest, Create) {
  auto tokenizer_or =
      SentencePieceTokenizer::CreateFromFile(GetSentencePieceModelPath());
  EXPECT_TRUE(tokenizer_or.ok());
}

TEST(SentencePieceTokenizerTest, GetTokenizerType) {
  ASSERT_OK_AND_ASSIGN(auto tokenizer, SentencePieceTokenizer::CreateFromFile(
                                           GetSentencePieceModelPath()));
  EXPECT_EQ(tokenizer->GetTokenizerType(), TokenizerType::kSentencePiece);
}

TEST(SentencePieceTokenizerTest, TextToTokenIds) {
  ASSERT_OK_AND_ASSIGN(auto tokenizer, SentencePieceTokenizer::CreateFromFile(
                                           GetSentencePieceModelPath()));

  absl::string_view text = "How's it going?";
  ASSERT_OK_AND_ASSIGN(auto ids, tokenizer->TextToTokenIds(text));

  EXPECT_THAT(ids, ::testing::ElementsAre(224, 24, 8, 66, 246, 18, 2295));
}

TEST(SentencePieceTokenizerTest, TokenToId) {
  ASSERT_OK_AND_ASSIGN(auto tokenizer, SentencePieceTokenizer::CreateFromFile(
                                           GetSentencePieceModelPath()));
  EXPECT_THAT(tokenizer->TokenToId("X"), IsOkAndHolds(882));
}

TEST(SentencePieceTokenizerTest, TokenToIdUnknownTokenReturnsError) {
  ASSERT_OK_AND_ASSIGN(auto tokenizer, SentencePieceTokenizer::CreateFromFile(
                                           GetSentencePieceModelPath()));
  EXPECT_THAT(tokenizer->TokenToId("unknown_token"),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST(SentencePieceTokenizerTest, TokenIdsToText) {
  ASSERT_OK_AND_ASSIGN(auto tokenizer, SentencePieceTokenizer::CreateFromFile(
                                           GetSentencePieceModelPath()));

  const std::vector<int> ids = {90, 547, 58, 735, 210, 466, 2294};
  ASSERT_OK_AND_ASSIGN(auto text, tokenizer->TokenIdsToText(ids));
  EXPECT_EQ(text, "Hello World!");
}

TEST(SentencePieceTokenizerTest, TokenIdsToTextSkipSpecialTokens) {
  ASSERT_OK_AND_ASSIGN(auto tokenizer, SentencePieceTokenizer::CreateFromFile(
                                           GetSentencePieceModelPath()));

  // 1 is <s>, 2 is </s>, 0 is <unk>
  const std::vector<int> ids = {1, 90, 547, 58, 735, 210, 466, 2294, 2};
  EXPECT_THAT(tokenizer->TokenIdsToText(ids, /*skip_special_tokens=*/true),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(SentencePieceTokenizerTest, TokenIdsToTextConsecutiveByteTokens) {
  ASSERT_OK_AND_ASSIGN(auto tokenizer, SentencePieceTokenizer::CreateFromFile(
                                           GetGemma3TokenizerModelPath()));

  std::vector<std::string> tokens = tokenizer->GetTokens();
  EXPECT_EQ(tokens.size(), 262144);

  // Consecutive byte token combination
  // <0xC2><0xB0> --> °
  EXPECT_EQ(tokens[432], "<0xC2>");
  EXPECT_EQ(tokens[414], "<0xB0>");

  // Pass to tokenizer in two separate calls to TokenIdsToText.
  ASSERT_OK_AND_ASSIGN(auto chunk_text, tokenizer->TokenIdsToText({432, 414}));
  EXPECT_EQ(chunk_text, "°");
}

TEST(SentencePieceTokenizerTest, TokenIdsToTextMixedConsecutiveByteTokens) {
  ASSERT_OK_AND_ASSIGN(auto tokenizer, SentencePieceTokenizer::CreateFromFile(
                                           GetGemma3TokenizerModelPath()));

  std::vector<std::string> tokens = tokenizer->GetTokens();
  EXPECT_EQ(tokens.size(), 262144);

  // Consecutive byte token combination
  // <0x6B><0x6D><0xC2><0xB2> --> km²
  EXPECT_EQ(tokens[345], "<0x6B>");
  EXPECT_EQ(tokens[347], "<0x6D>");
  EXPECT_EQ(tokens[432], "<0xC2>");
  EXPECT_EQ(tokens[416], "<0xB2>");

  auto chunk_token_ids = {345, 347, 432, 416};
  ASSERT_OK_AND_ASSIGN(auto text, tokenizer->TokenIdsToText(chunk_token_ids));
  EXPECT_EQ(text, "km²");
}

TEST(SentencePieceTokenizerTest, TokenIdsToTextMixedTokens) {
  ASSERT_OK_AND_ASSIGN(auto tokenizer, SentencePieceTokenizer::CreateFromFile(
                                           GetGemma3TokenizerModelPath()));

  const std::vector<int> ids = {345, 347, 432, 416, 964};
  ASSERT_OK_AND_ASSIGN(auto text, tokenizer->TokenIdsToText(ids));
  EXPECT_EQ(text, "km² were");
}

TEST(SentencePieceTokenizerTest,
     TokenIdsToTextStartByteReturnsReplacementCharacter) {
  ASSERT_OK_AND_ASSIGN(auto tokenizer, SentencePieceTokenizer::CreateFromFile(
                                           GetGemma3TokenizerModelPath()));

  // <0xC2> (432) is a start byte, invalid on its own.
  auto text_or = tokenizer->TokenIdsToText({432});
  ASSERT_OK(text_or);
  EXPECT_EQ(*text_or, "\xef\xbf\xbd");
}

TEST(SentencePieceTokenizerTest,
     TokenIdsToTextMixedTokensReturnsReplacementCharacter) {
  ASSERT_OK_AND_ASSIGN(auto tokenizer, SentencePieceTokenizer::CreateFromFile(
                                           GetGemma3TokenizerModelPath()));

  const std::vector<int> ids = {964, 345, 347, 432};  // miss byte token 416
  auto text_or = tokenizer->TokenIdsToText(ids);
  ASSERT_OK(text_or);
  EXPECT_EQ(*text_or, " werekm\xef\xbf\xbd");
}

TEST(SentencePieceTokenizerTest, GetTokens) {
  ASSERT_OK_AND_ASSIGN(auto tokenizer, SentencePieceTokenizer::CreateFromFile(
                                           GetSentencePieceModelPath()));
  std::vector<std::string> tokens = tokenizer->GetTokens();
  EXPECT_EQ(tokens.size(), 4000);

  // Verify 5 different tokens.
  EXPECT_EQ(tokens[0], "<unk>");
  EXPECT_EQ(tokens[1], "<s>");
  EXPECT_EQ(tokens[2], "</s>");
  EXPECT_EQ(tokens[224], "▁How");
  EXPECT_EQ(tokens[2295], "?");
}

TEST(SentencePieceTokenizerTest, TokensTokenIdsToTextOutOfRange) {
  ASSERT_OK_AND_ASSIGN(auto tokenizer, SentencePieceTokenizer::CreateFromFile(
                                           GetSentencePieceModelPath()));
  std::vector<std::string> tokens = tokenizer->GetTokens();
  EXPECT_THAT(tokenizer->TokenIdsToText({10000}),
              StatusIs(absl::StatusCode::kNotFound));
}

}  // namespace
}  // namespace litert::support
