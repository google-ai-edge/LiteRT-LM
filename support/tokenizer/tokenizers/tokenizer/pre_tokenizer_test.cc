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
//
// Unit tests for `tokenizer/pre_tokenizer.{h,cc}` (C++ port of
// `src/tokenizer/pre_tokenizer.rs` from the HuggingFace `tokenizers` crate
// v0.23.2).

#include "support/tokenizer/tokenizers/tokenizer/pre_tokenizer.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/tokenizer/encoding.h"
#include "support/tokenizer/tokenizers/tokenizer/normalizer.h"
#include "support/tokenizer/tokenizers/tokenizer/token.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace tokenizers {
namespace {

TEST(PreTokenizerTest, SplitNormalizeTokenizeIntoEncoding) {
  PreTokenizedString pts = PreTokenizedString::From("Hello  world");

  ASSERT_OK(pts.Split([](size_t, NormalizedString s)
                          -> absl::StatusOr<std::vector<NormalizedString>> {
    return s.Split(U' ', SplitDelimiterBehavior::kRemoved);
  }));

  ASSERT_OK(pts.Normalize([](NormalizedString& s) {
    s.Lowercase();
    return absl::OkStatus();
  }));

  auto splits = pts.GetSplits(OffsetReferential::kOriginal, OffsetType::kByte);
  ASSERT_EQ(splits.size(), 2u);
  EXPECT_EQ(std::get<0>(splits[0]), "hello");
  EXPECT_EQ(std::get<1>(splits[0]), (Offsets{0, 5}));
  EXPECT_EQ(std::get<0>(splits[1]), "world");
  EXPECT_EQ(std::get<1>(splits[1]), (Offsets{7, 12}));

  uint32_t next_id = 1;
  ASSERT_OK(pts.Tokenize([&next_id](const NormalizedString& s)
                             -> absl::StatusOr<std::vector<Token>> {
    return std::vector<Token>{
        Token(next_id++, std::string(s.Get()), Offsets{0, s.Len()})};
  }));

  ASSERT_OK_AND_ASSIGN(Encoding enc, std::move(pts).IntoEncoding(
                                         std::nullopt, 0, OffsetType::kByte));
  EXPECT_EQ(enc.GetIds(), (std::vector<uint32_t>{1, 2}));
  EXPECT_EQ(enc.GetTokens(), (std::vector<std::string>{"hello", "world"}));
  EXPECT_EQ(enc.GetOffsets(), (std::vector<Offsets>{{0, 5}, {7, 12}}));
  EXPECT_EQ(enc.GetWordIds(), (std::vector<std::optional<uint32_t>>{0, 1}));
}

TEST(PreTokenizerTest, CharOffsetsAndTokenizeWithLimit) {
  PreTokenizedString pts = PreTokenizedString::From("é a b");
  ASSERT_OK(pts.Split([](size_t, NormalizedString s)
                          -> absl::StatusOr<std::vector<NormalizedString>> {
    return s.Split(U' ', SplitDelimiterBehavior::kRemoved);
  }));

  ASSERT_OK(pts.TokenizeWithLimit(
      [](const NormalizedString& s) -> absl::StatusOr<std::vector<Token>> {
        return std::vector<Token>{
            Token(1, std::string(s.Get()), Offsets{0, s.Len()})};
      },
      /*max_tokens=*/2, TruncationDirection::kRight));

  ASSERT_OK_AND_ASSIGN(Encoding enc, std::move(pts).IntoEncoding(
                                         std::nullopt, 0, OffsetType::kChar));
  EXPECT_EQ(enc.GetTokens(), (std::vector<std::string>{"é", "a"}));
  EXPECT_EQ(enc.GetOffsets(), (std::vector<Offsets>{{0, 1}, {2, 3}}));
}

}  // namespace
}  // namespace tokenizers
