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
// C++ port of the `#[cfg(test)]` module of `src/tokenizer/encoding.rs` from
// the HuggingFace `tokenizers` crate v0.23.2
// (https://github.com/huggingface/tokenizers, Apache License 2.0,
// Copyright 2020 The HuggingFace Team). Kept line-by-line close to the Rust
// original on purpose; see
// support/tokenizer/tokenizers/README.md.

#include "support/tokenizer/tokenizers/tokenizer/encoding.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/tokenizer/token.h"
#include "support/tokenizer/tokenizers/utils/json.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace tokenizers {
namespace {

TEST(EncodingTest, MergeEncodings) {
  Encoding a(
      /*ids=*/{1},
      /*type_ids=*/{0},
      /*tokens=*/{"Hello "},
      /*words=*/{0},
      /*offsets=*/{{0, 6}},
      /*special_tokens_mask=*/{0},
      /*attention_mask=*/{1},
      /*overflowing=*/{},
      /*sequence_ranges=*/{});
  Encoding b(
      /*ids=*/{2},
      /*type_ids=*/{1},
      /*tokens=*/{"World!"},
      /*words=*/{0},
      /*offsets=*/{{0, 6}},
      /*special_tokens_mask=*/{0},
      /*attention_mask=*/{1},
      /*overflowing=*/{},
      /*sequence_ranges=*/{});
  a.MergeWith(b, true);

  Encoding expected(
      /*ids=*/{1, 2},
      /*type_ids=*/{0, 1},
      /*tokens=*/{"Hello ", "World!"},
      /*words=*/{0, 0},
      /*offsets=*/{{0, 6}, {6, 12}},
      /*special_tokens_mask=*/{0, 0},
      /*attention_mask=*/{1, 1},
      /*overflowing=*/{},
      /*sequence_ranges=*/{});
  EXPECT_EQ(a, expected);
}

TEST(EncodingTest, Truncate) {
  Encoding a(
      /*ids=*/{1, 2, 3},
      /*type_ids=*/{0, 0, 0},
      /*tokens=*/{"Hello", "World", "!"},
      /*words=*/{0, 1, 2},
      /*offsets=*/{{0, 5}, {6, 11}, {11, 12}},
      /*special_tokens_mask=*/{0, 0, 0},
      /*attention_mask=*/{1, 1, 1},
      /*overflowing=*/{},
      /*sequence_ranges=*/{});
  ASSERT_OK(a.Truncate(2, 0, TruncationDirection::kRight));

  Encoding expected(
      /*ids=*/{1, 2},
      /*type_ids=*/{0, 0},
      /*tokens=*/{"Hello", "World"},
      /*words=*/{0, 1},
      /*offsets=*/{{0, 5}, {6, 11}},
      /*special_tokens_mask=*/{0, 0},
      /*attention_mask=*/{1, 1},
      /*overflowing=*/
      {Encoding(
          /*ids=*/{3},
          /*type_ids=*/{0},
          /*tokens=*/{"!"},
          /*words=*/{2},
          /*offsets=*/{{11, 12}},
          /*special_tokens_mask=*/{0},
          /*attention_mask=*/{1},
          /*overflowing=*/{},
          /*sequence_ranges=*/{})},
      /*sequence_ranges=*/{});
  EXPECT_EQ(a, expected);
}

TEST(EncodingTest, TruncateToEmpty) {
  Encoding a(
      /*ids=*/{1, 2, 3},
      /*type_ids=*/{0, 0, 0},
      /*tokens=*/{"Hello", "World", "!"},
      /*words=*/{0, 1, 2},
      /*offsets=*/{{0, 5}, {6, 11}, {11, 12}},
      /*special_tokens_mask=*/{0, 0, 0},
      /*attention_mask=*/{1, 1, 1},
      /*overflowing=*/{},
      /*sequence_ranges=*/{});
  ASSERT_OK(a.Truncate(0, 0, TruncationDirection::kRight));

  Encoding expected(
      /*ids=*/{},
      /*type_ids=*/{},
      /*tokens=*/{},
      /*words=*/{},
      /*offsets=*/{},
      /*special_tokens_mask=*/{},
      /*attention_mask=*/{},
      /*overflowing=*/
      {Encoding(
          /*ids=*/{1, 2, 3},
          /*type_ids=*/{0, 0, 0},
          /*tokens=*/{"Hello", "World", "!"},
          /*words=*/{0, 1, 2},
          /*offsets=*/{{0, 5}, {6, 11}, {11, 12}},
          /*special_tokens_mask=*/{0, 0, 0},
          /*attention_mask=*/{1, 1, 1},
          /*overflowing=*/{},
          /*sequence_ranges=*/{})},
      /*sequence_ranges=*/{});
  EXPECT_EQ(a, expected);
}

TEST(EncodingTest, TruncateOverflowWithStride) {
  Encoding enc(
      /*ids=*/{1, 2, 3, 4, 5},
      /*type_ids=*/{0, 0, 0, 0, 0},
      /*tokens=*/{"42", "is", "the", "answer", "!"},
      /*words=*/{0, 1, 2, 3, 4},
      /*offsets=*/{{0, 2}, {2, 4}, {4, 7}, {7, 13}, {13, 14}},
      /*special_tokens_mask=*/{0, 0, 0, 0, 0},
      /*attention_mask=*/{1, 1, 1, 1, 1},
      /*overflowing=*/{},
      /*sequence_ranges=*/{});
  ASSERT_OK(enc.Truncate(4, 2, TruncationDirection::kRight));

  Encoding expected(
      /*ids=*/{1, 2, 3, 4},
      /*type_ids=*/{0, 0, 0, 0},
      /*tokens=*/{"42", "is", "the", "answer"},
      /*words=*/{0, 1, 2, 3},
      /*offsets=*/{{0, 2}, {2, 4}, {4, 7}, {7, 13}},
      /*special_tokens_mask=*/{0, 0, 0, 0},
      /*attention_mask=*/{1, 1, 1, 1},
      /*overflowing=*/
      {Encoding(
          /*ids=*/{3, 4, 5},
          /*type_ids=*/{0, 0, 0},
          /*tokens=*/{"the", "answer", "!"},
          /*words=*/{2, 3, 4},
          /*offsets=*/{{4, 7}, {7, 13}, {13, 14}},
          /*special_tokens_mask=*/{0, 0, 0},
          /*attention_mask=*/{1, 1, 1},
          /*overflowing=*/{},
          /*sequence_ranges=*/{})},
      /*sequence_ranges=*/{});
  EXPECT_EQ(enc, expected);
}

TEST(EncodingTest, TruncateLeft) {
  Encoding a(
      /*ids=*/{1, 2, 3},
      /*type_ids=*/{0, 0, 0},
      /*tokens=*/{"Hello", "World", "!"},
      /*words=*/{0, 1, 2},
      /*offsets=*/{{0, 5}, {6, 11}, {11, 12}},
      /*special_tokens_mask=*/{0, 0, 0},
      /*attention_mask=*/{1, 1, 1},
      /*overflowing=*/{},
      /*sequence_ranges=*/{});
  ASSERT_OK(a.Truncate(2, 0, TruncationDirection::kLeft));

  Encoding expected(
      /*ids=*/{2, 3},
      /*type_ids=*/{0, 0},
      /*tokens=*/{"World", "!"},
      /*words=*/{1, 2},
      /*offsets=*/{{6, 11}, {11, 12}},
      /*special_tokens_mask=*/{0, 0},
      /*attention_mask=*/{1, 1},
      /*overflowing=*/
      {Encoding(
          /*ids=*/{1},
          /*type_ids=*/{0},
          /*tokens=*/{"Hello"},
          /*words=*/{0},
          /*offsets=*/{{0, 5}},
          /*special_tokens_mask=*/{0},
          /*attention_mask=*/{1},
          /*overflowing=*/{},
          /*sequence_ranges=*/{})},
      /*sequence_ranges=*/{});
  EXPECT_EQ(a, expected);
}

TEST(EncodingTest, Mappings) {
  Encoding encoding(
      /*ids=*/std::vector<uint32_t>(11, 0),  // Needed for Encoding::len
      /*type_ids=*/{},
      /*tokens=*/
      {
          // First sequence:
          "He",
          "llo",
          "won",
          "der",
          "ful",
          "friend",
          "!",
          // Second sequence:
          "How",
          "are",
          "you",
          "?",
      },
      /*words=*/
      {
          // First sequence:
          0,
          0,
          1,
          1,
          1,
          2,
          3,
          // Second sequence:
          0,
          1,
          2,
          3,
      },
      /*offsets=*/
      {
          // First sequence:
          {0, 2},
          {2, 5},
          {7, 10},
          {10, 13},
          {13, 16},
          {17, 23},
          {23, 24},
          // Second sequence:
          {0, 3},
          {4, 7},
          {8, 11},
          {11, 12},
      },
      /*special_tokens_mask=*/{},
      /*attention_mask=*/{},
      /*overflowing=*/{},
      /*sequence_ranges=*/{{0, {0, 7}}, {1, {7, 11}}});

  EXPECT_EQ(encoding.WordToTokens(0, 0), (std::pair<size_t, size_t>(0, 2)));
  EXPECT_EQ(encoding.WordToTokens(1, 0), (std::pair<size_t, size_t>(2, 5)));
  EXPECT_EQ(encoding.WordToTokens(2, 0), (std::pair<size_t, size_t>(5, 6)));
  EXPECT_EQ(encoding.WordToTokens(3, 0), (std::pair<size_t, size_t>(6, 7)));
  EXPECT_EQ(encoding.WordToTokens(0, 1), (std::pair<size_t, size_t>(7, 8)));
  EXPECT_EQ(encoding.WordToTokens(1, 1), (std::pair<size_t, size_t>(8, 9)));
  EXPECT_EQ(encoding.WordToTokens(2, 1), (std::pair<size_t, size_t>(9, 10)));
  EXPECT_EQ(encoding.WordToTokens(3, 1), (std::pair<size_t, size_t>(10, 11)));

  EXPECT_EQ(encoding.WordToChars(0, 0), (Offsets{0, 5}));
  EXPECT_EQ(encoding.WordToChars(1, 0), (Offsets{7, 16}));
  EXPECT_EQ(encoding.WordToChars(0, 1), (Offsets{0, 3}));
  EXPECT_EQ(encoding.WordToChars(1, 1), (Offsets{4, 7}));

  EXPECT_EQ(encoding.TokenToChars(0), (std::pair<size_t, Offsets>(0, {0, 2})));
  EXPECT_EQ(encoding.TokenToChars(1), (std::pair<size_t, Offsets>(0, {2, 5})));
  EXPECT_EQ(encoding.TokenToChars(7), (std::pair<size_t, Offsets>(1, {0, 3})));
  EXPECT_EQ(encoding.TokenToChars(9), (std::pair<size_t, Offsets>(1, {8, 11})));

  EXPECT_EQ(encoding.TokenToWord(1), (std::pair<size_t, uint32_t>(0, 0)));
  EXPECT_EQ(encoding.TokenToWord(2), (std::pair<size_t, uint32_t>(0, 1)));
  EXPECT_EQ(encoding.TokenToWord(7), (std::pair<size_t, uint32_t>(1, 0)));
  EXPECT_EQ(encoding.TokenToWord(9), (std::pair<size_t, uint32_t>(1, 2)));
  EXPECT_EQ(encoding.TokenToWord(11), std::nullopt);

  EXPECT_EQ(encoding.CharToToken(3, 0), 1u);
  EXPECT_EQ(encoding.CharToToken(8, 0), 2u);
  EXPECT_EQ(encoding.CharToToken(16, 0), std::nullopt);
  EXPECT_EQ(encoding.CharToToken(23, 0), 6u);
  EXPECT_EQ(encoding.CharToToken(2, 1), 7u);
  EXPECT_EQ(encoding.CharToToken(9, 1), 9u);

  EXPECT_EQ(encoding.CharToWord(3, 0), 0u);
  EXPECT_EQ(encoding.CharToWord(8, 0), 1u);
  EXPECT_EQ(encoding.CharToWord(16, 0), std::nullopt);
  EXPECT_EQ(encoding.CharToWord(23, 0), 3u);
  EXPECT_EQ(encoding.CharToWord(2, 1), 0u);
  EXPECT_EQ(encoding.CharToWord(9, 1), 2u);
}

TEST(EncodingTest, Padding) {
  Encoding a(
      /*ids=*/{1},
      /*type_ids=*/{0},
      /*tokens=*/{"Hello "},
      /*words=*/{0},
      /*offsets=*/{{0, 6}},
      /*special_tokens_mask=*/{0},
      /*attention_mask=*/{1},
      /*overflowing=*/{},
      /*sequence_ranges=*/{{0, {0, 1}}});
  const size_t target_length = 2;
  const uint32_t pad_id = 99;
  const uint32_t pad_type_id = 0;
  const char* pad_token = "[PAD]";
  a.Pad(target_length, pad_id, pad_type_id, pad_token, PaddingDirection::kLeft);
  EXPECT_EQ(a.GetSequenceRanges(),
            (absl::flat_hash_map<size_t, SequenceRange>{{0, {1, 2}}}));
}

}  // namespace
}  // namespace tokenizers
