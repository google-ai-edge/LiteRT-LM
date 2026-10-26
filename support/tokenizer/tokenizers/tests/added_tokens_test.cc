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
// C++ port of `tests/added_tokens.rs` from the HuggingFace `tokenizers` crate
// v0.23.2 (https://github.com/huggingface/tokenizers, Apache License 2.0,
// Copyright 2020 The HuggingFace Team). Kept line-by-line close to the Rust
// original on purpose; see
// support/tokenizer/tokenizers/README.md.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "support/tokenizer/tokenizers/tests/common.h"
#include "support/tokenizer/tokenizers/tokenizer/added_vocabulary.h"
#include "support/tokenizer/tokenizers/tokenizer/encoding.h"
#include "support/tokenizer/tokenizers/tokenizer/token.h"
#include "support/tokenizer/tokenizers/tokenizer/tokenizer.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace tokenizers {
namespace {

using ::tokenizers::test_common::GetByteLevel;
using ::tokenizers::test_common::GetEmpty;

TEST(AddedTokensTest, AddTokens) {
  Tokenizer tokenizer = GetEmpty();

  ASSERT_OK_AND_ASSIGN(size_t added_special,
                       tokenizer.AddSpecialTokens({
                           AddedToken::From("<cls>", true),
                           AddedToken::From("<sep>", true),
                       }));
  EXPECT_EQ(added_special, 2u);
  EXPECT_EQ(tokenizer.TokenToId("<cls>"), std::optional<uint32_t>(0));
  EXPECT_EQ(tokenizer.TokenToId("<sep>"), std::optional<uint32_t>(1));

  ASSERT_OK_AND_ASSIGN(size_t added, tokenizer.AddTokens({
                                         AddedToken::From("hello", false),
                                         AddedToken::From("world", false),
                                     }));
  EXPECT_EQ(added, 2u);
  EXPECT_EQ(tokenizer.TokenToId("hello"), std::optional<uint32_t>(2));
  EXPECT_EQ(tokenizer.TokenToId("world"), std::optional<uint32_t>(3));
}

TEST(AddedTokensTest, LstripTokens) {
  Tokenizer tokenizer = GetByteLevel(true, false);
  ASSERT_OK(
      tokenizer
          .AddSpecialTokens({AddedToken::From("<mask>", true).Lstrip(true)})
          .status());

  const char* input = "I saw a <mask> 😺";
  ASSERT_OK_AND_ASSIGN(Encoding output, tokenizer.Encode(input, false));

  EXPECT_EQ(
      output.GetTokens(),
      (std::vector<std::string>{"ĠI", "Ġsaw", "Ġa", " <mask>", "ĠðŁĺ", "º"}));
  EXPECT_EQ(output.GetOffsets(),
            (std::vector<Offsets>{
                {0, 1}, {1, 5}, {5, 7}, {7, 14}, {14, 19}, {15, 19}}));
}

TEST(AddedTokensTest, RstripTokens) {
  Tokenizer tokenizer = GetByteLevel(false, false);
  ASSERT_OK(
      tokenizer
          .AddSpecialTokens({AddedToken::From("<mask>", true).Rstrip(true)})
          .status());

  const char* input = "I saw a <mask> 😺";
  ASSERT_OK_AND_ASSIGN(Encoding output, tokenizer.Encode(input, false));

  EXPECT_EQ(output.GetTokens(),
            (std::vector<std::string>{"I", "Ġsaw", "Ġa", "Ġ", "<mask> ", "ðŁĺ",
                                      "º"}));

  // When `add_prefix_space = true` rstrip cannot work as a prefix space is
  // added to the next token
  Tokenizer tokenizer2 = GetByteLevel(true, false);
  ASSERT_OK(
      tokenizer2
          .AddSpecialTokens({AddedToken::From("<mask>", true).Rstrip(true)})
          .status());

  ASSERT_OK_AND_ASSIGN(Encoding output2, tokenizer2.Encode(input, false));

  EXPECT_EQ(output2.GetTokens(),
            (std::vector<std::string>{"ĠI", "Ġsaw", "Ġa", "Ġ", "<mask> ",
                                      "ĠðŁĺ", "º"}));
}

TEST(AddedTokensTest, SingleWordTokens) {
  // If `single_word = true` it shouldn't split `dancing`
  Tokenizer tokenizer = GetByteLevel(false, false);
  ASSERT_OK(
      tokenizer
          .AddSpecialTokens({AddedToken::From("ing", true).SingleWord(true)})
          .status());

  const char* input = "I like dancing";
  ASSERT_OK_AND_ASSIGN(Encoding output, tokenizer.Encode(input, false));

  EXPECT_EQ(output.GetTokens(),
            (std::vector<std::string>{"I", "Ġlike", "Ġdancing"}));

  // If `single_word = false` it should split `dancing`
  Tokenizer tokenizer2 = GetByteLevel(false, false);
  ASSERT_OK(
      tokenizer2
          .AddSpecialTokens({AddedToken::From("ing", true).SingleWord(false)})
          .status());

  ASSERT_OK_AND_ASSIGN(Encoding output2, tokenizer2.Encode(input, false));

  EXPECT_EQ(output2.GetTokens(),
            (std::vector<std::string>{"I", "Ġlike", "Ġd", "anc", "ing"}));
}

TEST(AddedTokensTest, OverlappingTokens) {
  Tokenizer tokenizer = GetByteLevel(false, false);

  ASSERT_OK(
      tokenizer.AddSpecialTokens({AddedToken::From("danc", true)}).status());
  ASSERT_OK(
      tokenizer.AddSpecialTokens({AddedToken::From("nci", true)}).status());
  ASSERT_OK(
      tokenizer.AddSpecialTokens({AddedToken::From("ing", true)}).status());

  const char* input = "I like dancing";
  ASSERT_OK_AND_ASSIGN(Encoding output, tokenizer.Encode(input, false));

  EXPECT_EQ(output.GetTokens(),
            (std::vector<std::string>{"I", "Ġlike", "Ġ", "danc", "ing"}));

  Tokenizer tokenizer2 = GetByteLevel(false, false);

  ASSERT_OK(
      tokenizer2.AddSpecialTokens({AddedToken::From("nci", true)}).status());
  ASSERT_OK(
      tokenizer2.AddSpecialTokens({AddedToken::From("danc", true)}).status());
  ASSERT_OK(
      tokenizer2.AddSpecialTokens({AddedToken::From("ing", true)}).status());
  ASSERT_OK(
      tokenizer2.AddSpecialTokens({AddedToken::From("ike", true)}).status());

  ASSERT_OK_AND_ASSIGN(Encoding output2, tokenizer2.Encode(input, false));

  // Breaking change but following `transformers` breaking change.
  // This behavior is deemed not used in practice:
  // https://github.com/huggingface/transformers/pull/13220
  // Order does NOT matter. (We could make it work again but the trie
  // would need to keep insertion order too)
  //
  // assert_eq!(output.get_tokens(), &["I", "Ġlike", "Ġda", "nci", "ng"]);
  EXPECT_EQ(output2.GetTokens(),
            (std::vector<std::string>{"I", "Ġl", "ike", "Ġ", "danc", "ing"}));
}

}  // namespace
}  // namespace tokenizers
