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
// C++ port of `#[cfg(test)] mod tests` in `src/tokenizer/mod.rs` from the
// HuggingFace `tokenizers` crate v0.23.2
// (https://github.com/huggingface/tokenizers, Apache License 2.0,
// Copyright 2020 The HuggingFace Team). Kept line-by-line close to the Rust
// original on purpose; see
// support/tokenizer/tokenizers/README.md.

#include "support/tokenizer/tokenizers/tokenizer/tokenizer.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/models/wordlevel/wordlevel.h"
#include "support/tokenizer/tokenizers/pre_tokenizers/whitespace.h"
#include "support/tokenizer/tokenizers/tokenizer/added_vocabulary.h"
#include "support/tokenizer/tokenizers/tokenizer/encoding.h"
#include "support/tokenizer/tokenizers/tokenizer/truncation.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace tokenizers {
namespace {

using ::tokenizers::models::wordlevel::WordLevel;
using ::tokenizers::models::wordlevel::WordLevelBuilder;
using ::tokenizers::pre_tokenizers::WhitespaceSplit;

// Build a tokenizer with a known vocabulary: "a"=0, "b"=1, ... "j"=9,
// "<unk>"=10. Uses WhitespaceSplit so each space-separated word maps to one
// token.
Tokenizer TestTokenizer() {
  absl::flat_hash_map<std::string, uint32_t> vocab;
  for (char c = 'a'; c <= 'j'; ++c) {
    vocab.emplace(std::string(1, c), static_cast<uint32_t>(c - 'a'));
  }
  vocab.emplace("<unk>", 10);

  WordLevel model = WordLevelBuilder()
                        .Vocab(std::move(vocab))
                        .UnkToken("<unk>")
                        .Build()
                        .value();

  Tokenizer tokenizer(std::move(model));
  tokenizer.WithPreTokenizer(WhitespaceSplit());
  return tokenizer;
}

TEST(TokenizerTest, RightTruncationEarlyExitMatchesFullEncode) {
  // "a b c d e f g h i j" → 10 tokens [0,1,2,3,4,5,6,7,8,9]
  // Right truncation to 3 → [0,1,2]
  const char* input = "a b c d e f g h i j";

  Tokenizer tok = TestTokenizer();
  ASSERT_OK(tok.WithTruncation(TruncationParams{
      /*direction=*/TruncationDirection::kRight,
      /*max_length=*/3,
      /*strategy=*/TruncationStrategy::kLongestFirst,
      /*stride=*/0,
  }));
  ASSERT_OK_AND_ASSIGN(Encoding truncated, tok.Encode(input, false));

  Tokenizer tok_full = TestTokenizer();
  ASSERT_OK_AND_ASSIGN(Encoding full, tok_full.Encode(input, false));

  ASSERT_EQ(truncated.GetIds().size(), 3u);
  EXPECT_EQ(
      truncated.GetIds(),
      (std::vector<uint32_t>(full.GetIds().begin(), full.GetIds().begin() + 3)))
      << "Right-truncated should match first 3 tokens of full encoding";
}

TEST(TokenizerTest, LeftTruncationKeepsTailTokens) {
  // "a b c d e f g h i j" → 10 tokens [0,1,2,3,4,5,6,7,8,9]
  // Left truncation to 3 → [7,8,9]
  const char* input = "a b c d e f g h i j";

  Tokenizer tok = TestTokenizer();
  ASSERT_OK(tok.WithTruncation(TruncationParams{
      /*direction=*/TruncationDirection::kLeft,
      /*max_length=*/3,
      /*strategy=*/TruncationStrategy::kLongestFirst,
      /*stride=*/0,
  }));
  ASSERT_OK_AND_ASSIGN(Encoding truncated, tok.Encode(input, false));

  Tokenizer tok_full = TestTokenizer();
  ASSERT_OK_AND_ASSIGN(Encoding full, tok_full.Encode(input, false));
  size_t n = full.GetIds().size();

  ASSERT_EQ(truncated.GetIds().size(), 3u);
  EXPECT_EQ(truncated.GetIds(),
            (std::vector<uint32_t>(full.GetIds().begin() + (n - 3),
                                   full.GetIds().end())))
      << "Left-truncated should match last 3 tokens of full encoding";
}

TEST(TokenizerTest, PairRightTruncationLongestFirst) {
  // Seq A: "a b c d e f g h i j" → 10 tokens
  // Seq B: "a b c d e"           → 5 tokens
  // Total: 15, max_length: 6, LongestFirst → should trim A more than B
  const char* seq_a = "a b c d e f g h i j";
  const char* seq_b = "a b c d e";

  Tokenizer tok = TestTokenizer();
  ASSERT_OK(tok.WithTruncation(TruncationParams{
      /*direction=*/TruncationDirection::kRight,
      /*max_length=*/6,
      /*strategy=*/TruncationStrategy::kLongestFirst,
      /*stride=*/0,
  }));
  ASSERT_OK_AND_ASSIGN(Encoding truncated,
                       tok.Encode(EncodeInput(seq_a, seq_b), false));

  ASSERT_EQ(truncated.GetIds().size(), 6u)
      << "Pair encoding should have exactly max_length tokens";
  // First N tokens should match the start of the full encoding
  EXPECT_EQ(truncated.GetIds(), (std::vector<uint32_t>{0, 1, 2, 0, 1, 2}))
      << "Truncated pair should match prefix of full pair encoding";
}

TEST(TokenizerTest, PairOnlySecondDoesNotTruncateFirst) {
  // Seq A: "a b c d e" → 5 tokens
  // Seq B: "a b c d e f g h i j" → 10 tokens
  // max_length: 8, OnlySecond → A stays at 5, B truncated to 3
  const char* seq_a = "a b c d e";
  const char* seq_b = "a b c d e f g h i j";

  Tokenizer tok = TestTokenizer();
  ASSERT_OK(tok.WithTruncation(TruncationParams{
      /*direction=*/TruncationDirection::kRight,
      /*max_length=*/8,
      /*strategy=*/TruncationStrategy::kOnlySecond,
      /*stride=*/0,
  }));
  ASSERT_OK_AND_ASSIGN(Encoding truncated,
                       tok.Encode(EncodeInput(seq_a, seq_b), false));

  ASSERT_EQ(truncated.GetIds().size(), 8u)
      << "Pair encoding should have exactly max_length tokens";

  // First sequence should be fully preserved (5 tokens)
  Tokenizer tok_full = TestTokenizer();
  ASSERT_OK_AND_ASSIGN(Encoding full_a, tok_full.Encode(seq_a, false));
  EXPECT_EQ((std::vector<uint32_t>(truncated.GetIds().begin(),
                                   truncated.GetIds().begin() + 5)),
            full_a.GetIds())
      << "OnlySecond should not truncate the first sequence";
}

TEST(TokenizerTest, PairOnlyFirstDoesNotTruncateSecond) {
  // Seq A: "a b c d e f g h i j" → 10 tokens
  // Seq B: "a b c d e" → 5 tokens
  // max_length: 8, OnlyFirst → B stays at 5, A truncated to 3
  const char* seq_a = "a b c d e f g h i j";
  const char* seq_b = "a b c d e";

  Tokenizer tok = TestTokenizer();
  ASSERT_OK(tok.WithTruncation(TruncationParams{
      /*direction=*/TruncationDirection::kRight,
      /*max_length=*/8,
      /*strategy=*/TruncationStrategy::kOnlyFirst,
      /*stride=*/0,
  }));
  ASSERT_OK_AND_ASSIGN(Encoding truncated,
                       tok.Encode(EncodeInput(seq_a, seq_b), false));

  ASSERT_EQ(truncated.GetIds().size(), 8u)
      << "Pair encoding should have exactly max_length tokens";

  // Second sequence should be fully preserved (last 5 tokens)
  Tokenizer tok_full = TestTokenizer();
  ASSERT_OK_AND_ASSIGN(Encoding full_b, tok_full.Encode(seq_b, false));
  const std::vector<uint32_t>& ids = truncated.GetIds();
  EXPECT_EQ((std::vector<uint32_t>(ids.end() - 5, ids.end())), full_b.GetIds())
      << "OnlyFirst should not truncate the second sequence";
}

TEST(TokenizerTest, GetVocabSizeWithOverlappingAddedTokens) {
  // TestTokenizer() builds a WordLevel model with "a"-"j" + "<unk>" = 11 tokens
  Tokenizer tok = TestTokenizer();

  EXPECT_EQ(tok.GetVocabSize(false), 11u);
  EXPECT_EQ(tok.GetVocabSize(true), 11u);

  // Add "a" as a special token — it already exists in the model vocab (overlap)
  ASSERT_OK(tok.AddSpecialTokens({AddedToken::From("a", true)}).status());
  EXPECT_EQ(tok.GetVocabSize(false), 11u);
  EXPECT_EQ(tok.GetVocabSize(true), 11u);  // must not double-count

  // Add a genuinely new token
  ASSERT_OK(tok.AddTokens({AddedToken::From("new_token", false)}).status());
  EXPECT_EQ(tok.GetVocabSize(false), 11u);
  EXPECT_EQ(tok.GetVocabSize(true), 12u);
}

}  // namespace
}  // namespace tokenizers
