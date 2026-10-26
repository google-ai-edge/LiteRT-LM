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
// C++ port of `src/models/wordlevel/mod.rs` (`#[cfg(test)] mod tests`) from the
// HuggingFace `tokenizers` crate v0.23.2
// (https://github.com/huggingface/tokenizers, Apache License 2.0,
// Copyright 2020 The HuggingFace Team). Kept line-by-line close to the Rust
// original on purpose; see
// support/tokenizer/tokenizers/README.md.

#include "support/tokenizer/tokenizers/models/wordlevel/wordlevel.h"

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/tokenizer/api.h"
#include "support/util/status_macros.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace tokenizers::models::wordlevel {
namespace {

TEST(WordLevelTest, TestTokenizeUnk) {
  const Vocab vocab = {{"<unk>", 0}, {"a", 1}, {"b", 2}};
  ASSERT_OK_AND_ASSIGN(
      WordLevel wordlevel,
      WordLevelBuilder().Vocab(vocab).UnkToken("<unk>").Build());
  ASSERT_OK_AND_ASSIGN(std::vector<Token> tokens, wordlevel.Tokenize("c"));
  EXPECT_EQ(tokens, (std::vector<Token>{Token{0u, "<unk>", {0, 1}}}));

  ASSERT_OK_AND_ASSIGN(tokens, wordlevel.Tokenize("a"));
  EXPECT_EQ(tokens, (std::vector<Token>{Token{1u, "a", {0, 1}}}));
}

TEST(WordLevelTest, TestTokenizeMissingUnkToken) {
  const Vocab vocab = {{"a", 0}, {"b", 1}};
  ASSERT_OK_AND_ASSIGN(WordLevel wordlevel,
                       WordLevelBuilder().Vocab(vocab).Build());
  ASSERT_OK_AND_ASSIGN(std::vector<Token> tokens, wordlevel.Tokenize("a"));
  EXPECT_EQ(tokens, (std::vector<Token>{Token{0u, "a", {0, 1}}}));

  const absl::StatusOr<std::vector<Token>> error = wordlevel.Tokenize("c");
  ASSERT_FALSE(error.ok());
  EXPECT_EQ(error.status(), Error::MissingUnkToken());
}

}  // namespace
}  // namespace tokenizers::models::wordlevel
