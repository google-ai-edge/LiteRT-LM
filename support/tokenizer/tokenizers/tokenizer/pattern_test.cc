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
// C++ port of the `#[cfg(test)]` module of `src/tokenizer/pattern.rs` from the
// HuggingFace `tokenizers` crate v0.23.2
// (https://github.com/huggingface/tokenizers, Apache License 2.0,
// Copyright 2020 The HuggingFace Team). Kept line-by-line close to the Rust
// original on purpose; see
// support/tokenizer/tokenizers/README.md.

#include "support/tokenizer/tokenizers/tokenizer/pattern.h"

#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "support/tokenizer/tokenizers/tokenizer/token.h"
#include "support/tokenizer/tokenizers/utils/regex.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace tokenizers {
namespace {

using MatchList = std::vector<std::pair<Offsets, bool>>;

void DoTest(const char* inside, Pattern pattern, const MatchList& expected) {
  ASSERT_OK_AND_ASSIGN(MatchList matches, pattern.FindMatches(inside));
  EXPECT_EQ(matches, expected);

  MatchList inverted_expected;
  inverted_expected.reserve(expected.size());
  for (const auto& [offsets, flag] : expected) {
    inverted_expected.emplace_back(offsets, !flag);
  }
  ASSERT_OK_AND_ASSIGN(MatchList inverted,
                       Invert(std::move(pattern)).FindMatches(inside));
  EXPECT_EQ(inverted, inverted_expected);
}

TEST(PatternTest, Char) {
  DoTest("aba", U'a', {{{0, 1}, true}, {{1, 2}, false}, {{2, 3}, true}});
  DoTest("bbbba", U'a', {{{0, 4}, false}, {{4, 5}, true}});
  DoTest("aabbb", U'a', {{{0, 1}, true}, {{1, 2}, true}, {{2, 5}, false}});
  DoTest("", U'a', {{{0, 0}, false}});
  DoTest("aaa", U'b', {{{0, 3}, false}});
}

TEST(PatternTest, Str) {
  DoTest("aba", "a", {{{0, 1}, true}, {{1, 2}, false}, {{2, 3}, true}});
  DoTest("bbbba", "a", {{{0, 4}, false}, {{4, 5}, true}});
  DoTest("aabbb", "a", {{{0, 1}, true}, {{1, 2}, true}, {{2, 5}, false}});
  DoTest("aabbb", "ab", {{{0, 1}, false}, {{1, 3}, true}, {{3, 5}, false}});
  DoTest("aabbab", "ab",
         {{{0, 1}, false}, {{1, 3}, true}, {{3, 4}, false}, {{4, 6}, true}});
  DoTest("", "", {{{0, 0}, false}});
  DoTest("aaa", "", {{{0, 3}, false}});
  DoTest("aaa", "b", {{{0, 3}, false}});
}

TEST(PatternTest, Functions) {
  auto is_b = [](char32_t c) { return c == U'b'; };
  DoTest("aba", is_b, {{{0, 1}, false}, {{1, 2}, true}, {{2, 3}, false}});
  DoTest("aaaab", is_b, {{{0, 4}, false}, {{4, 5}, true}});
  DoTest("bbaaa", is_b, {{{0, 1}, true}, {{1, 2}, true}, {{2, 5}, false}});
  DoTest("", is_b, {{{0, 0}, false}});
  DoTest("aaa", is_b, {{{0, 3}, false}});
}

TEST(PatternTest, Regex) {
  ASSERT_OK_AND_ASSIGN(SysRegex is_whitespace, SysRegex::New(R"(\s+)"));
  DoTest("a   b", &is_whitespace,
         {{{0, 1}, false}, {{1, 4}, true}, {{4, 5}, false}});
  DoTest("   a   b   ", &is_whitespace,
         {{{0, 3}, true},
          {{3, 4}, false},
          {{4, 7}, true},
          {{7, 8}, false},
          {{8, 11}, true}});
  DoTest("", &is_whitespace, {{{0, 0}, false}});
  DoTest("𝔾𝕠𝕠𝕕 𝕞𝕠𝕣𝕟𝕚𝕟𝕘", &is_whitespace,
         {{{0, 16}, false}, {{16, 17}, true}, {{17, 45}, false}});
  DoTest("aaa", &is_whitespace, {{{0, 3}, false}});
}

TEST(PatternTest, SysRegex) {
  ASSERT_OK_AND_ASSIGN(SysRegex is_whitespace, SysRegex::New(R"(\s+)"));
  DoTest("a   b", &is_whitespace,
         {{{0, 1}, false}, {{1, 4}, true}, {{4, 5}, false}});
  DoTest("   a   b   ", &is_whitespace,
         {{{0, 3}, true},
          {{3, 4}, false},
          {{4, 7}, true},
          {{7, 8}, false},
          {{8, 11}, true}});
  DoTest("", &is_whitespace, {{{0, 0}, false}});
  DoTest("𝔾𝕠𝕠𝕕 𝕞𝕠𝕣𝕟𝕚𝕟𝕘", &is_whitespace,
         {{{0, 16}, false}, {{16, 17}, true}, {{17, 45}, false}});
  DoTest("aaa", &is_whitespace, {{{0, 3}, false}});
}

}  // namespace
}  // namespace tokenizers
