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
// Tests for the C++ port of `src/utils/onig.rs` from the HuggingFace
// `tokenizers` crate v0.23.2 (https://github.com/huggingface/tokenizers,
// Apache License 2.0, Copyright 2020 The HuggingFace Team). `onig.rs` has no
// unit tests of its own; these pin down the Oniguruma behaviour the rest of
// the port relies on (the patterns of the crate, Unicode classes, empty
// matches, escaping, thread-safety).

#include "support/tokenizer/tokenizers/utils/regex.h"

#include <cstddef>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace tokenizers {
namespace {

using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::Pair;
using ::testing::status::StatusIs;

using Match = std::pair<size_t, size_t>;

// The pre-tokenization regex of `pre_tokenizers/byte_level.rs` (GPT-2).
constexpr absl::string_view kGpt2Pattern =
    R"('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)";
// The pre-tokenization regex of Llama 3 (`Split` pre-tokenizer in its
// tokenizer.json).
constexpr absl::string_view kLlama3Pattern =
    R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";
// The regex of `pre_tokenizers/whitespace.rs`.
constexpr absl::string_view kWhitespacePattern = R"(\w+|[^\w\s]+)";

// The matched substrings, for readable expectations.
std::vector<std::string> Matches(const SysRegex& regex,
                                 absl::string_view text) {
  std::vector<std::string> out;
  for (const auto& [start, end] : regex.FindIter(text)) {
    out.push_back(std::string(text.substr(start, end - start)));
  }
  return out;
}

TEST(SysRegex, InvalidPatterns) {
  EXPECT_THAT(SysRegex::New("("),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("end pattern with unmatched parenthesis")));
  EXPECT_THAT(SysRegex::New("[a"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("premature end of char-class")));
  EXPECT_THAT(SysRegex::New(R"(\p{NotAProperty})"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("invalid character property name")));
  EXPECT_THAT(
      SysRegex::New("*a"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("target of repeat operator is not specified")));
  // The message mirrors `onig::Error`'s `Display`.
  EXPECT_THAT(SysRegex::New("(").status().message(),
              HasSubstr("Oniguruma error: "));
}

TEST(SysRegex, PatternAccessorAndMove) {
  ASSERT_OK_AND_ASSIGN(SysRegex a, SysRegex::New("a+"));
  EXPECT_EQ(a.pattern(), "a+");
  SysRegex b = std::move(a);
  EXPECT_EQ(b.pattern(), "a+");
  EXPECT_THAT(b.FindIter("xaayaaa"), ElementsAre(Pair(1, 3), Pair(4, 7)));
  ASSERT_OK_AND_ASSIGN(SysRegex c, SysRegex::New("b"));
  c = std::move(b);
  EXPECT_EQ(c.pattern(), "a+");
  EXPECT_THAT(c.FindIter("xaayaaa"), ElementsAre(Pair(1, 3), Pair(4, 7)));
}

TEST(SysRegex, EmptyPatternAndEmptyText) {
  ASSERT_OK_AND_ASSIGN(SysRegex empty, SysRegex::New(""));
  EXPECT_THAT(empty.FindIter(""), ElementsAre(Pair(0, 0)));
  EXPECT_THAT(empty.FindIter("ab"),
              ElementsAre(Pair(0, 0), Pair(1, 1), Pair(2, 2)));
  ASSERT_OK_AND_ASSIGN(SysRegex spaces, SysRegex::New(R"(\s+)"));
  EXPECT_THAT(spaces.FindIter(""), IsEmpty());
  ASSERT_OK_AND_ASSIGN(SysRegex star, SysRegex::New(R"(\s*)"));
  EXPECT_THAT(star.FindIter(""), ElementsAre(Pair(0, 0)));
}

// `onig::FindMatches` skips an empty match right after the previous match by
// advancing one *character*, never one byte.
TEST(SysRegex, EmptyMatchesAdvanceByOneCharacter) {
  ASSERT_OK_AND_ASSIGN(SysRegex star, SysRegex::New(R"(\s*)"));
  EXPECT_THAT(star.FindIter("a b"),
              ElementsAre(Pair(0, 0), Pair(1, 2), Pair(3, 3)));
  ASSERT_OK_AND_ASSIGN(SysRegex x_star, SysRegex::New("x*"));
  // "日本" is 2 characters of 3 bytes each.
  EXPECT_THAT(x_star.FindIter("日本"),
              ElementsAre(Pair(0, 0), Pair(3, 3), Pair(6, 6)));
  EXPECT_THAT(x_star.FindIter("xx日x"), ElementsAre(Pair(0, 2), Pair(5, 6)));
  // 😀 is one 4-byte character.
  EXPECT_THAT(x_star.FindIter("😀"), ElementsAre(Pair(0, 0), Pair(4, 4)));
}

TEST(SysRegex, Gpt2Pattern) {
  ASSERT_OK_AND_ASSIGN(SysRegex regex, SysRegex::New(kGpt2Pattern));
  EXPECT_THAT(Matches(regex, "Hello world! I'm 42."),
              ElementsAre("Hello", " world", "!", " I", "'m", " 42", "."));
  EXPECT_THAT(regex.FindIter("Hello world! I'm 42."),
              ElementsAre(Pair(0, 5), Pair(5, 11), Pair(11, 12), Pair(12, 14),
                          Pair(14, 16), Pair(16, 19), Pair(19, 20)));
  EXPECT_THAT(Matches(regex, "I've you'll we're they'd it's isn't"),
              ElementsAre("I", "'ve", " you", "'ll", " we", "'re", " they",
                          "'d", " it", "'s", " isn", "'t"));
  // `\s+(?!\S)` keeps the last space of a run attached to the next word.
  EXPECT_THAT(Matches(regex, "a  b"), ElementsAre("a", " ", " b"));
  EXPECT_THAT(Matches(regex, "a   b   "), ElementsAre("a", "  ", " b", "   "));
  EXPECT_THAT(Matches(regex, "\n\n  x"), ElementsAre("\n\n ", " x"));
  EXPECT_THAT(Matches(regex, "   "), ElementsAre("   "));
  EXPECT_THAT(regex.FindIter(""), IsEmpty());
}

TEST(SysRegex, Gpt2PatternUnicode) {
  ASSERT_OK_AND_ASSIGN(SysRegex regex, SysRegex::New(kGpt2Pattern));
  // CJK letters are \p{L}; the emoji is neither letter, number nor space.
  EXPECT_THAT(regex.FindIter("日本語 😀 café"),
              ElementsAre(Pair(0, 9), Pair(9, 14), Pair(14, 20)));
  EXPECT_THAT(Matches(regex, "日本語 😀 café"),
              ElementsAre("日本語", " 😀", " café"));
  // A combining mark (\p{M}) is not \p{L}: it is split off like GPT-2 does.
  EXPECT_THAT(regex.FindIter("e\u0301"), ElementsAre(Pair(0, 1), Pair(1, 3)));
  // Precomposed, it is a single letter.
  EXPECT_THAT(regex.FindIter("\u00e9"), ElementsAre(Pair(0, 2)));
  // Non-ASCII digits are \p{N}; non-ASCII spaces (\u00a0, \u3000) are \s (not
  // the literal ASCII ' ' prefix of ` ?\p{N}+` / ` ?\p{L}+`), so they match the
  // trailing `\s+` branch as their own token.
  EXPECT_THAT(Matches(regex, "x\u00a0\u0663\u0664"),
              ElementsAre("x", "\u00a0", "\u0663\u0664"));
  EXPECT_THAT(Matches(regex, "x\u3000y"), ElementsAre("x", "\u3000", "y"));
  EXPECT_THAT(Matches(regex, "Ünïcödé Straße"),
              ElementsAre("Ünïcödé", " Straße"));
}

TEST(SysRegex, Llama3Pattern) {
  ASSERT_OK_AND_ASSIGN(SysRegex regex, SysRegex::New(kLlama3Pattern));
  EXPECT_THAT(regex.FindIter("Hello World 12345 I'M\n\n ok"),
              ElementsAre(Pair(0, 5), Pair(5, 11), Pair(11, 12), Pair(12, 15),
                          Pair(15, 17), Pair(17, 19), Pair(19, 21),
                          Pair(21, 23), Pair(23, 26)));
  EXPECT_THAT(Matches(regex, "Hello World 12345 I'M\n\n ok"),
              ElementsAre("Hello", " World", " ", "123", "45", " I", "'M",
                          "\n\n", " ok"));
  // Numbers are split in groups of at most three digits.
  EXPECT_THAT(Matches(regex, "1234567"), ElementsAre("123", "456", "7"));
  // `(?i:...)` makes the contractions case-insensitive.
  EXPECT_THAT(Matches(regex, "DON'T Don't"),
              ElementsAre("DON", "'T", " Don", "'t"));
  // Punctuation swallows following newlines.
  EXPECT_THAT(Matches(regex, "end.\n\nNext"),
              ElementsAre("end", ".\n\n", "Next"));
  EXPECT_THAT(Matches(regex, "日本語の文章"), ElementsAre("日本語の文章"));
  EXPECT_THAT(Matches(regex, "a  b"), ElementsAre("a", " ", " b"));
}

TEST(SysRegex, WhitespacePattern) {
  ASSERT_OK_AND_ASSIGN(SysRegex regex, SysRegex::New(kWhitespacePattern));
  // Expectations of `pre_tokenizers/whitespace.rs`' `basic` test.
  EXPECT_THAT(regex.FindIter("Hey friend!     How are you?!?"),
              ElementsAre(Pair(0, 3), Pair(4, 10), Pair(10, 11), Pair(16, 19),
                          Pair(20, 23), Pair(24, 27), Pair(27, 30)));
  // `\w` is Unicode-aware (letters, marks, digits, connector punctuation).
  EXPECT_THAT(Matches(regex, "héllo 日本語 ٣٤ a_b e\u0301"),
              ElementsAre("héllo", "日本語", "٣٤", "a_b", "e\u0301"));
  EXPECT_THAT(Matches(regex, "!!a--b 😀"),
              ElementsAre("!!", "a", "--", "b", "😀"));
}

TEST(SysRegex, WhitespaceRun) {
  ASSERT_OK_AND_ASSIGN(SysRegex regex, SysRegex::New(R"(\s+)"));
  EXPECT_THAT(regex.FindIter("a b\tc\n\nd"),
              ElementsAre(Pair(1, 2), Pair(3, 4), Pair(5, 7)));
  // `\s` is the Unicode White_Space property: NBSP (2 bytes), ideographic
  // space (3 bytes), but not a zero width space.
  EXPECT_THAT(regex.FindIter("a\u00a0b\u3000c\u200bd"),
              ElementsAre(Pair(1, 3), Pair(4, 7)));
  EXPECT_THAT(regex.FindIter("nospace"), IsEmpty());
  EXPECT_THAT(regex.FindIter("  "), ElementsAre(Pair(0, 2)));
}

TEST(SysRegex, DigitsAndWordClasses) {
  ASSERT_OK_AND_ASSIGN(SysRegex digits, SysRegex::New(R"(\d+)"));
  // `\d` is \p{Nd}, like the `regex` crate's.
  EXPECT_THAT(Matches(digits, "a1b٢٣c"), ElementsAre("1", "٢٣"));
  ASSERT_OK_AND_ASSIGN(SysRegex p_n, SysRegex::New(R"(\p{N})"));
  // ² (U+00B2) is No, not Nd.
  EXPECT_THAT(Matches(p_n, "x²y"), ElementsAre("²"));
  EXPECT_THAT(Matches(digits, "x²y"), IsEmpty());
}

// Documented in regex.h: Oniguruma's default syntax has Ruby's line anchors.
TEST(SysRegex, AnchorsAreLineAnchors) {
  ASSERT_OK_AND_ASSIGN(SysRegex caret, SysRegex::New("^a"));
  EXPECT_THAT(caret.FindIter("b\na"), ElementsAre(Pair(2, 3)));
  ASSERT_OK_AND_ASSIGN(SysRegex text_start, SysRegex::New(R"(\Aa)"));
  EXPECT_THAT(text_start.FindIter("b\na"), IsEmpty());
  EXPECT_THAT(text_start.FindIter("ab"), ElementsAre(Pair(0, 1)));
  ASSERT_OK_AND_ASSIGN(SysRegex dollar, SysRegex::New("a$"));
  EXPECT_THAT(dollar.FindIter("a\nb"), ElementsAre(Pair(0, 1)));
  ASSERT_OK_AND_ASSIGN(SysRegex text_end, SysRegex::New(R"(a\z)"));
  EXPECT_THAT(text_end.FindIter("a\nb"), IsEmpty());
  EXPECT_THAT(text_end.FindIter("ba"), ElementsAre(Pair(1, 2)));
}

TEST(SysRegex, LookaroundAndCaseInsensitiveGroups) {
  ASSERT_OK_AND_ASSIGN(SysRegex lookahead, SysRegex::New(R"(\d+(?=px))"));
  EXPECT_THAT(Matches(lookahead, "10px 20em 30px"), ElementsAre("10", "30"));
  ASSERT_OK_AND_ASSIGN(SysRegex lookbehind, SysRegex::New(R"((?<=\$)\d+)"));
  EXPECT_THAT(Matches(lookbehind, "$10 €20 $30"), ElementsAre("10", "30"));
  ASSERT_OK_AND_ASSIGN(SysRegex insensitive, SysRegex::New("(?i:é|ß)"));
  EXPECT_THAT(Matches(insensitive, "É ẞ e"), ElementsAre("É", "ẞ"));
}

TEST(RegexEscape, EscapesExactlyTheRegexSyntaxMetaCharacters) {
  EXPECT_EQ(RegexEscape("a.b*c"), R"(a\.b\*c)");
  EXPECT_EQ(RegexEscape(R"(\.+*?()|[]{}^$#&-~)"),
            R"(\\\.\+\*\?\(\)\|\[\]\{\}\^\$\#\&\-\~)");
  // Everything else is left alone, including non-ASCII characters.
  EXPECT_EQ(RegexEscape("<br> / 'q' \"dq\" @%=!:;,_ 日本語▁😀"),
            "<br> / 'q' \"dq\" @%=!:;,_ 日本語▁😀");
  EXPECT_EQ(RegexEscape(""), "");
  EXPECT_EQ(RegexEscape("abc123_"), "abc123_");
}

TEST(RegexEscape, EscapedPatternMatchesLiterally) {
  const std::string literal =
      R"(!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~ ▁<|endoftext|>日本)";
  ASSERT_OK_AND_ASSIGN(SysRegex regex, SysRegex::New(RegexEscape(literal)));
  const std::string text = absl::StrCat("xx", literal, "yy", literal);
  EXPECT_THAT(regex.FindIter(text),
              ElementsAre(Pair(2, 2 + literal.size()),
                          Pair(2 + literal.size() + 2,
                               2 + literal.size() + 2 + literal.size())));
  // Unescaped, `<` and `>` are literals in this syntax too (no \< \> word
  // boundaries), which is what `Replace`/`Split` with a `String` pattern need.
  ASSERT_OK_AND_ASSIGN(SysRegex br, SysRegex::New(RegexEscape("<br>")));
  EXPECT_THAT(br.FindIter("a<br>b<br>"), ElementsAre(Pair(1, 5), Pair(6, 10)));
  ASSERT_OK_AND_ASSIGN(SysRegex meta, SysRegex::New(RegexEscape("▁")));
  EXPECT_THAT(meta.FindIter("▁Hello▁World"),
              ElementsAre(Pair(0, 3), Pair(8, 11)));
}

TEST(SysRegex, ConcurrentFindIter) {
  ASSERT_OK_AND_ASSIGN(SysRegex regex, SysRegex::New(kGpt2Pattern));
  const std::string text =
      "The quick brown fox jumps over the lazy dog 123 times! 日本語 😀 "
      "I'm sure we're done, aren't we?";
  const std::vector<Match> expected = regex.FindIter(text);
  ASSERT_FALSE(expected.empty());
  std::vector<std::thread> threads;
  std::vector<bool> ok(8, false);
  for (int t = 0; t < 8; ++t) {
    threads.emplace_back([&, t] {
      bool all_equal = true;
      for (int i = 0; i < 200; ++i) {
        if (regex.FindIter(text) != expected) all_equal = false;
      }
      ok[t] = all_equal;
    });
  }
  for (auto& thread : threads) thread.join();
  for (int t = 0; t < 8; ++t) EXPECT_TRUE(ok[t]) << "thread " << t;
}

TEST(SysRegex, ConcurrentNew) {
  std::vector<std::thread> threads;
  std::vector<absl::Status> statuses(8);
  for (int t = 0; t < 8; ++t) {
    threads.emplace_back([&, t] {
      for (int i = 0; i < 50; ++i) {
        absl::StatusOr<SysRegex> regex =
            SysRegex::New(t % 2 == 0 ? kGpt2Pattern : kLlama3Pattern);
        if (!regex.ok()) {
          statuses[t] = regex.status();
          return;
        }
        if (regex->FindIter("ab cd").size() != 2) {
          statuses[t] = absl::InternalError("unexpected match count");
          return;
        }
      }
    });
  }
  for (auto& thread : threads) thread.join();
  for (int t = 0; t < 8; ++t) EXPECT_OK(statuses[t]) << "thread " << t;
}

}  // namespace
}  // namespace tokenizers
