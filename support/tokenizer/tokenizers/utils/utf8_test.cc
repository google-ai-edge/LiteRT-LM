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

#include "support/tokenizer/tokenizers/utils/utf8.h"

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"  // from @com_google_absl

namespace tokenizers {
namespace {

TEST(Utf8Test, Utf8LenAndEncodeDecodeRoundTrip) {
  EXPECT_EQ(Utf8Len(U'\0'), 1u);
  EXPECT_EQ(Utf8Len(U'a'), 1u);
  EXPECT_EQ(Utf8Len(U'\x7F'), 1u);
  EXPECT_EQ(Utf8Len(U'\u0080'), 2u);
  EXPECT_EQ(Utf8Len(U'é'), 2u);
  EXPECT_EQ(Utf8Len(U'\u07FF'), 2u);
  EXPECT_EQ(Utf8Len(U'\u0800'), 3u);
  EXPECT_EQ(Utf8Len(U'▁'), 3u);
  EXPECT_EQ(Utf8Len(U'日'), 3u);
  EXPECT_EQ(Utf8Len(U'\uFFFF'), 3u);
  EXPECT_EQ(Utf8Len(U'\U00010000'), 4u);
  EXPECT_EQ(Utf8Len(U'😀'), 4u);
  EXPECT_EQ(Utf8Len(U'\U0010FFFF'), 4u);

  const std::vector<char32_t> code_points = {
      U'\0', U'A',  U'\x7F',   U'\u0080', U'é',  U'\u07FF',     U'\u0800',
      U'▁',  U'日', U'\uFFFD', U'\uFFFF', U'😀', U'\U00010000', U'\U0010FFFF',
  };
  std::string combined;
  for (char32_t cp : code_points) {
    const std::string one = ToUtf8(cp);
    EXPECT_EQ(one.size(), Utf8Len(cp));
    size_t pos = 0;
    EXPECT_EQ(DecodeUtf8(one, &pos), cp);
    EXPECT_EQ(pos, one.size());
    AppendUtf8(cp, &combined);
  }
  EXPECT_TRUE(IsValidUtf8(combined));
  EXPECT_EQ(CountChars(combined), code_points.size());

  // Non-scalar values (surrogates, > U+10FFFF) are replaced by U+FFFD.
  EXPECT_EQ(ToUtf8(0xD800), "\xEF\xBF\xBD");
  EXPECT_EQ(ToUtf8(0xDFFF), "\xEF\xBF\xBD");
  EXPECT_EQ(ToUtf8(0x110000), "\xEF\xBF\xBD");
}

TEST(Utf8Test, CharIndicesAndCountChars) {
  EXPECT_EQ(CountChars(""), 0u);
  const CharIndices empty("");
  EXPECT_EQ(empty.begin(), empty.end());

  // 'a' (1B @ 0), 'é' (2B @ 1), '日' (3B @ 3), '😀' (4B @ 6), 'z' (1B @ 10).
  constexpr absl::string_view kText = "aé日😀z";
  EXPECT_EQ(kText.size(), 11u);
  EXPECT_EQ(CountChars(kText), 5u);

  std::vector<std::pair<size_t, char32_t>> actual;
  for (auto item : CharIndices(kText)) {
    actual.push_back(item);
  }
  const std::vector<std::pair<size_t, char32_t>> expected = {
      {0, U'a'}, {1, U'é'}, {3, U'日'}, {6, U'😀'}, {10, U'z'},
  };
  EXPECT_EQ(actual, expected);

  // Post-increment and offset() accessor.
  CharIndices range(kText);
  auto it = range.begin();
  EXPECT_EQ(it.offset(), 0u);
  auto prev = it++;
  EXPECT_EQ(*prev, std::make_pair(size_t{0}, U'a'));
  EXPECT_EQ(*it, std::make_pair(size_t{1}, U'é'));
}

TEST(Utf8Test, IsValidUtf8AndFromUtf8Lossy) {
  EXPECT_TRUE(IsValidUtf8(""));
  EXPECT_TRUE(IsValidUtf8(absl::string_view("a\0b", 3)));
  EXPECT_TRUE(IsValidUtf8("Hello, 世界! 😀"));

  // Stray continuation byte.
  EXPECT_FALSE(IsValidUtf8("\x80"));
  EXPECT_EQ(FromUtf8Lossy("a\x80z"), "a\uFFFDz");

  // Overlong 2-byte encodings (0xC0, 0xC1 are never valid lead bytes).
  EXPECT_FALSE(IsValidUtf8("\xC0\xAF"));
  EXPECT_FALSE(IsValidUtf8("\xC1\xBF"));
  EXPECT_EQ(FromUtf8Lossy("\xC0\xAF"), "\uFFFD\uFFFD");

  // Overlong 3-byte encoding (0xE0 followed by < 0xA0).
  EXPECT_FALSE(IsValidUtf8("\xE0\x80\x80"));
  EXPECT_EQ(FromUtf8Lossy("\xE0\x80\x80"), "\uFFFD\uFFFD\uFFFD");

  // UTF-16 surrogate in UTF-8 (U+D800 = 0xED 0xA0 0x80): each byte is its own
  // maximal subpart.
  EXPECT_FALSE(IsValidUtf8("\xED\xA0\x80"));
  EXPECT_EQ(FromUtf8Lossy("\xED\xA0\x80"), "\uFFFD\uFFFD\uFFFD");

  // Overlong 4-byte encoding and > U+10FFFF.
  EXPECT_FALSE(IsValidUtf8("\xF0\x80\x80\x80"));
  EXPECT_FALSE(IsValidUtf8("\xF4\x90\x80\x80"));
  EXPECT_FALSE(IsValidUtf8("\xF5\x80\x80\x80"));

  // Truncated valid prefixes: the whole valid prefix is one maximal subpart,
  // and any following ASCII byte is preserved intact.
  EXPECT_FALSE(IsValidUtf8("\xC3"));
  EXPECT_EQ(FromUtf8Lossy("\xC3x"), "\uFFFDx");
  EXPECT_FALSE(IsValidUtf8("\xE2\x82"));
  EXPECT_EQ(FromUtf8Lossy("\xE2\x82x"), "\uFFFDx");
  EXPECT_FALSE(IsValidUtf8("\xF0\x9F\x98"));
  EXPECT_EQ(FromUtf8Lossy("\xF0\x9F\x98x"), "\uFFFDx");
}

}  // namespace
}  // namespace tokenizers
