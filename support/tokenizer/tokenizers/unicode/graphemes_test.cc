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
// Tests of the C++ port of `unicode-segmentation`'s grapheme segmentation:
// the `#[test]`s of `src/grapheme.rs`, `test_graphemes` of `tests/test.rs`
// (official GraphemeBreakTest.txt data via graphemes_test_data.h) and the
// forward/reverse consistency properties `tests/test.rs` checks with
// quickcheck.

#include "support/tokenizer/tokenizers/unicode/graphemes.h"

#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/unicode/graphemes_test_data.h"

namespace tokenizers::unicode {
namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::Pair;

using OptionalSize = std::optional<size_t>;
using SizeResult = GraphemeResult<OptionalSize>;
using BoolResult = GraphemeResult<bool>;

std::vector<std::string> ToStrings(
    const std::vector<absl::string_view>& views) {
  return std::vector<std::string>(views.begin(), views.end());
}

// `s.graphemes(is_extended).rev().collect()`.
std::vector<std::string> GraphemesReversed(absl::string_view s,
                                           bool is_extended) {
  std::vector<std::string> out;
  GraphemesIterator iter(s, is_extended);
  while (std::optional<absl::string_view> g = iter.NextBack()) {
    out.emplace_back(*g);
  }
  return out;
}

std::vector<std::string> Reversed(std::vector<std::string> v) {
  return std::vector<std::string>(v.rbegin(), v.rend());
}

// ---- src/grapheme.rs `#[test]`s --------------------------------------------

TEST(GraphemeCursorTest, TestGraphemeCursorRisPrecontext) {
  const absl::string_view s =
      "\U0001f1fa\U0001f1f8\U0001f1fa\U0001f1f8\U0001f1fa\U0001f1f8";
  GraphemeCursor c(8, s.size(), true);
  EXPECT_EQ(c.IsBoundary(s.substr(4), 4),
            BoolResult(GraphemeIncomplete::PreContext(4)));
  c.ProvideContext(s.substr(0, 4), 0);
  EXPECT_EQ(c.IsBoundary(s.substr(4), 4), BoolResult(true));
}

TEST(GraphemeCursorTest, TestGraphemeCursorChunkStartRequirePrecontext) {
  const absl::string_view s = "\r\n";
  GraphemeCursor c(1, s.size(), true);
  EXPECT_EQ(c.IsBoundary(s.substr(1), 1),
            BoolResult(GraphemeIncomplete::PreContext(1)));
  c.ProvideContext(s.substr(0, 1), 0);
  EXPECT_EQ(c.IsBoundary(s.substr(1), 1), BoolResult(false));
}

TEST(GraphemeCursorTest, TestGraphemeCursorPrevBoundary) {
  const absl::string_view s = "abcd";
  GraphemeCursor c(3, s.size(), true);
  EXPECT_EQ(c.PrevBoundary(s.substr(2), 2),
            SizeResult(GraphemeIncomplete::PrevChunk()));
  EXPECT_EQ(c.PrevBoundary(s.substr(0, 2), 0), SizeResult(OptionalSize(2)));
}

TEST(GraphemeCursorTest, TestGraphemeCursorPrevBoundaryChunkStart) {
  const absl::string_view s = "abcd";
  GraphemeCursor c(2, s.size(), true);
  EXPECT_EQ(c.PrevBoundary(s.substr(2), 2),
            SizeResult(GraphemeIncomplete::PrevChunk()));
  EXPECT_EQ(c.PrevBoundary(s.substr(0, 2), 0), SizeResult(OptionalSize(1)));
}

TEST(GraphemeCursorTest, TestGraphemeCursorBoundaryWithZwjOnChunkStart) {
  const absl::string_view chunk0 = "\U0001f469";        // 4 bytes
  const absl::string_view chunk1 = "\u200d\U0001f52c";  // 3 bytes + 4 bytes

  const size_t full_len = chunk0.size() + chunk1.size();

  GraphemeCursor cur(0, full_len, true);
  EXPECT_EQ(cur.NextBoundary(chunk0, 0),
            SizeResult(GraphemeIncomplete::NextChunk()));
  const SizeResult res = cur.NextBoundary(chunk1, chunk0.size());
  if (res.ok()) {
    EXPECT_EQ(res.value(), OptionalSize(11));
  } else if (res.error().kind == GraphemeIncomplete::Kind::kPreContext) {
    cur.ProvideContext(chunk0, 0);
    EXPECT_EQ(cur.NextBoundary(chunk1, chunk0.size()),
              SizeResult(OptionalSize(11)));
  } else {
    FAIL() << "unreachable: " << res;
  }
}

TEST(GraphemeCursorTest, TestGraphemeCursorEmojiNoZwj) {
  const absl::string_view chunk0 = "\U0001f352";  // 4 bytes
  const absl::string_view chunk1 = "\U0001f951";  // 4 bytes
  const size_t full_len = chunk0.size() + chunk1.size();

  GraphemeCursor c(0, full_len, true);
  EXPECT_EQ(c.NextBoundary(chunk0, 0),
            SizeResult(GraphemeIncomplete::NextChunk()));
  EXPECT_EQ(c.NextBoundary(chunk1, chunk0.size()),
            SizeResult(GraphemeIncomplete::PreContext(chunk0.size())));
  c.ProvideContext(chunk0, 0);
  EXPECT_EQ(c.NextBoundary(chunk1, chunk0.size()), SizeResult(OptionalSize(4)));
  EXPECT_EQ(c.NextBoundary(chunk1, chunk0.size()), SizeResult(OptionalSize(8)));
  EXPECT_EQ(c.NextBoundary(chunk1, chunk0.size()),
            SizeResult(OptionalSize(std::nullopt)));
}

TEST(GraphemeCursorTest, TestGraphemeCursorEmojiChunkBoundaryBeforeZwj) {
  const absl::string_view chunk0 = "\U0001f352";          // 4 bytes
  const absl::string_view chunk1 = "\u200d\U0001f951";    // 3 + 4 bytes
  const size_t full_len = chunk0.size() + chunk1.size();  // 11

  GraphemeCursor c(0, full_len, true);
  EXPECT_EQ(c.NextBoundary(chunk0, 0),
            SizeResult(GraphemeIncomplete::NextChunk()));
  EXPECT_EQ(c.NextBoundary(chunk1, chunk0.size()),
            SizeResult(GraphemeIncomplete::PreContext(chunk0.size())));
  c.ProvideContext(chunk0, 0);
  EXPECT_EQ(c.NextBoundary(chunk1, chunk0.size()),
            SizeResult(OptionalSize(11)));
  EXPECT_EQ(c.NextBoundary(chunk1, chunk0.size()),
            SizeResult(OptionalSize(std::nullopt)));
}

TEST(GraphemeCursorTest, TestGraphemeCursorEmojiChunkBoundaryAfterZwj) {
  const absl::string_view chunk0 = "\U0001f352\u200d";    // 4 + 3 bytes
  const absl::string_view chunk1 = "\U0001f951";          // 4 bytes
  const size_t full_len = chunk0.size() + chunk1.size();  // 11

  GraphemeCursor c(0, full_len, true);
  EXPECT_EQ(c.NextBoundary(chunk0, 0),
            SizeResult(GraphemeIncomplete::NextChunk()));
  EXPECT_EQ(c.NextBoundary(chunk1, chunk0.size()),
            SizeResult(GraphemeIncomplete::PreContext(chunk0.size())));
  c.ProvideContext(chunk0, 0);
  EXPECT_EQ(c.NextBoundary(chunk1, chunk0.size()),
            SizeResult(OptionalSize(11)));
  EXPECT_EQ(c.NextBoundary(chunk1, chunk0.size()),
            SizeResult(OptionalSize(std::nullopt)));
}

TEST(GraphemeCursorTest, TestGraphemeCursorEmojiZwjAcrossChunks) {
  const absl::string_view chunk0 = "\U0001f352";  // 4 bytes
  const absl::string_view chunk1 = "\u200d";      // 3 bytes
  const absl::string_view chunk2 = "\U0001f951";  // 4 bytes
  const size_t full_len = chunk0.size() + chunk1.size() + chunk2.size();  // 11
  const size_t chunk2_start = chunk0.size() + chunk1.size();

  GraphemeCursor c(0, full_len, true);
  EXPECT_EQ(c.NextBoundary(chunk0, 0),
            SizeResult(GraphemeIncomplete::NextChunk()));
  EXPECT_EQ(c.NextBoundary(chunk1, chunk0.size()),
            SizeResult(GraphemeIncomplete::NextChunk()));
  EXPECT_EQ(c.NextBoundary(chunk2, chunk2_start),
            SizeResult(GraphemeIncomplete::PreContext(chunk2_start)));
  c.ProvideContext(chunk1, chunk0.size());
  EXPECT_EQ(c.NextBoundary(chunk2, chunk2_start),
            SizeResult(GraphemeIncomplete::PreContext(chunk0.size())));
  c.ProvideContext(chunk0, 0);
  EXPECT_EQ(c.NextBoundary(chunk2, chunk2_start), SizeResult(OptionalSize(11)));
  EXPECT_EQ(c.NextBoundary(chunk2, chunk2_start),
            SizeResult(OptionalSize(std::nullopt)));
}

// ---- doc examples of src/grapheme.rs ---------------------------------------

TEST(GraphemeCursorTest, DocExamples) {
  {
    const absl::string_view s = "हिन्दी";
    GraphemeCursor legacy(0, s.size(), false);
    EXPECT_EQ(legacy.NextBoundary(s, 0),
              SizeResult(OptionalSize(absl::string_view("ह").size())));
    GraphemeCursor extended(0, s.size(), true);
    EXPECT_EQ(extended.NextBoundary(s, 0),
              SizeResult(OptionalSize(absl::string_view("हि").size())));
  }
  {
    const absl::string_view s = "abcd";
    GraphemeCursor cursor(0, s.size(), false);
    EXPECT_EQ(cursor.CurCursor(), 0);
    cursor.SetCursor(2);
    EXPECT_EQ(cursor.CurCursor(), 2);
  }
  // Two flags (🇷🇸🇮🇴), each flag is two RIS codepoints, each RIS is 4 bytes.
  const absl::string_view flags = "\U0001F1F7\U0001F1F8\U0001F1EE\U0001F1F4";
  {
    GraphemeCursor cursor(4, flags.size(), false);
    EXPECT_EQ(cursor.CurCursor(), 4);
    EXPECT_EQ(cursor.NextBoundary(flags, 0), SizeResult(OptionalSize(8)));
    EXPECT_EQ(cursor.CurCursor(), 8);
  }
  {
    GraphemeCursor cursor(8, flags.size(), false);
    // Not enough pre-context to decide if there's a boundary between the two
    // flags.
    EXPECT_EQ(cursor.IsBoundary(flags.substr(8), 8),
              BoolResult(GraphemeIncomplete::PreContext(8)));
    // Provide one more Regional Indicator Symbol of pre-context
    cursor.ProvideContext(flags.substr(4, 4), 4);
    // Still not enough context to decide.
    EXPECT_EQ(cursor.IsBoundary(flags.substr(8), 8),
              BoolResult(GraphemeIncomplete::PreContext(4)));
    // Provide additional requested context.
    cursor.ProvideContext(flags.substr(0, 4), 0);
    // That's enough to decide (it always is when context goes to the start of
    // the string)
    EXPECT_EQ(cursor.IsBoundary(flags.substr(8), 8), BoolResult(true));
  }
  {
    GraphemeCursor cursor(8, flags.size(), false);
    EXPECT_EQ(cursor.IsBoundary(flags, 0), BoolResult(true));
    cursor.SetCursor(12);
    EXPECT_EQ(cursor.IsBoundary(flags, 0), BoolResult(false));
  }
  {
    GraphemeCursor cursor(4, flags.size(), false);
    EXPECT_EQ(cursor.NextBoundary(flags, 0), SizeResult(OptionalSize(8)));
    EXPECT_EQ(cursor.NextBoundary(flags, 0), SizeResult(OptionalSize(16)));
    EXPECT_EQ(cursor.NextBoundary(flags, 0),
              SizeResult(OptionalSize(std::nullopt)));
  }
  {
    const absl::string_view s = "abcd";
    GraphemeCursor cursor(0, s.size(), false);
    EXPECT_EQ(cursor.NextBoundary(s.substr(0, 2), 0),
              SizeResult(OptionalSize(1)));
    EXPECT_EQ(cursor.NextBoundary(s.substr(0, 2), 0),
              SizeResult(GraphemeIncomplete::NextChunk()));
    EXPECT_EQ(cursor.NextBoundary(s.substr(2, 2), 2),
              SizeResult(OptionalSize(2)));
    EXPECT_EQ(cursor.NextBoundary(s.substr(2, 2), 2),
              SizeResult(OptionalSize(3)));
    EXPECT_EQ(cursor.NextBoundary(s.substr(2, 2), 2),
              SizeResult(OptionalSize(4)));
    EXPECT_EQ(cursor.NextBoundary(s.substr(2, 2), 2),
              SizeResult(OptionalSize(std::nullopt)));
  }
  {
    GraphemeCursor cursor(12, flags.size(), false);
    EXPECT_EQ(cursor.PrevBoundary(flags, 0), SizeResult(OptionalSize(8)));
    EXPECT_EQ(cursor.PrevBoundary(flags, 0), SizeResult(OptionalSize(0)));
    EXPECT_EQ(cursor.PrevBoundary(flags, 0),
              SizeResult(OptionalSize(std::nullopt)));
  }
  {
    const absl::string_view s = "abcd";
    GraphemeCursor cursor(4, s.size(), false);
    EXPECT_EQ(cursor.PrevBoundary(s.substr(2, 2), 2),
              SizeResult(OptionalSize(3)));
    EXPECT_EQ(cursor.PrevBoundary(s.substr(2, 2), 2),
              SizeResult(GraphemeIncomplete::PrevChunk()));
    EXPECT_EQ(cursor.PrevBoundary(s.substr(0, 2), 0),
              SizeResult(OptionalSize(2)));
    EXPECT_EQ(cursor.PrevBoundary(s.substr(0, 2), 0),
              SizeResult(OptionalSize(1)));
    EXPECT_EQ(cursor.PrevBoundary(s.substr(0, 2), 0),
              SizeResult(OptionalSize(0)));
    EXPECT_EQ(cursor.PrevBoundary(s.substr(0, 2), 0),
              SizeResult(OptionalSize(std::nullopt)));
  }
  {
    GraphemesIterator iter("abc", true);
    EXPECT_EQ(iter.AsStr(), "abc");
    iter.Next();
    EXPECT_EQ(iter.AsStr(), "bc");
    iter.Next();
    iter.Next();
    EXPECT_EQ(iter.AsStr(), "");
  }
  {
    GraphemeIndicesIterator iter("abc", true);
    EXPECT_EQ(iter.AsStr(), "abc");
    iter.Next();
    EXPECT_EQ(iter.AsStr(), "bc");
    iter.Next();
    iter.Next();
    EXPECT_EQ(iter.AsStr(), "");
  }
}

// ---- tests/test.rs `test_graphemes` ----------------------------------------

TEST(GraphemesTest, TestGraphemes) {
  // Official test suite doesn't include two Prepend chars between two other
  // chars. / Test for Prepend followed by two Any chars
  const std::vector<GraphemeTestDiff> extra_diff = {
      {"\u0020\u0600\u0600\u0020",
       {"\u0020", "\u0600\u0600\u0020"},
       {"\u0020", "\u0600", "\u0600", "\u0020"}},
      {"\u0600\u0020\u0020",
       {"\u0600\u0020", "\u0020"},
       {"\u0600", "\u0020", "\u0020"}},
  };

  const std::vector<GraphemeTestSame> extra_same = {
      // family emoji (more than two emoji joined by ZWJ)
      {"\U0001f468\u200d\U0001f467\u200d\U0001f466",
       {"\U0001f468\u200d\U0001f467\u200d\U0001f466"}},
      // cartwheel emoji followed by two fitzpatrick skin tone modifiers
      // (test case from issue #19)
      {"\U0001F938\U0001F3FE\U0001F3FE", {"\U0001F938\U0001F3FE\U0001F3FE"}},
  };

  const std::vector<GraphemeTestSame>& test_same = GraphemeTestSameCases();
  const std::vector<GraphemeTestDiff>& test_diff = GraphemeTestDiffCases();
  ASSERT_GT(test_same.size(), 500);
  ASSERT_GT(test_diff.size(), 50);

  const auto check_same = [](const GraphemeTestSame& t) {
    SCOPED_TRACE(t.input);
    // test forward iterator
    const std::vector<std::string> our_extended =
        ToStrings(Graphemes(t.input, true));
    const std::vector<std::string> our_legacy =
        ToStrings(Graphemes(t.input, false));
    EXPECT_EQ(our_extended, t.graphemes) << "extended";
    EXPECT_EQ(our_legacy, t.graphemes) << "legacy";

    // test reverse iterator
    EXPECT_EQ(GraphemesReversed(t.input, true), Reversed(t.graphemes));
    EXPECT_EQ(GraphemesReversed(t.input, false), Reversed(t.graphemes));
  };
  for (const GraphemeTestSame& t : test_same) check_same(t);
  for (const GraphemeTestSame& t : extra_same) check_same(t);

  const auto check_diff = [](const GraphemeTestDiff& t) {
    SCOPED_TRACE(t.input);
    // test forward iterator
    EXPECT_EQ(ToStrings(Graphemes(t.input, true)), t.extended);
    EXPECT_EQ(ToStrings(Graphemes(t.input, false)), t.legacy);

    // test reverse iterator
    EXPECT_EQ(GraphemesReversed(t.input, true), Reversed(t.extended));
    EXPECT_EQ(GraphemesReversed(t.input, false), Reversed(t.legacy));
  };
  for (const GraphemeTestDiff& t : test_diff) check_diff(t);
  for (const GraphemeTestDiff& t : extra_diff) check_diff(t);

  // test the indices iterators
  {
    const absl::string_view s = "a\u0310e\u0301o\u0308\u0332\r\n";
    EXPECT_THAT(GraphemeIndices(s, true),
                ElementsAre(Pair(0, "a\u0310"), Pair(3, "e\u0301"),
                            Pair(6, "o\u0308\u0332"), Pair(11, "\r\n")));
    std::vector<std::pair<size_t, std::string>> rev;
    GraphemeIndicesIterator rev_iter(s, true);
    while (std::optional<std::pair<size_t, absl::string_view>> g =
               rev_iter.NextBack()) {
      rev.emplace_back(g->first, std::string(g->second));
    }
    EXPECT_THAT(rev, ElementsAre(Pair(11, "\r\n"), Pair(6, "o\u0308\u0332"),
                                 Pair(3, "e\u0301"), Pair(0, "a\u0310")));
    GraphemeIndicesIterator gr_inds_iter(s, true);
    {
      const auto e1 = gr_inds_iter.SizeHint();
      EXPECT_EQ(e1, std::make_pair(size_t{1}, OptionalSize(13)));
      size_t c = 0;
      while (gr_inds_iter.Next().has_value()) ++c;
      EXPECT_EQ(c, 4);
    }
    const auto e2 = gr_inds_iter.SizeHint();
    EXPECT_EQ(e2, std::make_pair(size_t{0}, OptionalSize(0)));
  }

  // make sure the reverse iterator does the right thing with "\n" at beginning
  // of string
  {
    const absl::string_view s = "\n\r\n\r";
    EXPECT_THAT(GraphemesReversed(s, true), ElementsAre("\r", "\r\n", "\n"));
  }
}

// ---- tests/test.rs quickcheck properties, on the test corpus ---------------

TEST(GraphemesTest, ForwardReverseAndJoin) {
  std::vector<std::string> corpus = {"",
                                     "a",
                                     "abc",
                                     "\r\n",
                                     "\n\r\n\r",
                                     "a̐éö̲\r\n",
                                     "हिन्दी",
                                     "\U0001F1F7\U0001F1F8\U0001F1EE\U0001F1F4",
                                     "\U0001F1F7\U0001F1F8\U0001F1EE"};
  for (const GraphemeTestSame& t : GraphemeTestSameCases()) {
    corpus.push_back(t.input);
  }
  for (const GraphemeTestDiff& t : GraphemeTestDiffCases()) {
    corpus.push_back(t.input);
  }
  // Concatenations stress the pre-context handling over longer strings.
  std::string all;
  for (const std::string& s : corpus) all += s;
  corpus.push_back(all);

  for (const std::string& s : corpus) {
    SCOPED_TRACE(s);
    for (const bool is_extended : {true, false}) {
      // quickcheck_forward_reverse_graphemes_{extended,legacy}
      const std::vector<std::string> a = ToStrings(Graphemes(s, is_extended));
      const std::vector<std::string> b =
          Reversed(GraphemesReversed(s, is_extended));
      EXPECT_EQ(a, b) << is_extended;
      // quickcheck_join_graphemes
      std::string joined;
      for (const std::string& g : a) joined += g;
      EXPECT_EQ(joined, s);
    }
  }
}

TEST(GraphemesTest, EmptyAndAscii) {
  EXPECT_THAT(Graphemes("", true), IsEmpty());
  EXPECT_THAT(Graphemes("", false), IsEmpty());
  EXPECT_THAT(Graphemes("abc", true), ElementsAre("a", "b", "c"));
  EXPECT_THAT(GraphemeIndices("", true), IsEmpty());
  // The views alias the input.
  const absl::string_view s = "x\u0301y";
  const std::vector<absl::string_view> g = Graphemes(s, true);
  ASSERT_EQ(g.size(), 2);
  EXPECT_EQ(g[0].data(), s.data());
  EXPECT_EQ(g[1].data(), s.data() + 3);
}

}  // namespace
}  // namespace tokenizers::unicode
