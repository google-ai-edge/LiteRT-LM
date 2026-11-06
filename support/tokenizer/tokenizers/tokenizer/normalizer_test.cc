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
// C++ port of the `#[cfg(test)]` module of `src/tokenizer/normalizer.rs` from
// the HuggingFace `tokenizers` crate v0.23.2
// (https://github.com/huggingface/tokenizers, Apache License 2.0,
// Copyright 2020 The HuggingFace Team). Kept line-by-line close to the Rust
// original on purpose; see
// support/tokenizer/tokenizers/README.md.

#include "support/tokenizer/tokenizers/tokenizer/normalizer.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/tokenizer/pattern.h"
#include "support/tokenizer/tokenizers/unicode/categories.h"
#include "support/tokenizer/tokenizers/utils/regex.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace tokenizers {
namespace {

using PairVec = std::vector<std::pair<size_t, size_t>>;

TEST(NormalizerTest, TestLenRangeInclusive) {
  // `Range::Original(3..=7)` in half-open form is `3..8`.
  Range range = Range::Original(3, 8);
  EXPECT_EQ(range.Len(), 5u);
}

TEST(NormalizerTest, TestLenRangeExclusive) {
  Range range = Range::Original(3, 7);
  EXPECT_EQ(range.Len(), 4u);
}

TEST(NormalizerTest, NfdAddsNewChars) {
  NormalizedString n = NormalizedString::From("élégant");
  n.Nfd();
  EXPECT_EQ(n.Alignments(), (PairVec{{0, 2},
                                     {0, 2},
                                     {0, 2},
                                     {2, 3},
                                     {3, 5},
                                     {3, 5},
                                     {3, 5},
                                     {5, 6},
                                     {6, 7},
                                     {7, 8},
                                     {8, 9}}));
  EXPECT_EQ(n.AlignmentsOriginal(), (PairVec{{0, 3},
                                             {0, 3},
                                             {3, 4},
                                             {4, 7},
                                             {4, 7},
                                             {7, 8},
                                             {8, 9},
                                             {9, 10},
                                             {10, 11}}));
}

TEST(NormalizerTest, RemoveCharsAddedByNfd) {
  NormalizedString n = NormalizedString::From("élégant");
  n.Nfd().Filter([](char32_t c) { return !unicode::IsMarkNonspacing(c); });

  EXPECT_EQ(n.Get(), "elegant");
  EXPECT_EQ(n.Alignments(),
            (PairVec{{0, 2}, {2, 3}, {3, 5}, {5, 6}, {6, 7}, {7, 8}, {8, 9}}));
  EXPECT_EQ(n.AlignmentsOriginal(), (PairVec{{0, 1},
                                             {0, 1},
                                             {1, 2},
                                             {2, 3},
                                             {2, 3},
                                             {3, 4},
                                             {4, 5},
                                             {5, 6},
                                             {6, 7}}));
}

TEST(NormalizerTest, RemoveChars) {
  NormalizedString n = NormalizedString::From("élégant");
  n.Filter([](char32_t c) { return c != U'n'; });
  EXPECT_EQ(n.Get(), "élégat");
  EXPECT_EQ(n.Alignments(), (PairVec{{0, 2},
                                     {0, 2},
                                     {2, 3},
                                     {3, 5},
                                     {3, 5},
                                     {5, 6},
                                     {6, 7},
                                     // Skipped range
                                     {8, 9}}));
  EXPECT_EQ(n.AlignmentsOriginal(), (PairVec{{0, 2},
                                             {0, 2},
                                             {2, 3},
                                             {3, 5},
                                             {3, 5},
                                             {5, 6},
                                             {6, 7},
                                             {7, 7},  // Eaten n
                                             {7, 8}}));
}

TEST(NormalizerTest, MixedAdditionAndRemoval) {
  NormalizedString n = NormalizedString::From("élégant");
  n.Nfd().Filter(
      [](char32_t c) { return !unicode::IsMarkNonspacing(c) && c != U'n'; });
  EXPECT_EQ(n.Get(), "elegat");
  EXPECT_EQ(n.Alignments(),
            (PairVec{{0, 2}, {2, 3}, {3, 5}, {5, 6}, {6, 7}, {8, 9}}));
  EXPECT_EQ(n.AlignmentsOriginal(), (PairVec{{0, 1},
                                             {0, 1},
                                             {1, 2},
                                             {2, 3},
                                             {2, 3},
                                             {3, 4},  // g
                                             {4, 5},  // a
                                             {5, 5},  // Eaten n
                                             {5, 6}}));
}

TEST(NormalizerTest, RangeConversion) {
  NormalizedString n = NormalizedString::From("    __Hello__   ");
  n.Filter([](char32_t c) { return !unicode::IsWhitespace(c); }).Lowercase();
  std::optional<std::pair<size_t, size_t>> hello_n =
      n.ConvertOffsets(Range::Original(6, 11));
  ASSERT_EQ(hello_n, (std::pair<size_t, size_t>(2, 7)));
  EXPECT_EQ(n.GetRange(Range::Normalized(hello_n->first, hello_n->second)),
            "hello");
  EXPECT_EQ(
      n.GetRangeOriginal(Range::Normalized(hello_n->first, hello_n->second)),
      "Hello");
  EXPECT_EQ(n.GetRange(Range::Original(6, 11)), "hello");
  EXPECT_EQ(n.GetRangeOriginal(Range::Original(6, 11)), "Hello");

  // Make sure we get None only in specific cases
  EXPECT_EQ(n.ConvertOffsets(Range::Original(0, 0)),
            (std::pair<size_t, size_t>(0, 0)));
  EXPECT_EQ(n.ConvertOffsets(Range::Original(3, 3)),
            (std::pair<size_t, size_t>(3, 3)));
  EXPECT_EQ(n.ConvertOffsets(Range::OriginalFrom(15)),
            (std::pair<size_t, size_t>(9, 9)));
  EXPECT_EQ(n.ConvertOffsets(Range::OriginalFrom(16)),
            (std::pair<size_t, size_t>(16, 16)));
  EXPECT_EQ(n.ConvertOffsets(Range::OriginalFrom(17)), std::nullopt);
  EXPECT_EQ(n.ConvertOffsets(Range::Normalized(0, 0)),
            (std::pair<size_t, size_t>(0, 0)));
  EXPECT_EQ(n.ConvertOffsets(Range::Normalized(3, 3)),
            (std::pair<size_t, size_t>(3, 3)));
  EXPECT_EQ(n.ConvertOffsets(Range::NormalizedFrom(9)),
            (std::pair<size_t, size_t>(9, 9)));
  EXPECT_EQ(n.ConvertOffsets(Range::NormalizedFrom(10)), std::nullopt);
}

TEST(NormalizerTest, OriginalRange) {
  NormalizedString n = NormalizedString::From("Hello_______ World!");
  n.Filter([](char32_t c) { return c != U'_'; }).Lowercase();
  absl::string_view world_n = *n.GetRange(Range::Normalized(6, 11));
  absl::string_view world_o = *n.GetRangeOriginal(Range::Normalized(6, 11));
  EXPECT_EQ(world_n, "world");
  EXPECT_EQ(world_o, "World");
  std::pair<size_t, size_t> conv = *n.ConvertOffsets(Range::Normalized(6, 11));
  Range original_range = Range::Original(conv.first, conv.second);
  EXPECT_EQ(*n.GetRange(original_range), "world");
  EXPECT_EQ(*n.GetRangeOriginal(original_range), "World");
  EXPECT_EQ(original_range.IntoFullRange(n.LenOriginal()),
            (std::pair<size_t, size_t>(13, 18)));
}

TEST(NormalizerTest, AddedAroundEdges) {
  NormalizedString n = NormalizedString::From("Hello");
  n.Transform(
      {
          {U' ', 1},
          {U'H', 0},
          {U'e', 0},
          {U'l', 0},
          {U'l', 0},
          {U'o', 0},
          {U' ', 1},
      },
      0);

  EXPECT_EQ(n.Get(), " Hello ");
  EXPECT_EQ(n.GetRangeOriginal(Range::Normalized(1, n.Len() - 1)), "Hello");
}

TEST(NormalizerTest, AddedCharactersAlignment) {
  NormalizedString n = NormalizedString::From("野口 No");
  std::vector<std::pair<char32_t, int64_t>> transforms;
  n.ForEach([&transforms](char32_t c) {
    if (static_cast<size_t>(c) > 0x4E00) {
      transforms.push_back({U' ', 0});
      transforms.push_back({c, 1});
      transforms.push_back({U' ', 1});
    } else {
      transforms.push_back({c, 0});
    }
  });
  n.Transform(transforms, 0);

  EXPECT_EQ(n, NormalizedString("野口 No", " 野  口  No",
                                {
                                    {0, 3},
                                    {0, 3},
                                    {0, 3},
                                    {0, 3},
                                    {0, 3},
                                    {3, 6},
                                    {3, 6},
                                    {3, 6},
                                    {3, 6},
                                    {3, 6},
                                    {6, 7},
                                    {7, 8},
                                    {8, 9},
                                },
                                0));
  EXPECT_EQ(n.AlignmentsOriginal(), (PairVec{{0, 5},
                                             {0, 5},
                                             {0, 5},
                                             {5, 10},
                                             {5, 10},
                                             {5, 10},
                                             {10, 11},
                                             {11, 12},
                                             {12, 13}}));
}

TEST(NormalizerTest, RemoveAtBeginning) {
  NormalizedString n = NormalizedString::From("     Hello");
  n.Filter([](char32_t c) { return !unicode::IsWhitespace(c); });
  EXPECT_EQ(n.GetRangeOriginal(Range::Normalized(1, 5)), "ello");
  EXPECT_EQ(n.GetRangeOriginal(Range::Normalized(0, n.Len())), "Hello");
}

TEST(NormalizerTest, RemoveAtEnd) {
  NormalizedString n = NormalizedString::From("Hello    ");
  n.Filter([](char32_t c) { return !unicode::IsWhitespace(c); });
  EXPECT_EQ(n.GetRangeOriginal(Range::Normalized(0, 4)), "Hell");
  EXPECT_EQ(n.GetRangeOriginal(Range::Normalized(0, n.Len())), "Hello");
}

TEST(NormalizerTest, RemovedAroundBothEdges) {
  NormalizedString n = NormalizedString::From("  Hello  ");
  n.Filter([](char32_t c) { return !unicode::IsWhitespace(c); });
  EXPECT_EQ(n.Get(), "Hello");

  EXPECT_EQ(n.GetRangeOriginal(Range::Normalized(0, 5)), "Hello");
  EXPECT_EQ(n.GetRangeOriginal(Range::Normalized(1, 4)), "ell");
}

TEST(NormalizerTest, Lstrip) {
  NormalizedString n = NormalizedString::From("  This is an example  ");
  n.Lstrip();
  EXPECT_EQ(n.Get(), "This is an example  ");
  EXPECT_EQ(n.GetRangeOriginal(Range::Normalized(0, n.Len())),
            "This is an example  ");
}

TEST(NormalizerTest, Rstrip) {
  NormalizedString n = NormalizedString::From("  This is an example  ");
  n.Rstrip();
  EXPECT_EQ(n.Get(), "  This is an example");
  EXPECT_EQ(n.GetRangeOriginal(Range::Normalized(0, n.Len())),
            "  This is an example");
}

TEST(NormalizerTest, Strip) {
  NormalizedString n = NormalizedString::From("  This is an example  ");
  n.Strip();
  EXPECT_EQ(n.Get(), "This is an example");
  EXPECT_EQ(n.GetRangeOriginal(Range::Normalized(0, n.Len())),
            "This is an example");
}

TEST(NormalizerTest, StripUnicode) {
  NormalizedString n = NormalizedString::From("  你好asa \n");
  n.Strip();
  EXPECT_EQ(n.Get(), "你好asa");
  EXPECT_EQ(n.GetRangeOriginal(Range::Normalized(0, n.Len())), "你好asa");
}

TEST(NormalizerTest, Prepend) {
  NormalizedString n = NormalizedString::From("there");
  n.Prepend("Hey ");
  EXPECT_EQ(n.Get(), "Hey there");
  EXPECT_EQ(n.Alignments(), (PairVec{{0, 1},
                                     {0, 1},
                                     {0, 1},
                                     {0, 1},
                                     {0, 1},
                                     {1, 2},
                                     {2, 3},
                                     {3, 4},
                                     {4, 5}}));
  EXPECT_EQ(n.ConvertOffsets(Range::Normalized(0, 4)),
            (std::pair<size_t, size_t>(0, 1)));
}

TEST(NormalizerTest, Append) {
  NormalizedString n = NormalizedString::From("Hey");
  n.Append(" there");
  EXPECT_EQ(n.Get(), "Hey there");
  EXPECT_EQ(n.Alignments(), (PairVec{{0, 1},
                                     {1, 2},
                                     {2, 3},
                                     {2, 3},
                                     {2, 3},
                                     {2, 3},
                                     {2, 3},
                                     {2, 3},
                                     {2, 3}}));
  EXPECT_EQ(n.ConvertOffsets(Range::Normalized(3, 6)),
            (std::pair<size_t, size_t>(2, 3)));
}

TEST(NormalizerTest, GetRange) {
  std::string s = "Hello my name is John 👋";
  EXPECT_EQ(GetRangeOf(s, 0, std::nullopt), s);
  EXPECT_EQ(GetRangeOf(s, 17, std::nullopt), "John 👋");
}

TEST(NormalizerTest, Slice) {
  NormalizedString s = NormalizedString::From("𝔾𝕠𝕠𝕕 𝕞𝕠𝕣𝕟𝕚𝕟𝕘");
  s.Nfkc();

  NormalizedString original_slice = *s.Slice(Range::Original(0, 4));
  EXPECT_EQ(original_slice.Get(), "G");
  EXPECT_EQ(original_slice.GetOriginal(), "𝔾");

  NormalizedString normalized_slice = *s.Slice(Range::Normalized(0, 4));
  EXPECT_EQ(normalized_slice.Get(), "Good");
  EXPECT_EQ(normalized_slice.GetOriginal(), "𝔾𝕠𝕠𝕕");

  // Make sure the sliced NormalizedString is still aligned as expected
  NormalizedString s2 = NormalizedString::From("   Good Morning!   ");
  s2.Strip();

  // If we keep the whole slice
  NormalizedString slice = *s2.Slice(Range::OriginalAll());
  EXPECT_EQ(slice.GetRangeOriginal(Range::Normalized(0, 4)), "Good");
  slice = *s2.Slice(Range::NormalizedAll());
  EXPECT_EQ(slice.GetRangeOriginal(Range::Normalized(0, 4)), "Good");

  // If we keep after the modified piece
  slice = *s2.Slice(Range::Original(4, 15));
  EXPECT_EQ(slice.GetRangeOriginal(Range::Normalized(0, 3)), "ood");

  // If we keep only the modified piece
  slice = *s2.Slice(Range::Original(3, 16));
  EXPECT_EQ(slice.GetRangeOriginal(Range::Normalized(0, 4)), "Good");
}

TEST(NormalizerTest, Replace) {
  // Simple
  NormalizedString s = NormalizedString::From(" Hello   friend ");
  ASSERT_OK(s.Replace(U' ', "_"));
  EXPECT_EQ(s.Get(), "_Hello___friend_");
  s = NormalizedString::From("aaaab");
  ASSERT_OK(s.Replace(U'a', "b"));
  EXPECT_EQ(s.Get(), "bbbbb");

  // Overlapping
  s = NormalizedString::From("aaaab");
  ASSERT_OK(s.Replace("aaa", "b"));
  EXPECT_EQ(s.Get(), "bab");

  // Regex
  s = NormalizedString::From(" Hello   friend ");
  ASSERT_OK_AND_ASSIGN(SysRegex re, SysRegex::New(R"(\s+)"));
  ASSERT_OK(s.Replace(&re, "_"));
  EXPECT_EQ(s.Get(), "_Hello_friend_");
}

TEST(NormalizerTest, Split) {
  NormalizedString s = NormalizedString::From("The-final--countdown");

  auto test = [&s](SplitDelimiterBehavior behavior,
                   const std::vector<absl::string_view>& expected) {
    ASSERT_OK_AND_ASSIGN(std::vector<NormalizedString> splits,
                         s.Split(U'-', behavior));
    std::vector<absl::string_view> got;
    got.reserve(splits.size());
    for (const NormalizedString& n : splits) {
      got.push_back(n.Get());
    }
    EXPECT_EQ(got, expected);
  };

  test(SplitDelimiterBehavior::kRemoved, {"The", "final", "countdown"});
  test(SplitDelimiterBehavior::kIsolated,
       {"The", "-", "final", "-", "-", "countdown"});
  test(SplitDelimiterBehavior::kMergedWithPrevious,
       {"The-", "final-", "-", "countdown"});
  test(SplitDelimiterBehavior::kMergedWithNext,
       {"The", "-final", "-", "-countdown"});
  test(SplitDelimiterBehavior::kContiguous,
       {"The", "-", "final", "--", "countdown"});
}

TEST(NormalizerTest, TransformRangeSingleBytes) {
  const NormalizedString s = NormalizedString::From("Hello friend");

  // Removing at the beginning
  NormalizedString current = s;
  current.TransformRange(Range::Original(0, 4), {{U'Y', 0}}, 3);
  EXPECT_EQ(current, NormalizedString("Hello friend", "Yo friend",
                                      {{3, 4},
                                       {4, 5},
                                       {5, 6},
                                       {6, 7},
                                       {7, 8},
                                       {8, 9},
                                       {9, 10},
                                       {10, 11},
                                       {11, 12}},
                                      0));
  EXPECT_EQ(current.AlignmentsOriginal(), (PairVec{{0, 0},
                                                   {0, 0},
                                                   {0, 0},
                                                   {0, 1},
                                                   {1, 2},
                                                   {2, 3},
                                                   {3, 4},
                                                   {4, 5},
                                                   {5, 6},
                                                   {6, 7},
                                                   {7, 8},
                                                   {8, 9}}));

  // Removing in the middle
  current = s;
  current.TransformRange(Range::Original(3, 10),
                         {{U'_', 0}, {U'F', 0}, {U'R', -2}}, 2);
  EXPECT_EQ(
      current,
      NormalizedString(
          "Hello friend", "Hel_FRnd",
          {{0, 1}, {1, 2}, {2, 3}, {5, 6}, {6, 7}, {7, 8}, {10, 11}, {11, 12}},
          0));
  EXPECT_EQ(current.AlignmentsOriginal(), (PairVec{{0, 1},
                                                   {1, 2},
                                                   {2, 3},
                                                   {3, 3},
                                                   {3, 3},
                                                   {3, 4},
                                                   {4, 5},
                                                   {5, 6},
                                                   {6, 6},
                                                   {6, 6},
                                                   {6, 7},
                                                   {7, 8}}));

  // Removing at the end
  current = s;
  current.TransformRange(Range::OriginalFrom(5), {{U'_', 0}, {U'F', -5}}, 0);
  EXPECT_EQ(current,
            NormalizedString(
                "Hello friend", "Hello_F",
                {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6}, {6, 7}}, 0));
  EXPECT_EQ(current.AlignmentsOriginal(), (PairVec{{0, 1},
                                                   {1, 2},
                                                   {2, 3},
                                                   {3, 4},
                                                   {4, 5},
                                                   {5, 6},
                                                   {6, 7},
                                                   {7, 7},
                                                   {7, 7},
                                                   {7, 7},
                                                   {7, 7},
                                                   {7, 7}}));

  // Adding at the beginning
  current = s;
  current.TransformRange(Range::Original(0, 1), {{U'H', 1}, {U'H', 0}}, 0);
  EXPECT_EQ(current, NormalizedString("Hello friend", "HHello friend",
                                      {{0, 0},
                                       {0, 1},
                                       {1, 2},
                                       {2, 3},
                                       {3, 4},
                                       {4, 5},
                                       {5, 6},
                                       {6, 7},
                                       {7, 8},
                                       {8, 9},
                                       {9, 10},
                                       {10, 11},
                                       {11, 12}},
                                      0));
  EXPECT_EQ(current.AlignmentsOriginal(), (PairVec{{1, 2},
                                                   {2, 3},
                                                   {3, 4},
                                                   {4, 5},
                                                   {5, 6},
                                                   {6, 7},
                                                   {7, 8},
                                                   {8, 9},
                                                   {9, 10},
                                                   {10, 11},
                                                   {11, 12},
                                                   {12, 13}}));

  // Equivalent to the previous one
  current = s;
  current.TransformRange(Range::Original(0, 0), {{U'H', 1}}, 0);
  EXPECT_EQ(current, NormalizedString("Hello friend", "HHello friend",
                                      {{0, 0},
                                       {0, 1},
                                       {1, 2},
                                       {2, 3},
                                       {3, 4},
                                       {4, 5},
                                       {5, 6},
                                       {6, 7},
                                       {7, 8},
                                       {8, 9},
                                       {9, 10},
                                       {10, 11},
                                       {11, 12}},
                                      0));
  EXPECT_EQ(current.AlignmentsOriginal(), (PairVec{{1, 2},
                                                   {2, 3},
                                                   {3, 4},
                                                   {4, 5},
                                                   {5, 6},
                                                   {6, 7},
                                                   {7, 8},
                                                   {8, 9},
                                                   {9, 10},
                                                   {10, 11},
                                                   {11, 12},
                                                   {12, 13}}));

  // Adding as part of the first character
  current = s;
  current.TransformRange(Range::Original(0, 1), {{U'H', 0}, {U'H', 1}}, 0);
  EXPECT_EQ(current, NormalizedString("Hello friend", "HHello friend",
                                      {{0, 1},
                                       {0, 1},
                                       {1, 2},
                                       {2, 3},
                                       {3, 4},
                                       {4, 5},
                                       {5, 6},
                                       {6, 7},
                                       {7, 8},
                                       {8, 9},
                                       {9, 10},
                                       {10, 11},
                                       {11, 12}},
                                      0));
  EXPECT_EQ(current.AlignmentsOriginal(), (PairVec{{0, 2},
                                                   {2, 3},
                                                   {3, 4},
                                                   {4, 5},
                                                   {5, 6},
                                                   {6, 7},
                                                   {7, 8},
                                                   {8, 9},
                                                   {9, 10},
                                                   {10, 11},
                                                   {11, 12},
                                                   {12, 13}}));

  // Adding in the middle
  current = s;
  current.TransformRange(Range::Original(5, 6),
                         {{U'_', 0}, {U'm', 1}, {U'y', 1}, {U'_', 1}}, 0);
  EXPECT_EQ(current, NormalizedString("Hello friend", "Hello_my_friend",
                                      {{0, 1},
                                       {1, 2},
                                       {2, 3},
                                       {3, 4},
                                       {4, 5},
                                       {5, 6},
                                       {5, 6},
                                       {5, 6},
                                       {5, 6},
                                       {6, 7},
                                       {7, 8},
                                       {8, 9},
                                       {9, 10},
                                       {10, 11},
                                       {11, 12}},
                                      0));
  EXPECT_EQ(current.AlignmentsOriginal(), (PairVec{{0, 1},
                                                   {1, 2},
                                                   {2, 3},
                                                   {3, 4},
                                                   {4, 5},
                                                   {5, 9},
                                                   {9, 10},
                                                   {10, 11},
                                                   {11, 12},
                                                   {12, 13},
                                                   {13, 14},
                                                   {14, 15}}));

  // Adding at the end
  current = s;
  current.TransformRange(Range::OriginalFrom(11),
                         {{U'd', 0}, {U'_', 1}, {U'!', 1}}, 0);
  EXPECT_EQ(current, NormalizedString("Hello friend", "Hello friend_!",
                                      {{0, 1},
                                       {1, 2},
                                       {2, 3},
                                       {3, 4},
                                       {4, 5},
                                       {5, 6},
                                       {6, 7},
                                       {7, 8},
                                       {8, 9},
                                       {9, 10},
                                       {10, 11},
                                       {11, 12},
                                       {11, 12},
                                       {11, 12}},
                                      0));
  EXPECT_EQ(current.AlignmentsOriginal(), (PairVec{{0, 1},
                                                   {1, 2},
                                                   {2, 3},
                                                   {3, 4},
                                                   {4, 5},
                                                   {5, 6},
                                                   {6, 7},
                                                   {7, 8},
                                                   {8, 9},
                                                   {9, 10},
                                                   {10, 11},
                                                   {11, 14}}));
}

TEST(NormalizerTest, TransformRangeMultipleBytes) {
  const NormalizedString s = NormalizedString::From("𝔾𝕠𝕠𝕕");

  // Removing at the beginning
  NormalizedString current = s;
  current.TransformRange(Range::Original(0, 8), {{U'G', -1}}, 0);
  EXPECT_EQ(current, NormalizedString("𝔾𝕠𝕠𝕕", "G𝕠𝕕",
                                      {{0, 4},
                                       {8, 12},
                                       {8, 12},
                                       {8, 12},
                                       {8, 12},
                                       {12, 16},
                                       {12, 16},
                                       {12, 16},
                                       {12, 16}},
                                      0));
  EXPECT_EQ(current.AlignmentsOriginal(), (PairVec{{0, 1},
                                                   {0, 1},
                                                   {0, 1},
                                                   {0, 1},
                                                   {1, 1},
                                                   {1, 1},
                                                   {1, 1},
                                                   {1, 1},
                                                   {1, 5},
                                                   {1, 5},
                                                   {1, 5},
                                                   {1, 5},
                                                   {5, 9},
                                                   {5, 9},
                                                   {5, 9},
                                                   {5, 9}}));
  EXPECT_EQ(current.GetRange(Range::Original(0, 8)), "G");
  EXPECT_EQ(current.GetRange(Range::Original(0, 4)), "G");
  EXPECT_EQ(current.GetRangeOriginal(Range::Original(0, 4)), "𝔾");
  EXPECT_EQ(current.GetRangeOriginal(Range::Original(0, 8)), "𝔾𝕠");

  // Removing in the middle
  current = s;
  current.TransformRange(Range::Original(4, 12), {{U'o', -1}}, 0);
  EXPECT_EQ(current, NormalizedString("𝔾𝕠𝕠𝕕", "𝔾o𝕕",
                                      {{0, 4},
                                       {0, 4},
                                       {0, 4},
                                       {0, 4},
                                       {4, 8},
                                       {12, 16},
                                       {12, 16},
                                       {12, 16},
                                       {12, 16}},
                                      0));
  EXPECT_EQ(current.AlignmentsOriginal(), (PairVec{{0, 4},
                                                   {0, 4},
                                                   {0, 4},
                                                   {0, 4},
                                                   {4, 5},
                                                   {4, 5},
                                                   {4, 5},
                                                   {4, 5},
                                                   {5, 5},
                                                   {5, 5},
                                                   {5, 5},
                                                   {5, 5},
                                                   {5, 9},
                                                   {5, 9},
                                                   {5, 9},
                                                   {5, 9}}));

  // Removing at the end
  current = s;
  current.TransformRange(Range::OriginalFrom(12), {{U'd', 0}, {U'!', 1}}, 0);
  EXPECT_EQ(current, NormalizedString("𝔾𝕠𝕠𝕕", "𝔾𝕠𝕠d!",
                                      {{0, 4},
                                       {0, 4},
                                       {0, 4},
                                       {0, 4},
                                       {4, 8},
                                       {4, 8},
                                       {4, 8},
                                       {4, 8},
                                       {8, 12},
                                       {8, 12},
                                       {8, 12},
                                       {8, 12},
                                       {12, 16},
                                       {12, 16}},
                                      0));

  // Adding at the beginning
  current = s;
  current.TransformRange(Range::Original(0, 4), {{U'_', 1}, {U'𝔾', 0}}, 0);
  EXPECT_EQ(current, NormalizedString("𝔾𝕠𝕠𝕕", "_𝔾𝕠𝕠𝕕",
                                      {{0, 0},
                                       {0, 4},
                                       {0, 4},
                                       {0, 4},
                                       {0, 4},
                                       {4, 8},
                                       {4, 8},
                                       {4, 8},
                                       {4, 8},
                                       {8, 12},
                                       {8, 12},
                                       {8, 12},
                                       {8, 12},
                                       {12, 16},
                                       {12, 16},
                                       {12, 16},
                                       {12, 16}},
                                      0));
  EXPECT_EQ(current.AlignmentsOriginal(), (PairVec{{1, 5},
                                                   {1, 5},
                                                   {1, 5},
                                                   {1, 5},
                                                   {5, 9},
                                                   {5, 9},
                                                   {5, 9},
                                                   {5, 9},
                                                   {9, 13},
                                                   {9, 13},
                                                   {9, 13},
                                                   {9, 13},
                                                   {13, 17},
                                                   {13, 17},
                                                   {13, 17},
                                                   {13, 17}}));
  EXPECT_EQ(current.GetRange(Range::Original(0, 8)), "𝔾𝕠");
  EXPECT_EQ(current.GetRange(Range::Original(0, 4)), "𝔾");
  EXPECT_EQ(current.GetRangeOriginal(Range::Original(0, 4)), "𝔾");
  EXPECT_EQ(current.GetRangeOriginal(Range::Original(0, 8)), "𝔾𝕠");

  // Equivalent to the previous one
  current = s;
  current.TransformRange(Range::Original(0, 0), {{U'_', 1}}, 0);
  EXPECT_EQ(current, NormalizedString("𝔾𝕠𝕠𝕕", "_𝔾𝕠𝕠𝕕",
                                      {{0, 0},
                                       {0, 4},
                                       {0, 4},
                                       {0, 4},
                                       {0, 4},
                                       {4, 8},
                                       {4, 8},
                                       {4, 8},
                                       {4, 8},
                                       {8, 12},
                                       {8, 12},
                                       {8, 12},
                                       {8, 12},
                                       {12, 16},
                                       {12, 16},
                                       {12, 16},
                                       {12, 16}},
                                      0));
  EXPECT_EQ(current.AlignmentsOriginal(), (PairVec{{1, 5},
                                                   {1, 5},
                                                   {1, 5},
                                                   {1, 5},
                                                   {5, 9},
                                                   {5, 9},
                                                   {5, 9},
                                                   {5, 9},
                                                   {9, 13},
                                                   {9, 13},
                                                   {9, 13},
                                                   {9, 13},
                                                   {13, 17},
                                                   {13, 17},
                                                   {13, 17},
                                                   {13, 17}}));
  EXPECT_EQ(current.GetRange(Range::Original(0, 8)), "𝔾𝕠");
  EXPECT_EQ(current.GetRange(Range::Original(0, 4)), "𝔾");
  EXPECT_EQ(current.GetRangeOriginal(Range::Original(0, 4)), "𝔾");
  EXPECT_EQ(current.GetRangeOriginal(Range::Original(0, 8)), "𝔾𝕠");

  // Adding as part of the first character
  current = s;
  current.TransformRange(Range::Original(0, 4), {{U'𝔾', 0}, {U'o', 1}}, 0);
  EXPECT_EQ(current, NormalizedString("𝔾𝕠𝕠𝕕", "𝔾o𝕠𝕠𝕕",
                                      {{0, 4},
                                       {0, 4},
                                       {0, 4},
                                       {0, 4},
                                       {0, 4},
                                       {4, 8},
                                       {4, 8},
                                       {4, 8},
                                       {4, 8},
                                       {8, 12},
                                       {8, 12},
                                       {8, 12},
                                       {8, 12},
                                       {12, 16},
                                       {12, 16},
                                       {12, 16},
                                       {12, 16}},
                                      0));
  EXPECT_EQ(current.AlignmentsOriginal(), (PairVec{{0, 5},
                                                   {0, 5},
                                                   {0, 5},
                                                   {0, 5},
                                                   {5, 9},
                                                   {5, 9},
                                                   {5, 9},
                                                   {5, 9},
                                                   {9, 13},
                                                   {9, 13},
                                                   {9, 13},
                                                   {9, 13},
                                                   {13, 17},
                                                   {13, 17},
                                                   {13, 17},
                                                   {13, 17}}));
  EXPECT_EQ(current.GetRange(Range::Original(0, 8)), "𝔾o𝕠");
  EXPECT_EQ(current.GetRange(Range::Original(0, 4)), "𝔾o");
  EXPECT_EQ(current.GetRangeOriginal(Range::Original(0, 4)), "𝔾");
  EXPECT_EQ(current.GetRangeOriginal(Range::Original(0, 8)), "𝔾𝕠");

  // Adding in the middle
  current = s;
  current.TransformRange(Range::Original(4, 8),
                         {{U'𝕠', 0}, {U'o', 1}, {U'o', 1}, {U'o', 1}}, 0);
  EXPECT_EQ(current, NormalizedString("𝔾𝕠𝕠𝕕", "𝔾𝕠ooo𝕠𝕕",
                                      {{0, 4},
                                       {0, 4},
                                       {0, 4},
                                       {0, 4},
                                       {4, 8},
                                       {4, 8},
                                       {4, 8},
                                       {4, 8},
                                       {4, 8},
                                       {4, 8},
                                       {4, 8},
                                       {8, 12},
                                       {8, 12},
                                       {8, 12},
                                       {8, 12},
                                       {12, 16},
                                       {12, 16},
                                       {12, 16},
                                       {12, 16}},
                                      0));
  EXPECT_EQ(current.AlignmentsOriginal(), (PairVec{{0, 4},
                                                   {0, 4},
                                                   {0, 4},
                                                   {0, 4},
                                                   {4, 11},
                                                   {4, 11},
                                                   {4, 11},
                                                   {4, 11},
                                                   {11, 15},
                                                   {11, 15},
                                                   {11, 15},
                                                   {11, 15},
                                                   {15, 19},
                                                   {15, 19},
                                                   {15, 19},
                                                   {15, 19}}));

  // Adding at the end
  current = s;
  current.TransformRange(Range::OriginalFrom(16), {{U'!', 1}}, 0);
  EXPECT_EQ(current, NormalizedString("𝔾𝕠𝕠𝕕", "𝔾𝕠𝕠𝕕!",
                                      {{0, 4},
                                       {0, 4},
                                       {0, 4},
                                       {0, 4},
                                       {4, 8},
                                       {4, 8},
                                       {4, 8},
                                       {4, 8},
                                       {8, 12},
                                       {8, 12},
                                       {8, 12},
                                       {8, 12},
                                       {12, 16},
                                       {12, 16},
                                       {12, 16},
                                       {12, 16},
                                       {12, 16}},
                                      0));
  EXPECT_EQ(current.AlignmentsOriginal(), (PairVec{{0, 4},
                                                   {0, 4},
                                                   {0, 4},
                                                   {0, 4},
                                                   {4, 8},
                                                   {4, 8},
                                                   {4, 8},
                                                   {4, 8},
                                                   {8, 12},
                                                   {8, 12},
                                                   {8, 12},
                                                   {8, 12},
                                                   {12, 17},
                                                   {12, 17},
                                                   {12, 17},
                                                   {12, 17}}));
}

TEST(NormalizerTest, TransformCheck) {
  NormalizedString s = NormalizedString::From("abc…");
  s.Nfkd();
  std::vector<std::pair<char32_t, int64_t>> transforms = {
      {U'a', -2}, {U'.', 0}, {U'.', 0}, {U'.', 0}};
  s.Transform(transforms, 0);
  s.Lowercase();
  EXPECT_EQ(s.Get(), "a...");
}

TEST(NormalizerTest, TestAppendAfterClear) {
  NormalizedString n = NormalizedString::From("Hello");
  EXPECT_EQ(n.Get(), "Hello");

  n.Clear();
  EXPECT_EQ(n.Get(), "");

  n.Append(" World");
  EXPECT_EQ(n.Get(), " World");

  EXPECT_EQ(n.LenOriginal(), 5u);
  EXPECT_EQ(n.Len(), 6u);

  EXPECT_EQ(n.GetRangeOriginal(Range::Original(0, 5)), "Hello");
  EXPECT_EQ(n.GetRangeOriginal(Range::Normalized(0, 6)), "");

  EXPECT_EQ(n.GetRange(Range::Normalized(0, 6)), " World");
}

}  // namespace
}  // namespace tokenizers
