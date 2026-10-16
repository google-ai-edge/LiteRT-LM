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

#include "omni/text2speech/kokoro/cjk_blob.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "omni/text2speech/kokoro/cjk_test_blobs.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace litert::omni::tts::kokoro {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::IsEmpty;

// Packs `keys` into the (offset index, NUL-separated blob) pair the readers
// expect. Callers pass them already sorted, as the packer emits them.
std::pair<std::string, std::string> PackStrings(
    const std::vector<std::string>& keys) {
  std::string index;
  std::string blob;
  for (const std::string& key : keys) {
    AppendUint32(blob.size(), index);
    blob += key;
    blob.push_back('\0');
  }
  AppendUint32(blob.size(), index);
  return {index, blob};
}

// A Darts 0.32 double array holding "a" -> 5 and "ab" -> 7, laid out by hand.
//
// Traversal carries the parent's base value. From `base[0] = 1`, the child for
// 'a' (0x61) is at 1 + 0x61 + 1 = 99, whose base 100 becomes the new state; the
// value for "a" hangs off slot 100 itself. The child for 'b' (0x62) is then at
// 100 + 0x62 + 1 = 199, whose base 200 carries the value for "ab".
std::string BuildDoubleArray() {
  constexpr int kUnits = 201;
  std::vector<int32_t> base(kUnits, 0);
  std::vector<uint32_t> check(kUnits, 0xFFFFFFFFu);

  base[0] = 1;
  check[99] = 1;
  base[99] = 100;
  check[100] = 100;
  base[100] = -(5 << 8 | 1) - 1;  // one token at index 5
  check[199] = 100;
  base[199] = 200;
  check[200] = 200;
  base[200] = -(7 << 8 | 2) - 1;  // two tokens starting at index 7

  std::string units;
  for (int i = 0; i < kUnits; ++i) {
    AppendUint32(static_cast<uint32_t>(base[i]), units);
    AppendUint32(check[i], units);
  }
  return units;
}

MATCHER_P3(IsMatch, token_index, token_count, length, "") {
  return arg.token_index == token_index && arg.token_count == token_count &&
         arg.length == length;
}

TEST(CjkBlobTest, ReadsSectionsByName) {
  const std::string blob =
      BuildCjkBlob({{"w-key", "hello"}, {"p-val", "world!!"}});
  ASSERT_OK_AND_ASSIGN(const CjkBlob parsed, CjkBlob::Create(blob));

  EXPECT_TRUE(parsed.Contains("w-key"));
  EXPECT_TRUE(parsed.Contains("p-val"));
  EXPECT_FALSE(parsed.Contains("missing"));
  EXPECT_THAT(parsed.Section("w-key"), IsOkAndHolds("hello"));
  EXPECT_THAT(parsed.Section("p-val"), IsOkAndHolds("world!!"));
  EXPECT_THAT(
      parsed.Section("missing").status(),
      ::testing::Property(&absl::Status::code, absl::StatusCode::kNotFound));
}

TEST(CjkBlobTest, RejectsMalformedBlobs) {
  EXPECT_THAT(CjkBlob::Create("").status(),
              ::testing::Property(&absl::Status::code,
                                  absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(CjkBlob::Create(std::string(64, '\0')).status(),
              ::testing::Property(&absl::Status::code,
                                  absl::StatusCode::kInvalidArgument));

  // A section that runs off the end of the blob must be caught rather than
  // handed out as a view into whatever follows in memory.
  std::string truncated = BuildCjkBlob({{"w-key", "hello"}});
  truncated.resize(truncated.size() - 4);
  EXPECT_THAT(CjkBlob::Create(truncated).status(),
              ::testing::Property(&absl::Status::code,
                                  absl::StatusCode::kInvalidArgument));
}

TEST(SortedStringIndexTest, FindsExactKeys) {
  const auto [index, keys] = PackStrings({"一点", "中国", "你好", "语音"});
  ASSERT_OK_AND_ASSIGN(const SortedStringIndex sorted,
                       SortedStringIndex::Create(index, keys));

  EXPECT_EQ(sorted.Size(), 4);
  EXPECT_EQ(sorted.Key(0), "一点");
  EXPECT_EQ(sorted.Find("你好"), 2);
  EXPECT_EQ(sorted.Find("语音"), 3);
  EXPECT_EQ(sorted.Find("世界"), -1);
  EXPECT_EQ(sorted.Find(""), -1);
}

TEST(SortedStringIndexTest, AnswersThePrefixQuestionJiebaAsks) {
  // Building the DAG walks a fragment out one character at a time and stops as
  // soon as no entry starts with it, so the prefix test has to be exact about
  // partial multi-byte characters too.
  const auto [index, keys] = PackStrings({"中国", "中国人", "中心"});
  ASSERT_OK_AND_ASSIGN(const SortedStringIndex sorted,
                       SortedStringIndex::Create(index, keys));

  EXPECT_TRUE(sorted.HasPrefix("中"));
  EXPECT_TRUE(sorted.HasPrefix("中国"));
  EXPECT_TRUE(sorted.HasPrefix("中国人"));
  EXPECT_TRUE(sorted.HasPrefix(""));
  EXPECT_FALSE(sorted.HasPrefix("中国人民"));
  EXPECT_FALSE(sorted.HasPrefix("人"));

  // A key is a prefix of itself, but is not necessarily an entry.
  EXPECT_EQ(sorted.Find("中"), -1);
}

TEST(SortedStringIndexTest, RejectsMalformedIndexes) {
  EXPECT_THAT(SortedStringIndex::Create("", "").status(),
              ::testing::Property(&absl::Status::code,
                                  absl::StatusCode::kInvalidArgument));

  // Offsets must strictly ascend and stay inside the payload.
  std::string index;
  AppendUint32(0, index);
  AppendUint32(99, index);
  EXPECT_THAT(SortedStringIndex::Create(index, "ab").status(),
              ::testing::Property(&absl::Status::code,
                                  absl::StatusCode::kInvalidArgument));

  std::string duplicate_offsets;
  AppendUint32(0, duplicate_offsets);
  AppendUint32(2, duplicate_offsets);
  AppendUint32(2, duplicate_offsets);
  EXPECT_THAT(SortedStringIndex::Create(duplicate_offsets,
                                        absl::string_view("a\0b\0", 4))
                  .status(),
              ::testing::Property(&absl::Status::code,
                                  absl::StatusCode::kInvalidArgument));
}

TEST(StringValueTableTest, ReadsValuesByPosition) {
  const auto [index, values] = PackStrings({"ni↓xau↓", "ꭧʊ→ŋkwo↗"});
  ASSERT_OK_AND_ASSIGN(const StringValueTable table,
                       StringValueTable::Create(index, values));

  EXPECT_EQ(table.Size(), 2);
  EXPECT_EQ(table.Value(0), "ni↓xau↓");
  EXPECT_EQ(table.Value(1), "ꭧʊ→ŋkwo↗");
}

TEST(MecabDoubleArrayTest, FindsEveryPrefixOfTheQuery) {
  // `MecabDoubleArray` views into `units`, which must outlive it.
  const std::string units = BuildDoubleArray();
  ASSERT_OK_AND_ASSIGN(const MecabDoubleArray array,
                       MecabDoubleArray::Create(units));

  EXPECT_THAT(array.CommonPrefixSearch("abc"),
              ElementsAre(IsMatch(5, 1, 1), IsMatch(7, 2, 2)));
  EXPECT_THAT(array.CommonPrefixSearch("a"), ElementsAre(IsMatch(5, 1, 1)));
  EXPECT_THAT(array.CommonPrefixSearch("b"), IsEmpty());
  EXPECT_THAT(array.CommonPrefixSearch(""), IsEmpty());
}

TEST(MecabDoubleArrayTest, StaysInsideATruncatedArray) {
  // The blob is untrusted, and Darts itself does no bounds checking. A search
  // that walks off the end must come back empty rather than read out of range.
  std::string units = BuildDoubleArray();
  units.resize(8 * 100);
  ASSERT_OK_AND_ASSIGN(const MecabDoubleArray array,
                       MecabDoubleArray::Create(units));

  EXPECT_THAT(array.CommonPrefixSearch("abc"), IsEmpty());
}

TEST(MecabDoubleArrayTest, RejectsAPartialUnit) {
  EXPECT_THAT(MecabDoubleArray::Create(""),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(MecabDoubleArray::Create(std::string(12, '\0')),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

// Builds a `matrix.bin` whose every entry is `1000 * right_id + left_id`, so a
// transposed read shows up as a swapped pair rather than a plausible number.
std::string BuildProbeMatrix(int right_count, int left_count) {
  std::vector<int> costs;
  for (int left_id = 0; left_id < left_count; ++left_id) {
    for (int right_id = 0; right_id < right_count; ++right_id) {
      costs.push_back(1000 * right_id + left_id);
    }
  }
  return BuildMecabConnectionMatrix(right_count, left_count, costs);
}

TEST(MecabTokenTableTest, ReadsItsRows) {
  const std::string tokens = BuildMecabToken(11, 22, 33, -44, 0) +
                             BuildMecabToken(55, 66, 77, 8000, 7);
  const std::string features = std::string("名詞\0動詞", 13);

  ASSERT_OK_AND_ASSIGN(const MecabTokenTable table,
                       MecabTokenTable::Create(tokens, features));

  ASSERT_EQ(table.Size(), 2);
  const MecabToken first = table.Token(0);
  EXPECT_EQ(first.left_id, 11);
  EXPECT_EQ(first.right_id, 22);
  EXPECT_EQ(first.pos_id, 33);
  // Word costs are signed, and a great many real entries are negative.
  EXPECT_EQ(first.word_cost, -44);
  EXPECT_EQ(first.feature, "名詞");

  const MecabToken second = table.Token(1);
  EXPECT_EQ(second.word_cost, 8000);
  EXPECT_EQ(second.feature, "動詞");
}

TEST(MecabTokenTableTest, HasNoFeatureWhenTheOffsetIsOutOfRange) {
  const std::string tokens = BuildMecabToken(1, 2, 3, 4, 99);
  ASSERT_OK_AND_ASSIGN(const MecabTokenTable table,
                       MecabTokenTable::Create(tokens, "short"));
  EXPECT_THAT(table.Token(0).feature, IsEmpty());
}

TEST(MecabTokenTableTest, RejectsARaggedTable) {
  EXPECT_THAT(MecabTokenTable::Create(std::string(20, '\0'), "").status(),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(MecabConnectionMatrixTest, IndexesRowMajorOverTheLeftNodesRightContext) {
  // `MecabConnectionMatrix` views into `data`, which must outlive it.
  const std::string data =
      BuildProbeMatrix(/*right_count=*/5, /*left_count=*/3);
  ASSERT_OK_AND_ASSIGN(const MecabConnectionMatrix matrix,
                       MecabConnectionMatrix::Create(data));
  EXPECT_EQ(matrix.RightContextCount(), 5);
  EXPECT_EQ(matrix.LeftContextCount(), 3);

  EXPECT_EQ(matrix.Cost(/*right_id=*/0, /*left_id=*/0), 0);
  EXPECT_EQ(matrix.Cost(/*right_id=*/4, /*left_id=*/2), 4002);
  // Transposing would read 1003 here, which the non-square shape makes
  // impossible to confuse with the value above.
  EXPECT_EQ(matrix.Cost(/*right_id=*/3, /*left_id=*/1), 3001);
}

TEST(MecabConnectionMatrixTest, TreatsContextIdsOutsideTheMatrixAsFree) {
  const std::string data = BuildProbeMatrix(2, 2);
  ASSERT_OK_AND_ASSIGN(const MecabConnectionMatrix matrix,
                       MecabConnectionMatrix::Create(data));
  EXPECT_EQ(matrix.Cost(/*right_id=*/9, /*left_id=*/0), 0);
  EXPECT_EQ(matrix.Cost(/*right_id=*/0, /*left_id=*/9), 0);
  EXPECT_EQ(matrix.Cost(/*right_id=*/-1, /*left_id=*/0), 0);
}

TEST(MecabConnectionMatrixTest, RejectsABodyTooSmallForItsHeader) {
  std::string matrix = BuildProbeMatrix(4, 4);
  matrix.resize(10);
  EXPECT_THAT(MecabConnectionMatrix::Create(matrix).status(),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(MecabConnectionMatrix::Create("ab").status(),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(MecabCharPropertyTest, UnpacksTheBitfield) {
  // These are unidic-lite's own values for the two categories, and they are
  // what makes the test worth having: HIRAGANA and KATAKANA differ precisely
  // in `invoke`, so a reader that swaps `group` and `invoke` -- the order
  // MeCab declares them in is the reverse of the order they are usually named
  // in -- reads one of them wrong.
  const std::string data = BuildMecabCharProperty(
      {"DEFAULT", "HIRAGANA", "KATAKANA"},
      {PackMecabCharInfo(/*types=*/0b001, /*default_type=*/0, /*length=*/0,
                         /*group=*/true, /*invoke=*/false),
       PackMecabCharInfo(/*types=*/0b010, /*default_type=*/1, /*length=*/2,
                         /*group=*/true, /*invoke=*/false),
       PackMecabCharInfo(/*types=*/0b110, /*default_type=*/2, /*length=*/2,
                         /*group=*/true, /*invoke=*/true)},
      /*code_points=*/8);
  ASSERT_OK_AND_ASSIGN(const MecabCharProperty property,
                       MecabCharProperty::Create(data));

  const MecabCharProperty::CharInfo hiragana = property.Info(1);
  EXPECT_EQ(hiragana.types, 0b010);
  EXPECT_EQ(hiragana.default_type, 1);
  EXPECT_EQ(hiragana.length, 2);
  EXPECT_TRUE(hiragana.group);
  EXPECT_FALSE(hiragana.invoke);

  const MecabCharProperty::CharInfo katakana = property.Info(2);
  EXPECT_EQ(katakana.types, 0b110);
  EXPECT_TRUE(katakana.invoke);
}

TEST(MecabCharPropertyTest, NamesItsCategories) {
  const std::string data =
      BuildMecabCharProperty({"DEFAULT", "KANJI"}, {}, /*code_points=*/4);
  ASSERT_OK_AND_ASSIGN(const MecabCharProperty property,
                       MecabCharProperty::Create(data));
  EXPECT_EQ(property.CategoryCount(), 2);
  EXPECT_EQ(property.CategoryName(0), "DEFAULT");
  EXPECT_EQ(property.CategoryName(1), "KANJI");
}

TEST(MecabCharPropertyTest, FallsBackToTheDefaultAboveTheTable) {
  const std::string data =
      BuildMecabCharProperty({"DEFAULT", "KANJI"},
                             {PackMecabCharInfo(0b01, 0, 0, true, false),
                              PackMecabCharInfo(0b10, 1, 2, false, false)},
                             /*code_points=*/4);
  ASSERT_OK_AND_ASSIGN(const MecabCharProperty property,
                       MecabCharProperty::Create(data));
  // MeCab's own table stops at the end of the basic multilingual plane and it
  // reads past the end for anything above; the record for U+0000 is what that
  // was meant to say.
  EXPECT_EQ(property.Info(0x20000).default_type, 0);
  EXPECT_EQ(property.Info(0x20000).types, 0b01);
}

TEST(MecabCharPropertyTest, DecodesUtf8CharactersOfEveryLength) {
  const std::string data =
      BuildMecabCharProperty({"DEFAULT"}, {}, /*code_points=*/0x3100);
  ASSERT_OK_AND_ASSIGN(const MecabCharProperty property,
                       MecabCharProperty::Create(data));

  const absl::string_view text = "aéあ𠮟";
  EXPECT_EQ(property.InfoAt(text, 0).length, 1);
  EXPECT_EQ(property.InfoAt(text, 1).length, 2);
  EXPECT_EQ(property.InfoAt(text, 3).length, 3);
  EXPECT_EQ(property.InfoAt(text, 6).length, 4);
}

TEST(MecabCharPropertyTest, SpendsOneByteOnAMalformedSequence) {
  const std::string data = BuildMecabCharProperty(
      {"DEFAULT", "KANJI"}, {PackMecabCharInfo(0b01, 0, 0, true, false)},
      /*code_points=*/4);
  ASSERT_OK_AND_ASSIGN(const MecabCharProperty property,
                       MecabCharProperty::Create(data));

  // A lone continuation byte. Reporting anything but one byte would either
  // stall the tokenizer or skip past a character that follows.
  const MecabCharProperty::DecodedChar lone =
      property.InfoAt(absl::string_view("\x80z", 2), 0);
  EXPECT_EQ(lone.length, 1);
  EXPECT_EQ(lone.info.default_type, 0);

  // So would a truncated three-byte sequence at the end of the input.
  EXPECT_EQ(property.InfoAt(absl::string_view("\xE3\x81", 2), 0).length, 1);
}

TEST(MecabCharPropertyTest, RejectsATableThatDoesNotFit) {
  EXPECT_THAT(MecabCharProperty::Create("ab").status(),
              StatusIs(absl::StatusCode::kInvalidArgument));
  std::string data = BuildMecabCharProperty({"DEFAULT"}, {}, /*code_points=*/4);
  data.resize(20);
  EXPECT_THAT(MecabCharProperty::Create(data).status(),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace

}  // namespace litert::omni::tts::kokoro
