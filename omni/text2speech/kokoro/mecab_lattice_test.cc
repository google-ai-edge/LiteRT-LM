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

#include "omni/text2speech/kokoro/mecab_lattice.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "omni/text2speech/kokoro/cjk_blob.h"
#include "omni/text2speech/kokoro/cjk_lattice.h"
#include "omni/text2speech/kokoro/cjk_test_blobs.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace litert::omni::tts::kokoro {
namespace {

using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::UnorderedElementsAre;

// A dictionary small enough to reason about, arranged over ASCII so the tests
// stay readable, but shaped exactly like unidic-lite: four character
// categories that differ in the three flags the tokenizer branches on, a
// handful of words, and one surface form standing for two different readings.
//
//   ' '     SPACE     -- skipped, never a word
//   a - z   PLAIN     -- like KANJI: never guessed past, two lengths
//   A - Z   SHOUTY    -- like KATAKANA: guessed at even when the dictionary
//                        matched, and grouped across the whole run
//   other   DEFAULT
enum Category { kDefault = 0, kSpace = 1, kPlain = 2, kShouty = 3 };

// Context ids. Zero is the sentence boundary, as it is in MeCab.
constexpr int kContextCount = 4;
constexpr int kPlainContext = 1;
constexpr int kOtherContext = 2;

std::string BuildCharProperty() {
  std::vector<uint32_t> records(128);
  for (int code_point = 0; code_point < 128; ++code_point) {
    Category category = kDefault;
    if (code_point == ' ') {
      category = kSpace;
    } else if (code_point >= 'a' && code_point <= 'z') {
      category = kPlain;
    } else if (code_point >= 'A' && code_point <= 'Z') {
      category = kShouty;
    }
    switch (category) {
      case kSpace:
        records[code_point] =
            PackMecabCharInfo(1 << kSpace, kSpace, /*length=*/0, /*group=*/true,
                              /*invoke=*/false);
        break;
      case kPlain:
        records[code_point] = PackMecabCharInfo(1 << kPlain, kPlain,
                                                /*length=*/2, /*group=*/false,
                                                /*invoke=*/false);
        break;
      case kShouty:
        records[code_point] = PackMecabCharInfo(1 << kShouty, kShouty,
                                                /*length=*/2, /*group=*/true,
                                                /*invoke=*/true);
        break;
      case kDefault:
        records[code_point] = PackMecabCharInfo(1 << kDefault, kDefault,
                                                /*length=*/0, /*group=*/true,
                                                /*invoke=*/false);
        break;
    }
  }
  return BuildMecabCharProperty({"DEFAULT", "SPACE", "PLAIN", "SHOUTY"},
                                records, /*code_points=*/128);
}

// Builds a `ja-lexicon` blob. `connection` is indexed the way
// `Connector::cost` indexes MeCab's matrix, `[left_id * kContextCount +
// right_id]`, and `unknown_categories` lets a test leave one out.
std::string BuildJapaneseBlob(
    const std::vector<int>& connection,
    const std::vector<std::string>& unknown_categories = {"DEFAULT", "SPACE",
                                                          "PLAIN", "SHOUTY"}) {
  // "x" stands for two readings, which is the case that makes the connection
  // matrix matter: which one is cheapest depends on the neighbours.
  std::vector<uint32_t> offsets;
  const std::string features = BuildMecabFeatures(
      {"ab", "abc", "c", "x/plain", "x/other", "X"}, offsets);
  const std::string tokens =
      BuildMecabToken(kPlainContext, kPlainContext, 1, 100, offsets[0]) +
      BuildMecabToken(kPlainContext, kPlainContext, 1, 250, offsets[1]) +
      BuildMecabToken(kPlainContext, kPlainContext, 1, 100, offsets[2]) +
      BuildMecabToken(kPlainContext, kPlainContext, 1, 100, offsets[3]) +
      BuildMecabToken(kOtherContext, kOtherContext, 1, 100, offsets[4]) +
      BuildMecabToken(kPlainContext, kPlainContext, 1, 100, offsets[5]);
  const std::string index = BuildMecabDoubleArray({
      {"ab", 0 << 8 | 1},
      {"abc", 1 << 8 | 1},
      {"c", 2 << 8 | 1},
      {"x", 3 << 8 | 2},  // Two tokens, starting at row three.
      {"X", 5 << 8 | 1},
  });

  // Guessed words are dear, so the search only reaches for one where the
  // dictionary leaves it no choice.
  std::vector<uint32_t> unknown_offsets;
  const std::string unknown_features = BuildMecabFeatures(
      {"DEFAULT", "SPACE", "PLAIN", "SHOUTY"}, unknown_offsets);
  std::string unknown_tokens;
  std::vector<std::pair<std::string, int>> unknown_index;
  for (int i = 0; i < 4; ++i) {
    unknown_tokens += BuildMecabToken(kPlainContext, kPlainContext, 9, 3000,
                                      unknown_offsets[i]);
  }
  const std::vector<std::string> all = {"DEFAULT", "SPACE", "PLAIN", "SHOUTY"};
  for (int i = 0; i < 4; ++i) {
    for (const std::string& wanted : unknown_categories) {
      if (wanted == all[i]) unknown_index.push_back({all[i], i << 8 | 1});
    }
  }

  return BuildCjkBlob({
      {"sys-da", index},
      {"sys-tok", tokens},
      {"sys-feat", features},
      {"matrix",
       BuildMecabConnectionMatrix(kContextCount, kContextCount, connection)},
      {"char", BuildCharProperty()},
      {"unk-da", BuildMecabDoubleArray(unknown_index)},
      {"unk-tok", unknown_tokens},
      {"unk-feat", unknown_features},
  });
}

std::vector<std::string> Surfaces(const std::vector<LatticeNode>& path) {
  std::vector<std::string> surfaces;
  surfaces.reserve(path.size());
  for (const LatticeNode& node : path) {
    surfaces.emplace_back(node.surface);
  }
  return surfaces;
}

// The spans a model offers at one offset, as (start, end) pairs.
std::vector<std::pair<int, int>> Spans(
    const std::vector<LatticeNode>& candidates) {
  std::vector<std::pair<int, int>> spans;
  spans.reserve(candidates.size());
  for (const LatticeNode& node : candidates) {
    spans.push_back({node.start, node.end});
  }
  return spans;
}

// One case of `ConnectionCostsChooseBetweenReadingsOfOneSurface`.
struct ReadingCostCase {
  int plain_cost;
  int other_cost;
  absl::string_view expected;
};

TEST(MecabCostModelTest, SegmentsIntoTheCheapestDictionaryReading) {
  const std::string data = BuildJapaneseBlob({});
  ASSERT_OK_AND_ASSIGN(const CjkBlob blob, CjkBlob::Create(data));
  ASSERT_OK_AND_ASSIGN(const MecabCostModel model,
                       MecabCostModel::Create(blob));

  // "ab" plus "c" costs 200; "abc" on its own costs 250.
  ASSERT_OK_AND_ASSIGN(const std::vector<LatticeNode> path,
                       ViterbiSegment("abc", model));
  EXPECT_THAT(Surfaces(path), ElementsAre("ab", "c"));
  EXPECT_EQ(model.Token(path[0]).feature, "ab");
  EXPECT_EQ(model.Token(path[1]).feature, "c");
}

TEST(MecabCostModelTest, ConnectionCostsChooseBetweenReadingsOfOneSurface) {
  // Both readings of "x" cost the same on their own and differ only in their
  // context ids, so the boundary with the start of the sentence -- context
  // zero -- is the whole of the decision.
  for (const ReadingCostCase& test_case :
       {ReadingCostCase{0, 500, "x/plain"},
        ReadingCostCase{500, 0, "x/other"}}) {
    std::vector<int> connection(kContextCount * kContextCount, 0);
    connection[kPlainContext * kContextCount + 0] = test_case.plain_cost;
    connection[kOtherContext * kContextCount + 0] = test_case.other_cost;

    const std::string data = BuildJapaneseBlob(connection);
    ASSERT_OK_AND_ASSIGN(const CjkBlob blob, CjkBlob::Create(data));
    ASSERT_OK_AND_ASSIGN(const MecabCostModel model,
                         MecabCostModel::Create(blob));
    ASSERT_OK_AND_ASSIGN(const std::vector<LatticeNode> path,
                         ViterbiSegment("x", model));
    ASSERT_EQ(path.size(), 1);
    EXPECT_EQ(model.Token(path[0]).feature, test_case.expected);
  }
}

TEST(MecabCostModelTest, AttachesLeadingWhitespaceToTheWordThatFollows) {
  const std::string data = BuildJapaneseBlob({});
  ASSERT_OK_AND_ASSIGN(const CjkBlob blob, CjkBlob::Create(data));
  ASSERT_OK_AND_ASSIGN(const MecabCostModel model,
                       MecabCostModel::Create(blob));

  ASSERT_OK_AND_ASSIGN(const std::vector<LatticeNode> path,
                       ViterbiSegment("ab  c", model));
  EXPECT_THAT(Surfaces(path), ElementsAre("ab", "c"));
  // The two spaces are not a word, but they are still part of what the second
  // node consumes, which is what keeps the path covering every byte.
  EXPECT_EQ(path[1].start, 2);
  EXPECT_EQ(path[1].end, 5);
}

TEST(MecabCostModelTest, CoversAWhitespaceTail) {
  const std::string data = BuildJapaneseBlob({});
  ASSERT_OK_AND_ASSIGN(const CjkBlob blob, CjkBlob::Create(data));
  ASSERT_OK_AND_ASSIGN(const MecabCostModel model,
                       MecabCostModel::Create(blob));

  // MeCab yields no token for trailing whitespace at all. A lattice that
  // cannot reach the end of its input has no answer, so the run becomes a node
  // the front end can recognise and drop.
  ASSERT_OK_AND_ASSIGN(const std::vector<LatticeNode> path,
                       ViterbiSegment("ab  ", model));
  EXPECT_THAT(Surfaces(path), ElementsAre("ab", "  "));
  EXPECT_EQ(model.Token(path[1]).feature, "SPACE");
}

TEST(MecabCostModelTest, GuessesAWordWhereTheDictionaryHasNothing) {
  const std::string data = BuildJapaneseBlob({});
  ASSERT_OK_AND_ASSIGN(const CjkBlob blob, CjkBlob::Create(data));
  ASSERT_OK_AND_ASSIGN(const MecabCostModel model,
                       MecabCostModel::Create(blob));

  ASSERT_OK_AND_ASSIGN(const std::vector<LatticeNode> path,
                       ViterbiSegment("z", model));
  EXPECT_THAT(Surfaces(path), ElementsAre("z"));
  EXPECT_TRUE(path[0].is_unknown);
  EXPECT_EQ(model.Token(path[0]).feature, "PLAIN");
}

TEST(MecabCostModelTest, GuessesAcrossTheWholeRunWhenTheCategoryGroups) {
  const std::string data = BuildJapaneseBlob({});
  ASSERT_OK_AND_ASSIGN(const CjkBlob blob, CjkBlob::Create(data));
  ASSERT_OK_AND_ASSIGN(const MecabCostModel model,
                       MecabCostModel::Create(blob));

  std::vector<LatticeNode> candidates;
  model.CandidatesAt("ZZZ", 0, candidates);
  // The whole run, then each length the category allows.
  EXPECT_THAT(
      Spans(candidates),
      UnorderedElementsAre(std::pair(0, 3), std::pair(0, 1), std::pair(0, 2)));

  // One guess beats three, so the run stays whole.
  ASSERT_OK_AND_ASSIGN(const std::vector<LatticeNode> path,
                       ViterbiSegment("ZZZ", model));
  EXPECT_THAT(Surfaces(path), ElementsAre("ZZZ"));
}

TEST(MecabCostModelTest, GuessesEvenWhereTheDictionaryMatchedWhenInvokeIsSet) {
  const std::string data = BuildJapaneseBlob({});
  ASSERT_OK_AND_ASSIGN(const CjkBlob blob, CjkBlob::Create(data));
  ASSERT_OK_AND_ASSIGN(const MecabCostModel model,
                       MecabCostModel::Create(blob));

  // "X" is in the dictionary, but its category invokes the guesser anyway, so
  // the two-character run is offered as well. Without that, a long run would
  // be cut into whichever short entries happen to exist.
  std::vector<LatticeNode> candidates;
  model.CandidatesAt("XX", 0, candidates);
  EXPECT_THAT(
      Spans(candidates),
      UnorderedElementsAre(std::pair(0, 1), std::pair(0, 1), std::pair(0, 2)));

  // The same run in the category that does not invoke stops at the dictionary.
  candidates.clear();
  model.CandidatesAt("ab", 0, candidates);
  EXPECT_THAT(Spans(candidates), ElementsAre(std::pair(0, 2)));
}

TEST(MecabCostModelTest, RejectsADictionaryMissingAnUnknownCategory) {
  const std::string data = BuildJapaneseBlob(
      {}, /*unknown_categories=*/{"DEFAULT", "SPACE", "PLAIN"});
  ASSERT_OK_AND_ASSIGN(const CjkBlob blob, CjkBlob::Create(data));
  // A category with nowhere to guess from would be a word that silently
  // cannot be read, so it fails at load instead.
  EXPECT_THAT(MecabCostModel::Create(blob),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(MecabCostModelTest, RequiresEverySectionItReads) {
  const std::string data = BuildCjkBlob({{"sys-da", "unused"}});
  ASSERT_OK_AND_ASSIGN(const CjkBlob blob, CjkBlob::Create(data));
  EXPECT_THAT(MecabCostModel::Create(blob),
              StatusIs(absl::StatusCode::kNotFound));
}

}  // namespace
}  // namespace litert::omni::tts::kokoro
