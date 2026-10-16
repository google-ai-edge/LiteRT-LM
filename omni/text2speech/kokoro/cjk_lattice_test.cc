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

#include "omni/text2speech/kokoro/cjk_lattice.h"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace litert::omni::tts::kokoro {
namespace {

using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::IsEmpty;

// A word a `ToyCostModel` knows.
struct ToyWord {
  absl::string_view surface;
  double cost = 0.0;
  int left_id = 0;
  int right_id = 0;
};

struct ToyConfig {
  std::vector<ToyWord> words;

  // What a single-character fallback node costs, or negative to offer none.
  // Withholding it is how a test makes the lattice unsolvable.
  double character_cost = -1.0;

  // Boundary costs as a row-major `context_count` square, indexed the way
  // MeCab indexes its connection matrix: row by the left node's `right_id`,
  // column by the right node's `left_id`. Empty means boundaries are free.
  std::vector<double> connection;
  int context_count = 1;

  // Whether to hand leading spaces to the following word rather than making a
  // node out of them, as MeCab does.
  bool attach_spaces = false;
};

// Returns the byte length of the UTF-8 character starting at `start`, clamped
// to what is left of `text`.
int Utf8Length(absl::string_view text, int start) {
  const auto lead = static_cast<unsigned char>(text[start]);
  int length = 1;
  if (lead >= 0xF0) {
    length = 4;
  } else if (lead >= 0xE0) {
    length = 3;
  } else if (lead >= 0xC0) {
    length = 2;
  }
  return std::min<int>(length, static_cast<int>(text.size()) - start);
}

// A cost model small enough to reason about by hand, shaped like the real
// ones: a dictionary lookup, a fallback for what the dictionary misses, and a
// square connection matrix.
class ToyCostModel : public CostModel {
 public:
  explicit ToyCostModel(ToyConfig config);

  void CandidatesAt(absl::string_view text, int start,
                    std::vector<LatticeNode>& candidates) const override;
  double ConnectionCost(int right_id, int left_id) const override;

 private:
  ToyConfig config_;
};

ToyCostModel::ToyCostModel(ToyConfig config) : config_(std::move(config)) {}

void ToyCostModel::CandidatesAt(absl::string_view text, int start,
                                std::vector<LatticeNode>& candidates) const {
  int surface_start = start;
  if (config_.attach_spaces) {
    while (surface_start < static_cast<int>(text.size()) &&
           text[surface_start] == ' ') {
      ++surface_start;
    }
    if (surface_start == static_cast<int>(text.size())) return;
  }
  const absl::string_view rest = text.substr(surface_start);

  bool matched = false;
  for (const ToyWord& word : config_.words) {
    if (!rest.starts_with(word.surface)) continue;
    matched = true;
    candidates.push_back({
        .start = start,
        .end = surface_start + static_cast<int>(word.surface.size()),
        .surface = rest.substr(0, word.surface.size()),
        .left_id = word.left_id,
        .right_id = word.right_id,
        .word_cost = word.cost,
    });
  }
  if (matched || config_.character_cost < 0.0) return;

  const int length = Utf8Length(text, surface_start);
  candidates.push_back({
      .start = start,
      .end = surface_start + length,
      .surface = rest.substr(0, length),
      .word_cost = config_.character_cost,
      .is_unknown = true,
  });
}

double ToyCostModel::ConnectionCost(int right_id, int left_id) const {
  if (config_.connection.empty()) return 0.0;
  return config_.connection[right_id * config_.context_count + left_id];
}

// A model with the off-by-one a real one could plausibly have: it reports each
// node as starting one byte after the offset it was asked about, which would
// leave that byte unread by anything.
class MisalignedCostModel : public CostModel {
 public:
  void CandidatesAt(absl::string_view text, int start,
                    std::vector<LatticeNode>& candidates) const override;
  double ConnectionCost(int right_id, int left_id) const override;
};

void MisalignedCostModel::CandidatesAt(
    absl::string_view text, int start,
    std::vector<LatticeNode>& candidates) const {
  candidates.push_back({
      .start = start + 1,
      .end = start + 2,
      .surface = text.substr(start + 1, 1),
  });
}

double MisalignedCostModel::ConnectionCost(int right_id, int left_id) const {
  return 0.0;
}

std::vector<std::string> Surfaces(const std::vector<LatticeNode>& path) {
  std::vector<std::string> surfaces;
  surfaces.reserve(path.size());
  for (const LatticeNode& node : path) {
    surfaces.emplace_back(node.surface);
  }
  return surfaces;
}

TEST(ViterbiSegmentTest, EmptyTextHasAnEmptyPath) {
  const ToyCostModel model{ToyConfig{.character_cost = 1.0}};
  ASSERT_OK_AND_ASSIGN(const std::vector<LatticeNode> path,
                       ViterbiSegment("", model));
  EXPECT_THAT(path, IsEmpty());
}

TEST(ViterbiSegmentTest, PrefersTheCheaperSegmentation) {
  // "ab" + "c" costs 20; "abc" on its own costs 30, so the pair wins.
  const ToyConfig config = {.words = {{.surface = "ab", .cost = 10.0},
                                      {.surface = "abc", .cost = 30.0},
                                      {.surface = "c", .cost = 10.0}}};
  const ToyCostModel model{config};
  ASSERT_OK_AND_ASSIGN(const std::vector<LatticeNode> path,
                       ViterbiSegment("abc", model));
  EXPECT_THAT(Surfaces(path), ElementsAre("ab", "c"));
}

TEST(ViterbiSegmentTest, PrefersTheLongWordOnceItIsCheapEnough) {
  ToyConfig config = {.words = {{.surface = "ab", .cost = 10.0},
                                {.surface = "abc", .cost = 15.0},
                                {.surface = "c", .cost = 10.0}}};
  const ToyCostModel model{config};
  ASSERT_OK_AND_ASSIGN(const std::vector<LatticeNode> path,
                       ViterbiSegment("abc", model));
  EXPECT_THAT(Surfaces(path), ElementsAre("abc"));
}

TEST(ViterbiSegmentTest, ConnectionCostCanOutweighWordCost) {
  // Word costs alone would pick "ab" + "c", as above. Charging 100 for the
  // boundary between context 1 and context 2 -- the only boundary that pair
  // has -- makes the single long word the cheaper reading.
  ToyConfig config = {
      .words = {{.surface = "ab", .cost = 10.0, .left_id = 0, .right_id = 1},
                {.surface = "abc", .cost = 30.0},
                {.surface = "c", .cost = 10.0, .left_id = 2, .right_id = 0}},
      .connection = {0.0, 0.0, 0.0,    //
                     0.0, 0.0, 100.0,  //
                     0.0, 0.0, 0.0},
      .context_count = 3};
  const ToyCostModel model{config};
  ASSERT_OK_AND_ASSIGN(const std::vector<LatticeNode> path,
                       ViterbiSegment("abc", model));
  EXPECT_THAT(Surfaces(path), ElementsAre("abc"));
}

TEST(ViterbiSegmentTest, ScoresTheSentenceBoundariesAsContextZero) {
  // Both readings of "abc" cost 20 in words. Context 1 is free against its
  // neighbours but expensive against the sentence boundary, which is context
  // 0, so the reading that touches a boundary with it loses.
  ToyConfig config = {
      .words = {{.surface = "ab", .cost = 10.0, .left_id = 1, .right_id = 1},
                {.surface = "abc", .cost = 20.0, .left_id = 2, .right_id = 2},
                {.surface = "c", .cost = 10.0, .left_id = 1, .right_id = 1}},
      .connection = {0.0, 100.0, 0.0,  //
                     100.0, 0.0, 0.0,  //
                     0.0, 0.0, 0.0},
      .context_count = 3};
  const ToyCostModel model{config};
  ASSERT_OK_AND_ASSIGN(const std::vector<LatticeNode> path,
                       ViterbiSegment("abc", model));
  EXPECT_THAT(Surfaces(path), ElementsAre("abc"));
}

TEST(ViterbiSegmentTest, FallsBackToTheModelsOwnUnknownNodes) {
  const ToyConfig config = {.words = {{.surface = "b", .cost = 1.0}},
                            .character_cost = 5.0};
  const ToyCostModel model{config};
  ASSERT_OK_AND_ASSIGN(const std::vector<LatticeNode> path,
                       ViterbiSegment("abc", model));
  EXPECT_THAT(Surfaces(path), ElementsAre("a", "b", "c"));
  EXPECT_THAT(path[0].is_unknown, true);
  EXPECT_THAT(path[1].is_unknown, false);
  EXPECT_THAT(path[2].is_unknown, true);
}

TEST(ViterbiSegmentTest, OffsetsAreBytesAndNodesStayOnCharacterBoundaries) {
  const ToyConfig config = {.words = {{.surface = "日本語", .cost = 1.0}},
                            .character_cost = 5.0};
  const ToyCostModel model{config};
  ASSERT_OK_AND_ASSIGN(const std::vector<LatticeNode> path,
                       ViterbiSegment("あ日本語", model));
  EXPECT_THAT(Surfaces(path), ElementsAre("あ", "日本語"));
  EXPECT_EQ(path[0].start, 0);
  EXPECT_EQ(path[0].end, 3);
  EXPECT_EQ(path[1].start, 3);
  EXPECT_EQ(path[1].end, 12);
}

TEST(ViterbiSegmentTest, ASurfaceMayBeShorterThanTheSpanItConsumes) {
  const ToyConfig config = {
      .words = {{.surface = "ab", .cost = 1.0}, {.surface = "cd", .cost = 1.0}},
      .attach_spaces = true};
  const ToyCostModel model{config};
  ASSERT_OK_AND_ASSIGN(const std::vector<LatticeNode> path,
                       ViterbiSegment("ab   cd", model));
  EXPECT_THAT(Surfaces(path), ElementsAre("ab", "cd"));
  // The three spaces belong to the span of the word that follows them, so the
  // path still covers every byte of the input.
  EXPECT_EQ(path[1].start, 2);
  EXPECT_EQ(path[1].end, 7);
}

TEST(ViterbiSegmentTest, FailsWhenNoPathSpansTheInput) {
  // Without a character fallback there is nothing to read "c" with, so the
  // lattice stops one byte short of the end.
  const ToyConfig config = {.words = {{.surface = "ab", .cost = 1.0}}};
  const ToyCostModel model{config};
  EXPECT_THAT(ViterbiSegment("abc", model),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(ViterbiSegmentTest, RejectsANodeThatDoesNotStartWhereItWasAsked) {
  const MisalignedCostModel model;
  EXPECT_THAT(ViterbiSegment("abc", model),
              StatusIs(absl::StatusCode::kInternal));
}

}  // namespace
}  // namespace litert::omni::tts::kokoro
