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

#include "omni/speech2text/timestamp_text_merger.h"

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/cc/litert_macros.h"  // from @litert
#include "omni/base/stage.h"
#include "omni/speech2text/detokenizer.h"
#include "omni/speech2text/text_merger.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep for ASSERT_OK, ASSERT_OK_AND_ASSIGN

namespace litert::omni::asr {
namespace {

class DummyWordStage
    : public SingleThreadedStageWithDeque<std::vector<Detokenizer::Word>> {
 public:
  void PushWords(const std::vector<std::string>& strings) {
    std::vector<Detokenizer::Word> words;
    words.reserve(strings.size());
    for (const auto& s : strings) {
      words.push_back({s, std::nullopt});
    }
    PushOutput(std::move(words));
  }

  void PushWordsWithTimestamps(
      const std::vector<std::pair<std::string, int>>& word_ts_pairs) {
    std::vector<Detokenizer::Word> words;
    words.reserve(word_ts_pairs.size());
    for (const auto& pair : word_ts_pairs) {
      words.push_back({pair.first, pair.second});
    }
    PushOutput(std::move(words));
  }

  void PushWordsWithEndOfChunkTimestamp(
      const std::vector<std::pair<std::string, int>>& word_ts_pairs,
      int end_ts_ms, bool is_warm_start = false) {
    std::vector<Detokenizer::Word> words;
    words.reserve(word_ts_pairs.size() + 1);
    for (const auto& pair : word_ts_pairs) {
      words.push_back({pair.first, pair.second});
    }
    words.push_back({"", end_ts_ms, is_warm_start});
    PushOutput(std::move(words));
  }

 protected:
  bool NeedScheduleInternal() const override { return false; }
  absl::Status ScheduleInternal() override { return absl::OkStatus(); }
};

absl::StatusOr<TextMerger::MergeResult> GetMergeResult(
    TimestampTextMerger& merger) {
  LITERT_ASSIGN_OR_RETURN(auto out, merger.GetOutput());
  return std::get<TextMerger::MergeResult>(std::move(out));
}

absl::StatusOr<TextMerger::MergeResult> ProcessWords(
    DummyWordStage& word_stage, TimestampTextMerger& merger,
    const std::vector<std::string>& strings) {
  word_stage.PushWords(strings);
  LITERT_RETURN_IF_ERROR(merger.Schedule());
  return GetMergeResult(merger);
}

absl::StatusOr<TextMerger::MergeResult> ProcessWordsWithTimestamps(
    DummyWordStage& word_stage, TimestampTextMerger& merger,
    const std::vector<std::pair<std::string, int>>& word_ts_pairs) {
  word_stage.PushWordsWithTimestamps(word_ts_pairs);
  LITERT_RETURN_IF_ERROR(merger.Schedule());
  return GetMergeResult(merger);
}

TEST(TimestampTextMergerTest, InitialChunkReturnsUnconfirmedOnly) {
  DummyWordStage word_stage;
  TimestampTextMerger merger(&word_stage);

  auto result = ProcessWords(word_stage, merger, {"hello", "world"});
  ASSERT_OK(result);

  EXPECT_EQ(result->confirmed_text, "");
  EXPECT_EQ(result->unconfirmed_text, "hello world");
}

TEST(TimestampTextMergerTest, SequentialMergeFlowWithTimestamps) {
  DummyWordStage word_stage;
  TimestampTextMerger merger(&word_stage, /*overlap_ratio=*/0.5f);

  // Chunk 1 (0 to 5000 ms)
  auto res1 = ProcessWordsWithTimestamps(word_stage, merger,
                                         {{"the", 500},
                                          {"stale", 1000},
                                          {"smell", 1500},
                                          {"of", 2000},
                                          {"old", 2500},
                                          {"beer", 3000},
                                          {"lingers", 3500}});
  ASSERT_OK(res1);
  EXPECT_EQ(res1->confirmed_text, "");
  EXPECT_EQ(res1->unconfirmed_text, "the stale smell of old beer lingers");

  // Chunk 2 (2500 ms to 7500 ms relative to start of Chunk 2)
  auto res2 = ProcessWordsWithTimestamps(word_stage, merger,
                                         {{"old", 0},
                                          {"beer", 500},
                                          {"lingers", 1000},
                                          {"it", 1500},
                                          {"takes", 2000},
                                          {"heat", 2500}});
  ASSERT_OK(res2);
  EXPECT_FALSE(res2->confirmed_text.empty());
  EXPECT_FALSE(res2->unconfirmed_text.empty());

  ASSERT_OK(merger.Flush());
  auto res_flush = GetMergeResult(merger);
  ASSERT_OK(res_flush);
  EXPECT_FALSE(res_flush->confirmed_text.empty());
  EXPECT_EQ(res_flush->unconfirmed_text, "");
}

TEST(TimestampTextMergerTest, ResetClearsState) {
  DummyWordStage word_stage;
  TimestampTextMerger merger(&word_stage);

  auto dummy_res = ProcessWords(word_stage, merger, {"hello", "world"});
  EXPECT_TRUE(dummy_res.ok());
  EXPECT_FALSE(merger.unconfirmed_words_for_testing().empty());

  merger.Reset();
  EXPECT_TRUE(merger.unconfirmed_words_for_testing().empty());

  auto result = ProcessWords(word_stage, merger, {"new", "stream"});
  ASSERT_OK(result);
  EXPECT_EQ(result->confirmed_text, "");
  EXPECT_EQ(result->unconfirmed_text, "new stream");
}

TEST(TimestampTextMergerTest,
     SequentialMergeFlowUsesEndOfChunkTimestampForStep) {
  DummyWordStage word_stage;
  TimestampTextMerger merger(&word_stage, /*overlap_ratio=*/0.5f);

  // Chunk 1 has 5000 ms total duration (end of chunk ts = 5000)
  word_stage.PushWordsWithEndOfChunkTimestamp({{"the", 500},
                                               {"stale", 1000},
                                               {"smell", 1500},
                                               {"of", 2000},
                                               {"old", 2500},
                                               {"beer", 3000},
                                               {"lingers", 3500}},
                                              /*end_ts_ms=*/5000);
  ASSERT_OK(merger.Schedule());
  auto res1 = GetMergeResult(merger);
  ASSERT_OK(res1);

  // Chunk 2 starts at 2500 ms relative to start of Chunk 2
  word_stage.PushWordsWithEndOfChunkTimestamp({{"old", 0},
                                               {"beer", 500},
                                               {"lingers", 1000},
                                               {"it", 1500},
                                               {"takes", 2000},
                                               {"heat", 2500}},
                                              /*end_ts_ms=*/5000);
  ASSERT_OK(merger.Schedule());
  auto res2 = GetMergeResult(merger);
  ASSERT_OK(res2);
  EXPECT_FALSE(res2->confirmed_text.empty());
  EXPECT_FALSE(res2->unconfirmed_text.empty());
}

TEST(TimestampTextMergerTest, ReplacesEndOfChunkHallucinationWithGroundTruth) {
  DummyWordStage word_stage;
  TimestampTextMerger merger(&word_stage, /*overlap_ratio=*/0.4f);

  // Chunk 1 has hallucinated "huge" at the end of chunk
  word_stage.PushWordsWithEndOfChunkTimestamp({{"the", 1000},
                                               {"stale", 1600},
                                               {"smell", 2100},
                                               {"of", 2600},
                                               {"old", 3000},
                                               {"beer", 3400},
                                               {"lingers", 4100},
                                               {"it", 4600},
                                               {"takes", 5000},
                                               {"huge", 5400}},
                                              /*end_ts_ms=*/5500);
  ASSERT_OK(merger.Schedule());
  auto res1 = GetMergeResult(merger);
  ASSERT_OK(res1);
  EXPECT_EQ(res1->confirmed_text, "");

  // Chunk 2 decodes ground truth "heat" at 1800 ms
  word_stage.PushWordsWithEndOfChunkTimestamp({{"lingers", 300},
                                               {"it", 800},
                                               {"takes", 1300},
                                               {"heat", 1800},
                                               {"to", 2300},
                                               {"bring", 2700},
                                               {"out", 2800}},
                                              /*end_ts_ms=*/5500);
  ASSERT_OK(merger.Schedule());
  auto res2 = GetMergeResult(merger);
  ASSERT_OK(res2);
  EXPECT_EQ(res2->confirmed_text, "the stale smell of old beer");
  EXPECT_EQ(res2->unconfirmed_text, "lingers it takes heat to bring out");

  ASSERT_OK(merger.Flush());
  auto res_flush = GetMergeResult(merger);
  ASSERT_OK(res_flush);
  EXPECT_EQ(res_flush->confirmed_text, "lingers it takes heat to bring out");
  EXPECT_EQ(res_flush->unconfirmed_text, "");
}

TEST(TimestampTextMergerTest, SilencePausePreventsMisalignment) {
  DummyWordStage word_stage;
  TimestampTextMerger merger(&word_stage, /*overlap_ratio=*/0.5f);

  // Chunk 1 ends early with last word at 1000ms
  word_stage.PushWordsWithEndOfChunkTimestamp({{"hello", 500}, {"world", 1000}},
                                              /*end_ts_ms=*/5000);
  ASSERT_OK(merger.Schedule());
  auto res1 = GetMergeResult(merger);
  ASSERT_OK(res1);

  // Chunk 2 starts after long silence (first word at 4500ms). Its window
  // starts at 2500 ms, after both words of chunk 1, which are therefore final.
  word_stage.PushWordsWithEndOfChunkTimestamp(
      {{"hello", 4500}, {"again", 4800}},
      /*end_ts_ms=*/5000);
  ASSERT_OK(merger.Schedule());
  auto res2 = GetMergeResult(merger);
  ASSERT_OK(res2);
  EXPECT_EQ(res2->confirmed_text, "hello world");
  EXPECT_EQ(res2->unconfirmed_text, "hello again");

  ASSERT_OK(merger.Flush());
  auto res_flush = GetMergeResult(merger);
  ASSERT_OK(res_flush);
  EXPECT_EQ(res_flush->confirmed_text, "hello again");
  EXPECT_EQ(res_flush->unconfirmed_text, "");
}

TEST(TimestampTextMergerTest, KeepsPreviousWordsMissingFromCurrentChunkStart) {
  DummyWordStage word_stage;
  TimestampTextMerger merger(&word_stage, /*overlap_ratio=*/0.4f);

  word_stage.PushWordsWithEndOfChunkTimestamp({{"the", 500},
                                               {"stale", 1000},
                                               {"smell", 2000},
                                               {"of", 2500},
                                               {"old", 3200},
                                               {"beer", 3600},
                                               {"lingers", 4000},
                                               {"it", 4400},
                                               {"takes", 4800}},
                                              /*end_ts_ms=*/5000);
  ASSERT_OK(merger.Schedule());
  ASSERT_OK(GetMergeResult(merger));

  // Chunk 2 (from 3000 ms) did not recognize "old beer" at its start.
  word_stage.PushWordsWithEndOfChunkTimestamp(
      {{"lingers", 1000}, {"it", 1400}, {"takes", 1800}, {"heat", 2200}},
      /*end_ts_ms=*/5000);
  ASSERT_OK(merger.Schedule());
  auto res2 = GetMergeResult(merger);
  ASSERT_OK(res2);
  EXPECT_EQ(res2->confirmed_text, "the stale smell of");
  EXPECT_EQ(res2->unconfirmed_text, "old beer lingers it takes heat");
}

TEST(TimestampTextMergerTest, KeepsPreviousTailWhereCurrentChunkHasGap) {
  DummyWordStage word_stage;
  TimestampTextMerger merger(&word_stage, /*overlap_ratio=*/0.4f);

  word_stage.PushWordsWithEndOfChunkTimestamp({{"the", 500},
                                               {"stale", 1000},
                                               {"smell", 2000},
                                               {"of", 2500},
                                               {"old", 3200},
                                               {"beer", 3600},
                                               {"lingers", 4000},
                                               {"it", 4100},
                                               {"takes", 4500},
                                               {"heat", 4700}},
                                              /*end_ts_ms=*/5000);
  ASSERT_OK(merger.Schedule());
  ASSERT_OK(GetMergeResult(merger));

  // Chunk 2 (from 3000 ms) heard nothing between "it" and "to": the previous
  // chunk's "takes heat" is kept and the new word "to" is added.
  word_stage.PushWordsWithEndOfChunkTimestamp({{"old", 200},
                                               {"beer", 600},
                                               {"lingers", 1000},
                                               {"it", 1100},
                                               {"to", 2600}},
                                              /*end_ts_ms=*/5000);
  ASSERT_OK(merger.Schedule());
  auto res2 = GetMergeResult(merger);
  ASSERT_OK(res2);
  EXPECT_EQ(res2->confirmed_text, "the stale smell of");
  EXPECT_EQ(res2->unconfirmed_text, "old beer lingers it takes heat to");
}

TEST(TimestampTextMergerTest, CurrentChunkWinsAfterPivotWhereItHeardWords) {
  DummyWordStage word_stage;
  TimestampTextMerger merger(&word_stage, /*overlap_ratio=*/0.4f);

  // "ta kes" at the end of chunk 1 was heard as two words.
  word_stage.PushWordsWithEndOfChunkTimestamp({{"the", 500},
                                               {"stale", 1000},
                                               {"smell", 2000},
                                               {"of", 2500},
                                               {"old", 3200},
                                               {"beer", 3600},
                                               {"lingers", 4000},
                                               {"it", 4300},
                                               {"ta", 4600},
                                               {"kes", 4800}},
                                              /*end_ts_ms=*/5000);
  ASSERT_OK(merger.Schedule());
  ASSERT_OK(GetMergeResult(merger));

  word_stage.PushWordsWithEndOfChunkTimestamp({{"old", 200},
                                               {"beer", 600},
                                               {"lingers", 1000},
                                               {"it", 1300},
                                               {"takes", 1700},
                                               {"heat", 2200}},
                                              /*end_ts_ms=*/5000);
  ASSERT_OK(merger.Schedule());
  auto res2 = GetMergeResult(merger);
  ASSERT_OK(res2);
  EXPECT_EQ(res2->confirmed_text, "the stale smell of");
  EXPECT_EQ(res2->unconfirmed_text, "old beer lingers it takes heat");
}

TEST(TimestampTextMergerTest, ChunkAfterEmptyWindowConfirmsPreviousWords) {
  DummyWordStage word_stage;
  TimestampTextMerger merger(&word_stage, /*overlap_ratio=*/0.4f);

  word_stage.PushWordsWithEndOfChunkTimestamp({{"hello", 500}, {"world", 1000}},
                                              /*end_ts_ms=*/5000);
  ASSERT_OK(merger.Schedule());
  ASSERT_OK(GetMergeResult(merger));

  // Chunk 2 (3000-8000 ms) is silent.
  word_stage.PushWordsWithEndOfChunkTimestamp({}, /*end_ts_ms=*/5000);
  ASSERT_OK(merger.Schedule());
  auto res2 = GetMergeResult(merger);
  ASSERT_OK(res2);
  EXPECT_EQ(res2->confirmed_text, "");
  EXPECT_EQ(res2->unconfirmed_text, "hello world");

  // Chunk 3 (6000-11000 ms) does not overlap the window of "hello world".
  word_stage.PushWordsWithEndOfChunkTimestamp({{"new", 500}, {"words", 900}},
                                              /*end_ts_ms=*/5000);
  ASSERT_OK(merger.Schedule());
  auto res3 = GetMergeResult(merger);
  ASSERT_OK(res3);
  EXPECT_EQ(res3->confirmed_text, "hello world");
  EXPECT_EQ(res3->unconfirmed_text, "new words");
}

TEST(TimestampTextMergerTest, PartialWordAtWindowStartIsDropped) {
  DummyWordStage word_stage;
  TimestampTextMerger merger(&word_stage, /*overlap_ratio=*/0.4f);

  word_stage.PushWordsWithEndOfChunkTimestamp({{"the", 500},
                                               {"quick", 1000},
                                               {"brown", 1500},
                                               {"fox", 2000},
                                               {"jumps", 2900},
                                               {"over", 3400},
                                               {"the", 3800},
                                               {"lazy", 4200},
                                               {"dog", 4600}},
                                              /*end_ts_ms=*/5000);
  ASSERT_OK(merger.Schedule());
  ASSERT_OK(GetMergeResult(merger));

  // Chunk 2 starts at 3000 ms, in the middle of "jumps", and decodes its tail.
  word_stage.PushWordsWithEndOfChunkTimestamp({{"umps", 0},
                                               {"over", 400},
                                               {"the", 800},
                                               {"lazy", 1200},
                                               {"dog", 1600},
                                               {"and", 2200},
                                               {"runs", 2600}},
                                              /*end_ts_ms=*/5000);
  ASSERT_OK(merger.Schedule());
  auto res2 = GetMergeResult(merger);
  ASSERT_OK(res2);
  EXPECT_EQ(res2->confirmed_text, "the quick brown fox jumps");
  EXPECT_EQ(res2->unconfirmed_text, "over the lazy dog and runs");
}

TEST(TimestampTextMergerTest, StreamOffsetKeepsFractionalStep) {
  DummyWordStage word_stage;
  TimestampTextMerger merger(&word_stage, /*overlap_ratio=*/0.4f);

  // Encoder-frame timestamps: 63 frames per chunk, step 37.8 frames.
  for (int i = 0; i < 5; ++i) {
    word_stage.PushWordsWithEndOfChunkTimestamp({{"word", 10}},
                                                /*end_ts_ms=*/63);
    ASSERT_OK(merger.Schedule());
    ASSERT_OK(GetMergeResult(merger));
  }
  EXPECT_NEAR(merger.stream_timestamp_offset_for_testing(), 5 * 37.8, 0.01);
}

TEST(TimestampTextMergerTest, RepeatedWordsHandlingWithTimestamps) {
  DummyWordStage word_stage;
  TimestampTextMerger merger(&word_stage, /*overlap_ratio=*/0.5f);

  // Repeated words "the the the" with distinct timestamps
  word_stage.PushWordsWithEndOfChunkTimestamp(
      {{"the", 500}, {"the", 1500}, {"the", 2500}, {"end", 3000}},
      /*end_ts_ms=*/5000);
  ASSERT_OK(merger.Schedule());
  auto res1 = GetMergeResult(merger);
  ASSERT_OK(res1);

  word_stage.PushWordsWithEndOfChunkTimestamp(
      {{"the", 0}, {"end", 500}, {"next", 1000}},
      /*end_ts_ms=*/5000);
  ASSERT_OK(merger.Schedule());
  auto res2 = GetMergeResult(merger);
  ASSERT_OK(res2);

  ASSERT_OK(merger.Flush());
  auto res_flush = GetMergeResult(merger);
  ASSERT_OK(res_flush);
  EXPECT_EQ(res_flush->confirmed_text, "the end next");
  EXPECT_EQ(res_flush->unconfirmed_text, "");
}

TEST(TimestampTextMergerTest,
     EmptyChunkWithEndOfChunkSentinelRetainsUnconfirmedWords) {
  DummyWordStage word_stage;
  TimestampTextMerger merger(&word_stage, /*overlap_ratio=*/0.5f);

  word_stage.PushWordsWithEndOfChunkTimestamp({{"hello", 500}, {"world", 1000}},
                                              /*end_ts_ms=*/5000);
  ASSERT_OK(merger.Schedule());
  auto res1 = GetMergeResult(merger);
  ASSERT_OK(res1);
  EXPECT_EQ(res1->unconfirmed_text, "hello world");

  // Chunk 2 has only the end-of-chunk sentinel (no spoken words).
  word_stage.PushWordsWithEndOfChunkTimestamp({}, /*end_ts_ms=*/5000);
  ASSERT_OK(merger.Schedule());
  auto res2 = GetMergeResult(merger);
  ASSERT_OK(res2);
  EXPECT_EQ(res2->confirmed_text, "");
  EXPECT_EQ(res2->unconfirmed_text, "hello world");
}

TEST(TimestampTextMergerTest, SequentialMergeFlowWithoutTimestamps) {
  DummyWordStage word_stage;
  TimestampTextMerger merger(&word_stage, /*overlap_ratio=*/0.5f);

  auto res1 =
      ProcessWords(word_stage, merger, {"hello", "world", "this", "is"});
  ASSERT_OK(res1);
  EXPECT_EQ(res1->confirmed_text, "");
  EXPECT_EQ(res1->unconfirmed_text, "hello world this is");

  auto res2 = ProcessWords(word_stage, merger, {"this", "is", "a", "test"});
  ASSERT_OK(res2);
  EXPECT_EQ(res2->confirmed_text, "hello world");
  EXPECT_EQ(res2->unconfirmed_text, "this is a test");
}

TEST(TimestampTextMergerTest,
     ShortWordBetweenMatchedWordsBeforePivotIsPreserved) {
  DummyWordStage word_stage;
  TimestampTextMerger merger(&word_stage, /*overlap_ratio=*/0.5f);

  // Chunk 1 (frames 0..63): missed short word "I" between "were." (42) and
  // "know" (47). Both 42 and 47 are within tol.substitution (4.725 frames) of
  // frame 44, but > 1.0 frame away.
  word_stage.PushWordsWithEndOfChunkTimestamp(
      {{"as", 35}, {"it", 38}, {"were.", 42}, {"know", 47}, {"most", 52}},
      /*end_ts_ms=*/63);
  ASSERT_OK(merger.Schedule());
  ASSERT_OK(GetMergeResult(merger));

  // Chunk 2 (step=31.5 -> chunk_start_ms=32): decodes "were." (10 -> 42),
  // "I" (12 -> 44), "know" (15 -> 47), "most" (20 -> 52), "of" (24 -> 56).
  // Because "were." and "know" in prev are matched, they do not suppress "I"
  // (at 44 < pivot) via covered_by_prev.
  word_stage.PushWordsWithEndOfChunkTimestamp({{"as", 3},
                                               {"it", 6},
                                               {"were.", 10},
                                               {"I", 12},
                                               {"know", 15},
                                               {"most", 20},
                                               {"of", 24}},
                                              /*end_ts_ms=*/63);
  ASSERT_OK(merger.Schedule());
  ASSERT_OK(GetMergeResult(merger));

  ASSERT_OK(merger.Flush());
  ASSERT_OK_AND_ASSIGN(auto flushed, GetMergeResult(merger));
  EXPECT_EQ(flushed.confirmed_text, "as it were. I know most of");
}

TEST(TimestampTextMergerTest,
     MergedTimestampsRemainSortedWhenShiftedCurrentOnlyWordPrecedesPair) {
  DummyWordStage word_stage;
  TimestampTextMerger merger(&word_stage, /*overlap_ratio=*/0.4f);

  // Chunk 1 (0-5000).
  word_stage.PushWordsWithEndOfChunkTimestamp({{"the", 500},
                                               {"stale", 1000},
                                               {"smell", 2000},
                                               {"alpha", 3200},
                                               {"beta", 3600},
                                               {"gamma", 3800},
                                               {"delta", 4400},
                                               {"eps", 4800}},
                                              /*end_ts_ms=*/5000);
  ASSERT_OK(merger.Schedule());
  ASSERT_OK(GetMergeResult(merger));

  // Chunk 2 (3000-8000) emits every shared word 300 ms later than chunk 1
  // (shift = +300) and hears an extra word "new" between "beta" and "gamma".
  word_stage.PushWordsWithEndOfChunkTimestamp({{"alpha", 500},
                                               {"beta", 900},
                                               {"new", 950},
                                               {"gamma", 1100},
                                               {"delta", 1700},
                                               {"eps", 2100},
                                               {"zeta", 2500},
                                               {"eta", 3300},
                                               {"theta", 3700}},
                                              /*end_ts_ms=*/5000);
  ASSERT_OK(merger.Schedule());
  auto res2 = GetMergeResult(merger);
  ASSERT_OK(res2);
  EXPECT_EQ(res2->confirmed_text, "the stale smell");
  EXPECT_EQ(res2->unconfirmed_text,
            "alpha beta new gamma delta eps zeta eta theta");

  // Chunk 3 (6000-11000) hears "eta" but misses "theta" (gap) and adds
  // "kappa". With the time-based merge preserved, "theta" (6700 < pivot 7200)
  // is kept.
  word_stage.PushWordsWithEndOfChunkTimestamp({{"eta", 300}, {"kappa", 3500}},
                                              /*end_ts_ms=*/5000);
  ASSERT_OK(merger.Schedule());
  auto res3 = GetMergeResult(merger);
  ASSERT_OK(res3);
  EXPECT_EQ(res3->confirmed_text, "alpha beta new gamma delta eps zeta");
  EXPECT_EQ(res3->unconfirmed_text, "eta theta kappa");
}

TEST(TimestampTextMergerTest, SilentChunkWithoutSentinelAdvancesStreamOffset) {
  DummyWordStage word_stage;
  TimestampTextMerger merger(&word_stage, /*overlap_ratio=*/0.4f);

  // Chunk 1 (0-5000).
  word_stage.PushWordsWithEndOfChunkTimestamp({{"the", 500},
                                               {"stale", 1000},
                                               {"smell", 2000},
                                               {"of", 2900},
                                               {"lingers", 4000},
                                               {"it", 4400},
                                               {"takes", 4800}},
                                              /*end_ts_ms=*/5000);
  ASSERT_OK(merger.Schedule());
  ASSERT_OK(GetMergeResult(merger));
  EXPECT_NEAR(merger.stream_timestamp_offset_for_testing(), 3000, 0.01);

  // Chunk 2 (3000-8000) is silent: no words, no end-of-chunk sentinel.
  word_stage.PushWords({});
  ASSERT_OK(merger.Schedule());
  ASSERT_OK(GetMergeResult(merger));
  EXPECT_NEAR(merger.stream_timestamp_offset_for_testing(), 6000, 0.01);

  // Chunk 3 (6000-11000): "the end" right after the silence.
  word_stage.PushWordsWithEndOfChunkTimestamp({{"the", 200}, {"end", 600}},
                                              /*end_ts_ms=*/5000);
  ASSERT_OK(merger.Schedule());
  auto res3 = GetMergeResult(merger);
  ASSERT_OK(res3);
  EXPECT_EQ(res3->confirmed_text, "the stale smell of lingers it takes");
  EXPECT_EQ(res3->unconfirmed_text, "the end");
}

TEST(TimestampTextMergerTest,
     NewWordAtWindowStartNotDroppedAfterFullySpokenPreviousWord) {
  DummyWordStage word_stage;
  TimestampTextMerger merger(&word_stage, /*overlap_ratio=*/0.4f);

  // Chunk 1 (0-5000) heard "fox" at 2900 ms, missed "over" at 3000 ms, and
  // heard "the lazy dog" at 3800..4600 ms.
  word_stage.PushWordsWithEndOfChunkTimestamp({{"the", 500},
                                               {"quick", 1000},
                                               {"brown", 1500},
                                               {"fox", 2900},
                                               {"the", 3800},
                                               {"lazy", 4200},
                                               {"dog", 4600}},
                                              /*end_ts_ms=*/5000);
  ASSERT_OK(merger.Schedule());
  ASSERT_OK(GetMergeResult(merger));

  // Chunk 2 (3000-8000) hears "over" at 0 ms (3000 ms absolute), which is not
  // a partial suffix of "fox" and should be preserved.
  word_stage.PushWordsWithEndOfChunkTimestamp(
      {{"over", 0}, {"the", 800}, {"lazy", 1200}, {"dog", 1600}},
      /*end_ts_ms=*/5000);
  ASSERT_OK(merger.Schedule());
  auto res2 = GetMergeResult(merger);
  ASSERT_OK(res2);
  EXPECT_EQ(res2->confirmed_text, "the quick brown fox");
  EXPECT_EQ(res2->unconfirmed_text, "over the lazy dog");
}

TEST(TimestampTextMergerTest, MissedWordAfterMatchedInWindowWordIsKept) {
  DummyWordStage word_stage;
  TimestampTextMerger merger(&word_stage, /*overlap_ratio=*/0.4f);

  // Chunk 1 (0-5000) heard "the" at 3000 but missed "big" at 3250.
  word_stage.PushWordsWithEndOfChunkTimestamp(
      {{"see", 500}, {"the", 3000}, {"lazy", 3800}, {"dog", 4200}},
      /*end_ts_ms=*/5000);
  ASSERT_OK(merger.Schedule());
  ASSERT_OK(GetMergeResult(merger));

  // Chunk 2 (3000-8000) hears "the big lazy dog runs".
  word_stage.PushWordsWithEndOfChunkTimestamp(
      {{"the", 0}, {"big", 250}, {"lazy", 800}, {"dog", 1200}, {"runs", 2000}},
      /*end_ts_ms=*/5000);
  ASSERT_OK(merger.Schedule());
  auto res2 = GetMergeResult(merger);
  ASSERT_OK(res2);
  EXPECT_EQ(res2->confirmed_text, "see");
  EXPECT_EQ(res2->unconfirmed_text, "the big lazy dog runs");
}

TEST(TimestampTextMergerTest, WordCountPrevTailBranchReseedsTimestamps) {
  DummyWordStage word_stage;
  TimestampTextMerger merger(&word_stage, /*overlap_ratio=*/0.4f);

  // Chunk 1 without timestamps.
  word_stage.PushWords({"a", "b", "c", "d", "e", "f", "g", "h"});
  ASSERT_OK(merger.Schedule());
  ASSERT_OK(GetMergeResult(merger));

  // Chunk 2 (0-5000) with timestamps: only "e f g" (fewer words after the
  // pivot than the previous tail) -> MergeByWordCount, prev-tail branch.
  word_stage.PushWordsWithEndOfChunkTimestamp(
      {{"e", 100}, {"f", 3500}, {"g", 4000}},
      /*end_ts_ms=*/5000);
  ASSERT_OK(merger.Schedule());
  ASSERT_OK(GetMergeResult(merger));

  // Chunk 3 (3000-8000) with timestamps overlapping chunk 2's window: hears
  // "f" at 500 (3500 absolute) and "i" at 2500 (5500 absolute), missing "g"
  // and "h" (both synthesized at 4000 < pivot 4200). Because chunk 2 re-seeded
  // prev_timestamps_, chunk 3 uses MergeByTime and preserves "g h" before the
  // pivot.
  word_stage.PushWordsWithEndOfChunkTimestamp({{"f", 500}, {"i", 2500}},
                                              /*end_ts_ms=*/5000);
  ASSERT_OK(merger.Schedule());
  auto res3 = GetMergeResult(merger);
  ASSERT_OK(res3);
  EXPECT_EQ(res3->confirmed_text, "e");
  EXPECT_EQ(res3->unconfirmed_text, "f g h i");
}

TEST(TimestampTextMergerTest,
     EstimateTimestampShiftIgnoresSingleCharAndOutOfOverlapWords) {
  DummyWordStage word_stage;
  TimestampTextMerger merger(&word_stage, /*overlap_ratio=*/0.4f);

  // Chunk 1 (0-5000): "early" is at 2800 ms (before chunk 2's 3000 ms start),
  // followed by three single-char tokens at 3100, 3500, 3900 and two
  // multi-char words at 4200 and 4600.
  word_stage.PushWordsWithEndOfChunkTimestamp({{"early", 2800},
                                               {"a", 3100},
                                               {"i", 3500},
                                               {"o", 3900},
                                               {"signal", 4200},
                                               {"detected", 4600}},
                                              /*end_ts_ms=*/5000);
  ASSERT_OK(merger.Schedule());
  ASSERT_OK(GetMergeResult(merger));

  // Chunk 2 (3000-8000): the three single-char tokens appear at +300 ms
  // (3400, 3800, 4200), whereas the multi-char words "signal" and "detected"
  // have zero shift (4200, 4600). Because single-char words are ignored by
  // EstimateTimestampShift, shift is 0 (chunk_start = 3000) rather than +300
  // (which would lower chunk_start to 2700 and leave "early" unconfirmed).
  word_stage.PushWordsWithEndOfChunkTimestamp({{"a", 400},
                                               {"i", 800},
                                               {"o", 1200},
                                               {"signal", 1200},
                                               {"detected", 1600},
                                               {"now", 2500}},
                                              /*end_ts_ms=*/5000);
  ASSERT_OK(merger.Schedule());
  auto res2 = GetMergeResult(merger);
  ASSERT_OK(res2);
  EXPECT_EQ(res2->confirmed_text, "early");
}

TEST(TimestampTextMergerTest,
     BoundaryFragmentSharingSubstringOrFollowingNumericWordIsDropped) {
  DummyWordStage word_stage;
  TimestampTextMerger merger(&word_stage, /*overlap_ratio=*/0.4f);

  // Chunk 1 (0-5000): "station" starts at 2900 ms (just before chunk 2's
  // 3000 ms boundary).
  word_stage.PushWordsWithEndOfChunkTimestamp(
      {{"next", 1000}, {"station", 2900}, {"ahead", 3800}, {"now", 4400}},
      /*end_ts_ms=*/5000);
  ASSERT_OK(merger.Schedule());
  ASSERT_OK(GetMergeResult(merger));

  // Chunk 2 (3000-8000): hears cut fragment "ational" at 50 ms (3050 ms
  // absolute), which shares a 3-char substring ("ati"/"tio"/"ion") with
  // "station" at 2900 ms and is dropped by IsCutAtWindowStart.
  word_stage.PushWordsWithEndOfChunkTimestamp(
      {{"ational", 50}, {"ahead", 800}, {"now", 1400}, {"please", 2200}},
      /*end_ts_ms=*/5000);
  ASSERT_OK(merger.Schedule());
  auto res2 = GetMergeResult(merger);
  ASSERT_OK(res2);
  EXPECT_EQ(res2->confirmed_text, "next station");
  EXPECT_EQ(res2->unconfirmed_text, "ahead now please");

  // Also verify a boundary word following a numeric word at the window start:
  merger.Reset();
  word_stage.PushWordsWithEndOfChunkTimestamp(
      {{"room", 1000}, {"105", 2900}, {"is", 3800}, {"open", 4400}},
      /*end_ts_ms=*/5000);
  ASSERT_OK(merger.Schedule());
  ASSERT_OK(GetMergeResult(merger));

  // Chunk 2 hears "five" at 50 ms (3050 ms absolute) from the tail of "105".
  word_stage.PushWordsWithEndOfChunkTimestamp(
      {{"five", 50}, {"is", 800}, {"open", 1400}, {"today", 2200}},
      /*end_ts_ms=*/5000);
  ASSERT_OK(merger.Schedule());
  auto res_num = GetMergeResult(merger);
  ASSERT_OK(res_num);
  EXPECT_EQ(res_num->confirmed_text, "room 105");
  EXPECT_EQ(res_num->unconfirmed_text, "is open today");
}

TEST(TimestampTextMergerTest,
     DeduplicatesTrailingSuffixOfHyphenatedAndApostropheWords) {
  DummyWordStage word_stage;
  TimestampTextMerger merger(&word_stage, /*overlap_ratio=*/0.4f);

  // Chunk 1 (0-5000) decodes "twenty-five" at 3400 ms and "it's" at 3900 ms.
  word_stage.PushWordsWithEndOfChunkTimestamp(
      {{"count", 1000}, {"twenty-five", 3400}, {"it's", 3900}, {"ready", 4400}},
      /*end_ts_ms=*/5000);
  ASSERT_OK(merger.Schedule());
  ASSERT_OK(GetMergeResult(merger));

  // Chunk 2 (3000-8000) pairs "twenty-five" (before pivot 4200, so kept from
  // prev) and also emits an unpaired trailing "five" at 4250 ms and "s" right
  // after "it's". AppendMergedWord deduplicates both trailing suffixes.
  word_stage.PushWordsWithEndOfChunkTimestamp({{"twenty-five", 400},
                                               {"five", 1250},
                                               {"it's", 1300},
                                               {"s", 1350},
                                               {"ready", 1400},
                                               {"now", 2200}},
                                              /*end_ts_ms=*/5000);
  ASSERT_OK(merger.Schedule());
  auto res2 = GetMergeResult(merger);
  ASSERT_OK(res2);
  EXPECT_EQ(res2->confirmed_text, "count");
  EXPECT_EQ(res2->unconfirmed_text, "twenty-five it's ready now");
}

TEST(TimestampTextMergerTest,
     PairedSubstitutionAtPivotPrefersPreviousNumericWord) {
  DummyWordStage word_stage;
  TimestampTextMerger merger(&word_stage, /*overlap_ratio=*/0.4f);

  // Chunk 1 (0-63 frames, step=37.8, pivot=37.8 + 0.6*(63-37.8) = 52.92):
  // Numeric word "25" is at frame 53 (within pivot + 1.0).
  word_stage.PushWordsWithEndOfChunkTimestamp(
      {{"page", 20}, {"number", 42}, {"is", 48}, {"25", 53}, {"today", 59}},
      /*end_ts_ms=*/63);
  ASSERT_OK(merger.Schedule());
  ASSERT_OK(GetMergeResult(merger));

  // Chunk 2 (chunk_start_ms=38) pairs at frame 53 with non-numeric "twenty"
  // (15 + 38 = 53). Because prev_ov_words[i] starts with a digit and is within
  // pivot + 1.0, "25" is preferred over "twenty".
  word_stage.PushWordsWithEndOfChunkTimestamp(
      {{"number", 4}, {"is", 10}, {"twenty", 15}, {"today", 21}, {"ok", 35}},
      /*end_ts_ms=*/63);
  ASSERT_OK(merger.Schedule());
  auto res2 = GetMergeResult(merger);
  ASSERT_OK(res2);
  EXPECT_EQ(res2->confirmed_text, "page");
  EXPECT_EQ(res2->unconfirmed_text, "number is 25 today ok");
}

TEST(TimestampTextMergerTest,
     DropsShortUnpairedTailWordAtEndOfPreviousChunkWindow) {
  DummyWordStage word_stage;
  TimestampTextMerger merger(&word_stage, /*overlap_ratio=*/0.4f);

  // Chunk 1 (0-63 frames) has a short 2-char truncated word "th" right at the
  // end of the chunk (frame 62 >= prev_end_ms - 1.0).
  word_stage.PushWordsWithEndOfChunkTimestamp(
      {{"start", 20}, {"reading", 42}, {"now", 48}, {"th", 62}},
      /*end_ts_ms=*/63);
  ASSERT_OK(merger.Schedule());
  ASSERT_OK(GetMergeResult(merger));

  // Chunk 2 (chunk_start_ms=38) hears "reading" (42), "now" (48), and
  // continues with "complete" at frame 75 (37 + 38), leaving "th" at 62
  // unpaired. Because "th" is a short (<= 3 chars) final word of chunk 1 at
  // the window end, it is dropped rather than kept as a gap word.
  word_stage.PushWordsWithEndOfChunkTimestamp(
      {{"reading", 4}, {"now", 10}, {"complete", 37}},
      /*end_ts_ms=*/63);
  ASSERT_OK(merger.Schedule());
  auto res2 = GetMergeResult(merger);
  ASSERT_OK(res2);
  EXPECT_EQ(res2->confirmed_text, "start");
  EXPECT_EQ(res2->unconfirmed_text, "reading now complete");
}

TEST(TimestampTextMergerTest,
     BoundaryWordFollowingNumericWordIsNotDroppedWhenNotAnEcho) {
  DummyWordStage word_stage;
  TimestampTextMerger merger(&word_stage, /*overlap_ratio=*/0.4f);

  // Chunk 1 (0-5000): ends with "100" at 2900 ms.
  word_stage.PushWordsWithEndOfChunkTimestamp({{"cost", 1000}, {"100", 2900}},
                                              /*end_ts_ms=*/5000);
  ASSERT_OK(merger.Schedule());
  ASSERT_OK(GetMergeResult(merger));

  // Chunk 2 (3000-8000): begins with "dollars" at 50 ms (3050 ms absolute).
  // "dollars" is a valid word starting the next clause and not a spoken echo of
  // "100" (like "hundred"), so it must NOT be treated as a cut fragment.
  word_stage.PushWordsWithEndOfChunkTimestamp(
      {{"dollars", 50}, {"is", 800}, {"due", 1400}, {"today", 2200}},
      /*end_ts_ms=*/5000);
  ASSERT_OK(merger.Schedule());
  auto res = GetMergeResult(merger);
  ASSERT_OK(res);
  EXPECT_EQ(res->confirmed_text, "cost 100");
  EXPECT_EQ(res->unconfirmed_text, "dollars is due today");

  // Also verify "page 25 begins":
  merger.Reset();
  word_stage.PushWordsWithEndOfChunkTimestamp({{"page", 1000}, {"25", 2900}},
                                              /*end_ts_ms=*/5000);
  ASSERT_OK(merger.Schedule());
  ASSERT_OK(GetMergeResult(merger));

  // Chunk 2 begins with "begins" at 50 ms. "begins" is not an echo of "25"
  // (unlike "five" or "twenty"), so it must NOT be treated as a cut fragment.
  word_stage.PushWordsWithEndOfChunkTimestamp(
      {{"begins", 50}, {"with", 800}, {"chapter", 1400}, {"one", 2200}},
      /*end_ts_ms=*/5000);
  ASSERT_OK(merger.Schedule());
  auto res2 = GetMergeResult(merger);
  ASSERT_OK(res2);
  EXPECT_EQ(res2->confirmed_text, "page 25");
  EXPECT_EQ(res2->unconfirmed_text, "begins with chapter one");
}

TEST(TimestampTextMergerTest,
     DoesNotDropWordsSharingSubstringOrEndingWithUnrelatedBoundaryWord) {
  DummyWordStage word_stage;
  TimestampTextMerger merger(&word_stage, /*overlap_ratio=*/0.4f);

  // Chunk 1 (0-5000): ends with "weather" at 2900 ms (just before chunk 2's
  // 3000 ms boundary).
  word_stage.PushWordsWithEndOfChunkTimestamp(
      {{"the", 1000}, {"weather", 2900}},
      /*end_ts_ms=*/5000);
  ASSERT_OK(merger.Schedule());
  ASSERT_OK(GetMergeResult(merger));

  // Chunk 2 (3000-8000): starts with "the" at 50 ms (3050 ms absolute).
  // While "weather" and "the" both end in 'e' or share the trigram "the",
  // "the" is a valid word and not a boundary cut fragment of "weather".
  word_stage.PushWordsWithEndOfChunkTimestamp(
      {{"the", 50}, {"sky", 800}, {"is", 1400}, {"clear", 2000}},
      /*end_ts_ms=*/5000);
  ASSERT_OK(merger.Schedule());
  auto res = GetMergeResult(merger);
  ASSERT_OK(res);
  EXPECT_EQ(res->confirmed_text, "the weather");
  EXPECT_EQ(res->unconfirmed_text, "the sky is clear");
}

TEST(TimestampTextMergerTest,
     HyphenatedWordDoesNotDropSubstringsNotPrecededByDelimiter) {
  DummyWordStage word_stage;
  TimestampTextMerger merger(&word_stage, /*overlap_ratio=*/0.4f);

  // Chunk 1 (0-5000): decodes "re-create" and "well-planned".
  word_stage.PushWordsWithEndOfChunkTimestamp(
      {{"we", 500}, {"re-create", 2000}, {"well-planned", 3500}},
      /*end_ts_ms=*/5000);
  ASSERT_OK(merger.Schedule());
  ASSERT_OK(GetMergeResult(merger));

  // Chunk 2 (3000-8000): hears "well-planned" (3500), followed by valid words
  // "ed" and "ate". Because neither follows a delimiter in the preceding word,
  // neither is dropped.
  word_stage.PushWordsWithEndOfChunkTimestamp(
      {{"well-planned", 500}, {"ed", 1200}, {"ate", 1800}, {"lunch", 2500}},
      /*end_ts_ms=*/5000);
  ASSERT_OK(merger.Schedule());
  auto res = GetMergeResult(merger);
  ASSERT_OK(res);
  EXPECT_EQ(res->confirmed_text, "we re-create");
  EXPECT_EQ(res->unconfirmed_text, "well-planned ed ate lunch");
}

TEST(TimestampTextMergerTest, HyphenatedSuffixEchoRespectsProximityGap) {
  DummyWordStage word_stage;
  TimestampTextMerger merger(&word_stage, /*overlap_ratio=*/0.4f);

  // Chunk 1 (0-5000): decodes "twenty-five" at 3400 ms.
  word_stage.PushWordsWithEndOfChunkTimestamp(
      {{"page", 1000}, {"twenty-five", 3400}},
      /*end_ts_ms=*/5000);
  ASSERT_OK(merger.Schedule());
  ASSERT_OK(GetMergeResult(merger));

  // Chunk 2 (3000-8000, overlap_len=2000, tol.match_late=1000 ms):
  // "five" appears at 2500 ms (5500 ms absolute), well after the overlap and
  // past tol.match_late from 3400 ms (|5500 - 3400| = 2100 > 1000).
  // It represents a genuine new sentence ("Five people arrived") and must NOT
  // be dropped.
  word_stage.PushWordsWithEndOfChunkTimestamp({{"twenty-five", 400},
                                               {"five", 2500},
                                               {"people", 3100},
                                               {"arrived", 3700}},
                                              /*end_ts_ms=*/5000);
  ASSERT_OK(merger.Schedule());
  auto res = GetMergeResult(merger);
  ASSERT_OK(res);
  EXPECT_EQ(res->confirmed_text, "page");
  EXPECT_EQ(res->unconfirmed_text, "twenty-five five people arrived");
}

TEST(TimestampTextMergerTest,
     DropsShortUnpairedTailWordAtEndOfPreviousChunkWindowInMilliseconds) {
  DummyWordStage word_stage;
  TimestampTextMerger merger(&word_stage, /*overlap_ratio=*/0.4f);

  // Chunk 1 (0-5000 ms): has a short 2-char truncated word "th" at 4950 ms,
  // right at the end of the 5000 ms window (within 5000 - tol.substitution).
  word_stage.PushWordsWithEndOfChunkTimestamp(
      {{"start", 1000}, {"reading", 3200}, {"now", 4200}, {"th", 4950}},
      /*end_ts_ms=*/5000);
  ASSERT_OK(merger.Schedule());
  ASSERT_OK(GetMergeResult(merger));

  // Chunk 2 (3000-8000 ms, chunk_start_ms=3000): hears "reading" (200),
  // "now" (1200), and continues with "complete" at 2500 (5500 ms absolute),
  // leaving "th" at 4950 unpaired. Because "th" is at the end of chunk 1's
  // window, it is dropped as an unconfirmed prev tail.
  word_stage.PushWordsWithEndOfChunkTimestamp(
      {{"reading", 200}, {"now", 1200}, {"complete", 2500}},
      /*end_ts_ms=*/5000);
  ASSERT_OK(merger.Schedule());
  auto res2 = GetMergeResult(merger);
  ASSERT_OK(res2);
  EXPECT_EQ(res2->confirmed_text, "start");
  EXPECT_EQ(res2->unconfirmed_text, "reading now complete");
}

// Runs `windows` (one space-free CJK hypothesis per window, as emitted by
// Qwen3-ASR) through `merger` and returns all confirmed text plus the flush.
std::string MergeWindows(DummyWordStage& word_stage,
                         TimestampTextMerger& merger,
                         const std::vector<std::string>& windows) {
  std::string text;
  for (const std::string& window : windows) {
    auto result = ProcessWords(word_stage, merger, {window});
    EXPECT_OK(result);
    if (result.ok()) text += result->confirmed_text;
  }
  EXPECT_OK(merger.Flush());
  auto flushed = GetMergeResult(merger);
  EXPECT_OK(flushed);
  if (flushed.ok()) text += flushed->confirmed_text;
  return text;
}

TEST(TimestampTextMergerTest, ChineseWindowsWithoutSpacesAreNotDuplicated) {
  DummyWordStage word_stage;
  TimestampTextMerger merger(&word_stage, /*overlap_ratio=*/0.4f);

  // Live Qwen3-ASR output for 5 s windows with a 2 s overlap: each window
  // repeats the tail of the previous one, and Chinese has no spaces between
  // words.
  const std::string text = MergeWindows(
      word_stage, merger,
      {"你好吗？今天是晚上吃什么？北京有什么好玩的？",
       "北京有什么好玩的？这个秋天，是不是还可以？", "是不是还可以找个地方玩？",
       "这个中文的重复，为什么会中文会有重复呢？", "中文会有重复的。"});

  // The last window re-hears the previous window's tail ("中文会有重复呢？")
  // as "中文会有重复的。"; after the pivot the current window wins.
  EXPECT_EQ(text,
            "你好吗？今天是晚上吃什么？北京有什么好玩的？这个秋天，"
            "是不是还可以找个地方玩？这个中文的重复，为什么会中文会有重复的。");
}

TEST(TimestampTextMergerTest, ChineseWindowEndingMidSentenceIsMerged) {
  DummyWordStage word_stage;
  TimestampTextMerger merger(&word_stage, /*overlap_ratio=*/0.4f);

  // The first window ends mid-sentence, so the model closes it with a period
  // that the next window does not have.
  const std::string text = MergeWindows(
      word_stage, merger,
      {"你可以说中文吗？好像中文的。", "好像中文的质量还不错，可以做到实时吗？",
       "可以做到实时吗？可以的。"});

  EXPECT_EQ(text,
            "你可以说中文吗？好像中文的质量还不错，可以做到实时吗？可以的。");
}

TEST(TimestampTextMergerTest, MixedChineseAndLatinKeepsWordSpacing) {
  DummyWordStage word_stage;
  TimestampTextMerger merger(&word_stage, /*overlap_ratio=*/0.4f);

  auto result =
      ProcessWords(word_stage, merger,
                   {"欢迎体验", "Light", "RT", "Omni", "语音合成演示。"});
  ASSERT_OK(result);

  EXPECT_EQ(result->unconfirmed_text, "欢迎体验 Light RT Omni 语音合成演示。");
}

TEST(TimestampTextMergerTest, TimestampedChineseSentenceWordIsNotDuplicated) {
  DummyWordStage word_stage;
  TimestampTextMerger merger(&word_stage, /*overlap_ratio=*/0.4f);

  // A detokenizer that does not split CJK text emits each window as one word
  // carrying only its onset timestamp, plus the end of the chunk.
  word_stage.PushWordsWithEndOfChunkTimestamp(
      {{"你好吗？北京有什么好玩的？", 200}}, /*end_ts_ms=*/5000);
  ASSERT_OK(merger.Schedule());
  ASSERT_OK_AND_ASSIGN(auto res1, GetMergeResult(merger));
  word_stage.PushWordsWithEndOfChunkTimestamp(
      {{"北京有什么好玩的？这个秋天还可以。", 100}}, /*end_ts_ms=*/5000);
  ASSERT_OK(merger.Schedule());
  ASSERT_OK_AND_ASSIGN(auto res2, GetMergeResult(merger));
  ASSERT_OK(merger.Flush());
  ASSERT_OK_AND_ASSIGN(auto flushed, GetMergeResult(merger));

  EXPECT_EQ(res1.confirmed_text + res2.confirmed_text + flushed.confirmed_text,
            "你好吗？北京有什么好玩的？这个秋天还可以。");
}

TEST(TimestampTextMergerTest, ChineseCharactersWithTimestampsMergeByTime) {
  DummyWordStage word_stage;
  TimestampTextMerger merger(&word_stage, /*overlap_ratio=*/0.4f);

  // One word per character, as emitted by `TokenizerDetokenizer` for a
  // timestamped decoder. The second window starts at 3000 (= 5000 * 0.6), so
  // "北京好玩" (3200..3800) is heard by both windows.
  word_stage.PushWordsWithEndOfChunkTimestamp({{"你", 200},
                                               {"好", 1000},
                                               {"吗？", 2000},
                                               {"北", 3200},
                                               {"京", 3400},
                                               {"好", 3600},
                                               {"玩", 3800}},
                                              /*end_ts_ms=*/5000);
  ASSERT_OK(merger.Schedule());
  ASSERT_OK_AND_ASSIGN(auto res1, GetMergeResult(merger));
  word_stage.PushWordsWithEndOfChunkTimestamp({{"北", 200},
                                               {"京", 400},
                                               {"好", 600},
                                               {"玩", 800},
                                               {"的。", 1000},
                                               {"秋", 2500},
                                               {"天", 2700}},
                                              /*end_ts_ms=*/5000);
  ASSERT_OK(merger.Schedule());
  ASSERT_OK_AND_ASSIGN(auto res2, GetMergeResult(merger));
  ASSERT_OK(merger.Flush());
  ASSERT_OK_AND_ASSIGN(auto flushed, GetMergeResult(merger));

  EXPECT_EQ(res2.confirmed_text, "你好吗？");
  EXPECT_EQ(res1.confirmed_text + res2.confirmed_text + flushed.confirmed_text,
            "你好吗？北京好玩的。秋天");
}

TEST(TimestampTextMergerTest, WarmStartPrefixesAreReplacedByFirstFullWindow) {
  DummyWordStage word_stage;
  TimestampTextMerger merger(&word_stage, /*overlap_ratio=*/0.75f);

  // Warm-start 1 (0..1250ms, `is_warm_start = true`)
  word_stage.PushWordsWithEndOfChunkTimestamp({{"hello", 300}, {"wor", 900}},
                                              /*end_ts_ms=*/5000,
                                              /*is_warm_start=*/true);
  ASSERT_OK(merger.Schedule());
  ASSERT_OK_AND_ASSIGN(auto ws1, GetMergeResult(merger));
  EXPECT_EQ(ws1.confirmed_text, "");
  EXPECT_EQ(ws1.unconfirmed_text, "hello wor");

  // Warm-start 2 (0..2500ms, `is_warm_start = true`) corrects "wor" to "world"
  // without advancing stream_timestamp_offset_ms_.
  word_stage.PushWordsWithEndOfChunkTimestamp(
      {{"hello", 300}, {"world", 900}, {"this", 1800}},
      /*end_ts_ms=*/5000, /*is_warm_start=*/true);
  ASSERT_OK(merger.Schedule());
  ASSERT_OK_AND_ASSIGN(auto ws2, GetMergeResult(merger));
  EXPECT_EQ(ws2.confirmed_text, "");
  EXPECT_EQ(ws2.unconfirmed_text, "hello world this");

  // Warm-start 3 (empty partial window with `is_warm_start = true`) preserves
  // `prev_words_` so a transient decoder collapse on a longer prefix does not
  // blank out the displayed partial transcript before the first full window.
  word_stage.PushWordsWithEndOfChunkTimestamp({}, /*end_ts_ms=*/5000,
                                              /*is_warm_start=*/true);
  ASSERT_OK(merger.Schedule());
  ASSERT_OK_AND_ASSIGN(auto ws3, GetMergeResult(merger));
  EXPECT_EQ(ws3.confirmed_text, "");
  EXPECT_EQ(ws3.unconfirmed_text, "hello world this");

  // First full window (0..5000ms, `is_warm_start = false`) replaces the
  // warm-start hypothesis and advances stream_timestamp_offset_ms_ to 1250.
  word_stage.PushWordsWithEndOfChunkTimestamp({{"hello", 300},
                                               {"world", 900},
                                               {"this", 1800},
                                               {"is", 2600},
                                               {"parakeet", 3500}},
                                              /*end_ts_ms=*/5000);
  ASSERT_OK(merger.Schedule());
  ASSERT_OK_AND_ASSIGN(auto full1, GetMergeResult(merger));
  EXPECT_EQ(full1.confirmed_text, "");
  EXPECT_EQ(full1.unconfirmed_text, "hello world this is parakeet");

  // Second full window (1250..6250ms, `is_warm_start = false`) confirms words
  // before 1250ms ("hello world") plus stabilized overlap words before
  // `pivot - tol.substitution` ("this is") and updates the unconfirmed tail.
  word_stage.PushWordsWithEndOfChunkTimestamp(
      {{"this", 550}, {"is", 1350}, {"parakeet", 2250}, {"streaming", 4100}},
      /*end_ts_ms=*/5000);
  ASSERT_OK(merger.Schedule());
  ASSERT_OK_AND_ASSIGN(auto full2, GetMergeResult(merger));
  EXPECT_EQ(full2.confirmed_text, "hello world this is");
  EXPECT_EQ(full2.unconfirmed_text, "parakeet streaming");
}

TEST(TimestampTextMergerTest,
     FirstFullWindowWithTrimmedSilenceConfirmsWordsOnNextWindow) {
  DummyWordStage word_stage;
  TimestampTextMerger merger(&word_stage, /*overlap_ratio=*/0.75f);

  // First full window (0..5000ms) had trailing silence trimmed (e.g.
  // DecodeRange{0.0, 0.6}) so words only appear in the first 1500ms, with
  // `is_warm_start = false`.
  word_stage.PushWordsWithEndOfChunkTimestamp({{"call", 300}, {"mom", 800}},
                                              /*end_ts_ms=*/5000,
                                              /*is_warm_start=*/false);
  ASSERT_OK(merger.Schedule());
  ASSERT_OK_AND_ASSIGN(auto win1, GetMergeResult(merger));
  EXPECT_EQ(win1.confirmed_text, "");
  EXPECT_EQ(win1.unconfirmed_text, "call mom");

  // Next full window (1250..6250ms) contains new speech after the pause at
  // 3500ms relative (4750ms absolute). "call mom" must be confirmed, not
  // replaced.
  word_stage.PushWordsWithEndOfChunkTimestamp({{"right", 3500}, {"now", 4000}},
                                              /*end_ts_ms=*/5000,
                                              /*is_warm_start=*/false);
  ASSERT_OK(merger.Schedule());
  ASSERT_OK_AND_ASSIGN(auto win2, GetMergeResult(merger));
  EXPECT_EQ(win2.confirmed_text, "call mom");
  EXPECT_EQ(win2.unconfirmed_text, "right now");
}

TEST(TimestampTextMergerTest,
     ConsecutiveSilentChunksFinalizePreviousWordsDuringLiveSilence) {
  DummyWordStage word_stage;
  TimestampTextMerger merger(&word_stage, /*overlap_ratio=*/0.75f);

  // Full window (0..5000ms) with speech ending at 2100ms.
  word_stage.PushWordsWithEndOfChunkTimestamp(
      {{"finalize", 800}, {"the", 1400}, {"transcripts", 2100}},
      /*end_ts_ms=*/5000);
  ASSERT_OK(merger.Schedule());
  ASSERT_OK_AND_ASSIGN(auto win1, GetMergeResult(merger));
  EXPECT_EQ(win1.confirmed_text, "");
  EXPECT_EQ(win1.unconfirmed_text, "finalize the transcripts");

  // First silent follow-up window (1250..6250ms) retains unconfirmed words in
  // case of a transient empty decode while advancing the stream offset to
  // 2500ms.
  word_stage.PushWordsWithEndOfChunkTimestamp({}, /*end_ts_ms=*/5000);
  ASSERT_OK(merger.Schedule());
  ASSERT_OK_AND_ASSIGN(auto silent1, GetMergeResult(merger));
  EXPECT_EQ(silent1.confirmed_text, "");
  EXPECT_EQ(silent1.unconfirmed_text, "finalize the transcripts");

  // Second consecutive silent window (2500..7500ms) starts after the last
  // spoken word's timestamp (2100ms < 2500ms) and immediately finalizes the
  // pending words without waiting for the speaker to talk again or stop the
  // stream.
  word_stage.PushWords({});
  ASSERT_OK(merger.Schedule());
  ASSERT_OK_AND_ASSIGN(auto silent2, GetMergeResult(merger));
  EXPECT_EQ(silent2.confirmed_text, "finalize the transcripts");
  EXPECT_EQ(silent2.unconfirmed_text, "");

  // Subsequent silent windows emit no duplicate confirmed or interim text.
  word_stage.PushWords({});
  ASSERT_OK(merger.Schedule());
  ASSERT_OK_AND_ASSIGN(auto silent3, GetMergeResult(merger));
  EXPECT_EQ(silent3.confirmed_text, "");
  EXPECT_EQ(silent3.unconfirmed_text, "");

  // When speech resumes after the pause (e.g. at 3750..8750ms) and spread goes
  // across two windows, a single gated silent follow-up chunk (`PushWords({})`)
  // immediately finalizes the remaining unconfirmed words without waiting for a
  // second silent chunk.
  word_stage.PushWordsWithEndOfChunkTimestamp(
      {{"speaking", 1000}, {"again", 2000}}, /*end_ts_ms=*/5000);
  ASSERT_OK(merger.Schedule());
  ASSERT_OK_AND_ASSIGN(auto resume1, GetMergeResult(merger));
  EXPECT_EQ(resume1.confirmed_text, "");
  EXPECT_EQ(resume1.unconfirmed_text, "speaking again");

  word_stage.PushWords({});
  ASSERT_OK(merger.Schedule());
  ASSERT_OK_AND_ASSIGN(auto gated_silent, GetMergeResult(merger));
  EXPECT_EQ(gated_silent.confirmed_text, "speaking again");
  EXPECT_EQ(gated_silent.unconfirmed_text, "");
}

}  // namespace
}  // namespace litert::omni::asr
