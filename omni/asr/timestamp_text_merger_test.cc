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

#include "omni/asr/timestamp_text_merger.h"

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
#include "omni/asr/detokenizer.h"
#include "omni/asr/text_merger.h"
#include "omni/base/stage.h"
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
      int end_ts_ms) {
    std::vector<Detokenizer::Word> words;
    words.reserve(word_ts_pairs.size() + 1);
    for (const auto& pair : word_ts_pairs) {
      words.push_back({pair.first, pair.second});
    }
    words.push_back({"", end_ts_ms});
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

TEST(TimestampTextMergerTest, HallucinatedBurstAtPreviousWindowEndIsDropped) {
  DummyWordStage word_stage;
  TimestampTextMerger merger(&word_stage, /*overlap_ratio=*/0.4f);

  // Chunk 1 (0-5000 ms) ends with a burst of three words at one timestamp.
  word_stage.PushWordsWithEndOfChunkTimestamp({{"the", 500},
                                               {"stale", 1000},
                                               {"smell", 2000},
                                               {"of", 2500},
                                               {"old", 3200},
                                               {"beer", 3600},
                                               {"lingers", 4000},
                                               {"it", 4400},
                                               {"takes", 4800},
                                               {"a", 4900},
                                               {"little", 4900},
                                               {"bit", 4900}},
                                              /*end_ts_ms=*/5000);
  ASSERT_OK(merger.Schedule());
  ASSERT_OK(GetMergeResult(merger));

  // Chunk 2 starts at 3000 ms (step = 5000 * 0.6).
  word_stage.PushWordsWithEndOfChunkTimestamp({{"old", 200},
                                               {"beer", 600},
                                               {"lingers", 1000},
                                               {"it", 1400},
                                               {"takes", 1800},
                                               {"heat", 2200},
                                               {"to", 2600},
                                               {"bring", 3000},
                                               {"out", 3400}},
                                              /*end_ts_ms=*/5000);
  ASSERT_OK(merger.Schedule());
  auto res2 = GetMergeResult(merger);
  ASSERT_OK(res2);
  EXPECT_EQ(res2->confirmed_text, "the stale smell of");
  EXPECT_EQ(res2->unconfirmed_text,
            "old beer lingers it takes heat to bring out");
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

TEST(TimestampTextMergerTest, CompoundWordSplitAcrossPivotPrefersTwoWordSplit) {
  DummyWordStage word_stage;
  TimestampTextMerger merger(&word_stage, /*overlap_ratio=*/0.4f);

  // Chunk 1: "in the mudfaucers doesn't" near the end of the 5s window.
  word_stage.PushWordsWithEndOfChunkTimestamp(
      {{"in", 3000}, {"the", 3400}, {"mudfaucers", 3900}, {"doesn't", 4500}},
      /*end_ts_ms=*/5000);
  ASSERT_OK(merger.Schedule());
  ASSERT_OK(GetMergeResult(merger));

  // Chunk 2 (step=3000ms, overlap=[0, 2000ms], pivot=1200ms):
  // "mud" (at 900ms < pivot) + "fossils" (at 1300ms >= pivot) splits
  // "mudfaucers" across the pivot.
  word_stage.PushWordsWithEndOfChunkTimestamp({{"in", 0},
                                               {"the", 400},
                                               {"mud", 900},
                                               {"fossils", 1300},
                                               {"doesn't", 1600},
                                               {"erode", 2200}},
                                              /*end_ts_ms=*/5000);
  ASSERT_OK(merger.Schedule());
  ASSERT_OK(GetMergeResult(merger));

  ASSERT_OK(merger.Flush());
  ASSERT_OK_AND_ASSIGN(auto flushed, GetMergeResult(merger));
  EXPECT_EQ(flushed.confirmed_text, "in the mud fossils doesn't erode");
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

TEST(TimestampTextMergerTest, HyphenatedCompoundSuffixAtSeamIsDeduplicated) {
  DummyWordStage word_stage;
  TimestampTextMerger merger(&word_stage, /*overlap_ratio=*/0.4f);

  word_stage.PushWordsWithEndOfChunkTimestamp(
      {{"the", 2500}, {"two-stage", 3500}, {"trigger", 4300}},
      /*end_ts_ms=*/5000);
  ASSERT_OK(merger.Schedule());
  ASSERT_OK(GetMergeResult(merger));

  // Chunk 2 starts at 3000 ms and hears "stage" (600 -> 3600 ms) + "trigger"
  // (1300 -> 4300 ms).
  word_stage.PushWordsWithEndOfChunkTimestamp(
      {{"stage", 600}, {"trigger", 1300}, {"system", 2000}},
      /*end_ts_ms=*/5000);
  ASSERT_OK(merger.Schedule());
  ASSERT_OK_AND_ASSIGN(auto res2, GetMergeResult(merger));
  EXPECT_EQ(res2.confirmed_text, "the");
  EXPECT_EQ(res2.unconfirmed_text, "two-stage trigger system");
}

TEST(TimestampTextMergerTest,
     TruncatedFinalBoundaryWordInPrevIsDroppedWhenCurrContinues) {
  DummyWordStage word_stage;
  TimestampTextMerger merger(&word_stage, /*overlap_ratio=*/0.5f);

  // Chunk 1 ends with a truncated boundary artifact "on." at frame 60 of 63.
  word_stage.PushWordsWithEndOfChunkTimestamp(
      {{"Look", 20}, {"at", 25}, {"that.", 30}, {"Holy", 52}, {"on.", 60}},
      /*end_ts_ms=*/63);
  ASSERT_OK(merger.Schedule());
  ASSERT_OK(GetMergeResult(merger));

  // Chunk 2 (step=31.5 frames) decodes across both sides of frame 60: "Holy"
  // (20 -> 52) and "cow." (35 -> 67).
  word_stage.PushWordsWithEndOfChunkTimestamp(
      {{"Holy", 20}, {"cow.", 35}, {"Amazing.", 45}},
      /*end_ts_ms=*/63);
  ASSERT_OK(merger.Schedule());
  ASSERT_OK_AND_ASSIGN(auto res2, GetMergeResult(merger));
  EXPECT_EQ(res2.confirmed_text, "Look at that.");
  EXPECT_EQ(res2.unconfirmed_text, "Holy cow. Amazing.");
}

TEST(TimestampTextMergerTest,
     SpelledInitialismBeforePivotIsNotReplacedByCurrFragment) {
  DummyWordStage word_stage;
  TimestampTextMerger merger(&word_stage, /*overlap_ratio=*/0.5f);

  // Chunk 1 decodes spelled-out letters "d e p t" in the overlap region, where
  // "d" at 2800 ms is before the pivot (3750 ms) and is NOT the final word of
  // Chunk 1.
  word_stage.PushWordsWithEndOfChunkTimestamp(
      {{"spelled", 2000}, {"d", 2800}, {"e", 3200}, {"p", 3600}, {"t", 4200}},
      /*end_ts_ms=*/5000);
  ASSERT_OK(merger.Schedule());
  ASSERT_OK(GetMergeResult(merger));

  // Chunk 2 (starting at 2500 ms) hallucinates "dep" at the window start
  // (300 -> 2800 ms) followed by "e" (700 -> 3200 ms), "p" (1100 -> 3600 ms),
  // "t" (1700 -> 4200 ms). Because "d" in Chunk 1 is not at the end of Chunk 1,
  // it must be preserved rather than overwritten by "dep".
  word_stage.PushWordsWithEndOfChunkTimestamp(
      {{"dep", 300}, {"e", 700}, {"p", 1100}, {"t", 1700}},
      /*end_ts_ms=*/5000);
  ASSERT_OK(merger.Schedule());
  ASSERT_OK_AND_ASSIGN(auto res2, GetMergeResult(merger));
  EXPECT_EQ(res2.confirmed_text, "spelled");
  EXPECT_EQ(res2.unconfirmed_text, "d e p t");
}

TEST(TimestampTextMergerTest,
     WindowEndSingleLetterFragmentInPrevIsReplacedByCurrFullWord) {
  DummyWordStage word_stage;
  TimestampTextMerger merger(&word_stage, /*overlap_ratio=*/0.5f);

  // Chunk 1 ends with a truncated 1-letter word "b" at 3600 ms (just before
  // pivot=3750 ms) because the window ended mid-word.
  word_stage.PushWordsWithEndOfChunkTimestamp(
      {{"hello", 1500}, {"coyote", 3000}, {"b", 3600}},
      /*end_ts_ms=*/5000);
  ASSERT_OK(merger.Schedule());
  ASSERT_OK(GetMergeResult(merger));

  // Chunk 2 (starting at 2500 ms) has full context for the final word:
  // "coyote" (500 -> 3000 ms), "bill" (1100 -> 3600 ms), "today" (1800 ->
  // 4300 ms).
  word_stage.PushWordsWithEndOfChunkTimestamp(
      {{"coyote", 500}, {"bill", 1100}, {"today", 1800}},
      /*end_ts_ms=*/5000);
  ASSERT_OK(merger.Schedule());
  ASSERT_OK_AND_ASSIGN(auto res2, GetMergeResult(merger));
  EXPECT_EQ(res2.confirmed_text, "hello");
  EXPECT_EQ(res2.unconfirmed_text, "coyote bill today");
}

}  // namespace
}  // namespace litert::omni::asr
