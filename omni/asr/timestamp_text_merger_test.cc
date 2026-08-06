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

#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/cc/litert_macros.h"  // from @litert
#include "omni/asr/detokenizer.h"
#include "omni/asr/text_merger.h"
#include "omni/base/stage.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep for ASSERT_OK

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

absl::StatusOr<TextMerger::MergeResult> ProcessWords(
    DummyWordStage& word_stage, TimestampTextMerger& merger,
    const std::vector<std::string>& strings) {
  word_stage.PushWords(strings);
  LITERT_RETURN_IF_ERROR(merger.Schedule());
  return merger.GetOutput();
}

absl::StatusOr<TextMerger::MergeResult> ProcessWordsWithTimestamps(
    DummyWordStage& word_stage, TimestampTextMerger& merger,
    const std::vector<std::pair<std::string, int>>& word_ts_pairs) {
  word_stage.PushWordsWithTimestamps(word_ts_pairs);
  LITERT_RETURN_IF_ERROR(merger.Schedule());
  return merger.GetOutput();
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
  auto res_flush = merger.GetOutput();
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
  auto res1 = merger.GetOutput();
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
  auto res2 = merger.GetOutput();
  ASSERT_OK(res2);
  EXPECT_FALSE(res2->confirmed_text.empty());
  EXPECT_FALSE(res2->unconfirmed_text.empty());
}

}  // namespace
}  // namespace litert::omni::asr
