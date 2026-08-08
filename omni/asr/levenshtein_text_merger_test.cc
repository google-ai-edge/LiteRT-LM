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

#include "omni/asr/levenshtein_text_merger.h"

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/cc/litert_macros.h"  // from @litert
#include "omni/asr/detokenizer.h"
#include "omni/asr/text_merger.h"
#include "omni/base/stage.h"

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

 protected:
  bool NeedScheduleInternal() const override { return false; }
  absl::Status ScheduleInternal() override { return absl::OkStatus(); }
};

absl::StatusOr<TextMerger::MergeResult> ProcessWords(
    DummyWordStage& word_stage, LevenshteinTextMerger& merger,
    const std::vector<std::string>& strings) {
  word_stage.PushWords(strings);
  LITERT_RETURN_IF_ERROR(merger.Schedule());
  return merger.GetOutput();
}

TEST(LevenshteinTextMergerTest, InitialChunkReturnsUnconfirmedOnly) {
  DummyWordStage word_stage;
  LevenshteinTextMerger merger(&word_stage);

  auto result = ProcessWords(word_stage, merger, {"hello", "world"});
  ASSERT_TRUE(result.ok());

  EXPECT_EQ(result->confirmed_text, "");
  EXPECT_EQ(result->unconfirmed_text, "hello world");
}

TEST(LevenshteinTextMergerTest, SequentialMergeFlow) {
  DummyWordStage word_stage;
  LevenshteinTextMerger merger(&word_stage);

  // Chunk 1: "hello world this is"
  auto res1 =
      ProcessWords(word_stage, merger, {"hello", "world", "this", "is"});
  ASSERT_TRUE(res1.ok());
  EXPECT_EQ(res1->confirmed_text, "");
  EXPECT_EQ(res1->unconfirmed_text, "hello world this is");

  // Chunk 2: overlaps at "this is", adds "a test"
  auto res2 = ProcessWords(word_stage, merger, {"this", "is", "a", "test"});
  ASSERT_TRUE(res2.ok());
  EXPECT_EQ(res2->confirmed_text, "hello world");
  EXPECT_EQ(res2->unconfirmed_text, "this is a test");

  // Chunk 3: overlaps at "a test", adds "of streaming"
  auto res3 =
      ProcessWords(word_stage, merger, {"a", "test", "of", "streaming"});
  ASSERT_TRUE(res3.ok());
  EXPECT_EQ(res3->confirmed_text, "this is");
  EXPECT_EQ(res3->unconfirmed_text, "a test of streaming");

  // End of stream flush
  ASSERT_TRUE(merger.Flush().ok());
  auto res_flush = merger.GetOutput();
  ASSERT_TRUE(res_flush.ok());
  EXPECT_EQ(res_flush->confirmed_text, "a test of streaming");
  EXPECT_EQ(res_flush->unconfirmed_text, "");
}

TEST(LevenshteinTextMergerTest, ResetClearsState) {
  DummyWordStage word_stage;
  LevenshteinTextMerger merger(&word_stage);

  auto dummy_res = ProcessWords(word_stage, merger, {"hello", "world"});
  EXPECT_TRUE(dummy_res.ok());
  EXPECT_FALSE(merger.unconfirmed_words_for_testing().empty());

  merger.Reset();
  EXPECT_TRUE(merger.unconfirmed_words_for_testing().empty());

  // First merge after reset behaves as initial chunk
  auto result = ProcessWords(word_stage, merger, {"new", "stream"});
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result->confirmed_text, "");
  EXPECT_EQ(result->unconfirmed_text, "new stream");
}

TEST(LevenshteinTextMergerTest, NoOverlapConfirmsPreviousState) {
  DummyWordStage word_stage;
  LevenshteinTextMerger merger(&word_stage);

  auto res1 = ProcessWords(word_stage, merger, {"apple", "banana"});
  EXPECT_TRUE(res1.ok());

  // Chunk 2 has no overlap
  auto result = ProcessWords(word_stage, merger, {"cat", "dog"});
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result->confirmed_text, "apple banana");
  EXPECT_EQ(result->unconfirmed_text, "cat dog");
}

}  // namespace
}  // namespace litert::omni::asr
