// Copyright 2025 The ODML Authors.
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

#include "runtime/engine/io_types.h"

#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "litert/test/matchers.h"  // from @litert
#include "runtime/components/constrained_decoding/fake_constraint.h"
#include "runtime/proto/engine.pb.h"
#include "runtime/util/test_utils.h"  // NOLINT

namespace litert::lm {
namespace {

using ::testing::ContainsRegex;
using ::testing::ElementsAre;
using ::testing::status::StatusIs;

TEST(ResponsesTest, GetTaskState) {
  {
    Responses responses(TaskState::kCreated, {});
    EXPECT_EQ(responses.GetTaskState(), TaskState::kCreated);
  }
  {
    Responses responses(TaskState::kQueued, {});
    EXPECT_EQ(responses.GetTaskState(), TaskState::kQueued);
  }
  {
    Responses responses(TaskState::kProcessing, {});
    EXPECT_EQ(responses.GetTaskState(), TaskState::kProcessing);
  }
  {
    Responses responses(TaskState::kDone, {});
    EXPECT_EQ(responses.GetTaskState(), TaskState::kDone);
  }
  {
    Responses responses(TaskState::kMaxNumTokensReached, {});
    EXPECT_EQ(responses.GetTaskState(), TaskState::kMaxNumTokensReached);
  }
  {
    Responses responses(TaskState::kFailed, {});
    EXPECT_EQ(responses.GetTaskState(), TaskState::kFailed);
  }
  {
    Responses responses(TaskState::kDependentTaskFailed, {});
    EXPECT_EQ(responses.GetTaskState(), TaskState::kDependentTaskFailed);
  }
  {
    Responses responses(TaskState::kCancelled, {});
    EXPECT_EQ(responses.GetTaskState(), TaskState::kCancelled);
  }
  {
    Responses responses(TaskState::kDependentTaskCancelled, {});
    EXPECT_EQ(responses.GetTaskState(), TaskState::kDependentTaskCancelled);
  }
  {
    Responses responses(TaskState::kLastCallbackQueued, {});
    EXPECT_EQ(responses.GetTaskState(), TaskState::kLastCallbackQueued);
  }
  {
    Responses responses(TaskState::kUnknown, {});
    EXPECT_EQ(responses.GetTaskState(), TaskState::kUnknown);
  }
}

TEST(TaskStateTest, TaskStateToString) {
  {
    std::stringstream ss;
    ss << TaskState::kCreated;
    EXPECT_EQ(ss.str(), "Created");
  }
  {
    std::stringstream ss;
    ss << TaskState::kQueued;
    EXPECT_EQ(ss.str(), "Queued");
  }
  {
    std::stringstream ss;
    ss << TaskState::kProcessing;
    EXPECT_EQ(ss.str(), "Processing");
  }
  {
    std::stringstream ss;
    ss << TaskState::kDone;
    EXPECT_EQ(ss.str(), "Done");
  }
  {
    std::stringstream ss;
    ss << TaskState::kMaxNumTokensReached;
    EXPECT_EQ(ss.str(), "MaxNumTokensReached");
  }
  {
    std::stringstream ss;
    ss << TaskState::kFailed;
    EXPECT_EQ(ss.str(), "Failed");
  }
  {
    std::stringstream ss;
    ss << TaskState::kDependentTaskFailed;
    EXPECT_EQ(ss.str(), "DependentTaskFailed");
  }
  {
    std::stringstream ss;
    ss << TaskState::kCancelled;
    EXPECT_EQ(ss.str(), "Cancelled");
  }
  {
    std::stringstream ss;
    ss << TaskState::kDependentTaskCancelled;
    EXPECT_EQ(ss.str(), "DependentTaskCancelled");
  }
  {
    std::stringstream ss;
    ss << TaskState::kLastCallbackQueued;
    EXPECT_EQ(ss.str(), "LastCallbackQueued");
  }
  {
    std::stringstream ss;
    ss << TaskState::kUnknown;
    EXPECT_EQ(ss.str(), "Unknown");
  }
}

TEST(ResponsesTest, SetTaskState) {
  Responses responses(TaskState::kCreated, {});
  EXPECT_EQ(responses.GetTaskState(), TaskState::kCreated);
  responses.SetTaskState(TaskState::kQueued);
  EXPECT_EQ(responses.GetTaskState(), TaskState::kQueued);
  responses.SetTaskState(TaskState::kProcessing);
  EXPECT_EQ(responses.GetTaskState(), TaskState::kProcessing);
  responses.SetTaskState(TaskState::kDone);
  EXPECT_EQ(responses.GetTaskState(), TaskState::kDone);
  responses.SetTaskState(TaskState::kMaxNumTokensReached);
  EXPECT_EQ(responses.GetTaskState(), TaskState::kMaxNumTokensReached);
  responses.SetTaskState(TaskState::kFailed);
  EXPECT_EQ(responses.GetTaskState(), TaskState::kFailed);
  responses.SetTaskState(TaskState::kDependentTaskFailed);
  EXPECT_EQ(responses.GetTaskState(), TaskState::kDependentTaskFailed);
  responses.SetTaskState(TaskState::kCancelled);
  EXPECT_EQ(responses.GetTaskState(), TaskState::kCancelled);
  responses.SetTaskState(TaskState::kDependentTaskCancelled);
  EXPECT_EQ(responses.GetTaskState(), TaskState::kDependentTaskCancelled);
  responses.SetTaskState(TaskState::kLastCallbackQueued);
  EXPECT_EQ(responses.GetTaskState(), TaskState::kLastCallbackQueued);
  responses.SetTaskState(TaskState::kUnknown);
  EXPECT_EQ(responses.GetTaskState(), TaskState::kUnknown);
}

TEST(TaskStateTest, IsTaskEndState) {
  EXPECT_TRUE(IsTaskEndState(TaskState::kDone));
  EXPECT_TRUE(IsTaskEndState(TaskState::kMaxNumTokensReached));
  EXPECT_TRUE(IsTaskEndState(TaskState::kFailed));
  EXPECT_TRUE(IsTaskEndState(TaskState::kDependentTaskFailed));
  EXPECT_TRUE(IsTaskEndState(TaskState::kCancelled));
  EXPECT_TRUE(IsTaskEndState(TaskState::kDependentTaskCancelled));
  EXPECT_FALSE(IsTaskEndState(TaskState::kCreated));
  EXPECT_FALSE(IsTaskEndState(TaskState::kQueued));
  EXPECT_FALSE(IsTaskEndState(TaskState::kProcessing));
}

TEST(ResponsesTest, GetTexts) {
  Responses responses(TaskState::kProcessing,
                      {"Hello World!", "How's it going?"});

  EXPECT_THAT(responses.GetTexts(),
              ElementsAre("Hello World!", "How's it going?"));
}

TEST(ResponsesTest, GetScores) {
  Responses responses(TaskState::kProcessing, /*response_texts=*/{},
                      /*scores=*/{0.1f, 0.2f});

  EXPECT_THAT(responses.GetScores(), ElementsAre(0.1, 0.2));
}

TEST(ResponsesTest, GetMutableTexts) {
  Responses responses =
      Responses(TaskState::kProcessing, {"Hello World!", "How's it going?"});

  EXPECT_EQ(responses.GetMutableTexts().size(), 2);
  EXPECT_THAT(responses.GetMutableTexts()[0], "Hello World!");
  EXPECT_THAT(responses.GetMutableTexts()[1], "How's it going?");
}

TEST(ResponsesTest, GetMutableScores) {
  Responses responses = Responses(TaskState::kProcessing, /*response_texts=*/{},
                                  /*scores=*/{0.1f, 0.2f});

  EXPECT_EQ(responses.GetMutableScores().size(), 2);
  EXPECT_FLOAT_EQ(responses.GetMutableScores()[0], 0.1f);
  EXPECT_FLOAT_EQ(responses.GetMutableScores()[1], 0.2f);
}

TEST(ResponsesTest, GetTokenScores) {
  Responses responses(TaskState::kProcessing);
  EXPECT_FALSE(responses.GetTokenScores().has_value());
}

TEST(ResponsesTest, GetMutableTokenScores) {
  Responses responses = Responses(TaskState::kProcessing);
  responses.GetMutableTokenScores() =
      std::vector<std::vector<float>>{{0.1f, 0.2f}, {0.3f, 0.4f}};
  ASSERT_TRUE(responses.GetTokenScores().has_value());
  EXPECT_EQ(responses.GetTokenScores()->size(), 2);
  EXPECT_THAT(responses.GetTokenScores()->at(0), ElementsAre(0.1f, 0.2f));
  EXPECT_THAT(responses.GetTokenScores()->at(1), ElementsAre(0.3f, 0.4f));
}

TEST(ResponsesTest, GetTokenIds) {
  Responses responses(TaskState::kProcessing, /*response_texts=*/{},
                      /*scores=*/{}, /*token_lengths=*/{},
                      /*token_ids=*/{{1, 2, 3}, {4, 5}});

  EXPECT_THAT(responses.GetTokenIds(),
              ElementsAre(ElementsAre(1, 2, 3), ElementsAre(4, 5)));
}

TEST(ResponsesTest, GetMutableTokenIds) {
  Responses responses = Responses(TaskState::kProcessing);

  EXPECT_TRUE(responses.GetMutableTokenIds().empty());

  responses.GetMutableTokenIds() =
      std::vector<std::vector<int>>{{1, 2, 3}, {4, 5}};
  EXPECT_THAT(responses.GetTokenIds(),
              ElementsAre(ElementsAre(1, 2, 3), ElementsAre(4, 5)));
}

proto::BenchmarkParams GetBenchmarkParams() {
  proto::BenchmarkParams benchmark_params;
  benchmark_params.set_num_decode_tokens(100);
  benchmark_params.set_num_prefill_tokens(100);
  return benchmark_params;
}

// --- Test Init Phases ---
TEST(BenchmarkInfoTests, AddAndGetInitPhases) {
  BenchmarkInfo benchmark_info(GetBenchmarkParams());
  EXPECT_OK(benchmark_info.TimeInitPhaseStart(
      BenchmarkInfo::InitPhase::kModelAssets));
  EXPECT_OK(
      benchmark_info.TimeInitPhaseStart(BenchmarkInfo::InitPhase::kTokenizer));
  absl::SleepFor(absl::Milliseconds(50));
  EXPECT_OK(
      benchmark_info.TimeInitPhaseEnd(BenchmarkInfo::InitPhase::kTokenizer));
  absl::SleepFor(absl::Milliseconds(50));
  EXPECT_OK(
      benchmark_info.TimeInitPhaseEnd(BenchmarkInfo::InitPhase::kModelAssets));

  const auto& phases = benchmark_info.GetInitPhases();
  ASSERT_EQ(phases.size(), 2);
  // The time should be greater than 50ms.
  EXPECT_GT(phases.at(std::string(BenchmarkInfo::InitPhaseToString(
                BenchmarkInfo::InitPhase::kTokenizer))),
            absl::Milliseconds(50));
  // The time should be greater than 50 + 50 = 100ms.
  EXPECT_GT(phases.at(std::string(BenchmarkInfo::InitPhaseToString(
                BenchmarkInfo::InitPhase::kModelAssets))),
            absl::Milliseconds(100));
}

TEST(BenchmarkInfoTests, AddInitPhaseTwice) {
  BenchmarkInfo benchmark_info(GetBenchmarkParams());
  EXPECT_OK(benchmark_info.TimeInitPhaseStart(
      BenchmarkInfo::InitPhase::kModelAssets));
  // Starting the same phase twice should fail.
  EXPECT_THAT(
      benchmark_info.TimeInitPhaseStart(BenchmarkInfo::InitPhase::kModelAssets),
      StatusIs(absl::StatusCode::kInternal));

  // Ending a phase that has not started should fail.
  EXPECT_THAT(
      benchmark_info.TimeInitPhaseEnd(BenchmarkInfo::InitPhase::kTokenizer),
      StatusIs(absl::StatusCode::kInternal));
}

TEST(BenchmarkInfoTests, CandDoInitPhaseRecord) {
  BenchmarkInfo benchmark_info(GetBenchmarkParams());
  EXPECT_OK(benchmark_info.InitPhaseRecord(
      BenchmarkInfo::InitPhase::kModelAssets, absl::Milliseconds(50)));
  EXPECT_OK(benchmark_info.InitPhaseRecord(BenchmarkInfo::InitPhase::kTokenizer,
                                           absl::Milliseconds(100)));

  EXPECT_EQ(benchmark_info.GetInitPhases().at(
                std::string(BenchmarkInfo::InitPhaseToString(
                    BenchmarkInfo::InitPhase::kModelAssets))),
            absl::Milliseconds(50));
  EXPECT_EQ(benchmark_info.GetInitPhases().at(
                std::string(BenchmarkInfo::InitPhaseToString(
                    BenchmarkInfo::InitPhase::kTokenizer))),
            absl::Milliseconds(100));
}

TEST(BenchmarkInfoTests, AddInitPhaseError) {
  BenchmarkInfo benchmark_info(GetBenchmarkParams());
  // Recording the same phase twice should fail.
  EXPECT_OK(benchmark_info.InitPhaseRecord(
      BenchmarkInfo::InitPhase::kModelAssets, absl::Milliseconds(50)));
  EXPECT_THAT(
      benchmark_info.InitPhaseRecord(BenchmarkInfo::InitPhase::kModelAssets,
                                     absl::Milliseconds(50)),
      StatusIs(absl::StatusCode::kInternal));
}

TEST(BenchmarkInfoTests, AddPrefillTurn) {
  BenchmarkInfo benchmark_info(GetBenchmarkParams());
  EXPECT_OK(benchmark_info.TimePrefillTurnStart());
  EXPECT_OK(benchmark_info.TimePrefillTurnEnd(100));
  EXPECT_OK(benchmark_info.TimePrefillTurnStart());
  EXPECT_OK(benchmark_info.TimePrefillTurnEnd(200));
  EXPECT_EQ(benchmark_info.GetTotalPrefillTurns(), 2);
}

TEST(BenchmarkInfoTests, AddPrefillTurnError) {
  BenchmarkInfo benchmark_info(GetBenchmarkParams());
  EXPECT_OK(benchmark_info.TimePrefillTurnStart());
  // Starting the prefill turn twice should fail.
  EXPECT_THAT(benchmark_info.TimePrefillTurnStart(),
              StatusIs(absl::StatusCode::kInternal));

  EXPECT_OK(benchmark_info.TimePrefillTurnEnd(100));
  // Ending a prefill turn that has not started should fail.
  EXPECT_THAT(benchmark_info.TimePrefillTurnEnd(200),
              StatusIs(absl::StatusCode::kInternal));
}

TEST(BenchmarkInfoTests, AddDecodeTurn) {
  BenchmarkInfo benchmark_info(GetBenchmarkParams());
  EXPECT_OK(benchmark_info.TimeDecodeTurnStart());
  EXPECT_OK(benchmark_info.TimeDecodeTurnEnd(100));
  EXPECT_OK(benchmark_info.TimeDecodeTurnStart());
  EXPECT_OK(benchmark_info.TimeDecodeTurnEnd(200));
  EXPECT_EQ(benchmark_info.GetTotalDecodeTurns(), 2);
}

TEST(BenchmarkInfoTests, AddDecodeTurnError) {
  BenchmarkInfo benchmark_info(GetBenchmarkParams());
  EXPECT_OK(benchmark_info.TimeDecodeTurnStart());
  // Starting the decode turn twice should fail.
  EXPECT_THAT(benchmark_info.TimeDecodeTurnStart(),
              StatusIs(absl::StatusCode::kInternal));

  EXPECT_OK(benchmark_info.TimeDecodeTurnEnd(100));
  // Ending a decode turn that has not started should fail.
  EXPECT_THAT(benchmark_info.TimeDecodeTurnEnd(200),
              StatusIs(absl::StatusCode::kInternal));
}

TEST(BenchmarkInfoTests, AddTextToTokenIdsTurn) {
  BenchmarkInfo benchmark_info(GetBenchmarkParams());
  EXPECT_OK(benchmark_info.TimeTextToTokenIdsStart());
  EXPECT_OK(benchmark_info.TimeTextToTokenIdsEnd(10));
  EXPECT_OK(benchmark_info.TimeTextToTokenIdsStart());
  EXPECT_OK(benchmark_info.TimeTextToTokenIdsEnd(20));
  EXPECT_EQ(benchmark_info.GetTotalTextToTokenIdsTurns(), 2);
}

TEST(BenchmarkInfoTests, AddTextToTokenIdsTurnError) {
  BenchmarkInfo benchmark_info(GetBenchmarkParams());
  EXPECT_OK(benchmark_info.TimeTextToTokenIdsStart());
  // Starting the turn twice should fail.
  EXPECT_THAT(benchmark_info.TimeTextToTokenIdsStart(),
              StatusIs(absl::StatusCode::kInternal));

  EXPECT_OK(benchmark_info.TimeTextToTokenIdsEnd(10));
  // Ending a turn that has not started should fail.
  EXPECT_THAT(benchmark_info.TimeTextToTokenIdsEnd(20),
              StatusIs(absl::StatusCode::kInternal));
}

TEST(BenchmarkInfoTests, AddMarks) {
  BenchmarkInfo benchmark_info(GetBenchmarkParams());
  EXPECT_OK(benchmark_info.TimeMarkDelta("sampling"));
  absl::SleepFor(absl::Milliseconds(200));
  EXPECT_OK(benchmark_info.TimeMarkDelta("sampling"));
  absl::SleepFor(absl::Milliseconds(200));
  EXPECT_OK(benchmark_info.TimeMarkDelta("sampling"));
  EXPECT_EQ(benchmark_info.GetMarkDurations().size(), 1);

  // The time should record the duration between the 2nd and 3rd calls, which
  // should be slightly more than 200ms.
  EXPECT_GT(benchmark_info.GetMarkDurations().at("sampling"),
            absl::Milliseconds(200));
  // Verify that the time doesn't record the duration between the 1st and 3nd
  // calls, which is less than 200ms + 200ms = 400ms.
  EXPECT_LT(benchmark_info.GetMarkDurations().at("sampling"),
            absl::Milliseconds(400));
}

TEST(BenchmarkInfoTests, AddTwoMarks) {
  BenchmarkInfo benchmark_info(GetBenchmarkParams());
  EXPECT_OK(benchmark_info.TimeMarkDelta("tokenize"));
  EXPECT_OK(benchmark_info.TimeMarkDelta("sampling"));
  absl::SleepFor(absl::Milliseconds(50));
  EXPECT_OK(benchmark_info.TimeMarkDelta("sampling"));
  absl::SleepFor(absl::Milliseconds(50));
  EXPECT_OK(benchmark_info.TimeMarkDelta("tokenize"));
  EXPECT_EQ(benchmark_info.GetMarkDurations().size(), 2);

  // Time between two sampling calls should be more than 50ms.
  EXPECT_GT(benchmark_info.GetMarkDurations().at("sampling"),
            absl::Milliseconds(50));
  // Time between two tokenize calls should be more than 50ms + 50ms = 100ms.
  EXPECT_GT(benchmark_info.GetMarkDurations().at("tokenize"),
            absl::Milliseconds(100));
}

TEST(BenchmarkInfoTests, GetTimeToFirstTokenInvalid) {
  BenchmarkInfo benchmark_info(GetBenchmarkParams());
  EXPECT_OK(benchmark_info.TimePrefillTurnStart());
  EXPECT_OK(benchmark_info.TimePrefillTurnEnd(100));
  EXPECT_EQ(benchmark_info.GetTimeToFirstToken(), 0.0);
}

TEST(BenchmarkInfoTests, GetTimeToFirstTokenValid) {
  BenchmarkInfo benchmark_info(GetBenchmarkParams());
  // Simulating prefilling 100 tokens takes > 100ms.
  EXPECT_OK(benchmark_info.TimePrefillTurnStart());
  absl::SleepFor(absl::Milliseconds(100));
  EXPECT_OK(benchmark_info.TimePrefillTurnEnd(100));
  // Simulating decoding 50 tokens takes > 200ms.
  EXPECT_OK(benchmark_info.TimeDecodeTurnStart());
  absl::SleepFor(absl::Milliseconds(200));
  EXPECT_OK(benchmark_info.TimeDecodeTurnEnd(50));

  // The time to first token should be (larger than) 100ms + 200ms / 50 = 104ms.
  EXPECT_GT(benchmark_info.GetTimeToFirstToken(), 0.104);
}

TEST(BenchmarkInfoTests, OperatorOutputWithData) {
  BenchmarkInfo benchmark_info(GetBenchmarkParams());
  EXPECT_OK(
      benchmark_info.TimeInitPhaseStart(BenchmarkInfo::InitPhase::kTotal));
  EXPECT_OK(benchmark_info.TimeInitPhaseStart(
      BenchmarkInfo::InitPhase::kModelAssets));
  EXPECT_OK(
      benchmark_info.TimeInitPhaseStart(BenchmarkInfo::InitPhase::kTokenizer));
  EXPECT_OK(
      benchmark_info.TimeInitPhaseEnd(BenchmarkInfo::InitPhase::kModelAssets));
  EXPECT_OK(
      benchmark_info.TimeInitPhaseEnd(BenchmarkInfo::InitPhase::kTokenizer));
  EXPECT_OK(benchmark_info.TimeInitPhaseEnd(BenchmarkInfo::InitPhase::kTotal));

  EXPECT_OK(benchmark_info.TimePrefillTurnStart());
  EXPECT_OK(benchmark_info.TimePrefillTurnEnd(100));
  EXPECT_OK(benchmark_info.TimePrefillTurnStart());
  EXPECT_OK(benchmark_info.TimePrefillTurnEnd(200));

  EXPECT_OK(benchmark_info.TimeDecodeTurnStart());
  EXPECT_OK(benchmark_info.TimeDecodeTurnEnd(100));

  EXPECT_OK(benchmark_info.TimeTextToTokenIdsStart());
  EXPECT_OK(benchmark_info.TimeTextToTokenIdsEnd(50));

  std::stringstream ss;
  ss << benchmark_info;
  const std::string expected_output = R"(BenchmarkInfo:
  Init Phases \(3\):
    - Init Model assets: .* ms
    - Init Tokenizer: .* ms
    - Init Total: .* ms
--------------------------------------------------
  Time to first token: .* s
--------------------------------------------------
  Prefill Turns \(Total 2 turns\):
    Prefill Turn 1: Processed 100 tokens in .* duration.
      Prefill Speed: .* tokens/sec.
    Prefill Turn 2: Processed 200 tokens in .* duration.
      Prefill Speed: .* tokens/sec.
--------------------------------------------------
  Decode Turns \(Total 1 turns\):
    Decode Turn 1: Processed 100 tokens in .* duration.
      Decode Speed: .* tokens/sec.
--------------------------------------------------
  TextToTokenIds Turns \(Total 1 turns\):
    Turn 1: .*, 50 tokens
--------------------------------------------------
--------------------------------------------------
)";
  EXPECT_THAT(ss.str(), ContainsRegex(expected_output));
}

TEST(DecodeConfigTest, CreateDefault) {
  DecodeConfig decode_config = DecodeConfig::CreateDefault();
  EXPECT_EQ(decode_config.GetConstraint(), nullptr);
  EXPECT_EQ(decode_config.GetMaxOutputTokens(), std::nullopt);
}

TEST(DecodeConfigTest, SetAndGetConstraint) {
  DecodeConfig decode_config = DecodeConfig::CreateDefault();
  auto constraint = FakeConstraint({1, 2, 3}, /*vocabulary_size=*/10);
  decode_config.SetConstraint(&constraint);
  EXPECT_EQ(decode_config.GetConstraint(), &constraint);
}

TEST(DecodeConfigTest, SetAndGetMaxOutputTokens) {
  DecodeConfig decode_config = DecodeConfig::CreateDefault();
  EXPECT_EQ(decode_config.GetMaxOutputTokens(), std::nullopt);

  decode_config.SetMaxOutputTokens(42);
  EXPECT_EQ(decode_config.GetMaxOutputTokens(), 42);
}

TEST(VisionExecutorPropertiesTest, OperatorOutput) {
  VisionExecutorProperties properties;
  properties.num_tokens_per_image = 128;
  properties.patch_num_shrink_factor = 4;

  std::stringstream ss;
  ss << properties;
  EXPECT_THAT(ss.str(), ContainsRegex("num_tokens_per_image: 128"));
  EXPECT_THAT(ss.str(), ContainsRegex("patch_num_shrink_factor: 4"));
}

TEST(VisionExecutorPropertiesTest, OperatorOutputDefault) {
  VisionExecutorProperties properties;

  std::stringstream ss;
  ss << properties;
  EXPECT_THAT(ss.str(), ContainsRegex("num_tokens_per_image: 256"));
  EXPECT_THAT(ss.str(), ContainsRegex("patch_num_shrink_factor: not set"));
}

TEST(AudioExecutorPropertiesTest, OperatorOutput) {
  AudioExecutorProperties properties;
  properties.is_streaming_model = true;
  properties.streaming_chunk_size = 1024;
  properties.streaming_chunk_overlap_size = 256;
  properties.audio_shrink_factor = 8;

  std::stringstream ss;
  ss << properties;
  EXPECT_THAT(ss.str(), ContainsRegex("is_streaming_model: 1"));
  EXPECT_THAT(ss.str(), ContainsRegex("streaming_chunk_size: 1024"));
  EXPECT_THAT(ss.str(), ContainsRegex("streaming_chunk_overlap_size: 256"));
  EXPECT_THAT(ss.str(), ContainsRegex("audio_shrink_factor: 8"));
}

}  // namespace
}  // namespace litert::lm
