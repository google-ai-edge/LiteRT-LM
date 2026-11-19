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

#include "runtime/conversation/end_of_turn.h"

#include <cstdint>
#include <fstream>
#include <limits>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/conversation/io_types.h"
#include "runtime/executor/executor_stats.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

using ::testing::ElementsAre;
using ::testing::Pair;
using ::testing::status::IsOkAndHolds;
using ::testing::status::StatusIs;

ExecutorStats MtpStats() {
  ExecutorStats stats{.module_name = "LLM"};
  stats.AccumulatePlainStep(absl::Microseconds(51234));
  stats.AccumulateMtpRound(/*drafted_tokens=*/3, /*accepted_tokens=*/2,
                           /*emitted_tokens=*/3, absl::Microseconds(100),
                           absl::Microseconds(200), absl::Microseconds(300));
  stats.Accumulate("acceptance_rate", 0.5);
  stats.substats.push_back(ExecutorStats{.module_name = "Drafter"});
  stats.substats.back().Accumulate("drafter_steps", int64_t{3});
  return stats;
}

TEST(EndOfTurnTest, FinishReasonToString) {
  EXPECT_EQ(FinishReasonToString(FinishReason::kStop), "stop");
  EXPECT_EQ(FinishReasonToString(FinishReason::kMaxTokens), "max_tokens");
  EXPECT_EQ(FinishReasonToString(FinishReason::kAppended), "appended");
  EXPECT_EQ(FinishReasonToString(FinishReason::kCancelled), "cancelled");
}

TEST(EndOfTurnTest, CreateEndOfTurnMessageWithStats) {
  Message message =
      CreateEndOfTurnMessage(FinishReason::kStop, ExecutorStats{"LLM"});

  EXPECT_EQ(message, nlohmann::ordered_json::parse(R"json({
    "role": "assistant",
    "finish_reason": "stop",
    "executor_stats": {
      "module_name": "LLM", "metrics": {}, "latencies_us": {}, "substats": []
    }
  })json"));
}

TEST(EndOfTurnTest, CreateEndOfTurnMessageWithoutStats) {
  Message message =
      CreateEndOfTurnMessage(FinishReason::kAppended, std::nullopt);

  EXPECT_EQ(message, nlohmann::ordered_json::parse(R"json({
    "role": "assistant", "finish_reason": "appended", "executor_stats": null
  })json"));
}

TEST(EndOfTurnTest, IsEndOfTurnMessage) {
  EXPECT_TRUE(IsEndOfTurnMessage(
      CreateEndOfTurnMessage(FinishReason::kMaxTokens, MtpStats())));
  EXPECT_TRUE(IsEndOfTurnMessage(
      CreateEndOfTurnMessage(FinishReason::kAppended, std::nullopt)));
  // Reasons added later still end the turn.
  EXPECT_TRUE(IsEndOfTurnMessage(Message::parse(
      R"json({"role": "assistant", "finish_reason": "tool_calls"})json")));
  // The old end-of-turn marker.
  EXPECT_TRUE(IsEndOfTurnMessage(Message()));
}

TEST(EndOfTurnTest, IsEndOfTurnMessageIsFalseForChunks) {
  EXPECT_FALSE(IsEndOfTurnMessage(Message::parse(R"json({
    "role": "assistant", "content": [{"type": "text", "text": "Hi"}]
  })json")));
  EXPECT_FALSE(IsEndOfTurnMessage(Message::parse(
      R"json({"role": "assistant", "channels": {"thought": "Hmm"}})json")));
  // A null `finish_reason` counts as missing, like a null `executor_stats`.
  EXPECT_FALSE(IsEndOfTurnMessage(Message::parse(
      R"json({"role": "assistant", "finish_reason": null})json")));
  EXPECT_FALSE(IsEndOfTurnMessage(Message::object()));
  EXPECT_FALSE(IsEndOfTurnMessage(Message::array()));
}

TEST(EndOfTurnTest, ExecutorStatsToJson) {
  EXPECT_EQ(ExecutorStatsToJson(MtpStats()),
            nlohmann::ordered_json::parse(R"json({
    "module_name": "LLM",
    "metrics": {
      "plain_steps": 1,
      "mtp_rounds": 1,
      "mtp_num_draft_tokens": 3,
      "mtp_num_accepted_tokens": 2,
      "mtp_emitted_tokens": 3,
      "acceptance_rate": 0.5
    },
    "latencies_us": {
      "plain_step_time": 51234,
      "mtp_drafting_time": 100,
      "mtp_verify_time": 200,
      "mtp_round_time": 300
    },
    "substats": [{
      "module_name": "Drafter",
      "metrics": {"drafter_steps": 3},
      "latencies_us": {},
      "substats": []
    }]
  })json"));
}

TEST(EndOfTurnTest, ExecutorStatsRoundTrip) {
  ExecutorStats stats = MtpStats();

  // Go through the serialized string, as the bindings do.
  ASSERT_OK_AND_ASSIGN(ExecutorStats round_trip,
                       ExecutorStatsFromJson(nlohmann::ordered_json::parse(
                           ExecutorStatsToJson(stats).dump())));

  EXPECT_EQ(round_trip.module_name, stats.module_name);
  EXPECT_EQ(round_trip.metrics, stats.metrics);
  EXPECT_EQ(round_trip.latencies, stats.latencies);
  ASSERT_EQ(round_trip.substats.size(), 1);
  EXPECT_EQ(round_trip.substats[0].module_name, "Drafter");
  EXPECT_EQ(round_trip.substats[0].metrics, stats.substats[0].metrics);
  // Integer and double metrics keep their types.
  EXPECT_TRUE(
      std::holds_alternative<int64_t>(*round_trip.GetMetric("plain_steps")));
  EXPECT_TRUE(
      std::holds_alternative<double>(*round_trip.GetMetric("acceptance_rate")));
}

TEST(EndOfTurnTest, ExecutorStatsToJsonLeavesOutNonFiniteMetrics) {
  ExecutorStats stats{.module_name = "LLM"};
  stats.Accumulate("nan", std::numeric_limits<double>::quiet_NaN());
  stats.Accumulate("inf", std::numeric_limits<double>::infinity());
  stats.Accumulate("rate", 0.5);

  nlohmann::ordered_json json = ExecutorStatsToJson(stats);

  EXPECT_EQ(json["metrics"], nlohmann::ordered_json::parse(R"({"rate": 0.5})"));
  EXPECT_OK(
      ExecutorStatsFromJson(nlohmann::ordered_json::parse(json.dump())));
}

TEST(EndOfTurnTest, ExecutorStatsFromJsonLeavesMissingFieldsEmpty) {
  ASSERT_OK_AND_ASSIGN(ExecutorStats stats,
                       ExecutorStatsFromJson(
                           Message::parse(R"json({"metrics": {"a": 1}})json")));

  EXPECT_EQ(stats.module_name, "");
  EXPECT_THAT(stats.metrics, ElementsAre(Pair("a", MetricValue(int64_t{1}))));
  EXPECT_TRUE(stats.latencies.empty());
  EXPECT_TRUE(stats.substats.empty());
}

TEST(EndOfTurnTest, ExecutorStatsFromJsonRejectsWrongTypes) {
  for (const char* json : {
           R"json([])json",
           R"json({"module_name": 1})json",
           R"json({"metrics": []})json",
           R"json({"metrics": {"a": "1"}})json",
           R"json({"latencies_us": {"a": 1.5}})json",
           R"json({"substats": {}})json",
           R"json({"substats": [{"metrics": {"a": true}}]})json",
       }) {
    EXPECT_THAT(ExecutorStatsFromJson(Message::parse(json)),
                StatusIs(absl::StatusCode::kInvalidArgument))
        << json;
  }
}

TEST(EndOfTurnTest, GetExecutorStats) {
  ASSERT_OK_AND_ASSIGN(std::optional<ExecutorStats> stats,
                       GetExecutorStats(CreateEndOfTurnMessage(
                           FinishReason::kStop, MtpStats())));

  ASSERT_TRUE(stats.has_value());
  EXPECT_EQ(stats->plain_steps(), 1);
  EXPECT_EQ(stats->mtp_round_time(), absl::Microseconds(300));
}

TEST(EndOfTurnTest, GetExecutorStatsReturnsNulloptWithoutStats) {
  EXPECT_THAT(GetExecutorStats(CreateEndOfTurnMessage(FinishReason::kAppended,
                                                      std::nullopt)),
              IsOkAndHolds(::testing::Eq(std::nullopt)));
  EXPECT_THAT(GetExecutorStats(Message()),
              IsOkAndHolds(::testing::Eq(std::nullopt)));
  EXPECT_THAT(GetExecutorStats(
                  Message::parse(R"json({"finish_reason": "stop"})json")),
              IsOkAndHolds(::testing::Eq(std::nullopt)));
}

TEST(EndOfTurnTest, GetExecutorStatsReturnsErrorForInvalidStats) {
  EXPECT_THAT(
      GetExecutorStats(Message::parse(
          R"json({"finish_reason": "stop", "executor_stats": []})json")),
      StatusIs(absl::StatusCode::kInvalidArgument));
}

// The tests of the language bindings parse `runtime/testdata/end_of_turn.json`,
// so it has to stay what the writer emits.
TEST(EndOfTurnTest, BindingTestdataMatchesTheWriter) {
  std::ifstream file(absl::StrCat(
      testing::SrcDir(), "/",
      "litert_lm/runtime/testdata/end_of_turn.json"));
  ASSERT_TRUE(file.is_open());
  const Message message = Message::parse(file);

  EXPECT_TRUE(IsEndOfTurnMessage(message));
  EXPECT_EQ(message.at("finish_reason").get<std::string>(),
            FinishReasonToString(FinishReason::kMaxTokens));
  ASSERT_OK_AND_ASSIGN(std::optional<ExecutorStats> stats,
                       GetExecutorStats(message));
  ASSERT_TRUE(stats.has_value());
  EXPECT_EQ(ExecutorStatsToJson(*stats), message.at("executor_stats"));
}

}  // namespace
}  // namespace litert::lm
