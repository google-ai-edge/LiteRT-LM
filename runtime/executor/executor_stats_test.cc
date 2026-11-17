// Copyright 2026 Google LLC.
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

#include "runtime/executor/executor_stats.h"

#include <cstdint>
#include <optional>
#include <sstream>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl

namespace litert::lm {
namespace {

using ::testing::HasSubstr;

TEST(ExecutorStatsTest, AccumulateAndGetLatency) {
  ExecutorStats stats;
  EXPECT_EQ(stats.GetTotalLatency(), absl::ZeroDuration());
  EXPECT_FALSE(stats.GetLatency("Stage1").has_value());

  stats.Accumulate("Stage1", absl::Milliseconds(10));
  stats.Accumulate("Stage2", absl::Milliseconds(20));

  EXPECT_EQ(stats.GetLatency("Stage1"), absl::Milliseconds(10));
  EXPECT_EQ(stats.GetLatency("Stage2"), absl::Milliseconds(20));
  EXPECT_FALSE(stats.GetLatency("NonExistent").has_value());

  // Accumulation on duplicate key
  stats.Accumulate("Stage1", absl::Milliseconds(15));
  EXPECT_EQ(stats.GetLatency("Stage1"), absl::Milliseconds(25));
  EXPECT_EQ(stats.latencies.size(), 2);
}

TEST(ExecutorStatsTest, AccumulateAndGetMetric) {
  ExecutorStats stats;
  EXPECT_FALSE(stats.GetMetric("num_tokens").has_value());

  // int64_t metric
  stats.Accumulate("num_tokens", int64_t{5});
  stats.Accumulate("num_tokens", int64_t{3});
  auto metric = stats.GetMetric("num_tokens");
  ASSERT_TRUE(metric.has_value());
  EXPECT_EQ(std::get<int64_t>(*metric), 8);

  // double metric
  stats.Accumulate("throughput", 12.5);
  stats.Accumulate("throughput", 7.5);
  auto double_metric = stats.GetMetric("throughput");
  ASSERT_TRUE(double_metric.has_value());
  EXPECT_DOUBLE_EQ(std::get<double>(*double_metric), 20.0);

  // Type mismatch skips update
  stats.Accumulate("num_tokens", 3.14);
  metric = stats.GetMetric("num_tokens");
  ASSERT_TRUE(metric.has_value());
  EXPECT_EQ(std::get<int64_t>(*metric), 8);
}

TEST(ExecutorStatsTest, AccumulateStatHelpersWithOptional) {
  std::optional<ExecutorStats> inactive_stats = std::nullopt;
  AccumulateStat(inactive_stats, "Stage", absl::Milliseconds(10));
  AccumulateStat(inactive_stats, "Metric", int64_t{1});
  EXPECT_FALSE(inactive_stats.has_value());

  std::optional<ExecutorStats> active_stats = ExecutorStats();
  AccumulateStat(active_stats, "Stage", absl::Milliseconds(10));
  AccumulateStat(active_stats, "Metric", int64_t{1});
  EXPECT_EQ(active_stats->GetLatency("Stage"), absl::Milliseconds(10));
  EXPECT_EQ(std::get<int64_t>(*active_stats->GetMetric("Metric")), 1);
}

TEST(ExecutorStatsTest, ScopedLatencyWithOptional) {
  std::optional<ExecutorStats> inactive_stats = std::nullopt;
  {
    ScopedLatency scoped(inactive_stats, "Stage");
    absl::SleepFor(absl::Milliseconds(2));
  }
  EXPECT_FALSE(inactive_stats.has_value());

  std::optional<ExecutorStats> active_stats = ExecutorStats();
  {
    ScopedLatency scoped(active_stats, "Stage");
    absl::SleepFor(absl::Milliseconds(2));
  }
  EXPECT_TRUE(active_stats->GetLatency("Stage").has_value());
  EXPECT_GE(*active_stats->GetLatency("Stage"), absl::Milliseconds(1));
}

TEST(ExecutorStatsTest, StreamOperatorFormatting) {
  ExecutorStats empty_stats;
  std::ostringstream empty_oss;
  empty_oss << empty_stats;
  EXPECT_TRUE(empty_oss.str().empty());

  ExecutorStats stats;
  stats.module_name = "Embedding";
  stats.Accumulate(kTotalLatency, absl::Milliseconds(100));
  stats.Accumulate("Embedder lookup", absl::Milliseconds(20));
  stats.Accumulate("Text encoder inference", absl::Milliseconds(80));
  stats.Accumulate("(e2e) Embedding num tokens", int64_t{10});

  ExecutorStats vision_substats;
  vision_substats.module_name = "Vision";
  vision_substats.Accumulate(kTotalLatency, absl::Milliseconds(30));
  vision_substats.Accumulate("Vision num images", int64_t{1});
  stats.substats.push_back(std::move(vision_substats));

  std::ostringstream oss;
  oss << stats;
  std::string output = oss.str();

  EXPECT_THAT(output, HasSubstr("====== EMBEDDING STATS ======"));
  EXPECT_THAT(output, HasSubstr("Total Embedding latency [us]: 100000"));
  EXPECT_THAT(output, HasSubstr("(e2e) Embedding num tokens: 10"));
  EXPECT_THAT(output, HasSubstr("------ Embedding breakdown ------"));
  EXPECT_THAT(output, HasSubstr("Embedder lookup latency [us]: 20000 (20%)"));
  EXPECT_THAT(output,
              HasSubstr("Text encoder inference latency [us]: 80000 (80%)"));
  EXPECT_THAT(output, HasSubstr("====== VISION STATS ======"));
  EXPECT_THAT(output, HasSubstr("Total Vision latency [us]: 30000"));
  EXPECT_THAT(output, HasSubstr("Vision num images: 1"));
}

TEST(ExecutorStatsTest, AccumulateLogitsStepIsSeparateFromPlainStep) {
  ExecutorStats stats;
  EXPECT_EQ(stats.logits_steps(), 0);
  EXPECT_EQ(stats.logits_step_time(), absl::ZeroDuration());

  stats.AccumulateLogitsStep(absl::Milliseconds(8));
  stats.AccumulateLogitsStep(absl::Milliseconds(9));

  EXPECT_EQ(stats.logits_steps(), 2);
  EXPECT_EQ(stats.logits_step_time(), absl::Milliseconds(17));
  EXPECT_EQ(stats.plain_steps(), 0);
  EXPECT_EQ(stats.plain_step_time(), absl::ZeroDuration());
}

TEST(ExecutorStatsTest, AccumulateMtpRoundAndPlainStep) {
  ExecutorStats stats;
  EXPECT_EQ(stats.mtp_rounds(), 0);
  EXPECT_EQ(stats.mtp_num_draft_tokens(), 0);
  EXPECT_EQ(stats.mtp_num_accepted_tokens(), 0);
  EXPECT_EQ(stats.mtp_emitted_tokens(), 0);
  EXPECT_EQ(stats.plain_steps(), 0);
  EXPECT_EQ(stats.mtp_drafting_time(), absl::ZeroDuration());
  EXPECT_EQ(stats.mtp_verify_time(), absl::ZeroDuration());
  EXPECT_EQ(stats.mtp_round_time(), absl::ZeroDuration());
  EXPECT_EQ(stats.plain_step_time(), absl::ZeroDuration());

  stats.AccumulateMtpRound(/*drafted_tokens=*/3, /*accepted_tokens=*/2,
                           /*emitted_tokens=*/3,
                           /*drafting_time=*/absl::Milliseconds(11),
                           /*verify_time=*/absl::Milliseconds(10),
                           /*round_time=*/absl::Milliseconds(21));
  stats.AccumulateMtpRound(/*drafted_tokens=*/3, /*accepted_tokens=*/3,
                           /*emitted_tokens=*/5,
                           /*drafting_time=*/absl::Milliseconds(12),
                           /*verify_time=*/absl::Milliseconds(10),
                           /*round_time=*/absl::Milliseconds(32));
  stats.AccumulatePlainStep(absl::Milliseconds(10));
  stats.AccumulatePlainStep(absl::Milliseconds(11));

  EXPECT_EQ(stats.mtp_rounds(), 2);
  EXPECT_EQ(stats.mtp_num_draft_tokens(), 6);
  EXPECT_EQ(stats.mtp_num_accepted_tokens(), 5);
  EXPECT_EQ(stats.mtp_emitted_tokens(), 8);
  EXPECT_EQ(stats.plain_steps(), 2);
  EXPECT_EQ(stats.mtp_drafting_time(), absl::Milliseconds(23));
  EXPECT_EQ(stats.mtp_verify_time(), absl::Milliseconds(20));
  EXPECT_EQ(stats.mtp_round_time(), absl::Milliseconds(53));
  EXPECT_EQ(stats.plain_step_time(), absl::Milliseconds(21));

  EXPECT_EQ(stats.GetMetricAsInt64("missing", -1), -1);
  EXPECT_DOUBLE_EQ(stats.GetMetricAsDouble(kMtpRoundsMetric), 2.0);
  EXPECT_DOUBLE_EQ(stats.GetMetricAsDouble("missing", -2.5), -2.5);
}

TEST(ExecutorStatsTest, AccumulateMtpLenientAccepted) {
  ExecutorStats stats;
  stats.AccumulateMtpLenientAccepted(0);
  // Zero is not recorded, so the metric only appears once a draft token was
  // leniently accepted.
  EXPECT_FALSE(stats.GetMetric(kMtpLenientAcceptedTokensMetric).has_value());
  stats.AccumulateMtpLenientAccepted(2);
  stats.AccumulateMtpLenientAccepted(0);
  stats.AccumulateMtpLenientAccepted(3);
  EXPECT_EQ(stats.mtp_lenient_accepted_tokens(), 5);
}

TEST(ExecutorStatsTest, AccumulateDraftVerification) {
  ExecutorStats stats;
  EXPECT_EQ(stats.draft_verifications(), 0);
  EXPECT_EQ(stats.draft_verify_passes(), 0);
  EXPECT_EQ(stats.draft_verify_draft_tokens(), 0);
  EXPECT_EQ(stats.draft_verify_accepted_tokens(), 0);
  EXPECT_EQ(stats.draft_verify_emitted_tokens(), 0);
  EXPECT_EQ(stats.draft_verify_time(), absl::ZeroDuration());

  stats.AccumulateDraftVerification(/*passes=*/2, /*draft_tokens=*/20,
                                    /*accepted_tokens=*/12,
                                    /*emitted_tokens=*/13,
                                    /*verify_time=*/absl::Milliseconds(30));
  stats.AccumulateDraftVerification(/*passes=*/1, /*draft_tokens=*/4,
                                    /*accepted_tokens=*/4,
                                    /*emitted_tokens=*/4,
                                    /*verify_time=*/absl::Milliseconds(12));

  EXPECT_EQ(stats.draft_verifications(), 2);
  EXPECT_EQ(stats.draft_verify_passes(), 3);
  EXPECT_EQ(stats.draft_verify_draft_tokens(), 24);
  EXPECT_EQ(stats.draft_verify_accepted_tokens(), 16);
  EXPECT_EQ(stats.draft_verify_emitted_tokens(), 17);
  EXPECT_EQ(stats.draft_verify_time(), absl::Milliseconds(42));
  // Draft verification is separate from decode steps.
  EXPECT_EQ(stats.plain_steps(), 0);
  EXPECT_EQ(stats.mtp_rounds(), 0);
}

TEST(ExecutorStatsTest, AccumulateDraftVerifyLenientAccepted) {
  ExecutorStats stats;
  stats.AccumulateDraftVerifyLenientAccepted(0);
  // The metric is only recorded once a draft token has been leniently
  // accepted.
  EXPECT_FALSE(
      stats.GetMetric(kDraftVerifyLenientAcceptedTokensMetric).has_value());
  stats.AccumulateDraftVerifyLenientAccepted(2);
  stats.AccumulateDraftVerifyLenientAccepted(0);
  stats.AccumulateDraftVerifyLenientAccepted(3);
  EXPECT_EQ(stats.draft_verify_lenient_accepted_tokens(), 5);
  // MTP rounds count their lenient accepts separately.
  EXPECT_EQ(stats.mtp_lenient_accepted_tokens(), 0);
}

TEST(SubtractExecutorStatsTest, ReturnsWhatWasAccumulatedBetweenSnapshots) {
  ExecutorStats stats{.module_name = std::string(kLlmModuleName)};
  stats.AccumulatePlainStep(absl::Milliseconds(10));
  stats.AccumulateMtpRound(/*drafted_tokens=*/3, /*accepted_tokens=*/1,
                           /*emitted_tokens=*/2, absl::Milliseconds(1),
                           absl::Milliseconds(2), absl::Milliseconds(3));
  stats.AccumulateMtpLenientAccepted(1);
  stats.Accumulate("ratio", 0.25);
  const ExecutorStats before = stats;

  stats.AccumulatePlainStep(absl::Milliseconds(5));
  stats.AccumulatePlainStep(absl::Milliseconds(6));
  stats.AccumulateDraftVerification(/*passes=*/1, /*draft_tokens=*/4,
                                    /*accepted_tokens=*/3, /*emitted_tokens=*/4,
                                    absl::Milliseconds(7));
  stats.Accumulate("ratio", 0.5);
  stats.Accumulate("new_zero", int64_t{0});

  const ExecutorStats difference = SubtractExecutorStats(stats, before);
  EXPECT_EQ(difference.module_name, kLlmModuleName);
  EXPECT_EQ(difference.plain_steps(), 2);
  EXPECT_EQ(difference.plain_step_time(), absl::Milliseconds(11));
  EXPECT_EQ(difference.draft_verifications(), 1);
  EXPECT_EQ(difference.draft_verify_accepted_tokens(), 3);
  EXPECT_EQ(difference.draft_verify_time(), absl::Milliseconds(7));
  EXPECT_DOUBLE_EQ(difference.GetMetricAsDouble("ratio"), 0.5);
  // New entries are kept as recorded, even zero ones.
  EXPECT_EQ(difference.GetMetric("new_zero"), MetricValue(int64_t{0}));
  // The MTP round and lenient accept happened before `before`.
  EXPECT_FALSE(difference.GetMetric(kMtpRoundsMetric).has_value());
  EXPECT_FALSE(difference.GetLatency(kMtpRoundTimeLatency).has_value());
  EXPECT_FALSE(
      difference.GetMetric(kMtpLenientAcceptedTokensMetric).has_value());
}

TEST(SubtractExecutorStatsTest, ReturnsEmptyStatsForEqualSnapshots) {
  ExecutorStats stats{.module_name = "Embedding"};
  stats.AccumulatePlainStep(absl::Milliseconds(10));
  stats.substats.push_back(ExecutorStats{.module_name = "Vision"});
  stats.substats.back().Accumulate(kTotalLatency, absl::Milliseconds(4));

  const ExecutorStats difference = SubtractExecutorStats(stats, stats);
  EXPECT_EQ(difference.module_name, "Embedding");
  EXPECT_TRUE(difference.latencies.empty());
  EXPECT_TRUE(difference.metrics.empty());
  EXPECT_TRUE(difference.substats.empty());
}

TEST(SubtractExecutorStatsTest, SubtractsSubstatsByModuleName) {
  ExecutorStats before{.module_name = "Embedding"};
  before.substats.push_back(ExecutorStats{.module_name = "Vision"});
  before.substats.back().Accumulate(kTotalLatency, absl::Milliseconds(4));
  before.substats.push_back(ExecutorStats{.module_name = "Audio"});
  before.substats.back().Accumulate(kTotalLatency, absl::Milliseconds(2));

  ExecutorStats after = before;
  // The substats of `after` are in another order, and one is new.
  std::swap(after.substats[0], after.substats[1]);
  after.substats[1].Accumulate(kTotalLatency, absl::Milliseconds(3));
  after.substats.push_back(ExecutorStats{.module_name = "Text"});
  after.substats.back().Accumulate("tokens", int64_t{5});

  const ExecutorStats difference = SubtractExecutorStats(after, before);
  ASSERT_EQ(difference.substats.size(), 2);
  EXPECT_EQ(difference.substats[0].module_name, "Vision");
  EXPECT_EQ(difference.substats[0].GetTotalLatency(), absl::Milliseconds(3));
  EXPECT_EQ(difference.substats[1].module_name, "Text");
  EXPECT_EQ(difference.substats[1].GetMetricAsInt64("tokens"), 5);
}

TEST(SubtractExecutorStatsTest, LeavesOutNewEmptySubstats) {
  const ExecutorStats before{.module_name = "Embedding"};
  ExecutorStats after = before;
  after.substats.push_back(ExecutorStats{.module_name = "Vision"});

  const ExecutorStats difference = SubtractExecutorStats(after, before);
  EXPECT_TRUE(difference.empty());
}

}  // namespace
}  // namespace litert::lm
