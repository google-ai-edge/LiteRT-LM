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

}  // namespace
}  // namespace litert::lm
