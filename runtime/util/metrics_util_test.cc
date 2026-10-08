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

#include "runtime/util/metrics_util.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "runtime/engine/io_types.h"
#include "runtime/engine/litert_lm_lib.h"
#include "runtime/proto/engine.pb.h"
#include "runtime/proto/litert_lm_metrics.pb.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

using ::testing::Contains;
using ::testing::FloatEq;
using ::testing::Pair;
using ::testing::SizeIs;

TEST(MetricsUtilTest, EmptyMetrics) {
  LitertLmMetrics metrics;
  metrics.peak_mem_mb = 12.5f;
  metrics.peak_private_mb = 10.0f;
  metrics.peak_dmabuf_mb = 348.7f;
  metrics.dmabuf_buffer_count = 14;

  ASSERT_OK_AND_ASSIGN(auto proto, ToProto(metrics));

  EXPECT_THAT(proto.peak_mem_mb(), FloatEq(12.5f));
  EXPECT_THAT(proto.peak_private_mb(), FloatEq(10.0f));
  EXPECT_THAT(proto.peak_dmabuf_mb(), FloatEq(348.7f));
  EXPECT_EQ(proto.dmabuf_buffer_count(), 14);
  EXPECT_FALSE(proto.has_benchmark_params());
  EXPECT_THAT(proto.init_phase_durations_us(), SizeIs(0));
  EXPECT_THAT(proto.mark_durations_us(), SizeIs(0));
}

TEST(MetricsUtilTest, WithBenchmarkInfo) {
  proto::BenchmarkParams params;
  params.set_num_prefill_tokens(128);
  params.set_num_decode_tokens(256);

  BenchmarkInfo info(params);

  // Simulate some init phases & turns
  ASSERT_OK(info.InitPhaseRecord(BenchmarkInfo::InitPhase::kModelAssets,
                                 absl::Microseconds(100)));

  ASSERT_OK(info.TimePrefillTurnStart());
  absl::SleepFor(absl::Milliseconds(2));
  ASSERT_OK(info.TimePrefillTurnEnd(100));

  ASSERT_OK(info.TimeDecodeTurnStart());
  absl::SleepFor(absl::Milliseconds(2));
  ASSERT_OK(info.TimeDecodeTurnEnd(1));

  ASSERT_OK(info.TimeMarkDelta("test_mark"));
  absl::SleepFor(absl::Milliseconds(2));
  ASSERT_OK(info.TimeMarkDelta("test_mark"));

  LitertLmMetrics metrics;
  metrics.peak_mem_mb = 100.0f;
  metrics.peak_private_mb = 80.0f;
  metrics.benchmark_info = info;

  ASSERT_OK_AND_ASSIGN(auto proto, ToProto(metrics));

  EXPECT_THAT(proto.peak_mem_mb(), FloatEq(100.0f));
  EXPECT_THAT(proto.peak_private_mb(), FloatEq(80.0f));
  EXPECT_TRUE(proto.has_benchmark_params());
  EXPECT_EQ(proto.benchmark_params().num_prefill_tokens(), 128);
  EXPECT_EQ(proto.benchmark_params().num_decode_tokens(), 256);

  EXPECT_THAT(proto.init_phase_durations_us(), SizeIs(1));
  EXPECT_THAT(proto.init_phase_durations_us(),
              Contains(Pair("Init Model assets", 100)));
  EXPECT_THAT(proto.mark_durations_us().size(), 1);
  EXPECT_THAT(proto.mark_durations_us(), Contains(testing::Key("test_mark")));
  EXPECT_GT(proto.mark_durations_us().at("test_mark"), 0);

  EXPECT_EQ(proto.prefill_turns().size(), 1);
  EXPECT_EQ(proto.prefill_turns(0).num_tokens(), 100);
  EXPECT_EQ(proto.prefill_turns(0).tokens_per_second(),
            info.GetPrefillTokensPerSec(0));

  EXPECT_EQ(proto.decode_turns().size(), 1);
  EXPECT_EQ(proto.decode_turns(0).num_tokens(), 1);
  EXPECT_EQ(proto.decode_turns(0).tokens_per_second(),
            info.GetDecodeTokensPerSec(0));
  EXPECT_EQ(proto.text_to_token_ids_turns().size(), 0);

  EXPECT_EQ(proto.time_to_first_token_seconds(), info.GetTimeToFirstToken());
}

TEST(MetricsUtilTest, VectorToProtoList) {
  LitertLmMetrics m1;
  m1.peak_mem_mb = 1.0f;
  LitertLmMetrics m2;
  m2.peak_mem_mb = 2.0f;

  ASSERT_OK_AND_ASSIGN(auto proto, ToProtoList({m1, m2}));

  EXPECT_EQ(proto.metrics_size(), 2);
  EXPECT_THAT(proto.metrics(0).peak_mem_mb(), FloatEq(1.0f));
  EXPECT_THAT(proto.metrics(1).peak_mem_mb(), FloatEq(2.0f));
}

}  // namespace
}  // namespace litert::lm
