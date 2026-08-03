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

#include <optional>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "runtime/engine/io_types.h"
#include "runtime/engine/litert_lm_lib.h"
#include "runtime/proto/litert_lm_metrics.pb.h"
#include "runtime/util/status_macros.h"

namespace litert::lm {

namespace {

proto::BenchmarkTurnData TurnDataToProto(
    const BenchmarkTurnData& data,
    std::optional<double> tokens_per_second = std::nullopt) {
  proto::BenchmarkTurnData proto_data;
  proto_data.set_duration_us(absl::ToInt64Microseconds(data.duration));
  proto_data.set_num_tokens(data.num_tokens);

  if (tokens_per_second.has_value()) {
    proto_data.set_tokens_per_second(*tokens_per_second);
  } else {
    double turn_seconds = absl::ToDoubleSeconds(data.duration);
    if (turn_seconds > 0.0) {
      proto_data.set_tokens_per_second(static_cast<double>(data.num_tokens) /
                                       turn_seconds);
    } else {
      proto_data.set_tokens_per_second(0.0);
    }
  }

  return proto_data;
}

}  // namespace

absl::StatusOr<proto::LitertLmMetrics> ToProto(const LitertLmMetrics& metrics) {
  proto::LitertLmMetrics proto_metrics;

  proto_metrics.set_peak_mem_mb(metrics.peak_mem_mb);
  proto_metrics.set_peak_private_mb(metrics.peak_private_mb);

  if (metrics.benchmark_info.has_value()) {
    const auto& info = metrics.benchmark_info.value();

    *proto_metrics.mutable_benchmark_params() = info.GetBenchmarkParams();

    auto& init_durations_us = *proto_metrics.mutable_init_phase_durations_us();
    for (const auto& [phase, duration] : info.GetInitPhases()) {
      init_durations_us[phase] = absl::ToInt64Microseconds(duration);
    }

    auto& mark_durations_us = *proto_metrics.mutable_mark_durations_us();
    for (const auto& [mark, duration] : info.GetMarkDurations()) {
      mark_durations_us[mark] = absl::ToInt64Microseconds(duration);
    }

    for (int i = 0; i < info.GetTotalPrefillTurns(); ++i) {
      ASSIGN_OR_RETURN(auto turn, info.GetPrefillTurn(i));
      *proto_metrics.add_prefill_turns() =
          TurnDataToProto(turn, info.GetPrefillTokensPerSec(i));
    }

    for (int i = 0; i < info.GetTotalDecodeTurns(); ++i) {
      ASSIGN_OR_RETURN(auto turn, info.GetDecodeTurn(i));
      *proto_metrics.add_decode_turns() =
          TurnDataToProto(turn, info.GetDecodeTokensPerSec(i));
    }

    for (int i = 0; i < info.GetTotalTextToTokenIdsTurns(); ++i) {
      ASSIGN_OR_RETURN(auto turn, info.GetTextToTokenIdsTurn(i));
      *proto_metrics.add_text_to_token_ids_turns() = TurnDataToProto(turn);
    }

    proto_metrics.set_time_to_first_token_seconds(info.GetTimeToFirstToken());
  }

  return proto_metrics;
}

absl::StatusOr<proto::LitertLmMetricsList> ToProtoList(
    const std::vector<LitertLmMetrics>& metrics_list) {
  proto::LitertLmMetricsList proto_list;
  for (const auto& metrics : metrics_list) {
    ASSIGN_OR_RETURN(*proto_list.add_metrics(), ToProto(metrics));
  }
  return proto_list;
}

}  // namespace litert::lm
