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
#include <ostream>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/strings/ascii.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl

namespace litert::lm {

namespace {

double SafePercentage(absl::Duration part, absl::Duration total) {
  if (total <= absl::ZeroDuration()) return 0.0;
  return absl::FDivDuration(part, total) * 100.0;
}

}  // namespace

void ExecutorStats::Accumulate(absl::string_view name,
                               absl::Duration duration) {
  for (auto& [key, val] : latencies) {
    if (key == name) {
      val += duration;
      return;
    }
  }
  latencies.emplace_back(std::string(name), duration);
}

void ExecutorStats::Accumulate(absl::string_view name, MetricValue value) {
  for (auto& [key, val] : metrics) {
    if (key == name) {
      if (val.index() != value.index()) {
        ABSL_LOG(WARNING) << "Metric '" << name
                          << "' recorded with mismatched type (existing index "
                          << val.index() << ", new index " << value.index()
                          << "); skipping update.";
      } else {
        std::visit(
            [&val](auto&& v) {
              using T = std::decay_t<decltype(v)>;
              std::get<T>(val) += v;
            },
            value);
      }
      return;
    }
  }
  metrics.emplace_back(std::string(name), value);
}

std::optional<absl::Duration> ExecutorStats::GetLatency(
    absl::string_view name) const {
  for (const auto& [key, val] : latencies) {
    if (key == name) return val;
  }
  return std::nullopt;
}

std::optional<MetricValue> ExecutorStats::GetMetric(
    absl::string_view name) const {
  for (const auto& [key, val] : metrics) {
    if (key == name) return val;
  }
  return std::nullopt;
}

int64_t ExecutorStats::GetMetricAsInt64(absl::string_view name,
                                        int64_t default_value) const {
  auto metric = GetMetric(name);
  if (!metric.has_value()) {
    return default_value;
  }
  if (const auto* int_val = std::get_if<int64_t>(&*metric)) {
    return *int_val;
  }
  return default_value;
}

double ExecutorStats::GetMetricAsDouble(absl::string_view name,
                                        double default_value) const {
  auto metric = GetMetric(name);
  if (!metric.has_value()) {
    return default_value;
  }
  if (const auto* double_val = std::get_if<double>(&*metric)) {
    return *double_val;
  }
  if (const auto* int_val = std::get_if<int64_t>(&*metric)) {
    return static_cast<double>(*int_val);
  }
  return default_value;
}

void ExecutorStats::AccumulateMtpRound(int64_t drafted_tokens,
                                       int64_t accepted_tokens,
                                       int64_t emitted_tokens,
                                       absl::Duration drafting_time,
                                       absl::Duration verify_time,
                                       absl::Duration round_time) {
  Accumulate(kMtpRoundsMetric, int64_t{1});
  Accumulate(kMtpNumDraftTokensMetric, drafted_tokens);
  Accumulate(kMtpNumAcceptedTokensMetric, accepted_tokens);
  Accumulate(kMtpEmittedTokensMetric, emitted_tokens);
  Accumulate(kMtpDraftingTimeLatency, drafting_time);
  Accumulate(kMtpVerifyTimeLatency, verify_time);
  Accumulate(kMtpRoundTimeLatency, round_time);
}

void ExecutorStats::AccumulatePlainStep(absl::Duration step_time) {
  Accumulate(kPlainStepsMetric, int64_t{1});
  Accumulate(kPlainStepTimeLatency, step_time);
}

void ExecutorStats::AccumulateLogitsStep(absl::Duration step_time) {
  Accumulate(kLogitsStepsMetric, int64_t{1});
  Accumulate(kLogitsStepTimeLatency, step_time);
}

std::ostream& operator<<(std::ostream& os, const ExecutorStats& stats) {
  if (stats.latencies.empty() && stats.metrics.empty() &&
      stats.substats.empty()) {
    return os;
  }

  std::string name_upper = absl::AsciiStrToUpper(stats.module_name);

  os << "\n" << "====== " << name_upper << " STATS ======";

  absl::Duration total_lat = stats.GetTotalLatency();
  if (total_lat > absl::ZeroDuration()) {
    os << "\n"
       << "Total " << stats.module_name
       << " latency [us]: " << absl::ToInt64Microseconds(total_lat);
  }

  for (const auto& [label, val] : stats.metrics) {
    std::visit([&](auto&& v) { os << "\n" << label << ": " << v; }, val);
  }

  bool has_breakdown = false;
  for (const auto& [label, duration] : stats.latencies) {
    if (label != kTotalLatency) {
      has_breakdown = true;
      break;
    }
  }

  if (has_breakdown) {
    os << "\n" << "------ " << stats.module_name << " breakdown ------";

    for (const auto& [label, duration] : stats.latencies) {
      if (label == kTotalLatency) continue;
      os << "\n"
         << label << " latency [us]: " << absl::ToInt64Microseconds(duration);
      if (total_lat > absl::ZeroDuration()) {
        os << " (" << SafePercentage(duration, total_lat) << "%)";
      }
    }
  }

  for (const auto& sub : stats.substats) {
    os << sub;
  }
  return os;
}

}  // namespace litert::lm
