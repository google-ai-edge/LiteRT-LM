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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_EXECUTOR_STATS_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_EXECUTOR_STATS_H_

#include <cstdint>
#include <iosfwd>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl

namespace litert::lm {

inline constexpr absl::string_view kTotalLatency = "Total";

using MetricValue = std::variant<int64_t, double>;

// Holds latency and metrics statistics for an executor or submodule.
struct ExecutorStats {
  std::string module_name;

  // Latency records by stage/operation (preserves insertion order).
  // Total execution latency is stored under `kTotalLatency`.
  std::vector<std::pair<std::string, absl::Duration>> latencies;

  // Key-value metrics (counts, tokens, throughputs, etc.).
  std::vector<std::pair<std::string, MetricValue>> metrics;

  // Submodule stats (e.g., Vision / Audio stats within Embedding).
  std::vector<ExecutorStats> substats;

  // Accumulates latency or metrics for a stage/operation.
  void Accumulate(absl::string_view name, absl::Duration duration);
  void Accumulate(absl::string_view name, MetricValue value);

  // Gets recorded stats by name.
  std::optional<absl::Duration> GetLatency(absl::string_view name) const;
  std::optional<MetricValue> GetMetric(absl::string_view name) const;

  // Convenience accessor for total latency.
  absl::Duration GetTotalLatency() const {
    return GetLatency(kTotalLatency).value_or(absl::ZeroDuration());
  }
};

// Accumulates latency into an optional ExecutorStats if profiling is active.
inline void AccumulateStat(std::optional<ExecutorStats>& stats,
                           absl::string_view name, absl::Duration duration) {
  if (stats.has_value()) {
    stats->Accumulate(name, duration);
  }
}

// Accumulates metric into an optional ExecutorStats if profiling is active.
inline void AccumulateStat(std::optional<ExecutorStats>& stats,
                           absl::string_view name, MetricValue value) {
  if (stats.has_value()) {
    stats->Accumulate(name, std::move(value));
  }
}

std::ostream& operator<<(std::ostream& os, const ExecutorStats& stats);

// RAII timer that accumulates duration into an ExecutorStats (by name)
// on destruction. When passed an uninitialized optional ExecutorStats, timing
// is completely skipped.
class ScopedLatency {
 public:
  explicit ScopedLatency(std::optional<ExecutorStats>& stats,
                         absl::string_view name = kTotalLatency)
      : stats_(stats.has_value() ? &stats.value() : nullptr),
        name_(name),
        start_time_(stats_ ? absl::Now() : absl::InfinitePast()) {}

  ~ScopedLatency() {
    if (stats_ != nullptr) {
      stats_->Accumulate(name_, absl::Now() - start_time_);
    }
  }

  ScopedLatency(const ScopedLatency&) = delete;
  ScopedLatency& operator=(const ScopedLatency&) = delete;

 private:
  ExecutorStats* stats_ = nullptr;
  absl::string_view name_;
  absl::Time start_time_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_EXECUTOR_STATS_H_
