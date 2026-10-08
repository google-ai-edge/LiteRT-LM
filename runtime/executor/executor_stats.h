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
inline constexpr absl::string_view kLlmModuleName = "LLM";

// Speculative decoding / LLM decode metric names. An MTP round covers drafting
// plus verification only; the plain decode that re-primes the drafter before a
// round is counted as a plain step. Plain steps and MTP rounds include
// sampling. A logits step is a decode that returns logits to the caller
// without sampling (e.g. external sampling or scoring); it excludes sampling,
// so it is tracked separately from plain steps.
inline constexpr absl::string_view kMtpRoundsMetric = "mtp_rounds";
inline constexpr absl::string_view kMtpNumDraftTokensMetric =
    "mtp_num_draft_tokens";
inline constexpr absl::string_view kMtpNumAcceptedTokensMetric =
    "mtp_num_accepted_tokens";
inline constexpr absl::string_view kMtpEmittedTokensMetric =
    "mtp_emitted_tokens";
inline constexpr absl::string_view kPlainStepsMetric = "plain_steps";
inline constexpr absl::string_view kLogitsStepsMetric = "logits_steps";

// Speculative decoding / LLM decode latency names.
inline constexpr absl::string_view kMtpDraftingTimeLatency =
    "mtp_drafting_time";
inline constexpr absl::string_view kMtpVerifyTimeLatency = "mtp_verify_time";
inline constexpr absl::string_view kMtpRoundTimeLatency = "mtp_round_time";
inline constexpr absl::string_view kPlainStepTimeLatency = "plain_step_time";
inline constexpr absl::string_view kLogitsStepTimeLatency = "logits_step_time";

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

  // Typed convenience accessors returning `default_value` when absent or
  // recorded with a different type.
  int64_t GetMetricAsInt64(absl::string_view name,
                           int64_t default_value = 0) const;
  double GetMetricAsDouble(absl::string_view name,
                           double default_value = 0.0) const;
  absl::Duration GetLatencyOrZero(absl::string_view name) const {
    return GetLatency(name).value_or(absl::ZeroDuration());
  }

  // Convenience accessor for total latency.
  absl::Duration GetTotalLatency() const {
    return GetLatencyOrZero(kTotalLatency);
  }

  // Convenience helpers for LLM decode and speculative decoding stats.
  void AccumulateMtpRound(int64_t drafted_tokens, int64_t accepted_tokens,
                          int64_t emitted_tokens, absl::Duration drafting_time,
                          absl::Duration verify_time,
                          absl::Duration round_time);
  void AccumulatePlainStep(absl::Duration step_time);
  void AccumulateLogitsStep(absl::Duration step_time);

  int64_t mtp_rounds() const { return GetMetricAsInt64(kMtpRoundsMetric); }
  int64_t mtp_num_draft_tokens() const {
    return GetMetricAsInt64(kMtpNumDraftTokensMetric);
  }
  int64_t mtp_num_accepted_tokens() const {
    return GetMetricAsInt64(kMtpNumAcceptedTokensMetric);
  }
  int64_t mtp_emitted_tokens() const {
    return GetMetricAsInt64(kMtpEmittedTokensMetric);
  }
  int64_t plain_steps() const { return GetMetricAsInt64(kPlainStepsMetric); }
  int64_t logits_steps() const { return GetMetricAsInt64(kLogitsStepsMetric); }
  absl::Duration mtp_drafting_time() const {
    return GetLatencyOrZero(kMtpDraftingTimeLatency);
  }
  absl::Duration mtp_verify_time() const {
    return GetLatencyOrZero(kMtpVerifyTimeLatency);
  }
  absl::Duration mtp_round_time() const {
    return GetLatencyOrZero(kMtpRoundTimeLatency);
  }
  absl::Duration plain_step_time() const {
    return GetLatencyOrZero(kPlainStepTimeLatency);
  }
  absl::Duration logits_step_time() const {
    return GetLatencyOrZero(kLogitsStepTimeLatency);
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
