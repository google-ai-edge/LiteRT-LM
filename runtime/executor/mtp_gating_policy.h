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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_MTP_GATING_POLICY_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_MTP_GATING_POLICY_H_

#include <cstdint>
#include <optional>
#include <ostream>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl

namespace litert::lm {

// Configuration of the adaptive MTP (speculative decoding) gating policy.
//
// `MtpGatingPolicy` sanitizes the configuration so that its state machine
// always progresses: counts are raised to at least 1, and a `deadband` or
// `profit_margin` outside its valid range (see below) is replaced by its
// default.
//
struct MtpGatingConfig {
  // Half-life, in observations, of the exponential moving averages that track
  // the cost of speculative rounds and plain decode steps.
  double ema_half_life_steps = 32;

  // Minimum number of observations per branch before the policy acts.
  int min_samples = 16;

  // Deadband (δ). While speculating, speculation is disabled only if its cost
  // per token exceeds the plain decode cost per token by more than this
  // fraction. Must be non-negative.
  double deadband = 0.05;

  // While disabled, speculation is re-enabled only if a probe measures a cost
  // per token below `plain * (1 - profit_margin)`. Must be in [0, 1).
  double profit_margin = 0.05;

  // Number of plain steps between probes while speculation is disabled.
  int probe_interval = 64;

  // Number of speculative steps per probe. Probes on low-acceptance text are
  // slower than plain decoding, so they are kept short.
  int probe_length = 4;

  // Ceiling of the geometric back-off applied to `probe_interval` after
  // unsuccessful probes. A failed probe costs little (`probe_length` steps at
  // slightly worse than plain speed), while probing too rarely delays
  // re-enabling speculation when the content changes (e.g. from prose to
  // code), so this ceiling is kept low.
  int max_probe_interval = 128;

  // Flap damping: when speculation is disabled again within
  // `max_flap_probe_interval` steps of a probe re-enabling it, the probe
  // interval keeps doubling (up to this ceiling) instead of being reset. This
  // bounds switching on adversarial traces whose probes look better than
  // sustained speculation.
  int max_flap_probe_interval = 512;

  // Test/diagnostic only: if > 0, ignores all estimates and alternates between
  // speculative and plain decoding every `forced_switch_period` steps.
  int forced_switch_period = 0;

  bool operator==(const MtpGatingConfig& other) const = default;
};

// Prints a human-readable representation of `config` for logging and debugging.
std::ostream& operator<<(std::ostream& os, const MtpGatingConfig& config);

// Returns an `InvalidArgumentError` if a field of `config` is out of its valid
// range, i.e. if `MtpGatingPolicy` would have to sanitize it. Use this to
// reject user-supplied configurations instead of silently sanitizing them.
// Ceilings below `probe_interval` are not errors: the policy raises them.
absl::Status ValidateMtpGatingConfig(const MtpGatingConfig& config);

// Counters and current estimates of an `MtpGatingPolicy`. Counters are 64-bit
// because they accumulate across turns and conversations for the lifetime of
// the executor (matching `ExecutorStats::MetricValue`), whereas
// `MtpGatingConfig` holds small per-phase step thresholds.
struct MtpGatingStats {
  // Number of valid speculative rounds observed (lifetime).
  int64_t steps_speculated = 0;

  // Number of valid plain decode steps observed (lifetime).
  int64_t steps_plain = 0;

  // Number of regime changes (enable <-> disable decisions). Bootstrap plain
  // phases and probes are not counted as switches.
  int64_t switches = 0;

  // Number of probes started while speculation was disabled.
  int64_t probes = 0;

  // Number of bootstrap plain phases forced to measure the plain baseline.
  int64_t bootstrap_phases = 0;

  // Number of ignored (degenerate) observations.
  int64_t invalid_observations = 0;

  // Number of valid observations excluded from the estimates because they
  // were the first step of a turn or after a reset.
  int64_t skipped_observations = 0;

  // Number of `ResetEstimates()` calls.
  int64_t estimate_resets = 0;

  // Number of `OnNewConversation()` calls.
  int64_t conversation_resets = 0;

  // True if the policy currently considers speculation profitable.
  bool speculating = true;

  // Exponential moving average of wall-clock time per speculative round, in ms.
  // Zero if not yet initialized.
  double ema_round_ms = 0;

  // Exponential moving average of tokens emitted per speculative round. Zero if
  // not yet initialized.
  double ema_round_tokens = 0;

  // Exponential moving average of wall-clock time per token for plain decode
  // steps, in ms. Zero if not yet initialized.
  double ema_plain_ms = 0;
};

// Decides, step by step, whether speculative decoding (MTP) should be used.
//
// The policy compares the measured wall-clock cost per emitted token of
// speculative rounds against the cost of plain decode steps, and turns
// speculation off when it does not pay off. While off, it periodically probes
// speculation to detect content changes that make it profitable again.
//
// The class has no dependencies on LiteRT and is not thread-safe.
//
// The policy only decides whether to speculate. It deliberately has no access
// to the acceptance-test lenience (`SessionConfig::SetSpeculativeLenience`),
// which trades output quality for speed and is owned by the caller.
//
// Usage, for every decode step:
//   const bool speculate = policy.ShouldSpeculate();
//   ... run the speculative or plain branch ...
//   policy.Observe({speculate, tokens_emitted, elapsed});
//
class MtpGatingPolicy {
 public:
  // Outcome of a single decode step passed to `Observe()`.
  struct Observation {
    // Whether the step ran the speculative (MTP) branch.
    bool speculated = false;

    // Number of tokens emitted by the step. For speculative steps this is in
    // [1, γ + 2] (it includes the priming token when the round was primed).
    int tokens_emitted = 0;

    // Wall-clock time of the step.
    absl::Duration elapsed;
  };

  // Constructs a policy with the given configuration (sanitized to valid
  // ranges).
  explicit MtpGatingPolicy(MtpGatingConfig config = {});

  // Returns whether the next step should speculate. Call exactly once before
  // each step.
  bool ShouldSpeculate() const;

  // Records the outcome of a step. Call exactly once after each step.
  // Degenerate observations (tokens_emitted <= 0 or non-positive/infinite
  // elapsed time) are ignored for the estimates and counted in
  // `invalid_observations`; they still advance the step clock that drives
  // probing. The first observation after construction, `OnPrefill()` or
  // `ResetEstimates()` is also excluded from the estimates: the first decode
  // step of a turn carries one-off costs (priming, warm-up) that are not
  // representative of steady-state decoding.
  void Observe(const Observation& obs);

  // Signals the start of a new turn. Everything carries over: the estimates,
  // the probe back-off (including flap damping), and the steps spent in the
  // current phase, so that turns shorter than `probe_interval` are still
  // probed on schedule. The exception is an in-progress probe, which cannot be
  // evaluated across a prefill: it is retried once, right after the first step
  // of the new turn. The next observation is excluded from the estimates.
  void OnPrefill();

  // Signals the start of a new conversation on the same executor, i.e. a
  // prefill into an empty KV cache. The cost estimates (wall time of a
  // speculative round and of a plain step) depend only on the device and the
  // model, so they are kept. The yield estimate (tokens per speculative round)
  // depends on the content, so the previous conversation's value is dropped,
  // and the policy speculates until it has `min_samples` rounds of the new
  // conversation. Otherwise this behaves like `OnPrefill()`.
  void OnNewConversation();

  // Signals that the sampler or decoding configuration changed. Drops the
  // speculative estimates (the plain estimate is kept) and resumes speculating
  // until warm-up completes again. The next observation is excluded from the
  // estimates.
  void ResetEstimates();

  // Returns a snapshot of the policy's counters and current EMA estimates.
  MtpGatingStats stats() const;

  // Returns the sanitized configuration used by the policy.
  const MtpGatingConfig& config() const { return config_; }

 private:
  // Operating phase of the gating state machine.
  enum class Phase {
    // Speculating (including warm-up).
    kSpeculating,

    // Forcing plain steps once to measure the plain baseline.
    kBootstrapPlain,

    // Speculation disabled; running plain steps until the next probe.
    kDisabled,

    // Speculation disabled; running a short speculative probe.
    kProbing,
  };

  // Exponential moving average with a warm-up: until `1 / samples` drops below
  // the steady-state smoothing factor, the n-th sample is weighted by `1 / n`
  // (a running mean), so the first sample does not dominate for several
  // half-lives.
  struct Ema {
    // Current estimate. Only meaningful if at least one sample has been
    // incorporated since `value` was last seeded or cleared.
    double value = 0;

    // Number of samples incorporated since the last reset.
    int64_t samples = 0;

    // Incorporates `sample` with weight `max(steady_alpha, 1 / samples)`.
    void Update(double sample, double steady_alpha);
  };

  // Estimated wall-clock cost per emitted token of the speculative branch, in
  // ms (or +infinity if uninitialized).
  double MtpCostPerToken() const;

  // Estimated wall-clock cost per emitted token of the plain branch, in ms (or
  // +infinity if uninitialized).
  double PlainCostPerToken() const;

  // Whether both the speculative yield and plain estimates have at least
  // `min_samples` valid observations.
  bool IsWarm() const;

  // Whether speculation costs more per token than plain decoding, beyond the
  // deadband.
  bool SpeculationUnprofitable() const;

  // Whether the policy currently considers speculation profitable. Bootstrap
  // plain phases count as speculating; probes do not.
  bool InSpeculatingRegime() const;

  // Incorporates a valid observation into the speculative or plain EMA
  // estimates.
  void UpdateEstimates(const Observation& obs);

  // Advances the phase state machine after recording an observation.
  void AdvancePhase(const Observation& obs, bool valid);

  // Transitions to `kDisabled` and sets the next probe interval (applying flap
  // damping when applicable).
  void EnterDisabled();

  // Transitions to `kSpeculating`, recording whether re-entry was triggered by
  // a successful probe.
  void EnterSpeculating(bool by_probe);

  // Sanitized policy configuration.
  MtpGatingConfig config_;

  // Steady-state EMA smoothing factor derived from
  // `config_.ema_half_life_steps`.
  double alpha_;

  // Current phase of the gating state machine.
  Phase phase_ = Phase::kSpeculating;

  // Wall-clock time per speculative round, in ms. Samples are counted since
  // construction or the last `ResetEstimates()`.
  Ema round_ms_;

  // Emitted tokens per speculative round. Samples are counted since
  // construction or the last `ResetEstimates()` or `OnNewConversation()`.
  Ema round_tokens_;

  // Wall-clock time per token for plain decode steps, in ms. Never reset.
  Ema plain_ms_;

  // Whether the next valid observation is excluded from the estimates (set on
  // construction, `OnPrefill()`, and `ResetEstimates()`).
  bool skip_next_observation_ = true;

  // Steps spent in the current phase. 64-bit because it is not reset across
  // turns, so a long-lived speculating executor keeps counting.
  int64_t phase_steps_ = 0;

  // Number of plain decode steps to wait in `kDisabled` before starting the
  // next probe (subject to geometric back-off).
  int current_probe_interval_;

  // Flap damping: if the current `kSpeculating` period was started by a
  // successful probe, the probe interval in effect before that probe;
  // otherwise empty.
  std::optional<int> interval_before_reenable_;

  // Sum of elapsed times (in ms) over valid speculative observations in the
  // current probe.
  double probe_ms_sum_ = 0;

  // Sum of emitted tokens over valid speculative observations in the current
  // probe.
  int64_t probe_tokens_sum_ = 0;

  // Number of valid speculative observations recorded in the current probe.
  int probe_valid_steps_ = 0;

  // Whether the pending or current probe retries one interrupted by a prefill.
  bool probe_is_retry_ = false;

  // Total observations seen when `config_.forced_switch_period > 0`, used to
  // alternate between speculative and plain decoding deterministically.
  int64_t forced_step_ = 0;

  // Lifetime counters exposed via `stats()`. The `speculating` and EMA fields
  // are not stored here; `stats()` derives them from the policy state.
  MtpGatingStats stats_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_MTP_GATING_POLICY_H_
