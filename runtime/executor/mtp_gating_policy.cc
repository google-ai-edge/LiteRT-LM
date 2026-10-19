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

#include "runtime/executor/mtp_gating_policy.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <ostream>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl

namespace litert::lm {
namespace {

// Returns the steady-state EMA smoothing factor for a half-life expressed in
// observations: x <- x + alpha * (sample - x), with alpha = 1 - 2^(-1/h).
// Non-positive or NaN half-lives fall back to 1.0 (no smoothing).
double AlphaFromHalfLife(double half_life_steps) {
  if (!(half_life_steps > 0)) {
    return 1.0;
  }
  return 1.0 - std::exp2(-1.0 / half_life_steps);
}

}  // namespace

std::ostream& operator<<(std::ostream& os, const MtpGatingConfig& config) {
  os << "MtpGatingConfig{ema_half_life_steps=" << config.ema_half_life_steps
     << ", min_samples=" << config.min_samples
     << ", deadband=" << config.deadband
     << ", profit_margin=" << config.profit_margin
     << ", probe_interval=" << config.probe_interval
     << ", probe_length=" << config.probe_length
     << ", max_probe_interval=" << config.max_probe_interval
     << ", max_flap_probe_interval=" << config.max_flap_probe_interval
     << ", forced_switch_period=" << config.forced_switch_period << "}";
  return os;
}

absl::Status ValidateMtpGatingConfig(const MtpGatingConfig& config) {
  // The negated comparisons also reject NaN.
  if (!(config.ema_half_life_steps > 0)) {
    return absl::InvalidArgumentError(
        absl::StrCat("MtpGatingConfig.ema_half_life_steps must be positive, "
                     "got ",
                     config.ema_half_life_steps));
  }
  if (!(config.deadband >= 0)) {
    return absl::InvalidArgumentError(
        absl::StrCat("MtpGatingConfig.deadband must be non-negative, got ",
                     config.deadband));
  }
  if (!(config.profit_margin >= 0 && config.profit_margin < 1)) {
    return absl::InvalidArgumentError(
        absl::StrCat("MtpGatingConfig.profit_margin must be in [0, 1), got ",
                     config.profit_margin));
  }
  const struct {
    const char* name;
    int value;
    int min;
  } int_fields[] = {
      {"min_samples", config.min_samples, 1},
      {"probe_interval", config.probe_interval, 1},
      {"probe_length", config.probe_length, 1},
      {"max_probe_interval", config.max_probe_interval, 1},
      {"max_flap_probe_interval", config.max_flap_probe_interval, 1},
      {"forced_switch_period", config.forced_switch_period, 0},
  };
  for (const auto& field : int_fields) {
    if (field.value < field.min) {
      return absl::InvalidArgumentError(
          absl::StrCat("MtpGatingConfig.", field.name, " must be at least ",
                       field.min, ", got ", field.value));
    }
  }
  return absl::OkStatus();
}

MtpGatingPolicy::MtpGatingPolicy(MtpGatingConfig config)
    : config_(config), alpha_(AlphaFromHalfLife(config.ema_half_life_steps)) {
  const MtpGatingConfig defaults;
  config_.min_samples = std::max(1, config_.min_samples);
  config_.probe_interval = std::max(1, config_.probe_interval);
  config_.probe_length = std::max(1, config_.probe_length);
  config_.max_probe_interval =
      std::max(config_.probe_interval, config_.max_probe_interval);
  config_.max_flap_probe_interval =
      std::max(config_.max_probe_interval, config_.max_flap_probe_interval);
  // The negated comparisons also catch NaN.
  if (!(config_.deadband >= 0)) {
    config_.deadband = defaults.deadband;
  }
  if (!(config_.profit_margin >= 0 && config_.profit_margin < 1)) {
    config_.profit_margin = defaults.profit_margin;
  }
  current_probe_interval_ = config_.probe_interval;
}

void MtpGatingPolicy::Ema::Update(double sample, double steady_alpha) {
  ++samples;
  // With `samples == 1` the weight is 1, so the first sample replaces any
  // previous value.
  const double alpha =
      std::max(steady_alpha, 1.0 / static_cast<double>(samples));
  value += alpha * (sample - value);
}

double MtpGatingPolicy::MtpCostPerToken() const {
  if (round_ms_.samples == 0 || !(round_tokens_.value > 0)) {
    return std::numeric_limits<double>::infinity();
  }
  // Ratio of EMAs (rather than an EMA of per-round ms/tokens) so that rounds
  // are weighted by the number of tokens they emit.
  return round_ms_.value / round_tokens_.value;
}

double MtpGatingPolicy::PlainCostPerToken() const {
  return plain_ms_.samples > 0 ? plain_ms_.value
                               : std::numeric_limits<double>::infinity();
}

bool MtpGatingPolicy::IsWarm() const {
  // `round_tokens_.samples <= round_ms_.samples`, so this also requires
  // `min_samples` round-cost samples.
  return round_tokens_.samples >= config_.min_samples &&
         plain_ms_.samples >= config_.min_samples;
}

bool MtpGatingPolicy::SpeculationUnprofitable() const {
  return MtpCostPerToken() > PlainCostPerToken() * (1.0 + config_.deadband);
}

bool MtpGatingPolicy::InSpeculatingRegime() const {
  if (config_.forced_switch_period > 0) {
    return ShouldSpeculate();
  }
  return phase_ == Phase::kSpeculating || phase_ == Phase::kBootstrapPlain;
}

bool MtpGatingPolicy::ShouldSpeculate() const {
  if (config_.forced_switch_period > 0) {
    return (forced_step_ / config_.forced_switch_period) % 2 == 0;
  }
  switch (phase_) {
    case Phase::kSpeculating:
    case Phase::kProbing:
      return true;
    case Phase::kBootstrapPlain:
    case Phase::kDisabled:
      return false;
  }
  return true;
}

void MtpGatingPolicy::UpdateEstimates(const Observation& obs) {
  const double ms = absl::ToDoubleMilliseconds(obs.elapsed);
  if (obs.speculated) {
    round_ms_.Update(ms, alpha_);
    // The first sample after `OnNewConversation()` replaces the previous
    // conversation's yield.
    round_tokens_.Update(static_cast<double>(obs.tokens_emitted), alpha_);
  } else {
    // Plain steps emit one token; normalize defensively.
    plain_ms_.Update(ms / static_cast<double>(obs.tokens_emitted), alpha_);
  }
}

void MtpGatingPolicy::EnterDisabled() {
  if (InSpeculatingRegime()) {
    ++stats_.switches;
  }
  // Flap damping: if a probe re-enabled speculation and it turned out to be
  // unprofitable soon after, keep backing off from the interval that led to
  // that probe instead of probing again at the base rate.
  if (interval_before_reenable_.has_value() && phase_ == Phase::kSpeculating &&
      phase_steps_ < config_.max_flap_probe_interval) {
    current_probe_interval_ = std::min(config_.max_flap_probe_interval,
                                       2 * *interval_before_reenable_);
  } else {
    current_probe_interval_ = config_.probe_interval;
  }
  interval_before_reenable_.reset();
  probe_is_retry_ = false;
  phase_ = Phase::kDisabled;
  phase_steps_ = 0;
}

void MtpGatingPolicy::EnterSpeculating(bool by_probe) {
  if (!InSpeculatingRegime()) {
    ++stats_.switches;
  }
  probe_is_retry_ = false;
  // Preserve the active back-off interval before resetting it so
  // `EnterDisabled()` can continue backing off if this re-enable flaps.
  if (by_probe) {
    interval_before_reenable_ = current_probe_interval_;
  } else {
    interval_before_reenable_.reset();
  }
  phase_ = Phase::kSpeculating;
  phase_steps_ = 0;
  current_probe_interval_ = config_.probe_interval;
}

void MtpGatingPolicy::AdvancePhase(const Observation& obs, bool valid) {
  ++phase_steps_;
  switch (phase_) {
    case Phase::kSpeculating: {
      if (stats_.bootstrap_phases == 0 &&
          round_ms_.samples >= config_.min_samples &&
          plain_ms_.samples < config_.min_samples) {
        // No plain baseline yet: force `min_samples` plain steps, at most once
        // per policy lifetime.
        ++stats_.bootstrap_phases;
        phase_ = Phase::kBootstrapPlain;
        phase_steps_ = 0;
        return;
      }
      if (valid && obs.speculated && IsWarm() && SpeculationUnprofitable()) {
        EnterDisabled();
      }
      return;
    }
    case Phase::kBootstrapPlain: {
      if (plain_ms_.samples < config_.min_samples &&
          phase_steps_ < 4 * config_.min_samples) {
        // Keep measuring. The step bound guarantees progress if observations
        // keep being invalid.
        return;
      }
      if (IsWarm() && SpeculationUnprofitable()) {
        EnterDisabled();
      } else {
        // Returning to `kSpeculating` after bootstrap is not a regime switch
        // (bootstrap counts as speculating).
        phase_ = Phase::kSpeculating;
        phase_steps_ = 0;
      }
      return;
    }
    case Phase::kDisabled: {
      if (phase_steps_ >= current_probe_interval_) {
        ++stats_.probes;
        phase_ = Phase::kProbing;
        phase_steps_ = 0;
        probe_ms_sum_ = 0;
        probe_tokens_sum_ = 0;
        probe_valid_steps_ = 0;
      }
      return;
    }
    case Phase::kProbing: {
      if (valid && obs.speculated) {
        probe_ms_sum_ += absl::ToDoubleMilliseconds(obs.elapsed);
        probe_tokens_sum_ += obs.tokens_emitted;
        ++probe_valid_steps_;
      }
      if (phase_steps_ < config_.probe_length) {
        return;
      }
      probe_is_retry_ = false;
      // Evaluate the probe on its own samples only (ratio of sums, which
      // down-weights the priming step). Also overwrite the speculative EMAs —
      // which `UpdateEstimates()` blended with the pre-disable history — so the
      // first step after re-enabling is not judged against stale estimates.
      if (probe_tokens_sum_ > 0) {
        const double probe_ms_per_token =
            probe_ms_sum_ / static_cast<double>(probe_tokens_sum_);
        const double num_probe_steps = static_cast<double>(probe_valid_steps_);
        // `UpdateEstimates()` already counted the probe's rounds, so the
        // sample counts stay as they are.
        round_ms_.value = probe_ms_sum_ / num_probe_steps;
        round_tokens_.value =
            static_cast<double>(probe_tokens_sum_) / num_probe_steps;
        if (probe_ms_per_token <
            PlainCostPerToken() * (1.0 - config_.profit_margin)) {
          EnterSpeculating(/*by_probe=*/true);
          return;
        }
      }
      // Unsuccessful probe: back off geometrically, but never shorten an
      // interval raised by flap damping.
      current_probe_interval_ = std::max(
          current_probe_interval_,
          std::min(config_.max_probe_interval, 2 * current_probe_interval_));
      phase_ = Phase::kDisabled;
      phase_steps_ = 0;
      return;
    }
  }
}

void MtpGatingPolicy::Observe(const Observation& obs) {
  const bool valid = obs.tokens_emitted > 0 &&
                     obs.elapsed > absl::ZeroDuration() &&
                     obs.elapsed != absl::InfiniteDuration();
  if (valid) {
    if (obs.speculated) {
      ++stats_.steps_speculated;
    } else {
      ++stats_.steps_plain;
    }
    if (skip_next_observation_) {
      // Only a valid observation consumes the skip: an invalid one carries no
      // timing, so the next valid step is still the one to discard.
      ++stats_.skipped_observations;
      skip_next_observation_ = false;
    } else {
      UpdateEstimates(obs);
    }
  } else {
    ++stats_.invalid_observations;
  }
  if (config_.forced_switch_period > 0) {
    const bool was_speculating =
        (forced_step_ / config_.forced_switch_period) % 2 == 0;
    ++forced_step_;
    const bool will_speculate =
        (forced_step_ / config_.forced_switch_period) % 2 == 0;
    if (was_speculating != will_speculate) {
      ++stats_.switches;
    }
    return;
  }
  AdvancePhase(obs, valid);
}

void MtpGatingPolicy::OnPrefill() {
  skip_next_observation_ = true;
  if (phase_ != Phase::kProbing) {
    return;
  }
  // Restarting the full wait when a prefill cuts a probe short would starve
  // probing whenever the turn length divides the probe interval. Instead,
  // setting `phase_steps_ = current_probe_interval_ - 1` in `kDisabled` runs
  // the first step of the new turn plain (absorbing one-off post-prefill costs)
  // and starts a retry probe on the second step. If that retry is also cut
  // short, fall back to a full interval wait.
  phase_ = Phase::kDisabled;
  if (probe_is_retry_) {
    probe_is_retry_ = false;
    phase_steps_ = 0;
  } else {
    probe_is_retry_ = true;
    phase_steps_ = current_probe_interval_ - 1;
  }
}

void MtpGatingPolicy::OnNewConversation() {
  OnPrefill();
  round_tokens_.samples = 0;
  ++stats_.conversation_resets;
  if (config_.forced_switch_period > 0) {
    return;
  }
  // Speculate until the new conversation's yield estimate is warm. A pending
  // bootstrap still needs its plain samples first.
  if (phase_ != Phase::kBootstrapPlain) {
    EnterSpeculating(/*by_probe=*/false);
  }
}

void MtpGatingPolicy::ResetEstimates() {
  round_ms_ = {};
  round_tokens_ = {};
  skip_next_observation_ = true;
  ++stats_.estimate_resets;
  if (config_.forced_switch_period > 0) {
    return;
  }
  // Resume speculating to warm up the new estimates, unless the one-time plain
  // bootstrap is still in progress (it will return to `kSpeculating` once
  // done).
  if (phase_ != Phase::kBootstrapPlain) {
    EnterSpeculating(/*by_probe=*/false);
  }
}

MtpGatingStats MtpGatingPolicy::stats() const {
  MtpGatingStats stats = stats_;
  stats.speculating = InSpeculatingRegime();
  stats.ema_round_ms = round_ms_.samples > 0 ? round_ms_.value : 0;
  // `round_tokens_.samples <= round_ms_.samples`.
  stats.ema_round_tokens = round_tokens_.samples > 0 ? round_tokens_.value : 0;
  stats.ema_plain_ms = plain_ms_.samples > 0 ? plain_ms_.value : 0;
  return stats;
}

}  // namespace litert::lm
