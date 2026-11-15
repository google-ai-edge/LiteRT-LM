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
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <sstream>
#include <type_traits>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl

namespace litert::lm {
namespace {

using ::testing::AnyOf;
using ::testing::Each;
using ::testing::ElementsAre;
using ::testing::HasSubstr;

// Synthetic cost model matching the measured Gemma 4 2.3B GPU numbers: a plain
// decode step costs 10 ms and emits 1 token, a speculative round costs 22 ms
// and emits `tokens_per_round` tokens on average. A speculative step that
// follows a plain step (or is the first step) runs the priming decode first:
// it costs an extra plain step and emits one extra token.
constexpr double kPlainMs = 10.0;
constexpr double kRoundMs = 22.0;

class Simulator {
 public:
  // `tau(step)` returns the expected number of tokens per speculative round at
  // `step`. Adversarial traces can capture the simulator to also depend on
  // `steps_since_plain()`.
  using TauFn = std::function<double(int step)>;

  Simulator(MtpGatingPolicy& policy, TauFn tau) : policy_(policy), tau_(tau) {}

  // Runs `n` steps and returns the decisions made.
  std::vector<bool> Run(int n) {
    std::vector<bool> decisions;
    decisions.reserve(n);
    for (int i = 0; i < n; ++i) {
      const bool speculate = policy_.ShouldSpeculate();
      decisions.push_back(speculate);
      policy_.Observe(Step(speculate));
      ++step_;
    }
    return decisions;
  }

  int step() const { return step_; }
  int steps_since_plain() const { return steps_since_plain_; }

 private:
  MtpGatingPolicy::Observation Step(bool speculate) {
    if (!speculate) {
      primed_ = false;
      steps_since_plain_ = 0;
      return {.speculated = false,
              .tokens_emitted = 1,
              .elapsed = absl::Milliseconds(kPlainMs)};
    }
    ++steps_since_plain_;
    // Deterministically realize a fractional average number of tokens.
    const double tau = tau_(step_);
    const double next_acc = token_acc_ + tau;
    int tokens = static_cast<int>(std::floor(next_acc)) -
                 static_cast<int>(std::floor(token_acc_));
    token_acc_ = next_acc;
    double ms = kRoundMs;
    if (!primed_) {
      tokens += 1;
      ms += kPlainMs;
      primed_ = true;
    }
    return {.speculated = true,
            .tokens_emitted = tokens,
            .elapsed = absl::Microseconds(static_cast<int64_t>(ms * 1000))};
  }

  MtpGatingPolicy& policy_;
  TauFn tau_;
  int step_ = 0;
  int steps_since_plain_ = 0;
  bool primed_ = false;
  double token_acc_ = 0;
};

// Returns the lengths of the maximal runs of `value` in `decisions`.
std::vector<int> RunLengths(const std::vector<bool>& decisions, bool value) {
  std::vector<int> runs;
  int current = 0;
  for (bool d : decisions) {
    if (d == value) {
      ++current;
    } else if (current > 0) {
      runs.push_back(current);
      current = 0;
    }
  }
  if (current > 0) runs.push_back(current);
  return runs;
}

// The speculative decoding lenience is a quality/speed trade-off
// owned by the caller; the gating policy must not be able to read or change
// it. These detectors fail to compile-time-assert if a lenience knob is ever
// added to the policy or its configuration.
template <typename T, typename = void>
struct HasLenienceMember : std::false_type {};
template <typename T>
struct HasLenienceMember<T, std::void_t<decltype(std::declval<T&>().lenience)>>
    : std::true_type {};
template <typename T, typename = void>
struct HasSpeculativeLenienceMember : std::false_type {};
template <typename T>
struct HasSpeculativeLenienceMember<
    T, std::void_t<decltype(std::declval<T&>().speculative_lenience)>>
    : std::true_type {};
template <typename T, typename = void>
struct HasSetLenience : std::false_type {};
template <typename T>
struct HasSetLenience<
    T, std::void_t<decltype(std::declval<T&>().SetLenience(0.0f))>>
    : std::true_type {};

static_assert(!HasLenienceMember<MtpGatingConfig>::value);
static_assert(!HasSpeculativeLenienceMember<MtpGatingConfig>::value);
static_assert(!HasSetLenience<MtpGatingPolicy>::value);

TEST(MtpGatingConfigTest, Equality) {
  EXPECT_EQ(MtpGatingConfig(), MtpGatingConfig());
  EXPECT_NE(MtpGatingConfig(), MtpGatingConfig{.max_probe_interval = 256});
}

TEST(MtpGatingConfigTest, ValidateAcceptsDefaultsAndBoundaries) {
  EXPECT_TRUE(ValidateMtpGatingConfig({}).ok());
  EXPECT_TRUE(ValidateMtpGatingConfig({.deadband = 0, .profit_margin = 0}).ok());
  // An infinite half-life gives a running mean.
  EXPECT_TRUE(ValidateMtpGatingConfig(
                  {.ema_half_life_steps =
                       std::numeric_limits<double>::infinity()})
                  .ok());
  // A ceiling below `probe_interval` is raised by the policy.
  EXPECT_TRUE(
      ValidateMtpGatingConfig({.probe_interval = 256, .max_probe_interval = 1})
          .ok());
  EXPECT_TRUE(ValidateMtpGatingConfig({.forced_switch_period = 16}).ok());
}

TEST(MtpGatingConfigTest, ValidateRejectsOutOfRange) {
  constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
  const MtpGatingConfig invalid_configs[] = {
      {.ema_half_life_steps = 0},
      {.ema_half_life_steps = kNaN},
      {.min_samples = 0},
      {.deadband = -0.1},
      {.deadband = kNaN},
      {.profit_margin = -0.5},
      {.profit_margin = 1.0},
      {.profit_margin = kNaN},
      {.probe_interval = 0},
      {.probe_length = 0},
      {.max_probe_interval = 0},
      {.max_flap_probe_interval = 0},
      {.forced_switch_period = -1},
  };
  for (const MtpGatingConfig& config : invalid_configs) {
    EXPECT_EQ(ValidateMtpGatingConfig(config).code(),
              absl::StatusCode::kInvalidArgument)
        << config;
  }
}

TEST(MtpGatingPolicyTest, StartsSpeculating) {
  MtpGatingPolicy policy;
  EXPECT_TRUE(policy.ShouldSpeculate());
  EXPECT_TRUE(policy.stats().speculating);
}

TEST(MtpGatingPolicyTest, EmaFollowsHalfLifeDefinition) {
  MtpGatingPolicy policy({.ema_half_life_steps = 1});
  // The first observation is excluded from the estimates.
  policy.Observe({.speculated = true,
                  .tokens_emitted = 1,
                  .elapsed = absl::Milliseconds(500)});
  // alpha = 1 - 2^-1 = 0.5; the first included sample seeds the EMA.
  policy.Observe({.speculated = true,
                  .tokens_emitted = 2,
                  .elapsed = absl::Milliseconds(20)});
  policy.Observe({.speculated = true,
                  .tokens_emitted = 4,
                  .elapsed = absl::Milliseconds(10)});
  policy.Observe({.speculated = false,
                  .tokens_emitted = 1,
                  .elapsed = absl::Milliseconds(8)});
  const MtpGatingStats stats = policy.stats();
  EXPECT_DOUBLE_EQ(stats.ema_round_ms, 15.0);
  EXPECT_DOUBLE_EQ(stats.ema_round_tokens, 3.0);
  EXPECT_DOUBLE_EQ(stats.ema_plain_ms, 8.0);
  EXPECT_EQ(stats.steps_speculated, 3);
  EXPECT_EQ(stats.steps_plain, 1);
  EXPECT_EQ(stats.skipped_observations, 1);
}

TEST(MtpGatingPolicyTest, WarmUpAveragesSamplesEqually) {
  // With a long half-life the EMA starts as a running mean, so the first
  // sample does not dominate the estimate.
  MtpGatingPolicy policy({.ema_half_life_steps = 1000});
  policy.Observe({.speculated = false,
                  .tokens_emitted = 1,
                  .elapsed = absl::Milliseconds(10)});  // Skipped.
  for (int ms : {100, 20, 20, 20}) {
    policy.Observe({.speculated = true,
                    .tokens_emitted = 2,
                    .elapsed = absl::Milliseconds(ms)});
  }
  EXPECT_NEAR(policy.stats().ema_round_ms, 40.0, 1e-9);
  EXPECT_NEAR(policy.stats().ema_round_tokens, 2.0, 1e-9);
}

TEST(MtpGatingPolicyTest, FirstStepAfterPrefillIsExcludedFromEstimates) {
  MtpGatingPolicy policy;
  policy.Observe({.speculated = true,
                  .tokens_emitted = 2,
                  .elapsed = absl::Milliseconds(300)});  // Skipped.
  policy.Observe({.speculated = true,
                  .tokens_emitted = 2,
                  .elapsed = absl::Milliseconds(20)});
  EXPECT_DOUBLE_EQ(policy.stats().ema_round_ms, 20.0);

  // The first step of the next turn is excluded as well.
  policy.OnPrefill();
  policy.Observe({.speculated = true,
                  .tokens_emitted = 2,
                  .elapsed = absl::Milliseconds(300)});  // Skipped.
  EXPECT_DOUBLE_EQ(policy.stats().ema_round_ms, 20.0);
  EXPECT_EQ(policy.stats().skipped_observations, 2);
  EXPECT_EQ(policy.stats().steps_speculated, 3);
}

TEST(MtpGatingPolicyTest, SteadyHighAcceptanceStaysOn) {
  const MtpGatingConfig config;
  MtpGatingPolicy policy(config);
  Simulator sim(policy, [](int) { return 3.0; });
  const std::vector<bool> decisions = sim.Run(4000);

  const MtpGatingStats stats = policy.stats();
  EXPECT_TRUE(stats.speculating);
  EXPECT_EQ(stats.switches, 0);
  EXPECT_EQ(stats.probes, 0);
  // Only the one bootstrap plain phase is taken.
  EXPECT_EQ(stats.bootstrap_phases, 1);
  EXPECT_EQ(stats.steps_plain, config.min_samples);
  EXPECT_THAT(RunLengths(decisions, false), ElementsAre(config.min_samples));
}

TEST(MtpGatingPolicyTest, SteadyLowAcceptanceDisablesAndBacksOff) {
  const MtpGatingConfig config;
  MtpGatingPolicy policy(config);
  Simulator sim(policy, [](int) { return 1.87; });

  // Warm-up (min_samples speculative steps), bootstrap (min_samples plain
  // steps), then at most one half-life to decide.
  const int decide_by =
      2 * config.min_samples + static_cast<int>(config.ema_half_life_steps);
  sim.Run(decide_by);
  EXPECT_FALSE(policy.stats().speculating);
  EXPECT_EQ(policy.stats().switches, 1);

  const std::vector<bool> decisions = sim.Run(6000);
  const MtpGatingStats stats = policy.stats();
  EXPECT_FALSE(stats.speculating);
  EXPECT_EQ(stats.switches, 1);
  EXPECT_GE(stats.probes, 4);

  // Probes are exactly `probe_length` long, and the plain gaps between them
  // back off geometrically up to `max_probe_interval`.
  std::vector<int> probe_runs = RunLengths(decisions, true);
  for (int run : probe_runs) {
    EXPECT_EQ(run, config.probe_length);
  }
  std::vector<int> plain_runs = RunLengths(decisions, false);
  ASSERT_GE(plain_runs.size(), 3);
  EXPECT_EQ(plain_runs[plain_runs.size() - 2], config.max_probe_interval);
  // Speculative share is small: probes only, about one probe per
  // `max_probe_interval` plain steps.
  EXPECT_LT(stats.steps_speculated * config.max_probe_interval,
            2 * config.probe_length * stats.steps_plain);
}

// Runs `n` steps where speculative rounds cost 22 ms for 2 tokens (11 ms/token)
// and plain steps cost `plain_ms`, following the policy's decisions.
MtpGatingStats RunMarginalTrace(MtpGatingPolicy& policy, double plain_ms,
                                int n) {
  for (int i = 0; i < n; ++i) {
    if (policy.ShouldSpeculate()) {
      policy.Observe({.speculated = true,
                      .tokens_emitted = 2,
                      .elapsed = absl::Milliseconds(22)});
    } else {
      policy.Observe({.speculated = false,
                      .tokens_emitted = 1,
                      .elapsed = absl::Microseconds(
                          static_cast<int64_t>(plain_ms * 1000))});
    }
  }
  return policy.stats();
}

TEST(MtpGatingPolicyTest, MarginallySlowerSpeculationWithinDeadbandStaysOn) {
  // Speculation is ~3% slower per token than plain decoding, which is within
  // the default deadband.
  MtpGatingPolicy policy;
  const MtpGatingStats stats =
      RunMarginalTrace(policy, /*plain_ms=*/10.7, /*n=*/500);
  EXPECT_TRUE(stats.speculating);
  EXPECT_EQ(stats.switches, 0);
}

TEST(MtpGatingPolicyTest, SpeculationSlowerThanDeadbandDisables) {
  // Same trace, but speculation is ~22% slower per token than plain decoding.
  MtpGatingPolicy policy;
  const MtpGatingStats stats =
      RunMarginalTrace(policy, /*plain_ms=*/9.0, /*n=*/500);
  EXPECT_FALSE(stats.speculating);
  EXPECT_EQ(stats.switches, 1);
}

TEST(MtpGatingPolicyTest, StepFromLowToHighReenables) {
  const MtpGatingConfig config;
  constexpr int kChangeStep = 500;
  MtpGatingPolicy policy(config);
  Simulator sim(policy,
                [](int step) { return step < kChangeStep ? 1.87 : 3.0; });
  sim.Run(kChangeStep);
  ASSERT_FALSE(policy.stats().speculating);

  // A probe that straddles the change may fail; the next one must succeed.
  const int bound = config.max_probe_interval + 2 * config.probe_length;
  int reenabled_after = -1;
  for (int i = 0; i < bound; ++i) {
    sim.Run(1);
    if (policy.stats().speculating) {
      reenabled_after = i + 1;
      break;
    }
  }
  EXPECT_GT(reenabled_after, 0);
  EXPECT_LE(reenabled_after, bound);

  // Once re-enabled, it stays on.
  const int64_t switches = policy.stats().switches;
  sim.Run(2000);
  EXPECT_TRUE(policy.stats().speculating);
  EXPECT_EQ(policy.stats().switches, switches);
}

TEST(MtpGatingPolicyTest, SawtoothAroundBreakEvenHasFewSwitches) {
  MtpGatingPolicy policy;
  // tau oscillates between 1.9 and 2.7 tokens/round (11.6 to 8.1 ms/token)
  // around the break-even point of 2.2 tokens/round with a period of 400.
  Simulator sim(policy, [](int step) {
    const double phase = (step % 400) / 400.0;
    const double tri = phase < 0.5 ? 2 * phase : 2 * (1 - phase);
    return 1.9 + 0.8 * tri;
  });
  constexpr int kSteps = 20000;
  sim.Run(kSteps);
  EXPECT_LE(policy.stats().switches * 100, kSteps);
}

TEST(MtpGatingPolicyTest, AlternatingTraceHasBoundedSwitches) {
  MtpGatingPolicy policy;
  // tau alternates between 1 and 4 tokens/round every 37 steps.
  Simulator sim(policy, [](int step) { return (step / 37) % 2 ? 1.0 : 4.0; });
  constexpr int kSteps = 20000;
  sim.Run(kSteps);
  EXPECT_LE(policy.stats().switches * 100, kSteps);
}

TEST(MtpGatingPolicyTest, AdversarialProbeFriendlyTraceHasBoundedSwitches) {
  MtpGatingPolicy policy;
  // The adversary makes every probe look great (4 tokens/round for the first
  // speculative steps after plain decoding) and every sustained speculative
  // period look bad (1 token/round).
  Simulator* sim_ptr = nullptr;
  Simulator sim(policy, [&sim_ptr](int) {
    return sim_ptr->steps_since_plain() <= 8 ? 4.0 : 1.0;
  });
  sim_ptr = &sim;
  constexpr int kSteps = 20000;
  sim.Run(kSteps);
  // The adversary does cause flapping, but flap damping bounds it.
  EXPECT_GE(policy.stats().switches, 2);
  EXPECT_LE(policy.stats().switches * 100, kSteps);
}

TEST(MtpGatingPolicyTest,
     AdversarialProbeFriendlyTraceWithShortTurnsHasBoundedSwitches) {
  MtpGatingPolicy policy;
  // Same adversary, but with a new turn every 50 steps: flap damping must
  // carry over across turns.
  Simulator* sim_ptr = nullptr;
  Simulator sim(policy, [&sim_ptr](int) {
    return sim_ptr->steps_since_plain() <= 8 ? 4.0 : 1.0;
  });
  sim_ptr = &sim;
  constexpr int kSteps = 20000;
  constexpr int kTurnSteps = 50;
  for (int i = 0; i < kSteps / kTurnSteps; ++i) {
    policy.OnPrefill();
    sim.Run(kTurnSteps);
  }
  EXPECT_LE(policy.stats().switches * 100, kSteps);
}

TEST(MtpGatingPolicyTest, OnPrefillKeepsEstimates) {
  MtpGatingPolicy policy;
  Simulator sim(policy, [](int) { return 1.87; });
  sim.Run(200);
  ASSERT_FALSE(policy.stats().speculating);
  const MtpGatingStats before = policy.stats();

  policy.OnPrefill();
  const MtpGatingStats after = policy.stats();
  EXPECT_DOUBLE_EQ(after.ema_round_ms, before.ema_round_ms);
  EXPECT_DOUBLE_EQ(after.ema_round_tokens, before.ema_round_tokens);
  EXPECT_DOUBLE_EQ(after.ema_plain_ms, before.ema_plain_ms);
  EXPECT_FALSE(after.speculating);
  // No warm-up is needed: the policy keeps running plain steps until the next
  // probe. The steps already spent waiting before the prefill carry over, and
  // so does the back-off, which is at most `max_probe_interval`.
  int plain_steps = 0;
  while (!policy.ShouldSpeculate()) {
    sim.Run(1);
    ++plain_steps;
    ASSERT_LE(plain_steps, policy.config().max_probe_interval);
  }
}

TEST(MtpGatingPolicyTest, OnPrefillKeepsProbeBackOff) {
  const MtpGatingConfig config;
  MtpGatingPolicy policy(config);
  Simulator sim(policy, [](int) { return 1.87; });
  // Long enough for the back-off to reach `max_probe_interval`.
  sim.Run(4000);
  ASSERT_FALSE(policy.stats().speculating);

  // Many turns, each shorter than the probe interval.
  constexpr int kTurnSteps = 32;
  std::vector<bool> decisions;
  for (int turn = 0; turn < 8 * config.max_probe_interval / kTurnSteps;
       ++turn) {
    policy.OnPrefill();
    const std::vector<bool> turn_decisions = sim.Run(kTurnSteps);
    decisions.insert(decisions.end(), turn_decisions.begin(),
                     turn_decisions.end());
  }
  // The plain gaps between probes keep their backed-off length instead of
  // dropping back to `probe_interval` at every turn. A probe that a turn
  // boundary interrupts is retried after the first (plain) step of the next
  // turn, which adds a gap of 1 (or extends the gap by 1 if the probe had not
  // run any step yet). The first and last gaps are cut off by the window.
  const std::vector<int> plain_runs = RunLengths(decisions, false);
  ASSERT_GE(plain_runs.size(), 4);
  for (size_t i = 1; i + 1 < plain_runs.size(); ++i) {
    EXPECT_THAT(plain_runs[i], AnyOf(1, config.max_probe_interval,
                                     config.max_probe_interval + 1))
        << "gap " << i;
  }
  // Probes run to completion despite the turn boundaries.
  const std::vector<int> spec_runs = RunLengths(decisions, true);
  EXPECT_GE(std::count(spec_runs.begin(), spec_runs.end(), config.probe_length),
            4);
}

TEST(MtpGatingPolicyTest, ProbesAreNotStarvedWhenTurnLengthDividesInterval) {
  // Without back-off, a probe is due every 64 plain steps, and every turn is
  // 32 steps long. If an interrupted probe restarted the full wait, every
  // later probe would start in the last steps of a turn and be cut short.
  const MtpGatingConfig config = {
      .probe_interval = 64, .probe_length = 8, .max_probe_interval = 64};
  MtpGatingPolicy policy(config);
  Simulator sim(policy, [](int) { return 1.87; });
  sim.Run(500);
  ASSERT_FALSE(policy.stats().speculating);
  constexpr int kTurnSteps = 32;
  // Align the turns so that the next probe starts near the end of a turn.
  while (!policy.ShouldSpeculate()) sim.Run(1);
  sim.Run(config.probe_length - 2);
  std::vector<bool> decisions;
  for (int turn = 0; turn < 40; ++turn) {
    policy.OnPrefill();
    const std::vector<bool> turn_decisions = sim.Run(kTurnSteps);
    decisions.insert(decisions.end(), turn_decisions.begin(),
                     turn_decisions.end());
  }
  const std::vector<int> spec_runs = RunLengths(decisions, true);
  // About one complete probe per `probe_interval + probe_length` steps.
  EXPECT_GE(std::count(spec_runs.begin(), spec_runs.end(), config.probe_length),
            10);
}

TEST(MtpGatingPolicyTest, ShortTurnsStillGetProbed) {
  MtpGatingPolicy policy;
  Simulator sim(policy, [](int) { return 1.87; });
  sim.Run(200);
  ASSERT_FALSE(policy.stats().speculating);
  const int64_t probes_before = policy.stats().probes;
  // Many turns, each shorter than the probe interval, spanning more than the
  // longest backed-off interval.
  const int turn_steps = policy.config().probe_interval / 2;
  const int num_turns = 2 * policy.config().max_probe_interval / turn_steps;
  for (int turn = 0; turn < num_turns; ++turn) {
    policy.OnPrefill();
    sim.Run(turn_steps);
  }
  EXPECT_GT(policy.stats().probes, probes_before);
}

TEST(MtpGatingPolicyTest, OnPrefillRetriesInterruptedProbeOnce) {
  const MtpGatingConfig config;
  MtpGatingPolicy policy(config);
  Simulator sim(policy, [](int) { return 1.87; });
  // Run until a probe is in progress.
  sim.Run(200);
  while (!policy.ShouldSpeculate()) sim.Run(1);
  sim.Run(1);
  const int64_t probes = policy.stats().probes;

  // The interrupted probe is retried after the first (plain) step of the new
  // turn.
  policy.OnPrefill();
  EXPECT_FALSE(policy.ShouldSpeculate());
  sim.Run(1);
  EXPECT_TRUE(policy.ShouldSpeculate());
  EXPECT_EQ(policy.stats().probes, probes + 1);

  // If the retry is interrupted too, the policy waits a full interval.
  sim.Run(1);
  policy.OnPrefill();
  const std::vector<bool> decisions = sim.Run(config.max_probe_interval);
  EXPECT_FALSE(decisions[0]);
  EXPECT_FALSE(decisions[1]);
}

TEST(MtpGatingPolicyTest, ResetEstimatesRequiresWarmUpAgain) {
  const MtpGatingConfig config;
  MtpGatingPolicy policy(config);
  // Clearly unprofitable speculation (14.7 ms/token), so the policy disables
  // as soon as the estimates are warm.
  Simulator sim(policy, [](int) { return 1.5; });
  sim.Run(200);
  ASSERT_FALSE(policy.stats().speculating);

  policy.ResetEstimates();
  EXPECT_EQ(policy.stats().ema_round_ms, 0);
  EXPECT_EQ(policy.stats().ema_round_tokens, 0);
  EXPECT_GT(policy.stats().ema_plain_ms, 0);
  EXPECT_TRUE(policy.stats().speculating);

  // The policy speculates for exactly `min_samples` steps, plus the excluded
  // first step after the reset (the plain estimate is kept, so no bootstrap is
  // needed), before disabling again.
  const std::vector<bool> decisions = sim.Run(3 * config.min_samples);
  ASSERT_FALSE(decisions.empty());
  EXPECT_EQ(RunLengths(decisions, true).front(), config.min_samples + 1);
  EXPECT_FALSE(policy.stats().speculating);
  EXPECT_EQ(policy.stats().bootstrap_phases, 1);
  EXPECT_EQ(policy.stats().estimate_resets, 1);
}

TEST(MtpGatingPolicyTest, OnNewConversationKeepsCostsAndDropsYield) {
  MtpGatingPolicy policy;
  Simulator sim(policy, [](int) { return 1.5; });
  sim.Run(200);
  ASSERT_FALSE(policy.stats().speculating);
  const MtpGatingStats before = policy.stats();

  policy.OnNewConversation();
  const MtpGatingStats after = policy.stats();
  EXPECT_DOUBLE_EQ(after.ema_round_ms, before.ema_round_ms);
  EXPECT_DOUBLE_EQ(after.ema_plain_ms, before.ema_plain_ms);
  EXPECT_EQ(after.ema_round_tokens, 0);
  EXPECT_TRUE(after.speculating);
  EXPECT_TRUE(policy.ShouldSpeculate());
  EXPECT_EQ(after.conversation_resets, 1);
  EXPECT_EQ(after.estimate_resets, 0);
}

TEST(MtpGatingPolicyTest, NewConversationIsNotJudgedByThePreviousOne) {
  const MtpGatingConfig config;
  MtpGatingPolicy policy(config);
  // A low-acceptance conversation that is long enough to warm up and disable
  // speculation...
  double tau = 1.5;
  Simulator sim(policy, [&tau](int) { return tau; });
  sim.Run(200);
  ASSERT_FALSE(policy.stats().speculating);
  const int64_t plain_before = policy.stats().steps_plain;

  // ... followed by a high-acceptance one: the policy speculates from the
  // first step and never runs a plain step, because the plain estimate is
  // kept (no bootstrap) and the new yield is measured on the new content.
  policy.OnNewConversation();
  tau = 3.5;
  const std::vector<bool> decisions = sim.Run(500);
  EXPECT_THAT(RunLengths(decisions, false), ::testing::IsEmpty());
  EXPECT_TRUE(policy.stats().speculating);
  EXPECT_EQ(policy.stats().steps_plain, plain_before);
  EXPECT_EQ(policy.stats().bootstrap_phases, 1);
}

TEST(MtpGatingPolicyTest, NewLowAcceptanceConversationDisablesAfterWarmUp) {
  const MtpGatingConfig config;
  MtpGatingPolicy policy(config);
  Simulator sim(policy, [](int) { return 1.5; });
  sim.Run(200);
  ASSERT_FALSE(policy.stats().speculating);

  // The new conversation is unprofitable too. The policy speculates for
  // `min_samples` rounds of the new content, plus the excluded first step,
  // and disables again without a plain bootstrap.
  policy.OnNewConversation();
  const std::vector<bool> decisions = sim.Run(3 * config.min_samples);
  ASSERT_FALSE(decisions.empty());
  EXPECT_EQ(RunLengths(decisions, true).front(), config.min_samples + 1);
  EXPECT_FALSE(policy.stats().speculating);
  EXPECT_EQ(policy.stats().bootstrap_phases, 1);
}

TEST(MtpGatingPolicyTest, OnNewConversationKeepsPendingBootstrap) {
  const MtpGatingConfig config;
  MtpGatingPolicy policy(config);
  Simulator sim(policy, [](int) { return 3.0; });
  // Warm up the speculative estimates and enter the plain bootstrap.
  sim.Run(config.min_samples + 2);
  ASSERT_FALSE(policy.ShouldSpeculate());
  ASSERT_EQ(policy.stats().bootstrap_phases, 1);

  policy.OnNewConversation();
  // The plain baseline is still missing, so the bootstrap continues.
  EXPECT_FALSE(policy.ShouldSpeculate());
  sim.Run(4 * config.min_samples);
  EXPECT_TRUE(policy.stats().speculating);
  EXPECT_EQ(policy.stats().bootstrap_phases, 1);
}

TEST(MtpGatingPolicyTest, ForcedSwitchPeriodAlternates) {
  MtpGatingPolicy policy({.forced_switch_period = 4});
  Simulator sim(policy, [](int) { return 3.0; });
  const std::vector<bool> decisions = sim.Run(16);
  const std::vector<bool> expected = {true,  true,  true,  true, false, false,
                                      false, false, true,  true, true,  true,
                                      false, false, false, false};
  EXPECT_EQ(decisions, expected);
  EXPECT_EQ(policy.stats().switches, 4);
}

TEST(MtpGatingPolicyTest, ForcedSwitchPeriodIgnoresEstimates) {
  MtpGatingPolicy policy({.forced_switch_period = 3});
  // Terrible speculation would normally be disabled.
  Simulator sim(policy, [](int) { return 0.0; });
  const std::vector<bool> decisions = sim.Run(600);
  EXPECT_EQ(RunLengths(decisions, true).size(), 100);
  EXPECT_THAT(RunLengths(decisions, true), Each(3));
}

TEST(MtpGatingPolicyTest, InvalidObservationsAreIgnored) {
  MtpGatingPolicy policy;
  policy.Observe({.speculated = true,
                  .tokens_emitted = 0,
                  .elapsed = absl::Milliseconds(22)});
  policy.Observe({.speculated = true,
                  .tokens_emitted = -3,
                  .elapsed = absl::Milliseconds(22)});
  policy.Observe({.speculated = false,
                  .tokens_emitted = 1,
                  .elapsed = absl::ZeroDuration()});
  policy.Observe({.speculated = false,
                  .tokens_emitted = 1,
                  .elapsed = absl::Milliseconds(-1)});
  policy.Observe({.speculated = false,
                  .tokens_emitted = 1,
                  .elapsed = absl::InfiniteDuration()});
  const MtpGatingStats stats = policy.stats();
  EXPECT_EQ(stats.invalid_observations, 5);
  EXPECT_EQ(stats.steps_speculated, 0);
  EXPECT_EQ(stats.steps_plain, 0);
  EXPECT_EQ(stats.ema_round_ms, 0);
  EXPECT_EQ(stats.ema_plain_ms, 0);
  EXPECT_TRUE(policy.ShouldSpeculate());
}

TEST(MtpGatingPolicyTest, AllInvalidObservationsDoNotStallBootstrap) {
  const MtpGatingConfig config;
  MtpGatingPolicy policy(config);
  // One more than `min_samples`: the first observation is excluded.
  for (int i = 0; i < config.min_samples + 1; ++i) {
    policy.Observe({.speculated = true,
                    .tokens_emitted = 3,
                    .elapsed = absl::Milliseconds(22)});
  }
  ASSERT_FALSE(policy.ShouldSpeculate());  // Bootstrap started.
  for (int i = 0; i < 10 * config.min_samples; ++i) {
    policy.Observe({.speculated = false,
                    .tokens_emitted = 0,
                    .elapsed = absl::Milliseconds(10)});
  }
  // Without a plain baseline the policy falls back to speculating.
  EXPECT_TRUE(policy.ShouldSpeculate());
}

TEST(MtpGatingPolicyTest, DegenerateConfigDoesNotCrash) {
  MtpGatingPolicy policy({.ema_half_life_steps = 0,
                          .min_samples = 0,
                          .deadband = std::nan(""),
                          .profit_margin = 2.0,
                          .probe_interval = 0,
                          .probe_length = 0,
                          .max_probe_interval = 0});
  // Out-of-range thresholds are replaced by the defaults.
  EXPECT_EQ(policy.config().deadband, MtpGatingConfig().deadband);
  EXPECT_EQ(policy.config().profit_margin, MtpGatingConfig().profit_margin);
  Simulator sim(policy, [](int step) { return step % 2 ? 1.0 : 3.0; });
  sim.Run(1000);
  EXPECT_EQ(policy.stats().invalid_observations, 0);
}

TEST(MtpGatingPolicyTest, NegativeThresholdsAreReplacedByDefaults) {
  MtpGatingPolicy policy({.deadband = -0.1, .profit_margin = -0.1});
  EXPECT_EQ(policy.config().deadband, MtpGatingConfig().deadband);
  EXPECT_EQ(policy.config().profit_margin, MtpGatingConfig().profit_margin);
}

TEST(MtpGatingPolicyTest, ProbeEstimateIgnoresInvalidObservations) {
  const MtpGatingConfig config;
  MtpGatingPolicy policy(config);
  Simulator sim(policy, [](int) { return 1.87; });
  sim.Run(200);
  ASSERT_FALSE(policy.stats().speculating);
  while (!policy.ShouldSpeculate()) sim.Run(1);

  // A probe with a single valid round; the other steps are invalid.
  policy.Observe({.speculated = true,
                  .tokens_emitted = 3,
                  .elapsed = absl::Milliseconds(30)});
  for (int i = 1; i < config.probe_length; ++i) {
    policy.Observe({.speculated = true,
                    .tokens_emitted = 0,
                    .elapsed = absl::Milliseconds(30)});
  }
  // 10 ms/token is not profitable, so the probe fails, and the speculative
  // estimates are re-seeded with the valid round only.
  EXPECT_FALSE(policy.ShouldSpeculate());
  EXPECT_DOUBLE_EQ(policy.stats().ema_round_ms, 30.0);
  EXPECT_DOUBLE_EQ(policy.stats().ema_round_tokens, 3.0);
}

TEST(MtpGatingPolicyTest, ConfigIsPrintable) {
  std::stringstream ss;
  ss << MtpGatingConfig{.probe_interval = 7};
  EXPECT_THAT(ss.str(), HasSubstr("probe_interval=7"));
  EXPECT_THAT(ss.str(), HasSubstr("forced_switch_period=0"));
}

}  // namespace
}  // namespace litert::lm
