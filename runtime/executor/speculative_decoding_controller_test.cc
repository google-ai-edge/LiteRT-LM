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

#include "runtime/executor/speculative_decoding_controller.h"

#include <cstdint>
#include <optional>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "runtime/executor/mtp_gating_policy.h"
#include "runtime/executor/speculative_decoding_mode.h"
#include "runtime/proto/sampler_params.pb.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

using ::testing::status::StatusIs;
using Decision = SpeculativeDecodingController::Decision;

// Stands in for the executor's MTP drafter loading.
class FakeDrafter {
 public:
  explicit FakeDrafter(absl::Status load_status = absl::OkStatus(),
                       bool loaded = false)
      : load_status_(load_status), loaded_(loaded) {}

  absl::Status Load() {
    ++load_calls_;
    if (load_status_.ok()) {
      loaded_ = true;
    }
    return load_status_;
  }

  bool loaded() const { return loaded_; }
  int load_calls() const { return load_calls_; }

 private:
  absl::Status load_status_;
  bool loaded_;
  int load_calls_ = 0;
};

absl::StatusOr<Decision> Decide(SpeculativeDecodingController& controller,
                                std::optional<SpeculativeDecodingMode> mode,
                                FakeDrafter& drafter,
                                const std::optional<proto::SamplerParameters>&
                                    sampler_params = std::nullopt,
                                bool constrained = false) {
  return controller.Decide(
      mode, drafter.loaded(), [&drafter] { return drafter.Load(); },
      sampler_params, constrained);
}

proto::SamplerParameters TopP(int k) {
  proto::SamplerParameters params;
  params.set_type(proto::SamplerParameters::TOP_P);
  params.set_k(k);
  params.set_p(0.95f);
  params.set_temperature(1.0f);
  return params;
}

// Runs `num_steps` adaptive steps in which a speculative round costs 50 ms and
// emits a single token while a plain step costs 10 ms, so that speculation does
// not pay off. Returns the number of regime changes reported by `Observe()`.
int64_t RunUnprofitableAdaptiveSteps(SpeculativeDecodingController& controller,
                                     FakeDrafter& drafter, int num_steps) {
  int64_t reported_switches = 0;
  for (int i = 0; i < num_steps; ++i) {
    absl::StatusOr<Decision> decision =
        Decide(controller, SpeculativeDecodingMode::kAdaptive, drafter);
    if (!decision.ok()) {
      ADD_FAILURE() << decision.status();
      break;
    }
    EXPECT_TRUE(decision->adaptive);
    reported_switches += controller.Observe(
        *decision, /*tokens_emitted=*/1,
        decision->speculate ? absl::Milliseconds(50) : absl::Milliseconds(10));
  }
  return reported_switches;
}

TEST(SpeculativeDecodingControllerTest, UnsetModeDefaultsToOff) {
  SpeculativeDecodingController controller;
  FakeDrafter drafter;
  ASSERT_OK_AND_ASSIGN(const Decision decision,
                       Decide(controller, std::nullopt, drafter));
  EXPECT_FALSE(decision.speculate);
  EXPECT_FALSE(decision.adaptive);
  EXPECT_EQ(drafter.load_calls(), 0);
}

TEST(SpeculativeDecodingControllerTest, UnsetModeUsesDefaultMode) {
  SpeculativeDecodingController controller(MtpGatingConfig(),
                                           SpeculativeDecodingMode::kOn);
  FakeDrafter drafter(absl::OkStatus(), /*loaded=*/true);
  ASSERT_OK_AND_ASSIGN(const Decision decision,
                       Decide(controller, std::nullopt, drafter));
  EXPECT_TRUE(decision.speculate);
  EXPECT_FALSE(decision.adaptive);
  EXPECT_EQ(drafter.load_calls(), 0);
}

// A drafter lazily loaded for one session must not make sessions that request
// no mode speculate when the executor was created without it.
TEST(SpeculativeDecodingControllerTest,
     UnsetModeIgnoresDrafterLoadedForAnotherSession) {
  SpeculativeDecodingController controller(MtpGatingConfig(),
                                           SpeculativeDecodingMode::kOff);
  FakeDrafter drafter;
  ASSERT_OK_AND_ASSIGN(
      Decision decision,
      Decide(controller, SpeculativeDecodingMode::kOn, drafter));
  EXPECT_TRUE(decision.speculate);
  ASSERT_TRUE(drafter.loaded());

  ASSERT_OK_AND_ASSIGN(decision, Decide(controller, std::nullopt, drafter));
  EXPECT_FALSE(decision.speculate);
  EXPECT_FALSE(decision.adaptive);
  EXPECT_EQ(drafter.load_calls(), 1);
}

TEST(SpeculativeDecodingControllerTest, UnsetModeGatesWithAdaptiveDefault) {
  SpeculativeDecodingController controller(MtpGatingConfig(),
                                           SpeculativeDecodingMode::kAdaptive);
  FakeDrafter drafter(absl::OkStatus(), /*loaded=*/true);
  int64_t reported_switches = 0;
  for (int i = 0; i < 40; ++i) {
    ASSERT_OK_AND_ASSIGN(const Decision decision,
                         Decide(controller, std::nullopt, drafter));
    EXPECT_TRUE(decision.adaptive);
    reported_switches += controller.Observe(
        decision, /*tokens_emitted=*/1,
        decision.speculate ? absl::Milliseconds(50) : absl::Milliseconds(10));
  }
  EXPECT_EQ(reported_switches, 1);
  ASSERT_NE(controller.gating_policy(), nullptr);
  EXPECT_FALSE(controller.gating_policy()->stats().speculating);
  EXPECT_EQ(drafter.load_calls(), 0);
}

TEST(SpeculativeDecodingControllerTest,
     UnsetModeWithAdaptiveDefaultSpeculatesUnderConstraints) {
  SpeculativeDecodingController controller(MtpGatingConfig(),
                                           SpeculativeDecodingMode::kAdaptive);
  FakeDrafter drafter(absl::OkStatus(), /*loaded=*/true);
  ASSERT_OK_AND_ASSIGN(
      const Decision decision,
      Decide(controller, std::nullopt, drafter,
             /*sampler_params=*/std::nullopt, /*constrained=*/true));
  EXPECT_TRUE(decision.speculate);
  EXPECT_FALSE(decision.adaptive);
}

TEST(SpeculativeDecodingControllerTest, OffNeverLoadsDrafter) {
  SpeculativeDecodingController controller;
  FakeDrafter drafter;
  ASSERT_OK_AND_ASSIGN(
      const Decision decision,
      Decide(controller, SpeculativeDecodingMode::kOff, drafter));
  EXPECT_FALSE(decision.speculate);
  EXPECT_FALSE(decision.adaptive);
  EXPECT_EQ(drafter.load_calls(), 0);
}

TEST(SpeculativeDecodingControllerTest, OnLoadsDrafterOnce) {
  SpeculativeDecodingController controller;
  FakeDrafter drafter;
  for (int i = 0; i < 2; ++i) {
    ASSERT_OK_AND_ASSIGN(
        const Decision decision,
        Decide(controller, SpeculativeDecodingMode::kOn, drafter));
    EXPECT_TRUE(decision.speculate);
    EXPECT_FALSE(decision.adaptive);
  }
  EXPECT_EQ(drafter.load_calls(), 1);
  EXPECT_EQ(controller.gating_policy(), nullptr);
}

TEST(SpeculativeDecodingControllerTest, OnFailsIfDrafterFailsToLoad) {
  SpeculativeDecodingController controller;
  FakeDrafter drafter(absl::InternalError("no drafter"));
  EXPECT_THAT(Decide(controller, SpeculativeDecodingMode::kOn, drafter),
              StatusIs(absl::StatusCode::kInternal));
  EXPECT_THAT(Decide(controller, SpeculativeDecodingMode::kOn, drafter),
              StatusIs(absl::StatusCode::kInternal));
  EXPECT_EQ(drafter.load_calls(), 2);
}

TEST(SpeculativeDecodingControllerTest,
     AdaptiveWithoutDrafterDecodesPlainlyWithoutRetrying) {
  SpeculativeDecodingController controller;
  FakeDrafter drafter(absl::InternalError("no drafter"));
  for (int i = 0; i < 3; ++i) {
    ASSERT_OK_AND_ASSIGN(
        const Decision decision,
        Decide(controller, SpeculativeDecodingMode::kAdaptive, drafter));
    EXPECT_FALSE(decision.speculate);
    EXPECT_FALSE(decision.adaptive);
  }
  EXPECT_EQ(drafter.load_calls(), 1);
  EXPECT_EQ(controller.gating_policy(), nullptr);
}

TEST(SpeculativeDecodingControllerTest,
     AdaptiveWithConstraintSpeculatesWithoutGating) {
  SpeculativeDecodingController controller;
  FakeDrafter drafter;
  ASSERT_OK_AND_ASSIGN(
      const Decision decision,
      Decide(controller, SpeculativeDecodingMode::kAdaptive, drafter,
             /*sampler_params=*/std::nullopt, /*constrained=*/true));
  EXPECT_TRUE(decision.speculate);
  EXPECT_FALSE(decision.adaptive);
  EXPECT_EQ(drafter.load_calls(), 1);
  EXPECT_EQ(controller.gating_policy(), nullptr);
}

TEST(SpeculativeDecodingControllerTest, AdaptiveCreatesPolicyWithConfig) {
  SpeculativeDecodingController controller(
      MtpGatingConfig{.probe_interval = 7});
  FakeDrafter drafter;
  ASSERT_OK_AND_ASSIGN(
      const Decision decision,
      Decide(controller, SpeculativeDecodingMode::kAdaptive, drafter));
  // A new policy speculates until it has estimates.
  EXPECT_TRUE(decision.speculate);
  EXPECT_TRUE(decision.adaptive);
  EXPECT_EQ(drafter.load_calls(), 1);
  ASSERT_NE(controller.gating_policy(), nullptr);
  EXPECT_EQ(controller.gating_policy()->config().probe_interval, 7);
}

TEST(SpeculativeDecodingControllerTest, ObserveReturnsPolicySwitches) {
  SpeculativeDecodingController controller(
      MtpGatingConfig{.forced_switch_period = 2});
  FakeDrafter drafter;
  std::vector<bool> speculated;
  std::vector<int64_t> switches;
  for (int i = 0; i < 6; ++i) {
    ASSERT_OK_AND_ASSIGN(
        const Decision decision,
        Decide(controller, SpeculativeDecodingMode::kAdaptive, drafter));
    speculated.push_back(decision.speculate);
    switches.push_back(controller.Observe(decision, /*tokens_emitted=*/1,
                                          absl::Milliseconds(10)));
  }
  EXPECT_THAT(speculated,
              ::testing::ElementsAre(true, true, false, false, true, true));
  EXPECT_THAT(switches, ::testing::ElementsAre(0, 1, 0, 1, 0, 1));
}

TEST(SpeculativeDecodingControllerTest, ObserveIgnoresNonAdaptiveDecisions) {
  SpeculativeDecodingController controller;
  FakeDrafter drafter;
  ASSERT_OK(Decide(controller, SpeculativeDecodingMode::kAdaptive, drafter));
  ASSERT_NE(controller.gating_policy(), nullptr);

  EXPECT_EQ(controller.Observe({.speculate = true, .adaptive = false},
                               /*tokens_emitted=*/3, absl::Milliseconds(20)),
            0);
  const MtpGatingStats stats = controller.gating_policy()->stats();
  EXPECT_EQ(stats.steps_speculated, 0);
  EXPECT_EQ(stats.steps_plain, 0);
}

TEST(SpeculativeDecodingControllerTest,
     ReEnableForcedByNewConversationIsNotReported) {
  SpeculativeDecodingController controller;
  FakeDrafter drafter;
  // Speculation is turned off once both branches have `min_samples` estimates.
  EXPECT_EQ(RunUnprofitableAdaptiveSteps(controller, drafter, 40), 1);
  ASSERT_NE(controller.gating_policy(), nullptr);
  EXPECT_FALSE(controller.gating_policy()->stats().speculating);
  EXPECT_EQ(controller.gating_policy()->stats().switches, 1);

  // A new conversation re-enables speculation, which the policy counts as a
  // switch but is not a gating decision.
  controller.OnPrefill(/*new_conversation=*/true);
  EXPECT_EQ(controller.gating_policy()->stats().switches, 2);
  ASSERT_OK_AND_ASSIGN(
      const Decision decision,
      Decide(controller, SpeculativeDecodingMode::kAdaptive, drafter));
  EXPECT_TRUE(decision.speculate);
  EXPECT_EQ(controller.Observe(decision, /*tokens_emitted=*/1,
                               absl::Milliseconds(50)),
            0);
}

TEST(SpeculativeDecodingControllerTest,
     ReEnableForcedBySamplerChangeIsNotReported) {
  SpeculativeDecodingController controller;
  FakeDrafter drafter;
  EXPECT_EQ(RunUnprofitableAdaptiveSteps(controller, drafter, 40), 1);
  ASSERT_NE(controller.gating_policy(), nullptr);
  EXPECT_FALSE(controller.gating_policy()->stats().speculating);

  ASSERT_OK_AND_ASSIGN(const Decision decision,
                       Decide(controller, SpeculativeDecodingMode::kAdaptive,
                              drafter, TopP(/*k=*/40)));
  EXPECT_EQ(controller.gating_policy()->stats().estimate_resets, 1);
  EXPECT_TRUE(decision.speculate);
  EXPECT_EQ(controller.Observe(decision, /*tokens_emitted=*/1,
                               absl::Milliseconds(50)),
            0);
}

TEST(SpeculativeDecodingControllerTest, ResetsEstimatesOnlyOnSamplerChange) {
  SpeculativeDecodingController controller;
  FakeDrafter drafter;
  auto decide = [&](const std::optional<proto::SamplerParameters>& params) {
    ASSERT_OK(Decide(controller, SpeculativeDecodingMode::kAdaptive, drafter,
                     params));
  };

  decide(std::nullopt);
  ASSERT_NE(controller.gating_policy(), nullptr);
  // Unset and unspecified sampler parameters both select the default sampler.
  decide(proto::SamplerParameters());
  EXPECT_EQ(controller.gating_policy()->stats().estimate_resets, 0);

  decide(TopP(/*k=*/40));
  EXPECT_EQ(controller.gating_policy()->stats().estimate_resets, 1);
  decide(TopP(/*k=*/40));
  EXPECT_EQ(controller.gating_policy()->stats().estimate_resets, 1);
  decide(TopP(/*k=*/1));
  EXPECT_EQ(controller.gating_policy()->stats().estimate_resets, 2);
}

TEST(SpeculativeDecodingControllerTest, OnPrefillNotifiesPolicy) {
  SpeculativeDecodingController controller;
  // No-op before any adaptive step.
  controller.OnPrefill(/*new_conversation=*/true);
  EXPECT_EQ(controller.gating_policy(), nullptr);

  FakeDrafter drafter;
  ASSERT_OK_AND_ASSIGN(
      const Decision decision,
      Decide(controller, SpeculativeDecodingMode::kAdaptive, drafter));
  ASSERT_NE(controller.gating_policy(), nullptr);
  // The first observation of a policy is excluded from its estimates.
  controller.Observe(decision, /*tokens_emitted=*/3, absl::Milliseconds(20));
  controller.Observe(decision, /*tokens_emitted=*/3, absl::Milliseconds(20));
  EXPECT_EQ(controller.gating_policy()->stats().skipped_observations, 1);

  // A new turn excludes its first observation too.
  controller.OnPrefill(/*new_conversation=*/false);
  controller.Observe(decision, /*tokens_emitted=*/3, absl::Milliseconds(20));
  EXPECT_EQ(controller.gating_policy()->stats().skipped_observations, 2);
  EXPECT_EQ(controller.gating_policy()->stats().conversation_resets, 0);

  controller.OnPrefill(/*new_conversation=*/true);
  EXPECT_EQ(controller.gating_policy()->stats().conversation_resets, 1);
}

}  // namespace
}  // namespace litert::lm
