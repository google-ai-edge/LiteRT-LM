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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_SPECULATIVE_DECODING_CONTROLLER_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_SPECULATIVE_DECODING_CONTROLLER_H_

#include <cstdint>
#include <memory>
#include <optional>

#include "absl/functional/function_ref.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "runtime/executor/mtp_gating_policy.h"
#include "runtime/executor/speculative_decoding_mode.h"
#include "runtime/proto/sampler_params.pb.h"

namespace litert::lm {

// Decides, for every decode step, whether an executor runs a speculative (MTP)
// round or a plain decode step, according to the requested
// `SpeculativeDecodingMode`.
//
// In `SpeculativeDecodingMode::kAdaptive` (requested, or the default mode for
// steps that request none), the decision is delegated to an `MtpGatingPolicy`,
// which the controller creates on first use and keeps informed of turn and
// conversation boundaries and of sampler changes.
//
// The controller does not depend on LiteRT: the caller owns the MTP drafter and
// runs the decode steps, and passes a callback that loads the drafter.
//
// Usage, for every decode step:
//   ASSIGN_OR_RETURN(const Decision decision, controller.Decide(...));
//   ... run the speculative or plain branch ...
//   controller.Observe(decision, tokens_emitted, elapsed);
//
// Not thread-safe.
class SpeculativeDecodingController {
 public:
  // Decision for a single decode step.
  struct Decision {
    // Whether the step should run a speculative (MTP) round.
    bool speculate = false;

    // Whether the decision was made by the adaptive gating policy. If so, the
    // outcome of the step must be reported with `Observe()`.
    bool adaptive = false;
  };

  // `gating_config` configures the `MtpGatingPolicy` used in
  // `SpeculativeDecodingMode::kAdaptive`. `default_mode` is used for steps
  // that request no mode; it should reflect the executor's creation-time
  // configuration (e.g. the engine's mode if the drafter was loaded at
  // creation, `kOff` otherwise), so that a drafter lazily loaded for one
  // session does not change the behavior of steps that request no mode.
  explicit SpeculativeDecodingController(
      MtpGatingConfig gating_config = {},
      SpeculativeDecodingMode default_mode = SpeculativeDecodingMode::kOff);

  // Decides whether the next decode step should speculate.
  //
  // - `mode`: the requested mode. `std::nullopt` uses the `default_mode` given
  //   at construction.
  // - `drafter_loaded`: whether the MTP drafter is already loaded.
  // - `load_drafter`: loads the MTP drafter. Called only when `mode` needs the
  //   drafter and it is not loaded yet.
  // - `sampler_params`: the session's sampler parameters. The sampler affects
  //   the cost and acceptance of speculative rounds, so the adaptive policy's
  //   speculative estimates are reset when they change (see
  //   `SamplerParamsEquivalent()`).
  // - `constrained`: whether a decoding constraint is active. The constraint
  //   state is not carried across plain and speculative steps, so
  //   `SpeculativeDecodingMode::kAdaptive` does not gate constrained steps and
  //   speculates like `SpeculativeDecodingMode::kOn`.
  //
  // Returns the error of `load_drafter` in `SpeculativeDecodingMode::kOn`.
  // `SpeculativeDecodingMode::kAdaptive` is best effort instead: if the drafter
  // fails to load, it logs a warning and decodes plainly from then on, without
  // retrying.
  absl::StatusOr<Decision> Decide(
      std::optional<SpeculativeDecodingMode> mode, bool drafter_loaded,
      absl::FunctionRef<absl::Status()> load_drafter,
      const std::optional<proto::SamplerParameters>& sampler_params,
      bool constrained);

  // Reports the outcome of a step decided by `decision`: the number of tokens
  // the step emitted and its wall-clock time. No-op unless
  // `decision.adaptive`.
  //
  // Returns the number of regime changes (speculation turned on or off) that
  // the policy decided on this step. The re-enables that `OnPrefill()` or a
  // sampler change force are not gating decisions and are never counted.
  int64_t Observe(const Decision& decision, int tokens_emitted,
                  absl::Duration elapsed);

  // Signals a prefill. `new_conversation` is true for a prefill into an empty
  // KV cache (a new session, or a session that was reset): the policy then
  // drops the previous conversation's content-dependent estimates.
  void OnPrefill(bool new_conversation);

  // Returns the adaptive gating policy, or nullptr if no step has been decided
  // by it yet.
  const MtpGatingPolicy* gating_policy() const { return gating_policy_.get(); }

 private:
  // Decides a `SpeculativeDecodingMode::kAdaptive` step.
  absl::StatusOr<Decision> DecideAdaptive(
      bool drafter_loaded, absl::FunctionRef<absl::Status()> load_drafter,
      const std::optional<proto::SamplerParameters>& sampler_params,
      bool constrained);

  // Configuration of `gating_policy_`.
  MtpGatingConfig gating_config_;

  // Mode used for steps that request no mode.
  SpeculativeDecodingMode default_mode_;

  // Adaptive gating policy. Created on the first step decided in
  // `SpeculativeDecodingMode::kAdaptive`.
  std::unique_ptr<MtpGatingPolicy> gating_policy_;

  // Sampler parameters of the last adaptive step.
  std::optional<proto::SamplerParameters> gating_sampler_params_;

  // Set when `SpeculativeDecodingMode::kAdaptive` failed to load the drafter.
  bool drafter_unavailable_ = false;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_SPECULATIVE_DECODING_CONTROLLER_H_
