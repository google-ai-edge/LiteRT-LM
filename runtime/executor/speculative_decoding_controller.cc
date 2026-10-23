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
#include <memory>
#include <optional>

#include "absl/functional/function_ref.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "runtime/components/sampler_factory.h"
#include "runtime/executor/mtp_gating_policy.h"
#include "runtime/executor/speculative_decoding_mode.h"
#include "runtime/proto/sampler_params.pb.h"

namespace litert::lm {

SpeculativeDecodingController::SpeculativeDecodingController(
    MtpGatingConfig gating_config, SpeculativeDecodingMode default_mode)
    : gating_config_(gating_config), default_mode_(default_mode) {}

absl::StatusOr<SpeculativeDecodingController::Decision>
SpeculativeDecodingController::Decide(
    std::optional<SpeculativeDecodingMode> mode, bool drafter_loaded,
    absl::FunctionRef<absl::Status()> load_drafter,
    const std::optional<proto::SamplerParameters>& sampler_params,
    bool constrained) {
  const SpeculativeDecodingMode resolved_mode = mode.value_or(default_mode_);
  switch (resolved_mode) {
    case SpeculativeDecodingMode::kOff:
      return Decision{.speculate = false};
    case SpeculativeDecodingMode::kOn:
      if (!drafter_loaded) {
        ABSL_RETURN_IF_ERROR(load_drafter());
      }
      return Decision{.speculate = true};
    case SpeculativeDecodingMode::kAdaptive:
      return DecideAdaptive(drafter_loaded, load_drafter, sampler_params,
                            constrained);
  }
  return absl::InvalidArgumentError(absl::StrCat(
      "Unknown speculative decoding mode: ", static_cast<int>(resolved_mode)));
}

absl::StatusOr<SpeculativeDecodingController::Decision>
SpeculativeDecodingController::DecideAdaptive(
    bool drafter_loaded, absl::FunctionRef<absl::Status()> load_drafter,
    const std::optional<proto::SamplerParameters>& sampler_params,
    bool constrained) {
  if (!drafter_loaded) {
    // Adaptive mode is best effort: a model without a usable MTP drafter
    // decodes plainly instead of failing every step (kOn still fails).
    if (drafter_unavailable_) {
      return Decision{.speculate = false};
    }
    if (const absl::Status status = load_drafter(); !status.ok()) {
      ABSL_LOG(WARNING) << "Adaptive speculative decoding is unavailable; "
                           "falling back to plain decoding: "
                        << status;
      drafter_unavailable_ = true;
      return Decision{.speculate = false};
    }
  }
  if (constrained) {
    // The constraint state is not carried across plain and speculative steps,
    // so do not switch mid-turn under constraints.
    ABSL_LOG_FIRST_N(INFO, 1)
        << "Adaptive speculative decoding is not supported with decoding "
           "constraints; speculative decoding is always on.";
    return Decision{.speculate = true};
  }
  if (gating_policy_ == nullptr) {
    gating_policy_ = std::make_unique<MtpGatingPolicy>(gating_config_);
    ABSL_LOG(INFO) << "Adaptive speculative decoding enabled with "
                   << gating_policy_->config();
    gating_sampler_params_ = sampler_params;
  } else if (!SamplerParamsEquivalent(sampler_params, gating_sampler_params_)) {
    // The speculative estimates are only kept while the sampler configuration
    // is unchanged (e.g. across turns of the same session).
    gating_policy_->ResetEstimates();
    gating_sampler_params_ = sampler_params;
  }
  return Decision{.speculate = gating_policy_->ShouldSpeculate(),
                  .adaptive = true};
}

int64_t SpeculativeDecodingController::Observe(const Decision& decision,
                                               int tokens_emitted,
                                               absl::Duration elapsed) {
  if (!decision.adaptive || gating_policy_ == nullptr) {
    return 0;
  }
  // Counting the switches around `Observe()` only, rather than since the last
  // step, leaves out the re-enables forced by `OnNewConversation()` and
  // `ResetEstimates()`, which are not gating decisions.
  const int64_t switches_before = gating_policy_->stats().switches;
  gating_policy_->Observe({.speculated = decision.speculate,
                           .tokens_emitted = tokens_emitted,
                           .elapsed = elapsed});
  return gating_policy_->stats().switches - switches_before;
}

void SpeculativeDecodingController::OnPrefill(bool new_conversation) {
  if (gating_policy_ == nullptr) {
    return;
  }
  if (new_conversation) {
    // The policy is shared by all the conversations the executor runs, so it
    // would otherwise judge the new content by the previous conversation's
    // acceptance.
    gating_policy_->OnNewConversation();
  } else {
    gating_policy_->OnPrefill();
  }
}

}  // namespace litert::lm
