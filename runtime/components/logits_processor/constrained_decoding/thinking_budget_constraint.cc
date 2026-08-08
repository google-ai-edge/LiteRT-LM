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

#include "runtime/components/logits_processor/constrained_decoding/thinking_budget_constraint.h"

#include <memory>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/components/logits_processor/constrained_decoding/bitmap.h"
#include "runtime/components/logits_processor/constrained_decoding/constraint.h"
#include "runtime/util/status_macros.h"

namespace litert::lm {

std::unique_ptr<Constraint::State> ThinkingBudgetConstraint::Start() const {
  auto state = std::make_unique<ThinkingState>();
  state->thinking_token_count = 0;
  // If no start token IDs are provided, it means the start-of-thought token
  // was prefilled in the prompt. We bypass the start matching phase and
  // transition immediately to the active thinking phase.
  if (start_token_ids_.empty()) {
    state->in_thinking = true;
    state->matching_start_index = -1;
  } else {
    state->in_thinking = false;
    state->matching_start_index = 0;
  }
  state->natural_end_match_index = 0;
  state->forced_end_token_index = -1;
  return state;
}

bool ThinkingBudgetConstraint::IsEnded(const Constraint::State& state) const {
  const auto& s = static_cast<const ThinkingState&>(state);
  // We are not ended if we are still matching start tokens or actively
  // thinking.
  if (s.in_thinking || s.matching_start_index >= 0) {
    return false;
  }

  // We are not in thinking, delegate to the user_constraint_ if there is any.
  if (user_constraint_ != nullptr) {
    return user_constraint_->IsEnded(*s.user_state);
  }

  // Returns false to avoid restarting the thinking constraint.
  return false;
}

absl::StatusOr<std::unique_ptr<Constraint::State>>
ThinkingBudgetConstraint::ComputeNext(const Constraint::State& state,
                                      int token) const {
  const auto& s = static_cast<const ThinkingState&>(state);
  auto next_s = std::make_unique<ThinkingState>();
  next_s->thinking_token_count = s.thinking_token_count;
  next_s->in_thinking = s.in_thinking;
  next_s->forced_end_token_index = s.forced_end_token_index;
  next_s->matching_start_index = s.matching_start_index;

  // We are in content phase if we are not thinking and not matching start
  // tokens.
  const bool in_content_phase = !s.in_thinking && s.matching_start_index == -1;

  if (in_content_phase && user_constraint_ != nullptr) {
    ASSIGN_OR_RETURN(next_s->user_state,
                     user_constraint_->ComputeNext(*s.user_state, token));
  }

  int natural_end_match_index = s.natural_end_match_index;

  // Phase A: Matching start tokens
  if (next_s->matching_start_index >= 0) {
    if (ProcessStartMatching(*next_s, token)) {
      // Mismatch: skipped thinking. Transition to content phase.
      next_s->in_thinking = false;
      next_s->matching_start_index = -1;
      if (user_constraint_ != nullptr) {
        auto user_start = user_constraint_->Start();
        ASSIGN_OR_RETURN(next_s->user_state,
                         user_constraint_->ComputeNext(*user_start, token));
      }
    }
  } else if (next_s->in_thinking) {
    // Phase B: Actively thinking
    if (next_s->forced_end_token_index >= 0) {
      ProcessForcedEnd(*next_s, token);
    } else {
      next_s->thinking_token_count++;
      ProcessNaturalEnd(*next_s, natural_end_match_index, token);
      CheckBudget(*next_s);
    }
  }

  // If we just transitioned out of thinking to content phase.
  const bool transitioned_to_content = s.in_thinking && !next_s->in_thinking;
  if (transitioned_to_content && user_constraint_ != nullptr &&
      next_s->user_state == nullptr) {
    next_s->user_state = user_constraint_->Start();
  }

  next_s->natural_end_match_index = natural_end_match_index;
  return next_s;
}

absl::StatusOr<std::unique_ptr<Bitmap>> ThinkingBudgetConstraint::ComputeBitmap(
    const Constraint::State& state) const {
  const auto& s = static_cast<const ThinkingState&>(state);

  // Suspend user constraint during start matching and thinking.
  if (s.in_thinking || s.matching_start_index >= 0) {
    if (s.forced_end_token_index >= 0) {
      return std::make_unique<SingleAllowedTokenBitmap>(
          end_token_ids_[s.forced_end_token_index]);
    }
    return std::make_unique<AllAllowedBitmap>();
  }

  if (user_constraint_ != nullptr) {
    RET_CHECK(s.user_state != nullptr) << "User constraint state is null.";
    return user_constraint_->ComputeBitmap(*s.user_state);
  }

  return std::make_unique<AllAllowedBitmap>();
}

void ThinkingBudgetConstraint::ProcessForcedEnd(ThinkingState& state,
                                                int token) const {
  // Verify that the model generated the expected forced end token.
  // We only allow these tokens in ComputeBitmap, so it should match.
  if (token == end_token_ids_[state.forced_end_token_index]) {
    state.forced_end_token_index++;
    // If we have generated all forced end tokens, we are done thinking.
    if (state.forced_end_token_index >= end_token_ids_.size()) {
      state.in_thinking = false;
      state.forced_end_token_index = -1;
    }
  } else {
    // Should not happen under normal circumstances if ComputeBitmap restricts
    // vocabulary.
    state.forced_end_token_index = -1;
  }
}

bool ThinkingBudgetConstraint::ProcessStartMatching(ThinkingState& state,
                                                    int token) const {
  if (token == start_token_ids_[state.matching_start_index]) {
    state.matching_start_index++;
    // If we fully matched the start sequence, transition to thinking.
    if (state.matching_start_index >= start_token_ids_.size()) {
      state.matching_start_index = -1;
      state.in_thinking = true;
    }
  } else {
    // Mismatch: The model generated a token that is not part of the start
    // sequence.
    return true;  // Indicates thinking was skipped.
  }
  return false;
}

void ThinkingBudgetConstraint::ProcessNaturalEnd(ThinkingState& state,
                                                 int& natural_end_match_index,
                                                 int token) const {
  // Track if the model naturally generates the end-of-thinking sequence.
  if (token == end_token_ids_[natural_end_match_index]) {
    natural_end_match_index++;
    if (natural_end_match_index >= end_token_ids_.size()) {
      // Naturally reached the end of thinking.
      state.in_thinking = false;
      natural_end_match_index = 0;
    }
  } else if (token == end_token_ids_[0]) {
    // Restart matching from the first end token if we got a partial match
    // reset.
    natural_end_match_index = 1;
  } else {
    natural_end_match_index = 0;
  }
}

void ThinkingBudgetConstraint::CheckBudget(ThinkingState& state) const {
  // If we exceeded the budget, trigger the forced end sequence.
  if (budget_ >= 0 && state.in_thinking &&
      state.thinking_token_count >= budget_) {
    state.forced_end_token_index = 0;
  }
}
}  // namespace litert::lm
