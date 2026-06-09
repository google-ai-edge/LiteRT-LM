// Copyright 2025 The ODML Authors.
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

#include "runtime/components/constrained_decoding/thinking_budget_constraint.h"

#include <memory>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/components/constrained_decoding/bitmap.h"
#include "runtime/components/constrained_decoding/constraint.h"
#include "runtime/util/status_macros.h"

namespace litert::lm {

std::unique_ptr<Constraint::State> ThinkingBudgetConstraint::Start() const {
  auto state = std::make_unique<ThinkingState>();
  state->thinking_token_count = 0;
  state->in_thinking = true;
  state->natural_match_index = 0;
  state->matching_start_index = start_token_ids_.empty() ? -1 : 0;
  if (budget_ == 0) {
    state->forced_end_token_index = 0;
    if (end_token_ids_.empty()) {
      state->in_thinking = false;
      state->forced_end_token_index = -1;
    }
  } else {
    state->forced_end_token_index = -1;
  }
  if (!state->in_thinking && user_constraint_ != nullptr) {
    state->user_state = user_constraint_->Start();
  }
  return state;
}

bool ThinkingBudgetConstraint::IsEnded(const Constraint::State& state) const {
  const auto& s = static_cast<const ThinkingState&>(state);
  if (s.in_thinking) {
    return false;
  }
  if (user_constraint_ != nullptr) {
    return user_constraint_->IsEnded(*s.user_state);
  }
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

  if (!s.in_thinking && user_constraint_ != nullptr) {
    ASSIGN_OR_RETURN(next_s->user_state,
                     user_constraint_->ComputeNext(*s.user_state, token));
  }

  int natural_match_index = s.natural_match_index;

  if (next_s->in_thinking) {
    if (next_s->forced_end_token_index >= 0) {
      if (token == end_token_ids_[next_s->forced_end_token_index]) {
        next_s->forced_end_token_index++;
        if (next_s->forced_end_token_index >= end_token_ids_.size()) {
          next_s->in_thinking = false;
          next_s->forced_end_token_index = -1;
        }
      } else {
        next_s->forced_end_token_index = -1;
      }
    } else {
      if (next_s->matching_start_index >= 0 &&
          next_s->matching_start_index < start_token_ids_.size()) {
        if (token == start_token_ids_[next_s->matching_start_index]) {
          next_s->matching_start_index++;
          if (next_s->matching_start_index >= start_token_ids_.size()) {
            next_s->matching_start_index = -1;
          }
        } else {
          next_s->matching_start_index = -1;
          next_s->thinking_token_count++;
        }
      } else {
        next_s->thinking_token_count++;
      }

      if (!end_token_ids_.empty()) {
        if (token == end_token_ids_[natural_match_index]) {
          natural_match_index++;
          if (natural_match_index >= end_token_ids_.size()) {
            next_s->in_thinking = false;
            natural_match_index = 0;
          }
        } else if (token == end_token_ids_[0]) {
          natural_match_index = 1;
        } else {
          natural_match_index = 0;
        }
      }

      if (budget_ >= 0 && next_s->in_thinking &&
          next_s->thinking_token_count >= budget_) {
        next_s->forced_end_token_index = 0;
        if (end_token_ids_.empty()) {
          next_s->in_thinking = false;
          next_s->forced_end_token_index = -1;
        }
      }
    }

    if (!next_s->in_thinking && user_constraint_ != nullptr) {
      next_s->user_state = user_constraint_->Start();
    }
  }

  next_s->natural_match_index = natural_match_index;
  return next_s;
}

absl::StatusOr<std::unique_ptr<Bitmap>> ThinkingBudgetConstraint::ComputeBitmap(
    const Constraint::State& state) const {
  const auto& s = static_cast<const ThinkingState&>(state);

  if (s.in_thinking) {
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

}  // namespace litert::lm
