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

#include "runtime/components/constrained_decoding/repetition_penalty_constraint.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/components/constrained_decoding/constraint.h"
#include "runtime/components/constrained_decoding/logit_mask.h"
#include "runtime/components/constrained_decoding/repetition_penalty_config.h"

namespace litert::lm {

std::vector<SparseLogitMask::Entry> RepetitionPenaltyMask::ToSparseEntries(
    const std::vector<Entry>& entries) {
  std::vector<SparseLogitMask::Entry> sparse_entries;
  sparse_entries.reserve(entries.size());
  for (const auto& entry : entries) {
    // Only a penalty strictly greater than 1 rescales the logit; 1.0 or below
    // contributes nothing but its bias.
    const bool penalize = entry.repetition_penalty > 1.0f;
    sparse_entries.push_back({
        .token_id = entry.token_id,
        .weight = penalize ? 1.0f / entry.repetition_penalty : 1.0f,
        .bias = entry.bias,
        .sign_dependent_weight = penalize,
    });
  }
  return sparse_entries;
}

RepetitionPenaltyConstraint::RepetitionPenaltyConstraint(
    int vocab_size, RepetitionPenaltyConfig config)
    : vocab_size_(vocab_size), config_(std::move(config)) {}

std::unique_ptr<Constraint::State> RepetitionPenaltyConstraint::Start() const {
  auto state = std::make_unique<RepetitionPenaltyState>();
  if (config_.window_size() > 0) {
    state->token_history.resize(config_.window_size(), -1);
  }
  state->token_history_index = 0;
  return state;
}

bool RepetitionPenaltyConstraint::IsEnded(const State& state) const {
  return false;
}

absl::StatusOr<std::unique_ptr<Constraint::State>>
RepetitionPenaltyConstraint::ComputeNext(const State& state, int token) const {
  if (token < 0 || token >= vocab_size_) {
    return absl::InvalidArgumentError("Invalid token id.");
  }

  const auto& cur_state = static_cast<const RepetitionPenaltyState&>(state);
  auto next_state = std::make_unique<RepetitionPenaltyState>(cur_state);

  if (!config_.enabled()) {
    return next_state;
  }

  ++next_state->token_counts[token];

  if (config_.window_size() > 0) {
    int expired_token =
        next_state->token_history[next_state->token_history_index];
    next_state->token_history[next_state->token_history_index] = token;

    if (expired_token != -1) {
      auto it = next_state->token_counts.find(expired_token);
      if (it != next_state->token_counts.end()) {
        --it->second;
        if (it->second <= 0) {
          next_state->token_counts.erase(it);
        }
      }
    }

    next_state->token_history_index =
        (next_state->token_history_index + 1) % config_.window_size();
  }

  return next_state;
}

absl::StatusOr<std::unique_ptr<LogitMask>>
RepetitionPenaltyConstraint::ComputeMask(const State& state) const {
  const auto& cur_state = static_cast<const RepetitionPenaltyState&>(state);

  if (!config_.enabled() || cur_state.token_counts.empty()) {
    return std::make_unique<RepetitionPenaltyMask>(
        std::vector<RepetitionPenaltyMask::Entry>{});
  }

  std::vector<RepetitionPenaltyMask::Entry> entries;
  entries.reserve(cur_state.token_counts.size());

  const float rep_penalty = config_.repetition_penalty();

  for (const auto& [token_id, count] : cur_state.token_counts) {
    if (count <= 0 || token_id < 0 || token_id >= vocab_size_) {
      continue;
    }
    const float bias =
        -(config_.presence_penalty() + count * config_.frequency_penalty());
    entries.push_back({
        .token_id = token_id,
        .repetition_penalty = rep_penalty,
        .bias = bias,
    });
  }

  std::sort(entries.begin(), entries.end(),
            [](const RepetitionPenaltyMask::Entry& a,
               const RepetitionPenaltyMask::Entry& b) {
              return a.token_id < b.token_id;
            });

  return std::make_unique<RepetitionPenaltyMask>(entries);
}

}  // namespace litert::lm
