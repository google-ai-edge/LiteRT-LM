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
#include <cmath>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "runtime/components/constrained_decoding/constraint.h"
#include "runtime/components/constrained_decoding/logit_mask.h"
#include "runtime/components/constrained_decoding/repetition_penalty_config.h"
#include "tflite/types/half.h"  // from @litert

namespace litert::lm {

namespace {

template <typename T>
absl::Status ApplyRepetitionPenaltyImpl(
    absl::Span<const RepetitionPenaltyMask::Entry> entries,
    absl::Span<T> logits) {
  for (const auto& entry : entries) {
    if (entry.token_id < 0 ||
        entry.token_id >= static_cast<int>(logits.size())) {
      continue;
    }
    float val = static_cast<float>(logits[entry.token_id]);
    if constexpr (std::is_same_v<T, float>) {
      if ((std::isinf(val) && val < 0.0f) ||
          (val <= std::numeric_limits<float>::lowest())) {
        // Disallowed by a hard constraint, preserve -inf.
        continue;
      }
    } else {
      if (logits[entry.token_id] == tflite::half::min() ||
          (std::isinf(val) && val < 0.0f)) {
        // Disallowed by a hard constraint, preserve half::min().
        continue;
      }
    }
    if (entry.repetition_penalty > 1.0f) {
      if (val > 0.0f) {
        val /= entry.repetition_penalty;
      } else {
        val *= entry.repetition_penalty;
      }
    }
    val += entry.bias;
    logits[entry.token_id] = static_cast<T>(val);
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status RepetitionPenaltyMask::Apply(absl::Span<float> logits) const {
  return ApplyRepetitionPenaltyImpl(entries_, logits);
}

absl::Status RepetitionPenaltyMask::Apply(
    absl::Span<tflite::half> logits) const {
  return ApplyRepetitionPenaltyImpl(entries_, logits);
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

  return std::make_unique<RepetitionPenaltyMask>(std::move(entries));
}

}  // namespace litert::lm
