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

#include "runtime/components/constrained_decoding/no_repeat_ngram_constraint.h"

#include <atomic>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/fixed_array.h"  // from @com_google_absl
#include "absl/hash/hash.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "runtime/components/constrained_decoding/constraint.h"
#include "runtime/components/constrained_decoding/logit_mask.h"
#include "runtime/components/constrained_decoding/no_repeat_ngram_config.h"

namespace litert::lm {

NoRepeatNgramConstraint::NoRepeatNgramConstraint(int vocab_size,
                                                 NoRepeatNgramConfig config)
    : vocab_size_(vocab_size),
      config_(std::move(config)),
      prefix_length_(config_.no_repeat_ngram_size() - 1) {}

std::unique_ptr<Constraint::State> NoRepeatNgramConstraint::Start() const {
  auto state = std::make_unique<NoRepeatNgramState>();
  if (config_.enabled() && config_.window_size() > 0) {
    state->token_history.resize(config_.window_size(), -1);
    if (config_.no_repeat_ngram_size() > 1) {
      state->prefix_hash_history.resize(config_.window_size(), 0);
    }
  }
  state->num_tokens_updated = 0;
  state->token_history_index = 0;
  return state;
}

bool NoRepeatNgramConstraint::IsEnded(const State& state) const {
  return false;
}

absl::StatusOr<std::unique_ptr<Constraint::State>>
NoRepeatNgramConstraint::ComputeNext(const State& state, int token) const {
  if (token < 0 || token >= vocab_size_) {
    return absl::InvalidArgumentError("Invalid token id.");
  }

  const auto& cur_state = static_cast<const NoRepeatNgramState&>(state);
  auto next_state = std::make_unique<NoRepeatNgramState>(cur_state);

  if (!config_.enabled()) {
    return next_state;
  }

  int next_token_history_index =
      config_.window_size() > 0
          ? (next_state->token_history_index + 1) % config_.window_size()
          : next_state->token_history_index + 1;

  int expired_prefix_hash_history_index = 0;
  if (config_.window_size() > 0 && config_.no_repeat_ngram_size() > 1) {
    expired_prefix_hash_history_index =
        (next_state->token_history_index + prefix_length_) %
        config_.window_size();
  }

  if (config_.window_size() > 0 &&
      next_state->num_tokens_updated >= config_.window_size()) {
    if (prefix_length_ == 0) {
      int expired_token =
          next_state->token_history[next_state->token_history_index];
      if (expired_token != -1) {
        auto it = next_state->banned_tokens.find(expired_token);
        if (it != next_state->banned_tokens.end()) {
          --it->second;
          if (it->second <= 0) {
            next_state->banned_tokens.erase(it);
          }
        }
      }
    } else {
      PrefixHash expired_prefix_hash =
          next_state->prefix_hash_history[expired_prefix_hash_history_index];
      auto it =
          next_state->prefix_hash_to_end_indices.find(expired_prefix_hash);
      if (it != next_state->prefix_hash_to_end_indices.end()) {
        it->second.erase(expired_prefix_hash_history_index);
        if (it->second.empty()) {
          next_state->prefix_hash_to_end_indices.erase(it);
        }
      }
    }
  }

  if (config_.window_size() == 0) {
    next_state->token_history.push_back(token);
  } else {
    next_state->token_history[next_state->token_history_index] = token;
  }

  if (prefix_length_ == 0) {
    next_state->banned_tokens[token]++;
  } else if (next_state->num_tokens_updated + 1 >= prefix_length_) {
    PrefixHash prefix_hash = ComputePrefixHash(*next_state);
    next_state->prefix_hash_to_end_indices[prefix_hash].insert(
        next_token_history_index);
    if (config_.window_size() == 0) {
      next_state->latest_prefix_hash = prefix_hash;
    } else {
      next_state->prefix_hash_history[next_token_history_index] = prefix_hash;
    }
  }

  ++next_state->num_tokens_updated;
  next_state->token_history_index = next_token_history_index;

  return next_state;
}

absl::StatusOr<std::unique_ptr<LogitMask>> NoRepeatNgramConstraint::ComputeMask(
    const State& state) const {
  const auto& s = static_cast<const NoRepeatNgramState&>(state);

  if (!config_.enabled() ||
      s.num_tokens_updated < config_.no_repeat_ngram_size()) {
    return BitmapLogitMask::CreateAllAllowed(vocab_size_);
  }

  if (prefix_length_ == 0) {
    std::vector<int> disallowed;
    disallowed.reserve(s.banned_tokens.size());
    for (const auto& [token_id, count] : s.banned_tokens) {
      if (count > 0 && token_id >= 0 && token_id < vocab_size_) {
        disallowed.push_back(token_id);
      }
    }
    return BitmapLogitMask::CreateFromDisallowedTokens(
        vocab_size_, absl::MakeConstSpan(disallowed));
  }

  PrefixHash prefix_hash = (config_.window_size() == 0)
                               ? s.latest_prefix_hash
                               : s.prefix_hash_history[s.token_history_index];

  auto it = s.prefix_hash_to_end_indices.find(prefix_hash);
  if (it == s.prefix_hash_to_end_indices.end()) {
    return BitmapLogitMask::CreateAllAllowed(vocab_size_);
  }

  std::vector<int> disallowed;
  for (int end_index : it->second) {
    if (end_index == s.token_history_index) {
      continue;
    }
    if (!PrefixesMatch(s, s.token_history_index, end_index)) {
      prefix_hash_collision_count_.fetch_add(1, std::memory_order_relaxed);
      continue;
    }
    if (end_index >= 0 &&
        end_index < static_cast<int>(s.token_history.size())) {
      int banned_token = s.token_history[end_index];
      if (banned_token >= 0 && banned_token < vocab_size_) {
        disallowed.push_back(banned_token);
      }
    }
  }

  return BitmapLogitMask::CreateFromDisallowedTokens(
      vocab_size_, absl::MakeConstSpan(disallowed));
}

NoRepeatNgramConstraint::PrefixHash NoRepeatNgramConstraint::ComputePrefixHash(
    const NoRepeatNgramState& state) const {
  int start_idx = (state.token_history_index + 1) - prefix_length_;
  if (start_idx < 0) {
    start_idx += config_.window_size();
  }

  if (start_idx + prefix_length_ <=
      static_cast<int>(state.token_history.size())) {
    auto span =
        absl::MakeConstSpan(&state.token_history[start_idx], prefix_length_);
    return absl::Hash<decltype(span)>()(span);
  }

  absl::FixedArray<int> prefix_buffer(prefix_length_);
  int prefix_idx = 0;
  for (int token_idx = start_idx;
       token_idx < static_cast<int>(state.token_history.size());
       ++prefix_idx, ++token_idx) {
    prefix_buffer[prefix_idx] = state.token_history[token_idx];
  }
  for (int token_idx = 0; token_idx <= state.token_history_index;
       ++prefix_idx, ++token_idx) {
    prefix_buffer[prefix_idx] = state.token_history[token_idx];
  }

  auto span = absl::MakeConstSpan(prefix_buffer);
  return absl::Hash<decltype(span)>()(span);
}

bool NoRepeatNgramConstraint::PrefixesMatch(const NoRepeatNgramState& state,
                                            int end_index1,
                                            int end_index2) const {
  int start_index1 = end_index1 - prefix_length_;
  if (start_index1 < 0) {
    start_index1 += config_.window_size();
  }

  int start_index2 = end_index2 - prefix_length_;
  if (start_index2 < 0) {
    start_index2 += config_.window_size();
  }

  if (start_index1 + prefix_length_ <=
          static_cast<int>(state.token_history.size()) &&
      start_index2 + prefix_length_ <=
          static_cast<int>(state.token_history.size())) {
    auto span1 =
        absl::MakeConstSpan(&state.token_history[start_index1], prefix_length_);
    auto span2 =
        absl::MakeConstSpan(&state.token_history[start_index2], prefix_length_);
    return span1 == span2;
  }

  for (int i = 0; i < prefix_length_; ++i) {
    int idx1 = start_index1 + i;
    if (idx1 >= config_.window_size()) {
      idx1 -= config_.window_size();
    }

    int idx2 = start_index2 + i;
    if (idx2 >= config_.window_size()) {
      idx2 -= config_.window_size();
    }

    if (state.token_history[idx1] != state.token_history[idx2]) {
      return false;
    }
  }

  return true;
}

}  // namespace litert::lm
