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

#include "runtime/executor/gpu_artisan/ml_drift/heuristic_drafter.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "runtime/executor/gpu_artisan/ml_drift/llm.h"
#include "runtime/executor/gpu_artisan/ml_drift/speculative_utils.h"
#include "runtime/util/status_macros.h"

namespace odml::infra::gpu {

namespace {

using ml_drift::speculative_utils::TokenProbability;

absl::StatusOr<int32_t> ComputeRobustWeight(absl::Span<const int> tokens,
                                            size_t robust_match_idx,
                                            size_t robust_needle_idx) {
  RET_CHECK_GT(robust_needle_idx, robust_match_idx);

  int32_t robust_weight = 0;
  const int kRobustContextLength = 29;
  for (size_t i = 0; i < kRobustContextLength; ++i) {
    if (tokens[robust_match_idx] == tokens[robust_needle_idx]) {
      robust_weight |= (1 << (kRobustContextLength - 1 - i));
    }
    if (robust_match_idx-- == 0) break;
    --robust_needle_idx;
  }
  return robust_weight;
}

}  // namespace

HeuristicDrafter::HeuristicDrafter(int gamma) : gamma_(gamma) {}
HeuristicDrafter::~HeuristicDrafter() = default;

absl::Status HeuristicDrafter::InitInputTokens(
    const std::vector<int>& input_tokens, int current_step, bool,
    const std::atomic_bool*) {
  if (input_tokens.size() == 1 && input_tokens_.size() == current_step + 1 &&
      input_tokens_.back() == input_tokens[0]) {
    return absl::OkStatus();
  }
  if (input_tokens_.size() < current_step) {
    return absl::InternalError("current_step is too large.");
  }
  input_tokens_.resize(current_step);
  input_tokens_.insert(input_tokens_.end(), input_tokens.begin(),
                       input_tokens.end());
  return absl::OkStatus();
}

absl::StatusOr<std::vector<int>> HeuristicDrafter::DraftTokensAfter(
    const ml_drift::speculative_utils::TokenAtTimeStep& token) {
  if (token.time_step() > input_tokens_.size()) {
    return absl::InternalError("time_step past end of input.");
  }

  input_tokens_.resize(token.time_step() + 1);
  input_tokens_[token.time_step()] = token.token();
  size_t best_match_end = 0;
  size_t longest_exact_match_length = 0;
  int32_t best_robust_weight = 0;

  // We only start looking for matches at least `gamma_` tokens before the end
  // of the input sequence, since we need to suggest at least `gamma_` tokens.
  const int draft_length = gamma_ + 1;
  for (int match_end = input_tokens_.size() - draft_length; match_end > 0;
       --match_end) {
    // "match" i.e. [match_start, match_end) refers to a sequence of tokens from
    // history.
    size_t match_start = match_end - 1;
    // "needle" refers to the recent few tokens that we're trying to find
    // matches for.
    size_t needle_idx = input_tokens_.size() - 1;
    while (match_start >= 0 &&
           input_tokens_[match_start] == input_tokens_[needle_idx]) {
      --match_start;
      --needle_idx;
    }
    size_t exact_match_length = match_end - match_start - 1;
    if (exact_match_length < longest_exact_match_length) {
      // Don't bother calculating robust weight for the remaining context since
      // we've previously found a longer exact match.
      continue;
    }
    ABSL_ASSIGN_OR_RETURN(
        int32_t robust_weight,
        ComputeRobustWeight(input_tokens_, match_start, needle_idx));
    if (exact_match_length == longest_exact_match_length) {
      // During exact match length ties, compare robust weights.
      if (robust_weight > best_robust_weight) {
        best_robust_weight = robust_weight;
        best_match_end = match_end;
      }
    } else {
      // Current match is the longest exact match observed so far. This trumps
      // any robust weight difference, so just update everything.
      longest_exact_match_length = exact_match_length;
      best_robust_weight = robust_weight;
      best_match_end = match_end;
    }
  }

  std::vector<int> guesses(1 + gamma_, 0);
  guesses[0] = token.token();

  if (longest_exact_match_length > 0 || best_robust_weight > 0) {
    for (int32_t i = 1; i < guesses.size(); ++i) {
      guesses[i] = input_tokens_[i + best_match_end - 1];
    }
  } else {
    guesses.resize(1);
  }

  // Save guesses for later. If these are rejected, they'll be tossed out in
  // `RejectAfterResample()`.
  input_tokens_.insert(input_tokens_.end(), guesses.begin() + 1, guesses.end());
  return guesses;
}

absl::Status HeuristicDrafter::CleanCache(int time_step,
                                          bool skip_ringbuffer_delta_check) {
  if (time_step < input_tokens_.size()) {
    input_tokens_.resize(time_step);
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<TokenProbability>>
HeuristicDrafter::GetProbabilityCache(int time_step) {
  return std::vector<TokenProbability>{
      TokenProbability::CreateGreedy(input_tokens_[time_step])};
}

absl::Status HeuristicDrafter::RejectAfterResample(
    const ml_drift::speculative_utils::TokenAtTimeStep& token) {
  input_tokens_.resize(token.time_step() + 1);
  input_tokens_[token.time_step()] = token.token();
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<ml_drift::ContextData>>
HeuristicDrafter::SaveContext() {
  auto context_data = std::make_unique<ml_drift::ContextData>();
  context_data->tokens = input_tokens_;
  return context_data;
}

absl::StatusOr<std::unique_ptr<ml_drift::ContextData>>
HeuristicDrafter::SaveLocalContext() {
  return std::make_unique<ml_drift::ContextData>();
}

absl::Status HeuristicDrafter::RestoreContext(
    std::unique_ptr<ml_drift::ContextData> context_data) {
  input_tokens_ = std::move(context_data->tokens);
  return absl::OkStatus();
}

absl::Status HeuristicDrafter::RestoreLocalContext(
    std::unique_ptr<ml_drift::ContextData> context_data) {
  return absl::OkStatus();
}

absl::Status HeuristicDrafter::RestoreContextWithCacheCleaning(
    const ml_drift::ContextData& context_data) {
  input_tokens_ = context_data.tokens;
  return absl::OkStatus();
}

}  // namespace odml::infra::gpu
