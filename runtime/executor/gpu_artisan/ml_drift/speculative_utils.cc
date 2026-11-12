// Copyright 2024 The MediaPipe Authors.
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

#include "runtime/executor/gpu_artisan/ml_drift/speculative_utils.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <random>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl

namespace ml_drift {
namespace speculative_utils {

GreedyAcceptance AcceptGreedyDraft(absl::Span<const int> drafted,
                                   absl::Span<const int> verifier_top1) {
  int num_accepted = 0;
  while (num_accepted < static_cast<int>(drafted.size()) &&
         num_accepted < static_cast<int>(verifier_top1.size()) &&
         drafted[num_accepted] == verifier_top1[num_accepted]) {
    ++num_accepted;
  }
  const int next_token = num_accepted < static_cast<int>(verifier_top1.size())
                             ? verifier_top1[num_accepted]
                             : -1;
  return {num_accepted, next_token};
}

float GetProbFromDistribution(const std::vector<TokenProbability>& dist,
                              int token) {
  for (const auto& tp : dist) {
    if (tp.token() == token) return tp.probability();
  }
  return 0.0f;
}

absl::StatusOr<int> ResampleToken(
    const std::vector<TokenProbability>& main_model_dist,
    const std::vector<TokenProbability>& drafter_model_dist,
    std::default_random_engine* rand_gen) {
  if (rand_gen == nullptr) {
    return absl::InvalidArgumentError("rand_gen must not be null.");
  }
  absl::flat_hash_map<int, float> draft_prob_map;
  draft_prob_map.reserve(drafter_model_dist.size());
  for (const auto& dp : drafter_model_dist) {
    draft_prob_map[dp.token()] += dp.probability();
  }

  std::vector<std::pair<int, float>> adjusted_probs;
  adjusted_probs.reserve(main_model_dist.size());
  for (const auto& tp : main_model_dist) {
    auto it = draft_prob_map.find(tp.token());
    const float q_x = (it != draft_prob_map.end()) ? it->second : 0.0f;
    const float diff = std::max(0.0f, tp.probability() - q_x);
    if (diff > 0.0f) {
      adjusted_probs.push_back({tp.token(), diff});
    }
  }

  if (adjusted_probs.empty()) {
    for (const auto& tp : main_model_dist) {
      if (tp.probability() > 0.0f) {
        adjusted_probs.push_back({tp.token(), tp.probability()});
      }
    }
    if (adjusted_probs.empty()) {
      return absl::InvalidArgumentError(
          "main_model_dist has no positive mass.");
    }
  }

  // Sort by token ID so deterministic PRNG sampling order is invariant to the
  // input ordering of `main_model_dist`.
  absl::c_sort(adjusted_probs,
               [](const auto& a, const auto& b) { return a.first < b.first; });

  float sum_adjusted = 0.0f;
  for (const auto& [token, prob] : adjusted_probs) {
    sum_adjusted += prob;
  }

  float sample =
      std::uniform_real_distribution<float>(0.0f, sum_adjusted)(*rand_gen);
  for (const auto& [token, prob] : adjusted_probs) {
    sample -= prob;
    if (sample <= 0.0f) return token;
  }
  return adjusted_probs.back().first;  // float rounding fallback
}

void SpeculativeDecodingState::ResetAll() {
  token_override_.clear();
  probability_cache_.clear();
  last_overwrite_step_ = -1;
  last_score_base_time_step_ = -1;
  last_score_draft_size_ = 0;
  active_hidden_state_seeded_ = false;
  last_scoring_params_f32_.clear();
  total_cycles_ = 0;
  total_accepted_tokens_ = 0;
}

void SpeculativeDecodingState::ResetFromTimeStep(int time_step) {
  last_overwrite_step_ = -1;
  if (time_step <= 0) {
    probability_cache_.clear();
    token_override_.clear();
    active_hidden_state_seeded_ = false;
    last_score_base_time_step_ = -1;
    last_score_draft_size_ = 0;
  } else {
    auto ge_step = [time_step](const auto& kv) {
      return kv.first >= time_step;
    };
    absl::erase_if(probability_cache_, ge_step);
    absl::erase_if(token_override_, ge_step);
    // Rolling back into the last scored window invalidates the seeded hidden
    // state.
    if (last_score_base_time_step_ >= 0 &&
        time_step <= last_score_base_time_step_ + last_score_draft_size_) {
      last_score_base_time_step_ = -1;
      last_score_draft_size_ = 0;
      active_hidden_state_seeded_ = false;
    }
  }
}

std::optional<int> SpeculativeDecodingState::GetTokenAtStep(
    int step, absl::Span<const int> input_tokens) const {
  if (step < 0) return std::nullopt;
  if (auto it = token_override_.find(step); it != token_override_.end()) {
    return it->second;
  }
  return step < static_cast<int>(input_tokens.size())
             ? std::optional<int>(input_tokens[step])
             : std::nullopt;
}

absl::Status SpeculativeDecodingState::BackfillCommittedTokens(
    std::vector<int>* input_tokens, int* ready_tokens,
    int* num_output_tokens) const {
  if (input_tokens == nullptr || ready_tokens == nullptr) {
    return absl::InvalidArgumentError(
        "input_tokens and ready_tokens must not be null.");
  }
  const int old_size = static_cast<int>(input_tokens->size());
  const int current_time_step = old_size + (*ready_tokens - 1);
  if (old_size >= current_time_step) return absl::OkStatus();

  std::vector<int> backfilled;
  backfilled.reserve(current_time_step - old_size);
  for (int s = old_size; s < current_time_step; ++s) {
    std::optional<int> tok = GetTokenAtStep(s, *input_tokens);
    if (!tok.has_value()) {
      return absl::InternalError(absl::StrCat("Missing token override at step ",
                                              s, " during backfill."));
    }
    backfilled.push_back(*tok);
  }
  const int count = static_cast<int>(backfilled.size());
  if (num_output_tokens != nullptr) {
    if (*num_output_tokens < count) {
      return absl::InternalError(absl::StrCat(
          "num_output_tokens (", *num_output_tokens,
          ") underflow when subtracting backfilled tokens (", count, ")."));
    }
    *num_output_tokens -= count;
  }
  input_tokens->insert(input_tokens->end(), backfilled.begin(),
                       backfilled.end());
  *ready_tokens -= count;
  return absl::OkStatus();
}

void SpeculativeDecodingState::RecordTokenOverride(int time_step, int token) {
  token_override_[time_step] = token;
}

absl::StatusOr<PreparedScoreStep> SpeculativeDecodingState::PrepareScoreStep(
    int gamma, int seq_size, int time_step, absl::Span<const int> tokens,
    const odml::infra::proto::SamplerParameters& sampler_params,
    int f32_channels, std::default_random_engine* rand_gen) {
  if (sampler_params.k() < 1) {
    return absl::InvalidArgumentError(
        "sampler_params.k must be >= 1 for speculative scoring.");
  }
  const float temperature = sampler_params.temperature();
  const float p = sampler_params.p();
  if (!std::isfinite(temperature) || temperature < 0.0f || !std::isfinite(p) ||
      p < 0.0f) {
    return absl::InvalidArgumentError(
        "sampler_params temperature and p must be finite and >= 0.");
  }
  const bool is_greedy =
      sampler_params.k() == 1 || temperature == 0.0f;
  if (!is_greedy && f32_channels > 1 && rand_gen == nullptr) {
    return absl::InvalidArgumentError(
        "rand_gen is required for non-greedy scoring.");
  }
  const int draft_size = seq_size > 0 ? seq_size : (gamma + 1);
  const int actual_draft_size = static_cast<int>(tokens.size());
  if (actual_draft_size < 2 || actual_draft_size > draft_size) {
    return absl::InvalidArgumentError(
        absl::StrCat("ScoreTokens requires between 2 and ", draft_size,
                     " tokens, got ", actual_draft_size));
  }
  PreparedScoreStep result{draft_size, actual_draft_size,
                           std::vector<int>(draft_size, 0)};
  absl::c_copy(tokens, result.padded_tokens.begin());
  token_override_[time_step] = tokens[0];
  last_score_base_time_step_ = time_step;
  last_score_draft_size_ = actual_draft_size;
  result.llm_params = {.token_index_offset = time_step,
                       .active_tokens = time_step + actual_draft_size,
                       .topk = sampler_params.k()};
  if (f32_channels > 0) {
    std::vector<float> params(f32_channels, 0.0f);
    if (!is_greedy) {
      std::uniform_real_distribution<float> dist(0.0f, std::min(p, 1.0f));
      for (int i = 0; i < f32_channels - 1; ++i) params[i] = dist(*rand_gen);
    }
    params.back() = is_greedy ? 1.0f : 1.0f / temperature;
    if (params != last_scoring_params_f32_) {
      last_scoring_params_f32_ = params;
      result.updated_f32_params = std::move(params);
    }
  }
  return result;
}

absl::StatusOr<std::vector<TokenProbability>>
SpeculativeDecodingState::BuildNucleusClippedDistribution(
    absl::Span<const int32_t> tokens, absl::Span<const float> probs,
    int k_limit, float p) {
  const int limit = std::min({k_limit, static_cast<int>(tokens.size()),
                              static_cast<int>(probs.size())});
  if (limit <= 0) {
    return absl::InvalidArgumentError(
        "BuildNucleusClippedDistribution requires k_limit > 0 and non-empty "
        "inputs.");
  }
  if (std::isnan(p)) {
    return absl::InvalidArgumentError("p must not be NaN.");
  }
  double total_mass = 0.0;
  for (int j = 0; j < limit; ++j) {
    if (!std::isfinite(probs[j]) || probs[j] < 0.0f) {
      return absl::InvalidArgumentError(
          absl::StrCat("Invalid probability at index ", j));
    }
    total_mass += probs[j];
  }
  const double p_eff =
      (p <= 0.0f) ? 0.0
                  : ((p < 1.0f) ? std::min(static_cast<double>(p), total_mass)
                                : total_mass);
  std::vector<TokenProbability> dist;
  dist.reserve(limit);
  double cum_prob = 0.0;
  for (int j = 0; j < limit; ++j) {
    const double raw_p = probs[j];
    const float clipped_p =
        (p_eff > 0.0)
            ? static_cast<float>(std::clamp(
                  std::min(raw_p, std::max(0.0, p_eff - cum_prob)) / p_eff, 0.0,
                  1.0))
            : (j == 0 ? 1.0f : 0.0f);
    cum_prob += raw_p;
    ABSL_ASSIGN_OR_RETURN(TokenProbability tp,
                          TokenProbability::Create(tokens[j], clipped_p));
    dist.push_back(tp);
  }
  double sum_clipped = 0.0;
  for (const auto& tp : dist) {
    sum_clipped += tp.probability();
  }
  if (sum_clipped > 0.0 && std::abs(sum_clipped - 1.0) > 1e-6) {
    for (auto& tp : dist) {
      ABSL_RETURN_IF_ERROR(tp.set_probability(static_cast<float>(
          std::clamp(tp.probability() / sum_clipped, 0.0, 1.0))));
    }
  }
  return dist;
}

absl::StatusOr<ScoreVerificationResult>
SpeculativeDecodingState::VerifyScoredTokens(
    int base_time_step, absl::Span<const int> candidate_tokens,
    absl::Span<const int32_t> topk_tokens_data, int topk_tokens_channels,
    absl::Span<const float> topk_probs_data, int topk_probs_channels,
    absl::Span<const int32_t> sampled_tokens_data,
    const odml::infra::proto::SamplerParameters& sampler_params) {
  const int actual_draft_size = static_cast<int>(candidate_tokens.size());
  if (actual_draft_size < 2) {
    return absl::InvalidArgumentError(
        "candidate_tokens must have at least 2 tokens.");
  }
  if (sampler_params.k() < 1) {
    return absl::InvalidArgumentError(
        "sampler_params.k must be >= 1 for speculative verification.");
  }
  const size_t n = static_cast<size_t>(actual_draft_size);
  if (topk_tokens_channels <= 0 ||
      topk_tokens_data.size() < n * static_cast<size_t>(topk_tokens_channels)) {
    return absl::InvalidArgumentError(
        "topk_tokens_data buffer is smaller than required.");
  }
  const int effective_gamma = actual_draft_size - 1;
  const bool is_greedy =
      sampler_params.k() == 1 || sampler_params.temperature() == 0.0f;

  std::vector<std::vector<TokenProbability>> dists;
  if (!is_greedy) {
    if (topk_probs_channels <= 0 ||
        topk_probs_data.size() <
            n * static_cast<size_t>(topk_probs_channels)) {
      return absl::InvalidArgumentError(
          "topk_probs_data buffer is smaller than required for non-greedy "
          "verification.");
    }
    if (sampled_tokens_data.size() <= static_cast<size_t>(effective_gamma)) {
      return absl::InvalidArgumentError(
          "sampled_tokens_data too small for non-greedy bonus token.");
    }
    const int k_limit = std::min({topk_tokens_channels, topk_probs_channels,
                                  static_cast<int>(sampler_params.k())});
    dists.resize(actual_draft_size);
    for (int i = 0; i < actual_draft_size; ++i) {
      ABSL_ASSIGN_OR_RETURN(
          dists[i], BuildNucleusClippedDistribution(
                        topk_tokens_data.subspan(
                            i * static_cast<size_t>(topk_tokens_channels),
                            topk_tokens_channels),
                        topk_probs_data.subspan(
                            i * static_cast<size_t>(topk_probs_channels),
                            topk_probs_channels),
                        k_limit, sampler_params.p()));
    }
  }

  std::vector<int> verifier_top1(actual_draft_size);
  for (int i = 0; i < actual_draft_size; ++i) {
    verifier_top1[i] = topk_tokens_data[i * topk_tokens_channels];
  }

  // Only the current round is ever read; drop older distributions.
  absl::erase_if(probability_cache_, [base_time_step](const auto& kv) {
    return kv.first <= base_time_step;
  });

  last_score_base_time_step_ = base_time_step;
  last_score_draft_size_ = actual_draft_size;

  ScoreVerificationResult result;
  int accepted_slot = 0;
  if (is_greedy) {
    accepted_slot =
        AcceptGreedyDraft(candidate_tokens.subspan(1), verifier_top1)
            .num_accepted;
    for (int i = 0; i < accepted_slot; ++i) {
      token_override_[base_time_step + 1 + i] = verifier_top1[i];
    }
    for (int i = 0; i < actual_draft_size; ++i) {
      probability_cache_[base_time_step + 1 + i] = {
          TokenProbability::CreateGreedy(verifier_top1[i])};
    }
  } else {
    accepted_slot = effective_gamma;
    result.non_greedy_draft_tokens_to_restore.reserve(accepted_slot);
    for (int i = 0; i < accepted_slot; ++i) {
      const int32_t draft_tok = static_cast<int32_t>(candidate_tokens[i + 1]);
      token_override_[base_time_step + 1 + i] = draft_tok;
      result.non_greedy_draft_tokens_to_restore.push_back(draft_tok);
    }
    for (int i = 0; i < actual_draft_size; ++i) {
      probability_cache_[base_time_step + 1 + i] = std::move(dists[i]);
    }
  }

  ++total_cycles_;
  total_accepted_tokens_ += accepted_slot;

  result.accepted_slot = accepted_slot;
  result.valid_hidden_state_slot =
      std::clamp(accepted_slot, 0, effective_gamma);
  if (accepted_slot == effective_gamma && effective_gamma > 0) {
    const int32_t bonus_token =
        is_greedy ? verifier_top1[effective_gamma]
                  : sampled_tokens_data[effective_gamma];
    result.bonus_token_to_overwrite =
        TokenAtTimeStep(base_time_step + effective_gamma + 1, bonus_token);
  }
  active_hidden_state_seeded_ = true;
  return result;
}

absl::StatusOr<RejectRewindAction> SpeculativeDecodingState::RewindOnReject(
    const TokenAtTimeStep& resampled, int input_tokens_size, int* ready_tokens,
    int* num_output_tokens, int gamma,
    const odml::infra::proto::SamplerParameters& sampler_params) {
  if (ready_tokens == nullptr) {
    return absl::InvalidArgumentError("ready_tokens must not be null.");
  }
  if (resampled.time_step() <= 0) {
    return absl::InvalidArgumentError(
        "Time step must be positive for RewindOnReject.");
  }
  const int initial_time_step = input_tokens_size - 1 + *ready_tokens;
  const int dirty_slots = initial_time_step - resampled.time_step();
  if (dirty_slots < 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid reject time_step ", resampled.time_step(),
                     " beyond initial_time_step ", initial_time_step));
  }
  // Caller truncates input_tokens to time_step.
  const int post_clean_size =
      std::min(input_tokens_size, resampled.time_step());
  *ready_tokens = resampled.time_step() - (post_clean_size - 1);
  if (num_output_tokens != nullptr) *num_output_tokens = *ready_tokens - 1;

  auto ge_step = [step = resampled.time_step()](const auto& kv) {
    return kv.first >= step;
  };
  absl::erase_if(probability_cache_, ge_step);
  absl::erase_if(token_override_, ge_step);
  token_override_[resampled.time_step()] = resampled.token();

  RejectRewindAction action;
  action.rolled_back_params = {.token_index_offset = resampled.time_step(),
                               .active_tokens = resampled.time_step() + 1,
                               .topk = sampler_params.k()};
  action.dirty_slots = dirty_slots;

  const bool is_greedy =
      sampler_params.k() == 1 || sampler_params.temperature() == 0.0f;
  if (!is_greedy && dirty_slots > 0) {
    const int effective_gamma =
        last_score_draft_size_ > 0 ? (last_score_draft_size_ - 1) : gamma;
    const int actual_accepted = std::clamp(
        (last_score_base_time_step_ >= 0)
            ? (resampled.time_step() - last_score_base_time_step_ - 1)
            : (effective_gamma - dirty_slots),
        0, std::max(0, effective_gamma));
    total_accepted_tokens_ = std::max<int64_t>(
        0, total_accepted_tokens_ -
               std::max(0, effective_gamma - actual_accepted));
    action.reslice_hidden_state_slot = actual_accepted;
  }
  return action;
}

absl::StatusOr<std::vector<TokenProbability>>
SpeculativeDecodingState::GetProbabilityCache(
    int time_step, int chained_base_step,
    absl::Span<const int> chained_drafts) const {
  if (auto it = probability_cache_.find(time_step);
      it != probability_cache_.end() && !it->second.empty()) {
    return it->second;
  }
  if (chained_base_step >= 0 && !chained_drafts.empty()) {
    const int draft_idx = time_step - chained_base_step - 1;
    if (draft_idx >= 0 && draft_idx < static_cast<int>(chained_drafts.size())) {
      return std::vector<TokenProbability>{
          TokenProbability::CreateGreedy(chained_drafts[draft_idx])};
    }
  }
  return absl::NotFoundError(
      absl::StrCat("Probability cache not found at time step: ", time_step));
}

void SpeculativeDecodingState::SetProbabilityCache(
    int time_step, std::vector<TokenProbability> probs) {
  probability_cache_[time_step] = std::move(probs);
}

}  // namespace speculative_utils
}  // namespace ml_drift
