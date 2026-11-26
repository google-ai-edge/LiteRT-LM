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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_SPECULATIVE_UTILS_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_SPECULATIVE_UTILS_H_

// This file hosts any utilities or types used for speculative decoding that is
// shared by multiple backend.

#include <cstdint>
#include <optional>
#include <random>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "runtime/executor/gpu_artisan/ml_drift/runtime_params.h"
#include "runtime/executor/gpu_artisan/proto/sampler_params.pb.h"

namespace ml_drift {
namespace speculative_utils {

struct GreedyAcceptance {
  int num_accepted;
  int next_token;  // next_token = verifier top1 at slot num_accepted
};

GreedyAcceptance AcceptGreedyDraft(absl::Span<const int> drafted,
                                   absl::Span<const int> verifier_top1);

// Represents a token and the corresponding probability.
class TokenProbability {
 public:
  int token() const { return token_; }
  float probability() const { return probability_; }
  void set_token(int token) { token_ = token; }
  absl::Status set_probability(float probability) {
    if (!(probability >= 0.0f && probability <= 1.0f)) {
      return absl::InvalidArgumentError(
          "TokenProbability::set_probability: probability must be in [0.0, "
          "1.0]");
    }
    probability_ = probability;
    return absl::OkStatus();
  }
  static absl::StatusOr<TokenProbability> Create(int in_token,
                                                 float in_probability) {
    if (!(in_probability >= 0.0f && in_probability <= 1.0f)) {
      return absl::InvalidArgumentError(
          "TokenProbability::Create: probability must be in [0.0, 1.0]");
    }
    return TokenProbability(in_token, in_probability);
  }
  static TokenProbability CreateGreedy(int in_token) {
    return TokenProbability(in_token, 1.0);
  }
  bool operator==(const TokenProbability& other) const {
    return token_ == other.token_ && probability_ == other.probability_;
  }

 private:
  TokenProbability(int in_token, float in_probability)
      : token_(in_token), probability_(in_probability) {}
  int token_;
  float probability_;
};

class TokenAtTimeStep {
 public:
  TokenAtTimeStep(int in_time_step, int in_token)
      : time_step_(in_time_step), token_(in_token) {}
  int time_step() const { return time_step_; }
  int token() const { return token_; }
  void set_time_step(int time_step) { time_step_ = time_step; }
  void set_token(int token) { token_ = token; }
  bool operator==(const TokenAtTimeStep& other) const {
    return time_step_ == other.time_step_ && token_ == other.token_;
  }

 private:
  int time_step_;
  int token_;
};

// Retrieves the probability for the corresponding token from the given
// distribution. This runs in O(N) time whereby N is the number of distinct
// outcome in the probability distribution, or effectively, the TopK value.
// While it takes O(N) time, the TopK is usually small and for small input size
// looping usually has a smaller constant time (vs hash table).
float GetProbFromDistribution(const std::vector<TokenProbability>& dist,
                              int token);

// Runs speculative resampling algorithm from the paper.
// The paper: https://arxiv.org/pdf/2211.17192
// The main model is termed target model in the paper, while drafter model is
// termed approximation model in the paper.
absl::StatusOr<int> ResampleToken(
    const std::vector<TokenProbability>& main_model_dist,
    const std::vector<TokenProbability>& drafter_model_dist,
    std::default_random_engine* rand_gen);

struct PreparedScoreStep {
  int draft_size = 0;
  int actual_draft_size = 0;
  std::vector<int> padded_tokens;
  LlmRuntimeParams llm_params;
  // Populated when f32_channels > 0 and params changed since last score step.
  std::optional<std::vector<float>> updated_f32_params;
};

struct ScoreVerificationResult {
  int accepted_slot = 0;
  int valid_hidden_state_slot = 0;
  // Set when all drafted tokens are accepted and the bonus token at slot
  // effective_gamma must be written via OverwriteOneInputToken.
  std::optional<TokenAtTimeStep> bonus_token_to_overwrite;
  // Non-empty only for non-greedy scoring when token buffer needs drafted
  // tokens restored at [base_time_step + 1 .. base_time_step +
  // effective_gamma].
  std::vector<int32_t> non_greedy_draft_tokens_to_restore;
};

struct RejectRewindAction {
  LlmRuntimeParams rolled_back_params;
  int dirty_slots = 0;
  std::optional<int> reslice_hidden_state_slot;
};

// Backend-agnostic CPU-side speculative decoding state and verification logic
// shared across LlmWebGpu, LlmOpenCl, and LlmMetal.
class SpeculativeDecodingState {
 public:
  SpeculativeDecodingState() = default;

  // Full reset (used by RestoreContext and CleanCache(time_step <= 0)).
  void ResetAll();

  // Partial reset from time_step onwards (used by CleanCache(time_step)).
  void ResetFromTimeStep(int time_step);

  // Resolves the token at `step`, checking `token_override_` before falling
  // back to `input_tokens`. Returns std::nullopt if out of range and not
  // overridden.
  std::optional<int> GetTokenAtStep(int step,
                                    absl::Span<const int> input_tokens) const;

  // Backfills speculatively accepted tokens from `token_override_` into
  // `input_tokens` up to `input_tokens->size() + *ready_tokens - 1`, adjusting
  // `*ready_tokens` and `*num_output_tokens` (if non-null).
  absl::Status BackfillCommittedTokens(std::vector<int>* input_tokens,
                                       int* ready_tokens,
                                       int* num_output_tokens) const;

  // Records a token override at `time_step`.
  void RecordTokenOverride(int time_step, int token);

  // Prepares the padded input tokens, LlmRuntimeParams, and optional f32
  // sampler params for ScoreTokens, and records `token_override_[time_step]`
  // and `last_score_base_time_step_` / `last_score_draft_size_`.
  absl::StatusOr<PreparedScoreStep> PrepareScoreStep(
      int gamma, int seq_size, int time_step, absl::Span<const int> tokens,
      const odml::infra::proto::SamplerParameters& sampler_params,
      int f32_channels, std::default_random_engine* rand_gen);

  // Verifies scored tokens against `candidate_tokens`, populates
  // `probability_cache_` for slots [0..effective_gamma], updates telemetry and
  // `token_override_`, marks `active_hidden_state_seeded_ = true`, and returns
  // `ScoreVerificationResult`.
  absl::StatusOr<ScoreVerificationResult> VerifyScoredTokens(
      int base_time_step, absl::Span<const int> candidate_tokens,
      absl::Span<const int32_t> topk_tokens_data, int topk_tokens_channels,
      absl::Span<const float> topk_probs_data, int topk_probs_channels,
      absl::Span<const int32_t> sampled_tokens_data,
      const odml::infra::proto::SamplerParameters& sampler_params);

  // Rewinds `*ready_tokens`, `*num_output_tokens` (if non-null),
  // `token_override_`, and `probability_cache_` upon rejection at `resampled`,
  // updates non-greedy telemetry rollback, and returns `RejectRewindAction`.
  // Note: `input_tokens_size` is the pre-truncation size.
  absl::StatusOr<RejectRewindAction> RewindOnReject(
      const TokenAtTimeStep& resampled, int input_tokens_size,
      int* ready_tokens, int* num_output_tokens, int gamma,
      const odml::infra::proto::SamplerParameters& sampler_params);

  // Builds a nucleus-clipped Top-K distribution from raw tokens and probs.
  static absl::StatusOr<std::vector<TokenProbability>>
  BuildNucleusClippedDistribution(absl::Span<const int32_t> tokens,
                                  absl::Span<const float> probs, int k_limit,
                                  float p);

  // Probability cache accessors (with optional chained-draft fallback).
  absl::StatusOr<std::vector<TokenProbability>> GetProbabilityCache(
      int time_step, int chained_base_step = -1,
      absl::Span<const int> chained_drafts = {}) const;
  void SetProbabilityCache(int time_step, std::vector<TokenProbability> probs);
  void ClearProbabilityCache() { probability_cache_.clear(); }

  // Accessors and mutators for speculative state fields.
  int last_overwrite_step() const { return last_overwrite_step_; }
  void set_last_overwrite_step(int step) { last_overwrite_step_ = step; }
  bool active_hidden_state_seeded() const {
    return active_hidden_state_seeded_;
  }
  void set_active_hidden_state_seeded(bool seeded) {
    active_hidden_state_seeded_ = seeded;
  }
  int last_score_base_time_step() const { return last_score_base_time_step_; }
  int last_score_draft_size() const { return last_score_draft_size_; }
  void ClearLastScoringParamsF32() { last_scoring_params_f32_.clear(); }

  int64_t total_cycles() const { return total_cycles_; }
  int64_t total_accepted_tokens() const { return total_accepted_tokens_; }

 private:
  absl::flat_hash_map<int, int> token_override_;
  absl::flat_hash_map<int, std::vector<TokenProbability>> probability_cache_;
  int last_overwrite_step_ = -1;
  int last_score_base_time_step_ = -1;
  int last_score_draft_size_ = 0;
  bool active_hidden_state_seeded_ = false;
  std::vector<float> last_scoring_params_f32_;
  int64_t total_cycles_ = 0;
  int64_t total_accepted_tokens_ = 0;
};

}  // namespace speculative_utils
}  // namespace ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_SPECULATIVE_UTILS_H_
