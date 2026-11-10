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

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/executor/gpu_artisan/proto/sampler_params.pb.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace ml_drift {
namespace speculative_utils {
namespace {

using ::odml::infra::proto::SamplerParameters;
using ::testing::ElementsAre;
using ::testing::FloatNear;
using ::testing::status::IsOkAndHolds;
using ::testing::status::StatusIs;

TEST(SpeculativeUtilsTest, GetProbFromDistribution) {
  ASSERT_OK_AND_ASSIGN(TokenProbability p1, TokenProbability::Create(1, 0.1));
  ASSERT_OK_AND_ASSIGN(TokenProbability p2, TokenProbability::Create(2, 0.2));
  ASSERT_OK_AND_ASSIGN(TokenProbability p3, TokenProbability::Create(3, 0.3));
  ASSERT_OK_AND_ASSIGN(TokenProbability p4, TokenProbability::Create(4, 0.4));
  std::vector<TokenProbability> probs = {p1, p2, p3, p4};
  EXPECT_FLOAT_EQ(GetProbFromDistribution(probs, 1), 0.1);
  EXPECT_FLOAT_EQ(GetProbFromDistribution(probs, 2), 0.2);
  EXPECT_FLOAT_EQ(GetProbFromDistribution(probs, 3), 0.3);
  EXPECT_FLOAT_EQ(GetProbFromDistribution(probs, 4), 0.4);
  EXPECT_FLOAT_EQ(GetProbFromDistribution(probs, 5), 0.0);
}

TEST(SpeculativeUtilsTest, TokenProbabilityRejectsNaN) {
  EXPECT_THAT(TokenProbability::Create(1, NAN),
              StatusIs(absl::StatusCode::kInvalidArgument));
  ASSERT_OK_AND_ASSIGN(TokenProbability tp, TokenProbability::Create(1, 0.5f));
  EXPECT_THAT(tp.set_probability(NAN),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(SpeculativeUtilsTest, ResampleToken1) {
  ASSERT_OK_AND_ASSIGN(TokenProbability tp1, TokenProbability::Create(1, 0.1));
  ASSERT_OK_AND_ASSIGN(TokenProbability tp2, TokenProbability::Create(2, 0.2));
  ASSERT_OK_AND_ASSIGN(TokenProbability tp3, TokenProbability::Create(3, 0.3));
  ASSERT_OK_AND_ASSIGN(TokenProbability tp4, TokenProbability::Create(4, 0.4));
  std::vector<TokenProbability> verified_probs = {tp1, tp2, tp3, tp4};

  ASSERT_OK_AND_ASSIGN(TokenProbability tp5, TokenProbability::Create(1, 0.2));
  ASSERT_OK_AND_ASSIGN(TokenProbability tp6, TokenProbability::Create(2, 0.25));
  ASSERT_OK_AND_ASSIGN(TokenProbability tp7, TokenProbability::Create(3, 0.2));
  ASSERT_OK_AND_ASSIGN(TokenProbability tp8, TokenProbability::Create(5, 0.35));
  std::vector<TokenProbability> draft_probs = {tp5, tp6, tp7, tp8};

  std::default_random_engine rand = std::default_random_engine(42);
  EXPECT_THAT(ResampleToken(verified_probs, draft_probs, &rand),
              IsOkAndHolds(3));

  // Use a different random to verify that another possibility is possible.
  rand = std::default_random_engine(4);
  rand.discard(5);
  EXPECT_THAT(ResampleToken(verified_probs, draft_probs, &rand),
              IsOkAndHolds(4));
}

TEST(SpeculativeUtilsTest, ResampleTokenInvariantToInputOrder) {
  ASSERT_OK_AND_ASSIGN(TokenProbability tp1, TokenProbability::Create(1, 0.1f));
  ASSERT_OK_AND_ASSIGN(TokenProbability tp2, TokenProbability::Create(2, 0.2f));
  ASSERT_OK_AND_ASSIGN(TokenProbability tp3, TokenProbability::Create(3, 0.3f));
  ASSERT_OK_AND_ASSIGN(TokenProbability tp4, TokenProbability::Create(4, 0.4f));
  std::vector<TokenProbability> verified_probs = {tp1, tp2, tp3, tp4};
  std::vector<TokenProbability> verified_probs_rev = {tp4, tp3, tp2, tp1};

  ASSERT_OK_AND_ASSIGN(TokenProbability tp5, TokenProbability::Create(1, 0.2f));
  ASSERT_OK_AND_ASSIGN(TokenProbability tp6,
                       TokenProbability::Create(2, 0.25f));
  ASSERT_OK_AND_ASSIGN(TokenProbability tp7, TokenProbability::Create(3, 0.2f));
  ASSERT_OK_AND_ASSIGN(TokenProbability tp8,
                       TokenProbability::Create(5, 0.35f));
  std::vector<TokenProbability> draft_probs = {tp5, tp6, tp7, tp8};

  std::default_random_engine rand1(42);
  std::default_random_engine rand2(42);
  ASSERT_OK_AND_ASSIGN(int res1,
                       ResampleToken(verified_probs, draft_probs, &rand1));
  ASSERT_OK_AND_ASSIGN(int res2,
                       ResampleToken(verified_probs_rev, draft_probs, &rand2));
  EXPECT_EQ(res1, res2);
}

TEST(SpeculativeUtilsTest, ResampleTokenSumsDuplicateDrafterTokens) {
  ASSERT_OK_AND_ASSIGN(TokenProbability m0, TokenProbability::Create(0, 0.5f));
  ASSERT_OK_AND_ASSIGN(TokenProbability m1, TokenProbability::Create(1, 0.5f));
  std::vector<TokenProbability> main_dist = {m0, m1};

  ASSERT_OK_AND_ASSIGN(TokenProbability d0_a,
                       TokenProbability::Create(0, 0.3f));
  ASSERT_OK_AND_ASSIGN(TokenProbability d0_b,
                       TokenProbability::Create(0, 0.2f));
  ASSERT_OK_AND_ASSIGN(TokenProbability d0_c,
                       TokenProbability::Create(0, 0.0f));
  std::vector<TokenProbability> drafter_dist = {d0_a, d0_b, d0_c};

  std::default_random_engine rng(99);
  EXPECT_THAT(ResampleToken(main_dist, drafter_dist, &rng), IsOkAndHolds(1));
  EXPECT_THAT(ResampleToken(main_dist, drafter_dist, nullptr),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(SpeculativeUtilsTest, AcceptGreedyDraftAllAccepted) {
  std::vector<int> drafted = {10, 20, 30};
  std::vector<int> verifier = {10, 20, 30, 40};
  GreedyAcceptance result = AcceptGreedyDraft(drafted, verifier);
  EXPECT_EQ(result.num_accepted, 3);
  EXPECT_EQ(result.next_token, 40);
}

TEST(SpeculativeUtilsTest, AcceptGreedyDraftNoneAccepted) {
  std::vector<int> drafted = {10, 20, 30};
  std::vector<int> verifier = {99, 20, 30, 40};
  GreedyAcceptance result = AcceptGreedyDraft(drafted, verifier);
  EXPECT_EQ(result.num_accepted, 0);
  EXPECT_EQ(result.next_token, 99);
}

TEST(SpeculativeUtilsTest, AcceptGreedyDraftMismatchAtSlotK) {
  std::vector<int> drafted = {10, 20, 30};
  std::vector<int> verifier = {10, 99, 30, 40};
  GreedyAcceptance result = AcceptGreedyDraft(drafted, verifier);
  EXPECT_EQ(result.num_accepted, 1);
  EXPECT_EQ(result.next_token, 99);

  std::vector<int> verifier_k2 = {10, 20, 99, 40};
  GreedyAcceptance result_k2 = AcceptGreedyDraft(drafted, verifier_k2);
  EXPECT_EQ(result_k2.num_accepted, 2);
  EXPECT_EQ(result_k2.next_token, 99);
}

TEST(SpeculativeUtilsTest, AcceptGreedyDraftGamma1) {
  std::vector<int> drafted = {10};
  std::vector<int> verifier_accepted = {10, 20};
  GreedyAcceptance result1 = AcceptGreedyDraft(drafted, verifier_accepted);
  EXPECT_EQ(result1.num_accepted, 1);
  EXPECT_EQ(result1.next_token, 20);

  std::vector<int> verifier_rejected = {99, 20};
  GreedyAcceptance result2 = AcceptGreedyDraft(drafted, verifier_rejected);
  EXPECT_EQ(result2.num_accepted, 0);
  EXPECT_EQ(result2.next_token, 99);
}

TEST(SpeculativeUtilsTest, AcceptGreedyDraftNegativeAndExtremeIntIds) {
  std::vector<int> drafted = {-1, 0, 262143};
  std::vector<int> verifier_acc = {-1, 0, 262143, -100};
  GreedyAcceptance r_acc = AcceptGreedyDraft(drafted, verifier_acc);
  EXPECT_EQ(r_acc.num_accepted, 3);
  EXPECT_EQ(r_acc.next_token, -100);

  std::vector<int> verifier_rej = {-1, -2, 262143, -100};
  GreedyAcceptance r_rej = AcceptGreedyDraft(drafted, verifier_rej);
  EXPECT_EQ(r_rej.num_accepted, 1);
  EXPECT_EQ(r_rej.next_token, -2);
}

TEST(SpeculativeUtilsTest,
     AcceptGreedyDraftMismatchedSpanLengthsDraftedLongerThanVerifier) {
  std::vector<int> drafted = {10, 20, 30};
  GreedyAcceptance r_short = AcceptGreedyDraft(drafted, {10, 20});
  EXPECT_EQ(r_short.num_accepted, 2);
  EXPECT_EQ(r_short.next_token, -1);

  GreedyAcceptance r_eq = AcceptGreedyDraft(drafted, {10, 20, 30});
  EXPECT_EQ(r_eq.num_accepted, 3);
  EXPECT_EQ(r_eq.next_token, -1);
}

TEST(SpeculativeDecodingStateTest, GetTokenAtStepAndRecordOverride) {
  SpeculativeDecodingState state;
  std::vector<int> input_tokens = {100, 200, 300};

  EXPECT_EQ(state.GetTokenAtStep(-1, input_tokens), std::nullopt);
  EXPECT_EQ(state.GetTokenAtStep(0, input_tokens), 100);
  EXPECT_EQ(state.GetTokenAtStep(2, input_tokens), 300);
  EXPECT_EQ(state.GetTokenAtStep(3, input_tokens), std::nullopt);
  EXPECT_EQ(state.last_overwrite_step(), -1);

  state.RecordTokenOverride(2, 999);
  EXPECT_EQ(state.GetTokenAtStep(2, input_tokens), 999);
  EXPECT_EQ(state.last_overwrite_step(), -1);

  state.RecordTokenOverride(5, 555);
  state.set_last_overwrite_step(5);
  EXPECT_EQ(state.GetTokenAtStep(5, input_tokens), 555);
  EXPECT_EQ(state.last_overwrite_step(), 5);
}

TEST(SpeculativeDecodingStateTest, ResetFromTimeStepAndResetAll) {
  SpeculativeDecodingState state;
  for (int step : {2, 3, 4, 5}) {
    state.RecordTokenOverride(step, step * 10);
    state.set_last_overwrite_step(step);
    state.SetProbabilityCache(step,
                              {TokenProbability::CreateGreedy(step * 10)});
  }
  state.set_active_hidden_state_seeded(true);

  state.ResetFromTimeStep(4);
  EXPECT_EQ(state.last_overwrite_step(), -1);
  EXPECT_TRUE(state.active_hidden_state_seeded());
  EXPECT_EQ(state.GetTokenAtStep(2, {}), 20);
  EXPECT_EQ(state.GetTokenAtStep(3, {}), 30);
  EXPECT_EQ(state.GetTokenAtStep(4, {}), std::nullopt);
  EXPECT_EQ(state.GetTokenAtStep(5, {}), std::nullopt);
  EXPECT_OK(state.GetProbabilityCache(3));
  EXPECT_THAT(state.GetProbabilityCache(4),
              StatusIs(absl::StatusCode::kNotFound));

  state.ResetFromTimeStep(0);
  EXPECT_FALSE(state.active_hidden_state_seeded());
  EXPECT_EQ(state.GetTokenAtStep(2, {}), std::nullopt);
  EXPECT_THAT(state.GetProbabilityCache(2),
              StatusIs(absl::StatusCode::kNotFound));

  // VerifyScoredTokens at base 5 with 2 candidates: scored window is (5, 7].
  // ResetFromTimeStep clears last_score_base_time_step_ when
  // time_step <= base + draft_size.
  SamplerParameters greedy_params;
  greedy_params.set_k(1);
  greedy_params.set_temperature(0.0f);
  std::vector<int32_t> topk_tok = {20, 30};
  ASSERT_OK(state.VerifyScoredTokens(/*base_time_step=*/5, {10, 20}, topk_tok,
                                     /*topk_tokens_channels=*/1, {}, 0, {},
                                     greedy_params));
  EXPECT_EQ(state.last_score_base_time_step(), 5);
  EXPECT_TRUE(state.active_hidden_state_seeded());
  EXPECT_GT(state.total_cycles(), 0);

  state.ResetFromTimeStep(4);
  EXPECT_EQ(state.last_score_base_time_step(), -1);
  EXPECT_FALSE(state.active_hidden_state_seeded());

  // Inside-window rollback at step 7:
  ASSERT_OK(state.VerifyScoredTokens(/*base_time_step=*/5, {10, 20}, topk_tok,
                                     /*topk_tokens_channels=*/1, {}, 0, {},
                                     greedy_params));
  EXPECT_EQ(state.last_score_base_time_step(), 5);
  EXPECT_TRUE(state.active_hidden_state_seeded());
  state.ResetFromTimeStep(7);
  EXPECT_EQ(state.last_score_base_time_step(), -1);
  EXPECT_FALSE(state.active_hidden_state_seeded());

  // Outside-window rollback at step 8:
  ASSERT_OK(state.VerifyScoredTokens(/*base_time_step=*/5, {10, 20}, topk_tok,
                                     /*topk_tokens_channels=*/1, {}, 0, {},
                                     greedy_params));
  EXPECT_EQ(state.last_score_base_time_step(), 5);
  EXPECT_TRUE(state.active_hidden_state_seeded());
  state.ResetFromTimeStep(8);
  EXPECT_EQ(state.last_score_base_time_step(), 5);
  EXPECT_TRUE(state.active_hidden_state_seeded());

  state.RecordTokenOverride(1, 11);
  state.set_last_overwrite_step(1);
  state.set_active_hidden_state_seeded(true);
  state.ResetAll();
  EXPECT_FALSE(state.active_hidden_state_seeded());
  EXPECT_EQ(state.last_overwrite_step(), -1);
  EXPECT_EQ(state.GetTokenAtStep(1, {}), std::nullopt);
  EXPECT_EQ(state.total_cycles(), 0);
  EXPECT_EQ(state.total_accepted_tokens(), 0);
}

TEST(SpeculativeDecodingStateTest, BackfillCommittedTokens) {
  SpeculativeDecodingState state;
  state.RecordTokenOverride(2, 30);
  state.RecordTokenOverride(3, 40);

  std::vector<int> input_tokens = {10, 20};
  int ready_tokens = 3;
  int num_output_tokens = 5;

  ASSERT_OK(state.BackfillCommittedTokens(&input_tokens, &ready_tokens,
                                          &num_output_tokens));
  EXPECT_THAT(input_tokens, ElementsAre(10, 20, 30, 40));
  EXPECT_EQ(ready_tokens, 1);
  EXPECT_EQ(num_output_tokens, 3);

  // Also test nullable num_output_tokens (OpenCL path).
  state.RecordTokenOverride(4, 50);
  ready_tokens = 2;
  ASSERT_OK(
      state.BackfillCommittedTokens(&input_tokens, &ready_tokens, nullptr));
  EXPECT_THAT(input_tokens, ElementsAre(10, 20, 30, 40, 50));
  EXPECT_EQ(ready_tokens, 1);

  // Missing override returns InternalError.
  ready_tokens = 2;
  EXPECT_THAT(
      state.BackfillCommittedTokens(&input_tokens, &ready_tokens, nullptr),
      StatusIs(absl::StatusCode::kInternal));
}

TEST(SpeculativeDecodingStateTest, PrepareScoreStepGreedyAndNonGreedy) {
  SpeculativeDecodingState state;
  SamplerParameters greedy_params;
  greedy_params.set_k(1);
  greedy_params.set_temperature(0.0f);
  greedy_params.set_p(1.0f);

  // Too few or too many tokens should fail.
  EXPECT_THAT(state.PrepareScoreStep(/*gamma=*/2, /*seq_size=*/3,
                                     /*time_step=*/4, {10}, greedy_params,
                                     /*f32_channels=*/4, nullptr),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(
      state.PrepareScoreStep(/*gamma=*/2, /*seq_size=*/3,
                             /*time_step=*/4, {10, 20, 30, 40}, greedy_params,
                             /*f32_channels=*/4, nullptr),
      StatusIs(absl::StatusCode::kInvalidArgument));

  // k < 1 must fail fast with InvalidArgument.
  SamplerParameters bad_k_params;
  bad_k_params.set_k(0);
  bad_k_params.set_temperature(0.5f);
  EXPECT_THAT(state.PrepareScoreStep(/*gamma=*/2, /*seq_size=*/3,
                                     /*time_step=*/4, {10, 20}, bad_k_params,
                                     /*f32_channels=*/4, nullptr),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // NaN temperature or p must fail fast.
  SamplerParameters nan_params;
  nan_params.set_k(2);
  nan_params.set_temperature(NAN);
  EXPECT_THAT(state.PrepareScoreStep(/*gamma=*/2, /*seq_size=*/3,
                                     /*time_step=*/4, {10, 20}, nan_params,
                                     /*f32_channels=*/4, nullptr),
              StatusIs(absl::StatusCode::kInvalidArgument));
  nan_params.set_temperature(0.5f);
  nan_params.set_p(NAN);
  EXPECT_THAT(state.PrepareScoreStep(/*gamma=*/2, /*seq_size=*/3,
                                     /*time_step=*/4, {10, 20}, nan_params,
                                     /*f32_channels=*/4, nullptr),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // Non-greedy scoring without rand_gen must return InvalidArgument when
  // f32_channels > 1.
  SamplerParameters non_greedy_null_rng;
  non_greedy_null_rng.set_k(2);
  non_greedy_null_rng.set_temperature(0.5f);
  EXPECT_THAT(state.PrepareScoreStep(/*gamma=*/2, /*seq_size=*/3,
                                     /*time_step=*/4, {10, 20},
                                     non_greedy_null_rng,
                                     /*f32_channels=*/4, nullptr),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // Partial draft length (2 tokens for draft_size = 3) should pad with 0.
  ASSERT_OK_AND_ASSIGN(
      const PreparedScoreStep prep1,
      state.PrepareScoreStep(/*gamma=*/2, /*seq_size=*/3, /*time_step=*/4,
                             {10, 20}, greedy_params, /*f32_channels=*/4,
                             nullptr));
  EXPECT_EQ(prep1.draft_size, 3);
  EXPECT_EQ(prep1.actual_draft_size, 2);
  EXPECT_THAT(prep1.padded_tokens, ElementsAre(10, 20, 0));
  EXPECT_EQ(prep1.llm_params.token_index_offset, 4);
  EXPECT_EQ(prep1.llm_params.active_tokens, 6);
  EXPECT_EQ(prep1.llm_params.topk, 1);
  ASSERT_TRUE(prep1.updated_f32_params.has_value());
  EXPECT_THAT(*prep1.updated_f32_params, ElementsAre(0.0f, 0.0f, 0.0f, 1.0f));
  EXPECT_EQ(state.GetTokenAtStep(4, {}), 10);
  EXPECT_EQ(state.last_score_base_time_step(), 4);
  EXPECT_EQ(state.last_score_draft_size(), 2);

  // Identical second call should not re-emit updated_f32_params.
  ASSERT_OK_AND_ASSIGN(
      const PreparedScoreStep prep2,
      state.PrepareScoreStep(/*gamma=*/2, /*seq_size=*/3, /*time_step=*/5,
                             {20, 30, 40}, greedy_params, /*f32_channels=*/4,
                             nullptr));
  EXPECT_FALSE(prep2.updated_f32_params.has_value());

  // After ClearLastScoringParamsF32, updated_f32_params is populated again.
  state.ClearLastScoringParamsF32();
  ASSERT_OK_AND_ASSIGN(
      const PreparedScoreStep prep3,
      state.PrepareScoreStep(/*gamma=*/2, /*seq_size=*/3, /*time_step=*/5,
                             {20, 30, 40}, greedy_params, /*f32_channels=*/4,
                             nullptr));
  EXPECT_TRUE(prep3.updated_f32_params.has_value());

  // Non-greedy with RNG populates uniform samples in [0, p) and 1/temperature.
  SamplerParameters non_greedy_params;
  non_greedy_params.set_k(4);
  non_greedy_params.set_temperature(0.5f);
  non_greedy_params.set_p(0.8f);
  std::default_random_engine rng(123);
  ASSERT_OK_AND_ASSIGN(
      const PreparedScoreStep prep_ng,
      state.PrepareScoreStep(/*gamma=*/2, /*seq_size=*/3, /*time_step=*/6,
                             {20, 30, 40}, non_greedy_params,
                             /*f32_channels=*/4, &rng));
  ASSERT_TRUE(prep_ng.updated_f32_params.has_value());
  ASSERT_EQ(prep_ng.updated_f32_params->size(), 4u);
  for (int i = 0; i < 3; ++i) {
    EXPECT_GT((*prep_ng.updated_f32_params)[i], 0.0f);
    EXPECT_LT((*prep_ng.updated_f32_params)[i], 0.8f);
  }
  EXPECT_FLOAT_EQ(prep_ng.updated_f32_params->back(), 2.0f);
}

TEST(SpeculativeDecodingStateTest, VerifyScoredTokensGreedyAllAndPartial) {
  SpeculativeDecodingState state;
  SamplerParameters greedy_params;
  greedy_params.set_k(1);
  greedy_params.set_temperature(0.0f);

  // Cycle 1: All 2 drafted tokens accepted; bonus token is 99.
  std::vector<int> candidates = {10, 20, 30};
  std::vector<int32_t> topk_tokens = {20, 0, 30, 0, 99, 0};  // 2 channels/slot
  ASSERT_OK_AND_ASSIGN(
      ScoreVerificationResult res1,
      state.VerifyScoredTokens(/*base_time_step=*/5, candidates,
                               topk_tokens, /*topk_tokens_channels=*/2,
                               /*topk_probs_data=*/{},
                               /*topk_probs_channels=*/0,
                               /*sampled_tokens_data=*/{}, greedy_params));
  EXPECT_EQ(res1.accepted_slot, 2);
  EXPECT_EQ(res1.valid_hidden_state_slot, 2);
  ASSERT_TRUE(res1.bonus_token_to_overwrite.has_value());
  EXPECT_EQ(*res1.bonus_token_to_overwrite, TokenAtTimeStep(8, 99));
  EXPECT_TRUE(res1.non_greedy_draft_tokens_to_restore.empty());
  EXPECT_TRUE(state.active_hidden_state_seeded());
  EXPECT_EQ(state.GetTokenAtStep(6, {}), 20);
  EXPECT_EQ(state.GetTokenAtStep(7, {}), 30);
  EXPECT_THAT(state.GetProbabilityCache(6),
              IsOkAndHolds(ElementsAre(TokenProbability::CreateGreedy(20))));
  EXPECT_THAT(state.GetProbabilityCache(7),
              IsOkAndHolds(ElementsAre(TokenProbability::CreateGreedy(30))));
  EXPECT_THAT(state.GetProbabilityCache(8),
              IsOkAndHolds(ElementsAre(TokenProbability::CreateGreedy(99))));
  EXPECT_EQ(state.total_cycles(), 1);
  EXPECT_EQ(state.total_accepted_tokens(), 2);

  // Cycle 2: Only 1st drafted token accepted (2nd mismatches: 30 vs 77).
  std::vector<int32_t> topk_tokens_partial = {20, 0, 77, 0, 88, 0};
  ASSERT_OK_AND_ASSIGN(
      ScoreVerificationResult res2,
      state.VerifyScoredTokens(/*base_time_step=*/8, candidates,
                               topk_tokens_partial, /*topk_tokens_channels=*/2,
                               /*topk_probs_data=*/{},
                               /*topk_probs_channels=*/0,
                               /*sampled_tokens_data=*/{}, greedy_params));
  EXPECT_EQ(res2.accepted_slot, 1);
  EXPECT_EQ(res2.valid_hidden_state_slot, 1);
  EXPECT_FALSE(res2.bonus_token_to_overwrite.has_value());
  EXPECT_EQ(state.GetTokenAtStep(9, {}), 20);
  EXPECT_EQ(state.GetTokenAtStep(10, {}), std::nullopt);
  EXPECT_EQ(state.total_cycles(), 2);
  EXPECT_EQ(state.total_accepted_tokens(), 3);
}

TEST(SpeculativeDecodingStateTest,
     VerifyScoredTokensNonGreedyAndRewindOnReject) {
  SpeculativeDecodingState state;
  SamplerParameters non_greedy_params;
  non_greedy_params.set_k(2);
  non_greedy_params.set_temperature(1.0f);
  non_greedy_params.set_p(0.75f);

  std::vector<int> candidates = {10, 20, 30};
  // 3 slots x 2 channels
  std::vector<int32_t> topk_tokens = {20, 21, 30, 31, 40, 41};
  std::vector<float> topk_probs = {0.6f, 0.3f, 0.5f, 0.25f, 0.8f, 0.1f};
  std::vector<int32_t> sampled_tokens = {20, 31, 41};

  ASSERT_OK_AND_ASSIGN(
      ScoreVerificationResult res,
      state.VerifyScoredTokens(/*base_time_step=*/4, candidates,
                               topk_tokens, /*topk_tokens_channels=*/2,
                               topk_probs, /*topk_probs_channels=*/2,
                               sampled_tokens, non_greedy_params));
  EXPECT_EQ(res.accepted_slot, 2);
  EXPECT_EQ(res.valid_hidden_state_slot, 2);
  EXPECT_THAT(res.non_greedy_draft_tokens_to_restore, ElementsAre(20, 30));
  ASSERT_TRUE(res.bonus_token_to_overwrite.has_value());
  EXPECT_EQ(*res.bonus_token_to_overwrite, TokenAtTimeStep(7, 41));
  EXPECT_EQ(state.total_cycles(), 1);
  EXPECT_EQ(state.total_accepted_tokens(), 2);

  // Slot 0 (step 5): raw probs {0.6, 0.3}, p_eff = min(0.75, 0.9) = 0.75.
  // Token 20: 0.6 / 0.75 = 0.8; Token 21: (0.75 - 0.6) / 0.75 = 0.2.
  ASSERT_OK_AND_ASSIGN(std::vector<TokenProbability> probs5,
                       state.GetProbabilityCache(5));
  ASSERT_EQ(probs5.size(), 2u);
  EXPECT_EQ(probs5[0].token(), 20);
  EXPECT_THAT(probs5[0].probability(), FloatNear(0.8f, 1e-5f));
  EXPECT_EQ(probs5[1].token(), 21);
  EXPECT_THAT(probs5[1].probability(), FloatNear(0.2f, 1e-5f));

  // Simulate ScoreTokens advancing ready_tokens by actual_draft_size (3).
  // Prompt had 5 tokens (steps 0..4), ready_tokens was 1 -> now 4,
  // num_output_tokens was 0 -> now 3. initial_time_step = 5 - 1 + 4 = 8.
  // Suppose token at step 5 (20) was accepted, and token at step 6 (30) was
  // rejected and resampled to 31.
  int ready_tokens = 4;
  int num_output_tokens = 3;
  ASSERT_OK_AND_ASSIGN(
      RejectRewindAction rewind,
      state.RewindOnReject(TokenAtTimeStep(6, 31), /*input_tokens_size=*/5,
                           &ready_tokens, &num_output_tokens, /*gamma=*/2,
                           non_greedy_params));
  // initial_time_step = 8, resampled.time_step = 6 -> dirty_slots = 2.
  EXPECT_EQ(rewind.dirty_slots, 2);
  EXPECT_EQ(ready_tokens, 2);
  EXPECT_EQ(num_output_tokens, 1);
  EXPECT_EQ(rewind.rolled_back_params.token_index_offset, 6);
  EXPECT_EQ(rewind.rolled_back_params.active_tokens, 7);
  EXPECT_EQ(rewind.rolled_back_params.topk, 2);
  // actual_accepted = resampled.time_step(6) - base_time_step(4) - 1 = 1.
  ASSERT_TRUE(rewind.reslice_hidden_state_slot.has_value());
  EXPECT_EQ(*rewind.reslice_hidden_state_slot, 1);

  // Telemetry rolled back from 2 accepted to 1 accepted.
  EXPECT_EQ(state.total_cycles(), 1);
  EXPECT_EQ(state.total_accepted_tokens(), 1);

  // Overrides and probability cache pruned from step 6 onwards, step 6 set to
  // resampled token 31.
  EXPECT_EQ(state.GetTokenAtStep(5, {}), 20);
  EXPECT_EQ(state.GetTokenAtStep(6, {}), 31);
  EXPECT_EQ(state.GetTokenAtStep(7, {}), std::nullopt);
  EXPECT_OK(state.GetProbabilityCache(5));
  EXPECT_THAT(state.GetProbabilityCache(6),
              StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(state.GetProbabilityCache(7),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST(SpeculativeDecodingStateTest, VerifyScoredTokensGreedyFirstReject) {
  SpeculativeDecodingState state;
  SamplerParameters greedy_params;
  greedy_params.set_k(1);
  greedy_params.set_temperature(0.0f);

  // First drafted token (20) mismatches verifier top-1 at slot 0 (77).
  std::vector<int> candidates = {10, 20, 30};
  std::vector<int32_t> topk_tokens = {77, 0, 30, 0, 99, 0};
  ASSERT_OK_AND_ASSIGN(
      ScoreVerificationResult res,
      state.VerifyScoredTokens(/*base_time_step=*/5, candidates,
                               topk_tokens, /*topk_tokens_channels=*/2,
                               /*topk_probs_data=*/{},
                               /*topk_probs_channels=*/0,
                               /*sampled_tokens_data=*/{}, greedy_params));
  EXPECT_EQ(res.accepted_slot, 0);
  EXPECT_EQ(res.valid_hidden_state_slot, 0);
  EXPECT_FALSE(res.bonus_token_to_overwrite.has_value());
  EXPECT_TRUE(res.non_greedy_draft_tokens_to_restore.empty());
  EXPECT_EQ(state.GetTokenAtStep(6, {}), std::nullopt);
  EXPECT_EQ(state.GetTokenAtStep(7, {}), std::nullopt);
  EXPECT_THAT(state.GetProbabilityCache(6),
              IsOkAndHolds(ElementsAre(TokenProbability::CreateGreedy(77))));
  EXPECT_EQ(state.total_cycles(), 1);
  EXPECT_EQ(state.total_accepted_tokens(), 0);

  // Greedy RewindOnReject at step 6 sets override to 77 without reslicing or
  // double-decrementing greedy telemetry counters.
  int ready_tokens = 4;
  int num_output_tokens = 3;
  ASSERT_OK_AND_ASSIGN(
      RejectRewindAction rewind,
      state.RewindOnReject(TokenAtTimeStep(6, 77), /*input_tokens_size=*/3,
                           &ready_tokens, &num_output_tokens, /*gamma=*/2,
                           greedy_params));
  EXPECT_EQ(rewind.dirty_slots, 0);
  EXPECT_FALSE(rewind.reslice_hidden_state_slot.has_value());
  EXPECT_EQ(state.GetTokenAtStep(6, {}), 77);
  EXPECT_EQ(state.total_cycles(), 1);
  EXPECT_EQ(state.total_accepted_tokens(), 0);
}

TEST(SpeculativeDecodingStateTest, EmptyOrOutOfBoundsEdgeCases) {
  SpeculativeDecodingState state;
  SamplerParameters greedy_params;
  greedy_params.set_k(1);
  greedy_params.set_temperature(0.0f);

  // BackfillCommittedTokens null checks, no-op when ready_tokens <= 1, and
  // num_output_tokens underflow check.
  std::vector<int> input_tokens = {10, 20};
  int ready_tokens = 1;
  int num_output_tokens = 0;
  EXPECT_THAT(state.BackfillCommittedTokens(nullptr, &ready_tokens, nullptr),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(state.BackfillCommittedTokens(&input_tokens, nullptr, nullptr),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_OK(state.BackfillCommittedTokens(&input_tokens, &ready_tokens,
                                          &num_output_tokens));
  EXPECT_THAT(input_tokens, ElementsAre(10, 20));

  state.RecordTokenOverride(2, 30);
  ready_tokens = 2;
  num_output_tokens = 0;  // < backfilled_count (1) -> InternalError
  EXPECT_THAT(state.BackfillCommittedTokens(&input_tokens, &ready_tokens,
                                            &num_output_tokens),
              StatusIs(absl::StatusCode::kInternal));

  // PrepareScoreStep empty tokens error and seq_size <= 0 / f32_channels == 0.
  EXPECT_THAT(state.PrepareScoreStep(/*gamma=*/2, /*seq_size=*/3,
                                     /*time_step=*/2, {}, greedy_params,
                                     /*f32_channels=*/0, nullptr),
              StatusIs(absl::StatusCode::kInvalidArgument));
  ASSERT_OK_AND_ASSIGN(
      const PreparedScoreStep prep_no_seq,
      state.PrepareScoreStep(/*gamma=*/2, /*seq_size=*/0, /*time_step=*/2,
                             {10, 20}, greedy_params, /*f32_channels=*/0,
                             nullptr));
  EXPECT_EQ(prep_no_seq.draft_size, 3);
  EXPECT_FALSE(prep_no_seq.updated_f32_params.has_value());

  // VerifyScoredTokens empty/too-small candidate_tokens and buffer bounds.
  EXPECT_THAT(state.VerifyScoredTokens(/*base_time_step=*/2, {10},
                                       {20, 30}, /*topk_tokens_channels=*/1, {},
                                       0, {}, greedy_params),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(state.VerifyScoredTokens(
                  /*base_time_step=*/2, {10, 20}, {20},
                  /*topk_tokens_channels=*/1, {}, 0, {}, greedy_params),
              StatusIs(absl::StatusCode::kInvalidArgument));

  SamplerParameters non_greedy_params;
  non_greedy_params.set_k(2);
  non_greedy_params.set_temperature(1.0f);
  non_greedy_params.set_p(0.9f);
  EXPECT_THAT(state.VerifyScoredTokens(
                  /*base_time_step=*/2, {10, 20}, {20, 21, 30, 31},
                  /*topk_tokens_channels=*/2,
                  /*topk_probs_data=*/{0.5f},
                  /*topk_probs_channels=*/2, {20, 30}, non_greedy_params),
              StatusIs(absl::StatusCode::kInvalidArgument));
  SpeculativeDecodingState fresh_verify_state;
  EXPECT_THAT(fresh_verify_state.VerifyScoredTokens(
                  /*base_time_step=*/2, {10, 20}, {20, 21, 30, 31},
                  /*topk_tokens_channels=*/2,
                  /*topk_probs_data=*/{0.5f, 0.5f, 0.5f, 0.5f},
                  /*topk_probs_channels=*/2,
                  /*sampled_tokens_data=*/{20}, non_greedy_params),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_EQ(fresh_verify_state.last_score_base_time_step(), -1);
  EXPECT_EQ(fresh_verify_state.total_cycles(), 0);
  EXPECT_THAT(fresh_verify_state.GetProbabilityCache(3),
              StatusIs(absl::StatusCode::kNotFound));

  // RewindOnReject null ready_tokens, non-positive step, and out-of-bounds
  // step.
  EXPECT_THAT(
      state.RewindOnReject(TokenAtTimeStep(2, 30), /*input_tokens_size=*/2,
                           nullptr, nullptr, /*gamma=*/2, greedy_params),
      StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(
      state.RewindOnReject(TokenAtTimeStep(0, 30), /*input_tokens_size=*/2,
                           &ready_tokens, nullptr, /*gamma=*/2, greedy_params),
      StatusIs(absl::StatusCode::kInvalidArgument));
  ready_tokens = 2;  // initial_time_step = 2 - 1 + 2 = 3
  EXPECT_THAT(
      state.RewindOnReject(TokenAtTimeStep(4, 30), /*input_tokens_size=*/2,
                           &ready_tokens, nullptr, /*gamma=*/2, greedy_params),
      StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(SpeculativeUtilsTest, BuildNucleusClippedDistributionSumsToOne) {
  std::vector<int32_t> tokens = {10, 20, 30, 40, 50};
  std::vector<float> probs = {0.45f, 0.25f, 0.15f, 0.10f, 0.05f};

  for (float p : {0.50f, 0.70f, 0.85f, 0.95f, 1.0f}) {
    SCOPED_TRACE(absl::StrCat("p = ", p));
    ASSERT_OK_AND_ASSIGN(
        std::vector<TokenProbability> dist,
        SpeculativeDecodingState::BuildNucleusClippedDistribution(
            tokens, probs, /*k_limit=*/5, p));
    float sum_p = 0.0f;
    for (const auto& tp : dist) {
      sum_p += tp.probability();
    }
    EXPECT_NEAR(sum_p, 1.0f, 1e-5f);
  }
}

TEST(SpeculativeUtilsTest,
     BuildNucleusClippedDistributionTruncatesBeyondNucleus) {
  std::vector<int32_t> tokens = {1, 2, 3, 4};
  std::vector<float> probs = {0.60f, 0.30f, 0.08f, 0.02f};
  ASSERT_OK_AND_ASSIGN(
      std::vector<TokenProbability> dist,
      SpeculativeDecodingState::BuildNucleusClippedDistribution(
          tokens, probs, /*k_limit=*/4, /*p=*/0.75f));
  ASSERT_EQ(dist.size(), 4u);
  // Token 1: 0.60 / 0.75 = 0.80
  EXPECT_NEAR(dist[0].probability(), 0.80f, 1e-5f);
  // Token 2: (0.75 - 0.60) / 0.75 = 0.20
  EXPECT_NEAR(dist[1].probability(), 0.20f, 1e-5f);
  // Token 3 and 4 truncated beyond nucleus: 0.0
  EXPECT_FLOAT_EQ(dist[2].probability(), 0.0f);
  EXPECT_FLOAT_EQ(dist[3].probability(), 0.0f);
}

TEST(SpeculativeDecodingStateTest,
     VerifyScoredTokensNonGreedyFirstSlotRejectRollback) {
  SpeculativeDecodingState state;
  SamplerParameters non_greedy_params;
  non_greedy_params.set_k(2);
  non_greedy_params.set_temperature(1.0f);
  non_greedy_params.set_p(0.8f);

  std::vector<int> candidates = {10, 20, 30};
  std::vector<int32_t> topk_tokens = {20, 21, 30, 31, 40, 41};
  std::vector<float> topk_probs = {0.5f, 0.3f, 0.5f, 0.3f, 0.5f, 0.3f};
  std::vector<int32_t> sampled_tokens = {20, 30, 40};

  ASSERT_OK_AND_ASSIGN(
      ScoreVerificationResult res,
      state.VerifyScoredTokens(/*base_time_step=*/4, candidates,
                               topk_tokens, /*topk_tokens_channels=*/2,
                               topk_probs, /*topk_probs_channels=*/2,
                               sampled_tokens, non_greedy_params));
  EXPECT_EQ(res.accepted_slot, 2);
  EXPECT_EQ(state.total_accepted_tokens(), 2);

  // Reject at the very first slot (step 5, resampled to 99).
  int ready_tokens = 4;
  int num_output_tokens = 3;
  ASSERT_OK_AND_ASSIGN(
      RejectRewindAction rewind,
      state.RewindOnReject(TokenAtTimeStep(5, 99), /*input_tokens_size=*/5,
                           &ready_tokens, &num_output_tokens, /*gamma=*/2,
                           non_greedy_params));
  EXPECT_EQ(ready_tokens, 1);
  EXPECT_EQ(num_output_tokens, 0);
  ASSERT_TRUE(rewind.reslice_hidden_state_slot.has_value());
  EXPECT_EQ(*rewind.reslice_hidden_state_slot, 0);

  EXPECT_EQ(state.total_cycles(), 1);
  EXPECT_EQ(state.total_accepted_tokens(), 0);
}

// -----------------------------------------------------------------------------
// Adversarial Stress Tests: BuildNucleusClippedDistribution
// -----------------------------------------------------------------------------

TEST(SpeculativeUtilsAdversarialTest,
     BuildNucleusClippedDistributionTableDrivenEdgeCases) {
  const float denorm_min = std::numeric_limits<float>::denorm_min();
  struct TestCase {
    absl::string_view name;
    std::vector<int32_t> tokens;
    std::vector<float> probs;
    std::vector<float> p_values;
  };
  std::vector<TestCase> cases = {
      {"subnormal",
       {1, 2, 3, 4},
       {denorm_min, 2 * denorm_min, denorm_min, 4 * denorm_min},
       {0.0f, 0.1f, 0.5f, 0.9f, 1.0f}},
      {"sums_gt_1",
       {10, 20, 30, 40},
       {0.8f, 0.7f, 0.6f, 0.5f},
       {0.0f, 0.2f, 0.5f, 0.8f, 1.0f, 1.5f, 2.0f}},
      {"sums_lt_1",
       {10, 20, 30},
       {0.01f, 0.02f, 0.01f},
       {0.0f, 0.005f, 0.02f, 0.04f, 0.5f, 1.0f}},
      {"all_zeros", {10, 20, 30}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.5f, 1.0f}},
      {"single_elem", {1, 2, 3}, {0.999f, 0.0006f, 0.0004f}, {0.5f}},
      {"bounds", {10, 20, 30}, {0.70f, 0.20f, 0.10f}, {-0.5f, 0.0f, 1.0f}},
  };

  for (const auto& tc : cases) {
    for (float p : tc.p_values) {
      SCOPED_TRACE(absl::StrCat("Case: ", tc.name, ", p: ", p));
      ASSERT_OK_AND_ASSIGN(
          std::vector<TokenProbability> dist,
          SpeculativeDecodingState::BuildNucleusClippedDistribution(
              tc.tokens, tc.probs, tc.tokens.size(), p));
      ASSERT_EQ(dist.size(), tc.tokens.size());
      double sum_p = 0.0;
      for (const auto& tp : dist) {
        EXPECT_GE(tp.probability(), 0.0f);
        EXPECT_LE(tp.probability(), 1.0f);
        sum_p += tp.probability();
      }
      EXPECT_NEAR(sum_p, 1.0, 1e-5);
    }
  }

  // k_limit = 0 must fail with InvalidArgument.
  EXPECT_THAT(
      SpeculativeDecodingState::BuildNucleusClippedDistribution(
          {1, 2}, {0.5f, 0.5f}, /*k_limit=*/0, /*p=*/0.8f),
      StatusIs(absl::StatusCode::kInvalidArgument));

  // NaN prob must fail with InvalidArgument.
  EXPECT_THAT(
      SpeculativeDecodingState::BuildNucleusClippedDistribution(
          {1, 2}, {0.5f, NAN}, /*k_limit=*/2, /*p=*/0.8f),
      StatusIs(absl::StatusCode::kInvalidArgument));

  // Negative prob must fail with InvalidArgument.
  EXPECT_THAT(
      SpeculativeDecodingState::BuildNucleusClippedDistribution(
          {1, 2}, {0.5f, -0.1f}, /*k_limit=*/2, /*p=*/0.8f),
      StatusIs(absl::StatusCode::kInvalidArgument));

  // NaN p must fail with InvalidArgument.
  EXPECT_THAT(
      SpeculativeDecodingState::BuildNucleusClippedDistribution(
          {1, 2}, {0.5f, 0.5f}, /*k_limit=*/2, /*p=*/NAN),
      StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(SpeculativeUtilsAdversarialTest,
     BuildNucleusClippedDistributionLargeVocabulary256k) {
  const int kVocabSize = 256000;
  std::vector<int32_t> tokens(kVocabSize);
  std::vector<float> uniform_probs(kVocabSize, 1.0f / kVocabSize);
  std::vector<float> zipf_probs(kVocabSize);

  for (int i = 0; i < kVocabSize; ++i) {
    tokens[i] = i;
    zipf_probs[i] = 1.0f / (i + 1);
  }

  // Uniform distribution over 256k tokens
  for (float p : {0.1f, 0.5f, 0.9f, 1.0f}) {
    SCOPED_TRACE(absl::StrCat("256k uniform p = ", p));
    ASSERT_OK_AND_ASSIGN(
        std::vector<TokenProbability> dist,
        SpeculativeDecodingState::BuildNucleusClippedDistribution(
            tokens, uniform_probs, kVocabSize, p));
    ASSERT_EQ(dist.size(), static_cast<size_t>(kVocabSize));
    double sum_p = 0.0;
    for (const auto& tp : dist) {
      sum_p += tp.probability();
    }
    // With double precision accumulators in speculative_utils.cc,
    // sum_p passes strict 1e-5 tolerance.
    EXPECT_NEAR(sum_p, 1.0, 1e-5);
  }

  // Zipf distribution over 256k tokens
  for (float p : {0.1f, 0.5f, 0.9f, 1.0f}) {
    SCOPED_TRACE(absl::StrCat("256k Zipf p = ", p));
    ASSERT_OK_AND_ASSIGN(
        std::vector<TokenProbability> dist,
        SpeculativeDecodingState::BuildNucleusClippedDistribution(
            tokens, zipf_probs, kVocabSize, p));
    ASSERT_EQ(dist.size(), static_cast<size_t>(kVocabSize));
    double sum_p = 0.0;
    for (const auto& tp : dist) {
      sum_p += tp.probability();
    }
    EXPECT_NEAR(sum_p, 1.0, 1e-5);
  }
}

// -----------------------------------------------------------------------------
// Adversarial Stress Tests: RewindOnReject
// -----------------------------------------------------------------------------

TEST(SpeculativeUtilsAdversarialTest,
     RewindOnRejectAdversarialAllSlotsSimulation) {
  for (int gamma : {1, 2, 3, 5, 8}) {
    for (int reject_slot = 0; reject_slot <= gamma; ++reject_slot) {
      SCOPED_TRACE(
          absl::StrCat("Testing gamma=", gamma, ", reject_slot=", reject_slot));
      SpeculativeDecodingState state;
      SamplerParameters non_greedy_params;
      non_greedy_params.set_k(4);
      non_greedy_params.set_temperature(1.0f);
      non_greedy_params.set_p(0.8f);

      const int actual_draft_size = gamma + 1;
      std::vector<int> candidates(actual_draft_size);
      std::iota(candidates.begin(), candidates.end(), 100);

      std::vector<int32_t> topk_tokens(actual_draft_size * 4);
      std::vector<float> topk_probs(actual_draft_size * 4, 0.25f);
      for (int i = 0; i < actual_draft_size; ++i) {
        for (int c = 0; c < 4; ++c) {
          topk_tokens[i * 4 + c] = 100 + i + c;
        }
      }
      std::vector<int32_t> sampled_tokens(actual_draft_size);
      std::iota(sampled_tokens.begin(), sampled_tokens.end(), 100);

      const int base_time_step = 10;
      ASSERT_OK_AND_ASSIGN(
          ScoreVerificationResult res,
          state.VerifyScoredTokens(base_time_step, candidates,
                                   topk_tokens, /*topk_tokens_channels=*/4,
                                   topk_probs, /*topk_probs_channels=*/4,
                                   sampled_tokens, non_greedy_params));

      EXPECT_EQ(res.accepted_slot, gamma);
      EXPECT_EQ(state.total_accepted_tokens(), gamma);

      const int input_tokens_size = 10;
      int ready_tokens = gamma + 2;
      int num_output_tokens = ready_tokens - 1;
      const int resampled_step = base_time_step + 1 + reject_slot;

      ASSERT_OK_AND_ASSIGN(
          RejectRewindAction rewind,
          state.RewindOnReject(TokenAtTimeStep(resampled_step, 9999),
                               input_tokens_size, &ready_tokens,
                               &num_output_tokens, gamma, non_greedy_params));

      const int dirty_slots = (base_time_step + 1 + gamma) - resampled_step;
      EXPECT_EQ(rewind.dirty_slots, dirty_slots);

      if (dirty_slots > 0) {
        const int actual_accepted = reject_slot;
        ASSERT_TRUE(rewind.reslice_hidden_state_slot.has_value());
        EXPECT_EQ(*rewind.reslice_hidden_state_slot, actual_accepted);

        // Verify total_accepted_tokens_ decreased by (gamma - actual_accepted).
        EXPECT_EQ(state.total_accepted_tokens(), actual_accepted);
      } else {
        EXPECT_FALSE(rewind.reslice_hidden_state_slot.has_value());
        EXPECT_EQ(state.total_accepted_tokens(), gamma);
      }
    }
  }
}

TEST(SpeculativeUtilsAdversarialTest,
     RewindOnRejectMultiCycleConsecutiveRollbackSequence) {
  SpeculativeDecodingState state;
  SamplerParameters non_greedy_params;
  non_greedy_params.set_k(2);
  non_greedy_params.set_temperature(1.0f);
  non_greedy_params.set_p(0.85f);

  int cumulative_accepted = 0;
  const int gamma = 3;
  std::vector<int> candidates = {10, 20, 30, 40};
  std::vector<int32_t> topk_tokens = {20, 21, 30, 31, 40, 41, 50, 51};
  std::vector<float> topk_probs(8, 0.5f);
  std::vector<int32_t> sampled_tokens = {20, 30, 40, 50};

  struct CycleTest {
    int base_time_step;
    std::optional<int> reject_slot;  // nullopt = no reject
  };
  std::vector<CycleTest> cycles = {
      {0, std::nullopt},  // Cycle 1: full accept
      {10, 1},            // Cycle 2: reject at slot 1
      {20, 0},            // Cycle 3: reject at slot 0
  };

  for (const auto& c : cycles) {
    ASSERT_OK_AND_ASSIGN(
        auto res, state.VerifyScoredTokens(
                      c.base_time_step, candidates, topk_tokens, 2,
                      topk_probs, 2, sampled_tokens, non_greedy_params));
    EXPECT_EQ(res.accepted_slot, 3);
    if (!c.reject_slot.has_value()) {
      cumulative_accepted += gamma;
    } else {
      int ready_tokens = 5;
      int num_output_tokens = 4;
      ASSERT_OK_AND_ASSIGN(
          auto rewind,
          state.RewindOnReject(
              TokenAtTimeStep(c.base_time_step + 1 + *c.reject_slot, 999),
              /*input_tokens_size=*/c.base_time_step, &ready_tokens,
              &num_output_tokens, gamma, non_greedy_params));
      EXPECT_EQ(rewind.reslice_hidden_state_slot, *c.reject_slot);
      cumulative_accepted += *c.reject_slot;
    }
    EXPECT_EQ(state.total_accepted_tokens(), cumulative_accepted);
  }
  EXPECT_EQ(state.total_cycles(), 3);
}

TEST(SpeculativeDecodingStateTest,
     RewindOnRejectStandaloneDrafterTruncatedInputs) {
  SpeculativeDecodingState state;
  SamplerParameters greedy_params;
  greedy_params.set_k(1);
  greedy_params.set_temperature(0.0f);

  int ready_tokens = 1;
  int num_output_tokens = 0;
  ASSERT_OK_AND_ASSIGN(
      RejectRewindAction rewind,
      state.RewindOnReject(TokenAtTimeStep(6, 77), /*input_tokens_size=*/9,
                           &ready_tokens, &num_output_tokens, /*gamma=*/2,
                           greedy_params));
  EXPECT_EQ(rewind.dirty_slots, 3);
  EXPECT_EQ(ready_tokens, 1);
  EXPECT_EQ(num_output_tokens, 0);
  EXPECT_EQ(state.GetTokenAtStep(6, {}), 77);
}

TEST(SpeculativeDecodingStateTest, GetProbabilityCacheChainedDraftFallback) {
  SpeculativeDecodingState state;
  // In-range hit: base 5, drafts {7, 8}, time_step 7 -> draft_idx 1 -> token 8.
  EXPECT_THAT(
      state.GetProbabilityCache(/*time_step=*/7, /*chained_base_step=*/5,
                                /*chained_drafts=*/{7, 8}),
      IsOkAndHolds(ElementsAre(TokenProbability::CreateGreedy(8))));
  // Out-of-range miss: step 8 -> draft_idx 8 - 5 - 1 = 2 (size 2) -> NotFound
  EXPECT_THAT(
      state.GetProbabilityCache(/*time_step=*/8, /*chained_base_step=*/5,
                                /*chained_drafts=*/{7, 8}),
      StatusIs(absl::StatusCode::kNotFound));
  // Negative base with drafts gives NotFound for step 0
  EXPECT_THAT(
      state.GetProbabilityCache(/*time_step=*/0, /*chained_base_step=*/-1,
                                /*chained_drafts=*/{7, 8}),
      StatusIs(absl::StatusCode::kNotFound));
  // SetProbabilityCache entry takes precedence over drafts
  state.SetProbabilityCache(6, {TokenProbability::CreateGreedy(99)});
  EXPECT_THAT(
      state.GetProbabilityCache(/*time_step=*/6, /*chained_base_step=*/5,
                                /*chained_drafts=*/{7, 8}),
      IsOkAndHolds(ElementsAre(TokenProbability::CreateGreedy(99))));
}

TEST(SpeculativeDecodingStateTest,
     RewindOnRejectWithoutPriorScoreAndNullOutputCount) {
  SpeculativeDecodingState state;
  SamplerParameters non_greedy_params;
  non_greedy_params.set_k(4);
  non_greedy_params.set_temperature(1.0f);
  non_greedy_params.set_p(0.8f);

  int ready_tokens = 4;
  ASSERT_OK_AND_ASSIGN(
      RejectRewindAction rewind,
      state.RewindOnReject(TokenAtTimeStep(4, 99), /*input_tokens_size=*/3,
                           &ready_tokens, /*num_output_tokens=*/nullptr,
                           /*gamma=*/3, non_greedy_params));
  EXPECT_EQ(rewind.dirty_slots, 2);
  ASSERT_TRUE(rewind.reslice_hidden_state_slot.has_value());
  EXPECT_EQ(*rewind.reslice_hidden_state_slot, 1);
}

}  // namespace
}  // namespace speculative_utils
}  // namespace ml_drift
