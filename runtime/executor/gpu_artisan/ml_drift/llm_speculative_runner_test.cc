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

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "runtime/executor/gpu_artisan/ml_drift/llm.h"
#include "runtime/executor/gpu_artisan/ml_drift/llm_gpu_drafter.h"
#include "runtime/executor/gpu_artisan/ml_drift/llm_gpu_runner.h"
#include "runtime/executor/gpu_artisan/ml_drift/speculative_utils.h"
#include "runtime/executor/gpu_artisan/ml_drift/tensor_loaders/llm_tensor_loader.h"
#include "runtime/util/status_macros.h"
#include "runtime/util/test_utils.h"

namespace odml::infra::gpu {
namespace {

using ::ml_drift::speculative_utils::TokenAtTimeStep;
using ::ml_drift::speculative_utils::TokenProbability;
using ::testing::ElementsAreArray;

class UnimplementedLlmGpuRunner : public LlmGpuRunner {
 public:
  explicit UnimplementedLlmGpuRunner(int gamma) {
    // Set a sensible topk value.
    gamma_ = gamma;
  }
  absl::Status InitInputTokens(const std::vector<int>& input_tokens,
                               int current_step, bool wait_for_completion,
                               const std::atomic_bool* cancel) override {
    return absl::UnimplementedError("Not implemented.");
  };
  using LlmGpuRunner::CleanCache;
  absl::Status CleanCache(int time_step,
                          bool skip_ringbuffer_delta_check) override {
    return absl::UnimplementedError("Not implemented.");
  };

  absl::Status GetNextTokens(std::vector<std::vector<int>>* output_tokens,
                             UpdateConstraintFn update_constraint_fn) override {
    return absl::UnimplementedError("Not implemented.");
  };

  absl::Status LoadLoRA(
      uint32_t lora_id, int lora_rank,
      std::unique_ptr<LlmTensorLoader> tensor_loader) override {
    return absl::UnimplementedError("Not implemented.");
  }
  absl::Status UnloadLoRA(uint32_t lora_id) override {
    return absl::UnimplementedError("Not implemented.");
  }

  absl::Status UseLoRA(std::optional<uint32_t> lora_id) override {
    return absl::UnimplementedError("Not implemented.");
  }

  absl::Status UseLayers(const absl::flat_hash_set<int>& layers,
                         int final_ln) override {
    return absl::UnimplementedError("Not implemented.");
  }

  absl::Status UpdateSamplerParams(
      const proto::SamplerParameters& new_params,
      std::shared_ptr<std::default_random_engine> rand_gen) override {
    return absl::UnimplementedError("Not implemented.");
  }

  // Get the maximum possible Top-K value. Top-K provided to UpdateSamplerParams
  // should not be larger than max Top-K.
  int GetMaxTopK() override { return 16; }

 private:
  int gamma_;
};

// Fake for LLM GPU Runner.
class FakeLlmGpuRunner : public LlmGpuRunner {
 public:
  // Each TokenInferenceParameters represents a possible prefix (input) and the
  // corresponding parameters for generating the output.
  struct TokenInferenceParameters {
    // This set of parameter is active whenever the prefix matches exactly as
    // below.
    std::vector<int> prefix;

    // Parameters for Gaussian distribution.
    float mean;
    float stddev;

    // The random sampling value between 0 and 1.
    float sampling_value;
  };

  explicit FakeLlmGpuRunner(int gamma) {
    // Set a sensible topk value.
    sampler_params_.set_k(16);
    gamma_ = gamma;
  }

#ifdef __EMSCRIPTEN__
  absl::Status InitInputTokens(const std::vector<int>& input_tokens,
                               int current_step,
                               bool wait_for_completion) override {
    return absl::OkStatus();
  }
#else
  absl::Status InitInputTokens(const std::vector<int>& input_tokens,
                               int current_step, bool wait_for_completion,
                               const std::atomic_bool* cancel) override {
    RETURN_IF_ERROR(CleanCache(current_step));
    current_prefix_.insert(current_prefix_.end(), input_tokens.begin(),
                           input_tokens.end());
    return absl::OkStatus();
  }
#endif

  using LlmGpuRunner::CleanCache;
  absl::Status CleanCache(int time_step,
                          bool skip_ringbuffer_delta_check) override {
    // Clean current_prefix_ until only time_step is left, if it's too short,
    // we'll assert.
    ABSL_CHECK_GE(current_prefix_.size(), time_step);
    current_prefix_.resize(time_step);
    return absl::OkStatus();
  }

  absl::Status GetNextTokens(std::vector<std::vector<int>>* output_tokens,
                             UpdateConstraintFn update_constraint_fn) override {
    // Batching not supported.
    ABSL_CHECK_EQ(output_tokens->size(), 1);

    int expected_out_size = (*output_tokens)[0].size();
    for (int i = 0; i < expected_out_size; ++i) {
      ASSIGN_OR_RETURN(auto token, GetOneToken(false));
      (*output_tokens)[0][i] = token.token();
    }
    return absl::OkStatus();
  }

  absl::StatusOr<ml_drift::speculative_utils::TokenAtTimeStep> PeekNextToken()
      override {
    ASSIGN_OR_RETURN(auto token, GetOneToken(true));
    int time_step = current_prefix_.size();
    return ml_drift::speculative_utils::TokenAtTimeStep(time_step,
                                                        token.token());
  }

  absl::Status OverwriteOneInputToken(
      const ml_drift::speculative_utils::TokenAtTimeStep& token) override {
    token_override_[token.time_step()] = token.token();
    return absl::OkStatus();
  }

  absl::StatusOr<std::vector<int>> DraftTokens() override {
    std::vector<int> result;
    for (int i = 0; i < gamma_ + 1; ++i) {
      ASSIGN_OR_RETURN(auto token, GetOneToken(false));
      result.push_back(token.token());
    }
    return result;
  }

  absl::Status ScoreTokens(const std::vector<int>& tokens) override {
    current_prefix_.insert(current_prefix_.end(), tokens.begin(), tokens.end());
    return absl::OkStatus();
  }

  absl::StatusOr<std::vector<TokenProbability>> GetProbabilityCache(
      int time_step) override {
    RET_CHECK_GE(current_prefix_.size(), time_step)
        << "Time step is too large for GetProbabilityCache, expected "
        << current_prefix_.size() << " got " << time_step;
    std::vector<int> prefix(current_prefix_.begin(),
                            current_prefix_.begin() + time_step);
    for (const auto& params : result_database_) {
      if (params.prefix == prefix) {
        ASSIGN_OR_RETURN(auto probabilities,
                         GenerateTokenProbabilities(params));
        return probabilities;
      }
    }
    return absl::InternalError("No matching prefix found for " +
                               absl::StrJoin(prefix, ","));
  }

  absl::Status RejectAfterResample(
      const ml_drift::speculative_utils::TokenAtTimeStep& token) override {
    RETURN_IF_ERROR(CleanCache(token.time_step()));
    while (current_prefix_.size() > token.time_step()) {
      current_prefix_.pop_back();
    }
    TokenAtTimeStep overwrite_token(token.time_step(), token.token());
    RETURN_IF_ERROR(OverwriteOneInputToken(overwrite_token));
    return absl::OkStatus();
  }

  absl::Status UpdateSamplerParams(
      const proto::SamplerParameters& new_params,
      std::shared_ptr<std::default_random_engine> rand_gen) override {
    sampler_params_ = new_params;
    rand_gen_ = rand_gen;
    return absl::OkStatus();
  }

  absl::Status LoadLoRA(
      uint32_t lora_id, int lora_rank,
      std::unique_ptr<LlmTensorLoader> tensor_loader) override {
    return absl::UnimplementedError("Not implemented.");
  }

  absl::Status UnloadLoRA(uint32_t lora_id) override {
    return absl::UnimplementedError("Not implemented.");
  }

  int GetMaxTopK() override { return sampler_params_.k(); }

  absl::Status UseLoRA(std::optional<uint32_t> lora_id) override {
    return absl::UnimplementedError("Not implemented.");
  }

  absl::Status UseLayers(const absl::flat_hash_set<int>& layers,
                         int final_ln) override {
    return absl::UnimplementedError("Not implemented.");
  }

  // For testing only.
  void SetResultDatabase(
      std::vector<TokenInferenceParameters> result_database) {
    result_database_ = result_database;
  }

  absl::StatusOr<ml_drift::ContextData> SaveContext() override {
    ml_drift::ContextData context_data;
    context_data.tokens = current_prefix_;
    return context_data;
  }

  absl::StatusOr<ml_drift::ContextData> SaveLocalContext() override {
    ml_drift::ContextData context_data;
    context_data.tokens = current_prefix_;
    return context_data;
  }

  absl::Status RestoreContext(
      ml_drift::ContextData&& context_data) override {
    current_prefix_ = std::move(context_data.tokens);
    return absl::OkStatus();
  }

  absl::Status RestoreLocalContext(
      ml_drift::ContextData&& context_data) override {
    return absl::OkStatus();
  }

  absl::StatusOr<ml_drift::ContextData> CloneLocalContext(
      const ml_drift::ContextData& context_data) override {
    ml_drift::ContextData cloned;
    cloned.tokens = context_data.tokens;
    cloned.lora_id = context_data.lora_id;
    return cloned;
  }

 private:
  // The internal state.
  std::vector<int> current_prefix_;

  // Gamma for speculative decoding.
  int gamma_;

  // Internal token override for PeekNextToken and OverwriteOneInputToken.
  absl::flat_hash_map<int, int> token_override_;

  proto::SamplerParameters sampler_params_;
  std::shared_ptr<std::default_random_engine> rand_gen_;

  // A fixed database of parameters for possible prefixes encountered in the
  // test.
  std::vector<struct TokenInferenceParameters> result_database_;

  // Max token ID for this test.
  static constexpr int kMaxTokenId = 256 * 1024;

  // For a given set of parameters, generate the distribution of token
  // probabilities, taking the sampling parameters into account.
  absl::StatusOr<std::vector<ml_drift::speculative_utils::TokenProbability>>
  GenerateTokenProbabilities(const TokenInferenceParameters& params) {
    std::vector<int> tokens;
    std::vector<float> relative_probs;
    const int topk = sampler_params_.k();
    int center_token = floor(params.mean + 0.5);
    for (int i = -3 - topk; i < topk + 3; ++i) {
      int token = center_token + i;
      if (token <= 0) continue;
      if (token >= kMaxTokenId) continue;
      float start_x = static_cast<float>(token) - 0.5;
      float end_x = static_cast<float>(token) + 0.5;
      float cdf_start =
          0.5 * (1.0 + erf((start_x - params.mean) / params.stddev));
      float cdf_end = 0.5 * (1.0 + erf((end_x - params.mean) / params.stddev));
      float prob = cdf_end - cdf_start;
      tokens.push_back(token);
      relative_probs.push_back(prob);
    }

    std::vector<ml_drift::speculative_utils::TokenProbability> result;
    for (int i = 0; i < tokens.size(); ++i) {
      ASSIGN_OR_RETURN(auto token_probability,
                       ml_drift::speculative_utils::TokenProbability::Create(
                           tokens[i], relative_probs[i]));
      result.push_back(token_probability);
    }
    // Sort by probability, from high to low.
    std::sort(result.begin(), result.end(),
              [](const ml_drift::speculative_utils::TokenProbability& a,
                 const ml_drift::speculative_utils::TokenProbability& b) {
                return a.probability() > b.probability();
              });

    // Trim until we've top K tokens.
    while (result.size() > topk) {
      result.pop_back();
    }

    // Normalize the probabilities.
    float sum_probs = 0.0;
    for (const auto& token_probability : result) {
      sum_probs += token_probability.probability();
    }
    for (auto& token_probability : result) {
      RETURN_IF_ERROR(token_probability.set_probability(
          token_probability.probability() / sum_probs));
    }
    return result;
  }

  // Takes a token probability distribution and given sample value, sample from
  // the distribution and return the sampled token.
  absl::StatusOr<TokenProbability> SampleTokenWithProbability(
      const std::vector<ml_drift::speculative_utils::TokenProbability>& probs,
      float sampling_value) {
    float sum_probs = 0.0;
    for (const auto& prob : probs) {
      sum_probs += prob.probability();
      if (sum_probs >= sampling_value) {
        return prob;
      }
    }
    return absl::InternalError(
        "Nothing sampled for SampleTokenWithProbability");
  }

  // Attempts to find the probability of a corresponding token in a probability
  // distribution.
  float GetProbabilityForToken(
      const std::vector<ml_drift::speculative_utils::TokenProbability>& probs,
      int token) {
    for (const auto& prob : probs) {
      if (prob.token() == token) {
        return prob.probability();
      }
    }
    return 0.0;
  }

  // Attempt to find a match in the result database and if found generate the
  // token probability distribution and sample from it.
  // If peek is true, then do not change the current state.
  absl::StatusOr<TokenProbability> GetOneToken(bool peek) {
    // Check if there is a match in the database. This class is designed such
    // that the test should always populate the database such that there is a
    // always match, but when the behaviour we want to test changes, we might
    // not find a match in the database.
    for (const auto& params : result_database_) {
      if (params.prefix == current_prefix_) {
        ASSIGN_OR_RETURN(auto probabilities,
                         GenerateTokenProbabilities(params));

        ASSIGN_OR_RETURN(TokenProbability token_prob,
                         TokenProbability::Create(-1, 0.0));
        if (token_override_.contains(current_prefix_.size())) {
          int token = token_override_[current_prefix_.size()];
          float prob = GetProbabilityForToken(probabilities, token);
          ASSIGN_OR_RETURN(token_prob, TokenProbability::Create(token, prob));
        } else {
          // Sample the token.
          ASSIGN_OR_RETURN(
              token_prob,
              SampleTokenWithProbability(probabilities, params.sampling_value));
        }
        if (!peek) {
          current_prefix_.push_back(token_prob.token());
        }
        return token_prob;
      }
    }
    std::string prefix_str = absl::StrJoin(current_prefix_, ",");
    return absl::InternalError("No matching prefix found for " + prefix_str);
  }
};

// Create Test Fixtures for LlmSpeculativeRunner.
class LlmSpeculativeRunnerTest : public ::testing::Test {
 public:
  void SetUp() override {
    rand_gen_ = std::make_shared<std::default_random_engine>(0);
  }

  void CreateMainRunner(
      const std::vector<FakeLlmGpuRunner::TokenInferenceParameters>&
          main_responses) {
    owned_main_runner_ = std::make_unique<FakeLlmGpuRunner>(kGamma);
    owned_main_runner_->SetResultDatabase(main_responses);
    main_runner_ = owned_main_runner_.get();
  }

  void CreateDrafterRunner(
      const std::vector<FakeLlmGpuRunner::TokenInferenceParameters>&
          draft_responses) {
    auto fake_gpu_runner = std::make_unique<FakeLlmGpuRunner>(kGamma);
    fake_gpu_runner->SetResultDatabase(draft_responses);
    drafter_runner_ = fake_gpu_runner.get();
    owned_drafter_ =
        std::make_unique<LlmGpuDrafter>(std::move(fake_gpu_runner));
  }

  void CreateUnimplementedDrafterRunner() {
    auto fake_gpu_runner = std::make_unique<UnimplementedLlmGpuRunner>(kGamma);
    owned_drafter_ =
        std::make_unique<LlmGpuDrafter>(std::move(fake_gpu_runner));
  }

  void CreateSpeculativeRunner(int gamma) {
    ASSERT_OK_AND_ASSIGN(speculative_runner_,
                         LlmGpuRunner::CreateSpeculativeRunner(
                             std::move(owned_main_runner_),
                             std::move(owned_drafter_), gamma, rand_gen_));
    ASSERT_OK(speculative_runner_->EnableSpeculativeDecoding(true));
  }

 protected:
  FakeLlmGpuRunner* main_runner_;
  FakeLlmGpuRunner* drafter_runner_;
  std::unique_ptr<FakeLlmGpuRunner> owned_main_runner_;
  std::unique_ptr<LlmGpuDrafter> owned_drafter_;

  std::unique_ptr<LlmGpuRunner> speculative_runner_;

  std::shared_ptr<std::default_random_engine> rand_gen_;

  // A common gamma for most tests.
  static constexpr int kGamma = 4;
};

TEST_F(LlmSpeculativeRunnerTest, InitInputTokensTest) {
  const std::vector<FakeLlmGpuRunner::TokenInferenceParameters> main_responses =
      {{.prefix = {1, 2, 3},
        .mean = 115.3,
        .stddev = 0.1,
        .sampling_value = 0.5}};
  const std::vector<FakeLlmGpuRunner::TokenInferenceParameters>
      draft_responses = {{.prefix = {1, 2, 3},
                          .mean = 177.3,
                          .stddev = 0.1,
                          .sampling_value = 0.5}};

  CreateMainRunner(main_responses);
  CreateDrafterRunner(draft_responses);
  CreateSpeculativeRunner(/*gamma=*/4);

  std::vector<int> prefix = {1, 2, 3};
  std::atomic<bool> cancel = false;
  ASSERT_OK(speculative_runner_->InitInputTokens(prefix, 0, true, &cancel));
  // Set TopK to 40.
  proto::SamplerParameters sampler_params;
  sampler_params.set_k(40);
  ASSERT_OK(speculative_runner_->UpdateSamplerParams(
      sampler_params, std::make_shared<std::default_random_engine>()));

  std::vector<std::vector<int>> output_tokens;
  output_tokens.push_back(std::vector<int>{
      -1,
  });
  ASSERT_OK(
      main_runner_->GetNextTokens(&output_tokens, /*update_constraint_fn=*/{}));
  EXPECT_THAT(output_tokens[0], ElementsAreArray({115}));
  ASSERT_OK(drafter_runner_->GetNextTokens(&output_tokens,
                                           /*update_constraint_fn=*/{}));
  EXPECT_THAT(output_tokens[0], ElementsAreArray({177}));
}

TEST_F(LlmSpeculativeRunnerTest, CleanCacheTest) {
  const std::vector<FakeLlmGpuRunner::TokenInferenceParameters> main_responses =
      {{.prefix = {1, 2, 3},
        .mean = 115.3,
        .stddev = 0.1,
        .sampling_value = 0.5}};
  const std::vector<FakeLlmGpuRunner::TokenInferenceParameters>
      draft_responses = {{.prefix = {1, 2, 3},
                          .mean = 177.3,
                          .stddev = 0.1,
                          .sampling_value = 0.5}};

  CreateMainRunner(main_responses);
  CreateDrafterRunner(draft_responses);
  CreateSpeculativeRunner(/*gamma=*/4);

  std::vector<int> prefix = {1, 2, 3, 7, 8, 9};
  std::atomic<bool> cancel = false;
  ASSERT_OK(speculative_runner_->InitInputTokens(prefix, 0, true, &cancel));

  ASSERT_OK(speculative_runner_->CleanCache(3));

  std::vector<std::vector<int>> output_tokens;
  output_tokens.push_back(std::vector<int>{
      -1,
  });
  ASSERT_OK(
      main_runner_->GetNextTokens(&output_tokens, /*update_constraint_fn=*/{}));
  EXPECT_THAT(output_tokens[0], ElementsAreArray({115}));
  ASSERT_OK(drafter_runner_->GetNextTokens(&output_tokens,
                                           /*update_constraint_fn=*/{}));
  EXPECT_THAT(output_tokens[0], ElementsAreArray({177}));
}

TEST_F(LlmSpeculativeRunnerTest, UpdateSamplerParamsTest) {
  const std::vector<FakeLlmGpuRunner::TokenInferenceParameters> main_responses =
      {{.prefix = {1, 2, 3},
        .mean = 115.1,
        .stddev = 200,
        .sampling_value = 0.7}};
  const std::vector<FakeLlmGpuRunner::TokenInferenceParameters>
      draft_responses = {{.prefix = {1, 2, 3},
                          .mean = 177.1,
                          .stddev = 200,
                          .sampling_value = 0.7}};

  CreateMainRunner(main_responses);
  CreateDrafterRunner(draft_responses);
  CreateSpeculativeRunner(/*gamma=*/4);

  std::vector<int> prefix = {1, 2, 3};
  std::atomic<bool> cancel = false;
  ASSERT_OK(speculative_runner_->InitInputTokens(prefix, 0, true, &cancel));

  {
    // Set TopK to 40.
    proto::SamplerParameters sampler_params;
    sampler_params.set_k(40);
    ASSERT_OK(speculative_runner_->UpdateSamplerParams(
        sampler_params, std::make_shared<std::default_random_engine>()));

    // With TopK = 40 and a really large stddev, the output should not be the
    // mean.
    std::vector<std::vector<int>> output_tokens;
    output_tokens.push_back(std::vector<int>{
        -1,
    });
    ASSERT_OK(main_runner_->GetNextTokens(&output_tokens,
                                          /*update_constraint_fn=*/{}));
    EXPECT_NE(output_tokens[0][0], 115);
    ASSERT_OK(drafter_runner_->GetNextTokens(&output_tokens,
                                             /*update_constraint_fn=*/{}));
    EXPECT_NE(output_tokens[0][0], 177);
  }

  ASSERT_OK(speculative_runner_->CleanCache(3));
  {
    // Set TopK to 1.
    proto::SamplerParameters sampler_params;
    sampler_params.set_k(1);
    ASSERT_OK(speculative_runner_->UpdateSamplerParams(
        sampler_params, std::make_shared<std::default_random_engine>()));

    // With TopK = 1, the output should be the mean.
    std::vector<std::vector<int>> output_tokens;
    output_tokens.push_back(std::vector<int>{
        -1,
    });
    ASSERT_OK(main_runner_->GetNextTokens(&output_tokens,
                                          /*update_constraint_fn=*/{}));
    EXPECT_EQ(output_tokens[0][0], 115);
    ASSERT_OK(drafter_runner_->GetNextTokens(&output_tokens,
                                             /*update_constraint_fn=*/{}));
    EXPECT_EQ(output_tokens[0][0], 177);
  }
}

TEST_F(LlmSpeculativeRunnerTest, GetNextTokensFullRoundTest) {
  const std::vector<FakeLlmGpuRunner::TokenInferenceParameters> main_responses =
      {
          {.prefix = {1, 2, 3},
           .mean = 115.01,
           .stddev = 0.1,
           .sampling_value = 0.7},
          {.prefix = {1, 2, 3, 115},
           .mean = 180.01,
           .stddev = 3.0,
           .sampling_value = 0.5},
          {.prefix = {1, 2, 3, 115, 178},
           .mean = 133.01,
           .stddev = 0.1,
           .sampling_value = 0.3},
          {.prefix = {1, 2, 3, 115, 178, 133},
           .mean = 172.5,
           .stddev = 5.0,
           .sampling_value = 0.3},
          {.prefix = {1, 2, 3, 115, 178, 133, 174},
           .mean = 133.01,
           .stddev = 0.1,
           .sampling_value = 0.3},
      };
  const std::vector<FakeLlmGpuRunner::TokenInferenceParameters>
      draft_responses = {
          {.prefix = {1, 2, 3},
           .mean = 115.01,
           .stddev = 0.1,
           .sampling_value = 0.7},
          {.prefix = {1, 2, 3, 115},
           .mean = 180.01,
           .stddev = 5.0,
           .sampling_value = 0.5},
          {.prefix = {1, 2, 3, 115, 178},
           .mean = 133.01,
           .stddev = 0.1,
           .sampling_value = 0.3},
          {.prefix = {1, 2, 3, 115, 178, 133},
           .mean = 172.5,
           .stddev = 5.0,
           .sampling_value = 0.3},
          {.prefix = {1, 2, 3, 115, 178, 133, 174},
           .mean = 133.01,
           .stddev = 0.1,
           .sampling_value = 0.3},
      };

  CreateMainRunner(main_responses);
  CreateDrafterRunner(draft_responses);
  CreateSpeculativeRunner(kGamma);

  std::vector<int> prefix = {1, 2, 3};
  std::atomic<bool> cancel = false;
  ASSERT_OK(speculative_runner_->InitInputTokens(prefix, 0, true, &cancel));

  // Attempt to pull the next Gamma+1 tokens (the first round).
  std::vector<std::vector<int>> output_tokens;
  output_tokens.push_back(std::vector<int>(kGamma + 1, -1));
  ASSERT_OK(speculative_runner_->GetNextTokens(&output_tokens,
                                               /*update_constraint_fn=*/{}));
  ASSERT_EQ(output_tokens.size(), 1);
  EXPECT_THAT(output_tokens[0], ElementsAreArray({115, 178, 133, 174, 133}));
}

TEST_F(LlmSpeculativeRunnerTest, GetNextTokensPartialRoundTest) {
  const std::vector<FakeLlmGpuRunner::TokenInferenceParameters> main_responses =
      {
          {.prefix = {1, 2, 3},
           .mean = 115.01,
           .stddev = 0.1,
           .sampling_value = 0.7},
          {.prefix = {1, 2, 3, 115},
           .mean = 180.01,
           .stddev = 3.0,
           .sampling_value = 0.5},
          {.prefix = {1, 2, 3, 115, 178},
           .mean = 133.01,
           .stddev = 0.1,
           .sampling_value = 0.3},
          // Use a tight distribution around 177 so resampling after draft
          // rejection is deterministic across libc++, libstdc++, and MSVC STL.
          {.prefix = {1, 2, 3, 115, 178, 133},
           .mean = 177.01,
           .stddev = 0.1,
           .sampling_value = 0.3},
          {.prefix = {1, 2, 3, 115, 178, 133, 174},
           .mean = 133.01,
           .stddev = 0.1,
           .sampling_value = 0.3},

          // For the next round after rejection.
          {.prefix = {1, 2, 3, 115, 178, 133, 177},
           .mean = 125.0,
           .stddev = 0.1,
           .sampling_value = 0.5},
          {.prefix = {1, 2, 3, 115, 178, 133, 177, 125},
           .mean = 135.0,
           .stddev = 0.1,
           .sampling_value = 0.5},
          {.prefix = {1, 2, 3, 115, 178, 133, 177, 125, 135},
           .mean = 145.0,
           .stddev = 0.1,
           .sampling_value = 0.5},
          {.prefix = {1, 2, 3, 115, 178, 133, 177, 125, 135, 145},
           .mean = 155,
           .stddev = 0.1,
           .sampling_value = 0.5},
          {.prefix = {1, 2, 3, 115, 178, 133, 177, 125, 135, 145, 155},
           .mean = 165,
           .stddev = 0.1,
           .sampling_value = 0.5},
      };
  const std::vector<FakeLlmGpuRunner::TokenInferenceParameters>
      draft_responses = {
          {.prefix = {1, 2, 3},
           .mean = 115.01,
           .stddev = 0.1,
           .sampling_value = 0.7},
          {.prefix = {1, 2, 3, 115},
           .mean = 180.01,
           .stddev = 5.0,
           .sampling_value = 0.5},
          {.prefix = {1, 2, 3, 115, 178},
           .mean = 133.01,
           .stddev = 0.1,
           .sampling_value = 0.3},
          // This differs from main model so it should be rejected at this step.
          {.prefix = {1, 2, 3, 115, 178, 133},
           .mean = 220.5,
           .stddev = 5.0,
           .sampling_value = 0.3},
          {.prefix = {1, 2, 3, 115, 178, 133, 222},
           .mean = 133.01,
           .stddev = 0.1,
           .sampling_value = 0.3},

          // For the next round after rejection.
          {.prefix = {1, 2, 3, 115, 178, 133, 177},
           .mean = 125.0,
           .stddev = 0.1,
           .sampling_value = 0.5},
          {.prefix = {1, 2, 3, 115, 178, 133, 177, 125},
           .mean = 135.0,
           .stddev = 0.1,
           .sampling_value = 0.5},
          {.prefix = {1, 2, 3, 115, 178, 133, 177, 125, 135},
           .mean = 145.0,
           .stddev = 0.1,
           .sampling_value = 0.5},
          {.prefix = {1, 2, 3, 115, 178, 133, 177, 125, 135, 145},
           .mean = 155,
           .stddev = 0.1,
           .sampling_value = 0.5},
          {.prefix = {1, 2, 3, 115, 178, 133, 177, 125, 135, 145, 155},
           .mean = 165,
           .stddev = 0.1,
           .sampling_value = 0.5},
      };

  CreateMainRunner(main_responses);
  CreateDrafterRunner(draft_responses);
  CreateSpeculativeRunner(kGamma);

  std::vector<int> prefix = {1, 2, 3};
  std::atomic<bool> cancel = false;
  ASSERT_OK(speculative_runner_->InitInputTokens(prefix, 0, true, &cancel));

  // This will pull the first round but due to the rejection half way it'll pull
  // part of the second round as well.
  std::vector<std::vector<int>> output_tokens;
  output_tokens.push_back(std::vector<int>(kGamma + 1, -1));
  ASSERT_OK(speculative_runner_->GetNextTokens(&output_tokens,
                                               /*update_constraint_fn=*/{}));
  ASSERT_EQ(output_tokens.size(), 1);
  EXPECT_THAT(output_tokens[0], ElementsAreArray({115, 178, 133, 177, 125}));
}

TEST_F(LlmSpeculativeRunnerTest, MultipleInitInputTokensTest) {
  const std::vector<FakeLlmGpuRunner::TokenInferenceParameters> main_responses =
      {
          {.prefix = {1, 2, 3},
           .mean = 115.01,
           .stddev = 0.1,
           .sampling_value = 0.7},
          {.prefix = {1, 2, 3, 115},
           .mean = 180.01,
           .stddev = 3.0,
           .sampling_value = 0.5},
          {.prefix = {1, 2, 3, 115, 178},
           .mean = 133.01,
           .stddev = 0.1,
           .sampling_value = 0.3},
          {.prefix = {1, 2, 3, 115, 178, 133},
           .mean = 172.5,
           .stddev = 5.0,
           .sampling_value = 0.3},
          {.prefix = {1, 2, 3, 115, 178, 133, 174},
           .mean = 133.01,
           .stddev = 0.1,
           .sampling_value = 0.3},
          {.prefix = {4, 5, 6},
           .mean = 200.01,
           .stddev = 0.1,
           .sampling_value = 0.7},
          {.prefix = {4, 5, 6, 200},
           .mean = 321.01,
           .stddev = 3.0,
           .sampling_value = 0.5},
          {.prefix = {4, 5, 6, 200, 320},
           .mean = 442.01,
           .stddev = 0.1,
           .sampling_value = 0.3},
          {.prefix = {4, 5, 6, 200, 320, 442},
           .mean = 563.5,
           .stddev = 5.0,
           .sampling_value = 0.3},
          {.prefix = {4, 5, 6, 200, 320, 442, 565},
           .mean = 684.01,
           .stddev = 0.1,
           .sampling_value = 0.3},
      };
  const std::vector<FakeLlmGpuRunner::TokenInferenceParameters>
      draft_responses = {
          {.prefix = {1, 2, 3},
           .mean = 115.01,
           .stddev = 0.1,
           .sampling_value = 0.7},
          {.prefix = {1, 2, 3, 115},
           .mean = 180.01,
           .stddev = 5.0,
           .sampling_value = 0.5},
          {.prefix = {1, 2, 3, 115, 178},
           .mean = 133.01,
           .stddev = 0.1,
           .sampling_value = 0.3},
          {.prefix = {1, 2, 3, 115, 178, 133},
           .mean = 172.5,
           .stddev = 5.0,
           .sampling_value = 0.3},
          {.prefix = {1, 2, 3, 115, 178, 133, 174},
           .mean = 133.01,
           .stddev = 0.1,
           .sampling_value = 0.3},
          {.prefix = {4, 5, 6},
           .mean = 200.01,
           .stddev = 0.1,
           .sampling_value = 0.7},
          {.prefix = {4, 5, 6, 200},
           .mean = 321.01,
           .stddev = 3.0,
           .sampling_value = 0.5},
          {.prefix = {4, 5, 6, 200, 320},
           .mean = 442.01,
           .stddev = 0.1,
           .sampling_value = 0.3},
          {.prefix = {4, 5, 6, 200, 320, 442},
           .mean = 563.5,
           .stddev = 5.0,
           .sampling_value = 0.3},
          {.prefix = {4, 5, 6, 200, 320, 442, 565},
           .mean = 684.01,
           .stddev = 0.1,
           .sampling_value = 0.3},
      };

  CreateMainRunner(main_responses);
  CreateDrafterRunner(draft_responses);
  CreateSpeculativeRunner(kGamma);

  {
    std::vector<int> prefix = {1, 2, 3};
    std::atomic<bool> cancel = false;
    ASSERT_OK(speculative_runner_->InitInputTokens(prefix, 0, true, &cancel));

    // Attempt to pull less than kGamma + 1 tokens so that their are tokens left
    // in the ready_tokens.
    std::vector<std::vector<int>> output_tokens;
    output_tokens.push_back(std::vector<int>(kGamma, -1));
    ASSERT_OK(speculative_runner_->GetNextTokens(&output_tokens,
                                                /*update_constraint_fn=*/{}));
    ASSERT_EQ(output_tokens.size(), 1);
    EXPECT_THAT(output_tokens[0], ElementsAreArray({115, 178, 133, 174}));
  }

  {
    std::vector<int> prefix = {4, 5, 6};
    std::atomic<bool> cancel = false;
    ASSERT_OK(speculative_runner_->InitInputTokens(prefix, 0, true, &cancel));

    // Attempt to pull less than kGamma + 1 tokens so that their are tokens left
    // in the ready_tokens.
    std::vector<std::vector<int>> output_tokens;
    output_tokens.push_back(std::vector<int>(kGamma, -1));
    ASSERT_OK(speculative_runner_->GetNextTokens(&output_tokens,
                                                /*update_constraint_fn=*/{}));
    ASSERT_EQ(output_tokens.size(), 1);

    // There should be no left over tokens from the previous round in the
    // ready_tokens.
    EXPECT_THAT(output_tokens[0], ElementsAreArray({200, 320, 442, 565}));
  }
}

TEST_F(LlmSpeculativeRunnerTest, SaveAndRestoreContextTest) {
  const std::vector<FakeLlmGpuRunner::TokenInferenceParameters> main_responses =
      {
          // For the first InitInputTokens.
          {.prefix = {1, 2, 3},
           .mean = 115.01,
           .stddev = 0.1,
           .sampling_value = 0.7},
          {.prefix = {1, 2, 3, 115},
           .mean = 180.01,
           .stddev = 3.0,
           .sampling_value = 0.5},
          {.prefix = {1, 2, 3, 115, 178},
           .mean = 133.01,
           .stddev = 0.1,
           .sampling_value = 0.3},
          {.prefix = {1, 2, 3, 115, 178, 133},
           .mean = 172.5,
           .stddev = 5.0,
           .sampling_value = 0.3},
          {.prefix = {1, 2, 3, 115, 178, 133, 174},
           .mean = 133.01,
           .stddev = 0.1,
           .sampling_value = 0.3},
          // Context is saved and then restored here.
          {.prefix = {1, 2, 3, 115, 178, 133, 174, 133},
           .mean = 190.01,
           .stddev = 0.1,
           .sampling_value = 0.3},
          {.prefix = {1, 2, 3, 115, 178, 133, 174, 133, 190},
           .mean = 200.01,
           .stddev = 0.1,
           .sampling_value = 0.5},
          {.prefix = {1, 2, 3, 115, 178, 133, 174, 133, 190, 200},
           .mean = 210.01,
           .stddev = 0.1,
           .sampling_value = 0.5},
          {.prefix = {1, 2, 3, 115, 178, 133, 174, 133, 190, 200, 210},
           .mean = 220.01,
           .stddev = 0.1,
           .sampling_value = 0.5},
          {.prefix = {1, 2, 3, 115, 178, 133, 174, 133, 190, 200, 210, 220},
           .mean = 230.01,
           .stddev = 0.1,
           .sampling_value = 0.5},

          // For the second InitInputTokens.
          {.prefix = {4, 5, 6},
           .mean = 200.01,
           .stddev = 0.1,
           .sampling_value = 0.7},
          {.prefix = {4, 5, 6, 200},
           .mean = 321.01,
           .stddev = 3.0,
           .sampling_value = 0.5},
          {.prefix = {4, 5, 6, 200, 320},
           .mean = 442.01,
           .stddev = 0.1,
           .sampling_value = 0.3},
          {.prefix = {4, 5, 6, 200, 320, 442},
           .mean = 563.5,
           .stddev = 5.0,
           .sampling_value = 0.3},
          {.prefix = {4, 5, 6, 200, 320, 442, 565},
           .mean = 684.01,
           .stddev = 0.1,
           .sampling_value = 0.3},
      };
  const std::vector<FakeLlmGpuRunner::TokenInferenceParameters>
      draft_responses = {
          // For the first InitInputTokens.
          {.prefix = {1, 2, 3},
           .mean = 115.01,
           .stddev = 0.1,
           .sampling_value = 0.7},
          {.prefix = {1, 2, 3, 115},
           .mean = 180.01,
           .stddev = 5.0,
           .sampling_value = 0.5},
          {.prefix = {1, 2, 3, 115, 178},
           .mean = 133.01,
           .stddev = 0.1,
           .sampling_value = 0.3},
          {.prefix = {1, 2, 3, 115, 178, 133},
           .mean = 172.5,
           .stddev = 5.0,
           .sampling_value = 0.3},
          {.prefix = {1, 2, 3, 115, 178, 133, 174},
           .mean = 133.01,
           .stddev = 0.1,
           .sampling_value = 0.3},
          // Context is saved and then restored here.
          {.prefix = {1, 2, 3, 115, 178, 133, 174, 133},
           .mean = 190.01,
           .stddev = 0.1,
           .sampling_value = 0.3},
          {.prefix = {1, 2, 3, 115, 178, 133, 174, 133, 190},
           .mean = 200.01,
           .stddev = 0.1,
           .sampling_value = 0.5},
          {.prefix = {1, 2, 3, 115, 178, 133, 174, 133, 190, 200},
           .mean = 210.01,
           .stddev = 0.1,
           .sampling_value = 0.5},
          {.prefix = {1, 2, 3, 115, 178, 133, 174, 133, 190, 200, 210},
           .mean = 220.01,
           .stddev = 0.1,
           .sampling_value = 0.5},
          {.prefix = {1, 2, 3, 115, 178, 133, 174, 133, 190, 200, 210, 220},
           .mean = 230.01,
           .stddev = 0.1,
           .sampling_value = 0.5},

          // For the second InitInputTokens.
          {.prefix = {4, 5, 6},
           .mean = 200.01,
           .stddev = 0.1,
           .sampling_value = 0.7},
          {.prefix = {4, 5, 6, 200},
           .mean = 321.01,
           .stddev = 3.0,
           .sampling_value = 0.5},
          {.prefix = {4, 5, 6, 200, 320},
           .mean = 442.01,
           .stddev = 0.1,
           .sampling_value = 0.3},
          {.prefix = {4, 5, 6, 200, 320, 442},
           .mean = 563.5,
           .stddev = 5.0,
           .sampling_value = 0.3},
          {.prefix = {4, 5, 6, 200, 320, 442, 565},
           .mean = 684.01,
           .stddev = 0.1,
           .sampling_value = 0.3},
      };

  CreateMainRunner(main_responses);
  CreateDrafterRunner(draft_responses);
  CreateSpeculativeRunner(kGamma);

  {
    std::vector<int> prefix = {1, 2, 3};
    std::atomic<bool> cancel = false;
    ASSERT_OK(speculative_runner_->InitInputTokens(prefix, 0, true, &cancel));

    // Attempt to pull less than kGamma + 1 tokens so that their are tokens left
    // in the ready_tokens.
    std::vector<std::vector<int>> output_tokens;
    output_tokens.push_back(std::vector<int>(kGamma, -1));
    ASSERT_OK(speculative_runner_->GetNextTokens(&output_tokens,
                                                /*update_constraint_fn=*/{}));
    ASSERT_EQ(output_tokens.size(), 1);
    EXPECT_THAT(output_tokens[0], ElementsAreArray({115, 178, 133, 174}));
  }

  ASSERT_OK_AND_ASSIGN(auto context_data, speculative_runner_->SaveContext());

  {
    std::vector<int> prefix = {4, 5, 6};
    std::atomic<bool> cancel = false;
    ASSERT_OK(speculative_runner_->InitInputTokens(prefix, 0, true, &cancel));

    // Attempt to pull less than kGamma + 1 tokens so that their are tokens left
    // in the ready_tokens.
    std::vector<std::vector<int>> output_tokens;
    output_tokens.push_back(std::vector<int>(kGamma, -1));
    ASSERT_OK(speculative_runner_->GetNextTokens(&output_tokens,
                                                /*update_constraint_fn=*/{}));
    ASSERT_EQ(output_tokens.size(), 1);
    EXPECT_THAT(output_tokens[0], ElementsAreArray({200, 320, 442, 565}));
  }

  // Restoring the context should resume from the first InitInputTokens.
  ASSERT_OK(speculative_runner_->RestoreContext(std::move(context_data)));
  {
    // Attempt to pull less than kGamma + 1 tokens so that their are tokens left
    // in the ready_tokens.
    std::vector<std::vector<int>> output_tokens;
    output_tokens.push_back(std::vector<int>(kGamma, -1));
    ASSERT_OK(speculative_runner_->GetNextTokens(&output_tokens,
                                                /*update_constraint_fn=*/{}));
    ASSERT_EQ(output_tokens.size(), 1);
    EXPECT_THAT(output_tokens[0], ElementsAreArray({133, 190, 200, 210}));
  }
}

TEST_F(LlmSpeculativeRunnerTest, DisablingSpeculativeDecodingTest) {
  const std::vector<FakeLlmGpuRunner::TokenInferenceParameters> main_responses =
      {
          {.prefix = {1, 2, 3},
           .mean = 115.01,
           .stddev = 0.1,
           .sampling_value = 0.7},
          {.prefix = {1, 2, 3, 115},
           .mean = 180.01,
           .stddev = 3.0,
           .sampling_value = 0.5},
          {.prefix = {1, 2, 3, 115, 179},
           .mean = 133.01,
           .stddev = 0.1,
           .sampling_value = 0.3},
          {.prefix = {1, 2, 3, 115, 179, 133},
           .mean = 172.5,
           .stddev = 5.0,
           .sampling_value = 0.3},
          {.prefix = {1, 2, 3, 115, 179, 133, 174},
           .mean = 133.01,
           .stddev = 0.1,
           .sampling_value = 0.3},
      };


  // Create a drafter runner that returns unimplemented errors. This will ensure
  // that the speculative runner will not invoke any drafter methods if
  // speculative decoding is disabled.
  CreateUnimplementedDrafterRunner();

  CreateMainRunner(main_responses);
  CreateSpeculativeRunner(kGamma);

  // Disable speculative decoding.
  ASSERT_OK(speculative_runner_->EnableSpeculativeDecoding(false));

  std::vector<int> prefix = {1, 2, 3};
  std::atomic<bool> cancel = false;
  ASSERT_OK(speculative_runner_->InitInputTokens(prefix, 0, true, &cancel));

  // Attempt to pull the next Gamma+1 tokens (the first round).
  std::vector<std::vector<int>> output_tokens;
  output_tokens.push_back(std::vector<int>(kGamma + 1, -1));
  ASSERT_OK(speculative_runner_->GetNextTokens(&output_tokens,
                                               /*update_constraint_fn=*/{}));
  ASSERT_EQ(output_tokens.size(), 1);

  // The output should be the same as the main model.
  EXPECT_THAT(output_tokens[0], ElementsAreArray({115, 179, 133, 174, 133}));
}

}  // namespace
}  // namespace odml::infra::gpu
