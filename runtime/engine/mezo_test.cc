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

#include "runtime/engine/mezo.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl

namespace litert::lm {
namespace {

// --- MeZoConfig Tests ---

TEST(MeZoConfigTest, DefaultValues) {
  MeZoConfig config;
  EXPECT_FLOAT_EQ(config.GetLearningRate(), 1e-6f);
  EXPECT_FLOAT_EQ(config.GetEpsilon(), 1e-3f);
  EXPECT_FLOAT_EQ(config.GetWeightDecay(), 0.0f);
  EXPECT_EQ(config.GetSeed(), 0u);
}

TEST(MeZoConfigTest, SettersAndGetters) {
  MeZoConfig config;
  config.SetLearningRate(1e-4f);
  config.SetEpsilon(2e-3f);
  config.SetWeightDecay(0.01f);
  config.SetSeed(42);

  EXPECT_FLOAT_EQ(config.GetLearningRate(), 1e-4f);
  EXPECT_FLOAT_EQ(config.GetEpsilon(), 2e-3f);
  EXPECT_FLOAT_EQ(config.GetWeightDecay(), 0.01f);
  EXPECT_EQ(config.GetSeed(), 42u);
}

// --- MeZoFineTuner Creation Tests ---

TEST(MeZoFineTunerTest, CreateWithValidConfig) {
  MeZoConfig config;
  config.SetLearningRate(1e-5f);
  config.SetEpsilon(1e-3f);
  auto finetuner = MeZoFineTuner::Create(config);
  ASSERT_TRUE(finetuner.ok()) << finetuner.status();
  EXPECT_EQ((*finetuner)->GetStepCount(), 0u);
  EXPECT_FLOAT_EQ((*finetuner)->GetLearningRate(), 1e-5f);
}

TEST(MeZoFineTunerTest, CreateFailsWithNegativeEpsilon) {
  MeZoConfig config;
  config.SetEpsilon(-1e-3f);
  auto finetuner = MeZoFineTuner::Create(config);
  ASSERT_FALSE(finetuner.ok());
  EXPECT_EQ(finetuner.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(MeZoFineTunerTest, CreateFailsWithZeroEpsilon) {
  MeZoConfig config;
  config.SetEpsilon(0.0f);
  auto finetuner = MeZoFineTuner::Create(config);
  ASSERT_FALSE(finetuner.ok());
  EXPECT_EQ(finetuner.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(MeZoFineTunerTest, CreateFailsWithNegativeLearningRate) {
  MeZoConfig config;
  config.SetLearningRate(-1e-5f);
  auto finetuner = MeZoFineTuner::Create(config);
  ASSERT_FALSE(finetuner.ok());
  EXPECT_EQ(finetuner.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(MeZoFineTunerTest, CreateFailsWithNegativeWeightDecay) {
  MeZoConfig config;
  config.SetWeightDecay(-0.01f);
  auto finetuner = MeZoFineTuner::Create(config);
  ASSERT_FALSE(finetuner.ok());
  EXPECT_EQ(finetuner.status().code(), absl::StatusCode::kInvalidArgument);
}

// --- MeZoFineTuner Step Tests ---

TEST(MeZoFineTunerTest, StepUpdatesParameters) {
  MeZoConfig config;
  config.SetLearningRate(1e-2f);
  config.SetEpsilon(1e-3f);
  config.SetSeed(12345);
  auto finetuner = MeZoFineTuner::Create(config);
  ASSERT_TRUE(finetuner.ok()) << finetuner.status();

  // Create a small parameter buffer.
  constexpr size_t kNumElements = 8;
  std::vector<float> weights(kNumElements, 1.0f);
  std::vector<float> original_weights = weights;

  NamedParameter param;
  param.name = "test_weight";
  param.data = weights.data();
  param.num_elements = kNumElements;
  param.is_bias_or_layernorm = false;

  std::vector<NamedParameter> params = {param};

  // Loss function: returns sum of squares of parameters.
  int call_count = 0;
  auto loss_fn = [&]() -> absl::StatusOr<float> {
    ++call_count;
    float loss = 0.0f;
    for (size_t i = 0; i < kNumElements; ++i) {
      loss += weights[i] * weights[i];
    }
    return loss;
  };

  auto result = (*finetuner)->Step(params, std::move(loss_fn));
  ASSERT_TRUE(result.ok()) << result.status();

  // The loss function should have been called exactly twice.
  EXPECT_EQ(call_count, 2);

  // Parameters should have been updated (differ from original).
  bool any_changed = false;
  for (size_t i = 0; i < kNumElements; ++i) {
    if (weights[i] != original_weights[i]) {
      any_changed = true;
      break;
    }
  }
  EXPECT_TRUE(any_changed);

  // Step count should be 1.
  EXPECT_EQ((*finetuner)->GetStepCount(), 1u);

  // Returned loss should be finite and positive.
  EXPECT_TRUE(std::isfinite(*result));
  EXPECT_GT(*result, 0.0f);
}

TEST(MeZoFineTunerTest, StepWithEmptyParametersFails) {
  MeZoConfig config;
  config.SetSeed(1);
  auto finetuner = MeZoFineTuner::Create(config);
  ASSERT_TRUE(finetuner.ok()) << finetuner.status();

  std::vector<NamedParameter> empty_params;
  auto loss_fn = []() -> absl::StatusOr<float> { return 0.0f; };

  auto result = (*finetuner)->Step(empty_params, std::move(loss_fn));
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(MeZoFineTunerTest, StepWithNullDataFails) {
  MeZoConfig config;
  config.SetSeed(1);
  auto finetuner = MeZoFineTuner::Create(config);
  ASSERT_TRUE(finetuner.ok()) << finetuner.status();

  NamedParameter param;
  param.name = "null_param";
  param.data = nullptr;
  param.num_elements = 10;

  std::vector<NamedParameter> params = {param};
  auto loss_fn = []() -> absl::StatusOr<float> { return 0.0f; };

  auto result = (*finetuner)->Step(params, std::move(loss_fn));
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(MeZoFineTunerTest, ReproducibilityWithSameSeed) {
  auto run_step = [](uint64_t seed) -> std::vector<float> {
    MeZoConfig config;
    config.SetLearningRate(1e-2f);
    config.SetEpsilon(1e-3f);
    config.SetSeed(seed);
    auto finetuner = MeZoFineTuner::Create(config);
    if (!finetuner.ok()) return {};

    constexpr size_t kN = 4;
    std::vector<float> weights(kN, 1.0f);

    NamedParameter param;
    param.name = "w";
    param.data = weights.data();
    param.num_elements = kN;
    param.is_bias_or_layernorm = false;

    std::vector<NamedParameter> params = {param};
    auto loss_fn = [&]() -> absl::StatusOr<float> {
      float loss = 0.0f;
      for (size_t i = 0; i < kN; ++i) loss += weights[i] * weights[i];
      return loss;
    };

    auto result = (*finetuner)->Step(params, std::move(loss_fn));
    if (!result.ok()) return {};
    return weights;
  };

  // Two runs with the same seed should produce identical results.
  std::vector<float> run1 = run_step(99999);
  std::vector<float> run2 = run_step(99999);
  ASSERT_FALSE(run1.empty());
  ASSERT_FALSE(run2.empty());
  ASSERT_EQ(run1.size(), run2.size());
  for (size_t i = 0; i < run1.size(); ++i) {
    EXPECT_FLOAT_EQ(run1[i], run2[i]);
  }

  // A different seed should produce different results.
  std::vector<float> run3 = run_step(11111);
  ASSERT_FALSE(run3.empty());
  bool any_different = false;
  for (size_t i = 0; i < run1.size(); ++i) {
    if (run1[i] != run3[i]) {
      any_different = true;
      break;
    }
  }
  EXPECT_TRUE(any_different);
}

TEST(MeZoFineTunerTest, SetLearningRate) {
  MeZoConfig config;
  config.SetLearningRate(1e-5f);
  auto finetuner = MeZoFineTuner::Create(config);
  ASSERT_TRUE(finetuner.ok()) << finetuner.status();

  EXPECT_FLOAT_EQ((*finetuner)->GetLearningRate(), 1e-5f);
  (*finetuner)->SetLearningRate(1e-3f);
  EXPECT_FLOAT_EQ((*finetuner)->GetLearningRate(), 1e-3f);
}

TEST(MeZoFineTunerTest, WeightDecaySkippedForBiasParams) {
  // Run with weight decay and compare bias vs non-bias parameter updates.
  MeZoConfig config;
  config.SetLearningRate(1e-2f);
  config.SetEpsilon(1e-3f);
  config.SetWeightDecay(0.1f);
  config.SetSeed(42);

  constexpr size_t kN = 4;

  // Run with is_bias_or_layernorm = false (weight decay applied).
  auto finetuner1 = MeZoFineTuner::Create(config);
  ASSERT_TRUE(finetuner1.ok());
  std::vector<float> weights_wd(kN, 2.0f);
  NamedParameter param_wd;
  param_wd.name = "dense";
  param_wd.data = weights_wd.data();
  param_wd.num_elements = kN;
  param_wd.is_bias_or_layernorm = false;
  std::vector<NamedParameter> params_wd = {param_wd};
  auto loss_fn_wd = [&]() -> absl::StatusOr<float> {
    float l = 0;
    for (size_t i = 0; i < kN; ++i) l += weights_wd[i] * weights_wd[i];
    return l;
  };
  auto r1 = (*finetuner1)->Step(params_wd, std::move(loss_fn_wd));
  ASSERT_TRUE(r1.ok());

  // Run with is_bias_or_layernorm = true (weight decay NOT applied).
  auto finetuner2 = MeZoFineTuner::Create(config);
  ASSERT_TRUE(finetuner2.ok());
  std::vector<float> weights_no_wd(kN, 2.0f);
  NamedParameter param_no_wd;
  param_no_wd.name = "bias";
  param_no_wd.data = weights_no_wd.data();
  param_no_wd.num_elements = kN;
  param_no_wd.is_bias_or_layernorm = true;
  std::vector<NamedParameter> params_no_wd = {param_no_wd};
  auto loss_fn_no_wd = [&]() -> absl::StatusOr<float> {
    float l = 0;
    for (size_t i = 0; i < kN; ++i) l += weights_no_wd[i] * weights_no_wd[i];
    return l;
  };
  auto r2 = (*finetuner2)->Step(params_no_wd, std::move(loss_fn_no_wd));
  ASSERT_TRUE(r2.ok());

  // The updates should differ because weight decay is only applied to
  // the non-bias parameter.
  bool any_different = false;
  for (size_t i = 0; i < kN; ++i) {
    if (weights_wd[i] != weights_no_wd[i]) {
      any_different = true;
      break;
    }
  }
  EXPECT_TRUE(any_different);
}

TEST(MeZoFineTunerTest, LossFunctionFailureRestoresParameters) {
  MeZoConfig config;
  config.SetLearningRate(1e-2f);
  config.SetEpsilon(1e-3f);
  config.SetSeed(42);
  auto finetuner = MeZoFineTuner::Create(config);
  ASSERT_TRUE(finetuner.ok()) << finetuner.status();

  constexpr size_t kN = 4;
  std::vector<float> weights(kN, 1.5f);
  std::vector<float> original_weights = weights;

  NamedParameter param;
  param.name = "w";
  param.data = weights.data();
  param.num_elements = kN;
  param.is_bias_or_layernorm = false;

  std::vector<NamedParameter> params = {param};

  // Loss function that fails on the first call (positive perturbation).
  auto loss_fn = []() -> absl::StatusOr<float> {
    return absl::InternalError("loss computation failed");
  };

  auto result = (*finetuner)->Step(params, std::move(loss_fn));
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), absl::StatusCode::kInternal);

  // Parameters should be restored to their original values.
  for (size_t i = 0; i < kN; ++i) {
    EXPECT_FLOAT_EQ(weights[i], original_weights[i]);
  }
}

TEST(MeZoFineTunerTest, MultipleStepsDecreaseLoss) {
  MeZoConfig config;
  config.SetLearningRate(1e-1f);
  config.SetEpsilon(1e-3f);
  config.SetSeed(42);
  auto finetuner = MeZoFineTuner::Create(config);
  ASSERT_TRUE(finetuner.ok()) << finetuner.status();

  // Simple quadratic loss: L = sum(w^2). Minimum at w=0.
  constexpr size_t kN = 4;
  std::vector<float> weights(kN, 5.0f);

  NamedParameter param;
  param.name = "w";
  param.data = weights.data();
  param.num_elements = kN;
  param.is_bias_or_layernorm = false;

  std::vector<NamedParameter> params = {param};

  float first_loss = 0.0f;
  float last_loss = 0.0f;

  // Run 50 steps. With a quadratic loss and reasonable lr, loss should
  // generally decrease over many steps (though individual steps may not).
  for (int step = 0; step < 50; ++step) {
    auto loss_fn = [&]() -> absl::StatusOr<float> {
      float l = 0.0f;
      for (size_t i = 0; i < kN; ++i) l += weights[i] * weights[i];
      return l;
    };
    auto result = (*finetuner)->Step(params, std::move(loss_fn));
    ASSERT_TRUE(result.ok()) << result.status();
    if (step == 0) first_loss = *result;
    last_loss = *result;
  }

  EXPECT_EQ((*finetuner)->GetStepCount(), 50u);

  // After 50 steps of MeZO on a quadratic, the loss should have decreased.
  // Compute current loss directly (the returned loss is from the perturbed
  // point, so re-evaluate at the final parameters).
  float final_loss = 0.0f;
  for (size_t i = 0; i < kN; ++i) {
    final_loss += weights[i] * weights[i];
  }
  EXPECT_LT(final_loss, first_loss);
}

}  // namespace
}  // namespace litert::lm
