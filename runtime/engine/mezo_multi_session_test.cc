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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/components/lora_manager.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/mezo.h"

namespace litert::lm {
namespace {

// Simulates three sessions using MeZO and LoRA together:
//
//   Session A: MeZO training - perturbs and updates its own LoRA weight buffers.
//   Session B: LoRA inference - holds a separate pre-trained LoRA.
//   Session C: Base model - no LoRA adapter active.
//
// Verifies:
//   1. MeZO modifies only Session A's weight buffers.
//   2. Session B's LoRA buffers remain unchanged throughout.
//   3. Session C has no LoRA (lora_id is nullopt).
//   4. SessionConfig correctly carries lora_id per session.

constexpr size_t kNumElements = 8;

class MeZoMultiSessionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Session A: MeZO training LoRA buffers (start at 1.0).
    training_weights_.assign(kNumElements, 1.0f);

    // Session B: pre-trained LoRA buffers (start at 0.5).
    inference_weights_.assign(kNumElements, 0.5f);
    inference_weights_snapshot_ = inference_weights_;
  }

  std::vector<float> training_weights_;
  std::vector<float> inference_weights_;
  std::vector<float> inference_weights_snapshot_;
};

TEST_F(MeZoMultiSessionTest, ThreeSessionsWithLoraIsolation) {
  // --- Session configs carry per-session LoRA IDs ---
  auto config_a = SessionConfig::CreateDefault();
  ASSERT_TRUE(config_a.ok()) << config_a.status();
  config_a->SetLoraId(2);  // training LoRA

  auto config_b = SessionConfig::CreateDefault();
  ASSERT_TRUE(config_b.ok()) << config_b.status();
  config_b->SetLoraId(1);  // pre-trained LoRA

  auto config_c = SessionConfig::CreateDefault();
  ASSERT_TRUE(config_c.ok()) << config_c.status();
  // Session C: no LoRA (default nullopt)

  EXPECT_EQ(config_a->GetLoraId(), std::optional<uint32_t>(2));
  EXPECT_EQ(config_b->GetLoraId(), std::optional<uint32_t>(1));
  EXPECT_EQ(config_c->GetLoraId(), std::nullopt);

  // --- Session A: MeZO training on its LoRA buffers ---
  MeZoConfig mezo_config;
  mezo_config.SetLearningRate(1e-2f);
  mezo_config.SetEpsilon(1e-3f);
  mezo_config.SetSeed(42);
  auto finetuner = MeZoFineTuner::Create(mezo_config);
  ASSERT_TRUE(finetuner.ok()) << finetuner.status();

  NamedParameter training_param;
  training_param.name = "lora_a.query_weight";
  training_param.data = training_weights_.data();
  training_param.num_elements = kNumElements;
  training_param.is_bias_or_layernorm = false;

  std::vector<NamedParameter> training_params = {training_param};

  // MeZO step: loss = sum(w^2), called twice (perturbed + and -).
  auto loss_fn = [this]() -> absl::StatusOr<float> {
    float loss = 0.0f;
    for (size_t i = 0; i < kNumElements; ++i) {
      loss += training_weights_[i] * training_weights_[i];
    }
    return loss;
  };

  auto result = (*finetuner)->Step(training_params, std::move(loss_fn));
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ((*finetuner)->GetStepCount(), 1u);

  // Training weights should have been updated.
  bool training_weights_changed = false;
  for (size_t i = 0; i < kNumElements; ++i) {
    if (training_weights_[i] != 1.0f) {
      training_weights_changed = true;
      break;
    }
  }
  EXPECT_TRUE(training_weights_changed)
      << "MeZO should have updated Session A's training weights";

  // --- Session B: inference LoRA buffers are untouched ---
  for (size_t i = 0; i < kNumElements; ++i) {
    EXPECT_FLOAT_EQ(inference_weights_[i], inference_weights_snapshot_[i])
        << "Session B's LoRA weights should not have been modified at index "
        << i;
  }

  // --- Session C: no LoRA adapter ---
  EXPECT_EQ(config_c->GetLoraId(), std::nullopt);
}

TEST_F(MeZoMultiSessionTest, MultipleStepsOnlyAffectTrainingSession) {
  MeZoConfig mezo_config;
  mezo_config.SetLearningRate(1e-1f);
  mezo_config.SetEpsilon(1e-3f);
  mezo_config.SetSeed(123);
  auto finetuner = MeZoFineTuner::Create(mezo_config);
  ASSERT_TRUE(finetuner.ok()) << finetuner.status();

  NamedParameter training_param;
  training_param.name = "lora_a.query_weight";
  training_param.data = training_weights_.data();
  training_param.num_elements = kNumElements;
  training_param.is_bias_or_layernorm = false;

  std::vector<NamedParameter> training_params = {training_param};

  // Run 10 MeZO steps. After each step, verify Session B is unchanged.
  for (int step = 0; step < 10; ++step) {
    auto loss_fn = [this]() -> absl::StatusOr<float> {
      float loss = 0.0f;
      for (size_t i = 0; i < kNumElements; ++i) {
        loss += training_weights_[i] * training_weights_[i];
      }
      return loss;
    };

    auto result = (*finetuner)->Step(training_params, std::move(loss_fn));
    ASSERT_TRUE(result.ok()) << "Step " << step << " failed: "
                             << result.status();

    // Session B's buffers must remain unchanged after every step.
    for (size_t i = 0; i < kNumElements; ++i) {
      EXPECT_FLOAT_EQ(inference_weights_[i], inference_weights_snapshot_[i])
          << "Session B changed at step " << step << ", index " << i;
    }
  }

  EXPECT_EQ((*finetuner)->GetStepCount(), 10u);

  // After 10 steps of MeZO on quadratic loss, training weights should have
  // decreased toward zero.
  float final_loss = 0.0f;
  for (size_t i = 0; i < kNumElements; ++i) {
    final_loss += training_weights_[i] * training_weights_[i];
  }
  float initial_loss = static_cast<float>(kNumElements);  // sum(1.0^2)
  EXPECT_LT(final_loss, initial_loss)
      << "Training loss should decrease after 10 MeZO steps";
}

TEST_F(MeZoMultiSessionTest, SessionConfigLoraIdPersistsThroughCopy) {
  auto config = SessionConfig::CreateDefault();
  ASSERT_TRUE(config.ok()) << config.status();
  config->SetLoraId(42);

  // Copy the config.
  SessionConfig copied = *config;
  EXPECT_EQ(copied.GetLoraId(), std::optional<uint32_t>(42));

  // Modifying the copy should not affect the original.
  copied.SetLoraId(99);
  EXPECT_EQ(config->GetLoraId(), std::optional<uint32_t>(42));
  EXPECT_EQ(copied.GetLoraId(), std::optional<uint32_t>(99));

  // Clearing LoRA on a session.
  copied.SetLoraId(std::nullopt);
  EXPECT_EQ(copied.GetLoraId(), std::nullopt);
}

}  // namespace
}  // namespace litert::lm
