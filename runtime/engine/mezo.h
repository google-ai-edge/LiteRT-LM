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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_MEZO_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_MEZO_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl

namespace litert::lm {

// Configuration for MeZO (Memory-efficient Zeroth-Order) fine-tuning.
//
// MeZO estimates gradients using only forward passes, achieving the same
// memory footprint as inference. See: "Fine-Tuning Language Models with Just
// Forward Passes" (Malladi et al., NeurIPS 2023).
//
// Example:
//   MeZoConfig config;
//   config.SetLearningRate(1e-7f);
//   config.SetEpsilon(1e-3f);
//   auto finetuner = MeZoFineTuner::Create(config);
class MeZoConfig {
 public:
  MeZoConfig() = default;

  // Learning rate for parameter updates.
  float GetLearningRate() const { return learning_rate_; }
  void SetLearningRate(float learning_rate) { learning_rate_ = learning_rate; }

  // Perturbation scale for finite difference gradient estimation.
  float GetEpsilon() const { return epsilon_; }
  void SetEpsilon(float epsilon) { epsilon_ = epsilon; }

  // Weight decay coefficient applied to non-bias, non-layernorm parameters.
  float GetWeightDecay() const { return weight_decay_; }
  void SetWeightDecay(float weight_decay) { weight_decay_ = weight_decay; }

  // Random seed for reproducibility. A value of 0 uses a random seed.
  uint64_t GetSeed() const { return seed_; }
  void SetSeed(uint64_t seed) { seed_ = seed; }

 private:
  float learning_rate_ = 1e-6f;
  float epsilon_ = 1e-3f;
  float weight_decay_ = 0.0f;
  uint64_t seed_ = 0;
};

// A named parameter buffer for MeZO optimization. Represents a contiguous
// block of float32 model weights that can be perturbed and updated.
struct NamedParameter {
  // Name of the parameter (e.g., "attention.query_weight_0").
  std::string name;

  // Pointer to the mutable weight data. Must remain valid for the duration
  // of the MeZoFineTuner::Step call.
  float* data = nullptr;

  // Number of float elements in the parameter.
  size_t num_elements = 0;

  // Whether this parameter is a bias or layer normalization weight. When true,
  // weight decay is not applied to this parameter during updates.
  bool is_bias_or_layernorm = false;
};

// MeZO (Memory-efficient Zeroth-Order) fine-tuner for on-device LLMs.
//
// Implements the SPSA (Simultaneous Perturbation Stochastic Approximation)
// gradient estimator: perturbs model parameters with random Gaussian noise,
// evaluates the loss at two points, and estimates the gradient from the
// difference. The full perturbation vector is never stored; instead, a random
// seed is saved and used to regenerate the perturbations as needed.
//
// Memory overhead per step: O(1) (one scalar seed + one scalar gradient).
//
// Example:
//   MeZoConfig config;
//   config.SetLearningRate(1e-7f);
//   ASSIGN_OR_RETURN(auto finetuner, MeZoFineTuner::Create(config));
//
//   // Collect trainable parameters.
//   std::vector<NamedParameter> params = GetTrainableParameters(model);
//
//   // Run one optimization step.
//   auto loss_fn = [&]() -> absl::StatusOr<float> {
//     return ComputeLoss(session, input_data);
//   };
//   ASSIGN_OR_RETURN(float loss, finetuner->Step(params, std::move(loss_fn)));
class MeZoFineTuner {
 public:
  ~MeZoFineTuner();

  // Creates a MeZoFineTuner with the given configuration.
  static absl::StatusOr<std::unique_ptr<MeZoFineTuner>> Create(
      const MeZoConfig& config);

  // Performs one MeZO optimization step on the given parameters.
  //
  // The loss function is called twice per step (once with positive
  // perturbation, once with negative perturbation). Parameters are restored
  // to their original values before the update is applied.
  //
  // Returns the loss from the positive perturbation (f(theta + eps*z)).
  absl::StatusOr<float> Step(
      const std::vector<NamedParameter>& parameters,
      absl::AnyInvocable<absl::StatusOr<float>()> loss_fn);

  // Returns the number of optimization steps completed.
  uint64_t GetStepCount() const;

  // Updates the learning rate (e.g., for scheduling).
  void SetLearningRate(float learning_rate);

  // Returns the current learning rate.
  float GetLearningRate() const;

 private:
  class Impl;
  explicit MeZoFineTuner(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_MEZO_H_
