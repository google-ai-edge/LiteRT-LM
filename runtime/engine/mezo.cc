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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl

namespace litert::lm {

// Pimpl implementation of MeZoFineTuner.
class MeZoFineTuner::Impl {
 public:
  explicit Impl(const MeZoConfig& config)
      : learning_rate_(config.GetLearningRate()),
        epsilon_(config.GetEpsilon()),
        weight_decay_(config.GetWeightDecay()),
        step_count_(0) {
    if (config.GetSeed() != 0) {
      global_rng_.seed(config.GetSeed());
    } else {
      std::random_device rd;
      global_rng_.seed(rd());
    }
  }

  absl::StatusOr<float> Step(
      const std::vector<NamedParameter>& parameters,
      absl::AnyInvocable<absl::StatusOr<float>()> loss_fn) {
    if (parameters.empty()) {
      return absl::InvalidArgumentError("Parameters must not be empty.");
    }
    for (const auto& param : parameters) {
      if (param.data == nullptr || param.num_elements == 0) {
        return absl::InvalidArgumentError(
            "Parameter data must not be null and num_elements must be > 0.");
      }
    }

    // Sample a random seed for this step's perturbation vector.
    uint64_t step_seed = global_rng_();

    // Step 1: Perturb parameters positively: theta += eps * z.
    PerturbParameters(parameters, epsilon_, step_seed);

    // Step 2: Evaluate loss at theta + eps * z.
    absl::StatusOr<float> loss_plus = loss_fn();
    if (!loss_plus.ok()) {
      // Restore parameters before returning error.
      PerturbParameters(parameters, -epsilon_, step_seed);
      return loss_plus.status();
    }

    // Step 3: Perturb parameters negatively: theta -= 2 * eps * z.
    // Net effect from original: theta - eps * z.
    PerturbParameters(parameters, -2.0f * epsilon_, step_seed);

    // Step 4: Evaluate loss at theta - eps * z.
    absl::StatusOr<float> loss_minus = loss_fn();
    if (!loss_minus.ok()) {
      // Restore parameters before returning error.
      PerturbParameters(parameters, epsilon_, step_seed);
      return loss_minus.status();
    }

    // Step 5: Restore parameters to original: theta += eps * z.
    PerturbParameters(parameters, epsilon_, step_seed);

    // Step 6: Compute the projected gradient scalar.
    float projected_grad = (*loss_plus - *loss_minus) / (2.0f * epsilon_);

    // Step 7: Update parameters using the same seed to regenerate z.
    UpdateParameters(parameters, projected_grad, step_seed);

    ++step_count_;

    ABSL_LOG(INFO) << "MeZO step " << step_count_
                   << ": loss=" << *loss_plus
                   << ", projected_grad=" << projected_grad;

    return *loss_plus;
  }

  uint64_t GetStepCount() const { return step_count_; }

  void SetLearningRate(float learning_rate) { learning_rate_ = learning_rate; }

  float GetLearningRate() const { return learning_rate_; }

 private:
  // Perturbs all parameters in-place: param.data[i] += scale * z[i],
  // where z[i] ~ N(0, 1) generated from the given seed.
  void PerturbParameters(const std::vector<NamedParameter>& parameters,
                         float scale, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (const auto& param : parameters) {
      for (size_t i = 0; i < param.num_elements; ++i) {
        param.data[i] += scale * dist(rng);
      }
    }
  }

  // Updates parameters: theta_i -= lr * (projected_grad * z_i + wd * theta_i),
  // where z[i] is regenerated from the same seed used during perturbation.
  // Weight decay is skipped for bias and layer normalization parameters.
  void UpdateParameters(const std::vector<NamedParameter>& parameters,
                        float projected_grad, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (const auto& param : parameters) {
      if (!param.is_bias_or_layernorm && weight_decay_ != 0.0f) {
        for (size_t i = 0; i < param.num_elements; ++i) {
          float z = dist(rng);
          param.data[i] -= learning_rate_ *
              (projected_grad * z + weight_decay_ * param.data[i]);
        }
      } else {
        for (size_t i = 0; i < param.num_elements; ++i) {
          float z = dist(rng);
          param.data[i] -= learning_rate_ * projected_grad * z;
        }
      }
    }
  }

  float learning_rate_;
  float epsilon_;
  float weight_decay_;
  uint64_t step_count_;
  std::mt19937_64 global_rng_;
};

MeZoFineTuner::MeZoFineTuner(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

MeZoFineTuner::~MeZoFineTuner() = default;

absl::StatusOr<std::unique_ptr<MeZoFineTuner>> MeZoFineTuner::Create(
    const MeZoConfig& config) {
  if (config.GetEpsilon() <= 0.0f) {
    return absl::InvalidArgumentError("Epsilon must be positive.");
  }
  if (config.GetLearningRate() <= 0.0f) {
    return absl::InvalidArgumentError("Learning rate must be positive.");
  }
  if (config.GetWeightDecay() < 0.0f) {
    return absl::InvalidArgumentError("Weight decay must be non-negative.");
  }
  return absl::WrapUnique(
      new MeZoFineTuner(std::make_unique<Impl>(config)));
}

absl::StatusOr<float> MeZoFineTuner::Step(
    const std::vector<NamedParameter>& parameters,
    absl::AnyInvocable<absl::StatusOr<float>()> loss_fn) {
  return impl_->Step(parameters, std::move(loss_fn));
}

uint64_t MeZoFineTuner::GetStepCount() const {
  return impl_->GetStepCount();
}

void MeZoFineTuner::SetLearningRate(float learning_rate) {
  impl_->SetLearningRate(learning_rate);
}

float MeZoFineTuner::GetLearningRate() const {
  return impl_->GetLearningRate();
}

}  // namespace litert::lm
