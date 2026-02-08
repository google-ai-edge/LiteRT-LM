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
        use_conmezo_(config.GetUseConMeZo()),
        momentum_decay_(config.GetMomentumDecay()),
        cone_angle_(config.GetConeAngle()),
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

    // Determine total parameter count for ConMeZO momentum initialization.
    if (use_conmezo_ && !momentum_initialized_) {
      size_t total_elements = 0;
      for (const auto& param : parameters) {
        total_elements += param.num_elements;
      }
      momentum_.assign(total_elements, 0.0f);
      momentum_initialized_ = true;
    }

    // Choose perturbation method based on ConMeZO mode.
    const bool use_cone = use_conmezo_ && MomentumNonZero();

    // Step 1: Perturb parameters positively: theta += eps * z.
    if (use_cone) {
      ConePerturbParameters(parameters, epsilon_, step_seed);
    } else {
      PerturbParameters(parameters, epsilon_, step_seed);
    }

    // Step 2: Evaluate loss at theta + eps * z.
    absl::StatusOr<float> loss_plus = loss_fn();
    if (!loss_plus.ok()) {
      // Restore parameters before returning error.
      if (use_cone) {
        ConePerturbParameters(parameters, -epsilon_, step_seed);
      } else {
        PerturbParameters(parameters, -epsilon_, step_seed);
      }
      return loss_plus.status();
    }

    // Step 3: Perturb parameters negatively: theta -= 2 * eps * z.
    // Net effect from original: theta - eps * z.
    if (use_cone) {
      ConePerturbParameters(parameters, -2.0f * epsilon_, step_seed);
    } else {
      PerturbParameters(parameters, -2.0f * epsilon_, step_seed);
    }

    // Step 4: Evaluate loss at theta - eps * z.
    absl::StatusOr<float> loss_minus = loss_fn();
    if (!loss_minus.ok()) {
      // Restore parameters before returning error.
      if (use_cone) {
        ConePerturbParameters(parameters, epsilon_, step_seed);
      } else {
        PerturbParameters(parameters, epsilon_, step_seed);
      }
      return loss_minus.status();
    }

    // Step 5: Restore parameters to original: theta += eps * z.
    if (use_cone) {
      ConePerturbParameters(parameters, epsilon_, step_seed);
    } else {
      PerturbParameters(parameters, epsilon_, step_seed);
    }

    // Step 6: Compute the projected gradient scalar.
    float projected_grad = (*loss_plus - *loss_minus) / (2.0f * epsilon_);

    // Step 7: Update parameters and (if ConMeZO) update momentum.
    if (use_conmezo_) {
      ConeUpdateParameters(parameters, projected_grad, step_seed, use_cone);
    } else {
      UpdateParameters(parameters, projected_grad, step_seed);
    }

    ++step_count_;

    ABSL_LOG(INFO) << "MeZO step " << step_count_
                   << ": loss=" << *loss_plus
                   << ", projected_grad=" << projected_grad
                   << (use_conmezo_ ? " [ConMeZO]" : "");

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

  // Returns true if any element of the momentum vector is non-zero.
  bool MomentumNonZero() const {
    for (float m : momentum_) {
      if (m != 0.0f) return true;
    }
    return false;
  }

  // Computes the L2 norm of the momentum vector.
  float MomentumNorm() const {
    float sum_sq = 0.0f;
    for (float m : momentum_) {
      sum_sq += m * m;
    }
    return std::sqrt(sum_sq);
  }

  // Cone-constrained perturbation using three-pass seed replay.
  //
  // The perturbation direction is biased toward the momentum direction:
  //   z = cos(theta) * m_hat + sin(theta) * z_perp_hat
  // where z_perp = z_raw - (z_raw . m_hat) * m_hat is the component of
  // random noise orthogonal to the momentum. Three passes over the RNG
  // stream avoid allocating a d-sized temporary vector.
  void ConePerturbParameters(const std::vector<NamedParameter>& parameters,
                             float scale, uint64_t seed) {
    const float m_norm = MomentumNorm();
    if (m_norm == 0.0f) {
      PerturbParameters(parameters, scale, seed);
      return;
    }
    const float inv_m_norm = 1.0f / m_norm;
    const float cos_a = std::cos(cone_angle_);
    const float sin_a = std::sin(cone_angle_);

    // Pass 1: Compute dot(z_raw, m_hat).
    float dot = 0.0f;
    {
      std::mt19937_64 rng(seed);
      std::normal_distribution<float> dist(0.0f, 1.0f);
      size_t m_idx = 0;
      for (const auto& param : parameters) {
        for (size_t i = 0; i < param.num_elements; ++i) {
          float z_raw = dist(rng);
          dot += z_raw * momentum_[m_idx] * inv_m_norm;
          ++m_idx;
        }
      }
    }

    // Pass 2: Compute ||z_perp|| where z_perp = z_raw - dot * m_hat.
    float z_perp_norm_sq = 0.0f;
    {
      std::mt19937_64 rng(seed);
      std::normal_distribution<float> dist(0.0f, 1.0f);
      size_t m_idx = 0;
      for (const auto& param : parameters) {
        for (size_t i = 0; i < param.num_elements; ++i) {
          float z_raw = dist(rng);
          float m_hat_i = momentum_[m_idx] * inv_m_norm;
          float z_perp_i = z_raw - dot * m_hat_i;
          z_perp_norm_sq += z_perp_i * z_perp_i;
          ++m_idx;
        }
      }
    }
    const float z_perp_norm = std::sqrt(z_perp_norm_sq);
    const float inv_z_perp_norm =
        (z_perp_norm > 1e-10f) ? (1.0f / z_perp_norm) : 0.0f;

    // Pass 3: Apply z = cos(a) * m_hat + sin(a) * z_perp_hat, scaled.
    {
      std::mt19937_64 rng(seed);
      std::normal_distribution<float> dist(0.0f, 1.0f);
      size_t m_idx = 0;
      for (const auto& param : parameters) {
        for (size_t i = 0; i < param.num_elements; ++i) {
          float z_raw = dist(rng);
          float m_hat_i = momentum_[m_idx] * inv_m_norm;
          float z_perp_i = z_raw - dot * m_hat_i;
          float z_cone = cos_a * m_hat_i + sin_a * z_perp_i * inv_z_perp_norm;
          param.data[i] += scale * z_cone;
          ++m_idx;
        }
      }
    }
  }

  // ConMeZO parameter update with momentum EMA.
  // Updates parameters using the cone-perturbation direction z (or vanilla z
  // if momentum was zero at perturbation time), and updates the momentum
  // vector with the projected gradient.
  void ConeUpdateParameters(const std::vector<NamedParameter>& parameters,
                            float projected_grad, uint64_t seed,
                            bool used_cone) {
    const float m_norm = MomentumNorm();
    const float inv_m_norm = (m_norm > 1e-10f) ? (1.0f / m_norm) : 0.0f;

    float cos_a = std::cos(cone_angle_);
    float sin_a = std::sin(cone_angle_);

    // For cone mode, we need dot and z_perp_norm (same as ConePerturbParameters).
    float dot = 0.0f;
    float z_perp_norm = 1.0f;

    if (used_cone) {
      // Pass 1: dot
      {
        std::mt19937_64 rng(seed);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        size_t m_idx = 0;
        for (const auto& param : parameters) {
          for (size_t i = 0; i < param.num_elements; ++i) {
            float z_raw = dist(rng);
            dot += z_raw * momentum_[m_idx] * inv_m_norm;
            ++m_idx;
          }
        }
      }

      // Pass 2: z_perp_norm
      float z_perp_norm_sq = 0.0f;
      {
        std::mt19937_64 rng(seed);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        size_t m_idx = 0;
        for (const auto& param : parameters) {
          for (size_t i = 0; i < param.num_elements; ++i) {
            float z_raw = dist(rng);
            float m_hat_i = momentum_[m_idx] * inv_m_norm;
            float z_perp_i = z_raw - dot * m_hat_i;
            z_perp_norm_sq += z_perp_i * z_perp_i;
            ++m_idx;
          }
        }
      }
      z_perp_norm = std::sqrt(z_perp_norm_sq);
    }

    const float inv_z_perp_norm =
        (z_perp_norm > 1e-10f) ? (1.0f / z_perp_norm) : 0.0f;

    // Pass 3 (or single pass for vanilla): Update parameters and momentum.
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    size_t m_idx = 0;
    for (const auto& param : parameters) {
      for (size_t i = 0; i < param.num_elements; ++i) {
        float z_raw = dist(rng);
        float z;
        if (used_cone) {
          float m_hat_i = momentum_[m_idx] * inv_m_norm;
          float z_perp_i = z_raw - dot * m_hat_i;
          z = cos_a * m_hat_i + sin_a * z_perp_i * inv_z_perp_norm;
        } else {
          z = z_raw;
        }

        // Update momentum: EMA of projected_grad * z.
        momentum_[m_idx] = momentum_decay_ * momentum_[m_idx] +
                           (1.0f - momentum_decay_) * projected_grad * z;

        // Update parameter.
        float update = projected_grad * z;
        if (!param.is_bias_or_layernorm && weight_decay_ != 0.0f) {
          update += weight_decay_ * param.data[i];
        }
        param.data[i] -= learning_rate_ * update;

        ++m_idx;
      }
    }
  }

  float learning_rate_;
  float epsilon_;
  float weight_decay_;
  bool use_conmezo_;
  float momentum_decay_;
  float cone_angle_;
  uint64_t step_count_;
  std::mt19937_64 global_rng_;

  // ConMeZO state.
  std::vector<float> momentum_;
  bool momentum_initialized_ = false;
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
  if (config.GetMomentumDecay() < 0.0f || config.GetMomentumDecay() > 1.0f) {
    return absl::InvalidArgumentError(
        "Momentum decay must be in [0, 1].");
  }
  constexpr float kPiOver2 = 1.5707963f;
  if (config.GetConeAngle() < 0.0f || config.GetConeAngle() > kPiOver2) {
    return absl::InvalidArgumentError(
        "Cone angle must be in [0, pi/2].");
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
