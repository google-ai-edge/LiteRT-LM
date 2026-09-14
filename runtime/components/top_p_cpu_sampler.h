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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_TOP_P_CPU_SAMPLER_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_TOP_P_CPU_SAMPLER_H_

#include <memory>
#include <random>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/sampler.h"
#include "runtime/proto/sampler_params.pb.h"

namespace litert::lm {

class TopPSampler : public Sampler {
 public:
  // Creates a TopPSampler with the given input arguments:
  // - k: The number of top logits to consider.
  // - p: The top-p probability mass to consider.
  // - batch_size: The batch size of the input logits.
  // - sequence_size: The sequence length of the input logits.
  // - compute_exact_log_probs: Optional. If true, computes exact
  // full-vocabulary token log-probabilities when scores_tensor is requested.
  // Defaults to false to avoid penalizing other workloads.
  static absl::StatusOr<std::unique_ptr<TopPSampler>> Create(
      int k, float p, float temperature, int batch_size, int sequence_size,
      int seed, bool compute_exact_log_probs = false);

  // Given a batch of logits, samples a batch of token ids.
  // The expected shape of the logits is [batch_size, sequence_size,
  // vocab_size]. The output is a 2D litert::TensorBuffer of shape [batch_size,
  // sequence_size]. The scores_tensor is optional. If it is not nullptr, the
  // sampled scores are also written to it (in the same shape as the
  // ids_tensor). The scores are the log of the probability of the sampled
  // token.
  absl::Status SampleToIdAndScoreBuffer(const TensorBuffer& logits_tensor,
                                        TensorBuffer& ids_tensor,
                                        TensorBuffer* scores_tensor) override;

  // Updates the configs of the sampler.
  absl::Status UpdateConfig(
      const proto::SamplerParameters& sampler_params, int batch_size,
      std::shared_ptr<std::default_random_engine> rand_gen) override;

  // Sets whether to compute exact full-vocabulary token log-probabilities.
  void SetComputeExactLogProbs(bool compute_exact_log_probs) {
    compute_exact_log_probs_ = compute_exact_log_probs;
  }
  bool ComputeExactLogProbs() const { return compute_exact_log_probs_; }

 private:
  explicit TopPSampler(int k, float p, float temperature, int batch_size,
                       int sequence_size, int seed,
                       bool compute_exact_log_probs = false)
      : k_(k),
        p_(p),
        temperature_(temperature),
        batch_size_(batch_size),
        sequence_size_(sequence_size),
        compute_exact_log_probs_(compute_exact_log_probs) {
    generator_ = std::make_shared<std::default_random_engine>(seed);
  }

  // The parameters for the sampler.
  int k_;
  float p_;
  float temperature_;
  int batch_size_;
  int sequence_size_;
  bool compute_exact_log_probs_ = false;
  std::shared_ptr<std::default_random_engine> generator_;

  // The logits data to be used for sampling. Having it as a member to avoid
  // re-allocating the vector for each sampling call.
  std::vector<float> logits_data_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_TOP_P_CPU_SAMPLER_H_
