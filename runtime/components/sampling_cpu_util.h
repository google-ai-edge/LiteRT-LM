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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_SAMPLING_CPU_UTIL_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_SAMPLING_CPU_UTIL_H_

#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl

namespace litert::lm {

// Computes the top k token ids (a.k.a. indices of the given logits).
//   - logits: a 3D tensor (in a flattened buffer) of shape
//     [batch_size, sequence_size, vocab_size].
//   - k: the number of top k.
//   - batch_size: the batch size of the logits.
//   - sequence_size: the sequence length of the logits.
// The output is a vector of token ids of shape
//   [batch_size, sequence_size * k].
absl::StatusOr<std::vector<std::vector<int>>> TopKTokenIds(
    absl::Span<const float> logits, int k, int batch_size = 1,
    int sequence_size = 1);

// Computes the softmax of the given logits.
//   - logits: a 3D tensor (in a flattened buffer) of shape
//     [batch_size, sequence_size, vocab_size].
//   - topk_token_ids: a 3D tensor (in a flattened buffer) of shape
//     [batch_size, sequence_size, k]. The token ids of the top k logits.
//   - temperature: the temperature of the softmax.
//   - batch_size: the batch size of the logits.
//   - sequence_size: the sequence length of the logits.
//   - max_logit_values: this is an output parameter to store the max logit
//     values of each batch and sequence. It is a vector of shape [batch_size,
//     sequence_size].
// The output is a vector of probabilities of shape
//     [batch_size, sequence_size * vocab_size].
absl::StatusOr<std::vector<std::vector<float>>> Softmax(
    absl::Span<const float> logits, absl::Span<const int> topk_token_ids,
    float temperature, int batch_size, int sequence_size,
    std::vector<std::vector<float>>& max_logit_values);

// Samples a batch of token ids from the given probabilities.
//   - logits: a 3D tensor (in a flattened buffer) of shape
//     [batch_size, sequence_size, vocab_size].
//   - k: the number of top k.
//   - p: the probability threshold use by Top-P sampling.
//   - temperature: the temperature used for calculating the softmax.
//   - uniforms: batch_size * sequence_size values in [0, 1). The token of
//     (batch b, position s) is the one at which the cumulative probability of
//     the nucleus reaches uniforms[b * sequence_size + s] times its mass. The
//     values are checked but unused for greedy sampling (k <= 1).
//   - batch_size: the batch size of the logits.
//   - sequence_size: the sequence length of the logits.
//   - sampled_scores: this is an output parameter to store the sampled scores
//     (as probabilities between 0 and 1) of each batch. It is a vector of shape
//     [batch_size]. Note that the probabilities is only an approximation of the
//     true probabilities as they are calculated based on the top-k logits
//     which are not normalized across the entire vocab. When k == 1, the
//     sampled_scores are always 1.0.
absl::StatusOr<std::vector<std::vector<int>>> TopKTopPSampling(
    absl::Span<const float> logits, int k, float p, float temperature,
    absl::Span<const float> uniforms, int batch_size, int sequence_size,
    std::vector<std::vector<float>>& sampled_scores);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_SAMPLING_CPU_UTIL_H_
