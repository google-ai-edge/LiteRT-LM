// Copyright 2026 The ODML Authors.
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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_TOP_K_DISTRIBUTION_BUFFER_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_TOP_K_DISTRIBUTION_BUFFER_H_

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"  // from @litert

namespace litert::lm {

// Helpers for validating and populating the output TensorBuffers of
// `Sampler::SampleToIdAndTopKBuffer()`, shared by the CPU and GPU samplers.

// Returns K, the size of the last dimension of the top-k distribution buffers
// of `Sampler::SampleToIdAndTopKBuffer()`, after checking that `probs_tensor`
// is float32, `ids_tensor` is int32, and both have
// `batch_size * sequence_size * K` elements with K as the last dimension.
absl::StatusOr<int> GetTopKDistributionBufferSize(
    const TensorBuffer& probs_tensor, const TensorBuffer& ids_tensor,
    int batch_size, int sequence_size);

// Writes a top-k distribution to the output buffers of
// `Sampler::SampleToIdAndTopKBuffer()`.
//   - `probs` and `ids` hold `src_k` entries per (batch, position), in
//     [batch_size, sequence_size, src_k] order and sorted in descending order
//     of probability. Only the first `num_valid` entries of each row are
//     copied (`src_k` may exceed `num_valid` when reading back padded GPU
//     buffers); any remaining entries up to K are padded with 0.0 and -1.
//   - Returns `InvalidArgumentError` if the buffers are malformed (see
//     `GetTopKDistributionBufferSize()`) or their K is less than `num_valid`.
absl::Status WriteTopKDistribution(absl::Span<const float> probs,
                                   absl::Span<const int> ids, int src_k,
                                   int num_valid, int batch_size,
                                   int sequence_size,
                                   TensorBuffer& probs_tensor,
                                   TensorBuffer& ids_tensor);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_TOP_K_DISTRIBUTION_BUFFER_H_
