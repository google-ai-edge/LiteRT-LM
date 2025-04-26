#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_SAMPLING_CPU_UTIL_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_SAMPLING_CPU_UTIL_H_

#include <random>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl

namespace litert::lm {

// Computes the softmax of the given logits. The logits must be a 2D tensor (in
// a flattened buffer) of shape [batch_size, vocab_size]. The output is a vector
// of probabilities of shape [batch_size, vocab_size].
absl::StatusOr<std::vector<float>> Softmax(absl::Span<const float> logits,
                                           float temperature,
                                           int batch_size = 1);

// Samples a batch of token ids from the given probabilities. The probabilities
// must be a 2D tensor (in a flattened buffer) of shape [batch_size,
// vocab_size]. The output is a vector of sampled token ids of shape
// [batch_size].
absl::StatusOr<std::vector<int>> TopKTopPSampling(
    absl::Span<const float> probabilities, int k, float p, std::mt19937& rng,
    int batch_size = 1);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_SAMPLING_CPU_UTIL_H_
