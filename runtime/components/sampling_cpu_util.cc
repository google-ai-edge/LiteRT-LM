#include "runtime/components/sampling_cpu_util.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl

namespace litert::lm {

absl::StatusOr<std::vector<float>> Softmax(absl::Span<const float> logits,
                                           float temperature, int batch_size) {
  if (logits.empty()) {
    return absl::InvalidArgumentError("Logits vector cannot be empty.");
  }
  if (logits.size() % batch_size != 0) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Logits vector size must be a multiple of batch size. But got %d and "
        "%d.",
        logits.size(), batch_size));
  }
  if (temperature <= 0.0) {
      // A very small positive temperature can mimic greedy sampling,
      // but 0.0 would cause division by zero.
      return absl::InvalidArgumentError(
          absl::StrCat("Temperature must be positive, but got ", temperature));
  }
  const int vocab_size = logits.size() / batch_size;
  std::vector<float> probabilities(logits.size());
  for (size_t b = 0; b < batch_size; ++b) {
    // Use std::max_element to find the maximum value among the logits.
    // Dereference the iterator returned by std::max_element to get the value.
    float max_logit = *std::max_element(logits.begin() + b * vocab_size,
                                        logits.begin() + (b + 1) * vocab_size);

    float sum_of_exps = 0.0;
    for (size_t i = b * vocab_size; i < (b + 1) * vocab_size; ++i) {
      probabilities[i] = std::exp((logits[i] - max_logit) / temperature);
      sum_of_exps += probabilities[i];
    }

    if (sum_of_exps <= std::numeric_limits<float>::epsilon()) {
      // Handle potential zero sum (uniform distribution fallback)
      float uniform_prob = 1.0 / static_cast<float>(logits.size());
      std::fill(probabilities.begin(), probabilities.end(), uniform_prob);
    } else {
      // Normalize
      float inv_sum =
          1.0 / sum_of_exps;  // Calculate inverse once for slight speedup
      for (size_t i = b * vocab_size; i < (b + 1) * vocab_size; ++i) {
        probabilities[i] *= inv_sum;
      }
    }
  }
  return probabilities;
};

absl::StatusOr<std::vector<int>> TopKTopPSampling(
    absl::Span<const float> probabilities, int k, float p, std::mt19937& rng,
    int batch_size) {
  if (probabilities.empty()) {
    return absl::InvalidArgumentError("Logits vector cannot be empty.");
  }
  if (probabilities.size() % batch_size != 0) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Probabilities vector size must be a multiple of batch "
                        "size. But got %d and "
                        "%d.",
                        probabilities.size(), batch_size));
  }
  if (k <= 0) {
    return absl::InvalidArgumentError("k must be greater than 0.");
  }
  if (p < 0.0 || p > 1.0) {
    return absl::InvalidArgumentError("p must be in the range [0.0, 1.0].");
  }
  const int vocab_size = probabilities.size() / batch_size;
  // Ensure k is not larger than the number of probabilities
  k = std::min(k, vocab_size);

  std::vector<int> sampled_ids(batch_size);
  std::vector<int> indices(vocab_size);
  for (int b = 0; b < batch_size; ++b) {
    // Fill with 0, 1, 2,...
    std::iota(indices.begin(), indices.end(), 0);

    // Define the comparator for descending probability
    auto desc_prob_comp = [&probabilities, vocab_size, b](int i1, int i2) {
      return probabilities[b * vocab_size + i1] >
             probabilities[b * vocab_size + i2];
    };

    // Partition Top-K.
    // O(N) average time complexity.
    // Rearranges 'indices' such that the k elements with the highest
    // probabilities are in the range [indices.begin(), indices.begin() + k).
    // The element at indices[k] is not necessarily the (k+1)th largest.
    std::nth_element(indices.begin(), indices.begin() + k, indices.end(),
                     desc_prob_comp);

    // Sort Only the Top-K.
    // O(k log k) time complexity.
    // Sorts only the first k indices, which now correspond to the top k
    // probabilities.
    std::sort(indices.begin(), indices.begin() + k, desc_prob_comp);

    // Determine Top-P Cutoff Index within Top-K.
    // O(k) time complexity.
    double cumulative_prob = 0.0;
    double nucleus_sum = 0.0;
    int final_nucleus_size = 0;  // Actual number of elements to sample from

    for (int i = 0; i < k; ++i) {
      int current_original_index = b * vocab_size + indices[i];
      // Check if adding this probability would exceed the threshold p. It stops
      // when cumulative_prob >= p.
      cumulative_prob += probabilities[current_original_index];
      nucleus_sum += probabilities[current_original_index];
      final_nucleus_size = i + 1;  // Include this element

      if (cumulative_prob >= p) {
        break;  // Found the smallest set within Top-K satisfying Top-P
      }
    }
    // final_nucleus_size now holds min(p_cutoff_within_top_k, k)

    // Handle Edge Case: Zero Nucleus Sum.
    if (nucleus_sum <= std::numeric_limits<double>::epsilon()) {
      // Fallback: Return the index with the absolute highest probability
      // (indices[0] after sorting top-k).
      sampled_ids[b] = indices[0];
    }

    // O(final_nucleus_size) which is O(k) time complexity.
    std::uniform_real_distribution<double> dist(0.0, nucleus_sum);
    double random_sample = dist(rng);

    double current_cumulative = 0.0;
    for (int i = 0; i < final_nucleus_size; ++i) {
      int current_original_index = b * vocab_size + indices[i];
      current_cumulative += probabilities[current_original_index];
      if (random_sample <= current_cumulative) {
        sampled_ids[b] = current_original_index - b * vocab_size;
      }
    }
  }
  return sampled_ids;
}

}  // namespace litert::lm
