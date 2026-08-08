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

#include "runtime/components/sampling_cpu_util.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl

namespace litert::lm {

namespace {

// Equivalent to *std::max_element(x, x + kSize), but written in a way to coax
// the compiler into autovectorizing it.
template <int kSize>
inline float Max(const float* x) {
  constexpr int kTileSize = 8;
  static_assert(kSize % kTileSize == 0);
  float max[kTileSize];
  std::copy_n(x, kTileSize, max);
  for (int i = kTileSize; i < kSize; i += kTileSize) {
    for (int j = 0; j < kTileSize; ++j) {
      max[j] = std::max(max[j], x[i + j]);
    }
  }
  return *std::max_element(max, max + kTileSize);
}

// Similar to std::max_element, but more optimized.
inline int MaxElement(const float* logits, int size) {
  if (size <= 0) return 0;

  constexpr int kTileSize = 128;
  float max = logits[0];
  int max_idx = 0;

  int i = 0;
  for (; i + kTileSize <= size; i += kTileSize) {
    const float* tile_logits = logits + i;

    // Find the max in this tile.
    float tile_max = Max<kTileSize>(tile_logits);

    if (tile_max > max) {
      // Find first index of this max in the tile. We only do this if we have a
      // new max, which avoids the index search for most tiles.
      for (int j = 0; j < kTileSize; ++j) {
        if (tile_logits[j] == tile_max) {
          max = tile_max;
          max_idx = i + j;
          break;
        }
      }
    }
  }

  // Handle remaining elements
  for (; i < size; ++i) {
    if (logits[i] > max) {
      max = logits[i];
      max_idx = i;
    }
  }

  return max_idx;
}

}  // namespace

absl::StatusOr<std::vector<std::vector<int>>> TopKTokenIds(
    absl::Span<const float> logits, int k, int batch_size, int sequence_size) {
  if (logits.size() % (batch_size * sequence_size) != 0) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Logits vector size must be a multiple of batch "
                        "size * sequence_size. But got %d and "
                        "%d * %d.",
                        logits.size(), batch_size, sequence_size));
  }
  const int vocab_size = logits.size() / (batch_size * sequence_size);
  std::vector<std::vector<int>> output_indices(
      batch_size, std::vector<int>(sequence_size * k));
  if (k == 1) {  // Greedy sampling. Use MaxElement to be more efficient.
    for (int b = 0; b < batch_size; ++b) {
      for (int s = 0; s < sequence_size; ++s) {
        const float* sub_logits =
            logits.data() + (b * sequence_size + s) * vocab_size;
        output_indices[b][s] = MaxElement(sub_logits, vocab_size);
      }
    }
  } else if (k <= 1024) {
    struct LogitIndex {
      float logit;
      int index;
    };
    auto min_heap_comp = [](const LogitIndex& l, const LogitIndex& r) {
      return l.logit > r.logit;
    };
    for (int b = 0; b < batch_size; ++b) {
      for (int s = 0; s < sequence_size; ++s) {
        int offset = (b * sequence_size + s) * vocab_size;
        int actual_k = std::min(k, vocab_size);

        std::vector<LogitIndex> min_heap;
        min_heap.reserve(actual_k);
        for (int i = 0; i < actual_k; ++i) {
          min_heap.push_back({logits[offset + i], i});
        }
        absl::c_make_heap(min_heap, min_heap_comp);

        for (int i = actual_k; i < vocab_size; ++i) {
          float val = logits[offset + i];
          if (val > min_heap.front().logit) {
            absl::c_pop_heap(min_heap, min_heap_comp);
            min_heap.back() = {val, i};
            absl::c_push_heap(min_heap, min_heap_comp);
          }
        }

        std::sort(min_heap.begin(), min_heap.end(),
                  [](const LogitIndex& a, const LogitIndex& b) {
                    if (a.logit != b.logit) {
                      return a.logit > b.logit;
                    }
                    return a.index < b.index;
                  });
        for (int i = 0; i < actual_k; ++i) {
          output_indices[b][s * k + i] = min_heap[i].index;
        }
      }
    }
  } else {
    std::vector<int> indices(vocab_size);
    for (int b = 0; b < batch_size; ++b) {
      for (int s = 0; s < sequence_size; ++s) {
        // Fill with 0, 1, 2,...
        std::iota(indices.begin(), indices.end(), 0);

        // Define the comparator for descending probability
        auto desc_prob_comp = [&logits, vocab_size, b, s, sequence_size](
                                  int i1, int i2) {
          int offset = (b * sequence_size + s) * vocab_size;
          return logits[offset + i1] > logits[offset + i2];
        };
        // Partition Top-K.
        // O(N) average time complexity.
        // Rearranges 'indices' such that the k elements with the highest
        // probabilities are in the range [indices.begin(), indices.begin() +
        // k). The element at indices[k] is not necessarily the (k+1)th largest.
        std::nth_element(indices.begin(), indices.begin() + k, indices.end(),
                         desc_prob_comp);
        std::copy(indices.begin(), indices.begin() + k,
                  output_indices[b].begin() + s * k);
      }
    }
  }
  return output_indices;
}

absl::StatusOr<std::vector<std::vector<float>>> Softmax(
    absl::Span<const float> logits, absl::Span<const int> topk_token_ids,
    float temperature, int batch_size, int sequence_size,
    std::vector<std::vector<float>>& max_logit_values) {
  if (logits.empty()) {
    return absl::InvalidArgumentError("Logits vector cannot be empty.");
  }
  if (logits.size() % (batch_size * sequence_size) != 0) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Logits vector size must be a multiple of batch size * sequence "
        "length. But got %d and %d * %d.",
        logits.size(), batch_size, sequence_size));
  }
  if (temperature < 0.0) {
    // A very small positive temperature can mimic greedy sampling.
    // If temp = 0, we will clamp it to epsilon later.
    return absl::InvalidArgumentError(
        absl::StrCat("Temperature must be >= 0, but got ", temperature));
  }
  const int vocab_size = logits.size() / (batch_size * sequence_size);
  const int k = topk_token_ids.size() / (batch_size * sequence_size);
  std::vector<std::vector<float>> probabilities(
      batch_size, std::vector<float>(sequence_size * k));
  max_logit_values.assign(batch_size, std::vector<float>(sequence_size));
  for (size_t b = 0; b < batch_size; ++b) {
    for (size_t s = 0; s < sequence_size; ++s) {
      // Define the comparator for ascending logit to be used with
      // std::max_element. The comparator takes two indices from topk_token_ids
      // and compares the corresponding logit values.
      auto logit_comp = [&logits, vocab_size, b, s, sequence_size](
                            int topk_idx1, int topk_idx2) {
        int offset = (b * sequence_size + s) * vocab_size;
        return logits[offset + topk_idx1] < logits[offset + topk_idx2];
      };

      int topk_offset = (b * sequence_size + s) * k;
      // Use std::max_element to find the index in topk_token_ids that
      // corresponds to the maximum logit value.
      auto max_iterator = std::max_element(
          topk_token_ids.begin() + topk_offset,
          topk_token_ids.begin() + topk_offset + k, logit_comp);
      const int max_logit_idx =
          std::distance(topk_token_ids.begin() + topk_offset, max_iterator);
      int offset = (b * sequence_size + s) * vocab_size;
      max_logit_values[b][s] =
          logits[offset + topk_token_ids[topk_offset + max_logit_idx]];

      float sum_of_exps = 0.0;
      float current_temp =
          std::max(temperature, std::numeric_limits<float>::epsilon());
      for (size_t i = 0; i < k; ++i) {
        probabilities[b][s * k + i] =
            std::exp((logits[offset + topk_token_ids[topk_offset + i]] -
                      max_logit_values[b][s]) /
                     current_temp);
        sum_of_exps += probabilities[b][s * k + i];
      }

      if (sum_of_exps <= std::numeric_limits<float>::epsilon()) {
        // Handle potential zero sum (uniform distribution fallback)
        float uniform_prob = 1.0 / static_cast<float>(k);
        std::fill(probabilities[b].begin() + s * k,
                  probabilities[b].begin() + (s + 1) * k, uniform_prob);
      } else if (std::isinf(sum_of_exps)) {
        // Handle inf sum which is caused by very small temperature.
        // This is to avoid inf sum in the softmax calculation.
        std::fill(probabilities[b].begin() + s * k,
                  probabilities[b].begin() + (s + 1) * k, 0.0f);
        probabilities[b][s * k + max_logit_idx] = 1.0f;
      } else {
        // Normalize probabilities
        float inv_sum =
            1.0 / sum_of_exps;  // Calculate inverse once for slight speedup
        for (size_t i = 0; i < k; ++i) {
          probabilities[b][s * k + i] *= inv_sum;
        }
      }
    }
  }
  return probabilities;
};

absl::StatusOr<std::vector<std::vector<int>>> TopKTopPSampling(
    absl::Span<const float> logits, int k, float p, float temperature,
    std::shared_ptr<std::default_random_engine> rng, int batch_size,
    int sequence_size, std::vector<std::vector<float>>& sampled_scores) {
  if (logits.empty()) {
    return absl::InvalidArgumentError("Logits vector cannot be empty.");
  }
  if (logits.size() % (batch_size * sequence_size) != 0) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Logits vector size must be a multiple of batch "
                        "size * sequence length. But got %d and %d * %d.",
                        logits.size(), batch_size, sequence_size));
  }
  if (k <= 0) {
    return absl::InvalidArgumentError("k must be greater than 0.");
  }
  if (p < 0.0 || p > 1.0) {
    return absl::InvalidArgumentError("p must be in the range [0.0, 1.0].");
  }
  if (rng == nullptr) {
    return absl::InvalidArgumentError("rng cannot be nullptr.");
  }
  const int vocab_size = logits.size() / (batch_size * sequence_size);
  // Ensure k is not larger than the number of probabilities
  k = std::min(k, vocab_size);

  auto topk_token_ids = TopKTokenIds(logits, k, batch_size, sequence_size);
  if (!topk_token_ids.ok()) return topk_token_ids.status();

  std::vector<int> flat_topk_token_ids(batch_size * sequence_size * k);
  for (int b = 0; b < batch_size; ++b) {
    std::copy((*topk_token_ids)[b].begin(), (*topk_token_ids)[b].end(),
              flat_topk_token_ids.begin() + b * sequence_size * k);
  }

  std::vector<std::vector<float>> max_logit_values;
  auto probabilities = Softmax(logits, flat_topk_token_ids, temperature,
                               batch_size, sequence_size, max_logit_values);
  if (!probabilities.ok()) return probabilities.status();

  std::vector<std::vector<int>> sampled_ids(batch_size,
                                            std::vector<int>(sequence_size));
  sampled_scores.assign(batch_size, std::vector<float>(sequence_size));
  if (k == 1) {  // Greedy sampling. Return the topk_token_ids directly.
    for (int b = 0; b < batch_size; ++b) {
      for (int s = 0; s < sequence_size; ++s) {
        sampled_ids[b][s] = (*topk_token_ids)[b][s];
        sampled_scores[b][s] = 1.0f;
      }
    }
    return sampled_ids;
  }
  float current_temp =
      std::max(temperature, std::numeric_limits<float>::epsilon());
  for (int b = 0; b < batch_size; ++b) {
    for (int s = 0; s < sequence_size; ++s) {
      // Define the comparator for descending probability
      auto desc_prob_comp = [&probabilities, b, s, k](int i1, int i2) {
        return (*probabilities)[b][s * k + i1] >
               (*probabilities)[b][s * k + i2];
      };

      // Sort Only the Top-K.
      // O(k log k) time complexity.
      // Sorts the indices of the top k elements based on their probabilities.
      // - index_of_topk[i] stores the offset within the current batch's top-k
      // range.
      std::vector<int> index_of_topk(k);
      std::iota(index_of_topk.begin(), index_of_topk.end(), 0);
      std::sort(index_of_topk.begin(), index_of_topk.end(), desc_prob_comp);

      // Determine Top-P Cutoff Index within Top-K.
      // O(k) time complexity.
      double cumulative_prob = 0.0;
      int final_sample_size = 0;  // Actual number of elements to sample from

      for (int i = 0; i < k; ++i) {
        // Check if adding this probability would exceed the threshold p. It
        // stops when cumulative_prob >= p.
        cumulative_prob += (*probabilities)[b][s * k + index_of_topk[i]];
        final_sample_size = i + 1;  // Include this element

        if (cumulative_prob >= p) {
          break;  // Found the smallest set within Top-K satisfying Top-P
        }
      }
      // final_sample_size now holds min(p_cutoff_within_top_k, k)

      int topk_offset = b * sequence_size * k + s * k;
      int offset = (b * sequence_size + s) * vocab_size;

      // Handle Edge Case: Cumulative Probability is Zero.
      if (cumulative_prob <= std::numeric_limits<double>::epsilon()) {
        // Fallback: Return the index with the absolute highest probability
        // (indices[0] after sorting top-k).
        sampled_ids[b][s] = flat_topk_token_ids[topk_offset + index_of_topk[0]];
        sampled_scores[b][s] = std::exp(
            (logits[offset + sampled_ids[b][s]] - max_logit_values[b][s]) /
            current_temp);
        continue;
      }

      // O(final_sample_size) which is O(k) time complexity.
      std::uniform_real_distribution<double> dist(0.0, cumulative_prob);
      double random_sample = dist(*rng);
      double current_cumulative = 0.0;
      for (int i = 0; i < final_sample_size; ++i) {
        current_cumulative += (*probabilities)[b][s * k + index_of_topk[i]];
        if (random_sample <= current_cumulative) {
          sampled_ids[b][s] =
              flat_topk_token_ids[topk_offset + index_of_topk[i]];
          sampled_scores[b][s] = (*probabilities)[b][s * k + index_of_topk[i]];
          break;
        }
      }
    }
  }
  return sampled_ids;
}

}  // namespace litert::lm
