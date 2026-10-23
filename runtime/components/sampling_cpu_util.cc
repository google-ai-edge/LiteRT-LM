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
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
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

// The top-k distribution of each (batch, position), in the order that
// TopKTopPSampling() applies the top-p cut in.
struct SortedTopK {
  // min(k, vocab_size).
  int k = 0;
  // Shape [batch_size, sequence_size, k] (flattened). Each (batch, position)
  // is sorted in descending order of probability, with ties broken by
  // ascending token id so that the order, and hence the top-p cut, doesn't
  // depend on the order in which TopKTokenIds() returns the ids.
  std::vector<int> ids;
  std::vector<float> probs;
  // Shape [batch_size, sequence_size] (flattened): the max logit of each
  // (batch, position).
  std::vector<float> max_logits;
};

absl::StatusOr<SortedTopK> ComputeSortedTopK(absl::Span<const float> logits,
                                             int k, float temperature,
                                             int batch_size,
                                             int sequence_size) {
  if (logits.empty()) {
    return absl::InvalidArgumentError("Logits vector cannot be empty.");
  }
  if (batch_size <= 0 || sequence_size <= 0) {
    return absl::InvalidArgumentError(
        absl::StrFormat("batch_size and sequence_size must be positive, but "
                        "got %d and %d.",
                        batch_size, sequence_size));
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
  const int vocab_size = logits.size() / (batch_size * sequence_size);
  // Ensure k is not larger than the number of probabilities
  k = std::min(k, vocab_size);

  ABSL_ASSIGN_OR_RETURN(auto topk_token_ids,
                        TopKTokenIds(logits, k, batch_size, sequence_size));
  std::vector<int> flat_topk_token_ids(batch_size * sequence_size * k);
  for (int b = 0; b < batch_size; ++b) {
    std::copy(topk_token_ids[b].begin(), topk_token_ids[b].end(),
              flat_topk_token_ids.begin() + b * sequence_size * k);
  }
  std::vector<std::vector<float>> max_logit_values;
  ABSL_ASSIGN_OR_RETURN(auto probabilities,
                        Softmax(logits, flat_topk_token_ids, temperature,
                                batch_size, sequence_size, max_logit_values));

  const int num_rows = batch_size * sequence_size;
  SortedTopK sorted;
  sorted.k = k;
  sorted.ids.resize(num_rows * k);
  sorted.probs.resize(num_rows * k);
  sorted.max_logits.resize(num_rows);
  std::vector<int> order(k);
  for (int b = 0; b < batch_size; ++b) {
    for (int s = 0; s < sequence_size; ++s) {
      const int row = b * sequence_size + s;
      const float* row_probs = probabilities[b].data() + s * k;
      const int* row_ids = flat_topk_token_ids.data() + row * k;
      // O(k log k) time complexity.
      std::iota(order.begin(), order.end(), 0);
      std::sort(order.begin(), order.end(), [&](int i1, int i2) {
        if (row_probs[i1] != row_probs[i2]) {
          return row_probs[i1] > row_probs[i2];
        }
        return row_ids[i1] < row_ids[i2];
      });
      for (int i = 0; i < k; ++i) {
        sorted.probs[row * k + i] = row_probs[order[i]];
        sorted.ids[row * k + i] = row_ids[order[i]];
      }
      sorted.max_logits[row] = max_logit_values[b][s];
    }
  }
  return sorted;
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
    int sequence_size, std::vector<std::vector<float>>& sampled_scores,
    TopKDistribution* topk_distribution) {
  if (p < 0.0 || p > 1.0) {
    return absl::InvalidArgumentError("p must be in the range [0.0, 1.0].");
  }
  if (rng == nullptr) {
    return absl::InvalidArgumentError("rng cannot be nullptr.");
  }
  ABSL_ASSIGN_OR_RETURN(
      SortedTopK sorted_topk,
      ComputeSortedTopK(logits, k, temperature, batch_size, sequence_size));
  k = sorted_topk.k;
  const int vocab_size = logits.size() / (batch_size * sequence_size);

  std::vector<std::vector<int>> sampled_ids(batch_size,
                                            std::vector<int>(sequence_size));
  sampled_scores.assign(batch_size, std::vector<float>(sequence_size));
  float current_temp =
      std::max(temperature, std::numeric_limits<float>::epsilon());
  for (int b = 0; b < batch_size; ++b) {
    for (int s = 0; s < sequence_size; ++s) {
      const int row = b * sequence_size + s;
      const int* topk_ids = sorted_topk.ids.data() + row * k;
      const float* topk_probs = sorted_topk.probs.data() + row * k;
      if (k == 1) {  // Greedy sampling. Return the top token directly.
        sampled_ids[b][s] = topk_ids[0];
        sampled_scores[b][s] = 1.0f;
        continue;
      }

      // Determine Top-P Cutoff Index within Top-K.
      // O(k) time complexity.
      double cumulative_prob = 0.0;
      int final_sample_size = 0;  // Actual number of elements to sample from

      for (int i = 0; i < k; ++i) {
        // Check if adding this probability would exceed the threshold p. It
        // stops when cumulative_prob >= p.
        cumulative_prob += topk_probs[i];
        final_sample_size = i + 1;  // Include this element

        if (cumulative_prob >= p) {
          break;  // Found the smallest set within Top-K satisfying Top-P
        }
      }
      // final_sample_size now holds min(p_cutoff_within_top_k, k)

      const int offset = row * vocab_size;

      // Handle Edge Case: Cumulative Probability is Zero.
      if (cumulative_prob <= std::numeric_limits<double>::epsilon()) {
        // Fallback: Return the token with the highest probability.
        sampled_ids[b][s] = topk_ids[0];
        sampled_scores[b][s] = std::exp(
            (logits[offset + sampled_ids[b][s]] - sorted_topk.max_logits[row]) /
            current_temp);
        continue;
      }

      // O(final_sample_size) which is O(k) time complexity.
      std::uniform_real_distribution<double> dist(0.0, cumulative_prob);
      double random_sample = dist(*rng);
      double current_cumulative = 0.0;
      for (int i = 0; i < final_sample_size; ++i) {
        current_cumulative += topk_probs[i];
        if (random_sample <= current_cumulative) {
          sampled_ids[b][s] = topk_ids[i];
          sampled_scores[b][s] = topk_probs[i];
          break;
        }
      }
    }
  }
  if (topk_distribution != nullptr) {
    topk_distribution->k = k;
    topk_distribution->probs = std::move(sorted_topk.probs);
    topk_distribution->ids = std::move(sorted_topk.ids);
  }
  return sampled_ids;
}

absl::Status ComputeTopKDistribution(absl::Span<const float> logits, int k,
                                     float temperature, int batch_size,
                                     int sequence_size,
                                     std::vector<float>& probs,
                                     std::vector<int>& ids) {
  ABSL_ASSIGN_OR_RETURN(
      SortedTopK sorted_topk,
      ComputeSortedTopK(logits, k, temperature, batch_size, sequence_size));
  probs = std::move(sorted_topk.probs);
  ids = std::move(sorted_topk.ids);
  return absl::OkStatus();
}

}  // namespace litert::lm
