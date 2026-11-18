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

#include "omni/any2any/any2any_sampler.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl

namespace litert::omni::any2any {

Any2AnySampler::Any2AnySampler(Options options) : options_(options) { Reset(); }

bool Any2AnySampler::IsBetterCandidate(const Candidate& a, const Candidate& b) {
  return a.logit != b.logit ? a.logit > b.logit : a.index < b.index;
}

int Any2AnySampler::SelectGreedy(absl::Span<const float> logits) {
  int best_idx = 0;
  float best_val = -std::numeric_limits<float>::infinity();
  bool found_finite = false;
  for (size_t i = 0; i < logits.size(); ++i) {
    const float v = logits[i];
    if (std::isinf(v) && v > 0.0f) {
      return static_cast<int>(i);
    }
    if (std::isfinite(v) && (!found_finite || v > best_val)) {
      best_val = v;
      best_idx = static_cast<int>(i);
      found_finite = true;
    }
  }
  return best_idx;
}

int Any2AnySampler::Sample(absl::Span<const float> logits) {
  if (logits.empty()) {
    return 0;
  }
  if (!options_.do_sample || !(options_.temperature > 0.0f) ||
      options_.top_k == 1) {
    return SelectGreedy(logits);
  }

  const bool bounded_top_k =
      options_.top_k > 0 && static_cast<size_t>(options_.top_k) < logits.size();
  const size_t k =
      bounded_top_k ? static_cast<size_t>(options_.top_k) : logits.size();

  candidates_.clear();
  candidates_.reserve(k);

  if (bounded_top_k) {
    // Maintain a k-element min-heap of the best finite candidates (worst
    // candidate at `front()`, since `IsBetterCandidate` orders better items as
    // "less" for the max-heap invariant).
    for (size_t i = 0; i < logits.size(); ++i) {
      const float v = logits[i];
      if (std::isinf(v) && v > 0.0f) {
        return static_cast<int>(i);
      }
      if (!std::isfinite(v)) {
        continue;
      }
      const Candidate c{v, static_cast<int>(i)};
      if (candidates_.size() < k) {
        candidates_.push_back(c);
        if (candidates_.size() == k) {
          absl::c_make_heap(candidates_, IsBetterCandidate);
        }
      } else if (IsBetterCandidate(c, candidates_.front())) {
        absl::c_pop_heap(candidates_, IsBetterCandidate);
        candidates_.back() = c;
        absl::c_push_heap(candidates_, IsBetterCandidate);
      }
    }
  } else {
    for (size_t i = 0; i < logits.size(); ++i) {
      const float v = logits[i];
      if (std::isinf(v) && v > 0.0f) {
        return static_cast<int>(i);
      }
      if (std::isfinite(v)) {
        candidates_.push_back({v, static_cast<int>(i)});
      }
    }
  }

  if (candidates_.empty()) {
    return 0;
  }

  absl::c_sort(candidates_, IsBetterCandidate);
  if (candidates_.size() == 1) {
    return candidates_.front().index;
  }

  const double max_logit = static_cast<double>(candidates_.front().logit);
  const double temp = std::max<double>(options_.temperature, 1e-6);

  probs_.clear();
  probs_.reserve(candidates_.size());
  double sum = 0.0;
  for (const Candidate& c : candidates_) {
    const double p =
        std::exp((static_cast<double>(c.logit) - max_logit) / temp);
    probs_.push_back(p);
    sum += p;
  }

  size_t keep = candidates_.size();
  double nucleus_sum = sum;
  if (options_.top_p > 0.0f && options_.top_p < 1.0f) {
    const double threshold = static_cast<double>(options_.top_p) * sum;
    double cumulative = 0.0;
    size_t n = 0;
    for (; n < keep; ++n) {
      cumulative += probs_[n];
      if (cumulative >= threshold) {
        ++n;
        break;
      }
    }
    keep = std::max<size_t>(1, n);
    nucleus_sum = cumulative;
  }

  if (keep == 1) {
    return candidates_.front().index;
  }

  const double u =
      std::uniform_real_distribution<double>(0.0, nucleus_sum)(rng_);
  double cumulative = 0.0;
  int last_positive_idx = candidates_.front().index;
  for (size_t i = 0; i < keep; ++i) {
    if (probs_[i] <= 0.0) {
      continue;
    }
    last_positive_idx = candidates_[i].index;
    cumulative += probs_[i];
    if (u < cumulative) {
      return candidates_[i].index;
    }
  }
  return last_positive_idx;
}

void Any2AnySampler::Reset() { rng_.seed(options_.seed.value_or(42)); }

}  // namespace litert::omni::any2any
