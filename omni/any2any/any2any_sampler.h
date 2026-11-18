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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_ANY2ANY_SAMPLER_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_ANY2ANY_SAMPLER_H_

#include <cstdint>
#include <optional>
#include <random>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl

namespace litert::omni::any2any {

// Reusable, NaN/Inf-safe greedy and temperature/top-k/top-p token sampler
// shared by `Any2AnyLlmStage` and `Any2AnyTtsDecoder`.
//
// Retains scratch buffers across `Sample` calls to avoid per-step heap
// allocations on autoregressive decode loops, selects top-k finite candidates
// before applying `std::exp`, and samples via an inverse-CDF scan.
class Any2AnySampler {
 public:
  struct Options {
    // When false (or when `!(temperature > 0.0f)`), `Sample` falls back to
    // greedy selection via `SelectGreedy`.
    bool do_sample = false;
    // Softmax temperature for stochastic sampling; non-positive or `NaN`
    // values fall back to greedy selection.
    float temperature = 0.0f;
    // Number of highest finite logits to retain before softmax (`<= 0`
    // disables top-k truncation and considers all finite logits).
    int top_k = 0;
    // Nucleus (top-p) cumulative probability threshold; values outside `(0, 1)`
    // or `NaN` disable nucleus truncation.
    float top_p = 1.0f;
    // Optional RNG seed for reproducible stochastic sampling (defaults to 42).
    std::optional<uint64_t> seed = 42;
  };

  // Constructs a sampler with `options` and seeds the internal RNG via `Reset`.
  explicit Any2AnySampler(Options options);

  // Selects the first `+Inf` logit if present, else the argmax among finite
  // logits (earliest index on ties), or `0` if `logits` is empty or contains
  // only `NaN` / `-Inf`.
  static int SelectGreedy(absl::Span<const float> logits);

  // Selects the next token ID from `logits` using `SelectGreedy` when
  // `!options.do_sample || !(options.temperature > 0.0f) || options.top_k ==
  // 1`, or temperature-scaled top-k + nucleus (top-p) sampling otherwise.
  // Non-finite logits are handled safely: the first `+Inf` is returned
  // immediately, `NaN` and `-Inf` entries are ignored, and `0` is returned if
  // `logits` is empty or has no finite/`+Inf` entries.
  int Sample(absl::Span<const float> logits);

  // Reseeds the internal RNG from `options.seed` (defaulting to 42).
  void Reset();

 private:
  struct Candidate {
    float logit;
    int index;
  };

  static bool IsBetterCandidate(const Candidate& a, const Candidate& b);

  Options options_;
  std::mt19937_64 rng_;
  std::vector<Candidate> candidates_;
  std::vector<double> probs_;
};

}  // namespace litert::omni::any2any

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_ANY2ANY_SAMPLER_H_
