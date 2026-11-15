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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_CONSTRAINED_DECODING_REPETITION_PENALTY_CONSTRAINT_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_CONSTRAINED_DECODING_REPETITION_PENALTY_CONSTRAINT_H_

#include <memory>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/components/constrained_decoding/constraint.h"
#include "runtime/components/constrained_decoding/logit_mask.h"
#include "runtime/components/constrained_decoding/repetition_penalty_config.h"

namespace litert::lm {

// Immutable soft constraint mask representing token adjustments with
// repetition, presence, and frequency penalties.
// Multiplicative repetition penalty:
//   if z_i > 0: z'_i = z_i / repetition_penalty
//   if z_i <= 0: z'_i = z_i * repetition_penalty
// Additive presence and frequency penalties:
//   z'_i += bias (where bias = -(presence_penalty + count * frequency_penalty))
//
// The penalties are encoded as sign-dependent SparseLogitMask entries with
// weight = 1 / repetition_penalty, which is exactly the definition above.
// Reusing the sparse representation instead of a custom mask is what lets
// LiteRtLogitMaskRunner fold these penalties into its accelerated masking
// graphs; a custom mask can only be evaluated on host memory. `entries()`
// therefore returns the converted SparseLogitMask::Entry values, not the
// RepetitionPenaltyMask::Entry values the mask was constructed from.
class RepetitionPenaltyMask : public SparseLogitMask {
 public:
  struct Entry {
    int token_id;
    float repetition_penalty = 1.0f;  // Multiplicative penalty (>= 1.0)
    float bias = 0.0f;                // Additive bias

    bool operator==(const Entry& other) const {
      return token_id == other.token_id &&
             repetition_penalty == other.repetition_penalty &&
             bias == other.bias;
    }
  };

  explicit RepetitionPenaltyMask(const std::vector<Entry>& entries)
      : SparseLogitMask(ToSparseEntries(entries)) {}

 private:
  // Converts repetition penalty entries into their sign-dependent sparse
  // equivalents.
  static std::vector<SparseLogitMask::Entry> ToSparseEntries(
      const std::vector<Entry>& entries);
};

// A constraint that applies soft repetition, presence, and frequency penalties.
class RepetitionPenaltyConstraint : public Constraint {
 public:
  struct RepetitionPenaltyState : public Constraint::State {
    RepetitionPenaltyState() = default;
    RepetitionPenaltyState(const RepetitionPenaltyState&) = default;
    RepetitionPenaltyState& operator=(const RepetitionPenaltyState&) = default;

    // Token frequency count in the active window.
    absl::flat_hash_map<int, int> token_counts;

    // Circular buffer for windowed tracking.
    std::vector<int> token_history;

    // Current index in circular buffer.
    int token_history_index = 0;
  };

  RepetitionPenaltyConstraint(int vocab_size, RepetitionPenaltyConfig config);

  std::unique_ptr<State> Start() const override;

  bool IsEnded(const State& state) const override;

  int GetVocabularySize() const override { return vocab_size_; }

  absl::StatusOr<std::unique_ptr<State>> ComputeNext(const State& state,
                                                     int token) const override;

  absl::StatusOr<std::unique_ptr<LogitMask>> ComputeMask(
      const State& state) const override;

  const RepetitionPenaltyConfig& config() const { return config_; }

 private:
  const int vocab_size_;
  const RepetitionPenaltyConfig config_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_CONSTRAINED_DECODING_REPETITION_PENALTY_CONSTRAINT_H_
