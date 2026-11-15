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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_CONSTRAINED_DECODING_NO_REPEAT_NGRAM_CONSTRAINT_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_CONSTRAINED_DECODING_NO_REPEAT_NGRAM_CONSTRAINT_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/components/constrained_decoding/constraint.h"
#include "runtime/components/constrained_decoding/logit_mask.h"
#include "runtime/components/constrained_decoding/no_repeat_ngram_config.h"

namespace litert::lm {

// A constraint that forbids previously seen n-grams from repeating.
class NoRepeatNgramConstraint : public Constraint {
 public:
  using PrefixHash = size_t;

  struct NoRepeatNgramState : public Constraint::State {
    NoRepeatNgramState() = default;
    NoRepeatNgramState(const NoRepeatNgramState&) = default;
    NoRepeatNgramState& operator=(const NoRepeatNgramState&) = default;

    // If config_.window_size() > 0, circular buffer of size window_size.
    // Otherwise, stores all tokens generated so far.
    std::vector<int> token_history;

    // Used when config_.no_repeat_ngram_size() == 1.
    // Stores token_id -> count in window.
    absl::flat_hash_map<int, int> banned_tokens;

    // Used when config_.no_repeat_ngram_size() > 1.
    // Maps prefix hash -> set of end indices in token_history.
    absl::flat_hash_map<PrefixHash, absl::flat_hash_set<int>>
        prefix_hash_to_end_indices;

    // Used when config_.no_repeat_ngram_size() > 1 && config_.window_size() ==
    // 0.
    PrefixHash latest_prefix_hash = 0;

    // Used when config_.no_repeat_ngram_size() > 1 && config_.window_size() >
    // 0.
    std::vector<PrefixHash> prefix_hash_history;

    // Total tokens consumed so far.
    int64_t num_tokens_updated = 0;

    // Write index in circular buffer.
    int token_history_index = 0;
  };

  NoRepeatNgramConstraint(int vocab_size, NoRepeatNgramConfig config);

  std::unique_ptr<State> Start() const override;

  bool IsEnded(const State& state) const override;

  int GetVocabularySize() const override { return vocab_size_; }

  absl::StatusOr<std::unique_ptr<State>> ComputeNext(const State& state,
                                                     int token) const override;

  absl::StatusOr<std::unique_ptr<LogitMask>> ComputeMask(
      const State& state) const override;

  const NoRepeatNgramConfig& config() const { return config_; }

  // Helpers for testing prefix hash collisions.
  PrefixHash ComputePrefixHash(const NoRepeatNgramState& state) const;
  bool PrefixesMatch(const NoRepeatNgramState& state, int end_index1,
                     int end_index2) const;
  int prefix_hash_collision_count() const {
    return prefix_hash_collision_count_.load(std::memory_order_relaxed);
  }

 private:
  const int vocab_size_;
  const NoRepeatNgramConfig config_;
  const int prefix_length_;

  mutable std::atomic<int> prefix_hash_collision_count_{0};
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_CONSTRAINED_DECODING_NO_REPEAT_NGRAM_CONSTRAINT_H_
