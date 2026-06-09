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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_CONSTRAINED_DECODING_THINKING_BUDGET_CONSTRAINT_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_CONSTRAINED_DECODING_THINKING_BUDGET_CONSTRAINT_H_

#include <memory>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/components/constrained_decoding/bitmap.h"
#include "runtime/components/constrained_decoding/constraint.h"

namespace litert::lm {

// A bitmap that only allows a single token ID.
//
// Used to temporarily restrict the vocabulary to force the generation of a
// specific token (e.g., forcing end-of-thinking delimiters).
class SingleAllowedTokenBitmap : public Bitmap {
 public:
  explicit SingleAllowedTokenBitmap(int allowed_token_id)
      : allowed_token_id_(allowed_token_id) {}
  bool Get(int index) const override { return index == allowed_token_id_; }

 private:
  int allowed_token_id_;
};

// A constraint that enforces a thinking token budget limit during generation.
//
// It wraps an optional user-defined constraint (`user_constraint`) using the
// decorator pattern, forwarding constraint logic to it. When the budget of
// thinking tokens is exceeded, it overrides the allowed token list to force the
// generation of the thinking end-delimiters (`end_token_ids`, such as
// `<channel|>`), after which it transitions control back to the wrapped
// constraint for content generation.
class ThinkingBudgetConstraint : public Constraint {
 public:
  // State variables for tracking the thinking progress of a single sequence.
  struct ThinkingState : public Constraint::State {
    int thinking_token_count = 0;
    bool in_thinking = true;
    // Index of the forced end delimiter token we are currently generating,
    // or -1 if we are not currently forcing the end of the thinking channel.
    int forced_end_token_index = -1;
    // Current match progress of the end delimiters generated naturally by the
    // model.
    int natural_match_index = 0;
    // Index of the start delimiter token we are currently matching,
    // or -1 if we are not matching or have finished matching.
    int matching_start_index = 0;
    std::unique_ptr<Constraint::State> user_state = nullptr;
  };

  ThinkingBudgetConstraint(Constraint* absl_nullable user_constraint,
                           int budget, std::vector<int> start_token_ids,
                           std::vector<int> end_token_ids, int vocab_size)
      : user_constraint_(user_constraint),
        budget_(budget),
        start_token_ids_(std::move(start_token_ids)),
        end_token_ids_(std::move(end_token_ids)),
        vocab_size_(vocab_size) {}

  std::unique_ptr<Constraint::State> Start() const override;

  bool IsEnded(const Constraint::State& state) const override;

  int GetVocabularySize() const override { return vocab_size_; }

  absl::StatusOr<std::unique_ptr<Constraint::State>> ComputeNext(
      const Constraint::State& state, int token) const override;

  absl::StatusOr<std::unique_ptr<Bitmap>> ComputeBitmap(
      const Constraint::State& state) const override;

 private:
  Constraint* absl_nullable user_constraint_;
  const int budget_;
  const std::vector<int> start_token_ids_;
  const std::vector<int> end_token_ids_;
  const int vocab_size_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_CONSTRAINED_DECODING_THINKING_BUDGET_CONSTRAINT_H_
