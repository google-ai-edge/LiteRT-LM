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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_CONSTRAINED_DECODING_COMPOSITE_CONSTRAINT_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_CONSTRAINED_DECODING_COMPOSITE_CONSTRAINT_H_

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/components/constrained_decoding/constraint.h"
#include "runtime/components/constrained_decoding/logit_mask.h"
#include "runtime/components/constrained_decoding/bitmap.h"

namespace litert::lm {

// A composite constraint that combines multiple sub-constraints (both hard and
// soft) into a single unified constraint.
//
// It executes each sub-constraint's state transitions in parallel, and
// aggregates their logit masks into a CompositeLogitMask for single-pass
// evaluation.
class CompositeConstraint : public Constraint {
 public:
  // Holds the individual states of each sub-constraint in order.
  struct CompositeState : public Constraint::State {
    CompositeState() = default;
    explicit CompositeState(
        std::vector<std::unique_ptr<Constraint::State>> states)
        : sub_states(std::move(states)) {}

    std::vector<std::unique_ptr<Constraint::State>> sub_states;
  };

  // Creates an empty CompositeConstraint with an optional initial vocabulary
  // size. If the vocabulary size is not specified, it can be inferred later
  // from the sub-constraints.
  static absl::StatusOr<std::unique_ptr<CompositeConstraint>> Create(
      int vocab_size = 0);

  // Creates a CompositeConstraint with a list of constraints. Validates that
  // all constraints are non-null and that their vocabulary sizes are
  // consistent, inferring the vocabulary size from the constraints.
  static absl::StatusOr<std::unique_ptr<CompositeConstraint>> Create(
      std::vector<std::unique_ptr<Constraint>> constraints);

  // Adds a sub-constraint to the composite.
  absl::Status AddConstraint(std::unique_ptr<Constraint> constraint);

  // Adds an unowned sub-constraint to the composite.
  absl::Status AddUnownedConstraint(Constraint* constraint);

  const std::vector<std::unique_ptr<Constraint>>& constraints() const {
    return constraints_;
  }
  size_t size() const { return constraints_.size(); }
  bool empty() const { return constraints_.empty(); }

  // Constraint interface implementation:
  std::unique_ptr<State> Start() const override;

  bool IsEnded(const State& state) const override;

  int GetVocabularySize() const override { return vocab_size_; }

  absl::StatusOr<std::unique_ptr<State>> ComputeNext(const State& state,
                                                     int token) const override;

  absl::StatusOr<std::unique_ptr<LogitMask>> ComputeMask(
      const State& state) const override;

  absl::StatusOr<std::unique_ptr<Bitmap>> ComputeBitmap(
      const State& state) const override;

 private:
  explicit CompositeConstraint(int vocab_size) : vocab_size_(vocab_size) {}

  std::vector<std::unique_ptr<Constraint>> constraints_;
  int vocab_size_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_CONSTRAINED_DECODING_COMPOSITE_CONSTRAINT_H_
