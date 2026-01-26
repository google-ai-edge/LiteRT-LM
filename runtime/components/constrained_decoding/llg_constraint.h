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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_CONSTRAINED_DECODING_LLG_CONSTRAINT_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_CONSTRAINED_DECODING_LLG_CONSTRAINT_H_

#include <memory>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/components/constrained_decoding/bitmap.h"
#include "runtime/components/constrained_decoding/constraint.h"
#include "llguidance.h"

namespace litert::lm {

class LlgBitmap : public Bitmap {
 public:
  explicit LlgBitmap(std::vector<bool>&& mask) : mask_(std::move(mask)) {}

  bool Get(int index) const override { return mask_[index]; }

 private:
  const std::vector<bool> mask_;
};

// A wrapper class to own the ::LlgConstraint* pointer from llguidance.h.
class LlgConstraintOwner {
 public:
  explicit LlgConstraintOwner(::LlgConstraint* llg_constraint)
      : llg_constraint_(llg_constraint) {}

  ~LlgConstraintOwner() { llg_free_constraint(llg_constraint_); }

  ::LlgConstraint* llg_constraint() const { return llg_constraint_; }

 private:
  ::LlgConstraint* llg_constraint_;  // Owned.
};

// Represents a decoding constraint based on the LLGuidance library.
class LlgConstraint : public Constraint {
 public:
  class LlgState : public State {
   public:
    // LlgState takes ownership of llg_constraint.
    explicit LlgState(::LlgConstraint* llg_constraint) {
      llg_constraint_owner_ =
          std::make_shared<LlgConstraintOwner>(llg_constraint);
    }

    ::LlgConstraint* llg_constraint() const {
      return llg_constraint_owner_->llg_constraint();
    }

   private:
    // The shared_ptr is needed because Constraint::State must be copyable.
    std::shared_ptr<LlgConstraintOwner> llg_constraint_owner_;
  };

  // LlgConstraint takes ownership of llg_constraint.
  explicit LlgConstraint(::LlgConstraint* llg_constraint, int vocab_size,
                         int eos_token_id)
      : llg_constraint_owner_(llg_constraint),
        vocab_size_(vocab_size),
        eos_token_id_(eos_token_id) {}

  // Gets the start state of the constraint.
  std::unique_ptr<State> Start() const override;

  // Returns true if the constraint is at the end state.
  bool IsEnded(const State& state) const override;

  // Gets the vocabulary size of the constraint.
  int GetVocabularySize() const override;

  // Computes the next state given the current state and the latest decoded
  // token.
  absl::StatusOr<std::unique_ptr<State>> ComputeNext(const State& state,
                                                     int token) const override;

  // Computes the allowed tokens bitmap given the current state.
  absl::StatusOr<std::unique_ptr<Bitmap>> ComputeBitmap(
      const State& state) const override;

 private:
  LlgConstraintOwner llg_constraint_owner_;
  int vocab_size_;
  int eos_token_id_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_CONSTRAINED_DECODING_LLG_CONSTRAINT_H_
