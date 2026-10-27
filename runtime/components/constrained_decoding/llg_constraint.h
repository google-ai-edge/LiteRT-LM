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

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/components/constrained_decoding/bitmap.h"
#include "runtime/components/constrained_decoding/constraint.h"
#include "runtime/components/constrained_decoding/logit_mask.h"
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
// Copies deep-clone the underlying constraint, so copies can be advanced
// independently of each other.
class LlgConstraintOwner {
 public:
  explicit LlgConstraintOwner(::LlgConstraint* llg_constraint)
      : llg_constraint_(llg_constraint) {}

  LlgConstraintOwner(const LlgConstraintOwner& other)
      : llg_constraint_(other.llg_constraint_ != nullptr
                            ? llg_clone_constraint(other.llg_constraint_.get())
                            : nullptr) {}

  LlgConstraintOwner& operator=(const LlgConstraintOwner& other) {
    LlgConstraintOwner copy(other);
    std::swap(llg_constraint_, copy.llg_constraint_);
    return *this;
  }

  LlgConstraintOwner(LlgConstraintOwner&&) noexcept = default;
  LlgConstraintOwner& operator=(LlgConstraintOwner&&) noexcept = default;

  ::LlgConstraint* llg_constraint() const { return llg_constraint_.get(); }

 private:
  struct Deleter {
    void operator()(::LlgConstraint* llg_constraint) const {
      llg_free_constraint(llg_constraint);
    }
  };

  std::unique_ptr<::LlgConstraint, Deleter> llg_constraint_;
};

// Represents a decoding constraint based on the LLGuidance library.
class LlgConstraint : public Constraint {
 public:
  class LlgState : public State {
   public:
    // LlgState takes ownership of llg_constraint.
    explicit LlgState(::LlgConstraint* llg_constraint)
        : llg_constraint_owner_(llg_constraint) {}

    ::LlgConstraint* llg_constraint() const {
      return llg_constraint_owner_.llg_constraint();
    }

    bool mask_computed() const { return mask_computed_; }

    void set_mask_computed(bool mask_computed) const {
      mask_computed_ = mask_computed;
    }

   private:
    LlgConstraintOwner llg_constraint_owner_;

    // Whether llg_compute_mask() has run since the last commit. llguidance
    // requires compute_mask() before commit_token(), and the C API does not
    // expose this, so it is mirrored here. Mutable because ComputeMask() takes
    // a const State& (the llg constraint is already mutated through it).
    mutable bool mask_computed_ = false;
  };

  // LlgConstraint takes ownership of llg_constraint. Once the constraint has
  // ended, only `eos_token_ids` are allowed.
  explicit LlgConstraint(::LlgConstraint* llg_constraint, int vocab_size,
                         std::vector<int> eos_token_ids)
      : llg_constraint_owner_(llg_constraint),
        vocab_size_(vocab_size),
        eos_token_ids_(std::move(eos_token_ids)) {}

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

  // Computes the logit mask given the current state.
  absl::StatusOr<std::unique_ptr<LogitMask>> ComputeMask(
      const State& state) const override;

  // Computes the allowed tokens bitmap given the current state.
  absl::StatusOr<std::unique_ptr<Bitmap>> ComputeBitmap(
      const State& state) const override;

 private:
  // Commits `token` to `state`, computing the mask first if needed.
  absl::Status CommitToken(const LlgState& state, int token) const;

  LlgConstraintOwner llg_constraint_owner_;
  int vocab_size_;
  std::vector<int> eos_token_ids_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_CONSTRAINED_DECODING_LLG_CONSTRAINT_H_
