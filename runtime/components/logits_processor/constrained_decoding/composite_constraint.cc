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

#include "runtime/components/logits_processor/constrained_decoding/composite_constraint.h"

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "runtime/components/logits_processor/constrained_decoding/constraint.h"
#include "runtime/components/logits_processor/constrained_decoding/logit_mask.h"

namespace litert::lm {
namespace {

class UnownedConstraint : public Constraint {
 public:
  explicit UnownedConstraint(Constraint* constraint)
      : constraint_(*constraint) {}

  std::unique_ptr<State> Start() const override { return constraint_.Start(); }

  bool IsEnded(const State& state) const override {
    return constraint_.IsEnded(state);
  }

  int GetVocabularySize() const override {
    return constraint_.GetVocabularySize();
  }

  absl::StatusOr<std::unique_ptr<State>> ComputeNext(const State& state,
                                                     int token) const override {
    return constraint_.ComputeNext(state, token);
  }

  absl::StatusOr<std::unique_ptr<LogitMask>> ComputeMask(
      const State& state) const override {
    return constraint_.ComputeMask(state);
  }

 private:
  Constraint& constraint_;
};

}  // namespace

absl::StatusOr<std::unique_ptr<CompositeConstraint>>
CompositeConstraint::Create(int vocab_size) {
  if (vocab_size < 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("Vocabulary size cannot be negative: ", vocab_size));
  }
  return absl::WrapUnique(new CompositeConstraint(vocab_size));
}

absl::StatusOr<std::unique_ptr<CompositeConstraint>>
CompositeConstraint::Create(
    std::vector<std::unique_ptr<Constraint>> constraints) {
  auto composite = absl::WrapUnique(new CompositeConstraint(/*vocab_size=*/0));
  for (auto& constraint : constraints) {
    ABSL_RETURN_IF_ERROR(composite->AddConstraint(std::move(constraint)));
  }
  return composite;
}

absl::Status CompositeConstraint::AddConstraint(
    std::unique_ptr<Constraint> constraint) {
  if (constraint == nullptr) {
    return absl::InvalidArgumentError("Constraint cannot be null.");
  }
  if (vocab_size_ <= 0) {
    vocab_size_ = constraint->GetVocabularySize();
  } else if (constraint->GetVocabularySize() != vocab_size_) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Constraint vocabulary size (", constraint->GetVocabularySize(),
        ") does not match composite vocabulary size (", vocab_size_, ")."));
  }
  constraints_.push_back(std::move(constraint));
  return absl::OkStatus();
}

absl::Status CompositeConstraint::AddUnownedConstraint(Constraint* constraint) {
  if (constraint == nullptr) {
    return absl::InvalidArgumentError("Constraint cannot be null.");
  }
  return AddConstraint(std::make_unique<UnownedConstraint>(constraint));
}

std::unique_ptr<Constraint::State> CompositeConstraint::Start() const {
  std::vector<std::unique_ptr<Constraint::State>> sub_states;
  sub_states.reserve(constraints_.size());
  for (const auto& constraint : constraints_) {
    sub_states.push_back(constraint->Start());
  }
  return std::make_unique<CompositeState>(std::move(sub_states));
}

bool CompositeConstraint::IsEnded(const State& state) const {
  const auto& composite_state = static_cast<const CompositeState&>(state);
  for (size_t i = 0; i < constraints_.size(); ++i) {
    if (i < composite_state.sub_states.size() &&
        composite_state.sub_states[i] != nullptr) {
      if (constraints_[i]->IsEnded(*composite_state.sub_states[i])) {
        return true;
      }
    }
  }
  return false;
}

absl::StatusOr<std::unique_ptr<Constraint::State>>
CompositeConstraint::ComputeNext(const State& state, int token) const {
  if (vocab_size_ > 0 && (token < 0 || token >= vocab_size_)) {
    return absl::InvalidArgumentError("Invalid token id.");
  }
  const auto& composite_state = static_cast<const CompositeState&>(state);
  if (composite_state.sub_states.size() != constraints_.size()) {
    return absl::InvalidArgumentError(
        "CompositeState sub-states size does not match constraints count.");
  }

  std::vector<std::unique_ptr<Constraint::State>> next_sub_states;
  next_sub_states.reserve(constraints_.size());

  for (size_t i = 0; i < constraints_.size(); ++i) {
    if (composite_state.sub_states[i] == nullptr) {
      return absl::InvalidArgumentError("Encountered null sub-state.");
    }
    ABSL_ASSIGN_OR_RETURN(
        auto next_sub_state,
        constraints_[i]->ComputeNext(*composite_state.sub_states[i], token));
    next_sub_states.push_back(std::move(next_sub_state));
  }

  return std::make_unique<CompositeState>(std::move(next_sub_states));
}

absl::StatusOr<std::unique_ptr<LogitMask>> CompositeConstraint::ComputeMask(
    const State& state) const {
  const auto& composite_state = static_cast<const CompositeState&>(state);
  if (composite_state.sub_states.size() != constraints_.size()) {
    return absl::InvalidArgumentError(
        "CompositeState sub-states size does not match constraints count.");
  }

  auto composite_mask = std::make_unique<CompositeLogitMask>();

  for (size_t i = 0; i < constraints_.size(); ++i) {
    if (composite_state.sub_states[i] == nullptr) {
      return absl::InvalidArgumentError("Encountered null sub-state.");
    }
    ABSL_ASSIGN_OR_RETURN(auto mask, constraints_[i]->ComputeMask(
                                         *composite_state.sub_states[i]));
    if (mask != nullptr) {
      composite_mask->AddMask(std::move(mask));
    }
  }

  return composite_mask;
}

}  // namespace litert::lm
