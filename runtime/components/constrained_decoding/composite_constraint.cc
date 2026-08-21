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

#include "runtime/components/constrained_decoding/composite_constraint.h"

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "runtime/components/constrained_decoding/constraint.h"
#include "runtime/components/constrained_decoding/logit_mask.h"
#include "runtime/components/constrained_decoding/bitmap.h"

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

  absl::StatusOr<std::unique_ptr<Bitmap>> ComputeBitmap(
      const State& state) const override {
    return constraint_.ComputeBitmap(state);
  }

 private:
  Constraint& constraint_;
};

// A Bitmap that combines multiple underlying bitmaps. A token index is allowed
// only if it is allowed by all non-null underlying bitmaps (logical AND).
class CompositeBitmap : public Bitmap {
 public:
  explicit CompositeBitmap(std::vector<std::unique_ptr<Bitmap>> bitmaps)
      : bitmaps_(std::move(bitmaps)) {}

  bool Get(int index) const override {
    for (const auto& bitmap : bitmaps_) {
      if (bitmap != nullptr && !bitmap->Get(index)) {
        return false;
      }
    }
    return true;
  }

 private:
  std::vector<std::unique_ptr<Bitmap>> bitmaps_;
};

// A Bitmap adapter that wraps an existing BitmapLogitMask.
class BitmapLogitMaskWrapper : public Bitmap {
 public:
  explicit BitmapLogitMaskWrapper(std::unique_ptr<BitmapLogitMask> mask)
      : mask_(std::move(mask)) {}

  bool Get(int index) const override {
    if (mask_ == nullptr) {
      return true;
    }
    return mask_->IsAllowed(index);
  }

 private:
  std::unique_ptr<BitmapLogitMask> mask_;
};

// Converts a LogitMask to a Bitmap.
//
// - Returns AllAllowedBitmap if `mask` is null or of type kSparse (soft
//   constraint / penalty).
// - Wraps `mask` directly in BitmapLogitMaskWrapper if it is of type kBitmap.
// - Returns an InvalidArgumentError if `mask` is of any other type (such as
//   kComposite or kCustom) that cannot be directly converted to a Bitmap.
absl::StatusOr<std::unique_ptr<Bitmap>> LogitMaskToBitmap(
    std::unique_ptr<LogitMask> mask) {
  if (mask == nullptr) {
    return std::make_unique<AllAllowedBitmap>();
  }
  if (mask->GetType() == MaskType::kBitmap) {
    return std::make_unique<BitmapLogitMaskWrapper>(
        std::unique_ptr<BitmapLogitMask>(
            static_cast<BitmapLogitMask*>(mask.release())));
  }
  return absl::InvalidArgumentError(
      "Cannot convert non-bitmap LogitMask to Bitmap.");
}

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

absl::StatusOr<std::unique_ptr<Bitmap>> CompositeConstraint::ComputeBitmap(
    const State& state) const {
  if (constraints_.empty()) {
    return std::make_unique<AllAllowedBitmap>();
  }
  const auto& composite_state = static_cast<const CompositeState&>(state);
  if (composite_state.sub_states.size() != constraints_.size()) {
    return absl::InvalidArgumentError(
        "CompositeState sub-states size does not match constraints count.");
  }

  std::vector<std::unique_ptr<Bitmap>> bitmaps;
  bitmaps.reserve(constraints_.size());

  for (size_t i = 0; i < constraints_.size(); ++i) {
    if (composite_state.sub_states[i] == nullptr) {
      return absl::InvalidArgumentError("Encountered null sub-state.");
    }
    auto bitmap_or =
        constraints_[i]->ComputeBitmap(*composite_state.sub_states[i]);
    if (bitmap_or.ok()) {
      if (*bitmap_or != nullptr) {
        bitmaps.push_back(std::move(*bitmap_or));
      }
    } else if (bitmap_or.status().code() == absl::StatusCode::kUnimplemented) {
      // Fallback: If a sub-constraint does not implement ComputeBitmap, compute
      // its LogitMask and convert it to a Bitmap.
      ABSL_ASSIGN_OR_RETURN(auto mask, constraints_[i]->ComputeMask(
                                           *composite_state.sub_states[i]));
      ABSL_ASSIGN_OR_RETURN(auto mask_bitmap,
                            LogitMaskToBitmap(std::move(mask)));
      if (mask_bitmap != nullptr) {
        bitmaps.push_back(std::move(mask_bitmap));
      }
    } else {
      return bitmap_or.status();
    }
  }

  return std::make_unique<CompositeBitmap>(std::move(bitmaps));
}

}  // namespace litert::lm
