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

#include "runtime/components/constrained_decoding/llg_constraint.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "runtime/components/constrained_decoding/bitmap.h"
#include "runtime/components/constrained_decoding/constraint.h"
#include "runtime/components/constrained_decoding/logit_mask.h"
#include "llguidance.h"

namespace litert::lm {

namespace {

std::vector<bool> SampleMaskToVector(const uint32_t* sample_mask,
                                     size_t vocab_size, bool is_stop,
                                     absl::Span<const int> eos_token_ids) {
  if (sample_mask == nullptr) {
    if (is_stop) {
      // If stopped, only allow EOS.
      std::vector<bool> mask_vector(vocab_size, false);
      for (int eos_token_id : eos_token_ids) {
        if (eos_token_id >= 0 && eos_token_id < vocab_size) {
          mask_vector[eos_token_id] = true;
        }
      }
      return mask_vector;
    } else {
      // If not stopped but mask is null, it implies no constraints are active
      // (unconstrained), so we allow all tokens.
      return std::vector<bool>(vocab_size, true);
    }
  }
  std::vector<bool> mask_vector;
  mask_vector.reserve(vocab_size);
  for (size_t i = 0; i < vocab_size; ++i) {
    mask_vector.push_back(sample_mask[i / 32] & (1 << (i % 32)));
  }
  return mask_vector;
}

// Returns the mask for a stopped constraint, which only allows EOS.
std::unique_ptr<LogitMask> StoppedMask(int vocab_size,
                                       absl::Span<const int> eos_token_ids) {
  return BitmapLogitMask::CreateFromAllowedTokens(vocab_size, eos_token_ids);
}

// llguidance reports a constraint in an error state as stopped, so this must be
// checked before llg_is_stopped() to avoid treating errors as a normal end.
absl::Status CheckNoError(const ::LlgConstraint* llg_constraint) {
  if (const char* error = llg_get_error(llg_constraint); error != nullptr) {
    return absl::InternalError(
        absl::StrCat("Constraint is in an error state: ", error));
  }
  return absl::OkStatus();
}

}  // namespace

std::unique_ptr<Constraint::State> LlgConstraint::Start() const {
  ::LlgConstraint* llg_constraint =
      llg_clone_constraint(llg_constraint_owner_.llg_constraint());
  return std::make_unique<LlgConstraint::LlgState>(llg_constraint);
}

bool LlgConstraint::IsEnded(const LlgConstraint::State& state) const {
  const auto& llg_state = static_cast<const LlgConstraint::LlgState&>(state);
  return llg_is_stopped(llg_state.llg_constraint());
}

int LlgConstraint::GetVocabularySize() const { return vocab_size_; }

absl::Status LlgConstraint::CommitToken(const LlgState& state,
                                        int token) const {
  ::LlgConstraint* llg_constraint = state.llg_constraint();
  if (!state.mask_computed() && !llg_is_stopped(llg_constraint)) {
    LlgMaskResult mask_res;
    if (llg_compute_mask(llg_constraint, &mask_res) != 0) {
      std::string error_message = llg_get_error(llg_constraint);
      return absl::InternalError(
          absl::StrCat("Failed to compute mask: ", error_message));
    }
  }

  LlgCommitResult commit_res;
  if (llg_commit_token(llg_constraint, token, &commit_res) != 0) {
    std::string error_message = llg_get_error(llg_constraint);
    return absl::InternalError(
        absl::StrCat("Failed to commit token: ", error_message));
  }
  state.set_mask_computed(false);
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<Constraint::State>> LlgConstraint::ComputeNext(
    const Constraint::State& state, int token) const {
  const auto& llg_state = static_cast<const LlgConstraint::LlgState&>(state);
  auto next_state = std::make_unique<LlgConstraint::LlgState>(llg_state);
  ABSL_RETURN_IF_ERROR(CommitToken(*next_state, token));
  return next_state;
}

absl::StatusOr<std::unique_ptr<LogitMask>> LlgConstraint::ComputeMask(
    const Constraint::State& state) const {
  const auto& llg_state = static_cast<const LlgConstraint::LlgState&>(state);

  ABSL_RETURN_IF_ERROR(CheckNoError(llg_state.llg_constraint()));
  if (llg_is_stopped(llg_state.llg_constraint())) {
    llg_state.set_mask_computed(true);
    return StoppedMask(vocab_size_, eos_token_ids_);
  }

  LlgMaskResult mask_res;
  if (llg_compute_mask(llg_state.llg_constraint(), &mask_res) != 0) {
    std::string error_message = llg_get_error(llg_state.llg_constraint());
    return absl::InternalError(
        absl::StrCat("Failed to compute mask: ", error_message));
  }
  llg_state.set_mask_computed(true);

  if (mask_res.sample_mask == nullptr) {
    if (mask_res.is_stop) {
      return StoppedMask(vocab_size_, eos_token_ids_);
    } else {
      return BitmapLogitMask::CreateAllAllowed(vocab_size_);
    }
  }

  size_t num_u32 = (vocab_size_ + 31) / 32;
  return std::make_unique<BitmapLogitMask>(
      vocab_size_, absl::MakeConstSpan(mask_res.sample_mask, num_u32));
}

absl::StatusOr<std::unique_ptr<Bitmap>> LlgConstraint::ComputeBitmap(
    const Constraint::State& state) const {
  const auto& llg_state = static_cast<const LlgConstraint::LlgState&>(state);

  ABSL_RETURN_IF_ERROR(CheckNoError(llg_state.llg_constraint()));
  if (llg_is_stopped(llg_state.llg_constraint())) {
    llg_state.set_mask_computed(true);
    std::vector<bool> mask_vector = SampleMaskToVector(
        /*sample_mask=*/nullptr, vocab_size_, /*is_stop=*/true, eos_token_ids_);
    return std::make_unique<LlgBitmap>(std::move(mask_vector));
  }

  LlgMaskResult mask_res;
  if (llg_compute_mask(llg_state.llg_constraint(), &mask_res) != 0) {
    std::string error_message = llg_get_error(llg_state.llg_constraint());
    return absl::InternalError(
        absl::StrCat("Failed to compute mask: ", error_message));
  }
  llg_state.set_mask_computed(true);

  std::vector<bool> mask_vector = SampleMaskToVector(
      mask_res.sample_mask, vocab_size_, mask_res.is_stop, eos_token_ids_);
  return std::make_unique<LlgBitmap>(std::move(mask_vector));
}

}  // namespace litert::lm
