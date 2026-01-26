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
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "runtime/components/constrained_decoding/bitmap.h"
#include "runtime/components/constrained_decoding/constraint.h"
#include "llguidance.h"

namespace litert::lm {

namespace {

std::vector<bool> SampleMaskToVector(const uint32_t* sample_mask,
                                     size_t vocab_size, bool is_stop,
                                     int eos_token_id) {
  if (sample_mask == nullptr) {
    if (is_stop) {
      // If stopped, only allow EOS.
      std::vector<bool> mask_vector(vocab_size, false);
      if (eos_token_id >= 0 && eos_token_id < vocab_size) {
        mask_vector[eos_token_id] = true;
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

}  // namespace

std::unique_ptr<Constraint::State> LlgConstraint::Start() const {
  ::LlgConstraint* llg_constraint =
      llg_clone_constraint(llg_constraint_owner_.llg_constraint());
  return std::make_unique<LlgConstraint::LlgState>(llg_constraint);
}

bool LlgConstraint::IsEnded(const LlgConstraint::State& state) const {
  const auto& llg_state = static_cast<const LlgConstraint::LlgState&>(state);

  LlgMaskResult mask_res;
  if (llg_compute_mask(llg_state.llg_constraint(), &mask_res) != 0) {
    std::string error_message = llg_get_error(llg_state.llg_constraint());
    ABSL_LOG(ERROR) << "Failed to compute mask: " << error_message;
    return false;
  }
  return mask_res.is_stop;
  ;
}

int LlgConstraint::GetVocabularySize() const { return vocab_size_; }

absl::StatusOr<std::unique_ptr<Constraint::State>> LlgConstraint::ComputeNext(
    const Constraint::State& state, int token) const {
  const auto& llg_state = static_cast<const LlgConstraint::LlgState&>(state);

  LlgCommitResult commit_res;
  if (llg_commit_token(llg_state.llg_constraint(), token, &commit_res) != 0) {
    std::string error_message = llg_get_error(llg_state.llg_constraint());
    return absl::InternalError(
        absl::StrCat("Failed to commit token: ", error_message));
  }

  return std::make_unique<LlgConstraint::LlgState>(llg_state);
}

absl::StatusOr<std::unique_ptr<Bitmap>> LlgConstraint::ComputeBitmap(
    const Constraint::State& state) const {
  const auto& llg_state = static_cast<const LlgConstraint::LlgState&>(state);

  LlgMaskResult mask_res;
  if (llg_compute_mask(llg_state.llg_constraint(), &mask_res) != 0) {
    std::string error_message = llg_get_error(llg_state.llg_constraint());
    return absl::InternalError(
        absl::StrCat("Failed to compute mask: ", error_message));
  }

  std::vector<bool> mask_vector = SampleMaskToVector(
      mask_res.sample_mask, vocab_size_, mask_res.is_stop, eos_token_id_);
  return std::make_unique<LlgBitmap>(std::move(mask_vector));
}

}  // namespace litert::lm
