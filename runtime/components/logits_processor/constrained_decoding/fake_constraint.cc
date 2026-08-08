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

#include "runtime/components/logits_processor/constrained_decoding/fake_constraint.h"

#include <memory>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/components/logits_processor/constrained_decoding/bitmap.h"
#include "runtime/components/logits_processor/constrained_decoding/constraint.h"

namespace litert::lm {

std::unique_ptr<Constraint::State> FakeConstraint::Start() const {
  return std::make_unique<FakeState>(0);
}

bool FakeConstraint::IsEnded(const State& state) const {
  const auto& fake_state = static_cast<const FakeState&>(state);
  return fake_state.index() == token_ids_.size();
}

absl::StatusOr<std::unique_ptr<Constraint::State>> FakeConstraint::ComputeNext(
    const State& state, int token) const {
  const auto& fake_state = static_cast<const FakeState&>(state);
  if (fake_state.index() >= token_ids_.size()) {
    return absl::InvalidArgumentError("Invalid state");
  }

  return std::make_unique<FakeState>(fake_state.index() + 1);
}

absl::StatusOr<std::unique_ptr<Bitmap>> FakeConstraint::ComputeBitmap(
    const State& state) const {
  const auto& fake_state = static_cast<const FakeState&>(state);
  return std::make_unique<SingleAllowedTokenBitmap>(
      token_ids_[fake_state.index()]);
}

}  // namespace litert::lm
