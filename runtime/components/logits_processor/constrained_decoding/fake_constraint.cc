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

namespace {

// A bitmap implementation that allows only the one specified token.
class SingleAllowedTokenBitmap : public Bitmap {
 public:
  explicit SingleAllowedTokenBitmap(int allowed_token_id)
      : allowed_token_id_(allowed_token_id) {}

  bool Get(int index) const override { return index == allowed_token_id_; }

 private:
  const int allowed_token_id_;
};

}  // namespace

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
