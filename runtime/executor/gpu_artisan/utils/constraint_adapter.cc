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

#include "runtime/executor/gpu_artisan/utils/constraint_adapter.h"

#include <memory>
#include <utility>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/components/constrained_decoding/bitmap.h"
#include "runtime/components/constrained_decoding/constraint.h"
#include "runtime/executor/gpu_artisan/utils/bitmap.h"
#include "runtime/executor/gpu_artisan/utils/constraint_utils.h"
#include "runtime/util/status_macros.h"

namespace odml::infra::llm_utils {

BitmapAdapter::BitmapAdapter(std::unique_ptr<litert::lm::Bitmap> bitmap)
    : bitmap_(std::move(bitmap)) {}

bool BitmapAdapter::Get(int index) const { return bitmap_->Get(index); }

ConstraintAdapter::ConstraintAdapter(litert::lm::Constraint* constraint)
    : constraint_(constraint) {}

ConstraintAdapter::AdapterState::AdapterState(
    std::unique_ptr<litert::lm::Constraint::State> state)
    : state_(std::move(state)) {}

const litert::lm::Constraint::State& ConstraintAdapter::AdapterState::Get()
    const {
  return *state_;
}

std::shared_ptr<Constraint::State> ConstraintAdapter::Start() const {
  return std::make_shared<AdapterState>(constraint_->Start());
}

bool ConstraintAdapter::IsEnded(const Constraint::State& state) const {
  return constraint_->IsEnded(static_cast<const AdapterState&>(state).Get());
}

absl::StatusOr<std::shared_ptr<Constraint::State>>
ConstraintAdapter::ComputeNext(const Constraint::State& state,
                               int token) const {
  ASSIGN_OR_RETURN(auto next_state,
                   constraint_->ComputeNext(
                       static_cast<const AdapterState&>(state).Get(), token));
  return std::make_shared<AdapterState>(std::move(next_state));
}

absl::StatusOr<std::unique_ptr<Bitmap>> ConstraintAdapter::ComputeBitmap(
    const Constraint::State& state) const {
  ASSIGN_OR_RETURN(auto bitmap,
                   constraint_->ComputeBitmap(
                       static_cast<const AdapterState&>(state).Get()));
  return std::make_unique<BitmapAdapter>(std::move(bitmap));
}

}  // namespace odml::infra::llm_utils
