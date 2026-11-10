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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_UTILS_CONSTRAINT_ADAPTER_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_UTILS_CONSTRAINT_ADAPTER_H_

#include <memory>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/components/constrained_decoding/bitmap.h"
#include "runtime/components/constrained_decoding/constraint.h"
#include "runtime/executor/gpu_artisan/utils/bitmap.h"
#include "runtime/executor/gpu_artisan/utils/constraint_utils.h"

namespace odml::infra::llm_utils {

class BitmapAdapter : public odml::infra::llm_utils::Bitmap {
 public:
  explicit BitmapAdapter(std::unique_ptr<::litert::lm::Bitmap> bitmap);
  bool Get(int index) const override;

 private:
  std::unique_ptr<::litert::lm::Bitmap> bitmap_;
};

// Wraps a litert::lm::Constraint and exposes it as an
// odml::infra::llm_utils::Constraint.
class ConstraintAdapter : public Constraint {
 public:
  // `constraint` is not owned by the adapter.
  explicit ConstraintAdapter(::litert::lm::Constraint* constraint);

  class AdapterState : public Constraint::State {
   public:
    explicit AdapterState(
        std::unique_ptr<::litert::lm::Constraint::State> state);

    const ::litert::lm::Constraint::State& Get() const;

   private:
    std::unique_ptr<::litert::lm::Constraint::State> state_;
  };

  std::shared_ptr<Constraint::State> Start() const override;

  bool IsEnded(const Constraint::State& state) const override;

  absl::StatusOr<std::shared_ptr<Constraint::State>> ComputeNext(
      const Constraint::State& state, int token) const override;

  absl::StatusOr<std::unique_ptr<Bitmap>> ComputeBitmap(
      const Constraint::State& state) const override;

  ::litert::lm::Constraint* GetConstraint() const { return constraint_; }

 private:
  ::litert::lm::Constraint* constraint_;  // Not owned.
};

}  // namespace odml::infra::llm_utils

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_UTILS_CONSTRAINT_ADAPTER_H_
