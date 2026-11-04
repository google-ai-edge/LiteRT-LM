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

#include "runtime/executor/cancellation_latch.h"

#include "absl/functional/function_ref.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_compiled_model.h"  // from @litert

namespace litert::lm {

CancellationLatch::CancellationLatch(
    absl::FunctionRef<bool()> should_cancel,
    absl::Span<litert::CompiledModel* const> compiled_models)
    : should_cancel_(should_cancel),
      compiled_models_(compiled_models.begin(), compiled_models.end()) {
  for (litert::CompiledModel* compiled_model : compiled_models_) {
    compiled_model->SetCancellationFunction(this, [](void* data) {
      return static_cast<CancellationLatch*>(data)->IsCancelled();
    });
  }
}

CancellationLatch::~CancellationLatch() {
  for (litert::CompiledModel* compiled_model : compiled_models_) {
    compiled_model->SetCancellationFunction(/*data=*/nullptr,
                                            [](void*) { return false; });
  }
}

bool CancellationLatch::IsCancelled() {
  if (!cancelled_ && should_cancel_()) {
    cancelled_ = true;
  }
  return cancelled_;
}

}  // namespace litert::lm
