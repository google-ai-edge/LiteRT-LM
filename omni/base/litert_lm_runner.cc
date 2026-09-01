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

#include "omni/base/litert_lm_runner.h"

#include <memory>
#include <utility>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/executor/llm_executor_base.h"
#include "runtime/executor/llm_executor_io_types.h"

namespace litert::omni {

LiteRtLmRunnerImpl::LiteRtLmRunnerImpl(
    std::unique_ptr<lm::LlmExecutorBase> absl_nonnull executor,
    std::unique_ptr<lm::ModelResources> model_resources)
    : model_resources_(std::move(model_resources)),
      owned_executor_(std::move(executor)) {}

absl::Status LiteRtLmRunnerImpl::Prefill(const lm::ExecutorInputs& inputs) {
  return owned_executor_->Prefill(inputs);
}

absl::StatusOr<TensorBuffer> LiteRtLmRunnerImpl::Decode(
    const lm::ExecutorInputs& inputs) {
  LITERT_ASSIGN_OR_RETURN(auto output, owned_executor_->DecodeLogits(inputs));
  return output;
}

absl::Status LiteRtLmRunnerImpl::Reset() { return owned_executor_->Reset(); }

}  // namespace litert::omni
