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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_BASE_LITERT_LM_RUNNER_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_BASE_LITERT_LM_RUNNER_H_

#include <memory>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/executor/llm_executor_base.h"
#include "runtime/executor/llm_executor_io_types.h"

namespace litert::omni {

// Interface for LiteRT-LM model execution in omni pipelines.
class LiteRtLmRunner {
 public:
  virtual ~LiteRtLmRunner() = default;

  // Runs the prefill phase on input embeddings/tokens.
  virtual absl::Status Prefill(const lm::ExecutorInputs& inputs) = 0;

  // Runs one step of decoding and returns output logits TensorBuffer.
  virtual absl::StatusOr<TensorBuffer> Decode(
      const lm::ExecutorInputs& inputs) = 0;

  // Resets the runner state and KV cache.
  virtual absl::Status Reset() = 0;
};

// Implementation of LiteRtLmRunner wrapping an owned LlmExecutorBase instance.
class LiteRtLmRunnerImpl : public LiteRtLmRunner {
 public:
  explicit LiteRtLmRunnerImpl(
      std::unique_ptr<lm::LlmExecutorBase> absl_nonnull executor,
      std::unique_ptr<lm::ModelResources> model_resources = nullptr);

  ~LiteRtLmRunnerImpl() override = default;

  absl::Status Prefill(const lm::ExecutorInputs& inputs) override;

  absl::StatusOr<TensorBuffer> Decode(
      const lm::ExecutorInputs& inputs) override;

  absl::Status Reset() override;

 private:
  std::unique_ptr<lm::ModelResources> model_resources_;
  std::unique_ptr<lm::LlmExecutorBase> owned_executor_;
};

}  // namespace litert::omni

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_BASE_LITERT_LM_RUNNER_H_
