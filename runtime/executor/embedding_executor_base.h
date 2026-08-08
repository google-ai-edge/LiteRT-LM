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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_EMBEDDING_EXECUTOR_BASE_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_EMBEDDING_EXECUTOR_BASE_H_

#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "runtime/executor/llm_executor_io_types.h"

namespace litert::lm {

// The Embedding Executor serves as a wrapper around converted text encoder
// models across diverse hardware accelerators like CPUs, GPUs, and NPUs.
class EmbeddingExecutorBase {
 public:
  virtual ~EmbeddingExecutorBase() = default;

  // Computes the single fixed-size embedding vector for the given inputs.
  virtual absl::StatusOr<std::vector<float>> ComputeEmbedding(
      const ExecutorInputs& inputs) = 0;

  // Computes a batch of embedding vectors for the given batch of inputs.
  virtual absl::StatusOr<std::vector<std::vector<float>>> ComputeEmbeddingBatch(
      const std::vector<ExecutorInputs>& batch_inputs) = 0;

  virtual absl::string_view ExecutorBackendName() const = 0;

  // Get expected input dimension of the text encoder model.
  virtual absl::StatusOr<std::vector<int>> GetExpectedInputDimension()
      const = 0;

  // Get the embedding dimension output by the text encoder model.
  virtual absl::StatusOr<int> GetEmbeddingDimension() const = 0;

  // Gets the litert environment used by the executor.
  virtual ::litert::Environment* GetEnvironment() const = 0;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_EMBEDDING_EXECUTOR_BASE_H_
