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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_EMBEDDING_LITERT_COMPILED_MODEL_EXECUTOR_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_EMBEDDING_LITERT_COMPILED_MODEL_EXECUTOR_H_

#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/embedding_lookup/embedding_lookup_manager.h"
#include "runtime/components/model_resources.h"
#include "runtime/executor/embedding/embedding_executor_base.h"
#include "runtime/executor/embedding/embedding_executor_settings.h"
#include "runtime/executor/llm_executor_io_types.h"

namespace litert::lm {

// The EmbeddingLiteRtCompiledModelExecutor runs the two-stage embedding
// pipeline:
// Stage 1 (EMBEDDER): Maps token IDs + multimodal soft tokens to sequence
// embedding vectors (`[batch, seq_len, embed_dim]`) using
// EmbeddingLookupManager.
// Stage 2 (TEXT_ENCODER): Executes the transformer model on the sequence
// embedding vectors via litert::CompiledModel to produce a fixed-size
// embedding vector (`[batch, output_dim]`).
class EmbeddingLiteRtCompiledModelExecutor : public EmbeddingExecutorBase {
 public:
  // Creates an EmbeddingLiteRtCompiledModelExecutor from the given settings,
  // LiteRT environment, and model resources.
  static absl::StatusOr<std::unique_ptr<EmbeddingLiteRtCompiledModelExecutor>>
  Create(EmbeddingExecutorSettings executor_settings, litert::Environment& env,
         std::unique_ptr<ModelResources> resources);

  // Creates an EmbeddingLiteRtCompiledModelExecutor from the given settings and
  // LiteRT environment.
  static absl::StatusOr<std::unique_ptr<EmbeddingLiteRtCompiledModelExecutor>>
  Create(EmbeddingExecutorSettings executor_settings, litert::Environment& env);

  ~EmbeddingLiteRtCompiledModelExecutor() override = default;

  using EmbeddingExecutorBase::ComputeEmbedding;
  using EmbeddingExecutorBase::ComputeEmbeddingBatch;

  // Computes the embedding output for the given inputs.
  absl::StatusOr<EmbeddingOutput> ComputeEmbedding(
      const ExecutorInputs& inputs,
      const ComputeEmbeddingOptions& options) override;

  // Computes a batch of embedding outputs for the given batch of inputs.
  absl::StatusOr<std::vector<EmbeddingOutput>> ComputeEmbeddingBatch(
      const std::vector<ExecutorInputs>& batch_inputs,
      const ComputeEmbeddingOptions& options) override;

  // Returns the name of the hardware backend used by this executor.
  absl::string_view ExecutorBackendName() const override;

  // Returns the expected input dimensions of the text encoder model
  // (`[batch, seq_len, embed_dim]`).
  absl::StatusOr<std::vector<int>> GetExpectedInputDimension() const override;

  // Returns the dimension of the output embedding vector produced by the text
  // encoder model.
  absl::StatusOr<int> GetEmbeddingDimension() const override;

  // Returns a pointer to the LiteRT environment used by this executor.
  litert::Environment* GetEnvironment() const override;

 private:
  absl::StatusOr<std::vector<float>> RunEncoderForTokens(
      absl::Span<const int32_t> tokens);

  EmbeddingLiteRtCompiledModelExecutor(
      EmbeddingExecutorSettings executor_settings, litert::Environment& env,
      std::unique_ptr<ModelResources> resources,
      std::unique_ptr<EmbeddingLookupManager> embedding_lookup,
      std::unique_ptr<EmbeddingLookupManager> per_layer_embedding_lookup,
      std::unique_ptr<litert::CompiledModel> compiled_model,
      std::vector<int> expected_input_dimension, int embedding_dimension,
      std::map<int, size_t> encoder_signatures);

  EmbeddingExecutorSettings executor_settings_;
  litert::Environment& env_;
  std::unique_ptr<ModelResources> resources_;
  std::unique_ptr<EmbeddingLookupManager> embedding_lookup_;
  std::unique_ptr<EmbeddingLookupManager> per_layer_embedding_lookup_;
  std::unique_ptr<litert::CompiledModel> compiled_model_;
  std::vector<int> expected_input_dimension_;
  int embedding_dimension_;
  std::string backend_name_;

  // Text encoder signatures: Sequence Length -> Signature Index
  std::map<int, size_t> encoder_signatures_;

  // Input buffer map: Signature Index -> Input Buffer
  absl::flat_hash_map<size_t, std::vector<litert::TensorBuffer>>
      input_buffers_cache_;

  // Output buffer map: Signature Index -> Output Buffers
  absl::flat_hash_map<size_t, std::vector<litert::TensorBuffer>>
      output_buffers_cache_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_EMBEDDING_LITERT_COMPILED_MODEL_EXECUTOR_H_
