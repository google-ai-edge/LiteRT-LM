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
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/functional/function_ref.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/embedding_lookup/embedding_lookup_manager.h"
#include "runtime/components/model_resources.h"
#include "runtime/executor/cancellation_latch.h"
#include "runtime/executor/embedding/embedding_executor_base.h"
#include "runtime/executor/embedding/embedding_executor_settings.h"
#include "runtime/executor/executor_stats.h"
#include "runtime/executor/llm_executor_io_types.h"

namespace litert::lm {

// The EmbeddingLiteRtCompiledModelExecutor runs the two-stage embedding
struct CompiledTextEncoderInfo {
  std::unique_ptr<litert::CompiledModel> compiled_model;
  std::vector<int> expected_input_dimension;
  int embedding_dimension;
  // Sorted map from input sequence length to signature index. Sorted order is
  // required to use lower_bound() to select the smallest signature bucket that
  // fits a given input token length.
  std::map<int, size_t> encoder_signatures;
  absl::flat_hash_map<size_t, std::vector<litert::TensorBuffer>>
      input_buffers_cache;
  absl::flat_hash_map<size_t, std::vector<litert::TensorBuffer>>
      output_buffers_cache;
};

// EmbeddingLiteRtCompiledModelExecutor implements a two-stage embedding
// pipeline:
// Stage 1 (EMBEDDER): Maps token IDs + multimodal soft tokens to sequence
// embedding vectors (`[batch, seq_len, embed_dim]`) using
// EmbeddingLookupManager.
// Stage 2 (TEXT_ENCODER): Executes the transformer model on the sequence
// embedding vectors via litert::CompiledModel to produce a fixed-size
// embedding vector (`[batch, output_dim]`).
class EmbeddingLiteRtCompiledModelExecutor : public EmbeddingExecutorBase {
 public:
  // Compiles the text encoder model from the given resources.
  static absl::StatusOr<CompiledTextEncoderInfo> CompileTextEncoder(
      const EmbeddingExecutorSettings& executor_settings,
      litert::Environment& env, ModelResources& resources);

  // Creates an EmbeddingLiteRtCompiledModelExecutor from the given settings,
  // LiteRT environment, and model resources.
  //
  // The embedder model (kTfLiteEmbedder) is optional. Without it, the executor
  // can only compute embeddings from pre-computed input embeddings (see
  // `ComputeEmbeddingFromInputEmbeddings`), and `ComputeEmbedding` /
  // `ComputeEmbeddingBatch` return a FailedPreconditionError.
  //
  // The executor shares the ownership of `resources`, so the same resources
  // can be shared with other executors (e.g. a
  // `VisionLiteRtCompiledModelExecutor` created from the same model) that can
  // then be created and destroyed independently of this executor. `env` must
  // outlive the executor.
  //
  // Note: `ModelResources` is not thread-safe, and `Create()` mutates
  // `*resources` (e.g. it lazily loads the models). It must not be called
  // concurrently with other users of the same `ModelResources` instance across
  // threads without external synchronization.
  static absl::StatusOr<std::unique_ptr<EmbeddingLiteRtCompiledModelExecutor>>
  Create(EmbeddingExecutorSettings executor_settings, litert::Environment& env,
         std::shared_ptr<ModelResources> resources);

  // Creates an EmbeddingLiteRtCompiledModelExecutor with pre-compiled
  // components.
  static absl::StatusOr<std::unique_ptr<EmbeddingLiteRtCompiledModelExecutor>>
  Create(EmbeddingExecutorSettings executor_settings, litert::Environment& env,
         std::shared_ptr<ModelResources> resources,
         std::unique_ptr<EmbeddingLookupManager> embedding_lookup,
         std::unique_ptr<EmbeddingLookupManager> per_layer_embedding_lookup,
         CompiledTextEncoderInfo compiled_text_encoder_info);

  // Creates an EmbeddingLiteRtCompiledModelExecutor from the given settings and
  // LiteRT environment.
  static absl::StatusOr<std::unique_ptr<EmbeddingLiteRtCompiledModelExecutor>>
  Create(EmbeddingExecutorSettings executor_settings, litert::Environment& env);

  ~EmbeddingLiteRtCompiledModelExecutor() override = default;

  absl::Status StartProfiling() override;
  absl::StatusOr<ExecutorStats> StopProfiling() override;

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

  // Runs the text encoder on pre-computed input embeddings, skipping the
  // embedding lookup. This lets the embedding lookup (and the merging of
  // multimodal embeddings, e.g. vision soft tokens) run outside of this
  // executor, e.g. on another backend.
  //
  // `input_embeddings` holds `[sequence_length, embed_dim]` float values in
  // row-major order, where `embed_dim` is the last dimension of
  // `GetExpectedInputDimension()`. The text encoder signature with the
  // smallest sequence length that fits `sequence_length` is used; the
  // remaining positions are padded with zeros and masked out.
  //
  // Returns the output embedding vector of the text encoder, of size
  // `GetEmbeddingDimension()`. Returns an InvalidArgumentError if the size of
  // `input_embeddings` is not a positive multiple of `embed_dim` or if the
  // sequence is longer than the longest signature, and an UnimplementedError
  // if the text encoder requires per-layer embeddings (which are computed
  // from token IDs).
  absl::StatusOr<std::vector<float>> ComputeEmbeddingFromInputEmbeddings(
      absl::Span<const float> input_embeddings);

  // Same as above, but can be cancelled while it runs.
  //
  // `should_cancel` is checked before the text encoder runs and, for the
  // duration of this call only, after every op executed by the LiteRT
  // interpreter. Once it returns true, the call stops at the next check;
  // `should_cancel` is not called again for this call. An op delegated to an
  // accelerator (e.g. the NPU dispatch op running a whole compiled graph)
  // cannot be interrupted, so cancellation takes effect at op boundaries.
  // Calls without `should_cancel` do not pay the per-op check.
  //
  // Args:
  //   input_embeddings: `[sequence_length, embed_dim]` float input embeddings,
  //     as for the overload above.
  //   should_cancel: Returns true to request cancellation. Called on the thread
  //     running this method.
  //
  // Returns:
  //   The output embedding vector of the text encoder, a CancelledError if
  //   `should_cancel` returned true, or the errors of the overload above. The
  //   executor remains usable after a cancelled call.
  absl::StatusOr<std::vector<float>> ComputeEmbeddingFromInputEmbeddings(
      absl::Span<const float> input_embeddings,
      absl::FunctionRef<bool()> should_cancel);

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
  // The buffers and input indices of a text encoder run.
  struct EncoderRun {
    // The selected signature, i.e. the one with the smallest sequence length
    // that fits the input.
    size_t signature_index = 0;
    // The sequence length of the selected signature.
    size_t padded_sequence_length = 0;
    std::vector<litert::TensorBuffer>* input_buffers = nullptr;
    std::vector<litert::TensorBuffer>* output_buffers = nullptr;
    size_t embeddings_buffer_index = 0;
    std::optional<size_t> input_mask_buffer_index;
    std::optional<size_t> per_layer_embeddings_buffer_index;
  };

  // Selects the text encoder signature for `sequence_length` and gets (or
  // creates) its buffers.
  absl::StatusOr<EncoderRun> PrepareEncoderRun(size_t sequence_length);

  // Fills the input mask of `run`, runs the text encoder and returns its output
  // embedding vector. The embeddings (and per-layer embeddings, if any) inputs
  // of `run` must be filled.
  //
  // Args:
  //   run: The text encoder run, with its embeddings inputs filled.
  //   sequence_length: The number of input embeddings.
  //   latch: The latch of a cancellable call, or nullptr. If set, the call is
  //     cancelled before the run, and an interrupted run is reported as
  //     cancelled.
  //
  // Returns:
  //   The output embedding vector, or a CancelledError if `latch` is
  //   cancelled.
  absl::StatusOr<std::vector<float>> FinishEncoderRun(const EncoderRun& run,
                                                      size_t sequence_length,
                                                      CancellationLatch* latch);

  // Implements ComputeEmbeddingFromInputEmbeddings(), cancellable with `latch`
  // if not null (see FinishEncoderRun).
  absl::StatusOr<std::vector<float>> EncodeInputEmbeddings(
      absl::Span<const float> input_embeddings, CancellationLatch* latch);

  absl::StatusOr<std::vector<float>> RunEncoderForTokens(
      absl::Span<const int32_t> tokens);

  EmbeddingLiteRtCompiledModelExecutor(
      EmbeddingExecutorSettings executor_settings, litert::Environment& env,
      std::shared_ptr<ModelResources> resources,
      std::unique_ptr<EmbeddingLookupManager> embedding_lookup,
      std::unique_ptr<EmbeddingLookupManager> per_layer_embedding_lookup,
      std::unique_ptr<litert::CompiledModel> compiled_model,
      std::vector<int> expected_input_dimension, int embedding_dimension,
      std::map<int, size_t> encoder_signatures,
      absl::flat_hash_map<size_t, std::vector<litert::TensorBuffer>>
          input_buffers_cache,
      absl::flat_hash_map<size_t, std::vector<litert::TensorBuffer>>
          output_buffers_cache);

  EmbeddingExecutorSettings executor_settings_;
  litert::Environment& env_;
  std::shared_ptr<ModelResources> resources_;
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

  std::optional<ExecutorStats> latency_stats_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_EMBEDDING_LITERT_COMPILED_MODEL_EXECUTOR_H_
