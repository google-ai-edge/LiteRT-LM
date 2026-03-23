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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LITERT_NPU_COMPILED_MODEL_EXECUTOR_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LITERT_NPU_COMPILED_MODEL_EXECUTOR_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/embedding_lookup/embedding_lookup_manager.h"
#include "runtime/components/model_resources.h"
#include "runtime/executor/litert_compiled_model_executor_utils.h"
#include "runtime/executor/llm_executor.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/executor/llm_executor_processed_tokens.h"
#include "runtime/executor/llm_executor_settings.h"

namespace litert::lm {

// Component intended to be used with an NPU variant of Gemma3.
class LlmLiteRtNpuCompiledModelExecutor : public LlmExecutor {
 public:
  // Holds the latency breakdown stats for the executor.
  // TODO(b/405424188): Use 'litert::lm::BenchmarkInfo' instead.
  struct LatencyStats {
    // Prefill latency stats.
    uint64_t prefill_e2e_latency_us = 0;
    int prefill_num_tokens = 0;
    uint64_t prefill_prepare_input_latency_us = 0;
    uint64_t prefill_embedder_inference_latency_us = 0;
    std::optional<uint64_t> prefill_embedder_per_layer_inference_latency_us =
        std::nullopt;
    uint64_t prefill_mask_inference_latency_us = 0;
    uint64_t prefill_rope_inference_latency_us = 0;
    uint64_t prefill_llm_inference_latency_us = 0;
    uint64_t prefill_cache_update_inference_latency_us = 0;

    // Decode latency stats.
    uint64_t decode_e2e_latency_us = 0;
    int decode_num_tokens = 0;
    uint64_t decode_prepare_input_latency_us = 0;
    uint64_t decode_embedder_inference_latency_us = 0;
    std::optional<uint64_t> decode_embedder_per_layer_inference_latency_us =
        std::nullopt;
    uint64_t decode_mask_inference_latency_us = 0;
    uint64_t decode_rope_inference_latency_us = 0;
    uint64_t decode_llm_inference_latency_us = 0;
    uint64_t decode_cache_update_inference_latency_us = 0;
    uint64_t decode_sampling_latency_us = 0;
    std::string DebugString() const;
  };

  bool ShouldLogPerformance() const;

  // Creates an executor from the resources.
  static absl::StatusOr<std::unique_ptr<LlmLiteRtNpuCompiledModelExecutor>>
  Create(const LlmExecutorSettings& executor_settings,
         ModelResources& resources, Environment& env);

  // Input APIs:
  // Basic API to trigger the "prefill" or "prefix" process.
  // Input is token ids with shape `[batch, sequence_length]`
  absl::Status Prefill(const ExecutorInputs& inputs) override;

  // Advanced API to allow customized query parameters.
  // Input is token ids with shape `[batch, sequence_length]`
  absl::Status Prefill(const ExecutorInputs& inputs,
                       const ExecutorPrefillParams& params) override;

  // Output APIs:
  // Basic API to trigger the "decode" process.
  absl::StatusOr<std::vector<std::vector<int>>> Decode() override;

  absl::StatusOr<std::vector<std::vector<int>>> Decode(
      const ExecutorDecodeParams& decode_params);

  absl::string_view ExecutorBackendName() const override {
    return "LiteRT NPU Compiled Model";
  }

  ~LlmLiteRtNpuCompiledModelExecutor() override;

  // Gets the current step of the executor.
  // Public API, the return value is the current step that user expects (e.g.
  // users prefill 100 tokens, then they expect the current step to be 100).
  absl::StatusOr<int> GetCurrentStep() const override { return current_step_; }

  absl::StatusOr<int> GetVocabSize() override;

  absl::StatusOr<LlmExecutorSettings> GetExecutorSettings() const override {
    return executor_settings_;
  };

  ::litert::Environment* GetEnvironment() const override { return &env_; }
  // Prints the latency stats for the executor.  Intended to be used for
  // profiling.
  LatencyStats GetLatencyStats() const;

  // Resets all of the internal states.
  absl::Status Reset() override;

  absl::Status UseLoRA(std::optional<uint32_t> lora_id) override;

  void SetEmbeddingApiMode(absl::string_view mode) override {
    is_bulk_embedding_ = (mode == "bulk");
  }

 private:
  static litert::Expected<litert::Options> CreateLiteRtOptions(
      const LlmExecutorSettings& settings);

  static litert::Expected<litert::Options> CreateCpuLiteRtOptions();

  // Holds the tensor buffer maps for the inference of a precompiled model,
  // both for prefill and decode.
  struct InferenceContext {
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
        prefill_input_buffers;
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
        prefill_output_buffers;
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
        decode_input_buffers;
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
        decode_output_buffers;
    InferenceContext(
        absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
            prefill_input_buffers,
        absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
            prefill_output_buffers,
        absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
            decode_input_buffers,
        absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
            decode_output_buffers);
  };

  // Holds the context for the embedder model.
  struct EmbedderContext {
    ::litert::Model embedder_model;
    ::litert::CompiledModel embedder_compiled_model;
    InferenceContext inference_context;
    EmbedderContext(
        ::litert::CompiledModel embedder_compiled_model,
        absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
            prefill_input_buffers,
        absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
            prefill_output_buffers,
        absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
            decode_input_buffers,
        absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
            decode_output_buffers);
  };

  // Holds the context for the embedder per layer model.
  struct EmbedderPerLayerContext {
    ::litert::CompiledModel embedder_per_layer_compiled_model;
    InferenceContext inference_context;
    EmbedderPerLayerContext(
        ::litert::CompiledModel embedder_per_layer_compiled_model,
        absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
            prefill_input_buffers,
        absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
            prefill_output_buffers,
        absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
            decode_input_buffers,
        absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
            decode_output_buffers)
        : embedder_per_layer_compiled_model(
              std::move(embedder_per_layer_compiled_model)),
          inference_context(std::move(prefill_input_buffers),
                            std::move(prefill_output_buffers),
                            std::move(decode_input_buffers),
                            std::move(decode_output_buffers)) {}
  };

  // Holds the context for the NPU auxiliary model, which contains several
  // signatures for Mask, RoPE and KV cache update computation.
  struct NpuAuxiliaryContext {
    ::litert::CompiledModel npu_auxiliary_compiled_model;
    explicit NpuAuxiliaryContext(
        ::litert::CompiledModel npu_auxiliary_compiled_model);
  };

 protected:
  LlmLiteRtNpuCompiledModelExecutor(
      LlmExecutorSettings executor_settings, Environment& llm_env,
      EmbedderContext embedder_context,
      NpuAuxiliaryContext npu_auxiliary_context, InferenceContext mask_context,
      InferenceContext rope_context, ::litert::CompiledModel llm_compiled_model,
      InferenceContext llm_inference_context,
      InferenceContext cache_update_inference_context,
      SortedPrefillSignatureMap prefill_signature_map,
      std::optional<std::unique_ptr<EmbeddingLookupManager>>
          embedding_lookup_manager,
      std::optional<std::unique_ptr<EmbeddingLookupManager>>
          per_layer_embedding_lookup_manager,
      std::optional<EmbedderPerLayerContext> embedder_per_layer_context);

 private:
  // Prefill internal implementation, for one prefill call to the Interpreter
  // with a certain length.
  absl::Status PrefillInternal(absl::string_view prefill_signature,
                               absl::Span<const int> ids);

  // Decode internal implementation. Uses the specified 'token' as the input
  // token and uses the specified 'step' as the current time step.  The
  // logits from the decode step are stored in the 'logits' output buffer of
  // the transformer model when this function returns absl::OkStatus().
  absl::Status DecodeInternal(int step, std::shared_ptr<TokenData> token,
                              bool run_rope_and_mask = true);

  // Bulk prefill from raw embeddings directly into the Transformer graph.
  absl::Status PrefillInternalFromEmbeddings(
      absl::string_view prefill_signature, absl::Span<const float> embeddings,
      absl::Span<const float> ple_embeddings,
      absl::Span<const int32_t> seq_positions);

  // Creates the context for the embedder model.  Instead of creating new
  // output buffers for the embedder, the context will use the input buffers
  // of the provided 'gemma_prefill_input_buffers' and
  // 'gemma_decode_input_buffers'.  Similarly, instead of creating the buffers
  // for the input tokens the provided 'prefill_input_tokens' and
  // 'decode_input_tokens' will be duplicated and re-used as the input buffers.
  static absl::StatusOr<EmbedderContext> CreateEmbedderContextWithBufferSharing(
      ::litert::Environment& env, const litert::Model& embedder_model,
      const ::litert::TensorBuffer& prefill_input_tokens,
      const ::litert::TensorBuffer& decode_input_tokens,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          gemma_prefill_input_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          gemma_decode_input_buffers,
      const LlmExecutorSettings& settings);

  // Creates the context for the embedder per layer model.  Instead of creating
  // new output buffers for the embedder, the context will use the input buffers
  // of the provided 'gemma_prefill_input_buffers',
  // 'gemma_decode_input_buffers'.  Similarly, instead of creating the buffers
  // for the input tokens the provided 'prefill_input_tokens' and
  // 'decode_input_tokens' will be duplicated and re-used as the input buffers.
  static absl::StatusOr<
      LlmLiteRtNpuCompiledModelExecutor::EmbedderPerLayerContext>
  CreateEmbedderPerLayerContextWithBufferSharing(
      ::litert::Environment& env, const litert::Model& embedder_model,
      const ::litert::TensorBuffer& prefill_input_tokens,
      const ::litert::TensorBuffer& decode_input_tokens,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          gemma_prefill_input_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          gemma_decode_input_buffers,
      const LlmExecutorSettings& settings);

  // Creates the context for the NPU auxiliary model.
  static absl::StatusOr<NpuAuxiliaryContext> CreateNpuAuxiliaryContext(
      ::litert::Environment& env, const litert::Model& npu_auxiliary_model,
      const LlmExecutorSettings& settings);

  // Creates the context for the mask model.  Instead of creating new
  // output buffers for the mask model, the context will use the input buffers
  // of the provided 'gemma_prefill_input_buffers' and
  // 'gemma_decode_input_buffers'.
  static absl::StatusOr<InferenceContext> CreateMaskContextWithBufferSharing(
      NpuAuxiliaryContext& npu_auxiliary_context,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          gemma_prefill_input_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          gemma_decode_input_buffers);

  // Creates the context for the RoPE model.  Instead of creating new
  // output buffers for the RoPE model, the context will use the input buffers
  // of the provided 'gemma_prefill_input_buffers' and
  // 'gemma_decode_input_buffers'.
  static absl::StatusOr<InferenceContext> CreateRopeContextWithBufferSharing(
      NpuAuxiliaryContext& npu_auxiliary_context,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          gemma_prefill_input_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          gemma_decode_input_buffers);

  // Creates the context for the LLM model.
  static absl::StatusOr<InferenceContext>
  CreateLlmInferenceContextWithBufferSharing(
      ::litert::Environment& env, ::litert::CompiledModel& llm_compiled_model,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          input_kv_cache_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          prefill_output_kv_cache_slice_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          decode_output_kv_cache_slice_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          gemma_prefill_input_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          gemma_decode_input_buffers);

  static absl::StatusOr<InferenceContext>
  CreateCacheUpdateInferenceContextWithBufferSharing(
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          input_kv_cache_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          prefill_output_kv_cache_slice_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          decode_output_kv_cache_slice_buffers,
      ::litert::TensorBuffer prefill_input_pos,
      ::litert::TensorBuffer decode_input_pos);
  // Run a 'warmup' inference on every model (prefill and decode).  This is
  // intended to be called before the first actual inference.
  static absl::Status WarmupInference(
      ::litert::CompiledModel& compiled_model_llm,
      InferenceContext& llm_inference_context,
      ::litert::CompiledModel& compiled_model_auxiliary,
      const InferenceContext& rope_inference_context,
      const InferenceContext& mask_inference_context,
      const InferenceContext& cache_update_inference_context);

  // Clears all buffers in the provided 'buffers' map that belong to the KV
  // cache.
  static absl::Status ClearKVCache(
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>& buffers);

  bool UseEmbeddingLookupManager() const {
    return embedding_lookup_manager_.has_value();
  }

  // Allocates the transformer buffers. The buffers will be stored in the
  // provided output parameters.
  static absl::Status AllocateTransformerBuffers(
      litert::Environment& env, const litert::Model* transformer_model,
      CompiledModel& llm_compiled_model,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          gemma_prefill_input_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          gemma_decode_input_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          input_kv_cache_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          prefill_output_kv_cache_slice_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          decode_output_kv_cache_slice_buffers);

  // Create the executor for Gemma3n, with multi-modality support.
  static absl::StatusOr<std::unique_ptr<LlmLiteRtNpuCompiledModelExecutor>>
  CreateForModelHasPerLayerEmbedding(
      const LlmExecutorSettings& executor_settings, ModelResources& resources,
      litert::Environment& env, const litert::Model* transformer_model);

  // Create the executor for Gemma3.
  static absl::StatusOr<std::unique_ptr<LlmLiteRtNpuCompiledModelExecutor>>
  CreateForModelWithoutPerLayerEmbedding(
      const LlmExecutorSettings& executor_settings, ModelResources& resources,
      litert::Environment& env, const litert::Model* transformer_model);

  LlmExecutorSettings executor_settings_;
  ::litert::Environment& env_;
  std::unique_ptr<ModelResources> resources_;
  LatencyStats latency_stats_;
  EmbedderContext embedder_context_;
  NpuAuxiliaryContext npu_auxiliary_context_;
  InferenceContext mask_context_;
  InferenceContext rope_context_;
  ::litert::CompiledModel llm_compiled_model_;
  std::optional<std::unique_ptr<EmbeddingLookupManager>>
      embedding_lookup_manager_;
  std::optional<std::unique_ptr<EmbeddingLookupManager>>
      per_layer_embedding_lookup_manager_;
  std::optional<EmbedderPerLayerContext> embedder_per_layer_context_;
  InferenceContext llm_inference_context_;
  InferenceContext cache_update_inference_context_;
  SortedPrefillSignatureMap prefill_signature_map_;

  // The sampled ids to use for external sampling.
  // The layout is batch-major.
  // e.g. for output_batch_size=2, the layout is:
  // {batch_0_seq_0, batch_1_seq_0, batch_0_seq_1, batch_1_seq_1, ...}
  std::vector<int> sampled_ids_;

  // Internal timestep.
  int current_step_ = 0;

  bool is_bulk_embedding_ = false;

  // Logits quantization parameters.
  float logits_scale_ = 1.0f;
  int32_t logits_zero_point_ = 0;

  // The processed tokens.  This is also used to store the pending input token
  // for next prefill or decode steps.
  litert::lm::ProcessedTokens processed_tokens_;

  bool ShouldDump() const;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LITERT_NPU_COMPILED_MODEL_EXECUTOR_H_
