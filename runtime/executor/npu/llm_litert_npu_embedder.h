// Copyright 2026 Google LLC.
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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_NPU_LLM_LITERT_NPU_EMBEDDER_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_NPU_LLM_LITERT_NPU_EMBEDDER_H_

#include <cstddef>
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
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/embedding_lookup/embedding_lookup_manager.h"
#include "runtime/components/model_resources.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/executor/llm_executor_processed_tokens.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/executor/npu/llm_litert_npu_compiled_model_executor_utils.h"

namespace litert::lm {

// Signature names for the embedder.
struct EmbedderSignatures {
  static constexpr absl::string_view kDecodeEmbedder = "decode_embedder";
  static constexpr absl::string_view kVerifyEmbedder = "verify_embedder";
  // Prefill and decode use identical tensor signature names.
  static constexpr absl::string_view kEmbedderInput = "token_ids";
  static constexpr absl::string_view kEmbedderOutput = "embeddings";
};

struct EmbedderPerLayerSignatures {
  static constexpr absl::string_view kDecodeEmbedderPerLayer =
      "decode_per_layer_embedder";
  static constexpr absl::string_view kVerifyEmbedderPerLayer =
      "verify_per_layer_embedder";
  // Prefill and decode use identical tensor signature names.
  static constexpr absl::string_view kEmbedderInput = "token_ids";
  static constexpr absl::string_view kEmbedderOutput = "embeddings";
};

struct EmbedderContext {
  ::litert::Model embedder_model;
  ::litert::CompiledModel embedder_compiled_model;
  InferenceContext inference_context;
  EmbedderContext(::litert::CompiledModel embedder_compiled_model,
                  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
                      prefill_input_buffers,
                  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
                      prefill_output_buffers,
                  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
                      decode_input_buffers,
                  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
                      decode_output_buffers,
                  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
                      verify_input_buffers = {},
                  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
                      verify_output_buffers = {})
      : embedder_compiled_model(std::move(embedder_compiled_model)),
        inference_context(
            std::move(prefill_input_buffers), std::move(prefill_output_buffers),
            std::move(decode_input_buffers), std::move(decode_output_buffers),
            std::move(verify_input_buffers), std::move(verify_output_buffers)) {
  }
  EmbedderContext(EmbedderContext&&) = default;
  EmbedderContext& operator=(EmbedderContext&&) = default;
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
          decode_output_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
          verify_input_buffers = {},
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
          verify_output_buffers = {})
      : embedder_per_layer_compiled_model(
            std::move(embedder_per_layer_compiled_model)),
        inference_context(
            std::move(prefill_input_buffers), std::move(prefill_output_buffers),
            std::move(decode_input_buffers), std::move(decode_output_buffers),
            std::move(verify_input_buffers), std::move(verify_output_buffers)) {
  }
  EmbedderPerLayerContext(EmbedderPerLayerContext&&) = default;
  EmbedderPerLayerContext& operator=(EmbedderPerLayerContext&&) = default;
};

// Hardware quantization parameters for Per-Layer Embeddings.
struct HWQuantizationParams {
  const float* scales;
  bool is_per_channel;
};

// Holds hardware quantization parameters and lookup tables for Per-Layer
// Embeddings (PLE).
struct HWPleParams {
  bool use_hw_ple = false;
  std::vector<const uint8_t*> ple_table_ptrs;
  std::vector<HWQuantizationParams> ple_quant_params;
  std::vector<float> ple_per_tensor_scales;
  int num_tables = 0;
  int ple_embedding_dim = 0;
  litert::ElementType output_type = litert::ElementType::None;
  litert::ElementType ple_table_element_type = litert::ElementType::None;
  float mul_scale = 1.0f;
  float output_scale = 1.0f;
  int32_t final_zero_point = 0;
};

// =============================================================================
// NpuEmbedder Usage Guide:
//
// 1. Multimodal Lifecycle (Optional/No-op for text-only):
//    embedder_.UpdateMultiModalEmbeddings(inputs);
//    ... prefill ...
//    embedder_.CleanupMultiModalEmbeddings();
//
// 2. Regular Prefill:
//    embedder_.RunPrefill(signature, pending_token, processed_tokens,
//    last_input_token); if (embedder_.HasPerLayerEmbeddings()) {
//      embedder_.RunPrefillPerLayer(ple_signature, tokens_to_embed);
//    }
//    * Note on Prefill: All 3 embedding steps (copy previous pending token to
//      slot 0, lookup processed input tokens, and lookup/cache the new holdback
//      token) are encapsulated in a single `RunPrefill` call because all tokens
//      in the prefill chunk are known together at the time of execution.
//
// 3. Regular Single-Token Decode:
//    // Step A (End of Step T): Look up embedding for newly sampled token and
//    // attach it to TokenData before enqueuing into the pending token queue:
//    embedder_.LookupDecode(token);
//
//    // Step B (Start of Step T+1): When popped from queue, copy embedding to
//    // decode buffer (or run compiled embedder model) before text decoder
//    runs: embedder_.RunDecode(*token); if (embedder_.HasPerLayerEmbeddings())
//    {
//      embedder_.RunDecodePerLayer(token->id());
//    }
//    * Note on Decode: Decode separates `LookupDecode` and `RunDecode` across
//      step boundaries because token generation is autoregressive: the newly
//      sampled token's embedding is looked up immediately upon generation at
//      the end of step T, whereas hardware buffer loading (`RunDecode`) occurs
//      at the start of step T+1 when preparing decoder inputs.
//
// 4. MTP Speculative Decoding - Draft Token Embedding:
//    embedder_.LookupDecode(draft_token_id, draft_embedding);
//
// 5. MTP Speculative Decoding - Verification:
//    embedder_.RunVerify(verify_ids);
//    if (embedder_.HasPerLayerEmbeddings()) {
//      embedder_.RunVerifyPerLayer(verify_ids);
//    }
//
// 6. Dynamic Context Switching:
//    When the text decoder switches from one context length to the next,
//    its input buffer bindings change. Rebind Embedder output buffers:
//    embedder_.UpdateOutputBuffers(
//        active_group.text_decoder_inference_context.prefill_input_buffers,
//        active_group.text_decoder_inference_context.decode_input_buffers,
//        active_group.text_decoder_inference_context.verify_input_buffers);
// =============================================================================
class NpuEmbedder {
 public:
  NpuEmbedder() = default;
  NpuEmbedder(const NpuEmbedder&) = delete;
  NpuEmbedder& operator=(const NpuEmbedder&) = delete;
  NpuEmbedder(NpuEmbedder&&) = default;
  NpuEmbedder& operator=(NpuEmbedder&&) = default;

  // --- Lifecycle & Creation ---
  static absl::StatusOr<NpuEmbedder> Create(
      ::litert::Environment& env, ModelResources& resources,
      const LlmExecutorSettings& executor_settings,
      const ResolvedPrefillSignatures& prefill_signatures,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          text_decoder_prefill_input_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          text_decoder_decode_input_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          text_decoder_verify_input_buffers,
      bool has_per_layer_embeddings);

  absl::Status UpdateMultiModalEmbeddings(const ExecutorInputs& inputs);
  absl::Status CleanupMultiModalEmbeddings();
  std::vector<float> GetDefaultEmbeddingVector() const;

  // --- Stage 1: Prefill ---
  absl::Status RunPrefill(absl::string_view embedder_signature = "",
                          const TokenData* pending_token = nullptr,
                          absl::Span<const int> processed_input_tokens = {},
                          TokenData* last_input_token = nullptr);
  absl::Status RunPrefillPerLayer(absl::string_view signature = "",
                                  absl::Span<const int> tokens_to_embed = {});
  absl::Status WriteAndPadPleEmbeddings(::litert::Environment& env,
                                        absl::Span<const float> ple_embeddings);

  // --- Stage 2: Decode ---
  absl::Status LookupDecode(TokenData* token);
  absl::Status LookupDecode(int32_t token_id,
                            std::vector<float>& out_embedding);
  absl::Status RunDecode(const TokenData& token);
  absl::Status RunDecodePerLayer(int32_t token_id);
  absl::Status WriteDecodePleEmbeddings(absl::Span<const float> ple_embeddings);

  // --- Stage 3: Speculative Decoding (Verification) ---
  absl::Status RunVerify(absl::Span<const int> verify_ids);
  absl::Status RunVerifyPerLayer(absl::Span<const int> verify_ids);

  // --- Query & Introspection ---
  bool HasPerLayerEmbeddings() const {
    return (ple_params_.use_hw_ple && !ple_params_.ple_table_ptrs.empty()) ||
           embedder_per_layer_context_.has_value();
  }
  bool UseHwPle() const {
    return ple_params_.use_hw_ple && !ple_params_.ple_table_ptrs.empty();
  }
  const HWPleParams& PleParams() const { return ple_params_; }

  // --- Warmup & Context Accessors ---
  absl::Status RunPrefillEmbedder(absl::string_view signature);
  absl::Status RunDecodeEmbedder();

  // Updates the internal Embedder output buffer bindings to point to the
  // newly active context group's text decoder input buffers when switching
  // from one context length to the next.
  absl::Status UpdateOutputBuffers(
      const absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          text_decoder_prefill_input_buffers,
      const absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          text_decoder_decode_input_buffers,
      const absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          text_decoder_verify_input_buffers);

  EmbedderContext* MutableEmbedderContext() {
    return embedder_context_.has_value() ? &*embedder_context_ : nullptr;
  }
  EmbedderPerLayerContext* MutableEmbedderPerLayerContext() {
    return embedder_per_layer_context_.has_value()
               ? &*embedder_per_layer_context_
               : nullptr;
  }
  const EmbedderPerLayerContext* GetEmbedderPerLayerContext() const {
    return embedder_per_layer_context_.has_value()
               ? &*embedder_per_layer_context_
               : nullptr;
  }

 private:
  absl::Status LookupHwPle(const int* token_ids, int num_tokens,
                           void* output_buffer) const;

  static absl::StatusOr<NpuEmbedder> Create(
      std::unique_ptr<EmbeddingLookupManager> embedding_lookup_manager,
      std::optional<EmbedderContext> embedder_context,
      std::optional<EmbedderPerLayerContext> embedder_per_layer_context,
      std::unique_ptr<EmbeddingLookupManager>
          per_layer_embedding_lookup_manager,
      const litert::Model* embedder_per_layer_model, HWPleParams ple_params,
      std::optional<::litert::TensorBuffer> prefill_embeddings_buffer =
          std::nullopt,
      std::optional<::litert::TensorBuffer> decode_embeddings_buffer =
          std::nullopt,
      std::optional<::litert::TensorBuffer> verify_embeddings_buffer =
          std::nullopt,
      std::optional<::litert::TensorBuffer> prefill_ple_buffer = std::nullopt,
      std::optional<::litert::TensorBuffer> decode_ple_buffer = std::nullopt,
      std::optional<::litert::TensorBuffer> verify_ple_buffer = std::nullopt);

  absl::Status RunVerifyEmbedder(absl::Span<const int> verify_ids);
  absl::Status WriteAndPadPleEmbeddings(::litert::Environment& env,
                                        ::litert::TensorBuffer& buffer,
                                        absl::Span<const float> ple_embeddings);

  static absl::StatusOr<EmbedderContext> CreateEmbedderContext(
      ::litert::Environment& env, const litert::Model& embedder_model,
      const ResolvedPrefillSignatures& prefill_signatures,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          text_decoder_prefill_input_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          text_decoder_decode_input_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          text_decoder_verify_input_buffers,
      const LlmExecutorSettings& settings);

  static absl::StatusOr<EmbedderPerLayerContext> CreateEmbedderPerLayerContext(
      ::litert::Environment& env, const litert::Model& embedder_model,
      const ResolvedPrefillSignatures& prefill_signatures,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          text_decoder_prefill_input_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          text_decoder_decode_input_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          text_decoder_verify_input_buffers,
      const LlmExecutorSettings& settings);

  bool HasEmbeddingLookupManager() const {
    return embedding_lookup_manager_ != nullptr;
  }

  absl::Status CopyEmbeddingToBuffer(
      absl::Span<const float> embedding,
      ::litert::TensorBuffer& destination_buffer);

  std::unique_ptr<EmbeddingLookupManager> embedding_lookup_manager_;
  std::optional<EmbedderContext> embedder_context_;
  std::optional<EmbedderPerLayerContext> embedder_per_layer_context_;
  std::unique_ptr<EmbeddingLookupManager> per_layer_embedding_lookup_manager_;
  const litert::Model* embedder_per_layer_model_ = nullptr;
  HWPleParams ple_params_;
  std::optional<::litert::TensorBuffer> prefill_embeddings_buffer_;
  std::optional<::litert::TensorBuffer> decode_embeddings_buffer_;
  std::optional<::litert::TensorBuffer> verify_embeddings_buffer_;
  std::optional<::litert::TensorBuffer> prefill_ple_buffer_;
  std::optional<::litert::TensorBuffer> decode_ple_buffer_;
  std::optional<::litert::TensorBuffer> verify_ple_buffer_;
};

// Performs manual per-layer embedding lookup.
absl::Status HWPerLayerEmbeddingLookup(
    const int* token_ids, int num_tokens, const uint8_t* const* table_ptrs,
    const HWQuantizationParams* quant_params, int num_tables,
    int ple_embedding_dim, void* output_buffer, litert::ElementType output_type,
    litert::ElementType ple_table_element_type, float mul_scale = 1.0f,
    float output_scale = 1.0f, int32_t final_zero_point = 0);

// Writes PLE embeddings to the buffer, quantizing them to Int16 if needed.
absl::Status WritePleEmbeddings(::litert::TensorBuffer& buffer,
                                absl::Span<const float> ple_embeddings,
                                litert::ElementType output_type,
                                float final_scale, int32_t final_zero_point);

// Writes PLE embeddings to the buffer, quantizing them to Int16 if needed,
// and pads the remaining buffer space with a default embedding.
absl::Status WriteAndPadPleEmbeddings(::litert::TensorBuffer& buffer,
                                      absl::Span<const float> ple_embeddings,
                                      size_t ple_dim, size_t seq_pos_size,
                                      const std::vector<float>& default_ple_emb,
                                      litert::ElementType output_type,
                                      float final_scale,
                                      int32_t final_zero_point);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_NPU_LLM_LITERT_NPU_EMBEDDER_H_
