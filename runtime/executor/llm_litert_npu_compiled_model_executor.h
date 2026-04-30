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

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <ostream>
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
#include "litert/cc/litert_expected.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_options.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/embedding_lookup/embedding_lookup_manager.h"
#include "runtime/components/model_resources.h"
#include "runtime/executor/litert_compiled_model_executor_utils.h"
#include "runtime/executor/llm_executor.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/executor/llm_executor_processed_tokens.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/executor/llm_litert_npu_compiled_model_executor_utils.h"

namespace litert::lm {

// Component intended to be used with an NPU variant of Gemma3.
class LlmLiteRtNpuCompiledModelExecutor : public LlmExecutor {
 public:
  struct LogitsQuantizationParams {
    float scale = 1.0f;
    int32_t zero_point = 0;
  };

  // Holds the latency breakdown stats for the executor.
  // TODO(b/405424188): Use 'litert::lm::BenchmarkInfo' instead.
  struct LatencyStats {
    // Prefill latency stats.
    uint64_t prefill_e2e_latency_us = 0;
    int prefill_num_tokens = 0;
    uint64_t prefill_prepare_input_latency_us = 0;
    uint64_t prefill_embedder_inference_latency_us = 0;
    uint64_t prefill_embedder_per_layer_inference_latency_us = 0;
    uint64_t prefill_mask_inference_latency_us = 0;
    uint64_t prefill_rope_inference_latency_us = 0;
    uint64_t prefill_llm_inference_latency_us = 0;
    uint64_t prefill_cache_update_inference_latency_us = 0;

    // Decode latency stats.
    uint64_t decode_e2e_latency_us = 0;
    int decode_num_tokens = 0;
    uint64_t decode_prepare_input_latency_us = 0;
    uint64_t decode_embedder_inference_latency_us = 0;
    uint64_t decode_embedder_per_layer_inference_latency_us = 0;
    uint64_t decode_mask_inference_latency_us = 0;
    uint64_t decode_rope_inference_latency_us = 0;
    uint64_t decode_llm_inference_latency_us = 0;
    uint64_t decode_drafter_inference_latency_us = 0;
    uint64_t decode_cache_update_inference_latency_us = 0;
    uint64_t decode_sampling_latency_us = 0;
    uint64_t decode_mtp_rejection_sampling_latency_us = 0;
    uint64_t decode_mtp_activation_copy_latency_us = 0;
    uint64_t decode_token_queue_latency_us = 0;

    // MTP / Speculative Decoding latency stats.
    int mtp_num_draft_tokens = 0;
    int mtp_num_accepted_tokens = 0;
  };

  // Creates an executor from the resources.
  static absl::StatusOr<std::unique_ptr<LlmLiteRtNpuCompiledModelExecutor>>
  Create(const LlmExecutorSettings& executor_settings,
         ModelResources& resources, Environment& env);

  ~LlmLiteRtNpuCompiledModelExecutor() override;

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
      const ExecutorDecodeParams& decode_params) override;

  absl::string_view ExecutorBackendName() const override {
    return "LiteRT NPU Compiled Model";
  }

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
  const LatencyStats& GetLatencyStats() const;

  // Resets all of the internal states.
  absl::Status Reset() override;

 private:
  static litert::Expected<litert::Options> CreateLiteRtNpuOptions(
      const LlmExecutorSettings& settings);

  static litert::Expected<litert::Options> CreateLiteRtCpuOptions(
      const LlmExecutorSettings& settings);

  // Holds the tensor buffer maps for the inference of a precompiled model,
  // both for prefill and decode.

  enum class SpeculativeDecodingType {
    kNone,
    kMTP,
  };

  enum class KVCacheUpdateMethod {
    kModel,
    kWH,
  };

  enum class MaskUpdateMethod {
    kModel,
    kWH,
  };

  struct InferenceContext {
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
        prefill_input_buffers;
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
        prefill_output_buffers;
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
        decode_input_buffers;
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
        decode_output_buffers;
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
        verify_input_buffers;
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
        verify_output_buffers;
    InferenceContext(
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
            verify_output_buffers = {});
  };

  // Holds the context for the drafter model.
  struct DrafterContext {
    ::litert::CompiledModel mtp_compiled_model;
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
        mtp_input_buffers;
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
        mtp_output_buffers;
    DrafterContext(
        ::litert::CompiledModel mtp_compiled_model,
        absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
            mtp_input_buffers,
        absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
            mtp_output_buffers)
        : mtp_compiled_model(std::move(mtp_compiled_model)),
          mtp_input_buffers(std::move(mtp_input_buffers)),
          mtp_output_buffers(std::move(mtp_output_buffers)) {}
  };

  // Holds the context for the drafter auxiliary model.
  struct DrafterAuxContext {
    ::litert::CompiledModel mtp_aux_compiled_model;
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
        rope_input_buffers;
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
        rope_output_buffers;
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
        mask_input_buffers;
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
        mask_output_buffers;
    DrafterAuxContext(
        ::litert::CompiledModel mtp_aux_compiled_model,
        absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
            rope_input_buffers,
        absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
            rope_output_buffers,
        absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
            mask_input_buffers,
        absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
            mask_output_buffers)
        : mtp_aux_compiled_model(std::move(mtp_aux_compiled_model)),
          rope_input_buffers(std::move(rope_input_buffers)),
          rope_output_buffers(std::move(rope_output_buffers)),
          mask_input_buffers(std::move(mask_input_buffers)),
          mask_output_buffers(std::move(mask_output_buffers)) {}
  };

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
            decode_output_buffers,
        absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
            verify_input_buffers = {},
        absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
            verify_output_buffers = {});
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
          inference_context(std::move(prefill_input_buffers),
                            std::move(prefill_output_buffers),
                            std::move(decode_input_buffers),
                            std::move(decode_output_buffers),
                            std::move(verify_input_buffers),
                            std::move(verify_output_buffers)) {}
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
      std::unique_ptr<EmbeddingLookupManager> embedding_lookup_manager,
      std::optional<EmbedderPerLayerContext> embedder_per_layer_context,
      LogitsQuantizationParams quantization_params,
      std::vector<const uint8_t*> ple_table_ptrs = {},
      std::vector<HWQuantizationParams> ple_quant_params = {},
      std::vector<float> ple_per_tensor_scales = {}, int num_tables = 0,
      litert::ElementType output_type = litert::ElementType::None,
      float final_scale = 1.0f, int32_t final_zero_point = 0,
      SpeculativeDecodingType speculative_decoding_type =
          SpeculativeDecodingType::kNone,
      std::optional<DrafterContext> drafter_context = std::nullopt,
      std::optional<DrafterAuxContext> drafter_aux_context = std::nullopt)
      : executor_settings_(std::move(executor_settings)),
        env_(llm_env),
        embedder_context_(std::move(embedder_context)),
        npu_auxiliary_context_(std::move(npu_auxiliary_context)),
        mask_context_(std::move(mask_context)),
        rope_context_(std::move(rope_context)),
        llm_compiled_model_(std::move(llm_compiled_model)),
        embedding_lookup_manager_(std::move(embedding_lookup_manager)),
        embedder_per_layer_context_(std::move(embedder_per_layer_context)),
        llm_inference_context_(std::move(llm_inference_context)),
        cache_update_inference_context_(
            std::move(cache_update_inference_context)),
        prefill_signature_map_(std::move(prefill_signature_map)),
        ple_table_ptrs_(std::move(ple_table_ptrs)),
        ple_quant_params_(std::move(ple_quant_params)),
        ple_per_tensor_scales_(std::move(ple_per_tensor_scales)),
        num_tables_(num_tables),
        output_type_(output_type),
        final_scale_(final_scale),
        final_zero_point_(final_zero_point),
        speculative_decoding_type_(speculative_decoding_type),
        drafter_context_(std::move(drafter_context)),
        drafter_aux_context_(std::move(drafter_aux_context)),
        per_tensor_logits_scale_(quantization_params.scale),
        per_tensor_logits_zero_point_(quantization_params.zero_point) {
    auto npu_config_status = executor_settings_.GetBackendConfig<NpuConfig>();
    if (npu_config_status.ok()) {
      npu_config_ = *npu_config_status;
      if (npu_config_.use_hw_masking_for_npu) {
        prefill_mask_update_method_ = MaskUpdateMethod::kWH;
        decode_mask_update_method_ = MaskUpdateMethod::kWH;
        mtp_mask_update_method_ = MaskUpdateMethod::kWH;
        verify_mask_update_method_ = MaskUpdateMethod::kWH;
      }
      if (npu_config_.use_hw_cache_update_for_npu) {
        prefill_kv_cache_update_method_ = KVCacheUpdateMethod::kWH;
        decode_kv_cache_update_method_ = KVCacheUpdateMethod::kWH;
      }
      if (npu_config_.use_hw_ple_for_npu) {
        use_hw_ple_for_npu_ = true;
      }
    }
    if (embedder_per_layer_context_.has_value()) {
      latency_stats_.prefill_embedder_per_layer_inference_latency_us = 0;
      latency_stats_.decode_embedder_per_layer_inference_latency_us = 0;
    }
  }

 private:
  // Prefill internal implementation, for one prefill call to the Interpreter
  // with a certain length.
  absl::Status PrefillInternal(absl::string_view prefill_signature,
                               absl::Span<const int> ids);

  // Decode internal implementation. Uses the specified 'token' as the input
  // token and uses the specified 'step' as the current time step.  The
  // logits from the decode step are stored in the 'logits' output buffer of
  // the transformer model when this function returns absl::OkStatus().
  //
  // Arguments:
  // - step: The current time step.
  // - token: The input token to decode.
  absl::Status DecodeInternal(int step, std::shared_ptr<TokenData> token);

  // Run the drafter loop for MTP.
  // Persistent buffer to store the sliced activations from the last verifier
  // run, to be used as input for the next speculative cycle.
  std::vector<uint8_t> last_verify_activations_;
  bool has_valid_verify_activations_ = false;

  // Queue of token IDs that have been accepted by the verifier but not yet
  // returned to the engine.
  std::vector<int> pending_accepted_tokens_;

  // Run the drafter loop for MTP, returning the draft tokens.
  absl::StatusOr<std::vector<int>> RunDrafterLoop(int start_step,
                                                  int current_token_id);
  // Run the verifier batch for MTP.
  absl::Status RunVerifierBatch(int start_step, int current_token_id,
                                const std::vector<int>& draft_tokens);

  // Rejection sampling for MTP. Returns (num_accepted, bonus_token_id).
  struct RejectionSamplingResult {
    int num_accepted;
    int bonus_token_id;
  };
  absl::StatusOr<RejectionSamplingResult> PerformRejectionSampling(
      const std::vector<int>& draft_tokens,
      const ::litert::TensorBuffer& verifier_logits_buffer);

  // Commit the verified KV cache for MTP.
  absl::Status CommitVerifiedKVCache(int start_step);

  // Creates the context for the embedder model.  Instead of creating new
  // output buffers for the embedder, the context will use the input buffers
  // of the provided 'gemma_prefill_input_buffers' and
  // 'gemma_decode_input_buffers'.  Similarly, instead of creating the buffers
  // for the input tokens the provided 'prefill_input_tokens' and
  // 'decode_input_tokens' will be duplicated and re-used as the input buffers.
  //
  // Arguments:
  // - env: The LiteRT environment.
  // - embedder_model: The embedder model.
  // - prefill_input_tokens: The tensor buffer for prefill input tokens.
  // - decode_input_tokens: The tensor buffer for decode input tokens.
  // - gemma_prefill_input_buffers: A map of input buffers for the Gemma prefill
  //   model, keyed by input name. The output buffers of the embedder model
  //   will be shared with these input buffers.
  // - gemma_decode_input_buffers: A map of input buffers for the Gemma decode
  //   model, keyed by input name. The output buffers of the embedder model
  //   will be shared with these input buffers.
  // - settings: The executor settings.
  static absl::StatusOr<EmbedderContext> CreateEmbedderContextWithBufferSharing(
      ::litert::Environment& env, const litert::Model& embedder_model,
      const ::litert::TensorBuffer& prefill_input_tokens,
      const ::litert::TensorBuffer& decode_input_tokens,
      const ::litert::TensorBuffer& verify_input_tokens,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          gemma_prefill_input_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          gemma_decode_input_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          gemma_verify_input_buffers,
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
      const ::litert::TensorBuffer& verify_input_tokens,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          gemma_prefill_input_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          gemma_decode_input_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          gemma_verify_input_buffers,
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
      const NpuAuxiliaryContext& npu_auxiliary_context,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          gemma_prefill_input_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          gemma_decode_input_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          gemma_verify_input_buffers);

  static absl::StatusOr<InferenceContext> CreateRopeContextWithBufferSharing(
      const NpuAuxiliaryContext& npu_auxiliary_context,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          gemma_prefill_input_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          gemma_decode_input_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          gemma_verify_input_buffers);

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
          verify_output_kv_cache_slice_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          gemma_prefill_input_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          gemma_decode_input_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          gemma_verify_input_buffers);

  static absl::StatusOr<InferenceContext>
  CreateCacheUpdateInferenceContextWithBufferSharing(
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          input_kv_cache_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          prefill_output_kv_cache_slice_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          decode_output_kv_cache_slice_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          verify_output_kv_cache_slice_buffers,
      ::litert::TensorBuffer prefill_input_pos,
      ::litert::TensorBuffer decode_input_pos,
      ::litert::TensorBuffer verify_input_pos);
  // Run a 'warmup' inference on every model (prefill and decode).  This is
  // intended to be called before the first actual inference.
  static absl::Status WarmupInference(
      ::litert::CompiledModel& compiled_model_llm,
      InferenceContext& llm_inference_context,
      ::litert::CompiledModel& compiled_model_auxiliary,
      const InferenceContext& rope_inference_context,
      const InferenceContext& mask_inference_context,
      const InferenceContext& cache_update_inference_context);

  // Run a 'warmup' inference on the drafter model.  This is intended to be
  // called before the first actual inference.
  static absl::Status WarmupDrafterInference(
      const DrafterContext& drafter_context,
      const DrafterAuxContext& drafter_aux_context);

  // Clears all buffers in the provided 'buffers' map that belong to the KV
  // cache.
  static absl::Status ClearKVCache(
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>& buffers);

  bool UseEmbeddingLookupManager() const {
    return embedding_lookup_manager_ != nullptr;
  }

  // Allocates the MTP drafter buffers. The buffers will be stored in the
  // provided output parameters.
  static absl::StatusOr<DrafterContext>
  CreateDrafterInferenceContextWithBufferSharing(
      ::litert::Environment& env, const litert::Model& mtp_drafter_model,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          drafter_input_kv_cache_buffers,
      ::litert::TensorBuffer& output_activations_buffers);

  static absl::StatusOr<DrafterAuxContext>
  CreateDrafterInferenceAuxContextWithBufferSharing(
      ::litert::Environment& env, const litert::Model& mtp_aux_model,
      const absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          drafter_aux_output_buffers);

  static absl::Status AllocateTransformerBuffers(
      litert::Environment& env, const litert::Model* transformer_model,
      CompiledModel& llm_compiled_model,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          gemma_prefill_input_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          gemma_decode_input_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          gemma_verify_input_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          input_kv_cache_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          prefill_output_kv_cache_slice_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          decode_output_kv_cache_slice_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          verify_output_kv_cache_slice_buffers);

  // Create the executor for Gemma3n, with multi-modality support.
  static absl::StatusOr<std::unique_ptr<LlmLiteRtNpuCompiledModelExecutor>>
  CreateForModelHasPerLayerEmbedding(
      const LlmExecutorSettings& executor_settings, ModelResources& resources,
      litert::Environment& env, const litert::Model* transformer_model,
      LogitsQuantizationParams quantization_params);

  // Create the executor for Gemma3.
  static absl::StatusOr<std::unique_ptr<LlmLiteRtNpuCompiledModelExecutor>>
  CreateForModelWithoutPerLayerEmbedding(
      const LlmExecutorSettings& executor_settings, ModelResources& resources,
      litert::Environment& env, const litert::Model* transformer_model,
      LogitsQuantizationParams quantization_params);

  LlmExecutorSettings executor_settings_;
  NpuConfig npu_config_;
  KVCacheUpdateMethod prefill_kv_cache_update_method_ =
      KVCacheUpdateMethod::kModel;
  KVCacheUpdateMethod decode_kv_cache_update_method_ =
      KVCacheUpdateMethod::kModel;
  MaskUpdateMethod prefill_mask_update_method_ = MaskUpdateMethod::kModel;
  MaskUpdateMethod decode_mask_update_method_ = MaskUpdateMethod::kModel;
  MaskUpdateMethod mtp_mask_update_method_ = MaskUpdateMethod::kModel;
  MaskUpdateMethod verify_mask_update_method_ = MaskUpdateMethod::kModel;

  ::litert::Environment& env_;
  std::unique_ptr<ModelResources> resources_;
  LatencyStats latency_stats_;
  EmbedderContext embedder_context_;
  NpuAuxiliaryContext npu_auxiliary_context_;
  InferenceContext mask_context_;
  InferenceContext rope_context_;
  ::litert::CompiledModel llm_compiled_model_;
  std::unique_ptr<EmbeddingLookupManager> embedding_lookup_manager_;
  std::optional<EmbedderPerLayerContext> embedder_per_layer_context_;
  InferenceContext llm_inference_context_;
  InferenceContext cache_update_inference_context_;
  SortedPrefillSignatureMap prefill_signature_map_;

  bool use_hw_ple_for_npu_ = false;
  std::vector<const uint8_t*> ple_table_ptrs_;
  std::vector<HWQuantizationParams> ple_quant_params_;
  std::vector<float> ple_per_tensor_scales_;
  int num_tables_ = 0;
  litert::ElementType output_type_ = litert::ElementType::None;
  float final_scale_ = 1.0f;
  int32_t final_zero_point_ = 0;

  // MTP / Speculative Decoding members.
  SpeculativeDecodingType speculative_decoding_type_ =
      SpeculativeDecodingType::kNone;
  std::optional<DrafterContext> drafter_context_;
  std::optional<DrafterAuxContext> drafter_aux_context_;

  // The sampled ids to use for external sampling.
  // The layout is batch-major.
  // e.g. for output_batch_size=2, the layout is:
  // {batch_0_seq_0, batch_1_seq_0, batch_0_seq_1, batch_1_seq_1, ...}
  std::vector<int> sampled_ids_;

  // Internal timestep.
  int current_step_ = 0;

  // Per-tensor quantization parameters for the logits tensor of the 'decode'
  // signature.
  float per_tensor_logits_scale_;
  int32_t per_tensor_logits_zero_point_;

  // The processed tokens.  This is also used to store the pending input token
  // for next prefill or decode steps.
  litert::lm::ProcessedTokens processed_tokens_;
};

std::ostream& operator<<(
    std::ostream& os,
    const LlmLiteRtNpuCompiledModelExecutor::LatencyStats& stats);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LITERT_NPU_COMPILED_MODEL_EXECUTOR_H_
