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
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_options.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/constrained_decoding/constrained_decoder.h"
#include "runtime/components/model_resources.h"
#include "runtime/executor/litert_compiled_model_executor_utils.h"
#include "runtime/executor/llm_executor.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/executor/llm_executor_processed_tokens.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/executor/npu/llm_litert_npu_compiled_model_executor_utils.h"
#include "runtime/executor/npu/llm_litert_npu_embedder.h"
#include "runtime/executor/npu/llm_litert_npu_kv_cache.h"
#include "runtime/executor/npu/llm_litert_npu_mask.h"
#include "runtime/executor/npu/llm_litert_npu_rope.h"

namespace litert::lm {

// Holds the context for the drafter auxiliary model (MTP RoPE & Mask).
struct DrafterAuxContext {
  ::litert::CompiledModel mtp_aux_compiled_model;
  NpuRope drafter_rope;
  NpuMask drafter_mask;

  DrafterAuxContext(::litert::CompiledModel mtp_aux_compiled_model,
                    NpuRope drafter_rope, NpuMask drafter_mask)
      : mtp_aux_compiled_model(std::move(mtp_aux_compiled_model)),
        drafter_rope(std::move(drafter_rope)),
        drafter_mask(std::move(drafter_mask)) {
    this->drafter_rope.SetCompiledModel(&this->mtp_aux_compiled_model);
    this->drafter_mask.SetCompiledModel(&this->mtp_aux_compiled_model);
  }
  DrafterAuxContext(DrafterAuxContext&& other) noexcept
      : mtp_aux_compiled_model(std::move(other.mtp_aux_compiled_model)),
        drafter_rope(std::move(other.drafter_rope)),
        drafter_mask(std::move(other.drafter_mask)) {
    drafter_rope.SetCompiledModel(&mtp_aux_compiled_model);
    drafter_mask.SetCompiledModel(&mtp_aux_compiled_model);
  }
  DrafterAuxContext& operator=(DrafterAuxContext&& other) noexcept {
    if (this != &other) {
      mtp_aux_compiled_model = std::move(other.mtp_aux_compiled_model);
      drafter_rope = std::move(other.drafter_rope);
      drafter_mask = std::move(other.drafter_mask);
      drafter_rope.SetCompiledModel(&mtp_aux_compiled_model);
      drafter_mask.SetCompiledModel(&mtp_aux_compiled_model);
    }
    return *this;
  }

  // Compiles the MTP auxiliary model and initializes drafter_rope and
  // drafter_mask.
  static absl::StatusOr<DrafterAuxContext> Create(
      ::litert::Environment& env, const litert::Model& mtp_aux_model,
      const absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          drafter_aux_output_buffers,
      MaskUpdateMethod mtp_mask_update_method);
};

// Component intended to be used with an NPU variant of Gemma3.
class LlmLiteRtNpuCompiledModelExecutor : public LlmExecutor {
 public:
  struct LogitsQuantizationParams {
    float scale = 1.0f;
    int32_t zero_point = 0;
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

  absl::StatusOr<::litert::TensorBuffer> DecodeLogits(
      const ExecutorInputs& inputs) override;

  absl::StatusOr<::litert::TensorBuffer> DecodeLogits(
      const ExecutorInputs& inputs, const ExecutorDecodeParams& decode_params);

  absl::string_view ExecutorBackendName() const override {
    return "LiteRT NPU Compiled Model";
  }

  // Set the current step of the executor.
  absl::Status SetCurrentStep(int new_step) override;

  absl::StatusOr<const ProcessedTokens*> GetProcessedTokens() const override;

  // Updates the runtime configuration.
  absl::Status UpdateRuntimeConfig(
      const RuntimeConfig& runtime_config) override {
    return absl::OkStatus();
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

  absl::StatusOr<std::unique_ptr<LlmContext>> CreateNewContext(
      std::optional<uint32_t> lora_id,
      RuntimeConfig runtime_config) const override;

  absl::StatusOr<std::unique_ptr<LlmContext>> CloneContext() const override;

  absl::Status RestoreContext(
      std::unique_ptr<LlmContext> context_data) override;

 private:
  enum class SpeculativeDecodingType {
    kNone,
    kMTP,
  };

 protected:
  LlmLiteRtNpuCompiledModelExecutor(
      LlmExecutorSettings executor_settings, Environment& llm_env,
      NpuAuxiliaryContext npu_auxiliary_context,
      ::litert::CompiledModel text_decoder_compiled_model,
      std::vector<ContextGroup> context_groups,
      std::vector<int> sorted_supported_context_sizes,
      SortedPrefillSignatureMap prefill_signature_map,
      ResolvedPrefillSignatures prefill_signatures,
      LogitsQuantizationParams quantization_params, int64_t kv_cache_init_value,
      SpeculativeDecodingType speculative_decoding_type,
      std::optional<DrafterContext> drafter_context,
      std::optional<DrafterAuxContext> drafter_aux_context,
      NpuEmbedder main_embedder, NpuRope main_rope, NpuMask main_mask,
      NpuKVCache main_cache)
      : executor_settings_(std::move(executor_settings)),
        env_(llm_env),
        main_embedder_(std::move(main_embedder)),
        main_rope_(std::move(main_rope)),
        main_mask_(std::move(main_mask)),
        main_cache_(std::move(main_cache)),
        npu_auxiliary_context_(std::move(npu_auxiliary_context)),
        text_decoder_compiled_model_(std::move(text_decoder_compiled_model)),
        context_groups_(std::move(context_groups)),
        sorted_supported_context_sizes_(
            std::move(sorted_supported_context_sizes)),
        prefill_signature_map_(std::move(prefill_signature_map)),
        prefill_signatures_(std::move(prefill_signatures)),
        speculative_decoding_type_(speculative_decoding_type),
        drafter_context_(std::move(drafter_context)),
        drafter_aux_context_(std::move(drafter_aux_context)),
        per_tensor_logits_scale_(quantization_params.scale),
        per_tensor_logits_zero_point_(quantization_params.zero_point),
        kv_cache_init_value_(kv_cache_init_value) {
    auto npu_config_status = executor_settings_.GetBackendConfig<NpuConfig>();
    if (npu_config_status.ok()) {
      npu_config_ = *npu_config_status;
    }
    if (main_embedder_.HasPerLayerEmbeddings()) {
      latency_stats_.prefill_embedder_per_layer_inference_latency_us = 0;
      latency_stats_.decode_embedder_per_layer_inference_latency_us = 0;
    }
    main_rope_.SetCompiledModel(
        &npu_auxiliary_context_.npu_auxiliary_compiled_model);
    main_mask_.SetCompiledModel(
        &npu_auxiliary_context_.npu_auxiliary_compiled_model);
    main_cache_.SetCompiledModel(
        &npu_auxiliary_context_.npu_auxiliary_compiled_model);
    if (drafter_aux_context_.has_value()) {
      drafter_aux_context_->drafter_rope.SetCompiledModel(
          &drafter_aux_context_->mtp_aux_compiled_model);
      drafter_aux_context_->drafter_mask.SetCompiledModel(
          &drafter_aux_context_->mtp_aux_compiled_model);
    }
  }

 private:
  const ContextGroup& ActiveContextGroup() const {
    return context_groups_[active_context_group_index_];
  }
  ContextGroup& ActiveContextGroup() {
    return context_groups_[active_context_group_index_];
  }
  ContextGroup& MutableActiveContextGroup() {
    return context_groups_[active_context_group_index_];
  }
  absl::Status SwitchContextSizeIfRequired(int required_tokens);

  // Prefill internal implementation, for one prefill call to the Interpreter
  // with a certain length.
  absl::Status PrefillInternal(absl::string_view prefill_signature,
                               absl::Span<const int> ids);

  // Prefill internal implementation using embeddings as input.
  absl::Status PrefillInternalFromEmbeddings(
      absl::string_view prefill_signature,
      absl::Span<const int32_t> sliced_tokens,
      absl::Span<const float> embeddings,
      absl::Span<const float> ple_embeddings,
      absl::Span<const int32_t> seq_positions);

  // Runs the common downstream prefill pipeline (RoPE, Masking, LLM execution,
  // and KV Cache updates) using the pre-populated active buffers.
  absl::Status PrefillCommonPipeline(const ResolvedPrefillSignatures& sigs);

  // Decode internal implementation. Uses the specified 'token' as the input
  // token and uses the specified 'step' as the current time step.  The
  // logits from the decode step are stored in the 'logits' output buffer of
  // the transformer model when this function returns absl::OkStatus().
  //
  // Arguments:
  // - step: The current time step.
  // - token: The input token to decode.
  absl::Status DecodeInternal(int step, std::shared_ptr<TokenData> token);

  // Helper to extract token data and execute DecodeInternal for a single token
  // index.
  absl::Status DecodeSingleToken(size_t idx,
                                 absl::Span<const int32_t> seq_pos_span,
                                 absl::Span<const int32_t> tokens_span,
                                 absl::Span<const float> embeddings,
                                 size_t embedding_dim,
                                 absl::Span<const float> ple_embeddings,
                                 size_t ple_dim);

  // Standard non-speculative decode step.
  absl::StatusOr<std::vector<std::vector<int>>> DecodeNonSpeculative(
      const ExecutorDecodeParams& decode_params, absl::Time start_time);

  // Run the drafter loop for MTP.
  // Persistent buffer to store the sliced activations from the last verifier
  // run, to be used as input for the next speculative cycle.
  std::vector<uint8_t> last_verify_activations_;
  bool has_valid_verify_activations_ = false;

  // Queue of token IDs that have been accepted by the verifier but not yet
  // returned to the engine.
  std::vector<int> pending_accepted_tokens_;

  // Pops the next token from pending_accepted_tokens_, updates state, and
  // returns it.
  absl::StatusOr<std::vector<std::vector<int>>> PopPendingAcceptedToken(
      absl::Time start_time);

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

  // Determine the maximum sequence length that the NPU model supports.
  //
  // 1. Query the 'LlmMetadata' in the provided 'resources' and check if a max
  //    sequence length (referred to as 'max_num_tokens') has been set.
  // 2. Otherwise determine the maximum sequence length supported by checking
  //    the KV cache tensor buffers.
  //
  // Note: If `executor_settings.GetMaxNumTokens()` is set, the minimum of it
  // and the model's supported limit will be returned. A warning is logged if
  // the user requested limit is greater than the model's supported limit.
  static absl::StatusOr<int> DetermineMaxSequenceLength(
      const LlmExecutorSettings& executor_settings, ModelResources& resources,
      const litert::Model& text_decoder_model);

  // Creates the context for the text decoder model.
  static absl::StatusOr<InferenceContext> CreateTextDecoderInferenceContext(
      ::litert::Environment& env,
      ::litert::CompiledModel& text_decoder_compiled_model,
      const ResolvedPrefillSignatures& prefill_signatures,
      absl::string_view decode_signature, absl::string_view verify_signature,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          input_kv_cache_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          prefill_output_kv_cache_slice_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          decode_output_kv_cache_slice_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          verify_output_kv_cache_slice_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          text_decoder_prefill_input_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          text_decoder_decode_input_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          text_decoder_verify_input_buffers);

  static absl::Status WarmupInference(
      ::litert::CompiledModel& text_decoder_compiled_model,
      std::vector<ContextGroup>& context_groups,
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
  absl::Status ClearKVCache(
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>& buffers)
      const;

  // Allocates input and output buffers for the text decoder compiled model:
  // - text_decoder_{prefill,decode,verify}_input_buffers: Non-KV input tensor
  //   buffers (embeddings, attention masks, RoPE, sequence positions) that will
  //   later be shared with the outputs of auxiliary subgraphs (Embedder, Mask,
  //   RoPE).
  // - input_kv_cache_buffers: Persistent full KV cache input tensor buffers
  //   (K, V, and compressed KV) across all layers, shared across all phases and
  //   initialized via FillKVCacheBuffer.
  // - {prefill,decode,verify}_output_kv_cache_slice_buffers: The newly computed
  //   KV cache slice outputs generated by the text decoder, to be written into
  //   input_kv_cache_buffers during the KV cache update step.
  static absl::Status AllocateTextDecoderBuffers(
      litert::Environment& env, const litert::Model* text_decoder_model,
      CompiledModel& text_decoder_compiled_model,
      const ResolvedPrefillSignatures& prefill_signatures,
      absl::string_view decode_signature, absl::string_view verify_signature,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          text_decoder_prefill_input_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          text_decoder_decode_input_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          text_decoder_verify_input_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          input_kv_cache_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          prefill_output_kv_cache_slice_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          decode_output_kv_cache_slice_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          verify_output_kv_cache_slice_buffers,
      absl::flat_hash_map<absl::string_view, HWQuantParams>& kv_quant_params,
      int64_t kv_cache_init_value);

  LlmExecutorSettings executor_settings_;
  ::litert::Environment& env_;
  NpuConfig npu_config_;
  NpuEmbedder main_embedder_;
  NpuRope main_rope_;
  NpuMask main_mask_;
  NpuKVCache main_cache_;

  LatencyStats latency_stats_;
  NpuAuxiliaryContext npu_auxiliary_context_;
  ::litert::CompiledModel text_decoder_compiled_model_;
  std::vector<ContextGroup> context_groups_;
  std::vector<int> sorted_supported_context_sizes_;
  int active_context_group_index_ = 0;
  SortedPrefillSignatureMap prefill_signature_map_;
  // The prefill-family signature names resolved for the prefill length the
  // model was compiled with.
  ResolvedPrefillSignatures prefill_signatures_;

  // MTP / Speculative Decoding members.
  SpeculativeDecodingType speculative_decoding_type_ =
      SpeculativeDecodingType::kNone;
  std::optional<DrafterContext> drafter_context_;
  std::optional<DrafterAuxContext> drafter_aux_context_;

  // The sampler parameters to use for internal sampling.
  proto::SamplerParameters sampler_params_;

  // Internal timestep.
  int current_step_ = 0;

  // Per-tensor quantization parameters for the logits tensor of the 'decode'
  // signature.
  float per_tensor_logits_scale_;
  int32_t per_tensor_logits_zero_point_;

  // The processed tokens.  This is also used to store the pending input token
  // for next prefill or decode steps.
  litert::lm::ProcessedTokens processed_tokens_;

  // Tracks whether a decode step was run so we know how to update the
  // constraint state.
  bool ran_decode_ = false;
  int64_t kv_cache_init_value_ = 0;
  absl::Mutex execution_mutex_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LITERT_NPU_COMPILED_MODEL_EXECUTOR_H_
