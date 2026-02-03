// Copyright 2024 The ODML Authors.
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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LLM_LITERT_COMPILED_MODEL_EXECUTOR_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LLM_LITERT_COMPILED_MODEL_EXECUTOR_H_

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
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
#include "runtime/components/lora_manager.h"
#include "runtime/components/sampler.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/litert_compiled_model_executor_utils.h"
#include "runtime/executor/llm_executor.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/executor/llm_executor_processed_tokens.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/executor/llm_processed_context.h"

namespace litert::lm {

// GPU executor that implements the shared functionalities for all GPU backends
// (OpenCl/WebGpu/Metal/etc.). Note that this class itself is not instantiable,
// since the Create() function is not implemented.
// TODO: b/361667248 - Add test for LlmTfLiteGpuExecutor.
class LlmLiteRtCompiledModelExecutorBase : public LlmExecutor {
 public:
  using LlmExecutor::Prefill;

  // Input APIs:
  // Basic API to trigger the "prefill" or "prefix" process.
  // Input is token ids with shape `[batch, sequence_length]`
  absl::Status Prefill(const ExecutorInputs& inputs) override {
    ExecutorPrefillParams params;
    return Prefill(inputs, params);
  };

  // Output APIs:
  // Basic API to trigger the "decode" process.
  absl::Status Decode(::litert::TensorBuffer& output_tokens) override;

  // Advanced API to allow customized query parameters.
  absl::Status Decode(::litert::TensorBuffer& output_tokens,
                      const ExecutorDecodeParams& decode_params) override;

  // Basic API to trigger the "decode" process but without sampling.
  // Input is token ids with shape `[batch, sequence_length]`
  // Output is logits with shape `[batch, sequence_length, vocab_size]`
  // TODO: b/355310550 - Shall we change the function naming here to not
  // overload Decode?
  absl::Status Decode(const ExecutorInputs& inputs,
                      ::litert::TensorBuffer& output_logits) override;

  absl::StatusOr<::litert::TensorBuffer> DecodeLogits(
      const ExecutorInputs& inputs) override;

  absl::StatusOr<::litert::TensorBuffer> DecodeLogits(
      const ExecutorInputs& inputs, const ExecutorDecodeParams& decode_params);

  absl::string_view ExecutorBackendName() const override {
    return "LiteRT Compiled Model";
  }

  // Gets the executor settings.
  absl::StatusOr<LlmExecutorSettings> GetExecutorSettings() const override {
    return executor_settings_;
  }

  // Gets the current step of the executor.
  // Public API, the return value is the current step that user expects (e.g.
  // users prefill 100 tokens, then they expect the current step to be 100). It
  // is different from the internal current step.
  absl::StatusOr<int> GetCurrentStep() const override {
    return llm_context_->runtime_state().current_step;
  }

  // Sets the current step of the executor.
  absl::Status SetCurrentStep(int new_step) override;

  // Resets all of the internal states.
  absl::Status Reset() override;

  absl::StatusOr<int> GetVocabSize() override;

  // Initializes the sampler.
  // `logits_data_type` is optional because the executor usually knows the
  // logits data type from initialization. If it is not provided, the executor
  // uses the internally stored `logits_data_type_`.
  absl::Status InitializeSampler(
      std::optional<ActivationDataType> logits_data_type = std::nullopt);

  using LogitsDataType = ActivationDataType;

  // Sets the LoRA manager used for injecting LoRA buffers into forward passes.
  // The caller retains ownership of the LoraManager.
  void SetLoraManager(LoraManager* lora_manager) {
    lora_manager_ = lora_manager;
  }

  const ProcessedTokens& processed_tokens_for_testing() const {
    return llm_context_->processed_context().processed_tokens();
  }

 protected:
  LlmLiteRtCompiledModelExecutorBase(
      LlmExecutorSettings executor_settings, ::litert::Environment& env,
      const ::litert::Model* absl_nonnull model,
      ::litert::CompiledModel compiled_model,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
          decode_input_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
          decode_output_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
          input_kv_cache_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
          output_kv_cache_buffers,
      std::optional<
          absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>>
          decode_input_kv_cache_buffers,
      std::optional<
          absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>>
          decode_output_kv_cache_buffers,
      ModelSignatures signatures, int output_batch_size,
      std::string weight_cache_path,
      std::unique_ptr<EmbeddingLookupManager> embedding_lookup,
      std::unique_ptr<EmbeddingLookupManager> per_layer_embedding_lookup,
      bool use_fp16_precision, LogitsDataType logits_data_type)
      : executor_settings_(std::move(executor_settings)),
        env_(env),
        model_(*model),
        compiled_model_(std::move(compiled_model)),
        decode_input_buffers_(std::move(decode_input_buffers)),
        decode_output_buffers_(std::move(decode_output_buffers)),
        kv_cache_buffers_1_(std::move(input_kv_cache_buffers)),
        kv_cache_buffers_2_(std::move(output_kv_cache_buffers)),
        input_kv_cache_buffers_(&kv_cache_buffers_1_),
        output_kv_cache_buffers_(&kv_cache_buffers_2_),
        decode_kv_cache_buffers_1_(std::move(decode_input_kv_cache_buffers)),
        decode_kv_cache_buffers_2_(std::move(decode_output_kv_cache_buffers)),
        signatures_(signatures),
        weight_cache_path_(std::move(weight_cache_path)),
        embedding_lookup_(std::move(embedding_lookup)),
        per_layer_embedding_lookup_(std::move(per_layer_embedding_lookup)),
        use_fp16_precision_(use_fp16_precision),
        logits_data_type_(logits_data_type) {
    auto processed_context = std::make_unique<LlmProcessedContext>(
        std::nullopt,
        absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>(),
        ProcessedTokens());
    auto runtime_config = std::make_unique<RuntimeConfig>();
    runtime_config->output_heads = output_batch_size;
    auto runtime_state = std::make_unique<RuntimeState>();
    llm_context_ = std::make_unique<LlmContext>(std::move(processed_context),
                                                std::move(runtime_config),
                                                std::move(runtime_state));
  }

 protected:
  // Rolls back the processed tokens to the current step.
  absl::Status RollBackProcessedTokens();

  // Swaps the input tensors before Sampling when the sampler handles input.
  // Current input_pos and mask tensors in decode_input_buffers_ are swapped
  // with decode_prev_input_pos_ and decode_prev_mask_, i.e. current ones become
  // previous ones, and new current ones will be calculated from the previous
  // ones by the sampler.
  absl::Status SwapSamplerInputTensors();
  // Sets or resets the input tensors and inference function for the sampler.
  absl::Status SetSamplerInputHandling(bool reset);

  // Samples output logits and write to ids_tensor.
  absl::Status SampleLogits(const TensorBuffer& logits,
                            TensorBuffer& ids_tensor);

  // Prefill internal implementation, for one prefill call to the Interpreter
  // with a certain length synchronously or asynchronously.
  absl::Status PrefillInternal(
      absl::string_view prefill_signature,
      absl::flat_hash_map<absl::string_view /*input_name*/,
                          ::litert::TensorBuffer>& prefill_input_buffers,
      absl::Span<const int> ids, bool async);

  // Decode internal implementation. Uses the specified 'token' as the input
  // token and uses the specified 'step' as the current time step.  The
  // logits from the decode step are stored in the 'logits' output buffer of
  // the transformer model when this function returns absl::OkStatus().
  virtual absl::Status DecodeInternal(
      const std::vector<std::shared_ptr<TokenData>>& token,
      TensorBuffer& output_logits);

  // Helper function of DecodeInternal to bind input/output tensors for decode
  // and run decode signature.
  absl::Status BindTensorsAndRunDecode(
      const absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          input_buffers,
      TensorBuffer* output_logits);
  // Static version of BindTensorsAndRunDecode to be used as a callback for
  // sampler.
  static int BindTensorsAndRunDecodeStatic(void* arg);

  // Creates Prefill input buffers for a given signature.
  absl::Status CreatePrefillInputBuffers(
      absl::string_view prefill_signature, int sequence_length,
      int context_length,
      absl::flat_hash_map<absl::string_view, TensorBuffer>&
          prefill_input_buffers);

  // Fills the input buffer from the unprocessed token.
  absl::Status FillInputBufferWithToken(
      const std::vector<std::shared_ptr<TokenData>>& unprocessed_token,
      ::litert::TensorBuffer& input_buffer,
      bool is_per_layer_embedding = false);

  // Prepares the first prefill step possibly after decode.
  // When output_batch_size_ > 1, It selects only one set of KV cache buffers.
  absl::Status PrepareFirstPrefillAfterDecode(int token_index_to_reduce);

  // Prepares the first decode step.
  // When output_batch_size_ > 1, It broadcasts KV cache buffers to
  // output_batch_size_ times for the rest of the decode steps.
  // When output_batch_size_ == 1, It doesn't do anything.
  absl::Status PrepareFirstDecode();

  // Gets the token to decode. If there is id provided in the inputs, it will be
  // returned as the token to decode. Otherwise, the next unprocessed token will
  // be returned.
  absl::StatusOr<ProcessedTokens::StepAndToken> GetTokenToDecode(
      const ExecutorInputs& inputs);

  // Mark the pending token as processed if there is one, or adds the token as a
  // processed token.
  absl::Status ConsumePendingOrAddProcessedToken(
      const std::vector<std::shared_ptr<TokenData>>& token);

  LlmExecutorSettings executor_settings_;
  ::litert::Environment& env_;
  const ::litert::Model& model_;
  ::litert::CompiledModel compiled_model_;

  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      decode_input_buffers_;
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      decode_output_buffers_;
  // KV cache double buffers because some GPU backends can't allocate one buffer
  // for both read and write at the same time.
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      kv_cache_buffers_1_;
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      kv_cache_buffers_2_;
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>*
      input_kv_cache_buffers_;
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>*
      output_kv_cache_buffers_;
  // KV cache (double) buffers used during decode when output_batch_size_ > 1.
  std::optional<absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>>
      decode_kv_cache_buffers_1_;
  std::optional<absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>>
      decode_kv_cache_buffers_2_;

  // The signatures of the model.
  ModelSignatures signatures_;

  // The context of the executor, which contains
  // 1. The configuration settings.
  // 2. The internal states.
  // 3. The processed tokens.(e.g. KVCache)
  std::unique_ptr<LlmContext> llm_context_;

  // Whether the executor needs to prepare the kvcache buffers before execution.
  bool force_prepare_needed_ = false;

  // Sampler for sampling logits.
  // For now, only CPU sampler is supported.
  std::unique_ptr<Sampler> sampler_;
  bool sampler_handles_input_ = true;
  // Extra input tensors to swap for decode when sampler handles input tensors.
  ::litert::TensorBuffer decode_prev_input_pos_;
  ::litert::TensorBuffer decode_prev_mask_;

  // The path to the weight cache directory. Executor will take the ownership of
  // this path to maintain the path lifecycle.
  std::string weight_cache_path_;

  // The embedding lookup for the optional embedder model.
  std::unique_ptr<EmbeddingLookupManager> embedding_lookup_;

  // The embedding lookup for the optional per layer embedder model.
  std::unique_ptr<EmbeddingLookupManager> per_layer_embedding_lookup_;

  // Whether to use FP16 precision for the calculation.
  bool use_fp16_precision_;

  // The logits data type of the model, used to determine the data type of the
  // logits tensor for gpu sampling.
  LogitsDataType logits_data_type_;

  // GPU optimized single buffer cache
  bool gpu_optimized_single_buffer_cache_ = false;

  // Optional LoRA manager for injecting LoRA buffers into forward passes.
  // Not owned by the executor.
  LoraManager* lora_manager_ = nullptr;
};

// The static executor for the prefill-decode compiled model.
// This variant is instantiated when the model is statically shaped.
class LlmLiteRtCompiledModelExecutorStatic
    : public LlmLiteRtCompiledModelExecutorBase {
 public:
  static absl::StatusOr<std::unique_ptr<LlmLiteRtCompiledModelExecutorStatic>>
  Create(LlmExecutorSettings executor_settings, Environment& lrt_env,
         ModelResources& resources);

  using LlmLiteRtCompiledModelExecutorBase::Prefill;

  absl::Status Prefill(const ExecutorInputs& inputs,
                       const ExecutorPrefillParams& params) override;

 private:
  LlmLiteRtCompiledModelExecutorStatic(
      LlmExecutorSettings executor_settings, ::litert::Environment& env,
      const ::litert::Model* absl_nonnull model,
      ::litert::CompiledModel compiled_model,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
          decode_input_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
          decode_output_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
          input_kv_cache_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
          output_kv_cache_buffers,
      std::optional<
          absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>>
          decode_input_kv_cache_buffers,
      std::optional<
          absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>>
          decode_output_kv_cache_buffers,
      SortedPrefillSignatureMap prefill_signature_map,
      ModelSignatures signatures, int output_batch_size,
      std::string weight_cache_path,
      std::unique_ptr<EmbeddingLookupManager> embedding_lookup = nullptr,
      std::unique_ptr<EmbeddingLookupManager> per_layer_embedding_lookup =
          nullptr,
      bool use_fp16_precision = true,
      LogitsDataType logits_data_type = LogitsDataType::FLOAT32)
      : LlmLiteRtCompiledModelExecutorBase(
            std::move(executor_settings), env, model, std::move(compiled_model),
            std::move(decode_input_buffers), std::move(decode_output_buffers),
            std::move(input_kv_cache_buffers),
            std::move(output_kv_cache_buffers),
            std::move(decode_input_kv_cache_buffers),
            std::move(decode_output_kv_cache_buffers), signatures,
            output_batch_size, std::move(weight_cache_path),
            std::move(embedding_lookup), std::move(per_layer_embedding_lookup),
            use_fp16_precision, logits_data_type),
        prefill_signature_map_(std::move(prefill_signature_map)) {}

  SortedPrefillSignatureMap prefill_signature_map_;
  // Signature names are unique across all signatures in a model so it is safe
  // to refer to them by just their unique name.
  absl::flat_hash_map<std::string /*prefill_signature_name*/,
                      absl::flat_hash_map<absl::string_view /*input_name*/,
                                          ::litert::TensorBuffer>>
      prefill_input_buffers_;
};

// The dynamic executor for the prefill-decode compiled model.
// This variant is instantiated when the model is dynamically shaped, in
// particular, input sequence length and KV cache size are dynamic.
class LlmLiteRtCompiledModelExecutorDynamic
    : public LlmLiteRtCompiledModelExecutorBase {
 public:
  static absl::StatusOr<std::unique_ptr<LlmLiteRtCompiledModelExecutorDynamic>>
  Create(LlmExecutorSettings executor_settings, Environment& lrt_env,
         ModelResources& resources);

  using LlmLiteRtCompiledModelExecutorBase::Prefill;

  absl::Status Prefill(const ExecutorInputs& inputs,
                       const ExecutorPrefillParams& params) override;

 private:
  LlmLiteRtCompiledModelExecutorDynamic(
      LlmExecutorSettings executor_settings, ::litert::Environment& env,
      const ::litert::Model* absl_nonnull model,
      ::litert::CompiledModel compiled_model,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
          decode_input_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
          decode_output_buffers,
      int prefill_chunk_size, int key_dynamic_dim_index,
      int value_dynamic_dim_index, int kv_increament_size,
      std::vector<std::string> key_cache_input_names,
      std::vector<std::string> value_cache_input_names,
      ModelSignatures signatures, int output_batch_size,
      std::string weight_cache_path,
      std::unique_ptr<EmbeddingLookupManager> embedding_lookup = nullptr,
      std::unique_ptr<EmbeddingLookupManager> per_layer_embedding_lookup =
          nullptr,
      bool use_fp16_precision = true,
      LogitsDataType logits_data_type = LogitsDataType::FLOAT32)
      : LlmLiteRtCompiledModelExecutorBase(
            std::move(executor_settings), env, model, std::move(compiled_model),
            std::move(decode_input_buffers), std::move(decode_output_buffers),
            /*input_kv_cache_buffers=*/{},
            /*output_kv_cache_buffers=*/{},
            /*decode_input_kv_cache_buffers=*/std::nullopt,
            /*decode_output_kv_cache_buffers=*/std::nullopt, signatures,
            output_batch_size, std::move(weight_cache_path),
            std::move(embedding_lookup), std::move(per_layer_embedding_lookup),
            use_fp16_precision, logits_data_type),
        prefill_chunk_size_(prefill_chunk_size),
        key_dynamic_dim_index_(key_dynamic_dim_index),
        value_dynamic_dim_index_(value_dynamic_dim_index),
        kv_increament_size_(kv_increament_size),
        key_cache_input_names_(std::move(key_cache_input_names)),
        value_cache_input_names_(std::move(value_cache_input_names)) {}

  absl::Status PrefillInternal(absl::Span<int> ids,
                               const ExecutorPrefillParams& params);

  // Extends the base class DecodeInternal to handle KV cache buffers.
  absl::Status DecodeInternal(
      const std::vector<std::shared_ptr<TokenData>>& token,
      TensorBuffer& output_logits) override;

  int prefill_chunk_size_;
  int key_dynamic_dim_index_;
  int value_dynamic_dim_index_;
  uint32_t kv_increament_size_;
  std::vector<std::string> key_cache_input_names_;
  std::vector<std::string> value_cache_input_names_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LLM_LITERT_COMPILED_MODEL_EXECUTOR_H_
