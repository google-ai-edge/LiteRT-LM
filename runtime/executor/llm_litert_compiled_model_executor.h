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
#include "runtime/components/sampler.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/litert_compiled_model_executor_utils.h"
#include "runtime/executor/llm_executor.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/executor/llm_executor_processed_tokens.h"
#include "runtime/executor/llm_executor_settings.h"

namespace litert::lm {

// GPU executor that implements the shared functionalities for all GPU backends
// (OpenCl/WebGpu/Metal/etc.). Note that this class itself is not instantiable,
// since the Create() function is not implemented.
// TODO: b/361667248 - Add test for LlmTfLiteGpuExecutor.
class LlmLiteRtCompiledModelExecutor : public LlmExecutor {
 public:
  // Creates a LlmLiteRtCompiledModelExecutor from a LiteRt model.
  static absl::StatusOr<std::unique_ptr<LlmLiteRtCompiledModelExecutor>> Create(
      LlmExecutorSettings executor_settings, Environment& lrt_env,
      ModelResources& resources);

  // Input APIs:
  // Basic API to trigger the "prefill" or "prefix" process.
  // Input is token ids with shape `[batch, sequence_length]`
  absl::Status Prefill(const ExecutorInputs& inputs) override {
    ExecutorPrefillParams params;
    return Prefill(inputs, params);
  };

  // Advanced API to allow customized query parameters.
  // Input is token ids with shape `[batch, sequence_length]`
  absl::Status Prefill(const ExecutorInputs& inputs,
                       const ExecutorPrefillParams& params) override;

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
    return processed_tokens_.TokenCount();
  }

  // Resets all of the internal states.
  absl::Status Reset() override;

  absl::StatusOr<int> GetVocabSize() override;

  // Initializes the sampler.
  absl::Status InitializeSampler(int batch_size);

  using LogitsDataType = ActivationDataType;

 protected:
  LlmLiteRtCompiledModelExecutor(
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
      SortedPrefillSignatureMap prefill_signature_map,
      ModelSignatures signatures, std::string weight_cache_path,
      std::unique_ptr<EmbeddingLookupManager> embedding_lookup = nullptr,
      std::unique_ptr<EmbeddingLookupManager> per_layer_embedding_lookup =
          nullptr,
      LogitsDataType logits_data_type = LogitsDataType::FLOAT32)
      : executor_settings_(std::move(executor_settings)),
        env_(env),
        compiled_model_(std::move(compiled_model)),
        decode_input_buffers_(std::move(decode_input_buffers)),
        decode_output_buffers_(std::move(decode_output_buffers)),
        kv_cache_buffers_1_(std::move(input_kv_cache_buffers)),
        kv_cache_buffers_2_(std::move(output_kv_cache_buffers)),
        input_kv_cache_buffers_(&kv_cache_buffers_1_),
        output_kv_cache_buffers_(&kv_cache_buffers_2_),
        prefill_signature_map_(std::move(prefill_signature_map)),
        signatures_(signatures),
        weight_cache_path_(std::move(weight_cache_path)),
        embedding_lookup_(std::move(embedding_lookup)),
        per_layer_embedding_lookup_(std::move(per_layer_embedding_lookup)),
        logits_data_type_(logits_data_type) {}

 private:
  // Samples output logits and write to ids_tensor.
  absl::Status SampleLogits(const TensorBuffer& logits,
                            TensorBuffer& ids_tensor);

  // Prefill internal implementation, for one prefill call to the Interpreter
  // with a certain length.
  absl::Status PrefillInternal(absl::string_view prefill_signature,
                               absl::Span<const int> ids);

  // Decode internal implementation. Uses the specified 'token' as the input
  // token and uses the specified 'step' as the current time step.  The
  // logits from the decode step are stored in the 'logits' output buffer of
  // the transformer model when this function returns absl::OkStatus().
  absl::Status DecodeInternal(int step, std::shared_ptr<TokenData> token,
                              TensorBuffer& output_logits);

  // Create Prefill input buffers for a given signature.
  absl::Status CreatePrefillInputBuffers(absl::string_view prefill_signature);

  // Fills the input buffer from the unprocessed token.
  absl::Status FillInputBufferWithToken(
      std::shared_ptr<TokenData> unprocessed_token,
      ::litert::TensorBuffer& input_buffer,
      bool is_per_layer_embedding = false);

  // Gets the token to decode. If there is id provided in the inputs, it will be
  // returned as the token to decode. Otherwise, the next unprocessed token will
  // be returned.
  absl::StatusOr<ProcessedTokens::StepAndToken> GetTokenToDecode(
      const ExecutorInputs& inputs);

  // Mark the pending token as processed if there is one, or adds the token as a
  // processed token.
  absl::Status ConsumePendingOrAddProcessedToken(
      std::shared_ptr<TokenData> token);

  LlmExecutorSettings executor_settings_;
  ::litert::Environment& env_;
  ::litert::CompiledModel compiled_model_;
  // The prefill input buffers for each signature.
  // The key is the prefill signature name, and the value is a map of input
  // name to input buffer.
  //
  // Signature names are unique across all signatures in a model so it is safe
  // to refer to them by just their unique name.
  absl::flat_hash_map<std::string /*prefill_signature_name*/,
                      absl::flat_hash_map<absl::string_view /*input_name*/,
                                          ::litert::TensorBuffer>>
      prefill_input_buffers_;
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      decode_input_buffers_;
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      decode_output_buffers_;
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      kv_cache_buffers_1_;
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      kv_cache_buffers_2_;
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>*
      input_kv_cache_buffers_;
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>*
      output_kv_cache_buffers_;

  SortedPrefillSignatureMap prefill_signature_map_;

  // The signatures of the model.
  ModelSignatures signatures_;

  // The sampled ids to use for external sampling.
  // The layout is batch-major.
  // e.g. for output_batch_size=2, the layout is:
  // {batch_0_seq_0, batch_1_seq_0, batch_0_seq_1, batch_1_seq_1, ...}
  std::vector<int> sampled_ids_;

  // Sampler for sampling logits.
  // For now, only CPU sampler is supported.
  std::unique_ptr<Sampler> sampler_;

  // Internal timestep.
  int current_step_ = 0;

  // Keeps track of processed tokens during the LLM execution. This also keeps
  // track of the pending input token, if any.
  ProcessedTokens processed_tokens_;

  // The path to the weight cache directory. Executor will take the ownership of
  // this path to maintain the path lifecycle.
  std::string weight_cache_path_;

  // The embedding lookup for the optional embedder model.
  std::unique_ptr<EmbeddingLookupManager> embedding_lookup_;

  // The embedding lookup for the optional per layer embedder model.
  std::unique_ptr<EmbeddingLookupManager> per_layer_embedding_lookup_;

  // The logits data type of the model, used to determine the data type of the
  // logits tensor for gpu sampling.
  LogitsDataType logits_data_type_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LLM_LITERT_COMPILED_MODEL_EXECUTOR_H_
