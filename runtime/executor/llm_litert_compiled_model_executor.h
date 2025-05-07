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

#ifndef THIRD_PARTY_ODML_INFRA_GENAI_INFERENCE_EXECUTOR_LLM_TFLITE_GPU_EXECUTOR_H_
#define THIRD_PARTY_ODML_INFRA_GENAI_INFERENCE_EXECUTOR_LLM_TFLITE_GPU_EXECUTOR_H_

#include <memory>
#include <utility>
#include <vector>

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
#include "runtime/components/sampler.h"
#include "runtime/executor/litert_compiled_model_executor_utils.h"
#include "runtime/executor/llm_executor.h"
#include "runtime/executor/llm_executor_io_types.h"
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
      const LlmExecutorSettings& executor_settings,
      ::litert::Model& litert_model);

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

  // Basic API to trigger the "decode" process but without sampling.
  // Input is token ids with shape `[batch, sequence_length]`
  // Output is logits with shape `[batch, sequence_length, vocab_size]`
  // TODO: b/355310550 - Shall we change the function naming here to not
  // overload Decode?
  absl::Status Decode(const ExecutorInputs& inputs,
                      ::litert::TensorBuffer& output_logits) override;

  absl::StatusOr<::litert::TensorBuffer> DecodeLogits(
      const ExecutorInputs& inputs) override;

  absl::string_view ExecutorBackendName() const override {
    return "LiteRT Compiled Model";
  }

  absl::StatusOr<int> GetVocabSize() override;

 protected:
  LlmLiteRtCompiledModelExecutor(
      ::litert::Environment env, ::litert::Model model,
      ::litert::CompiledModel compiled_model,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
          prefill_input_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
          prefill_output_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
          decode_input_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
          decode_output_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
          input_kv_cache_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
          output_kv_cache_buffers,
      SortedPrefillSignatureMap prefill_signature_map,
      ModelSignatures signatures, int batch_size)
      : env_(std::move(env)),
        model_(std::move(model)),
        compiled_model_(std::move(compiled_model)),
        prefill_input_buffers_(std::move(prefill_input_buffers)),
        prefill_output_buffers_(std::move(prefill_output_buffers)),
        decode_input_buffers_(std::move(decode_input_buffers)),
        decode_output_buffers_(std::move(decode_output_buffers)),
        kv_cache_buffers_1_(std::move(input_kv_cache_buffers)),
        kv_cache_buffers_2_(std::move(output_kv_cache_buffers)),
        input_kv_cache_buffers_(&kv_cache_buffers_1_),
        output_kv_cache_buffers_(&kv_cache_buffers_2_),
        prefill_signature_map_(std::move(prefill_signature_map)),
        signatures_(signatures),
        output_batch_size_(batch_size) {}

 private:
  // Samples output logits on CPU and get tokens.
  absl::StatusOr<std::vector<int>> SampleLogits(absl::Span<const float> logits);

  // Prefill internal implementation, for one prefill call to the Interpreter
  // with a certain length.
  absl::Status PrefillInternal(absl::string_view prefill_signature,
                               absl::Span<const int> ids);

  // Decode internal implementation, without result downloading.
  // Caller of this function is responsible for capturing the output.
  absl::Status DecodeInternal(ExecutorInputs inputs);

  ::litert::Environment env_;
  ::litert::Model model_;
  ::litert::CompiledModel compiled_model_;
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      prefill_input_buffers_;
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      prefill_output_buffers_;
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
  // Output batch size for the sampled ids.
  int output_batch_size_ = 0;

  // Sampler for sampling logits.
  // For now, only CPU sampler is supported.
  std::unique_ptr<::litert::lm::Sampler> sampler_;

  // Internal timestep.
  int current_step_ = 0;

  // TODO: b/404625243 - To be implemented.
  // The processed tokens.
  std::vector<int> processed_tokens_;

  // The token served as the first input token to the model for next Prefill or
  // Decode.
  int next_input_token_id_ = -1;

  // A tensor buffer to store the logits decoded before sampling the final
  // tokens. It's to avoid creating a new tensor buffer for each Decode() call.
  ::litert::TensorBuffer decoded_logits_;

  // A vector to store the logits decoded before sampling the final tokens.
  // It's to avoid creating a new vector for each Decode() call.
  std::vector<float> decoded_logits_vector_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_INFRA_GENAI_INFERENCE_EXECUTOR_LLM_TFLITE_GPU_EXECUTOR_H_
