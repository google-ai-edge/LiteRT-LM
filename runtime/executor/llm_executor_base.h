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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LLM_EXECUTOR_BASE_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LLM_EXECUTOR_BASE_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/executor/llm_executor_processed_tokens.h"
#include "runtime/executor/llm_executor_settings.h"

namespace litert::lm {

// The LLM Executor serves as a lightweight and portable wrapper around various
// converted LLM model formats, i.e. LiteRT. It aims to provide a general,
// minimal-dependency interface for the users, abstracting the complexities of
// executing models across diverse hardware accelerators like CPUs, GPUs, and
// other ML accelerators.
class LlmExecutorBase {
 public:
  virtual ~LlmExecutorBase() = default;

  // ------------Input APIs------------:
  // Basic API to trigger the "prefill" or "prefix" process.
  // Input is token ids with shape `[batch, sequence_length]`
  virtual absl::Status Prefill(const ExecutorInputs& inputs) = 0;

  // Advanced API to allow customized query parameters.
  // Input is token ids with shape `[batch, sequence_length]`
  virtual absl::Status Prefill(const ExecutorInputs& inputs,
                               const ExecutorPrefillParams& prefill_params) {
    return absl::UnimplementedError(absl::StrCat(
        "Prefill with prefill params not implemented for backend: ",
        ExecutorBackendName()));
  };

  // ------------Output APIs------------:
  // Basic API to trigger the "decode" process. On success, will return a vector
  // of token ids of generated output tokens, one per candidate.
  virtual absl::StatusOr<std::vector<std::vector<int>>> Decode() = 0;

  // Advanced API to trigger the decode and sampling process with custom
  // parameters. On success, will return a vector of token ids of generated
  // output tokens, one per candidate.
  virtual absl::StatusOr<std::vector<std::vector<int>>> Decode(
      const ExecutorDecodeParams& decode_params) {
    return absl::UnimplementedError(
        absl::StrCat("Decode with decode params not implemented for backend: ",
                     ExecutorBackendName()));
  }

  // [Deprecated]Basic API to trigger the "decode" process but without sampling.
  // Input is token ids with shape `[batch, sequence_length]`
  // Output is logits with shape `[batch, sequence_length, vocab_size]` of
  // float32_t.
  // TODO(b/416293708): remove this API once the new API is ready and tested.
  virtual absl::Status Decode(const ExecutorInputs& inputs,
                              ::litert::TensorBuffer& output_logits) {
    return absl::UnimplementedError(
        absl::StrCat("Decode for logits output not implemented for backend: ",
                     ExecutorBackendName()));
  };

  // Basic API to trigger the "decode" process but without sampling.
  // Input is token ids with shape `[batch, sequence_length]`
  // Output is logits with shape `[batch, sequence_length, vocab_size]` of
  // float32_t on the host memory.
  virtual absl::StatusOr<::litert::TensorBuffer> DecodeLogits(
      const ExecutorInputs& inputs) {
    return absl::UnimplementedError(
        absl::StrCat("Decode for logits output not implemented for backend: ",
                     ExecutorBackendName()));
  };

  virtual absl::string_view ExecutorBackendName() const = 0;

  // Get vocabulary size used to build tensor buffers for decode functions.
  virtual absl::StatusOr<int> GetVocabSize() {
    return absl::UnimplementedError(absl::StrCat(
        "GetVocabSize not implemented for backend: ", ExecutorBackendName()));
  };

  // Gets the current step of the executor.
  virtual absl::StatusOr<int> GetCurrentStep() const {
    return absl::UnimplementedError(absl::StrCat(
        "GetCurrentStep not implemented for backend: ", ExecutorBackendName()));
  };

  virtual absl::Status SetCurrentStep(int current_step) {
    return absl::UnimplementedError(absl::StrCat(
        "SetCurrentStep not implemented for backend: ", ExecutorBackendName()));
  };

  // Gets the executor settings of the executor.
  virtual absl::StatusOr<LlmExecutorSettings> GetExecutorSettings() const {
    return absl::UnimplementedError(
        absl::StrCat("GetExecutorSettings not implemented for backend: ",
                     ExecutorBackendName()));
  };

  // Updates the executor settings.
  virtual absl::Status UpdateExecutorSettings(
      const LlmExecutorSettings& executor_settings) {
    return absl::OkStatus();
  }

  // Gets the runtime configuration.
  virtual absl::StatusOr<RuntimeConfig> GetRuntimeConfig() const {
    return absl::UnimplementedError(
        absl::StrCat("GetRuntimeConfig not implemented for backend: ",
                     ExecutorBackendName()));
  };

  // Updates the runtime configuration.
  virtual absl::Status UpdateRuntimeConfig(
      const RuntimeConfig& runtime_config) {
    return absl::UnimplementedError(
        absl::StrCat("UpdateRuntimeConfig not implemented for backend: ",
                     ExecutorBackendName()));
  };

  // Gets the runtime state.
  virtual absl::StatusOr<RuntimeState> GetRuntimeState() const {
    return absl::UnimplementedError(
        absl::StrCat("GetRuntimeState not implemented for backend: ",
                     ExecutorBackendName()));
  };

  // Updates the runtime state.
  virtual absl::Status UpdateRuntimeState(const RuntimeState& runtime_state) {
    return absl::UnimplementedError(
        absl::StrCat("UpdateRuntimeState not implemented for backend: ",
                     ExecutorBackendName()));
  };

  // Gets the processed tokens of the executor. This is used by resource manager
  // to check if processed context copying is needed.
  virtual absl::StatusOr<const ProcessedTokens*> GetProcessedTokens() const {
    return absl::UnimplementedError(
        absl::StrCat("processed_tokens not implemented for backend: ",
                     ExecutorBackendName()));
  }

  // Gets the litert environment used by the executor.
  // This is used by AICore's EmbeddingModelMldrift to convert tokens to
  // TensorBuffers.
  virtual ::litert::Environment* GetEnvironment() const { return nullptr; };

  // ------------Vision APIs------------:
  // This function will populate the GPU tensors with the vision embeddings and
  // vision per layer embeddings. This should only be used before the
  // prefill/prefix stage.
  // vision_input: The vision embeddings to populate the GPU tensors with.
  // image_index: The index of the image in the batch. It must be non-negative
  // and less than `max_num_images`. This will overwrite the vision embeddings
  // for the given image index if it is already populated.
  virtual absl::Status FillVisionEmbeddings(
      const ExecutorVisionData& vision_input, int image_index) {
    return absl::UnimplementedError(
        absl::StrCat("FillVisionEmbeddings not implemented for backend: ",
                     ExecutorBackendName()));
  };

  // Resets all of the internal states (e.g. KVCache). Loaded and used LoRA
  // models are not affected (remain loaded and in use).
  virtual absl::Status Reset() {
    return absl::UnimplementedError(absl::StrCat(
        "Reset not implemented for backend: ", ExecutorBackendName()));
  };

  // ------------State/context management APIs------------:
  // Creates a new context with the given configs.
  virtual absl::StatusOr<std::unique_ptr<LlmContext>> CreateNewContext(
      std::optional<uint32_t> lora_id, RuntimeConfig runtime_config) const {
    return absl::UnimplementedError(
        absl::StrCat("CreateNewContext not implemented for backend: ",
                     ExecutorBackendName()));
  };

  // Performs necessary operations to clone the current llm context from the
  // executor and returns it to the caller.
  virtual absl::StatusOr<std::unique_ptr<LlmContext>> CloneContext() const {
    return absl::UnimplementedError(absl::StrCat(
        "GetContext not implemented for backend: ", ExecutorBackendName()));
  };

  // Sets the llm_context and performs necessary operations to make sure the
  // model is restored with the provided llm context.
  virtual absl::Status RestoreContext(
      std::unique_ptr<LlmContext> context_data) {
    return absl::UnimplementedError(absl::StrCat(
        "RestoreContext not implemented for backend: ", ExecutorBackendName()));
  };
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LLM_EXECUTOR_BASE_H_
