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

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/executor/llm_executor_io_types.h"

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
  // Basic API to trigger the "decode" process. On success, fills output tokens
  // tensor buffer of shape `[batch, sequence_length]` of int32_t.
  virtual absl::Status Decode(::litert::TensorBuffer& output_tokens) = 0;

  // Basic API to trigger the "decode" process but without sampling.
  // Input is token ids with shape `[batch, sequence_length]`
  // Output is logits with shape `[batch, sequence_length, vocab_size]` of
  // float32_t.
  virtual absl::Status Decode(const ExecutorInputs& inputs,
                              ::litert::TensorBuffer& output_logits) {
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
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LLM_EXECUTOR_BASE_H_
