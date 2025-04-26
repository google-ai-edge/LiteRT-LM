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

#include <atomic>
#include <iostream>
#include <optional>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/litert/cc/litert_tensor_buffer.h"  // from @litert

namespace litert::lm {

// Special tokens are token ids place holders for vision or audio embeddings
// input.
constexpr int kVisionSpecialToken = -1;
constexpr int kAudioSpecialToken = -2;

struct TextInput {
  // New tokens to be processed. Shape `[batch_size, tokens_per_batch]`.
  ::litert::TensorBuffer token_ids;
};
std::ostream& operator<<(std::ostream& os, const TextInput& text_input);

// Struct to host the vision embeddings input
// embeddings: Flattened vision embedding matrix with shape
//   [vision_tokens_num, model_dimension].
// per_layer_embeddings: Flattened vision per layer embeddings tensor with
//   shape [stack_size, vision_tokens_num, per_layer_embedding_dimension].
struct VisionInput {
  std::optional<::litert::TensorBuffer> embeddings;
  std::optional<::litert::TensorBuffer> per_layer_embeddings;
};
std::ostream& operator<<(std::ostream& os, const VisionInput& vision_input);

// Struct to host the audio embeddings input
// embeddings: Flattened audio embedding matrix with shape
//   [audio_tokens_num, model_dimension].
// per_layer_embeddings: Flattened audio per layer embeddings tensor with
//   shape [stack_size, audio_tokens_num, per_layer_embedding_dimension].
struct AudioInput {
  std::optional<::litert::TensorBuffer> embeddings;
  std::optional<::litert::TensorBuffer> per_layer_embeddings;
};
std::ostream& operator<<(std::ostream& os, const AudioInput& audio_input);

struct Inputs {
  // New tokens to be processed. Shape `[batch_size, tokens_per_batch]`.
  TextInput text_input;

  // Embeddings and per-layer embeddings for visual input.
  //
  // vision_input.embeddings must have a number of rows equal to the count of
  // kVisionSpecialToken in text_input.token_ids.
  // Each kVisionSpecialToken in text_input.token_ids indicates the position for
  // one corresponding row in the visual embeddings.
  // The shape of vision_input.embeddings is:
  // [num_vision_tokens, model_dimension].
  //
  // Similarly, vision_input.per_layer_embeddings must also correspond to the
  // kVisionSpecialToken count.
  // The shape of vision_input.per_layer_embeddings is:
  // [num_layers, num_vision_tokens, per_layer_embedding_dimension].
  //
  // Example:
  // text_input.token_ids = [2, kVisionSpecialToken, kVisionSpecialToken,
  //                         kVisionSpecialToken, 106]
  // (contains 3 vision tokens)
  //
  // Then, vision_input.embeddings should have shape [3, model_dimension]:
  // vision_input.embeddings =
  // [[0.1, ...],  // Embedding for the 1st kVisionSpecialToken
  //  [0.5, ...],  // Embedding for the 2nd kVisionSpecialToken
  //  [0.9, ...]]  // Embedding for the 3rd kVisionSpecialToken
  //
  // And vision_input.per_layer_embeddings should have shape
  // [num_layers, 3, per_layer_embedding_dimension]:
  // vision_input.per_layer_embeddings =
  // [[[0.01, ...], [0.06, ...], [0.11, ...]], // Layer 1 embeddings
  //  [[0.02, ...], [0.07, ...], [0.12, ...]], // Layer 2 embeddings
  //  [..., ...]]
  std::optional<VisionInput> vision_input;

  // Embeddings and per-layer embeddings for audio input.
  //
  // audio_input.embeddings must have a number of rows equal to the count of
  // kAudioSpecialToken in text_input.token_ids.
  // Each kAudioSpecialToken in text_input.token_ids indicates the position for
  // one corresponding row in the audio embeddings.
  // The shape of audio_input.embeddings is:
  // [num_audio_tokens, model_dimension].
  //
  // Similarly, audio_input.per_layer_embeddings must also correspond to the
  // kAudioSpecialToken count.
  // The shape of audio_input.per_layer_embeddings is:
  // [num_layers, num_audio_tokens, per_layer_embedding_dimension].
  //
  // Example: Similar to vision input.
  std::optional<AudioInput> audio_input;
};
std::ostream& operator<<(std::ostream& os, const Inputs& inputs);

// Struct to host the parameters for Prefill.
struct PrefillQueryParams {
  // The current step to prefill.
  int current_step;

  // Whether to wait for the prefill to complete before returning.
  bool wait_for_completion;

  // A cancel flag to cancel the prefill remotely.
  const std::atomic_bool* cancel;
};
std::ostream& operator<<(std::ostream& os, const PrefillQueryParams& params);

// TODO(b/412847331): provide better documentation.
class LlmExecutorBase {
 public:
  virtual ~LlmExecutorBase() = default;

  // ------------Input APIs------------:
  // Basic API to trigger the "prefill" or "prefix" process.
  // Input is token ids with shape `[batch, sequence_length]`
  virtual absl::Status Prefill(const Inputs& inputs) = 0;

  // Advanced API to allow customized query parameters.
  // Input is token ids with shape `[batch, sequence_length]`
  virtual absl::Status Prefill(const Inputs& inputs,
                               const PrefillQueryParams& prefill_query_params) {
    return absl::UnimplementedError(
        absl::StrCat("Prefill with query params not implemented for backend: ",
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
  virtual absl::Status Decode(const Inputs& inputs,
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
  virtual absl::Status FillVisionEmbeddings(const VisionInput& vision_input,
                                            int image_index) {
    return absl::UnimplementedError(
        absl::StrCat("FillVisionEmbeddings not implemented for backend: ",
                     ExecutorBackendName()));
  };
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LLM_EXECUTOR_BASE_H_
