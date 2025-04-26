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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LLM_EXECUTOR_IO_TYPES_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LLM_EXECUTOR_IO_TYPES_H_

#include <atomic>
#include <optional>
#include <ostream>
#include <utility>

#include "litert/cc/litert_tensor_buffer.h"  // from @litert

namespace litert::lm {

// Note: Use the operator << to print values only for debugging purposes. It may
// creates copy of the underlying TensorBuffer and make the memory consumption
// high and increase the latency.

// Class to host the text input
class ExecutorTextData {
 public:
  // Default constructor
  ExecutorTextData() = default;

  // Constructor that moves a TensorBuffer
  // New tokens to be processed. Shape `[batch_size, tokens_per_batch]`.
  explicit ExecutorTextData(::litert::TensorBuffer&& token_ids)
      : token_ids_(std::move(token_ids)) {}

  // Getter for token_ids
  const ::litert::TensorBuffer& GetTokenIds() const { return token_ids_; }
  // Getter for mutable token_ids
  ::litert::TensorBuffer& GetMutableTokenIds() { return token_ids_; }

  // Setter for token_ids (moves the input)
  void SetTokenIds(::litert::TensorBuffer&& token_ids) {
    token_ids_ = std::move(token_ids);
  }

 private:
  ::litert::TensorBuffer token_ids_;
};
std::ostream& operator<<(std::ostream& os, const ExecutorTextData& text_data);

// Class to host the vision embeddings input
class ExecutorVisionData {
 public:
  // Special tokens are token ids place holders for vision embeddings
  // input.
  static constexpr int kSpecialToken = -1;

  // Default constructor
  ExecutorVisionData() = default;

  // Constructor that moves optional TensorBuffers. Note that the embeddings are
  // optional and different model may require both or some of them. It is the
  // caller's responsibility to prepare the necessary embeddings in order for
  // the model to function properly.
  // embeddings: Flattened vision embedding matrix with shape
  //   [vision_tokens_num, model_dimension].
  // per_layer_embeddings: Flattened vision per layer embeddings tensor with
  //   shape [stack_size, vision_tokens_num, per_layer_embedding_dimension].
  ExecutorVisionData(
      std::optional<::litert::TensorBuffer>&& embeddings,
      std::optional<::litert::TensorBuffer>&& per_layer_embeddings)
      : embeddings_(std::move(embeddings)),
        per_layer_embeddings_(std::move(per_layer_embeddings)) {}

  // Getters
  const std::optional<::litert::TensorBuffer>& GetEmbeddings() const {
    return embeddings_;
  }
  std::optional<::litert::TensorBuffer>& GetMutableEmbeddings() {
    return embeddings_;
  }

  const std::optional<::litert::TensorBuffer>& GetPerLayerEmbeddings() const {
    return per_layer_embeddings_;
  }
  std::optional<::litert::TensorBuffer>& GetMutablePerLayerEmbeddings() {
    return per_layer_embeddings_;
  }

  // Setters
  void SetEmbeddings(std::optional<::litert::TensorBuffer>&& embeddings) {
    embeddings_ = std::move(embeddings);
  }
  void SetPerLayerEmbeddings(
      std::optional<::litert::TensorBuffer>&& per_layer_embeddings) {
    per_layer_embeddings_ = std::move(per_layer_embeddings);
  }

 private:
  std::optional<::litert::TensorBuffer> embeddings_;
  std::optional<::litert::TensorBuffer> per_layer_embeddings_;
};
std::ostream& operator<<(std::ostream& os,
                         const ExecutorVisionData& vision_data);

// Class to host the audio embeddings input
class ExecutorAudioData {
 public:
  // Special tokens are token ids place holders for vision or audio embeddings
  // input.
  static constexpr int kSpecialToken = -2;

  // Default constructor
  ExecutorAudioData() = default;

  // Constructor that moves optional TensorBuffers
  // embeddings: Flattened audio embedding matrix with shape
  //   [audio_tokens_num, model_dimension].
  // per_layer_embeddings: Flattened audio per layer embeddings tensor with
  //   shape [stack_size, audio_tokens_num, per_layer_embedding_dimension].
  ExecutorAudioData(
      std::optional<::litert::TensorBuffer>&& embeddings,
      std::optional<::litert::TensorBuffer>&& per_layer_embeddings)
      : embeddings_(std::move(embeddings)),
        per_layer_embeddings_(std::move(per_layer_embeddings)) {}

  // Getters
  const std::optional<::litert::TensorBuffer>& GetEmbeddings() const {
    return embeddings_;
  }
  std::optional<::litert::TensorBuffer>& GetMutableEmbeddings() {
    return embeddings_;
  }

  const std::optional<::litert::TensorBuffer>& GetPerLayerEmbeddings() const {
    return per_layer_embeddings_;
  }
  std::optional<::litert::TensorBuffer>& GetMutablePerLayerEmbeddings() {
    return per_layer_embeddings_;
  }

  // Setters
  void SetEmbeddings(std::optional<::litert::TensorBuffer>&& embeddings) {
    embeddings_ = std::move(embeddings);
  }
  void SetPerLayerEmbeddings(
      std::optional<::litert::TensorBuffer>&& per_layer_embeddings) {
    per_layer_embeddings_ = std::move(per_layer_embeddings);
  }

 private:
  std::optional<::litert::TensorBuffer> embeddings_;
  std::optional<::litert::TensorBuffer> per_layer_embeddings_;
};
std::ostream& operator<<(std::ostream& os, const ExecutorAudioData& audio_data);

// Class to bundle all executor inputs
class ExecutorInputs {
 public:
  // Default constructor
  ExecutorInputs() = default;

  // Constructor moving all components
  ExecutorInputs(std::optional<ExecutorTextData>&& text_data,
                 std::optional<ExecutorVisionData>&& vision_data,
                 std::optional<ExecutorAudioData>&& audio_data)
      : text_data_(std::move(text_data)),
        vision_data_(std::move(vision_data)),
        audio_data_(std::move(audio_data)) {}

  // Getter/Setter for text_data
  // New tokens to be processed. Shape `[batch_size, tokens_per_batch]`.
  const std::optional<ExecutorTextData>& GetTextData() const {
    return text_data_;
  }
  std::optional<ExecutorTextData>& GetMutableTextData() { return text_data_; }
  void SetTextData(ExecutorTextData&& text_data) {
    text_data_ = std::move(text_data);
  }

  // Getter/Setter for vision_data
  // Embeddings and per-layer embeddings for visual input.
  //
  // GetVisionData().value().GetEmbeddings() (if present) must have a number of
  // rows equal to the count of kVisionSpecialToken in
  // GetTextData().GetTokenIds(). Each kVisionSpecialToken in
  // GetTextData().GetTokenIds() indicates the position for one corresponding
  // row in the visual embeddings. The shape of
  // GetVisionData().value().GetEmbeddings().value() is: [num_vision_tokens,
  // model_dimension].
  //
  // Similarly, GetVisionData().value().GetPerLayerEmbeddings() must also
  // correspond to the kVisionSpecialToken count. The shape of
  // GetVisionData().value().GetPerLayerEmbeddings().value() is: [num_layers,
  // num_vision_tokens, per_layer_embedding_dimension].
  //
  // Example:
  // GetTextData().GetTokenIds() = [2, kVisionSpecialToken, kVisionSpecialToken,
  // kVisionSpecialToken, 106, 77, (otehr text token ids)...] (contains 3 vision
  // tokens)
  //
  // Then, the vision embeddings should have shape [3,
  // model_dimension]: GetVisionData().value().GetEmbeddings().value() =
  // [[0.1, ...],  // Embedding for the 1st kVisionSpecialToken
  //  [0.5, ...],  // Embedding for the 2nd kVisionSpecialToken
  //  [0.9, ...]]  // Embedding for the 3rd kVisionSpecialToken
  //
  // And the per_layer_embeddings should have shape [num_layers, 3,
  // per_layer_embedding_dimension]:
  // GetVisionData().value().GetPerLayerEmbeddings().value() =
  // [[[0.01, ...], [0.06, ...], [0.11, ...]], // Layer 1 embeddings
  //  [[0.02, ...], [0.07, ...], [0.12, ...]], // Layer 2 embeddings
  //  [..., ...]]
  const std::optional<ExecutorVisionData>& GetVisionData() const {
    return vision_data_;
  }
  std::optional<ExecutorVisionData>& GetMutableVisionData() {
    return vision_data_;
  }
  void SetVisionData(std::optional<ExecutorVisionData>&& vision_data) {
    vision_data_ = std::move(vision_data);
  }

  // Getter/Setter for audio_data
  // Embeddings and per-layer embeddings for audio input.
  //
  // GetAudioData().value().GetEmbeddings() (if present) must have a number of
  // rows equal to the count of kAudioSpecialToken in
  // GetTextData().GetTokenIds(). Each kAudioSpecialToken in
  // GetTextData().GetTokenIds() indicates the position for one corresponding
  // row in the audio embeddings. The shape of
  // GetAudioData().value().GetEmbeddings().value() is: [num_audio_tokens,
  // model_dimension].
  //
  // Similarly, GetAudioData().value().GetPerLayerEmbeddings() must also
  // correspond to the kAudioSpecialToken count. The shape of
  // GetAudioData().value().GetPerLayerEmbeddings().value() is: [num_layers,
  // num_audio_tokens, per_layer_embedding_dimension].
  //
  // Example: Similar to vision input.
  const std::optional<ExecutorAudioData>& GetAudioData() const {
    return audio_data_;
  }
  std::optional<ExecutorAudioData>& GetMutableAudioData() {
    return audio_data_;
  }
  void SetAudioData(std::optional<ExecutorAudioData>&& audio_data) {
    audio_data_ = std::move(audio_data);
  }

 private:
  std::optional<ExecutorTextData> text_data_;
  std::optional<ExecutorVisionData> vision_data_;
  std::optional<ExecutorAudioData> audio_data_;
};
std::ostream& operator<<(std::ostream& os, const ExecutorInputs& inputs);

// Class to host the parameters for Prefill.
class ExecutorPrefillParams {
 public:
  // Default constructor: Initializes members to default values.
  // - current_step: 0
  // - wait_for_completion: false
  // - cancel: nullptr
  ExecutorPrefillParams() = default;

  // Parameterized constructor for all values
  ExecutorPrefillParams(int current_step, bool wait_for_completion,
                        const std::atomic_bool* cancel)
      : current_step_(current_step),
        wait_for_completion_(wait_for_completion),
        cancel_(cancel) {}

  // Getter for current_step
  int GetCurrentStep() const { return current_step_; }
  // Setter for current_step
  void SetCurrentStep(int current_step) { current_step_ = current_step; }

  // Getter for wait_for_completion
  bool GetWaitForCompletion() const { return wait_for_completion_; }
  // Setter for wait_for_completion
  void SetWaitForCompletion(bool wait_for_completion) {
    wait_for_completion_ = wait_for_completion;
  }

  // Getter for cancel flag
  const std::atomic_bool* GetCancelFlag() const { return cancel_; }
  // Setter for cancel flag
  void SetCancelFlag(const std::atomic_bool* cancel) { cancel_ = cancel; }

 private:
  // The current step to prefill.
  int current_step_ = 0;

  // Whether to wait for the prefill to complete before returning.
  bool wait_for_completion_ = false;

  // A cancel flag to cancel the prefill remotely. This is a pointer to an
  // external atomic_bool that the users provides. If the users change the value
  // to true, the Executor is responsible to cancel the Prefill process as soon
  // as possible.
  const std::atomic_bool* cancel_ = nullptr;
};
std::ostream& operator<<(std::ostream& os, const ExecutorPrefillParams& params);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LLM_EXECUTOR_IO_TYPES_H_
