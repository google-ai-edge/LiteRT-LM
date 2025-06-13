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

#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"  // from @litert

namespace litert::lm {

// Note: Use the operator << to print values only for debugging purposes. It may
// create a copy of the underlying TensorBuffer and make the memory consumption
// high and increase the latency.

// Class to host the text input data.
class ExecutorTextData {
 public:
  // Default constructor
  ExecutorTextData() = default;

  // Constructor that moves a TensorBuffer
  // New tokens to be processed. Shape `[batch_size, tokens_per_batch]`.
  explicit ExecutorTextData(::litert::TensorBuffer&& token_ids);

  // Getter for token_ids.
  const ::litert::TensorBuffer& GetTokenIds() const;
  // Getter for mutable token_ids.
  ::litert::TensorBuffer& GetMutableTokenIds();

  // Setter for token_ids (moves the input).
  void SetTokenIds(::litert::TensorBuffer&& token_ids);

 private:
  ::litert::TensorBuffer token_ids_;
};
std::ostream& operator<<(std::ostream& os, const ExecutorTextData& text_data);

// ExecutorVisionData (if present) must have a number of
// rows equal to the count of kVisionSpecialToken in
// ExecutorTextData.GetTokenIds(). Each kVisionSpecialToken in
// ExecutorTextData.GetTokenIds() indicates the position for one corresponding
// row in the visual embeddings. The shape of
// ExecutorVisionData.GetEmbeddings() is: [num_vision_tokens, model_dimension].
//
// Similarly, ExecutorVisionData.GetPerLayerEmbeddings() must also
// correspond to the kVisionSpecialToken count. The shape of
// per_layer_embeddings is: [num_layers, num_vision_tokens,
// per_layer_embedding_dimension].
//
// Example:
// token_ids = [2, kSpecialToken, kSpecialToken, kSpecialToken, 106, 77, (other
// text token ids)...] (contains 3 vision tokens)
//
// Then, the vision embeddings should have shape [3,
// model_dimension]:
// [[0.1, ...],  // Embedding for the 1st kVisionSpecialToken
//  [0.5, ...],  // Embedding for the 2nd kVisionSpecialToken
//  [0.9, ...]]  // Embedding for the 3rd kVisionSpecialToken
//
// And the per_layer_embeddings should have shape [num_layers, 3,
// per_layer_embedding_dimension]:
// [[[0.01, ...], [0.06, ...], [0.11, ...]], // Layer 1 embeddings
//  [[0.02, ...], [0.07, ...], [0.12, ...]], // Layer 2 embeddings
//  [..., ...]]
class ExecutorVisionData {
 public:
  // Special tokens are token ids placeholders for vision embeddings
  // input.
  static constexpr int kSpecialToken = -1;

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
      std::optional<::litert::TensorBuffer>&& per_layer_embeddings);

  // Getters:
  absl::StatusOr<const ::litert::TensorBuffer*> GetEmbeddingsPtr() const;
  absl::StatusOr<::litert::TensorBuffer*> GetMutableEmbeddingsPtr();
  absl::StatusOr<const ::litert::TensorBuffer*> GetPerLayerEmbeddingsPtr()
      const;
  absl::StatusOr<::litert::TensorBuffer*> GetMutablePerLayerEmbeddingsPtr();

  // Setters:
  void SetEmbeddings(std::optional<::litert::TensorBuffer>&& embeddings);
  void SetPerLayerEmbeddings(
      std::optional<::litert::TensorBuffer>&& per_layer_embeddings);

 private:
  std::optional<::litert::TensorBuffer> embeddings_;
  std::optional<::litert::TensorBuffer> per_layer_embeddings_;
};
std::ostream& operator<<(std::ostream& os,
                         const ExecutorVisionData& vision_data);

// ExecutorAudioData.GetEmbeddings() (if present) must have a number of
// rows equal to the count of ExecutorAudioData::kSpecialToken in
// ExecutorTextData.GetTokenIds(). Each kSpecialToken in
// ExecutorTextData.GetTokenIds() indicates the position for one corresponding
// row in the audio embeddings. The shape of
// ExecutorAudioData.GetEmbeddings() is: [num_audio_tokens,
// model_dimension].
//
// Similarly, ExecutorAudioData.GetPerLayerEmbeddings() must also
// correspond to the kSpecialToken count. The shape of
// ExecutorAudioData.GetPerLayerEmbeddings() is: [num_layers,
// num_audio_tokens, per_layer_embedding_dimension].
//
// Example: Similar to vision input.
class ExecutorAudioData {
 public:
  // Special tokens are token ids place holders for vision or audio embeddings
  // input.
  static constexpr int kSpecialToken = -2;
  // TODO: b/410085735 - Define this to be -4 after the end of audio embedding
  // lookup model is ready.
  static constexpr int kEndToken = -4;

  ExecutorAudioData() = default;

  // Constructor that moves optional TensorBuffers
  // embeddings: Flattened audio embedding matrix with shape
  //   [audio_tokens_num, model_dimension].
  // per_layer_embeddings: Flattened audio per layer embeddings tensor with
  //   shape [stack_size, audio_tokens_num, per_layer_embedding_dimension].
  ExecutorAudioData(
      std::optional<::litert::TensorBuffer>&& embeddings,
      std::optional<::litert::TensorBuffer>&& per_layer_embeddings);

  // Getters:
  absl::StatusOr<const ::litert::TensorBuffer*> GetEmbeddingsPtr() const;
  absl::StatusOr<::litert::TensorBuffer*> GetMutableEmbeddingsPtr();
  absl::StatusOr<const ::litert::TensorBuffer*> GetPerLayerEmbeddingsPtr()
      const;
  absl::StatusOr<::litert::TensorBuffer*> GetMutablePerLayerEmbeddingsPtr();

  // Setters:
  void SetEmbeddings(std::optional<::litert::TensorBuffer>&& embeddings);
  void SetPerLayerEmbeddings(
      std::optional<::litert::TensorBuffer>&& per_layer_embeddings);

 private:
  std::optional<::litert::TensorBuffer> embeddings_;
  std::optional<::litert::TensorBuffer> per_layer_embeddings_;
};
std::ostream& operator<<(std::ostream& os, const ExecutorAudioData& audio_data);

class ExecutorInputs {
 public:
  ExecutorInputs() = default;

  ExecutorInputs(std::optional<ExecutorTextData>&& text_data,
                 std::optional<ExecutorVisionData>&& vision_data,
                 std::optional<ExecutorAudioData>&& audio_data);

  // Getters for top-level optional members
  absl::StatusOr<const ExecutorTextData*> GetTextDataPtr() const;
  absl::StatusOr<ExecutorTextData*> GetMutableTextDataPtr();
  absl::StatusOr<const ExecutorVisionData*> GetVisionDataPtr() const;
  absl::StatusOr<ExecutorVisionData*> GetMutableVisionDataPtr();
  absl::StatusOr<const ExecutorAudioData*> GetAudioDataPtr() const;
  absl::StatusOr<ExecutorAudioData*> GetMutableAudioDataPtr();

  // Getters for NESTED members
  absl::StatusOr<const ::litert::TensorBuffer*> GetTextTokenIdsPtr() const;
  absl::StatusOr<::litert::TensorBuffer*> GetMutableTextTokenIdsPtr();
  absl::StatusOr<const ::litert::TensorBuffer*> GetVisionEmbeddingsPtr() const;
  absl::StatusOr<::litert::TensorBuffer*> GetMutableVisionEmbeddingsPtr();
  absl::StatusOr<const ::litert::TensorBuffer*> GetVisionPerLayerEmbeddingsPtr()
      const;
  absl::StatusOr<::litert::TensorBuffer*>
  GetMutableVisionPerLayerEmbeddingsPtr();
  absl::StatusOr<const ::litert::TensorBuffer*> GetAudioEmbeddingsPtr() const;
  absl::StatusOr<::litert::TensorBuffer*> GetMutableAudioEmbeddingsPtr();
  absl::StatusOr<const ::litert::TensorBuffer*> GetAudioPerLayerEmbeddingsPtr()
      const;
  absl::StatusOr<::litert::TensorBuffer*>
  GetMutableAudioPerLayerEmbeddingsPtr();

  // Setters:
  void SetTextData(ExecutorTextData&& text_data);
  void SetVisionData(std::optional<ExecutorVisionData>&& vision_data);
  void SetAudioData(std::optional<ExecutorAudioData>&& audio_data);

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
  // - current_step: -1
  // - wait_for_completion: false
  // - cancel: nullptr
  ExecutorPrefillParams() = default;

  // Parameterized constructor for all values
  ExecutorPrefillParams(int current_step, bool wait_for_completion,
                        const std::atomic_bool* cancel);

  int GetCurrentStep() const;
  void SetCurrentStep(int current_step);

  bool GetWaitForCompletion() const;
  void SetWaitForCompletion(bool wait_for_completion);

  const std::atomic_bool* GetCancelFlag() const;
  void SetCancelFlag(const std::atomic_bool* cancel);

 private:
  // The current step to prefill.
  int current_step_ = -1;

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
