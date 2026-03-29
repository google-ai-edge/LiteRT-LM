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
#include <cstdint>
#include <memory>
#include <optional>
#include <ostream>
#include <random>
#include <utility>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/constrained_decoding/constrained_decoder.h"
#include "runtime/executor/llm_executor_processed_tokens.h"
#include "runtime/executor/llm_executor_settings.h"

namespace litert::lm {

// KVCache direct related context container.
class ProcessedContext {
 public:
  virtual ~ProcessedContext() = default;

  // Gets the LoRA id.
  virtual std::optional<uint32_t> lora_id() const = 0;

  // Sets the LoRA id.
  virtual void set_lora_id(std::optional<uint32_t> lora_id) = 0;

  // Gets the processed tokens.
  virtual ProcessedTokens& processed_tokens() = 0;

 protected:
  ProcessedContext() = default;
  ProcessedContext(const ProcessedContext&) = default;
  ProcessedContext(ProcessedContext&&) noexcept = default;
  ProcessedContext& operator=(const ProcessedContext&) = default;
  ProcessedContext& operator=(ProcessedContext&&) noexcept = default;
};

// Context for the audio executor.
class AudioContext {
 public:
  virtual ~AudioContext() = default;

  // Clones the audio context for stateful models. Notice the operation deep
  // copy the underlying data, and expect it might be expensive.
  virtual absl::StatusOr<std::unique_ptr<AudioContext>> Clone() const = 0;

 protected:
  AudioContext() = default;
  AudioContext(const AudioContext&) = default;
  AudioContext(AudioContext&&) noexcept = default;
  AudioContext& operator=(const AudioContext&) = default;
  AudioContext& operator=(AudioContext&&) noexcept = default;
};

// Struct to host the internal state for the executor.
// State will be changed by the executor while executing task.
// Noticed: The states here are all the internal states excluded those that are
// directly related to the KVCache.
struct RuntimeState {
  // The current time step.
  int current_step = 0;

  // Random generator for sampling step.
  std::shared_ptr<std::default_random_engine> rand_gen;

  // Whether decode has been run ever after prefill.
  // This is only used by the compiled model executor to determine whether
  // KVCache preparation for prefill or decode should be done.
  bool ran_decode = false;
};

// A resource interface to hold the llm context.
class LlmContext {
 public:
  explicit LlmContext(std::unique_ptr<ProcessedContext> processed_context,
                      std::unique_ptr<RuntimeConfig> runtime_config,
                      std::unique_ptr<RuntimeState> runtime_state)
      : processed_context_(std::move(processed_context)),
        runtime_config_(std::move(runtime_config)),
        runtime_state_(std::move(runtime_state)) {};

  ~LlmContext() = default;

  // Gets the processed context.
  ProcessedContext& processed_context() { return *processed_context_; };

  // Gets the process state.
  RuntimeConfig& runtime_config() { return *runtime_config_; };

  // Gets the runtime state.
  RuntimeState& runtime_state() { return *runtime_state_; };

  // Gets the audio processed context.
  AudioContext* audio_processed_context() {
    return audio_processed_context_.get();
  };

  // Retrieves the processed context, the caller will take the ownership of the
  // returned processed context and it will no longer be available in the
  // LlmContext. This is useful for non-duplicating the processed context while
  // extracting it.
  absl::StatusOr<std::unique_ptr<ProcessedContext>> RetrieveProcessedContext() {
    return std::move(processed_context_);
  };

  // Retrieves the runtime state, the caller will take the ownership of the
  // returned runtime state and it will no longer be available in the
  // LlmContext. This is useful for non-duplicating the runtime state while
  // extracting it.
  absl::StatusOr<std::unique_ptr<RuntimeState>> RetrieveRuntimeState() {
    return std::move(runtime_state_);
  };

  // Retrieves the runtime config, the caller will take the ownership of the
  // returned runtime config and it will no longer be available in the
  // LlmContext. This is useful for non-duplicating the runtime config while
  // extracting it.
  absl::StatusOr<std::unique_ptr<RuntimeConfig>> RetrieveRuntimeConfig() {
    return std::move(runtime_config_);
  };

  // Retrieves the audio processed context, the caller will take the ownership
  // of the returned audio processed context and it will no longer be available
  // in the LlmContext. This is useful for non-duplicating the audio processed
  // context while extracting it.
  absl::StatusOr<std::unique_ptr<AudioContext>>
  RetrieveAudioProcessedContext() {
    return std::move(audio_processed_context_);
  };

 private:
  std::unique_ptr<ProcessedContext> processed_context_;
  std::unique_ptr<RuntimeConfig> runtime_config_;
  std::unique_ptr<RuntimeState> runtime_state_;
  std::unique_ptr<AudioContext> audio_processed_context_;

 protected:
  LlmContext() = default;
  LlmContext(LlmContext&&) noexcept = default;
  LlmContext& operator=(LlmContext&&) noexcept = default;
};

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

  // Special token for the end of image.
  static constexpr int kEndToken = -3;

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
  // valid_tokens: The number of valid tokens in the audio embeddings.
  ExecutorAudioData(
      std::optional<::litert::TensorBuffer>&& embeddings,
      std::optional<::litert::TensorBuffer>&& per_layer_embeddings,
      int valid_tokens = -1);

  // Getters:
  absl::StatusOr<const ::litert::TensorBuffer*> GetEmbeddingsPtr() const;
  absl::StatusOr<::litert::TensorBuffer*> GetMutableEmbeddingsPtr();
  absl::StatusOr<const ::litert::TensorBuffer*> GetPerLayerEmbeddingsPtr()
      const;
  absl::StatusOr<::litert::TensorBuffer*> GetMutablePerLayerEmbeddingsPtr();
  int GetValidTokens() const;

  // Setters:
  void SetEmbeddings(std::optional<::litert::TensorBuffer>&& embeddings);
  void SetPerLayerEmbeddings(
      std::optional<::litert::TensorBuffer>&& per_layer_embeddings);
  void SetValidTokens(int valid_tokens);

 private:
  std::optional<::litert::TensorBuffer> embeddings_;
  std::optional<::litert::TensorBuffer> per_layer_embeddings_;

  // The number of valid tokens in the audio embeddings. This is used to
  // determine the number of audio tokens to be actually used.
  // -1 means all the tokens are valid.
  // TODO: b/431028796 - Remove this field once tensorbuffer can be sliced
  // efficiently.
  int valid_tokens_ = -1;
};
std::ostream& operator<<(std::ostream& os, const ExecutorAudioData& audio_data);

class ExecutorInputs {
 public:
  ExecutorInputs() = default;

  ExecutorInputs(std::optional<ExecutorTextData>&& text_data,
                 std::optional<ExecutorVisionData>&& vision_data,
                 std::optional<ExecutorAudioData>&& audio_data);

  // Getters for top-level optional members
  int GetNextPosition() const;
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
  void SetNextPosition(int next_position);
  void SetTextData(ExecutorTextData&& text_data);
  void SetVisionData(std::optional<ExecutorVisionData>&& vision_data);
  void SetAudioData(std::optional<ExecutorAudioData>&& audio_data);

 private:
  // The next prefill/decode position to be used.
  int next_position_ = 0;
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
  ExecutorPrefillParams(
      int current_step, bool wait_for_completion,
      const std::atomic_bool* cancel,
      std::optional<int> max_prefill_sequence_length = std::nullopt);

  int GetCurrentStep() const;
  void SetCurrentStep(int current_step);

  bool GetWaitForCompletion() const;
  void SetWaitForCompletion(bool wait_for_completion);

  const std::atomic_bool* GetCancelFlag() const;
  void SetCancelFlag(const std::atomic_bool* cancel);

  absl::StatusOr<int> GetMaxPrefillSequenceLength() const;
  void SetMaxPrefillSequenceLength(
      std::optional<int> max_prefill_sequence_length);

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

  // Maximum sequence length of the prefill signatures allowed to be used for
  // this prefill call. Invoking a prefill signature with long sequence length
  // will result in a long waiting time. To ensure in-time cancellation, we may
  // need to limit the maximum sequence length used for prefill. If not set, all
  // prefill signatures are considered during prefill.
  std::optional<int> max_prefill_sequence_length_;
};
std::ostream& operator<<(std::ostream& os, const ExecutorPrefillParams& params);

// Class to host the parameters for Decode.
class ExecutorDecodeParams {
 public:
  ExecutorDecodeParams() = default;

  // Sets the constraint decoder. The caller retains ownership of the constraint
  // decoder and must ensure it outlives the ExecutorDecodeParams.
  void SetConstraintDecoder(ConstrainedDecoder* constraint);

  // Returns true if the constraint decoder is set.
  bool HasConstraintDecoder() const;

  // Returns the constraint decoder if it exists. Otherwise, returns nullptr.
  ConstrainedDecoder* GetConstraintDecoder() const;

 private:
  ConstrainedDecoder* absl_nullable constraint_decoder_ = nullptr;
};
std::ostream& operator<<(std::ostream& os, const ExecutorDecodeParams& params);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LLM_EXECUTOR_IO_TYPES_H_
