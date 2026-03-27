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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_AUDIO_LITERT_COMPILED_MODEL_EXECUTOR_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_AUDIO_LITERT_COMPILED_MODEL_EXECUTOR_H_

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/audio_executor.h"
#include "runtime/executor/audio_executor_settings.h"
#include "runtime/executor/llm_executor_io_types.h"

namespace litert::lm {

// The context for streaming audio encoder model, which contains
// the state buffers of the audio encoder model.
class AudioStreamingContext : public AudioContext {
 public:
  explicit AudioStreamingContext(
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
          state_buffers)
      : state_buffers_(std::move(state_buffers)) {};

  absl::StatusOr<std::unique_ptr<AudioContext>> Clone() const override;

  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
  state_buffers() {
    return state_buffers_;
  }

 private:
  // The state buffers of the audio encoder model. It includes the kv caches and
  // the convolution features and masks of the last timestamp.
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer> state_buffers_;
};

// The Audio Executor that uses the LiteRT CompiledModel to run the audio
// encoder and audio adapter models to encode the spectrogram tensor into audio
// soft token embeddings.
class AudioLiteRtCompiledModelExecutor : public AudioExecutor {
 public:
  // Create an AudioLiteRtCompiledModelExecutor to encode the spectrogram
  // LiteRT TensorBuffer into audio embeddings LiteRT TensorBuffer.
  // Args:
  //   - executor_settings: The audio executor settings.
  //   - env: The LiteRT environment.
  // Returns:
  //   A unique pointer to the AudioLiteRtCompiledModelExecutor if successful,
  //   or an error status if failed.
  static absl::StatusOr<std::unique_ptr<AudioLiteRtCompiledModelExecutor>>
  Create(AudioExecutorSettings executor_settings, Environment& env);

  // Run the audio encoder and audio adapter models to encode the spectrogram
  // tensor into audio embeddings. It is caller's responsibility to ensure the
  // spectrogram tensor is valid and has the correct shape. It is assumed that
  // all the timestamps in the spectrogram tensor are valid.
  // Args:
  //   - spectrogram_tensor: The spectrogram tensor to encode, in shape of
  //     [..., timestamp, frequency_bins].
  // Returns:
  //   A ExecutorAudioData object containing the audio embeddings and the
  //   number of valid tokens.
  absl::StatusOr<ExecutorAudioData> Encode(
      const TensorBuffer& spectrogram_tensor) override;

  // Run the audio encoder and audio adapter models to encode the spectrogram
  // tensor into audio embeddings. It is caller's responsibility to ensure the
  // spectrogram tensor is valid and has the correct shape.
  // The spectrogram mask is used to indicate the valid timestamps in the
  // spectrogram tensor.
  // Args:
  //   - spectrogram_tensor: The spectrogram tensor to encode, in shape of
  //     [..., timestamp, frequency_bins].
  //   - spectrogram_mask: The spectrogram mask to indicate the valid timestamps
  //     in the spectrogram tensor, in shape of [..., timestamp].
  // Returns:
  //   A ExecutorAudioData object containing the audio embeddings and the
  //   number of valid tokens.
  absl::StatusOr<ExecutorAudioData> Encode(
      const TensorBuffer& spectrogram_tensor,
      const TensorBuffer& spectrogram_mask);

  // Reset the audio encoder, which will be a stateful object when streaming
  // model is used.
  absl::Status Reset() override { return audio_encoder_->Reset(); }

  // Get the audio executor properties.
  absl::StatusOr<AudioExecutorProperties> GetAudioExecutorProperties()
      const override {
    return executor_properties_;
  }

  // Create a new audio context for the audio executor.
  absl::StatusOr<std::unique_ptr<AudioContext>> CreateNewContext() override;

  // Clone the audio context for the audio executor.
  absl::StatusOr<std::unique_ptr<AudioContext>> CloneContext() override;

  // Clone the audio context from the given audio context.
  absl::StatusOr<std::unique_ptr<AudioContext>> CloneContext(
      const AudioContext& audio_context) override;

  // Restore the audio context for the audio executor.
  absl::Status RestoreContext(
      std::unique_ptr<AudioContext> audio_context) override;

 private:
  // The Audio Encoder LiteRT CompiledModel wrapper manage the input and
  // output buffers of the audio encoder model. It is not expected to be used
  // directly by the user. It is used by the AudioLiteRtCompiledModelExecutor
  // to encode the spectrogram tensor into audio embeddings. The user should
  // use the AudioLiteRtCompiledModelExecutor instead.
  class AudioEncoder {
   public:
    virtual ~AudioEncoder() = default;

    virtual absl::Status Initialize() = 0;

    virtual absl::Status ClearInputBuffers() = 0;

    virtual absl::Status Reset() = 0;

    const CompiledModel& GetCompiledModel() const { return compiled_model_; }

    CompiledModel& GetMutableCompiledModel() { return compiled_model_; }

    const absl::flat_hash_map<absl::string_view, TensorBuffer>&
    GetInputBuffersMap() const {
      return input_buffers_map_;
    }

    absl::flat_hash_map<absl::string_view, TensorBuffer>&
    GetMutableInputBuffersMap() {
      return input_buffers_map_;
    }

    const absl::flat_hash_map<absl::string_view, TensorBuffer>&
    GetOutputBuffersMap() const {
      return output_buffers_map_;
    }

    absl::flat_hash_map<absl::string_view, TensorBuffer>&
    GetMutableOutputBuffersMap() {
      return output_buffers_map_;
    }

    const TensorBuffer& GetInputMaskBuffer() const {
      return *input_mask_buffer_;
    }

    TensorBuffer& GetMutableInputMaskBuffer() { return *input_mask_buffer_; }

    const TensorBuffer& GetInputSpectrogramBuffer() const {
      return *spectrogram_buffer_;
    }

    TensorBuffer& GetMutableInputSpectrogramBuffer() {
      return *spectrogram_buffer_;
    }

    const TensorBuffer& GetOutputMaskBuffer() const {
      return *output_mask_buffer_;
    }

    TensorBuffer& GetMutableOutputMaskBuffer() { return *output_mask_buffer_; }

    const TensorBuffer& GetOutputFeaturesBuffer() const {
      return *output_features_buffer_;
    }

    TensorBuffer& GetMutableOutputFeaturesBuffer() {
      return *output_features_buffer_;
    }

   protected:
    CompiledModel compiled_model_;

    // The input buffer for the spectrogram mask.
    TensorBuffer* input_mask_buffer_;
    // The input buffer for the spectrogram tensor.
    TensorBuffer* spectrogram_buffer_;
    // The output buffer for the valid tokens mask.
    TensorBuffer* output_mask_buffer_;
    // The output buffer for the features.
    TensorBuffer* output_features_buffer_;

    // The input names for the audio encoder model.
    std::vector<std::string> input_names_;

    // The output names for the audio encoder model.
    std::vector<std::string> output_names_;

    // The input buffers map for the audio encoder model.
    absl::flat_hash_map<absl::string_view, TensorBuffer> input_buffers_map_;
    // The output buffers map for the audio encoder model.
    absl::flat_hash_map<absl::string_view, TensorBuffer> output_buffers_map_;
  };

  // Audio Encoder for static LiteRT model, where the whole audio is provided at
  // once.
  class AudioStaticEncoder : public AudioEncoder {
   public:
    // Create an AudioStaticEncoder to run audio static encoder LiteRT
    // CompiledModel.
    // Args:
    //   - env: The LiteRT environment.
    //   - model: The audio encoder model.
    // Returns:
    //   A unique pointer to the AudioStaticEncoder if successful, or an error
    //   status if failed.
    static absl::StatusOr<std::unique_ptr<AudioStaticEncoder>> Create(
        const AudioExecutorSettings& executor_settings, Environment& env,
        const Model* absl_nonnull model);

    // Initialize the AudioStaticEncoder, which will create the input and output
    // buffers for the audio encoder model.
    absl::Status Initialize() override;

    absl::Status ClearInputBuffers() override;

    absl::Status Reset() override { return ClearInputBuffers(); }

   private:
    AudioStaticEncoder(const AudioExecutorSettings& executor_settings,
                       Environment& env, const Model* absl_nonnull model)
        : executor_settings_(executor_settings), env_(env), model_(*model) {}

    const AudioExecutorSettings& executor_settings_;
    Environment& env_;
    const Model& model_;
  };

  // Audio Encoder for streaming LiteRT model, where the audio is provided in
  // streaming fashion.
  //
  // For streaming audio encoder model, the input buffers map contains two
  // parts:
  // 1. The inputs from the new audio segment. It includes
  //  - segment_values: The spectrogram segment.
  //  - segment_mask: The spectrogram mask.
  // 2. The inputs from the internal state. It includes
  //  - prev_features: The previous features.
  //  - prev_mask: The previous mask.
  //  - prev_conv_out_mask: The previous conv out mask.
  //  and for each transformer layer (12 layers for gemma3n):
  //    - prev_q_{layer_idx}: The previous q tensor.
  //    - prev_k_{layer_idx}: The previous k tensor.
  //    - prev_v_{layer_idx}: The previous v tensor.
  //    - conv_padding_{layer_idx}: The conv padding.
  //  and for each subsample layer (2 layers for gemma3n):
  //  - feature_states_{layer_idx}: The feature states.
  //
  // For streaming audio encoder model, the output buffers map contains two
  // parts:
  // 1. The outputs from the new audio segment. It includes
  //  - features: The features.
  //  - mask: The valid tokens mask.
  // 2. The outputs from the internal state, and are used for next round of
  //    input. It includes
  //  - prev_features: The previous features.
  //  - prev_mask: The previous mask.
  //  - prev_conv_out_mask: The previous conv out mask.
  //  and for each transformer layer (12 layers for gemma3n):
  //    - prev_q_{layer_idx}: The previous q tensor.
  //    - prev_k_{layer_idx}: The previous k tensor.
  //    - prev_v_{layer_idx}: The previous v tensor.
  //    - conv_padding_{layer_idx}: The conv padding.
  //  and for each subsample layer (2 layers for gemma3n):
  //  -
  class AudioStreamingEncoder : public AudioEncoder {
   public:
    // Create an AudioStreamingEncoder to run audio streaming encoder LiteRT
    // CompiledModel.
    // Args:
    //   - env: The LiteRT environment.
    //   - model: The audio encoder model.
    // Returns:
    //   A unique pointer to the AudioStreamingEncoder if successful, or an
    //   error status if failed.
    static absl::StatusOr<std::unique_ptr<AudioStreamingEncoder>> Create(
        const AudioExecutorSettings& executor_settings, Environment& env,
        const Model* absl_nonnull model);

    // Initialize the AudioStreamingEncoder, which will create the input and
    // output buffers for the audio encoder model.
    absl::Status Initialize() override;

    int GetOverlapSize() const { return overlap_size_; }

    // Swap the internal state buffers between input and output buffers map, so
    // the previous state will be used for the current state.
    absl::Status SwapInternalStateBuffers();

    absl::Status ClearInputBuffers() override;

    absl::Status Reset() override;

    absl::StatusOr<std::unique_ptr<AudioStreamingContext>> CreateNewContext();

    absl::StatusOr<std::unique_ptr<AudioStreamingContext>> CloneContext();

    absl::Status RestoreContext(
        std::unique_ptr<AudioStreamingContext> audio_streaming_context);

   private:
    AudioStreamingEncoder(const AudioExecutorSettings& executor_settings,
                          Environment& env, const Model* absl_nonnull model)
        : executor_settings_(executor_settings), env_(env), model_(*model) {}

    const AudioExecutorSettings& executor_settings_;
    Environment& env_;
    const Model& model_;
    int overlap_size_;
  };

  // The Audio Adapter LiteRT CompiledModel wrapper manage the input and
  // output buffers of the audio adapter model. It is not expected to be used
  // directly by the user. It is used by the AudioLiteRtCompiledModelExecutor to
  // encode the audio embeddings into audio soft tokens. The user should use the
  // AudioLiteRtCompiledModelExecutor instead.
  class AudioAdapter {
   public:
    // Create an AudioAdapter to run audio adapter LiteRT CompiledModel.
    // Args:
    //   - env: The LiteRT environment.
    //   - model: The audio adapter model.
    // Returns:
    //   A unique pointer to the AudioAdapter if successful, or an error status
    //   if failed.
    static absl::StatusOr<std::unique_ptr<AudioAdapter>> Create(
        const AudioExecutorSettings& executor_settings, Environment& env,
        const Model* absl_nonnull model);

    // Initialize the AudioAdapter, which will create the input and output
    // buffers for the audio adapter model.
    absl::Status Initialize();

    const CompiledModel& GetCompiledModel() const { return compiled_model_; }

    CompiledModel& GetMutableCompiledModel() { return compiled_model_; }

    const std::vector<TensorBuffer>& GetInputBuffers() const {
      return input_buffers_;
    }

    std::vector<TensorBuffer>& GetMutableInputBuffers() {
      return input_buffers_;
    }

    const TensorBuffer& GetFeaturesBuffer() const { return *features_buffer_; }

    TensorBuffer& GetMutableFeaturesBuffer() { return *features_buffer_; }

    const TensorBuffer& GetMaskBuffer() const { return *mask_buffer_; }

    TensorBuffer& GetMutableMaskBuffer() { return *mask_buffer_; }

    const std::vector<TensorBuffer>& GetOutputBuffers() const {
      return output_buffers_;
    }
    std::vector<TensorBuffer>& GetMutableOutputBuffers() {
      return output_buffers_;
    }

   private:
    AudioAdapter(const AudioExecutorSettings& executor_settings,
                 Environment& env, const Model* absl_nonnull model)
        : executor_settings_(executor_settings), env_(env), model_(*model) {}

    const AudioExecutorSettings& executor_settings_;
    Environment& env_;
    const Model& model_;
    CompiledModel compiled_model_;
    // The input buffers for the audio adapter model.
    std::vector<TensorBuffer> input_buffers_;
    // The input buffers for the input features.
    TensorBuffer* features_buffer_;
    // The input buffer for the input mask.
    TensorBuffer* mask_buffer_;
    // The output buffers for the audio adapter model.
    std::vector<TensorBuffer> output_buffers_;
  };

  explicit AudioLiteRtCompiledModelExecutor(
      AudioExecutorSettings executor_settings,
      AudioExecutorProperties executor_properties, Environment& env,
      std::unique_ptr<ModelResources> resources,
      std::unique_ptr<AudioEncoder> audio_encoder,
      std::unique_ptr<AudioAdapter> audio_adapter, int sequence_length,
      int spectrogram_feature_dimensions, int audio_embedding_dimensions,
      int encoder_shrinking_factor)
      : sequence_length_(sequence_length),
        spectrogram_feature_dimensions_(spectrogram_feature_dimensions),
        audio_embedding_dimensions_(audio_embedding_dimensions),
        encoder_shrinking_factor_(encoder_shrinking_factor),
        executor_settings_(std::move(executor_settings)),
        executor_properties_(std::move(executor_properties)),
        env_(env),
        resources_(std::move(resources)),
        audio_encoder_(std::move(audio_encoder)),
        audio_adapter_(std::move(audio_adapter)) {}

  // Run the audio encoder and audio adapter models to encode the spectrogram
  // tensor into audio embeddings.
  // Args:
  //   - spectrogram_tensor: The spectrogram tensor buffer to encode.
  //   - spectrogram_mask: The spectrogram mask buffer to indicate the valid
  //   timestamps.
  //   - audio_embeddings: The output buffer for the audio embeddings to write
  //   into.
  // Returns:
  //   The number of valid tokens in the audio embeddings.
  absl::StatusOr<int> EncodeInternal(absl::Span<float> spectrogram_tensor,
                                     absl::Span<uint8_t> spectrogram_mask,
                                     absl::Span<float> audio_embeddings);
  int sequence_length_;
  int spectrogram_feature_dimensions_;
  int audio_embedding_dimensions_;
  int encoder_shrinking_factor_;
  AudioExecutorSettings executor_settings_;
  AudioExecutorProperties executor_properties_;
  /// The LiteRT environment.
  Environment& env_;
  std::unique_ptr<ModelResources> resources_;
  std::unique_ptr<AudioEncoder> audio_encoder_;
  std::unique_ptr<AudioAdapter> audio_adapter_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_AUDIO_LITERT_COMPILED_MODEL_EXECUTOR_H_
