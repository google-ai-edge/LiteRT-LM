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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_AUDIO_EXECUTOR_BASE_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_AUDIO_EXECUTOR_BASE_H_

#include <cstdint>
#include <memory>
#include <optional>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/engine/io_types.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/llm_executor_io_types.h"

namespace litert::lm {

class AudioExecutorBase {
 public:
  virtual ~AudioExecutorBase() = default;

  // ------------Encode APIs------------:
  // Basic API to trigger the "encode" process.
  // Input is audio spectrogram tensor with shape `[batch, 1, frame,
  // num_channels]`. Output is audio data which contains main embeddings with
  // shape `[batch, 1, num_audio_tokens, model_dimension]`.
  virtual absl::StatusOr<::litert::lm::ExecutorAudioData> Encode(
      const litert::TensorBuffer& spectrogram_tensor) = 0;

  // Reset the audio executor to its initial state. It must be called for
  // streaming audio models after finishing an audio stream.
  virtual absl::Status Reset() {
    return absl::UnimplementedError("Not implemented.");
  }

  // Flush any buffered spectrogram frames from intermediate streaming Encode()
  // calls. This should be called when the audio stream ends (e.g., when
  // InputAudioEnd is encountered) to process remaining buffered frames with
  // zero-padding and produce the final audio embeddings. Returns an empty
  // ExecutorAudioData with 0 valid tokens if no frames are buffered.
  virtual absl::StatusOr<::litert::lm::ExecutorAudioData> Flush() {
    return absl::UnimplementedError("Not implemented.");
  }

  // Create a new audio context for the audio executor.
  virtual absl::StatusOr<std::unique_ptr<AudioContext>> CreateNewContext() {
    return absl::UnimplementedError("Not implemented.");
  };

  // Clone the audio context for the audio executor.
  virtual absl::StatusOr<std::unique_ptr<AudioContext>> CloneContext() {
    return absl::UnimplementedError("Not implemented.");
  }

  // Clone the audio context from the given audio context.
  virtual absl::StatusOr<std::unique_ptr<AudioContext>> CloneContext(
      const AudioContext& audio_context) {
    return absl::UnimplementedError("Not implemented.");
  }

  // Restore the audio context for the audio executor.
  virtual absl::Status RestoreContext(
      std::unique_ptr<AudioContext> audio_context) {
    return absl::UnimplementedError("Not implemented.");
  }

  // Get the audio executor properties.
  virtual absl::StatusOr<AudioExecutorProperties> GetAudioExecutorProperties()
      const {
    return absl::UnimplementedError("Not implemented.");
  }

  // Loads the LoRA model into the audio executor.
  virtual absl::Status LoadLoRA(uint32_t lora_id,
                                const ModelAssets& model_assets) {
    return absl::UnimplementedError("Not implemented.");
  }

  // Sets the current LoRA ID to use.
  virtual absl::Status UseLoRA(std::optional<uint32_t> lora_id) {
    return absl::UnimplementedError("Not implemented.");
  }
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_AUDIO_EXECUTOR_BASE_H_
