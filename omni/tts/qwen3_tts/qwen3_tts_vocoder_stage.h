// Copyright 2026 The ODML Authors.
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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_QWEN3_TTS_QWEN3_TTS_VOCODER_STAGE_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_QWEN3_TTS_QWEN3_TTS_VOCODER_STAGE_H_

#include <memory>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "omni/base/model_resources.h"
#include "omni/base/stage.h"
#include "omni/tts/qwen3_tts/qwen3_tts_io_types.h"
#include "omni/tts/qwen3_tts/qwen3_tts_model_config.h"
#include "omni/tts/vocoder.h"

namespace litert::omni::tts {

// Stage 4: Neural Vocoder (Windowed Codec Audio Synthesis)
class Qwen3TtsVocoderStage : public Vocoder {
 public:
  // Creates a Qwen3TtsVocoderStage instance and initializes its resources.
  //
  // args
  // - latent_decoder: Stage providing input Qwen3TtsLatentOutput data.
  // - config: Configuration for the Qwen3TtsVocoderStage.
  // - resources: Shared ModelResources container with compiled models.
  //
  // returns
  // - Unique pointer to created Qwen3TtsVocoderStage on success, or error
  // status
  //   on failure.
  static absl::StatusOr<std::unique_ptr<Qwen3TtsVocoderStage>> Create(
      Stage<Qwen3TtsLatentOutput>* absl_nonnull latent_decoder,
      const Qwen3TtsModelConfig& config,
      std::shared_ptr<ModelResources> absl_nonnull resources);

  ~Qwen3TtsVocoderStage() override = default;

  // Flushes remaining buffered audio frames and synthesizes audio.
  //
  // returns
  // - absl::OkStatus() on success, or error status on failure.
  absl::Status Flush() override;

 protected:
  void ResetInternal() override;

  bool NeedScheduleInternal() const override {
    return latent_decoder_.HasOutput();
  }

  // Executes one step of vocoder stage processing asynchronously.
  //
  // returns
  // - absl::OkStatus() on success, or error status on failure.
  absl::Status ScheduleInternal() override;

 private:
  Qwen3TtsVocoderStage(Stage<Qwen3TtsLatentOutput>* absl_nonnull latent_decoder,
                       Qwen3TtsModelConfig config,
                       std::shared_ptr<ModelResources> absl_nonnull resources)
      : latent_decoder_(*latent_decoder),
        config_(std::move(config)),
        resources_(std::move(resources)) {}

  Stage<Qwen3TtsLatentOutput>& latent_decoder_;

  // Decodes a fixed chunk of discrete RVQ codebook frames into raw audio PCM
  // waveform samples.
  //
  // args
  // - window_frames: Matrix of RVQ codebook frame token IDs for the window.
  // - c: Overlap context offset (in frames).
  // - valid_frames_count: Number of valid frames in the window to output audio
  //   for.
  //
  // returns
  // - Audio PCM waveform float vector on success, or error status on failure.
  absl::StatusOr<std::vector<float>> DecodeChunk(
      const std::vector<std::vector<int>>& window_frames, int c,
      int valid_frames_count);

  // Processes buffered pending frames into decoded audio chunks.
  // If flush_remaining is true, also decodes any remaining sub-chunk frames.
  absl::Status ProcessPendingChunks(bool flush_remaining);

  Qwen3TtsModelConfig config_;
  std::shared_ptr<ModelResources> resources_;
  std::shared_ptr<CompiledModel> codec_model_;
  std::vector<TensorBuffer> input_buffers_;
  std::vector<TensorBuffer> output_buffers_;
  int codec_chunk_ = 100;
  int upsample_ = 1920;
  std::vector<std::vector<int>> pending_frames_;
  bool is_first_chunk_ = true;
};

}  // namespace litert::omni::tts

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_QWEN3_TTS_QWEN3_TTS_VOCODER_STAGE_H_
