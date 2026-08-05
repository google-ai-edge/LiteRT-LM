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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_QWEN3_TTS_QWEN3_VOCODER_STAGE_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_QWEN3_TTS_QWEN3_VOCODER_STAGE_H_

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "omni/base/stage.h"
#include "omni/tts/latent_decoder.h"
#include "omni/tts/qwen3_tts/qwen3_stage_options.h"
#include "omni/tts/vocoder.h"

namespace litert::omni::tts {

// Stage 4: Neural Vocoder (Windowed Codec Audio Synthesis)
class Qwen3VocoderStage : public Vocoder {
 public:
  // Creates a Qwen3VocoderStage instance and initializes its resources.
  //
  // args
  // - latent_decoder: Stage providing input LatentOutput data.
  // - options: Options for configuring the Qwen3VocoderStage.
  // - env: Environment to use for compiled models.
  //
  // returns
  // - Unique pointer to created Qwen3VocoderStage on success, or error status
  //   on failure.
  static absl::StatusOr<std::unique_ptr<Qwen3VocoderStage>> Create(
      Stage<LatentOutput>* absl_nonnull latent_decoder,
      Qwen3StageOptions options, std::shared_ptr<Environment> absl_nonnull env);

  ~Qwen3VocoderStage() override = default;

  // Resets the stage state back to idle and clears any pending outputs.
  void Reset() override;

  // Flushes remaining buffered audio frames and synthesizes audio.
  //
  // returns
  // - absl::OkStatus() on success, or error status on failure.
  absl::Status Flush() override;

 protected:
  // Executes one step of vocoder stage processing asynchronously.
  //
  // returns
  // - absl::OkStatus() on success, or error status on failure.
  absl::Status ScheduleInternal() override;

 private:
  Qwen3VocoderStage(Stage<LatentOutput>* absl_nonnull latent_decoder,
                    Qwen3StageOptions options,
                    std::shared_ptr<Environment> absl_nonnull env)
      : Vocoder(latent_decoder),
        options_(std::move(options)),
        env_(std::move(env)) {}

  // Decodes discrete RVQ codebook frames into raw audio PCM waveform samples.
  //
  // args
  // - frames: Matrix of RVQ codebook frame token IDs.
  //
  // returns
  // - Audio PCM waveform float vector on success, or error status on failure.
  absl::StatusOr<std::vector<float>> DecodeCodes(
      const std::vector<std::vector<int>>& frames);

  Qwen3StageOptions options_;
  std::shared_ptr<Environment> env_;
  std::optional<CompiledModel> codec_model_;
  std::vector<TensorBuffer> input_buffers_;
  std::vector<TensorBuffer> output_buffers_;
  int codec_chunk_ = 100;
  int upsample_ = 1920;
  std::vector<std::vector<int>> pending_frames_;
};

}  // namespace litert::omni::tts

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_QWEN3_TTS_QWEN3_VOCODER_STAGE_H_
