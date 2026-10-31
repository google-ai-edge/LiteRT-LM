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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_MINICPM_O_MINICPM_O_VOCODER_STAGE_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_MINICPM_O_MINICPM_O_VOCODER_STAGE_H_

#include <memory>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "omni/any2any/minicpm_o/minicpm_o_model_config.h"
#include "omni/any2any/minicpm_o/minicpm_o_tts_stage.h"
#include "omni/base/io_types.h"
#include "omni/base/litert_runner.h"
#include "omni/base/stage.h"

namespace litert::omni::any2any {
namespace minicpm_o {

// Maximum number of left-context mel frames overlapped and crossfaded between
// consecutive vocoder chunks to avoid chunk-boundary discontinuities.
inline constexpr int kVocoderOverlapFrames = 16;

}  // namespace minicpm_o

// Stage 4: Neural Mel-to-Waveform Vocoder & Output Stage for MiniCPM-o 4.5.
// Emits `TextOutput` for the generated assistant text and, when audio
// generation is enabled, runs `audio_vocoder.tflite` (the 24 kHz Vocos/HiFi-GAN
// mel-to-waveform vocoder of the CosyVoice2/Token2Wav TTS pipeline) with
// overlapping mel context and linear crossfading to synthesize 24 kHz
// `AudioOutput` PCM waveforms from the 100-bin log-mel spectrogram frames
// emitted by `MiniCpmOTtsStage`.
class MiniCpmOVocoderStage : public SingleThreadedStageWithDeque<Output> {
 public:
  static absl::StatusOr<std::unique_ptr<MiniCpmOVocoderStage>> Create(
      Stage<MiniCpmOTtsOutput>* absl_nonnull tts_stage,
      const MiniCpmOModelConfig& config,
      std::unique_ptr<LiteRtRunner> absl_nullable vocoder_runner = nullptr);

  ~MiniCpmOVocoderStage() override = default;

 protected:
  absl::Status FlushInternal() override;

  bool NeedScheduleInternal() const override { return tts_stage_.HasOutput(); }

  absl::Status ScheduleInternal() override;

 private:
  MiniCpmOVocoderStage(
      Stage<MiniCpmOTtsOutput>* absl_nonnull tts_stage,
      MiniCpmOModelConfig config,
      std::unique_ptr<LiteRtRunner> absl_nullable vocoder_runner)
      : tts_stage_(*tts_stage),
        config_(std::move(config)),
        vocoder_runner_(std::move(vocoder_runner)) {}

  absl::Status ProcessTtsOutput(const MiniCpmOTtsOutput& tts_out);

  absl::StatusOr<std::vector<float>> SynthesizeWaveform(
      const MiniCpmOTtsOutput& tts_out);

  Stage<MiniCpmOTtsOutput>& tts_stage_;
  MiniCpmOModelConfig config_;

  std::unique_ptr<LiteRtRunner> vocoder_runner_;
  std::vector<TensorBuffer> input_buffers_;
  std::vector<TensorBuffer> output_buffers_;

  int mel_chunk_frames_ = 100;
  int upsample_hop_ = minicpm_o::kVocoderHopSize;
};

}  // namespace litert::omni::any2any

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_MINICPM_O_MINICPM_O_VOCODER_STAGE_H_
