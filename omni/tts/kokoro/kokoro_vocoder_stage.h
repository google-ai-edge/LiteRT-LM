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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_KOKORO_KOKORO_VOCODER_STAGE_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_KOKORO_KOKORO_VOCODER_STAGE_H_

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "kiss_fftr.h"  // from @kissfft
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "omni/base/model_resources.h"
#include "omni/base/stage.h"
#include "omni/tts/kokoro/kokoro_io_types.h"
#include "omni/tts/vocoder.h"

namespace litert::omni::tts {

// Neural Vocoder and iSTFT Audio Synthesis for Kokoro-82M TTS.
// Converts acoustic predictor features (ASR representations, pitch, energy,
// style) into time-domain float PCM audio samples via neural spectrogram
// prediction and inverse Short-Time Fourier Transform (iSTFT) synthesis.
class KokoroVocoderStage : public Vocoder {
 public:
  // Creates a KokoroVocoderStage instance and initializes its resources.
  //
  // args
  // - acoustic_predictor: Stage providing input KokoroAcousticOutput data.
  // - resources: Shared ModelResources container with compiled models.
  //
  // returns
  // - Unique pointer to created KokoroVocoderStage on success, or error status
  //   on failure.
  static absl::StatusOr<std::unique_ptr<KokoroVocoderStage>> Create(
      Stage<KokoroAcousticOutput>* absl_nonnull acoustic_predictor,
      std::shared_ptr<ModelResources> absl_nonnull resources);

  ~KokoroVocoderStage() override;

  // Flushes remaining buffered audio frames and synthesizes audio.
  //
  // returns
  // - absl::OkStatus() on success, or error status on failure.
  absl::Status Flush() override;

 protected:
  bool NeedScheduleInternal() const override {
    return acoustic_predictor_.HasOutput();
  }

  // Executes one step of vocoder stage processing asynchronously.
  //
  // returns
  // - absl::OkStatus() on success, or error status on failure.
  absl::Status ScheduleInternal() override;

 private:
  struct InputIndices {
    size_t acoustic_features = 0;
    size_t pitch_contour = 1;
    size_t energy_contour = 2;
    size_t speaker_style = 3;
    size_t speech_frame_length = 4;
  };

  struct OutputIndices {
    size_t magnitude_spectrogram = 0;
    size_t phase_spectrogram = 1;
  };

  KokoroVocoderStage(Stage<KokoroAcousticOutput>* acoustic_predictor,
                     std::shared_ptr<ModelResources> resources)
      : acoustic_predictor_(*acoustic_predictor),
        resources_(std::move(resources)) {}

  // Synthesizes time-domain PCM audio from magnitude and phase spectrograms
  // using persistent KissFFT configuration and Hann overlap-add synthesis
  // window.
  absl::StatusOr<std::vector<float>> SynthesizeIstftAudio(
      absl::Span<const float> magnitude_spectrogram,
      absl::Span<const float> phase_spectrogram, int active_subframes);

  Stage<KokoroAcousticOutput>& acoustic_predictor_;
  std::shared_ptr<ModelResources> resources_;

  std::shared_ptr<CompiledModel> vocoder_model_;

  InputIndices input_indices_;
  OutputIndices output_indices_;

  std::vector<TensorBuffer> vocoder_input_buffers_;
  std::vector<TensorBuffer> vocoder_output_buffers_;

  // Pre-allocated KissFFT real-inverse FFT configuration plan.
  kiss_fftr_cfg fft_cfg_ = nullptr;
  // Precomputed inverse Hann synthesis window with overlap-add normalization.
  std::vector<float> inv_window_;
};

}  // namespace litert::omni::tts

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_KOKORO_KOKORO_VOCODER_STAGE_H_
