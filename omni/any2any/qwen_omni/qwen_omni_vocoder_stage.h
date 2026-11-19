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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_QWEN_OMNI_QWEN_OMNI_VOCODER_STAGE_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_QWEN_OMNI_QWEN_OMNI_VOCODER_STAGE_H_

#include <memory>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "omni/any2any/any2any_vocoder_stage.h"
#include "omni/any2any/io_types.h"
#include "omni/any2any/qwen_omni/qwen_omni_model_config.h"
#include "omni/base/io_types.h"
#include "omni/base/litert_runner.h"
#include "omni/base/stage.h"

namespace litert::omni::any2any {

// Stage 4: Neural Speech-Codec-to-Waveform Vocoder & Output Stage for
// Qwen2.5-Omni-3B Token2Wav.
// Wraps `Any2AnyVocoderStage` to forward assistant `TextOutput`, validate
// discrete 50 Hz speech codec token IDs against the codebook range, and run
// `audio_vocoder.tflite` over edge-clamped sliding windows (without seam
// crossfade, supporting inferred output layouts) to synthesize 24 kHz
// `AudioOutput` PCM waveforms and `EndOfOutputTurn`.
class QwenOmniVocoderStage : public SingleThreadedStageWithDeque<Output> {
 public:
  // Creates and initializes a `QwenOmniVocoderStage` with the given upstream
  // TTS stage, model configuration, and optional vocoder runner.
  //
  // Args:
  //   tts_stage: Non-null upstream TTS stage producing assistant text and
  //     discrete 50 Hz speech codec tokens.
  //   config: Qwen-Omni model configuration specifying vocoder token window
  //     dimensions, codebook upper bound, and output sample rate.
  //   vocoder_runner: Runner for `tf_lite_audio_vocoder` (required when
  //     `config.generate_audio` is true).
  //
  // Returns:
  //   A newly initialized `QwenOmniVocoderStage` instance, or an `absl::Status`
  //   error if configuration or buffer validation fails.
  static absl::StatusOr<std::unique_ptr<QwenOmniVocoderStage>> Create(
      Stage<Any2AnyTtsOutput>* absl_nonnull tts_stage,
      const QwenOmniModelConfig& config,
      std::unique_ptr<LiteRtRunner> absl_nullable vocoder_runner = nullptr);

  ~QwenOmniVocoderStage() override = default;

 protected:
  absl::Status FlushInternal() override;

  bool NeedScheduleInternal() const override;

  void ResetInternal() override;

  absl::Status ScheduleInternal() override;

 private:
  explicit QwenOmniVocoderStage(
      std::unique_ptr<Any2AnyVocoderStage> absl_nonnull vocoder_stage);

  std::unique_ptr<Any2AnyVocoderStage> vocoder_stage_;
};

}  // namespace litert::omni::any2any

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_QWEN_OMNI_QWEN_OMNI_VOCODER_STAGE_H_
