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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_QWEN_OMNI_QWEN_OMNI_AUDIO_STAGE_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_QWEN_OMNI_QWEN_OMNI_AUDIO_STAGE_H_

#include <memory>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "omni/any2any/any2any_whisper_audio_stage.h"
#include "omni/any2any/io_types.h"
#include "omni/any2any/qwen_omni/qwen_omni_io_types.h"
#include "omni/any2any/qwen_omni/qwen_omni_model_config.h"
#include "omni/base/fan_out_stage.h"
#include "omni/base/io_types.h"
#include "omni/base/litert_runner.h"
#include "omni/base/sequenced.h"
#include "omni/base/stage.h"
#include "omni/omni_session.h"

namespace litert::omni::any2any {

// Audio modality stage for Qwen2.5-Omni-3B:
// - Wraps `Any2AnyWhisperAudioStage` to consume `Sequenced<AudioInput>` and
//   `Sequenced<AudioInputMetadata>` from `Any2AnyInputFanOutStage`.
// - Resamples incoming PCM audio (defaulting to
//   `config.audio_input_sample_rate_hz`) to 16 kHz
//   (`qwen_omni::kAudioInputSampleRateHz`) and extracts 128-bin Whisper-style
//   log-mel spectrograms.
// - Runs the `tf_lite_audio_encoder` runner (`Qwen2_5OmniAudioEncoder` with
//   Conv1d stride-2 + avg-pool stride-2 = 4x temporal downsampling) in
//   200-frame (`2 * n_window`) chunks and frames the emitted embeddings into a
//   single `Sequenced<QwenOmniEmbeddingSegment>` per clip with `<|audio_bos|>`
//   (`151647`), `<|AUDIO|>` (`151646`), and `<|audio_eos|>` (`151648`).
class QwenOmniAudioStage
    : public SingleThreadedStageWithDeque<Sequenced<QwenOmniEmbeddingSegment>> {
 public:
  // Creates and initializes a `QwenOmniAudioStage` from a `LiteRtRunner`
  // executing `tf_lite_audio_encoder`.
  static absl::StatusOr<std::unique_ptr<QwenOmniAudioStage>> Create(
      FanOutOutputQueue<OmniSession::Input, AudioInput>* absl_nonnull
          audio_queue,
      FanOutOutputQueue<OmniSession::Input, AudioInputMetadata>* absl_nonnull
          audio_input_metadata_queue,
      std::unique_ptr<LiteRtRunner> absl_nonnull audio_encoder_runner,
      const QwenOmniModelConfig& config = {});

  ~QwenOmniAudioStage() override;

 protected:
  bool NeedScheduleInternal() const override;
  absl::Status ScheduleInternal() override;
  void ResetInternal() override;

 private:
  QwenOmniAudioStage(
      std::unique_ptr<Any2AnyWhisperAudioStage> absl_nonnull audio_stage,
      const QwenOmniModelConfig& config);

  QwenOmniEmbeddingSegment BuildAudioSegment(
      std::vector<EncodedAudioChunk> chunks) const;

  std::unique_ptr<Any2AnyWhisperAudioStage> audio_stage_;
  QwenOmniModelConfig config_;
};

}  // namespace litert::omni::any2any

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_QWEN_OMNI_QWEN_OMNI_AUDIO_STAGE_H_
