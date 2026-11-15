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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_MINICPM_O_MINICPM_O_AUDIO_STAGE_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_MINICPM_O_MINICPM_O_AUDIO_STAGE_H_

#include <memory>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "omni/any2any/minicpm_o/minicpm_o_io_types.h"
#include "omni/any2any/minicpm_o/minicpm_o_model_config.h"
#include "omni/base/fan_out_stage.h"
#include "omni/base/io_types.h"
#include "omni/base/litert_runner.h"
#include "omni/base/sequenced.h"
#include "omni/base/stage.h"
#include "omni/omni_session.h"
#include "omni/utils/audio_resampler.h"
#include "omni/utils/whisper_log_mel_extractor.h"

namespace litert::omni::any2any {

// Audio input processing stage for MiniCPM-o:
// - Consumes `Sequenced<AudioInput>` and `Sequenced<AudioInputMetadata>` from
//   `MiniCpmOInputFanOutStage` in ascending `seqnum` order.
// - Updates the input sample rate and resets the resampler when
//   `AudioInputMetadata` is received.
// - Resamples mono PCM audio to 16 kHz via `AudioResampler`.
// - Computes 80-bin Whisper-compatible log-mel spectrogram features via
//   `WhisperLogMelExtractor`.
// - Runs the neural audio encoder model per chunk.
// - Outputs `Sequenced<MiniCpmOEmbeddingSegment>` containing the continuous
//   audio embeddings and `<|audio_start|>`, placeholder (`-2`), and
//   `<|audio_end|>` token IDs for downstream reordering and chat-template
//   framing in `MiniCpmOFrontendStage`.
class MiniCpmOAudioStage
    : public SingleThreadedStageWithDeque<Sequenced<MiniCpmOEmbeddingSegment>> {
 public:
  // Creates a `MiniCpmOAudioStage` that consumes sequenced audio and audio
  // metadata inputs in ascending `seqnum` order, computes Whisper-compatible
  // log-mel spectrogram features, runs `audio_encoder_runner`, and outputs
  // `Sequenced<MiniCpmOEmbeddingSegment>`.
  //
  // Args:
  //   audio_queue: Non-null output queue from `MiniCpmOInputFanOutStage`
  //     providing `Sequenced<AudioInput>` items.
  //   audio_input_metadata_queue: Non-null output queue from
  //     `MiniCpmOInputFanOutStage` providing `Sequenced<AudioInputMetadata>`
  //     items.
  //   audio_encoder_runner: Non-null runner executing the Whisper audio
  //     encoder and projector model. Owned exclusively by this stage and
  //     invoked only from this stage's single execution thread.
  //   config: Model configuration specifying `llm_hidden_size` and default
  //     `audio_input_sample_rate_hz`.
  //
  // Returns:
  //   A newly initialized `MiniCpmOAudioStage` on success, or an error status
  //   if `config.llm_hidden_size <= 0`, if
  //   `config.audio_input_sample_rate_hz <= 0`, if creating input/output
  //   buffers or the log-mel extractor fails, if any buffer list is empty, if
  //   input buffer byte size is not a positive multiple of
  //   `minicpm_o::kAudioMelBins * sizeof(float)`, or if output buffer byte size
  //   is not a positive multiple of `config.llm_hidden_size * sizeof(float)`.
  static absl::StatusOr<std::unique_ptr<MiniCpmOAudioStage>> Create(
      FanOutOutputQueue<OmniSession::Input, AudioInput>* absl_nonnull
          audio_queue,
      FanOutOutputQueue<OmniSession::Input, AudioInputMetadata>* absl_nonnull
          audio_input_metadata_queue,
      std::unique_ptr<LiteRtRunner> absl_nonnull audio_encoder_runner,
      const MiniCpmOModelConfig& config = {});

  ~MiniCpmOAudioStage() override;

 protected:
  bool NeedScheduleInternal() const override;
  absl::Status ScheduleInternal() override;
  void ResetInternal() override;

 private:
  MiniCpmOAudioStage(
      FanOutOutputQueue<OmniSession::Input, AudioInput>* absl_nonnull
          audio_queue,
      FanOutOutputQueue<OmniSession::Input, AudioInputMetadata>* absl_nonnull
          audio_input_metadata_queue,
      int chunk_frames,
      std::unique_ptr<WhisperLogMelExtractor> absl_nonnull log_mel_extractor,
      std::unique_ptr<LiteRtRunner> absl_nonnull audio_encoder_runner,
      std::vector<TensorBuffer> audio_enc_input_buffers,
      std::vector<TensorBuffer> audio_enc_output_buffers,
      const MiniCpmOModelConfig& config);

  // Processes raw audio PCM samples into continuous audio encoder embeddings
  // and `<|audio_start|>`, placeholder (`-2`), and `<|audio_end|>` token IDs.
  //
  // Args:
  //   audio: Input audio waveform containing mono PCM samples at
  //     `input_sample_rate_hz_`.
  //
  // Returns:
  //   A `MiniCpmOEmbeddingSegment` containing the continuous audio embeddings
  //   and unembedded audio boundary/placeholder token IDs, or an error status
  //   if resampling, preprocessing, or runner execution fails.
  absl::StatusOr<MiniCpmOEmbeddingSegment> ProcessAudio(
      const AudioInput& audio);

  FanOutOutputQueue<OmniSession::Input, AudioInput>& audio_queue_;
  FanOutOutputQueue<OmniSession::Input, AudioInputMetadata>&
      audio_input_metadata_queue_;
  MiniCpmOModelConfig config_;
  int hidden_dim_;
  int chunk_frames_;
  int input_sample_rate_hz_;

  AudioResampler resampler_;
  std::unique_ptr<WhisperLogMelExtractor> log_mel_extractor_;
  std::unique_ptr<LiteRtRunner> audio_encoder_runner_;
  std::vector<TensorBuffer> audio_enc_input_buffers_;
  std::vector<TensorBuffer> audio_enc_output_buffers_;
};

}  // namespace litert::omni::any2any

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_MINICPM_O_MINICPM_O_AUDIO_STAGE_H_
