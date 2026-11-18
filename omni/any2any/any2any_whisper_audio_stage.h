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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_ANY2ANY_WHISPER_AUDIO_STAGE_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_ANY2ANY_WHISPER_AUDIO_STAGE_H_

#include <functional>
#include <memory>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "omni/any2any/io_types.h"
#include "omni/base/fan_out_stage.h"
#include "omni/base/io_types.h"
#include "omni/base/litert_runner.h"
#include "omni/base/sequenced.h"
#include "omni/base/stage.h"
#include "omni/omni_session.h"
#include "omni/utils/audio_resampler.h"
#include "omni/utils/whisper_log_mel_extractor.h"

namespace litert::omni::any2any {

// Concrete Whisper-style audio encoder stage shared across Any2Any model
// families:
// - Consumes `Sequenced<AudioInput>` and `Sequenced<AudioInputMetadata>` in
//   ascending `seqnum` order.
// - Validates metadata, updates the active input sample rate, and resets both
//   the `AudioResampler` and `WhisperLogMelExtractor`.
// - Resamples mono PCM audio to `target_sample_rate_hz` (skipping resampling
//   when the active input rate already matches), extracts Whisper-style
//   log-mel spectrograms, slices into fixed `chunk_frames` windows, and runs
//   `audio_encoder_runner`.
// - Outputs `Sequenced<std::vector<EncodedAudioChunk>>` (empty vector for
//   metadata events and empty `AudioInput` items). Model-specific audio stages
//   wrap this stage to frame the emitted chunk embeddings into their
//   model-specific embedding segments.
class Any2AnyWhisperAudioStage
    : public SingleThreadedStageWithDeque<
          Sequenced<std::vector<EncodedAudioChunk>>> {
 public:
  // Configuration for `Any2AnyWhisperAudioStage`, shared across Any2Any model
  // families (e.g., MiniCPM-o, Qwen2.5-Omni).
  struct Options {
    // Hidden dimension of the downstream LLM embedding space.
    int llm_hidden_size = 0;
    // Initial input PCM sample rate in Hz (restored on `Reset()`).
    int initial_input_sample_rate_hz = kDefaultInputSampleRateHz;
    // Target sample rate in Hz expected by the Whisper log-mel extractor.
    int target_sample_rate_hz = kDefaultInputSampleRateHz;
    // Number of mel filterbank bins expected by the audio encoder (e.g. 80 or
    // 128).
    int num_mel_bins = 80;
    // Optional callback computing the number of valid output tokens for a chunk
    // containing `valid_frames` real mel frames out of `chunk_frames`. Return
    // values are clamped to `[1, max_chunk_tokens]`. When unset, defaults to
    // `DefaultComputeValidAudioTokens()`.
    std::function<int(int valid_frames, int chunk_frames, int max_chunk_tokens)>
        compute_valid_tokens;
  };

  static absl::StatusOr<std::unique_ptr<Any2AnyWhisperAudioStage>> Create(
      FanOutOutputQueue<OmniSession::Input, AudioInput>* absl_nonnull
          audio_queue,
      FanOutOutputQueue<OmniSession::Input, AudioInputMetadata>* absl_nonnull
          audio_input_metadata_queue,
      std::unique_ptr<LiteRtRunner> absl_nonnull audio_encoder_runner,
      Options options);

  ~Any2AnyWhisperAudioStage() override;

  // Default proportional ceiling downsampling formula mapping `valid_frames`
  // out of `chunk_frames` to `[1, max_chunk_tokens]`.
  static int DefaultComputeValidAudioTokens(int valid_frames, int chunk_frames,
                                            int max_chunk_tokens);

 protected:
  bool NeedScheduleInternal() const override;
  absl::Status ScheduleInternal() override;
  void ResetInternal() override;

 private:
  Any2AnyWhisperAudioStage(
      FanOutOutputQueue<OmniSession::Input, AudioInput>* absl_nonnull
          audio_queue,
      FanOutOutputQueue<OmniSession::Input, AudioInputMetadata>* absl_nonnull
          audio_input_metadata_queue,
      int chunk_frames, int max_chunk_tokens,
      std::unique_ptr<WhisperLogMelExtractor> absl_nonnull log_mel_extractor,
      std::unique_ptr<LiteRtRunner> absl_nonnull audio_encoder_runner,
      std::vector<TensorBuffer> audio_enc_input_buffers,
      std::vector<TensorBuffer> audio_enc_output_buffers, Options options);

  absl::StatusOr<std::vector<EncodedAudioChunk>> ProcessAudio(
      const AudioInput& audio);

  FanOutOutputQueue<OmniSession::Input, AudioInput>& audio_queue_;
  FanOutOutputQueue<OmniSession::Input, AudioInputMetadata>&
      audio_input_metadata_queue_;
  Options options_;
  int chunk_frames_;
  int max_chunk_tokens_;
  int input_sample_rate_hz_;

  AudioResampler resampler_;
  std::unique_ptr<WhisperLogMelExtractor> log_mel_extractor_;
  std::unique_ptr<LiteRtRunner> audio_encoder_runner_;
  std::vector<TensorBuffer> audio_enc_input_buffers_;
  std::vector<TensorBuffer> audio_enc_output_buffers_;
};

}  // namespace litert::omni::any2any

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_ANY2ANY_WHISPER_AUDIO_STAGE_H_
