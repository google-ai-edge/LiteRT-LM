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

#include "omni/any2any/any2any_whisper_audio_stage.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_macros.h"  // from @litert
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
namespace {

constexpr int kWhisperFrameLength = 400;
constexpr int kWhisperHopLength = 160;
constexpr float kWhisperMelFloor = 1e-10f;

}  // namespace

int Any2AnyWhisperAudioStage::DefaultComputeValidAudioTokens(
    int valid_frames, int chunk_frames, int max_chunk_tokens) {
  return std::clamp(
      (valid_frames * max_chunk_tokens + chunk_frames - 1) / chunk_frames, 1,
      max_chunk_tokens);
}

Any2AnyWhisperAudioStage::Any2AnyWhisperAudioStage(
    FanOutOutputQueue<OmniSession::Input, AudioInput>* absl_nonnull audio_queue,
    FanOutOutputQueue<OmniSession::Input, AudioInputMetadata>* absl_nonnull
        audio_input_metadata_queue,
    int chunk_frames, int max_chunk_tokens,
    std::unique_ptr<WhisperLogMelExtractor> absl_nonnull log_mel_extractor,
    std::unique_ptr<LiteRtRunner> absl_nonnull audio_encoder_runner,
    std::vector<TensorBuffer> audio_enc_input_buffers,
    std::vector<TensorBuffer> audio_enc_output_buffers, Options options)
    : audio_queue_(*audio_queue),
      audio_input_metadata_queue_(*audio_input_metadata_queue),
      options_(std::move(options)),
      chunk_frames_(chunk_frames),
      max_chunk_tokens_(max_chunk_tokens),
      input_sample_rate_hz_(options_.initial_input_sample_rate_hz),
      log_mel_extractor_(std::move(log_mel_extractor)),
      audio_encoder_runner_(std::move(audio_encoder_runner)),
      audio_enc_input_buffers_(std::move(audio_enc_input_buffers)),
      audio_enc_output_buffers_(std::move(audio_enc_output_buffers)) {}

Any2AnyWhisperAudioStage::~Any2AnyWhisperAudioStage() = default;

absl::StatusOr<std::unique_ptr<Any2AnyWhisperAudioStage>>
Any2AnyWhisperAudioStage::Create(
    FanOutOutputQueue<OmniSession::Input, AudioInput>* absl_nonnull audio_queue,
    FanOutOutputQueue<OmniSession::Input, AudioInputMetadata>* absl_nonnull
        audio_input_metadata_queue,
    std::unique_ptr<LiteRtRunner> absl_nonnull audio_encoder_runner,
    Options options) {
  if (options.llm_hidden_size <= 0) {
    return absl::InvalidArgumentError(
        "Any2AnyWhisperAudioStage llm_hidden_size must be positive.");
  }
  if (options.initial_input_sample_rate_hz <= 0) {
    return absl::InvalidArgumentError(
        "Any2AnyWhisperAudioStage initial_input_sample_rate_hz must be "
        "positive.");
  }
  if (options.target_sample_rate_hz <= 0) {
    return absl::InvalidArgumentError(
        "Any2AnyWhisperAudioStage target_sample_rate_hz must be positive.");
  }
  if (options.num_mel_bins <= 0) {
    return absl::InvalidArgumentError(
        "Any2AnyWhisperAudioStage num_mel_bins must be positive.");
  }

  ABSL_ASSIGN_OR_RETURN(std::vector<TensorBuffer> audio_enc_input_buffers,
                        audio_encoder_runner->CreateInputBuffers(""));
  ABSL_ASSIGN_OR_RETURN(std::vector<TensorBuffer> audio_enc_output_buffers,
                        audio_encoder_runner->CreateOutputBuffers(""));
  if (audio_enc_input_buffers.empty() || audio_enc_output_buffers.empty()) {
    return absl::InvalidArgumentError(
        "Any2AnyWhisperAudioStage audio_encoder_runner must provide non-empty "
        "input and output buffers.");
  }

  LITERT_ASSIGN_OR_RETURN(size_t audio_enc_out_bytes,
                          audio_enc_output_buffers[0].Size());
  const size_t token_emb_bytes =
      static_cast<size_t>(options.llm_hidden_size) * sizeof(float);
  if (audio_enc_out_bytes == 0 || audio_enc_out_bytes % token_emb_bytes != 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Any2AnyWhisperAudioStage audio_encoder_runner output buffer size (",
        audio_enc_out_bytes,
        " bytes) must be a positive multiple of llm_hidden_size (",
        options.llm_hidden_size, ") * sizeof(float) (", token_emb_bytes,
        " bytes)."));
  }
  const int max_chunk_tokens =
      static_cast<int>(audio_enc_out_bytes / token_emb_bytes);

  LITERT_ASSIGN_OR_RETURN(size_t in_bytes, audio_enc_input_buffers[0].Size());
  const size_t mel_frame_bytes =
      static_cast<size_t>(options.num_mel_bins) * sizeof(float);
  if (in_bytes == 0 || in_bytes % mel_frame_bytes != 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Any2AnyWhisperAudioStage audio_encoder_runner input buffer size (",
        in_bytes, " bytes) must be a positive multiple of num_mel_bins (",
        options.num_mel_bins, ") * sizeof(float) (", mel_frame_bytes,
        " bytes)."));
  }
  const int chunk_frames = static_cast<int>(in_bytes / mel_frame_bytes);

  const WhisperLogMelExtractor::Config mel_config{
      .sample_rate_hz = options.target_sample_rate_hz,
      .n_fft = kWhisperFrameLength,
      .hop_length = kWhisperHopLength,
      .n_mels = options.num_mel_bins,
      .n_frames = chunk_frames,
      .mel_floor = kWhisperMelFloor,
      .transpose = false,
  };
  ABSL_ASSIGN_OR_RETURN(
      std::unique_ptr<WhisperLogMelExtractor> log_mel_extractor,
      WhisperLogMelExtractor::Create(mel_config));

  return absl::WrapUnique(new Any2AnyWhisperAudioStage(
      audio_queue, audio_input_metadata_queue, chunk_frames, max_chunk_tokens,
      std::move(log_mel_extractor), std::move(audio_encoder_runner),
      std::move(audio_enc_input_buffers), std::move(audio_enc_output_buffers),
      std::move(options)));
}

bool Any2AnyWhisperAudioStage::NeedScheduleInternal() const {
  return !audio_queue_.Empty() || !audio_input_metadata_queue_.Empty();
}

absl::StatusOr<std::vector<EncodedAudioChunk>>
Any2AnyWhisperAudioStage::ProcessAudio(const AudioInput& audio) {
  std::vector<float> resampled_pcm;
  absl::Span<const float> pcm_target;
  if (input_sample_rate_hz_ == options_.target_sample_rate_hz) {
    pcm_target = absl::MakeConstSpan(audio.pcm_samples);
  } else {
    ABSL_ASSIGN_OR_RETURN(
        resampled_pcm,
        resampler_.Resample(audio.pcm_samples, input_sample_rate_hz_,
                            options_.target_sample_rate_hz));
    pcm_target = absl::MakeConstSpan(resampled_pcm);
  }
  if (pcm_target.size() < static_cast<size_t>(kWhisperHopLength)) {
    return absl::InvalidArgumentError(
        "AudioInput is too short to produce any mel spectrogram frames.");
  }

  const int total_frames =
      static_cast<int>(pcm_target.size()) / kWhisperHopLength;
  const int num_chunks = (total_frames + chunk_frames_ - 1) / chunk_frames_;
  const int total_padded_frames = num_chunks * chunk_frames_;
  if (log_mel_extractor_->GetConfig().n_frames != total_padded_frames) {
    WhisperLogMelExtractor::Config mel_config = log_mel_extractor_->GetConfig();
    mel_config.n_frames = total_padded_frames;
    ABSL_ASSIGN_OR_RETURN(log_mel_extractor_,
                          WhisperLogMelExtractor::Create(mel_config));
  }
  const absl::Span<const float> active_pcm = pcm_target.subspan(
      0, static_cast<size_t>(total_frames) * kWhisperHopLength);
  ABSL_ASSIGN_OR_RETURN(const std::vector<float> full_mel,
                        log_mel_extractor_->Extract(active_pcm));

  std::vector<EncodedAudioChunk> chunks;
  chunks.reserve(num_chunks);
  for (int chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
    const int frame_offset = chunk_idx * chunk_frames_;
    const int valid_frames =
        std::min(chunk_frames_, total_frames - frame_offset);
    {
      LITERT_ASSIGN_OR_RETURN(
          auto lock_and_addr,
          TensorBufferScopedLock::Create<float>(
              audio_enc_input_buffers_[0], TensorBuffer::LockMode::kWrite));
      float* dst = lock_and_addr.second;
      if (num_chunks == 1) {
        absl::c_copy(full_mel, dst);
      } else {
        for (int b = 0; b < options_.num_mel_bins; ++b) {
          const absl::Span<const float> src_row =
              absl::MakeConstSpan(full_mel).subspan(
                  static_cast<size_t>(b) * total_padded_frames + frame_offset,
                  chunk_frames_);
          absl::c_copy(src_row, dst + static_cast<size_t>(b) * chunk_frames_);
        }
      }
    }

    ABSL_RETURN_IF_ERROR(audio_encoder_runner_->Run(
        "", audio_enc_input_buffers_, audio_enc_output_buffers_));

    const int raw_valid_tokens =
        options_.compute_valid_tokens
            ? options_.compute_valid_tokens(valid_frames, chunk_frames_,
                                            max_chunk_tokens_)
            : DefaultComputeValidAudioTokens(valid_frames, chunk_frames_,
                                             max_chunk_tokens_);
    const int valid_tokens = std::clamp(raw_valid_tokens, 1, max_chunk_tokens_);
    const size_t valid_floats =
        static_cast<size_t>(valid_tokens) * options_.llm_hidden_size;

    std::vector<float> chunk_out;
    {
      LITERT_ASSIGN_OR_RETURN(
          auto lock_and_addr,
          TensorBufferScopedLock::Create<const float>(
              audio_enc_output_buffers_[0], TensorBuffer::LockMode::kRead));
      const float* src = lock_and_addr.second;
      chunk_out.assign(src, src + valid_floats);
    }

    chunks.push_back(EncodedAudioChunk{
        .embeddings = std::move(chunk_out),
        .valid_tokens = valid_tokens,
    });
  }

  return chunks;
}

absl::Status Any2AnyWhisperAudioStage::ScheduleInternal() {
  absl::Cleanup cleanup = [this] { SetState(State::kIdle); };
  if (!NeedScheduleInternal()) {
    return absl::NotFoundError("Audio queues have no output.");
  }

  while (NeedScheduleInternal()) {
    bool take_metadata = false;
    if (!audio_input_metadata_queue_.Empty()) {
      if (audio_queue_.Empty()) {
        take_metadata = true;
      } else {
        ABSL_ASSIGN_OR_RETURN(
            const Sequenced<AudioInputMetadata>* peeked_metadata,
            audio_input_metadata_queue_.PeekFront());
        ABSL_ASSIGN_OR_RETURN(const Sequenced<AudioInput>* peeked_audio,
                              audio_queue_.PeekFront());
        take_metadata = peeked_metadata->seqnum < peeked_audio->seqnum;
      }
    }

    if (take_metadata) {
      ABSL_ASSIGN_OR_RETURN(const Sequenced<AudioInputMetadata>* metadata,
                            audio_input_metadata_queue_.PeekFront());
      if (metadata->item.sample_rate_hz <= 0) {
        return absl::InvalidArgumentError(absl::StrCat(
            "AudioInputMetadata sample_rate_hz must be positive, got ",
            metadata->item.sample_rate_hz, "."));
      }
      if (metadata->item.num_channels != 1) {
        return absl::InvalidArgumentError(
            absl::StrCat("Any2AnyWhisperAudioStage only supports mono audio "
                         "(num_channels == 1), got ",
                         metadata->item.num_channels, "."));
      }
      input_sample_rate_hz_ = metadata->item.sample_rate_hz;
      resampler_.Reset();
      log_mel_extractor_->Reset();
      const uint64_t seqnum = metadata->seqnum;
      ABSL_RETURN_IF_ERROR(audio_input_metadata_queue_.PopFront().status());
      PushOutput(Sequenced<std::vector<EncodedAudioChunk>>{
          .seqnum = seqnum,
          .item = {},
      });
    } else {
      ABSL_ASSIGN_OR_RETURN(const Sequenced<AudioInput>* sequenced_input,
                            audio_queue_.PeekFront());
      const uint64_t seqnum = sequenced_input->seqnum;
      if (sequenced_input->item.pcm_samples.empty()) {
        ABSL_RETURN_IF_ERROR(audio_queue_.PopFront().status());
        PushOutput(Sequenced<std::vector<EncodedAudioChunk>>{
            .seqnum = seqnum,
            .item = {},
        });
        continue;
      }
      ABSL_ASSIGN_OR_RETURN(std::vector<EncodedAudioChunk> chunks,
                            ProcessAudio(sequenced_input->item));
      ABSL_RETURN_IF_ERROR(audio_queue_.PopFront().status());
      PushOutput(Sequenced<std::vector<EncodedAudioChunk>>{
          .seqnum = seqnum,
          .item = std::move(chunks),
      });
    }
  }
  return absl::OkStatus();
}

void Any2AnyWhisperAudioStage::ResetInternal() {
  SingleThreadedStageWithDeque::ResetInternal();
  input_sample_rate_hz_ = options_.initial_input_sample_rate_hz;
  resampler_.Reset();
  log_mel_extractor_->Reset();
}

}  // namespace litert::omni::any2any
