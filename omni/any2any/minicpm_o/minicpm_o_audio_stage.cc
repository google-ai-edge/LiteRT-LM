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

#include "omni/any2any/minicpm_o/minicpm_o_audio_stage.h"

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
namespace {

constexpr int kWhisperFrameLength = 400;
constexpr int kWhisperHopLength = 160;

}  // namespace

MiniCpmOAudioStage::MiniCpmOAudioStage(
    FanOutOutputQueue<OmniSession::Input, AudioInput>* absl_nonnull audio_queue,
    FanOutOutputQueue<OmniSession::Input, AudioInputMetadata>* absl_nonnull
        audio_input_metadata_queue,
    int chunk_frames,
    std::unique_ptr<WhisperLogMelExtractor> absl_nonnull log_mel_extractor,
    std::unique_ptr<LiteRtRunner> absl_nonnull audio_encoder_runner,
    std::vector<TensorBuffer> audio_enc_input_buffers,
    std::vector<TensorBuffer> audio_enc_output_buffers,
    const MiniCpmOModelConfig& config)
    : audio_queue_(*audio_queue),
      audio_input_metadata_queue_(*audio_input_metadata_queue),
      config_(config),
      hidden_dim_(config.llm_hidden_size),
      chunk_frames_(chunk_frames),
      input_sample_rate_hz_(config.audio_input_sample_rate_hz),
      log_mel_extractor_(std::move(log_mel_extractor)),
      audio_encoder_runner_(std::move(audio_encoder_runner)),
      audio_enc_input_buffers_(std::move(audio_enc_input_buffers)),
      audio_enc_output_buffers_(std::move(audio_enc_output_buffers)) {}

MiniCpmOAudioStage::~MiniCpmOAudioStage() = default;

absl::StatusOr<std::unique_ptr<MiniCpmOAudioStage>> MiniCpmOAudioStage::Create(
    FanOutOutputQueue<OmniSession::Input, AudioInput>* absl_nonnull audio_queue,
    FanOutOutputQueue<OmniSession::Input, AudioInputMetadata>* absl_nonnull
        audio_input_metadata_queue,
    std::unique_ptr<LiteRtRunner> absl_nonnull audio_encoder_runner,
    const MiniCpmOModelConfig& config) {
  if (config.llm_hidden_size <= 0) {
    return absl::InvalidArgumentError(
        "MiniCpmOAudioStage config.llm_hidden_size must be positive.");
  }
  if (config.audio_input_sample_rate_hz <= 0) {
    return absl::InvalidArgumentError(
        "MiniCpmOAudioStage config.audio_input_sample_rate_hz must be "
        "positive.");
  }

  ABSL_ASSIGN_OR_RETURN(std::vector<TensorBuffer> audio_enc_input_buffers,
                        audio_encoder_runner->CreateInputBuffers(""));
  ABSL_ASSIGN_OR_RETURN(std::vector<TensorBuffer> audio_enc_output_buffers,
                        audio_encoder_runner->CreateOutputBuffers(""));
  if (audio_enc_input_buffers.empty() || audio_enc_output_buffers.empty()) {
    return absl::InvalidArgumentError(
        "MiniCpmOAudioStage audio_encoder_runner must provide non-empty input "
        "and output buffers.");
  }
  LITERT_ASSIGN_OR_RETURN(size_t audio_enc_out_bytes,
                          audio_enc_output_buffers[0].Size());
  const size_t token_emb_bytes =
      static_cast<size_t>(config.llm_hidden_size) * sizeof(float);
  if (audio_enc_out_bytes == 0 || audio_enc_out_bytes % token_emb_bytes != 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "MiniCpmOAudioStage audio_encoder_runner output buffer size (",
        audio_enc_out_bytes,
        " bytes) must be a positive multiple of llm_hidden_size (",
        config.llm_hidden_size, ") * sizeof(float) (", token_emb_bytes,
        " bytes)."));
  }

  LITERT_ASSIGN_OR_RETURN(size_t in_bytes, audio_enc_input_buffers[0].Size());
  const int num_bins = minicpm_o::kAudioMelBins;
  const size_t mel_frame_bytes = static_cast<size_t>(num_bins) * sizeof(float);
  if (in_bytes == 0 || in_bytes % mel_frame_bytes != 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "MiniCpmOAudioStage audio_encoder_runner input buffer size (", in_bytes,
        " bytes) must be a positive multiple of kAudioMelBins (", num_bins,
        ") * sizeof(float) (", mel_frame_bytes, " bytes)."));
  }
  const int chunk_frames = static_cast<int>(in_bytes / mel_frame_bytes);

  const WhisperLogMelExtractor::Config mel_config{
      .sample_rate_hz = minicpm_o::kAudioInputSampleRateHz,
      .n_fft = kWhisperFrameLength,
      .hop_length = kWhisperHopLength,
      .n_mels = minicpm_o::kAudioMelBins,
      .n_frames = chunk_frames,
      .mel_floor = 1e-10f,
      .transpose = false,
  };
  ABSL_ASSIGN_OR_RETURN(
      std::unique_ptr<WhisperLogMelExtractor> log_mel_extractor,
      WhisperLogMelExtractor::Create(mel_config));

  return absl::WrapUnique(new MiniCpmOAudioStage(
      audio_queue, audio_input_metadata_queue, chunk_frames,
      std::move(log_mel_extractor), std::move(audio_encoder_runner),
      std::move(audio_enc_input_buffers), std::move(audio_enc_output_buffers),
      config));
}

absl::StatusOr<MiniCpmOEmbeddingSegment> MiniCpmOAudioStage::ProcessAudio(
    const AudioInput& audio) {
  if (audio.pcm_samples.empty()) {
    return MiniCpmOEmbeddingSegment{};
  }

  ABSL_ASSIGN_OR_RETURN(
      std::vector<float> pcm_16k,
      resampler_.Resample(audio.pcm_samples, input_sample_rate_hz_,
                          minicpm_o::kAudioInputSampleRateHz));
  if (pcm_16k.size() < static_cast<size_t>(kWhisperHopLength)) {
    return absl::InvalidArgumentError(
        "AudioInput is too short to produce any mel spectrogram frames.");
  }

  const int total_frames = static_cast<int>(pcm_16k.size()) / kWhisperHopLength;
  const int num_chunks = (total_frames + chunk_frames_ - 1) / chunk_frames_;
  const int total_padded_frames = num_chunks * chunk_frames_;
  if (log_mel_extractor_->GetConfig().n_frames != total_padded_frames) {
    WhisperLogMelExtractor::Config mel_config = log_mel_extractor_->GetConfig();
    mel_config.n_frames = total_padded_frames;
    ABSL_ASSIGN_OR_RETURN(log_mel_extractor_,
                          WhisperLogMelExtractor::Create(mel_config));
  }
  const absl::Span<const float> active_pcm =
      absl::MakeConstSpan(pcm_16k).subspan(
          0, static_cast<size_t>(total_frames) * kWhisperHopLength);
  ABSL_ASSIGN_OR_RETURN(const std::vector<float> full_mel,
                        log_mel_extractor_->Extract(active_pcm));

  MiniCpmOEmbeddingSegment segment;
  std::vector<float> chunk_input(static_cast<size_t>(minicpm_o::kAudioMelBins) *
                                 chunk_frames_);
  for (int chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
    const int frame_offset = chunk_idx * chunk_frames_;
    const int valid_frames =
        std::min(chunk_frames_, total_frames - frame_offset);
    for (int b = 0; b < minicpm_o::kAudioMelBins; ++b) {
      const absl::Span<const float> src_row =
          absl::MakeConstSpan(full_mel).subspan(
              static_cast<size_t>(b) * total_padded_frames + frame_offset,
              chunk_frames_);
      absl::c_copy(src_row, chunk_input.begin() +
                                static_cast<size_t>(b) * chunk_frames_);
    }

    LITERT_RETURN_IF_ERROR(audio_enc_input_buffers_[0].Write<float>(
        absl::MakeConstSpan(chunk_input)));
    ABSL_RETURN_IF_ERROR(audio_encoder_runner_->Run(
        "", audio_enc_input_buffers_, audio_enc_output_buffers_));

    LITERT_ASSIGN_OR_RETURN(size_t out_bytes,
                            audio_enc_output_buffers_[0].Size());
    std::vector<float> chunk_out(out_bytes / sizeof(float));
    LITERT_RETURN_IF_ERROR(
        audio_enc_output_buffers_[0].Read<float>(absl::MakeSpan(chunk_out)));

    const int max_chunk_tokens =
        std::max(1, static_cast<int>(chunk_out.size()) / hidden_dim_);
    const int valid_tokens = std::min(
        max_chunk_tokens,
        std::max(1, (valid_frames * max_chunk_tokens + chunk_frames_ - 1) /
                        chunk_frames_));

    segment.token_ids.push_back(minicpm_o::kAudioStartTokenId);
    segment.token_ids.insert(segment.token_ids.end(), valid_tokens,
                             minicpm_o::kAudioEmbedPlaceholderTokenId);
    segment.token_ids.push_back(minicpm_o::kAudioEndTokenId);
    segment.embeddings.insert(
        segment.embeddings.end(), chunk_out.begin(),
        chunk_out.begin() + static_cast<size_t>(valid_tokens) * hidden_dim_);
    segment.num_tokens += valid_tokens;
  }

  return segment;
}

bool MiniCpmOAudioStage::NeedScheduleInternal() const {
  return !audio_queue_.Empty() || !audio_input_metadata_queue_.Empty();
}

absl::Status MiniCpmOAudioStage::ScheduleInternal() {
  absl::Cleanup cleanup = [this] { SetState(State::kIdle); };

  if (audio_queue_.Empty() && audio_input_metadata_queue_.Empty()) {
    return absl::NotFoundError("Audio queues have no output.");
  }

  while (!audio_queue_.Empty() || !audio_input_metadata_queue_.Empty()) {
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
        return absl::InvalidArgumentError(absl::StrCat(
            "MiniCpmOAudioStage only supports mono audio (num_channels == 1), "
            "got ",
            metadata->item.num_channels, "."));
      }
      if (metadata->item.sample_rate_hz != input_sample_rate_hz_) {
        input_sample_rate_hz_ = metadata->item.sample_rate_hz;
        resampler_.Reset();
      }
      const uint64_t seqnum = metadata->seqnum;
      ABSL_RETURN_IF_ERROR(audio_input_metadata_queue_.PopFront().status());
      this->PushOutput(Sequenced<MiniCpmOEmbeddingSegment>{
          .seqnum = seqnum,
          .item = MiniCpmOEmbeddingSegment{},
      });
    } else {
      ABSL_ASSIGN_OR_RETURN(const Sequenced<AudioInput>* sequenced_input,
                            audio_queue_.PeekFront());
      ABSL_ASSIGN_OR_RETURN(MiniCpmOEmbeddingSegment segment,
                            ProcessAudio(sequenced_input->item));
      const uint64_t seqnum = sequenced_input->seqnum;
      ABSL_RETURN_IF_ERROR(audio_queue_.PopFront().status());
      this->PushOutput(Sequenced<MiniCpmOEmbeddingSegment>{
          .seqnum = seqnum,
          .item = std::move(segment),
      });
    }
  }
  return absl::OkStatus();
}

void MiniCpmOAudioStage::ResetInternal() {
  SingleThreadedStageWithDeque::ResetInternal();
  input_sample_rate_hz_ = config_.audio_input_sample_rate_hz;
  resampler_.Reset();
  log_mel_extractor_->Reset();
}

}  // namespace litert::omni::any2any
