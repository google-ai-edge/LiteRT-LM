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

#include "omni/any2any/minicpm_o/minicpm_o_vocoder_stage.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_macros.h"  // from @litert
#include "omni/any2any/minicpm_o/minicpm_o_model_config.h"
#include "omni/any2any/minicpm_o/minicpm_o_tts_stage.h"
#include "omni/base/io_types.h"
#include "omni/base/litert_runner.h"
#include "omni/base/stage.h"
#include "support/util/convert_tensor_buffer.h"

namespace litert::omni::any2any {

absl::StatusOr<std::unique_ptr<MiniCpmOVocoderStage>>
MiniCpmOVocoderStage::Create(
    Stage<MiniCpmOTtsOutput>* absl_nonnull tts_stage,
    const MiniCpmOModelConfig& config,
    std::unique_ptr<LiteRtRunner> absl_nullable vocoder_runner) {
  auto stage = absl::WrapUnique(
      new MiniCpmOVocoderStage(tts_stage, config, std::move(vocoder_runner)));

  if (stage->vocoder_runner_ != nullptr) {
    ABSL_ASSIGN_OR_RETURN(stage->input_buffers_,
                          stage->vocoder_runner_->CreateInputBuffers(""));
    ABSL_ASSIGN_OR_RETURN(stage->output_buffers_,
                          stage->vocoder_runner_->CreateOutputBuffers(""));
    if (stage->input_buffers_.empty() || stage->output_buffers_.empty()) {
      return absl::InternalError(
          "audio_vocoder model must have at least 1 input and 1 output "
          "buffer.");
    }

    LITERT_ASSIGN_OR_RETURN(size_t in_bytes, stage->input_buffers_[0].Size());
    LITERT_ASSIGN_OR_RETURN(size_t out_bytes, stage->output_buffers_[0].Size());
    const int num_bins = std::max(1, stage->config_.tts_num_mel_bins);
    const int in_floats =
        std::max(1, static_cast<int>(in_bytes / sizeof(float)));
    stage->mel_chunk_frames_ = std::max(1, in_floats / num_bins);
    const int total_samples =
        std::max(1, static_cast<int>(out_bytes / sizeof(float)));
    stage->upsample_hop_ =
        std::max(1, total_samples / stage->mel_chunk_frames_);
  } else if (stage->config_.generate_audio) {
    return absl::NotFoundError(
        "MiniCpmOVocoderStage requires vocoder_runner when generate_audio is "
        "enabled.");
  }

  return stage;
}

absl::StatusOr<std::vector<float>> MiniCpmOVocoderStage::SynthesizeWaveform(
    const MiniCpmOTtsOutput& tts_out) {
  if (vocoder_runner_ == nullptr || tts_out.num_frames <= 0 ||
      tts_out.mel_features.empty()) {
    return std::vector<float>{};
  }

  const int num_bins = std::max(1, tts_out.num_mel_bins);
  const int total_frames = tts_out.num_frames;
  const int chunk_frames = std::max(1, mel_chunk_frames_);
  const int overlap_frames =
      (chunk_frames > 1)
          ? std::max(
                1, std::min(minicpm_o::kVocoderOverlapFrames, chunk_frames / 4))
          : 0;

  std::vector<float> waveform;
  waveform.reserve(static_cast<size_t>(total_frames) * upsample_hop_);

  int emitted_frames = 0;
  while (emitted_frames < total_frames) {
    const int cur_overlap_frames =
        (emitted_frames == 0) ? 0 : std::min(overlap_frames, emitted_frames);
    const int chunk_start = emitted_frames - cur_overlap_frames;
    const int valid_frames = std::min(chunk_frames, total_frames - chunk_start);

    std::vector<float> chunk_mel(static_cast<size_t>(num_bins) * chunk_frames,
                                 0.0f);
    for (int b = 0; b < num_bins; ++b) {
      for (int f = 0; f < chunk_frames; ++f) {
        // Clamp trailing frames of a partial final chunk to the last valid mel
        // frame rather than zero-padding to avoid boundary filter ringing.
        const int src_frame = std::min(chunk_start + f, total_frames - 1);
        const size_t src_idx =
            static_cast<size_t>(b) * total_frames + src_frame;
        if (src_idx < tts_out.mel_features.size()) {
          chunk_mel[static_cast<size_t>(b) * chunk_frames + f] =
              tts_out.mel_features[src_idx];
        }
      }
    }

    LITERT_RETURN_IF_ERROR(
        input_buffers_[0].Write<float>(absl::MakeConstSpan(chunk_mel)));
    ABSL_RETURN_IF_ERROR(
        vocoder_runner_->Run("", input_buffers_, output_buffers_));

    LITERT_ASSIGN_OR_RETURN(
        std::vector<float> chunk_pcm,
        support::CopyFromTensorBuffer<float>(output_buffers_[0]));
    const size_t valid_samples = std::min(
        chunk_pcm.size(), static_cast<size_t>(valid_frames) * upsample_hop_);
    const size_t overlap_samples =
        std::min({static_cast<size_t>(cur_overlap_frames) * upsample_hop_,
                  valid_samples, waveform.size()});

    if (overlap_samples > 0) {
      const size_t crossfade_start = waveform.size() - overlap_samples;
      for (size_t i = 0; i < overlap_samples; ++i) {
        const float alpha =
            static_cast<float>(i + 1) / static_cast<float>(overlap_samples + 1);
        waveform[crossfade_start + i] =
            (1.0f - alpha) * waveform[crossfade_start + i] +
            alpha * chunk_pcm[i];
      }
    }
    waveform.insert(waveform.end(), chunk_pcm.begin() + overlap_samples,
                    chunk_pcm.begin() + valid_samples);
    emitted_frames = chunk_start + valid_frames;
  }

  return waveform;
}

absl::Status MiniCpmOVocoderStage::ProcessTtsOutput(
    const MiniCpmOTtsOutput& tts_out) {
  if (tts_out.has_text && !tts_out.text.empty()) {
    PushOutput(TextOutput{.confirmed_text = tts_out.text});
  }

  if (tts_out.num_frames > 0 && !tts_out.mel_features.empty() &&
      vocoder_runner_ != nullptr) {
    ABSL_ASSIGN_OR_RETURN(std::vector<float> pcm, SynthesizeWaveform(tts_out));
    if (!pcm.empty()) {
      PushOutput(AudioOutput{
          .pcm_samples = std::move(pcm),
          .sample_rate_hz = config_.audio_output_sample_rate_hz,
      });
    }
  }

  if (tts_out.is_text_final) {
    PushOutput(EndOfOutputTurn{});
  }
  return absl::OkStatus();
}

absl::Status MiniCpmOVocoderStage::ScheduleInternal() {
  absl::Cleanup cleanup = [this] { SetState(State::kIdle); };
  auto tts_out = tts_stage_.GetOutput();
  if (absl::IsNotFound(tts_out.status())) {
    return absl::OkStatus();
  }
  if (!tts_out.ok()) {
    return tts_out.status();
  }
  return ProcessTtsOutput(*tts_out);
}

absl::Status MiniCpmOVocoderStage::FlushInternal() {
  while (true) {
    auto tts_out = tts_stage_.GetOutput();
    if (!tts_out.ok()) break;
    ABSL_RETURN_IF_ERROR(ProcessTtsOutput(*tts_out));
  }
  return absl::OkStatus();
}

}  // namespace litert::omni::any2any
