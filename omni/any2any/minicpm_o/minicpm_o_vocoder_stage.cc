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
#include <cstdint>
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
  if (config.vocoder_history_tokens < 0 || config.vocoder_chunk_tokens <= 0 ||
      config.vocoder_lookahead_tokens < 0 ||
      config.vocoder_samples_per_token <= 0) {
    return absl::InvalidArgumentError(
        "MiniCpmOVocoderStage token window and sample counts must be valid.");
  }

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

    stage->history_tokens_ = stage->config_.vocoder_history_tokens;
    stage->chunk_tokens_ = stage->config_.vocoder_chunk_tokens;
    stage->lookahead_tokens_ = stage->config_.vocoder_lookahead_tokens;
    stage->samples_per_token_ = stage->config_.vocoder_samples_per_token;

    const size_t expected_in_bytes =
        static_cast<size_t>(stage->history_tokens_ + stage->chunk_tokens_ +
                            stage->lookahead_tokens_) *
        sizeof(int32_t);
    const size_t expected_out_bytes =
        static_cast<size_t>(stage->history_tokens_ + stage->chunk_tokens_) *
        stage->samples_per_token_ * sizeof(float);
    if (in_bytes != expected_in_bytes || out_bytes != expected_out_bytes) {
      return absl::InvalidArgumentError(
          "audio_vocoder buffer sizes do not match configured vocoder token "
          "window and PCM sample dimensions.");
    }
  } else if (stage->config_.generate_audio) {
    return absl::NotFoundError(
        "MiniCpmOVocoderStage requires vocoder_runner when generate_audio is "
        "enabled.");
  }

  return stage;
}

absl::StatusOr<std::vector<float>> MiniCpmOVocoderStage::SynthesizeWaveform(
    const MiniCpmOTtsOutput& tts_out) {
  if (vocoder_runner_ == nullptr || tts_out.audio_codes.empty()) {
    return std::vector<float>{};
  }

  const int total_tokens = static_cast<int>(tts_out.audio_codes.size());
  const int window_tokens = history_tokens_ + chunk_tokens_ + lookahead_tokens_;
  const size_t history_samples =
      static_cast<size_t>(history_tokens_) * samples_per_token_;

  std::vector<float> waveform;
  waveform.reserve(static_cast<size_t>(total_tokens) * samples_per_token_);

  int emitted_tokens = 0;
  while (emitted_tokens < total_tokens) {
    const int valid_chunk_tokens =
        std::min(chunk_tokens_, total_tokens - emitted_tokens);
    std::vector<int32_t> window_codes(window_tokens,
                                      minicpm_o::kTtsSilenceTokenId);
    for (int i = 0; i < history_tokens_; ++i) {
      const int src_idx = emitted_tokens - history_tokens_ + i;
      if (src_idx >= 0 && src_idx < total_tokens) {
        window_codes[i] = static_cast<int32_t>(tts_out.audio_codes[src_idx]);
      }
    }
    for (int j = 0; j < chunk_tokens_ + lookahead_tokens_; ++j) {
      const int src_idx = emitted_tokens + j;
      if (src_idx >= 0 && src_idx < total_tokens) {
        window_codes[history_tokens_ + j] =
            static_cast<int32_t>(tts_out.audio_codes[src_idx]);
      }
    }

    LITERT_RETURN_IF_ERROR(
        input_buffers_[0].Write<int32_t>(absl::MakeConstSpan(window_codes)));
    ABSL_RETURN_IF_ERROR(
        vocoder_runner_->Run("", input_buffers_, output_buffers_));

    LITERT_ASSIGN_OR_RETURN(
        std::vector<float> chunk_pcm,
        support::CopyFromTensorBuffer<float>(output_buffers_[0]));

    // Crossfade only a short seam window (up to 48 samples = 2 ms at 24 kHz)
    // immediately preceding `chunk_pcm[history_samples]` to prevent boundary
    // clicks without superimposing 160 ms (`3840` samples) of independently
    // phased HiFTGenerator harmonic waveforms.
    constexpr size_t kMaxSeamCrossfadeSamples = 48;
    const size_t crossfade_samples =
        std::min(history_samples, kMaxSeamCrossfadeSamples);
    if (emitted_tokens > 0 && crossfade_samples > 0 &&
        waveform.size() >= crossfade_samples &&
        chunk_pcm.size() >= history_samples) {
      const size_t waveform_seam_start = waveform.size() - crossfade_samples;
      const size_t chunk_seam_start = history_samples - crossfade_samples;
      for (size_t i = 0; i < crossfade_samples; ++i) {
        const float alpha = static_cast<float>(i + 1) /
                            static_cast<float>(crossfade_samples + 1);
        waveform[waveform_seam_start + i] =
            (1.0f - alpha) * waveform[waveform_seam_start + i] +
            alpha * chunk_pcm[chunk_seam_start + i];
      }
    }

    const size_t valid_new_samples =
        static_cast<size_t>(valid_chunk_tokens) * samples_per_token_;
    const size_t copy_start = std::min(history_samples, chunk_pcm.size());
    const size_t copy_end =
        std::min(history_samples + valid_new_samples, chunk_pcm.size());
    waveform.insert(waveform.end(), chunk_pcm.begin() + copy_start,
                    chunk_pcm.begin() + copy_end);
    emitted_tokens += valid_chunk_tokens;
  }

  return waveform;
}

absl::Status MiniCpmOVocoderStage::ProcessTtsOutput(
    const MiniCpmOTtsOutput& tts_out) {
  if (tts_out.has_text && !tts_out.text.empty()) {
    PushOutput(TextOutput{.confirmed_text = tts_out.text});
  }

  if (!tts_out.audio_codes.empty() && vocoder_runner_ != nullptr) {
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
