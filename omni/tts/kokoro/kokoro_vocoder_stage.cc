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

#include "omni/tts/kokoro/kokoro_vocoder_stage.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "kiss_fftr.h"  // from @kissfft
#include "litert/cc/litert_macros.h"  // from @litert
#include "omni/base/io_types.h"
#include "omni/base/model_resources.h"
#include "omni/base/model_utils.h"
#include "omni/base/stage.h"
#include "omni/tts/kokoro/common.h"
#include "omni/tts/kokoro/kokoro_io_types.h"

namespace litert::omni::tts {

absl::StatusOr<std::unique_ptr<KokoroVocoderStage>> KokoroVocoderStage::Create(
    Stage<KokoroAcousticOutput>* absl_nonnull acoustic_predictor,
    std::shared_ptr<ModelResources> absl_nonnull resources) {
  auto stage = std::unique_ptr<KokoroVocoderStage>(
      new KokoroVocoderStage(acoustic_predictor, std::move(resources)));

  // Load the compiled neural vocoder model.
  LITERT_ASSIGN_OR_RETURN(
      stage->vocoder_model_,
      stage->resources_->GetCompiledModel("kokoro_vocoder"));

  // Allocate reusable input and output tensor buffers for the vocoder model.
  LITERT_ASSIGN_OR_RETURN(stage->vocoder_input_buffers_,
                          stage->vocoder_model_->CreateInputBuffers());
  LITERT_ASSIGN_OR_RETURN(stage->vocoder_output_buffers_,
                          stage->vocoder_model_->CreateOutputBuffers());

  if (stage->vocoder_input_buffers_.size() < 5) {
    return absl::InternalError(absl::StrFormat(
        "kokoro_vocoder expected at least 5 input buffers, but got %d",
        stage->vocoder_input_buffers_.size()));
  }
  if (stage->vocoder_output_buffers_.empty()) {
    return absl::InternalError(
        "kokoro_vocoder expected non-empty output buffers");
  }

  // Resolve input tensor indices by signature name, falling back to canonical
  // positions.
  LITERT_ASSIGN_OR_RETURN(
      stage->input_indices_.acoustic_features,
      ResolveInputIndex(*stage->vocoder_model_, "acoustic_features"));
  LITERT_ASSIGN_OR_RETURN(
      stage->input_indices_.pitch_contour,
      ResolveInputIndex(*stage->vocoder_model_, "pitch_contour"));
  LITERT_ASSIGN_OR_RETURN(
      stage->input_indices_.energy_contour,
      ResolveInputIndex(*stage->vocoder_model_, "energy_contour"));
  LITERT_ASSIGN_OR_RETURN(
      stage->input_indices_.speaker_style,
      ResolveInputIndex(*stage->vocoder_model_, "speaker_style"));
  LITERT_ASSIGN_OR_RETURN(
      stage->input_indices_.speech_frame_length,
      ResolveInputIndex(*stage->vocoder_model_, "speech_frame_length"));

  // Resolve output tensor indices by signature name.
  LITERT_ASSIGN_OR_RETURN(
      stage->output_indices_.magnitude_spectrogram,
      ResolveOutputIndex(*stage->vocoder_model_, "magnitude_spectrogram"));
  LITERT_ASSIGN_OR_RETURN(
      stage->output_indices_.phase_spectrogram,
      ResolveOutputIndex(*stage->vocoder_model_, "phase_spectrogram"));

  // Initialize persistent KissFFT real-inverse FFT configuration plan.
  stage->fft_cfg_ =
      kiss_fftr_alloc(kokoro::kIstftFrameLen, 1, nullptr, nullptr);
  if (!stage->fft_cfg_) {
    return absl::InternalError(
        "Failed to allocate KissFFT configuration for Kokoro vocoder");
  }

  // Precompute periodic Hann synthesis window with overlap-add normalization.
  // Overlap-add sum with hop=5 is 1.5, normalized by N (20): scale = 1 / (20
  // * 1.5) = 1/30.
  stage->inv_window_.resize(kokoro::kIstftFrameLen);
  const float window_scale =
      1.0f / (kokoro::kIstftFrameLen * kokoro::kHannOverlapAddScale);
  for (int j = 0; j < kokoro::kIstftFrameLen; ++j) {
    float hann =
        0.5f * (1.0f - std::cos(2.0f * M_PI * j / kokoro::kIstftFrameLen));
    stage->inv_window_[j] = hann * window_scale;
  }

  return stage;
}

KokoroVocoderStage::~KokoroVocoderStage() {
  if (fft_cfg_) {
    free(fft_cfg_);
  }
}

absl::Status KokoroVocoderStage::Flush() { return absl::OkStatus(); }

absl::StatusOr<std::vector<float>> KokoroVocoderStage::SynthesizeIstftAudio(
    absl::Span<const float> magnitude_spectrogram,
    absl::Span<const float> phase_spectrogram, int active_subframes) {
  if (active_subframes <= 0 || magnitude_spectrogram.empty() ||
      phase_spectrogram.empty()) {
    return std::vector<float>();
  }

  const size_t total_subframes =
      magnitude_spectrogram.size() / kokoro::kIstftBins;
  if (magnitude_spectrogram.size() < kokoro::kIstftBins * total_subframes ||
      phase_spectrogram.size() < kokoro::kIstftBins * total_subframes) {
    return absl::InvalidArgumentError(
        "Invalid spectrogram buffer sizes for iSTFT synthesis");
  }
  active_subframes =
      std::min<int>(active_subframes, static_cast<int>(total_subframes));

  // Allocate overlap-add accumulation buffer.
  const size_t max_audio_len =
      static_cast<size_t>(active_subframes) * kokoro::kIstftHop +
      kokoro::kIstftFrameLen;
  std::vector<float> pcm_accum(max_audio_len, 0.0f);

  kiss_fft_cpx freq_data[kokoro::kIstftBins];
  float time_data[kokoro::kIstftFrameLen];

  // Perform frame-by-frame inverse FFT transform.
  for (int m = 0; m < active_subframes; ++m) {
    // Convert polar magnitude and phase to complex frequency bins.
    for (int k = 0; k < kokoro::kIstftBins; ++k) {
      float mag = magnitude_spectrogram[k * total_subframes + m];
      float ph = phase_spectrogram[k * total_subframes + m];
      if (std::isnan(mag) || std::isinf(mag)) mag = 0.0f;
      if (std::isnan(ph) || std::isinf(ph)) ph = 0.0f;

      freq_data[k].r = mag * std::cos(ph);
      freq_data[k].i = mag * std::sin(ph);
    }

    // Execute inverse real FFT for this frame.
    kiss_fftri(fft_cfg_, freq_data, time_data);

    // Apply normalized synthesis window and accumulate into overlap-add buffer.
    const size_t out_start = static_cast<size_t>(m) * kokoro::kIstftHop;
    for (int j = 0; j < kokoro::kIstftFrameLen; ++j) {
      float sample = time_data[j] * inv_window_[j];
      if (!std::isnan(sample) && !std::isinf(sample)) {
        pcm_accum[out_start + j] += sample;
      }
    }
  }

  // STFT center=True uses pad_len = kIstftFrameLen / 2 padding at both ends.
  // Trim pad_len from start to align synthesized waveform with time 0.
  constexpr int kPadLen = kokoro::kIstftFrameLen / 2;
  const size_t trimmed_len =
      (active_subframes > 0)
          ? (static_cast<size_t>(active_subframes - 1) * kokoro::kIstftHop)
          : 0;
  if (pcm_accum.size() < kPadLen + trimmed_len) {
    return std::vector<float>();
  }

  std::vector<float> pcm_output(trimmed_len);
  std::copy(pcm_accum.begin() + kPadLen,
            pcm_accum.begin() + kPadLen + trimmed_len, pcm_output.begin());
  return pcm_output;
}

absl::Status KokoroVocoderStage::ScheduleInternal() {
  absl::Cleanup cleanup = [this] { SetState(State::kIdle); };
  ABSL_VLOG(2) << "[TRACE] Starting KokoroVocoderStage::ScheduleInternal";

  // Retrieve input acoustic features from upstream acoustic predictor stage.
  auto acoustic_res = acoustic_predictor_.GetOutput();
  if (absl::IsNotFound(acoustic_res.status())) {
    return absl::OkStatus();
  } else if (!acoustic_res.ok()) {
    return acoustic_res.status();
  }
  KokoroAcousticOutput payload = std::move(*acoustic_res);

  // Step 1: Write input acoustic representation, pitch, energy, style, and
  // length tensors.
  std::vector<int64_t> speech_len_vec = {
      static_cast<int64_t>(payload.l_speech)};

  LITERT_RETURN_IF_ERROR(
      vocoder_input_buffers_[input_indices_.acoustic_features].Write<float>(
          absl::MakeConstSpan(payload.asr_data)));
  LITERT_RETURN_IF_ERROR(
      vocoder_input_buffers_[input_indices_.pitch_contour].Write<float>(
          absl::MakeConstSpan(payload.f0_n_data)));
  LITERT_RETURN_IF_ERROR(
      vocoder_input_buffers_[input_indices_.energy_contour].Write<float>(
          absl::MakeConstSpan(payload.n_aux_data)));
  LITERT_RETURN_IF_ERROR(
      vocoder_input_buffers_[input_indices_.speaker_style].Write<float>(
          absl::MakeConstSpan(payload.ref_s_decoder)));
  LITERT_RETURN_IF_ERROR(
      vocoder_input_buffers_[input_indices_.speech_frame_length].Write<int64_t>(
          absl::MakeConstSpan(speech_len_vec)));

  // Step 2: Execute neural vocoder inference.
  LITERT_RETURN_IF_ERROR(
      vocoder_model_->Run(vocoder_input_buffers_, vocoder_output_buffers_));

  // Step 3: Extract spectrogram representations and synthesize time-domain PCM
  // samples.
  std::vector<float> pcm_output;
  size_t target_samples = std::min<size_t>(
      kokoro::kMaxSpeechFrames * kokoro::kAudioHop,
      static_cast<size_t>(payload.l_speech * kokoro::kAudioHop));
  if (target_samples == 0) target_samples = kokoro::kAudioHop;

  // Query output tensor buffer sizes dynamically to avoid overflow or
  // mismatch with model output dimensions.
  LITERT_ASSIGN_OR_RETURN(
      auto mag_packed_size,
      vocoder_output_buffers_[output_indices_.magnitude_spectrogram]
          .PackedSize());
  LITERT_ASSIGN_OR_RETURN(
      auto phase_packed_size,
      vocoder_output_buffers_[output_indices_.phase_spectrogram].PackedSize());

  std::vector<float> magnitude_spectrogram(mag_packed_size / sizeof(float),
                                           0.0f);
  std::vector<float> phase_spectrogram(phase_packed_size / sizeof(float),
                                       0.0f);

  LITERT_RETURN_IF_ERROR(
      vocoder_output_buffers_[output_indices_.magnitude_spectrogram]
          .Read<float>(absl::MakeSpan(magnitude_spectrogram)));
  LITERT_RETURN_IF_ERROR(
      vocoder_output_buffers_[output_indices_.phase_spectrogram].Read<float>(
          absl::MakeSpan(phase_spectrogram)));

  const int active_subframes =
      static_cast<int>(
          payload.l_speech * (kokoro::kAudioHop / kokoro::kIstftHop) + 1);

  ABSL_ASSIGN_OR_RETURN(
      pcm_output, SynthesizeIstftAudio(magnitude_spectrogram, phase_spectrogram,
                                       active_subframes));
  pcm_output.resize(target_samples, 0.0f);

  // Step 4: Package synthesized audio payload and push downstream.
  AudioOutput output;
  output.pcm_samples = std::move(pcm_output);
  output.sample_rate_hz = kokoro::kSampleRate;

  PushOutput(std::move(output));
  return absl::OkStatus();
}

}  // namespace litert::omni::tts
