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

  // Pre-derive speech frame length integer width and vocoder frame capacity.
  LITERT_ASSIGN_OR_RETURN(
      size_t sfl_size,
      stage->vocoder_input_buffers_[stage->input_indices_.speech_frame_length]
          .PackedSize());
  stage->speech_frame_length_is_int32_ = (sfl_size == sizeof(int32_t));

  LITERT_ASSIGN_OR_RETURN(
      size_t asr_in_bytes,
      stage->vocoder_input_buffers_[stage->input_indices_.acoustic_features]
          .PackedSize());
  stage->frame_capacity_ = kokoro::FrameCapacityFromPackedSize(asr_in_bytes);
  if (stage->frame_capacity_ <= 0) {
    return absl::InternalError("Invalid acoustic_features frame capacity");
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

  // Step 1: Determine acoustic output frame stride (`t_acoustic`) vs. vocoder
  // input frame capacity (`t_vocoder`) for tensors of shape [1, C, T].
  LITERT_ASSIGN_OR_RETURN(
      auto voc_asr_bytes,
      vocoder_input_buffers_[input_indices_.acoustic_features].PackedSize());
  const int t_vocoder =
      std::max<int>(1, kokoro::FrameCapacityFromPackedSize(voc_asr_bytes));
  const int t_acoustic = std::max<int>(
      1,
      static_cast<int>(payload.asr_data.size() / kokoro::kAcousticFeatureDim));
  const int total_speech_frames = std::clamp(payload.l_speech, 1, t_acoustic);

  LITERT_ASSIGN_OR_RETURN(
      auto mag_packed_size,
      vocoder_output_buffers_[output_indices_.magnitude_spectrogram]
          .PackedSize());
  LITERT_ASSIGN_OR_RETURN(
      auto phase_packed_size,
      vocoder_output_buffers_[output_indices_.phase_spectrogram].PackedSize());

  std::vector<float> magnitude_spectrogram(mag_packed_size / sizeof(float),
                                           0.0f);
  std::vector<float> phase_spectrogram(phase_packed_size / sizeof(float), 0.0f);

  LITERT_ASSIGN_OR_RETURN(
      auto voc_f0_bytes,
      vocoder_input_buffers_[input_indices_.pitch_contour].PackedSize());
  LITERT_ASSIGN_OR_RETURN(
      auto voc_n_bytes,
      vocoder_input_buffers_[input_indices_.energy_contour].PackedSize());
  const int f0_subframes_per_frame = std::max<int>(
      1, static_cast<int>(voc_f0_bytes / sizeof(float)) / t_vocoder);

  std::vector<float> voc_asr_buf(kokoro::kAcousticFeatureDim * t_vocoder, 0.0f);
  std::vector<float> voc_f0_buf(voc_f0_bytes / sizeof(float), 0.0f);
  std::vector<float> voc_n_buf(voc_n_bytes / sizeof(float), 0.0f);

  // Slice `total_speech_frames` into chunks of at most `t_vocoder` frames so
  // compact bucketed vocoders (e.g. `kokoro_vocoder_64.tflite`) can synthesize
  // utterances of any length without buffer mismatch or truncation.
  for (int frame_start = 0; frame_start < total_speech_frames;
       frame_start += t_vocoder) {
    const int chunk_frames =
        std::min(t_vocoder, total_speech_frames - frame_start);

    std::fill(voc_asr_buf.begin(), voc_asr_buf.end(), 0.0f);
    std::fill(voc_f0_buf.begin(), voc_f0_buf.end(), 0.0f);
    std::fill(voc_n_buf.begin(), voc_n_buf.end(), 0.0f);

    for (int c = 0; c < kokoro::kAcousticFeatureDim; ++c) {
      const float* src = payload.asr_data.data() + c * t_acoustic + frame_start;
      float* dst = voc_asr_buf.data() + c * t_vocoder;
      std::copy(src, src + chunk_frames, dst);
    }

    const int f0_start = frame_start * f0_subframes_per_frame;
    const int f0_count = std::min<int>(
        chunk_frames * f0_subframes_per_frame,
        std::min<int>(static_cast<int>(voc_f0_buf.size()),
                      static_cast<int>(payload.f0_n_data.size()) - f0_start));
    if (f0_count > 0) {
      std::copy(payload.f0_n_data.data() + f0_start,
                payload.f0_n_data.data() + f0_start + f0_count,
                voc_f0_buf.data());
    }

    const int n_count = std::min<int>(
        chunk_frames * f0_subframes_per_frame,
        std::min<int>(static_cast<int>(voc_n_buf.size()),
                      static_cast<int>(payload.n_aux_data.size()) - f0_start));
    if (n_count > 0) {
      std::copy(payload.n_aux_data.data() + f0_start,
                payload.n_aux_data.data() + f0_start + n_count,
                voc_n_buf.data());
    }

    LITERT_RETURN_IF_ERROR(
        vocoder_input_buffers_[input_indices_.acoustic_features].Write<float>(
            absl::MakeConstSpan(voc_asr_buf)));
    LITERT_RETURN_IF_ERROR(
        vocoder_input_buffers_[input_indices_.pitch_contour].Write<float>(
            absl::MakeConstSpan(voc_f0_buf)));
    LITERT_RETURN_IF_ERROR(
        vocoder_input_buffers_[input_indices_.energy_contour].Write<float>(
            absl::MakeConstSpan(voc_n_buf)));
    LITERT_RETURN_IF_ERROR(
        vocoder_input_buffers_[input_indices_.speaker_style].Write<float>(
            absl::MakeConstSpan(payload.ref_s_decoder)));

    if (speech_frame_length_is_int32_) {
      std::vector<int32_t> speech_len_vec = {
          static_cast<int32_t>(chunk_frames)};
      LITERT_RETURN_IF_ERROR(
          vocoder_input_buffers_[input_indices_.speech_frame_length]
              .Write<int32_t>(absl::MakeConstSpan(speech_len_vec)));
    } else {
      std::vector<int64_t> speech_len_vec = {
          static_cast<int64_t>(chunk_frames)};
      LITERT_RETURN_IF_ERROR(
          vocoder_input_buffers_[input_indices_.speech_frame_length]
              .Write<int64_t>(absl::MakeConstSpan(speech_len_vec)));
    }

    // Step 2: Execute neural vocoder inference for this frame slice.
    LITERT_RETURN_IF_ERROR(
        vocoder_model_->Run(vocoder_input_buffers_, vocoder_output_buffers_));

    // Step 3: Extract spectrogram representations and synthesize time-domain
    // PCM samples.
    size_t target_samples =
        static_cast<size_t>(chunk_frames) * kokoro::kAudioHop;
    if (target_samples == 0) target_samples = kokoro::kAudioHop;

    LITERT_RETURN_IF_ERROR(
        vocoder_output_buffers_[output_indices_.magnitude_spectrogram]
            .Read<float>(absl::MakeSpan(magnitude_spectrogram)));
    LITERT_RETURN_IF_ERROR(
        vocoder_output_buffers_[output_indices_.phase_spectrogram].Read<float>(
            absl::MakeSpan(phase_spectrogram)));

    const int active_subframes = static_cast<int>(
        chunk_frames * (kokoro::kAudioHop / kokoro::kIstftHop) + 1);

    std::vector<float> pcm_output;
    ABSL_ASSIGN_OR_RETURN(
        pcm_output, SynthesizeIstftAudio(magnitude_spectrogram,
                                         phase_spectrogram, active_subframes));
    pcm_output.resize(target_samples, 0.0f);

    kokoro::TrimSliceJoinSilence(pcm_output, payload.join_before,
                                 payload.join_after);

    // Step 4: Package synthesized audio payload and push downstream.
    AudioOutput output;
    output.pcm_samples = std::move(pcm_output);
    output.sample_rate_hz = kokoro::kSampleRate;

    PushOutput(std::move(output));
  }
  return absl::OkStatus();
}

}  // namespace litert::omni::tts
