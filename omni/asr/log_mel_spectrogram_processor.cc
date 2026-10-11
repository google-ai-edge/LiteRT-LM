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

#include "omni/asr/log_mel_spectrogram_processor.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/log/absl_vlog_is_on.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "omni/asr/audio_preprocessor.h"
#include "omni/asr/utils.h"
#include "omni/base/stage.h"
#include "support/preprocessor/audio_preprocessor.h"
#include "support/preprocessor/audio_preprocessor_miniaudio.h"
#include "support/util/io_types.h"

namespace litert::omni::asr {
namespace {

constexpr float kEpsilon = 1e-5f;
constexpr float kLn10 = 2.302585092994046f;
// Minimum number of real-audio frames required to compute per-feature mean and
// (sample) standard deviation; shorter chunks are skipped like silence.
constexpr int kMinValidFrames = 2;

int GetSmallestPowerOfTwoGreaterOrEqualTo(int n) {
  if (n > 0 && (n & (n - 1)) == 0) {
    return n;
  }
  int p = 1;
  while (p < n) {
    p <<= 1;
  }
  return p;
}

std::vector<float> ComputeMean(const std::vector<std::vector<float>>& mel_spec,
                               int n_frames) {
  std::vector<float> result(mel_spec.size(), 0.0f);
  if (n_frames <= 0) return result;
  for (size_t m = 0; m < mel_spec.size(); ++m) {
    float sum = 0.0f;
    int count = std::min(static_cast<int>(mel_spec[m].size()), n_frames);
    if (count <= 0) continue;
    for (int f = 0; f < count; ++f) {
      sum += mel_spec[m][f];
    }
    result[m] = sum / count;
  }
  return result;
}

std::vector<float> ComputeStd(const std::vector<std::vector<float>>& mel_spec,
                              const std::vector<float>& mean, int n_frames) {
  std::vector<float> result(mel_spec.size(), 0.0f);
  if (n_frames <= 1) return result;
  for (size_t m = 0; m < mel_spec.size(); ++m) {
    float sum_sq = 0.0f;
    int count = std::min(static_cast<int>(mel_spec[m].size()), n_frames);
    if (count <= 1) continue;
    for (int f = 0; f < count; ++f) {
      float diff = mel_spec[m][f] - mean[m];
      sum_sq += diff * diff;
    }
    // Sample variance (N - 1)
    result[m] = std::sqrt(sum_sq / (count - 1));
  }
  return result;
}

}  // namespace

absl::StatusOr<std::unique_ptr<LogMelSpectrogramProcessor>>
LogMelSpectrogramProcessor::Create(
    int sample_rate_hz, const LogMelSpectrogramConfig& config,
    Stage<std::vector<float>>* absl_nonnull audio_source) {
  if (sample_rate_hz <= 0) {
    return absl::InvalidArgumentError("Sample rate must be positive.");
  }
  if (config.n_fft <= 0) {
    return absl::InvalidArgumentError("n_fft must be positive.");
  }
  if (config.hop_length <= 0) {
    return absl::InvalidArgumentError("hop_length must be positive.");
  }
  if (config.n_mels <= 0) {
    return absl::InvalidArgumentError("n_mels must be positive.");
  }
  const bool is_whisper = config.norm_type == NormType::kWhisper;
  int fft_length =
      config.fft_length > 0
          ? config.fft_length
          : (is_whisper ? config.n_fft
                        : GetSmallestPowerOfTwoGreaterOrEqualTo(config.n_fft));
  float mel_high_hz =
      config.mel_high_hz > 0.0f ? config.mel_high_hz : sample_rate_hz / 2.0f;
  litert::support::AudioPreprocessorConfig preprocessor_config =
      litert::support::AudioPreprocessorConfig::Create(
          sample_rate_hz, /*num_channels=*/1,
          /*frame_length=*/config.n_fft,
          /*hop_length=*/config.hop_length,
          /*fft_length=*/fft_length,
          /*input_scale=*/config.input_scale,
          /*pre_emphasis_factor=*/config.preemphasis,
          /*num_mel_bins=*/config.n_mels,
          /*mel_low_hz=*/config.mel_low_hz,
          /*mel_high_hz=*/mel_high_hz,
          /*mel_floor=*/config.mel_floor,
          /*normalize_mel=*/config.normalize_mel,
          /*add_floor_to_mel_before_log=*/config.add_floor_to_mel_before_log,
          /*semicausal_padding=*/false,
          /*non_zero_hanning=*/!is_whisper, /*periodic_hanning=*/true,
          litert::support::AudioPreprocessorConfig::FftPaddingType::kRight);
  preprocessor_config.SetSlaneyMel(config.slaney_mel);
  preprocessor_config.SetPowerSpectrum(config.power_spectrum);

  ABSL_ASSIGN_OR_RETURN(
      auto preprocessor,
      litert::support::AudioPreprocessorMiniAudio::Create(preprocessor_config));
  return absl::WrapUnique(new LogMelSpectrogramProcessor(
      sample_rate_hz, config, audio_source, std::move(preprocessor)));
}

LogMelSpectrogramProcessor::LogMelSpectrogramProcessor(
    int sample_rate_hz, const LogMelSpectrogramConfig& config,
    Stage<std::vector<float>>* absl_nonnull audio_source,
    std::unique_ptr<litert::support::AudioPreprocessorMiniAudio> absl_nonnull
        preprocessor)
    : AudioPreprocessor(audio_source),
      sample_rate_hz_(sample_rate_hz),
      config_(config),
      preprocessor_(std::move(preprocessor)) {}

void LogMelSpectrogramProcessor::ResetInternal() {
  if (preprocessor_) {
    preprocessor_->Reset();
  }
  previous_window_.clear();
  last_valid_frames_ = 0;
  last_start_fraction_ = 0.0f;
  peak_speech_rms_ = 0.0f;
}

absl::Status LogMelSpectrogramProcessor::ScheduleInternal() {
  auto cleanup = absl::MakeCleanup([this]() { SetState(State::kIdle); });
  ABSL_ASSIGN_OR_RETURN(auto pcm_samples, audio_source_.GetOutput());
  ABSL_ASSIGN_OR_RETURN(auto features, Process(std::move(pcm_samples)));
  PushOutput(std::move(features));
  return absl::OkStatus();
}

absl::StatusOr<std::vector<float>> LogMelSpectrogramProcessor::Process(
    std::vector<float> raw_speech) {
  const size_t window_samples =
      config_.n_frames > 0
          ? static_cast<size_t>(config_.n_frames) * config_.hop_length
          : 0;
  size_t overlap = 0;
  if (window_samples > 0 && previous_window_.size() == window_samples) {
    // Overlapping audio sources (`AudioInputSource`, `FileAudioSource`) prefix
    // follow-up chunks with the retained overlap samples from the end of
    // `previous_window_`. Identify that exact overlap so shifting a short final
    // chunk back to a full window splices the contiguous samples preceding
    // `raw_speech.begin()` rather than duplicating the overlap region.
    const size_t max_candidate = std::min(raw_speech.size(), window_samples);
    const size_t min_overlap = static_cast<size_t>(config_.hop_length);
    for (size_t candidate = max_candidate; candidate >= min_overlap;
         --candidate) {
      if (std::equal(previous_window_.end() - candidate, previous_window_.end(),
                     raw_speech.begin())) {
        overlap = candidate;
        break;
      }
    }
    if (overlap == raw_speech.size()) {
      if (preprocessor_) {
        preprocessor_->Reset();
      }
      return std::vector<float>();
    }
  }
  if (window_samples > 0 && raw_speech.size() >= window_samples) {
    // Keep the most recent full window so a short final chunk can be shifted
    // back over it (see below); a silent window is still part of the stream.
    previous_window_.assign(raw_speech.end() - window_samples,
                            raw_speech.end());
  }

  if (IsSilentAudio(raw_speech)) {
    // AudioPreprocessorMiniAudio buffers trailing PCM samples in its internal
    // input_queue_ across calls for STFT window overlap. Reset it when
    // skipping a silent chunk so stale pre-silence samples are not stitched
    // onto the start of the next non-silent chunk.
    if (preprocessor_) {
      preprocessor_->Reset();
    }
    return std::vector<float>();
  }

  // Number of frames backed by real audio. The final chunk of a stream may be
  // shorter than the model window (the audio source does not pad it), so
  // `valid_frames` can be smaller than `config_.n_frames`.
  int valid_frames = raw_speech.size() / config_.hop_length;
  if (valid_frames < kMinValidFrames) {
    // Too short to compute meaningful per-utterance statistics (and to carry
    // any speech); treat like silence so the recognizer skips the chunk.
    if (preprocessor_) {
      preprocessor_->Reset();
    }
    return std::vector<float>();
  }
  if (!preprocessor_) {
    return absl::InternalError(
        "AudioPreprocessorMiniAudio is not initialized.");
  }

  last_start_fraction_ = 0.0f;
  // Some recordings pad a short utterance (~0.5-1 s) with several seconds of
  // low-level room noise (~-50 dBFS) inside a full 5 s window. Normalizing over
  // 4+ s of room noise shifts the silent tail to negative sigmas and collapses
  // the exported encoder to all blanks; treat >= 2 s of trailing room noise
  // (< 6e-3 RMS and < 5% of peak 100 ms speech RMS) like trailing padding.
  constexpr int kBlockFrames = 10;
  constexpr int kMinTrailingSilentBlocks = 20;
  constexpr int kTrailingMarginBlocks = 1;
  constexpr int kMinTrimmedBlocks = 7;
  constexpr float kTrailingSilenceMaxRms = 6e-3f;
  constexpr float kTrailingSilenceRelativeRms = 0.05f;
  bool trimmed_trailing_silence = false;
  if (valid_frames >= kBlockFrames) {
    const int num_blocks = valid_frames / kBlockFrames;
    const size_t block_samples =
        static_cast<size_t>(kBlockFrames) * config_.hop_length;
    std::vector<float> block_rms(num_blocks, 0.0f);
    float max_block_rms = 0.0f;
    float max_new_block_rms = 0.0f;
    const int check_start_block =
        ((!ShiftsShortChunkBack() || config_.normalize_mel) &&
         overlap >= 2 * block_samples)
            ? static_cast<int>(overlap / block_samples) - 2
            : 0;
    for (int b = 0; b < num_blocks; ++b) {
      double sum_sq = 0.0;
      const size_t offset = b * block_samples;
      for (size_t i = 0; i < block_samples; ++i) {
        const float s = raw_speech[offset + i];
        sum_sq += static_cast<double>(s) * s;
      }
      block_rms[b] = static_cast<float>(std::sqrt(sum_sq / block_samples));
      max_block_rms = std::max(max_block_rms, block_rms[b]);
      if (b >= check_start_block) {
        max_new_block_rms = std::max(max_new_block_rms, block_rms[b]);
      }
    }
    if (peak_speech_rms_ > 0.0f &&
        max_new_block_rms <=
            std::min(kTrailingSilenceMaxRms,
                     kTrailingSilenceRelativeRms * peak_speech_rms_)) {
      // Follow-up chunk consists entirely of low-level room noise relative to
      // earlier speech in the stream; skip it like silence so normalizing over
      // room noise alone does not hallucinate tokens.
      if (preprocessor_) {
        preprocessor_->Reset();
      }
      return std::vector<float>();
    }
    peak_speech_rms_ = std::max(peak_speech_rms_, max_block_rms);
    if ((ShiftsShortChunkBack() || config_.normalize_mel) &&
        valid_frames >=
            (kMinTrailingSilentBlocks + kTrailingMarginBlocks) * kBlockFrames) {
      const float silence_rms_limit = std::min(
          kTrailingSilenceMaxRms, kTrailingSilenceRelativeRms * max_block_rms);
      int last_active_block = 0;
      for (int b = 0; b < num_blocks; ++b) {
        if (block_rms[b] > silence_rms_limit) {
          last_active_block = b + 1;
        }
      }
      if (last_active_block > 0 &&
          num_blocks - last_active_block >= kMinTrailingSilentBlocks) {
        const int kept_blocks = std::max(
            kMinTrimmedBlocks, last_active_block + kTrailingMarginBlocks);
        valid_frames = std::min(valid_frames, kept_blocks * kBlockFrames);
        const size_t active_samples =
            static_cast<size_t>(valid_frames) * config_.hop_length;
        if (active_samples < raw_speech.size()) {
          std::fill(raw_speech.begin() + active_samples, raw_speech.end(),
                    0.0f);
        }
        previous_window_.clear();
        trimmed_trailing_silence = true;
      }
    }
  }

  if (window_samples > 0 && raw_speech.size() < window_samples) {
    const size_t missing = window_samples - raw_speech.size();
    if (!trimmed_trailing_silence && ShiftsShortChunkBack() &&
        (!config_.normalize_mel || missing * 6 >= window_samples) &&
        previous_window_.size() == window_samples) {
      // Final chunk of a long stream: instead of padding it, shift it back so
      // it ends exactly at the end of the audio, i.e. feed the model the last
      // `window_samples` of the stream. The model then always sees a full
      // window of real audio (padding of any kind perturbs the exported
      // encoder, which has no length mask); the decoder drops the tokens of
      // the first `missing` samples, which the previous window already
      // reported.
      std::vector<float> window;
      window.reserve(window_samples);
      window.insert(window.end(), previous_window_.end() - overlap - missing,
                    previous_window_.end() - overlap);
      window.insert(window.end(), raw_speech.begin(), raw_speech.end());
      raw_speech = std::move(window);
      previous_window_ = raw_speech;
      valid_frames = config_.n_frames;
      last_start_fraction_ =
          static_cast<float>(missing) / static_cast<float>(window_samples);
      ABSL_VLOG(1) << "LogMel chunk shifted back by " << missing
                   << " samples (overlap=" << overlap
                   << ", start_fraction=" << last_start_fraction_ << ")";
    } else {
      // Stream shorter than one window: zero-pad up to the window *before*
      // the STFT so framing is identical to a full chunk. The normalization
      // statistics below are computed over `valid_frames` only.
      raw_speech.resize(window_samples, 0.0f);
    }
  }
  if (config_.norm_type == NormType::kWhisper &&
      raw_speech.size() > static_cast<size_t>(config_.n_fft)) {
    const size_t left_pad = static_cast<size_t>(config_.n_fft / 2);
    const size_t right_pad = static_cast<size_t>(
        std::max(0, config_.n_fft - config_.hop_length - config_.n_fft / 2));
    std::vector<float> padded;
    padded.reserve(left_pad + raw_speech.size() + right_pad);
    for (size_t i = left_pad; i >= 1; --i) {
      padded.push_back(raw_speech[i]);
    }
    padded.insert(padded.end(), raw_speech.begin(), raw_speech.end());
    for (size_t i = 1; i <= right_pad; ++i) {
      padded.push_back(raw_speech[raw_speech.size() - 1 - i]);
    }
    raw_speech = std::move(padded);
  }

  // Each call to `Process` frames a self-contained (and generally overlapping
  // or shifted-back) window, so clear `AudioPreprocessorMiniAudio`'s trailing
  // `input_queue_` samples from the previous window before running the STFT.
  preprocessor_->Reset();
  litert::support::InputAudio input_audio(std::move(raw_speech));
  ABSL_ASSIGN_OR_RETURN(auto processed_audio,
                        preprocessor_->Preprocess(input_audio));
  ABSL_ASSIGN_OR_RETURN(const auto* tensor_buffer,
                        processed_audio.GetPreprocessedAudioTensor());
  LITERT_ASSIGN_OR_RETURN(auto tensor_type, tensor_buffer->TensorType());
  LITERT_ASSIGN_OR_RETURN(auto elements, tensor_type.Layout().NumElements());
  std::vector<float> log_mel_flat(elements);
  LITERT_RETURN_IF_ERROR(const_cast<litert::TensorBuffer*>(tensor_buffer)
                             ->Read<float>(absl::MakeSpan(log_mel_flat)));

  int num_frames = log_mel_flat.size() / config_.n_mels;
  const int stat_frames = std::min(valid_frames, num_frames);
  std::vector<std::vector<float>> mel_spec(config_.n_mels,
                                           std::vector<float>(num_frames));
  for (int f = 0; f < num_frames; ++f) {
    for (int m = 0; m < config_.n_mels; ++m) {
      mel_spec[m][f] = log_mel_flat[f * config_.n_mels + m];
    }
  }

  if (ABSL_VLOG_IS_ON(2)) {
    // Raw (pre-normalization) feature statistics over the valid frames, for
    // calibrating the feature pipeline against a reference implementation.
    float raw_min = std::numeric_limits<float>::infinity();
    float raw_max = -std::numeric_limits<float>::infinity();
    double raw_sum = 0.0;
    int constant_features = 0;
    for (size_t m = 0; m < mel_spec.size(); ++m) {
      float feature_min = std::numeric_limits<float>::infinity();
      float feature_max = -std::numeric_limits<float>::infinity();
      for (int f = 0; f < stat_frames; ++f) {
        const float v = mel_spec[m][f];
        feature_min = std::min(feature_min, v);
        feature_max = std::max(feature_max, v);
        raw_sum += v;
      }
      raw_min = std::min(raw_min, feature_min);
      raw_max = std::max(raw_max, feature_max);
      if (feature_max - feature_min < 1e-6f) ++constant_features;
    }
    ABSL_VLOG(2) << "LogMel raw stats over " << stat_frames
                 << " valid frames: min=" << raw_min << " max=" << raw_max
                 << " mean="
                 << (stat_frames > 0 ? raw_sum / (stat_frames * mel_spec.size())
                                     : 0.0)
                 << " constant_features=" << constant_features << "/"
                 << mel_spec.size();
  }

  if (config_.norm_type == NormType::kWhisper) {
    // log10 scale, then clip to (max - 8) and rescale. The max is taken over
    // valid frames only; padded frames sit at the mel floor and are clipped.
    float max_val = -std::numeric_limits<float>::infinity();
    for (size_t m = 0; m < mel_spec.size(); ++m) {
      for (int f = 0; f < num_frames; ++f) {
        mel_spec[m][f] = mel_spec[m][f] / kLn10;
        if (f < stat_frames && mel_spec[m][f] > max_val) {
          max_val = mel_spec[m][f];
        }
      }
    }

    float clip_val = max_val - 8.0f;
    for (size_t m = 0; m < mel_spec.size(); ++m) {
      for (int f = 0; f < num_frames; ++f) {
        mel_spec[m][f] = (std::max(mel_spec[m][f], clip_val) + 4.0f) / 4.0f;
      }
    }
  } else if (config_.normalize_mel) {
    // AudioPreprocessorMiniAudio applies kUsmMelMean and kUsmMelStdDev; set
    // padded frames to 0 (the per-feature USM mean) instead of the normalized
    // mel floor (~-3 sigma).
    for (size_t m = 0; m < mel_spec.size(); ++m) {
      for (int f = stat_frames; f < num_frames; ++f) {
        mel_spec[m][f] = 0.0f;
      }
    }
  } else {
    // Per-feature mean/std over valid frames only, as in the reference (NeMo)
    // pipeline, which normalizes each utterance over its true length.
    //
    // Only a stream shorter than one window is padded (a long stream's final
    // chunk is shifted back instead, see above). Padded frames are set to 0
    // (the per-feature mean) after normalization, like the reference's
    // zero-padded normalized features. The exported encoder has no length
    // input, so unlike the reference it does see the padding, and it is very
    // sensitive to what it sees there; of the fills tried, zeros are the only
    // one that keeps short utterances recognizable:
    //  - normalizing the padded digital silence with the speech statistics
    //    (extreme negative values: the mel floor is many sigmas below speech
    //    for low-variance features) or any constant below about -0.5 sigma
    //    makes the encoder emit only blanks for most short utterances;
    //  - continuing the utterance's own background (quietest frames) or
    //    mirroring its tail into the padding is worse still.
    // The remaining failure mode of zeros, re-emitting words over the padded
    // region, is handled by bounding the decoder to the valid frames (see
    // `Decoder::SetDecodeRange`).
    auto means = ComputeMean(mel_spec, stat_frames);
    auto stds = ComputeStd(mel_spec, means, stat_frames);
    for (size_t m = 0; m < mel_spec.size(); ++m) {
      float mean = means[m];
      float std = stds[m] + kEpsilon;
      for (int f = 0; f < num_frames; ++f) {
        mel_spec[m][f] =
            (f < stat_frames) ? (mel_spec[m][f] - mean) / std : 0.0f;
      }
    }
  }
  ABSL_VLOG(1) << "LogMel chunk: valid_frames=" << valid_frames
               << " num_frames=" << num_frames
               << " n_frames=" << config_.n_frames;

  int n_frames = (config_.n_frames > 0)
                     ? config_.n_frames
                     : (mel_spec.empty() ? 0 : mel_spec[0].size());
  // Clamp to `n_frames` (which equals `num_frames` when `config_.n_frames <= 0`
  // and `config_.n_frames` when a fixed window is configured, so an unpadded
  // full window of `n_frames * hop_length` samples reports `n_frames`).
  last_valid_frames_ = std::min(valid_frames, n_frames);
  std::vector<float> result(n_frames * config_.n_mels, 0.0f);
  for (size_t m = 0; m < mel_spec.size(); ++m) {
    for (size_t f = 0; f < mel_spec[m].size(); ++f) {
      if (f >= n_frames) break;
      float value = mel_spec[m][f];
      if (config_.transpose) {
        result[f * config_.n_mels + m] = value;
      } else {
        result[m * n_frames + f] = value;
      }
    }
  }

  return result;
}

std::vector<float> LogMelSpectrogramProcessor::last_chunk_centered_features()
    const {
  const size_t window_samples =
      config_.n_frames > 0
          ? static_cast<size_t>(config_.n_frames) * config_.hop_length
          : 0;
  const int centered_frame_length =
      std::min(config_.n_fft, sample_rate_hz_ / 40);
  if (!ShiftsShortChunkBack() || window_samples == 0 ||
      previous_window_.size() != window_samples ||
      last_valid_frames_ != config_.n_frames || centered_frame_length <= 0) {
    return {};
  }
  LogMelSpectrogramConfig centered_cfg = config_;
  centered_cfg.n_fft = centered_frame_length;
  centered_cfg.fft_length =
      config_.fft_length > 0
          ? config_.fft_length
          : GetSmallestPowerOfTwoGreaterOrEqualTo(config_.n_fft);
  auto proc = Create(sample_rate_hz_, centered_cfg, &audio_source_);
  if (!proc.ok()) return {};
  const size_t left_pad = static_cast<size_t>(centered_frame_length / 2);
  std::vector<float> padded(left_pad, 0.0f);
  padded.insert(padded.end(), previous_window_.begin(), previous_window_.end());
  padded.insert(padded.end(), centered_frame_length - left_pad, 0.0f);
  auto res = (*proc)->Process(std::move(padded));
  return res.ok() ? *std::move(res) : std::vector<float>();
}

}  // namespace litert::omni::asr
