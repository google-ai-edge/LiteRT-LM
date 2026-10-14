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
#include <optional>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
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

// Returns the FFT size for STFT computation. If config.fft_length is unset,
// Whisper uses config.n_fft directly (e.g., 400 -> 201 frequency bins) to
// match WhisperFeatureExtractor's Slaney mel filterbank resolution, whereas
// other models round config.n_fft up to the next power of two.
int GetFftLength(
    const LogMelSpectrogramProcessor::LogMelSpectrogramConfig& config) {
  if (config.fft_length > 0) {
    return config.fft_length;
  }
  if (config.norm_type == LogMelSpectrogramProcessor::NormType::kWhisper) {
    return config.n_fft;
  }
  return GetSmallestPowerOfTwoGreaterOrEqualTo(config.n_fft);
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

// Trims >= 2 s (20 x 100 ms blocks) of trailing room noise (< 6e-3 RMS and
// < 5% of peak block RMS) so normalizing a short utterance in a 5 s window
// does not include seconds of background silence.
int TrimTrailingRoomNoise(std::vector<float>& raw_speech, int valid_frames,
                          int hop_length) {
  constexpr int kBlockFrames = 10;
  constexpr int kMinSilentBlocks = 20;
  if (valid_frames < (kMinSilentBlocks + 1) * kBlockFrames) {
    return valid_frames;
  }
  const int num_blocks = valid_frames / kBlockFrames;
  const size_t block_samples = static_cast<size_t>(kBlockFrames) * hop_length;
  std::vector<float> block_rms(num_blocks, 0.0f);
  for (int b = 0; b < num_blocks; ++b) {
    double sum_sq = 0.0;
    for (size_t i = 0; i < block_samples; ++i) {
      const float s = raw_speech[b * block_samples + i];
      sum_sq += static_cast<double>(s) * s;
    }
    block_rms[b] = static_cast<float>(std::sqrt(sum_sq / block_samples));
  }
  const float limit = std::min(
      6e-3f, 0.05f * *std::max_element(block_rms.begin(), block_rms.end()));
  int last_active = 0;
  for (int b = 0; b < num_blocks; ++b) {
    if (block_rms[b] > limit) last_active = b + 1;
  }
  if (last_active > 0 && num_blocks - last_active >= kMinSilentBlocks) {
    valid_frames =
        std::min(valid_frames, std::max(7, last_active + 1) * kBlockFrames);
    raw_speech.resize(static_cast<size_t>(valid_frames) * hop_length);
  }
  return valid_frames;
}

// Applies center=True reflect padding around raw_speech so each STFT window of
// length n_fft is centered at frame * hop_length, matching
// WhisperFeatureExtractor (torch.stft(..., center=True,
// pad_mode="reflect")[:, :-1]). Standard/USM models (kStandard / normalize_mel)
// use unpadded causal STFT framing directly from AudioPreprocessorMiniAudio
// without pre-centering.
void PutSamplesInCenter(std::vector<float>& raw_speech, int n_fft,
                        int hop_length) {
  if (raw_speech.size() <= static_cast<size_t>(n_fft)) {
    return;
  }
  const size_t left_pad = static_cast<size_t>(n_fft / 2);
  const size_t right_pad =
      static_cast<size_t>(std::max(0, n_fft - hop_length - n_fft / 2));
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

// Returns true if a follow-up chunk (`peak_speech_rms > 0`) contains only
// low-level background room noise relative to earlier speech in the stream, and
// updates `peak_speech_rms` with the maximum 100 ms block RMS otherwise.
bool IsRoomNoiseFollowUpChunk(absl::Span<const float> raw_speech,
                              int valid_frames, int hop_length,
                              float& peak_speech_rms) {
  constexpr int kBlockFrames = 10;
  constexpr float kTrailingSilenceMaxRms = 6e-3f;
  constexpr float kTrailingSilenceRelativeRms = 0.05f;
  const int num_blocks = valid_frames / kBlockFrames;
  if (num_blocks <= 0) return false;
  const size_t block_samples = static_cast<size_t>(kBlockFrames) * hop_length;
  float max_block_rms = 0.0f;
  for (int b = 0; b < num_blocks; ++b) {
    double sum_sq = 0.0;
    for (size_t i = 0; i < block_samples; ++i) {
      const float s = raw_speech[b * block_samples + i];
      sum_sq += static_cast<double>(s) * s;
    }
    const float rms = static_cast<float>(std::sqrt(sum_sq / block_samples));
    max_block_rms = std::max(max_block_rms, rms);
  }
  if (peak_speech_rms > 0.0f &&
      max_block_rms <=
          std::min(kTrailingSilenceMaxRms,
                   kTrailingSilenceRelativeRms * peak_speech_rms)) {
    return true;
  }
  peak_speech_rms = std::max(peak_speech_rms, max_block_rms);
  return false;
}

// Shifts a short final chunk `raw_speech` back over `previous_window` to form a
// full `window_samples`-length window ending at `raw_speech.end()`,
// deduplicating any overlap prefix already shared with the tail of
// `previous_window`. Updates `previous_window` to the reconstructed window and
// returns the fraction of the window that precedes the new samples.
float ShiftShortChunkBackOverPreviousWindow(std::vector<float>& raw_speech,
                                            std::vector<float>& previous_window,
                                            size_t window_samples,
                                            int hop_length) {
  const size_t missing = window_samples - raw_speech.size();
  // Overlapping audio sources (`AudioInputSource`, `FileAudioSource`) prefix
  // follow-up chunks with the retained overlap samples from the end of
  // `previous_window`. Identify that exact overlap so shifting a short final
  // chunk back to a full window splices the contiguous samples preceding
  // `raw_speech.begin()` rather than duplicating the overlap.
  size_t overlap = 0;
  const size_t min_overlap = static_cast<size_t>(hop_length);
  for (size_t candidate = raw_speech.size(); candidate >= min_overlap;
       --candidate) {
    if (std::equal(previous_window.end() - candidate, previous_window.end(),
                   raw_speech.begin())) {
      overlap = candidate;
      break;
    }
  }
  std::vector<float> window;
  window.reserve(window_samples);
  window.insert(window.end(), previous_window.end() - overlap - missing,
                previous_window.end() - overlap);
  window.insert(window.end(), raw_speech.begin(), raw_speech.end());
  raw_speech = std::move(window);
  previous_window = raw_speech;
  return static_cast<float>(missing) / static_cast<float>(window_samples);
}

void NormalizeWhisperMelSpectrogram(std::vector<std::vector<float>>& mel_spec,
                                    int stat_frames, int num_frames) {
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
}

void ZeroPadNormalizedMelSpectrogram(std::vector<std::vector<float>>& mel_spec,
                                     int stat_frames, int num_frames) {
  // AudioPreprocessorMiniAudio applies fixed kUsmMelMean and kUsmMelStdDev
  // normalization across all frames, turning zero-padded audio in
  // [stat_frames, num_frames) into non-zero negative values
  // ((log(mel_floor) - mean) / std). Overwrite the padded tail with 0.0f so it
  // represents the normalized mean and does not distort encoders without a
  // length mask.
  for (size_t m = 0; m < mel_spec.size(); ++m) {
    for (int f = stat_frames; f < num_frames; ++f) {
      mel_spec[m][f] = 0.0f;
    }
  }
}

void NormalizeStandardMelSpectrogram(std::vector<std::vector<float>>& mel_spec,
                                     int stat_frames, int num_frames) {
  auto means = ComputeMean(mel_spec, stat_frames);
  auto stds = ComputeStd(mel_spec, means, stat_frames);
  for (size_t m = 0; m < mel_spec.size(); ++m) {
    float mean = means[m];
    float std = stds[m] + kEpsilon;
    for (int f = 0; f < num_frames; ++f) {
      mel_spec[m][f] = (f < stat_frames) ? (mel_spec[m][f] - mean) / std : 0.0f;
    }
  }
}

std::vector<float> FlattenMelSpectrogram(
    const std::vector<std::vector<float>>& mel_spec, int n_frames, int n_mels,
    bool transpose) {
  std::vector<float> result(n_frames * n_mels, 0.0f);
  for (size_t m = 0; m < mel_spec.size(); ++m) {
    for (size_t f = 0;
         f < mel_spec[m].size() && f < static_cast<size_t>(n_frames); ++f) {
      if (transpose) {
        result[f * n_mels + m] = mel_spec[m][f];
      } else {
        result[m * n_frames + f] = mel_spec[m][f];
      }
    }
  }
  return result;
}

litert::support::AudioPreprocessorConfig CreateMiniAudioConfig(
    int sample_rate_hz,
    const LogMelSpectrogramProcessor::LogMelSpectrogramConfig& config,
    int frame_length) {
  const float mel_high_hz =
      config.mel_high_hz > 0.0f ? config.mel_high_hz : sample_rate_hz / 2.0f;
  litert::support::AudioPreprocessorConfig preprocessor_config =
      litert::support::AudioPreprocessorConfig::Create(
          sample_rate_hz, /*num_channels=*/1,
          /*frame_length=*/frame_length,
          /*hop_length=*/config.hop_length,
          /*fft_length=*/GetFftLength(config),
          /*input_scale=*/config.input_scale,
          /*pre_emphasis_factor=*/config.preemphasis,
          /*num_mel_bins=*/config.n_mels,
          /*mel_low_hz=*/config.mel_low_hz,
          /*mel_high_hz=*/mel_high_hz,
          /*mel_floor=*/config.mel_floor,
          /*normalize_mel=*/config.normalize_mel,
          /*add_floor_to_mel_before_log=*/config.add_floor_to_mel_before_log,
          /*semicausal_padding=*/false,
          /*non_zero_hanning=*/
          config.norm_type != LogMelSpectrogramProcessor::NormType::kWhisper,
          /*periodic_hanning=*/true,
          litert::support::AudioPreprocessorConfig::FftPaddingType::kRight);
  preprocessor_config.SetSlaneyMel(config.slaney_mel);
  preprocessor_config.SetPowerSpectrum(config.power_spectrum);
  return preprocessor_config;
}

absl::StatusOr<std::vector<float>> ExtractMelFeatures(
    litert::support::AudioPreprocessorMiniAudio& preprocessor,
    const LogMelSpectrogramProcessor::LogMelSpectrogramConfig& config,
    std::vector<float> raw_speech, int valid_frames) {
  preprocessor.Reset();
  litert::support::InputAudio input_audio(std::move(raw_speech));
  ABSL_ASSIGN_OR_RETURN(auto processed_audio,
                        preprocessor.Preprocess(input_audio));
  ABSL_ASSIGN_OR_RETURN(const auto* tensor_buffer,
                        processed_audio.GetPreprocessedAudioTensor());
  LITERT_ASSIGN_OR_RETURN(auto tensor_type, tensor_buffer->TensorType());
  LITERT_ASSIGN_OR_RETURN(auto elements, tensor_type.Layout().NumElements());
  std::vector<float> log_mel_flat(elements);
  LITERT_RETURN_IF_ERROR(const_cast<litert::TensorBuffer*>(tensor_buffer)
                             ->Read<float>(absl::MakeSpan(log_mel_flat)));

  const int num_frames = log_mel_flat.size() / config.n_mels;
  const int stat_frames = std::min(valid_frames, num_frames);
  std::vector<std::vector<float>> mel_spec(config.n_mels,
                                           std::vector<float>(num_frames));
  for (int f = 0; f < num_frames; ++f) {
    for (int m = 0; m < config.n_mels; ++m) {
      mel_spec[m][f] = log_mel_flat[f * config.n_mels + m];
    }
  }

  if (config.norm_type == LogMelSpectrogramProcessor::NormType::kWhisper) {
    NormalizeWhisperMelSpectrogram(mel_spec, stat_frames, num_frames);
  } else if (config.normalize_mel) {
    ZeroPadNormalizedMelSpectrogram(mel_spec, stat_frames, num_frames);
  } else {
    NormalizeStandardMelSpectrogram(mel_spec, stat_frames, num_frames);
  }

  const int n_frames = (config.n_frames > 0)
                           ? config.n_frames
                           : (mel_spec.empty() ? 0 : mel_spec[0].size());
  return FlattenMelSpectrogram(mel_spec, n_frames, config.n_mels,
                               config.transpose);
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

  ABSL_ASSIGN_OR_RETURN(
      auto preprocessor,
      litert::support::AudioPreprocessorMiniAudio::Create(
          CreateMiniAudioConfig(sample_rate_hz, config, config.n_fft)));
  std::unique_ptr<litert::support::AudioPreprocessorMiniAudio>
      centered_preprocessor;
  const int centered_frame_length = std::min(config.n_fft, sample_rate_hz / 40);
  if (config.n_frames > 0 && config.norm_type == NormType::kStandard &&
      centered_frame_length > 0) {
    ABSL_ASSIGN_OR_RETURN(centered_preprocessor,
                          litert::support::AudioPreprocessorMiniAudio::Create(
                              CreateMiniAudioConfig(sample_rate_hz, config,
                                                    centered_frame_length)));
  }
  return absl::WrapUnique(new LogMelSpectrogramProcessor(
      sample_rate_hz, config, audio_source, std::move(preprocessor),
      std::move(centered_preprocessor)));
}

LogMelSpectrogramProcessor::LogMelSpectrogramProcessor(
    int sample_rate_hz, const LogMelSpectrogramConfig& config,
    Stage<std::vector<float>>* absl_nonnull audio_source,
    std::unique_ptr<litert::support::AudioPreprocessorMiniAudio> absl_nonnull
    preprocessor,
    std::unique_ptr<litert::support::AudioPreprocessorMiniAudio> absl_nullable
    centered_preprocessor)
    : AudioPreprocessor(audio_source),
      sample_rate_hz_(sample_rate_hz),
      config_(config),
      window_samples_(config_.n_frames > 0
                          ? static_cast<size_t>(config_.n_frames) *
                                config_.hop_length
                          : 0),
      preprocessor_(std::move(preprocessor)),
      centered_preprocessor_(std::move(centered_preprocessor)) {}

bool LogMelSpectrogramProcessor::ShiftsShortChunkBack() const {
  return config_.n_frames > 0 && config_.norm_type == NormType::kStandard;
}

void LogMelSpectrogramProcessor::ResetInternal() {
  if (preprocessor_) {
    preprocessor_->Reset();
  }
  if (centered_preprocessor_) {
    centered_preprocessor_->Reset();
  }
  previous_window_.clear();
  peak_speech_rms_ = 0.0f;
}

absl::Status LogMelSpectrogramProcessor::ScheduleInternal() {
  auto cleanup = absl::MakeCleanup([this]() { SetState(State::kIdle); });
  ABSL_ASSIGN_OR_RETURN(auto pcm_samples, audio_source_.GetOutput());
  ABSL_ASSIGN_OR_RETURN(auto output, Process(std::move(pcm_samples)));
  PushOutput(std::move(output));
  return absl::OkStatus();
}

absl::StatusOr<PreprocessedAudio> LogMelSpectrogramProcessor::Process(
    std::vector<float> raw_speech) {
  if (!preprocessor_) {
    return absl::InternalError(
        "AudioPreprocessorMiniAudio is not initialized.");
  }

  if (IsSilentAudio(raw_speech)) {
    // AudioPreprocessorMiniAudio buffers trailing PCM samples in its internal
    // input_queue_ across calls for STFT window overlap. Reset it and clear
    // `previous_window_` when skipping a silent chunk so stale pre-silence
    // samples are not stitched onto the start of the next non-silent chunk.
    preprocessor_->Reset();
    if (centered_preprocessor_) {
      centered_preprocessor_->Reset();
    }
    previous_window_.clear();
    return PreprocessedAudio{};
  }

  int valid_frames = raw_speech.size() / config_.hop_length;
  if (IsRoomNoiseFollowUpChunk(raw_speech, valid_frames, config_.hop_length,
                               peak_speech_rms_)) {
    preprocessor_->Reset();
    if (centered_preprocessor_) {
      centered_preprocessor_->Reset();
    }
    previous_window_.clear();
    return PreprocessedAudio{};
  }

  const bool is_short_chunk =
      window_samples_ > 0 && raw_speech.size() < window_samples_;
  if (ShiftsShortChunkBack() && !is_short_chunk) {
    // Keep the most recent full non-silent window so a subsequent short final
    // chunk can be shifted back over it.
    previous_window_.assign(raw_speech.end() - window_samples_,
                            raw_speech.end());
  }

  bool trimmed_trailing_silence = false;
  // Only trim trailing room noise for kStandard (via ShiftsShortChunkBack())
  // and normalize_mel, where per-utterance statistics are computed over
  // [0, stat_frames) and frames >= stat_frames are zeroed out for encoders
  // without a length mask. Whisper (kWhisper) normalizes by the global peak and
  // is trained on 30 s windows where padded silence is clipped to
  // (max_val - 4.0) / 4.0 rather than 0.0.
  if (ShiftsShortChunkBack() || config_.normalize_mel) {
    const int trimmed =
        TrimTrailingRoomNoise(raw_speech, valid_frames, config_.hop_length);
    if (trimmed < valid_frames) {
      valid_frames = trimmed;
      trimmed_trailing_silence = true;
      if (!is_short_chunk) {
        previous_window_.clear();
      }
    }
  }

  float start_fraction = 0.0f;
  if (window_samples_ > 0 && raw_speech.size() < window_samples_) {
    const size_t missing = window_samples_ - raw_speech.size();
    if (is_short_chunk && ShiftsShortChunkBack() &&
        (!config_.normalize_mel || missing * 6 >= window_samples_) &&
        previous_window_.size() == window_samples_) {
      start_fraction = ShiftShortChunkBackOverPreviousWindow(
          raw_speech, previous_window_, window_samples_, config_.hop_length);
      valid_frames = config_.n_frames;
    } else {
      raw_speech.resize(window_samples_, 0.0f);
    }
  }
  if (config_.norm_type == NormType::kWhisper) {
    PutSamplesInCenter(raw_speech, config_.n_fft, config_.hop_length);
  }

  ABSL_ASSIGN_OR_RETURN(
      std::vector<float> features,
      ExtractMelFeatures(*preprocessor_, config_, std::move(raw_speech),
                         valid_frames));

  const int n_frames = (config_.n_frames > 0)
                           ? config_.n_frames
                           : static_cast<int>(features.size() / config_.n_mels);
  // Record the active audio span `[start_fraction, valid_fraction)` so
  // timestamped decoders can skip already-emitted prefix frames on shifted-back
  // final chunks (`start_fraction > 0`) and padded tail frames on short chunks
  // (`valid_fraction < 1`).
  const int bounded_valid_frames = std::min(valid_frames, n_frames);
  const float valid_fraction =
      (config_.n_frames > 0 && bounded_valid_frames > 0)
          ? std::min(1.0f, static_cast<float>(bounded_valid_frames) /
                               static_cast<float>(config_.n_frames))
          : 1.0f;
  // For full valid windows on processors that shift short chunks back, also
  // compute centered 25 ms STFT features so the recognizer can retry if the
  // primary unpadded 512-sample STFT pass collapses at the window boundaries.
  std::optional<std::vector<float>> centered_features;
  if (ShiftsShortChunkBack() && bounded_valid_frames == config_.n_frames) {
    centered_features = ComputeCenteredFeatures();
  }
  if (trimmed_trailing_silence) {
    previous_window_.clear();
  }
  return PreprocessedAudio{
      .features = std::move(features),
      .centered_features = std::move(centered_features),
      .decode_range = DecodeRange{.start_fraction = start_fraction,
                                  .valid_fraction = valid_fraction},
  };
}

std::optional<std::vector<float>>
LogMelSpectrogramProcessor::ComputeCenteredFeatures() {
  const int centered_frame_length =
      std::min(config_.n_fft, sample_rate_hz_ / 40);
  if (!centered_preprocessor_ || window_samples_ == 0 ||
      previous_window_.size() != window_samples_ ||
      centered_frame_length <= 0) {
    return std::nullopt;
  }

  const size_t left_pad = static_cast<size_t>(centered_frame_length / 2);
  std::vector<float> padded(left_pad, 0.0f);
  padded.insert(padded.end(), previous_window_.begin(), previous_window_.end());
  padded.insert(padded.end(), centered_frame_length - left_pad, 0.0f);
  const int valid_frames = padded.size() / config_.hop_length;
  auto features = ExtractMelFeatures(*centered_preprocessor_, config_,
                                     std::move(padded), valid_frames);
  if (!features.ok() || features->empty()) {
    return std::nullopt;
  }
  return *std::move(features);
}

}  // namespace litert::omni::asr
