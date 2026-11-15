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

#include "omni/speech2text/log_mel_spectrogram_processor.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
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
#include "omni/base/stage.h"
#include "omni/speech2text/audio_preprocessor.h"
#include "omni/speech2text/audio_source.h"
#include "omni/speech2text/utils.h"
#include "omni/utils/audio_utils.h"
#include "support/preprocessor/audio_preprocessor.h"
#include "support/preprocessor/audio_preprocessor_miniaudio.h"
#include "support/util/io_types.h"

namespace litert::omni::asr {
namespace {

constexpr float kEpsilon = 1e-5f;
// Minimum peak amplitude for unnormalized standard log-mel inputs before
// scaling up to prevent the power spectrum (`|X[k]|^2`) from collapsing into
// `mel_floor` (`5.96e-8`) prior to per-utterance mean/std normalization.
constexpr float kMinStandardNormPeak = 0.02f;
constexpr float kTargetStandardNormPeak = 0.1f;

// Number of 10 ms STFT frames per 100 ms RMS evaluation block.
constexpr int kBlockFrames = 10;

// Max-to-percentile ratios for steady stationary background noise.
constexpr float kStationaryMaxToP10Ratio = 1.6f;
constexpr float kStationaryMaxToMedianRatio = 1.35f;

// Thresholds for gating ultra-quiet noise (`max_rms <= kUltraQuietMaxRms`)
// while preserving multi-block ultra-quiet speech with a clear crest over a
// positive floor.
constexpr float kUltraQuietMaxRms = 8e-4f;
constexpr float kUltraQuietMinFloorRms = 2e-5f;
constexpr float kUltraQuietSpeechToP10Ratio = 3.0f;
constexpr float kUltraQuietSpeechToMedianRatio = 1.6f;

// Thresholds for stationary noise with 1-2 isolated transient clicks/taps.
constexpr size_t kMinBlocksForExtendedNoiseChecks = 10;
constexpr size_t kMaxTransientOutlierBlocks = 2;
constexpr float kMaxTransientNoiseRms = 6e-3f;

// Thresholds for low-energy ambient noise with slow level drift.
constexpr float kMaxDriftNoiseRms = 3.5e-3f;
constexpr float kDriftMaxToMinRatio = 3.0f;
constexpr float kDriftMaxToMedianRatio = 1.7f;
constexpr float kDriftMaxMeanStepToRangeRatio = 0.12f;

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

std::vector<float> ComputeBlockRms(absl::Span<const float> raw_speech,
                                   int valid_frames, int hop_length) {
  const int num_blocks = valid_frames / kBlockFrames;
  if (num_blocks <= 0 || hop_length <= 0) return {};
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
  return block_rms;
}

struct OutlierStats {
  size_t count = 0;
  bool has_adjacent = false;
};

// Counts blocks exceeding the stationary noise ceiling (`min(1.6 * p10_rms,
// 1.35 * median_rms)`) and whether any two outlier blocks are adjacent.
OutlierStats CountOutlierBlocks(absl::Span<const float> blocks, float p10_rms,
                                float median_rms) {
  const float outlier_limit =
      std::min(kStationaryMaxToP10Ratio * p10_rms,
               kStationaryMaxToMedianRatio * median_rms);
  OutlierStats stats;
  bool prev_was_outlier = false;
  for (const float b : blocks) {
    const bool is_outlier = b > outlier_limit;
    if (is_outlier) {
      ++stats.count;
      if (prev_was_outlier) stats.has_adjacent = true;
    }
    prev_was_outlier = is_outlier;
  }
  return stats;
}

// Classification of a 100 ms block-RMS span as active audio/speech vs.
// background microphone or room noise.
enum class NoiseSpanKind {
  // Active audio or speech that does not match any noise pattern below.
  kNotNoise,
  // Ultra-quiet floor (`max_rms <= kUltraQuietMaxRms`):
  // - Condition: Lacks a continuous positive floor (`p10_rms >= 2e-5`) or a
  //   multi-block speech crest (`max_rms > 3.0 * p10_rms`, `max_rms > 1.6 *
  //   median_rms`, and `> 2` outlier blocks with adjacent outliers).
  // - Rationale: Gates near-digital silence, mute-to-unmute steps, and
  //   single-block taps on an ultra-quiet floor while preserving ultra-quiet
  //   speech.
  kUltraQuiet,
  // Stationary noise (`blocks.size() >= 3` and `max_rms <= max_allowed_rms`):
  // - Condition: `max_rms <= 1.6 * p10_rms` and `max_rms <= 1.35 *
  //   median_rms`.
  // - Rationale: Steady room tone has a flat RMS distribution across blocks.
  kStationary,
  // Transient tap (`blocks.size() >= 10` and
  // `max_rms <= kMaxTransientNoiseRms`):
  // - Condition: At most 2 non-adjacent blocks exceed the stationary ceiling.
  // - Rationale: Ignores single-block mic taps without gating multi-block
  //   spoken words.
  kTransientTap,
  // Slow drift (`blocks.size() >= 10` and `max_rms <= kMaxDriftNoiseRms`):
  // - Condition: Bounded spread (`max_rms <= 3.0 * min_rms` and `1.7 *
  //   median_rms`) with small block-to-block steps (`mean_abs_delta <= 0.12 *
  //   (max_rms - min_rms)`).
  // - Rationale: HVAC or AGC drift varies smoothly, unlike syllabic speech.
  kSlowDrift,
};

NoiseSpanKind ClassifyNoiseSpan(absl::Span<const float> blocks,
                                float max_allowed_rms,
                                float* out_median_rms = nullptr) {
  if (blocks.empty()) return NoiseSpanKind::kNotNoise;
  std::vector<float> sorted(blocks.begin(), blocks.end());
  std::sort(sorted.begin(), sorted.end());
  const float min_rms = sorted.front();
  const float max_rms = sorted.back();
  const float median_rms = sorted[sorted.size() / 2];
  if (out_median_rms != nullptr) *out_median_rms = median_rms;
  const float p10_rms = sorted[sorted.size() / 10];
  if (max_rms <= kUltraQuietMaxRms) {
    if (p10_rms < kUltraQuietMinFloorRms ||
        max_rms <= kUltraQuietSpeechToP10Ratio * p10_rms ||
        max_rms <= kUltraQuietSpeechToMedianRatio * median_rms) {
      return NoiseSpanKind::kUltraQuiet;
    }
    const OutlierStats outliers =
        CountOutlierBlocks(blocks, p10_rms, median_rms);
    return (outliers.count > kMaxTransientOutlierBlocks &&
            outliers.has_adjacent)
               ? NoiseSpanKind::kNotNoise
               : NoiseSpanKind::kUltraQuiet;
  }
  if (blocks.size() < 3 || max_rms > max_allowed_rms) {
    return NoiseSpanKind::kNotNoise;
  }
  if (max_rms <= kStationaryMaxToP10Ratio * p10_rms &&
      max_rms <= kStationaryMaxToMedianRatio * median_rms) {
    return NoiseSpanKind::kStationary;
  }
  if (blocks.size() >= kMinBlocksForExtendedNoiseChecks) {
    // Stationary noise with at most 2 isolated (non-adjacent) single-block
    // transient clicks/taps above the stationary ceiling.
    if (max_rms <= kMaxTransientNoiseRms) {
      const OutlierStats outliers =
          CountOutlierBlocks(blocks, p10_rms, median_rms);
      if (outliers.count <= kMaxTransientOutlierBlocks &&
          !outliers.has_adjacent) {
        return NoiseSpanKind::kTransientTap;
      }
    }
    // Low-energy ambient noise with slow level drift across the window, which
    // lacks both the deep inter-syllable silences (`max_rms > 3.0 * min_rms`)
    // and the rapid block-to-block envelope modulation
    // (`mean_abs_delta > 0.12 * (max_rms - min_rms)`) of quiet speech.
    if (max_rms <= kMaxDriftNoiseRms &&
        max_rms <= kDriftMaxToMinRatio * min_rms &&
        max_rms <= kDriftMaxToMedianRatio * median_rms) {
      float sum_abs_delta = 0.0f;
      for (size_t i = 1; i < blocks.size(); ++i) {
        sum_abs_delta += std::abs(blocks[i] - blocks[i - 1]);
      }
      const float mean_abs_delta =
          sum_abs_delta / static_cast<float>(blocks.size() - 1);
      if (mean_abs_delta <=
          kDriftMaxMeanStepToRangeRatio * (max_rms - min_rms)) {
        return NoiseSpanKind::kSlowDrift;
      }
    }
  }
  return NoiseSpanKind::kNotNoise;
}

bool IsStationaryTailNoise(absl::Span<const float> blocks,
                           float max_allowed_rms, float* out_median_rms) {
  const NoiseSpanKind kind =
      ClassifyNoiseSpan(blocks, max_allowed_rms, out_median_rms);
  return kind == NoiseSpanKind::kUltraQuiet ||
         kind == NoiseSpanKind::kStationary;
}

// Trims >= 2 s (20 x 100 ms blocks, or 10 blocks on a warm-start prefix) of
// trailing room noise so normalizing a short utterance in a 5 s window does
// not include seconds of background silence. When `decode_valid_frames` is
// non-null and full trimming does not trigger, bounds `decode_valid_frames` to
// exclude >= 300 ms (3 blocks) of trailing silence/noise without altering
// `valid_frames`.
int TrimTrailingRoomNoise(std::vector<float>& raw_speech, int valid_frames,
                          int hop_length, int* decode_valid_frames = nullptr,
                          float peak_speech_rms = 0.0f,
                          bool is_warm_start = false,
                          int* out_trailing_silent_blocks = nullptr) {
  constexpr int kMinSilentBlocks = 20;
  constexpr int kMinWarmStartSilentBlocks = 10;
  constexpr int kMinCheckSilentBlocks = 7;
  constexpr int kMinDigitalSilentBlocks = 3;
  constexpr float kMaxStationaryNoiseRms = 0.025f;
  if (decode_valid_frames != nullptr) *decode_valid_frames = valid_frames;
  if (out_trailing_silent_blocks != nullptr) *out_trailing_silent_blocks = 0;
  if (valid_frames < (kMinDigitalSilentBlocks + 1) * kBlockFrames) {
    return valid_frames;
  }
  const std::vector<float> block_rms =
      ComputeBlockRms(raw_speech, valid_frames, hop_length);
  const int num_blocks = static_cast<int>(block_rms.size());
  if (num_blocks <= kMinDigitalSilentBlocks) return valid_frames;
  const float max_rms = *std::max_element(block_rms.begin(), block_rms.end());
  const float ref_max_rms = std::max(max_rms, peak_speech_rms);
  float limit = std::min(6e-3f, 0.05f * ref_max_rms);
  const int min_silent_blocks =
      is_warm_start ? kMinWarmStartSilentBlocks : kMinSilentBlocks;
  if (num_blocks > min_silent_blocks) {
    float tail_median_rms = 0.0f;
    const float max_allowed_tail_rms =
        std::min(kMaxStationaryNoiseRms, 0.6f * ref_max_rms);
    if (IsStationaryTailNoise(absl::MakeConstSpan(block_rms).subspan(
                                  num_blocks - min_silent_blocks),
                              max_allowed_tail_rms, &tail_median_rms)) {
      limit = std::max(limit,
                       std::min(max_allowed_tail_rms, 1.45f * tail_median_rms));
    }
  }
  if (out_trailing_silent_blocks != nullptr) {
    float silent_limit = limit;
    if (num_blocks > kMinCheckSilentBlocks) {
      float tail_short_median_rms = 0.0f;
      const float max_allowed_tail_rms =
          std::min(kMaxStationaryNoiseRms, 0.6f * ref_max_rms);
      if (IsStationaryTailNoise(absl::MakeConstSpan(block_rms).subspan(
                                    num_blocks - kMinCheckSilentBlocks),
                                max_allowed_tail_rms, &tail_short_median_rms)) {
        silent_limit = std::max(
            silent_limit,
            std::min(max_allowed_tail_rms, 1.45f * tail_short_median_rms));
      }
    }
    int last_active_for_count = 0;
    for (int b = 0; b < num_blocks; ++b) {
      if (block_rms[b] > silent_limit) last_active_for_count = b + 1;
    }
    *out_trailing_silent_blocks = num_blocks - last_active_for_count;
  }
  int last_active = 0;
  for (int b = 0; b < num_blocks; ++b) {
    if (block_rms[b] > limit) last_active = b + 1;
  }
  if (last_active > 0 && num_blocks - last_active >= min_silent_blocks) {
    valid_frames =
        std::min(valid_frames, std::max(7, last_active + 1) * kBlockFrames);
    raw_speech.resize(static_cast<size_t>(valid_frames) * hop_length);
    if (decode_valid_frames != nullptr) *decode_valid_frames = valid_frames;
    return valid_frames;
  }
  if (decode_valid_frames != nullptr) {
    float digital_limit = std::min(5e-4f, 0.01f * ref_max_rms);
    float tail3_median_rms = 0.0f;
    const float max_allowed_tail_rms =
        std::min(kMaxStationaryNoiseRms, 0.6f * ref_max_rms);
    if (IsStationaryTailNoise(absl::MakeConstSpan(block_rms).subspan(
                                  num_blocks - kMinDigitalSilentBlocks),
                              max_allowed_tail_rms, &tail3_median_rms)) {
      digital_limit =
          std::max(digital_limit,
                   std::min(max_allowed_tail_rms, 1.45f * tail3_median_rms));
    }
    int last_digital = 0;
    for (int b = 0; b < num_blocks; ++b) {
      if (block_rms[b] > digital_limit) last_digital = b + 1;
    }
    if (last_digital > 0 &&
        num_blocks - last_digital >= kMinDigitalSilentBlocks) {
      *decode_valid_frames = std::min(
          valid_frames, std::max(std::max(7, last_digital + 1) * kBlockFrames,
                                 valid_frames - 2 * kBlockFrames));
    }
  }
  return valid_frames;
}

// Returns true if a chunk contains only stationary background room noise (or
// low-level room noise relative to earlier speech in the stream), and updates
// `peak_speech_rms` with the maximum 100 ms block RMS otherwise.
bool IsRoomNoiseFollowUpChunk(absl::Span<const float> raw_speech,
                              int valid_frames, int hop_length,
                              float& peak_speech_rms) {
  constexpr float kTrailingSilenceMaxRms = 6e-3f;
  constexpr float kTrailingSilenceRelativeRms = 0.05f;
  constexpr float kMaxStationaryNoiseRms = 0.025f;
  const std::vector<float> block_rms =
      ComputeBlockRms(raw_speech, valid_frames, hop_length);
  if (block_rms.empty()) return false;
  const float max_block_rms =
      *std::max_element(block_rms.begin(), block_rms.end());
  if (peak_speech_rms > 0.0f &&
      max_block_rms <=
          std::min(kTrailingSilenceMaxRms,
                   kTrailingSilenceRelativeRms * peak_speech_rms)) {
    return true;
  }
  const float allowed_stationary_rms =
      peak_speech_rms > 0.0f ? std::min(kMaxStationaryNoiseRms,
                                        std::max(0.02f, 0.6f * peak_speech_rms))
                             : 0.02f;
  if ((block_rms.size() >= 5 || max_block_rms <= kUltraQuietMaxRms) &&
      ClassifyNoiseSpan(block_rms, allowed_stationary_rms) !=
          NoiseSpanKind::kNotNoise) {
    return true;
  }
  if (peak_speech_rms > 0.0f && block_rms.size() >= 20 &&
      max_block_rms <= 0.5f * peak_speech_rms &&
      ClassifyNoiseSpan(absl::MakeConstSpan(block_rms).subspan(
                            block_rms.size() >= 45 ? 4 : 2),
                        std::min(kMaxStationaryNoiseRms,
                                 std::max(0.02f, 0.5f * peak_speech_rms))) !=
          NoiseSpanKind::kNotNoise) {
    return true;
  }
  peak_speech_rms = std::max(peak_speech_rms, max_block_rms);
  return false;
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

litert::support::AudioPreprocessorConfig CreateMiniAudioConfig(
    int sample_rate_hz,
    const LogMelSpectrogramProcessor::LogMelSpectrogramConfig& config,
    int frame_length) {
  if (config.norm_type == LogMelSpectrogramProcessor::NormType::kWhisper &&
      config.fft_length <= 0 && config.input_scale == 1.0f &&
      config.preemphasis == 0.0f && config.mel_low_hz == 0.0f &&
      config.mel_high_hz <= 0.0f && !config.normalize_mel &&
      config.add_floor_to_mel_before_log) {
    litert::support::AudioPreprocessorConfig preprocessor_config =
        CreateWhisperAudioPreprocessorConfig(sample_rate_hz, frame_length,
                                             config.hop_length, config.n_mels,
                                             config.mel_floor);
    preprocessor_config.SetSlaneyMel(config.slaney_mel);
    preprocessor_config.SetPowerSpectrum(config.power_spectrum);
    return preprocessor_config;
  }
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

// Scales active samples `[0, valid_frames * hop_length)` of quiet unnormalized
// standard log-mel inputs (`0 < peak < kMinStandardNormPeak`) up to
// `kTargetStandardNormPeak` so that the STFT power spectrum (`|X[k]|^2`) does
// not collapse into `mel_floor` before per-utterance mean/std normalization.
void ScaleQuietStandardNormAudio(
    const LogMelSpectrogramProcessor::LogMelSpectrogramConfig& config,
    std::vector<float>& raw_speech, int valid_frames) {
  if (config.norm_type != LogMelSpectrogramProcessor::NormType::kStandard ||
      config.normalize_mel || valid_frames <= 0 || config.hop_length <= 0) {
    return;
  }
  const size_t active_samples =
      std::min(raw_speech.size(), static_cast<size_t>(valid_frames) *
                                      static_cast<size_t>(config.hop_length));
  float peak = 0.0f;
  for (size_t i = 0; i < active_samples; ++i) {
    peak = std::max(peak, std::abs(raw_speech[i]));
  }
  if (peak > 0.0f && peak < kMinStandardNormPeak) {
    const float gain = kTargetStandardNormPeak / peak;
    for (size_t i = 0; i < active_samples; ++i) {
      raw_speech[i] *= gain;
    }
  }
}

absl::StatusOr<std::vector<float>> ExtractMelFeatures(
    litert::support::AudioPreprocessorMiniAudio& preprocessor,
    const LogMelSpectrogramProcessor::LogMelSpectrogramConfig& config,
    std::vector<float> raw_speech, int valid_frames) {
  preprocessor.Reset();
  ScaleQuietStandardNormAudio(config, raw_speech, valid_frames);
  litert::support::InputAudio input_audio(std::move(raw_speech));
  ABSL_ASSIGN_OR_RETURN(auto processed_audio,
                        preprocessor.Preprocess(input_audio));
  ABSL_ASSIGN_OR_RETURN(const auto* tensor_buffer,
                        processed_audio.GetPreprocessedAudioTensor());
  LITERT_ASSIGN_OR_RETURN(auto tensor_type, tensor_buffer->TensorType());
  LITERT_ASSIGN_OR_RETURN(auto elements, tensor_type.Layout().NumElements());
  const int num_frames = static_cast<int>(elements) / config.n_mels;
  const int stat_frames = std::min(valid_frames, num_frames);
  std::vector<std::vector<float>> mel_spec(config.n_mels,
                                           std::vector<float>(num_frames));
  {
    LITERT_ASSIGN_OR_RETURN(
        auto lock_and_addr,
        ::litert::TensorBufferScopedLock::Create<const float>(
            *const_cast<litert::TensorBuffer*>(tensor_buffer),
            ::litert::TensorBuffer::LockMode::kRead));
    const float* log_mel_flat = lock_and_addr.second;
    for (int f = 0; f < num_frames; ++f) {
      for (int m = 0; m < config.n_mels; ++m) {
        mel_spec[m][f] = log_mel_flat[f * config.n_mels + m];
      }
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

absl::StatusOr<std::unique_ptr<LogMelSpectrogramProcessor>>
LogMelSpectrogramProcessor::Create(int sample_rate_hz,
                                   const LogMelSpectrogramConfig& config,
                                   AudioSource* absl_nonnull audio_source) {
  ABSL_ASSIGN_OR_RETURN(
      auto processor,
      Create(sample_rate_hz, config,
             static_cast<Stage<std::vector<float>>*>(audio_source)));
  processor->typed_audio_source_ = audio_source;
  return processor;
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
  prev_trailing_silent_blocks_ = 0;
}

absl::Status LogMelSpectrogramProcessor::ScheduleInternal() {
  auto cleanup = absl::MakeCleanup([this]() { SetState(State::kIdle); });
  ABSL_ASSIGN_OR_RETURN(auto pcm_samples, audio_source_.GetOutput());
  const bool is_warm_start = IsLastInputWarmStart();
  ABSL_ASSIGN_OR_RETURN(auto output,
                        Process(std::move(pcm_samples), is_warm_start));
  output.is_warm_start = is_warm_start;
  PushOutput(std::move(output));
  return absl::OkStatus();
}

absl::StatusOr<PreprocessedAudio> LogMelSpectrogramProcessor::Process(
    std::vector<float> raw_speech, bool is_warm_start) {
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
    if (!is_warm_start) {
      prev_trailing_silent_blocks_ =
          static_cast<int>(raw_speech.size() / config_.hop_length) / 10;
    }
    return PreprocessedAudio{};
  }

  int valid_frames = raw_speech.size() / config_.hop_length;
  const bool is_short_chunk =
      window_samples_ > 0 && raw_speech.size() < window_samples_;
  float short_chunk_peak_rms = peak_speech_rms_;
  if (IsRoomNoiseFollowUpChunk(
          raw_speech, valid_frames, config_.hop_length,
          is_short_chunk ? short_chunk_peak_rms : peak_speech_rms_)) {
    preprocessor_->Reset();
    if (centered_preprocessor_) {
      centered_preprocessor_->Reset();
    }
    previous_window_.clear();
    if (!is_warm_start && !is_short_chunk) {
      prev_trailing_silent_blocks_ = valid_frames / 10;
    }
    return PreprocessedAudio{};
  }

  if (ShiftsShortChunkBack() && !is_short_chunk) {
    // Keep the most recent full non-silent window so a subsequent short final
    // chunk can be shifted back over it.
    previous_window_.assign(raw_speech.end() - window_samples_,
                            raw_speech.end());
  }

  bool trimmed_trailing_silence = false;
  int decode_valid_frames = valid_frames;
  int trailing_silent_blocks = 0;
  // Only trim trailing room noise for kStandard (via ShiftsShortChunkBack())
  // and normalize_mel, where per-utterance statistics are computed over
  // [0, stat_frames) and frames >= stat_frames are zeroed out for encoders
  // without a length mask. Whisper (kWhisper) normalizes by the global peak and
  // is trained on 30 s windows where padded silence is clipped to
  // (max_val - 4.0) / 4.0 rather than 0.0.
  if (ShiftsShortChunkBack() || config_.normalize_mel) {
    const int trimmed = TrimTrailingRoomNoise(
        raw_speech, valid_frames, config_.hop_length,
        (is_short_chunk && ShiftsShortChunkBack()) ? &decode_valid_frames
                                                   : nullptr,
        peak_speech_rms_, is_warm_start, &trailing_silent_blocks);
    if (!is_warm_start && !is_short_chunk &&
        prev_trailing_silent_blocks_ >= 7 && trailing_silent_blocks >= 20 &&
        trailing_silent_blocks >= prev_trailing_silent_blocks_ + 8) {
      prev_trailing_silent_blocks_ = trailing_silent_blocks;
      preprocessor_->Reset();
      if (centered_preprocessor_) {
        centered_preprocessor_->Reset();
      }
      previous_window_.clear();
      return PreprocessedAudio{};
    }
    if (!is_warm_start && !is_short_chunk) {
      prev_trailing_silent_blocks_ = trailing_silent_blocks;
    }
    if (trimmed < valid_frames) {
      valid_frames = trimmed;
      decode_valid_frames = trimmed;
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
      decode_valid_frames = valid_frames;
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
  const int bounded_decode_valid_frames =
      std::min(decode_valid_frames, n_frames);
  const float valid_fraction =
      (config_.n_frames > 0 && bounded_decode_valid_frames > 0)
          ? std::min(1.0f, static_cast<float>(bounded_decode_valid_frames) /
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
