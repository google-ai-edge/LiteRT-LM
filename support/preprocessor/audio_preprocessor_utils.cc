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

#include "support/preprocessor/audio_preprocessor_utils.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "support/preprocessor/audio_preprocessor.h"
#include "support/preprocessor/mel_filterbank.h"
#include "support/util/status_macros.h"

namespace litert::support {

absl::Status PadOrTruncateForFft(
    int fft_length, AudioPreprocessorConfig::FftPaddingType padding_type,
    std::vector<float>& frame) {
  int input_dim = frame.size();
  if (input_dim == fft_length) {
    return absl::OkStatus();
  }

  if (input_dim < fft_length) {
    int pad_amount = fft_length - input_dim;
    if (padding_type == AudioPreprocessorConfig::FftPaddingType::kCenter) {
      int pad_left = pad_amount / 2;
      int pad_right = pad_amount - pad_left;
      frame.insert(frame.begin(), pad_left, 0.0f);
      frame.insert(frame.end(), pad_right, 0.0f);
    } else if (padding_type ==
               AudioPreprocessorConfig::FftPaddingType::kRight) {
      frame.resize(fft_length, 0.0f);
    } else {
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported padding: ", padding_type));
    }
  } else {
    int trim_left = 0;
    if (padding_type == AudioPreprocessorConfig::FftPaddingType::kCenter) {
      trim_left = (input_dim - fft_length) / 2;
    } else if (padding_type ==
               AudioPreprocessorConfig::FftPaddingType::kRight) {
      trim_left = 0;
    } else {
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported padding: ", padding_type));
    }
    frame.erase(frame.begin(), frame.begin() + trim_left);
    frame.resize(fft_length);
  }
  return absl::OkStatus();
}

std::vector<float> GetHanningWindow(int window_length,
                                    bool use_periodic_hanning,
                                    bool non_zero_hanning) {
  if (window_length <= 0) {
    return {};
  }
  const int n = (use_periodic_hanning && (window_length % 2 == 0))
                    ? window_length
                    : window_length - 1;
  float arg = M_PI * 2.0 / n;
  std::vector<float> hanning_window;
  hanning_window.reserve(window_length);
  const float shift = non_zero_hanning ? 0.5 : 0.0;
  for (int i = 0; i < window_length; ++i) {
    hanning_window.push_back(0.5 - (0.5 * cos(arg * (i + shift))));
  }
  return hanning_window;
}

bool GetNextWindowOfSamples(const AudioPreprocessorConfig& config,
                            const std::vector<float>& pcm_frames,
                            int& input_start, std::vector<float>& input_queue,
                            int& samples_to_next_step) {
  auto input_it = pcm_frames.begin() + input_start;
  int input_remaining = pcm_frames.end() - input_it;
  if (samples_to_next_step > input_remaining) {
    // Copy in as many samples are left and return false, no full window.
    input_queue.insert(input_queue.end(), input_it, pcm_frames.end());
    input_start += input_remaining;  // Increases it to input.size().
    samples_to_next_step -= input_remaining;
    return false;  // Not enough for a full window.
  } else {
    // Copy just enough into queue to make a new window.
    if (samples_to_next_step < config.GetFrameLength()) {
      input_queue.erase(input_queue.begin(),
                        input_queue.begin() + input_queue.size() -
                            (config.GetFrameLength() - samples_to_next_step));
      input_queue.insert(input_queue.end(), input_it,
                         input_it + samples_to_next_step);
    } else {
      input_queue.assign(
          input_it + samples_to_next_step - config.GetFrameLength(),
          input_it + samples_to_next_step);
    }
    input_start += samples_to_next_step;
    samples_to_next_step = config.GetHopLength();  // Be ready for next step.
    return true;  // Yes, input_queue now contains exactly a window-full.
  }
}

absl::StatusOr<std::vector<std::vector<float>>> GetFramedSegments(
    const AudioPreprocessorConfig& config, const std::vector<float>& pcm_frames,
    std::vector<float>& input_queue, int& samples_to_next_step) {
  std::vector<std::vector<float>> windowed_signals;
  int input_start = 0;
  while (GetNextWindowOfSamples(config, pcm_frames, input_start, input_queue,
                                samples_to_next_step)) {
    if (input_queue.size() != config.GetFrameLength()) {
      return absl::InternalError(absl::StrCat(
          "Input queue size is not equal to frame length: ", input_queue.size(),
          " vs ", config.GetFrameLength()));
    }
    windowed_signals.push_back(input_queue);
  }

  if (!config.BufferLastFrame() && !input_queue.empty() &&
      input_queue.size() < config.GetFrameLength()) {
    input_queue.resize(config.GetFrameLength(), 0.0f);
    windowed_signals.push_back(input_queue);
    input_queue.clear();
    if (config.GetSemicausalPadding()) {
      samples_to_next_step = config.GetFrameLength() - config.GetHopLength();
      input_queue.resize(config.GetHopLength(), 0.0f);
    } else {
      samples_to_next_step = config.GetFrameLength();
    }
  }
  return windowed_signals;
}

absl::StatusOr<std::vector<std::vector<float>>> GetWindowedSignalsForFft(
    const AudioPreprocessorConfig& config, absl::Span<const float> pcm_frames,
    std::vector<float>& input_queue, int& samples_to_next_step) {
  const float input_scale = config.GetInputScale();
  const float pre_emphasis_factor = config.GetPreEmphasisFactor();
  std::vector<float> scaled_pcm_frames(pcm_frames.size(), 0);
  absl::c_transform(pcm_frames, scaled_pcm_frames.begin(),
                    [&input_scale](float x) { return x * input_scale; });

  LITERT_ASSIGN_OR_RETURN(auto raw_windowed_signals,
                          GetFramedSegments(config, scaled_pcm_frames,
                                            input_queue, samples_to_next_step));

  std::vector<std::vector<float>> windowed_signals;
  windowed_signals.reserve(raw_windowed_signals.size());
  for (int i = 0; i < raw_windowed_signals.size(); ++i) {
    const auto& input_frame = raw_windowed_signals[i];
    windowed_signals.push_back(std::vector<float>(config.GetFrameLength(), 0));
    std::vector<float>& current_frame = windowed_signals.back();
    current_frame[0] = input_frame[0] * (1 - pre_emphasis_factor);
    for (int j = 1; j < config.GetFrameLength(); ++j) {
      current_frame[j] =
          input_frame[j] - pre_emphasis_factor * input_frame[j - 1];
    }
  }
  const std::vector<float> hanning_window =
      GetHanningWindow(config.GetFrameLength(), config.GetPeriodicHanning(),
                       config.GetNonZeroHanning());
  for (int i = 0; i < windowed_signals.size(); ++i) {
    std::vector<float>& current_frame = windowed_signals[i];
    for (int j = 0; j < current_frame.size(); ++j) {
      current_frame[j] *= hanning_window[j];
    }
    LITERT_RETURN_IF_ERROR(PadOrTruncateForFft(
        config.GetFftLength(), config.GetFftPaddingType(), current_frame));
  }
  return windowed_signals;
}

absl::Status ToLogMelSpectrogram(const AudioPreprocessorConfig& config,
                                 const MelFilterbank& mel_filterbank,
                                 const std::vector<float>& spectrograms,
                                 std::vector<float>& log_mel_spectrograms) {
  std::vector<double> spectrograms_double(spectrograms.begin(),
                                          spectrograms.end());
  int fft_bins = config.GetFftBins();
  const int frames = spectrograms.size() / fft_bins;
  log_mel_spectrograms.reserve(frames * config.GetNumMelBins());
  std::vector<double> tmp_log_mel(config.GetNumMelBins(), 0);
  for (int i = 0; i < frames; ++i) {
    LITERT_RETURN_IF_ERROR(mel_filterbank.ToMelSpectrum(
        absl::MakeSpan(spectrograms_double.data() + i * fft_bins, fft_bins),
        &tmp_log_mel));
    for (int j = 0; j < tmp_log_mel.size(); ++j) {
      float log_mel;
      if (config.GetAddFloorToMelBeforeLog()) {
        log_mel =
            std::log(static_cast<float>(tmp_log_mel[j]) + config.GetMelFloor());
      } else {
        log_mel = std::max(std::log(static_cast<float>(tmp_log_mel[j])),
                           config.GetMelFloor());
      }
      if (config.GetNormalizeMel()) {
        log_mel = (log_mel - AudioPreprocessorConfig::kUsmMelMean[j]) /
                  AudioPreprocessorConfig::kUsmMelStdDev[j];
      }
      log_mel_spectrograms.push_back(log_mel);
    }
  }
  return absl::OkStatus();
}

}  // namespace litert::support
