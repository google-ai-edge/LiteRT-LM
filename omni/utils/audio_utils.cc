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

#include "omni/utils/audio_utils.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

#include "support/preprocessor/audio_preprocessor.h"

namespace litert::omni {
namespace {

constexpr float kLn10 = 2.302585092994046f;

}  // namespace

litert::support::AudioPreprocessorConfig CreateWhisperAudioPreprocessorConfig(
    int sample_rate_hz, int n_fft, int hop_length, int n_mels,
    float mel_floor) {
  litert::support::AudioPreprocessorConfig preprocessor_config =
      litert::support::AudioPreprocessorConfig::Create(
          sample_rate_hz, /*num_channels=*/1,
          /*frame_length=*/n_fft,
          /*hop_length=*/hop_length,
          /*fft_length=*/n_fft,
          /*input_scale=*/1.0f,
          /*pre_emphasis_factor=*/0.0f,
          /*num_mel_bins=*/n_mels,
          /*mel_low_hz=*/0.0f,
          /*mel_high_hz=*/sample_rate_hz / 2.0f,
          /*mel_floor=*/mel_floor,
          /*normalize_mel=*/false,
          /*add_floor_to_mel_before_log=*/true,
          /*semicausal_padding=*/false,
          /*non_zero_hanning=*/false,
          /*periodic_hanning=*/true,
          litert::support::AudioPreprocessorConfig::FftPaddingType::kRight);
  preprocessor_config.SetSlaneyMel(true);
  preprocessor_config.SetPowerSpectrum(true);
  return preprocessor_config;
}

void PutSamplesInCenter(std::vector<float>& raw_speech, int n_fft,
                        int hop_length) {
  if (n_fft <= 0 || hop_length <= 0 ||
      raw_speech.size() <= static_cast<size_t>(n_fft)) {
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

void NormalizeWhisperMelSpectrogram(std::vector<std::vector<float>>& mel_spec,
                                    int stat_frames, int num_frames) {
  if (num_frames <= 0) {
    return;
  }
  const int effective_stat_frames = std::max(1, stat_frames);
  float max_val = -std::numeric_limits<float>::infinity();
  for (size_t m = 0; m < mel_spec.size(); ++m) {
    const int frames =
        std::min(static_cast<int>(mel_spec[m].size()), num_frames);
    for (int f = 0; f < frames; ++f) {
      mel_spec[m][f] = mel_spec[m][f] / kLn10;
      if (f < effective_stat_frames && mel_spec[m][f] > max_val) {
        max_val = mel_spec[m][f];
      }
    }
  }
  const float clip_val = max_val - 8.0f;
  for (size_t m = 0; m < mel_spec.size(); ++m) {
    const int frames =
        std::min(static_cast<int>(mel_spec[m].size()), num_frames);
    for (int f = 0; f < frames; ++f) {
      mel_spec[m][f] = (std::max(mel_spec[m][f], clip_val) + 4.0f) / 4.0f;
    }
  }
}

std::vector<float> FlattenMelSpectrogram(
    const std::vector<std::vector<float>>& mel_spec, int n_frames, int n_mels,
    bool transpose) {
  if (n_frames <= 0 || n_mels <= 0) {
    return {};
  }
  std::vector<float> result(static_cast<size_t>(n_frames) * n_mels, 0.0f);
  const int mels_to_copy = std::min(static_cast<int>(mel_spec.size()), n_mels);
  for (int m = 0; m < mels_to_copy; ++m) {
    const int frames_to_copy =
        std::min(static_cast<int>(mel_spec[m].size()), n_frames);
    for (int f = 0; f < frames_to_copy; ++f) {
      if (transpose) {
        result[f * n_mels + m] = mel_spec[m][f];
      } else {
        result[m * n_frames + f] = mel_spec[m][f];
      }
    }
  }
  return result;
}

}  // namespace litert::omni
