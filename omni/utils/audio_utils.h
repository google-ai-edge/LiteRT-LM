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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_UTILS_AUDIO_UTILS_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_UTILS_AUDIO_UTILS_H_

#include <vector>

#include "support/preprocessor/audio_preprocessor.h"

namespace litert::omni {

// Creates an `AudioPreprocessorConfig` configured for Whisper-compatible
// Slaney power-spectrum log-mel extraction with periodic Hann windowing.
//
// Args:
//   sample_rate_hz: Input PCM sample rate in Hz.
//   n_fft: STFT window and FFT size in samples.
//   hop_length: STFT hop length in samples.
//   n_mels: Number of mel frequency bins.
//   mel_floor: Additive floor applied to mel energies before taking the log.
//
// Returns:
//   Configured `litert::support::AudioPreprocessorConfig` instance.
litert::support::AudioPreprocessorConfig CreateWhisperAudioPreprocessorConfig(
    int sample_rate_hz = 16000, int n_fft = 400, int hop_length = 160,
    int n_mels = 80, float mel_floor = 1e-10f);

// Applies `center=True` reflect padding in-place around `raw_speech` so each
// STFT window of length `n_fft` is centered at `frame * hop_length`, matching
// `WhisperFeatureExtractor` (`torch.stft(..., center=True,
// pad_mode="reflect")[:, :-1]`). No-op if `n_fft <= 0`, `hop_length <= 0`, or
// `raw_speech.size() <= static_cast<size_t>(n_fft)`.
//
// Args:
//   raw_speech: Input/output PCM audio samples modified in-place.
//   n_fft: STFT window size in samples.
//   hop_length: STFT hop length in samples.
void PutSamplesInCenter(std::vector<float>& raw_speech, int n_fft,
                        int hop_length);

// Normalizes a `[n_mels][num_frames]` natural-log mel spectrogram in-place
// using Whisper's log10 dynamic-range compression (`max(x, max_val - 8.0) + 4`
// divided by `4.0`), computing `max_val` across frames `[0, stat_frames)`.
//
// Args:
//   mel_spec: Input/output `[n_mels][num_frames]` natural-log mel spectrogram.
//   stat_frames: Number of leading frames used to compute the peak value.
//   num_frames: Total number of frames per mel bin to normalize.
void NormalizeWhisperMelSpectrogram(std::vector<std::vector<float>>& mel_spec,
                                    int stat_frames, int num_frames);

// Flattens a `[n_mels][frames]` 2D mel spectrogram into a 1D vector of size
// `n_frames * n_mels`, zero-padding or truncating the frame axis to `n_frames`.
//
// Args:
//   mel_spec: 2D mel spectrogram indexed as `mel_spec[mel_bin][frame]`.
//   n_frames: Target number of output frames.
//   n_mels: Target number of mel bins.
//   transpose: If true, outputs frame-major `[n_frames, n_mels]`; otherwise
//     outputs mel-major `[n_mels, n_frames]`.
//
// Returns:
//   Flattened spectrogram vector of length `n_frames * n_mels`.
std::vector<float> FlattenMelSpectrogram(
    const std::vector<std::vector<float>>& mel_spec, int n_frames, int n_mels,
    bool transpose);

}  // namespace litert::omni

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_UTILS_AUDIO_UTILS_H_
