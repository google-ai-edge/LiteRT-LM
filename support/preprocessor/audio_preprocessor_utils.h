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

#ifndef THIRD_PARTY_ODML_LITERT_LM_SUPPORT_PREPROCESSOR_AUDIO_PREPROCESSOR_UTILS_H_
#define THIRD_PARTY_ODML_LITERT_LM_SUPPORT_PREPROCESSOR_AUDIO_PREPROCESSOR_UTILS_H_

#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "support/preprocessor/audio_preprocessor.h"
#include "support/preprocessor/mel_filterbank.h"

namespace litert::support {

// Pads or truncates the frame vector in-place to the given fft_length.
absl::Status PadOrTruncateForFft(
    int fft_length, AudioPreprocessorConfig::FftPaddingType padding_type,
    std::vector<float>& frame);

// Generates a Hanning window of length `window_length`.
std::vector<float> GetHanningWindow(int window_length,
                                    bool use_periodic_hanning = true,
                                    bool non_zero_hanning = true);

// Fills `input_queue` with the next window of samples from `pcm_frames`.
// Returns true if a full window is available in `input_queue`.
bool GetNextWindowOfSamples(const AudioPreprocessorConfig& config,
                            const std::vector<float>& pcm_frames,
                            int& input_start, std::vector<float>& input_queue,
                            int& samples_to_next_step);

// Frames `pcm_frames` into windowed segments according to `config`.
absl::StatusOr<std::vector<std::vector<float>>> GetFramedSegments(
    const AudioPreprocessorConfig& config, const std::vector<float>& pcm_frames,
    std::vector<float>& input_queue, int& samples_to_next_step);

// Prepares windowed signals for FFT processing:
// 1. Applies input scaling.
// 2. Extracts framed segments.
// 3. Applies pre-emphasis filtering.
// 4. Applies Hanning windowing.
// 5. Performs FFT padding or truncating.
absl::StatusOr<std::vector<std::vector<float>>> GetWindowedSignalsForFft(
    const AudioPreprocessorConfig& config, absl::Span<const float> pcm_frames,
    std::vector<float>& input_queue, int& samples_to_next_step);

// Converts power spectrograms to log Mel spectrograms using `mel_filterbank`
// and `config`.
absl::Status ToLogMelSpectrogram(const AudioPreprocessorConfig& config,
                                 const MelFilterbank& mel_filterbank,
                                 const std::vector<float>& spectrograms,
                                 std::vector<float>& log_mel_spectrograms);

}  // namespace litert::support

#endif  // THIRD_PARTY_ODML_LITERT_LM_SUPPORT_PREPROCESSOR_AUDIO_PREPROCESSOR_UTILS_H_
