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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_UTILS_AUDIO_RESAMPLER_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_UTILS_AUDIO_RESAMPLER_H_

#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "miniaudio.h"  // from @miniaudio

namespace litert::omni {

// Resamples mono 32-bit float PCM audio between sample rates using miniaudio.
class AudioResampler {
 public:
  AudioResampler() = default;
  ~AudioResampler();

  AudioResampler(const AudioResampler&) = delete;
  AudioResampler& operator=(const AudioResampler&) = delete;

  // Resets internal filter state and cached history without uninitializing the
  // resampler, preparing it for a new independent audio stream.
  void Reset();

  // Resamples `input_pcm` from `input_sample_rate_hz` to
  // `target_sample_rate_hz`. If `input_sample_rate_hz ==
  // target_sample_rate_hz`, returns `input_pcm` unmodified.
  //
  // Args:
  //   input_pcm: Mono 32-bit float PCM samples to resample.
  //   input_sample_rate_hz: Input PCM sample rate in Hz (must be positive).
  //   target_sample_rate_hz: Target output sample rate in Hz (must be
  //     positive).
  //
  // Returns:
  //   Resampled mono 32-bit float PCM samples on success,
  //   `absl::InvalidArgumentError` if either sample rate is non-positive, or
  //   `absl::InternalError` if miniaudio resampler initialization or frame
  //   processing fails.
  absl::StatusOr<std::vector<float>> Resample(absl::Span<const float> input_pcm,
                                              int input_sample_rate_hz,
                                              int target_sample_rate_hz);

 private:
  absl::Status EnsureInitialized(int in_rate, int out_rate);

  ma_resampler resampler_{};
  int input_sample_rate_hz_ = 0;
  int target_sample_rate_hz_ = 0;
  bool is_initialized_ = false;
};

}  // namespace litert::omni

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_UTILS_AUDIO_RESAMPLER_H_
