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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_UTILS_WHISPER_LOG_MEL_EXTRACTOR_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_UTILS_WHISPER_LOG_MEL_EXTRACTOR_H_

#include <memory>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "support/preprocessor/audio_preprocessor_miniaudio.h"

namespace litert::omni {

// Synchronous library helper that extracts Whisper-style normalized log-mel
// spectrograms from mono float PCM audio using `AudioPreprocessorMiniAudio`.
class WhisperLogMelExtractor {
 public:
  // Configuration parameters for `WhisperLogMelExtractor`.
  struct Config {
    // Input PCM audio sample rate in Hz.
    int sample_rate_hz = 16000;
    // STFT window and FFT size in samples.
    int n_fft = 400;
    // STFT hop length in samples.
    int hop_length = 160;
    // Number of mel frequency bins.
    int n_mels = 80;
    // Number of output mel frames per extracted window.
    int n_frames = 3000;
    // Additive floor applied to mel energies before taking the log.
    float mel_floor = 1e-10f;
    // If true, outputs frame-major `[n_frames, n_mels]`; otherwise outputs
    // mel-major `[n_mels, n_frames]`.
    bool transpose = false;
  };

  // Creates and initializes a `WhisperLogMelExtractor`.
  //
  // Args:
  //   config: Extraction configuration parameters.
  //
  // Returns:
  //   A unique pointer to `WhisperLogMelExtractor` on success, or
  //   `absl::InvalidArgumentError` if any configuration parameter is invalid.
  static absl::StatusOr<std::unique_ptr<WhisperLogMelExtractor>> Create(
      const Config& config);
  static absl::StatusOr<std::unique_ptr<WhisperLogMelExtractor>> Create();

  ~WhisperLogMelExtractor();

  WhisperLogMelExtractor(const WhisperLogMelExtractor&) = delete;
  WhisperLogMelExtractor& operator=(const WhisperLogMelExtractor&) = delete;

  // Extracts a flattened Whisper-normalized log-mel spectrogram of shape
  // `[n_mels, n_frames]` (or `[n_frames, n_mels]` when `config.transpose` is
  // true) from `pcm_samples`.
  //
  // Resets the underlying preprocessor before extraction and zero-pads or
  // truncates `pcm_samples` to `config.n_frames * config.hop_length` samples
  // (frames beyond the active input are normalized to the Whisper silence
  // floor).
  //
  // Args:
  //   pcm_samples: Mono 32-bit float PCM samples at `config.sample_rate_hz`.
  //
  // Returns:
  //   Flattened spectrogram vector of size `config.n_frames * config.n_mels` on
  //   success, or an error status if preprocessing fails.
  absl::StatusOr<std::vector<float>> Extract(
      absl::Span<const float> pcm_samples);

  // Resets the underlying `AudioPreprocessorMiniAudio` state (note: `Extract`
  // also resets the preprocessor automatically before each window).
  void Reset();

  // Returns the configuration used by this extractor.
  const Config& GetConfig() const;

 private:
  WhisperLogMelExtractor(
      const Config& config,
      std::unique_ptr<litert::support::AudioPreprocessorMiniAudio> absl_nonnull
      preprocessor);

  Config config_;
  std::unique_ptr<litert::support::AudioPreprocessorMiniAudio> preprocessor_;
};

}  // namespace litert::omni

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_UTILS_WHISPER_LOG_MEL_EXTRACTOR_H_
