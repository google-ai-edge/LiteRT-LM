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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_LOG_MEL_SPECTROGRAM_PROCESSOR_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_LOG_MEL_SPECTROGRAM_PROCESSOR_H_

#include <algorithm>
#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "omni/asr/audio_preprocessor.h"
#include "omni/base/stage.h"
#include "support/preprocessor/audio_preprocessor_miniaudio.h"

namespace litert::omni::asr {

// Converts raw PCM audio speech chunks into Log-Mel Spectrogram model input
// features.
class LogMelSpectrogramProcessor : public AudioPreprocessor {
 public:
  enum class NormType {
    kStandard = 0,
    kWhisper = 1,
  };

  // Represents the configuration for Log-Mel Spectrogram preprocessing.
  struct LogMelSpectrogramConfig {
    int n_fft = 512;
    int n_mels = 80;
    int hop_length = 160;
    int n_frames = 0;
    bool transpose = false;
    float preemphasis = 0.0f;
    int fft_length = 0;
    float input_scale = 1.0f;
    float mel_low_hz = 0.0f;
    float mel_high_hz = 0.0f;
    float mel_floor = 1e-5f;
    bool normalize_mel = false;
    bool add_floor_to_mel_before_log = true;
    NormType norm_type = NormType::kStandard;
    // librosa-compatible mel (Slaney scale + normalization, fractional
    // triangles) summed over the power spectrum; what Whisper and NeMo use.
    bool slaney_mel = false;
    bool power_spectrum = false;
  };

  static absl::StatusOr<std::unique_ptr<LogMelSpectrogramProcessor>> Create(
      int sample_rate_hz, const LogMelSpectrogramConfig& config,
      Stage<std::vector<float>>* absl_nonnull audio_source);

  ~LogMelSpectrogramProcessor() override = default;

  const LogMelSpectrogramConfig& config() const { return config_; }

 protected:
  void ResetInternal() override;
  absl::Status ScheduleInternal() override;

 private:
  LogMelSpectrogramProcessor(
      int sample_rate_hz, const LogMelSpectrogramConfig& config,
      Stage<std::vector<float>>* absl_nonnull audio_source,
      std::unique_ptr<litert::support::AudioPreprocessorMiniAudio> absl_nonnull
      preprocessor,
      std::unique_ptr<litert::support::AudioPreprocessorMiniAudio> absl_nullable
      centered_preprocessor);

  // Whether a chunk shorter than the model window that follows a full window
  // (the final chunk of a long stream) is shifted back to end at the end of
  // the audio instead of being padded. Done for the standard per-feature
  // normalization (the encoders it feeds have no length mask and their
  // output degrades on padded windows); the Whisper-style path keeps padding,
  // as Whisper was trained on padded 30 s windows.
  bool ShiftsShortChunkBack() const;

  // Processes raw speech (float vector) into model input features and decode
  // range.
  absl::StatusOr<PreprocessedAudio> Process(std::vector<float> raw_speech);

  std::optional<std::vector<float>> ComputeCenteredFeatures();

  const int sample_rate_hz_;
  const LogMelSpectrogramConfig config_;
  const size_t window_samples_;
  std::unique_ptr<litert::support::AudioPreprocessorMiniAudio> absl_nonnull
      preprocessor_;
  std::unique_ptr<litert::support::AudioPreprocessorMiniAudio> absl_nullable
  centered_preprocessor_;
  // Last `n_frames * hop_length` samples of the most recent full-size chunk.
  std::vector<float> previous_window_;
  float peak_speech_rms_ = 0.0f;
};

}  // namespace litert::omni::asr

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_LOG_MEL_SPECTROGRAM_PROCESSOR_H_
