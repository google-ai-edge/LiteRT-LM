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

#include <memory>
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
    NormType norm_type = NormType::kStandard;
  };

  static absl::StatusOr<std::unique_ptr<LogMelSpectrogramProcessor>> Create(
      int sample_rate_hz, const LogMelSpectrogramConfig& config,
      Stage<std::vector<float>>* absl_nonnull audio_source);

  ~LogMelSpectrogramProcessor() override = default;

 protected:
  void ResetInternal() override;
  absl::Status ScheduleInternal() override;

 private:
  LogMelSpectrogramProcessor(
      int sample_rate_hz, const LogMelSpectrogramConfig& config,
      Stage<std::vector<float>>* absl_nonnull audio_source,
      std::unique_ptr<litert::support::AudioPreprocessorMiniAudio>
          preprocessor);

  // Processes raw speech (float vector) into model input features.
  absl::StatusOr<std::vector<float>> Process(std::vector<float> raw_speech);

  const int sample_rate_hz_;
  const LogMelSpectrogramConfig config_;
  std::unique_ptr<litert::support::AudioPreprocessorMiniAudio> preprocessor_;
};

}  // namespace litert::omni::asr

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_LOG_MEL_SPECTROGRAM_PROCESSOR_H_
