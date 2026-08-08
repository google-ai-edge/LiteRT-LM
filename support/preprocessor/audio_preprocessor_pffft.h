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

#ifndef THIRD_PARTY_ODML_LITERT_LM_SUPPORT_PREPROCESSOR_AUDIO_PREPROCESSOR_PFFFT_H_
#define THIRD_PARTY_ODML_LITERT_LM_SUPPORT_PREPROCESSOR_AUDIO_PREPROCESSOR_PFFFT_H_

#include <memory>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "support/preprocessor/audio_preprocessor.h"
#include "support/preprocessor/mel_filterbank.h"
#include "support/util/io_types.h"

namespace litert::support {

// Audio preprocessor implementation using PFFFT library.
class AudioPreprocessorPffft : public AudioPreprocessor {
 public:
  ~AudioPreprocessorPffft() override;

  // Creates an AudioPreprocessorPffft instance.
  // Args:
  //   - config: The configuration of the audio preprocessor.
  // Returns:
  //   A non-null unique pointer to the AudioPreprocessorPffft instance upon
  //   success.
  static absl::StatusOr<absl_nonnull std::unique_ptr<AudioPreprocessorPffft>>
  Create(const AudioPreprocessorConfig& config);

  // Preprocesses the audio (PCM frames or pre-processed audio) and returns the
  // preprocessed audio mel spectrograms.
  absl::StatusOr<InputAudio> Preprocess(const InputAudio& input_audio) override;

  // Exposes PcmFramesToSpectrogram for testing purposes.
  absl::Status PcmFramesToSpectrogramForTesting(
      absl::Span<const float> pcm_frames, std::vector<float>& spectrograms) {
    return PcmFramesToSpectrogram(pcm_frames, spectrograms);
  }

  // Resets the preprocessor to its initial state.
  void Reset() override {
    input_queue_.clear();
    if (config_.GetSemicausalPadding()) {
      samples_to_next_step_ = config_.GetFrameLength() - config_.GetHopLength();
      input_queue_.resize(config_.GetHopLength(), 0.0f);
    } else {
      samples_to_next_step_ = config_.GetFrameLength();
    }
  }

  // Copy constructor for cloning the audio preprocessor.
  AudioPreprocessorPffft(const AudioPreprocessorPffft& other)
      : config_(other.config_),
        mel_filterbank_(nullptr),
        input_queue_(other.input_queue_),
        samples_to_next_step_(other.samples_to_next_step_) {
    mel_filterbank_ = std::make_unique<MelFilterbank>();
    auto status = mel_filterbank_->Initialize(
        other.config_.GetFftBins(), other.config_.GetSampleRateHz(),
        other.config_.GetNumMelBins(), other.config_.GetMelLowHz(),
        other.config_.GetMelHighHz());
    if (!status.ok()) {
      ABSL_LOG(ERROR) << "Failed to initialize mel filterbank: " << status;
    }
  }

  // Copy assignment operator for cloning the audio preprocessor.
  AudioPreprocessorPffft& operator=(const AudioPreprocessorPffft& other) {
    if (this == &other) {
      return *this;
    }
    config_ = other.config_;
    mel_filterbank_ = std::make_unique<MelFilterbank>();
    auto status = mel_filterbank_->Initialize(
        other.config_.GetFftBins(), other.config_.GetSampleRateHz(),
        other.config_.GetNumMelBins(), other.config_.GetMelLowHz(),
        other.config_.GetMelHighHz());
    if (!status.ok()) {
      ABSL_LOG(ERROR) << "Failed to initialize mel filterbank: " << status;
    }
    input_queue_ = other.input_queue_;
    samples_to_next_step_ = other.samples_to_next_step_;
    return *this;
  }

 private:
  explicit AudioPreprocessorPffft(
      const AudioPreprocessorConfig& config,
      std::unique_ptr<MelFilterbank> mel_filterbank)
      : config_(config),
        mel_filterbank_(std::move(mel_filterbank)),
        input_queue_(std::vector<float>()) {
    if (config.GetSemicausalPadding()) {
      samples_to_next_step_ = config.GetFrameLength() - config.GetHopLength();
      input_queue_.resize(config.GetHopLength(), 0.0f);
    } else {
      samples_to_next_step_ = config.GetFrameLength();
    }
  }

  absl::Status PcmFramesToSpectrogram(absl::Span<const float> pcm_frames,
                                      std::vector<float>& spectrograms);

  AudioPreprocessorConfig config_;
  std::unique_ptr<MelFilterbank> mel_filterbank_;
  std::vector<float> input_queue_;
  int samples_to_next_step_;
};

}  // namespace litert::support

#endif  // THIRD_PARTY_ODML_LITERT_LM_SUPPORT_PREPROCESSOR_AUDIO_PREPROCESSOR_PFFFT_H_
