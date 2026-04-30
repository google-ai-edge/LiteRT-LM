// Copyright 2025 The ODML Authors.
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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_PREPROCESSOR_AUDIO_PREPROCESSOR_MINIAUDIO_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_PREPROCESSOR_AUDIO_PREPROCESSOR_MINIAUDIO_H_

#include <memory>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "runtime/components/preprocessor/audio_preprocessor.h"
#include "runtime/components/preprocessor/mel_filterbank.h"
#include "runtime/engine/io_types.h"

namespace litert::lm {

// Audio preprocessor implementation using MiniAudio library and kissfft
// library.
class AudioPreprocessorMiniAudio : public AudioPreprocessor {
 public:
  // Creates an AudioPreprocessorMiniAudio instance.
  // Args:
  //   - config: The configuration of the audio preprocessor.
  // Returns:
  //   A unique pointer to the AudioPreprocessorMiniAudio instance.
  static absl::StatusOr<std::unique_ptr<AudioPreprocessorMiniAudio>> Create(
      const AudioPreprocessorConfig& config);

  // Decodes the raw audio bytes to PCM frames using MiniAudio library.
  // Args:
  //   - audio_bytes: The raw audio bytes read from the audio file to decode.
  //     The supported audio format from MiniAudio library is WAV, FLAC and MP3.
  //   - num_channels: The number of channels in the audio to be decoded. Only
  //     mono audio (num_channels = 1) is supported for now.
  //   - sample_rate_hz: The sample rate of the audio to be resampled to.
  //   - pcm_frames: The decoded PCM frames.
  static absl::Status DecodeAudio(absl::string_view audio_bytes,
                                  int num_channels, int sample_rate_hz,
                                  std::vector<float>& pcm_frames);

  // Preprocesses the undecoded raw audio bytes and returns the preprocessed
  // audio mel spectrograms.
  // Args:
  //   - input_audio: The input audio to preprocess. The supported audio
  //     format from MiniAudio library is WAV, FLAC and MP3.
  //     If the input audio is already preprocessed, it will be returned
  //     directly.
  // Returns:
  //   An InputAudio object containing the preprocessed audio mel spectrograms
  //   with shape (1, num_frames, num_mel_bins).
  absl::StatusOr<InputAudio> Preprocess(const InputAudio& input_audio) override;

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
  AudioPreprocessorMiniAudio(const AudioPreprocessorMiniAudio& other)
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
  AudioPreprocessorMiniAudio& operator=(
      const AudioPreprocessorMiniAudio& other) {
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
  explicit AudioPreprocessorMiniAudio(
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

  absl::Status ToLogMelSpectrogram(const std::vector<float>& spectrograms,
                                   std::vector<float>& log_mel_spectrograms);

  bool GetNextWindowOfSamples(const std::vector<float>& pcm_frames,
                              int& input_start);

  AudioPreprocessorConfig config_;
  std::unique_ptr<MelFilterbank> mel_filterbank_;
  std::vector<float> input_queue_;
  int samples_to_next_step_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_PREPROCESSOR_AUDIO_PREPROCESSOR_MINIAUDIO_H_
