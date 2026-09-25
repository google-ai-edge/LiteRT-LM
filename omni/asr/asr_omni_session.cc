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

#include "omni/asr/asr_omni_session.h"

#include <cstddef>
#include <memory>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "omni/asr/asr_engine.h"
#include "omni/asr/audio_source.h"
#include "omni/omni_session.h"

namespace litert::omni::asr {
namespace {

// AudioSource implementation that pulls `AudioInput` buffers from an
// `OmniSession::InputSource` and yields PCM audio chunks to an `AsrSession`.
class AudioInputSource : public AudioSource {
 public:
  AudioInputSource(
      std::unique_ptr<OmniSession::InputSource> absl_nonnull input_source,
      int sample_rate_hz, int num_channels, int samples_per_interval,
      int overlap_samples)
      : input_source_(std::move(input_source)),
        sample_rate_hz_(sample_rate_hz),
        num_channels_(num_channels),
        samples_per_interval_(static_cast<size_t>(samples_per_interval)),
        overlap_samples_(overlap_samples > 0 &&
                                 overlap_samples < samples_per_interval
                             ? static_cast<size_t>(overlap_samples)
                             : 0) {}

  ~AudioInputSource() override = default;

  int GetSampleRateHz() const override { return sample_rate_hz_; }
  int GetNumChannels() const override { return num_channels_; }

 protected:
  void ResetInternal() override {
    input_source_->Reset();
    buffer_.clear();
  }

  bool NeedScheduleInternal() const override {
    return input_source_->NeedSchedule() || input_source_->HasOutput() ||
           buffer_.size() >= samples_per_interval_;
  }

  absl::Status ScheduleInternal() override {
    SetState(State::kRunning);
    absl::Cleanup cleanup = [this] { SetState(State::kIdle); };

    while (buffer_.size() < samples_per_interval_) {
      if (input_source_->NeedSchedule()) {
        ABSL_RETURN_IF_ERROR(input_source_->Schedule());
      }
      if (!input_source_->HasOutput()) {
        return absl::OutOfRangeError("End of audio stream reached.");
      }
      ABSL_ASSIGN_OR_RETURN(OmniSession::Input input,
                            input_source_->GetOutput());
      if (std::holds_alternative<OmniSession::EndOfInput>(input)) {
        if (buffer_.empty()) {
          return absl::OutOfRangeError("End of audio stream reached.");
        }
        buffer_.resize(samples_per_interval_, 0.0f);
        std::vector<float> output;
        std::swap(output, buffer_);
        PushOutput(std::move(output));
        return absl::OkStatus();
      }
      if (const auto* metadata =
              std::get_if<OmniSession::AudioInputMetadata>(&input)) {
        if (metadata->sample_rate_hz > 0 &&
            metadata->sample_rate_hz != sample_rate_hz_) {
          return absl::InvalidArgumentError(
              "AudioInputMetadata sample_rate_hz does not match "
              "AudioInputSource.");
        }
        if (metadata->num_channels > 0 &&
            metadata->num_channels != num_channels_) {
          return absl::InvalidArgumentError(
              "AudioInputMetadata num_channels does not match "
              "AudioInputSource.");
        }
        continue;
      }
      auto* audio_input = std::get_if<OmniSession::AudioInput>(&input);
      if (audio_input == nullptr) {
        return absl::InvalidArgumentError("ASR Session requires AudioInput.");
      }
      if (buffer_.empty()) {
        std::swap(buffer_, audio_input->pcm_samples);
      } else {
        buffer_.insert(buffer_.end(), audio_input->pcm_samples.begin(),
                       audio_input->pcm_samples.end());
      }
    }

    const size_t step = samples_per_interval_ - overlap_samples_;
    // Copy remaining content into output first, then swap to reduce data copy.
    std::vector<float> output(buffer_.begin() + step, buffer_.end());
    std::swap(output, buffer_);
    output.resize(samples_per_interval_);
    PushOutput(std::move(output));
    return absl::OkStatus();
  }

 private:
  std::unique_ptr<OmniSession::InputSource> absl_nonnull input_source_;
  const int sample_rate_hz_;
  const int num_channels_;
  const size_t samples_per_interval_;
  const size_t overlap_samples_;
  std::vector<float> buffer_;
};

}  // namespace

std::unique_ptr<AudioSource> AsrOmniSessionFactory::CreateAudioInputSource(
    std::unique_ptr<OmniSession::InputSource> absl_nonnull input_source,
    int sample_rate_hz, int num_channels, int samples_per_interval,
    int overlap_samples) {
  return std::make_unique<AudioInputSource>(
      std::move(input_source), sample_rate_hz, num_channels,
      samples_per_interval, overlap_samples);
}

absl::StatusOr<std::unique_ptr<OmniSessionFactory>>
AsrOmniSessionFactory::CreateFactory(AsrEngineConfig config) {
  ABSL_ASSIGN_OR_RETURN(auto asr_engine, AsrEngine::Create(std::move(config)));
  return std::unique_ptr<OmniSessionFactory>(
      new AsrOmniSessionFactory(std::move(asr_engine)));
}

AsrOmniSessionFactory::AsrOmniSessionFactory(
    std::unique_ptr<AsrEngine> absl_nonnull asr_engine)
    : asr_engine_(std::move(asr_engine)) {}

absl::StatusOr<std::unique_ptr<OmniSession>> AsrOmniSessionFactory::Create(
    std::unique_ptr<OmniSession::InputSource> absl_nonnull input_source) {
  const auto& config = asr_engine_->config();
  int samples_per_interval = static_cast<int>(
      config.sample_rate_hz * (config.input_milliseconds / 1000.0));
  int overlap_samples =
      static_cast<int>(samples_per_interval * config.overlap_ratio);
  auto audio_source = CreateAudioInputSource(
      std::move(input_source), config.sample_rate_hz, /*num_channels=*/1,
      samples_per_interval, overlap_samples);
  return asr_engine_->CreateSession(std::move(audio_source));
}

}  // namespace litert::omni::asr
