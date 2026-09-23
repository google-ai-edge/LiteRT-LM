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
#include <deque>
#include <memory>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/thread_annotations.h"  // from @com_google_absl
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "omni/asr/asr_engine.h"
#include "omni/asr/asr_session.h"
#include "omni/asr/audio_source.h"
#include "omni/omni_session.h"

namespace litert::omni::asr {
namespace {

// AudioSource implementation that accepts `AudioInput` buffers and yields
// PCM audio chunks to an `AsrSession`.
class AudioInputSource : public AudioSource {
 public:
  AudioInputSource(int sample_rate_hz, int num_channels,
                   int samples_per_interval, int overlap_samples)
      : sample_rate_hz_(sample_rate_hz),
        num_channels_(num_channels),
        samples_per_interval_(static_cast<size_t>(samples_per_interval)),
        overlap_samples_(overlap_samples > 0 &&
                                 overlap_samples < samples_per_interval
                             ? static_cast<size_t>(overlap_samples)
                             : 0) {}

  ~AudioInputSource() override = default;

  int GetSampleRateHz() const override { return sample_rate_hz_; }
  int GetNumChannels() const override { return num_channels_; }

  absl::Status PushAudio(const OmniSession::AudioInput& input) {
    if (input.sample_rate_hz > 0 && input.sample_rate_hz != sample_rate_hz_) {
      return absl::InvalidArgumentError(
          "AudioInput sample_rate_hz does not match AudioInputSource.");
    }
    if (input.num_channels > 0 && input.num_channels != num_channels_) {
      return absl::InvalidArgumentError(
          "AudioInput num_channels does not match AudioInputSource.");
    }
    if (input.pcm_samples.empty()) {
      return absl::OkStatus();
    }

    absl::MutexLock lock(buffer_mutex_);
    pending_chunks_.push_back(input.pcm_samples);
    return absl::OkStatus();
  }

  void Flush() {
    absl::MutexLock lock(buffer_mutex_);
    pending_chunks_.clear();
  }

 protected:
  void ResetInternal() override { Flush(); }

  bool NeedScheduleInternal() const override {
    absl::MutexLock lock(buffer_mutex_);
    size_t total_size = 0;
    for (const auto& chunk : pending_chunks_) {
      total_size += chunk.size();
      if (total_size >= samples_per_interval_) {
        return true;
      }
    }
    return false;
  }

  absl::Status ScheduleInternal() override {
    SetState(State::kRunning);
    absl::Cleanup cleanup = [this] { SetState(State::kIdle); };

    std::deque<std::vector<float>> queue;
    {
      absl::MutexLock lock(buffer_mutex_);
      queue.swap(pending_chunks_);
    }

    if (queue.empty()) {
      return absl::OutOfRangeError("End of audio stream reached.");
    }

    std::vector<float> output = std::move(queue.front());
    queue.pop_front();
    while (output.size() < samples_per_interval_ && !queue.empty()) {
      const auto& next = queue.front();
      output.insert(output.end(), next.begin(), next.end());
      queue.pop_front();
    }

    if (output.size() < samples_per_interval_) {
      queue.push_front(std::move(output));
      RestoreQueue(std::move(queue));
      return absl::OutOfRangeError("End of audio stream reached.");
    }

    const size_t step = samples_per_interval_ - overlap_samples_;
    if (output.size() > step) {
      queue.push_front(std::vector<float>(output.begin() + step, output.end()));
    }
    output.resize(samples_per_interval_);

    RestoreQueue(std::move(queue));

    PushOutput(std::move(output));
    return absl::OkStatus();
  }

 private:
  void RestoreQueue(std::deque<std::vector<float>> queue) {
    absl::MutexLock lock(buffer_mutex_);
    pending_chunks_.swap(queue);
    for (auto& chunk : queue) {
      pending_chunks_.push_back(std::move(chunk));
    }
  }

  const int sample_rate_hz_;
  const int num_channels_;
  const size_t samples_per_interval_;
  const size_t overlap_samples_;

  mutable absl::Mutex buffer_mutex_;
  std::deque<std::vector<float>> pending_chunks_ ABSL_GUARDED_BY(buffer_mutex_);
};

// `OmniSession` implementation backed by `AsrSession`.
class AsrOmniSession : public OmniSession {
 public:
  explicit AsrOmniSession(std::unique_ptr<AsrSession> asr_session)
      : asr_session_(std::move(asr_session)),
        audio_source_(dynamic_cast<AudioInputSource*>(
            asr_session_->components().audio_source.get())) {}
  ~AsrOmniSession() override = default;

  void Reset() override { asr_session_->Reset(); }

  absl::StatusOr<Output> Process(Input input) override {
    const auto* audio_input = std::get_if<AudioInput>(&input);
    if (audio_input == nullptr) {
      return absl::InvalidArgumentError(
          "ASR OmniSession::Process() requires AudioInput.");
    }
    ABSL_RETURN_IF_ERROR(audio_source_->PushAudio(*audio_input));
    TextOutput combined_output;
    while (audio_source_->NeedSchedule()) {
      ABSL_ASSIGN_OR_RETURN(TextOutput chunk_out,
                            asr_session_->ProcessNextChunk());
      absl::StrAppend(&combined_output.confirmed_text,
                      chunk_out.confirmed_text);
      combined_output.unconfirmed_text = std::move(chunk_out.unconfirmed_text);
    }
    return Output(std::move(combined_output));
  }

  absl::Status ProcessAsync(Input input, OutputCallback callback) override {
    const auto* audio_input = std::get_if<AudioInput>(&input);
    if (audio_input == nullptr) {
      return absl::InvalidArgumentError(
          "ASR OmniSession::ProcessAsync() requires AudioInput.");
    }
    ABSL_RETURN_IF_ERROR(audio_source_->PushAudio(*audio_input));
    // TODO(b/538727793): Avoid flushing TextMerger when intermediate chunks
    // drain in streaming ProcessAsync calls before AsrOmniSession::Flush().
    absl::Status status = asr_session_->ProcessAsync(
        [cb = std::move(callback)](
            absl::StatusOr<TextOutput> result) mutable -> absl::Status {
          if (!result.ok()) {
            return cb(result.status());
          }
          return cb(Output(*std::move(result)));
        });
    if (absl::IsAlreadyExists(status)) {
      return absl::OkStatus();
    }
    return status;
  }

  absl::StatusOr<Output> Flush() override {
    audio_source_->Flush();
    ABSL_ASSIGN_OR_RETURN(TextOutput flushed_out, asr_session_->Flush());
    return Output(std::move(flushed_out));
  }

 private:
  std::unique_ptr<AsrSession> asr_session_;
  AudioInputSource* const audio_source_ = nullptr;
};

}  // namespace

struct AsrOmniSessionTestingPeer {
  static std::unique_ptr<AudioSource> CreateAudioInputSource(
      int sample_rate_hz, int num_channels, int samples_per_interval,
      int overlap_samples);
  static std::unique_ptr<OmniSession> CreateSession(
      std::unique_ptr<AsrSession> asr_session);
};

std::unique_ptr<AudioSource> AsrOmniSessionTestingPeer::CreateAudioInputSource(
    int sample_rate_hz, int num_channels, int samples_per_interval,
    int overlap_samples) {
  return std::make_unique<AudioInputSource>(
      sample_rate_hz, num_channels, samples_per_interval, overlap_samples);
}

std::unique_ptr<OmniSession> AsrOmniSessionTestingPeer::CreateSession(
    std::unique_ptr<AsrSession> asr_session) {
  return std::make_unique<AsrOmniSession>(std::move(asr_session));
}

absl::StatusOr<std::unique_ptr<OmniSessionFactory>>
AsrOmniSessionFactory::CreateFactory(AsrEngineConfig config) {
  ABSL_ASSIGN_OR_RETURN(auto asr_engine, AsrEngine::Create(std::move(config)));
  return std::unique_ptr<OmniSessionFactory>(
      new AsrOmniSessionFactory(std::move(asr_engine)));
}

AsrOmniSessionFactory::AsrOmniSessionFactory(
    std::unique_ptr<AsrEngine> asr_engine)
    : asr_engine_(std::move(asr_engine)) {}

absl::StatusOr<std::unique_ptr<OmniSession>> AsrOmniSessionFactory::Create() {
  const auto& config = asr_engine_->config();
  int samples_per_interval = static_cast<int>(
      config.sample_rate_hz * (config.input_milliseconds / 1000.0));
  int overlap_samples =
      static_cast<int>(samples_per_interval * config.overlap_ratio);
  auto audio_source = std::make_unique<AudioInputSource>(
      config.sample_rate_hz, /*num_channels=*/1, samples_per_interval,
      overlap_samples);
  ABSL_ASSIGN_OR_RETURN(auto asr_session,
                        asr_engine_->CreateSession(std::move(audio_source)));
  return std::make_unique<AsrOmniSession>(std::move(asr_session));
}

}  // namespace litert::omni::asr
