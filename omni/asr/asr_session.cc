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

#include "omni/asr/asr_session.h"

#include <algorithm>
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
#include "litert/cc/litert_environment.h"  // from @litert
#include "omni/asr/audio_preprocessor.h"
#include "omni/asr/audio_source.h"
#include "omni/asr/dummy_preprocessor.h"
#include "omni/asr/levenshtein_text_merger.h"
#include "omni/asr/litert_speech_recognizer.h"
#include "omni/asr/log_mel_spectrogram_processor.h"
#include "omni/asr/text_merger.h"
#include "omni/asr/timestamp_text_merger.h"
#include "omni/asr/tokenizer_detokenizer.h"
#include "omni/base/litert_runner.h"
#include "omni/base/stage.h"
#include "omni/multi_staged_session.h"
#include "omni/omni_engine.h"
#include "omni/omni_session.h"
#include "runtime/components/model_resources.h"
#include "runtime/framework/threadpool.h"
#include "runtime/proto/asr_metadata.pb.h"
#include "runtime/proto/asr_session_config.pb.h"
#include "support/tokenizer/tokenizer.h"

namespace litert::omni::asr {
namespace {

constexpr float kDefaultOverlapRatio = 0.4f;

lm::proto::AsrSessionConfig ResolveAsrSessionConfig(
    const MessageLite* absl_nullable session_config) {
  lm::proto::AsrSessionConfig resolved;
  if (session_config != nullptr) {
    resolved.CheckTypeAndMergeFrom(*session_config);
  }
  if (!resolved.has_overlap_ratio()) {
    resolved.set_overlap_ratio(kDefaultOverlapRatio);
  }
  if (!resolved.has_text_merger_type() ||
      resolved.text_merger_type() ==
          lm::proto::AsrSessionConfig::TEXT_MERGER_TYPE_UNSPECIFIED) {
    resolved.set_text_merger_type(
        lm::proto::AsrSessionConfig::TEXT_MERGER_TYPE_TIMESTAMP);
  }
  return resolved;
}

lm::proto::AsrMetadata ResolveAsrMetadata(
    const MessageLite* absl_nullable metadata, int default_vocab_size) {
  lm::proto::AsrMetadata resolved;
  if (metadata != nullptr) {
    resolved.CheckTypeAndMergeFrom(*metadata);
  }
  if (!resolved.has_sample_rate_hz()) {
    resolved.set_sample_rate_hz(16000);
  }
  if (!resolved.has_input_milliseconds()) {
    resolved.set_input_milliseconds(5000);
  }
  if (!resolved.has_decode_stop_token_id()) {
    resolved.set_decode_stop_token_id(-1);
  }
  if (!resolved.has_decode_skip_until_token_id()) {
    resolved.set_decode_skip_until_token_id(-1);
  }
  if (!resolved.has_decode_statefully_after()) {
    resolved.set_decode_statefully_after(4);
  }
  if (resolved.vocab_size() == 0 && default_vocab_size > 0) {
    resolved.set_vocab_size(default_vocab_size);
  }
  return resolved;
}

LogMelSpectrogramProcessor::LogMelSpectrogramConfig ToLogMelConfig(
    const lm::proto::LogMelSpectrogramConfig& l) {
  LogMelSpectrogramProcessor::LogMelSpectrogramConfig cfg;
  if (l.has_n_fft()) cfg.n_fft = l.n_fft();
  if (l.has_n_mels()) cfg.n_mels = l.n_mels();
  if (l.has_hop_length()) cfg.hop_length = l.hop_length();
  if (l.has_n_frames()) cfg.n_frames = l.n_frames();
  if (l.has_fft_length()) cfg.fft_length = l.fft_length();
  if (l.has_input_scale()) cfg.input_scale = l.input_scale();
  if (l.has_preemphasis()) cfg.preemphasis = l.preemphasis();
  if (l.has_mel_low_hz()) cfg.mel_low_hz = l.mel_low_hz();
  if (l.has_mel_high_hz()) cfg.mel_high_hz = l.mel_high_hz();
  if (l.has_mel_floor()) cfg.mel_floor = l.mel_floor();
  if (l.has_normalize_mel()) cfg.normalize_mel = l.normalize_mel();
  if (l.has_add_floor_to_mel_before_log()) {
    cfg.add_floor_to_mel_before_log = l.add_floor_to_mel_before_log();
  }
  if (l.has_transpose()) cfg.transpose = l.transpose();
  if (l.has_norm_type()) {
    if (l.norm_type() ==
        lm::proto::LogMelSpectrogramConfig::NORM_TYPE_WHISPER) {
      cfg.norm_type = LogMelSpectrogramProcessor::NormType::kWhisper;
    } else {
      cfg.norm_type = LogMelSpectrogramProcessor::NormType::kStandard;
    }
  }
  if (l.has_slaney_mel()) cfg.slaney_mel = l.slaney_mel();
  if (l.has_power_spectrum()) cfg.power_spectrum = l.power_spectrum();
  return cfg;
}

// AudioSource implementation that pulls `AudioInput` buffers from an
// `OmniSession::InputSource` and yields PCM audio chunks to an `OmniSession`.
class AudioInputSource : public AudioSource {
 public:
  AudioInputSource(
      std::unique_ptr<OmniSession::InputSource> absl_nonnull input_source,
      int sample_rate_hz, int num_channels, int samples_per_interval,
      int overlap_samples)
      : input_source_(std::move(input_source)),
        sample_rate_hz_(sample_rate_hz),
        num_channels_(num_channels),
        samples_per_interval_(
            static_cast<size_t>(std::max(1, samples_per_interval))),
        overlap_samples_(overlap_samples > 0 &&
                                 static_cast<size_t>(overlap_samples) <
                                     samples_per_interval_
                             ? static_cast<size_t>(overlap_samples)
                             : 0) {}

  ~AudioInputSource() override = default;

  int GetSampleRateHz() const override { return sample_rate_hz_; }
  int GetNumChannels() const override { return num_channels_; }

 protected:
  void ResetInternal() override {
    input_source_->Reset();
    buffer_.clear();
    is_finished_ = false;
  }

  bool NeedScheduleInternal() const override {
    return (!is_finished_ &&
            (input_source_->NeedSchedule() || input_source_->HasOutput())) ||
           buffer_.size() >= samples_per_interval_ ||
           (is_finished_ && !buffer_.empty());
  }

  absl::Status ScheduleInternal() override {
    SetState(State::kRunning);
    absl::Cleanup cleanup = [this] { SetState(State::kIdle); };

    ABSL_RETURN_IF_ERROR(DrainAvailableInputs());
    if (buffer_.size() >= samples_per_interval_) {
      PushNextInterval();
      return absl::OkStatus();
    }
    if (is_finished_) {
      if (buffer_.empty()) {
        return absl::OutOfRangeError("End of audio stream reached.");
      }
      PushPaddedRemainder();
      return absl::OkStatus();
    }
    return absl::NotFoundError("Not enough audio samples for an interval.");
  }

  absl::Status FlushInternal() override {
    ABSL_RETURN_IF_ERROR(input_source_->Flush());
    ABSL_RETURN_IF_ERROR(DrainAvailableInputs());

    while (buffer_.size() >= samples_per_interval_) {
      PushNextInterval();
      ABSL_RETURN_IF_ERROR(DrainAvailableInputs());
    }

    if (!buffer_.empty()) {
      PushPaddedRemainder();
    }
    return absl::OkStatus();
  }

 private:
  // Drains available inputs from `input_source_` into `buffer_` until at least
  // `samples_per_interval_` samples are buffered, coalescing consecutive
  // `AudioInput` chunks and marking `is_finished_` when `EndOfInput` or source
  // `OutOfRangeError` is reached.
  absl::Status DrainAvailableInputs() {
    while (!is_finished_) {
      if (buffer_.size() >= samples_per_interval_) {
        break;
      }
      if (!input_source_->HasOutput()) {
        if (!input_source_->NeedSchedule()) {
          break;
        }
        absl::Status status = input_source_->Schedule();
        if (absl::IsOutOfRange(status)) {
          is_finished_ = true;
          break;
        }
        if (absl::IsNotFound(status)) {
          break;
        }
        ABSL_RETURN_IF_ERROR(status);
        if (!input_source_->HasOutput()) {
          break;
        }
      }
      ABSL_ASSIGN_OR_RETURN(OmniSession::Input input,
                            input_source_->GetOutput());
      if (std::holds_alternative<OmniSession::EndOfInput>(input)) {
        is_finished_ = true;
        break;
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
    return absl::OkStatus();
  }

  void PushNextInterval() {
    const size_t step = samples_per_interval_ - overlap_samples_;
    // Copy remaining content into output first, then swap to reduce data copy.
    std::vector<float> output(buffer_.begin() + step, buffer_.end());
    std::swap(output, buffer_);
    output.resize(samples_per_interval_);
    PushOutput(std::move(output));
  }

  void PushPaddedRemainder() {
    // Zero-pad any remaining samples in `buffer_` (including retained
    // overlap samples from the previous chunk) to `samples_per_interval_`
    // and emit a final chunk so the recognizer can process the tail with
    // trailing silence context.
    buffer_.resize(samples_per_interval_, 0.0f);
    std::vector<float> output;
    std::swap(output, buffer_);
    PushOutput(std::move(output));
  }

  std::unique_ptr<OmniSession::InputSource> absl_nonnull input_source_;
  const int sample_rate_hz_;
  const int num_channels_;
  const size_t samples_per_interval_;
  const size_t overlap_samples_;
  std::vector<float> buffer_;
  bool is_finished_ = false;
};

}  // namespace

std::unique_ptr<AudioSource> AsrSessionFactory::CreateAudioInputSource(
    std::unique_ptr<OmniSession::InputSource> absl_nonnull input_source,
    int sample_rate_hz, int num_channels, int samples_per_interval,
    int overlap_samples) {
  return std::make_unique<AudioInputSource>(
      std::move(input_source), sample_rate_hz, num_channels,
      samples_per_interval, overlap_samples);
}

AsrSessionFactory::AsrSessionFactory(
    const OmniEngine::Settings& settings,
    const MessageLite* absl_nullable session_config,
    const MessageLite* absl_nullable metadata,
    ::litert::Environment env,
    std::shared_ptr<lm::ModelResources> absl_nullable model_resources,
    std::unique_ptr<::litert::support::Tokenizer> absl_nonnull tokenizer)
    : session_config_(ResolveAsrSessionConfig(session_config)),
      metadata_(ResolveAsrMetadata(metadata, tokenizer->GetVocabSize())),
      environment_(std::make_unique<::litert::Environment>(std::move(env))),
      model_resources_(std::move(model_resources)),
      tokenizer_(std::move(tokenizer)),
      thread_pool_(std::make_unique<::litert::lm::ThreadPool>(
          "asr_engine_pool",
          settings.num_threads > 0 ? settings.num_threads : 4)) {}

AsrSessionFactory::~AsrSessionFactory() = default;

absl::StatusOr<std::unique_ptr<OmniSession>> AsrSessionFactory::Create(
    std::unique_ptr<OmniSession::InputSource> absl_nonnull input_source) {
  const int sample_rate_hz = metadata_.sample_rate_hz();
  const int samples_per_interval = static_cast<int>(
      sample_rate_hz * (metadata_.input_milliseconds() / 1000.0));
  const int overlap_samples =
      static_cast<int>(samples_per_interval * session_config_.overlap_ratio());
  auto audio_source = CreateAudioInputSource(
      std::move(input_source), sample_rate_hz, /*num_channels=*/1,
      samples_per_interval, overlap_samples);
  return CreateWithAudioSource(std::move(audio_source));
}

absl::StatusOr<std::unique_ptr<MultiStagedSession>>
AsrSessionFactory::CreateMultiStagedSession(
    std::unique_ptr<AudioSource> absl_nonnull audio_source,
    std::unique_ptr<LiteRtRunner> absl_nonnull encoder,
    std::unique_ptr<LiteRtSpeechRecognizer::Decoder> absl_nonnull decoder) {
  AudioSource* raw_audio_source = audio_source.get();
  std::unique_ptr<AudioPreprocessor> preprocessor;
  if (metadata_.has_log_mel_spectrogram_config()) {
    ABSL_ASSIGN_OR_RETURN(
        preprocessor,
        LogMelSpectrogramProcessor::Create(
            metadata_.sample_rate_hz(),
            ToLogMelConfig(metadata_.log_mel_spectrogram_config()),
            raw_audio_source));
  } else {
    preprocessor = std::make_unique<DummyPreprocessor>(raw_audio_source);
  }

  AudioPreprocessor* raw_preprocessor = preprocessor.get();
  ABSL_ASSIGN_OR_RETURN(
      auto speech_recognizer,
      LiteRtSpeechRecognizer::Create(std::move(encoder), raw_preprocessor,
                                     std::move(decoder)));

  LiteRtSpeechRecognizer* raw_speech_recognizer = speech_recognizer.get();
  auto detokenizer = std::make_unique<TokenizerDetokenizer>(
      raw_speech_recognizer, tokenizer_.get());

  TokenizerDetokenizer* raw_detokenizer = detokenizer.get();
  std::unique_ptr<TextMerger> text_merger;
  switch (session_config_.text_merger_type()) {
    case lm::proto::AsrSessionConfig::TEXT_MERGER_TYPE_LEVENSHTEIN:
      text_merger = std::make_unique<LevenshteinTextMerger>(raw_detokenizer);
      break;
    case lm::proto::AsrSessionConfig::TEXT_MERGER_TYPE_TIMESTAMP:
    case lm::proto::AsrSessionConfig::TEXT_MERGER_TYPE_UNSPECIFIED:
    default:
      text_merger = std::make_unique<TimestampTextMerger>(
          raw_detokenizer, session_config_.overlap_ratio());
      break;
  }

  TextMerger* raw_text_merger = text_merger.get();
  std::vector<std::unique_ptr<internal::StageBase>> stages;
  stages.reserve(5);
  stages.push_back(std::move(audio_source));
  stages.push_back(std::move(preprocessor));
  stages.push_back(std::move(speech_recognizer));
  stages.push_back(std::move(detokenizer));
  stages.push_back(std::move(text_merger));

  return MultiStagedSession::Create(std::move(stages), raw_text_merger,
                                    thread_pool_.get());
}

}  // namespace litert::omni::asr
