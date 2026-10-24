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

#include "omni/speech2text/asr_session.h"

#include <algorithm>
#include <cstddef>
#include <deque>
#include <memory>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/base/thread_annotations.h"  // from @com_google_absl
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "omni/base/litert_runner.h"
#include "omni/base/stage.h"
#include "omni/multi_staged_session.h"
#include "omni/omni_engine.h"
#include "omni/omni_session.h"
#include "omni/speech2text/audio_preprocessor.h"
#include "omni/speech2text/audio_resampler.h"
#include "omni/speech2text/audio_source.h"
#include "omni/speech2text/dummy_preprocessor.h"
#include "omni/speech2text/levenshtein_text_merger.h"
#include "omni/speech2text/litert_speech_recognizer.h"
#include "omni/speech2text/log_mel_spectrogram_processor.h"
#include "omni/speech2text/text_merger.h"
#include "omni/speech2text/timestamp_text_merger.h"
#include "omni/speech2text/tokenizer_detokenizer.h"
#include "runtime/components/model_resources.h"
#include "runtime/framework/threadpool.h"
#include "runtime/proto/speech2text_metadata.pb.h"
#include "runtime/proto/speech2text_session_config.pb.h"
#include "support/tokenizer/tokenizer.h"

namespace litert::omni::asr {
namespace {

constexpr float kDefaultOverlapRatio = 0.75f;

lm::proto::Speech2TextSessionConfig ResolveSpeech2TextSessionConfig(
    const MessageLite* absl_nullable session_config) {
  lm::proto::Speech2TextSessionConfig resolved;
  if (session_config != nullptr) {
    resolved.CheckTypeAndMergeFrom(*session_config);
  }
  if (!resolved.has_overlap_ratio()) {
    resolved.set_overlap_ratio(kDefaultOverlapRatio);
  }
  if (!resolved.has_text_merger_type() ||
      resolved.text_merger_type() ==
          lm::proto::Speech2TextSessionConfig::TEXT_MERGER_TYPE_UNSPECIFIED) {
    resolved.set_text_merger_type(
        lm::proto::Speech2TextSessionConfig::TEXT_MERGER_TYPE_TIMESTAMP);
  }
  return resolved;
}

lm::proto::Speech2TextMetadata ResolveSpeech2TextMetadata(
    const MessageLite* absl_nullable metadata, int default_vocab_size) {
  lm::proto::Speech2TextMetadata resolved;
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
      int overlap_samples, bool enable_warm_start = false)
      : input_source_(std::move(input_source)),
        sample_rate_hz_(sample_rate_hz),
        input_sample_rate_hz_(sample_rate_hz),
        num_channels_(num_channels),
        samples_per_interval_(
            static_cast<size_t>(std::max(1, samples_per_interval))),
        overlap_samples_(overlap_samples > 0 &&
                                 static_cast<size_t>(overlap_samples) <
                                     samples_per_interval_
                             ? static_cast<size_t>(overlap_samples)
                             : 0),
        enable_warm_start_(enable_warm_start && overlap_samples_ > 0),
        next_warm_start_samples_(samples_per_interval_ - overlap_samples_) {}

  ~AudioInputSource() override = default;

  int GetSampleRateHz() const override { return sample_rate_hz_; }
  int GetNumChannels() const override { return num_channels_; }

  absl::StatusOr<std::vector<float>> GetOutput() override {
    absl::MutexLock lock(mutex_);
    if (outputs_.empty()) {
      return absl::NotFoundError("Deque has no output available.");
    }
    std::vector<float> item = std::move(outputs_.front());
    outputs_.pop_front();
    if (!warm_start_flags_.empty()) {
      last_output_was_warm_start_ = warm_start_flags_.front();
      warm_start_flags_.pop_front();
    } else {
      last_output_was_warm_start_ = false;
    }
    return item;
  }

  bool IsLastOutputWarmStart() const override {
    absl::MutexLock lock(mutex_);
    return last_output_was_warm_start_;
  }

 protected:
  void ResetInternal() override {
    input_source_->Reset();
    input_sample_rate_hz_ = sample_rate_hz_;
    resampler_.Reset();
    buffer_.clear();
    has_unpushed_samples_ = false;
    has_pushed_full_interval_ = false;
    next_warm_start_samples_ = samples_per_interval_ - overlap_samples_;
    is_finished_ = false;
    {
      absl::MutexLock lock(mutex_);
      warm_start_flags_.clear();
      last_output_was_warm_start_ = false;
    }
  }

  bool NeedScheduleInternal() const override {
    return (!is_finished_ &&
            (input_source_->NeedSchedule() || input_source_->HasOutput() ||
             ShouldPushWarmStart())) ||
           buffer_.size() >= samples_per_interval_ ||
           (is_finished_ && has_unpushed_samples_);
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
      if (!has_unpushed_samples_) {
        buffer_.clear();
        return absl::OutOfRangeError("End of audio stream reached.");
      }
      PushRemainder();
      return absl::OkStatus();
    }
    if (ShouldPushWarmStart()) {
      PushWarmStart();
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

    if (has_unpushed_samples_) {
      PushRemainder();
    } else if (is_finished_) {
      buffer_.clear();
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
        if (metadata->sample_rate_hz <= 0) {
          return absl::InvalidArgumentError(
              "AudioInputMetadata sample_rate_hz must be positive.");
        }
        if (metadata->num_channels <= 0 ||
            metadata->num_channels != num_channels_) {
          return absl::InvalidArgumentError(
              "AudioInputMetadata num_channels does not match "
              "AudioInputSource.");
        }
        if (metadata->sample_rate_hz != input_sample_rate_hz_) {
          input_sample_rate_hz_ = metadata->sample_rate_hz;
          resampler_.Reset();
        }
        continue;
      }
      auto* audio_input = std::get_if<OmniSession::AudioInput>(&input);
      if (audio_input == nullptr) {
        return absl::InvalidArgumentError("ASR Session requires AudioInput.");
      }
      std::vector<float> pcm_samples = std::move(audio_input->pcm_samples);
      if (!pcm_samples.empty() && input_sample_rate_hz_ != sample_rate_hz_) {
        ABSL_ASSIGN_OR_RETURN(  // NOLINT - prefer ABSL_ as consistency
            pcm_samples, resampler_.Resample(pcm_samples, input_sample_rate_hz_,
                                             sample_rate_hz_));
      }
      if (!pcm_samples.empty()) {
        // Newly arrived PCM samples in `buffer_` have not been emitted in any
        // interval yet. Tracking this explicitly is needed because when
        // `buffer_.size() <= overlap_samples_`, `buffer_.size()` alone cannot
        // distinguish unpushed samples of a short stream from the
        // already-pushed `overlap_samples_` tail retained by
        // `PushNextInterval()`.
        has_unpushed_samples_ = true;
      }
      if (buffer_.empty()) {
        std::swap(buffer_, pcm_samples);
      } else {
        buffer_.insert(buffer_.end(), pcm_samples.begin(), pcm_samples.end());
      }
    }
    return absl::OkStatus();
  }

  bool ShouldPushWarmStart() const {
    return enable_warm_start_ && !has_pushed_full_interval_ && !is_finished_ &&
           has_unpushed_samples_ &&
           buffer_.size() >= next_warm_start_samples_ &&
           buffer_.size() < samples_per_interval_;
  }

  void PushChunk(std::vector<float> chunk, bool is_warm_start) {
    absl::MutexLock lock(mutex_);
    outputs_.push_back(std::move(chunk));
    warm_start_flags_.push_back(is_warm_start);
  }

  void PushWarmStart() {
    const size_t step = samples_per_interval_ - overlap_samples_;
    next_warm_start_samples_ = ((buffer_.size() / step) + 1) * step;
    has_unpushed_samples_ = false;
    PushChunk(buffer_, /*is_warm_start=*/true);
  }

  void PushNextInterval() {
    has_pushed_full_interval_ = true;
    const size_t step = samples_per_interval_ - overlap_samples_;
    // Copy remaining content into output first, then swap to reduce data copy.
    std::vector<float> output(buffer_.begin() + step, buffer_.end());
    std::swap(output, buffer_);
    output.resize(samples_per_interval_);
    has_unpushed_samples_ = buffer_.size() > overlap_samples_;
    PushChunk(std::move(output), /*is_warm_start=*/false);
  }

  void PushRemainder() {
    // Emit remaining samples unpadded; the downstream AudioPreprocessor pads to
    // the model window after computing normalization statistics over real
    // audio.
    std::vector<float> output;
    std::swap(output, buffer_);
    has_unpushed_samples_ = false;
    has_pushed_full_interval_ = false;
    next_warm_start_samples_ = samples_per_interval_ - overlap_samples_;
    PushChunk(std::move(output), /*is_warm_start=*/false);
  }

  std::unique_ptr<OmniSession::InputSource> absl_nonnull input_source_;
  const int sample_rate_hz_;
  int input_sample_rate_hz_;
  const int num_channels_;
  const size_t samples_per_interval_;
  const size_t overlap_samples_;
  const bool enable_warm_start_;
  size_t next_warm_start_samples_;
  AudioResampler resampler_;
  std::vector<float> buffer_;
  // True when `buffer_` contains samples that have not yet been emitted by
  // `PushNextInterval()`, `PushWarmStart()`, or `PushRemainder()` (as opposed
  // to only the retained `overlap_samples_` tail from the previous full
  // interval).
  bool has_unpushed_samples_ = false;
  bool has_pushed_full_interval_ = false;
  bool is_finished_ = false;
  std::deque<bool> warm_start_flags_ ABSL_GUARDED_BY(mutex_);
  bool last_output_was_warm_start_ ABSL_GUARDED_BY(mutex_) = false;
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
    const MessageLite* absl_nullable metadata, ::litert::Environment env,
    std::shared_ptr<lm::ModelResources> absl_nullable model_resources,
    std::unique_ptr<::litert::support::Tokenizer> absl_nonnull tokenizer)
    : session_config_(ResolveSpeech2TextSessionConfig(session_config)),
      metadata_(
          ResolveSpeech2TextMetadata(metadata, tokenizer->GetVocabSize())),
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
  const bool enable_warm_start =
      ShouldEnableWarmStart() &&
      session_config_.text_merger_type() ==
          lm::proto::Speech2TextSessionConfig::TEXT_MERGER_TYPE_TIMESTAMP;
  auto audio_source = std::make_unique<AudioInputSource>(
      std::move(input_source), sample_rate_hz, /*num_channels=*/1,
      samples_per_interval, overlap_samples, enable_warm_start);
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
    // Raw-PCM models take a fixed `input_milliseconds` window; the audio source
    // leaves the final chunk of a stream unpadded, so pad it here.
    const int samples_per_interval = static_cast<int>(
        metadata_.sample_rate_hz() * (metadata_.input_milliseconds() / 1000.0));
    preprocessor = std::make_unique<DummyPreprocessor>(raw_audio_source,
                                                       samples_per_interval);
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
    case lm::proto::Speech2TextSessionConfig::TEXT_MERGER_TYPE_LEVENSHTEIN:
      text_merger = std::make_unique<LevenshteinTextMerger>(raw_detokenizer);
      break;
    case lm::proto::Speech2TextSessionConfig::TEXT_MERGER_TYPE_TIMESTAMP:
    case lm::proto::Speech2TextSessionConfig::TEXT_MERGER_TYPE_UNSPECIFIED:
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
