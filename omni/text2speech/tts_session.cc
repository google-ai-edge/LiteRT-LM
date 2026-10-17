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

#include "omni/text2speech/tts_session.h"

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/base/thread_annotations.h"  // from @com_google_absl
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "omni/base/model_resources.h"
#include "omni/omni_engine.h"
#include "omni/omni_session.h"
#include "omni/text2speech/stream_text_source.h"
#include "omni/text2speech/text_chunk_utils.h"
#include "runtime/framework/threadpool.h"
#include "runtime/proto/text2speech_metadata.pb.h"
#include "runtime/proto/text2speech_session_config.pb.h"

namespace litert::omni::tts {
namespace {

lm::proto::Text2SpeechSessionConfig ResolveText2SpeechSessionConfig(
    const MessageLite* absl_nullable session_config) {
  lm::proto::Text2SpeechSessionConfig resolved;
  if (session_config != nullptr) {
    resolved.CheckTypeAndMergeFrom(*session_config);
  }
  if (!resolved.has_language()) {
    resolved.set_language("en-US");
  }
  return resolved;
}

// StreamTextSource subclass that pulls `TextInput` payloads from an
// `OmniSession::InputSource` and yields text chunks to an `OmniSession`.
class TextInputSource : public StreamTextSource {
 public:
  explicit TextInputSource(
      std::unique_ptr<OmniSession::InputSource> absl_nonnull input_source,
      TextChunkConfig config = {})
      : StreamTextSource(std::move(config)),
        input_source_(std::move(input_source)) {}

  ~TextInputSource() override = default;

 protected:
  void ResetInternal() override {
    input_source_->Reset();
    StreamTextSource::ResetInternal();
  }

  bool NeedScheduleInternal() const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_) override {
    return (!IsFinishedLocked() &&
            (input_source_->NeedSchedule() || input_source_->HasOutput())) ||
           StreamTextSource::NeedScheduleInternal();
  }

  absl::Status ScheduleInternal() ABSL_NO_THREAD_SAFETY_ANALYSIS override {
    SetState(State::kRunning);
    absl::Cleanup cleanup = [this] { SetState(State::kIdle); };

    ABSL_RETURN_IF_ERROR(DrainAvailableInputs());
    return StreamTextSource::ScheduleInternal();
  }

  absl::Status FlushInternal() ABSL_NO_THREAD_SAFETY_ANALYSIS override {
    ABSL_RETURN_IF_ERROR(input_source_->Flush());
    ABSL_RETURN_IF_ERROR(DrainAvailableInputs());
    return StreamTextSource::FlushInternal();
  }

 private:
  // Drains all currently available inputs from `input_source_` into
  // `StreamTextSource`, coalescing consecutive `TextInput`s and calling
  // `Finish()` when `EndOfInput` or source `OutOfRangeError` is reached.
  absl::Status DrainAvailableInputs() ABSL_NO_THREAD_SAFETY_ANALYSIS {
    while (!IsFinished()) {
      if (!input_source_->HasOutput()) {
        if (!input_source_->NeedSchedule()) {
          break;
        }
        absl::Status status = input_source_->Schedule();
        if (absl::IsOutOfRange(status)) {
          Finish();
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
        Finish();
        break;
      }
      const auto* text_input = std::get_if<OmniSession::TextInput>(&input);
      if (text_input == nullptr) {
        return absl::InvalidArgumentError("TTS Session requires TextInput.");
      }
      AppendText(text_input->text);
    }
    return absl::OkStatus();
  }

  std::unique_ptr<OmniSession::InputSource> absl_nonnull input_source_;
};

}  // namespace

TextChunkConfig ToTextChunkConfig(
    const lm::proto::Text2SpeechTextChunkConfig& proto) {
  TextChunkConfig config;
  if (!proto.delimiters().empty()) {
    config.delimiters.assign(proto.delimiters().begin(),
                             proto.delimiters().end());
  }
  if (proto.has_include_delimiter()) {
    config.include_delimiter = proto.include_delimiter();
  }
  if (proto.has_max_buffer_size() && proto.max_buffer_size() > 0) {
    config.max_buffer_size = static_cast<size_t>(proto.max_buffer_size());
  }
  return config;
}

std::unique_ptr<StreamTextSource> TtsSessionFactory::CreateTextInputSource(
    std::unique_ptr<OmniSession::InputSource> absl_nonnull input_source,
    TextChunkConfig config) {
  return std::make_unique<TextInputSource>(std::move(input_source),
                                           std::move(config));
}

TtsSessionFactory::TtsSessionFactory(
    const OmniEngine::Settings& settings,
    const MessageLite* absl_nullable session_config,
    const lm::proto::Text2SpeechMetadata& metadata,
    std::vector<std::string> available_voices,
    std::shared_ptr<ModelResources> absl_nullable model_resources)
    : engine_settings_(settings),
      session_config_(ResolveText2SpeechSessionConfig(session_config)),
      metadata_(metadata),
      available_voices_(std::move(available_voices)),
      model_resources_(std::move(model_resources)),
      thread_pool_(std::make_unique<lm::ThreadPool>(
          "tts_engine_pool",
          settings.num_threads > 0 ? settings.num_threads : 4)) {}

TtsSessionFactory::~TtsSessionFactory() = default;

absl::StatusOr<std::unique_ptr<OmniSession>> TtsSessionFactory::Create(
    std::unique_ptr<OmniSession::InputSource> absl_nonnull input_source) {
  auto text_source = CreateTextInputSource(
      std::move(input_source), ResolveTextChunkConfig(session_config_));
  return CreateWithTextSource(session_config_, std::move(text_source));
}

std::vector<std::string> TtsSessionFactory::GetAvailableLanguages() const {
  return {metadata_.supported_languages().begin(),
          metadata_.supported_languages().end()};
}

std::vector<std::string> TtsSessionFactory::GetAvailableVoices() const {
  return GetAvailableVoices("");
}

std::vector<std::string> TtsSessionFactory::GetAvailableVoices(
    absl::string_view language) const {
  if (language.empty()) {
    return available_voices_;
  }
  return {};
}

bool TtsSessionFactory::HasVoice(absl::string_view voice) const {
  if (voice.empty()) return false;
  absl::string_view voice_name = voice;
  size_t last_slash = voice_name.find_last_of("/\\");
  if (last_slash != absl::string_view::npos) {
    voice_name = voice_name.substr(last_slash + 1);
  }
  if (absl::EndsWith(voice_name, ".bin")) {
    voice_name = voice_name.substr(0, voice_name.size() - 4);
  }
  for (const auto& v : available_voices_) {
    if (v == voice || v == voice_name) {
      return true;
    }
  }
  return false;
}

std::string TtsSessionFactory::GetDefaultVoice(
    absl::string_view /*language*/) const {
  if (!available_voices_.empty()) {
    return available_voices_.front();
  }
  return "";
}

TextChunkConfig TtsSessionFactory::ResolveTextChunkConfig(
    const lm::proto::Text2SpeechSessionConfig& session_config) const {
  lm::proto::Text2SpeechSessionConfig effective =
      MergeSessionConfig(session_config);
  return ToTextChunkConfig(effective.text_chunk_config());
}

lm::proto::Text2SpeechSessionConfig TtsSessionFactory::MergeSessionConfig(
    const lm::proto::Text2SpeechSessionConfig& session_config) const {
  lm::proto::Text2SpeechSessionConfig effective = session_config_;
  effective.MergeFrom(session_config);
  if (session_config.has_text_chunk_config() &&
      !session_config.text_chunk_config().delimiters().empty()) {
    *effective.mutable_text_chunk_config()->mutable_delimiters() =
        session_config.text_chunk_config().delimiters();
  }
  return effective;
}

}  // namespace litert::omni::tts
