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

#include "omni/tts/tts_omni_session.h"

#include <memory>
#include <utility>
#include <variant>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "omni/omni_session.h"
#include "omni/tts/tts_engine.h"
#include "omni/tts/tts_session.h"

namespace litert::omni::tts {
namespace {

// `OmniSession` implementation backed by `TtsSession`.
class TtsOmniSession : public OmniSession {
 public:
  explicit TtsOmniSession(std::unique_ptr<TtsSession> tts_session)
      : tts_session_(std::move(tts_session)) {}
  ~TtsOmniSession() override = default;

  void Reset() override { tts_session_->Reset(); }

  absl::StatusOr<Output> Process(const Input& input) override {
    const auto* text_input = std::get_if<TextInput>(&input);
    if (text_input == nullptr) {
      return absl::InvalidArgumentError(
          "TTS OmniSession::Process() requires TextInput.");
    }
    ABSL_ASSIGN_OR_RETURN(AudioOutput audio_out,
                          tts_session_->Synthesize(text_input->text));
    return Output(std::move(audio_out));
  }

  absl::Status ProcessAsync(const Input& input,
                            AsyncCallback callback) override {
    const auto* text_input = std::get_if<TextInput>(&input);
    if (text_input == nullptr) {
      return absl::InvalidArgumentError(
          "TTS OmniSession::ProcessAsync() requires TextInput.");
    }
    return tts_session_->SynthesizeAsync(
        text_input->text,
        [cb = std::move(callback)](
            absl::StatusOr<AudioOutput> result) mutable -> absl::Status {
          if (!result.ok()) {
            return cb(result.status());
          }
          return cb(Output(*std::move(result)));
        });
  }

  absl::StatusOr<Output> Flush() override {
    ABSL_ASSIGN_OR_RETURN(AudioOutput audio_out, tts_session_->Flush());
    return Output(std::move(audio_out));
  }

 private:
  std::unique_ptr<TtsSession> tts_session_;
};

}  // namespace

struct TtsOmniSessionTestingPeer {
  static std::unique_ptr<OmniSession> CreateSession(
      std::unique_ptr<TtsSession> tts_session);
};

std::unique_ptr<OmniSession> TtsOmniSessionTestingPeer::CreateSession(
    std::unique_ptr<TtsSession> tts_session) {
  return std::make_unique<TtsOmniSession>(std::move(tts_session));
}

absl::StatusOr<std::unique_ptr<OmniSessionFactory>>
TtsOmniSessionFactory::CreateFactory(TtsEngineSettings settings) {
  ABSL_ASSIGN_OR_RETURN(auto tts_engine,
                        TtsEngine::Create(std::move(settings)));
  return std::unique_ptr<OmniSessionFactory>(
      new TtsOmniSessionFactory(std::move(tts_engine)));
}

TtsOmniSessionFactory::TtsOmniSessionFactory(
    std::unique_ptr<TtsEngine> tts_engine)
    : tts_engine_(std::move(tts_engine)) {}

absl::StatusOr<std::unique_ptr<OmniSession>> TtsOmniSessionFactory::Create() {
  ABSL_ASSIGN_OR_RETURN(auto tts_session,
                        tts_engine_->CreateSession(TtsSessionConfig{}));
  return std::make_unique<TtsOmniSession>(std::move(tts_session));
}

}  // namespace litert::omni::tts
