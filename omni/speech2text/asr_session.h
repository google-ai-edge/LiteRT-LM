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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_SPEECH2TEXT_ASR_SESSION_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_SPEECH2TEXT_ASR_SESSION_H_

#include <memory>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "omni/base/litert_runner.h"
#include "omni/multi_staged_session.h"
#include "omni/omni_engine.h"
#include "omni/omni_session.h"
#include "omni/speech2text/audio_source.h"
#include "omni/speech2text/litert_speech_recognizer.h"
#include "runtime/components/model_resources.h"
#include "runtime/framework/threadpool.h"
#include "runtime/proto/speech2text_metadata.pb.h"
#include "runtime/proto/speech2text_session_config.pb.h"
#include "support/tokenizer/tokenizer.h"

namespace litert::omni {
class OmniSessionTest;
}  // namespace litert::omni

namespace litert::omni::asr {

// Base `OmniSessionFactory` implementation for ASR models.
class AsrSessionFactory : public OmniSessionFactory {
 public:
  AsrSessionFactory(
      const OmniEngine::Settings& settings,
      const MessageLite* absl_nullable session_config,
      const MessageLite* absl_nullable metadata,
      ::litert::Environment env,
      std::shared_ptr<lm::ModelResources> absl_nullable model_resources,
      std::unique_ptr<::litert::support::Tokenizer> absl_nonnull tokenizer);

  ~AsrSessionFactory() override;

  absl::StatusOr<std::unique_ptr<OmniSession>> Create(
      std::unique_ptr<OmniSession::InputSource> absl_nonnull input_source)
      override;

  const lm::proto::Speech2TextMetadata& metadata() const { return metadata_; }
  const lm::proto::Speech2TextSessionConfig& session_config() const {
    return session_config_;
  }

 protected:
  // Whether `AudioInputSource` should emit progressive partial-window prefixes
  // before the first full `input_milliseconds` window is accumulated.
  virtual bool ShouldEnableWarmStart() const { return false; }

  // Subclasses override this to call `CreateMultiStagedSession` below with
  // `audio_source` built from `input_source` by this class, and `encoder` and
  // `decoder` instantiated by subclasses.
  virtual absl::StatusOr<std::unique_ptr<MultiStagedSession>>
  CreateWithAudioSource(
      std::unique_ptr<AudioSource> absl_nonnull audio_source) = 0;

  // Builds the ASR stage pipeline (`AudioSource` -> `AudioPreprocessor` ->
  // `LiteRtSpeechRecognizer` -> `TokenizerDetokenizer` -> `TextMerger`) and
  // returns a `MultiStagedSession`.
  absl::StatusOr<std::unique_ptr<MultiStagedSession>> CreateMultiStagedSession(
      std::unique_ptr<AudioSource> absl_nonnull audio_source,
      std::unique_ptr<LiteRtRunner> absl_nonnull encoder,
      std::unique_ptr<LiteRtSpeechRecognizer::Decoder> absl_nonnull decoder);

 private:
  friend class ::litert::omni::OmniSessionTest;
  friend class AsrSessionTest;

  static std::unique_ptr<AudioSource> CreateAudioInputSource(
      std::unique_ptr<OmniSession::InputSource> absl_nonnull input_source,
      int sample_rate_hz, int num_channels, int samples_per_interval,
      int overlap_samples);

  lm::proto::Speech2TextSessionConfig session_config_;
  lm::proto::Speech2TextMetadata metadata_;
  std::unique_ptr<::litert::Environment> absl_nonnull environment_;
  std::shared_ptr<lm::ModelResources> absl_nullable model_resources_;
  std::unique_ptr<::litert::support::Tokenizer> absl_nonnull tokenizer_;
  std::unique_ptr<::litert::lm::ThreadPool> absl_nonnull thread_pool_;
};

}  // namespace litert::omni::asr

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_SPEECH2TEXT_ASR_SESSION_H_
