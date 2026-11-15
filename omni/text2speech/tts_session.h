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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2SPEECH_TTS_SESSION_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2SPEECH_TTS_SESSION_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "omni/base/model_resources.h"
#include "omni/multi_staged_session.h"
#include "omni/omni_engine.h"
#include "omni/omni_session.h"
#include "omni/text2speech/stream_text_source.h"
#include "omni/text2speech/text_chunk_utils.h"
#include "runtime/framework/threadpool.h"
#include "runtime/proto/text2speech_metadata.pb.h"
#include "runtime/proto/text2speech_session_config.pb.h"

namespace litert::omni {
class OmniSessionTest;
}  // namespace litert::omni

namespace litert::omni::tts {

// Converts a `Text2SpeechTextChunkConfig` proto into a `TextChunkConfig`
// struct.
TextChunkConfig ToTextChunkConfig(
    const lm::proto::Text2SpeechTextChunkConfig& proto);

// Base `OmniSessionFactory` implementation for Text-to-Speech (TTS) models.
class TtsSessionFactory : public OmniSessionFactory {
 public:
  TtsSessionFactory(
      const OmniEngine::Settings& settings,
      const MessageLite* absl_nullable session_config,
      const lm::proto::Text2SpeechMetadata& metadata,
      std::vector<std::string> available_voices,
      std::shared_ptr<ModelResources> absl_nullable model_resources);

  ~TtsSessionFactory() override;

  absl::StatusOr<std::unique_ptr<OmniSession>> Create(
      std::unique_ptr<OmniSession::InputSource> absl_nonnull input_source)
      override;

  const lm::proto::Text2SpeechMetadata& metadata() const { return metadata_; }
  const lm::proto::Text2SpeechSessionConfig& session_config() const {
    return session_config_;
  }
  const OmniEngine::Settings& engine_settings() const {
    return engine_settings_;
  }
  std::shared_ptr<ModelResources> model_resources() const {
    return model_resources_;
  }
  const std::vector<std::string>& available_voices() const {
    return available_voices_;
  }

  // Returns the list of available languages in BCP-47 format supported by the
  // model and its available voices.
  virtual std::vector<std::string> GetAvailableLanguages() const;

  // Returns the list of available voice profile names.
  virtual std::vector<std::string> GetAvailableVoices() const;

  // Returns the list of available voice profile names. If `language` is
  // non-empty, filters to return only voices compatible with that language.
  virtual std::vector<std::string> GetAvailableVoices(
      absl::string_view language) const;

  // Checks whether the specified voice is available.
  bool HasVoice(absl::string_view voice) const;

  // Returns the default voice profile name for the specified BCP-47 language.
  virtual std::string GetDefaultVoice(absl::string_view language) const;

  // Resolves the `TextChunkConfig` for `session_config`, applying any
  // model-specific revisions.
  virtual TextChunkConfig ResolveTextChunkConfig(
      const lm::proto::Text2SpeechSessionConfig& session_config) const;

 protected:
  virtual absl::StatusOr<std::unique_ptr<MultiStagedSession>>
  CreateWithTextSource(
      const lm::proto::Text2SpeechSessionConfig& session_config,
      std::unique_ptr<StreamTextSource> absl_nullable text_source) = 0;

  lm::proto::Text2SpeechSessionConfig MergeSessionConfig(
      const lm::proto::Text2SpeechSessionConfig& session_config) const;

  lm::ThreadPool* thread_pool() const { return thread_pool_.get(); }

 private:
  friend class ::litert::omni::OmniSessionTest;
  friend class TtsEngine;
  friend class TtsSessionTest;

  static std::unique_ptr<StreamTextSource> CreateTextInputSource(
      std::unique_ptr<OmniSession::InputSource> absl_nonnull input_source,
      TextChunkConfig config = {});

  OmniEngine::Settings engine_settings_;
  lm::proto::Text2SpeechSessionConfig session_config_;
  lm::proto::Text2SpeechMetadata metadata_;
  std::vector<std::string> available_voices_;
  std::shared_ptr<ModelResources> absl_nullable model_resources_;
  std::unique_ptr<lm::ThreadPool> absl_nonnull thread_pool_;
};

}  // namespace litert::omni::tts

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2SPEECH_TTS_SESSION_H_
