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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2SPEECH_KOKORO_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2SPEECH_KOKORO_H_

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
#include "omni/text2speech/kokoro/kokoro_model_config.h"
#include "omni/text2speech/stream_text_source.h"
#include "omni/text2speech/text_chunk_utils.h"
#include "omni/text2speech/tts_session.h"
#include "runtime/proto/text2speech_metadata.pb.h"
#include "runtime/proto/text2speech_session_config.pb.h"

namespace litert::omni::tts {

// `TtsSessionFactory` implementation for Kokoro-82M TTS models.
class KokoroSessionFactory : public TtsSessionFactory {
 public:
  KokoroSessionFactory(
      const OmniEngine::Settings& settings,
      const MessageLite* absl_nullable session_config,
      const lm::proto::Text2SpeechMetadata& metadata,
      std::vector<std::string> available_voices,
      std::shared_ptr<ModelResources> absl_nullable model_resources,
      KokoroModelConfig config = {});

  std::vector<std::string> GetAvailableLanguages() const override;

  using TtsSessionFactory::GetAvailableVoices;

  std::vector<std::string> GetAvailableVoices(
      absl::string_view language) const override;

  std::string GetDefaultVoice(absl::string_view language) const override;

  TextChunkConfig ResolveTextChunkConfig(
      const lm::proto::Text2SpeechSessionConfig& session_config) const override;

 protected:
  absl::StatusOr<std::unique_ptr<MultiStagedSession>> CreateWithTextSource(
      const lm::proto::Text2SpeechSessionConfig& session_config,
      std::unique_ptr<StreamTextSource> absl_nullable text_source) override;

 private:
  KokoroModelConfig config_;
};

}  // namespace litert::omni::tts

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2SPEECH_KOKORO_H_
