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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_TTS_OMNI_SESSION_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_TTS_OMNI_SESSION_H_

#include <memory>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "omni/omni_session.h"
#include "omni/tts/stream_text_source.h"
#include "omni/tts/text_chunk_utils.h"
#include "omni/tts/tts_engine.h"

namespace litert::omni {
class OmniSessionTest;
}  // namespace litert::omni

namespace litert::omni::tts {

// `OmniSessionFactory` implementation backed by `TtsEngine`.
class TtsOmniSessionFactory : public OmniSessionFactory {
 public:
  static absl::StatusOr<std::unique_ptr<OmniSessionFactory>> CreateFactory(
      TtsEngineSettings settings);
  ~TtsOmniSessionFactory() override = default;

  absl::StatusOr<std::unique_ptr<OmniSession>> Create(
      std::unique_ptr<OmniSession::InputSource> absl_nonnull input_source)
      override;

 private:
  friend class ::litert::omni::OmniSessionTest;

  static std::unique_ptr<StreamTextSource> CreateTextInputSource(
      std::unique_ptr<OmniSession::InputSource> absl_nonnull input_source,
      TextChunkConfig config = {});

  explicit TtsOmniSessionFactory(
      std::unique_ptr<TtsEngine> absl_nonnull tts_engine);

  std::unique_ptr<TtsEngine> tts_engine_;
};

}  // namespace litert::omni::tts

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_TTS_OMNI_SESSION_H_
