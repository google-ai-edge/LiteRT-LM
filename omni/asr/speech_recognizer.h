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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_SPEECH_RECOGNIZER_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_SPEECH_RECOGNIZER_H_

#include <optional>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "omni/base/stage.h"

namespace litert::omni::asr {

// Represents a decoded token ID and its optional timestamp.
struct DecodedToken {
  int token_id;
  std::optional<int> timestamp_ms;

  static constexpr int kEndOfChunkTokenId = -1;

  bool IsEndOfChunk() const { return token_id == kEndOfChunkTokenId; }
};

// Abstract interface for recognizing preprocessed audio features into tokens.
class SpeechRecognizer
    : public SingleThreadedStageWithDeque<std::vector<DecodedToken>> {
 public:
  using DecodedToken = ::litert::omni::asr::DecodedToken;

  explicit SpeechRecognizer(
      Stage<std::vector<float>>* absl_nonnull audio_preprocessor)
      : audio_preprocessor_(*audio_preprocessor) {}

  ~SpeechRecognizer() override = default;

 protected:
  bool NeedScheduleInternal() const override {
    return audio_preprocessor_.HasOutput();
  }

  Stage<std::vector<float>>& audio_preprocessor_;
};

}  // namespace litert::omni::asr

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_SPEECH_RECOGNIZER_H_
