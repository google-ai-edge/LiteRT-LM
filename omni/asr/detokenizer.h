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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_DETOKENIZER_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_DETOKENIZER_H_

#include <optional>
#include <string>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "omni/asr/speech_recognizer.h"
#include "omni/base/stage.h"

namespace litert::omni::asr {

// Represents a decoded word and its optional timestamp.
struct Word {
  std::string text;
  std::optional<int> timestamp_ms;

  // An empty text indicates an end-of-chunk word.
  bool IsEndOfChunk() const { return text.empty(); }
};

// Abstract interface for converting decoded tokens into words.
class Detokenizer : public SingleThreadedStageWithDeque<std::vector<Word>> {
 public:
  using Word = ::litert::omni::asr::Word;

  explicit Detokenizer(
      Stage<std::vector<SpeechRecognizer::DecodedToken>>* absl_nonnull
          speech_recognizer)
      : speech_recognizer_(*speech_recognizer) {}

  ~Detokenizer() override = default;

 protected:
  bool NeedScheduleInternal() const override {
    return speech_recognizer_.HasOutput();
  }

  Stage<std::vector<SpeechRecognizer::DecodedToken>>& speech_recognizer_;
};

}  // namespace litert::omni::asr

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_DETOKENIZER_H_
