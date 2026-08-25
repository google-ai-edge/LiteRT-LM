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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_TOKENIZER_DETOKENIZER_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_TOKENIZER_DETOKENIZER_H_

#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "omni/asr/detokenizer.h"
#include "omni/asr/speech_recognizer.h"
#include "omni/base/stage.h"
#include "support/tokenizer/tokenizer.h"

namespace litert::omni::asr {

// Converts SpeechRecognizer tokens into words using litert::support::Tokenizer.
class TokenizerDetokenizer : public Detokenizer {
 public:
  TokenizerDetokenizer(
      Stage<std::vector<SpeechRecognizer::DecodedToken>>* absl_nonnull
          speech_recognizer,
      ::litert::support::Tokenizer* absl_nonnull tokenizer);

  ~TokenizerDetokenizer() override = default;

 protected:
  absl::Status ScheduleInternal() override;

 private:
  absl::StatusOr<std::vector<Detokenizer::Word>> Detokenize(
      const std::vector<SpeechRecognizer::DecodedToken>& tokens);

  ::litert::support::Tokenizer* const tokenizer_;
};

}  // namespace litert::omni::asr

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_TOKENIZER_DETOKENIZER_H_
