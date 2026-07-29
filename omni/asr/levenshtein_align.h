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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_LEVENSHTEIN_ALIGN_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_LEVENSHTEIN_ALIGN_H_

#include <string>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl

namespace litert::omni::asr {

// Represents the edit decision code for string sequence alignment.
enum class AlignCode {
  kInsertion = 0,
  kDeletion = 1,
  kSubstitution = 2,
  kCorrect = 3,
};

// Computes the Levenshtein sequence alignment between ref_tokens and
// hyp_tokens. Returns the optimal alignment decision sequence.
std::vector<AlignCode> AlignTokens(absl::Span<const std::string> ref_tokens,
                                   absl::Span<const std::string> hyp_tokens);

}  // namespace litert::omni::asr

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_LEVENSHTEIN_ALIGN_H_
