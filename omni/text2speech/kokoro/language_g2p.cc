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

#include "omni/text2speech/kokoro/language_g2p.h"

#include <string>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl

namespace litert::omni::tts::kokoro {

LanguageG2p::~LanguageG2p() = default;

absl::StatusOr<std::string> LanguageG2p::TextToIpa(
    absl::string_view text,
    const absl::flat_hash_map<std::string, std::string>& custom_lexicon) const {
  return DoTextToIpa(text, custom_lexicon);
}

absl::StatusOr<std::string> LanguageG2p::WordToIpa(
    absl::string_view word,
    const absl::flat_hash_map<std::string, std::string>& custom_lexicon) const {
  return DoWordToIpa(word, custom_lexicon);
}

}  // namespace litert::omni::tts::kokoro
