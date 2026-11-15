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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2SPEECH_KOKORO_LANGUAGE_G2P_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2SPEECH_KOKORO_LANGUAGE_G2P_H_

#include <string>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl

namespace litert::omni::tts::kokoro {

// Common interface of the self-contained, rule-based G2P engines that need no
// bundled lexicon (Spanish, Hindi, Portuguese, Italian, and French), so that
// `KokoroPhonemizer` can hold them and dispatch to them uniformly.
//
// Engines implement the private `DoTextToIpa` / `DoWordToIpa` hooks; callers
// use the public non-virtual `TextToIpa` / `WordToIpa`.
class LanguageG2p {
 public:
  virtual ~LanguageG2p();

  // Converts text (a word, phrase, or sentence) in the engine's language into
  // a space-joined Kokoro phoneme string with punctuation preserved.
  //
  // args
  // - text: UTF-8 input text to phonemize.
  // - custom_lexicon: Optional lowercase word -> Kokoro IPA override map.
  //
  // returns
  // - The Kokoro phoneme string for `text`, or an error status on failure.
  absl::StatusOr<std::string> TextToIpa(
      absl::string_view text,
      const absl::flat_hash_map<std::string, std::string>& custom_lexicon = {})
      const;

  // Converts a single word, phonemized in isolation, into a Kokoro phoneme
  // string.
  //
  // args
  // - word: UTF-8 word token to phonemize.
  // - custom_lexicon: Optional lowercase word -> Kokoro IPA override map.
  //
  // returns
  // - The Kokoro phoneme string for `word`, or an error status on failure.
  absl::StatusOr<std::string> WordToIpa(
      absl::string_view word,
      const absl::flat_hash_map<std::string, std::string>& custom_lexicon = {})
      const;

 private:
  // Implements `TextToIpa`.
  virtual absl::StatusOr<std::string> DoTextToIpa(
      absl::string_view text,
      const absl::flat_hash_map<std::string, std::string>& custom_lexicon)
      const = 0;

  // Implements `WordToIpa`.
  virtual absl::StatusOr<std::string> DoWordToIpa(
      absl::string_view word,
      const absl::flat_hash_map<std::string, std::string>& custom_lexicon)
      const = 0;
};

}  // namespace litert::omni::tts::kokoro

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2SPEECH_KOKORO_LANGUAGE_G2P_H_
