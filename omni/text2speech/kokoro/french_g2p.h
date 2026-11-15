// Copyright 2026 Google LLC.
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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2SPEECH_KOKORO_FRENCH_G2P_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2SPEECH_KOKORO_FRENCH_G2P_H_

#include <string>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "omni/text2speech/kokoro/language_g2p.h"

namespace litert::omni::tts::kokoro {

// Self-contained French (`fr-fr`) grapheme-to-phoneme engine for Kokoro TTS.
//
// Implements French orthographic-to-phonemic rules (silent 3rd-person plural
// `-ent` verb endings vs. nasal `/ɑ̃/` nouns/adjectives, context-sensitive
// homograph disambiguation, vowel/consonant digraphs, nasal vowels, mute-e
// schwa deletion, clause-level liaison and `h aspiré` blocking, function-word
// destressing, consonant-only acronym letter-by-letter spelling, and
// cardinal/ordinal/decimal/time/currency/symbol verbalization) emitting
// Kokoro's French phoneme inventory directly.
class FrenchG2p : public LanguageG2p {
 public:
  // Creates a `FrenchG2p` instance.
  //
  // args:
  //   (none)
  //
  // returns:
  //   An initialized `FrenchG2p` instance.
  static absl::StatusOr<FrenchG2p> Create();

 private:
  explicit FrenchG2p(int unused);

  // LanguageG2p:
  absl::StatusOr<std::string> DoTextToIpa(
      absl::string_view text,
      const absl::flat_hash_map<std::string, std::string>& custom_lexicon)
      const override;
  absl::StatusOr<std::string> DoWordToIpa(
      absl::string_view word,
      const absl::flat_hash_map<std::string, std::string>& custom_lexicon)
      const override;
};

}  // namespace litert::omni::tts::kokoro

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2SPEECH_KOKORO_FRENCH_G2P_H_
