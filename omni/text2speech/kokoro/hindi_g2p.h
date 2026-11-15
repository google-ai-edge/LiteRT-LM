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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2SPEECH_KOKORO_HINDI_G2P_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2SPEECH_KOKORO_HINDI_G2P_H_

#include <string>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "omni/text2speech/kokoro/language_g2p.h"

namespace litert::omni::tts::kokoro {

// Self-contained rule-based Hindi (Devanagari, `hi`) grapheme-to-phoneme (G2P)
// converter for Kokoro-82M TTS.
//
// Produces Kokoro vocabulary IPA strings (`MisakiFlavor::kGeneric`)
// without requiring any external lexicon blob.
// Covers:
// - Devanagari consonant, independent vowel, matra, halant (virama), nukta
//   (both combining U+093C and precomposed U+0958..U+095F), anusvara,
//   chandrabindu, and visarga decoding
// - Special conjunct realizations (`क्ष` -> `kʃ`, `ज्ञ` -> `ɡj`, reph `र्` ->
//   trill `r` before coronal/labial/fricative consonants)
// - Weight-based Hindi stress assignment on the underlying syllable structure
//   (heavy syllables closed by conjuncts/nasals or containing long vowels)
// - Word-final and right-to-left medial schwa deletion (`VC_CV` / `VC_CCV`)
// - Homorganic nasal assimilation and vowel nasalization (`◌̃`), word-final
//   `-अय` fronting (`ɛj`), and consonant gemination (`ː`)
// - Function-word destressing, Devanagari/ASCII number verbalization, and
//   common symbols (`%`, `/`, `+`, `₹`, `।`, `॥`)
class HindiG2p : public LanguageG2p {
 public:
  // Creates a `HindiG2p` instance.
  //
  // returns
  // - Initialized `HindiG2p` instance.
  static absl::StatusOr<HindiG2p> Create();

 private:
  explicit HindiG2p(int unused);

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

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2SPEECH_KOKORO_HINDI_G2P_H_
