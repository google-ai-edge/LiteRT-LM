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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2SPEECH_KOKORO_PORTUGUESE_G2P_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2SPEECH_KOKORO_PORTUGUESE_G2P_H_

#include <string>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "omni/text2speech/kokoro/language_g2p.h"

namespace litert::omni::tts::kokoro {

// Self-contained Brazilian Portuguese (`pt-br`) grapheme-to-phoneme engine for
// Kokoro TTS.
//
// Implements Brazilian Portuguese orthographic-to-phonemic rules
// (syllabification with falling diphthongs and hiatus glides, Acordo
// Ortografico primary and pretonic secondary stress placement, open vs. closed
// mid vowels /ɛ, ɔ/ vs. /e, o/, nasal vowels and coda nasalization, coda /l/
// vocalization, palatalization of /t, d/ before [i]/[y], final vowel reduction,
// cross-word sandhi, abbreviations, and cardinal/ordinal/currency/symbol
// verbalization) emitting Kokoro's Portuguese phoneme inventory directly.
class PortugueseG2p : public LanguageG2p {
 public:
  // Creates a `PortugueseG2p` instance.
  //
  // args:
  //   (none)
  //
  // returns:
  //   An initialized `PortugueseG2p` instance.
  static absl::StatusOr<PortugueseG2p> Create();

 private:
  explicit PortugueseG2p(int unused);

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

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2SPEECH_KOKORO_PORTUGUESE_G2P_H_
