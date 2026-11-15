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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2SPEECH_KOKORO_SPANISH_G2P_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2SPEECH_KOKORO_SPANISH_G2P_H_

#include <string>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "omni/text2speech/kokoro/language_g2p.h"

namespace litert::omni::tts::kokoro {

// Self-contained Spanish (`es`) grapheme-to-phoneme engine for Kokoro TTS.
//
// Implements Spanish orthographic-to-phonemic rules (syllabification with
// glide/hiatus resolution, RAE primary and pretonic secondary stress placement,
// `-mente` double stress, unstressed function-word reduction, within-word and
// cross-word spirantization of /b, d, g/ and nasal place assimilation,
// abbreviations, Roman numerals, and cardinal/currency/percentage/symbol
// verbalization) emitting Kokoro's Spanish phoneme inventory directly.
class SpanishG2p : public LanguageG2p {
 public:
  // Creates a `SpanishG2p` instance.
  //
  // args:
  //   (none)
  //
  // returns:
  //   An initialized `SpanishG2p` instance.
  static absl::StatusOr<SpanishG2p> Create();

 private:
  explicit SpanishG2p(int unused);

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

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2SPEECH_KOKORO_SPANISH_G2P_H_
