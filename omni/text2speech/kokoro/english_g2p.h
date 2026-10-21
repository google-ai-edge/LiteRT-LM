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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2SPEECH_KOKORO_ENGLISH_G2P_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2SPEECH_KOKORO_ENGLISH_G2P_H_

#include <string>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "omni/text2speech/kokoro/cjk_blob.h"

namespace litert::omni::tts::kokoro {

// Converts American (`en-us`) and British (`en-gb`) English text into Kokoro's
// IPA phoneme notation without calling `spaCy` or `num2words`.
//
// The pipeline mirrors `misaki.en.G2P`:
//   1. Markdown link preprocessing (`[word](/ipa/)`, `[word](stress)`,
//      `[123](#flags#)`).
//   2. Rule-based tokenization and context-driven part-of-speech tagging.
//   3. Right-to-left lexicon lookup over the `en-lexicon` `LTTSCJK1` blob
//      (`us-*` and `gb-*` sections), including `grow_dictionary` case-flip
//      resolution, POS-dependent heteronym dictionaries, context-sensitive
//      function words (`the` -> `ðə`/`ði`, `to` -> `tə`/`tʊ`/`tu`),
//      morphological inflection stemming (`-s`, `-ed`, `-ing`), and cardinal,
//      ordinal, year, decimal, and currency number expansion.
//   4. Subtoken compound stress resolution (`resolve_tokens`).
//   5. Self-contained C++ fallback for out-of-vocabulary tokens (clock times,
//      temperatures, units, abbreviations, vowelless acronyms, productive
//      prefix/suffix stripping, and English letter-to-sound rules).
//
// All methods on `EnglishG2p` are thread-safe after construction and hold
// non-owning views into the `CjkBlob` passed to `Create`.
class EnglishG2p {
 public:
  // Creates an `EnglishG2p` with empty lexicon tables, using built-in
  // function-word rules, abbreviations, and English letter-to-sound rules.
  //
  // returns:
  //   Initialized `EnglishG2p` on success, or an error status on failure.
  static absl::StatusOr<EnglishG2p> Create();

  // Creates an `EnglishG2p` backed by an `en-lexicon` `CjkBlob` containing
  // `us-idx`, `us-key`, `us-vidx`, `us-val`, `gb-idx`, `gb-key`, `gb-vidx`,
  // and `gb-val` sections.
  //
  // args:
  //   blob: Parsed `LTTSCJK1` container holding the English lexicon tables.
  //     Must outlive the returned `EnglishG2p`.
  //
  // returns:
  //   Initialized `EnglishG2p` on success, or `absl::InvalidArgumentError` /
  //   `absl::NotFoundError` if any required section is missing or malformed.
  static absl::StatusOr<EnglishG2p> Create(const CjkBlob& blob);

  // Converts an English text string into a Kokoro IPA string.
  //
  // args:
  //   text: UTF-8 English input text (words, numbers, currencies, and
  //     punctuation).
  //   british: If true, uses the British (`en-gb`) lexicon and phonological
  //     rules; otherwise uses American (`en-us`).
  //   custom_lexicon: Optional lowercase word -> IPA override map checked
  //     before the bundled lexicon.
  //
  // returns:
  //   Phonemized string in Kokoro's English vocabulary, or an error status on
  //   failure.
  absl::StatusOr<std::string> TextToIpa(
      absl::string_view text, bool british = false,
      const absl::flat_hash_map<std::string, std::string>& custom_lexicon = {})
      const;

  // Converts a single English word token into a Kokoro IPA string.
  //
  // args:
  //   word: UTF-8 English word.
  //   british: If true, uses the British (`en-gb`) lexicon and phonological
  //     rules; otherwise uses American (`en-us`).
  //
  // returns:
  //   Kokoro IPA string for `word`, or an error status on failure.
  absl::StatusOr<std::string> WordToIpa(absl::string_view word,
                                        bool british = false) const;

 private:
  EnglishG2p(SortedStringIndex us_keys, StringValueTable us_values,
             SortedStringIndex gb_keys, StringValueTable gb_values);

  SortedStringIndex us_keys_;
  StringValueTable us_values_;
  SortedStringIndex gb_keys_;
  StringValueTable gb_values_;
};

}  // namespace litert::omni::tts::kokoro

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2SPEECH_KOKORO_ENGLISH_G2P_H_
