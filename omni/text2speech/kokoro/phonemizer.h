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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2SPEECH_KOKORO_PHONEMIZER_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2SPEECH_KOKORO_PHONEMIZER_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "omni/text2speech/kokoro/chinese_g2p.h"
#include "omni/text2speech/kokoro/cjk_blob.h"
#include "omni/text2speech/kokoro/english_g2p.h"
#include "omni/text2speech/kokoro/japanese_g2p.h"
#include "omni/text2speech/kokoro/language_g2p.h"
#include "omni/text2speech/kokoro/traditional_chinese_converter.h"
#include "omni/text2speech/text_normalizer.h"

namespace litert::omni::tts {

// Returns the global mapping of IPA phoneme graphemes to vocabulary token IDs.
//
// returns
// - Const reference to flat_hash_map of phoneme strings to integer token IDs.
const absl::flat_hash_map<std::string_view, int>& GetKokoroVocabMap();

// Normalization flavors corresponding to Misaki G2P post-processing modes.
enum class MisakiFlavor {
  // Misaki English US fallback normalization (lang_code 'a', voice prefix
  // "af"/"am").
  kEnglishUs,
  // Misaki English GB fallback normalization (lang_code 'b', voice prefix
  // "bf"/"bm").
  kEnglishGb,
  // Misaki generic G2P normalization (es, fr-fr, hi, it, pt-br).
  // Preserves Spanish jota, trills, and vowel length markers.
  kGeneric,
  // Misaki Mandarin Chinese normalization (lang_code 'z', voice prefix
  // "zf"/"zm", language code "cmn").
  kChinese,
  // Misaki Japanese normalization (lang_code 'j', voice prefix "jf"/"jm").
  kJapanese,
};

// Maps a canonical language code ("en-us", "en-gb", "es", "cmn", etc.) to
// its corresponding Misaki normalization flavor.
//
// args
// - language_code: Canonical language identifier (as returned by
//   `NormalizeLanguageCode`, e.g. "en-us", "en-gb", "es", "cmn").
//
// returns
// - The `MisakiFlavor` governing phoneme post-processing for `language_code`.
MisakiFlavor FlavorForLanguage(absl::string_view language_code);

// Normalizes raw IPA phonemes to Kokoro-compatible vocabulary symbols according
// to the target language's Misaki flavor.
//
// args
// - raw_ipa: Raw IPA phoneme string view.
// - flavor: Target normalization flavor (default: kEnglishUs).
//
// returns
// - Normalized IPA string matching Kokoro phoneme vocabulary.
std::string NormalizeMisakiPhonemes(
    absl::string_view raw_ipa, MisakiFlavor flavor = MisakiFlavor::kEnglishUs);

// Resolves the canonical language code implied by a Kokoro voice identifier
// (e.g. "ef_dora" -> "es", "am_adam" -> "en-us", "hf_alpha.bin" -> "hi",
// "zf_xiaobei" -> "cmn").
//
// args
// - voice_name: Kokoro voice identifier or filename (e.g. "af_heart",
//   "zf_xiaobei.bin").
//
// returns
// - Canonical language code (e.g. "en-us", "es", "cmn"), or an empty
//   string if `voice_name` does not match the `<lang><gender>_<name>` prefix
//   convention.
std::string LanguageForVoiceName(absl::string_view voice_name);

// Validates that the requested language matches the voice's language.
// If `language` is empty, auto-detects from `voice_name` (or defaults to
// "en-us"). If `language` is specified and conflicts with `voice_name`, returns
// an InvalidArgumentError.
//
// args
// - voice_name: Kokoro voice identifier or filename (e.g. "zf_xiaobei").
// - language: Optional user-requested language tag or code (e.g. "zh-CN",
//   "cmn", or "" to infer from `voice_name`).
//
// returns
// - Resolved canonical language code (e.g. "cmn", "en-us"), or an
//   `absl::InvalidArgumentError` if `language` conflicts with `voice_name`.
absl::StatusOr<std::string> ResolveAndValidateLanguage(
    absl::string_view voice_name, absl::string_view language);

// Checks whether a Unicode codepoint forms part of a spoken word across
// supported languages (ASCII alphanumeric, Latin Extended, Devanagari, CJK
// Unified Ideographs, Hiragana, and Katakana).
//
// args
// - cp: Unicode codepoint (`char32_t`) to classify.
//
// returns
// - `true` if `cp` belongs to a spoken word token, or `false` if it should be
//   treated as punctuation or whitespace.
bool IsWordCodePoint(char32_t cp);

// Normalizes a language name or Kokoro 1-letter voice code to a canonical
// language code. Examples:
// - "a", "en", "en-us" -> "en-us"
// - "b", "en-gb" -> "en-gb"
// - "e", "es", "spanish" -> "es"
// - "f", "fr", "fr-fr" -> "fr-fr"
// - "h", "hi", "hindi" -> "hi"
// - "i", "it", "italian" -> "it"
// - "p", "pt", "pt-br" -> "pt-br"
// - "j", "ja", "japanese" -> "ja"
// - "z", "zh", "zh-CN", "chinese" -> "cmn"
//
// args
// - language_code: Language identifier or single-letter code.
//
// returns
// - Canonical language code.
std::string NormalizeLanguageCode(absl::string_view language_code);

// Encapsulates the G2P (Grapheme-to-Phoneme) engines, lexicon front ends, and
// Misaki phoneme normalizer for phonemization and token ID encoding across
// languages.
//
// Usage:
// ```cpp
// ABSL_ASSIGN_OR_RETURN(
//     std::unique_ptr<KokoroPhonemizer> phonemizer,
//     KokoroPhonemizer::Create(/*data_dir=*/"", "cmn", /*custom_lexicon=*/{},
//                              text_norm_rules, zh_lexicon_bytes));
// ABSL_ASSIGN_OR_RETURN(std::vector<int> ids,
//                       phonemizer->TextToPhonemeIds("你好，世界！"));
// ```
class KokoroPhonemizer {
 public:
  // Creates and initializes a KokoroPhonemizer instance using the specified
  // data directory, target language, custom lexicon, text normalization rules,
  // and packed CJK lexicon blob.
  //
  // args
  // - data_dir: Path to auxiliary phonemizer data directory or model directory.
  // - language: Language code or voice prefix (e.g. "en-us", "en-gb", "es",
  //   "fr-fr", "hi", "it", "pt-br", "ja", "zh", "cmn", "a", "b", "e", etc.).
  //   Defaults to "en-us".
  // - custom_lexicon: Custom dictionary mapping words to IPA pronunciations.
  // - text_norm_rules: Optional text normalization rule table for `language`,
  //   in the format documented in omni/text2speech/text_normalizer.h. Applied
  //   to text before phonemization. The table is bound to `language`, so it
  //   stops being applied if SetLanguage later switches to another language.
  // - cjk_lexicon: Optional raw bytes of a packed `LCJK` binary container
  //   (e.g. the `"zh-lexicon"` section from the `.litertlm` package). Must
  //   remain mapped and valid for the lifetime of the returned
  //   `KokoroPhonemizer`. Required when `language` is Mandarin ("cmn").
  // - zh_t2s_dictionary: Optional Traditional -> Simplified Chinese conversion
  //   table (e.g. the `"zh-t2s"` section), in the format documented in
  //   kokoro/traditional_chinese_converter.h. When given, Mandarin text is
  //   converted to Simplified Chinese before text normalization and G2P.
  //
  // returns
  // - Unique pointer to `KokoroPhonemizer` on success, or an error status if
  //   required phonemizer data is missing, `text_norm_rules` or
  //   `zh_t2s_dictionary` is malformed, or `cjk_lexicon` is missing/corrupt for
  //   a CJK language.
  static absl::StatusOr<std::unique_ptr<KokoroPhonemizer>> Create(
      absl::string_view data_dir, absl::string_view language = "en-us",
      const absl::flat_hash_map<std::string, std::string>& custom_lexicon = {},
      absl::string_view text_norm_rules = "",
      absl::string_view cjk_lexicon = "",
      absl::string_view zh_t2s_dictionary = "");

  // Overload for creating KokoroPhonemizer with default "en-us" language and
  // a custom lexicon.
  //
  // args
  // - data_dir: Path to auxiliary phonemizer data directory or model directory.
  // - custom_lexicon: Custom dictionary mapping lowercase words to IPA
  //   pronunciations.
  //
  // returns
  // - Unique pointer to `KokoroPhonemizer` on success, or an error status on
  //   failure.
  static absl::StatusOr<std::unique_ptr<KokoroPhonemizer>> Create(
      absl::string_view data_dir,
      const absl::flat_hash_map<std::string, std::string>& custom_lexicon);

  ~KokoroPhonemizer() = default;

  // Updates the active language/voice for phonemization.
  // Normalizes the language code (e.g., "en-gb", "b", "es", "spanish").
  //
  // args
  // - language: Target language code or voice prefix to switch to.
  void SetLanguage(absl::string_view language);

  // Converts a word in the configured language to IPA pronunciation via custom
  // lexicon or the active G2P front end.
  //
  // args
  // - word: Input word string view.
  //
  // returns
  // - IPA pronunciation string, or error status on failure.
  absl::StatusOr<std::string> WordToIpa(absl::string_view word) const;

  // Converts raw text in the configured language to normalized
  // Kokoro-compatible IPA phoneme transcript.
  //
  // args
  // - text: Input raw text string view.
  //
  // returns
  // - Normalized IPA phoneme transcription string, or error status on
  // failure.
  absl::StatusOr<std::string> TextToIpa(absl::string_view text) const;

  // Converts raw text into framed sequence of Kokoro phoneme token IDs (with
  // BOS/EOS).
  //
  // args
  // - text: Input raw text string view.
  //
  // returns
  // - Vector of integer phoneme token IDs framed with BOS (0) and EOS (0), or
  // error status on failure.
  absl::StatusOr<std::vector<int>> TextToPhonemeIds(
      absl::string_view text) const;

  // Returns the active normalized language code.
  //
  // returns
  // - View of the active canonical language code (e.g. "en-us", "es", "cmn").
  absl::string_view Language() const;

 private:
  KokoroPhonemizer() = default;

  // Converts an accumulated word token into its IPA representation and appends
  // it to the combined IPA string.
  //
  // args
  // - current_word: The current word being processed.
  // - combined_ipa: The combined IPA string to append to.
  //
  // returns
  // - Error status on failure.
  absl::Status FlushWordToIpa(std::string& current_word,
                              std::string& combined_ipa) const;

  std::string data_dir_;
  std::string language_ = "en-us";
  absl::flat_hash_map<std::string, std::string> custom_lexicon_;
  absl::flat_hash_map<std::string, std::string> merged_lexicon_;
  // Applied to text before phonemization. Null when no rule table was given.
  std::unique_ptr<TextNormalizer> text_normalizer_;
  // Normalized language the rule table was written for. SetLanguage can move
  // `language_` away from it, and normalizing English with, say, Mandarin
  // rules would corrupt the text, so the two must agree for the normalizer to
  // run.
  std::string text_norm_language_;
  // Section directory of the bundled lexicon blob, plus the language
  // front end built over it (`en-lexicon` -> `english_g2p_`, `ja-lexicon` ->
  // `japanese_g2p_`, `zh-lexicon` -> `chinese_g2p_`).
  std::unique_ptr<kokoro::CjkBlob> cjk_blob_;
  std::unique_ptr<kokoro::EnglishG2p> english_g2p_;
  // Rule-based G2P engines that need no bundled lexicon, keyed by normalized
  // language code ("es", "hi", "pt-br", "it", "fr-fr").
  absl::flat_hash_map<std::string, std::unique_ptr<kokoro::LanguageG2p>>
      language_g2ps_;
  std::unique_ptr<kokoro::JapaneseG2p> japanese_g2p_;
  std::unique_ptr<kokoro::ChineseG2p> chinese_g2p_;
  // Converts Traditional Chinese to Simplified before Mandarin text
  // normalization and G2P. Null when no conversion table was given.
  std::unique_ptr<kokoro::TraditionalChineseConverter>
      traditional_chinese_converter_;
};

}  // namespace litert::omni::tts

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2SPEECH_KOKORO_PHONEMIZER_H_
