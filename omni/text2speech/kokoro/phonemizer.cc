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

#include "omni/text2speech/kokoro/phonemizer.h"

#include <cstddef>
#include <filesystem>  // NOLINT: Required for path manipulation.
#include <memory>
#include <string>
#include <system_error>  // NOLINT: Required by std::filesystem.
#include <utility>
#include <vector>

#include "absl/base/attributes.h"  // from @com_google_absl
#include "absl/base/const_init.h"  // from @com_google_absl
#include "absl/base/no_destructor.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/ascii.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/str_replace.h"  // from @com_google_absl
#include "absl/strings/str_split.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "omni/text2speech/kokoro/chinese_g2p.h"
#include "omni/text2speech/kokoro/cjk_blob.h"
#include "omni/text2speech/kokoro/common.h"
#include "omni/text2speech/kokoro/english_g2p.h"
#include "omni/text2speech/kokoro/french_g2p.h"
#include "omni/text2speech/kokoro/hindi_g2p.h"
#include "omni/text2speech/kokoro/italian_g2p.h"
#include "omni/text2speech/kokoro/japanese_g2p.h"
#include "omni/text2speech/kokoro/language_g2p.h"
#include "omni/text2speech/kokoro/portuguese_g2p.h"
#include "omni/text2speech/kokoro/spanish_g2p.h"
#include "omni/text2speech/kokoro/traditional_chinese_converter.h"
#include "omni/text2speech/kokoro/unicode_utils.h"
#include "omni/text2speech/text_normalizer.h"

namespace litert::omni::tts {

using ::litert::omni::tts::kokoro::DecodeUtf8CodePoint;
using ::litert::omni::tts::kokoro::Utf8CodePoint;

namespace {

constexpr absl::string_view kMissingJapaneseLexiconError =
    "Japanese TTS requires a bundled 'ja-lexicon' section in the .litertlm "
    "model container.";

// Checks whether a character is an opening delimiter (e.g. '(', '[', '"').
// Used during text phonemization to avoid inserting unnecessary leading spaces
// immediately after an opening bracket or quotation mark.
bool IsOpenPunctuation(char ch) {
  return ch == '(' || ch == '[' || ch == '{' || ch == '"' || ch == '\'';
}

// Creates the rule-based G2P engine `G2p` and registers it for `language`.
template <typename G2p>
absl::Status AddLanguageG2p(
    absl::string_view language,
    absl::flat_hash_map<std::string, std::unique_ptr<kokoro::LanguageG2p>>&
        g2ps) {
  ABSL_ASSIGN_OR_RETURN(G2p g2p, G2p::Create());
  g2ps[language] = std::make_unique<G2p>(std::move(g2p));
  return absl::OkStatus();
}

}  // namespace

const absl::flat_hash_map<std::string_view, int>& GetKokoroVocabMap() {
  // Static token mapping table representing the 178-token Kokoro-82M phoneme
  // vocabulary.
  // - Index 0: BOS / EOS token (`kBosTokenId` / `kEosTokenId`).
  // - Indices 1..15: Clause and sentence punctuation marks (;, :, ,, ., !, ?).
  // - Index 16: Inter-word whitespace delimiter (' ').
  // - Indices 24..41: Uppercase single-character representations of English
  //   diphthongs produced by Misaki normalization (e.g., A=eɪ, I=aɪ, O=oʊ,
  //   Q=əʊ, W=aʊ, Y=ɔɪ).
  // - Remaining indices: Standard IPA vowels, consonants, stress modifiers
  //   (ˈ, ˌ), and pitch direction markers.
  static const absl::NoDestructor<absl::flat_hash_map<std::string_view, int>>
      vocab_map({{";", 1},   {":", 2},   {",", 3},   {".", 4},   {"!", 5},
                 {"?", 6},   {"—", 9},   {"…", 10},  {"\"", 11}, {"(", 12},
                 {")", 13},  {"“", 14},  {"”", 15},  {" ", 16},  {"̃", 17},
                 {"ʣ", 18},  {"ʥ", 19},  {"ʦ", 20},  {"ʨ", 21},  {"ᵝ", 22},
                 {"ꭧ", 23},  {"A", 24},  {"I", 25},  {"O", 31},  {"Q", 33},
                 {"S", 35},  {"T", 36},  {"W", 39},  {"Y", 41},  {"ᵊ", 42},
                 {"a", 43},  {"b", 44},  {"c", 45},  {"d", 46},  {"e", 47},
                 {"f", 48},  {"h", 50},  {"i", 51},  {"j", 52},  {"k", 53},
                 {"l", 54},  {"m", 55},  {"n", 56},  {"o", 57},  {"p", 58},
                 {"q", 59},  {"r", 60},  {"s", 61},  {"t", 62},  {"u", 63},
                 {"v", 64},  {"w", 65},  {"x", 66},  {"y", 67},  {"z", 68},
                 {"ɑ", 69},  {"ɐ", 70},  {"ɒ", 71},  {"æ", 72},  {"β", 75},
                 {"ɔ", 76},  {"ɕ", 77},  {"ç", 78},  {"ɖ", 80},  {"ð", 81},
                 {"ʤ", 82},  {"ə", 83},  {"ɚ", 85},  {"ɛ", 86},  {"ɜ", 87},
                 {"ɟ", 90},  {"ɡ", 92},  {"ɥ", 99},  {"ɨ", 101}, {"ɪ", 102},
                 {"ʝ", 103}, {"ɯ", 110}, {"ɰ", 111}, {"ŋ", 112}, {"ɳ", 113},
                 {"ɲ", 114}, {"ɴ", 115}, {"ø", 116}, {"ɸ", 118}, {"θ", 119},
                 {"œ", 120}, {"ɹ", 123}, {"ɾ", 125}, {"ɻ", 126}, {"ʁ", 128},
                 {"ɽ", 129}, {"ʂ", 130}, {"ʃ", 131}, {"ʈ", 132}, {"ʧ", 133},
                 {"ʊ", 135}, {"ʋ", 136}, {"ʌ", 138}, {"ɣ", 139}, {"ɤ", 140},
                 {"χ", 142}, {"ʎ", 143}, {"ʒ", 147}, {"ʔ", 148}, {"ˈ", 156},
                 {"ˌ", 157}, {"ː", 158}, {"ʰ", 162}, {"ʲ", 164}, {"↓", 169},
                 {"→", 171}, {"↗", 172}, {"↘", 173}, {"ᵻ", 177}});
  return *vocab_map;
}

bool IsHanziCodePoint(char32_t cp) {
  return cp == 0x3007 || (cp >= 0x4E00 && cp <= 0x9FFF) ||
         (cp >= 0x3400 && cp <= 0x4DBF) || (cp >= 0x20000 && cp <= 0x2A6DF);
}

MisakiFlavor FlavorForLanguage(absl::string_view language_code) {
  std::string normalized = NormalizeLanguageCode(language_code);
  if (normalized == "en-us") {
    return MisakiFlavor::kEnglishUs;
  }
  if (normalized == "en-gb") {
    return MisakiFlavor::kEnglishGb;
  }
  if (normalized == "cmn" || normalized == "zh") {
    return MisakiFlavor::kChinese;
  }
  if (normalized == "ja") {
    return MisakiFlavor::kJapanese;
  }
  return MisakiFlavor::kGeneric;
}

std::string NormalizeMisakiPhonemes(absl::string_view raw_ipa,
                                    MisakiFlavor flavor) {
  if (flavor == MisakiFlavor::kGeneric) {
    // Generic multilingual G2P normalization rules (used for es, fr-fr, hi, it,
    // pt-br). They only merge phoneme pairs that are tied into one phoneme
    // (`t͡s`, `a͡ʊ`); an untied `ts` / `aʊ` spans two phonemes (vingt-six,
    // causata) and must stay as is. Preserves Spanish jota ('x'), trill ('r'),
    // and vowel length marker ('ː').
    std::string tied = absl::StrReplaceAll(raw_ipa, {{"^", "\u0361"}});
    tied = absl::StrReplaceAll(tied, {
                                         {"a\u0361ɪ", "I"},
                                         {"a\u0361ʊ", "W"},
                                         {"d\u0361z", "ʣ"},
                                         {"d\u0361ʒ", "ʤ"},
                                         {"e\u0361ɪ", "A"},
                                         {"o\u0361ʊ", "O"},
                                         {"ə\u0361ʊ", "Q"},
                                         {"s\u0361s", "S"},
                                         {"t\u0361s", "ʦ"},
                                         {"t\u0361ʃ", "ʧ"},
                                         {"ɔ\u0361ɪ", "Y"},
                                     });
    return absl::StrReplaceAll(tied, {{"\u0361", ""}, {"-", ""}});
  }

  // Step 1: Remove the Unicode tie-bar accent (U+0361 COMBINING DOUBLE INVERTED
  // BREVE, encoded in UTF-8 as 2 bytes: 0xCD 0xA1) and '^' tie marker.
  // Tie bars connect multi-character affricates (e.g., "t͡ʃ", "d͡ʒ"). Kokoro
  // vocabulary uses individual characters ("ʧ", "ʤ"), so stripping the tie bar
  // enables clean mapping in Step 2.
  std::string clean_ipa = absl::StrReplaceAll(
      raw_ipa, {{"\u0361", ""}, {"\xCD\xA1", ""}, {"^", ""}});

  if (flavor == MisakiFlavor::kJapanese) {
    return clean_ipa;
  }
  if (flavor == MisakiFlavor::kChinese) {
    return absl::StrReplaceAll(clean_ipa, {
                                              {"aɪ", "I"},
                                              {"aʊ", "W"},
                                              {"eɪ", "A"},
                                              {"ɔɪ", "Y"},
                                              {"oʊ", "O"},
                                              {"əʊ", "Q"},
                                              {"tʃ", "ʧ"},
                                              {"dʒ", "ʤ"},
                                          });
  }

  if (flavor == MisakiFlavor::kEnglishGb) {
    // British English normalization:
    return absl::StrReplaceAll(clean_ipa, {
                                              {"aɪ", "I"},
                                              {"aʊ", "W"},
                                              {"eɪ", "A"},
                                              {"ɔɪ", "Y"},
                                              {"eə", "ɛː"},
                                              {"iə", "ɪə"},
                                              {"əʊ", "Q"},
                                              {"tʃ", "ʧ"},
                                              {"dʒ", "ʤ"},
                                              {"əl", "ᵊl"},
                                              {"ɚ", "əɹ"},
                                              {"r", "ɹ"},
                                              {"x", "k"},
                                              {"ç", "k"},
                                              {"ɐ", "ə"},
                                              {"ɬ", "l"},
                                              {"o", "ɔ"},
                                          });
  }

  // Step 2: Apply Misaki G2P normalization rules to convert raw IPA sequences
  // into Kokoro's compact vocabulary representations (English US):
  // - English diphthongs -> single uppercase characters (e.g. "aɪ" -> 'I', "aʊ"
  // -> 'W',
  //   "eɪ" -> 'A', "ɔɪ" -> 'Y', "oʊ" -> 'O', "əʊ" -> 'Q').
  // - Affricate pairs -> single phonetic symbols ("tʃ" -> 'ʧ', "dʒ" -> 'ʤ').
  // - Syllabic & rhotic consonants -> normalized forms ("əl" -> 'ᵊl', "ɚ" ->
  // "əɹ", "r" -> 'ɹ').
  // - Rare fricatives & near-open vowels -> standard substitutes ("x"/"ç" ->
  // 'k', "ɐ" -> 'ə', "ɬ" -> 'l').
  // - Vowel length marker -> stripped ("ː" -> "").
  return absl::StrReplaceAll(clean_ipa, {
                                            {"aɪ", "I"},
                                            {"aʊ", "W"},
                                            {"eɪ", "A"},
                                            {"ɔɪ", "Y"},
                                            {"oʊ", "O"},
                                            {"əʊ", "Q"},
                                            {"tʃ", "ʧ"},
                                            {"dʒ", "ʤ"},
                                            {"əl", "ᵊl"},
                                            {"ɚ", "əɹ"},
                                            {"r", "ɹ"},
                                            {"x", "k"},
                                            {"ç", "k"},
                                            {"ɐ", "ə"},
                                            {"ɬ", "l"},
                                            {"ː", ""},
                                        });
}

bool IsWordCodePoint(char32_t cp) {
  // ASCII Alphanumeric, apostrophe, hyphen
  if ((cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z') ||
      (cp >= '0' && cp <= '9') || cp == '\'' || cp == '-') {
    return true;
  }
  // Latin-1 Letters (accented letters like á, é, í, ó, ú, ñ, ç): 0x00C0..0x00FF
  // (excluding × and ÷)
  if (cp >= 0x00C0 && cp <= 0x00FF) {
    return cp != 0x00D7 && cp != 0x00F7;
  }
  // Latin Extended (A, B, Additional)
  if ((cp >= 0x0100 && cp <= 0x024F) || (cp >= 0x1E00 && cp <= 0x1EFF)) {
    return true;
  }
  // Devanagari (Hindi): U+0900..U+097F (excluding dandas U+0964, U+0965)
  if (cp >= 0x0900 && cp <= 0x097F) {
    return cp != 0x0964 && cp != 0x0965;
  }
  // CJK Unified Ideographs (U+4E00 to U+9FFF) and Ideographic Number Zero 〇
  // (U+3007)
  if (cp == 0x3007 || (cp >= 0x4E00 && cp <= 0x9FFF)) {
    return true;
  }
  // Hiragana & Katakana: U+3040..U+30FF, excluding the katakana middle dot
  // U+30FB (・), which is a word *separator*. Leaving it classified as a word
  // character glues it onto the preceding token, where the G2P engine cannot
  // pronounce it and the boundary it marks is lost.
  if (cp >= 0x3040 && cp <= 0x30FF) {
    return cp != 0x30FB;
  }
  // Fullwidth Latin & Digits
  if ((cp >= 0xFF21 && cp <= 0xFF3A) || (cp >= 0xFF41 && cp <= 0xFF5A) ||
      (cp >= 0xFF10 && cp <= 0xFF19)) {
    return true;
  }
  return false;
}

// Returns the byte length of the UTF-8 sequence starting at `text[pos]`,
// clamped so it never runs past the end of `text`.
size_t Utf8SequenceLength(absl::string_view text, size_t pos) {
  return DecodeUtf8CodePoint(text, pos).length;
}

// Checks if the given text contains any Chinese (Hanzi) characters.
bool ContainsHanzi(absl::string_view text) {
  size_t i = 0;
  while (i < text.size()) {
    const Utf8CodePoint decoded = DecodeUtf8CodePoint(text, i);
    if (IsHanziCodePoint(decoded.code_point)) {
      return true;
    }
    i += decoded.length;
  }
  return false;
}

// Multi-byte Unicode punctuation that Kokoro's vocabulary or phonetic mapping
// expects, mapped to its target replacement representation.
const absl::flat_hash_map<absl::string_view, absl::string_view>&
GetUnicodePunctuationMap() {
  static const absl::NoDestructor<
      absl::flat_hash_map<absl::string_view, absl::string_view>>
      kMap({
          {"—", "—"},
          {"–", "—"},
          {"…", "…"},
          {"“", "“"},
          {"”", "”"},
          {"，", ","},
          {"。", "."},
          {"！", "!"},
          {"？", "?"},
          {"।", "."},
          {"॥", "."},
          // CJK punctuation. Without these the characters reach
          // TextToPhonemeIds as raw UTF-8, match nothing in the vocabulary and
          // are dropped, so the prosodic break they encode is lost entirely.
          {"、", ","},
          {"；", ";"},
          {"：", ":"},
          {"（", "("},
          {"）", ")"},
          {"《", "“"},
          {"》", "”"},
          {"〈", "("},
          {"〉", ")"},
          {"「", "“"},
          {"」", "”"},
          {"『", "“"},
          {"』", "”"},
          {"・", " "},
          {"〜", "—"},
          {"～", "—"},
          {"〖", "("},
          {"〗", ")"},
          {"【", "("},
          {"】", ")"},
      });
  return *kMap;
}

std::string NormalizeLanguageCode(absl::string_view language_code) {
  if (language_code.empty()) return "en-us";
  std::string lower = absl::AsciiStrToLower(language_code);
  for (char& c : lower) {
    if (c == '_') c = '-';
  }

  static const absl::NoDestructor<absl::flat_hash_map<std::string, std::string>>
      kLanguageMap({
          // American English ('a')
          {"a", "en-us"},
          {"en", "en-us"},
          {"en-us", "en-us"},
          {"american", "en-us"},
          {"english", "en-us"},
          {"american english", "en-us"},
          // British English ('b')
          {"b", "en-gb"},
          {"en-gb", "en-gb"},
          {"en-uk", "en-gb"},
          {"british", "en-gb"},
          {"british english", "en-gb"},
          // Spanish ('e')
          {"e", "es"},
          {"es", "es"},
          {"es-es", "es"},
          {"es-419", "es"},
          {"spanish", "es"},
          // French ('f')
          {"f", "fr-fr"},
          {"fr", "fr-fr"},
          {"fr-fr", "fr-fr"},
          {"french", "fr-fr"},
          // Hindi ('h')
          {"h", "hi"},
          {"hi", "hi"},
          {"hindi", "hi"},
          // Italian ('i')
          {"i", "it"},
          {"it", "it"},
          {"italian", "it"},
          // Brazilian Portuguese ('p')
          {"p", "pt-br"},
          {"pt", "pt-br"},
          {"pt-br", "pt-br"},
          {"portuguese", "pt-br"},
          // Japanese ('j')
          {"j", "ja"},
          {"ja", "ja"},
          {"japanese", "ja"},
          // Mandarin Chinese ('z' -> "cmn")
          {"z", "cmn"},
          {"zh", "cmn"},
          {"cmn", "cmn"},
          {"zh-cmn", "cmn"},
          {"zh-cn", "cmn"},
          {"zh-tw", "cmn"},
          {"zh-hans", "cmn"},
          {"zh-hant", "cmn"},
          {"chinese", "cmn"},
          {"mandarin", "cmn"},
      });

  auto it = kLanguageMap->find(lower);
  if (it != kLanguageMap->end()) {
    return it->second;
  }
  size_t dash = lower.find('-');
  if (dash != std::string::npos) {
    it = kLanguageMap->find(lower.substr(0, dash));
    if (it != kLanguageMap->end()) {
      return it->second;
    }
  }
  return lower;
}

std::string LanguageForVoiceName(absl::string_view voice_name) {
  if (voice_name.empty()) return "";
  // Extract filename stem: strip directory and extension.
  absl::string_view filename = voice_name;
  size_t last_slash = filename.find_last_of("/\\");
  if (last_slash != absl::string_view::npos) {
    filename = filename.substr(last_slash + 1);
  }
  if (absl::EndsWith(filename, ".bin")) {
    filename = filename.substr(0, filename.size() - 4);
  }

  // Kokoro voice naming convention: 2-character prefix + '_' + name
  // First char indicates language:
  // 'a' -> "en-us"
  // 'b' -> "en-gb"
  // 'e' -> "es"
  // 'f' -> "fr-fr"
  // 'h' -> "hi"
  // 'i' -> "it"
  // 'p' -> "pt-br"
  // 'j' -> "ja"
  // 'z' -> "cmn"
  if (filename.size() >= 3 && filename[2] == '_') {
    char lang_char = absl::ascii_tolower(filename[0]);
    switch (lang_char) {
      case 'a':
        return "en-us";
      case 'b':
        return "en-gb";
      case 'e':
        return "es";
      case 'f':
        return "fr-fr";
      case 'h':
        return "hi";
      case 'i':
        return "it";
      case 'p':
        return "pt-br";
      case 'j':
        return "ja";
      case 'z':
        return "cmn";
      default:
        break;
    }
  }
  return "";
}

absl::StatusOr<std::string> ResolveAndValidateLanguage(
    absl::string_view voice_name, absl::string_view language) {
  std::string voice_lang = LanguageForVoiceName(voice_name);
  if (language.empty()) {
    return voice_lang.empty() ? "en-us" : voice_lang;
  }
  std::string normalized_lang = NormalizeLanguageCode(language);
  if (!voice_lang.empty() && normalized_lang != voice_lang) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Voice '%s' (language: %s) conflicts with requested language '%s' "
        "(normalized: %s)",
        voice_name, voice_lang, language, normalized_lang));
  }
  return normalized_lang;
}

absl::StatusOr<std::unique_ptr<KokoroPhonemizer>> KokoroPhonemizer::Create(
    absl::string_view data_dir, absl::string_view language,
    const absl::flat_hash_map<std::string, std::string>& custom_lexicon,
    absl::string_view text_norm_rules, absl::string_view cjk_lexicon,
    absl::string_view zh_t2s_dictionary) {
  const std::string normalized_language = NormalizeLanguageCode(language);
    std::string resolved_data_dir(data_dir);

  auto phonemizer = std::unique_ptr<KokoroPhonemizer>(new KokoroPhonemizer());
  phonemizer->data_dir_ = resolved_data_dir;
  phonemizer->language_ = normalized_language;
  // Every rule-based engine is created up front so that `SetLanguage` can
  // switch to any of their languages later.
  ABSL_RETURN_IF_ERROR(
      AddLanguageG2p<kokoro::SpanishG2p>("es", phonemizer->language_g2ps_));
  ABSL_RETURN_IF_ERROR(
      AddLanguageG2p<kokoro::HindiG2p>("hi", phonemizer->language_g2ps_));
  ABSL_RETURN_IF_ERROR(AddLanguageG2p<kokoro::PortugueseG2p>(
      "pt-br", phonemizer->language_g2ps_));
  ABSL_RETURN_IF_ERROR(
      AddLanguageG2p<kokoro::ItalianG2p>("it", phonemizer->language_g2ps_));
  ABSL_RETURN_IF_ERROR(
      AddLanguageG2p<kokoro::FrenchG2p>("fr-fr", phonemizer->language_g2ps_));

  if (!text_norm_rules.empty()) {
    ABSL_ASSIGN_OR_RETURN(phonemizer->text_normalizer_,
                          TextNormalizer::Create(text_norm_rules));
    phonemizer->text_norm_language_ = phonemizer->language_;
  }
  if (!zh_t2s_dictionary.empty()) {
    ABSL_ASSIGN_OR_RETURN(
        phonemizer->traditional_chinese_converter_,
        kokoro::TraditionalChineseConverter::Create(zh_t2s_dictionary));
  }

  // Determine the parent cache directory path expected by ChineseG2p (which
  // unpacks the cppjieba HMM model file into cache_dir).
  std::string parent_dir = phonemizer->data_dir_;
  if (parent_dir.empty()) {
    parent_dir = ".";
  }

  if (!cjk_lexicon.empty()) {
    ABSL_ASSIGN_OR_RETURN(kokoro::CjkBlob blob,
                          kokoro::CjkBlob::Create(cjk_lexicon));
    phonemizer->cjk_blob_ = std::make_unique<kokoro::CjkBlob>(std::move(blob));
    if (phonemizer->cjk_blob_->Contains("us-idx")) {
      ABSL_ASSIGN_OR_RETURN(kokoro::EnglishG2p en_g2p,
                            kokoro::EnglishG2p::Create(*phonemizer->cjk_blob_));
      phonemizer->english_g2p_ =
          std::make_unique<kokoro::EnglishG2p>(std::move(en_g2p));
    } else if (phonemizer->language_ == "ja" ||
               phonemizer->cjk_blob_->Contains("sys-da")) {
      ABSL_ASSIGN_OR_RETURN(
          kokoro::JapaneseG2p ja_g2p,
          kokoro::JapaneseG2p::Create(*phonemizer->cjk_blob_));
      phonemizer->japanese_g2p_ =
          std::make_unique<kokoro::JapaneseG2p>(std::move(ja_g2p));
    } else if (phonemizer->language_ == "cmn" ||
               phonemizer->cjk_blob_->Contains("w-frq")) {
      ABSL_ASSIGN_OR_RETURN(
          kokoro::ChineseG2p zh_g2p,
          kokoro::ChineseG2p::Create(*phonemizer->cjk_blob_, parent_dir));
      phonemizer->chinese_g2p_ =
          std::make_unique<kokoro::ChineseG2p>(std::move(zh_g2p));
    }
  }
    if (phonemizer->english_g2p_ == nullptr) {
    ABSL_ASSIGN_OR_RETURN(kokoro::EnglishG2p en_g2p,
                          kokoro::EnglishG2p::Create());
    phonemizer->english_g2p_ =
        std::make_unique<kokoro::EnglishG2p>(std::move(en_g2p));
  }

  // Pre-seed default project-specific pronunciation overrides.
  phonemizer->merged_lexicon_ = {
      {"litert", "lˌItˌɑɹtˈi"},
      {"tts", "tˌiːtˌiːˈɛs"},
      {"kokoro", "kˈoʊkəɹoʊ"},
      {"github", "ɡˈɪthʌb"},
  };
  for (const auto& [w, ipa] : custom_lexicon) {
    const std::string low_w = absl::AsciiStrToLower(w);
    phonemizer->custom_lexicon_[low_w] =
        NormalizeMisakiPhonemes(ipa, FlavorForLanguage(phonemizer->language_));
    phonemizer->merged_lexicon_[low_w] = ipa;
  }

  phonemizer->SetLanguage(language);
  ABSL_LOG(INFO) << "KokoroPhonemizer initialized successfully for language '"
                 << phonemizer->language_
                 << "' from: " << phonemizer->data_dir_;

  return phonemizer;
}

absl::StatusOr<std::unique_ptr<KokoroPhonemizer>> KokoroPhonemizer::Create(
    absl::string_view data_dir,
    const absl::flat_hash_map<std::string, std::string>& custom_lexicon) {
  return Create(data_dir, "en-us", custom_lexicon);
}

absl::string_view KokoroPhonemizer::Language() const { return language_; }

void KokoroPhonemizer::SetLanguage(absl::string_view language) {
  language_ = NormalizeLanguageCode(language);
}

absl::StatusOr<std::string> KokoroPhonemizer::WordToIpa(
    absl::string_view word) const {
  if (word.empty()) {
    return "";
  }
  std::string lower_word = absl::AsciiStrToLower(word);
  auto it = merged_lexicon_.find(lower_word);
  if (it != merged_lexicon_.end()) {
    return it->second;
  }

  if (language_ == "ja") {
    if (japanese_g2p_ == nullptr) {
      return absl::FailedPreconditionError(kMissingJapaneseLexiconError);
    }
    return japanese_g2p_->TextToIpa(word);
  }
  if ((language_ == "en-us" || language_ == "en-gb") &&
      english_g2p_ != nullptr) {
    return english_g2p_->WordToIpa(word, /*british=*/language_ == "en-gb");
  }
  if (const auto it = language_g2ps_.find(language_);
      it != language_g2ps_.end()) {
    return it->second->WordToIpa(word, custom_lexicon_);
  }

  if (ContainsHanzi(word)) {
    if (chinese_g2p_ == nullptr) {
      return absl::FailedPreconditionError(
          "Chinese TTS requires a bundled 'zh-lexicon' section in the "
          ".litertlm model container.");
    }
    ABSL_ASSIGN_OR_RETURN(const std::vector<absl::string_view> subwords,
                          chinese_g2p_->Segment(word));
    std::vector<std::string> pieces;
    pieces.reserve(subwords.size());
    for (absl::string_view subword : subwords) {
      std::string sub_ipa;
      if (!chinese_g2p_->WordToIpa(subword, sub_ipa)) {
        return absl::NotFoundError(
            absl::StrCat("Failed to phonemize Chinese token '", subword,
                         "' in zh-lexicon."));
      }
      if (absl::StartsWith(sub_ipa, "i")) {
        sub_ipa.insert(0, 1, 'j');
      }
      if (subword.size() > 3 && absl::EndsWith(subword, "儿") &&
          absl::EndsWith(sub_ipa, "nɚ↗")) {
        sub_ipa.resize(sub_ipa.size() - std::string("nɚ↗").size());
        sub_ipa.append("ɚ↗");
      }
      pieces.push_back(std::move(sub_ipa));
    }
    return absl::StrJoin(pieces, " ");
  }
  if (language_ == "cmn" && english_g2p_ != nullptr) {
    return english_g2p_->WordToIpa(word, /*british=*/false);
  }

  return absl::FailedPreconditionError(
      absl::StrCat("Unsupported language: ", language_));
}

absl::Status KokoroPhonemizer::FlushWordToIpa(std::string& current_word,
                                              std::string& combined_ipa) const {
  if (current_word.empty()) return absl::OkStatus();
  std::string lower_word = absl::AsciiStrToLower(current_word);
  auto it = merged_lexicon_.find(lower_word);
  std::string word_ipa;
  if (it != merged_lexicon_.end()) {
    word_ipa = it->second;
  } else {
    ABSL_ASSIGN_OR_RETURN(word_ipa, WordToIpa(current_word));
  }
  if (!word_ipa.empty()) {
    // Add separating whitespace between words unless following open
    // punctuation.
    if (!combined_ipa.empty() && combined_ipa.back() != ' ' &&
        !IsOpenPunctuation(combined_ipa.back())) {
      combined_ipa += ' ';
    }
    combined_ipa += word_ipa;
  }
  current_word.clear();
  return absl::OkStatus();
}

absl::StatusOr<std::string> KokoroPhonemizer::TextToIpa(
    absl::string_view text) const {
  // The Mandarin rule table and lexicon are Simplified Chinese; bring
  // Traditional input to Simplified first so that both apply.
  std::string simplified_text;
  if (language_ == "cmn" && traditional_chinese_converter_ != nullptr) {
    simplified_text = traditional_chinese_converter_->Convert(text);
    text = simplified_text;
  }

  // Rewrite digits and symbols into pronounceable words before tokenizing, so
  // that the rules can see a number whole rather than one word at a time.
  std::string normalized_text;
  if (text_normalizer_ != nullptr && language_ == text_norm_language_) {
    normalized_text = text_normalizer_->Normalize(text);
    text = normalized_text;
  }

  if ((language_ == "en-us" || language_ == "en-gb") &&
      english_g2p_ != nullptr) {
    return english_g2p_->TextToIpa(text, /*british=*/language_ == "en-gb",
                                   custom_lexicon_);
  }
  if (const auto it = language_g2ps_.find(language_);
      it != language_g2ps_.end()) {
    return it->second->TextToIpa(text, custom_lexicon_);
  }

  if (language_ == "ja") {
    if (japanese_g2p_ == nullptr) {
      return absl::FailedPreconditionError(kMissingJapaneseLexiconError);
    }
    return japanese_g2p_->TextToIpa(text);
  }
  if (language_ == "cmn" && chinese_g2p_ == nullptr) {
    return absl::FailedPreconditionError(
        "Chinese TTS requires a bundled 'zh-lexicon' section in the "
        ".litertlm model container.");
  }

  if (language_ != "cmn" && language_ != "ja") {
    return absl::FailedPreconditionError(
        absl::StrCat("Unsupported language: ", language_));
  }

  std::string combined_ipa;
  std::string current_word;

  const auto& unicode_punct_map = GetUnicodePunctuationMap();

  size_t i = 0;
  while (i < text.size()) {
    const Utf8CodePoint decoded = DecodeUtf8CodePoint(text, i);
    const char32_t cp = decoded.code_point;
    const size_t char_len = decoded.length;
    const unsigned char c = static_cast<unsigned char>(text[i]);
    absl::string_view symbol = text.substr(i, char_len);
    i += char_len;

    // 1. Whitespace handling (' ', '\t', '\n', '\r', etc.)
    if (char_len == 1 && absl::ascii_isspace(c)) {
      ABSL_RETURN_IF_ERROR(FlushWordToIpa(current_word, combined_ipa));
      if (!combined_ipa.empty() && combined_ipa.back() != ' ') {
        combined_ipa += ' ';
      }
      continue;
    }

    // 2. Curly apostrophe: U+2019 (’ : 0xE2 0x80 0x99) or U+2018 (‘)
    // If inside a word (e.g. don’t), treat as an ASCII apostrophe '\''.
    if (symbol == "’" || symbol == "‘") {
      if (!current_word.empty()) {
        current_word += '\'';
        continue;
      }
    }

    // 3. Known multi-byte Unicode punctuation
    if (auto it = unicode_punct_map.find(symbol);
        it != unicode_punct_map.end()) {
      ABSL_RETURN_IF_ERROR(FlushWordToIpa(current_word, combined_ipa));
      absl::StrAppend(&combined_ipa, it->second);
      continue;
    }

    // 4. Word characters (ASCII alnum, Latin Extended, Devanagari, CJK, Kana)
    if (IsWordCodePoint(cp)) {
      if (!current_word.empty() &&
          ContainsHanzi(current_word) != IsHanziCodePoint(cp)) {
        ABSL_RETURN_IF_ERROR(FlushWordToIpa(current_word, combined_ipa));
      }
      current_word.append(symbol.data(), symbol.size());
      continue;
    }

    // 5. Standard ASCII punctuation / delimiters (e.g. '.', ',', '!', '?', ';',
    // ':', '(', ')')
    ABSL_RETURN_IF_ERROR(FlushWordToIpa(current_word, combined_ipa));
    combined_ipa.append(symbol.data(), symbol.size());
  }
  ABSL_RETURN_IF_ERROR(FlushWordToIpa(current_word, combined_ipa));

  // Normalize IPA output to match Kokoro's vocabulary symbols.
  return NormalizeMisakiPhonemes(combined_ipa, FlavorForLanguage(language_));
}

absl::StatusOr<std::vector<int>> KokoroPhonemizer::TextToPhonemeIds(
    absl::string_view text) const {
  if (absl::StripAsciiWhitespace(text).empty()) {
    return std::vector<int>();
  }

  // Convert raw input text to normalized IPA phoneme transcript.
  ABSL_ASSIGN_OR_RETURN(std::string ipa_str, TextToIpa(text));
  if (absl::StripAsciiWhitespace(ipa_str).empty()) {
    return std::vector<int>();
  }

  // Start with the BOS token (ID 0). The sequence is not capped here:
  // `KokoroAcousticStage` slices it to the model capacity, so a cap would only
  // drop the tail of long chunks silently.
  std::vector<int> ids;
  ids.reserve(ipa_str.size() + 2);
  ids.push_back(kokoro::kBosTokenId);
  const auto& vocab = GetKokoroVocabMap();

  size_t pos = 0;
  bool has_phonetic_token = false;
  int dropped_symbols = 0;

  // Iterate over the UTF-8 encoded IPA string character by character (Unicode
  // codepoints).
  while (pos < ipa_str.size()) {
    const size_t char_len = Utf8SequenceLength(ipa_str, pos);
    absl::string_view symbol = absl::string_view(ipa_str).substr(pos, char_len);
    pos += char_len;

    // Map the Unicode symbol to its corresponding integer token ID in Kokoro's
    // vocabulary.
    auto it = vocab.find(symbol);
    if (it != vocab.end()) {
      ids.push_back(it->second);
      if (it->second != kokoro::kSpaceTokenId) {
        has_phonetic_token = true;
      }
    } else if (symbol == " ") {
      ids.push_back(kokoro::kSpaceTokenId);
    } else {
      // Silently dropping symbols degrades pronunciation invisibly (e.g.
      // lowered/centralized diacritics for Japanese), so surface a single
      // summary instead of discarding them without a trace.
      ++dropped_symbols;
    }
  }

  if (dropped_symbols > 0) {
    ABSL_LOG(WARNING) << "Dropped " << dropped_symbols
                      << " phoneme symbol(s) outside the Kokoro vocabulary "
                         "while encoding language \""
                      << language_ << "\".";
  }

  // If no actual phonetic or punctuation tokens were mapped, return empty.
  if (!has_phonetic_token) {
    return std::vector<int>();
  }

  // Append the trailing EOS token.
  ids.push_back(kokoro::kEosTokenId);
  return ids;
}

}  // namespace litert::omni::tts
