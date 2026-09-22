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

#include "omni/tts/kokoro/phonemizer.h"

#include <cstddef>
#include <filesystem>  // NOLINT: Required for path manipulation.
#include <memory>
#include <string>
#include <system_error>  // NOLINT: Required by std::filesystem.
#include <vector>

#include "absl/base/attributes.h"  // from @com_google_absl
#include "absl/base/const_init.h"  // from @com_google_absl
#include "absl/base/no_destructor.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/ascii.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/str_replace.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "espeak-ng/espeak_ng.h"  // from @espeak_ng
#include "espeak-ng/speak_lib.h"  // from @espeak_ng
#include "litert/cc/litert_macros.h"  // from @litert
#include "omni/tts/kokoro/common.h"

namespace litert::omni::tts {

ABSL_CONST_INIT absl::Mutex KokoroPhonemizer::espeak_mutex_(absl::kConstInit);

namespace {

// Thread-safe one-time initialization of the process-global espeak-ng C
// library.
//
// The third-party `libespeak-ng` library is written in legacy C without
// instance context handles, storing internal phonetic lookup tables, voice
// structures, and dictionary file pointers in process-global C static
// variables.
absl::Status EnsureEspeakInitialized(absl::string_view parent_dir) {
  static const absl::NoDestructor<absl::Status> init_status([parent_dir]() {
    std::string path_str(parent_dir);
    espeak_ng_InitializePath(path_str.c_str());
    int status =
        espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS, 0, path_str.c_str(),
                          espeakINITIALIZE_DONT_EXIT);
    if (status <= 0) {
      return absl::InternalError(absl::StrCat(
          "espeak_Initialize failed with parent_dir: ", parent_dir));
    }
    return absl::OkStatus();
  }());
  return *init_status;
}

// Checks whether a character is an opening delimiter (e.g. '(', '[', '"').
// Used during text phonemization to avoid inserting unnecessary leading spaces
// immediately after an opening bracket or quotation mark.
bool IsOpenPunctuation(char ch) {
  return ch == '(' || ch == '[' || ch == '{' || ch == '"' || ch == '\'';
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

MisakiFlavor FlavorForLanguage(absl::string_view espeak_voice) {
  std::string normalized = NormalizeLanguageCode(espeak_voice);
  if (normalized == "en-us") {
    return MisakiFlavor::kEnglishUs;
  }
  if (normalized == "en-gb") {
    return MisakiFlavor::kEnglishGb;
  }
  return MisakiFlavor::kEspeakGeneric;
}

std::string NormalizeMisakiPhonemes(absl::string_view raw_ipa,
                                    MisakiFlavor flavor) {
  // Step 1: Remove the Unicode tie-bar accent (U+0361 COMBINING DOUBLE INVERTED
  // BREVE, encoded in UTF-8 as 2 bytes: 0xCD 0xA1) and '^' tie marker.
  // eSpeak-ng uses this tie bar to connect multi-character affricates (e.g.,
  // "t͡ʃ", "d͡ʒ"). Kokoro vocabulary uses individual characters ("ʧ", "ʤ"), so
  // stripping the tie bar enables clean mapping in Step 2.
  std::string clean_ipa = absl::StrReplaceAll(
      raw_ipa, {{"\u0361", ""}, {"\xCD\xA1", ""}, {"^", ""}});

  if (flavor == MisakiFlavor::kEspeakGeneric) {
    // misaki.espeak.EspeakG2P.E2M rules (used for es, fr-fr, hi, it, pt-br):
    // Preserves Spanish jota ('x'), trill ('r'), and vowel length marker ('ː').
    return absl::StrReplaceAll(clean_ipa, {
                                              {"aɪ", "I"},
                                              {"aʊ", "W"},
                                              {"dz", "ʣ"},
                                              {"dʒ", "ʤ"},
                                              {"eɪ", "A"},
                                              {"oʊ", "O"},
                                              {"əʊ", "Q"},
                                              {"ss", "S"},
                                              {"ts", "ʦ"},
                                              {"tʃ", "ʧ"},
                                              {"ɔɪ", "Y"},
                                              {"-", ""},
                                          });
  }

  if (flavor == MisakiFlavor::kEnglishGb) {
    // misaki.espeak.EspeakFallback (british=True):
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

// Decodes the UTF-8 codepoint starting at `text[pos]`, advancing `char_len`.
char32_t DecodeUtf8Char(absl::string_view text, size_t pos, size_t* char_len) {
  if (pos >= text.size()) {
    *char_len = 0;
    return 0;
  }
  const unsigned char c0 = static_cast<unsigned char>(text[pos]);
  if ((c0 & 0x80) == 0) {
    *char_len = 1;
    return c0;
  }
  if ((c0 & 0xE0) == 0xC0 && pos + 1 < text.size()) {
    *char_len = 2;
    const unsigned char c1 = static_cast<unsigned char>(text[pos + 1]);
    return (static_cast<char32_t>(c0 & 0x1F) << 6) |
           static_cast<char32_t>(c1 & 0x3F);
  }
  if ((c0 & 0xF0) == 0xE0 && pos + 2 < text.size()) {
    *char_len = 3;
    const unsigned char c1 = static_cast<unsigned char>(text[pos + 1]);
    const unsigned char c2 = static_cast<unsigned char>(text[pos + 2]);
    return (static_cast<char32_t>(c0 & 0x0F) << 12) |
           (static_cast<char32_t>(c1 & 0x3F) << 6) |
           static_cast<char32_t>(c2 & 0x3F);
  }
  if ((c0 & 0xF8) == 0xF0 && pos + 3 < text.size()) {
    *char_len = 4;
    const unsigned char c1 = static_cast<unsigned char>(text[pos + 1]);
    const unsigned char c2 = static_cast<unsigned char>(text[pos + 2]);
    const unsigned char c3 = static_cast<unsigned char>(text[pos + 3]);
    return (static_cast<char32_t>(c0 & 0x07) << 18) |
           (static_cast<char32_t>(c1 & 0x3F) << 12) |
           (static_cast<char32_t>(c2 & 0x3F) << 6) |
           static_cast<char32_t>(c3 & 0x3F);
  }
  *char_len = 1;
  return c0;
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
  // CJK Unified Ideographs: U+4E00..U+9FFF
  if (cp >= 0x4E00 && cp <= 0x9FFF) {
    return true;
  }
  // Hiragana & Katakana: U+3040..U+30FF
  if (cp >= 0x3040 && cp <= 0x30FF) {
    return true;
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
  size_t len = 0;
  DecodeUtf8Char(text, pos, &len);
  return len;
}

// Multi-byte Unicode punctuation that Kokoro's vocabulary or phonetic mapping
// expects, mapped to its target replacement representation.
const absl::flat_hash_map<absl::string_view, absl::string_view>&
GetUnicodePunctuationMap() {
  static const auto* kMap =
      new absl::flat_hash_map<absl::string_view, absl::string_view>{
          {"—", "—"},  {"–", "—"},  {"…", "…"},  {"“", "“"},
          {"”", "”"},  {"，", ","}, {"。", "."}, {"！", "!"},
          {"？", "?"}, {"।", "."},  {"॥", "."},
      };
  return *kMap;
}

std::string ResolveEspeakDataDir(absl::string_view path) {
  if (path.empty()) return "";

  // Check candidate paths: the path itself, a subfolder named
  // "espeak-ng-data", or the parent directory's "espeak-ng-data" when `path`
  // is a `.litertlm` model file.
  std::filesystem::path base_path = std::string(path);
  std::vector<std::filesystem::path> candidates = {
      base_path,
      base_path / "espeak-ng-data",
  };
  std::error_code ec;
  if (std::filesystem::is_regular_file(base_path, ec) && !ec) {
    candidates.push_back(base_path.parent_path() / "espeak-ng-data");
  }
  for (const auto& candidate : candidates) {
    // A valid espeak-ng data directory MUST contain the "phontab" table file.
    if (std::filesystem::exists(candidate / "phontab", ec) && !ec) {
      return candidate.string();
    }
  }
  return "";
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
          // Mandarin Chinese ('z' -> "cmn" in espeak-ng)
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

std::string EspeakVoiceForLanguage(absl::string_view language_code) {
  // `espeak_SetVoiceByName` resolves a name against the voice file identifier
  // (e.g. "roa/es", "gmw/en-US"), never against the `language` attributes
  // declared inside the file. French is the only Kokoro language whose file
  // name differs from its misaki code: `roa/fr` declares "fr-fr" as a language
  // attribute only, so "fr-fr" resolves to nothing and every French word fails
  // to phonemize.
  if (language_code == "fr-fr") {
    return "fr";
  }
  if (language_code == "en-gb") {
    return "en";
  }
  return std::string(language_code);
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
    absl::string_view espeak_data_dir, absl::string_view language,
    const absl::flat_hash_map<std::string, std::string>& custom_lexicon) {
  if (espeak_data_dir.empty()) {
    return absl::InvalidArgumentError("espeak_data_dir cannot be empty");
  }

  std::string data_dir = ResolveEspeakDataDir(espeak_data_dir);
  if (data_dir.empty()) {
    return absl::NotFoundError(absl::StrCat(
        "Could not find espeak-ng-data directory with phontab. Given path: '",
        espeak_data_dir, "'"));
  }

  auto phonemizer = std::unique_ptr<KokoroPhonemizer>(new KokoroPhonemizer());
  phonemizer->data_dir_ = data_dir;
  phonemizer->language_ = NormalizeLanguageCode(language);

  // Pre-seed default project-specific pronunciation overrides.
  phonemizer->merged_lexicon_ = {
      {"litert", "lˌItˌɑɹtˈi"},
      {"tts", "tˌiːtˌiːˈɛs"},
      {"kokoro", "kˈoʊkəɹoʊ"},
      {"github", "ɡˈɪthʌb"},
  };
  for (const auto& [w, ipa] : custom_lexicon) {
    phonemizer->merged_lexicon_[absl::AsciiStrToLower(w)] = ipa;
  }

  // Determine the parent directory path expected by espeak_Initialize.
  // If data_dir_ is "/path/to/espeak-ng-data", parent_dir is "/path/to".
  std::string parent_dir = phonemizer->data_dir_;
  if (parent_dir.size() >= 15 &&
      absl::EndsWith(parent_dir, "/espeak-ng-data")) {
    parent_dir = parent_dir.substr(0, parent_dir.size() - 15);
  }
  if (parent_dir.empty()) {
    parent_dir = ".";
  }

  // Initialize the process-global espeak-ng G2P C library with the explicit
  // parent path. The class-level static mutex protects one-time library
  // initialization and voice table selection.
  auto init_status = EnsureEspeakInitialized(parent_dir);
  if (!init_status.ok()) {
    return init_status;
  }

  phonemizer->SetLanguage(language);
  ABSL_LOG(INFO) << "espeak-ng G2P initialized successfully for language '"
                 << phonemizer->language_
                 << "' from: " << phonemizer->data_dir_;

  return phonemizer;
}

void KokoroPhonemizer::SetLanguage(absl::string_view language) {
  language_ = NormalizeLanguageCode(language);
  espeak_voice_ = EspeakVoiceForLanguage(language_);
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

  absl::MutexLock lock(espeak_mutex_);
  // Ensure active espeak voice matches this phonemizer's target language.
  if (espeak_SetVoiceByName(espeak_voice_.c_str()) != EE_OK) {
    return absl::InternalError(absl::StrCat("Failed to set espeak voice '",
                                            espeak_voice_,
                                            "' for language: ", language_));
  }

  std::string word_str(word);
  const void* text_ptr = word_str.c_str();

  // text_mode = espeakCHARS_AUTO: Auto-detects input character encoding (UTF-8
  // / ASCII).
  constexpr int text_mode = espeakCHARS_AUTO;

  // phoneme_mode = 0x02: Flag bit 1 (espeakPHONEMES_IPA = 0x02) directs
  // espeak_TextToPhonemes to produce standard International Phonetic Alphabet
  // (IPA) UTF-8 strings rather than espeak's internal ASCII phonetic
  // representation.
  constexpr int phoneme_mode = 0x02;

  std::string res;
  while (text_ptr != nullptr && *static_cast<const char*>(text_ptr) != '\0') {
    const void* prev_ptr = text_ptr;
    const char* ph = espeak_TextToPhonemes(&text_ptr, text_mode, phoneme_mode);
    if (ph != nullptr) {
      res += ph;
    }
    // Prevent infinite loop if text_ptr is not advanced by espeak.
    if (text_ptr == prev_ptr) {
      text_ptr = static_cast<const char*>(text_ptr) + 1;
    }
  }
  return res;
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
    LITERT_ASSIGN_OR_RETURN(word_ipa, WordToIpa(current_word));
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
  std::string combined_ipa;
  std::string current_word;

  const auto& unicode_punct_map = GetUnicodePunctuationMap();

  size_t i = 0;
  while (i < text.size()) {
    size_t char_len = 0;
    const char32_t cp = DecodeUtf8Char(text, i, &char_len);
    const unsigned char c = static_cast<unsigned char>(text[i]);
    absl::string_view symbol = text.substr(i, char_len);
    i += char_len;

    // 1. Whitespace handling (' ', '\t', '\n', '\r', etc.)
    if (char_len == 1 && absl::ascii_isspace(c)) {
      LITERT_RETURN_IF_ERROR(FlushWordToIpa(current_word, combined_ipa));
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
      LITERT_RETURN_IF_ERROR(FlushWordToIpa(current_word, combined_ipa));
      absl::StrAppend(&combined_ipa, it->second);
      continue;
    }

    // 4. Word characters (ASCII alnum, Latin Extended, Devanagari, CJK, Kana)
    if (IsWordCodePoint(cp)) {
      current_word.append(symbol.data(), symbol.size());
      continue;
    }

    // 5. Standard ASCII punctuation / delimiters (e.g. '.', ',', '!', '?', ';',
    // ':', '(', ')')
    LITERT_RETURN_IF_ERROR(FlushWordToIpa(current_word, combined_ipa));
    combined_ipa.append(symbol.data(), symbol.size());
  }
  LITERT_RETURN_IF_ERROR(FlushWordToIpa(current_word, combined_ipa));

  // Normalize IPA output to match Kokoro's vocabulary symbols.
  return NormalizeMisakiPhonemes(combined_ipa, FlavorForLanguage(language_));
}

absl::StatusOr<std::vector<int>> KokoroPhonemizer::TextToPhonemeIds(
    absl::string_view text) const {
  if (absl::StripAsciiWhitespace(text).empty()) {
    return std::vector<int>();
  }

  // Convert raw input text to normalized IPA phoneme transcript.
  LITERT_ASSIGN_OR_RETURN(std::string ipa_str, TextToIpa(text));
  if (absl::StripAsciiWhitespace(ipa_str).empty()) {
    return std::vector<int>();
  }

  // Initialize output token buffer with BOS token (ID 0) at index 0.
  std::vector<int> ids(kokoro::kMaxTokens, kokoro::kBosTokenId);
  const auto& vocab = GetKokoroVocabMap();

  int idx = 1;
  size_t pos = 0;
  bool has_phonetic_token = false;

  // Iterate over the UTF-8 encoded IPA string character by character (Unicode
  // codepoints).
  while (pos < ipa_str.size() && idx < kokoro::kMaxTokens - 1) {
    const size_t char_len = Utf8SequenceLength(ipa_str, pos);
    absl::string_view symbol = absl::string_view(ipa_str).substr(pos, char_len);
    pos += char_len;

    // Map the Unicode symbol to its corresponding integer token ID in Kokoro's
    // vocabulary.
    auto it = vocab.find(symbol);
    if (it != vocab.end()) {
      ids[idx++] = it->second;
      if (it->second != kokoro::kSpaceTokenId) {
        has_phonetic_token = true;
      }
    } else if (symbol == " ") {
      ids[idx++] = kokoro::kSpaceTokenId;
    }
  }

  // If no actual phonetic or punctuation tokens were mapped, return empty.
  if (!has_phonetic_token) {
    return std::vector<int>();
  }

  // Resize token array to include the trailing EOS token (ID 0).
  ids.resize(idx + 1);
  return ids;
}

}  // namespace litert::omni::tts
