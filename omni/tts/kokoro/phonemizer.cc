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

#include <unistd.h>

#include <cctype>
#include <cstddef>
#include <filesystem>  // NOLINT: Required for path manipulation.
#include <memory>
#include <string>
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

std::string NormalizeMisakiPhonemes(absl::string_view raw_ipa) {
  // Step 1: Remove the Unicode tie-bar accent (U+0361 COMBINING DOUBLE INVERTED
  // BREVE, encoded in UTF-8 as 2 bytes: 0xCD 0xA1).
  // eSpeak-ng uses this tie bar to connect multi-character affricates (e.g.,
  // "t͡ʃ", "d͡ʒ"). Kokoro vocabulary uses individual characters ("ʧ", "ʤ"), so
  // stripping the tie bar enables clean mapping in Step 2.
  std::string clean_ipa =
      absl::StrReplaceAll(raw_ipa, {{"\u0361", ""}, {"\xCD\xA1", ""}});

  // Step 2: Apply Misaki G2P normalization rules to convert raw IPA sequences
  // into Kokoro's compact vocabulary representations:
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

std::string ResolveEspeakDataDir(absl::string_view path) {
  if (path.empty()) return "";

  // Check candidate paths: the path itself or a subfolder named
  // "espeak-ng-data".
  std::filesystem::path base_path = std::string(path);
  std::vector<std::filesystem::path> candidates = {
      base_path,
      base_path / "espeak-ng-data",
  };
  for (const auto& candidate : candidates) {
    // A valid espeak-ng data directory MUST contain the "phontab" table file.
    if (std::filesystem::exists(candidate / "phontab")) {
      return candidate.string();
    }
  }
  return "";
}

std::string NormalizeLanguageCode(absl::string_view language_code) {
  if (language_code.empty()) return "en-us";
  std::string lower = absl::AsciiStrToLower(language_code);

  static const absl::NoDestructor<absl::flat_hash_map<std::string, std::string>>
      kLanguageMap({
          // American English ('a')
          {"a", "en-us"},
          {"en", "en-us"},
          {"en-us", "en-us"},
          {"en_us", "en-us"},
          {"american", "en-us"},
          {"english", "en-us"},
          {"american english", "en-us"},
          // British English ('b')
          {"b", "en-gb"},
          {"en-gb", "en-gb"},
          {"en_gb", "en-gb"},
          {"british", "en-gb"},
          {"british english", "en-gb"},
          // Spanish ('e')
          {"e", "es"},
          {"es", "es"},
          {"spanish", "es"},
          // French ('f')
          {"f", "fr-fr"},
          {"fr", "fr-fr"},
          {"fr-fr", "fr-fr"},
          {"fr_fr", "fr-fr"},
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
          {"pt_br", "pt-br"},
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
          {"chinese", "cmn"},
          {"mandarin", "cmn"},
      });

  auto it = kLanguageMap->find(lower);
  if (it != kLanguageMap->end()) {
    return it->second;
  }
  return lower;
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
  if (espeak_SetVoiceByName(language_.c_str()) != EE_OK) {
    return absl::InternalError(
        absl::StrCat("Failed to set espeak voice for language: ", language_));
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

  // Tokenize input sentence by word characters (alphanumeric, non-ASCII UTF-8
  // bytes, apostrophe, hyphen) and punctuation, converting each word to IPA
  // while preserving punctuation marks.
  for (size_t i = 0; i < text.size(); ++i) {
    char c = text[i];
    unsigned char uc = static_cast<unsigned char>(c);
    if (std::isalnum(uc) || uc >= 0x80 || c == '\'' || c == '-') {
      current_word += c;
    } else {
      LITERT_RETURN_IF_ERROR(FlushWordToIpa(current_word, combined_ipa));
      if (c == ' ') {
        if (!combined_ipa.empty() && combined_ipa.back() != ' ') {
          combined_ipa += ' ';
        }
      } else {
        combined_ipa += c;
      }
    }
  }
  LITERT_RETURN_IF_ERROR(FlushWordToIpa(current_word, combined_ipa));

  // Normalize IPA output to match Kokoro's vocabulary symbols.
  return NormalizeMisakiPhonemes(combined_ipa);
}

absl::StatusOr<std::vector<int>> KokoroPhonemizer::TextToPhonemeIds(
    absl::string_view text) const {
  // Initialize output token buffer with BOS token (ID 0) at index 0.
  std::vector<int> ids(kokoro::kMaxTokens, kokoro::kBosTokenId);
  const auto& vocab = GetKokoroVocabMap();

  // Convert raw input text to normalized IPA phoneme transcript.
  LITERT_ASSIGN_OR_RETURN(std::string ipa_str, TextToIpa(text));
  if (ipa_str.empty()) {
    ipa_str = std::string(text);
  }

  int idx = 1;
  size_t pos = 0;

  // Iterate over the UTF-8 encoded IPA string character by character (Unicode
  // codepoints). Multi-byte UTF-8 sequences (such as IPA phonetic characters
  // 'ɑ', 'ə', 'ɪ', and punctuation '—', '…') are parsed using standard UTF-8
  // leading-byte bitmasks:
  while (pos < ipa_str.size() && idx < kokoro::kMaxTokens - 1) {
    unsigned char c = ipa_str[pos];
    size_t char_len = 1;

    // Bitwise check on UTF-8 leading byte header:
    // - 0xxxxxxx: 1-byte ASCII character (0x00..0x7F).
    // - 110xxxxx (c & 0xE0 == 0xC0): 2-byte sequence (e.g. IPA letters: ɑ, ə,
    // ɪ, ð, ʃ).
    // - 1110xxxx (c & 0xF0 == 0xE0): 3-byte sequence (e.g. punctuation: —, …,
    // modifiers: ᵊ).
    // - 11110xxx (c & 0xF8 == 0xF0): 4-byte sequence (e.g. rare Unicode plane 1
    // symbols: ꭧ).
    if ((c & 0x80) == 0) {
      char_len = 1;
    } else if ((c & 0xE0) == 0xC0) {
      char_len = 2;
    } else if ((c & 0xF0) == 0xE0) {
      char_len = 3;
    } else if ((c & 0xF8) == 0xF0) {
      char_len = 4;
    }

    // Boundary check to prevent reading past the end of the string.
    if (pos + char_len > ipa_str.size()) char_len = ipa_str.size() - pos;
    absl::string_view symbol = absl::string_view(ipa_str).substr(pos, char_len);
    pos += char_len;

    // Map the Unicode symbol to its corresponding integer token ID in Kokoro's
    // vocabulary.
    auto it = vocab.find(symbol);
    if (it != vocab.end()) {
      ids[idx++] = it->second;
    } else if (symbol == " ") {
      ids[idx++] = kokoro::kSpaceTokenId;
    }
  }

  // Resize token array to include the trailing EOS token (ID 0).
  ids.resize(idx + 1);
  return ids;
}

}  // namespace litert::omni::tts
