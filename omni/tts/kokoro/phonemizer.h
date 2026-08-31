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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_KOKORO_PHONEMIZER_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_KOKORO_PHONEMIZER_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl

namespace litert::omni::tts {

// Returns the global mapping of IPA phoneme graphemes to vocabulary token IDs.
//
// returns
// - Const reference to flat_hash_map of phoneme strings to integer token IDs.
const absl::flat_hash_map<std::string_view, int>& GetKokoroVocabMap();

// Normalizes raw IPA phonemes to Kokoro-compatible vocabulary symbols.
//
// args
// - raw_ipa: Raw IPA phoneme string view.
//
// returns
// - Normalized IPA string matching Kokoro phoneme vocabulary.
std::string NormalizeMisakiPhonemes(absl::string_view raw_ipa);

// Resolves the espeak-ng data directory path from a given directory.
// Checks if `path` or `path/espeak-ng-data` contains valid espeak-ng data
// (phontab).
//
// args
// - path: Path to espeak-ng-data directory or parent model directory.
//
// returns
// - Canonical path to valid espeak-ng-data directory, or empty string if not
// found.
std::string ResolveEspeakDataDir(absl::string_view path);

// Normalizes a language name or Kokoro 1-letter voice code to an espeak-ng
// voice name. Examples:
// - "a", "en", "en-us" -> "en-us"
// - "b", "en-gb" -> "en-gb"
// - "e", "es", "spanish" -> "es"
// - "f", "fr", "fr-fr" -> "fr-fr"
// - "h", "hi", "hindi" -> "hi"
// - "i", "it", "italian" -> "it"
// - "p", "pt", "pt-br" -> "pt-br"
// - "j", "ja", "japanese" -> "ja"
// - "z", "zh", "chinese" -> "zh"
//
// args
// - language_code: Language identifier or single-letter code.
//
// returns
// - Canonical espeak-ng voice name.
std::string NormalizeLanguageCode(absl::string_view language_code);

// Encapsulates the espeak-ng G2P (Grapheme-to-Phoneme) engine and Misaki
// phoneme normalizer for phonemization and token ID encoding across languages.
//
// Thread Safety Architecture:
// The underlying third-party `libespeak-ng` library is written in C and relies
// on process-wide global static state (e.g. active voice structures, phonetic
// lookup tables, and language dictionary files in `speech.c` and
// `translate.c`). To guarantee complete thread safety and prevent cross-talk
// when multiple phonemizer instances or concurrent TTS sessions synthesize
// audio across different languages (e.g., English and Spanish), all direct
// interactions with the libespeak-ng C library are serialized via the
// class-level static `espeak_mutex_`.
class KokoroPhonemizer {
 public:
  // Creates and initializes a KokoroPhonemizer instance using the specified
  // data directory, target language, and custom lexicon.
  //
  // args
  // - espeak_data_dir: Path to espeak-ng-data directory or model directory.
  // - language: Language code or voice prefix (e.g. "en-us", "en-gb", "es",
  //   "fr-fr", "hi", "it", "pt-br", "ja", "zh", "a", "b", "e", etc.). Defaults
  //   to "en-us".
  // - custom_lexicon: Custom dictionary mapping words to IPA pronunciations.
  //
  // returns
  // - Unique pointer to KokoroPhonemizer on success, or error status on
  // failure.
  static absl::StatusOr<std::unique_ptr<KokoroPhonemizer>> Create(
      absl::string_view espeak_data_dir, absl::string_view language = "en-us",
      const absl::flat_hash_map<std::string, std::string>& custom_lexicon = {});

  // Overload for creating KokoroPhonemizer with default "en-us" language and
  // custom lexicon.
  static absl::StatusOr<std::unique_ptr<KokoroPhonemizer>> Create(
      absl::string_view espeak_data_dir,
      const absl::flat_hash_map<std::string, std::string>& custom_lexicon) {
    return Create(espeak_data_dir, "en-us", custom_lexicon);
  }

  ~KokoroPhonemizer() = default;

  // Updates the active language/voice for phonemization.
  // Normalizes the language code (e.g., "en-gb", "b", "es", "spanish").
  void SetLanguage(absl::string_view language);

  // Converts a word in the configured language to IPA pronunciation via custom
  // lexicon or espeak-ng.
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

  // Returns the active espeak-ng language / voice name.
  absl::string_view language() const { return language_; }

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

  // Process-wide mutex protecting the non-reentrant libespeak-ng C library.
  // Because libespeak-ng maintains process-global static state, all library
  // initialization, voice selection, and phonemization calls must be
  // serialized.
  static absl::Mutex espeak_mutex_;

  std::string data_dir_;
  std::string language_ = "en-us";
  absl::flat_hash_map<std::string, std::string> merged_lexicon_;
};

}  // namespace litert::omni::tts

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_KOKORO_PHONEMIZER_H_
