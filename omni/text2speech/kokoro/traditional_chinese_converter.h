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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2SPEECH_KOKORO_TRADITIONAL_CHINESE_CONVERTER_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2SPEECH_KOKORO_TRADITIONAL_CHINESE_CONVERTER_H_

#include <cstddef>
#include <memory>
#include <string>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl

namespace litert::omni::tts::kokoro {

// Converts Traditional Chinese text to Simplified Chinese with a dictionary of
// phrase and character mappings (OpenCC's TSPhrases + TSCharacters, shipped as
// the "zh-t2s" section of a Kokoro .litertlm container).
//
// The Mandarin front end (jieba segmentation, pypinyin phrase readings, and the
// zh text normalization table) is built from Simplified Chinese data. Running
// Traditional input through it falls back to per-character readings and loses
// polyphone context (銀行 read as yín-xíng instead of yín-háng). Converting to
// Simplified first lets the existing data apply unchanged.
//
// Conversion is greedy longest match: at each position the longest dictionary
// key that matches is replaced, and text that matches no key is copied
// through. Phrase entries that map to themselves (乾坤 -> 乾坤) keep their
// characters from being converted individually (乾 -> 干).
//
// Dictionary format: UTF-8 text, one `traditional<TAB>simplified` entry per
// line. Blank lines and lines starting with '#' are ignored. If an entry lists
// several space-separated candidates, the first one is used.
class TraditionalChineseConverter {
 public:
  // Not copyable: `entries_` holds views into `dictionary_`, so a copy would
  // point into the source object's dictionary.
  TraditionalChineseConverter(const TraditionalChineseConverter&) = delete;
  TraditionalChineseConverter& operator=(const TraditionalChineseConverter&) =
      delete;

  // Parses a conversion dictionary.
  //
  // args
  // - dictionary: Dictionary text in the format described above. The converter
  //   keeps its own copy.
  //
  // returns
  // - A converter, or an InvalidArgumentError naming the offending line if an
  //   entry has no tab-separated value or the dictionary has no entries.
  static absl::StatusOr<std::unique_ptr<TraditionalChineseConverter>> Create(
      absl::string_view dictionary);

  // Converts `text` to Simplified Chinese.
  //
  // args
  // - text: UTF-8 input text. Non-Chinese text passes through unchanged.
  //
  // returns
  // - The converted text.
  std::string Convert(absl::string_view text) const;

  // Returns the number of dictionary entries. Exposed for diagnostics and
  // testing.
  //
  // returns
  // - Number of distinct keys parsed from the dictionary.
  size_t EntryCount() const;

 private:
  TraditionalChineseConverter() = default;

  // Owned copy of the dictionary text; `entries_` holds views into it.
  std::string dictionary_;
  absl::flat_hash_map<absl::string_view, absl::string_view> entries_;
  // Length in UTF-8 characters of the longest key.
  size_t max_key_chars_ = 0;
};

}  // namespace litert::omni::tts::kokoro

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2SPEECH_KOKORO_TRADITIONAL_CHINESE_CONVERTER_H_
