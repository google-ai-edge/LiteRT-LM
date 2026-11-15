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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2SPEECH_KOKORO_JAPANESE_G2P_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2SPEECH_KOKORO_JAPANESE_G2P_H_

#include <optional>
#include <string>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "omni/text2speech/kokoro/cjk_blob.h"
#include "omni/text2speech/kokoro/mecab_lattice.h"

namespace litert::omni::tts::kokoro {

// Japanese grapheme-to-phoneme front end ported from `misaki.ja.JAG2P`.
//
// Combines `MecabCostModel` + `ViterbiSegment` over the packed `ja-lexicon`
// blob with `misaki/ja.py`'s normalization, `num2kana` digit reading,
// `ja_words` compound grouping, and Hepburn kana-to-IPA rules (`っ` -> `ʔ`,
// `ー` -> `ː`, context-dependent `ん` nasal assimilation, digraphs, sutegana,
// and odoriji).
class JapaneseG2p {
 public:
  // Builds the G2P engine from a `ja-lexicon` blob.
  //
  // args
  // - blob: Parsed `ja-lexicon` blob, which must outlive the result. The
  //   optional `w-idx` / `w-key` sections enable `ja_words` compound grouping.
  //
  // returns
  // - The engine, or an error if the MeCab sections are missing or malformed.
  static absl::StatusOr<JapaneseG2p> Create(const CjkBlob& blob);

  // Converts Japanese text into a Kokoro IPA phoneme string.
  //
  // args
  // - text: UTF-8 Japanese text.
  //
  // returns
  // - The IPA string (empty for empty or whitespace-only `text`), or an error
  //   if segmentation fails.
  absl::StatusOr<std::string> TextToIpa(absl::string_view text) const;

 private:
  JapaneseG2p(MecabCostModel model,
              std::optional<SortedStringIndex> words_index);

  MecabCostModel model_;
  std::optional<SortedStringIndex> words_index_;
};

}  // namespace litert::omni::tts::kokoro

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2SPEECH_KOKORO_JAPANESE_G2P_H_
