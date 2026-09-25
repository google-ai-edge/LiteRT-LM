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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_KOKORO_CHINESE_G2P_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_KOKORO_CHINESE_G2P_H_

#include <memory>
#include <string>
#include <vector>

#include "cppjieba/DictTrie.hpp"
#include "cppjieba/HMMModel.hpp"
#include "cppjieba/MixSegment.hpp"
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "omni/tts/kokoro/cjk_blob.h"

namespace litert::omni::tts::kokoro {

// Mandarin Chinese grapheme-to-phoneme front end ported from `misaki.zh.ZHG2P`.
//
// Segments input text with `cppjieba::MixSegment` (maximum-probability DAG over
// `DictTrie` plus Viterbi `HMMModel` for out-of-vocabulary spans), then
// converts each segmented word into Kokoro IPA by running `pypinyin`'s forward
// maximum-matching (`mmseg`) over the packed phrase dictionary
// (`p-idx`/`p-key`/`p-vidx`/`p-val`) and falling back per character to the
// packed single-character dictionary (`c-idx`/`c-key`/`c-vidx`/`c-val`).
//
// Usage:
//   ASSERT_OK_AND_ASSIGN(CjkBlob blob, CjkBlob::Create(zh_lexicon_bytes));
//   ASSERT_OK_AND_ASSIGN(ChineseG2p g2p, ChineseG2p::Create(blob, cache_dir));
//   ASSERT_OK_AND_ASSIGN(std::string ipa, g2p.TextToIpa("今天天气"));
//
// Lifetime:
//   Because `SortedStringIndex` and `StringValueTable` read zero-copy slices
//   from `blob`, both `blob` and its underlying memory buffer must outlive the
//   `ChineseG2p` instance.
class ChineseG2p {
 public:
  ChineseG2p(ChineseG2p&&) noexcept = default;
  ChineseG2p& operator=(ChineseG2p&&) noexcept = default;
  ~ChineseG2p() = default;

  // Builds a `ChineseG2p` instance from a parsed `zh-lexicon` `CjkBlob`,
  // unpacking the HMM model file required by `cppjieba::HMMModel` into
  // `cache_dir` (or the system temporary directory when `cache_dir` is empty or
  // read-only).
  //
  // args
  // - blob: Parsed `CjkBlob` wrapping the `"zh-lexicon"` container sections
  //   (`w-*`, `hmm-*`, `p-*`, `c-*`). Must outlive the returned `ChineseG2p`.
  // - cache_dir: Optional directory path where the unpacked
  //   `kokoro_zh_hmm_model.utf8` file is cached for `cppjieba::HMMModel`.
  //
  // returns
  // - Initialized `ChineseG2p` instance on success, or an error status
  //   (`InvalidArgumentError`, `NotFoundError`, or `InternalError`) if any
  //   required lexicon section is missing, malformed, or cannot be unpacked.
  static absl::StatusOr<ChineseG2p> Create(const CjkBlob& blob,
                                           absl::string_view cache_dir = "");

  // Segments `text` into `jieba.lcut(..., cut_all=False)` words using
  // `cppjieba::MixSegment` over Hanzi/alphanumeric spans while keeping
  // non-Hanzi runs intact.
  //
  // args
  // - text: UTF-8 input text to segment. Must outlive the returned
  //   `absl::string_view` slices, which point directly into `text`.
  //
  // returns
  // - Vector of `absl::string_view` word slices in `text` order, or an error
  //   status if segmentation fails.
  absl::StatusOr<std::vector<absl::string_view>> Segment(
      absl::string_view text) const;

  // Converts a single Chinese word token to Kokoro IPA and appends it to
  // `out`.
  //
  // Applies `pypinyin` intra-word forward maximum-matching (`mmseg`) over the
  // phrase table first so multi-character sandhi phrases inside compound words
  // (e.g. `"一点"` inside `"一万多一点"`) resolve to their contextual phrase
  // pronunciations before falling back to single-character lookups.
  //
  // args
  // - word: Single UTF-8 Chinese word token to transcribe.
  // - out: Output string to which the Kokoro IPA transcription is appended.
  //   Left unmodified if this method returns `false`.
  //
  // returns
  // - `true` when every phrase and character in `word` was resolved and
  //   appended to `out`; `false` when `word` contains a character absent from
  //   both the phrase and character dictionaries (e.g. U+74E7 `瓧`).
  bool WordToIpa(absl::string_view word, std::string& out) const;

  // Segments `text` via `Segment()` and joins the per-word Kokoro IPA
  // transcriptions with single spaces.
  //
  // args
  // - text: UTF-8 Mandarin Chinese text to segment and transcribe.
  //
  // returns
  // - Space-separated Kokoro IPA transcription string on success, or
  //   `absl::NotFoundError` if any word contains a character missing from the
  //   lexicon.
  absl::StatusOr<std::string> TextToIpa(absl::string_view text) const;

 private:
  ChineseG2p(std::unique_ptr<cppjieba::DictTrie> dict_trie,
             std::unique_ptr<cppjieba::HMMModel> hmm_model,
             std::unique_ptr<cppjieba::MixSegment> segmenter,
             SortedStringIndex phrases, StringValueTable phrase_values,
             SortedStringIndex characters, StringValueTable character_values);

  // Applies `pypinyin.seg.mmseg.Seg.cut` (`no_non_phrases=True`) over
  // `phrases_`, appending slices pointing into `word` to `pieces`.
  void MmsegCut(absl::string_view word,
                std::vector<absl::string_view>& pieces) const;

  std::unique_ptr<cppjieba::DictTrie> dict_trie_;
  std::unique_ptr<cppjieba::HMMModel> hmm_model_;
  std::unique_ptr<cppjieba::MixSegment> segmenter_;
  SortedStringIndex phrases_;
  StringValueTable phrase_values_;
  SortedStringIndex characters_;
  StringValueTable character_values_;
};

}  // namespace litert::omni::tts::kokoro

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_KOKORO_CHINESE_G2P_H_
