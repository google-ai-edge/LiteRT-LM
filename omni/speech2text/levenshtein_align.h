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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_SPEECH2TEXT_LEVENSHTEIN_ALIGN_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_SPEECH2TEXT_LEVENSHTEIN_ALIGN_H_

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl

namespace litert::omni::asr {

using WordIndexPair = std::pair<size_t, size_t>;

// Represents the edit decision code for string sequence alignment.
enum class AlignCode {
  kInsertion = 0,
  kDeletion = 1,
  kSubstitution = 2,
  kCorrect = 3,
};

// Computes the Levenshtein sequence alignment between ref_tokens and
// hyp_tokens. Returns the optimal alignment decision sequence.
std::vector<AlignCode> AlignTokens(absl::Span<const std::string> ref_tokens,
                                   absl::Span<const std::string> hyp_tokens);

// Computes the Levenshtein edit distance between two string views.
int ComputeLevenshteinDistanceStr(absl::string_view s1, absl::string_view s2);

// Computes the word-level Levenshtein edit distance between two word sequences.
int ComputeLevenshteinDistanceWords(absl::Span<const std::string> words1,
                                    absl::Span<const std::string> words2);

// Lowercases a word and strips punctuation (ASCII and common Unicode
// punctuation such as CJK and fullwidth marks).
std::string CanonicalizeStr(absl::string_view word);

// Lowercases each word in the span and strips punctuation.
std::vector<std::string> CanonicalizeWords(absl::Span<const std::string> words);

// Returns true if two already-canonicalized string views match exactly (when
// the shorter has length <= 3) or have Levenshtein edit distance <= 1.
bool CloseEnoughCanonical(absl::string_view c1, absl::string_view c2);

// Returns true if two string views are close enough after canonicalization.
bool CloseEnoughStr(absl::string_view s1, absl::string_view s2);

// Returns true if two word spans have equal size and corresponding words are
// close enough.
bool CloseEnoughWords(absl::Span<const std::string> l1,
                      absl::Span<const std::string> l2);

// Finds max overlapping suffix of prev_words and prefix of curr_words within
// search_window using CloseEnoughWords, and returns curr_words with the
// overlapping prefix removed.
std::vector<std::string> DedupWords(absl::Span<const std::string> prev_words,
                                    absl::Span<const std::string> curr_words,
                                    int search_window = 2);

// Finds the best "seam run" in the overlap region between
// `prev_words[min_prev..]` and `curr_words[0..max_curr)`, returning a vector
// of `(prev_idx, curr_idx)` word index pairs in `prev_words` and `curr_words`,
// or an empty vector if no valid seam match exists.
//
// A "seam run" is a contiguous sequence of matching words (`CloseEnoughStr`)
// at the boundary ("seam") where two consecutive overlapping ASR chunks are
// stitched together. Stitching at a multi-word contiguous run avoids dropping
// words when an isolated function word (e.g., "the") happens to match earlier
// in `prev_words`:
//
//   prev_words: ["read", "the", "book", "on", "the", "table", "today"]
//                          |                    |       |        |
//   curr_words:          ["the",              "the", "table", "today", "now"]
//                          ^                    ^^^^^^^^^^^^^^^^^^^^^
//              isolated 1-word match            seam run (length 3)
//
// The best seam run is selected in three steps:
//   1. Global Levenshtein alignment: Runs `AlignTokens` on
//      `prev_words[min_prev..]` and `curr_words[0..max_curr)` and extracts the
//      longest contiguous run of `AlignCode::kCorrect` matches (preferring
//      later runs in `prev_words` on ties of length >= 2).
//   2. Fallback boundary scan (when step 1 finds no run of length >= 2):
//      Because substitution cost equals insertion + deletion cost (2 = 1 + 1)
//      in `AlignTokens`, non-overlapping prefixes/suffixes of similar length
//      can pair as substitutions and break up a genuine seam overlap. We scan
//      directly for the longest contiguous `CloseEnoughStr` run of length >= 2
//      near the boundary (`curr_idx <= 3` or within the last 2 words of
//      `prev_words`), and if still < 2, look for a pair of aligned matches
//      separated by a small gap near the boundary where at least one word is a
//      content word (>= 6 chars).
//   3. Single-word boundary validation: A 1-word match is accepted only if it
//      sits at the chunk boundary (the last word of `prev_words` and first
//      word of `curr_words`, or a >= 5-char word within the last 3 words of
//      `prev_words` and first 2 words of `curr_words`).
std::vector<WordIndexPair> FindBestSeamMatchRun(
    absl::Span<const std::string> prev_words, size_t min_prev,
    absl::Span<const std::string> curr_words, size_t max_curr);

}  // namespace litert::omni::asr

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_SPEECH2TEXT_LEVENSHTEIN_ALIGN_H_
