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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_LEVENSHTEIN_ALIGN_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_LEVENSHTEIN_ALIGN_H_

#include <string>
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl

namespace litert::omni::asr {

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

// Lowercases a word and strips punctuation.
std::string CanonicalizeStr(absl::string_view word);

// Lowercases each word in the span and strips punctuation.
std::vector<std::string> CanonicalizeWords(absl::Span<const std::string> words);

// Returns true if two string views match on prefix or have edit distance <= 1.
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

}  // namespace litert::omni::asr

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_LEVENSHTEIN_ALIGN_H_
