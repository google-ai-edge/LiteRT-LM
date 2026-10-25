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
//
// Replacement for the `daachorse` crate (`DoubleArrayAhoCorasick<u32>` built
// with `MatchKind::LeftmostLongest`, `leftmost_find_iter`) used by
// `src/tokenizer/added_vocabulary.rs` of the HuggingFace `tokenizers` crate
// v0.23.2; see
// support/tokenizer/tokenizers/README.md.
//
// PORT NOTE: this is not a double-array automaton. It is a plain byte trie
// that, for every candidate start position, walks the trie to find the
// longest pattern starting there, which yields exactly the leftmost-longest,
// non-overlapping match sequence of `leftmost_find_iter`: the match with the
// smallest start wins, ties are broken by length, and the search resumes at
// the end of the reported match. Complexity is O(text * longest pattern),
// which is fine for added-token vocabularies.

#ifndef THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_TOKENIZER_AHO_CORASICK_H_
#define THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_TOKENIZER_AHO_CORASICK_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl

namespace tokenizers {

// `daachorse::Match`
struct AhoCorasickMatch {
  // Byte offsets `[start, end)` into the searched text.
  size_t start = 0;
  size_t end = 0;
  // The value registered with the matched pattern (`Match::value()`).
  uint32_t value = 0;

  friend bool operator==(const AhoCorasickMatch& a, const AhoCorasickMatch& b) {
    return a.start == b.start && a.end == b.end && a.value == b.value;
  }
  friend bool operator!=(const AhoCorasickMatch& a, const AhoCorasickMatch& b) {
    return !(a == b);
  }
};

// Leftmost-longest multi-pattern matcher over bytes (UTF-8 strings are
// matched as byte sequences, exactly like `daachorse`).
class AhoCorasick {
 public:
  // `DoubleArrayAhoCorasickBuilder::new().match_kind(MatchKind::LeftmostLongest)
  // .build_with_values(patvals)`. Fails like `daachorse` on an empty pattern
  // set, an empty pattern or a duplicate pattern.
  static absl::StatusOr<AhoCorasick> BuildWithValues(
      const std::vector<std::pair<std::string, uint32_t>>& patvals);

  // `leftmost_find_iter(haystack)`, fully materialized: non-overlapping
  // matches ordered by start offset; for each start the longest pattern wins.
  std::vector<AhoCorasickMatch> LeftmostFindIter(
      absl::string_view haystack) const;

  // Number of patterns.
  size_t NumPatterns() const { return num_patterns_; }

 private:
  struct Node {
    // Transitions on the next byte.
    absl::flat_hash_map<uint8_t, uint32_t> children;
    // Value of the pattern ending at this node, if any.
    bool is_terminal = false;
    uint32_t value = 0;
  };

  AhoCorasick() = default;

  std::vector<Node> nodes_;
  size_t num_patterns_ = 0;
};

}  // namespace tokenizers

#endif  // THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_TOKENIZER_AHO_CORASICK_H_
