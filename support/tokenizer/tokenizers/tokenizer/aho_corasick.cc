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

#include "support/tokenizer/tokenizers/tokenizer/aho_corasick.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl

namespace tokenizers {

absl::StatusOr<AhoCorasick> AhoCorasick::BuildWithValues(
    const std::vector<std::pair<std::string, uint32_t>>& patvals) {
  if (patvals.empty()) {
    return absl::InvalidArgumentError(
        "AhoCorasick::BuildWithValues: patvals must not be empty");
  }

  AhoCorasick ac;
  ac.nodes_.emplace_back();

  for (const auto& [pattern, value] : patvals) {
    if (pattern.empty()) {
      return absl::InvalidArgumentError(
          "AhoCorasick::BuildWithValues: pattern must not be empty");
    }
    uint32_t cur = 0;
    for (char ch : pattern) {
      const uint8_t b = static_cast<uint8_t>(ch);
      auto it = ac.nodes_[cur].children.find(b);
      if (it != ac.nodes_[cur].children.end()) {
        cur = it->second;
      } else {
        const uint32_t next = static_cast<uint32_t>(ac.nodes_.size());
        ac.nodes_[cur].children.emplace(b, next);
        ac.nodes_.emplace_back();
        cur = next;
      }
    }
    if (ac.nodes_[cur].is_terminal) {
      return absl::InvalidArgumentError(absl::StrCat(
          "AhoCorasick::BuildWithValues: duplicate pattern `", pattern, "`"));
    }
    ac.nodes_[cur].is_terminal = true;
    ac.nodes_[cur].value = value;
  }

  ac.num_patterns_ = patvals.size();
  return ac;
}

std::vector<AhoCorasickMatch> AhoCorasick::LeftmostFindIter(
    absl::string_view haystack) const {
  std::vector<AhoCorasickMatch> matches;
  if (nodes_.empty() || haystack.empty()) {
    return matches;
  }

  for (size_t start = 0; start < haystack.size();) {
    uint32_t cur = 0;
    size_t pos = start;
    bool found = false;
    size_t best_end = start;
    uint32_t best_value = 0;

    while (pos < haystack.size()) {
      const uint8_t b = static_cast<uint8_t>(haystack[pos]);
      auto it = nodes_[cur].children.find(b);
      if (it == nodes_[cur].children.end()) {
        break;
      }
      cur = it->second;
      ++pos;
      if (nodes_[cur].is_terminal) {
        found = true;
        best_end = pos;
        best_value = nodes_[cur].value;
      }
    }

    if (found) {
      matches.push_back(AhoCorasickMatch{start, best_end, best_value});
      start = best_end;
    } else {
      ++start;
    }
  }

  return matches;
}

}  // namespace tokenizers
