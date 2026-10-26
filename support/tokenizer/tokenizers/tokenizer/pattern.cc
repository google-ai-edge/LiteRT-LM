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
// C++ port of `src/tokenizer/pattern.rs` from the HuggingFace `tokenizers`
// crate v0.23.2 (https://github.com/huggingface/tokenizers, Apache License 2.0,
// Copyright 2020 The HuggingFace Team). Kept line-by-line close to the Rust
// original on purpose; see
// support/tokenizer/tokenizers/README.md.

#include "support/tokenizer/tokenizers/tokenizer/pattern.h"

#include <cstddef>
#include <functional>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/tokenizer/token.h"
#include "support/tokenizer/tokenizers/utils/regex.h"
#include "support/tokenizer/tokenizers/utils/utf8.h"
#include "support/util/status_macros.h"

namespace tokenizers {

namespace {

using Matches = std::vector<std::pair<Offsets, bool>>;

// `impl Pattern for &SysRegex` (identical to `impl Pattern for &Regex`)
absl::StatusOr<Matches> FindMatchesRegex(const SysRegex& regex,
                                         absl::string_view inside) {
  if (inside.empty()) {
    return Matches{{{0, 0}, false}};
  }

  size_t prev = 0;
  Matches splits;
  splits.reserve(inside.size());
  for (const auto& [start, end] : regex.FindIter(inside)) {
    if (prev != start) {
      splits.push_back({{prev, start}, false});
    }
    splits.push_back({{start, end}, true});
    prev = end;
  }
  if (prev != inside.size()) {
    splits.push_back({{prev, inside.size()}, false});
  }
  return splits;
}

// `impl Pattern for &str`
absl::StatusOr<Matches> FindMatchesStr(absl::string_view pattern,
                                       absl::string_view inside) {
  if (pattern.empty()) {
    // If we try to find the matches with an empty string, just don't match
    // anything
    // PORT NOTE: the Rust code really uses the *char* count here (not the
    // byte length) for the end offset; kept as is.
    return Matches{{{0, CountChars(inside)}, false}};
  }

  ASSIGN_OR_RETURN(SysRegex re, SysRegex::New(RegexEscape(pattern)));
  return FindMatchesRegex(re, inside);
}

// `impl<F> Pattern for F where F: Fn(char) -> bool`
absl::StatusOr<Matches> FindMatchesFn(const std::function<bool(char32_t)>& f,
                                      absl::string_view inside) {
  if (inside.empty()) {
    return Matches{{{0, 0}, false}};
  }

  size_t last_offset = 0;
  size_t last_seen = 0;

  Matches matches;
  for (const auto [b, c] : CharIndices(inside)) {
    const size_t len_utf8 = Utf8Len(c);
    last_seen = b + len_utf8;
    if (f(c)) {
      if (last_offset < b) {
        // We need to emit what was before this match
        matches.push_back({{last_offset, b}, false});
      }
      matches.push_back({{b, b + len_utf8}, true});
      last_offset = b + len_utf8;
    }
  }

  // Do not forget the last potential split
  if (last_seen > last_offset) {
    matches.push_back({{last_offset, last_seen}, false});
  }

  return matches;
}

}  // namespace

Pattern::Pattern(char32_t c) : pattern_(c) {}

Pattern::Pattern(absl::string_view s) : pattern_(std::string(s)) {}

Pattern::Pattern(const SysRegex* regex) : pattern_(regex) {}

Pattern::Pattern(std::function<bool(char32_t)> f) : pattern_(std::move(f)) {}

absl::StatusOr<std::vector<std::pair<Offsets, bool>>> Pattern::FindMatches(
    absl::string_view inside) const {
  absl::StatusOr<Matches> matches = std::visit(
      [inside](const auto& pattern) -> absl::StatusOr<Matches> {
        using T = std::decay_t<decltype(pattern)>;
        if constexpr (std::is_same_v<T, char32_t>) {
          // `impl Pattern for char`
          const char32_t self = pattern;
          return FindMatchesFn([self](char32_t c) { return c == self; },
                               inside);
        } else if constexpr (std::is_same_v<T, std::string>) {
          return FindMatchesStr(pattern, inside);
        } else if constexpr (std::is_same_v<T, const SysRegex*>) {
          return FindMatchesRegex(*pattern, inside);
        } else {
          return FindMatchesFn(pattern, inside);
        }
      },
      pattern_);
  if (!matches.ok()) return matches.status();
  if (invert_) {
    // `impl<P: Pattern> Pattern for Invert<P>`
    for (auto& [offsets, flag] : *matches) {
      flag = !flag;
    }
  }
  return *std::move(matches);
}

Pattern Invert(Pattern pattern) {
  pattern.invert_ = !pattern.invert_;
  return pattern;
}

}  // namespace tokenizers
