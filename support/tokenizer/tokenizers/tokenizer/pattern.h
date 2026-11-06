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

#ifndef THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_TOKENIZER_PATTERN_H_
#define THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_TOKENIZER_PATTERN_H_

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

namespace tokenizers {

// Pattern used to split a NormalizedString
//
// PORT NOTE: the Rust `trait Pattern` is implemented for `char`, `&str`,
// `&String`, `&Regex` (the `regex` crate), `&SysRegex` and `Fn(char) -> bool`,
// plus the `Invert<P>` wrapper. This is a small type-erased value class with
// one constructor per `impl` (`&Regex` and `&SysRegex` both map to
// `const SysRegex*`); `Invert(pattern)` toggles the `is_match` flags.
class Pattern {
 public:
  // `impl Pattern for char`
  Pattern(char32_t c);  // NOLINT(google-explicit-constructor)
  // `impl Pattern for &str` / `impl Pattern for &String`
  Pattern(absl::string_view s);  // NOLINT(google-explicit-constructor)
  Pattern(const char* s)         // NOLINT(google-explicit-constructor)
      : Pattern(absl::string_view(s)) {}
  Pattern(const std::string& s)  // NOLINT(google-explicit-constructor)
      : Pattern(absl::string_view(s)) {}
  // `impl Pattern for &SysRegex` (and `&Regex`). The regex must outlive the
  // pattern.
  Pattern(const SysRegex* regex);  // NOLINT(google-explicit-constructor)
  // `impl<F> Pattern for F where F: Fn(char) -> bool`
  Pattern(
      std::function<bool(char32_t)> f);  // NOLINT(google-explicit-constructor)
  template <
      typename F,
      typename = std::enable_if_t<
          std::is_invocable_r_v<bool, F, char32_t> &&
          !std::is_same_v<std::decay_t<F>, Pattern> &&
          !std::is_same_v<std::decay_t<F>, std::function<bool(char32_t)>>>>
  Pattern(F&& f)  // NOLINT(google-explicit-constructor)
      : Pattern(std::function<bool(char32_t)>(std::forward<F>(f))) {}

  // Slice the given string in a list of pattern match positions, with
  // a boolean indicating whether this is a match or not.
  //
  // This method *must* cover the whole string in its outputs, with
  // contiguous ordered slices.
  absl::StatusOr<std::vector<std::pair<Offsets, bool>>> FindMatches(
      absl::string_view inside) const;

  friend Pattern Invert(Pattern pattern);

 private:
  std::variant<char32_t, std::string, const SysRegex*,
               std::function<bool(char32_t)>>
      pattern_;
  bool invert_ = false;
};

// Invert the `is_match` flags for the wrapped Pattern. This is useful
// for example when we use a regex that matches words instead of a delimiter,
// and we want to match the delimiter.
//
// `pub struct Invert<P: Pattern>(pub P);`
Pattern Invert(Pattern pattern);

}  // namespace tokenizers

#endif  // THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_TOKENIZER_PATTERN_H_
