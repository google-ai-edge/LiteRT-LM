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
// C++ port of `src/utils/onig.rs` (and the `SysRegex` re-export of
// `src/utils/mod.rs`) from the HuggingFace `tokenizers` crate v0.23.2
// (https://github.com/huggingface/tokenizers, Apache License 2.0,
// Copyright 2020 The HuggingFace Team). Kept line-by-line close to the Rust
// original on purpose; see
// support/tokenizer/tokenizers/README.md.
//
// The crate is built with its `onig` feature, i.e. every regular expression
// (`SysRegex`, and the `regex` crate uses of the inference path) is evaluated
// by Oniguruma. This header wraps Oniguruma the way the `onig` crate does.

#ifndef THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_UTILS_REGEX_H_
#define THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_UTILS_REGEX_H_

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl

// Oniguruma's `regex_t` (`OnigRegexType`), kept opaque so that users of this
// header do not see Oniguruma's headers (and their `UChar` macro).
struct re_pattern_buffer;

namespace tokenizers {

// `pub struct SysRegex { regex: onig::Regex }`
//
// A regular expression compiled by Oniguruma with `ONIG_ENCODING_UTF8`,
// `ONIG_OPTION_NONE` and `ONIG_SYNTAX_DEFAULT` (= `ONIG_SYNTAX_ONIGURUMA`, the
// Ruby-like syntax), exactly like `onig::Regex::new`. Consequences worth
// knowing when porting patterns written for the Rust `regex` crate:
//   * `\p{L}`, `\p{N}`, `(?i:...)`, look-around and `\s`/`\w`/`\d` with
//     Unicode semantics all work.
//   * `^` and `$` are *line* anchors in this syntax (Ruby semantics); use
//     `\A` and `\z` for the `regex` crate's default text anchors.
//   * `\h` is a hex digit, `\R` a generic newline, `\X` a grapheme cluster.
//
// A compiled `SysRegex` is immutable: `FindIter` may be called concurrently
// from several threads on the same object.
class SysRegex {
 public:
  // `SysRegex::new(regex_str)`: compiles `pattern`. Oniguruma compilation
  // errors become `absl::InvalidArgumentError("Oniguruma error: <text>")`,
  // mirroring the `Display` of `onig::Error`.
  static absl::StatusOr<SysRegex> New(absl::string_view pattern);

  ~SysRegex();
  SysRegex(SysRegex&& other) noexcept;
  SysRegex& operator=(SysRegex&& other) noexcept;
  SysRegex(const SysRegex&) = delete;
  SysRegex& operator=(const SysRegex&) = delete;

  // `self.regex.find_iter(inside)`, collected: the byte offsets `(start, end)`
  // of the successive non-overlapping matches of the regex in `text`, with the
  // iteration semantics of `onig::FindMatches`: each search starts where the
  // previous match ended; an empty match found exactly where the previous match
  // ended is skipped by advancing the search start by one character.
  // PORT NOTE: `onig` panics when Oniguruma reports a search error (e.g. the
  // retry limit of a catastrophic pattern); here the iteration stops instead.
  std::vector<std::pair<size_t, size_t>> FindIter(absl::string_view text) const;

  // The pattern this regex was compiled from.
  absl::string_view pattern() const { return pattern_; }

 private:
  SysRegex(re_pattern_buffer* regex, std::string pattern)
      : regex_(regex), pattern_(std::move(pattern)) {}

  re_pattern_buffer* regex_ = nullptr;
  std::string pattern_;
};

// `regex::escape(text)` (regex-syntax 0.8): escapes all regular expression
// meta characters in `text` so that it can be used as a literal in a pattern.
// The set of meta characters is exactly the one of `regex_syntax::
// is_meta_character`: `\ . + * ? ( ) | [ ] { } ^ $ # & - ~`. Every one of them
// escaped is a literal in Oniguruma's syntax too, so `SysRegex::New(
// RegexEscape(s))` matches `s` literally.
std::string RegexEscape(absl::string_view text);

}  // namespace tokenizers

#endif  // THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_UTILS_REGEX_H_
