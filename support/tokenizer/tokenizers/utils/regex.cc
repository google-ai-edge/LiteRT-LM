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
// The parts of the `onig` crate (v6.4, `Regex::new` and `FindMatches`) and of
// `regex-syntax` (`escape`) that the crate relies on are inlined here, since
// they are what defines the behaviour of `SysRegex`.

#include "support/tokenizer/tokenizers/utils/regex.h"

#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"  // from @com_google_absl
#include "absl/base/call_once.h"  // from @com_google_absl
#include "absl/base/const_init.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/utils/utf8.h"

// Keep oniguruma.h from defining `UChar` as a macro (it collides with ICU).
#define ONIG_ESCAPE_UCHAR_COLLISION
#include "oniguruma.h"  // from @oniguruma

namespace tokenizers {
namespace {

// Oniguruma must be initialized once per process before the first `onig_new`
// (it would otherwise do so lazily, racily, from `onig_new`).
void InitializeOniguruma() {
  static absl::once_flag once;
  absl::call_once(once, [] {
    OnigEncoding encodings[] = {ONIG_ENCODING_UTF8};
    onig_initialize(encodings, 1);
  });
}

// `static REGEX_NEW_MUTEX: Lazy<Mutex<()>>` of the `onig` crate: Oniguruma's
// compiler is not thread-safe, searching compiled regexes is.
ABSL_CONST_INIT absl::Mutex regex_new_mutex(absl::kConstInit);

const OnigUChar* AsOnigString(absl::string_view s) {
  // `absl::string_view::data()` may be null for an empty view; Oniguruma does
  // pointer arithmetic on it, so hand it a valid empty string instead.
  static constexpr char kEmpty[] = "";
  return reinterpret_cast<const OnigUChar*>(s.empty() ? kEmpty : s.data());
}

}  // namespace

// `onig::Regex::new(pattern)` ==
// `Regex::with_options_and_encoding(pattern, REGEX_OPTION_NONE,
//                                   Syntax::default())`
absl::StatusOr<SysRegex> SysRegex::New(absl::string_view pattern) {
  InitializeOniguruma();

  const OnigUChar* pattern_start = AsOnigString(pattern);
  const OnigUChar* pattern_end = pattern_start + pattern.size();
  // We can use this later to get an error message to pass back
  // if regex creation fails.
  OnigErrorInfo error_info;
  error_info.enc = nullptr;
  error_info.par = nullptr;
  error_info.par_end = nullptr;
  re_pattern_buffer* regex = nullptr;
  int result;
  {
    // Grab a lock to make sure that `onig_new` isn't called by
    // more than one thread at a time.
    absl::MutexLock lock(&regex_new_mutex);
    result = onig_new(&regex, pattern_start, pattern_end, ONIG_OPTION_NONE,
                      ONIG_ENCODING_UTF8, ONIG_SYNTAX_DEFAULT, &error_info);
  }
  if (result != ONIG_NORMAL) {
    // `onig::Error::from_code_and_info` + its `Display` impl.
    OnigUChar buffer[ONIG_MAX_ERROR_MESSAGE_LEN];
    const int length = onig_error_code_to_str(buffer, result, &error_info);
    return absl::InvalidArgumentError(absl::StrCat(
        "Oniguruma error: ",
        absl::string_view(reinterpret_cast<const char*>(buffer),
                          length > 0 ? static_cast<size_t>(length) : 0)));
  }
  return SysRegex(regex, std::string(pattern));
}

SysRegex::~SysRegex() {
  if (regex_ != nullptr) onig_free(regex_);
}

SysRegex::SysRegex(SysRegex&& other) noexcept
    : regex_(std::exchange(other.regex_, nullptr)),
      pattern_(std::move(other.pattern_)) {}

SysRegex& SysRegex::operator=(SysRegex&& other) noexcept {
  if (this != &other) {
    if (regex_ != nullptr) onig_free(regex_);
    regex_ = std::exchange(other.regex_, nullptr);
    pattern_ = std::move(other.pattern_);
  }
  return *this;
}

// `onig::FindMatches::next`, run to exhaustion.
std::vector<std::pair<size_t, size_t>> SysRegex::FindIter(
    absl::string_view text) const {
  std::vector<std::pair<size_t, size_t>> matches;
  const OnigUChar* str = AsOnigString(text);
  const OnigUChar* end = str + text.size();
  // A fresh region per call keeps concurrent `FindIter` calls independent.
  OnigRegion region;
  onig_region_init(&region);

  size_t last_end = 0;
  std::optional<size_t> last_match_end;
  while (last_end <= text.size()) {
    onig_region_clear(&region);
    const int result = onig_search(regex_, str, end, str + last_end, end,
                                   &region, ONIG_OPTION_NONE);
    // ONIG_MISMATCH ends the iteration; so does an error (see the PORT NOTE
    // on `FindIter`).
    if (result < 0) break;
    const size_t s = static_cast<size_t>(region.beg[0]);
    const size_t e = static_cast<size_t>(region.end[0]);

    // Don't accept empty matches immediately following the last match.
    // i.e., no infinite loops please.
    if (e == s && last_match_end.has_value() && *last_match_end == e) {
      // `self.text[self.last_end..].chars().next()
      //      .map(|c| c.len_utf8()).unwrap_or(1)`
      if (last_end < text.size()) {
        DecodeUtf8(text, &last_end);
      } else {
        last_end += 1;
      }
      continue;
    } else {
      last_end = e;
      last_match_end = e;
    }

    matches.emplace_back(s, e);
  }

  onig_region_free(&region, /*free_self=*/0);
  return matches;
}

namespace {

// `regex_syntax::is_meta_character` (regex-syntax 0.8).
bool IsMetaCharacter(char c) {
  switch (c) {
    case '\\':
    case '.':
    case '+':
    case '*':
    case '?':
    case '(':
    case ')':
    case '|':
    case '[':
    case ']':
    case '{':
    case '}':
    case '^':
    case '$':
    case '#':
    case '&':
    case '-':
    case '~':
      return true;
    default:
      return false;
  }
}

}  // namespace

// `regex_syntax::escape` / `escape_into`. Iterating over bytes instead of
// chars is equivalent: every meta character is ASCII, and the bytes of a
// multi-byte UTF-8 sequence are never ASCII.
std::string RegexEscape(absl::string_view text) {
  std::string quoted;
  quoted.reserve(text.size());
  for (char c : text) {
    if (IsMetaCharacter(c)) {
      quoted.push_back('\\');
    }
    quoted.push_back(c);
  }
  return quoted;
}

}  // namespace tokenizers
