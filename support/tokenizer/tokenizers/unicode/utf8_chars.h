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
// Minimal UTF-8 decoding used by the ports of the Unicode crates (the
// replacement for `str::chars()` / `Chars::next_back()` / `char::len_utf8()`
// of the Rust standard library). Internal to the `unicode` package.
//
// PORT NOTE: Rust strings are valid UTF-8 by construction; the C++ port is
// given `absl::string_view`s that are assumed to be valid UTF-8 as well, but
// never reads past the end of the view: a malformed or truncated sequence is
// decoded as a single byte yielding U+FFFD.

#ifndef THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_UNICODE_UTF8_CHARS_H_
#define THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_UNICODE_UTF8_CHARS_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

#include "absl/strings/string_view.h"  // from @com_google_absl

namespace tokenizers::unicode::internal {

inline constexpr char32_t kReplacementCharacter = 0xFFFD;

// `char::len_utf8()`: the number of bytes of the UTF-8 encoding of `c`.
inline size_t Utf8Len(char32_t c) {
  if (c < 0x80) return 1;
  if (c < 0x800) return 2;
  if (c < 0x10000) return 3;
  return 4;
}

// Decodes the character whose UTF-8 encoding starts at `s[*pos]` and advances
// `*pos` past it. Precondition: `*pos < s.size()`.
inline char32_t DecodeUtf8(absl::string_view s, size_t* pos) {
  const size_t i = *pos;
  const uint8_t b0 = static_cast<uint8_t>(s[i]);
  size_t len;
  char32_t cp;
  if (b0 < 0x80) {
    *pos = i + 1;
    return b0;
  } else if ((b0 & 0xE0) == 0xC0) {
    len = 2;
    cp = b0 & 0x1F;
  } else if ((b0 & 0xF0) == 0xE0) {
    len = 3;
    cp = b0 & 0x0F;
  } else if ((b0 & 0xF8) == 0xF0) {
    len = 4;
    cp = b0 & 0x07;
  } else {
    *pos = i + 1;
    return kReplacementCharacter;
  }
  if (i + len > s.size()) {
    *pos = i + 1;
    return kReplacementCharacter;
  }
  for (size_t k = 1; k < len; ++k) {
    const uint8_t b = static_cast<uint8_t>(s[i + k]);
    if ((b & 0xC0) != 0x80) {
      *pos = i + 1;
      return kReplacementCharacter;
    }
    cp = (cp << 6) | (b & 0x3F);
  }
  *pos = i + len;
  return cp;
}

// Decodes the character whose UTF-8 encoding ends right before `s[*end]`
// (`Chars::next_back()`) and moves `*end` to its first byte.
// Precondition: `0 < *end <= s.size()`.
inline char32_t DecodeUtf8Backward(absl::string_view s, size_t* end) {
  size_t start = *end - 1;
  // Skip at most three continuation bytes.
  for (size_t k = 0;
       k < 3 && start > 0 && (static_cast<uint8_t>(s[start]) & 0xC0) == 0x80;
       ++k) {
    --start;
  }
  size_t pos = start;
  const char32_t c = DecodeUtf8(s, &pos);
  if (pos != *end) {
    // Malformed: decode the last byte alone.
    *end -= 1;
    return kReplacementCharacter;
  }
  *end = start;
  return c;
}

// Appends the UTF-8 encoding of `c` to `out` (`String::push(char)`).
inline void AppendUtf8(char32_t c, std::string* out) {
  if (c < 0x80) {
    out->push_back(static_cast<char>(c));
  } else if (c < 0x800) {
    out->push_back(static_cast<char>(0xC0 | (c >> 6)));
    out->push_back(static_cast<char>(0x80 | (c & 0x3F)));
  } else if (c < 0x10000) {
    out->push_back(static_cast<char>(0xE0 | (c >> 12)));
    out->push_back(static_cast<char>(0x80 | ((c >> 6) & 0x3F)));
    out->push_back(static_cast<char>(0x80 | (c & 0x3F)));
  } else {
    out->push_back(static_cast<char>(0xF0 | (c >> 18)));
    out->push_back(static_cast<char>(0x80 | ((c >> 12) & 0x3F)));
    out->push_back(static_cast<char>(0x80 | ((c >> 6) & 0x3F)));
    out->push_back(static_cast<char>(0x80 | (c & 0x3F)));
  }
}

// `str::chars()`: a forward iterator over the characters of a UTF-8 string.
// Like a `Fuse`d iterator it keeps returning `nullopt` once exhausted.
class Utf8Chars {
 public:
  explicit Utf8Chars(absl::string_view s) : s_(s), pos_(0) {}

  std::optional<char32_t> Next() {
    if (pos_ >= s_.size()) return std::nullopt;
    return DecodeUtf8(s_, &pos_);
  }

  // The number of bytes not consumed yet (an upper bound on the number of
  // remaining characters, used as a capacity hint).
  size_t RemainingBytes() const { return s_.size() - pos_; }

 private:
  absl::string_view s_;
  size_t pos_;
};

}  // namespace tokenizers::unicode::internal

#endif  // THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_UNICODE_UTF8_CHARS_H_
