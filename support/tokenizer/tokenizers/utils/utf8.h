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
// Support code for the C++ port of the HuggingFace `tokenizers` crate v0.23.2
// (https://github.com/huggingface/tokenizers, Apache License 2.0,
// Copyright 2020 The HuggingFace Team); see
// support/tokenizer/tokenizers/README.md.
//
// The Rust crate works on `&str`/`String` (always valid UTF-8) and `char`
// (a Unicode scalar value) and relies on the standard library for the
// conversions between the two: `str::chars()`, `str::char_indices()`,
// `char::len_utf8()`, `String::push(char)`, `String::from_utf8_lossy()`, ...
// This header provides the equivalents used by the port. Strings are UTF-8
// (`absl::string_view` / `std::string`), characters are `char32_t`, and byte
// offsets mean exactly what they mean in Rust.

#ifndef THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_UTILS_UTF8_H_
#define THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_UTILS_UTF8_H_

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <string>
#include <utility>

#include "absl/strings/string_view.h"  // from @com_google_absl

namespace tokenizers {

// `char::REPLACEMENT_CHARACTER`.
inline constexpr char32_t kReplacementCharacter = U'\uFFFD';

namespace utf8_internal {

// Decodes the UTF-8 sequence starting at `s[pos]` (`pos < s.size()` required).
// On success stores the code point in `*code_point`, its encoded length in
// `*len` and returns true. On failure returns false and stores in `*len` the
// length (>= 1) of the "maximal subpart" of an ill-formed sequence, which is
// exactly the span `String::from_utf8_lossy` replaces with one U+FFFD (the
// `error_len` of `core::str::Utf8Error`, or everything up to the end of the
// input for a truncated sequence).
inline bool DecodeOne(absl::string_view s, size_t pos, char32_t* code_point,
                      size_t* len) {
  const auto* bytes = reinterpret_cast<const unsigned char*>(s.data());
  const unsigned char lead = bytes[pos];
  if (lead < 0x80) {
    *code_point = lead;
    *len = 1;
    return true;
  }
  size_t need;              // number of continuation bytes
  unsigned char lo = 0x80;  // allowed range of the first continuation byte
  unsigned char hi = 0xBF;
  char32_t cp;
  if (lead >= 0xC2 && lead <= 0xDF) {
    need = 1;
    cp = lead & 0x1F;
  } else if (lead >= 0xE0 && lead <= 0xEF) {
    need = 2;
    cp = lead & 0x0F;
    if (lead == 0xE0) lo = 0xA0;  // no overlong encodings
    if (lead == 0xED) hi = 0x9F;  // no surrogates
  } else if (lead >= 0xF0 && lead <= 0xF4) {
    need = 3;
    cp = lead & 0x07;
    if (lead == 0xF0) lo = 0x90;  // no overlong encodings
    if (lead == 0xF4) hi = 0x8F;  // nothing above U+10FFFF
  } else {
    // 0x80..0xC1 (stray continuation byte / overlong lead) or 0xF5..0xFF.
    *len = 1;
    return false;
  }
  for (size_t k = 1; k <= need; ++k) {
    if (pos + k >= s.size()) {
      *len = k;  // truncated sequence at the end of the input
      return false;
    }
    const unsigned char b = bytes[pos + k];
    if (k == 1 ? (b < lo || b > hi) : (b < 0x80 || b > 0xBF)) {
      *len = k;
      return false;
    }
    cp = (cp << 6) | (b & 0x3F);
  }
  *code_point = cp;
  *len = need + 1;
  return true;
}

}  // namespace utf8_internal

// `char::len_utf8()`: the number of bytes of the UTF-8 encoding of `c`.
inline size_t Utf8Len(char32_t c) {
  if (c < 0x80) return 1;
  if (c < 0x800) return 2;
  if (c < 0x10000) return 3;
  return 4;
}

// `String::push(c)`: appends the UTF-8 encoding of `c` to `*out`.
// PORT NOTE: a Rust `char` is always a Unicode scalar value; values that are
// not (surrogates, > U+10FFFF) are encoded as U+FFFD here.
inline void AppendUtf8(char32_t c, std::string* out) {
  if ((c >= 0xD800 && c <= 0xDFFF) || c > 0x10FFFF) c = kReplacementCharacter;
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

// `char::to_string()`: the UTF-8 encoding of `c`.
inline std::string ToUtf8(char32_t c) {
  std::string out;
  AppendUtf8(c, &out);
  return out;
}

// Decodes the character starting at byte offset `*pos` of `s` and advances
// `*pos` past it (`*pos < s.size()` required). This is what `s[*pos..].chars()
// .next()` does in Rust.
// PORT NOTE: Rust strings are always valid UTF-8. For an ill-formed sequence
// this returns U+FFFD and advances past its maximal subpart, like
// `String::from_utf8_lossy` would.
inline char32_t DecodeUtf8(absl::string_view s, size_t* pos) {
  char32_t code_point = kReplacementCharacter;
  size_t len = 1;
  if (!utf8_internal::DecodeOne(s, *pos, &code_point, &len)) {
    code_point = kReplacementCharacter;
  }
  *pos += len;
  return code_point;
}

// `str::from_utf8(bytes).is_ok()`: whether `s` is well-formed UTF-8.
inline bool IsValidUtf8(absl::string_view s) {
  size_t pos = 0;
  while (pos < s.size()) {
    char32_t code_point;
    size_t len;
    if (!utf8_internal::DecodeOne(s, pos, &code_point, &len)) return false;
    pos += len;
  }
  return true;
}

// `String::from_utf8_lossy(bytes).into_owned()`: `s` with every maximal
// subpart of an ill-formed sequence replaced by U+FFFD.
inline std::string FromUtf8Lossy(absl::string_view s) {
  std::string out;
  out.reserve(s.size());
  size_t pos = 0;
  while (pos < s.size()) {
    char32_t code_point;
    size_t len;
    if (utf8_internal::DecodeOne(s, pos, &code_point, &len)) {
      out.append(s.data() + pos, len);
    } else {
      AppendUtf8(kReplacementCharacter, &out);
    }
    pos += len;
  }
  return out;
}

// `s.chars().count()`: the number of characters (code points) in `s`.
inline size_t CountChars(absl::string_view s) {
  size_t count = 0;
  size_t pos = 0;
  while (pos < s.size()) {
    DecodeUtf8(s, &pos);
    ++count;
  }
  return count;
}

// `str::char_indices()`: a range of `(byte_offset, char)` pairs, one per
// character of the string:
//
//   for (auto [offset, c] : CharIndices(s)) { ... }
//
// The iterator is a forward iterator; `*it` is a `std::pair<size_t, char32_t>`
// by value. `s.chars()` is `CharIndices(s)` ignoring the offsets.
class CharIndices {
 public:
  using value_type = std::pair<size_t, char32_t>;

  class Iterator {
   public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = CharIndices::value_type;
    using difference_type = std::ptrdiff_t;
    using pointer = const value_type*;
    using reference = value_type;

    Iterator() = default;
    Iterator(absl::string_view s, size_t pos) : s_(s), pos_(pos) { Load(); }

    value_type operator*() const { return {pos_, c_}; }
    Iterator& operator++() {
      pos_ = next_pos_;
      Load();
      return *this;
    }
    Iterator operator++(int) {
      Iterator copy = *this;
      ++*this;
      return copy;
    }
    bool operator==(const Iterator& other) const { return pos_ == other.pos_; }
    bool operator!=(const Iterator& other) const { return pos_ != other.pos_; }

    // The byte offset of the current character, or `s.size()` at the end.
    size_t offset() const { return pos_; }

   private:
    void Load() {
      next_pos_ = pos_;
      c_ = pos_ < s_.size() ? DecodeUtf8(s_, &next_pos_) : 0;
    }

    absl::string_view s_;
    size_t pos_ = 0;
    size_t next_pos_ = 0;
    char32_t c_ = 0;
  };

  using iterator = Iterator;
  using const_iterator = Iterator;

  explicit CharIndices(absl::string_view s) : s_(s) {}

  Iterator begin() const { return Iterator(s_, 0); }
  Iterator end() const { return Iterator(s_, s_.size()); }

 private:
  absl::string_view s_;
};

}  // namespace tokenizers

#endif  // THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_UTILS_UTF8_H_
