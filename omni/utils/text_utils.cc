/*
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "omni/utils/text_utils.h"

#include <cctype>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl

namespace litert::omni {

char32_t DecodeUtf8CodePoint(absl::string_view text, size_t& pos) {
  if (pos >= text.size()) return 0;
  const unsigned char c0 = static_cast<unsigned char>(text[pos]);
  int len = 1;
  char32_t cp = c0;
  if (c0 >= 0xC0 && c0 < 0xE0) {
    len = 2;
    cp = c0 & 0x1F;
  } else if (c0 >= 0xE0 && c0 < 0xF0) {
    len = 3;
    cp = c0 & 0x0F;
  } else if (c0 >= 0xF0 && c0 < 0xF8) {
    len = 4;
    cp = c0 & 0x07;
  }
  if (len == 1 || pos + len > text.size()) {
    ++pos;
    return c0;
  }
  for (int k = 1; k < len; ++k) {
    const unsigned char ck = static_cast<unsigned char>(text[pos + k]);
    if ((ck & 0xC0) != 0x80) {
      ++pos;
      return c0;
    }
    cp = (cp << 6) | (ck & 0x3F);
  }
  pos += len;
  return cp;
}

bool IsCjkCharacter(char32_t code_point) {
  const char32_t cp = code_point;
  return (cp >= 0x3005 && cp <= 0x3007) ||  // Iteration marks, 〆, 〇
         (cp >= 0x3021 && cp <= 0x3029) ||  // Hangzhou numerals
         (cp >= 0x3031 && cp <= 0x3035) ||  // Kana repeat marks
         (cp >= 0x3038 && cp <= 0x303C) ||  // Hangzhou numerals, 〻, 〼
         (cp >= 0x3040 && cp <= 0x30FF) ||  // Hiragana, Katakana
         (cp >= 0x3400 && cp <= 0x4DBF) ||  // CJK Extension A
         (cp >= 0x4E00 && cp <= 0x9FFF) ||  // CJK Unified Ideographs
         (cp >= 0xF900 && cp <= 0xFAFF) ||  // CJK Compatibility Ideographs
         (cp >= 0x20000 && cp <= 0x2FA1F);  // CJK Extensions B-F
}

bool IsCjkPunctuation(char32_t code_point) {
  const char32_t cp = code_point;
  // U+3001..U+303F except the letters and numerals in that block (see
  // `IsCjkCharacter`). U+3000 is the ideographic space, not punctuation.
  if (cp >= 0x3001 && cp <= 0x303F) return !IsCjkCharacter(cp);
  return (cp >= 0xFF01 && cp <= 0xFF0F) || (cp >= 0xFF1A && cp <= 0xFF20) ||
         (cp >= 0xFF3B && cp <= 0xFF40) || (cp >= 0xFF5B && cp <= 0xFF65);
}

bool IsPunctuation(char32_t code_point) {
  const char32_t cp = code_point;
  if (cp < 0x80) return std::ispunct(static_cast<int>(cp)) != 0;
  return IsCjkPunctuation(cp) || (cp >= 0x2010 && cp <= 0x205E) ||
         cp == 0x00A1 || cp == 0x00AB || cp == 0x00BB || cp == 0x00BF;
}

std::vector<std::string> SplitCjkCharacters(
    absl::Span<const std::string> words) {
  std::vector<std::string> pieces;
  pieces.reserve(words.size());
  for (const std::string& word : words) {
    // Non-CJK text accumulated since the last CJK character of this word.
    std::string pending;
    // Whether the last piece pushed for this word ends with a CJK character,
    // so that directly following punctuation attaches to it.
    bool attach_punctuation = false;
    size_t pos = 0;
    while (pos < word.size()) {
      const size_t start = pos;
      const char32_t cp = DecodeUtf8CodePoint(word, pos);
      const absl::string_view ch(word.data() + start, pos - start);
      if (IsCjkCharacter(cp)) {
        if (!pending.empty()) pieces.push_back(std::move(pending));
        pending.clear();
        pieces.emplace_back(ch);
        attach_punctuation = true;
      } else if (attach_punctuation && IsPunctuation(cp)) {
        pieces.back().append(ch);
      } else {
        pending.append(ch);
        attach_punctuation = false;
      }
    }
    if (!pending.empty()) pieces.push_back(std::move(pending));
  }
  return pieces;
}

std::string JoinWithCjkAwareSpacing(absl::Span<const std::string> pieces) {
  std::string text;
  for (const std::string& piece : pieces) {
    if (piece.empty()) continue;
    if (!text.empty()) {
      // Decode the last code point of `text` and the first one of `piece`.
      size_t last_start = text.size() - 1;
      while (last_start > 0 &&
             (static_cast<unsigned char>(text[last_start]) & 0xC0) == 0x80) {
        --last_start;
      }
      const char32_t last = DecodeUtf8CodePoint(text, last_start);
      size_t first_pos = 0;
      const char32_t first = DecodeUtf8CodePoint(piece, first_pos);
      // Fullwidth CJK punctuation carries its own spacing, so it never gets an
      // extra space on either side ("好，OK", "（LiteRT）").
      const bool glued = IsCjkPunctuation(last) || IsCjkPunctuation(first) ||
                         (IsCjkCharacter(last) && IsCjkCharacter(first));
      if (!glued) text.push_back(' ');
    }
    text.append(piece);
  }
  return text;
}

}  // namespace litert::omni
