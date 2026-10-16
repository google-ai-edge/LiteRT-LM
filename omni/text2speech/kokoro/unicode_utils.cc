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

#include "omni/text2speech/kokoro/unicode_utils.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl

namespace litert::omni::tts::kokoro {
namespace {

bool IsContinuationByte(uint8_t byte) { return (byte & 0xC0) == 0x80; }

uint8_t ByteAt(absl::string_view text, size_t pos) {
  return static_cast<uint8_t>(text[pos]);
}

}  // namespace

Utf8CodePoint DecodeUtf8CodePoint(absl::string_view text, size_t pos) {
  if (pos >= text.size()) {
    return {0, 0};
  }
  const uint8_t b0 = ByteAt(text, pos);
  if (b0 < 0x80) {
    return {b0, 1};
  }
  if ((b0 & 0xE0) == 0xC0 && pos + 1 < text.size()) {
    const uint8_t b1 = ByteAt(text, pos + 1);
    if (IsContinuationByte(b1)) {
      return {static_cast<char32_t>(((b0 & 0x1F) << 6) | (b1 & 0x3F)), 2};
    }
  }
  if ((b0 & 0xF0) == 0xE0 && pos + 2 < text.size()) {
    const uint8_t b1 = ByteAt(text, pos + 1);
    const uint8_t b2 = ByteAt(text, pos + 2);
    if (IsContinuationByte(b1) && IsContinuationByte(b2)) {
      return {static_cast<char32_t>(((b0 & 0x0F) << 12) | ((b1 & 0x3F) << 6) |
                                    (b2 & 0x3F)),
              3};
    }
  }
  if ((b0 & 0xF8) == 0xF0 && pos + 3 < text.size()) {
    const uint8_t b1 = ByteAt(text, pos + 1);
    const uint8_t b2 = ByteAt(text, pos + 2);
    const uint8_t b3 = ByteAt(text, pos + 3);
    if (IsContinuationByte(b1) && IsContinuationByte(b2) &&
        IsContinuationByte(b3)) {
      return {static_cast<char32_t>(((b0 & 0x07) << 18) | ((b1 & 0x3F) << 12) |
                                    ((b2 & 0x3F) << 6) | (b3 & 0x3F)),
              4};
    }
  }
  return {b0, 1};
}

bool ConsumeUtf8CodePoint(absl::string_view text, size_t& pos,
                          char32_t& code_point) {
  const Utf8CodePoint decoded = DecodeUtf8CodePoint(text, pos);
  if (decoded.length == 0) {
    return false;
  }
  code_point = decoded.code_point;
  pos += decoded.length;
  return true;
}

void AppendUtf8(char32_t code_point, std::string& out) {
  if (code_point < 0x80) {
    out.push_back(static_cast<char>(code_point));
  } else if (code_point < 0x800) {
    out.reserve(out.size() + 2);
    out.push_back(static_cast<char>(0xC0 | (code_point >> 6)));
    out.push_back(static_cast<char>(0x80 | (code_point & 0x3F)));
  } else if (code_point < 0x10000) {
    out.reserve(out.size() + 3);
    out.push_back(static_cast<char>(0xE0 | (code_point >> 12)));
    out.push_back(static_cast<char>(0x80 | ((code_point >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (code_point & 0x3F)));
  } else {
    out.reserve(out.size() + 4);
    out.push_back(static_cast<char>(0xF0 | ((code_point >> 18) & 0x07)));
    out.push_back(static_cast<char>(0x80 | ((code_point >> 12) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | ((code_point >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (code_point & 0x3F)));
  }
}

std::u32string Utf8ToU32(absl::string_view text) {
  std::u32string out;
  out.reserve(text.size());
  size_t pos = 0;
  char32_t code_point = 0;
  while (ConsumeUtf8CodePoint(text, pos, code_point)) {
    out.push_back(code_point);
  }
  return out;
}

std::vector<char32_t> Utf8ToCodePoints(absl::string_view text) {
  std::vector<char32_t> out;
  out.reserve(text.size());
  size_t pos = 0;
  char32_t code_point = 0;
  while (ConsumeUtf8CodePoint(text, pos, code_point)) {
    out.push_back(code_point);
  }
  return out;
}

std::string CodePointsToUtf8(absl::Span<const char32_t> code_points) {
  std::string out;
  out.reserve(code_points.size() * 2);
  for (const char32_t code_point : code_points) {
    AppendUtf8(code_point, out);
  }
  return out;
}

char32_t ToLowerLatin(char32_t code_point) {
  if (code_point >= U'A' && code_point <= U'Z') {
    return code_point + (U'a' - U'A');
  }
  if (code_point >= 0xC0 && code_point <= 0xDE && code_point != 0xD7) {
    return code_point + 0x20;
  }
  if (code_point == 0x152) {  // Œ
    return 0x153;
  }
  if (code_point == 0x178) {  // Ÿ
    return 0xFF;
  }
  return code_point;
}

std::string ToLowerUtf8(absl::string_view text) {
  std::string out;
  out.reserve(text.size());
  size_t pos = 0;
  char32_t code_point = 0;
  while (ConsumeUtf8CodePoint(text, pos, code_point)) {
    AppendUtf8(ToLowerLatin(code_point), out);
  }
  return out;
}

bool StartsWithAny(absl::string_view text,
                   absl::Span<const absl::string_view> prefixes) {
  for (const absl::string_view prefix : prefixes) {
    if (absl::StartsWith(text, prefix)) {
      return true;
    }
  }
  return false;
}

bool EndsWithAny(absl::string_view text,
                 absl::Span<const absl::string_view> suffixes) {
  for (const absl::string_view suffix : suffixes) {
    if (absl::EndsWith(text, suffix)) {
      return true;
    }
  }
  return false;
}

}  // namespace litert::omni::tts::kokoro
