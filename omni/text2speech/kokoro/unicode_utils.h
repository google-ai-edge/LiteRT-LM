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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2SPEECH_KOKORO_UNICODE_UTILS_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2SPEECH_KOKORO_UNICODE_UTILS_H_

#include <cstddef>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl

// UTF-8, case, and string helpers shared by the Kokoro phonemizer and its G2P
// engines (English, Spanish, Hindi, Portuguese, Italian, French, Japanese,
// and Mandarin Chinese).
//
// Malformed UTF-8 is never an error: a lead byte that does not start a
// complete, well-formed sequence decodes as the single code point equal to
// that byte, so callers always make progress.

namespace litert::omni::tts::kokoro {

// A code point decoded from UTF-8 and the number of bytes it occupied.
struct Utf8CodePoint {
  char32_t code_point = 0;
  size_t length = 0;
};

// Decodes the UTF-8 code point that starts at byte `pos` of `text`.
//
// args
// - text: UTF-8 text.
// - pos: Byte offset of the first byte of the code point.
//
// returns
// - The decoded code point and its byte length; `length` is 0 when `pos` is at
//   or past the end of `text`, and 1 for a malformed lead byte.
Utf8CodePoint DecodeUtf8CodePoint(absl::string_view text, size_t pos);

// Decodes the code point at byte `pos` of `text` and advances `pos` past it.
//
// args
// - text: UTF-8 text.
// - pos: Byte offset to decode at; advanced past the decoded code point.
// - code_point: Set to the decoded code point.
//
// returns
// - False, leaving `pos` and `code_point` unchanged, when `pos` is at or past
//   the end of `text`; true otherwise.
bool ConsumeUtf8CodePoint(absl::string_view text, size_t& pos,
                          char32_t& code_point);

// Appends the UTF-8 encoding of `code_point` to `out`.
//
// args
// - code_point: Unicode code point to encode.
// - out: String to append the 1 to 4 encoded bytes to.
void AppendUtf8(char32_t code_point, std::string& out);

// Decodes UTF-8 `text` into code points.
//
// args
// - text: UTF-8 text.
//
// returns
// - The code points of `text` as a UTF-32 string.
std::u32string Utf8ToU32(absl::string_view text);

// Decodes UTF-8 `text` into code points.
//
// args
// - text: UTF-8 text.
//
// returns
// - The code points of `text`.
std::vector<char32_t> Utf8ToCodePoints(absl::string_view text);

// Encodes code points as UTF-8.
//
// args
// - code_points: Code points to encode, e.g. a `std::u32string` or a
//   `std::vector<char32_t>`.
//
// returns
// - The UTF-8 encoding of `code_points`.
std::string CodePointsToUtf8(absl::Span<const char32_t> code_points);

// Lowercases an uppercase Latin letter: ASCII A-Z, the Latin-1 letters
// U+00C0..U+00DE (except U+00D7 ×), Œ, and Ÿ, which covers the uppercase
// letters of every Latin-script Kokoro language.
//
// args
// - code_point: Code point to lowercase.
//
// returns
// - The lowercase letter, or `code_point` unchanged if it is not one of the
//   letters above.
char32_t ToLowerLatin(char32_t code_point);

// Lowercases the Latin letters of UTF-8 `text` with `ToLowerLatin`.
//
// args
// - text: UTF-8 text.
//
// returns
// - `text` with every letter handled by `ToLowerLatin` lowercased.
std::string ToLowerUtf8(absl::string_view text);

// Returns whether `text` starts with any of `prefixes`.
//
// args
// - text: String to test.
// - prefixes: Candidate prefixes.
//
// returns
// - True if some prefix in `prefixes` is a prefix of `text`.
bool StartsWithAny(absl::string_view text,
                   absl::Span<const absl::string_view> prefixes);

// Returns whether `text` ends with any of `suffixes`.
//
// args
// - text: String to test.
// - suffixes: Candidate suffixes.
//
// returns
// - True if some suffix in `suffixes` is a suffix of `text`.
bool EndsWithAny(absl::string_view text,
                 absl::Span<const absl::string_view> suffixes);

}  // namespace litert::omni::tts::kokoro

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2SPEECH_KOKORO_UNICODE_UTILS_H_
