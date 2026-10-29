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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_UTILS_TEXT_UTILS_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_UTILS_TEXT_UTILS_H_

#include <cstddef>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl

// Unicode text helpers shared by Omni models, e.g. for languages written
// without spaces between words (Chinese, Japanese).
namespace litert::omni {

// Decodes the UTF-8 code point starting at byte `pos` of `text` and advances
// `pos` past it. An invalid or truncated sequence decodes as its first byte
// (and advances by one byte), so iteration always terminates. Returns 0 and
// leaves `pos` unchanged when `pos >= text.size()`.
char32_t DecodeUtf8CodePoint(absl::string_view text, size_t& pos);

// Returns true for Han ideographs, Hiragana and Katakana, and the letters and
// numerals of the CJK Symbols and Punctuation block (e.g. "々" in "人々", "〇"
// in "二〇二六"): characters of scripts written without spaces between words.
bool IsCjkCharacter(char32_t code_point);

// Returns true for CJK punctuation and symbols (U+3001..U+303F except the
// letters and numerals there, e.g. "。、「」") and fullwidth ASCII punctuation
// (e.g. "，？！："). The ideographic space U+3000 is not punctuation.
bool IsCjkPunctuation(char32_t code_point);

// Returns true for ASCII punctuation, CJK and fullwidth punctuation, and the
// General Punctuation block (e.g. curly quotes, dashes, ellipsis).
bool IsPunctuation(char32_t code_point);

// Splits `words` so that every CJK character (see `IsCjkCharacter`) is its own
// piece, keeping the punctuation that directly follows it attached ("吗？" is
// one piece). Other text, e.g. Latin words, is kept as is. This turns
// space-separated "words" into units that can be aligned or compared for any
// script: a decoder that splits on spaces otherwise yields a whole Chinese
// sentence as one word.
std::vector<std::string> SplitCjkCharacters(
    absl::Span<const std::string> words);

// Joins `pieces` with a space between them, except between two CJK characters
// and next to CJK punctuation, which carries its own spacing. Inverts
// `SplitCjkCharacters` for text that separates words with spaces, puts spaces
// around Latin words inside CJK text (e.g. "欢迎体验 LiteRT 演示。") and none
// next to fullwidth punctuation (e.g. "请看（LiteRT）演示").
std::string JoinWithCjkAwareSpacing(absl::Span<const std::string> pieces);

}  // namespace litert::omni

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_UTILS_TEXT_UTILS_H_
