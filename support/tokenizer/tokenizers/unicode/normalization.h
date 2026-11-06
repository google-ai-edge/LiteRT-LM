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
// C++ port of the `unicode_normalization_alignments` crate (the files
// `google/{lib,decompose,recompose}.rs` of the HuggingFace `tokenizers` crate
// v0.23.2, https://github.com/huggingface/tokenizers, Apache License 2.0,
// Copyright 2020 The HuggingFace Team) and of the parts of the
// `unicode-normalization` crate v0.1.25 it builds on (`src/{lookups,
// perfect_hash,normalize}.rs`, https://github.com/unicode-rs/unicode-
// normalization, MIT OR Apache-2.0, Copyright 2012-2019 The Rust Project
// Developers). Kept line-by-line close to the Rust original on purpose; see
// support/tokenizer/tokenizers/README.md.
//
// Unicode version of the tables: 17.0.0 (see normalization_tables.h).

#ifndef THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_UNICODE_NORMALIZATION_H_
#define THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_UNICODE_NORMALIZATION_H_

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/functional/function_ref.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl

namespace tokenizers::unicode {

// ---- `unicode_normalization::char` -----------------------------------------

// Look up the canonical combining class for a codepoint.
//
// The value returned is as defined in the Unicode Character Database.
uint8_t CanonicalCombiningClass(char32_t c);

// Return whether the given character is a combining mark
// (`General_Category=Mark`).
bool IsCombiningMark(char32_t c);

// Compute canonical Unicode decomposition for character.
// See [Unicode Standard Annex #15](http://www.unicode.org/reports/tr15/)
// for more information.
void DecomposeCanonical(char32_t c,
                        absl::FunctionRef<void(char32_t)> emit_char);

// Compute canonical or compatible Unicode decomposition for character.
// See [Unicode Standard Annex #15](http://www.unicode.org/reports/tr15/)
// for more information.
void DecomposeCompatible(char32_t c,
                         absl::FunctionRef<void(char32_t)> emit_char);

// Compose two characters into a single character, if possible.
// See [Unicode Standard Annex #15](http://www.unicode.org/reports/tr15/)
// for more information.
std::optional<char32_t> Compose(char32_t a, char32_t b);

// ---- `unicode_normalization_alignments::UnicodeNormalization` --------------
//
// Each of the functions below normalizes the (valid UTF-8) string `s` and
// returns the normalized characters together with the alignment delta that the
// Rust iterators yield as the second tuple element: `0` for a character that
// starts the normalization of an original character, `1` for every additional
// character produced by a decomposition, and `k.1 + change - 1` (i.e. possibly
// negative) for a character produced by composing several characters. This is
// exactly the input `NormalizedString::transform()` expects.
//
// PORT NOTE: the Rust crate exposes lazy iterators (`Decompositions`,
// `Recompositions`) implementing `Iterator<Item = (char, isize)>`; the
// tokenizers crate only ever drains them, so the port returns the drained
// vector.

// Returns the string in Unicode Normalization Form D (canonical decomposition).
std::vector<std::pair<char32_t, int64_t>> Nfd(absl::string_view s);

// Returns the string in Unicode Normalization Form KD (compatibility
// decomposition).
std::vector<std::pair<char32_t, int64_t>> Nfkd(absl::string_view s);

// Returns the string in Unicode Normalization Form C (canonical decomposition
// followed by canonical composition).
std::vector<std::pair<char32_t, int64_t>> Nfc(absl::string_view s);

// Returns the string in Unicode Normalization Form KC (compatibility
// decomposition followed by canonical composition).
std::vector<std::pair<char32_t, int64_t>> Nfkc(absl::string_view s);

}  // namespace tokenizers::unicode

#endif  // THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_UNICODE_NORMALIZATION_H_
