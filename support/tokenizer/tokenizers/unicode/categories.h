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
// C++ port of the `unicode_categories` crate v0.1.1 (`src/lib.rs`,
// https://github.com/swgillespie/unicode-categories, MIT OR Apache-2.0,
// Copyright 2016 Sean Gillespie), i.e. the `UnicodeCategories` trait the
// HuggingFace `tokenizers` crate v0.23.2 uses, plus the Rust standard library
// `char` predicates and case conversions the tokenizers crate uses
// (`library/core/src/char/methods.rs` of rustc, MIT OR Apache-2.0, Copyright
// The Rust Project Developers). Kept line-by-line close to the Rust original
// on purpose; see
// support/tokenizer/tokenizers/README.md.
//
// Table provenance (see categories_tables.h for details):
//  * the general-category predicates use the tables of the `unicode_categories`
//    crate itself (so that behaviour is identical to the Rust tokenizers);
//  * `IsAlphabetic` / `IsNumeric` use the Unicode 17.0.0 `Alphabetic` and `N`
//    range tables of the `unicode-segmentation` crate v1.13.3;
//  * `IsWhitespace`, `ToLowercase` and `ToUppercase` use the tables of the Rust
//    standard library (`library/core/src/unicode/unicode_data.rs`).

#ifndef THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_UNICODE_CATEGORIES_H_
#define THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_UNICODE_CATEGORIES_H_

#include "absl/container/inlined_vector.h"  // from @com_google_absl

namespace tokenizers::unicode {

// ---- `unicode_categories::UnicodeCategories` -------------------------------

// Returns `true` if this value is a member
// of the "Other, Control" (Cc) category.
bool IsOtherControl(char32_t c);

// Returns `true` if this value is a member
// of the "Other, Format" (Cf) category.
bool IsOtherFormat(char32_t c);

// Returns true if this value is a member
// of the "Other, Private Use" (Co) category.
bool IsOtherPrivateUse(char32_t c);

// Returns true if this value is a member
// of the "Letter, Lowercase" (Ll) category.
bool IsLetterLowercase(char32_t c);

// Returns true if this value is a member of
// the "Letter, Modifier" (Lm) category.
bool IsLetterModifier(char32_t c);

// Returns true if this value is a member of
// the "Letter, Other" (Lo) category.
bool IsLetterOther(char32_t c);

// Returns true if this value is a member of
// the "Letter, Titlecase" (Lt) category.
bool IsLetterTitlecase(char32_t c);

// Returns true if this value is a member of
// the "Letter, Uppercase" (Lu) category.
bool IsLetterUppercase(char32_t c);

// Returns true if this value is a member of
// the "Mark, Spacing Combining" (Mc) category.
bool IsMarkSpacingCombining(char32_t c);

// Returns true if this value is a member of
// the "Mark, Enclosing" (Me) category.
bool IsMarkEnclosing(char32_t c);

// Returns true if this value is a member of
// the "Mark, Nonspacing" (Mn) category.
bool IsMarkNonspacing(char32_t c);

// Returns true if this value is a member of
// the "Number, Decimal Digit" (Nd) category.
bool IsNumberDecimalDigit(char32_t c);

// Returns true if this value is a member of
// the "Number, Letter" (Nl) category.
bool IsNumberLetter(char32_t c);

// Returns true if this value is a member of
// the "Number, Other" (No) category.
bool IsNumberOther(char32_t c);

// Returns true if this value is a member of
// the "Punctuation, Connector" (Pc) category.
bool IsPunctuationConnector(char32_t c);

// Returns true if this value is a member of
// the "Punctuation, Dash" (Pd) category.
bool IsPunctuationDash(char32_t c);

// Returns true if this value is a member of
// the "Punctuation, Close" (Pe) category.
bool IsPunctuationClose(char32_t c);

// Returns true if this value is a member of
// the "Punctuation, Final Quote" (Pf) category.
bool IsPunctuationFinalQuote(char32_t c);

// Returns true if this value is a member of
// the "Punctuation, Initial Quote" (Pi) category.
bool IsPunctuationInitialQuote(char32_t c);

// Returns true if this value is a member of
// the "Punctuation, Other" (Po) category.
bool IsPunctuationOther(char32_t c);

// Returns true if this value is a member of
// the "Punctuation, Open" (Ps) category.
bool IsPunctuationOpen(char32_t c);

// Returns true if this value is a member of
// the "Symbol, Currency" (Sc) category.
bool IsSymbolCurrency(char32_t c);

// Returns true if this value is a member of
// the "Symbol, Modifier" (Sk) category.
bool IsSymbolModifier(char32_t c);

// Returns true if this value is a member of
// the "Symbol, Math" (Sm) category.
bool IsSymbolMath(char32_t c);

// Returns true if this value is a member of
// the "Symbol, Other" (So) category.
bool IsSymbolOther(char32_t c);

// Returns true if this value is a member of
// the "Separator, Line" (Zl) category.
bool IsSeparatorLine(char32_t c);

// Returns true if this value is a member of
// the "Separator, Paragraph" (Zp) category.
bool IsSeparatorParagraph(char32_t c);

// Returns true if this value is a member of
// the "Separator, Space" (Zs) category.
bool IsSeparatorSpace(char32_t c);

// Returns true if this value is a member of
// a "Other" category: Cc, Cf, Cn, or Co.
// Surrogates cannot be `chars` in Rust, so
// they are not included.
bool IsOther(char32_t c);

// Returns true if this value is a member of
// a "Letter" category: Lc, Ll, Lm, Lo, Lt, or Lu.
bool IsLetter(char32_t c);

// Returns true if this value is a member of a
// "Mark" category: Mc, Me, or Mn.
bool IsMark(char32_t c);

// Returns true if this value is a member of a
// "Number" category: Nd, Nl, or No.
bool IsNumber(char32_t c);

// Returns true if this value is a member of a
// "Punctuation" category: Pc, Pd, Pe, Pf, Pi, Po, or Ps.
bool IsPunctuation(char32_t c);

// Returns true if this value is a member of a
// "Symbol" category: Sc, Sk, Sm, or So.
bool IsSymbol(char32_t c);

// Returns true if this value is a member of a
// "Separator" category: Zl, Zp, or Zs.
bool IsSeparator(char32_t c);

// ---- Rust standard library `char` methods ---------------------------------

// `char::is_whitespace`: returns `true` if this `char` has the `White_Space`
// property.
bool IsWhitespace(char32_t c);

// `char::is_alphabetic`: returns `true` if this `char` has the `Alphabetic`
// property.
bool IsAlphabetic(char32_t c);

// `char::is_numeric`: returns `true` if this `char` has one of the general
// categories for numbers (`Nd`, `Nl`, `No`).
bool IsNumeric(char32_t c);

// `char::is_alphanumeric`: returns `true` if this `char` satisfies either
// `IsAlphabetic()` or `IsNumeric()`.
bool IsAlphanumeric(char32_t c);

// `char::is_control`: returns `true` if this `char` has the general category
// for control codes (`Cc`).
bool IsControl(char32_t c);

// `char::is_ascii_punctuation`: checks if the value is an ASCII punctuation
// character: U+0021 ..= U+002F, U+003A ..= U+0040, U+005B ..= U+0060 or
// U+007B ..= U+007E.
bool IsAsciiPunctuation(char32_t c);

// The characters produced by `char::to_lowercase()` / `char::to_uppercase()`
// (one to three code points; `char::MAX_LEN_UTF8`-style guarantee of the
// Unicode standard: case mappings expand to at most three code points).
using CaseMapping = absl::InlinedVector<char32_t, 3>;

// `char::to_lowercase`: returns the lowercase mapping of this `char` as per
// `UnicodeData.txt` and `SpecialCasing.txt` (unconditional mappings only),
// e.g. U+0130 'İ' -> "i\u0307".
CaseMapping ToLowercase(char32_t c);

// `char::to_uppercase`: returns the uppercase mapping of this `char` as per
// `UnicodeData.txt` and `SpecialCasing.txt` (unconditional mappings only),
// e.g. U+00DF 'ß' -> "SS".
CaseMapping ToUppercase(char32_t c);

}  // namespace tokenizers::unicode

#endif  // THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_UNICODE_CATEGORIES_H_
