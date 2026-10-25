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
// Copyright 2016 Sean Gillespie) plus the Rust standard library `char`
// predicates and case conversions the HuggingFace `tokenizers` crate v0.23.2
// uses (`library/core/src/char/methods.rs` and the `conversions` module of
// `library/core/src/unicode/unicode_data.rs` of rustc, MIT OR Apache-2.0,
// Copyright The Rust Project Developers). Kept line-by-line close to the Rust
// original on purpose; see
// support/tokenizer/tokenizers/README.md.

#include "support/tokenizer/tokenizers/unicode/categories.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <utility>

#include "absl/types/span.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/unicode/categories_tables.h"

namespace tokenizers::unicode {
namespace {

// ===========================================================================
// unicode_categories/src/lib.rs
// ===========================================================================

bool TableBinarySearch(char32_t target, absl::Span<const char32_t> table) {
  return std::binary_search(table.begin(), table.end(), target);
}

// ===========================================================================
// rustc library/core/src/unicode/unicode_data.rs, `mod conversions`
// ===========================================================================

// `[char; 3]`: a case mapping padded with '\0'.
using CaseChars = std::array<char32_t, 3>;

// `Range::start()`.
inline uint16_t RangeStart(const CaseRange& range) { return range.start; }

// `Range::end()`.
inline uint16_t RangeEnd(const CaseRange& range) {
  return range.start + static_cast<uint16_t>(range.len);
}

std::pair<uint16_t, uint16_t> Deconstruct(char32_t c) {
  const uint32_t u = static_cast<uint32_t>(c);
  const uint16_t plane = static_cast<uint16_t>(u >> 16);
  const uint16_t low = static_cast<uint16_t>(u);
  return std::make_pair(plane, low);
}

char32_t Reconstruct(uint16_t plane, uint16_t low) {
  // SAFETY: The caller must ensure that the result is a valid `char`.
  return static_cast<char32_t>((static_cast<uint32_t>(plane) << 16) |
                               static_cast<uint32_t>(low));
}

std::optional<CaseChars> Lookup(char32_t input, const CaseL1Lut& l1_lut) {
  const auto [input_high, input_low] = Deconstruct(input);
  // `l1_lut.l2_luts.get(input_high as usize)`.
  if (input_high >= 2) {
    return std::nullopt;
  }
  const CaseL2Lut& l2_lut = l1_lut.l2_luts[input_high];

  // `singles.binary_search_by(...)`: Greater if input_low < range.start(),
  // Less if input_low > range.end(), else Equal.
  const auto it =
      std::lower_bound(l2_lut.singles.begin(), l2_lut.singles.end(), input_low,
                       [](const CaseSingle& single, uint16_t low) {
                         return RangeEnd(single.range) < low;
                       });
  if (it != l2_lut.singles.end() && RangeStart(it->range) <= input_low &&
      input_low <= RangeEnd(it->range)) {
    // SAFETY: binary search guarantees that the index is in bounds.
    const CaseRange& range = it->range;
    const int16_t output_delta = it->delta;
    const uint16_t mask = range.parity ? 1 : 0;
    if ((input_low & mask) == (RangeStart(range) & mask)) {
      // `input_low.wrapping_add_signed(output_delta)`.
      const uint16_t output_low = static_cast<uint16_t>(
          static_cast<uint32_t>(input_low) +
          static_cast<uint32_t>(static_cast<int32_t>(output_delta)));
      // SAFETY: Table data are guaranteed to be valid Unicode.
      const char32_t output = Reconstruct(input_high, output_low);
      return CaseChars{output, U'\0', U'\0'};
    }
  }

  // `multis.binary_search_by_key(&input_low, |&(p, _)| p)`.
  const auto mit = std::lower_bound(
      l2_lut.multis.begin(), l2_lut.multis.end(), input_low,
      [](const CaseMulti& multi, uint16_t low) { return multi.key < low; });
  if (mit != l2_lut.multis.end() && mit->key == input_low) {
    // SAFETY: binary search guarantees that the index is in bounds.
    // SAFETY: Table data are guaranteed to be valid Unicode.
    return CaseChars{Reconstruct(input_high, mit->out[0]),
                     Reconstruct(input_high, mit->out[1]),
                     Reconstruct(input_high, mit->out[2])};
  }

  for (const CaseException& exception : l2_lut.exceptions) {
    if (exception.key == input_low) {
      return CaseChars{exception.out[0], exception.out[1], exception.out[2]};
    }
  }

  return std::nullopt;
}

// `char::to_ascii_lowercase`.
char32_t ToAsciiLowercase(char32_t c) {
  return (U'A' <= c && c <= U'Z') ? c + 0x20 : c;
}

// `char::to_ascii_uppercase`.
char32_t ToAsciiUppercase(char32_t c) {
  return (U'a' <= c && c <= U'z') ? c - 0x20 : c;
}

// `conversions::to_lower`.
CaseChars ToLower(char32_t c) {
  // https://util.unicode.org/UnicodeJsps/list-unicodeset.jsp?a=[:Changes_When_Lowercased:]-[:ASCII:]&abb=on
  if (c < 0xC0) {
    return CaseChars{ToAsciiLowercase(c), U'\0', U'\0'};
  }

  // The rest of Latin-1: À..=Þ (except ×) add 0x20, everything else maps to
  // itself.
  if (c <= 0xFF) {
    if (0xC0 <= c && c <= 0xDE && c != 0xD7) {
      // SAFETY: U+00C0..=U+00DE plus 0x20 lands in U+00E0..=U+00FE.
      return CaseChars{static_cast<char32_t>(c + 0x20), U'\0', U'\0'};
    }
    return CaseChars{c, U'\0', U'\0'};
  }

  return Lookup(c, kLowercaseLut).value_or(CaseChars{c, U'\0', U'\0'});
}

// `conversions::to_upper`.
CaseChars ToUpper(char32_t c) {
  // https://util.unicode.org/UnicodeJsps/list-unicodeset.jsp?a=[:Changes_When_Uppercased:]-[:ASCII:]&abb=on
  if (c < 0xB5) {
    return CaseChars{ToAsciiUppercase(c), U'\0', U'\0'};
  }
  // The rest of Latin-1: µ maps to U+039C, ß expands to SS,
  // à..=þ (except ÷) subtract 0x20, ÿ maps to U+0178.
  if (c <= 0xFF) {
    switch (c) {
      case 0xB5:
        return CaseChars{U'\u039C', U'\0', U'\0'};
      case 0xDF:
        return CaseChars{U'S', U'S', U'\0'};
      case 0xFF:
        return CaseChars{U'\u0178', U'\0', U'\0'};
      default:
        if (0xE0 <= c && c <= 0xFE && c != 0xF7) {
          // SAFETY: U+00E0..=U+00FE minus 0x20 lands in U+00C0..=U+00DE.
          return CaseChars{static_cast<char32_t>(c - 0x20), U'\0', U'\0'};
        }
        return CaseChars{c, U'\0', U'\0'};
    }
  }

  return Lookup(c, kUppercaseLut).value_or(CaseChars{c, U'\0', U'\0'});
}

// `CaseMappingIter::new`: the mapping without its '\0' padding.
CaseMapping ToCaseMapping(const CaseChars& chars) {
  CaseMapping mapping(chars.begin(), chars.end());
  if (chars[2] == U'\0') {
    mapping.pop_back();
    if (chars[1] == U'\0') {
      mapping.pop_back();

      // Deliberately don't check `chars[0]`,
      // as '\0' lowercases to itself
    }
  }
  return mapping;
}

}  // namespace

// ---- `impl UnicodeCategories for char` ------------------------------------

bool IsOtherControl(char32_t c) { return TableBinarySearch(c, kOtherControl); }

bool IsOtherFormat(char32_t c) { return TableBinarySearch(c, kOtherFormat); }

bool IsOtherPrivateUse(char32_t c) {
  // Private Use
  if (0xE000 <= c && c <= 0xF8FF) return true;
  // Plane 15, Private Use
  if (0xF0000 <= c && c <= 0xFFFFD) return true;
  // Plane 16, private Use
  if (0x100000 <= c && c <= 0x10FFFD) return true;
  return TableBinarySearch(c, kOtherPrivateUse);
}

bool IsLetterLowercase(char32_t c) {
  return TableBinarySearch(c, kLetterLowercased);
}

bool IsLetterModifier(char32_t c) {
  return TableBinarySearch(c, kLetterModifier);
}

bool IsLetterOther(char32_t c) {
  // CJK Ideograph Extension A
  if (0x3400 <= c && c <= 0x4DB5) return true;
  // CJK Ideograph
  if (0x4E00 <= c && c <= 0x9FD5) return true;
  // Hangul Syllable
  if (0xAC00 <= c && c <= 0xD7A3) return true;
  // Tangut Ideograph
  if (0x17000 <= c && c <= 0x187EC) return true;
  // CJK Ideograph Extension B
  if (0x20000 <= c && c <= 0x2A6D6) return true;
  // CJK Ideograph Extension C
  if (0x2A700 <= c && c <= 0x2B734) return true;
  // CJK Ideograph Extension D
  if (0x2B740 <= c && c <= 0x2B81D) return true;
  // CJK Ideograph Extension E
  if (0x2B820 <= c && c <= 0x2CEA1) return true;
  return TableBinarySearch(c, kLetterOther);
}

bool IsLetterTitlecase(char32_t c) {
  return TableBinarySearch(c, kLetterTitlecase);
}

bool IsLetterUppercase(char32_t c) {
  return TableBinarySearch(c, kLetterUppercase);
}

bool IsMarkSpacingCombining(char32_t c) {
  return TableBinarySearch(c, kMarkSpaceCombining);
}

bool IsMarkEnclosing(char32_t c) {
  return TableBinarySearch(c, kMarkEnclosing);
}

bool IsMarkNonspacing(char32_t c) {
  return TableBinarySearch(c, kMarkNonspacing);
}

bool IsNumberDecimalDigit(char32_t c) {
  return TableBinarySearch(c, kNumberDecimalDigit);
}

bool IsNumberLetter(char32_t c) { return TableBinarySearch(c, kNumberLetter); }

bool IsNumberOther(char32_t c) { return TableBinarySearch(c, kNumberOther); }

bool IsPunctuationConnector(char32_t c) {
  return TableBinarySearch(c, kPunctuationConnector);
}

bool IsPunctuationDash(char32_t c) {
  return TableBinarySearch(c, kPunctuationDash);
}

bool IsPunctuationClose(char32_t c) {
  return TableBinarySearch(c, kPunctuationClose);
}

bool IsPunctuationFinalQuote(char32_t c) {
  return TableBinarySearch(c, kPunctuationFinalQuote);
}

bool IsPunctuationInitialQuote(char32_t c) {
  return TableBinarySearch(c, kPunctuationInitialQuote);
}

bool IsPunctuationOther(char32_t c) {
  return TableBinarySearch(c, kPunctuationOther);
}

bool IsPunctuationOpen(char32_t c) {
  return TableBinarySearch(c, kPunctuationOpen);
}

bool IsSymbolCurrency(char32_t c) {
  return TableBinarySearch(c, kSymbolCurrency);
}

bool IsSymbolModifier(char32_t c) {
  return TableBinarySearch(c, kSymbolModifier);
}

bool IsSymbolMath(char32_t c) { return TableBinarySearch(c, kSymbolMath); }

bool IsSymbolOther(char32_t c) { return TableBinarySearch(c, kSymbolOther); }

bool IsSeparatorLine(char32_t c) {
  return TableBinarySearch(c, kSeparatorLine);
}

bool IsSeparatorParagraph(char32_t c) {
  return TableBinarySearch(c, kSeparatorParagraph);
}

bool IsSeparatorSpace(char32_t c) {
  return TableBinarySearch(c, kSeparatorSpace);
}

// ---- the provided methods of `trait UnicodeCategories` --------------------

bool IsOther(char32_t c) {
  return IsOtherControl(c) || IsOtherFormat(c) || IsOtherPrivateUse(c);
}

bool IsLetter(char32_t c) {
  return IsLetterLowercase(c) || IsLetterModifier(c) || IsLetterOther(c) ||
         IsLetterTitlecase(c) || IsLetterUppercase(c);
}

bool IsMark(char32_t c) {
  return IsMarkSpacingCombining(c) || IsMarkEnclosing(c) || IsMarkNonspacing(c);
}

bool IsNumber(char32_t c) {
  return IsNumberDecimalDigit(c) || IsNumberLetter(c) || IsNumberOther(c);
}

bool IsPunctuation(char32_t c) {
  // (The Rust code lists `is_punctuation_close()` twice.)
  return IsPunctuationConnector(c) || IsPunctuationDash(c) ||
         IsPunctuationClose(c) || IsPunctuationFinalQuote(c) ||
         IsPunctuationInitialQuote(c) || IsPunctuationOther(c) ||
         IsPunctuationOpen(c);
}

bool IsSymbol(char32_t c) {
  return IsSymbolCurrency(c) || IsSymbolModifier(c) || IsSymbolMath(c) ||
         IsSymbolOther(c);
}

bool IsSeparator(char32_t c) {
  return IsSeparatorLine(c) || IsSeparatorParagraph(c) || IsSeparatorSpace(c);
}

// ---- rustc library/core/src/char/methods.rs -------------------------------

bool IsWhitespace(char32_t c) {
  // ' ' | '\x09'..='\x0d' => true,
  if (c == U' ' || (0x09 <= c && c <= 0x0d)) return true;
  // '\0'..='\u{84}' => false,
  if (c <= 0x84) return false;
  // _ => unicode::White_Space(self),
  return WhiteSpaceLookup(c);
}

bool IsAlphabetic(char32_t c) {
  // 'a'..='z' | 'A'..='Z' => true,
  if ((U'a' <= c && c <= U'z') || (U'A' <= c && c <= U'Z')) return true;
  // '\0'..='\u{A9}' => false,
  if (c <= 0xA9) return false;
  // _ => unicode::Alphabetic(self),
  return DerivedPropertyAlphabetic(c);
}

bool IsNumeric(char32_t c) {
  // '0'..='9' => true,
  if (U'0' <= c && c <= U'9') return true;
  // '\0'..='\u{B1}' => false,
  if (c <= 0xB1) return false;
  // _ => unicode::N(self),
  return GeneralCategoryN(c);
}

bool IsAlphanumeric(char32_t c) {
  // 'a'..='z' | 'A'..='Z' | '0'..='9' => true,
  if ((U'a' <= c && c <= U'z') || (U'A' <= c && c <= U'Z') ||
      (U'0' <= c && c <= U'9')) {
    return true;
  }
  // '\0'..='\u{A9}' => false,
  if (c <= 0xA9) return false;
  // _ => unicode::Alphabetic(self) || unicode::N(self),
  return DerivedPropertyAlphabetic(c) || GeneralCategoryN(c);
}

bool IsControl(char32_t c) {
  // According to
  // https://www.unicode.org/policies/stability_policy.html#Property_Value,
  // the set of codepoints in `Cc` will never change.
  // So we can just hard-code the patterns to match against instead of using a
  // table.
  return c <= 0x1f || (0x7f <= c && c <= 0x9f);
}

bool IsAsciiPunctuation(char32_t c) {
  return (U'!' <= c && c <= U'/') || (U':' <= c && c <= U'@') ||
         (U'[' <= c && c <= U'`') || (U'{' <= c && c <= U'~');
}

CaseMapping ToLowercase(char32_t c) { return ToCaseMapping(ToLower(c)); }

CaseMapping ToUppercase(char32_t c) { return ToCaseMapping(ToUpper(c)); }

}  // namespace tokenizers::unicode
