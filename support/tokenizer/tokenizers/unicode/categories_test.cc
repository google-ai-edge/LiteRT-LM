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
// Tests of the C++ port of `unicode_categories` (the crate's own tests in
// `src/lib.rs`) and of the Rust standard library `char` methods.

#include "support/tokenizer/tokenizers/unicode/categories.h"

#include <cstdint>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace tokenizers::unicode {
namespace {

using ::testing::ElementsAre;

// ---- unicode_categories/src/lib.rs `mod tests` -----------------------------

TEST(UnicodeCategories, IsOtherControl) {
  EXPECT_TRUE(IsOtherControl(U'\0'));
  EXPECT_TRUE(IsOtherControl(U'\u007F'));
  EXPECT_FALSE(IsOtherControl(U'f'));
}

TEST(UnicodeCategories, IsOtherFormat) {
  EXPECT_TRUE(IsOtherFormat(U'\u0601'));  // '؁'
  EXPECT_FALSE(IsOtherFormat(U'0'));
}

TEST(UnicodeCategories, IsOtherPrivateUse) {
  EXPECT_TRUE(IsOtherPrivateUse(U'\uF8FF'));
  EXPECT_FALSE(IsOtherPrivateUse(U'n'));
}

TEST(UnicodeCategories, IsLetterLowercase) {
  EXPECT_TRUE(IsLetterLowercase(U'q'));
  EXPECT_FALSE(IsLetterLowercase(U'N'));
}

TEST(UnicodeCategories, IsLetterModifier) {
  EXPECT_TRUE(IsLetterModifier(U'\u02E2'));  // 'ˢ'
  EXPECT_FALSE(IsLetterModifier(U'm'));
}

TEST(UnicodeCategories, IsLetterRange) {
  EXPECT_TRUE(IsLetterOther(U'\u754C'));  // '界'
}

// ---- additional checks of every predicate ----------------------------------

TEST(UnicodeCategories, Letters) {
  EXPECT_TRUE(IsLetterUppercase(U'A'));
  EXPECT_TRUE(IsLetterUppercase(U'\u00C9'));
  EXPECT_FALSE(IsLetterUppercase(U'a'));
  EXPECT_TRUE(IsLetterTitlecase(U'\u01C5'));  // 'ǅ'
  EXPECT_FALSE(IsLetterTitlecase(U'\u01C4'));
  EXPECT_TRUE(IsLetterOther(U'\u05D0'));      // 'א'
  EXPECT_TRUE(IsLetterOther(U'\uAC00'));      // Hangul syllable range
  EXPECT_TRUE(IsLetterOther(U'\U00020000'));  // CJK Extension B range
  EXPECT_FALSE(IsLetterOther(U'a'));
  EXPECT_TRUE(IsLetter(U'a'));
  EXPECT_TRUE(IsLetter(U'\u02E2'));
  EXPECT_FALSE(IsLetter(U'1'));
  EXPECT_FALSE(IsLetter(U' '));
}

TEST(UnicodeCategories, Marks) {
  EXPECT_TRUE(IsMarkNonspacing(U'\u0301'));
  EXPECT_FALSE(IsMarkNonspacing(U'\u0903'));
  EXPECT_TRUE(IsMarkSpacingCombining(U'\u0903'));
  EXPECT_TRUE(IsMarkEnclosing(U'\u20DD'));
  EXPECT_FALSE(IsMarkEnclosing(U'\u0301'));
  EXPECT_TRUE(IsMark(U'\u0301'));
  EXPECT_TRUE(IsMark(U'\u0903'));
  EXPECT_TRUE(IsMark(U'\u20DD'));
  EXPECT_FALSE(IsMark(U'a'));
}

TEST(UnicodeCategories, Numbers) {
  EXPECT_TRUE(IsNumberDecimalDigit(U'7'));
  EXPECT_TRUE(IsNumberDecimalDigit(U'\u0663'));  // '٣'
  EXPECT_FALSE(IsNumberDecimalDigit(U'\u00BD'));
  EXPECT_TRUE(IsNumberLetter(U'\u216B'));  // 'Ⅻ'
  EXPECT_FALSE(IsNumberLetter(U'7'));
  EXPECT_TRUE(IsNumberOther(U'\u00BD'));  // '½'
  EXPECT_TRUE(IsNumberOther(U'\u00B2'));  // '²'
  EXPECT_FALSE(IsNumberOther(U'7'));
  EXPECT_TRUE(IsNumber(U'7'));
  EXPECT_TRUE(IsNumber(U'\u216B'));
  EXPECT_TRUE(IsNumber(U'\u00BD'));
  EXPECT_FALSE(IsNumber(U'a'));
}

TEST(UnicodeCategories, Punctuation) {
  EXPECT_TRUE(IsPunctuationConnector(U'_'));
  EXPECT_TRUE(IsPunctuationDash(U'-'));
  EXPECT_TRUE(IsPunctuationDash(U'\u2014'));
  EXPECT_TRUE(IsPunctuationOpen(U'('));
  EXPECT_TRUE(IsPunctuationClose(U')'));
  EXPECT_TRUE(IsPunctuationInitialQuote(U'\u00AB'));  // '«'
  EXPECT_TRUE(IsPunctuationFinalQuote(U'\u00BB'));    // '»'
  EXPECT_TRUE(IsPunctuationOther(U'!'));
  EXPECT_TRUE(IsPunctuationOther(U'\u3002'));  // '。'
  EXPECT_FALSE(IsPunctuationOther(U'a'));
  EXPECT_TRUE(IsPunctuation(U'_'));
  EXPECT_TRUE(IsPunctuation(U'-'));
  EXPECT_TRUE(IsPunctuation(U'('));
  EXPECT_TRUE(IsPunctuation(U')'));
  EXPECT_TRUE(IsPunctuation(U'\u00AB'));
  EXPECT_TRUE(IsPunctuation(U'\u00BB'));
  EXPECT_TRUE(IsPunctuation(U'!'));
  EXPECT_TRUE(IsPunctuation(U'\u3002'));
  // Symbols are not punctuation (`$`, `+`, `^` are Sc, Sm, Sk).
  EXPECT_FALSE(IsPunctuation(U'$'));
  EXPECT_FALSE(IsPunctuation(U'+'));
  EXPECT_FALSE(IsPunctuation(U'^'));
  EXPECT_FALSE(IsPunctuation(U'a'));
}

TEST(UnicodeCategories, Symbols) {
  EXPECT_TRUE(IsSymbolCurrency(U'$'));
  EXPECT_TRUE(IsSymbolCurrency(U'\u20AC'));
  EXPECT_TRUE(IsSymbolModifier(U'^'));
  EXPECT_TRUE(IsSymbolMath(U'+'));
  EXPECT_TRUE(IsSymbolMath(U'\u2200'));
  EXPECT_TRUE(IsSymbolOther(U'\u00A9'));
  EXPECT_TRUE(IsSymbolOther(U'\U0001F600'));
  EXPECT_FALSE(IsSymbolOther(U'a'));
  EXPECT_TRUE(IsSymbol(U'$'));
  EXPECT_TRUE(IsSymbol(U'^'));
  EXPECT_TRUE(IsSymbol(U'+'));
  EXPECT_TRUE(IsSymbol(U'\u00A9'));
  EXPECT_FALSE(IsSymbol(U'!'));
}

TEST(UnicodeCategories, Separators) {
  EXPECT_TRUE(IsSeparatorSpace(U' '));
  EXPECT_TRUE(IsSeparatorSpace(U'\u00A0'));
  EXPECT_TRUE(IsSeparatorSpace(U'\u3000'));
  EXPECT_FALSE(IsSeparatorSpace(U'\n'));
  EXPECT_TRUE(IsSeparatorLine(U'\u2028'));
  EXPECT_TRUE(IsSeparatorParagraph(U'\u2029'));
  EXPECT_TRUE(IsSeparator(U' '));
  EXPECT_TRUE(IsSeparator(U'\u2028'));
  EXPECT_TRUE(IsSeparator(U'\u2029'));
  EXPECT_FALSE(IsSeparator(U'\t'));
}

TEST(UnicodeCategories, Other) {
  EXPECT_TRUE(IsOther(U'\0'));
  EXPECT_TRUE(IsOther(U'\t'));
  EXPECT_TRUE(IsOther(U'\u0601'));
  EXPECT_TRUE(IsOther(U'\u200B'));  // ZERO WIDTH SPACE is Cf
  EXPECT_TRUE(IsOther(U'\uE000'));
  EXPECT_TRUE(IsOther(U'\U0010FFFD'));
  EXPECT_FALSE(IsOther(U'a'));
  EXPECT_FALSE(IsOther(U' '));
  EXPECT_TRUE(IsOtherPrivateUse(U'\U000F0000'));
  EXPECT_FALSE(IsOtherPrivateUse(U'\U000FFFFE'));
}

// ---- Rust `char` methods ---------------------------------------------------

TEST(RustChar, IsWhitespace) {
  // The 25 White_Space code points.
  const char32_t kWhiteSpace[] = {
      0x09,   0x0A,   0x0B,   0x0C,   0x0D,   0x20,   0x85,   0xA0,   0x1680,
      0x2000, 0x2001, 0x2002, 0x2003, 0x2004, 0x2005, 0x2006, 0x2007, 0x2008,
      0x2009, 0x200A, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000};
  int count = 0;
  for (uint32_t c = 0; c <= 0x10FFFF; ++c) {
    bool expected = false;
    for (char32_t w : kWhiteSpace) expected |= (w == c);
    ASSERT_EQ(IsWhitespace(static_cast<char32_t>(c)), expected) << c;
    count += expected;
  }
  EXPECT_EQ(count, 25);
  // Not White_Space: ZERO WIDTH SPACE, MONGOLIAN VOWEL SEPARATOR, ZWNBSP.
  EXPECT_FALSE(IsWhitespace(U'\u200B'));
  EXPECT_FALSE(IsWhitespace(U'\u180E'));
  EXPECT_FALSE(IsWhitespace(U'\uFEFF'));
}

TEST(RustChar, IsAlphabetic) {
  EXPECT_TRUE(IsAlphabetic(U'a'));
  EXPECT_TRUE(IsAlphabetic(U'Z'));
  EXPECT_TRUE(IsAlphabetic(U'\u00AA'));  // 'ª' (Lo)
  EXPECT_TRUE(IsAlphabetic(U'\u00E9'));
  EXPECT_TRUE(IsAlphabetic(U'\u216B'));  // 'Ⅻ' (Nl, Alphabetic)
  EXPECT_TRUE(IsAlphabetic(
      U'\u0345'));  // COMBINING GREEK YPOGEGRAMMENI (Other_Alphabetic)
  EXPECT_TRUE(IsAlphabetic(U'\u4E2D'));  // '中'
  EXPECT_TRUE(IsAlphabetic(U'\uAC00'));
  EXPECT_TRUE(IsAlphabetic(U'\U00010400'));
  EXPECT_FALSE(IsAlphabetic(U'1'));
  EXPECT_FALSE(IsAlphabetic(U'_'));
  EXPECT_FALSE(IsAlphabetic(U' '));
  EXPECT_FALSE(IsAlphabetic(U'\u00A9'));
  EXPECT_FALSE(IsAlphabetic(U'\u0301'));
  EXPECT_FALSE(IsAlphabetic(U'\U0001F600'));
}

TEST(RustChar, IsNumeric) {
  EXPECT_TRUE(IsNumeric(U'0'));
  EXPECT_TRUE(IsNumeric(U'9'));
  EXPECT_TRUE(IsNumeric(U'\u00B2'));      // '²' (No)
  EXPECT_TRUE(IsNumeric(U'\u00BD'));      // '½' (No)
  EXPECT_TRUE(IsNumeric(U'\u0663'));      // '٣' (Nd)
  EXPECT_TRUE(IsNumeric(U'\u216B'));      // 'Ⅻ' (Nl)
  EXPECT_TRUE(IsNumeric(U'\uFF11'));      // '１'
  EXPECT_TRUE(IsNumeric(U'\U0001D7CE'));  // MATHEMATICAL BOLD DIGIT ZERO
  EXPECT_FALSE(IsNumeric(U'a'));
  EXPECT_FALSE(IsNumeric(U'\u4E00'));  // '一' is Lo, not N
  EXPECT_FALSE(IsNumeric(U'.'));
  EXPECT_FALSE(IsNumeric(U'\u00B1'));
}

TEST(RustChar, IsAlphanumeric) {
  EXPECT_TRUE(IsAlphanumeric(U'a'));
  EXPECT_TRUE(IsAlphanumeric(U'7'));
  EXPECT_TRUE(IsAlphanumeric(U'\u00E9'));
  EXPECT_TRUE(IsAlphanumeric(U'\u00BD'));
  EXPECT_FALSE(IsAlphanumeric(U'_'));
  EXPECT_FALSE(IsAlphanumeric(U' '));
  EXPECT_FALSE(IsAlphanumeric(U'\u00A9'));
}

TEST(RustChar, IsControl) {
  EXPECT_TRUE(IsControl(U'\0'));
  EXPECT_TRUE(IsControl(U'\t'));
  EXPECT_TRUE(IsControl(U'\u001F'));
  EXPECT_TRUE(IsControl(U'\u007F'));
  EXPECT_TRUE(IsControl(U'\u0085'));
  EXPECT_TRUE(IsControl(U'\u009F'));
  EXPECT_FALSE(IsControl(U' '));
  EXPECT_FALSE(IsControl(U'\u00A0'));
  EXPECT_FALSE(IsControl(U'\u200B'));
}

TEST(RustChar, IsAsciiPunctuation) {
  int count = 0;
  for (uint32_t c = 0; c <= 0x10FFFF; ++c) {
    const bool expected = (0x21 <= c && c <= 0x2F) ||
                          (0x3A <= c && c <= 0x40) ||
                          (0x5B <= c && c <= 0x60) || (0x7B <= c && c <= 0x7E);
    ASSERT_EQ(IsAsciiPunctuation(static_cast<char32_t>(c)), expected) << c;
    count += expected;
  }
  EXPECT_EQ(count, 32);
}

TEST(RustChar, ToLowercase) {
  EXPECT_THAT(ToLowercase(U'A'), ElementsAre(U'a'));
  EXPECT_THAT(ToLowercase(U'a'), ElementsAre(U'a'));
  EXPECT_THAT(ToLowercase(U'1'), ElementsAre(U'1'));
  EXPECT_THAT(ToLowercase(U'\0'), ElementsAre(U'\0'));
  EXPECT_THAT(ToLowercase(U'\u00C9'), ElementsAre(U'\u00E9'));        // 'É'
  EXPECT_THAT(ToLowercase(U'\u00D7'), ElementsAre(U'\u00D7'));        // '×'
  EXPECT_THAT(ToLowercase(U'\u00DF'), ElementsAre(U'\u00DF'));        // 'ß'
  EXPECT_THAT(ToLowercase(U'\u0130'), ElementsAre(U'i', U'\u0307'));  // 'İ'
  EXPECT_THAT(ToLowercase(U'\u0100'),
              ElementsAre(U'\u0101'));  // 'Ā' (step_by_2)
  EXPECT_THAT(ToLowercase(U'\u0101'),
              ElementsAre(U'\u0101'));  // 'ā' (odd one out)
  EXPECT_THAT(ToLowercase(U'\u0178'), ElementsAre(U'\u00FF'));  // 'Ÿ'
  EXPECT_THAT(ToLowercase(U'\u01C4'), ElementsAre(U'\u01C6'));  // 'Ǆ'
  EXPECT_THAT(ToLowercase(U'\u01C5'), ElementsAre(U'\u01C6'));  // 'ǅ'
  EXPECT_THAT(ToLowercase(U'\u03A3'),
              ElementsAre(U'\u03C3'));  // 'Σ' (no final-sigma rule)
  EXPECT_THAT(ToLowercase(U'\u0410'), ElementsAre(U'\u0430'));  // 'А'
  EXPECT_THAT(ToLowercase(U'\u13A0'), ElementsAre(U'\uAB70'));  // Cherokee 'Ꭰ'
  EXPECT_THAT(ToLowercase(U'\u1E9E'), ElementsAre(U'\u00DF'));  // 'ẞ'
  EXPECT_THAT(ToLowercase(U'\u2126'), ElementsAre(U'\u03C9'));  // 'Ω' OHM SIGN
  EXPECT_THAT(ToLowercase(U'\u212A'), ElementsAre(U'k'));       // KELVIN SIGN
  EXPECT_THAT(ToLowercase(U'\u2160'), ElementsAre(U'\u2170'));  // 'Ⅰ'
  EXPECT_THAT(ToLowercase(U'\uFF21'), ElementsAre(U'\uFF41'));  // 'Ａ'
  EXPECT_THAT(ToLowercase(U'\U00010400'),
              ElementsAre(U'\U00010428'));  // Deseret
  EXPECT_THAT(ToLowercase(U'\U0001E900'), ElementsAre(U'\U0001E922'));  // Adlam
  EXPECT_THAT(ToLowercase(U'\u4E2D'), ElementsAre(U'\u4E2D'));          // '中'
  EXPECT_THAT(ToLowercase(U'\U0001F600'), ElementsAre(U'\U0001F600'));
  EXPECT_THAT(ToLowercase(U'\U0010FFFF'), ElementsAre(U'\U0010FFFF'));
}

TEST(RustChar, ToUppercase) {
  EXPECT_THAT(ToUppercase(U'a'), ElementsAre(U'A'));
  EXPECT_THAT(ToUppercase(U'A'), ElementsAre(U'A'));
  EXPECT_THAT(ToUppercase(U'1'), ElementsAre(U'1'));
  EXPECT_THAT(ToUppercase(U'\0'), ElementsAre(U'\0'));
  EXPECT_THAT(ToUppercase(U'\u00B5'), ElementsAre(U'\u039C'));        // 'µ'
  EXPECT_THAT(ToUppercase(U'\u00DF'), ElementsAre(U'S', U'S'));       // 'ß'
  EXPECT_THAT(ToUppercase(U'\u00E9'), ElementsAre(U'\u00C9'));        // 'é'
  EXPECT_THAT(ToUppercase(U'\u00F7'), ElementsAre(U'\u00F7'));        // '÷'
  EXPECT_THAT(ToUppercase(U'\u00FF'), ElementsAre(U'\u0178'));        // 'ÿ'
  EXPECT_THAT(ToUppercase(U'\u0101'), ElementsAre(U'\u0100'));        // 'ā'
  EXPECT_THAT(ToUppercase(U'\u0131'), ElementsAre(U'I'));             // 'ı'
  EXPECT_THAT(ToUppercase(U'\u0149'), ElementsAre(U'\u02BC', U'N'));  // 'ŉ'
  EXPECT_THAT(ToUppercase(U'\u017F'), ElementsAre(U'S'));             // 'ſ'
  EXPECT_THAT(ToUppercase(U'\u01C5'), ElementsAre(U'\u01C4'));        // 'ǅ'
  EXPECT_THAT(ToUppercase(U'\u01C6'), ElementsAre(U'\u01C4'));        // 'ǆ'
  EXPECT_THAT(ToUppercase(U'\u01F0'), ElementsAre(U'J', U'\u030C'));  // 'ǰ'
  EXPECT_THAT(ToUppercase(U'\u0390'),
              ElementsAre(U'\u0399', U'\u0308', U'\u0301'));    // 'ΐ'
  EXPECT_THAT(ToUppercase(U'\u03C2'), ElementsAre(U'\u03A3'));  // 'ς'
  EXPECT_THAT(ToUppercase(U'\u0587'),
              ElementsAre(U'\u0535', U'\u0552'));               // 'և'
  EXPECT_THAT(ToUppercase(U'\u1E9B'), ElementsAre(U'\u1E60'));  // 'ẛ'
  EXPECT_THAT(ToUppercase(U'\u1F80'),
              ElementsAre(U'\u1F08', U'\u0399'));  // 'ᾀ'
  EXPECT_THAT(ToUppercase(U'\u1FB7'),
              ElementsAre(U'\u0391', U'\u0342', U'\u0399'));    // 'ᾷ'
  EXPECT_THAT(ToUppercase(U'\u2170'), ElementsAre(U'\u2160'));  // 'ⅰ'
  EXPECT_THAT(ToUppercase(U'\uAB70'),
              ElementsAre(U'\u13A0'));  // Cherokee small
  EXPECT_THAT(ToUppercase(U'\uFB03'), ElementsAre(U'F', U'F', U'I'));  // 'ﬃ'
  EXPECT_THAT(ToUppercase(U'\uFF41'), ElementsAre(U'\uFF21'));         // 'ａ'
  EXPECT_THAT(ToUppercase(U'\U00010428'),
              ElementsAre(U'\U00010400'));  // Deseret
  EXPECT_THAT(ToUppercase(U'\U0001E922'), ElementsAre(U'\U0001E900'));  // Adlam
  EXPECT_THAT(ToUppercase(U'\u4E2D'), ElementsAre(U'\u4E2D'));          // '中'
  EXPECT_THAT(ToUppercase(U'\U0010FFFF'), ElementsAre(U'\U0010FFFF'));
}

TEST(RustChar, CaseMappingRoundTrips) {
  // Every simple (1:1) mapping of ASCII and Latin-1 round-trips.
  for (char32_t c = U'a'; c <= U'z'; ++c) {
    ASSERT_THAT(ToLowercase(ToUppercase(c)[0]), ElementsAre(c));
  }
  for (char32_t c = U'\u00E0'; c <= U'\u00FE'; ++c) {
    if (c == U'\u00F7') continue;
    ASSERT_THAT(ToLowercase(ToUppercase(c)[0]), ElementsAre(c))
        << static_cast<uint32_t>(c);
  }
  // Mappings never produce more than three code points and never '\0' padding.
  for (uint32_t c = 0x80; c <= 0x10FFFF; ++c) {
    if (0xD800 <= c && c <= 0xDFFF) continue;
    const CaseMapping lower = ToLowercase(static_cast<char32_t>(c));
    const CaseMapping upper = ToUppercase(static_cast<char32_t>(c));
    ASSERT_GE(lower.size(), 1u) << c;
    ASSERT_LE(lower.size(), 3u) << c;
    ASSERT_GE(upper.size(), 1u) << c;
    ASSERT_LE(upper.size(), 3u) << c;
    for (char32_t l : lower) ASSERT_NE(l, U'\0') << c;
    for (char32_t u : upper) ASSERT_NE(u, U'\0') << c;
  }
}

}  // namespace
}  // namespace tokenizers::unicode
