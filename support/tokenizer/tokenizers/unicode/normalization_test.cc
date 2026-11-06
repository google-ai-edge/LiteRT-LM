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
// Tests of the C++ port of `unicode_normalization_alignments` /
// `unicode-normalization`: the crate's own tests (`src/test.rs`,
// `src/normalize.rs`), alignment-delta checks, and the Unicode conformance
// test file NormalizationTest.txt.

#include "support/tokenizer/tokenizers/unicode/normalization.h"

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_split.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/unicode/normalization_test_data.h"
#include "support/tokenizer/tokenizers/unicode/utf8_chars.h"

namespace tokenizers::unicode {
namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::Pair;

// `.collect::<String>()` of the characters of a normalization result.
std::string ToString(const std::vector<std::pair<char32_t, int64_t>>& chars) {
  std::string out;
  for (const auto& [c, delta] : chars) {
    internal::AppendUtf8(c, &out);
  }
  return out;
}

std::string NfdString(absl::string_view s) { return ToString(Nfd(s)); }
std::string NfkdString(absl::string_view s) { return ToString(Nfkd(s)); }
std::string NfcString(absl::string_view s) { return ToString(Nfc(s)); }
std::string NfkcString(absl::string_view s) { return ToString(Nfkc(s)); }

// ---- unicode-normalization/src/test.rs -------------------------------------

TEST(UnicodeNormalization, TestNfd) {
  const auto t = [](absl::string_view input, absl::string_view expected) {
    EXPECT_EQ(NfdString(input), expected) << input;
  };
  t("abc", "abc");
  t("\u1e0b\u01c4", "d\u0307\u01c4");
  t("\u2026", "\u2026");
  t("\u2126", "\u03a9");
  t("\u1e0b\u0323", "d\u0323\u0307");
  t("\u1e0d\u0307", "d\u0323\u0307");
  t("a\u0301", "a\u0301");
  t("\u0301a", "\u0301a");
  t("\ud4db", "\u1111\u1171\u11b6");
  t("\uac1c", "\u1100\u1162");
}

TEST(UnicodeNormalization, TestNfkd) {
  const auto t = [](absl::string_view input, absl::string_view expected) {
    EXPECT_EQ(NfkdString(input), expected) << input;
  };
  t("abc", "abc");
  t("\u1e0b\u01c4", "d\u0307DZ\u030c");
  t("\u2026", "...");
  t("\u2126", "\u03a9");
  t("\u1e0b\u0323", "d\u0323\u0307");
  t("\u1e0d\u0307", "d\u0323\u0307");
  t("a\u0301", "a\u0301");
  t("\u0301a", "\u0301a");
  t("\ud4db", "\u1111\u1171\u11b6");
  t("\uac1c", "\u1100\u1162");
}

TEST(UnicodeNormalization, TestNfc) {
  const auto t = [](absl::string_view input, absl::string_view expected) {
    EXPECT_EQ(NfcString(input), expected) << input;
  };
  t("abc", "abc");
  t("\u1e0b\u01c4", "\u1e0b\u01c4");
  t("\u2026", "\u2026");
  t("\u2126", "\u03a9");
  t("\u1e0b\u0323", "\u1e0d\u0307");
  t("\u1e0d\u0307", "\u1e0d\u0307");
  t("a\u0301", "\u00e1");
  t("\u0301a", "\u0301a");
  t("\ud4db", "\ud4db");
  t("\uac1c", "\uac1c");
  t("a\u0300\u0305\u0315\u05aeb", "\u00e0\u05ae\u0305\u0315b");
}

TEST(UnicodeNormalization, TestNfkc) {
  const auto t = [](absl::string_view input, absl::string_view expected) {
    EXPECT_EQ(NfkcString(input), expected) << input;
  };
  t("abc", "abc");
  t("\u1e0b\u01c4", "\u1e0bD\u017d");
  t("\u2026", "...");
  t("\u2126", "\u03a9");
  t("\u1e0b\u0323", "\u1e0d\u0307");
  t("\u1e0d\u0307", "\u1e0d\u0307");
  t("a\u0301", "\u00e1");
  t("\u0301a", "\u0301a");
  t("\ud4db", "\ud4db");
  t("\uac1c", "\uac1c");
  t("a\u0300\u0305\u0315\u05aeb", "\u00e0\u05ae\u0305\u0315b");
}

TEST(UnicodeNormalization, TestNormalizeChar) {
  // `'\u{2126}'.nfd().to_string()`
  EXPECT_EQ(NfdString("\u2126"), "\u03a9");
}

TEST(UnicodeNormalization, TestIsCombiningMarkAscii) {
  for (uint32_t cp = 0; cp < 0x7f; ++cp) {
    EXPECT_FALSE(IsCombiningMark(static_cast<char32_t>(cp))) << cp;
  }
}

TEST(UnicodeNormalization, TestIsCombiningMarkMisc) {
  // https://github.com/unicode-rs/unicode-normalization/issues/16
  // U+11C3A BHAIKSUKI VOWEL SIGN O
  // Category: Mark, Nonspacing [Mn]
  EXPECT_TRUE(IsCombiningMark(U'\U00011C3A'));

  // U+11C3F BHAIKSUKI SIGN VIRAMA
  // Category: Mark, Nonspacing [Mn]
  EXPECT_TRUE(IsCombiningMark(U'\U00011C3F'));
}

// ---- unicode-normalization/src/normalize.rs --------------------------------

// Regression test from a bugfix where we were composing an LV_Syllable with
// T_BASE directly. (We should only compose an LV_Syllable with a character
// in the range `T_BASE + 1 ..= T_LAST`.)
TEST(UnicodeNormalization, TestHangulComposition) {
  EXPECT_EQ(Compose(U'\uc8e0', U'\u11a7'), std::nullopt);
}

// ---- additional checks of the lookups ------------------------------------

TEST(UnicodeNormalization, CanonicalCombiningClass) {
  EXPECT_EQ(CanonicalCombiningClass(U'a'), 0);
  EXPECT_EQ(CanonicalCombiningClass(U'\u0301'), 230);
  EXPECT_EQ(CanonicalCombiningClass(U'\u0323'), 220);
  EXPECT_EQ(CanonicalCombiningClass(U'\u0334'), 1);
  EXPECT_EQ(CanonicalCombiningClass(U'\u3099'), 8);
  EXPECT_EQ(CanonicalCombiningClass(U'\u05ae'), 228);
  EXPECT_EQ(CanonicalCombiningClass(U'\u0f71'), 129);
  EXPECT_EQ(CanonicalCombiningClass(U'\U0001D165'), 216);
  EXPECT_EQ(CanonicalCombiningClass(U'\uac00'), 0);
  EXPECT_EQ(CanonicalCombiningClass(U'\U0010FFFF'), 0);
}

TEST(UnicodeNormalization, IsCombiningMark) {
  EXPECT_TRUE(IsCombiningMark(U'\u0301'));  // Mn
  EXPECT_TRUE(IsCombiningMark(U'\u0903'));  // Mc
  EXPECT_TRUE(IsCombiningMark(U'\u20dd'));  // Me
  EXPECT_FALSE(IsCombiningMark(U'a'));
  EXPECT_FALSE(IsCombiningMark(U'\u00e9'));
  EXPECT_FALSE(IsCombiningMark(U'\U0010FFFF'));
}

TEST(UnicodeNormalization, Compose) {
  EXPECT_EQ(Compose(U'A', U'\u030a'), U'\u00c5');
  EXPECT_EQ(Compose(U'A', U'\u0301'), U'\u00c1');
  EXPECT_EQ(Compose(U'\u1100', U'\u1161'), U'\uac00');
  EXPECT_EQ(Compose(U'\uac00', U'\u11a8'), U'\uac01');
  // composition_table_astral
  EXPECT_EQ(Compose(U'\U00011099', U'\U000110ba'), U'\U0001109a');
  EXPECT_EQ(Compose(U'\U00011131', U'\U00011127'), U'\U0001112e');
  EXPECT_EQ(Compose(U'a', U'b'), std::nullopt);
  EXPECT_EQ(Compose(U'\U00011099', U'a'), std::nullopt);
  // Composition exclusions: U+0958 decomposes to U+0915 U+093C but does not
  // compose back.
  EXPECT_EQ(Compose(U'\u0915', U'\u093c'), std::nullopt);
}

TEST(UnicodeNormalization, DecomposeCanonical) {
  const auto decompose = [](char32_t c) {
    std::vector<char32_t> out;
    DecomposeCanonical(c, [&](char32_t d) { out.push_back(d); });
    return out;
  };
  EXPECT_THAT(decompose(U'a'), ElementsAre(U'a'));
  EXPECT_THAT(decompose(U'\u00c5'), ElementsAre(U'A', U'\u030a'));
  EXPECT_THAT(decompose(U'\u2126'), ElementsAre(U'\u03a9'));
  EXPECT_THAT(decompose(U'\u1e0b'), ElementsAre(U'd', U'\u0307'));
  EXPECT_THAT(decompose(U'\u1e69'), ElementsAre(U's', U'\u0323', U'\u0307'));
  EXPECT_THAT(decompose(U'\uac00'), ElementsAre(U'\u1100', U'\u1161'));
  EXPECT_THAT(decompose(U'\ud4db'),
              ElementsAre(U'\u1111', U'\u1171', U'\u11b6'));
  EXPECT_THAT(decompose(U'\ufb01'), ElementsAre(U'\ufb01'));
  EXPECT_THAT(decompose(U'\U0001D15E'),
              ElementsAre(U'\U0001D157', U'\U0001D165'));
  EXPECT_THAT(decompose(U'\U0002F800'), ElementsAre(U'\u4e3d'));
}

TEST(UnicodeNormalization, DecomposeCompatible) {
  const auto decompose = [](char32_t c) {
    std::vector<char32_t> out;
    DecomposeCompatible(c, [&](char32_t d) { out.push_back(d); });
    return out;
  };
  EXPECT_THAT(decompose(U'a'), ElementsAre(U'a'));
  EXPECT_THAT(decompose(U'\u00c5'), ElementsAre(U'A', U'\u030a'));
  EXPECT_THAT(decompose(U'\ufb01'), ElementsAre(U'f', U'i'));
  EXPECT_THAT(decompose(U'\u2026'), ElementsAre(U'.', U'.', U'.'));
  EXPECT_THAT(decompose(U'\u01c4'), ElementsAre(U'D', U'Z', U'\u030c'));
  EXPECT_THAT(decompose(U'\u3392'), ElementsAre(U'M', U'H', U'z'));
  EXPECT_THAT(decompose(U'\u3336'), ElementsAre(U'\u30d8', U'\u30af', U'\u30bf',
                                                U'\u30fc', U'\u30eb'));
  EXPECT_THAT(decompose(U'\uac00'), ElementsAre(U'\u1100', U'\u1161'));
}

// ---- alignment deltas (`unicode_normalization_alignments`) -----------------

TEST(UnicodeNormalizationAlignments, DecompositionDeltas) {
  // The first character produced by an original character carries 0, every
  // further one carries 1.
  EXPECT_THAT(Nfd("abc"),
              ElementsAre(Pair(U'a', 0), Pair(U'b', 0), Pair(U'c', 0)));
  EXPECT_THAT(Nfd("\u00e9"), ElementsAre(Pair(U'e', 0), Pair(U'\u0301', 1)));
  EXPECT_THAT(Nfd("\u1e69"), ElementsAre(Pair(U's', 0), Pair(U'\u0323', 1),
                                         Pair(U'\u0307', 1)));
  EXPECT_THAT(Nfd("\ud4db"), ElementsAre(Pair(U'\u1111', 0), Pair(U'\u1171', 1),
                                         Pair(U'\u11b6', 1)));
  EXPECT_THAT(Nfkd("\ufb01"), ElementsAre(Pair(U'f', 0), Pair(U'i', 1)));
  EXPECT_THAT(Nfkd("\u2026"),
              ElementsAre(Pair(U'.', 0), Pair(U'.', 1), Pair(U'.', 1)));
  // Canonical reordering keeps the deltas attached to their characters:
  // "\u1e0b\u0323" -> d(0) \u0307(1) \u0323(0) -> sorted by ccc: d \u0323
  // \u0307.
  EXPECT_THAT(
      Nfd("\u1e0b\u0323"),
      ElementsAre(Pair(U'd', 0), Pair(U'\u0323', 0), Pair(U'\u0307', 1)));
  EXPECT_THAT(Nfd(""), IsEmpty());
}

TEST(UnicodeNormalizationAlignments, CompositionDeltas) {
  // Composing k and ch gives `k.1 + change - 1`.
  EXPECT_THAT(Nfc("e\u0301"), ElementsAre(Pair(U'\u00e9', -1)));
  EXPECT_THAT(Nfc("\u00e9"), ElementsAre(Pair(U'\u00e9', 0)));
  EXPECT_THAT(Nfc("\u1111\u1171\u11b6"), ElementsAre(Pair(U'\ud4db', -2)));
  EXPECT_THAT(Nfc("\ud4db"), ElementsAre(Pair(U'\ud4db', 0)));
  EXPECT_THAT(
      Nfc("a\u0300\u0305\u0315\u05aeb"),
      ElementsAre(Pair(U'\u00e0', -1), Pair(U'\u05ae', 0), Pair(U'\u0305', 0),
                  Pair(U'\u0315', 0), Pair(U'b', 0)));
  // A starter that does not compose is emitted with its own delta, the
  // blocked combining marks follow in canonical order.
  EXPECT_THAT(Nfc("\u0301a"), ElementsAre(Pair(U'\u0301', 0), Pair(U'a', 0)));
  EXPECT_THAT(Nfkc("\ufb01"), ElementsAre(Pair(U'f', 0), Pair(U'i', 1)));
  // "\u01c4" -> D(0) Z(1) \u030c(1); Z and \u030c compose to \u017d with
  // delta 1 + 1 - 1 = 1.
  EXPECT_THAT(
      Nfkc("\u1e0b\u01c4"),
      ElementsAre(Pair(U'\u1e0b', 0), Pair(U'D', 0), Pair(U'\u017d', 1)));
  EXPECT_THAT(Nfc(""), IsEmpty());
}

// ---- Unicode conformance test (NormalizationTest.txt) ----------------------

void CheckConformance(absl::string_view part, const std::string& c1,
                      const std::string& c2, const std::string& c3,
                      const std::string& c4, const std::string& c5) {
  SCOPED_TRACE(absl::StrCat(part, " source=", c1));
  // NFC
  EXPECT_EQ(NfcString(c1), c2);
  EXPECT_EQ(NfcString(c2), c2);
  EXPECT_EQ(NfcString(c3), c2);
  EXPECT_EQ(NfcString(c4), c4);
  EXPECT_EQ(NfcString(c5), c4);
  // NFD
  EXPECT_EQ(NfdString(c1), c3);
  EXPECT_EQ(NfdString(c2), c3);
  EXPECT_EQ(NfdString(c3), c3);
  EXPECT_EQ(NfdString(c4), c5);
  EXPECT_EQ(NfdString(c5), c5);
  // NFKC
  EXPECT_EQ(NfkcString(c1), c4);
  EXPECT_EQ(NfkcString(c2), c4);
  EXPECT_EQ(NfkcString(c3), c4);
  EXPECT_EQ(NfkcString(c4), c4);
  EXPECT_EQ(NfkcString(c5), c4);
  // NFKD
  EXPECT_EQ(NfkdString(c1), c5);
  EXPECT_EQ(NfkdString(c2), c5);
  EXPECT_EQ(NfkdString(c3), c5);
  EXPECT_EQ(NfkdString(c4), c5);
  EXPECT_EQ(NfkdString(c5), c5);
}

TEST(UnicodeNormalization, ConformanceSample) {
  const std::vector<NormalizationTestCase>& cases = NormalizationTestCases();
  ASSERT_GT(cases.size(), 500);
  for (const NormalizationTestCase& c : cases) {
    CheckConformance(c.part, c.source, c.nfc, c.nfd, c.nfkc, c.nfkd);
  }
}

// Decodes a column of NormalizationTest.txt ("0041 0301") into UTF-8.
std::string DecodeColumn(absl::string_view column) {
  std::string out;
  for (absl::string_view cp :
       absl::StrSplit(column, ' ', absl::SkipWhitespace())) {
    internal::AppendUtf8(static_cast<char32_t>(std::strtoul(
                             std::string(cp).c_str(), nullptr, 16)),
                         &out);
  }
  return out;
}

// Runs the whole conformance file if it is available as test data.
TEST(UnicodeNormalization, ConformanceFull) {
  const char* srcdir = std::getenv("TEST_SRCDIR");
  if (srcdir == nullptr) {
    GTEST_SKIP() << "TEST_SRCDIR not set";
  }
  const std::string path =
      absl::StrCat(srcdir,
                   "/third_party/unicode/ucd/NormalizationTest.txt");
  std::ifstream file(path);
  if (!file.is_open()) {
    GTEST_SKIP() << "NormalizationTest.txt not available at " << path;
  }
  std::string line;
  std::string part = "?";
  int count = 0;
  while (std::getline(file, line)) {
    if (line.rfind("@Part", 0) == 0) {
      part = line.substr(0, line.find(' '));
      continue;
    }
    if (line.empty() || line[0] == '#') continue;
    const std::vector<std::string> fields =
        absl::StrSplit(absl::string_view(line).substr(0, line.find('#')), ';');
    ASSERT_GE(fields.size(), 5) << line;
    CheckConformance(part, DecodeColumn(fields[0]), DecodeColumn(fields[1]),
                     DecodeColumn(fields[2]), DecodeColumn(fields[3]),
                     DecodeColumn(fields[4]));
    ++count;
  }
  EXPECT_GT(count, 10000);
}

}  // namespace
}  // namespace tokenizers::unicode
