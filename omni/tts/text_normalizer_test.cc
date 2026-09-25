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

#include "omni/tts/text_normalizer.h"

#include <fstream>
#include <ios>
#include <iterator>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace litert::omni::tts {
namespace {

using ::testing::HasSubstr;

// A minimal table exercising both substitution forms, used by the tests that
// are about the interpreter rather than about any particular language.
constexpr char kToyTable[] =
    "# comment line\n"
    "\n"
    "digit\t0\tzero\n"
    "digit\t1\tone\n"
    "digit\t2\ttwo\n"
    "digit\t3\tthree\n"
    "digit\t4\tfour\n"
    "digit\t5\tfive\n"
    "digit\t6\tsix\n"
    "digit\t7\tseven\n"
    "digit\t8\teight\n"
    "digit\t9\tnine\n"
    "rule\t(\\d+)%\tpct \\1\n"
    "rule\t\\[(\\d+)\\]\t#1\n";

// Reads the Mandarin rule table that ships with the model, so that the tests
// below cover the data we actually serve rather than a copy of it.
std::string ReadMandarinRuleTable() {
  const std::string base_dir = ::testing::SrcDir();
  const std::vector<std::string> candidate_paths = {
      "omni/tts/data/zh_textnorm.txt",
      absl::StrCat(base_dir, "/",
                   "odml/litert_lm/omni/tts/data/zh_textnorm.txt"),
      absl::StrCat(base_dir,
                   "/litert_lm/omni/tts/data/"
                   "zh_textnorm.txt"),
  };
  for (const std::string& path : candidate_paths) {
    std::ifstream file(path, std::ios::binary);
    if (!file.good()) continue;
    return std::string(std::istreambuf_iterator<char>(file),
                       std::istreambuf_iterator<char>());
  }
  return "";
}

TEST(TextNormalizerTest, EmptyTableLeavesTextAlone) {
  ASSERT_OK_AND_ASSIGN(auto normalizer, TextNormalizer::Create(""));
  EXPECT_EQ(normalizer->RuleCount(), 0);
  EXPECT_EQ(normalizer->Normalize("2026年 3.14"), "2026年 3.14");
}

TEST(TextNormalizerTest, SubstitutesGroupsVerbatimAndDigitByDigit) {
  ASSERT_OK_AND_ASSIGN(auto normalizer, TextNormalizer::Create(kToyTable));
  EXPECT_EQ(normalizer->RuleCount(), 2);
  EXPECT_EQ(normalizer->Normalize("50%"), "pct 50");
  EXPECT_EQ(normalizer->Normalize("[2026]"), "twozerotwosix");
  // Every match is rewritten, and text around them is preserved.
  EXPECT_EQ(normalizer->Normalize("a 1% b 2% c"), "a pct 1 b pct 2 c");
}

TEST(TextNormalizerTest, HonorsEscapes) {
  ASSERT_OK_AND_ASSIGN(
      auto normalizer,
      TextNormalizer::Create("rule\t(a)\t\\\\ \\# \\@ \\1 literal#\n"));
  EXPECT_EQ(normalizer->Normalize("a"), "\\ # @ a literal#");
}

TEST(TextNormalizerTest, AppliesRulesInTableOrder) {
  // The second rule sees the output of the first, which is what lets a table
  // layer rewrites.
  ASSERT_OK_AND_ASSIGN(auto normalizer,
                       TextNormalizer::Create("rule\tone\ttwo\n"
                                              "rule\ttwo\tthree\n"));
  EXPECT_EQ(normalizer->Normalize("one"), "three");
}

TEST(TextNormalizerTest, EmptyMatchDoesNotHang) {
  // `\d*` matches the empty string at every position. The scan has to make
  // forward progress anyway, and must not split a multi-byte character.
  ASSERT_OK_AND_ASSIGN(auto normalizer,
                       TextNormalizer::Create("rule\t(\\d*)\t[\\1]\n"));
  EXPECT_EQ(normalizer->Normalize("a1"), "[]a[1][]");
  EXPECT_EQ(normalizer->Normalize("中"), "[]中[]");
}

TEST(TextNormalizerTest, RejectsMalformedTables) {
  EXPECT_THAT(
      TextNormalizer::Create("bogus\tx\ty\n").status(),
      ::testing::AllOf(::testing::Property(&absl::Status::code,
                                           absl::StatusCode::kInvalidArgument),
                       ::testing::Property(&absl::Status::message,
                                           HasSubstr("unknown record type"))));
  EXPECT_THAT(
      TextNormalizer::Create("rule\tonly_one_field\n").status(),
      ::testing::Property(&absl::Status::message, HasSubstr("`rule` takes 2")));
  EXPECT_THAT(TextNormalizer::Create("digit\txx\tzero\n").status(),
              ::testing::Property(&absl::Status::message,
                                  HasSubstr("single ASCII digit")));
  EXPECT_THAT(
      TextNormalizer::Create("digit\t0\ta\ndigit\t0\tb\n").status(),
      ::testing::Property(&absl::Status::message, HasSubstr("defined twice")));
  EXPECT_THAT(
      TextNormalizer::Create("unit\t十\t百\t千\n").status(),
      ::testing::Property(&absl::Status::message, HasSubstr("`unit` takes 5")));
  EXPECT_THAT(TextNormalizer::Create("rule\t(unclosed\tx\n").status(),
              ::testing::Property(&absl::Status::message,
                                  HasSubstr("cannot compile pattern")));
  // The line number makes a bad table findable.
  EXPECT_THAT(TextNormalizer::Create("# ok\nrule\ta\tb\nbogus\n").status(),
              ::testing::Property(&absl::Status::message, HasSubstr("line 3")));
}

TEST(TextNormalizerTest, RejectsDigitExpansionWithoutADigitTable) {
  EXPECT_THAT(TextNormalizer::Create("rule\t(\\d+)\t#1\n").status(),
              ::testing::Property(&absl::Status::message,
                                  HasSubstr("defines 0 of the 10 digits")));
  // An escaped '#' is a literal, not an expansion, so it needs no digits.
  EXPECT_OK(TextNormalizer::Create("rule\t(\\d+)\t\\#1\n"));
}

TEST(TextNormalizerTest, RejectsCardinalExpansionWithoutUnitRecord) {
  std::string digits_only =
      "digit\t0\t零\ndigit\t1\t一\ndigit\t2\t二\ndigit\t3\t三\ndigit\t4\t四\n"
      "digit\t5\t五\ndigit\t6\t六\ndigit\t7\t七\ndigit\t8\t八\ndigit\t9\t九\n"
      "rule\t(\\d+)\t@1\n";
  EXPECT_THAT(TextNormalizer::Create(digits_only).status(),
              ::testing::Property(&absl::Status::message,
                                  HasSubstr("defines no `unit` record")));
}

TEST(TextNormalizerTest, MandarinTableParses) {
  const std::string table = ReadMandarinRuleTable();
  ASSERT_FALSE(table.empty()) << "could not locate zh_textnorm.txt";
  ASSERT_OK_AND_ASSIGN(auto normalizer, TextNormalizer::Create(table));
  EXPECT_GT(normalizer->RuleCount(), 0);
}

TEST(TextNormalizerTest, MandarinTableNormalizesYearsDatesAndDecimals) {
  const std::string table = ReadMandarinRuleTable();
  ASSERT_FALSE(table.empty()) << "could not locate zh_textnorm.txt";
  ASSERT_OK_AND_ASSIGN(auto normalizer, TextNormalizer::Create(table));

  // Four-digit years are read digit-by-digit using 零 (U+96F6), while months
  // and days are read as cardinals.
  EXPECT_EQ(normalizer->Normalize("2026年"), "二零二六年");
  EXPECT_EQ(normalizer->Normalize("二〇二六年"), "二零二六年");
  EXPECT_EQ(normalizer->Normalize("2026年1月1日"), "二零二六年一月一日");
  EXPECT_EQ(normalizer->Normalize("2026年9月24號"), "二零二六年九月二十四號");
  EXPECT_EQ(normalizer->Normalize("今天是2026年9月24日"),
            "今天是二零二六年九月二十四日");

  // Decimals: cardinal integer part + 点 + digit-by-digit fractional part.
  EXPECT_EQ(normalizer->Normalize("3.14"), "三点一四");
  EXPECT_EQ(normalizer->Normalize("12.34"), "十二点三四");
  EXPECT_EQ(normalizer->Normalize("0.05"), "零点零五");

  // Percentages and fractions.
  EXPECT_EQ(normalizer->Normalize("50%"), "百分之五十");
  EXPECT_EQ(normalizer->Normalize("3.14%"), "百分之三点一四");
  EXPECT_EQ(normalizer->Normalize("3/4"), "四分之三");

  // Temperatures, angles, and currencies.
  EXPECT_EQ(normalizer->Normalize("25℃"), "二十五摄氏度");
  EXPECT_EQ(normalizer->Normalize("25°C"), "二十五摄氏度");
  EXPECT_EQ(normalizer->Normalize("25 °C"), "二十五摄氏度");
  EXPECT_EQ(normalizer->Normalize("-5°C"), "负五摄氏度");
  EXPECT_EQ(normalizer->Normalize("单位：°C"), "单位：摄氏度");
  EXPECT_EQ(normalizer->Normalize("室温(°F)"), "室温(华氏度)");
  EXPECT_EQ(normalizer->Normalize("90°"), "九十度");
  EXPECT_EQ(normalizer->Normalize("￥100"), "一百元");
  EXPECT_EQ(normalizer->Normalize("$100"), "一百美元");
}

TEST(TextNormalizerTest, MandarinTableFormatsCardinalsInChinese) {
  const std::string table = ReadMandarinRuleTable();
  ASSERT_FALSE(table.empty()) << "could not locate zh_textnorm.txt";
  ASSERT_OK_AND_ASSIGN(auto normalizer, TextNormalizer::Create(table));

  // Five-or-more-digit numbers before 年 are duration cardinals, not calendar
  // years, so they are read as cardinals (一万年, 五万年) rather than
  // digit-by-digit.
  EXPECT_EQ(normalizer->Normalize("10000年"), "一万年");
  EXPECT_EQ(normalizer->Normalize("50000年"), "五万年");
  EXPECT_EQ(normalizer->Normalize("一共有123个"),
            "一共有\xe4\xb8\x80百二十三个");
  EXPECT_EQ(normalizer->Normalize("下午3点30分"), "下午三点三十分");
  EXPECT_EQ(normalizer->Normalize("第1名"), "第一名");
  EXPECT_EQ(normalizer->Normalize("他有5个苹果"), "他有五个苹果");
  EXPECT_EQ(normalizer->Normalize("105"), "一百零五");
  EXPECT_EQ(normalizer->Normalize("10005"), "一万零五");
  // Text with nothing to normalize is returned untouched.
  EXPECT_EQ(normalizer->Normalize("你好，世界"), "你好，世界");
}

}  // namespace
}  // namespace litert::omni::tts
