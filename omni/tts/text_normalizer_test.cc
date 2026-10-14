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
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
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
std::string ReadRuleTable(absl::string_view file_name) {
  const std::string base_dir = ::testing::SrcDir();
  const std::vector<std::string> candidate_paths = {
      absl::StrCat("omni/tts/data/", file_name),
      absl::StrCat(base_dir, "/",
                   "odml/litert_lm/omni/tts/data/",
                   file_name),
      absl::StrCat(base_dir, "/litert_lm/omni/tts/",
                   "data/", file_name),
  };
  for (const std::string& path : candidate_paths) {
    std::ifstream file(path, std::ios::binary);
    if (!file.good()) continue;
    return std::string(std::istreambuf_iterator<char>(file),
                       std::istreambuf_iterator<char>());
  }
  return "";
}

std::string ReadMandarinRuleTable() { return ReadRuleTable("zh_textnorm.txt"); }

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

TEST(TextNormalizerTest, LooksUpMapEntries) {
  ASSERT_OK_AND_ASSIGN(auto normalizer,
                       TextNormalizer::Create("map\tmonth\t05\tmaggio\n"
                                              "map\tmonth\t12\tdicembre\n"
                                              "rule\t(\\d+)/(\\d+)\t\\1 "
                                              "{month:\\2}\n"));
  EXPECT_EQ(normalizer->Normalize("il 12/05 e il 3/12"),
            "il 12 maggio e il 3 dicembre");
  // A key missing from the map is copied through verbatim.
  EXPECT_EQ(normalizer->Normalize("13/13"), "13 13");
}

TEST(TextNormalizerTest, MapValuesMayBeEmptyAndBracesStayLiteral) {
  ASSERT_OK_AND_ASSIGN(auto normalizer, TextNormalizer::Create(
                                            "map\tdrop\tx\t\n"
                                            "rule\t<(\\w)>\t{drop:\\1}{a}\n"));
  EXPECT_EQ(normalizer->Normalize("<x><y>"), "{a}y{a}");
}

TEST(TextNormalizerTest, RejectsUndefinedOrDuplicateMaps) {
  EXPECT_THAT(TextNormalizer::Create("rule\t(a)\t{nope:\\1}\n").status(),
              ::testing::Property(&absl::Status::message,
                                  HasSubstr("undefined map `nope`")));
  EXPECT_THAT(TextNormalizer::Create("map\tm\tk\tv\nmap\tm\tk\tw\n").status(),
              ::testing::Property(&absl::Status::message,
                                  HasSubstr("defines key \"k\" twice")));
  EXPECT_THAT(
      TextNormalizer::Create("map\tm\tk\n").status(),
      ::testing::Property(&absl::Status::message, HasSubstr("`map` takes 3")));
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

class EnglishTableTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const std::string table = ReadRuleTable("en_textnorm.txt");
    ASSERT_FALSE(table.empty()) << "could not locate en_textnorm.txt";
    ASSERT_OK_AND_ASSIGN(normalizer_, TextNormalizer::Create(table));
  }

  std::unique_ptr<TextNormalizer> normalizer_;
};

TEST_F(EnglishTableTest, ReadsFourDigitNumbersAsYears) {
  EXPECT_EQ(normalizer_->Normalize("the 2026 budget"), "the 20 26 budget");
  EXPECT_EQ(normalizer_->Normalize("out of 1500 people"),
            "out of 15 hundred people");
  EXPECT_EQ(normalizer_->Normalize("in 1905."), "in 19 oh 5.");
  EXPECT_EQ(normalizer_->Normalize("in 2005,"), "in two thousand and 5,");
  EXPECT_EQ(normalizer_->Normalize("in 2000"), "in 2 thousand");
  EXPECT_EQ(normalizer_->Normalize("In the 1990s, the"),
            "In the 19 nineties, the");
  // Not years: separators, decimals, times, and longer numbers.
  EXPECT_EQ(normalizer_->Normalize("1,500 people"), "1500 people");
  EXPECT_EQ(normalizer_->Normalize("at 12:30"), "at 12:30");
  EXPECT_EQ(normalizer_->Normalize("12345"), "12345");
}

TEST_F(EnglishTableTest, ReadsCurrencyWithUnits) {
  EXPECT_EQ(normalizer_->Normalize("$1,245.99"), "1245 dollars and 99 cents");
  EXPECT_EQ(normalizer_->Normalize("£45.50, about €53.20."),
            "45 pounds and 50 pence, about 53 euros and 20 cents.");
  EXPECT_EQ(normalizer_->Normalize("$1 or $6.05"),
            "1 dollar or 6 dollars and 5 cents");
}

TEST_F(EnglishTableTest, ReadsDecimalsDigitByDigit) {
  EXPECT_EQ(normalizer_->Normalize("0.05% and 12.5"),
            "0 point 0 5% and 12 point 5");
}

TEST_F(EnglishTableTest, DropsSlashesAndHyphensBetweenDigits) {
  EXPECT_EQ(normalizer_->Normalize("1-800-555-0199"), "1 800 555 0199");
}

TEST_F(EnglishTableTest, ReadsFractionsDatesTimesAndCodes) {
  EXPECT_EQ(normalizer_->Normalize("2 1/2 pounds"), "2 and a half pounds");
  EXPECT_EQ(normalizer_->Normalize("to 3/4x + 1"), "to three quarters x + 1");
  EXPECT_EQ(normalizer_->Normalize("on 08/24/2026 at 09:15"),
            "on August 24th 20 26 at 9:15");
  EXPECT_EQ(normalizer_->Normalize("by 5:00 p.m."), "by 5 PM");
  EXPECT_EQ(normalizer_->Normalize("a 40 km/h zone"),
            "a 40 kilometers per hour zone");
  EXPECT_EQ(normalizer_->Normalize("model ABC-123-XYZ-45."),
            "model ABC  1 2 3 XYZ  4 5.");
  EXPECT_EQ(normalizer_->Normalize("555-0143, ext. 99"),
            "555 0143, extension 99");
}

TEST_F(EnglishTableTest, ExpandsAbbreviations) {
  EXPECT_EQ(normalizer_->Normalize("bring e.g., apples, etc., i.e., fruit"),
            "bring E G, apples, etc, I E, fruit");
  EXPECT_EQ(normalizer_->Normalize("at 8 p.m. or 10 a.m. sharp"),
            "at 8 PM or 10 AM sharp");
  EXPECT_EQ(normalizer_->Normalize("Dr. Jones in Washington, D.C."),
            "Doctor Jones in Washington, DC");
  EXPECT_EQ(normalizer_->Normalize("see p. 23, ASAP"), "see p 23, A-sap");
}

struct TableExample {
  const char* table;
  const char* input;
  const char* expected;
};

class EuropeanTableTest : public ::testing::TestWithParam<TableExample> {};

TEST_P(EuropeanTableTest, Normalizes) {
  const TableExample& example = GetParam();
  const std::string table = ReadRuleTable(example.table);
  ASSERT_FALSE(table.empty()) << "could not locate " << example.table;
  ASSERT_OK_AND_ASSIGN(auto normalizer, TextNormalizer::Create(table));
  EXPECT_EQ(normalizer->Normalize(example.input), example.expected);
}

INSTANTIATE_TEST_SUITE_P(
    Tables, EuropeanTableTest,
    ::testing::Values(
        TableExample{"es_textnorm.txt", "Los EE.UU. y la Sra. Gómez",
                     "Los Estados Unidos y la señora Gómez"},
        TableExample{"es_textnorm.txt", "cuesta €1.500,50 hoy",
                     "cuesta 1500 euros con 50 céntimos hoy"},
        TableExample{"es_textnorm.txt", "a las 15:30 h. o a las 18:00",
                     "a las tres y 30 de la tarde o a las seis de la tarde"},
        TableExample{"es_textnorm.txt", "bajó a -5 °C y subió un 12,5%",
                     "bajó a 5 grados Celsius bajo cero y subió un 12 coma 5 "
                     "por ciento"},
        TableExample{"es_textnorm.txt", "el 12/05/1998 con 3/4 de taza",
                     "el 12 de mayo de 1998 con tres cuartos de taza"},
        TableExample{"it_textnorm.txt", "il 12/05/2026 alle 14:30",
                     "il dodici maggio due mila ventisei alle quattordici e "
                     "trenta"},
        TableExample{"it_textnorm.txt", "è di €1.500,50 al 1º piano",
                     "è di mille cinquecento euro e cinquanta centesimi al "
                     "primo piano"},
        TableExample{"it_textnorm.txt", "Il dott. Rossi della Rossi S.p.A.",
                     "Il dottor Rossi della Rossi esse pi a"},
        TableExample{"it_textnorm.txt", "alle 10:15 a -5 °C",
                     "alle dieci e un quarto a meno cinque gradi Celsius"},
        TableExample{"fr_textnorm.txt", "un parking pour 15 000 € à 10h15",
                     "un parking pour 15000 euros à 10 heures 15"},
        TableExample{"fr_textnorm.txt", "2/3 des voix, -5,5 °C, 3,75%",
                     "deux tiers des voix, moins 5 virgule 5 degrés Celsius, "
                     "3 virgule 75 pour cent"},
        TableExample{"pt_textnorm.txt", "custou R$ 1.500,50 às 14:30",
                     "custou 1500 reais e 50 centavos às duas e meia da tarde"},
        TableExample{"pt_textnorm.txt", "no dia 01/01/2026 às 09:00",
                     "no dia primeiro de janeiro de 2026 às 9 horas da manhã"},
        TableExample{"pt_textnorm.txt", "O Sr. Silva chegou em 2º lugar",
                     "O senhor Silva chegou em segundo lugar"},
        TableExample{"es_textnorm.txt", "el +34 91 123 4567",
                     "el más  3 4  9 1  1 2 3  4 5 6 7"},
        TableExample{"es_textnorm.txt", "llegó en el 1.º puesto, el 12.º piso",
                     "llegó en el primero puesto, el duodécimo piso"},
        TableExample{"es_textnorm.txt", "D. Juan, Dª. María y el Ing. Ruiz",
                     "don Juan, doña María y el ingeniero Ruiz"},
        TableExample{"es_textnorm.txt", "en la pág. 45", "en la página 45"},
        TableExample{"it_textnorm.txt", "il numero è +39 02 1234567.",
                     "il numero è più  tre nove  zero due  uno due tre quattro "
                     "cinque sei sette."},
        TableExample{"it_textnorm.txt",
                     "Carlo V e il XXI secolo, Giovanni XXIII",
                     "Carlo V e il ventunesimo secolo, Giovanni ventitreesimo"},
        TableExample{"it_textnorm.txt", "Ho 8 gatti, 108 pecore e 1500 galline",
                     "Ho otto gatti, cento otto pecore e mille cinquecento "
                     "galline"},
        TableExample{"it_textnorm.txt",
                     "Nel 2500000 e 1001000, il 21º posto e 007",
                     "Nel due milioni cinquecento mila e un milione mille, il "
                     "ventunesimo posto e  zero zero sette"},
        TableExample{"it_textnorm.txt", "un foglio A4 a 5°C e -3 punti",
                     "un foglio A quattro a cinque gradi Celsius e meno tre "
                     "punti"},
        TableExample{"fr_textnorm.txt", "le 2e ou le 3ème étage",
                     "le deuxième ou le troisième étage"},
        TableExample{"pt_textnorm.txt", "é +55 (11) 91234-5678.",
                     "é mais  5 5  1 1  9 1 2 3 4  5 6 7 8."},
        TableExample{"hi_textnorm.txt", "उसने १,००,००० रुपये और ₹४५०.५० दिए",
                     "उसने 100000 रुपये और 450 रुपए और 50 पैसे दिए"},
        TableExample{"hi_textnorm.txt", "सुबह १०:३० बजे, २:३० बजे, ५०%",
                     "सुबह साढ़े 10 बजे, ढाई बजे, 50 प्रतिशत"},
        TableExample{"hi_textnorm.txt", "२.५ किलो और ३/४ कप",
                     "2 दशमलव 5 किलो और तीन चौथाई कप"},
        TableExample{"hi_textnorm.txt", "नंबर +९१ ९८७६५-४३२१० है",
                     "नंबर  9 1  9 8 7 6 5  4 3 2 1 0 है"},
        TableExample{"pt_textnorm.txt", "um orçamento de R$ 50M e 0,05%",
                     "um orçamento de 50 milhões de reais e 0 vírgula 05 por "
                     "cento"},
        TableExample{"ja_textnorm.txt", "荷物を1個口、3個口で発送する。",
                     "荷物を いっこぐち、 さんこぐちで発送する。"},
        TableExample{"ja_textnorm.txt", "会議室には1人、2人、または4人の学生",
                     "会議室には ひとり、 ふたり、または よにんの学生"},
        TableExample{
            "ja_textnorm.txt", "105、110、1100、11000、2026",
            " ひゃくご、 ひゃくじゅう、 せんひゃく、 いちまんいっせん、 "
            "にせんにじゅうろく"},
        TableExample{"ja_textnorm.txt", "1500円で300本、8000人、600匹",
                     " せんごひゃく円で さんびゃっぽん、 はっせんにん、 "
                     "ろっぴゃっぴき"},
        TableExample{"ja_textnorm.txt", "10時10分に始まり、12時に終了",
                     " じゅうじ じゅっぷんに始まり、 じゅうにじに終了"},
        TableExample{"ja_textnorm.txt", "4月4日の7時、9時、3分",
                     " しがつ よっかの しちじ、 くじ、 さんぷん"},
        TableExample{"ja_textnorm.txt", "3/4、2.5%、¥1,500、-5℃",
                     " よんぶんの さん、 にてんごパーセント、 せんごひゃく円、"
                     "マイナス ご度"},
        TableExample{"ja_textnorm.txt", "３〜５個、20歳、1つ、8回、3階",
                     " さんから ごこ、 はたち、 ひとつ、 はっかい、 さんがい"},
        TableExample{
            "ja_textnorm.txt", "2026/10/13 10:00と14:05",
            " にせんにじゅうろく年 じゅうがつ じゅうさん日  じゅうじと "
            "じゅうよじ ごふん"}));

}  // namespace
}  // namespace litert::omni::tts
