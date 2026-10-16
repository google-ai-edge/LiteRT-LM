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

#include "omni/text2speech/kokoro/traditional_chinese_converter.h"

#include <fstream>
#include <ios>
#include <iterator>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace litert::omni::tts::kokoro {
namespace {

using ::absl_testing::StatusIs;

constexpr char kDictionary[] =
    "# comment\n"
    "\n"
    "乾坤\t乾坤\n"
    "銀行\t银行\n"
    "乾\t干 乾\n"
    "銀\t银\n"
    "號\t号\r\n";

// Reads the conversion table that ships with the model (generated from OpenCC
// by the //omni/text2speech:zh_t2s_dictionary
// genrule) from the test runfiles, so the tests cover the data we actually
// serve.
std::string ReadBundledDictionary() {
  const std::string base_dir = ::testing::SrcDir();
  const std::vector<std::string> candidate_paths = {
      "omni/text2speech/data/zh_t2s.txt",
      absl::StrCat(base_dir, "/",
                   "odml/litert_lm/omni/text2speech/data/zh_t2s.txt"),
      absl::StrCat(base_dir,
                   "/litert_lm/omni/text2speech/"
                   "data/zh_t2s.txt"),
  };
  for (const std::string& path : candidate_paths) {
    std::ifstream file(path, std::ios::binary);
    if (!file.good()) continue;
    return std::string(std::istreambuf_iterator<char>(file),
                       std::istreambuf_iterator<char>());
  }
  return "";
}

TEST(TraditionalChineseConverterTest, ConvertsWithLongestMatch) {
  ASSERT_OK_AND_ASSIGN(auto converter,
                       TraditionalChineseConverter::Create(kDictionary));
  EXPECT_EQ(converter->EntryCount(), 5);
  // 乾坤 is kept as a phrase, 乾 alone takes the first candidate.
  EXPECT_EQ(converter->Convert("乾坤與乾"), "乾坤與干");
  EXPECT_EQ(converter->Convert("銀行的銀子"), "银行的银子");
  EXPECT_EQ(converter->Convert("房號1025, OK!"), "房号1025, OK!");
}

TEST(TraditionalChineseConverterTest, PassesThroughUnmatchedText) {
  ASSERT_OK_AND_ASSIGN(auto converter,
                       TraditionalChineseConverter::Create(kDictionary));
  EXPECT_EQ(converter->Convert(""), "");
  EXPECT_EQ(converter->Convert("Hello, 世界 é"), "Hello, 世界 é");
}

TEST(TraditionalChineseConverterTest, RejectsMalformedDictionaries) {
  EXPECT_THAT(TraditionalChineseConverter::Create("銀行 银行\n"),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(TraditionalChineseConverter::Create("銀行\t\n"),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(TraditionalChineseConverter::Create("# only a comment\n"),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(TraditionalChineseConverterTest, BundledDictionaryConvertsSentences) {
  const std::string dictionary = ReadBundledDictionary();
  if (dictionary.empty()) {
    // The table is generated from OpenCC's dictionaries at build time and is
    // not part of the open-source build.
    GTEST_SKIP() << "zh_t2s.txt is not available in this build";
  }
  ASSERT_OK_AND_ASSIGN(auto converter,
                       TraditionalChineseConverter::Create(dictionary));
  EXPECT_EQ(converter->Convert("銀行在重慶，我們去了三次。"),
            "银行在重庆，我们去了三次。");
  EXPECT_EQ(converter->Convert("這個問題很複雜，需要長時間討論。"),
            "这个问题很复杂，需要长时间讨论。");
  EXPECT_EQ(converter->Convert("房間號是1025，價格是兩千元。"),
            "房间号是1025，价格是两千元。");
  // Identity phrases keep their characters from being converted on their own.
  EXPECT_EQ(converter->Convert("乾坤"), "乾坤");
  // Simplified text is left alone.
  EXPECT_EQ(converter->Convert("银行在重庆。"), "银行在重庆。");
  EXPECT_GT(converter->EntryCount(), 3000);
}

}  // namespace
}  // namespace litert::omni::tts::kokoro
