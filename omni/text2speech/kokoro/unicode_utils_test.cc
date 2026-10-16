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

#include "omni/text2speech/kokoro/unicode_utils.h"

#include <cstddef>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"  // from @com_google_absl

namespace litert::omni::tts::kokoro {
namespace {

using ::testing::ElementsAre;

TEST(UnicodeUtilsTest, DecodesEveryEncodingLength) {
  // "aé€😀": 1-, 2-, 3- and 4-byte code points.
  constexpr absl::string_view kText = "a\xC3\xA9\xE2\x82\xAC\xF0\x9F\x98\x80";
  const Utf8CodePoint a = DecodeUtf8CodePoint(kText, 0);
  EXPECT_EQ(a.code_point, U'a');
  EXPECT_EQ(a.length, 1);
  const Utf8CodePoint e_acute = DecodeUtf8CodePoint(kText, 1);
  EXPECT_EQ(e_acute.code_point, U'é');
  EXPECT_EQ(e_acute.length, 2);
  const Utf8CodePoint euro = DecodeUtf8CodePoint(kText, 3);
  EXPECT_EQ(euro.code_point, U'€');
  EXPECT_EQ(euro.length, 3);
  const Utf8CodePoint emoji = DecodeUtf8CodePoint(kText, 6);
  EXPECT_EQ(emoji.code_point, U'\U0001F600');
  EXPECT_EQ(emoji.length, 4);
  EXPECT_EQ(DecodeUtf8CodePoint(kText, kText.size()).length, 0);
}

TEST(UnicodeUtilsTest, MalformedLeadByteDecodesAsItself) {
  // A truncated 2-byte sequence and a lead byte without continuation.
  EXPECT_EQ(DecodeUtf8CodePoint("\xC3", 0).code_point, 0xC3);
  EXPECT_EQ(DecodeUtf8CodePoint("\xC3", 0).length, 1);
  EXPECT_EQ(DecodeUtf8CodePoint("\xE2x", 0).code_point, 0xE2);
  EXPECT_EQ(DecodeUtf8CodePoint("\xE2x", 0).length, 1);
}

TEST(UnicodeUtilsTest, ConsumeAdvancesUntilTheEnd) {
  constexpr absl::string_view kText = "h\xC3\xA9";
  size_t pos = 0;
  char32_t code_point = 0;
  ASSERT_TRUE(ConsumeUtf8CodePoint(kText, pos, code_point));
  EXPECT_EQ(code_point, U'h');
  ASSERT_TRUE(ConsumeUtf8CodePoint(kText, pos, code_point));
  EXPECT_EQ(code_point, U'é');
  EXPECT_EQ(pos, kText.size());
  EXPECT_FALSE(ConsumeUtf8CodePoint(kText, pos, code_point));
  EXPECT_EQ(code_point, U'é');
}

TEST(UnicodeUtilsTest, RoundTripsThroughCodePoints) {
  const std::string text = "ðə kwˈɪk 😀 पूर्व";
  const std::u32string u32 = Utf8ToU32(text);
  const std::vector<char32_t> code_points = Utf8ToCodePoints(text);
  EXPECT_EQ(std::u32string(code_points.begin(), code_points.end()), u32);
  EXPECT_EQ(CodePointsToUtf8(u32), text);
  EXPECT_EQ(CodePointsToUtf8(code_points), text);
  EXPECT_THAT(Utf8ToCodePoints("aé"), ElementsAre(U'a', U'é'));
}

TEST(UnicodeUtilsTest, AppendUtf8EncodesEveryLength) {
  std::string out;
  AppendUtf8(U'a', out);
  AppendUtf8(U'é', out);
  AppendUtf8(U'€', out);
  AppendUtf8(U'\U0001F600', out);
  EXPECT_EQ(out, "a\xC3\xA9\xE2\x82\xAC\xF0\x9F\x98\x80");
}

TEST(UnicodeUtilsTest, ToLowerLatin) {
  EXPECT_EQ(ToLowerLatin(U'A'), U'a');
  EXPECT_EQ(ToLowerLatin(U'À'), U'à');
  EXPECT_EQ(ToLowerLatin(U'Ç'), U'ç');
  EXPECT_EQ(ToLowerLatin(U'Þ'), U'þ');
  EXPECT_EQ(ToLowerLatin(U'Œ'), U'œ');
  EXPECT_EQ(ToLowerLatin(U'Ÿ'), U'ÿ');
  // Not letters, already lowercase, or outside the handled scripts.
  EXPECT_EQ(ToLowerLatin(U'×'), U'×');
  EXPECT_EQ(ToLowerLatin(U'ß'), U'ß');
  EXPECT_EQ(ToLowerLatin(U'1'), U'1');
  EXPECT_EQ(ToLowerLatin(U'Ω'), U'Ω');
  EXPECT_EQ(ToLowerUtf8("ÉCOLE Œuvre ŸSER 東京"), "école œuvre ÿser 東京");
}

TEST(UnicodeUtilsTest, StartsAndEndsWithAny) {
  EXPECT_TRUE(StartsWithAny("prendre", {"re", "pr"}));
  EXPECT_FALSE(StartsWithAny("prendre", {"re", "dr"}));
  EXPECT_TRUE(EndsWithAny("velocemente", {"zione", "mente"}));
  EXPECT_FALSE(EndsWithAny("velocemente", {}));
}

}  // namespace
}  // namespace litert::omni::tts::kokoro
