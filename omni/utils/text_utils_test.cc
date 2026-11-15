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

#include "omni/utils/text_utils.h"

#include <cstddef>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl

namespace litert::omni {
namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;

std::vector<char32_t> DecodeAll(absl::string_view text) {
  std::vector<char32_t> code_points;
  size_t pos = 0;
  while (pos < text.size())
    code_points.push_back(DecodeUtf8CodePoint(text, pos));
  return code_points;
}

TEST(TextUtilsTest, DecodeUtf8CodePointDecodesAllLengths) {
  EXPECT_THAT(DecodeAll("a\u00e9\u4e2d\U00020000"),
              ElementsAre(U'a', U'\u00e9', U'\u4e2d', U'\U00020000'));
}

TEST(TextUtilsTest, DecodeUtf8CodePointAdvancesOverInvalidBytes) {
  // A lone continuation byte and a truncated 3-byte sequence.
  EXPECT_THAT(DecodeAll("\x80"
                        "a"
                        "\xE4\xB8"),
              ElementsAre(0x80, U'a', 0xE4, 0xB8));
  // A lead byte followed by a non-continuation byte.
  EXPECT_THAT(DecodeAll("\xE4"
                        "ab"),
              ElementsAre(0xE4, U'a', U'b'));
  // 0xF8 is not a valid lead byte, even when continuation bytes follow.
  EXPECT_THAT(DecodeAll("\xF8\x80\x80\x80"),
              ElementsAre(0xF8, 0x80, 0x80, 0x80));
}

TEST(TextUtilsTest, DecodeUtf8CodePointAtEndReturnsZero) {
  size_t pos = 1;
  EXPECT_EQ(DecodeUtf8CodePoint("a", pos), 0);
  EXPECT_EQ(pos, 1);
}

TEST(TextUtilsTest, ClassifiesCharacters) {
  EXPECT_TRUE(IsCjkCharacter(U'中'));
  EXPECT_TRUE(IsCjkCharacter(U'こ'));
  EXPECT_TRUE(IsCjkCharacter(U'テ'));
  EXPECT_TRUE(IsCjkCharacter(U'\U00020000'));  // CJK Extension B.
  EXPECT_TRUE(IsCjkCharacter(U'々'));
  EXPECT_TRUE(IsCjkCharacter(U'〇'));
  EXPECT_FALSE(IsCjkCharacter(U'a'));
  EXPECT_FALSE(IsCjkCharacter(U'。'));

  EXPECT_TRUE(IsCjkPunctuation(U'。'));
  EXPECT_TRUE(IsCjkPunctuation(U'？'));
  EXPECT_TRUE(IsCjkPunctuation(U'「'));
  EXPECT_TRUE(IsCjkPunctuation(U'｛'));
  EXPECT_TRUE(IsCjkPunctuation(U'｡'));  // Halfwidth ideographic full stop.
  EXPECT_FALSE(IsCjkPunctuation(U'?'));
  EXPECT_FALSE(IsCjkPunctuation(U'々'));
  EXPECT_FALSE(IsCjkPunctuation(U'〇'));
  EXPECT_FALSE(IsCjkPunctuation(U'\u3000'));  // Ideographic space.

  EXPECT_TRUE(IsPunctuation(U','));
  EXPECT_TRUE(IsPunctuation(U'，'));
  EXPECT_TRUE(IsPunctuation(U'\u201C'));  // Left double quotation mark.
  EXPECT_FALSE(IsPunctuation(U'中'));
  EXPECT_FALSE(IsPunctuation(U'a'));
}

TEST(TextUtilsTest, SplitCjkCharactersSplitsCjkAndKeepsPunctuation) {
  EXPECT_THAT(SplitCjkCharacters({"你好吗？今天"}),
              ElementsAre("你", "好", "吗？", "今", "天"));
  EXPECT_THAT(SplitCjkCharacters({"欢迎体验", "LiteRT", "演示。"}),
              ElementsAre("欢", "迎", "体", "验", "LiteRT", "演", "示。"));
  EXPECT_THAT(
      SplitCjkCharacters({"用LiteRT做", "これはテスト"}),
      ElementsAre("用", "LiteRT", "做", "こ", "れ", "は", "テ", "ス", "ト"));
  // Leading punctuation has nothing to attach to.
  EXPECT_THAT(SplitCjkCharacters({"「中文」"}),
              ElementsAre("「", "中", "文」"));
  // Iteration marks and ideographic numerals are characters, not punctuation.
  EXPECT_THAT(SplitCjkCharacters({"人々"}), ElementsAre("人", "々"));
  EXPECT_THAT(SplitCjkCharacters({"二〇二六年"}),
              ElementsAre("二", "〇", "二", "六", "年"));
}

TEST(TextUtilsTest, SplitCjkCharactersKeepsLatinWords) {
  EXPECT_THAT(SplitCjkCharacters({"Hello,", "world!", "don't"}),
              ElementsAre("Hello,", "world!", "don't"));
  EXPECT_THAT(SplitCjkCharacters({}), IsEmpty());
}

TEST(TextUtilsTest, JoinWithCjkAwareSpacingInvertsSplit) {
  for (const std::vector<std::string>& words :
       std::vector<std::vector<std::string>>{
           {"欢迎体验", "Light", "RT", "Omni", "语音合成演示。"},
           {"hello", "world,", "this", "is", "a", "test."},
           {"你好吗？今天是晚上吃什么？"},
           {"请看（LiteRT）演示。"},
       }) {
    EXPECT_EQ(JoinWithCjkAwareSpacing(SplitCjkCharacters(words)),
              absl::StrJoin(words, " "));
  }
}

TEST(TextUtilsTest, JoinWithCjkAwareSpacingGluesOnlyCjk) {
  EXPECT_EQ(JoinWithCjkAwareSpacing({"吗？", "好"}), "吗？好");
  EXPECT_EQ(JoinWithCjkAwareSpacing({"中", "OK"}), "中 OK");
  EXPECT_EQ(JoinWithCjkAwareSpacing({"OK。", "好"}), "OK。好");
  EXPECT_EQ(JoinWithCjkAwareSpacing({"好，", "OK"}), "好，OK");
  EXPECT_EQ(JoinWithCjkAwareSpacing({"OK", "「好」"}), "OK「好」");
  EXPECT_EQ(JoinWithCjkAwareSpacing({"a", "", "b"}), "a b");
  EXPECT_EQ(JoinWithCjkAwareSpacing({}), "");
}

}  // namespace
}  // namespace litert::omni
