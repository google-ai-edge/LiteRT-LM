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

#include "omni/text2speech/kokoro/japanese_g2p.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "omni/text2speech/kokoro/cjk_blob.h"
#include "omni/text2speech/kokoro/cjk_test_blobs.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace litert::omni::tts::kokoro {
namespace {

using ::absl_testing::IsOkAndHolds;

// Builds a synthetic `ja-lexicon` blob with UniDic's 11 character categories
// (`DEFAULT`=0, `SPACE`=1, `KANJI`=2, `SYMBOL`=3, `NUMERIC`=4, `ALPHA`=5,
// `HIRAGANA`=6, `KATAKANA`=7, ...) and pruned 8-column feature records
// (`pos1,pos2,pos3,pos4,cType,pron,kana,aType`).
std::string BuildSyntheticJapaneseG2pBlob() {
  std::vector<uint32_t> char_records(0x3100, 0);
  const uint32_t default_rec =
      PackMecabCharInfo(1 << 0, 0, /*length=*/0, /*group=*/true,
                        /*invoke=*/false);
  for (uint32_t& r : char_records) r = default_rec;
  char_records[' '] = PackMecabCharInfo(1 << 1, 1, /*length=*/0, /*group=*/true,
                                        /*invoke=*/false);
  for (int cp = 'a'; cp <= 'z'; ++cp) {
    char_records[cp] =
        PackMecabCharInfo(1 << 5, 5, /*length=*/0, /*group=*/true,
                          /*invoke=*/true);
  }
  for (int cp = 'A'; cp <= 'Z'; ++cp) {
    char_records[cp] =
        PackMecabCharInfo(1 << 5, 5, /*length=*/0, /*group=*/true,
                          /*invoke=*/true);
  }
  for (int cp = 0x3040; cp <= 0x309F; ++cp) {
    char_records[cp] =
        PackMecabCharInfo(1 << 6, 6, /*length=*/2, /*group=*/true,
                          /*invoke=*/false);
  }
  for (int cp = 0x30A0; cp <= 0x30FF; ++cp) {
    char_records[cp] =
        PackMecabCharInfo(1 << 7, 7, /*length=*/2, /*group=*/true,
                          /*invoke=*/true);
  }
  char_records[0x3001] =
      PackMecabCharInfo(1 << 3, 3, /*length=*/0, /*group=*/true,
                        /*invoke=*/false);
  char_records[0x3002] =
      PackMecabCharInfo(1 << 3, 3, /*length=*/0, /*group=*/true,
                        /*invoke=*/false);

  const std::vector<std::string> categories = {
      "DEFAULT",  "SPACE",    "KANJI",        "SYMBOL", "NUMERIC", "ALPHA",
      "HIRAGANA", "KATAKANA", "KANJINUMERIC", "GREEK",  "CYRILLIC"};
  const std::string char_bin =
      BuildMecabCharProperty(categories, char_records, 0x3100);

  std::vector<uint32_t> sys_offsets;
  const std::string sys_features = BuildMecabFeatures(
      {
          "名詞,普通名詞,一般,*,*,キョー,キョウ,1",
          "助詞,係助詞,*,*,*,ワ,ハ,*",
          "名詞,普通名詞,一般,*,*,テンキ,テンキ,1",
          "助動詞,*,*,*,助動詞-デス,デス,デス,*",
          "補助記号,句点,*,*,*,*,,*",
          "名詞,普通名詞,一般,*,*,サンポ,サンポ,0",
          "名詞,普通名詞,一般,*,*,シンカンセン,シンカンセン,3",
          "名詞,普通名詞,一般,*,*,ガッコー,ガッコウ,0",
          "名詞,数詞,*,*,*,ニ,ニ,1",
          "名詞,数詞,*,*,*,ジュー,ジュウ,1",
          "名詞,数詞,*,*,*,サン,サン,0",
          "形容詞,非自立可能,*,*,形容詞,イー,イイ,1",
      },
      sys_offsets);

  std::string sys_tokens;
  for (uint32_t offset : sys_offsets) {
    sys_tokens += BuildMecabToken(1, 1, 1, 100, offset);
  }

  const std::string sys_da = BuildMecabDoubleArray({
      {"今日", (0 << 8) | 1},
      {"は", (1 << 8) | 1},
      {"天気", (2 << 8) | 1},
      {"です", (3 << 8) | 1},
      {"。", (4 << 8) | 1},
      {"散歩", (5 << 8) | 1},
      {"新幹線", (6 << 8) | 1},
      {"学校", (7 << 8) | 1},
      {"に", (8 << 8) | 1},
      {"じゅう", (9 << 8) | 1},
      {"さん", (10 << 8) | 1},
      {"いい", (11 << 8) | 1},
  });

  std::vector<uint32_t> unk_offsets;
  std::vector<std::string> unk_records;
  std::string unk_tokens;
  std::vector<std::pair<std::string, int>> unk_entries;
  for (size_t i = 0; i < categories.size(); ++i) {
    unk_records.push_back("名詞,普通名詞,一般,*,*,*,*");
  }
  const std::string unk_features = BuildMecabFeatures(unk_records, unk_offsets);
  for (size_t i = 0; i < categories.size(); ++i) {
    unk_tokens += BuildMecabToken(1, 1, 1, 5000, unk_offsets[i]);
    unk_entries.push_back({categories[i], static_cast<int>((i << 8) | 1)});
  }

  return BuildCjkBlob({
      {"sys-da", sys_da},
      {"sys-tok", sys_tokens},
      {"sys-feat", sys_features},
      {"matrix", BuildMecabConnectionMatrix(2, 2, {0, 0, 0, 0})},
      {"char", char_bin},
      {"unk-da", BuildMecabDoubleArray(unk_entries)},
      {"unk-tok", unk_tokens},
      {"unk-feat", unk_features},
  });
}

TEST(JapaneseG2pTest, ConvertsKanjiAndParticlesUsingPronColumn) {
  const std::string data = BuildSyntheticJapaneseG2pBlob();
  ASSERT_OK_AND_ASSIGN(const CjkBlob blob, CjkBlob::Create(data));
  ASSERT_OK_AND_ASSIGN(const JapaneseG2p g2p, JapaneseG2p::Create(blob));

  // The topic particle "は" has `pron=ワ`, read as "wa": misaki (and so Kokoro)
  // writes the Japanese "w" as `β`, so わ is `βa`, whereas ば would be `ba`.
  // "今日" has `pron=キョー` (`kʲoː`), "いい" has `pron=イー` (`iː`), and "。"
  // attaches to the preceding word without an intervening space.
  EXPECT_THAT(g2p.TextToIpa("今日はいい天気です。"),
              IsOkAndHolds("kʲoː βa iː teŋkʲi desɨ."));
}

TEST(JapaneseG2pTest, AssimilatesMoraicNasalAndGlottalStop) {
  const std::string data = BuildSyntheticJapaneseG2pBlob();
  ASSERT_OK_AND_ASSIGN(const CjkBlob blob, CjkBlob::Create(data));
  ASSERT_OK_AND_ASSIGN(const JapaneseG2p g2p, JapaneseG2p::Create(blob));

  // 散歩 (サンポ) -> `ん` assimilates to 'm' before 'p'; 学校 (ガッコー) ->
  // `っ` becomes 'ʔ' with no adjacent spaces and `ー` lengthens the vowel.
  EXPECT_THAT(g2p.TextToIpa("散歩"), IsOkAndHolds("sampo"));
  EXPECT_THAT(g2p.TextToIpa("学校"), IsOkAndHolds("ɡaʔkoː"));
}

TEST(JapaneseG2pTest, NormalizesDigitsViaNumToKana) {
  const std::string data = BuildSyntheticJapaneseG2pBlob();
  ASSERT_OK_AND_ASSIGN(const CjkBlob blob, CjkBlob::Create(data));
  ASSERT_OK_AND_ASSIGN(const JapaneseG2p g2p, JapaneseG2p::Create(blob));

  // "23" -> "にじゅうさん" -> segmented as "に", "じゅう", "さん".
  EXPECT_THAT(g2p.TextToIpa("23"), IsOkAndHolds("ɲi ʥɨː saɴ"));
}

TEST(JapaneseG2pTest, ComposesHalfwidthKatakanaDakutenOfEitherParity) {
  const std::string data = BuildSyntheticJapaneseG2pBlob();
  ASSERT_OK_AND_ASSIGN(const CjkBlob blob, CjkBlob::Create(data));
  ASSERT_OK_AND_ASSIGN(const JapaneseG2p g2p, JapaneseG2p::Create(blob));

  // テ, ト and ツ sit at even code points (after small ッ), unlike カ..チ, and
  // must still compose with a half-width dakuten into デ, ド and ヅ.
  EXPECT_THAT(g2p.TextToIpa("ﾃﾞﾄﾞﾂﾞ"), IsOkAndHolds("dedozɨ"));
  EXPECT_THAT(g2p.TextToIpa("ｶﾞﾊﾟ"), IsOkAndHolds("ɡapa"));
}

TEST(JapaneseG2pTest, ExpandsIterationMarksToIpa) {
  const std::string data = BuildSyntheticJapaneseG2pBlob();
  ASSERT_OK_AND_ASSIGN(const CjkBlob blob, CjkBlob::Create(data));
  ASSERT_OK_AND_ASSIGN(const JapaneseG2p g2p, JapaneseG2p::Create(blob));

  // `ゝ` repeats the previous kana as IPA rather than as raw kana, and `ゞ`
  // voices it, including `つ` -> `づ`, whose code point parity differs from
  // `か`..`ち`.
  EXPECT_THAT(g2p.TextToIpa("すゝ"), IsOkAndHolds("sɨsɨ"));
  EXPECT_THAT(g2p.TextToIpa("つゞ"), IsOkAndHolds("ʦɨzɨ"));
  EXPECT_THAT(g2p.TextToIpa("かゞ"), IsOkAndHolds("kaɡa"));
}

}  // namespace
}  // namespace litert::omni::tts::kokoro
