// Copyright 2026 Google LLC.
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

#include "omni/text2speech/kokoro/english_g2p.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "omni/text2speech/kokoro/cjk_blob.h"
#include "omni/text2speech/kokoro/cjk_test_blobs.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace litert::omni::tts::kokoro {
namespace {

using ::testing::status::IsOkAndHolds;
using ::testing::status::StatusIs;

void PackStringMap(std::vector<std::pair<std::string, std::string>> items,
                   std::string& k_idx, std::string& k_blob, std::string& v_idx,
                   std::string& v_blob) {
  std::sort(items.begin(), items.end());
  k_idx.reserve((items.size() + 1) * sizeof(uint32_t));
  v_idx.reserve((items.size() + 1) * sizeof(uint32_t));
  for (const auto& [k, v] : items) {
    AppendUint32(static_cast<uint32_t>(k_blob.size()), k_idx);
    k_blob.append(k);
    k_blob.push_back('\0');
    AppendUint32(static_cast<uint32_t>(v_blob.size()), v_idx);
    v_blob.append(v);
    v_blob.push_back('\0');
  }
  AppendUint32(static_cast<uint32_t>(k_blob.size()), k_idx);
  AppendUint32(static_cast<uint32_t>(v_blob.size()), v_idx);
}

std::string BuildSyntheticEnglishG2pBlob() {
  const std::vector<std::pair<std::string, std::string>> us_entries = {
      // Letters for get_NNP
      {"A", "4ˈA"},
      {"B", "4bˈi"},
      {"C", "4sˈi"},
      {"D", "4dˈi"},
      {"E", "4ˈi"},
      {"F", "4ˈɛf"},
      {"G", "4ʤˈi"},
      {"H", "4ˈAʧ"},
      {"I", "4ˈI"},
      {"J", "4ʤˈA"},
      {"K", "4kˈA"},
      {"L", "4ˈɛl"},
      {"M", "4ˈɛm"},
      {"N", "4ˈɛn"},
      {"O", "4ˈO"},
      {"P", "4pˈi"},
      {"S", "4ˈɛs"},
      {"T", "4tˈi"},
      {"U", "4jˈu"},
      {"V", "4vˈi"},
      // Function words and numbers
      {"am", "4ˈæm"},
      {"and", "4ænd"},
      {"apple", "4ˈæpᵊl"},
      {"at", "4æt"},
      {"be", "4DEFAULT=bi|None=bˈi"},
      {"became", "4bəkˈAm"},
      {"before", "4bəfˈɔɹ"},
      {"book", "4bˈʊk"},
      {"by", "4bI"},
      {"cent", "4sˈɛnt"},
      {"close", "4DEFAULT=klˈOs|NOUN=klˈOz|VERB=klˈOz"},
      {"content", "4DEFAULT=kəntˈɛnt|NOUN=kˈɑntɛnt"},
      {"doctor", "4dˈɑktəɹ"},
      {"dollar", "4dˈɑləɹ"},
      {"don't", "4dˈOnt"},
      {"door", "4dˈɔɹ"},
      {"edit", "4ˈɛdət"},
      {"eight", "4ˈAt"},
      {"equals", "4ˈikwᵊlz"},
      {"euro", "4jˈʊɹO"},
      {"fifteen", "4fˌɪftˈin"},
      {"fifty", "4fˈɪfti"},
      {"file", "4fˈIl"},
      {"first", "4fˈɜɹst"},
      {"five", "4fˈIv"},
      {"forty", "4fˈɔɹɾi"},
      {"four", "4fˈɔɹ"},
      {"get", "4ɡɛt"},
      {"go", "4ɡˈO"},
      {"he", "4DEFAULT=hi|None=hˈi"},
      {"hello", "4həlˈO"},
      {"his", "4hɪz"},
      {"honor", "4ˈɑnəɹ"},
      {"hundred", "4hˈʌndɹəd"},
      {"in", "4ɪn"},
      {"invalid", "4DEFAULT=ɪnvˈælɪd|NOUN=ˈɪnvəlɪd"},
      {"is", "4ɪz"},
      {"it", "4ɪt"},
      {"live", "4DEFAULT=lˈIv|VERB=lˈɪv"},
      {"lives", "4DEFAULT=lˈIvz|VERB=lˈɪvz"},
      {"micro", "4mˈIkɹə"},
      {"minus", "4mˈInəs"},
      {"nine", "4nˈIn"},
      {"ninety", "4nˈIndi"},
      {"noble", "4nˈObᵊl"},
      {"now", "4nˈW"},
      {"of", "4ʌv"},
      {"oh", "4ˈO"},
      {"on", "4ˌɔn"},
      {"one", "4wˈʌn"},
      {"only", "4ˈOnli"},
      {"passenger", "4pˈæsᵊnʤəɹ"},
      {"pence", "4pˈɛns"},
      {"percent", "4pəɹsˈɛnt"},
      {"please", "4plˈiz"},
      {"plus", "4plˈʌs"},
      {"point", "4pˈYnt"},
      {"pound", "4pˈWnd"},
      {"present", "4DEFAULT=pɹˈɛzᵊnt|VERB=pɹizˈɛnt"},
      {"read", "4ADJ=ɹˈɛd|DEFAULT=ɹˈid|VBD=ɹˈɛd|VBN=ɹˈɛd|VBP=ɹˈɛd"},
      {"record", "4DEFAULT=ɹˈɛkəɹd|VERB=ɹəkˈɔɹd"},
      {"say", "4sˈA"},
      {"second", "4sˈɛkənd"},
      {"slash", "4slˈæʃ"},
      {"smith", "4smˈɪθ"},
      {"so", "4sˌO"},
      {"soft", "4sˈɑft"},
      {"street", "4stɹˈit"},
      {"subject", "4DEFAULT=sˈʌbʤɛkt|VERB=səbʤˈɛkt"},
      {"ten", "4tˈɛn"},
      {"that", "4DEFAULT=ðæt|None=ðˈæt"},
      {"thirty", "4θˈɜɹɾi"},
      {"thousand", "4θˈWzᵊnd"},
      {"three", "4θɹˈi"},
      {"to", "4tu"},
      {"too", "4tˈu"},
      {"touch", "4tˈʌʧ"},
      {"twelfth", "4twˈɛlfθ"},
      {"twelve", "4twˈɛlv"},
      {"twenty", "4twˈɛnti"},
      {"two", "4tˈu"},
      {"used", "4DEFAULT=jˈuzd|VBD=jˈust"},
      {"versus", "4vˈɜɹsəs"},
      {"wait", "4wˈAt"},
      {"was", "4wʌz"},
      {"water", "4wˈɔɾəɹ"},
      {"we", "4DEFAULT=wi|None=wˈi"},
      {"will", "4wɪl"},
      {"window", "4wˈɪndO"},
      {"with", "4wɪð"},
      {"world", "3wˈɜɹld"},
      {"yesterday", "4jˈɛstəɹdˌA"},
      {"you", "4ju"},
      {"zero", "4zˈɪɹO"},
  };

  const std::vector<std::pair<std::string, std::string>> gb_entries = {
      {"A", "4ˈA"},         {"B", "4bˈiː"},        {"C", "4sˈiː"},
      {"E", "4ˈiː"},        {"F", "4ˈɛf"},         {"I", "4ˈI"},
      {"M", "4ˈɛm"},        {"P", "4pˈiː"},        {"S", "4ˈɛs"},
      {"T", "4tˈiː"},       {"apple", "4ˈapᵊl"},   {"doctor", "4dˈɒktə"},
      {"edit", "4ˈɛdɪt"},   {"five", "4fˈIv"},     {"four", "4fˈɔː"},
      {"hello", "4həlˈQ"},  {"micro", "4mˈIkɹə"},  {"minus", "4mˈInəs"},
      {"noble", "4nˈQblᵊ"}, {"one", "4wˈʌn"},      {"pound", "4pˈWnd"},
      {"soft", "4sˈɒft"},   {"thirty", "4θˈɜːti"}, {"three", "4θɹˈiː"},
      {"to", "4tuː"},       {"twelve", "4twˈɛlv"}, {"twenty", "4twˈɛnti"},
      {"two", "4tˈuː"},     {"wait", "4wˈAt"},     {"water", "4wˈɔːtə"},
      {"world", "3wˈɜːld"}, {"zero", "4zˈɪəɹQ"},
  };

  std::string us_idx;
  std::string us_key;
  std::string us_vidx;
  std::string us_val;
  PackStringMap(us_entries, us_idx, us_key, us_vidx, us_val);

  std::string gb_idx;
  std::string gb_key;
  std::string gb_vidx;
  std::string gb_val;
  PackStringMap(gb_entries, gb_idx, gb_key, gb_vidx, gb_val);

  return BuildCjkBlob({
      {"us-idx", us_idx},
      {"us-key", us_key},
      {"us-vidx", us_vidx},
      {"us-val", us_val},
      {"gb-idx", gb_idx},
      {"gb-key", gb_key},
      {"gb-vidx", gb_vidx},
      {"gb-val", gb_val},
  });
}

TEST(EnglishG2pTest, RejectsMissingSections) {
  const std::string raw = BuildCjkBlob({{"us-idx", std::string(4, '\0')}});
  ASSERT_OK_AND_ASSIGN(const CjkBlob blob, CjkBlob::Create(raw));
  EXPECT_THAT(EnglishG2p::Create(blob), StatusIs(absl::StatusCode::kNotFound));
}

TEST(EnglishG2pTest, HandlesEmptyAndWhitespaceText) {
  const std::string raw = BuildSyntheticEnglishG2pBlob();
  ASSERT_OK_AND_ASSIGN(const CjkBlob blob, CjkBlob::Create(raw));
  ASSERT_OK_AND_ASSIGN(const EnglishG2p g2p, EnglishG2p::Create(blob));

  EXPECT_THAT(g2p.TextToIpa(""), IsOkAndHolds(""));
  EXPECT_THAT(g2p.TextToIpa("   \t\n "), IsOkAndHolds(""));
}

TEST(EnglishG2pTest, DistinguishesAmericanAndBritishAccents) {
  const std::string raw = BuildSyntheticEnglishG2pBlob();
  ASSERT_OK_AND_ASSIGN(const CjkBlob blob, CjkBlob::Create(raw));
  ASSERT_OK_AND_ASSIGN(const EnglishG2p g2p, EnglishG2p::Create(blob));

  EXPECT_THAT(g2p.TextToIpa("Hello world, water!", /*british=*/false),
              IsOkAndHolds("həlˈO wˈɜɹld, wˈɔɾəɹ!"));
  EXPECT_THAT(g2p.TextToIpa("Hello world, water!", /*british=*/true),
              IsOkAndHolds("həlˈQ wˈɜːld, wˈɔːtə!"));
}

TEST(EnglishG2pTest, ContextSensitiveFunctionWords) {
  const std::string raw = BuildSyntheticEnglishG2pBlob();
  ASSERT_OK_AND_ASSIGN(const CjkBlob blob, CjkBlob::Create(raw));
  ASSERT_OK_AND_ASSIGN(const EnglishG2p g2p, EnglishG2p::Create(blob));

  // "the" -> ðə before consonant ("book"), ði before vowel ("apple").
  // "to" -> tə before consonant ("get"), tʊ before vowel ("edit"), tu at end.
  EXPECT_THAT(g2p.TextToIpa("the book and the apple"),
              IsOkAndHolds("ðə bˈʊk ænd ði ˈæpᵊl"));
  EXPECT_THAT(g2p.TextToIpa("to get, to edit, to."),
              IsOkAndHolds("tə ɡɛt, tʊ ˈɛdət, tu."));
  EXPECT_THAT(g2p.TextToIpa("I used to go, he used it."),
              IsOkAndHolds("ˌI jˈust tə ɡˈO, hi jˈuzd ɪt."));
}

TEST(EnglishG2pTest, DisambiguatesPosHeteronyms) {
  const std::string raw = BuildSyntheticEnglishG2pBlob();
  ASSERT_OK_AND_ASSIGN(const CjkBlob blob, CjkBlob::Create(raw));
  ASSERT_OK_AND_ASSIGN(const EnglishG2p g2p, EnglishG2p::Create(blob));

  EXPECT_THAT(
      g2p.TextToIpa(
          "Please close the door before you get too close to the window."),
      IsOkAndHolds(
          "plˈiz klˈOz ðə dˈɔɹ bəfˈɔɹ ju ɡɛt tˈu klˈOs tə ðə wˈɪndO."));
  EXPECT_THAT(g2p.TextToIpa("He was content with the content of the book."),
              IsOkAndHolds("hˌi wʌz kəntˈɛnt wɪð ðə kˈɑntɛnt ʌv ðə bˈʊk."));
  EXPECT_THAT(g2p.TextToIpa("We read that the read-only file was invalid."),
              IsOkAndHolds("wˌi ɹˈɛd ðæt ðə ɹˌidˈOnli fˈIl wʌz ɪnvˈælɪd."));
  EXPECT_THAT(g2p.TextToIpa("The invalid passenger will present the present."),
              IsOkAndHolds("ði ɪnvˈælɪd pˈæsᵊnʤəɹ wɪl pɹizˈɛnt ðə pɹˈɛzᵊnt."));
  EXPECT_THAT(g2p.TextToIpa("Dr. Smith lives on the street."),
              IsOkAndHolds("dˈɑktəɹ. smˈɪθ lˈɪvz ˌɔn ðə stɹˈit."));
}

TEST(EnglishG2pTest, InflectsMorphologicalSuffixes) {
  const std::string raw = BuildSyntheticEnglishG2pBlob();
  ASSERT_OK_AND_ASSIGN(const CjkBlob blob, CjkBlob::Create(raw));
  ASSERT_OK_AND_ASSIGN(const EnglishG2p g2p, EnglishG2p::Create(blob));

  // US flapping on -ed / -ing after US_TAUS ('t' -> 'ɾᵻd' / 'ɾɪŋ') vs GB 'ɪd'.
  EXPECT_THAT(g2p.WordToIpa("waited", /*british=*/false),
              IsOkAndHolds("wˈAɾᵻd"));
  EXPECT_THAT(g2p.WordToIpa("waited", /*british=*/true),
              IsOkAndHolds("wˈAtɪd"));
  EXPECT_THAT(g2p.WordToIpa("editing", /*british=*/false),
              IsOkAndHolds("ˈɛdəɾɪŋ"));
  EXPECT_THAT(g2p.WordToIpa("apples", /*british=*/false),
              IsOkAndHolds("ˈæpᵊlz"));
}

TEST(EnglishG2pTest, ExpandsNumbersCurrenciesTimesAndUnits) {
  const std::string raw = BuildSyntheticEnglishG2pBlob();
  ASSERT_OK_AND_ASSIGN(const CjkBlob blob, CjkBlob::Create(raw));
  ASSERT_OK_AND_ASSIGN(const EnglishG2p g2p, EnglishG2p::Create(blob));

  EXPECT_THAT(g2p.TextToIpa("12.5% and $1,245.99 on 12th."),
              IsOkAndHolds("twˈɛlv pYnt fˈIv pəɹsˈɛnt ænd wˈʌn θˈWzᵊnd tˈu "
                           "hˈʌndɹəd fˈɔɹɾi fˈIv dˈɑləɹz ænd nˈIndi nˈIn "
                           "sˈɛnts ˌɔn twˈɛlfθ."));
  EXPECT_THAT(g2p.TextToIpa("23°F at 4:30 PM."),
              IsOkAndHolds("twˈɛnti θɹˈi dᵻɡɹˈiz ˈɛf æt fˈɔɹ:θˈɜɹɾi pˌiˈɛm."));
  EXPECT_THAT(g2p.TextToIpa("10 lbs and 1 tsp at Microsoft."),
              IsOkAndHolds("tˈɛn pˈWndz ænd wˈʌn tˌiˌɛspˈi æt mˈIkɹəsˌɑft."));
}

TEST(EnglishG2pTest, SupportsMarkdownOverridesCustomLexiconAndLts) {
  const std::string raw = BuildSyntheticEnglishG2pBlob();
  ASSERT_OK_AND_ASSIGN(const CjkBlob blob, CjkBlob::Create(raw));
  ASSERT_OK_AND_ASSIGN(const EnglishG2p g2p, EnglishG2p::Create(blob));

  EXPECT_THAT(g2p.TextToIpa("Say [Kokoro](/kˈOkəɹO/) now."),
              IsOkAndHolds("sˈA kˈOkəɹO nˈW."));
  const absl::flat_hash_map<std::string, std::string> custom = {
      {"litert", "lˈItˌɑɹtˈi"},
  };
  EXPECT_THAT(g2p.TextToIpa("Hello LiteRT", /*british=*/false, custom),
              IsOkAndHolds("həlˈO lˈItˌɑɹtˈi"));
  // British spelling fallback ("honour" -> "honor") and comparative ("nobler").
  EXPECT_THAT(g2p.WordToIpa("honour", /*british=*/false),
              IsOkAndHolds("ˈɑnəɹ"));
  EXPECT_THAT(g2p.WordToIpa("nobler", /*british=*/false),
              IsOkAndHolds("nˈObləɹ"));
}

}  // namespace
}  // namespace litert::omni::tts::kokoro
