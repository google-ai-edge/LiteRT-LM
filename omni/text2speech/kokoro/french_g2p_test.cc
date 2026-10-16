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

#include "omni/text2speech/kokoro/french_g2p.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl

namespace litert::omni::tts::kokoro {
namespace {

using ::absl_testing::IsOkAndHolds;

TEST(FrenchG2pTest, BasicWordsNasalsDigraphsAndSilentEnt) {
  const auto g2p_result = FrenchG2p::Create();
  ASSERT_TRUE(g2p_result.ok());
  const FrenchG2p& g2p = *g2p_result;

  EXPECT_THAT(g2p.WordToIpa("bonjour"), IsOkAndHolds("bɔ̃ʒˈuʁ"));
  EXPECT_THAT(g2p.WordToIpa("français"), IsOkAndHolds("fʁɑ̃sˈɛ"));
  EXPECT_THAT(g2p.WordToIpa("château"), IsOkAndHolds("ʃatˈo"));
  EXPECT_THAT(g2p.WordToIpa("président"), IsOkAndHolds("pʁezidˈɑ̃"));
  EXPECT_THAT(g2p.WordToIpa("parlent"), IsOkAndHolds("pˈaʁl"));
  EXPECT_THAT(g2p.WordToIpa("soleil"), IsOkAndHolds("solˈɛj"));
  EXPECT_THAT(g2p.WordToIpa("montagne"), IsOkAndHolds("mɔ̃tˈaɲ"));
  EXPECT_THAT(g2p.WordToIpa("aujourd'hui"), IsOkAndHolds("oʒuʁdyˈi"));
}

TEST(FrenchG2pTest, SentencesLiaisonHomographsAndNumbers) {
  const auto g2p_result = FrenchG2p::Create();
  ASSERT_TRUE(g2p_result.ok());
  const FrenchG2p& g2p = *g2p_result;

  EXPECT_THAT(
      g2p.TextToIpa("Les poules du couvent couvent tranquillement leurs œufs."),
      IsOkAndHolds("le pˈul dy kuvˈɑ̃ kˈuv tʁɑ̃kilmˈɑ̃ lœʁz ˈø."));
  EXPECT_THAT(g2p.TextToIpa(
                  "Les policiers pressent le pas car ils pressent un danger."),
              IsOkAndHolds("le polisjˈe pʁˈɛs lə pˈa kaʁ il pʁˈɛst œ̃ dɑ̃ʒˈe."));
  EXPECT_THAT(
      g2p.TextToIpa("Le train partira à 14h30 de la gare de Lyon."),
      IsOkAndHolds("lə tʁˈɛ̃ paʁtiʁˈa a katˈɔʁz ˈaʃ tʁˈɑ̃t də la ɡˈaʁ də ljˈɔ̃."));
}

TEST(FrenchG2pTest, CustomLexiconOverride) {
  const auto g2p_result = FrenchG2p::Create();
  ASSERT_TRUE(g2p_result.ok());
  const FrenchG2p& g2p = *g2p_result;
  const absl::flat_hash_map<std::string, std::string> custom = {
      {"kokoro", "kokoʁˈo"},
  };
  EXPECT_THAT(g2p.WordToIpa("Kokoro", custom), IsOkAndHolds("kokoʁˈo"));
  EXPECT_THAT(g2p.TextToIpa("Bonjour Kokoro.", custom),
              IsOkAndHolds("bɔ̃ʒˈuʁ kokoʁˈo."));
}

}  // namespace
}  // namespace litert::omni::tts::kokoro
