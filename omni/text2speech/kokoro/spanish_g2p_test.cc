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

#include "omni/text2speech/kokoro/spanish_g2p.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl

namespace litert::omni::tts::kokoro {
namespace {

using ::absl_testing::IsOkAndHolds;

TEST(SpanishG2pTest, BasicWordsAndAllophony) {
  const auto g2p_result = SpanishG2p::Create();
  ASSERT_TRUE(g2p_result.ok());
  const SpanishG2p& g2p = *g2p_result;

  EXPECT_THAT(g2p.WordToIpa("hola"), IsOkAndHolds("ˈola"));
  EXPECT_THAT(g2p.WordToIpa("mundo"), IsOkAndHolds("mˈundo"));
  EXPECT_THAT(g2p.WordToIpa("español"), IsOkAndHolds("ˌespaɲˈol"));
  EXPECT_THAT(g2p.WordToIpa("zorro"), IsOkAndHolds("θˈoro"));
  EXPECT_THAT(g2p.WordToIpa("llave"), IsOkAndHolds("ʎˈaβe"));
  EXPECT_THAT(g2p.WordToIpa("chocolate"), IsOkAndHolds("ʧˌokolˈate"));
  EXPECT_THAT(g2p.WordToIpa("guerra"), IsOkAndHolds("ɡˈera"));
  EXPECT_THAT(g2p.WordToIpa("pingüino"), IsOkAndHolds("pˌiŋɡuˈino"));
  EXPECT_THAT(g2p.WordToIpa("antigüedad"), IsOkAndHolds("ˌantiɣueðˈad"));
  EXPECT_THAT(g2p.WordToIpa("rápidamente"), IsOkAndHolds("rˈapiðamˈente"));
  EXPECT_THAT(g2p.WordToIpa("México"), IsOkAndHolds("mˈɛxiko"));
}

TEST(SpanishG2pTest, SentenceSandhiAndFunctionWords) {
  const auto g2p_result = SpanishG2p::Create();
  ASSERT_TRUE(g2p_result.ok());
  const SpanishG2p& g2p = *g2p_result;

  // Cross-word spirantization (de -> ðe after vowel, el gato -> el ɣˈato) and
  // unstressed function words (el, y, en).
  EXPECT_THAT(g2p.TextToIpa("El zorro y el gato en el ojo."),
              IsOkAndHolds("el θˈoro i el ɣˈato en el ˈoxo."));

  // Final -d is /ð/ before a vowel and /d/ before a consonant or pause.
  EXPECT_THAT(
      g2p.TextToIpa(
          "Busco el límite de velocidad antes de que la policía me limite."),
      IsOkAndHolds(
          "bˈusko el lˈimite ðe βˌeloθiðˈað ˈantes ðe ke la pˌoliθˈia me "
          "limˈite."));
  EXPECT_THAT(
      g2p.TextToIpa("El caos en la ciudad comenzó después de la tormenta."),
      IsOkAndHolds(
          "el kˈaos en la θjuðˈad kˌomenθˈo ðespwˈes ðe la toɾmˈɛnta."));

  // Cross-word nasal assimilation (un gran -> ˈuŋ ɡɾˈan, en Barcelona -> em
  // bˌaɾθelˈona) and inverted punctuation stripping (¿ / ¡).
  EXPECT_THAT(g2p.TextToIpa("¿Dónde vive un gran amigo?"),
              IsOkAndHolds("dˈonde βˈiβe ˈuŋ ɡɾˈan amˈiɣo?"));
}

TEST(SpanishG2pTest, NumbersAbbreviationsSymbolsAndRomanNumerals) {
  const auto g2p_result = SpanishG2p::Create();
  ASSERT_TRUE(g2p_result.ok());
  const SpanishG2p& g2p = *g2p_result;

  EXPECT_THAT(
      g2p.TextToIpa("El artículo fue publicado en la pág. 45 del tomo II."),
      IsOkAndHolds(
          "el aɾtˈikulo fwˈe pˌuβlikˈaðo en la pˈaɡ. kwaɾˌɛntaiθˈinko ðel "
          "tˈomo ðˈos."));
  EXPECT_THAT(
      g2p.TextToIpa("El costo total fue de €150,50 euros."),
      IsOkAndHolds(
          "el kˈosto totˈal fwˈe ðe ˈeʊɾo θjˈento θinkwˈɛnta,θinkwˈɛnta "
          "ˈeʊɾos."));
  EXPECT_THAT(
      g2p.TextToIpa("Hay un descuento del 10% en todos los artículos, etc."),
      IsOkAndHolds(
          "ˈI ˈun deskwˈɛnto ðel ðjˈɛθ poɾθjˈɛnto en tˈoðos los aɾtˈikulos, "
          "etθˈeteɾa."));
  EXPECT_THAT(
      g2p.TextToIpa("La empresa tiene su sede en Barcelona, S.A. de C.V."),
      IsOkAndHolds("la empɾˈesa tjˈene su sˈeðe em bˌaɾθelˈona, ˈɛse.ˈa. de "
                   "θˈe.ˈuβe."));
}

TEST(SpanishG2pTest, CustomLexiconOverride) {
  const auto g2p_result = SpanishG2p::Create();
  ASSERT_TRUE(g2p_result.ok());
  const SpanishG2p& g2p = *g2p_result;
  const absl::flat_hash_map<std::string, std::string> custom = {
      {"kokoro", "kokˈoɾo"},
  };
  EXPECT_THAT(g2p.WordToIpa("Kokoro", custom), IsOkAndHolds("kokˈoɾo"));
  EXPECT_THAT(g2p.TextToIpa("Hola Kokoro!", custom),
              IsOkAndHolds("ˈola kokˈoɾo!"));
}

}  // namespace
}  // namespace litert::omni::tts::kokoro
