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

#include "omni/text2speech/kokoro/portuguese_g2p.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl

namespace litert::omni::tts::kokoro {
namespace {

using ::absl_testing::IsOkAndHolds;

TEST(PortugueseG2pTest, BasicWordsPalatalizationNasalAndCodaL) {
  const auto g2p_result = PortugueseG2p::Create();
  ASSERT_TRUE(g2p_result.ok());
  const PortugueseG2p& g2p = *g2p_result;

  EXPECT_THAT(g2p.WordToIpa("pão"), IsOkAndHolds("pˈɐ̃ʊ̃"));
  EXPECT_THAT(g2p.WordToIpa("leite"), IsOkAndHolds("lˈAʧy"));
  EXPECT_THAT(g2p.WordToIpa("cidade"), IsOkAndHolds("sˌidˈaʤy"));
  EXPECT_THAT(g2p.WordToIpa("brasil"), IsOkAndHolds("brazˈiʊ"));
  EXPECT_THAT(g2p.WordToIpa("muito"), IsOkAndHolds("mwˈiŋtʊ"));
  EXPECT_THAT(g2p.WordToIpa("coração"), IsOkAndHolds("kˌoɾasˈɐ̃ʊ̃"));
  EXPECT_THAT(g2p.WordToIpa("escola"), IsOkAndHolds("ˌeskˈɔlæ"));
  EXPECT_THAT(g2p.WordToIpa("projeto"), IsOkAndHolds("prˌoʒˈɛtʊ"));
}

TEST(PortugueseG2pTest, SentencesSandhiAndNumbers) {
  const auto g2p_result = PortugueseG2p::Create();
  ASSERT_TRUE(g2p_result.ok());
  const PortugueseG2p& g2p = *g2p_result;

  EXPECT_THAT(
      g2p.TextToIpa("O pão que comprei na padaria estava muito gostoso."),
      IsOkAndHolds(
          "ʊ pˈɐ̃ʊ̃ ky komprˈA na pˌadaɾˈiæ estˌavæ mwˈiŋtʊ ɡˌostˈozʊ."));
  EXPECT_THAT(
      g2p.TextToIpa("O preço do livro é de €15,50 em Portugal."),
      IsOkAndHolds(
          "ʊ prˈesʊ dʊ lˈivrw ɛ ʤy ˈeʊɾʊ kˈiŋzy,sˌiŋkwˈAŋtæ ˈAŋ pˌoɾətuɡˈW."));
  EXPECT_THAT(
      g2p.TextToIpa("A temperatura média no inverno é de -5 °C."),
      IsOkAndHolds("a tˌAmpeɾatˈuɾæ mˈɛʤjæ nʊ ˌiŋvˈɛɾənw ɛ ʤy mˈenʊs sˈiŋkʊ "
                   "ɡrˈW sˈe."));
}

TEST(PortugueseG2pTest, CustomLexiconOverride) {
  const auto g2p_result = PortugueseG2p::Create();
  ASSERT_TRUE(g2p_result.ok());
  const PortugueseG2p& g2p = *g2p_result;
  const absl::flat_hash_map<std::string, std::string> custom = {
      {"kokoro", "kokˈoɾʊ"},
  };
  EXPECT_THAT(g2p.WordToIpa("Kokoro", custom), IsOkAndHolds("kokˈoɾʊ"));
  EXPECT_THAT(g2p.TextToIpa("Olá Kokoro.", custom),
              IsOkAndHolds("olˈa kokˈoɾʊ."));
}

TEST(PortugueseG2pTest, OrdinalsMatchHostKokoro) {
  const auto g2p_result = PortugueseG2p::Create();
  ASSERT_TRUE(g2p_result.ok());
  const PortugueseG2p& g2p = *g2p_result;

  EXPECT_THAT(g2p.TextToIpa("1º"), IsOkAndHolds("prˌimˈAɾʊ"));
  EXPECT_THAT(g2p.TextToIpa("3ª"), IsOkAndHolds("tˌeɾəsˈAɾæ"));
  EXPECT_THAT(g2p.TextToIpa("4º"), IsOkAndHolds("kwˈaɾətʊ"));
  EXPECT_THAT(g2p.TextToIpa("4ª"), IsOkAndHolds("kwˈaɾətæ"));
  EXPECT_THAT(g2p.TextToIpa("10º"), IsOkAndHolds("dˈɛsimʊ"));
  EXPECT_THAT(g2p.TextToIpa("12ª"), IsOkAndHolds("dˈɛsimˌæseɡˈũŋdæ"));
  EXPECT_THAT(g2p.TextToIpa("18º"), IsOkAndHolds("dˈɛsimˌʊoɪtˈavʊ"));
  EXPECT_THAT(g2p.TextToIpa("19º"), IsOkAndHolds("dˈɛsimʊnˈonʊ"));
  EXPECT_THAT(g2p.TextToIpa("21º"), IsOkAndHolds("vˌiʒˈɛzimˌʊprimˈAɾʊ"));
  EXPECT_THAT(g2p.TextToIpa("99º"), IsOkAndHolds("nˌonaʒˈɛzimʊnˈonʊ"));
  EXPECT_THAT(g2p.TextToIpa("100ª"), IsOkAndHolds("sˌAŋtˈɛzim æ"));
  EXPECT_THAT(g2p.TextToIpa("101º"), IsOkAndHolds("sˌAŋtˈɛzimˌʊ primˈAɾʊ"));
  EXPECT_THAT(g2p.TextToIpa("110º"), IsOkAndHolds("sˌAŋtˈɛzimʊ dˈɛsimʊ"));
  EXPECT_THAT(g2p.TextToIpa("123º"),
              IsOkAndHolds("sˌAŋtˈɛzimˌʊ viʒˈɛzimˌʊteɾəsˈAɾʊ"));
}

TEST(PortugueseG2pTest, CurrencySymbols) {
  const auto g2p_result = PortugueseG2p::Create();
  ASSERT_TRUE(g2p_result.ok());
  const PortugueseG2p& g2p = *g2p_result;

  EXPECT_THAT(g2p.TextToIpa("R$ 10"), IsOkAndHolds("xeˈW dˈɛs"));
  EXPECT_THAT(g2p.TextToIpa("$ 5"), IsOkAndHolds("sifrˈɐ̃ʊ̃ sˈiŋkʊ"));
}

}  // namespace
}  // namespace litert::omni::tts::kokoro
