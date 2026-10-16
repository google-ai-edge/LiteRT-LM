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

#include "omni/text2speech/kokoro/italian_g2p.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl

namespace litert::omni::tts::kokoro {
namespace {

using ::absl_testing::IsOkAndHolds;

TEST(ItalianG2pTest, BasicWordsGeminationVelarNasalAndVowels) {
  const auto g2p_result = ItalianG2p::Create();
  ASSERT_TRUE(g2p_result.ok());
  const ItalianG2p& g2p = *g2p_result;

  EXPECT_THAT(g2p.WordToIpa("ciao"), IsOkAndHolds("ʧˈao"));
  EXPECT_THAT(g2p.WordToIpa("pizza"), IsOkAndHolds("pˈiʦːa"));
  EXPECT_THAT(g2p.WordToIpa("pranzo"), IsOkAndHolds("prˈaŋʣo"));
  EXPECT_THAT(g2p.WordToIpa("ogni"), IsOkAndHolds("ˈoɲɲɪ"));
  EXPECT_THAT(g2p.WordToIpa("significa"), IsOkAndHolds("siɲˈifika"));
  EXPECT_THAT(g2p.WordToIpa("tecnologia"), IsOkAndHolds("teknoloʤˈia"));
  EXPECT_THAT(g2p.WordToIpa("scienza"), IsOkAndHolds("ʃˈɛnʦa"));
  EXPECT_THAT(g2p.WordToIpa("famiglia"), IsOkAndHolds("famˈiʎa"));
}

TEST(ItalianG2pTest, SentencesCliticsSandhiAndNumbers) {
  const auto g2p_result = ItalianG2p::Create();
  ASSERT_TRUE(g2p_result.ok());
  const ItalianG2p& g2p = *g2p_result;

  EXPECT_THAT(
      g2p.TextToIpa("Il treno per Milano parte dal binario 4."),
      IsOkAndHolds("il trˈɛno per milˈano pˈarte dal binˈario kwˈatːro."));
  EXPECT_THAT(
      g2p.TextToIpa("L'opera è iniziata alle 20:30 in punto."),
      IsOkAndHolds("lˈɔpera e iniʦjˈata ˌalle vˈentɪ:trˈeːnta in pˈunto."));
  EXPECT_THAT(
      g2p.TextToIpa("Il biglietto costa €12,50 a persona."),
      IsOkAndHolds("il biʎˈetːo kˈɔsta ˈɛʊro dˈodiʧɪ,ʧinkwˈanta a persˈona."));
}

TEST(ItalianG2pTest, CustomLexiconOverride) {
  const auto g2p_result = ItalianG2p::Create();
  ASSERT_TRUE(g2p_result.ok());
  const ItalianG2p& g2p = *g2p_result;
  const absl::flat_hash_map<std::string, std::string> custom = {
      {"kokoro", "kokˈoro"},
  };
  EXPECT_THAT(g2p.WordToIpa("Kokoro", custom), IsOkAndHolds("kokˈoro"));
  EXPECT_THAT(g2p.TextToIpa("Ciao Kokoro.", custom),
              IsOkAndHolds("ʧˈao kokˈoro."));
}

}  // namespace
}  // namespace litert::omni::tts::kokoro
