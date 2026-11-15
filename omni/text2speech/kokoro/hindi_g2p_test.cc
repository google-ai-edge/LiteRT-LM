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

#include "omni/text2speech/kokoro/hindi_g2p.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl

namespace litert::omni::tts::kokoro {
namespace {

using ::absl_testing::IsOkAndHolds;

TEST(HindiG2pTest, BasicWordsSchwaDeletionAndNukta) {
  const auto g2p_result = HindiG2p::Create();
  ASSERT_TRUE(g2p_result.ok());
  const HindiG2p& g2p = *g2p_result;

  EXPECT_THAT(g2p.WordToIpa("नमस्ते"), IsOkAndHolds("nəmˈʌsteː"));
  EXPECT_THAT(g2p.WordToIpa("भारत"), IsOkAndHolds("bʰˈaːɾət"));
  EXPECT_THAT(g2p.WordToIpa("समझदार"), IsOkAndHolds("sˌəməɟʰdˈaːɾ"));
  EXPECT_THAT(g2p.WordToIpa("दरवाज़ा"), IsOkAndHolds("dˌəɾʋˈaːzaː"));
  EXPECT_THAT(g2p.WordToIpa("सुंदर"), IsOkAndHolds("sˈũdəɾ"));
  EXPECT_THAT(g2p.WordToIpa("महिलाओं"), IsOkAndHolds("mˌəhɪlˈaːw"));
  EXPECT_THAT(g2p.WordToIpa("अच्छा"), IsOkAndHolds("ˈʌcʰcʰaː"));
  EXPECT_THAT(g2p.WordToIpa("युद्ध"), IsOkAndHolds("jˈʊdʰː"));
}

TEST(HindiG2pTest, SentencesDandaAndNumbers) {
  const auto g2p_result = HindiG2p::Create();
  ASSERT_TRUE(g2p_result.ok());
  const HindiG2p& g2p = *g2p_result;

  EXPECT_THAT(g2p.TextToIpa("यह मेरी पसंदीदा पुस्तक है।"),
              IsOkAndHolds("jˌəh mˌeːɾi pˌəsndˈiːdaː pˈʊstək hɛː."));
  EXPECT_THAT(
      g2p.TextToIpa("उसकी परीक्षा में ५०% अंक आए हैं।"),
      IsOkAndHolds("ˌʊskˌi pəɾˈiːkʃaː mẽː pəcˈaːs pətˈɪʃət ˈʌŋk ˈaːeː hɛ̃."));
  EXPECT_THAT(g2p.TextToIpa("इस मोबाइल की कीमत १२,९९९ रुपये है।"),
              IsOkAndHolds("ˈɪs moːbˈaːɪl ki kˈiːmət bˈaːɾəh,nˈɔːsˈɔː "
                           "nɪɲjˈaːnəbˌeː ɾˈʊpjˌeː hɛː."));
}

TEST(HindiG2pTest, CustomLexiconOverride) {
  const auto g2p_result = HindiG2p::Create();
  ASSERT_TRUE(g2p_result.ok());
  const HindiG2p& g2p = *g2p_result;
  const absl::flat_hash_map<std::string, std::string> custom = {
      {"कोकोरो", "koːkˈoːɾoː"},
  };
  EXPECT_THAT(g2p.WordToIpa("कोकोरो", custom), IsOkAndHolds("koːkˈoːɾoː"));
  EXPECT_THAT(g2p.TextToIpa("नमस्ते कोकोरो।", custom),
              IsOkAndHolds("nəmˈʌsteː koːkˈoːɾoː."));
}

}  // namespace
}  // namespace litert::omni::tts::kokoro
