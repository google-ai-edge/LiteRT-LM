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

#include "omni/tts/kokoro/common.h"

#include <gtest/gtest.h>

namespace litert::omni::tts {
namespace {

TEST(CommonTest, ConstantsSanity) {
  EXPECT_EQ(kokoro::kMaxTokens, 256);
  EXPECT_EQ(kokoro::kSampleRate, 24000);
  EXPECT_EQ(kokoro::kAudioHop, 600);
  EXPECT_EQ(kokoro::kVoicePackSize, 510 * 256);
  EXPECT_EQ(kokoro::kBosTokenId, 0);
  EXPECT_EQ(kokoro::kEosTokenId, 0);
  EXPECT_EQ(kokoro::kSpaceTokenId, 16);
}

TEST(CommonTest, LoadVoiceEmbeddingNotFound) {
  auto embed =
      kokoro::LoadVoiceEmbedding("/non_existent_dir", "non_existent_voice");
  EXPECT_FALSE(embed.ok());
}

}  // namespace
}  // namespace litert::omni::tts
