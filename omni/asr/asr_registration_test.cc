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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "omni/omni_registry.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep for ASSERT_OK

namespace litert::omni::asr {
namespace {

using ::testing::UnorderedElementsAre;

TEST(AsrRegistrationTest, OnlyLinkedModelIsRegistered) {
  EXPECT_THAT(
      OmniRegistry::Get().GetRegisteredModels(),
      UnorderedElementsAre("parakeet-tdt-0.6b-v3", "parakeet-tdt-0.6b-v3-int8",
                           "parakeet-tdt", "parakeet-tdt-int8"));
  EXPECT_TRUE(OmniRegistry::Get().IsModelRegistered("parakeet-tdt-0.6b-v3"));
  EXPECT_TRUE(
      OmniRegistry::Get().IsModelRegistered("parakeet-tdt-0.6b-v3-int8"));
  EXPECT_TRUE(OmniRegistry::Get().IsModelRegistered("parakeet-tdt"));
  EXPECT_TRUE(OmniRegistry::Get().IsModelRegistered("parakeet-tdt-int8"));
  EXPECT_FALSE(OmniRegistry::Get().IsModelRegistered("parakeet-ctc-0.6b"));
  EXPECT_FALSE(OmniRegistry::Get().IsModelRegistered("moonshine-tiny"));
  EXPECT_FALSE(OmniRegistry::Get().IsModelRegistered("whisper-tiny"));
  EXPECT_FALSE(OmniRegistry::Get().IsModelRegistered("qwen3-asr-0.6b"));
  EXPECT_FALSE(OmniRegistry::Get().IsModelRegistered("tinygemma-asr"));

  ASSERT_OK_AND_ASSIGN(auto reg_model,
                       OmniRegistry::Get().GetRegisteredModel("parakeet-tdt"));
  EXPECT_EQ(reg_model.model_name, "parakeet-tdt");
  EXPECT_EQ(reg_model.canonical_model_name, "parakeet-tdt-0.6b-v3");
  EXPECT_EQ(reg_model.model_remote_url,
            "https://huggingface.co/litert-community/parakeet-tdt-0.6b-v3/"
            "resolve/main/parakeet_tdt_0.6b_v3_5s_i8_stateful.litertlm");

  EXPECT_TRUE(absl::IsNotFound(
      OmniRegistry::Get().GetRegisteredModel("whisper-tiny").status()));
}

}  // namespace
}  // namespace litert::omni::asr
