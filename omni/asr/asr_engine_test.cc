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

#include "omni/asr/asr_engine.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_options.h"  // from @litert
#include "omni/asr/model_metadata.h"
#include "omni/omni_engine.h"
#include "omni/omni_registry.h"
#include "omni/omni_session.h"
#include "runtime/components/model_resources.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep for ASSERT_OK

namespace litert::omni::asr {
namespace {

using ::testing::IsSupersetOf;

TEST(AsrEngineRegistrationTest, AllDefaultModelsAreRegisteredWithUmbrellaDep) {
  EXPECT_THAT(OmniRegistry::Get().GetRegisteredModels(),
              IsSupersetOf({"parakeet-tdt-0.6b-v3",
                            "parakeet-tdt-0.6b-v3-int8",
                            "parakeet-tdt",
                            "parakeet-tdt-int8",
                            "parakeet-ctc-0.6b",
                            "parakeet-ctc-0.6b-int8",
                            "parakeet-ctc",
                            "parakeet-ctc-int8",
                            "moonshine-tiny",
                            "moonshine-tiny-int8",
                            "moonshine",
                            "moonshine-int8",
                            "whisper-tiny",
                            "whisper-tiny-int8",
                            "whisper",
                            "whisper-int8",
                            "qwen3-asr-0.6b",
                            "qwen3-asr-0.6b-int8",
                            "qwen3-asr",
                            "qwen3-asr-int8",
                            "tinygemma-asr",
                            "tinygemma-asr-int8"}));

  EXPECT_TRUE(OmniRegistry::Get().IsModelRegistered("parakeet-tdt-0.6b-v3"));
  EXPECT_TRUE(OmniRegistry::Get().IsModelRegistered("parakeet-ctc-0.6b"));
  EXPECT_TRUE(OmniRegistry::Get().IsModelRegistered("moonshine-tiny"));
  EXPECT_TRUE(OmniRegistry::Get().IsModelRegistered("whisper-tiny"));
  EXPECT_TRUE(OmniRegistry::Get().IsModelRegistered("qwen3-asr-0.6b"));
  EXPECT_TRUE(OmniRegistry::Get().IsModelRegistered("tinygemma-asr"));
  EXPECT_FALSE(OmniRegistry::Get().IsModelRegistered("unknown-model"));
}

TEST(AsrEngineRegistrationTest, PopulatesConfigForRegisteredModels) {
  AsrEngineConfig tdt_config;
  ASSERT_OK(
      PopulateConfigFromMetadataJson("parakeet-tdt-0.6b-v3", "{}", tdt_config));
  EXPECT_EQ(tdt_config.metadata.sample_rate_hz(), 16000);
  EXPECT_EQ(tdt_config.metadata.input_milliseconds(), 5000);

  AsrEngineConfig whisper_config;
  ASSERT_OK(PopulateConfigFromMetadataJson("whisper", "{}", whisper_config));
  EXPECT_EQ(whisper_config.metadata.sample_rate_hz(), 16000);
  EXPECT_EQ(whisper_config.metadata.input_milliseconds(), 30000);
}

TEST(AsrEngineRegistrationTest, RejectsUnknownModelOnCreate) {
  EXPECT_TRUE(
      absl::IsNotFound(AsrEngine::Create("unknown-model", {}).status()));
}

TEST(AsrEngineRegistrationTest, DynamicRegisterAndUnregisterModel) {
  ASSERT_OK(
      OmniRegistry::Get().RegisterSessionFactory(OmniRegistry::RegistrySpec{
          .model_names = {"custom-test-asr-ctc"},
          .get_session_factory =
              [](absl::string_view /*model_name*/,
                 const OmniEngine::Settings& /*settings*/,
                 const MessageLite* absl_nullable /*session_config*/,
                 const MessageLite* absl_nullable /*metadata*/,
                 ::litert::Environment& /*env*/, ::litert::Options& /*options*/,
                 std::shared_ptr<::litert::lm::ModelResources> absl_nullable
                 /*lm_resources*/)
              -> absl::StatusOr<std::unique_ptr<OmniSessionFactory>> {
            return absl::UnimplementedError("Custom session factory invoked");
          },
          .session_config_type_id = "Asr",
      }));
  EXPECT_TRUE(OmniRegistry::Get().IsModelRegistered("custom-test-asr-ctc"));

  EXPECT_TRUE(OmniRegistry::Get().UnregisterModel("custom-test-asr-ctc"));
  EXPECT_FALSE(OmniRegistry::Get().IsModelRegistered("custom-test-asr-ctc"));
}

}  // namespace
}  // namespace litert::omni::asr
