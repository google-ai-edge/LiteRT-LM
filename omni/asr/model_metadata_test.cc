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

#include "omni/asr/model_metadata.h"

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "omni/asr/asr_engine.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep for ASSERT_OK

namespace litert::omni::asr {
namespace {

TEST(ModelMetadataTest, EmbeddedJsonIsNonEmpty) {
  EXPECT_FALSE(GetEmbeddedModelMetadataJson().empty());
}

TEST(ModelMetadataTest, LoadsSupportedModelsFromEmbeddedJson) {
  const std::vector<std::string> models = {
      "parakeet-tdt-0.6b-v3", "parakeet-ctc-0.6b", "moonshine-tiny",
      "whisper-tiny",         "qwen3-asr-0.6b",    "tinygemma-asr",
  };

  for (const auto& model_name : models) {
    ASSERT_OK_AND_ASSIGN(auto config, GetConfigFromMetadataJson(model_name));
    EXPECT_EQ(config.model_name, model_name);
    EXPECT_GT(config.input_milliseconds, 0);
    EXPECT_EQ(config.decoder_type, AsrEngineConfig::DecoderType::kUnspecified);
    EXPECT_THAT(config.model_url, ::testing::EndsWith(".litertlm"));
    EXPECT_TRUE(config.tokenizer_url.empty());
  }
}

TEST(ModelMetadataTest, ResolvesDecoderTypeForTfliteModels) {
  constexpr absl::string_view kTfliteMetadataJson = R"json({
    "parakeet-tdt-0.6b-v3": {
      "modelRemoteUrl": "https://example.com/parakeet_tdt.tflite",
      "inputMilliseconds": 5000
    },
    "parakeet-ctc-0.6b": {
      "modelRemoteUrl": "https://example.com/parakeet_ctc.tflite",
      "inputMilliseconds": 5000
    },
    "moonshine-tiny": {
      "modelRemoteUrl": "https://example.com/moonshine_tiny.tflite",
      "inputMilliseconds": 5000
    }
  })json";

  ASSERT_OK_AND_ASSIGN(
      auto tdt_config,
      GetConfigFromMetadataJson("parakeet-tdt-0.6b-v3", kTfliteMetadataJson));
  EXPECT_EQ(tdt_config.decoder_type, AsrEngineConfig::DecoderType::kTdt);

  ASSERT_OK_AND_ASSIGN(
      auto ctc_config,
      GetConfigFromMetadataJson("parakeet-ctc-0.6b", kTfliteMetadataJson));
  EXPECT_EQ(ctc_config.decoder_type, AsrEngineConfig::DecoderType::kCtc);

  ASSERT_OK_AND_ASSIGN(
      auto stateless_config,
      GetConfigFromMetadataJson("moonshine-tiny", kTfliteMetadataJson));
  EXPECT_EQ(stateless_config.decoder_type,
            AsrEngineConfig::DecoderType::kStateless);
}

TEST(ModelMetadataTest, RejectsUnknownModel) {
  EXPECT_TRUE(absl::IsNotFound(
      GetConfigFromMetadataJson("nonexistent-model").status()));
}

TEST(ModelMetadataTest, RejectsInvalidJson) {
  EXPECT_TRUE(absl::IsInvalidArgument(
      GetConfigFromMetadataJson("whisper-tiny", "{invalid json").status()));
}

}  // namespace
}  // namespace litert::omni::asr
