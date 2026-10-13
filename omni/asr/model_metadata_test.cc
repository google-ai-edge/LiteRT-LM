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
#include "support/util/test_utils.h"  // IWYU pragma: keep for ASSERT_OK

namespace litert::omni::asr {
namespace {

TEST(ModelMetadataTest, PopulatesSupportedModelsFromRegistry) {
  const std::vector<std::string> models = {
      "parakeet-tdt-0.6b-v3", "parakeet-ctc-0.6b", "moonshine-tiny",
      "whisper-tiny",         "qwen3-asr-0.6b",    "tinygemma-asr",
  };

  for (const auto& model_name : models) {
    AsrEngineConfig config;
    ASSERT_OK(PopulateConfigFromMetadataJson(model_name, "{}", config));
    EXPECT_EQ(config.metadata.sample_rate_hz(), 16000);
    EXPECT_EQ(config.metadata.input_milliseconds(),
              model_name == "whisper-tiny" ? 30000 : 5000);
    EXPECT_THAT(config.model_url, ::testing::EndsWith(".litertlm"));
    EXPECT_TRUE(config.tokenizer_url.empty());
  }
}

TEST(ModelMetadataTest, PreservesCallerOverrides) {
  AsrEngineConfig config;
  config.model_url = "https://example.com/custom.tflite";
  config.metadata.set_sample_rate_hz(24000);
  config.metadata.set_input_milliseconds(10000);

  ASSERT_OK(
      PopulateConfigFromMetadataJson("parakeet-tdt-0.6b-v3", "{}", config));
  EXPECT_EQ(config.model_url, "https://example.com/custom.tflite");
  EXPECT_EQ(config.metadata.sample_rate_hz(), 24000);
  EXPECT_EQ(config.metadata.input_milliseconds(), 10000);
}

TEST(ModelMetadataTest, RejectsUnknownModel) {
  AsrEngineConfig config;
  EXPECT_TRUE(absl::IsNotFound(
      PopulateConfigFromMetadataJson("nonexistent-model", "{}", config)));
}

}  // namespace
}  // namespace litert::omni::asr
