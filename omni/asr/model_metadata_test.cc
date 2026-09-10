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

  for (const auto& model : models) {
    ASSERT_OK_AND_ASSIGN(auto config, GetConfigFromMetadataJson(model));
    EXPECT_EQ(config.model_name, model);
    EXPECT_GT(config.input_milliseconds, 0);
  }
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
