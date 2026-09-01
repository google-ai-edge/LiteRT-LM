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

#include "runtime/engine/embedding_engine_settings.h"

#include <optional>
#include <sstream>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "runtime/executor/embedding/embedding_executor_settings.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/proto/embedding_metadata.pb.h"
#include "runtime/proto/embedding_model_type.pb.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

using ::testing::Eq;
using ::testing::HasSubstr;

TEST(EmbeddingEngineSettingsTest, CreateDefaultBasic) {
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create("test_embedding_model.tflite"));
  ASSERT_OK_AND_ASSIGN(auto settings, EmbeddingEngineSettings::CreateDefault(
                                          model_assets, Backend::CPU));

  EXPECT_THAT(settings.GetMainExecutorSettings().GetBackend(),
              Eq(Backend::CPU));
  EXPECT_FALSE(settings.GetVisionExecutorSettings().has_value());
  EXPECT_FALSE(settings.GetAudioExecutorSettings().has_value());
}

TEST(EmbeddingEngineSettingsTest, CreateDefaultMultimodal) {
  ASSERT_OK_AND_ASSIGN(
      auto model_assets,
      ModelAssets::Create("test_multimodal_embedding.litertlm"));
  ASSERT_OK_AND_ASSIGN(auto settings, EmbeddingEngineSettings::CreateDefault(
                                          model_assets, Backend::CPU,
                                          Backend::CPU, Backend::CPU));

  EXPECT_THAT(settings.GetMainExecutorSettings().GetBackend(),
              Eq(Backend::CPU));
  EXPECT_TRUE(settings.GetVisionExecutorSettings().has_value());
  EXPECT_TRUE(settings.GetAudioExecutorSettings().has_value());
}

TEST(EmbeddingEngineSettingsTest, BenchmarkSettings) {
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create("test_embedding_model.tflite"));
  ASSERT_OK_AND_ASSIGN(auto settings, EmbeddingEngineSettings::CreateDefault(
                                          model_assets, Backend::CPU));

  EXPECT_FALSE(settings.IsBenchmarkEnabled());
  EXPECT_FALSE(settings.GetBenchmarkParams().has_value());

  settings.GetMutableBenchmarkParams();

  EXPECT_TRUE(settings.IsBenchmarkEnabled());
  EXPECT_TRUE(settings.GetBenchmarkParams().has_value());
}

TEST(EmbeddingEngineSettingsTest, StreamOutputFormatting) {
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create("test_embedding_model.tflite"));
  ASSERT_OK_AND_ASSIGN(auto settings, EmbeddingEngineSettings::CreateDefault(
                                          model_assets, Backend::CPU));

  std::ostringstream os;
  os << settings;
  EXPECT_THAT(os.str(), HasSubstr("EmbeddingEngineSettings:"));
}

TEST(EmbeddingEngineSettingsTest, ModifyMainExecutorSettings) {
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create("test_embedding_model.tflite"));
  ASSERT_OK_AND_ASSIGN(auto settings, EmbeddingEngineSettings::CreateDefault(
                                          model_assets, Backend::CPU));

  EXPECT_EQ(settings.GetMainExecutorSettings().GetNumThreads(), 4);
  settings.GetMutableMainExecutorSettings().SetNumThreads(8);
  EXPECT_EQ(settings.GetMainExecutorSettings().GetNumThreads(), 8);

  settings.GetMutableMainExecutorSettings().SetCacheDir("/tmp/cache");
  EXPECT_EQ(settings.GetMainExecutorSettings().GetCacheDir(), "/tmp/cache");
}

TEST(EmbeddingEngineSettingsTest, MaxInputLengthGetterAndSetter) {
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create("test_embedding_model.tflite"));
  ASSERT_OK_AND_ASSIGN(auto settings, EmbeddingEngineSettings::CreateDefault(
                                          model_assets, Backend::CPU));

  EXPECT_EQ(settings.GetMaxInputLength(), std::nullopt);
  settings.SetMaxInputLength(512);
  EXPECT_EQ(settings.GetMaxInputLength(), 512);
}

TEST(EmbeddingEngineSettingsTest, VisionTokensPerImageGetterAndSetter) {
  ASSERT_OK_AND_ASSIGN(
      auto model_assets,
      ModelAssets::Create("test_multimodal_embedding.litertlm"));
  ASSERT_OK_AND_ASSIGN(auto settings, EmbeddingEngineSettings::CreateDefault(
                                          model_assets, Backend::CPU,
                                          Backend::CPU, std::nullopt));

  EXPECT_EQ(settings.GetVisionTokensPerImage(), std::nullopt);
  settings.SetVisionTokensPerImage(70);
  EXPECT_EQ(settings.GetVisionTokensPerImage(), 70);
}

}  // namespace
}  // namespace litert::lm
