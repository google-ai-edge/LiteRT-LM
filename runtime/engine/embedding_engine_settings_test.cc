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

TEST(EmbeddingEngineSettingsTest, ResolveDefaultsUserSettingTakesPrecedence) {
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create("test_embedding_model.tflite"));
  ASSERT_OK_AND_ASSIGN(auto settings, EmbeddingEngineSettings::CreateDefault(
                                          model_assets, Backend::GPU));
  // User explicitly sets activation data type in code.
  settings.GetMutableMainExecutorSettings().SetActivationDataType(
      ActivationDataType::FLOAT32);

  // Even though prefer_activation_type is fp16 and backend is GPU, user setting
  // wins.
  EXPECT_OK(settings.ResolveDefaults(/*text_prefer_activation_type=*/"fp16"));
  EXPECT_EQ(settings.GetMainExecutorSettings().GetActivationDataType(),
            ActivationDataType::FLOAT32);
}

TEST(EmbeddingEngineSettingsTest,
     ResolveDefaultsPreferActivationTypeTakesPrecedenceOverGpuDefault) {
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create("test_embedding_model.tflite"));
  ASSERT_OK_AND_ASSIGN(auto settings, EmbeddingEngineSettings::CreateDefault(
                                          model_assets, Backend::GPU));

  // User did not set it in code. prefer_activation_type is set to "fp32".
  EXPECT_OK(settings.ResolveDefaults(/*text_prefer_activation_type=*/"fp32"));
  EXPECT_EQ(settings.GetMainExecutorSettings().GetActivationDataType(),
            ActivationDataType::FLOAT32);
}

TEST(EmbeddingEngineSettingsTest,
     ResolveDefaultsPreferActivationTypeMixedPrecision) {
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create("test_embedding_model.tflite"));
  ASSERT_OK_AND_ASSIGN(auto settings, EmbeddingEngineSettings::CreateDefault(
                                          model_assets, Backend::GPU));

  // prefer_activation_type "fp32_fp16" should set FLOAT32 and enable mixed
  // precision.
  EXPECT_OK(
      settings.ResolveDefaults(/*text_prefer_activation_type=*/"fp32_fp16"));
  EXPECT_EQ(settings.GetMainExecutorSettings().GetActivationDataType(),
            ActivationDataType::FLOAT32);
  EXPECT_TRUE(settings.GetMainExecutorSettings().IsMixedPrecisionEnabled());
}

TEST(EmbeddingEngineSettingsTest, ResolveDefaultsGpuFallbackToFloat16) {
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create("test_embedding_model.tflite"));
  ASSERT_OK_AND_ASSIGN(auto settings, EmbeddingEngineSettings::CreateDefault(
                                          model_assets, Backend::GPU));

  // Neither user set nor prefer_activation_type provided. Falls back to FLOAT16
  // on GPU.
  EXPECT_OK(
      settings.ResolveDefaults(/*text_prefer_activation_type=*/std::nullopt));
  EXPECT_EQ(settings.GetMainExecutorSettings().GetActivationDataType(),
            ActivationDataType::FLOAT16);
}

TEST(EmbeddingEngineSettingsTest, ResolveDefaultsCpuNoDefaultActivationType) {
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create("test_embedding_model.tflite"));
  ASSERT_OK_AND_ASSIGN(auto settings, EmbeddingEngineSettings::CreateDefault(
                                          model_assets, Backend::CPU));

  // On CPU with no prefer_activation_type, activation data type remains
  // nullopt.
  EXPECT_OK(
      settings.ResolveDefaults(/*text_prefer_activation_type=*/std::nullopt));
  EXPECT_EQ(settings.GetMainExecutorSettings().GetActivationDataType(),
            std::nullopt);
}

TEST(EmbeddingEngineSettingsTest, ResolveDefaultsMultimodal) {
  ASSERT_OK_AND_ASSIGN(
      auto model_assets,
      ModelAssets::Create("test_multimodal_embedding.litertlm"));
  ASSERT_OK_AND_ASSIGN(auto settings, EmbeddingEngineSettings::CreateDefault(
                                          model_assets, Backend::GPU,
                                          Backend::CPU, Backend::GPU));

  EXPECT_OK(settings.ResolveDefaults(
      /*text_prefer_activation_type=*/"fp16",
      /*vision_prefer_activation_type=*/"fp32",
      /*audio_prefer_activation_type=*/std::nullopt));

  EXPECT_EQ(settings.GetMainExecutorSettings().GetActivationDataType(),
            ActivationDataType::FLOAT16);
  ASSERT_TRUE(settings.GetVisionExecutorSettings().has_value());
  EXPECT_EQ(settings.GetVisionExecutorSettings()->GetActivationDataType(),
            ActivationDataType::FLOAT32);
  ASSERT_TRUE(settings.GetAudioExecutorSettings().has_value());
  EXPECT_EQ(settings.GetAudioExecutorSettings()->GetActivationDataType(),
            ActivationDataType::FLOAT16);
}

TEST(EmbeddingEngineSettingsTest, ValidateBackendConstraintSuccess) {
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create("test_embedding_model.tflite"));
  ASSERT_OK_AND_ASSIGN(auto settings, EmbeddingEngineSettings::CreateDefault(
                                          model_assets, Backend::GPU));

  EXPECT_OK(settings.Validate(/*text_backend_constraint=*/"cpu,gpu"));
}

TEST(EmbeddingEngineSettingsTest, ValidateBackendConstraintMismatch) {
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create("test_embedding_model.tflite"));
  ASSERT_OK_AND_ASSIGN(auto settings, EmbeddingEngineSettings::CreateDefault(
                                          model_assets, Backend::GPU));

  EXPECT_FALSE(settings.Validate(/*text_backend_constraint=*/"cpu").ok());
}

TEST(EmbeddingEngineSettingsTest, ValidateInvalidCacheDir) {
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create("test_embedding_model.tflite"));
  ASSERT_OK_AND_ASSIGN(auto settings, EmbeddingEngineSettings::CreateDefault(
                                          model_assets, Backend::CPU));
  settings.GetMutableMainExecutorSettings().SetCacheDir(
      "/non_existent_directory_for_test/invalid");

  EXPECT_FALSE(settings.Validate().ok());
}

}  // namespace
}  // namespace litert::lm
