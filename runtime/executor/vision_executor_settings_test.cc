// Copyright 2025 The ODML Authors.
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

#include "runtime/executor/vision_executor_settings.h"
#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "runtime/executor/executor_settings_base.h"
#include "litert/cc/internal/scoped_file.h"  // from @litert
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

using ::testing::status::StatusIs;

TEST(VisionExecutorSettingsTest, GetModelAssets) {
  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets, ModelAssets::Create("/tmp"));
  ASSERT_OK_AND_ASSIGN(
      VisionExecutorSettings settings,
      VisionExecutorSettings::CreateDefault(model_assets,
                                            /*encoder_backend=*/Backend::GPU,
                                            /*adapter_backend=*/Backend::GPU));
  auto new_model_assets = settings.GetModelAssets();
  ASSERT_OK_AND_ASSIGN(auto path, new_model_assets.GetPath());
  EXPECT_EQ(path, "/tmp");
}

TEST(VisionExecutorSettingsTest, GetAndSetEncoderBackend) {
  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets, ModelAssets::Create(""));
  ASSERT_OK_AND_ASSIGN(
      VisionExecutorSettings settings,
      VisionExecutorSettings::CreateDefault(model_assets,
                                            /*encoder_backend=*/Backend::GPU,
                                            /*adapter_backend=*/Backend::GPU));
  EXPECT_EQ(settings.GetEncoderBackend(), Backend::GPU);
  EXPECT_OK(settings.SetEncoderBackend(Backend::CPU));
  EXPECT_EQ(settings.GetEncoderBackend(), Backend::CPU);
}

TEST(VisionExecutorSettingsTest, GetAndSetAdapterBackend) {
  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets, ModelAssets::Create(""));
  ASSERT_OK_AND_ASSIGN(
      VisionExecutorSettings settings,
      VisionExecutorSettings::CreateDefault(model_assets,
                                            /*encoder_backend=*/Backend::GPU,
                                            /*adapter_backend=*/Backend::GPU));
  EXPECT_EQ(settings.GetAdapterBackend(), Backend::GPU);
  EXPECT_OK(settings.SetAdapterBackend(Backend::CPU));
  EXPECT_EQ(settings.GetAdapterBackend(), Backend::CPU);
}

TEST(VisionExecutorSettingsTest, CreateDefaultWithInvalidBackend) {
  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets, ModelAssets::Create(""));
  // Vision encoder supports GPU, CPU and NPU backends.
  EXPECT_THAT(
      VisionExecutorSettings::CreateDefault(model_assets,
                                            /*encoder_backend=*/Backend::CPU,
                                            /*adapter_backend=*/Backend::NPU),
      StatusIs(absl::StatusCode::kInvalidArgument,
               "Unsupported adapter backend: 6"));
  EXPECT_THAT(
      VisionExecutorSettings::CreateDefault(model_assets,
                                            /*encoder_backend=*/Backend::GPU,
                                            /*adapter_backend=*/Backend::NPU),
      StatusIs(absl::StatusCode::kInvalidArgument,
               "Unsupported adapter backend: 6"));
  EXPECT_THAT(VisionExecutorSettings::CreateDefault(
                  model_assets, /*encoder_backend=*/Backend::GPU_ARTISAN,
                  /*adapter_backend=*/Backend::GPU),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "Unsupported encoder backend: 2"));
  EXPECT_THAT(VisionExecutorSettings::CreateDefault(
                  model_assets, /*encoder_backend=*/Backend::CPU,
                  /*adapter_backend=*/Backend::CPU_ARTISAN),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "Unsupported adapter backend: 1"));
};

TEST(VisionExecutorSettingsTest, CreateDefaultWithValidBackend) {
  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets, ModelAssets::Create(""));
  // Valid combinations.
  EXPECT_OK(VisionExecutorSettings::CreateDefault(model_assets, Backend::CPU,
                                                  Backend::GPU));
  EXPECT_OK(VisionExecutorSettings::CreateDefault(model_assets, Backend::GPU,
                                                  Backend::CPU));
  EXPECT_OK(VisionExecutorSettings::CreateDefault(model_assets, Backend::CPU,
                                                  Backend::CPU));
  EXPECT_OK(VisionExecutorSettings::CreateDefault(model_assets, Backend::GPU,
                                                  Backend::GPU));
  EXPECT_OK(VisionExecutorSettings::CreateDefault(model_assets, Backend::NPU,
                                                  Backend::GPU));
  EXPECT_OK(VisionExecutorSettings::CreateDefault(model_assets, Backend::NPU,
                                                  Backend::CPU));
}

TEST(VisionExecutorSettingsTest, GetAndSetScopedFiles) {
  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets, ModelAssets::Create(""));
  ASSERT_OK_AND_ASSIGN(
      VisionExecutorSettings settings,
      VisionExecutorSettings::CreateDefault(model_assets, Backend::GPU,
                                            Backend::GPU));

  auto encoder_cache = std::make_shared<litert::ScopedFile>();
  auto adapter_cache = std::make_shared<litert::ScopedFile>();
  auto encoder_program = std::make_shared<litert::ScopedFile>();
  auto adapter_program = std::make_shared<litert::ScopedFile>();

  settings.SetScopedEncoderCacheFile(encoder_cache);
  settings.SetScopedAdapterCacheFile(adapter_cache);
  settings.SetScopedEncoderProgramCacheFile(encoder_program);
  settings.SetScopedAdapterProgramCacheFile(adapter_program);

  EXPECT_EQ(settings.GetScopedEncoderCacheFile(), encoder_cache);
  EXPECT_EQ(settings.GetScopedAdapterCacheFile(), adapter_cache);
  EXPECT_EQ(settings.GetScopedEncoderProgramCacheFile(), encoder_program);
  EXPECT_EQ(settings.GetScopedAdapterProgramCacheFile(), adapter_program);
}

TEST(VisionExecutorSettingsTest, GetWeightCacheFile) {
  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets, ModelAssets::Create(""));
  ASSERT_OK_AND_ASSIGN(
      VisionExecutorSettings settings,
      VisionExecutorSettings::CreateDefault(model_assets, Backend::GPU,
                                            Backend::GPU));

  auto encoder_cache = std::make_shared<litert::ScopedFile>();
  auto adapter_cache = std::make_shared<litert::ScopedFile>();

  settings.SetScopedEncoderCacheFile(encoder_cache);
  settings.SetScopedAdapterCacheFile(adapter_cache);

  ASSERT_OK_AND_ASSIGN(
      auto result1,
      settings.GetWeightCacheFile(VisionExecutorSettings::kEncoderName));
  EXPECT_EQ(std::get<std::shared_ptr<litert::ScopedFile>>(result1),
            encoder_cache);

  ASSERT_OK_AND_ASSIGN(
      auto result2,
      settings.GetWeightCacheFile(VisionExecutorSettings::kAdapterName));
  EXPECT_EQ(std::get<std::shared_ptr<litert::ScopedFile>>(result2),
            adapter_cache);
}

TEST(VisionExecutorSettingsTest, GetProgramCacheFile) {
  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets, ModelAssets::Create(""));
  ASSERT_OK_AND_ASSIGN(
      VisionExecutorSettings settings,
      VisionExecutorSettings::CreateDefault(model_assets, Backend::GPU,
                                            Backend::GPU));

  auto encoder_program = std::make_shared<litert::ScopedFile>();
  auto adapter_program = std::make_shared<litert::ScopedFile>();

  settings.SetScopedEncoderProgramCacheFile(encoder_program);
  settings.SetScopedAdapterProgramCacheFile(adapter_program);

  ASSERT_OK_AND_ASSIGN(
      auto result1,
      settings.GetProgramCacheFile(VisionExecutorSettings::kEncoderName));
  EXPECT_EQ(std::get<std::shared_ptr<litert::ScopedFile>>(result1),
            encoder_program);

  ASSERT_OK_AND_ASSIGN(
      auto result2,
      settings.GetProgramCacheFile(VisionExecutorSettings::kAdapterName));
  EXPECT_EQ(std::get<std::shared_ptr<litert::ScopedFile>>(result2),
            adapter_program);
}


}  // namespace
}  // namespace litert::lm

