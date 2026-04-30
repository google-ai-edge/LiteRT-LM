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

#include "runtime/executor/audio_executor_settings.h"
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

TEST(AudioExecutorSettingsTest, GetModelAssets) {
  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets, ModelAssets::Create("/tmp"));
  ASSERT_OK_AND_ASSIGN(AudioExecutorSettings settings,
                       AudioExecutorSettings::CreateDefault(
                           model_assets, 10, Backend::GPU_ARTISAN));
  auto new_model_assets = settings.GetModelAssets();
  ASSERT_OK_AND_ASSIGN(auto path, new_model_assets.GetPath());
  EXPECT_EQ(path, "/tmp");
}

TEST(AudioExecutorSettingsTest, GetAndSetMaxSequenceLength) {
  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets, ModelAssets::Create(""));
  ASSERT_OK_AND_ASSIGN(AudioExecutorSettings settings,
                       AudioExecutorSettings::CreateDefault(
                           model_assets, 10, Backend::GPU_ARTISAN));
  EXPECT_EQ(settings.GetMaxSequenceLength(), 10);
  settings.SetMaxSequenceLength(20);
  EXPECT_EQ(settings.GetMaxSequenceLength(), 20);
}

TEST(AudioExecutorSettingsTest, GetAndSetBackend) {
  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets, ModelAssets::Create(""));
  ASSERT_OK_AND_ASSIGN(AudioExecutorSettings settings,
                       AudioExecutorSettings::CreateDefault(
                           model_assets, 10, Backend::GPU_ARTISAN));
  EXPECT_EQ(settings.GetBackend(), Backend::GPU_ARTISAN);
  EXPECT_OK(settings.SetBackend(Backend::GPU_ARTISAN));
  EXPECT_EQ(settings.GetBackend(), Backend::GPU_ARTISAN);
}

TEST(AudioExecutorSettingsTest, GetAndSetNumThreads) {
  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets, ModelAssets::Create(""));
  ASSERT_OK_AND_ASSIGN(
      AudioExecutorSettings settings,
      AudioExecutorSettings::CreateDefault(model_assets, 10, Backend::CPU));
  EXPECT_EQ(settings.GetNumThreads(), 4);
  settings.SetNumThreads(8);
  EXPECT_EQ(settings.GetNumThreads(), 8);
}

TEST(AudioExecutorSettingsTest, CreateDefaultWithInvalidBackend) {
  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets, ModelAssets::Create(""));
  EXPECT_THAT(AudioExecutorSettings::CreateDefault(model_assets, 10,
                                                   Backend::CPU_ARTISAN),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(AudioExecutorSettings::CreateDefault(
                  model_assets, 10, Backend::GOOGLE_TENSOR_ARTISAN),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(
      AudioExecutorSettings::CreateDefault(model_assets, 10, Backend::NPU),
      StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(AudioExecutorSettingsTest, GetAndSetBundledWithMainModel) {
  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets, ModelAssets::Create(""));
  ASSERT_OK_AND_ASSIGN(AudioExecutorSettings settings,
                       AudioExecutorSettings::CreateDefault(
                           model_assets, 10, Backend::GPU_ARTISAN));
  EXPECT_TRUE(settings.GetBundledWithMainModel());
  settings.SetBundledWithMainModel(false);
  EXPECT_FALSE(settings.GetBundledWithMainModel());
}

TEST(AudioExecutorSettingsTest, GetAndSetScopedFiles) {
  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets, ModelAssets::Create(""));
  ASSERT_OK_AND_ASSIGN(AudioExecutorSettings settings,
                       AudioExecutorSettings::CreateDefault(
                           model_assets, 10, Backend::GPU_ARTISAN));

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

TEST(AudioExecutorSettingsTest, GetWeightCacheFile) {
  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets, ModelAssets::Create(""));
  ASSERT_OK_AND_ASSIGN(AudioExecutorSettings settings,
                       AudioExecutorSettings::CreateDefault(
                           model_assets, 10, Backend::GPU_ARTISAN));

  auto encoder_cache = std::make_shared<litert::ScopedFile>();
  auto adapter_cache = std::make_shared<litert::ScopedFile>();

  settings.SetScopedEncoderCacheFile(encoder_cache);
  settings.SetScopedAdapterCacheFile(adapter_cache);

  ASSERT_OK_AND_ASSIGN(
      auto result1,
      settings.GetWeightCacheFile(AudioExecutorSettings::kEncoderName));
  EXPECT_EQ(std::get<std::shared_ptr<litert::ScopedFile>>(result1),
            encoder_cache);

  ASSERT_OK_AND_ASSIGN(
      auto result2,
      settings.GetWeightCacheFile(AudioExecutorSettings::kAdapterName));
  EXPECT_EQ(std::get<std::shared_ptr<litert::ScopedFile>>(result2),
            adapter_cache);
}

TEST(AudioExecutorSettingsTest, GetProgramCacheFile) {
  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets, ModelAssets::Create(""));
  ASSERT_OK_AND_ASSIGN(AudioExecutorSettings settings,
                       AudioExecutorSettings::CreateDefault(
                           model_assets, 10, Backend::GPU_ARTISAN));

  auto encoder_program = std::make_shared<litert::ScopedFile>();
  auto adapter_program = std::make_shared<litert::ScopedFile>();

  settings.SetScopedEncoderProgramCacheFile(encoder_program);
  settings.SetScopedAdapterProgramCacheFile(adapter_program);

  ASSERT_OK_AND_ASSIGN(
      auto result1,
      settings.GetProgramCacheFile(AudioExecutorSettings::kEncoderName));
  EXPECT_EQ(std::get<std::shared_ptr<litert::ScopedFile>>(result1),
            encoder_program);

  ASSERT_OK_AND_ASSIGN(
      auto result2,
      settings.GetProgramCacheFile(AudioExecutorSettings::kAdapterName));
  EXPECT_EQ(std::get<std::shared_ptr<litert::ScopedFile>>(result2),
            adapter_program);
}

}  // namespace
}  // namespace litert::lm

