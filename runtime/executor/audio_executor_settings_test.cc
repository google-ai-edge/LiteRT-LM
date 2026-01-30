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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "runtime/executor/executor_settings_base.h"
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

}  // namespace
}  // namespace litert::lm
