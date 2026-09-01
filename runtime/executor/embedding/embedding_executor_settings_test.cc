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

#include "runtime/executor/embedding/embedding_executor_settings.h"

#include <memory>
#include <sstream>
#include <string>
#include <variant>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "runtime/executor/executor_settings_base.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

using ::testing::ElementsAre;
using ::testing::HasSubstr;

TEST(EmbeddingExecutorSettingsTest, GetModelAssets) {
  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets,
                       ModelAssets::Create("/tmp/model"));
  ASSERT_OK_AND_ASSIGN(EmbeddingExecutorSettings settings,
                       EmbeddingExecutorSettings::CreateDefault(
                           model_assets, Backend::CPU));
  auto new_model_assets = settings.GetModelAssets();
  ASSERT_OK_AND_ASSIGN(auto path, new_model_assets.GetPath());
  EXPECT_EQ(path, "/tmp/model");
}

TEST(EmbeddingExecutorSettingsTest, GetAndSetBackend) {
  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets, ModelAssets::Create(""));
  ASSERT_OK_AND_ASSIGN(EmbeddingExecutorSettings settings,
                       EmbeddingExecutorSettings::CreateDefault(
                           model_assets, Backend::CPU));
  EXPECT_EQ(settings.GetBackend(), Backend::CPU);
  EXPECT_OK(settings.SetBackend(Backend::GPU));
  EXPECT_EQ(settings.GetBackend(), Backend::GPU);
}

TEST(EmbeddingExecutorSettingsTest, GetAndSetNumThreads) {
  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets, ModelAssets::Create(""));
  ASSERT_OK_AND_ASSIGN(EmbeddingExecutorSettings settings,
                       EmbeddingExecutorSettings::CreateDefault(
                           model_assets, Backend::CPU));

  EXPECT_EQ(settings.GetNumThreads(), 4);
  settings.SetNumThreads(8);
  EXPECT_EQ(settings.GetNumThreads(), 8);
}

TEST(EmbeddingExecutorSettingsTest, GetAndSetScopedFiles) {
  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets, ModelAssets::Create(""));
  ASSERT_OK_AND_ASSIGN(EmbeddingExecutorSettings settings,
                       EmbeddingExecutorSettings::CreateDefault(
                           model_assets, Backend::CPU));

  auto encoder_cache = std::make_shared<litert::lm::ScopedFile>();
  auto encoder_program = std::make_shared<litert::lm::ScopedFile>();

  settings.SetScopedEncoderCacheFile(encoder_cache);
  settings.SetScopedEncoderProgramCacheFile(encoder_program);

  EXPECT_EQ(settings.GetScopedEncoderCacheFile(), encoder_cache);
  EXPECT_EQ(settings.GetScopedEncoderProgramCacheFile(), encoder_program);
}

TEST(EmbeddingExecutorSettingsTest, GetWeightCacheFileWithScopedFile) {
  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets, ModelAssets::Create(""));
  ASSERT_OK_AND_ASSIGN(EmbeddingExecutorSettings settings,
                       EmbeddingExecutorSettings::CreateDefault(
                           model_assets, Backend::CPU));

  auto encoder_cache = std::make_shared<litert::lm::ScopedFile>();
  settings.SetScopedEncoderCacheFile(encoder_cache);

  ASSERT_OK_AND_ASSIGN(
      auto result,
      settings.GetWeightCacheFile(EmbeddingExecutorSettings::kEncoderName));
  EXPECT_EQ(std::get<std::shared_ptr<litert::lm::ScopedFile>>(result),
            encoder_cache);
}

TEST(EmbeddingExecutorSettingsTest, GetProgramCacheFileWithScopedFile) {
  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets, ModelAssets::Create(""));
  ASSERT_OK_AND_ASSIGN(EmbeddingExecutorSettings settings,
                       EmbeddingExecutorSettings::CreateDefault(
                           model_assets, Backend::CPU));

  auto encoder_program = std::make_shared<litert::lm::ScopedFile>();
  settings.SetScopedEncoderProgramCacheFile(encoder_program);

  ASSERT_OK_AND_ASSIGN(
      auto result,
      settings.GetProgramCacheFile(EmbeddingExecutorSettings::kEncoderName));
  EXPECT_EQ(std::get<std::shared_ptr<litert::lm::ScopedFile>>(result),
            encoder_program);
}

TEST(EmbeddingExecutorSettingsTest, GetWeightCacheFileFallback) {
  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets,
                       ModelAssets::Create("/tmp/model.tflite"));
  ASSERT_OK_AND_ASSIGN(EmbeddingExecutorSettings settings,
                       EmbeddingExecutorSettings::CreateDefault(
                           model_assets, Backend::CPU));
  settings.SetCacheDir("/tmp/cache");

  ASSERT_OK_AND_ASSIGN(
      auto result,
      settings.GetWeightCacheFile(EmbeddingExecutorSettings::kEncoderName));
  EXPECT_TRUE(std::holds_alternative<std::string>(result));
}

TEST(EmbeddingExecutorSettingsTest, GetProgramCacheFileFallback) {
  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets,
                       ModelAssets::Create("/tmp/model.tflite"));
  ASSERT_OK_AND_ASSIGN(EmbeddingExecutorSettings settings,
                       EmbeddingExecutorSettings::CreateDefault(
                           model_assets, Backend::GPU));
  settings.SetCacheDir("/tmp/cache");

  ASSERT_OK_AND_ASSIGN(
      auto result,
      settings.GetProgramCacheFile(EmbeddingExecutorSettings::kEncoderName));
  EXPECT_TRUE(std::holds_alternative<std::string>(result));
}

TEST(EmbeddingExecutorSettingsTest, StreamOutputFormatting) {
  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets, ModelAssets::Create(""));
  ASSERT_OK_AND_ASSIGN(EmbeddingExecutorSettings settings,
                       EmbeddingExecutorSettings::CreateDefault(
                           model_assets, Backend::CPU));

  std::ostringstream os;
  os << settings;
  EXPECT_THAT(os.str(), HasSubstr("backend: "));
}

TEST(EmbeddingExecutorSettingsTest, SelectedSignatures) {
  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets, ModelAssets::Create(""));
  ASSERT_OK_AND_ASSIGN(
      EmbeddingExecutorSettings settings,
      EmbeddingExecutorSettings::CreateDefault(model_assets, Backend::CPU));
  EXPECT_TRUE(settings.GetSelectedSignatures().empty());

  settings.SetSelectedSignatures({"encoder_256", "encoder_512"});
  EXPECT_THAT(settings.GetSelectedSignatures(),
              ElementsAre("encoder_256", "encoder_512"));
}

}  // namespace
}  // namespace litert::lm
