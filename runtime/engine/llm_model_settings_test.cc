#include "third_party/odml/litert_lm/runtime/engine/llm_model_settings.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "third_party/odml/litert_lm/runtime/executor/llm_executor_config.h"

namespace litert::lm {
namespace {

using ::litert::lm::LlmExecutorConfig;
using ::litert::lm::LlmModelSettings;
using ::testing::Eq;

class LlmModelSettingsTest : public ::testing::Test {
};

TEST_F(LlmModelSettingsTest, GetModelPath) {
  ModelAssets model_assets;
  model_assets.model_paths.push_back("test_model_path_1");
  LlmExecutorConfig executor_settings(model_assets);
  LlmModelSettings settings(executor_settings);

  EXPECT_EQ(settings.GetMainExecutorSettings().GetModelAssets().model_paths[0],
            "test_model_path_1");
}

TEST_F(LlmModelSettingsTest, SetAndGetCacheDir) {
  ModelAssets model_assets;
  model_assets.model_paths.push_back("test_model_path_1");
  LlmExecutorConfig executor_settings(model_assets);
  executor_settings.SetCacheDir("test_cache_dir");
  LlmModelSettings settings(executor_settings);
  EXPECT_EQ(settings.GetMainExecutorSettings().GetCacheDir(), "test_cache_dir");
}

TEST_F(LlmModelSettingsTest, SetAndGetMaxNumTokens) {
  ModelAssets model_assets;
  model_assets.model_paths.push_back("test_model_path_1");
  LlmExecutorConfig executor_settings(model_assets);
  executor_settings.SetMaxNumTokens(128);
  LlmModelSettings settings(executor_settings);
  EXPECT_EQ(settings.GetMainExecutorSettings().GetMaxNumTokens(), 128);
}

TEST_F(LlmModelSettingsTest, SetAndGetExecutorBackend) {
  ModelAssets model_assets;
  model_assets.model_paths.push_back("test_model_path_1");
  LlmExecutorConfig executor_settings(model_assets);
  executor_settings.SetBackend(Backend::GPU);
  LlmModelSettings settings(executor_settings);
  EXPECT_THAT(settings.GetMainExecutorSettings().GetBackend(),
              Eq(Backend::GPU));
}

TEST_F(LlmModelSettingsTest, DefaultExecutorBackend) {
  ModelAssets model_assets;
  model_assets.model_paths.push_back("test_model_path_1");
  LlmExecutorConfig executor_settings(model_assets);
  LlmModelSettings settings(executor_settings);
  EXPECT_THAT(settings.GetMainExecutorSettings().GetBackend(),
              Eq(Backend::CPU));
}

}  // namespace
}  // namespace litert::lm
