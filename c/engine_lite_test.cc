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

#include <memory>
#include <string>

#include <gtest/gtest.h>
#include "c/engine.h"
#include "c/engine_internal.h"
#include "c/error_reporter.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/llm_executor_settings.h"

namespace {

using EngineSettingsPtr =
    std::unique_ptr<LiteRtLmEngineSettings,
                    decltype(&litert_lm_engine_settings_delete)>;
using EnginePtr =
    std::unique_ptr<LiteRtLmEngine, decltype(&litert_lm_engine_delete)>;
using SessionPtr =
    std::unique_ptr<LiteRtLmSession, decltype(&litert_lm_session_delete)>;
using ResponsesPtr =
    std::unique_ptr<LiteRtLmResponses, decltype(&litert_lm_responses_delete)>;
using InputDataPtr =
    std::unique_ptr<LiteRtLmInputData, decltype(&litert_lm_input_data_delete)>;
using SessionConfigPtr =
    std::unique_ptr<LiteRtLmSessionConfig,
                    decltype(&litert_lm_session_config_delete)>;
using SamplerParamsPtr =
    std::unique_ptr<LiteRtLmSamplerParams,
                    decltype(&litert_lm_sampler_params_delete)>;

// Creates engine settings through the status + out-parameter C API and returns
// them (or NULL on failure) wrapped in a smart pointer.
EngineSettingsPtr CreateEngineSettings(const char* model_path,
                                       const char* backend_str,
                                       const char* vision_backend_str,
                                       const char* audio_backend_str) {
  LiteRtLmEngineSettings* settings = nullptr;
  EXPECT_EQ(litert_lm_engine_settings_create(model_path, backend_str,
                                             vision_backend_str,
                                             audio_backend_str, &settings),
            kLiteRtLmStatusOk);
  return EngineSettingsPtr(settings, &litert_lm_engine_settings_delete);
}

TEST(EngineLiteTest, CreateSettingsWithNoVisionAndAudioBackend) {
  const std::string task_path = "test_model_path_1";
  EngineSettingsPtr settings =
      CreateEngineSettings(task_path.c_str(), "cpu",
                           /* vision_backend_str */ nullptr,
                           /* audio_backend_str */ nullptr);
  ASSERT_NE(settings, nullptr);
  EXPECT_FALSE(settings->settings->GetVisionExecutorSettings().has_value());
  EXPECT_FALSE(settings->settings->GetAudioExecutorSettings().has_value());
}

TEST(EngineLiteTest, CreateSettingsWithVisionAndAudioBackend) {
  const std::string task_path = "test_model_path_1";
  EngineSettingsPtr settings =
      CreateEngineSettings(task_path.c_str(), "cpu",
                           /* vision_backend_str */ "gpu",
                           /* audio_backend_str */ "cpu");
  ASSERT_NE(settings, nullptr);
  EXPECT_TRUE(settings->settings->GetVisionExecutorSettings().has_value());
  EXPECT_TRUE(settings->settings->GetAudioExecutorSettings().has_value());
  EXPECT_EQ(settings->settings->GetVisionExecutorSettings()->GetBackend(),
            litert::lm::Backend::GPU);
  EXPECT_EQ(settings->settings->GetAudioExecutorSettings()->GetBackend(),
            litert::lm::Backend::CPU);
}

TEST(EngineLiteTest, SetCacheDir) {
  const std::string task_path = "test_model_path_1";
  EngineSettingsPtr settings =
      CreateEngineSettings(task_path.c_str(), "cpu",
                           /* vision_backend_str */ nullptr,
                           /* audio_backend_str */ nullptr);
  ASSERT_NE(settings, nullptr);
  const std::string cache_dir = "test_cache_dir";
  litert_lm_engine_settings_set_cache_dir(settings.get(), cache_dir.c_str());
  EXPECT_EQ(settings->settings->GetMainExecutorSettings().GetCacheDir(),
            cache_dir);
}

TEST(EngineLiteTest, SamplerParamsCreateAndDelete) {
  LiteRtLmSamplerParams* raw_params = nullptr;
  ASSERT_EQ(
      litert_lm_sampler_params_create(kLiteRtLmSamplerTypeTopK, &raw_params),
      kLiteRtLmStatusOk);
  SamplerParamsPtr params(raw_params, &litert_lm_sampler_params_delete);
  ASSERT_NE(params, nullptr);
  EXPECT_EQ(params->type, kLiteRtLmSamplerTypeTopK);
  litert_lm_sampler_params_set_top_k(params.get(), 50);
  EXPECT_EQ(params->top_k, 50);
}

}  // namespace
