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

#include "runtime/executor/llm_executor_settings.h"

#include <sstream>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "runtime/util/test_utils.h"  // NOLINT

namespace litert::lm {
namespace {

using absl::StatusCode::kInvalidArgument;
using ::testing::status::StatusIs;

TEST(LlmExecutorConfigTest, Backend) {
  Backend backend;
  std::stringstream oss;
  backend = Backend::CPU_ARTISAN;
  oss << backend;
  EXPECT_EQ(oss.str(), "CPU_ARTISAN");

  backend = Backend::GPU_ARTISAN;
  oss.str("");
  oss << backend;
  EXPECT_EQ(oss.str(), "GPU_ARTISAN");

  backend = Backend::GPU;
  oss.str("");
  oss << backend;
  EXPECT_EQ(oss.str(), "GPU");

  backend = Backend::CPU;
  oss.str("");
  oss << backend;
  EXPECT_EQ(oss.str(), "CPU");
}

TEST(LlmExecutorConfigTest, ActivatonDataType) {
  ActivationDataType act;
  std::stringstream oss;
  act = ActivationDataType::FLOAT32;
  oss << act;
  EXPECT_EQ(oss.str(), "FLOAT32");

  act = ActivationDataType::FLOAT16;
  oss.str("");
  oss << act;
  EXPECT_EQ(oss.str(), "FLOAT16");
}

TEST(LlmExecutorConfigTest, FakeWeightsMode) {
  FakeWeightsMode fake_weights_mode;
  std::stringstream oss;
  fake_weights_mode = FakeWeightsMode::FAKE_WEIGHTS_NONE;
  oss << fake_weights_mode;
  EXPECT_EQ(oss.str(), "FAKE_WEIGHTS_NONE");

  fake_weights_mode = FakeWeightsMode::FAKE_WEIGHTS_8BITS_ALL_LAYERS;
  oss.str("");
  oss << fake_weights_mode;
  EXPECT_EQ(oss.str(), "FAKE_WEIGHTS_8BITS_ALL_LAYERS");

  fake_weights_mode = FakeWeightsMode::FAKE_WEIGHTS_ATTN_8_FFN_4_EMB_4;
  oss.str("");
  oss << fake_weights_mode;
  EXPECT_EQ(oss.str(), "FAKE_WEIGHTS_ATTN_8_FFN_4_EMB_4");
}

TEST(LlmExecutorConfigTest, ModelAssets) {
  ModelAssets model_assets;
  std::stringstream oss;
  model_assets.model_paths = {"/path/to/model1", "/path/to/model2"};
  model_assets.fake_weights_mode = FakeWeightsMode::FAKE_WEIGHTS_NONE;
  oss << model_assets;
  const std::string expected_output = R"(model_paths:
  /path/to/model1
  /path/to/model2
fake_weights_mode: FAKE_WEIGHTS_NONE
)";
  EXPECT_EQ(oss.str(), expected_output);
}

GpuArtisanConfig CreateGpuArtisanConfig() {
  GpuArtisanConfig config;
  config.num_output_candidates = 1;
  config.wait_for_weight_uploads = true;
  config.num_decode_steps_per_sync = 3;
  config.sequence_batch_size = 16;
  config.supported_lora_ranks = {4, 16};
  config.max_top_k = 40;
  config.enable_decode_logits = true;
  return config;
}

TEST(LlmExecutorConfigTest, GpuArtisanConfig) {
  GpuArtisanConfig config = CreateGpuArtisanConfig();
  std::stringstream oss;
  oss << config;
  const std::string expected_output = R"(num_output_candidates: 1
wait_for_weight_uploads: 1
num_decode_steps_per_sync: 3
sequence_batch_size: 16
supported_lora_ranks: vector of 2 elements: [4, 16]
max_top_k: 40
enable_decode_logits: 1
)";
  EXPECT_EQ(oss.str(), expected_output);
}

TEST(LlmExecutorConfigTest, LlmExecutorSettings) {
  ModelAssets model_assets;
  model_assets.model_paths = {"/path/to/model1"};
  model_assets.fake_weights_mode = FakeWeightsMode::FAKE_WEIGHTS_NONE;
  LlmExecutorSettings config(model_assets);
  config.SetBackend(Backend::GPU_ARTISAN);
  config.SetBackendConfig(CreateGpuArtisanConfig());
  config.SetMaxNumTokens(1024);
  config.SetActivationDataType(ActivationDataType::FLOAT16);
  config.SetMaxNumImages(1);
  config.SetCacheDir("/path/to/cache");

  std::stringstream oss;
  oss << config;
  const std::string expected_output = R"(backend: GPU_ARTISAN
backend_config: num_output_candidates: 1
wait_for_weight_uploads: 1
num_decode_steps_per_sync: 3
sequence_batch_size: 16
supported_lora_ranks: vector of 2 elements: [4, 16]
max_top_k: 40
enable_decode_logits: 1

max_tokens: 1024
activation_data_type: FLOAT16
max_num_images: 1
cache_dir: /path/to/cache
model_assets: model_paths:
  /path/to/model1
fake_weights_mode: FAKE_WEIGHTS_NONE

)";
  EXPECT_EQ(oss.str(), expected_output);
}

TEST(LlmExecutorConfigTest, GetBackendConfig) {
  ModelAssets model_assets;
  LlmExecutorSettings config(model_assets);
  config.SetBackendConfig(CreateGpuArtisanConfig());

  auto gpu_config = config.GetBackendConfig<GpuArtisanConfig>();
  EXPECT_OK(gpu_config);
  EXPECT_EQ(gpu_config->num_output_candidates, 1);
  EXPECT_THAT(config.GetBackendConfig<CpuConfig>(), StatusIs(kInvalidArgument));
}

TEST(LlmExecutorConfigTest, MutableBackendConfig) {
  ModelAssets model_assets;
  LlmExecutorSettings config(model_assets);
  config.SetBackendConfig(CreateGpuArtisanConfig());

  auto gpu_config = config.MutableBackendConfig<GpuArtisanConfig>();
  EXPECT_OK(gpu_config);
  gpu_config->num_output_candidates = 2;
  config.SetBackendConfig(gpu_config.value());

  auto gpu_config_after_change = config.GetBackendConfig<GpuArtisanConfig>();
  EXPECT_EQ(gpu_config_after_change->num_output_candidates, 2);
  EXPECT_THAT(config.MutableBackendConfig<CpuConfig>(),
              StatusIs(kInvalidArgument));
}
}  // namespace
}  // namespace litert::lm
