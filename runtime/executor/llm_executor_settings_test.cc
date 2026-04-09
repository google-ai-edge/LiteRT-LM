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

#include <filesystem>  // NOLINT: Required for path manipulation.
#include <memory>
#include <sstream>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/executor/executor_settings_base.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

#if defined(_WIN32)
constexpr absl::string_view kPathToModel1 = "\\path\\to\\model1";
constexpr absl::string_view kPathToModel1Tflite = "\\path\\to\\model1.tflite";
constexpr absl::string_view kPathToModel1TfliteCache =
    "\\path\\to\\model1.tflite.cache";
constexpr absl::string_view kPathToCache = "\\path\\to\\cache";
constexpr absl::string_view kWeightCachePath = "\\weight\\cache\\path";
constexpr absl::string_view kWeightCachePathWithSeparator =
    "\\weight\\cache\\path\\";
constexpr absl::string_view kWeightCachePathFile =
    "\\weight\\cache\\path\\model1.tflite.cache";
constexpr absl::string_view kWeightCachePathXnnpackFile =
    "\\weight\\cache\\path\\model1.tflite.xnnpack_cache";
constexpr absl::string_view kModel1TfliteCustomSuffix =
    "\\path\\to\\model1.tflite.custom_suffix";
#else
constexpr absl::string_view kPathToModel1 = "/path/to/model1";
constexpr absl::string_view kPathToModel1Tflite = "/path/to/model1.tflite";
constexpr absl::string_view kPathToModel1TfliteCache =
    "/path/to/model1.tflite.cache";
constexpr absl::string_view kPathToCache = "/path/to/cache";
constexpr absl::string_view kWeightCachePath = "/weight/cache/path";
constexpr absl::string_view kWeightCachePathWithSeparator =
    "/weight/cache/path/";
constexpr absl::string_view kWeightCachePathFile =
    "/weight/cache/path/model1.tflite.cache";
constexpr absl::string_view kWeightCachePathXnnpackFile =
    "/weight/cache/path/model1.tflite.xnnpack_cache";
constexpr absl::string_view kModel1TfliteCustomSuffix =
    "/path/to/model1.tflite.custom_suffix";
#endif

using absl::StatusCode::kInvalidArgument;
using ::testing::VariantWith;
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

  backend = Backend::GOOGLE_TENSOR_ARTISAN;
  oss.str("");
  oss << backend;
  EXPECT_EQ(oss.str(), "GOOGLE_TENSOR_ARTISAN");

  backend = Backend::NPU;
  oss.str("");
  oss << backend;
  EXPECT_EQ(oss.str(), "NPU");
}

TEST(LlmExecutorConfigTest, StringToBackend) {
  ASSERT_OK_AND_ASSIGN(auto backend, GetBackendFromString("cpu_artisan"));
  EXPECT_EQ(backend, Backend::CPU_ARTISAN);
  ASSERT_OK_AND_ASSIGN(backend, GetBackendFromString("gpu_artisan"));
  EXPECT_EQ(backend, Backend::GPU_ARTISAN);
  ASSERT_OK_AND_ASSIGN(backend, GetBackendFromString("gpu"));
  EXPECT_EQ(backend, Backend::GPU);
  ASSERT_OK_AND_ASSIGN(backend, GetBackendFromString("cpu"));
  EXPECT_EQ(backend, Backend::CPU);
  ASSERT_OK_AND_ASSIGN(backend, GetBackendFromString("google_tensor_artisan"));
  EXPECT_EQ(backend, Backend::GOOGLE_TENSOR_ARTISAN);
  ASSERT_OK_AND_ASSIGN(backend, GetBackendFromString("npu"));
  EXPECT_EQ(backend, Backend::NPU);
}

TEST(LlmExecutorConfigTest, ActivationDataType) {
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

TEST(LlmExecutorConfigTest, FileFormat) {
  std::stringstream oss;

  oss.str("");
  oss << FileFormat::TFLITE;
  EXPECT_EQ(oss.str(), "TFLITE");

  oss.str("");
  oss << FileFormat::TASK;
  EXPECT_EQ(oss.str(), "TASK");

  oss.str("");
  oss << FileFormat::LITERT_LM;
  EXPECT_EQ(oss.str(), "LITERT_LM");
}

TEST(LlmExecutorConfigTest, ModelAssets) {
  auto model_assets = ModelAssets::Create(kPathToModel1);
  ASSERT_OK(model_assets);
  std::stringstream oss;
  oss << *model_assets;
  const std::string expected_output =
      absl::StrCat("model_path: ", kPathToModel1,
                   "\nfake_weights_mode: FAKE_WEIGHTS_NONE\n");
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
  config.use_submodel = true;
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
enable_external_embeddings: 0
use_submodel: 1
)";
  EXPECT_EQ(oss.str(), expected_output);
}

TEST(LlmExecutorConfigTest, LlmExecutorSettings) {
  auto model_assets = ModelAssets::Create(kPathToModel1);
  ASSERT_OK(model_assets);
  ASSERT_OK_AND_ASSIGN(auto settings,
                       LlmExecutorSettings::CreateDefault(
                           *std::move(model_assets), Backend::GPU_ARTISAN));
  settings.SetBackendConfig(CreateGpuArtisanConfig());
  settings.SetMaxNumTokens(1024);
  settings.SetActivationDataType(ActivationDataType::FLOAT16);
  settings.SetMaxNumImages(1);
  settings.SetCacheDir(std::string(kPathToCache));

  std::stringstream oss;
  oss << settings;
  const std::string expected_output = absl::StrCat(
      R"(backend: GPU_ARTISAN
backend_config:
num_output_candidates: 1
wait_for_weight_uploads: 1
num_decode_steps_per_sync: 3
sequence_batch_size: 16
supported_lora_ranks: vector of 2 elements: [4, 16]
max_top_k: 40
enable_decode_logits: 1
enable_external_embeddings: 0
use_submodel: 1

max_tokens: 1024
activation_data_type: FLOAT16
max_num_images: 1
lora_rank: 0
cache_dir: )",
      kPathToCache, R"(
cache_file: Not set
litert_dispatch_lib_dir: Not set
model_assets: model_path: )",
      kPathToModel1, R"(
fake_weights_mode: FAKE_WEIGHTS_NONE

advanced_settings: Not set
)");
  EXPECT_EQ(oss.str(), expected_output);
}

TEST(LlmExecutorConfigTest, LlmExecutorSettingsWithAdvancedSettings) {
  auto model_assets = ModelAssets::Create(kPathToModel1);
  ASSERT_OK(model_assets);
  ASSERT_OK_AND_ASSIGN(auto settings,
                       LlmExecutorSettings::CreateDefault(
                           *std::move(model_assets), Backend::GPU_ARTISAN));
  settings.SetBackendConfig(CreateGpuArtisanConfig());
  settings.SetMaxNumTokens(1024);
  settings.SetActivationDataType(ActivationDataType::FLOAT16);
  settings.SetMaxNumImages(1);
  settings.SetCacheDir(std::string(kPathToCache));
  settings.SetAdvancedSettings(AdvancedSettings{
      .prefill_batch_sizes = {128, 256},
      .num_output_candidates = 3,
      .configure_magic_numbers = true,
      .verify_magic_numbers = true,
      .clear_kv_cache_before_prefill = false,
      .num_logits_to_print_after_decode = 10,
      .gpu_madvise_original_shared_tensors = true,
      .is_benchmark = true,
      .preferred_device_substr = "nvidia",
      .num_threads_to_upload = 4,
      .num_threads_to_compile = 2,
      .convert_weights_on_gpu = true,
      .wait_for_weights_conversion_complete_in_benchmark = false,
      .optimize_shader_compilation = false,
      .cache_compiled_shaders_only = true,
      .share_constant_tensors = false,
      .sampler_handles_input = false,
      .allow_src_quantized_fc_conv_ops = true,
      .hint_waiting_for_completion = false,
      .enable_speculative_decoding = false,
      .disable_delegate_clustering = false,
      .hint_gpu_flush_period = 10,
  });

  std::stringstream oss;
  oss << settings;
  const std::string expected_output = absl::StrCat(
      R"(backend: GPU_ARTISAN
backend_config:
num_output_candidates: 1
wait_for_weight_uploads: 1
num_decode_steps_per_sync: 3
sequence_batch_size: 16
supported_lora_ranks: vector of 2 elements: [4, 16]
max_top_k: 40
enable_decode_logits: 1
enable_external_embeddings: 0
use_submodel: 1

max_tokens: 1024
activation_data_type: FLOAT16
max_num_images: 1
lora_rank: 0
cache_dir: )",
      kPathToCache, R"(
cache_file: Not set
litert_dispatch_lib_dir: Not set
model_assets: model_path: )",
      kPathToModel1, R"(
fake_weights_mode: FAKE_WEIGHTS_NONE

advanced_settings: prefill_batch_sizes: [128, 256]
num_output_candidates: 3
configure_magic_numbers: 1
verify_magic_numbers: 1
clear_kv_cache_before_prefill: 0
num_logits_to_print_after_decode: 10
gpu_madvise_original_shared_tensors: 1
is_benchmark: 1
preferred_device_substr: nvidia
num_threads_to_upload: 4
num_threads_to_compile: 2
convert_weights_on_gpu: 1
wait_for_weights_conversion_complete_in_benchmark: 0
optimize_shader_compilation: 0
cache_compiled_shaders_only: 1
share_constant_tensors: 0
sampler_handles_input: 0
allow_src_quantized_fc_conv_ops: 1
hint_waiting_for_completion: 0
gpu_context_low_priority: Not set
enable_speculative_decoding: 0
disable_delegate_clustering: 0
hint_gpu_flush_period: 10

)");
  EXPECT_EQ(oss.str(), expected_output);
}

TEST(LlmExecutorConfigTest, AdvancedSettingsWithGpuContextLowPriority) {
  auto model_assets = ModelAssets::Create(kPathToModel1);
  ASSERT_OK(model_assets);
  ASSERT_OK_AND_ASSIGN(auto settings,
                       LlmExecutorSettings::CreateDefault(
                           *std::move(model_assets), Backend::GPU_ARTISAN));
  settings.SetAdvancedSettings(AdvancedSettings{
      .gpu_context_low_priority = true,
  });

  std::stringstream oss;
  oss << settings;
  EXPECT_THAT(oss.str(), ::testing::HasSubstr("gpu_context_low_priority: 1"));

  settings.SetAdvancedSettings(AdvancedSettings{
      .gpu_context_low_priority = false,
  });
  oss.str("");
  oss << settings;
  EXPECT_THAT(oss.str(), ::testing::HasSubstr("gpu_context_low_priority: 0"));
}

TEST(LlmExecutorConfigTest, AdvancedSettingsWithHintGpuFlushPeriod) {
  auto model_assets = ModelAssets::Create(kPathToModel1);
  ASSERT_OK(model_assets);
  ASSERT_OK_AND_ASSIGN(auto settings,
                       LlmExecutorSettings::CreateDefault(
                           *std::move(model_assets), Backend::GPU_ARTISAN));
  settings.SetAdvancedSettings(AdvancedSettings{
      .hint_gpu_flush_period = 10,
  });

  std::stringstream oss;
  oss << settings;
  EXPECT_THAT(oss.str(), ::testing::HasSubstr("hint_gpu_flush_period: 10"));

  settings.SetAdvancedSettings(AdvancedSettings{
      .hint_gpu_flush_period = -1,
  });
  oss.str("");
  oss << settings;
  EXPECT_THAT(oss.str(), ::testing::HasSubstr("hint_gpu_flush_period: -1"));
}

TEST(GetWeightCacheFileTest, CacheDirAndModelPath) {
  auto model_assets = ModelAssets::Create(kPathToModel1Tflite);
  ASSERT_OK(model_assets);
  ASSERT_OK_AND_ASSIGN(auto settings, LlmExecutorSettings::CreateDefault(
                                          *std::move(model_assets)));
  settings.SetCacheDir(std::string(kWeightCachePath));

  ASSERT_OK_AND_ASSIGN(auto weight_cache_file, settings.GetWeightCacheFile());
  EXPECT_THAT(weight_cache_file,
              VariantWith<std::string>(std::string(kWeightCachePathFile)));
}

TEST(GetWeightCacheFileTest, CacheDirHasTrailingSeparator) {
  auto model_assets = ModelAssets::Create(kPathToModel1Tflite);
  ASSERT_OK(model_assets);
  ASSERT_OK_AND_ASSIGN(auto settings, LlmExecutorSettings::CreateDefault(
                                          *std::move(model_assets)));
  settings.SetCacheDir(std::string(kWeightCachePathWithSeparator));

  ASSERT_OK_AND_ASSIGN(auto weight_cache_file, settings.GetWeightCacheFile());
  EXPECT_THAT(weight_cache_file,
              VariantWith<std::string>(std::string(kWeightCachePathFile)));
}

TEST(GetWeightCacheFileTest, CacheDirAndModelPathAndCustomSuffix) {
  auto model_assets = ModelAssets::Create(kPathToModel1Tflite);
  ASSERT_OK(model_assets);
  ASSERT_OK_AND_ASSIGN(auto settings, LlmExecutorSettings::CreateDefault(
                                          *std::move(model_assets)));
  settings.SetCacheDir(std::string(kWeightCachePath));

  ASSERT_OK_AND_ASSIGN(auto weight_cache_file,
                       settings.GetWeightCacheFile(".xnnpack_cache"));
  EXPECT_THAT(weight_cache_file, VariantWith<std::string>(
                                     std::string(kWeightCachePathXnnpackFile)));
}

TEST(LlmExecutorConfigTest, ModelPathOnly) {
  auto model_assets = ModelAssets::Create(kPathToModel1Tflite);
  ASSERT_OK(model_assets);
  ASSERT_OK_AND_ASSIGN(auto settings, LlmExecutorSettings::CreateDefault(
                                          *std::move(model_assets)));

  ASSERT_OK_AND_ASSIGN(auto weight_cache_file, settings.GetWeightCacheFile());
  EXPECT_THAT(weight_cache_file,
              VariantWith<std::string>(std::string(kPathToModel1TfliteCache)));
}

TEST(GetWeightCacheFileTest, ModelPathAndSuffix) {
  auto model_assets = ModelAssets::Create(kPathToModel1Tflite);
  ASSERT_OK(model_assets);
  ASSERT_OK_AND_ASSIGN(auto settings, LlmExecutorSettings::CreateDefault(
                                          *std::move(model_assets)));

  ASSERT_OK_AND_ASSIGN(auto weight_cache_file,
                       settings.GetWeightCacheFile(".custom_suffix"));
  EXPECT_THAT(weight_cache_file,
              VariantWith<std::string>(std::string(kModel1TfliteCustomSuffix)));
}

TEST(GetWeightCacheFileTest, PreferScopedCacheFileToCacheDir) {
  const auto cache_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.cache";

  ASSERT_OK_AND_ASSIGN(auto cache_file, ScopedFile::Open(cache_path.string()));
  auto shared_cache_file = std::make_shared<ScopedFile>(std::move(cache_file));

  auto model_assets = ModelAssets::Create(kPathToModel1Tflite);
  ASSERT_OK(model_assets);
  ASSERT_OK_AND_ASSIGN(auto settings, LlmExecutorSettings::CreateDefault(
                                          *std::move(model_assets)));
  settings.SetScopedCacheFile(shared_cache_file);
  settings.SetCacheDir(std::string(kWeightCachePath));

  ASSERT_OK_AND_ASSIGN(auto weight_cache_file, settings.GetWeightCacheFile());
  EXPECT_THAT(weight_cache_file,
              VariantWith<std::shared_ptr<ScopedFile>>(shared_cache_file));
}

TEST(GetWeightCacheFileTest, PreferScopedCacheFileToScopedModelFile) {
  const auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.litertlm";
  const auto cache_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.cache";

  ASSERT_OK_AND_ASSIGN(auto model_file, ScopedFile::Open(model_path.string()));
  ASSERT_OK_AND_ASSIGN(auto cache_file, ScopedFile::Open(cache_path.string()));
  auto shared_cache_file = std::make_shared<ScopedFile>(std::move(cache_file));

  auto model_assets =
      ModelAssets::Create(std::make_shared<ScopedFile>(std::move(model_file)));
  ASSERT_OK(model_assets);
  ASSERT_OK_AND_ASSIGN(auto settings, LlmExecutorSettings::CreateDefault(
                                          *std::move(model_assets)));
  settings.SetScopedCacheFile(shared_cache_file);

  ASSERT_OK_AND_ASSIGN(auto weight_cache_file, settings.GetWeightCacheFile());
  EXPECT_THAT(weight_cache_file,
              VariantWith<std::shared_ptr<ScopedFile>>(shared_cache_file));
}

TEST(GetWeightCacheFileTest, EmptyModelPath) {
  auto model_assets = ModelAssets::Create("");
  ASSERT_OK(model_assets);
  ASSERT_OK_AND_ASSIGN(auto settings, LlmExecutorSettings::CreateDefault(
                                          *std::move(model_assets)));
  settings.SetCacheDir(std::string(kWeightCachePath));

  EXPECT_THAT(settings.GetWeightCacheFile(".xnnpack_cache"),
              StatusIs(kInvalidArgument));
}

TEST(GetWeightCacheFileTest, CacheDisabled) {
  const auto cache_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.cache";

  ASSERT_OK_AND_ASSIGN(auto cache_file, ScopedFile::Open(cache_path.string()));

  auto model_assets = ModelAssets::Create(kPathToModel1Tflite);
  ASSERT_OK(model_assets);
  ASSERT_OK_AND_ASSIGN(auto settings, LlmExecutorSettings::CreateDefault(
                                          *std::move(model_assets)));
  settings.SetCacheDir(":nocache");
  // This should be ignored in favor of the explicitly disabled cache dir.
  settings.SetScopedCacheFile(
      std::make_shared<ScopedFile>(std::move(cache_file)));

  EXPECT_THAT(settings.GetWeightCacheFile(), StatusIs(kInvalidArgument));
}

TEST(LlmExecutorConfigTest, GetBackendConfig) {
  auto model_assets = ModelAssets::Create(kPathToModel1);
  ASSERT_OK(model_assets);
  ASSERT_OK_AND_ASSIGN(auto settings,
                       LlmExecutorSettings::CreateDefault(
                           *std::move(model_assets), Backend::GPU_ARTISAN));

  settings.SetBackendConfig(CreateGpuArtisanConfig());

  ASSERT_OK_AND_ASSIGN(auto gpu_config,
                       settings.GetBackendConfig<GpuArtisanConfig>());
  EXPECT_EQ(gpu_config.num_output_candidates, 1);
  EXPECT_TRUE(gpu_config.use_submodel);

  // Test setting via MutableBackendConfig
  ASSERT_OK_AND_ASSIGN(auto mutable_gpu_config,
                       settings.MutableBackendConfig<GpuArtisanConfig>());
  mutable_gpu_config.use_submodel = false;
  settings.SetBackendConfig(mutable_gpu_config);
  ASSERT_OK_AND_ASSIGN(auto updated_gpu_config,
                       settings.GetBackendConfig<GpuArtisanConfig>());
  EXPECT_FALSE(updated_gpu_config.use_submodel);

  EXPECT_THAT(settings.GetBackendConfig<CpuConfig>(),
              StatusIs(kInvalidArgument));
}

TEST(LlmExecutorConfigTest, MutableBackendConfig) {
  auto model_assets = ModelAssets::Create(kPathToModel1);
  ASSERT_OK(model_assets);
  ASSERT_OK_AND_ASSIGN(auto settings,
                       LlmExecutorSettings::CreateDefault(
                           *std::move(model_assets), Backend::GPU_ARTISAN));
  settings.SetBackendConfig(CreateGpuArtisanConfig());

  ASSERT_OK_AND_ASSIGN(auto gpu_config,
                       settings.MutableBackendConfig<GpuArtisanConfig>());
  gpu_config.num_output_candidates = 2;
  settings.SetBackendConfig(gpu_config);

  ASSERT_OK_AND_ASSIGN(auto gpu_config_after_change,
                       settings.GetBackendConfig<GpuArtisanConfig>());
  EXPECT_EQ(gpu_config_after_change.num_output_candidates, 2);
  EXPECT_THAT(settings.MutableBackendConfig<CpuConfig>(),
              StatusIs(kInvalidArgument));
}

TEST(LlmExecutorConfigTest, SetSupportedLoraRanks) {
  auto model_assets = ModelAssets::Create(kPathToModel1);
  ASSERT_OK(model_assets);
  ASSERT_OK_AND_ASSIGN(auto settings, LlmExecutorSettings::CreateDefault(
                                          *std::move(model_assets),
                                          Backend::GPU_ARTISAN, Backend::GPU));
  EXPECT_EQ(settings.GetSamplerBackend(), Backend::GPU);
}

}  // namespace
}  // namespace litert::lm
