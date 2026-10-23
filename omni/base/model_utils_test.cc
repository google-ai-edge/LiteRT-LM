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

#include "omni/base/model_utils.h"

#include <cstdint>
#include <filesystem>  // NOLINT
#include <fstream>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_options.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/cc/litert_tensor_buffer_types.h"  // from @litert
#include "omni/omni_engine.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/proto/speech2text_metadata.pb.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace litert::omni {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;

TEST(ModelUtilsTest, ResolveLitertLmPathFromFileAndDirectory) {
  std::filesystem::path temp_dir =
      std::filesystem::path(::testing::TempDir()) / "resolve_litertlm_test";
  std::filesystem::create_directories(temp_dir);
  EXPECT_EQ(ResolveLitertLmPath(temp_dir.string()), "");
  EXPECT_EQ(ResolveLitertLmPath("/non/existent/path"), "");

  std::filesystem::path model_file = temp_dir / "test_model.litertlm";
  {
    std::ofstream out(model_file);
    out << "dummy";
  }
  EXPECT_EQ(ResolveLitertLmPath(model_file.string()), model_file.string());
  EXPECT_EQ(ResolveLitertLmPath(temp_dir.string()), model_file.string());
  std::filesystem::remove_all(temp_dir);
}

TEST(ModelUtilsTest, CheckFileReadableNonExistent) {
  EXPECT_THAT(CheckFileReadable("/non/existent/file.bin"),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST(ModelUtilsTest, LoadFileNonExistent) {
  EXPECT_THAT(LoadFile("/non/existent/dir", "model.tflite"),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST(ModelUtilsTest, CreateCompiledModelNonExistent) {
  auto env_expected = Environment::Create({});
  if (env_expected.HasValue()) {
    ModelOptions options;
    options.model_dir = "/non/existent/dir";
    EXPECT_THAT(CreateCompiledModel(*env_expected, options, "model.tflite"),
                StatusIs(absl::StatusCode::kNotFound));
  }
}

TEST(ModelUtilsTest, CreateCompiledModelConfiguresModelCaches) {
  auto env = Environment::Create({});
  ASSERT_TRUE(env.HasValue());
  const std::filesystem::path root =
      std::filesystem::path(::testing::TempDir()) / "model_utils_cache_test";
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root);
  std::ofstream((root / "model.tflite").string()) << "dummy";
  std::ofstream((root / "bundle.litertlm").string()) << "dummy";

  // Unwritable explicit cache_dir fails before model compilation.
  ModelOptions bad_cache{.model_dir = root.string(),
                         .cache_dir = "/non/existent/cache/dir"};
  EXPECT_THAT(CreateCompiledModel(*env, bad_cache, "model.tflite"),
              StatusIs(absl::StatusCode::kPermissionDenied));
  EXPECT_THAT(
      CreateCompiledModelFromBuffer(*env, bad_cache, "dummy", "kokoro_vocoder"),
      StatusIs(absl::StatusCode::kPermissionDenied));

  // Writable cache_dir cleans stale CPU and GPU caches of the same model.
  std::ofstream((root / "model.tflite_1_2.xnnpack_cache").string()) << "x";
  std::ofstream(
      (root / "kokoro_vocoder_1_2_mldrift_program_cache.bin").string())
      << "x";
  ModelOptions cpu_opts{.model_dir = root.string(),
                        .cache_dir = root.string(),
                        .backend = lm::Backend::CPU};
  EXPECT_FALSE(CreateCompiledModel(*env, cpu_opts, "model.tflite").ok());
  EXPECT_FALSE(
      std::filesystem::exists(root / "model.tflite_1_2.xnnpack_cache"));

  ModelOptions gpu_opts{.model_dir = root.string(),
                        .cache_dir = root.string(),
                        .backend = lm::Backend::GPU};
  EXPECT_FALSE(
      CreateCompiledModelFromBuffer(*env, gpu_opts, "dummy", "kokoro_vocoder")
          .ok());
  EXPECT_FALSE(std::filesystem::exists(
      root / "kokoro_vocoder_1_2_mldrift_program_cache.bin"));
  EXPECT_FALSE(CreateLmRunner(*env, cpu_opts, "bundle.litertlm").ok());
  EXPECT_FALSE(CreateLmEngineRunner(*env, cpu_opts, "bundle.litertlm").ok());
  std::filesystem::remove_all(root);
}

TEST(ModelUtilsTest, CreateExecutorInputsWithText) {
  auto env = Environment::Create({});
  ASSERT_TRUE(env.HasValue());
  RankedTensorType type(ElementType::Int32, Layout(Dimensions({1, 4})));
  auto buffer = TensorBuffer::CreateManaged(*env, TensorBufferType::kHostMemory,
                                            type, 4 * sizeof(int32_t));
  ASSERT_TRUE(buffer.HasValue());

  auto inputs = CreateExecutorInputsWithText(*buffer);
  ASSERT_THAT(inputs, IsOk());
  EXPECT_THAT(inputs->GetTextDataPtr(), IsOk());
}

TEST(ModelUtilsTest, CreateExecutorInputsWithAudio) {
  auto env = Environment::Create({});
  ASSERT_TRUE(env.HasValue());
  RankedTensorType type(ElementType::Float32, Layout(Dimensions({1, 1024})));
  auto buffer = TensorBuffer::CreateManaged(*env, TensorBufferType::kHostMemory,
                                            type, 1024 * sizeof(float));
  ASSERT_TRUE(buffer.HasValue());

  auto inputs = CreateExecutorInputsWithAudio(*buffer);
  ASSERT_THAT(inputs, IsOk());
  EXPECT_THAT(inputs->GetAudioDataPtr(), IsOk());
}

TEST(ModelUtilsTest, CreateExecutorInputsWithVision) {
  auto env = Environment::Create({});
  ASSERT_TRUE(env.HasValue());
  RankedTensorType type(ElementType::Float32, Layout(Dimensions({1, 512})));
  auto buffer = TensorBuffer::CreateManaged(*env, TensorBufferType::kHostMemory,
                                            type, 512 * sizeof(float));
  ASSERT_TRUE(buffer.HasValue());

  auto inputs = CreateExecutorInputsWithVision(*buffer);
  ASSERT_THAT(inputs, IsOk());
  EXPECT_THAT(inputs->GetVisionDataPtr(), IsOk());
}

TEST(ModelUtilsTest, CreateCompiledModelForStatefulRunnerCpuNonExistent) {
  auto env = Environment::Create({});
  ASSERT_TRUE(env.HasValue());
  ModelOptions options;
  options.model_dir = "/non/existent/dir";
  options.backend = lm::Backend::CPU;
  EXPECT_THAT(CreateCompiledModelForStatefulRunner(
                  *env, options, "model.tflite",
                  /*signature_name=*/"", /*num_non_state_inputs=*/1,
                  /*num_non_state_outputs=*/1),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST(ModelUtilsTest, CreateCompiledModelForStatefulRunnerGpuNonExistent) {
  auto env = Environment::Create({});
  ASSERT_TRUE(env.HasValue());
  ModelOptions options;
  options.model_dir = "/non/existent/dir";
  options.backend = lm::Backend::GPU;
  EXPECT_THAT(CreateCompiledModelForStatefulRunner(
                  *env, options, "model.tflite",
                  /*signature_name=*/"", /*num_non_state_inputs=*/1,
                  /*num_non_state_outputs=*/1),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST(ModelUtilsTest, ModelOptionsExternalTensorPatterns) {
  ModelOptions options;
  EXPECT_TRUE(options.external_tensor_patterns.empty());
  options.external_tensor_patterns.push_back("kv_cache_");
  EXPECT_EQ(options.external_tensor_patterns.size(), 1);
  EXPECT_EQ(options.external_tensor_patterns[0], "kv_cache_");
}

TEST(ModelUtilsTest, CreateLmEngineRunnerNonExistent) {
  auto env_expected = Environment::Create({});
  if (env_expected.HasValue()) {
    ModelOptions options;
    options.model_dir = "/non/existent/dir";
    EXPECT_THAT(CreateLmEngineRunner(*env_expected, options, "model.litertlm"),
                StatusIs(absl::StatusCode::kNotFound));
  }
}

TEST(ModelUtilsTest, ParseOmniBackend) {
  auto cpu = ParseOmniBackend("cpu");
  ASSERT_THAT(cpu, IsOk());
  EXPECT_EQ(*cpu, OmniEngine::Settings::Backend::kCpu);

  auto gpu = ParseOmniBackend("GPU");
  ASSERT_THAT(gpu, IsOk());
  EXPECT_EQ(*gpu, OmniEngine::Settings::Backend::kGpu);

  auto npu = ParseOmniBackend("Npu");
  ASSERT_THAT(npu, IsOk());
  EXPECT_EQ(*npu, OmniEngine::Settings::Backend::kNpu);

  EXPECT_THAT(ParseOmniBackend("unknown"),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(ModelUtilsTest, ToLmModelOptionsMapsEngineSettings) {
  OmniEngine::Settings settings{
      .backend = OmniEngine::Settings::Backend::kGpu,
      .cache_dir = "/tmp/custom_cache",
      .num_threads = 6,
  };
  ModelOptions opts = ToLmModelOptions(settings);
  EXPECT_EQ(opts.backend, lm::Backend::GPU);
  EXPECT_EQ(opts.cache_dir, "/tmp/custom_cache");
  EXPECT_EQ(opts.num_threads, 6);
  EXPECT_FALSE(opts.force_fp32_on_gpu);

  settings.backend = OmniEngine::Settings::Backend::kNpu;
  settings.cache_dir.clear();
  settings.num_threads = 0;
  settings.force_fp32_on_gpu = true;
  opts = ToLmModelOptions(settings);
  EXPECT_EQ(opts.backend, lm::Backend::NPU);
  EXPECT_EQ(opts.cache_dir, ":nocache");
  EXPECT_EQ(opts.num_threads, 4);
  EXPECT_TRUE(opts.force_fp32_on_gpu);
}

TEST(ModelUtilsTest, CreateTokenizerReturnsInvalidArgumentWhenMissing) {
  EXPECT_THAT(CreateTokenizer(/*resources=*/nullptr, /*tokenizer_path=*/""),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(ModelUtilsTest, CreateCompiledModelFromEngineSettingsNonExistent) {
  auto env = Environment::Create({});
  ASSERT_TRUE(env.HasValue());
  auto options = Options::Create();
  ASSERT_TRUE(options.HasValue());
  OmniEngine::Settings settings{
      .model_path = "/non/existent/model.tflite",
  };
  EXPECT_THAT(CreateCompiledModel(
                  settings, *env, *options, /*lm_resources=*/nullptr,
                  lm::proto::Speech2TextMetadata::TF_LITE_ENCODER_DECODER),
              StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(
      CreateCompiledModel(settings, *env, *options,
                          /*lm_resources=*/nullptr,
                          lm::proto::LlmMetadata::TF_LITE_AUDIO_ENCODER_HW),
      StatusIs(absl::StatusCode::kNotFound));
}

}  // namespace
}  // namespace litert::omni
