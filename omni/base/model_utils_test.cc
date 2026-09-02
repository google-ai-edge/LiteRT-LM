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
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/cc/litert_tensor_buffer_types.h"  // from @litert
#include "runtime/executor/executor_settings_base.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace litert::omni {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;

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

}  // namespace
}  // namespace litert::omni
