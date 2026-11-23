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

#include "omni/any2any/any2any_test_utils.h"

#include <cstdint>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "omni/base/model_utils.h"
#include "runtime/executor/executor_settings_base.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep for ASSERT_OK_AND_ASSIGN
#include "tflite/schema/schema_generated.h"  // from @litert

namespace litert::omni::any2any {
namespace {

using ::testing::ElementsAre;
using ::testing::FloatEq;

TEST(Any2AnyTestUtilsTest, BuildMinimalTfliteModelEmitsZerosAndFloatValues) {
  auto env = Environment::Create({});
  ASSERT_TRUE(env.HasValue());

  // Include a value with all four distinct non-zero IEEE 754 bytes (0x40490fdb)
  // to catch any byte-ordering bug in FLOAT32 FlatBuffer packing.
  const std::vector<float> expected_values = {-2.5f, 0.0f, 1.25f, 3.1415927f};
  const std::string model_bytes = BuildMinimalTfliteModelBuffer(
      {
          {"tokens", {1, 2}, tflite::TensorType_INT32},
          {"features", {1, 3}, tflite::TensorType_FLOAT32},
      },
      {
          {"zero_out", {1, 3}, tflite::TensorType_FLOAT32},
          {"const_out", {2, 2}, tflite::TensorType_FLOAT32, expected_values},
      });

  ModelOptions options;
  options.backend = lm::Backend::CPU;
  ASSERT_OK_AND_ASSIGN(
      CompiledModel compiled,
      CreateCompiledModelFromBuffer(*env, options, model_bytes, "test_model"));

  auto inputs = compiled.CreateInputBuffers("serving_default");
  ASSERT_TRUE(inputs.HasValue());
  ASSERT_EQ(inputs->size(), 2);
  auto outputs = compiled.CreateOutputBuffers("serving_default");
  ASSERT_TRUE(outputs.HasValue());
  ASSERT_EQ(outputs->size(), 2);

  const std::vector<int32_t> tokens = {1, 2};
  ASSERT_TRUE(
      (*inputs)[0].Write<int32_t>(absl::MakeConstSpan(tokens)).HasValue());
  const std::vector<float> features = {0.1f, 0.2f, 0.3f};
  ASSERT_TRUE(
      (*inputs)[1].Write<float>(absl::MakeConstSpan(features)).HasValue());

  ASSERT_TRUE(compiled.Run("serving_default", *inputs, *outputs).HasValue());

  std::vector<float> zero_out(3, 99.0f);
  ASSERT_TRUE((*outputs)[0].Read<float>(absl::MakeSpan(zero_out)).HasValue());
  EXPECT_THAT(zero_out,
              ElementsAre(FloatEq(0.0f), FloatEq(0.0f), FloatEq(0.0f)));

  std::vector<float> const_out(4, 0.0f);
  ASSERT_TRUE((*outputs)[1].Read<float>(absl::MakeSpan(const_out)).HasValue());
  EXPECT_THAT(const_out, ElementsAre(FloatEq(-2.5f), FloatEq(0.0f),
                                     FloatEq(1.25f), FloatEq(3.1415927f)));
}

}  // namespace
}  // namespace litert::omni::any2any
