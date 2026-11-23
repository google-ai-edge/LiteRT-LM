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

#include "omni/utils/tensor_utils.h"

#include <cstdint>
#include <utility>

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
#include "support/util/test_utils.h"  // IWYU pragma: keep for ASSERT_OK

namespace litert::omni {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

TEST(TensorUtilsTest, ValidateFloatBufferChecksTypeAndCapacity) {
  auto env = Environment::Create({});
  ASSERT_TRUE(env.HasValue());

  RankedTensorType float_type(ElementType::Float32, Layout(Dimensions({2, 4})));
  auto float_buf = TensorBuffer::CreateManaged(
      *env, TensorBufferType::kHostMemory, std::move(float_type),
      /*buffer_size=*/8 * sizeof(float));
  ASSERT_TRUE(float_buf.HasValue());

  EXPECT_OK(ValidateFloatBuffer(*float_buf, /*expected_elements=*/8, "hidden"));
  EXPECT_THAT(
      ValidateFloatBuffer(*float_buf, /*expected_elements=*/4, "hidden"),
      StatusIs(absl::StatusCode::kInvalidArgument));

  RankedTensorType int_type(ElementType::Int32, Layout(Dimensions({2, 4})));
  auto int_buf = TensorBuffer::CreateManaged(
      *env, TensorBufferType::kHostMemory, std::move(int_type),
      /*buffer_size=*/8 * sizeof(int32_t));
  ASSERT_TRUE(int_buf.HasValue());

  EXPECT_THAT(ValidateFloatBuffer(*int_buf, /*expected_elements=*/8, "hidden"),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(TensorUtilsTest, ValidateTensorBufferChecksTypeAndElementCount) {
  auto env = Environment::Create({});
  ASSERT_TRUE(env.HasValue());

  RankedTensorType float_type(ElementType::Float32, Layout(Dimensions({2, 3})));
  auto float_buf = TensorBuffer::CreateManaged(
      *env, TensorBufferType::kHostMemory, std::move(float_type),
      /*buffer_size=*/6 * sizeof(float));
  ASSERT_TRUE(float_buf.HasValue());

  EXPECT_OK(ValidateTensorBuffer(*float_buf, ElementType::Float32,
                                 /*expected_elements=*/6, "embeds"));
  EXPECT_THAT(ValidateTensorBuffer(*float_buf, ElementType::Int32,
                                   /*expected_elements=*/6, "embeds"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("embeds buffer element type")));
  EXPECT_THAT(ValidateTensorBuffer(*float_buf, ElementType::Float32,
                                   /*expected_elements=*/4, "embeds"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("embeds buffer element count")));

  RankedTensorType int_type(ElementType::Int32, Layout(Dimensions({1, 5})));
  auto int_buf = TensorBuffer::CreateManaged(
      *env, TensorBufferType::kHostMemory, std::move(int_type),
      /*buffer_size=*/5 * sizeof(int32_t));
  ASSERT_TRUE(int_buf.HasValue());

  EXPECT_OK(ValidateTensorBuffer(*int_buf, ElementType::Int32,
                                 /*expected_elements=*/5, "token_ids"));
  EXPECT_THAT(ValidateTensorBuffer(*int_buf, ElementType::Int64,
                                   /*expected_elements=*/5, "token_ids"),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(ValidateTensorBuffer(*int_buf, ElementType::Int32,
                                   /*expected_elements=*/10, "token_ids"),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace litert::omni
