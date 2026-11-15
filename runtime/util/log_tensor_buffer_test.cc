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

#include "runtime/util/log_tensor_buffer.h"

#include <cstdint>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/cc/litert_tensor_buffer_types.h"  // from @litert
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

TEST(LogTensorBufferTest, LogTensor_Float) {
  struct alignas(kHostMemoryBufferAlignment) {
    float d[5] = {1.1, 2.2, -3.3, 4.4, 5.5};
  } data;

  auto tensor_buffer = TensorBuffer::CreateFromHostMemory(
      RankedTensorType(ElementType::Float32, Layout(Dimensions({5}))), data.d,
      5 * sizeof(float));
  ASSERT_TRUE(tensor_buffer.HasValue());

  EXPECT_OK(LogTensor(*tensor_buffer, 2, "Float Prefix: "));
}

TEST(LogTensorBufferTest, LogTensor_Int8) {
  struct alignas(kHostMemoryBufferAlignment) {
    int8_t d[5] = {1, 2, -3, 4, 5};
  } data;

  auto tensor_buffer = TensorBuffer::CreateFromHostMemory(
      RankedTensorType(ElementType::Int8, Layout(Dimensions({5}))), data.d,
      5 * sizeof(int8_t));
  ASSERT_TRUE(tensor_buffer.HasValue());

  EXPECT_OK(LogTensor(*tensor_buffer, 5, "Int8 Prefix: "));
}

TEST(LogTensorBufferTest, LogTensor_Bool) {
  struct alignas(kHostMemoryBufferAlignment) {
    bool d[3] = {true, false, true};
  } data;

  auto tensor_buffer = TensorBuffer::CreateFromHostMemory(
      RankedTensorType(ElementType::Bool, Layout(Dimensions({3}))), data.d,
      3 * sizeof(bool));
  ASSERT_TRUE(tensor_buffer.HasValue());

  EXPECT_OK(LogTensor(*tensor_buffer, 3, "Bool Prefix: "));
}

}  // namespace
}  // namespace litert::lm
