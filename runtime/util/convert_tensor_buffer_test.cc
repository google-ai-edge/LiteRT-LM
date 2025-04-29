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

#include "runtime/util/convert_tensor_buffer.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"  // from @litert
#include "litert/c/litert_tensor_buffer_types.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/test/matchers.h"  // from @litert

namespace litert::lm {
namespace {

using ::testing::ElementsAre;
using ::testing::litert::IsError;
using ::testing::litert::IsOkAndHolds;

MATCHER_P(LayoutDimensionsAre, n, "") {
  return ::testing::ExplainMatchResult(::testing::Eq(::litert::Dimensions(n)),
                                       arg.Layout().Dimensions(),
                                       result_listener);
};

TEST(ConvertTensorBufferTest, CreateTensorBuffer_Success) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto tensor_buffer,
                              CreateTensorBuffer<int8_t>({2, 5}));
  EXPECT_THAT(tensor_buffer.TensorType(),
              IsOkAndHolds(LayoutDimensionsAre(Dimensions({2, 5}))));
  EXPECT_THAT(tensor_buffer.Size(), IsOkAndHolds(10));
  EXPECT_THAT(tensor_buffer.BufferType(),
              IsOkAndHolds(kLiteRtTensorBufferTypeHostMemory));
}

TEST(ConvertTensorBufferTest, CreateTensorBuffer_Success_MultipleBytes) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto tensor_buffer,
                              CreateTensorBuffer<int32_t>({2, 5}));
  EXPECT_THAT(tensor_buffer.TensorType(),
              IsOkAndHolds(LayoutDimensionsAre(Dimensions({2, 5}))));
  EXPECT_THAT(tensor_buffer.Size(), IsOkAndHolds(40));
  EXPECT_THAT(tensor_buffer.BufferType(),
              IsOkAndHolds(kLiteRtTensorBufferTypeHostMemory));
}

TEST(ConvertTensorBufferTest, CopyToTensorBuffer_Success) {
  std::vector<int8_t> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  LITERT_ASSERT_OK_AND_ASSIGN(auto tensor_buffer,
                              CopyToTensorBuffer<int8_t>(data, {2, 5}));
  EXPECT_THAT(tensor_buffer.TensorType(),
              IsOkAndHolds(LayoutDimensionsAre(Dimensions({2, 5}))));
  EXPECT_THAT(tensor_buffer.Size(), IsOkAndHolds(10));
  EXPECT_THAT(tensor_buffer.BufferType(),
              IsOkAndHolds(kLiteRtTensorBufferTypeHostMemory));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto lock_and_addr,
      ::litert::TensorBufferScopedLock::Create(tensor_buffer));
  LITERT_ASSERT_OK_AND_ASSIGN(const size_t buffer_size, tensor_buffer.Size());
  const auto span = absl::MakeConstSpan(
      static_cast<int8_t*>(lock_and_addr.second), buffer_size);
  EXPECT_THAT(span, ElementsAre(1, 2, 3, 4, 5, 6, 7, 8, 9, 10));
}

TEST(ConvertTensorBufferTest, CopyToTensorBuffer_Success_MultipleBytes) {
  std::vector<int32_t> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  LITERT_ASSERT_OK_AND_ASSIGN(auto tensor_buffer,
                              CopyToTensorBuffer<int32_t>(data, {2, 5}));
  EXPECT_THAT(tensor_buffer.TensorType(),
              IsOkAndHolds(LayoutDimensionsAre(Dimensions({2, 5}))));
  EXPECT_THAT(tensor_buffer.Size(), IsOkAndHolds(40));
  EXPECT_THAT(tensor_buffer.BufferType(),
              IsOkAndHolds(kLiteRtTensorBufferTypeHostMemory));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto lock_and_addr,
      ::litert::TensorBufferScopedLock::Create(tensor_buffer));
  LITERT_ASSERT_OK_AND_ASSIGN(const size_t buffer_size, tensor_buffer.Size());
  auto span = absl::MakeConstSpan(static_cast<int32_t*>(lock_and_addr.second),
                                  buffer_size / sizeof(int32_t));
  EXPECT_THAT(span, ElementsAre(1, 2, 3, 4, 5, 6, 7, 8, 9, 10));
}

TEST(ConvertTensorBufferTest, ConvertAndCopyToTensorBuffer_ToInt8) {
  std::vector<int32_t> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto tensor_buffer,
      ConvertAndCopyToTensorBuffer<int8_t>(absl::MakeConstSpan(data), {2, 5}));
  EXPECT_THAT(tensor_buffer.TensorType(),
              IsOkAndHolds(LayoutDimensionsAre(Dimensions({2, 5}))));
  EXPECT_THAT(tensor_buffer.Size(), IsOkAndHolds(10));
  EXPECT_THAT(tensor_buffer.BufferType(),
              IsOkAndHolds(kLiteRtTensorBufferTypeHostMemory));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto lock_and_addr,
      ::litert::TensorBufferScopedLock::Create(tensor_buffer));
  LITERT_ASSERT_OK_AND_ASSIGN(const size_t buffer_size, tensor_buffer.Size());
  auto span = absl::MakeConstSpan(static_cast<int8_t*>(lock_and_addr.second),
                                  buffer_size / sizeof(int8_t));
  EXPECT_THAT(span, ElementsAre(1, 2, 3, 4, 5, 6, 7, 8, 9, 10));
}

TEST(ConvertTensorBufferTest, ConvertAndCopyToTensorBuffer_ToInt32) {
  std::vector<int8_t> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto tensor_buffer,
      ConvertAndCopyToTensorBuffer<int32_t>(absl::MakeConstSpan(data), {2, 5}));
  EXPECT_THAT(tensor_buffer.TensorType(),
              IsOkAndHolds(LayoutDimensionsAre(Dimensions({2, 5}))));
  EXPECT_THAT(tensor_buffer.Size(), IsOkAndHolds(40));
  EXPECT_THAT(tensor_buffer.BufferType(),
              IsOkAndHolds(kLiteRtTensorBufferTypeHostMemory));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto lock_and_addr,
      ::litert::TensorBufferScopedLock::Create(tensor_buffer));
  LITERT_ASSERT_OK_AND_ASSIGN(const size_t buffer_size, tensor_buffer.Size());
  auto span = absl::MakeConstSpan(static_cast<int32_t*>(lock_and_addr.second),
                                  buffer_size / sizeof(int32_t));
  EXPECT_THAT(span, ElementsAre(1, 2, 3, 4, 5, 6, 7, 8, 9, 10));
}

TEST(ConvertTensorBufferTest, ConvertAndCopyToTensorBuffer_ToFloat) {
  std::vector<int32_t> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto tensor_buffer,
      ConvertAndCopyToTensorBuffer<float>(absl::MakeConstSpan(data), {2, 5}));
  EXPECT_THAT(tensor_buffer.TensorType(),
              IsOkAndHolds(LayoutDimensionsAre(Dimensions({2, 5}))));
  EXPECT_THAT(tensor_buffer.Size(), IsOkAndHolds(40));
  EXPECT_THAT(tensor_buffer.BufferType(),
              IsOkAndHolds(kLiteRtTensorBufferTypeHostMemory));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto lock_and_addr,
      ::litert::TensorBufferScopedLock::Create(tensor_buffer));
  LITERT_ASSERT_OK_AND_ASSIGN(const size_t buffer_size, tensor_buffer.Size());
  auto span = absl::MakeConstSpan(static_cast<float*>(lock_and_addr.second),
                                  buffer_size / sizeof(float));
  EXPECT_THAT(span, ElementsAre(1., 2., 3., 4., 5., 6., 7., 8., 9., 10.));
}

TEST(ConvertTensorBufferTest, ReferTensorBufferAsSpan_Success) {
  std::vector<int8_t> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  LITERT_ASSERT_OK_AND_ASSIGN(auto tensor_buffer,
                              CopyToTensorBuffer<int8_t>(data, {2, 5}));
  EXPECT_THAT(ReferTensorBufferAsSpan<int8_t>(tensor_buffer),
              IsOkAndHolds(ElementsAre(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)));
}

TEST(ConvertTensorBufferTest, ReferTensorBufferAsSpan_Success_Const) {
  std::vector<int8_t> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  LITERT_ASSERT_OK_AND_ASSIGN(auto tensor_buffer,
                              CopyToTensorBuffer<int8_t>(data, {2, 5}));
  const ::litert::TensorBuffer& const_tensor_buffer = tensor_buffer;
  EXPECT_THAT(ReferTensorBufferAsSpan<int8_t>(const_tensor_buffer),
              IsOkAndHolds(ElementsAre(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)));
}

TEST(ConvertTensorBufferTest, ReferTensorBufferAsSpan_NonHostMemory) {
  ::litert::TensorBuffer tensor_buffer;
  EXPECT_THAT(ReferTensorBufferAsSpan<int8_t>(tensor_buffer),
              IsError(kLiteRtStatusErrorInvalidArgument,
                      "Tensor buffer is not in the host memory."));
}

TEST(ConvertTensorBufferTest, ReferTensorBufferAsSpan_IncompatibleElementType) {
  std::vector<int32_t> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  LITERT_ASSERT_OK_AND_ASSIGN(auto tensor_buffer,
                              CopyToTensorBuffer<int32_t>(data, {2, 5}));
  EXPECT_THAT(ReferTensorBufferAsSpan<float>(tensor_buffer),
              IsError(kLiteRtStatusErrorInvalidArgument,
                      "Element type is not compatible to the target type."));
}

TEST(ConvertTensorBufferTest, CopyFromTensorBuffer_Success) {
  std::vector<int8_t> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  LITERT_ASSERT_OK_AND_ASSIGN(auto tensor_buffer,
                              CopyToTensorBuffer<int8_t>(data, {2, 5}));
  EXPECT_THAT(CopyFromTensorBuffer<int8_t>(tensor_buffer),
              IsOkAndHolds(ElementsAre(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)));
}

TEST(ConvertTensorBufferTest, CopyFromTensorBuffer_Success_Const) {
  std::vector<int8_t> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  LITERT_ASSERT_OK_AND_ASSIGN(auto tensor_buffer,
                              CopyToTensorBuffer<int8_t>(data, {2, 5}));
  const ::litert::TensorBuffer& const_tensor_buffer = tensor_buffer;
  EXPECT_THAT(CopyFromTensorBuffer<int8_t>(const_tensor_buffer),
              IsOkAndHolds(ElementsAre(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)));
}

TEST(ConvertTensorBufferTest, CopyFromTensorBuffer_IncompatibleElementType) {
  std::vector<int32_t> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  LITERT_ASSERT_OK_AND_ASSIGN(auto tensor_buffer,
                              CopyToTensorBuffer<int32_t>(data, {2, 5}));
  EXPECT_THAT(CopyFromTensorBuffer<float>(tensor_buffer),
              IsError(kLiteRtStatusErrorInvalidArgument,
                      "Element type is not compatible to the target type."));
}

TEST(ConvertTensorBufferTest, CopyFromTensorBuffer2D_Success) {
  std::vector<int8_t> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  LITERT_ASSERT_OK_AND_ASSIGN(auto tensor_buffer,
                              CopyToTensorBuffer<int8_t>(data, {2, 5}));

  LITERT_ASSERT_OK_AND_ASSIGN(auto copied_data,
                              CopyFromTensorBuffer2D<int8_t>(tensor_buffer));
  EXPECT_EQ(copied_data.size(), 2);
  EXPECT_THAT(copied_data[0], ElementsAre(1, 2, 3, 4, 5));
  EXPECT_THAT(copied_data[1], ElementsAre(6, 7, 8, 9, 10));
}

TEST(ConvertTensorBufferTest, CopyFromTensorBuffer2D_Success_Const) {
  std::vector<int8_t> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  LITERT_ASSERT_OK_AND_ASSIGN(auto tensor_buffer,
                              CopyToTensorBuffer<int8_t>(data, {2, 5}));

  const ::litert::TensorBuffer& const_tensor_buffer = tensor_buffer;

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto copied_data, CopyFromTensorBuffer2D<int8_t>(const_tensor_buffer));
  EXPECT_EQ(copied_data.size(), 2);
  EXPECT_THAT(copied_data[0], ElementsAre(1, 2, 3, 4, 5));
  EXPECT_THAT(copied_data[1], ElementsAre(6, 7, 8, 9, 10));
}

TEST(ConvertTensorBufferTest, CopyFromTensorBuffer2D_IncompatibleElementType) {
  std::vector<int32_t> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  LITERT_ASSERT_OK_AND_ASSIGN(auto tensor_buffer,
                              CopyToTensorBuffer<int32_t>(data, {2, 5}));
  EXPECT_THAT(CopyFromTensorBuffer2D<float>(tensor_buffer),
              IsError(kLiteRtStatusErrorInvalidArgument,
                      "Element type is not compatible to the target type."));
}

TEST(ConvertTensorBufferTest, CopyFromTensorBuffer2D_Not2DTensor) {
  std::vector<int8_t> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  LITERT_ASSERT_OK_AND_ASSIGN(auto tensor_buffer,
                              CopyToTensorBuffer<int8_t>(data, {2, 3, 2}));
  EXPECT_THAT(CopyFromTensorBuffer2D<int8_t>(tensor_buffer),
              IsError(kLiteRtStatusErrorInvalidArgument,
                      "Tensor buffer must have 2 dimensions."));
}

}  // namespace
}  // namespace litert::lm
