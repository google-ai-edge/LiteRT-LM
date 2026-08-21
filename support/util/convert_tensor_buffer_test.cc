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

#include "support/util/convert_tensor_buffer.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_common.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/cc/litert_tensor_buffer_types.h"  // from @litert
#include "litert/test/matchers.h"  // from @litert

namespace litert::support {
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
              IsOkAndHolds(::litert::TensorBufferType::kHostMemory));
}

TEST(ConvertTensorBufferTest, CreateTensorBuffer_Success_MultipleBytes) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto tensor_buffer,
                              CreateTensorBuffer<int32_t>({2, 5}));
  EXPECT_THAT(tensor_buffer.TensorType(),
              IsOkAndHolds(LayoutDimensionsAre(Dimensions({2, 5}))));
  EXPECT_THAT(tensor_buffer.Size(), IsOkAndHolds(40));
  EXPECT_THAT(tensor_buffer.BufferType(),
              IsOkAndHolds(::litert::TensorBufferType::kHostMemory));
}

TEST(ConvertTensorBufferTest, CopyToTensorBuffer_Success) {
  std::vector<int8_t> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  LITERT_ASSERT_OK_AND_ASSIGN(auto tensor_buffer,
                              CopyToTensorBuffer<int8_t>(data, {2, 5}));
  EXPECT_THAT(tensor_buffer.TensorType(),
              IsOkAndHolds(LayoutDimensionsAre(Dimensions({2, 5}))));
  EXPECT_THAT(tensor_buffer.Size(), IsOkAndHolds(10));
  EXPECT_THAT(tensor_buffer.BufferType(),
              IsOkAndHolds(::litert::TensorBufferType::kHostMemory));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto lock_and_addr, ::litert::TensorBufferScopedLock::Create(
                              tensor_buffer, TensorBuffer::LockMode::kRead));
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
              IsOkAndHolds(::litert::TensorBufferType::kHostMemory));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto lock_and_addr, ::litert::TensorBufferScopedLock::Create(
                              tensor_buffer, TensorBuffer::LockMode::kRead));
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
              IsOkAndHolds(::litert::TensorBufferType::kHostMemory));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto lock_and_addr, ::litert::TensorBufferScopedLock::Create(
                              tensor_buffer, TensorBuffer::LockMode::kRead));
  LITERT_ASSERT_OK_AND_ASSIGN(const size_t buffer_size, tensor_buffer.Size());
  auto span = absl::MakeConstSpan(static_cast<int8_t*>(lock_and_addr.second),
                                  buffer_size / sizeof(int8_t));
  EXPECT_THAT(span, ElementsAre(1, 2, 3, 4, 5, 6, 7, 8, 9, 10));
}

TEST(ConvertTensorBufferTest, ConvertAndCopyToTensorBuffer_ToInt33) {
  std::vector<int8_t> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto tensor_buffer,
      ConvertAndCopyToTensorBuffer<int32_t>(absl::MakeConstSpan(data), {2, 5}));
  EXPECT_THAT(tensor_buffer.TensorType(),
              IsOkAndHolds(LayoutDimensionsAre(Dimensions({2, 5}))));
  EXPECT_THAT(tensor_buffer.Size(), IsOkAndHolds(40));
  EXPECT_THAT(tensor_buffer.BufferType(),
              IsOkAndHolds(::litert::TensorBufferType::kHostMemory));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto lock_and_addr, ::litert::TensorBufferScopedLock::Create(
                              tensor_buffer, TensorBuffer::LockMode::kRead));
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
              IsOkAndHolds(::litert::TensorBufferType::kHostMemory));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto lock_and_addr, ::litert::TensorBufferScopedLock::Create(
                              tensor_buffer, TensorBuffer::LockMode::kRead));
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
              IsError(::litert::Status::kErrorInvalidArgument,
                      "Tensor buffer is not in the host memory."));
}

TEST(ConvertTensorBufferTest, ReferTensorBufferAsSpan_IncompatibleElementType) {
  std::vector<int32_t> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  LITERT_ASSERT_OK_AND_ASSIGN(auto tensor_buffer,
                              CopyToTensorBuffer<int32_t>(data, {2, 5}));
  EXPECT_THAT(ReferTensorBufferAsSpan<float>(tensor_buffer),
              IsError(::litert::Status::kErrorInvalidArgument,
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
              IsError(::litert::Status::kErrorInvalidArgument,
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
              IsError(::litert::Status::kErrorInvalidArgument,
                      "Element type is not compatible to the target type."));
}

TEST(ConvertTensorBufferTest, CopyFromTensorBuffer2D_Not2DTensor) {
  std::vector<int8_t> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  LITERT_ASSERT_OK_AND_ASSIGN(auto tensor_buffer,
                              CopyToTensorBuffer<int8_t>(data, {2, 3, 2}));
  EXPECT_THAT(CopyFromTensorBuffer2D<int8_t>(tensor_buffer),
              IsError(::litert::Status::kErrorInvalidArgument,
                      "Tensor buffer must have 2 dimensions."));
}

TEST(ConvertTensorBufferTest, DropTokensfromTensorBuffer_Success) {
  std::vector<int32_t> source_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  LITERT_ASSERT_OK_AND_ASSIGN(auto source_tensor_buffer,
                              CopyToTensorBuffer<int32_t>(source_data, {10}));
  LITERT_ASSERT_OK(
      DropTokensfromTensorBuffer<int32_t>(source_tensor_buffer, 4, 0));
  EXPECT_THAT(source_tensor_buffer.TensorType(),
              IsOkAndHolds(LayoutDimensionsAre(Dimensions({10}))));
  EXPECT_THAT(source_tensor_buffer.Size(), IsOkAndHolds(40));
  EXPECT_THAT(ReferTensorBufferAsSpan<int32_t>(source_tensor_buffer),
              IsOkAndHolds(ElementsAre(5, 6, 7, 8, 9, 10, 0, 0, 0, 0)));
}

TEST(ConvertTensorBufferTest, DropTokensfromTensorBuffer2D_Success) {
  std::vector<int32_t> source_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  LITERT_ASSERT_OK_AND_ASSIGN(auto source_tensor_buffer,
                              CopyToTensorBuffer<int32_t>(source_data, {2, 5}));
  LITERT_ASSERT_OK(DropTokensfromTensorBuffer<int32_t>(source_tensor_buffer,
                                                       /*num_tokens_to_drop=*/2,
                                                       /*dimension=*/1));
  EXPECT_THAT(source_tensor_buffer.TensorType(),
              IsOkAndHolds(LayoutDimensionsAre(Dimensions({2, 5}))));
  EXPECT_THAT(source_tensor_buffer.Size(), IsOkAndHolds(40));
  EXPECT_THAT(ReferTensorBufferAsSpan<int32_t>(source_tensor_buffer),
              IsOkAndHolds(ElementsAre(3, 4, 5, 0, 0, 8, 9, 10, 0, 0)));
}

TEST(ConvertTensorBufferTest, DropTokensfromTensorBuffer_InvalidTokenSize) {
  std::vector<int32_t> source_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  LITERT_ASSERT_OK_AND_ASSIGN(auto source_tensor_buffer,
                              CopyToTensorBuffer<int32_t>(source_data, {10}));
  EXPECT_THAT(
      DropTokensfromTensorBuffer<int32_t>(source_tensor_buffer,
                                          /*num_tokens_to_drop=*/11,
                                          /*dimension=*/0),
      IsError(::litert::Status::kErrorInvalidArgument,
              "num_tokens_to_drop is larger than the target dimension."));
}

TEST(ConvertTensorBufferTest, DropTokensfromTensorBuffer_InvalidDropSize) {
  std::vector<int32_t> source_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  LITERT_ASSERT_OK_AND_ASSIGN(auto source_tensor_buffer,
                              CopyToTensorBuffer<int32_t>(source_data, {10}));
  EXPECT_THAT(DropTokensfromTensorBuffer<int32_t>(source_tensor_buffer, 2, 10),
              IsError(::litert::Status::kErrorInvalidArgument,
                      "Target dimension is out of range."));
}

TEST(ConvertTensorBufferTest, DropTokensfromTensorBuffer4D_Dim_2_Success) {
  std::vector<int32_t> source_data = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                                      11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                      21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                      31, 32, 33, 34, 35, 36, 37, 38, 39, 40};
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto source_tensor_buffer,
      CopyToTensorBuffer<int32_t>(source_data, {2, 1, 4, 5}));
  LITERT_ASSERT_OK(DropTokensfromTensorBuffer<int32_t>(source_tensor_buffer,
                                                       /*num_tokens_to_drop=*/2,
                                                       /*dimension=*/2));
  EXPECT_THAT(source_tensor_buffer.TensorType(),
              IsOkAndHolds(LayoutDimensionsAre(Dimensions({2, 1, 4, 5}))));
  EXPECT_THAT(source_tensor_buffer.Size(), IsOkAndHolds(160));
  EXPECT_THAT(
      ReferTensorBufferAsSpan<int32_t>(source_tensor_buffer),
      IsOkAndHolds(ElementsAre(11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 31, 32, 33, 34, 35, 36, 37,
                               38, 39, 40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)));
}

TEST(ConvertTensorBufferTest,
     DropTokensfromTensorBuffer4D_Dim_2_Offset_1_Retain_neg1_Failure) {
  std::vector<int32_t> source_data = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                                      11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                      21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                      31, 32, 33, 34, 35, 36, 37, 38, 39, 40};
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto source_tensor_buffer,
      CopyToTensorBuffer<int32_t>(source_data, {2, 1, 4, 5}));
  EXPECT_THAT(DropTokensfromTensorBuffer<int32_t>(source_tensor_buffer,
                                                  /*num_tokens_to_drop=*/2,
                                                  /*dimension=*/2,
                                                  /*init_tokens_to_retain=*/-1),
              IsError(::litert::Status::kErrorInvalidArgument,
                      "init_tokens_to_retain is negative."));
}

TEST(ConvertTensorBufferTest,
     DropTokensfromTensorBuffer4D_Dim_2_Offset_1_Retain_too_large_Failure) {
  std::vector<int32_t> source_data = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                                      11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                      21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                      31, 32, 33, 34, 35, 36, 37, 38, 39, 40};
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto source_tensor_buffer,
      CopyToTensorBuffer<int32_t>(source_data, {2, 1, 4, 5}));
  EXPECT_THAT(
      DropTokensfromTensorBuffer<int32_t>(source_tensor_buffer,
                                          /*num_tokens_to_drop=*/2,
                                          /*dimension=*/2,
                                          /*init_tokens_to_retain=*/10),
      IsError(::litert::Status::kErrorInvalidArgument,
              "init_tokens_to_retain is larger than the target dimension."));
}

TEST(ConvertTensorBufferTest, DropTokensFromTensorBuffer_TotalTokens_TooLarge) {
  std::vector<int32_t> source_data = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                                      11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                      21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                      31, 32, 33, 34, 35, 36, 37, 38, 39, 40};
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto source_tensor_buffer,
      CopyToTensorBuffer<int32_t>(source_data, {2, 1, 4, 5}));
  EXPECT_THAT(
      DropTokensfromTensorBuffer<int32_t>(source_tensor_buffer,
                                          /*num_tokens_to_drop=*/3,
                                          /*dimension=*/2,
                                          /*init_tokens_to_retain=*/2),
      IsError(::litert::Status::kErrorInvalidArgument,
              "the total number of tokens retained and dropped is greater than "
              "the target dimension. This will result in an out of bounds "
              "access."));
}

TEST(ConvertTensorBufferTest,
     DropTokensfromTensorBuffer4D_Dim_2_Offset_1_Success) {
  std::vector<int32_t> source_data = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                                      11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                      21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                      31, 32, 33, 34, 35, 36, 37, 38, 39, 40};
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto source_tensor_buffer,
      CopyToTensorBuffer<int32_t>(source_data, {2, 1, 4, 5}));
  LITERT_ASSERT_OK(
      DropTokensfromTensorBuffer<int32_t>(source_tensor_buffer,
                                          /*num_tokens_to_drop=*/2,
                                          /*dimension=*/2,
                                          /*init_tokens_to_retain=*/1));
  EXPECT_THAT(source_tensor_buffer.TensorType(),
              IsOkAndHolds(LayoutDimensionsAre(Dimensions({2, 1, 4, 5}))));
  EXPECT_THAT(source_tensor_buffer.Size(), IsOkAndHolds(160));
  EXPECT_THAT(
      ReferTensorBufferAsSpan<int32_t>(source_tensor_buffer),
      IsOkAndHolds(ElementsAre(1, 2, 3, 4, 5, 16, 17, 18, 19, 20, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 21, 22, 23, 24, 25, 36, 37, 38,
                               39, 40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)));
}

TEST(ConvertTensorBufferTest, DropTokensfromTensorBuffer4D_Dim_3_Success) {
  std::vector<int32_t> source_data = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                                      11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                      21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                      31, 32, 33, 34, 35, 36, 37, 38, 39, 40};
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto source_tensor_buffer,
      CopyToTensorBuffer<int32_t>(source_data, {2, 1, 4, 5}));
  LITERT_ASSERT_OK(DropTokensfromTensorBuffer<int32_t>(source_tensor_buffer,
                                                       /*num_tokens_to_drop=*/2,
                                                       /*dimension=*/3));
  EXPECT_THAT(source_tensor_buffer.TensorType(),
              IsOkAndHolds(LayoutDimensionsAre(Dimensions({2, 1, 4, 5}))));
  EXPECT_THAT(source_tensor_buffer.Size(), IsOkAndHolds(160));
  EXPECT_THAT(
      ReferTensorBufferAsSpan<int32_t>(source_tensor_buffer),
      IsOkAndHolds(ElementsAre(3, 4, 5, 0, 0, 8, 9, 10, 0, 0, 13, 14, 15, 0, 0,
                               18, 19, 20, 0, 0, 23, 24, 25, 0, 0, 28, 29, 30,
                               0, 0, 33, 34, 35, 0, 0, 38, 39, 40, 0, 0)));
}

TEST(ConvertTensorBufferTest,
     DropTokensfromTensorBuffer4D_Dim_3_Offset_1_Success) {
  std::vector<int32_t> source_data = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                                      11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                      21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                      31, 32, 33, 34, 35, 36, 37, 38, 39, 40};
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto source_tensor_buffer,
      CopyToTensorBuffer<int32_t>(source_data, {2, 1, 4, 5}));
  LITERT_ASSERT_OK(
      DropTokensfromTensorBuffer<int32_t>(source_tensor_buffer,
                                          /*num_tokens_to_drop=*/2,
                                          /*dimension=*/3,
                                          /*init_tokens_to_retain=*/1));
  EXPECT_THAT(source_tensor_buffer.TensorType(),
              IsOkAndHolds(LayoutDimensionsAre(Dimensions({2, 1, 4, 5}))));
  EXPECT_THAT(source_tensor_buffer.Size(), IsOkAndHolds(160));
  EXPECT_THAT(
      ReferTensorBufferAsSpan<int32_t>(source_tensor_buffer),
      IsOkAndHolds(ElementsAre(1, 4, 5, 0, 0, 6, 9, 10, 0, 0, 11, 14, 15, 0, 0,
                               16, 19, 20, 0, 0, 21, 24, 25, 0, 0, 26, 29, 30,
                               0, 0, 31, 34, 35, 0, 0, 36, 39, 40, 0, 0)));
}

TEST(ConvertTensorBufferTest, CopyToTensorBuffer_Bool_Success) {
  bool data[] = {true,  false, true,  false, true,
                 false, true,  false, true,  false};
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto tensor_buffer,
      CopyToTensorBuffer<bool>(absl::MakeConstSpan(data), {2, 5}));
  EXPECT_THAT(tensor_buffer.TensorType(),
              IsOkAndHolds(LayoutDimensionsAre(Dimensions({2, 5}))));
  EXPECT_THAT(tensor_buffer.Size(), IsOkAndHolds(10));
  EXPECT_THAT(tensor_buffer.BufferType(),
              IsOkAndHolds(::litert::TensorBufferType::kHostMemory));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto lock_and_addr, ::litert::TensorBufferScopedLock::Create(
                              tensor_buffer, TensorBuffer::LockMode::kRead));
  LITERT_ASSERT_OK_AND_ASSIGN(const size_t buffer_size, tensor_buffer.Size());
  const auto span = absl::MakeConstSpan(
      static_cast<bool*>(lock_and_addr.second), buffer_size / sizeof(bool));
  EXPECT_THAT(span, ElementsAre(true, false, true, false, true, false, true,
                                false, true, false));
}

TEST(ConvertTensorBufferTest, CopyFromTensorBuffer_Bool_Success) {
  bool data[] = {true,  false, true,  false, true,
                 false, true,  false, true,  false};
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto tensor_buffer,
      CopyToTensorBuffer<bool>(absl::MakeConstSpan(data), {2, 5}));
  EXPECT_THAT(CopyFromTensorBuffer<bool>(tensor_buffer),
              IsOkAndHolds(ElementsAre(true, false, true, false, true, false,
                                       true, false, true, false)));
}

TEST(ConvertTensorBufferTest, CopyFromTensorBuffer2D_Bool_Success) {
  bool data[] = {true,  false, true,  false, true,
                 false, true,  false, true,  false};
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto tensor_buffer,
      CopyToTensorBuffer<bool>(absl::MakeConstSpan(data), {2, 5}));

  LITERT_ASSERT_OK_AND_ASSIGN(auto copied_data,
                              CopyFromTensorBuffer2D<bool>(tensor_buffer));
  EXPECT_EQ(copied_data.size(), 2);
  EXPECT_THAT(copied_data[0], ElementsAre(true, false, true, false, true));
  EXPECT_THAT(copied_data[1], ElementsAre(false, true, false, true, false));
}

}  // namespace
}  // namespace litert::support
