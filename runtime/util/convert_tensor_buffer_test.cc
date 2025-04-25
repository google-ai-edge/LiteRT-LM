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

#include "third_party/odml/litert_lm/runtime/util/convert_tensor_buffer.h"

#include <cstdint>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "third_party/absl/types/span.h"
#include "litert/c/litert_common.h"
#include "litert/cc/litert_model.h"
#include "litert/cc/litert_tensor_buffer.h"

namespace litert::lm {
namespace {

using ::testing::ElementsAre;

TEST(ConvertTensorBufferTest, CreateTensorBuffer_Success) {
  auto tensor_buffer = CreateTensorBuffer<int8_t>({2, 5});
  EXPECT_TRUE(tensor_buffer.HasValue());
  EXPECT_EQ(tensor_buffer->TensorType()->Layout().Dimensions(),
            ::litert::Dimensions({2, 5}));
  EXPECT_EQ(*tensor_buffer->Size(), 10);
  EXPECT_EQ(*tensor_buffer->BufferType(), kLiteRtTensorBufferTypeHostMemory);
}

TEST(ConvertTensorBufferTest, CreateTensorBuffer_Success_MultipleBytes) {
  auto tensor_buffer = CreateTensorBuffer<int32_t>({2, 5});
  EXPECT_TRUE(tensor_buffer.HasValue());
  EXPECT_EQ(tensor_buffer->TensorType()->Layout().Dimensions(),
            ::litert::Dimensions({2, 5}));
  EXPECT_EQ(*tensor_buffer->Size(), 40);
  EXPECT_EQ(*tensor_buffer->BufferType(), kLiteRtTensorBufferTypeHostMemory);
}

TEST(ConvertTensorBufferTest, CopyToTensorBuffer_Success) {
  std::vector<int8_t> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  auto tensor_buffer = CopyToTensorBuffer<int8_t>(data, {2, 5});
  EXPECT_TRUE(tensor_buffer.HasValue());
  EXPECT_EQ(tensor_buffer->TensorType()->Layout().Dimensions(),
            ::litert::Dimensions({2, 5}));
  EXPECT_EQ(*tensor_buffer->Size(), 10);
  EXPECT_EQ(*tensor_buffer->BufferType(), kLiteRtTensorBufferTypeHostMemory);

  auto lock_and_addr = ::litert::TensorBufferScopedLock::Create(*tensor_buffer);
  EXPECT_TRUE(lock_and_addr.HasValue());
  auto span = absl::MakeConstSpan(static_cast<int8_t*>(lock_and_addr->second),
                                  *tensor_buffer->Size());
  EXPECT_THAT(span, ElementsAre(1, 2, 3, 4, 5, 6, 7, 8, 9, 10));
}

TEST(ConvertTensorBufferTest, CopyToTensorBuffer_Success_MultipleBytes) {
  std::vector<int32_t> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  auto tensor_buffer = CopyToTensorBuffer<int32_t>(data, {2, 5});
  EXPECT_TRUE(tensor_buffer.HasValue());
  EXPECT_EQ(tensor_buffer->TensorType()->Layout().Dimensions(),
            ::litert::Dimensions({2, 5}));
  EXPECT_EQ(*tensor_buffer->Size(), 40);
  EXPECT_EQ(*tensor_buffer->BufferType(), kLiteRtTensorBufferTypeHostMemory);

  auto lock_and_addr = ::litert::TensorBufferScopedLock::Create(*tensor_buffer);
  EXPECT_TRUE(lock_and_addr.HasValue());
  auto span = absl::MakeConstSpan(static_cast<int32_t*>(lock_and_addr->second),
                                  *tensor_buffer->Size() / sizeof(int32_t));
  EXPECT_THAT(span, ElementsAre(1, 2, 3, 4, 5, 6, 7, 8, 9, 10));
}

TEST(ConvertTensorBufferTest, ConvertAndCopyToTensorBuffer_ToInt8) {
  std::vector<int32_t> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  auto tensor_buffer =
      ConvertAndCopyToTensorBuffer<int8_t>(absl::MakeConstSpan(data), {2, 5});
  EXPECT_TRUE(tensor_buffer.HasValue());
  EXPECT_EQ(tensor_buffer->TensorType()->Layout().Dimensions(),
            ::litert::Dimensions({2, 5}));
  EXPECT_EQ(*tensor_buffer->Size(), 10);
  EXPECT_EQ(*tensor_buffer->BufferType(), kLiteRtTensorBufferTypeHostMemory);

  auto lock_and_addr = ::litert::TensorBufferScopedLock::Create(*tensor_buffer);
  EXPECT_TRUE(lock_and_addr.HasValue());
  auto span = absl::MakeConstSpan(static_cast<int8_t*>(lock_and_addr->second),
                                  *tensor_buffer->Size() / sizeof(int8_t));
  EXPECT_THAT(span, ElementsAre(1, 2, 3, 4, 5, 6, 7, 8, 9, 10));
}

TEST(ConvertTensorBufferTest, ConvertAndCopyToTensorBuffer_ToInt32) {
  std::vector<int8_t> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  auto tensor_buffer =
      ConvertAndCopyToTensorBuffer<int32_t>(absl::MakeConstSpan(data), {2, 5});
  EXPECT_TRUE(tensor_buffer.HasValue());
  EXPECT_EQ(tensor_buffer->TensorType()->Layout().Dimensions(),
            ::litert::Dimensions({2, 5}));
  EXPECT_EQ(*tensor_buffer->Size(), 40);
  EXPECT_EQ(*tensor_buffer->BufferType(), kLiteRtTensorBufferTypeHostMemory);

  auto lock_and_addr = ::litert::TensorBufferScopedLock::Create(*tensor_buffer);
  EXPECT_TRUE(lock_and_addr.HasValue());
  auto span = absl::MakeConstSpan(static_cast<int32_t*>(lock_and_addr->second),
                                  *tensor_buffer->Size() / sizeof(int32_t));
  EXPECT_THAT(span, ElementsAre(1, 2, 3, 4, 5, 6, 7, 8, 9, 10));
}

TEST(ConvertTensorBufferTest, ConvertAndCopyToTensorBuffer_ToFloat) {
  std::vector<int32_t> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  auto tensor_buffer =
      ConvertAndCopyToTensorBuffer<float>(absl::MakeConstSpan(data), {2, 5});
  EXPECT_TRUE(tensor_buffer.HasValue());
  EXPECT_EQ(tensor_buffer->TensorType()->Layout().Dimensions(),
            ::litert::Dimensions({2, 5}));
  EXPECT_EQ(*tensor_buffer->Size(), 40);
  EXPECT_EQ(*tensor_buffer->BufferType(), kLiteRtTensorBufferTypeHostMemory);

  auto lock_and_addr = ::litert::TensorBufferScopedLock::Create(*tensor_buffer);
  EXPECT_TRUE(lock_and_addr.HasValue());
  auto span = absl::MakeConstSpan(static_cast<float*>(lock_and_addr->second),
                                  *tensor_buffer->Size() / sizeof(float));
  EXPECT_THAT(span, ElementsAre(1., 2., 3., 4., 5., 6., 7., 8., 9., 10.));
}

TEST(ConvertTensorBufferTest, ReferTensorBufferAsSpan_Success) {
  std::vector<int8_t> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  auto tensor_buffer = CopyToTensorBuffer<int8_t>(data, {2, 5});
  EXPECT_TRUE(tensor_buffer.HasValue());
  EXPECT_EQ(tensor_buffer->TensorType()->Layout().Dimensions(),
            ::litert::Dimensions({2, 5}));
  EXPECT_EQ(*tensor_buffer->Size(), 10);
  EXPECT_EQ(*tensor_buffer->BufferType(), kLiteRtTensorBufferTypeHostMemory);

  auto span = ReferTensorBufferAsSpan<int8_t>(*tensor_buffer);
  EXPECT_TRUE(span.HasValue());
  EXPECT_THAT(*span, ElementsAre(1, 2, 3, 4, 5, 6, 7, 8, 9, 10));
}

TEST(ConvertTensorBufferTest, ReferTensorBufferAsSpan_Success_Const) {
  std::vector<int8_t> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  auto tensor_buffer = CopyToTensorBuffer<int8_t>(data, {2, 5});
  EXPECT_TRUE(tensor_buffer.HasValue());
  EXPECT_EQ(tensor_buffer->TensorType()->Layout().Dimensions(),
            ::litert::Dimensions({2, 5}));
  EXPECT_EQ(*tensor_buffer->Size(), 10);
  EXPECT_EQ(*tensor_buffer->BufferType(), kLiteRtTensorBufferTypeHostMemory);

  const ::litert::TensorBuffer& const_tensor_buffer = *tensor_buffer;
  auto span = ReferTensorBufferAsSpan<int8_t>(const_tensor_buffer);
  EXPECT_TRUE(span.HasValue());
  EXPECT_THAT(*span, ElementsAre(1, 2, 3, 4, 5, 6, 7, 8, 9, 10));
}

TEST(ConvertTensorBufferTest, ReferTensorBufferAsSpan_NonHostMemory) {
  ::litert::TensorBuffer tensor_buffer;

  auto span = ReferTensorBufferAsSpan<int8_t>(tensor_buffer);
  EXPECT_FALSE(span.HasValue());
  EXPECT_EQ(span.Error().Status(), kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(span.Error().Message(), "Tensor buffer is not in the host memory.");
}

TEST(ConvertTensorBufferTest, ReferTensorBufferAsSpan_IncompatibleElementType) {
  std::vector<int32_t> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  auto tensor_buffer = CopyToTensorBuffer<int32_t>(data, {2, 5});
  EXPECT_TRUE(tensor_buffer.HasValue());
  EXPECT_EQ(tensor_buffer->TensorType()->Layout().Dimensions(),
            ::litert::Dimensions({2, 5}));
  EXPECT_EQ(*tensor_buffer->Size(), 40);
  EXPECT_EQ(*tensor_buffer->BufferType(), kLiteRtTensorBufferTypeHostMemory);

  auto span = ReferTensorBufferAsSpan<float>(*tensor_buffer);
  EXPECT_EQ(span.Error().Status(), kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(span.Error().Message(),
            "Element type is not compatible to the target type.");
}

TEST(ConvertTensorBufferTest, CopyFromTensorBuffer_Success) {
  std::vector<int8_t> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  auto tensor_buffer = CopyToTensorBuffer<int8_t>(data, {2, 5});
  EXPECT_TRUE(tensor_buffer.HasValue());
  EXPECT_EQ(tensor_buffer->TensorType()->Layout().Dimensions(),
            ::litert::Dimensions({2, 5}));
  EXPECT_EQ(*tensor_buffer->Size(), 10);
  EXPECT_EQ(*tensor_buffer->BufferType(), kLiteRtTensorBufferTypeHostMemory);

  auto copied_data = CopyFromTensorBuffer<int8_t>(*tensor_buffer);
  EXPECT_TRUE(copied_data.HasValue());
  EXPECT_THAT(*copied_data, ElementsAre(1, 2, 3, 4, 5, 6, 7, 8, 9, 10));
}

TEST(ConvertTensorBufferTest, CopyFromTensorBuffer_Success_Const) {
  std::vector<int8_t> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  auto tensor_buffer = CopyToTensorBuffer<int8_t>(data, {2, 5});
  EXPECT_TRUE(tensor_buffer.HasValue());
  EXPECT_EQ(tensor_buffer->TensorType()->Layout().Dimensions(),
            ::litert::Dimensions({2, 5}));
  EXPECT_EQ(*tensor_buffer->Size(), 10);
  EXPECT_EQ(*tensor_buffer->BufferType(), kLiteRtTensorBufferTypeHostMemory);

  const ::litert::TensorBuffer& const_tensor_buffer = *tensor_buffer;

  auto copied_data = CopyFromTensorBuffer<int8_t>(const_tensor_buffer);
  EXPECT_TRUE(copied_data.HasValue());
  EXPECT_THAT(*copied_data, ElementsAre(1, 2, 3, 4, 5, 6, 7, 8, 9, 10));
}

TEST(ConvertTensorBufferTest, CopyFromTensorBuffer_IncompatibleElementType) {
  std::vector<int32_t> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  auto tensor_buffer = CopyToTensorBuffer<int32_t>(data, {2, 5});
  EXPECT_TRUE(tensor_buffer.HasValue());
  EXPECT_EQ(tensor_buffer->TensorType()->Layout().Dimensions(),
            ::litert::Dimensions({2, 5}));
  EXPECT_EQ(*tensor_buffer->Size(), 40);
  EXPECT_EQ(*tensor_buffer->BufferType(), kLiteRtTensorBufferTypeHostMemory);

  auto copied_data = CopyFromTensorBuffer<float>(*tensor_buffer);
  EXPECT_EQ(copied_data.Error().Status(), kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(copied_data.Error().Message(),
            "Element type is not compatible to the target type.");
}

TEST(ConvertTensorBufferTest, CopyFromTensorBuffer2D_Success) {
  std::vector<int8_t> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  auto tensor_buffer = CopyToTensorBuffer<int8_t>(data, {2, 5});
  EXPECT_TRUE(tensor_buffer.HasValue());
  EXPECT_EQ(tensor_buffer->TensorType()->Layout().Dimensions(),
            ::litert::Dimensions({2, 5}));
  EXPECT_EQ(*tensor_buffer->Size(), 10);
  EXPECT_EQ(*tensor_buffer->BufferType(), kLiteRtTensorBufferTypeHostMemory);

  auto copied_data = CopyFromTensorBuffer2D<int8_t>(*tensor_buffer);
  EXPECT_TRUE(copied_data.HasValue());
  EXPECT_EQ(copied_data->size(), 2);
  EXPECT_THAT((*copied_data)[0], ElementsAre(1, 2, 3, 4, 5));
  EXPECT_THAT((*copied_data)[1], ElementsAre(6, 7, 8, 9, 10));
}

TEST(ConvertTensorBufferTest, CopyFromTensorBuffer2D_Success_Const) {
  std::vector<int8_t> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  auto tensor_buffer = CopyToTensorBuffer<int8_t>(data, {2, 5});
  EXPECT_TRUE(tensor_buffer.HasValue());
  EXPECT_EQ(tensor_buffer->TensorType()->Layout().Dimensions(),
            ::litert::Dimensions({2, 5}));
  EXPECT_EQ(*tensor_buffer->Size(), 10);
  EXPECT_EQ(*tensor_buffer->BufferType(), kLiteRtTensorBufferTypeHostMemory);

  const ::litert::TensorBuffer& const_tensor_buffer = *tensor_buffer;

  auto copied_data = CopyFromTensorBuffer2D<int8_t>(const_tensor_buffer);
  EXPECT_TRUE(copied_data.HasValue());
  EXPECT_EQ(copied_data->size(), 2);
  EXPECT_THAT((*copied_data)[0], ElementsAre(1, 2, 3, 4, 5));
  EXPECT_THAT((*copied_data)[1], ElementsAre(6, 7, 8, 9, 10));
}

TEST(ConvertTensorBufferTest, CopyFromTensorBuffer2D_IncompatibleElementType) {
  std::vector<int32_t> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  auto tensor_buffer = CopyToTensorBuffer<int32_t>(data, {2, 5});
  EXPECT_TRUE(tensor_buffer.HasValue());
  EXPECT_EQ(tensor_buffer->TensorType()->Layout().Dimensions(),
            ::litert::Dimensions({2, 5}));
  EXPECT_EQ(*tensor_buffer->Size(), 40);
  EXPECT_EQ(*tensor_buffer->BufferType(), kLiteRtTensorBufferTypeHostMemory);

  auto copied_data = CopyFromTensorBuffer2D<float>(*tensor_buffer);
  EXPECT_EQ(copied_data.Error().Status(), kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(copied_data.Error().Message(),
            "Element type is not compatible to the target type.");
}

TEST(ConvertTensorBufferTest, CopyFromTensorBuffer2D_Not2DTensor) {
  std::vector<int8_t> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  auto tensor_buffer = CopyToTensorBuffer<int8_t>(data, {2, 3, 2});
  EXPECT_TRUE(tensor_buffer.HasValue());
  EXPECT_EQ(tensor_buffer->TensorType()->Layout().Dimensions(),
            ::litert::Dimensions({2, 3, 2}));
  EXPECT_EQ(*tensor_buffer->Size(), 12);
  EXPECT_EQ(*tensor_buffer->BufferType(), kLiteRtTensorBufferTypeHostMemory);

  auto copied_data = CopyFromTensorBuffer2D<int8_t>(*tensor_buffer);
  EXPECT_EQ(copied_data.Error().Status(), kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(copied_data.Error().Message(),
            "Tensor buffer must have 2 dimensions.");
}

}  // namespace
}  // namespace litert::lm
