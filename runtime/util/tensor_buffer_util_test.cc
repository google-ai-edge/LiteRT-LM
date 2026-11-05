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

#include "runtime/util/tensor_buffer_util.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/cc/litert_tensor_buffer_types.h"  // from @litert
#include "litert/test/matchers.h"  // from @litert
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::litert::IsOkAndHolds;
using ::testing::status::StatusIs;

TEST(TensorBufferUtilTest, NumSignificantDims) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto tensor_buffer,
                              CreateTensorBuffer<int8_t>({2, 5}));
  EXPECT_THAT(NumSignificantDims(tensor_buffer), IsOkAndHolds(Eq(2)));
  LITERT_ASSERT_OK_AND_ASSIGN(tensor_buffer,
                              CreateTensorBuffer<int8_t>({2, 1, 5}));
  EXPECT_THAT(NumSignificantDims(tensor_buffer), IsOkAndHolds(Eq(2)));
  LITERT_ASSERT_OK_AND_ASSIGN(tensor_buffer,
                              CreateTensorBuffer<int8_t>({1, 1, 5}));
  EXPECT_THAT(NumSignificantDims(tensor_buffer), IsOkAndHolds(Eq(1)));
}

TEST(TensorBufferUtilTest, TensorBufferDims) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto tensor_buffer,
                              CreateTensorBuffer<int8_t>({2, 5}));
  EXPECT_THAT(TensorBufferDims(tensor_buffer), IsOkAndHolds(ElementsAre(2, 5)));
  LITERT_ASSERT_OK_AND_ASSIGN(tensor_buffer,
                              CreateTensorBuffer<int8_t>({2, 1, 5}));
  EXPECT_THAT(TensorBufferDims(tensor_buffer),
              IsOkAndHolds(ElementsAre(2, 1, 5)));
  LITERT_ASSERT_OK_AND_ASSIGN(tensor_buffer,
                              CreateTensorBuffer<int8_t>({1, 1, 5}));
  EXPECT_THAT(TensorBufferDims(tensor_buffer),
              IsOkAndHolds(ElementsAre(1, 1, 5)));
}

TEST(TensorBufferUtilTest, CopyTensorBuffer) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, litert::Environment::Create({}));
  std::vector<int8_t> data = {1, 2, 3, 4, 5};
  LITERT_ASSERT_OK_AND_ASSIGN(auto tensor_buffer,
                              CreateTensorBuffer<int8_t>({5}));
  LITERT_ASSERT_OK(tensor_buffer.Write<int8_t>(data));

  LITERT_ASSERT_OK_AND_ASSIGN(auto copy, CopyTensorBuffer(env, tensor_buffer));

  EXPECT_THAT(copy.BufferType(),
              IsOkAndHolds(Eq(TensorBufferType::kHostMemory)));
  EXPECT_THAT(TensorBufferDims(copy), IsOkAndHolds(ElementsAre(5)));
  LITERT_ASSERT_OK_AND_ASSIGN(auto copy_data,
                              CopyFromTensorBuffer<int8_t>(copy));
  EXPECT_THAT(copy_data, ElementsAre(1, 2, 3, 4, 5));

  // Verify deep copy: modify original, copy should not change.
  std::vector<int8_t> new_data = {10, 20, 30, 40, 50};
  LITERT_ASSERT_OK(tensor_buffer.Write<int8_t>(new_data));
  LITERT_ASSERT_OK_AND_ASSIGN(copy_data, CopyFromTensorBuffer<int8_t>(copy));
  EXPECT_THAT(copy_data, ElementsAre(1, 2, 3, 4, 5));

  // Verify deep copy: modify copy, original should not change.
  std::vector<int8_t> copy_new_data = {5, 4, 3, 2, 1};
  LITERT_ASSERT_OK(copy.Write<int8_t>(copy_new_data));
  LITERT_ASSERT_OK_AND_ASSIGN(auto original_data,
                              CopyFromTensorBuffer<int8_t>(tensor_buffer));
  EXPECT_THAT(original_data, ElementsAre(10, 20, 30, 40, 50));
}

TEST(TensorBufferUtilTest, CopyTensorBufferPreservesBufferTypeAndPaddedSize) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, litert::Environment::Create({}));
  const RankedTensorType tensor_type = MakeRankedTensorType<int32_t>({5});
  constexpr size_t kPackedSize = 5 * sizeof(int32_t);
  constexpr size_t kPaddedSize = 128;

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto padded_buffer,
      TensorBuffer::CreateManaged(env, TensorBufferType::kHostMemory,
                                  tensor_type, kPaddedSize));
  const std::vector<int32_t> data = {10, 20, 30, 40, 50};
  LITERT_ASSERT_OK(padded_buffer.Write<int32_t>(data));

  LITERT_ASSERT_OK_AND_ASSIGN(auto copy, CopyTensorBuffer(env, padded_buffer));

  EXPECT_THAT(copy.BufferType(),
              IsOkAndHolds(Eq(TensorBufferType::kHostMemory)));
  EXPECT_THAT(copy.Size(), IsOkAndHolds(Eq(kPaddedSize)));
  EXPECT_THAT(copy.PackedSize(), IsOkAndHolds(Eq(kPackedSize)));
  EXPECT_THAT(CopyFromTensorBuffer<int32_t>(copy),
              IsOkAndHolds(ElementsAre(10, 20, 30, 40, 50)));

  const std::vector<int32_t> updated_data = {60, 70, 80, 90, 100};
  LITERT_ASSERT_OK(padded_buffer.Write<int32_t>(updated_data));
  EXPECT_THAT(CopyFromTensorBuffer<int32_t>(copy),
              IsOkAndHolds(ElementsAre(10, 20, 30, 40, 50)));
}

TEST(TensorBufferUtilTest, CopyTensorBufferToHostMemoryCompactsPaddedBuffer) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, litert::Environment::Create({}));
  const RankedTensorType tensor_type = MakeRankedTensorType<int32_t>({5});
  constexpr size_t kPackedSize = 5 * sizeof(int32_t);
  constexpr size_t kPaddedSize = 128;

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto padded_buffer,
      TensorBuffer::CreateManaged(env, TensorBufferType::kHostMemory,
                                  tensor_type, kPaddedSize));
  const std::vector<int32_t> data = {10, 20, 30, 40, 50};
  LITERT_ASSERT_OK(padded_buffer.Write<int32_t>(data));

  LITERT_ASSERT_OK_AND_ASSIGN(auto host_copy,
                              CopyTensorBufferToHostMemory(env, padded_buffer));

  EXPECT_THAT(host_copy.BufferType(),
              IsOkAndHolds(Eq(TensorBufferType::kHostMemory)));
  EXPECT_THAT(host_copy.Size(), IsOkAndHolds(Eq(kPackedSize)));
  EXPECT_THAT(host_copy.PackedSize(), IsOkAndHolds(Eq(kPackedSize)));
  EXPECT_THAT(TensorBufferDims(host_copy), IsOkAndHolds(ElementsAre(5)));
  EXPECT_THAT(CopyFromTensorBuffer<int32_t>(host_copy),
              IsOkAndHolds(ElementsAre(10, 20, 30, 40, 50)));

  const std::vector<int32_t> updated_data = {60, 70, 80, 90, 100};
  LITERT_ASSERT_OK(padded_buffer.Write<int32_t>(updated_data));
  EXPECT_THAT(CopyFromTensorBuffer<int32_t>(host_copy),
              IsOkAndHolds(ElementsAre(10, 20, 30, 40, 50)));
}

TEST(TensorBufferUtilTest,
     CopyAndRestoreStridedTensorBufferPreservesStridedSpan) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, litert::Environment::Create({}));
  // Shape [2, 3] with row stride 6 (3 active + 3 padded elements per row):
  // packed = 6 * 4 = 24 bytes, strided active span = ((2 - 1) * 6 + 3) * 4 =
  // 36 bytes, allocated inside a 128-byte hardware-aligned buffer.
  const RankedTensorType strided_type(
      ElementType::Int32, Layout(Dimensions({2, 3}), Strides({6, 1})));
  constexpr size_t kPackedSize = 6 * sizeof(int32_t);
  constexpr size_t kStridedSpanElements = 9;
  constexpr size_t kStridedSpanBytes = kStridedSpanElements * sizeof(int32_t);
  constexpr size_t kPaddedSize = 128;

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto strided_buffer,
      TensorBuffer::CreateManaged(env, TensorBufferType::kHostMemory,
                                  strided_type, kPaddedSize));
  const std::vector<int32_t> strided_words = {11, 22, 33, -1, -2,
                                              -3, 44, 55, 66};
  {
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto lock_and_addr,
        TensorBufferScopedLock::Create(strided_buffer,
                                       TensorBuffer::LockMode::kWrite));
    std::memcpy(lock_and_addr.second, strided_words.data(), kStridedSpanBytes);
  }

  auto read_words = [](TensorBuffer& buf, size_t count) {
    auto lock_and_addr =
        TensorBufferScopedLock::Create(buf, TensorBuffer::LockMode::kRead);
    const auto* ptr = static_cast<const int32_t*>(lock_and_addr->second);
    return std::vector<int32_t>(ptr, ptr + count);
  };

  LITERT_ASSERT_OK_AND_ASSIGN(auto full_copy,
                              CopyTensorBuffer(env, strided_buffer));
  EXPECT_THAT(full_copy.Size(), IsOkAndHolds(Eq(kPaddedSize)));
  EXPECT_THAT(full_copy.PackedSize(), IsOkAndHolds(Eq(kPackedSize)));
  EXPECT_THAT(read_words(full_copy, kStridedSpanElements),
              ElementsAre(11, 22, 33, -1, -2, -3, 44, 55, 66));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto host_copy, CopyTensorBufferToHostMemory(env, strided_buffer));
  EXPECT_THAT(host_copy.BufferType(),
              IsOkAndHolds(Eq(TensorBufferType::kHostMemory)));
  EXPECT_THAT(host_copy.Size(), IsOkAndHolds(Eq(kStridedSpanBytes)));
  EXPECT_THAT(host_copy.PackedSize(), IsOkAndHolds(Eq(kPackedSize)));
  LITERT_ASSERT_OK_AND_ASSIGN(auto host_type, host_copy.TensorType());
  EXPECT_TRUE(host_type.Layout().HasStrides());
  EXPECT_THAT(host_type.Layout().Strides(), ElementsAre(6, 1));
  EXPECT_THAT(read_words(host_copy, kStridedSpanElements),
              ElementsAre(11, 22, 33, -1, -2, -3, 44, 55, 66));

  // Overwrite `strided_buffer` and restore in place from the compacted host
  // snapshot without reallocating the destination buffer handle.
  const auto* original_handle = strided_buffer.Get();
  {
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto lock_and_addr,
        TensorBufferScopedLock::Create(strided_buffer,
                                       TensorBuffer::LockMode::kWrite));
    std::memset(lock_and_addr.second, 0, kPaddedSize);
  }
  ASSERT_OK(CopyTensorBufferContents(strided_buffer, host_copy));
  EXPECT_EQ(strided_buffer.Get(), original_handle);
  EXPECT_THAT(strided_buffer.Size(), IsOkAndHolds(Eq(kPaddedSize)));
  EXPECT_THAT(read_words(strided_buffer, kStridedSpanElements),
              ElementsAre(11, 22, 33, -1, -2, -3, 44, 55, 66));
}

// Reads the first `count` bytes exposed by locking `buffer`.
std::vector<uint8_t> ReadLockedBytes(TensorBuffer& buffer, size_t count) {
  auto lock_and_addr =
      TensorBufferScopedLock::Create(buffer, TensorBuffer::LockMode::kRead);
  const auto* ptr = static_cast<const uint8_t*>(lock_and_addr->second);
  return std::vector<uint8_t>(ptr, ptr + count);
}

// Overwrites the first `bytes.size()` bytes exposed by locking `buffer`.
void WriteLockedBytes(TensorBuffer& buffer, absl::Span<const uint8_t> bytes) {
  auto lock_and_addr =
      TensorBufferScopedLock::Create(buffer, TensorBuffer::LockMode::kWrite);
  ASSERT_TRUE(lock_and_addr);
  std::memcpy(lock_and_addr->second, bytes.data(), bytes.size());
}

TEST(TensorBufferUtilTest,
     CopyTensorBufferToHostMemoryZeroStrideZeroesTailPadding) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, litert::Environment::Create({}));
  // Shape [2, 3] with a broadcast row stride of 0: both rows alias the same 3
  // elements, so the active span is 3 * 4 = 12 bytes, smaller than the packed
  // size of 6 * 4 = 24 bytes.
  const RankedTensorType broadcast_type(
      ElementType::Int32, Layout(Dimensions({2, 3}), Strides({0, 1})));
  constexpr size_t kPackedSize = 6 * sizeof(int32_t);
  constexpr size_t kActiveSpanBytes = 3 * sizeof(int32_t);

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto broadcast_buffer,
      TensorBuffer::CreateManaged(env, TensorBufferType::kHostMemory,
                                  broadcast_type, kPackedSize));
  const std::vector<uint8_t> source_bytes(kPackedSize, 0xAB);
  ASSERT_NO_FATAL_FAILURE(WriteLockedBytes(broadcast_buffer, source_bytes));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto host_copy, CopyTensorBufferToHostMemory(env, broadcast_buffer));
  EXPECT_THAT(host_copy.BufferType(),
              IsOkAndHolds(Eq(TensorBufferType::kHostMemory)));
  EXPECT_THAT(host_copy.Size(), IsOkAndHolds(Eq(kPackedSize)));
  EXPECT_THAT(host_copy.PackedSize(), IsOkAndHolds(Eq(kPackedSize)));
  std::vector<uint8_t> expected_bytes(kPackedSize, 0);
  std::fill(expected_bytes.begin(), expected_bytes.begin() + kActiveSpanBytes,
            0xAB);
  EXPECT_EQ(ReadLockedBytes(host_copy, kPackedSize), expected_bytes);

  // Restoring copies only the 12-byte active span back in place.
  ASSERT_NO_FATAL_FAILURE(
      WriteLockedBytes(broadcast_buffer, std::vector<uint8_t>(kPackedSize, 0)));
  ASSERT_OK(CopyTensorBufferContents(broadcast_buffer, host_copy));
  EXPECT_EQ(ReadLockedBytes(broadcast_buffer, kPackedSize), expected_bytes);
}

TEST(TensorBufferUtilTest, CopyAndRestoreOddSpanStridedInt4TensorBuffer) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, litert::Environment::Create({}));
  // Shape [2, 3] Int4 with row stride 4: packed = ceil(6 / 2) = 3 bytes, odd
  // active span = (2 - 1) * 4 + 3 = 7 elements = ceil(7 / 2) = 4 bytes,
  // allocated inside an 8-byte buffer.
  const RankedTensorType int4_type(ElementType::Int4,
                                   Layout(Dimensions({2, 3}), Strides({4, 1})));
  constexpr size_t kPackedSize = 3;
  constexpr size_t kStridedSpanBytes = 4;
  constexpr size_t kPaddedSize = 8;

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto int4_buffer,
      TensorBuffer::CreateManaged(env, TensorBufferType::kHostMemory, int4_type,
                                  kPaddedSize));
  const std::vector<uint8_t> source_bytes = {0x21, 0x43, 0x65, 0x87,
                                             0xA9, 0xCB, 0xED, 0x0F};
  ASSERT_NO_FATAL_FAILURE(WriteLockedBytes(int4_buffer, source_bytes));

  LITERT_ASSERT_OK_AND_ASSIGN(auto host_copy,
                              CopyTensorBufferToHostMemory(env, int4_buffer));
  EXPECT_THAT(host_copy.Size(), IsOkAndHolds(Eq(kStridedSpanBytes)));
  EXPECT_THAT(host_copy.PackedSize(), IsOkAndHolds(Eq(kPackedSize)));
  EXPECT_THAT(ReadLockedBytes(host_copy, kStridedSpanBytes),
              ElementsAre(0x21, 0x43, 0x65, 0x87));

  ASSERT_NO_FATAL_FAILURE(
      WriteLockedBytes(int4_buffer, std::vector<uint8_t>(kPaddedSize, 0)));
  ASSERT_OK(CopyTensorBufferContents(int4_buffer, host_copy));
  EXPECT_THAT(ReadLockedBytes(int4_buffer, kPaddedSize),
              ElementsAre(0x21, 0x43, 0x65, 0x87, 0, 0, 0, 0));
}

TEST(TensorBufferUtilTest, CopyAndRestoreEmptyStridedTensorBuffer) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, litert::Environment::Create({}));
  const RankedTensorType empty_type(
      ElementType::Int32, Layout(Dimensions({0, 3}), Strides({3, 1})));
  constexpr size_t kPaddedSize = 16;

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto empty_buffer,
      TensorBuffer::CreateManaged(env, TensorBufferType::kHostMemory,
                                  empty_type, kPaddedSize));

  LITERT_ASSERT_OK_AND_ASSIGN(auto full_copy,
                              CopyTensorBuffer(env, empty_buffer));
  EXPECT_THAT(full_copy.Size(), IsOkAndHolds(Eq(kPaddedSize)));
  EXPECT_THAT(full_copy.PackedSize(), IsOkAndHolds(Eq(0)));

  LITERT_ASSERT_OK_AND_ASSIGN(auto host_copy,
                              CopyTensorBufferToHostMemory(env, empty_buffer));
  EXPECT_THAT(host_copy.BufferType(),
              IsOkAndHolds(Eq(TensorBufferType::kHostMemory)));
  EXPECT_THAT(host_copy.Size(), IsOkAndHolds(Eq(0)));
  EXPECT_THAT(host_copy.PackedSize(), IsOkAndHolds(Eq(0)));

  EXPECT_OK(CopyTensorBufferContents(empty_buffer, host_copy));
}

TEST(TensorBufferUtilTest, RejectsUndersizedStridedOrMismatchedTensorBuffers) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, litert::Environment::Create({}));
  const RankedTensorType strided_type(
      ElementType::Int32, Layout(Dimensions({2, 3}), Strides({6, 1})));
  constexpr size_t kPackedOnlySize = 6 * sizeof(int32_t);
  constexpr size_t kValidStridedSize = 9 * sizeof(int32_t);

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto undersized_strided,
      TensorBuffer::CreateManaged(env, TensorBufferType::kHostMemory,
                                  strided_type, kPackedOnlySize));
  EXPECT_THAT(CopyTensorBuffer(env, undersized_strided),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(CopyTensorBufferToHostMemory(env, undersized_strided),
              StatusIs(absl::StatusCode::kInvalidArgument));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto valid_strided,
      TensorBuffer::CreateManaged(env, TensorBufferType::kHostMemory,
                                  strided_type, kValidStridedSize));
  EXPECT_THAT(CopyTensorBufferContents(valid_strided, undersized_strided),
              StatusIs(absl::StatusCode::kInvalidArgument));

  LITERT_ASSERT_OK_AND_ASSIGN(auto unstrided,
                              CreateTensorBuffer<int32_t>({2, 3}));
  EXPECT_THAT(CopyTensorBufferContents(valid_strided, unstrided),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(TensorBufferUtilTest,
     WrapOrCreateTensorBufferFromHostMemory_SizeMismatch) {
  auto tensor_type = MakeRankedTensorType<float>({1, 1536});
  std::vector<float> host_data(2560, 0.0f);
  absl::Span<float> host_span = absl::MakeSpan(host_data);

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto maybe_wrapped,
      WrapOrCreateTensorBufferFromHostMemory(tensor_type, host_span));

  // The behavior depends on whether the underlying runtime allows wrapping
  // mismatched sizes. If it fails, it should return a non-wrapped managed
  // buffer.
  if (!maybe_wrapped.wrapped) {
    LITERT_ASSERT_OK_AND_ASSIGN(auto packed_size,
                                maybe_wrapped.buffer.PackedSize());
    EXPECT_EQ(packed_size, 1536 * sizeof(float));
  }
}

}  // namespace
}  // namespace litert::lm
