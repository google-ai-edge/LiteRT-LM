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

#include <cstdint>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/test/matchers.h"  // from @litert
#include "runtime/util/convert_tensor_buffer.h"

namespace litert::lm {
namespace {

using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::litert::IsOkAndHolds;

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
