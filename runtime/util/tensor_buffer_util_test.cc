#include "runtime/util/tensor_buffer_util.h"

#include <cstdint>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "litert/test/matchers.h"  // from @litert
#include "runtime/util/convert_tensor_buffer.h"

namespace litert::lm {
namespace {

using ::testing::Eq;

TEST(TensorBufferUtilTest, NumSignificantDims) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto tensor_buffer,
                              CreateTensorBuffer<int8_t>({2, 5}));
  EXPECT_THAT(NumSignificantDims(tensor_buffer), Eq(2));
  LITERT_ASSERT_OK_AND_ASSIGN(tensor_buffer,
                              CreateTensorBuffer<int8_t>({2, 1, 5}));
  EXPECT_THAT(NumSignificantDims(tensor_buffer), Eq(2));
  LITERT_ASSERT_OK_AND_ASSIGN(tensor_buffer,
                              CreateTensorBuffer<int8_t>({1, 1, 5}));
  EXPECT_THAT(NumSignificantDims(tensor_buffer), Eq(1));
}

}  // namespace
}  // namespace litert::lm
