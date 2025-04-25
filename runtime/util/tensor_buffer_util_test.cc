#include "third_party/odml/litert_lm/runtime/util/tensor_buffer_util.h"
#include <cstdint>

#include <gtest/gtest.h>
#include "third_party/odml/litert_lm/runtime/util/convert_tensor_buffer.h"

namespace litert::lm {
namespace {

TEST(TensorBufferUtilTest, NumSignificantDims) {
  auto tensor_buffer = CreateTensorBuffer<int8_t>({2, 5});
  EXPECT_EQ(NumSignificantDims(*tensor_buffer), 2);
  tensor_buffer = CreateTensorBuffer<int8_t>({2, 1, 5});
  EXPECT_EQ(NumSignificantDims(*tensor_buffer), 2);
  tensor_buffer = CreateTensorBuffer<int8_t>({1, 1, 5});
  EXPECT_EQ(NumSignificantDims(*tensor_buffer), 1);
}

}  // namespace
}  // namespace litert::lm
