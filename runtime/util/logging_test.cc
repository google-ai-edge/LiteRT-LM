#include "runtime/util/logging.h"

#include <cstdint>
#include <optional>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

#include <gtest/gtest.h>
#include "litert/c/litert_tensor_buffer.h"  // from @litert
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/util/logging_tensor_buffer.h"

namespace litert::lm {
namespace {

TEST(LoggingTest, LogVector) {
  std::vector<int> data = {1, 2, 3, 4, 5};
  std::stringstream oss;
  oss << data;
  EXPECT_EQ(oss.str(), "vector of 5 elements: [1, 2, 3, 4, 5]");
}

TEST(LoggingTest, LogOptional) {
  std::optional<int> data = std::nullopt;
  std::stringstream oss;
  oss << data;
  EXPECT_EQ(oss.str(), "Not set.");

  // Test with a value.
  oss.str("");
  data = 10;
  oss << data;
}

TEST(LoggingTest, LogVariant) {
  std::variant<int, std::string> data1 = 10;
  std::stringstream oss;
  oss << data1;
  EXPECT_EQ(oss.str(), "10");

  // Test with a string.
  std::variant<int, std::string> data2 = "hello";
  oss.str("");
  oss << data2;
  EXPECT_EQ(oss.str(), "hello");
}

TEST(LoggingTest, LogTensorBuffer_None) {
  std::stringstream oss;
  oss << ::litert::TensorBuffer();
  EXPECT_EQ(oss.str(), "TensorBuffer: [tensor in non-host memory type=0]");
}

TEST(LoggingTest, LogTensorBuffer_Vector) {
  struct alignas(LITERT_HOST_MEMORY_BUFFER_ALIGNMENT) {
    int d[5] = {1, 2, -3, 4, 5};
  } data;

  auto tensor_buffer = ::litert::TensorBuffer::CreateFromHostMemory(
      ::litert::RankedTensorType(::litert::ElementType::Int32,
                                 ::litert::Layout(::litert::Dimensions({5}))),
      data.d, 5 * sizeof(int));
  EXPECT_TRUE(tensor_buffer.HasValue());

  std::stringstream oss;
  oss << *tensor_buffer;
  EXPECT_EQ(oss.str(), "TensorBuffer: [1, 2, -3, 4, 5] shape=(5)");
}

TEST(LoggingTest, LogTensorBuffer_Vector_Int8) {
  struct alignas(LITERT_HOST_MEMORY_BUFFER_ALIGNMENT) {
    int8_t d[5] = {1, 2, -3, 4, 5};
  } data;

  auto tensor_buffer = ::litert::TensorBuffer::CreateFromHostMemory(
      ::litert::RankedTensorType(::litert::ElementType::Int8,
                                 ::litert::Layout(::litert::Dimensions({5}))),
      data.d, 5 * sizeof(int8_t));
  EXPECT_TRUE(tensor_buffer.HasValue());

  std::stringstream oss;
  oss << *tensor_buffer;
  EXPECT_EQ(oss.str(), "TensorBuffer: [1, 2, -3, 4, 5] shape=(5)");
}

TEST(LoggingTest, LogTensorBuffer_Vector_Int16) {
  struct alignas(LITERT_HOST_MEMORY_BUFFER_ALIGNMENT) {
    int16_t d[5] = {1, 2, -3, 4, 5};
  } data;

  auto tensor_buffer = ::litert::TensorBuffer::CreateFromHostMemory(
      ::litert::RankedTensorType(::litert::ElementType::Int16,
                                 ::litert::Layout(::litert::Dimensions({5}))),
      data.d, 5 * sizeof(int16_t));
  EXPECT_TRUE(tensor_buffer.HasValue());

  std::stringstream oss;
  oss << *tensor_buffer;
  EXPECT_EQ(oss.str(), "TensorBuffer: [1, 2, -3, 4, 5] shape=(5)");
}

TEST(LoggingTest, LogTensorBuffer_Vector_Float) {
  struct alignas(LITERT_HOST_MEMORY_BUFFER_ALIGNMENT) {
    float d[5] = {1.1, 2.2, -3.3, 4.4, 5.5};
  } data;

  auto tensor_buffer = ::litert::TensorBuffer::CreateFromHostMemory(
      ::litert::RankedTensorType(::litert::ElementType::Float32,
                                 ::litert::Layout(::litert::Dimensions({5}))),
      data.d, 5 * sizeof(float));
  EXPECT_TRUE(tensor_buffer.HasValue());

  std::stringstream oss;
  oss << *tensor_buffer;
  EXPECT_EQ(oss.str(), "TensorBuffer: [1.1, 2.2, -3.3, 4.4, 5.5] shape=(5)");
}

TEST(LoggingTest, LogTensorBuffer_Matrix) {
  struct alignas(LITERT_HOST_MEMORY_BUFFER_ALIGNMENT) {
    int d[12] = {1, 2, -3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  } data;

  auto tensor_buffer = ::litert::TensorBuffer::CreateFromHostMemory(
      ::litert::RankedTensorType(
          ::litert::ElementType::Int32,
          ::litert::Layout(::litert::Dimensions({3, 4}))),
      data.d, 12 * sizeof(int));
  EXPECT_TRUE(tensor_buffer.HasValue());

  std::stringstream oss;
  oss << *tensor_buffer;
  EXPECT_EQ(oss.str(),
            "TensorBuffer: [[1, 2, -3, 4], [5, 6, 7, 8], [9, 10, 11, 12]] "
            "shape=(3, 4)");
}

TEST(LoggingTest, LogTensorBuffer_Tensor) {
  struct alignas(LITERT_HOST_MEMORY_BUFFER_ALIGNMENT) {
    float d[12] = {1.1, 2.2, -3.3, 4.4,   5.5,   6.6,
                   7.7, 8.8, 9.9,  10.10, 11.11, 12.12};
  } data;

  auto tensor_buffer = ::litert::TensorBuffer::CreateFromHostMemory(
      ::litert::RankedTensorType(
          ::litert::ElementType::Float32,
          ::litert::Layout(::litert::Dimensions({2, 3, 2}))),
      data.d, 12 * sizeof(float));
  EXPECT_TRUE(tensor_buffer.HasValue());

  std::stringstream oss;
  oss << *tensor_buffer;
  EXPECT_EQ(oss.str(),
            "TensorBuffer: [[[1.1, 2.2], [-3.3, 4.4], [5.5, 6.6]], [[7.7, 8.8],"
            " [9.9, 10.1], [11.11, 12.12]]] shape=(2, 3, 2)");
}

}  // namespace
}  // namespace litert::lm
