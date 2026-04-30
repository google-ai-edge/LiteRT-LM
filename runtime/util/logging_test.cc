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

#include "runtime/util/logging.h"

#include <cstdint>
#include <optional>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

#include <gtest/gtest.h>
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/cc/litert_tensor_buffer_types.h"  // from @litert
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
  EXPECT_EQ(oss.str(), "Not set");

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
  struct alignas(::litert::kHostMemoryBufferAlignment) {
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
  struct alignas(::litert::kHostMemoryBufferAlignment) {
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
  struct alignas(::litert::kHostMemoryBufferAlignment) {
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
  struct alignas(::litert::kHostMemoryBufferAlignment) {
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
  struct alignas(::litert::kHostMemoryBufferAlignment) {
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
  struct alignas(::litert::kHostMemoryBufferAlignment) {
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

TEST(LoggingTest, MinLogSeverity) {
  // Initially should be nullopt or whatever it was set to by previous tests.
  // We'll set it and verify.
  SetMinLogSeverity(LogSeverity::kInfo);
  auto severity = GetMinLogSeverity();
  ASSERT_TRUE(severity.has_value());
  // kLiteRtLogSeverityInfo is usually 1 (check litert_logging.h if unsure,
  // but we can just check it's consistent).
  auto info_val = *severity;

  SetMinLogSeverity(LogSeverity::kError);
  severity = GetMinLogSeverity();
  ASSERT_TRUE(severity.has_value());
  EXPECT_NE(*severity, info_val);

  SetMinLogSeverity(LogSeverity::kSilent);
  severity = GetMinLogSeverity();
  ASSERT_TRUE(severity.has_value());
  EXPECT_NE(*severity, info_val);
}

}  // namespace
}  // namespace litert::lm
