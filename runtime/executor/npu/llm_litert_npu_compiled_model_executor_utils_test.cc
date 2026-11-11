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

#include "runtime/executor/npu/llm_litert_npu_compiled_model_executor_utils.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "litert/cc/litert_buffer_ref.h"  // from @litert
#include "litert/cc/litert_common.h"  // from @litert
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_expected.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/cc/litert_tensor_buffer_types.h"  // from @litert
#include "litert/test/matchers.h"  // from @litert
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/litert_compiled_model_executor_utils.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/executor/llm_executor_processed_tokens.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/executor/llm_litert_npu_compiled_model_executor.h"
#include "runtime/executor/npu/llm_litert_npu_embedder.h"
#include "runtime/executor/npu/llm_litert_npu_kv_cache.h"
#include "runtime/executor/npu/llm_litert_npu_mask.h"
#include "runtime/executor/npu/llm_litert_npu_rope.h"
#include "runtime/proto/executor_metadata.pb.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep
#include "tflite/schema/schema_generated.h"  // from @litert

namespace litert::lm {
namespace {

using ::litert::ElementType;
using ::litert::Layout;
using ::litert::RankedTensorType;
using ::litert::TensorBuffer;
using ::litert::TensorBufferScopedLock;
using ::testing::_;  // NOLINT: Required by ASSERT_OK_AND_ASSIGN().
using ::testing::Each;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::HasSubstr;
using ::testing::status::StatusIs;

template <typename T>
int ReferenceFindMaxIndex(const std::vector<T>& data) {
  if (data.empty()) return 0;
  int max_idx = 0;
  T max_val = std::numeric_limits<T>::lowest();
  for (int i = 0; i < (int)data.size(); ++i) {
    if (data[i] > max_val) {
      max_val = data[i];
      max_idx = i;
    }
  }
  return max_idx;
}

class ExecutorUtilsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto env_expected = ::litert::Environment::Create({});
    ASSERT_TRUE(env_expected.HasValue());
    env_.emplace(std::move(*env_expected));
  }

  template <typename T>
  TensorBuffer CreateTensorBuffer(const std::vector<T>& data,
                                  ElementType type) {
    return CreateTensorBufferWithDims(data, type, {1, 1, (int32_t)data.size()});
  }

  template <typename T>
  TensorBuffer CreateTensorBufferWithDims(const std::vector<T>& data,
                                          ElementType type,
                                          std::vector<int32_t> dims) {
    ::litert::Dimensions dimensions;
    for (auto d : dims) dimensions.push_back(d);
    RankedTensorType tensor_type(type, Layout(std::move(dimensions)));
    auto buffer_expected = TensorBuffer::CreateManaged(
        *env_, ::litert::TensorBufferType::kHostMemory, tensor_type,
        data.size() * sizeof(T));
    ABSL_CHECK(buffer_expected.HasValue());
    TensorBuffer buffer = std::move(*buffer_expected);
    auto lock_expected = TensorBufferScopedLock::Create<T>(
        buffer, TensorBuffer::LockMode::kWrite);
    ABSL_CHECK(lock_expected.HasValue());
    std::memcpy(lock_expected->second, data.data(), data.size() * sizeof(T));
    return buffer;
  }

  template <typename T>
  void RunSophisticatedTest(ElementType type, int size) {
    std::vector<T> data(size);
    std::mt19937 gen(42);
    if constexpr (std::is_floating_point_v<T>) {
      std::uniform_real_distribution<T> dis(-100.0, 100.0);
      for (int i = 0; i < size; ++i) data[i] = dis(gen);
    } else {
      std::uniform_int_distribution<int> dis(
          static_cast<int>(std::numeric_limits<T>::lowest()),
          static_cast<int>(std::numeric_limits<T>::max()) - 2);
      for (int i = 0; i < size; ++i) data[i] = static_cast<T>(dis(gen));
    }

    for (bool use_neon : {false, true}) {
      // Edge cases: max at start, middle, end
      for (int pos : {0, size / 2, size - 1}) {
        std::vector<T> current_data = data;
        T current_max =
            *std::max_element(current_data.begin(), current_data.end());
        current_data[pos] = current_max + 1;
        TensorBuffer buffer = CreateTensorBuffer(current_data, type);
        LITERT_ASSERT_OK_AND_ASSIGN(auto max_idx,
                                    FindMaxIndex<T>(buffer, use_neon));
        EXPECT_EQ(max_idx, pos) << "Failed at pos " << pos << " for size "
                                << size << " use_neon=" << use_neon;
      }

      // Multiple occurrences
      std::vector<T> current_data = data;
      T current_max =
          *std::max_element(current_data.begin(), current_data.end());
      int first_pos = size / 4;
      int second_pos = size / 2;
      current_data[first_pos] = current_max + 2;
      current_data[second_pos] = current_max + 2;
      TensorBuffer buffer = CreateTensorBuffer(current_data, type);
      LITERT_ASSERT_OK_AND_ASSIGN(auto max_idx,
                                  FindMaxIndex<T>(buffer, use_neon));
      // Our implementation should return the first occurrence
      EXPECT_EQ(max_idx, first_pos) << "Failed multiple occurrences for size "
                                    << size << " use_neon=" << use_neon;
    }
  }

  std::optional<::litert::Environment> env_;
};

TEST_F(ExecutorUtilsTest, FindMaxIndexFloat32Large) {
  RunSophisticatedTest<float>(ElementType::Float32, 1027);
}

TEST_F(ExecutorUtilsTest, FindMaxIndexInt16Large) {
  RunSophisticatedTest<int16_t>(ElementType::Int16, 1033);
}

TEST_F(ExecutorUtilsTest, FindMaxIndexInt8Large) {
  RunSophisticatedTest<int8_t>(ElementType::Int8, 1041);
}

TEST_F(ExecutorUtilsTest, CrossVerifyFloat32) {
  int size = 512;
  std::vector<float> data(size);
  std::mt19937 gen(123);
  std::uniform_real_distribution<float> dis(-1.0, 1.0);
  for (int i = 0; i < size; ++i) data[i] = dis(gen);

  TensorBuffer buffer = CreateTensorBuffer(data, ElementType::Float32);
  for (bool use_neon : {false, true}) {
    LITERT_ASSERT_OK_AND_ASSIGN(auto max_idx,
                                FindMaxIndex<float>(buffer, use_neon));
    EXPECT_EQ(max_idx, ReferenceFindMaxIndex(data)) << "use_neon=" << use_neon;
  }
}

TEST_F(ExecutorUtilsTest, CrossVerifyInt16) {
  int size = 512;
  std::vector<int16_t> data(size);
  std::mt19937 gen(123);
  std::uniform_int_distribution<int16_t> dis(-1000, 1000);
  for (int i = 0; i < size; ++i) data[i] = dis(gen);

  TensorBuffer buffer = CreateTensorBuffer(data, ElementType::Int16);
  for (bool use_neon : {false, true}) {
    LITERT_ASSERT_OK_AND_ASSIGN(auto max_idx,
                                FindMaxIndex<int16_t>(buffer, use_neon));
    EXPECT_EQ(max_idx, ReferenceFindMaxIndex(data)) << "use_neon=" << use_neon;
  }
}

TEST_F(ExecutorUtilsTest, CrossVerifyInt8) {
  int size = 512;
  std::vector<int8_t> data(size);
  std::mt19937 gen(123);
  std::uniform_int_distribution<int> dis(-100, 100);
  for (int i = 0; i < size; ++i) data[i] = static_cast<int8_t>(dis(gen));

  TensorBuffer buffer = CreateTensorBuffer(data, ElementType::Int8);
  for (bool use_neon : {false, true}) {
    LITERT_ASSERT_OK_AND_ASSIGN(auto max_idx,
                                FindMaxIndex<int8_t>(buffer, use_neon));
    EXPECT_EQ(max_idx, ReferenceFindMaxIndex(data)) << "use_neon=" << use_neon;
  }
}

TEST_F(ExecutorUtilsTest, ApplyGreedySamplingCrossVerify) {
  std::vector<float> data = {0.1f, 0.9f, 0.4f};
  TensorBuffer buffer = CreateTensorBuffer(data, ElementType::Float32);
  for (bool use_neon : {false, true}) {
    LITERT_ASSERT_OK_AND_ASSIGN(auto sample_idx,
                                ApplyGreedySampling(buffer, use_neon));
    EXPECT_EQ(sample_idx, 1) << "use_neon=" << use_neon;
  }
}

#if defined(__x86_64__) || defined(_M_X64)

TEST_F(ExecutorUtilsTest, FindMaxIndexSse2FloatBasic) {
  // 17 elements: 4 SIMD iterations + 1 scalar tail.
  std::vector<float> data = {1.0f,  3.0f, 2.0f,  5.0f, 4.0f, 0.0f,
                             -1.0f, 2.5f, 3.5f,  4.5f, 9.0f, 1.5f,
                             2.0f,  0.5f, -2.0f, 3.0f, 7.0f};
  EXPECT_EQ(FindMaxIndexSse2Float(data.data(), data.size()), 10);
}

TEST_F(ExecutorUtilsTest, FindMaxIndexSse2FloatEdgeCases) {
  // Empty.
  EXPECT_EQ(FindMaxIndexSse2Float(nullptr, 0), 0);

  // Single element.
  std::vector<float> single = {42.0f};
  EXPECT_EQ(FindMaxIndexSse2Float(single.data(), single.size()), 0);

  // Max at start (18 elements: 4 SIMD iterations + 2 scalar tail).
  std::vector<float> start(18, 1.0f);
  start[0] = 10.0f;
  EXPECT_EQ(FindMaxIndexSse2Float(start.data(), start.size()), 0);

  // Max at end.
  std::vector<float> end(18, 1.0f);
  end[17] = 10.0f;
  EXPECT_EQ(FindMaxIndexSse2Float(end.data(), end.size()), 17);

  // Duplicate max returns first occurrence.
  std::vector<float> dup(18, 1.0f);
  dup[5] = 5.0f;
  dup[13] = 5.0f;
  EXPECT_EQ(FindMaxIndexSse2Float(dup.data(), dup.size()), 5);

  // Negative values.
  std::vector<float> neg(18, -5.0f);
  neg[11] = -1.0f;
  EXPECT_EQ(FindMaxIndexSse2Float(neg.data(), neg.size()), 11);
}

TEST_F(ExecutorUtilsTest, FindMaxIndexSse2FloatLarge) {
  int size = 1027;  // Not a multiple of 4 to test scalar tail.
  std::vector<float> data(size);
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dis(-100.0f, 100.0f);
  for (int i = 0; i < size; ++i) data[i] = dis(gen);

  int expected = ReferenceFindMaxIndex(data);
  EXPECT_EQ(FindMaxIndexSse2Float(data.data(), size), expected);

  // Place max at various positions.
  for (int pos : {0, size / 2, size - 1}) {
    std::vector<float> d = data;
    float mx = *std::max_element(d.begin(), d.end());
    d[pos] = mx + 1.0f;
    EXPECT_EQ(FindMaxIndexSse2Float(d.data(), size), pos) << "pos=" << pos;
  }
}

TEST_F(ExecutorUtilsTest, FindMaxIndexSse2Int16Basic) {
  // 35 elements: 4 SIMD iterations + 3 scalar tail.
  std::vector<int16_t> data(35, 0);
  data[27] = 500;
  EXPECT_EQ(FindMaxIndexSse2Int16(data.data(), data.size()), 27);
}

TEST_F(ExecutorUtilsTest, FindMaxIndexSse2Int16EdgeCases) {
  // Empty.
  EXPECT_EQ(FindMaxIndexSse2Int16(nullptr, 0), 0);

  // Single element.
  std::vector<int16_t> single = {42};
  EXPECT_EQ(FindMaxIndexSse2Int16(single.data(), single.size()), 0);

  // Max at start (34 elements: 4 SIMD iterations + 2 scalar tail).
  std::vector<int16_t> start(34, 1);
  start[0] = 1000;
  EXPECT_EQ(FindMaxIndexSse2Int16(start.data(), start.size()), 0);

  // Max at end.
  std::vector<int16_t> end(34, 1);
  end[33] = 1000;
  EXPECT_EQ(FindMaxIndexSse2Int16(end.data(), end.size()), 33);

  // Duplicate max returns first occurrence.
  std::vector<int16_t> dup(34, 1);
  dup[9] = 500;
  dup[25] = 500;
  EXPECT_EQ(FindMaxIndexSse2Int16(dup.data(), dup.size()), 9);

  // Negative values.
  std::vector<int16_t> neg(34, -500);
  neg[20] = -100;
  EXPECT_EQ(FindMaxIndexSse2Int16(neg.data(), neg.size()), 20);

  // Extreme values (34 elements).
  std::vector<int16_t> extreme(34, 0);
  extreme[0] = std::numeric_limits<int16_t>::lowest();
  extreme[17] = std::numeric_limits<int16_t>::max();
  EXPECT_EQ(FindMaxIndexSse2Int16(extreme.data(), extreme.size()), 17);
}

TEST_F(ExecutorUtilsTest, FindMaxIndexSse2Int16Large) {
  int size = 1033;  // Not a multiple of 8 to test scalar tail.
  std::vector<int16_t> data(size);
  std::mt19937 gen(42);
  std::uniform_int_distribution<int16_t> dis(-1000, 1000);
  for (int i = 0; i < size; ++i) data[i] = dis(gen);

  int expected = ReferenceFindMaxIndex(data);
  EXPECT_EQ(FindMaxIndexSse2Int16(data.data(), size), expected);

  // Place max at various positions.
  for (int pos : {0, size / 2, size - 1}) {
    std::vector<int16_t> d = data;
    int16_t mx = *std::max_element(d.begin(), d.end());
    d[pos] = mx + 1;
    EXPECT_EQ(FindMaxIndexSse2Int16(d.data(), size), pos) << "pos=" << pos;
  }
}

TEST_F(ExecutorUtilsTest, FindMaxIndexSse2Int8Basic) {
  // 67 elements: 4 SIMD iterations + 3 scalar tail.
  std::vector<int8_t> data(67, 0);
  data[50] = 100;
  EXPECT_EQ(FindMaxIndexSse2Int8(data.data(), data.size()), 50);
}

TEST_F(ExecutorUtilsTest, FindMaxIndexSse2Int8EdgeCases) {
  // Empty.
  EXPECT_EQ(FindMaxIndexSse2Int8(nullptr, 0), 0);

  // Single element.
  std::vector<int8_t> single = {42};
  EXPECT_EQ(FindMaxIndexSse2Int8(single.data(), single.size()), 0);

  // Max at start (65 elements: 4 SIMD iterations + 1 scalar tail).
  std::vector<int8_t> start(65, 0);
  start[0] = 100;
  EXPECT_EQ(FindMaxIndexSse2Int8(start.data(), start.size()), 0);

  // Max at end.
  std::vector<int8_t> end(65, 0);
  end[64] = 100;
  EXPECT_EQ(FindMaxIndexSse2Int8(end.data(), end.size()), 64);

  // Duplicate max returns first occurrence.
  std::vector<int8_t> dup(65, 1);
  dup[18] = 50;
  dup[45] = 50;
  EXPECT_EQ(FindMaxIndexSse2Int8(dup.data(), dup.size()), 18);

  // Negative values.
  std::vector<int8_t> neg(65, -50);
  neg[40] = -10;
  EXPECT_EQ(FindMaxIndexSse2Int8(neg.data(), neg.size()), 40);

  // Extreme values including signed boundary (65 elements).
  std::vector<int8_t> extreme(65, 0);
  extreme[0] = std::numeric_limits<int8_t>::lowest();
  extreme[33] = std::numeric_limits<int8_t>::max();
  EXPECT_EQ(FindMaxIndexSse2Int8(extreme.data(), extreme.size()), 33);
}

TEST_F(ExecutorUtilsTest, FindMaxIndexSse2Int8Large) {
  int size = 1041;  // Not a multiple of 16 to test scalar tail.
  std::vector<int8_t> data(size);
  std::mt19937 gen(42);
  std::uniform_int_distribution<int> dis(-100, 100);
  for (int i = 0; i < size; ++i) data[i] = static_cast<int8_t>(dis(gen));

  int expected = ReferenceFindMaxIndex(data);
  EXPECT_EQ(FindMaxIndexSse2Int8(data.data(), size), expected);

  // Place max at various positions.
  for (int pos : {0, size / 2, size - 1}) {
    std::vector<int8_t> d = data;
    int8_t mx = *std::max_element(d.begin(), d.end());
    d[pos] = mx + 1;
    EXPECT_EQ(FindMaxIndexSse2Int8(d.data(), size), pos) << "pos=" << pos;
  }
}

#endif  // defined(__x86_64__) || defined(_M_X64)





TEST_F(ExecutorUtilsTest, DequantizeLogitsInt16) {
  std::vector<int16_t> quantized_data = {100, 200, -100, -200};
  float scale = 0.5f;
  int32_t zero_point = 10;

  TensorBuffer src = CreateTensorBuffer(quantized_data, ElementType::Int16);
  TensorBuffer dst =
      CreateTensorBuffer(std::vector<float>(4, 0.0f), ElementType::Float32);

  ASSERT_TRUE(DequantizeLogits(src, dst, scale, zero_point, false).ok());

  auto lock_expected =
      TensorBufferScopedLock::Create<float>(dst, TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(lock_expected.HasValue());
  auto& lock = *lock_expected;

  for (size_t i = 0; i < quantized_data.size(); ++i) {
    float expected =
        scale * (static_cast<float>(quantized_data[i]) - zero_point);
    EXPECT_NEAR(lock.second[i], expected, 1e-5);
  }
}

TEST_F(ExecutorUtilsTest, DequantizeLogitsInt8) {
  std::vector<int8_t> quantized_data = {10, 20, -10, -20};
  float scale = 0.25f;
  int32_t zero_point = -5;

  TensorBuffer src = CreateTensorBuffer(quantized_data, ElementType::Int8);
  TensorBuffer dst =
      CreateTensorBuffer(std::vector<float>(4, 0.0f), ElementType::Float32);

  ASSERT_TRUE(DequantizeLogits(src, dst, scale, zero_point, false).ok());

  auto lock_expected =
      TensorBufferScopedLock::Create<float>(dst, TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(lock_expected.HasValue());
  auto& lock = *lock_expected;

  for (size_t i = 0; i < quantized_data.size(); ++i) {
    float expected =
        scale * (static_cast<float>(quantized_data[i]) - zero_point);
    EXPECT_NEAR(lock.second[i], expected, 1e-5);
  }
}



TEST(ExecutorUtilsQuantizeTest, QuantizeInt16) {
  EXPECT_EQ(Quantize<int16_t>(10.0f, 2.0f, 5), 10);
  EXPECT_EQ(Quantize<int16_t>(9.0f, 2.0f, 5), 10);
  EXPECT_EQ(Quantize<int16_t>(-9.0f, 2.0f, 5), 0);
  EXPECT_EQ(Quantize<int16_t>(100000.0f, 1.0f, 0), 32767);
  EXPECT_EQ(Quantize<int16_t>(-100000.0f, 1.0f, 0), -32768);
}

TEST(ExecutorUtilsQuantizeTest, QuantizeInt8) {
  EXPECT_EQ(Quantize<int8_t>(10.0f, 2.0f, 5), 10);
  EXPECT_EQ(Quantize<int8_t>(9.0f, 2.0f, 5), 10);
  EXPECT_EQ(Quantize<int8_t>(-9.0f, 2.0f, 5), 0);
  EXPECT_EQ(Quantize<int8_t>(1000.0f, 1.0f, 0), 127);
  EXPECT_EQ(Quantize<int8_t>(-1000.0f, 1.0f, 0), -128);
}

TEST(ExecutorUtilsFormatFirstNTest, FormatFirstNEmpty) {
  std::vector<int> empty;
  EXPECT_EQ(FormatFirstN<int>(empty), "[]");
}

TEST(ExecutorUtilsFormatFirstNTest, FormatFirstNLessOrEqualThanLimit) {
  std::vector<int> data = {1, 2, 3, 4, 5};
  EXPECT_EQ(FormatFirstN<int>(data, 5), "[1, 2, 3, 4, 5]");
  EXPECT_EQ(FormatFirstN<int>(data, 10), "[1, 2, 3, 4, 5]");
}

TEST(ExecutorUtilsFormatFirstNTest, FormatFirstNMoreThanLimit) {
  std::vector<int> data = {1, 2, 3, 4, 5, 6};
  EXPECT_EQ(FormatFirstN<int>(data, 3), "[1, 2, 3, ...]");
  EXPECT_EQ(FormatFirstN<int>(data, 5), "[1, 2, 3, 4, 5, ...]");
}

TEST(ExecutorUtilsFormatFirstNTest, FormatFirstNFloat) {
  std::vector<float> data = {1.5f, 2.5f, 3.5f};
  EXPECT_EQ(FormatFirstN<float>(data, 2), "[1.5, 2.5, ...]");
}

TEST_F(ExecutorUtilsTest, DetectUsesRingbuffer_UniformCacheSizesReturnsFalse) {
  absl::flat_hash_map<absl::string_view, TensorBuffer> buffers;
  std::vector<float> data(64, 0.0f);
  TensorBuffer k0 =
      CreateTensorBufferWithDims(data, ElementType::Float32, {1, 1, 16, 4});
  TensorBuffer v0 =
      CreateTensorBufferWithDims(data, ElementType::Float32, {1, 1, 16, 4});
  TensorBuffer k1 =
      CreateTensorBufferWithDims(data, ElementType::Float32, {1, 1, 16, 4});
  TensorBuffer v1 =
      CreateTensorBufferWithDims(data, ElementType::Float32, {1, 1, 16, 4});
  buffers.emplace("kv_cache_k_0", std::move(k0));
  buffers.emplace("kv_cache_v_0", std::move(v0));
  buffers.emplace("kv_cache_k_1", std::move(k1));
  buffers.emplace("kv_cache_v_1", std::move(v1));

  EXPECT_FALSE(DetectUsesRingbuffer(buffers));
}

TEST_F(ExecutorUtilsTest, DetectUsesRingbuffer_MixedCacheSizesReturnsTrue) {
  absl::flat_hash_map<absl::string_view, TensorBuffer> buffers;
  std::vector<float> data_16(64, 0.0f);
  std::vector<float> data_8(32, 0.0f);
  TensorBuffer k0 =
      CreateTensorBufferWithDims(data_16, ElementType::Float32, {1, 1, 16, 4});
  TensorBuffer v0 =
      CreateTensorBufferWithDims(data_16, ElementType::Float32, {1, 1, 16, 4});
  TensorBuffer k1 =
      CreateTensorBufferWithDims(data_8, ElementType::Float32, {1, 1, 8, 4});
  TensorBuffer v1 =
      CreateTensorBufferWithDims(data_8, ElementType::Float32, {1, 1, 8, 4});
  buffers.emplace("kv_cache_k_0", std::move(k0));
  buffers.emplace("kv_cache_v_0", std::move(v0));
  buffers.emplace("kv_cache_k_1", std::move(k1));
  buffers.emplace("kv_cache_v_1", std::move(v1));

  EXPECT_TRUE(DetectUsesRingbuffer(buffers));
}

TEST_F(ExecutorUtilsTest, DetectUsesRingbuffer_MetadataLocalCacheReturnsTrue) {
  absl::flat_hash_map<absl::string_view, TensorBuffer> buffers;
  std::vector<float> data(64, 0.0f);
  TensorBuffer k0 =
      CreateTensorBufferWithDims(data, ElementType::Float32, {1, 1, 16, 4});
  buffers.emplace("kv_cache_k_0", std::move(k0));

  proto::ExecutorMetadata metadata;
  auto* sb = metadata.mutable_llm_executor_metadata()->add_state_buffers();
  sb->set_type(proto::StateBuffer::TYPE_LOCAL_KEY_CACHE);

  EXPECT_TRUE(DetectUsesRingbuffer(buffers, &metadata));
}

TEST_F(ExecutorUtilsTest,
       DetectUsesRingbuffer_MetadataGlobalCacheReturnsFalse) {
  absl::flat_hash_map<absl::string_view, TensorBuffer> buffers;
  std::vector<float> data_16(64, 0.0f);
  std::vector<float> data_8(32, 0.0f);
  TensorBuffer k0 =
      CreateTensorBufferWithDims(data_16, ElementType::Float32, {1, 1, 16, 4});
  TensorBuffer k1 =
      CreateTensorBufferWithDims(data_8, ElementType::Float32, {1, 1, 8, 4});
  buffers.emplace("kv_cache_k_0", std::move(k0));
  buffers.emplace("kv_cache_k_1", std::move(k1));

  proto::ExecutorMetadata metadata;
  auto* sb = metadata.mutable_llm_executor_metadata()->add_state_buffers();
  sb->set_type(proto::StateBuffer::TYPE_GLOBAL_KEY_CACHE);

  // Even though buffer sizes are mixed (16 vs 8), metadata says global, so
  // returns false.
  EXPECT_FALSE(DetectUsesRingbuffer(buffers, &metadata));
}

TEST_F(ExecutorUtilsTest,
       DetectUsesRingbuffer_MetadataNullFallsBackToHeuristic) {
  absl::flat_hash_map<absl::string_view, TensorBuffer> uniform_buffers;
  std::vector<float> data(64, 0.0f);
  uniform_buffers.emplace(
      "kv_cache_k_0",
      CreateTensorBufferWithDims(data, ElementType::Float32, {1, 1, 16, 4}));
  uniform_buffers.emplace(
      "kv_cache_k_1",
      CreateTensorBufferWithDims(data, ElementType::Float32, {1, 1, 16, 4}));

  EXPECT_FALSE(DetectUsesRingbuffer(uniform_buffers, nullptr));

  absl::flat_hash_map<absl::string_view, TensorBuffer> mixed_buffers;
  std::vector<float> data_8(32, 0.0f);
  mixed_buffers.emplace(
      "kv_cache_k_0",
      CreateTensorBufferWithDims(data, ElementType::Float32, {1, 1, 16, 4}));
  mixed_buffers.emplace(
      "kv_cache_k_1",
      CreateTensorBufferWithDims(data_8, ElementType::Float32, {1, 1, 8, 4}));

  EXPECT_TRUE(DetectUsesRingbuffer(mixed_buffers, nullptr));
}

TEST_F(ExecutorUtilsTest, ResolveGraphLayout_Tier3FullMetadata) {
  absl::flat_hash_map<absl::string_view, TensorBuffer> buffers;
  proto::ExecutorMetadata metadata;
  auto* llm_meta = metadata.mutable_llm_executor_metadata();
  auto* sb = llm_meta->add_state_buffers();
  sb->set_type(proto::StateBuffer::TYPE_LOCAL_KEY_CACHE);
  sb->set_minimum_sequence_length(256);
  sb->set_maximum_sequence_length(1024);
  llm_meta->mutable_attention_mask_settings()->set_sliding_window_size(256);

  NpuGraphLayout layout =
      NpuGraphLayout::Resolve(128, 1024, buffers, &metadata);
  EXPECT_EQ(layout.prefill_chunk_size, 128);
  EXPECT_EQ(layout.global_cache_length, 1024);
  EXPECT_TRUE(layout.uses_ringbuffer);
  ASSERT_TRUE(layout.local_cache_length.has_value());
  EXPECT_EQ(*layout.local_cache_length, 256);
  EXPECT_EQ(layout.sliding_window_size, 256);
}

TEST_F(ExecutorUtilsTest,
       ResolveGraphLayout_Tier2PartialMetadataMissingMaskSettings) {
  absl::flat_hash_map<absl::string_view, TensorBuffer> buffers;
  proto::ExecutorMetadata metadata;
  auto* llm_meta = metadata.mutable_llm_executor_metadata();
  auto* sb = llm_meta->add_state_buffers();
  sb->set_type(proto::StateBuffer::TYPE_LOCAL_VALUE_CACHE);
  sb->set_minimum_sequence_length(768);
  sb->set_maximum_sequence_length(2048);

  NpuGraphLayout layout =
      NpuGraphLayout::Resolve(512, 2048, buffers, &metadata);
  EXPECT_EQ(layout.prefill_chunk_size, 512);
  EXPECT_EQ(layout.global_cache_length, 2048);
  EXPECT_TRUE(layout.uses_ringbuffer);
  ASSERT_TRUE(layout.local_cache_length.has_value());
  EXPECT_EQ(*layout.local_cache_length, 768);
  // Default sliding window to local cache length when attention_mask_settings
  // is omitted
  EXPECT_EQ(layout.sliding_window_size, 768);
}

TEST_F(ExecutorUtilsTest, ResolveGraphLayout_Tier1NoMetadataHeuristicFallback) {
  absl::flat_hash_map<absl::string_view, TensorBuffer> buffers;
  std::vector<float> data_global(64, 0.0f);
  std::vector<float> data_local(16, 0.0f);
  buffers.emplace("kv_cache_k_0",
                  CreateTensorBufferWithDims(data_global, ElementType::Float32,
                                             {1, 1, 16, 4}));
  buffers.emplace("kv_cache_k_1",
                  CreateTensorBufferWithDims(data_local, ElementType::Float32,
                                             {1, 1, 4, 4}));

  NpuGraphLayout layout = NpuGraphLayout::Resolve(256, 16, buffers, nullptr);
  EXPECT_EQ(layout.prefill_chunk_size, 256);
  EXPECT_EQ(layout.global_cache_length, 16);
  EXPECT_TRUE(layout.uses_ringbuffer);
  ASSERT_TRUE(layout.local_cache_length.has_value());
  EXPECT_EQ(*layout.local_cache_length, 4);
  EXPECT_EQ(layout.sliding_window_size, 4);
}

TEST_F(ExecutorUtilsTest, ResolveGraphLayout_UniformNoRingbuffer) {
  absl::flat_hash_map<absl::string_view, TensorBuffer> buffers;
  std::vector<float> data(64, 0.0f);
  buffers.emplace(
      "kv_cache_k_0",
      CreateTensorBufferWithDims(data, ElementType::Float32, {1, 1, 16, 4}));
  buffers.emplace(
      "kv_cache_k_1",
      CreateTensorBufferWithDims(data, ElementType::Float32, {1, 1, 16, 4}));

  NpuGraphLayout layout = NpuGraphLayout::Resolve(256, 16, buffers, nullptr);
  EXPECT_EQ(layout.prefill_chunk_size, 256);
  EXPECT_EQ(layout.global_cache_length, 16);
  EXPECT_FALSE(layout.uses_ringbuffer);
  EXPECT_FALSE(layout.local_cache_length.has_value());
  EXPECT_EQ(layout.sliding_window_size, 512);
}

// Builds a drafter-like model with one signature per entry of `signatures`
// (signature key -> context size). Every signature has the inputs
// `activations` [1, 1, 8], `input_pos` [1] (unless `with_input_pos` is false),
// `mask_local` [1, 1, 1, N], `kv_cache_k_0` [1, 1, N, 4] and `kv_cache_v_0`
// [1, 1, 4, N] for context size N, and computes the output `logits` [1, 1, 8]
// as ABS(activations).
std::vector<uint8_t> BuildDrafterModelBuffer(
    const std::vector<std::pair<std::string, int32_t>>& signatures,
    bool with_input_pos = true) {
  flatbuffers::FlatBufferBuilder builder;
  auto opcode =
      tflite::CreateOperatorCode(builder, tflite::BuiltinOperator_ABS);
  auto opcodes_vec = builder.CreateVector({opcode});

  std::vector<flatbuffers::Offset<tflite::SubGraph>> subgraphs;
  std::vector<flatbuffers::Offset<tflite::SignatureDef>> sig_defs;
  for (size_t i = 0; i < signatures.size(); ++i) {
    const int32_t context_size = signatures[i].second;
    auto create_tensor = [&](std::vector<int32_t> dims, tflite::TensorType type,
                             absl::string_view name) {
      return tflite::CreateTensor(
          builder, builder.CreateVector(dims), type,
          /*buffer=*/0, builder.CreateString(name.data(), name.size()));
    };
    auto tensors_vec = builder.CreateVector(
        {create_tensor({1, 1, 8}, tflite::TensorType_FLOAT32, "activations"),
         create_tensor({1}, tflite::TensorType_INT32, "input_pos"),
         create_tensor({1, 1, 1, context_size}, tflite::TensorType_FLOAT32,
                       "mask_local"),
         create_tensor({1, 1, context_size, 4}, tflite::TensorType_FLOAT32,
                       "kv_cache_k_0"),
         create_tensor({1, 1, 4, context_size}, tflite::TensorType_FLOAT32,
                       "kv_cache_v_0"),
         create_tensor({1, 1, 8}, tflite::TensorType_FLOAT32, "logits")});

    std::vector<int32_t> op_inputs = {0};
    std::vector<int32_t> op_outputs = {5};
    auto op = tflite::CreateOperator(builder, /*opcode_index=*/0,
                                     builder.CreateVector(op_inputs),
                                     builder.CreateVector(op_outputs));
    auto ops_vec = builder.CreateVector({op});
    std::vector<int32_t> sg_inputs = {0, 2, 3, 4};
    if (with_input_pos) sg_inputs.insert(sg_inputs.begin() + 1, 1);
    std::vector<int32_t> sg_outputs = {5};
    subgraphs.push_back(tflite::CreateSubGraph(
        builder, tensors_vec, builder.CreateVector(sg_inputs),
        builder.CreateVector(sg_outputs), ops_vec,
        builder.CreateString(signatures[i].first)));

    auto create_map = [&](absl::string_view name, uint32_t tensor_index) {
      return tflite::CreateTensorMap(
          builder, builder.CreateString(name.data(), name.size()),
          tensor_index);
    };
    std::vector<flatbuffers::Offset<tflite::TensorMap>> inputs_map = {
        create_map("activations", 0), create_map("mask_local", 2),
        create_map("kv_cache_k_0", 3), create_map("kv_cache_v_0", 4)};
    if (with_input_pos) inputs_map.push_back(create_map("input_pos", 1));
    auto inputs_map_vec = builder.CreateVector(inputs_map);
    auto outputs_map_vec = builder.CreateVector({create_map("logits", 5)});
    sig_defs.push_back(tflite::CreateSignatureDef(
        builder, inputs_map_vec, outputs_map_vec,
        builder.CreateString(signatures[i].first), /*subgraph_index=*/i));
  }

  auto subgraphs_vec = builder.CreateVector(subgraphs);
  auto sig_defs_vec = builder.CreateVector(sig_defs);
  auto buffer = tflite::CreateBuffer(builder);
  auto buffers_vec = builder.CreateVector({buffer});
  auto model =
      tflite::CreateModel(builder, /*version=*/3, opcodes_vec, subgraphs_vec,
                          builder.CreateString("drafter_model"), buffers_vec,
                          /*metadata_buffer=*/0, /*metadata=*/0, sig_defs_vec);
  tflite::FinishModelBuffer(builder, model);
  return std::vector<uint8_t>(builder.GetBufferPointer(),
                              builder.GetBufferPointer() + builder.GetSize());
}

RankedTensorType MakeType(ElementType type, const std::vector<int32_t>& dims) {
  ::litert::Dimensions dimensions;
  for (auto d : dims) dimensions.push_back(d);
  return RankedTensorType(type, Layout(std::move(dimensions)));
}

std::vector<int32_t> DimensionsOf(const TensorBuffer& buffer) {
  auto type = buffer.TensorType();
  ABSL_CHECK(type.HasValue());
  auto dims = type->Layout().Dimensions();
  return std::vector<int32_t>(dims.begin(), dims.end());
}

const void* HostAddressOf(const TensorBuffer& buffer) {
  auto lock =
      TensorBufferScopedLock::Create(buffer, TensorBuffer::LockMode::kRead);
  ABSL_CHECK(lock.HasValue());
  return lock->second;
}

// Returns true when both buffers are backed by the same host memory. The
// buffers are locked one after the other as duplicates share a single lock.
bool SharesMemory(const TensorBuffer& a, const TensorBuffer& b) {
  return HostAddressOf(a) == HostAddressOf(b);
}

class DrafterContextTest : public ExecutorUtilsTest {
 protected:
  // Loads `buffer` as a model. The model does not own the bytes, so they are
  // kept alive by the fixture.
  ::litert::Model LoadModel(std::vector<uint8_t> buffer) {
    model_buffers_.push_back(std::move(buffer));
    const std::vector<uint8_t>& stored = model_buffers_.back();
    auto model = ::litert::Model::CreateFromBuffer(
        *env_, ::litert::BufferRef<uint8_t>(stored.data(), stored.size()));
    ABSL_CHECK(model.HasValue()) << model.Error().Message();
    return std::move(*model);
  }

  // A text decoder context group of `context_size` with KV cache buffers
  // matching the drafter model of `BuildDrafterModelBuffer`.
  NpuContextGroup MakeContextGroup(int context_size) {
    NpuContextGroup group;
    group.context_size = context_size;
    group.layout.global_cache_length = context_size;
    std::vector<float> data(context_size * 4, 0.0f);
    group.input_kv_cache_buffers.emplace(
        "kv_cache_k_0", CreateTensorBufferWithDims(data, ElementType::Float32,
                                                   {1, 1, context_size, 4}));
    group.input_kv_cache_buffers.emplace(
        "kv_cache_v_0", CreateTensorBufferWithDims(data, ElementType::Float32,
                                                   {1, 1, 4, context_size}));
    return group;
  }

  std::vector<std::vector<uint8_t>> model_buffers_;
};

TEST_F(DrafterContextTest, ResolveMtpSignaturesPrefersPerContextSignatures) {
  ::litert::Model model =
      LoadModel(BuildDrafterModelBuffer({{"mtp_drafter_cache_640", 640},
                                         {"mtp_drafter_cache_4096", 4096},
                                         {"mask_cache_640", 640},
                                         {"mask", 4096}}));
  EXPECT_EQ(ResolveMtpDrafterSignature(model, 640), "mtp_drafter_cache_640");
  EXPECT_EQ(ResolveMtpDrafterSignature(model, 4096), "mtp_drafter_cache_4096");
  EXPECT_EQ(ResolveMtpMaskSignature(model, 640), "mask_cache_640");
  // Context sizes without a dedicated signature, including the context size 0
  // of dynamic models, resolve to the single-context names.
  EXPECT_EQ(ResolveMtpDrafterSignature(model, 1024), "mtp_drafter");
  EXPECT_EQ(ResolveMtpDrafterSignature(model, 0), "mtp_drafter");
  EXPECT_EQ(ResolveMtpMaskSignature(model, 4096), "mask");
  EXPECT_EQ(ResolveMtpMaskSignature(model, 0), "mask");
}

TEST_F(DrafterContextTest, ResolveMtpSignaturesSingleContextModel) {
  ::litert::Model model =
      LoadModel(BuildDrafterModelBuffer({{"mtp_drafter", 4096}}));
  EXPECT_EQ(ResolveMtpDrafterSignature(model, 4096), "mtp_drafter");
  EXPECT_EQ(ResolveMtpDrafterSignature(model, 640), "mtp_drafter");
  EXPECT_EQ(ResolveMtpMaskSignature(model, 4096), "mask");
}

TEST(IsCompatibleTensorTypeTest, ComparesElementTypeRankAndStaticDimensions) {
  const RankedTensorType static_640 =
      MakeType(ElementType::Float32, {1, 1, 1, 640});
  const RankedTensorType static_4096 =
      MakeType(ElementType::Float32, {1, 1, 1, 4096});
  const RankedTensorType dynamic =
      MakeType(ElementType::Float32, {1, 1, 1, -1});
  const RankedTensorType rank_3 = MakeType(ElementType::Float32, {1, 1, 640});
  const RankedTensorType int8_640 = MakeType(ElementType::Int8, {1, 1, 1, 640});

  EXPECT_TRUE(IsCompatibleTensorType(static_640, static_640));
  EXPECT_FALSE(IsCompatibleTensorType(static_4096, static_640));
  EXPECT_FALSE(IsCompatibleTensorType(static_640, static_4096));
  // A dynamic signature dimension accepts any buffer size, but a static one
  // does not accept a dynamic buffer dimension.
  EXPECT_TRUE(IsCompatibleTensorType(dynamic, static_640));
  EXPECT_TRUE(IsCompatibleTensorType(dynamic, static_4096));
  EXPECT_FALSE(IsCompatibleTensorType(static_640, dynamic));
  EXPECT_FALSE(IsCompatibleTensorType(static_640, rank_3));
  EXPECT_FALSE(IsCompatibleTensorType(static_640, int8_640));
}

TEST_F(DrafterContextTest, CreateBindsOneDrafterSignaturePerContextGroup) {
  ::litert::Model model = LoadModel(BuildDrafterModelBuffer(
      {{"mtp_drafter_cache_640", 640}, {"mtp_drafter_cache_1024", 1024}}));
  std::vector<NpuContextGroup> groups;
  groups.push_back(MakeContextGroup(640));
  groups.push_back(MakeContextGroup(1024));

  ASSERT_OK_AND_ASSIGN(DrafterContext drafter_context,
                       DrafterContext::Create(*env_, model, groups));

  // Starts out bound to the first context group.
  ASSERT_EQ(drafter_context.context_bindings.size(), 2);
  EXPECT_EQ(drafter_context.context_bindings[0].signature,
            "mtp_drafter_cache_640");
  EXPECT_EQ(drafter_context.context_bindings[1].signature,
            "mtp_drafter_cache_1024");
  EXPECT_EQ(drafter_context.signature, "mtp_drafter_cache_640");
  EXPECT_TRUE(drafter_context.mtp_output_buffers.contains("logits"));

  // KV cache inputs are not allocated by the drafter; the bound ones alias the
  // KV cache buffers of the bound context group.
  EXPECT_FALSE(drafter_context.context_bindings[0].input_buffers.contains(
      "kv_cache_k_0"));
  EXPECT_TRUE(
      SharesMemory(drafter_context.mtp_input_buffers.at("kv_cache_k_0"),
                   groups[0].input_kv_cache_buffers.at("kv_cache_k_0")));
  EXPECT_TRUE(
      SharesMemory(drafter_context.mtp_input_buffers.at("kv_cache_v_0"),
                   groups[0].input_kv_cache_buffers.at("kv_cache_v_0")));
  // Context size dependent inputs get a buffer per context group; the others
  // are shared between the context groups.
  EXPECT_THAT(DimensionsOf(drafter_context.mtp_input_buffers.at("mask_local")),
              ElementsAre(1, 1, 1, 640));
  EXPECT_FALSE(SharesMemory(
      drafter_context.context_bindings[0].input_buffers.at("mask_local"),
      drafter_context.context_bindings[1].input_buffers.at("mask_local")));
  EXPECT_TRUE(SharesMemory(
      drafter_context.context_bindings[0].input_buffers.at("activations"),
      drafter_context.context_bindings[1].input_buffers.at("activations")));
  EXPECT_TRUE(SharesMemory(
      drafter_context.context_bindings[0].input_buffers.at("input_pos"),
      drafter_context.context_bindings[1].input_buffers.at("input_pos")));
  LITERT_EXPECT_OK(drafter_context.mtp_compiled_model.Run(
      drafter_context.signature, drafter_context.mtp_input_buffers,
      drafter_context.mtp_output_buffers));

  // Binding the second context group switches the signature, the KV cache
  // aliases and the context size dependent inputs.
  ASSERT_OK(drafter_context.BindContextGroup(1, groups[1]));
  EXPECT_EQ(drafter_context.signature, "mtp_drafter_cache_1024");
  EXPECT_TRUE(
      SharesMemory(drafter_context.mtp_input_buffers.at("kv_cache_k_0"),
                   groups[1].input_kv_cache_buffers.at("kv_cache_k_0")));
  EXPECT_THAT(DimensionsOf(drafter_context.mtp_input_buffers.at("mask_local")),
              ElementsAre(1, 1, 1, 1024));
  EXPECT_TRUE(SharesMemory(
      drafter_context.mtp_input_buffers.at("activations"),
      drafter_context.context_bindings[0].input_buffers.at("activations")));
  LITERT_EXPECT_OK(drafter_context.mtp_compiled_model.Run(
      drafter_context.signature, drafter_context.mtp_input_buffers,
      drafter_context.mtp_output_buffers));

  // Re-binding picks up re-allocated KV cache buffers of a context group.
  groups[1] = MakeContextGroup(1024);
  ASSERT_OK(drafter_context.BindContextGroup(1, groups[1]));
  EXPECT_TRUE(
      SharesMemory(drafter_context.mtp_input_buffers.at("kv_cache_v_0"),
                   groups[1].input_kv_cache_buffers.at("kv_cache_v_0")));

  EXPECT_THAT(drafter_context.BindContextGroup(2, groups[1]),
              StatusIs(absl::StatusCode::kInternal));
}

TEST_F(DrafterContextTest, CreateSingleContextDrafter) {
  ::litert::Model model =
      LoadModel(BuildDrafterModelBuffer({{"mtp_drafter", 640}}));
  std::vector<NpuContextGroup> groups;
  groups.push_back(MakeContextGroup(640));

  ASSERT_OK_AND_ASSIGN(DrafterContext drafter_context,
                       DrafterContext::Create(*env_, model, groups));
  EXPECT_EQ(drafter_context.signature, "mtp_drafter");
  EXPECT_TRUE(
      SharesMemory(drafter_context.mtp_input_buffers.at("kv_cache_k_0"),
                   groups[0].input_kv_cache_buffers.at("kv_cache_k_0")));
  LITERT_EXPECT_OK(drafter_context.mtp_compiled_model.Run(
      drafter_context.signature, drafter_context.mtp_input_buffers,
      drafter_context.mtp_output_buffers));
}

TEST_F(DrafterContextTest, SetInputPosWritesPositionWhenInputExists) {
  ::litert::Model model =
      LoadModel(BuildDrafterModelBuffer({{"mtp_drafter", 640}}));
  std::vector<NpuContextGroup> groups;
  groups.push_back(MakeContextGroup(640));

  ASSERT_OK_AND_ASSIGN(DrafterContext drafter_context,
                       DrafterContext::Create(*env_, model, groups));
  ASSERT_OK(drafter_context.SetInputPos(42));
  EXPECT_EQ(*static_cast<const int32_t*>(HostAddressOf(
                drafter_context.mtp_input_buffers.at("input_pos"))),
            42);
}

TEST_F(DrafterContextTest, SetInputPosIsNoOpWithoutInputPosInput) {
  // Converters that drop unused inputs leave `input_pos` out of drafters with
  // externalized RoPE and masks.
  ::litert::Model model = LoadModel(BuildDrafterModelBuffer(
      {{"mtp_drafter", 640}}, /*with_input_pos=*/false));
  std::vector<NpuContextGroup> groups;
  groups.push_back(MakeContextGroup(640));

  ASSERT_OK_AND_ASSIGN(DrafterContext drafter_context,
                       DrafterContext::Create(*env_, model, groups));
  EXPECT_FALSE(drafter_context.mtp_input_buffers.contains("input_pos"));
  EXPECT_OK(drafter_context.SetInputPos(42));
  LITERT_EXPECT_OK(drafter_context.mtp_compiled_model.Run(
      drafter_context.signature, drafter_context.mtp_input_buffers,
      drafter_context.mtp_output_buffers));
}

TEST_F(DrafterContextTest, CreateRejectsDrafterWithoutSignatureForContextSize) {
  ::litert::Model model =
      LoadModel(BuildDrafterModelBuffer({{"mtp_drafter_cache_640", 640}}));
  std::vector<NpuContextGroup> groups;
  groups.push_back(MakeContextGroup(640));
  groups.push_back(MakeContextGroup(1024));

  EXPECT_THAT(
      DrafterContext::Create(*env_, model, groups),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("neither signature \"mtp_drafter_cache_1024\" nor "
                         "\"mtp_drafter\" for text decoder context size "
                         "1024")));
}

TEST(MissingContextSignatureErrorTest, NamesBothCandidatesForContextSize) {
  EXPECT_THAT(MissingContextSignatureError("MTP drafter model", "mtp_drafter",
                                           "mtp_drafter", 1024),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "MTP drafter model has neither signature "
                       "\"mtp_drafter_cache_1024\" nor \"mtp_drafter\" for "
                       "text decoder context size 1024."));
  // Context size 0 (dynamic models) only ever looks up the base signature.
  EXPECT_THAT(MissingContextSignatureError("MTP auxiliary model", "mask",
                                           "mask", 0),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "MTP auxiliary model has no signature \"mask\" for "
                       "text decoder context size 0."));
  EXPECT_THAT(MissingContextSignatureError("MTP auxiliary model", "mask",
                                           "mask_cache_640", 640),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "MTP auxiliary model has no signature "
                       "\"mask_cache_640\" for text decoder context size "
                       "640."));
}

TEST_F(DrafterContextTest, CreateRejectsKVCacheShapeMismatch) {
  // A single-context drafter built for 4096 tokens cannot attend over the KV
  // cache of a 640 token context group.
  ::litert::Model model =
      LoadModel(BuildDrafterModelBuffer({{"mtp_drafter", 4096}}));
  std::vector<NpuContextGroup> groups;
  groups.push_back(MakeContextGroup(640));

  EXPECT_THAT(DrafterContext::Create(*env_, model, groups),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("provides shape [1, 1, 640, 4]")));
}

// Builds an MTP auxiliary model with a "rope" signature (`input_pos` [1] ->
// `rope_output_name` [1, 1, 8]) and one mask signature per entry of
// `mask_signatures` (signature key -> context size N, mapping `time_step` [1]
// and optional `input_tokens` [1] to `mask_output_name` [1, 1, 1, N]). Each
// signature casts its scalar int32 position/step input to float32 and adds a
// zero constant of the output shape so the output buffer is filled with the
// scalar input value.
std::vector<uint8_t> BuildMtpAuxModelBuffer(
    const std::vector<std::pair<std::string, int32_t>>& mask_signatures,
    bool with_input_tokens = true,
    absl::string_view rope_output_name = "activations",
    absl::string_view mask_output_name = "mask_local") {
  flatbuffers::FlatBufferBuilder builder;
  auto cast_opcode =
      tflite::CreateOperatorCode(builder, tflite::BuiltinOperator_CAST);
  auto add_opcode =
      tflite::CreateOperatorCode(builder, tflite::BuiltinOperator_ADD);
  auto opcodes_vec = builder.CreateVector({cast_opcode, add_opcode});

  std::vector<flatbuffers::Offset<tflite::Buffer>> buffers;
  buffers.push_back(tflite::CreateBuffer(builder));  // Buffer 0: empty.
  auto add_zeros_buffer = [&](size_t num_floats) -> uint32_t {
    const std::vector<uint8_t> zero_bytes(num_floats * sizeof(float), 0);
    buffers.push_back(
        tflite::CreateBuffer(builder, builder.CreateVector(zero_bytes)));
    return buffers.size() - 1;
  };
  auto create_tensor = [&](std::vector<int32_t> dims, tflite::TensorType type,
                           uint32_t buffer_idx, absl::string_view name) {
    return tflite::CreateTensor(
        builder, builder.CreateVector(dims), type, buffer_idx,
        builder.CreateString(name.data(), name.size()));
  };
  auto create_map = [&](absl::string_view name, uint32_t tensor_index) {
    return tflite::CreateTensorMap(
        builder, builder.CreateString(name.data(), name.size()), tensor_index);
  };

  std::vector<flatbuffers::Offset<tflite::SubGraph>> subgraphs;
  std::vector<flatbuffers::Offset<tflite::SignatureDef>> sig_defs;

  // Subgraph 0: "rope".
  {
    const uint32_t zeros_buf = add_zeros_buffer(8);
    auto tensors_vec = builder.CreateVector(
        {create_tensor({1}, tflite::TensorType_INT32, 0, "input_pos"),
         create_tensor({1}, tflite::TensorType_FLOAT32, 0, "pos_f32"),
         create_tensor({1, 1, 8}, tflite::TensorType_FLOAT32, zeros_buf,
                       "rope_zeros"),
         create_tensor({1, 1, 8}, tflite::TensorType_FLOAT32, 0,
                       rope_output_name)});
    auto cast_op = tflite::CreateOperator(builder, /*opcode_index=*/0,
                                          builder.CreateVector<int32_t>({0}),
                                          builder.CreateVector<int32_t>({1}));
    auto add_op = tflite::CreateOperator(builder, /*opcode_index=*/1,
                                         builder.CreateVector<int32_t>({1, 2}),
                                         builder.CreateVector<int32_t>({3}));
    subgraphs.push_back(tflite::CreateSubGraph(
        builder, tensors_vec, builder.CreateVector<int32_t>({0}),
        builder.CreateVector<int32_t>({3}),
        builder.CreateVector({cast_op, add_op}), builder.CreateString("rope")));
    sig_defs.push_back(tflite::CreateSignatureDef(
        builder, builder.CreateVector({create_map("input_pos", 0)}),
        builder.CreateVector({create_map(rope_output_name, 3)}),
        builder.CreateString("rope"), /*subgraph_index=*/0));
  }

  // Subgraphs 1..K: mask signatures.
  for (size_t i = 0; i < mask_signatures.size(); ++i) {
    const int32_t context_size = mask_signatures[i].second;
    const uint32_t zeros_buf = add_zeros_buffer(context_size);
    auto tensors_vec = builder.CreateVector(
        {create_tensor({1}, tflite::TensorType_INT32, 0, "time_step"),
         create_tensor({1}, tflite::TensorType_INT32, 0, "input_tokens"),
         create_tensor({1}, tflite::TensorType_FLOAT32, 0, "step_f32"),
         create_tensor({1, 1, 1, context_size}, tflite::TensorType_FLOAT32,
                       zeros_buf, "mask_zeros"),
         create_tensor({1, 1, 1, context_size}, tflite::TensorType_FLOAT32, 0,
                       mask_output_name)});
    auto cast_op = tflite::CreateOperator(builder, /*opcode_index=*/0,
                                          builder.CreateVector<int32_t>({0}),
                                          builder.CreateVector<int32_t>({2}));
    auto add_op = tflite::CreateOperator(builder, /*opcode_index=*/1,
                                         builder.CreateVector<int32_t>({2, 3}),
                                         builder.CreateVector<int32_t>({4}));
    std::vector<int32_t> sg_inputs = {0};
    std::vector<flatbuffers::Offset<tflite::TensorMap>> inputs_map = {
        create_map("time_step", 0)};
    if (with_input_tokens) {
      sg_inputs.push_back(1);
      inputs_map.push_back(create_map("input_tokens", 1));
    }
    const uint32_t subgraph_index = subgraphs.size();
    subgraphs.push_back(tflite::CreateSubGraph(
        builder, tensors_vec, builder.CreateVector(sg_inputs),
        builder.CreateVector<int32_t>({4}),
        builder.CreateVector({cast_op, add_op}),
        builder.CreateString(mask_signatures[i].first)));
    sig_defs.push_back(tflite::CreateSignatureDef(
        builder, builder.CreateVector(inputs_map),
        builder.CreateVector({create_map(mask_output_name, 4)}),
        builder.CreateString(mask_signatures[i].first), subgraph_index));
  }

  auto model = tflite::CreateModel(
      builder, /*version=*/3, opcodes_vec, builder.CreateVector(subgraphs),
      builder.CreateString("mtp_aux_model"), builder.CreateVector(buffers),
      /*metadata_buffer=*/0, /*metadata=*/0, builder.CreateVector(sig_defs));
  tflite::FinishModelBuffer(builder, model);
  return std::vector<uint8_t>(builder.GetBufferPointer(),
                              builder.GetBufferPointer() + builder.GetSize());
}

TEST_F(DrafterContextTest,
       RunDrafterMaskInvokesExplicitAndDefaultFallbackSignature) {
  const std::vector<uint8_t> aux_buffer =
      BuildMtpAuxModelBuffer({{"mask_cache_640", 640}, {"mask", 640}});
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto compiled_model,
      ::litert::CompiledModel::Create(
          *env_,
          ::litert::BufferRef<uint8_t>(aux_buffer.data(), aux_buffer.size()),
          ::litert::HwAccelerators::kCpu));

  absl::flat_hash_map<absl::string_view, TensorBuffer> in_buffers;
  absl::flat_hash_map<absl::string_view, TensorBuffer> out_buffers;
  LITERT_ASSERT_OK_AND_ASSIGN(
      in_buffers[MaskSignatures::kMaskInputTimeStep],
      compiled_model.CreateInputBuffer("mask_cache_640",
                                       MaskSignatures::kMaskInputTimeStep));
  LITERT_ASSERT_OK_AND_ASSIGN(
      in_buffers[MaskSignatures::kMaskInputTokens],
      compiled_model.CreateInputBuffer("mask_cache_640",
                                       MaskSignatures::kMaskInputTokens));
  LITERT_ASSERT_OK_AND_ASSIGN(
      out_buffers["mask_local"],
      compiled_model.CreateOutputBuffer("mask_cache_640", "mask_local"));

  NpuGraphLayout layout;
  layout.global_cache_length = 640;
  ASSERT_OK_AND_ASSIGN(NpuMask mask, NpuMask::CreateForDrafter(
                                         MaskUpdateMethod::kModel,
                                         &compiled_model, std::move(in_buffers),
                                         std::move(out_buffers), &layout));

  ASSERT_OK(mask.SetDecodeInput(/*step=*/7, /*token_id=*/11));
  ASSERT_OK(mask.RunDrafter("mask_cache_640"));
  EXPECT_EQ(*static_cast<const float*>(
                HostAddressOf(mask.Context().decode_output_buffers.at(
                    "mask_local"))),
            7.0f);

  // An empty signature falls back to the single-context "mask" signature.
  ASSERT_OK(mask.SetDecodeInput(/*step=*/13, /*token_id=*/17));
  ASSERT_OK(mask.RunDrafter());
  EXPECT_EQ(*static_cast<const float*>(
                HostAddressOf(mask.Context().decode_output_buffers.at(
                    "mask_local"))),
            13.0f);
}

TEST_F(DrafterContextTest,
       DrafterAuxContextBindsMaskSignaturesAndRealiasesOutputsAcrossGroups) {
  ::litert::Model drafter_model = LoadModel(BuildDrafterModelBuffer(
      {{"mtp_drafter_cache_640", 640}, {"mtp_drafter_cache_1024", 1024}}));
  ::litert::Model aux_model = LoadModel(BuildMtpAuxModelBuffer(
      {{"mask_cache_640", 640}, {"mask_cache_1024", 1024}}));
  std::vector<NpuContextGroup> groups;
  groups.push_back(MakeContextGroup(640));
  groups.push_back(MakeContextGroup(1024));

  ASSERT_OK_AND_ASSIGN(DrafterContext drafter_context,
                       DrafterContext::Create(*env_, drafter_model, groups));
  ASSERT_OK_AND_ASSIGN(
      DrafterAuxContext drafter_aux_context,
      DrafterAuxContext::Create(*env_, aux_model, drafter_context, groups,
                                MaskUpdateMethod::kModel));

  EXPECT_THAT(drafter_aux_context.mask_signatures,
              ElementsAre("mask_cache_640", "mask_cache_1024"));
  EXPECT_EQ(drafter_aux_context.mask_signature, "mask_cache_640");
  EXPECT_TRUE(SharesMemory(
      drafter_aux_context.drafter_rope.Context().decode_output_buffers.at(
          "activations"),
      drafter_context.mtp_input_buffers.at("activations")));
  EXPECT_TRUE(SharesMemory(
      drafter_aux_context.drafter_mask.Context().decode_output_buffers.at(
          "mask_local"),
      drafter_context.context_bindings[0].input_buffers.at("mask_local")));

  ASSERT_OK(drafter_aux_context.drafter_rope.SetDecodePosition(19));
  ASSERT_OK(drafter_aux_context.drafter_rope.RunDrafter());
  EXPECT_EQ(*static_cast<const float*>(HostAddressOf(
                drafter_context.mtp_input_buffers.at("activations"))),
            19.0f);

  ASSERT_OK(drafter_aux_context.drafter_mask.SetDecodeInput(23, 0));
  ASSERT_OK(drafter_aux_context.drafter_mask.RunDrafter(
      drafter_aux_context.mask_signature));
  EXPECT_EQ(*static_cast<const float*>(HostAddressOf(
                drafter_context.mtp_input_buffers.at("mask_local"))),
            23.0f);

  // Switching to the second context group updates the active mask signature,
  // re-points the mask output at the 1024-token drafter mask input buffer, and
  // updates the layout pointer used by kWH mask generation.
  ASSERT_OK(drafter_context.BindContextGroup(1, groups[1]));
  ASSERT_OK(drafter_aux_context.BindContextGroup(
      1, groups[1], drafter_context.mtp_input_buffers));
  EXPECT_EQ(drafter_aux_context.mask_signature, "mask_cache_1024");
  EXPECT_TRUE(SharesMemory(
      drafter_aux_context.drafter_mask.Context().decode_output_buffers.at(
          "mask_local"),
      drafter_context.context_bindings[1].input_buffers.at("mask_local")));
  ASSERT_OK(drafter_aux_context.drafter_mask.SetDecodeInput(31, 0));
  ASSERT_OK(drafter_aux_context.drafter_mask.RunDrafter(
      drafter_aux_context.mask_signature));
  EXPECT_EQ(*static_cast<const float*>(HostAddressOf(
                drafter_context.mtp_input_buffers.at("mask_local"))),
            31.0f);
}

TEST_F(DrafterContextTest,
       DrafterAuxContextToleratesOmittedInputTokensInMaskSignature) {
  ::litert::Model drafter_model =
      LoadModel(BuildDrafterModelBuffer({{"mtp_drafter", 640}}));
  ::litert::Model aux_model = LoadModel(
      BuildMtpAuxModelBuffer({{"mask", 640}}, /*with_input_tokens=*/false));
  std::vector<NpuContextGroup> groups;
  groups.push_back(MakeContextGroup(640));

  ASSERT_OK_AND_ASSIGN(DrafterContext drafter_context,
                       DrafterContext::Create(*env_, drafter_model, groups));
  ASSERT_OK_AND_ASSIGN(
      DrafterAuxContext drafter_aux_context,
      DrafterAuxContext::Create(*env_, aux_model, drafter_context, groups,
                                MaskUpdateMethod::kModel));
  EXPECT_FALSE(drafter_aux_context.drafter_mask.Context()
                   .decode_input_buffers.contains("input_tokens"));
  ASSERT_OK(drafter_aux_context.drafter_mask.SetDecodeInput(11, 99));
  ASSERT_OK(drafter_aux_context.drafter_mask.RunDrafter(
      drafter_aux_context.mask_signature));
  EXPECT_EQ(*static_cast<const float*>(HostAddressOf(
                drafter_context.mtp_input_buffers.at("mask_local"))),
            11.0f);
}

TEST_F(DrafterContextTest,
       DrafterAuxContextRejectsMismatchedRopeOrMaskOutputs) {
  ::litert::Model drafter_model = LoadModel(BuildDrafterModelBuffer(
      {{"mtp_drafter_cache_640", 640}, {"mtp_drafter_cache_1024", 1024}}));
  std::vector<NpuContextGroup> groups;
  groups.push_back(MakeContextGroup(640));
  groups.push_back(MakeContextGroup(1024));
  ASSERT_OK_AND_ASSIGN(DrafterContext drafter_context,
                       DrafterContext::Create(*env_, drafter_model, groups));

  // Missing RoPE output in the drafter inputs.
  ::litert::Model bad_rope_aux = LoadModel(BuildMtpAuxModelBuffer(
      {{"mask_cache_640", 640}, {"mask_cache_1024", 1024}},
      /*with_input_tokens=*/true, /*rope_output_name=*/"unknown_rope_out"));
  EXPECT_THAT(DrafterAuxContext::Create(*env_, bad_rope_aux, drafter_context,
                                        groups, MaskUpdateMethod::kModel),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("has no input \"unknown_rope_out\"")));

  // Missing mask output in the drafter inputs.
  ::litert::Model bad_mask_name_aux = LoadModel(BuildMtpAuxModelBuffer(
      {{"mask_cache_640", 640}, {"mask_cache_1024", 1024}},
      /*with_input_tokens=*/true, /*rope_output_name=*/"activations",
      /*mask_output_name=*/"unknown_mask_out"));
  EXPECT_THAT(
      DrafterAuxContext::Create(*env_, bad_mask_name_aux, drafter_context,
                                groups, MaskUpdateMethod::kModel),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("has no input \"unknown_mask_out\"")));

  // Single-context mask signature whose output shape only matches 640 fails
  // validation against the 1024 context group.
  ::litert::Model bad_mask_shape_aux =
      LoadModel(BuildMtpAuxModelBuffer({{"mask", 640}}));
  EXPECT_THAT(
      DrafterAuxContext::Create(*env_, bad_mask_shape_aux, drafter_context,
                                groups, MaskUpdateMethod::kModel),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("produces \"mask_local\" with shape [1, 1, 1, 640] "
                         "but MTP drafter signature \"mtp_drafter_cache_1024\" "
                         "expects shape [1, 1, 1, 1024]")));
}

TEST_F(ExecutorUtilsTest, ResolveGraphLayout_Gemma4_12bAttentionMaskTypes) {
  // Mirrors the attention_mask_settings of the Gemma4 12B executor metadata:
  // causal global layers and vision-bidirectional local (ring buffer) layers.
  absl::flat_hash_map<absl::string_view, TensorBuffer> buffers;
  proto::ExecutorMetadata metadata;
  auto* llm_meta = metadata.mutable_llm_executor_metadata();
  auto* sb = llm_meta->add_state_buffers();
  sb->set_type(proto::StateBuffer::TYPE_LOCAL_KEY_CACHE);
  sb->set_minimum_sequence_length(1024);
  sb->set_maximum_sequence_length(1024);
  auto* mask_settings = llm_meta->mutable_attention_mask_settings();
  mask_settings->set_attention_mask_type(proto::ATTENTION_MASK_TYPE_CAUSAL);
  mask_settings->set_local_attention_mask_type(
      proto::ATTENTION_MASK_TYPE_VISION_BIDIRECTIONAL);
  mask_settings->set_sliding_window_size(1024);

  NpuGraphLayout layout =
      NpuGraphLayout::Resolve(1024, 16383, buffers, &metadata);
  EXPECT_TRUE(layout.uses_ringbuffer);
  EXPECT_EQ(layout.sliding_window_size, 1024);
  EXPECT_EQ(layout.global_mask_type, proto::ATTENTION_MASK_TYPE_CAUSAL);
  EXPECT_EQ(layout.local_mask_type,
            proto::ATTENTION_MASK_TYPE_VISION_BIDIRECTIONAL);
}

TEST_F(ExecutorUtilsTest,
       ResolveGraphLayout_LocalMaskTypeDefaultsToGlobalMaskType) {
  absl::flat_hash_map<absl::string_view, TensorBuffer> buffers;
  proto::ExecutorMetadata metadata;
  metadata.mutable_llm_executor_metadata()
      ->mutable_attention_mask_settings()
      ->set_attention_mask_type(proto::ATTENTION_MASK_TYPE_BIDIRECTIONAL);

  NpuGraphLayout layout =
      NpuGraphLayout::Resolve(128, 1024, buffers, &metadata);
  EXPECT_EQ(layout.global_mask_type, proto::ATTENTION_MASK_TYPE_BIDIRECTIONAL);
  EXPECT_EQ(layout.local_mask_type, proto::ATTENTION_MASK_TYPE_BIDIRECTIONAL);
}

TEST_F(ExecutorUtilsTest, ResolveGraphLayout_MaskTypesDefaultToCausal) {
  absl::flat_hash_map<absl::string_view, TensorBuffer> buffers;
  NpuGraphLayout layout = NpuGraphLayout::Resolve(128, 1024, buffers, nullptr);
  EXPECT_EQ(layout.global_mask_type, proto::ATTENTION_MASK_TYPE_CAUSAL);
  EXPECT_EQ(layout.local_mask_type, proto::ATTENTION_MASK_TYPE_CAUSAL);
}

void SetFirstFloat(TensorBuffer& buffer, float value) {
  auto lock =
      TensorBufferScopedLock::Create(buffer, TensorBuffer::LockMode::kWrite);
  ABSL_CHECK(lock.HasValue());
  static_cast<float*>(lock->second)[0] = value;
}

float GetFirstFloat(const TensorBuffer& buffer) {
  return *static_cast<const float*>(HostAddressOf(buffer));
}

}  // namespace

// Gives the tests access to the private warm-up and buffer setup functions of
// LlmLiteRtNpuCompiledModelExecutor.
class LlmLiteRtNpuWarmupTest : public DrafterContextTest {
 protected:
  static absl::Status WarmupInference(
      ::litert::CompiledModel& text_decoder_compiled_model,
      std::vector<NpuContextGroup>& context_groups,
      ::litert::CompiledModel& compiled_model_auxiliary,
      const InferenceContext& rope_inference_context = {},
      const InferenceContext& mask_inference_context = {},
      const InferenceContext& cache_update_inference_context = {},
      MaskUpdateMethod mask_update_method = MaskUpdateMethod::kModel,
      KVCacheUpdateMethod cache_update_method = KVCacheUpdateMethod::kModel) {
    return LlmLiteRtNpuCompiledModelExecutor::WarmupInference(
        text_decoder_compiled_model, context_groups, compiled_model_auxiliary,
        rope_inference_context, mask_inference_context,
        cache_update_inference_context, mask_update_method,
        cache_update_method);
  }

  static absl::Status WarmupDrafterInference(
      DrafterContext& drafter_context, DrafterAuxContext& drafter_aux_context,
      absl::Span<const NpuContextGroup> context_groups) {
    return LlmLiteRtNpuCompiledModelExecutor::WarmupDrafterInference(
        drafter_context, drafter_aux_context, context_groups);
  }

  static absl::StatusOr<int> AllocateTextDecoderBuffers(
      ::litert::Environment& env, const ::litert::Model* text_decoder_model,
      ::litert::CompiledModel& text_decoder_compiled_model,
      const ResolvedPrefillSignatures& prefill_signatures,
      absl::string_view decode_signature, absl::string_view verify_signature,
      absl::flat_hash_map<absl::string_view, TensorBuffer>&
          text_decoder_prefill_input_buffers,
      absl::flat_hash_map<absl::string_view, TensorBuffer>&
          text_decoder_decode_input_buffers,
      absl::flat_hash_map<absl::string_view, TensorBuffer>&
          text_decoder_verify_input_buffers,
      absl::flat_hash_map<absl::string_view, TensorBuffer>&
          input_kv_cache_buffers,
      absl::flat_hash_map<absl::string_view, TensorBuffer>&
          prefill_output_kv_cache_slice_buffers,
      absl::flat_hash_map<absl::string_view, TensorBuffer>&
          decode_output_kv_cache_slice_buffers,
      absl::flat_hash_map<absl::string_view, TensorBuffer>&
          verify_output_kv_cache_slice_buffers,
      absl::flat_hash_map<absl::string_view, HWQuantParams>& kv_quant_params,
      int64_t kv_cache_init_value) {
    return LlmLiteRtNpuCompiledModelExecutor::AllocateTextDecoderBuffers(
        env, text_decoder_model, text_decoder_compiled_model,
        prefill_signatures, decode_signature, verify_signature,
        text_decoder_prefill_input_buffers, text_decoder_decode_input_buffers,
        text_decoder_verify_input_buffers, input_kv_cache_buffers,
        prefill_output_kv_cache_slice_buffers,
        decode_output_kv_cache_slice_buffers,
        verify_output_kv_cache_slice_buffers, kv_quant_params,
        kv_cache_init_value);
  }

  static absl::StatusOr<InferenceContext> CreateTextDecoderInferenceContext(
      ::litert::Environment& env,
      ::litert::CompiledModel& text_decoder_compiled_model,
      const ResolvedPrefillSignatures& prefill_signatures,
      absl::string_view decode_signature, absl::string_view verify_signature,
      absl::flat_hash_map<absl::string_view, TensorBuffer>&
          input_kv_cache_buffers,
      absl::flat_hash_map<absl::string_view, TensorBuffer>&
          prefill_output_kv_cache_slice_buffers,
      absl::flat_hash_map<absl::string_view, TensorBuffer>&
          decode_output_kv_cache_slice_buffers,
      absl::flat_hash_map<absl::string_view, TensorBuffer>&
          verify_output_kv_cache_slice_buffers,
      absl::flat_hash_map<absl::string_view, TensorBuffer>&
          text_decoder_prefill_input_buffers,
      absl::flat_hash_map<absl::string_view, TensorBuffer>&
          text_decoder_decode_input_buffers,
      absl::flat_hash_map<absl::string_view, TensorBuffer>&
          text_decoder_verify_input_buffers) {
    return LlmLiteRtNpuCompiledModelExecutor::CreateTextDecoderInferenceContext(
        env, text_decoder_compiled_model, prefill_signatures, decode_signature,
        verify_signature, input_kv_cache_buffers,
        prefill_output_kv_cache_slice_buffers,
        decode_output_kv_cache_slice_buffers,
        verify_output_kv_cache_slice_buffers,
        text_decoder_prefill_input_buffers, text_decoder_decode_input_buffers,
        text_decoder_verify_input_buffers);
  }

  static absl::Status ApplyLegacyKvCacheWorkarounds(
      const ::litert::CompiledModel& text_decoder_compiled_model,
      absl::string_view prefill_signature,
      const proto::LlmMetadata* llm_metadata,
      absl::Span<const int> sorted_supported_context_sizes,
      bool has_per_layer_embeddings, int64_t kv_cache_init_value,
      bool enable_npu_debug_logging,
      InferenceContext& text_decoder_inference_context) {
    return LlmLiteRtNpuCompiledModelExecutor::ApplyLegacyKvCacheWorkarounds(
        text_decoder_compiled_model, prefill_signature, llm_metadata,
        sorted_supported_context_sizes, has_per_layer_embeddings,
        kv_cache_init_value, enable_npu_debug_logging,
        text_decoder_inference_context);
  }
};

namespace {

TEST_F(LlmLiteRtNpuWarmupTest, WarmupInferenceOnlyRunsFirstContextGroup) {
  // The text decoder only has the signatures of the 640 context group: running
  // any of the 1024 context group's signatures would fail.
  const std::vector<uint8_t> decoder_buffer = BuildDrafterModelBuffer(
      {{"prefill_640", 640}, {"decode_640", 640}});
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto decoder, ::litert::CompiledModel::Create(
                        *env_,
                        ::litert::BufferRef<uint8_t>(decoder_buffer.data(),
                                                     decoder_buffer.size()),
                        ::litert::HwAccelerators::kCpu));
  const std::vector<uint8_t> aux_buffer = BuildMtpAuxModelBuffer({});
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto aux, ::litert::CompiledModel::Create(
                    *env_,
                    ::litert::BufferRef<uint8_t>(aux_buffer.data(),
                                                 aux_buffer.size()),
                    ::litert::HwAccelerators::kCpu));

  std::vector<NpuContextGroup> groups;
  groups.push_back(MakeContextGroup(640));
  groups.push_back(MakeContextGroup(1024));
  groups[0].prefill_signatures.prefill = "prefill_640";
  groups[0].decode_signature = "decode_640";
  groups[1].prefill_signatures.prefill = "prefill_1024";
  groups[1].decode_signature = "decode_1024";
  // The auxiliary model has none of the groups' auxiliary signatures, so their
  // warm-up is skipped.
  for (NpuContextGroup& group : groups) {
    const std::string suffix = absl::StrCat("_", group.context_size);
    group.prefill_signatures.rope = absl::StrCat("prefill_rope", suffix);
    group.prefill_signatures.mask = absl::StrCat("prefill_mask", suffix);
    group.prefill_signatures.cache_update =
        absl::StrCat("prefill_cache_update", suffix);
    group.decode_aux_signatures.rope = absl::StrCat("decode_rope", suffix);
    group.decode_aux_signatures.mask = absl::StrCat("decode_mask", suffix);
    group.decode_aux_signatures.cache_update =
        absl::StrCat("decode_cache_update", suffix);
    group.verify_signature = absl::StrCat("verify", suffix);
  }
  ASSERT_FALSE(decoder.FindSignature(groups[1].prefill_signatures.prefill));
  ASSERT_FALSE(decoder.FindSignature(groups[1].decode_signature));

  InferenceContext& context = groups[0].text_decoder_inference_context;
  for (absl::string_view name : {"activations", "input_pos", "mask_local"}) {
    LITERT_ASSERT_OK_AND_ASSIGN(context.prefill_input_buffers[name],
                                decoder.CreateInputBuffer("prefill_640", name));
    LITERT_ASSERT_OK_AND_ASSIGN(context.decode_input_buffers[name],
                                decoder.CreateInputBuffer("decode_640", name));
  }
  for (absl::string_view name : {"kv_cache_k_0", "kv_cache_v_0"}) {
    LITERT_ASSERT_OK_AND_ASSIGN(
        context.prefill_input_buffers[name],
        groups[0].input_kv_cache_buffers.at(name).Duplicate());
    LITERT_ASSERT_OK_AND_ASSIGN(
        context.decode_input_buffers[name],
        groups[0].input_kv_cache_buffers.at(name).Duplicate());
  }
  LITERT_ASSERT_OK_AND_ASSIGN(context.prefill_output_buffers["logits"],
                              decoder.CreateOutputBuffer("prefill_640",
                                                         "logits"));
  LITERT_ASSERT_OK_AND_ASSIGN(context.decode_output_buffers["logits"],
                              decoder.CreateOutputBuffer("decode_640",
                                                         "logits"));
  SetFirstFloat(context.prefill_input_buffers.at("activations"), -2.0f);
  SetFirstFloat(context.decode_input_buffers.at("activations"), -3.0f);

  ASSERT_OK(WarmupInference(decoder, groups, aux));

  // The 640 context group's prefill and decode signatures ran.
  EXPECT_EQ(GetFirstFloat(context.prefill_output_buffers.at("logits")), 2.0f);
  EXPECT_EQ(GetFirstFloat(context.decode_output_buffers.at("logits")), 3.0f);
}

TEST_F(LlmLiteRtNpuWarmupTest, WarmupDrafterInferenceOnlyRunsFirstContextGroup) {
  ::litert::Model drafter_model = LoadModel(BuildDrafterModelBuffer(
      {{"mtp_drafter_cache_640", 640}, {"mtp_drafter_cache_1024", 1024}}));
  ::litert::Model aux_model = LoadModel(BuildMtpAuxModelBuffer(
      {{"mask_cache_640", 640}, {"mask_cache_1024", 1024}}));
  std::vector<NpuContextGroup> groups;
  groups.push_back(MakeContextGroup(640));
  groups.push_back(MakeContextGroup(1024));
  ASSERT_OK_AND_ASSIGN(DrafterContext drafter_context,
                       DrafterContext::Create(*env_, drafter_model, groups));
  ASSERT_OK_AND_ASSIGN(
      DrafterAuxContext drafter_aux_context,
      DrafterAuxContext::Create(*env_, aux_model, drafter_context, groups,
                                MaskUpdateMethod::kModel));

  // The mask signatures write the time step into the drafter's mask input of
  // their context group.
  SetFirstFloat(
      drafter_context.context_bindings[0].input_buffers.at("mask_local"), -1);
  SetFirstFloat(
      drafter_context.context_bindings[1].input_buffers.at("mask_local"), -1);
  ASSERT_OK(drafter_aux_context.drafter_mask.SetDecodeInput(7, 0));

  ASSERT_OK(WarmupDrafterInference(drafter_context, drafter_aux_context,
                                   groups));

  // Only the 640 context group's mask ran, and the drafter stays bound to it.
  EXPECT_EQ(GetFirstFloat(drafter_context.context_bindings[0].input_buffers.at(
                "mask_local")),
            7.0f);
  EXPECT_EQ(GetFirstFloat(drafter_context.context_bindings[1].input_buffers.at(
                "mask_local")),
            -1.0f);
  EXPECT_EQ(drafter_context.signature, "mtp_drafter_cache_640");
  EXPECT_EQ(drafter_aux_context.mask_signature, "mask_cache_640");
}

TEST(NeedsVisionAlignedPrefillChunksTest, OnlyForVisionBidirectionalMasks) {
  NpuGraphLayout layout;
  EXPECT_FALSE(NeedsVisionAlignedPrefillChunks(layout));

  layout.local_mask_type = proto::ATTENTION_MASK_TYPE_VISION_BIDIRECTIONAL;
  EXPECT_TRUE(NeedsVisionAlignedPrefillChunks(layout));

  layout.local_mask_type = proto::ATTENTION_MASK_TYPE_CAUSAL;
  layout.global_mask_type = proto::ATTENTION_MASK_TYPE_VISION_BIDIRECTIONAL;
  EXPECT_TRUE(NeedsVisionAlignedPrefillChunks(layout));

  layout.global_mask_type = proto::ATTENTION_MASK_TYPE_BIDIRECTIONAL;
  EXPECT_FALSE(NeedsVisionAlignedPrefillChunks(layout));
}

constexpr int kTextToken = 7;
constexpr int kVisionToken = ExecutorVisionData::kSpecialToken;
// A pending input token from the previous prefill or decode step.
constexpr std::optional<int> kPendingText = kTextToken;
constexpr std::optional<int> kNoPending = std::nullopt;

// Builds token ids from (token, count) segments.
std::vector<int> MakeIds(const std::vector<std::pair<int, int>>& segments) {
  std::vector<int> ids;
  for (const auto& [token, count] : segments) {
    ids.insert(ids.end(), count, token);
  }
  return ids;
}

std::vector<int> GroupSizes(
    const std::vector<std::pair<std::string, int>>& work_groups) {
  std::vector<int> sizes;
  for (const auto& [signature, size] : work_groups) {
    sizes.push_back(size);
  }
  return sizes;
}

// Returns, for each work group, the indices into the ids of the tokens its
// prefill chunk processes, as `PrefillInternal` does: the pending input token
// (index -1 before the first work group) followed by all but the last id of
// the work group, which becomes the next pending input token.
std::vector<std::vector<int>> ProcessedIdIndices(
    const std::vector<std::pair<std::string, int>>& work_groups,
    std::optional<int> pending_token_id) {
  std::vector<std::vector<int>> chunks;
  int next_id = 0;
  std::optional<int> pending_index =
      pending_token_id.has_value() ? std::optional<int>(-1) : std::nullopt;
  for (const auto& [signature, size] : work_groups) {
    std::vector<int> chunk;
    if (pending_index.has_value()) {
      chunk.push_back(*pending_index);
    }
    for (int i = next_id; i < next_id + size - 1; ++i) {
      chunk.push_back(i);
    }
    pending_index = next_id + size - 1;
    next_id += size;
    chunks.push_back(std::move(chunk));
  }
  return chunks;
}

// Returns the index of the chunk that processes each vision token of `ids`
// (or -1 if no chunk processes it).
std::vector<int> ChunkOfEachVisionToken(
    const std::vector<int>& ids,
    const std::vector<std::vector<int>>& processed_id_indices) {
  std::vector<int> chunk_of_id(ids.size(), -1);
  for (int c = 0; c < processed_id_indices.size(); ++c) {
    for (const int index : processed_id_indices[c]) {
      if (index >= 0) {
        chunk_of_id[index] = c;
      }
    }
  }
  std::vector<int> result;
  for (int i = 0; i < ids.size(); ++i) {
    if (ids[i] == kVisionToken) {
      result.push_back(chunk_of_id[i]);
    }
  }
  return result;
}

const SortedPrefillSignatureMap& Prefill128() {
  static const auto* const kMap =
      new SortedPrefillSignatureMap({{128, "prefill_128"}});
  return *kMap;
}

TEST(PlanVisionAlignedPrefillWorkGroupsTest, TextOnlyUsesFullChunks) {
  const std::vector<int> ids = MakeIds({{kTextToken, 300}});
  for (const std::optional<int> pending_token_id : {kPendingText, kNoPending}) {
    ASSERT_OK_AND_ASSIGN(auto work_groups,
                         PlanVisionAlignedPrefillWorkGroups(Prefill128(), ids,
                                                            pending_token_id));
    EXPECT_THAT(GroupSizes(work_groups), ElementsAre(128, 128, 44));
    for (const auto& [signature, size] : work_groups) {
      EXPECT_EQ(signature, "prefill_128");
    }
  }
}

TEST(PlanVisionAlignedPrefillWorkGroupsTest,
     ImageAcrossChunkBoundaryStartsNewChunk) {
  // 100 text tokens, a 70-token image and 50 text tokens: with full 128-token
  // chunks, the image would be split after 27 or 28 of its tokens.
  const std::vector<int> ids =
      MakeIds({{kTextToken, 100}, {kVisionToken, 70}, {kTextToken, 50}});

  // With a pending token, the first chunk processes it and ids 0 to 99.
  ASSERT_OK_AND_ASSIGN(auto with_pending, PlanVisionAlignedPrefillWorkGroups(
                                              Prefill128(), ids, kPendingText));
  EXPECT_THAT(GroupSizes(with_pending), ElementsAre(101, 119));
  EXPECT_THAT(ChunkOfEachVisionToken(
                  ids, ProcessedIdIndices(with_pending, kPendingText)),
              Each(1));

  // Without a pending token, the first chunk processes ids 0 to 99 and leaves
  // id 100 (the first image token) pending for the second chunk.
  ASSERT_OK_AND_ASSIGN(
      auto without_pending,
      PlanVisionAlignedPrefillWorkGroups(Prefill128(), ids, kNoPending));
  EXPECT_THAT(GroupSizes(without_pending), ElementsAre(101, 119));
  EXPECT_THAT(ChunkOfEachVisionToken(
                  ids, ProcessedIdIndices(without_pending, kNoPending)),
              Each(1));
}

TEST(PlanVisionAlignedPrefillWorkGroupsTest, ImageEndingAtChunkEndIsUnchanged) {
  // With a pending token, the first chunk processes it and ids 0 to 126: the
  // image (ids 57 to 126) ends exactly with the chunk.
  const std::vector<int> ids =
      MakeIds({{kTextToken, 57}, {kVisionToken, 70}, {kTextToken, 50}});
  ASSERT_OK_AND_ASSIGN(auto work_groups, PlanVisionAlignedPrefillWorkGroups(
                                             Prefill128(), ids, kPendingText));
  EXPECT_THAT(GroupSizes(work_groups), ElementsAre(128, 49));
  EXPECT_THAT(ChunkOfEachVisionToken(
                  ids, ProcessedIdIndices(work_groups, kPendingText)),
              Each(0));
}

TEST(PlanVisionAlignedPrefillWorkGroupsTest, ImageLongerThanChunkIsSplit) {
  const std::vector<int> ids =
      MakeIds({{kTextToken, 20}, {kVisionToken, 280}, {kTextToken, 20}});
  ASSERT_OK_AND_ASSIGN(auto work_groups, PlanVisionAlignedPrefillWorkGroups(
                                             Prefill128(), ids, kPendingText));
  // The same chunks as without the image.
  EXPECT_THAT(GroupSizes(work_groups), ElementsAre(128, 128, 64));
}

TEST(PlanVisionAlignedPrefillWorkGroupsTest, AlignsEachImage) {
  const std::vector<int> ids = MakeIds({{kTextToken, 100},
                                        {kVisionToken, 70},
                                        {kTextToken, 20},
                                        {kVisionToken, 70},
                                        {kTextToken, 10}});
  ASSERT_OK_AND_ASSIGN(auto work_groups, PlanVisionAlignedPrefillWorkGroups(
                                             Prefill128(), ids, kPendingText));
  // Chunk 0: pending + ids [0, 100). Chunk 1: ids [100, 190) (the first image
  // and the text after it); the second image (ids [190, 260)) would not fit,
  // so it starts chunk 2.
  EXPECT_THAT(GroupSizes(work_groups), ElementsAre(101, 90, 79));
  EXPECT_THAT(ChunkOfEachVisionToken(
                  ids, ProcessedIdIndices(work_groups, kPendingText)),
              ElementsAreArray(MakeIds({{1, 70}, {2, 70}})));
}

TEST(PlanVisionAlignedPrefillWorkGroupsTest,
     PendingVisionTokenBelongsToFollowingImage) {
  // The previous step left the first token of an image pending, e.g. because
  // the image was longer than the remaining space of a split input. It is
  // processed first, so the image is 1 + 69 tokens long and starts the first
  // chunk: it cannot move and fits.
  const std::vector<int> ids = MakeIds({{kVisionToken, 69}, {kTextToken, 100}});
  ASSERT_OK_AND_ASSIGN(auto work_groups, PlanVisionAlignedPrefillWorkGroups(
                                             Prefill128(), ids, kVisionToken));
  EXPECT_THAT(GroupSizes(work_groups), ElementsAre(128, 41));
  EXPECT_THAT(ChunkOfEachVisionToken(
                  ids, ProcessedIdIndices(work_groups, kVisionToken)),
              Each(0));
}

TEST(PlanVisionAlignedPrefillWorkGroupsTest, UsesSmallestFittingSignature) {
  const SortedPrefillSignatureMap prefill_signature_map = {
      {1024, "prefill_1024"}, {128, "prefill_128"}};
  const std::vector<int> ids =
      MakeIds({{kTextToken, 1000}, {kVisionToken, 280}, {kTextToken, 20}});
  ASSERT_OK_AND_ASSIGN(auto work_groups,
                       PlanVisionAlignedPrefillWorkGroups(prefill_signature_map,
                                                          ids, kPendingText));
  EXPECT_THAT(work_groups,
              ElementsAre(std::pair<std::string, int>("prefill_1024", 1001),
                          std::pair<std::string, int>("prefill_1024", 299)));
  EXPECT_THAT(ChunkOfEachVisionToken(
                  ids, ProcessedIdIndices(work_groups, kPendingText)),
              Each(1));

  const std::vector<int> short_ids =
      MakeIds({{kTextToken, 20}, {kVisionToken, 70}, {kTextToken, 20}});
  ASSERT_OK_AND_ASSIGN(auto short_work_groups,
                       PlanVisionAlignedPrefillWorkGroups(
                           prefill_signature_map, short_ids, kPendingText));
  EXPECT_THAT(short_work_groups,
              ElementsAre(std::pair<std::string, int>("prefill_128", 110)));
}

TEST(PlanVisionAlignedPrefillWorkGroupsTest, SingleId) {
  ASSERT_OK_AND_ASSIGN(auto with_pending,
                       PlanVisionAlignedPrefillWorkGroups(
                           Prefill128(), {kTextToken}, kPendingText));
  EXPECT_THAT(GroupSizes(with_pending), ElementsAre(1));
  ASSERT_OK_AND_ASSIGN(auto without_pending,
                       PlanVisionAlignedPrefillWorkGroups(
                           Prefill128(), {kTextToken}, kNoPending));
  EXPECT_THAT(GroupSizes(without_pending), ElementsAre(1));
}

TEST(PlanVisionAlignedPrefillWorkGroupsTest, RejectsInvalidInputs) {
  EXPECT_THAT(
      PlanVisionAlignedPrefillWorkGroups(Prefill128(), {}, kPendingText),
      StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(PlanVisionAlignedPrefillWorkGroups(SortedPrefillSignatureMap(),
                                                 {kTextToken}, kPendingText),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

// Random prompts with images that fit in a chunk: every image is processed by
// a single chunk, every chunk fits its signature, and the work groups cover
// all ids.
TEST(PlanVisionAlignedPrefillWorkGroupsTest,
     RandomPromptsKeepImagesInOneChunk) {
  std::mt19937 rng(42);
  std::uniform_int_distribution<int> text_length(1, 300);
  std::uniform_int_distribution<int> image_length(1, 128);
  std::uniform_int_distribution<int> num_images(0, 4);
  for (int trial = 0; trial < 500; ++trial) {
    std::vector<std::pair<int, int>> segments = {
        {kTextToken, text_length(rng)}};
    std::vector<int> image_lengths;
    for (int i = num_images(rng); i > 0; --i) {
      image_lengths.push_back(image_length(rng));
      segments.push_back({kVisionToken, image_lengths.back()});
      segments.push_back({kTextToken, text_length(rng)});
    }
    const std::vector<int> ids = MakeIds(segments);
    const std::optional<int> pending_token_id =
        trial % 2 == 0 ? kPendingText : kNoPending;
    ASSERT_OK_AND_ASSIGN(auto work_groups,
                         PlanVisionAlignedPrefillWorkGroups(Prefill128(), ids,
                                                            pending_token_id));

    int total = 0;
    for (const auto& [signature, size] : work_groups) {
      EXPECT_GT(size, 0);
      EXPECT_LE(size, 128);
      total += size;
    }
    EXPECT_EQ(total, ids.size());
    const auto processed = ProcessedIdIndices(work_groups, pending_token_id);
    for (const auto& chunk : processed) {
      EXPECT_LE(chunk.size(), 128);
    }
    const std::vector<int> chunks = ChunkOfEachVisionToken(ids, processed);
    int offset = 0;
    for (const int length : image_lengths) {
      for (int i = 1; i < length; ++i) {
        EXPECT_EQ(chunks[offset + i], chunks[offset])
            << "trial " << trial << ": image split across chunks";
      }
      offset += length;
    }
  }
}

TEST(GetNpuPrefillWorkGroupsTest, AlignsOnlyForVisionMasksAndImages) {
  const std::vector<int> image_ids =
      MakeIds({{kTextToken, 100}, {kVisionToken, 70}, {kTextToken, 50}});
  const std::vector<int> text_ids = MakeIds({{kTextToken, 220}});
  NpuGraphLayout causal_layout;
  NpuGraphLayout vision_layout;
  vision_layout.local_mask_type =
      proto::ATTENTION_MASK_TYPE_VISION_BIDIRECTIONAL;

  // Causal masks: the image may be split, as the chunking does not matter.
  ASSERT_OK_AND_ASSIGN(auto causal,
                       GetNpuPrefillWorkGroups(Prefill128(), causal_layout,
                                               image_ids, kPendingText));
  EXPECT_THAT(GroupSizes(causal), ElementsAre(128, 92));

  // Vision-bidirectional masks: the image starts a new chunk.
  ASSERT_OK_AND_ASSIGN(auto vision,
                       GetNpuPrefillWorkGroups(Prefill128(), vision_layout,
                                               image_ids, kPendingText));
  EXPECT_THAT(GroupSizes(vision), ElementsAre(101, 119));

  // Vision-bidirectional masks without images: the regular chunking.
  ASSERT_OK_AND_ASSIGN(
      auto text, GetNpuPrefillWorkGroups(Prefill128(), vision_layout, text_ids,
                                         kPendingText));
  EXPECT_THAT(GroupSizes(text), ElementsAre(128, 92));
}

}  // namespace

// Exercises LlmLiteRtNpuCompiledModelExecutor::CloneContext and RestoreContext
// directly on context groups with decode-only KV cache buffers (e.g. Gemma 4
// E2B KV-shared layers) and Float32 prefill dummy KV buffers (b/416702118).
class LlmLiteRtNpuCloneContextTest : public DrafterContextTest {
 protected:
  std::unique_ptr<LlmLiteRtNpuCompiledModelExecutor> MakeExecutor(
      std::vector<NpuContextGroup> groups,
      std::vector<int> supported_context_sizes, int current_step,
      int max_num_tokens = 1024, int64_t kv_cache_init_value = 0,
      ResolvedPrefillSignatures prefill_signatures = {},
      NpuEmbedder embedder = NpuEmbedder()) {
    absl::StatusOr<ModelAssets> model_assets =
        ModelAssets::Create("dummy_model");
    ABSL_CHECK_OK(model_assets);
    absl::StatusOr<LlmExecutorSettings> settings =
        LlmExecutorSettings::CreateDefault(*std::move(model_assets),
                                           Backend::NPU);
    ABSL_CHECK_OK(settings);
    settings->SetMaxNumTokens(max_num_tokens);

    const std::vector<uint8_t> model_buffer = BuildMtpAuxModelBuffer({});
    ::litert::Expected<::litert::CompiledModel> decoder =
        ::litert::CompiledModel::Create(
            *env_,
            ::litert::BufferRef<uint8_t>(model_buffer.data(),
                                         model_buffer.size()),
            ::litert::HwAccelerators::kCpu);
    ABSL_CHECK(decoder.HasValue());
    ::litert::Expected<::litert::CompiledModel> aux =
        ::litert::CompiledModel::Create(
            *env_,
            ::litert::BufferRef<uint8_t>(model_buffer.data(),
                                         model_buffer.size()),
            ::litert::HwAccelerators::kCpu);
    ABSL_CHECK(aux.HasValue());

    NpuGraphLayout initial_layout =
        !groups.empty() ? groups[0].layout : NpuGraphLayout();
    std::unique_ptr<LlmLiteRtNpuCompiledModelExecutor> executor =
        absl::WrapUnique(new LlmLiteRtNpuCompiledModelExecutor(
            std::move(*settings), *env_, NpuAuxiliaryContext(std::move(*aux)),
            std::move(*decoder), std::move(groups),
            std::move(supported_context_sizes), /*prefill_signature_map=*/{},
            std::move(prefill_signatures),
            LlmLiteRtNpuCompiledModelExecutor::LogitsQuantizationParams{
                .scale = 1.0f, .zero_point = 0},
            kv_cache_init_value,
            LlmLiteRtNpuCompiledModelExecutor::SpeculativeDecodingType::kNone,
            /*drafter_context=*/std::nullopt,
            /*drafter_aux_context=*/std::nullopt, std::move(embedder),
            NpuRope(), NpuMask(), NpuKVCache(), /*text_decoder_model=*/nullptr,
            /*llm_metadata=*/nullptr, /*executor_metadata=*/nullptr,
            /*is_dynamic_model=*/false, /*dynamic_kv_cache_capacity=*/0,
            std::move(initial_layout)));
    executor->current_step_ = current_step;
    return executor;
  }

  static NpuContextGroup& MutableContextGroup(
      LlmLiteRtNpuCompiledModelExecutor& executor, int index = 0) {
    return executor.context_groups_[index];
  }

  static int ActiveContextGroupIndex(
      const LlmLiteRtNpuCompiledModelExecutor& executor) {
    return executor.active_context_group_index_;
  }

  static void SetActiveContextGroupIndex(
      LlmLiteRtNpuCompiledModelExecutor& executor, int index) {
    executor.active_context_group_index_ = index;
  }

  static ProcessedTokens& MutableProcessedTokens(
      LlmLiteRtNpuCompiledModelExecutor& executor) {
    return executor.processed_tokens_;
  }

  static absl::StatusOr<ResolvedPrefillSignatures> PreparePrefillSignatures(
      LlmLiteRtNpuCompiledModelExecutor& executor,
      absl::string_view prefill_signature, size_t num_tokens) {
    return executor.PreparePrefillSignatures(prefill_signature, num_tokens);
  }

  static absl::StatusOr<std::vector<std::vector<int>>> FinishDecodeStep(
      LlmLiteRtNpuCompiledModelExecutor& executor, int token_id,
      absl::Time start_time, bool advance_step = false,
      bool mark_previous_processed = false) {
    return executor.FinishDecodeStep(
        token_id, start_time,
        {.advance_step = advance_step,
         .mark_previous_processed = mark_previous_processed});
  }

  static void EnqueuePendingAcceptedToken(
      LlmLiteRtNpuCompiledModelExecutor& executor, int token_id) {
    executor.pending_accepted_tokens_.push_back(token_id);
  }
};

namespace {

TEST_F(LlmLiteRtNpuCloneContextTest,
       CloneAndRestoreSavesDecodeOnlyKVCacheBuffers) {
  constexpr int kSeqLen = 4;
  constexpr int kHeadDim = 2;
  constexpr int kNumElems = kSeqLen * kHeadDim;

  NpuContextGroup group;
  group.context_size = kSeqLen;
  // Layer 0 is in both prefill and decode; layer 14 is decode-only (omitted
  // from prefill_input_buffers, matching Gemma 4 E2B KV-shared layers).
  group.input_kv_cache_buffers.emplace(
      "kv_cache_k_0",
      CreateTensorBufferWithDims(std::vector<int16_t>(kNumElems, 10),
                                 ElementType::Int16,
                                 {1, 1, kSeqLen, kHeadDim}));
  group.input_kv_cache_buffers.emplace(
      "kv_cache_v_0",
      CreateTensorBufferWithDims(std::vector<int16_t>(kNumElems, 11),
                                 ElementType::Int16,
                                 {1, 1, kHeadDim, kSeqLen}));
  group.input_kv_cache_buffers.emplace(
      "kv_cache_k_14",
      CreateTensorBufferWithDims(std::vector<int16_t>(kNumElems, 14),
                                 ElementType::Int16,
                                 {1, 1, kSeqLen, kHeadDim}));
  group.input_kv_cache_buffers.emplace(
      "kv_cache_v_14",
      CreateTensorBufferWithDims(std::vector<int16_t>(kNumElems, 15),
                                 ElementType::Int16,
                                 {1, 1, kHeadDim, kSeqLen}));
  group.layout = NpuGraphLayout::Resolve(/*prefill_chunk_size=*/2, kSeqLen,
                                         group.input_kv_cache_buffers, nullptr);

  InferenceContext& ctx = group.text_decoder_inference_context;
  for (absl::string_view name : {"kv_cache_k_0", "kv_cache_v_0"}) {
    LITERT_ASSERT_OK_AND_ASSIGN(
        ctx.prefill_input_buffers[name],
        group.input_kv_cache_buffers.at(name).Duplicate());
  }
  for (absl::string_view name :
       {"kv_cache_k_0", "kv_cache_v_0", "kv_cache_k_14", "kv_cache_v_14"}) {
    LITERT_ASSERT_OK_AND_ASSIGN(
        ctx.decode_input_buffers[name],
        group.input_kv_cache_buffers.at(name).Duplicate());
  }

  std::vector<NpuContextGroup> groups;
  groups.push_back(std::move(group));
  std::unique_ptr<LlmLiteRtNpuCompiledModelExecutor> executor =
      MakeExecutor(std::move(groups), {kSeqLen}, /*current_step=*/3);

  // Clone session A's context.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<LlmContext> session_a_context,
                       executor->CloneContext());

  // Simulate session B overwriting every KV cache buffer, including the
  // decode-only layer 14 buffers.
  for (auto& [name, buf] :
       MutableContextGroup(*executor).input_kv_cache_buffers) {
    auto lock = TensorBufferScopedLock::Create<int16_t>(
        buf, TensorBuffer::LockMode::kWrite);
    ASSERT_TRUE(lock.HasValue());
    std::fill_n(lock->second, kNumElems, static_cast<int16_t>(99));
  }

  // Restore session A's context and verify all layers (including decode-only
  // kv_cache_{k,v}_14) hold session A's values rather than session B's 99s.
  ASSERT_OK(executor->RestoreContext(std::move(session_a_context)));

  auto expect_all_equal = [&](absl::string_view name, int16_t expected) {
    auto& buf = MutableContextGroup(*executor).input_kv_cache_buffers.at(name);
    auto lock = TensorBufferScopedLock::Create<int16_t>(
        buf, TensorBuffer::LockMode::kRead);
    ABSL_CHECK(lock.HasValue());
    for (int i = 0; i < kNumElems; ++i) {
      EXPECT_EQ(lock->second[i], expected) << name << "[" << i << "]";
    }
  };
  expect_all_equal("kv_cache_k_0", 10);
  expect_all_equal("kv_cache_v_0", 11);
  expect_all_equal("kv_cache_k_14", 14);
  expect_all_equal("kv_cache_v_14", 15);
}

TEST_F(LlmLiteRtNpuCloneContextTest,
       CloneAndRestoreUsesDecodeQuantizedCacheInsteadOfPrefillDummy) {
  constexpr int kSeqLen = 4;
  constexpr int kHeadDim = 2;
  constexpr int kNumElems = kSeqLen * kHeadDim;
  constexpr int16_t kGuardCanary = 0x2345;

  // Back the Int16 decode KV cache with a double-length master buffer so that
  // even if a buggy implementation attempts to copy a Float32 buffer into it,
  // the overflow lands inside `decode_master`'s guard half rather than
  // corrupting the heap.
  std::vector<int16_t> initial_values(kNumElems * 2, kGuardCanary);
  for (int i = 0; i < kNumElems; ++i) {
    initial_values[i] = static_cast<int16_t>(100 + i);
  }
  TensorBuffer decode_master = CreateTensorBufferWithDims(
      initial_values, ElementType::Int16, {1, 1, kSeqLen * 2, kHeadDim});
  RankedTensorType int16_type(
      ElementType::Int16,
      Layout(::litert::Dimensions{1, 1, kSeqLen, kHeadDim}));
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer decode_kv_buf,
      CreateAliasBuffer(*env_, decode_master, int16_type));

  NpuContextGroup group;
  group.context_size = kSeqLen;
  group.input_kv_cache_buffers.emplace("kv_cache_k_31",
                                       std::move(decode_kv_buf));
  group.layout = NpuGraphLayout::Resolve(/*prefill_chunk_size=*/2, kSeqLen,
                                         group.input_kv_cache_buffers, nullptr);

  // Mirror CreateTextDecoderInferenceContext's b/416702118 workaround:
  // prefill_input_buffers holds an unconnected zeroed Float32 dummy while
  // input_kv_cache_buffers / decode_input_buffers hold the real Int16 KV cache.
  InferenceContext& ctx = group.text_decoder_inference_context;
  ctx.prefill_input_buffers.emplace(
      "kv_cache_k_31",
      CreateTensorBufferWithDims(std::vector<float>(kNumElems, 0.0f),
                                 ElementType::Float32,
                                 {1, 1, kSeqLen, kHeadDim}));
  LITERT_ASSERT_OK_AND_ASSIGN(
      ctx.decode_input_buffers["kv_cache_k_31"],
      group.input_kv_cache_buffers.at("kv_cache_k_31").Duplicate());

  std::vector<NpuContextGroup> groups;
  groups.push_back(std::move(group));
  std::unique_ptr<LlmLiteRtNpuCompiledModelExecutor> executor =
      MakeExecutor(std::move(groups), {kSeqLen}, /*current_step=*/4);

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<LlmContext> session_a_context,
                       executor->CloneContext());

  // Overwrite the active Int16 buffer with session B values.
  {
    auto write_lock = TensorBufferScopedLock::Create<int16_t>(
        decode_master, TensorBuffer::LockMode::kWrite);
    ASSERT_TRUE(write_lock.HasValue());
    for (int i = 0; i < kNumElems; ++i) {
      write_lock->second[i] = static_cast<int16_t>(200 + i);
    }
  }

  ASSERT_OK(executor->RestoreContext(std::move(session_a_context)));

  auto read_lock = TensorBufferScopedLock::Create<int16_t>(
      decode_master, TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(read_lock.HasValue());
  for (int i = 0; i < kNumElems; ++i) {
    EXPECT_EQ(read_lock->second[i], static_cast<int16_t>(100 + i)) << i;
  }
  for (int i = kNumElems; i < kNumElems * 2; ++i) {
    EXPECT_EQ(read_lock->second[i], kGuardCanary) << "Guard clobbered at " << i;
  }
}

TEST(KvCacheTensorNamePredicatesTest, MatchesExpectedPrefixes) {
  EXPECT_TRUE(IsKvCacheTensorName("kv_cache_k_0"));
  EXPECT_TRUE(IsKvCacheTensorName("kv_cache_v_15"));
  EXPECT_TRUE(IsKvCacheTensorName("kv_cache_c_3"));
  EXPECT_FALSE(IsKvCacheTensorName("kv_slice_k_0"));
  EXPECT_FALSE(IsKvCacheTensorName("embeddings"));

  EXPECT_TRUE(IsKvCacheSliceTensorName("kv_slice_k_0"));
  EXPECT_TRUE(IsKvCacheSliceTensorName("kv_slice_v_15"));
  EXPECT_TRUE(IsKvCacheSliceTensorName("kv_slice_c_3"));
  EXPECT_FALSE(IsKvCacheSliceTensorName("kv_cache_k_0"));
  EXPECT_FALSE(IsKvCacheSliceTensorName("logits"));
}

// Builds a minimal text decoder TFLite model with `prefill_4`, `decode`, and
// `verify` signatures to exercise AllocateTextDecoderBuffers and
// CreateTextDecoderInferenceContext on CPU.
std::vector<uint8_t> BuildTextDecoderBufferTestModel(
    tflite::TensorType prefill_unconnected_kv_type) {
  flatbuffers::FlatBufferBuilder builder;
  auto add_opcode =
      tflite::CreateOperatorCode(builder, tflite::BuiltinOperator_ADD);
  auto reshape_opcode =
      tflite::CreateOperatorCode(builder, tflite::BuiltinOperator_RESHAPE);
  auto opcodes_vec = builder.CreateVector({add_opcode, reshape_opcode});

  std::vector<flatbuffers::Offset<tflite::Buffer>> buffers;
  buffers.push_back(tflite::CreateBuffer(builder));
  auto add_zero_floats = [&](size_t count) -> uint32_t {
    std::vector<uint8_t> bytes(count * sizeof(float), 0);
    buffers.push_back(
        tflite::CreateBuffer(builder, builder.CreateVector(bytes)));
    return buffers.size() - 1;
  };
  auto add_shape_buf = [&](std::vector<int32_t> shape) -> uint32_t {
    const uint8_t* raw = reinterpret_cast<const uint8_t*>(shape.data());
    std::vector<uint8_t> bytes(raw, raw + shape.size() * sizeof(int32_t));
    buffers.push_back(
        tflite::CreateBuffer(builder, builder.CreateVector(bytes)));
    return buffers.size() - 1;
  };
  auto make_tensor = [&](std::vector<int32_t> dims, tflite::TensorType type,
                         uint32_t buf_idx, absl::string_view name) {
    return tflite::CreateTensor(builder, builder.CreateVector(dims), type,
                                buf_idx,
                                builder.CreateString(name.data(), name.size()));
  };
  auto make_map = [&](absl::string_view name, uint32_t idx) {
    return tflite::CreateTensorMap(
        builder, builder.CreateString(name.data(), name.size()), idx);
  };

  std::vector<flatbuffers::Offset<tflite::SubGraph>> subgraphs;
  std::vector<flatbuffers::Offset<tflite::SignatureDef>> sig_defs;

  // Subgraph 0: "prefill_4" with shared kv_cache_k_0 (INT16) and unconnected
  // kv_cache_k_25 (`prefill_unconnected_kv_type`).
  {
    uint32_t zero_buf = add_zero_floats(16);
    uint32_t shape_slice = add_shape_buf({1, 1, 4, 4});
    auto tensors = builder.CreateVector({
        make_tensor({1, 4, 4}, tflite::TensorType_FLOAT32, 0, "embeddings"),
        make_tensor({1, 1, 8, 4}, tflite::TensorType_INT16, 0, "kv_cache_k_0"),
        make_tensor({1, 1, 8, 4}, prefill_unconnected_kv_type, 0,
                    "kv_cache_k_25"),
        make_tensor({1, 4, 4}, tflite::TensorType_FLOAT32, zero_buf, "zeros"),
        make_tensor({1, 4, 4}, tflite::TensorType_FLOAT32, 0, "unused_out"),
        make_tensor({4}, tflite::TensorType_INT32, shape_slice, "slice_shape"),
        make_tensor({1, 1, 4, 4}, tflite::TensorType_INT16, 0, "kv_slice_k_0"),
        make_tensor({1, 1, 4, 4}, tflite::TensorType_INT16, 0, "kv_in_slice"),
    });
    // Pass embeddings -> unused_out via ADD, and reshape a 16-elem INT16 input
    // to kv_slice_k_0 so all signature tensors are valid graph tensors.
    auto add_op = tflite::CreateOperator(builder, 0,
                                         builder.CreateVector<int32_t>({0, 3}),
                                         builder.CreateVector<int32_t>({4}));
    auto reshape_op = tflite::CreateOperator(
        builder, 1, builder.CreateVector<int32_t>({7, 5}),
        builder.CreateVector<int32_t>({6}));
    subgraphs.push_back(tflite::CreateSubGraph(
        builder, tensors, builder.CreateVector<int32_t>({0, 1, 2, 7}),
        builder.CreateVector<int32_t>({4, 6}),
        builder.CreateVector({add_op, reshape_op}),
        builder.CreateString("prefill_4")));
    sig_defs.push_back(tflite::CreateSignatureDef(
        builder,
        builder.CreateVector(
            {make_map("embeddings", 0), make_map("kv_cache_k_0", 1),
             make_map("kv_cache_k_25", 2), make_map("aux_in", 7)}),
        builder.CreateVector(
            {make_map("unused_out", 4), make_map("kv_slice_k_0", 6)}),
        builder.CreateString("prefill_4"), 0));
  }

  // Subgraph 1: "decode" with INT16 kv_cache_k_0 and kv_cache_k_25, plus
  // logits, activations, and kv_slice_k_0 outputs.
  {
    uint32_t zero_buf = add_zero_floats(4);
    uint32_t shape_slice = add_shape_buf({1, 1, 1, 4});
    auto tensors = builder.CreateVector({
        make_tensor({1, 1, 4}, tflite::TensorType_FLOAT32, 0, "embeddings"),
        make_tensor({1, 1, 8, 4}, tflite::TensorType_INT16, 0, "kv_cache_k_0"),
        make_tensor({1, 1, 8, 4}, tflite::TensorType_INT16, 0, "kv_cache_k_25"),
        make_tensor({1, 1, 4}, tflite::TensorType_FLOAT32, zero_buf, "zeros"),
        make_tensor({1, 1, 4}, tflite::TensorType_FLOAT32, 0, "logits"),
        make_tensor({1, 1, 4}, tflite::TensorType_FLOAT32, 0, "activations"),
        make_tensor({4}, tflite::TensorType_INT32, shape_slice, "slice_shape"),
        make_tensor({1, 1, 1, 4}, tflite::TensorType_INT16, 0, "kv_in_slice"),
        make_tensor({1, 1, 1, 4}, tflite::TensorType_INT16, 0, "kv_slice_k_0"),
    });
    auto add_logits = tflite::CreateOperator(
        builder, 0, builder.CreateVector<int32_t>({0, 3}),
        builder.CreateVector<int32_t>({4}));
    auto add_acts = tflite::CreateOperator(
        builder, 0, builder.CreateVector<int32_t>({0, 3}),
        builder.CreateVector<int32_t>({5}));
    auto reshape_op = tflite::CreateOperator(
        builder, 1, builder.CreateVector<int32_t>({7, 6}),
        builder.CreateVector<int32_t>({8}));
    subgraphs.push_back(tflite::CreateSubGraph(
        builder, tensors, builder.CreateVector<int32_t>({0, 1, 2, 7}),
        builder.CreateVector<int32_t>({4, 5, 8}),
        builder.CreateVector({add_logits, add_acts, reshape_op}),
        builder.CreateString("decode")));
    sig_defs.push_back(tflite::CreateSignatureDef(
        builder,
        builder.CreateVector(
            {make_map("embeddings", 0), make_map("kv_cache_k_0", 1),
             make_map("kv_cache_k_25", 2), make_map("aux_in", 7)}),
        builder.CreateVector({make_map("logits", 4), make_map("activations", 5),
                              make_map("kv_slice_k_0", 8)}),
        builder.CreateString("decode"), 1));
  }

  // Subgraph 2: "verify" with embeddings, kv_cache_k_0 -> logits, activations,
  // kv_slice_k_0.
  {
    uint32_t zero_buf = add_zero_floats(8);
    uint32_t shape_slice = add_shape_buf({1, 1, 2, 4});
    auto tensors = builder.CreateVector({
        make_tensor({1, 2, 4}, tflite::TensorType_FLOAT32, 0, "embeddings"),
        make_tensor({1, 1, 8, 4}, tflite::TensorType_INT16, 0, "kv_cache_k_0"),
        make_tensor({1, 2, 4}, tflite::TensorType_FLOAT32, zero_buf, "zeros"),
        make_tensor({1, 2, 4}, tflite::TensorType_FLOAT32, 0, "logits"),
        make_tensor({1, 2, 4}, tflite::TensorType_FLOAT32, 0, "activations"),
        make_tensor({4}, tflite::TensorType_INT32, shape_slice, "slice_shape"),
        make_tensor({1, 1, 2, 4}, tflite::TensorType_INT16, 0, "kv_in_slice"),
        make_tensor({1, 1, 2, 4}, tflite::TensorType_INT16, 0, "kv_slice_k_0"),
    });
    auto add_logits = tflite::CreateOperator(
        builder, 0, builder.CreateVector<int32_t>({0, 2}),
        builder.CreateVector<int32_t>({3}));
    auto add_acts = tflite::CreateOperator(
        builder, 0, builder.CreateVector<int32_t>({0, 2}),
        builder.CreateVector<int32_t>({4}));
    auto reshape_op = tflite::CreateOperator(
        builder, 1, builder.CreateVector<int32_t>({6, 5}),
        builder.CreateVector<int32_t>({7}));
    subgraphs.push_back(tflite::CreateSubGraph(
        builder, tensors, builder.CreateVector<int32_t>({0, 1, 6}),
        builder.CreateVector<int32_t>({3, 4, 7}),
        builder.CreateVector({add_logits, add_acts, reshape_op}),
        builder.CreateString("verify")));
    sig_defs.push_back(tflite::CreateSignatureDef(
        builder,
        builder.CreateVector({make_map("embeddings", 0),
                              make_map("kv_cache_k_0", 1),
                              make_map("aux_in", 6)}),
        builder.CreateVector({make_map("logits", 3), make_map("activations", 4),
                              make_map("kv_slice_k_0", 7)}),
        builder.CreateString("verify"), 2));
  }

  auto model = tflite::CreateModel(
      builder, 3, opcodes_vec, builder.CreateVector(subgraphs),
      builder.CreateString("decoder_test"), builder.CreateVector(buffers),
      /*metadata_buffer=*/0, /*metadata=*/0, builder.CreateVector(sig_defs));
  builder.Finish(model, "TFL3");
  return std::vector<uint8_t>(builder.GetBufferPointer(),
                              builder.GetBufferPointer() + builder.GetSize());
}

TEST_F(
    LlmLiteRtNpuWarmupTest,
    AllocateAndCreateDecoderInferenceContextHandlesMismatchedFloat32Prefill) {
  const std::vector<uint8_t> buf =
      BuildTextDecoderBufferTestModel(tflite::TensorType_FLOAT32);
  ::litert::Model model = LoadModel(buf);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto compiled,
      ::litert::CompiledModel::Create(
          *env_, ::litert::BufferRef<uint8_t>(buf.data(), buf.size()),
          ::litert::HwAccelerators::kCpu));

  ResolvedPrefillSignatures prefill_sigs = BuildResolvedPrefillSignatures(4);
  absl::flat_hash_map<absl::string_view, TensorBuffer> prefill_in, decode_in,
      verify_in, kv_in, prefill_slice_out, decode_slice_out, verify_slice_out;
  absl::flat_hash_map<absl::string_view, HWQuantParams> kv_quant_params;

  ASSERT_OK_AND_ASSIGN(
      int resolved_ctx,
      AllocateTextDecoderBuffers(
          *env_, &model, compiled, prefill_sigs, "decode", "verify", prefill_in,
          decode_in, verify_in, kv_in, prefill_slice_out, decode_slice_out,
          verify_slice_out, kv_quant_params, /*kv_cache_init_value=*/7));
  EXPECT_EQ(resolved_ctx, 8);
  EXPECT_TRUE(kv_in.contains("kv_cache_k_0"));
  EXPECT_TRUE(kv_in.contains("kv_cache_k_25"));
  EXPECT_TRUE(prefill_slice_out.contains("kv_slice_k_0"));
  EXPECT_TRUE(decode_slice_out.contains("kv_slice_k_0"));
  EXPECT_TRUE(verify_slice_out.contains("kv_slice_k_0"));

  // Persistent KV buffers were allocated from decode (Int16) and filled with 7.
  {
    auto lock = TensorBufferScopedLock::Create<int16_t>(
        kv_in.at("kv_cache_k_25"), TensorBuffer::LockMode::kRead);
    ASSERT_TRUE(lock.HasValue());
    EXPECT_EQ(lock->second[0], 7);
  }

  ASSERT_OK_AND_ASSIGN(InferenceContext ctx,
                       CreateTextDecoderInferenceContext(
                           *env_, compiled, prefill_sigs, "decode", "verify",
                           kv_in, prefill_slice_out, decode_slice_out,
                           verify_slice_out, prefill_in, decode_in, verify_in));

  // Shared Int16 kv_cache_k_0 aliases `kv_in`, while Float32 prefill
  // kv_cache_k_25 gets a separate cleared Float32 dummy buffer (b/416702118).
  EXPECT_TRUE(SharesMemory(ctx.prefill_input_buffers.at("kv_cache_k_0"),
                           kv_in.at("kv_cache_k_0")));
  EXPECT_FALSE(SharesMemory(ctx.prefill_input_buffers.at("kv_cache_k_25"),
                            kv_in.at("kv_cache_k_25")));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto prefill_k25_type,
      ctx.prefill_input_buffers.at("kv_cache_k_25").TensorType());
  EXPECT_EQ(prefill_k25_type.ElementType(), ElementType::Float32);

  // Decode and verify outputs include both slice buffers and
  // logits/activations.
  EXPECT_TRUE(ctx.decode_output_buffers.contains("logits"));
  EXPECT_TRUE(ctx.decode_output_buffers.contains("activations"));
  EXPECT_TRUE(ctx.verify_output_buffers.contains("logits"));
  EXPECT_TRUE(ctx.verify_output_buffers.contains("activations"));
}

TEST_F(LlmLiteRtNpuWarmupTest,
       CreateTextDecoderInferenceContextRejectsIncompatibleKvTypes) {
  // Int8 in prefill vs Int16 in decode is rejected as incompatible.
  const std::vector<uint8_t> buf =
      BuildTextDecoderBufferTestModel(tflite::TensorType_INT8);
  ::litert::Model model = LoadModel(buf);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto compiled,
      ::litert::CompiledModel::Create(
          *env_, ::litert::BufferRef<uint8_t>(buf.data(), buf.size()),
          ::litert::HwAccelerators::kCpu));

  ResolvedPrefillSignatures prefill_sigs = BuildResolvedPrefillSignatures(4);
  absl::flat_hash_map<absl::string_view, TensorBuffer> prefill_in, decode_in,
      verify_in, kv_in, prefill_slice_out, decode_slice_out, verify_slice_out;
  absl::flat_hash_map<absl::string_view, HWQuantParams> kv_quant_params;
  ASSERT_OK(AllocateTextDecoderBuffers(
      *env_, &model, compiled, prefill_sigs, "decode", "verify", prefill_in,
      decode_in, verify_in, kv_in, prefill_slice_out, decode_slice_out,
      verify_slice_out, kv_quant_params, /*kv_cache_init_value=*/0));

  EXPECT_THAT(CreateTextDecoderInferenceContext(
                  *env_, compiled, prefill_sigs, "decode", "verify", kv_in,
                  prefill_slice_out, decode_slice_out, verify_slice_out,
                  prefill_in, decode_in, verify_in),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Incompatible KV cache tensor types for "
                                 "'kv_cache_k_25'")));
}

// Builds a multi-signature auxiliary model where each signature `(sig, in_name,
// out_name, bias)` computes `out = in + bias` on a `[1, 1, 4]` Float32 tensor.
struct SimpleAuxSigSpec {
  std::string sig_name;
  std::string in_name;
  std::string out_name;
  float bias;
};

std::vector<uint8_t> BuildSimpleAuxModelBuffer(
    const std::vector<SimpleAuxSigSpec>& specs) {
  flatbuffers::FlatBufferBuilder builder;
  auto add_opcode =
      tflite::CreateOperatorCode(builder, tflite::BuiltinOperator_ADD);
  auto opcodes_vec = builder.CreateVector({add_opcode});

  std::vector<flatbuffers::Offset<tflite::Buffer>> buffers;
  buffers.push_back(tflite::CreateBuffer(builder));
  std::vector<flatbuffers::Offset<tflite::SubGraph>> subgraphs;
  std::vector<flatbuffers::Offset<tflite::SignatureDef>> sig_defs;

  for (size_t i = 0; i < specs.size(); ++i) {
    std::vector<float> bias_vals(4, specs[i].bias);
    const uint8_t* raw = reinterpret_cast<const uint8_t*>(bias_vals.data());
    buffers.push_back(tflite::CreateBuffer(
        builder, builder.CreateVector(std::vector<uint8_t>(raw, raw + 16))));
    const uint32_t bias_buf = buffers.size() - 1;

    auto t_in = tflite::CreateTensor(
        builder, builder.CreateVector<int32_t>({1, 1, 4}),
        tflite::TensorType_FLOAT32, 0, builder.CreateString(specs[i].in_name));
    auto t_bias = tflite::CreateTensor(
        builder, builder.CreateVector<int32_t>({1, 1, 4}),
        tflite::TensorType_FLOAT32, bias_buf, builder.CreateString("bias"));
    auto t_out = tflite::CreateTensor(
        builder, builder.CreateVector<int32_t>({1, 1, 4}),
        tflite::TensorType_FLOAT32, 0, builder.CreateString(specs[i].out_name));
    auto op = tflite::CreateOperator(builder, 0,
                                     builder.CreateVector<int32_t>({0, 1}),
                                     builder.CreateVector<int32_t>({2}));
    subgraphs.push_back(tflite::CreateSubGraph(
        builder, builder.CreateVector({t_in, t_bias, t_out}),
        builder.CreateVector<int32_t>({0}), builder.CreateVector<int32_t>({2}),
        builder.CreateVector({op}), builder.CreateString(specs[i].sig_name)));
    auto in_map = tflite::CreateTensorMap(
        builder, builder.CreateString(specs[i].in_name), 0);
    auto out_map = tflite::CreateTensorMap(
        builder, builder.CreateString(specs[i].out_name), 2);
    sig_defs.push_back(
        tflite::CreateSignatureDef(builder, builder.CreateVector({in_map}),
                                   builder.CreateVector({out_map}),
                                   builder.CreateString(specs[i].sig_name), i));
  }

  auto model = tflite::CreateModel(
      builder, 3, opcodes_vec, builder.CreateVector(subgraphs),
      builder.CreateString("aux_test"), builder.CreateVector(buffers), 0, 0,
      builder.CreateVector(sig_defs));
  builder.Finish(model, "TFL3");
  return std::vector<uint8_t>(builder.GetBufferPointer(),
                              builder.GetBufferPointer() + builder.GetSize());
}

TEST_F(LlmLiteRtNpuWarmupTest,
       WarmupInferenceAppliesAuxOverridesSkipsWHAndClearsMasterKVCache) {
  const std::vector<uint8_t> decoder_buf = BuildSimpleAuxModelBuffer({
      {"prefill_640", "embeddings", "logits", 0.0f},
      {"decode_640", "embeddings", "logits", 0.0f},
  });
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto decoder,
      ::litert::CompiledModel::Create(
          *env_,
          ::litert::BufferRef<uint8_t>(decoder_buf.data(), decoder_buf.size()),
          ::litert::HwAccelerators::kCpu));

  const std::vector<uint8_t> aux_buf = BuildSimpleAuxModelBuffer({
      {"prefill_rope", "rope_in", "rope_out", 10.0f},
      {"prefill_mask", "mask_in", "mask_local", 20.0f},
      {"prefill_cache_update", "kv_cache_k_0", "kv_cache_v_0", 30.0f},
      {"decode_rope", "rope_in", "rope_out", 40.0f},
      {"decode_mask", "mask_in", "mask_local", 50.0f},
      {"decode_cache_update", "kv_cache_k_0", "kv_cache_v_0", 60.0f},
      {"verify_rope", "rope_in", "rope_out", 70.0f},
      {"verify_mask", "mask_in", "mask_local", 80.0f},
      {"verify_cache_update", "cu_in", "cu_out", 90.0f},
  });
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto aux,
      ::litert::CompiledModel::Create(
          *env_, ::litert::BufferRef<uint8_t>(aux_buf.data(), aux_buf.size()),
          ::litert::HwAccelerators::kCpu));

  auto make_f32_buf = [&](float val) {
    return CreateTensorBufferWithDims(std::vector<float>(4, val),
                                      ElementType::Float32, {1, 1, 4});
  };

  NpuContextGroup group;
  group.context_size = 640;
  group.prefill_signatures.prefill = "prefill_640";
  group.prefill_signatures.rope = "prefill_rope";
  group.prefill_signatures.mask = "prefill_mask";
  group.prefill_signatures.cache_update = "prefill_cache_update";
  group.decode_signature = "decode_640";
  group.decode_aux_signatures.rope = "decode_rope";
  group.decode_aux_signatures.mask = "decode_mask";
  group.decode_aux_signatures.cache_update = "decode_cache_update";
  group.verify_signature = "verify_absent";
  group.input_kv_cache_buffers.emplace("kv_cache_k_0", make_f32_buf(5.0f));
  group.input_kv_cache_buffers.emplace("kv_cache_v_0", make_f32_buf(6.0f));

  InferenceContext& dec_ctx = group.text_decoder_inference_context;
  dec_ctx.prefill_input_buffers.emplace("embeddings", make_f32_buf(0.0f));
  dec_ctx.prefill_input_buffers.emplace("rope_out", make_f32_buf(-1.0f));
  dec_ctx.prefill_input_buffers.emplace("mask_local", make_f32_buf(-1.0f));
  LITERT_ASSERT_OK_AND_ASSIGN(
      dec_ctx.prefill_input_buffers["kv_cache_k_0"],
      group.input_kv_cache_buffers.at("kv_cache_k_0").Duplicate());
  dec_ctx.prefill_output_buffers.emplace("logits", make_f32_buf(0.0f));

  dec_ctx.decode_input_buffers.emplace("embeddings", make_f32_buf(0.0f));
  dec_ctx.decode_input_buffers.emplace("rope_out", make_f32_buf(-1.0f));
  dec_ctx.decode_input_buffers.emplace("mask_local", make_f32_buf(-1.0f));
  LITERT_ASSERT_OK_AND_ASSIGN(
      dec_ctx.decode_input_buffers["kv_cache_v_0"],
      group.input_kv_cache_buffers.at("kv_cache_v_0").Duplicate());
  dec_ctx.decode_output_buffers.emplace("logits", make_f32_buf(0.0f));

  InferenceContext rope_ctx, mask_ctx, cu_ctx;
  rope_ctx.prefill_input_buffers.emplace("rope_in", make_f32_buf(1.0f));
  rope_ctx.prefill_output_buffers.emplace("rope_out", make_f32_buf(-99.0f));
  rope_ctx.decode_input_buffers.emplace("rope_in", make_f32_buf(2.0f));
  rope_ctx.decode_output_buffers.emplace("rope_out", make_f32_buf(-99.0f));
  rope_ctx.verify_input_buffers.emplace("rope_in", make_f32_buf(3.0f));
  rope_ctx.verify_output_buffers.emplace("rope_out", make_f32_buf(-99.0f));

  mask_ctx.prefill_input_buffers.emplace("mask_in", make_f32_buf(1.0f));
  mask_ctx.prefill_output_buffers.emplace("mask_local", make_f32_buf(-99.0f));
  mask_ctx.decode_input_buffers.emplace("mask_in", make_f32_buf(2.0f));
  mask_ctx.decode_output_buffers.emplace("mask_local", make_f32_buf(-99.0f));
  mask_ctx.verify_input_buffers.emplace("mask_in", make_f32_buf(3.0f));
  mask_ctx.verify_output_buffers.emplace("mask_local", make_f32_buf(-99.0f));

  cu_ctx.prefill_input_buffers.emplace("kv_cache_k_0", make_f32_buf(-99.0f));
  cu_ctx.prefill_output_buffers.emplace("kv_cache_v_0", make_f32_buf(-99.0f));
  cu_ctx.decode_input_buffers.emplace("kv_cache_k_0", make_f32_buf(-99.0f));
  cu_ctx.decode_output_buffers.emplace("kv_cache_v_0", make_f32_buf(-99.0f));
  cu_ctx.verify_input_buffers.emplace("cu_in", make_f32_buf(4.0f));
  cu_ctx.verify_output_buffers.emplace("cu_out", make_f32_buf(-99.0f));

  std::vector<NpuContextGroup> groups;
  groups.push_back(std::move(group));

  // 1. With kWH mask and kWH cache update, only RoPE aux signatures run;
  // embeddings are filled with 1.0f before decoder warmup; KV caches are
  // cleared to 0.0f at the end.
  ASSERT_OK(WarmupInference(decoder, groups, aux, rope_ctx, mask_ctx, cu_ctx,
                            MaskUpdateMethod::kWH, KVCacheUpdateMethod::kWH));
  EXPECT_EQ(
      GetFirstFloat(
          groups[0].text_decoder_inference_context.prefill_output_buffers.at(
              "logits")),
      1.0f);
  EXPECT_EQ(
      GetFirstFloat(
          groups[0].text_decoder_inference_context.prefill_input_buffers.at(
              "rope_out")),
      11.0f);
  EXPECT_EQ(
      GetFirstFloat(
          groups[0].text_decoder_inference_context.decode_input_buffers.at(
              "rope_out")),
      42.0f);
  EXPECT_EQ(GetFirstFloat(rope_ctx.verify_output_buffers.at("rope_out")),
            73.0f);
  // Mask and cache update were skipped under kWH.
  EXPECT_EQ(
      GetFirstFloat(
          groups[0].text_decoder_inference_context.prefill_input_buffers.at(
              "mask_local")),
      -1.0f);
  EXPECT_EQ(GetFirstFloat(mask_ctx.verify_output_buffers.at("mask_local")),
            -99.0f);
  EXPECT_EQ(GetFirstFloat(cu_ctx.verify_output_buffers.at("cu_out")), -99.0f);
  // Master KV caches on `groups.back()` were cleared at the end of warmup.
  EXPECT_EQ(GetFirstFloat(groups[0].input_kv_cache_buffers.at("kv_cache_k_0")),
            0.0f);
  EXPECT_EQ(GetFirstFloat(groups[0].input_kv_cache_buffers.at("kv_cache_v_0")),
            0.0f);

  // 2. With kModel mask and kModel cache update, mask and cache_update
  // overrides also run.
  ASSERT_OK(WarmupInference(decoder, groups, aux, rope_ctx, mask_ctx, cu_ctx,
                            MaskUpdateMethod::kModel,
                            KVCacheUpdateMethod::kModel));
  EXPECT_EQ(
      GetFirstFloat(
          groups[0].text_decoder_inference_context.prefill_input_buffers.at(
              "mask_local")),
      21.0f);
  EXPECT_EQ(
      GetFirstFloat(
          groups[0].text_decoder_inference_context.decode_input_buffers.at(
              "mask_local")),
      52.0f);
  EXPECT_EQ(GetFirstFloat(mask_ctx.verify_output_buffers.at("mask_local")),
            83.0f);
  EXPECT_EQ(GetFirstFloat(cu_ctx.verify_output_buffers.at("cu_out")), 94.0f);
}

TEST_F(LlmLiteRtNpuCloneContextTest,
       PreparePrefillSignaturesValidatesBoundsAndResolvesChunkSize) {
  std::vector<NpuContextGroup> groups;
  groups.push_back(MakeContextGroup(640));
  auto executor = MakeExecutor(std::move(groups), {640}, /*current_step=*/600,
                               /*max_num_tokens=*/630);

  // Exceeds user max_num_tokens (600 + 32 > 630).
  EXPECT_THAT(PreparePrefillSignatures(*executor, "prefill_64", 32),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("exceeds max sequence length (630)")));

  // Fits max_num_tokens (600 + 16 <= 630), but padded runner_size 64 overflows
  // NPU hardware context_size (600 + 64 > 640).
  EXPECT_THAT(PreparePrefillSignatures(*executor, "prefill_64_cache_640", 16),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Padded prefill length (64) plus current step "
                                 "(600) exceeds NPU context size (640)")));

  // Fits both max_num_tokens and context_size (600 + 32 <= 640).
  ASSERT_OK_AND_ASSIGN(
      ResolvedPrefillSignatures sigs,
      PreparePrefillSignatures(*executor, "prefill_32_cache_640", 10));
  EXPECT_EQ(sigs.size, 32);
  EXPECT_EQ(sigs.context_size, 640);
  EXPECT_EQ(sigs.prefill, "prefill_32_cache_640");
}

TEST_F(LlmLiteRtNpuCloneContextTest,
       ResetClearsAllContextGroupKVCachesAndResetsActiveGroup) {
  std::vector<NpuContextGroup> groups;
  for (int ctx_size : {64, 128}) {
    NpuContextGroup g;
    g.context_size = ctx_size;
    g.input_kv_cache_buffers.emplace(
        "kv_cache_k_0",
        CreateTensorBufferWithDims(std::vector<int16_t>(16, 42),
                                   ElementType::Int16, {1, 1, 4, 4}));
    LITERT_ASSERT_OK_AND_ASSIGN(
        g.text_decoder_inference_context.decode_input_buffers["kv_cache_k_0"],
        g.input_kv_cache_buffers.at("kv_cache_k_0").Duplicate());
    LITERT_ASSERT_OK_AND_ASSIGN(
        g.text_decoder_inference_context.prefill_input_buffers["kv_cache_k_0"],
        g.input_kv_cache_buffers.at("kv_cache_k_0").Duplicate());
    groups.push_back(std::move(g));
  }

  auto executor =
      MakeExecutor(std::move(groups), {64, 128}, /*current_step=*/50,
                   /*max_num_tokens=*/128, /*kv_cache_init_value=*/-3);
  SetActiveContextGroupIndex(*executor, 1);

  ASSERT_OK(executor->Reset());
  EXPECT_EQ(ActiveContextGroupIndex(*executor), 0);
  ASSERT_OK_AND_ASSIGN(int step_after_reset, executor->GetCurrentStep());
  EXPECT_EQ(step_after_reset, 0);

  for (int idx : {0, 1}) {
    auto lock = TensorBufferScopedLock::Create<int16_t>(
        MutableContextGroup(*executor, idx)
            .input_kv_cache_buffers.at("kv_cache_k_0"),
        TensorBuffer::LockMode::kRead);
    ASSERT_TRUE(lock.HasValue());
    EXPECT_EQ(lock->second[0], -3) << "group " << idx;
  }

  // Overwrite group 1 again and verify RestoreContext with an empty context
  // (`current_step == 0`) also clears all context group KV caches.
  {
    auto write_lock = TensorBufferScopedLock::Create<int16_t>(
        MutableContextGroup(*executor, 1)
            .input_kv_cache_buffers.at("kv_cache_k_0"),
        TensorBuffer::LockMode::kWrite);
    ASSERT_TRUE(write_lock.HasValue());
    write_lock->second[0] = 99;
  }
  SetActiveContextGroupIndex(*executor, 1);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<LlmContext> empty_ctx,
                       executor->CreateNewContext(std::nullopt, {}));
  ASSERT_OK(executor->RestoreContext(std::move(empty_ctx)));
  EXPECT_EQ(ActiveContextGroupIndex(*executor), 0);
  {
    auto lock = TensorBufferScopedLock::Create<int16_t>(
        MutableContextGroup(*executor, 1)
            .input_kv_cache_buffers.at("kv_cache_k_0"),
        TensorBuffer::LockMode::kRead);
    ASSERT_TRUE(lock.HasValue());
    EXPECT_EQ(lock->second[0], -3);
  }
}

TEST_F(LlmLiteRtNpuCloneContextTest,
       FinishDecodeStepEnqueuesTokenAndOptionallyAdvancesStep) {
  std::vector<NpuContextGroup> groups;
  groups.push_back(MakeContextGroup(64));
  auto executor = MakeExecutor(std::move(groups), {64}, /*current_step=*/5);

  // Non-speculative path: keeps `current_step_` as already incremented by
  // caller and enqueues token 11 as pending.
  ASSERT_OK_AND_ASSIGN(auto out1,
                       FinishDecodeStep(*executor, /*token_id=*/11, absl::Now(),
                                        /*advance_step=*/false));
  EXPECT_THAT(out1, ElementsAre(ElementsAre(11)));
  ASSERT_OK_AND_ASSIGN(int step1, executor->GetCurrentStep());
  EXPECT_EQ(step1, 5);
  auto [first_step, first_pending] =
      MutableProcessedTokens(*executor).GetNextUnprocessedToken();
  ASSERT_EQ(first_pending.size(), 1);
  EXPECT_EQ(first_pending[0]->id(), 11);

  // Queued speculative token path: marks previous pending token as processed,
  // enqueues token 22, and advances `current_step_`.
  ASSERT_OK_AND_ASSIGN(auto out2,
                       FinishDecodeStep(*executor, /*token_id=*/22, absl::Now(),
                                        /*advance_step=*/true,
                                        /*mark_previous_processed=*/true));
  EXPECT_THAT(out2, ElementsAre(ElementsAre(22)));
  ASSERT_OK_AND_ASSIGN(int step2, executor->GetCurrentStep());
  EXPECT_EQ(step2, 6);
  auto [next_step, pending] =
      MutableProcessedTokens(*executor).GetNextUnprocessedToken();
  ASSERT_EQ(pending.size(), 1);
  EXPECT_EQ(pending[0]->id(), 22);

  // Public Decode() pops from `pending_accepted_tokens_` via
  // PopPendingAcceptedToken and advances `current_step_`.
  MutableProcessedTokens(*executor).AddProcessedTokens({0, 1, 2, 3, 4});
  EnqueuePendingAcceptedToken(*executor, 33);
  ASSERT_OK_AND_ASSIGN(auto out3, executor->Decode());
  EXPECT_THAT(out3, ElementsAre(ElementsAre(33)));
  ASSERT_OK_AND_ASSIGN(int step3, executor->GetCurrentStep());
  EXPECT_EQ(step3, 7);
}

TEST_F(LlmLiteRtNpuWarmupTest,
       ApplyLegacyKvCacheWorkaroundsRecreatesDecodeBuffersForLegacyLayers) {
  struct LayerCase {
    std::string k_name;
    std::string v_name;
  };
  std::vector<LayerCase> cases = {
      {"kv_cache_k_25", "kv_cache_v_25"},
      {"kv_cache_k_23", "kv_cache_v_23"},
  };

  for (const LayerCase& tc : cases) {
    flatbuffers::FlatBufferBuilder builder;
    auto reshape_op =
        tflite::CreateOperatorCode(builder, tflite::BuiltinOperator_RESHAPE);
    auto opcodes = builder.CreateVector({reshape_op});
    std::vector<flatbuffers::Offset<tflite::Buffer>> buffers;
    buffers.push_back(tflite::CreateBuffer(builder));
    const std::vector<int32_t> shape = {1, 1, 4, 4};
    const uint8_t* raw = reinterpret_cast<const uint8_t*>(shape.data());
    buffers.push_back(tflite::CreateBuffer(
        builder, builder.CreateVector(std::vector<uint8_t>(raw, raw + 16))));

    struct SubgraphAndSig {
      flatbuffers::Offset<tflite::SubGraph> subgraph;
      flatbuffers::Offset<tflite::SignatureDef> sig_def;
    };
    auto make_sub = [&](absl::string_view sig, tflite::TensorType kv_type,
                        uint32_t idx) {
      auto t_k =
          tflite::CreateTensor(builder, builder.CreateVector(shape), kv_type, 0,
                               builder.CreateString(tc.k_name));
      auto t_v =
          tflite::CreateTensor(builder, builder.CreateVector(shape), kv_type, 0,
                               builder.CreateString(tc.v_name));
      auto t_s = tflite::CreateTensor(
          builder, builder.CreateVector<int32_t>({4}), tflite::TensorType_INT32,
          1, builder.CreateString("s"));
      auto t_out =
          tflite::CreateTensor(builder, builder.CreateVector(shape), kv_type, 0,
                               builder.CreateString("out"));
      auto op = tflite::CreateOperator(builder, 0,
                                       builder.CreateVector<int32_t>({0, 2}),
                                       builder.CreateVector<int32_t>({3}));
      auto sg = tflite::CreateSubGraph(
          builder, builder.CreateVector({t_k, t_v, t_s, t_out}),
          builder.CreateVector<int32_t>({0, 1}),
          builder.CreateVector<int32_t>({3}), builder.CreateVector({op}),
          builder.CreateString(sig.data(), sig.size()));
      auto sd = tflite::CreateSignatureDef(
          builder,
          builder.CreateVector({
              tflite::CreateTensorMap(builder, builder.CreateString(tc.k_name),
                                      0),
              tflite::CreateTensorMap(builder, builder.CreateString(tc.v_name),
                                      1),
          }),
          builder.CreateVector({tflite::CreateTensorMap(
              builder, builder.CreateString("out"), 3)}),
          builder.CreateString(sig.data(), sig.size()), idx);
      return SubgraphAndSig{sg, sd};
    };

    auto [sg_prefill, sd_prefill] =
        make_sub("prefill_4", tflite::TensorType_FLOAT32, 0);
    auto [sg_decode, sd_decode] =
        make_sub("decode", tflite::TensorType_INT16, 1);
    auto m = tflite::CreateModel(
        builder, 3, opcodes, builder.CreateVector({sg_prefill, sg_decode}),
        builder.CreateString("legacy"), builder.CreateVector(buffers), 0, 0,
        builder.CreateVector({sd_prefill, sd_decode}));
    builder.Finish(m, "TFL3");
    std::vector<uint8_t> model_bytes(
        builder.GetBufferPointer(),
        builder.GetBufferPointer() + builder.GetSize());

    LITERT_ASSERT_OK_AND_ASSIGN(
        auto compiled, ::litert::CompiledModel::Create(
                           *env_,
                           ::litert::BufferRef<uint8_t>(model_bytes.data(),
                                                        model_bytes.size()),
                           ::litert::HwAccelerators::kCpu));

    InferenceContext ctx;
    // Give prefill_input_buffers a mismatched Int16 buffer (32 bytes vs
    // Float32's 64 bytes) so `correct_mismatched_prefill_buffers` also runs.
    ctx.prefill_input_buffers.emplace(
        tc.k_name,
        CreateTensorBufferWithDims(std::vector<int16_t>(16, 0),
                                   ElementType::Int16, {1, 1, 4, 4}));
    ctx.prefill_input_buffers.emplace(
        tc.v_name,
        CreateTensorBufferWithDims(std::vector<int16_t>(16, 0),
                                   ElementType::Int16, {1, 1, 4, 4}));

    ASSERT_OK(ApplyLegacyKvCacheWorkarounds(
        compiled, "prefill_4", /*llm_metadata=*/nullptr, {64},
        /*has_per_layer_embeddings=*/false, /*kv_cache_init_value=*/9,
        /*enable_npu_debug_logging=*/true, ctx));

    ASSERT_TRUE(ctx.decode_input_buffers.contains(tc.k_name)) << tc.k_name;
    ASSERT_TRUE(ctx.decode_input_buffers.contains(tc.v_name)) << tc.v_name;
    auto lock = TensorBufferScopedLock::Create<int16_t>(
        ctx.decode_input_buffers.at(tc.k_name), TensorBuffer::LockMode::kRead);
    ASSERT_TRUE(lock.HasValue());
    EXPECT_EQ(lock->second[0], 9) << tc.k_name;
  }
}

}  // namespace
}  // namespace litert::lm
