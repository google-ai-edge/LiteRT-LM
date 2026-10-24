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
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "litert/cc/litert_buffer_ref.h"  // from @litert
#include "litert/cc/litert_common.h"  // from @litert
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/cc/litert_tensor_buffer_types.h"  // from @litert
#include "litert/test/matchers.h"  // from @litert
#include "runtime/executor/llm_litert_npu_compiled_model_executor.h"
#include "runtime/executor/npu/llm_litert_npu_mask.h"
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
using ::testing::ElementsAre;
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

}  // namespace
}  // namespace litert::lm
