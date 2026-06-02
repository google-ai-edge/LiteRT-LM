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

#include "runtime/executor/llm_litert_npu_compiled_model_executor_utils.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <random>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/cc/litert_tensor_buffer_types.h"  // from @litert

namespace litert::lm {
namespace {

using ::litert::ElementType;
using ::litert::Layout;
using ::litert::RankedTensorType;
using ::litert::TensorBuffer;
using ::litert::TensorBufferScopedLock;

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

std::vector<uint8_t> PackInt4(const std::vector<int8_t>& unpacked) {
  std::vector<uint8_t> packed;
  for (size_t i = 0; i < unpacked.size(); i += 2) {
    int8_t low = unpacked[i] & 0xF;
    int8_t high = 0;
    if (i + 1 < unpacked.size()) {
      high = unpacked[i + 1] & 0xF;
    }
    packed.push_back((high << 4) | low);
  }
  return packed;
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
    TensorBuffer buffer = std::move(*buffer_expected);
    auto lock_expected = TensorBufferScopedLock::Create<T>(
        buffer, TensorBuffer::LockMode::kWrite);
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
        auto result = FindMaxIndex<T>(buffer, use_neon);
        ASSERT_TRUE(result.ok());
        EXPECT_EQ(*result, pos) << "Failed at pos " << pos << " for size "
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
      auto result = FindMaxIndex<T>(buffer, use_neon);
      ASSERT_TRUE(result.ok());
      // Our implementation should return the first occurrence
      EXPECT_EQ(*result, first_pos) << "Failed multiple occurrences for size "
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
    auto result = FindMaxIndex<float>(buffer, use_neon);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(*result, ReferenceFindMaxIndex(data)) << "use_neon=" << use_neon;
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
    auto result = FindMaxIndex<int16_t>(buffer, use_neon);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(*result, ReferenceFindMaxIndex(data)) << "use_neon=" << use_neon;
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
    auto result = FindMaxIndex<int8_t>(buffer, use_neon);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(*result, ReferenceFindMaxIndex(data)) << "use_neon=" << use_neon;
  }
}

TEST_F(ExecutorUtilsTest, ApplyGreedySamplingCrossVerify) {
  std::vector<float> data = {0.1f, 0.9f, 0.4f};
  TensorBuffer buffer = CreateTensorBuffer(data, ElementType::Float32);
  for (bool use_neon : {false, true}) {
    auto result = ApplyGreedySampling(buffer, use_neon);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(*result, 1) << "use_neon=" << use_neon;
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

TEST_F(ExecutorUtilsTest, HWKVCacheUpdateBasic) {
  int hidden_dim = 4;
  int cache_seq = 10;
  int slice_seq = 2;
  int start_pos = 3;

  std::vector<float> cache_data(hidden_dim * cache_seq, 0.0f);
  std::vector<float> slice_data = {1.0f, 2.0f, 3.0f, 4.0f,
                                   5.0f, 6.0f, 7.0f, 8.0f};
  std::vector<int32_t> pos_data = {start_pos};

  absl::flat_hash_map<absl::string_view, TensorBuffer> in_buffers;
  in_buffers.emplace("input_pos",
                     CreateTensorBuffer(pos_data, ElementType::Int32));
  in_buffers.emplace("kv_cache_k_0", CreateTensorBufferWithDims(
                                         cache_data, ElementType::Float32,
                                         {1, cache_seq, hidden_dim}));
  in_buffers.emplace("kv_cache_v_0", CreateTensorBufferWithDims(
                                         cache_data, ElementType::Float32,
                                         {1, cache_seq, hidden_dim}));
  in_buffers.emplace("kv_slice_k_0", CreateTensorBufferWithDims(
                                         slice_data, ElementType::Float32,
                                         {1, slice_seq, hidden_dim}));
  in_buffers.emplace("kv_slice_v_0", CreateTensorBufferWithDims(
                                         slice_data, ElementType::Float32,
                                         {1, slice_seq, hidden_dim}));

  absl::flat_hash_map<absl::string_view, TensorBuffer> out_buffers;

  ASSERT_TRUE(HWKVCacheUpdate(in_buffers, out_buffers).ok());

  auto lock_expected = TensorBufferScopedLock::Create<float>(
      in_buffers.at("kv_cache_k_0"), TensorBuffer::LockMode::kRead);
  auto& lock = *lock_expected;
  for (int i = 0; i < slice_seq * hidden_dim; ++i) {
    EXPECT_EQ(lock.second[start_pos * hidden_dim + i], slice_data[i]);
  }
}

TEST_F(ExecutorUtilsTest, HWKVCacheUpdateTransposedInt8) {
  // Tests the path where last_dim_matches == false (transposed layout)
  int hidden_dim = 32;  // multiple of 16 for NEON
  int cache_seq = 64;
  int start_pos = 5;

  std::vector<int8_t> cache_data(hidden_dim * cache_seq, 0);
  std::vector<int8_t> slice_data(hidden_dim);
  for (int i = 0; i < hidden_dim; ++i) slice_data[i] = i + 1;
  std::vector<int32_t> pos_data = {start_pos};

  absl::flat_hash_map<absl::string_view, TensorBuffer> in_buffers;
  in_buffers.emplace("input_pos",
                     CreateTensorBuffer(pos_data, ElementType::Int32));
  in_buffers.emplace("kv_cache_k_0",
                     CreateTensorBufferWithDims(cache_data, ElementType::Int8,
                                                {1, hidden_dim, cache_seq}));
  in_buffers.emplace("kv_cache_v_0",
                     CreateTensorBufferWithDims(cache_data, ElementType::Int8,
                                                {1, hidden_dim, cache_seq}));
  in_buffers.emplace("kv_slice_k_0",
                     CreateTensorBufferWithDims(slice_data, ElementType::Int8,
                                                {1, 1, 1, hidden_dim}));
  in_buffers.emplace("kv_slice_v_0",
                     CreateTensorBufferWithDims(slice_data, ElementType::Int8,
                                                {1, 1, 1, hidden_dim}));

  absl::flat_hash_map<absl::string_view, TensorBuffer> out_buffers;

  ASSERT_TRUE(HWKVCacheUpdate(in_buffers, out_buffers).ok());

  auto lock_expected = TensorBufferScopedLock::Create<int8_t>(
      in_buffers.at("kv_cache_k_0"), TensorBuffer::LockMode::kRead);
  auto& lock = *lock_expected;
  for (int h = 0; h < hidden_dim; ++h) {
    EXPECT_EQ(lock.second[h * cache_seq + start_pos], slice_data[h]);
  }
}

TEST_F(ExecutorUtilsTest, HWKVCacheUpdateTransposedInt16) {
  int hidden_dim = 16;  // multiple of 8 for NEON
  int cache_seq = 32;
  int start_pos = 10;

  std::vector<int16_t> cache_data(hidden_dim * cache_seq, 0);
  std::vector<int16_t> slice_data(hidden_dim);
  for (int i = 0; i < hidden_dim; ++i) slice_data[i] = i + 100;
  std::vector<int32_t> pos_data = {start_pos};

  absl::flat_hash_map<absl::string_view, TensorBuffer> in_buffers;
  in_buffers.emplace("input_pos",
                     CreateTensorBuffer(pos_data, ElementType::Int32));
  in_buffers.emplace("kv_cache_k_0",
                     CreateTensorBufferWithDims(cache_data, ElementType::Int16,
                                                {1, hidden_dim, cache_seq}));
  in_buffers.emplace("kv_cache_v_0",
                     CreateTensorBufferWithDims(cache_data, ElementType::Int16,
                                                {1, hidden_dim, cache_seq}));
  in_buffers.emplace("kv_slice_k_0",
                     CreateTensorBufferWithDims(slice_data, ElementType::Int16,
                                                {1, 1, 1, hidden_dim}));
  in_buffers.emplace("kv_slice_v_0",
                     CreateTensorBufferWithDims(slice_data, ElementType::Int16,
                                                {1, 1, 1, hidden_dim}));

  absl::flat_hash_map<absl::string_view, TensorBuffer> out_buffers;

  ASSERT_TRUE(HWKVCacheUpdate(in_buffers, out_buffers).ok());

  auto lock_expected = TensorBufferScopedLock::Create<int16_t>(
      in_buffers.at("kv_cache_k_0"), TensorBuffer::LockMode::kRead);
  auto& lock = *lock_expected;
  for (int h = 0; h < hidden_dim; ++h) {
    EXPECT_EQ(lock.second[h * cache_seq + start_pos], slice_data[h]);
  }
}

TEST_F(ExecutorUtilsTest, HWKVCacheUpdateOutOfRange) {
  int hidden_dim = 4;
  int cache_seq = 5;
  int slice_seq = 2;
  int start_pos = 4;  // 4 + 2 > 5, should error

  std::vector<float> cache_data(hidden_dim * cache_seq, 0.0f);
  std::vector<float> slice_data(hidden_dim * slice_seq, 1.0f);
  std::vector<int32_t> pos_data = {start_pos};

  absl::flat_hash_map<absl::string_view, TensorBuffer> in_buffers;
  in_buffers.emplace("input_pos",
                     CreateTensorBuffer(pos_data, ElementType::Int32));
  in_buffers.emplace("kv_cache_k_0", CreateTensorBufferWithDims(
                                         cache_data, ElementType::Float32,
                                         {1, cache_seq, hidden_dim}));
  in_buffers.emplace("kv_cache_v_0", CreateTensorBufferWithDims(
                                         cache_data, ElementType::Float32,
                                         {1, cache_seq, hidden_dim}));
  in_buffers.emplace("kv_slice_k_0", CreateTensorBufferWithDims(
                                         slice_data, ElementType::Float32,
                                         {1, slice_seq, hidden_dim}));
  in_buffers.emplace("kv_slice_v_0", CreateTensorBufferWithDims(
                                         slice_data, ElementType::Float32,
                                         {1, slice_seq, hidden_dim}));

  absl::flat_hash_map<absl::string_view, TensorBuffer> out_buffers;

  EXPECT_FALSE(HWKVCacheUpdate(in_buffers, out_buffers).ok());
}

TEST_F(ExecutorUtilsTest, HWKVCacheUpdateGemma3nPrefill) {
  int hidden_dim = 256;
  int cache_seq = 2048;
  int slice_seq = 128;

  std::vector<int16_t> cache_data(2 * hidden_dim * cache_seq, 0);
  std::vector<int16_t> slice_data(2 * hidden_dim * slice_seq);
  for (int i = 0; i < 2 * hidden_dim * slice_seq; ++i) slice_data[i] = i + 1;
  std::vector<int32_t> pos_data(slice_seq, 0);

  absl::flat_hash_map<absl::string_view, TensorBuffer> in_buffers;
  in_buffers.emplace(
      "input_pos",
      CreateTensorBufferWithDims(pos_data, ElementType::Int32, {slice_seq}));
  in_buffers.emplace("kv_cache_k_0",
                     CreateTensorBufferWithDims(cache_data, ElementType::Int16,
                                                {1, 2, cache_seq, hidden_dim}));
  in_buffers.emplace("kv_cache_v_0",
                     CreateTensorBufferWithDims(cache_data, ElementType::Int16,
                                                {1, 2, hidden_dim, cache_seq}));
  in_buffers.emplace("kv_slice_k_0",
                     CreateTensorBufferWithDims(slice_data, ElementType::Int16,
                                                {1, 2, slice_seq, hidden_dim}));
  in_buffers.emplace("kv_slice_v_0",
                     CreateTensorBufferWithDims(slice_data, ElementType::Int16,
                                                {1, 2, hidden_dim, slice_seq}));

  absl::flat_hash_map<absl::string_view, TensorBuffer> out_buffers;

  ASSERT_TRUE(HWKVCacheUpdate(in_buffers, out_buffers).ok());

  auto k_lock_expected = TensorBufferScopedLock::Create<int16_t>(
      in_buffers.at("kv_cache_k_0"), TensorBuffer::LockMode::kRead);
  auto& k_lock = *k_lock_expected;
  for (int o = 0; o < 2; ++o) {
    for (int i = 0; i < slice_seq * hidden_dim; ++i) {
      EXPECT_EQ(k_lock.second[o * cache_seq * hidden_dim + i],
                slice_data[o * slice_seq * hidden_dim + i]);
    }
  }

  auto v_lock_expected = TensorBufferScopedLock::Create<int16_t>(
      in_buffers.at("kv_cache_v_0"), TensorBuffer::LockMode::kRead);
  auto& v_lock = *v_lock_expected;
  for (int h = 0; h < 2 * hidden_dim; ++h) {
    for (int s = 0; s < slice_seq; ++s) {
      EXPECT_EQ(v_lock.second[h * cache_seq + s],
                slice_data[h * slice_seq + s]);
    }
  }
}

TEST_F(ExecutorUtilsTest, HWMaskUpdateInt8) {
  int seq_q = 1;
  int seq_k = 4096 + 4;  // capacity + batch
  int time_step = 100;
  int8_t valid_val = 127;
  int8_t masked_val = -128;

  absl::flat_hash_map<absl::string_view, TensorBuffer> in_buffers;
  std::vector<int32_t> time_step_data = {time_step};
  in_buffers.emplace("time_step",
                     CreateTensorBuffer(time_step_data, ElementType::Int32));

  absl::flat_hash_map<absl::string_view, TensorBuffer> out_buffers;
  std::vector<int8_t> mask_data(seq_q * seq_k, 0);
  out_buffers.emplace("mask_local",
                      CreateTensorBufferWithDims(mask_data, ElementType::Int8,
                                                 {1, seq_q, seq_k}));
  out_buffers.emplace("mask_global",
                      CreateTensorBufferWithDims(mask_data, ElementType::Int8,
                                                 {1, seq_q, seq_k}));

  ASSERT_TRUE(HWMaskUpdate(in_buffers, out_buffers).ok());

  auto local_lock_expected = TensorBufferScopedLock::Create<int8_t>(
      out_buffers.at("mask_local"), TensorBuffer::LockMode::kRead);
  auto global_lock_expected = TensorBufferScopedLock::Create<int8_t>(
      out_buffers.at("mask_global"), TensorBuffer::LockMode::kRead);
  auto& local_lock = *local_lock_expected;
  auto& global_lock = *global_lock_expected;

  // Check KV cache part (0 to 4095)
  for (int k = 0; k < 4096; ++k) {
    if (k < time_step) {
      EXPECT_EQ(global_lock.second[k], valid_val) << "k=" << k;
      if (k >= time_step - 511) {
        EXPECT_EQ(local_lock.second[k], valid_val) << "k=" << k;
      } else {
        EXPECT_EQ(local_lock.second[k], masked_val) << "k=" << k;
      }
    } else {
      EXPECT_EQ(global_lock.second[k], masked_val) << "k=" << k;
      EXPECT_EQ(local_lock.second[k], masked_val) << "k=" << k;
    }
  }
}

TEST_F(ExecutorUtilsTest, HWMaskUpdateInt16) {
  int seq_q = 4;  // Verify case
  int seq_k = 4096 + 4;
  int time_step = 2000;
  int16_t valid_val = 0;
  int16_t masked_val = -32767;

  absl::flat_hash_map<absl::string_view, TensorBuffer> in_buffers;
  std::vector<int32_t> time_step_data = {time_step};
  in_buffers.emplace("time_step",
                     CreateTensorBuffer(time_step_data, ElementType::Int32));
  std::vector<int32_t> tokens_data = {1, 2, 3, -1};  // last token is invalid
  in_buffers.emplace("input_tokens",
                     CreateTensorBuffer(tokens_data, ElementType::Int32));

  absl::flat_hash_map<absl::string_view, TensorBuffer> out_buffers;
  std::vector<int16_t> mask_data(seq_q * seq_k, 0);
  out_buffers.emplace("mask_local",
                      CreateTensorBufferWithDims(mask_data, ElementType::Int16,
                                                 {1, seq_q, seq_k}));
  out_buffers.emplace("mask_global",
                      CreateTensorBufferWithDims(mask_data, ElementType::Int16,
                                                 {1, seq_q, seq_k}));

  ASSERT_TRUE(HWMaskUpdate(in_buffers, out_buffers).ok());

  auto local_lock_expected = TensorBufferScopedLock::Create<int16_t>(
      out_buffers.at("mask_local"), TensorBuffer::LockMode::kRead);
  auto global_lock_expected = TensorBufferScopedLock::Create<int16_t>(
      out_buffers.at("mask_global"), TensorBuffer::LockMode::kRead);
  auto& local_lock = *local_lock_expected;
  auto& global_lock = *global_lock_expected;

  for (int q = 0; q < seq_q; ++q) {
    // Check batch part (4096 to 4099)
    for (int k_rel = 0; k_rel < 4; ++k_rel) {
      int k = 4096 + k_rel;
      if (k_rel <= q && tokens_data[k_rel] != -1) {
        EXPECT_EQ(global_lock.second[q * seq_k + k], valid_val)
            << "q=" << q << " k=" << k;
        EXPECT_EQ(local_lock.second[q * seq_k + k], valid_val)
            << "q=" << q << " k=" << k;
      } else {
        EXPECT_EQ(global_lock.second[q * seq_k + k], masked_val)
            << "q=" << q << " k=" << k;
        EXPECT_EQ(local_lock.second[q * seq_k + k], masked_val)
            << "q=" << q << " k=" << k;
      }
    }
  }
}

TEST_F(ExecutorUtilsTest, HWMaskUpdateGemma3Prefill) {
  // Gemma 3 Prefill: capacity 1280, prefill 128 -> seq_k = 1408
  int seq_q = 128;
  int seq_k = 1408;
  int time_step = 0;  // First prefill
  int8_t valid_val = 127;
  int8_t masked_val = -128;

  absl::flat_hash_map<absl::string_view, TensorBuffer> in_buffers;
  std::vector<int32_t> time_step_data = {time_step};
  in_buffers.emplace("time_step",
                     CreateTensorBuffer(time_step_data, ElementType::Int32));

  absl::flat_hash_map<absl::string_view, TensorBuffer> out_buffers;
  std::vector<int8_t> mask_data(seq_q * seq_k, 0);
  out_buffers.emplace("mask_local",
                      CreateTensorBufferWithDims(mask_data, ElementType::Int8,
                                                 {1, seq_q, seq_k}));
  out_buffers.emplace("mask_global",
                      CreateTensorBufferWithDims(mask_data, ElementType::Int8,
                                                 {1, seq_q, seq_k}));

  ASSERT_TRUE(HWMaskUpdate(in_buffers, out_buffers).ok());

  auto global_lock_expected = TensorBufferScopedLock::Create<int8_t>(
      out_buffers.at("mask_global"), TensorBuffer::LockMode::kRead);
  auto& global_lock = *global_lock_expected;

  // Capacity 1280.
  // Draft part should start at 1280.
  // For prefill chunk, token q attends to tokens 0..q within the chunk.
  int q = 10;
  int k_chunk = 5;
  int k_global = 1280 + k_chunk;
  EXPECT_EQ(global_lock.second[q * seq_k + k_global], valid_val)
      << "q=" << q << " k=" << k_global;

  k_chunk = 15;
  k_global = 1280 + k_chunk;
  EXPECT_EQ(global_lock.second[q * seq_k + k_global], masked_val)
      << "q=" << q << " k=" << k_global;
}

TEST_F(ExecutorUtilsTest, HWMaskUpdateGemma3Decode) {
  // Gemma 3 Decode: capacity 1280, batch 1 -> seq_k = 1281
  int seq_q = 1;
  int seq_k = 1281;
  int time_step = 500;
  int8_t valid_val = 127;

  absl::flat_hash_map<absl::string_view, TensorBuffer> in_buffers;
  std::vector<int32_t> time_step_data = {time_step};
  in_buffers.emplace("time_step",
                     CreateTensorBuffer(time_step_data, ElementType::Int32));

  absl::flat_hash_map<absl::string_view, TensorBuffer> out_buffers;
  std::vector<int8_t> mask_data(seq_q * seq_k, 0);
  out_buffers.emplace("mask_local",
                      CreateTensorBufferWithDims(mask_data, ElementType::Int8,
                                                 {1, seq_q, seq_k}));
  out_buffers.emplace("mask_global",
                      CreateTensorBufferWithDims(mask_data, ElementType::Int8,
                                                 {1, seq_q, seq_k}));

  ASSERT_TRUE(HWMaskUpdate(in_buffers, out_buffers).ok());

  auto global_lock_expected = TensorBufferScopedLock::Create<int8_t>(
      out_buffers.at("mask_global"), TensorBuffer::LockMode::kRead);
  auto& global_lock = *global_lock_expected;

  // Check KV cache part
  EXPECT_EQ(global_lock.second[100], valid_val);
  EXPECT_EQ(global_lock.second[600], -128);

  // Check new token part (at index 1280)
  EXPECT_EQ(global_lock.second[1280], valid_val);
}

TEST_F(ExecutorUtilsTest, HWPerLayerEmbeddingLookupFloat32) {
  constexpr int kNumTables = 2;
  constexpr int kColSize = 4;

  std::vector<int8_t> table0_unpacked = {0, 1, 2, 3, -1, -2, -3, -4,
                                         4, 5, 6, 7, -5, -6, -7, -8};
  std::vector<int8_t> table1_unpacked = {0,  -1, -2, -3, 4, 5, 6, 7,
                                         -4, -5, -6, -7, 1, 2, 3, -8};

  std::vector<uint8_t> table0_packed = PackInt4(table0_unpacked);
  std::vector<uint8_t> table1_packed = PackInt4(table1_unpacked);

  std::vector<const uint8_t*> table_ptrs = {table0_packed.data(),
                                            table1_packed.data()};

  std::vector<float> scales0 = {1.0f, 2.0f, 0.5f, 1.0f};
  std::vector<float> scales1 = {0.5f};

  HWQuantizationParams qp[kNumTables];
  qp[0].scales = scales0.data();
  qp[0].is_per_channel = true;
  qp[1].scales = scales1.data();
  qp[1].is_per_channel = false;

  std::vector<int32_t> token_ids = {1, 2};
  int num_tokens = token_ids.size();

  std::vector<float> output(num_tokens * kNumTables * kColSize, 0.0f);

  auto status = HWPerLayerEmbeddingLookup(
      token_ids.data(), num_tokens, table_ptrs.data(), qp, kNumTables, kColSize,
      output.data(), litert::ElementType::Float32);

  ASSERT_TRUE(status.ok());

  std::vector<float> expected_output = {-2.0f, -4.0f, -6.0f, -8.0f, 2.0f, 2.5f,
                                        3.0f,  3.5f,  2.0f,  2.5f,  3.0f, 3.5f,
                                        -2.0f, -2.5f, -3.0f, -3.5f};

  for (size_t i = 0; i < output.size(); ++i) {
    EXPECT_NEAR(output[i], expected_output[i], 1e-5) << "Index " << i;
  }
}

TEST_F(ExecutorUtilsTest, HWPerLayerEmbeddingLookupInt16) {
  constexpr int kNumTables = 1;
  constexpr int kColSize = 4;

  std::vector<int8_t> table0_unpacked = {0, 1, 2, 3, -1, -2, -3, -4,
                                         4, 5, 6, 7, -5, -6, -7, -8};

  std::vector<uint8_t> table0_packed = PackInt4(table0_unpacked);
  std::vector<const uint8_t*> table_ptrs = {table0_packed.data()};

  std::vector<float> scales0 = {1.0f};

  HWQuantizationParams qp[kNumTables];
  qp[0].scales = scales0.data();
  qp[0].is_per_channel = false;

  std::vector<int32_t> token_ids = {1};
  int num_tokens = token_ids.size();

  std::vector<int16_t> output(num_tokens * kNumTables * kColSize, 0);

  float final_scale = 0.5f;
  int32_t final_zero_point = 10;

  auto status = HWPerLayerEmbeddingLookup(
      token_ids.data(), num_tokens, table_ptrs.data(), qp, kNumTables, kColSize,
      output.data(), litert::ElementType::Int16, final_scale, final_zero_point);

  ASSERT_TRUE(status.ok());

  std::vector<int16_t> expected_output = {8, 6, 4, 2};

  for (size_t i = 0; i < output.size(); ++i) {
    EXPECT_EQ(output[i], expected_output[i]) << "Index " << i;
  }
}

TEST_F(ExecutorUtilsTest, HWPerLayerEmbeddingLookupNeon) {
  constexpr int kNumTables = 1;
  constexpr int kColSize = 32;

  std::vector<int8_t> table0_unpacked(kColSize);
  for (int i = 0; i < kColSize; ++i) {
    table0_unpacked[i] = (i % 16) - 8;
  }

  std::vector<uint8_t> table0_packed = PackInt4(table0_unpacked);
  std::vector<const uint8_t*> table_ptrs = {table0_packed.data()};

  std::vector<float> scales0 = {1.0f};

  HWQuantizationParams qp[kNumTables];
  qp[0].scales = scales0.data();
  qp[0].is_per_channel = false;

  std::vector<int32_t> token_ids = {0};
  int num_tokens = token_ids.size();

  std::vector<float> output(num_tokens * kNumTables * kColSize, 0.0f);

  auto status = HWPerLayerEmbeddingLookup(
      token_ids.data(), num_tokens, table_ptrs.data(), qp, kNumTables, kColSize,
      output.data(), litert::ElementType::Float32);

  ASSERT_TRUE(status.ok());

  for (size_t i = 0; i < output.size(); ++i) {
    EXPECT_NEAR(output[i], static_cast<float>(table0_unpacked[i]), 1e-5)
        << "Index " << i;
  }
}

TEST_F(ExecutorUtilsTest, HWKVCacheUpdateInvalidPos) {
  int hidden_dim = 4;
  int cache_seq = 5;
  int slice_seq = 2;
  int start_pos = -1;  // Invalid negative start_pos

  std::vector<float> cache_data(hidden_dim * cache_seq, 0.0f);
  std::vector<float> slice_data(hidden_dim * slice_seq, 1.0f);
  std::vector<int32_t> pos_data = {start_pos};

  absl::flat_hash_map<absl::string_view, TensorBuffer> in_buffers;
  in_buffers.emplace("input_pos",
                     CreateTensorBuffer(pos_data, ElementType::Int32));
  in_buffers.emplace("kv_cache_k_0", CreateTensorBufferWithDims(
                                         cache_data, ElementType::Float32,
                                         {1, cache_seq, hidden_dim}));
  in_buffers.emplace("kv_cache_v_0", CreateTensorBufferWithDims(
                                         cache_data, ElementType::Float32,
                                         {1, cache_seq, hidden_dim}));
  in_buffers.emplace("kv_slice_k_0", CreateTensorBufferWithDims(
                                         slice_data, ElementType::Float32,
                                         {1, slice_seq, hidden_dim}));
  in_buffers.emplace("kv_slice_v_0", CreateTensorBufferWithDims(
                                         slice_data, ElementType::Float32,
                                         {1, slice_seq, hidden_dim}));

  absl::flat_hash_map<absl::string_view, TensorBuffer> out_buffers;
  EXPECT_FALSE(HWKVCacheUpdate(in_buffers, out_buffers).ok());
}

TEST_F(ExecutorUtilsTest, HWKVCacheUpdateMismatchedOuterDims) {
  int hidden_dim = 4;
  int cache_seq = 5;
  int slice_seq = 2;
  int start_pos = 0;

  // Cache outer_size = 2 (dim0 = 2)
  std::vector<float> cache_data(2 * hidden_dim * cache_seq, 0.0f);
  // Slice outer_size = 1 (dim0 = 1)
  std::vector<float> slice_data(1 * hidden_dim * slice_seq, 1.0f);
  std::vector<int32_t> pos_data = {start_pos};

  absl::flat_hash_map<absl::string_view, TensorBuffer> in_buffers;
  in_buffers.emplace("input_pos",
                     CreateTensorBuffer(pos_data, ElementType::Int32));
  in_buffers.emplace("kv_cache_k_0", CreateTensorBufferWithDims(
                                         cache_data, ElementType::Float32,
                                         {2, cache_seq, hidden_dim}));
  in_buffers.emplace("kv_cache_v_0", CreateTensorBufferWithDims(
                                         cache_data, ElementType::Float32,
                                         {2, cache_seq, hidden_dim}));
  in_buffers.emplace("kv_slice_k_0", CreateTensorBufferWithDims(
                                         slice_data, ElementType::Float32,
                                         {1, slice_seq, hidden_dim}));
  in_buffers.emplace("kv_slice_v_0", CreateTensorBufferWithDims(
                                         slice_data, ElementType::Float32,
                                         {1, slice_seq, hidden_dim}));

  absl::flat_hash_map<absl::string_view, TensorBuffer> out_buffers;
  EXPECT_FALSE(HWKVCacheUpdate(in_buffers, out_buffers).ok());
}

TEST_F(ExecutorUtilsTest, HWKVCacheUpdateMismatchedElementTypes) {
  int hidden_dim = 4;
  int cache_seq = 5;
  int slice_seq = 2;
  int start_pos = 0;

  std::vector<float> cache_data(hidden_dim * cache_seq, 0.0f);
  std::vector<int8_t> slice_data(hidden_dim * slice_seq, 1);
  std::vector<int32_t> pos_data = {start_pos};

  absl::flat_hash_map<absl::string_view, TensorBuffer> in_buffers;
  in_buffers.emplace("input_pos",
                     CreateTensorBuffer(pos_data, ElementType::Int32));
  in_buffers.emplace("kv_cache_k_0", CreateTensorBufferWithDims(
                                         cache_data, ElementType::Float32,
                                         {1, cache_seq, hidden_dim}));
  in_buffers.emplace("kv_cache_v_0", CreateTensorBufferWithDims(
                                         cache_data, ElementType::Float32,
                                         {1, cache_seq, hidden_dim}));
  in_buffers.emplace("kv_slice_k_0",
                     CreateTensorBufferWithDims(slice_data, ElementType::Int8,
                                                {1, slice_seq, hidden_dim}));
  in_buffers.emplace("kv_slice_v_0",
                     CreateTensorBufferWithDims(slice_data, ElementType::Int8,
                                                {1, slice_seq, hidden_dim}));

  absl::flat_hash_map<absl::string_view, TensorBuffer> out_buffers;
  EXPECT_FALSE(HWKVCacheUpdate(in_buffers, out_buffers).ok());
}

TEST_F(ExecutorUtilsTest, HWKVCacheUpdateDequantizeInt16ToFloat32) {
  int hidden_dim = 4;
  int cache_seq = 5;
  int slice_seq = 2;
  int start_pos = 1;

  std::vector<float> cache_data(hidden_dim * cache_seq, 0.0f);
  std::vector<int16_t> slice_data = {
      100, 200, 300, 400,  // step 0
      500, 600, 700, 800   // step 1
  };
  std::vector<int32_t> pos_data = {start_pos};

  absl::flat_hash_map<absl::string_view, TensorBuffer> in_buffers;
  in_buffers.emplace("input_pos",
                     CreateTensorBuffer(pos_data, ElementType::Int32));
  in_buffers.emplace("kv_cache_k_0", CreateTensorBufferWithDims(
                                         cache_data, ElementType::Float32,
                                         {1, cache_seq, hidden_dim}));
  in_buffers.emplace("kv_cache_v_0", CreateTensorBufferWithDims(
                                         cache_data, ElementType::Float32,
                                         {1, cache_seq, hidden_dim}));
  in_buffers.emplace("kv_slice_k_0",
                     CreateTensorBufferWithDims(slice_data, ElementType::Int16,
                                                {1, slice_seq, hidden_dim}));
  in_buffers.emplace("kv_slice_v_0",
                     CreateTensorBufferWithDims(slice_data, ElementType::Int16,
                                                {1, slice_seq, hidden_dim}));

  absl::flat_hash_map<absl::string_view, TensorBuffer> out_buffers;

  absl::flat_hash_map<absl::string_view, HWQuantParams> quant_params;
  HWQuantParams k_params;
  k_params.scale = 0.1f;
  k_params.zero_point = 50;
  quant_params["kv_slice_k_0"] = k_params;

  HWQuantParams v_params;
  v_params.scale = 0.2f;
  v_params.zero_point = 100;
  quant_params["kv_slice_v_0"] = v_params;

  ASSERT_TRUE(HWKVCacheUpdate(in_buffers, out_buffers, quant_params).ok());

  // Verify K cache (dequantized using scale=0.1, zp=50)
  auto k_lock_expected = TensorBufferScopedLock::Create<float>(
      in_buffers.at("kv_cache_k_0"), TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(k_lock_expected.HasValue());
  auto& k_lock = *k_lock_expected;
  for (int i = 0; i < slice_seq * hidden_dim; ++i) {
    float expected = (static_cast<float>(slice_data[i]) - 50.0f) * 0.1f;
    EXPECT_NEAR(k_lock.second[start_pos * hidden_dim + i], expected, 1e-5)
        << "Index " << i;
  }

  // Verify V cache (dequantized using scale=0.2, zp=100)
  auto v_lock_expected = TensorBufferScopedLock::Create<float>(
      in_buffers.at("kv_cache_v_0"), TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(v_lock_expected.HasValue());
  auto& v_lock = *v_lock_expected;
  for (int i = 0; i < slice_seq * hidden_dim; ++i) {
    float expected = (static_cast<float>(slice_data[i]) - 100.0f) * 0.2f;
    EXPECT_NEAR(v_lock.second[start_pos * hidden_dim + i], expected, 1e-5)
        << "Index " << i;
  }
}

TEST_F(ExecutorUtilsTest, HWKVCacheUpdateConvolution) {
  int hidden_dim = 4;
  int cache_seq = 10;

  std::vector<float> cache_data(hidden_dim * cache_seq, 0.0f);
  std::vector<float> slice_data(hidden_dim * cache_seq, 1.0f);
  std::vector<int32_t> pos_data = {0};

  absl::flat_hash_map<absl::string_view, TensorBuffer> in_buffers;
  in_buffers.emplace("input_pos",
                     CreateTensorBuffer(pos_data, ElementType::Int32));
  in_buffers.emplace("kv_cache_c_0", CreateTensorBufferWithDims(
                                         cache_data, ElementType::Float32,
                                         {1, cache_seq, hidden_dim}));
  in_buffers.emplace("kv_slice_c_0", CreateTensorBufferWithDims(
                                         slice_data, ElementType::Float32,
                                         {1, cache_seq, hidden_dim}));

  absl::flat_hash_map<absl::string_view, TensorBuffer> out_buffers;

  ASSERT_TRUE(HWKVCacheUpdate(in_buffers, out_buffers).ok());

  auto lock_expected = TensorBufferScopedLock::Create<float>(
      in_buffers.at("kv_cache_c_0"), TensorBuffer::LockMode::kRead);
  auto& lock = *lock_expected;
  for (int i = 0; i < (int)slice_data.size(); ++i) {
    EXPECT_EQ(lock.second[i], slice_data[i]);
  }
}

TEST_F(ExecutorUtilsTest, HWKVCacheUpdateConvolutionOutBuffer) {
  int hidden_dim = 4;
  int cache_seq = 5;

  std::vector<float> in_cache_data(hidden_dim * cache_seq, 0.0f);
  std::vector<float> out_cache_data(hidden_dim * cache_seq, 0.0f);
  std::vector<float> slice_data(hidden_dim * cache_seq, 2.0f);
  std::vector<int32_t> pos_data = {0};

  absl::flat_hash_map<absl::string_view, TensorBuffer> in_buffers;
  in_buffers.emplace("input_pos",
                     CreateTensorBuffer(pos_data, ElementType::Int32));
  in_buffers.emplace("kv_cache_c_1", CreateTensorBufferWithDims(
                                         in_cache_data, ElementType::Float32,
                                         {1, cache_seq, hidden_dim}));
  in_buffers.emplace("kv_slice_c_1", CreateTensorBufferWithDims(
                                         slice_data, ElementType::Float32,
                                         {1, cache_seq, hidden_dim}));

  absl::flat_hash_map<absl::string_view, TensorBuffer> out_buffers;
  out_buffers.emplace("kv_cache_c_1", CreateTensorBufferWithDims(
                                          out_cache_data, ElementType::Float32,
                                          {1, cache_seq, hidden_dim}));

  ASSERT_TRUE(HWKVCacheUpdate(in_buffers, out_buffers).ok());

  // Check in_buffer
  {
    auto lock_expected = TensorBufferScopedLock::Create<float>(
        in_buffers.at("kv_cache_c_1"), TensorBuffer::LockMode::kRead);
    auto& lock = *lock_expected;
    for (int i = 0; i < (int)slice_data.size(); ++i) {
      EXPECT_EQ(lock.second[i], 2.0f);
    }
  }

  // Check out_buffer
  {
    auto lock_expected = TensorBufferScopedLock::Create<float>(
        out_buffers.at("kv_cache_c_1"), TensorBuffer::LockMode::kRead);
    auto& lock = *lock_expected;
    for (int i = 0; i < (int)slice_data.size(); ++i) {
      EXPECT_EQ(lock.second[i], 2.0f);
    }
  }
}

TEST_F(ExecutorUtilsTest, HWMaskUpdateFloat16) {
  int seq_q = 1;
  int seq_k = 4096 + 4;
  int time_step = 100;
  uint16_t valid_val = 0x0000;
  uint16_t masked_val = 0xFC00;

  absl::flat_hash_map<absl::string_view, TensorBuffer> in_buffers;
  std::vector<int32_t> time_step_data = {time_step};
  in_buffers.emplace("time_step",
                     CreateTensorBuffer(time_step_data, ElementType::Int32));

  absl::flat_hash_map<absl::string_view, TensorBuffer> out_buffers;
  std::vector<uint16_t> mask_data(seq_q * seq_k, 0);
  out_buffers.emplace(
      "mask_global", CreateTensorBufferWithDims(mask_data, ElementType::Float16,
                                                {1, seq_q, seq_k}));

  ASSERT_TRUE(HWMaskUpdate(in_buffers, out_buffers).ok());

  auto global_lock_expected = TensorBufferScopedLock::Create<uint16_t>(
      out_buffers.at("mask_global"), TensorBuffer::LockMode::kRead);
  auto& global_lock = *global_lock_expected;

  // Check KV cache part (0 to 4095)
  for (int k = 0; k < 4096; ++k) {
    if (k < time_step) {
      EXPECT_EQ(global_lock.second[k], valid_val) << "k=" << k;
    } else {
      EXPECT_EQ(global_lock.second[k], masked_val) << "k=" << k;
    }
  }
}

TEST_F(ExecutorUtilsTest, HWMaskUpdateBFloat16) {
  int seq_q = 1;
  int seq_k = 4096 + 4;
  int time_step = 100;
  uint16_t valid_val = 0x0000;
  uint16_t masked_val = 0xFF80;

  absl::flat_hash_map<absl::string_view, TensorBuffer> in_buffers;
  std::vector<int32_t> time_step_data = {time_step};
  in_buffers.emplace("time_step",
                     CreateTensorBuffer(time_step_data, ElementType::Int32));

  absl::flat_hash_map<absl::string_view, TensorBuffer> out_buffers;
  std::vector<uint16_t> mask_data(seq_q * seq_k, 0);
  out_buffers.emplace("mask_global",
                      CreateTensorBufferWithDims(
                          mask_data, ElementType::BFloat16, {1, seq_q, seq_k}));

  ASSERT_TRUE(HWMaskUpdate(in_buffers, out_buffers).ok());

  auto global_lock_expected = TensorBufferScopedLock::Create<uint16_t>(
      out_buffers.at("mask_global"), TensorBuffer::LockMode::kRead);
  auto& global_lock = *global_lock_expected;

  // Check KV cache part (0 to 4095)
  for (int k = 0; k < 4096; ++k) {
    if (k < time_step) {
      EXPECT_EQ(global_lock.second[k], valid_val) << "k=" << k;
    } else {
      EXPECT_EQ(global_lock.second[k], masked_val) << "k=" << k;
    }
  }
}

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

}  // namespace
}  // namespace litert::lm
