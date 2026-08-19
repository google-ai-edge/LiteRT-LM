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

#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/cc/litert_tensor_buffer_types.h"  // from @litert
#include "litert/test/matchers.h"  // from @litert

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
  ASSERT_TRUE(lock_expected.HasValue());
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
  ASSERT_TRUE(lock_expected.HasValue());
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
  ASSERT_TRUE(lock_expected.HasValue());
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
  ASSERT_TRUE(k_lock_expected.HasValue());
  auto& k_lock = *k_lock_expected;
  for (int o = 0; o < 2; ++o) {
    for (int i = 0; i < slice_seq * hidden_dim; ++i) {
      EXPECT_EQ(k_lock.second[o * cache_seq * hidden_dim + i],
                slice_data[o * slice_seq * hidden_dim + i]);
    }
  }

  auto v_lock_expected = TensorBufferScopedLock::Create<int16_t>(
      in_buffers.at("kv_cache_v_0"), TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(v_lock_expected.HasValue());
  auto& v_lock = *v_lock_expected;
  for (int h = 0; h < 2 * hidden_dim; ++h) {
    for (int s = 0; s < slice_seq; ++s) {
      EXPECT_EQ(v_lock.second[h * cache_seq + s],
                slice_data[h * slice_seq + s]);
    }
  }
}

TEST_F(ExecutorUtilsTest, HWKVCacheUpdateSWADecode) {
  int hidden_dim = 4;
  int cache_seq = 8;
  int slice_seq = 1;
  int start_pos = 9;  // 9 % 8 = 1

  std::vector<float> cache_data(hidden_dim * cache_seq, 0.0f);
  std::vector<float> slice_data = {1.0f, 2.0f, 3.0f, 4.0f};
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

  ASSERT_TRUE(
      HWKVCacheUpdate(in_buffers, out_buffers, {}, /*enable_swa=*/true).ok());

  auto lock_expected = TensorBufferScopedLock::Create<float>(
      in_buffers.at("kv_cache_k_0"), TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(lock_expected.HasValue());
  auto& lock = *lock_expected;

  // Verify that only index 1 (wrapped from 9) is updated
  int target_pos = start_pos % cache_seq;
  for (int i = 0; i < cache_seq; ++i) {
    for (int h = 0; h < hidden_dim; ++h) {
      float expected = (i == target_pos) ? slice_data[h] : 0.0f;
      EXPECT_EQ(lock.second[i * hidden_dim + h], expected)
          << "Mismatch at seq " << i << " head " << h;
    }
  }
}

TEST_F(ExecutorUtilsTest, HWKVCacheUpdateSWADecodeTransposed) {
  int hidden_dim = 4;
  int cache_seq = 8;
  int slice_seq = 1;
  int start_pos = 9;  // 9 % 8 = 1

  std::vector<float> cache_data(hidden_dim * cache_seq, 0.0f);
  std::vector<float> slice_data = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<int32_t> pos_data = {start_pos};

  absl::flat_hash_map<absl::string_view, TensorBuffer> in_buffers;
  in_buffers.emplace("input_pos",
                     CreateTensorBuffer(pos_data, ElementType::Int32));
  // Transposed cache: [1, hidden_dim, cache_seq]
  in_buffers.emplace("kv_cache_k_0", CreateTensorBufferWithDims(
                                         cache_data, ElementType::Float32,
                                         {1, hidden_dim, cache_seq}));
  in_buffers.emplace("kv_cache_v_0", CreateTensorBufferWithDims(
                                         cache_data, ElementType::Float32,
                                         {1, hidden_dim, cache_seq}));
  in_buffers.emplace("kv_slice_k_0", CreateTensorBufferWithDims(
                                         slice_data, ElementType::Float32,
                                         {1, 1, slice_seq, hidden_dim}));
  in_buffers.emplace("kv_slice_v_0", CreateTensorBufferWithDims(
                                         slice_data, ElementType::Float32,
                                         {1, 1, slice_seq, hidden_dim}));

  absl::flat_hash_map<absl::string_view, TensorBuffer> out_buffers;

  ASSERT_TRUE(
      HWKVCacheUpdate(in_buffers, out_buffers, {}, /*enable_swa=*/true).ok());

  auto lock_expected = TensorBufferScopedLock::Create<float>(
      in_buffers.at("kv_cache_k_0"), TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(lock_expected.HasValue());
  auto& lock = *lock_expected;

  int target_pos = start_pos % cache_seq;
  for (int h = 0; h < hidden_dim; ++h) {
    for (int s = 0; s < cache_seq; ++s) {
      float expected = (s == target_pos) ? slice_data[h] : 0.0f;
      EXPECT_EQ(lock.second[h * cache_seq + s], expected)
          << "Mismatch at head " << h << " seq " << s;
    }
  }
}

TEST_F(ExecutorUtilsTest, HWKVCacheUpdateSWAPrefillWrap) {
  int hidden_dim = 4;
  int cache_seq = 8;
  int slice_seq = 4;
  int start_pos = 6;  // 6 + 4 = 10 > 8. Wraps to 6, 7, 0, 1.

  std::vector<float> cache_data(hidden_dim * cache_seq, 0.0f);
  std::vector<float> slice_data = {
      1.0f,  2.0f,  3.0f,  4.0f,   // token 0
      5.0f,  6.0f,  7.0f,  8.0f,   // token 1
      9.0f,  10.0f, 11.0f, 12.0f,  // token 2
      13.0f, 14.0f, 15.0f, 16.0f   // token 3
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
  in_buffers.emplace("kv_slice_k_0", CreateTensorBufferWithDims(
                                         slice_data, ElementType::Float32,
                                         {1, slice_seq, hidden_dim}));
  in_buffers.emplace("kv_slice_v_0", CreateTensorBufferWithDims(
                                         slice_data, ElementType::Float32,
                                         {1, slice_seq, hidden_dim}));

  absl::flat_hash_map<absl::string_view, TensorBuffer> out_buffers;

  ASSERT_TRUE(
      HWKVCacheUpdate(in_buffers, out_buffers, {}, /*enable_swa=*/true).ok());

  auto lock_expected = TensorBufferScopedLock::Create<float>(
      in_buffers.at("kv_cache_k_0"), TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(lock_expected.HasValue());
  auto& lock = *lock_expected;

  // Expected mapping:
  // cache[6] <- slice[0] (1..4)
  // cache[7] <- slice[1] (5..8)
  // cache[0] <- slice[2] (9..12)
  // cache[1] <- slice[3] (13..16)

  std::vector<int> expected_slice_idx = {
      2,   // cache[0]
      3,   // cache[1]
      -1,  // cache[2]
      -1,  // cache[3]
      -1,  // cache[4]
      -1,  // cache[5]
      0,   // cache[6]
      1    // cache[7]
  };

  for (int i = 0; i < cache_seq; ++i) {
    int s_idx = expected_slice_idx[i];
    for (int h = 0; h < hidden_dim; ++h) {
      float expected =
          (s_idx == -1) ? 0.0f : slice_data[s_idx * hidden_dim + h];
      EXPECT_EQ(lock.second[i * hidden_dim + h], expected)
          << "Mismatch at seq " << i << " head " << h;
    }
  }
}

TEST_F(ExecutorUtilsTest, HWKVCacheUpdateSWAPrefillWrapTransposed) {
  int hidden_dim = 4;
  int cache_seq = 8;
  int slice_seq = 2;
  int start_pos = 7;  // 7 + 2 = 9 > 8. Wraps to 7 and 0.

  std::vector<float> cache_data(hidden_dim * cache_seq, 0.0f);
  // Slice layout is [seq, hidden] (seq-major).
  std::vector<float> slice_data = {
      1.0f, 2.0f, 3.0f, 4.0f,  // seq 0, h=0..3
      5.0f, 6.0f, 7.0f, 8.0f   // seq 1, h=0..3
  };
  std::vector<int32_t> pos_data = {start_pos};

  absl::flat_hash_map<absl::string_view, TensorBuffer> in_buffers;
  in_buffers.emplace("input_pos",
                     CreateTensorBuffer(pos_data, ElementType::Int32));
  in_buffers.emplace("kv_cache_k_0", CreateTensorBufferWithDims(
                                         cache_data, ElementType::Float32,
                                         {1, hidden_dim, cache_seq}));
  in_buffers.emplace("kv_cache_v_0", CreateTensorBufferWithDims(
                                         cache_data, ElementType::Float32,
                                         {1, hidden_dim, cache_seq}));
  in_buffers.emplace("kv_slice_k_0", CreateTensorBufferWithDims(
                                         slice_data, ElementType::Float32,
                                         {1, slice_seq, hidden_dim}));
  in_buffers.emplace("kv_slice_v_0", CreateTensorBufferWithDims(
                                         slice_data, ElementType::Float32,
                                         {1, slice_seq, hidden_dim}));

  absl::flat_hash_map<absl::string_view, TensorBuffer> out_buffers;

  ASSERT_TRUE(
      HWKVCacheUpdate(in_buffers, out_buffers, {}, /*enable_swa=*/true).ok());

  auto lock_expected = TensorBufferScopedLock::Create<float>(
      in_buffers.at("kv_cache_k_0"), TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(lock_expected.HasValue());
  auto& lock = *lock_expected;

  // Expected mapping:
  // cache[h][7] <- slice[0][h] (1..4)
  // cache[h][0] <- slice[1][h] (5..8)

  std::vector<int> expected_slice_seq_idx = {
      1,   // cache seq 0
      -1,  // cache seq 1
      -1,  // cache seq 2
      -1,  // cache seq 3
      -1,  // cache seq 4
      -1,  // cache seq 5
      -1,  // cache seq 6
      0    // cache seq 7
  };

  for (int h = 0; h < hidden_dim; ++h) {
    for (int s = 0; s < cache_seq; ++s) {
      int s_seq = expected_slice_seq_idx[s];
      float expected =
          (s_seq == -1) ? 0.0f : slice_data[s_seq * hidden_dim + h];
      EXPECT_EQ(lock.second[h * cache_seq + s], expected)
          << "Mismatch at head " << h << " seq " << s;
    }
  }
}

TEST_F(ExecutorUtilsTest, HWKVCacheUpdateSWAPrefillWithValidMask) {
  int hidden_dim = 2;
  int cache_seq = 8;
  int slice_seq = 8;

  // Initialize cache with distinct values
  std::vector<float> cache_data(hidden_dim * cache_seq);
  for (int i = 0; i < cache_seq; ++i) {
    for (int h = 0; h < hidden_dim; ++h) {
      cache_data[i * hidden_dim + h] = static_cast<float>(100 + i * 10 + h);
    }
  }

  // Slice data: 6 real tokens, 2 padding tokens (value 999.0)
  std::vector<float> slice_data = {
      1.0f,   2.0f,    // token 0 (real, maps to pos 104)
      3.0f,   4.0f,    // token 1 (real, maps to pos 105)
      5.0f,   6.0f,    // token 2 (real, maps to pos 106)
      7.0f,   8.0f,    // token 3 (real, maps to pos 107)
      9.0f,   10.0f,   // token 4 (real, maps to pos 108)
      11.0f,  12.0f,   // token 5 (real, maps to pos 109)
      999.0f, 999.0f,  // token 6 (padding)
      999.0f, 999.0f   // token 7 (padding)
  };

  std::vector<int32_t> pos_data = {100, 101, 102, 103, 104, 105,
                                   106, 107, 108, 109, 0,   0};

  std::vector<uint8_t> valid_mask_data = {true, true, true, true, true,  true,
                                          true, true, true, true, false, false};

  absl::flat_hash_map<absl::string_view, TensorBuffer> in_buffers;
  in_buffers.emplace("input_pos",
                     CreateTensorBuffer(pos_data, ElementType::Int32));
  in_buffers.emplace("valid_mask",
                     CreateTensorBuffer(valid_mask_data, ElementType::Bool));
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

  ASSERT_TRUE(
      HWKVCacheUpdate(in_buffers, out_buffers, {}, /*enable_swa=*/true).ok());

  auto lock_expected = TensorBufferScopedLock::Create<float>(
      in_buffers.at("kv_cache_k_0"), TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(lock_expected.HasValue());
  auto& lock = *lock_expected;

  // Expected cache content:
  // cache[0] <- slice[0] (1, 2)  -- maps to pos 104
  // cache[1] <- slice[1] (3, 4)  -- maps to pos 105
  // cache[2] <- slice[2] (5, 6)  -- maps to pos 106
  // cache[3] <- slice[3] (7, 8)  -- maps to pos 107
  // cache[4] <- slice[4] (9, 10) -- maps to pos 108
  // cache[5] <- slice[5] (11, 12)-- maps to pos 109
  // cache[6] <- original cache[6] (160, 161)
  // cache[7] <- original cache[7] (170, 171)

  for (int i = 0; i < 6; ++i) {
    EXPECT_EQ(lock.second[i * hidden_dim + 0], slice_data[i * hidden_dim + 0])
        << "Mismatch at seq " << i;
    EXPECT_EQ(lock.second[i * hidden_dim + 1], slice_data[i * hidden_dim + 1])
        << "Mismatch at seq " << i;
  }
  // Check that original values are preserved for indices 6 and 7
  for (int i = 6; i < 8; ++i) {
    EXPECT_EQ(lock.second[i * hidden_dim + 0], cache_data[i * hidden_dim + 0])
        << "Overwritten at seq " << i;
    EXPECT_EQ(lock.second[i * hidden_dim + 1], cache_data[i * hidden_dim + 1])
        << "Overwritten at seq " << i;
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
  ASSERT_TRUE(local_lock_expected.HasValue());
  auto global_lock_expected = TensorBufferScopedLock::Create<int8_t>(
      out_buffers.at("mask_global"), TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(global_lock_expected.HasValue());
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
  ASSERT_TRUE(local_lock_expected.HasValue());
  auto global_lock_expected = TensorBufferScopedLock::Create<int16_t>(
      out_buffers.at("mask_global"), TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(global_lock_expected.HasValue());
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
  ASSERT_TRUE(global_lock_expected.HasValue());
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
  ASSERT_TRUE(global_lock_expected.HasValue());
  auto& global_lock = *global_lock_expected;

  // Check KV cache part
  EXPECT_EQ(global_lock.second[100], valid_val);
  EXPECT_EQ(global_lock.second[600], -128);

  // Check new token part (at index 1280)
  EXPECT_EQ(global_lock.second[1280], valid_val);
}

TEST_F(ExecutorUtilsTest, HWMaskUpdateWithValidMask) {
  int seq_q = 128;
  int seq_k = 1408;
  int time_step = 0;

  absl::flat_hash_map<absl::string_view, TensorBuffer> in_buffers;
  std::vector<int32_t> time_step_data = {time_step};
  in_buffers.emplace("time_step",
                     CreateTensorBuffer(time_step_data, ElementType::Int32));

  // 50 valid tokens in valid_mask, rest are padding (false)
  std::vector<uint8_t> valid_mask_data(seq_q, false);
  for (int i = 0; i < 50; ++i) {
    valid_mask_data[i] = true;
  }
  in_buffers.emplace("valid_mask",
                     CreateTensorBuffer(valid_mask_data, ElementType::Bool));

  // We pass input_tokens with ALL valid (0) to ensure it uses valid_mask
  // instead
  std::vector<int32_t> input_tokens_data(seq_q, 0);
  in_buffers.emplace("input_tokens",
                     CreateTensorBuffer(input_tokens_data, ElementType::Int32));

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
  ASSERT_TRUE(global_lock_expected.HasValue());
  ASSERT_TRUE(global_lock_expected);
  auto& global_lock = *global_lock_expected;

  int8_t valid_val = 127;
  int8_t masked_val = -128;

  // Check row q = 100
  // k_rel = k - 1280.
  // We expect valid_val for k_rel < 50.
  // We expect masked_val for 50 <= k_rel <= 100.
  int64_t row_offset = 100 * seq_k;
  for (int k_rel = 0; k_rel < 50; ++k_rel) {
    EXPECT_EQ(global_lock.second[row_offset + 1280 + k_rel], valid_val)
        << "k_rel " << k_rel;
  }
  for (int k_rel = 50; k_rel <= 100; ++k_rel) {
    EXPECT_EQ(global_lock.second[row_offset + 1280 + k_rel], masked_val)
        << "k_rel " << k_rel;
  }
}

TEST_F(ExecutorUtilsTest, HWMaskUpdateSWADecode_TimestepSmallerLocalWindow) {
  int seq_q = 1;
  int seq_k_local = 513;    // capacity 512 + 1 batch
  int seq_k_global = 4097;  // capacity 4096 + 1 batch
  int time_step = 300;
  float valid_val = 0.0f;
  float masked_val = -1e9f;

  absl::flat_hash_map<absl::string_view, TensorBuffer> in_buffers;
  std::vector<int32_t> time_step_data = {time_step};
  in_buffers.emplace("time_step",
                     CreateTensorBuffer(time_step_data, ElementType::Int32));

  absl::flat_hash_map<absl::string_view, TensorBuffer> out_buffers;
  std::vector<float> mask_local_data(seq_q * seq_k_local, 0.0f);
  std::vector<float> mask_global_data(seq_q * seq_k_global, 0.0f);

  out_buffers.emplace("mask_local", CreateTensorBufferWithDims(
                                        mask_local_data, ElementType::Float32,
                                        {1, seq_q, seq_k_local}));
  out_buffers.emplace("mask_global", CreateTensorBufferWithDims(
                                         mask_global_data, ElementType::Float32,
                                         {1, seq_q, seq_k_global}));

  ASSERT_TRUE(HWMaskUpdate(in_buffers, out_buffers).ok());

  auto local_lock_expected = TensorBufferScopedLock::Create<float>(
      out_buffers.at("mask_local"), TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(local_lock_expected.HasValue());
  auto global_lock_expected = TensorBufferScopedLock::Create<float>(
      out_buffers.at("mask_global"), TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(global_lock_expected.HasValue());
  ASSERT_TRUE(local_lock_expected);
  ASSERT_TRUE(global_lock_expected);
  float* local_ptr = local_lock_expected->second;
  float* global_ptr = global_lock_expected->second;

  // Check local mask (SWA capacity 512, time_step 300)
  // No wrapping. Valid range in history: [0, 299].
  EXPECT_EQ(local_ptr[0], valid_val);
  EXPECT_EQ(local_ptr[299], valid_val);
  EXPECT_EQ(local_ptr[300], masked_val);
  EXPECT_EQ(local_ptr[511], masked_val);
  // Batch part (index 512)
  EXPECT_EQ(local_ptr[512], valid_val);

  // Check global mask (capacity 4096, time_step 300)
  // No wrapping. Valid range in history: [0, 299].
  EXPECT_EQ(global_ptr[0], valid_val);
  EXPECT_EQ(global_ptr[299], valid_val);
  EXPECT_EQ(global_ptr[300], masked_val);
  EXPECT_EQ(global_ptr[4095], masked_val);
  // Batch part (index 4096)
  EXPECT_EQ(global_ptr[4096], valid_val);
}

TEST_F(ExecutorUtilsTest, HWMaskUpdateSWADecode_TimestepLargerLocalWindow) {
  int seq_q = 1;
  int seq_k_local = 513;    // capacity 512 + 1 batch
  int seq_k_global = 4097;  // capacity 4096 + 1 batch
  int time_step = 1000;
  float valid_val = 0.0f;
  float masked_val = -1e9f;

  absl::flat_hash_map<absl::string_view, TensorBuffer> in_buffers;
  std::vector<int32_t> time_step_data = {time_step};
  in_buffers.emplace("time_step",
                     CreateTensorBuffer(time_step_data, ElementType::Int32));

  absl::flat_hash_map<absl::string_view, TensorBuffer> out_buffers;
  std::vector<float> mask_local_data(seq_q * seq_k_local, 0.0f);
  std::vector<float> mask_global_data(seq_q * seq_k_global, 0.0f);

  out_buffers.emplace(
      "mask_local",
      CreateTensorBufferWithDims(mask_local_data, ElementType::Float32,
                                 {1, seq_q, seq_k_local}));
  out_buffers.emplace(
      "mask_global",
      CreateTensorBufferWithDims(mask_global_data, ElementType::Float32,
                                 {1, seq_q, seq_k_global}));

  ASSERT_TRUE(HWMaskUpdate(in_buffers, out_buffers).ok());

  auto local_lock_expected = TensorBufferScopedLock::Create<float>(
      out_buffers.at("mask_local"), TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(local_lock_expected.HasValue());
  auto global_lock_expected = TensorBufferScopedLock::Create<float>(
      out_buffers.at("mask_global"), TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(global_lock_expected.HasValue());
  ASSERT_TRUE(local_lock_expected);
  ASSERT_TRUE(global_lock_expected);
  float* local_ptr = local_lock_expected->second;
  float* global_ptr = global_lock_expected->second;

  // Check local mask (SWA capacity 512, time_step 1000)
  // Wrapped. Valid range in history: [489, 999] (relative to time_step).
  // t_k = 999 - ((999 - k) % 512)
  // Index 488: t_k = 488 (invalid)
  // Index 489: t_k = 489 (valid)
  // Index 0: t_k = 512 (valid)
  // Index 487: t_k = 999 (valid)
  EXPECT_EQ(local_ptr[488], masked_val);
  EXPECT_EQ(local_ptr[489], valid_val);
  EXPECT_EQ(local_ptr[0], valid_val);
  EXPECT_EQ(local_ptr[487], valid_val);
  // Batch part (index 512)
  EXPECT_EQ(local_ptr[512], valid_val);

  // Check global mask (capacity 4096, time_step 1000)
  // Not wrapped. Valid range: 0..999.
  EXPECT_EQ(global_ptr[0], valid_val);
  EXPECT_EQ(global_ptr[999], valid_val);
  EXPECT_EQ(global_ptr[1000], masked_val);
  EXPECT_EQ(global_ptr[4095], masked_val);
  // Batch part (index 4096)
  EXPECT_EQ(global_ptr[4096], valid_val);
}

TEST_F(ExecutorUtilsTest, HWMaskUpdateCapacity4096Uses512AsWindow) {
  int seq_q = 1;
  int seq_k_local = 4097;   // capacity 4096 + 1 batch
  int seq_k_global = 4097;  // capacity 4096 + 1 batch
  int time_step = 1000;
  float valid_val = 0.0f;
  float masked_val = -1e9f;

  absl::flat_hash_map<absl::string_view, TensorBuffer> in_buffers;
  std::vector<int32_t> time_step_data = {time_step};
  in_buffers.emplace("time_step",
                     CreateTensorBuffer(time_step_data, ElementType::Int32));

  absl::flat_hash_map<absl::string_view, TensorBuffer> out_buffers;
  std::vector<float> mask_local_data(seq_q * seq_k_local, 0.0f);
  std::vector<float> mask_global_data(seq_q * seq_k_global, 0.0f);

  out_buffers.emplace("mask_local", CreateTensorBufferWithDims(
                                        mask_local_data, ElementType::Float32,
                                        {1, seq_q, seq_k_local}));
  out_buffers.emplace("mask_global", CreateTensorBufferWithDims(
                                         mask_global_data, ElementType::Float32,
                                         {1, seq_q, seq_k_global}));

  ASSERT_TRUE(HWMaskUpdate(in_buffers, out_buffers).ok());

  auto local_lock_expected = TensorBufferScopedLock::Create<float>(
      out_buffers.at("mask_local"), TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(local_lock_expected.HasValue());
  ASSERT_TRUE(local_lock_expected);
  float* local_ptr = local_lock_expected->second;

  // Check local mask (SWA capacity 4096, time_step 1000)
  // Uses 512 as window size.
  // Valid range in history: [489, 999].
  EXPECT_EQ(local_ptr[0], masked_val);
  EXPECT_EQ(local_ptr[488], masked_val);
  EXPECT_EQ(local_ptr[489], valid_val);
  EXPECT_EQ(local_ptr[999], valid_val);
  EXPECT_EQ(local_ptr[1000], masked_val);
}

TEST_F(ExecutorUtilsTest, HWMaskUpdateSWAPrefill) {
  int seq_q = 128;
  int seq_k_local = 640;    // capacity 512 + 128 batch
  int seq_k_global = 4224;  // capacity 4096 + 128 batch
  int time_step = 0;
  float valid_val = 0.0f;
  float masked_val = -1e9f;

  absl::flat_hash_map<absl::string_view, TensorBuffer> in_buffers;
  std::vector<int32_t> time_step_data = {time_step};
  in_buffers.emplace("time_step",
                     CreateTensorBuffer(time_step_data, ElementType::Int32));

  absl::flat_hash_map<absl::string_view, TensorBuffer> out_buffers;
  std::vector<float> mask_local_data(seq_q * seq_k_local, 0.0f);
  std::vector<float> mask_global_data(seq_q * seq_k_global, 0.0f);

  out_buffers.emplace(
      "mask_local",
      CreateTensorBufferWithDims(mask_local_data, ElementType::Float32,
                                 {1, seq_q, seq_k_local}));
  out_buffers.emplace(
      "mask_global",
      CreateTensorBufferWithDims(mask_global_data, ElementType::Float32,
                                 {1, seq_q, seq_k_global}));

  ASSERT_TRUE(HWMaskUpdate(in_buffers, out_buffers).ok());

  auto local_lock_expected = TensorBufferScopedLock::Create<float>(
      out_buffers.at("mask_local"), TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(local_lock_expected.HasValue());
  auto global_lock_expected = TensorBufferScopedLock::Create<float>(
      out_buffers.at("mask_global"), TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(global_lock_expected.HasValue());
  ASSERT_TRUE(local_lock_expected);
  ASSERT_TRUE(global_lock_expected);
  float* local_ptr = local_lock_expected->second;
  float* global_ptr = global_lock_expected->second;

  // For row q = 10:
  // History is all masked.
  // Batch part: k_rel in 0..127.
  // Valid if k_rel <= q (0..10).
  int q = 10;
  int64_t row_offset_local = q * seq_k_local;
  int64_t row_offset_global = q * seq_k_global;

  // Local history (0..511)
  EXPECT_EQ(local_ptr[row_offset_local + 0], masked_val);
  EXPECT_EQ(local_ptr[row_offset_local + 511], masked_val);
  // Local batch (512..639)
  // k_rel = 0 <= 10
  EXPECT_EQ(local_ptr[row_offset_local + 512 + 0], valid_val);
  // k_rel = 10 <= 10
  EXPECT_EQ(local_ptr[row_offset_local + 512 + 10], valid_val);
  // k_rel = 11 > 10
  EXPECT_EQ(local_ptr[row_offset_local + 512 + 11], masked_val);

  // Global history (0..4095)
  EXPECT_EQ(global_ptr[row_offset_global + 0], masked_val);
  EXPECT_EQ(global_ptr[row_offset_global + 4095], masked_val);
  // Global batch (4096..4223)
  // k_rel = 0 <= 10
  EXPECT_EQ(global_ptr[row_offset_global + 4096 + 0], valid_val);
  // k_rel = 10 <= 10
  EXPECT_EQ(global_ptr[row_offset_global + 4096 + 10], valid_val);
  // k_rel = 11 > 10
  EXPECT_EQ(global_ptr[row_offset_global + 4096 + 11], masked_val);
}

TEST_F(ExecutorUtilsTest, HWMaskUpdateSWAPrefill_TimestepSmallerLocalWindow) {
  int seq_q = 128;
  int seq_k_local = 640;    // capacity 512 + 128 batch
  int seq_k_global = 4224;  // capacity 4096 + 128 batch
  int time_step = 300;
  float valid_val = 0.0f;
  float masked_val = -1e9f;

  absl::flat_hash_map<absl::string_view, TensorBuffer> in_buffers;
  std::vector<int32_t> time_step_data = {time_step};
  in_buffers.emplace("time_step",
                     CreateTensorBuffer(time_step_data, ElementType::Int32));

  absl::flat_hash_map<absl::string_view, TensorBuffer> out_buffers;
  std::vector<float> mask_local_data(seq_q * seq_k_local, 0.0f);
  std::vector<float> mask_global_data(seq_q * seq_k_global, 0.0f);

  out_buffers.emplace("mask_local", CreateTensorBufferWithDims(
                                        mask_local_data, ElementType::Float32,
                                        {1, seq_q, seq_k_local}));
  out_buffers.emplace("mask_global", CreateTensorBufferWithDims(
                                         mask_global_data, ElementType::Float32,
                                         {1, seq_q, seq_k_global}));

  ASSERT_TRUE(HWMaskUpdate(in_buffers, out_buffers).ok());

  auto local_lock_expected = TensorBufferScopedLock::Create<float>(
      out_buffers.at("mask_local"), TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(local_lock_expected.HasValue());
  auto global_lock_expected = TensorBufferScopedLock::Create<float>(
      out_buffers.at("mask_global"), TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(global_lock_expected.HasValue());
  ASSERT_TRUE(local_lock_expected);
  ASSERT_TRUE(global_lock_expected);
  float* local_ptr = local_lock_expected->second;
  float* global_ptr = global_lock_expected->second;

  // For row q = 10:
  // History valid range: [0, 299].
  // Batch part: k_rel in 0..127. Valid if k_rel <= q (0..10).
  int q = 10;
  int64_t row_offset_local = q * seq_k_local;
  int64_t row_offset_global = q * seq_k_global;

  // Local history (0..511)
  // First 300 tokens are valid historical tokens.
  EXPECT_EQ(local_ptr[row_offset_local + 0], valid_val);
  EXPECT_EQ(local_ptr[row_offset_local + 299], valid_val);
  // Remaining slots in the history capacity (300 to 511) are empty/masked.
  EXPECT_EQ(local_ptr[row_offset_local + 300], masked_val);
  EXPECT_EQ(local_ptr[row_offset_local + 511], masked_val);
  // Local batch (512..639)
  // 512 is the offset where the current prefill batch starts in the KV cache.
  // We check causality for query q = 10 (which is global token 310).
  // Batch token 0 (global 300) is before query, so it's valid.
  EXPECT_EQ(local_ptr[row_offset_local + 512 + 0], valid_val);
  // Batch token 10 (global 310, itself) is valid.
  EXPECT_EQ(local_ptr[row_offset_local + 512 + 10], valid_val);
  // Batch token 11 (global 311) is in the future relative to q=10, so it's
  // masked.
  EXPECT_EQ(local_ptr[row_offset_local + 512 + 11], masked_val);

  // Global history (0..4095)
  // First 300 tokens are valid historical tokens.
  EXPECT_EQ(global_ptr[row_offset_global + 0], valid_val);
  EXPECT_EQ(global_ptr[row_offset_global + 299], valid_val);
  // Remaining slots in global capacity (300 to 4095) are empty/masked.
  EXPECT_EQ(global_ptr[row_offset_global + 300], masked_val);
  EXPECT_EQ(global_ptr[row_offset_global + 4095], masked_val);
  // Global batch (4096..4223)
  // 4096 is the offset where the current prefill batch starts in the global
  // KV cache.
  EXPECT_EQ(global_ptr[row_offset_global + 4096 + 0], valid_val);
  EXPECT_EQ(global_ptr[row_offset_global + 4096 + 10], valid_val);
  EXPECT_EQ(global_ptr[row_offset_global + 4096 + 11], masked_val);
}

TEST_F(ExecutorUtilsTest, HWMaskUpdateSWAPrefill_TimestepLargerLocalWindow) {
  int seq_q = 128;
  int seq_k_local = 640;    // capacity 512 + 128 batch
  int seq_k_global = 4224;  // capacity 4096 + 128 batch
  int time_step = 1000;
  float valid_val = 0.0f;
  float masked_val = -1e9f;

  absl::flat_hash_map<absl::string_view, TensorBuffer> in_buffers;
  std::vector<int32_t> time_step_data = {time_step};
  in_buffers.emplace("time_step",
                     CreateTensorBuffer(time_step_data, ElementType::Int32));

  absl::flat_hash_map<absl::string_view, TensorBuffer> out_buffers;
  std::vector<float> mask_local_data(seq_q * seq_k_local, 0.0f);
  std::vector<float> mask_global_data(seq_q * seq_k_global, 0.0f);

  out_buffers.emplace("mask_local", CreateTensorBufferWithDims(
                                        mask_local_data, ElementType::Float32,
                                        {1, seq_q, seq_k_local}));
  out_buffers.emplace("mask_global", CreateTensorBufferWithDims(
                                         mask_global_data, ElementType::Float32,
                                         {1, seq_q, seq_k_global}));

  ASSERT_TRUE(HWMaskUpdate(in_buffers, out_buffers).ok());

  auto local_lock_expected = TensorBufferScopedLock::Create<float>(
      out_buffers.at("mask_local"), TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(local_lock_expected.HasValue());
  auto global_lock_expected = TensorBufferScopedLock::Create<float>(
      out_buffers.at("mask_global"), TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(global_lock_expected.HasValue());
  ASSERT_TRUE(local_lock_expected);
  ASSERT_TRUE(global_lock_expected);
  float* local_ptr = local_lock_expected->second;
  float* global_ptr = global_lock_expected->second;

  // For row q = 10:
  // Valid range in history: [499, 999] (logical).
  // Corresponding physical indices: [0, 487] and [499, 511].
  // Masked physical indices: [488, 498].
  int q = 10;
  int64_t row_offset_local = q * seq_k_local;
  int64_t row_offset_global = q * seq_k_global;

  // Local history (0..511)
  // SWA window is 512. For q = 10 (global 1010), valid history is [499, 999].
  // SWA circular buffer wrapped. Token 999 is at slot 487. Token 499 is at
  // slot 499.
  // Token 488 (slot 488) is out of window, so it's masked.
  EXPECT_EQ(local_ptr[row_offset_local + 488], masked_val);
  // Token 498 (slot 498) is out of window, so it's masked.
  EXPECT_EQ(local_ptr[row_offset_local + 498], masked_val);
  // Token 499 (slot 499) is the oldest valid token in window.
  EXPECT_EQ(local_ptr[row_offset_local + 499], valid_val);
  // Token 999 (slot 487) is the newest historical token, valid.
  EXPECT_EQ(local_ptr[row_offset_local + 487], valid_val);
  // Token 512 (slot 0) is valid.
  EXPECT_EQ(local_ptr[row_offset_local + 0], valid_val);
  // Local batch (512..639)
  // 512 is the offset where the current prefill batch starts in the KV cache.
  // We check causality for query q = 10 (which is global token 1010).
  // Batch token 0 (global 1000) is before query, so it's valid.
  EXPECT_EQ(local_ptr[row_offset_local + 512 + 0], valid_val);
  // Batch token 10 (global 1010, itself) is valid.
  EXPECT_EQ(local_ptr[row_offset_local + 512 + 10], valid_val);
  // Batch token 11 (global 1011) is in the future relative to q=10, so it's
  // masked.
  EXPECT_EQ(local_ptr[row_offset_local + 512 + 11], masked_val);

  // Check global mask (capacity 4096, time_step 1000)
  // Not wrapped. Valid range: 0..999.
  EXPECT_EQ(global_ptr[row_offset_global + 0], valid_val);
  EXPECT_EQ(global_ptr[row_offset_global + 999], valid_val);
  EXPECT_EQ(global_ptr[row_offset_global + 1000], masked_val);
  EXPECT_EQ(global_ptr[row_offset_global + 4095], masked_val);
  // Global batch (4096..4223)
  // 4096 is the offset where the current prefill batch starts in the global
  // KV cache.
  EXPECT_EQ(global_ptr[row_offset_global + 4096 + 0], valid_val);
  // Batch token 10 (global 1010, itself) is valid.
  EXPECT_EQ(global_ptr[row_offset_global + 4096 + 10], valid_val);
  // Batch token 11 (global 1011) is in the future relative to q=10, so it's
  // masked.
  EXPECT_EQ(global_ptr[row_offset_global + 4096 + 11], masked_val);
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
      output.data(), litert::ElementType::Float32, litert::ElementType::Int4);

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
      output.data(), litert::ElementType::Int16, litert::ElementType::Int4,
      1.0f, final_scale, final_zero_point);

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
      output.data(), litert::ElementType::Float32, litert::ElementType::Int4);

  ASSERT_TRUE(status.ok());

  for (size_t i = 0; i < output.size(); ++i) {
    EXPECT_NEAR(output[i], static_cast<float>(table0_unpacked[i]), 1e-5)
        << "Index " << i;
  }
}

TEST_F(ExecutorUtilsTest, HWPerLayerEmbeddingLookupInt8Float32WithScale) {
  constexpr int kNumTables = 2;
  constexpr int kColSize = 4;

  // Raw Int8 table data (no packing needed).
  std::vector<int8_t> table0 = {0, 1, 2, 3, -1, -2, -3, -4,
                                4, 5, 6, 7, -5, -6, -7, -8};
  std::vector<int8_t> table1 = {0,  -1, -2, -3, 4, 5, 6, 7,
                                -4, -5, -6, -7, 1, 2, 3, -8};

  std::vector<const uint8_t*> table_ptrs = {
      reinterpret_cast<const uint8_t*>(table0.data()),
      reinterpret_cast<const uint8_t*>(table1.data())};

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

  // Apply a final scaling factor of 16.0f (mimicking Gemma's sqrt(d_model)).
  float final_scale = 16.0f;

  auto status = HWPerLayerEmbeddingLookup(
      token_ids.data(), num_tokens, table_ptrs.data(), qp, kNumTables, kColSize,
      output.data(), litert::ElementType::Float32, litert::ElementType::Int8,
      final_scale);

  ASSERT_TRUE(status.ok());

  // Expected output:
  // For Token 1 (t=0):
  //   Table 0: row=[-1, -2, -3, -4], scale=scales0[1]=2.0, final_scale=16.0 ->
  //   val * 32.0
  //            expected = [-32.0, -64.0, -96.0, -128.0]
  //   Table 1: row=[4, 5, 6, 7], scale=scales1[0]=0.5, final_scale=16.0 -> val
  //   * 8.0
  //            expected = [32.0, 40.0, 48.0, 56.0]
  // For Token 2 (t=1):
  //   Table 0: row=[4, 5, 6, 7], scale=scales0[2]=0.5, final_scale=16.0 -> val
  //   * 8.0
  //            expected = [32.0, 40.0, 48.0, 56.0]
  //   Table 1: row=[-4, -5, -6, -7], scale=scales1[0]=0.5, final_scale=16.0 ->
  //   val * 8.0
  //            expected = [-32.0, -40.0, -48.0, -56.0]
  std::vector<float> expected_output = {
      -32.0f, -64.0f, -96.0f, -128.0f, 32.0f,  40.0f,  48.0f,  56.0f,
      32.0f,  40.0f,  48.0f,  56.0f,   -32.0f, -40.0f, -48.0f, -56.0f};

  for (size_t i = 0; i < output.size(); ++i) {
    EXPECT_NEAR(output[i], expected_output[i], 1e-5) << "Index " << i;
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
  ASSERT_TRUE(lock_expected.HasValue());
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
    ASSERT_TRUE(lock_expected.HasValue());
    auto& lock = *lock_expected;
    for (int i = 0; i < (int)slice_data.size(); ++i) {
      EXPECT_EQ(lock.second[i], 2.0f);
    }
  }

  // Check out_buffer
  {
    auto lock_expected = TensorBufferScopedLock::Create<float>(
        out_buffers.at("kv_cache_c_1"), TensorBuffer::LockMode::kRead);
    ASSERT_TRUE(lock_expected.HasValue());
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
  ASSERT_TRUE(global_lock_expected.HasValue());
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
  ASSERT_TRUE(global_lock_expected.HasValue());
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

TEST_F(ExecutorUtilsTest, WritePleEmbeddingsFloat32) {
  std::vector<float> ple_embeddings = {1.0f, 2.0f, 3.0f, 4.0f};
  TensorBuffer buffer =
      CreateTensorBuffer(std::vector<float>(4, 0.0f), ElementType::Float32);

  ASSERT_TRUE(
      WritePleEmbeddings(buffer, ple_embeddings, ElementType::Float32, 1.0f, 0)
          .ok());

  auto lock_expected = TensorBufferScopedLock::Create<float>(
      buffer, TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(lock_expected.HasValue());
  auto& lock = *lock_expected;

  for (size_t i = 0; i < ple_embeddings.size(); ++i) {
    EXPECT_EQ(lock.second[i], ple_embeddings[i]);
  }
}

TEST_F(ExecutorUtilsTest, WritePleEmbeddingsInt16) {
  std::vector<float> ple_embeddings = {1.0f, -2.0f, 3.5f, -4.5f};
  float scale = 0.5f;
  int32_t zero_point = 10;
  TensorBuffer buffer =
      CreateTensorBuffer(std::vector<int16_t>(4, 0), ElementType::Int16);

  ASSERT_TRUE(WritePleEmbeddings(buffer, ple_embeddings, ElementType::Int16,
                                 scale, zero_point)
                  .ok());

  auto lock_expected = TensorBufferScopedLock::Create<int16_t>(
      buffer, TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(lock_expected.HasValue());
  auto& lock = *lock_expected;

  for (size_t i = 0; i < ple_embeddings.size(); ++i) {
    int16_t expected = Quantize<int16_t>(ple_embeddings[i], scale, zero_point);
    EXPECT_EQ(lock.second[i], expected);
  }
}

TEST_F(ExecutorUtilsTest, WritePleEmbeddingsInt16InsufficientCapacity) {
  std::vector<float> ple_embeddings = {1.0f, -2.0f, 3.5f, -4.5f};
  float scale = 0.5f;
  int32_t zero_point = 10;
  // Buffer size 3 instead of 4
  TensorBuffer buffer =
      CreateTensorBuffer(std::vector<int16_t>(3, 0), ElementType::Int16);

  EXPECT_FALSE(WritePleEmbeddings(buffer, ple_embeddings, ElementType::Int16,
                                  scale, zero_point)
                   .ok());
}

TEST_F(ExecutorUtilsTest, WriteAndPadPleEmbeddingsFloat32) {
  std::vector<float> ple_embeddings = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  size_t ple_dim = 3;
  size_t seq_pos_size = 2;
  std::vector<float> default_ple_emb = {0.1f, 0.2f, 0.3f};

  TensorBuffer buffer = CreateTensorBufferWithDims(
      std::vector<float>(4 * ple_dim, 0.0f), ElementType::Float32,
      {1, 4, (int32_t)ple_dim});

  ASSERT_TRUE(WriteAndPadPleEmbeddings(buffer, ple_embeddings, ple_dim,
                                       seq_pos_size, default_ple_emb,
                                       ElementType::Float32, 1.0f, 0)
                  .ok());

  auto lock_expected = TensorBufferScopedLock::Create<float>(
      buffer, TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(lock_expected.HasValue());
  auto& lock = *lock_expected;

  for (size_t i = 0; i < ple_embeddings.size(); ++i) {
    EXPECT_EQ(lock.second[i], ple_embeddings[i]);
  }

  for (size_t t = seq_pos_size; t < 4; ++t) {
    for (size_t d = 0; d < ple_dim; ++d) {
      EXPECT_EQ(lock.second[t * ple_dim + d], default_ple_emb[d]);
    }
  }
}

TEST_F(ExecutorUtilsTest, WriteAndPadPleEmbeddingsInt16) {
  std::vector<float> ple_embeddings = {1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f};
  size_t ple_dim = 3;
  size_t seq_pos_size = 2;
  std::vector<float> default_ple_emb = {0.5f, -0.5f, 1.5f};
  float scale = 0.5f;
  int32_t zero_point = 10;

  TensorBuffer buffer =
      CreateTensorBufferWithDims(std::vector<int16_t>(4 * ple_dim, 0),
                                 ElementType::Int16, {1, 4, (int32_t)ple_dim});

  ASSERT_TRUE(WriteAndPadPleEmbeddings(buffer, ple_embeddings, ple_dim,
                                       seq_pos_size, default_ple_emb,
                                       ElementType::Int16, scale, zero_point)
                  .ok());

  auto lock_expected = TensorBufferScopedLock::Create<int16_t>(
      buffer, TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(lock_expected.HasValue());
  auto& lock = *lock_expected;

  for (size_t i = 0; i < ple_embeddings.size(); ++i) {
    int16_t expected = Quantize<int16_t>(ple_embeddings[i], scale, zero_point);
    EXPECT_EQ(lock.second[i], expected);
  }

  for (size_t t = seq_pos_size; t < 4; ++t) {
    for (size_t d = 0; d < ple_dim; ++d) {
      int16_t expected =
          Quantize<int16_t>(default_ple_emb[d], scale, zero_point);
      EXPECT_EQ(lock.second[t * ple_dim + d], expected);
    }
  }
}

TEST_F(ExecutorUtilsTest, WriteAndPadPleEmbeddingsFloat32NoDefault) {
  std::vector<float> ple_embeddings = {1.0f, 2.0f, 3.0f, -4.0f, 5.0f, 6.0f};
  size_t ple_dim = 3;
  size_t seq_pos_size = 2;
  std::vector<float> default_ple_emb = {};

  TensorBuffer buffer = CreateTensorBufferWithDims(
      std::vector<float>(4 * ple_dim, -1.0f), ElementType::Float32,
      {1, 4, (int32_t)ple_dim});

  ASSERT_TRUE(WriteAndPadPleEmbeddings(buffer, ple_embeddings, ple_dim,
                                       seq_pos_size, default_ple_emb,
                                       ElementType::Float32, 1.0f, 0)
                  .ok());

  auto lock_expected = TensorBufferScopedLock::Create<float>(
      buffer, TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(lock_expected.HasValue());
  auto& lock = *lock_expected;

  for (size_t i = 0; i < ple_embeddings.size(); ++i) {
    EXPECT_EQ(lock.second[i], ple_embeddings[i]);
  }

  for (size_t t = seq_pos_size; t < 4; ++t) {
    for (size_t d = 0; d < ple_dim; ++d) {
      EXPECT_EQ(lock.second[t * ple_dim + d], 0.0f);
    }
  }
}

TEST_F(ExecutorUtilsTest, WriteAndPadPleEmbeddingsInt16NoDefault) {
  std::vector<float> ple_embeddings = {1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f};
  size_t ple_dim = 3;
  size_t seq_pos_size = 2;
  std::vector<float> default_ple_emb = {};
  float scale = 0.5f;
  int32_t zero_point = 10;

  TensorBuffer buffer =
      CreateTensorBufferWithDims(std::vector<int16_t>(4 * ple_dim, -1),
                                 ElementType::Int16, {1, 4, (int32_t)ple_dim});

  ASSERT_TRUE(WriteAndPadPleEmbeddings(buffer, ple_embeddings, ple_dim,
                                       seq_pos_size, default_ple_emb,
                                       ElementType::Int16, scale, zero_point)
                  .ok());

  auto lock_expected = TensorBufferScopedLock::Create<int16_t>(
      buffer, TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(lock_expected.HasValue());
  auto& lock = *lock_expected;

  for (size_t i = 0; i < ple_embeddings.size(); ++i) {
    int16_t expected = Quantize<int16_t>(ple_embeddings[i], scale, zero_point);
    EXPECT_EQ(lock.second[i], expected);
  }

  int16_t expected_padding = Quantize<int16_t>(0.0f, scale, zero_point);
  for (size_t t = seq_pos_size; t < 4; ++t) {
    for (size_t d = 0; d < ple_dim; ++d) {
      EXPECT_EQ(lock.second[t * ple_dim + d], expected_padding);
    }
  }
}

TEST_F(ExecutorUtilsTest, WriteAndPadPleEmbeddingsInt16SizeMismatch) {
  std::vector<float> ple_embeddings = {1.0f, -2.0f, 3.0f, -4.0f, 5.0f};
  size_t ple_dim = 3;
  size_t seq_pos_size = 2;
  std::vector<float> default_ple_emb = {0.5f, -0.5f, 1.5f};
  float scale = 0.5f;
  int32_t zero_point = 10;

  TensorBuffer buffer =
      CreateTensorBufferWithDims(std::vector<int16_t>(4 * ple_dim, 0),
                                 ElementType::Int16, {1, 4, (int32_t)ple_dim});

  EXPECT_FALSE(WriteAndPadPleEmbeddings(buffer, ple_embeddings, ple_dim,
                                        seq_pos_size, default_ple_emb,
                                        ElementType::Int16, scale, zero_point)
                   .ok());
}

TEST_F(ExecutorUtilsTest, WriteAndPadPleEmbeddingsInt16InsufficientCapacity) {
  std::vector<float> ple_embeddings = {1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f};
  size_t ple_dim = 3;
  size_t seq_pos_size = 2;
  std::vector<float> default_ple_emb = {0.5f, -0.5f, 1.5f};
  float scale = 0.5f;
  int32_t zero_point = 10;

  // Buffer only has capacity for 1 token, but seq_pos_size is 2.
  TensorBuffer buffer =
      CreateTensorBufferWithDims(std::vector<int16_t>(1 * ple_dim, 0),
                                 ElementType::Int16, {1, 1, (int32_t)ple_dim});

  EXPECT_FALSE(WriteAndPadPleEmbeddings(buffer, ple_embeddings, ple_dim,
                                        seq_pos_size, default_ple_emb,
                                        ElementType::Int16, scale, zero_point)
                   .ok());
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

TEST_F(ExecutorUtilsTest,
       NpuMaskCreateFailsWhenCompiledModelNullForModelMethod) {
  InferenceContext ctx;
  auto mask_or =
      NpuMask::CreateForTest(MaskUpdateMethod::kModel, nullptr, std::move(ctx));
  EXPECT_FALSE(mask_or.ok());
}

TEST_F(ExecutorUtilsTest,
       NpuMaskCreateSucceedsForHwMethodWithoutCompiledModel) {
  InferenceContext ctx;
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto mask,
      NpuMask::CreateForTest(MaskUpdateMethod::kWH, nullptr, std::move(ctx)));
  EXPECT_EQ(mask.GetMethod(), MaskUpdateMethod::kWH);
}

TEST_F(ExecutorUtilsTest, NpuMaskSettersAndAccessors) {
  InferenceContext ctx;
  ctx.decode_input_buffers[MaskSignatures::kMaskInputTimeStep] =
      CreateTensorBufferWithDims(std::vector<int32_t>{0}, ElementType::Int32,
                                 {1});
  ctx.decode_input_buffers[MaskSignatures::kMaskInputTokens] =
      CreateTensorBufferWithDims(std::vector<int32_t>{0}, ElementType::Int32,
                                 {1});
  ctx.decode_input_buffers[MaskSignatures::kMaskInputValidMask] =
      CreateTensorBufferWithDims(std::vector<uint8_t>{0}, ElementType::Bool,
                                 {1});

  ctx.verify_input_buffers[MaskSignatures::kMaskInputTimeStep] =
      CreateTensorBufferWithDims(std::vector<int32_t>{0}, ElementType::Int32,
                                 {1});
  ctx.verify_input_buffers[MaskSignatures::kMaskInputTokens] =
      CreateTensorBufferWithDims(std::vector<int32_t>{0, 0, 0},
                                 ElementType::Int32, {1, 3});

  ctx.prefill_input_buffers[MaskSignatures::kMaskInputTimeStep] =
      CreateTensorBufferWithDims(std::vector<int32_t>{0}, ElementType::Int32,
                                 {1});

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto mask,
      NpuMask::CreateForTest(MaskUpdateMethod::kWH, nullptr, std::move(ctx)));

  // Test SetDecodeInput.
  EXPECT_TRUE(mask.SetDecodeInput(42, 101).ok());
  auto time_step_lock = TensorBufferScopedLock::Create<int32_t>(
      mask.Context().decode_input_buffers.at(
          MaskSignatures::kMaskInputTimeStep),
      TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(time_step_lock.HasValue());
  EXPECT_EQ(time_step_lock->second[0], 42);

  auto tokens_lock = TensorBufferScopedLock::Create<int32_t>(
      mask.Context().decode_input_buffers.at(MaskSignatures::kMaskInputTokens),
      TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(tokens_lock.HasValue());
  EXPECT_EQ(tokens_lock->second[0], 101);

  // Test SetVerifyInput.
  std::vector<int> verify_tokens = {10, 20, 30};
  EXPECT_TRUE(mask.SetVerifyInput(50, verify_tokens).ok());
  auto verify_step_lock = TensorBufferScopedLock::Create<int32_t>(
      mask.Context().verify_input_buffers.at(
          MaskSignatures::kMaskInputTimeStep),
      TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(verify_step_lock.HasValue());
  EXPECT_EQ(verify_step_lock->second[0], 50);

  auto verify_tokens_lock = TensorBufferScopedLock::Create<int32_t>(
      mask.Context().verify_input_buffers.at(MaskSignatures::kMaskInputTokens),
      TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(verify_tokens_lock.HasValue());
  EXPECT_EQ(verify_tokens_lock->second[0], 10);
  EXPECT_EQ(verify_tokens_lock->second[1], 20);
  EXPECT_EQ(verify_tokens_lock->second[2], 30);
}

TEST_F(ExecutorUtilsTest, NpuMaskCreateForDrafter) {
  absl::flat_hash_map<absl::string_view, TensorBuffer> in_bufs;
  absl::flat_hash_map<absl::string_view, TensorBuffer> out_bufs;
  in_bufs[MaskSignatures::kMaskInputTimeStep] = CreateTensorBufferWithDims(
      std::vector<int32_t>{0}, ElementType::Int32, {1});
  in_bufs[MaskSignatures::kMaskInputTokens] = CreateTensorBufferWithDims(
      std::vector<int32_t>{0}, ElementType::Int32, {1});

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto mask,
      NpuMask::CreateForDrafter(MaskUpdateMethod::kWH, nullptr,
                                std::move(in_bufs), std::move(out_bufs)));
  EXPECT_TRUE(mask.SetDecodeInput(77, 88).ok());

  auto step_lock = TensorBufferScopedLock::Create<int32_t>(
      mask.Context().decode_input_buffers.at(
          MaskSignatures::kMaskInputTimeStep),
      TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(step_lock.HasValue());
  EXPECT_EQ(step_lock->second[0], 77);

  auto token_lock = TensorBufferScopedLock::Create<int32_t>(
      mask.Context().decode_input_buffers.at(MaskSignatures::kMaskInputTokens),
      TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(token_lock.HasValue());
  EXPECT_EQ(token_lock->second[0], 88);
}

TEST_F(ExecutorUtilsTest, NpuMaskSetPrefillInput) {
  InferenceContext ctx;
  ctx.prefill_input_buffers[MaskSignatures::kMaskInputTimeStep] =
      CreateTensorBufferWithDims(std::vector<int32_t>{0}, ElementType::Int32,
                                 {1});
  ctx.prefill_input_buffers[MaskSignatures::kMaskInputTokens] =
      CreateTensorBufferWithDims(std::vector<int32_t>{0, 0, 0, 0, 0},
                                 ElementType::Int32, {1, 5});
  ctx.prefill_input_buffers[MaskSignatures::kMaskInputValidMask] =
      CreateTensorBufferWithDims(std::vector<uint8_t>{0, 0, 0, 0, 0},
                                 ElementType::Bool, {1, 5});

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto mask,
      NpuMask::CreateForTest(MaskUpdateMethod::kWH, nullptr, std::move(ctx)));

  std::vector<int> tokens = {101, -5, 202};
  EXPECT_TRUE(mask.SetPrefillInput(15, tokens).ok());

  auto step_lock = TensorBufferScopedLock::Create<int32_t>(
      mask.Context().prefill_input_buffers.at(
          MaskSignatures::kMaskInputTimeStep),
      TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(step_lock.HasValue());
  EXPECT_EQ(step_lock->second[0], 15);

  auto tokens_lock = TensorBufferScopedLock::Create<int32_t>(
      mask.Context().prefill_input_buffers.at(MaskSignatures::kMaskInputTokens),
      TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(tokens_lock.HasValue());
  EXPECT_EQ(tokens_lock->second[0], 101);
  EXPECT_EQ(tokens_lock->second[1], 0);  // negative clamped to 0
  EXPECT_EQ(tokens_lock->second[2], 202);
  EXPECT_EQ(tokens_lock->second[3], 0);  // zero padded
  EXPECT_EQ(tokens_lock->second[4], 0);

  auto valid_mask_lock = TensorBufferScopedLock::Create<bool>(
      mask.Context().prefill_input_buffers.at(
          MaskSignatures::kMaskInputValidMask),
      TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(valid_mask_lock.HasValue());
  EXPECT_TRUE(valid_mask_lock->second[0]);
  EXPECT_TRUE(valid_mask_lock->second[1]);
  EXPECT_TRUE(valid_mask_lock->second[2]);
  EXPECT_FALSE(valid_mask_lock->second[3]);
  EXPECT_FALSE(valid_mask_lock->second[4]);
}

TEST_F(ExecutorUtilsTest,
       NpuKVCacheCreateFailsWhenCompiledModelNullForModelMethod) {
  InferenceContext ctx;
  auto update_or = NpuKVCache::CreateForTest(KVCacheUpdateMethod::kModel,
                                             nullptr, std::move(ctx));
  EXPECT_FALSE(update_or.ok());
}

TEST_F(ExecutorUtilsTest,
       NpuKVCacheCreateSucceedsForHwMethodWithoutCompiledModel) {
  InferenceContext ctx;
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto update, NpuKVCache::CreateForTest(KVCacheUpdateMethod::kWH, nullptr,
                                             std::move(ctx)));
  EXPECT_EQ(update.GetMethod(), KVCacheUpdateMethod::kWH);
}

TEST_F(ExecutorUtilsTest, NpuKVCacheCommitVerifiedKVCacheSetsPosition) {
  InferenceContext ctx;
  ctx.verify_input_buffers[CacheUpdateSignatures::kInputPos] =
      CreateTensorBufferWithDims(std::vector<int32_t>{0, 0, 0},
                                 ElementType::Int32, {3});

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto update, NpuKVCache::CreateForTest(KVCacheUpdateMethod::kWH, nullptr,
                                             std::move(ctx)));

  EXPECT_TRUE(update.SetVerifyPos(100).ok());

  auto read_lock = TensorBufferScopedLock::Create<int32_t>(
      update.Context().verify_input_buffers.at(
          CacheUpdateSignatures::kInputPos),
      TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(read_lock.HasValue());
  EXPECT_EQ(read_lock->second[0], 100);
  EXPECT_EQ(read_lock->second[1], 101);
  EXPECT_EQ(read_lock->second[2], 102);
}

TEST_F(ExecutorUtilsTest, NpuRopeCreateFailsWhenCompiledModelNull) {
  InferenceContext ctx;
  auto rope_or = NpuRope::CreateForTest(nullptr, std::move(ctx));
  EXPECT_FALSE(rope_or.ok());
}

TEST_F(ExecutorUtilsTest, NpuRopeSetDecodePosition) {
  InferenceContext ctx;
  ctx.decode_input_buffers[RopeSignatures::kInputPos] =
      CreateTensorBufferWithDims(std::vector<int32_t>{0}, ElementType::Int32,
                                 {1});

  // For testing SetDecodePosition, create with a dummy compiled model pointer
  // or cast
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto rope, NpuRope::CreateForTest(
                     reinterpret_cast<const ::litert::CompiledModel*>(0x1234),
                     std::move(ctx)));

  EXPECT_TRUE(rope.SetDecodePosition(42).ok());

  auto read_lock = TensorBufferScopedLock::Create<int32_t>(
      rope.Context().decode_input_buffers.at(RopeSignatures::kInputPos),
      TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(read_lock.HasValue());
  EXPECT_EQ(read_lock->second[0], 42);
}

TEST_F(ExecutorUtilsTest, NpuRopeDrafterSetDecodePosition) {
  absl::flat_hash_map<absl::string_view, TensorBuffer> in_buffers;
  in_buffers[RopeSignatures::kInputPos] = CreateTensorBufferWithDims(
      std::vector<int32_t>{0}, ElementType::Int32, {1});

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto rope, NpuRope::CreateForDrafter(
                     reinterpret_cast<const ::litert::CompiledModel*>(0x1234),
                     std::move(in_buffers), {}));

  EXPECT_TRUE(rope.SetDecodePosition(77).ok());

  auto read_lock = TensorBufferScopedLock::Create<int32_t>(
      rope.Context().decode_input_buffers.at(RopeSignatures::kInputPos),
      TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(read_lock.HasValue());
  EXPECT_EQ(read_lock->second[0], 77);
}

TEST_F(ExecutorUtilsTest, NpuRopeSetVerifyPositions) {
  InferenceContext ctx;
  ctx.verify_input_buffers[RopeSignatures::kInputPos] =
      CreateTensorBufferWithDims(std::vector<int32_t>{0, 0, 0, 0},
                                 ElementType::Int32, {4});

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto rope, NpuRope::CreateForTest(
                     reinterpret_cast<const ::litert::CompiledModel*>(0x1234),
                     std::move(ctx)));

  EXPECT_TRUE(rope.SetVerifyPositions(10, 3).ok());

  auto read_lock = TensorBufferScopedLock::Create<int32_t>(
      rope.Context().verify_input_buffers.at(RopeSignatures::kInputPos),
      TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(read_lock.HasValue());
  EXPECT_EQ(read_lock->second[0], 10);
  EXPECT_EQ(read_lock->second[1], 11);
  EXPECT_EQ(read_lock->second[2], 12);
  EXPECT_EQ(read_lock->second[3], 0);
}

TEST_F(ExecutorUtilsTest, NpuRopeSetPrefillPositionsAndZeroPads) {
  InferenceContext ctx;
  ctx.prefill_input_buffers[RopeSignatures::kInputPos] =
      CreateTensorBufferWithDims(std::vector<int32_t>{99, 99, 99, 99, 99},
                                 ElementType::Int32, {5});

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto rope, NpuRope::CreateForTest(
                     reinterpret_cast<const ::litert::CompiledModel*>(0x1234),
                     std::move(ctx)));

  std::vector<int32_t> positions = {5, 6, 7};
  EXPECT_TRUE(rope.SetPrefillPositions(positions).ok());

  auto read_lock = TensorBufferScopedLock::Create<int32_t>(
      rope.Context().prefill_input_buffers.at(RopeSignatures::kInputPos),
      TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(read_lock.HasValue());
  EXPECT_EQ(read_lock->second[0], 5);
  EXPECT_EQ(read_lock->second[1], 6);
  EXPECT_EQ(read_lock->second[2], 7);
  EXPECT_EQ(read_lock->second[3], 0);
  EXPECT_EQ(read_lock->second[4], 0);
}

}  // namespace
}  // namespace litert::lm
