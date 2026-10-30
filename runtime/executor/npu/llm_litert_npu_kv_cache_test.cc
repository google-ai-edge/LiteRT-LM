// Copyright 2026 Google LLC.
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

#include "runtime/executor/npu/llm_litert_npu_kv_cache.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/cc/litert_tensor_buffer_types.h"  // from @litert
#include "litert/test/matchers.h"  // from @litert
#include "runtime/executor/npu/llm_litert_npu_compiled_model_executor_utils.h"
#include "runtime/proto/executor_metadata.pb.h"

namespace litert::lm {
namespace {

using ::litert::ElementType;
using ::litert::Layout;
using ::litert::RankedTensorType;
using ::litert::TensorBuffer;
using ::litert::TensorBufferScopedLock;

class NpuKVCacheTest : public ::testing::Test {
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
    for (int32_t dim : dims) {
      dimensions.push_back(dim);
    }
    RankedTensorType tensor_type(type, Layout(dimensions));
    auto buffer = TensorBuffer::CreateManaged(
        *env_, ::litert::TensorBufferType::kHostMemory, tensor_type,
        data.size() * sizeof(T));
    ABSL_CHECK(buffer.HasValue());

    auto lock =
        TensorBufferScopedLock::Create(*buffer, TensorBuffer::LockMode::kWrite);
    ABSL_CHECK(lock.HasValue());
    std::memcpy(lock->second, data.data(), data.size() * sizeof(T));

    return std::move(*buffer);
  }

  std::optional<::litert::Environment> env_;
};

TEST_F(NpuKVCacheTest, HWKVCacheUpdateBasic) {
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

  NpuGraphLayout layout = NpuGraphLayout::Resolve(0, 0, in_buffers, nullptr);
  ASSERT_TRUE(HWKVCacheUpdate(in_buffers, out_buffers, &layout).ok());

  auto lock_expected = TensorBufferScopedLock::Create<float>(
      in_buffers.at("kv_cache_k_0"), TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(lock_expected.HasValue());
  auto& lock = *lock_expected;
  for (int i = 0; i < slice_seq * hidden_dim; ++i) {
    EXPECT_EQ(lock.second[start_pos * hidden_dim + i], slice_data[i]);
  }
}

TEST_F(NpuKVCacheTest, HWKVCacheUpdateTransposedInt8) {
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

  NpuGraphLayout layout = NpuGraphLayout::Resolve(0, 0, in_buffers, nullptr);
  ASSERT_TRUE(HWKVCacheUpdate(in_buffers, out_buffers, &layout).ok());

  auto lock_expected = TensorBufferScopedLock::Create<int8_t>(
      in_buffers.at("kv_cache_k_0"), TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(lock_expected.HasValue());
  auto& lock = *lock_expected;
  for (int h = 0; h < hidden_dim; ++h) {
    EXPECT_EQ(lock.second[h * cache_seq + start_pos], slice_data[h]);
  }
}

TEST_F(NpuKVCacheTest, HWKVCacheUpdateTransposedInt16) {
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

  NpuGraphLayout layout = NpuGraphLayout::Resolve(0, 0, in_buffers, nullptr);
  ASSERT_TRUE(HWKVCacheUpdate(in_buffers, out_buffers, &layout).ok());

  auto lock_expected = TensorBufferScopedLock::Create<int16_t>(
      in_buffers.at("kv_cache_k_0"), TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(lock_expected.HasValue());
  auto& lock = *lock_expected;
  for (int h = 0; h < hidden_dim; ++h) {
    EXPECT_EQ(lock.second[h * cache_seq + start_pos], slice_data[h]);
  }
}

TEST_F(NpuKVCacheTest, HWKVCacheUpdateOutOfRange) {
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

  NpuGraphLayout layout = NpuGraphLayout::Resolve(0, 0, in_buffers, nullptr);
  EXPECT_FALSE(HWKVCacheUpdate(in_buffers, out_buffers, &layout).ok());
}

TEST_F(NpuKVCacheTest, HWKVCacheUpdateGemma3nPrefill) {
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

  NpuGraphLayout layout = NpuGraphLayout::Resolve(0, 0, in_buffers, nullptr);
  ASSERT_TRUE(HWKVCacheUpdate(in_buffers, out_buffers, &layout).ok());

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

TEST_F(NpuKVCacheTest, HWKVCacheUpdateSWADecode) {
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

  NpuGraphLayout layout = NpuGraphLayout::Resolve(0, 0, in_buffers, nullptr);
  layout.uses_ringbuffer = true;
  ASSERT_TRUE(HWKVCacheUpdate(in_buffers, out_buffers, &layout).ok());

  auto lock_expected = TensorBufferScopedLock::Create<float>(
      in_buffers.at("kv_cache_k_0"), TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(lock_expected.HasValue());
  auto& lock = *lock_expected;

  int target_pos = start_pos % cache_seq;
  for (int i = 0; i < cache_seq; ++i) {
    for (int h = 0; h < hidden_dim; ++h) {
      float expected = (i == target_pos) ? slice_data[h] : 0.0f;
      EXPECT_EQ(lock.second[i * hidden_dim + h], expected)
          << "Mismatch at seq " << i << " head " << h;
    }
  }
}

TEST_F(NpuKVCacheTest, HWKVCacheUpdateSWADecodeTransposed) {
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

  NpuGraphLayout layout = NpuGraphLayout::Resolve(0, 0, in_buffers, nullptr);
  layout.uses_ringbuffer = true;
  ASSERT_TRUE(HWKVCacheUpdate(in_buffers, out_buffers, &layout).ok());

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

TEST_F(NpuKVCacheTest, HWKVCacheUpdateSWAPrefillWrap) {
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

  NpuGraphLayout layout = NpuGraphLayout::Resolve(0, 0, in_buffers, nullptr);
  layout.uses_ringbuffer = true;
  ASSERT_TRUE(HWKVCacheUpdate(in_buffers, out_buffers, &layout).ok());

  auto lock_expected = TensorBufferScopedLock::Create<float>(
      in_buffers.at("kv_cache_k_0"), TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(lock_expected.HasValue());
  auto& lock = *lock_expected;

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

  for (int s = 0; s < cache_seq; ++s) {
    int slice_idx = expected_slice_idx[s];
    for (int h = 0; h < hidden_dim; ++h) {
      float expected =
          (slice_idx != -1) ? slice_data[slice_idx * hidden_dim + h] : 0.0f;
      EXPECT_EQ(lock.second[s * hidden_dim + h], expected)
          << "Mismatch at seq " << s << " head " << h;
    }
  }
}

TEST_F(NpuKVCacheTest, HWKVCacheUpdateSWAPrefillWrapTransposed) {
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

  NpuGraphLayout layout = NpuGraphLayout::Resolve(0, 0, in_buffers, nullptr);
  layout.uses_ringbuffer = true;
  ASSERT_TRUE(HWKVCacheUpdate(in_buffers, out_buffers, &layout).ok());

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

TEST_F(NpuKVCacheTest, HWKVCacheUpdateSWAPrefillWithValidMask) {
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

  NpuGraphLayout layout = NpuGraphLayout::Resolve(0, 0, in_buffers, nullptr);
  layout.uses_ringbuffer = true;
  ASSERT_TRUE(HWKVCacheUpdate(in_buffers, out_buffers, &layout).ok());

  auto lock_expected = TensorBufferScopedLock::Create<float>(
      in_buffers.at("kv_cache_k_0"), TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(lock_expected.HasValue());
  auto& lock = *lock_expected;

  for (int i = 0; i < 6; ++i) {
    EXPECT_EQ(lock.second[i * hidden_dim + 0], slice_data[i * hidden_dim + 0])
        << "Mismatch at seq " << i;
    EXPECT_EQ(lock.second[i * hidden_dim + 1], slice_data[i * hidden_dim + 1])
        << "Mismatch at seq " << i;
  }
  for (int i = 6; i < 8; ++i) {
    EXPECT_EQ(lock.second[i * hidden_dim + 0], cache_data[i * hidden_dim + 0])
        << "Overwritten at seq " << i;
    EXPECT_EQ(lock.second[i * hidden_dim + 1], cache_data[i * hidden_dim + 1])
        << "Overwritten at seq " << i;
  }
}

TEST_F(NpuKVCacheTest, HWKVCacheUpdateInvalidPos) {
  int hidden_dim = 4;
  int cache_seq = 5;
  int slice_seq = 2;
  int start_pos = -1;

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
  NpuGraphLayout layout = NpuGraphLayout::Resolve(0, 0, in_buffers, nullptr);
  EXPECT_FALSE(HWKVCacheUpdate(in_buffers, out_buffers, &layout).ok());
}

TEST_F(NpuKVCacheTest, HWKVCacheUpdateMismatchedOuterDims) {
  int hidden_dim = 4;
  int cache_seq = 5;
  int slice_seq = 2;
  int start_pos = 0;

  std::vector<float> cache_data(2 * hidden_dim * cache_seq, 0.0f);
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
  NpuGraphLayout layout = NpuGraphLayout::Resolve(0, 0, in_buffers, nullptr);
  EXPECT_FALSE(HWKVCacheUpdate(in_buffers, out_buffers, &layout).ok());
}

TEST_F(NpuKVCacheTest, HWKVCacheUpdateMismatchedElementTypes) {
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
  NpuGraphLayout layout = NpuGraphLayout::Resolve(0, 0, in_buffers, nullptr);
  EXPECT_FALSE(HWKVCacheUpdate(in_buffers, out_buffers, &layout).ok());
}

TEST_F(NpuKVCacheTest, HWKVCacheUpdateDequantizeInt16ToFloat32) {
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

  NpuGraphLayout layout = NpuGraphLayout::Resolve(0, 0, in_buffers, nullptr);
  ASSERT_TRUE(
      HWKVCacheUpdate(in_buffers, out_buffers, &layout, quant_params).ok());

  auto k_lock_expected = TensorBufferScopedLock::Create<float>(
      in_buffers.at("kv_cache_k_0"), TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(k_lock_expected.HasValue());
  auto& k_lock = *k_lock_expected;
  for (int i = 0; i < slice_seq * hidden_dim; ++i) {
    float expected = (static_cast<float>(slice_data[i]) - 50.0f) * 0.1f;
    EXPECT_NEAR(k_lock.second[start_pos * hidden_dim + i], expected, 1e-5)
        << "Index " << i;
  }

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

TEST_F(NpuKVCacheTest, HWKVCacheUpdateConvolution) {
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

  NpuGraphLayout layout = NpuGraphLayout::Resolve(0, 0, in_buffers, nullptr);
  ASSERT_TRUE(HWKVCacheUpdate(in_buffers, out_buffers, &layout).ok());

  auto lock_expected = TensorBufferScopedLock::Create<float>(
      in_buffers.at("kv_cache_c_0"), TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(lock_expected.HasValue());
  auto& lock = *lock_expected;
  for (int i = 0; i < (int)slice_data.size(); ++i) {
    EXPECT_EQ(lock.second[i], slice_data[i]);
  }
}

TEST_F(NpuKVCacheTest, HWKVCacheUpdateConvolutionOutBuffer) {
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

  NpuGraphLayout layout2 = NpuGraphLayout::Resolve(0, 0, in_buffers, nullptr);
  ASSERT_TRUE(HWKVCacheUpdate(in_buffers, out_buffers, &layout2).ok());

  {
    auto lock_expected = TensorBufferScopedLock::Create<float>(
        in_buffers.at("kv_cache_c_1"), TensorBuffer::LockMode::kRead);
    ASSERT_TRUE(lock_expected.HasValue());
    auto& lock = *lock_expected;
    for (int i = 0; i < (int)slice_data.size(); ++i) {
      EXPECT_EQ(lock.second[i], 2.0f);
    }
  }

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

TEST_F(NpuKVCacheTest,
       NpuKVCacheCreateFailsWhenCompiledModelNullForModelMethod) {
  InferenceContext ctx;
  NpuGraphLayout layout;
  auto update_or = NpuKVCache::CreateForTest(KVCacheUpdateMethod::kModel,
                                             nullptr, std::move(ctx),
                                             /*kv_quant_params=*/{}, &layout);
  EXPECT_FALSE(update_or.ok());
}

TEST_F(NpuKVCacheTest,
       NpuKVCacheCreateSucceedsForHwMethodWithoutCompiledModel) {
  InferenceContext ctx;
  NpuGraphLayout layout;
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto update, NpuKVCache::CreateForTest(KVCacheUpdateMethod::kWH, nullptr,
                                             std::move(ctx),
                                             /*kv_quant_params=*/{}, &layout));
  EXPECT_EQ(update.GetMethod(), KVCacheUpdateMethod::kWH);
}

TEST_F(NpuKVCacheTest, NpuKVCacheCommitVerifiedKVCacheSetsPosition) {
  InferenceContext ctx;
  ctx.verify_input_buffers[CacheUpdateSignatures::kInputPos] =
      CreateTensorBufferWithDims(std::vector<int32_t>{0, 0, 0},
                                 ElementType::Int32, {3});

  NpuGraphLayout layout;
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto update, NpuKVCache::CreateForTest(KVCacheUpdateMethod::kWH, nullptr,
                                             std::move(ctx),
                                             /*kv_quant_params=*/{}, &layout));

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

TEST_F(NpuKVCacheTest, FillKVCacheBufferHonorsInitValueForInt8) {
  TensorBuffer buffer =
      CreateTensorBuffer(std::vector<int8_t>(16, 7), ElementType::Int8);
  LITERT_ASSERT_OK(FillKVCacheBuffer(buffer, /*init_value=*/-1));
  auto lock = TensorBufferScopedLock::Create<int8_t>(
      buffer, TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(lock.HasValue());
  for (int i = 0; i < 16; ++i) EXPECT_EQ(lock->second[i], -1);
}

TEST_F(NpuKVCacheTest, FillKVCacheBufferHonorsInitValueForUInt8) {
  TensorBuffer buffer =
      CreateTensorBuffer(std::vector<uint8_t>(16, 7), ElementType::UInt8);
  LITERT_ASSERT_OK(FillKVCacheBuffer(buffer, /*init_value=*/128));
  auto lock = TensorBufferScopedLock::Create<uint8_t>(
      buffer, TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(lock.HasValue());
  for (int i = 0; i < 16; ++i) EXPECT_EQ(lock->second[i], 128);
}

TEST_F(NpuKVCacheTest, FillKVCacheBufferHonorsInitValueForInt16) {
  TensorBuffer buffer =
      CreateTensorBuffer(std::vector<int16_t>(16, 7), ElementType::Int16);
  LITERT_ASSERT_OK(FillKVCacheBuffer(buffer, /*init_value=*/-300));
  auto lock = TensorBufferScopedLock::Create<int16_t>(
      buffer, TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(lock.HasValue());
  for (int i = 0; i < 16; ++i) EXPECT_EQ(lock->second[i], -300);
}

TEST_F(NpuKVCacheTest, FillKVCacheBufferHonorsInitValueForUInt16) {
  TensorBuffer buffer =
      CreateTensorBuffer(std::vector<uint16_t>(16, 7), ElementType::UInt16);
  LITERT_ASSERT_OK(FillKVCacheBuffer(buffer, /*init_value=*/32768));
  auto lock = TensorBufferScopedLock::Create<uint16_t>(
      buffer, TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(lock.HasValue());
  for (int i = 0; i < 16; ++i) EXPECT_EQ(lock->second[i], 32768);
}

TEST_F(NpuKVCacheTest, FillKVCacheBufferZerosIntegerBufferForZeroInitValue) {
  TensorBuffer buffer =
      CreateTensorBuffer(std::vector<int16_t>(16, 7), ElementType::Int16);
  LITERT_ASSERT_OK(FillKVCacheBuffer(buffer, /*init_value=*/0));
  auto lock = TensorBufferScopedLock::Create<int16_t>(
      buffer, TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(lock.HasValue());
  for (int i = 0; i < 16; ++i) EXPECT_EQ(lock->second[i], 0);
}

TEST_F(NpuKVCacheTest, FillKVCacheBufferIgnoresInitValueForFloat) {
  // The init value is in the integer storage domain; float KV is zeroed.
  TensorBuffer buffer =
      CreateTensorBuffer(std::vector<float>(16, 7.0f), ElementType::Float32);
  LITERT_ASSERT_OK(FillKVCacheBuffer(buffer, /*init_value=*/32768));
  auto lock = TensorBufferScopedLock::Create<float>(
      buffer, TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(lock.HasValue());
  for (int i = 0; i < 16; ++i) EXPECT_EQ(lock->second[i], 0.0f);
}

TEST_F(NpuKVCacheTest, ClearKVCacheBuffersOnlyFillsKVCacheBuffers) {
  absl::flat_hash_map<absl::string_view, TensorBuffer> buffers;
  buffers.emplace("kv_cache_k_0", CreateTensorBuffer(std::vector<int8_t>(8, 7),
                                                     ElementType::Int8));
  buffers.emplace("kv_cache_v_0", CreateTensorBuffer(std::vector<int8_t>(8, 7),
                                                     ElementType::Int8));
  buffers.emplace("input_pos", CreateTensorBuffer(std::vector<int8_t>(8, 7),
                                                  ElementType::Int8));
  LITERT_ASSERT_OK(ClearKVCacheBuffers(buffers, /*init_value=*/-1));
  for (absl::string_view name : {"kv_cache_k_0", "kv_cache_v_0", "input_pos"}) {
    auto lock = TensorBufferScopedLock::Create<int8_t>(
        buffers.at(name), TensorBuffer::LockMode::kRead);
    ASSERT_TRUE(lock.HasValue());
    const int8_t expected = name == "input_pos" ? 7 : -1;
    for (int i = 0; i < 8; ++i) EXPECT_EQ(lock->second[i], expected) << name;
  }
}

TEST_F(NpuKVCacheTest, CopySingleKVCacheBufferPadsFloatWithZeros) {
  // [1, 1, seq, 2] float: grow from 2 to 4 with 1 active token; the init value
  // does not apply to float, so the padding is zero.
  TensorBuffer src =
      CreateTensorBufferWithDims(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f},
                                 ElementType::Float32, {1, 1, 2, 2});
  TensorBuffer dst = CreateTensorBufferWithDims(
      std::vector<float>(8, 9.0f), ElementType::Float32, {1, 1, 4, 2});
  LITERT_ASSERT_OK(NpuKVCache::CopySingleKVCacheBuffer(
      src, dst, /*active_seq_len=*/1, /*kv_cache_init_value=*/32768,
      /*sequence_axis=*/2));
  auto lock =
      TensorBufferScopedLock::Create<float>(dst, TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(lock.HasValue());
  const std::vector<float> expected = {1.0f, 2.0f, 0.0f, 0.0f,
                                       0.0f, 0.0f, 0.0f, 0.0f};
  for (int i = 0; i < 8; ++i) EXPECT_EQ(lock->second[i], expected[i]) << i;
}

TEST_F(NpuKVCacheTest, SingleHeadCopyAndClearKVCache) {
  constexpr int kOldSeqLen = 640;
  constexpr int kNewSeqLen = 1024;
  constexpr int kHeadDim = 256;
  constexpr int8_t kInitVal = -1;

  // 1. Allocate Master 1024 Buffers
  // K: int8[1, 1, 1024, 256] -> size: 1024 * 256
  std::vector<int8_t> master_k_init(kNewSeqLen * kHeadDim, kInitVal);
  TensorBuffer master_k_buf = CreateTensorBufferWithDims(
      master_k_init, ElementType::Int8, {1, 1, kNewSeqLen, kHeadDim});

  // V: int8[1, 1, 256, 1024] -> size: 256 * 1024
  std::vector<int8_t> master_v_init(kHeadDim * kNewSeqLen, kInitVal);
  TensorBuffer master_v_buf = CreateTensorBufferWithDims(
      master_v_init, ElementType::Int8, {1, 1, kHeadDim, kNewSeqLen});

  // 2. Create Aliased 640 Views into Master
  RankedTensorType k_640_type(
      ElementType::Int8,
      Layout(::litert::Dimensions{1, 1, kOldSeqLen, kHeadDim}));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto alias_k_buf, CreateAliasBuffer(*env_, master_k_buf, k_640_type));

  RankedTensorType v_640_type(
      ElementType::Int8,
      Layout(::litert::Dimensions{1, 1, kHeadDim, kOldSeqLen}));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto alias_v_buf, CreateAliasBuffer(*env_, master_v_buf, v_640_type));

  // 3. Verify aliased 640 views start with -1
  {
    auto k_lock = TensorBufferScopedLock::Create<int8_t>(
        alias_k_buf, TensorBuffer::LockMode::kRead);
    ASSERT_TRUE(k_lock.HasValue());
    for (int i = 0; i < kOldSeqLen * kHeadDim; ++i) {
      EXPECT_EQ(k_lock->second[i], kInitVal);
    }

    auto v_lock = TensorBufferScopedLock::Create<int8_t>(
        alias_v_buf, TensorBuffer::LockMode::kRead);
    ASSERT_TRUE(v_lock.HasValue());
    for (int i = 0; i < kHeadDim * kOldSeqLen; ++i) {
      EXPECT_EQ(v_lock->second[i], kInitVal);
    }
  }

  // 4. Simulate Writing 640 tokens into the 640 views
  //
  // Layout representation:
  // K buffer shape: [1, 1, 640 (seq_len), 256 (head_dim)]
  // Row (token t), Column (channel d):
  // [
  //   [1,   1,   ..., 1  ],  // token 0
  //   [2,   2,   ..., 2  ],  // token 1
  //   ...
  //   [120, 120, ..., 120],  // token 119
  //   [1,   1,   ..., 1  ],  // token 120
  //   ...
  // ]
  //
  // V buffer shape (transposed): [1, 1, 256 (head_dim), 640 (seq_len)]
  // Row (channel d), Column (token t):
  // [
  //   [1, 2, ..., 120, 1, 2, ...],  // channel 0
  //   [1, 2, ..., 120, 1, 2, ...],  // channel 1
  //   ...
  //   [1, 2, ..., 120, 1, 2, ...]   // channel 255
  // ]
  {
    auto k_lock = TensorBufferScopedLock::Create<int8_t>(
        alias_k_buf, TensorBuffer::LockMode::kWrite);
    ASSERT_TRUE(k_lock.HasValue());
    for (int t = 0; t < kOldSeqLen; ++t) {
      int8_t val = static_cast<int8_t>((t % 120) + 1);
      for (int d = 0; d < kHeadDim; ++d) {
        k_lock->second[t * kHeadDim + d] = val;
      }
    }

    auto v_lock = TensorBufferScopedLock::Create<int8_t>(
        alias_v_buf, TensorBuffer::LockMode::kWrite);
    ASSERT_TRUE(v_lock.HasValue());
    for (int t = 0; t < kOldSeqLen; ++t) {
      int8_t val = static_cast<int8_t>((t % 120) + 1);
      for (int d = 0; d < kHeadDim; ++d) {
        // Transposed: [1, 1, HeadDim, SeqLen]
        v_lock->second[d * kOldSeqLen + t] = val;
      }
    }
  }

  // 5. Execute CopyKVCache (Migration 640 -> 1024)
  absl::flat_hash_map<absl::string_view, TensorBuffer> src_map;
  src_map.emplace("kv_cache_k_10", std::move(alias_k_buf));
  src_map.emplace("kv_cache_v_10", std::move(alias_v_buf));

  absl::flat_hash_map<absl::string_view, TensorBuffer> dst_map;
  LITERT_ASSERT_OK_AND_ASSIGN(auto master_k_dst, master_k_buf.Duplicate());
  LITERT_ASSERT_OK_AND_ASSIGN(auto master_v_dst, master_v_buf.Duplicate());
  dst_map.emplace("kv_cache_k_10", std::move(master_k_dst));
  dst_map.emplace("kv_cache_v_10", std::move(master_v_dst));

  InferenceContext dummy_ctx;
  NpuGraphLayout layout;
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto kv_cache,
      NpuKVCache::CreateForTest(KVCacheUpdateMethod::kWH, nullptr,
                                std::move(dummy_ctx), /*kv_quant_params=*/{},
                                /*layout=*/&layout,
                                /*kv_cache_init_value=*/kInitVal));

  LITERT_ASSERT_OK(
      kv_cache.CopyKVCache(src_map, dst_map, /*active_seq_len=*/kOldSeqLen));

  // 6. Verify 1024 Master Buffers
  //
  // After migration from 640 -> 1024, the memory layout looks as follows:
  //
  // K buffer shape: [1, 1, 1024 (seq_len), 256 (head_dim)]
  // Row (token t), Column (channel d):
  // [
  //   // --- Active tokens [0, 640) ---
  //   [  1,   1, ...,   1],  // token 0
  //   [  2,   2, ...,   2],  // token 1
  //   ...
  //   [120, 120, ..., 120],  // token 119
  //   [  1,   1, ...,   1],  // token 120
  //   ...
  //   [ 40,  40, ...,  40],  // token 639 (val = (639%120)+1)
  //   // --- Padded / newly exposed slots [640, 1024) ---
  //   [ -1,  -1, ...,  -1],  // token 640
  //   ...
  //   [ -1,  -1, ...,  -1]   // token 1023
  // ]
  //
  // V buffer shape (transposed): [1, 1, 256 (head_dim), 1024 (seq_len)]
  // Row (channel d), Column (token t):
  // [
  //   // Columns:  0, 1 ... 639 (val=40) | 640, 641 ... 1023
  //   //          [-- Active tokens ---] | [---- Padded ----]
  //   [            1, 2 ... 40,             -1,  -1  ... -1 ],  // ch 0
  //   [            1, 2 ... 40,             -1,  -1  ... -1 ],  // ch 1
  //   ...
  //   [            1, 2 ... 40,             -1,  -1  ... -1 ]   // ch 255
  // ]
  //
  // Notice that before migration, ch 1's active data was at elements
  // [640, 1280). In the 1024 layout, ch 1's active data moved to [1024, 1664).
  // Elements [640, 1024) (ch 0's padding) previously held ch 1's old
  // data, and are now verified to be cleanly reset to kInitVal (-1).

  // Verify K: [1, 1, 1024, 256]
  {
    auto k_lock = TensorBufferScopedLock::Create<int8_t>(
        master_k_buf, TensorBuffer::LockMode::kRead);
    ASSERT_TRUE(k_lock.HasValue());
    // Active tokens [0, 640)
    for (int t = 0; t < kOldSeqLen; ++t) {
      int8_t expected_val = static_cast<int8_t>((t % 120) + 1);
      for (int d = 0; d < kHeadDim; ++d) {
        EXPECT_EQ(k_lock->second[t * kHeadDim + d], expected_val);
      }
    }
    // Exposed padded slots [640, 1024)
    for (int t = kOldSeqLen; t < kNewSeqLen; ++t) {
      for (int d = 0; d < kHeadDim; ++d) {
        EXPECT_EQ(k_lock->second[t * kHeadDim + d], kInitVal);
      }
    }
  }

  // Verify V (Transposed): [1, 1, 256, 1024]
  {
    auto v_lock = TensorBufferScopedLock::Create<int8_t>(
        master_v_buf, TensorBuffer::LockMode::kRead);
    ASSERT_TRUE(v_lock.HasValue());
    for (int d = 0; d < kHeadDim; ++d) {
      // Active tokens [0, 640)
      for (int t = 0; t < kOldSeqLen; ++t) {
        int8_t expected_val = static_cast<int8_t>((t % 120) + 1);
        EXPECT_EQ(v_lock->second[d * kNewSeqLen + t], expected_val);
      }
      // Exposed padded slots [640, 1024)
      for (int t = kOldSeqLen; t < kNewSeqLen; ++t) {
        EXPECT_EQ(v_lock->second[d * kNewSeqLen + t], kInitVal);
      }
    }
  }
}

TEST_F(NpuKVCacheTest, CascadingMultiTierCopyKVCache) {
  constexpr int kTier0 = 640;
  constexpr int kTier1 = 1024;
  constexpr int kTier2 = 4096;
  constexpr int kHeadDim = 256;
  constexpr int8_t kInitVal = -1;

  // 1. Allocate Master 4096 Buffers (Tier 2)
  std::vector<int8_t> master_k_init(kTier2 * kHeadDim, kInitVal);
  TensorBuffer master_k_buf = CreateTensorBufferWithDims(
      master_k_init, ElementType::Int8, {1, 1, kTier2, kHeadDim});

  std::vector<int8_t> master_v_init(kHeadDim * kTier2, kInitVal);
  TensorBuffer master_v_buf = CreateTensorBufferWithDims(
      master_v_init, ElementType::Int8, {1, 1, kHeadDim, kTier2});

  // 2. Create Aliased Views for Tier 0 (640) and Tier 1 (1024)
  RankedTensorType k_640_type(
      ElementType::Int8, Layout(::litert::Dimensions{1, 1, kTier0, kHeadDim}));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto alias_k_640, CreateAliasBuffer(*env_, master_k_buf, k_640_type));

  RankedTensorType v_640_type(
      ElementType::Int8, Layout(::litert::Dimensions{1, 1, kHeadDim, kTier0}));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto alias_v_640, CreateAliasBuffer(*env_, master_v_buf, v_640_type));

  RankedTensorType k_1024_type(
      ElementType::Int8, Layout(::litert::Dimensions{1, 1, kTier1, kHeadDim}));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto alias_k_1024, CreateAliasBuffer(*env_, master_k_buf, k_1024_type));

  RankedTensorType v_1024_type(
      ElementType::Int8, Layout(::litert::Dimensions{1, 1, kHeadDim, kTier1}));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto alias_v_1024, CreateAliasBuffer(*env_, master_v_buf, v_1024_type));

  // 3. Write 640 tokens into Tier 0 view
  {
    auto k_lock = TensorBufferScopedLock::Create<int8_t>(
        alias_k_640, TensorBuffer::LockMode::kWrite);
    ASSERT_TRUE(k_lock.HasValue());
    for (int t = 0; t < kTier0; ++t) {
      int8_t val = static_cast<int8_t>((t % 120) + 1);
      for (int d = 0; d < kHeadDim; ++d) {
        k_lock->second[t * kHeadDim + d] = val;
      }
    }

    auto v_lock = TensorBufferScopedLock::Create<int8_t>(
        alias_v_640, TensorBuffer::LockMode::kWrite);
    ASSERT_TRUE(v_lock.HasValue());
    for (int t = 0; t < kTier0; ++t) {
      int8_t val = static_cast<int8_t>((t % 120) + 1);
      for (int d = 0; d < kHeadDim; ++d) {
        v_lock->second[d * kTier0 + t] = val;
      }
    }
  }

  InferenceContext dummy_ctx;
  NpuGraphLayout layout;
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto kv_cache,
      NpuKVCache::CreateForTest(KVCacheUpdateMethod::kWH, nullptr,
                                std::move(dummy_ctx), /*kv_quant_params=*/{},
                                /*layout=*/&layout,
                                /*kv_cache_init_value=*/kInitVal));

  // 4. Migrate Tier 0 (640) -> Tier 1 (1024)
  {
    absl::flat_hash_map<absl::string_view, TensorBuffer> src_map;
    src_map.emplace("kv_cache_k_10", std::move(alias_k_640));
    src_map.emplace("kv_cache_v_10", std::move(alias_v_640));

    absl::flat_hash_map<absl::string_view, TensorBuffer> dst_map;
    LITERT_ASSERT_OK_AND_ASSIGN(auto k_dup, alias_k_1024.Duplicate());
    LITERT_ASSERT_OK_AND_ASSIGN(auto v_dup, alias_v_1024.Duplicate());
    dst_map.emplace("kv_cache_k_10", std::move(k_dup));
    dst_map.emplace("kv_cache_v_10", std::move(v_dup));

    LITERT_ASSERT_OK(
        kv_cache.CopyKVCache(src_map, dst_map, /*active_seq_len=*/kTier0));
  }

  // 5. Decode/write additional 300 tokens into Tier 1 (total active = 940)
  constexpr int kTotalActive = 940;
  {
    auto k_lock = TensorBufferScopedLock::Create<int8_t>(
        alias_k_1024, TensorBuffer::LockMode::kWrite);
    ASSERT_TRUE(k_lock.HasValue());
    for (int t = kTier0; t < kTotalActive; ++t) {
      int8_t val = static_cast<int8_t>((t % 120) + 1);
      for (int d = 0; d < kHeadDim; ++d) {
        k_lock->second[t * kHeadDim + d] = val;
      }
    }

    auto v_lock = TensorBufferScopedLock::Create<int8_t>(
        alias_v_1024, TensorBuffer::LockMode::kWrite);
    ASSERT_TRUE(v_lock.HasValue());
    for (int t = kTier0; t < kTotalActive; ++t) {
      int8_t val = static_cast<int8_t>((t % 120) + 1);
      for (int d = 0; d < kHeadDim; ++d) {
        v_lock->second[d * kTier1 + t] = val;
      }
    }
  }

  // 6. Migrate Tier 1 (1024) -> Tier 2 (4096)
  {
    absl::flat_hash_map<absl::string_view, TensorBuffer> src_map;
    src_map.emplace("kv_cache_k_10", std::move(alias_k_1024));
    src_map.emplace("kv_cache_v_10", std::move(alias_v_1024));

    absl::flat_hash_map<absl::string_view, TensorBuffer> dst_map;
    LITERT_ASSERT_OK_AND_ASSIGN(auto k_dup, master_k_buf.Duplicate());
    LITERT_ASSERT_OK_AND_ASSIGN(auto v_dup, master_v_buf.Duplicate());
    dst_map.emplace("kv_cache_k_10", std::move(k_dup));
    dst_map.emplace("kv_cache_v_10", std::move(v_dup));

    LITERT_ASSERT_OK(kv_cache.CopyKVCache(src_map, dst_map,
                                          /*active_seq_len=*/kTotalActive));
  }

  // 7. Verify Tier 2 (4096) Master Buffers
  {
    auto k_lock = TensorBufferScopedLock::Create<int8_t>(
        master_k_buf, TensorBuffer::LockMode::kRead);
    ASSERT_TRUE(k_lock.HasValue());
    // Active tokens [0, 940)
    for (int t = 0; t < kTotalActive; ++t) {
      int8_t expected_val = static_cast<int8_t>((t % 120) + 1);
      for (int d = 0; d < kHeadDim; ++d) {
        EXPECT_EQ(k_lock->second[t * kHeadDim + d], expected_val);
      }
    }
    // Exposed padded slots [940, 4096)
    for (int t = kTotalActive; t < kTier2; ++t) {
      for (int d = 0; d < kHeadDim; ++d) {
        EXPECT_EQ(k_lock->second[t * kHeadDim + d], kInitVal);
      }
    }
  }

  {
    auto v_lock = TensorBufferScopedLock::Create<int8_t>(
        master_v_buf, TensorBuffer::LockMode::kRead);
    ASSERT_TRUE(v_lock.HasValue());
    for (int d = 0; d < kHeadDim; ++d) {
      // Active tokens [0, 940)
      for (int t = 0; t < kTotalActive; ++t) {
        int8_t expected_val = static_cast<int8_t>((t % 120) + 1);
        EXPECT_EQ(v_lock->second[d * kTier2 + t], expected_val);
      }
      // Exposed padded slots [940, 4096)
      for (int t = kTotalActive; t < kTier2; ++t) {
        EXPECT_EQ(v_lock->second[d * kTier2 + t], kInitVal);
      }
    }
  }
}

TEST_F(NpuKVCacheTest, PartialMidBucketCopyKVCache) {
  constexpr int kOldSeqLen = 640;
  constexpr int kNewSeqLen = 1024;
  constexpr int kActiveTokens = 450;
  constexpr int kHeadDim = 256;
  constexpr int8_t kInitVal = -1;

  // 1. Allocate Master 1024 Buffers
  std::vector<int8_t> master_k_init(kNewSeqLen * kHeadDim, kInitVal);
  TensorBuffer master_k_buf = CreateTensorBufferWithDims(
      master_k_init, ElementType::Int8, {1, 1, kNewSeqLen, kHeadDim});

  std::vector<int8_t> master_v_init(kHeadDim * kNewSeqLen, kInitVal);
  TensorBuffer master_v_buf = CreateTensorBufferWithDims(
      master_v_init, ElementType::Int8, {1, 1, kHeadDim, kNewSeqLen});

  // 2. Create Aliased 640 Views into Master
  RankedTensorType k_640_type(
      ElementType::Int8,
      Layout(::litert::Dimensions{1, 1, kOldSeqLen, kHeadDim}));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto alias_k_buf, CreateAliasBuffer(*env_, master_k_buf, k_640_type));

  RankedTensorType v_640_type(
      ElementType::Int8,
      Layout(::litert::Dimensions{1, 1, kHeadDim, kOldSeqLen}));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto alias_v_buf, CreateAliasBuffer(*env_, master_v_buf, v_640_type));

  // 3. Write only 450 tokens into 640 view (leaving 450..639 as kInitVal)
  {
    auto k_lock = TensorBufferScopedLock::Create<int8_t>(
        alias_k_buf, TensorBuffer::LockMode::kWrite);
    ASSERT_TRUE(k_lock.HasValue());
    for (int t = 0; t < kActiveTokens; ++t) {
      int8_t val = static_cast<int8_t>((t % 120) + 1);
      for (int d = 0; d < kHeadDim; ++d) {
        k_lock->second[t * kHeadDim + d] = val;
      }
    }

    auto v_lock = TensorBufferScopedLock::Create<int8_t>(
        alias_v_buf, TensorBuffer::LockMode::kWrite);
    ASSERT_TRUE(v_lock.HasValue());
    for (int t = 0; t < kActiveTokens; ++t) {
      int8_t val = static_cast<int8_t>((t % 120) + 1);
      for (int d = 0; d < kHeadDim; ++d) {
        v_lock->second[d * kOldSeqLen + t] = val;
      }
    }
  }

  // 4. Migrate 640 -> 1024 with active_seq_len = 450
  absl::flat_hash_map<absl::string_view, TensorBuffer> src_map;
  src_map.emplace("kv_cache_k_10", std::move(alias_k_buf));
  src_map.emplace("kv_cache_v_10", std::move(alias_v_buf));

  absl::flat_hash_map<absl::string_view, TensorBuffer> dst_map;
  LITERT_ASSERT_OK_AND_ASSIGN(auto master_k_dst, master_k_buf.Duplicate());
  LITERT_ASSERT_OK_AND_ASSIGN(auto master_v_dst, master_v_buf.Duplicate());
  dst_map.emplace("kv_cache_k_10", std::move(master_k_dst));
  dst_map.emplace("kv_cache_v_10", std::move(master_v_dst));

  InferenceContext dummy_ctx;
  NpuGraphLayout layout;
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto kv_cache,
      NpuKVCache::CreateForTest(KVCacheUpdateMethod::kWH, nullptr,
                                std::move(dummy_ctx), /*kv_quant_params=*/{},
                                /*layout=*/&layout,
                                /*kv_cache_init_value=*/kInitVal));

  LITERT_ASSERT_OK(kv_cache.CopyKVCache(src_map, dst_map,
                                        /*active_seq_len=*/kActiveTokens));

  // 5. Verify 1024 Master Buffers
  // K: [1, 1, 1024, 256]
  {
    auto k_lock = TensorBufferScopedLock::Create<int8_t>(
        master_k_buf, TensorBuffer::LockMode::kRead);
    ASSERT_TRUE(k_lock.HasValue());
    // Active tokens [0, 450)
    for (int t = 0; t < kActiveTokens; ++t) {
      int8_t expected_val = static_cast<int8_t>((t % 120) + 1);
      for (int d = 0; d < kHeadDim; ++d) {
        EXPECT_EQ(k_lock->second[t * kHeadDim + d], expected_val);
      }
    }
    // Padded slots [450, 1024)
    for (int t = kActiveTokens; t < kNewSeqLen; ++t) {
      for (int d = 0; d < kHeadDim; ++d) {
        EXPECT_EQ(k_lock->second[t * kHeadDim + d], kInitVal);
      }
    }
  }

  // V (Transposed): [1, 1, 256, 1024]
  {
    auto v_lock = TensorBufferScopedLock::Create<int8_t>(
        master_v_buf, TensorBuffer::LockMode::kRead);
    ASSERT_TRUE(v_lock.HasValue());
    for (int d = 0; d < kHeadDim; ++d) {
      // Active tokens [0, 450)
      for (int t = 0; t < kActiveTokens; ++t) {
        int8_t expected_val = static_cast<int8_t>((t % 120) + 1);
        EXPECT_EQ(v_lock->second[d * kNewSeqLen + t], expected_val);
      }
      // Padded slots [450, 1024)
      for (int t = kActiveTokens; t < kNewSeqLen; ++t) {
        EXPECT_EQ(v_lock->second[d * kNewSeqLen + t], kInitVal);
      }
    }
  }
}

TEST_F(NpuKVCacheTest, HWKVCacheUpdate_WithKVCacheBufferInfo_LocalAndGlobal) {
  int hidden_dim = 4;
  int cache_seq = 10;
  int slice_seq = 2;

  // Layer 0 is local (ringbuffer), Layer 1 is global (linear).
  // K shape: [1, 1, seq, dim] -> sequence_axis = 2
  // V shape: [1, 1, dim, seq] -> sequence_axis = 3
  NpuGraphLayout layout;
  auto& buffer_info_map = layout.kv_cache_buffers;
  buffer_info_map["kv_cache_k_0"] =
      NpuGraphLayout::KVCacheBuffer{.sequence_axis = 2, .is_local = true};
  buffer_info_map["kv_cache_v_0"] =
      NpuGraphLayout::KVCacheBuffer{.sequence_axis = 3, .is_local = true};
  buffer_info_map["kv_slice_k_0"] =
      NpuGraphLayout::KVCacheBuffer{.sequence_axis = 2, .is_local = true};
  buffer_info_map["kv_slice_v_0"] =
      NpuGraphLayout::KVCacheBuffer{.sequence_axis = 3, .is_local = true};

  buffer_info_map["kv_cache_k_1"] =
      NpuGraphLayout::KVCacheBuffer{.sequence_axis = 2, .is_local = false};
  buffer_info_map["kv_cache_v_1"] =
      NpuGraphLayout::KVCacheBuffer{.sequence_axis = 3, .is_local = false};
  buffer_info_map["kv_slice_k_1"] =
      NpuGraphLayout::KVCacheBuffer{.sequence_axis = 2, .is_local = false};
  buffer_info_map["kv_slice_v_1"] =
      NpuGraphLayout::KVCacheBuffer{.sequence_axis = 3, .is_local = false};

  std::vector<float> k_cache_data(cache_seq * hidden_dim, 0.0f);
  std::vector<float> v_cache_data(hidden_dim * cache_seq, 0.0f);
  std::vector<float> k_slice_data(slice_seq * hidden_dim, 1.0f);
  std::vector<float> v_slice_data(hidden_dim * slice_seq, 2.0f);

  // 1. Within capacity: start_pos = 7
  {
    absl::flat_hash_map<absl::string_view, TensorBuffer> in_buffers;
    std::vector<int32_t> pos_data = {7};
    in_buffers.emplace("input_pos",
                       CreateTensorBuffer(pos_data, ElementType::Int32));
    in_buffers.emplace("kv_cache_k_0", CreateTensorBufferWithDims(
                                           k_cache_data, ElementType::Float32,
                                           {1, 1, cache_seq, hidden_dim}));
    in_buffers.emplace("kv_cache_v_0", CreateTensorBufferWithDims(
                                           v_cache_data, ElementType::Float32,
                                           {1, 1, hidden_dim, cache_seq}));
    in_buffers.emplace("kv_slice_k_0", CreateTensorBufferWithDims(
                                           k_slice_data, ElementType::Float32,
                                           {1, 1, slice_seq, hidden_dim}));
    in_buffers.emplace("kv_slice_v_0", CreateTensorBufferWithDims(
                                           v_slice_data, ElementType::Float32,
                                           {1, 1, hidden_dim, slice_seq}));

    in_buffers.emplace("kv_cache_k_1", CreateTensorBufferWithDims(
                                           k_cache_data, ElementType::Float32,
                                           {1, 1, cache_seq, hidden_dim}));
    in_buffers.emplace("kv_cache_v_1", CreateTensorBufferWithDims(
                                           v_cache_data, ElementType::Float32,
                                           {1, 1, hidden_dim, cache_seq}));
    in_buffers.emplace("kv_slice_k_1", CreateTensorBufferWithDims(
                                           k_slice_data, ElementType::Float32,
                                           {1, 1, slice_seq, hidden_dim}));
    in_buffers.emplace("kv_slice_v_1", CreateTensorBufferWithDims(
                                           v_slice_data, ElementType::Float32,
                                           {1, 1, hidden_dim, slice_seq}));

    absl::flat_hash_map<absl::string_view, TensorBuffer> out_buffers;
    LITERT_ASSERT_OK(HWKVCacheUpdate(in_buffers, out_buffers, &layout));
  }

  // 2. Beyond capacity: start_pos = 11.
  // Local layer 0 should wrap to 11 % 10 = 1.
  // Global layer 1 should fail because 11 + 2 > 10.
  {
    absl::flat_hash_map<absl::string_view, TensorBuffer> in_buffers;
    std::vector<int32_t> pos_data = {11};
    in_buffers.emplace("input_pos",
                       CreateTensorBuffer(pos_data, ElementType::Int32));
    in_buffers.emplace("kv_cache_k_0", CreateTensorBufferWithDims(
                                           k_cache_data, ElementType::Float32,
                                           {1, 1, cache_seq, hidden_dim}));
    in_buffers.emplace("kv_cache_v_0", CreateTensorBufferWithDims(
                                           v_cache_data, ElementType::Float32,
                                           {1, 1, hidden_dim, cache_seq}));
    in_buffers.emplace("kv_slice_k_0", CreateTensorBufferWithDims(
                                           k_slice_data, ElementType::Float32,
                                           {1, 1, slice_seq, hidden_dim}));
    in_buffers.emplace("kv_slice_v_0", CreateTensorBufferWithDims(
                                           v_slice_data, ElementType::Float32,
                                           {1, 1, hidden_dim, slice_seq}));

    absl::flat_hash_map<absl::string_view, TensorBuffer> out_buffers;
    // Layer 0 alone succeeds because it wraps
    LITERT_ASSERT_OK(HWKVCacheUpdate(in_buffers, out_buffers, &layout));

    // Adding global layer 1 should cause an error
    in_buffers.emplace("kv_cache_k_1", CreateTensorBufferWithDims(
                                           k_cache_data, ElementType::Float32,
                                           {1, 1, cache_seq, hidden_dim}));
    in_buffers.emplace("kv_cache_v_1", CreateTensorBufferWithDims(
                                           v_cache_data, ElementType::Float32,
                                           {1, 1, hidden_dim, cache_seq}));
    in_buffers.emplace("kv_slice_k_1", CreateTensorBufferWithDims(
                                           k_slice_data, ElementType::Float32,
                                           {1, 1, slice_seq, hidden_dim}));
    in_buffers.emplace("kv_slice_v_1", CreateTensorBufferWithDims(
                                           v_slice_data, ElementType::Float32,
                                           {1, 1, hidden_dim, slice_seq}));

    EXPECT_FALSE(HWKVCacheUpdate(in_buffers, out_buffers, &layout).ok());
  }
}

// =============================================================================
// Sliding window models with ring buffer KV caches.
//
// These tests use the exact KV cache shapes of the Gemma 4 12B and 26B A4B NPU
// models. Both keep the sliding window layers in a 1024-slot ring buffer (8 KV
// heads of dimension 256) and the global layers in a cache that spans the whole
// context (1 or 2 KV heads of dimension 512). The KV caches are f16, K caches
// are [1, heads, seq, dim] and V caches are [1, heads, dim, seq].
//   - Gemma 4 12B: 128-token prefill chunks, a 16383-token global cache and no
//     ExecutorMetadata, so the layout is inferred from the tensor shapes.
//   - Gemma 4 26B A4B: 1024-token prefill chunks, i.e. as long as the ring
//     buffer, a 32768-token global cache and ExecutorMetadata.
//
// Each model is reduced to one sliding window layer (0) and one global layer
// (5). The layout is resolved the way the executor resolves it: from the KV
// caches only (without the slices), with the global cache length as
// `global_cache_length`. The cache updates go through `NpuKVCache`, which the
// executor drives the same way.
// =============================================================================

struct SlidingWindowModel {
  absl::string_view name;
  int prefill_chunk_size;
  // Length of the sliding window layers' ring buffer KV cache.
  int ring_buffer_size;
  // Length of the global layers' KV cache.
  int global_cache_length;
  int num_local_heads;
  int local_head_dim;
  int num_global_heads;
  int global_head_dim;
  bool has_executor_metadata;
};

constexpr SlidingWindowModel kSlidingWindowModels[] = {
    {.name = "Gemma4_12B",
     .prefill_chunk_size = 128,
     .ring_buffer_size = 1024,
     .global_cache_length = 16383,
     .num_local_heads = 8,
     .local_head_dim = 256,
     .num_global_heads = 1,
     .global_head_dim = 512,
     .has_executor_metadata = false},
    {.name = "Gemma4_26B_A4B",
     .prefill_chunk_size = 1024,
     .ring_buffer_size = 1024,
     .global_cache_length = 32768,
     .num_local_heads = 8,
     .local_head_dim = 256,
     .num_global_heads = 2,
     .global_head_dim = 512,
     .has_executor_metadata = true},
};

void PrintTo(const SlidingWindowModel& model, std::ostream* os) {
  *os << model.name;
}

// One of the four KV caches of the reduced model, together with its slice.
struct KVTensor {
  absl::string_view cache_name;
  absl::string_view slice_name;
  bool is_ring_buffer;
  // V tensors have the sequence on the last axis: [1, heads, dim, seq].
  bool is_value;
  int num_heads;
  int head_dim;
  int cache_length;
  // Makes the values stored in each of the four tensors distinct.
  int id;
};

std::vector<KVTensor> KVTensors(const SlidingWindowModel& model) {
  return {
      {.cache_name = "kv_cache_k_0",
       .slice_name = "kv_slice_k_0",
       .is_ring_buffer = true,
       .is_value = false,
       .num_heads = model.num_local_heads,
       .head_dim = model.local_head_dim,
       .cache_length = model.ring_buffer_size,
       .id = 0},
      {.cache_name = "kv_cache_v_0",
       .slice_name = "kv_slice_v_0",
       .is_ring_buffer = true,
       .is_value = true,
       .num_heads = model.num_local_heads,
       .head_dim = model.local_head_dim,
       .cache_length = model.ring_buffer_size,
       .id = 1},
      {.cache_name = "kv_cache_k_5",
       .slice_name = "kv_slice_k_5",
       .is_ring_buffer = false,
       .is_value = false,
       .num_heads = model.num_global_heads,
       .head_dim = model.global_head_dim,
       .cache_length = model.global_cache_length,
       .id = 2},
      {.cache_name = "kv_cache_v_5",
       .slice_name = "kv_slice_v_5",
       .is_ring_buffer = false,
       .is_value = true,
       .num_heads = model.num_global_heads,
       .head_dim = model.global_head_dim,
       .cache_length = model.global_cache_length,
       .id = 3},
  };
}

// Returns the shape of `tensor` for `seq_len` tokens.
std::vector<int32_t> Dims(const KVTensor& tensor, int seq_len) {
  if (tensor.is_value) {
    return {1, tensor.num_heads, tensor.head_dim, seq_len};
  }
  return {1, tensor.num_heads, seq_len, tensor.head_dim};
}

// Returns the index of an element of `tensor` with room for `seq_len` tokens.
int64_t ElementIndex(const KVTensor& tensor, int seq_len, int head, int seq,
                     int channel) {
  if (tensor.is_value) {
    return (int64_t{head} * tensor.head_dim + channel) * seq_len + seq;
  }
  return (int64_t{head} * seq_len + seq) * tensor.head_dim + channel;
}

// Returns the f16 bit pattern that `tensor` holds for the token at `position`,
// in `head` and `channel`. The pattern is never 0, which marks a slot that was
// never written, and it changes with every argument, so that a value that lands
// in the wrong slot, head, channel or tensor is detected.
uint16_t KVValue(const KVTensor& tensor, int position, int head, int channel) {
  uint32_t x = static_cast<uint32_t>(position) * 0x9E3779B1u;
  x ^= static_cast<uint32_t>(head + 1) * 0x85EBCA77u;
  x ^= static_cast<uint32_t>(channel + 1) * 0xC2B2AE3Du;
  x ^= static_cast<uint32_t>(tensor.id + 1) * 0x27D4EB2Fu;
  x ^= x >> 15;
  x *= 0x2C1B3C6Du;
  x ^= x >> 12;
  return static_cast<uint16_t>(x % 0xFFFFu + 1);
}

// Positions of the padding tokens of a partially filled prefill slice, which
// must never reach the KV caches.
constexpr int kPaddingPosition = 1 << 24;

// The token position that each KV cache slot is expected to hold, or -1 for a
// slot that was never written.
class ExpectedKVCache {
 public:
  ExpectedKVCache(int ring_buffer_size, int global_cache_length)
      : ring_buffer_slots_(ring_buffer_size, -1),
        global_slots_(global_cache_length, -1) {}

  void Write(int position) {
    ring_buffer_slots_[position % ring_buffer_slots_.size()] = position;
    global_slots_[position] = position;
  }

  const std::vector<int>& Slots(const KVTensor& tensor) const {
    return tensor.is_ring_buffer ? ring_buffer_slots_ : global_slots_;
  }

 private:
  std::vector<int> ring_buffer_slots_;
  std::vector<int> global_slots_;
};

// Returns ExecutorMetadata like the 26B A4B model's: one state buffer per KV
// cache, with its sequence axis and its (ring buffer or global) cache length.
proto::ExecutorMetadata CreateExecutorMetadata(
    const SlidingWindowModel& model) {
  proto::ExecutorMetadata metadata;
  proto::LlmExecutorMetadata& llm_metadata =
      *metadata.mutable_llm_executor_metadata();
  llm_metadata.set_max_history_size(model.global_cache_length);
  for (const KVTensor& tensor : KVTensors(model)) {
    proto::StateBuffer& state_buffer = *llm_metadata.add_state_buffers();
    const std::string name(tensor.cache_name);
    state_buffer.set_prefill_input_name(name);
    state_buffer.set_prefill_output_name(name);
    state_buffer.set_decode_input_name(name);
    state_buffer.set_decode_output_name(name);
    if (tensor.is_ring_buffer) {
      state_buffer.set_type(tensor.is_value
                                ? proto::StateBuffer::TYPE_LOCAL_VALUE_CACHE
                                : proto::StateBuffer::TYPE_LOCAL_KEY_CACHE);
      state_buffer.set_minimum_sequence_length(tensor.cache_length);
    } else {
      state_buffer.set_type(tensor.is_value
                                ? proto::StateBuffer::TYPE_GLOBAL_VALUE_CACHE
                                : proto::StateBuffer::TYPE_GLOBAL_KEY_CACHE);
    }
    state_buffer.set_sequence_axis(tensor.is_value ? 3 : 2);
    state_buffer.set_maximum_sequence_length(tensor.cache_length);
  }
  return metadata;
}

class SlidingWindowModelKVCacheTest
    : public NpuKVCacheTest,
      public ::testing::WithParamInterface<SlidingWindowModel> {
 protected:
  void SetUp() override {
    NpuKVCacheTest::SetUp();
    const SlidingWindowModel& model = GetParam();
    for (const KVTensor& tensor : KVTensors(model)) {
      kv_caches_.emplace(tensor.cache_name,
                         CreateF16Buffer(Dims(tensor, tensor.cache_length)));
    }
    if (model.has_executor_metadata) {
      executor_metadata_ = CreateExecutorMetadata(model);
    }
    // Like the executor, resolve the layout from the KV caches, which do not
    // include the slices, with the length of the longest KV cache as the
    // global cache length.
    layout_ = NpuGraphLayout::Resolve(
        model.prefill_chunk_size, model.global_cache_length, kv_caches_,
        model.has_executor_metadata ? &executor_metadata_ : nullptr);
  }

  // Returns a zero-initialized f16 buffer.
  TensorBuffer CreateF16Buffer(const std::vector<int32_t>& dims) {
    int64_t num_elements = 1;
    for (int32_t dim : dims) {
      num_elements *= dim;
    }
    return CreateTensorBufferWithDims(std::vector<uint16_t>(num_elements, 0),
                                      ElementType::Float16, dims);
  }

  // Fills `slice`, which has room for `slice_length` tokens, with the values of
  // the `num_tokens` tokens from `start_position` on, followed by padding.
  void FillSlice(TensorBuffer& slice, const KVTensor& tensor, int slice_length,
                 int start_position, int num_tokens) {
    auto lock = TensorBufferScopedLock::Create<uint16_t>(
        slice, TensorBuffer::LockMode::kWrite);
    ASSERT_TRUE(lock.HasValue());
    uint16_t* data = lock->second;
    for (int head = 0; head < tensor.num_heads; ++head) {
      for (int i = 0; i < slice_length; ++i) {
        const int position =
            i < num_tokens ? start_position + i : kPaddingPosition + i;
        for (int channel = 0; channel < tensor.head_dim; ++channel) {
          data[ElementIndex(tensor, slice_length, head, i, channel)] =
              KVValue(tensor, position, head, channel);
        }
      }
    }
  }

  // Expects that every slot of the ring buffers and the slots
  // [`global_begin`, `global_end`) of the global caches hold the values of the
  // tokens in `expected`, and stops the test otherwise.
  void ExpectKVCachesHold(const ExpectedKVCache& expected, int global_begin,
                          int global_end) {
    for (const KVTensor& tensor : KVTensors(GetParam())) {
      if (tensor.is_ring_buffer) {
        ExpectKVCacheHolds(tensor, expected.Slots(tensor), 0,
                           tensor.cache_length);
      } else {
        ExpectKVCacheHolds(tensor, expected.Slots(tensor), global_begin,
                           std::min(global_end, tensor.cache_length));
      }
    }
    ASSERT_FALSE(HasFailure());
  }

  void ExpectKVCacheHolds(const KVTensor& tensor,
                          const std::vector<int>& slot_positions,
                          int begin_slot, int end_slot) {
    auto lock = TensorBufferScopedLock::Create<uint16_t>(
        kv_caches_.at(tensor.cache_name), TensorBuffer::LockMode::kRead);
    ASSERT_TRUE(lock.HasValue());
    const uint16_t* data = lock->second;
    int64_t num_mismatches = 0;
    std::string first_mismatch;
    for (int head = 0; head < tensor.num_heads; ++head) {
      for (int slot = begin_slot; slot < end_slot; ++slot) {
        const int position = slot_positions[slot];
        for (int channel = 0; channel < tensor.head_dim; ++channel) {
          const uint16_t expected =
              position < 0 ? 0 : KVValue(tensor, position, head, channel);
          const uint16_t actual = data[ElementIndex(tensor, tensor.cache_length,
                                                    head, slot, channel)];
          if (actual != expected && num_mismatches++ == 0) {
            first_mismatch =
                absl::StrCat("head ", head, ", slot ", slot, ", channel ",
                             channel, " (expected position ", position, ")");
          }
        }
      }
    }
    EXPECT_EQ(num_mismatches, 0)
        << tensor.cache_name << ": first mismatch at " << first_mismatch;
  }

  absl::flat_hash_map<absl::string_view, TensorBuffer> kv_caches_;
  proto::ExecutorMetadata executor_metadata_;
  NpuGraphLayout layout_;
};

TEST_P(SlidingWindowModelKVCacheTest, ResolvesLayoutFromKVCaches) {
  const SlidingWindowModel& model = GetParam();
  EXPECT_TRUE(layout_.uses_ringbuffer);
  EXPECT_EQ(layout_.global_cache_length, model.global_cache_length);
  ASSERT_TRUE(layout_.local_cache_length.has_value());
  EXPECT_EQ(*layout_.local_cache_length, model.ring_buffer_size);
  EXPECT_EQ(layout_.sliding_window_size, model.ring_buffer_size);

  for (const KVTensor& tensor : KVTensors(model)) {
    // The slices are not among the buffers that the layout is resolved from,
    // so they get the sequence axis of their KV cache.
    for (absl::string_view name : {tensor.cache_name, tensor.slice_name}) {
      SCOPED_TRACE(name);
      auto it = layout_.kv_cache_buffers.find(name);
      ASSERT_TRUE(it != layout_.kv_cache_buffers.end());
      EXPECT_EQ(it->second.sequence_axis, tensor.is_value ? 3 : 2);
      EXPECT_EQ(it->second.is_local, tensor.is_ring_buffer);
    }
  }
}

// Runs a conversation through the cache updates: a first turn that fills only
// part of a prefill chunk, full chunks through two wrap-arounds of the ring
// buffer (for the 26B A4B model, each of them overwrites the whole ring
// buffer), a full chunk that starts 24 slots before the end of the ring buffer,
// decode steps across the end of the ring buffer, a short follow-up turn, and a
// decode step at the last position of the global cache. After every step, every
// slot of the ring buffers holds its latest token and the global caches hold
// every token at its position, and nothing else: neither the padding of a
// partial prefill chunk nor values in the wrong place.
TEST_P(SlidingWindowModelKVCacheTest, PrefillAndDecodeWrapAroundRingBuffer) {
  const SlidingWindowModel& model = GetParam();
  const int chunk = model.prefill_chunk_size;
  const int ring = model.ring_buffer_size;

  // Set up the cache update buffers like `NpuKVCache::Create` does: prefill and
  // decode share the KV caches, and each has its own slices (the outputs of the
  // text decoder), input positions and valid mask.
  absl::flat_hash_map<absl::string_view, TensorBuffer> prefill_slices;
  absl::flat_hash_map<absl::string_view, TensorBuffer> decode_slices;
  InferenceContext context;
  for (const KVTensor& tensor : KVTensors(model)) {
    prefill_slices.emplace(tensor.slice_name,
                           CreateF16Buffer(Dims(tensor, chunk)));
    decode_slices.emplace(tensor.slice_name, CreateF16Buffer(Dims(tensor, 1)));
    TensorBuffer& kv_cache = kv_caches_.at(tensor.cache_name);
    LITERT_ASSERT_OK_AND_ASSIGN(
        context.prefill_input_buffers[tensor.cache_name], kv_cache.Duplicate());
    LITERT_ASSERT_OK_AND_ASSIGN(
        context.prefill_output_buffers[tensor.cache_name],
        kv_cache.Duplicate());
    LITERT_ASSERT_OK_AND_ASSIGN(context.decode_input_buffers[tensor.cache_name],
                                kv_cache.Duplicate());
    LITERT_ASSERT_OK_AND_ASSIGN(
        context.decode_output_buffers[tensor.cache_name], kv_cache.Duplicate());
    LITERT_ASSERT_OK_AND_ASSIGN(
        context.prefill_input_buffers[tensor.slice_name],
        prefill_slices.at(tensor.slice_name).Duplicate());
    LITERT_ASSERT_OK_AND_ASSIGN(
        context.decode_input_buffers[tensor.slice_name],
        decode_slices.at(tensor.slice_name).Duplicate());
  }
  context.prefill_input_buffers[CacheUpdateSignatures::kInputPos] =
      CreateTensorBufferWithDims(std::vector<int32_t>(chunk, 0),
                                 ElementType::Int32, {chunk});
  context.prefill_input_buffers[CacheUpdateSignatures::kInputValidMask] =
      CreateTensorBufferWithDims(std::vector<uint8_t>(chunk, 0),
                                 ElementType::Bool, {1, chunk});
  context.decode_input_buffers[CacheUpdateSignatures::kInputPos] =
      CreateTensorBufferWithDims(std::vector<int32_t>{0}, ElementType::Int32,
                                 {1});
  context.decode_input_buffers[CacheUpdateSignatures::kInputValidMask] =
      CreateTensorBufferWithDims(std::vector<uint8_t>{0}, ElementType::Bool,
                                 {1, 1});
  LITERT_ASSERT_OK_AND_ASSIGN(
      NpuKVCache kv_cache_update,
      NpuKVCache::CreateForTest(KVCacheUpdateMethod::kWH,
                                /*compiled_model=*/nullptr, std::move(context),
                                /*kv_quant_params=*/{}, &layout_));

  ExpectedKVCache expected(ring, model.global_cache_length);
  int position = 0;
  auto prefill = [&](int num_tokens) {
    SCOPED_TRACE(absl::StrCat("Prefill of ", num_tokens, " tokens at position ",
                              position));
    for (const KVTensor& tensor : KVTensors(model)) {
      ASSERT_NO_FATAL_FAILURE(FillSlice(prefill_slices.at(tensor.slice_name),
                                        tensor, chunk, position, num_tokens));
    }
    std::vector<int32_t> positions(num_tokens);
    std::iota(positions.begin(), positions.end(), position);
    LITERT_ASSERT_OK(kv_cache_update.SetPrefillPositions(positions));
    LITERT_ASSERT_OK(kv_cache_update.RunPrefill());
    for (int i = 0; i < num_tokens; ++i) {
      expected.Write(position + i);
    }
    // Also check where the padding of the chunk would land.
    ASSERT_NO_FATAL_FAILURE(
        ExpectKVCachesHold(expected, position, position + chunk));
    position += num_tokens;
  };
  auto prefill_until = [&](int end_position) {
    while (position < end_position) {
      ASSERT_NO_FATAL_FAILURE(
          prefill(std::min(chunk, end_position - position)));
    }
  };
  auto decode = [&]() {
    SCOPED_TRACE(absl::StrCat("Decode at position ", position));
    for (const KVTensor& tensor : KVTensors(model)) {
      ASSERT_NO_FATAL_FAILURE(FillSlice(decode_slices.at(tensor.slice_name),
                                        tensor, /*slice_length=*/1, position,
                                        /*num_tokens=*/1));
    }
    LITERT_ASSERT_OK(kv_cache_update.SetDecodePosition(position));
    LITERT_ASSERT_OK(kv_cache_update.RunDecode());
    expected.Write(position);
    ASSERT_NO_FATAL_FAILURE(
        ExpectKVCachesHold(expected, position, position + 1));
    ++position;
  };
  // Returns the first position of the next pass over the ring buffer.
  auto next_ring_buffer_pass = [&]() { return (position / ring + 1) * ring; };

  // A first turn that fills only part of a prefill chunk.
  ASSERT_NO_FATAL_FAILURE(prefill(37));
  ASSERT_NO_FATAL_FAILURE(decode());
  ASSERT_NO_FATAL_FAILURE(decode());
  // Full prefill chunks through two wrap-arounds of the ring buffer.
  while (position < 2 * ring + 39) {
    ASSERT_NO_FATAL_FAILURE(prefill(chunk));
  }
  // A full prefill chunk that starts 24 slots before the end of the ring
  // buffer.
  ASSERT_NO_FATAL_FAILURE(prefill_until(next_ring_buffer_pass() - 24));
  ASSERT_NO_FATAL_FAILURE(prefill(chunk));
  // Decode steps across the end of the ring buffer.
  ASSERT_NO_FATAL_FAILURE(prefill_until(next_ring_buffer_pass() - 3));
  for (int i = 0; i < 6; ++i) {
    ASSERT_NO_FATAL_FAILURE(decode());
  }
  // A short follow-up turn.
  ASSERT_NO_FATAL_FAILURE(prefill(5));
  // The last position of the global cache.
  position = model.global_cache_length - 1;
  ASSERT_NO_FATAL_FAILURE(decode());
  // Nothing outside the written positions changed in the global caches.
  ASSERT_NO_FATAL_FAILURE(
      ExpectKVCachesHold(expected, 0, model.global_cache_length));
}

INSTANTIATE_TEST_SUITE_P(
    SlidingWindowModels, SlidingWindowModelKVCacheTest,
    ::testing::ValuesIn(kSlidingWindowModels),
    [](const ::testing::TestParamInfo<SlidingWindowModel>& info) {
      return std::string(info.param.name);
    });

}  // namespace
}  // namespace litert::lm
