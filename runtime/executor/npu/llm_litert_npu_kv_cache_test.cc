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

#include <cstdint>
#include <cstring>
#include <optional>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/cc/litert_tensor_buffer_types.h"  // from @litert
#include "litert/test/matchers.h"  // from @litert
#include "runtime/executor/npu/llm_litert_npu_compiled_model_executor_utils.h"

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

  ASSERT_TRUE(HWKVCacheUpdate(in_buffers, out_buffers).ok());

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

  ASSERT_TRUE(HWKVCacheUpdate(in_buffers, out_buffers).ok());

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

  ASSERT_TRUE(HWKVCacheUpdate(in_buffers, out_buffers).ok());

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

  EXPECT_FALSE(HWKVCacheUpdate(in_buffers, out_buffers).ok());
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

  ASSERT_TRUE(
      HWKVCacheUpdate(in_buffers, out_buffers, {}, /*enable_swa=*/true).ok());

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

  ASSERT_TRUE(
      HWKVCacheUpdate(in_buffers, out_buffers, {}, /*enable_swa=*/true).ok());

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

  ASSERT_TRUE(
      HWKVCacheUpdate(in_buffers, out_buffers, {}, /*enable_swa=*/true).ok());

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
  EXPECT_FALSE(HWKVCacheUpdate(in_buffers, out_buffers).ok());
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
  EXPECT_FALSE(HWKVCacheUpdate(in_buffers, out_buffers).ok());
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
  EXPECT_FALSE(HWKVCacheUpdate(in_buffers, out_buffers).ok());
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

  ASSERT_TRUE(HWKVCacheUpdate(in_buffers, out_buffers, quant_params).ok());

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

  ASSERT_TRUE(HWKVCacheUpdate(in_buffers, out_buffers).ok());

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

  ASSERT_TRUE(HWKVCacheUpdate(in_buffers, out_buffers).ok());

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
  auto update_or = NpuKVCache::CreateForTest(KVCacheUpdateMethod::kModel,
                                             nullptr, std::move(ctx));
  EXPECT_FALSE(update_or.ok());
}

TEST_F(NpuKVCacheTest,
       NpuKVCacheCreateSucceedsForHwMethodWithoutCompiledModel) {
  InferenceContext ctx;
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto update, NpuKVCache::CreateForTest(KVCacheUpdateMethod::kWH, nullptr,
                                             std::move(ctx)));
  EXPECT_EQ(update.GetMethod(), KVCacheUpdateMethod::kWH);
}

TEST_F(NpuKVCacheTest, NpuKVCacheCommitVerifiedKVCacheSetsPosition) {
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
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto kv_cache,
      NpuKVCache::CreateForTest(KVCacheUpdateMethod::kWH, nullptr,
                                std::move(dummy_ctx), /*kv_quant_params=*/{},
                                /*has_sliding_window_attention=*/false,
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
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto kv_cache,
      NpuKVCache::CreateForTest(KVCacheUpdateMethod::kWH, nullptr,
                                std::move(dummy_ctx), /*kv_quant_params=*/{},
                                /*has_sliding_window_attention=*/false,
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
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto kv_cache,
      NpuKVCache::CreateForTest(KVCacheUpdateMethod::kWH, nullptr,
                                std::move(dummy_ctx), /*kv_quant_params=*/{},
                                /*has_sliding_window_attention=*/false,
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

}  // namespace
}  // namespace litert::lm
