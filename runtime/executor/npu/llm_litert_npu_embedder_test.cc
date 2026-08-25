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

#include "runtime/executor/npu/llm_litert_npu_embedder.h"

#include <cstdint>
#include <cstring>
#include <optional>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/cc/litert_tensor_buffer_types.h"  // from @litert
#include "runtime/executor/npu/llm_litert_npu_compiled_model_executor_utils.h"

namespace litert::lm {
namespace {

using ::litert::ElementType;
using ::litert::Layout;
using ::litert::RankedTensorType;
using ::litert::TensorBuffer;
using ::litert::TensorBufferScopedLock;

std::vector<uint8_t> PackInt4(const std::vector<int8_t>& unpacked) {
  std::vector<uint8_t> packed((unpacked.size() + 1) / 2, 0);
  for (size_t i = 0; i < unpacked.size(); ++i) {
    uint8_t nibble = static_cast<uint8_t>(unpacked[i]) & 0x0F;
    if (i % 2 == 0) {
      packed[i / 2] |= nibble;
    } else {
      packed[i / 2] |= (nibble << 4);
    }
  }
  return packed;
}

class NpuEmbedderTest : public ::testing::Test {
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

TEST_F(NpuEmbedderTest, HWPerLayerEmbeddingLookupFloat32) {
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

TEST_F(NpuEmbedderTest, HWPerLayerEmbeddingLookupInt16) {
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

TEST_F(NpuEmbedderTest, HWPerLayerEmbeddingLookupNeon) {
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

TEST_F(NpuEmbedderTest, HWPerLayerEmbeddingLookupInt8Float32WithScale) {
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

TEST_F(NpuEmbedderTest, WritePleEmbeddingsFloat32) {
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

TEST_F(NpuEmbedderTest, WritePleEmbeddingsInt16) {
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

TEST_F(NpuEmbedderTest, WritePleEmbeddingsInt16InsufficientCapacity) {
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

TEST_F(NpuEmbedderTest, WriteAndPadPleEmbeddingsFloat32) {
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

TEST_F(NpuEmbedderTest, WriteAndPadPleEmbeddingsInt16) {
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

TEST_F(NpuEmbedderTest, WriteAndPadPleEmbeddingsFloat32NoDefault) {
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

TEST_F(NpuEmbedderTest, WriteAndPadPleEmbeddingsInt16NoDefault) {
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

}  // namespace
}  // namespace litert::lm
