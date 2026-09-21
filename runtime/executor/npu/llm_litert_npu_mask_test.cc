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

#include "runtime/executor/npu/llm_litert_npu_mask.h"

#include <cstdint>
#include <cstring>
#include <optional>
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

namespace litert::lm {
namespace {

using ::litert::ElementType;
using ::litert::Layout;
using ::litert::RankedTensorType;
using ::litert::TensorBuffer;
using ::litert::TensorBufferScopedLock;

class NpuMaskTest : public ::testing::Test {
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

TEST_F(NpuMaskTest, NpuMaskCreateFailsWhenCompiledModelNullForModelMethod) {
  InferenceContext ctx;
  NpuModelGeometry geometry;
  auto mask_or = NpuMask::CreateForTest(MaskUpdateMethod::kModel, nullptr,
                                        std::move(ctx), &geometry);
  EXPECT_FALSE(mask_or.ok());
}

TEST_F(NpuMaskTest, NpuMaskCreateSucceedsForHwMethodWithoutCompiledModel) {
  InferenceContext ctx;
  NpuModelGeometry geometry;
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto mask, NpuMask::CreateForTest(MaskUpdateMethod::kWH, nullptr,
                                        std::move(ctx), &geometry));
  EXPECT_EQ(mask.GetMethod(), MaskUpdateMethod::kWH);
}

TEST_F(NpuMaskTest, NpuMaskSettersAndAccessors) {
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

  NpuModelGeometry geometry;
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto mask, NpuMask::CreateForTest(MaskUpdateMethod::kWH, nullptr,
                                        std::move(ctx), &geometry));

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

TEST_F(NpuMaskTest, NpuMaskCreateForDrafter) {
  absl::flat_hash_map<absl::string_view, TensorBuffer> in_bufs;
  absl::flat_hash_map<absl::string_view, TensorBuffer> out_bufs;
  in_bufs[MaskSignatures::kMaskInputTimeStep] = CreateTensorBufferWithDims(
      std::vector<int32_t>{0}, ElementType::Int32, {1});
  in_bufs[MaskSignatures::kMaskInputTokens] = CreateTensorBufferWithDims(
      std::vector<int32_t>{0}, ElementType::Int32, {1});

  NpuModelGeometry geometry;
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto mask, NpuMask::CreateForDrafter(MaskUpdateMethod::kWH, nullptr,
                                           std::move(in_bufs),
                                           std::move(out_bufs), &geometry));
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

TEST_F(NpuMaskTest, NpuMaskSetPrefillInput) {
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

  NpuModelGeometry geometry;
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto mask, NpuMask::CreateForTest(MaskUpdateMethod::kWH, nullptr,
                                        std::move(ctx), &geometry));

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

TEST_F(NpuMaskTest, HWMaskUpdateInt8) {
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

TEST_F(NpuMaskTest, HWMaskUpdateInt16) {
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

TEST_F(NpuMaskTest, HWMaskUpdateGemma3Prefill) {
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

TEST_F(NpuMaskTest, HWMaskUpdateGemma3Decode) {
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

TEST_F(NpuMaskTest, HWMaskUpdateWithValidMask) {
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

TEST_F(NpuMaskTest, HWMaskUpdateSWADecode_TimestepSmallerLocalWindow) {
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

TEST_F(NpuMaskTest, HWMaskUpdateSWADecode_TimestepLargerLocalWindow) {
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

TEST_F(NpuMaskTest, HWMaskUpdateCapacity4096Uses512AsWindow) {
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

TEST_F(NpuMaskTest, HWMaskUpdateSWAPrefill) {
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

TEST_F(NpuMaskTest, HWMaskUpdateSWAPrefill_TimestepSmallerLocalWindow) {
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

TEST_F(NpuMaskTest, HWMaskUpdateSWAPrefill_TimestepLargerLocalWindow) {
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

TEST_F(NpuMaskTest, HWMaskUpdateFloat16) {
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

TEST_F(NpuMaskTest, HWMaskUpdateBFloat16) {
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

TEST_F(NpuMaskTest, HWMaskUpdateCustomSlidingWindowSize) {
  int seq_q = 1;
  int seq_k = 256;
  int time_step = 100;
  int64_t custom_window = 32;

  absl::flat_hash_map<absl::string_view, TensorBuffer> in_buffers;
  std::vector<int32_t> time_step_data = {time_step};
  in_buffers.emplace("time_step",
                     CreateTensorBuffer(time_step_data, ElementType::Int32));

  absl::flat_hash_map<absl::string_view, TensorBuffer> out_buffers;
  std::vector<int8_t> mask_data(seq_q * seq_k, 0);
  out_buffers.emplace("mask_local",
                      CreateTensorBufferWithDims(mask_data, ElementType::Int8,
                                                 {1, seq_q, seq_k}));

  LITERT_ASSERT_OK(HWMaskUpdate(in_buffers, out_buffers, custom_window));

  auto local_lock_expected = TensorBufferScopedLock::Create<int8_t>(
      out_buffers.at("mask_local"), TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(local_lock_expected.HasValue());
  auto& local_lock = *local_lock_expected;

  // Window is 32 tokens: for effective_pos = 100, attends to [100 - 32 + 1,
  // 100] = [69, 100]
  for (int k = 0; k < seq_k; ++k) {
    if (k >= 100 - custom_window + 1 && k <= 100) {
      EXPECT_EQ(local_lock.second[k], 127) << "k=" << k;
    } else {
      EXPECT_EQ(local_lock.second[k], -128) << "k=" << k;
    }
  }
}

// =============================================================================
// Ringbuffer mask corner cases.
//
// The configurations below mirror the two lightweight verification models:
//   Config A: gemma3_1b_1024global_256local_128prefill
//   Config B: gemma3_1b_2048global_768local_512prefill
//
// The local mask is indexed by *physical* ring slot while the attention window
// is defined over *logical* token positions, so the mask has to invert the same
// slot = token % capacity mapping that HWKVCacheUpdate uses on the writer side.
// RingSlotToken below is an independent statement of that mapping: if the mask
// and the cache update ever disagree, these tests fail.
// =============================================================================

constexpr float kValidMaskValue = 0.0f;
constexpr float kMaskedMaskValue = -1e9f;

// Returns the logical token index held in physical ring slot `slot` after
// tokens [0, time_step) have been written to a ring of `capacity` slots, or -1
// if the slot has never been written.
int64_t RingSlotToken(int64_t slot, int64_t time_step, int64_t capacity) {
  int64_t token = -1;
  for (int64_t t = slot; t < time_step; t += capacity) {
    token = t;
  }
  return token;
}

struct SwaConfig {
  absl::string_view name;
  int local_capacity;
  int global_capacity;
};

// Sweeps the time step across several ring wraps and checks every physical slot
// of the local mask against the ring reference model. The interesting time
// steps are the ones on either side of a wrap, where the mapping from slot to
// token changes discontinuously.
TEST_F(NpuMaskTest, HWMaskUpdateRingMaskMatchesPhysicalCacheSlots) {
  const SwaConfig kConfigs[] = {
      {"1024global_256local", 256, 1024},
      {"2048global_768local", 768, 2048},
  };

  for (const SwaConfig& config : kConfigs) {
    SCOPED_TRACE(config.name);
    const int64_t window = config.local_capacity;
    const int capacity = config.local_capacity;

    for (int time_step : {1, capacity - 1, capacity, capacity + 1, 2 * capacity,
                          2 * capacity + 1, 3 * capacity + 7}) {
      if (time_step >= config.global_capacity) {
        continue;  // Beyond the global cache the session would have ended.
      }
      SCOPED_TRACE(absl::StrCat("time_step=", time_step));

      constexpr int kSeqQ = 1;
      const int seq_k_local = capacity + kSeqQ;
      const int seq_k_global = config.global_capacity + kSeqQ;

      absl::flat_hash_map<absl::string_view, TensorBuffer> in_buffers;
      const std::vector<int32_t> time_step_data = {time_step};
      in_buffers.emplace(
          "time_step", CreateTensorBuffer(time_step_data, ElementType::Int32));

      absl::flat_hash_map<absl::string_view, TensorBuffer> out_buffers;
      out_buffers.emplace("mask_local",
                          CreateTensorBufferWithDims(
                              std::vector<float>(kSeqQ * seq_k_local, 0.0f),
                              ElementType::Float32, {1, kSeqQ, seq_k_local}));
      out_buffers.emplace("mask_global",
                          CreateTensorBufferWithDims(
                              std::vector<float>(kSeqQ * seq_k_global, 0.0f),
                              ElementType::Float32, {1, kSeqQ, seq_k_global}));

      LITERT_ASSERT_OK(HWMaskUpdate(in_buffers, out_buffers, window));

      auto local_lock_expected = TensorBufferScopedLock::Create<float>(
          out_buffers.at("mask_local"), TensorBuffer::LockMode::kRead);
      ASSERT_TRUE(local_lock_expected.HasValue());
      const float* local_ptr = local_lock_expected->second;

      for (int k = 0; k < capacity; ++k) {
        const int64_t token = RingSlotToken(k, time_step, capacity);
        const bool expect_valid = token >= 0 && token >= time_step - window + 1;
        EXPECT_EQ(local_ptr[k],
                  expect_valid ? kValidMaskValue : kMaskedMaskValue)
            << "local slot " << k << " holds token " << token;
      }
      // The token being decoded lives in the batch slot appended after the
      // ring and is always attended to.
      EXPECT_EQ(local_ptr[capacity], kValidMaskValue);

      auto global_lock_expected = TensorBufferScopedLock::Create<float>(
          out_buffers.at("mask_global"), TensorBuffer::LockMode::kRead);
      ASSERT_TRUE(global_lock_expected.HasValue());
      const float* global_ptr = global_lock_expected->second;
      for (int k = 0; k < config.global_capacity; ++k) {
        EXPECT_EQ(global_ptr[k],
                  k < time_step ? kValidMaskValue : kMaskedMaskValue)
            << "global slot " << k;
      }
      EXPECT_EQ(global_ptr[config.global_capacity], kValidMaskValue);
    }
  }
}

// A local cache buffer padded beyond the attention window: 768 physical slots
// but only the newest 512 tokens may be attended to. Nothing in the tensor
// shapes expresses this, so the window has to come from the metadata
// (attention_mask_settings.sliding_window_size, or the local state buffer's
// minimum_sequence_length). Without it the window defaults to the full
// capacity and 256 stale tokens leak into the attention.
TEST_F(NpuMaskTest, HWMaskUpdateWindowSmallerThanLocalRingCapacity) {
  constexpr int kLocalCapacity = 768;
  constexpr int kGlobalCapacity = 2048;
  constexpr int64_t kWindow = 512;
  constexpr int kTimeStep = 1000;
  constexpr int kSeqQ = 1;

  const int seq_k_local = kLocalCapacity + kSeqQ;
  const int seq_k_global = kGlobalCapacity + kSeqQ;

  absl::flat_hash_map<absl::string_view, TensorBuffer> in_buffers;
  const std::vector<int32_t> time_step_data = {kTimeStep};
  in_buffers.emplace("time_step",
                     CreateTensorBuffer(time_step_data, ElementType::Int32));

  absl::flat_hash_map<absl::string_view, TensorBuffer> out_buffers;
  out_buffers.emplace("mask_local",
                      CreateTensorBufferWithDims(
                          std::vector<float>(kSeqQ * seq_k_local, 0.0f),
                          ElementType::Float32, {1, kSeqQ, seq_k_local}));
  out_buffers.emplace("mask_global",
                      CreateTensorBufferWithDims(
                          std::vector<float>(kSeqQ * seq_k_global, 0.0f),
                          ElementType::Float32, {1, kSeqQ, seq_k_global}));

  LITERT_ASSERT_OK(HWMaskUpdate(in_buffers, out_buffers, kWindow));

  auto local_lock_expected = TensorBufferScopedLock::Create<float>(
      out_buffers.at("mask_local"), TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(local_lock_expected.HasValue());
  const float* local_ptr = local_lock_expected->second;

  for (int k = 0; k < kLocalCapacity; ++k) {
    const int64_t token = RingSlotToken(k, kTimeStep, kLocalCapacity);
    const bool expect_valid = token >= kTimeStep - kWindow + 1;
    EXPECT_EQ(local_ptr[k], expect_valid ? kValidMaskValue : kMaskedMaskValue)
        << "local slot " << k << " holds token " << token;
  }

  // Slots 0..231 hold tokens 768..999 and slots 232..767 hold tokens 232..767,
  // so the in window range [489, 999] covers slots [0, 231] and [489, 767]. The
  // 257 slots in between are physically resident but out of window.
  EXPECT_EQ(local_ptr[231], kValidMaskValue);   // token 999
  EXPECT_EQ(local_ptr[232], kMaskedMaskValue);  // token 232
  EXPECT_EQ(local_ptr[488], kMaskedMaskValue);  // token 488
  EXPECT_EQ(local_ptr[489], kValidMaskValue);   // token 489
}

// A prefill chunk issued after the ring has already wrapped. Each query row has
// its own window, so the oldest visible slot moves as the row advances, and it
// does not sit at slot 0.
TEST_F(NpuMaskTest, HWMaskUpdateRingMaskDuringPrefillChunkAfterWrap) {
  constexpr int kLocalCapacity = 768;
  constexpr int kGlobalCapacity = 2048;
  constexpr int kPrefillSeq = 512;
  constexpr int64_t kWindow = kLocalCapacity;
  // Two 512 token chunks have already been prefilled, so the 768 slot ring has
  // wrapped once and holds tokens 256..1023.
  constexpr int kTimeStep = 1024;

  const int seq_k_local = kLocalCapacity + kPrefillSeq;
  const int seq_k_global = kGlobalCapacity + kPrefillSeq;

  absl::flat_hash_map<absl::string_view, TensorBuffer> in_buffers;
  const std::vector<int32_t> time_step_data = {kTimeStep};
  in_buffers.emplace("time_step",
                     CreateTensorBuffer(time_step_data, ElementType::Int32));

  absl::flat_hash_map<absl::string_view, TensorBuffer> out_buffers;
  out_buffers.emplace("mask_local",
                      CreateTensorBufferWithDims(
                          std::vector<float>(kPrefillSeq * seq_k_local, 0.0f),
                          ElementType::Float32, {1, kPrefillSeq, seq_k_local}));
  out_buffers.emplace(
      "mask_global", CreateTensorBufferWithDims(
                         std::vector<float>(kPrefillSeq * seq_k_global, 0.0f),
                         ElementType::Float32, {1, kPrefillSeq, seq_k_global}));

  LITERT_ASSERT_OK(HWMaskUpdate(in_buffers, out_buffers, kWindow));

  auto local_lock_expected = TensorBufferScopedLock::Create<float>(
      out_buffers.at("mask_local"), TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(local_lock_expected.HasValue());
  const float* local_ptr = local_lock_expected->second;

  for (int q : {0, 255, kPrefillSeq - 1}) {
    SCOPED_TRACE(absl::StrCat("q=", q));
    const int64_t logical_pos = kTimeStep + q;
    const float* row = local_ptr + q * seq_k_local;

    for (int k = 0; k < kLocalCapacity; ++k) {
      const int64_t token = RingSlotToken(k, kTimeStep, kLocalCapacity);
      const bool expect_valid =
          token >= 0 && token >= logical_pos - kWindow + 1;
      EXPECT_EQ(row[k], expect_valid ? kValidMaskValue : kMaskedMaskValue)
          << "local slot " << k << " holds token " << token;
    }
    // Causal within the chunk being prefilled.
    for (int k_rel = 0; k_rel < kPrefillSeq; ++k_rel) {
      EXPECT_EQ(row[kLocalCapacity + k_rel],
                k_rel <= q ? kValidMaskValue : kMaskedMaskValue)
          << "batch slot " << k_rel;
    }
  }

  // For the first row of the chunk the oldest in window token is 257, which
  // lives in slot 257. Slot 256 still holds token 256 and must be masked, even
  // though slot 0 (holding token 768) is visible.
  EXPECT_EQ(local_ptr[0], kValidMaskValue);
  EXPECT_EQ(local_ptr[256], kMaskedMaskValue);
  EXPECT_EQ(local_ptr[257], kValidMaskValue);
}

// =============================================================================
// NpuModelGeometry overload coverage.
//
// Production code reaches HWMaskUpdate exclusively through the
// NpuModelGeometry overload, but every other test in this file calls the
// explicit-parameter overload, so the unpacking in the geometry overload is
// otherwise unexercised.
// =============================================================================

// The geometry overload must be a pure repackaging of the explicit parameters.
// Running both against identical inputs and comparing the full mask pins that
// equivalence, so a future change to the unpacking cannot silently alter
// behaviour for production callers while leaving every existing test green.
TEST_F(NpuMaskTest, HWMaskUpdateGeometryOverloadMatchesExplicitParameters) {
  constexpr int kLocalCapacity = 256;
  constexpr int kGlobalCapacity = 1024;
  constexpr int kWindow = 256;
  constexpr int kTimeStep = 300;
  constexpr int kSeqQ = 1;

  const int seq_k_local = kLocalCapacity + kSeqQ;
  const int seq_k_global = kGlobalCapacity + kSeqQ;

  // Builds a fresh input/output buffer pair so the two invocations cannot
  // observe each other's writes.
  auto make_buffers =
      [&](absl::flat_hash_map<absl::string_view, TensorBuffer>& in_buffers,
          absl::flat_hash_map<absl::string_view, TensorBuffer>& out_buffers) {
        const std::vector<int32_t> time_step_data = {kTimeStep};
        in_buffers.emplace("time_step", CreateTensorBuffer(time_step_data,
                                                           ElementType::Int32));
        out_buffers.emplace("mask_local",
                            CreateTensorBufferWithDims(
                                std::vector<float>(kSeqQ * seq_k_local, 0.0f),
                                ElementType::Float32, {1, kSeqQ, seq_k_local}));
        out_buffers.emplace(
            "mask_global", CreateTensorBufferWithDims(
                               std::vector<float>(kSeqQ * seq_k_global, 0.0f),
                               ElementType::Float32, {1, kSeqQ, seq_k_global}));
      };

  auto read_mask =
      [](absl::flat_hash_map<absl::string_view, TensorBuffer>& out_buffers,
         absl::string_view name, int count) {
        auto lock_expected = TensorBufferScopedLock::Create<float>(
            out_buffers.at(name), TensorBuffer::LockMode::kRead);
        ABSL_CHECK(lock_expected.HasValue());
        return std::vector<float>(lock_expected->second,
                                  lock_expected->second + count);
      };

  absl::flat_hash_map<absl::string_view, TensorBuffer> explicit_in;
  absl::flat_hash_map<absl::string_view, TensorBuffer> explicit_out;
  make_buffers(explicit_in, explicit_out);
  LITERT_ASSERT_OK(HWMaskUpdate(explicit_in, explicit_out, kWindow,
                                /*uses_ringbuffer=*/true, kGlobalCapacity,
                                kLocalCapacity));

  NpuModelGeometry geometry;
  geometry.global_cache_length = kGlobalCapacity;
  geometry.local_cache_length = kLocalCapacity;
  geometry.sliding_window_size = kWindow;
  geometry.uses_ringbuffer = true;

  absl::flat_hash_map<absl::string_view, TensorBuffer> geometry_in;
  absl::flat_hash_map<absl::string_view, TensorBuffer> geometry_out;
  make_buffers(geometry_in, geometry_out);
  LITERT_ASSERT_OK(HWMaskUpdate(geometry_in, geometry_out, geometry));

  EXPECT_EQ(read_mask(geometry_out, "mask_local", kSeqQ * seq_k_local),
            read_mask(explicit_out, "mask_local", kSeqQ * seq_k_local));
  EXPECT_EQ(read_mask(geometry_out, "mask_global", kSeqQ * seq_k_global),
            read_mask(explicit_out, "mask_global", kSeqQ * seq_k_global));
}

// A geometry with no resolved local cache length falls back to the global
// cache length for the local capacity. This happens whenever
// ResolveModelGeometry cannot find a local state buffer, and it silently
// claims the local ring is as large as the global cache, so the behaviour is
// worth pinning explicitly rather than leaving it implied by
// `local_cache_length.value_or(global_cache_length)`.
TEST_F(NpuMaskTest,
       HWMaskUpdateGeometryUnsetLocalCacheLengthFallsBackToGlobal) {
  constexpr int kGlobalCapacity = 512;
  constexpr int kTimeStep = 100;
  constexpr int kSeqQ = 1;

  // Both masks are sized for the global capacity, which is what the fallback
  // assumes the local ring to be.
  const int seq_k = kGlobalCapacity + kSeqQ;

  absl::flat_hash_map<absl::string_view, TensorBuffer> in_buffers;
  const std::vector<int32_t> time_step_data = {kTimeStep};
  in_buffers.emplace("time_step",
                     CreateTensorBuffer(time_step_data, ElementType::Int32));

  absl::flat_hash_map<absl::string_view, TensorBuffer> out_buffers;
  out_buffers.emplace(
      "mask_local",
      CreateTensorBufferWithDims(std::vector<float>(kSeqQ * seq_k, 0.0f),
                                 ElementType::Float32, {1, kSeqQ, seq_k}));
  out_buffers.emplace(
      "mask_global",
      CreateTensorBufferWithDims(std::vector<float>(kSeqQ * seq_k, 0.0f),
                                 ElementType::Float32, {1, kSeqQ, seq_k}));

  NpuModelGeometry geometry;
  geometry.global_cache_length = kGlobalCapacity;
  geometry.local_cache_length = std::nullopt;
  geometry.sliding_window_size = kGlobalCapacity;
  geometry.uses_ringbuffer = true;

  LITERT_ASSERT_OK(HWMaskUpdate(in_buffers, out_buffers, geometry));

  auto local_lock_expected = TensorBufferScopedLock::Create<float>(
      out_buffers.at("mask_local"), TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(local_lock_expected.HasValue());
  const float* local_ptr = local_lock_expected->second;

  // Treated as a 512 slot ring that has not yet wrapped at time step 100.
  for (int k = 0; k < kGlobalCapacity; ++k) {
    EXPECT_EQ(local_ptr[k], k < kTimeStep ? kValidMaskValue : kMaskedMaskValue)
        << "local slot " << k;
  }
  EXPECT_EQ(local_ptr[kGlobalCapacity], kValidMaskValue);
}

// =============================================================================
// Regression tests.
//
// Each test below pins a failure mode that HWMaskUpdate and
// UpdateInterleavedSWAMasks have to keep ruling out: a mask that is silently
// left unwritten, and a write past the end of the mask tensor. Neither
// announces itself in production -- no error is returned and no sanitizer
// fires -- so both are asserted explicitly here.
// =============================================================================

// A model whose geometry says uses_ringbuffer = true but which only binds a
// single mask buffer (MTP/drafter and global-only signatures both do this).
//
// HWMaskUpdate sets `is_interleaved_swa = uses_ringbuffer` without checking
// whether a local mask buffer actually exists, so it routes to
// UpdateInterleavedSWAMasks even when only the global mask is bound. The query
// length it forwards therefore has to come from whichever mask is present:
// reading it from the local mask alone leaves it at 0, because seq_q_local is
// only assigned inside `if (mask_local_buf)`. Every fill loop is bounded by
// seq_q, so a zero there writes nothing at all and returns OkStatus over
// whatever the buffer previously held. This test seeds the buffer with a
// sentinel so that outcome is visible rather than silent.
TEST_F(NpuMaskTest,
       HWMaskUpdateGeometryRingbufferWithOnlyGlobalMaskStillWrites) {
  constexpr int kGlobalCapacity = 1024;
  constexpr int kLocalCapacity = 256;
  constexpr int kTimeStep = 300;
  constexpr int kSeqQ = 1;
  constexpr float kSentinel = 42.0f;

  const int seq_k_global = kGlobalCapacity + kSeqQ;

  absl::flat_hash_map<absl::string_view, TensorBuffer> in_buffers;
  const std::vector<int32_t> time_step_data = {kTimeStep};
  in_buffers.emplace("time_step",
                     CreateTensorBuffer(time_step_data, ElementType::Int32));

  absl::flat_hash_map<absl::string_view, TensorBuffer> out_buffers;
  out_buffers.emplace("mask_global",
                      CreateTensorBufferWithDims(
                          std::vector<float>(kSeqQ * seq_k_global, kSentinel),
                          ElementType::Float32, {1, kSeqQ, seq_k_global}));

  NpuModelGeometry geometry;
  geometry.global_cache_length = kGlobalCapacity;
  geometry.local_cache_length = kLocalCapacity;
  geometry.sliding_window_size = kLocalCapacity;
  geometry.uses_ringbuffer = true;

  LITERT_ASSERT_OK(HWMaskUpdate(in_buffers, out_buffers, geometry));

  auto global_lock_expected = TensorBufferScopedLock::Create<float>(
      out_buffers.at("mask_global"), TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(global_lock_expected.HasValue());
  const float* global_ptr = global_lock_expected->second;

  // The global mask must have been populated, not left at the sentinel.
  for (int k = 0; k < kGlobalCapacity; ++k) {
    EXPECT_EQ(global_ptr[k], k < kTimeStep ? kValidMaskValue : kMaskedMaskValue)
        << "global slot " << k;
  }
  EXPECT_EQ(global_ptr[kGlobalCapacity], kValidMaskValue);
}

// A capacity that exceeds the mask tensor must not be written past the end of
// that tensor.
//
// The ring branch of FillMaskSingle iterates `for (k = 0; k < capacity; ++k)`
// and writes `row[k]` with no clamp against seq_k; only the batch loop below it
// guards with `if (k >= seq_k) break;`. The capacity it trusts comes from
// NpuModelGeometry -- UpdateInterleavedSWAMasks substitutes `seq_k - seq_q`
// only when the supplied capacity is <= 0 -- and geometry.global_cache_length
// is a token budget rather than a tensor dimension, so it can legitimately
// exceed the mask width. UpdateInterleavedSWAMasks therefore has to reject a
// capacity that does not fit the bound buffer instead of trusting it.
//
// To demonstrate that safely, the local mask is backed by an allocation far
// larger than its declared tensor width and the slack is filled with a
// sentinel. A correct implementation touches only the first seq_k elements, so
// the sentinel must survive. Note that a sanitizer does NOT catch this: the
// backing store is a LiteRT managed host buffer, not an instrumented malloc,
// so the overrun is invisible to ASAN and has to be asserted explicitly.
TEST_F(NpuMaskTest, HWMaskUpdateGeometryRejectsCapacityLargerThanMaskTensor) {
  constexpr int kLocalCapacity = 4096;  // Far wider than the mask below.
  constexpr int kGlobalCapacity = 4096;
  // Must be >= kLocalCapacity so FillMaskSingle takes the aging branch. Below
  // capacity the `t_k < time_step` causality guard suppresses every write past
  // the time step, which masks the missing bound.
  constexpr int kTimeStep = 5000;
  constexpr int kSeqQ = 1;

  constexpr int kSeqKLocal = 513;  // Declared local mask: 512 slots + 1 batch.
  constexpr int kLocalAllocation = 8192;  // Backing store, with slack.
  constexpr float kSentinel = 42.0f;
  const int seq_k_global = kGlobalCapacity + kSeqQ;

  absl::flat_hash_map<absl::string_view, TensorBuffer> in_buffers;
  const std::vector<int32_t> time_step_data = {kTimeStep};
  in_buffers.emplace("time_step",
                     CreateTensorBuffer(time_step_data, ElementType::Int32));

  // Declared as {1, 1, 513} but backed by 8192 floats of sentinel, so any write
  // beyond the declared width lands in the slack instead of corrupting the
  // heap, and is directly observable.
  absl::flat_hash_map<absl::string_view, TensorBuffer> out_buffers;
  out_buffers.emplace("mask_local",
                      CreateTensorBufferWithDims(
                          std::vector<float>(kLocalAllocation, kSentinel),
                          ElementType::Float32, {1, kSeqQ, kSeqKLocal}));
  out_buffers.emplace("mask_global",
                      CreateTensorBufferWithDims(
                          std::vector<float>(kSeqQ * seq_k_global, 0.0f),
                          ElementType::Float32, {1, kSeqQ, seq_k_global}));

  NpuModelGeometry geometry;
  geometry.global_cache_length = kGlobalCapacity;
  geometry.local_cache_length = kLocalCapacity;
  geometry.sliding_window_size = kLocalCapacity;
  geometry.uses_ringbuffer = true;

  // Either reject the mismatch outright or clamp to the tensor; silently
  // writing past the declared width is not acceptable.
  EXPECT_FALSE(HWMaskUpdate(in_buffers, out_buffers, geometry).ok());

  auto local_lock_expected = TensorBufferScopedLock::Create<float>(
      out_buffers.at("mask_local"), TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(local_lock_expected.HasValue());
  const float* local_ptr = local_lock_expected->second;

  for (int k = kSeqKLocal; k < kLocalAllocation; ++k) {
    ASSERT_EQ(local_ptr[k], kSentinel)
        << "wrote past the declared mask width at index " << k;
  }
}

// A mask whose key dimension is narrower than its query dimension must be
// rejected rather than turned into a negative capacity.
//
// When the geometry supplies no capacity, UpdateInterleavedSWAMasks falls back
// to `seq_k - seq_q`, which goes negative for such a mask. The width check
// alone cannot catch that: substituting the fallback reduces
// `capacity + seq_q <= seq_k` to `seq_k <= seq_k`, which always holds. A
// negative capacity then reaches FillMaskSingle, whose batch loop writes
// `row[capacity + k_rel]` and only guards the upper end with
// `if (k >= seq_k) break;`, so the first row is written *below* the start of
// the buffer.
//
// Such a mask is malformed, but that is exactly what these checks are for, and
// the write is upstream of the allocation rather than merely past its end.
TEST_F(NpuMaskTest, HWMaskUpdateGeometryRejectsKeyDimNarrowerThanQueryDim) {
  constexpr int kSeqQ = 8;
  constexpr int kSeqKLocal = 4;  // Narrower than the query dimension.
  constexpr int kTimeStep = 3;

  absl::flat_hash_map<absl::string_view, TensorBuffer> in_buffers;
  const std::vector<int32_t> time_step_data = {kTimeStep};
  in_buffers.emplace("time_step",
                     CreateTensorBuffer(time_step_data, ElementType::Int32));

  absl::flat_hash_map<absl::string_view, TensorBuffer> out_buffers;
  out_buffers.emplace("mask_local",
                      CreateTensorBufferWithDims(
                          std::vector<float>(kSeqQ * kSeqKLocal, 0.0f),
                          ElementType::Float32, {1, kSeqQ, kSeqKLocal}));

  // No local capacity, so UpdateInterleavedSWAMasks takes the seq_k - seq_q
  // fallback.
  NpuModelGeometry geometry;
  geometry.local_cache_length = 0;
  geometry.sliding_window_size = 0;
  geometry.uses_ringbuffer = true;

  EXPECT_FALSE(HWMaskUpdate(in_buffers, out_buffers, geometry).ok());
}

}  // namespace
}  // namespace litert::lm
