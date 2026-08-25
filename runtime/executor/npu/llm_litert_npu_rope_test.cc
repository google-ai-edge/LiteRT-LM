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

#include "runtime/executor/npu/llm_litert_npu_rope.h"

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
#include "litert/cc/litert_compiled_model.h"  // from @litert
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

class NpuRopeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto env_expected = ::litert::Environment::Create({});
    ASSERT_TRUE(env_expected.HasValue());
    env_.emplace(std::move(*env_expected));
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

  std::optional<::litert::Environment> env_;
};

TEST_F(NpuRopeTest, NpuRopeCreateFailsWhenCompiledModelNull) {
  InferenceContext ctx;
  auto rope_or = NpuRope::CreateForTest(nullptr, std::move(ctx));
  EXPECT_FALSE(rope_or.ok());
}

TEST_F(NpuRopeTest, NpuRopeSetDecodePosition) {
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

TEST_F(NpuRopeTest, NpuRopeDrafterSetDecodePosition) {
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

TEST_F(NpuRopeTest, NpuRopeSetVerifyPositions) {
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

TEST_F(NpuRopeTest, NpuRopeSetPrefillPositionsAndZeroPads) {
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
