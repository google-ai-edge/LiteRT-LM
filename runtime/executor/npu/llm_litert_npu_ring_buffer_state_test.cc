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

#include "runtime/executor/npu/llm_litert_npu_ring_buffer_state.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/cc/litert_tensor_buffer_types.h"  // from @litert
#include "runtime/executor/litert/legacy_map_state.h"
#include "runtime/executor/npu/llm_litert_npu_compiled_model_executor_utils.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

using ::litert::TensorBuffer;
using ::testing::Each;
using ::testing::IsEmpty;
using ::testing::UnorderedElementsAre;
using ::testing::status::StatusIs;

constexpr int kGlobalLen = 16;
constexpr int kRingLen = 8;
constexpr int kDim = 4;
constexpr int kPrefillLen = 4;

class RingBufferStateTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto env = ::litert::Environment::Create({});
    ASSERT_TRUE(env.HasValue());
    env_ = std::make_unique<::litert::Environment>(std::move(*env));
  }

  // Creates a host memory float buffer of shape `dims` filled with `value`.
  TensorBuffer MakeBuffer(const std::vector<int32_t>& dims, float value) {
    ::litert::Dimensions dimensions(dims.begin(), dims.end());
    int64_t num_elements = 1;
    for (int32_t d : dims) num_elements *= d;
    auto buffer = TensorBuffer::CreateManaged(
        *env_, ::litert::TensorBufferType::kHostMemory,
        ::litert::RankedTensorType(::litert::ElementType::Float32,
                                   ::litert::Layout(std::move(dimensions))),
        num_elements * sizeof(float));
    ABSL_CHECK(buffer.HasValue());
    Fill(*buffer, value);
    return std::move(*buffer);
  }

  static void Fill(TensorBuffer& buffer, float value) {
    auto size = buffer.PackedSize();
    ABSL_CHECK(size.HasValue());
    std::vector<float> values(*size / sizeof(float), value);
    ABSL_CHECK(buffer.Write<float>(values).HasValue());
  }

  static std::vector<float> Read(const TensorBuffer& buffer) {
    auto size = buffer.PackedSize();
    ABSL_CHECK(size.HasValue());
    std::vector<float> values(*size / sizeof(float));
    ABSL_CHECK(const_cast<TensorBuffer&>(buffer)
                   .Read<float>(absl::MakeSpan(values))
                   .HasValue());
    return values;
  }

  // KV cache buffers of a model with a global layer (0) and, with
  // `with_ring_buffer`, a ring buffer layer (1); otherwise layer 1 is a local
  // layer as long as the global one.
  absl::flat_hash_map<absl::string_view, TensorBuffer> MakeKvCache(
      bool with_ring_buffer, float value) {
    const int local_len = with_ring_buffer ? kRingLen : kGlobalLen;
    absl::flat_hash_map<absl::string_view, TensorBuffer> buffers;
    buffers.insert_or_assign("kv_cache_k_0",
                             MakeBuffer({1, 1, kGlobalLen, kDim}, value));
    buffers.insert_or_assign("kv_cache_v_0",
                             MakeBuffer({1, 1, kDim, kGlobalLen}, value));
    buffers.insert_or_assign("kv_cache_k_1",
                             MakeBuffer({1, 1, local_len, kDim}, value));
    buffers.insert_or_assign("kv_cache_v_1",
                             MakeBuffer({1, 1, kDim, local_len}, value));
    return buffers;
  }

  static NpuGraphLayout ResolveLayout(
      const absl::flat_hash_map<absl::string_view, TensorBuffer>& buffers) {
    return NpuGraphLayout::Resolve(kPrefillLen, kGlobalLen, buffers);
  }

  std::unique_ptr<::litert::Environment> env_;
};

TEST_F(RingBufferStateTest, SavesAndRestoresOnlyTheRingBuffers) {
  auto kv_cache = MakeKvCache(/*with_ring_buffer=*/true, /*value=*/1.0f);
  const NpuGraphLayout layout = ResolveLayout(kv_cache);
  EXPECT_THAT(RingBufferState::GetNames(layout, kv_cache),
              UnorderedElementsAre("kv_cache_k_1", "kv_cache_v_1"));

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<LegacyMapState> saved,
                       RingBufferState::Save(*env_, layout, kv_cache));
  EXPECT_EQ(saved->buffers().size(), 2);

  // Later steps overwrite all buffers.
  for (auto& [name, buffer] : kv_cache) Fill(buffer, 2.0f);
  ASSERT_OK(RingBufferState::Restore(*saved, layout, kv_cache));

  EXPECT_THAT(Read(kv_cache.at("kv_cache_k_1")), Each(1.0f));
  EXPECT_THAT(Read(kv_cache.at("kv_cache_v_1")), Each(1.0f));
  // The global KV cache is position indexed and not restored.
  EXPECT_THAT(Read(kv_cache.at("kv_cache_k_0")), Each(2.0f));
  EXPECT_THAT(Read(kv_cache.at("kv_cache_v_0")), Each(2.0f));
}

TEST_F(RingBufferStateTest, WithoutRingBufferSavesNothing) {
  auto kv_cache = MakeKvCache(/*with_ring_buffer=*/false, /*value=*/1.0f);
  const NpuGraphLayout layout = ResolveLayout(kv_cache);
  EXPECT_THAT(RingBufferState::GetNames(layout, kv_cache), IsEmpty());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<LegacyMapState> saved,
                       RingBufferState::Save(*env_, layout, kv_cache));
  EXPECT_THAT(saved->buffers(), IsEmpty());
  ASSERT_OK(RingBufferState::Restore(*saved, layout, kv_cache));
}

// A context saved without the ring buffers (e.g. in a context size without
// one) can't restore them.
TEST_F(RingBufferStateTest, RestoreFailsForOtherRingBuffers) {
  auto kv_cache = MakeKvCache(/*with_ring_buffer=*/true, /*value=*/1.0f);
  const NpuGraphLayout layout = ResolveLayout(kv_cache);
  const LegacyMapState empty({});
  EXPECT_THAT(RingBufferState::Restore(empty, layout, kv_cache),
              StatusIs(absl::StatusCode::kFailedPrecondition));

  absl::flat_hash_map<std::string, TensorBuffer> other;
  other.insert_or_assign("kv_cache_k_0",
                         MakeBuffer({1, 1, kGlobalLen, kDim}, 3.0f));
  other.insert_or_assign("kv_cache_k_1",
                         MakeBuffer({1, 1, kRingLen, kDim}, 3.0f));
  EXPECT_THAT(RingBufferState::Restore(LegacyMapState(std::move(other)), layout,
                                       kv_cache),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  // Nothing was copied.
  for (const auto& [name, buffer] : kv_cache) {
    EXPECT_THAT(Read(buffer), Each(1.0f)) << name;
  }
}

TEST_F(RingBufferStateTest, RestoreFailsForOtherShapes) {
  auto kv_cache = MakeKvCache(/*with_ring_buffer=*/true, /*value=*/1.0f);
  const NpuGraphLayout layout = ResolveLayout(kv_cache);
  absl::flat_hash_map<std::string, TensorBuffer> other;
  other.insert_or_assign("kv_cache_k_1",
                         MakeBuffer({1, 1, kRingLen, kDim}, 3.0f));
  other.insert_or_assign("kv_cache_v_1",
                         MakeBuffer({1, 1, kDim, kRingLen / 2}, 3.0f));
  EXPECT_THAT(RingBufferState::Restore(LegacyMapState(std::move(other)), layout,
                                       kv_cache),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  for (const auto& [name, buffer] : kv_cache) {
    EXPECT_THAT(Read(buffer), Each(1.0f)) << name;
  }
}

}  // namespace
}  // namespace litert::lm
