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

#include "runtime/executor/gpu_artisan/ml_drift/mtp_binding.h"

#include <cstdint>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_tensor.h"  // from @ml_drift
#include "ml_drift/common/tensor_handle.h"  // from @ml_drift
#include "runtime/executor/gpu_artisan/ml_drift/speculative_shared_state.h"
#include "runtime/util/test_utils.h"

namespace odml::infra::gpu {
namespace {

using ::testing::ElementsAre;
using ::testing::status::StatusIs;

ml_drift::TensorHandle MakeHandle(uint32_t id) {
  ml_drift::TensorHandle h;
  h.id = id;
  return h;
}

TEST(MtpBindingTest, SuccessBindingExactMapContents) {
  ml_drift::MtpDrafterGraphHandles g;
  g.key_cache = {MakeHandle(101), MakeHandle(102)};
  g.value_cache = {MakeHandle(103), MakeHandle(104)};
  g.params_i32 = MakeHandle(105);
  g.tokens_ids = MakeHandle(106);
  g.input_activations = MakeHandle(107);

  // Sentinels for tensors
  auto* dummy_k_local = reinterpret_cast<ml_drift::GpuSpatialTensor*>(0x1000);
  auto* dummy_k_global = reinterpret_cast<ml_drift::GpuSpatialTensor*>(0x2000);
  auto* dummy_v_local = reinterpret_cast<ml_drift::GpuSpatialTensor*>(0x3000);
  auto* dummy_v_global = reinterpret_cast<ml_drift::GpuSpatialTensor*>(0x4000);
  auto* dummy_params = reinterpret_cast<ml_drift::GpuSpatialTensor*>(0x5000);
  auto* dummy_tokens = reinterpret_cast<ml_drift::GpuSpatialTensor*>(0x6000);
  auto* dummy_hidden = reinterpret_cast<ml_drift::GpuSpatialTensor*>(0x7000);

  MtpSharedInputs in;
  in.kv_shared_start_layer = 15;
  // local = 13, global = 14
  in.key_cache.resize(16, nullptr);
  in.value_cache.resize(16, nullptr);
  in.key_cache[13] = dummy_k_local;
  in.key_cache[14] = dummy_k_global;
  in.value_cache[13] = dummy_v_local;
  in.value_cache[14] = dummy_v_global;
  in.params_i32 = dummy_params;
  in.tokens_ids = dummy_tokens;
  in.last_accepted_hidden_state = dummy_hidden;

  ml_drift::CreateGpuModelInfo info;
  ASSERT_OK(BindMtpDrafterExternals(g, in, &info));

  EXPECT_EQ(info.external_immutable_tensors.size(), 7);
  EXPECT_EQ(info.external_immutable_tensors[101], dummy_k_local);
  EXPECT_EQ(info.external_immutable_tensors[102], dummy_k_global);
  EXPECT_EQ(info.external_immutable_tensors[103], dummy_v_local);
  EXPECT_EQ(info.external_immutable_tensors[104], dummy_v_global);
  EXPECT_EQ(info.external_immutable_tensors[105], dummy_params);
  EXPECT_EQ(info.external_immutable_tensors[106], dummy_tokens);
  EXPECT_EQ(info.external_immutable_tensors[107], dummy_hidden);
}

TEST(MtpBindingTest, InvalidInputsReturnInvalidArgument) {
  auto* dummy = reinterpret_cast<ml_drift::GpuSpatialTensor*>(0x1000);
  auto make_valid_handles = []() {
    ml_drift::MtpDrafterGraphHandles g;
    g.key_cache = {MakeHandle(1), MakeHandle(2)};
    g.value_cache = {MakeHandle(3), MakeHandle(4)};
    g.params_i32 = MakeHandle(5);
    g.tokens_ids = MakeHandle(6);
    g.input_activations = MakeHandle(7);
    return g;
  };
  auto make_valid_input = [&]() {
    MtpSharedInputs in;
    in.kv_shared_start_layer = 2;
    in.key_cache = {dummy, dummy};
    in.value_cache = {dummy, dummy};
    in.params_i32 = dummy;
    in.tokens_ids = dummy;
    in.last_accepted_hidden_state = dummy;
    return in;
  };

  ml_drift::CreateGpuModelInfo info;
  EXPECT_THAT(BindMtpDrafterExternals(make_valid_handles(), make_valid_input(),
                                      nullptr),
              StatusIs(absl::StatusCode::kInvalidArgument));

  for (bool bad_key : {true, false}) {
    auto g = make_valid_handles();
    if (bad_key) {
      g.key_cache = {MakeHandle(1)};
    } else {
      g.value_cache = {MakeHandle(3)};
    }
    EXPECT_THAT(BindMtpDrafterExternals(g, make_valid_input(), &info),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }

  {
    auto in = make_valid_input();
    in.kv_shared_start_layer = 1;
    EXPECT_THAT(BindMtpDrafterExternals(make_valid_handles(), in, &info),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }

  for (bool short_key : {true, false}) {
    auto in = make_valid_input();
    in.kv_shared_start_layer = 15;
    in.key_cache.resize(short_key ? 10 : 16, dummy);
    in.value_cache.resize(short_key ? 16 : 14, dummy);
    EXPECT_THAT(BindMtpDrafterExternals(make_valid_handles(), in, &info),
                StatusIs(absl::StatusCode::kFailedPrecondition));
  }

  for (int field = 0; field < 7; ++field) {
    auto in = make_valid_input();
    ml_drift::GpuSpatialTensor** targets[7] = {
        &in.key_cache[0],
        &in.key_cache[1],
        &in.value_cache[0],
        &in.value_cache[1],
        &in.params_i32,
        &in.tokens_ids,
        &in.last_accepted_hidden_state,
    };
    *targets[field] = nullptr;
    EXPECT_THAT(BindMtpDrafterExternals(make_valid_handles(), in, &info),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }
}

struct FakeHandle {
  ml_drift::GpuSpatialTensor* p;
  ml_drift::GpuSpatialTensor* get() const { return p; }
};

struct FakeCache {
  std::vector<FakeHandle> key_cached_tensors;
  std::vector<FakeHandle> value_cached_tensors;
};

TEST(MtpBindingTest, PopulateMtpCacheTensorsReplacesPreviousContents) {
  auto* dummy1 = reinterpret_cast<ml_drift::GpuSpatialTensor*>(0x1000);
  auto* dummy2 = reinterpret_cast<ml_drift::GpuSpatialTensor*>(0x2000);
  auto* dummy3 = reinterpret_cast<ml_drift::GpuSpatialTensor*>(0x3000);

  FakeCache cache1;
  cache1.key_cached_tensors = {FakeHandle{dummy1}, FakeHandle{dummy2}};
  cache1.value_cached_tensors = {FakeHandle{dummy1}, FakeHandle{dummy2}};

  MtpSharedInputs in;
  PopulateMtpCacheTensors(cache1, &in);
  EXPECT_THAT(in.key_cache, ElementsAre(dummy1, dummy2));
  EXPECT_THAT(in.value_cache, ElementsAre(dummy1, dummy2));

  FakeCache cache2;
  cache2.key_cached_tensors = {FakeHandle{dummy3}};
  cache2.value_cached_tensors = {FakeHandle{dummy3}};

  PopulateMtpCacheTensors(cache2, &in);
  EXPECT_THAT(in.key_cache, ElementsAre(dummy3));
  EXPECT_THAT(in.value_cache, ElementsAre(dummy3));
}

}  // namespace
}  // namespace odml::infra::gpu
