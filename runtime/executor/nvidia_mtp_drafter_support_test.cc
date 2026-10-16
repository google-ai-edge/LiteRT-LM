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

#include "runtime/executor/nvidia_mtp_drafter_support.h"

#include <cstdint>
#include <cstdlib>
#include <filesystem>  // NOLINT(build/c++17)
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_common.h"  // from @litert
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_opaque_options.h"  // from @litert
#include "litert/cc/litert_options.h"  // from @litert
#include "litert/cc/litert_tensor_buffer_requirements.h"  // from @litert
#include "litert/cc/litert_tensor_buffer_types.h"  // from @litert
#include "litert/test/matchers.h"  // from @litert
#include "litert/vendors/nvidia/cache_layout.h"  // from @litert
#include "runtime/executor/executor_settings_base.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

using ::testing::status::StatusIs;

constexpr auto kCudaBuffer =
    static_cast<TensorBufferType>(nvidia::kNvidiaCudaTensorBufferType);

TEST(NvidiaMtpDrafterSupportTest, RecognizesTransposedValueCache) {
  const Layout layout(Dimensions{1, 8, 256, 1152});
  const std::vector<TensorBufferType> buffer_types = {
      kCudaBuffer, TensorBufferType::kHostMemory};
  const std::vector<uint32_t> strides = {2359296, 294912, 1, 256};
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto requirements,
      TensorBufferRequirements::Create(buffer_types, 4718592, strides));

  ASSERT_OK_AND_ASSIGN(bool transposed,
                       UsesNvidiaTransposedValueCache(layout, requirements));
  EXPECT_TRUE(transposed);
}

TEST(NvidiaMtpDrafterSupportTest, LeavesDenseCudaValueCacheUnchanged) {
  const Layout layout(Dimensions{1, 8, 256, 1152});
  const std::vector<TensorBufferType> buffer_types = {kCudaBuffer};
  for (const auto& strides :
       std::vector<std::vector<uint32_t>>{{}, {2359296, 294912, 1152, 1}}) {
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto requirements,
        TensorBufferRequirements::Create(buffer_types, 4718592, strides));

    ASSERT_OK_AND_ASSIGN(bool transposed,
                         UsesNvidiaTransposedValueCache(layout, requirements));
    EXPECT_FALSE(transposed);
  }
}

TEST(NvidiaMtpDrafterSupportTest, LeavesOtherBackendStridesUnchanged) {
  const Layout layout(Dimensions{1, 8, 256, 1152});
  const std::vector<TensorBufferType> buffer_types = {
      TensorBufferType::kOpenClBuffer};
  const std::vector<uint32_t> strides = {3000000, 375000, 1200, 1};
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto requirements,
      TensorBufferRequirements::Create(buffer_types, 6000000, strides));

  ASSERT_OK_AND_ASSIGN(bool transposed,
                       UsesNvidiaTransposedValueCache(layout, requirements));
  EXPECT_FALSE(transposed);
}

TEST(NvidiaMtpDrafterSupportTest, RejectsUnsupportedCudaValueCacheStrides) {
  const Layout layout(Dimensions{1, 8, 256, 1152});
  const std::vector<TensorBufferType> buffer_types = {kCudaBuffer};
  for (const auto& strides : std::vector<std::vector<uint32_t>>{
           {2359296, 294912, 2, 256}, {294912, 1, 256}}) {
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto requirements,
        TensorBufferRequirements::Create(buffer_types, 4718592, strides));

    EXPECT_THAT(UsesNvidiaTransposedValueCache(layout, requirements),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }
}

TEST(NvidiaMtpDrafterSupportTest, ReadOnlyValueCacheOptionsOwnSortedNames) {
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto options,
      CreateNvidiaReadOnlyValueCacheOptions({"kv_cache_v_2", "kv_cache_v_0"}));
  LITERT_ASSERT_OK_AND_ASSIGN(auto identifier, options.GetIdentifier());
  EXPECT_EQ(std::string(identifier.data(), identifier.size()),
            LITERT_NVIDIA_READ_ONLY_VALUE_CACHE_OPTIONS_ID);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto* payload, options.GetData<LiteRtNvidiaReadOnlyValueCacheOptions>());
  ASSERT_EQ(payload->num_inputs, 2u);
  EXPECT_STREQ(payload->input_names[0], "kv_cache_v_0");
  EXPECT_STREQ(payload->input_names[1], "kv_cache_v_2");
}

TEST(NvidiaMtpDrafterSupportTest, ReadOnlyValueCacheOptionsHashTheNameSet) {
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto options,
      CreateNvidiaReadOnlyValueCacheOptions({"kv_cache_v_2", "kv_cache_v_0"}));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto same_set,
      CreateNvidiaReadOnlyValueCacheOptions({"kv_cache_v_0", "kv_cache_v_2"}));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto other_set, CreateNvidiaReadOnlyValueCacheOptions({"kv_cache_v_1"}));
  LITERT_ASSERT_OK_AND_ASSIGN(auto hash, options.Hash());
  LITERT_ASSERT_OK_AND_ASSIGN(auto same_hash, same_set.Hash());
  LITERT_ASSERT_OK_AND_ASSIGN(auto other_hash, other_set.Hash());
  EXPECT_EQ(hash, same_hash);
  EXPECT_NE(hash, other_hash);
}

TEST(NvidiaMtpDrafterSupportTest, SharedCacheInputsOwnSortedNames) {
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto options,
      CreateNvidiaSharedCacheInputs({"kv_cache_v_2", "kv_cache_k_2"}));
  LITERT_ASSERT_OK_AND_ASSIGN(auto identifier, options.GetIdentifier());
  EXPECT_EQ(std::string(identifier.data(), identifier.size()),
            LITERT_NVIDIA_SHARED_CACHE_INPUTS_ID);
  LITERT_ASSERT_OK_AND_ASSIGN(auto* payload,
                              options.GetData<LiteRtNvidiaSharedCacheInputs>());
  ASSERT_EQ(payload->num_inputs, 2u);
  EXPECT_STREQ(payload->input_names[0], "kv_cache_k_2");
  EXPECT_STREQ(payload->input_names[1], "kv_cache_v_2");
}

TEST(NvidiaMtpDrafterSupportTest, SharedCacheInputsHashTheNameSet) {
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto options,
      CreateNvidiaSharedCacheInputs({"kv_cache_v_2", "kv_cache_k_2"}));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto same_set,
      CreateNvidiaSharedCacheInputs({"kv_cache_k_2", "kv_cache_v_2"}));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto read_only,
      CreateNvidiaReadOnlyValueCacheOptions({"kv_cache_k_2", "kv_cache_v_2"}));
  LITERT_ASSERT_OK_AND_ASSIGN(auto hash, options.Hash());
  LITERT_ASSERT_OK_AND_ASSIGN(auto same_hash, same_set.Hash());
  LITERT_ASSERT_OK_AND_ASSIGN(auto read_only_hash, read_only.Hash());
  EXPECT_EQ(hash, same_hash);
  // The same names under the other identifier hash the same: the identifier
  // tells them apart in the options chain.
  EXPECT_EQ(hash, read_only_hash);
}

TEST(NvidiaMtpDrafterSupportTest, SharedValueCacheOptionsOnlyOnTheNpuRoute) {
  const std::filesystem::path model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/magic_test_decode_batch.tflite";
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));
  LITERT_ASSERT_OK_AND_ASSIGN(auto drafter_model,
                              Model::CreateFromFile(model_path.string()));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto base_model, CompiledModel::Create(env, model_path.string(),
                                             litert::HwAccelerators::kCpu));
  // CPU and GPU never produce NVIDIA caches: the options are left alone
  // without looking at the models.
  for (Backend backend : {Backend::CPU, Backend::GPU}) {
    LITERT_ASSERT_OK_AND_ASSIGN(auto options, Options::Create());
    EXPECT_OK(AddNvidiaSharedValueCacheOptions(backend, drafter_model,
                                               base_model, options));
  }
}

TEST(NvidiaMtpDrafterSupportTest, GreedySamplerFallsBackOffNvidiaBuffers) {
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));
  // Host logits on the CPU backend, and host logits on the NPU route (whose
  // sampler decides on the first logits it sees), both sample on the host.
  for (Backend backend : {Backend::CPU, Backend::NPU}) {
    SCOPED_TRACE(static_cast<int>(backend));
    proto::SamplerParameters params;
    params.set_type(proto::SamplerParameters::TOP_P);
    params.set_k(1);
    params.set_p(0.0f);
    params.set_temperature(1.0f);
    params.set_seed(0);
    ASSERT_OK_AND_ASSIGN(auto sampler,
                         CreateMtpGreedySampler(
                             backend, /*batch_size=*/1, params, std::cref(env),
                             /*sequence_size=*/1, /*vocab_size=*/8,
                             ActivationDataType::FLOAT32));
    ASSERT_NE(sampler, nullptr);
    const std::vector<float> values = {0.1f, 0.2f, 0.7f, 0.3f,
                                       0.0f, 0.4f, 0.5f, 0.6f};
    LITERT_ASSERT_OK_AND_ASSIGN(auto logits,
                                CreateTensorBuffer<float>({1, 1, 8}));
    LITERT_ASSERT_OK(logits.Write<float>(absl::MakeConstSpan(values)));
    LITERT_ASSERT_OK_AND_ASSIGN(auto ids, CreateTensorBuffer<int32_t>({1, 1}));
    for (int round = 0; round < 2; ++round) {
      ASSERT_OK(sampler->SampleToIdAndScoreBuffer(logits, ids, nullptr));
      std::vector<int32_t> id(1);
      LITERT_ASSERT_OK(ids.Read<int32_t>(absl::MakeSpan(id)));
      EXPECT_EQ(id[0], 2);
    }
  }
}

proto::SamplerParameters GreedyParameters() {
  proto::SamplerParameters params;
  params.set_type(proto::SamplerParameters::TOP_P);
  params.set_k(1);
  params.set_p(0.0f);
  params.set_temperature(1.0f);
  return params;
}

TEST(NvidiaMtpDrafterSupportTest, SupportsDecodeAndBatchedVerifierFloatLogits) {
  for (auto dtype : {ElementType::Float16, ElementType::Float32}) {
    for (int rows : {1, 4}) {
      ASSERT_OK_AND_ASSIGN(
          bool supported,
          SupportsNvidiaGreedySampler(
              Backend::NPU, GreedyParameters(), kCudaBuffer,
              RankedTensorType(dtype, Layout(Dimensions{1, rows, 262144}))));
      EXPECT_TRUE(supported);
    }
  }
}

TEST(NvidiaMtpDrafterSupportTest,
     LeavesOtherBackendsTypesAndConfigsOnExistingPath) {
  const RankedTensorType type(ElementType::Float32,
                              Layout(Dimensions{1, 4, 8}));
  for (Backend backend : {Backend::CPU, Backend::GPU}) {
    ASSERT_OK_AND_ASSIGN(
        bool supported, SupportsNvidiaGreedySampler(backend, GreedyParameters(),
                                                    kCudaBuffer, type));
    EXPECT_FALSE(supported);
  }
  ASSERT_OK_AND_ASSIGN(
      bool host_supported,
      SupportsNvidiaGreedySampler(Backend::NPU, GreedyParameters(),
                                  TensorBufferType::kHostMemory, type));
  EXPECT_FALSE(host_supported);
  ASSERT_OK_AND_ASSIGN(
      bool int_supported,
      SupportsNvidiaGreedySampler(
          Backend::NPU, GreedyParameters(), kCudaBuffer,
          RankedTensorType(ElementType::Int32, Layout(type.Layout()))));
  EXPECT_FALSE(int_supported);
  for (int variant = 0; variant < 3; ++variant) {
    auto params = GreedyParameters();
    if (variant == 0) {
      params.set_k(2);
    } else if (variant == 1) {
      params.set_p(0.9f);
    } else {
      params.set_type(proto::SamplerParameters::TOP_K);
    }
    ASSERT_OK_AND_ASSIGN(
        bool supported,
        SupportsNvidiaGreedySampler(Backend::NPU, params, kCudaBuffer, type));
    EXPECT_FALSE(supported);
  }
}

TEST(NvidiaMtpDrafterSupportTest, RejectsMalformedMatchingLogitsShape) {
  for (const Dimensions& dims :
       std::vector<Dimensions>{{4, 8}, {2, 4, 8}, {1, 0, 8}, {1, 4, -1}}) {
    EXPECT_THAT(SupportsNvidiaGreedySampler(
                    Backend::NPU, GreedyParameters(), kCudaBuffer,
                    RankedTensorType(ElementType::Float32, Layout(dims))),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }
}

TEST(NvidiaMtpDrafterSupportTest, AcceptsDenseStridesAndRejectsPaddedLogits) {
  const Dimensions dims{1, 4, 8};
  const Strides dense{32, 8, 1};
  ASSERT_OK_AND_ASSIGN(
      bool supported,
      SupportsNvidiaGreedySampler(
          Backend::NPU, GreedyParameters(), kCudaBuffer,
          RankedTensorType(ElementType::Float16, Layout(dims, dense))));
  EXPECT_TRUE(supported);
  const Strides padded{36, 9, 1};
  EXPECT_THAT(SupportsNvidiaGreedySampler(
                  Backend::NPU, GreedyParameters(), kCudaBuffer,
                  RankedTensorType(ElementType::Float16, Layout(dims, padded))),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

#if defined(__linux__)
class ScopedSamplingFlag {
 public:
  explicit ScopedSamplingFlag(const char* value) {
    if (const char* old = std::getenv(kName)) {
      old_ = old;
    }
    if (value) {
      setenv(kName, value, 1);
    } else {
      unsetenv(kName);
    }
  }
  ~ScopedSamplingFlag() {
    if (old_) {
      setenv(kName, old_->c_str(), 1);
    } else {
      unsetenv(kName);
    }
  }

 private:
  static constexpr char kName[] = "LITERT_NVIDIA_MTP_GPU_SAMPLING";
  std::optional<std::string> old_;
};

TEST(NvidiaMtpDrafterSupportTest, HostLogitsOrOptOutKeepExistingSampler) {
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto logits,
      TensorBuffer::CreateManagedHostMemory(
          RankedTensorType(ElementType::Float32, Layout(Dimensions{1, 1, 8})),
          8 * sizeof(float)));
  for (const char* value : std::vector<const char*>{nullptr, "0", "", "true"}) {
    ScopedSamplingFlag flag(value);
    ASSERT_OK_AND_ASSIGN(
        auto sampler, TryCreateNvidiaGreedySampler(env, Backend::NPU,
                                                   GreedyParameters(), logits));
    EXPECT_EQ(sampler, nullptr);
  }
}

TEST(NvidiaMtpDrafterSupportTest, StillRequiresActualCudaBuffer) {
  ScopedSamplingFlag flag("1");
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto logits,
      TensorBuffer::CreateManagedHostMemory(
          RankedTensorType(ElementType::Float32, Layout(Dimensions{1, 1, 8})),
          8 * sizeof(float)));
  ASSERT_OK_AND_ASSIGN(
      auto sampler, TryCreateNvidiaGreedySampler(env, Backend::NPU,
                                                 GreedyParameters(), logits));
  EXPECT_EQ(sampler, nullptr);
}
#endif  // defined(__linux__)

}  // namespace
}  // namespace litert::lm
