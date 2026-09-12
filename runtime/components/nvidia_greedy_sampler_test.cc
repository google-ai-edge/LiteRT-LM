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

#include "runtime/components/nvidia_greedy_sampler.h"

#include <cstdlib>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/test/matchers.h"  // from @litert
#include "litert/vendors/nvidia/cache_layout.h"  // from @litert
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

using ::testing::status::StatusIs;
constexpr auto kCudaBuffer =
    static_cast<TensorBufferType>(nvidia::kNvidiaCudaTensorBufferType);

proto::SamplerParameters GreedyParameters() {
  proto::SamplerParameters params;
  params.set_type(proto::SamplerParameters::TOP_P);
  params.set_k(1);
  params.set_p(0.0f);
  params.set_temperature(1.0f);
  return params;
}

TEST(NvidiaGreedySamplerTest, SupportsDecodeAndBatchedVerifierFloatLogits) {
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

TEST(NvidiaGreedySamplerTest, LeavesOtherBackendsTypesAndConfigsOnExistingPath) {
  const RankedTensorType type(ElementType::Float32,
                              Layout(Dimensions{1, 4, 8}));
  for (Backend backend : {Backend::CPU, Backend::GPU}) {
    ASSERT_OK_AND_ASSIGN(bool supported,
                         SupportsNvidiaGreedySampler(
                             backend, GreedyParameters(), kCudaBuffer, type));
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
    ASSERT_OK_AND_ASSIGN(bool supported,
                         SupportsNvidiaGreedySampler(Backend::NPU, params,
                                                     kCudaBuffer, type));
    EXPECT_FALSE(supported);
  }
}

TEST(NvidiaGreedySamplerTest, RejectsMalformedMatchingLogitsShape) {
  for (const Dimensions& dims :
       std::vector<Dimensions>{{4, 8}, {2, 4, 8}, {1, 0, 8}, {1, 4, -1}}) {
    EXPECT_THAT(SupportsNvidiaGreedySampler(
                    Backend::NPU, GreedyParameters(), kCudaBuffer,
                    RankedTensorType(ElementType::Float32, Layout(dims))),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }
}

TEST(NvidiaGreedySamplerTest, AcceptsDenseStridesAndRejectsPaddedLogits) {
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

TEST(NvidiaGreedySamplerTest, DisabledFlagKeepsExistingSampler) {
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto logits,
      TensorBuffer::CreateManagedHostMemory(
          RankedTensorType(ElementType::Float32, Layout(Dimensions{1, 1, 8})),
          8 * sizeof(float)));
  for (const char* value : std::vector<const char*>{nullptr, "0", "", "true"}) {
    ScopedSamplingFlag flag(value);
    ASSERT_OK_AND_ASSIGN(auto sampler,
                         TryCreateNvidiaGreedySampler(
                             env, Backend::NPU, GreedyParameters(), logits));
    EXPECT_EQ(sampler, nullptr);
  }
}

TEST(NvidiaGreedySamplerTest, EnabledFlagStillRequiresActualCudaBuffer) {
  ScopedSamplingFlag flag("1");
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto logits,
      TensorBuffer::CreateManagedHostMemory(
          RankedTensorType(ElementType::Float32, Layout(Dimensions{1, 1, 8})),
          8 * sizeof(float)));
  ASSERT_OK_AND_ASSIGN(auto sampler,
                       TryCreateNvidiaGreedySampler(
                           env, Backend::NPU, GreedyParameters(), logits));
  EXPECT_EQ(sampler, nullptr);
}
#endif  // defined(__linux__)

}  // namespace
}  // namespace litert::lm
