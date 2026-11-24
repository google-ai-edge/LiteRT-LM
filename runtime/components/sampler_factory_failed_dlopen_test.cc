// Copyright 2025 The ODML Authors.
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

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "litert/c/litert_any.h"  // from @litert
#include "litert/c/litert_common.h"  // from @litert
#include "litert/c/litert_environment_options.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_expected.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/test/matchers.h"  // from @litert
#include "runtime/components/sampler_factory.h"
#include "runtime/components/top_p_cpu_sampler.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/proto/sampler_params.pb.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

// Creates a bare environment with a non-null `tag` handle, so that
// CreateSampler() tries to load the matching GPU sampler library, without
// triggering LiteRT's GpuEnvironment initialization on the bogus handle.
litert::Expected<litert::Environment> CreateEnvWithOption(
    LiteRtEnvOptionTag tag) {
  LITERT_ASSIGN_OR_RETURN(auto env, litert::Environment::Create({}));
  LiteRtAny value{};
  value.type = kLiteRtAnyTypeInt;
  value.int_value = 1;
  const LiteRtEnvOption option = {tag, value};
  const auto holder = env.GetHolder();
  LITERT_RETURN_IF_ERROR(holder.runtime->AddEnvironmentOptions(
      holder.handle, /*num_options=*/1, &option, /*overwrite=*/true));
  return env;
}

TEST(SamplerFactoryFailedDlopenTest,
     CreateSamplerForGpuFallsBackToCpuIfUnavailable) {
  std::vector<LiteRtEnvOptionTag> tags;
#if LITERT_HAS_WEBGPU_SUPPORT
  tags.push_back(kLiteRtEnvOptionTagWebGpuDevice);
#endif  // LITERT_HAS_WEBGPU_SUPPORT
#if LITERT_HAS_OPENCL_SUPPORT
  tags.push_back(kLiteRtEnvOptionTagOpenClContext);
#endif  // LITERT_HAS_OPENCL_SUPPORT
#if LITERT_HAS_METAL_SUPPORT
  tags.push_back(kLiteRtEnvOptionTagMetalDevice);
#endif  // LITERT_HAS_METAL_SUPPORT

  for (const LiteRtEnvOptionTag tag : tags) {
    SCOPED_TRACE(tag);
    LITERT_ASSERT_OK_AND_ASSIGN(auto litert_env, CreateEnvWithOption(tag));

    proto::SamplerParameters sampler_params;
    sampler_params.set_k(1);
    sampler_params.set_p(0.0);
    sampler_params.set_temperature(1.0);
    sampler_params.set_seed(12345);
    sampler_params.set_type(proto::SamplerParameters::TOP_P);
    auto result =
        CreateSampler(Backend::GPU, /*batch_size=*/1, std::move(sampler_params),
                      litert_env, /*sequence_size=*/1, /*vocab_size=*/201,
                      /*activation_data_type=*/std::nullopt);

    // The sampler library fails to load, so it should fall back to CPU.
    ASSERT_OK(result);
    auto sampler = std::move(result.value());
    TopPSampler* top_p_sampler = dynamic_cast<TopPSampler*>(sampler.get());
    EXPECT_NE(top_p_sampler, nullptr);
  }
}

}  // namespace
}  // namespace litert::lm
