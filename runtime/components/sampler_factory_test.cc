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

#include "runtime/components/sampler_factory.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/c/litert_common.h"  // from @litert
#include "litert/cc/litert_api_types.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_environment_options.h"  // from @litert
#include "litert/test/matchers.h"  // from @litert
#include "runtime/components/sampler.h"
#include "runtime/components/top_k_webgpu_sampler.h"
#include "runtime/components/top_p_cpu_sampler.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/proto/sampler_params.pb.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

extern "C" {
extern int (*LiteRtTopKOpenClSampler_Create_Static)(
    const struct LiteRtRuntimeCApiStruct*, LiteRtEnvironment, int, int, int,
    const void*, const void*, void**, char**);
extern void (*LiteRtTopKOpenClSampler_Destroy_Static)(void*);
extern int (*LiteRtTopKOpenClSampler_SampleToIdAndScoreBuffer_Static)(
    void*, LiteRtTensorBuffer, LiteRtTensorBuffer, const LiteRtTensorBuffer*,
    char**);
extern int (*LiteRtTopKOpenClSampler_UpdateConfig_Static)(void*, const void*,
                                                          int, void*, char**);
extern int (*LiteRtTopKOpenClSampler_CanHandleInput_Static)(void*);
extern int (*LiteRtTopKOpenClSampler_HandlesInput_Static)(void*);
extern int (*LiteRtTopKOpenClSampler_SetInferenceFuncAndInputTensors_Static)(
    void*, int (*)(void*), void*, LiteRtTensorBuffer, LiteRtTensorBuffer,
    LiteRtTensorBuffer, LiteRtTensorBuffer, LiteRtTensorBuffer,
    LiteRtTensorBuffer, LiteRtTensorBuffer, char**);

extern int (*LiteRtTopKWebGpuSampler_Create_Static)(
    const struct LiteRtRuntimeCApiStruct*, LiteRtEnvironment, int, int, int,
    const void*, const void*, void**, char**);
extern void (*LiteRtTopKWebGpuSampler_Destroy_Static)(void*);
extern int (*LiteRtTopKWebGpuSampler_SampleToIdAndScoreBuffer_Static)(
    void*, LiteRtTensorBuffer, LiteRtTensorBuffer, const LiteRtTensorBuffer*,
    char**);
extern int (*LiteRtTopKWebGpuSampler_UpdateConfig_Static)(void*, const void*,
                                                          int, void*, char**);
extern int (*LiteRtTopKWebGpuSampler_CanHandleInput_Static)(void*);
extern int (*LiteRtTopKWebGpuSampler_HandlesInput_Static)(void*);
extern int (*LiteRtTopKWebGpuSampler_SetInferenceFuncAndInputTensors_Static)(
    void*, int (*)(void*), void*, LiteRtTensorBuffer, LiteRtTensorBuffer,
    LiteRtTensorBuffer, LiteRtTensorBuffer, LiteRtTensorBuffer,
    LiteRtTensorBuffer, LiteRtTensorBuffer, char**);
}  // extern "C"

namespace litert::lm {
namespace {

using ::testing::HasSubstr;
using ::testing::status::StatusIs;

TEST(SamplerFactoryTest, CreateSamplerForCpuWorksCorrectly) {
  proto::SamplerParameters sampler_params;
  sampler_params.set_k(1);
  sampler_params.set_p(0.0);
  sampler_params.set_temperature(1.0);
  sampler_params.set_seed(12345);
  sampler_params.set_type(proto::SamplerParameters::TOP_P);
  ASSERT_OK_AND_ASSIGN(
      auto sampler, CreateSampler(Backend::CPU,
                                  /*batch_size=*/1, std::move(sampler_params)));
  EXPECT_NE(sampler, nullptr);
  // Make sure the factory creates the correct sampler.
  TopPSampler* top_p_sampler = dynamic_cast<TopPSampler*>(sampler.get());
  EXPECT_NE(top_p_sampler, nullptr);
  EXPECT_FALSE(top_p_sampler->ComputeExactLogProbs());
}

TEST(SamplerFactoryTest, CreateSamplerForCpuWithComputeExactLogProbs) {
  proto::SamplerParameters sampler_params;
  sampler_params.set_k(1);
  sampler_params.set_p(0.0);
  sampler_params.set_temperature(1.0);
  sampler_params.set_seed(12345);
  sampler_params.set_type(proto::SamplerParameters::TOP_P);
  sampler_params.set_compute_exact_log_probs(true);
  ASSERT_OK_AND_ASSIGN(
      auto sampler, CreateSampler(Backend::CPU,
                                  /*batch_size=*/1, std::move(sampler_params)));
  EXPECT_NE(sampler, nullptr);
  TopPSampler* top_p_sampler = dynamic_cast<TopPSampler*>(sampler.get());
  ASSERT_NE(top_p_sampler, nullptr);
  EXPECT_TRUE(top_p_sampler->ComputeExactLogProbs());
}

TEST(SamplerFactoryTest, CreateSamplerForCpuSupportsGreedyAndTopK) {
  proto::SamplerParameters greedy_params;
  greedy_params.set_type(proto::SamplerParameters::GREEDY);
  ASSERT_OK_AND_ASSIGN(
      auto greedy_sampler,
      CreateSampler(Backend::CPU, /*batch_size=*/1, std::move(greedy_params)));
  EXPECT_NE(dynamic_cast<TopPSampler*>(greedy_sampler.get()), nullptr);

  proto::SamplerParameters top_k_params;
  top_k_params.set_type(proto::SamplerParameters::TOP_K);
  top_k_params.set_k(10);
  top_k_params.set_temperature(0.8f);
  ASSERT_OK_AND_ASSIGN(
      auto top_k_sampler,
      CreateSampler(Backend::CPU, /*batch_size=*/1, std::move(top_k_params)));
  EXPECT_NE(dynamic_cast<TopPSampler*>(top_k_sampler.get()), nullptr);

  proto::SamplerParameters invalid_top_k;
  invalid_top_k.set_type(proto::SamplerParameters::TOP_K);
  invalid_top_k.set_k(0);
  EXPECT_THAT(
      CreateSampler(Backend::CPU, /*batch_size=*/1, std::move(invalid_top_k)),
      StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(SamplerFactoryTest, CreateSamplerForCpuWithUnsupportedSamplerTypeFails) {
  proto::SamplerParameters sampler_params;
  sampler_params.set_k(1);
  sampler_params.set_p(0.0);
  sampler_params.set_temperature(1.0);
  sampler_params.set_seed(12345);
  sampler_params.set_type(static_cast<proto::SamplerParameters::Type>(99));
  auto result = CreateSampler(Backend::CPU,
                              /*batch_size=*/1, std::move(sampler_params));
  EXPECT_THAT(result, StatusIs(absl::StatusCode::kUnimplemented));
  EXPECT_THAT(result.status().message(), HasSubstr("not implemented yet."));
}

TEST(SamplerFactoryTest,
     CreateSamplerForCpuWithUnspecifiedSamplerTypeReturnsNullptr) {
  proto::SamplerParameters sampler_params;
  sampler_params.set_k(1);
  sampler_params.set_p(0.0);
  sampler_params.set_temperature(1.0);
  sampler_params.set_seed(12345);
  sampler_params.set_type(proto::SamplerParameters::TYPE_UNSPECIFIED);
  ASSERT_OK_AND_ASSIGN(
      auto sampler,
      CreateSampler(Backend::CPU, /*batch_size=*/1, std::move(sampler_params)));
  EXPECT_EQ(sampler, nullptr);
}

proto::SamplerParameters TopPParams(int k, float p, float temperature) {
  proto::SamplerParameters sampler_params;
  sampler_params.set_type(proto::SamplerParameters::TOP_P);
  sampler_params.set_k(k);
  sampler_params.set_p(p);
  sampler_params.set_temperature(temperature);
  sampler_params.set_seed(0);
  return sampler_params;
}

TEST(SamplerFactoryTest, UpdateOrRecreateSamplerCreatesMissingSampler) {
  std::unique_ptr<Sampler> sampler;
  int max_top_k = 0;
  ASSERT_OK(UpdateOrRecreateSampler(sampler, max_top_k, Backend::CPU,
                                    /*batch_size=*/1,
                                    TopPParams(1, 0.0f, 1.0f)));
  EXPECT_NE(dynamic_cast<TopPSampler*>(sampler.get()), nullptr);
  EXPECT_EQ(max_top_k, 1);
}

TEST(SamplerFactoryTest, UpdateOrRecreateSamplerRecreatesOnlyWhenTopKGrows) {
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Sampler> sampler,
      CreateSampler(Backend::CPU, /*batch_size=*/1, TopPParams(1, 0.0f, 1.0f)));
  int max_top_k = 1;

  // A larger top-k recreates the sampler.
  const Sampler* original = sampler.get();
  ASSERT_OK(UpdateOrRecreateSampler(sampler, max_top_k, Backend::CPU,
                                    /*batch_size=*/1,
                                    TopPParams(40, 0.95f, 1.0f)));
  EXPECT_NE(sampler.get(), original);
  EXPECT_EQ(max_top_k, 40);

  // A smaller or equal top-k reconfigures the sampler in place.
  const Sampler* recreated = sampler.get();
  ASSERT_OK(UpdateOrRecreateSampler(sampler, max_top_k, Backend::CPU,
                                    /*batch_size=*/1,
                                    TopPParams(10, 0.9f, 0.7f)));
  EXPECT_EQ(sampler.get(), recreated);
  EXPECT_EQ(max_top_k, 40);

  // Switching to GREEDY (even with k = 0) normalizes to k = 1 and updates in
  // place without failing or shrinking max_top_k.
  proto::SamplerParameters greedy;
  greedy.set_type(proto::SamplerParameters::GREEDY);
  ASSERT_OK(UpdateOrRecreateSampler(sampler, max_top_k, Backend::CPU,
                                    /*batch_size=*/1, greedy));
  EXPECT_EQ(sampler.get(), recreated);
  EXPECT_EQ(max_top_k, 40);
}

TEST(SamplerFactoryTest, UpdateOrRecreateSamplerFailsWithoutSampler) {
  std::unique_ptr<Sampler> sampler;
  int max_top_k = 0;
  proto::SamplerParameters unspecified;
  unspecified.set_k(1);
  EXPECT_THAT(UpdateOrRecreateSampler(sampler, max_top_k, Backend::CPU,
                                      /*batch_size=*/1, unspecified),
              StatusIs(absl::StatusCode::kInternal));
  EXPECT_EQ(max_top_k, 0);
}

int g_fake_opencl_create_calls = 0;
int g_fake_webgpu_create_calls = 0;

int FakeOpenClCreate(const struct LiteRtRuntimeCApiStruct*, LiteRtEnvironment,
                     int, int, int, const void*, const void*,
                     void** sampler_out, char**) {
  ++g_fake_opencl_create_calls;
  static int dummy_cl_sampler = 1;
  *sampler_out = &dummy_cl_sampler;
  return 0;
}

int FakeWebGpuCreate(const struct LiteRtRuntimeCApiStruct*, LiteRtEnvironment,
                     int, int, int, const void*, const void*,
                     void** sampler_out, char**) {
  ++g_fake_webgpu_create_calls;
  static int dummy_wgpu_sampler = 2;
  *sampler_out = &dummy_wgpu_sampler;
  return 0;
}

void FakeDestroy(void*) {}
int FakeSample(void*, LiteRtTensorBuffer, LiteRtTensorBuffer,
               const LiteRtTensorBuffer*, char**) {
  return 0;
}
int FakeUpdateConfig(void*, const void*, int, void*, char**) { return 0; }
int FakeBoolStub(void*) { return 0; }
int FakeSetInference(void*, int (*)(void*), void*, LiteRtTensorBuffer,
                     LiteRtTensorBuffer, LiteRtTensorBuffer, LiteRtTensorBuffer,
                     LiteRtTensorBuffer, LiteRtTensorBuffer, LiteRtTensorBuffer,
                     char**) {
  return 0;
}

TEST(SamplerFactoryTest,
     CreateGpuSamplerSelectsMatchingBackendWhenMultipleStaticSamplersLinked) {
  auto prev_cl_create = LiteRtTopKOpenClSampler_Create_Static;
  auto prev_cl_destroy = LiteRtTopKOpenClSampler_Destroy_Static;
  auto prev_cl_sample = LiteRtTopKOpenClSampler_SampleToIdAndScoreBuffer_Static;
  auto prev_cl_update = LiteRtTopKOpenClSampler_UpdateConfig_Static;
  auto prev_cl_can_handle = LiteRtTopKOpenClSampler_CanHandleInput_Static;
  auto prev_cl_handles = LiteRtTopKOpenClSampler_HandlesInput_Static;
  auto prev_cl_set_inf =
      LiteRtTopKOpenClSampler_SetInferenceFuncAndInputTensors_Static;

  auto prev_wgpu_create = LiteRtTopKWebGpuSampler_Create_Static;
  auto prev_wgpu_destroy = LiteRtTopKWebGpuSampler_Destroy_Static;
  auto prev_wgpu_sample =
      LiteRtTopKWebGpuSampler_SampleToIdAndScoreBuffer_Static;
  auto prev_wgpu_update = LiteRtTopKWebGpuSampler_UpdateConfig_Static;
  auto prev_wgpu_can_handle = LiteRtTopKWebGpuSampler_CanHandleInput_Static;
  auto prev_wgpu_handles = LiteRtTopKWebGpuSampler_HandlesInput_Static;
  auto prev_wgpu_set_inf =
      LiteRtTopKWebGpuSampler_SetInferenceFuncAndInputTensors_Static;

  absl::Cleanup restore = [&] {
    LiteRtTopKOpenClSampler_Create_Static = prev_cl_create;
    LiteRtTopKOpenClSampler_Destroy_Static = prev_cl_destroy;
    LiteRtTopKOpenClSampler_SampleToIdAndScoreBuffer_Static = prev_cl_sample;
    LiteRtTopKOpenClSampler_UpdateConfig_Static = prev_cl_update;
    LiteRtTopKOpenClSampler_CanHandleInput_Static = prev_cl_can_handle;
    LiteRtTopKOpenClSampler_HandlesInput_Static = prev_cl_handles;
    LiteRtTopKOpenClSampler_SetInferenceFuncAndInputTensors_Static =
        prev_cl_set_inf;

    LiteRtTopKWebGpuSampler_Create_Static = prev_wgpu_create;
    LiteRtTopKWebGpuSampler_Destroy_Static = prev_wgpu_destroy;
    LiteRtTopKWebGpuSampler_SampleToIdAndScoreBuffer_Static = prev_wgpu_sample;
    LiteRtTopKWebGpuSampler_UpdateConfig_Static = prev_wgpu_update;
    LiteRtTopKWebGpuSampler_CanHandleInput_Static = prev_wgpu_can_handle;
    LiteRtTopKWebGpuSampler_HandlesInput_Static = prev_wgpu_handles;
    LiteRtTopKWebGpuSampler_SetInferenceFuncAndInputTensors_Static =
        prev_wgpu_set_inf;
  };

  LiteRtTopKOpenClSampler_Create_Static = &FakeOpenClCreate;
  LiteRtTopKOpenClSampler_Destroy_Static = &FakeDestroy;
  LiteRtTopKOpenClSampler_SampleToIdAndScoreBuffer_Static = &FakeSample;
  LiteRtTopKOpenClSampler_UpdateConfig_Static = &FakeUpdateConfig;
  LiteRtTopKOpenClSampler_CanHandleInput_Static = &FakeBoolStub;
  LiteRtTopKOpenClSampler_HandlesInput_Static = &FakeBoolStub;
  LiteRtTopKOpenClSampler_SetInferenceFuncAndInputTensors_Static =
      &FakeSetInference;

  LiteRtTopKWebGpuSampler_Create_Static = &FakeWebGpuCreate;
  LiteRtTopKWebGpuSampler_Destroy_Static = &FakeDestroy;
  LiteRtTopKWebGpuSampler_SampleToIdAndScoreBuffer_Static = &FakeSample;
  LiteRtTopKWebGpuSampler_UpdateConfig_Static = &FakeUpdateConfig;
  LiteRtTopKWebGpuSampler_CanHandleInput_Static = &FakeBoolStub;
  LiteRtTopKWebGpuSampler_HandlesInput_Static = &FakeBoolStub;
  LiteRtTopKWebGpuSampler_SetInferenceFuncAndInputTensors_Static =
      &FakeSetInference;

  g_fake_opencl_create_calls = 0;
  g_fake_webgpu_create_calls = 0;

  // 1. Empty environment has neither OpenCL nor WebGPU device tag, so GPU
  // sampler creation falls back cleanly to CPU TopPSampler.
  LITERT_ASSERT_OK_AND_ASSIGN(auto empty_env, litert::Environment::Create({}));
  ASSERT_OK_AND_ASSIGN(
      auto cpu_fallback,
      CreateSampler(Backend::GPU, /*batch_size=*/1, TopPParams(1, 0.0f, 1.0f),
                    std::cref(empty_env), /*sequence_size=*/1,
                    /*vocab_size=*/128));
  EXPECT_NE(dynamic_cast<TopPSampler*>(cpu_fallback.get()), nullptr);
  EXPECT_EQ(g_fake_opencl_create_calls, 0);
  EXPECT_EQ(g_fake_webgpu_create_calls, 0);

#if LITERT_HAS_OPENCL_SUPPORT
  // 2. Environment publishing kOpenClDeviceId selects the OpenCL sampler.
  const std::vector<litert::EnvironmentOptions::Option> cl_opts = {
      {litert::EnvironmentOptions::Tag::kOpenClDeviceId, int64_t{1}},
  };
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto cl_env,
      litert::Environment::Create(litert::EnvironmentOptions(cl_opts)));
  ASSERT_OK_AND_ASSIGN(
      auto cl_sampler,
      CreateSampler(Backend::GPU, /*batch_size=*/1, TopPParams(1, 0.0f, 1.0f),
                    std::cref(cl_env), /*sequence_size=*/1,
                    /*vocab_size=*/128));
  EXPECT_EQ(dynamic_cast<TopPSampler*>(cl_sampler.get()), nullptr);
  EXPECT_EQ(g_fake_opencl_create_calls, 1);
  EXPECT_EQ(g_fake_webgpu_create_calls, 0);
#endif  // LITERT_HAS_OPENCL_SUPPORT

#if LITERT_HAS_WEBGPU_SUPPORT
  // 3. Environment publishing kWebGpuDevice selects the WebGPU sampler.
  const std::vector<litert::EnvironmentOptions::Option> wgpu_opts = {
      {litert::EnvironmentOptions::Tag::kWebGpuDevice, int64_t{1}},
  };
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto wgpu_env,
      litert::Environment::Create(litert::EnvironmentOptions(wgpu_opts)));
  ASSERT_OK_AND_ASSIGN(
      auto wgpu_sampler,
      CreateSampler(Backend::GPU, /*batch_size=*/1, TopPParams(1, 0.0f, 1.0f),
                    std::cref(wgpu_env), /*sequence_size=*/1,
                    /*vocab_size=*/128));
  EXPECT_EQ(dynamic_cast<TopPSampler*>(wgpu_sampler.get()), nullptr);
  EXPECT_EQ(g_fake_webgpu_create_calls, 1);
#endif  // LITERT_HAS_WEBGPU_SUPPORT
}

TEST(SamplerFactoryTest, TopKWebGpuSamplerReadsRuntimeLibraryDirOption) {
  const std::string empty_dir = testing::TempDir();
  const std::vector<litert::EnvironmentOptions::Option> opts = {
      {litert::EnvironmentOptions::Tag::kRuntimeLibraryDir,
       litert::StringView(empty_dir)},
  };
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, litert::Environment::Create(litert::EnvironmentOptions(opts)));
  auto status_or = TopKWebGpuSampler::Create(
      env, /*batch_size=*/1, /*sequence_size=*/1, /*vocab_size=*/128,
      /*activation_data_type=*/std::nullopt, TopPParams(1, 0.0f, 1.0f));
  EXPECT_FALSE(status_or.ok());
}

}  // namespace
}  // namespace litert::lm
