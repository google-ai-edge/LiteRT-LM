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

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/c/litert_any.h"  // from @litert
#include "litert/c/litert_common.h"  // from @litert
#include "litert/c/litert_environment_options.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_expected.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/test/matchers.h"  // from @litert
#include "runtime/components/sampler.h"
#include "runtime/components/top_p_cpu_sampler.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/proto/sampler_params.pb.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace litert::lm {

using LiteRtTopKSampler_Sampler = void;
using LiteRtTopKSampler_ActivationDataType = void;
using LiteRtTopKSampler_SamplerParameters = void;

extern "C" int (*LiteRtTopKOpenClSampler_Create_Static)(
    const struct LiteRtRuntimeCApiStruct* runtime_c_api, LiteRtEnvironment env,
    int batch_size, int sequence_size, int vocab_size,
    const LiteRtTopKSampler_ActivationDataType* activation_data_type,
    const LiteRtTopKSampler_SamplerParameters* sampler_params,
    LiteRtTopKSampler_Sampler** sampler_out, char** error_msg);
extern "C" void (*LiteRtTopKOpenClSampler_Destroy_Static)(
    LiteRtTopKSampler_Sampler* sampler);
extern "C" int (*LiteRtTopKOpenClSampler_SampleToIdAndScoreBuffer_Static)(
    LiteRtTopKSampler_Sampler* sampler, LiteRtTensorBuffer logits_tensor,
    LiteRtTensorBuffer ids_tensor, const LiteRtTensorBuffer* scores_tensor,
    char** error_msg);
extern "C" int (*LiteRtTopKOpenClSampler_UpdateConfig_Static)(
    LiteRtTopKSampler_Sampler* sampler,
    const LiteRtTopKSampler_SamplerParameters* sampler_params, int batch_size,
    void* rand_gen_shared_ptr, char** error_msg);
extern "C" int (*LiteRtTopKOpenClSampler_CanHandleInput_Static)(
    LiteRtTopKSampler_Sampler* sampler);
extern "C" int (*LiteRtTopKOpenClSampler_HandlesInput_Static)(
    LiteRtTopKSampler_Sampler* sampler);
extern "C" int (
    *LiteRtTopKOpenClSampler_SetInferenceFuncAndInputTensors_Static)(
    LiteRtTopKSampler_Sampler* sampler, int (*run_inference_func)(void* arg),
    void* arg, LiteRtTensorBuffer ids_tensor,
    LiteRtTensorBuffer prev_input_positions_tensor,
    LiteRtTensorBuffer input_positions_tensor,
    LiteRtTensorBuffer prev_mask_tensor, LiteRtTensorBuffer mask_tensor,
    LiteRtTensorBuffer prev_param_tensor, LiteRtTensorBuffer param_tensor,
    char** error_msg);

extern "C" int (*LiteRtTopKWebGpuSampler_Create_Static)(
    const struct LiteRtRuntimeCApiStruct* runtime_c_api, LiteRtEnvironment env,
    int batch_size, int sequence_size, int vocab_size,
    const LiteRtTopKSampler_ActivationDataType* activation_data_type,
    const LiteRtTopKSampler_SamplerParameters* sampler_params,
    LiteRtTopKSampler_Sampler** sampler_out, char** error_msg);
extern "C" void (*LiteRtTopKWebGpuSampler_Destroy_Static)(
    LiteRtTopKSampler_Sampler* sampler);
extern "C" int (*LiteRtTopKWebGpuSampler_SampleToIdAndScoreBuffer_Static)(
    LiteRtTopKSampler_Sampler* sampler, LiteRtTensorBuffer logits_tensor,
    LiteRtTensorBuffer ids_tensor, const LiteRtTensorBuffer* scores_tensor,
    char** error_msg);
extern "C" int (*LiteRtTopKWebGpuSampler_UpdateConfig_Static)(
    LiteRtTopKSampler_Sampler* sampler,
    const LiteRtTopKSampler_SamplerParameters* sampler_params, int batch_size,
    void* rand_gen_shared_ptr, char** error_msg);
extern "C" int (*LiteRtTopKWebGpuSampler_CanHandleInput_Static)(
    LiteRtTopKSampler_Sampler* sampler);
extern "C" int (*LiteRtTopKWebGpuSampler_HandlesInput_Static)(
    LiteRtTopKSampler_Sampler* sampler);
extern "C" int (
    *LiteRtTopKWebGpuSampler_SetInferenceFuncAndInputTensors_Static)(
    LiteRtTopKSampler_Sampler* sampler, int (*run_inference_func)(void* arg),
    void* arg, LiteRtTensorBuffer ids_tensor,
    LiteRtTensorBuffer prev_input_positions_tensor,
    LiteRtTensorBuffer input_positions_tensor,
    LiteRtTensorBuffer prev_mask_tensor, LiteRtTensorBuffer mask_tensor,
    LiteRtTensorBuffer prev_param_tensor, LiteRtTensorBuffer param_tensor,
    char** error_msg);

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

struct FakeSamplerConfig {
  bool install = false;
  absl::StatusCode status = absl::StatusCode::kOk;
  const char* error = "";
};

char* DuplicateErrorString(const char* msg) {
  const size_t size = std::strlen(msg) + 1;
  char* copy = static_cast<char*>(std::malloc(size));
  if (copy != nullptr) {
    std::memcpy(copy, msg, size);
  }
  return copy;
}

// Attaches a backend option tag with `handle` to an existing environment
// without triggering LiteRT's GpuEnvironment initialization on bogus handles.
litert::Expected<void> AddEnvOption(const litert::Environment& env,
                                    LiteRtEnvOptionTag tag,
                                    int64_t handle = 1) {
  LiteRtAny value{};
  value.type = kLiteRtAnyTypeInt;
  value.int_value = handle;
  const LiteRtEnvOption option = {tag, value};
  const auto holder = env.GetHolder();
  LITERT_RETURN_IF_ERROR(holder.runtime->AddEnvironmentOptions(
      holder.handle, /*num_options=*/1, &option, /*overwrite=*/true));
  return {};
}

// Creates a bare environment with a single backend option tag.
litert::Expected<litert::Environment> CreateEnvWithOption(
    LiteRtEnvOptionTag tag, int64_t handle = 1) {
  LITERT_ASSIGN_OR_RETURN(auto env, litert::Environment::Create({}));
  LITERT_RETURN_IF_ERROR(AddEnvOption(env, tag, handle));
  return env;
}

int g_opencl_create_calls = 0;
int g_opencl_create_status = static_cast<int>(absl::StatusCode::kUnknown);
const char* g_opencl_create_error =
    "Can not find OpenCL library on this device";

int g_webgpu_create_calls = 0;
int g_webgpu_create_status = static_cast<int>(absl::StatusCode::kNotFound);
const char* g_webgpu_create_error = "Option was not set for this environment.";

int FakeOpenClCreate(
    const struct LiteRtRuntimeCApiStruct* /*runtime_c_api*/,
    LiteRtEnvironment /*env*/, int /*batch_size*/, int /*sequence_size*/,
    int /*vocab_size*/,
    const LiteRtTopKSampler_ActivationDataType* /*activation_data_type*/,
    const LiteRtTopKSampler_SamplerParameters* /*sampler_params*/,
    LiteRtTopKSampler_Sampler** /*sampler_out*/, char** error_msg) {
  ++g_opencl_create_calls;
  if (error_msg != nullptr) {
    *error_msg = DuplicateErrorString(g_opencl_create_error);
  }
  return g_opencl_create_status;
}

int FakeWebGpuCreate(
    const struct LiteRtRuntimeCApiStruct* /*runtime_c_api*/,
    LiteRtEnvironment /*env*/, int /*batch_size*/, int /*sequence_size*/,
    int /*vocab_size*/,
    const LiteRtTopKSampler_ActivationDataType* /*activation_data_type*/,
    const LiteRtTopKSampler_SamplerParameters* /*sampler_params*/,
    LiteRtTopKSampler_Sampler** /*sampler_out*/, char** error_msg) {
  ++g_webgpu_create_calls;
  if (error_msg != nullptr) {
    *error_msg = DuplicateErrorString(g_webgpu_create_error);
  }
  return g_webgpu_create_status;
}

void FakeDestroy(LiteRtTopKSampler_Sampler* /*sampler*/) {}

int FakeSample(LiteRtTopKSampler_Sampler* /*sampler*/,
               LiteRtTensorBuffer /*logits_tensor*/,
               LiteRtTensorBuffer /*ids_tensor*/,
               const LiteRtTensorBuffer* /*scores_tensor*/,
               char** /*error_msg*/) {
  return 0;
}

int FakeUpdateConfig(LiteRtTopKSampler_Sampler* /*sampler*/,
                     const LiteRtTopKSampler_SamplerParameters* /*params*/,
                     int /*batch_size*/, void* /*rand_gen_shared_ptr*/,
                     char** /*error_msg*/) {
  return 0;
}

int FakeCanHandleInput(LiteRtTopKSampler_Sampler* /*sampler*/) { return 0; }
int FakeHandlesInput(LiteRtTopKSampler_Sampler* /*sampler*/) { return 0; }

int FakeSetInferenceFuncAndInputTensors(
    LiteRtTopKSampler_Sampler* /*sampler*/,
    int (* /*run_inference_func*/)(void*), void* /*arg*/,
    LiteRtTensorBuffer /*ids_tensor*/,
    LiteRtTensorBuffer /*prev_input_positions_tensor*/,
    LiteRtTensorBuffer /*input_positions_tensor*/,
    LiteRtTensorBuffer /*prev_mask_tensor*/, LiteRtTensorBuffer /*mask_tensor*/,
    LiteRtTensorBuffer /*prev_param_tensor*/,
    LiteRtTensorBuffer /*param_tensor*/, char** /*error_msg*/) {
  return 0;
}

auto InstallFakeStaticGpuSamplers(FakeSamplerConfig opencl,
                                  FakeSamplerConfig webgpu) {
  const int prev_opencl_status = g_opencl_create_status;
  const char* prev_opencl_error = g_opencl_create_error;
  const int prev_webgpu_status = g_webgpu_create_status;
  const char* prev_webgpu_error = g_webgpu_create_error;

  g_opencl_create_calls = 0;
  g_webgpu_create_calls = 0;
  g_opencl_create_status = static_cast<int>(opencl.status);
  g_opencl_create_error = opencl.error;
  g_webgpu_create_status = static_cast<int>(webgpu.status);
  g_webgpu_create_error = webgpu.error;

  auto prev_opencl_create = LiteRtTopKOpenClSampler_Create_Static;
  auto prev_opencl_destroy = LiteRtTopKOpenClSampler_Destroy_Static;
  auto prev_opencl_sample =
      LiteRtTopKOpenClSampler_SampleToIdAndScoreBuffer_Static;
  auto prev_opencl_update = LiteRtTopKOpenClSampler_UpdateConfig_Static;
  auto prev_opencl_can_handle = LiteRtTopKOpenClSampler_CanHandleInput_Static;
  auto prev_opencl_handles = LiteRtTopKOpenClSampler_HandlesInput_Static;
  auto prev_opencl_set_fn =
      LiteRtTopKOpenClSampler_SetInferenceFuncAndInputTensors_Static;

  auto prev_webgpu_create = LiteRtTopKWebGpuSampler_Create_Static;
  auto prev_webgpu_destroy = LiteRtTopKWebGpuSampler_Destroy_Static;
  auto prev_webgpu_sample =
      LiteRtTopKWebGpuSampler_SampleToIdAndScoreBuffer_Static;
  auto prev_webgpu_update = LiteRtTopKWebGpuSampler_UpdateConfig_Static;
  auto prev_webgpu_can_handle = LiteRtTopKWebGpuSampler_CanHandleInput_Static;
  auto prev_webgpu_handles = LiteRtTopKWebGpuSampler_HandlesInput_Static;
  auto prev_webgpu_set_fn =
      LiteRtTopKWebGpuSampler_SetInferenceFuncAndInputTensors_Static;

  if (opencl.install) {
    LiteRtTopKOpenClSampler_Create_Static = &FakeOpenClCreate;
    LiteRtTopKOpenClSampler_Destroy_Static = &FakeDestroy;
    LiteRtTopKOpenClSampler_SampleToIdAndScoreBuffer_Static = &FakeSample;
    LiteRtTopKOpenClSampler_UpdateConfig_Static = &FakeUpdateConfig;
    LiteRtTopKOpenClSampler_CanHandleInput_Static = &FakeCanHandleInput;
    LiteRtTopKOpenClSampler_HandlesInput_Static = &FakeHandlesInput;
    LiteRtTopKOpenClSampler_SetInferenceFuncAndInputTensors_Static =
        &FakeSetInferenceFuncAndInputTensors;
  }
  if (webgpu.install) {
    LiteRtTopKWebGpuSampler_Create_Static = &FakeWebGpuCreate;
    LiteRtTopKWebGpuSampler_Destroy_Static = &FakeDestroy;
    LiteRtTopKWebGpuSampler_SampleToIdAndScoreBuffer_Static = &FakeSample;
    LiteRtTopKWebGpuSampler_UpdateConfig_Static = &FakeUpdateConfig;
    LiteRtTopKWebGpuSampler_CanHandleInput_Static = &FakeCanHandleInput;
    LiteRtTopKWebGpuSampler_HandlesInput_Static = &FakeHandlesInput;
    LiteRtTopKWebGpuSampler_SetInferenceFuncAndInputTensors_Static =
        &FakeSetInferenceFuncAndInputTensors;
  }

  return absl::MakeCleanup([=]() {
    g_opencl_create_status = prev_opencl_status;
    g_opencl_create_error = prev_opencl_error;
    g_webgpu_create_status = prev_webgpu_status;
    g_webgpu_create_error = prev_webgpu_error;

    LiteRtTopKOpenClSampler_Create_Static = prev_opencl_create;
    LiteRtTopKOpenClSampler_Destroy_Static = prev_opencl_destroy;
    LiteRtTopKOpenClSampler_SampleToIdAndScoreBuffer_Static =
        prev_opencl_sample;
    LiteRtTopKOpenClSampler_UpdateConfig_Static = prev_opencl_update;
    LiteRtTopKOpenClSampler_CanHandleInput_Static = prev_opencl_can_handle;
    LiteRtTopKOpenClSampler_HandlesInput_Static = prev_opencl_handles;
    LiteRtTopKOpenClSampler_SetInferenceFuncAndInputTensors_Static =
        prev_opencl_set_fn;

    LiteRtTopKWebGpuSampler_Create_Static = prev_webgpu_create;
    LiteRtTopKWebGpuSampler_Destroy_Static = prev_webgpu_destroy;
    LiteRtTopKWebGpuSampler_SampleToIdAndScoreBuffer_Static =
        prev_webgpu_sample;
    LiteRtTopKWebGpuSampler_UpdateConfig_Static = prev_webgpu_update;
    LiteRtTopKWebGpuSampler_CanHandleInput_Static = prev_webgpu_can_handle;
    LiteRtTopKWebGpuSampler_HandlesInput_Static = prev_webgpu_handles;
    LiteRtTopKWebGpuSampler_SetInferenceFuncAndInputTensors_Static =
        prev_webgpu_set_fn;
  });
}

#if LITERT_HAS_WEBGPU_SUPPORT
TEST(SamplerFactoryTest,
     CreateSamplerForGpuWebGpuEnvSkipsStaticOpenClAndFallsBackToCpu) {
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, CreateEnvWithOption(kLiteRtEnvOptionTagWebGpuDevice));

  auto cleanup = InstallFakeStaticGpuSamplers(
      /*opencl=*/{.install = true,
                  .status = absl::StatusCode::kUnknown,
                  .error = "Can not find OpenCL library on this device"},
      /*webgpu=*/{.install = true,
                  .status = absl::StatusCode::kUnavailable,
                  .error = "WebGPU sampler unavailable."});

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Sampler> sampler,
      CreateSampler(Backend::GPU, /*batch_size=*/1, TopPParams(1, 0.0f, 1.0f),
                    env, /*sequence_size=*/1, /*vocab_size=*/201,
                    /*activation_data_type=*/std::nullopt));
  EXPECT_EQ(g_webgpu_create_calls, 1);
  EXPECT_EQ(g_opencl_create_calls, 0);
  EXPECT_NE(dynamic_cast<TopPSampler*>(sampler.get()), nullptr);
}
#endif  // LITERT_HAS_WEBGPU_SUPPORT

#if LITERT_HAS_OPENCL_SUPPORT
TEST(SamplerFactoryTest,
     CreateSamplerForGpuOpenClEnvSkipsWebGpuAndFallsBackWhenUnavailable) {
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, CreateEnvWithOption(kLiteRtEnvOptionTagOpenClContext));

  auto cleanup = InstallFakeStaticGpuSamplers(
      /*opencl=*/{.install = true,
                  .status = absl::StatusCode::kUnavailable,
                  .error = "OpenCL library unavailable: Can not find OpenCL "
                           "library on this device"},
      /*webgpu=*/{.install = true,
                  .status = absl::StatusCode::kNotFound,
                  .error = "Option was not set for this environment."});

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Sampler> sampler,
      CreateSampler(Backend::GPU, /*batch_size=*/1, TopPParams(1, 0.0f, 1.0f),
                    env, /*sequence_size=*/1, /*vocab_size=*/201,
                    /*activation_data_type=*/std::nullopt));
  EXPECT_EQ(g_webgpu_create_calls, 0);
  EXPECT_EQ(g_opencl_create_calls, 1);
  EXPECT_NE(dynamic_cast<TopPSampler*>(sampler.get()), nullptr);
}
#endif  // LITERT_HAS_OPENCL_SUPPORT

TEST(SamplerFactoryTest,
     CreateSamplerForGpuBareEnvSkipsGpuSamplersAndFallsBackToCpu) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, litert::Environment::Create({}));

  auto cleanup = InstallFakeStaticGpuSamplers(
      /*opencl=*/{.install = true,
                  .status = absl::StatusCode::kUnknown,
                  .error = "Can not find OpenCL library on this device"},
      /*webgpu=*/{.install = true,
                  .status = absl::StatusCode::kNotFound,
                  .error = "Option was not set for this environment."});

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Sampler> sampler,
      CreateSampler(Backend::GPU, /*batch_size=*/1, TopPParams(1, 0.0f, 1.0f),
                    env, /*sequence_size=*/1, /*vocab_size=*/201,
                    /*activation_data_type=*/std::nullopt));
  EXPECT_EQ(g_opencl_create_calls, 0);
  EXPECT_EQ(g_webgpu_create_calls, 0);
  EXPECT_NE(dynamic_cast<TopPSampler*>(sampler.get()), nullptr);
}

TEST(SamplerFactoryTest,
     CreateSamplerForGpuNullHandleEnvSkipsGpuSamplersAndFallsBackToCpu) {
  for (const LiteRtEnvOptionTag tag :
       {kLiteRtEnvOptionTagOpenClContext, kLiteRtEnvOptionTagWebGpuDevice}) {
    SCOPED_TRACE(tag);
    LITERT_ASSERT_OK_AND_ASSIGN(auto env,
                                CreateEnvWithOption(tag, /*handle=*/0));

    // Fail with a non-kUnavailable error so that trying either GPU sampler
    // fails the test instead of falling back to CPU.
    auto cleanup = InstallFakeStaticGpuSamplers(
        /*opencl=*/{.install = true,
                    .status = absl::StatusCode::kUnknown,
                    .error = "OpenCL sampler should not be tried."},
        /*webgpu=*/{.install = true,
                    .status = absl::StatusCode::kUnknown,
                    .error = "WebGPU sampler should not be tried."});

    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<Sampler> sampler,
        CreateSampler(Backend::GPU, /*batch_size=*/1, TopPParams(1, 0.0f, 1.0f),
                      env, /*sequence_size=*/1, /*vocab_size=*/201,
                      /*activation_data_type=*/std::nullopt));
    EXPECT_EQ(g_opencl_create_calls, 0);
    EXPECT_EQ(g_webgpu_create_calls, 0);
    EXPECT_NE(dynamic_cast<TopPSampler*>(sampler.get()), nullptr);
  }
}

#if LITERT_HAS_OPENCL_SUPPORT
TEST(SamplerFactoryTest,
     CreateSamplerForGpuPropagatesNonUnavailableErrorFromMatchingBackend) {
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, CreateEnvWithOption(kLiteRtEnvOptionTagOpenClContext));

  auto cleanup = InstallFakeStaticGpuSamplers(
      /*opencl=*/{.install = true,
                  .status = absl::StatusCode::kInvalidArgument,
                  .error = "Batch size must be positive."},
      /*webgpu=*/{.install = false});

  EXPECT_THAT(
      CreateSampler(Backend::GPU, /*batch_size=*/1, TopPParams(1, 0.0f, 1.0f),
                    env, /*sequence_size=*/1, /*vocab_size=*/201,
                    /*activation_data_type=*/std::nullopt),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Batch size must be positive.")));
  EXPECT_EQ(g_opencl_create_calls, 1);
}
#endif  // LITERT_HAS_OPENCL_SUPPORT

#if LITERT_HAS_WEBGPU_SUPPORT
TEST(SamplerFactoryTest,
     CreateSamplerForGpuPropagatesNonUnavailableErrorFromWebGpu) {
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, CreateEnvWithOption(kLiteRtEnvOptionTagWebGpuDevice));

  auto cleanup = InstallFakeStaticGpuSamplers(
      /*opencl=*/{.install = false},
      /*webgpu=*/{.install = true,
                  .status = absl::StatusCode::kInvalidArgument,
                  .error = "Batch size must be positive."});

  EXPECT_THAT(
      CreateSampler(Backend::GPU, /*batch_size=*/1, TopPParams(1, 0.0f, 1.0f),
                    env, /*sequence_size=*/1, /*vocab_size=*/201,
                    /*activation_data_type=*/std::nullopt),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Batch size must be positive.")));
  EXPECT_EQ(g_webgpu_create_calls, 1);
}
#endif  // LITERT_HAS_WEBGPU_SUPPORT

#if LITERT_HAS_WEBGPU_SUPPORT && LITERT_HAS_OPENCL_SUPPORT
TEST(SamplerFactoryTest, CreateSamplerForGpuTriesWebGpuBeforeOpenCl) {
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, CreateEnvWithOption(kLiteRtEnvOptionTagWebGpuDevice));
  LITERT_ASSERT_OK(AddEnvOption(env, kLiteRtEnvOptionTagOpenClContext));

  {
    // A WebGPU error other than kUnavailable is returned without trying
    // OpenCL.
    auto cleanup = InstallFakeStaticGpuSamplers(
        /*opencl=*/{.install = true,
                    .status = absl::StatusCode::kUnknown,
                    .error = "OpenCL sampler should not be tried."},
        /*webgpu=*/{.install = true,
                    .status = absl::StatusCode::kInvalidArgument,
                    .error = "Batch size must be positive."});
    EXPECT_THAT(
        CreateSampler(Backend::GPU, /*batch_size=*/1, TopPParams(1, 0.0f, 1.0f),
                      env, /*sequence_size=*/1, /*vocab_size=*/201,
                      /*activation_data_type=*/std::nullopt),
        StatusIs(absl::StatusCode::kInvalidArgument));
    EXPECT_EQ(g_webgpu_create_calls, 1);
    EXPECT_EQ(g_opencl_create_calls, 0);
  }
  {
    // If both are unavailable, both are tried before falling back to CPU.
    auto cleanup = InstallFakeStaticGpuSamplers(
        /*opencl=*/{.install = true,
                    .status = absl::StatusCode::kUnavailable,
                    .error = "OpenCL sampler unavailable."},
        /*webgpu=*/{.install = true,
                    .status = absl::StatusCode::kUnavailable,
                    .error = "WebGPU sampler unavailable."});
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<Sampler> sampler,
        CreateSampler(Backend::GPU, /*batch_size=*/1, TopPParams(1, 0.0f, 1.0f),
                      env, /*sequence_size=*/1, /*vocab_size=*/201,
                      /*activation_data_type=*/std::nullopt));
    EXPECT_EQ(g_webgpu_create_calls, 1);
    EXPECT_EQ(g_opencl_create_calls, 1);
    EXPECT_NE(dynamic_cast<TopPSampler*>(sampler.get()), nullptr);
  }
}
#endif  // LITERT_HAS_WEBGPU_SUPPORT && LITERT_HAS_OPENCL_SUPPORT

}  // namespace
}  // namespace litert::lm
