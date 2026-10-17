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

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "devtools/build/runtime/get_runfiles_dir.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_buffer_ref.h"  // from @litert
#include "litert/cc/litert_common.h"  // from @litert
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_expected.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_options.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/cc/litert_tensor_buffer_types.h"  // from @litert
#include "litert/cc/options/litert_gpu_options.h"  // from @litert
#include "litert/test/matchers.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/components/model_resources_task.h"
#include "runtime/components/top_k_webgpu_sampler.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/proto/sampler_params.pb.h"
#include "runtime/util/model_asset_bundle_resources.h"
#include "runtime/util/scoped_file.h"
#include "util/task/contrib/status_macros/ret_check.h"
#include "util/task/status_macros.h"

#if !MEMORY_SANITIZER && !THREAD_SANITIZER
namespace litert::lm {
namespace {

using ::testing::HasSubstr;

constexpr absl::string_view kTestLmTaskPath =
    "litert_lm/runtime/testdata/test_lm.task";

absl::StatusOr<std::tuple<std::unique_ptr<litert::lm::ModelResources>,
                          litert::Environment, litert::CompiledModel>>
CreateLiteRTResources(absl::string_view file_path = kTestLmTaskPath) {
  std::string model_path = devtools_build::GetDataDependencyFilepath(file_path);
  auto scoped_file = ScopedFile::Open(model_path);
  auto resources = ModelAssetBundleResources::Create(
      /*tag=*/"", std::move(*scoped_file));
  ABSL_ASSIGN_OR_RETURN(auto model_resources,
                        ModelResourcesTask::Create(std::move(*resources)));
  ABSL_ASSIGN_OR_RETURN(auto model_buffer,
                        model_resources->GetTFLiteModelBuffer(
                            proto::LlmMetadata::TF_LITE_PREFILL_DECODE));
  LITERT_ASSIGN_OR_RETURN(auto env, litert::Environment::Create({}));

  LITERT_ASSIGN_OR_RETURN(auto compilation_options,
                          ::litert::Options::Create());
  LITERT_ASSIGN_OR_RETURN(
      auto& gpu_compilation_options,
      compilation_options.GetOptions<::litert::GpuOptions>());
  gpu_compilation_options.SetBackend(GpuOptions::Backend::kWebGpu);
  gpu_compilation_options.EnableConstantTensorSharing(true);
  gpu_compilation_options.EnableInfiniteFloatCapping(true);
  gpu_compilation_options.EnableAllowSrcQuantizedFcConvOps(true);
  gpu_compilation_options.SetPrecision(GpuOptions::Precision::kFp32);
  gpu_compilation_options.SetPreferTextureWeights(true);
  compilation_options.SetHardwareAccelerators(litert::HwAccelerators::kGpu);

  litert::BufferRef<uint8_t> model_buffer_ref(model_buffer.data(),
                                              model_buffer.size());
  LITERT_ASSIGN_OR_RETURN(auto compiled_model,
                          litert::CompiledModel::Create(env, model_buffer_ref,
                                                        compilation_options));
  return std::make_tuple(std::move(model_resources), std::move(env),
                         std::move(compiled_model));
}

TEST(TopKWebGpuSamplerCApiTest, CanCreateSuccessfully) {
  ASSERT_OK_AND_ASSIGN(auto ret, CreateLiteRTResources(kTestLmTaskPath));
  auto [litert_model, litert_env, compiled_model] = std::move(ret);
  proto::SamplerParameters sampler_params;
  sampler_params.set_k(1);
  sampler_params.set_p(0.5);
  sampler_params.set_temperature(1.0);
  LiteRtTopKWebGpuSampler_Sampler* sampler = nullptr;
  char* error_msg = nullptr;
  auto [runtime_c_api, env_handle] = litert_env.GetHolderForCApi();
  int error_code = LiteRtTopKWebGpuSampler_Create(
      runtime_c_api, env_handle, /*batch_size=*/1, /*sequence_size=*/1,
      /*vocab_size=*/201,
      /*activation_data_type=*/nullptr, &sampler_params, &sampler, &error_msg);
  free(error_msg);
  ASSERT_EQ(error_code, 0);
  ASSERT_NE(sampler, nullptr);
  LiteRtTopKWebGpuSampler_Destroy(sampler);
}

TEST(TopKWebGpuSamplerCApiTest, CreationFailedWithInvalidInputs) {
  ASSERT_OK_AND_ASSIGN(auto ret, CreateLiteRTResources(kTestLmTaskPath));
  auto [litert_model, litert_env, compiled_model] = std::move(ret);
  LiteRtTopKWebGpuSampler_Sampler* sampler = nullptr;
  char* error_msg = nullptr;
  auto [runtime_c_api, env_handle] = litert_env.GetHolderForCApi();
  int error_code = LiteRtTopKWebGpuSampler_Create(
      /*runtime_c_api=*/nullptr, env_handle, /*batch_size=*/1,
      /*sequence_size=*/1, /*vocab_size=*/201,
      /*activation_data_type=*/nullptr, /*sampler_params=*/nullptr, &sampler,
      &error_msg);
  ASSERT_NE(error_code, 0);
  std::string error_msg_str(error_msg);
  EXPECT_THAT(error_msg_str, HasSubstr("Input LiteRT C API not provided."));
  ASSERT_EQ(sampler, nullptr);
  free(error_msg);

  error_code = LiteRtTopKWebGpuSampler_Create(
      runtime_c_api, /*env=*/nullptr, /*batch_size=*/1,
      /*sequence_size=*/1, /*vocab_size=*/201,
      /*activation_data_type=*/nullptr, /*sampler_params=*/nullptr, &sampler,
      &error_msg);
  ASSERT_NE(error_code, 0);
  error_msg_str = error_msg;
  EXPECT_THAT(error_msg_str,
              HasSubstr("Input LiteRT Environment not provided."));
  ASSERT_EQ(sampler, nullptr);
  free(error_msg);

  // Error message is nullable.
  error_code = LiteRtTopKWebGpuSampler_Create(
      runtime_c_api, env_handle, /*batch_size=*/1, /*sequence_size=*/1,
      /*vocab_size=*/201,
      /*activation_data_type=*/nullptr, /*sampler_params=*/nullptr,
      /*sampler_out=*/nullptr, /*error_msg=*/nullptr);
  ASSERT_NE(error_code, 0);
  ASSERT_EQ(sampler, nullptr);
}

absl::Status CreateStatus(int error_code, const char* error_msg) {
  absl::StatusCode code = static_cast<absl::StatusCode>(error_code);
  return absl::Status(code, error_msg);
}

absl::StatusOr<std::vector<int>> Sample(
    Environment* litert_env, TensorBuffer& logits_tensor,
    proto::SamplerParameters sampler_params,
    const std::vector<int>& correct_indices = {},
    const std::vector<int>& incorrect_indices = {}) {
  LITERT_ASSIGN_OR_RETURN(auto tensor_type, logits_tensor.TensorType());
  RET_CHECK_EQ(tensor_type.Layout().Rank(), 3);
  RET_CHECK_EQ(tensor_type.Layout().Dimensions()[0], 1);
  RET_CHECK_EQ(tensor_type.Layout().Dimensions()[1], 1);
  int vocab_size = tensor_type.Layout().Dimensions()[2];
  RET_CHECK_GT(vocab_size, 1);

  LiteRtTopKWebGpuSampler_Sampler* sampler = nullptr;
  char* error_msg = nullptr;
  auto [runtime_c_api, env_handle] = litert_env->GetHolderForCApi();
  int error_code = LiteRtTopKWebGpuSampler_Create(
      runtime_c_api, env_handle, /*batch_size=*/1, /*sequence_size=*/1,
      vocab_size, /*activation_data_type=*/nullptr, &sampler_params, &sampler,
      &error_msg);
  if (error_code != 0) {
    auto status = CreateStatus(error_code, error_msg);
    free(error_msg);
    return status;
  }
  RET_CHECK_NE(sampler, nullptr);

  std::vector<float> logits_vec(vocab_size, 0.0f);
  for (auto corrent_index : correct_indices) {
    RET_CHECK_LT(corrent_index, vocab_size);
    logits_vec[corrent_index] = 0.01f;
  }
  for (auto incorrect_index : incorrect_indices) {
    RET_CHECK_LT(incorrect_index, vocab_size);
    logits_vec[incorrect_index] = -0.01f;
  }
  LITERT_RETURN_IF_ERROR(logits_tensor.Write(absl::MakeConstSpan(logits_vec)));

  LITERT_ASSIGN_OR_RETURN(
      TensorBuffer ids_tensor,
      litert::TensorBuffer::CreateManaged(
          *litert_env, litert::TensorBufferType::kWebGpuBuffer,
          litert::RankedTensorType(litert::ElementType::Float32,
                                   litert::Layout(litert::Dimensions({1}))),
          /*buffer_size=*/1));

  error_code = LiteRtTopKWebGpuSampler_SampleToIdAndScoreBuffer(
      sampler, logits_tensor.Get(), ids_tensor.Get(),
      /*scores_tensor=*/nullptr, &error_msg);
  if (error_code != 0) {
    auto status = CreateStatus(error_code, error_msg);
    free(error_msg);
    return status;
  }
  std::vector<int> ids(1, -1);
  LITERT_RETURN_IF_ERROR(ids_tensor.Read(absl::MakeSpan(ids)));

  LiteRtTopKWebGpuSampler_Destroy(sampler);
  return ids;
}

TEST(TopKWebGpuSamplerCApiTest, SamplingWorksAsExpected) {
  ASSERT_OK_AND_ASSIGN(auto ret, CreateLiteRTResources(kTestLmTaskPath));
  auto [litert_model, litert_env, compiled_model] = std::move(ret);

  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer logits_tensor,
      compiled_model.CreateOutputBuffer("decode", "logits"));

  proto::SamplerParameters sampler_params;
  sampler_params.set_k(1);
  sampler_params.set_p(0.5);
  sampler_params.set_temperature(1.0);
  sampler_params.set_seed(0);

  int correct_index = 1, incorrect_index = 0;
  ASSERT_OK_AND_ASSIGN(auto ids,
                       Sample(&litert_env, logits_tensor, sampler_params,
                              {correct_index}, {incorrect_index}));
  ASSERT_EQ(ids.size(), 1);
  EXPECT_EQ(ids[0], correct_index);
}

TEST(TopKWebGpuSamplerCApiTest, SamplingWorksAsExpectedWithDifferentSeeds) {
  ASSERT_OK_AND_ASSIGN(auto ret, CreateLiteRTResources(kTestLmTaskPath));
  auto [litert_model, litert_env, compiled_model] = std::move(ret);

  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer logits_tensor,
      compiled_model.CreateOutputBuffer("decode", "logits"));

  proto::SamplerParameters sampler_params;
  sampler_params.set_k(3);
  // Tokens 0 and 1 have probability 0.334 each and token 2 has 0.331. Top-p
  // keeps the smallest prefix whose cumulative probability reaches p, i.e.
  // tokens {0, 1} (0.669 >= 0.6), so token 2 can never be sampled.
  sampler_params.set_p(0.6);
  sampler_params.set_temperature(1.0);

  // Test with two different seeds.
  sampler_params.set_seed(0);
  int correct_index_1 = 0, correct_index_2 = 1;
  ASSERT_OK_AND_ASSIGN(auto ids,
                       Sample(&litert_env, logits_tensor, sampler_params,
                              {correct_index_1, correct_index_2}));
  ASSERT_EQ(ids.size(), 1);
  EXPECT_EQ(ids[0], correct_index_1);

  sampler_params.set_seed(22000);
  ASSERT_OK_AND_ASSIGN(ids, Sample(&litert_env, logits_tensor, sampler_params,
                                   {correct_index_1, correct_index_2}));
  ASSERT_EQ(ids.size(), 1);
  EXPECT_EQ(ids[0], correct_index_2);
}

}  // namespace
}  // namespace litert::lm
#endif  // !MEMORY_SANITIZER && !THREAD_SANITIZER
