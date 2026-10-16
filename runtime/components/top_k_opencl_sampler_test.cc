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

#include "runtime/components/top_k_opencl_sampler.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "devtools/build/runtime/get_runfiles_dir.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
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
#include "runtime/components/top_p_cpu_sampler.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/proto/sampler_params.pb.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/model_asset_bundle_resources.h"
#include "runtime/util/scoped_file.h"
#include "util/task/contrib/status_macros/ret_check.h"
#include "util/task/status_macros.h"

#if !MEMORY_SANITIZER && !THREAD_SANITIZER
namespace litert::lm {
namespace {

using ::litert::ElementType;
using ::litert::TensorBufferScopedLock;

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

TEST(TopKOpenClSamplerTest, CanCreateSuccessfully) {
  ASSERT_OK_AND_ASSIGN(auto ret, CreateLiteRTResources(kTestLmTaskPath));
  auto [litert_model, litert_env, compiled_model] = std::move(ret);
  proto::SamplerParameters sampler_params;
  sampler_params.set_k(1);
  sampler_params.set_p(0.5);
  sampler_params.set_temperature(1.0);
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<TopKOpenClSampler> sampler,
      TopKOpenClSampler::Create(
          /*env=*/litert_env, /*batch_size=*/1,
          /*sequence_size=*/1, /*vocab_size=*/201,
          /*activation_data_type=*/std::nullopt, sampler_params));
  EXPECT_NE(sampler, nullptr);
}

TEST(TopKOpenClSamplerTest, CanCreateSuccessfullyWithFloat16) {
  ASSERT_OK_AND_ASSIGN(auto ret, CreateLiteRTResources(kTestLmTaskPath));
  auto [litert_model, litert_env, compiled_model] = std::move(ret);
  proto::SamplerParameters sampler_params;
  sampler_params.set_k(1);
  sampler_params.set_p(0.5);
  sampler_params.set_temperature(1.0);
  auto sampler_or = TopKOpenClSampler::Create(
      /*env=*/litert_env, /*batch_size=*/1, /*sequence_size=*/1,
      /*vocab_size=*/201,
      /*activation_data_type=*/ActivationDataType::FLOAT16, sampler_params);
  if (!sampler_or.ok()) {
    GTEST_SKIP() << "Skipping FP16 test due to creation failure: "
                 << sampler_or.status();
  }
  EXPECT_NE(sampler_or.value(), nullptr);
}

absl::StatusOr<std::vector<int>> Sample(
    Environment* litert_env, TensorBuffer& logits_tensor,
    const proto::SamplerParameters& sampler_params,
    const std::vector<int>& correct_indices = {},
    const std::vector<int>& incorrect_indices = {}) {
  LITERT_ASSIGN_OR_RETURN(auto tensor_type, logits_tensor.TensorType());
  RET_CHECK_EQ(tensor_type.Layout().Rank(), 3);
  RET_CHECK_EQ(tensor_type.Layout().Dimensions()[0], 1);
  RET_CHECK_EQ(tensor_type.Layout().Dimensions()[1], 1);
  int vocab_size = tensor_type.Layout().Dimensions()[2];
  RET_CHECK_GT(vocab_size, 1);

  ABSL_ASSIGN_OR_RETURN(
      std::unique_ptr<TopKOpenClSampler> sampler,
      TopKOpenClSampler::Create(
          /*env=*/*litert_env, /*batch_size=*/1, /*sequence_size=*/1,
          vocab_size, /*activation_data_type=*/std::nullopt, sampler_params));
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
          *litert_env, litert::TensorBufferType::kOpenClBuffer,
          litert::RankedTensorType(litert::ElementType::Float32,
                                   litert::Layout(litert::Dimensions({1}))),
          /*buffer_size=*/1));

  RET_CHECK_OK(sampler->SampleToIdAndScoreBuffer(logits_tensor, ids_tensor,
                                                 /*scores_tensor=*/nullptr));
  std::vector<int> ids(1, -1);
  LITERT_RETURN_IF_ERROR(ids_tensor.Read(absl::MakeSpan(ids)));

  return ids;
}

absl::StatusOr<std::vector<int>> Sample(
    Environment* litert_env, TensorBuffer& logits_tensor,
    TopKOpenClSampler& sampler, const std::vector<int>& correct_indices = {},
    const std::vector<int>& incorrect_indices = {}) {
  LITERT_ASSIGN_OR_RETURN(auto tensor_type, logits_tensor.TensorType());
  RET_CHECK_EQ(tensor_type.Layout().Rank(), 3);
  RET_CHECK_EQ(tensor_type.Layout().Dimensions()[0], 1);
  RET_CHECK_EQ(tensor_type.Layout().Dimensions()[1], 1);
  int vocab_size = tensor_type.Layout().Dimensions()[2];
  RET_CHECK_GT(vocab_size, 1);

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
          *litert_env, litert::TensorBufferType::kOpenClBuffer,
          litert::RankedTensorType(litert::ElementType::Int32,
                                   litert::Layout(litert::Dimensions({1}))),
          /*buffer_size=*/sizeof(int32_t)));

  RET_CHECK_OK(sampler.SampleToIdAndScoreBuffer(logits_tensor, ids_tensor,
                                                /*scores_tensor=*/nullptr));
  std::vector<int> ids(1, -1);
  LITERT_RETURN_IF_ERROR(ids_tensor.Read(absl::MakeSpan(ids)));

  return ids;
}

TEST(TopKOpenClSamplerTest, SamplingWorksAsExpected) {
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

TEST(TopKOpenClSamplerTest, SamplingWorksAsExpectedWithDifferentSeeds) {
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

TEST(TopKOpenClSamplerTest, SamplingWorksAsExpectedWithDifferentModes) {
  ASSERT_OK_AND_ASSIGN(auto ret, CreateLiteRTResources(kTestLmTaskPath));
  auto [litert_model, litert_env, compiled_model] = std::move(ret);

  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer logits_tensor,
      compiled_model.CreateOutputBuffer("decode", "logits"));
  int correct_index_1 = 0, correct_index_2 = 1;

  // Mode 1: Greedy (k = 1)
  proto::SamplerParameters sampler_params_greedy;
  sampler_params_greedy.set_k(1);
  sampler_params_greedy.set_p(1.0);
  sampler_params_greedy.set_temperature(1.0);
  sampler_params_greedy.set_seed(22000);
  ASSERT_OK_AND_ASSIGN(auto ids_greedy,
                       Sample(&litert_env, logits_tensor, sampler_params_greedy,
                              {correct_index_1, correct_index_2}));
  ASSERT_EQ(ids_greedy.size(), 1);
  EXPECT_EQ(ids_greedy[0], correct_index_1);

  // Mode 2: Top K (k > 1 and p = 1.0). Only the two highest logits (tokens 0
  // and 1, with probability 0.5 each) are kept.
  proto::SamplerParameters sampler_params_topk;
  sampler_params_topk.set_k(2);
  sampler_params_topk.set_p(1.0);
  sampler_params_topk.set_temperature(1.0);
  sampler_params_topk.set_seed(22000);
  ASSERT_OK_AND_ASSIGN(auto ids_topk,
                       Sample(&litert_env, logits_tensor, sampler_params_topk,
                              {correct_index_1, correct_index_2}));
  ASSERT_EQ(ids_topk.size(), 1);
  EXPECT_EQ(ids_topk[0], correct_index_2);
}

TEST(TopKOpenClSamplerTest, SamplingWorksAsExpectedWithTemperatureZero) {
  ASSERT_OK_AND_ASSIGN(auto ret, CreateLiteRTResources(kTestLmTaskPath));
  auto [litert_model, litert_env, compiled_model] = std::move(ret);

  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer logits_tensor,
      compiled_model.CreateOutputBuffer("decode", "logits"));

  proto::SamplerParameters sampler_params;
  sampler_params.set_k(3);
  sampler_params.set_p(1.0);
  sampler_params.set_temperature(0.0);
  sampler_params.set_seed(0);

  int correct_index = 10, incorrect_index = 3;
  ASSERT_OK_AND_ASSIGN(auto ids,
                       Sample(&litert_env, logits_tensor, sampler_params,
                              {correct_index}, {incorrect_index}));
  ASSERT_EQ(ids.size(), 1);
  EXPECT_EQ(ids[0], correct_index);
}

TEST(TopKOpenClSamplerTest, UpdateConfigWorksEndToEnd) {
  ASSERT_OK_AND_ASSIGN(auto ret, CreateLiteRTResources(kTestLmTaskPath));
  auto [litert_model, litert_env, compiled_model] = std::move(ret);

  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer logits_tensor,
      compiled_model.CreateOutputBuffer("decode", "logits"));
  LITERT_ASSERT_OK_AND_ASSIGN(auto tensor_type, logits_tensor.TensorType());
  int vocab_size = tensor_type.Layout().Dimensions()[2];

  proto::SamplerParameters sampler_params;
  sampler_params.set_k(2);
  sampler_params.set_p(1.0);
  sampler_params.set_temperature(1.0);
  sampler_params.set_seed(22000);

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<TopKOpenClSampler> sampler,
      TopKOpenClSampler::Create(
          /*env=*/litert_env, /*batch_size=*/1, /*sequence_size=*/1, vocab_size,
          /*activation_data_type=*/std::nullopt, sampler_params));
  ASSERT_NE(sampler, nullptr);

  std::vector<float> logits_vec(vocab_size, 0.0f);
  const int target_index = 4;
  logits_vec[target_index] = 0.01f;

  LITERT_ASSERT_OK(logits_tensor.Write(absl::MakeConstSpan(logits_vec)));

  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer ids_tensor,
      litert::TensorBuffer::CreateManaged(
          litert_env, litert::TensorBufferType::kOpenClBuffer,
          litert::RankedTensorType(litert::ElementType::Int32,
                                   litert::Layout(litert::Dimensions({1}))),
          /*buffer_size=*/sizeof(int32_t)));

  // With k=2, the target and a zero logit have about the same probability, so
  // only k=1 picks the target every time.
  auto expect_always_argmax = [&] {
    for (int i = 0; i < 20; ++i) {
      ASSERT_OK(sampler->SampleToIdAndScoreBuffer(logits_tensor, ids_tensor,
                                                  /*scores_tensor=*/nullptr));
      std::vector<int> ids(1, -1);
      LITERT_ASSERT_OK(ids_tensor.Read(absl::MakeSpan(ids)));
      EXPECT_EQ(ids[0], target_index);
    }
  };

  // The creation-time k is accepted.
  EXPECT_OK(sampler->UpdateConfig(sampler_params, /*batch_size=*/1, nullptr));

  // Lowering k takes effect.
  sampler_params.set_k(1);
  EXPECT_OK(sampler->UpdateConfig(sampler_params, /*batch_size=*/1, nullptr));
  expect_always_argmax();

  // A k above the creation-time k is rejected and leaves the sampler on k=1.
  sampler_params.set_k(vocab_size);
  EXPECT_THAT(sampler->UpdateConfig(sampler_params, /*batch_size=*/1, nullptr),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
  expect_always_argmax();
}

TEST(TopKOpenClSamplerTest, MatchesCpuSampler) {
  ASSERT_OK_AND_ASSIGN(auto ret, CreateLiteRTResources(kTestLmTaskPath));
  auto [litert_model, litert_env, compiled_model] = std::move(ret);

  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer logits_tensor,
      compiled_model.CreateOutputBuffer("decode", "logits"));
  LITERT_ASSERT_OK_AND_ASSIGN(auto tensor_type, logits_tensor.TensorType());
  const int vocab_size = tensor_type.Layout().Dimensions()[2];

  // The GPU sampler must implement the same top-k/top-p sampling as the CPU
  // sampler and consume the random number generator in the same way, so that
  // the same seed yields the same tokens on both backends.
  constexpr int kTopK = 8;
  constexpr float kTopP = 0.8f;
  constexpr float kTemperature = 0.7f;
  constexpr int kSeed = 1234;
  proto::SamplerParameters sampler_params;
  sampler_params.set_k(kTopK);
  sampler_params.set_p(kTopP);
  sampler_params.set_temperature(kTemperature);
  sampler_params.set_seed(kSeed);
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<TopKOpenClSampler> gpu_sampler,
      TopKOpenClSampler::Create(litert_env, /*batch_size=*/1,
                                /*sequence_size=*/1, vocab_size,
                                ActivationDataType::FLOAT32, sampler_params));
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<TopPSampler> cpu_sampler,
      TopPSampler::Create(kTopK, kTopP, kTemperature, /*batch_size=*/1,
                          /*sequence_size=*/1, kSeed));

  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer gpu_ids_tensor,
      litert::TensorBuffer::CreateManaged(
          litert_env, litert::TensorBufferType::kOpenClBuffer,
          litert::RankedTensorType(litert::ElementType::Int32,
                                   litert::Layout(litert::Dimensions({1}))),
          /*buffer_size=*/sizeof(int32_t)));
  const std::vector<int> initial_ids = {-1};
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer cpu_ids_tensor,
      CopyToTensorBuffer<int>(absl::MakeConstSpan(initial_ids), {1}));

  constexpr int kNumSteps = 200;
  std::mt19937 logits_generator(42);
  std::uniform_real_distribution<float> logits_distribution(-2.0f, 2.0f);
  std::vector<float> logits(vocab_size);
  int num_mismatches = 0;
  for (int step = 0; step < kNumSteps; ++step) {
    for (float& logit : logits) {
      logit = logits_distribution(logits_generator);
    }
    LITERT_ASSERT_OK(logits_tensor.Write(absl::MakeConstSpan(logits)));
    LITERT_ASSERT_OK_AND_ASSIGN(
        TensorBuffer cpu_logits_tensor,
        CopyToTensorBuffer<float>(absl::MakeConstSpan(logits),
                                  {1, 1, vocab_size}));

    ASSERT_OK(gpu_sampler->SampleToIdAndScoreBuffer(
        logits_tensor, gpu_ids_tensor, /*scores_tensor=*/nullptr));
    ASSERT_OK(cpu_sampler->SampleToIdAndScoreBuffer(
        cpu_logits_tensor, cpu_ids_tensor, /*scores_tensor=*/nullptr));

    std::vector<int> gpu_ids(1, -1);
    LITERT_ASSERT_OK(gpu_ids_tensor.Read(absl::MakeSpan(gpu_ids)));
    LITERT_ASSERT_OK_AND_ASSIGN(std::vector<int> cpu_ids,
                                CopyFromTensorBuffer<int>(cpu_ids_tensor));
    ASSERT_EQ(cpu_ids.size(), 1);
    if (gpu_ids[0] != cpu_ids[0]) {
      ++num_mismatches;
    }
  }
  // Floating-point differences between the backends can only flip a draw that
  // lands right on a cumulative probability boundary, which should be rare.
  EXPECT_LE(num_mismatches, kNumSteps / 100);
}

TEST(TopKOpenClSamplerTest, SamplingWorksAsExpectedWithFloat16) {
  ASSERT_OK_AND_ASSIGN(auto ret, CreateLiteRTResources(kTestLmTaskPath));
  auto [litert_model, litert_env, compiled_model] = std::move(ret);

  proto::SamplerParameters sampler_params;
  sampler_params.set_k(1);
  sampler_params.set_p(0.5);
  sampler_params.set_temperature(1.0);
  sampler_params.set_seed(0);

  int vocab_size = 201;
  auto sampler_or = TopKOpenClSampler::Create(
      /*env=*/litert_env, /*batch_size=*/1, /*sequence_size=*/1, vocab_size,
      /*activation_data_type=*/ActivationDataType::FLOAT16, sampler_params);
  if (!sampler_or.ok()) {
    GTEST_SKIP() << "Skipping FP16 test due to creation failure: "
                 << sampler_or.status();
  }
  std::unique_ptr<TopKOpenClSampler> sampler = std::move(sampler_or.value());

  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer logits_tensor,
      litert::TensorBuffer::CreateManaged(
          litert_env, litert::TensorBufferType::kOpenClBuffer,
          litert::RankedTensorType(
              litert::ElementType::Float16,
              litert::Layout(litert::Dimensions({1, 1, vocab_size}))),
          /*buffer_size=*/vocab_size * sizeof(uint16_t)));

  std::vector<uint16_t> logits_vec(vocab_size, 0);  // 0.0
  int correct_index = 1;
  logits_vec[correct_index] = 0x3c00;  // 1.0 in Float16

  LITERT_ASSERT_OK(logits_tensor.Write(absl::MakeConstSpan(logits_vec)));

  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer ids_tensor,
      litert::TensorBuffer::CreateManaged(
          litert_env, litert::TensorBufferType::kOpenClBuffer,
          litert::RankedTensorType(litert::ElementType::Int32,
                                   litert::Layout(litert::Dimensions({1}))),
          /*buffer_size=*/sizeof(int32_t)));

  ASSERT_OK(sampler->SampleToIdAndScoreBuffer(logits_tensor, ids_tensor,
                                              /*scores_tensor=*/nullptr));
  std::vector<int> ids(1, -1);
  LITERT_ASSERT_OK(ids_tensor.Read(absl::MakeSpan(ids)));

  ASSERT_EQ(ids.size(), 1);
  EXPECT_EQ(ids[0], correct_index);
}

TEST(TopKOpenClSamplerTest, SamplingWorksAsExpectedWithMultiTokens) {
  ASSERT_OK_AND_ASSIGN(auto ret, CreateLiteRTResources(kTestLmTaskPath));
  auto [litert_model, litert_env, compiled_model] = std::move(ret);

  int batch_size = 1;
  int sequence_size = 2;

  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer logits_tensor_meta,
      compiled_model.CreateOutputBuffer("decode", "logits"));
  LITERT_ASSERT_OK_AND_ASSIGN(auto tensor_type,
                              logits_tensor_meta.TensorType());
  int vocab_size = tensor_type.Layout().Dimensions()[2];

  proto::SamplerParameters sampler_params;
  sampler_params.set_k(1);
  sampler_params.set_p(1.0);
  sampler_params.set_temperature(1.0);
  sampler_params.set_seed(0);

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<TopKOpenClSampler> sampler,
      TopKOpenClSampler::Create(
          /*env=*/litert_env, batch_size, sequence_size, vocab_size,
          /*activation_data_type=*/std::nullopt, sampler_params));
  ASSERT_NE(sampler, nullptr);

  // Re-create the logits tensor with the correct sequence size.
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer logits_tensor,
      litert::TensorBuffer::CreateManaged(
          litert_env, litert::TensorBufferType::kOpenClBuffer,
          litert::RankedTensorType(
              litert::ElementType::Float32,
              litert::Layout(
                  litert::Dimensions({batch_size, sequence_size, vocab_size}))),
          /*buffer_size=*/batch_size * sequence_size * vocab_size *
              sizeof(float)));

  std::vector<float> logits_vec(batch_size * sequence_size * vocab_size, 0.0f);
  // Sequence 0, favor token 10
  int target_0 = 10;
  logits_vec[target_0] = 100.0f;
  // Sequence 1, favor token 20
  int target_1 = 20;
  logits_vec[vocab_size + target_1] = 100.0f;

  LITERT_ASSERT_OK(logits_tensor.Write(absl::MakeConstSpan(logits_vec)));

  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer ids_tensor,
      litert::TensorBuffer::CreateManaged(
          litert_env, litert::TensorBufferType::kOpenClBuffer,
          litert::RankedTensorType(
              litert::ElementType::Int32,
              litert::Layout(litert::Dimensions({batch_size * sequence_size}))),
          /*buffer_size=*/batch_size * sequence_size * sizeof(int32_t)));

  ASSERT_OK(sampler->SampleToIdAndScoreBuffer(logits_tensor, ids_tensor,
                                              /*scores_tensor=*/nullptr));

  std::vector<int> ids(batch_size * sequence_size, -1);
  LITERT_ASSERT_OK(ids_tensor.Read(absl::MakeSpan(ids)));

  ASSERT_EQ(ids.size(), 2);
  EXPECT_EQ(ids[0], target_0);
  EXPECT_EQ(ids[1], target_1);
}

int MarkBooleanTrue(void* arg) {
  *static_cast<bool*>(arg) = true;
  return 0;
}

TEST(TopKOpenClSamplerTest, HandlesInputAfterSampling) {
  constexpr int kInputPos = 10;

  ASSERT_OK_AND_ASSIGN(auto ret, CreateLiteRTResources(kTestLmTaskPath));
  auto [litert_model, litert_env, compiled_model] = std::move(ret);

  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer logits_tensor,
      compiled_model.CreateOutputBuffer("decode", "logits"));
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer tokens_tensor,
      compiled_model.CreateInputBuffer("decode", "tokens"));
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer prev_input_pos_tensor,
      compiled_model.CreateInputBuffer("decode", "input_pos"));
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer input_pos_tensor,
      compiled_model.CreateInputBuffer("decode", "input_pos"));
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer prev_mask_tensor,
      compiled_model.CreateInputBuffer("decode", "mask"));
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer mask_tensor,
      compiled_model.CreateInputBuffer("decode", "mask"));

  LITERT_ASSERT_OK_AND_ASSIGN(auto tensor_type, logits_tensor.TensorType());
  int vocab_size = tensor_type.Layout().Dimensions()[2];

  proto::SamplerParameters sampler_params;
  sampler_params.set_k(1);
  sampler_params.set_p(0.5);
  sampler_params.set_temperature(1.0);
  sampler_params.set_seed(0);

  bool inference_called = false;
  ASSERT_OK_AND_ASSIGN(
      auto sampler,
      TopKOpenClSampler::Create(
          litert_env, /*batch_size=*/1, /*sequence_size=*/1, vocab_size,
          /*activation_data_type=*/std::nullopt, sampler_params));

  ASSERT_OK(sampler->SetInferenceFuncAndInputTensors(
      MarkBooleanTrue, &inference_called, &tokens_tensor,
      &prev_input_pos_tensor, &input_pos_tensor, &prev_mask_tensor,
      &mask_tensor, nullptr, nullptr));
  EXPECT_FALSE(inference_called);

  {
    LITERT_ASSERT_OK_AND_ASSIGN(auto input_pos_type,
                                prev_input_pos_tensor.TensorType());
    EXPECT_EQ(input_pos_type.ElementType(), ElementType::Int32);
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto prev_input_pos_lock_and_addr,
        TensorBufferScopedLock::Create(prev_input_pos_tensor,
                                       TensorBuffer::LockMode::kWrite));
    static_cast<int*>(prev_input_pos_lock_and_addr.second)[0] = kInputPos;

    LITERT_ASSERT_OK_AND_ASSIGN(auto mask_type, prev_mask_tensor.TensorType());
    EXPECT_EQ(mask_type.ElementType(), ElementType::Float32);
    LITERT_ASSERT_OK_AND_ASSIGN(auto num_mask_elements,
                                mask_type.Layout().NumElements());
    // Need 1 more element than kInputPos to check mask generation.
    EXPECT_GT(num_mask_elements, kInputPos + 2);
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto prev_mask_lock_and_addr,
        TensorBufferScopedLock::Create(prev_mask_tensor,
                                       TensorBuffer::LockMode::kWrite));
    for (int i = 0; i <= kInputPos; ++i) {
      static_cast<float*>(prev_mask_lock_and_addr.second)[i] = 0.0f;
    }
    for (int i = kInputPos + 1; i < num_mask_elements; ++i) {
      static_cast<float*>(prev_mask_lock_and_addr.second)[i] = -1.0f;
    }
  }

  int correct_index = 1, incorrect_index = 0;
  ASSERT_OK_AND_ASSIGN(auto ids, Sample(&litert_env, logits_tensor, *sampler,
                                        {correct_index}, {incorrect_index}));
  ASSERT_EQ(ids.size(), 1);
  EXPECT_EQ(ids[0], correct_index);

  // Inference should be called before Sample() returns.
  EXPECT_TRUE(inference_called);

  {
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto tokens_lock_and_addr,
        TensorBufferScopedLock::Create(tokens_tensor,
                                       TensorBuffer::LockMode::kRead));
    EXPECT_EQ(static_cast<int*>(tokens_lock_and_addr.second)[0], correct_index);

    LITERT_ASSERT_OK_AND_ASSIGN(
        auto input_pos_lock_and_addr,
        TensorBufferScopedLock::Create(input_pos_tensor,
                                       TensorBuffer::LockMode::kRead));
    EXPECT_EQ(static_cast<int*>(input_pos_lock_and_addr.second)[0],
              kInputPos + 1);

    LITERT_ASSERT_OK_AND_ASSIGN(
        auto mask_lock_and_addr,
        TensorBufferScopedLock::Create(mask_tensor,
                                       TensorBuffer::LockMode::kRead));
    static_cast<float*>(mask_lock_and_addr.second)[kInputPos + 1] = 0.0f;
  }
}

TEST(TopKOpenClSamplerTest, HandlesInputMultiTurn) {
  constexpr int kTurn1StartPos = 10;
  constexpr int kTurn2StartPos = 25;

  ASSERT_OK_AND_ASSIGN(auto ret, CreateLiteRTResources(kTestLmTaskPath));
  auto [litert_model, litert_env, compiled_model] = std::move(ret);

  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer logits_tensor,
      compiled_model.CreateOutputBuffer("decode", "logits"));
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer tokens_tensor,
      compiled_model.CreateInputBuffer("decode", "tokens"));
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer prev_input_pos_tensor,
      compiled_model.CreateInputBuffer("decode", "input_pos"));
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer input_pos_tensor,
      compiled_model.CreateInputBuffer("decode", "input_pos"));
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer prev_mask_tensor,
      compiled_model.CreateInputBuffer("decode", "mask"));
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer mask_tensor,
      compiled_model.CreateInputBuffer("decode", "mask"));

  LITERT_ASSERT_OK_AND_ASSIGN(auto tensor_type, logits_tensor.TensorType());
  int vocab_size = tensor_type.Layout().Dimensions()[2];

  proto::SamplerParameters sampler_params;
  sampler_params.set_k(1);
  sampler_params.set_p(0.5);
  sampler_params.set_temperature(1.0);
  sampler_params.set_seed(0);

  int inference_call_count = 0;
  auto count_inference = [](void* arg) -> int {
    (*static_cast<int*>(arg))++;
    return 0;
  };

  ASSERT_OK_AND_ASSIGN(
      auto sampler,
      TopKOpenClSampler::Create(
          litert_env, /*batch_size=*/1, /*sequence_size=*/1, vocab_size,
          /*activation_data_type=*/std::nullopt, sampler_params));

  EXPECT_TRUE(sampler->CanHandleInput());
  EXPECT_FALSE(sampler->HandlesInput());

  // === Turn 1 ===
  ASSERT_OK(sampler->SetInferenceFuncAndInputTensors(
      count_inference, &inference_call_count, &tokens_tensor,
      &prev_input_pos_tensor, &input_pos_tensor, &prev_mask_tensor,
      &mask_tensor, nullptr, nullptr));
  EXPECT_TRUE(sampler->HandlesInput());

  {
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto prev_input_pos_lock_and_addr,
        TensorBufferScopedLock::Create(prev_input_pos_tensor,
                                       TensorBuffer::LockMode::kWrite));
    static_cast<int*>(prev_input_pos_lock_and_addr.second)[0] = kTurn1StartPos;
  }

  int turn1_token1 = 1;
  ASSERT_OK_AND_ASSIGN(
      auto ids1, Sample(&litert_env, logits_tensor, *sampler, {turn1_token1}));
  ASSERT_EQ(ids1.size(), 1);
  EXPECT_EQ(ids1[0], turn1_token1);
  EXPECT_EQ(inference_call_count, 1);

  {
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto input_pos_lock_and_addr,
        TensorBufferScopedLock::Create(input_pos_tensor,
                                       TensorBuffer::LockMode::kRead));
    EXPECT_EQ(static_cast<int*>(input_pos_lock_and_addr.second)[0],
              kTurn1StartPos + 1);
  }

  {
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto prev_input_pos_lock_and_addr,
        TensorBufferScopedLock::Create(prev_input_pos_tensor,
                                       TensorBuffer::LockMode::kWrite));
    static_cast<int*>(prev_input_pos_lock_and_addr.second)[0] =
        kTurn1StartPos + 1;
  }

  int turn1_token2 = 2;
  ASSERT_OK_AND_ASSIGN(
      auto ids2, Sample(&litert_env, logits_tensor, *sampler, {turn1_token2}));
  ASSERT_EQ(ids2.size(), 1);
  EXPECT_EQ(ids2[0], turn1_token2);
  EXPECT_EQ(inference_call_count, 2);

  {
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto input_pos_lock_and_addr,
        TensorBufferScopedLock::Create(input_pos_tensor,
                                       TensorBuffer::LockMode::kRead));
    EXPECT_EQ(static_cast<int*>(input_pos_lock_and_addr.second)[0],
              kTurn1StartPos + 2);
  }

  // === Reset input handling between turns ===
  ASSERT_OK(sampler->SetInferenceFuncAndInputTensors(
      nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
      nullptr));
  EXPECT_FALSE(sampler->HandlesInput());

  int prefill_token = 3;
  ASSERT_OK_AND_ASSIGN(auto ids_prefill, Sample(&litert_env, logits_tensor,
                                                *sampler, {prefill_token}));
  ASSERT_EQ(ids_prefill.size(), 1);
  EXPECT_EQ(ids_prefill[0], prefill_token);
  EXPECT_EQ(inference_call_count, 2);

  // === Turn 2 ===
  ASSERT_OK(sampler->SetInferenceFuncAndInputTensors(
      count_inference, &inference_call_count, &tokens_tensor,
      &prev_input_pos_tensor, &input_pos_tensor, &prev_mask_tensor,
      &mask_tensor, nullptr, nullptr));
  EXPECT_TRUE(sampler->HandlesInput());

  {
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto prev_input_pos_lock_and_addr,
        TensorBufferScopedLock::Create(prev_input_pos_tensor,
                                       TensorBuffer::LockMode::kWrite));
    static_cast<int*>(prev_input_pos_lock_and_addr.second)[0] = kTurn2StartPos;
  }

  int turn2_token1 = 4;
  ASSERT_OK_AND_ASSIGN(auto ids_t2_1, Sample(&litert_env, logits_tensor,
                                             *sampler, {turn2_token1}));
  ASSERT_EQ(ids_t2_1.size(), 1);
  EXPECT_EQ(ids_t2_1[0], turn2_token1);
  EXPECT_EQ(inference_call_count, 3);

  {
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto input_pos_lock_and_addr,
        TensorBufferScopedLock::Create(input_pos_tensor,
                                       TensorBuffer::LockMode::kRead));
    EXPECT_EQ(static_cast<int*>(input_pos_lock_and_addr.second)[0],
              kTurn2StartPos + 1);
  }
}

}  // namespace
}  // namespace litert::lm
#endif  // !MEMORY_SANITIZER && !THREAD_SANITIZER
