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

#include "runtime/executor/llm_litert_compiled_model_executor.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>  // NOLINT: Required for path manipulation.
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <system_error>  // NOLINT: Required for std::error_code used with std::filesystem.
#include <tuple>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/test/matchers.h"  // from @litert
#include "runtime/components/constrained_decoding/constrained_decoder.h"
#include "runtime/components/constrained_decoding/fake_constraint.h"
#include "runtime/components/model_resources.h"
#include "runtime/components/model_resources_litert_lm.h"
#include "runtime/components/sampler.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/litert_lm_loader.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/status_macros.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

using ::testing::status::StatusIs;

constexpr char kTestStaticModelPath[] =
    "litert_lm/runtime/testdata/test_lm.litertlm";

// Test model with dynamic sequence and context length dimensions.
constexpr char kTestDynamicModelPath[] =
    "litert_lm/runtime/testdata/test_lm_dynamic.litertlm";

const int kMaxNumTokens = 32;
const int kNumThreads = 4;

absl::StatusOr<std::unique_ptr<ModelResources>>
CreateExecutorModelResourcesLitertLm(absl::string_view model_path) {
  ABSL_ASSIGN_OR_RETURN(auto scoped_file, ScopedFile::Open(model_path));
  ABSL_ASSIGN_OR_RETURN(auto loader,
                        LitertLmLoader::Create(std::move(scoped_file)));
  return ModelResourcesLitertLm::Create(std::move(loader));
}

TEST(LlmLiteRtCompiledModelExecutorStaticTest,
     CreateExecutorTest_WithoutCache) {
  auto model_path =
      std::filesystem::path(::testing::SrcDir()) / kTestStaticModelPath;
  ASSERT_OK_AND_ASSIGN(
      auto model_resources,
      CreateExecutorModelResourcesLitertLm(model_path.string()));
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create(model_path.string()));
  auto executor_settings =
      LlmExecutorSettings::CreateDefault(model_assets, Backend::CPU);
  executor_settings->SetCacheDir(":nocache");
  executor_settings->SetMaxNumTokens(kMaxNumTokens);
  ::litert::lm::CpuConfig config;
  config.number_of_threads = kNumThreads;
  executor_settings->SetBackendConfig(config);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));
  auto executor = LlmLiteRtCompiledModelExecutorStatic::Create(
      *executor_settings, env, *model_resources);
  ASSERT_OK(executor);
  ASSERT_NE(*executor, nullptr);
}

TEST(LlmLiteRtCompiledModelExecutorStaticTest,
     CreateExecutorTest_WithProfiling) {
  auto model_path =
      std::filesystem::path(::testing::SrcDir()) / kTestStaticModelPath;
  ASSERT_OK_AND_ASSIGN(
      auto model_resources,
      CreateExecutorModelResourcesLitertLm(model_path.string()));
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create(model_path.string()));
  ASSERT_OK_AND_ASSIGN(
      auto executor_settings,
      LlmExecutorSettings::CreateDefault(model_assets, Backend::CPU));
  executor_settings.SetCacheDir(":nocache");
  executor_settings.SetMaxNumTokens(kMaxNumTokens);
  executor_settings.SetAdvancedSettings(AdvancedSettings{
      .enable_profiling = true,
  });
  ::litert::lm::CpuConfig config;
  config.number_of_threads = kNumThreads;
  executor_settings.SetBackendConfig(config);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));
  ASSERT_OK_AND_ASSIGN(auto executor,
                       LlmLiteRtCompiledModelExecutorStatic::Create(
                           executor_settings, env, *model_resources));
  ASSERT_NE(executor, nullptr);

  ExecutorInputs inputs;
  const std::vector<int> input_tokens = {1, 2, 0};
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto input_tokens_buffer,
      CopyToTensorBuffer<int>(absl::MakeSpan(input_tokens), {1, 3}));
  inputs.SetTextData(ExecutorTextData(std::move(input_tokens_buffer)));

  EXPECT_OK(executor->Prefill(inputs));

  ASSERT_OK_AND_ASSIGN(auto summary, executor->GetProfileSummary());
  EXPECT_FALSE(summary.empty());
}

TEST(LlmLiteRtCompiledModelExecutorStaticTest,
     CreateExecutorTest_WithoutProfiling) {
  auto model_path =
      std::filesystem::path(::testing::SrcDir()) / kTestStaticModelPath;
  ASSERT_OK_AND_ASSIGN(
      auto model_resources,
      CreateExecutorModelResourcesLitertLm(model_path.string()));
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create(model_path.string()));
  ASSERT_OK_AND_ASSIGN(
      auto executor_settings,
      LlmExecutorSettings::CreateDefault(model_assets, Backend::CPU));
  executor_settings.SetCacheDir(":nocache");
  executor_settings.SetMaxNumTokens(kMaxNumTokens);
  executor_settings.SetAdvancedSettings(AdvancedSettings{
      .enable_profiling = false,
  });
  ::litert::lm::CpuConfig config;
  config.number_of_threads = kNumThreads;
  executor_settings.SetBackendConfig(config);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));
  ASSERT_OK_AND_ASSIGN(auto executor,
                       LlmLiteRtCompiledModelExecutorStatic::Create(
                           executor_settings, env, *model_resources));
  ASSERT_NE(executor, nullptr);

  ExecutorInputs inputs;
  const std::vector<int> input_tokens = {1, 2, 0};
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto input_tokens_buffer,
      CopyToTensorBuffer<int>(absl::MakeSpan(input_tokens), {1, 3}));
  inputs.SetTextData(ExecutorTextData(std::move(input_tokens_buffer)));

  EXPECT_OK(executor->Prefill(inputs));

  auto summary = executor->GetProfileSummary();
  EXPECT_EQ(summary.status().code(), absl::StatusCode::kFailedPrecondition);
}

TEST(LlmLiteRtCompiledModelExecutorStaticTest, PrefillTest) {
  auto model_path =
      std::filesystem::path(::testing::SrcDir()) / kTestStaticModelPath;
  ASSERT_OK_AND_ASSIGN(
      auto model_resources,
      CreateExecutorModelResourcesLitertLm(model_path.string()));
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create(model_path.string()));
  auto executor_settings =
      LlmExecutorSettings::CreateDefault(model_assets, Backend::CPU);
  executor_settings->SetCacheDir(":nocache");
  executor_settings->SetMaxNumTokens(kMaxNumTokens);
  ::litert::lm::CpuConfig config;
  config.number_of_threads = kNumThreads;
  executor_settings->SetBackendConfig(config);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));
  ASSERT_OK_AND_ASSIGN(auto executor,
                       LlmLiteRtCompiledModelExecutorStatic::Create(
                           *executor_settings, env, *model_resources));
  ASSERT_NE(executor, nullptr);

  ExecutorInputs inputs;
  // Create a tensor buffer with 3 elements but only the first two elements
  // are actually processed.
  const std::vector<int> input_tokens = {1, 2, 0};
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto input_tokens_buffer,
      CopyToTensorBuffer<int>(absl::MakeSpan(input_tokens), {1, 3}));
  inputs.SetTextData(ExecutorTextData(std::move(input_tokens_buffer)));

  EXPECT_OK(executor->Prefill(inputs));

  ASSERT_OK_AND_ASSIGN(auto current_step, executor->GetCurrentStep());

  EXPECT_EQ(current_step, 3);
}

TEST(LlmLiteRtCompiledModelExecutorStaticTest, ResetTest) {
  auto model_path =
      std::filesystem::path(::testing::SrcDir()) / kTestStaticModelPath;
  ASSERT_OK_AND_ASSIGN(
      auto model_resources,
      CreateExecutorModelResourcesLitertLm(model_path.string()));
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create(model_path.string()));
  ASSERT_OK_AND_ASSIGN(
      auto executor_settings,
      LlmExecutorSettings::CreateDefault(model_assets, Backend::CPU));
  executor_settings.SetCacheDir(":nocache");
  executor_settings.SetMaxNumTokens(kMaxNumTokens);
  ::litert::lm::CpuConfig config;
  config.number_of_threads = kNumThreads;
  executor_settings.SetBackendConfig(config);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));
  ASSERT_OK_AND_ASSIGN(auto executor,
                       LlmLiteRtCompiledModelExecutorStatic::Create(
                           executor_settings, env, *model_resources));
  ASSERT_NE(executor, nullptr);

  ExecutorInputs inputs;
  const std::vector<int> input_tokens = {1, 2, 0};
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto input_tokens_buffer,
      CopyToTensorBuffer<int>(absl::MakeSpan(input_tokens), {1, 3}));
  inputs.SetTextData(ExecutorTextData(std::move(input_tokens_buffer)));

  EXPECT_OK(executor->Prefill(inputs));

  {
    ASSERT_OK_AND_ASSIGN(auto current_step, executor->GetCurrentStep());
    EXPECT_EQ(current_step, 3);
  }

  EXPECT_OK(executor->Reset());

  {
    ASSERT_OK_AND_ASSIGN(auto current_step, executor->GetCurrentStep());
    EXPECT_EQ(current_step, 0);
  }

  const std::vector<int> second_input_tokens = {3, 4, 5};
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto second_input_tokens_buffer,
      CopyToTensorBuffer<int>(absl::MakeSpan(second_input_tokens), {1, 3}));
  ExecutorInputs second_inputs;
  second_inputs.SetTextData(
      ExecutorTextData(std::move(second_input_tokens_buffer)));

  EXPECT_OK(executor->Prefill(second_inputs));

  {
    ASSERT_OK_AND_ASSIGN(auto current_step, executor->GetCurrentStep());
    EXPECT_EQ(current_step, 3);
  }
}

TEST(LlmLiteRtCompiledModelExecutorStaticTest,
     PrefillExceedsStateEntriesErrorTest) {
  auto model_path =
      std::filesystem::path(::testing::SrcDir()) / kTestStaticModelPath;
  ASSERT_OK_AND_ASSIGN(
      auto model_resources,
      CreateExecutorModelResourcesLitertLm(model_path.string()));
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create(model_path.string()));
  ASSERT_OK_AND_ASSIGN(
      auto executor_settings,
      LlmExecutorSettings::CreateDefault(model_assets, Backend::CPU));
  executor_settings.SetCacheDir(":nocache");
  executor_settings.SetMaxNumTokens(kMaxNumTokens);
  ::litert::lm::CpuConfig config;
  config.number_of_threads = kNumThreads;
  executor_settings.SetBackendConfig(config);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));
  ASSERT_OK_AND_ASSIGN(auto executor,
                       LlmLiteRtCompiledModelExecutorStatic::Create(
                           executor_settings, env, *model_resources));
  ASSERT_NE(executor, nullptr);

  // Create a tensor buffer with 165 tokens, which requires work groups total >
  // max state entries (160).
  const std::vector<int> input_tokens(165, 1);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto input_tokens_buffer,
      CopyToTensorBuffer<int>(absl::MakeSpan(input_tokens), {1, 165}));
  ExecutorInputs inputs;
  inputs.SetTextData(ExecutorTextData(std::move(input_tokens_buffer)));

  EXPECT_THAT(executor->Prefill(inputs),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(LlmLiteRtCompiledModelExecutorStaticTest,
     PrefillExceedsRemainingStateEntriesErrorTest) {
  auto model_path =
      std::filesystem::path(::testing::SrcDir()) / kTestStaticModelPath;
  ASSERT_OK_AND_ASSIGN(
      auto model_resources,
      CreateExecutorModelResourcesLitertLm(model_path.string()));
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create(model_path.string()));
  ASSERT_OK_AND_ASSIGN(
      auto executor_settings,
      LlmExecutorSettings::CreateDefault(model_assets, Backend::CPU));
  executor_settings.SetCacheDir(":nocache");
  executor_settings.SetMaxNumTokens(kMaxNumTokens);
  ::litert::lm::CpuConfig config;
  config.number_of_threads = kNumThreads;
  executor_settings.SetBackendConfig(config);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));
  ASSERT_OK_AND_ASSIGN(auto executor,
                       LlmLiteRtCompiledModelExecutorStatic::Create(
                           executor_settings, env, *model_resources));
  ASSERT_NE(executor, nullptr);

  // First prefill of 100 tokens (within 128 capacity).
  const std::vector<int> first_input_tokens(100, 1);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto first_buffer,
      CopyToTensorBuffer<int>(absl::MakeSpan(first_input_tokens), {1, 100}));
  ExecutorInputs first_inputs;
  first_inputs.SetTextData(ExecutorTextData(std::move(first_buffer)));
  EXPECT_OK(executor->Prefill(first_inputs));

  // Second prefill of 30 tokens. Current step is 100, remaining capacity is 28.
  const std::vector<int> second_input_tokens(30, 1);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto second_buffer,
      CopyToTensorBuffer<int>(absl::MakeSpan(second_input_tokens), {1, 30}));
  ExecutorInputs second_inputs;
  second_inputs.SetTextData(ExecutorTextData(std::move(second_buffer)));
  EXPECT_THAT(executor->Prefill(second_inputs),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(LlmLiteRtCompiledModelExecutorStaticTest, DecodeTest) {
  auto model_path =
      std::filesystem::path(::testing::SrcDir()) / kTestStaticModelPath;
  ASSERT_OK_AND_ASSIGN(
      auto model_resources,
      CreateExecutorModelResourcesLitertLm(model_path.string()));
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create(model_path.string()));
  auto executor_settings =
      LlmExecutorSettings::CreateDefault(model_assets, Backend::CPU);
  executor_settings->SetCacheDir(":nocache");
  executor_settings->SetMaxNumTokens(kMaxNumTokens);
  ::litert::lm::CpuConfig config;
  config.number_of_threads = kNumThreads;
  executor_settings->SetBackendConfig(config);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));
  ASSERT_OK_AND_ASSIGN(auto executor,
                       LlmLiteRtCompiledModelExecutorStatic::Create(
                           *executor_settings, env, *model_resources));
  ASSERT_NE(executor, nullptr);

  // An explicitly present but unspecified sampler config should use the
  // default CPU sampler rather than disabling sampling.
  RuntimeConfig runtime_config;
  runtime_config.sampler_params = proto::SamplerParameters();
  ASSERT_OK_AND_ASSIGN(
      auto context, executor->CreateNewContext(std::nullopt, runtime_config));
  ASSERT_OK(executor->RestoreContext(std::move(context)));

  ExecutorInputs inputs;
  // Create a tensor buffer with 3 elements.
  const std::vector<int> input_tokens = {1, 2, 0};
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto input_tokens_buffer,
      CopyToTensorBuffer<int>(absl::MakeSpan(input_tokens), {1, 3}));
  inputs.SetTextData(ExecutorTextData(std::move(input_tokens_buffer)));

  EXPECT_OK(executor->Prefill(inputs));

  {
    ASSERT_OK_AND_ASSIGN(auto current_step, executor->GetCurrentStep());
    EXPECT_EQ(current_step, 3);
  }

  {
    ASSERT_OK_AND_ASSIGN(auto output_tokens, executor->Decode());

    ASSERT_OK_AND_ASSIGN(auto current_step, executor->GetCurrentStep());
    EXPECT_EQ(current_step, 4);

    ASSERT_EQ(output_tokens.size(), 1);
    ASSERT_EQ(output_tokens[0].size(), 1);
    EXPECT_EQ(output_tokens[0][0], 126670);
  }

  {
    ASSERT_OK_AND_ASSIGN(auto output_tokens, executor->Decode());

    ASSERT_OK_AND_ASSIGN(auto current_step, executor->GetCurrentStep());
    EXPECT_EQ(current_step, 5);

    ASSERT_EQ(output_tokens.size(), 1);
    ASSERT_EQ(output_tokens[0].size(), 1);
    EXPECT_EQ(output_tokens[0][0], 126670);
  }
}

class FakeInvalidTokenSampler : public Sampler {
 public:
  absl::Status SampleToIdAndScoreBuffer(const TensorBuffer& logits_tensor,
                                        TensorBuffer& ids_tensor,
                                        TensorBuffer* scores_tensor) override {
    LITERT_ASSIGN_OR_RETURN(auto lock_and_addr,
                            TensorBufferScopedLock::Create(
                                ids_tensor, TensorBuffer::LockMode::kWrite));
    int* ptr = static_cast<int*>(const_cast<void*>(lock_and_addr.second));
    ptr[0] = -1;
    return absl::OkStatus();
  }

  absl::Status UpdateConfig(
      const proto::SamplerParameters& sampler_params, int batch_size,
      std::shared_ptr<std::default_random_engine> rand_gen) override {
    return absl::OkStatus();
  }
};

class FakeSamplerWithInputHandling : public Sampler {
 public:
  bool CanHandleInput() const override { return true; }
  bool HandlesInput() const override { return inference_func_ != nullptr; }

  absl::Status SetInferenceFuncAndInputTensors(
      int (*run_inference_func)(void* arg), void* arg,
      const TensorBuffer* ids_tensor,
      const TensorBuffer* prev_input_positions_tensor,
      const TensorBuffer* input_positions_tensor,
      const TensorBuffer* prev_mask_tensor, const TensorBuffer* mask_tensor,
      const TensorBuffer* prev_param_tensor,
      const TensorBuffer* param_tensor) override {
    inference_func_ = run_inference_func;
    arg_ = arg;
    prev_input_positions_tensor_ = prev_input_positions_tensor;
    input_positions_tensor_ = input_positions_tensor;
    if (run_inference_func != nullptr) {
      set_count_++;
    } else {
      reset_count_++;
    }
    return absl::OkStatus();
  }

  absl::Status SampleToIdAndScoreBuffer(const TensorBuffer& logits_tensor,
                                        TensorBuffer& ids_tensor,
                                        TensorBuffer* scores_tensor) override {
    sample_count_++;
    LITERT_ASSIGN_OR_RETURN(auto lock_and_addr,
                            TensorBufferScopedLock::Create(
                                ids_tensor, TensorBuffer::LockMode::kWrite));
    int* ptr = static_cast<int*>(const_cast<void*>(lock_and_addr.second));
    ptr[0] = sample_count_;

    if (inference_func_ != nullptr) {
      inference_called_count_++;
    }
    return absl::OkStatus();
  }

  absl::Status UpdateConfig(
      const proto::SamplerParameters& sampler_params, int batch_size,
      std::shared_ptr<std::default_random_engine> rand_gen) override {
    return absl::OkStatus();
  }

  int set_count() const { return set_count_; }
  int reset_count() const { return reset_count_; }
  int sample_count() const { return sample_count_; }
  int inference_called_count() const { return inference_called_count_; }

 private:
  int (*inference_func_)(void* arg) = nullptr;
  void* arg_ = nullptr;
  const TensorBuffer* prev_input_positions_tensor_ = nullptr;
  const TensorBuffer* input_positions_tensor_ = nullptr;
  int set_count_ = 0;
  int reset_count_ = 0;
  int sample_count_ = 0;
  int inference_called_count_ = 0;
};

class TestableLlmLiteRtCompiledModelExecutorStatic
    : public LlmLiteRtCompiledModelExecutorStatic {
 public:
  static void SetSamplerForTest(LlmLiteRtCompiledModelExecutorStatic* executor,
                                std::unique_ptr<Sampler> sampler) {
    auto* testable =
        static_cast<TestableLlmLiteRtCompiledModelExecutorStatic*>(executor);
    testable->sampler_ = std::move(sampler);
    testable->sampler_handles_input_ = false;
  }

  static void SetSamplerWithInputHandlingForTest(
      LlmLiteRtCompiledModelExecutorStatic* executor,
      std::unique_ptr<Sampler> sampler) {
    auto* testable =
        static_cast<TestableLlmLiteRtCompiledModelExecutorStatic*>(executor);
    testable->sampler_ = std::move(sampler);
    testable->sampler_handles_input_ = true;
  }
};

TEST(LlmLiteRtCompiledModelExecutorStaticTest,
     SamplerInputHandlingMultiTurnTest) {
  auto model_path =
      std::filesystem::path(::testing::SrcDir()) / kTestStaticModelPath;
  ASSERT_OK_AND_ASSIGN(
      auto model_resources,
      CreateExecutorModelResourcesLitertLm(model_path.string()));
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create(model_path.string()));
  ASSERT_OK_AND_ASSIGN(
      auto executor_settings,
      LlmExecutorSettings::CreateDefault(model_assets, Backend::CPU));
  executor_settings.SetCacheDir(":nocache");
  executor_settings.SetMaxNumTokens(kMaxNumTokens);
  executor_settings.SetAdvancedSettings(AdvancedSettings{
      .sampler_handles_input = true,
  });
  ::litert::lm::CpuConfig config;
  config.number_of_threads = kNumThreads;
  executor_settings.SetBackendConfig(config);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));
  ASSERT_OK_AND_ASSIGN(auto executor,
                       LlmLiteRtCompiledModelExecutorStatic::Create(
                           executor_settings, env, *model_resources));
  ASSERT_NE(executor, nullptr);

  auto fake_sampler = std::make_unique<FakeSamplerWithInputHandling>();
  auto* fake_sampler_ptr = fake_sampler.get();
  TestableLlmLiteRtCompiledModelExecutorStatic::
      SetSamplerWithInputHandlingForTest(executor.get(),
                                         std::move(fake_sampler));

  // Turn 1 Prefill
  ExecutorInputs inputs1;
  const std::vector<int> turn1_tokens = {1, 2, 3};
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto input_tokens_buffer1,
      CopyToTensorBuffer<int>(absl::MakeSpan(turn1_tokens), {1, 3}));
  inputs1.SetTextData(ExecutorTextData(std::move(input_tokens_buffer1)));
  EXPECT_OK(executor->Prefill(inputs1));

  // Turn 1 Decode - Step 1
  ASSERT_OK_AND_ASSIGN(auto out_tokens1_step1, executor->Decode());
  EXPECT_EQ(fake_sampler_ptr->set_count(), 1);
  EXPECT_EQ(fake_sampler_ptr->inference_called_count(), 1);

  // Turn 1 Decode - Step 2
  ASSERT_OK_AND_ASSIGN(auto out_tokens1_step2, executor->Decode());
  EXPECT_EQ(fake_sampler_ptr->inference_called_count(), 2);

  // Turn 2 Prefill (should reset sampler input handling)
  ExecutorInputs inputs2;
  const std::vector<int> turn2_tokens = {4, 5};
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto input_tokens_buffer2,
      CopyToTensorBuffer<int>(absl::MakeSpan(turn2_tokens), {1, 2}));
  inputs2.SetTextData(ExecutorTextData(std::move(input_tokens_buffer2)));
  EXPECT_OK(executor->Prefill(inputs2));

  // Sampler input handling should be reset during prefill
  EXPECT_GT(fake_sampler_ptr->reset_count(), 0);
  EXPECT_FALSE(fake_sampler_ptr->HandlesInput());

  // Turn 2 Decode - Step 1
  ASSERT_OK_AND_ASSIGN(auto out_tokens2_step1, executor->Decode());
  // Input handling should be re-enabled on first decode step of Turn 2
  EXPECT_GT(fake_sampler_ptr->set_count(), 1);
  EXPECT_EQ(fake_sampler_ptr->inference_called_count(), 3);

  // Turn 2 Decode - Step 2
  ASSERT_OK_AND_ASSIGN(auto out_tokens2_step2, executor->Decode());
  EXPECT_EQ(fake_sampler_ptr->inference_called_count(), 4);
}

TEST(LlmLiteRtCompiledModelExecutorStaticTest,
     ErrorOnInvalidSampledTokenId_True) {
  auto model_path =
      std::filesystem::path(::testing::SrcDir()) / kTestStaticModelPath;
  ASSERT_OK_AND_ASSIGN(
      auto model_resources,
      CreateExecutorModelResourcesLitertLm(model_path.string()));
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create(model_path.string()));
  ASSERT_OK_AND_ASSIGN(
      auto executor_settings,
      LlmExecutorSettings::CreateDefault(model_assets, Backend::CPU));
  executor_settings.SetCacheDir(":nocache");
  executor_settings.SetMaxNumTokens(kMaxNumTokens);
  executor_settings.SetAdvancedSettings(AdvancedSettings{
      .error_on_invalid_sampled_token_id = true,
  });
  ::litert::lm::CpuConfig config;
  config.number_of_threads = kNumThreads;
  executor_settings.SetBackendConfig(config);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));
  ASSERT_OK_AND_ASSIGN(auto executor,
                       LlmLiteRtCompiledModelExecutorStatic::Create(
                           executor_settings, env, *model_resources));
  ASSERT_NE(executor, nullptr);

  ExecutorInputs inputs;
  const std::vector<int> input_tokens = {1, 2, 0};
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto input_tokens_buffer,
      CopyToTensorBuffer<int>(absl::MakeSpan(input_tokens), {1, 3}));
  inputs.SetTextData(ExecutorTextData(std::move(input_tokens_buffer)));

  EXPECT_OK(executor->Prefill(inputs));

  TestableLlmLiteRtCompiledModelExecutorStatic::SetSamplerForTest(
      executor.get(), std::make_unique<FakeInvalidTokenSampler>());

  auto decode_status = executor->Decode();
  EXPECT_EQ(decode_status.status().code(), absl::StatusCode::kInternal);
  EXPECT_THAT(
      decode_status.status().message(),
      ::testing::HasSubstr(
          "Invalid decode and sample result. The sampled token is negative."));
}

TEST(LlmLiteRtCompiledModelExecutorStaticTest,
     ErrorOnInvalidSampledTokenId_False) {
  auto model_path =
      std::filesystem::path(::testing::SrcDir()) / kTestStaticModelPath;
  ASSERT_OK_AND_ASSIGN(
      auto model_resources,
      CreateExecutorModelResourcesLitertLm(model_path.string()));
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create(model_path.string()));
  ASSERT_OK_AND_ASSIGN(
      auto executor_settings,
      LlmExecutorSettings::CreateDefault(model_assets, Backend::CPU));
  executor_settings.SetCacheDir(":nocache");
  executor_settings.SetMaxNumTokens(kMaxNumTokens);
  executor_settings.SetAdvancedSettings(AdvancedSettings{
      .error_on_invalid_sampled_token_id = false,
  });
  ::litert::lm::CpuConfig config;
  config.number_of_threads = kNumThreads;
  executor_settings.SetBackendConfig(config);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));
  ASSERT_OK_AND_ASSIGN(auto executor,
                       LlmLiteRtCompiledModelExecutorStatic::Create(
                           executor_settings, env, *model_resources));
  ASSERT_NE(executor, nullptr);

  ExecutorInputs inputs;
  const std::vector<int> input_tokens = {1, 2, 0};
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto input_tokens_buffer,
      CopyToTensorBuffer<int>(absl::MakeSpan(input_tokens), {1, 3}));
  inputs.SetTextData(ExecutorTextData(std::move(input_tokens_buffer)));

  EXPECT_OK(executor->Prefill(inputs));

  TestableLlmLiteRtCompiledModelExecutorStatic::SetSamplerForTest(
      executor.get(), std::make_unique<FakeInvalidTokenSampler>());

  ASSERT_OK_AND_ASSIGN(auto output_tokens, executor->Decode());
  ASSERT_EQ(output_tokens.size(), 1);
  ASSERT_EQ(output_tokens[0].size(), 1);
  EXPECT_EQ(output_tokens[0][0], 0);
}

TEST(LlmLiteRtCompiledModelExecutorStaticTest, ConstrainedDecodeTest) {
  auto model_path =
      std::filesystem::path(::testing::SrcDir()) / kTestStaticModelPath;
  ASSERT_OK_AND_ASSIGN(
      auto model_resources,
      CreateExecutorModelResourcesLitertLm(model_path.string()));
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create(model_path.string()));
  auto executor_settings =
      LlmExecutorSettings::CreateDefault(model_assets, Backend::CPU);
  executor_settings->SetCacheDir(":nocache");
  executor_settings->SetMaxNumTokens(kMaxNumTokens);
  ::litert::lm::CpuConfig config;
  config.number_of_threads = kNumThreads;
  executor_settings->SetBackendConfig(config);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));
  ASSERT_OK_AND_ASSIGN(auto executor,
                       LlmLiteRtCompiledModelExecutorStatic::Create(
                           *executor_settings, env, *model_resources));
  ASSERT_NE(executor, nullptr);

  ExecutorInputs inputs;
  // Create a tensor buffer with 3 elements.
  const std::vector<int> input_tokens = {1, 2, 0};
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto input_tokens_buffer,
      CopyToTensorBuffer<int>(absl::MakeSpan(input_tokens), {1, 3}));
  inputs.SetTextData(ExecutorTextData(std::move(input_tokens_buffer)));

  EXPECT_OK(executor->Prefill(inputs));

  {
    ASSERT_OK_AND_ASSIGN(auto current_step, executor->GetCurrentStep());
    EXPECT_EQ(current_step, 3);
  }

  ExecutorDecodeParams params;

  auto constraint = FakeConstraint({2, 3}, /*vocabulary_size=*/262144);
  ConstrainedDecoder constrained_decoder(&constraint, /*batch_size=*/1);
  params.SetConstrainedDecoder(&constrained_decoder);

  {
    ASSERT_OK_AND_ASSIGN(auto output_tokens, executor->Decode(params));

    ASSERT_OK_AND_ASSIGN(auto current_step, executor->GetCurrentStep());
    EXPECT_EQ(current_step, 4);

    ASSERT_EQ(output_tokens.size(), 1);
    ASSERT_EQ(output_tokens[0].size(), 1);
    EXPECT_EQ(output_tokens[0][0], 2);
  }

  {
    ASSERT_OK_AND_ASSIGN(auto output_tokens, executor->Decode(params));

    ASSERT_OK_AND_ASSIGN(auto current_step, executor->GetCurrentStep());
    EXPECT_EQ(current_step, 5);

    ASSERT_EQ(output_tokens.size(), 1);
    ASSERT_EQ(output_tokens[0].size(), 1);
    EXPECT_EQ(output_tokens[0][0], 3);
  }
}

TEST(LlmLiteRtCompiledModelExecutorStaticTest, DecodeLogitsTest) {
  auto model_path =
      std::filesystem::path(::testing::SrcDir()) / kTestStaticModelPath;
  ASSERT_OK_AND_ASSIGN(
      auto model_resources,
      CreateExecutorModelResourcesLitertLm(model_path.string()));
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create(model_path.string()));
  auto executor_settings =
      LlmExecutorSettings::CreateDefault(model_assets, Backend::CPU);
  executor_settings->SetCacheDir(":nocache");
  executor_settings->SetMaxNumTokens(kMaxNumTokens);
  ::litert::lm::CpuConfig config;
  config.number_of_threads = kNumThreads;
  executor_settings->SetBackendConfig(config);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));
  ASSERT_OK_AND_ASSIGN(auto executor,
                       LlmLiteRtCompiledModelExecutorStatic::Create(
                           *executor_settings, env, *model_resources));
  ASSERT_NE(executor, nullptr);

  ExecutorInputs inputs;
  // Create a tensor buffer with 1 element.
  const std::vector<int> input_tokens = {1};
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto input_tokens_buffer,
      CopyToTensorBuffer<int>(absl::MakeSpan(input_tokens), {1, 1}));
  inputs.SetTextData(ExecutorTextData(std::move(input_tokens_buffer)));

  EXPECT_OK(executor->Prefill(inputs));

  {
    ASSERT_OK_AND_ASSIGN(auto current_step, executor->GetCurrentStep());
    EXPECT_EQ(current_step, 1);
  }

  LITERT_ASSERT_OK_AND_ASSIGN(auto output_tokens,
                              CreateTensorBuffer<int>({1, 1}));

  {
    ASSERT_OK_AND_ASSIGN(auto output_logits, executor->DecodeLogits(inputs));

    ASSERT_OK_AND_ASSIGN(auto current_step, executor->GetCurrentStep());
    EXPECT_EQ(current_step, 2);

    auto output_logits_span = ReferTensorBufferAsSpan<float>(output_logits);
    EXPECT_TRUE(output_logits_span.HasValue());
  }
}

TEST(LlmLiteRtCompiledModelExecutorStaticTest, UpdateExecutorSettingsTest) {
  auto model_path =
      std::filesystem::path(::testing::SrcDir()) / kTestStaticModelPath;
  ASSERT_OK_AND_ASSIGN(
      auto model_resources,
      CreateExecutorModelResourcesLitertLm(model_path.string()));
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create(model_path.string()));
  auto executor_settings =
      LlmExecutorSettings::CreateDefault(model_assets, Backend::CPU);
  ASSERT_OK(executor_settings);
  executor_settings->SetMaxNumTokens(kMaxNumTokens);

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));
  ASSERT_OK_AND_ASSIGN(auto executor,
                       LlmLiteRtCompiledModelExecutorStatic::Create(
                           *executor_settings, env, *model_resources));

  auto new_executor_settings =
      LlmExecutorSettings::CreateDefault(model_assets, Backend::GPU);
  ASSERT_OK(new_executor_settings);
  new_executor_settings->SetMaxNumTokens(kMaxNumTokens + 1);

  EXPECT_OK(executor->UpdateExecutorSettings(*new_executor_settings));

  ASSERT_OK_AND_ASSIGN(auto updated_settings, executor->GetExecutorSettings());
  EXPECT_EQ(updated_settings.GetBackend(), Backend::GPU);
  EXPECT_EQ(updated_settings.GetMaxNumTokens(), kMaxNumTokens + 1);
}

TEST(LlmLiteRtCompiledModelExecutorStaticTest, CreateExecutorTest_WithCache) {
  auto cache_path = std::filesystem::path(::testing::TempDir()) /
                    absl::StrCat("cache-", std::rand());
  std::filesystem::remove_all(cache_path);
  absl::Cleanup remove_cache = [cache_path] {
    std::filesystem::remove_all(cache_path);
  };

  auto model_path =
      std::filesystem::path(::testing::SrcDir()) / kTestStaticModelPath;
  ASSERT_OK_AND_ASSIGN(
      auto model_resources,
      CreateExecutorModelResourcesLitertLm(model_path.string()));
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create(model_path.string()));
  auto executor_settings =
      LlmExecutorSettings::CreateDefault(model_assets, Backend::CPU);
  executor_settings->SetCacheDir(cache_path.string());
  executor_settings->SetMaxNumTokens(kMaxNumTokens);
  ::litert::lm::CpuConfig config;
  config.number_of_threads = kNumThreads;
  executor_settings->SetBackendConfig(config);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));
  auto executor = LlmLiteRtCompiledModelExecutorStatic::Create(
      *executor_settings, env, *model_resources);
  ASSERT_OK(executor);
  ASSERT_NE(*executor, nullptr);
}

TEST(LlmLiteRtCompiledModelExecutorStaticTest,
     CreateExecutorTest_WithFileDescriptorCache) {
  auto cache_path =
      std::filesystem::path(::testing::TempDir()) /
      absl::StrCat(
          ::testing::UnitTest::GetInstance()->current_test_info()->name(),
          ".cache");
  std::error_code ec;
  std::filesystem::remove_all(cache_path, ec);
  ASSERT_FALSE(ec);
  {
    // Create an empty file - ScopedFile expects the file to exist.
    std::ofstream cache_file(cache_path.string());
  }
  absl::Cleanup remove_cache = [cache_path] {
    std::error_code ec;
    std::filesystem::remove_all(cache_path, ec);
  };
  ASSERT_OK_AND_ASSIGN(auto scoped_cache_file,
                       ScopedFile::OpenWritable(cache_path.string()));
  auto shared_scoped_cache_file =
      std::make_shared<ScopedFile>(std::move(scoped_cache_file));

  auto model_path =
      std::filesystem::path(::testing::SrcDir()) / kTestStaticModelPath;
  ASSERT_OK_AND_ASSIGN(
      auto model_resources,
      CreateExecutorModelResourcesLitertLm(model_path.string()));
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create(model_path.string()));
  auto executor_settings =
      LlmExecutorSettings::CreateDefault(model_assets, Backend::CPU);
  executor_settings->SetScopedCacheFile(shared_scoped_cache_file);
  executor_settings->SetMaxNumTokens(kMaxNumTokens);
  ::litert::lm::CpuConfig config;
  config.number_of_threads = kNumThreads;
  executor_settings->SetBackendConfig(config);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));
  auto executor = LlmLiteRtCompiledModelExecutorStatic::Create(
      *executor_settings, env, *model_resources);
  ASSERT_OK(executor);
  ASSERT_NE(*executor, nullptr);
}

// Decode batch size and vocab size of magic_test_decode_batch.tflite.
constexpr int kMaxDecodeBatchSize = 11;
constexpr int kVocabSize = 16000;

class TfLiteModelResources : public ModelResources {
 public:
  static absl::StatusOr<std::unique_ptr<TfLiteModelResources>> Create(
      const ModelAssets& model_assets, bool with_mtp_drafter = false) {
    LITERT_ASSIGN_OR_RETURN(auto path, model_assets.GetPath());
    LITERT_ASSIGN_OR_RETURN(auto model,
                            Model::CreateFromFile(std::string(path)));
    return absl::WrapUnique(
        new TfLiteModelResources(std::move(model), with_mtp_drafter));
  }

 private:
  explicit TfLiteModelResources(Model model, bool with_mtp_drafter = false)
      : model_(std::move(model)), with_mtp_drafter_(with_mtp_drafter) {}

 public:
  // ModelResources implementation:
  absl::StatusOr<const Model*> GetTFLiteModel(ModelType model_type) override {
    if (model_type == ModelType::kTfLitePrefillDecode) {
      return &model_;
    }
    if (model_type == ModelType::kTfLiteMtpDrafter) {
      if (with_mtp_drafter_) {
        // Reuse the same model for testing MTP drafter creation
        return &model_;
      } else {
        return absl::NotFoundError("MTP Drafter model not found");
      }
    }
    return absl::UnimplementedError("Unsupported model type");
  }

  absl::StatusOr<absl::string_view> GetTFLiteModelBuffer(
      ModelType model_type) override {
    return absl::UnimplementedError("GetTFLiteModelBuffer not implemented.");
  }

  absl::StatusOr<std::unique_ptr<Tokenizer>> GetTokenizer() override {
    return absl::UnimplementedError("GetTokenizer not implemented.");
  }

  absl::StatusOr<const proto::LlmMetadata*> GetLlmMetadata() override {
    return absl::UnimplementedError("GetLlmMetadata not implemented.");
  }

  absl::StatusOr<const proto::ExecutorMetadata*> GetExecutorMetadata()
      override {
    return absl::UnimplementedError("GetExecutorMetadata not implemented.");
  }

  std::optional<std::string> GetTFLiteModelBackendConstraint(
      ModelType model_type) override {
    return std::nullopt;
  }

  std::optional<std::string> GetTFLiteModelPreferActivationType(
      ModelType model_type) override {
    return std::nullopt;
  }

  absl::StatusOr<std::reference_wrapper<ScopedFile>> GetScopedFile() override {
    return absl::UnimplementedError("GetScopedFile not implemented.");
  }

  absl::StatusOr<std::pair<size_t, size_t>> GetWeightsSectionOffset(
      ModelType model_type) override {
    return absl::UnimplementedError("GetWeightsSectionOffset not implemented.");
  }

  absl::StatusOr<FileRegion> GetTFLiteModelSectionFileRegion(
      ModelType model_type) override {
    return absl::UnimplementedError(
        "GetTFLiteModelSectionFileRegion not implemented.");
  }

 private:
  Model model_;
  bool with_mtp_drafter_;
};

TEST(LlmLiteRtCompiledModelExecutorStaticTest,
     CreateExecutorTest_WithMtpDrafter) {
  const std::filesystem::path model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/magic_test_decode_batch.tflite";
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create(model_path.string()));
  ASSERT_OK_AND_ASSIGN(auto executor_settings,
                       LlmExecutorSettings::CreateDefault(model_assets));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));
  ASSERT_OK_AND_ASSIGN(auto model_resources,
                       TfLiteModelResources::Create(model_assets,
                                                    /*with_mtp_drafter=*/true));
  ASSERT_OK_AND_ASSIGN(
      auto executor, LlmLiteRtCompiledModelExecutorStatic::Create(
                         std::move(executor_settings), env, *model_resources));
  EXPECT_TRUE(executor);
}

TEST(LlmLiteRtCompiledModelExecutorStaticTest, MultipleOutput_Decode) {
  const std::filesystem::path model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/magic_test_decode_batch.tflite";
  auto model_assets = ModelAssets::Create(model_path.string());
  ASSERT_OK(model_assets);
  auto executor_settings = LlmExecutorSettings::CreateDefault(*model_assets);
  ASSERT_OK(executor_settings);
  auto env = Environment::Create(std::vector<Environment::Option>());
  LITERT_ASSERT_OK(env);
  ASSERT_OK_AND_ASSIGN(auto model_resources,
                       TfLiteModelResources::Create(*model_assets));

  ASSERT_OK_AND_ASSIGN(
      auto executor,
      LlmLiteRtCompiledModelExecutorStatic::Create(
          std::move(*executor_settings), *env, *model_resources));
  auto step = executor->GetCurrentStep();
  EXPECT_OK(step);
  EXPECT_EQ(*step, 0);

  // Prefill 5 tokens.
  ExecutorInputs inputs;
  const std::vector<int> input_tokens = {1, 2, 3, 4, 5};
  auto input_tokens_buffer =
      CopyToTensorBuffer<int>(absl::MakeSpan(input_tokens), {1, 5});
  EXPECT_TRUE(input_tokens_buffer);
  inputs.SetTextData(ExecutorTextData(std::move(*input_tokens_buffer)));
  EXPECT_OK(executor->Prefill(inputs));
  step = executor->GetCurrentStep();
  EXPECT_OK(step);
  EXPECT_EQ(*step, 5);
  auto step_and_token =
      executor->processed_tokens_for_testing().GetNextUnprocessedToken();
  EXPECT_EQ(step_and_token.step, 4);
  EXPECT_EQ(step_and_token.token.size(), 1);
  EXPECT_EQ(step_and_token.token[0]->id(), 5);

  // Decode 20 tokens.
  constexpr int kDecodeSteps = 20;
  for (int i = 0; i < kDecodeSteps; ++i) {
    ASSERT_OK_AND_ASSIGN(auto output_tokens, executor->Decode());
    EXPECT_EQ(output_tokens.size(), kMaxDecodeBatchSize);
    // All tokens should be the same since sampling is strict.
    for (int j = 1; j < kMaxDecodeBatchSize; ++j) {
      EXPECT_EQ(output_tokens[0][0], output_tokens[j][0]);
    }
  }
  step = executor->GetCurrentStep();
  EXPECT_OK(step);
  EXPECT_EQ(*step, 5 + kDecodeSteps);
  step_and_token =
      executor->processed_tokens_for_testing().GetNextUnprocessedToken();
  EXPECT_EQ(step_and_token.step, 4 + kDecodeSteps);
  EXPECT_EQ(step_and_token.token.size(), kMaxDecodeBatchSize);
  // All tokens should be the same since sampling is strict.
  for (int i = 1; i < kMaxDecodeBatchSize; ++i) {
    EXPECT_EQ(step_and_token.token[0]->id(), step_and_token.token[i]->id());
  }

  // Prefill once again, and see the token candidate is reduced one.
  ExecutorInputs inputs_next;
  input_tokens_buffer =
      CopyToTensorBuffer<int>(absl::MakeSpan(input_tokens.data(), 1), {1, 1});
  EXPECT_TRUE(input_tokens_buffer);
  inputs_next.SetTextData(ExecutorTextData(std::move(*input_tokens_buffer)));
  EXPECT_OK(executor->Prefill(inputs_next));
  step = executor->GetCurrentStep();
  EXPECT_OK(step);
  EXPECT_EQ(*step, 5 + kDecodeSteps + 1);
  step_and_token =
      executor->processed_tokens_for_testing().GetNextUnprocessedToken();
  EXPECT_EQ(step_and_token.step, 4 + kDecodeSteps + 1);
  EXPECT_EQ(step_and_token.token.size(), 1);
}

TEST(LlmLiteRtCompiledModelExecutorStaticTest,
     MultipleOutput_DecodeLogits_EmptyInputs) {
  const std::filesystem::path model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/magic_test_decode_batch.tflite";
  auto model_assets = ModelAssets::Create(model_path.string());
  ASSERT_OK(model_assets);
  auto executor_settings = LlmExecutorSettings::CreateDefault(*model_assets);
  ASSERT_OK(executor_settings);
  auto env = Environment::Create(std::vector<Environment::Option>());
  LITERT_ASSERT_OK(env);
  ASSERT_OK_AND_ASSIGN(auto model_resources,
                       TfLiteModelResources::Create(*model_assets));
  auto executor = LlmLiteRtCompiledModelExecutorStatic::Create(
      std::move(*executor_settings), *env, *model_resources);
  EXPECT_OK(executor);
  EXPECT_TRUE(*executor);
  auto step = (*executor)->GetCurrentStep();
  EXPECT_OK(step);
  EXPECT_EQ(*step, 0);

  // Prefill 5 tokens.
  ExecutorInputs inputs;
  const std::vector<int> input_tokens = {1, 2, 3, 4, 5};
  auto input_tokens_buffer =
      CopyToTensorBuffer<int>(absl::MakeSpan(input_tokens), {1, 5});
  inputs.SetTextData(ExecutorTextData(std::move(*input_tokens_buffer)));
  EXPECT_OK((*executor)->Prefill(inputs));
  step = (*executor)->GetCurrentStep();
  EXPECT_OK(step);
  EXPECT_EQ(*step, 5);
  auto step_and_token =
      (*executor)->processed_tokens_for_testing().GetNextUnprocessedToken();
  EXPECT_EQ(step_and_token.step, 4);
  EXPECT_EQ(step_and_token.token.size(), 1);
  EXPECT_EQ(step_and_token.token[0]->id(), 5);

  // Decode 20 tokens.
  constexpr int kDecodeSteps = 20;
  ExecutorInputs decode_inputs;
  for (int i = 0; i < kDecodeSteps; ++i) {
    auto logits = (*executor)->DecodeLogits(decode_inputs);
    EXPECT_OK(logits);
    auto logits_type = logits->TensorType();
    EXPECT_TRUE(logits_type);
    EXPECT_EQ(logits_type->ElementType(), ElementType::Float32);
    EXPECT_EQ(logits_type->Layout().Dimensions(),
              Dimensions({kMaxDecodeBatchSize, 1, kVocabSize}));
    auto logits_span = ReferTensorBufferAsSpan<float>(*logits);
    EXPECT_TRUE(logits_span);
    EXPECT_EQ(logits_span->size(), kMaxDecodeBatchSize * kVocabSize);
    // Check the first logit of the first candidate vs the first logit of the
    // last candidate. They are the same only for first decode step.
    if (i == 0) {
      EXPECT_EQ(logits_span->at(0),
                logits_span->at((kMaxDecodeBatchSize - 1) * kVocabSize));
    } else if (logits_span->at(0) != -std::numeric_limits<float>::infinity() &&
               logits_span->at(0) != std::numeric_limits<float>::infinity()) {
      EXPECT_NE(logits_span->at(0),
                logits_span->at((kMaxDecodeBatchSize - 1) * kVocabSize));
    }

    // Prepare for next decode.
    std::vector<int> decode_input_tokens{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    input_tokens_buffer = CopyToTensorBuffer<int>(
        absl::MakeSpan(decode_input_tokens), {kMaxDecodeBatchSize, 1});
    EXPECT_TRUE(input_tokens_buffer);
    decode_inputs.SetTextData(
        ExecutorTextData(std::move(*input_tokens_buffer)));
  }
  step = (*executor)->GetCurrentStep();
  EXPECT_OK(step);
  // First pending tokens were processed.
  EXPECT_EQ(*step, 5 + kDecodeSteps);
  step_and_token =
      (*executor)->processed_tokens_for_testing().GetNextUnprocessedToken();
  EXPECT_EQ(step_and_token.step, 4 + kDecodeSteps);
  // No pending input token left with DecodeLogits.
  EXPECT_TRUE(step_and_token.token.empty());

  // Prefill once again, and see the token candidate is reduced one.
  ExecutorInputs inputs_next;
  input_tokens_buffer =
      CopyToTensorBuffer<int>(absl::MakeSpan(input_tokens.data(), 1), {1, 1});
  inputs_next.SetTextData(ExecutorTextData(std::move(*input_tokens_buffer)));
  EXPECT_OK((*executor)->Prefill(inputs_next));
  step = (*executor)->GetCurrentStep();
  EXPECT_OK(step);
  EXPECT_EQ(*step, 5 + kDecodeSteps + 1);
  step_and_token =
      (*executor)->processed_tokens_for_testing().GetNextUnprocessedToken();
  EXPECT_EQ(step_and_token.step, 4 + kDecodeSteps);
  // The prefilled token is added as pending input token.
  EXPECT_EQ(step_and_token.token.size(), 1);
}

TEST(LlmLiteRtCompiledModelExecutorStaticTest,
     MultipleOutput_DecodeLogits_ValidInputs) {
  const std::filesystem::path model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/magic_test_decode_batch.tflite";
  auto model_assets = ModelAssets::Create(model_path.string());
  ASSERT_OK(model_assets);
  auto executor_settings = LlmExecutorSettings::CreateDefault(*model_assets);
  ASSERT_OK(executor_settings);
  auto env = Environment::Create(std::vector<Environment::Option>());
  LITERT_ASSERT_OK(env);
  ASSERT_OK_AND_ASSIGN(auto model_resources,
                       TfLiteModelResources::Create(*model_assets));
  auto executor = LlmLiteRtCompiledModelExecutorStatic::Create(
      std::move(*executor_settings), *env, *model_resources);
  EXPECT_OK(executor);
  EXPECT_TRUE(*executor);
  auto step = (*executor)->GetCurrentStep();
  EXPECT_OK(step);
  EXPECT_EQ(*step, 0);

  // Prefill 5 tokens.
  ExecutorInputs inputs;
  const std::vector<int> input_tokens = {1, 2, 3, 4, 5};
  auto input_tokens_buffer =
      CopyToTensorBuffer<int>(absl::MakeSpan(input_tokens), {1, 5});
  inputs.SetTextData(ExecutorTextData(std::move(*input_tokens_buffer)));
  EXPECT_OK((*executor)->Prefill(inputs));
  step = (*executor)->GetCurrentStep();
  EXPECT_OK(step);
  EXPECT_EQ(*step, 5);
  auto step_and_token =
      (*executor)->processed_tokens_for_testing().GetNextUnprocessedToken();
  EXPECT_EQ(step_and_token.step, 4);
  EXPECT_EQ(step_and_token.token.size(), 1);
  EXPECT_EQ(step_and_token.token[0]->id(), 5);

  // Decode 20 tokens.
  constexpr int kDecodeSteps = 20;
  ExecutorInputs decode_inputs;
  const std::vector<int> decode_input_tokens{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  input_tokens_buffer = CopyToTensorBuffer<int>(
      absl::MakeSpan(decode_input_tokens), {kMaxDecodeBatchSize, 1});
  EXPECT_TRUE(input_tokens_buffer);
  decode_inputs.SetTextData(ExecutorTextData(std::move(*input_tokens_buffer)));
  for (int i = 0; i < kDecodeSteps; ++i) {
    auto logits = (*executor)->DecodeLogits(decode_inputs);
    EXPECT_OK(logits);
    auto logits_type = logits->TensorType();
    EXPECT_TRUE(logits_type);
    EXPECT_EQ(logits_type->ElementType(), ElementType::Float32);
    EXPECT_EQ(logits_type->Layout().Dimensions(),
              Dimensions({kMaxDecodeBatchSize, 1, kVocabSize}));
    auto logits_span = ReferTensorBufferAsSpan<float>(*logits);
    EXPECT_TRUE(logits_span);
    EXPECT_EQ(logits_span->size(), kMaxDecodeBatchSize * kVocabSize);
    // Check the first logit of the first candidate vs the first logit of the
    // last candidate. They are different for all decode steps.
    if (logits_span->at(0) != -std::numeric_limits<float>::infinity() &&
        logits_span->at(0) != std::numeric_limits<float>::infinity()) {
      EXPECT_NE(logits_span->at(0),
                logits_span->at((kMaxDecodeBatchSize - 1) * kVocabSize));
    }
  }
  step = (*executor)->GetCurrentStep();
  EXPECT_OK(step);
  // First pending tokens were ignored.
  EXPECT_EQ(*step, 5 + kDecodeSteps);
  step_and_token =
      (*executor)->processed_tokens_for_testing().GetNextUnprocessedToken();
  EXPECT_EQ(step_and_token.step, 4 + kDecodeSteps);
  // No pending input token left with DecodeLogits.
  EXPECT_TRUE(step_and_token.token.empty());

  // Prefill once again, and see the token candidate is reduced one.
  ExecutorInputs inputs_next;
  input_tokens_buffer =
      CopyToTensorBuffer<int>(absl::MakeSpan(input_tokens.data(), 1), {1, 1});
  inputs_next.SetTextData(ExecutorTextData(std::move(*input_tokens_buffer)));
  EXPECT_OK((*executor)->Prefill(inputs_next));
  step = (*executor)->GetCurrentStep();
  EXPECT_OK(step);
  EXPECT_EQ(*step, 5 + kDecodeSteps + 1);
  step_and_token =
      (*executor)->processed_tokens_for_testing().GetNextUnprocessedToken();
  EXPECT_EQ(step_and_token.step, 4 + kDecodeSteps);
  // The prefilled token is added as pending input token.
  EXPECT_EQ(step_and_token.token.size(), 1);
}

absl::StatusOr<
    std::pair<std::unique_ptr<ModelResources>,
              std::unique_ptr<LlmLiteRtCompiledModelExecutorDynamic>>>
CreateDynamicExecutor(Environment& env, absl::string_view model_path,
                      uint32_t kv_increment_size = 8,
                      int prefill_chunk_size = -1) {
  auto path = std::filesystem::path(::testing::SrcDir()) / model_path;
  ABSL_ASSIGN_OR_RETURN(auto model_resources,
                        CreateExecutorModelResourcesLitertLm(path.string()));
  ABSL_ASSIGN_OR_RETURN(auto model_assets, ModelAssets::Create(path.string()));
  auto executor_settings =
      LlmExecutorSettings::CreateDefault(model_assets, Backend::CPU);
  executor_settings->SetCacheDir(":nocache");
  executor_settings->SetMaxNumTokens(kMaxNumTokens);
  CpuConfig config;
  config.number_of_threads = kNumThreads;
  config.kv_increment_size = kv_increment_size;
  config.prefill_chunk_size = prefill_chunk_size;
  executor_settings->SetBackendConfig(config);
  ABSL_ASSIGN_OR_RETURN(auto executor,
                        LlmLiteRtCompiledModelExecutorDynamic::Create(
                            *executor_settings, env, *model_resources));
  return std::make_pair(std::move(model_resources), std::move(executor));
}

TEST(LlmLiteRtCompiledModelExecutorDynamicTest, PrefillTest) {
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));
  std::unique_ptr<ModelResources> model_resources;
  std::unique_ptr<LlmLiteRtCompiledModelExecutorDynamic> executor;
  {
    ASSERT_OK_AND_ASSIGN(auto p,
                         CreateDynamicExecutor(env, kTestDynamicModelPath));
    std::tie(model_resources, executor) = std::move(p);
  }

  ExecutorInputs inputs;
  // Create a tensor buffer with 3 elements but only the first two elements
  // match the expected prefill tokens.
  const std::vector<int> input_tokens = {1, 2, 0};
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto input_tokens_buffer,
      CopyToTensorBuffer<int>(absl::MakeSpan(input_tokens), {1, 3}));
  inputs.SetTextData(ExecutorTextData(std::move(input_tokens_buffer)));

  for (int i = 0; i < 10; ++i) {
    EXPECT_OK(executor->Prefill(inputs));
    ASSERT_OK_AND_ASSIGN(auto current_step, executor->GetCurrentStep());
    EXPECT_EQ(current_step, 3 * (i + 1));
  }
}

TEST(LlmLiteRtCompiledModelExecutorDynamicTest, ResetTest) {
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));
  std::unique_ptr<ModelResources> model_resources;
  std::unique_ptr<LlmLiteRtCompiledModelExecutorDynamic> executor;
  {
    ASSERT_OK_AND_ASSIGN(auto p,
                         CreateDynamicExecutor(env, kTestDynamicModelPath));
    std::tie(model_resources, executor) = std::move(p);
  }

  ExecutorInputs inputs;
  const std::vector<int> input_tokens = {1, 2, 0};
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto input_tokens_buffer,
      CopyToTensorBuffer<int>(absl::MakeSpan(input_tokens), {1, 3}));
  inputs.SetTextData(ExecutorTextData(std::move(input_tokens_buffer)));

  EXPECT_OK(executor->Prefill(inputs));

  {
    ASSERT_OK_AND_ASSIGN(auto current_step, executor->GetCurrentStep());
    EXPECT_EQ(current_step, 3);
  }

  EXPECT_OK(executor->Reset());

  {
    ASSERT_OK_AND_ASSIGN(auto current_step, executor->GetCurrentStep());
    EXPECT_EQ(current_step, 0);
  }

  const std::vector<int> second_input_tokens = {3, 4, 5};
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto second_input_tokens_buffer,
      CopyToTensorBuffer<int>(absl::MakeSpan(second_input_tokens), {1, 3}));
  ExecutorInputs second_inputs;
  second_inputs.SetTextData(
      ExecutorTextData(std::move(second_input_tokens_buffer)));

  EXPECT_OK(executor->Prefill(second_inputs));

  {
    ASSERT_OK_AND_ASSIGN(auto current_step, executor->GetCurrentStep());
    EXPECT_EQ(current_step, 3);
  }
}

TEST(LlmLiteRtCompiledModelExecutorDynamicTest, DecodeTest) {
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));
  std::unique_ptr<ModelResources> model_resources;
  std::unique_ptr<LlmLiteRtCompiledModelExecutorDynamic> executor;
  {
    ASSERT_OK_AND_ASSIGN(auto p,
                         CreateDynamicExecutor(env, kTestDynamicModelPath));
    std::tie(model_resources, executor) = std::move(p);
  }

  ExecutorInputs inputs;
  // Create a tensor buffer with 3 elements but only the first two elements
  // match the expected prefill tokens.
  const std::vector<int> input_tokens = {1, 2, 0};
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto input_tokens_buffer,
      CopyToTensorBuffer<int>(absl::MakeSpan(input_tokens), {1, 3}));
  inputs.SetTextData(ExecutorTextData(std::move(input_tokens_buffer)));

  EXPECT_OK(executor->Prefill(inputs));
  {
    ASSERT_OK_AND_ASSIGN(auto current_step, executor->GetCurrentStep());
    EXPECT_EQ(current_step, input_tokens.size());
  }

  for (int i = 0; i < 16; ++i) {
    ASSERT_OK_AND_ASSIGN(auto output_tokens, executor->Decode());
    ASSERT_OK_AND_ASSIGN(auto current_step, executor->GetCurrentStep());
    EXPECT_EQ(current_step, input_tokens.size() + (i + 1));
  }
}

}  // namespace
}  // namespace litert::lm
