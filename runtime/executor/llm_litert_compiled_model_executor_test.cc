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

  // Same, but recording the params the sampler stands for -- what the executor
  // does for a sampler it built itself, and what a context switch compares
  // against.
  static void SetSamplerWithInputHandlingAndParamsForTest(
      LlmLiteRtCompiledModelExecutorStatic* executor,
      std::unique_ptr<Sampler> sampler,
      const proto::SamplerParameters& sampler_params) {
    auto* testable =
        static_cast<TestableLlmLiteRtCompiledModelExecutorStatic*>(executor);
    testable->sampler_ = std::move(sampler);
    testable->sampler_handles_input_ = true;
    // A single-head CPU context with the default input handling, which is what
    // these tests restore.
    testable->sampler_built_from_ = SamplerConstruction{
        .params = sampler_params,
        .output_heads = 1,
        .backend = Backend::CPU,
        .handles_input_requested = true,
    };
  }

  // Same, for a sampler that stands for a given number of output heads.
  static void SetSamplerWithParamsAndOutputHeadsForTest(
      LlmLiteRtCompiledModelExecutorStatic* executor,
      std::unique_ptr<Sampler> sampler,
      const proto::SamplerParameters& sampler_params, int output_heads) {
    auto* testable =
        static_cast<TestableLlmLiteRtCompiledModelExecutorStatic*>(executor);
    testable->sampler_ = std::move(sampler);
    testable->sampler_handles_input_ = true;
    testable->sampler_built_from_ = SamplerConstruction{
        .params = sampler_params,
        .output_heads = output_heads,
        .backend = Backend::CPU,
        .handles_input_requested = true,
    };
  }

  static bool HasSamplerForTest(
      LlmLiteRtCompiledModelExecutorStatic* executor) {
    return static_cast<TestableLlmLiteRtCompiledModelExecutorStatic*>(executor)
               ->sampler_ != nullptr;
  }

  static std::optional<proto::SamplerParameters> SamplerParamsInUseForTest(
      LlmLiteRtCompiledModelExecutorStatic* executor) {
    const auto& built =
        static_cast<TestableLlmLiteRtCompiledModelExecutorStatic*>(executor)
            ->sampler_built_from_;
    if (!built.has_value()) return std::nullopt;
    return built->params;
  }
};

// A handling sampler that reports through counters which outlive it. The
// release under test destroys the sampler, so the object itself cannot be
// asked afterwards what happened to it.
class FakeSamplerReportingTeardown : public FakeSamplerWithInputHandling {
 public:
  FakeSamplerReportingTeardown(int* disconnect_count, bool* destroyed)
      : disconnect_count_(disconnect_count), destroyed_(destroyed) {}

  ~FakeSamplerReportingTeardown() override { *destroyed_ = true; }

  absl::Status SetInferenceFuncAndInputTensors(
      int (*run_inference_func)(void* arg), void* arg,
      const TensorBuffer* ids_tensor,
      const TensorBuffer* prev_input_positions_tensor,
      const TensorBuffer* input_positions_tensor,
      const TensorBuffer* prev_mask_tensor, const TensorBuffer* mask_tensor,
      const TensorBuffer* prev_param_tensor,
      const TensorBuffer* param_tensor) override {
    if (run_inference_func == nullptr) {
      ++*disconnect_count_;
    }
    return FakeSamplerWithInputHandling::SetInferenceFuncAndInputTensors(
        run_inference_func, arg, ids_tensor, prev_input_positions_tensor,
        input_positions_tensor, prev_mask_tensor, mask_tensor,
        prev_param_tensor, param_tensor);
  }

 private:
  int* disconnect_count_;
  bool* destroyed_;
};

// Sampler params a session would carry. Two of them, differing in every field
// the executor builds a sampler from.
proto::SamplerParameters GreedySamplerParams() {
  proto::SamplerParameters params;
  params.set_type(proto::SamplerParameters::TOP_P);
  params.set_k(1);
  params.set_p(0.0f);
  params.set_temperature(1.0f);
  params.set_seed(1);
  return params;
}

proto::SamplerParameters StochasticSamplerParams() {
  proto::SamplerParameters params;
  params.set_type(proto::SamplerParameters::TOP_P);
  params.set_k(40);
  params.set_p(0.95f);
  params.set_temperature(1.5f);
  params.set_seed(7);
  return params;
}

// A second context with different sampler params must not keep sampling with
// the sampler the first context built (#2080): the executor caches one
// sampler, and before this a stochastic session opened after a greedy one on
// the same engine stayed greedy for the engine's whole lifetime.
TEST(LlmLiteRtCompiledModelExecutorStaticTest,
     SamplerFollowsActiveContextParams) {
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

  auto run_session = [&](const proto::SamplerParameters& params) {
    RuntimeConfig runtime_config;
    runtime_config.sampler_params = params;
    ASSERT_OK_AND_ASSIGN(
        auto context, executor->CreateNewContext(std::nullopt, runtime_config));
    ASSERT_OK(executor->RestoreContext(std::move(context)));
    const std::vector<int> input_tokens = {1, 2, 3};
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto input_tokens_buffer,
        CopyToTensorBuffer<int>(absl::MakeSpan(input_tokens), {1, 3}));
    ExecutorInputs inputs;
    inputs.SetTextData(ExecutorTextData(std::move(input_tokens_buffer)));
    ASSERT_OK(executor->Prefill(inputs));
    ASSERT_OK(executor->Decode().status());
  };
  auto params_in_use = [&]() -> proto::SamplerParameters {
    const auto& in_use =
        TestableLlmLiteRtCompiledModelExecutorStatic::SamplerParamsInUseForTest(
            executor.get());
    EXPECT_TRUE(in_use.has_value());
    return in_use.value_or(proto::SamplerParameters());
  };

  // First session: greedy. The sampler is built from this context's params.
  run_session(GreedySamplerParams());
  EXPECT_EQ(params_in_use().k(), 1);
  EXPECT_FLOAT_EQ(params_in_use().temperature(), 1.0f);
  EXPECT_EQ(params_in_use().seed(), 1);

  // Second session on the same engine asks for stochastic sampling. (A rebuilt
  // sampler may well land at the freed one's address, so the params it was
  // built with are the observable, not the pointer.)
  run_session(StochasticSamplerParams());
  EXPECT_EQ(params_in_use().k(), 40);
  EXPECT_FLOAT_EQ(params_in_use().p(), 0.95f);
  EXPECT_FLOAT_EQ(params_in_use().temperature(), 1.5f);
  EXPECT_EQ(params_in_use().seed(), 7);

  // And back: a greedy session after a stochastic one is greedy again.
  run_session(GreedySamplerParams());
  EXPECT_EQ(params_in_use().k(), 1);
  EXPECT_FLOAT_EQ(params_in_use().temperature(), 1.0f);
  EXPECT_EQ(params_in_use().seed(), 1);
}

// The release happens when the context is restored, NOT when logits are
// sampled. A sampler that handles input is bound into the decode graph, and
// DecodeLogits() reads its presence as proof that the step already ran -- so a
// sampler dropped only at sampling time would hand the new session a token the
// previous context's graph produced.
TEST(LlmLiteRtCompiledModelExecutorStaticTest,
     SamplerIsReleasedWhenTheRestoredContextChangesParams) {
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

  // A first session, with a sampler that handles input and stands for greedy
  // params -- as one the executor built for this context would.
  RuntimeConfig greedy_config;
  greedy_config.sampler_params = GreedySamplerParams();
  ASSERT_OK_AND_ASSIGN(auto greedy_context,
                       executor->CreateNewContext(std::nullopt, greedy_config));
  ASSERT_OK(executor->RestoreContext(std::move(greedy_context)));
  int disconnect_count = 0;
  bool destroyed = false;
  auto fake_sampler = std::make_unique<FakeSamplerReportingTeardown>(
      &disconnect_count, &destroyed);
  auto* fake_sampler_ptr = fake_sampler.get();
  TestableLlmLiteRtCompiledModelExecutorStatic::
      SetSamplerWithInputHandlingAndParamsForTest(
          executor.get(), std::move(fake_sampler), GreedySamplerParams());

  ExecutorInputs inputs;
  const std::vector<int> tokens = {1, 2, 3};
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto input_tokens_buffer,
      CopyToTensorBuffer<int>(absl::MakeSpan(tokens), {1, 3}));
  inputs.SetTextData(ExecutorTextData(std::move(input_tokens_buffer)));
  ASSERT_OK(executor->Prefill(inputs));
  ASSERT_OK(executor->Decode().status());
  // The decode bound the sampler into the graph -- the state that makes a late
  // release unsafe.
  ASSERT_TRUE(fake_sampler_ptr->HandlesInput());
  ASSERT_EQ(disconnect_count, 0);

  // Restoring a context that asks for different sampling releases it, and
  // disconnects it from the decode graph on the way out.
  RuntimeConfig stochastic_config;
  stochastic_config.sampler_params = StochasticSamplerParams();
  ASSERT_OK_AND_ASSIGN(
      auto stochastic_context,
      executor->CreateNewContext(std::nullopt, stochastic_config));
  ASSERT_OK(executor->RestoreContext(std::move(stochastic_context)));
  EXPECT_FALSE(
      TestableLlmLiteRtCompiledModelExecutorStatic::HasSamplerForTest(
          executor.get()));
  EXPECT_TRUE(destroyed);
  // Disconnected from the decode graph before it was dropped, not left bound
  // to a model that outlives it.
  EXPECT_EQ(disconnect_count, 1);
  EXPECT_FALSE(
      TestableLlmLiteRtCompiledModelExecutorStatic::SamplerParamsInUseForTest(
          executor.get())
          .has_value());
}

// The mirror of the test above: a context asking for the SAME sampling keeps
// the sampler, so an ordinary second turn neither rebuilds it nor restarts its
// random stream.
TEST(LlmLiteRtCompiledModelExecutorStaticTest,
     SamplerSurvivesARestoreThatKeepsTheParams) {
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

  RuntimeConfig runtime_config;
  runtime_config.sampler_params = GreedySamplerParams();
  ASSERT_OK_AND_ASSIGN(auto context,
                       executor->CreateNewContext(std::nullopt, runtime_config));
  ASSERT_OK(executor->RestoreContext(std::move(context)));
  auto fake_sampler = std::make_unique<FakeSamplerWithInputHandling>();
  auto* fake_sampler_ptr = fake_sampler.get();
  TestableLlmLiteRtCompiledModelExecutorStatic::
      SetSamplerWithInputHandlingAndParamsForTest(
          executor.get(), std::move(fake_sampler), GreedySamplerParams());
  const int reset_count_before = fake_sampler_ptr->reset_count();

  ASSERT_OK_AND_ASSIGN(auto same_params_context,
                       executor->CreateNewContext(std::nullopt, runtime_config));
  ASSERT_OK(executor->RestoreContext(std::move(same_params_context)));
  EXPECT_TRUE(TestableLlmLiteRtCompiledModelExecutorStatic::HasSamplerForTest(
      executor.get()));
  EXPECT_EQ(fake_sampler_ptr->reset_count(), reset_count_before);

  // An in-place config change is the other way the active context's sampling
  // moves, and it releases the sampler too.
  RuntimeConfig changed_config;
  changed_config.sampler_params = StochasticSamplerParams();
  ASSERT_OK(executor->UpdateRuntimeConfig(changed_config));
  EXPECT_FALSE(
      TestableLlmLiteRtCompiledModelExecutorStatic::HasSamplerForTest(
          executor.get()));
}

// `output_heads` is a construction parameter too -- it is the sampler's batch
// size, and the CPU sampler validates the logits and ids it is handed against
// it. So a context that asks for the same sampling but a different number of
// heads must not keep the sampler either, in either direction.
TEST(LlmLiteRtCompiledModelExecutorStaticTest,
     SamplerIsReleasedWhenTheRestoredContextChangesOutputHeads) {
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

  // Restore a context, then install a sampler standing for that context: same
  // sampling parameters, built for `heads` output heads.
  auto install_sampler_for = [&](std::optional<int> heads) {
    RuntimeConfig runtime_config;
    runtime_config.sampler_params = GreedySamplerParams();
    runtime_config.output_heads = heads;
    ASSERT_OK_AND_ASSIGN(
        auto context, executor->CreateNewContext(std::nullopt, runtime_config));
    ASSERT_OK(executor->RestoreContext(std::move(context)));
    TestableLlmLiteRtCompiledModelExecutorStatic::
        SetSamplerWithParamsAndOutputHeadsForTest(
            executor.get(), std::make_unique<FakeSamplerWithInputHandling>(),
            GreedySamplerParams(), heads.value_or(1));
  };
  auto restore_with_heads = [&](std::optional<int> heads) {
    RuntimeConfig runtime_config;
    runtime_config.sampler_params = GreedySamplerParams();
    runtime_config.output_heads = heads;
    ASSERT_OK_AND_ASSIGN(
        auto context, executor->CreateNewContext(std::nullopt, runtime_config));
    ASSERT_OK(executor->RestoreContext(std::move(context)));
  };

  // One head -> four heads.
  install_sampler_for(std::nullopt);
  restore_with_heads(4);
  EXPECT_FALSE(
      TestableLlmLiteRtCompiledModelExecutorStatic::HasSamplerForTest(
          executor.get()));

  // And back, four heads -> one.
  install_sampler_for(4);
  restore_with_heads(1);
  EXPECT_FALSE(
      TestableLlmLiteRtCompiledModelExecutorStatic::HasSamplerForTest(
          executor.get()));

  // The same head count keeps it, so this is not just "release on every
  // restore".
  install_sampler_for(4);
  restore_with_heads(4);
  EXPECT_TRUE(TestableLlmLiteRtCompiledModelExecutorStatic::HasSamplerForTest(
      executor.get()));

  // An unset head count is one head, so it matches an explicit one.
  install_sampler_for(std::nullopt);
  restore_with_heads(1);
  EXPECT_TRUE(TestableLlmLiteRtCompiledModelExecutorStatic::HasSamplerForTest(
      executor.get()));
}

// The sampler backend and the requested input-handling mode come from the
// executor settings rather than the context, and a live executor can have its
// settings replaced. A sampler built under the old settings must not survive a
// change to either -- while an update that touches neither must not rebuild
// it for nothing.
TEST(LlmLiteRtCompiledModelExecutorStaticTest,
     SamplerIsReleasedWhenExecutorSettingsChangeHowItIsBuilt) {
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

  RuntimeConfig runtime_config;
  runtime_config.sampler_params = GreedySamplerParams();
  ASSERT_OK_AND_ASSIGN(auto context,
                       executor->CreateNewContext(std::nullopt, runtime_config));
  ASSERT_OK(executor->RestoreContext(std::move(context)));
  auto install_sampler = [&]() {
    TestableLlmLiteRtCompiledModelExecutorStatic::
        SetSamplerWithInputHandlingAndParamsForTest(
            executor.get(), std::make_unique<FakeSamplerWithInputHandling>(),
            GreedySamplerParams());
  };

  // An update to a setting the sampler is not built from keeps it.
  install_sampler();
  {
    LlmExecutorSettings settings = executor_settings;
    AdvancedSettings advanced_settings;
    advanced_settings.gpu_enable_metal_residency_set = true;
    settings.SetAdvancedSettings(advanced_settings);
    ASSERT_OK(executor->UpdateExecutorSettings(settings));
  }
  EXPECT_TRUE(TestableLlmLiteRtCompiledModelExecutorStatic::HasSamplerForTest(
      executor.get()));

  // A different sampler backend releases it.
  {
    LlmExecutorSettings settings = executor_settings;
    settings.SetSamplerBackend(Backend::GPU);
    ASSERT_OK(executor->UpdateExecutorSettings(settings));
  }
  EXPECT_FALSE(
      TestableLlmLiteRtCompiledModelExecutorStatic::HasSamplerForTest(
          executor.get()));

  // So does turning off the sampler's handling of decode input.
  ASSERT_OK(executor->UpdateExecutorSettings(executor_settings));
  install_sampler();
  {
    LlmExecutorSettings settings = executor_settings;
    settings.SetAdvancedSettings(
        AdvancedSettings{.sampler_handles_input = false});
    ASSERT_OK(executor->UpdateExecutorSettings(settings));
  }
  EXPECT_FALSE(
      TestableLlmLiteRtCompiledModelExecutorStatic::HasSamplerForTest(
          executor.get()));
}

// A sampler installed from outside records no params, so a context switch must
// leave it alone rather than replace it with one of its own.
TEST(LlmLiteRtCompiledModelExecutorStaticTest,
     AnExternallyInstalledSamplerIsNotReleased) {
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

  auto fake_sampler = std::make_unique<FakeSamplerWithInputHandling>();
  TestableLlmLiteRtCompiledModelExecutorStatic::
      SetSamplerWithInputHandlingForTest(executor.get(),
                                         std::move(fake_sampler));

  RuntimeConfig runtime_config;
  runtime_config.sampler_params = StochasticSamplerParams();
  ASSERT_OK_AND_ASSIGN(auto context,
                       executor->CreateNewContext(std::nullopt, runtime_config));
  ASSERT_OK(executor->RestoreContext(std::move(context)));
  EXPECT_TRUE(TestableLlmLiteRtCompiledModelExecutorStatic::HasSamplerForTest(
      executor.get()));
}

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
  void set_with_mtp_drafter(bool with_mtp_drafter) {
    with_mtp_drafter_ = with_mtp_drafter;
  }

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

TEST(LlmLiteRtCompiledModelExecutorStaticTest,
     Decode_ExplicitSpeculativeDecoding) {
  const std::filesystem::path model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/magic_test_none.tflite";
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create(model_path.string()));
  ASSERT_OK_AND_ASSIGN(auto executor_settings,
                       LlmExecutorSettings::CreateDefault(model_assets));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));
  ASSERT_OK_AND_ASSIGN(auto model_resources, TfLiteModelResources::Create(
                                                 model_assets,
                                                 /*with_mtp_drafter=*/false));
  ASSERT_OK_AND_ASSIGN(
      auto executor, LlmLiteRtCompiledModelExecutorStatic::Create(
                         std::move(executor_settings), env, *model_resources));
  ASSERT_TRUE(executor);

  // Prefill 5 tokens.
  ExecutorInputs inputs;
  const std::vector<int> input_tokens = {1, 2, 3, 4, 5};
  auto input_tokens_buffer =
      CopyToTensorBuffer<int>(absl::MakeSpan(input_tokens), {1, 5});
  ASSERT_TRUE(input_tokens_buffer);
  inputs.SetTextData(ExecutorTextData(std::move(*input_tokens_buffer)));
  ASSERT_OK(executor->Prefill(inputs));

  // Decode with default/nullopt enable_speculative_decoding succeeds.
  ASSERT_OK(executor->Decode());

  // Decode with enable_speculative_decoding = false succeeds.
  ExecutorDecodeParams decode_params_disabled;
  decode_params_disabled.SetEnableSpeculativeDecoding(false);
  ASSERT_OK(executor->Decode(decode_params_disabled));

  // Decode with enable_speculative_decoding = true attempts lazy load and fails
  // when model does not meet MTP requirements (e.g. embedding lookup).
  ExecutorDecodeParams decode_params_enabled;
  decode_params_enabled.SetEnableSpeculativeDecoding(true);
  EXPECT_THAT(executor->Decode(decode_params_enabled),
              StatusIs(absl::StatusCode::kInternal));
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
