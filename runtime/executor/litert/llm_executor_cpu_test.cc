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

#include <cstdint>
#include <filesystem>  // NOLINT: Required for path manipulation.
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/test/matchers.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/components/model_resources_litert_lm.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/litert/llm_executor.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/litert_lm_loader.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

constexpr char kTestLmStaticModelPath[] =
    "litert_lm/runtime/testdata/test_lm_static.litertlm";

const int kNumThreads = 4;
const int kMaxNumTokens = 32;

absl::StatusOr<std::unique_ptr<ModelResources>>
CreateExecutorModelResourcesLitertLm(absl::string_view model_path) {
  ASSIGN_OR_RETURN(auto scoped_file, ScopedFile::Open(model_path));
  return ModelResourcesLitertLm::Create(
      std::make_unique<LitertLmLoader>(std::move(scoped_file)));
}

/* ===========================================================================*/
/* LlmExecutorExternalSamplerStatic */
/* ===========================================================================*/

absl::StatusOr<std::pair<std::unique_ptr<ModelResources>,
                         std::unique_ptr<LlmExecutorExternalSamplerStatic>>>
CreateExternalSamplerStaticExecutor(Environment& env,
                                    const std::string& model_path) {
  auto path = std::filesystem::path(::testing::SrcDir()) / model_path;
  ASSIGN_OR_RETURN(auto model_resources,
                   CreateExecutorModelResourcesLitertLm(path.string()));
  ASSIGN_OR_RETURN(auto model_assets, ModelAssets::Create(path.string()));
  ASSIGN_OR_RETURN(auto executor_settings, LlmExecutorSettings::CreateDefault(
                                               model_assets, Backend::CPU));
  executor_settings.SetCacheDir(":nocache");
  executor_settings.SetMaxNumTokens(kMaxNumTokens);
  CpuConfig config;
  config.number_of_threads = kNumThreads;
  executor_settings.SetBackendConfig(config);
  ASSIGN_OR_RETURN(auto executor,
                   LlmExecutorExternalSamplerStatic::Create(
                       executor_settings, env, *model_resources));
  return std::make_pair(std::move(model_resources), std::move(executor));
}

TEST(LlmExecutorExternalSamplerStaticTest, CorrectnessTest) {
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));
  ASSERT_OK_AND_ASSIGN(
      (auto [model_resources, executor]),
      CreateExternalSamplerStaticExecutor(env, kTestLmStaticModelPath));
  ASSERT_OK_AND_ASSIGN(auto kv_cache, executor->CreateKVCache());
  int next_position = 0;
  {
    ExecutorInputs inputs;
    const std::vector<int> input_tokens = {2, 128, 256, 512};
    const int num_input_tokens = input_tokens.size();
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto input_tokens_buffer,
        CopyToTensorBuffer<int>(absl::MakeSpan(input_tokens),
                                {1, num_input_tokens}));
    inputs.SetTextData(ExecutorTextData(std::move(input_tokens_buffer)));
    inputs.SetNextPosition(next_position);
    ASSERT_OK(executor->Prefill(std::move(inputs), *kv_cache,
                                /*lora_id=*/std::nullopt));
    next_position += num_input_tokens;
  }
  const std::vector<int> tokens = {1024, 2048};
  // The first three logits for each token
  const std::vector<std::vector<float>> expected_logits = {
      {-0.110427, -0.169406, -0.31386}, {0.233391, 0.0977644, 0.0840598}};
  for (int i = 0; i < tokens.size(); ++i) {
    ExecutorInputs inputs;
    std::vector<int> input_tokens = {tokens[i]};
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto input_tokens_buffer,
        CopyToTensorBuffer<int>(absl::MakeSpan(input_tokens), {1, 1}));
    inputs.SetTextData(ExecutorTextData(std::move(input_tokens_buffer)));
    inputs.SetNextPosition(next_position);
    ASSERT_OK_AND_ASSIGN(
        auto logits,
        executor->Step(std::move(inputs), *kv_cache, /*lora_id=*/std::nullopt));
    LITERT_ASSERT_OK_AND_ASSIGN(auto logits_span,
                                ReferTensorBufferAsSpan<float>(logits));
    EXPECT_NEAR(logits_span[0], expected_logits[i][0], 1e-2);
    EXPECT_NEAR(logits_span[1], expected_logits[i][1], 1e-2);
    EXPECT_NEAR(logits_span[2], expected_logits[i][2], 1e-2);
    next_position += 1;
  }
}

TEST(LlmExecutorExternalSamplerStaticTest, IncrementalPrefillCorrectnessTest) {
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));
  ASSERT_OK_AND_ASSIGN(
      (auto [model_resources, executor]),
      CreateExternalSamplerStaticExecutor(env, kTestLmStaticModelPath));
  ASSERT_OK_AND_ASSIGN(auto kv_cache, executor->CreateKVCache());
  int next_position = 0;
  const std::vector<std::vector<int>> prefill_inputs = {{2, 128}, {256, 512}};
  for (const auto& input_tokens : prefill_inputs) {
    ExecutorInputs inputs;
    const int num_input_tokens = input_tokens.size();
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto input_tokens_buffer,
        CopyToTensorBuffer<int>(absl::MakeSpan(input_tokens),
                                {1, num_input_tokens}));
    inputs.SetTextData(ExecutorTextData(std::move(input_tokens_buffer)));
    inputs.SetNextPosition(next_position);
    ASSERT_OK(executor->Prefill(std::move(inputs), *kv_cache,
                                /*lora_id=*/std::nullopt));
    next_position += num_input_tokens;
  }

  const std::vector<int> tokens = {1024, 2048};
  // The first three logits for each token
  const std::vector<std::vector<float>> expected_logits = {
      {-0.110427, -0.169406, -0.31386}, {0.233391, 0.0977644, 0.0840598}};
  for (int i = 0; i < tokens.size(); ++i) {
    ExecutorInputs inputs;
    std::vector<int> input_tokens = {tokens[i]};
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto input_tokens_buffer,
        CopyToTensorBuffer<int>(absl::MakeSpan(input_tokens), {1, 1}));
    inputs.SetTextData(ExecutorTextData(std::move(input_tokens_buffer)));
    inputs.SetNextPosition(next_position);
    ASSERT_OK_AND_ASSIGN(
        auto logits,
        executor->Step(std::move(inputs), *kv_cache, /*lora_id=*/std::nullopt));
    LITERT_ASSERT_OK_AND_ASSIGN(auto logits_span,
                                ReferTensorBufferAsSpan<float>(logits));
    EXPECT_NEAR(logits_span[0], expected_logits[i][0], 1e-2);
    EXPECT_NEAR(logits_span[1], expected_logits[i][1], 1e-2);
    EXPECT_NEAR(logits_span[2], expected_logits[i][2], 1e-2);
    next_position += 1;
  }
}

/* ===========================================================================*/
/* LlmExecutorExternalSamplerDynamic */
/* ===========================================================================*/

absl::StatusOr<std::pair<std::unique_ptr<ModelResources>,
                         std::unique_ptr<LlmExecutorExternalSamplerDynamic>>>
CreateExternalSamplerDynamicExecutor(Environment& env,
                                     const std::string& model_path,
                                     uint32_t kv_increment_size = 8,
                                     int prefill_chunk_size = -1) {
  auto path = std::filesystem::path(::testing::SrcDir()) / model_path;
  ASSIGN_OR_RETURN(auto model_resources,
                   CreateExecutorModelResourcesLitertLm(path.string()));
  ASSIGN_OR_RETURN(auto model_assets, ModelAssets::Create(path.string()));
  ASSIGN_OR_RETURN(auto executor_settings, LlmExecutorSettings::CreateDefault(
                                               model_assets, Backend::CPU));
  executor_settings.SetCacheDir(":nocache");
  executor_settings.SetMaxNumTokens(kMaxNumTokens);
  CpuConfig config;
  config.number_of_threads = kNumThreads;
  config.kv_increment_size = kv_increment_size;
  config.prefill_chunk_size = prefill_chunk_size;
  executor_settings.SetBackendConfig(config);
  ASSIGN_OR_RETURN(auto executor,
                   LlmExecutorExternalSamplerDynamic::Create(
                       executor_settings, env, *model_resources));
  return std::make_pair(std::move(model_resources), std::move(executor));
}

/* ===========================================================================*/
/* LlmExecutorInternalSamplerStatic */
/* ===========================================================================*/

absl::StatusOr<std::pair<std::unique_ptr<ModelResources>,
                         std::unique_ptr<LlmExecutorInternalSamplerStatic>>>
CreateInternalSamplerStaticExecutor(Environment& env,
                                    const std::string& model_path) {
  auto path = std::filesystem::path(::testing::SrcDir()) / model_path;
  ASSIGN_OR_RETURN(auto model_resources,
                   CreateExecutorModelResourcesLitertLm(path.string()));
  ASSIGN_OR_RETURN(auto model_assets, ModelAssets::Create(path.string()));
  ASSIGN_OR_RETURN(auto executor_settings, LlmExecutorSettings::CreateDefault(
                                               model_assets, Backend::CPU));
  executor_settings.SetCacheDir(":nocache");
  executor_settings.SetMaxNumTokens(kMaxNumTokens);
  CpuConfig config;
  config.number_of_threads = kNumThreads;
  executor_settings.SetBackendConfig(config);
  ASSIGN_OR_RETURN(auto executor,
                   LlmExecutorInternalSamplerStatic::Create(
                       executor_settings, env, *model_resources));
  return std::make_pair(std::move(model_resources), std::move(executor));
}

TEST(LlmExecutorInternalSamplerStaticTest, CorrectnessTest) {
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));
  ASSERT_OK_AND_ASSIGN(
      (auto [model_resources, executor]),
      CreateInternalSamplerStaticExecutor(env, kTestLmStaticModelPath));
  ASSERT_OK_AND_ASSIGN(auto kv_cache, executor->CreateKVCache());
  int next_position = 0;
  {
    ExecutorInputs inputs;
    const std::vector<int> input_tokens = {2, 128, 256, 512};
    const int num_input_tokens = input_tokens.size();
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto input_tokens_buffer,
        CopyToTensorBuffer<int>(absl::MakeSpan(input_tokens),
                                {1, num_input_tokens}));
    inputs.SetTextData(ExecutorTextData(std::move(input_tokens_buffer)));
    inputs.SetNextPosition(next_position);
    ASSERT_OK(executor->Prefill(std::move(inputs), *kv_cache,
                                /*lora_id=*/std::nullopt));
    next_position += num_input_tokens;
  }
  const std::vector<int> tokens = {1024, 2048};
  const std::vector<float> expected_output_tokens = {7857, 5148};
  for (int i = 0; i < tokens.size(); ++i) {
    ExecutorInputs inputs;
    std::vector<int> input_tokens = {tokens[i]};
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto input_tokens_buffer,
        CopyToTensorBuffer<int>(absl::MakeSpan(input_tokens), {1, 1}));
    inputs.SetTextData(ExecutorTextData(std::move(input_tokens_buffer)));
    inputs.SetNextPosition(next_position);
    ASSERT_OK_AND_ASSIGN(
        std::vector<std::vector<int>> tokens,
        executor->SampleTokens(/*num_steps=*/1, std::move(inputs), *kv_cache,
                               /*lora_id=*/std::nullopt));

    ASSERT_EQ(tokens.size(), 1);
    ASSERT_EQ(tokens[0].size(), 1);
    EXPECT_EQ(tokens[0][0], expected_output_tokens[i]);
    next_position += 1;
  }
}

TEST(LlmExecutorInternalSamplerStaticTest, MultiStepSampleTest) {
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));
  ASSERT_OK_AND_ASSIGN(
      (auto [model_resources, executor]),
      CreateInternalSamplerStaticExecutor(env, kTestLmStaticModelPath));
  ASSERT_OK_AND_ASSIGN(auto kv_cache, executor->CreateKVCache());
  int next_position = 0;
  {
    ExecutorInputs inputs;
    const std::vector<int> input_tokens = {2, 128, 256, 512};
    const int num_input_tokens = input_tokens.size();
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto input_tokens_buffer,
        CopyToTensorBuffer<int>(absl::MakeSpan(input_tokens),
                                {1, num_input_tokens}));
    inputs.SetTextData(ExecutorTextData(std::move(input_tokens_buffer)));
    inputs.SetNextPosition(next_position);
    ASSERT_OK(executor->Prefill(std::move(inputs), *kv_cache,
                                /*lora_id=*/std::nullopt));
    next_position += num_input_tokens;
  }
  const std::vector<float> expected_output_tokens = {7857, 10470, 322};
  ExecutorInputs inputs;
  std::vector<int> input_tokens = {1024};
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto input_tokens_buffer,
      CopyToTensorBuffer<int>(absl::MakeSpan(input_tokens), {1, 1}));
  inputs.SetTextData(ExecutorTextData(std::move(input_tokens_buffer)));
  inputs.SetNextPosition(next_position);
  ASSERT_OK_AND_ASSIGN(
      std::vector<std::vector<int>> tokens,
      executor->SampleTokens(/*num_steps=*/3, std::move(inputs), *kv_cache,
                             /*lora_id=*/std::nullopt));

  ASSERT_EQ(tokens.size(), 1);
  ASSERT_EQ(tokens[0].size(), expected_output_tokens.size());
  for (int i = 0; i < tokens[0].size(); ++i) {
    EXPECT_EQ(tokens[0][i], expected_output_tokens[i]);
  }
}

}  // namespace
}  // namespace litert::lm
