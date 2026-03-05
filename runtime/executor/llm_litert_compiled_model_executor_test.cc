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

#include <cstdint>
#include <cstdlib>
#include <filesystem>  // NOLINT: Required for path manipulation.
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <system_error>  // NOLINT: Required for std::error_code used with std::filesystem.
#include <tuple>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/test/matchers.h"  // from @litert
#include "runtime/components/constrained_decoding/constrained_decoder.h"
#include "runtime/components/constrained_decoding/fake_constraint.h"
#include "runtime/components/model_resources.h"
#include "runtime/components/model_resources_litert_lm.h"
#include "runtime/components/model_resources_task.h"
#include "runtime/components/tokenizer.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/litert_lm_loader.h"
#include "runtime/util/model_asset_bundle_resources.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/status_macros.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

constexpr char kTestStaticModelPath[] =
    "litert_lm/runtime/testdata/test_lm.task";

// Test model with dynamic sequence and context length dimensions.
constexpr char kTestDynamicModelPath[] =
    "litert_lm/runtime/testdata/test_lm_dynamic.litertlm";

const int kMaxNumTokens = 32;
const int kNumThreads = 4;

absl::StatusOr<std::unique_ptr<ModelResources>>
CreateExecutorModelResourcesTask(absl::string_view model_path) {
  auto scoped_file = ScopedFile::Open(model_path);
  auto resources = ModelAssetBundleResources::Create(
      /*tag=*/"", std::move(*scoped_file));
  auto model_resources = ModelResourcesTask::Create(std::move(*resources));
  return model_resources;
}

absl::StatusOr<std::unique_ptr<ModelResources>>
CreateExecutorModelResourcesLitertLm(absl::string_view model_path) {
  ASSIGN_OR_RETURN(auto scoped_file, ScopedFile::Open(model_path));
  return ModelResourcesLitertLm::Create(
      std::make_unique<LitertLmLoader>(std::move(scoped_file)));
}

TEST(LlmLiteRtCompiledModelExecutorStaticTest,
     CreateExecutorTest_WithoutCache) {
  auto model_path =
      std::filesystem::path(::testing::SrcDir()) / kTestStaticModelPath;
  ASSERT_OK_AND_ASSIGN(auto model_resources,
                       CreateExecutorModelResourcesTask(model_path.string()));
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

TEST(LlmLiteRtCompiledModelExecutorStaticTest, PrefillTest) {
  auto model_path =
      std::filesystem::path(::testing::SrcDir()) / kTestStaticModelPath;
  ASSERT_OK_AND_ASSIGN(auto model_resources,
                       CreateExecutorModelResourcesTask(model_path.string()));
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

TEST(LlmLiteRtCompiledModelExecutorStaticTest, DecodeTest) {
  auto model_path =
      std::filesystem::path(::testing::SrcDir()) / kTestStaticModelPath;
  ASSERT_OK_AND_ASSIGN(auto model_resources,
                       CreateExecutorModelResourcesTask(model_path.string()));
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

  {
    ASSERT_OK_AND_ASSIGN(auto output_tokens, executor->Decode());

    ASSERT_OK_AND_ASSIGN(auto current_step, executor->GetCurrentStep());
    EXPECT_EQ(current_step, 4);

    ASSERT_EQ(output_tokens.size(), 1);
    ASSERT_EQ(output_tokens[0].size(), 1);
    EXPECT_EQ(output_tokens[0][0], 8005);
  }

  {
    ASSERT_OK_AND_ASSIGN(auto output_tokens, executor->Decode());

    ASSERT_OK_AND_ASSIGN(auto current_step, executor->GetCurrentStep());
    EXPECT_EQ(current_step, 5);

    ASSERT_EQ(output_tokens.size(), 1);
    ASSERT_EQ(output_tokens[0].size(), 1);
    EXPECT_EQ(output_tokens[0][0], 52530);
  }
}

TEST(LlmLiteRtCompiledModelExecutorStaticTest, ConstrainedDecodeTest) {
  auto model_path =
      std::filesystem::path(::testing::SrcDir()) / kTestStaticModelPath;
  ASSERT_OK_AND_ASSIGN(auto model_resources,
                       CreateExecutorModelResourcesTask(model_path.string()));
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
  ConstrainedDecoder constraint_decoder =
      ConstrainedDecoder(&constraint, /*batch_size=*/1);
  params.SetConstraintDecoder(&constraint_decoder);

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
  ASSERT_OK_AND_ASSIGN(auto model_resources,
                       CreateExecutorModelResourcesTask(model_path.string()));
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

TEST(LlmLiteRtCompiledModelExecutorStaticTest, CreateExecutorTest_WithCache) {
  auto cache_path = std::filesystem::path(::testing::TempDir()) /
                    absl::StrCat("cache-", std::rand());
  std::filesystem::remove_all(cache_path);
  absl::Cleanup remove_cache = [cache_path] {
    std::filesystem::remove_all(cache_path);
  };

  auto model_path =
      std::filesystem::path(::testing::SrcDir()) / kTestStaticModelPath;
  ASSERT_OK_AND_ASSIGN(auto model_resources,
                       CreateExecutorModelResourcesTask(model_path.string()));
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
  ASSERT_OK_AND_ASSIGN(auto model_resources,
                       CreateExecutorModelResourcesTask(model_path.string()));
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
  explicit TfLiteModelResources(const ModelAssets& model_assets,
                                bool with_mtp_drafter = false)
      : with_mtp_drafter_(with_mtp_drafter) {
    LITERT_ASSIGN_OR_ABORT(auto path, model_assets.GetPath());
    LITERT_ASSIGN_OR_ABORT(model_, Model::CreateFromFile(std::string(path)));
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

  std::optional<std::string> GetTFLiteModelBackendConstraint(
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
  TfLiteModelResources model_resources(model_assets,
                                       /*with_mtp_drafter=*/true);
  ASSERT_OK_AND_ASSIGN(auto executor,
                       LlmLiteRtCompiledModelExecutorStatic::Create(
                           std::move(executor_settings), env, model_resources));
  EXPECT_TRUE(executor);
}

TEST(LlmLiteRtCompiledModelExecutorStaticTest, MultipleOutput_Decode) {
  const std::filesystem::path model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/magic_test_decode_batch.tflite";
  auto model_assets = ModelAssets::Create(model_path.string());
  EXPECT_OK(model_assets);
  auto executor_settings = LlmExecutorSettings::CreateDefault(*model_assets);
  EXPECT_OK(executor_settings);
  auto env = Environment::Create(std::vector<Environment::Option>());
  EXPECT_TRUE(env);
  TfLiteModelResources model_resources(*model_assets);

  ASSERT_OK_AND_ASSIGN(
      auto executor, LlmLiteRtCompiledModelExecutorStatic::Create(
                         std::move(*executor_settings), *env, model_resources));
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
  EXPECT_OK(model_assets);
  auto executor_settings = LlmExecutorSettings::CreateDefault(*model_assets);
  EXPECT_OK(executor_settings);
  auto env = Environment::Create(std::vector<Environment::Option>());
  EXPECT_TRUE(env);
  TfLiteModelResources model_resources(*model_assets);
  auto executor = LlmLiteRtCompiledModelExecutorStatic::Create(
      std::move(*executor_settings), *env, model_resources);
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
  EXPECT_OK(model_assets);
  auto executor_settings = LlmExecutorSettings::CreateDefault(*model_assets);
  EXPECT_OK(executor_settings);
  auto env = Environment::Create(std::vector<Environment::Option>());
  EXPECT_TRUE(env);
  TfLiteModelResources model_resources(*model_assets);
  auto executor = LlmLiteRtCompiledModelExecutorStatic::Create(
      std::move(*executor_settings), *env, model_resources);
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
  ASSIGN_OR_RETURN(auto model_resources,
                   CreateExecutorModelResourcesLitertLm(path.string()));
  ASSIGN_OR_RETURN(auto model_assets, ModelAssets::Create(path.string()));
  auto executor_settings =
      LlmExecutorSettings::CreateDefault(model_assets, Backend::CPU);
  executor_settings->SetCacheDir(":nocache");
  executor_settings->SetMaxNumTokens(kMaxNumTokens);
  CpuConfig config;
  config.number_of_threads = kNumThreads;
  config.kv_increment_size = kv_increment_size;
  config.prefill_chunk_size = prefill_chunk_size;
  executor_settings->SetBackendConfig(config);
  ASSIGN_OR_RETURN(auto executor,
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
