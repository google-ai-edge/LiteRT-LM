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

// Runs LlmLiteRtCompiledModelExecutorStatic on CPU with the checksum fake LM
// test models (testdata/checksum_fake_lm*.tflite).
// The fake LM predicts the checksum (c[0] + ... + c[p] + p) % 32 as the token
// after the context c[0], ..., c[p], so every decoded token checks the
// positions, attention mask and KV cache contents that the executor used to
// produce it.

#include <filesystem>  // NOLINT(build/c++17)
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/test/matchers.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/components/sampler_factory.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/executor/llm_litert_compiled_model_executor.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

constexpr char kTestdataDir[] =
    "litert_lm/runtime/testdata";

// Vocabulary and KV cache sizes of the fake LM.
constexpr int kFakeLmVocabSize = 32;
constexpr int kFakeLmContextSize = 64;

std::string FakeLmModelPath(absl::string_view file_name) {
  return (std::filesystem::path(::testing::SrcDir()) / kTestdataDir /
          std::string(file_name))
      .string();
}

// Returns the token that the fake LM predicts after `context`.
int FakeLmNextToken(absl::Span<const int> context) {
  const int sum = std::accumulate(context.begin(), context.end(), 0);
  return (sum + static_cast<int>(context.size()) - 1) % kFakeLmVocabSize;
}

// Serves a fake LM as the prefill/decode model of a bundle, and optionally the
// fake embedder as its text embedder.
class FakeLmModelResources : public ModelResources {
 public:
  static absl::StatusOr<std::unique_ptr<FakeLmModelResources>> Create(
      absl::string_view model_file,
      std::optional<absl::string_view> embedder_file) {
    LITERT_ASSIGN_OR_RETURN(auto model,
                            Model::CreateFromFile(FakeLmModelPath(model_file)));
    std::optional<Model> embedder;
    if (embedder_file.has_value()) {
      LITERT_ASSIGN_OR_RETURN(
          embedder, Model::CreateFromFile(FakeLmModelPath(*embedder_file)));
    }
    return absl::WrapUnique(
        new FakeLmModelResources(std::move(model), std::move(embedder)));
  }

  using ModelResources::HasTFLiteModel;
  bool HasTFLiteModel(proto::LlmMetadata::TfLiteModelType model_type) override {
    return model_type == proto::LlmMetadata::TF_LITE_PREFILL_DECODE ||
           (model_type == proto::LlmMetadata::TF_LITE_EMBEDDER &&
            embedder_.has_value());
  }

  using ModelResources::GetTFLiteModel;
  absl::StatusOr<const Model*> GetTFLiteModel(
      proto::LlmMetadata::TfLiteModelType model_type) override {
    if (model_type == proto::LlmMetadata::TF_LITE_PREFILL_DECODE) {
      return &model_;
    }
    if (model_type == proto::LlmMetadata::TF_LITE_EMBEDDER &&
        embedder_.has_value()) {
      return &*embedder_;
    }
    return absl::NotFoundError("The fake LM bundle has no such model.");
  }

  absl::StatusOr<std::unique_ptr<Tokenizer>> GetTokenizer() override {
    return absl::UnimplementedError("The fake LM has no tokenizer.");
  }

  absl::StatusOr<const proto::LlmMetadata*> GetLlmMetadata() override {
    return absl::UnimplementedError("The fake LM has no LLM metadata.");
  }

 private:
  FakeLmModelResources(Model model, std::optional<Model> embedder)
      : model_(std::move(model)), embedder_(std::move(embedder)) {}

  Model model_;
  std::optional<Model> embedder_;
};

struct FakeLmTestCase {
  std::string name;
  std::string model_file;
  std::optional<std::string> embedder_file;
};

class FakeLmExecutorTest : public ::testing::TestWithParam<FakeLmTestCase> {
 protected:
  void SetUp() override {
    const FakeLmTestCase& test_case = GetParam();
    ASSERT_OK_AND_ASSIGN(resources_,
                         FakeLmModelResources::Create(test_case.model_file,
                                                      test_case.embedder_file));
    ASSERT_OK_AND_ASSIGN(
        auto model_assets,
        ModelAssets::Create(FakeLmModelPath(test_case.model_file)));
    ASSERT_OK_AND_ASSIGN(auto settings, LlmExecutorSettings::CreateDefault(
                                            model_assets, Backend::CPU));
    settings.SetCacheDir(":nocache");
    settings.SetMaxNumTokens(kFakeLmContextSize);
    CpuConfig cpu_config;
    cpu_config.number_of_threads = 1;
    settings.SetBackendConfig(cpu_config);
    LITERT_ASSERT_OK_AND_ASSIGN(auto env, Environment::Create({}));
    env_ = std::make_unique<Environment>(std::move(env));
    ASSERT_OK_AND_ASSIGN(
        executor_, LlmLiteRtCompiledModelExecutorStatic::Create(settings, *env_,
                                                                *resources_));

    RuntimeConfig runtime_config;
    runtime_config.sampler_params = GreedySamplerParams();
    ASSERT_OK_AND_ASSIGN(auto context, executor_->CreateNewContext(
                                           std::nullopt, runtime_config));
    ASSERT_OK(executor_->RestoreContext(std::move(context)));
  }

  // Prefills `tokens` and appends them to `context_`.
  absl::Status Prefill(absl::Span<const int> tokens) {
    LITERT_ASSIGN_OR_RETURN(
        auto tokens_buffer,
        CopyToTensorBuffer<int>(tokens, {1, static_cast<int>(tokens.size())}));
    ExecutorInputs inputs;
    inputs.SetTextData(ExecutorTextData(std::move(tokens_buffer)));
    ABSL_RETURN_IF_ERROR(executor_->Prefill(inputs));
    context_.insert(context_.end(), tokens.begin(), tokens.end());
    return absl::OkStatus();
  }

  // Decodes `num_steps` tokens, expecting each to be the fake LM's prediction
  // and the executor step to count the context.
  void DecodeAndExpectFakeLmTokens(int num_steps) {
    for (int step = 0; step < num_steps; ++step) {
      SCOPED_TRACE(::testing::Message() << "Context size: " << context_.size());
      ASSERT_OK_AND_ASSIGN(auto current_step, executor_->GetCurrentStep());
      ASSERT_EQ(current_step, context_.size());
      ASSERT_OK_AND_ASSIGN(auto output_tokens, executor_->Decode());
      ASSERT_EQ(output_tokens.size(), 1);
      ASSERT_EQ(output_tokens[0].size(), 1);
      EXPECT_EQ(output_tokens[0][0], FakeLmNextToken(context_));
      context_.push_back(output_tokens[0][0]);
    }
    ASSERT_OK_AND_ASSIGN(auto current_step, executor_->GetCurrentStep());
    EXPECT_EQ(current_step, context_.size());
  }

  // Rewinds the executor to `step` tokens and drops the rewound tokens from
  // `context_`.
  absl::Status Rewind(int step) {
    ABSL_RETURN_IF_ERROR(executor_->SetCurrentStep(step));
    context_.resize(step);
    return absl::OkStatus();
  }

  std::unique_ptr<Environment> env_;
  std::unique_ptr<FakeLmModelResources> resources_;
  std::unique_ptr<LlmLiteRtCompiledModelExecutorStatic> executor_;
  // The tokens that the executor has processed or holds pending.
  std::vector<int> context_;
};

TEST_P(FakeLmExecutorTest, DecodeAfterPrefill) {
  ASSERT_OK(Prefill({3, 1, 4}));
  ASSERT_NO_FATAL_FAILURE(DecodeAndExpectFakeLmTokens(/*num_steps=*/6));
}

TEST_P(FakeLmExecutorTest, DecodeAfterPrefillOfSeveralChunks) {
  std::vector<int> tokens(21);
  for (int i = 0; i < tokens.size(); ++i) {
    tokens[i] = (7 * i + 2) % kFakeLmVocabSize;
  }
  ASSERT_OK(Prefill(tokens));
  ASSERT_NO_FATAL_FAILURE(DecodeAndExpectFakeLmTokens(/*num_steps=*/4));
}

TEST_P(FakeLmExecutorTest, DecodeAfterPrefillThatFollowsPrefill) {
  ASSERT_OK(Prefill({3, 1, 4}));
  ASSERT_OK(Prefill({1, 5, 9, 2, 6}));
  ASSERT_NO_FATAL_FAILURE(DecodeAndExpectFakeLmTokens(/*num_steps=*/3));
}

TEST_P(FakeLmExecutorTest, DecodeAfterPrefillThatFollowsDecode) {
  ASSERT_OK(Prefill({3, 1, 4}));
  ASSERT_NO_FATAL_FAILURE(DecodeAndExpectFakeLmTokens(/*num_steps=*/2));
  ASSERT_OK(Prefill({1, 5, 9, 2, 6}));
  ASSERT_NO_FATAL_FAILURE(DecodeAndExpectFakeLmTokens(/*num_steps=*/3));
}

TEST_P(FakeLmExecutorTest, DecodeAfterRewind) {
  ASSERT_OK(Prefill({3, 1, 4, 1, 5, 9}));
  ASSERT_NO_FATAL_FAILURE(DecodeAndExpectFakeLmTokens(/*num_steps=*/4));
  // Rewinding into the decoded tokens leaves stale KV cache entries behind the
  // new step, which the attention mask must hide.
  ASSERT_OK(Rewind(8));
  ASSERT_NO_FATAL_FAILURE(DecodeAndExpectFakeLmTokens(/*num_steps=*/3));
  // Rewinding into the prefilled tokens.
  ASSERT_OK(Rewind(3));
  ASSERT_NO_FATAL_FAILURE(DecodeAndExpectFakeLmTokens(/*num_steps=*/2));
}

TEST_P(FakeLmExecutorTest, DecodeAfterPrefillThatFollowsRewind) {
  ASSERT_OK(Prefill({3, 1, 4, 1, 5, 9}));
  ASSERT_NO_FATAL_FAILURE(DecodeAndExpectFakeLmTokens(/*num_steps=*/2));
  // The rewind makes the first decoded token pending again.
  ASSERT_OK(Rewind(7));
  ASSERT_OK(Prefill({2, 6}));
  ASSERT_NO_FATAL_FAILURE(DecodeAndExpectFakeLmTokens(/*num_steps=*/2));
  // The rewind makes a prefilled token pending again, and the prefill of a
  // single token only processes that pending token.
  ASSERT_OK(Rewind(3));
  ASSERT_OK(Prefill({5}));
  ASSERT_NO_FATAL_FAILURE(DecodeAndExpectFakeLmTokens(/*num_steps=*/2));
}

TEST_P(FakeLmExecutorTest, DecodeUntilTheContextFillsTheKvCache) {
  ASSERT_OK(Prefill({3, 1, 4, 1, 5, 9, 2, 6, 5, 3}));
  // Ends with kFakeLmContextSize tokens in the context, the last one pending:
  // decoding it would leave a pending token without a KV cache slot.
  ASSERT_NO_FATAL_FAILURE(
      DecodeAndExpectFakeLmTokens(/*num_steps=*/kFakeLmContextSize - 10));
}

INSTANTIATE_TEST_SUITE_P(
    FakeLms, FakeLmExecutorTest,
    ::testing::Values(
        FakeLmTestCase{.name = "TokenInput",
                       .model_file = "checksum_fake_lm.tflite"},
        FakeLmTestCase{.name = "TokenInputWithoutVerify",
                       .model_file = "checksum_fake_lm_no_verify.tflite"},
        FakeLmTestCase{.name = "EmbeddingInput",
                       .model_file = "checksum_fake_lm_embedding_input.tflite",
                       .embedder_file = "checksum_fake_lm_embedder.tflite"}),
    [](const ::testing::TestParamInfo<FakeLmTestCase>& info) {
      return info.param.name;
    });

}  // namespace
}  // namespace litert::lm
