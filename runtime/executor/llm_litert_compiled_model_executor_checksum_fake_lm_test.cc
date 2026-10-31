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

#include <array>
#include <atomic>
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
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/test/matchers.h"  // from @litert
#include "runtime/components/constrained_decoding/constrained_decoder.h"
#include "runtime/components/constrained_decoding/constraint.h"
#include "runtime/components/constrained_decoding/fake_constraint.h"
#include "runtime/components/model_resources.h"
#include "runtime/components/sampler_factory.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/executor_stats.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/executor/llm_litert_compiled_model_executor.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

using ::testing::AnyOf;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::Eq;
using ::testing::FieldsAre;
using ::testing::HasSubstr;
using ::testing::status::IsOkAndHolds;
using ::testing::status::StatusIs;

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

// Tokens that the constraints of the tests force. After the prompt {3, 1, 4}
// and the forced tokens before it, the fake LM predicts none of them.
constexpr std::array<int, 20> kForcedTokens = {
    5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};

std::vector<int> ForcedTokens() {
  return std::vector<int>(kForcedTokens.begin(), kForcedTokens.end());
}

// Like FakeConstraint, which forces a sequence of tokens, but fails the
// transitions by the tokens that it disallows, as LLGuidance does.
class StrictFakeConstraint : public FakeConstraint {
 public:
  StrictFakeConstraint(std::vector<int> token_ids, int vocabulary_size)
      : FakeConstraint(token_ids, vocabulary_size),
        token_ids_(std::move(token_ids)) {}

  absl::StatusOr<std::unique_ptr<State>> ComputeNext(const State& state,
                                                     int token) const override {
    const int index = static_cast<const FakeState&>(state).index();
    if (index < token_ids_.size() && token != token_ids_[index]) {
      return absl::InvalidArgumentError(
          absl::StrCat("The constraint disallows token ", token, "."));
    }
    return FakeConstraint::ComputeNext(state, token);
  }

 private:
  std::vector<int> token_ids_;
};

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
  // Whether the model has a verify signature that draft verification can use.
  bool supports_draft_verification = true;
  // The width of the widest verify signature that verifies drafts.
  int widest_verify_width = 4;
};

class FakeLmExecutorTest : public ::testing::TestWithParam<FakeLmTestCase> {
 protected:
  void SetUp() override {
    ASSERT_NO_FATAL_FAILURE(CreateExecutor(/*selected_signatures=*/{}));
  }

  // (Re)creates `executor_` for the model of the test case, restricted to
  // `selected_signatures` if not empty, and clears `context_`.
  void CreateExecutor(std::vector<std::string> selected_signatures) {
    const FakeLmTestCase& test_case = GetParam();
    executor_.reset();
    context_.clear();
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
    settings.SetSelectedSignatures(std::move(selected_signatures));
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

  // Returns the `num_tokens` tokens that decode steps would produce next.
  std::vector<int> FakeLmContinuation(int num_tokens) const {
    std::vector<int> context = context_;
    for (int i = 0; i < num_tokens; ++i) {
      context.push_back(FakeLmNextToken(context));
    }
    return std::vector<int>(context.end() - num_tokens, context.end());
  }

  // Verifies `draft` and appends the new tokens to `context_`.
  absl::StatusOr<DraftVerificationResult> VerifyDraft(
      absl::Span<const int> draft,
      const ExecutorDecodeParams& params = ExecutorDecodeParams()) {
    ABSL_ASSIGN_OR_RETURN(DraftVerificationResult result,
                          executor_->VerifyDraft(draft, params));
    context_.insert(context_.end(), result.tokens.begin(), result.tokens.end());
    ABSL_ASSIGN_OR_RETURN(int current_step, executor_->GetCurrentStep());
    if (current_step != context_.size()) {
      return absl::InternalError(absl::StrCat("Step ", current_step,
                                              " after verification, expected ",
                                              context_.size(), "."));
    }
    return result;
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

TEST_P(FakeLmExecutorTest, ChecksDraftVerificationSupport) {
  if (GetParam().supports_draft_verification) {
    EXPECT_OK(executor_->CheckDraftVerificationSupport());
  } else {
    EXPECT_THAT(
        executor_->CheckDraftVerificationSupport(),
        StatusIs(absl::StatusCode::kFailedPrecondition, HasSubstr("has none")));
  }
}

TEST_P(FakeLmExecutorTest, DraftVerificationNeedsASelectedVerifySignature) {
  ASSERT_NO_FATAL_FAILURE(CreateExecutor({"decode", "prefill_8"}));
  // The executor works, but can't verify drafts.
  ASSERT_OK(Prefill({3, 1, 4}));
  DecodeAndExpectFakeLmTokens(/*num_steps=*/1);
  EXPECT_THAT(executor_->CheckDraftVerificationSupport(),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(
      executor_->VerifyDraft(FakeLmContinuation(2), ExecutorDecodeParams()),
      StatusIs(absl::StatusCode::kFailedPrecondition));
}

// Draft verification on a model whose verify signatures can verify drafts.
class FakeLmDraftVerificationTest : public FakeLmExecutorTest {
 protected:
  void SetUp() override {
    if (!GetParam().supports_draft_verification) {
      GTEST_SKIP() << "The model can't verify drafts.";
    }
    FakeLmExecutorTest::SetUp();
  }
};

TEST_P(FakeLmDraftVerificationTest, AcceptsTheTokensThatDecodingProduces) {
  ASSERT_OK(Prefill({3, 1, 4}));
  for (int draft_size : {1, 2, 3, 4, 5, 7, 8, 9, 12, 20}) {
    SCOPED_TRACE(::testing::Message() << "Draft size: " << draft_size);
    const std::vector<int> continuation = FakeLmContinuation(draft_size + 1);
    const std::vector<int> draft(continuation.begin(), continuation.end() - 1);
    ASSERT_OK_AND_ASSIGN(DraftVerificationResult result, VerifyDraft(draft));
    EXPECT_EQ(result.num_accepted, draft_size);
    // The token after the draft, unless the draft ends at a pass boundary.
    ASSERT_THAT(result.tokens.size(),
                AnyOf(Eq(draft_size), Eq(draft_size + 1)));
    EXPECT_THAT(result.tokens,
                ElementsAreArray(continuation.begin(),
                                 continuation.begin() + result.tokens.size()));
    // Decoding continues as if every token had been decoded.
    DecodeAndExpectFakeLmTokens(/*num_steps=*/2);
    ASSERT_OK(Rewind(3));
  }
}

TEST_P(FakeLmDraftVerificationTest, AcceptsTheLastDraftTokenAtAPassBoundary) {
  // The widest pass takes the pending token and W - 1 draft tokens, and its
  // last token accepts draft token W; the draft token stays pending.
  const int width = GetParam().widest_verify_width;
  ASSERT_OK(Prefill({3, 1, 4}));
  const std::vector<int> draft = FakeLmContinuation(width);
  EXPECT_THAT(VerifyDraft(draft),
              IsOkAndHolds(FieldsAre(ElementsAreArray(draft), width)));
  DecodeAndExpectFakeLmTokens(/*num_steps=*/2);
}

TEST_P(FakeLmDraftVerificationTest, ReturnsTheCorrectionOfTheFirstMismatch) {
  ASSERT_OK(Prefill({3, 1, 4}));
  for (int mismatch : {0, 1, 3, 4, 7, 8, 11}) {
    SCOPED_TRACE(::testing::Message() << "Mismatch: " << mismatch);
    const std::vector<int> continuation = FakeLmContinuation(12);
    std::vector<int> draft = continuation;
    draft[mismatch] = (draft[mismatch] + 1) % kFakeLmVocabSize;
    EXPECT_THAT(VerifyDraft(draft),
                IsOkAndHolds(FieldsAre(
                    ElementsAreArray(continuation.begin(),
                                     continuation.begin() + mismatch + 1),
                    mismatch)));
    // The KV cache entries of the rejected draft tokens don't matter.
    DecodeAndExpectFakeLmTokens(/*num_steps=*/3);
    ASSERT_OK(Rewind(3));
  }
}

TEST_P(FakeLmDraftVerificationTest, VerifiesAfterDecode) {
  ASSERT_OK(Prefill({3, 1, 4}));
  DecodeAndExpectFakeLmTokens(/*num_steps=*/2);
  const std::vector<int> draft = FakeLmContinuation(5);
  ASSERT_OK_AND_ASSIGN(DraftVerificationResult result, VerifyDraft(draft));
  EXPECT_EQ(result.num_accepted, 5);
  DecodeAndExpectFakeLmTokens(/*num_steps=*/2);
}

TEST_P(FakeLmDraftVerificationTest, FailsIfNoPassFits) {
  std::vector<int> tokens(62);
  for (int i = 0; i < tokens.size(); ++i) {
    tokens[i] = (5 * i + 1) % kFakeLmVocabSize;
  }
  ASSERT_OK(Prefill(tokens));
  // The draft and the token after it fit into the context, but a pass of
  // width 4 at position 61 would end past the 64 KV cache entries.
  EXPECT_THAT(
      executor_->VerifyDraft(FakeLmContinuation(1), ExecutorDecodeParams()),
      StatusIs(absl::StatusCode::kResourceExhausted,
               HasSubstr("No verify pass fits")));
  DecodeAndExpectFakeLmTokens(/*num_steps=*/2);
}

TEST_P(FakeLmDraftVerificationTest, StopsAtThePassThatDoesntFit) {
  std::vector<int> tokens(58);
  for (int i = 0; i < tokens.size(); ++i) {
    tokens[i] = (5 * i + 1) % kFakeLmVocabSize;
  }
  ASSERT_OK(Prefill(tokens));
  // A pass of width 4 at position 57 accepts 4 draft tokens. The next one, at
  // position 61, would end past the 64 KV cache entries.
  const std::vector<int> draft = FakeLmContinuation(5);
  EXPECT_THAT(VerifyDraft(draft),
              IsOkAndHolds(FieldsAre(
                  ElementsAreArray(draft.begin(), draft.begin() + 4), 4)));
  std::optional<ExecutorStats> stats = executor_->GetExecutorStats();
  ASSERT_TRUE(stats.has_value());
  EXPECT_EQ(stats->draft_verify_passes(), 1);
  EXPECT_EQ(stats->draft_verify_draft_tokens(), 4);
  // Decoding fills the context.
  DecodeAndExpectFakeLmTokens(/*num_steps=*/2);
  EXPECT_EQ(context_.size(), kFakeLmContextSize);
}

TEST_P(FakeLmDraftVerificationTest,
       FailsEarlyIfTheDraftDoesntFitIntoTheContext) {
  // The context ends at the end of the KV cache, or before it.
  for (int max_num_tokens : {kFakeLmContextSize, kFakeLmContextSize / 2}) {
    SCOPED_TRACE(::testing::Message() << "Max num tokens: " << max_num_tokens);
    ASSERT_NO_FATAL_FAILURE(CreateExecutor(/*selected_signatures=*/{}));
    ASSERT_OK_AND_ASSIGN(LlmExecutorSettings settings,
                         executor_->GetExecutorSettings());
    settings.SetMaxNumTokens(max_num_tokens);
    ASSERT_OK(executor_->UpdateExecutorSettings(settings));
    std::vector<int> tokens(max_num_tokens - 3);
    for (int i = 0; i < tokens.size(); ++i) {
      tokens[i] = (5 * i + 1) % kFakeLmVocabSize;
    }
    ASSERT_OK(Prefill(tokens));
    // A pass of width 4 fits, but the token after the draft would be pending
    // past the context.
    EXPECT_THAT(
        executor_->VerifyDraft(FakeLmContinuation(3), ExecutorDecodeParams()),
        StatusIs(absl::StatusCode::kResourceExhausted,
                 HasSubstr("no room for the token after it")));
    // A shorter draft fills the context.
    ASSERT_OK(VerifyDraft(FakeLmContinuation(2)));
    EXPECT_EQ(context_.size(), max_num_tokens);
  }
}

TEST_P(FakeLmDraftVerificationTest, RejectsInvalidDrafts) {
  ASSERT_OK(Prefill({3, 1, 4}));
  EXPECT_THAT(executor_->VerifyDraft({}, ExecutorDecodeParams()),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(
      executor_->VerifyDraft({1, kFakeLmVocabSize}, ExecutorDecodeParams()),
      StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(executor_->VerifyDraft({-1}, ExecutorDecodeParams()),
              StatusIs(absl::StatusCode::kInvalidArgument));
  DecodeAndExpectFakeLmTokens(/*num_steps=*/2);
}

TEST_P(FakeLmDraftVerificationTest, StopsWhenCancelled) {
  ASSERT_OK(Prefill({3, 1, 4}));
  std::atomic<bool> cancelled = true;
  ExecutorDecodeParams params;
  params.SetCancelled(&cancelled);
  EXPECT_THAT(executor_->VerifyDraft(FakeLmContinuation(5), params),
              StatusIs(absl::StatusCode::kCancelled));
  DecodeAndExpectFakeLmTokens(/*num_steps=*/2);
}

TEST_P(FakeLmDraftVerificationTest,
       CancelledVerificationKeepsThePendingTokenFromThePrefill) {
  // The strict constraint would fail on the prompt's last token if a
  // constrained decode fed it.
  StrictFakeConstraint constraint(ForcedTokens(), kFakeLmVocabSize);
  ASSERT_OK(Prefill({3, 1, 4}));
  ASSERT_OK_AND_ASSIGN(auto decoder,
                       ConstrainedDecoder::CreateForHost(&constraint));
  std::atomic<bool> cancelled = true;
  ExecutorDecodeParams params;
  params.SetConstrainedDecoder(decoder.get());
  params.SetCancelled(&cancelled);
  EXPECT_THAT(executor_->VerifyDraft(ForcedTokens(), params),
              StatusIs(absl::StatusCode::kCancelled));
  cancelled = false;
  ASSERT_OK_AND_ASSIGN(auto output_tokens, executor_->Decode(params));
  EXPECT_THAT(output_tokens, ElementsAre(ElementsAre(kForcedTokens[0])));
}

TEST_P(FakeLmDraftVerificationTest, ReturnsTheCommittedPassesWhenCancelled) {
  ASSERT_OK(Prefill({3, 1, 4}));
  std::atomic<bool> cancelled = false;
  executor_->UpdatePostGraphRunCallback(
      [&cancelled](absl::string_view signature_name, int /*current_step*/,
                   auto& /*tensor_buffers*/) {
        if (absl::StartsWith(signature_name, "verify")) {
          cancelled = true;
        }
      });
  ExecutorDecodeParams params;
  params.SetCancelled(&cancelled);
  // The first pass accepts as many draft tokens as it has rows, the last one
  // pending, and the cancellation skips the other passes.
  const int width = GetParam().widest_verify_width;
  const std::vector<int> draft = FakeLmContinuation(20);
  EXPECT_THAT(
      VerifyDraft(draft, params),
      IsOkAndHolds(FieldsAre(
          ElementsAreArray(draft.begin(), draft.begin() + width), width)));
  executor_->UpdatePostGraphRunCallback(nullptr);
  DecodeAndExpectFakeLmTokens(/*num_steps=*/2);

  // Only completed verifications count.
  std::optional<ExecutorStats> stats = executor_->GetExecutorStats();
  ASSERT_TRUE(stats.has_value());
  EXPECT_EQ(stats->draft_verifications(), 0);
  EXPECT_EQ(stats->plain_steps(), 2);
}

TEST_P(FakeLmDraftVerificationTest, GraphRunCallbacksSeeTheVerifyPasses) {
  // Verify passes run like prefill and decode runs: the graph run callbacks
  // get each run with all of its buffers, including the KV cache.
  std::vector<std::string> pre_runs;
  std::vector<std::string> post_runs;
  executor_->UpdatePreGraphRunCallback(
      [&pre_runs](absl::string_view signature_name, int /*current_step*/,
                  auto& tensor_buffers) {
        pre_runs.push_back(std::string(signature_name));
        EXPECT_TRUE(tensor_buffers.contains("kv_cache_k_0")) << signature_name;
      });
  executor_->UpdatePostGraphRunCallback(
      [&post_runs](absl::string_view signature_name, int /*current_step*/,
                   auto& tensor_buffers) {
        post_runs.push_back(std::string(signature_name));
        EXPECT_TRUE(tensor_buffers.contains("kv_cache_k_0")) << signature_name;
        if (absl::StartsWith(signature_name, "verify")) {
          EXPECT_TRUE(tensor_buffers.contains("logits"));
        }
      });
  ASSERT_OK(Prefill({3, 1, 4}));
  // One pass of the narrowest verify signature.
  ASSERT_OK(VerifyDraft(FakeLmContinuation(2)));
  DecodeAndExpectFakeLmTokens(/*num_steps=*/1);
  executor_->UpdatePreGraphRunCallback(nullptr);
  executor_->UpdatePostGraphRunCallback(nullptr);

  EXPECT_THAT(pre_runs, ElementsAre("prefill_8", "verify", "decode"));
  EXPECT_THAT(post_runs, ElementsAre("prefill_8", "verify", "decode"));
}

TEST_P(FakeLmDraftVerificationTest, InvalidatesTheMtpDrafterState) {
  ASSERT_OK(Prefill({3, 1, 4}));
  executor_->set_mtp_primed_for_testing(true);
  ASSERT_OK(VerifyDraft(FakeLmContinuation(3)));
  EXPECT_FALSE(executor_->mtp_primed_for_testing());
}

TEST_P(FakeLmDraftVerificationTest, AccumulatesTheCompletedVerifications) {
  ASSERT_OK(Prefill({3, 1, 4}));
  // The widest pass, then a pass that rejects the last draft token.
  const int width = GetParam().widest_verify_width;
  std::vector<int> draft = FakeLmContinuation(width + 2);
  draft.back() = (draft.back() + 1) % kFakeLmVocabSize;
  ASSERT_OK(VerifyDraft(draft));
  // Failed verifications don't count.
  EXPECT_THAT(
      executor_->VerifyDraft({kFakeLmVocabSize}, ExecutorDecodeParams()),
      StatusIs(absl::StatusCode::kInvalidArgument));

  std::optional<ExecutorStats> stats = executor_->GetExecutorStats();
  ASSERT_TRUE(stats.has_value());
  EXPECT_EQ(stats->draft_verifications(), 1);
  EXPECT_EQ(stats->draft_verify_passes(), 2);
  EXPECT_EQ(stats->draft_verify_draft_tokens(), width + 2);
  EXPECT_EQ(stats->draft_verify_accepted_tokens(), width + 1);
  EXPECT_EQ(stats->draft_verify_emitted_tokens(), width + 2);
  EXPECT_GT(stats->draft_verify_time(), absl::ZeroDuration());
  EXPECT_EQ(stats->plain_steps(), 0);
}

TEST_P(FakeLmDraftVerificationTest,
       ConstraintCutsTheDraftAtTheFirstDisallowedToken) {
  // A permissive constraint rejects a disallowed draft token through the mask
  // of its row, a strict one also through the failed transition.
  FakeConstraint permissive_constraint(ForcedTokens(), kFakeLmVocabSize);
  StrictFakeConstraint strict_constraint(ForcedTokens(), kFakeLmVocabSize);
  for (Constraint* constraint :
       {static_cast<Constraint*>(&permissive_constraint),
        static_cast<Constraint*>(&strict_constraint)}) {
    // Cuts in the first pass, in a later pass of the narrowest signature, and
    // at the token after the last draft input of a pass.
    for (int cut : {0, 2, 3, 5}) {
      SCOPED_TRACE(::testing::Message()
                   << "Strict: " << (constraint == &strict_constraint)
                   << ", cut: " << cut);
      // A new executor, since a rewind wouldn't reset whether the pending
      // token comes from a decode.
      ASSERT_NO_FATAL_FAILURE(CreateExecutor(/*selected_signatures=*/{}));
      ASSERT_OK(Prefill({3, 1, 4}));
      ASSERT_OK_AND_ASSIGN(auto decoder,
                           ConstrainedDecoder::CreateForHost(constraint));
      ExecutorDecodeParams params;
      params.SetConstrainedDecoder(decoder.get());
      std::vector<int> draft(kForcedTokens.begin(), kForcedTokens.begin() + 8);
      draft[cut] = 0;
      EXPECT_THAT(VerifyDraft(draft, params),
                  IsOkAndHolds(FieldsAre(
                      ElementsAreArray(kForcedTokens.begin(),
                                       kForcedTokens.begin() + cut + 1),
                      cut)));
      // Decoding with the same decoder continues the forced tokens.
      ASSERT_OK_AND_ASSIGN(auto output_tokens, executor_->Decode(params));
      EXPECT_THAT(output_tokens,
                  ElementsAre(ElementsAre(kForcedTokens[cut + 1])));
    }
  }
}

TEST_P(FakeLmDraftVerificationTest, ConstraintAcceptsTheDraftsThatFollowIt) {
  StrictFakeConstraint constraint(ForcedTokens(), kFakeLmVocabSize);
  ASSERT_OK(Prefill({3, 1, 4}));
  ASSERT_OK_AND_ASSIGN(auto decoder,
                       ConstrainedDecoder::CreateForHost(&constraint));
  ExecutorDecodeParams params;
  params.SetConstrainedDecoder(decoder.get());
  ASSERT_OK_AND_ASSIGN(auto output_tokens, executor_->Decode(params));
  ASSERT_THAT(output_tokens, ElementsAre(ElementsAre(kForcedTokens[0])));
  context_.push_back(kForcedTokens[0]);
  int num_forced = 1;

  // After a decode, over several passes, and right after a verification.
  for (int draft_size : {10, 2}) {
    SCOPED_TRACE(::testing::Message() << "Draft size: " << draft_size);
    const std::vector<int> draft(
        kForcedTokens.begin() + num_forced,
        kForcedTokens.begin() + num_forced + draft_size);
    ASSERT_OK_AND_ASSIGN(DraftVerificationResult result,
                         VerifyDraft(draft, params));
    EXPECT_EQ(result.num_accepted, draft_size);
    EXPECT_THAT(result.tokens,
                ElementsAreArray(
                    kForcedTokens.begin() + num_forced,
                    kForcedTokens.begin() + num_forced + result.tokens.size()));
    num_forced += result.tokens.size();
  }

  // Decoding with the same decoder continues the forced tokens.
  ASSERT_OK_AND_ASSIGN(output_tokens, executor_->Decode(params));
  EXPECT_THAT(output_tokens,
              ElementsAre(ElementsAre(kForcedTokens[num_forced])));
}

INSTANTIATE_TEST_SUITE_P(
    FakeLms, FakeLmExecutorTest,
    ::testing::Values(
        FakeLmTestCase{.name = "TokenInput",
                       .model_file = "checksum_fake_lm.tflite"},
        FakeLmTestCase{.name = "TokenInputWithoutVerify",
                       .model_file = "checksum_fake_lm_no_verify.tflite",
                       .supports_draft_verification = false},
        // Invalid verify signatures disturb neither decoding nor the valid
        // verify signature.
        FakeLmTestCase{.name = "TokenInputWithInvalidVerify",
                       .model_file = "checksum_fake_lm_invalid_verify.tflite"},
        FakeLmTestCase{.name = "EmbeddingInput",
                       .model_file = "checksum_fake_lm_embedding_input.tflite",
                       .embedder_file = "checksum_fake_lm_embedder.tflite"}),
    [](const ::testing::TestParamInfo<FakeLmTestCase>& info) {
      return info.param.name;
    });

INSTANTIATE_TEST_SUITE_P(
    FakeLms, FakeLmDraftVerificationTest,
    ::testing::Values(
        FakeLmTestCase{.name = "TokenInput",
                       .model_file = "checksum_fake_lm.tflite",
                       .widest_verify_width = 8},
        FakeLmTestCase{.name = "TokenInputWithInvalidVerify",
                       .model_file = "checksum_fake_lm_invalid_verify.tflite"},
        FakeLmTestCase{.name = "EmbeddingInput",
                       .model_file = "checksum_fake_lm_embedding_input.tflite",
                       .embedder_file = "checksum_fake_lm_embedder.tflite"}),
    [](const ::testing::TestParamInfo<FakeLmTestCase>& info) {
      return info.param.name;
    });

}  // namespace
}  // namespace litert::lm
