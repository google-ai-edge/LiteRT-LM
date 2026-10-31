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

#include "runtime/executor/llm_litert_verifier.h"

#include <filesystem>  // NOLINT(build/c++17)
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_common.h"  // from @litert
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/test/matchers.h"  // from @litert
#include "runtime/components/constrained_decoding/constrained_decoder.h"
#include "runtime/components/constrained_decoding/constraint.h"
#include "runtime/components/constrained_decoding/fake_constraint.h"
#include "runtime/components/constrained_decoding/logit_mask.h"
#include "runtime/components/embedding_lookup/embedding_lookup_manager.h"
#include "runtime/components/sampler_factory.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/litert/state.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/executor/state_interface.h"
#include "runtime/proto/sampler_params.pb.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::IsOkAndHolds;
using ::testing::AllOf;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::Eq;
using ::testing::Field;
using ::testing::FieldsAre;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::IsNull;
using ::testing::Ne;
using ::testing::Optional;
using ::testing::UnorderedElementsAre;
using ::testing::status::StatusIs;

constexpr char kTestdataDir[] =
    "litert_lm/runtime/testdata";

// A fake LM test model. Both have a `verify` signature of width 4.
struct FakeLm {
  const char* name;
  const char* model_file;
  // Null for the model that takes token ids.
  const char* embedder_file;
};
constexpr FakeLm kTokenInputLm = {"TokenInput", "checksum_fake_lm.tflite",
                                  nullptr};
constexpr FakeLm kEmbeddingInputLm = {"EmbeddingInput",
                                      "checksum_fake_lm_embedding_input.tflite",
                                      "checksum_fake_lm_embedder.tflite"};
// Without verify signatures.
constexpr FakeLm kNoVerifyLm = {"NoVerify", "checksum_fake_lm_no_verify.tflite",
                                nullptr};
// With a valid `verify` signature and verify signatures that each break one
// rule of the contract.
constexpr FakeLm kInvalidVerifyLm = {
    "InvalidVerify", "checksum_fake_lm_invalid_verify.tflite", nullptr};
// The same for embedding inputs.
constexpr FakeLm kEmbeddingInputInvalidVerifyLm = {
    "EmbeddingInputInvalidVerify",
    "checksum_fake_lm_embedding_input_invalid_verify.tflite",
    "checksum_fake_lm_embedder.tflite"};

constexpr int kFakeLmVocabSize = 32;
constexpr int kFakeLmContextSize = 64;
constexpr int kVerifyWidth = 4;

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

// Returns what a pass over `context[position, position + num_tokens)` returns:
// the fake LM's next token after each of these context prefixes.
std::vector<int> FakeLmPassTokens(absl::Span<const int> context, int position,
                                  int num_tokens = kVerifyWidth) {
  std::vector<int> tokens;
  for (int i = 0; i < num_tokens; ++i) {
    tokens.push_back(FakeLmNextToken(context.subspan(0, position + i + 1)));
  }
  return tokens;
}

proto::SamplerParameters SampledParams(int seed) {
  proto::SamplerParameters params;
  params.set_type(proto::SamplerParameters::TOP_P);
  params.set_k(kFakeLmVocabSize);
  params.set_p(1.0f);
  params.set_temperature(1.0f);
  params.set_seed(seed);
  return params;
}

// Returns `start` followed by the `num_tokens` tokens that the fake LM
// predicts after it.
std::vector<int> FakeLmContinuation(std::vector<int> start, int num_tokens) {
  for (int i = 0; i < num_tokens; ++i) {
    start.push_back(FakeLmNextToken(start));
  }
  return start;
}

// Like FakeConstraint, which forces a sequence of tokens, but fails the
// transitions by the tokens that it disallows, as LLGuidance does. Counts its
// mask computations.
class StrictFakeConstraint : public FakeConstraint {
 public:
  explicit StrictFakeConstraint(std::vector<int> token_ids)
      : FakeConstraint(token_ids, kFakeLmVocabSize),
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

  absl::StatusOr<std::unique_ptr<LogitMask>> ComputeMask(
      const State& state) const override {
    ++num_masks_;
    return FakeConstraint::ComputeMask(state);
  }

  int num_masks() const { return num_masks_; }

 private:
  std::vector<int> token_ids_;
  mutable int num_masks_ = 0;
};

// Returns the number of forced tokens that `state` of a FakeConstraint
// follows.
int NumForcedTokens(const Constraint::State& state) {
  return static_cast<const FakeConstraint::FakeState&>(state).index();
}

// Returns whether `mask` allows `token` and no other token.
bool AllowsOnly(const LogitMask* mask, int token) {
  const auto* bitmap = dynamic_cast<const BitmapLogitMask*>(mask);
  if (bitmap == nullptr) {
    return false;
  }
  for (int t = 0; t < kFakeLmVocabSize; ++t) {
    if (bitmap->IsAllowed(t) != (t == token)) {
      return false;
    }
  }
  return true;
}

class LlmLiteRtVerifierTest : public ::testing::Test {
 protected:
  // Loads `lm` with a fresh KV cache.
  void LoadFakeLm(const FakeLm& lm) {
    lm_ = lm;
    LITERT_ASSERT_OK_AND_ASSIGN(auto env, Environment::Create({}));
    env_ = std::make_unique<Environment>(std::move(env));
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto model, CompiledModel::Create(*env_, FakeLmModelPath(lm.model_file),
                                          HwAccelerators::kCpu));
    model_ = std::make_unique<CompiledModel>(std::move(model));
    if (lm.embedder_file != nullptr) {
      LITERT_ASSERT_OK_AND_ASSIGN(
          auto embedder,
          Model::CreateFromFile(FakeLmModelPath(lm.embedder_file)));
      embedder_ = std::make_unique<Model>(std::move(embedder));
      ASSERT_OK_AND_ASSIGN(
          embedding_lookup_,
          EmbeddingLookupManager::Create(*env_, embedder_.get(),
                                         /*fully_supports_multi_modal=*/false));
    }
    ASSERT_OK_AND_ASSIGN(
        state_, LitertState::Create(*env_, *model_, "prefill_8",
                                    /*executor_metadata=*/nullptr,
                                    LitertState::AllocationPolicy::kInplace,
                                    /*batch_size=*/1));
  }

  absl::StatusOr<std::unique_ptr<LlmLiteRtVerifier>> CreateVerifier(
      absl::string_view signature_key, int pad_token_id = -1,
      LlmLiteRtVerifier::PassRunner pass_runner = nullptr) {
    ABSL_ASSIGN_OR_RETURN(auto model_assets,
                          ModelAssets::Create(FakeLmModelPath(lm_.model_file)));
    ABSL_ASSIGN_OR_RETURN(auto settings, LlmExecutorSettings::CreateDefault(
                                             model_assets, Backend::CPU));
    settings.SetPadTokenId(pad_token_id);
    return LlmLiteRtVerifier::Create(
        *env_, *model_, signature_key, settings, embedding_lookup_.get(),
        /*per_layer_embedding_lookup=*/nullptr,
        /*executor_metadata=*/nullptr, std::move(pass_runner));
  }

  // Runs a pass of `context[position, position + num_tokens)`.
  absl::Status RunPass(LlmLiteRtVerifier& verifier,
                       absl::Span<const int> context, int position,
                       int num_tokens) {
    ABSL_ASSIGN_OR_RETURN(
        auto state_buffers,
        state_->GetStateBuffers(*model_, verifier.signature_key()));
    return verifier.Run(position, context.subspan(position, num_tokens),
                        state_buffers);
  }

  FakeLm lm_;
  std::unique_ptr<Environment> env_;
  std::unique_ptr<CompiledModel> model_;
  std::unique_ptr<Model> embedder_;
  std::unique_ptr<EmbeddingLookupManager> embedding_lookup_;
  std::unique_ptr<LitertState> state_;
};

class LlmLiteRtVerifierFakeLmTest
    : public LlmLiteRtVerifierTest,
      public ::testing::WithParamInterface<FakeLm> {
 protected:
  void SetUp() override { ASSERT_NO_FATAL_FAILURE(LoadFakeLm(GetParam())); }
};

TEST_P(LlmLiteRtVerifierFakeLmTest, CreateReadsWidthAndVocabSize) {
  ASSERT_OK_AND_ASSIGN(auto verifier, CreateVerifier("verify"));
  EXPECT_EQ(verifier->signature_key(), "verify");
  EXPECT_EQ(verifier->width(), kVerifyWidth);
  EXPECT_EQ(verifier->vocab_size(), kFakeLmVocabSize);
}

TEST_P(LlmLiteRtVerifierFakeLmTest, CreateFailsForMissingSignature) {
  EXPECT_THAT(CreateVerifier("verify_16"),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST_P(LlmLiteRtVerifierFakeLmTest, CreateFailsForSignatureWithoutLogits) {
  EXPECT_THAT(CreateVerifier("prefill_8"),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_P(LlmLiteRtVerifierFakeLmTest, FullPassesReturnTheNextTokenOfEveryRow) {
  ASSERT_OK_AND_ASSIGN(auto verifier, CreateVerifier("verify"));
  const std::vector<int> context = {3, 1, 4, 1, 5, 9, 2, 6};
  // Each pass writes the KV cache entries of its positions, which the rows of
  // the next pass attend to.
  for (int position : {0, 4}) {
    SCOPED_TRACE(::testing::Message() << "Position: " << position);
    ASSERT_OK(RunPass(*verifier, context, position, kVerifyWidth));
    EXPECT_THAT(
        verifier->MaskAndSample({}),
        IsOkAndHolds(ElementsAreArray(FakeLmPassTokens(context, position))));
  }
}

TEST_P(LlmLiteRtVerifierFakeLmTest, PaddedPassesReturnTheNextTokenOfTheirRows) {
  // A pad token id other than 0 makes stale padding entries change the sums
  // of the token input model if a mask exposed them.
  ASSERT_OK_AND_ASSIGN(auto verifier,
                       CreateVerifier("verify", /*pad_token_id=*/7));
  const std::vector<int> context = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5};
  // Each pass starts at the first position that the previous pass padded.
  struct Pass {
    int position;
    int num_tokens;
  };
  for (const Pass& pass :
       {Pass{0, 2}, Pass{2, 1}, Pass{3, 4}, Pass{7, 3}, Pass{10, 1}}) {
    SCOPED_TRACE(::testing::Message() << "Position: " << pass.position
                                      << ", tokens: " << pass.num_tokens);
    ASSERT_OK(RunPass(*verifier, context, pass.position, pass.num_tokens));
    EXPECT_THAT(verifier->MaskAndSample({}),
                IsOkAndHolds(ElementsAreArray(FakeLmPassTokens(
                    context, pass.position, pass.num_tokens))));
  }
}

TEST_P(LlmLiteRtVerifierFakeLmTest, MaskAndSampleAppliesTheMaskOfEachRow) {
  ASSERT_OK_AND_ASSIGN(auto verifier, CreateVerifier("verify"));
  const std::vector<int> context = {3, 1, 4, 1};
  std::vector<int> expected = FakeLmPassTokens(context, /*position=*/0);
  const int forced_token = (expected[1] + 1) % kFakeLmVocabSize;
  std::vector<std::unique_ptr<LogitMask>> masks(kVerifyWidth);
  masks[1] =
      BitmapLogitMask::CreateSingleAllowedToken(kFakeLmVocabSize, forced_token);
  expected[1] = forced_token;

  ASSERT_OK(RunPass(*verifier, context, /*position=*/0, kVerifyWidth));
  EXPECT_THAT(verifier->MaskAndSample(masks),
              IsOkAndHolds(ElementsAreArray(expected)));
}

TEST_P(LlmLiteRtVerifierFakeLmTest, MaskAndSampleRejectsAPartialMaskSequence) {
  ASSERT_OK_AND_ASSIGN(auto verifier, CreateVerifier("verify"));
  const std::vector<int> context = {3, 1, 4};
  // Masks cover the padding rows too.
  ASSERT_OK(RunPass(*verifier, context, /*position=*/0, /*num_tokens=*/3));
  std::vector<std::unique_ptr<LogitMask>> masks(3);
  EXPECT_THAT(verifier->MaskAndSample(masks),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_P(LlmLiteRtVerifierFakeLmTest, MaskAndSampleFailsBeforeAnyPass) {
  ASSERT_OK_AND_ASSIGN(auto verifier, CreateVerifier("verify"));
  EXPECT_THAT(verifier->MaskAndSample({}),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST_P(LlmLiteRtVerifierFakeLmTest, RunRejectsEmptyAndOversizedPasses) {
  ASSERT_OK_AND_ASSIGN(auto verifier, CreateVerifier("verify"));
  ASSERT_OK_AND_ASSIGN(auto state_buffers,
                       state_->GetStateBuffers(*model_, "verify"));
  EXPECT_THAT(verifier->Run(/*position=*/0, {}, state_buffers),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(verifier->Run(/*position=*/0, {3, 1, 4, 1, 5}, state_buffers),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_P(LlmLiteRtVerifierFakeLmTest, PassRunnerStartsThePasses) {
  std::vector<std::string> run_signatures;
  ASSERT_OK_AND_ASSIGN(
      auto verifier,
      CreateVerifier("verify", /*pad_token_id=*/-1,
                     [&](absl::string_view signature_key,
                         absl::flat_hash_map<absl::string_view, TensorBuffer>&
                             input_buffers,
                         absl::flat_hash_map<absl::string_view, TensorBuffer>&
                             output_buffers) -> absl::Status {
                       run_signatures.push_back(std::string(signature_key));
                       // The runner gets all the buffers of the pass, including
                       // the KV cache.
                       EXPECT_TRUE(input_buffers.contains("kv_cache_k_0"));
                       EXPECT_TRUE(output_buffers.contains("kv_cache_k_0"));
                       EXPECT_TRUE(output_buffers.contains("logits"));
                       bool async = true;
                       LITERT_RETURN_IF_ERROR(
                           model_->RunAsync(signature_key, input_buffers,
                                            output_buffers, async));
                       return absl::OkStatus();
                     }));
  const std::vector<int> context = {3, 1, 4, 1, 5, 9, 2, 6};
  for (int position : {0, 4}) {
    SCOPED_TRACE(::testing::Message() << "Position: " << position);
    ASSERT_OK(RunPass(*verifier, context, position, kVerifyWidth));
    EXPECT_THAT(
        verifier->MaskAndSample({}),
        IsOkAndHolds(ElementsAreArray(FakeLmPassTokens(context, position))));
  }
  EXPECT_THAT(run_signatures, ElementsAre("verify", "verify"));
}

TEST_P(LlmLiteRtVerifierFakeLmTest, RunFailsIfThePassRunnerFails) {
  ASSERT_OK_AND_ASSIGN(
      auto verifier,
      CreateVerifier(
          "verify", /*pad_token_id=*/-1,
          [](absl::string_view /*signature_key*/,
             absl::flat_hash_map<absl::string_view, TensorBuffer>& /*inputs*/,
             absl::flat_hash_map<absl::string_view, TensorBuffer>&
             /*outputs*/) {
            return absl::UnavailableError("The pass can't start.");
          }));
  EXPECT_THAT(RunPass(*verifier, {3, 1, 4, 1}, /*position=*/0, kVerifyWidth),
              StatusIs(absl::StatusCode::kUnavailable));
  // No pass has started.
  EXPECT_THAT(verifier->MaskAndSample({}),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST_P(LlmLiteRtVerifierFakeLmTest, GetOutputBufferReturnsNonKvCacheOutputs) {
  ASSERT_OK_AND_ASSIGN(auto verifier, CreateVerifier("verify"));
  EXPECT_OK(verifier->GetOutputBuffer("logits"));
  EXPECT_THAT(verifier->GetOutputBuffer("kv_cache_k_0"),
              StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(verifier->GetOutputBuffer("activations"),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST_P(LlmLiteRtVerifierFakeLmTest, SamplerSamplesWithTheGivenGenerator) {
  ASSERT_OK_AND_ASSIGN(auto verifier, CreateVerifier("verify"));
  auto rand_gen = std::make_shared<std::default_random_engine>(7);
  EXPECT_THAT(
      verifier->UpdateSamplerConfig(SampledParams(/*seed=*/0), rand_gen),
      IsOkAndHolds(rand_gen));

  ASSERT_OK(RunPass(*verifier, {3, 1, 4, 1}, /*position=*/0, kVerifyWidth));
  const std::default_random_engine before_sampling = *rand_gen;
  ASSERT_OK(verifier->MaskAndSample({}));
  EXPECT_THAT(*rand_gen, Ne(before_sampling));
}

TEST_P(LlmLiteRtVerifierFakeLmTest,
       SamplerWithoutGeneratorUsesOneGeneratorPerSeed) {
  ASSERT_OK_AND_ASSIGN(auto verifier, CreateVerifier("verify"));
  ASSERT_OK_AND_ASSIGN(auto first, verifier->UpdateSamplerConfig(
                                       SampledParams(/*seed=*/1), nullptr));
  ASSERT_NE(first, nullptr);
  EXPECT_THAT(verifier->UpdateSamplerConfig(SampledParams(/*seed=*/1), nullptr),
              IsOkAndHolds(first));
  EXPECT_THAT(verifier->UpdateSamplerConfig(SampledParams(/*seed=*/2), nullptr),
              IsOkAndHolds(Ne(first)));
}

TEST_P(LlmLiteRtVerifierFakeLmTest, GreedySamplerConfigKeepsExactPredictions) {
  ASSERT_OK_AND_ASSIGN(auto verifier, CreateVerifier("verify"));
  ASSERT_OK(verifier->UpdateSamplerConfig(SampledParams(/*seed=*/0), nullptr));
  ASSERT_OK(verifier->UpdateSamplerConfig(GreedySamplerParams(), nullptr));
  const std::vector<int> context = {3, 1, 4, 1};
  ASSERT_OK(RunPass(*verifier, context, /*position=*/0, kVerifyWidth));
  EXPECT_THAT(verifier->MaskAndSample({}),
              IsOkAndHolds(
                  ElementsAreArray(FakeLmPassTokens(context, /*position=*/0))));
}

TEST_P(LlmLiteRtVerifierFakeLmTest, VerifyAcceptsTheDraftTokensThatItPredicts) {
  ASSERT_OK_AND_ASSIGN(auto verifier, CreateVerifier("verify"));
  const std::vector<int> context = FakeLmContinuation({3}, /*num_tokens=*/5);
  const std::vector<int> drafts(context.begin() + 1, context.end());
  ASSERT_OK_AND_ASSIGN(auto state_buffers,
                       state_->GetStateBuffers(*model_, "verify"));
  ASSERT_OK_AND_ASSIGN(
      VerifyPassResult result,
      verifier->Verify(/*position=*/0, context[0], drafts, state_buffers,
                       /*constraint_states=*/nullptr));
  // The pass takes 3 draft tokens, and its last row accepts the fourth, which
  // becomes the pending token of the next pass.
  EXPECT_THAT(result.outcome, FieldsAre(4, Eq(std::nullopt), 0));
  EXPECT_EQ(result.constraint_state, nullptr);
}

TEST_P(LlmLiteRtVerifierFakeLmTest, VerifyReturnsTheCorrectionOfAMismatch) {
  ASSERT_OK_AND_ASSIGN(auto verifier, CreateVerifier("verify"));
  const std::vector<int> context = FakeLmContinuation({3}, /*num_tokens=*/5);
  std::vector<int> drafts(context.begin() + 1, context.end());
  drafts[1] = (drafts[1] + 1) % kFakeLmVocabSize;
  ASSERT_OK_AND_ASSIGN(auto state_buffers,
                       state_->GetStateBuffers(*model_, "verify"));
  EXPECT_THAT(
      verifier->Verify(/*position=*/0, context[0], drafts, state_buffers,
                       /*constraint_states=*/nullptr),
      IsOkAndHolds(Field(&VerifyPassResult::outcome,
                         FieldsAre(1, Optional(context[2]), 0))));
}

TEST_P(LlmLiteRtVerifierFakeLmTest, VerifyPadsAPassOfAShortDraft) {
  ASSERT_OK_AND_ASSIGN(auto verifier, CreateVerifier("verify"));
  const std::vector<int> context = FakeLmContinuation({3}, /*num_tokens=*/3);
  const std::vector<int> drafts = {context[1], context[2]};
  ASSERT_OK_AND_ASSIGN(auto state_buffers,
                       state_->GetStateBuffers(*model_, "verify"));
  EXPECT_THAT(
      verifier->Verify(/*position=*/0, context[0], drafts, state_buffers,
                       /*constraint_states=*/nullptr),
      IsOkAndHolds(Field(&VerifyPassResult::outcome,
                         FieldsAre(2, Optional(context[3]), 0))));
}

TEST_P(LlmLiteRtVerifierFakeLmTest, VerifyAcceptsMismatchesWithLenience) {
  ASSERT_OK_AND_ASSIGN(auto verifier, CreateVerifier("verify"));
  // None of the draft tokens is predicted, but full lenience accepts them.
  const std::vector<int> context = FakeLmContinuation({3}, /*num_tokens=*/1);
  const std::vector<int> drafts = {(context[1] + 1) % kFakeLmVocabSize};
  ASSERT_OK_AND_ASSIGN(auto state_buffers,
                       state_->GetStateBuffers(*model_, "verify"));
  int num_uniforms = 0;
  EXPECT_THAT(verifier->Verify(/*position=*/0, context[0], drafts,
                               state_buffers, /*constraint_states=*/nullptr,
                               /*lenience=*/1.0f,
                               [&](int) {
                                 ++num_uniforms;
                                 return 0.5;
                               }),
              IsOkAndHolds(Field(&VerifyPassResult::outcome,
                                 FieldsAre(1, Ne(std::nullopt), 1))));
  EXPECT_EQ(num_uniforms, 1);
}

TEST_P(LlmLiteRtVerifierFakeLmTest, VerifyMasksTheRowsWithTheConstraint) {
  ASSERT_OK_AND_ASSIGN(auto verifier, CreateVerifier("verify"));
  // The fake LM predicts none of the forced tokens after the pending token 3.
  StrictFakeConstraint constraint({10, 11, 12, 13, 14});
  const std::unique_ptr<Constraint::State> start = constraint.Start();
  DraftConstraintStates constraint_states(constraint, *start);
  ASSERT_OK_AND_ASSIGN(auto state_buffers,
                       state_->GetStateBuffers(*model_, "verify"));
  ASSERT_OK_AND_ASSIGN(
      VerifyPassResult result,
      verifier->Verify(/*position=*/0, /*pending_token=*/3, {10, 11, 12, 13},
                       state_buffers, &constraint_states));
  EXPECT_THAT(result.outcome, FieldsAre(4, Eq(std::nullopt), 0));
  // The state follows the draft tokens that the pass took, but not the one
  // that becomes the pending token.
  ASSERT_NE(result.constraint_state, nullptr);
  EXPECT_EQ(NumForcedTokens(*result.constraint_state), 3);
}

TEST_P(LlmLiteRtVerifierFakeLmTest,
       VerifyEndsTheDraftAtATokenThatTheConstraintDisallows) {
  ASSERT_OK_AND_ASSIGN(auto verifier, CreateVerifier("verify"));
  StrictFakeConstraint constraint({10, 11, 12, 13});
  const std::unique_ptr<Constraint::State> start = constraint.Start();
  DraftConstraintStates constraint_states(constraint, *start);
  ASSERT_OK_AND_ASSIGN(auto state_buffers,
                       state_->GetStateBuffers(*model_, "verify"));
  ASSERT_OK_AND_ASSIGN(
      VerifyPassResult result,
      verifier->Verify(/*position=*/0, /*pending_token=*/3, {10, 0, 12},
                       state_buffers, &constraint_states));
  // The masked row of the disallowed draft token decodes the forced token.
  EXPECT_THAT(result.outcome, FieldsAre(1, Optional(11), 0));
  ASSERT_NE(result.constraint_state, nullptr);
  EXPECT_EQ(NumForcedTokens(*result.constraint_state), 1);
}

TEST_P(LlmLiteRtVerifierFakeLmTest, VerifyUsesTheMasksOfAppendedDraftTokens) {
  ASSERT_OK_AND_ASSIGN(auto verifier, CreateVerifier("verify"));
  StrictFakeConstraint constraint({10, 11, 12, 13});
  const std::unique_ptr<Constraint::State> start = constraint.Start();
  DraftConstraintStates constraint_states(constraint, *start);
  // Like a drafter, which drafts each token under its mask.
  const std::vector<int> drafts = {10, 11, 12};
  for (int token : drafts) {
    ASSERT_OK(constraint_states.Append(token));
  }
  ASSERT_EQ(constraint.num_masks(), 3);
  ASSERT_OK_AND_ASSIGN(auto state_buffers,
                       state_->GetStateBuffers(*model_, "verify"));
  ASSERT_OK_AND_ASSIGN(
      VerifyPassResult result,
      verifier->Verify(/*position=*/0, /*pending_token=*/3, drafts,
                       state_buffers, &constraint_states));
  EXPECT_THAT(result.outcome, FieldsAre(3, Optional(13), 0));
  // Only the mask of the last row is new.
  EXPECT_EQ(constraint.num_masks(), 4);
  ASSERT_NE(result.constraint_state, nullptr);
  EXPECT_EQ(NumForcedTokens(*result.constraint_state), 3);
}

TEST_P(LlmLiteRtVerifierFakeLmTest, LenienceNeedsDraftTokensSampledUnderMasks) {
  ASSERT_OK_AND_ASSIGN(auto verifier, CreateVerifier("verify"));
  StrictFakeConstraint constraint({10, 11, 12, 13});
  const std::unique_ptr<Constraint::State> start = constraint.Start();
  DraftConstraintStates constraint_states(constraint, *start);
  ASSERT_OK_AND_ASSIGN(auto state_buffers,
                       state_->GetStateBuffers(*model_, "verify"));
  // The draft tokens weren't appended, i.e. sampled under their masks, so
  // lenience could accept disallowed ones.
  EXPECT_THAT(verifier->Verify(/*position=*/0, /*pending_token=*/3, {10, 11},
                               state_buffers, &constraint_states,
                               /*lenience=*/1.0f, [](int) { return 0.5; }),
              StatusIs(absl::StatusCode::kInternal));
}

INSTANTIATE_TEST_SUITE_P(FakeLms, LlmLiteRtVerifierFakeLmTest,
                         ::testing::Values(kTokenInputLm, kEmbeddingInputLm),
                         [](const ::testing::TestParamInfo<FakeLm>& info) {
                           return std::string(info.param.name);
                         });

TEST_F(LlmLiteRtVerifierTest, SignaturesOfDifferentWidthsShareTheKvCache) {
  ASSERT_NO_FATAL_FAILURE(LoadFakeLm(kTokenInputLm));
  ASSERT_OK_AND_ASSIGN(auto verifier, CreateVerifier("verify"));
  ASSERT_OK_AND_ASSIGN(auto wide_verifier, CreateVerifier("verify_8"));
  EXPECT_EQ(wide_verifier->width(), 8);
  const std::vector<int> context = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9};

  ASSERT_OK(RunPass(*wide_verifier, context, /*position=*/0,
                    /*num_tokens=*/8));
  EXPECT_THAT(wide_verifier->MaskAndSample({}),
              IsOkAndHolds(ElementsAreArray(FakeLmPassTokens(
                  context, /*position=*/0, /*num_tokens=*/8))));
  ASSERT_OK(RunPass(*verifier, context, /*position=*/8, kVerifyWidth));
  EXPECT_THAT(verifier->MaskAndSample({}),
              IsOkAndHolds(
                  ElementsAreArray(FakeLmPassTokens(context, /*position=*/8))));
  ASSERT_OK(RunPass(*wide_verifier, context, /*position=*/12,
                    /*num_tokens=*/1));
  EXPECT_THAT(wide_verifier->MaskAndSample({}),
              IsOkAndHolds(ElementsAreArray(FakeLmPassTokens(
                  context, /*position=*/12, /*num_tokens=*/1))));
}

TEST_F(LlmLiteRtVerifierTest, CreateFailsWithoutTheEmbeddingLookup) {
  ASSERT_NO_FATAL_FAILURE(LoadFakeLm(kEmbeddingInputLm));
  embedding_lookup_.reset();
  EXPECT_THAT(CreateVerifier("verify"),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST_F(LlmLiteRtVerifierTest, PositionsInputMayBeNamedPositions) {
  ASSERT_NO_FATAL_FAILURE(LoadFakeLm(kInvalidVerifyLm));
  ASSERT_OK_AND_ASSIGN(auto verifier, CreateVerifier("verify_positions"));
  EXPECT_EQ(verifier->width(), kVerifyWidth);
  const std::vector<int> context = {3, 1, 4};
  ASSERT_OK(RunPass(*verifier, context, /*position=*/0, /*num_tokens=*/3));
  EXPECT_THAT(verifier->MaskAndSample({}),
              IsOkAndHolds(ElementsAreArray(FakeLmPassTokens(
                  context, /*position=*/0, /*num_tokens=*/3))));
}

class FindVerifySignaturesTest : public LlmLiteRtVerifierTest {
 protected:
  absl::StatusOr<std::vector<VerifySignatureInfo>> Find(
      std::vector<std::string> selected_signatures = {},
      bool has_embedding_lookup = false) {
    return FindVerifySignatures(*model_, "decode", *state_, selected_signatures,
                                has_embedding_lookup,
                                /*has_per_layer_embedding_lookup=*/false);
  }
};

::testing::Matcher<VerifySignatureInfo> Usable(absl::string_view key,
                                               int width) {
  return AllOf(Field(&VerifySignatureInfo::key, std::string(key)),
               Field(&VerifySignatureInfo::width, width),
               Field(&VerifySignatureInfo::status, IsOk()));
}

::testing::Matcher<VerifySignatureInfo> Unusable(absl::string_view key,
                                                 absl::string_view reason) {
  return AllOf(Field(&VerifySignatureInfo::key, std::string(key)),
               Field(&VerifySignatureInfo::width, 0),
               Field(&VerifySignatureInfo::status,
                     StatusIs(absl::StatusCode::kFailedPrecondition,
                              HasSubstr(std::string(reason)))));
}

TEST_F(FindVerifySignaturesTest, FindsEveryVerifySignature) {
  ASSERT_NO_FATAL_FAILURE(LoadFakeLm(kTokenInputLm));
  EXPECT_THAT(Find(), IsOkAndHolds(UnorderedElementsAre(
                          Usable("verify", 4), Usable("verify_8", 8))));
}

TEST_F(FindVerifySignaturesTest, FindsNoneInAModelWithoutVerifySignatures) {
  ASSERT_NO_FATAL_FAILURE(LoadFakeLm(kNoVerifyLm));
  EXPECT_THAT(Find(), IsOkAndHolds(IsEmpty()));
}

TEST_F(FindVerifySignaturesTest, OnlySelectedSignaturesAreUsable) {
  ASSERT_NO_FATAL_FAILURE(LoadFakeLm(kTokenInputLm));
  EXPECT_THAT(Find({"prefill_8", "decode", "verify_8"}),
              IsOkAndHolds(UnorderedElementsAre(Unusable("verify", "selected"),
                                                Usable("verify_8", 8))));
}

TEST_F(FindVerifySignaturesTest, EmbeddingInputsNeedTheEmbeddingLookup) {
  ASSERT_NO_FATAL_FAILURE(LoadFakeLm(kEmbeddingInputLm));
  EXPECT_THAT(Find({}, /*has_embedding_lookup=*/true),
              IsOkAndHolds(ElementsAre(Usable("verify", 4))));
  EXPECT_THAT(
      Find({}, /*has_embedding_lookup=*/false),
      IsOkAndHolds(ElementsAre(Unusable("verify", "embedding lookup"))));
}

TEST_F(FindVerifySignaturesTest, ReportsWhySignaturesBreakTheContract) {
  ASSERT_NO_FATAL_FAILURE(LoadFakeLm(kInvalidVerifyLm));
  EXPECT_THAT(Find(),
              IsOkAndHolds(UnorderedElementsAre(
                  Usable("verify", 4), Usable("verify_positions", 4),
                  Unusable("verify_vocab_16", "vocabulary size of decode"),
                  Unusable("verify_rank2", "shape [W]"),
                  Unusable("verify_int64_positions", "positions must be int32"),
                  Unusable("verify_int64_tokens", "tokens must be int32"),
                  Unusable("verify_short_kv", "same KV cache tensors"),
                  Unusable("verify_no_logits", "no logits output"),
                  Unusable("verify_local_mask", "local attention mask"),
                  Unusable("verify_lora", "same LoRA inputs"),
                  Unusable("verify_tokens_and_embeddings",
                           "both tokens and embeddings"),
                  Unusable("verify_param_tensor", "param tensor"),
                  Unusable("verify_unknown_input", "unknown input: unknown"))));
}

TEST_F(FindVerifySignaturesTest, EmbeddingsMustHaveTheShapeOfDecodes) {
  ASSERT_NO_FATAL_FAILURE(LoadFakeLm(kEmbeddingInputInvalidVerifyLm));
  EXPECT_THAT(Find({}, /*has_embedding_lookup=*/true),
              IsOkAndHolds(UnorderedElementsAre(
                  Usable("verify", 4),
                  Unusable("verify_embedding_rows",
                           "embeddings must have the shape [1, 4, 4]"),
                  Unusable("verify_embedding_dim",
                           "embeddings must have the shape [1, 4, 4]"))));
}

// Holds the tensors of another KV cache, but reports fewer entries.
class FewerEntriesState : public StateInterface {
 public:
  FewerEntriesState(const StateInterface& state, int num_entries)
      : state_(state), num_entries_(num_entries) {}

  int GetNumEntries() const override { return num_entries_; }
  int GetBatchSize() const override { return state_.GetBatchSize(); }
  bool Contains(absl::string_view tensor_name) const override {
    return state_.Contains(tensor_name);
  }
  absl::StatusOr<std::string> Serialize() const override {
    return absl::UnimplementedError("Serialize");
  }
  absl::Status Load(absl::string_view) override {
    return absl::UnimplementedError("Load");
  }
  absl::Status SelectAndCopyFrom(StateInterface&, int) override {
    return absl::UnimplementedError("SelectAndCopyFrom");
  }
  absl::Status BroadcastAndCopyFrom(StateInterface&) override {
    return absl::UnimplementedError("BroadcastAndCopyFrom");
  }
  absl::StatusOr<std::unique_ptr<StateInterface>> DeepCopy() const override {
    return absl::UnimplementedError("DeepCopy");
  }
  absl::Status Clear() override { return absl::UnimplementedError("Clear"); }

 private:
  const StateInterface& state_;
  const int num_entries_;
};

TEST_F(FindVerifySignaturesTest, WidthMustNotExceedTheKvCacheSize) {
  ASSERT_NO_FATAL_FAILURE(LoadFakeLm(kTokenInputLm));
  const FewerEntriesState state(*state_, /*num_entries=*/6);
  EXPECT_THAT(FindVerifySignatures(*model_, "decode", state,
                                   /*selected_signatures=*/{},
                                   /*has_embedding_lookup=*/false,
                                   /*has_per_layer_embedding_lookup=*/false),
              IsOkAndHolds(UnorderedElementsAre(
                  Usable("verify", 4),
                  Unusable("verify_8", "exceeds the KV cache size 6"))));
}

class LlmLiteRtDraftVerifierTest : public LlmLiteRtVerifierTest {
 protected:
  // Loads `lm` and creates `draft_verifier_` for it, restricted to
  // `selected_signatures` if not empty. The draft verifier records the
  // signature of every pass in `passes_`.
  void CreateDraftVerifier(const FakeLm& lm,
                           std::vector<std::string> selected_signatures = {}) {
    ASSERT_NO_FATAL_FAILURE(LoadFakeLm(lm));
    ASSERT_OK_AND_ASSIGN(auto model_assets,
                         ModelAssets::Create(FakeLmModelPath(lm.model_file)));
    ASSERT_OK_AND_ASSIGN(auto settings, LlmExecutorSettings::CreateDefault(
                                            model_assets, Backend::CPU));
    settings.SetSelectedSignatures(std::move(selected_signatures));
    settings_ = std::make_unique<LlmExecutorSettings>(std::move(settings));
    ASSERT_OK_AND_ASSIGN(
        draft_verifier_,
        LlmLiteRtDraftVerifier::Create(
            *env_, *model_, "decode", *state_, *settings_,
            embedding_lookup_.get(),
            /*per_layer_embedding_lookup=*/nullptr,
            /*executor_metadata=*/nullptr,
            [this](absl::string_view signature_key,
                   absl::flat_hash_map<absl::string_view, TensorBuffer>&
                       input_buffers,
                   absl::flat_hash_map<absl::string_view, TensorBuffer>&
                       output_buffers) -> absl::Status {
              passes_.push_back(std::string(signature_key));
              bool async = true;
              LITERT_RETURN_IF_ERROR(model_->RunAsync(
                  signature_key, input_buffers, output_buffers, async));
              return absl::OkStatus();
            }));
  }

  std::unique_ptr<LlmExecutorSettings> settings_;
  std::unique_ptr<LlmLiteRtDraftVerifier> draft_verifier_;
  // The signatures of the passes, in order.
  std::vector<std::string> passes_;
};

TEST_F(LlmLiteRtDraftVerifierTest, SupportsModelsWithAUsableVerifySignature) {
  ASSERT_NO_FATAL_FAILURE(CreateDraftVerifier(kTokenInputLm));
  EXPECT_OK(draft_verifier_->CheckSupport());
}

TEST_F(LlmLiteRtDraftVerifierTest, DoesntSupportModelsWithoutVerifySignatures) {
  ASSERT_NO_FATAL_FAILURE(CreateDraftVerifier(kNoVerifyLm));
  EXPECT_THAT(
      draft_verifier_->CheckSupport(),
      StatusIs(absl::StatusCode::kFailedPrecondition, HasSubstr("has none")));
}

TEST_F(LlmLiteRtDraftVerifierTest, ReportsWhyNoVerifySignatureIsUsable) {
  ASSERT_NO_FATAL_FAILURE(
      CreateDraftVerifier(kTokenInputLm, {"prefill_8", "decode"}));
  EXPECT_THAT(draft_verifier_->CheckSupport(),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       AllOf(HasSubstr("verify: "), HasSubstr("verify_8: "))));
}

TEST_F(LlmLiteRtDraftVerifierTest, ChecksWhetherAPassFitsIntoTheKvCache) {
  ASSERT_NO_FATAL_FAILURE(CreateDraftVerifier(kTokenInputLm));
  EXPECT_TRUE(draft_verifier_->PassFits(/*position=*/3, kFakeLmContextSize));
  EXPECT_TRUE(draft_verifier_->PassFits(/*position=*/60, kFakeLmContextSize));
  // A pass of width 4 at position 61 would end past the 64 KV cache entries.
  EXPECT_FALSE(draft_verifier_->PassFits(/*position=*/61, kFakeLmContextSize));
}

TEST_F(LlmLiteRtDraftVerifierTest,
       PassesUseTheNarrowestWidthThatTakesTheDraft) {
  ASSERT_NO_FATAL_FAILURE(CreateDraftVerifier(kTokenInputLm));
  const std::vector<int> context = FakeLmContinuation({3}, /*num_tokens=*/11);
  const absl::Span<const int> drafts =
      absl::MakeConstSpan(context).subspan(1, 10);
  // The widest pass takes 7 draft tokens, and its last row accepts the eighth,
  // which becomes the pending token of the next pass.
  EXPECT_THAT(draft_verifier_->VerifyPass(
                  /*position=*/0, context[0], drafts, *state_,
                  /*constrained_decoder=*/nullptr,
                  /*feed_pending_token=*/false),
              IsOkAndHolds(FieldsAre(8, Eq(std::nullopt), 0)));
  // The narrowest pass takes the last 2 draft tokens.
  EXPECT_THAT(draft_verifier_->VerifyPass(
                  /*position=*/8, context[8], drafts.subspan(8), *state_,
                  /*constrained_decoder=*/nullptr,
                  /*feed_pending_token=*/false),
              IsOkAndHolds(FieldsAre(2, Optional(context[11]), 0)));
  EXPECT_THAT(passes_, ElementsAre("verify_8", "verify"));
}

TEST_F(LlmLiteRtDraftVerifierTest, PassesFailIfNoneFitsIntoTheKvCache) {
  ASSERT_NO_FATAL_FAILURE(CreateDraftVerifier(kTokenInputLm));
  EXPECT_THAT(draft_verifier_->VerifyPass(
                  /*position=*/61, /*pending_token=*/3, {1}, *state_,
                  /*constrained_decoder=*/nullptr,
                  /*feed_pending_token=*/false),
              StatusIs(absl::StatusCode::kResourceExhausted));
  EXPECT_THAT(passes_, IsEmpty());
}

TEST_F(LlmLiteRtDraftVerifierTest, PassesSampleWithTheSamplerConfig) {
  ASSERT_NO_FATAL_FAILURE(CreateDraftVerifier(kTokenInputLm));
  auto rand_gen = std::make_shared<std::default_random_engine>(7);
  draft_verifier_->SetSamplerConfig(SampledParams(/*seed=*/0), rand_gen);
  const std::default_random_engine before_sampling = *rand_gen;
  ASSERT_OK(draft_verifier_->VerifyPass(
      /*position=*/0, /*pending_token=*/3, {1, 4}, *state_,
      /*constrained_decoder=*/nullptr, /*feed_pending_token=*/false));
  EXPECT_NE(*rand_gen, before_sampling);
}

TEST_F(LlmLiteRtDraftVerifierTest, PassesMaskTheRowsWithTheConstrainedDecoder) {
  ASSERT_NO_FATAL_FAILURE(CreateDraftVerifier(kTokenInputLm));
  // The fake LM predicts none of the forced tokens.
  FakeConstraint constraint({10, 11, 12, 13, 14, 15}, kFakeLmVocabSize);
  ASSERT_OK_AND_ASSIGN(auto decoder,
                       ConstrainedDecoder::CreateForHost(&constraint));
  // A pending token from a prefill is not fed to the decoder.
  EXPECT_THAT(draft_verifier_->VerifyPass(
                  /*position=*/0, /*pending_token=*/3, {10, 11}, *state_,
                  decoder.get(), /*feed_pending_token=*/false),
              IsOkAndHolds(FieldsAre(2, Optional(12), 0)));
  // The decoder's state follows the accepted draft tokens, but not the new
  // pending token.
  ASSERT_OK_AND_ASSIGN(const Constraint::State* state, decoder->GetState(0));
  EXPECT_EQ(NumForcedTokens(*state), 2);

  // A decoded pending token is fed to the decoder first.
  EXPECT_THAT(draft_verifier_->VerifyPass(
                  /*position=*/3, /*pending_token=*/12, {13, 0}, *state_,
                  decoder.get(), /*feed_pending_token=*/true),
              IsOkAndHolds(FieldsAre(1, Optional(14), 0)));
  ASSERT_OK_AND_ASSIGN(state, decoder->GetState(0));
  EXPECT_EQ(NumForcedTokens(*state), 4);
}

TEST_F(LlmLiteRtDraftVerifierTest,
       PassesKeepTheDecodersStateIfNoDraftTokenIsAccepted) {
  ASSERT_NO_FATAL_FAILURE(CreateDraftVerifier(kTokenInputLm));
  FakeConstraint constraint({10, 11, 12}, kFakeLmVocabSize);
  ASSERT_OK_AND_ASSIGN(auto decoder,
                       ConstrainedDecoder::CreateForHost(&constraint));
  ASSERT_OK_AND_ASSIGN(const Constraint::State* state, decoder->GetState(0));
  EXPECT_THAT(draft_verifier_->VerifyPass(
                  /*position=*/0, /*pending_token=*/3, {0}, *state_,
                  decoder.get(), /*feed_pending_token=*/false),
              IsOkAndHolds(FieldsAre(0, Optional(10), 0)));
  ASSERT_OK_AND_ASSIGN(const Constraint::State* new_state,
                       decoder->GetState(0));
  EXPECT_EQ(new_state, state);
  EXPECT_EQ(NumForcedTokens(*new_state), 0);
}

TEST(DraftConstraintStatesTest, MasksFollowTheDraftTokens) {
  StrictFakeConstraint constraint({5, 6, 7, 8});
  const std::unique_ptr<Constraint::State> start = constraint.Start();
  DraftConstraintStates states(constraint, *start);
  EXPECT_EQ(states.num_tokens(), 0);
  ASSERT_OK_AND_ASSIGN(const LogitMask* mask, states.NextMask());
  EXPECT_TRUE(AllowsOnly(mask, 5));
  ASSERT_OK(states.Append(5));
  // Computes the mask of the token first.
  ASSERT_OK(states.Append(6));
  EXPECT_EQ(states.num_tokens(), 2);

  ASSERT_OK_AND_ASSIGN(auto masks, states.PassMasks(/*width=*/4));
  ASSERT_EQ(masks.size(), 4);
  EXPECT_TRUE(AllowsOnly(masks[0].get(), 5));
  EXPECT_TRUE(AllowsOnly(masks[1].get(), 6));
  EXPECT_TRUE(AllowsOnly(masks[2].get(), 7));
  // The padding row.
  EXPECT_EQ(masks[3], nullptr);
  // Every mask is computed once.
  EXPECT_EQ(constraint.num_masks(), 3);
}

TEST(DraftConstraintStatesTest, AppendFailsForADisallowedToken) {
  StrictFakeConstraint constraint({5, 6, 7});
  const std::unique_ptr<Constraint::State> start = constraint.Start();
  DraftConstraintStates states(constraint, *start);
  ASSERT_OK(states.Append(5));
  EXPECT_THAT(states.Append(9), StatusIs(absl::StatusCode::kInvalidArgument));
  // The draft is unchanged.
  EXPECT_EQ(states.num_tokens(), 1);
  ASSERT_OK(states.Append(6));
  EXPECT_EQ(states.num_tokens(), 2);
}

TEST(DraftConstraintStatesTest, PassMasksNeedARowForTheNextToken) {
  StrictFakeConstraint constraint({5, 6, 7});
  const std::unique_ptr<Constraint::State> start = constraint.Start();
  DraftConstraintStates states(constraint, *start);
  ASSERT_OK(states.Append(5));
  ASSERT_OK(states.Append(6));
  EXPECT_THAT(states.PassMasks(/*width=*/2),
              StatusIs(absl::StatusCode::kInternal));
}

TEST(DraftConstraintStatesTest, TakeStateReturnsTheStateAfterTheDraftTokens) {
  StrictFakeConstraint constraint({5, 6, 7});
  const std::unique_ptr<Constraint::State> start = constraint.Start();
  DraftConstraintStates states(constraint, *start);
  ASSERT_OK(states.Append(5));
  ASSERT_OK(states.Append(6));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Constraint::State> state,
                       states.TakeState(1));
  ASSERT_NE(state, nullptr);
  EXPECT_EQ(NumForcedTokens(*state), 1);
  EXPECT_THAT(states.TakeState(3), StatusIs(absl::StatusCode::kInternal));
}

TEST(DraftConstraintStatesTest, TakeStateReturnsNullForTheCallersStartState) {
  StrictFakeConstraint constraint({5, 6, 7});
  const std::unique_ptr<Constraint::State> start = constraint.Start();
  DraftConstraintStates states(constraint, *start);
  ASSERT_OK(states.Append(5));
  EXPECT_THAT(states.TakeState(0), IsOkAndHolds(IsNull()));
}

TEST(DraftConstraintStatesTest, TakeStateMovesOutAnOwnedStartState) {
  StrictFakeConstraint constraint({5, 6, 7});
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Constraint::State> start,
                       constraint.ComputeNext(*constraint.Start(), 5));
  DraftConstraintStates states(constraint, std::move(start));
  ASSERT_OK_AND_ASSIGN(const LogitMask* mask, states.NextMask());
  EXPECT_TRUE(AllowsOnly(mask, 6));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Constraint::State> state,
                       states.TakeState(0));
  ASSERT_NE(state, nullptr);
  EXPECT_EQ(NumForcedTokens(*state), 1);
}

TEST(SeededRandomEngineTest, KeepsStreamWhileSeedIsUnchanged) {
  SeededRandomEngine engine;
  std::default_random_engine reference(7);
  const auto first = engine.Get(7);
  ASSERT_NE(first, nullptr);
  EXPECT_EQ((*first)(), reference());
  const auto again = engine.Get(7);
  EXPECT_EQ(again, first);
  EXPECT_EQ((*again)(), reference());
}

TEST(SeededRandomEngineTest, ReseedsWhenSeedChanges) {
  SeededRandomEngine engine;
  const auto seven = engine.Get(7);
  seven->discard(1);
  const auto eight = engine.Get(8);
  EXPECT_NE(eight, seven);
  std::default_random_engine reference_eight(8);
  EXPECT_EQ((*eight)(), reference_eight());
  // Going back to a previous seed restarts its stream.
  const auto seven_again = engine.Get(7);
  std::default_random_engine reference_seven(7);
  EXPECT_EQ((*seven_again)(), reference_seven());
}

constexpr int kLargeKvCache = 1000;

TEST(SelectVerifyWidthTest, PicksTheSmallestWidthThatTakesAllDrafts) {
  // A pass takes the pending token and the drafts.
  EXPECT_THAT(SelectVerifyWidth({4, 8, 16}, /*num_drafts=*/3, /*position=*/0,
                                kLargeKvCache),
              Optional(4));
  EXPECT_THAT(SelectVerifyWidth({4, 8, 16}, /*num_drafts=*/4, /*position=*/0,
                                kLargeKvCache),
              Optional(8));
  EXPECT_THAT(SelectVerifyWidth({16, 8, 4}, /*num_drafts=*/15, /*position=*/0,
                                kLargeKvCache),
              Optional(16));
}

TEST(SelectVerifyWidthTest, PicksTheLargestWidthIfNoneTakesAllDrafts) {
  EXPECT_THAT(SelectVerifyWidth({4, 16, 8}, /*num_drafts=*/16, /*position=*/0,
                                kLargeKvCache),
              Optional(16));
}

TEST(SelectVerifyWidthTest, OnlyCountsWidthsThatFitIntoTheKvCache) {
  // At position 50 of 64 entries, a pass of width 16 would write past the end.
  EXPECT_THAT(SelectVerifyWidth({4, 8, 16}, /*num_drafts=*/10, /*position=*/50,
                                /*kv_cache_size=*/64),
              Optional(8));
  EXPECT_THAT(SelectVerifyWidth({4, 8, 16}, /*num_drafts=*/10, /*position=*/60,
                                /*kv_cache_size=*/64),
              Optional(4));
  EXPECT_EQ(SelectVerifyWidth({4, 8, 16}, /*num_drafts=*/1, /*position=*/61,
                              /*kv_cache_size=*/64),
            std::nullopt);
  EXPECT_EQ(
      SelectVerifyWidth({}, /*num_drafts=*/1, /*position=*/0, kLargeKvCache),
      std::nullopt);
}

TEST(EvaluateVerifyPassTest, MismatchYieldsTheCorrection) {
  // Inputs: pending, 10, 20, 30.
  EXPECT_THAT(EvaluateVerifyPass({10, 25, 30, 40}, {10, 20, 30, 40},
                                 /*num_draft_inputs=*/3),
              FieldsAre(1, Optional(25), 0));
  EXPECT_THAT(EvaluateVerifyPass({11, 20, 30, 40}, {10, 20, 30},
                                 /*num_draft_inputs=*/3),
              FieldsAre(0, Optional(11), 0));
}

TEST(EvaluateVerifyPassTest, WholeDraftInsideThePassYieldsTheNextToken) {
  EXPECT_THAT(EvaluateVerifyPass({10, 20, 99}, {10, 20},
                                 /*num_draft_inputs=*/2),
              FieldsAre(2, Optional(99), 0));
}

TEST(EvaluateVerifyPassTest, TheLastTokenChecksTheDraftTokenAfterTheInputs) {
  // Accepted: the next draft token becomes the pending token.
  EXPECT_THAT(EvaluateVerifyPass({10, 20, 30, 40}, {10, 20, 30, 40, 50},
                                 /*num_draft_inputs=*/3),
              FieldsAre(4, Eq(std::nullopt), 0));
  // Rejected: the token is the correction.
  EXPECT_THAT(EvaluateVerifyPass({10, 20, 30, 41}, {10, 20, 30, 40, 50},
                                 /*num_draft_inputs=*/3),
              FieldsAre(3, Optional(41), 0));
}

TEST(EvaluateVerifyPassTest, PassOfOnlyThePendingToken) {
  EXPECT_THAT(EvaluateVerifyPass({7}, {7}, /*num_draft_inputs=*/0),
              FieldsAre(1, Eq(std::nullopt), 0));
  EXPECT_THAT(EvaluateVerifyPass({7}, {}, /*num_draft_inputs=*/0),
              FieldsAre(0, Optional(7), 0));
}

// Lenient acceptance: r = (1 - c) * p + c * delta(draft).

TEST(EvaluateVerifyPassTest, LenienceZeroIsExact) {
  std::mt19937_64 rng(42);
  std::uniform_int_distribution<int> token(0, 3);
  for (int trial = 0; trial < 1000; ++trial) {
    std::vector<int> drafts(4);
    std::vector<int> pass_tokens(5);
    for (int& t : drafts) t = token(rng);
    for (int& t : pass_tokens) t = token(rng);
    int uniform_calls = 0;
    const VerifyPassOutcome outcome =
        EvaluateVerifyPass(pass_tokens, drafts, /*num_draft_inputs=*/4,
                           /*lenience=*/0.0f, [&](int) {
                             ++uniform_calls;
                             return 0.0;
                           });
    const VerifyPassOutcome exact =
        EvaluateVerifyPass(pass_tokens, drafts, /*num_draft_inputs=*/4);
    EXPECT_EQ(outcome.num_accepted, exact.num_accepted);
    EXPECT_EQ(outcome.next_token, exact.next_token);
    EXPECT_EQ(outcome.num_lenient_accepted, 0);
    EXPECT_EQ(uniform_calls, 0);
  }
}

TEST(EvaluateVerifyPassTest, LenienceOneAcceptsEverything) {
  EXPECT_THAT(EvaluateVerifyPass({11, 21, 30, 99}, {10, 20, 30},
                                 /*num_draft_inputs=*/3, /*lenience=*/1.0f,
                                 [](int) { return 0.999999; }),
              FieldsAre(3, Optional(99), 2));
}

TEST(EvaluateVerifyPassTest, LenienceDrawsAUniformOnlyOnMismatch) {
  std::vector<int> positions;
  EXPECT_THAT(EvaluateVerifyPass({10, 25, 31, 99}, {10, 20, 30},
                                 /*num_draft_inputs=*/3, /*lenience=*/0.5f,
                                 [&](int i) {
                                   positions.push_back(i);
                                   // Accepts draft token 1, rejects 2.
                                   return i == 1 ? 0.1 : 0.9;
                                 }),
              FieldsAre(2, Optional(31), 1));
  EXPECT_THAT(positions, ElementsAre(1, 2));
}

TEST(EvaluateVerifyPassTest, LenienceAppliesToTheDraftTokenAfterTheInputs) {
  EXPECT_THAT(EvaluateVerifyPass({10, 20, 31}, {10, 20, 30, 40},
                                 /*num_draft_inputs=*/2, /*lenience=*/0.5f,
                                 [](int) { return 0.1; }),
              FieldsAre(3, Eq(std::nullopt), 1));
}

// A mismatching draft token at position i is accepted iff uniform(i) < c.
// Each trial evaluates one random pass at increasing lenience values c, all
// reading the same uniforms. With the same uniforms, raising c only turns
// rejections into acceptances, so the accepted prefix can't shrink.
TEST(EvaluateVerifyPassTest, AcceptanceIsMonotoneInLenience) {
  std::mt19937_64 rng(7);
  // A 3-token vocabulary makes a draft token match about 1/3 of the time, so
  // passes mix exact matches, lenient acceptances and rejections.
  std::uniform_int_distribution<int> token(0, 2);
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  const std::vector<float> lenience = {0.0f, 0.1f, 0.25f, 0.5f, 0.75f, 1.0f};
  for (int trial = 0; trial < 2000; ++trial) {
    std::vector<int> drafts(5);
    // The pass samples one token per draft token plus one after the last.
    std::vector<int> pass_tokens(6);
    // Drawn once per trial and shared by all c, instead of drawing fresh
    // uniforms per call, which would make the property hold only on average.
    std::vector<double> uniforms(5);
    for (int& t : drafts) t = token(rng);
    for (int& t : pass_tokens) t = token(rng);
    for (double& u : uniforms) u = unit(rng);
    int previous = -1;
    for (const float c : lenience) {
      const int accepted =
          EvaluateVerifyPass(pass_tokens, drafts, /*num_draft_inputs=*/5, c,
                             [&](int i) { return uniforms[i]; })
              .num_accepted;
      EXPECT_GE(accepted, previous);
      previous = accepted;
    }
  }
}

TEST(ValidateDraftTokenIdsTest, AcceptsTokenIdsOfTheVocabulary) {
  EXPECT_OK(ValidateDraftTokenIds({0, 5, 31}, /*vocab_size=*/32));
}

TEST(ValidateDraftTokenIdsTest, RejectsEmptyDrafts) {
  EXPECT_THAT(ValidateDraftTokenIds({}, /*vocab_size=*/32),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(ValidateDraftTokenIdsTest, RejectsTokenIdsOutsideTheVocabulary) {
  EXPECT_THAT(ValidateDraftTokenIds({1, 32}, /*vocab_size=*/32),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(ValidateDraftTokenIds({-1}, /*vocab_size=*/32),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace litert::lm
