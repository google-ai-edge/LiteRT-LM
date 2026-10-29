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
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_common.h"  // from @litert
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/test/matchers.h"  // from @litert
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
using ::testing::Field;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::Ne;
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
      absl::string_view signature_key, int pad_token_id = -1) {
    ABSL_ASSIGN_OR_RETURN(auto model_assets,
                          ModelAssets::Create(FakeLmModelPath(lm_.model_file)));
    ABSL_ASSIGN_OR_RETURN(auto settings, LlmExecutorSettings::CreateDefault(
                                             model_assets, Backend::CPU));
    settings.SetPadTokenId(pad_token_id);
    return LlmLiteRtVerifier::Create(*env_, *model_, signature_key, settings,
                                     embedding_lookup_.get(),
                                     /*per_layer_embedding_lookup=*/nullptr);
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

}  // namespace
}  // namespace litert::lm
