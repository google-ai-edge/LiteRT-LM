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
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
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
#include "runtime/proto/sampler_params.pb.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

using ::testing::ElementsAreArray;
using ::testing::Ne;
using ::testing::status::IsOkAndHolds;
using ::testing::status::StatusIs;

constexpr char kTestdataDir[] =
    "litert_lm/runtime/testdata";

// The fake LM with an embeddings input, whose `verify` signature has width 4.
constexpr char kFakeLmFile[] = "checksum_fake_lm_embedding_input.tflite";
constexpr char kFakeLmEmbedderFile[] = "checksum_fake_lm_embedder.tflite";
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

// Returns what a pass over `context[position, position + kVerifyWidth)`
// returns: the fake LM's next token after each of these context prefixes.
std::vector<int> FakeLmPassTokens(absl::Span<const int> context, int position) {
  std::vector<int> tokens;
  for (int i = 0; i < kVerifyWidth; ++i) {
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
  void SetUp() override {
    LITERT_ASSERT_OK_AND_ASSIGN(auto env, Environment::Create({}));
    env_ = std::make_unique<Environment>(std::move(env));
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto model, CompiledModel::Create(*env_, FakeLmModelPath(kFakeLmFile),
                                          HwAccelerators::kCpu));
    model_ = std::make_unique<CompiledModel>(std::move(model));
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto embedder,
        Model::CreateFromFile(FakeLmModelPath(kFakeLmEmbedderFile)));
    embedder_ = std::make_unique<Model>(std::move(embedder));
    ASSERT_OK_AND_ASSIGN(
        embedding_lookup_,
        EmbeddingLookupManager::Create(*env_, embedder_.get(),
                                       /*fully_supports_multi_modal=*/false));
    ASSERT_OK_AND_ASSIGN(
        state_, LitertState::Create(*env_, *model_, "prefill_8",
                                    /*executor_metadata=*/nullptr,
                                    LitertState::AllocationPolicy::kInplace,
                                    /*batch_size=*/1));
  }

  absl::StatusOr<std::unique_ptr<LlmLiteRtVerifier>> CreateVerifier(
      absl::string_view signature_key) {
    ABSL_ASSIGN_OR_RETURN(auto model_assets,
                          ModelAssets::Create(FakeLmModelPath(kFakeLmFile)));
    ABSL_ASSIGN_OR_RETURN(auto settings, LlmExecutorSettings::CreateDefault(
                                             model_assets, Backend::CPU));
    return LlmLiteRtVerifier::Create(*env_, *model_, signature_key, settings,
                                     *embedding_lookup_,
                                     /*per_layer_embedding_lookup=*/
                                     std::nullopt);
  }

  // Runs a pass of `context[position, position + width)`.
  absl::Status RunPass(LlmLiteRtVerifier& verifier,
                       absl::Span<const int> context, int position) {
    ABSL_ASSIGN_OR_RETURN(
        auto state_buffers,
        state_->GetStateBuffers(*model_, verifier.signature_key()));
    return verifier.Run(position, context.subspan(position, verifier.width()),
                        state_buffers);
  }

  std::unique_ptr<Environment> env_;
  std::unique_ptr<CompiledModel> model_;
  std::unique_ptr<Model> embedder_;
  std::unique_ptr<EmbeddingLookupManager> embedding_lookup_;
  std::unique_ptr<LitertState> state_;
};

TEST_F(LlmLiteRtVerifierTest, CreateReadsWidthAndVocabSize) {
  ASSERT_OK_AND_ASSIGN(auto verifier, CreateVerifier("verify"));
  EXPECT_EQ(verifier->signature_key(), "verify");
  EXPECT_EQ(verifier->width(), kVerifyWidth);
  EXPECT_EQ(verifier->vocab_size(), kFakeLmVocabSize);
}

TEST_F(LlmLiteRtVerifierTest, CreateFailsForMissingSignature) {
  EXPECT_THAT(CreateVerifier("verify_8"),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST_F(LlmLiteRtVerifierTest, CreateFailsForSignatureWithoutLogits) {
  EXPECT_THAT(CreateVerifier("prefill_8"),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(LlmLiteRtVerifierTest, PassesReturnTheNextTokenOfEveryRow) {
  ASSERT_OK_AND_ASSIGN(auto verifier, CreateVerifier("verify"));
  const std::vector<int> context = {3, 1, 4, 1, 5, 9, 2, 6};
  // Each pass writes the KV cache entries of its positions, which the rows of
  // the next pass attend to.
  for (int position : {0, 4}) {
    SCOPED_TRACE(::testing::Message() << "Position: " << position);
    ASSERT_OK(RunPass(*verifier, context, position));
    EXPECT_THAT(
        verifier->MaskAndSample({}),
        IsOkAndHolds(ElementsAreArray(FakeLmPassTokens(context, position))));
  }
}

TEST_F(LlmLiteRtVerifierTest, MaskAndSampleAppliesTheMaskOfEachRow) {
  ASSERT_OK_AND_ASSIGN(auto verifier, CreateVerifier("verify"));
  const std::vector<int> context = {3, 1, 4, 1};
  std::vector<int> expected = FakeLmPassTokens(context, /*position=*/0);
  const int forced_token = (expected[1] + 1) % kFakeLmVocabSize;
  std::vector<std::unique_ptr<LogitMask>> masks(kVerifyWidth);
  masks[1] =
      BitmapLogitMask::CreateSingleAllowedToken(kFakeLmVocabSize, forced_token);
  expected[1] = forced_token;

  ASSERT_OK(RunPass(*verifier, context, /*position=*/0));
  EXPECT_THAT(verifier->MaskAndSample(masks),
              IsOkAndHolds(ElementsAreArray(expected)));
}

TEST_F(LlmLiteRtVerifierTest, MaskAndSampleRejectsAPartialMaskSequence) {
  ASSERT_OK_AND_ASSIGN(auto verifier, CreateVerifier("verify"));
  ASSERT_OK(RunPass(*verifier, {3, 1, 4, 1}, /*position=*/0));
  std::vector<std::unique_ptr<LogitMask>> masks(kVerifyWidth - 1);
  EXPECT_THAT(verifier->MaskAndSample(masks),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(LlmLiteRtVerifierTest, RunRejectsAWrongNumberOfTokens) {
  ASSERT_OK_AND_ASSIGN(auto verifier, CreateVerifier("verify"));
  ASSERT_OK_AND_ASSIGN(auto state_buffers,
                       state_->GetStateBuffers(*model_, "verify"));
  EXPECT_THAT(verifier->Run(/*position=*/0, {3, 1, 4}, state_buffers),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(LlmLiteRtVerifierTest, GetOutputBufferReturnsNonKvCacheOutputs) {
  ASSERT_OK_AND_ASSIGN(auto verifier, CreateVerifier("verify"));
  EXPECT_OK(verifier->GetOutputBuffer("logits"));
  EXPECT_THAT(verifier->GetOutputBuffer("kv_cache_k_0"),
              StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(verifier->GetOutputBuffer("activations"),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST_F(LlmLiteRtVerifierTest, SamplerSamplesWithTheGivenGenerator) {
  ASSERT_OK_AND_ASSIGN(auto verifier, CreateVerifier("verify"));
  auto rand_gen = std::make_shared<std::default_random_engine>(7);
  EXPECT_THAT(
      verifier->UpdateSamplerConfig(SampledParams(/*seed=*/0), rand_gen),
      IsOkAndHolds(rand_gen));

  ASSERT_OK(RunPass(*verifier, {3, 1, 4, 1}, /*position=*/0));
  const std::default_random_engine before_sampling = *rand_gen;
  ASSERT_OK(verifier->MaskAndSample({}));
  EXPECT_THAT(*rand_gen, Ne(before_sampling));
}

TEST_F(LlmLiteRtVerifierTest, SamplerWithoutGeneratorUsesOneGeneratorPerSeed) {
  ASSERT_OK_AND_ASSIGN(auto verifier, CreateVerifier("verify"));
  ASSERT_OK_AND_ASSIGN(auto first, verifier->UpdateSamplerConfig(
                                       SampledParams(/*seed=*/1), nullptr));
  ASSERT_NE(first, nullptr);
  EXPECT_THAT(verifier->UpdateSamplerConfig(SampledParams(/*seed=*/1), nullptr),
              IsOkAndHolds(first));
  EXPECT_THAT(verifier->UpdateSamplerConfig(SampledParams(/*seed=*/2), nullptr),
              IsOkAndHolds(Ne(first)));
}

TEST_F(LlmLiteRtVerifierTest, GreedySamplerConfigKeepsExactPredictions) {
  ASSERT_OK_AND_ASSIGN(auto verifier, CreateVerifier("verify"));
  ASSERT_OK(verifier->UpdateSamplerConfig(SampledParams(/*seed=*/0), nullptr));
  ASSERT_OK(verifier->UpdateSamplerConfig(GreedySamplerParams(), nullptr));
  const std::vector<int> context = {3, 1, 4, 1};
  ASSERT_OK(RunPass(*verifier, context, /*position=*/0));
  EXPECT_THAT(verifier->MaskAndSample({}),
              IsOkAndHolds(
                  ElementsAreArray(FakeLmPassTokens(context, /*position=*/0))));
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
