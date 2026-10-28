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

#include "runtime/executor/llm_litert_mtp_drafter.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17)
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_common.h"  // from @litert
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_options.h"  // from @litert
#include "litert/test/matchers.h"  // from @litert
#include "runtime/components/embedding_lookup/embedding_lookup_manager.h"
#include "runtime/components/model_resources.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

using ::testing::status::StatusIs;

class TfLiteModelResources : public ModelResources {
 public:
  using ModelResources::HasTFLiteModel;

  static absl::StatusOr<std::unique_ptr<TfLiteModelResources>> Create(
      const ModelAssets& model_assets, bool with_mtp_drafter = false) {
    LITERT_ASSIGN_OR_RETURN(auto path, model_assets.GetPath());
    LITERT_ASSIGN_OR_RETURN(auto model,
                            Model::CreateFromFile(std::string(path)));
    return absl::WrapUnique(
        new TfLiteModelResources(std::move(model), with_mtp_drafter));
  }

  bool HasTFLiteModel(proto::LlmMetadata::TfLiteModelType model_type) override {
    return model_type == proto::LlmMetadata::TF_LITE_PREFILL_DECODE ||
           (model_type == proto::LlmMetadata::TF_LITE_MTP_DRAFTER &&
            with_mtp_drafter_);
  }

  absl::StatusOr<const Model*> GetTFLiteModel(
      proto::LlmMetadata::TfLiteModelType model_type) override {
    if (model_type == proto::LlmMetadata::TF_LITE_PREFILL_DECODE) {
      return &model_;
    }
    if (model_type == proto::LlmMetadata::TF_LITE_MTP_DRAFTER) {
      if (with_mtp_drafter_) {
        return &model_;
      } else {
        return absl::NotFoundError("MTP Drafter model not found");
      }
    }
    return absl::UnimplementedError("Unsupported model type");
  }

  absl::StatusOr<absl::string_view> GetTFLiteModelBuffer(
      proto::LlmMetadata::TfLiteModelType model_type) override {
    return absl::UnimplementedError("GetTFLiteModelBuffer not implemented.");
  }

  absl::StatusOr<std::unique_ptr<Tokenizer>> GetTokenizer() override {
    return absl::UnimplementedError("GetTokenizer not implemented.");
  }

  absl::StatusOr<const proto::LlmMetadata*> GetLlmMetadata() override {
    return absl::UnimplementedError("GetLlmMetadata not implemented.");
  }

  absl::StatusOr<std::pair<size_t, size_t>> GetWeightsSectionOffset(
      proto::LlmMetadata::TfLiteModelType model_type) override {
    return absl::UnimplementedError(
        "GetTFLiteModelSectionFileRegion not implemented.");
  }

  absl::StatusOr<const proto::ExecutorMetadata*> GetExecutorMetadata()
      override {
    return absl::UnimplementedError("GetExecutorMetadata not implemented.");
  }

  std::optional<std::string> GetTFLiteModelBackendConstraint(
      proto::LlmMetadata::TfLiteModelType model_type) override {
    return std::nullopt;
  }

  std::optional<std::string> GetTFLiteModelPreferActivationType(
      proto::LlmMetadata::TfLiteModelType model_type) override {
    return std::nullopt;
  }

  absl::StatusOr<std::reference_wrapper<ScopedFile>> GetScopedFile() override {
    return absl::UnimplementedError("GetScopedFile not implemented.");
  }

  absl::StatusOr<FileRegion> GetTFLiteModelSectionFileRegion(
      proto::LlmMetadata::TfLiteModelType model_type) override {
    return absl::UnimplementedError(
        "GetTFLiteModelSectionFileRegion not implemented.");
  }

 private:
  explicit TfLiteModelResources(Model model, bool with_mtp_drafter = false)
      : model_(std::move(model)), with_mtp_drafter_(with_mtp_drafter) {}

  Model model_;
  bool with_mtp_drafter_;
};

TEST(LlmLiteRtMtpDrafterTest, CreateFromModelResources_MissingVerifySignature) {
  const std::filesystem::path model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/magic_test_decode_batch.tflite";
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create(model_path.string()));
  ASSERT_OK_AND_ASSIGN(auto executor_settings,
                       LlmExecutorSettings::CreateDefault(model_assets));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto base_model, CompiledModel::Create(env, model_path.string(),
                                             litert::HwAccelerators::kCpu));
  ASSERT_OK_AND_ASSIGN(auto model_resources,
                       TfLiteModelResources::Create(model_assets,
                                                    /*with_mtp_drafter=*/true));

  const std::filesystem::path embedder_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/dummy_embedder.tflite";
  LITERT_ASSERT_OK_AND_ASSIGN(auto embedder_model,
                              Model::CreateFromFile(embedder_path.string()));
  ASSERT_OK_AND_ASSIGN(
      auto embedding_lookup,
      EmbeddingLookupManager::Create(env, &embedder_model,
                                     /*fully_supports_multi_modal=*/false));

  auto drafter_or = LlmLiteRtMtpDrafter::Create(
      env, *model_resources, executor_settings, base_model, *embedding_lookup,
      /*ple_manager=*/std::nullopt);
  EXPECT_THAT(drafter_or, StatusIs(absl::StatusCode::kNotFound));
}

TEST(LlmLiteRtMtpDrafterTest,
     CreateFromPreCompiledModel_MissingVerifySignature) {
  const std::filesystem::path model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/magic_test_decode_batch.tflite";
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create(model_path.string()));
  ASSERT_OK_AND_ASSIGN(auto executor_settings,
                       LlmExecutorSettings::CreateDefault(model_assets));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));
  LITERT_ASSERT_OK_AND_ASSIGN(auto mtp_model,
                              Model::CreateFromFile(model_path.string()));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto compiled_mtp_model,
      CompiledModel::Create(env, model_path.string(),
                            litert::HwAccelerators::kCpu));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto base_model, CompiledModel::Create(env, model_path.string(),
                                             litert::HwAccelerators::kCpu));

  const std::filesystem::path embedder_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/dummy_embedder.tflite";
  LITERT_ASSERT_OK_AND_ASSIGN(auto embedder_model,
                              Model::CreateFromFile(embedder_path.string()));
  ASSERT_OK_AND_ASSIGN(
      auto embedding_lookup,
      EmbeddingLookupManager::Create(env, &embedder_model,
                                     /*fully_supports_multi_modal=*/false));

  auto drafter_or = LlmLiteRtMtpDrafter::Create(
      env, std::move(compiled_mtp_model), executor_settings, base_model,
      mtp_model, *embedding_lookup,
      /*ple_manager=*/std::nullopt);
  EXPECT_THAT(drafter_or, StatusIs(absl::StatusCode::kNotFound));
}

TEST(LlmLiteRtMtpDrafterTest, UpdateCompilationOptions) {
  auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.litertlm";
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create(model_path.string()));
  ASSERT_OK_AND_ASSIGN(auto settings_gpu, LlmExecutorSettings::CreateDefault(
                                              model_assets, Backend::GPU));
  LITERT_ASSERT_OK_AND_ASSIGN(auto options, Options::Create());
  EXPECT_OK(UpdateCompilationOptions(settings_gpu, options));

  ASSERT_OK_AND_ASSIGN(auto settings_cpu, LlmExecutorSettings::CreateDefault(
                                              model_assets, Backend::CPU));
  EXPECT_OK(UpdateCompilationOptions(settings_cpu, options));
}

TEST(LlmLiteRtMtpDrafterTest, CountAcceptedPrefixAllAccepted) {
  const std::vector<int> drafted = {10, 20, 30};
  const std::vector<int> verifier_ids = {10, 20, 30, 99};
  ASSERT_EQ(verifier_ids.size(), drafted.size() + 1);
  const int accepted = CountAcceptedPrefix(verifier_ids, drafted);
  EXPECT_EQ(accepted, 3);
  EXPECT_EQ(verifier_ids[accepted], 99);
}

TEST(LlmLiteRtMtpDrafterTest, CountAcceptedPrefixNoneAccepted) {
  const std::vector<int> drafted = {10, 20, 30};
  const std::vector<int> verifier_ids = {11, 20, 30, 99};
  ASSERT_EQ(verifier_ids.size(), drafted.size() + 1);
  const int accepted = CountAcceptedPrefix(verifier_ids, drafted);
  EXPECT_EQ(accepted, 0);
  EXPECT_EQ(verifier_ids[accepted], 11);
}

TEST(LlmLiteRtMtpDrafterTest, CountAcceptedPrefixMismatchInMiddle) {
  const std::vector<int> drafted = {10, 20, 30};
  const std::vector<int> verifier_ids = {10, 25, 30, 99};
  ASSERT_EQ(verifier_ids.size(), drafted.size() + 1);
  const int accepted = CountAcceptedPrefix(verifier_ids, drafted);
  EXPECT_EQ(accepted, 1);
  EXPECT_EQ(verifier_ids[accepted], 25);
}

TEST(LlmLiteRtMtpDrafterTest, CountAcceptedPrefixEmptyDraftedWithBonusOnly) {
  const std::vector<int> drafted = {};
  const std::vector<int> verifier_ids = {42};
  ASSERT_EQ(verifier_ids.size(), drafted.size() + 1);
  const int accepted = CountAcceptedPrefix(verifier_ids, drafted);
  EXPECT_EQ(accepted, 0);
  EXPECT_EQ(verifier_ids[accepted], 42);
}

TEST(LlmLiteRtMtpDrafterTest,
     CountAcceptedPrefixDefensivelyClampsIfVerifierShorterThanDrafted) {
  // Callers enforce verifier_ids.size() == drafted.size() + 1 via RET_CHECK,
  // but CountAcceptedPrefix safely avoids out-of-bounds reads if misused.
  const std::vector<int> drafted = {10, 20, 30};
  const std::vector<int> short_verifier = {10, 20};
  EXPECT_EQ(CountAcceptedPrefix(short_verifier, drafted), 2);
}

// A toy first-order "language model" over a small vocabulary, used to check
// that exact-match verification with a *sampling* verifier and a *greedy*
// drafter emits tokens from the verifier's distribution (speculative sampling).
constexpr int kVocab = 5;
using Distribution = std::array<double, kVocab>;

// Target (verifier) next-token distributions, indexed by the previous token.
constexpr std::array<Distribution, kVocab> kTarget = {{
    {0.10, 0.40, 0.20, 0.20, 0.10},
    {0.30, 0.05, 0.25, 0.10, 0.30},
    {0.15, 0.15, 0.15, 0.40, 0.15},
    {0.50, 0.10, 0.10, 0.10, 0.20},
    {0.05, 0.20, 0.45, 0.10, 0.20},
}};

// Greedy drafter proposals, indexed by the previous token. They differ from
// the argmax of `kTarget` for most rows, so rejections are frequent.
constexpr std::array<int, kVocab> kDraft = {2, 0, 3, 4, 1};

enum class Verifier { kSampled, kGreedy };

// The target distributions of `kTarget` as samplers, built once.
std::array<std::discrete_distribution<int>, kVocab> TargetSamplers() {
  std::array<std::discrete_distribution<int>, kVocab> samplers;
  for (int i = 0; i < kVocab; ++i) {
    samplers[i] =
        std::discrete_distribution<int>(kTarget[i].begin(), kTarget[i].end());
  }
  return samplers;
}

int Argmax(const Distribution& p) {
  return static_cast<int>(
      std::distance(p.begin(), std::max_element(p.begin(), p.end())));
}

// Generates the first `num_tokens` tokens after `start` with option-B MTP
// rounds of `num_draft_steps` drafted tokens each. A `kSampled` verifier
// samples every position independently from the target distribution
// conditioned on the drafted prefix; a `kGreedy` verifier picks its argmax.
std::vector<int> GenerateWithMtpRounds(
    int start, int num_tokens, int num_draft_steps, Verifier verifier,
    std::array<std::discrete_distribution<int>, kVocab>& target_samplers,
    std::mt19937_64& rng) {
  std::vector<int> emitted;
  int last = start;
  while (static_cast<int>(emitted.size()) < num_tokens) {
    std::vector<int> drafted(num_draft_steps);
    for (int i = 0; i < num_draft_steps; ++i) {
      drafted[i] = kDraft[i == 0 ? last : drafted[i - 1]];
    }
    std::vector<int> verifier_ids(num_draft_steps + 1);
    for (int i = 0; i <= num_draft_steps; ++i) {
      const int previous = i == 0 ? last : drafted[i - 1];
      verifier_ids[i] = verifier == Verifier::kSampled
                            ? target_samplers[previous](rng)
                            : Argmax(kTarget[previous]);
    }
    const int accepted = CountAcceptedPrefix(verifier_ids, drafted);
    emitted.insert(emitted.end(), drafted.begin(), drafted.begin() + accepted);
    emitted.push_back(verifier_ids[accepted]);
    last = emitted.back();
  }
  emitted.resize(num_tokens);
  return emitted;
}

// Returns the counts of the first token emitted after `start` over `trials`
// generations.
std::vector<int64_t> CountFirstTokens(int start, Verifier verifier,
                                      int64_t trials, std::mt19937_64& rng) {
  auto target_samplers = TargetSamplers();
  std::vector<int64_t> counts(kVocab, 0);
  for (int64_t t = 0; t < trials; ++t) {
    ++counts[GenerateWithMtpRounds(start, /*num_tokens=*/1,
                                   /*num_draft_steps=*/3, verifier,
                                   target_samplers, rng)[0]];
  }
  return counts;
}

double ChiSquared(const std::vector<int64_t>& observed,
                  const std::vector<double>& expected_probs, int64_t trials) {
  double chi2 = 0.0;
  for (int i = 0; i < static_cast<int>(observed.size()); ++i) {
    const double expected = expected_probs[i] * trials;
    const double diff = observed[i] - expected;
    chi2 += diff * diff / expected;
  }
  return chi2;
}

TEST(LlmLiteRtMtpDrafterTest, SampledVerificationFirstTokenFollowsTarget) {
  constexpr int64_t kTrials = 200000;
  constexpr int kStart = 0;
  std::mt19937_64 rng(1234);
  const std::vector<int64_t> counts =
      CountFirstTokens(kStart, Verifier::kSampled, kTrials, rng);
  const std::vector<double> expected(kTarget[kStart].begin(),
                                     kTarget[kStart].end());
  // Critical value of the chi-squared distribution with 4 degrees of freedom
  // at p = 0.001.
  EXPECT_LT(ChiSquared(counts, expected, kTrials), 18.47);
}

TEST(LlmLiteRtMtpDrafterTest,
     SampledVerificationJointOfTwoTokensFollowsTarget) {
  // The second token is emitted either in the same round (after an accepted
  // draft) or in the next round (after a rejection); both paths must combine
  // into the target joint distribution p(x1 | start) * p(x2 | x1).
  constexpr int64_t kTrials = 200000;
  constexpr int kStart = 3;
  std::mt19937_64 rng(5678);
  auto target_samplers = TargetSamplers();
  std::vector<int64_t> counts(kVocab * kVocab, 0);
  for (int64_t t = 0; t < kTrials; ++t) {
    const std::vector<int> tokens =
        GenerateWithMtpRounds(kStart, /*num_tokens=*/2, /*num_draft_steps=*/3,
                              Verifier::kSampled, target_samplers, rng);
    ++counts[tokens[0] * kVocab + tokens[1]];
  }
  std::vector<double> expected(kVocab * kVocab);
  for (int a = 0; a < kVocab; ++a) {
    for (int b = 0; b < kVocab; ++b) {
      expected[a * kVocab + b] = kTarget[kStart][a] * kTarget[a][b];
    }
  }
  // Critical value of the chi-squared distribution with 24 degrees of freedom
  // at p = 0.001.
  EXPECT_LT(ChiSquared(counts, expected, kTrials), 51.18);
}

TEST(LlmLiteRtMtpDrafterTest, GreedyVerificationDoesNotFollowTarget) {
  // Control: with an argmax verifier the emitted token is deterministic, so the
  // chi-squared check of `SampledVerificationFirstTokenFollowsTarget` must
  // reject it.
  constexpr int64_t kTrials = 200000;
  constexpr int kStart = 0;
  std::mt19937_64 rng(1234);
  const std::vector<int64_t> counts =
      CountFirstTokens(kStart, Verifier::kGreedy, kTrials, rng);
  EXPECT_EQ(counts[Argmax(kTarget[kStart])], kTrials);
  const std::vector<double> expected(kTarget[kStart].begin(),
                                     kTarget[kStart].end());
  EXPECT_GT(ChiSquared(counts, expected, kTrials), 18.47);
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

TEST(LlmLiteRtMtpDrafterTest, LenienceRandomEngineRestartsWithVerifierStream) {
  auto first_gen = std::make_shared<std::default_random_engine>(7);
  LenienceRandomEngine lenience;
  lenience.Update(/*seed=*/7, first_gen);
  const uint64_t first_draw = lenience.engine()();
  const uint64_t second_draw = lenience.engine()();

  // The same seed and generator continue the stream.
  lenience.Update(/*seed=*/7, first_gen);
  const uint64_t third_draw = lenience.engine()();
  EXPECT_NE(third_draw, first_draw);

  // A new generator with the same seed (e.g. a new session) restarts it.
  auto second_gen = std::make_shared<std::default_random_engine>(7);
  lenience.Update(/*seed=*/7, second_gen);
  EXPECT_EQ(lenience.engine()(), first_draw);
  EXPECT_EQ(lenience.engine()(), second_draw);

  // So does a seed change, and going back to the old seed.
  lenience.Update(/*seed=*/8, second_gen);
  EXPECT_NE(lenience.engine()(), first_draw);
  lenience.Update(/*seed=*/7, second_gen);
  EXPECT_EQ(lenience.engine()(), first_draw);
}

TEST(LlmLiteRtMtpDrafterTest, LenienceRandomEngineDefaultsToSeedZero) {
  LenienceRandomEngine never_updated;
  LenienceRandomEngine seed_zero;
  seed_zero.Update(/*seed=*/0, std::make_shared<std::default_random_engine>(0));
  EXPECT_EQ(never_updated.engine()(), seed_zero.engine()());
}

// Lenient acceptance: r = (1 - c) * p + c * delta(draft).

TEST(LlmLiteRtMtpDrafterTest, LenienceZeroMatchesExactAcceptance) {
  std::mt19937_64 rng(42);
  std::uniform_int_distribution<int> token(0, 3);
  for (int trial = 0; trial < 1000; ++trial) {
    std::vector<int> drafted(4);
    std::vector<int> verifier_ids(5);
    for (int& t : drafted) t = token(rng);
    for (int& t : verifier_ids) t = token(rng);
    int uniform_calls = 0;
    const LenientAcceptResult result = CountAcceptedPrefixWithLenience(
        verifier_ids, drafted, /*lenience=*/0.0f, [&](int) {
          ++uniform_calls;
          return 0.0;
        });
    EXPECT_EQ(result.accepted, CountAcceptedPrefix(verifier_ids, drafted));
    EXPECT_EQ(result.lenient_accepted, 0);
    EXPECT_EQ(uniform_calls, 0);
  }
}

TEST(LlmLiteRtMtpDrafterTest, LenienceOneAcceptsEverything) {
  const std::vector<int> drafted = {10, 20, 30};
  const std::vector<int> verifier_ids = {11, 21, 30, 99};
  const LenientAcceptResult result = CountAcceptedPrefixWithLenience(
      verifier_ids, drafted, /*lenience=*/1.0f, [](int) { return 0.999999; });
  EXPECT_EQ(result.accepted, 3);
  EXPECT_EQ(result.lenient_accepted, 2);
  EXPECT_EQ(verifier_ids[result.accepted], 99);
}

TEST(LlmLiteRtMtpDrafterTest, LenienceUsesUniformOnlyOnMismatch) {
  const std::vector<int> drafted = {10, 20, 30};
  const std::vector<int> verifier_ids = {10, 25, 31, 99};
  std::vector<int> positions;
  const LenientAcceptResult result = CountAcceptedPrefixWithLenience(
      verifier_ids, drafted, /*lenience=*/0.5f, [&](int i) {
        positions.push_back(i);
        return i == 1 ? 0.1 : 0.9;  // Accept position 1, reject position 2.
      });
  EXPECT_EQ(result.accepted, 2);
  EXPECT_EQ(result.lenient_accepted, 1);
  EXPECT_EQ(verifier_ids[result.accepted], 31);
  EXPECT_EQ(positions, (std::vector<int>{1, 2}));
}

TEST(LlmLiteRtMtpDrafterTest, AcceptanceIsMonotoneInLenience) {
  // With shared randomness, raising c can only extend the accepted prefix.
  std::mt19937_64 rng(7);
  std::uniform_int_distribution<int> token(0, 2);
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  const std::vector<float> lenience = {0.0f, 0.1f, 0.25f, 0.5f, 0.75f, 1.0f};
  for (int trial = 0; trial < 2000; ++trial) {
    std::vector<int> drafted(5);
    std::vector<int> verifier_ids(6);
    std::vector<double> uniforms(5);
    for (int& t : drafted) t = token(rng);
    for (int& t : verifier_ids) t = token(rng);
    for (double& u : uniforms) u = unit(rng);
    int previous = -1;
    for (const float c : lenience) {
      const int accepted =
          CountAcceptedPrefixWithLenience(verifier_ids, drafted, c, [&](int i) {
            return uniforms[i];
          }).accepted;
      EXPECT_GE(accepted, previous);
      previous = accepted;
    }
  }
}

TEST(LlmLiteRtMtpDrafterTest, LenientAcceptanceSamplesFromMixture) {
  // The first emitted token must follow r = (1 - c) * p + c * delta(draft).
  constexpr int64_t kTrials = 200000;
  constexpr int kStart = 0;
  constexpr float kLenience = 0.3f;
  std::mt19937_64 rng(2468);
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  const Distribution& p = kTarget[kStart];
  std::discrete_distribution<int> verifier(p.begin(), p.end());
  const std::vector<int> drafted = {kDraft[kStart]};
  std::vector<int64_t> counts(kVocab, 0);
  for (int64_t t = 0; t < kTrials; ++t) {
    // The bonus position is irrelevant for the first token.
    const std::vector<int> verifier_ids = {verifier(rng), 0};
    const LenientAcceptResult result = CountAcceptedPrefixWithLenience(
        verifier_ids, drafted, kLenience, [&](int) { return unit(rng); });
    ++counts[result.accepted == 1 ? drafted[0] : verifier_ids[0]];
  }
  std::vector<double> expected(kVocab);
  for (int v = 0; v < kVocab; ++v) {
    expected[v] = (1.0 - kLenience) * p[v] + (v == drafted[0] ? kLenience : 0);
  }
  // Critical value of the chi-squared distribution with 4 degrees of freedom
  // at p = 0.001.
  EXPECT_LT(ChiSquared(counts, expected, kTrials), 18.47);
  // And it is distinguishable from the unmodified target.
  const std::vector<double> target(p.begin(), p.end());
  EXPECT_GT(ChiSquared(counts, target, kTrials), 18.47);
}

}  // namespace
}  // namespace litert::lm
