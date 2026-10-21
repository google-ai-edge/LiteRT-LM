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

#include "runtime/components/sampler_factory.h"

#include <memory>
#include <optional>
#include <random>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "runtime/components/sampler.h"
#include "runtime/components/top_p_cpu_sampler.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/proto/sampler_params.pb.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

using ::testing::HasSubstr;
using ::testing::status::StatusIs;

TEST(SamplerFactoryTest, CreateSamplerForCpuWorksCorrectly) {
  proto::SamplerParameters sampler_params;
  sampler_params.set_k(1);
  sampler_params.set_p(0.0);
  sampler_params.set_temperature(1.0);
  sampler_params.set_seed(12345);
  sampler_params.set_type(proto::SamplerParameters::TOP_P);
  ASSERT_OK_AND_ASSIGN(
      auto sampler, CreateSampler(Backend::CPU,
                                  /*batch_size=*/1, std::move(sampler_params)));
  EXPECT_NE(sampler, nullptr);
  // Make sure the factory creates the correct sampler.
  TopPSampler* top_p_sampler = dynamic_cast<TopPSampler*>(sampler.get());
  EXPECT_NE(top_p_sampler, nullptr);
  EXPECT_FALSE(top_p_sampler->ComputeExactLogProbs());
}

TEST(SamplerFactoryTest, CreateSamplerForCpuWithComputeExactLogProbs) {
  proto::SamplerParameters sampler_params;
  sampler_params.set_k(1);
  sampler_params.set_p(0.0);
  sampler_params.set_temperature(1.0);
  sampler_params.set_seed(12345);
  sampler_params.set_type(proto::SamplerParameters::TOP_P);
  sampler_params.set_compute_exact_log_probs(true);
  ASSERT_OK_AND_ASSIGN(
      auto sampler, CreateSampler(Backend::CPU,
                                  /*batch_size=*/1, std::move(sampler_params)));
  EXPECT_NE(sampler, nullptr);
  TopPSampler* top_p_sampler = dynamic_cast<TopPSampler*>(sampler.get());
  ASSERT_NE(top_p_sampler, nullptr);
  EXPECT_TRUE(top_p_sampler->ComputeExactLogProbs());
}

TEST(SamplerFactoryTest, CreateSamplerForCpuWithUnsupportedSamplerTypeFails) {
  proto::SamplerParameters sampler_params;
  sampler_params.set_k(1);
  sampler_params.set_p(0.0);
  sampler_params.set_temperature(1.0);
  sampler_params.set_seed(12345);
  sampler_params.set_type(proto::SamplerParameters::TOP_K);
  auto result = CreateSampler(Backend::CPU,
                              /*batch_size=*/1, std::move(sampler_params));
  EXPECT_THAT(result, StatusIs(absl::StatusCode::kUnimplemented));
  EXPECT_THAT(result.status().message(), HasSubstr("not implemented yet."));
}

TEST(SamplerFactoryTest,
     CreateSamplerForCpuWithUnspecifiedSamplerTypeReturnsNullptr) {
  proto::SamplerParameters sampler_params;
  sampler_params.set_k(1);
  sampler_params.set_p(0.0);
  sampler_params.set_temperature(1.0);
  sampler_params.set_seed(12345);
  sampler_params.set_type(proto::SamplerParameters::TYPE_UNSPECIFIED);
  ASSERT_OK_AND_ASSIGN(
      auto sampler,
      CreateSampler(Backend::CPU, /*batch_size=*/1, std::move(sampler_params)));
  EXPECT_EQ(sampler, nullptr);
}

proto::SamplerParameters TopPParams(int k, float p, float temperature) {
  proto::SamplerParameters sampler_params;
  sampler_params.set_type(proto::SamplerParameters::TOP_P);
  sampler_params.set_k(k);
  sampler_params.set_p(p);
  sampler_params.set_temperature(temperature);
  sampler_params.set_seed(0);
  return sampler_params;
}

TEST(SamplerFactoryTest, UpdateOrRecreateSamplerCreatesMissingSampler) {
  std::unique_ptr<Sampler> sampler;
  int max_top_k = 0;
  ASSERT_OK(UpdateOrRecreateSampler(sampler, max_top_k, Backend::CPU,
                                    /*batch_size=*/1, TopPParams(1, 0.0f, 1.0f),
                                    /*rand_gen=*/nullptr));
  EXPECT_NE(dynamic_cast<TopPSampler*>(sampler.get()), nullptr);
  EXPECT_EQ(max_top_k, 1);
}

TEST(SamplerFactoryTest, UpdateOrRecreateSamplerRecreatesOnlyWhenTopKGrows) {
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Sampler> sampler,
      CreateSampler(Backend::CPU, /*batch_size=*/1, TopPParams(1, 0.0f, 1.0f)));
  int max_top_k = 1;
  auto rand_gen = std::make_shared<std::default_random_engine>(42);

  // A larger top-k recreates the sampler.
  const Sampler* original = sampler.get();
  ASSERT_OK(UpdateOrRecreateSampler(sampler, max_top_k, Backend::CPU,
                                    /*batch_size=*/1,
                                    TopPParams(40, 0.95f, 1.0f), rand_gen));
  EXPECT_NE(sampler.get(), original);
  EXPECT_EQ(max_top_k, 40);

  // A smaller or equal top-k reconfigures the sampler in place.
  const Sampler* recreated = sampler.get();
  ASSERT_OK(UpdateOrRecreateSampler(sampler, max_top_k, Backend::CPU,
                                    /*batch_size=*/1,
                                    TopPParams(10, 0.9f, 0.7f), rand_gen));
  EXPECT_EQ(sampler.get(), recreated);
  EXPECT_EQ(max_top_k, 40);
}

absl::StatusOr<int> SampleOne(Sampler& sampler,
                              const std::vector<float>& logits) {
  auto logits_tensor =
      CopyToTensorBuffer<float>(logits, {1, static_cast<int>(logits.size())});
  if (!logits_tensor) {
    return absl::InternalError("Failed to create the logits tensor.");
  }
  std::vector<int> ids_vector(1);
  auto ids_tensor =
      CopyToTensorBuffer<int>(absl::MakeConstSpan(ids_vector), {1});
  if (!ids_tensor) {
    return absl::InternalError("Failed to create the ids tensor.");
  }
  ABSL_RETURN_IF_ERROR(sampler.SampleToIdAndScoreBuffer(
      *logits_tensor, *ids_tensor, /*scores_tensor=*/nullptr));
  auto ids = CopyFromTensorBuffer<int>(*ids_tensor);
  if (!ids) {
    return absl::InternalError("Failed to read the ids tensor.");
  }
  return (*ids)[0];
}

TEST(SamplerFactoryTest, UpdateOrRecreateSamplerSharesRandomGenerator) {
  // Samplers `a` and `b` share one generator, so alternating between them
  // yields the same draws as a single sampler `c` with its own generator
  // seeded identically.
  const auto params = TopPParams(8, 1.0f, 1.0f);
  auto shared = std::make_shared<std::default_random_engine>(7);
  auto reference = std::make_shared<std::default_random_engine>(7);
  std::unique_ptr<Sampler> a;
  std::unique_ptr<Sampler> b;
  std::unique_ptr<Sampler> c;
  int max_top_k_a = 0;
  int max_top_k_b = 0;
  int max_top_k_c = 0;
  ASSERT_OK(UpdateOrRecreateSampler(a, max_top_k_a, Backend::CPU,
                                    /*batch_size=*/1, params, shared));
  ASSERT_OK(UpdateOrRecreateSampler(b, max_top_k_b, Backend::CPU,
                                    /*batch_size=*/1, params, shared));
  ASSERT_OK(UpdateOrRecreateSampler(c, max_top_k_c, Backend::CPU,
                                    /*batch_size=*/1, params, reference));

  const std::vector<float> logits(8, 0.0f);  // Uniform over 8 tokens.
  std::vector<int> alternating;
  std::vector<int> single;
  for (int i = 0; i < 64; ++i) {
    ASSERT_OK_AND_ASSIGN(int id, SampleOne(i % 2 == 0 ? *a : *b, logits));
    alternating.push_back(id);
    ASSERT_OK_AND_ASSIGN(int ref_id, SampleOne(*c, logits));
    single.push_back(ref_id);
  }
  EXPECT_EQ(alternating, single);
}

TEST(SamplerFactoryTest, UpdateOrRecreateSamplerFailsWithoutSampler) {
  std::unique_ptr<Sampler> sampler;
  int max_top_k = 0;
  proto::SamplerParameters unspecified;
  unspecified.set_k(1);
  EXPECT_THAT(UpdateOrRecreateSampler(sampler, max_top_k, Backend::CPU,
                                      /*batch_size=*/1, unspecified,
                                      /*rand_gen=*/nullptr),
              StatusIs(absl::StatusCode::kInternal));
  EXPECT_EQ(max_top_k, 0);
}

TEST(SamplerFactoryTest, GreedySamplerParamsAreGreedyTopP) {
  const proto::SamplerParameters params = GreedySamplerParams();
  EXPECT_EQ(params.type(), proto::SamplerParameters::TOP_P);
  EXPECT_EQ(params.k(), 1);
  EXPECT_EQ(params.temperature(), 1.0f);
  EXPECT_EQ(params.seed(), 0);
}

TEST(SamplerFactoryTest, SamplerParamsEquivalentTreatsUnspecifiedAsDefault) {
  // Unspecified parameters select the default sampler like unset ones, whatever
  // the other fields hold.
  proto::SamplerParameters unspecified;
  unspecified.set_k(40);
  unspecified.set_temperature(0.7f);
  EXPECT_TRUE(SamplerParamsEquivalent(std::nullopt, std::nullopt));
  EXPECT_TRUE(SamplerParamsEquivalent(std::nullopt, unspecified));
  EXPECT_TRUE(SamplerParamsEquivalent(unspecified, std::nullopt));
  EXPECT_TRUE(SamplerParamsEquivalent(unspecified, proto::SamplerParameters()));

  EXPECT_FALSE(SamplerParamsEquivalent(std::nullopt, GreedySamplerParams()));
  EXPECT_FALSE(SamplerParamsEquivalent(GreedySamplerParams(), unspecified));
}

TEST(SamplerFactoryTest, SamplerParamsEquivalentComparesEveryField) {
  proto::SamplerParameters base;
  base.set_type(proto::SamplerParameters::TOP_P);
  base.set_k(40);
  base.set_p(0.95f);
  base.set_temperature(1.0f);
  base.set_seed(7);
  EXPECT_TRUE(SamplerParamsEquivalent(base, base));

  proto::SamplerParameters unset_seed = base;
  unset_seed.clear_seed();
  proto::SamplerParameters zero_seed = base;
  zero_seed.set_seed(0);
  EXPECT_TRUE(SamplerParamsEquivalent(unset_seed, zero_seed));

  proto::SamplerParameters changed = base;
  changed.set_type(proto::SamplerParameters::TOP_K);
  EXPECT_FALSE(SamplerParamsEquivalent(changed, base));
  changed = base;
  changed.set_k(10);
  EXPECT_FALSE(SamplerParamsEquivalent(changed, base));
  changed = base;
  changed.set_p(0.5f);
  EXPECT_FALSE(SamplerParamsEquivalent(changed, base));
  changed = base;
  changed.set_temperature(0.7f);
  EXPECT_FALSE(SamplerParamsEquivalent(changed, base));
  changed = base;
  changed.set_seed(8);
  EXPECT_FALSE(SamplerParamsEquivalent(changed, base));
  changed = base;
  changed.set_backend(proto::SamplerParameters::CPU);
  EXPECT_FALSE(SamplerParamsEquivalent(changed, base));
  changed = base;
  changed.set_compute_exact_log_probs(true);
  EXPECT_FALSE(SamplerParamsEquivalent(changed, base));
}

}  // namespace
}  // namespace litert::lm
