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
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/components/sampler.h"
#include "runtime/components/top_p_cpu_sampler.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/proto/sampler_params.pb.h"
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

TEST(SamplerFactoryTest, CreateSamplerForCpuSupportsGreedyAndTopK) {
  proto::SamplerParameters greedy_params;
  greedy_params.set_type(proto::SamplerParameters::GREEDY);
  ASSERT_OK_AND_ASSIGN(
      auto greedy_sampler,
      CreateSampler(Backend::CPU, /*batch_size=*/1, std::move(greedy_params)));
  EXPECT_NE(dynamic_cast<TopPSampler*>(greedy_sampler.get()), nullptr);

  proto::SamplerParameters top_k_params;
  top_k_params.set_type(proto::SamplerParameters::TOP_K);
  top_k_params.set_k(10);
  top_k_params.set_temperature(0.8f);
  ASSERT_OK_AND_ASSIGN(
      auto top_k_sampler,
      CreateSampler(Backend::CPU, /*batch_size=*/1, std::move(top_k_params)));
  EXPECT_NE(dynamic_cast<TopPSampler*>(top_k_sampler.get()), nullptr);

  proto::SamplerParameters invalid_top_k;
  invalid_top_k.set_type(proto::SamplerParameters::TOP_K);
  invalid_top_k.set_k(0);
  EXPECT_THAT(
      CreateSampler(Backend::CPU, /*batch_size=*/1, std::move(invalid_top_k)),
      StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(SamplerFactoryTest, CreateSamplerForCpuWithUnsupportedSamplerTypeFails) {
  proto::SamplerParameters sampler_params;
  sampler_params.set_k(1);
  sampler_params.set_p(0.0);
  sampler_params.set_temperature(1.0);
  sampler_params.set_seed(12345);
  sampler_params.set_type(static_cast<proto::SamplerParameters::Type>(99));
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
                                    /*batch_size=*/1,
                                    TopPParams(1, 0.0f, 1.0f)));
  EXPECT_NE(dynamic_cast<TopPSampler*>(sampler.get()), nullptr);
  EXPECT_EQ(max_top_k, 1);
}

TEST(SamplerFactoryTest, UpdateOrRecreateSamplerRecreatesOnlyWhenTopKGrows) {
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Sampler> sampler,
      CreateSampler(Backend::CPU, /*batch_size=*/1, TopPParams(1, 0.0f, 1.0f)));
  int max_top_k = 1;

  // A larger top-k recreates the sampler.
  const Sampler* original = sampler.get();
  ASSERT_OK(UpdateOrRecreateSampler(sampler, max_top_k, Backend::CPU,
                                    /*batch_size=*/1,
                                    TopPParams(40, 0.95f, 1.0f)));
  EXPECT_NE(sampler.get(), original);
  EXPECT_EQ(max_top_k, 40);

  // A smaller or equal top-k reconfigures the sampler in place.
  const Sampler* recreated = sampler.get();
  ASSERT_OK(UpdateOrRecreateSampler(sampler, max_top_k, Backend::CPU,
                                    /*batch_size=*/1,
                                    TopPParams(10, 0.9f, 0.7f)));
  EXPECT_EQ(sampler.get(), recreated);
  EXPECT_EQ(max_top_k, 40);

  // Switching to GREEDY (even with k = 0) normalizes to k = 1 and updates in
  // place without failing or shrinking max_top_k.
  proto::SamplerParameters greedy;
  greedy.set_type(proto::SamplerParameters::GREEDY);
  ASSERT_OK(UpdateOrRecreateSampler(sampler, max_top_k, Backend::CPU,
                                    /*batch_size=*/1, greedy));
  EXPECT_EQ(sampler.get(), recreated);
  EXPECT_EQ(max_top_k, 40);
}

TEST(SamplerFactoryTest, UpdateOrRecreateSamplerFailsWithoutSampler) {
  std::unique_ptr<Sampler> sampler;
  int max_top_k = 0;
  proto::SamplerParameters unspecified;
  unspecified.set_k(1);
  EXPECT_THAT(UpdateOrRecreateSampler(sampler, max_top_k, Backend::CPU,
                                      /*batch_size=*/1, unspecified),
              StatusIs(absl::StatusCode::kInternal));
  EXPECT_EQ(max_top_k, 0);
}

}  // namespace
}  // namespace litert::lm
