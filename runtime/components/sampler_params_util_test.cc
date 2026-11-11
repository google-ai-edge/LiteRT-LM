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

#include "runtime/components/sampler_params_util.h"

#include <limits>
#include <optional>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "runtime/proto/sampler_params.pb.h"

namespace litert::lm {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;

TEST(SamplerParamsUtilTest, NormalizeSamplerParamsLowersGreedyAndTopK) {
  proto::SamplerParameters greedy;
  greedy.set_type(proto::SamplerParameters::GREEDY);
  greedy.set_k(40);
  greedy.set_p(0.8f);
  greedy.set_temperature(0.7f);
  greedy.set_seed(99);
  greedy.set_backend(proto::SamplerParameters::CPU);
  greedy.set_compute_exact_log_probs(true);
  const proto::SamplerParameters normalized_greedy =
      NormalizeSamplerParams(greedy);
  EXPECT_EQ(normalized_greedy.type(), proto::SamplerParameters::TOP_P);
  EXPECT_EQ(normalized_greedy.k(), 1);
  EXPECT_FLOAT_EQ(normalized_greedy.p(), 0.0f);
  EXPECT_FLOAT_EQ(normalized_greedy.temperature(), 1.0f);
  EXPECT_EQ(normalized_greedy.seed(), 99);
  EXPECT_EQ(normalized_greedy.backend(), proto::SamplerParameters::CPU);
  EXPECT_TRUE(normalized_greedy.compute_exact_log_probs());

  proto::SamplerParameters top_k;
  top_k.set_type(proto::SamplerParameters::TOP_K);
  top_k.set_k(32);
  top_k.set_p(0.0f);
  top_k.set_temperature(0.6f);
  top_k.set_seed(7);
  top_k.set_backend(proto::SamplerParameters::GPU);
  top_k.set_compute_exact_log_probs(true);
  const proto::SamplerParameters normalized_top_k =
      NormalizeSamplerParams(top_k);
  EXPECT_EQ(normalized_top_k.type(), proto::SamplerParameters::TOP_P);
  EXPECT_EQ(normalized_top_k.k(), 32);
  EXPECT_FLOAT_EQ(normalized_top_k.p(), 1.0f);
  EXPECT_FLOAT_EQ(normalized_top_k.temperature(), 0.6f);
  EXPECT_EQ(normalized_top_k.seed(), 7);
  EXPECT_EQ(normalized_top_k.backend(), proto::SamplerParameters::GPU);
  EXPECT_TRUE(normalized_top_k.compute_exact_log_probs());

  proto::SamplerParameters unspecified;
  unspecified.set_type(proto::SamplerParameters::TYPE_UNSPECIFIED);
  EXPECT_EQ(NormalizeSamplerParams(unspecified).type(),
            proto::SamplerParameters::TYPE_UNSPECIFIED);
}

TEST(SamplerParamsUtilTest, GreedySamplerParamsAreGreedyTopP) {
  const proto::SamplerParameters params = GreedySamplerParams();
  EXPECT_EQ(params.type(), proto::SamplerParameters::TOP_P);
  EXPECT_EQ(params.k(), 1);
  EXPECT_EQ(params.p(), 0.0f);
  EXPECT_EQ(params.temperature(), 1.0f);
  EXPECT_EQ(params.seed(), 0);
}

TEST(SamplerParamsUtilTest, ValidateTopPSamplingParamsAcceptsValidParams) {
  EXPECT_THAT(ValidateTopPSamplingParams(proto::SamplerParameters::TOP_P,
                                         /*k=*/1, /*p=*/0.0f,
                                         /*temperature=*/0.0f),
              IsOk());
  EXPECT_THAT(ValidateTopPSamplingParams(proto::SamplerParameters::TOP_P,
                                         /*k=*/40, /*p=*/1.0f,
                                         /*temperature=*/1.5f),
              IsOk());
  EXPECT_THAT(ValidateTopPSamplingParams(
                  proto::SamplerParameters::TYPE_UNSPECIFIED, /*k=*/40,
                  /*p=*/0.95f, /*temperature=*/1.0f),
              IsOk());
}

TEST(SamplerParamsUtilTest, ValidateTopPSamplingParamsRejectsOtherTypes) {
  // GREEDY and TOP_K must be normalized first.
  EXPECT_THAT(ValidateTopPSamplingParams(proto::SamplerParameters::GREEDY,
                                         /*k=*/1, /*p=*/0.0f,
                                         /*temperature=*/1.0f),
              StatusIs(absl::StatusCode::kUnimplemented));
  EXPECT_THAT(ValidateTopPSamplingParams(proto::SamplerParameters::TOP_K,
                                         /*k=*/40, /*p=*/1.0f,
                                         /*temperature=*/1.0f),
              StatusIs(absl::StatusCode::kUnimplemented));
}

TEST(SamplerParamsUtilTest, ValidateTopPSamplingParamsRejectsOutOfBounds) {
  constexpr auto kTopP = proto::SamplerParameters::TOP_P;
  constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
  constexpr float kInf = std::numeric_limits<float>::infinity();
  EXPECT_THAT(ValidateTopPSamplingParams(kTopP, /*k=*/0, /*p=*/0.5f,
                                         /*temperature=*/1.0f),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(ValidateTopPSamplingParams(kTopP, /*k=*/-1, /*p=*/0.5f,
                                         /*temperature=*/1.0f),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(ValidateTopPSamplingParams(kTopP, /*k=*/1, /*p=*/-0.1f,
                                         /*temperature=*/1.0f),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(ValidateTopPSamplingParams(kTopP, /*k=*/1, /*p=*/1.1f,
                                         /*temperature=*/1.0f),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(ValidateTopPSamplingParams(kTopP, /*k=*/1, /*p=*/kNaN,
                                         /*temperature=*/1.0f),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(ValidateTopPSamplingParams(kTopP, /*k=*/1, /*p=*/0.5f,
                                         /*temperature=*/-0.1f),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(ValidateTopPSamplingParams(kTopP, /*k=*/1, /*p=*/0.5f,
                                         /*temperature=*/kNaN),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(ValidateTopPSamplingParams(kTopP, /*k=*/1, /*p=*/0.5f,
                                         /*temperature=*/kInf),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(SamplerParamsUtilTest, SamplerParamsEquivalentTreatsUnspecifiedAsDefault) {
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

TEST(SamplerParamsUtilTest, SamplerParamsEquivalentComparesEveryField) {
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

TEST(SamplerParamsUtilTest, SamplerParamsEquivalentCanIgnoreSeed) {
  proto::SamplerParameters base;
  base.set_type(proto::SamplerParameters::TOP_P);
  base.set_k(40);
  base.set_p(0.95f);
  base.set_temperature(1.0f);
  base.set_seed(7);

  proto::SamplerParameters changed = base;
  changed.set_seed(8);
  EXPECT_TRUE(SamplerParamsEquivalent(changed, base, /*ignore_seed=*/true));
  changed.clear_seed();
  EXPECT_TRUE(SamplerParamsEquivalent(changed, base, /*ignore_seed=*/true));

  // The other fields are still compared.
  changed.set_k(10);
  EXPECT_FALSE(SamplerParamsEquivalent(changed, base, /*ignore_seed=*/true));
}

TEST(SamplerParamsUtilTest, SamplerParamsEquivalentNormalizesGreedyAndTopK) {
  // GREEDY with any stale k/p/temperature normalizes to GreedySamplerParams().
  proto::SamplerParameters greedy;
  greedy.set_type(proto::SamplerParameters::GREEDY);
  greedy.set_k(40);
  greedy.set_p(0.8f);
  greedy.set_temperature(0.5f);
  EXPECT_TRUE(SamplerParamsEquivalent(greedy, GreedySamplerParams()));

  proto::SamplerParameters default_greedy;
  default_greedy.set_type(proto::SamplerParameters::GREEDY);
  EXPECT_TRUE(SamplerParamsEquivalent(greedy, default_greedy));

  // TOP_K ignores p and is equivalent to TOP_P with p = 1.0f.
  proto::SamplerParameters top_k_a;
  top_k_a.set_type(proto::SamplerParameters::TOP_K);
  top_k_a.set_k(10);
  top_k_a.set_p(0.2f);
  top_k_a.set_temperature(0.8f);

  proto::SamplerParameters top_k_b = top_k_a;
  top_k_b.set_p(0.9f);
  EXPECT_TRUE(SamplerParamsEquivalent(top_k_a, top_k_b));

  proto::SamplerParameters equiv_top_p;
  equiv_top_p.set_type(proto::SamplerParameters::TOP_P);
  equiv_top_p.set_k(10);
  equiv_top_p.set_p(1.0f);
  equiv_top_p.set_temperature(0.8f);
  EXPECT_TRUE(SamplerParamsEquivalent(top_k_a, equiv_top_p));
}

}  // namespace
}  // namespace litert::lm
