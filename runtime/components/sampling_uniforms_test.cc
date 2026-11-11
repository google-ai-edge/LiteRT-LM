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

#include "runtime/components/sampling_uniforms.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "runtime/proto/sampler_params.pb.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

using ::absl_testing::StatusIs;
using ::testing::Each;

proto::SamplerParameters ParamsWithK(int k) {
  proto::SamplerParameters params;
  params.set_type(proto::SamplerParameters::TOP_P);
  params.set_k(k);
  params.set_p(0.95f);
  params.set_temperature(1.0f);
  return params;
}

TEST(SamplingUniformsTest, DrawUniformsMatchesUniformRealDistribution) {
  std::default_random_engine rng(42);
  std::vector<float> uniforms(16);
  DrawUniforms(rng, absl::MakeSpan(uniforms));

  std::default_random_engine expected_rng(42);
  std::uniform_real_distribution<double> distribution(0.0, 1.0);
  for (float uniform : uniforms) {
    EXPECT_EQ(uniform, static_cast<float>(distribution(expected_rng)));
  }
  // Both engines are in the same state afterwards.
  EXPECT_EQ(rng, expected_rng);
}

TEST(SamplingUniformsTest, DrawSamplingUniformsDrawsForNonGreedySampling) {
  std::default_random_engine rng(3);
  std::vector<float> uniforms(4);
  DrawSamplingUniforms(ParamsWithK(40), rng, absl::MakeSpan(uniforms));

  std::default_random_engine expected_rng(3);
  std::vector<float> expected(4);
  DrawUniforms(expected_rng, absl::MakeSpan(expected));
  EXPECT_EQ(uniforms, expected);
  EXPECT_EQ(rng, expected_rng);
}

TEST(SamplingUniformsTest, DrawSamplingUniformsSkipsGreedySampling) {
  std::default_random_engine rng(3);
  std::vector<float> uniforms(4, 0.5f);
  DrawSamplingUniforms(ParamsWithK(1), rng, absl::MakeSpan(uniforms));
  EXPECT_THAT(uniforms, Each(0.0f));
  EXPECT_EQ(rng, std::default_random_engine(3));

  proto::SamplerParameters greedy_with_stale_k;
  greedy_with_stale_k.set_type(proto::SamplerParameters::GREEDY);
  greedy_with_stale_k.set_k(40);
  std::fill(uniforms.begin(), uniforms.end(), 0.5f);
  DrawSamplingUniforms(greedy_with_stale_k, rng, absl::MakeSpan(uniforms));
  EXPECT_THAT(uniforms, Each(0.0f));
  EXPECT_EQ(rng, std::default_random_engine(3));
}

TEST(SamplingUniformsTest, ValidateUniformsAcceptsValuesInRange) {
  const std::vector<float> uniforms = {0.0f, 0.5f, 0x1.fffffep-1f};
  EXPECT_OK(ValidateUniforms(uniforms, /*expected_count=*/3));
}

TEST(SamplingUniformsTest, ValidateUniformsRejectsWrongCount) {
  const std::vector<float> uniforms = {0.5f};
  EXPECT_THAT(ValidateUniforms(uniforms, /*expected_count=*/2),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(SamplingUniformsTest, ValidateUniformsRejectsValuesOutOfRange) {
  for (float uniform : {-0.1f, 1.0f, 1.5f, std::nanf("")}) {
    const std::vector<float> uniforms = {0.5f, uniform};
    EXPECT_THAT(ValidateUniforms(uniforms, /*expected_count=*/2),
                StatusIs(absl::StatusCode::kInvalidArgument))
        << "uniform=" << uniform;
  }
}

}  // namespace
}  // namespace litert::lm
