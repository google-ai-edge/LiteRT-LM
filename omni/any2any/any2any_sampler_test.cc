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

#include "omni/any2any/any2any_sampler.h"

#include <limits>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace litert::omni::any2any {
namespace {

using ::testing::Contains;

TEST(Any2AnySamplerTest, SelectGreedyHandlesEmptyTiesAndNonFiniteValues) {
  constexpr float kNan = std::numeric_limits<float>::quiet_NaN();
  constexpr float kInf = std::numeric_limits<float>::infinity();

  EXPECT_EQ(Any2AnySampler::SelectGreedy({}), 0);
  EXPECT_EQ(Any2AnySampler::SelectGreedy({kNan, -kInf, kNan}), 0);
  EXPECT_EQ(Any2AnySampler::SelectGreedy({kNan, -kInf, -1.5f, -3.0f}), 2);
  // Ties select the earliest index.
  EXPECT_EQ(Any2AnySampler::SelectGreedy({1.0f, 3.0f, 3.0f, 2.0f}), 1);
  // First `+Inf` wins immediately.
  EXPECT_EQ(Any2AnySampler::SelectGreedy({100.0f, kInf, kInf}), 1);
}

TEST(Any2AnySamplerTest, FallsBackToGreedyOnDisabledNonPositiveNanTempOrTopK1) {
  constexpr float kNan = std::numeric_limits<float>::quiet_NaN();
  constexpr float kInf = std::numeric_limits<float>::infinity();
  const std::vector<float> logits = {1.0f, 5.0f, 2.0f};

  {
    Any2AnySampler sampler({.do_sample = false, .temperature = 1.0f});
    EXPECT_EQ(sampler.Sample({}), 0);
    EXPECT_EQ(sampler.Sample(logits), 1);
  }
  {
    Any2AnySampler sampler({.do_sample = true, .temperature = 0.0f});
    EXPECT_EQ(sampler.Sample(logits), 1);
  }
  {
    Any2AnySampler sampler({.do_sample = true, .temperature = -1.0f});
    EXPECT_EQ(sampler.Sample(logits), 1);
  }
  {
    Any2AnySampler sampler({.do_sample = true, .temperature = kNan});
    for (int i = 0; i < 8; ++i) {
      EXPECT_EQ(sampler.Sample(logits), 1);
    }
    EXPECT_EQ(sampler.Sample({kNan, -kInf, -2.0f, -5.0f}), 2);
    EXPECT_EQ(sampler.Sample({10.0f, kInf, 20.0f}), 1);
  }
  {
    Any2AnySampler sampler(
        {.do_sample = true, .temperature = 1.0f, .top_k = 1});
    for (int i = 0; i < 8; ++i) {
      EXPECT_EQ(sampler.Sample(logits), 1);
    }
  }
}

TEST(Any2AnySamplerTest, TopKHeapSelectionTopPAndResetDeterminism) {
  constexpr float kNan = std::numeric_limits<float>::quiet_NaN();
  constexpr float kInf = std::numeric_limits<float>::infinity();

  Any2AnySampler sampler({
      .do_sample = true,
      .temperature = 1.0f,
      .top_k = 2,
      .top_p = 0.8f,
      .seed = 777,
  });

  EXPECT_EQ(sampler.Sample({kNan, kInf, 5.0f}), 1);
  // `+Inf` appearing after the top-k min-heap is filled still wins immediately.
  EXPECT_EQ(sampler.Sample({1.0f, 2.0f, 3.0f, kInf, 4.0f}), 3);
  EXPECT_EQ(sampler.Sample({kNan, -kInf}), 0);
  // Single finite entry among non-finite entries returns it directly.
  EXPECT_EQ(sampler.Sample({kNan, -kInf, 3.0f, kNan}), 2);

  // Bounded top-k=2 min-heap selects {1, 4} (equal top logits, discarding
  // index 5 with equal logit 2.0f in favor of smaller indices {1, 4}).
  const std::vector<float> logits = {kNan, 2.0f, -10.0f, 1.0f,
                                     2.0f, 2.0f, -kInf};
  std::vector<int> first_run;
  for (int i = 0; i < 24; ++i) {
    const int code = sampler.Sample(logits);
    EXPECT_TRUE(code == 1 || code == 4);
    first_run.push_back(code);
  }
  EXPECT_THAT(first_run, Contains(1));
  EXPECT_THAT(first_run, Contains(4));

  sampler.Reset();
  std::vector<int> second_run;
  for (int i = 0; i < 24; ++i) {
    second_run.push_back(sampler.Sample(logits));
  }
  EXPECT_EQ(first_run, second_run);

  // Full-vocab (`top_k = 0`) with `top_p = 0.5` keeps only the dominant token.
  Any2AnySampler top_p_sampler({
      .do_sample = true,
      .temperature = 1.0f,
      .top_k = 0,
      .top_p = 0.5f,
      .seed = 123,
  });
  const std::vector<float> skewed_logits = {1.0f, 0.0f, -10.0f};
  for (int i = 0; i < 32; ++i) {
    EXPECT_EQ(top_p_sampler.Sample(skewed_logits), 0);
  }

  // Full-vocab (`top_k = 0`) with `+Inf` temperature and extreme finite logits
  // stays finite and samples uniformly across both indices.
  Any2AnySampler inf_temp_sampler({
      .do_sample = true,
      .temperature = kInf,
      .top_k = 0,
      .top_p = 1.0f,
      .seed = 99,
  });
  constexpr float kMaxFloat = std::numeric_limits<float>::max();
  std::vector<int> extreme_draws;
  for (int i = 0; i < 24; ++i) {
    const int code = inf_temp_sampler.Sample({-kMaxFloat, kMaxFloat});
    EXPECT_TRUE(code == 0 || code == 1);
    extreme_draws.push_back(code);
  }
  EXPECT_THAT(extreme_draws, Contains(0));
  EXPECT_THAT(extreme_draws, Contains(1));
}

}  // namespace
}  // namespace litert::omni::any2any
