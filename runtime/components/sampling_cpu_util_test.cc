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

#include "runtime/components/sampling_cpu_util.h"

#include <cstdio>
#include <memory>
#include <random>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl

namespace litert::lm {
namespace {

using ::testing::ElementsAre;
using ::testing::UnorderedElementsAre;

TEST(SamplingCpuUtilTest, TopKTokenIds_BatchSize1) {
  const std::vector<float> logits = {0.1, 0.5, 0.4, 0.2};
  auto topk_token_ids = TopKTokenIds(absl::MakeConstSpan(logits), /*k=*/2,
                                     /*batch_size=*/1, /*sequence_size=*/1);
  ASSERT_TRUE(topk_token_ids.ok());
  EXPECT_EQ(topk_token_ids->size(), 1);
  EXPECT_THAT((*topk_token_ids)[0], UnorderedElementsAre(1, 2));
}

TEST(SamplingCpuUtilTest, TopKTokenIds_BatchSize2) {
  const std::vector<float> logits = {0.1, 0.5, 0.4, 0.2};
  auto topk_token_ids = TopKTokenIds(absl::MakeConstSpan(logits), /*k=*/1,
                                     /*batch_size=*/2, /*sequence_size=*/1);
  ASSERT_TRUE(topk_token_ids.ok());
  EXPECT_EQ(topk_token_ids->size(), 2);
  EXPECT_THAT((*topk_token_ids)[0], ElementsAre(1));
  EXPECT_THAT((*topk_token_ids)[1], ElementsAre(0));
}

TEST(SamplingCpuUtilTest, TopKTokenIds_SequenceLength2) {
  const std::vector<float> logits = {0.1, 0.5, 0.4, 0.2, 0.6, 0.3};
  auto topk_token_ids = TopKTokenIds(absl::MakeConstSpan(logits), /*k=*/1,
                                     /*batch_size=*/1,
                                     /*sequence_size=*/2);
  ASSERT_TRUE(topk_token_ids.ok());
  EXPECT_EQ(topk_token_ids->size(), 1);
  EXPECT_THAT((*topk_token_ids)[0], ElementsAre(1, 1));
}

TEST(SamplingCpuUtilTest, Softmax_BatchSize1) {
  const std::vector<float> logits = {0.1f, 0.1f};
  const std::vector<int> topk_indices = {0, 1};
  std::vector<std::vector<float>> max_logit_values;
  auto probabilities =
      Softmax(absl::MakeConstSpan(logits), absl::MakeConstSpan(topk_indices),
              /*temperature=*/1.0, /*batch_size=*/1, /*sequence_size=*/1,
              max_logit_values);
  ASSERT_TRUE(probabilities.ok());
  EXPECT_EQ(probabilities->size(), 1);
  EXPECT_THAT((*probabilities)[0], ElementsAre(0.5, 0.5));
  EXPECT_EQ(max_logit_values.size(), 1);
  EXPECT_THAT(max_logit_values[0], ElementsAre(0.1f));
}

TEST(SamplingCpuUtilTest, Softmax_AllZeroLogits) {
  const std::vector<float> logits = {0.0f, 0.0f};
  const std::vector<int> topk_indices = {0, 1};
  std::vector<std::vector<float>> max_logit_values;
  auto probabilities =
      Softmax(absl::MakeConstSpan(logits), absl::MakeConstSpan(topk_indices),
              /*temperature=*/1.0, /*batch_size=*/1, /*sequence_size=*/1,
              max_logit_values);
  ASSERT_TRUE(probabilities.ok());
  EXPECT_EQ(probabilities->size(), 1);
  EXPECT_THAT((*probabilities)[0], ElementsAre(0.5, 0.5));
  EXPECT_EQ(max_logit_values.size(), 1);
  EXPECT_THAT(max_logit_values[0], ElementsAre(0.0f));
}

TEST(SamplingCpuUtilTest, Softmax_TemperatureVerySmall) {
  const std::vector<float> logits = {0.0f, 1.0f, 2.0f};
  const std::vector<int> topk_indices = {0, 1, 2};
  std::vector<std::vector<float>> max_logit_values;
  auto probabilities =
      Softmax(absl::MakeConstSpan(logits), absl::MakeConstSpan(topk_indices),
              /*temperature=*/0.00000001f, /*batch_size=*/1,
              /*sequence_size=*/1, max_logit_values);
  ASSERT_TRUE(probabilities.ok());
  // Very small temperature should mimic greedy sampling.
  EXPECT_EQ(probabilities->size(), 1);
  EXPECT_THAT((*probabilities)[0], ElementsAre(0.0f, 0.0f, 1.0f));
  EXPECT_EQ(max_logit_values.size(), 1);
  EXPECT_THAT(max_logit_values[0], ElementsAre(2.0f));
}

TEST(SamplingCpuUtilTest, Softmax_TemperatureExactlyZero) {
  const std::vector<float> logits = {0.0f, 1.0f, 2.0f};
  const std::vector<int> topk_indices = {0, 1, 2};
  std::vector<std::vector<float>> max_logit_values;
  auto probabilities =
      Softmax(absl::MakeConstSpan(logits), absl::MakeConstSpan(topk_indices),
              /*temperature=*/0.0f, /*batch_size=*/1, /*sequence_size=*/1,
              max_logit_values);
  ASSERT_TRUE(probabilities.ok());
  // Exactly zero temperature should mimic greedy sampling.
  EXPECT_EQ(probabilities->size(), 1);
  EXPECT_THAT((*probabilities)[0], ElementsAre(0.0f, 0.0f, 1.0f));
  EXPECT_EQ(max_logit_values.size(), 1);
  EXPECT_THAT(max_logit_values[0], ElementsAre(2.0f));
}

TEST(SamplingCpuUtilTest, Softmax_TemperatureInf) {
  const std::vector<float> logits = {0.0f, 1.0f, 2.0f, 3.0f};
  const std::vector<int> topk_indices = {0, 1, 2, 3};
  std::vector<std::vector<float>> max_logit_values;
  auto probabilities =
      Softmax(absl::MakeConstSpan(logits), absl::MakeConstSpan(topk_indices),
              /*temperature=*/100000000000.0f, /*batch_size=*/1,
              /*sequence_size=*/1, max_logit_values);
  ASSERT_TRUE(probabilities.ok());
  // Very large temperature should mimic uniform sampling.
  EXPECT_EQ(probabilities->size(), 1);
  EXPECT_THAT((*probabilities)[0], ElementsAre(0.25f, 0.25f, 0.25f, 0.25f));
  EXPECT_EQ(max_logit_values.size(), 1);
  EXPECT_THAT(max_logit_values[0], ElementsAre(3.0f));
}

TEST(SamplingCpuUtilTest, Softmax_BatchSize3) {
  // Batch size of 3, vocab size of 2.
  const std::vector<float> logits = {0.1f, 0.1f, 0.0f, 5.0f, 1.0f, 0.0f};
  absl::Span<const float> logits_span = absl::MakeConstSpan(logits);
  const std::vector<int> topk_indices = {0, 1, 0, 1, 0, 1};
  absl::Span<const int> topk_indices_span = absl::MakeConstSpan(topk_indices);
  std::vector<std::vector<float>> max_logit_values;
  auto probabilities = Softmax(logits_span, topk_indices_span,
                               /*temperature=*/1.0f, /*batch_size=*/3,
                               /*sequence_size=*/1, max_logit_values);
  ASSERT_TRUE(probabilities.ok());
  EXPECT_EQ(probabilities->size(), 3);
  EXPECT_EQ((*probabilities)[0].size(), 2);
  EXPECT_THAT((*probabilities)[0], ElementsAre(0.5f, 0.5f));
  EXPECT_EQ((*probabilities)[1].size(), 2);
  EXPECT_THAT((*probabilities)[1], ElementsAre(0.00669285096f, 0.993307173f));
  EXPECT_EQ((*probabilities)[2].size(), 2);
  EXPECT_THAT((*probabilities)[2], ElementsAre(0.731058598f, 0.268941432f));
  EXPECT_EQ(max_logit_values.size(), 3);
  EXPECT_EQ(max_logit_values[0].size(), 1);
  EXPECT_THAT(max_logit_values[0], ElementsAre(0.1f));
  EXPECT_EQ(max_logit_values[1].size(), 1);
  EXPECT_THAT(max_logit_values[1], ElementsAre(5.0f));
  EXPECT_EQ(max_logit_values[2].size(), 1);
  EXPECT_THAT(max_logit_values[2], ElementsAre(1.0f));
}

TEST(SamplingCpuUtilTest, TopKTopPSampling_InvalidInputs) {
  const std::vector<float> probabilities = {0.0, 0.0, 0.3};
  auto rng = std::make_shared<std::default_random_engine>(0);
  // Negative k.
  std::vector<std::vector<float>> sampled_scores;
  auto sampled_ids =
      TopKTopPSampling(absl::MakeConstSpan(probabilities), /*k=*/-1,
                       /*p=*/0.5,
                       /*temperature=*/1.0, rng, /*batch_size=*/1,
                       /*sequence_size=*/1, sampled_scores);
  EXPECT_FALSE(sampled_ids.ok());
  // Negative p.
  sampled_ids = TopKTopPSampling(absl::MakeConstSpan(probabilities),
                                 /*k=*/1,
                                 /*p=*/-0.5, /*temperature=*/1.0f, rng,
                                 /*sequence_size=*/1,
                                 /*batch_size=*/1, sampled_scores);
  EXPECT_FALSE(sampled_ids.ok());
}

TEST(SamplingCpuUtilTest, TopKTopPSampling_BatchSize1) {
  const std::vector<float> probabilities = {0.0, 0.0, 0.3};
  auto rng = std::make_shared<std::default_random_engine>(0);
  std::vector<std::vector<float>> sampled_scores;
  auto sampled_ids =
      TopKTopPSampling(absl::MakeConstSpan(probabilities), /*k=*/1,
                       /*p=*/0.5,
                       /*temperature=*/1.0f, rng, /*batch_size=*/1,
                       /*sequence_size=*/1, sampled_scores);
  ASSERT_TRUE(sampled_ids.ok());
  EXPECT_EQ(sampled_ids->size(), 1);
  EXPECT_THAT((*sampled_ids)[0], ElementsAre(2));
  EXPECT_EQ(sampled_scores.size(), 1);
  EXPECT_THAT(sampled_scores[0], ElementsAre(1.0));
}

TEST(SamplingCpuUtilTest, TopKTopPSampling_BatchSize1_TopK) {
  // Test that the sampler does return a sampled token from the top k
  // instead of always returning the first or the last token.
  const std::vector<float> logits = {-1.0e7f, 1.0f, -1e3f};
  auto rng = std::make_shared<std::default_random_engine>(0);
  std::vector<std::vector<float>> sampled_scores;
  auto sampled_ids =
      TopKTopPSampling(absl::MakeConstSpan(logits), /*k=*/3,
                       /*p=*/1.0,
                       /*temperature=*/1.0f, rng,
                       /*batch_size=*/1, /*sequence_size=*/1, sampled_scores);
  ASSERT_TRUE(sampled_ids.ok());
  EXPECT_EQ(sampled_ids->size(), 1);
  EXPECT_THAT((*sampled_ids)[0], ElementsAre(1));
  EXPECT_EQ(sampled_scores.size(), 1);
  EXPECT_THAT(sampled_scores[0], ElementsAre(1.0));
}

TEST(SamplingCpuUtilTest, TopKTopPSampling_BatchSize3) {
  // Batch of 3, vocab size of 3. The sampled ids are 2, 1, 0.
  const std::vector<float> logits = {0.0, 0.0, 1.0, 0.0, 1.0,
                                     0.0, 1.0, 0.0, 0.0};
  auto rng = std::make_shared<std::default_random_engine>(0);
  std::vector<std::vector<float>> sampled_scores;
  auto sampled_ids =
      TopKTopPSampling(absl::MakeConstSpan(logits), /*k=*/2, /*p=*/0.5,
                       /*temperature=*/0.00001f, rng, /*batch_size=*/3,
                       /*sequence_size=*/1, sampled_scores);
  ASSERT_TRUE(sampled_ids.ok());
  EXPECT_EQ(sampled_ids->size(), 3);
  EXPECT_EQ((*sampled_ids)[0][0], 2);
  EXPECT_EQ((*sampled_ids)[1][0], 1);
  EXPECT_EQ((*sampled_ids)[2][0], 0);
  EXPECT_EQ(sampled_scores.size(), 3);
  EXPECT_EQ(sampled_scores[0].size(), 1);
  EXPECT_EQ(sampled_scores[1].size(), 1);
  EXPECT_EQ(sampled_scores[2].size(), 1);
}

TEST(SamplingCpuUtilTest, TopKTopPSampling_LargeVocabIndices) {
  // Tests that sampling works correctly when top-k topk_token_ids are larger
  // than k. This exposes a bug where vocab topk_token_ids were incorrectly used
  // as offsets.
  std::vector<float> logits = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                               1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0};
  auto rng = std::make_shared<std::default_random_engine>(0);
  std::vector<std::vector<float>> sampled_scores;
  // Top p = 0.0001f, should always return the token with the highest logit,
  // which is the 14th token in this case.
  auto sampled_ids =
      TopKTopPSampling(absl::MakeConstSpan(logits), /*k=*/15, /*p=*/0.0001f,
                       /*temperature=*/1.0f, rng, /*batch_size=*/1,
                       /*sequence_size=*/1, sampled_scores);
  ASSERT_TRUE(sampled_ids.ok());
  // With very small temperature, it should pick the logit with the highest
  // value, which is at index 11.
  EXPECT_EQ(sampled_ids->size(), 1);
  EXPECT_THAT((*sampled_ids)[0], ElementsAre(14));
  EXPECT_EQ(sampled_scores.size(), 1);
  EXPECT_THAT(sampled_scores[0],
              ElementsAre(testing::FloatNear(0.99827528f, 1e-5f)));
}

TEST(SamplingCpuUtilTest, Softmax_BatchSize2SequenceLength2) {
  // Batch size 2, sequence length 2, vocab size 3.
  const std::vector<float> logits = {
      0.1f, 0.1f, 0.2f,  // Batch 0, Sequence 0
      0.0f, 5.0f, 1.0f,  // Batch 0, Sequence 1
      1.0f, 0.0f, 2.0f,  // Batch 1, Sequence 0
      0.0f, 0.0f, 0.0f   // Batch 1, Sequence 1
  };
  absl::Span<const float> logits_span = absl::MakeConstSpan(logits);
  const std::vector<int> topk_indices = {
      1, 2,  // Batch 0, Sequence 0
      1, 2,  // Batch 0, Sequence 1
      0, 2,  // Batch 1, Sequence 0
      0, 1   // Batch 1, Sequence 1
  };
  absl::Span<const int> topk_indices_span = absl::MakeConstSpan(topk_indices);
  std::vector<std::vector<float>> max_logit_values;
  auto probabilities = Softmax(logits_span, topk_indices_span,
                               /*temperature=*/1.0f, /*batch_size=*/2,
                               /*sequence_size=*/2, max_logit_values);
  ASSERT_TRUE(probabilities.ok());
  EXPECT_EQ(probabilities->size(), 2);
  EXPECT_EQ((*probabilities)[0].size(), 4);
  EXPECT_THAT((*probabilities)[0],
              ElementsAre(testing::FloatNear(0.47502f, 1e-4f),
                          testing::FloatNear(0.52498f, 1e-4f),
                          testing::FloatNear(0.98201f, 1e-4f),
                          testing::FloatNear(0.017986f, 1e-4f)));
  EXPECT_EQ((*probabilities)[1].size(), 4);
  EXPECT_THAT((*probabilities)[1],
              ElementsAre(testing::FloatNear(0.26894f, 1e-4f),
                          testing::FloatNear(0.73106f, 1e-4f),
                          testing::FloatNear(0.5f, 1e-4f),
                          testing::FloatNear(0.5f, 1e-4f)));

  EXPECT_EQ(max_logit_values.size(), 2);
  EXPECT_EQ(max_logit_values[0].size(), 2);
  EXPECT_THAT(max_logit_values[0], ElementsAre(0.2f, 5.0f));
  EXPECT_EQ(max_logit_values[1].size(), 2);
  EXPECT_THAT(max_logit_values[1], ElementsAre(2.0f, 0.0f));
}

TEST(SamplingCpuUtilTest, TopKTopPSampling_BatchSize2SequenceLength2) {
  // Batch of 2, sequence length 2, vocab size of 3.
  const std::vector<float> logits = {
      0.0f, 0.0f, 1.0f,  // b0, s0: top is 2
      1.0f, 0.0f, 0.0f,  // b0, s1: top is 0
      0.0f, 1.0f, 0.0f,  // b1, s0: top is 1
      0.0f, 0.0f, 1.0f   // b1, s1: top is 2
  };
  auto rng = std::make_shared<std::default_random_engine>(0);
  std::vector<std::vector<float>> sampled_scores;
  auto sampled_ids =
      TopKTopPSampling(absl::MakeConstSpan(logits), /*k=*/2, /*p=*/0.5f,
                       /*temperature=*/0.00001f, rng, /*batch_size=*/2,
                       /*sequence_size=*/2, sampled_scores);
  ASSERT_TRUE(sampled_ids.ok());
  EXPECT_EQ(sampled_ids->size(), 2);
  EXPECT_THAT((*sampled_ids)[0], ElementsAre(2, 0));
  EXPECT_THAT((*sampled_ids)[1], ElementsAre(1, 2));

  EXPECT_EQ(sampled_scores.size(), 2);
  EXPECT_EQ(sampled_scores[0].size(), 2);
  EXPECT_EQ(sampled_scores[1].size(), 2);
}

TEST(SamplingCpuUtilTest, BenchmarkTopKTokenIds) {
  constexpr int kVocabSize = 256000;  // Gemma vocab size
  constexpr int kIterations = 10;
  std::vector<float> logits(kVocabSize);
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dis(-5.0f, 5.0f);
  for (float& val : logits) {
    val = dis(gen);
  }

  absl::Time start = absl::Now();
  for (int i = 0; i < kIterations; ++i) {
    auto res = TopKTokenIds(absl::MakeConstSpan(logits), /*k=*/1,
                            /*batch_size=*/1, /*sequence_size=*/1);
    ASSERT_TRUE(res.ok());
  }
  absl::Time end = absl::Now();
  absl::Duration duration = end - start;
  std::printf("\n[BENCHMARK] TopK=1 (greedy) execution took: %.3f ms\n",
              absl::ToDoubleMilliseconds(duration) / kIterations);

  start = absl::Now();
  for (int i = 0; i < kIterations; ++i) {
    auto res = TopKTokenIds(absl::MakeConstSpan(logits), /*k=*/256,
                            /*batch_size=*/1, /*sequence_size=*/1);
    ASSERT_TRUE(res.ok());
  }
  end = absl::Now();
  duration = end - start;
  std::printf("[BENCHMARK] TopK=256 (min-heap) execution took: %.3f ms\n",
              absl::ToDoubleMilliseconds(duration) / kIterations);

  start = absl::Now();
  for (int i = 0; i < kIterations; ++i) {
    auto res = TopKTokenIds(absl::MakeConstSpan(logits), /*k=*/1025,
                            /*batch_size=*/1, /*sequence_size=*/1);
    ASSERT_TRUE(res.ok());
  }
  end = absl::Now();
  duration = end - start;
  std::printf("[BENCHMARK] TopK=1025 (full-sort) execution took: %.3f ms\n\n",
              absl::ToDoubleMilliseconds(duration) / kIterations);
}

}  // namespace
}  // namespace litert::lm
