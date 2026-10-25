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

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
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

TEST(SamplingCpuUtilTest, ComputeTopKDistribution_SortedSoftmaxOverTopK) {
  // Two positions with vocab 5, k = 3, temperature 0.5.
  const std::vector<float> logits = {0.0f, 2.0f, 1.0f, 3.0f,  -1.0f,
                                     1.0f, 1.0f, 0.0f, -2.0f, 0.5f};
  std::vector<float> probs;
  std::vector<int> ids;
  ASSERT_TRUE(ComputeTopKDistribution(logits, /*k=*/3, /*temperature=*/0.5f,
                                      /*batch_size=*/1, /*sequence_size=*/2,
                                      probs, ids)
                  .ok());
  ASSERT_EQ(probs.size(), 6);
  // Position 0: top-3 ids 3, 1, 2 with logits 3, 2, 1.
  EXPECT_THAT(std::vector<int>(ids.begin(), ids.begin() + 3),
              ElementsAre(3, 1, 2));
  const double z0 = 1.0 + std::exp(-2.0) + std::exp(-4.0);
  EXPECT_NEAR(probs[0], 1.0 / z0, 1e-6);
  EXPECT_NEAR(probs[1], std::exp(-2.0) / z0, 1e-6);
  EXPECT_NEAR(probs[2], std::exp(-4.0) / z0, 1e-6);
  // Position 1: ids 0 and 1 tie; ties are ordered by ascending id.
  EXPECT_THAT(std::vector<int>(ids.begin() + 3, ids.end()),
              ElementsAre(0, 1, 4));
  EXPECT_NEAR(probs[3], probs[4], 1e-7);
  EXPECT_NEAR(probs[3] + probs[4] + probs[5], 1.0, 1e-6);
}

TEST(SamplingCpuUtilTest, ComputeTopKDistribution_ClampsKAndSortsLargeK) {
  // k > 1024 takes the partial-sort path of TopKTokenIds(), whose output is
  // not sorted; the distribution must still be sorted.
  constexpr int kVocab = 1100;
  std::vector<float> logits(kVocab);
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
  for (float& logit : logits) logit = dist(rng);
  std::vector<float> probs;
  std::vector<int> ids;
  ASSERT_TRUE(ComputeTopKDistribution(logits, /*k=*/5000, /*temperature=*/1.0f,
                                      /*batch_size=*/1, /*sequence_size=*/1,
                                      probs, ids)
                  .ok());
  ASSERT_EQ(probs.size(), kVocab);
  double total = 0.0;
  for (int i = 0; i < kVocab; ++i) {
    total += probs[i];
    if (i > 0) {
      EXPECT_GE(probs[i - 1], probs[i]) << i;
      // Near-equal logits can round to equal probabilities.
      EXPECT_GE(logits[ids[i - 1]], logits[ids[i]] - 1e-5f) << i;
    }
  }
  EXPECT_NEAR(total, 1.0, 1e-4);
}

TEST(SamplingCpuUtilTest, ComputeTopKDistribution_InvalidArguments) {
  std::vector<float> probs;
  std::vector<int> ids;
  const std::vector<float> logits = {0.0f, 1.0f, 2.0f};
  EXPECT_EQ(
      ComputeTopKDistribution(logits, /*k=*/0, 1.0f, 1, 1, probs, ids).code(),
      absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(
      ComputeTopKDistribution(logits, /*k=*/1, 1.0f, 1, 2, probs, ids).code(),
      absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(ComputeTopKDistribution({}, /*k=*/1, 1.0f, 1, 1, probs, ids).code(),
            absl::StatusCode::kInvalidArgument);
}

// Logits of shape [2, 2, 4], with k = 3 and temperature 1. The top-3 logits of
// the four (batch, position) rows, in descending order, are:
//   - row 0: ids 3, 1, 2 with logits 3, 2, 1.
//   - row 1: ids 0, 2, 3 with logits 5, 4, 0.
//   - row 2: ids 0, 1, 2 with logits 1, 1, 0 (ids 0 and 1 tie).
//   - row 3: ids 2, 3, 1 with logits 2, 1, 0.
constexpr int kDistBatch = 2;
constexpr int kDistSeq = 2;
constexpr int kDistK = 3;
std::vector<float> DistLogits() {
  return {0.0f, 2.0f, 1.0f, 3.0f,  5.0f,  -1.0f, 4.0f, 0.0f,
          1.0f, 1.0f, 0.0f, -2.0f, -3.0f, 0.0f,  2.0f, 1.0f};
}

TEST(SamplingCpuUtilTest, ComputeTopKDistribution_BatchSize2SequenceLength2) {
  std::vector<float> probs;
  std::vector<int> ids;
  ASSERT_TRUE(ComputeTopKDistribution(DistLogits(), kDistK,
                                      /*temperature=*/1.0f, kDistBatch,
                                      kDistSeq, probs, ids)
                  .ok());
  EXPECT_THAT(ids, ElementsAre(3, 1, 2, 0, 2, 3, 0, 1, 2, 2, 3, 1));
  // Expected unnormalized probabilities exp(logit - max logit) of each row.
  const std::vector<double> unnormalized = {1.0,
                                            std::exp(-1.0),
                                            std::exp(-2.0),
                                            1.0,
                                            std::exp(-1.0),
                                            std::exp(-5.0),
                                            1.0,
                                            1.0,
                                            std::exp(-1.0),
                                            1.0,
                                            std::exp(-1.0),
                                            std::exp(-2.0)};
  ASSERT_EQ(probs.size(), unnormalized.size());
  for (int row = 0; row < kDistBatch * kDistSeq; ++row) {
    double z = 0.0;
    for (int i = 0; i < kDistK; ++i) z += unnormalized[row * kDistK + i];
    for (int i = 0; i < kDistK; ++i) {
      EXPECT_NEAR(probs[row * kDistK + i], unnormalized[row * kDistK + i] / z,
                  1e-6)
          << "row=" << row << " i=" << i;
    }
  }
}

TEST(SamplingCpuUtilTest, TopKTopPSampling_OutputsTopKDistribution) {
  constexpr float kTopP = 0.9f;
  auto rng = std::make_shared<std::default_random_engine>(5);
  std::vector<std::vector<float>> sampled_scores;
  TopKDistribution distribution;
  ASSERT_OK_AND_ASSIGN(
      auto sampled_ids,
      TopKTopPSampling(DistLogits(), kDistK, kTopP, /*temperature=*/1.0f, rng,
                       kDistBatch, kDistSeq, sampled_scores, &distribution));

  // The distribution is the same as the one ComputeTopKDistribution() returns.
  std::vector<float> expected_probs;
  std::vector<int> expected_ids;
  ASSERT_TRUE(ComputeTopKDistribution(DistLogits(), kDistK,
                                      /*temperature=*/1.0f, kDistBatch,
                                      kDistSeq, expected_probs, expected_ids)
                  .ok());
  EXPECT_EQ(distribution.k, kDistK);
  EXPECT_EQ(distribution.ids, expected_ids);
  EXPECT_EQ(distribution.probs, expected_probs);

  // Each sampled token is in the top-p prefix of its row of the distribution,
  // and its score is its probability.
  for (int b = 0; b < kDistBatch; ++b) {
    for (int s = 0; s < kDistSeq; ++s) {
      const int row = b * kDistSeq + s;
      double mass = 0.0;
      int prefix_size = 0;
      while (prefix_size < kDistK && mass < kTopP) {
        mass += distribution.probs[row * kDistK + prefix_size];
        ++prefix_size;
      }
      const auto prefix_begin = distribution.ids.begin() + row * kDistK;
      const auto it = std::find(prefix_begin, prefix_begin + prefix_size,
                                sampled_ids[b][s]);
      ASSERT_NE(it, prefix_begin + prefix_size) << "b=" << b << " s=" << s;
      EXPECT_EQ(sampled_scores[b][s],
                distribution.probs[it - distribution.ids.begin()]);
    }
  }
}

TEST(SamplingCpuUtilTest, TopKTopPSampling_DistributionDoesNotChangeSampling) {
  auto rng_with = std::make_shared<std::default_random_engine>(9);
  auto rng_without = std::make_shared<std::default_random_engine>(9);
  for (int i = 0; i < 16; ++i) {
    std::vector<std::vector<float>> scores_with;
    std::vector<std::vector<float>> scores_without;
    TopKDistribution distribution;
    ASSERT_OK_AND_ASSIGN(
        auto ids_with,
        TopKTopPSampling(DistLogits(), kDistK, /*p=*/0.95f,
                         /*temperature=*/1.0f, rng_with, kDistBatch, kDistSeq,
                         scores_with, &distribution));
    ASSERT_OK_AND_ASSIGN(
        auto ids_without,
        TopKTopPSampling(DistLogits(), kDistK, /*p=*/0.95f,
                         /*temperature=*/1.0f, rng_without, kDistBatch,
                         kDistSeq, scores_without));
    EXPECT_EQ(ids_with, ids_without) << "i=" << i;
    EXPECT_EQ(scores_with, scores_without) << "i=" << i;
  }
}

TEST(SamplingCpuUtilTest, TopKTopPSampling_TopPCutBreaksTiesByTokenId) {
  // The top-k tokens all have the same probability, and k > 1024 takes the
  // nth_element path of TopKTokenIds(), which returns them in no particular
  // order. With p below a single token's probability, the top-p cut keeps
  // only the first token of the sorted distribution: the smallest id.
  constexpr int kVocab = 1100;
  constexpr int kK = 1050;
  std::vector<float> logits(kVocab, 0.0f);
  std::fill(logits.begin(), logits.begin() + kK, 1.0f);
  auto rng = std::make_shared<std::default_random_engine>(3);
  for (int i = 0; i < 8; ++i) {
    std::vector<std::vector<float>> sampled_scores;
    TopKDistribution distribution;
    ASSERT_OK_AND_ASSIGN(
        auto sampled_ids,
        TopKTopPSampling(logits, kK, /*p=*/1e-4f, /*temperature=*/1.0f, rng,
                         /*batch_size=*/1,
                         /*sequence_size=*/1, sampled_scores, &distribution));
    EXPECT_THAT(sampled_ids, ElementsAre(ElementsAre(0))) << "i=" << i;
    ASSERT_EQ(distribution.ids.size(), kK);
    for (int j = 0; j < kK; ++j) {
      ASSERT_EQ(distribution.ids[j], j);
    }
  }
}

}  // namespace
}  // namespace litert::lm
