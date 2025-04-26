#include "runtime/components/sampling_cpu_util.h"

#include <random>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"  // from @com_google_absl

namespace litert::lm {
namespace {

using ::testing::ElementsAre;

TEST(SamplingCpuUtilTest, Softmax_BatchSize1) {
  const std::vector<float> logits = {0.1, 0.1};
  auto probabilities =
      Softmax(absl::MakeConstSpan(logits), /*temperature=*/1.0);
  EXPECT_TRUE(probabilities.ok());
  EXPECT_THAT(*probabilities, ElementsAre(0.5, 0.5));
}

TEST(SamplingCpuUtilTest, Softmax_AllZeroLogits) {
  const std::vector<float> logits = {0.0, 0.0};
  auto probabilities =
      Softmax(absl::MakeConstSpan(logits), /*temperature=*/1.0);
  EXPECT_TRUE(probabilities.ok());
  EXPECT_THAT(*probabilities, ElementsAre(0.5, 0.5));
}

TEST(SamplingCpuUtilTest, Softmax_TemperatureZero) {
  const std::vector<float> logits = {0.0, 1.0, 2.0};
  auto probabilities =
      Softmax(absl::MakeConstSpan(logits), /*temperature=*/0.00000001f);
  EXPECT_TRUE(probabilities.ok());
  // Very small temperature should mimic greedy sampling.
  EXPECT_THAT(*probabilities, ElementsAre(0.0, 0.0, 1.0));
}

TEST(SamplingCpuUtilTest, Softmax_TemperatureInf) {
  const std::vector<float> logits = {0.0, 1.0, 2.0, 3.0};
  auto probabilities =
      Softmax(absl::MakeConstSpan(logits), /*temperature=*/100000000000.0f);
  EXPECT_TRUE(probabilities.ok());
  // Very large temperature should mimic uniform sampling.
  EXPECT_THAT(*probabilities, ElementsAre(0.25, 0.25, 0.25, 0.25));
}

TEST(SamplingCpuUtilTest, Softmax_BatchSize3) {
  // Batch size of 3, vocab size of 2.
  const std::vector<float> logits = {0.1, 0.1, 0, 5, 1, 0};
  absl::Span<const float> logits_span = absl::MakeConstSpan(logits);
  auto probabilities = Softmax(absl::MakeConstSpan(logits_span),
                               /*temperature=*/1.0, /*batch_size=*/3);
  EXPECT_TRUE(probabilities.ok());
  EXPECT_THAT(*probabilities, ElementsAre(0.5, 0.5, 0.00669285096, 0.993307173,
                                          0.731058598, 0.268941432));
}

TEST(SamplingCpuUtilTest, TopKTopPSampling_InvalidInputs) {
  const std::vector<float> probabilities = {0.0, 0.0, 0.3};
  std::random_device rd;
  std::mt19937 rng(rd());
  // Negative k.
  auto sampled_ids =
      TopKTopPSampling(absl::MakeConstSpan(probabilities), -1, 0.5, rng, 1);
  EXPECT_FALSE(sampled_ids.ok());
  // Negative p.
  sampled_ids =
      TopKTopPSampling(absl::MakeConstSpan(probabilities), 1, -0.5, rng, 1);
  EXPECT_FALSE(sampled_ids.ok());
}

TEST(SamplingCpuUtilTest, TopKTopPSampling_BatchSize1) {
  const std::vector<float> probabilities = {0.0, 0.0, 0.3};
  std::random_device rd;
  std::mt19937 rng(rd());
  auto sampled_ids =
      TopKTopPSampling(absl::MakeConstSpan(probabilities), 1, 0.5, rng, 1);
  EXPECT_TRUE(sampled_ids.ok());
  EXPECT_THAT(*sampled_ids, ElementsAre(2));
}

TEST(SamplingCpuUtilTest, TopKTopPSampling_BatchSize2) {
  // Batch of 3, vocab size of 3. The sampled ids are 2, 1, 0.
  const std::vector<float> probabilities = {0.0, 0.0, 1.0, 0.0, 1.0,
                                            0.0, 1.0, 0.0, 0.0};
  std::random_device rd;
  std::mt19937 rng(rd());
  auto sampled_ids =
      TopKTopPSampling(absl::MakeConstSpan(probabilities), 2, 0.5, rng, 3);
  EXPECT_TRUE(sampled_ids.ok());
  EXPECT_THAT(*sampled_ids, ElementsAre(2, 1, 0));
}

}  // namespace
}  // namespace litert::lm
