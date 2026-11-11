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

#include "runtime/components/top_p_cpu_sampler.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/internal/litert_runtime_c_api.h"  // from @litert
#include "litert/c/litert_common.h"  // from @litert
#include "litert/c/litert_tensor_buffer_types.h"  // from @litert
#include "litert/cc/internal/litert_handle.h"  // from @litert
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_expected.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/cc/litert_tensor_buffer_types.h"  // from @litert
#include "runtime/proto/sampler_params.pb.h"
#include "runtime/util/convert_tensor_buffer.h"  // IWYU pragma: keep
#include "runtime/util/test_utils.h"  // IWYU pragma: keep
#include "tflite/types/half.h"  // from @litert

namespace litert::lm {
namespace {

using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::FloatNear;

// Uniforms for `batch_size * sequence_size` tokens. Their values don't matter
// for greedy sampling (k = 1).
std::vector<float> ZeroUniforms(int batch_size, int sequence_size) {
  return std::vector<float>(batch_size * sequence_size, 0.0f);
}

Expected<void> WriteFp16ToTensorBuffer(absl::Span<const float> data,
                                       TensorBuffer& tensor_buffer) {
  std::vector<uint16_t> fp16_data;
  fp16_data.reserve(data.size());
  for (float val : data) {
    tflite::half half_val(val);
    uint16_t fp16_val;
    std::memcpy(&fp16_val, &half_val, sizeof(uint16_t));
    fp16_data.push_back(fp16_val);
  }
  return tensor_buffer.Write(absl::MakeConstSpan(fp16_data));
}

Expected<TensorBuffer> CopyFp16ToTensorBuffer(absl::Span<const float> data,
                                              absl::Span<const int> dims) {
  LITERT_ASSIGN_OR_RETURN(
      auto tensor_buffer,
      TensorBuffer::CreateManagedHostMemory(
          RankedTensorType(ElementType::Float16,
                           Layout(Dimensions(dims.begin(), dims.end()))),
          data.size() * sizeof(uint16_t)));
  LITERT_RETURN_IF_ERROR(WriteFp16ToTensorBuffer(data, tensor_buffer));
  return tensor_buffer;
}

TEST(TopPSamplerTest, Create) {
  auto sampler_or = TopPSampler::Create(/*k=*/1, /*p=*/0.5, /*temperature=*/1.0,
                                        /*batch_size=*/1, /*sequence_size=*/1);
  EXPECT_TRUE(sampler_or.ok());
}

TEST(TopPSamplerTest, CreateWithZeroTemp) {
  auto sampler_or = TopPSampler::Create(/*k=*/1, /*p=*/0.5, /*temperature=*/0.0,
                                        /*batch_size=*/1, /*sequence_size=*/1);
  EXPECT_TRUE(sampler_or.ok());
}

TEST(TopPSamplerTest, CreateWithNegativeTemp) {
  auto sampler_or =
      TopPSampler::Create(/*k=*/1, /*p=*/0.5, /*temperature=*/-1.0,
                          /*batch_size=*/1, /*sequence_size=*/1);
  EXPECT_FALSE(sampler_or.ok());
  EXPECT_THAT(sampler_or.status().message(),
              testing::HasSubstr("Temperature must be >= 0"));
}

TEST(TopPSamplerTest, SampleToIdAndScoreBuffer_IdsOnly_BatchSize2) {
  auto sampler_or = TopPSampler::Create(/*k=*/1, /*p=*/0.5, /*temperature=*/1.0,
                                        /*batch_size=*/2, /*sequence_size=*/1);
  ASSERT_TRUE(sampler_or.ok());
  auto sampler = *std::move(sampler_or);

  const std::vector<float> logits = {0.0, 0.0, 10.0, 0.0, 11.0, 12.0, 1.0, 2.0};
  auto logits_tensor = CopyToTensorBuffer<float>(logits, {2, 4});

  std::vector<int> ids_vector(2);
  auto ids_tensor =
      CopyToTensorBuffer<int>(absl::MakeConstSpan(ids_vector), {2});
  auto status = sampler->SampleToIdAndScoreBuffer(*logits_tensor, *ids_tensor,
                                                  /*scores_tensor=*/nullptr,
                                                  ZeroUniforms(2, 1));
  EXPECT_TRUE(status.ok());

  auto ids = CopyFromTensorBuffer<int>(*ids_tensor);
  ASSERT_TRUE(ids.HasValue());
  // The sampled id is 2 and 1.
  EXPECT_THAT(*ids, ElementsAre(2, 1));
}

TEST(TopPSamplerTest, SampleToIdAndScoreBuffer_BatchSize2) {
  auto sampler_or = TopPSampler::Create(/*k=*/1, /*p=*/0.5, /*temperature=*/1.0,
                                        /*batch_size=*/2, /*sequence_size=*/1);
  ASSERT_TRUE(sampler_or.ok());
  auto sampler = *std::move(sampler_or);

  const std::vector<float> logits = {
      std::numeric_limits<float>::min(), std::numeric_limits<float>::min(),
      std::numeric_limits<float>::max(), std::numeric_limits<float>::min(),
      std::numeric_limits<float>::min(), std::numeric_limits<float>::max(),
      std::numeric_limits<float>::min(), std::numeric_limits<float>::min()};
  auto logits_tensor = CopyToTensorBuffer<float>(logits, {2, 4});
  ASSERT_TRUE(logits_tensor.HasValue());

  std::vector<int> ids_vector(2);
  auto ids_tensor =
      CopyToTensorBuffer<int>(absl::MakeConstSpan(ids_vector), {2});
  ASSERT_TRUE(ids_tensor.HasValue());
  std::vector<float> scores_vector(2);
  auto scores_tensor =
      CopyToTensorBuffer<float>(absl::MakeConstSpan(scores_vector), {2});
  ASSERT_TRUE(scores_tensor.HasValue());
  auto status = sampler->SampleToIdAndScoreBuffer(
      *logits_tensor, *ids_tensor, &(*scores_tensor), ZeroUniforms(2, 1));
  EXPECT_TRUE(status.ok());
  auto ids = CopyFromTensorBuffer<int>(*ids_tensor);
  ASSERT_TRUE(ids.HasValue());
  // The sampled id is 2 and 1.
  EXPECT_THAT(*ids, ElementsAre(2, 1));

  auto scores = CopyFromTensorBuffer<float>(*scores_tensor);
  ASSERT_TRUE(scores.HasValue());
  // The scores are the log of the probability of the sampled token.
  EXPECT_THAT(*scores, ElementsAre(std::log(1.0f), std::log(1.0f)));
}

TEST(TopPSamplerTest, SampleToIdAndScoreBuffer_BatchSize2SequenceLength2) {
  auto sampler_or = TopPSampler::Create(/*k=*/1, /*p=*/0.5, /*temperature=*/1.0,
                                        /*batch_size=*/2, /*sequence_size=*/2);
  ASSERT_TRUE(sampler_or.ok());
  auto sampler = *std::move(sampler_or);

  const std::vector<float> logits = {
      0.0,  0.0,  10.0, 0.0,  // b0, s0: top is 2
      11.0, 0.0,  0.0,  0.0,  // b0, s1: top is 0
      0.0,  12.0, 0.0,  0.0,  // b1, s0: top is 1
      0.0,  0.0,  0.0,  13.0  // b1, s1: top is 3
  };
  auto logits_tensor = CopyToTensorBuffer<float>(logits, {2, 2, 4});
  ASSERT_TRUE(logits_tensor.HasValue());

  std::vector<int> ids_vector(4);
  auto ids_tensor =
      CopyToTensorBuffer<int>(absl::MakeConstSpan(ids_vector), {2, 2});
  ASSERT_TRUE(ids_tensor.HasValue());
  std::vector<float> scores_vector(4);
  auto scores_tensor =
      CopyToTensorBuffer<float>(absl::MakeConstSpan(scores_vector), {2, 2});
  ASSERT_TRUE(scores_tensor.HasValue());
  auto status = sampler->SampleToIdAndScoreBuffer(
      *logits_tensor, *ids_tensor, &(*scores_tensor), ZeroUniforms(2, 2));
  EXPECT_TRUE(status.ok());
  auto ids = CopyFromTensorBuffer<int>(*ids_tensor);
  ASSERT_TRUE(ids.HasValue());
  // The sampled ids are 2, 0, 1, 3.
  EXPECT_THAT(*ids, ElementsAre(2, 0, 1, 3));

  auto scores = CopyFromTensorBuffer<float>(*scores_tensor);
  ASSERT_TRUE(scores.HasValue());
  // The scores are the log of the probability of the sampled token.
  EXPECT_THAT(*scores, ElementsAre(std::log(1.0f), std::log(1.0f),
                                   std::log(1.0f), std::log(1.0f)));
}

// Overwriting and sampling from the same Float16 TensorBuffer across calls
// verifies both that the read lock is released and that `logits_data_` is
// refreshed on every call.
TEST(TopPSamplerTest,
     SampleToIdAndScoreBufferFp16_RepeatedCallsOnSameLogitsTensor) {
  ASSERT_OK_AND_ASSIGN(
      auto sampler, TopPSampler::Create(/*k=*/1, /*p=*/0.5, /*temperature=*/1.0,
                                        /*batch_size=*/2, /*sequence_size=*/1));
  auto logits_tensor =
      CopyFp16ToTensorBuffer(std::vector<float>(8, 0.0f), {2, 4});
  ASSERT_TRUE(logits_tensor.HasValue());
  std::vector<int> ids_vector(2);
  auto ids_tensor =
      CopyToTensorBuffer<int>(absl::MakeConstSpan(ids_vector), {2});
  ASSERT_TRUE(ids_tensor.HasValue());

  int step = 0;
  for (const auto& [logits, expected_ids] :
       std::vector<std::pair<std::vector<float>, std::vector<int>>>{
           {{0.0f, 0.0f, 10.0f, 0.0f, 11.0f, 12.0f, 1.0f, 2.0f}, {2, 1}},
           {{9.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 2.0f, 8.0f}, {0, 3}},
           {{0.0f, 7.0f, 1.0f, 0.0f, 0.0f, 1.0f, 6.0f, 2.0f}, {1, 2}},
       }) {
    SCOPED_TRACE(step++);
    ASSERT_TRUE(WriteFp16ToTensorBuffer(logits, *logits_tensor).HasValue());
    ASSERT_OK(sampler->SampleToIdAndScoreBuffer(*logits_tensor, *ids_tensor,
                                                /*scores_tensor=*/nullptr,
                                                ZeroUniforms(2, 1)));
    auto ids = CopyFromTensorBuffer<int>(*ids_tensor);
    ASSERT_TRUE(ids.HasValue());
    EXPECT_EQ(*ids, expected_ids);
  }
}

TEST(TopPSamplerTest, SampleToIdAndScoreBufferFp16_IdsOnly_BatchSize2) {
  auto sampler_or = TopPSampler::Create(/*k=*/1, /*p=*/0.5, /*temperature=*/1.0,
                                        /*batch_size=*/2, /*sequence_size=*/1);
  ASSERT_TRUE(sampler_or.ok());
  auto sampler = *std::move(sampler_or);

  const std::vector<float> logits = {0.0, 0.0, 10.0, 0.0, 11.0, 12.0, 1.0, 2.0};
  auto logits_tensor = CopyFp16ToTensorBuffer(logits, {2, 4});
  ASSERT_TRUE(logits_tensor.HasValue());
  std::vector<int> ids_vector(2);
  auto ids_tensor =
      CopyToTensorBuffer<int>(absl::MakeConstSpan(ids_vector), {2});
  ASSERT_TRUE(ids_tensor.HasValue());
  auto status = sampler->SampleToIdAndScoreBuffer(*logits_tensor, *ids_tensor,
                                                  /*scores_tensor=*/nullptr,
                                                  ZeroUniforms(2, 1));
  EXPECT_TRUE(status.ok());

  auto ids = CopyFromTensorBuffer<int>(*ids_tensor);
  ASSERT_TRUE(ids.HasValue());
  // The sampled id is 2 and 1.
  EXPECT_THAT(*ids, ElementsAre(2, 1));
}

TEST(TopPSamplerTest, SampleToIdAndScoreBufferFp16_BatchSize2) {
  auto sampler_or = TopPSampler::Create(/*k=*/1, /*p=*/0.5, /*temperature=*/1.0,
                                        /*batch_size=*/2, /*sequence_size=*/1);
  ASSERT_TRUE(sampler_or.ok());
  auto sampler = *std::move(sampler_or);

  const std::vector<float> logits = {
      std::numeric_limits<float>::min(), std::numeric_limits<float>::min(),
      std::numeric_limits<float>::max(), std::numeric_limits<float>::min(),
      std::numeric_limits<float>::min(), std::numeric_limits<float>::max(),
      std::numeric_limits<float>::min(), std::numeric_limits<float>::min()};
  auto logits_tensor = CopyFp16ToTensorBuffer(logits, {2, 4});
  ASSERT_TRUE(logits_tensor.HasValue());

  std::vector<int> ids_vector(2);
  auto ids_tensor =
      CopyToTensorBuffer<int>(absl::MakeConstSpan(ids_vector), {2});
  ASSERT_TRUE(ids_tensor.HasValue());
  std::vector<float> scores_vector(2);
  auto scores_tensor =
      CopyToTensorBuffer<float>(absl::MakeConstSpan(scores_vector), {2});
  ASSERT_TRUE(scores_tensor.HasValue());

  auto status = sampler->SampleToIdAndScoreBuffer(
      *logits_tensor, *ids_tensor, &(*scores_tensor), ZeroUniforms(2, 1));
  EXPECT_TRUE(status.ok());

  auto ids = CopyFromTensorBuffer<int>(*ids_tensor);
  ASSERT_TRUE(ids.HasValue());
  // The sampled id is 2 and 1.
  EXPECT_THAT(*ids, ElementsAre(2, 1));

  auto scores = CopyFromTensorBuffer<float>(*scores_tensor);
  ASSERT_TRUE(scores.HasValue());
  // The scores are the log of the probability of the sampled token.
  EXPECT_THAT(*scores, ElementsAre(std::log(1.0f), std::log(1.0f)));
}

TEST(TopPSamplerTest, SampleToIdAndScoreBufferFp16_BatchSize2SequenceLength2) {
  auto sampler_or = TopPSampler::Create(/*k=*/1, /*p=*/0.5, /*temperature=*/1.0,
                                        /*batch_size=*/2, /*sequence_size=*/2);
  ASSERT_TRUE(sampler_or.ok());
  auto sampler = *std::move(sampler_or);

  const std::vector<float> logits = {
      0.0,  0.0,  10.0, 0.0,  // b0, s0: top is 2
      11.0, 0.0,  0.0,  0.0,  // b0, s1: top is 0
      0.0,  12.0, 0.0,  0.0,  // b1, s0: top is 1
      0.0,  0.0,  0.0,  13.0  // b1, s1: top is 3
  };
  auto logits_tensor = CopyFp16ToTensorBuffer(logits, {2, 2, 4});
  ASSERT_TRUE(logits_tensor.HasValue());

  std::vector<int> ids_vector(4);
  auto ids_tensor =
      CopyToTensorBuffer<int>(absl::MakeConstSpan(ids_vector), {2, 2});
  ASSERT_TRUE(ids_tensor.HasValue());
  std::vector<float> scores_vector(4);
  auto scores_tensor =
      CopyToTensorBuffer<float>(absl::MakeConstSpan(scores_vector), {2, 2});
  ASSERT_TRUE(scores_tensor.HasValue());

  auto status = sampler->SampleToIdAndScoreBuffer(
      *logits_tensor, *ids_tensor, &(*scores_tensor), ZeroUniforms(2, 2));
  EXPECT_TRUE(status.ok());

  auto ids = CopyFromTensorBuffer<int>(*ids_tensor);
  ASSERT_TRUE(ids.HasValue());
  // The sampled ids are 2, 0, 1, 3.
  EXPECT_THAT(*ids, ElementsAre(2, 0, 1, 3));

  auto scores = CopyFromTensorBuffer<float>(*scores_tensor);
  ASSERT_TRUE(scores.HasValue());
  // The scores are the log of the probability of the sampled token.
  EXPECT_THAT(*scores, ElementsAre(std::log(1.0f), std::log(1.0f),
                                   std::log(1.0f), std::log(1.0f)));
}

TEST(TopPSamplerTest, UpdateConfig) {
  auto sampler_or = TopPSampler::Create(/*k=*/1, /*p=*/0.5, /*temperature=*/1.0,
                                        /*batch_size=*/1, /*sequence_size=*/1);
  ASSERT_TRUE(sampler_or.ok());
  auto sampler = *std::move(sampler_or);

  proto::SamplerParameters sampler_params;
  sampler_params.set_k(1);
  sampler_params.set_p(1.0);
  sampler_params.set_temperature(100.0);

  auto status = sampler->UpdateConfig(sampler_params, /*batch_size=*/1);
  ASSERT_TRUE(status.ok());

  const std::vector<float> logits = {0.0, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0};
  auto logits_tensor = CopyToTensorBuffer<float>(logits, {1, 8});

  std::vector<int> ids_vector = {0, 1, 2, 3, 4, 5, 6, 7};
  auto ids_tensor =
      CopyToTensorBuffer<int>(absl::MakeConstSpan(ids_vector), {1, 8});
  ASSERT_TRUE(ids_tensor.HasValue());

  status = sampler->SampleToIdAndScoreBuffer(*logits_tensor, ids_tensor.Value(),
                                             /*scores_tensor=*/nullptr,
                                             ZeroUniforms(1, 1));
  ASSERT_TRUE(status.ok());

  auto ids = CopyFromTensorBuffer<int>(ids_tensor.Value());
  ASSERT_TRUE(ids.HasValue());
  // With topk=1, it should pick the argmax.
  EXPECT_EQ(ids.Value()[0], 4);

  // Update the config again.
  sampler_params.set_k(8);

  status = sampler->UpdateConfig(sampler_params, /*batch_size=*/1);
  ASSERT_TRUE(status.ok());

  const std::vector<float> uniforms = {0.99f};
  status =
      sampler->SampleToIdAndScoreBuffer(*logits_tensor, ids_tensor.Value(),
                                        /*scores_tensor=*/nullptr, uniforms);
  ASSERT_TRUE(status.ok());

  ids = CopyFromTensorBuffer<int>(ids_tensor.Value());

  // With topk=8 and high temperature, the distribution is nearly flat, so a
  // uniform close to 1 picks a token other than the argmax.
  EXPECT_NE(ids.Value()[0], 4);
}

TEST(TopPSamplerTest, UpdateConfigRejectsNonPositiveBatchSize) {
  ASSERT_OK_AND_ASSIGN(auto sampler,
                       TopPSampler::Create(
                           /*k=*/3, /*p=*/1.0, /*temperature=*/1.0,
                           /*batch_size=*/1, /*sequence_size=*/1));
  proto::SamplerParameters sampler_params;
  sampler_params.set_k(3);
  sampler_params.set_p(1.0);
  sampler_params.set_temperature(1.0);
  for (int batch_size : {0, -1}) {
    EXPECT_THAT(sampler->UpdateConfig(sampler_params, batch_size),
                StatusIs(absl::StatusCode::kInvalidArgument))
        << "batch_size=" << batch_size;
  }

  // The rejected updates leave the sampler usable with its batch size of 1.
  const std::vector<float> logits = {0.0f, 1.0f, 0.0f};
  auto logits_tensor = CopyToTensorBuffer<float>(logits, {1, 3});
  ASSERT_TRUE(logits_tensor.HasValue());
  std::vector<int> ids_vector(1);
  auto ids_tensor =
      CopyToTensorBuffer<int>(absl::MakeConstSpan(ids_vector), {1});
  ASSERT_TRUE(ids_tensor.HasValue());
  EXPECT_OK(sampler->SampleToIdAndScoreBuffer(*logits_tensor, *ids_tensor,
                                              /*scores_tensor=*/nullptr,
                                              ZeroUniforms(1, 1)));
}

TEST(TopPSamplerTest,
     SampleToIdAndScoreBuffer_GreedyScoresTensorNullAndNonNull) {
  auto sampler_or = TopPSampler::Create(/*k=*/1, /*p=*/1.0, /*temperature=*/0.0,
                                        /*batch_size=*/1, /*sequence_size=*/1);
  ASSERT_TRUE(sampler_or.ok());
  if (!sampler_or.ok()) {
    return;
  }
  auto sampler = std::move(*sampler_or);
  EXPECT_FALSE(sampler->ComputeExactLogProbs());

  // Vocabulary of size 4: logits = {0.0, 1.0, 2.0, 3.0}
  // Argmax token is 3.
  const std::vector<float> logits = {0.0f, 1.0f, 2.0f, 3.0f};
  auto logits_tensor = CopyToTensorBuffer<float>(logits, {1, 4});
  ASSERT_TRUE(logits_tensor.HasValue());

  // 1. Fast-path greedy decode: scores_tensor == nullptr
  std::vector<int> ids_vector_fast(1);
  auto ids_tensor_fast =
      CopyToTensorBuffer<int>(absl::MakeConstSpan(ids_vector_fast), {1});
  ASSERT_TRUE(ids_tensor_fast.HasValue());

  auto status_fast = sampler->SampleToIdAndScoreBuffer(
      *logits_tensor, *ids_tensor_fast, /*scores_tensor=*/nullptr,
      ZeroUniforms(1, 1));
  EXPECT_TRUE(status_fast.ok());

  auto ids_fast = CopyFromTensorBuffer<int>(*ids_tensor_fast);
  ASSERT_TRUE(ids_fast.HasValue());
  EXPECT_THAT(*ids_fast, ElementsAre(3));

  // 2. Decode with scores requested: default compute_exact_log_probs == false.
  // Preserves fast default Top-K score behavior (std::log(1.0f) = 0.0f).
  std::vector<int> ids_vector_scores(1);
  auto ids_tensor_scores =
      CopyToTensorBuffer<int>(absl::MakeConstSpan(ids_vector_scores), {1});
  ASSERT_TRUE(ids_tensor_scores.HasValue());

  std::vector<float> scores_vector(1);
  auto scores_tensor =
      CopyToTensorBuffer<float>(absl::MakeConstSpan(scores_vector), {1});
  ASSERT_TRUE(scores_tensor.HasValue());

  auto status_scores =
      sampler->SampleToIdAndScoreBuffer(*logits_tensor, *ids_tensor_scores,
                                        &(*scores_tensor), ZeroUniforms(1, 1));
  EXPECT_TRUE(status_scores.ok());

  auto ids_scores = CopyFromTensorBuffer<int>(*ids_tensor_scores);
  ASSERT_TRUE(ids_scores.HasValue());
  // Token ID selection is identical whether scores are requested or not.
  EXPECT_EQ(ids_fast.Value()[0], ids_scores.Value()[0]);
  EXPECT_THAT(*ids_scores, ElementsAre(3));

  auto scores_default = CopyFromTensorBuffer<float>(*scores_tensor);
  ASSERT_TRUE(scores_default.HasValue());
  EXPECT_THAT(*scores_default, ElementsAre(std::log(1.0f)));

  // 3. Opt-in exact full-vocabulary log-probabilities.
  // log_prob = 3.0 - (3.0 + ln(exp(-3) + exp(-2) + exp(-1) + exp(0)))
  //          = 3.0 - (3.0 + ln(1.5529918)) = -0.4401868
  sampler->SetComputeExactLogProbs(true);
  EXPECT_TRUE(sampler->ComputeExactLogProbs());

  status_scores =
      sampler->SampleToIdAndScoreBuffer(*logits_tensor, *ids_tensor_scores,
                                        &(*scores_tensor), ZeroUniforms(1, 1));
  EXPECT_TRUE(status_scores.ok());

  auto scores_exact = CopyFromTensorBuffer<float>(*scores_tensor);
  ASSERT_TRUE(scores_exact.HasValue());
  EXPECT_THAT(*scores_exact, ElementsAre(FloatNear(-0.44018677f, 1e-5)));
}

TEST(TopPSamplerTest, SampleToIdAndScoreBuffer_ExactLogProbsViaCreate) {
  auto sampler_or = TopPSampler::Create(/*k=*/1, /*p=*/1.0, /*temperature=*/0.0,
                                        /*batch_size=*/1, /*sequence_size=*/1,
                                        /*compute_exact_log_probs=*/true);
  ASSERT_TRUE(sampler_or.ok());
  if (!sampler_or.ok()) {
    return;
  }
  auto sampler = std::move(*sampler_or);
  EXPECT_TRUE(sampler->ComputeExactLogProbs());

  const std::vector<float> logits = {0.0f, 1.0f, 2.0f, 3.0f};
  auto logits_tensor = CopyToTensorBuffer<float>(logits, {1, 4});
  ASSERT_TRUE(logits_tensor.HasValue());

  std::vector<int> ids_vector(1);
  auto ids_tensor =
      CopyToTensorBuffer<int>(absl::MakeConstSpan(ids_vector), {1});
  ASSERT_TRUE(ids_tensor.HasValue());

  std::vector<float> scores_vector(1);
  auto scores_tensor =
      CopyToTensorBuffer<float>(absl::MakeConstSpan(scores_vector), {1});
  ASSERT_TRUE(scores_tensor.HasValue());

  auto status = sampler->SampleToIdAndScoreBuffer(
      *logits_tensor, *ids_tensor, &(*scores_tensor), ZeroUniforms(1, 1));
  EXPECT_TRUE(status.ok());

  auto scores = CopyFromTensorBuffer<float>(*scores_tensor);
  ASSERT_TRUE(scores.HasValue());
  EXPECT_THAT(*scores, ElementsAre(FloatNear(-0.44018677f, 1e-5)));
}

// Probabilities 0.5, 0.3 and 0.2 for tokens 0, 1 and 2.
std::vector<float> ThreeTokenLogits() {
  return {std::log(0.5f), std::log(0.3f), std::log(0.2f)};
}

TEST(TopPSamplerTest, Sample_PerPosition) {
  ASSERT_OK_AND_ASSIGN(
      auto sampler, TopPSampler::Create(/*k=*/3, /*p=*/1.0, /*temperature=*/1.0,
                                        /*batch_size=*/2, /*sequence_size=*/2));
  std::vector<float> logits;
  for (int i = 0; i < 4; ++i) {
    const std::vector<float> row = ThreeTokenLogits();
    logits.insert(logits.end(), row.begin(), row.end());
  }
  auto logits_tensor = CopyToTensorBuffer<float>(logits, {2, 2, 3});
  ASSERT_TRUE(logits_tensor.HasValue());
  std::vector<int> ids_vector(4);
  auto ids_tensor =
      CopyToTensorBuffer<int>(absl::MakeConstSpan(ids_vector), {2, 2});
  ASSERT_TRUE(ids_tensor.HasValue());

  // Batch-major: (b0, s0), (b0, s1), (b1, s0), (b1, s1).
  const std::vector<float> uniforms = {0.1f, 0.6f, 0.9f, 0.3f};
  EXPECT_OK(sampler->SampleToIdAndScoreBuffer(*logits_tensor, *ids_tensor,
                                              /*scores_tensor=*/nullptr,
                                              absl::MakeConstSpan(uniforms)));
  auto ids = CopyFromTensorBuffer<int>(*ids_tensor);
  ASSERT_TRUE(ids.HasValue());
  EXPECT_THAT(*ids, ElementsAre(0, 1, 2, 0));
}

// Member declaration order is load-bearing: `buffer` borrows `non_host_env`,
// which borrows `custom_c_api` and `base_env`, so members must be destroyed in
// reverse declaration order.
struct NonHostFloatTensorBuffer {
  Environment base_env;
  std::unique_ptr<LiteRtRuntimeCApiStruct> custom_c_api;
  Environment non_host_env;
  TensorBuffer buffer;
};

// Creates a mappable Float32 TensorBuffer whose BufferType() reports non-host
// memory (kAhwb) so ReferTensorBufferAsSpan<float>() declines zero-copy span
// access and SampleToIdAndScoreBuffer() exercises the bulk-memcpy download path
// on any host platform without requiring device-specific buffer drivers.
Expected<NonHostFloatTensorBuffer> CreateNonHostFloatTensorBuffer(
    absl::Span<const float> data, absl::Span<const int> dims) {
  LITERT_ASSIGN_OR_RETURN(auto base_env, Environment::Create({}));
  auto [base_c_api, env_handle] = base_env.GetHolderForCApi();
  auto custom_c_api = std::make_unique<LiteRtRuntimeCApiStruct>(*base_c_api);
  custom_c_api->litert_get_tensor_buffer_type =
      [](LiteRtTensorBuffer /*tensor_buffer*/,
         LiteRtTensorBufferType* buffer_type) -> LiteRtStatus {
    *buffer_type = kLiteRtTensorBufferTypeAhwb;
    return kLiteRtStatusOk;
  };
  Environment non_host_env =
      Environment::WrapCObject(custom_c_api.get(), env_handle, OwnHandle::kNo);
  LITERT_ASSIGN_OR_RETURN(
      auto buffer,
      TensorBuffer::CreateManaged(
          non_host_env, TensorBufferType::kHostMemory,
          RankedTensorType(ElementType::Float32,
                           Layout(Dimensions(dims.begin(), dims.end()))),
          data.size() * sizeof(float)));
  LITERT_RETURN_IF_ERROR(buffer.Write<float>(data));
  return NonHostFloatTensorBuffer{
      std::move(base_env),
      std::move(custom_c_api),
      std::move(non_host_env),
      std::move(buffer),
  };
}

TEST(TopPSamplerTest,
     SampleToIdAndScoreBuffer_NonHostFloat32ReusesBufferAcrossSteps) {
  ASSERT_OK_AND_ASSIGN(
      auto sampler,
      TopPSampler::Create(/*k=*/1, /*p=*/0.5, /*temperature=*/1.0,
                          /*batch_size=*/2, /*sequence_size=*/1));

  // Step 1: batch 0 peaks at token 2, batch 1 peaks at token 1.
  const std::vector<float> step1_logits = {
      0.0f, 0.0f, 10.0f, 0.0f, 1.0f, 12.0f, 2.0f, 0.0f,
  };
  auto non_host_logits =
      CreateNonHostFloatTensorBuffer(step1_logits, {2, 1, 4});
  ASSERT_TRUE(non_host_logits.HasValue());
  auto buffer_type = non_host_logits->buffer.BufferType();
  ASSERT_TRUE(buffer_type.HasValue());
  EXPECT_EQ(*buffer_type, TensorBufferType::kAhwb);
  EXPECT_FALSE(
      ReferTensorBufferAsSpan<float>(non_host_logits->buffer).HasValue());

  const std::vector<int> zero_ids(2, 0);
  auto ids_tensor = CopyToTensorBuffer<int>(zero_ids, {2, 1});
  ASSERT_TRUE(ids_tensor.HasValue());
  const std::vector<float> zero_scores(2, 0.0f);
  auto scores_tensor = CopyToTensorBuffer<float>(zero_scores, {2, 1});
  ASSERT_TRUE(scores_tensor.HasValue());

  EXPECT_OK(sampler->SampleToIdAndScoreBuffer(
      non_host_logits->buffer, *ids_tensor, /*scores_tensor=*/nullptr,
      ZeroUniforms(2, 1)));
  auto step1_ids = CopyFromTensorBuffer<int>(*ids_tensor);
  ASSERT_TRUE(step1_ids.HasValue());
  EXPECT_THAT(*step1_ids, ElementsAre(2, 1));

  EXPECT_OK(sampler->SampleToIdAndScoreBuffer(
      non_host_logits->buffer, *ids_tensor, &(*scores_tensor),
      ZeroUniforms(2, 1)));
  auto step1_scores = CopyFromTensorBuffer<float>(*scores_tensor);
  ASSERT_TRUE(step1_scores.HasValue());
  EXPECT_THAT(*step1_scores, ElementsAre(0.0f, 0.0f));

  // Step 2: overwrite the same non-host buffer and verify `logits_data_` is
  // refreshed in place (batch 0 -> token 0, batch 1 -> token 3) with exact
  // log-probabilities enabled.
  const std::vector<float> step2_logits = {
      9.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 2.0f, 3.0f,
  };
  ASSERT_TRUE(non_host_logits->buffer.Write<float>(step2_logits).HasValue());
  sampler->SetComputeExactLogProbs(true);

  EXPECT_OK(sampler->SampleToIdAndScoreBuffer(
      non_host_logits->buffer, *ids_tensor, &(*scores_tensor),
      ZeroUniforms(2, 1)));
  auto step2_ids = CopyFromTensorBuffer<int>(*ids_tensor);
  ASSERT_TRUE(step2_ids.HasValue());
  EXPECT_THAT(*step2_ids, ElementsAre(0, 3));

  const float expected_batch0_log_prob =
      -std::log(1.0f + std::exp(-8.0f) + 2.0f * std::exp(-9.0f));
  auto step2_scores = CopyFromTensorBuffer<float>(*scores_tensor);
  ASSERT_TRUE(step2_scores.HasValue());
  EXPECT_THAT(*step2_scores,
              ElementsAre(FloatNear(expected_batch0_log_prob, 1e-5f),
                          FloatNear(-0.44018677f, 1e-5f)));

  // Step 3: overwrite once more and sample without scores.
  const std::vector<float> step3_logits = {
      0.0f, 5.0f, 0.0f, 0.0f, 0.0f, 0.0f, 7.0f, 0.0f,
  };
  ASSERT_TRUE(non_host_logits->buffer.Write<float>(step3_logits).HasValue());
  EXPECT_OK(sampler->SampleToIdAndScoreBuffer(
      non_host_logits->buffer, *ids_tensor, /*scores_tensor=*/nullptr,
      ZeroUniforms(2, 1)));
  auto step3_ids = CopyFromTensorBuffer<int>(*ids_tensor);
  ASSERT_TRUE(step3_ids.HasValue());
  EXPECT_THAT(*step3_ids, ElementsAre(1, 2));
}

TEST(TopPSamplerTest, UpdateConfigSupportsGreedyAndTopKAndValidatesBounds) {
  ASSERT_OK_AND_ASSIGN(
      auto sampler,
      TopPSampler::Create(/*k=*/4, /*p=*/0.1f, /*temperature=*/1.0f,
                          /*batch_size=*/1, /*sequence_size=*/1));

  // Logits [10.0, 9.9, -100.0, -100.0]: with p=0.1 token 0 (~0.525 mass) forms
  // a 1-token nucleus, whereas TOP_K (k=2, p=0.0 normalized to p=1.0) includes
  // token 1 as well, so uniform 0.9 picks token 1.
  const std::vector<float> logits = {10.0f, 9.9f, -100.0f, -100.0f};
  auto logits_tensor = CopyToTensorBuffer<float>(logits, {1, 1, 4});
  ASSERT_TRUE(logits_tensor.HasValue());
  auto ids_tensor = CopyToTensorBuffer<int>(std::vector<int>{-1}, {1, 1});
  ASSERT_TRUE(ids_tensor.HasValue());

  proto::SamplerParameters top_k_params;
  top_k_params.set_type(proto::SamplerParameters::TOP_K);
  top_k_params.set_k(2);
  top_k_params.set_p(0.0f);
  top_k_params.set_temperature(1.0f);
  ASSERT_OK(sampler->UpdateConfig(top_k_params, /*batch_size=*/1));
  const std::vector<float> high_uniform = {0.9f};
  ASSERT_OK(sampler->SampleToIdAndScoreBuffer(*logits_tensor, *ids_tensor,
                                              /*scores_tensor=*/nullptr,
                                              high_uniform));
  auto top_k_ids = CopyFromTensorBuffer<int>(*ids_tensor);
  ASSERT_TRUE(top_k_ids.HasValue());
  EXPECT_THAT(*top_k_ids, ElementsAre(1));

  // GREEDY with default k=0 normalizes to k=1 and always picks argmax token 0,
  // even with uniform 0.9.
  proto::SamplerParameters greedy_params;
  greedy_params.set_type(proto::SamplerParameters::GREEDY);
  ASSERT_OK(sampler->UpdateConfig(greedy_params, /*batch_size=*/1));
  ASSERT_OK(sampler->SampleToIdAndScoreBuffer(*logits_tensor, *ids_tensor,
                                              /*scores_tensor=*/nullptr,
                                              high_uniform));
  auto greedy_ids = CopyFromTensorBuffer<int>(*ids_tensor);
  ASSERT_TRUE(greedy_ids.HasValue());
  EXPECT_THAT(*greedy_ids, ElementsAre(0));

  // Invalid updates fail without mutating the sampler's state.
  proto::SamplerParameters bad_top_k;
  bad_top_k.set_type(proto::SamplerParameters::TOP_K);
  bad_top_k.set_k(0);
  EXPECT_THAT(sampler->UpdateConfig(bad_top_k, /*batch_size=*/1),
              StatusIs(absl::StatusCode::kInvalidArgument));

  proto::SamplerParameters nan_temp;
  nan_temp.set_type(proto::SamplerParameters::TOP_P);
  nan_temp.set_k(2);
  nan_temp.set_p(0.9f);
  nan_temp.set_temperature(std::numeric_limits<float>::quiet_NaN());
  EXPECT_THAT(sampler->UpdateConfig(nan_temp, /*batch_size=*/1),
              StatusIs(absl::StatusCode::kInvalidArgument));

  proto::SamplerParameters unknown_type;
  unknown_type.set_type(static_cast<proto::SamplerParameters::Type>(99));
  unknown_type.set_k(1);
  EXPECT_THAT(sampler->UpdateConfig(unknown_type, /*batch_size=*/1),
              StatusIs(absl::StatusCode::kUnimplemented));

  ASSERT_OK(sampler->SampleToIdAndScoreBuffer(*logits_tensor, *ids_tensor,
                                              /*scores_tensor=*/nullptr,
                                              high_uniform));
  auto preserved_ids = CopyFromTensorBuffer<int>(*ids_tensor);
  ASSERT_TRUE(preserved_ids.HasValue());
  EXPECT_THAT(*preserved_ids, ElementsAre(0));
}

}  // namespace
}  // namespace litert::lm
