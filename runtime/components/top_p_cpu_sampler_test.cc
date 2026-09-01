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
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_expected.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/proto/sampler_params.pb.h"
#include "runtime/util/convert_tensor_buffer.h"  // IWYU pragma: keep
#include "tflite/types/half.h"  // from @litert

namespace litert::lm {
namespace {

using ::testing::ElementsAre;
using ::testing::FloatNear;

Expected<TensorBuffer> CopyFp16ToTensorBuffer(absl::Span<const float> data,
                                              absl::Span<const int> dims) {
  std::vector<uint16_t> fp16_data;
  fp16_data.reserve(data.size());
  for (float val : data) {
    tflite::half half_val(val);
    uint16_t fp16_val;
    std::memcpy(&fp16_val, &half_val, sizeof(uint16_t));
    fp16_data.push_back(fp16_val);
  }

  LITERT_ASSIGN_OR_RETURN(
      auto tensor_buffer,
      TensorBuffer::CreateManagedHostMemory(
          RankedTensorType(ElementType::Float16,
                           Layout(Dimensions(dims.begin(), dims.end()))),
          fp16_data.size() * sizeof(uint16_t)));
  tensor_buffer.Write(absl::MakeConstSpan(fp16_data));
  return tensor_buffer;
}

TEST(TopPSamplerTest, Create) {
  auto sampler_or = TopPSampler::Create(/*k=*/1, /*p=*/0.5, /*temperature=*/1.0,
                                        /*batch_size=*/1, /*sequence_size=*/1,
                                        /*seed=*/1);
  EXPECT_TRUE(sampler_or.ok());
}

TEST(TopPSamplerTest, CreateWithZeroTemp) {
  auto sampler_or = TopPSampler::Create(/*k=*/1, /*p=*/0.5, /*temperature=*/0.0,
                                        /*batch_size=*/1, /*sequence_size=*/1,
                                        /*seed=*/1);
  EXPECT_TRUE(sampler_or.ok());
}

TEST(TopPSamplerTest, CreateWithNegativeTemp) {
  auto sampler_or =
      TopPSampler::Create(/*k=*/1, /*p=*/0.5, /*temperature=*/-1.0,
                          /*batch_size=*/1, /*sequence_size=*/1, /*seed=*/1);
  EXPECT_FALSE(sampler_or.ok());
  EXPECT_THAT(sampler_or.status().message(),
              testing::HasSubstr("Temperature must be >= 0"));
}

TEST(TopPSamplerTest, SampleToIdAndScoreBuffer_IdsOnly_BatchSize2) {
  auto sampler_or = TopPSampler::Create(/*k=*/1, /*p=*/0.5, /*temperature=*/1.0,
                                        /*batch_size=*/2, /*sequence_size=*/1,
                                        /*seed=*/1);
  ASSERT_TRUE(sampler_or.ok());
  auto sampler = *std::move(sampler_or);

  const std::vector<float> logits = {0.0, 0.0, 10.0, 0.0, 11.0, 12.0, 1.0, 2.0};
  auto logits_tensor = CopyToTensorBuffer<float>(logits, {2, 4});

  std::vector<int> ids_vector(2);
  auto ids_tensor =
      CopyToTensorBuffer<int>(absl::MakeConstSpan(ids_vector), {2});
  auto status = sampler->SampleToIdAndScoreBuffer(*logits_tensor, *ids_tensor,
                                                  /*scores_tensor=*/nullptr);
  EXPECT_TRUE(status.ok());

  auto ids = CopyFromTensorBuffer<int>(*ids_tensor);
  ASSERT_TRUE(ids.HasValue());
  // The sampled id is 2 and 1.
  EXPECT_THAT(*ids, ElementsAre(2, 1));
}

TEST(TopPSamplerTest, SampleToIdAndScoreBuffer_BatchSize2) {
  auto sampler_or = TopPSampler::Create(/*k=*/1, /*p=*/0.5, /*temperature=*/1.0,
                                        /*batch_size=*/2, /*sequence_size=*/1,
                                        /*seed=*/1);
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
  auto status = sampler->SampleToIdAndScoreBuffer(*logits_tensor, *ids_tensor,
                                                  &(*scores_tensor));
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
                                        /*batch_size=*/2, /*sequence_size=*/2,
                                        /*seed=*/1);
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
  auto status = sampler->SampleToIdAndScoreBuffer(*logits_tensor, *ids_tensor,
                                                  &(*scores_tensor));
  EXPECT_TRUE(status.ok());
  auto ids = CopyFromTensorBuffer<int>(*ids_tensor);
  ASSERT_TRUE(ids.HasValue());
  // The sampled ids are 2, 0, 1, 3.
  EXPECT_THAT(*ids, ElementsAre(2, 0, 1, 3));

  auto scores = CopyFromTensorBuffer<float>(*scores_tensor);
  ASSERT_TRUE(scores.HasValue());
  // The scores are the log of the probability of the sampled token.
  EXPECT_THAT(
      *scores,
      ElementsAre(
          FloatNear(-0.00013619f, 1e-6), FloatNear(-5.010385e-05f, 1e-6),
          FloatNear(-1.843247e-05f, 1e-6), FloatNear(-6.780965e-06f, 1e-6)));
}

TEST(TopPSamplerTest, SampleToIdAndScoreBufferFp16_IdsOnly_BatchSize2) {
  auto sampler_or = TopPSampler::Create(/*k=*/1, /*p=*/0.5, /*temperature=*/1.0,
                                        /*batch_size=*/2, /*sequence_size=*/1,
                                        /*seed=*/1);
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
                                                  /*scores_tensor=*/nullptr);
  EXPECT_TRUE(status.ok());

  auto ids = CopyFromTensorBuffer<int>(*ids_tensor);
  ASSERT_TRUE(ids.HasValue());
  // The sampled id is 2 and 1.
  EXPECT_THAT(*ids, ElementsAre(2, 1));
}

TEST(TopPSamplerTest, SampleToIdAndScoreBufferFp16_BatchSize2) {
  auto sampler_or = TopPSampler::Create(/*k=*/1, /*p=*/0.5, /*temperature=*/1.0,
                                        /*batch_size=*/2, /*sequence_size=*/1,
                                        /*seed=*/1);
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

  auto status = sampler->SampleToIdAndScoreBuffer(*logits_tensor, *ids_tensor,
                                                  &(*scores_tensor));
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
                                        /*batch_size=*/2, /*sequence_size=*/2,
                                        /*seed=*/1);
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

  auto status = sampler->SampleToIdAndScoreBuffer(*logits_tensor, *ids_tensor,
                                                  &(*scores_tensor));
  EXPECT_TRUE(status.ok());

  auto ids = CopyFromTensorBuffer<int>(*ids_tensor);
  ASSERT_TRUE(ids.HasValue());
  // The sampled ids are 2, 0, 1, 3.
  EXPECT_THAT(*ids, ElementsAre(2, 0, 1, 3));

  auto scores = CopyFromTensorBuffer<float>(*scores_tensor);
  ASSERT_TRUE(scores.HasValue());
  // The scores are the log of the probability of the sampled token.
  EXPECT_THAT(
      *scores,
      ElementsAre(
          FloatNear(-0.00013619f, 1e-5), FloatNear(-5.010385e-05f, 1e-5),
          FloatNear(-1.843247e-05f, 1e-5), FloatNear(-6.780965e-06f, 1e-5)));
}

TEST(TopPSamplerTest, UpdateConfig) {
  auto sampler_or = TopPSampler::Create(/*k=*/1, /*p=*/0.5, /*temperature=*/1.0,
                                        /*batch_size=*/1, /*sequence_size=*/1,
                                        /*seed=*/2);
  ASSERT_TRUE(sampler_or.ok());
  auto sampler = *std::move(sampler_or);

  proto::SamplerParameters sampler_params;
  sampler_params.set_k(1);
  sampler_params.set_p(1.0);
  sampler_params.set_temperature(100.0);

  auto status = sampler->UpdateConfig(sampler_params,
                                      /*batch_size=*/1, nullptr);
  ASSERT_TRUE(status.ok());

  const std::vector<float> logits = {0.0, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0};
  auto logits_tensor = CopyToTensorBuffer<float>(logits, {1, 8});

  std::vector<int> ids_vector = {0, 1, 2, 3, 4, 5, 6, 7};
  auto ids_tensor =
      CopyToTensorBuffer<int>(absl::MakeConstSpan(ids_vector), {1, 8});
  ASSERT_TRUE(ids_tensor.HasValue());

  status = sampler->SampleToIdAndScoreBuffer(*logits_tensor, ids_tensor.Value(),
                                             /*scores_tensor=*/nullptr);
  ASSERT_TRUE(status.ok());

  auto ids = CopyFromTensorBuffer<int>(ids_tensor.Value());
  ASSERT_TRUE(ids.HasValue());
  // With topk=1, it should pick the argmax.
  EXPECT_EQ(ids.Value()[0], 4);

  // Update the config again.
  sampler_params.set_k(8);

  status = sampler->UpdateConfig(sampler_params, /*batch_size=*/1, nullptr);
  ASSERT_TRUE(status.ok());

  status = sampler->SampleToIdAndScoreBuffer(*logits_tensor, ids_tensor.Value(),
                                             /*scores_tensor=*/nullptr);
  ASSERT_TRUE(status.ok());

  ids = CopyFromTensorBuffer<int>(ids_tensor.Value());

  // With topk=8 and high temperature, it might pick a random token. With
  // seed=2, it should consistently not pick 4.
  EXPECT_NE(ids.Value()[0], 4);
}

}  // namespace
}  // namespace litert::lm
