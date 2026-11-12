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

#include "runtime/executor/gpu_artisan/gpu_artisan_executor_utils.h"

#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/executor/gpu_artisan/proto/sampler_params.pb.h"
#include "runtime/proto/sampler_params.pb.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/test_utils.h"

namespace odml::infra {
namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;

TEST(GpuArtisanExecutorUtilsTest,
     CreateMLDriftTensorEmbeddingFromLitertTensorBuffer) {
  // Prepare input TensorBuffer
  std::vector<float> embedding_data = {1.0f,  2.0f, 3.0f, 4.0f, 5.0f,
                                       6.0f,  7.0f, 8.0f, 9.0f, 10.0f,
                                       11.0f, 12.0f};  // Shape [1, 1, 3, 4]
  auto tensor_buffer_status = ::litert::lm::CopyToTensorBuffer<float>(
      absl::MakeConstSpan(embedding_data), {1, 1, 3, 4});
  EXPECT_EQ(tensor_buffer_status.HasValue(), true);
  auto tensor_buffer = std::move(tensor_buffer_status.Value());

  // Call the function under test
  ASSERT_OK_AND_ASSIGN(
      ml_drift::TensorFloat32 result_tensor,
      CreateMLDriftTensorEmbeddingFromLitertTensorBuffer(tensor_buffer));

  // Verify the result
  EXPECT_EQ(result_tensor.shape, ml_drift::BHWC(1, 1, 3, 4));
  EXPECT_THAT(result_tensor.spanned_data, ElementsAreArray(embedding_data));
}

TEST(GpuArtisanExecutorUtilsTest,
     CreateMLDriftTensorPLEFromLitertTensorBuffer) {
  // Prepare input TensorBuffer
  std::vector<float> ple_data = {
      1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,  9.0f,
      10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f,
      19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f};  // Shape [1, 2, 3, 4]
  auto tensor_buffer_status = ::litert::lm::CopyToTensorBuffer<float>(
      absl::MakeConstSpan(ple_data), {1, 2, 3, 4});
  EXPECT_EQ(tensor_buffer_status.HasValue(), true);
  auto tensor_buffer = std::move(tensor_buffer_status.Value());

  // Call the function under test
  ASSERT_OK_AND_ASSIGN(
      std::vector<ml_drift::TensorFloat32> result_vec,
      CreateMLDriftTensorPLEFromLitertTensorBuffer(tensor_buffer));

  // Verify the result
  ASSERT_EQ(result_vec.size(), 2);

  // Verify first tensor in the vector
  EXPECT_EQ(result_vec[0].shape, ml_drift::BHWC(1, 1, 3, 4));
  EXPECT_THAT(result_vec[0].spanned_data,
              ElementsAreArray({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                                9.0f, 10.0f, 11.0f, 12.0f}));

  // Verify second tensor in the vector
  EXPECT_EQ(result_vec[1].shape, ml_drift::BHWC(1, 1, 3, 4));
  EXPECT_THAT(result_vec[1].spanned_data,
              ElementsAreArray({13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f,
                                20.0f, 21.0f, 22.0f, 23.0f, 24.0f}));
}

TEST(GpuArtisanExecutorUtilsTest,
     CreateMLDriftTensorEmbeddingFromLitertTensorBuffer_3DTensor) {
  // Prepare input TensorBuffer
  std::vector<float> embedding_data = {1.0f,  2.0f, 3.0f, 4.0f, 5.0f,
                                       6.0f,  7.0f, 8.0f, 9.0f, 10.0f,
                                       11.0f, 12.0f};  // Shape [1, 3, 4]
  auto tensor_buffer_status = ::litert::lm::CopyToTensorBuffer<float>(
      absl::MakeConstSpan(embedding_data), {1, 3, 4});
  EXPECT_EQ(tensor_buffer_status.HasValue(), true);
  auto tensor_buffer = std::move(tensor_buffer_status.Value());

  // Call the function under test
  ASSERT_OK_AND_ASSIGN(
      ml_drift::TensorFloat32 result_tensor,
      CreateMLDriftTensorEmbeddingFromLitertTensorBuffer(tensor_buffer));

  // Verify the result
  EXPECT_EQ(result_tensor.shape, ml_drift::BHWC(1, 1, 3, 4));
  EXPECT_THAT(result_tensor.spanned_data, ElementsAreArray(embedding_data));
}

TEST(GpuArtisanExecutorUtilsTest, GetImageTokenInfos) {
  // Case 1: Normal images with kEndToken.
  const std::vector<int> tokens1 = {101, -1, -1, -3, 102, -1, -1, -1, -3, 103};
  auto res1 = GetImageTokenInfos(tokens1, /*vision_token_id=*/-1,
                                 /*vision_end_token_id=*/-3);
  ASSERT_EQ(res1.size(), 2);
  EXPECT_EQ(res1[0].active_tokens, 2);
  EXPECT_EQ(res1[0].has_eoi, true);
  EXPECT_EQ(res1[0].total_tokens(), 3);

  EXPECT_EQ(res1[1].active_tokens, 3);
  EXPECT_EQ(res1[1].has_eoi, true);
  EXPECT_EQ(res1[1].total_tokens(), 4);

  // Case 2: Normal images without kEndToken.
  const std::vector<int> tokens2 = {101, -1, -1, 102, -1, -1, -1, 103};
  auto res2 = GetImageTokenInfos(tokens2, /*vision_token_id=*/-1,
                                 /*vision_end_token_id=*/-3);
  ASSERT_EQ(res2.size(), 2);
  EXPECT_EQ(res2[0].active_tokens, 2);
  EXPECT_EQ(res2[0].has_eoi, false);
  EXPECT_EQ(res2[0].total_tokens(), 2);

  EXPECT_EQ(res2[1].active_tokens, 3);
  EXPECT_EQ(res2[1].has_eoi, false);
  EXPECT_EQ(res2[1].total_tokens(), 3);

  // Case 3: String ending with special tokens (no trailing text).
  const std::vector<int> tokens3 = {101, -1, -1, -1};
  auto res3 = GetImageTokenInfos(tokens3, /*vision_token_id=*/-1,
                                 /*vision_end_token_id=*/-3);
  ASSERT_EQ(res3.size(), 1);
  EXPECT_EQ(res3[0].active_tokens, 3);
  EXPECT_EQ(res3[0].has_eoi, false);
  EXPECT_EQ(res3[0].total_tokens(), 3);

  // Case 4: Empty range.
  const std::vector<int> tokens4 = {};
  EXPECT_THAT(GetImageTokenInfos(tokens4, /*vision_token_id=*/-1,
                                 /*vision_end_token_id=*/-3),
              ::testing::IsEmpty());

  // Case 5: No special tokens.
  const std::vector<int> tokens5 = {101, 102, 103};
  EXPECT_THAT(GetImageTokenInfos(tokens5, /*vision_token_id=*/-1,
                                 /*vision_end_token_id=*/-3),
              ::testing::IsEmpty());
}

MATCHER_P6(ChunkIs, token_begin, token_end, image_begin, image_end,
           audio_token_begin, audio_token_end, "") {
  return arg.token_begin == token_begin && arg.token_end == token_end &&
         arg.image_begin == image_begin && arg.image_end == image_end &&
         arg.audio_token_begin == audio_token_begin &&
         arg.audio_token_end == audio_token_end;
}

constexpr int kVision = -1;
constexpr int kAudio = -2;
constexpr int kVisionEnd = -3;
constexpr int kAudioEnd = -4;
constexpr int kVisionTokensPerImage = 3;

std::vector<PrefillChunkRange> Split(const std::vector<int>& tokens,
                                     int max_images_per_chunk,
                                     int max_num_audio_tokens = 0) {
  return SplitTokensForPrefill(tokens, kVision, kAudio, kAudioEnd,
                               max_images_per_chunk, kVisionTokensPerImage,
                               max_num_audio_tokens);
}

TEST(SplitTokensForPrefillTest, NoSplitNeeded) {
  EXPECT_THAT(Split({101, kVision, kVision, kVisionEnd, 102, kVision, kVision,
                     kVisionEnd, 103},
                    /*max_images_per_chunk=*/2),
              ElementsAre(ChunkIs(0, 9, 0, 2, 0, 0)));
}

TEST(SplitTokensForPrefillTest, SplitsAtImageStart) {
  // Three images: [1, 3], [5, 7], [9, 10] (end-of-image tokens included).
  const std::vector<int> tokens = {101, kVision, kVision,    kVisionEnd,
                                   102, kVision, kVision,    kVisionEnd,
                                   103, kVision, kVisionEnd, 104};
  EXPECT_THAT(Split(tokens, /*max_images_per_chunk=*/1),
              ElementsAre(ChunkIs(0, 5, 0, 1, 0, 0), ChunkIs(5, 9, 1, 2, 0, 0),
                          ChunkIs(9, 12, 2, 3, 0, 0)));
  EXPECT_THAT(
      Split(tokens, /*max_images_per_chunk=*/2),
      ElementsAre(ChunkIs(0, 9, 0, 2, 0, 0), ChunkIs(9, 12, 2, 3, 0, 0)));
}

TEST(SplitTokensForPrefillTest, AdjacentImagesWithoutEndToken) {
  EXPECT_THAT(
      Split({kVision, kVision, 101, kVision, kVision},
            /*max_images_per_chunk=*/1),
      ElementsAre(ChunkIs(0, 3, 0, 1, 0, 0), ChunkIs(3, 5, 1, 2, 0, 0)));
}

TEST(SplitTokensForPrefillTest, NoImageLimit) {
  EXPECT_THAT(Split({kVision, 101, kVision, 102, kVision},
                    /*max_images_per_chunk=*/0),
              ElementsAre(ChunkIs(0, 5, 0, 3, 0, 0)));
}

TEST(SplitTokensForPrefillTest, EmptyTokens) {
  EXPECT_THAT(Split({}, /*max_images_per_chunk=*/1),
              ElementsAre(ChunkIs(0, 0, 0, 0, 0, 0)));
}

TEST(SplitTokensForPrefillTest, EachAudioClipIsInItsOwnChunk) {
  // Audio clips: [1, 2], [4, 4], [6, 8].
  EXPECT_THAT(
      Split({101, kAudio, kAudio, 102, kAudio, 103, kAudio, kAudio, kAudio},
            /*max_images_per_chunk=*/0),
      ElementsAre(ChunkIs(0, 4, 0, 0, 0, 2), ChunkIs(4, 6, 0, 0, 2, 3),
                  ChunkIs(6, 9, 0, 0, 3, 6)));
}

TEST(SplitTokensForPrefillTest, ChunkWithAudioHasRoomForFewerImages) {
  // Images: [0, 1], [3, 4], [9, 10]. Audio: [6, 7]. With room for 2 images
  // (6 tokens), the 2 audio tokens leave room for 1 image.
  EXPECT_THAT(
      Split({kVision, kVisionEnd, 101, kVision, kVisionEnd, 102, kAudio, kAudio,
             103, kVision, kVisionEnd, 104},
            /*max_images_per_chunk=*/2),
      ElementsAre(ChunkIs(0, 6, 0, 2, 0, 0), ChunkIs(6, 12, 2, 3, 0, 2)));
}

TEST(SplitTokensForPrefillTest, ImageAndAudioThatDoNotFitAreSplit) {
  // With room for 1 image (3 tokens), an audio clip leaves no room for images.
  EXPECT_THAT(
      Split({kVision, kVisionEnd, kAudio, 101},
            /*max_images_per_chunk=*/1),
      ElementsAre(ChunkIs(0, 2, 0, 1, 0, 0), ChunkIs(2, 4, 1, 1, 0, 1)));
  EXPECT_THAT(
      Split({kAudio, kAudio, 101, kVision, kVisionEnd},
            /*max_images_per_chunk=*/1),
      ElementsAre(ChunkIs(0, 3, 0, 0, 0, 2), ChunkIs(3, 5, 0, 1, 2, 2)));
}

TEST(SplitTokensForPrefillTest, ImagesBetweenAudioClips) {
  EXPECT_THAT(Split({kAudio, 101, kVision, kVisionEnd, kAudio, 102},
                    /*max_images_per_chunk=*/1),
              ElementsAre(ChunkIs(0, 2, 0, 0, 0, 1), ChunkIs(2, 4, 0, 1, 1, 1),
                          ChunkIs(4, 6, 1, 1, 1, 2)));
}

TEST(SplitTokensForPrefillTest, EndOfAudioTokenCountsTowardsTheAudioClip) {
  // With room for 2 images (6 tokens), 3 audio tokens leave room for 1 image,
  // but not with the end-of-audio token.
  EXPECT_THAT(Split({kVision, kVisionEnd, kAudio, kAudio, kAudio, 101},
                    /*max_images_per_chunk=*/2),
              ElementsAre(ChunkIs(0, 6, 0, 1, 0, 3)));
  EXPECT_THAT(
      Split({kVision, kVisionEnd, kAudio, kAudio, kAudio, kAudioEnd, 101},
            /*max_images_per_chunk=*/2),
      ElementsAre(ChunkIs(0, 2, 0, 1, 0, 0), ChunkIs(2, 7, 1, 1, 0, 3)));
}

TEST(SplitTokensForPrefillTest, ReservedAudioRoomFitsImagesWithAudio) {
  // With room for 4 audio embeddings after the image slots, a chunk can have
  // `max_images_per_chunk` images and an audio clip of up to 4 tokens.
  EXPECT_THAT(Split({kVision, kVisionEnd, kAudio, kAudio, kAudioEnd, 101,
                     kVision, kVisionEnd},
                    /*max_images_per_chunk=*/2, /*max_num_audio_tokens=*/4),
              ElementsAre(ChunkIs(0, 8, 0, 2, 0, 2)));
  // Two audio clips are still prefilled separately.
  EXPECT_THAT(
      Split({kAudio, kAudioEnd, 101, kAudio, kAudioEnd},
            /*max_images_per_chunk=*/2, /*max_num_audio_tokens=*/4),
      ElementsAre(ChunkIs(0, 3, 0, 0, 0, 1), ChunkIs(3, 5, 0, 0, 1, 2)));
}

TEST(GpuArtisanExecutorUtilsTest, ConvertToOdmlSamplerParams) {
  ::litert::lm::proto::SamplerParameters litert_params;
  litert_params.set_type(::litert::lm::proto::SamplerParameters::TOP_P);
  litert_params.set_k(40);
  litert_params.set_p(0.95f);
  litert_params.set_temperature(0.7f);
  litert_params.set_seed(42);

  proto::SamplerParameters odml_params =
      ConvertToOdmlSamplerParams(litert_params);
  EXPECT_EQ(odml_params.type(), proto::SamplerParameters::TOP_P);
  EXPECT_EQ(odml_params.k(), 40);
  EXPECT_FLOAT_EQ(odml_params.p(), 0.95f);
  EXPECT_FLOAT_EQ(odml_params.temperature(), 0.7f);
  EXPECT_TRUE(odml_params.has_seed());
  EXPECT_EQ(odml_params.seed(), 42);

  litert_params.set_type(::litert::lm::proto::SamplerParameters::TOP_K);
  litert_params.clear_seed();
  odml_params = ConvertToOdmlSamplerParams(litert_params);
  EXPECT_EQ(odml_params.type(), proto::SamplerParameters::TOP_K);
  EXPECT_FALSE(odml_params.has_seed());

  litert_params.set_type(::litert::lm::proto::SamplerParameters::GREEDY);
  odml_params = ConvertToOdmlSamplerParams(litert_params);
  EXPECT_EQ(odml_params.type(), proto::SamplerParameters::GREEDY);

  litert_params.set_type(
      ::litert::lm::proto::SamplerParameters::TYPE_UNSPECIFIED);
  odml_params = ConvertToOdmlSamplerParams(litert_params);
  EXPECT_EQ(odml_params.type(), proto::SamplerParameters::TYPE_UNSPECIFIED);
}

}  // namespace
}  // namespace odml::infra
