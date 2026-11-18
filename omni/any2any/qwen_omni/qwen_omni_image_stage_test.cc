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

#include "omni/any2any/qwen_omni/qwen_omni_image_stage.h"

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "omni/any2any/any2any_input_fan_out_stage.h"
#include "omni/any2any/any2any_test_utils.h"
#include "omni/any2any/qwen_omni/qwen_omni_io_types.h"
#include "omni/any2any/qwen_omni/qwen_omni_model_config.h"
#include "omni/any2any/qwen_omni/qwen_omni_test_utils.h"
#include "omni/base/fan_out_stage.h"
#include "omni/base/io_types.h"
#include "omni/base/litert_runner.h"
#include "omni/base/mock_litert_runner.h"
#include "omni/base/model_utils.h"
#include "omni/omni_session.h"
#include "runtime/components/preprocessor/pil_resize.h"
#include "runtime/executor/executor_settings_base.h"
#include "support/util/convert_tensor_buffer.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep
#include "tflite/schema/schema_generated.h"  // from @litert

namespace litert::omni::any2any {
namespace {

using ::absl_testing::StatusIs;
using ::testing::_;
using ::testing::ElementsAre;
using ::testing::FloatNear;
using ::testing::NiceMock;

constexpr float kQwenClipMean[3] = {0.48145466f, 0.4578275f, 0.40821073f};
constexpr float kQwenClipStd[3] = {0.26862954f, 0.26130258f, 0.27577711f};

// Valid 2x2 RGB PNG image:
// (0,0)=(255,0,0), (1,0)=(0,255,0), (0,1)=(0,0,255), (1,1)=(255,255,255).
constexpr uint8_t kValid2x2RgbPng[] = {
    0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00,
    0x0d, 0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00,
    0x00, 0x02, 0x08, 0x02, 0x00, 0x00, 0x00, 0xfd, 0xd4, 0x9a, 0x73,
    0x00, 0x00, 0x00, 0x12, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9c, 0x63,
    0xf8, 0xcf, 0xc0, 0xc0, 0x00, 0xc2, 0x0c, 0xff, 0x81, 0x00, 0x00,
    0x1f, 0xee, 0x05, 0xfb, 0x0b, 0xd9, 0x68, 0x8b, 0x00, 0x00, 0x00,
    0x00, 0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82,
};

std::unique_ptr<NiceMock<MockLiteRtRunner>> MakeMockVisionRunner(
    int image_size, int num_vision_tokens, int hidden_dim) {
  auto runner = std::make_unique<NiceMock<MockLiteRtRunner>>();
  ON_CALL(*runner, CreateInputBuffers(absl::string_view("")))
      .WillByDefault([image_size](absl::string_view)
                         -> absl::StatusOr<std::vector<TensorBuffer>> {
        LITERT_ASSIGN_OR_RETURN(auto buf, support::CreateTensorBuffer<float>(
                                              {1, 3, image_size, image_size}));
        std::vector<TensorBuffer> bufs;
        bufs.push_back(std::move(buf));
        return bufs;
      });
  ON_CALL(*runner, CreateOutputBuffers(absl::string_view("")))
      .WillByDefault([num_vision_tokens, hidden_dim](absl::string_view)
                         -> absl::StatusOr<std::vector<TensorBuffer>> {
        LITERT_ASSIGN_OR_RETURN(auto buf,
                                support::CreateTensorBuffer<float>(
                                    {1, num_vision_tokens, hidden_dim}));
        std::vector<TensorBuffer> bufs;
        bufs.push_back(std::move(buf));
        return bufs;
      });
  ON_CALL(*runner, Run(absl::string_view(""), _, _))
      .WillByDefault(
          [num_vision_tokens, hidden_dim](
              absl::string_view, absl::Span<const TensorBuffer>,
              absl::Span<const TensorBuffer> outputs) -> absl::Status {
            std::vector<float> out(
                static_cast<size_t>(num_vision_tokens) * hidden_dim, 0.5f);
            auto& out_buf = const_cast<TensorBuffer&>(outputs[0]);
            LITERT_RETURN_IF_ERROR(
                out_buf.Write<float>(absl::MakeConstSpan(out)));
            return absl::OkStatus();
          });
  return runner;
}

TEST(QwenOmniImageStageTest, ImageStagePreprocessesRawImagesAndFramesTokens) {
  // Arrange
  auto input_source = std::make_unique<PushInputSource>();
  auto fan_out = std::make_unique<Any2AnyInputFanOutStage>(input_source.get());
  QwenOmniModelConfig config =
      MakeSmallQwenOmniTestConfig(/*generate_audio=*/false);

  auto env = ::litert::Environment::Create({});
  ASSERT_TRUE(env.HasValue());
  ModelOptions opts{.backend = lm::Backend::CPU, .num_threads = 1};

  std::string vision_buf = BuildMinimalTfliteModelBuffer(
      {{"pixel_values", {1, 3, 28, 28}, tflite::TensorType_FLOAT32}},
      {{"vision_embeds", {1, 2, 4}, tflite::TensorType_FLOAT32}});
  ASSERT_OK_AND_ASSIGN(
      auto vision_model,
      CreateCompiledModelFromBuffer(*env, opts, vision_buf, "vision_encoder"));
  ASSERT_OK_AND_ASSIGN(
      auto image_stage,
      QwenOmniImageStage::Create(
          &fan_out->ImageQueue(),
          std::make_unique<LiteRtRunnerImpl>(&vision_model), config));

  // Act: push RGB, RGBA, Grayscale, and empty images.
  ASSERT_OK(input_source->PushInput(ImageInput{
      .image_bytes = std::vector<uint8_t>(4 * 4 * 3, 200),
      .width = 4,
      .height = 4,
      .channels = 3,
  }));
  ASSERT_OK(input_source->PushInput(ImageInput{
      .image_bytes = std::vector<uint8_t>(2 * 2 * 4, 100),
      .width = 2,
      .height = 2,
      .channels = 4,
  }));
  ASSERT_OK(input_source->PushInput(ImageInput{
      .image_bytes = {10, 20, 30, 40},
      .width = 2,
      .height = 2,
      .channels = 1,
  }));
  ASSERT_OK(input_source->PushInput(ImageInput{}));
  while (fan_out->NeedSchedule()) {
    ASSERT_OK(fan_out->Schedule());
  }
  ASSERT_OK(image_stage->Schedule());

  // Assert
  for (int i = 0; i < 3; ++i) {
    ASSERT_OK_AND_ASSIGN(auto img_seg, image_stage->GetOutput());
    EXPECT_EQ(img_seg.item.modality,
              QwenOmniEmbeddingSegment::Modality::kVision);
    EXPECT_EQ(img_seg.item.num_tokens, 2);
    EXPECT_EQ(img_seg.item.embeddings.size(), 8);
    EXPECT_THAT(img_seg.item.token_ids,
                ElementsAre(config.vision_bos_token_id, config.image_token_id,
                            config.image_token_id, config.vision_eos_token_id));
  }
  ASSERT_OK_AND_ASSIGN(auto empty_seg, image_stage->GetOutput());
  EXPECT_EQ(empty_seg.item.num_tokens, 0);
  EXPECT_TRUE(empty_seg.item.token_ids.empty());

  // Invalid byte count is rejected.
  ASSERT_OK(input_source->PushInput(ImageInput{
      .image_bytes = {1, 2, 3}, .width = 4, .height = 4, .channels = 3}));
  ASSERT_OK(fan_out->Schedule());
  EXPECT_THAT(image_stage->Schedule(),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(QwenOmniImageStageTest, CreateValidatesConfigAndRunnerBuffers) {
  SequencedVariantQueue<OmniSession::Input, ImageInput> image_queue;
  QwenOmniModelConfig config =
      MakeSmallQwenOmniTestConfig(/*generate_audio=*/false);

  QwenOmniModelConfig bad_hidden = config;
  bad_hidden.llm_hidden_size = 0;
  EXPECT_THAT(QwenOmniImageStage::Create(
                  &image_queue, MakeMockVisionRunner(28, 2, 4), bad_hidden),
              StatusIs(absl::StatusCode::kInvalidArgument));

  QwenOmniModelConfig bad_patch = config;
  bad_patch.vision_patch_size = 0;
  EXPECT_THAT(QwenOmniImageStage::Create(
                  &image_queue, MakeMockVisionRunner(28, 2, 4), bad_patch),
              StatusIs(absl::StatusCode::kInvalidArgument));

  QwenOmniModelConfig bad_merge = config;
  bad_merge.vision_spatial_merge_size = 0;
  EXPECT_THAT(QwenOmniImageStage::Create(
                  &image_queue, MakeMockVisionRunner(28, 2, 4), bad_merge),
              StatusIs(absl::StatusCode::kInvalidArgument));

  auto empty_runner = std::make_unique<NiceMock<MockLiteRtRunner>>();
  ON_CALL(*empty_runner, CreateInputBuffers(absl::string_view("")))
      .WillByDefault(
          [](absl::string_view) { return std::vector<TensorBuffer>{}; });
  ON_CALL(*empty_runner, CreateOutputBuffers(absl::string_view("")))
      .WillByDefault(
          [](absl::string_view) { return std::vector<TensorBuffer>{}; });
  EXPECT_THAT(
      QwenOmniImageStage::Create(&image_queue, std::move(empty_runner), config),
      StatusIs(absl::StatusCode::kInvalidArgument));

  // Non-Float32 input buffer is rejected even with matching shape [1, 3, 28,
  // 28].
  auto int32_in = MakeMockVisionRunner(28, 2, config.llm_hidden_size);
  ON_CALL(*int32_in, CreateInputBuffers(absl::string_view("")))
      .WillByDefault(
          [](absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
            LITERT_ASSIGN_OR_RETURN(
                auto buf, support::CreateTensorBuffer<int32_t>({1, 3, 28, 28}));
            std::vector<TensorBuffer> bufs;
            bufs.push_back(std::move(buf));
            return bufs;
          });
  EXPECT_THAT(
      QwenOmniImageStage::Create(&image_queue, std::move(int32_in), config),
      StatusIs(absl::StatusCode::kInvalidArgument));

  // Wrong rank ([1, 3, 784]).
  auto wrong_rank_in = MakeMockVisionRunner(28, 2, config.llm_hidden_size);
  ON_CALL(*wrong_rank_in, CreateInputBuffers(absl::string_view("")))
      .WillByDefault(
          [](absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
            LITERT_ASSIGN_OR_RETURN(
                auto buf, support::CreateTensorBuffer<float>({1, 3, 784}));
            std::vector<TensorBuffer> bufs;
            bufs.push_back(std::move(buf));
            return bufs;
          });
  EXPECT_THAT(QwenOmniImageStage::Create(&image_queue, std::move(wrong_rank_in),
                                         config),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // Batch size != 1 ([2, 3, 28, 28]).
  auto bad_batch_in = MakeMockVisionRunner(28, 2, config.llm_hidden_size);
  ON_CALL(*bad_batch_in, CreateInputBuffers(absl::string_view("")))
      .WillByDefault(
          [](absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
            LITERT_ASSIGN_OR_RETURN(
                auto buf, support::CreateTensorBuffer<float>({2, 3, 28, 28}));
            std::vector<TensorBuffer> bufs;
            bufs.push_back(std::move(buf));
            return bufs;
          });
  EXPECT_THAT(
      QwenOmniImageStage::Create(&image_queue, std::move(bad_batch_in), config),
      StatusIs(absl::StatusCode::kInvalidArgument));

  // NHWC layout ([1, 28, 28, 3]) is rejected.
  auto nhwc_in = MakeMockVisionRunner(28, 2, config.llm_hidden_size);
  ON_CALL(*nhwc_in, CreateInputBuffers(absl::string_view("")))
      .WillByDefault(
          [](absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
            LITERT_ASSIGN_OR_RETURN(
                auto buf, support::CreateTensorBuffer<float>({1, 28, 28, 3}));
            std::vector<TensorBuffer> bufs;
            bufs.push_back(std::move(buf));
            return bufs;
          });
  EXPECT_THAT(
      QwenOmniImageStage::Create(&image_queue, std::move(nhwc_in), config),
      StatusIs(absl::StatusCode::kInvalidArgument));

  // Non-square spatial dimensions whose product is a perfect square
  // (14 * 56 == 28 * 28) are rejected.
  auto non_square_in = MakeMockVisionRunner(28, 2, config.llm_hidden_size);
  ON_CALL(*non_square_in, CreateInputBuffers(absl::string_view("")))
      .WillByDefault(
          [](absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
            LITERT_ASSIGN_OR_RETURN(
                auto buf, support::CreateTensorBuffer<float>({1, 3, 14, 56}));
            std::vector<TensorBuffer> bufs;
            bufs.push_back(std::move(buf));
            return bufs;
          });
  EXPECT_THAT(QwenOmniImageStage::Create(&image_queue, std::move(non_square_in),
                                         config),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // Square image_size not divisible by vision_patch_size *
  // vision_spatial_merge_size (30 % 28 != 0) is rejected.
  EXPECT_THAT(
      QwenOmniImageStage::Create(
          &image_queue,
          MakeMockVisionRunner(/*image_size=*/30,
                               /*num_vision_tokens=*/2, config.llm_hidden_size),
          config),
      StatusIs(absl::StatusCode::kInvalidArgument));

  // Output buffer byte size not divisible by llm_hidden_size * sizeof(float).
  EXPECT_THAT(
      QwenOmniImageStage::Create(
          &image_queue, MakeMockVisionRunner(28, 1, /*hidden_dim=*/3), config),
      StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(QwenOmniImageStageTest,
     DecodesValidEncodedImageResizesWithPilBicubicAndNormalizesToNchw) {
  SequencedVariantQueue<OmniSession::Input, ImageInput> image_queue;
  QwenOmniModelConfig config =
      MakeSmallQwenOmniTestConfig(/*generate_audio=*/false);
  config.vision_patch_size = 1;
  config.vision_spatial_merge_size = 2;

  // Case 1: Same-size encoded 2x2 PNG (2 == image_size, skips resizing).
  std::vector<float> captured_nchw;
  auto runner_2x2 =
      MakeMockVisionRunner(/*image_size=*/2,
                           /*num_vision_tokens=*/2, config.llm_hidden_size);
  ON_CALL(*runner_2x2, Run(absl::string_view(""), _, _))
      .WillByDefault(
          [&captured_nchw, hidden_dim = config.llm_hidden_size](
              absl::string_view, absl::Span<const TensorBuffer> inputs,
              absl::Span<const TensorBuffer> outputs) -> absl::Status {
            captured_nchw.assign(3 * 2 * 2, 0.0f);
            auto& in_buf = const_cast<TensorBuffer&>(inputs[0]);
            LITERT_RETURN_IF_ERROR(
                in_buf.Read<float>(absl::MakeSpan(captured_nchw)));
            std::vector<float> out(static_cast<size_t>(2) * hidden_dim, 0.5f);
            auto& out_buf = const_cast<TensorBuffer&>(outputs[0]);
            LITERT_RETURN_IF_ERROR(
                out_buf.Write<float>(absl::MakeConstSpan(out)));
            return absl::OkStatus();
          });

  ASSERT_OK_AND_ASSIGN(
      auto stage_2x2,
      QwenOmniImageStage::Create(&image_queue, std::move(runner_2x2), config));

  ASSERT_OK(image_queue.PushBack(
      0, ImageInput{
             .image_bytes = std::vector<uint8_t>(std::begin(kValid2x2RgbPng),
                                                 std::end(kValid2x2RgbPng)),
             .width = 0,
             .height = 0,
             .channels = 0,
         }));
  EXPECT_TRUE(stage_2x2->NeedSchedule());
  ASSERT_OK(stage_2x2->Schedule());

  ASSERT_OK_AND_ASSIGN(auto seg, stage_2x2->GetOutput());
  EXPECT_EQ(seg.seqnum, 0);
  EXPECT_EQ(seg.item.modality, QwenOmniEmbeddingSegment::Modality::kVision);
  EXPECT_EQ(seg.item.num_tokens, 2);
  ASSERT_EQ(captured_nchw.size(), 12);
  // Pixel (0,0) is pure red (255, 0, 0).
  EXPECT_THAT(captured_nchw[0],
              FloatNear((1.0f - kQwenClipMean[0]) / kQwenClipStd[0], 1e-5f));
  EXPECT_THAT(captured_nchw[4],
              FloatNear((0.0f - kQwenClipMean[1]) / kQwenClipStd[1], 1e-5f));
  EXPECT_THAT(captured_nchw[8],
              FloatNear((0.0f - kQwenClipMean[2]) / kQwenClipStd[2], 1e-5f));
  // Pixel (1,1) is pure white (255, 255, 255).
  EXPECT_THAT(captured_nchw[3],
              FloatNear((1.0f - kQwenClipMean[0]) / kQwenClipStd[0], 1e-5f));
  EXPECT_THAT(captured_nchw[7],
              FloatNear((1.0f - kQwenClipMean[1]) / kQwenClipStd[1], 1e-5f));
  EXPECT_THAT(captured_nchw[11],
              FloatNear((1.0f - kQwenClipMean[2]) / kQwenClipStd[2], 1e-5f));

  // Invalid encoded bytes return an error.
  ASSERT_OK(image_queue.PushBack(1, ImageInput{
                                        .image_bytes = {0x00, 0x01, 0x02, 0x03},
                                        .width = 0,
                                        .height = 0,
                                        .channels = 0,
                                    }));
  EXPECT_FALSE(stage_2x2->Schedule().ok());
  ASSERT_OK(image_queue.PopFront());

  // Case 2: Downscaling non-uniform 6x4 RGB image to 4x4 matches
  // `lm::PilResizeBicubicRgb` + Qwen CLIP NCHW normalization.
  constexpr int kSrcW = 6;
  constexpr int kSrcH = 4;
  constexpr int kDstSize = 4;
  std::vector<uint8_t> src_rgb(kSrcW * kSrcH * 3);
  for (int y = 0; y < kSrcH; ++y) {
    for (int x = 0; x < kSrcW; ++x) {
      const size_t idx = (static_cast<size_t>(y) * kSrcW + x) * 3;
      src_rgb[idx] = static_cast<uint8_t>((x * 40 + y * 15) % 256);
      src_rgb[idx + 1] = static_cast<uint8_t>((x * 20 + y * 50) % 256);
      src_rgb[idx + 2] = static_cast<uint8_t>((x * 10 + y * 30 + 70) % 256);
    }
  }
  const std::vector<uint8_t> expected_resized =
      lm::PilResizeBicubicRgb(src_rgb.data(), kSrcW, kSrcH, kDstSize, kDstSize);

  std::vector<float> captured_resized_nchw;
  auto runner_4x4 =
      MakeMockVisionRunner(/*image_size=*/kDstSize,
                           /*num_vision_tokens=*/2, config.llm_hidden_size);
  ON_CALL(*runner_4x4, Run(absl::string_view(""), _, _))
      .WillByDefault(
          [&captured_resized_nchw, hidden_dim = config.llm_hidden_size](
              absl::string_view, absl::Span<const TensorBuffer> inputs,
              absl::Span<const TensorBuffer> outputs) -> absl::Status {
            captured_resized_nchw.assign(3 * kDstSize * kDstSize, 0.0f);
            auto& in_buf = const_cast<TensorBuffer&>(inputs[0]);
            LITERT_RETURN_IF_ERROR(
                in_buf.Read<float>(absl::MakeSpan(captured_resized_nchw)));
            std::vector<float> out(static_cast<size_t>(2) * hidden_dim, 0.5f);
            auto& out_buf = const_cast<TensorBuffer&>(outputs[0]);
            LITERT_RETURN_IF_ERROR(
                out_buf.Write<float>(absl::MakeConstSpan(out)));
            return absl::OkStatus();
          });

  ASSERT_OK_AND_ASSIGN(
      auto stage_4x4,
      QwenOmniImageStage::Create(&image_queue, std::move(runner_4x4), config));
  ASSERT_OK(image_queue.PushBack(2, ImageInput{
                                        .image_bytes = src_rgb,
                                        .width = kSrcW,
                                        .height = kSrcH,
                                        .channels = 3,
                                    }));
  ASSERT_OK(stage_4x4->Schedule());
  ASSERT_OK(stage_4x4->GetOutput());

  constexpr size_t kPlaneSize = kDstSize * kDstSize;
  ASSERT_EQ(captured_resized_nchw.size(), 3 * kPlaneSize);
  for (size_t i = 0; i < kPlaneSize; ++i) {
    for (int c = 0; c < 3; ++c) {
      const float expected =
          (expected_resized[i * 3 + c] / 255.0f - kQwenClipMean[c]) /
          kQwenClipStd[c];
      EXPECT_THAT(captured_resized_nchw[c * kPlaneSize + i],
                  FloatNear(expected, 1e-5f));
    }
  }
}

TEST(QwenOmniImageStageTest,
     RejectsInvalidDimensionsChannelsOverflowEmptyScheduleAndResets) {
  SequencedVariantQueue<OmniSession::Input, ImageInput> image_queue;
  QwenOmniModelConfig config =
      MakeSmallQwenOmniTestConfig(/*generate_audio=*/false);
  config.vision_patch_size = 2;
  config.vision_spatial_merge_size = 2;
  ASSERT_OK_AND_ASSIGN(
      auto stage,
      QwenOmniImageStage::Create(
          &image_queue, MakeMockVisionRunner(4, 2, config.llm_hidden_size),
          config));

  EXPECT_FALSE(stage->NeedSchedule());
  EXPECT_THAT(stage->Schedule(), StatusIs(absl::StatusCode::kNotFound));

  // Negative width / height and mismatched zero vs positive dimensions.
  const std::vector<ImageInput> bad_inputs = {
      ImageInput{
          .image_bytes = {1, 2, 3}, .width = -1, .height = 1, .channels = 3},
      ImageInput{
          .image_bytes = {1, 2, 3}, .width = 1, .height = -1, .channels = 3},
      ImageInput{
          .image_bytes = {1, 2, 3}, .width = 2, .height = 0, .channels = 3},
      ImageInput{
          .image_bytes = {1, 2, 3}, .width = 0, .height = 2, .channels = 3},
      // Unsupported channel count (2).
      ImageInput{
          .image_bytes = {1, 2, 3, 4}, .width = 2, .height = 1, .channels = 2},
      // Overflow-inducing dimensions where width * height * channels exceeds
      // 32-bit size_t range (65536 * 65536 * 3 = 12 GiB; wraps to 0 in 32-bit).
      ImageInput{.image_bytes = {1, 2, 3},
                 .width = 65536,
                 .height = 65536,
                 .channels = 3},
      // Large INT_MAX dimensions also safely rejected without overflow.
      ImageInput{.image_bytes = {1},
                 .width = std::numeric_limits<int>::max(),
                 .height = std::numeric_limits<int>::max(),
                 .channels = 4},
  };

  for (uint64_t i = 0; i < bad_inputs.size(); ++i) {
    ASSERT_OK(image_queue.PushBack(i, bad_inputs[i]));
    EXPECT_THAT(stage->Schedule(),
                StatusIs(absl::StatusCode::kInvalidArgument));
    ASSERT_OK(image_queue.PopFront());
  }

  // Reset clears buffered outputs.
  ASSERT_OK(image_queue.PushBack(
      100, ImageInput{
               .image_bytes = std::vector<uint8_t>(4 * 4 * 3, 128),
               .width = 4,
               .height = 4,
               .channels = 3,
           }));
  ASSERT_OK(stage->Schedule());
  EXPECT_TRUE(stage->HasOutput());
  stage->Reset();
  EXPECT_FALSE(stage->HasOutput());
  EXPECT_FALSE(stage->NeedSchedule());
}

}  // namespace
}  // namespace litert::omni::any2any
