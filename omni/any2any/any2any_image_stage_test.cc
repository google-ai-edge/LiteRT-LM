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

#include "omni/any2any/any2any_image_stage.h"

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "omni/any2any/io_types.h"
#include "omni/base/fan_out_stage.h"
#include "omni/base/io_types.h"
#include "omni/base/mock_litert_runner.h"
#include "omni/base/sequenced.h"
#include "omni/omni_session.h"
#include "runtime/components/preprocessor/image_preprocessor.h"
#include "support/util/convert_tensor_buffer.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace litert::omni::any2any {
namespace {

using ::absl_testing::StatusIs;
using ::testing::_;
using ::testing::ElementsAre;
using ::testing::NiceMock;
using ::testing::Return;

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

struct ObservedImage {
  int width = 0;
  int height = 0;
  const uint8_t* pixels_ptr = nullptr;
  std::vector<uint8_t> pixels;
};

class FakeImagePreprocessor : public lm::ImagePreprocessor {
 public:
  explicit FakeImagePreprocessor(lm::DecodedImage result)
      : result_(std::move(result)) {}

  absl::StatusOr<lm::DecodedImage> Decode(
      absl::string_view /*image_bytes*/) const override {
    return result_;
  }

 private:
  lm::DecodedImage result_;
};

Any2AnyImageStage::EncodeImageFn MakeRecordingEncoder(
    int num_output_tokens, std::vector<ObservedImage>* observed,
    absl::Status* encode_error = nullptr) {
  return [num_output_tokens, observed, encode_error](
             const Any2AnyImageStage::RgbImageView& image,
             absl::Span<TensorBuffer> input_buffers,
             Any2AnyImageStage::RunEncoderFn run_encoder)
             -> absl::StatusOr<Any2AnyImageEmbeddings> {
    if (encode_error != nullptr && !encode_error->ok()) {
      return *encode_error;
    }
    if (observed != nullptr) {
      observed->push_back(ObservedImage{
          .width = image.width,
          .height = image.height,
          .pixels_ptr = image.pixels.data(),
          .pixels =
              std::vector<uint8_t>(image.pixels.begin(), image.pixels.end()),
      });
    }
    const std::vector<float> dummy_input(4, 1.0f);
    LITERT_RETURN_IF_ERROR(
        input_buffers[0].Write<float>(absl::MakeConstSpan(dummy_input)));
    ABSL_ASSIGN_OR_RETURN(std::vector<float> embeddings, run_encoder());
    return Any2AnyImageEmbeddings{
        .embeddings = std::move(embeddings),
        .num_tokens = num_output_tokens,
        .tokens_per_slice = num_output_tokens,
    };
  };
}

std::unique_ptr<NiceMock<MockLiteRtRunner>> MakeMockVisionRunner(
    int num_tokens, int hidden_dim) {
  auto runner = std::make_unique<NiceMock<MockLiteRtRunner>>();
  ON_CALL(*runner, CreateInputBuffers(absl::string_view("")))
      .WillByDefault(
          [](absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
            LITERT_ASSIGN_OR_RETURN(auto buf,
                                    support::CreateTensorBuffer<float>({1, 4}));
            std::vector<TensorBuffer> bufs;
            bufs.push_back(std::move(buf));
            return bufs;
          });
  ON_CALL(*runner, CreateOutputBuffers(absl::string_view("")))
      .WillByDefault([num_tokens, hidden_dim](absl::string_view)
                         -> absl::StatusOr<std::vector<TensorBuffer>> {
        LITERT_ASSIGN_OR_RETURN(auto buf, support::CreateTensorBuffer<float>(
                                              {1, num_tokens, hidden_dim}));
        std::vector<TensorBuffer> bufs;
        bufs.push_back(std::move(buf));
        return bufs;
      });
  ON_CALL(*runner, Run(absl::string_view(""), _, _))
      .WillByDefault([num_tokens, hidden_dim](
                         absl::string_view, absl::Span<const TensorBuffer>,
                         absl::Span<const TensorBuffer> outputs)
                         -> absl::Status {
        std::vector<float> out(static_cast<size_t>(num_tokens) * hidden_dim);
        for (size_t i = 0; i < out.size(); ++i) {
          out[i] = static_cast<float>(i + 1);
        }
        auto& out_buf = const_cast<TensorBuffer&>(outputs[0]);
        LITERT_RETURN_IF_ERROR(out_buf.Write<float>(absl::MakeConstSpan(out)));
        return absl::OkStatus();
      });
  return runner;
}

TEST(Any2AnyImageStageTest, CreateValidatesOptionsBuffersAndEncoderCallback) {
  SequencedVariantQueue<OmniSession::Input, ImageInput> image_queue;
  auto default_init = [](absl::Span<const TensorBuffer>, int,
                         int num_output_tokens)
      -> absl::StatusOr<Any2AnyImageStage::EncodeImageFn> {
    return MakeRecordingEncoder(num_output_tokens, /*observed=*/nullptr);
  };

  EXPECT_THAT(
      Any2AnyImageStage::Create(
          &image_queue, MakeMockVisionRunner(2, 4),
          Any2AnyImageStage::Options{.llm_hidden_size = 0}, default_init),
      StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(
      Any2AnyImageStage::Create(
          &image_queue, MakeMockVisionRunner(2, 4),
          Any2AnyImageStage::Options{.llm_hidden_size = -1}, default_init),
      StatusIs(absl::StatusCode::kInvalidArgument));

  auto empty_runner = std::make_unique<NiceMock<MockLiteRtRunner>>();
  ON_CALL(*empty_runner, CreateInputBuffers(absl::string_view("")))
      .WillByDefault(
          [](absl::string_view) { return std::vector<TensorBuffer>{}; });
  ON_CALL(*empty_runner, CreateOutputBuffers(absl::string_view("")))
      .WillByDefault(
          [](absl::string_view) { return std::vector<TensorBuffer>{}; });
  EXPECT_THAT(
      Any2AnyImageStage::Create(
          &image_queue, std::move(empty_runner),
          Any2AnyImageStage::Options{.llm_hidden_size = 4}, default_init),
      StatusIs(absl::StatusCode::kInvalidArgument));

  // Output buffer byte size not divisible by llm_hidden_size * sizeof(float).
  EXPECT_THAT(
      Any2AnyImageStage::Create(
          &image_queue,
          MakeMockVisionRunner(/*num_tokens=*/1, /*hidden_dim=*/3),
          Any2AnyImageStage::Options{.llm_hidden_size = 4}, default_init),
      StatusIs(absl::StatusCode::kInvalidArgument));

  // Failing init_encoder propagates error.
  EXPECT_THAT(Any2AnyImageStage::Create(
                  &image_queue, MakeMockVisionRunner(2, 4),
                  Any2AnyImageStage::Options{.llm_hidden_size = 4},
                  [](absl::Span<const TensorBuffer>, int,
                     int) -> absl::StatusOr<Any2AnyImageStage::EncodeImageFn> {
                    return absl::FailedPreconditionError("bad input tensor");
                  }),
              StatusIs(absl::StatusCode::kFailedPrecondition));

  // Null EncodeImageFn returned by init_encoder is rejected.
  EXPECT_THAT(Any2AnyImageStage::Create(
                  &image_queue, MakeMockVisionRunner(2, 4),
                  Any2AnyImageStage::Options{.llm_hidden_size = 4},
                  [](absl::Span<const TensorBuffer>, int,
                     int) -> absl::StatusOr<Any2AnyImageStage::EncodeImageFn> {
                    return Any2AnyImageStage::EncodeImageFn{};
                  }),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(Any2AnyImageStageTest,
     DecodesRawRgbZeroCopyRgbaGrayscaleAndEmptyInputsInOrder) {
  SequencedVariantQueue<OmniSession::Input, ImageInput> image_queue;
  std::vector<ObservedImage> observed;
  ASSERT_OK_AND_ASSIGN(
      auto stage, Any2AnyImageStage::Create(
                      &image_queue, MakeMockVisionRunner(2, 4),
                      Any2AnyImageStage::Options{.llm_hidden_size = 4},
                      [&observed](absl::Span<const TensorBuffer> input_buffers,
                                  int, int num_output_tokens)
                          -> absl::StatusOr<Any2AnyImageStage::EncodeImageFn> {
                        EXPECT_EQ(input_buffers.size(), 1);
                        return MakeRecordingEncoder(num_output_tokens,
                                                    &observed);
                      }));

  EXPECT_EQ(stage->HiddenDim(), 4);
  EXPECT_EQ(stage->NumOutputTokens(), 2);
  EXPECT_FALSE(stage->NeedSchedule());
  EXPECT_THAT(stage->Schedule(), StatusIs(absl::StatusCode::kNotFound));

  // Direct DecodeImageToRgb returns a zero-copy view over raw RGB888 bytes.
  const ImageInput raw_rgb{
      .image_bytes = {10, 20, 30, 40, 50, 60},
      .width = 2,
      .height = 1,
      .channels = 3,
  };
  std::vector<uint8_t> storage;
  ASSERT_OK_AND_ASSIGN(const Any2AnyImageStage::RgbImageView direct_view,
                       stage->DecodeImageToRgb(raw_rgb, storage));
  EXPECT_EQ(direct_view.pixels.data(), raw_rgb.image_bytes.data());
  EXPECT_TRUE(storage.empty());

  // seqnum 0: empty image -> empty embeddings.
  ASSERT_OK(image_queue.PushBack(0, ImageInput{}));
  ASSERT_OK(stage->Schedule());
  ASSERT_OK_AND_ASSIGN(auto seg0, stage->GetOutput());
  EXPECT_EQ(seg0.seqnum, 0);
  EXPECT_EQ(seg0.item.num_tokens, 0);
  EXPECT_TRUE(seg0.item.embeddings.empty());

  // seqnum 1: raw RGB888 (2x1) -> zero-copy view into queued image_bytes.
  ASSERT_OK(image_queue.PushBack(1, raw_rgb));
  ASSERT_OK_AND_ASSIGN(const Sequenced<ImageInput>* peeked_rgb,
                       image_queue.PeekFront());
  const uint8_t* queued_rgb_ptr = peeked_rgb->item.image_bytes.data();

  // seqnum 2: raw RGBA8888 (2x1) -> alpha stripped.
  ASSERT_OK(
      image_queue.PushBack(2, ImageInput{
                                  .image_bytes = {1, 2, 3, 255, 4, 5, 6, 128},
                                  .width = 2,
                                  .height = 1,
                                  .channels = 4,
                              }));
  // seqnum 3: raw grayscale (2x1) -> replicated across RGB.
  ASSERT_OK(image_queue.PushBack(3, ImageInput{
                                        .image_bytes = {7, 9},
                                        .width = 2,
                                        .height = 1,
                                        .channels = 1,
                                    }));

  EXPECT_TRUE(stage->NeedSchedule());
  ASSERT_OK(stage->Schedule());
  EXPECT_FALSE(stage->NeedSchedule());

  ASSERT_EQ(observed.size(), 3);
  EXPECT_EQ(observed[0].width, 2);
  EXPECT_EQ(observed[0].height, 1);
  EXPECT_EQ(observed[0].pixels_ptr, queued_rgb_ptr);
  EXPECT_THAT(observed[0].pixels, ElementsAre(10, 20, 30, 40, 50, 60));
  EXPECT_THAT(observed[1].pixels, ElementsAre(1, 2, 3, 4, 5, 6));
  EXPECT_THAT(observed[2].pixels, ElementsAre(7, 7, 7, 9, 9, 9));

  ASSERT_OK_AND_ASSIGN(auto seg1, stage->GetOutput());
  EXPECT_EQ(seg1.seqnum, 1);
  EXPECT_EQ(seg1.item.num_tokens, 2);
  EXPECT_THAT(seg1.item.embeddings, ElementsAre(1, 2, 3, 4, 5, 6, 7, 8));

  ASSERT_OK_AND_ASSIGN(auto seg2, stage->GetOutput());
  EXPECT_EQ(seg2.seqnum, 2);
  EXPECT_EQ(seg2.item.num_tokens, 2);

  ASSERT_OK_AND_ASSIGN(auto seg3, stage->GetOutput());
  EXPECT_EQ(seg3.seqnum, 3);
  EXPECT_EQ(seg3.item.num_tokens, 2);
}

TEST(Any2AnyImageStageTest,
     DecodesCompressedPngAndRejectsInvalidOrMissingDecoder) {
  SequencedVariantQueue<OmniSession::Input, ImageInput> image_queue;
  std::vector<ObservedImage> observed;
  auto init = [&observed](absl::Span<const TensorBuffer>, int,
                          int num_output_tokens)
      -> absl::StatusOr<Any2AnyImageStage::EncodeImageFn> {
    return MakeRecordingEncoder(num_output_tokens, &observed);
  };

  ASSERT_OK_AND_ASSIGN(
      auto stage, Any2AnyImageStage::Create(
                      &image_queue, MakeMockVisionRunner(2, 4),
                      Any2AnyImageStage::Options{.llm_hidden_size = 4}, init));

  ASSERT_OK(image_queue.PushBack(
      0, ImageInput{
             .image_bytes = std::vector<uint8_t>(std::begin(kValid2x2RgbPng),
                                                 std::end(kValid2x2RgbPng)),
             .width = 0,
             .height = 0,
             .channels = 0,
         }));
  ASSERT_OK(stage->Schedule());

  ASSERT_OK_AND_ASSIGN(auto seg, stage->GetOutput());
  EXPECT_EQ(seg.seqnum, 0);
  EXPECT_EQ(seg.item.num_tokens, 2);
  ASSERT_EQ(observed.size(), 1);
  EXPECT_EQ(observed[0].width, 2);
  EXPECT_EQ(observed[0].height, 2);
  EXPECT_THAT(observed[0].pixels,
              ElementsAre(255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255));

  // Corrupt encoded bytes fail and leave the item queued.
  ASSERT_OK(image_queue.PushBack(1, ImageInput{
                                        .image_bytes = {0x00, 0x01, 0x02, 0x03},
                                        .width = 0,
                                        .height = 0,
                                    }));
  EXPECT_FALSE(stage->Schedule().ok());
  EXPECT_TRUE(stage->NeedSchedule());
  ASSERT_OK(image_queue.PopFront());

  // Null ImagePreprocessor fails with FailedPreconditionError on encoded input.
  ASSERT_OK_AND_ASSIGN(
      auto null_preprocessor_stage,
      Any2AnyImageStage::Create(
          &image_queue, MakeMockVisionRunner(2, 4),
          Any2AnyImageStage::Options{.llm_hidden_size = 4}, init, nullptr));
  ASSERT_OK(image_queue.PushBack(
      2, ImageInput{
             .image_bytes = std::vector<uint8_t>(std::begin(kValid2x2RgbPng),
                                                 std::end(kValid2x2RgbPng)),
         }));
  EXPECT_THAT(null_preprocessor_stage->Schedule(),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  ASSERT_OK(image_queue.PopFront());

  // Decoder returning inconsistent dimensions/pixel count fails with Internal.
  ASSERT_OK_AND_ASSIGN(
      auto bad_decoder_stage,
      Any2AnyImageStage::Create(
          &image_queue, MakeMockVisionRunner(2, 4),
          Any2AnyImageStage::Options{.llm_hidden_size = 4}, init,
          std::make_unique<FakeImagePreprocessor>(lm::DecodedImage{
              .pixels = {1, 2, 3},
              .width = 2,
              .height = 2,
          })));
  ASSERT_OK(image_queue.PushBack(3, ImageInput{.image_bytes = {0x01, 0x02}}));
  EXPECT_THAT(bad_decoder_stage->Schedule(),
              StatusIs(absl::StatusCode::kInternal));
  ASSERT_OK(image_queue.PopFront());
}

TEST(Any2AnyImageStageTest, RejectsInvalidRawDimensionsChannelsAndOverflow) {
  SequencedVariantQueue<OmniSession::Input, ImageInput> image_queue;
  ASSERT_OK_AND_ASSIGN(
      auto stage,
      Any2AnyImageStage::Create(
          &image_queue, MakeMockVisionRunner(2, 4),
          Any2AnyImageStage::Options{.llm_hidden_size = 4},
          [](absl::Span<const TensorBuffer>, int, int num_output_tokens)
              -> absl::StatusOr<Any2AnyImageStage::EncodeImageFn> {
            return MakeRecordingEncoder(num_output_tokens,
                                        /*observed=*/nullptr);
          }));

  // Direct DecodeImageToRgb rejects empty bytes.
  std::vector<uint8_t> storage;
  EXPECT_THAT(stage->DecodeImageToRgb(ImageInput{}, storage),
              StatusIs(absl::StatusCode::kInvalidArgument));

  const std::vector<ImageInput> bad_inputs = {
      ImageInput{
          .image_bytes = {1, 2, 3}, .width = -1, .height = 1, .channels = 3},
      ImageInput{
          .image_bytes = {1, 2, 3}, .width = 1, .height = -1, .channels = 3},
      ImageInput{
          .image_bytes = {1, 2, 3}, .width = 2, .height = 0, .channels = 3},
      ImageInput{
          .image_bytes = {1, 2, 3}, .width = 0, .height = 2, .channels = 3},
      ImageInput{
          .image_bytes = {1, 2, 3, 4}, .width = 2, .height = 1, .channels = 2},
      ImageInput{
          .image_bytes = {1, 2, 3}, .width = 2, .height = 2, .channels = 3},
      // 32-bit overflow dimensions (65536 * 65536 * 3 wraps to 0 on 32-bit).
      ImageInput{.image_bytes = {1, 2, 3},
                 .width = 65536,
                 .height = 65536,
                 .channels = 3},
      ImageInput{.image_bytes = {1},
                 .width = std::numeric_limits<int>::max(),
                 .height = std::numeric_limits<int>::max(),
                 .channels = 4},
  };

  for (uint64_t i = 0; i < bad_inputs.size(); ++i) {
    ASSERT_OK(image_queue.PushBack(i, bad_inputs[i]));
    EXPECT_THAT(stage->Schedule(),
                StatusIs(absl::StatusCode::kInvalidArgument));
    EXPECT_TRUE(stage->NeedSchedule());
    ASSERT_OK(image_queue.PopFront());
  }
}

TEST(Any2AnyImageStageTest,
     PropagatesEncodeAndRunnerErrorsWithoutPoppingQueuedInput) {
  SequencedVariantQueue<OmniSession::Input, ImageInput> image_queue;
  auto runner = MakeMockVisionRunner(2, 4);
  MockLiteRtRunner* runner_ptr = runner.get();
  absl::Status encode_error = absl::OkStatus();
  ASSERT_OK_AND_ASSIGN(
      auto stage, Any2AnyImageStage::Create(
                      &image_queue, std::move(runner),
                      Any2AnyImageStage::Options{.llm_hidden_size = 4},
                      [&encode_error](absl::Span<const TensorBuffer>, int,
                                      int num_output_tokens)
                          -> absl::StatusOr<Any2AnyImageStage::EncodeImageFn> {
                        return MakeRecordingEncoder(num_output_tokens,
                                                    /*observed=*/nullptr,
                                                    &encode_error);
                      }));

  ASSERT_OK(image_queue.PushBack(0, ImageInput{
                                        .image_bytes = {1, 2, 3},
                                        .width = 1,
                                        .height = 1,
                                        .channels = 3,
                                    }));

  // Encoder callback error leaves input at front of queue.
  encode_error = absl::ResourceExhaustedError("transient failure");
  EXPECT_THAT(stage->Schedule(),
              StatusIs(absl::StatusCode::kResourceExhausted));
  EXPECT_FALSE(stage->HasOutput());
  EXPECT_TRUE(stage->NeedSchedule());
  encode_error = absl::OkStatus();

  // Vision encoder runner failure also leaves input queued.
  EXPECT_CALL(*runner_ptr, Run(absl::string_view(""), _, _))
      .WillOnce(Return(absl::InternalError("runner failed")));
  EXPECT_THAT(stage->Schedule(), StatusIs(absl::StatusCode::kInternal));
  EXPECT_FALSE(stage->HasOutput());
  EXPECT_TRUE(stage->NeedSchedule());

  // Retry succeeds once the runner recovers.
  EXPECT_CALL(*runner_ptr, Run(absl::string_view(""), _, _))
      .WillOnce(Return(absl::OkStatus()));
  ASSERT_OK(stage->Schedule());
  ASSERT_OK_AND_ASSIGN(auto seg, stage->GetOutput());
  EXPECT_EQ(seg.seqnum, 0);
  EXPECT_EQ(seg.item.num_tokens, 2);
}

}  // namespace
}  // namespace litert::omni::any2any
