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

#include "omni/any2any/any2any_input_fan_out_stage.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "omni/base/io_types.h"
#include "omni/base/sequenced.h"
#include "omni/omni_session.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace litert::omni::any2any {
namespace {

using ::absl_testing::StatusIs;
using ::testing::ElementsAre;

TEST(Any2AnyInputFanOutStageTest, RoutesAllSupportedInputTypesInSequenceOrder) {
  PushInputSource input_source;
  Any2AnyInputFanOutStage fan_out(&input_source);

  ASSERT_OK(input_source.PushInput(TextInput{.text = "Hello"}));
  ASSERT_OK(input_source.PushInput(AudioInput{.pcm_samples = {0.25f}}));
  ASSERT_OK(input_source.PushInput(
      AudioInputMetadata{.sample_rate_hz = 16000, .num_channels = 1}));
  ASSERT_OK(input_source.PushInput(ImageInput{
      .image_bytes = {1, 2, 3}, .width = 1, .height = 1, .channels = 3}));
  ASSERT_OK(input_source.PushInput(EndOfInputTurn{}));
  ASSERT_OK(input_source.PushInput(EndOfInput{}));

  while (fan_out.NeedSchedule()) {
    ASSERT_OK(fan_out.Schedule());
  }

  ASSERT_OK_AND_ASSIGN(Sequenced<TextInput> text_item,
                       fan_out.TextQueue().PopFront());
  EXPECT_EQ(text_item.seqnum, 0);
  EXPECT_EQ(text_item.item.text, "Hello");

  ASSERT_OK_AND_ASSIGN(Sequenced<AudioInput> audio_item,
                       fan_out.AudioQueue().PopFront());
  EXPECT_EQ(audio_item.seqnum, 1);
  EXPECT_THAT(audio_item.item.pcm_samples, ElementsAre(0.25f));

  ASSERT_OK_AND_ASSIGN(Sequenced<AudioInputMetadata> meta_item,
                       fan_out.AudioInputMetadataQueue().PopFront());
  EXPECT_EQ(meta_item.seqnum, 2);
  EXPECT_EQ(meta_item.item.sample_rate_hz, 16000);
  EXPECT_EQ(meta_item.item.num_channels, 1);

  ASSERT_OK_AND_ASSIGN(Sequenced<ImageInput> image_item,
                       fan_out.ImageQueue().PopFront());
  EXPECT_EQ(image_item.seqnum, 3);
  EXPECT_EQ(image_item.item.width, 1);
  EXPECT_THAT(image_item.item.image_bytes, ElementsAre(1, 2, 3));

  ASSERT_OK_AND_ASSIGN(Sequenced<EndOfInputTurn> eot_item,
                       fan_out.EndOfTurnQueue().PopFront());
  EXPECT_EQ(eot_item.seqnum, 4);

  ASSERT_OK_AND_ASSIGN(Sequenced<EndOfInput> eoi_item,
                       fan_out.EndOfInputQueue().PopFront());
  EXPECT_EQ(eoi_item.seqnum, 5);
}

TEST(Any2AnyInputFanOutStageTest,
     RejectsDisabledModalitiesAndImageGenMetadataWithoutAdvancingSeqnum) {
  PushInputSource input_source;
  Any2AnyInputFanOutStage fan_out(&input_source, /*has_audio_encoder=*/false,
                                  /*has_vision_encoder=*/false);

  ASSERT_OK(input_source.PushInput(AudioInput{.pcm_samples = {0.1f}}));
  EXPECT_THAT(fan_out.Schedule(),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_TRUE(fan_out.AudioQueue().Empty());

  ASSERT_OK(input_source.PushInput(
      AudioInputMetadata{.sample_rate_hz = 16000, .num_channels = 1}));
  EXPECT_THAT(fan_out.Schedule(),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_TRUE(fan_out.AudioInputMetadataQueue().Empty());

  ASSERT_OK(input_source.PushInput(ImageInput{.image_bytes = {10, 20, 30, 40},
                                              .width = 2,
                                              .height = 2,
                                              .channels = 1}));
  EXPECT_THAT(fan_out.Schedule(),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_TRUE(fan_out.ImageQueue().Empty());

  ASSERT_OK(input_source.PushInput(ImageGenInputMetadata{.width = 256}));
  EXPECT_THAT(fan_out.Schedule(), StatusIs(absl::StatusCode::kUnimplemented));

  // Rejected inputs must not advance the sequence number.
  ASSERT_OK(input_source.PushInput(TextInput{.text = "Still seqnum 0"}));
  ASSERT_OK(fan_out.Schedule());
  ASSERT_OK_AND_ASSIGN(Sequenced<TextInput> text_item,
                       fan_out.TextQueue().PopFront());
  EXPECT_EQ(text_item.seqnum, 0);
  EXPECT_EQ(text_item.item.text, "Still seqnum 0");
}

TEST(Any2AnyInputFanOutStageTest, RejectsAudioOnlyWhenAudioEncoderDisabled) {
  PushInputSource input_source;
  Any2AnyInputFanOutStage fan_out(&input_source, /*has_audio_encoder=*/false,
                                  /*has_vision_encoder=*/true);

  ASSERT_OK(input_source.PushInput(AudioInput{.pcm_samples = {0.1f}}));
  EXPECT_THAT(fan_out.Schedule(),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_TRUE(fan_out.AudioQueue().Empty());

  ASSERT_OK(input_source.PushInput(
      AudioInputMetadata{.sample_rate_hz = 16000, .num_channels = 1}));
  EXPECT_THAT(fan_out.Schedule(),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_TRUE(fan_out.AudioInputMetadataQueue().Empty());

  ASSERT_OK(input_source.PushInput(ImageInput{.image_bytes = {10, 20, 30, 40},
                                              .width = 2,
                                              .height = 2,
                                              .channels = 1}));
  ASSERT_OK(fan_out.Schedule());
  ASSERT_OK_AND_ASSIGN(Sequenced<ImageInput> image_item,
                       fan_out.ImageQueue().PopFront());
  EXPECT_EQ(image_item.seqnum, 0);
  EXPECT_THAT(image_item.item.image_bytes, ElementsAre(10, 20, 30, 40));
}

TEST(Any2AnyInputFanOutStageTest, RejectsVisionOnlyWhenVisionEncoderDisabled) {
  PushInputSource input_source;
  Any2AnyInputFanOutStage fan_out(&input_source, /*has_audio_encoder=*/true,
                                  /*has_vision_encoder=*/false);

  ASSERT_OK(input_source.PushInput(ImageInput{.image_bytes = {10, 20, 30, 40},
                                              .width = 2,
                                              .height = 2,
                                              .channels = 1}));
  EXPECT_THAT(fan_out.Schedule(),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_TRUE(fan_out.ImageQueue().Empty());

  ASSERT_OK(input_source.PushInput(AudioInput{.pcm_samples = {0.1f}}));
  ASSERT_OK(input_source.PushInput(
      AudioInputMetadata{.sample_rate_hz = 16000, .num_channels = 1}));
  ASSERT_OK(fan_out.Schedule());

  ASSERT_OK_AND_ASSIGN(Sequenced<AudioInput> audio_item,
                       fan_out.AudioQueue().PopFront());
  EXPECT_EQ(audio_item.seqnum, 0);
  EXPECT_THAT(audio_item.item.pcm_samples, ElementsAre(0.1f));

  ASSERT_OK_AND_ASSIGN(Sequenced<AudioInputMetadata> meta_item,
                       fan_out.AudioInputMetadataQueue().PopFront());
  EXPECT_EQ(meta_item.seqnum, 1);
  EXPECT_EQ(meta_item.item.sample_rate_hz, 16000);
}

}  // namespace
}  // namespace litert::omni::any2any
