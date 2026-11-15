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

#include "support/preprocessor/audio_preprocessor_utils.h"

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "support/preprocessor/audio_preprocessor.h"
#include "support/util/test_utils.h"  // NOLINT

namespace litert::support {
namespace {

TEST(AudioPreprocessorUtilsTest, HanningWindowLengthZeroOrNegative) {
  EXPECT_TRUE(GetHanningWindow(0).empty());
  EXPECT_TRUE(GetHanningWindow(-5).empty());
}

TEST(AudioPreprocessorUtilsTest, HanningWindowValues) {
  auto window = GetHanningWindow(4, /*use_periodic_hanning=*/true,
                                 /*non_zero_hanning=*/false);
  EXPECT_EQ(window.size(), 4);
  // With periodic hanning (n=4) and non_zero_hanning=false (shift=0):
  // arg = 2*pi/4 = pi/2
  // i=0: 0.5 - 0.5*cos(0) = 0
  // i=1: 0.5 - 0.5*cos(pi/2) = 0.5
  // i=2: 0.5 - 0.5*cos(pi) = 1.0
  // i=3: 0.5 - 0.5*cos(3pi/2) = 0.5
  EXPECT_NEAR(window[0], 0.0f, 1e-5);
  EXPECT_NEAR(window[1], 0.5f, 1e-5);
  EXPECT_NEAR(window[2], 1.0f, 1e-5);
  EXPECT_NEAR(window[3], 0.5f, 1e-5);
}

TEST(AudioPreprocessorUtilsTest, PadOrTruncateRightPadding) {
  std::vector<float> frame = {1.0f, 2.0f, 3.0f};
  ASSERT_OK(PadOrTruncateForFft(
      5, AudioPreprocessorConfig::FftPaddingType::kRight, frame));
  EXPECT_THAT(frame, ::testing::ElementsAre(1.0f, 2.0f, 3.0f, 0.0f, 0.0f));
}

TEST(AudioPreprocessorUtilsTest, PadOrTruncateCenterPadding) {
  std::vector<float> frame = {1.0f, 2.0f, 3.0f};
  ASSERT_OK(PadOrTruncateForFft(
      5, AudioPreprocessorConfig::FftPaddingType::kCenter, frame));
  EXPECT_THAT(frame, ::testing::ElementsAre(0.0f, 1.0f, 2.0f, 3.0f, 0.0f));
}

TEST(AudioPreprocessorUtilsTest, PadOrTruncateCenterTruncate) {
  std::vector<float> frame = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  ASSERT_OK(PadOrTruncateForFft(
      3, AudioPreprocessorConfig::FftPaddingType::kCenter, frame));
  // Trim left = (5 - 3) / 2 = 1. Erase 1, resize to 3.
  EXPECT_THAT(frame, ::testing::ElementsAre(2.0f, 3.0f, 4.0f));
}

TEST(AudioPreprocessorUtilsTest, GetWindowedSignalsForFftSuccess) {
  auto config = AudioPreprocessorConfig::CreateDefaultUsmConfig();
  std::vector<float> pcm_data(1000, 1.0f);
  std::vector<float> input_queue;
  int samples_to_next_step = config.GetFrameLength();

  ASSERT_OK_AND_ASSIGN(auto windowed_signals,
                       GetWindowedSignalsForFft(config, pcm_data, input_queue,
                                                samples_to_next_step));
  EXPECT_FALSE(windowed_signals.empty());
  for (const auto& sig : windowed_signals) {
    EXPECT_EQ(sig.size(), config.GetFftLength());
  }
}

TEST(AudioPreprocessorUtilsTest, CountFramedSegmentsMatchesGetFramedSegments) {
  for (bool semicausal_padding : {false, true}) {
    for (bool buffer_last_frame : {false, true}) {
      AudioPreprocessorConfig config =
          AudioPreprocessorConfig::CreateDefaultUsmConfig();
      config.SetSemicausalPadding(semicausal_padding);
      config.SetBufferLastFrame(buffer_last_frame);
      for (int num_samples :
           {0, 1, 159, 160, 161, 319, 320, 321, 480, 1000, 16000, 16001}) {
        SCOPED_TRACE(absl::StrCat("semicausal_padding=", semicausal_padding,
                                  " buffer_last_frame=", buffer_last_frame,
                                  " num_samples=", num_samples));
        // Start from the initial state of AudioPreprocessorPffft.
        std::vector<float> input_queue;
        int samples_to_next_step = config.GetFrameLength();
        if (semicausal_padding) {
          samples_to_next_step =
              config.GetFrameLength() - config.GetHopLength();
          input_queue.resize(config.GetHopLength(), 0.0f);
        }
        ASSERT_OK_AND_ASSIGN(
            int count,
            CountFramedSegments(config, num_samples, input_queue.size(),
                                samples_to_next_step));
        std::vector<float> pcm_frames(num_samples, 0.5f);
        ASSERT_OK_AND_ASSIGN(std::vector<std::vector<float>> segments,
                             GetFramedSegments(config, pcm_frames, input_queue,
                                               samples_to_next_step));
        EXPECT_EQ(count, segments.size());
      }
    }
  }
}

TEST(AudioPreprocessorUtilsTest, CountFramedSegmentsInvalidHopLength) {
  AudioPreprocessorConfig config =
      AudioPreprocessorConfig::CreateDefaultUsmConfig();
  config.SetHopLength(0);
  EXPECT_THAT(CountFramedSegments(config, /*num_samples=*/1000,
                                  /*input_queue_size=*/0,
                                  /*samples_to_next_step=*/1),
              testing::status::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(AudioPreprocessorUtilsTest, GetPreprocessedAudioDimensionsForNumSamples) {
  AudioPreprocessorConfig config =
      AudioPreprocessorConfig::CreateDefaultUsmConfig();
  const int samples_to_next_step = config.GetFrameLength();
  ASSERT_OK_AND_ASSIGN(
      int num_frames,
      CountFramedSegments(config, /*num_samples=*/16000,
                          /*input_queue_size=*/0, samples_to_next_step));
  ASSERT_GT(num_frames, 0);

  ASSERT_OK_AND_ASSIGN(auto mel_dimensions,
                       GetPreprocessedAudioDimensionsForNumSamples(
                           config, /*num_samples=*/16000,
                           /*input_queue_size=*/0, samples_to_next_step));
  EXPECT_THAT(mel_dimensions,
              ::testing::ElementsAre(1, num_frames, config.GetNumMelBins()));

  config.SetSkipMelSpectrogramExtraction(true);
  ASSERT_OK_AND_ASSIGN(auto frame_dimensions,
                       GetPreprocessedAudioDimensionsForNumSamples(
                           config, /*num_samples=*/16000,
                           /*input_queue_size=*/0, samples_to_next_step));
  EXPECT_THAT(frame_dimensions,
              ::testing::ElementsAre(1, num_frames, config.GetFrameLength()));

  // Without any frame to emit, fails as Preprocess() does when mel spectrogram
  // extraction is skipped.
  config.SetBufferLastFrame(true);
  EXPECT_THAT(GetPreprocessedAudioDimensionsForNumSamples(
                  config, /*num_samples=*/10, /*input_queue_size=*/0,
                  samples_to_next_step),
              testing::status::StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(AudioPreprocessorUtilsTest, CountFramedSegmentsInvalidStateArguments) {
  AudioPreprocessorConfig config =
      AudioPreprocessorConfig::CreateDefaultUsmConfig();
  EXPECT_THAT(CountFramedSegments(config, /*num_samples=*/-1,
                                  /*input_queue_size=*/0,
                                  /*samples_to_next_step=*/1),
              testing::status::StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(CountFramedSegments(config, /*num_samples=*/1000,
                                  /*input_queue_size=*/-1,
                                  /*samples_to_next_step=*/1),
              testing::status::StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(CountFramedSegments(config, /*num_samples=*/1000,
                                  /*input_queue_size=*/0,
                                  /*samples_to_next_step=*/0),
              testing::status::StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(CountFramedSegments(config, /*num_samples=*/1000,
                                  /*input_queue_size=*/0,
                                  /*samples_to_next_step=*/-1),
              testing::status::StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace litert::support
