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

}  // namespace
}  // namespace litert::support
