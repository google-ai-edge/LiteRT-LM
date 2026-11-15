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

#include "omni/utils/audio_resampler.h"

#include <cmath>
#include <cstddef>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "support/util/test_utils.h"  // IWYU pragma: keep for ASSERT_OK

namespace litert::omni {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::AllOf;
using ::testing::FloatEq;
using ::testing::Ge;
using ::testing::IsEmpty;
using ::testing::Le;
using ::testing::Pointwise;

TEST(AudioResamplerTest, RejectsInvalidSampleRates) {
  AudioResampler resampler;
  std::vector<float> pcm(100, 0.1f);
  EXPECT_THAT(resampler.Resample(pcm, 0, 16000),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(resampler.Resample(pcm, 16000, 0),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(resampler.Resample(pcm, -16000, 16000),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(AudioResamplerTest, EmptyInputReturnsEmptyOutput) {
  AudioResampler resampler;
  EXPECT_THAT(resampler.Resample({}, 24000, 16000), IsOkAndHolds(IsEmpty()));
}

TEST(AudioResamplerTest, SameSampleRateReturnsIdenticalCopy) {
  AudioResampler resampler;
  std::vector<float> input = {0.1f, -0.2f, 0.5f, 0.0f, -0.8f};
  EXPECT_THAT(resampler.Resample(input, 16000, 16000), IsOkAndHolds(input));
}

TEST(AudioResamplerTest, DownsamplesAudioCorrectly) {
  AudioResampler resampler;
  // 1 second of 440 Hz sine wave at 24 kHz.
  std::vector<float> input_24k(24000);
  for (int i = 0; i < input_24k.size(); ++i) {
    input_24k[i] = std::sin(2.0 * M_PI * 440.0 * i / 24000.0);
  }

  ASSERT_OK_AND_ASSIGN(std::vector<float> output_16k,
                       resampler.Resample(input_24k, 24000, 16000));
  // 1 second at 16 kHz should be ~16000 samples.
  EXPECT_THAT(output_16k.size(), AllOf(Ge(15990), Le(16010)));
}

TEST(AudioResamplerTest, UpsamplesAudioCorrectly) {
  AudioResampler resampler;
  // 1 second of 440 Hz sine wave at 8 kHz.
  std::vector<float> input_8k(8000);
  for (int i = 0; i < input_8k.size(); ++i) {
    input_8k[i] = std::sin(2.0 * M_PI * 440.0 * i / 8000.0);
  }

  ASSERT_OK_AND_ASSIGN(std::vector<float> output_16k,
                       resampler.Resample(input_8k, 8000, 16000));
  // 1 second at 16 kHz should be ~16000 samples.
  EXPECT_THAT(output_16k.size(), AllOf(Ge(15990), Le(16010)));
}

TEST(AudioResamplerTest, ResamplingMultipleTimesProducesConsistentOutput) {
  AudioResampler resampler;
  std::vector<float> input_24k(24000, 0.25f);

  ASSERT_OK_AND_ASSIGN(std::vector<float> out1,
                       resampler.Resample(input_24k, 24000, 16000));
  resampler.Reset();
  ASSERT_OK_AND_ASSIGN(std::vector<float> out2,
                       resampler.Resample(input_24k, 24000, 16000));

  EXPECT_THAT(out1, Pointwise(FloatEq(), out2));
}

}  // namespace
}  // namespace litert::omni
