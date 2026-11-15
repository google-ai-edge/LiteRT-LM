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

#include "omni/utils/audio_utils.h"

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "support/preprocessor/audio_preprocessor.h"

namespace litert::omni {
namespace {

using ::testing::ElementsAre;
using ::testing::FloatNear;
using ::testing::IsEmpty;

TEST(AudioUtilsTest, CreateWhisperAudioPreprocessorConfigPopulatesFields) {
  const litert::support::AudioPreprocessorConfig config =
      CreateWhisperAudioPreprocessorConfig(/*sample_rate_hz=*/16000,
                                           /*n_fft=*/400, /*hop_length=*/160,
                                           /*n_mels=*/80, /*mel_floor=*/1e-10f);

  EXPECT_EQ(config.GetSampleRateHz(), 16000);
  EXPECT_EQ(config.GetNumChannels(), 1);
  EXPECT_EQ(config.GetFrameLength(), 400);
  EXPECT_EQ(config.GetHopLength(), 160);
  EXPECT_EQ(config.GetFftLength(), 400);
  EXPECT_EQ(config.GetNumMelBins(), 80);
  EXPECT_FLOAT_EQ(config.GetMelLowHz(), 0.0f);
  EXPECT_FLOAT_EQ(config.GetMelHighHz(), 8000.0f);
  EXPECT_FLOAT_EQ(config.GetMelFloor(), 1e-10f);
  EXPECT_FALSE(config.GetNormalizeMel());
  EXPECT_TRUE(config.GetAddFloorToMelBeforeLog());
  EXPECT_FALSE(config.GetNonZeroHanning());
  EXPECT_TRUE(config.GetPeriodicHanning());
  EXPECT_TRUE(config.GetSlaneyMel());
  EXPECT_TRUE(config.GetPowerSpectrum());
}

TEST(AudioUtilsTest, PutSamplesInCenterReflectPadsWhenLongerThanNfft) {
  std::vector<float> samples = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

  // n_fft = 4, hop_length = 1 -> left_pad = 2, right_pad = 4 - 1 - 2 = 1.
  PutSamplesInCenter(samples, /*n_fft=*/4, /*hop_length=*/1);

  EXPECT_THAT(samples,
              ElementsAre(3.0f, 2.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 4.0f));
}

TEST(AudioUtilsTest, PutSamplesInCenterIsNoOpWhenShortOrInvalidParams) {
  std::vector<float> short_samples = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> invalid_nfft_samples = {1.0f, 2.0f, 3.0f};
  std::vector<float> invalid_hop_samples = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

  PutSamplesInCenter(short_samples, /*n_fft=*/4, /*hop_length=*/2);
  PutSamplesInCenter(invalid_nfft_samples, /*n_fft=*/0, /*hop_length=*/1);
  PutSamplesInCenter(invalid_hop_samples, /*n_fft=*/4, /*hop_length=*/-10);

  EXPECT_THAT(short_samples, ElementsAre(1.0f, 2.0f, 3.0f, 4.0f));
  EXPECT_THAT(invalid_nfft_samples, ElementsAre(1.0f, 2.0f, 3.0f));
  EXPECT_THAT(invalid_hop_samples, ElementsAre(1.0f, 2.0f, 3.0f, 4.0f, 5.0f));
}

TEST(AudioUtilsTest, NormalizeWhisperMelSpectrogramClipsAndScalesByPeak) {
  constexpr float kLn10 = 2.302585092994046f;
  // In log10 domain: mel 0 has [0.0, -4.0], mel 1 has [-10.0, 2.0].
  // With stat_frames = 1, max_val across f < 1 is 0.0 (frame 1's 2.0 is
  // excluded from peak calculation), so clip_val = -8.0.
  std::vector<std::vector<float>> mel_spec = {
      {0.0f * kLn10, -4.0f * kLn10},
      {-10.0f * kLn10, 2.0f * kLn10},
  };

  NormalizeWhisperMelSpectrogram(mel_spec, /*stat_frames=*/1, /*num_frames=*/2);

  // (0.0 + 4.0) / 4.0 = 1.0, (-4.0 + 4.0) / 4.0 = 0.0
  // (max(-10.0, -8.0) + 4.0) / 4.0 = -1.0, (2.0 + 4.0) / 4.0 = 1.5
  EXPECT_THAT(mel_spec[0],
              ElementsAre(FloatNear(1.0f, 1e-5f), FloatNear(0.0f, 1e-5f)));
  EXPECT_THAT(mel_spec[1],
              ElementsAre(FloatNear(-1.0f, 1e-5f), FloatNear(1.5f, 1e-5f)));
}

TEST(AudioUtilsTest, NormalizeWhisperMelSpectrogramHandlesBoundaryParameters) {
  constexpr float kLn10 = 2.302585092994046f;
  // stat_frames <= 0 is clamped to 1, and num_frames > mel_spec[m].size()
  // is bounded to mel_spec[m].size().
  std::vector<std::vector<float>> mel_spec = {
      {0.0f * kLn10, -10.0f * kLn10},
  };

  NormalizeWhisperMelSpectrogram(mel_spec, /*stat_frames=*/0,
                                 /*num_frames=*/10);

  EXPECT_THAT(mel_spec[0],
              ElementsAre(FloatNear(1.0f, 1e-5f), FloatNear(-1.0f, 1e-5f)));
}

TEST(AudioUtilsTest, FlattenMelSpectrogramHandlesMelMajorAndFrameMajor) {
  const std::vector<std::vector<float>> mel_spec = {
      {1.0f, 2.0f, 3.0f},
      {4.0f, 5.0f, 6.0f},
  };

  const std::vector<float> mel_major = FlattenMelSpectrogram(
      mel_spec, /*n_frames=*/2, /*n_mels=*/2, /*transpose=*/false);
  const std::vector<float> frame_major = FlattenMelSpectrogram(
      mel_spec, /*n_frames=*/2, /*n_mels=*/2, /*transpose=*/true);
  const std::vector<float> mel_major_padded = FlattenMelSpectrogram(
      mel_spec, /*n_frames=*/4, /*n_mels=*/2, /*transpose=*/false);
  const std::vector<float> frame_major_padded = FlattenMelSpectrogram(
      mel_spec, /*n_frames=*/4, /*n_mels=*/2, /*transpose=*/true);
  const std::vector<float> empty_result = FlattenMelSpectrogram(
      mel_spec, /*n_frames=*/0, /*n_mels=*/2, /*transpose=*/false);

  EXPECT_THAT(mel_major, ElementsAre(1.0f, 2.0f, 4.0f, 5.0f));
  EXPECT_THAT(frame_major, ElementsAre(1.0f, 4.0f, 2.0f, 5.0f));
  EXPECT_THAT(mel_major_padded,
              ElementsAre(1.0f, 2.0f, 3.0f, 0.0f, 4.0f, 5.0f, 6.0f, 0.0f));
  EXPECT_THAT(frame_major_padded,
              ElementsAre(1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f, 0.0f, 0.0f));
  EXPECT_THAT(empty_result, IsEmpty());
}

}  // namespace
}  // namespace litert::omni
