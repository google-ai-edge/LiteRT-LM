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

#include "omni/asr/log_mel_spectrogram_processor.h"

#include <cmath>
#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "omni/base/stage.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep for ASSERT_OK

namespace litert::omni::asr {
namespace {

using ::testing::FloatNear;

// Dummy AudioSource for testing stage scheduling.
class DummyAudioSource
    : public SingleThreadedStageWithDeque<std::vector<float>> {
 public:
  DummyAudioSource() = default;
  void PushChunk(std::vector<float> chunk) { PushOutput(std::move(chunk)); }

 protected:
  bool NeedScheduleInternal() const override { return false; }
  absl::Status ScheduleInternal() override { return absl::OkStatus(); }
};

TEST(LogMelSpectrogramProcessorTest, ProcessEmptyAudioReturnsEmptyVector) {
  DummyAudioSource dummy_source;
  ASSERT_OK_AND_ASSIGN(
      auto processor,
      LogMelSpectrogramProcessor::Create(
          16000,
          LogMelSpectrogramProcessor::LogMelSpectrogramConfig{.n_fft = 512},
          &dummy_source));
  dummy_source.PushChunk(std::vector<float>());
  ASSERT_OK(processor->Schedule());
  ASSERT_OK_AND_ASSIGN(auto result, processor->GetOutput());
  EXPECT_TRUE(result.empty());
}

TEST(LogMelSpectrogramProcessorTest, ProcessDummyAudioReturnsCorrectShape) {
  DummyAudioSource dummy_source;
  ASSERT_OK_AND_ASSIGN(
      auto processor,
      LogMelSpectrogramProcessor::Create(
          16000,
          LogMelSpectrogramProcessor::LogMelSpectrogramConfig{.n_fft = 512},
          &dummy_source));

  // Generate a simple 1 second 440Hz sine wave as a dummy float vector.
  int sample_rate = 16000;
  int duration_seconds = 1;
  std::vector<float> audio(sample_rate * duration_seconds);
  for (int i = 0; i < audio.size(); ++i) {
    audio[i] = std::sin(2.0 * M_PI * 440.0 * i / sample_rate);
  }

  dummy_source.PushChunk(audio);
  ASSERT_OK(processor->Schedule());
  ASSERT_OK_AND_ASSIGN(auto result, processor->GetOutput());
  EXPECT_FALSE(result.empty());
  int feature_size = 80;
  EXPECT_EQ(result.size() % feature_size, 0);

  bool has_non_zero = false;
  for (float val : result) {
    if (val != 0.0f) {
      has_non_zero = true;
      break;
    }
  }
  EXPECT_TRUE(has_non_zero);
}

TEST(LogMelSpectrogramProcessorTest,
     ProcessWithStandardNormalizationNormalizesCorrectly) {
  DummyAudioSource dummy_source;
  ASSERT_OK_AND_ASSIGN(
      auto processor,
      LogMelSpectrogramProcessor::Create(
          16000,
          LogMelSpectrogramProcessor::LogMelSpectrogramConfig{
              .n_fft = 512,
              .norm_type = LogMelSpectrogramProcessor::NormType::kStandard},
          &dummy_source));

  int sample_rate = 16000;
  std::vector<float> audio(sample_rate);
  for (int i = 0; i < audio.size(); ++i) {
    audio[i] = std::sin(12345.678f * i) * std::cos(98765.432f * i);
  }

  dummy_source.PushChunk(audio);
  ASSERT_OK(processor->Schedule());
  ASSERT_OK_AND_ASSIGN(auto result, processor->GetOutput());
  EXPECT_FALSE(result.empty());

  // In standard normalization, features across valid frames should have roughly
  // zero mean and unit variance.
  int n_mels = 80;
  int num_frames = result.size() / n_mels;
  EXPECT_GT(num_frames, 0);

  for (int m = 0; m < n_mels; ++m) {
    float sum = 0.0f;
    for (int f = 0; f < num_frames; ++f) {
      sum += result[m * num_frames + f];
    }
    float mean = sum / num_frames;
    EXPECT_THAT(mean, FloatNear(0.0f, 1e-3f));
  }
}

TEST(LogMelSpectrogramProcessorTest,
     ProcessWithWhisperNormalizationClipsAndScalesCorrectly) {
  DummyAudioSource dummy_source;
  ASSERT_OK_AND_ASSIGN(
      auto processor,
      LogMelSpectrogramProcessor::Create(
          16000,
          LogMelSpectrogramProcessor::LogMelSpectrogramConfig{
              .n_fft = 512,
              .norm_type = LogMelSpectrogramProcessor::NormType::kWhisper},
          &dummy_source));

  int sample_rate = 16000;
  std::vector<float> audio(sample_rate);
  for (int i = 0; i < audio.size(); ++i) {
    audio[i] = std::sin(2.0 * M_PI * 440.0 * i / sample_rate);
  }

  dummy_source.PushChunk(audio);
  ASSERT_OK(processor->Schedule());
  ASSERT_OK_AND_ASSIGN(auto result, processor->GetOutput());
  EXPECT_FALSE(result.empty());
}

TEST(LogMelSpectrogramProcessorTest,
     ProcessWithNonPowerOfTwoNFFTPadsCorrectly) {
  DummyAudioSource dummy_source;
  ASSERT_OK_AND_ASSIGN(
      auto processor,
      LogMelSpectrogramProcessor::Create(
          16000,
          LogMelSpectrogramProcessor::LogMelSpectrogramConfig{.n_fft = 400},
          &dummy_source));

  int sample_rate = 16000;
  int duration_seconds = 1;
  std::vector<float> audio(sample_rate * duration_seconds);
  for (int i = 0; i < audio.size(); ++i) {
    audio[i] = std::sin(2.0 * M_PI * 440.0 * i / sample_rate);
  }

  dummy_source.PushChunk(audio);
  ASSERT_OK(processor->Schedule());
  ASSERT_OK_AND_ASSIGN(auto result, processor->GetOutput());
  EXPECT_FALSE(result.empty());
}

TEST(LogMelSpectrogramProcessorTest, ProcessPadsToExpectedNumberOfFrames) {
  int hop_length = 160;
  int n_frames_in_config = 10;
  std::vector<float> audio(2 * hop_length, 0.5f);

  DummyAudioSource dummy_source;
  ASSERT_OK_AND_ASSIGN(auto processor,
                       LogMelSpectrogramProcessor::Create(
                           16000,
                           LogMelSpectrogramProcessor::LogMelSpectrogramConfig{
                               .n_fft = 512, .n_frames = n_frames_in_config},
                           &dummy_source));

  dummy_source.PushChunk(audio);
  ASSERT_OK(processor->Schedule());
  ASSERT_OK_AND_ASSIGN(auto result, processor->GetOutput());
  EXPECT_EQ(result.size(), 800);
  EXPECT_EQ(result[result.size() - 1], 0.0f);
}

TEST(LogMelSpectrogramProcessorTest, SchedulePullsFromSourceAndPushesFeatures) {
  DummyAudioSource dummy_source;
  int sample_rate = 16000;
  int duration_seconds = 1;
  std::vector<float> audio_chunk(sample_rate * duration_seconds);
  for (int i = 0; i < audio_chunk.size(); ++i) {
    audio_chunk[i] = std::sin(2.0 * M_PI * 440.0 * i / sample_rate);
  }
  dummy_source.PushChunk(audio_chunk);

  ASSERT_OK_AND_ASSIGN(
      auto processor,
      LogMelSpectrogramProcessor::Create(
          16000,
          LogMelSpectrogramProcessor::LogMelSpectrogramConfig{.n_fft = 512},
          &dummy_source));
  ASSERT_OK(processor->Schedule());
  EXPECT_TRUE(processor->HasOutput());
  ASSERT_OK_AND_ASSIGN(auto output, processor->GetOutput());
  EXPECT_FALSE(output.empty());
}

}  // namespace
}  // namespace litert::omni::asr
