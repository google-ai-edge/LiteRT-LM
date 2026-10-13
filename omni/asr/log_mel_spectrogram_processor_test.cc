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
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "omni/asr/dummy_preprocessor.h"
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

TEST(LogMelSpectrogramProcessorTest, ProcessSilenceAudioReturnsEmptyVector) {
  DummyAudioSource dummy_source;
  ASSERT_OK_AND_ASSIGN(
      auto processor,
      LogMelSpectrogramProcessor::Create(
          16000,
          LogMelSpectrogramProcessor::LogMelSpectrogramConfig{.n_fft = 512},
          &dummy_source));
  dummy_source.PushChunk(std::vector<float>(16000, 0.0f));
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
}

std::vector<float> MakeTestAudio(int num_samples) {
  std::vector<float> audio(num_samples);
  uint32_t state = 12345u;
  for (int i = 0; i < num_samples; ++i) {
    state = state * 1664525u + 1013904223u;
    const float noise = static_cast<float>(state >> 8) / 16777216.0f - 0.5f;
    const float t = static_cast<float>(i) / 16000.0f;
    audio[i] =
        (0.3f + 0.7f * std::abs(std::sin(2.0f * M_PI * 1.5f * t))) * noise;
  }
  return audio;
}

TEST(LogMelSpectrogramProcessorTest,
     ShortChunkIsNormalizedOverValidFramesOnlyAndPaddedWithSilence) {
  constexpr int kHop = 160;
  constexpr int kNMels = 80;
  constexpr int kWindowFrames = 500;
  constexpr int kSpeechSamples = 16000 * 6 / 5;
  constexpr int kValidFrames = kSpeechSamples / kHop;

  DummyAudioSource source;
  ASSERT_OK_AND_ASSIGN(auto processor,
                       LogMelSpectrogramProcessor::Create(
                           16000,
                           LogMelSpectrogramProcessor::LogMelSpectrogramConfig{
                               .n_fft = 512,
                               .n_mels = kNMels,
                               .hop_length = kHop,
                               .n_frames = kWindowFrames},
                           &source));
  source.PushChunk(MakeTestAudio(kSpeechSamples));
  ASSERT_OK(processor->Schedule());
  ASSERT_OK_AND_ASSIGN(auto result, processor->GetOutput());
  ASSERT_EQ(result.size(), kWindowFrames * kNMels);

  for (int m = 0; m < kNMels; ++m) {
    double sum = 0.0;
    for (int f = 0; f < kValidFrames; ++f) {
      sum += result[m * kWindowFrames + f];
    }
    const double mean = sum / kValidFrames;
    double sum_sq = 0.0;
    for (int f = 0; f < kValidFrames; ++f) {
      const float v = result[m * kWindowFrames + f];
      sum_sq += (v - mean) * (v - mean);
    }
    EXPECT_THAT(mean, testing::DoubleNear(0.0, 1e-3)) << "mel=" << m;
    EXPECT_THAT(std::sqrt(sum_sq / (kValidFrames - 1)),
                testing::DoubleNear(1.0, 1e-2))
        << "mel=" << m;
    for (int f = kValidFrames; f < kWindowFrames; ++f) {
      ASSERT_EQ(result[m * kWindowFrames + f], 0.0f);
    }
  }
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

TEST(LogMelSpectrogramProcessorTest,
     ProcessWithUsmNormalizationProduces128Mels) {
  DummyAudioSource dummy_source;
  ASSERT_OK_AND_ASSIGN(auto processor,
                       LogMelSpectrogramProcessor::Create(
                           16000,
                           LogMelSpectrogramProcessor::LogMelSpectrogramConfig{
                               .n_fft = 512,
                               .n_mels = 128,
                               .transpose = true,
                               .preemphasis = 0.97f,
                               .fft_length = 1024,
                               .input_scale = 32768.0f,
                               .mel_low_hz = 125.0f,
                               .mel_high_hz = 7500.0f,
                               .mel_floor = 1e-6f,
                               .normalize_mel = true,
                               .add_floor_to_mel_before_log = false},
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
  constexpr int kUsmMelBins = 128;
  EXPECT_EQ(result.size() % kUsmMelBins, 0);
  EXPECT_TRUE(processor->config().transpose);
}

TEST(LogMelSpectrogramProcessorTest,
     OverlappingWindowsDoNotCarryTrailingStftSamplesAcrossChunks) {
  constexpr int kHop = 160, kNMels = 80, kWindowFrames = 500;
  const std::vector<float> first = MakeTestAudio(kWindowFrames * kHop);
  std::vector<float> second = first;
  for (float& s : second) s *= 0.25f;
  const LogMelSpectrogramProcessor::LogMelSpectrogramConfig config{
      .n_fft = 512,
      .n_mels = kNMels,
      .hop_length = kHop,
      .n_frames = kWindowFrames};

  DummyAudioSource seq_source;
  ASSERT_OK_AND_ASSIGN(auto seq_processor, LogMelSpectrogramProcessor::Create(
                                               16000, config, &seq_source));
  seq_source.PushChunk(first);
  ASSERT_OK(seq_processor->Schedule());
  ASSERT_OK(seq_processor->GetOutput().status());
  seq_source.PushChunk(second);
  ASSERT_OK(seq_processor->Schedule());
  ASSERT_OK_AND_ASSIGN(auto second_after_first, seq_processor->GetOutput());

  DummyAudioSource fresh_source;
  ASSERT_OK_AND_ASSIGN(auto fresh_processor, LogMelSpectrogramProcessor::Create(
                                                 16000, config, &fresh_source));
  fresh_source.PushChunk(second);
  ASSERT_OK(fresh_processor->Schedule());
  ASSERT_OK_AND_ASSIGN(auto second_fresh, fresh_processor->GetOutput());
  EXPECT_EQ(second_after_first, second_fresh);
}

TEST(LogMelSpectrogramProcessorTest,
     ShortUtteranceWithTrailingRoomNoiseTrimsValidFrames) {
  constexpr int kHop = 160, kNMels = 80, kWindowFrames = 500;
  std::vector<float> audio = MakeTestAudio(kWindowFrames * kHop);
  for (int i = 16000; i < kWindowFrames * kHop; ++i) audio[i] *= 0.005f;

  DummyAudioSource source;
  ASSERT_OK_AND_ASSIGN(auto processor,
                       LogMelSpectrogramProcessor::Create(
                           16000,
                           LogMelSpectrogramProcessor::LogMelSpectrogramConfig{
                               .n_fft = 512,
                               .n_mels = kNMels,
                               .hop_length = kHop,
                               .n_frames = kWindowFrames},
                           &source));
  source.PushChunk(audio);
  ASSERT_OK(processor->Schedule());
  ASSERT_OK_AND_ASSIGN(auto result, processor->GetOutput());
  ASSERT_EQ(result.size(), kWindowFrames * kNMels);
  // 10 active 100 ms blocks + 1 margin block = 110 valid frames.
  EXPECT_NE(result[109], 0.0f);
  for (int m = 0; m < kNMels; ++m) {
    for (int f = 110; f < kWindowFrames; ++f) {
      ASSERT_EQ(result[m * kWindowFrames + f], 0.0f);
    }
  }
}

TEST(LogMelSpectrogramProcessorTest,
     DummyPreprocessorPadsShortChunkWithDitherAndSuppressesSilentChunk) {
  constexpr int kWindowSamples = 16000, kSpeechSamples = 4800;
  DummyAudioSource source;
  DummyPreprocessor processor(&source, kWindowSamples);

  source.PushChunk(std::vector<float>(kSpeechSamples, 0.0f));
  ASSERT_OK(processor.Schedule());
  ASSERT_OK_AND_ASSIGN(auto silent_out, processor.GetOutput());
  EXPECT_TRUE(silent_out.empty());

  const std::vector<float> speech = MakeTestAudio(kSpeechSamples);
  source.PushChunk(speech);
  ASSERT_OK(processor.Schedule());
  ASSERT_OK_AND_ASSIGN(auto padded_out, processor.GetOutput());
  ASSERT_EQ(padded_out.size(), kWindowSamples);
  for (int i = 0; i < kSpeechSamples; ++i) {
    ASSERT_FLOAT_EQ(padded_out[i], speech[i]);
  }
  bool has_non_zero_dither = false;
  for (int i = kSpeechSamples; i < kWindowSamples; ++i) {
    EXPECT_LE(std::abs(padded_out[i]), 3.5e-3f);
    has_non_zero_dither = has_non_zero_dither || (padded_out[i] != 0.0f);
  }
  EXPECT_TRUE(has_non_zero_dither);
}

}  // namespace
}  // namespace litert::omni::asr
