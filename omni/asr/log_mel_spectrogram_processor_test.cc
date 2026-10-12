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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
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
  EXPECT_EQ(processor->last_valid_frames(), 2);
}

// Generates `num_samples` of deterministic, broadband, amplitude-modulated
// noise so that every mel bin carries energy well above the mel floor and
// varies over time (like speech), unlike a pure tone.
std::vector<float> MakeTestAudio(int num_samples) {
  std::vector<float> audio(num_samples);
  uint32_t state = 12345u;
  for (int i = 0; i < num_samples; ++i) {
    state = state * 1664525u + 1013904223u;  // LCG, deterministic.
    const float noise = static_cast<float>(state >> 8) / 16777216.0f - 0.5f;
    const float t = static_cast<float>(i) / 16000.0f;
    const float envelope =
        0.3f + 0.7f * std::abs(std::sin(2.0f * M_PI * 1.5f * t));
    audio[i] = envelope * noise;
  }
  return audio;
}

TEST(LogMelSpectrogramProcessorTest,
     ShortChunkIsNormalizedOverValidFramesOnlyAndPaddedWithSilence) {
  constexpr int kHop = 160;
  constexpr int kNMels = 80;
  constexpr int kWindowFrames = 500;             // 5 s model window.
  constexpr int kSpeechSamples = 16000 * 6 / 5;  // 1.2 s -> 120 valid frames.
  constexpr int kValidFrames = kSpeechSamples / kHop;
  const std::vector<float> audio = MakeTestAudio(kSpeechSamples);

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
  EXPECT_EQ(processor->last_valid_frames(), kValidFrames);

  for (int m = 0; m < kNMels; ++m) {
    // Per-feature statistics must come from the real-audio frames only: those
    // frames have zero mean and unit (sample) variance after normalization.
    // With the previous behavior (statistics over the zero-padded 5 s window)
    // the speech frames instead sit around +1.5 sigma.
    double sum = 0.0;
    for (int f = 0; f < kValidFrames; ++f) {
      sum += result[m * kWindowFrames + f];
    }
    const double mean = sum / kValidFrames;
    double sum_sq = 0.0;
    float speech_min = std::numeric_limits<float>::infinity();
    for (int f = 0; f < kValidFrames; ++f) {
      const float v = result[m * kWindowFrames + f];
      sum_sq += (v - mean) * (v - mean);
      speech_min = std::min(speech_min, v);
    }
    const double std = std::sqrt(sum_sq / (kValidFrames - 1));
    EXPECT_THAT(mean, testing::DoubleNear(0.0, 1e-3)) << "mel=" << m;
    EXPECT_THAT(std, testing::DoubleNear(1.0, 1e-2)) << "mel=" << m;

    // Padded frames are exactly 0 (the per-feature mean) after normalization,
    // like the reference pipeline's zero-padded normalized features.
    for (int f = kValidFrames; f < kWindowFrames; ++f) {
      ASSERT_EQ(result[m * kWindowFrames + f], 0.0f)
          << "mel=" << m << " frame=" << f;
    }
  }
}

TEST(LogMelSpectrogramProcessorTest, FullWindowChunkIsUnchangedByPaddingLogic) {
  constexpr int kHop = 160;
  constexpr int kNMels = 80;
  constexpr int kWindowFrames = 500;
  const std::vector<float> audio = MakeTestAudio(kWindowFrames * kHop);

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
  EXPECT_EQ(processor->last_valid_frames(), kWindowFrames);

  // Standard normalization over all (valid) frames: ~zero mean per feature.
  for (int m = 0; m < kNMels; ++m) {
    float sum = 0.0f;
    int count = 0;
    for (int f = 0; f < kWindowFrames; ++f) {
      // The STFT yields slightly fewer than 500 frames; the last couple of
      // frames are structural zero padding and excluded from the mean check.
      if (f >= kWindowFrames - 3) break;
      sum += result[m * kWindowFrames + f];
      ++count;
    }
    EXPECT_THAT(sum / count, FloatNear(0.0f, 2e-2f)) << "mel=" << m;
  }
}

TEST(LogMelSpectrogramProcessorTest, TinyChunkIsSkippedLikeSilence) {
  DummyAudioSource source;
  ASSERT_OK_AND_ASSIGN(auto processor,
                       LogMelSpectrogramProcessor::Create(
                           16000,
                           LogMelSpectrogramProcessor::LogMelSpectrogramConfig{
                               .n_fft = 512, .n_frames = 500},
                           &source));
  // 100 samples (< 2 hops): no meaningful statistics, so no features.
  std::vector<float> audio(100);
  for (int i = 0; i < audio.size(); ++i) {
    audio[i] = std::sin(0.3f * i);
  }
  source.PushChunk(audio);
  ASSERT_OK(processor->Schedule());
  ASSERT_OK_AND_ASSIGN(auto result, processor->GetOutput());
  EXPECT_TRUE(result.empty());
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
  constexpr int kHop = 160;
  constexpr int kNMels = 80;
  constexpr int kWindowFrames = 500;
  constexpr int kWindowSamples = kWindowFrames * kHop;
  const std::vector<float> first = MakeTestAudio(kWindowSamples);
  std::vector<float> second = MakeTestAudio(kWindowSamples);
  for (float& s : second) {
    s *= 0.25f;
  }
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

  ASSERT_EQ(second_after_first.size(), second_fresh.size());
  for (size_t i = 0; i < second_fresh.size(); ++i) {
    ASSERT_FLOAT_EQ(second_after_first[i], second_fresh[i]) << "i=" << i;
  }
}

TEST(LogMelSpectrogramProcessorTest,
     ShortUtteranceWithTrailingRoomNoiseTrimsValidFrames) {
  constexpr int kHop = 160;
  constexpr int kNMels = 80;
  constexpr int kWindowFrames = 500;
  constexpr int kWindowSamples = kWindowFrames * kHop;
  constexpr int kSpeechSamples = 16000;  // 1.0 s active speech (10 blocks).
  std::vector<float> audio = MakeTestAudio(kWindowSamples);
  // Attenuate the trailing 4.0 s (40 blocks >= 20 blocks) to low-level room
  // noise (~1e-3 RMS, well below 6e-3 and < 5% of peak speech RMS).
  for (int i = kSpeechSamples; i < kWindowSamples; ++i) {
    audio[i] *= 0.005f;
  }

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
  EXPECT_EQ(processor->last_valid_frames(), 110);
  EXPECT_FLOAT_EQ(processor->last_chunk_valid_fraction(), 110.0f / 500.0f);
  // Frames after `last_valid_frames()` are zeroed after normalization.
  for (int m = 0; m < kNMels; ++m) {
    for (int f = processor->last_valid_frames(); f < kWindowFrames; ++f) {
      ASSERT_EQ(result[m * kWindowFrames + f], 0.0f)
          << "mel=" << m << " frame=" << f;
    }
  }
}

TEST(LogMelSpectrogramProcessorTest,
     DummyPreprocessorPadsShortChunkWithDitherAndSuppressesSilentChunk) {
  constexpr int kWindowSamples = 16000;
  constexpr int kSpeechSamples = 4800;
  DummyAudioSource source;
  DummyPreprocessor processor(&source, kWindowSamples);

  // Silent chunk is suppressed to an empty feature vector.
  source.PushChunk(std::vector<float>(kSpeechSamples, 0.0f));
  ASSERT_OK(processor.Schedule());
  ASSERT_OK_AND_ASSIGN(auto silent_out, processor.GetOutput());
  EXPECT_TRUE(silent_out.empty());

  // Short speech chunk is padded to `kWindowSamples` with non-zero dither.
  const std::vector<float> speech = MakeTestAudio(kSpeechSamples);
  source.PushChunk(speech);
  ASSERT_OK(processor.Schedule());
  ASSERT_OK_AND_ASSIGN(auto padded_out, processor.GetOutput());
  ASSERT_EQ(padded_out.size(), kWindowSamples);
  EXPECT_FLOAT_EQ(processor.last_chunk_start_fraction(), 0.0f);
  EXPECT_FLOAT_EQ(processor.last_chunk_valid_fraction(),
                  static_cast<float>(kSpeechSamples) / kWindowSamples);
  for (int i = 0; i < kSpeechSamples; ++i) {
    ASSERT_FLOAT_EQ(padded_out[i], speech[i]) << "i=" << i;
  }
  bool has_non_zero_dither = false;
  for (int i = kSpeechSamples; i < kWindowSamples; ++i) {
    EXPECT_LE(std::abs(padded_out[i]), 3.5e-3f);
    if (padded_out[i] != 0.0f) {
      has_non_zero_dither = true;
    }
  }
  EXPECT_TRUE(has_non_zero_dither);
}

TEST(LogMelSpectrogramProcessorTest,
     WhisperNormalizationUsesCenteredReflectPaddingFraming) {
  constexpr int kHop = 160;
  constexpr int kNMels = 80;
  // 1.0 s -> 100 centered frames (vs 98 unpadded).
  constexpr int kSamples = 16000;
  const std::vector<float> audio = MakeTestAudio(kSamples);

  DummyAudioSource source;
  ASSERT_OK_AND_ASSIGN(
      auto processor,
      LogMelSpectrogramProcessor::Create(
          16000,
          LogMelSpectrogramProcessor::LogMelSpectrogramConfig{
              .n_fft = 400,
              .n_mels = kNMels,
              .hop_length = kHop,
              .norm_type = LogMelSpectrogramProcessor::NormType::kWhisper},
          &source));
  source.PushChunk(audio);
  ASSERT_OK(processor->Schedule());
  ASSERT_OK_AND_ASSIGN(auto result, processor->GetOutput());
  // With `center=True` reflect padding of `n_fft / 2 = 200` on the left and
  // `n_fft - hop_length - n_fft / 2 = 40` on the right, 16,000 samples yield
  // exactly `16,000 / 160 = 100` STFT frames.
  EXPECT_EQ(result.size(), (kSamples / kHop) * kNMels);
}

TEST(LogMelSpectrogramProcessorTest,
     ShortChunkAfterFullWindowIsShiftedBackOverPreviousWindow) {
  constexpr int kHop = 160;
  constexpr int kNMels = 80;
  constexpr int kWindowFrames = 500;
  constexpr int kWindowSamples = kWindowFrames * kHop;
  constexpr int kTailSamples = 16000 * 2;  // 2 s final chunk.
  // A 7 s stream: one full 5 s window followed by a 2 s final chunk.
  const std::vector<float> stream =
      MakeTestAudio(kWindowSamples + kTailSamples);
  const std::vector<float> first(stream.begin(),
                                 stream.begin() + kWindowSamples);
  const std::vector<float> tail(stream.begin() + kWindowSamples, stream.end());
  const LogMelSpectrogramProcessor::LogMelSpectrogramConfig config{
      .n_fft = 512,
      .n_mels = kNMels,
      .hop_length = kHop,
      .n_frames = kWindowFrames};

  DummyAudioSource source;
  ASSERT_OK_AND_ASSIGN(auto processor, LogMelSpectrogramProcessor::Create(
                                           16000, config, &source));
  ASSERT_TRUE(processor->ShiftsShortChunkBack());
  source.PushChunk(first);
  ASSERT_OK(processor->Schedule());
  ASSERT_OK_AND_ASSIGN(auto first_result, processor->GetOutput());
  EXPECT_EQ(processor->last_chunk_start_fraction(), 0.0f);
  EXPECT_EQ(processor->last_chunk_valid_fraction(), 1.0f);

  source.PushChunk(tail);
  ASSERT_OK(processor->Schedule());
  ASSERT_OK_AND_ASSIGN(auto shifted_result, processor->GetOutput());
  ASSERT_EQ(shifted_result.size(), kWindowFrames * kNMels);
  // The chunk was rebuilt as the last 5 s of the stream: 3 s of it were
  // already covered by the previous window, all of it is real audio.
  EXPECT_THAT(processor->last_chunk_start_fraction(),
              FloatNear(static_cast<float>(kWindowSamples - kTailSamples) /
                            kWindowSamples,
                        1e-6f));
  EXPECT_EQ(processor->last_valid_frames(), kWindowFrames);
  EXPECT_EQ(processor->last_chunk_valid_fraction(), 1.0f);

  // Same features as processing the stream's last window directly on a fresh
  // processor, i.e. no padding or stale STFT queue samples are involved.
  DummyAudioSource reference_source;
  ASSERT_OK_AND_ASSIGN(auto reference, LogMelSpectrogramProcessor::Create(
                                           16000, config, &reference_source));
  reference_source.PushChunk(
      std::vector<float>(stream.end() - kWindowSamples, stream.end()));
  ASSERT_OK(reference->Schedule());
  ASSERT_OK_AND_ASSIGN(auto reference_result, reference->GetOutput());
  EXPECT_EQ(reference->last_chunk_start_fraction(), 0.0f);
  ASSERT_EQ(reference_result.size(), shifted_result.size());
  for (size_t i = 0; i < shifted_result.size(); ++i) {
    ASSERT_FLOAT_EQ(shifted_result[i], reference_result[i]) << "i=" << i;
  }

  // Reset forgets the previous window: a short chunk is padded again.
  processor->Reset();
  source.PushChunk(tail);
  ASSERT_OK(processor->Schedule());
  ASSERT_OK_AND_ASSIGN(auto padded_result, processor->GetOutput());
  EXPECT_EQ(processor->last_chunk_start_fraction(), 0.0f);
  EXPECT_EQ(processor->last_valid_frames(), kTailSamples / kHop);
  EXPECT_EQ(padded_result[0 * kWindowFrames + kWindowFrames - 1], 0.0f);
}

TEST(LogMelSpectrogramProcessorTest, WhisperShortChunkIsPaddedNotShifted) {
  constexpr int kHop = 160;
  constexpr int kNMels = 80;
  constexpr int kWindowFrames = 500;
  constexpr int kWindowSamples = kWindowFrames * kHop;
  const std::vector<float> stream = MakeTestAudio(kWindowSamples + 16000);

  DummyAudioSource source;
  ASSERT_OK_AND_ASSIGN(
      auto processor,
      LogMelSpectrogramProcessor::Create(
          16000,
          LogMelSpectrogramProcessor::LogMelSpectrogramConfig{
              .n_fft = 400,
              .n_mels = kNMels,
              .hop_length = kHop,
              .n_frames = kWindowFrames,
              .norm_type = LogMelSpectrogramProcessor::NormType::kWhisper},
          &source));
  EXPECT_FALSE(processor->ShiftsShortChunkBack());
  source.PushChunk(
      std::vector<float>(stream.begin(), stream.begin() + kWindowSamples));
  ASSERT_OK(processor->Schedule());
  ASSERT_OK(processor->GetOutput().status());
  source.PushChunk(
      std::vector<float>(stream.begin() + kWindowSamples, stream.end()));
  ASSERT_OK(processor->Schedule());
  ASSERT_OK(processor->GetOutput().status());
  EXPECT_EQ(processor->last_chunk_start_fraction(), 0.0f);
  EXPECT_EQ(processor->last_valid_frames(), 16000 / kHop);
}

TEST(LogMelSpectrogramProcessorTest,
     OverlappingShortFinalChunkDeduplicatesOverlapWhenShiftedBack) {
  constexpr int kHop = 160;
  constexpr int kNMels = 80;
  constexpr int kWindowFrames = 500;
  constexpr int kWindowSamples = kWindowFrames * kHop;
  constexpr int kOverlapSamples = 200 * kHop;  // 2 s overlap.
  constexpr int kNewSamples = 100 * kHop;      // 1 s new samples.
  // 6 s stream: 5 s first window, then 3 s final chunk (2 s overlap + 1 s new).
  const std::vector<float> stream = MakeTestAudio(kWindowSamples + kNewSamples);
  const std::vector<float> first(stream.begin(),
                                 stream.begin() + kWindowSamples);
  const std::vector<float> overlapping_tail(
      stream.begin() + kWindowSamples - kOverlapSamples, stream.end());
  const LogMelSpectrogramProcessor::LogMelSpectrogramConfig config{
      .n_fft = 512,
      .n_mels = kNMels,
      .hop_length = kHop,
      .n_frames = kWindowFrames};

  DummyAudioSource source;
  ASSERT_OK_AND_ASSIGN(auto processor, LogMelSpectrogramProcessor::Create(
                                           16000, config, &source));
  source.PushChunk(first);
  ASSERT_OK(processor->Schedule());
  ASSERT_OK(processor->GetOutput().status());

  source.PushChunk(overlapping_tail);
  ASSERT_OK(processor->Schedule());
  ASSERT_OK_AND_ASSIGN(auto shifted_result, processor->GetOutput());

  DummyAudioSource reference_source;
  ASSERT_OK_AND_ASSIGN(auto reference, LogMelSpectrogramProcessor::Create(
                                           16000, config, &reference_source));
  reference_source.PushChunk(
      std::vector<float>(stream.end() - kWindowSamples, stream.end()));
  ASSERT_OK(reference->Schedule());
  ASSERT_OK_AND_ASSIGN(auto reference_result, reference->GetOutput());
  ASSERT_EQ(shifted_result.size(), reference_result.size());
  for (size_t i = 0; i < shifted_result.size(); ++i) {
    ASSERT_FLOAT_EQ(shifted_result[i], reference_result[i]) << "i=" << i;
  }
  EXPECT_FALSE(processor->last_chunk_centered_features().empty());
}

TEST(LogMelSpectrogramProcessorTest,
     DummyPreprocessorShiftsShortFinalChunkBackAndSuppressesTrailingRoomNoise) {
  constexpr int kWindowSamples = 16000;
  constexpr int kOverlapSamples = 3200;
  constexpr int kNewSamples = 3200;
  const std::vector<float> stream = MakeTestAudio(kWindowSamples + kNewSamples);
  DummyAudioSource source;
  DummyPreprocessor processor(&source, kWindowSamples);

  // Full first window.
  source.PushChunk(
      std::vector<float>(stream.begin(), stream.begin() + kWindowSamples));
  ASSERT_OK(processor.Schedule());
  ASSERT_OK_AND_ASSIGN(auto first_out, processor.GetOutput());
  ASSERT_EQ(first_out.size(), kWindowSamples);
  EXPECT_FLOAT_EQ(processor.last_chunk_start_fraction(), 0.0f);
  EXPECT_FLOAT_EQ(processor.last_chunk_valid_fraction(), 1.0f);

  // Overlapping short final chunk (overlap + new = 6400 < 16000) is shifted
  // back over the previous window without duplicating the overlap.
  source.PushChunk(std::vector<float>(
      stream.begin() + kWindowSamples - kOverlapSamples, stream.end()));
  ASSERT_OK(processor.Schedule());
  ASSERT_OK_AND_ASSIGN(auto shifted_out, processor.GetOutput());
  ASSERT_EQ(shifted_out.size(), kWindowSamples);
  EXPECT_FLOAT_EQ(
      processor.last_chunk_start_fraction(),
      static_cast<float>(kWindowSamples - (kOverlapSamples + kNewSamples)) /
          kWindowSamples);
  EXPECT_FLOAT_EQ(processor.last_chunk_valid_fraction(), 1.0f);
  for (int i = 0; i < kWindowSamples; ++i) {
    ASSERT_FLOAT_EQ(shifted_out[i], stream[kNewSamples + i]) << "i=" << i;
  }

  // Follow-up chunk containing only low-level room noise relative to earlier
  // speech RMS is suppressed.
  std::vector<float> room_noise = MakeTestAudio(kWindowSamples);
  for (float& s : room_noise) s *= 0.005f;
  source.PushChunk(room_noise);
  ASSERT_OK(processor.Schedule());
  ASSERT_OK_AND_ASSIGN(auto noise_out, processor.GetOutput());
  EXPECT_TRUE(noise_out.empty());
}

}  // namespace
}  // namespace litert::omni::asr
