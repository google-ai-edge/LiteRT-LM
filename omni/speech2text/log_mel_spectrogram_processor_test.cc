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

#include "omni/speech2text/log_mel_spectrogram_processor.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "omni/speech2text/audio_source.h"
#include "omni/speech2text/dummy_preprocessor.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep for ASSERT_OK

namespace litert::omni::asr {
namespace {

using ::testing::FloatNear;

// Dummy AudioSource for testing stage scheduling.
class DummyAudioSource : public AudioSource {
 public:
  DummyAudioSource() = default;
  int GetSampleRateHz() const override { return 16000; }
  int GetNumChannels() const override { return 1; }
  bool IsLastOutputWarmStart() const override { return last_was_warm_start_; }

  void PushChunk(std::vector<float> chunk, bool is_warm_start = false) {
    last_was_warm_start_ = is_warm_start;
    PushOutput(std::move(chunk));
  }

 protected:
  bool NeedScheduleInternal() const override { return false; }
  absl::Status ScheduleInternal() override { return absl::OkStatus(); }

 private:
  bool last_was_warm_start_ = false;
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
  EXPECT_TRUE(result.features.empty());
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
  EXPECT_TRUE(result.features.empty());
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
  EXPECT_FALSE(result.features.empty());
  int feature_size = 80;
  EXPECT_EQ(result.features.size() % feature_size, 0);

  bool has_non_zero = false;
  for (float val : result.features) {
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
  EXPECT_FALSE(result.features.empty());

  // In standard normalization, features across valid frames should have roughly
  // zero mean and unit variance.
  int n_mels = 80;
  int num_frames = result.features.size() / n_mels;
  EXPECT_GT(num_frames, 0);

  for (int m = 0; m < n_mels; ++m) {
    float sum = 0.0f;
    for (int f = 0; f < num_frames; ++f) {
      sum += result.features[m * num_frames + f];
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
  EXPECT_FALSE(result.features.empty());
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
  EXPECT_FALSE(result.features.empty());
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
  EXPECT_EQ(result.features.size(), 800);
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
  ASSERT_EQ(result.features.size(), kWindowFrames * kNMels);

  for (int m = 0; m < kNMels; ++m) {
    double sum = 0.0;
    for (int f = 0; f < kValidFrames; ++f) {
      sum += result.features[m * kWindowFrames + f];
    }
    const double mean = sum / kValidFrames;
    double sum_sq = 0.0;
    for (int f = 0; f < kValidFrames; ++f) {
      const float v = result.features[m * kWindowFrames + f];
      sum_sq += (v - mean) * (v - mean);
    }
    EXPECT_THAT(mean, testing::DoubleNear(0.0, 1e-3)) << "mel=" << m;
    EXPECT_THAT(std::sqrt(sum_sq / (kValidFrames - 1)),
                testing::DoubleNear(1.0, 1e-2))
        << "mel=" << m;
    for (int f = kValidFrames; f < kWindowFrames; ++f) {
      ASSERT_EQ(result.features[m * kWindowFrames + f], 0.0f);
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
  EXPECT_FALSE(output.features.empty());
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
  EXPECT_FALSE(result.features.empty());
  constexpr int kUsmMelBins = 128;
  EXPECT_EQ(result.features.size() % kUsmMelBins, 0);
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
  EXPECT_EQ(second_after_first.features, second_fresh.features);
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
  ASSERT_EQ(result.features.size(), kWindowFrames * kNMels);
  // 10 active 100 ms blocks + 1 margin block = 110 valid frames.
  EXPECT_NE(result.features[109], 0.0f);
  ASSERT_TRUE(result.decode_range.has_value());
  EXPECT_FLOAT_EQ(result.decode_range->valid_fraction, 110.0f / 500.0f);
  for (int m = 0; m < kNMels; ++m) {
    for (int f = 110; f < kWindowFrames; ++f) {
      ASSERT_EQ(result.features[m * kWindowFrames + f], 0.0f);
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
  EXPECT_TRUE(silent_out.features.empty());

  const std::vector<float> speech = MakeTestAudio(kSpeechSamples);
  source.PushChunk(speech);
  ASSERT_OK(processor.Schedule());
  ASSERT_OK_AND_ASSIGN(auto padded_out, processor.GetOutput());
  ASSERT_EQ(padded_out.features.size(), kWindowSamples);
  ASSERT_TRUE(padded_out.decode_range.has_value());
  EXPECT_EQ(padded_out.decode_range->start_fraction, 0.0f);
  EXPECT_FLOAT_EQ(padded_out.decode_range->valid_fraction,
                  static_cast<float>(kSpeechSamples) / kWindowSamples);
  for (int i = 0; i < kSpeechSamples; ++i) {
    ASSERT_FLOAT_EQ(padded_out.features[i], speech[i]);
  }
  bool has_non_zero_dither = false;
  for (int i = kSpeechSamples; i < kWindowSamples; ++i) {
    EXPECT_LE(std::abs(padded_out.features[i]), 3.5e-3f);
    has_non_zero_dither =
        has_non_zero_dither || (padded_out.features[i] != 0.0f);
  }
  EXPECT_TRUE(has_non_zero_dither);
}

TEST(LogMelSpectrogramProcessorTest,
     DummyPreprocessorShiftsShortFollowUpChunkBackOverPreviousWindow) {
  constexpr int kWindowSamples = 16000 * 5;
  constexpr int kOverlapSamples = 16000 * 2;
  constexpr int kNewSamples = 16000;
  const std::vector<float> stream = MakeTestAudio(kWindowSamples + kNewSamples);
  const std::vector<float> first(stream.begin(),
                                 stream.begin() + kWindowSamples);
  const std::vector<float> overlapping_tail(
      stream.begin() + kWindowSamples - kOverlapSamples, stream.end());

  DummyAudioSource source;
  DummyPreprocessor processor(&source, kWindowSamples);

  source.PushChunk(first);
  ASSERT_OK(processor.Schedule());
  ASSERT_OK_AND_ASSIGN(auto first_out, processor.GetOutput());
  EXPECT_EQ(first_out.features, first);
  ASSERT_TRUE(first_out.decode_range.has_value());
  EXPECT_EQ(first_out.decode_range->start_fraction, 0.0f);
  EXPECT_EQ(first_out.decode_range->valid_fraction, 1.0f);

  source.PushChunk(overlapping_tail);
  ASSERT_OK(processor.Schedule());
  ASSERT_OK_AND_ASSIGN(auto shifted_out, processor.GetOutput());
  EXPECT_EQ(shifted_out.features,
            std::vector<float>(stream.end() - kWindowSamples, stream.end()));
  ASSERT_TRUE(shifted_out.decode_range.has_value());
  EXPECT_FLOAT_EQ(shifted_out.decode_range->start_fraction,
                  static_cast<float>(kWindowSamples - overlapping_tail.size()) /
                      static_cast<float>(kWindowSamples));
  EXPECT_EQ(shifted_out.decode_range->valid_fraction, 1.0f);

  // A silent chunk clears `previous_window_`, so the next short chunk is
  // padded rather than shifted back over pre-silence audio.
  source.PushChunk(std::vector<float>(kWindowSamples, 0.0f));
  ASSERT_OK(processor.Schedule());
  ASSERT_OK_AND_ASSIGN(auto silent_out, processor.GetOutput());
  EXPECT_TRUE(silent_out.features.empty());

  source.PushChunk(overlapping_tail);
  ASSERT_OK(processor.Schedule());
  ASSERT_OK_AND_ASSIGN(auto after_silence_out, processor.GetOutput());
  ASSERT_EQ(after_silence_out.features.size(), kWindowSamples);
  ASSERT_TRUE(after_silence_out.decode_range.has_value());
  EXPECT_EQ(after_silence_out.decode_range->start_fraction, 0.0f);
  EXPECT_FLOAT_EQ(after_silence_out.decode_range->valid_fraction,
                  static_cast<float>(overlapping_tail.size()) /
                      static_cast<float>(kWindowSamples));

  // Reset also clears `previous_window_`.
  source.PushChunk(first);
  ASSERT_OK(processor.Schedule());
  ASSERT_OK(processor.GetOutput().status());
  processor.Reset();

  source.PushChunk(overlapping_tail);
  ASSERT_OK(processor.Schedule());
  ASSERT_OK_AND_ASSIGN(auto after_reset_out, processor.GetOutput());
  ASSERT_TRUE(after_reset_out.decode_range.has_value());
  EXPECT_EQ(after_reset_out.decode_range->start_fraction, 0.0f);
  EXPECT_FLOAT_EQ(after_reset_out.decode_range->valid_fraction,
                  static_cast<float>(overlapping_tail.size()) /
                      static_cast<float>(kWindowSamples));
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
  EXPECT_EQ(result.features.size(), (kSamples / kHop) * kNMels);
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
  source.PushChunk(first);
  ASSERT_OK(processor->Schedule());
  ASSERT_OK_AND_ASSIGN(auto first_result, processor->GetOutput());
  ASSERT_TRUE(first_result.decode_range.has_value());
  EXPECT_EQ(first_result.decode_range->start_fraction, 0.0f);
  EXPECT_EQ(first_result.decode_range->valid_fraction, 1.0f);

  source.PushChunk(tail);
  ASSERT_OK(processor->Schedule());
  ASSERT_OK_AND_ASSIGN(auto shifted_result, processor->GetOutput());
  ASSERT_EQ(shifted_result.features.size(), kWindowFrames * kNMels);
  // The chunk was rebuilt as the last 5 s of the stream: 3 s of it were
  // already covered by the previous window, all of it is real audio.
  ASSERT_TRUE(shifted_result.decode_range.has_value());
  EXPECT_THAT(shifted_result.decode_range->start_fraction,
              FloatNear(static_cast<float>(kWindowSamples - kTailSamples) /
                            kWindowSamples,
                        1e-6f));
  EXPECT_EQ(shifted_result.decode_range->valid_fraction, 1.0f);

  // Same features as processing the stream's last window directly on a fresh
  // processor, i.e. no padding or stale STFT queue samples are involved.
  DummyAudioSource reference_source;
  ASSERT_OK_AND_ASSIGN(auto reference, LogMelSpectrogramProcessor::Create(
                                           16000, config, &reference_source));
  reference_source.PushChunk(
      std::vector<float>(stream.end() - kWindowSamples, stream.end()));
  ASSERT_OK(reference->Schedule());
  ASSERT_OK_AND_ASSIGN(auto reference_result, reference->GetOutput());
  ASSERT_TRUE(reference_result.decode_range.has_value());
  EXPECT_EQ(reference_result.decode_range->start_fraction, 0.0f);
  EXPECT_EQ(shifted_result.features, reference_result.features);

  // Reset forgets the previous window: a short chunk is padded again.
  processor->Reset();
  source.PushChunk(tail);
  ASSERT_OK(processor->Schedule());
  ASSERT_OK_AND_ASSIGN(auto padded_result, processor->GetOutput());
  ASSERT_TRUE(padded_result.decode_range.has_value());
  EXPECT_EQ(padded_result.decode_range->start_fraction, 0.0f);
  EXPECT_FLOAT_EQ(padded_result.decode_range->valid_fraction,
                  static_cast<float>(kTailSamples / kHop) / kWindowFrames);
  EXPECT_EQ(padded_result.features[0 * kWindowFrames + kWindowFrames - 1],
            0.0f);
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
  source.PushChunk(
      std::vector<float>(stream.begin(), stream.begin() + kWindowSamples));
  ASSERT_OK(processor->Schedule());
  ASSERT_OK(processor->GetOutput().status());
  source.PushChunk(
      std::vector<float>(stream.begin() + kWindowSamples, stream.end()));
  ASSERT_OK(processor->Schedule());
  ASSERT_OK_AND_ASSIGN(auto tail_result, processor->GetOutput());
  ASSERT_TRUE(tail_result.decode_range.has_value());
  EXPECT_EQ(tail_result.decode_range->start_fraction, 0.0f);
  EXPECT_FLOAT_EQ(tail_result.decode_range->valid_fraction,
                  static_cast<float>(16000 / kHop) / kWindowFrames);
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
  EXPECT_EQ(shifted_result.features, reference_result.features);
  ASSERT_TRUE(shifted_result.centered_features.has_value());
  EXPECT_FALSE(shifted_result.centered_features->empty());
}

TEST(LogMelSpectrogramProcessorTest,
     SkippedSilentOrRoomNoiseChunkClearsPreviousWindowForSubsequentShortChunk) {
  constexpr int kHop = 160;
  constexpr int kNMels = 80;
  constexpr int kWindowFrames = 500;
  constexpr int kWindowSamples = kWindowFrames * kHop;
  constexpr int kShortSamples = 200 * kHop;
  const LogMelSpectrogramProcessor::LogMelSpectrogramConfig config{
      .n_fft = 512,
      .n_mels = kNMels,
      .hop_length = kHop,
      .n_frames = kWindowFrames};

  DummyAudioSource source;
  ASSERT_OK_AND_ASSIGN(auto processor, LogMelSpectrogramProcessor::Create(
                                           16000, config, &source));

  // Full speech chunk followed by a full silent chunk, then a short speech
  // chunk. The skipped silent chunk breaks stream continuity, so the short
  // chunk must be zero-padded rather than shifted back over the pre-silence
  // window.
  source.PushChunk(MakeTestAudio(kWindowSamples));
  ASSERT_OK(processor->Schedule());
  ASSERT_OK(processor->GetOutput().status());

  source.PushChunk(std::vector<float>(kWindowSamples, 0.0f));
  ASSERT_OK(processor->Schedule());
  ASSERT_OK_AND_ASSIGN(auto silent_result, processor->GetOutput());
  EXPECT_TRUE(silent_result.features.empty());

  source.PushChunk(MakeTestAudio(kShortSamples));
  ASSERT_OK(processor->Schedule());
  ASSERT_OK_AND_ASSIGN(auto after_silence_result, processor->GetOutput());
  ASSERT_TRUE(after_silence_result.decode_range.has_value());
  EXPECT_EQ(after_silence_result.decode_range->start_fraction, 0.0f);
  EXPECT_FLOAT_EQ(after_silence_result.decode_range->valid_fraction,
                  200.0f / kWindowFrames);

  // Likewise, a full room-noise follow-up chunk skipped by
  // IsRoomNoiseFollowUpChunk must clear `previous_window_`.
  source.PushChunk(MakeTestAudio(kWindowSamples));
  ASSERT_OK(processor->Schedule());
  ASSERT_OK(processor->GetOutput().status());

  std::vector<float> room_noise = MakeTestAudio(kWindowSamples);
  for (float& s : room_noise) s *= 0.005f;
  source.PushChunk(room_noise);
  ASSERT_OK(processor->Schedule());
  ASSERT_OK_AND_ASSIGN(auto noise_result, processor->GetOutput());
  EXPECT_TRUE(noise_result.features.empty());

  source.PushChunk(MakeTestAudio(kShortSamples));
  ASSERT_OK(processor->Schedule());
  ASSERT_OK_AND_ASSIGN(auto after_noise_result, processor->GetOutput());
  ASSERT_TRUE(after_noise_result.decode_range.has_value());
  EXPECT_EQ(after_noise_result.decode_range->start_fraction, 0.0f);
  EXPECT_FLOAT_EQ(after_noise_result.decode_range->valid_fraction,
                  200.0f / kWindowFrames);
}

TEST(LogMelSpectrogramProcessorTest,
     ShortFinalChunkWithTrailingRoomNoiseTrimsSilenceAndShiftsBack) {
  constexpr int kHop = 160;
  constexpr int kNMels = 80;
  constexpr int kWindowFrames = 500;
  constexpr int kWindowSamples = kWindowFrames * kHop;
  constexpr int kActiveTailFrames = 100;  // 1.0 s speech.
  constexpr int kSilentTailFrames = 250;  // 2.5 s room noise (>= 2.0 s).
  constexpr int kTailFrames = kActiveTailFrames + kSilentTailFrames;
  const std::vector<float> stream =
      MakeTestAudio((kWindowFrames + kTailFrames) * kHop);
  const std::vector<float> first(stream.begin(),
                                 stream.begin() + kWindowSamples);
  std::vector<float> tail(stream.begin() + kWindowSamples, stream.end());
  for (int i = kActiveTailFrames * kHop; i < kTailFrames * kHop; ++i) {
    tail[i] *= 0.005f;
  }
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

  source.PushChunk(tail);
  ASSERT_OK(processor->Schedule());
  ASSERT_OK_AND_ASSIGN(auto result, processor->GetOutput());
  // 10 active 100 ms blocks + 1 margin block = 110 trimmed frames shifted back
  // over the 500-frame previous window -> start_fraction = (500 - 110) / 500.
  ASSERT_TRUE(result.decode_range.has_value());
  EXPECT_FLOAT_EQ(result.decode_range->start_fraction, 390.0f / kWindowFrames);
  EXPECT_FLOAT_EQ(result.decode_range->valid_fraction, 1.0f);
  EXPECT_NE(result.features[496], 0.0f);
  ASSERT_TRUE(result.centered_features.has_value());
  EXPECT_NE((*result.centered_features)[kWindowFrames - 1], 0.0f);
}

TEST(LogMelSpectrogramProcessorTest,
     ShiftedShortChunkDoesNotTrimReconstructedWindowInCenteredFeatures) {
  constexpr int kHop = 160;
  constexpr int kNMels = 80;
  constexpr int kWindowFrames = 500;
  constexpr int kWindowSamples = kWindowFrames * kHop;
  constexpr int kOverlapFrames = 200;
  constexpr int kNewFrames = 5;
  std::vector<float> stream =
      MakeTestAudio((kWindowFrames + kNewFrames) * kHop);
  // 195 frames of quiet audio at the tail of the first window (< 20 silent
  // blocks, so the first window is not trimmed) plus 5 new quiet frames in the
  // 205-frame overlapping final chunk (< 210 frames, so the final chunk is not
  // trimmed on entry, while its first block [300, 310) still contains active
  // speech). The reconstructed 500-frame window [5, 505) has 20 trailing silent
  // blocks.
  for (int i = 305 * kHop; i < (kWindowFrames + kNewFrames) * kHop; ++i) {
    stream[i] *= 0.005f;
  }
  const std::vector<float> first(stream.begin(),
                                 stream.begin() + kWindowSamples);
  const std::vector<float> tail(
      stream.begin() + (kWindowFrames - kOverlapFrames) * kHop, stream.end());
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

  source.PushChunk(tail);
  ASSERT_OK(processor->Schedule());
  ASSERT_OK_AND_ASSIGN(auto result, processor->GetOutput());
  ASSERT_TRUE(result.decode_range.has_value());
  EXPECT_FLOAT_EQ(
      result.decode_range->start_fraction,
      static_cast<float>(kWindowFrames - (kOverlapFrames + kNewFrames)) /
          kWindowFrames);
  EXPECT_FLOAT_EQ(result.decode_range->valid_fraction, 1.0f);
  ASSERT_TRUE(result.centered_features.has_value());
  ASSERT_EQ(result.centered_features->size(), kWindowFrames * kNMels);
  // Centered features must not run TrimTrailingRoomNoise on the reconstructed
  // window and zero out the trailing frames in [start_fraction, 1.0).
  EXPECT_NE((*result.centered_features)[kWindowFrames - 1], 0.0f);
}

TEST(LogMelSpectrogramProcessorTest,
     LowAmplitudeStandardNormAudioScalesAboveMelFloorBeforeNormalization) {
  constexpr int kHop = 160;
  constexpr int kNMels = 80;
  constexpr int kWindowFrames = 500;
  const LogMelSpectrogramProcessor::LogMelSpectrogramConfig config{
      .n_fft = 512,
      .n_mels = kNMels,
      .hop_length = kHop,
      .n_frames = kWindowFrames,
      .preemphasis = 0.97f,
      .mel_floor = 5.960464477539063e-08f,
      .normalize_mel = false,
      .add_floor_to_mel_before_log = true,
      .norm_type = LogMelSpectrogramProcessor::NormType::kStandard,
      .slaney_mel = true,
      .power_spectrum = true};

  std::vector<float> normal_audio = MakeTestAudio(16000 * 2);
  float max_abs = 0.0f;
  for (float s : normal_audio) {
    max_abs = std::max(max_abs, std::abs(s));
  }
  // Scale normal_audio to exact peak = 0.1f (>= kMinStandardNormPeak = 0.02f).
  for (float& s : normal_audio) {
    s *= 0.1f / max_abs;
  }
  std::vector<float> quiet_audio = normal_audio;
  // Scale down by 12.5x (peak = 0.008f < kMinStandardNormPeak = 0.02f, which
  // ScaleQuietStandardNormAudio scales back up to kTargetStandardNormPeak =
  // 0.1f, while peak >= kSilenceMaxPeak so audio is not dropped as silence).
  for (float& s : quiet_audio) {
    s *= 0.08f;
  }

  DummyAudioSource normal_source;
  ASSERT_OK_AND_ASSIGN(
      auto normal_processor,
      LogMelSpectrogramProcessor::Create(16000, config, &normal_source));
  normal_source.PushChunk(normal_audio);
  ASSERT_OK(normal_processor->Schedule());
  ASSERT_OK_AND_ASSIGN(auto normal_result, normal_processor->GetOutput());

  DummyAudioSource quiet_source;
  ASSERT_OK_AND_ASSIGN(auto quiet_processor, LogMelSpectrogramProcessor::Create(
                                                 16000, config, &quiet_source));
  quiet_source.PushChunk(quiet_audio);
  ASSERT_OK(quiet_processor->Schedule());
  ASSERT_OK_AND_ASSIGN(auto quiet_result, quiet_processor->GetOutput());

  ASSERT_EQ(quiet_result.features.size(), normal_result.features.size());
  for (size_t i = 0; i < quiet_result.features.size(); ++i) {
    EXPECT_THAT(quiet_result.features[i],
                FloatNear(normal_result.features[i], 1e-4f));
  }
}

TEST(LogMelSpectrogramProcessorTest,
     ShortUtteranceWithTrailingDigitalSilenceTrimsValidFrames) {
  constexpr int kHop = 160;
  constexpr int kNMels = 80;
  constexpr int kWindowFrames = 500;
  constexpr int kTotalFrames = 230;   // 2.3 s short utterance.
  constexpr int kActiveFrames = 200;  // 2.0 s speech + 0.3 s digital silence.
  std::vector<float> audio = MakeTestAudio(kTotalFrames * kHop);
  for (int i = kActiveFrames * kHop; i < kTotalFrames * kHop; ++i) {
    audio[i] *= 0.0002f;
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
  // 20 active 100 ms blocks + 1 margin block = 210 decode valid frames, while
  // mel features remain populated through kTotalFrames (230) so encoder
  // per-band normalization statistics are unchanged.
  ASSERT_TRUE(result.decode_range.has_value());
  EXPECT_FLOAT_EQ(result.decode_range->valid_fraction, 210.0f / kWindowFrames);
  EXPECT_NE(result.features[210], 0.0f);
  EXPECT_EQ(result.features[kTotalFrames], 0.0f);
}

TEST(LogMelSpectrogramProcessorTest,
     ShortWarmStartChunkDoesNotInflatePeakSpeechRmsForSubsequentFullWindow) {
  constexpr int kHop = 160;
  constexpr int kNMels = 80;
  constexpr int kWindowFrames = 500;
  constexpr int kWarmStartFrames = 125;
  // Warm-start chunk has loud speech (`peak = 1.0f`), followed by a full
  // 500-frame window whose speech is quieter (`0.004f` RMS < `0.05 * warm_rms`
  // and `< 6e-3f`). Because the warm-start chunk is shorter than
  // `window_samples_`, it must not update `peak_speech_rms_` and cause the
  // first full window to be dropped as room noise.
  std::vector<float> warm_start = MakeTestAudio(kWarmStartFrames * kHop);
  for (float& s : warm_start) {
    s *= 2.0f;
  }
  std::vector<float> full_window = MakeTestAudio(kWindowFrames * kHop);
  for (float& s : full_window) {
    s *= 0.008f;
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
  source.PushChunk(warm_start, /*is_warm_start=*/true);
  ASSERT_OK(processor->Schedule());
  ASSERT_OK_AND_ASSIGN(auto warm_result, processor->GetOutput());
  EXPECT_FALSE(warm_result.features.empty());
  EXPECT_TRUE(warm_result.is_warm_start);

  source.PushChunk(full_window, /*is_warm_start=*/false);
  ASSERT_OK(processor->Schedule());
  ASSERT_OK_AND_ASSIGN(auto full_result, processor->GetOutput());
  EXPECT_FALSE(full_result.features.empty());
  EXPECT_FALSE(full_result.is_warm_start);
}

std::vector<float> MakeStationaryRoomNoise(int num_samples, float rms) {
  std::vector<float> audio(num_samples);
  // Uniform noise in [-sqrt(3) * rms, +sqrt(3) * rms] has exact RMS = `rms`.
  const float half_range = std::sqrt(3.0f) * rms;
  uint32_t state = 67890u;
  for (int i = 0; i < num_samples; ++i) {
    state = state * 1664525u + 1013904223u;
    const float u = static_cast<float>(state >> 8) / 8388608.0f - 1.0f;
    audio[i] = half_range * u;
  }
  return audio;
}

TEST(LogMelSpectrogramProcessorTest,
     StationaryRoomNoiseChunksAreSkippedBeforeAndAfterSpeech) {
  constexpr int kHop = 160;
  constexpr int kNMels = 80;
  constexpr int kWindowFrames = 500;
  const LogMelSpectrogramProcessor::LogMelSpectrogramConfig config{
      .n_fft = 512,
      .n_mels = kNMels,
      .hop_length = kHop,
      .n_frames = kWindowFrames};

  DummyAudioSource source;
  ASSERT_OK_AND_ASSIGN(auto processor, LogMelSpectrogramProcessor::Create(
                                           16000, config, &source));

  // Initial stationary ambient noise (`rms = 0.0006f` and `rms = 0.010f`)
  // before any speech (`peak_speech_rms_ == 0.0f`) is gated as room noise on
  // both warm-start and full windows.
  source.PushChunk(MakeStationaryRoomNoise(250 * kHop, 0.0006f),
                   /*is_warm_start=*/true);
  ASSERT_OK(processor->Schedule());
  ASSERT_OK_AND_ASSIGN(auto quiet_warm, processor->GetOutput());
  EXPECT_TRUE(quiet_warm.features.empty());
  EXPECT_TRUE(quiet_warm.is_warm_start);

  source.PushChunk(MakeStationaryRoomNoise(kWindowFrames * kHop, 0.010f),
                   /*is_warm_start=*/false);
  ASSERT_OK(processor->Schedule());
  ASSERT_OK_AND_ASSIGN(auto stationary_full, processor->GetOutput());
  EXPECT_TRUE(stationary_full.features.empty());

  // Speech followed by trailing stationary mic noise (`rms = 0.010f` > `6e-3f`)
  // trims the trailing noise on both a warm-start chunk (`>= 10` silent blocks)
  // and a full 5-second window (`>= 20` silent blocks).
  std::vector<float> warm_speech_plus_noise =
      MakeStationaryRoomNoise(250 * kHop, 0.010f);
  const std::vector<float> speech_1s = MakeTestAudio(100 * kHop);
  std::copy(speech_1s.begin(), speech_1s.end(), warm_speech_plus_noise.begin());
  source.PushChunk(warm_speech_plus_noise, /*is_warm_start=*/true);
  ASSERT_OK(processor->Schedule());
  ASSERT_OK_AND_ASSIGN(auto warm_trimmed, processor->GetOutput());
  EXPECT_FALSE(warm_trimmed.features.empty());
  ASSERT_TRUE(warm_trimmed.decode_range.has_value());
  EXPECT_FLOAT_EQ(warm_trimmed.decode_range->valid_fraction,
                  110.0f / kWindowFrames);

  std::vector<float> full_speech_plus_noise =
      MakeStationaryRoomNoise(kWindowFrames * kHop, 0.010f);
  std::copy(speech_1s.begin(), speech_1s.end(), full_speech_plus_noise.begin());
  source.PushChunk(full_speech_plus_noise, /*is_warm_start=*/false);
  ASSERT_OK(processor->Schedule());
  ASSERT_OK_AND_ASSIGN(auto full_trimmed, processor->GetOutput());
  EXPECT_FALSE(full_trimmed.features.empty());
  ASSERT_TRUE(full_trimmed.decode_range.has_value());
  EXPECT_FLOAT_EQ(full_trimmed.decode_range->valid_fraction,
                  110.0f / kWindowFrames);

  // When a full window ends with >= 7 blocks (0.7s+) of trailing stationary
  // noise (e.g. 1.2s of silence after 3.8s of speech), the immediately
  // following sliding window whose trailing silence grew by >= 8 blocks to
  // >= 20 blocks (2.45s of silence after 2.55s of already-decoded speech) is
  // gated as a silent follow-up chunk so the stream finalizes without redundant
  // encoder passes.
  std::vector<float> win_with_12_silent_blocks =
      MakeStationaryRoomNoise(kWindowFrames * kHop, 0.010f);
  const std::vector<float> speech_380 = MakeTestAudio(380 * kHop);
  std::copy(speech_380.begin(), speech_380.end(),
            win_with_12_silent_blocks.begin());
  source.PushChunk(win_with_12_silent_blocks, /*is_warm_start=*/false);
  ASSERT_OK(processor->Schedule());
  ASSERT_OK_AND_ASSIGN(auto first_pause_win, processor->GetOutput());
  EXPECT_FALSE(first_pause_win.features.empty());

  std::vector<float> win_with_25_silent_blocks =
      MakeStationaryRoomNoise(kWindowFrames * kHop, 0.010f);
  const std::vector<float> speech_250 = MakeTestAudio(250 * kHop);
  std::copy(speech_250.begin(), speech_250.end(),
            win_with_25_silent_blocks.begin());
  source.PushChunk(win_with_25_silent_blocks, /*is_warm_start=*/false);
  ASSERT_OK(processor->Schedule());
  ASSERT_OK_AND_ASSIGN(auto second_pause_win, processor->GetOutput());
  EXPECT_TRUE(second_pause_win.features.empty());

  // Follow-up window with only 1 block of weak residual tail (`0.03f` RMS <=
  // `0.25 * peak_speech_rms_`) followed by 49 blocks of `0.010f` stationary
  // room noise is skipped.
  std::vector<float> residual_plus_noise =
      MakeStationaryRoomNoise(kWindowFrames * kHop, 0.010f);
  for (int i = 0; i < 10 * kHop; ++i) {
    residual_plus_noise[i] = speech_1s[i] * 0.1f;
  }
  source.PushChunk(residual_plus_noise, /*is_warm_start=*/false);
  ASSERT_OK(processor->Schedule());
  ASSERT_OK_AND_ASSIGN(auto residual_out, processor->GetOutput());
  EXPECT_TRUE(residual_out.features.empty());
}

TEST(LogMelSpectrogramProcessorTest,
     DriftingOrTransientRoomNoiseIsSkippedWhileUltraQuietSpeechIsKept) {
  constexpr int kHop = 160;
  constexpr int kNMels = 80;
  constexpr int kWindowFrames = 500;
  const LogMelSpectrogramProcessor::LogMelSpectrogramConfig config{
      .n_fft = 512,
      .n_mels = kNMels,
      .hop_length = kHop,
      .n_frames = kWindowFrames};

  DummyAudioSource source;
  ASSERT_OK_AND_ASSIGN(auto processor, LogMelSpectrogramProcessor::Create(
                                           16000, config, &source));

  // 1. Slowly drifting low-level ambient room noise (`~0.0012` to `0.0028` RMS
  // with no deep inter-syllable silence) is gated as stationary noise.
  std::vector<float> drifting_noise =
      MakeStationaryRoomNoise(kWindowFrames * kHop, 0.0012f);
  for (int b = 0; b < 50; ++b) {
    const float scale = 1.0f + 1.2f * (static_cast<float>(b) / 49.0f);
    for (int i = b * 10 * kHop; i < (b + 1) * 10 * kHop; ++i) {
      drifting_noise[i] *= scale;
    }
  }
  source.PushChunk(drifting_noise, /*is_warm_start=*/false);
  ASSERT_OK(processor->Schedule());
  ASSERT_OK_AND_ASSIGN(auto drifting_out, processor->GetOutput());
  EXPECT_TRUE(drifting_out.features.empty());

  // 2. Stationary low-level room noise (`0.0018` RMS) with an isolated
  // single-block transient tap (`0.0045` RMS) is also gated.
  std::vector<float> transient_noise =
      MakeStationaryRoomNoise(kWindowFrames * kHop, 0.0018f);
  for (int i = 20 * 10 * kHop; i < 21 * kHop * 10; ++i) {
    transient_noise[i] *= 2.5f;
  }
  source.PushChunk(transient_noise, /*is_warm_start=*/false);
  ASSERT_OK(processor->Schedule());
  ASSERT_OK_AND_ASSIGN(auto transient_out, processor->GetOutput());
  EXPECT_TRUE(transient_out.features.empty());

  // 3. An ultra-quiet room floor (`0.0001` RMS) with an isolated single-block
  // tap (`0.0004` RMS), 2 adjacent tap blocks, or 3 non-adjacent taps is gated
  // rather than mistaken for ultra-quiet speech crest.
  std::vector<float> ultra_quiet_one_tap =
      MakeStationaryRoomNoise(kWindowFrames * kHop, 0.0001f);
  for (int i = 20 * 10 * kHop; i < 21 * 10 * kHop; ++i) {
    ultra_quiet_one_tap[i] *= 4.0f;
  }
  source.PushChunk(ultra_quiet_one_tap, /*is_warm_start=*/false);
  ASSERT_OK(processor->Schedule());
  ASSERT_OK_AND_ASSIGN(auto uq_one_tap_out, processor->GetOutput());
  EXPECT_TRUE(uq_one_tap_out.features.empty());

  std::vector<float> ultra_quiet_two_adj =
      MakeStationaryRoomNoise(kWindowFrames * kHop, 0.0001f);
  for (int i = 20 * 10 * kHop; i < 22 * 10 * kHop; ++i) {
    ultra_quiet_two_adj[i] *= 4.0f;
  }
  source.PushChunk(ultra_quiet_two_adj, /*is_warm_start=*/false);
  ASSERT_OK(processor->Schedule());
  ASSERT_OK_AND_ASSIGN(auto uq_two_adj_out, processor->GetOutput());
  EXPECT_TRUE(uq_two_adj_out.features.empty());

  std::vector<float> ultra_quiet_three_isolated =
      MakeStationaryRoomNoise(kWindowFrames * kHop, 0.0001f);
  for (int b : {10, 25, 40}) {
    for (int i = b * 10 * kHop; i < (b + 1) * 10 * kHop; ++i) {
      ultra_quiet_three_isolated[i] *= 4.0f;
    }
  }
  source.PushChunk(ultra_quiet_three_isolated, /*is_warm_start=*/false);
  ASSERT_OK(processor->Schedule());
  ASSERT_OK_AND_ASSIGN(auto uq_three_iso_out, processor->GetOutput());
  EXPECT_TRUE(uq_three_iso_out.features.empty());

  // 4. Digital silence (1 s, so `p10_rms == 0`) followed by a two-stage
  // ultra-quiet noise floor (`0.0002` RMS for 3 s then `0.0005` RMS for 1 s,
  // e.g. mute-to-unmute) is gated by `p10_rms < kUltraQuietMinFloorRms`.
  std::vector<float> mute_to_unmute_noise =
      MakeStationaryRoomNoise(kWindowFrames * kHop, 0.0002f);
  std::fill(mute_to_unmute_noise.begin(),
            mute_to_unmute_noise.begin() + 100 * kHop, 0.0f);
  for (int i = 400 * kHop; i < kWindowFrames * kHop; ++i) {
    mute_to_unmute_noise[i] *= 2.5f;
  }
  source.PushChunk(mute_to_unmute_noise, /*is_warm_start=*/false);
  ASSERT_OK(processor->Schedule());
  ASSERT_OK_AND_ASSIGN(auto mute_unmute_out, processor->GetOutput());
  EXPECT_TRUE(mute_unmute_out.features.empty());

  // 5. A 400 ms (4 contiguous blocks) quiet word (`~0.0029` RMS) on a `0.0012`
  // RMS background floor is not mistaken for an isolated transient click/tap or
  // slow drift.
  std::vector<float> short_quiet_word =
      MakeStationaryRoomNoise(kWindowFrames * kHop, 0.0012f);
  for (int i = 20 * 10 * kHop; i < 24 * 10 * kHop; ++i) {
    short_quiet_word[i] *= 2.4f;
  }
  source.PushChunk(short_quiet_word, /*is_warm_start=*/false);
  ASSERT_OK(processor->Schedule());
  ASSERT_OK_AND_ASSIGN(auto short_word_out, processor->GetOutput());
  EXPECT_FALSE(short_word_out.features.empty());

  // 6. Multi-second low-SNR continuous quiet speech (35 syllabic speech blocks
  // modulating between `0.0020` and `0.0034` RMS over 15 background floor
  // blocks at `0.0015` RMS) is preserved rather than dropped as drift noise.
  DummyAudioSource low_snr_source;
  ASSERT_OK_AND_ASSIGN(
      auto low_snr_processor,
      LogMelSpectrogramProcessor::Create(16000, config, &low_snr_source));
  std::vector<float> low_snr_speech =
      MakeStationaryRoomNoise(kWindowFrames * kHop, 0.0015f);
  for (int b = 15; b < 50; ++b) {
    const float target_rms = (b % 2 == 0) ? 0.0033f : 0.0021f;
    const float scale = target_rms / 0.0015f;
    for (int i = b * 10 * kHop; i < (b + 1) * 10 * kHop; ++i) {
      low_snr_speech[i] *= scale;
    }
  }
  low_snr_source.PushChunk(low_snr_speech, /*is_warm_start=*/false);
  ASSERT_OK(low_snr_processor->Schedule());
  ASSERT_OK_AND_ASSIGN(auto low_snr_out, low_snr_processor->GetOutput());
  EXPECT_FALSE(low_snr_out.features.empty());

  // Smoothly varying chunks that exceed `kDriftMaxToMinRatio * min_rms` (e.g.
  // a silent onset before sustained voice) or `kMaxDriftNoiseRms` are also
  // preserved.
  DummyAudioSource onset_source;
  ASSERT_OK_AND_ASSIGN(
      auto onset_processor,
      LogMelSpectrogramProcessor::Create(16000, config, &onset_source));
  std::vector<float> speech_after_silence =
      MakeStationaryRoomNoise(kWindowFrames * kHop, 0.0022f);
  for (int i = 0; i < 10 * 10 * kHop; ++i) {
    speech_after_silence[i] *= (0.0005f / 0.0022f);
  }
  onset_source.PushChunk(speech_after_silence, /*is_warm_start=*/false);
  ASSERT_OK(onset_processor->Schedule());
  ASSERT_OK_AND_ASSIGN(auto onset_out, onset_processor->GetOutput());
  EXPECT_FALSE(onset_out.features.empty());

  DummyAudioSource louder_source;
  ASSERT_OK_AND_ASSIGN(
      auto louder_processor,
      LogMelSpectrogramProcessor::Create(16000, config, &louder_source));
  std::vector<float> louder_drifting =
      MakeStationaryRoomNoise(kWindowFrames * kHop, 0.0025f);
  for (int b = 0; b < 50; ++b) {
    const float scale = 1.0f + 1.2f * (static_cast<float>(b) / 49.0f);
    for (int i = b * 10 * kHop; i < (b + 1) * 10 * kHop; ++i) {
      louder_drifting[i] *= scale;
    }
  }
  louder_source.PushChunk(louder_drifting, /*is_warm_start=*/false);
  ASSERT_OK(louder_processor->Schedule());
  ASSERT_OK_AND_ASSIGN(auto louder_out, louder_processor->GetOutput());
  EXPECT_FALSE(louder_out.features.empty());

  // 7. Ultra-quiet speech (`max_rms <= 8e-4f`) over a continuous floor that
  // exhibits a multi-block speech crest (`max_rms > 3.0 * p10_rms` and
  // `max_rms > 1.6 * median_rms`) is preserved rather than gated.
  DummyAudioSource quiet_source;
  ASSERT_OK_AND_ASSIGN(auto quiet_processor, LogMelSpectrogramProcessor::Create(
                                                 16000, config, &quiet_source));
  std::vector<float> ultra_quiet_speech =
      MakeStationaryRoomNoise(kWindowFrames * kHop, 0.0001f);
  const std::vector<float> speech_2s = MakeTestAudio(200 * kHop);
  for (size_t i = 0; i < speech_2s.size(); ++i) {
    ultra_quiet_speech[i] = speech_2s[i] * 0.0018f;
  }
  quiet_source.PushChunk(ultra_quiet_speech, /*is_warm_start=*/false);
  ASSERT_OK(quiet_processor->Schedule());
  ASSERT_OK_AND_ASSIGN(auto quiet_speech_out, quiet_processor->GetOutput());
  EXPECT_FALSE(quiet_speech_out.features.empty());
}

}  // namespace
}  // namespace litert::omni::asr
