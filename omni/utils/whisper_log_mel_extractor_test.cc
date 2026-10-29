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

#include "omni/utils/whisper_log_mel_extractor.h"

#include <cmath>
#include <cstddef>
#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "support/util/test_utils.h"  // IWYU pragma: keep for ASSERT_OK

namespace litert::omni {
namespace {

using ::absl_testing::StatusIs;

TEST(WhisperLogMelExtractorTest, RejectsInvalidConfig) {
  WhisperLogMelExtractor::Config bad_sample_rate{.sample_rate_hz = 0};
  WhisperLogMelExtractor::Config bad_n_fft{.n_fft = 0};
  WhisperLogMelExtractor::Config bad_hop_length{.hop_length = 0};
  WhisperLogMelExtractor::Config bad_n_mels{.n_mels = 0};
  WhisperLogMelExtractor::Config bad_n_frames{.n_frames = 0};
  WhisperLogMelExtractor::Config bad_mel_floor{.mel_floor = 0.0f};

  EXPECT_THAT(WhisperLogMelExtractor::Create(bad_sample_rate),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(WhisperLogMelExtractor::Create(bad_n_fft),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(WhisperLogMelExtractor::Create(bad_hop_length),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(WhisperLogMelExtractor::Create(bad_n_mels),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(WhisperLogMelExtractor::Create(bad_n_frames),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(WhisperLogMelExtractor::Create(bad_mel_floor),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(WhisperLogMelExtractorTest, DefaultConfigCreatesValidExtractor) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<WhisperLogMelExtractor> extractor,
                       WhisperLogMelExtractor::Create());

  EXPECT_EQ(extractor->GetConfig().sample_rate_hz, 16000);
  EXPECT_EQ(extractor->GetConfig().n_fft, 400);
  EXPECT_EQ(extractor->GetConfig().hop_length, 160);
  EXPECT_EQ(extractor->GetConfig().n_mels, 80);
  EXPECT_EQ(extractor->GetConfig().n_frames, 3000);
  EXPECT_FLOAT_EQ(extractor->GetConfig().mel_floor, 1e-10f);
  EXPECT_FALSE(extractor->GetConfig().transpose);
}

TEST(WhisperLogMelExtractorTest, SupportsSmallNFramesAndReset) {
  WhisperLogMelExtractor::Config config{
      .sample_rate_hz = 16000,
      .n_fft = 400,
      .hop_length = 160,
      .n_mels = 80,
      .n_frames = 2,
      .mel_floor = 1e-10f,
      .transpose = false,
  };
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<WhisperLogMelExtractor> extractor,
                       WhisperLogMelExtractor::Create(config));
  std::vector<float> pcm(320, 0.0f);
  for (size_t i = 0; i < pcm.size(); ++i) {
    pcm[i] = 0.5f * std::sin(2.0f * static_cast<float>(M_PI) * 440.0f *
                             static_cast<float>(i) / 16000.0f);
  }

  ASSERT_OK_AND_ASSIGN(std::vector<float> mel_first, extractor->Extract(pcm));
  extractor->Reset();
  ASSERT_OK_AND_ASSIGN(std::vector<float> mel_second, extractor->Extract(pcm));

  ASSERT_EQ(mel_first.size(), static_cast<size_t>(2 * 80));
  EXPECT_EQ(mel_first, mel_second);
  for (size_t i = 0; i < mel_first.size(); ++i) {
    EXPECT_TRUE(std::isfinite(mel_first[i])) << "at index " << i;
  }
}

TEST(WhisperLogMelExtractorTest, HandlesEmptyAndLongInputs) {
  WhisperLogMelExtractor::Config config{
      .sample_rate_hz = 16000,
      .n_fft = 400,
      .hop_length = 160,
      .n_mels = 80,
      .n_frames = 10,
      .mel_floor = 1e-10f,
      .transpose = true,
  };
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<WhisperLogMelExtractor> extractor,
                       WhisperLogMelExtractor::Create(config));
  std::vector<float> long_pcm(3200, 0.25f);

  ASSERT_OK_AND_ASSIGN(std::vector<float> empty_mel, extractor->Extract({}));
  ASSERT_OK_AND_ASSIGN(std::vector<float> long_mel,
                       extractor->Extract(long_pcm));

  EXPECT_EQ(empty_mel.size(), static_cast<size_t>(10 * 80));
  EXPECT_EQ(long_mel.size(), static_cast<size_t>(10 * 80));
  EXPECT_NE(empty_mel, long_mel);
}

}  // namespace
}  // namespace litert::omni
