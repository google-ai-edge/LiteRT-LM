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

#include "omni/asr/file_audio_source.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "support/util/test_utils.h"  // IWYU pragma: keep for ASSERT_OK
#include "miniaudio.h"  // from @miniaudio

namespace litert::omni::asr {
namespace {

using ::absl_testing::StatusIs;
using ::testing::Eq;
using ::testing::SizeIs;

void WriteWavFile(const std::string& path,
                  const std::vector<int16_t>& pcm_samples,
                  int sample_rate = 16000) {
  ma_encoder_config config = ma_encoder_config_init(
      ma_encoding_format_wav, ma_format_s16, 1, sample_rate);
  ma_encoder encoder;
  ASSERT_EQ(ma_encoder_init_file(path.c_str(), &config, &encoder), MA_SUCCESS);
  ma_uint64 frames_written = 0;
  ASSERT_EQ(ma_encoder_write_pcm_frames(&encoder, pcm_samples.data(),
                                        pcm_samples.size(), &frames_written),
            MA_SUCCESS);
  EXPECT_EQ(frames_written, pcm_samples.size());
  ma_encoder_uninit(&encoder);
}

class FileAudioSourceTest : public ::testing::Test {
 protected:
  void SetUp() override { temp_dir_ = testing::TempDir(); }

  std::string GetTempWavPath(const std::string& filename) {
    return absl::StrCat(temp_dir_, "/", filename);
  }

  std::string temp_dir_;
};

TEST_F(FileAudioSourceTest, Create_CorrectChunks) {
  int sample_rate = 16000;
  std::vector<int16_t> samples(1600);
  for (int i = 0; i < 1600; ++i) {
    samples[i] = static_cast<int16_t>(i * 10);
  }
  std::string path = GetTempWavPath("test_correct_chunks.wav");
  WriteWavFile(path, samples, sample_rate);

  ASSERT_OK_AND_ASSIGN(auto source,
                       FileAudioSource::Create(
                           path, /*interval=*/absl::Milliseconds(100),
                           /*overlap=*/absl::ZeroDuration(), sample_rate, 1));
  EXPECT_TRUE(source->NeedSchedule());
  ASSERT_OK(source->Schedule());
  EXPECT_FALSE(source->NeedSchedule());
  EXPECT_THAT(source->Schedule(), StatusIs(absl::StatusCode::kOutOfRange));

  ASSERT_OK_AND_ASSIGN(auto chunk, source->GetOutput());
  EXPECT_THAT(chunk, SizeIs(1600));
  EXPECT_NEAR(chunk[0], 0.0f / 32768.0f, 1e-4f);
  EXPECT_NEAR(chunk[1599], 15990.0f / 32768.0f, 1e-4f);
}

TEST_F(FileAudioSourceTest, Create_HandlesOverlap) {
  int sample_rate = 16000;
  std::vector<int16_t> samples(3200);
  for (int i = 0; i < 3200; ++i) {
    samples[i] = static_cast<int16_t>(i);
  }
  std::string path = GetTempWavPath("test_overlap.wav");
  WriteWavFile(path, samples, sample_rate);

  ASSERT_OK_AND_ASSIGN(auto source,
                       FileAudioSource::Create(
                           path, /*interval=*/absl::Milliseconds(100),
                           /*overlap=*/absl::Milliseconds(50), sample_rate, 1));
  std::vector<std::vector<float>> chunks;
  while (source->NeedSchedule()) {
    absl::Status status = source->Schedule();
    if (absl::IsOutOfRange(status)) {
      break;
    }
    ASSERT_OK(status);
    ASSERT_OK_AND_ASSIGN(auto chunk, source->GetOutput());
    chunks.push_back(std::move(chunk));
  }

  EXPECT_THAT(chunks, SizeIs(3));

  EXPECT_THAT(chunks[0], SizeIs(1600));
  EXPECT_NEAR(chunks[0][0], 0.0f / 32768.0f, 1e-4f);
  EXPECT_NEAR(chunks[0][1599], 1599.0f / 32768.0f, 1e-4f);

  EXPECT_THAT(chunks[1], SizeIs(1600));
  EXPECT_NEAR(chunks[1][0], 800.0f / 32768.0f, 1e-4f);
  EXPECT_NEAR(chunks[1][1599], 2399.0f / 32768.0f, 1e-4f);

  EXPECT_THAT(chunks[2], SizeIs(1600));
  EXPECT_NEAR(chunks[2][0], 1600.0f / 32768.0f, 1e-4f);
  EXPECT_NEAR(chunks[2][1599], 3199.0f / 32768.0f, 1e-4f);
}

TEST_F(FileAudioSourceTest, Create_HandlesNonDivisibleOverlap) {
  int sample_rate = 16000;
  // 4000 samples total (samples 0..3999)
  std::vector<int16_t> samples(4000);
  for (int i = 0; i < 4000; ++i) {
    samples[i] = static_cast<int16_t>(i);
  }
  std::string path = GetTempWavPath("test_non_divisible_overlap.wav");
  WriteWavFile(path, samples, sample_rate);

  // interval = 100ms (1600 samples), overlap = 40ms (640 samples), step = 960
  // samples
  ASSERT_OK_AND_ASSIGN(auto source,
                       FileAudioSource::Create(
                           path, /*interval=*/absl::Milliseconds(100),
                           /*overlap=*/absl::Milliseconds(40), sample_rate, 1));
  std::vector<std::vector<float>> chunks;
  while (source->NeedSchedule()) {
    absl::Status status = source->Schedule();
    if (absl::IsOutOfRange(status)) {
      break;
    }
    ASSERT_OK(status);
    ASSERT_OK_AND_ASSIGN(auto chunk, source->GetOutput());
    chunks.push_back(std::move(chunk));
  }

  // Chunk 0: samples 0..1599
  // Chunk 1: samples 960..2559
  // Chunk 2: samples 1920..3519
  ASSERT_GE(chunks.size(), 3);

  EXPECT_THAT(chunks[0], SizeIs(1600));
  EXPECT_NEAR(chunks[0][0], 0.0f / 32768.0f, 1e-4f);
  EXPECT_NEAR(chunks[0][1599], 1599.0f / 32768.0f, 1e-4f);

  EXPECT_THAT(chunks[1], SizeIs(1600));
  EXPECT_NEAR(chunks[1][0], 960.0f / 32768.0f, 1e-4f);
  EXPECT_NEAR(chunks[1][1599], 2559.0f / 32768.0f, 1e-4f);

  EXPECT_THAT(chunks[2], SizeIs(1600));
  EXPECT_NEAR(chunks[2][0], 1920.0f / 32768.0f, 1e-4f);
  EXPECT_NEAR(chunks[2][1599], 3519.0f / 32768.0f, 1e-4f);
}

TEST_F(FileAudioSourceTest, Create_ShortAudio_ReturnsWhatItCan) {
  int sample_rate = 16000;
  std::vector<int16_t> samples(800, 1000);
  std::string path = GetTempWavPath("test_short.wav");
  WriteWavFile(path, samples, sample_rate);

  ASSERT_OK_AND_ASSIGN(auto source,
                       FileAudioSource::Create(
                           path, /*interval=*/absl::Milliseconds(100),
                           /*overlap=*/absl::ZeroDuration(), sample_rate, 1));
  EXPECT_TRUE(source->NeedSchedule());
  ASSERT_OK(source->Schedule());
  EXPECT_FALSE(source->NeedSchedule());

  ASSERT_OK_AND_ASSIGN(auto chunk, source->GetOutput());
  EXPECT_THAT(chunk, SizeIs(1600));
  EXPECT_NEAR(chunk[0], 1000.0f / 32768.0f, 1e-4f);
  EXPECT_NEAR(chunk[799], 1000.0f / 32768.0f, 1e-4f);
  EXPECT_THAT(chunk[800], Eq(0.0f));
  EXPECT_THAT(chunk[1599], Eq(0.0f));
}

TEST_F(FileAudioSourceTest, Create_InvalidIntervalOverlap) {
  EXPECT_THAT(FileAudioSource::Create("non_existent.wav", absl::Seconds(1),
                                      absl::Seconds(1), 16000, 1),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(FileAudioSourceTest, Create_NonExistentFile_ReturnsNotFound) {
  EXPECT_THAT(FileAudioSource::Create("non_existent_file_12345.wav",
                                      absl::Milliseconds(100),
                                      absl::ZeroDuration(), 16000, 1),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST_F(FileAudioSourceTest, Reset_ClearsStateAndRestarts) {
  int sample_rate = 16000;
  std::vector<int16_t> samples(1600, 1000);
  std::string path = GetTempWavPath("test_reset.wav");
  WriteWavFile(path, samples, sample_rate);

  ASSERT_OK_AND_ASSIGN(auto source,
                       FileAudioSource::Create(
                           path, /*interval=*/absl::Milliseconds(100),
                           /*overlap=*/absl::ZeroDuration(), sample_rate, 1));
  ASSERT_OK(source->Schedule());
  EXPECT_FALSE(source->NeedSchedule());

  source->Reset();
  EXPECT_TRUE(source->NeedSchedule());
  EXPECT_FALSE(source->HasOutput());

  ASSERT_OK(source->Schedule());
  ASSERT_OK_AND_ASSIGN(auto chunk, source->GetOutput());
  EXPECT_THAT(chunk, SizeIs(1600));
  EXPECT_NEAR(chunk[0], 1000.0f / 32768.0f, 1e-4f);
}

}  // namespace
}  // namespace litert::omni::asr
