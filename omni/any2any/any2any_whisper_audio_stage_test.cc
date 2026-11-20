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

#include "omni/any2any/any2any_whisper_audio_stage.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/cc/litert_tensor_buffer_types.h"  // from @litert
#include "omni/base/fan_out_stage.h"
#include "omni/base/io_types.h"
#include "omni/base/mock_litert_runner.h"
#include "omni/omni_session.h"
#include "support/util/convert_tensor_buffer.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace litert::omni::any2any {
namespace {

using ::absl_testing::StatusIs;
using ::testing::_;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::NiceMock;

std::unique_ptr<NiceMock<MockLiteRtRunner>> MakeMockRunner(
    int hidden_dim, int num_mel_bins = 80, int chunk_frames = 10,
    int out_tokens = 4,
    std::vector<std::vector<float>>* captured_mels = nullptr) {
  auto runner = std::make_unique<NiceMock<MockLiteRtRunner>>();
  ON_CALL(*runner, CreateInputBuffers(absl::string_view("")))
      .WillByDefault([num_mel_bins, chunk_frames](absl::string_view)
                         -> absl::StatusOr<std::vector<TensorBuffer>> {
        LITERT_ASSIGN_OR_RETURN(auto buf, support::CreateTensorBuffer<float>(
                                              {1, num_mel_bins, chunk_frames}));
        std::vector<TensorBuffer> bufs;
        bufs.push_back(std::move(buf));
        return bufs;
      });
  ON_CALL(*runner, CreateOutputBuffers(absl::string_view("")))
      .WillByDefault([out_tokens, hidden_dim](absl::string_view)
                         -> absl::StatusOr<std::vector<TensorBuffer>> {
        LITERT_ASSIGN_OR_RETURN(auto buf, support::CreateTensorBuffer<float>(
                                              {1, out_tokens, hidden_dim}));
        std::vector<TensorBuffer> bufs;
        bufs.push_back(std::move(buf));
        return bufs;
      });
  ON_CALL(*runner, Run(absl::string_view(""), _, _))
      .WillByDefault(
          [out_tokens, hidden_dim, captured_mels](
              absl::string_view, absl::Span<const TensorBuffer> inputs,
              absl::Span<const TensorBuffer> outputs) -> absl::Status {
            if (captured_mels != nullptr) {
              LITERT_ASSIGN_OR_RETURN(
                  std::vector<float> mel,
                  support::CopyFromTensorBuffer<float>(inputs[0]));
              captured_mels->push_back(std::move(mel));
            }
            std::vector<float> out(static_cast<size_t>(out_tokens) *
                                   hidden_dim);
            for (int t = 0; t < out_tokens; ++t) {
              for (int d = 0; d < hidden_dim; ++d) {
                out[t * hidden_dim + d] = static_cast<float>(t * 10 + d + 1);
              }
            }
            auto& out_buf = const_cast<TensorBuffer&>(outputs[0]);
            LITERT_RETURN_IF_ERROR(
                out_buf.Write<float>(absl::MakeConstSpan(out)));
            return absl::OkStatus();
          });
  return runner;
}

// Non-constant waveform whose length (not a multiple of 3) and strong trailing
// DC step leave non-trivial fractional phase and low-pass filter history in a
// 24 kHz -> 16 kHz resampler unless `AudioResampler::Reset()` is invoked.
std::vector<float> MakeNonConstantAudioClip(size_t num_samples) {
  std::vector<float> pcm(num_samples);
  for (size_t i = 0; i < num_samples; ++i) {
    const float wave = 0.4f * std::sin(0.07f * static_cast<float>(i));
    const float tail_step = (i + 200 >= num_samples) ? 0.5f : 0.0f;
    pcm[i] = wave + tail_step;
  }
  return pcm;
}

TEST(Any2AnyWhisperAudioStageTest, CreateValidatesOptionsAndBuffers) {
  SequencedVariantQueue<OmniSession::Input, AudioInput> audio_queue;
  SequencedVariantQueue<OmniSession::Input, AudioInputMetadata> meta_queue;

  Any2AnyWhisperAudioStage::Options opts{
      .llm_hidden_size = 4,
      .initial_input_sample_rate_hz = 16000,
      .target_sample_rate_hz = 16000,
      .num_mel_bins = 80,
  };

  auto bad_hidden = opts;
  bad_hidden.llm_hidden_size = 0;
  EXPECT_THAT(Any2AnyWhisperAudioStage::Create(&audio_queue, &meta_queue,
                                               MakeMockRunner(4), bad_hidden),
              StatusIs(absl::StatusCode::kInvalidArgument));

  auto bad_init_sr = opts;
  bad_init_sr.initial_input_sample_rate_hz = 0;
  EXPECT_THAT(Any2AnyWhisperAudioStage::Create(&audio_queue, &meta_queue,
                                               MakeMockRunner(4), bad_init_sr),
              StatusIs(absl::StatusCode::kInvalidArgument));

  auto bad_target_sr = opts;
  bad_target_sr.target_sample_rate_hz = 0;
  EXPECT_THAT(Any2AnyWhisperAudioStage::Create(
                  &audio_queue, &meta_queue, MakeMockRunner(4), bad_target_sr),
              StatusIs(absl::StatusCode::kInvalidArgument));

  auto bad_mels = opts;
  bad_mels.num_mel_bins = 0;
  EXPECT_THAT(Any2AnyWhisperAudioStage::Create(&audio_queue, &meta_queue,
                                               MakeMockRunner(4), bad_mels),
              StatusIs(absl::StatusCode::kInvalidArgument));

  auto empty_runner = std::make_unique<NiceMock<MockLiteRtRunner>>();
  ON_CALL(*empty_runner, CreateInputBuffers(absl::string_view("")))
      .WillByDefault(
          [](absl::string_view) { return std::vector<TensorBuffer>{}; });
  ON_CALL(*empty_runner, CreateOutputBuffers(absl::string_view("")))
      .WillByDefault(
          [](absl::string_view) { return std::vector<TensorBuffer>{}; });
  EXPECT_THAT(Any2AnyWhisperAudioStage::Create(&audio_queue, &meta_queue,
                                               std::move(empty_runner), opts),
              StatusIs(absl::StatusCode::kInvalidArgument));

  EXPECT_THAT(Any2AnyWhisperAudioStage::Create(
                  &audio_queue, &meta_queue,
                  MakeMockRunner(/*hidden_dim=*/3, /*num_mel_bins=*/80,
                                 /*chunk_frames=*/10, /*out_tokens=*/1),
                  opts),
              StatusIs(absl::StatusCode::kInvalidArgument));

  auto bad_in_runner = MakeMockRunner(4);
  ON_CALL(*bad_in_runner, CreateInputBuffers(absl::string_view("")))
      .WillByDefault(
          [](absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
            LITERT_ASSIGN_OR_RETURN(auto buf,
                                    support::CreateTensorBuffer<float>({1, 7}));
            std::vector<TensorBuffer> bufs;
            bufs.push_back(std::move(buf));
            return bufs;
          });
  EXPECT_THAT(Any2AnyWhisperAudioStage::Create(&audio_queue, &meta_queue,
                                               std::move(bad_in_runner), opts),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(Any2AnyWhisperAudioStageTest,
     OrdersMetadataAndAudioResamplesResetsAndClampsTokens) {
  SequencedVariantQueue<OmniSession::Input, AudioInput> audio_queue;
  SequencedVariantQueue<OmniSession::Input, AudioInputMetadata> meta_queue;
  Any2AnyWhisperAudioStage::Options opts{
      .llm_hidden_size = 4,
      .initial_input_sample_rate_hz = 16000,
      .target_sample_rate_hz = 16000,
      .num_mel_bins = 80,
  };
  ASSERT_OK_AND_ASSIGN(
      auto stage, Any2AnyWhisperAudioStage::Create(
                      &audio_queue, &meta_queue,
                      MakeMockRunner(/*hidden_dim=*/4, /*num_mel_bins=*/80,
                                     /*chunk_frames=*/10, /*out_tokens=*/4),
                      opts));

  EXPECT_FALSE(stage->NeedSchedule());
  EXPECT_THAT(stage->Schedule(), StatusIs(absl::StatusCode::kNotFound));

  // Push audio at seqnum 0 (16 kHz, 15 mel frames -> 4 + 2 tokens), metadata at
  // seqnum 1 (24 kHz), and 24 kHz audio at seqnum 2 (2400 samples -> 10 mel
  // frames -> 4 tokens), plus empty audio at seqnum 3.
  ASSERT_OK(audio_queue.PushBack(
      0, AudioInput{.pcm_samples = std::vector<float>(2400, 0.1f)}));
  ASSERT_OK(meta_queue.PushBack(
      1, AudioInputMetadata{.sample_rate_hz = 24000, .num_channels = 1}));
  ASSERT_OK(audio_queue.PushBack(
      2, AudioInput{.pcm_samples = std::vector<float>(2400, 0.2f)}));
  ASSERT_OK(audio_queue.PushBack(3, AudioInput{}));

  EXPECT_TRUE(stage->NeedSchedule());
  ASSERT_OK(stage->Schedule());

  ASSERT_OK_AND_ASSIGN(auto seg0, stage->GetOutput());
  EXPECT_EQ(seg0.seqnum, 0);
  ASSERT_EQ(seg0.item.size(), 2);
  EXPECT_EQ(seg0.item[0].valid_tokens, 4);
  EXPECT_EQ(seg0.item[0].embeddings.size(), 16);
  EXPECT_EQ(seg0.item[1].valid_tokens, 2);
  EXPECT_EQ(seg0.item[1].embeddings.size(), 8);
  EXPECT_THAT(seg0.item[1].embeddings, ElementsAre(1, 2, 3, 4, 11, 12, 13, 14));

  ASSERT_OK_AND_ASSIGN(auto seg1, stage->GetOutput());
  EXPECT_EQ(seg1.seqnum, 1);
  EXPECT_TRUE(seg1.item.empty());

  ASSERT_OK_AND_ASSIGN(auto seg2, stage->GetOutput());
  EXPECT_EQ(seg2.seqnum, 2);
  ASSERT_EQ(seg2.item.size(), 1);
  EXPECT_EQ(seg2.item[0].valid_tokens, 4);
  EXPECT_EQ(seg2.item[0].embeddings.size(), 16);

  ASSERT_OK_AND_ASSIGN(auto seg3, stage->GetOutput());
  EXPECT_EQ(seg3.seqnum, 3);
  EXPECT_TRUE(seg3.item.empty());

  // Reset restores initial sample rate (16 kHz) and clears resampler/mel state.
  stage->Reset();
  ASSERT_OK(audio_queue.PushBack(
      4, AudioInput{.pcm_samples = std::vector<float>(1600, 0.1f)}));
  ASSERT_OK(stage->Schedule());
  ASSERT_OK_AND_ASSIGN(auto seg4, stage->GetOutput());
  EXPECT_EQ(seg4.seqnum, 4);
  ASSERT_EQ(seg4.item.size(), 1);
  EXPECT_EQ(seg4.item[0].valid_tokens, 4);

  // Invalid metadata stays queued until popped; too-short audio fails.
  ASSERT_OK(meta_queue.PushBack(
      5, AudioInputMetadata{.sample_rate_hz = 0, .num_channels = 1}));
  EXPECT_THAT(stage->Schedule(), StatusIs(absl::StatusCode::kInvalidArgument));
  ASSERT_OK(meta_queue.PopFront());

  ASSERT_OK(meta_queue.PushBack(
      6, AudioInputMetadata{.sample_rate_hz = 16000, .num_channels = 2}));
  EXPECT_THAT(stage->Schedule(), StatusIs(absl::StatusCode::kInvalidArgument));
  ASSERT_OK(meta_queue.PopFront());

  ASSERT_OK(audio_queue.PushBack(
      7, AudioInput{.pcm_samples = std::vector<float>(50, 0.1f)}));
  EXPECT_THAT(stage->Schedule(), StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(Any2AnyWhisperAudioStageTest, ClampsOutOfRangeCustomTokenCounts) {
  SequencedVariantQueue<OmniSession::Input, AudioInput> audio_queue;
  SequencedVariantQueue<OmniSession::Input, AudioInputMetadata> meta_queue;
  Any2AnyWhisperAudioStage::Options high_opts{
      .llm_hidden_size = 4,
      .initial_input_sample_rate_hz = 16000,
      .target_sample_rate_hz = 16000,
      .num_mel_bins = 80,
      .compute_valid_tokens = [](int, int, int) { return 99; },
  };
  ASSERT_OK_AND_ASSIGN(
      auto high_stage,
      Any2AnyWhisperAudioStage::Create(
          &audio_queue, &meta_queue,
          MakeMockRunner(/*hidden_dim=*/4, /*num_mel_bins=*/80,
                         /*chunk_frames=*/10, /*out_tokens=*/4),
          std::move(high_opts)));

  ASSERT_OK(audio_queue.PushBack(
      0, AudioInput{.pcm_samples = std::vector<float>(1600, 0.1f)}));
  ASSERT_OK(high_stage->Schedule());
  ASSERT_OK_AND_ASSIGN(auto high_seg, high_stage->GetOutput());
  ASSERT_EQ(high_seg.item.size(), 1);
  EXPECT_EQ(high_seg.item[0].valid_tokens, 4);
  EXPECT_EQ(high_seg.item[0].embeddings.size(), 16);

  Any2AnyWhisperAudioStage::Options low_opts{
      .llm_hidden_size = 4,
      .initial_input_sample_rate_hz = 16000,
      .target_sample_rate_hz = 16000,
      .num_mel_bins = 80,
      .compute_valid_tokens = [](int, int, int) { return 0; },
  };
  ASSERT_OK_AND_ASSIGN(
      auto low_stage, Any2AnyWhisperAudioStage::Create(
                          &audio_queue, &meta_queue,
                          MakeMockRunner(/*hidden_dim=*/4, /*num_mel_bins=*/80,
                                         /*chunk_frames=*/10, /*out_tokens=*/4),
                          std::move(low_opts)));

  ASSERT_OK(audio_queue.PushBack(
      1, AudioInput{.pcm_samples = std::vector<float>(1600, 0.1f)}));
  ASSERT_OK(low_stage->Schedule());
  ASSERT_OK_AND_ASSIGN(auto low_seg, low_stage->GetOutput());
  ASSERT_EQ(low_seg.item.size(), 1);
  EXPECT_EQ(low_seg.item[0].valid_tokens, 1);
  EXPECT_EQ(low_seg.item[0].embeddings.size(), 4);
}

TEST(Any2AnyWhisperAudioStageTest, ResetsResamplerStateOnSameRateMetadata) {
  SequencedVariantQueue<OmniSession::Input, AudioInput> audio_queue;
  SequencedVariantQueue<OmniSession::Input, AudioInputMetadata> meta_queue;
  Any2AnyWhisperAudioStage::Options opts{
      .llm_hidden_size = 4,
      .initial_input_sample_rate_hz = 16000,
      .target_sample_rate_hz = 16000,
      .num_mel_bins = 80,
  };
  std::vector<std::vector<float>> captured_mels;
  ASSERT_OK_AND_ASSIGN(
      auto stage,
      Any2AnyWhisperAudioStage::Create(
          &audio_queue, &meta_queue,
          MakeMockRunner(/*hidden_dim=*/4, /*num_mel_bins=*/80,
                         /*chunk_frames=*/10, /*out_tokens=*/4, &captured_mels),
          opts));

  const std::vector<float> clip = MakeNonConstantAudioClip(2401);

  // Clip 1 at 24 kHz -> 16 kHz from clean resampler state.
  ASSERT_OK(meta_queue.PushBack(
      0, AudioInputMetadata{.sample_rate_hz = 24000, .num_channels = 1}));
  ASSERT_OK(audio_queue.PushBack(1, AudioInput{.pcm_samples = clip}));
  ASSERT_OK(stage->Schedule());
  ASSERT_OK(stage->GetOutput());  // metadata
  ASSERT_OK(stage->GetOutput());  // clip 1
  ASSERT_EQ(captured_mels.size(), 1);

  // Clip 2 without intervening metadata: leftover resampler filter/phase state
  // must visibly alter the extracted mel spectrogram (negative control).
  ASSERT_OK(audio_queue.PushBack(2, AudioInput{.pcm_samples = clip}));
  ASSERT_OK(stage->Schedule());
  ASSERT_OK(stage->GetOutput());  // clip 2
  ASSERT_EQ(captured_mels.size(), 2);
  EXPECT_NE(captured_mels[1], captured_mels[0]);

  // Clip 3 preceded by same-rate AudioInputMetadata{24000}: resampler resets
  // unconditionally and reproduces the exact clean-state mel spectrogram.
  ASSERT_OK(meta_queue.PushBack(
      3, AudioInputMetadata{.sample_rate_hz = 24000, .num_channels = 1}));
  ASSERT_OK(audio_queue.PushBack(4, AudioInput{.pcm_samples = clip}));
  ASSERT_OK(stage->Schedule());
  ASSERT_OK(stage->GetOutput());  // metadata
  ASSERT_OK(stage->GetOutput());  // clip 3
  ASSERT_EQ(captured_mels.size(), 3);
  EXPECT_EQ(captured_mels[2], captured_mels[0]);
}

TEST(Any2AnyWhisperAudioStageTest, ResetsResamplerStateOnStageReset) {
  SequencedVariantQueue<OmniSession::Input, AudioInput> audio_queue;
  SequencedVariantQueue<OmniSession::Input, AudioInputMetadata> meta_queue;
  Any2AnyWhisperAudioStage::Options opts{
      .llm_hidden_size = 4,
      .initial_input_sample_rate_hz = 24000,
      .target_sample_rate_hz = 16000,
      .num_mel_bins = 80,
  };
  std::vector<std::vector<float>> captured_mels;
  ASSERT_OK_AND_ASSIGN(
      auto stage,
      Any2AnyWhisperAudioStage::Create(
          &audio_queue, &meta_queue,
          MakeMockRunner(/*hidden_dim=*/4, /*num_mel_bins=*/80,
                         /*chunk_frames=*/10, /*out_tokens=*/4, &captured_mels),
          opts));

  const std::vector<float> clip = MakeNonConstantAudioClip(2401);

  ASSERT_OK(audio_queue.PushBack(0, AudioInput{.pcm_samples = clip}));
  ASSERT_OK(stage->Schedule());
  ASSERT_OK(stage->GetOutput());
  ASSERT_EQ(captured_mels.size(), 1);

  // Consecutive clip without Reset() sees leftover resampler state.
  ASSERT_OK(audio_queue.PushBack(1, AudioInput{.pcm_samples = clip}));
  ASSERT_OK(stage->Schedule());
  ASSERT_OK(stage->GetOutput());
  ASSERT_EQ(captured_mels.size(), 2);
  EXPECT_NE(captured_mels[1], captured_mels[0]);

  // Reset() clears resampler state so the same clip reproduces clean-state mel.
  stage->Reset();
  ASSERT_OK(audio_queue.PushBack(2, AudioInput{.pcm_samples = clip}));
  ASSERT_OK(stage->Schedule());
  ASSERT_OK(stage->GetOutput());
  ASSERT_EQ(captured_mels.size(), 3);
  EXPECT_EQ(captured_mels[2], captured_mels[0]);
}

TEST(Any2AnyWhisperAudioStageTest, CreateRejectsInvalidInputTensorShapeOrType) {
  SequencedVariantQueue<OmniSession::Input, AudioInput> audio_queue;
  SequencedVariantQueue<OmniSession::Input, AudioInputMetadata> meta_queue;
  const Any2AnyWhisperAudioStage::Options opts{
      .llm_hidden_size = 4,
      .initial_input_sample_rate_hz = 16000,
      .target_sample_rate_hz = 16000,
      .num_mel_bins = 80,
  };

  // Non-Float32 input buffer is rejected.
  {
    auto runner = MakeMockRunner(4);
    ON_CALL(*runner, CreateInputBuffers(absl::string_view("")))
        .WillByDefault(
            [](absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
              LITERT_ASSIGN_OR_RETURN(
                  auto buf, support::CreateTensorBuffer<int32_t>({1, 80, 10}));
              std::vector<TensorBuffer> bufs;
              bufs.push_back(std::move(buf));
              return bufs;
            });
    EXPECT_THAT(
        Any2AnyWhisperAudioStage::Create(&audio_queue, &meta_queue,
                                         std::move(runner), opts),
        StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("Float32")));
  }

  auto expect_rejected_float_shape = [&](const Dimensions& shape,
                                         absl::string_view expected_substr) {
    auto runner = MakeMockRunner(4);
    ON_CALL(*runner, CreateInputBuffers(absl::string_view("")))
        .WillByDefault([shape](absl::string_view)
                           -> absl::StatusOr<std::vector<TensorBuffer>> {
          LITERT_ASSIGN_OR_RETURN(
              auto buf, support::CreateTensorBuffer<float>(Dimensions(shape)));
          std::vector<TensorBuffer> bufs;
          bufs.push_back(std::move(buf));
          return bufs;
        });
    EXPECT_THAT(Any2AnyWhisperAudioStage::Create(&audio_queue, &meta_queue,
                                                 std::move(runner), opts),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr(expected_substr)));
  };

  // Neither spatial dim matches num_mel_bins=80 (rank-3 and rank-2).
  expect_rejected_float_shape({1, 64, 10}, "does not match num_mel_bins");
  expect_rejected_float_shape({64, 10}, "does not match num_mel_bins");

  // Batch size != 1 and zero-sized rank-2 shapes are rejected.
  expect_rejected_float_shape({2, 80, 10}, "positive shape");
  expect_rejected_float_shape({2, 80, 1}, "positive shape");
  expect_rejected_float_shape({80, 0}, "positive shape");
  expect_rejected_float_shape({0, 80}, "positive shape");

  // Non-Float32 output buffer is rejected.
  {
    auto runner = MakeMockRunner(4);
    ON_CALL(*runner, CreateOutputBuffers(absl::string_view("")))
        .WillByDefault(
            [](absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
              LITERT_ASSIGN_OR_RETURN(
                  auto buf, support::CreateTensorBuffer<int32_t>({1, 4, 4}));
              std::vector<TensorBuffer> bufs;
              bufs.push_back(std::move(buf));
              return bufs;
            });
    EXPECT_THAT(
        Any2AnyWhisperAudioStage::Create(&audio_queue, &meta_queue,
                                         std::move(runner), opts),
        StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("Float32")));
  }
}

TEST(Any2AnyWhisperAudioStageTest, SupportsTimeMajorSingleAndMultiChunkLayout) {
  constexpr int kHiddenDim = 4;
  constexpr int kNumMelBins = 80;
  constexpr int kChunkFrames = 10;
  constexpr int kOutTokens = 4;

  const Any2AnyWhisperAudioStage::Options opts{
      .llm_hidden_size = kHiddenDim,
      .initial_input_sample_rate_hz = 16000,
      .target_sample_rate_hz = 16000,
      .num_mel_bins = kNumMelBins,
  };
  // 2400 samples @ 16 kHz = 15 mel frames = 2 chunks of 10 frames.
  const std::vector<float> multi_chunk_clip = MakeNonConstantAudioClip(2400);
  // 1600 samples @ 16 kHz = 10 mel frames = 1 chunk.
  const std::vector<float> single_chunk_clip = MakeNonConstantAudioClip(1600);

  auto run_with_shape = [&](const Dimensions& input_shape,
                            const std::vector<float>& clip)
      -> absl::StatusOr<std::vector<std::vector<float>>> {
    SequencedVariantQueue<OmniSession::Input, AudioInput> audio_queue;
    SequencedVariantQueue<OmniSession::Input, AudioInputMetadata> meta_queue;
    std::vector<std::vector<float>> captured_mels;
    auto runner = MakeMockRunner(kHiddenDim, kNumMelBins, kChunkFrames,
                                 kOutTokens, &captured_mels);
    ON_CALL(*runner, CreateInputBuffers(absl::string_view("")))
        .WillByDefault([input_shape](absl::string_view)
                           -> absl::StatusOr<std::vector<TensorBuffer>> {
          LITERT_ASSIGN_OR_RETURN(auto buf, support::CreateTensorBuffer<float>(
                                                Dimensions(input_shape)));
          std::vector<TensorBuffer> bufs;
          bufs.push_back(std::move(buf));
          return bufs;
        });
    ABSL_ASSIGN_OR_RETURN(
        std::unique_ptr<Any2AnyWhisperAudioStage> stage,
        Any2AnyWhisperAudioStage::Create(&audio_queue, &meta_queue,
                                         std::move(runner), opts));
    ABSL_RETURN_IF_ERROR(
        audio_queue.PushBack(0, AudioInput{.pcm_samples = clip}));
    ABSL_RETURN_IF_ERROR(stage->Schedule());
    ABSL_RETURN_IF_ERROR(stage->GetOutput().status());
    return captured_mels;
  };

  // Multi-chunk: compare [1, M, T] against [M, T], [1, T, M], and [T, M].
  ASSERT_OK_AND_ASSIGN(
      const std::vector<std::vector<float>> mel_major_multi,
      run_with_shape({1, kNumMelBins, kChunkFrames}, multi_chunk_clip));
  ASSERT_OK_AND_ASSIGN(
      const std::vector<std::vector<float>> rank2_mel_major_multi,
      run_with_shape({kNumMelBins, kChunkFrames}, multi_chunk_clip));
  ASSERT_OK_AND_ASSIGN(
      const std::vector<std::vector<float>> time_major_multi,
      run_with_shape({1, kChunkFrames, kNumMelBins}, multi_chunk_clip));
  ASSERT_OK_AND_ASSIGN(
      const std::vector<std::vector<float>> rank2_time_major_multi,
      run_with_shape({kChunkFrames, kNumMelBins}, multi_chunk_clip));

  ASSERT_EQ(mel_major_multi.size(), 2);
  ASSERT_EQ(time_major_multi.size(), 2);
  EXPECT_EQ(rank2_mel_major_multi, mel_major_multi);
  EXPECT_EQ(rank2_time_major_multi, time_major_multi);
  for (size_t c = 0; c < mel_major_multi.size(); ++c) {
    ASSERT_EQ(mel_major_multi[c].size(),
              static_cast<size_t>(kNumMelBins) * kChunkFrames);
    ASSERT_EQ(time_major_multi[c].size(),
              static_cast<size_t>(kNumMelBins) * kChunkFrames);
    for (int f = 0; f < kChunkFrames; ++f) {
      for (int m = 0; m < kNumMelBins; ++m) {
        EXPECT_FLOAT_EQ(time_major_multi[c][f * kNumMelBins + m],
                        mel_major_multi[c][m * kChunkFrames + f]);
      }
    }
  }

  // Multi-chunk time-major writes strictly `chunk_frames * num_mel_bins` floats
  // per chunk without overflowing past the end of the host tensor buffer.
  {
    auto env = ::litert::Environment::Create({});
    ASSERT_TRUE(env.HasValue());
    constexpr size_t kChunkElems =
        static_cast<size_t>(kChunkFrames) * kNumMelBins;
    constexpr float kCanary = 12345.0f;
    alignas(kHostMemoryBufferAlignment) std::array<float, kChunkElems + 1>
        backing;
    backing.fill(kCanary);

    SequencedVariantQueue<OmniSession::Input, AudioInput> audio_queue;
    SequencedVariantQueue<OmniSession::Input, AudioInputMetadata> meta_queue;
    std::vector<std::vector<float>> captured_mels;
    auto runner = MakeMockRunner(kHiddenDim, kNumMelBins, kChunkFrames,
                                 kOutTokens, &captured_mels);
    ON_CALL(*runner, CreateInputBuffers(absl::string_view("")))
        .WillByDefault([&env, &backing](absl::string_view)
                           -> absl::StatusOr<std::vector<TensorBuffer>> {
          RankedTensorType type(
              ElementType::Float32,
              Layout(Dimensions({1, kChunkFrames, kNumMelBins})));
          LITERT_ASSIGN_OR_RETURN(
              TensorBuffer buf,
              TensorBuffer::CreateFromHostMemory(*env, type, backing.data(),
                                                 kChunkElems * sizeof(float)));
          std::vector<TensorBuffer> bufs;
          bufs.push_back(std::move(buf));
          return bufs;
        });
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<Any2AnyWhisperAudioStage> stage,
        Any2AnyWhisperAudioStage::Create(&audio_queue, &meta_queue,
                                         std::move(runner), opts));
    ASSERT_OK(
        audio_queue.PushBack(0, AudioInput{.pcm_samples = multi_chunk_clip}));
    ASSERT_OK(stage->Schedule());
    ASSERT_OK(stage->GetOutput());
    EXPECT_EQ(captured_mels, time_major_multi);
    EXPECT_EQ(backing[kChunkElems], kCanary);
  }

  // Single-chunk: compare [1, M, T] against [1, T, M].
  ASSERT_OK_AND_ASSIGN(
      const std::vector<std::vector<float>> mel_major_single,
      run_with_shape({1, kNumMelBins, kChunkFrames}, single_chunk_clip));
  ASSERT_OK_AND_ASSIGN(
      const std::vector<std::vector<float>> time_major_single,
      run_with_shape({1, kChunkFrames, kNumMelBins}, single_chunk_clip));
  ASSERT_EQ(mel_major_single.size(), 1);
  ASSERT_EQ(time_major_single.size(), 1);
  for (int f = 0; f < kChunkFrames; ++f) {
    for (int m = 0; m < kNumMelBins; ++m) {
      EXPECT_FLOAT_EQ(time_major_single[0][f * kNumMelBins + m],
                      mel_major_single[0][m * kChunkFrames + f]);
    }
  }

  // Padded input buffers (Size() > PackedSize()) are accepted using
  // PackedSize().
  {
    auto env = ::litert::Environment::Create({});
    ASSERT_TRUE(env.HasValue());
    constexpr size_t kPackedBytes =
        static_cast<size_t>(kNumMelBins) * kChunkFrames * sizeof(float);
    constexpr size_t kPaddedBytes = kPackedBytes + 64;

    SequencedVariantQueue<OmniSession::Input, AudioInput> audio_queue;
    SequencedVariantQueue<OmniSession::Input, AudioInputMetadata> meta_queue;
    std::vector<std::vector<float>> captured_mels;
    auto runner = MakeMockRunner(kHiddenDim, kNumMelBins, kChunkFrames,
                                 kOutTokens, &captured_mels);
    ON_CALL(*runner, CreateInputBuffers(absl::string_view("")))
        .WillByDefault([&env](absl::string_view)
                           -> absl::StatusOr<std::vector<TensorBuffer>> {
          RankedTensorType type(
              ElementType::Float32,
              Layout(Dimensions({1, kNumMelBins, kChunkFrames})));
          LITERT_ASSIGN_OR_RETURN(
              TensorBuffer buf,
              TensorBuffer::CreateManaged(*env, TensorBufferType::kHostMemory,
                                          type, kPaddedBytes));
          LITERT_ASSIGN_OR_RETURN(size_t alloc_size, buf.Size());
          LITERT_ASSIGN_OR_RETURN(size_t packed_size, buf.PackedSize());
          EXPECT_GT(alloc_size, packed_size);
          std::vector<TensorBuffer> bufs;
          bufs.push_back(std::move(buf));
          return bufs;
        });
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<Any2AnyWhisperAudioStage> stage,
        Any2AnyWhisperAudioStage::Create(&audio_queue, &meta_queue,
                                         std::move(runner), opts));
    ASSERT_OK(
        audio_queue.PushBack(0, AudioInput{.pcm_samples = single_chunk_clip}));
    ASSERT_OK(stage->Schedule());
    ASSERT_OK(stage->GetOutput());
    EXPECT_EQ(captured_mels, mel_major_single);
  }
}

TEST(Any2AnyWhisperAudioStageTest, HandlesPaddedOutputTensorBuffer) {
  constexpr int kHiddenDim = 4;
  constexpr int kNumMelBins = 80;
  constexpr int kChunkFrames = 10;
  constexpr int kOutTokens = 4;
  static constexpr size_t kPackedBytes =
      static_cast<size_t>(kOutTokens) * kHiddenDim * sizeof(float);

  auto env = ::litert::Environment::Create({});
  ASSERT_TRUE(env.HasValue());
  const std::vector<float> clip = MakeNonConstantAudioClip(1600);

  // Test both padding that is NOT a multiple of token_emb_bytes (16 B) and
  // padding that IS a multiple of token_emb_bytes (64 B = 4 extra tokens).
  for (const size_t padding_bytes : {size_t{20}, size_t{64}}) {
    const size_t padded_bytes = kPackedBytes + padding_bytes;
    int captured_max_chunk_tokens = -1;
    SequencedVariantQueue<OmniSession::Input, AudioInput> audio_queue;
    SequencedVariantQueue<OmniSession::Input, AudioInputMetadata> meta_queue;
    const Any2AnyWhisperAudioStage::Options opts{
        .llm_hidden_size = kHiddenDim,
        .initial_input_sample_rate_hz = 16000,
        .target_sample_rate_hz = 16000,
        .num_mel_bins = kNumMelBins,
        .compute_valid_tokens =
            [&captured_max_chunk_tokens](int /*valid_frames*/,
                                         int /*chunk_frames*/,
                                         int max_chunk_tokens) {
              captured_max_chunk_tokens = max_chunk_tokens;
              return max_chunk_tokens;
            },
    };
    auto runner =
        MakeMockRunner(kHiddenDim, kNumMelBins, kChunkFrames, kOutTokens);
    ON_CALL(*runner, CreateOutputBuffers(absl::string_view("")))
        .WillByDefault([&env, padded_bytes](absl::string_view)
                           -> absl::StatusOr<std::vector<TensorBuffer>> {
          RankedTensorType type(
              ElementType::Float32,
              Layout(Dimensions({1, kOutTokens, kHiddenDim})));
          LITERT_ASSIGN_OR_RETURN(
              TensorBuffer buf,
              TensorBuffer::CreateManaged(*env, TensorBufferType::kHostMemory,
                                          type, padded_bytes));
          LITERT_ASSIGN_OR_RETURN(size_t alloc_size, buf.Size());
          LITERT_ASSIGN_OR_RETURN(size_t packed_size, buf.PackedSize());
          EXPECT_GT(alloc_size, packed_size);
          EXPECT_EQ(packed_size, kPackedBytes);
          // Fill the entire allocation (including trailing padding) with a
          // sentinel so any over-read is detectable.
          {
            LITERT_ASSIGN_OR_RETURN(auto lock_and_addr,
                                    TensorBufferScopedLock::Create<float>(
                                        buf, TensorBuffer::LockMode::kWrite));
            float* data = lock_and_addr.second;
            for (size_t i = 0; i < alloc_size / sizeof(float); ++i) {
              data[i] = -999.0f;
            }
          }
          std::vector<TensorBuffer> bufs;
          bufs.push_back(std::move(buf));
          return bufs;
        });

    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<Any2AnyWhisperAudioStage> stage,
        Any2AnyWhisperAudioStage::Create(&audio_queue, &meta_queue,
                                         std::move(runner), opts));
    ASSERT_OK(audio_queue.PushBack(0, AudioInput{.pcm_samples = clip}));
    ASSERT_OK(stage->Schedule());
    ASSERT_OK_AND_ASSIGN(auto seg, stage->GetOutput());
    EXPECT_EQ(captured_max_chunk_tokens, kOutTokens);
    ASSERT_EQ(seg.item.size(), 1);
    EXPECT_EQ(seg.item[0].valid_tokens, kOutTokens);
    EXPECT_EQ(seg.item[0].embeddings.size(),
              static_cast<size_t>(kOutTokens) * kHiddenDim);
    EXPECT_THAT(seg.item[0].embeddings,
                ElementsAre(1, 2, 3, 4, 11, 12, 13, 14, 21, 22, 23, 24, 31, 32,
                            33, 34));
  }
}

}  // namespace
}  // namespace litert::omni::any2any
