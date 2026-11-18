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

#include "omni/any2any/qwen_omni/qwen_omni_audio_stage.h"

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "omni/any2any/any2any_input_fan_out_stage.h"
#include "omni/any2any/any2any_test_utils.h"
#include "omni/any2any/qwen_omni/qwen_omni_io_types.h"
#include "omni/any2any/qwen_omni/qwen_omni_model_config.h"
#include "omni/any2any/qwen_omni/qwen_omni_test_utils.h"
#include "omni/base/fan_out_stage.h"
#include "omni/base/io_types.h"
#include "omni/base/litert_runner.h"
#include "omni/base/mock_litert_runner.h"
#include "omni/base/model_utils.h"
#include "omni/omni_session.h"
#include "runtime/executor/executor_settings_base.h"
#include "support/util/convert_tensor_buffer.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep
#include "tflite/schema/schema_generated.h"  // from @litert

namespace litert::omni::any2any {
namespace {

using ::absl_testing::StatusIs;
using ::testing::_;
using ::testing::ElementsAre;
using ::testing::NiceMock;

std::unique_ptr<NiceMock<MockLiteRtRunner>> MakeMockAudioRunner(
    int hidden_dim, int num_mel_bins, int chunk_frames, int out_tokens) {
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
          [out_tokens, hidden_dim](
              absl::string_view, absl::Span<const TensorBuffer>,
              absl::Span<const TensorBuffer> outputs) -> absl::Status {
            std::vector<float> out(static_cast<size_t>(out_tokens) * hidden_dim,
                                   1.0f);
            auto& out_buf = const_cast<TensorBuffer&>(outputs[0]);
            LITERT_RETURN_IF_ERROR(
                out_buf.Write<float>(absl::MakeConstSpan(out)));
            return absl::OkStatus();
          });
  return runner;
}

TEST(QwenOmniAudioStageTest,
     AudioStageValidatesInputsResamplesAndFramesMultiWindowClip) {
  // Arrange
  auto input_source = std::make_unique<PushInputSource>();
  auto fan_out = std::make_unique<Any2AnyInputFanOutStage>(input_source.get());
  QwenOmniModelConfig config =
      MakeSmallQwenOmniTestConfig(/*generate_audio=*/false);

  auto env = ::litert::Environment::Create({});
  ASSERT_TRUE(env.HasValue());
  ModelOptions opts{.backend = lm::Backend::CPU, .num_threads = 1};

  std::string audio_buf = BuildMinimalTfliteModelBuffer(
      {{"mel_features", {1, 128, 10}, tflite::TensorType_FLOAT32}},
      {{"audio_embeds", {1, 2, 4}, tflite::TensorType_FLOAT32}});
  ASSERT_OK_AND_ASSIGN(
      auto audio_model,
      CreateCompiledModelFromBuffer(*env, opts, audio_buf, "audio_encoder"));
  ASSERT_OK_AND_ASSIGN(
      auto audio_stage,
      QwenOmniAudioStage::Create(
          &fan_out->AudioQueue(), &fan_out->AudioInputMetadataQueue(),
          std::make_unique<LiteRtRunnerImpl>(&audio_model), config));

  // Act: push 24 kHz audio metadata + 4800 samples (200 ms -> 20 mel frames =
  // two 10-frame encoder windows), verify single <|audio_bos|>...<|audio_eos|>
  // framing across both windows.
  ASSERT_OK(input_source->PushInput(
      AudioInputMetadata{.sample_rate_hz = 24000, .num_channels = 1}));
  ASSERT_OK(input_source->PushInput(
      AudioInput{.pcm_samples = std::vector<float>(4800, 0.2f)}));
  while (fan_out->NeedSchedule()) {
    ASSERT_OK(fan_out->Schedule());
  }
  ASSERT_OK(audio_stage->Schedule());

  // Assert
  ASSERT_OK_AND_ASSIGN(auto meta_seg, audio_stage->GetOutput());
  EXPECT_EQ(meta_seg.item.num_tokens, 0);
  ASSERT_OK_AND_ASSIGN(auto audio_seg, audio_stage->GetOutput());
  EXPECT_EQ(audio_seg.item.modality,
            QwenOmniEmbeddingSegment::Modality::kAudio);
  EXPECT_EQ(audio_seg.item.num_tokens, 4);
  EXPECT_EQ(audio_seg.item.embeddings.size(), 16);
  EXPECT_THAT(audio_seg.item.token_ids,
              ElementsAre(config.audio_bos_token_id, config.audio_token_id,
                          config.audio_token_id, config.audio_token_id,
                          config.audio_token_id, config.audio_eos_token_id));
}

TEST(QwenOmniAudioStageTest, CreateValidatesConfigAndRunnerBuffers) {
  SequencedVariantQueue<OmniSession::Input, AudioInput> audio_queue;
  SequencedVariantQueue<OmniSession::Input, AudioInputMetadata> meta_queue;
  QwenOmniModelConfig config =
      MakeSmallQwenOmniTestConfig(/*generate_audio=*/false);

  QwenOmniModelConfig bad_hidden = config;
  bad_hidden.llm_hidden_size = 0;
  EXPECT_THAT(
      QwenOmniAudioStage::Create(
          &audio_queue, &meta_queue,
          MakeMockAudioRunner(config.llm_hidden_size, config.audio_num_mel_bins,
                              /*chunk_frames=*/12, /*out_tokens=*/3),
          bad_hidden),
      StatusIs(absl::StatusCode::kInvalidArgument));

  QwenOmniModelConfig bad_sr = config;
  bad_sr.audio_input_sample_rate_hz = 0;
  EXPECT_THAT(
      QwenOmniAudioStage::Create(
          &audio_queue, &meta_queue,
          MakeMockAudioRunner(config.llm_hidden_size, config.audio_num_mel_bins,
                              /*chunk_frames=*/12, /*out_tokens=*/3),
          bad_sr),
      StatusIs(absl::StatusCode::kInvalidArgument));

  QwenOmniModelConfig bad_mels = config;
  bad_mels.audio_num_mel_bins = 0;
  EXPECT_THAT(
      QwenOmniAudioStage::Create(
          &audio_queue, &meta_queue,
          MakeMockAudioRunner(config.llm_hidden_size, config.audio_num_mel_bins,
                              /*chunk_frames=*/12, /*out_tokens=*/3),
          bad_mels),
      StatusIs(absl::StatusCode::kInvalidArgument));

  auto empty_runner = std::make_unique<NiceMock<MockLiteRtRunner>>();
  ON_CALL(*empty_runner, CreateInputBuffers(absl::string_view("")))
      .WillByDefault(
          [](absl::string_view) { return std::vector<TensorBuffer>{}; });
  ON_CALL(*empty_runner, CreateOutputBuffers(absl::string_view("")))
      .WillByDefault(
          [](absl::string_view) { return std::vector<TensorBuffer>{}; });
  EXPECT_THAT(QwenOmniAudioStage::Create(&audio_queue, &meta_queue,
                                         std::move(empty_runner), config),
              StatusIs(absl::StatusCode::kInvalidArgument));

  EXPECT_THAT(
      QwenOmniAudioStage::Create(
          &audio_queue, &meta_queue,
          MakeMockAudioRunner(/*hidden_dim=*/3, config.audio_num_mel_bins,
                              /*chunk_frames=*/12, /*out_tokens=*/1),
          config),
      StatusIs(absl::StatusCode::kInvalidArgument));

  auto bad_in_runner = MakeMockAudioRunner(config.llm_hidden_size,
                                           config.audio_num_mel_bins, 12, 3);
  ON_CALL(*bad_in_runner, CreateInputBuffers(absl::string_view("")))
      .WillByDefault(
          [](absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
            LITERT_ASSIGN_OR_RETURN(auto buf,
                                    support::CreateTensorBuffer<float>({1, 7}));
            std::vector<TensorBuffer> bufs;
            bufs.push_back(std::move(buf));
            return bufs;
          });
  EXPECT_THAT(QwenOmniAudioStage::Create(&audio_queue, &meta_queue,
                                         std::move(bad_in_runner), config),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(QwenOmniAudioStageTest,
     ComputeValidAudioTokensHandlesFourXDownsamplingAndProportionalFallback) {
  SequencedVariantQueue<OmniSession::Input, AudioInput> audio_queue;
  SequencedVariantQueue<OmniSession::Input, AudioInputMetadata> meta_queue;
  QwenOmniModelConfig config =
      MakeSmallQwenOmniTestConfig(/*generate_audio=*/false);

  // Case 1: 4x downsampling branch (`chunk_frames == max_chunk_tokens * 4`,
  // 12 frames -> 3 tokens).
  ASSERT_OK_AND_ASSIGN(
      auto four_x_stage,
      QwenOmniAudioStage::Create(
          &audio_queue, &meta_queue,
          MakeMockAudioRunner(config.llm_hidden_size, config.audio_num_mel_bins,
                              /*chunk_frames=*/12, /*out_tokens=*/3),
          config));

  // 1 frame (160 samples): after_cnn = 1 (< 2) -> 1 token.
  ASSERT_OK(audio_queue.PushBack(
      0, AudioInput{.pcm_samples = std::vector<float>(160, 0.1f)}));
  // 7 frames (1120 samples): after_cnn = 4 (>= 2) -> (4 - 2) / 2 + 1 = 2.
  ASSERT_OK(audio_queue.PushBack(
      1, AudioInput{.pcm_samples = std::vector<float>(1120, 0.1f)}));
  // 12 frames (1920 samples): full chunk -> 3 tokens.
  ASSERT_OK(audio_queue.PushBack(
      2, AudioInput{.pcm_samples = std::vector<float>(1920, 0.1f)}));
  ASSERT_OK(four_x_stage->Schedule());

  ASSERT_OK_AND_ASSIGN(auto seg_1f, four_x_stage->GetOutput());
  EXPECT_EQ(seg_1f.item.num_tokens, 1);
  EXPECT_EQ(seg_1f.item.embeddings.size(), config.llm_hidden_size);

  ASSERT_OK_AND_ASSIGN(auto seg_7f, four_x_stage->GetOutput());
  EXPECT_EQ(seg_7f.item.num_tokens, 2);
  EXPECT_EQ(seg_7f.item.embeddings.size(), 2 * config.llm_hidden_size);

  ASSERT_OK_AND_ASSIGN(auto seg_12f, four_x_stage->GetOutput());
  EXPECT_EQ(seg_12f.item.num_tokens, 3);
  EXPECT_EQ(seg_12f.item.embeddings.size(), 3 * config.llm_hidden_size);

  // Case 2: Proportional ceiling fallback (`chunk_frames != max_chunk_tokens *
  // 4`, 10 frames -> 2 tokens).
  ASSERT_OK_AND_ASSIGN(
      auto fallback_stage,
      QwenOmniAudioStage::Create(
          &audio_queue, &meta_queue,
          MakeMockAudioRunner(config.llm_hidden_size, config.audio_num_mel_bins,
                              /*chunk_frames=*/10, /*out_tokens=*/2),
          config));

  // 3 frames (480 samples): ceil(3 * 2 / 10) = 1 token.
  ASSERT_OK(audio_queue.PushBack(
      3, AudioInput{.pcm_samples = std::vector<float>(480, 0.1f)}));
  // 6 frames (960 samples): ceil(6 * 2 / 10) = 2 tokens.
  ASSERT_OK(audio_queue.PushBack(
      4, AudioInput{.pcm_samples = std::vector<float>(960, 0.1f)}));
  ASSERT_OK(fallback_stage->Schedule());

  ASSERT_OK_AND_ASSIGN(auto seg_3f, fallback_stage->GetOutput());
  EXPECT_EQ(seg_3f.item.num_tokens, 1);
  ASSERT_OK_AND_ASSIGN(auto seg_6f, fallback_stage->GetOutput());
  EXPECT_EQ(seg_6f.item.num_tokens, 2);
}

TEST(QwenOmniAudioStageTest,
     HandlesEmptyAudioTooShortAudioInvalidMetadataEmptyScheduleAndReset) {
  SequencedVariantQueue<OmniSession::Input, AudioInput> audio_queue;
  SequencedVariantQueue<OmniSession::Input, AudioInputMetadata> meta_queue;
  QwenOmniModelConfig config =
      MakeSmallQwenOmniTestConfig(/*generate_audio=*/false);

  ASSERT_OK_AND_ASSIGN(
      auto stage,
      QwenOmniAudioStage::Create(
          &audio_queue, &meta_queue,
          MakeMockAudioRunner(config.llm_hidden_size, config.audio_num_mel_bins,
                              /*chunk_frames=*/10, /*out_tokens=*/2),
          config));

  // Empty queue: NeedSchedule is false and Schedule returns NotFound.
  EXPECT_FALSE(stage->NeedSchedule());
  EXPECT_THAT(stage->Schedule(), StatusIs(absl::StatusCode::kNotFound));

  // Empty AudioInput emits an empty segment without framing tokens.
  ASSERT_OK(audio_queue.PushBack(0, AudioInput{}));
  EXPECT_TRUE(stage->NeedSchedule());
  ASSERT_OK(stage->Schedule());
  ASSERT_OK_AND_ASSIGN(auto empty_seg, stage->GetOutput());
  EXPECT_EQ(empty_seg.seqnum, 0);
  EXPECT_EQ(empty_seg.item.num_tokens, 0);
  EXPECT_TRUE(empty_seg.item.token_ids.empty());
  EXPECT_TRUE(empty_seg.item.embeddings.empty());

  // Too-short AudioInput (< 160 samples at 16 kHz) fails with InvalidArgument.
  ASSERT_OK(audio_queue.PushBack(
      1, AudioInput{.pcm_samples = std::vector<float>(50, 0.1f)}));
  EXPECT_THAT(stage->Schedule(), StatusIs(absl::StatusCode::kInvalidArgument));
  ASSERT_OK(audio_queue.PopFront());

  // Invalid AudioInputMetadata (sample_rate_hz <= 0 or num_channels != 1).
  ASSERT_OK(meta_queue.PushBack(
      2, AudioInputMetadata{.sample_rate_hz = 0, .num_channels = 1}));
  EXPECT_THAT(stage->Schedule(), StatusIs(absl::StatusCode::kInvalidArgument));
  ASSERT_OK(meta_queue.PopFront());

  ASSERT_OK(meta_queue.PushBack(
      3, AudioInputMetadata{.sample_rate_hz = 16000, .num_channels = 2}));
  EXPECT_THAT(stage->Schedule(), StatusIs(absl::StatusCode::kInvalidArgument));
  ASSERT_OK(meta_queue.PopFront());

  // Switch sample rate to 24 kHz, feed 24 kHz audio, then Reset() and verify
  // buffered outputs are cleared and sample rate returns to 16 kHz default.
  ASSERT_OK(meta_queue.PushBack(
      4, AudioInputMetadata{.sample_rate_hz = 24000, .num_channels = 1}));
  ASSERT_OK(audio_queue.PushBack(
      5, AudioInput{.pcm_samples = std::vector<float>(2400, 0.2f)}));
  ASSERT_OK(stage->Schedule());
  EXPECT_TRUE(stage->HasOutput());

  stage->Reset();
  EXPECT_FALSE(stage->HasOutput());
  EXPECT_FALSE(stage->NeedSchedule());

  // After Reset(), 1600 samples at the restored 16 kHz default produces a full
  // 10-frame chunk (2 tokens) instead of being downsampled from 24 kHz.
  ASSERT_OK(audio_queue.PushBack(
      6, AudioInput{.pcm_samples = std::vector<float>(1600, 0.2f)}));
  ASSERT_OK(stage->Schedule());
  ASSERT_OK_AND_ASSIGN(auto post_reset_seg, stage->GetOutput());
  EXPECT_EQ(post_reset_seg.seqnum, 6);
  EXPECT_EQ(post_reset_seg.item.num_tokens, 2);
}

TEST(QwenOmniAudioStageTest, Non16kInitialConfigSampleRateResamplesTo16k) {
  SequencedVariantQueue<OmniSession::Input, AudioInput> audio_queue;
  SequencedVariantQueue<OmniSession::Input, AudioInputMetadata> meta_queue;
  QwenOmniModelConfig config =
      MakeSmallQwenOmniTestConfig(/*generate_audio=*/false);
  config.audio_input_sample_rate_hz = 24000;

  ASSERT_OK_AND_ASSIGN(
      auto stage,
      QwenOmniAudioStage::Create(
          &audio_queue, &meta_queue,
          MakeMockAudioRunner(config.llm_hidden_size, config.audio_num_mel_bins,
                              /*chunk_frames=*/10, /*out_tokens=*/2),
          config));

  // 2400 samples at 24 kHz (100 ms) must resample to 1600 samples at 16 kHz
  // (10 mel frames = 1 chunk = 2 tokens), rather than being treated as 15 mel
  // frames (2 chunks = 3 tokens) at 16 kHz.
  ASSERT_OK(audio_queue.PushBack(
      0, AudioInput{.pcm_samples = std::vector<float>(2400, 0.2f)}));
  ASSERT_OK(stage->Schedule());
  ASSERT_OK_AND_ASSIGN(auto seg, stage->GetOutput());
  EXPECT_EQ(seg.seqnum, 0);
  EXPECT_EQ(seg.item.num_tokens, 2);
  EXPECT_EQ(seg.item.embeddings.size(), 2 * config.llm_hidden_size);

  // Switch temporarily to 16 kHz via metadata, then Reset() and verify the
  // configured 24 kHz initial input rate is restored (resampling to 16 kHz).
  ASSERT_OK(meta_queue.PushBack(
      1, AudioInputMetadata{.sample_rate_hz = 16000, .num_channels = 1}));
  ASSERT_OK(stage->Schedule());
  ASSERT_OK(stage->GetOutput());

  stage->Reset();
  ASSERT_OK(audio_queue.PushBack(
      2, AudioInput{.pcm_samples = std::vector<float>(2400, 0.2f)}));
  ASSERT_OK(stage->Schedule());
  ASSERT_OK_AND_ASSIGN(auto post_reset_seg, stage->GetOutput());
  EXPECT_EQ(post_reset_seg.seqnum, 2);
  EXPECT_EQ(post_reset_seg.item.num_tokens, 2);
}

}  // namespace
}  // namespace litert::omni::any2any
