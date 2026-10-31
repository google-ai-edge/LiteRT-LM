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

#include "omni/any2any/minicpm_o/minicpm_o_vocoder_stage.h"

#include <memory>
#include <utility>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "omni/any2any/minicpm_o/minicpm_o_model_config.h"
#include "omni/any2any/minicpm_o/minicpm_o_tts_stage.h"
#include "omni/base/io_types.h"
#include "omni/base/mock_litert_runner.h"
#include "omni/base/stage.h"
#include "support/util/convert_tensor_buffer.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace litert::omni::any2any {
namespace {

using ::absl_testing::StatusIs;
using ::testing::_;
using ::testing::ElementsAre;
using ::testing::FloatNear;
using ::testing::NiceMock;
using ::testing::Return;

class FakeTtsStage : public SingleThreadedStageWithDeque<MiniCpmOTtsOutput> {
 public:
  void Push(MiniCpmOTtsOutput out) { PushOutput(std::move(out)); }

 protected:
  bool NeedScheduleInternal() const override { return false; }
  absl::Status ScheduleInternal() override { return absl::OkStatus(); }
};

TEST(MiniCpmOVocoderStageTest,
     CreateRejectsMissingVocoderWhenAudioGenerationEnabled) {
  MiniCpmOModelConfig config;
  config.generate_audio = true;
  FakeTtsStage tts_stage;

  EXPECT_THAT(MiniCpmOVocoderStage::Create(&tts_stage, config,
                                           /*vocoder_runner=*/nullptr)
                  .status(),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST(MiniCpmOVocoderStageTest, CreateRejectsVocoderWithEmptyBuffers) {
  MiniCpmOModelConfig config;
  config.generate_audio = true;
  FakeTtsStage tts_stage;

  auto vocoder_runner = std::make_unique<NiceMock<MockLiteRtRunner>>();
  ON_CALL(*vocoder_runner, CreateInputBuffers(absl::string_view("")))
      .WillByDefault(
          [](absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
            return std::vector<TensorBuffer>{};
          });
  ON_CALL(*vocoder_runner, CreateOutputBuffers(absl::string_view("")))
      .WillByDefault(
          [](absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
            return std::vector<TensorBuffer>{};
          });

  EXPECT_THAT(MiniCpmOVocoderStage::Create(&tts_stage, config,
                                           std::move(vocoder_runner))
                  .status(),
              StatusIs(absl::StatusCode::kInternal));
}

TEST(MiniCpmOVocoderStageTest,
     EmitsTextOutputAndEndOfOutputTurnWhenNoMelFeatures) {
  MiniCpmOModelConfig config;
  config.generate_audio = false;
  FakeTtsStage tts_stage;

  ASSERT_OK_AND_ASSIGN(auto vocoder_stage,
                       MiniCpmOVocoderStage::Create(&tts_stage, config));

  tts_stage.Push(MiniCpmOTtsOutput{
      .text = "Hello world",
      .has_text = true,
      .is_text_final = true,
  });

  ASSERT_OK(vocoder_stage->Schedule());
  ASSERT_OK_AND_ASSIGN(Output first_out, vocoder_stage->GetOutput());
  ASSERT_TRUE(std::holds_alternative<TextOutput>(first_out));
  EXPECT_EQ(std::get<TextOutput>(first_out).confirmed_text, "Hello world");

  ASSERT_OK_AND_ASSIGN(Output second_out, vocoder_stage->GetOutput());
  EXPECT_TRUE(std::holds_alternative<EndOfOutputTurn>(second_out));
  EXPECT_FALSE(vocoder_stage->HasOutput());
}

TEST(MiniCpmOVocoderStageTest,
     SynthesizesWaveformWithOverlappingContextCrossfadeAndEdgePadding) {
  MiniCpmOModelConfig config;
  config.generate_audio = true;
  config.tts_num_mel_bins = 2;
  config.audio_output_sample_rate_hz = 24000;
  FakeTtsStage tts_stage;

  // Input buffer: [1, 2 bins, 3 chunk_frames] -> mel_chunk_frames = 3,
  // overlap_frames = max(1, min(16, 3 / 4)) = 1.
  // Output buffer: [1, 6 samples] -> upsample_hop = 6 / 3 = 2 samples/frame.
  auto vocoder_runner = std::make_unique<NiceMock<MockLiteRtRunner>>();
  ON_CALL(*vocoder_runner, CreateInputBuffers(absl::string_view("")))
      .WillByDefault(
          [](absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
            LITERT_ASSIGN_OR_RETURN(
                auto in_buf, support::CreateTensorBuffer<float>({1, 2, 3}));
            std::vector<TensorBuffer> bufs;
            bufs.push_back(std::move(in_buf));
            return bufs;
          });
  ON_CALL(*vocoder_runner, CreateOutputBuffers(absl::string_view("")))
      .WillByDefault(
          [](absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
            LITERT_ASSIGN_OR_RETURN(auto out_buf,
                                    support::CreateTensorBuffer<float>({1, 6}));
            std::vector<TensorBuffer> bufs;
            bufs.push_back(std::move(out_buf));
            return bufs;
          });

  std::vector<std::vector<float>> captured_chunks;
  int call_idx = 0;
  EXPECT_CALL(*vocoder_runner, Run(absl::string_view(""), _, _))
      .Times(2)
      .WillRepeatedly(
          [&captured_chunks, &call_idx](
              absl::string_view, absl::Span<const TensorBuffer> inputs,
              absl::Span<const TensorBuffer> outputs) -> absl::Status {
            std::vector<float> chunk_in(6, 0.0f);
            auto& in_buf = const_cast<TensorBuffer&>(inputs[0]);
            LITERT_RETURN_IF_ERROR(
                in_buf.Read<float>(absl::MakeSpan(chunk_in)));
            captured_chunks.push_back(chunk_in);

            // Add +3.0f to chunk 1's output so the linear crossfade on the
            // overlapped frame (2 samples, alpha = 1/3 and 2/3) is verifiable.
            const float bias = (call_idx == 1) ? 3.0f : 0.0f;
            ++call_idx;

            // Emit 2 samples per frame based on bin 0 and bin 1 of each frame.
            // chunk_in layout is [2 bins, 3 frames]:
            // frame f: (chunk_in[f], chunk_in[3 + f])
            std::vector<float> pcm = {
                chunk_in[0] + bias, chunk_in[3] + bias, chunk_in[1] + bias,
                chunk_in[4] + bias, chunk_in[2] + bias, chunk_in[5] + bias,
            };
            auto& out_buf = const_cast<TensorBuffer&>(outputs[0]);
            LITERT_RETURN_IF_ERROR(
                out_buf.Write<float>(absl::MakeConstSpan(pcm)));
            return absl::OkStatus();
          });

  ASSERT_OK_AND_ASSIGN(auto vocoder_stage,
                       MiniCpmOVocoderStage::Create(&tts_stage, config,
                                                    std::move(vocoder_runner)));

  // 4 total frames across 2 bins in bin-major [2, 4]:
  // bin 0: {1.0, 2.0, 3.0, 4.0}
  // bin 1: {10.0, 20.0, 30.0, 40.0}
  tts_stage.Push(MiniCpmOTtsOutput{
      .text = "Audio",
      .has_text = true,
      .is_text_final = true,
      .mel_features = {1.0f, 2.0f, 3.0f, 4.0f, 10.0f, 20.0f, 30.0f, 40.0f},
      .num_mel_bins = 2,
      .num_frames = 4,
  });

  ASSERT_OK(vocoder_stage->Schedule());

  ASSERT_EQ(captured_chunks.size(), 2);
  // Chunk 0 (frames 0..2): bin 0 {1, 2, 3}, bin 1 {10, 20, 30}.
  EXPECT_THAT(captured_chunks[0],
              ElementsAre(1.0f, 2.0f, 3.0f, 10.0f, 20.0f, 30.0f));
  // Chunk 1 starts at frame 2 (1-frame overlap with chunk 0), covers valid
  // frames 2..3, and pads slot 2 by repeating the final valid frame (frame 3)
  // rather than zero-padding: bin 0 {3, 4, 4}, bin 1 {30, 40, 40}.
  EXPECT_THAT(captured_chunks[1],
              ElementsAre(3.0f, 4.0f, 4.0f, 30.0f, 40.0f, 40.0f));

  ASSERT_OK_AND_ASSIGN(Output text_out, vocoder_stage->GetOutput());
  ASSERT_TRUE(std::holds_alternative<TextOutput>(text_out));
  EXPECT_EQ(std::get<TextOutput>(text_out).confirmed_text, "Audio");

  ASSERT_OK_AND_ASSIGN(Output audio_out, vocoder_stage->GetOutput());
  ASSERT_TRUE(std::holds_alternative<AudioOutput>(audio_out));
  const auto& audio = std::get<AudioOutput>(audio_out);
  EXPECT_EQ(audio.sample_rate_hz, 24000);
  // 4 valid frames * 2 samples/frame = 8 valid PCM samples.
  // Frame 2 (samples 4 and 5) is linearly crossfaded between chunk 0 ({3, 30})
  // and chunk 1 ({6, 33}) with alpha = 1/3 and 2/3 -> {4.0, 32.0}.
  // Frame 3 (samples 6 and 7) comes from chunk 1 ({4 + 3, 40 + 3} = {7, 43}).
  EXPECT_THAT(audio.pcm_samples,
              ElementsAre(FloatNear(1.0f, 1e-5f), FloatNear(10.0f, 1e-5f),
                          FloatNear(2.0f, 1e-5f), FloatNear(20.0f, 1e-5f),
                          FloatNear(4.0f, 1e-5f), FloatNear(32.0f, 1e-5f),
                          FloatNear(7.0f, 1e-5f), FloatNear(43.0f, 1e-5f)));

  ASSERT_OK_AND_ASSIGN(Output eot_out, vocoder_stage->GetOutput());
  EXPECT_TRUE(std::holds_alternative<EndOfOutputTurn>(eot_out));
}

TEST(MiniCpmOVocoderStageTest,
     FlushPreservesTextAudioAndEndOfOutputTurnWhenBuffered) {
  MiniCpmOModelConfig config;
  config.generate_audio = true;
  config.tts_num_mel_bins = 2;
  config.audio_output_sample_rate_hz = 24000;
  FakeTtsStage tts_stage;

  auto vocoder_runner = std::make_unique<NiceMock<MockLiteRtRunner>>();
  ON_CALL(*vocoder_runner, CreateInputBuffers(absl::string_view("")))
      .WillByDefault(
          [](absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
            LITERT_ASSIGN_OR_RETURN(
                auto in_buf, support::CreateTensorBuffer<float>({1, 2, 1}));
            std::vector<TensorBuffer> bufs;
            bufs.push_back(std::move(in_buf));
            return bufs;
          });
  ON_CALL(*vocoder_runner, CreateOutputBuffers(absl::string_view("")))
      .WillByDefault(
          [](absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
            LITERT_ASSIGN_OR_RETURN(auto out_buf,
                                    support::CreateTensorBuffer<float>({1, 2}));
            std::vector<TensorBuffer> bufs;
            bufs.push_back(std::move(out_buf));
            return bufs;
          });
  EXPECT_CALL(*vocoder_runner, Run(absl::string_view(""), _, _))
      .WillOnce([](absl::string_view, absl::Span<const TensorBuffer> /*inputs*/,
                   absl::Span<const TensorBuffer> outputs) -> absl::Status {
        std::vector<float> pcm = {0.25f, -0.25f};
        auto& out_buf = const_cast<TensorBuffer&>(outputs[0]);
        LITERT_RETURN_IF_ERROR(out_buf.Write<float>(absl::MakeConstSpan(pcm)));
        return absl::OkStatus();
      });

  ASSERT_OK_AND_ASSIGN(auto vocoder_stage,
                       MiniCpmOVocoderStage::Create(&tts_stage, config,
                                                    std::move(vocoder_runner)));

  tts_stage.Push(MiniCpmOTtsOutput{
      .text = "Hello",
      .has_text = true,
      .is_text_final = true,
      .mel_features = {1.0f, 2.0f},
      .num_mel_bins = 2,
      .num_frames = 1,
  });

  ASSERT_OK(vocoder_stage->Flush());

  ASSERT_OK_AND_ASSIGN(Output text_out, vocoder_stage->GetOutput());
  ASSERT_TRUE(std::holds_alternative<TextOutput>(text_out));
  EXPECT_EQ(std::get<TextOutput>(text_out).confirmed_text, "Hello");

  ASSERT_OK_AND_ASSIGN(Output audio_out, vocoder_stage->GetOutput());
  ASSERT_TRUE(std::holds_alternative<AudioOutput>(audio_out));
  EXPECT_THAT(std::get<AudioOutput>(audio_out).pcm_samples,
              ElementsAre(0.25f, -0.25f));

  ASSERT_OK_AND_ASSIGN(Output eot_out, vocoder_stage->GetOutput());
  EXPECT_TRUE(std::holds_alternative<EndOfOutputTurn>(eot_out));
  EXPECT_FALSE(vocoder_stage->HasOutput());
}

TEST(MiniCpmOVocoderStageTest, PropagatesVocoderRunnerError) {
  MiniCpmOModelConfig config;
  config.generate_audio = true;
  config.tts_num_mel_bins = 2;
  FakeTtsStage tts_stage;

  auto vocoder_runner = std::make_unique<NiceMock<MockLiteRtRunner>>();
  ON_CALL(*vocoder_runner, CreateInputBuffers(absl::string_view("")))
      .WillByDefault(
          [](absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
            LITERT_ASSIGN_OR_RETURN(
                auto in_buf, support::CreateTensorBuffer<float>({1, 2, 1}));
            std::vector<TensorBuffer> bufs;
            bufs.push_back(std::move(in_buf));
            return bufs;
          });
  ON_CALL(*vocoder_runner, CreateOutputBuffers(absl::string_view("")))
      .WillByDefault(
          [](absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
            LITERT_ASSIGN_OR_RETURN(auto out_buf,
                                    support::CreateTensorBuffer<float>({1, 2}));
            std::vector<TensorBuffer> bufs;
            bufs.push_back(std::move(out_buf));
            return bufs;
          });
  EXPECT_CALL(*vocoder_runner, Run(absl::string_view(""), _, _))
      .WillOnce(Return(absl::InternalError("vocoder execution failed")));

  ASSERT_OK_AND_ASSIGN(auto vocoder_stage,
                       MiniCpmOVocoderStage::Create(&tts_stage, config,
                                                    std::move(vocoder_runner)));

  tts_stage.Push(MiniCpmOTtsOutput{
      .mel_features = {1.0f, 2.0f},
      .num_mel_bins = 2,
      .num_frames = 1,
  });

  EXPECT_THAT(vocoder_stage->Schedule(), StatusIs(absl::StatusCode::kInternal));
}

}  // namespace
}  // namespace litert::omni::any2any
