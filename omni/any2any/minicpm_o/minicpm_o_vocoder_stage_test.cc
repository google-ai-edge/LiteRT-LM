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

#include <cstdint>
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

TEST(MiniCpmOVocoderStageTest,
     CreateRejectsVocoderWithEmptyOrMismatchedBuffers) {
  MiniCpmOModelConfig config;
  config.generate_audio = true;
  FakeTtsStage tts_stage;

  MiniCpmOModelConfig invalid_chunk_config = config;
  invalid_chunk_config.vocoder_chunk_tokens = 0;
  EXPECT_THAT(MiniCpmOVocoderStage::Create(&tts_stage, invalid_chunk_config,
                                           /*vocoder_runner=*/nullptr)
                  .status(),
              StatusIs(absl::StatusCode::kInvalidArgument));

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

  // Mismatched input buffer size only (valid output buffer size).
  auto mismatched_in_runner = std::make_unique<NiceMock<MockLiteRtRunner>>();
  ON_CALL(*mismatched_in_runner, CreateInputBuffers(absl::string_view("")))
      .WillByDefault(
          [](absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
            LITERT_ASSIGN_OR_RETURN(
                auto in_buf, support::CreateTensorBuffer<int32_t>({1, 4}));
            std::vector<TensorBuffer> bufs;
            bufs.push_back(std::move(in_buf));
            return bufs;
          });
  ON_CALL(*mismatched_in_runner, CreateOutputBuffers(absl::string_view("")))
      .WillByDefault(
          [](absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
            LITERT_ASSIGN_OR_RETURN(
                auto out_buf,
                support::CreateTensorBuffer<float>(
                    {1, minicpm_o::kVocoderOutputSamples}));
            std::vector<TensorBuffer> bufs;
            bufs.push_back(std::move(out_buf));
            return bufs;
          });

  EXPECT_THAT(MiniCpmOVocoderStage::Create(&tts_stage, config,
                                           std::move(mismatched_in_runner))
                  .status(),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // Mismatched output buffer size only (valid input buffer size).
  auto mismatched_out_runner = std::make_unique<NiceMock<MockLiteRtRunner>>();
  ON_CALL(*mismatched_out_runner, CreateInputBuffers(absl::string_view("")))
      .WillByDefault(
          [](absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
            LITERT_ASSIGN_OR_RETURN(
                auto in_buf,
                support::CreateTensorBuffer<int32_t>(
                    {1, minicpm_o::kVocoderInputTokens}));
            std::vector<TensorBuffer> bufs;
            bufs.push_back(std::move(in_buf));
            return bufs;
          });
  ON_CALL(*mismatched_out_runner, CreateOutputBuffers(absl::string_view("")))
      .WillByDefault(
          [](absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
            LITERT_ASSIGN_OR_RETURN(auto out_buf,
                                    support::CreateTensorBuffer<float>({1, 6}));
            std::vector<TensorBuffer> bufs;
            bufs.push_back(std::move(out_buf));
            return bufs;
          });

  EXPECT_THAT(MiniCpmOVocoderStage::Create(&tts_stage, config,
                                           std::move(mismatched_out_runner))
                  .status(),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(MiniCpmOVocoderStageTest,
     EmitsTextOutputAndEndOfOutputTurnWhenNoAudioCodes) {
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
     SynthesizesWaveformWithHistoryLookaheadSilencePaddingAndCrossfade) {
  MiniCpmOModelConfig config;
  config.generate_audio = true;
  config.vocoder_history_tokens = 1;
  config.vocoder_chunk_tokens = 2;
  config.vocoder_lookahead_tokens = 1;
  config.vocoder_samples_per_token = 2;
  config.audio_output_sample_rate_hz = 24000;
  FakeTtsStage tts_stage;

  // Input buffer: int32[1, 4] (1 history + 2 chunk + 1 lookahead).
  // Output buffer: float[1, 6] ((1 history + 2 chunk) * 2 samples/token).
  auto vocoder_runner = std::make_unique<NiceMock<MockLiteRtRunner>>();
  ON_CALL(*vocoder_runner, CreateInputBuffers(absl::string_view("")))
      .WillByDefault(
          [](absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
            LITERT_ASSIGN_OR_RETURN(
                auto in_buf, support::CreateTensorBuffer<int32_t>({1, 4}));
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

  std::vector<std::vector<int32_t>> captured_windows;
  int call_idx = 0;
  EXPECT_CALL(*vocoder_runner, Run(absl::string_view(""), _, _))
      .Times(2)
      .WillRepeatedly(
          [&captured_windows, &call_idx](
              absl::string_view, absl::Span<const TensorBuffer> inputs,
              absl::Span<const TensorBuffer> outputs) -> absl::Status {
            std::vector<int32_t> window_in(4, 0);
            auto& in_buf = const_cast<TensorBuffer&>(inputs[0]);
            LITERT_RETURN_IF_ERROR(
                in_buf.Read<int32_t>(absl::MakeSpan(window_in)));
            captured_windows.push_back(window_in);

            // Add +3.0f to chunk 1's output so the linear crossfade on the
            // 1-token history overlap (2 samples, alpha = 1/3 and 2/3) is
            // verifiable.
            const float bias = (call_idx == 1) ? 3.0f : 0.0f;
            ++call_idx;

            // Emit 2 samples for each of the first 3 tokens (history + chunk).
            std::vector<float> pcm = {
                static_cast<float>(window_in[0]) + bias,
                static_cast<float>(window_in[0] * 10) + bias,
                static_cast<float>(window_in[1]) + bias,
                static_cast<float>(window_in[1] * 10) + bias,
                static_cast<float>(window_in[2]) + bias,
                static_cast<float>(window_in[2] * 10) + bias,
            };
            auto& out_buf = const_cast<TensorBuffer&>(outputs[0]);
            LITERT_RETURN_IF_ERROR(
                out_buf.Write<float>(absl::MakeConstSpan(pcm)));
            return absl::OkStatus();
          });

  ASSERT_OK_AND_ASSIGN(auto vocoder_stage,
                       MiniCpmOVocoderStage::Create(&tts_stage, config,
                                                    std::move(vocoder_runner)));

  // 3 audio codes: {1, 2, 3}.
  // Chunk 0 (emitted=0, valid=2): history={4218}, chunk={1, 2}, lookahead={3}.
  // Chunk 1 (emitted=2, valid=1): history={2}, chunk={3, 4218},
  // lookahead={4218}.
  tts_stage.Push(MiniCpmOTtsOutput{
      .text = "Audio",
      .has_text = true,
      .is_text_final = true,
      .audio_codes = {1, 2, 3},
  });

  ASSERT_OK(vocoder_stage->Schedule());

  ASSERT_EQ(captured_windows.size(), 2);
  EXPECT_THAT(captured_windows[0],
              ElementsAre(minicpm_o::kTtsSilenceTokenId, 1, 2, 3));
  EXPECT_THAT(captured_windows[1],
              ElementsAre(2, 3, minicpm_o::kTtsSilenceTokenId,
                          minicpm_o::kTtsSilenceTokenId));

  ASSERT_OK_AND_ASSIGN(Output text_out, vocoder_stage->GetOutput());
  ASSERT_TRUE(std::holds_alternative<TextOutput>(text_out));
  EXPECT_EQ(std::get<TextOutput>(text_out).confirmed_text, "Audio");

  ASSERT_OK_AND_ASSIGN(Output audio_out, vocoder_stage->GetOutput());
  ASSERT_TRUE(std::holds_alternative<AudioOutput>(audio_out));
  const auto& audio = std::get<AudioOutput>(audio_out);
  EXPECT_EQ(audio.sample_rate_hz, 24000);
  // 3 valid tokens * 2 samples/token = 6 valid PCM samples.
  // Token 0: {1.0, 10.0}.
  // Token 1 (crossfaded between chunk 0's {2.0, 20.0} and chunk 1's history
  // {5.0, 23.0} with alpha = 1/3 and 2/3): {3.0, 22.0}.
  // Token 2 (from chunk 1): {3.0 + 3.0, 30.0 + 3.0} = {6.0, 33.0}.
  EXPECT_THAT(audio.pcm_samples,
              ElementsAre(FloatNear(1.0f, 1e-5f), FloatNear(10.0f, 1e-5f),
                          FloatNear(3.0f, 1e-5f), FloatNear(22.0f, 1e-5f),
                          FloatNear(6.0f, 1e-5f), FloatNear(33.0f, 1e-5f)));

  ASSERT_OK_AND_ASSIGN(Output eot_out, vocoder_stage->GetOutput());
  EXPECT_TRUE(std::holds_alternative<EndOfOutputTurn>(eot_out));
}

TEST(MiniCpmOVocoderStageTest,
     FlushPreservesTextAudioAndEndOfOutputTurnWhenBuffered) {
  MiniCpmOModelConfig config;
  config.generate_audio = true;
  config.vocoder_history_tokens = 0;
  config.vocoder_chunk_tokens = 1;
  config.vocoder_lookahead_tokens = 0;
  config.vocoder_samples_per_token = 2;
  config.audio_output_sample_rate_hz = 24000;
  FakeTtsStage tts_stage;

  auto vocoder_runner = std::make_unique<NiceMock<MockLiteRtRunner>>();
  ON_CALL(*vocoder_runner, CreateInputBuffers(absl::string_view("")))
      .WillByDefault(
          [](absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
            LITERT_ASSIGN_OR_RETURN(
                auto in_buf, support::CreateTensorBuffer<int32_t>({1, 1}));
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
      .audio_codes = {100},
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
  config.vocoder_history_tokens = 0;
  config.vocoder_chunk_tokens = 1;
  config.vocoder_lookahead_tokens = 0;
  config.vocoder_samples_per_token = 2;
  FakeTtsStage tts_stage;

  auto vocoder_runner = std::make_unique<NiceMock<MockLiteRtRunner>>();
  ON_CALL(*vocoder_runner, CreateInputBuffers(absl::string_view("")))
      .WillByDefault(
          [](absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
            LITERT_ASSIGN_OR_RETURN(
                auto in_buf, support::CreateTensorBuffer<int32_t>({1, 1}));
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
      .audio_codes = {42},
  });

  EXPECT_THAT(vocoder_stage->Schedule(), StatusIs(absl::StatusCode::kInternal));
}

TEST(MiniCpmOVocoderStageTest, CapsSeamCrossfadeTo48SamplesOnLongHistory) {
  MiniCpmOModelConfig config;
  config.generate_audio = true;
  config.vocoder_history_tokens = 1;
  config.vocoder_chunk_tokens = 1;
  config.vocoder_lookahead_tokens = 0;
  config.vocoder_samples_per_token = 64;
  config.audio_output_sample_rate_hz = 24000;
  FakeTtsStage tts_stage;

  auto vocoder_runner = std::make_unique<NiceMock<MockLiteRtRunner>>();
  ON_CALL(*vocoder_runner, CreateInputBuffers(absl::string_view("")))
      .WillByDefault(
          [](absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
            LITERT_ASSIGN_OR_RETURN(
                auto in_buf, support::CreateTensorBuffer<int32_t>({1, 2}));
            std::vector<TensorBuffer> bufs;
            bufs.push_back(std::move(in_buf));
            return bufs;
          });
  ON_CALL(*vocoder_runner, CreateOutputBuffers(absl::string_view("")))
      .WillByDefault(
          [](absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
            LITERT_ASSIGN_OR_RETURN(
                auto out_buf, support::CreateTensorBuffer<float>({1, 128}));
            std::vector<TensorBuffer> bufs;
            bufs.push_back(std::move(out_buf));
            return bufs;
          });

  int call_idx = 0;
  EXPECT_CALL(*vocoder_runner, Run(absl::string_view(""), _, _))
      .Times(2)
      .WillRepeatedly([&call_idx](absl::string_view,
                                  absl::Span<const TensorBuffer> /*inputs*/,
                                  absl::Span<const TensorBuffer> outputs)
                          -> absl::Status {
        const float val = (call_idx == 0) ? 1.0f : 2.0f;
        ++call_idx;
        std::vector<float> pcm(128, val);
        auto& out_buf = const_cast<TensorBuffer&>(outputs[0]);
        LITERT_RETURN_IF_ERROR(out_buf.Write<float>(absl::MakeConstSpan(pcm)));
        return absl::OkStatus();
      });

  ASSERT_OK_AND_ASSIGN(auto vocoder_stage,
                       MiniCpmOVocoderStage::Create(&tts_stage, config,
                                                    std::move(vocoder_runner)));

  tts_stage.Push(MiniCpmOTtsOutput{
      .is_text_final = true,
      .audio_codes = {10, 20},
  });

  ASSERT_OK(vocoder_stage->Schedule());
  ASSERT_OK_AND_ASSIGN(Output audio_out, vocoder_stage->GetOutput());
  ASSERT_TRUE(std::holds_alternative<AudioOutput>(audio_out));
  const auto& pcm = std::get<AudioOutput>(audio_out).pcm_samples;
  ASSERT_EQ(pcm.size(), 128);
  // First 16 samples (64 - 48) of chunk 0 must remain untouched (1.0f), while
  // samples 16..63 are crossfaded between 1.0f and 2.0f.
  for (int i = 0; i < 16; ++i) {
    EXPECT_FLOAT_EQ(pcm[i], 1.0f);
  }
  EXPECT_GT(pcm[16], 1.0f);
  EXPECT_LT(pcm[63], 2.0f);
  EXPECT_FLOAT_EQ(pcm[64], 2.0f);
}

}  // namespace
}  // namespace litert::omni::any2any
