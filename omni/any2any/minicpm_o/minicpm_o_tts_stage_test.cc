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

#include "omni/any2any/minicpm_o/minicpm_o_tts_stage.h"

#include <cstdint>
#include <memory>
#include <utility>
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
#include "omni/any2any/minicpm_o/minicpm_o_llm_stage.h"
#include "omni/any2any/minicpm_o/minicpm_o_model_config.h"
#include "omni/base/mock_litert_runner.h"
#include "omni/base/stage.h"
#include "support/util/convert_tensor_buffer.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace litert::omni::any2any {
namespace {

using ::absl_testing::StatusIs;
using ::testing::_;
using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::NiceMock;
using ::testing::Return;

class FakeLlmStage : public SingleThreadedStageWithDeque<MiniCpmOLlmOutput> {
 public:
  void Push(MiniCpmOLlmOutput out) { PushOutput(std::move(out)); }

 protected:
  bool NeedScheduleInternal() const override { return false; }
  absl::Status ScheduleInternal() override { return absl::OkStatus(); }
};

std::unique_ptr<NiceMock<MockLiteRtRunner>> MakeValidProjectorRunner(
    int llm_hidden_size, int tts_hidden_size) {
  auto runner = std::make_unique<NiceMock<MockLiteRtRunner>>();
  ON_CALL(*runner, CreateInputBuffers(absl::string_view("")))
      .WillByDefault(
          [llm_hidden_size](
              absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
            LITERT_ASSIGN_OR_RETURN(
                auto in_buf,
                support::CreateTensorBuffer<float>({1, llm_hidden_size}));
            std::vector<TensorBuffer> bufs;
            bufs.push_back(std::move(in_buf));
            return bufs;
          });
  ON_CALL(*runner, CreateOutputBuffers(absl::string_view("")))
      .WillByDefault(
          [tts_hidden_size](
              absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
            LITERT_ASSIGN_OR_RETURN(
                auto out_buf,
                support::CreateTensorBuffer<float>({1, tts_hidden_size}));
            std::vector<TensorBuffer> bufs;
            bufs.push_back(std::move(out_buf));
            return bufs;
          });
  return runner;
}

std::unique_ptr<NiceMock<MockLiteRtRunner>> MakeValidDecoderRunner(
    int tts_hidden_size, int tts_vocab_size, int tts_num_mel_bins) {
  auto runner = std::make_unique<NiceMock<MockLiteRtRunner>>();
  ON_CALL(*runner, CreateInputBuffers(absl::string_view("")))
      .WillByDefault(
          [tts_hidden_size](
              absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
            LITERT_ASSIGN_OR_RETURN(
                auto cond_buf,
                support::CreateTensorBuffer<float>({1, tts_hidden_size}));
            LITERT_ASSIGN_OR_RETURN(
                auto code_buf, support::CreateTensorBuffer<int32_t>({1, 1}));
            LITERT_ASSIGN_OR_RETURN(
                auto step_buf, support::CreateTensorBuffer<int32_t>({1, 1}));
            std::vector<TensorBuffer> bufs;
            bufs.push_back(std::move(cond_buf));
            bufs.push_back(std::move(code_buf));
            bufs.push_back(std::move(step_buf));
            return bufs;
          });
  ON_CALL(*runner, CreateOutputBuffers(absl::string_view("")))
      .WillByDefault(
          [tts_vocab_size, tts_num_mel_bins](
              absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
            LITERT_ASSIGN_OR_RETURN(
                auto logits_buf,
                support::CreateTensorBuffer<float>({1, tts_vocab_size}));
            LITERT_ASSIGN_OR_RETURN(
                auto mel_buf,
                support::CreateTensorBuffer<float>({1, tts_num_mel_bins}));
            std::vector<TensorBuffer> bufs;
            bufs.push_back(std::move(logits_buf));
            bufs.push_back(std::move(mel_buf));
            return bufs;
          });
  return runner;
}

TEST(MiniCpmOTtsStageTest,
     CreateRejectsMissingProjectorOrDecoderWhenAudioGenerationEnabled) {
  MiniCpmOModelConfig config;
  config.generate_audio = true;
  config.llm_hidden_size = 4;
  config.tts_hidden_size = 2;
  config.tts_vocab_size = minicpm_o::kTtsVocabSize;
  config.tts_num_mel_bins = 2;
  FakeLlmStage llm_stage;

  EXPECT_THAT(
      MiniCpmOTtsStage::Create(&llm_stage, config,
                               /*tts_projector_runner=*/nullptr,
                               MakeValidDecoderRunner(2, config.tts_vocab_size,
                                                      2))
          .status(),
      StatusIs(absl::StatusCode::kNotFound));

  EXPECT_THAT(
      MiniCpmOTtsStage::Create(&llm_stage, config,
                               MakeValidProjectorRunner(4, 2),
                               /*tts_decoder_runner=*/nullptr)
          .status(),
      StatusIs(absl::StatusCode::kNotFound));
}

TEST(MiniCpmOTtsStageTest, CreateRejectsEmptyOrMismatchedBuffers) {
  MiniCpmOModelConfig config;
  config.generate_audio = true;
  config.llm_hidden_size = 4;
  config.tts_hidden_size = 2;
  config.tts_vocab_size = minicpm_o::kTtsVocabSize;
  config.tts_num_mel_bins = 2;
  FakeLlmStage llm_stage;

  auto empty_decoder = std::make_unique<NiceMock<MockLiteRtRunner>>();
  ON_CALL(*empty_decoder, CreateInputBuffers(absl::string_view("")))
      .WillByDefault([](absl::string_view)
                         -> absl::StatusOr<std::vector<TensorBuffer>> {
        return std::vector<TensorBuffer>{};
      });
  ON_CALL(*empty_decoder, CreateOutputBuffers(absl::string_view("")))
      .WillByDefault([](absl::string_view)
                         -> absl::StatusOr<std::vector<TensorBuffer>> {
        return std::vector<TensorBuffer>{};
      });

  EXPECT_THAT(
      MiniCpmOTtsStage::Create(&llm_stage, config,
                               MakeValidProjectorRunner(4, 2),
                               std::move(empty_decoder))
          .status(),
      StatusIs(absl::StatusCode::kInvalidArgument));

  // Wrong projector input dimension (8 instead of 4).
  EXPECT_THAT(
      MiniCpmOTtsStage::Create(&llm_stage, config,
                               MakeValidProjectorRunner(8, 2),
                               MakeValidDecoderRunner(2, config.tts_vocab_size,
                                                      2))
          .status(),
      StatusIs(absl::StatusCode::kInvalidArgument));

  // Wrong decoder mel dimension (4 instead of 2).
  EXPECT_THAT(
      MiniCpmOTtsStage::Create(&llm_stage, config,
                               MakeValidProjectorRunner(4, 2),
                               MakeValidDecoderRunner(2, config.tts_vocab_size,
                                                      4))
          .status(),
      StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(MiniCpmOTtsStageTest,
     PassesThroughTextWithoutRunningDecoderWhenAudioGenerationDisabled) {
  MiniCpmOModelConfig config;
  config.generate_audio = false;
  config.tts_num_mel_bins = 2;
  FakeLlmStage llm_stage;

  ASSERT_OK_AND_ASSIGN(
      auto tts_stage,
      MiniCpmOTtsStage::Create(&llm_stage, config,
                               /*tts_projector_runner=*/nullptr,
                               /*tts_decoder_runner=*/nullptr));

  llm_stage.Push(MiniCpmOLlmOutput{
      .text = "Hello world",
      .llm_hidden_states = {1.0f, 2.0f, 3.0f, 4.0f},
      .num_tokens = 2,
      .generate_audio = true,
      .is_final = true,
  });

  ASSERT_OK(tts_stage->Schedule());
  ASSERT_TRUE(tts_stage->HasOutput());
  ASSERT_OK_AND_ASSIGN(MiniCpmOTtsOutput out, tts_stage->GetOutput());
  EXPECT_EQ(out.text, "Hello world");
  EXPECT_TRUE(out.has_text);
  EXPECT_TRUE(out.is_text_final);
  EXPECT_EQ(out.num_frames, 0);
  EXPECT_THAT(out.mel_features, IsEmpty());
  EXPECT_THAT(out.audio_codes, IsEmpty());
}

TEST(MiniCpmOTtsStageTest,
     ProjectsAndDecodesAutoregressivelyWithPrevCodeAndStepFeedback) {
  MiniCpmOModelConfig config;
  config.generate_audio = true;
  config.do_sample = false;
  config.llm_hidden_size = 4;
  config.tts_hidden_size = 2;
  config.tts_vocab_size = minicpm_o::kTtsVocabSize;
  config.tts_num_mel_bins = 2;
  config.max_tts_tokens = 4;
  FakeLlmStage llm_stage;

  auto projector_runner = MakeValidProjectorRunner(4, 2);
  EXPECT_CALL(*projector_runner, Run(absl::string_view(""), _, _))
      .Times(2)
      .WillRepeatedly([](absl::string_view,
                         absl::Span<const TensorBuffer> inputs,
                         absl::Span<const TensorBuffer> outputs)
                          -> absl::Status {
        std::vector<float> in_vec(4, 0.0f);
        auto& in_buf = const_cast<TensorBuffer&>(inputs[0]);
        LITERT_RETURN_IF_ERROR(in_buf.Read<float>(absl::MakeSpan(in_vec)));
        std::vector<float> out_vec = {in_vec[0] + in_vec[1],
                                      in_vec[2] + in_vec[3]};
        auto& out_buf = const_cast<TensorBuffer&>(outputs[0]);
        LITERT_RETURN_IF_ERROR(
            out_buf.Write<float>(absl::MakeConstSpan(out_vec)));
        return absl::OkStatus();
      });

  auto decoder_runner = MakeValidDecoderRunner(2, config.tts_vocab_size, 2);
  std::vector<std::vector<float>> captured_conds;
  std::vector<int32_t> captured_prev_codes;
  std::vector<int32_t> captured_steps;
  EXPECT_CALL(*decoder_runner, Run(absl::string_view(""), _, _))
      .Times(3)
      .WillRepeatedly([&](absl::string_view,
                          absl::Span<const TensorBuffer> inputs,
                          absl::Span<const TensorBuffer> outputs)
                          -> absl::Status {
        std::vector<float> cond(2, 0.0f);
        int32_t prev_code = -1;
        int32_t step_idx = -1;
        LITERT_RETURN_IF_ERROR(const_cast<TensorBuffer&>(inputs[0]).Read<float>(
            absl::MakeSpan(cond)));
        LITERT_RETURN_IF_ERROR(
            const_cast<TensorBuffer&>(inputs[1]).Read<int32_t>(
                absl::MakeSpan(&prev_code, 1)));
        LITERT_RETURN_IF_ERROR(
            const_cast<TensorBuffer&>(inputs[2]).Read<int32_t>(
                absl::MakeSpan(&step_idx, 1)));
        captured_conds.push_back(cond);
        captured_prev_codes.push_back(prev_code);
        captured_steps.push_back(step_idx);

        // Transition purely based on fed-back prev_code and step_idx.
        std::vector<float> logits(config.tts_vocab_size, -10.0f);
        std::vector<float> mel(2, 0.0f);
        if (prev_code == minicpm_o::kTtsBosTokenId && step_idx == 0) {
          logits[7] = 10.0f;
          mel = {10.0f, 20.0f};
        } else if (prev_code == 7 && step_idx == 1) {
          logits[9] = 10.0f;
          mel = {11.0f, 21.0f};
        } else if (prev_code == 9 && step_idx == 2) {
          logits[minicpm_o::kTtsEosTokenId] = 10.0f;
        }
        LITERT_RETURN_IF_ERROR(
            const_cast<TensorBuffer&>(outputs[0])
                .Write<float>(absl::MakeConstSpan(logits)));
        LITERT_RETURN_IF_ERROR(const_cast<TensorBuffer&>(outputs[1])
                                   .Write<float>(absl::MakeConstSpan(mel)));
        return absl::OkStatus();
      });

  ASSERT_OK_AND_ASSIGN(
      auto tts_stage,
      MiniCpmOTtsStage::Create(&llm_stage, config, std::move(projector_runner),
                               std::move(decoder_runner)));

  // Token 0: {1, 2, 3, 4} -> projected {3, 7}.
  // Token 1: {5, 6, 7, 8} -> projected {11, 15}.
  llm_stage.Push(MiniCpmOLlmOutput{
      .text = "Hi",
      .llm_hidden_states = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
      .num_tokens = 2,
      .generate_audio = true,
      .is_final = true,
  });

  ASSERT_OK(tts_stage->Schedule());
  ASSERT_OK_AND_ASSIGN(MiniCpmOTtsOutput out, tts_stage->GetOutput());

  EXPECT_EQ(out.text, "Hi");
  EXPECT_EQ(out.num_frames, 2);
  EXPECT_EQ(out.num_mel_bins, 2);
  EXPECT_THAT(out.audio_codes, ElementsAre(7, 9));
  // Frame 0: {10, 20}, Frame 1: {11, 21} -> Bin-major [2, 2]: {10, 11, 20, 21}.
  EXPECT_THAT(out.mel_features, ElementsAre(10.0f, 11.0f, 20.0f, 21.0f));

  ASSERT_EQ(captured_conds.size(), 3);
  EXPECT_THAT(captured_conds[0], ElementsAre(3.0f, 7.0f));
  EXPECT_THAT(captured_conds[1], ElementsAre(11.0f, 15.0f));
  EXPECT_THAT(captured_conds[2], ElementsAre(11.0f, 15.0f));
  EXPECT_THAT(captured_prev_codes,
              ElementsAre(minicpm_o::kTtsBosTokenId, 7, 9));
  EXPECT_THAT(captured_steps, ElementsAre(0, 1, 2));
}

TEST(MiniCpmOTtsStageTest, SupportsStochasticTopKTopPSamplingWithReset) {
  MiniCpmOModelConfig config;
  config.generate_audio = true;
  config.do_sample = true;
  config.tts_temperature = 0.5f;
  config.tts_top_k = 1;
  config.tts_top_p = 0.9f;
  config.seed = 123;
  config.llm_hidden_size = 2;
  config.tts_hidden_size = 2;
  config.tts_vocab_size = minicpm_o::kTtsVocabSize;
  config.tts_num_mel_bins = 2;
  config.max_tts_tokens = 2;
  FakeLlmStage llm_stage;

  auto projector_runner = MakeValidProjectorRunner(2, 2);
  EXPECT_CALL(*projector_runner, Run(absl::string_view(""), _, _))
      .WillOnce([](absl::string_view, absl::Span<const TensorBuffer> inputs,
                   absl::Span<const TensorBuffer> outputs) -> absl::Status {
        std::vector<float> in_vec(2, 0.0f);
        LITERT_RETURN_IF_ERROR(const_cast<TensorBuffer&>(inputs[0]).Read<float>(
            absl::MakeSpan(in_vec)));
        LITERT_RETURN_IF_ERROR(
            const_cast<TensorBuffer&>(outputs[0])
                .Write<float>(absl::MakeConstSpan(in_vec)));
        return absl::OkStatus();
      });

  auto decoder_runner = MakeValidDecoderRunner(2, config.tts_vocab_size, 2);
  EXPECT_CALL(*decoder_runner, Run(absl::string_view(""), _, _))
      .Times(2)
      .WillRepeatedly([&](absl::string_view,
                          absl::Span<const TensorBuffer> inputs,
                          absl::Span<const TensorBuffer> outputs)
                          -> absl::Status {
        int32_t step_idx = 0;
        LITERT_RETURN_IF_ERROR(
            const_cast<TensorBuffer&>(inputs[2]).Read<int32_t>(
                absl::MakeSpan(&step_idx, 1)));
        std::vector<float> logits(config.tts_vocab_size, -20.0f);
        if (step_idx == 0) {
          logits[42] = 10.0f;
          logits[43] = 5.0f;
        } else {
          logits[minicpm_o::kTtsEosTokenId] = 10.0f;
        }
        std::vector<float> mel = {1.0f, 2.0f};
        LITERT_RETURN_IF_ERROR(
            const_cast<TensorBuffer&>(outputs[0])
                .Write<float>(absl::MakeConstSpan(logits)));
        LITERT_RETURN_IF_ERROR(const_cast<TensorBuffer&>(outputs[1])
                                   .Write<float>(absl::MakeConstSpan(mel)));
        return absl::OkStatus();
      });

  ASSERT_OK_AND_ASSIGN(
      auto tts_stage,
      MiniCpmOTtsStage::Create(&llm_stage, config, std::move(projector_runner),
                               std::move(decoder_runner)));

  llm_stage.Push(MiniCpmOLlmOutput{
      .text = "Sample",
      .llm_hidden_states = {1.0f, 2.0f},
      .num_tokens = 1,
      .generate_audio = true,
      .is_final = true,
  });

  ASSERT_OK(tts_stage->Schedule());
  ASSERT_OK_AND_ASSIGN(MiniCpmOTtsOutput out, tts_stage->GetOutput());
  EXPECT_THAT(out.audio_codes, ElementsAre(42));
  EXPECT_EQ(out.num_frames, 1);
  tts_stage->Reset();
}

TEST(MiniCpmOTtsStageTest, PropagatesErrorOnHiddenStateDimensionMismatch) {
  MiniCpmOModelConfig config;
  config.generate_audio = true;
  config.llm_hidden_size = 4;
  config.tts_hidden_size = 2;
  config.tts_vocab_size = minicpm_o::kTtsVocabSize;
  config.tts_num_mel_bins = 2;
  FakeLlmStage llm_stage;

  ASSERT_OK_AND_ASSIGN(
      auto tts_stage,
      MiniCpmOTtsStage::Create(
          &llm_stage, config, MakeValidProjectorRunner(4, 2),
          MakeValidDecoderRunner(2, config.tts_vocab_size, 2)));

  // num_tokens=2 expects 8 floats, but only 6 are provided.
  llm_stage.Push(MiniCpmOLlmOutput{
      .text = "Mismatch",
      .llm_hidden_states = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
      .num_tokens = 2,
      .generate_audio = true,
      .is_final = true,
  });

  EXPECT_THAT(tts_stage->Schedule(),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(MiniCpmOTtsStageTest, PropagatesDecoderRunnerErrors) {
  MiniCpmOModelConfig config;
  config.generate_audio = true;
  config.llm_hidden_size = 2;
  config.tts_hidden_size = 2;
  config.tts_vocab_size = minicpm_o::kTtsVocabSize;
  config.tts_num_mel_bins = 2;
  FakeLlmStage llm_stage;

  auto projector_runner = MakeValidProjectorRunner(2, 2);
  EXPECT_CALL(*projector_runner, Run(absl::string_view(""), _, _))
      .WillOnce(Return(absl::OkStatus()));

  auto decoder_runner = MakeValidDecoderRunner(2, config.tts_vocab_size, 2);
  EXPECT_CALL(*decoder_runner, Run(absl::string_view(""), _, _))
      .WillOnce(Return(absl::InternalError("decoder execution failed")));

  ASSERT_OK_AND_ASSIGN(
      auto tts_stage,
      MiniCpmOTtsStage::Create(&llm_stage, config, std::move(projector_runner),
                               std::move(decoder_runner)));

  llm_stage.Push(MiniCpmOLlmOutput{
      .text = "Error",
      .llm_hidden_states = {1.0f, 2.0f},
      .num_tokens = 1,
      .generate_audio = true,
      .is_final = true,
  });

  EXPECT_THAT(tts_stage->Schedule(), StatusIs(absl::StatusCode::kInternal));
}

}  // namespace
}  // namespace litert::omni::any2any
