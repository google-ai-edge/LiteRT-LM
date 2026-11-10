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
      .WillByDefault([llm_hidden_size](absl::string_view)
                         -> absl::StatusOr<std::vector<TensorBuffer>> {
        LITERT_ASSIGN_OR_RETURN(
            auto hidden_buf,
            support::CreateTensorBuffer<float>({1, 1, llm_hidden_size}));
        LITERT_ASSIGN_OR_RETURN(auto token_buf,
                                support::CreateTensorBuffer<int32_t>({1, 1}));
        std::vector<TensorBuffer> bufs;
        bufs.push_back(std::move(hidden_buf));
        bufs.push_back(std::move(token_buf));
        return bufs;
      });
  ON_CALL(*runner, CreateOutputBuffers(absl::string_view("")))
      .WillByDefault([tts_hidden_size](absl::string_view)
                         -> absl::StatusOr<std::vector<TensorBuffer>> {
        LITERT_ASSIGN_OR_RETURN(
            auto out_buf,
            support::CreateTensorBuffer<float>({1, 1, tts_hidden_size}));
        std::vector<TensorBuffer> bufs;
        bufs.push_back(std::move(out_buf));
        return bufs;
      });
  return runner;
}

std::unique_ptr<NiceMock<MockLiteRtRunner>> MakeValidDecoderRunner(
    int tts_hidden_size, int tts_vocab_size, int tts_num_layers,
    int tts_max_cache_len) {
  auto runner = std::make_unique<NiceMock<MockLiteRtRunner>>();
  ON_CALL(*runner, CreateInputBuffers(absl::string_view("")))
      .WillByDefault(
          [tts_hidden_size, tts_num_layers, tts_max_cache_len](
              absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
            LITERT_ASSIGN_OR_RETURN(
                auto cond_buf,
                support::CreateTensorBuffer<float>({1, 1, tts_hidden_size}));
            LITERT_ASSIGN_OR_RETURN(
                auto code_buf, support::CreateTensorBuffer<int32_t>({1, 1}));
            LITERT_ASSIGN_OR_RETURN(auto step_buf,
                                    support::CreateTensorBuffer<int32_t>({1}));
            LITERT_ASSIGN_OR_RETURN(
                auto kv_buf, support::CreateTensorBuffer<float>(
                                 {1, 2 * tts_num_layers, tts_max_cache_len,
                                  tts_hidden_size}));
            std::vector<TensorBuffer> bufs;
            bufs.push_back(std::move(cond_buf));
            bufs.push_back(std::move(code_buf));
            bufs.push_back(std::move(step_buf));
            bufs.push_back(std::move(kv_buf));
            return bufs;
          });
  ON_CALL(*runner, CreateOutputBuffers(absl::string_view("")))
      .WillByDefault(
          [tts_hidden_size, tts_vocab_size, tts_num_layers, tts_max_cache_len](
              absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
            LITERT_ASSIGN_OR_RETURN(
                auto logits_buf,
                support::CreateTensorBuffer<float>({1, 1, tts_vocab_size}));
            LITERT_ASSIGN_OR_RETURN(
                auto kv_buf, support::CreateTensorBuffer<float>(
                                 {1, 2 * tts_num_layers, tts_max_cache_len,
                                  tts_hidden_size}));
            std::vector<TensorBuffer> bufs;
            bufs.push_back(std::move(logits_buf));
            bufs.push_back(std::move(kv_buf));
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
  config.tts_num_layers = 1;
  config.tts_max_cache_len = 8;
  FakeLlmStage llm_stage;

  EXPECT_THAT(MiniCpmOTtsStage::Create(
                  &llm_stage, config,
                  /*tts_projector_runner=*/nullptr,
                  MakeValidDecoderRunner(2, config.tts_vocab_size, 1, 8))
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
  config.tts_num_layers = 1;
  config.tts_max_cache_len = 8;
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
  EXPECT_THAT(MiniCpmOTtsStage::Create(
                  &llm_stage, config, MakeValidProjectorRunner(8, 2),
                  MakeValidDecoderRunner(2, config.tts_vocab_size, 1, 8))
                  .status(),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // Wrong projector output dimension only (6 instead of 2).
  EXPECT_THAT(MiniCpmOTtsStage::Create(
                  &llm_stage, config, MakeValidProjectorRunner(4, 6),
                  MakeValidDecoderRunner(2, config.tts_vocab_size, 1, 8))
                  .status(),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // Wrong projector token_id buffer size only ({1, 2} instead of {1, 1}).
  auto bad_token_proj = std::make_unique<NiceMock<MockLiteRtRunner>>();
  ON_CALL(*bad_token_proj, CreateInputBuffers(absl::string_view("")))
      .WillByDefault(
          [](absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
            LITERT_ASSIGN_OR_RETURN(
                auto hidden_buf, support::CreateTensorBuffer<float>({1, 1, 4}));
            LITERT_ASSIGN_OR_RETURN(
                auto token_buf, support::CreateTensorBuffer<int32_t>({1, 2}));
            std::vector<TensorBuffer> bufs;
            bufs.push_back(std::move(hidden_buf));
            bufs.push_back(std::move(token_buf));
            return bufs;
          });
  ON_CALL(*bad_token_proj, CreateOutputBuffers(absl::string_view("")))
      .WillByDefault(
          [](absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
            LITERT_ASSIGN_OR_RETURN(
                auto out_buf, support::CreateTensorBuffer<float>({1, 1, 2}));
            std::vector<TensorBuffer> bufs;
            bufs.push_back(std::move(out_buf));
            return bufs;
          });
  EXPECT_THAT(MiniCpmOTtsStage::Create(
                  &llm_stage, config, std::move(bad_token_proj),
                  MakeValidDecoderRunner(2, config.tts_vocab_size, 1, 8))
                  .status(),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // Decoder with 4 inputs but only 1 output buffer.
  auto one_output_decoder = std::make_unique<NiceMock<MockLiteRtRunner>>();
  ON_CALL(*one_output_decoder, CreateInputBuffers(absl::string_view("")))
      .WillByDefault(
          [](absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
            std::vector<TensorBuffer> bufs;
            for (int i = 0; i < 4; ++i) {
              LITERT_ASSIGN_OR_RETURN(
                  auto buf, support::CreateTensorBuffer<float>({1}));
              bufs.push_back(std::move(buf));
            }
            return bufs;
          });
  ON_CALL(*one_output_decoder, CreateOutputBuffers(absl::string_view("")))
      .WillByDefault(
          [](absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
            std::vector<TensorBuffer> bufs;
            LITERT_ASSIGN_OR_RETURN(
                auto buf, support::CreateTensorBuffer<float>({1}));
            bufs.push_back(std::move(buf));
            return bufs;
          });
  EXPECT_THAT(
      MiniCpmOTtsStage::Create(&llm_stage, config,
                               MakeValidProjectorRunner(4, 2),
                               std::move(one_output_decoder))
          .status(),
      StatusIs(absl::StatusCode::kInvalidArgument));

  // Wrong decoder KV cache length (16 instead of 8).
  EXPECT_THAT(MiniCpmOTtsStage::Create(
                  &llm_stage, config, MakeValidProjectorRunner(4, 2),
                  MakeValidDecoderRunner(2, config.tts_vocab_size, 1, 16))
                  .status(),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // Wrong decoder logits vocab size only.
  EXPECT_THAT(MiniCpmOTtsStage::Create(
                  &llm_stage, config, MakeValidProjectorRunner(4, 2),
                  MakeValidDecoderRunner(2, config.tts_vocab_size + 1, 1, 8))
                  .status(),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // Wrong decoder output KV cache size only (input KV cache size is valid).
  auto bad_kv_out_decoder = std::make_unique<NiceMock<MockLiteRtRunner>>();
  ON_CALL(*bad_kv_out_decoder, CreateInputBuffers(absl::string_view("")))
      .WillByDefault(
          [](absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
            LITERT_ASSIGN_OR_RETURN(
                auto cond_buf, support::CreateTensorBuffer<float>({1, 1, 2}));
            LITERT_ASSIGN_OR_RETURN(
                auto code_buf, support::CreateTensorBuffer<int32_t>({1, 1}));
            LITERT_ASSIGN_OR_RETURN(auto step_buf,
                                    support::CreateTensorBuffer<int32_t>({1}));
            LITERT_ASSIGN_OR_RETURN(
                auto kv_buf, support::CreateTensorBuffer<float>({1, 2, 8, 2}));
            std::vector<TensorBuffer> bufs;
            bufs.push_back(std::move(cond_buf));
            bufs.push_back(std::move(code_buf));
            bufs.push_back(std::move(step_buf));
            bufs.push_back(std::move(kv_buf));
            return bufs;
          });
  ON_CALL(*bad_kv_out_decoder, CreateOutputBuffers(absl::string_view("")))
      .WillByDefault(
          [&config](
              absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
            LITERT_ASSIGN_OR_RETURN(
                auto logits_buf, support::CreateTensorBuffer<float>(
                                     {1, 1, config.tts_vocab_size}));
            LITERT_ASSIGN_OR_RETURN(
                auto kv_buf, support::CreateTensorBuffer<float>({1, 2, 16, 2}));
            std::vector<TensorBuffer> bufs;
            bufs.push_back(std::move(logits_buf));
            bufs.push_back(std::move(kv_buf));
            return bufs;
          });
  EXPECT_THAT(MiniCpmOTtsStage::Create(&llm_stage, config,
                                       MakeValidProjectorRunner(4, 2),
                                       std::move(bad_kv_out_decoder))
                  .status(),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(MiniCpmOTtsStageTest,
     PassesThroughTextWithoutRunningDecoderWhenAudioGenerationDisabled) {
  MiniCpmOModelConfig config;
  config.generate_audio = false;
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
  EXPECT_THAT(out.audio_codes, IsEmpty());
}

TEST(MiniCpmOTtsStageTest,
     ProjectsTokensWithControlFramingAndDecodesWithKvCacheFeedback) {
  MiniCpmOModelConfig config;
  config.generate_audio = true;
  config.do_sample = false;
  config.llm_hidden_size = 4;
  config.tts_hidden_size = 2;
  config.tts_vocab_size = minicpm_o::kTtsVocabSize;
  config.tts_num_layers = 1;
  config.tts_max_cache_len = 8;
  config.max_tts_tokens = 4;
  FakeLlmStage llm_stage;

  auto projector_runner = MakeValidProjectorRunner(4, 2);
  std::vector<int32_t> captured_proj_tokens;
  EXPECT_CALL(*projector_runner, Run(absl::string_view(""), _, _))
      .Times(4)
      .WillRepeatedly(
          [&](absl::string_view, absl::Span<const TensorBuffer> inputs,
              absl::Span<const TensorBuffer> outputs) -> absl::Status {
            std::vector<float> in_vec(4, 0.0f);
            int32_t token_id = -1;
            LITERT_RETURN_IF_ERROR(
                const_cast<TensorBuffer&>(inputs[0]).Read<float>(
                    absl::MakeSpan(in_vec)));
            LITERT_RETURN_IF_ERROR(
                const_cast<TensorBuffer&>(inputs[1]).Read<int32_t>(
                    absl::MakeSpan(&token_id, 1)));
            captured_proj_tokens.push_back(token_id);

            std::vector<float> out_vec = {
                in_vec[0] + in_vec[1] + static_cast<float>(token_id),
                in_vec[2] + in_vec[3]};
            LITERT_RETURN_IF_ERROR(
                const_cast<TensorBuffer&>(outputs[0])
                    .Write<float>(absl::MakeConstSpan(out_vec)));
            return absl::OkStatus();
          });

  auto decoder_runner = MakeValidDecoderRunner(2, config.tts_vocab_size, 1, 8);
  std::vector<std::vector<float>> captured_conds;
  std::vector<int32_t> captured_prev_codes;
  std::vector<int32_t> captured_steps;
  EXPECT_CALL(*decoder_runner, Run(absl::string_view(""), _, _))
      .Times(6)
      .WillRepeatedly([&](absl::string_view,
                          absl::Span<const TensorBuffer> inputs,
                          absl::Span<const TensorBuffer> outputs)
                          -> absl::Status {
        std::vector<float> cond(2, 0.0f);
        int32_t prev_code = -999;
        int32_t step_idx = -1;
        std::vector<float> kv_cache(2 * 8 * 2, 0.0f);
        LITERT_RETURN_IF_ERROR(const_cast<TensorBuffer&>(inputs[0]).Read<float>(
            absl::MakeSpan(cond)));
        LITERT_RETURN_IF_ERROR(
            const_cast<TensorBuffer&>(inputs[1]).Read<int32_t>(
                absl::MakeSpan(&prev_code, 1)));
        LITERT_RETURN_IF_ERROR(
            const_cast<TensorBuffer&>(inputs[2]).Read<int32_t>(
                absl::MakeSpan(&step_idx, 1)));
        LITERT_RETURN_IF_ERROR(const_cast<TensorBuffer&>(inputs[3]).Read<float>(
            absl::MakeSpan(kv_cache)));
        captured_conds.push_back(cond);
        captured_prev_codes.push_back(prev_code);
        captured_steps.push_back(step_idx);

        // Update kv_cache[step_idx] so we can verify KV cache propagation.
        kv_cache[step_idx] = static_cast<float>(step_idx + 1);

        std::vector<float> logits(config.tts_vocab_size, -10.0f);
        if (step_idx == 3 && prev_code == -1) {
          // End of conditioning prefill (<|audio_bos|>): predict first code 7.
          logits[7] = 10.0f;
        } else if (step_idx == 4 && prev_code == 7) {
          EXPECT_FLOAT_EQ(kv_cache[3], 4.0f);
          logits[9] = 10.0f;
        } else if (step_idx == 5 && prev_code == 9) {
          EXPECT_FLOAT_EQ(kv_cache[4], 5.0f);
          logits[minicpm_o::kTtsEosTokenId] = 10.0f;
        }
        LITERT_RETURN_IF_ERROR(
            const_cast<TensorBuffer&>(outputs[0])
                .Write<float>(absl::MakeConstSpan(logits)));
        LITERT_RETURN_IF_ERROR(
            const_cast<TensorBuffer&>(outputs[1])
                .Write<float>(absl::MakeConstSpan(kv_cache)));
        return absl::OkStatus();
      });

  ASSERT_OK_AND_ASSIGN(
      auto tts_stage,
      MiniCpmOTtsStage::Create(&llm_stage, config, std::move(projector_runner),
                               std::move(decoder_runner)));

  llm_stage.Push(MiniCpmOLlmOutput{
      .text = "Hi",
      .generated_token_ids = {10, 20},
      .llm_hidden_states = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
      .num_tokens = 2,
      .generate_audio = true,
      .is_final = true,
  });

  ASSERT_OK(tts_stage->Schedule());
  ASSERT_OK_AND_ASSIGN(MiniCpmOTtsOutput out, tts_stage->GetOutput());

  EXPECT_EQ(out.text, "Hi");
  EXPECT_THAT(out.audio_codes, ElementsAre(7, 9));
  EXPECT_THAT(captured_proj_tokens,
              ElementsAre(10, 20, minicpm_o::kTtsTextEosTokenId,
                          minicpm_o::kTtsAudioBosTokenId));

  ASSERT_EQ(captured_conds.size(), 6);
  EXPECT_THAT(captured_conds[0], ElementsAre(13.0f, 7.0f));
  EXPECT_THAT(captured_conds[1], ElementsAre(31.0f, 15.0f));
  EXPECT_THAT(
      captured_conds[2],
      ElementsAre(static_cast<float>(minicpm_o::kTtsTextEosTokenId), 0.0f));
  EXPECT_THAT(
      captured_conds[3],
      ElementsAre(static_cast<float>(minicpm_o::kTtsAudioBosTokenId), 0.0f));
  EXPECT_THAT(captured_conds[4], ElementsAre(0.0f, 0.0f));
  EXPECT_THAT(captured_conds[5], ElementsAre(0.0f, 0.0f));
  EXPECT_THAT(captured_prev_codes, ElementsAre(-1, -1, -1, -1, 7, 9));
  EXPECT_THAT(captured_steps, ElementsAre(0, 1, 2, 3, 4, 5));
}

TEST(MiniCpmOTtsStageTest,
     SlidingRecomputeReplaysPreviousChunkContextAcrossMultiChunkText) {
  MiniCpmOModelConfig config;
  config.generate_audio = true;
  config.do_sample = false;
  config.llm_hidden_size = 2;
  config.tts_hidden_size = 2;
  config.tts_vocab_size = minicpm_o::kTtsVocabSize;
  config.tts_num_layers = 1;
  config.tts_max_cache_len = 16;
  config.tts_text_chunk_size = 1;
  config.max_tts_tokens = 4;
  FakeLlmStage llm_stage;

  auto projector_runner = MakeValidProjectorRunner(2, 2);
  EXPECT_CALL(*projector_runner, Run(absl::string_view(""), _, _))
      .Times(5)
      .WillRepeatedly(
          [](absl::string_view, absl::Span<const TensorBuffer> inputs,
             absl::Span<const TensorBuffer> outputs) -> absl::Status {
            int32_t token_id = 0;
            LITERT_RETURN_IF_ERROR(
                const_cast<TensorBuffer&>(inputs[1]).Read<int32_t>(
                    absl::MakeSpan(&token_id, 1)));
            std::vector<float> out_vec = {static_cast<float>(token_id), 1.0f};
            LITERT_RETURN_IF_ERROR(
                const_cast<TensorBuffer&>(outputs[0])
                    .Write<float>(absl::MakeConstSpan(out_vec)));
            return absl::OkStatus();
          });

  auto decoder_runner = MakeValidDecoderRunner(2, config.tts_vocab_size, 1, 16);
  std::vector<int32_t> captured_steps;
  std::vector<int32_t> captured_prev_codes;
  int call_count = 0;
  EXPECT_CALL(*decoder_runner, Run(absl::string_view(""), _, _))
      .WillRepeatedly(
          [&](absl::string_view, absl::Span<const TensorBuffer> inputs,
              absl::Span<const TensorBuffer> outputs) -> absl::Status {
            int32_t prev_code = -999;
            int32_t step_idx = -1;
            LITERT_RETURN_IF_ERROR(
                const_cast<TensorBuffer&>(inputs[1]).Read<int32_t>(
                    absl::MakeSpan(&prev_code, 1)));
            LITERT_RETURN_IF_ERROR(
                const_cast<TensorBuffer&>(inputs[2]).Read<int32_t>(
                    absl::MakeSpan(&step_idx, 1)));
            captured_steps.push_back(step_idx);
            captured_prev_codes.push_back(prev_code);

            std::vector<float> logits(config.tts_vocab_size, -10.0f);
            // Chunk 0: 2 cond steps (token 0, audio_bos) -> emits code 11 on
            // call 1
            // -> EOS on call 2.
            // Chunk 1: replays chunk 0 (2 cond steps + 1 code step = 3 steps) +
            // 3 cond steps (token 1, text_eos, audio_bos) -> emits code 22 on
            // call 8 -> EOS on call 9.
            if (call_count == 1) {
              logits[11] = 10.0f;
            } else if (call_count == 2) {
              logits[minicpm_o::kTtsEosTokenId] = 10.0f;
            } else if (call_count == 8) {
              logits[22] = 10.0f;
            } else if (call_count == 9) {
              logits[minicpm_o::kTtsEosTokenId] = 10.0f;
            }
            ++call_count;

            LITERT_RETURN_IF_ERROR(
                const_cast<TensorBuffer&>(outputs[0])
                    .Write<float>(absl::MakeConstSpan(logits)));
            return absl::OkStatus();
          });

  ASSERT_OK_AND_ASSIGN(
      auto tts_stage,
      MiniCpmOTtsStage::Create(&llm_stage, config, std::move(projector_runner),
                               std::move(decoder_runner)));

  llm_stage.Push(MiniCpmOLlmOutput{
      .text = "Two chunks",
      .generated_token_ids = {101, 102},
      .llm_hidden_states = {1.0f, 1.0f, 2.0f, 2.0f},
      .num_tokens = 2,
      .generate_audio = true,
      .is_final = true,
  });

  ASSERT_OK(tts_stage->Schedule());
  ASSERT_OK_AND_ASSIGN(MiniCpmOTtsOutput out, tts_stage->GetOutput());
  EXPECT_THAT(out.audio_codes, ElementsAre(11, 22));
  // Chunk 0 steps: 0, 1, 2. Chunk 1 steps (with sliding_recompute): 0..6.
  EXPECT_THAT(captured_steps, ElementsAre(0, 1, 2, 0, 1, 2, 3, 4, 5, 6));
  EXPECT_THAT(captured_prev_codes,
              ElementsAre(-1, -1, 11, -1, -1, 11, -1, -1, -1, 22));
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
  config.tts_num_layers = 1;
  config.tts_max_cache_len = 8;
  config.max_tts_tokens = 2;
  FakeLlmStage llm_stage;

  auto projector_runner = MakeValidProjectorRunner(2, 2);
  EXPECT_CALL(*projector_runner, Run(absl::string_view(""), _, _))
      .Times(3)
      .WillRepeatedly(Return(absl::OkStatus()));

  auto decoder_runner = MakeValidDecoderRunner(2, config.tts_vocab_size, 1, 8);
  EXPECT_CALL(*decoder_runner, Run(absl::string_view(""), _, _))
      .Times(4)
      .WillRepeatedly([&](absl::string_view,
                          absl::Span<const TensorBuffer> inputs,
                          absl::Span<const TensorBuffer> outputs)
                          -> absl::Status {
        int32_t step_idx = 0;
        LITERT_RETURN_IF_ERROR(
            const_cast<TensorBuffer&>(inputs[2]).Read<int32_t>(
                absl::MakeSpan(&step_idx, 1)));
        std::vector<float> logits(config.tts_vocab_size, -20.0f);
        if (step_idx == 2) {
          logits[42] = 10.0f;
          logits[43] = 5.0f;
        } else if (step_idx == 3) {
          logits[minicpm_o::kTtsEosTokenId] = 10.0f;
        }
        LITERT_RETURN_IF_ERROR(
            const_cast<TensorBuffer&>(outputs[0])
                .Write<float>(absl::MakeConstSpan(logits)));
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
  tts_stage->Reset();
}

TEST(MiniCpmOTtsStageTest, AppliesFrequencyBasedRepetitionPenaltyOverWindow) {
  MiniCpmOModelConfig config;
  config.generate_audio = true;
  config.do_sample = false;
  config.tts_repetition_penalty = 2.0f;
  config.tts_repetition_window = 4;
  config.llm_hidden_size = 2;
  config.tts_hidden_size = 2;
  config.tts_vocab_size = minicpm_o::kTtsVocabSize;
  config.tts_num_layers = 1;
  config.tts_max_cache_len = 16;
  config.tts_text_chunk_size = 1;
  config.max_tts_tokens = 8;
  FakeLlmStage llm_stage;

  auto projector_runner = MakeValidProjectorRunner(2, 2);
  EXPECT_CALL(*projector_runner, Run(absl::string_view(""), _, _))
      .Times(5)
      .WillRepeatedly(Return(absl::OkStatus()));

  auto decoder_runner = MakeValidDecoderRunner(2, config.tts_vocab_size, 1, 16);
  int call_count = 0;
  EXPECT_CALL(*decoder_runner, Run(absl::string_view(""), _, _))
      .WillRepeatedly([&](absl::string_view,
                          absl::Span<const TensorBuffer> inputs,
                          absl::Span<const TensorBuffer> outputs)
                          -> absl::Status {
        int32_t step_idx = 0;
        LITERT_RETURN_IF_ERROR(
            const_cast<TensorBuffer&>(inputs[2]).Read<int32_t>(
                absl::MakeSpan(&step_idx, 1)));
        std::vector<float> logits(config.tts_vocab_size, -20.0f);
        if (call_count < 4) {
          // Chunk 0 (2 cond steps + 2 code steps = calls 0..3):
          // Code 5 starts at 12.0 and EOS at 4.0.
          // Call 1 (step 1, freq=0): 12.0 > 4.0 -> emits 5
          // Call 2 (step 2, freq=1): 12.0 / 2^1 = 6.0 > 4.0 -> emits 5
          // Call 3 (step 3, freq=2): 12.0 / 2^2 = 3.0 < 4.0 -> emits EOS!
          if (step_idx >= 1) {
            logits[5] = 12.0f;
            logits[minicpm_o::kTtsEosTokenId] = 4.0f;
          }
        } else {
          // Chunk 1 (replays 4 steps of chunk 0 + 3 cond steps = calls 4..10):
          // On call 10 (step 6), `out.audio_codes` already contains `{5, 5}`
          // from chunk 0, so code 5 is penalized across chunk boundaries:
          // Call 10 (step 6, freq(5)=2): 12.0 / 2^2 = 3.0 < 3.5 -> emits 6!
          // Call 11 (step 7, freq(5)=2, freq(6)=1): 3.0 and 1.75 < 3.2 -> EOS!
          if (step_idx >= 6) {
            logits[5] = 12.0f;
            logits[6] = 3.5f;
            logits[minicpm_o::kTtsEosTokenId] = 3.2f;
          }
        }
        ++call_count;
        LITERT_RETURN_IF_ERROR(
            const_cast<TensorBuffer&>(outputs[0])
                .Write<float>(absl::MakeConstSpan(logits)));
        return absl::OkStatus();
      });

  ASSERT_OK_AND_ASSIGN(
      auto tts_stage,
      MiniCpmOTtsStage::Create(&llm_stage, config, std::move(projector_runner),
                               std::move(decoder_runner)));

  llm_stage.Push(MiniCpmOLlmOutput{
      .text = "Repeat across chunks",
      .llm_hidden_states = {1.0f, 2.0f, 3.0f, 4.0f},
      .num_tokens = 2,
      .generate_audio = true,
      .is_final = true,
  });

  ASSERT_OK(tts_stage->Schedule());
  ASSERT_OK_AND_ASSIGN(MiniCpmOTtsOutput out, tts_stage->GetOutput());
  EXPECT_THAT(out.audio_codes, ElementsAre(5, 5, 6));

  // Verify that `tts_repetition_penalty <= 1.0f` and `tts_repetition_window <=
  // 0` disable repetition penalty rather than scaling logits.
  auto expect_disabled_penalty = [&](float penalty, int window,
                                     std::vector<int> expected_codes) {
    MiniCpmOModelConfig disabled_config = config;
    disabled_config.tts_repetition_penalty = penalty;
    disabled_config.tts_repetition_window = window;
    disabled_config.max_tts_tokens = 2;
    auto proj = MakeValidProjectorRunner(2, 2);
    ON_CALL(*proj, Run(absl::string_view(""), _, _))
        .WillByDefault(Return(absl::OkStatus()));
    auto dec =
        MakeValidDecoderRunner(2, disabled_config.tts_vocab_size, 1, 16);
    ON_CALL(*dec, Run(absl::string_view(""), _, _))
        .WillByDefault([&](absl::string_view,
                           absl::Span<const TensorBuffer> inputs,
                           absl::Span<const TensorBuffer> outputs)
                           -> absl::Status {
          int32_t step_idx = 0;
          LITERT_RETURN_IF_ERROR(
              const_cast<TensorBuffer&>(inputs[2]).Read<int32_t>(
                  absl::MakeSpan(&step_idx, 1)));
          std::vector<float> logits(disabled_config.tts_vocab_size, -20.0f);
          if (step_idx == 2) {
            logits[5] = 12.0f;
          } else if (step_idx == 3) {
            // If `penalty = 0.5f` were applied (`12.0 / 0.5 = 24.0 > 15.0`), it
            // would emit 5 instead of 8; if `window = -1` were cast to `size_t`
            // and `penalty = 2.0f` applied (`16.0 / 2.0 = 8.0 < 15.0`), it
            // would emit 8 instead of 5.
            logits[5] = (penalty < 1.0f) ? 12.0f : 16.0f;
            logits[8] = 15.0f;
          }
          LITERT_RETURN_IF_ERROR(
              const_cast<TensorBuffer&>(outputs[0])
                  .Write<float>(absl::MakeConstSpan(logits)));
          return absl::OkStatus();
        });
    ASSERT_OK_AND_ASSIGN(
        auto stage,
        MiniCpmOTtsStage::Create(&llm_stage, disabled_config, std::move(proj),
                                 std::move(dec)));
    llm_stage.Push(MiniCpmOLlmOutput{
        .text = "DisabledPenalty",
        .llm_hidden_states = {1.0f, 2.0f},
        .num_tokens = 1,
        .generate_audio = true,
        .is_final = true,
    });
    ASSERT_OK(stage->Schedule());
    ASSERT_OK_AND_ASSIGN(MiniCpmOTtsOutput res, stage->GetOutput());
    EXPECT_EQ(res.audio_codes, expected_codes);
  };

  expect_disabled_penalty(0.5f, 4, {5, 8});
  expect_disabled_penalty(2.0f, -1, {5, 5});
}

TEST(MiniCpmOTtsStageTest, PropagatesErrorOnHiddenStateDimensionMismatch) {
  MiniCpmOModelConfig config;
  config.generate_audio = true;
  config.llm_hidden_size = 4;
  config.tts_hidden_size = 2;
  config.tts_vocab_size = minicpm_o::kTtsVocabSize;
  config.tts_num_layers = 1;
  config.tts_max_cache_len = 8;
  FakeLlmStage llm_stage;

  ASSERT_OK_AND_ASSIGN(
      auto tts_stage,
      MiniCpmOTtsStage::Create(
          &llm_stage, config, MakeValidProjectorRunner(4, 2),
          MakeValidDecoderRunner(2, config.tts_vocab_size, 1, 8)));

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

  // num_tokens=2 with 8 floats, but generated_token_ids has only 1 element.
  llm_stage.Push(MiniCpmOLlmOutput{
      .text = "TokenCountMismatch",
      .generated_token_ids = {10},
      .llm_hidden_states = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
      .num_tokens = 2,
      .generate_audio = true,
      .is_final = true,
  });

  EXPECT_THAT(tts_stage->Schedule(),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // Conditioning length (7 tokens + 2 control tokens = 9 steps) exceeds
  // `tts_max_cache_len = 8`.
  llm_stage.Push(MiniCpmOLlmOutput{
      .text = "ExceedsCacheLen",
      .llm_hidden_states = std::vector<float>(7 * 4, 1.0f),
      .num_tokens = 7,
      .generate_audio = true,
      .is_final = true,
  });

  EXPECT_THAT(tts_stage->Schedule(),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(MiniCpmOTtsStageTest, PropagatesDecoderRunnerErrors) {
  MiniCpmOModelConfig config;
  config.generate_audio = true;
  config.llm_hidden_size = 2;
  config.tts_hidden_size = 2;
  config.tts_vocab_size = minicpm_o::kTtsVocabSize;
  config.tts_num_layers = 1;
  config.tts_max_cache_len = 8;
  FakeLlmStage llm_stage;

  auto projector_runner = MakeValidProjectorRunner(2, 2);
  EXPECT_CALL(*projector_runner, Run(absl::string_view(""), _, _))
      .Times(3)
      .WillRepeatedly(Return(absl::OkStatus()));

  auto decoder_runner = MakeValidDecoderRunner(2, config.tts_vocab_size, 1, 8);
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

TEST(MiniCpmOTtsStageTest,
     SkipsLeadingPostThinkNewlineAndAlignsPostTokenHiddenStates) {
  MiniCpmOModelConfig config;
  config.generate_audio = true;
  config.do_sample = false;
  config.llm_hidden_size = 2;
  config.tts_hidden_size = 2;
  config.tts_vocab_size = minicpm_o::kTtsVocabSize;
  config.tts_num_layers = 1;
  config.tts_max_cache_len = 16;
  config.max_tts_tokens = 2;
  FakeLlmStage llm_stage;

  auto projector_runner = MakeValidProjectorRunner(2, 2);
  std::vector<int32_t> captured_proj_tokens;
  std::vector<std::vector<float>> captured_proj_hiddens;
  EXPECT_CALL(*projector_runner, Run(absl::string_view(""), _, _))
      .Times(4)
      .WillRepeatedly(
          [&](absl::string_view, absl::Span<const TensorBuffer> inputs,
              absl::Span<const TensorBuffer> outputs) -> absl::Status {
            std::vector<float> in_vec(2, 0.0f);
            int32_t token_id = -1;
            LITERT_RETURN_IF_ERROR(
                const_cast<TensorBuffer&>(inputs[0]).Read<float>(
                    absl::MakeSpan(in_vec)));
            LITERT_RETURN_IF_ERROR(
                const_cast<TensorBuffer&>(inputs[1]).Read<int32_t>(
                    absl::MakeSpan(&token_id, 1)));
            captured_proj_hiddens.push_back(in_vec);
            captured_proj_tokens.push_back(token_id);

            LITERT_RETURN_IF_ERROR(
                const_cast<TensorBuffer&>(outputs[0])
                    .Write<float>(absl::MakeConstSpan(in_vec)));
            return absl::OkStatus();
          });

  auto decoder_runner = MakeValidDecoderRunner(2, config.tts_vocab_size, 1, 16);
  EXPECT_CALL(*decoder_runner, Run(absl::string_view(""), _, _))
      .WillRepeatedly([&](absl::string_view,
                          absl::Span<const TensorBuffer> inputs,
                          absl::Span<const TensorBuffer> outputs)
                          -> absl::Status {
        int32_t step_idx = 0;
        LITERT_RETURN_IF_ERROR(
            const_cast<TensorBuffer&>(inputs[2]).Read<int32_t>(
                absl::MakeSpan(&step_idx, 1)));
        std::vector<float> logits(config.tts_vocab_size, -20.0f);
        if (step_idx == 3) {
          logits[17] = 10.0f;
        } else if (step_idx >= 4) {
          logits[minicpm_o::kTtsEosTokenId] = 10.0f;
        }
        LITERT_RETURN_IF_ERROR(
            const_cast<TensorBuffer&>(outputs[0])
                .Write<float>(absl::MakeConstSpan(logits)));
        return absl::OkStatus();
      });

  ASSERT_OK_AND_ASSIGN(
      auto tts_stage,
      MiniCpmOTtsStage::Create(&llm_stage, config, std::move(projector_runner),
                               std::move(decoder_runner)));

  // Token 0 is 271 ("\n\n" after "</think>"), followed by spoken tokens 785
  // ("The") and 2168 (" image").
  llm_stage.Push(MiniCpmOLlmOutput{
      .text = "The image",
      .generated_token_ids = {minicpm_o::kQwenDoubleNewlineTokenId, 785, 2168},
      .llm_hidden_states = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
      .num_tokens = 3,
      .generate_audio = true,
      .is_final = true,
  });

  ASSERT_OK(tts_stage->Schedule());
  ASSERT_OK_AND_ASSIGN(MiniCpmOTtsOutput out, tts_stage->GetOutput());
  EXPECT_THAT(out.audio_codes, ElementsAre(17));
  EXPECT_THAT(captured_proj_tokens,
              ElementsAre(785, 2168, minicpm_o::kTtsTextEosTokenId,
                          minicpm_o::kTtsAudioBosTokenId));
  ASSERT_EQ(captured_proj_hiddens.size(), 4);
  EXPECT_THAT(captured_proj_hiddens[0], ElementsAre(5.0f, 6.0f));
  EXPECT_THAT(captured_proj_hiddens[1], ElementsAre(5.0f, 6.0f));
}

TEST(MiniCpmOTtsStageTest,
     NonFinalChunkWithoutEosReservesKvCacheHeadroomForNextChunk) {
  MiniCpmOModelConfig config;
  config.generate_audio = true;
  config.do_sample = false;
  config.tts_repetition_penalty = 1.0f;
  config.llm_hidden_size = 2;
  config.tts_hidden_size = 2;
  config.tts_vocab_size = minicpm_o::kTtsVocabSize;
  config.tts_num_layers = 1;
  // Use `tts_max_cache_len = 64` (>= 64) with `tts_text_chunk_size = 1` (< 4)
  // to verify both safe clamping when `chunk_size < 4` and KV-cache headroom
  // reservation on a non-final chunk that never samples EOS.
  config.tts_max_cache_len = 64;
  config.tts_text_chunk_size = 1;
  config.max_tts_tokens = 40;
  FakeLlmStage llm_stage;

  auto projector_runner = MakeValidProjectorRunner(2, 2);
  EXPECT_CALL(*projector_runner, Run(absl::string_view(""), _, _))
      .Times(5)
      .WillRepeatedly(
          [](absl::string_view, absl::Span<const TensorBuffer> inputs,
             absl::Span<const TensorBuffer> outputs) -> absl::Status {
            int32_t token_id = 0;
            LITERT_RETURN_IF_ERROR(
                const_cast<TensorBuffer&>(inputs[1]).Read<int32_t>(
                    absl::MakeSpan(&token_id, 1)));
            std::vector<float> out_vec = {static_cast<float>(token_id), 1.0f};
            LITERT_RETURN_IF_ERROR(
                const_cast<TensorBuffer&>(outputs[0])
                    .Write<float>(absl::MakeConstSpan(out_vec)));
            return absl::OkStatus();
          });

  auto decoder_runner = MakeValidDecoderRunner(2, config.tts_vocab_size, 1, 64);
  std::vector<float> chunk1_cond_first_elements;
  int chunk_idx = 0;
  int last_step_idx = -1;
  EXPECT_CALL(*decoder_runner, Run(absl::string_view(""), _, _))
      .WillRepeatedly([&](absl::string_view,
                          absl::Span<const TensorBuffer> inputs,
                          absl::Span<const TensorBuffer> outputs)
                          -> absl::Status {
        std::vector<float> cond(2, 0.0f);
        int32_t prev_code = -999;
        int32_t step_idx = -1;
        LITERT_RETURN_IF_ERROR(const_cast<TensorBuffer&>(inputs[0]).Read<float>(
            absl::MakeSpan(cond)));
        LITERT_RETURN_IF_ERROR(
            const_cast<TensorBuffer&>(inputs[1]).Read<int32_t>(
                absl::MakeSpan(&prev_code, 1)));
        LITERT_RETURN_IF_ERROR(
            const_cast<TensorBuffer&>(inputs[2]).Read<int32_t>(
                absl::MakeSpan(&step_idx, 1)));
        if (step_idx == 0 && last_step_idx >= 0) {
          ++chunk_idx;
        }
        last_step_idx = step_idx;

        std::vector<float> logits(config.tts_vocab_size, -20.0f);
        if (chunk_idx == 0) {
          // Chunk 0 never emits EOS; always prefers code 15.
          // With `max_cache_len = 64`, `reserved_next_chunk_steps = 32`, and
          // `cur_cond_len = 2`, chunk 0 is capped at `64 - 32 - 2 = 30` codes.
          logits[15] = 10.0f;
        } else {
          // Chunk 1 replays 32 steps of chunk 0 (`step_idx = 0..31`), then
          // feeds chunk 1's 3 conditioning steps (`step_idx = 32, 33, 34`).
          if (step_idx >= 32 && prev_code == -1) {
            chunk1_cond_first_elements.push_back(cond[0]);
          }
          if (step_idx == 34 && prev_code == -1) {
            logits[99] = 10.0f;
          } else if (step_idx == 35 && prev_code == 99) {
            logits[minicpm_o::kTtsEosTokenId] = 10.0f;
          } else {
            logits[15] = 10.0f;
          }
        }
        LITERT_RETURN_IF_ERROR(
            const_cast<TensorBuffer&>(outputs[0])
                .Write<float>(absl::MakeConstSpan(logits)));
        return absl::OkStatus();
      });

  ASSERT_OK_AND_ASSIGN(
      auto tts_stage,
      MiniCpmOTtsStage::Create(&llm_stage, config, std::move(projector_runner),
                               std::move(decoder_runner)));

  llm_stage.Push(MiniCpmOLlmOutput{
      .text = "No EOS on chunk 0",
      .generated_token_ids = {501, 502},
      .llm_hidden_states = {1.0f, 1.0f, 2.0f, 2.0f},
      .num_tokens = 2,
      .generate_audio = true,
      .is_final = true,
  });

  ASSERT_OK(tts_stage->Schedule());
  ASSERT_OK_AND_ASSIGN(MiniCpmOTtsOutput out, tts_stage->GetOutput());
  EXPECT_THAT(
      chunk1_cond_first_elements,
      ElementsAre(502.0f, static_cast<float>(minicpm_o::kTtsTextEosTokenId),
                  static_cast<float>(minicpm_o::kTtsAudioBosTokenId)));
  ASSERT_EQ(out.audio_codes.size(), 31);
  for (int i = 0; i < 30; ++i) {
    EXPECT_EQ(out.audio_codes[i], 15);
  }
  EXPECT_EQ(out.audio_codes[30], 99);
}

}  // namespace
}  // namespace litert::omni::any2any
