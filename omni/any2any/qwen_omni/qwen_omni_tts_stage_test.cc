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

#include "omni/any2any/qwen_omni/qwen_omni_tts_stage.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
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
#include "omni/any2any/any2any_test_utils.h"
#include "omni/any2any/io_types.h"
#include "omni/any2any/qwen_omni/qwen_omni_model_config.h"
#include "omni/any2any/qwen_omni/qwen_omni_test_utils.h"
#include "omni/base/litert_runner.h"
#include "omni/base/mock_litert_runner.h"
#include "omni/base/model_utils.h"
#include "omni/base/stage.h"
#include "runtime/executor/executor_settings_base.h"
#include "support/util/convert_tensor_buffer.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep
#include "tflite/schema/schema_generated.h"  // from @litert

namespace litert::omni::any2any {
namespace {

using ::absl_testing::StatusIs;
using ::testing::_;
using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::NiceMock;

class FakeLlmStage : public SingleThreadedStageWithDeque<Any2AnyLlmOutput> {
 public:
  void Push(Any2AnyLlmOutput out) { PushOutput(std::move(out)); }

 protected:
  bool NeedScheduleInternal() const override { return false; }
  absl::Status ScheduleInternal() override { return absl::OkStatus(); }
};

struct DecoderCall {
  int32_t prev_audio_code = 0;
  int32_t step_idx = 0;
};

std::unique_ptr<NiceMock<MockLiteRtRunner>> MakeMockProjectorRunner(
    int llm_hidden_size, int tts_hidden_size,
    std::vector<int32_t>* recorded_token_ids = nullptr) {
  auto runner = std::make_unique<NiceMock<MockLiteRtRunner>>();
  ON_CALL(*runner, CreateInputBuffers(absl::string_view("")))
      .WillByDefault([llm_hidden_size](absl::string_view)
                         -> absl::StatusOr<std::vector<TensorBuffer>> {
        LITERT_ASSIGN_OR_RETURN(
            auto hidden_buf,
            support::CreateTensorBuffer<float>({1, 1, llm_hidden_size}));
        LITERT_ASSIGN_OR_RETURN(auto tok_buf,
                                support::CreateTensorBuffer<int32_t>({1, 1}));
        std::vector<TensorBuffer> bufs;
        bufs.push_back(std::move(hidden_buf));
        bufs.push_back(std::move(tok_buf));
        return bufs;
      });
  ON_CALL(*runner, CreateOutputBuffers(absl::string_view("")))
      .WillByDefault([tts_hidden_size](absl::string_view)
                         -> absl::StatusOr<std::vector<TensorBuffer>> {
        LITERT_ASSIGN_OR_RETURN(
            auto cond_buf,
            support::CreateTensorBuffer<float>({1, 1, tts_hidden_size}));
        std::vector<TensorBuffer> bufs;
        bufs.push_back(std::move(cond_buf));
        return bufs;
      });
  ON_CALL(*runner, Run(absl::string_view(""), _, _))
      .WillByDefault(
          [tts_hidden_size, recorded_token_ids](
              absl::string_view, absl::Span<const TensorBuffer> inputs,
              absl::Span<const TensorBuffer> outputs) -> absl::Status {
            LITERT_ASSIGN_OR_RETURN(
                std::vector<int32_t> tok_ids,
                support::CopyFromTensorBuffer<int32_t>(inputs[1]));
            const int32_t tok_id = tok_ids.empty() ? 0 : tok_ids[0];
            if (recorded_token_ids != nullptr) {
              recorded_token_ids->push_back(tok_id);
            }
            std::vector<float> cond(static_cast<size_t>(tts_hidden_size),
                                    static_cast<float>(tok_id));
            auto& out_buf = const_cast<TensorBuffer&>(outputs[0]);
            LITERT_RETURN_IF_ERROR(
                out_buf.Write<float>(absl::MakeConstSpan(cond)));
            return absl::OkStatus();
          });
  return runner;
}

std::unique_ptr<NiceMock<MockLiteRtRunner>> MakeMockDecoderRunner(
    int tts_hidden_size, int tts_vocab_size, int num_layers, int max_cache_len,
    int kv_dim,
    std::function<std::vector<float>(int32_t prev_audio_code, int32_t step_idx)>
        logits_fn,
    std::vector<DecoderCall>* recorded_calls = nullptr) {
  auto runner = std::make_unique<NiceMock<MockLiteRtRunner>>();
  ON_CALL(*runner, CreateInputBuffers(absl::string_view("")))
      .WillByDefault(
          [tts_hidden_size, num_layers, max_cache_len, kv_dim](
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
                                 {1, 2 * num_layers, max_cache_len, kv_dim}));
            std::vector<TensorBuffer> bufs;
            bufs.push_back(std::move(cond_buf));
            bufs.push_back(std::move(code_buf));
            bufs.push_back(std::move(step_buf));
            bufs.push_back(std::move(kv_buf));
            return bufs;
          });
  ON_CALL(*runner, CreateOutputBuffers(absl::string_view("")))
      .WillByDefault(
          [tts_vocab_size, num_layers, max_cache_len, kv_dim](
              absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
            LITERT_ASSIGN_OR_RETURN(
                auto logits_buf,
                support::CreateTensorBuffer<float>({1, 1, tts_vocab_size}));
            LITERT_ASSIGN_OR_RETURN(
                auto kv_buf, support::CreateTensorBuffer<float>(
                                 {1, 2 * num_layers, max_cache_len, kv_dim}));
            std::vector<TensorBuffer> bufs;
            bufs.push_back(std::move(logits_buf));
            bufs.push_back(std::move(kv_buf));
            return bufs;
          });
  ON_CALL(*runner, Run(absl::string_view(""), _, _))
      .WillByDefault(
          [logits_fn = std::move(logits_fn), recorded_calls](
              absl::string_view, absl::Span<const TensorBuffer> inputs,
              absl::Span<const TensorBuffer> outputs) -> absl::Status {
            LITERT_ASSIGN_OR_RETURN(
                std::vector<int32_t> code_vec,
                support::CopyFromTensorBuffer<int32_t>(inputs[1]));
            LITERT_ASSIGN_OR_RETURN(
                std::vector<int32_t> step_vec,
                support::CopyFromTensorBuffer<int32_t>(inputs[2]));
            const int32_t prev_code = code_vec.empty() ? 0 : code_vec[0];
            const int32_t step_idx = step_vec.empty() ? 0 : step_vec[0];
            if (recorded_calls != nullptr) {
              recorded_calls->push_back(DecoderCall{
                  .prev_audio_code = prev_code, .step_idx = step_idx});
            }
            std::vector<float> logits = logits_fn(prev_code, step_idx);
            auto& out_logits = const_cast<TensorBuffer&>(outputs[0]);
            LITERT_RETURN_IF_ERROR(
                out_logits.Write<float>(absl::MakeConstSpan(logits)));
            return absl::OkStatus();
          });
  return runner;
}

TEST(QwenOmniTtsStageTest, TtsStageSupportsGqaKvCacheAndTalkerPrefillProtocol) {
  // Arrange: GQA Talker where tts_hidden_size=8, tts_num_kv_heads=2,
  // tts_head_dim=2 -> kv_cache last dim is 4 (not 8).
  QwenOmniModelConfig config =
      MakeSmallQwenOmniTestConfig(/*generate_audio=*/true);
  config.llm_hidden_size = 4;
  config.tts_hidden_size = 8;
  config.tts_num_kv_heads = 2;
  config.tts_head_dim = 2;
  config.tts_num_layers = 1;
  config.tts_max_cache_len = 16;
  config.tts_vocab_size = 8;
  config.max_tts_tokens = 3;

  auto env = ::litert::Environment::Create({});
  ASSERT_TRUE(env.HasValue());
  ModelOptions opts{.backend = lm::Backend::CPU, .num_threads = 1};

  std::string proj_buf = BuildMinimalTfliteModelBuffer(
      {{"a_llm_hidden", {1, 1, 4}, tflite::TensorType_FLOAT32},
       {"b_token_id", {1, 1}, tflite::TensorType_INT32}},
      {{"cond_emb", {1, 1, 8}, tflite::TensorType_FLOAT32}});
  ASSERT_OK_AND_ASSIGN(
      auto proj_model,
      CreateCompiledModelFromBuffer(*env, opts, proj_buf, "tts_projector"));

  std::string dec_buf = BuildMinimalTfliteModelBuffer(
      {{"a_cond_emb", {1, 1, 8}, tflite::TensorType_FLOAT32},
       {"b_prev_audio_code", {1, 1}, tflite::TensorType_INT32},
       {"c_step_idx", {1}, tflite::TensorType_INT32},
       {"d_kv_cache", {1, 2, 16, 4}, tflite::TensorType_FLOAT32}},
      {{"code_logits", {1, 1, 8}, tflite::TensorType_FLOAT32},
       {"new_kv_cache", {1, 2, 16, 4}, tflite::TensorType_FLOAT32}});
  ASSERT_OK_AND_ASSIGN(auto dec_model, CreateCompiledModelFromBuffer(
                                           *env, opts, dec_buf, "tts_decoder"));

  FakeLlmStage llm_stage;
  ASSERT_OK_AND_ASSIGN(
      auto tts_stage,
      QwenOmniTtsStage::Create(&llm_stage, config,
                               std::make_unique<LiteRtRunnerImpl>(&proj_model),
                               std::make_unique<LiteRtRunnerImpl>(&dec_model)));

  // Act: push a 2-token prompt + 2-token reply LLM chunk and end-of-turn.
  llm_stage.Push(Any2AnyLlmOutput{
      .text = "Hi",
      .prompt_token_ids = {1, 3},
      .prompt_hidden_states = std::vector<float>(2 * 4, 0.25f),
      .generated_token_ids = {10, 11},
      .llm_hidden_states = std::vector<float>(2 * 4, 0.5f),
      .num_tokens = 2,
      .generate_audio = true,
      .is_final = true,
  });
  ASSERT_OK(tts_stage->Schedule());

  // Assert
  ASSERT_OK_AND_ASSIGN(Any2AnyTtsOutput out, tts_stage->GetOutput());
  EXPECT_TRUE(out.has_text);
  EXPECT_EQ(out.text, "Hi");
  EXPECT_EQ(out.audio_codes.size(), 3);
  EXPECT_TRUE(out.is_text_final);
}

TEST(QwenOmniTtsStageTest, TtsStageRejectsMissingModelsWhenAudioEnabled) {
  QwenOmniModelConfig config =
      MakeSmallQwenOmniTestConfig(/*generate_audio=*/true);
  FakeLlmStage llm_stage;

  // Missing projector runner.
  EXPECT_THAT(QwenOmniTtsStage::Create(&llm_stage, config).status(),
              StatusIs(absl::StatusCode::kNotFound));

  // Projector present, decoder missing.
  auto proj =
      MakeMockProjectorRunner(config.llm_hidden_size, config.tts_hidden_size);
  EXPECT_THAT(
      QwenOmniTtsStage::Create(&llm_stage, config, std::move(proj), nullptr)
          .status(),
      StatusIs(absl::StatusCode::kNotFound));
}

TEST(QwenOmniTtsStageTest, RejectsInvalidConfigDimensionsAndBufferShapes) {
  FakeLlmStage llm_stage;
  {
    QwenOmniModelConfig cfg = MakeSmallQwenOmniTestConfig(false);
    cfg.llm_hidden_size = 0;
    EXPECT_THAT(QwenOmniTtsStage::Create(&llm_stage, cfg).status(),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }
  {
    QwenOmniModelConfig cfg = MakeSmallQwenOmniTestConfig(false);
    cfg.tts_max_cache_len = 0;
    EXPECT_THAT(QwenOmniTtsStage::Create(&llm_stage, cfg).status(),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }

  QwenOmniModelConfig config =
      MakeSmallQwenOmniTestConfig(/*generate_audio=*/true);

  // Projector wrong buffer count.
  {
    auto bad_proj = std::make_unique<NiceMock<MockLiteRtRunner>>();
    ON_CALL(*bad_proj, CreateInputBuffers(absl::string_view("")))
        .WillByDefault(
            [](absl::string_view) { return std::vector<TensorBuffer>{}; });
    ON_CALL(*bad_proj, CreateOutputBuffers(absl::string_view("")))
        .WillByDefault(
            [](absl::string_view) { return std::vector<TensorBuffer>{}; });
    EXPECT_THAT(
        QwenOmniTtsStage::Create(&llm_stage, config, std::move(bad_proj))
            .status(),
        StatusIs(absl::StatusCode::kInvalidArgument));
  }

  // Projector wrong hidden byte size.
  {
    auto bad_proj = MakeMockProjectorRunner(config.llm_hidden_size + 1,
                                            config.tts_hidden_size);
    EXPECT_THAT(
        QwenOmniTtsStage::Create(&llm_stage, config, std::move(bad_proj))
            .status(),
        StatusIs(absl::StatusCode::kInvalidArgument));
  }

  // Decoder wrong buffer count.
  {
    auto proj =
        MakeMockProjectorRunner(config.llm_hidden_size, config.tts_hidden_size);
    auto bad_dec = std::make_unique<NiceMock<MockLiteRtRunner>>();
    ON_CALL(*bad_dec, CreateInputBuffers(absl::string_view("")))
        .WillByDefault(
            [](absl::string_view) { return std::vector<TensorBuffer>{}; });
    ON_CALL(*bad_dec, CreateOutputBuffers(absl::string_view("")))
        .WillByDefault(
            [](absl::string_view) { return std::vector<TensorBuffer>{}; });
    EXPECT_THAT(QwenOmniTtsStage::Create(&llm_stage, config, std::move(proj),
                                         std::move(bad_dec))
                    .status(),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }

  // Decoder wrong logits size.
  {
    auto proj =
        MakeMockProjectorRunner(config.llm_hidden_size, config.tts_hidden_size);
    auto bad_dec = MakeMockDecoderRunner(
        config.tts_hidden_size, config.tts_vocab_size + 1,
        config.tts_num_layers, config.tts_max_cache_len, config.tts_hidden_size,
        [](int32_t, int32_t) { return std::vector<float>(9, 0.0f); });
    EXPECT_THAT(QwenOmniTtsStage::Create(&llm_stage, config, std::move(proj),
                                         std::move(bad_dec))
                    .status(),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }
}

TEST(QwenOmniTtsStageTest, AutoInfersKvCacheShapeFrom4DTensorType) {
  QwenOmniModelConfig config =
      MakeSmallQwenOmniTestConfig(/*generate_audio=*/true);
  config.tts_num_layers = 1;
  config.tts_max_cache_len = 8;
  config.max_tts_tokens = 20;

  // Decoder KV cache is actually 2 layers x max_cache_len=5 (total byte size
  // differs from 1 layer x max_cache_len=8), so Create() infers the dimensions
  // from the 4D tensor shape [1, 4, 5, tts_hidden_size].
  FakeLlmStage llm_stage;
  auto proj =
      MakeMockProjectorRunner(config.llm_hidden_size, config.tts_hidden_size);
  auto dec = MakeMockDecoderRunner(
      config.tts_hidden_size, config.tts_vocab_size, /*num_layers=*/2,
      /*max_cache_len=*/5, config.tts_hidden_size,
      [vocab = config.tts_vocab_size](int32_t, int32_t) {
        std::vector<float> logits(vocab, 0.0f);
        logits[1] = 2.0f;
        return logits;
      });
  ASSERT_OK_AND_ASSIGN(
      auto tts_stage, QwenOmniTtsStage::Create(
                          &llm_stage, config, std::move(proj), std::move(dec)));

  // With inferred max_cache_len=5: BOS (step 0) + first reply token (step 1) +
  // 3 decoded steps (steps 2, 3, 4) -> stops at step_idx=5 after emitting 4
  // codes.
  llm_stage.Push(Any2AnyLlmOutput{
      .text = "Ok",
      .generated_token_ids = {2},
      .llm_hidden_states = std::vector<float>(config.llm_hidden_size, 0.5f),
      .num_tokens = 1,
      .generate_audio = true,
      .is_final = true,
  });
  ASSERT_OK(tts_stage->Schedule());
  ASSERT_OK_AND_ASSIGN(Any2AnyTtsOutput out, tts_stage->GetOutput());
  EXPECT_EQ(out.audio_codes.size(), 4);
}

TEST(QwenOmniTtsStageTest,
     AllowsNullRunnersWhenAudioDisabledAndHandlesNeedSchedule) {
  FakeLlmStage llm_stage;
  QwenOmniModelConfig config =
      MakeSmallQwenOmniTestConfig(/*generate_audio=*/false);
  ASSERT_OK_AND_ASSIGN(auto tts_stage,
                       QwenOmniTtsStage::Create(&llm_stage, config));

  EXPECT_FALSE(tts_stage->NeedSchedule());
  EXPECT_OK(tts_stage->Schedule());
  EXPECT_THAT(tts_stage->GetOutput().status(),
              StatusIs(absl::StatusCode::kNotFound));

  llm_stage.Push(Any2AnyLlmOutput{
      .text = "Text only",
      .generated_token_ids = {1},
      .llm_hidden_states = std::vector<float>(config.llm_hidden_size, 1.0f),
      .num_tokens = 1,
      .generate_audio = true,
      .is_final = true,
  });
  EXPECT_TRUE(tts_stage->NeedSchedule());
  ASSERT_OK(tts_stage->Schedule());

  ASSERT_OK_AND_ASSIGN(Any2AnyTtsOutput out, tts_stage->GetOutput());
  EXPECT_TRUE(out.has_text);
  EXPECT_EQ(out.text, "Text only");
  EXPECT_THAT(out.audio_codes, IsEmpty());
  EXPECT_TRUE(out.is_text_final);
}

TEST(QwenOmniTtsStageTest, ValidatesLlmInputsCacheLenAndVocabBounds) {
  FakeLlmStage llm_stage;
  QwenOmniModelConfig config =
      MakeSmallQwenOmniTestConfig(/*generate_audio=*/true);

  auto make_stage = [&](const QwenOmniModelConfig& cfg) {
    return QwenOmniTtsStage::Create(
        &llm_stage, cfg,
        MakeMockProjectorRunner(cfg.llm_hidden_size, cfg.tts_hidden_size),
        MakeMockDecoderRunner(cfg.tts_hidden_size, cfg.tts_vocab_size,
                              cfg.tts_num_layers, cfg.tts_max_cache_len,
                              cfg.tts_hidden_size,
                              [v = cfg.tts_vocab_size](int32_t, int32_t) {
                                return std::vector<float>(v, 0.0f);
                              }));
  };

  ASSERT_OK_AND_ASSIGN(auto stage, make_stage(config));

  // Empty llm_hidden_states with num_tokens > 0 and generate_audio = true.
  llm_stage.Push(Any2AnyLlmOutput{
      .text = "A",
      .generated_token_ids = {1},
      .llm_hidden_states = {},
      .num_tokens = 1,
      .generate_audio = true,
  });
  EXPECT_THAT(stage->Schedule(), StatusIs(absl::StatusCode::kInvalidArgument));

  // Mismatched llm_hidden_states size.
  llm_stage.Push(Any2AnyLlmOutput{
      .text = "A",
      .generated_token_ids = {1},
      .llm_hidden_states = {0.1f},
      .num_tokens = 1,
      .generate_audio = true,
  });
  EXPECT_THAT(stage->Schedule(), StatusIs(absl::StatusCode::kInvalidArgument));

  // Mismatched generated_token_ids size.
  llm_stage.Push(Any2AnyLlmOutput{
      .text = "A",
      .generated_token_ids = {1, 2},
      .llm_hidden_states = std::vector<float>(config.llm_hidden_size, 0.1f),
      .num_tokens = 1,
      .generate_audio = true,
  });
  EXPECT_THAT(stage->Schedule(), StatusIs(absl::StatusCode::kInvalidArgument));

  // Mismatched prompt_hidden_states size.
  llm_stage.Push(Any2AnyLlmOutput{
      .text = "A",
      .prompt_token_ids = {1, 2},
      .prompt_hidden_states = {0.1f},
      .generated_token_ids = {1},
      .llm_hidden_states = std::vector<float>(config.llm_hidden_size, 0.1f),
      .num_tokens = 1,
      .generate_audio = true,
  });
  EXPECT_THAT(stage->Schedule(), StatusIs(absl::StatusCode::kInvalidArgument));

  // Out-of-range text token ID >= llm_vocab_size.
  llm_stage.Push(Any2AnyLlmOutput{
      .text = "A",
      .generated_token_ids = {config.llm_vocab_size},
      .llm_hidden_states = std::vector<float>(config.llm_hidden_size, 0.1f),
      .num_tokens = 1,
      .generate_audio = true,
  });
  EXPECT_THAT(stage->Schedule(), StatusIs(absl::StatusCode::kInvalidArgument));

  // tts_max_cache_len < 2 -> FailedPrecondition.
  {
    QwenOmniModelConfig tiny_cache = config;
    tiny_cache.tts_max_cache_len = 1;
    ASSERT_OK_AND_ASSIGN(auto tiny_stage, make_stage(tiny_cache));
    llm_stage.Push(Any2AnyLlmOutput{
        .text = "A",
        .generated_token_ids = {1},
        .llm_hidden_states = std::vector<float>(config.llm_hidden_size, 0.1f),
        .num_tokens = 1,
        .generate_audio = true,
    });
    EXPECT_THAT(tiny_stage->Schedule(),
                StatusIs(absl::StatusCode::kFailedPrecondition));
  }

  // Out-of-range tts_silence_token_id >= tts_vocab_size -> InvalidArgument.
  {
    QwenOmniModelConfig bad_silence = config;
    bad_silence.tts_silence_token_id = bad_silence.tts_vocab_size;
    ASSERT_OK_AND_ASSIGN(auto bad_stage, make_stage(bad_silence));
    llm_stage.Push(Any2AnyLlmOutput{
        .text = "A",
        .generated_token_ids = {1},
        .llm_hidden_states = std::vector<float>(config.llm_hidden_size, 0.1f),
        .num_tokens = 1,
        .generate_audio = true,
    });
    EXPECT_THAT(bad_stage->Schedule(),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }
}

TEST(QwenOmniTtsStageTest,
     PrefillZeroesMultimodalPlaceholdersAndTruncatesPromptAtCacheCap) {
  FakeLlmStage llm_stage;
  QwenOmniModelConfig config =
      MakeSmallQwenOmniTestConfig(/*generate_audio=*/true);
  config.tts_max_cache_len = 6;
  config.max_tts_tokens = 4;

  std::vector<int32_t> proj_tokens;
  std::vector<DecoderCall> dec_calls;
  auto proj = MakeMockProjectorRunner(config.llm_hidden_size,
                                      config.tts_hidden_size, &proj_tokens);
  auto dec = MakeMockDecoderRunner(
      config.tts_hidden_size, config.tts_vocab_size, config.tts_num_layers,
      config.tts_max_cache_len, config.tts_hidden_size,
      [vocab = config.tts_vocab_size](int32_t, int32_t) {
        std::vector<float> logits(vocab, 0.0f);
        logits[2] = 5.0f;
        return logits;
      },
      &dec_calls);
  ASSERT_OK_AND_ASSIGN(
      auto tts_stage, QwenOmniTtsStage::Create(
                          &llm_stage, config, std::move(proj), std::move(dec)));

  // 5 prompt tokens with audio/image/video placeholders, but max_cache_len=6
  // reserves 2 slots for BOS + first reply token, so only 4 prompt tokens fit.
  llm_stage.Push(Any2AnyLlmOutput{
      .text = "Hi",
      .prompt_token_ids = {1, config.audio_token_id, config.image_token_id,
                           config.video_token_id, 2},
      .prompt_hidden_states =
          std::vector<float>(5 * config.llm_hidden_size, 0.25f),
      .generated_token_ids = {3},
      .llm_hidden_states = std::vector<float>(config.llm_hidden_size, 0.5f),
      .num_tokens = 1,
      .generate_audio = true,
      .is_final = true,
  });
  ASSERT_OK(tts_stage->Schedule());

  ASSERT_OK_AND_ASSIGN(Any2AnyTtsOutput out, tts_stage->GetOutput());
  // Step 0..3: 4 prompt tokens; Step 4: BOS; Step 5: first reply token ->
  // produces 1 code then hits step_idx >= max_cache_len (6).
  EXPECT_THAT(out.audio_codes, ElementsAre(2));
  EXPECT_THAT(proj_tokens,
              ElementsAre(1, -1, -1, -1, config.tts_text_bos_token_id, 3));
  ASSERT_EQ(dec_calls.size(), 6);
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(dec_calls[i].prev_audio_code, -1);
    EXPECT_EQ(dec_calls[i].step_idx, i);
  }
  EXPECT_EQ(dec_calls[4].prev_audio_code, config.tts_silence_token_id);
  EXPECT_EQ(dec_calls[4].step_idx, 4);
  EXPECT_EQ(dec_calls[5].prev_audio_code, config.tts_audio_bos_token_id);
  EXPECT_EQ(dec_calls[5].step_idx, 5);
}

TEST(QwenOmniTtsStageTest,
     DecodeAppliesRepetitionPenaltySuppressionCachedPadAndStopsOnEosOrSilence) {
  FakeLlmStage llm_stage;
  QwenOmniModelConfig config =
      MakeSmallQwenOmniTestConfig(/*generate_audio=*/true);
  config.max_tts_tokens = 10;
  config.tts_repetition_penalty = 2.0f;

  std::vector<int32_t> proj_tokens;
  int decode_call = 0;
  auto proj = MakeMockProjectorRunner(config.llm_hidden_size,
                                      config.tts_hidden_size, &proj_tokens);
  auto dec = MakeMockDecoderRunner(
      config.tts_hidden_size, config.tts_vocab_size, config.tts_num_layers,
      config.tts_max_cache_len, config.tts_hidden_size, [&](int32_t, int32_t) {
        std::vector<float> logits(config.tts_vocab_size, -5.0f);
        ++decode_call;
        if (decode_call == 2) {
          // Output of first reply token step: audio BOS has huge logit (must be
          // suppressed!), code 2 wins over code 3.
          logits[config.tts_audio_bos_token_id] = 100.0f;
          logits[2] = 4.0f;
          logits[3] = 3.0f;
        } else if (decode_call == 3) {
          // After repetition penalty (2.0): code 2 becomes 4.0/2.0 = 2.0 < 3.0,
          // and negative logit -1.0 for code 2 would multiply by 2.0. Code 3
          // wins!
          logits[2] = 4.0f;
          logits[3] = 3.0f;
        } else if (decode_call == 4) {
          // Test negative logit repetition penalty: code 2 & 3 are in history,
          // both negative: -0.5 * 2.0 = -1.0 < -0.8 for code 4 -> code 4 wins!
          logits[2] = -0.5f;
          logits[3] = -0.5f;
          logits[4] = -0.8f;
        } else if (decode_call == 5) {
          // Emit code 1 so next step reuses cached text_pad_cond without
          // calling projector again.
          logits[1] = 2.0f;
        } else {
          // Stop on EOS.
          logits[config.tts_eos_token_id] = 10.0f;
        }
        return logits;
      });
  ASSERT_OK_AND_ASSIGN(
      auto tts_stage, QwenOmniTtsStage::Create(
                          &llm_stage, config, std::move(proj), std::move(dec)));

  llm_stage.Push(Any2AnyLlmOutput{
      .text = "Hello",
      .generated_token_ids = {1},
      .llm_hidden_states = std::vector<float>(config.llm_hidden_size, 0.5f),
      .num_tokens = 1,
      .generate_audio = true,
      .is_final = true,
  });
  ASSERT_OK(tts_stage->Schedule());

  ASSERT_OK_AND_ASSIGN(Any2AnyTtsOutput out, tts_stage->GetOutput());
  EXPECT_THAT(out.audio_codes, ElementsAre(2, 3, 4, 1));
  // Projector calls: text_bos, reply[0]=1, text_eos, text_pad (cached once!).
  EXPECT_THAT(proj_tokens, ElementsAre(config.tts_text_bos_token_id, 1,
                                       config.tts_text_eos_token_id,
                                       config.tts_text_pad_token_id));
}

TEST(QwenOmniTtsStageTest, SuppressesSpecialCodecTokensAboveVocoderCodebook) {
  FakeLlmStage llm_stage;
  QwenOmniModelConfig config =
      MakeSmallQwenOmniTestConfig(/*generate_audio=*/true);
  config.tts_vocab_size = qwen_omni::kVocoderCodebookSize + 4;
  config.tts_silence_token_id = qwen_omni::kVocoderCodebookSize;
  config.tts_audio_bos_token_id = qwen_omni::kVocoderCodebookSize + 1;
  config.tts_eos_token_id = qwen_omni::kVocoderCodebookSize + 2;
  const int kOtherSpecialCodecId = qwen_omni::kVocoderCodebookSize + 3;
  config.max_tts_tokens = 1024;  // Also exercises >= 1024 cap calculation.

  int call_count = 0;
  auto proj =
      MakeMockProjectorRunner(config.llm_hidden_size, config.tts_hidden_size);
  auto dec = MakeMockDecoderRunner(
      config.tts_hidden_size, config.tts_vocab_size, config.tts_num_layers,
      config.tts_max_cache_len, config.tts_hidden_size, [&](int32_t, int32_t) {
        std::vector<float> logits(config.tts_vocab_size, -1.0f);
        ++call_count;
        if (call_count == 2) {
          // Both BOS and extra special token >= kVocoderCodebookSize are
          // suppressed, so code 42 wins.
          logits[config.tts_audio_bos_token_id] = 50.0f;
          logits[kOtherSpecialCodecId] = 40.0f;
          logits[42] = 5.0f;
        } else {
          // Silence token is NOT suppressed and stops generation.
          logits[config.tts_silence_token_id] = 10.0f;
        }
        return logits;
      });
  ASSERT_OK_AND_ASSIGN(
      auto tts_stage, QwenOmniTtsStage::Create(
                          &llm_stage, config, std::move(proj), std::move(dec)));

  llm_stage.Push(Any2AnyLlmOutput{
      .text = "Hi",
      .generated_token_ids = {1},
      .llm_hidden_states = std::vector<float>(config.llm_hidden_size, 0.5f),
      .num_tokens = 1,
      .generate_audio = true,
      .is_final = true,
  });
  ASSERT_OK(tts_stage->Schedule());

  ASSERT_OK_AND_ASSIGN(Any2AnyTtsOutput out, tts_stage->GetOutput());
  EXPECT_THAT(out.audio_codes, ElementsAre(42));
}

TEST(QwenOmniTtsStageTest, SamplingTopKTopPNonFiniteLogitsAndResetDeterminism) {
  FakeLlmStage llm_stage;
  QwenOmniModelConfig config =
      MakeSmallQwenOmniTestConfig(/*generate_audio=*/true);
  config.tts_do_sample = true;
  config.tts_temperature = 0.8f;
  config.tts_top_k = 3;
  config.tts_top_p = 0.9f;
  config.seed = 777;
  config.max_tts_tokens = 4;

  const float kNan = std::numeric_limits<float>::quiet_NaN();
  const float kInf = std::numeric_limits<float>::infinity();

  int step = 0;
  auto proj =
      MakeMockProjectorRunner(config.llm_hidden_size, config.tts_hidden_size);
  auto dec = MakeMockDecoderRunner(
      config.tts_hidden_size, config.tts_vocab_size, config.tts_num_layers,
      config.tts_max_cache_len, config.tts_hidden_size, [&](int32_t, int32_t) {
        std::vector<float> logits(config.tts_vocab_size, -10.0f);
        ++step;
        if (step == 2) {
          // Normal sampling with NaN and -Inf mixed in.
          logits[0] = kNan;
          logits[1] = 1.0f;
          logits[2] = 1.1f;
          logits[3] = 0.9f;
          logits[4] = -kInf;
        } else if (step == 3) {
          // +Inf immediately selects index 3.
          logits[1] = 10.0f;
          logits[3] = kInf;
        } else if (step == 4) {
          // All NaN / -Inf falls back safely to 0.
          logits.assign(config.tts_vocab_size, kNan);
        } else {
          // Stop on EOS.
          logits[config.tts_eos_token_id] = 100.0f;
        }
        return logits;
      });
  ASSERT_OK_AND_ASSIGN(
      auto tts_stage, QwenOmniTtsStage::Create(
                          &llm_stage, config, std::move(proj), std::move(dec)));

  const Any2AnyLlmOutput input{
      .text = "Sample",
      .generated_token_ids = {1},
      .llm_hidden_states = std::vector<float>(config.llm_hidden_size, 0.5f),
      .num_tokens = 1,
      .generate_audio = true,
      .is_final = true,
  };

  llm_stage.Push(input);
  ASSERT_OK(tts_stage->Schedule());
  ASSERT_OK_AND_ASSIGN(Any2AnyTtsOutput first_run, tts_stage->GetOutput());
  ASSERT_EQ(first_run.audio_codes.size(), 3);
  EXPECT_EQ(first_run.audio_codes[1], 3);
  EXPECT_EQ(first_run.audio_codes[2], 0);

  // Reset restores RNG seed -> identical sampled sequence.
  step = 0;
  tts_stage->Reset();
  llm_stage.Push(input);
  ASSERT_OK(tts_stage->Schedule());
  ASSERT_OK_AND_ASSIGN(Any2AnyTtsOutput second_run, tts_stage->GetOutput());
  EXPECT_EQ(second_run.audio_codes, first_run.audio_codes);
}

TEST(QwenOmniTtsStageTest, GreedySelectionHandlesNonFiniteLogits) {
  FakeLlmStage llm_stage;
  QwenOmniModelConfig config =
      MakeSmallQwenOmniTestConfig(/*generate_audio=*/true);
  config.tts_do_sample = false;
  config.max_tts_tokens = 3;

  const float kNan = std::numeric_limits<float>::quiet_NaN();
  const float kInf = std::numeric_limits<float>::infinity();

  int step = 0;
  auto proj =
      MakeMockProjectorRunner(config.llm_hidden_size, config.tts_hidden_size);
  auto dec = MakeMockDecoderRunner(
      config.tts_hidden_size, config.tts_vocab_size, config.tts_num_layers,
      config.tts_max_cache_len, config.tts_hidden_size, [&](int32_t, int32_t) {
        std::vector<float> logits(config.tts_vocab_size, kNan);
        ++step;
        if (step == 2) {
          // NaN skipped, finite max at index 4 selected.
          logits[0] = kNan;
          logits[2] = -kInf;
          logits[4] = -1.5f;
        } else if (step == 3) {
          // +Inf at index 2 selected immediately.
          logits[1] = 100.0f;
          logits[2] = kInf;
        } else {
          // All NaN -> falls back to 0.
        }
        return logits;
      });
  ASSERT_OK_AND_ASSIGN(
      auto tts_stage, QwenOmniTtsStage::Create(
                          &llm_stage, config, std::move(proj), std::move(dec)));

  llm_stage.Push(Any2AnyLlmOutput{
      .text = "Greedy",
      .generated_token_ids = {1},
      .llm_hidden_states = std::vector<float>(config.llm_hidden_size, 0.5f),
      .num_tokens = 1,
      .generate_audio = true,
      .is_final = true,
  });
  ASSERT_OK(tts_stage->Schedule());
  ASSERT_OK_AND_ASSIGN(Any2AnyTtsOutput out, tts_stage->GetOutput());
  EXPECT_THAT(out.audio_codes, ElementsAre(4, 2, 0));
}

}  // namespace
}  // namespace litert::omni::any2any
