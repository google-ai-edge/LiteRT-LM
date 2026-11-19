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

#include "omni/any2any/qwen_omni/qwen_omni_vocoder_stage.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "omni/any2any/any2any_test_utils.h"
#include "omni/any2any/io_types.h"
#include "omni/any2any/qwen_omni/qwen_omni_model_config.h"
#include "omni/any2any/qwen_omni/qwen_omni_test_utils.h"
#include "omni/base/io_types.h"
#include "omni/base/litert_runner.h"
#include "omni/base/mock_litert_runner.h"
#include "omni/base/model_utils.h"
#include "runtime/executor/executor_settings_base.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep
#include "tflite/schema/schema_generated.h"  // from @litert

namespace litert::omni::any2any {
namespace {

using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::FloatEq;
using ::testing::NiceMock;

TEST(QwenOmniVocoderStageTest, VocoderStageRejectsOutOfRangeAudioCodes) {
  // Arrange
  QwenOmniModelConfig config =
      MakeSmallQwenOmniTestConfig(/*generate_audio=*/true);
  auto env = ::litert::Environment::Create({});
  ASSERT_TRUE(env.HasValue());
  ModelOptions opts{.backend = lm::Backend::CPU, .num_threads = 1};

  std::string voc_buf = BuildMinimalTfliteModelBuffer(
      {{"audio_codes", {1, 4}, tflite::TensorType_INT32}},
      {{"waveform", {1, 8}, tflite::TensorType_FLOAT32}});
  ASSERT_OK_AND_ASSIGN(
      auto voc_model,
      CreateCompiledModelFromBuffer(*env, opts, voc_buf, "audio_vocoder"));

  FakeTtsStage tts_stage;
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<QwenOmniVocoderStage> vocoder_stage,
      QwenOmniVocoderStage::Create(
          &tts_stage, config, std::make_unique<LiteRtRunnerImpl>(&voc_model)));

  // Act & Assert: push out-of-range code 8 (tts_vocab_size = 8) with text and
  // verify the wrapper still drains `TextOutput` before returning the error.
  tts_stage.Push(Any2AnyTtsOutput{
      .text = "BeforeError",
      .has_text = true,
      .is_text_final = true,
      .audio_codes = {0, 8},
  });
  EXPECT_THAT(vocoder_stage->Schedule(),
              StatusIs(absl::StatusCode::kInvalidArgument));
  ASSERT_OK_AND_ASSIGN(Output drained_text, vocoder_stage->GetOutput());
  ASSERT_TRUE(std::holds_alternative<TextOutput>(drained_text));
  EXPECT_EQ(std::get<TextOutput>(drained_text).confirmed_text, "BeforeError");
  EXPECT_FALSE(vocoder_stage->HasOutput());

  tts_stage.Push(Any2AnyTtsOutput{
      .is_text_final = true,
      .audio_codes = {-1},
  });
  EXPECT_THAT(vocoder_stage->Schedule(),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(QwenOmniVocoderStageTest,
     CapsMaxValidCodeToVocoderCodebookSizeAndHandlesNonPositiveVocabSize) {
  FakeTtsStage tts_stage;

  // Real default `tts_vocab_size = 8448` exceeds `kVocoderCodebookSize = 8193`,
  // so `max_valid_code` must be capped at `kVocoderCodebookSize`: `8192` is
  // valid, while `8193` and special token `kTtsCodecPadTokenId` (8292) must be
  // rejected.
  {
    QwenOmniModelConfig config =
        MakeSmallQwenOmniTestConfig(/*generate_audio=*/true);
    config.tts_vocab_size = qwen_omni::kTtsVocabSize;
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<QwenOmniVocoderStage> vocoder_stage,
        QwenOmniVocoderStage::Create(
            &tts_stage, config,
            MakeMockVocoderRunner(/*input_tokens=*/4, /*output_samples=*/8)));

    tts_stage.Push(Any2AnyTtsOutput{
        .is_text_final = false,
        .audio_codes = {qwen_omni::kVocoderCodebookSize - 1},
    });
    ASSERT_OK(vocoder_stage->Schedule());
    ASSERT_OK_AND_ASSIGN(Output out, vocoder_stage->GetOutput());
    ASSERT_TRUE(std::holds_alternative<AudioOutput>(out));
    EXPECT_FALSE(std::get<AudioOutput>(out).pcm_samples.empty());

    tts_stage.Push(Any2AnyTtsOutput{
        .is_text_final = false,
        .audio_codes = {qwen_omni::kVocoderCodebookSize},
    });
    EXPECT_THAT(vocoder_stage->Schedule(),
                StatusIs(absl::StatusCode::kInvalidArgument));

    tts_stage.Push(Any2AnyTtsOutput{
        .is_text_final = false,
        .audio_codes = {qwen_omni::kTtsCodecPadTokenId},
    });
    EXPECT_THAT(vocoder_stage->Schedule(),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }

  // Non-positive `tts_vocab_size` (`0` or negative) falls back to
  // `kVocoderCodebookSize`.
  for (int non_positive_vocab_size : {0, -1}) {
    QwenOmniModelConfig config =
        MakeSmallQwenOmniTestConfig(/*generate_audio=*/true);
    config.tts_vocab_size = non_positive_vocab_size;
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<QwenOmniVocoderStage> vocoder_stage,
        QwenOmniVocoderStage::Create(
            &tts_stage, config,
            MakeMockVocoderRunner(/*input_tokens=*/4, /*output_samples=*/8)));

    tts_stage.Push(Any2AnyTtsOutput{
        .is_text_final = false,
        .audio_codes = {qwen_omni::kVocoderCodebookSize - 1},
    });
    ASSERT_OK(vocoder_stage->Schedule());
    ASSERT_OK_AND_ASSIGN(Output out, vocoder_stage->GetOutput());
    ASSERT_TRUE(std::holds_alternative<AudioOutput>(out));
    EXPECT_FALSE(std::get<AudioOutput>(out).pcm_samples.empty());

    tts_stage.Push(Any2AnyTtsOutput{
        .is_text_final = false,
        .audio_codes = {qwen_omni::kVocoderCodebookSize},
    });
    EXPECT_THAT(vocoder_stage->Schedule(),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }
}

TEST(QwenOmniVocoderStageTest,
     VocoderStageRejectsMissingModelWhenAudioEnabled) {
  // Arrange
  QwenOmniModelConfig config;
  config.generate_audio = true;

  // Act & Assert
  FakeTtsStage tts_stage;
  EXPECT_THAT(QwenOmniVocoderStage::Create(&tts_stage, config).status(),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST(QwenOmniVocoderStageTest, RejectsInvalidWindowConfiguration) {
  FakeTtsStage tts_stage;
  {
    QwenOmniModelConfig cfg = MakeSmallQwenOmniTestConfig(false);
    cfg.vocoder_history_tokens = -1;
    EXPECT_THAT(QwenOmniVocoderStage::Create(&tts_stage, cfg).status(),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }
  {
    QwenOmniModelConfig cfg = MakeSmallQwenOmniTestConfig(false);
    cfg.vocoder_chunk_tokens = 0;
    EXPECT_THAT(QwenOmniVocoderStage::Create(&tts_stage, cfg).status(),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }
  {
    QwenOmniModelConfig cfg = MakeSmallQwenOmniTestConfig(false);
    cfg.vocoder_lookahead_tokens = -1;
    EXPECT_THAT(QwenOmniVocoderStage::Create(&tts_stage, cfg).status(),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }
  {
    QwenOmniModelConfig cfg = MakeSmallQwenOmniTestConfig(false);
    cfg.vocoder_samples_per_token = 0;
    EXPECT_THAT(QwenOmniVocoderStage::Create(&tts_stage, cfg).status(),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }
}

TEST(QwenOmniVocoderStageTest, RejectsEmptyAndIncompatibleVocoderBuffers) {
  FakeTtsStage tts_stage;
  QwenOmniModelConfig config =
      MakeSmallQwenOmniTestConfig(/*generate_audio=*/true);

  {
    auto runner = std::make_unique<NiceMock<MockLiteRtRunner>>();
    ON_CALL(*runner, CreateInputBuffers(absl::string_view("")))
        .WillByDefault(
            [](absl::string_view) { return std::vector<TensorBuffer>{}; });
    ON_CALL(*runner, CreateOutputBuffers(absl::string_view("")))
        .WillByDefault(
            [](absl::string_view) { return std::vector<TensorBuffer>{}; });
    EXPECT_THAT(
        QwenOmniVocoderStage::Create(&tts_stage, config, std::move(runner))
            .status(),
        StatusIs(absl::StatusCode::kInternal));
  }

  {
    // 4 input tokens, 5 output floats (not divisible by window=4 or trimmed=3).
    std::unique_ptr<NiceMock<MockLiteRtRunner>> runner =
        MakeMockVocoderRunner(/*input_tokens=*/4, /*output_samples=*/5);
    EXPECT_THAT(
        QwenOmniVocoderStage::Create(&tts_stage, config, std::move(runner))
            .status(),
        StatusIs(absl::StatusCode::kInvalidArgument));
  }
}

TEST(QwenOmniVocoderStageTest,
     AllowsNullRunnerWhenAudioDisabledAndForwardsTextAndTurnEnd) {
  FakeTtsStage tts_stage;
  QwenOmniModelConfig config =
      MakeSmallQwenOmniTestConfig(/*generate_audio=*/false);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<QwenOmniVocoderStage> vocoder_stage,
                       QwenOmniVocoderStage::Create(&tts_stage, config));

  tts_stage.Push(Any2AnyTtsOutput{
      .text = "Hello",
      .has_text = true,
      .is_text_final = true,
      .audio_codes = {1, 2},
  });
  ASSERT_OK(vocoder_stage->Schedule());

  ASSERT_OK_AND_ASSIGN(Output out1, vocoder_stage->GetOutput());
  ASSERT_TRUE(std::holds_alternative<TextOutput>(out1));
  EXPECT_EQ(std::get<TextOutput>(out1).confirmed_text, "Hello");

  ASSERT_OK_AND_ASSIGN(Output out2, vocoder_stage->GetOutput());
  EXPECT_TRUE(std::holds_alternative<EndOfOutputTurn>(out2));
  EXPECT_THAT(vocoder_stage->GetOutput().status(),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST(QwenOmniVocoderStageTest,
     SynthesizesMultiChunkSlidingWindowWithBoundaryPadding) {
  FakeTtsStage tts_stage;
  QwenOmniModelConfig config =
      MakeSmallQwenOmniTestConfig(/*generate_audio=*/true);
  config.vocoder_history_tokens = 1;
  config.vocoder_chunk_tokens = 2;
  config.vocoder_lookahead_tokens = 1;
  config.vocoder_samples_per_token = 2;
  config.audio_output_sample_rate_hz = 24000;

  std::vector<std::vector<int32_t>> recorded_windows;
  // Window = 1 + 2 + 1 = 4 tokens, full output = 4 * 2 = 8 samples.
  std::unique_ptr<NiceMock<MockLiteRtRunner>> runner = MakeMockVocoderRunner(
      /*input_tokens=*/4, /*output_samples=*/8, &recorded_windows);
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<QwenOmniVocoderStage> vocoder_stage,
      QwenOmniVocoderStage::Create(&tts_stage, config, std::move(runner)));

  tts_stage.Push(Any2AnyTtsOutput{
      .text = "Hi",
      .has_text = true,
      .is_text_final = true,
      .audio_codes = {1, 2, 3, 4, 5},
  });
  ASSERT_OK(vocoder_stage->Schedule());

  // 3 sliding windows for 5 tokens with history=1, chunk=2, lookahead=1:
  // Chunk 0 (codes 0..1): [pad=1, 1, 2, 3]
  // Chunk 1 (codes 2..3): [2, 3, 4, 5]
  // Chunk 2 (code 4):     [4, 5, pad=5, pad=5]
  ASSERT_EQ(recorded_windows.size(), 3);
  EXPECT_THAT(recorded_windows[0], ElementsAre(1, 1, 2, 3));
  EXPECT_THAT(recorded_windows[1], ElementsAre(2, 3, 4, 5));
  EXPECT_THAT(recorded_windows[2], ElementsAre(4, 5, 5, 5));

  ASSERT_OK_AND_ASSIGN(Output out1, vocoder_stage->GetOutput());
  ASSERT_TRUE(std::holds_alternative<TextOutput>(out1));
  EXPECT_EQ(std::get<TextOutput>(out1).confirmed_text, "Hi");

  ASSERT_OK_AND_ASSIGN(Output out2, vocoder_stage->GetOutput());
  ASSERT_TRUE(std::holds_alternative<AudioOutput>(out2));
  const AudioOutput& audio = std::get<AudioOutput>(out2);
  EXPECT_EQ(audio.sample_rate_hz, 24000);
  EXPECT_THAT(audio.pcm_samples,
              ElementsAre(FloatEq(10.0f), FloatEq(11.0f), FloatEq(20.0f),
                          FloatEq(21.0f), FloatEq(30.0f), FloatEq(31.0f),
                          FloatEq(40.0f), FloatEq(41.0f), FloatEq(50.0f),
                          FloatEq(51.0f)));

  ASSERT_OK_AND_ASSIGN(Output out3, vocoder_stage->GetOutput());
  EXPECT_TRUE(std::holds_alternative<EndOfOutputTurn>(out3));
}

TEST(QwenOmniVocoderStageTest, InfersSamplesPerTokenAndWindowFromBuffers) {
  FakeTtsStage tts_stage;
  QwenOmniModelConfig config =
      MakeSmallQwenOmniTestConfig(/*generate_audio=*/true);
  config.vocoder_history_tokens = 1;
  config.vocoder_chunk_tokens = 2;
  config.vocoder_lookahead_tokens = 1;
  config.vocoder_samples_per_token = 4;  // Deliberately mismatched.

  // Branch 1: matching input window (4 tokens), output (20 samples) does not
  // match expected trimmed (12) or full (16) bytes, and is divisible by
  // window_tokens (4) -> samples_per_token = 20 / 4 = 5.
  {
    std::unique_ptr<NiceMock<MockLiteRtRunner>> runner =
        MakeMockVocoderRunner(/*input_tokens=*/4, /*output_samples=*/20);
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<QwenOmniVocoderStage> stage,
        QwenOmniVocoderStage::Create(&tts_stage, config, std::move(runner)));
    tts_stage.Push(Any2AnyTtsOutput{
        .is_text_final = false,
        .audio_codes = {1, 2},
    });
    ASSERT_OK(stage->Schedule());
    ASSERT_OK_AND_ASSIGN(Output out, stage->GetOutput());
    ASSERT_TRUE(std::holds_alternative<AudioOutput>(out));
    EXPECT_EQ(std::get<AudioOutput>(out).pcm_samples.size(), 10);
  }

  // Branch 2: matching input window (4 tokens), output divisible by
  // trimmed_tokens (history + chunk = 3) -> samples_per_token = 9 / 3 = 3.
  {
    std::unique_ptr<NiceMock<MockLiteRtRunner>> runner =
        MakeMockVocoderRunner(/*input_tokens=*/4, /*output_samples=*/9);
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<QwenOmniVocoderStage> stage,
        QwenOmniVocoderStage::Create(&tts_stage, config, std::move(runner)));
    tts_stage.Push(Any2AnyTtsOutput{
        .is_text_final = false,
        .audio_codes = {1, 2},
    });
    ASSERT_OK(stage->Schedule());
    ASSERT_OK_AND_ASSIGN(Output out, stage->GetOutput());
    ASSERT_TRUE(std::holds_alternative<AudioOutput>(out));
    EXPECT_EQ(std::get<AudioOutput>(out).pcm_samples.size(), 6);
  }

  // Branch 3: non-matching input tokens (3 tokens), output divisible by
  // in_tokens -> history=0, chunk=3, lookahead=0, samples_per_token = 6 / 3
  // = 2.
  {
    std::vector<std::vector<int32_t>> recorded_windows;
    std::unique_ptr<NiceMock<MockLiteRtRunner>> runner = MakeMockVocoderRunner(
        /*input_tokens=*/3, /*output_samples=*/6, &recorded_windows);
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<QwenOmniVocoderStage> stage,
        QwenOmniVocoderStage::Create(&tts_stage, config, std::move(runner)));
    tts_stage.Push(Any2AnyTtsOutput{
        .is_text_final = false,
        .audio_codes = {1, 2},
    });
    ASSERT_OK(stage->Schedule());
    ASSERT_EQ(recorded_windows.size(), 1);
    EXPECT_THAT(recorded_windows[0], ElementsAre(1, 2, 2));
    ASSERT_OK_AND_ASSIGN(Output out, stage->GetOutput());
    ASSERT_TRUE(std::holds_alternative<AudioOutput>(out));
    EXPECT_EQ(std::get<AudioOutput>(out).pcm_samples.size(), 4);
  }
}

TEST(QwenOmniVocoderStageTest,
     NeedScheduleEmptyScheduleAndFlushDrainAllPendingOutputs) {
  FakeTtsStage tts_stage;
  QwenOmniModelConfig config =
      MakeSmallQwenOmniTestConfig(/*generate_audio=*/false);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<QwenOmniVocoderStage> vocoder_stage,
                       QwenOmniVocoderStage::Create(&tts_stage, config));

  EXPECT_FALSE(vocoder_stage->NeedSchedule());
  EXPECT_OK(vocoder_stage->Schedule());
  EXPECT_THAT(vocoder_stage->GetOutput().status(),
              StatusIs(absl::StatusCode::kNotFound));

  tts_stage.Push(Any2AnyTtsOutput{
      .text = "Part1",
      .has_text = true,
      .is_text_final = false,
  });
  tts_stage.Push(Any2AnyTtsOutput{
      .text = "Part2",
      .has_text = true,
      .is_text_final = true,
  });
  EXPECT_TRUE(vocoder_stage->NeedSchedule());

  ASSERT_OK(vocoder_stage->Flush());

  ASSERT_OK_AND_ASSIGN(Output out1, vocoder_stage->GetOutput());
  ASSERT_TRUE(std::holds_alternative<TextOutput>(out1));
  EXPECT_EQ(std::get<TextOutput>(out1).confirmed_text, "Part1");

  ASSERT_OK_AND_ASSIGN(Output out2, vocoder_stage->GetOutput());
  ASSERT_TRUE(std::holds_alternative<TextOutput>(out2));
  EXPECT_EQ(std::get<TextOutput>(out2).confirmed_text, "Part2");

  ASSERT_OK_AND_ASSIGN(Output out3, vocoder_stage->GetOutput());
  EXPECT_TRUE(std::holds_alternative<EndOfOutputTurn>(out3));

  tts_stage.Push(Any2AnyTtsOutput{
      .text = "Pending",
      .has_text = true,
      .is_text_final = true,
  });
  ASSERT_OK(vocoder_stage->Schedule());
  ASSERT_TRUE(vocoder_stage->HasOutput());
  vocoder_stage->Reset();
  EXPECT_FALSE(vocoder_stage->HasOutput());
}

}  // namespace
}  // namespace litert::omni::any2any
