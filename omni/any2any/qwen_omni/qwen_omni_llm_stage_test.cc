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

#include "omni/any2any/qwen_omni/qwen_omni_llm_stage.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "omni/any2any/io_types.h"
#include "omni/any2any/qwen_omni/qwen_omni_model_config.h"
#include "omni/base/mock_litert_lm_runner.h"
#include "omni/base/stage.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "support/tokenizer/tokenizer.h"
#include "support/util/convert_tensor_buffer.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep
#include "tflite/types/half.h"  // from @litert

namespace litert::omni::any2any {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::_;
using ::testing::AnyOf;
using ::testing::ElementsAre;
using ::testing::NiceMock;

class FakeFrontendStage
    : public SingleThreadedStageWithDeque<Any2AnyFrontendOutput> {
 public:
  void Push(Any2AnyFrontendOutput prompt) { PushOutput(std::move(prompt)); }

 protected:
  bool NeedScheduleInternal() const override { return false; }
  absl::Status ScheduleInternal() override { return absl::OkStatus(); }
};

class FakeTokenizer : public support::Tokenizer {
 public:
  explicit FakeTokenizer(bool fail_text_to_token_ids = false)
      : fail_text_to_token_ids_(fail_text_to_token_ids) {}

  support::TokenizerType GetTokenizerType() const override {
    return support::TokenizerType::kHuggingFace;
  }

  absl::StatusOr<support::TokenIds> TextToTokenIds(
      absl::string_view text) override {
    if (!fail_text_to_token_ids_ && text == "\n") {
      return support::TokenIds{5};
    }
    return absl::NotFoundError("not implemented");
  }

  absl::StatusOr<int> TokenToId(absl::string_view token) override {
    if (token == "<|endoftext|>") return 0;
    if (token == "<|im_end|>") return 2;
    if (token == "\n") return 5;
    return absl::NotFoundError("not found");
  }

  absl::StatusOr<std::string> TokenIdsToText(
      absl::Span<const int> token_ids, bool /*skip_special_tokens*/) override {
    std::string out = " ";
    for (size_t i = 0; i < token_ids.size(); ++i) {
      if (i > 0) absl::StrAppend(&out, " ");
      if (token_ids[i] == 3) {
        absl::StrAppend(&out, "Hello");
      } else if (token_ids[i] == 4) {
        absl::StrAppend(&out, "world");
      } else {
        absl::StrAppend(&out, "tok", token_ids[i]);
      }
    }
    return out;
  }

  std::vector<std::string> GetTokens() const override { return {}; }
  int GetVocabSize() const override { return 8; }

 private:
  bool fail_text_to_token_ids_ = false;
};

absl::StatusOr<TensorBuffer> MakeFloatTensorBuffer(
    absl::Span<const float> values) {
  LITERT_ASSIGN_OR_RETURN(auto buf,
                          support::CreateTensorBuffer<float>(
                              {1, 1, static_cast<int>(values.size())}));
  LITERT_RETURN_IF_ERROR(buf.Write<float>(values));
  return buf;
}

absl::StatusOr<TensorBuffer> MakeFloat16TensorBuffer(
    absl::Span<const float> values) {
  std::vector<tflite::half> fp16(values.begin(), values.end());
  LITERT_ASSIGN_OR_RETURN(auto buf, support::CreateTensorBuffer<tflite::half>(
                                        {1, 1, static_cast<int>(fp16.size())}));
  LITERT_RETURN_IF_ERROR(buf.Write<tflite::half>(absl::MakeConstSpan(fp16)));
  return buf;
}

TEST(QwenOmniLlmStageTest, CreateValidatesParametersAndNewlineFallback) {
  // Arrange
  FakeFrontendStage frontend;
  auto tokenizer = std::make_shared<FakeTokenizer>();
  QwenOmniModelConfig config;
  config.llm_hidden_size = 4;
  config.llm_vocab_size = 8;

  // Act & Assert
  QwenOmniModelConfig bad_hidden = config;
  bad_hidden.llm_hidden_size = 0;
  EXPECT_THAT(QwenOmniLlmStage::Create(
                  &frontend, tokenizer,
                  std::make_shared<NiceMock<MockLiteRtLmRunner>>(), bad_hidden),
              StatusIs(absl::StatusCode::kInvalidArgument));

  QwenOmniModelConfig bad_vocab = config;
  bad_vocab.llm_vocab_size = 0;
  EXPECT_THAT(QwenOmniLlmStage::Create(
                  &frontend, tokenizer,
                  std::make_shared<NiceMock<MockLiteRtLmRunner>>(), bad_vocab),
              StatusIs(absl::StatusCode::kInvalidArgument));

  EXPECT_THAT(QwenOmniLlmStage::Create(
                  &frontend, tokenizer,
                  std::make_shared<NiceMock<MockLiteRtLmRunner>>(), config),
              IsOk());

  // Fallback to TokenToId("\n") when TextToTokenIds("\n") fails.
  EXPECT_THAT(
      QwenOmniLlmStage::Create(
          &frontend,
          std::make_shared<FakeTokenizer>(/*fail_text_to_token_ids=*/true),
          std::make_shared<NiceMock<MockLiteRtLmRunner>>(), config),
      IsOk());
}

TEST(QwenOmniLlmStageTest, TextOnlyPrefillAndGreedyDecodeStopsOnImEnd) {
  // Arrange
  FakeFrontendStage frontend;
  QwenOmniModelConfig config;
  config.llm_hidden_size = 4;
  config.llm_vocab_size = 8;
  config.max_new_tokens = 4;
  config.generate_audio = false;
  config.do_sample = false;

  auto llm_runner = std::make_shared<NiceMock<MockLiteRtLmRunner>>();
  ON_CALL(*llm_runner, Prefill(_)).WillByDefault([](const lm::ExecutorInputs&) {
    return absl::OkStatus();
  });

  std::vector<int32_t> observed_tokens;
  int step = 0;
  ON_CALL(*llm_runner, Decode(_))
      .WillByDefault([&step, &observed_tokens](const lm::ExecutorInputs& inputs)
                         -> absl::StatusOr<TensorBuffer> {
        ABSL_ASSIGN_OR_RETURN(const TensorBuffer* tok_buf,
                              inputs.GetTextTokenIdsPtr());
        int32_t tok = -999;
        LITERT_RETURN_IF_ERROR(
            const_cast<TensorBuffer*>(tok_buf)->Read<int32_t>(
                absl::MakeSpan(&tok, 1)));
        observed_tokens.push_back(tok);

        std::vector<float> logits(8, -10.0f);
        if (step == 0) {
          logits[3] = 10.0f;  // "Hello"
        } else if (step == 1) {
          logits[4] = 10.0f;  // "world"
        } else {
          logits[2] = 10.0f;  // <|im_end|>
        }
        ++step;
        return MakeFloatTensorBuffer(logits);
      });

  ASSERT_OK_AND_ASSIGN(
      auto stage,
      QwenOmniLlmStage::Create(&frontend, std::make_shared<FakeTokenizer>(),
                               llm_runner, config));

  EXPECT_FALSE(stage->NeedSchedule());
  frontend.Push(Any2AnyFrontendOutput{
      .prompt_embeddings = std::vector<float>(2 * 4, 0.5f),
      .prompt_len = 2,
      .token_ids = {1, 3},
      // Verify output generate_audio respects config_.generate_audio == false.
      .generate_audio = true,
  });
  EXPECT_TRUE(stage->NeedSchedule());

  // Act
  ASSERT_OK(stage->Schedule());

  // Assert
  ASSERT_OK_AND_ASSIGN(Any2AnyLlmOutput out, stage->GetOutput());
  EXPECT_FALSE(out.generate_audio);
  EXPECT_EQ(out.text, "Hello world");
  EXPECT_THAT(out.generated_token_ids, ElementsAre(3, 4));
  EXPECT_EQ(out.num_tokens, 2);
  EXPECT_TRUE(out.prompt_hidden_states.empty());
  EXPECT_TRUE(out.llm_hidden_states.empty());
  // Decode inputs: -1 (after Prefill), 3 ("Hello"), 4 ("world"), then turn-end
  // suffix 2 (<|im_end|>) and 5 ("\n").
  EXPECT_THAT(observed_tokens, ElementsAre(-1, 3, 4, 2, 5));
}

TEST(QwenOmniLlmStageTest, GenerateAudioCollectsPromptAndReplyHiddenStates) {
  // Arrange
  FakeFrontendStage frontend;
  QwenOmniModelConfig config;
  config.llm_hidden_size = 4;
  config.llm_vocab_size = 8;
  config.max_new_tokens = 2;
  config.generate_audio = true;
  config.do_sample = false;

  auto llm_runner = std::make_shared<NiceMock<MockLiteRtLmRunner>>();
  ON_CALL(*llm_runner, Prefill(_)).WillByDefault([](const lm::ExecutorInputs&) {
    return absl::OkStatus();
  });

  int decode_call = 0;
  ON_CALL(*llm_runner, Decode(_))
      .WillByDefault([&decode_call](const lm::ExecutorInputs&)
                         -> absl::StatusOr<TensorBuffer> {
        // Output size = vocab_size(8) + hidden_dim(4) = 12.
        std::vector<float> raw(12, -10.0f);
        if (decode_call == 1) {
          raw[3] = 10.0f;  // After last prompt token -> emit token 3 ("Hello").
        } else if (decode_call == 2) {
          raw[2] = 10.0f;  // After token 3 -> stop (<|im_end|>).
        } else {
          raw[0] = 1.0f;
        }
        const float hidden_val = static_cast<float>(decode_call + 1);
        for (int h = 0; h < 4; ++h) {
          raw[8 + h] = hidden_val;
        }
        ++decode_call;
        return MakeFloatTensorBuffer(raw);
      });

  ASSERT_OK_AND_ASSIGN(
      auto stage,
      QwenOmniLlmStage::Create(&frontend, std::make_shared<FakeTokenizer>(),
                               llm_runner, config));

  frontend.Push(Any2AnyFrontendOutput{
      .prompt_embeddings = std::vector<float>(2 * 4, 0.25f),
      .prompt_len = 2,
      .token_ids = {1, 3},
      .generate_audio = true,
  });

  // Act
  ASSERT_OK(stage->Schedule());

  // Assert
  ASSERT_OK_AND_ASSIGN(Any2AnyLlmOutput out, stage->GetOutput());
  EXPECT_TRUE(out.generate_audio);
  EXPECT_EQ(out.text, "Hello");
  EXPECT_THAT(out.prompt_token_ids, ElementsAre(1, 3));
  EXPECT_THAT(out.generated_token_ids, ElementsAre(3));
  // Prompt hidden states from calls 0 and 1.
  EXPECT_THAT(out.prompt_hidden_states,
              ElementsAre(1.0f, 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 2.0f));
  // Reply hidden state from call 2 (when token 3 is fed as input).
  EXPECT_THAT(out.llm_hidden_states, ElementsAre(3.0f, 3.0f, 3.0f, 3.0f));
}

TEST(QwenOmniLlmStageTest, GenerateAudioRejectsRunnerWithoutHiddenStates) {
  // Arrange
  FakeFrontendStage frontend;
  QwenOmniModelConfig config;
  config.llm_hidden_size = 4;
  config.llm_vocab_size = 8;
  config.generate_audio = true;

  auto llm_runner = std::make_shared<NiceMock<MockLiteRtLmRunner>>();
  ON_CALL(*llm_runner, Prefill(_)).WillByDefault([](const lm::ExecutorInputs&) {
    return absl::OkStatus();
  });
  ON_CALL(*llm_runner, Decode(_)).WillByDefault([](const lm::ExecutorInputs&) {
    return MakeFloatTensorBuffer(std::vector<float>(8, 0.0f));
  });

  ASSERT_OK_AND_ASSIGN(
      auto stage,
      QwenOmniLlmStage::Create(&frontend, std::make_shared<FakeTokenizer>(),
                               llm_runner, config));
  frontend.Push(Any2AnyFrontendOutput{
      .prompt_embeddings = std::vector<float>(4, 0.1f),
      .prompt_len = 1,
      .token_ids = {1},
      .generate_audio = true,
  });

  // Act & Assert: missing hidden state on prompt step 0.
  EXPECT_THAT(stage->Schedule(),
              StatusIs(absl::StatusCode::kFailedPrecondition));

  // Missing hidden state on prompt step 1.
  int call = 0;
  ON_CALL(*llm_runner, Decode(_))
      .WillByDefault([&call](const lm::ExecutorInputs&) {
        ++call;
        if (call == 1) {
          return MakeFloatTensorBuffer(std::vector<float>(12, 0.0f));
        }
        return MakeFloatTensorBuffer(std::vector<float>(8, 0.0f));
      });
  frontend.Push(Any2AnyFrontendOutput{
      .prompt_embeddings = std::vector<float>(2 * 4, 0.1f),
      .prompt_len = 2,
      .token_ids = {1, 3},
      .generate_audio = true,
  });
  EXPECT_THAT(stage->Schedule(),
              StatusIs(absl::StatusCode::kFailedPrecondition));

  // Missing hidden state on reply decode step.
  call = 0;
  ON_CALL(*llm_runner, Decode(_))
      .WillByDefault([&call](const lm::ExecutorInputs&) {
        ++call;
        if (call == 1) {
          std::vector<float> raw(12, -10.0f);
          raw[3] = 10.0f;
          return MakeFloatTensorBuffer(raw);
        }
        return MakeFloatTensorBuffer(std::vector<float>(8, 0.0f));
      });
  frontend.Push(Any2AnyFrontendOutput{
      .prompt_embeddings = std::vector<float>(4, 0.1f),
      .prompt_len = 1,
      .token_ids = {1},
      .generate_audio = true,
  });
  EXPECT_THAT(stage->Schedule(),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(QwenOmniLlmStageTest, ValidatesPromptBuffersFloat16AndDecodeOutputErrors) {
  FakeFrontendStage frontend;
  QwenOmniModelConfig config;
  config.llm_hidden_size = 4;
  config.llm_vocab_size = 8;
  config.max_new_tokens = 2;
  config.generate_audio = false;
  config.do_sample = false;

  auto llm_runner = std::make_shared<NiceMock<MockLiteRtLmRunner>>();
  ON_CALL(*llm_runner, Prefill(_)).WillByDefault([](const lm::ExecutorInputs&) {
    return absl::OkStatus();
  });
  ASSERT_OK_AND_ASSIGN(
      auto stage,
      QwenOmniLlmStage::Create(&frontend, std::make_shared<FakeTokenizer>(),
                               llm_runner, config));

  // Empty prompt fails with InvalidArgument.
  frontend.Push(Any2AnyFrontendOutput{
      .prompt_embeddings = {}, .prompt_len = 0, .generate_audio = false});
  EXPECT_THAT(stage->Schedule(), StatusIs(absl::StatusCode::kInvalidArgument));

  // Mismatched prompt_embeddings size fails with InvalidArgument.
  frontend.Push(Any2AnyFrontendOutput{
      .prompt_embeddings = {1.0f, 2.0f},
      .prompt_len = 2,
      .generate_audio = false,
  });
  EXPECT_THAT(stage->Schedule(), StatusIs(absl::StatusCode::kInvalidArgument));

  // Float16 decode output works and immediate stop token produces empty text.
  ON_CALL(*llm_runner, Decode(_)).WillByDefault([](const lm::ExecutorInputs&) {
    std::vector<float> logits(8, -5.0f);
    logits[2] = 5.0f;  // <|im_end|>
    return MakeFloat16TensorBuffer(logits);
  });
  frontend.Push(Any2AnyFrontendOutput{
      .prompt_embeddings = std::vector<float>(4, 0.1f),
      .prompt_len = 1,
      .token_ids = {1},
      .generate_audio = false,
  });
  ASSERT_OK(stage->Schedule());
  ASSERT_OK_AND_ASSIGN(auto fp16_out, stage->GetOutput());
  EXPECT_TRUE(fp16_out.generated_token_ids.empty());
  EXPECT_TRUE(fp16_out.text.empty());

  // Unsupported decode buffer element type (Int32) fails with InvalidArgument.
  ON_CALL(*llm_runner, Decode(_))
      .WillByDefault(
          [](const lm::ExecutorInputs&) -> absl::StatusOr<TensorBuffer> {
            LITERT_ASSIGN_OR_RETURN(
                auto buf, support::CreateTensorBuffer<int32_t>({1, 1, 8}));
            return buf;
          });
  frontend.Push(Any2AnyFrontendOutput{
      .prompt_embeddings = std::vector<float>(4, 0.1f),
      .prompt_len = 1,
      .generate_audio = false,
  });
  EXPECT_THAT(stage->Schedule(), StatusIs(absl::StatusCode::kInvalidArgument));

  // Mismatched decode output size (neither vocab_size nor vocab_size +
  // hidden_dim) fails with InvalidArgument.
  ON_CALL(*llm_runner, Decode(_)).WillByDefault([](const lm::ExecutorInputs&) {
    return MakeFloatTensorBuffer(std::vector<float>(5, 0.0f));
  });
  frontend.Push(Any2AnyFrontendOutput{
      .prompt_embeddings = std::vector<float>(4, 0.1f),
      .prompt_len = 1,
      .generate_audio = false,
  });
  EXPECT_THAT(stage->Schedule(), StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(QwenOmniLlmStageTest, SamplingTopKTopPNonFiniteLogitsAndResetDeterminism) {
  FakeFrontendStage frontend;
  QwenOmniModelConfig config;
  config.llm_hidden_size = 4;
  config.llm_vocab_size = 8;
  config.max_new_tokens = 1;
  config.generate_audio = false;
  config.do_sample = true;
  config.temperature = 0.8f;
  config.top_k = 3;
  config.top_p = 0.7f;
  config.seed = 12345;

  auto llm_runner = std::make_shared<NiceMock<MockLiteRtLmRunner>>();
  ON_CALL(*llm_runner, Prefill(_)).WillByDefault([](const lm::ExecutorInputs&) {
    return absl::OkStatus();
  });
  EXPECT_CALL(*llm_runner, Reset()).WillOnce([]() { return absl::OkStatus(); });

  const float kNaN = std::numeric_limits<float>::quiet_NaN();
  const float kInf = std::numeric_limits<float>::infinity();

  // Mixed finite + NaN + -Inf logits across 4 equal-logit non-stop candidates
  // {3, 4, 5, 7} over 8 tokens: sampled tokens must never be NaN/-Inf indices,
  // consecutive runs without Reset() diverge, and Reset() deterministically
  // reproduces the exact 8-token sequence.
  QwenOmniModelConfig reset_cfg = config;
  reset_cfg.max_new_tokens = 8;
  reset_cfg.top_k = 4;
  reset_cfg.top_p = 0.99f;
  ON_CALL(*llm_runner, Decode(_))
      .WillByDefault([kNaN, kInf](const lm::ExecutorInputs&) {
        std::vector<float> logits = {kNaN, -kInf, -10.0f, 4.0f,
                                     4.0f, 4.0f,  kNaN,   4.0f};
        return MakeFloatTensorBuffer(logits);
      });

  ASSERT_OK_AND_ASSIGN(
      auto reset_stage,
      QwenOmniLlmStage::Create(&frontend, std::make_shared<FakeTokenizer>(),
                               llm_runner, reset_cfg));

  auto run_once = [&](QwenOmniLlmStage& s) {
    frontend.Push(Any2AnyFrontendOutput{
        .prompt_embeddings = std::vector<float>(4, 0.1f),
        .prompt_len = 1,
        .generate_audio = false,
    });
    EXPECT_OK(s.Schedule());
    return s.GetOutput();
  };

  ASSERT_OK_AND_ASSIGN(auto sampled_out1, run_once(*reset_stage));
  ASSERT_EQ(sampled_out1.generated_token_ids.size(), 8);
  for (int tok : sampled_out1.generated_token_ids) {
    EXPECT_THAT(tok, AnyOf(3, 4, 5, 7));
  }

  ASSERT_OK_AND_ASSIGN(auto unreset_out, run_once(*reset_stage));
  ASSERT_EQ(unreset_out.generated_token_ids.size(), 8);
  EXPECT_NE(unreset_out.generated_token_ids, sampled_out1.generated_token_ids);

  // Reset reseeds RNG deterministically and resets runner.
  reset_stage->Reset();
  ASSERT_OK_AND_ASSIGN(auto sampled_out2, run_once(*reset_stage));
  EXPECT_EQ(sampled_out2.generated_token_ids, sampled_out1.generated_token_ids);

  ASSERT_OK_AND_ASSIGN(
      auto stage,
      QwenOmniLlmStage::Create(&frontend, std::make_shared<FakeTokenizer>(),
                               llm_runner, config));

  // Top-p truncation with mixed finite + NaN + -Inf logits.
  ON_CALL(*llm_runner, Decode(_))
      .WillByDefault([kNaN, kInf](const lm::ExecutorInputs&) {
        std::vector<float> logits = {kNaN, -kInf,  -10.0f, 4.0f,
                                     3.8f, -10.0f, kNaN,   -10.0f};
        return MakeFloatTensorBuffer(logits);
      });
  ASSERT_OK_AND_ASSIGN(auto top_p_out, run_once(*stage));
  ASSERT_EQ(top_p_out.generated_token_ids.size(), 1);
  EXPECT_THAT(top_p_out.generated_token_ids[0], AnyOf(3, 4));

  // +Inf logit deterministically selects the +Inf token without NaN/UB.
  ON_CALL(*llm_runner, Decode(_))
      .WillByDefault([kNaN, kInf](const lm::ExecutorInputs&) {
        std::vector<float> logits = {kNaN, 1.0f, -10.0f, 2.0f,
                                     kInf, 0.0f, 0.0f,   0.0f};
        return MakeFloatTensorBuffer(logits);
      });
  frontend.Push(Any2AnyFrontendOutput{
      .prompt_embeddings = std::vector<float>(4, 0.1f),
      .prompt_len = 1,
      .generate_audio = false,
  });
  ASSERT_OK(stage->Schedule());
  ASSERT_OK_AND_ASSIGN(auto inf_out, stage->GetOutput());
  EXPECT_THAT(inf_out.generated_token_ids, ElementsAre(4));

  // All non-finite logits safely fall back to token 0 (which is <|endoftext|>,
  // a stop token, producing empty generated_token_ids without UB).
  ON_CALL(*llm_runner, Decode(_))
      .WillByDefault([kNaN, kInf](const lm::ExecutorInputs&) {
        std::vector<float> logits(8, kNaN);
        logits[1] = -kInf;
        return MakeFloatTensorBuffer(logits);
      });
  frontend.Push(Any2AnyFrontendOutput{
      .prompt_embeddings = std::vector<float>(4, 0.1f),
      .prompt_len = 1,
      .generate_audio = false,
  });
  ASSERT_OK(stage->Schedule());
  ASSERT_OK_AND_ASSIGN(auto all_nan_out, stage->GetOutput());
  EXPECT_TRUE(all_nan_out.generated_token_ids.empty());

  // Also test greedy mode with leading NaN and +Inf, plus top_k == 0 sampling.
  QwenOmniModelConfig greedy_cfg = config;
  greedy_cfg.do_sample = false;
  ASSERT_OK_AND_ASSIGN(
      auto greedy_stage,
      QwenOmniLlmStage::Create(&frontend, std::make_shared<FakeTokenizer>(),
                               llm_runner, greedy_cfg));
  ON_CALL(*llm_runner, Decode(_))
      .WillByDefault([kNaN](const lm::ExecutorInputs&) {
        std::vector<float> logits = {kNaN, -5.0f, -10.0f, 3.0f,
                                     1.0f, -5.0f, -5.0f,  -5.0f};
        return MakeFloatTensorBuffer(logits);
      });
  frontend.Push(Any2AnyFrontendOutput{
      .prompt_embeddings = std::vector<float>(4, 0.1f),
      .prompt_len = 1,
      .generate_audio = false,
  });
  ASSERT_OK(greedy_stage->Schedule());
  ASSERT_OK_AND_ASSIGN(auto greedy_nan_out, greedy_stage->GetOutput());
  EXPECT_THAT(greedy_nan_out.generated_token_ids, ElementsAre(3));

  ON_CALL(*llm_runner, Decode(_))
      .WillByDefault([kNaN, kInf](const lm::ExecutorInputs&) {
        std::vector<float> logits = {kNaN, 1.0f, -10.0f, 2.0f,
                                     kInf, 0.0f, 0.0f,   0.0f};
        return MakeFloatTensorBuffer(logits);
      });
  frontend.Push(Any2AnyFrontendOutput{
      .prompt_embeddings = std::vector<float>(4, 0.1f),
      .prompt_len = 1,
      .generate_audio = false,
  });
  ASSERT_OK(greedy_stage->Schedule());
  ASSERT_OK_AND_ASSIGN(auto greedy_inf_out, greedy_stage->GetOutput());
  EXPECT_THAT(greedy_inf_out.generated_token_ids, ElementsAre(4));

  QwenOmniModelConfig full_vocab_sample_cfg = config;
  full_vocab_sample_cfg.top_k = 0;
  full_vocab_sample_cfg.top_p = 1.0f;
  ASSERT_OK_AND_ASSIGN(
      auto full_sample_stage,
      QwenOmniLlmStage::Create(&frontend, std::make_shared<FakeTokenizer>(),
                               llm_runner, full_vocab_sample_cfg));
  ON_CALL(*llm_runner, Decode(_)).WillByDefault([](const lm::ExecutorInputs&) {
    std::vector<float> logits(8, -100.0f);
    logits[3] = 10.0f;
    return MakeFloatTensorBuffer(logits);
  });
  frontend.Push(Any2AnyFrontendOutput{
      .prompt_embeddings = std::vector<float>(4, 0.1f),
      .prompt_len = 1,
      .generate_audio = false,
  });
  ASSERT_OK(full_sample_stage->Schedule());
  ASSERT_OK_AND_ASSIGN(auto full_sample_out, full_sample_stage->GetOutput());
  EXPECT_THAT(full_sample_out.generated_token_ids, ElementsAre(3));
}

}  // namespace
}  // namespace litert::omni::any2any
