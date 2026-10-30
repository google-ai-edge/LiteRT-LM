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

#include "omni/any2any/minicpm_o/minicpm_o_llm_stage.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
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
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/cc/litert_tensor_buffer_types.h"  // from @litert
#include "omni/any2any/minicpm_o/minicpm_o_io_types.h"
#include "omni/any2any/minicpm_o/minicpm_o_model_config.h"
#include "omni/base/mock_litert_lm_runner.h"
#include "omni/base/stage.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "support/tokenizer/tokenizer.h"
#include "support/util/convert_tensor_buffer.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace litert::omni::any2any {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::_;
using ::testing::NiceMock;

class FakeFrontendStage
    : public SingleThreadedStageWithDeque<MiniCpmOFrontendOutput> {
 public:
  void Push(MiniCpmOFrontendOutput prompt) { PushOutput(std::move(prompt)); }

 protected:
  bool NeedScheduleInternal() const override { return false; }
  absl::Status ScheduleInternal() override { return absl::OkStatus(); }
};

class FakeTokenizer : public support::Tokenizer {
 public:
  support::TokenizerType GetTokenizerType() const override {
    return support::TokenizerType::kHuggingFace;
  }

  absl::StatusOr<support::TokenIds> TextToTokenIds(
      absl::string_view text) override {
    if (text == "\n") {
      return support::TokenIds{8};
    }
    return absl::NotFoundError("not implemented");
  }

  absl::StatusOr<int> TokenToId(absl::string_view token) override {
    if (token == "<|im_end|>") {
      return 2;
    }
    if (token == "</s>") {
      return 7;
    }
    if (token == "<think>") {
      return 5;
    }
    if (token == "</think>") {
      return 6;
    }
    if (token == "\n") {
      return 8;
    }
    return absl::NotFoundError("not found");
  }

  absl::StatusOr<std::string> TokenIdsToText(
      absl::Span<const int> token_ids, bool skip_special_tokens) override {
    if (custom_text_.has_value()) {
      return *custom_text_;
    }
    std::string out;
    for (size_t i = 0; i < token_ids.size(); ++i) {
      if (i > 0) absl::StrAppend(&out, " ");
      if (token_ids[i] == 3) {
        absl::StrAppend(&out, "Hello");
      } else if (token_ids[i] == 4) {
        absl::StrAppend(&out, "world");
      } else if (token_ids[i] == 5) {
        absl::StrAppend(&out, "<think>");
      } else if (token_ids[i] == 6) {
        absl::StrAppend(&out, "</think>");
      } else {
        absl::StrAppend(&out, "tok", token_ids[i]);
      }
    }
    return out;
  }

  std::vector<std::string> GetTokens() const override { return {}; }

  int GetVocabSize() const override { return 16; }

  void SetCustomText(std::string text) { custom_text_ = std::move(text); }

 private:
  std::optional<std::string> custom_text_;
};

TEST(MiniCpmOLlmStageTest, CreateValidatesParameters) {
  // Arrange
  FakeFrontendStage frontend;
  auto tokenizer = std::make_shared<FakeTokenizer>();
  MiniCpmOModelConfig config;
  config.llm_hidden_size = 4;
  config.llm_vocab_size = 8;

  // Act & Assert: non-positive llm_hidden_size fails with InvalidArgumentError.
  MiniCpmOModelConfig bad_hidden_config = config;
  bad_hidden_config.llm_hidden_size = 0;
  EXPECT_THAT(
      MiniCpmOLlmStage::Create(
          &frontend, tokenizer,
          std::make_shared<NiceMock<MockLiteRtLmRunner>>(), bad_hidden_config),
      StatusIs(absl::StatusCode::kInvalidArgument));

  // Act & Assert: non-positive llm_vocab_size fails with InvalidArgumentError.
  MiniCpmOModelConfig bad_vocab_config = config;
  bad_vocab_config.llm_vocab_size = 0;
  EXPECT_THAT(
      MiniCpmOLlmStage::Create(
          &frontend, tokenizer,
          std::make_shared<NiceMock<MockLiteRtLmRunner>>(), bad_vocab_config),
      StatusIs(absl::StatusCode::kInvalidArgument));

  // Act & Assert: valid parameters succeed.
  auto llm_runner = std::make_shared<NiceMock<MockLiteRtLmRunner>>();
  EXPECT_THAT(
      MiniCpmOLlmStage::Create(&frontend, tokenizer, llm_runner, config),
      IsOk());
}

TEST(MiniCpmOLlmStageTest, NeedScheduleAndPromptValidation) {
  // Arrange
  FakeFrontendStage frontend;
  MiniCpmOModelConfig config;
  config.llm_hidden_size = 4;
  config.llm_vocab_size = 8;
  auto llm_runner = std::make_shared<NiceMock<MockLiteRtLmRunner>>();
  ASSERT_OK_AND_ASSIGN(
      auto stage, MiniCpmOLlmStage::Create(
                      &frontend, std::make_shared<FakeTokenizer>(), llm_runner,
                      config));

  // Act & Assert: no output in frontend -> NeedSchedule() is false and
  // Schedule() returns NotFoundError.
  EXPECT_FALSE(stage->NeedSchedule());
  EXPECT_THAT(stage->Schedule(), StatusIs(absl::StatusCode::kNotFound));

  // Arrange: push an empty prompt.
  frontend.Push(MiniCpmOFrontendOutput{.prompt_len = 0});
  EXPECT_TRUE(stage->NeedSchedule());

  // Act & Assert: empty prompt returns InvalidArgumentError.
  EXPECT_THAT(stage->Schedule(), StatusIs(absl::StatusCode::kInvalidArgument));

  // Arrange: push a prompt whose embeddings size does not match
  // prompt_len * llm_hidden_size.
  MiniCpmOFrontendOutput misaligned_prompt;
  misaligned_prompt.prompt_embeddings = {1.0f, 2.0f};
  misaligned_prompt.prompt_len = 1;
  frontend.Push(std::move(misaligned_prompt));

  // Act & Assert: misaligned prompt returns InvalidArgumentError.
  EXPECT_THAT(stage->Schedule(), StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(MiniCpmOLlmStageTest,
     GenerateWithLmRunnerGreedyDecodingProducesExpectedTokens) {
  // Arrange
  FakeFrontendStage frontend;
  MiniCpmOModelConfig config;
  config.llm_hidden_size = 4;
  config.llm_vocab_size = 8;
  config.max_new_tokens = 4;
  config.do_sample = false;

  auto llm_runner = std::make_shared<NiceMock<MockLiteRtLmRunner>>();
  ON_CALL(*llm_runner, Prefill(_)).WillByDefault([](const lm::ExecutorInputs&) {
    return absl::OkStatus();
  });
  std::vector<int32_t> observed_input_tokens;
  int decode_step = 0;
  ON_CALL(*llm_runner, Decode(_))
      .WillByDefault([&decode_step,
                      &observed_input_tokens](const lm::ExecutorInputs& inputs)
                         -> absl::StatusOr<TensorBuffer> {
        ABSL_ASSIGN_OR_RETURN(const TensorBuffer* tok_buf,
                              inputs.GetTextTokenIdsPtr());
        int32_t tok = -999;
        LITERT_RETURN_IF_ERROR(
            const_cast<TensorBuffer*>(tok_buf)->Read<int32_t>(
                absl::MakeSpan(&tok, 1)));
        observed_input_tokens.push_back(tok);

        std::vector<float> logits(8, -10.0f);
        if (decode_step == 0) {
          logits[3] = 10.0f;  // "Hello"
        } else if (decode_step == 1) {
          logits[4] = 10.0f;  // "world"
        } else {
          logits[2] = 10.0f;  // "<|im_end|>" stop token
        }
        ++decode_step;
        LITERT_ASSIGN_OR_RETURN(auto buf,
                                support::CreateTensorBuffer<float>({1, 8}));
        LITERT_RETURN_IF_ERROR(buf.Write<float>(absl::MakeConstSpan(logits)));
        return buf;
      });

  ASSERT_OK_AND_ASSIGN(
      auto stage, MiniCpmOLlmStage::Create(
                      &frontend, std::make_shared<FakeTokenizer>(), llm_runner,
                      config));

  MiniCpmOFrontendOutput prompt;
  prompt.prompt_embeddings.assign(2 * config.llm_hidden_size, 0.5f);
  prompt.prompt_len = 2;
  prompt.generate_audio = false;
  frontend.Push(std::move(prompt));

  // Act
  ASSERT_OK(stage->Schedule());

  // Assert: Decode receives -1 (last prompt token), 3 ("Hello"), 4 ("world"),
  // followed by the turn-closing tokens 2 ("<|im_end|>") and 8 ("\n").
  EXPECT_EQ(observed_input_tokens, (std::vector<int32_t>{-1, 3, 4, 2, 8}));
  ASSERT_TRUE(stage->HasOutput());
  ASSERT_OK_AND_ASSIGN(MiniCpmOLlmOutput out, stage->GetOutput());
  EXPECT_EQ(out.text, "Hello world");
  EXPECT_EQ(out.generated_token_ids, (std::vector<int>{3, 4}));
  EXPECT_EQ(out.num_tokens, 2);
  EXPECT_FALSE(out.generate_audio);
  EXPECT_TRUE(out.is_final);
  EXPECT_TRUE(out.llm_hidden_states.empty());
}

TEST(MiniCpmOLlmStageTest,
     GenerateWithLmRunnerPrefillsFullPromptAndExtractsCombinedHiddenState) {
  // Arrange
  FakeFrontendStage frontend;
  auto llm_runner = std::make_shared<NiceMock<MockLiteRtLmRunner>>();

  MiniCpmOModelConfig config;
  config.llm_hidden_size = 4;
  config.llm_vocab_size = 8;
  config.max_new_tokens = 2;
  config.do_sample = false;

  std::vector<size_t> prefill_call_tokens;
  ON_CALL(*llm_runner, Prefill(_))
      .WillByDefault([&prefill_call_tokens](
                         const lm::ExecutorInputs& inputs) -> absl::Status {
        ABSL_ASSIGN_OR_RETURN(const TensorBuffer* tok_buf,
                              inputs.GetTextTokenIdsPtr());
        LITERT_ASSIGN_OR_RETURN(size_t bytes, tok_buf->PackedSize());
        prefill_call_tokens.push_back(bytes / sizeof(int32_t));
        return absl::OkStatus();
      });

  int decode_calls = 0;
  ON_CALL(*llm_runner, Decode(_))
      .WillByDefault([&decode_calls](const lm::ExecutorInputs&)
                         -> absl::StatusOr<TensorBuffer> {
        ++decode_calls;
        if (decode_calls == 1) {
          // Return combined [vocab_size (8) + hidden_dim (4) = 12] floats.
          std::vector<float> combined(12, -5.0f);
          combined[3] = 9.0f;  // "Hello"
          combined[8] = 1.5f;
          combined[9] = 2.5f;
          combined[10] = 3.5f;
          combined[11] = 4.5f;
          LITERT_ASSIGN_OR_RETURN(auto buf,
                                  support::CreateTensorBuffer<float>({1, 12}));
          LITERT_RETURN_IF_ERROR(
              buf.Write<float>(absl::MakeConstSpan(combined)));
          return buf;
        }
        // Stop token "</s>" (id 7 via TokenToId).
        std::vector<float> stop_logits(8, -5.0f);
        stop_logits[7] = 9.0f;
        LITERT_ASSIGN_OR_RETURN(auto buf,
                                support::CreateTensorBuffer<float>({1, 8}));
        LITERT_RETURN_IF_ERROR(
            buf.Write<float>(absl::MakeConstSpan(stop_logits)));
        return buf;
      });

  ASSERT_OK_AND_ASSIGN(
      auto stage, MiniCpmOLlmStage::Create(
                      &frontend, std::make_shared<FakeTokenizer>(), llm_runner,
                      config));

  // All 5 prompt tokens are passed in a single Prefill call.
  MiniCpmOFrontendOutput prompt;
  prompt.prompt_embeddings.assign(5 * config.llm_hidden_size, 0.25f);
  prompt.prompt_len = 5;
  prompt.generate_audio = true;
  frontend.Push(std::move(prompt));

  // Act
  ASSERT_OK(stage->Schedule());

  // Assert: 1 Prefill call of 5 tokens, 2 generation Decode calls + 2
  // turn-closing Decode calls (<|im_end|> and \n).
  EXPECT_EQ(prefill_call_tokens, (std::vector<size_t>{5}));
  EXPECT_EQ(decode_calls, 4);
  ASSERT_OK_AND_ASSIGN(MiniCpmOLlmOutput out, stage->GetOutput());
  EXPECT_EQ(out.text, "Hello");
  EXPECT_EQ(out.generated_token_ids, std::vector<int>{3});
  EXPECT_EQ(out.llm_hidden_states,
            (std::vector<float>{1.5f, 2.5f, 3.5f, 4.5f}));
}

TEST(MiniCpmOLlmStageTest,
     MultiTurnConversationCommitsTurnEndTokensBeforeNextTurnPrefill) {
  // Arrange
  FakeFrontendStage frontend;
  auto llm_runner = std::make_shared<NiceMock<MockLiteRtLmRunner>>();

  MiniCpmOModelConfig config;
  config.llm_hidden_size = 4;
  config.llm_vocab_size = 8;
  config.max_new_tokens = 2;
  config.do_sample = false;

  // Record the sequence of events ("P:<num_tokens>" for Prefill and
  // "D:<input_token>" for Decode) to verify that Turn 1's final generated
  // token and `<|im_end|>\n` (tokens 2 and 8) land in the KV cache before
  // Turn 2 is prefilled.
  std::vector<std::string> call_trace;
  ON_CALL(*llm_runner, Prefill(_))
      .WillByDefault([&call_trace](
                         const lm::ExecutorInputs& inputs) -> absl::Status {
        ABSL_ASSIGN_OR_RETURN(const TensorBuffer* tok_buf,
                              inputs.GetTextTokenIdsPtr());
        LITERT_ASSIGN_OR_RETURN(size_t bytes, tok_buf->PackedSize());
        const size_t num_tokens = bytes / sizeof(int32_t);
        call_trace.push_back(absl::StrCat("P:", num_tokens));
        return absl::OkStatus();
      });

  int decode_call_idx = 0;
  ON_CALL(*llm_runner, Decode(_))
      .WillByDefault([&call_trace, &decode_call_idx](
                         const lm::ExecutorInputs& inputs)
                         -> absl::StatusOr<TensorBuffer> {
        ABSL_ASSIGN_OR_RETURN(const TensorBuffer* tok_buf,
                              inputs.GetTextTokenIdsPtr());
        int32_t tok = -999;
        LITERT_RETURN_IF_ERROR(
            const_cast<TensorBuffer*>(tok_buf)->Read<int32_t>(
                absl::MakeSpan(&tok, 1)));
        call_trace.push_back(absl::StrCat("D:", tok));

        std::vector<float> logits(8, -10.0f);
        // Turn 1 hits max_new_tokens = 2 without sampling a stop token:
        //   call 0 (D:-1) -> samples 3 ("Hello")
        //   call 1 (D:3)  -> samples 4 ("world")
        //   call 2..4     -> CommitAssistantTurnEnd decodes 4, 2 (<|im_end|>),
        //                    8 ("\n")
        // Turn 2 stops on <|im_end|>:
        //   call 5 (D:-1) -> samples 3 ("Hello")
        //   call 6 (D:3)  -> samples 2 ("<|im_end|>")
        //   call 7..8     -> CommitAssistantTurnEnd decodes 2 (<|im_end|>),
        //                    8 ("\n")
        if (decode_call_idx == 0 || decode_call_idx == 5) {
          logits[3] = 10.0f;  // "Hello"
        } else if (decode_call_idx == 1) {
          logits[4] = 10.0f;  // "world"
        } else {
          logits[2] = 10.0f;  // "<|im_end|>"
        }
        ++decode_call_idx;
        LITERT_ASSIGN_OR_RETURN(auto buf,
                                support::CreateTensorBuffer<float>({1, 8}));
        LITERT_RETURN_IF_ERROR(buf.Write<float>(absl::MakeConstSpan(logits)));
        return buf;
      });

  ASSERT_OK_AND_ASSIGN(
      auto stage, MiniCpmOLlmStage::Create(
                      &frontend, std::make_shared<FakeTokenizer>(), llm_runner,
                      config));

  // Act: Turn 1 (hits max_new_tokens = 2).
  MiniCpmOFrontendOutput turn1_prompt;
  turn1_prompt.prompt_embeddings.assign(3 * config.llm_hidden_size, 1.0f);
  turn1_prompt.prompt_len = 3;
  turn1_prompt.generate_audio = false;
  frontend.Push(std::move(turn1_prompt));
  ASSERT_OK(stage->Schedule());
  ASSERT_OK_AND_ASSIGN(MiniCpmOLlmOutput turn1_out, stage->GetOutput());
  EXPECT_EQ(turn1_out.text, "Hello world");
  EXPECT_EQ(turn1_out.generated_token_ids, (std::vector<int>{3, 4}));

  // Act: Turn 2 (without Reset(); stops on <|im_end|>).
  MiniCpmOFrontendOutput turn2_prompt;
  turn2_prompt.prompt_embeddings.assign(4 * config.llm_hidden_size, 2.0f);
  turn2_prompt.prompt_len = 4;
  turn2_prompt.generate_audio = false;
  frontend.Push(std::move(turn2_prompt));
  ASSERT_OK(stage->Schedule());
  ASSERT_OK_AND_ASSIGN(MiniCpmOLlmOutput turn2_out, stage->GetOutput());
  EXPECT_EQ(turn2_out.text, "Hello");
  EXPECT_EQ(turn2_out.generated_token_ids, (std::vector<int>{3}));

  // Assert: Turn 1 commits token 4 ("world"), 2 ("<|im_end|>"), and 8 ("\n")
  // before Turn 2 ("P:4") is prefilled, and Turn 2 commits 2 ("<|im_end|>")
  // and 8 ("\n") at the end of Turn 2.
  EXPECT_EQ(call_trace,
            (std::vector<std::string>{"P:3", "D:-1", "D:3", "D:4", "D:2",
                                      "D:8", "P:4", "D:-1", "D:3", "D:2",
                                      "D:8"}));
}

TEST(MiniCpmOLlmStageTest,
     GenerateWithLmRunnerDecodesFloat16LogitsAndStripsThinkingTokensAndStates) {
  // Arrange
  FakeFrontendStage frontend;
  MiniCpmOModelConfig config;
  config.llm_hidden_size = 4;
  config.llm_vocab_size = 8;
  config.max_new_tokens = 4;
  config.do_sample = false;

  auto env = Environment::Create({});
  ASSERT_TRUE(env.HasValue());
  auto tokenizer = std::make_shared<FakeTokenizer>();

  auto llm_runner = std::make_shared<NiceMock<MockLiteRtLmRunner>>();
  ON_CALL(*llm_runner, Prefill(_)).WillByDefault([](const lm::ExecutorInputs&) {
    return absl::OkStatus();
  });
  int decode_step = 0;
  ON_CALL(*llm_runner, Decode(_))
      .WillByDefault([&decode_step, &env](const lm::ExecutorInputs&)
                         -> absl::StatusOr<TensorBuffer> {
        // 8 Float16 logit bit patterns + 4 Float16 hidden-state values:
        // 0x0000 (+0.0), 0xBC00 (-1.0), 0x3C00 (+1.0), 0x4000 (+2.0).
        std::vector<uint16_t> fp16_bits = {
            0x0000, 0xBC00, 0xBC00, 0xBC00, 0xBC00, 0xBC00,
            0xBC00, 0xBC00, 0x3C00, 0x3C00, 0x3C00, 0x3C00};
        if (decode_step == 0) {
          fp16_bits[5] = 0x4000;  // "<think>" (token 5)
        } else if (decode_step == 1) {
          fp16_bits[3] = 0x4000;  // "Hello" inside think block
        } else if (decode_step == 2) {
          fp16_bits[6] = 0x4000;  // "</think>" (token 6)
        } else {
          fp16_bits[4] = 0x4000;  // "world" after think block
          // Distinct hidden state (+2.0f) for the kept post-think token.
          fp16_bits[8] = 0x4000;
          fp16_bits[9] = 0x4000;
          fp16_bits[10] = 0x4000;
          fp16_bits[11] = 0x4000;
        }
        ++decode_step;
        RankedTensorType fp16_type(ElementType::Float16,
                                   Layout(Dimensions({1, 12})));
        LITERT_ASSIGN_OR_RETURN(
            auto buf,
            TensorBuffer::CreateManaged(*env, TensorBufferType::kHostMemory,
                                        fp16_type,
                                        fp16_bits.size() * sizeof(uint16_t)));
        LITERT_RETURN_IF_ERROR(
            buf.Write<uint16_t>(absl::MakeConstSpan(fp16_bits)));
        return buf;
      });

  ASSERT_OK_AND_ASSIGN(
      auto stage, MiniCpmOLlmStage::Create(&frontend, tokenizer, llm_runner,
                                           config));

  MiniCpmOFrontendOutput prompt;
  prompt.prompt_embeddings.assign(config.llm_hidden_size, 0.1f);
  prompt.prompt_len = 1;
  prompt.generate_audio = true;
  frontend.Push(prompt);

  // Act
  ASSERT_OK(stage->Schedule());

  // Assert: tokens 5 (<think>), 3 (Hello), 6 (</think>) and their hidden
  // states are stripped, leaving only token 4 ("world").
  ASSERT_OK_AND_ASSIGN(MiniCpmOLlmOutput out, stage->GetOutput());
  EXPECT_EQ(out.generated_token_ids, std::vector<int>{4});
  EXPECT_EQ(out.num_tokens, 1);
  EXPECT_EQ(out.text, "world");
  EXPECT_EQ(out.llm_hidden_states,
            (std::vector<float>{2.0f, 2.0f, 2.0f, 2.0f}));

  // Also verify string-level <think>...</think> stripping when emitted inside
  // detokenized text without discrete <think> token IDs.
  tokenizer->SetCustomText("<think>\ninternal reasoning\n</think>\nClean text");
  frontend.Push(std::move(prompt));
  ASSERT_OK(stage->Schedule());
  ASSERT_OK_AND_ASSIGN(MiniCpmOLlmOutput out2, stage->GetOutput());
  EXPECT_EQ(out2.text, "Clean text");
}

TEST(MiniCpmOLlmStageTest,
     GenerateWithLmRunnerRejectsInt16LogitsAndMissingAudioHiddenStates) {
  // Arrange
  FakeFrontendStage frontend;
  MiniCpmOModelConfig config;
  config.llm_hidden_size = 4;
  config.llm_vocab_size = 8;
  config.max_new_tokens = 1;
  config.do_sample = false;

  auto llm_runner = std::make_shared<NiceMock<MockLiteRtLmRunner>>();
  ON_CALL(*llm_runner, Prefill(_)).WillByDefault([](const lm::ExecutorInputs&) {
    return absl::OkStatus();
  });
  enum class DecodeOutputMode {
    kInt16,
    kMismatchedFloatLength,
    kLogitsOnlyFloat,
  };
  DecodeOutputMode mode = DecodeOutputMode::kInt16;
  ON_CALL(*llm_runner, Decode(_))
      .WillByDefault([&mode](const lm::ExecutorInputs&)
                         -> absl::StatusOr<TensorBuffer> {
        if (mode == DecodeOutputMode::kInt16) {
          std::vector<int16_t> i16(8, 1);
          LITERT_ASSIGN_OR_RETURN(auto buf,
                                  support::CreateTensorBuffer<int16_t>({1, 8}));
          LITERT_RETURN_IF_ERROR(buf.Write<int16_t>(absl::MakeConstSpan(i16)));
          return buf;
        }
        if (mode == DecodeOutputMode::kMismatchedFloatLength) {
          // Size 10 matches neither llm_vocab_size (8) nor
          // llm_vocab_size + llm_hidden_size (12).
          std::vector<float> bad_len(10, 1.0f);
          LITERT_ASSIGN_OR_RETURN(auto buf,
                                  support::CreateTensorBuffer<float>({1, 10}));
          LITERT_RETURN_IF_ERROR(
              buf.Write<float>(absl::MakeConstSpan(bad_len)));
          return buf;
        }
        std::vector<float> logits(8, -10.0f);
        logits[3] = 10.0f;
        LITERT_ASSIGN_OR_RETURN(auto buf,
                                support::CreateTensorBuffer<float>({1, 8}));
        LITERT_RETURN_IF_ERROR(buf.Write<float>(absl::MakeConstSpan(logits)));
        return buf;
      });

  ASSERT_OK_AND_ASSIGN(
      auto stage, MiniCpmOLlmStage::Create(
                      &frontend, std::make_shared<FakeTokenizer>(), llm_runner,
                      config));

  MiniCpmOFrontendOutput prompt;
  prompt.prompt_embeddings.assign(config.llm_hidden_size, 0.1f);
  prompt.prompt_len = 1;
  prompt.generate_audio = true;
  frontend.Push(prompt);

  // Act & Assert: Int16 TensorBuffer is rejected with InvalidArgumentError.
  EXPECT_THAT(stage->Schedule(), StatusIs(absl::StatusCode::kInvalidArgument));

  // Act & Assert: when Decode output size matches neither llm_vocab_size nor
  // llm_vocab_size + llm_hidden_size, returns InvalidArgumentError even when
  // generate_audio is false.
  mode = DecodeOutputMode::kMismatchedFloatLength;
  MiniCpmOFrontendOutput text_only_prompt = prompt;
  text_only_prompt.generate_audio = false;
  frontend.Push(std::move(text_only_prompt));
  EXPECT_THAT(stage->Schedule(), StatusIs(absl::StatusCode::kInvalidArgument));

  // Act & Assert: when generate_audio is true and combined hidden states are
  // absent from `llm_runner` output, returns FailedPreconditionError.
  mode = DecodeOutputMode::kLogitsOnlyFloat;
  frontend.Push(std::move(prompt));
  EXPECT_THAT(stage->Schedule(),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(MiniCpmOLlmStageTest,
     StochasticSamplingAndResetProduceDeterministicReproducibleTokens) {
  // Arrange
  FakeFrontendStage frontend;
  MiniCpmOModelConfig config;
  config.llm_hidden_size = 4;
  config.llm_vocab_size = 8;
  config.max_new_tokens = 2;
  config.do_sample = true;
  config.temperature = 0.7f;
  config.top_k = 2;
  config.top_p = 0.6f;
  config.seed = 12345;

  int reset_calls = 0;
  auto llm_runner = std::make_shared<NiceMock<MockLiteRtLmRunner>>();
  ON_CALL(*llm_runner, Reset()).WillByDefault([&reset_calls]() {
    ++reset_calls;
    return absl::OkStatus();
  });
  ON_CALL(*llm_runner, Prefill(_)).WillByDefault([](const lm::ExecutorInputs&) {
    return absl::OkStatus();
  });
  ON_CALL(*llm_runner, Decode(_))
      .WillByDefault(
          [](const lm::ExecutorInputs&) -> absl::StatusOr<TensorBuffer> {
            std::vector<float> logits = {-100.0f, -100.0f, -100.0f, 5.0f,
                                         4.9f,    -100.0f, -100.0f, -100.0f};
            LITERT_ASSIGN_OR_RETURN(auto buf,
                                    support::CreateTensorBuffer<float>({1, 8}));
            LITERT_RETURN_IF_ERROR(
                buf.Write<float>(absl::MakeConstSpan(logits)));
            return buf;
          });

  ASSERT_OK_AND_ASSIGN(
      auto stage, MiniCpmOLlmStage::Create(
                      &frontend, std::make_shared<FakeTokenizer>(), llm_runner,
                      config));

  MiniCpmOFrontendOutput prompt1;
  prompt1.prompt_embeddings.assign(config.llm_hidden_size, 1.0f);
  prompt1.prompt_len = 1;
  prompt1.generate_audio = false;
  frontend.Push(prompt1);

  // Act: run first turn, then Reset(), then run identical turn.
  ASSERT_OK(stage->Schedule());
  ASSERT_OK_AND_ASSIGN(MiniCpmOLlmOutput out1, stage->GetOutput());

  stage->Reset();
  frontend.Push(prompt1);
  ASSERT_OK(stage->Schedule());
  ASSERT_OK_AND_ASSIGN(MiniCpmOLlmOutput out2, stage->GetOutput());

  // Assert
  EXPECT_EQ(reset_calls, 1);
  EXPECT_EQ(out1.num_tokens, 2);
  EXPECT_EQ(out1.generated_token_ids, out2.generated_token_ids);
  EXPECT_EQ(out1.text, out2.text);
}

}  // namespace
}  // namespace litert::omni::any2any
