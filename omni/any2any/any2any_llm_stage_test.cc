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

#include "omni/any2any/any2any_llm_stage.h"

#include <cstddef>
#include <cstdint>
#include <limits>
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
#include "omni/any2any/io_types.h"
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
  explicit FakeTokenizer(bool fail_text_to_token_ids = false,
                         bool fail_im_end_lookup = false)
      : fail_text_to_token_ids_(fail_text_to_token_ids),
        fail_im_end_lookup_(fail_im_end_lookup) {}

  support::TokenizerType GetTokenizerType() const override {
    return support::TokenizerType::kHuggingFace;
  }

  absl::StatusOr<support::TokenIds> TextToTokenIds(
      absl::string_view text) override {
    if (!fail_text_to_token_ids_ && text == "\n") {
      return support::TokenIds{8};
    }
    return absl::NotFoundError("not implemented");
  }

  absl::StatusOr<int> TokenToId(absl::string_view token) override {
    if (token == "<|endoftext|>") {
      return 0;
    }
    if (token == "<|im_end|>" && !fail_im_end_lookup_) {
      return 2;
    }
    if (token == "</s>") {
      return 7;
    }
    if (token == "\n") {
      return 8;
    }
    return absl::NotFoundError("not found");
  }

  absl::StatusOr<std::string> TokenIdsToText(
      absl::Span<const int> token_ids, bool /*skip_special_tokens*/) override {
    ++detokenize_calls_;
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
      } else {
        absl::StrAppend(&out, "tok", token_ids[i]);
      }
    }
    return out;
  }

  std::vector<std::string> GetTokens() const override { return {}; }

  int GetVocabSize() const override { return 16; }

  void SetCustomText(std::string text) { custom_text_ = std::move(text); }
  int detokenize_calls() const { return detokenize_calls_; }

 private:
  bool fail_text_to_token_ids_ = false;
  bool fail_im_end_lookup_ = false;
  int detokenize_calls_ = 0;
  std::optional<std::string> custom_text_;
};

TEST(Any2AnyLlmStageTest, CreateValidatesParametersAndTokenFallbacks) {
  // Arrange
  FakeFrontendStage frontend;
  auto tokenizer = std::make_shared<FakeTokenizer>();
  Any2AnyLlmStage::Options options;
  options.llm_hidden_size = 4;
  options.llm_vocab_size = 8;
  options.default_im_end_token_id = 2;
  options.default_eos_token_id = 0;

  // Act & Assert: non-positive llm_hidden_size fails with InvalidArgumentError.
  Any2AnyLlmStage::Options bad_hidden_options = options;
  bad_hidden_options.llm_hidden_size = 0;
  EXPECT_THAT(
      Any2AnyLlmStage::Create(&frontend, tokenizer,
                              std::make_shared<NiceMock<MockLiteRtLmRunner>>(),
                              bad_hidden_options),
      StatusIs(absl::StatusCode::kInvalidArgument));

  // Act & Assert: non-positive llm_vocab_size fails with InvalidArgumentError.
  Any2AnyLlmStage::Options bad_vocab_options = options;
  bad_vocab_options.llm_vocab_size = 0;
  EXPECT_THAT(
      Any2AnyLlmStage::Create(&frontend, tokenizer,
                              std::make_shared<NiceMock<MockLiteRtLmRunner>>(),
                              bad_vocab_options),
      StatusIs(absl::StatusCode::kInvalidArgument));

  // Act & Assert: valid parameters succeed.
  auto llm_runner = std::make_shared<NiceMock<MockLiteRtLmRunner>>();
  EXPECT_THAT(
      Any2AnyLlmStage::Create(&frontend, tokenizer, llm_runner, options),
      IsOk());

  // Act & Assert: falls back to default_im_end_token_id and TokenToId("\n")
  // when tokenizer lookups fail.
  auto fallback_runner = std::make_shared<NiceMock<MockLiteRtLmRunner>>();
  ON_CALL(*fallback_runner, Prefill(_))
      .WillByDefault(
          [](const lm::ExecutorInputs&) { return absl::OkStatus(); });
  std::vector<int32_t> observed_tokens;
  ON_CALL(*fallback_runner, Decode(_))
      .WillByDefault([&observed_tokens](const lm::ExecutorInputs& inputs)
                         -> absl::StatusOr<TensorBuffer> {
        ABSL_ASSIGN_OR_RETURN(const TensorBuffer* tok_buf,
                              inputs.GetTextTokenIdsPtr());
        int32_t tok = -999;
        LITERT_RETURN_IF_ERROR(
            const_cast<TensorBuffer*>(tok_buf)->Read<int32_t>(
                absl::MakeSpan(&tok, 1)));
        observed_tokens.push_back(tok);
        std::vector<float> logits(8, -10.0f);
        logits[2] = 10.0f;  // Immediate stop via default_im_end_token_id.
        LITERT_ASSIGN_OR_RETURN(auto buf,
                                support::CreateTensorBuffer<float>({1, 8}));
        LITERT_RETURN_IF_ERROR(buf.Write<float>(absl::MakeConstSpan(logits)));
        return buf;
      });
  ASSERT_OK_AND_ASSIGN(
      auto fallback_stage,
      Any2AnyLlmStage::Create(
          &frontend,
          std::make_shared<FakeTokenizer>(/*fail_text_to_token_ids=*/true,
                                          /*fail_im_end_lookup=*/true),
          fallback_runner, options));
  frontend.Push(Any2AnyFrontendOutput{
      .prompt_embeddings = std::vector<float>(4, 0.5f),
      .prompt_len = 1,
      .generate_audio = false,
  });
  ASSERT_OK(fallback_stage->Schedule());
  // Receives -1 (decode), then fallback default_im_end_token_id (2) and
  // TokenToId("\n") (8).
  EXPECT_THAT(observed_tokens, ElementsAre(-1, 2, 8));
}

TEST(Any2AnyLlmStageTest, NeedScheduleAndPromptValidation) {
  // Arrange
  FakeFrontendStage frontend;
  Any2AnyLlmStage::Options options;
  options.llm_hidden_size = 4;
  options.llm_vocab_size = 8;
  auto llm_runner = std::make_shared<NiceMock<MockLiteRtLmRunner>>();
  ASSERT_OK_AND_ASSIGN(
      auto stage,
      Any2AnyLlmStage::Create(&frontend, std::make_shared<FakeTokenizer>(),
                              llm_runner, options));

  // Act & Assert: no output in frontend -> NeedSchedule() is false and
  // Schedule() returns NotFoundError.
  EXPECT_FALSE(stage->NeedSchedule());
  EXPECT_THAT(stage->Schedule(), StatusIs(absl::StatusCode::kNotFound));

  // Arrange: push an empty prompt.
  frontend.Push(Any2AnyFrontendOutput{.prompt_len = 0});
  EXPECT_TRUE(stage->NeedSchedule());

  // Act & Assert: empty prompt returns InvalidArgumentError.
  EXPECT_THAT(stage->Schedule(), StatusIs(absl::StatusCode::kInvalidArgument));

  // Arrange: push a prompt whose embeddings size does not match
  // prompt_len * llm_hidden_size.
  Any2AnyFrontendOutput misaligned_prompt;
  misaligned_prompt.prompt_embeddings = {1.0f, 2.0f};
  misaligned_prompt.prompt_len = 1;
  frontend.Push(std::move(misaligned_prompt));

  // Act & Assert: misaligned prompt returns InvalidArgumentError.
  EXPECT_THAT(stage->Schedule(), StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(Any2AnyLlmStageTest,
     GenerateWithLmRunnerGreedyDecodingProducesExpectedTokens) {
  // Arrange
  FakeFrontendStage frontend;
  Any2AnyLlmStage::Options options;
  options.llm_hidden_size = 4;
  options.llm_vocab_size = 8;
  options.max_new_tokens = 4;
  options.do_sample = false;

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
      auto stage,
      Any2AnyLlmStage::Create(&frontend, std::make_shared<FakeTokenizer>(),
                              llm_runner, options));

  Any2AnyFrontendOutput prompt;
  prompt.prompt_embeddings.assign(2 * options.llm_hidden_size, 0.5f);
  prompt.prompt_len = 2;
  prompt.generate_audio = false;
  frontend.Push(std::move(prompt));

  // Act
  ASSERT_OK(stage->Schedule());

  // Assert: Decode receives -1 (last prompt token), 3 ("Hello"), 4 ("world"),
  // followed by the turn-closing tokens 2 ("<|im_end|>") and 8 ("\n").
  EXPECT_EQ(observed_input_tokens, (std::vector<int32_t>{-1, 3, 4, 2, 8}));
  ASSERT_TRUE(stage->HasOutput());
  ASSERT_OK_AND_ASSIGN(Any2AnyLlmOutput out, stage->GetOutput());
  EXPECT_EQ(out.text, "Hello world");
  EXPECT_EQ(out.generated_token_ids, (std::vector<int>{3, 4}));
  EXPECT_EQ(out.num_tokens, 2);
  EXPECT_FALSE(out.generate_audio);
  EXPECT_TRUE(out.is_final);
  EXPECT_TRUE(out.llm_hidden_states.empty());
}

TEST(Any2AnyLlmStageTest,
     GenerateWithLmRunnerPrefillsFullPromptAndExtractsCombinedHiddenState) {
  // Arrange
  FakeFrontendStage frontend;
  auto llm_runner = std::make_shared<NiceMock<MockLiteRtLmRunner>>();

  Any2AnyLlmStage::Options options;
  options.llm_hidden_size = 4;
  options.llm_vocab_size = 8;
  options.max_new_tokens = 2;
  options.do_sample = false;

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
      auto stage,
      Any2AnyLlmStage::Create(&frontend, std::make_shared<FakeTokenizer>(),
                              llm_runner, options));

  // All 5 prompt tokens are passed in a single Prefill call.
  Any2AnyFrontendOutput prompt;
  prompt.prompt_embeddings.assign(5 * options.llm_hidden_size, 0.25f);
  prompt.prompt_len = 5;
  prompt.generate_audio = true;
  frontend.Push(std::move(prompt));

  // Act
  ASSERT_OK(stage->Schedule());

  // Assert: 1 Prefill call of 5 tokens, 2 generation Decode calls + 2
  // turn-closing Decode calls (<|im_end|> and \n).
  EXPECT_EQ(prefill_call_tokens, (std::vector<size_t>{5}));
  EXPECT_EQ(decode_calls, 4);
  ASSERT_OK_AND_ASSIGN(Any2AnyLlmOutput out, stage->GetOutput());
  EXPECT_TRUE(out.generate_audio);
  EXPECT_EQ(out.text, "Hello");
  EXPECT_EQ(out.generated_token_ids, std::vector<int>{3});
  EXPECT_TRUE(out.prompt_token_ids.empty());
  EXPECT_TRUE(out.prompt_hidden_states.empty());
  EXPECT_EQ(out.llm_hidden_states,
            (std::vector<float>{1.5f, 2.5f, 3.5f, 4.5f}));
}

TEST(Any2AnyLlmStageTest,
     MultiTurnConversationCommitsTurnEndTokensAndGatesAudioConditioning) {
  // Arrange
  FakeFrontendStage frontend;
  auto llm_runner = std::make_shared<NiceMock<MockLiteRtLmRunner>>();

  Any2AnyLlmStage::Options options;
  options.llm_hidden_size = 4;
  options.llm_vocab_size = 8;
  options.max_new_tokens = 2;
  options.do_sample = false;
  options.enable_audio_conditioning = false;

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
      auto stage,
      Any2AnyLlmStage::Create(&frontend, std::make_shared<FakeTokenizer>(),
                              llm_runner, options));

  // Act: Turn 1 hits max_new_tokens = 2; prompt.generate_audio = true is gated
  // off by options.enable_audio_conditioning = false.
  Any2AnyFrontendOutput turn1_prompt;
  turn1_prompt.prompt_embeddings.assign(3 * options.llm_hidden_size, 1.0f);
  turn1_prompt.prompt_len = 3;
  turn1_prompt.generate_audio = true;
  frontend.Push(std::move(turn1_prompt));
  ASSERT_OK(stage->Schedule());
  ASSERT_OK_AND_ASSIGN(Any2AnyLlmOutput turn1_out, stage->GetOutput());
  EXPECT_FALSE(turn1_out.generate_audio);
  EXPECT_TRUE(turn1_out.llm_hidden_states.empty());
  EXPECT_EQ(turn1_out.text, "Hello world");
  EXPECT_EQ(turn1_out.generated_token_ids, (std::vector<int>{3, 4}));

  // Act: Turn 2 (without Reset(); stops on <|im_end|>).
  Any2AnyFrontendOutput turn2_prompt;
  turn2_prompt.prompt_embeddings.assign(4 * options.llm_hidden_size, 2.0f);
  turn2_prompt.prompt_len = 4;
  turn2_prompt.generate_audio = false;
  frontend.Push(std::move(turn2_prompt));
  ASSERT_OK(stage->Schedule());
  ASSERT_OK_AND_ASSIGN(Any2AnyLlmOutput turn2_out, stage->GetOutput());
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

TEST(Any2AnyLlmStageTest,
     GenerateWithLmRunnerDecodesFloat16LogitsAndHonorsDetokenizeOutputFlag) {
  // Arrange
  FakeFrontendStage frontend;
  Any2AnyLlmStage::Options options;
  options.llm_hidden_size = 4;
  options.llm_vocab_size = 8;
  options.max_new_tokens = 2;
  options.do_sample = false;
  options.detokenize_output = false;

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
        if (decode_step % 2 == 0) {
          fp16_bits[3] = 0x4000;  // "Hello"
        } else {
          fp16_bits[4] = 0x4000;  // "world"
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
      auto stage,
      Any2AnyLlmStage::Create(&frontend, tokenizer, llm_runner, options));

  Any2AnyFrontendOutput prompt;
  prompt.prompt_embeddings.assign(options.llm_hidden_size, 0.1f);
  prompt.prompt_len = 1;
  prompt.generate_audio = true;
  frontend.Push(prompt);

  // Act: detokenize_output = false skips TokenIdsToText() and keeps Float16
  // hidden states aligned with generated tokens.
  ASSERT_OK(stage->Schedule());

  // Assert
  ASSERT_OK_AND_ASSIGN(Any2AnyLlmOutput out, stage->GetOutput());
  EXPECT_TRUE(out.generate_audio);
  EXPECT_TRUE(out.text.empty());
  EXPECT_EQ(tokenizer->detokenize_calls(), 0);
  EXPECT_EQ(out.generated_token_ids, (std::vector<int>{3, 4}));
  EXPECT_EQ(out.num_tokens, 2);
  EXPECT_EQ(
      out.llm_hidden_states,
      (std::vector<float>{1.0f, 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 2.0f}));

  // Also verify detokenize_output = true strips leading ASCII whitespace.
  options.detokenize_output = true;
  tokenizer->SetCustomText("  \nClean text");
  ASSERT_OK_AND_ASSIGN(
      auto detok_stage,
      Any2AnyLlmStage::Create(&frontend, tokenizer, llm_runner, options));
  frontend.Push(std::move(prompt));
  ASSERT_OK(detok_stage->Schedule());
  ASSERT_OK_AND_ASSIGN(Any2AnyLlmOutput out2, detok_stage->GetOutput());
  EXPECT_EQ(tokenizer->detokenize_calls(), 1);
  EXPECT_EQ(out2.text, "Clean text");
}

TEST(Any2AnyLlmStageTest,
     GenerateWithLmRunnerRejectsInt16LogitsAndMissingAudioHiddenStates) {
  // Arrange
  FakeFrontendStage frontend;
  Any2AnyLlmStage::Options options;
  options.llm_hidden_size = 4;
  options.llm_vocab_size = 8;
  options.max_new_tokens = 1;
  options.do_sample = false;

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
      auto stage,
      Any2AnyLlmStage::Create(&frontend, std::make_shared<FakeTokenizer>(),
                              llm_runner, options));

  Any2AnyFrontendOutput prompt;
  prompt.prompt_embeddings.assign(options.llm_hidden_size, 0.1f);
  prompt.prompt_len = 1;
  prompt.generate_audio = true;
  frontend.Push(prompt);

  // Act & Assert: Int16 TensorBuffer is rejected with InvalidArgumentError.
  EXPECT_THAT(stage->Schedule(), StatusIs(absl::StatusCode::kInvalidArgument));

  // Act & Assert: when Decode output size matches neither llm_vocab_size nor
  // llm_vocab_size + llm_hidden_size, returns InvalidArgumentError even when
  // generate_audio is false.
  mode = DecodeOutputMode::kMismatchedFloatLength;
  Any2AnyFrontendOutput text_only_prompt = prompt;
  text_only_prompt.generate_audio = false;
  frontend.Push(std::move(text_only_prompt));
  EXPECT_THAT(stage->Schedule(), StatusIs(absl::StatusCode::kInvalidArgument));

  // Act & Assert: when generate_audio is true and combined hidden states are
  // absent from `llm_runner` output, returns FailedPreconditionError.
  mode = DecodeOutputMode::kLogitsOnlyFloat;
  frontend.Push(prompt);
  EXPECT_THAT(stage->Schedule(),
              StatusIs(absl::StatusCode::kFailedPrecondition));

  // Act & Assert: missing hidden states in kPostTokenWithPrompt across prompt
  // step 0, prompt step 1, and reply step.
  Any2AnyLlmStage::Options post_options = options;
  post_options.hidden_state_mode =
      Any2AnyLlmStage::HiddenStateMode::kPostTokenWithPrompt;
  ASSERT_OK_AND_ASSIGN(
      auto post_stage,
      Any2AnyLlmStage::Create(&frontend, std::make_shared<FakeTokenizer>(),
                              llm_runner, post_options));

  frontend.Push(Any2AnyFrontendOutput{
      .prompt_embeddings = std::vector<float>(4, 0.1f),
      .prompt_len = 1,
      .token_ids = {1},
      .generate_audio = true,
  });
  EXPECT_THAT(post_stage->Schedule(),
              StatusIs(absl::StatusCode::kFailedPrecondition));

  int post_call = 0;
  ON_CALL(*llm_runner, Decode(_))
      .WillByDefault([&post_call](const lm::ExecutorInputs&)
                         -> absl::StatusOr<TensorBuffer> {
        ++post_call;
        const int len = (post_call == 1) ? 12 : 8;
        std::vector<float> raw(len, 0.0f);
        LITERT_ASSIGN_OR_RETURN(auto buf,
                                support::CreateTensorBuffer<float>({1, len}));
        LITERT_RETURN_IF_ERROR(buf.Write<float>(absl::MakeConstSpan(raw)));
        return buf;
      });
  frontend.Push(Any2AnyFrontendOutput{
      .prompt_embeddings = std::vector<float>(2 * 4, 0.1f),
      .prompt_len = 2,
      .token_ids = {1, 3},
      .generate_audio = true,
  });
  EXPECT_THAT(post_stage->Schedule(),
              StatusIs(absl::StatusCode::kFailedPrecondition));

  post_call = 0;
  ON_CALL(*llm_runner, Decode(_))
      .WillByDefault([&post_call](const lm::ExecutorInputs&)
                         -> absl::StatusOr<TensorBuffer> {
        ++post_call;
        if (post_call == 1) {
          std::vector<float> raw(12, -10.0f);
          raw[3] = 10.0f;
          LITERT_ASSIGN_OR_RETURN(auto buf,
                                  support::CreateTensorBuffer<float>({1, 12}));
          LITERT_RETURN_IF_ERROR(buf.Write<float>(absl::MakeConstSpan(raw)));
          return buf;
        }
        std::vector<float> raw(8, 0.0f);
        LITERT_ASSIGN_OR_RETURN(auto buf,
                                support::CreateTensorBuffer<float>({1, 8}));
        LITERT_RETURN_IF_ERROR(buf.Write<float>(absl::MakeConstSpan(raw)));
        return buf;
      });
  frontend.Push(Any2AnyFrontendOutput{
      .prompt_embeddings = std::vector<float>(4, 0.1f),
      .prompt_len = 1,
      .token_ids = {1},
      .generate_audio = true,
  });
  EXPECT_THAT(post_stage->Schedule(),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(Any2AnyLlmStageTest,
     StochasticSamplingAndResetProduceDeterministicReproducibleTokens) {
  // Arrange
  FakeFrontendStage frontend;
  Any2AnyLlmStage::Options options;
  options.llm_hidden_size = 4;
  options.llm_vocab_size = 8;
  options.max_new_tokens = 8;
  options.do_sample = true;
  options.temperature = 0.8f;
  options.top_k = 4;
  options.top_p = 0.99f;
  options.seed = 12345;
  options.default_im_end_token_id = 2;
  options.default_eos_token_id = 0;

  const float kNaN = std::numeric_limits<float>::quiet_NaN();
  const float kInf = std::numeric_limits<float>::infinity();

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
      .WillByDefault([kNaN, kInf](const lm::ExecutorInputs&)
                         -> absl::StatusOr<TensorBuffer> {
        std::vector<float> logits = {kNaN, -kInf, -10.0f, 4.0f,
                                     4.0f, 4.0f,  4.0f,   kNaN};
        LITERT_ASSIGN_OR_RETURN(auto buf,
                                support::CreateTensorBuffer<float>({1, 8}));
        LITERT_RETURN_IF_ERROR(buf.Write<float>(absl::MakeConstSpan(logits)));
        return buf;
      });

  ASSERT_OK_AND_ASSIGN(
      auto stage,
      Any2AnyLlmStage::Create(&frontend, std::make_shared<FakeTokenizer>(),
                              llm_runner, options));

  auto run_once = [&](Any2AnyLlmStage& s) {
    Any2AnyFrontendOutput prompt;
    prompt.prompt_embeddings.assign(options.llm_hidden_size, 1.0f);
    prompt.prompt_len = 1;
    prompt.generate_audio = false;
    frontend.Push(std::move(prompt));
    EXPECT_OK(s.Schedule());
    return s.GetOutput();
  };

  // Act: run first turn, second turn without Reset(), then Reset() and re-run.
  ASSERT_OK_AND_ASSIGN(Any2AnyLlmOutput out1, run_once(*stage));
  ASSERT_EQ(out1.generated_token_ids.size(), 8);
  for (int tok : out1.generated_token_ids) {
    EXPECT_THAT(tok, AnyOf(3, 4, 5, 6));
  }
  ASSERT_OK_AND_ASSIGN(Any2AnyLlmOutput unreset_out, run_once(*stage));
  EXPECT_NE(unreset_out.generated_token_ids, out1.generated_token_ids);

  stage->Reset();
  ASSERT_OK_AND_ASSIGN(Any2AnyLlmOutput out2, run_once(*stage));

  // Assert
  EXPECT_EQ(reset_calls, 1);
  EXPECT_EQ(out1.num_tokens, 8);
  EXPECT_EQ(out1.generated_token_ids, out2.generated_token_ids);
  EXPECT_EQ(out1.text, out2.text);

  // Top-p truncation, +Inf, and all-non-finite fallback in stochastic mode.
  options.max_new_tokens = 1;
  options.top_k = 3;
  options.top_p = 0.7f;
  ASSERT_OK_AND_ASSIGN(
      auto sample_stage,
      Any2AnyLlmStage::Create(&frontend, std::make_shared<FakeTokenizer>(),
                              llm_runner, options));

  auto set_decode_logits = [&llm_runner](std::vector<float> logits) {
    ON_CALL(*llm_runner, Decode(_))
        .WillByDefault([logits = std::move(logits)](const lm::ExecutorInputs&)
                           -> absl::StatusOr<TensorBuffer> {
          LITERT_ASSIGN_OR_RETURN(auto buf,
                                  support::CreateTensorBuffer<float>(
                                      {1, static_cast<int>(logits.size())}));
          LITERT_RETURN_IF_ERROR(buf.Write<float>(absl::MakeConstSpan(logits)));
          return buf;
        });
  };

  set_decode_logits({kNaN, -kInf, -10.0f, 4.0f, 3.8f, -10.0f, kNaN, -10.0f});
  ASSERT_OK_AND_ASSIGN(auto top_p_out, run_once(*sample_stage));
  ASSERT_EQ(top_p_out.generated_token_ids.size(), 1);
  EXPECT_THAT(top_p_out.generated_token_ids[0], AnyOf(3, 4));

  set_decode_logits({kNaN, 1.0f, -10.0f, 2.0f, kInf, 0.0f, 0.0f, 0.0f});
  ASSERT_OK_AND_ASSIGN(auto inf_out, run_once(*sample_stage));
  EXPECT_THAT(inf_out.generated_token_ids, ElementsAre(4));

  set_decode_logits({kNaN, -kInf, kNaN, kNaN, kNaN, kNaN, kNaN, kNaN});
  ASSERT_OK_AND_ASSIGN(auto all_nan_out, run_once(*sample_stage));
  EXPECT_TRUE(all_nan_out.generated_token_ids.empty());

  // Greedy mode with leading NaN and +Inf, plus top_k == 0 full-vocab sampling.
  Any2AnyLlmStage::Options greedy_opts = options;
  greedy_opts.do_sample = false;
  ASSERT_OK_AND_ASSIGN(
      auto greedy_stage,
      Any2AnyLlmStage::Create(&frontend, std::make_shared<FakeTokenizer>(),
                              llm_runner, greedy_opts));
  set_decode_logits({kNaN, -5.0f, -10.0f, 3.0f, 1.0f, -5.0f, -5.0f, -5.0f});
  ASSERT_OK_AND_ASSIGN(auto greedy_nan_out, run_once(*greedy_stage));
  EXPECT_THAT(greedy_nan_out.generated_token_ids, ElementsAre(3));

  set_decode_logits({kNaN, 1.0f, -10.0f, 2.0f, kInf, 0.0f, 0.0f, 0.0f});
  ASSERT_OK_AND_ASSIGN(auto greedy_inf_out, run_once(*greedy_stage));
  EXPECT_THAT(greedy_inf_out.generated_token_ids, ElementsAre(4));

  Any2AnyLlmStage::Options full_vocab_opts = options;
  full_vocab_opts.top_k = 0;
  full_vocab_opts.top_p = 1.0f;
  ASSERT_OK_AND_ASSIGN(
      auto full_sample_stage,
      Any2AnyLlmStage::Create(&frontend, std::make_shared<FakeTokenizer>(),
                              llm_runner, full_vocab_opts));
  set_decode_logits(
      {-100.0f, -100.0f, -100.0f, 10.0f, -100.0f, -100.0f, -100.0f, -100.0f});
  ASSERT_OK_AND_ASSIGN(auto full_out, run_once(*full_sample_stage));
  EXPECT_THAT(full_out.generated_token_ids, ElementsAre(3));
}

TEST(Any2AnyLlmStageTest, PostTokenWithPromptModeTextOnlyAndAudioConditioning) {
  // Arrange
  FakeFrontendStage frontend;
  auto llm_runner = std::make_shared<NiceMock<MockLiteRtLmRunner>>();
  ON_CALL(*llm_runner, Prefill(_)).WillByDefault([](const lm::ExecutorInputs&) {
    return absl::OkStatus();
  });

  // Case 1: generate_audio = false -> batch prefill + post-token decode.
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
          logits[3] = 10.0f;
        } else {
          logits[2] = 10.0f;
        }
        ++step;
        LITERT_ASSIGN_OR_RETURN(auto buf,
                                support::CreateTensorBuffer<float>({1, 8}));
        LITERT_RETURN_IF_ERROR(buf.Write<float>(absl::MakeConstSpan(logits)));
        return buf;
      });

  Any2AnyLlmStage::Options options;
  options.llm_hidden_size = 4;
  options.llm_vocab_size = 8;
  options.max_new_tokens = 4;
  options.do_sample = false;
  options.enable_audio_conditioning = true;
  options.hidden_state_mode =
      Any2AnyLlmStage::HiddenStateMode::kPostTokenWithPrompt;
  options.default_im_end_token_id = 2;
  options.default_eos_token_id = 0;

  ASSERT_OK_AND_ASSIGN(
      auto stage,
      Any2AnyLlmStage::Create(&frontend, std::make_shared<FakeTokenizer>(),
                              llm_runner, options));

  frontend.Push(Any2AnyFrontendOutput{
      .prompt_embeddings = std::vector<float>(2 * 4, 0.5f),
      .prompt_len = 2,
      .token_ids = {1, 3},
      .generate_audio = false,
  });
  ASSERT_OK(stage->Schedule());
  ASSERT_OK_AND_ASSIGN(Any2AnyLlmOutput text_only_out, stage->GetOutput());
  EXPECT_FALSE(text_only_out.generate_audio);
  EXPECT_EQ(text_only_out.text, "Hello");
  EXPECT_TRUE(text_only_out.prompt_hidden_states.empty());
  EXPECT_TRUE(text_only_out.llm_hidden_states.empty());
  EXPECT_THAT(observed_tokens, ElementsAre(-1, 3, 2, 8));

  // Case 2: generate_audio = true -> token-by-token prompt prefill collecting
  // prompt_hidden_states, then reply tokens paired with input-step hidden
  // states.
  int audio_call = 0;
  ON_CALL(*llm_runner, Decode(_))
      .WillByDefault([&audio_call](const lm::ExecutorInputs&)
                         -> absl::StatusOr<TensorBuffer> {
        std::vector<float> raw(12, -10.0f);
        if (audio_call == 1) {
          raw[3] = 10.0f;  // After last prompt token -> emit token 3 ("Hello").
        } else if (audio_call == 2) {
          raw[2] = 10.0f;  // After token 3 -> stop (<|im_end|>).
        } else {
          raw[0] = 1.0f;
        }
        const float hidden_val = static_cast<float>(audio_call + 1);
        for (int h = 0; h < 4; ++h) {
          raw[8 + h] = hidden_val;
        }
        ++audio_call;
        LITERT_ASSIGN_OR_RETURN(auto buf,
                                support::CreateTensorBuffer<float>({1, 12}));
        LITERT_RETURN_IF_ERROR(buf.Write<float>(absl::MakeConstSpan(raw)));
        return buf;
      });

  frontend.Push(Any2AnyFrontendOutput{
      .prompt_embeddings = std::vector<float>(2 * 4, 0.25f),
      .prompt_len = 2,
      .token_ids = {1, 3},
      .generate_audio = true,
  });
  ASSERT_OK(stage->Schedule());
  ASSERT_OK_AND_ASSIGN(Any2AnyLlmOutput audio_out, stage->GetOutput());
  EXPECT_TRUE(audio_out.generate_audio);
  EXPECT_EQ(audio_out.text, "Hello");
  EXPECT_THAT(audio_out.prompt_token_ids, ElementsAre(1, 3));
  EXPECT_THAT(audio_out.generated_token_ids, ElementsAre(3));
  EXPECT_THAT(audio_out.prompt_hidden_states,
              ElementsAre(1.0f, 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 2.0f));
  EXPECT_THAT(audio_out.llm_hidden_states, ElementsAre(3.0f, 3.0f, 3.0f, 3.0f));
}

}  // namespace
}  // namespace litert::omni::any2any
