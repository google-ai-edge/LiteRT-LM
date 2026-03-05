// Copyright 2025 The ODML Authors.
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

#include "runtime/executor/fake_llm_executor.h"

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/test/matchers.h"  // from @litert
#include "runtime/components/constrained_decoding/constrained_decoder.h"
#include "runtime/components/constrained_decoding/fake_constraint.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

using ::testing::status::StatusIs;

TEST(FakeLlmExecutorTest, ExecutorSettings) {
  const std::vector<std::vector<int>> prefill_tokens_set = {{1, 2, 3}};
  const std::vector<std::vector<int>> decode_tokens_set = {{3, 2}, {0, 0}};
  FakeLlmExecutor fake_llm_executor(3, prefill_tokens_set, decode_tokens_set);
  EXPECT_OK(fake_llm_executor.GetExecutorSettings());
  EXPECT_EQ(fake_llm_executor.GetExecutorSettings()->GetMaxNumTokens(), 1024);

  // Set the max num tokens to 100.
  fake_llm_executor.GetMutableExecutorSettings().value()->SetMaxNumTokens(100);
  EXPECT_EQ(fake_llm_executor.GetExecutorSettings()->GetMaxNumTokens(), 100);
}

TEST(FakeLlmExecutorTest, Prefill) {
  const std::vector<std::vector<int>> prefill_tokens_set = {{1, 2, 3}};
  const std::vector<std::vector<int>> decode_tokens_set = {{3, 2}, {0, 0}};
  FakeLlmExecutor fake_llm_executor(3, prefill_tokens_set, decode_tokens_set);

  ExecutorInputs inputs;
  // Create a tensor buffer with 3 elements but only the first two elements
  // match the expected prefill tokens.
  const std::vector<int> input_tokens = {1, 2, 0};
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto input_tokens_buffer,
      CopyToTensorBuffer<int>(absl::MakeSpan(input_tokens), {1, 3}));
  inputs.SetTextData(ExecutorTextData(std::move(input_tokens_buffer)));

  // Fail because the input tokens do not match the expected prefill tokens.
  EXPECT_THAT(fake_llm_executor.Prefill(inputs),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // Succeed because the input tokens match the expected prefill tokens.
  auto ids_span = ReferTensorBufferAsSpan<int>(*(*inputs.GetTextTokenIdsPtr()));

  (*ids_span)[2] = 3;
  EXPECT_OK(fake_llm_executor.Prefill(inputs));
  EXPECT_EQ(fake_llm_executor.GetCurrentStep().value(), 3);
}

TEST(FakeLlmExecutorTest, PrefillWithAudio) {
  const std::vector<std::vector<int>> prefill_tokens_set = {{1, 2, 3}};
  const std::vector<std::vector<int>> decode_tokens_set = {{3, 2}, {0, 0}};
  std::vector<float> audio_embeddings_set = {1.0f, 2.0f, 3.0f, 4.0f};
  FakeLlmExecutor fake_llm_executor(3, prefill_tokens_set, decode_tokens_set,
                                    /*batch_size=*/1, audio_embeddings_set);

  ExecutorInputs inputs;
  // Create a tensor buffer with 3 elements but only the first two elements
  // match the expected prefill tokens.
  const std::vector<int> input_tokens = {1, 2, 3};
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto input_tokens_buffer,
      CopyToTensorBuffer<int>(absl::MakeSpan(input_tokens), {1, 3}));
  inputs.SetTextData(ExecutorTextData(std::move(input_tokens_buffer)));

  const std::vector<float> input_audio_embedding = {1.0f, 2.0f, 3.0f, 0.0f};
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto input_audio_embedding_buffer,
      CopyToTensorBuffer<float>(absl::MakeSpan(input_audio_embedding),
                                {1, 4, 1}));
  inputs.SetAudioData(
      ExecutorAudioData(std::move(input_audio_embedding_buffer), std::nullopt));

  // Fail because the input audio embedding does not match the expected the
  // audio embedding set.
  EXPECT_THAT(fake_llm_executor.Prefill(inputs),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // Succeed because the input audio embedding matches the expected audio
  // embedding set.
  auto audio_embedding_span =
      ReferTensorBufferAsSpan<float>(*(*inputs.GetAudioEmbeddingsPtr()));
  (*audio_embedding_span)[3] = 4.0f;

  EXPECT_OK(fake_llm_executor.Prefill(inputs));
  EXPECT_EQ(fake_llm_executor.GetCurrentStep().value(), 3);
}

TEST(FakeLlmExecutorTest, DecodeWithoutPrefillFailed) {
  const std::vector<std::vector<int>> prefill_tokens_set = {{1, 2, 3}};
  const std::vector<std::vector<int>> decode_tokens_set = {{3}, {0}};
  FakeLlmExecutor fake_llm_executor(/*vocab_size=*/4, prefill_tokens_set,
                                    decode_tokens_set);

  EXPECT_THAT(fake_llm_executor.Decode(),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(FakeLlmExecutorTest, DecodeToIds) {
  const std::vector<std::vector<int>> prefill_tokens_set = {{1, 2, 3}};
  const std::vector<std::vector<int>> decode_tokens_set = {{3}, {0}};
  FakeLlmExecutor fake_llm_executor(4, prefill_tokens_set, decode_tokens_set);

  ExecutorInputs inputs;
  const std::vector<int> input_tokens = {1, 2, 3};
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto input_tokens_buffer,
      CopyToTensorBuffer<int>(absl::MakeSpan(input_tokens), {1, 3}));
  inputs.SetTextData(ExecutorTextData(std::move(input_tokens_buffer)));
  EXPECT_OK(fake_llm_executor.Prefill(inputs));
  EXPECT_EQ(fake_llm_executor.GetCurrentStep().value(), 3);

  // Call Decode for the 1st time. The output tokens should be the 1st decode
  // tokens: 3.
  ASSERT_OK_AND_ASSIGN(auto output_tokens, fake_llm_executor.Decode());
  EXPECT_EQ(fake_llm_executor.GetCurrentStep().value(), 4);
  EXPECT_EQ(output_tokens[0][0], 3);

  // Call Decode for the 2nd time. The output tokens should be the 2nd decode
  // tokens: 0.
  ASSERT_OK_AND_ASSIGN(output_tokens, fake_llm_executor.Decode());
  EXPECT_EQ(fake_llm_executor.GetCurrentStep().value(), 5);
  EXPECT_EQ(output_tokens[0][0], 0);

  // Call Decode for the 3nd time. Should fail.
  EXPECT_THAT(fake_llm_executor.Decode(),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(FakeLlmExecutorTest, DecodeToLogits) {
  const std::vector<std::vector<int>> prefill_tokens_set = {{1, 2, 3}};
  const std::vector<std::vector<int>> decode_tokens_set = {{3}, {0}};
  FakeLlmExecutor fake_llm_executor(/*vocab_size=*/4, prefill_tokens_set,
                                    decode_tokens_set);

  ExecutorInputs inputs;
  const std::vector<int> input_tokens = {1, 2, 3};
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto input_tokens_buffer,
      CopyToTensorBuffer<int>(absl::MakeSpan(input_tokens), {1, 3}));
  inputs.SetTextData(ExecutorTextData(std::move(input_tokens_buffer)));
  EXPECT_OK(fake_llm_executor.Prefill(inputs));
  EXPECT_EQ(fake_llm_executor.GetCurrentStep().value(), 3);

  // Create a tensor buffer with 3 elements but only the first two elements
  // match the expected prefill tokens.
  const std::vector<int> decode_input_tokens = {3};
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto decode_input_tokens_buffer,
      CopyToTensorBuffer<int>(absl::MakeSpan(decode_input_tokens), {1, 1}));
  inputs.SetTextData(ExecutorTextData(std::move(decode_input_tokens_buffer)));

  auto output_logits = CreateTensorBuffer<float>({1, 1, 4});
  // Call Decode for the 1st time. The output logits should have values:
  // [-inf, -inf, -inf, inf].
  EXPECT_OK(fake_llm_executor.Decode(inputs, *output_logits));
  EXPECT_EQ(fake_llm_executor.GetCurrentStep().value(), 4);
  auto output_logits_span = ReferTensorBufferAsSpan<float>(*output_logits);
  EXPECT_LE((*output_logits_span)[0], 0.0f);
  EXPECT_LE((*output_logits_span)[1], 0.0f);
  EXPECT_LE((*output_logits_span)[2], 0.0f);
  EXPECT_GE((*output_logits_span)[3], 0.0f);

  // Call Decode for the 2nd time. The output logits should have values:
  // [inf, -inf, -inf, -inf].
  EXPECT_OK(fake_llm_executor.Decode(inputs, *output_logits));
  EXPECT_EQ(fake_llm_executor.GetCurrentStep().value(), 5);
  EXPECT_GE((*output_logits_span)[0], 0.0f);
  EXPECT_LE((*output_logits_span)[1], 0.0f);
  EXPECT_LE((*output_logits_span)[2], 0.0f);
  EXPECT_LE((*output_logits_span)[3], 0.0f);

  // Call Decode for the 3nd time. Should fail.
  EXPECT_THAT(fake_llm_executor.Decode(inputs, *output_logits),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(FakeLlmExecutorTest, DecodeLogits) {
  const std::vector<std::vector<int>> prefill_tokens_set = {{1, 2, 3}};
  const std::vector<std::vector<int>> decode_tokens_set = {{3}, {0}};
  FakeLlmExecutor fake_llm_executor(/*vocab_size=*/4, prefill_tokens_set,
                                    decode_tokens_set);

  ExecutorInputs inputs;
  const std::vector<int> input_tokens = {1, 2, 3};
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto input_tokens_buffer,
      CopyToTensorBuffer<int>(absl::MakeSpan(input_tokens), {1, 3}));
  inputs.SetTextData(ExecutorTextData(std::move(input_tokens_buffer)));
  EXPECT_OK(fake_llm_executor.Prefill(inputs));
  EXPECT_EQ(fake_llm_executor.GetCurrentStep().value(), 3);

  // Create a tensor buffer with 3 elements but only the first two elements
  // match the expected prefill tokens.
  const std::vector<int> decode_input_tokens = {3};
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto decode_input_tokens_buffer,
      CopyToTensorBuffer<int>(absl::MakeSpan(decode_input_tokens), {1, 1}));
  inputs.SetTextData(ExecutorTextData(std::move(decode_input_tokens_buffer)));

  auto output_logits = fake_llm_executor.DecodeLogits(inputs);
  // Call Decode for the 1st time. The output logits should have values:
  // [-inf, -inf, -inf, inf].
  EXPECT_TRUE(output_logits.ok());
  EXPECT_EQ(fake_llm_executor.GetCurrentStep().value(), 4);
  auto output_logits_span = ReferTensorBufferAsSpan<float>(*output_logits);
  EXPECT_LE((*output_logits_span)[0], 0.0f);
  EXPECT_LE((*output_logits_span)[1], 0.0f);
  EXPECT_LE((*output_logits_span)[2], 0.0f);
  EXPECT_GE((*output_logits_span)[3], 0.0f);

  output_logits = fake_llm_executor.DecodeLogits(inputs);
  // Call Decode for the 2nd time. The output logits should have values:
  // [inf, -inf, -inf, -inf].
  EXPECT_TRUE(output_logits.ok());
  EXPECT_EQ(fake_llm_executor.GetCurrentStep().value(), 5);
  output_logits_span = ReferTensorBufferAsSpan<float>(*output_logits);
  EXPECT_GE((*output_logits_span)[0], 0.0f);
  EXPECT_LE((*output_logits_span)[1], 0.0f);
  EXPECT_LE((*output_logits_span)[2], 0.0f);
  EXPECT_LE((*output_logits_span)[3], 0.0f);

  // Call Decode for the 3nd time. Should fail.
  EXPECT_THAT(fake_llm_executor.Decode(inputs, *output_logits),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(FakeLlmExecutorTest, DecodeDelay) {
  const std::vector<std::vector<int>> prefill_tokens_set = {{1, 2, 3}};
  const std::vector<std::vector<int>> decode_tokens_set = {{3}, {0}};
  FakeLlmExecutor fake_llm_executor(/*vocab_size=*/4, prefill_tokens_set,
                                    decode_tokens_set);

  constexpr absl::Duration delay = absl::Milliseconds(100);
  fake_llm_executor.SetDecodeDelay(delay);

  ExecutorInputs inputs;
  const std::vector<int> input_tokens = {1, 2, 3};
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto input_tokens_buffer,
      CopyToTensorBuffer<int>(absl::MakeSpan(input_tokens), {1, 3}));
  inputs.SetTextData(ExecutorTextData(std::move(input_tokens_buffer)));
  EXPECT_OK(fake_llm_executor.Prefill(inputs));

  const absl::Time start = absl::Now();
  ASSERT_OK_AND_ASSIGN(auto output_tokens, fake_llm_executor.Decode());
  const absl::Duration elapsed = absl::Now() - start;
  EXPECT_GE(elapsed, delay);
}

TEST(FakeLlmExecutorTest, MultiplePrefillTriggers) {
  const std::vector<std::vector<int>> prefill_tokens_set = {{1, 2, 3}, {4, 5}};
  const std::vector<std::vector<int>> decode_tokens_set = {{6}, {7}, {8}, {9}};
  FakeLlmExecutor fake_llm_executor(/*vocab_size=*/10, prefill_tokens_set,
                                    decode_tokens_set);

  // Trigger the first prefill/decode sequence.
  {
    ExecutorInputs inputs;
    const std::vector<int> input_tokens = {1, 2, 3};
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto input_tokens_buffer,
        CopyToTensorBuffer<int>(absl::MakeSpan(input_tokens), {1, 3}));
    inputs.SetTextData(ExecutorTextData(std::move(input_tokens_buffer)));
    EXPECT_OK(fake_llm_executor.Prefill(inputs));
    EXPECT_EQ(fake_llm_executor.GetCurrentStep().value(), 3);

    ASSERT_OK_AND_ASSIGN(auto output_tokens, fake_llm_executor.Decode());
    EXPECT_EQ(fake_llm_executor.GetCurrentStep().value(), 4);
    EXPECT_EQ(output_tokens[0][0], 6);
    ASSERT_OK_AND_ASSIGN(output_tokens, fake_llm_executor.Decode());
    EXPECT_EQ(fake_llm_executor.GetCurrentStep().value(), 5);
    EXPECT_EQ(output_tokens[0][0], 7);
  }

  // Trigger the second prefill/decode sequence.
  {
    ExecutorInputs inputs;
    const std::vector<int> input_tokens = {4, 5};
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto input_tokens_buffer,
        CopyToTensorBuffer<int>(absl::MakeSpan(input_tokens), {1, 2}));
    inputs.SetTextData(ExecutorTextData(std::move(input_tokens_buffer)));
    EXPECT_OK(fake_llm_executor.Prefill(inputs));
    EXPECT_EQ(fake_llm_executor.GetCurrentStep().value(), 7);

    ASSERT_OK_AND_ASSIGN(auto output_tokens, fake_llm_executor.Decode());
    EXPECT_EQ(fake_llm_executor.GetCurrentStep().value(), 8);
    EXPECT_EQ(output_tokens[0][0], 8);
    ASSERT_OK_AND_ASSIGN(output_tokens, fake_llm_executor.Decode());
    EXPECT_EQ(fake_llm_executor.GetCurrentStep().value(), 9);
    EXPECT_EQ(output_tokens[0][0], 9);
  }

  // Call Prefill for the 3rd time. Should fail.
  {
    ExecutorInputs inputs;
    const std::vector<int> input_tokens = {6};
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto input_tokens_buffer,
        CopyToTensorBuffer<int>(absl::MakeSpan(input_tokens), {1, 1}));
    inputs.SetTextData(ExecutorTextData(std::move(input_tokens_buffer)));
    EXPECT_THAT(fake_llm_executor.Prefill(inputs),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }
}

TEST(FakeLlmExecutorTest, DecodeWithConstraint) {
  const std::vector<std::vector<int>> prefill_tokens_set = {{1, 2, 3}};
  const std::vector<std::vector<int>> decode_tokens_set = {{4}, {0}, {4}, {0}};
  FakeLlmExecutor fake_llm_executor(/*vocab_size=*/10, prefill_tokens_set,
                                    decode_tokens_set);

  // Fake constraint that expects "4, 0".
  const std::vector<int> expected_token_ids = {4, 0};
  auto constraint = FakeConstraint(expected_token_ids, /*vocabulary_size=*/10);

  ExecutorInputs inputs;
  const std::vector<int> input_tokens = {1, 2, 3};
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto input_tokens_buffer,
      CopyToTensorBuffer<int>(absl::MakeSpan(input_tokens), {1, 3}));
  inputs.SetTextData(ExecutorTextData(std::move(input_tokens_buffer)));
  EXPECT_OK(fake_llm_executor.Prefill(inputs));
  EXPECT_EQ(fake_llm_executor.GetCurrentStep().value(), 3);

  auto constrained_decoder =
      std::make_unique<ConstrainedDecoder>(&constraint,
                                           /*num_output_candidates=*/1);
  auto decode_params = ExecutorDecodeParams();
  decode_params.SetConstraintDecoder(constrained_decoder.get());
  // Call Decode for the 1st time. The output tokens should be the 1st decode
  // tokens: 4. (first constraint token)
  ASSERT_OK_AND_ASSIGN(auto output_tokens,
                       fake_llm_executor.Decode(decode_params));
  EXPECT_EQ(fake_llm_executor.GetCurrentStep().value(), 4);
  EXPECT_EQ(output_tokens[0][0], 4);

  // Call Decode for the 2nd time. The output tokens should be the 2nd decode
  // tokens: 0. (second constraint token)
  ASSERT_OK_AND_ASSIGN(output_tokens, fake_llm_executor.Decode(decode_params));
  EXPECT_EQ(fake_llm_executor.GetCurrentStep().value(), 5);
  EXPECT_EQ(output_tokens[0][0], 0);

  // Call Decode for the 3rd time. The output tokens should be the 3rd decode
  // tokens: 4. (first constraint token again)
  ASSERT_OK_AND_ASSIGN(output_tokens, fake_llm_executor.Decode(decode_params));
  EXPECT_EQ(fake_llm_executor.GetCurrentStep().value(), 6);
  EXPECT_EQ(output_tokens[0][0], 4);

  // Call Decode for the 2nd time. The output tokens should be the 2nd decode
  // tokens: 0. (second constraint token again)
  ASSERT_OK_AND_ASSIGN(output_tokens, fake_llm_executor.Decode(decode_params));
  EXPECT_EQ(fake_llm_executor.GetCurrentStep().value(), 7);
  EXPECT_EQ(output_tokens[0][0], 0);

  // Call Decode for the 5nd time. Should fail.
  EXPECT_THAT(fake_llm_executor.Decode(decode_params),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace litert::lm
