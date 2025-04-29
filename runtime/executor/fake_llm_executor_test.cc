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

#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/test/matchers.h"  // from @litert
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/test_utils.h"  // NOLINT

namespace litert::lm {
namespace {

using ::testing::status::StatusIs;

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
  ;
  (*ids_span)[2] = 3;
  EXPECT_OK(fake_llm_executor.Prefill(inputs));
}

TEST(FakeLlmExecutorTest, DecodeToIds) {
  const std::vector<std::vector<int>> prefill_tokens_set = {{1, 2, 3}};
  const std::vector<std::vector<int>> decode_tokens_set = {{3}, {0}};
  FakeLlmExecutor fake_llm_executor(4, prefill_tokens_set, decode_tokens_set);

  LITERT_ASSERT_OK_AND_ASSIGN(auto output_tokens,
                              CreateTensorBuffer<int>({1, 1}));
  // Call Decode for the 1st time. The output tokens should be the 1st decode
  // tokens: 3.
  EXPECT_OK(fake_llm_executor.Decode(output_tokens));
  auto output_tokens_span = ReferTensorBufferAsSpan<int>(output_tokens);
  EXPECT_EQ((*output_tokens_span)[0], 3);

  // Call Decode for the 2nd time. The output tokens should be the 2nd decode
  // tokens: 0.
  EXPECT_OK(fake_llm_executor.Decode(output_tokens));
  output_tokens_span = ReferTensorBufferAsSpan<int>(output_tokens);
  EXPECT_EQ((*output_tokens_span)[0], 0);

  // Call Decode for the 3nd time. Should fail.
  EXPECT_THAT(fake_llm_executor.Decode(output_tokens),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(FakeLlmExecutorTest, DecodeToLogits) {
  const std::vector<std::vector<int>> prefill_tokens_set = {{1, 2, 3}};
  const std::vector<std::vector<int>> decode_tokens_set = {{3}, {0}};
  FakeLlmExecutor fake_llm_executor(/*vocab_size=*/4, prefill_tokens_set,
                                    decode_tokens_set);

  ExecutorInputs inputs;
  // Create a tensor buffer with 3 elements but only the first two elements
  // match the expected prefill tokens.
  const std::vector<int> input_tokens = {3};
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto input_tokens_buffer,
      CopyToTensorBuffer<int>(absl::MakeSpan(input_tokens), {1, 1}));
  inputs.SetTextData(ExecutorTextData(std::move(input_tokens_buffer)));

  auto output_logits = CreateTensorBuffer<float>({1, 1, 4});
  // Call Decode for the 1st time. The output logits should have values:
  // [-inf, -inf, -inf, inf].
  EXPECT_OK(fake_llm_executor.Decode(inputs, *output_logits));
  auto output_logits_span = ReferTensorBufferAsSpan<float>(*output_logits);
  EXPECT_LE((*output_logits_span)[0], 0.0f);
  EXPECT_LE((*output_logits_span)[1], 0.0f);
  EXPECT_LE((*output_logits_span)[2], 0.0f);
  EXPECT_GE((*output_logits_span)[3], 0.0f);

  // Call Decode for the 2nd time. The output logits should have values:
  // [inf, -inf, -inf, -inf].
  EXPECT_OK(fake_llm_executor.Decode(inputs, *output_logits));
  EXPECT_GE((*output_logits_span)[0], 0.0f);
  EXPECT_LE((*output_logits_span)[1], 0.0f);
  EXPECT_LE((*output_logits_span)[2], 0.0f);
  EXPECT_LE((*output_logits_span)[3], 0.0f);

  // Call Decode for the 3nd time. Should fail.
  EXPECT_THAT(fake_llm_executor.Decode(inputs, *output_logits),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace litert::lm
