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

#include "runtime/core/tasks.h"

#include <atomic>
#include <filesystem>  // NOLINT: Required for path manipulation.
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "runtime/components/constrained_decoding/fake_constraint.h"
#include "runtime/components/sentencepiece_tokenizer.h"
#include "runtime/components/stop_token_detector.h"
#include "runtime/components/tokenizer.h"
#include "runtime/components/top_p_cpu_sampler.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/fake_llm_executor.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/framework/threadpool.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/status_macros.h"
#include "runtime/util/test_utils.h"  // NOLINT

namespace litert::lm {
namespace {

using ::testing::status::StatusIs;

constexpr char kTestdataDir[] =
    "litert_lm/runtime/components/testdata/";

class BytePairEncodingTokenizer : public Tokenizer {
 public:
  MOCK_METHOD(absl::StatusOr<std::vector<int>>, TextToTokenIds,
              (absl::string_view text), (override));
  MOCK_METHOD(absl::StatusOr<std::string>, TokenIdsToText,
              (const std::vector<int>& token_ids), (override));
  MOCK_METHOD(absl::StatusOr<int>, TokenToId, (absl::string_view token),
              (override));
  MOCK_METHOD(TokenizerType, GetTokenizerType, (), (const, override));
  MOCK_METHOD(std::vector<std::string>, GetTokens, (), (const, override));
};

absl::AnyInvocable<void(absl::StatusOr<Responses>)> CreateTestCallback(
    std::vector<std::string>& responses_ref, absl::Status& status_ref,
    bool& done_ref, bool delay_on_next = false) {
  return [&responses_ref, &status_ref, &done_ref,
          delay_on_next](absl::StatusOr<Responses> responses) mutable {
    // If the responses is not ok, the error status is returned.
    if (!responses.ok()) {
      status_ref = std::move(responses.status());
      done_ref = true;
      return;
    }
    // If the responses is done, the done reference is set to true.
    if (responses->GetTaskState() == TaskState::kDone ||
        responses->GetTaskState() == TaskState::kMaxNumTokensReached) {
      if (responses->GetTaskState() == TaskState::kMaxNumTokensReached) {
        status_ref = absl::InternalError(
            "Maximum kv-cache size reached. Please exit and re-start.");
      }
      EXPECT_FALSE(done_ref);
      done_ref = true;
      return;
    }
    // Accumulate the responses.
    for (int i = 0; i < responses->GetTexts().size(); ++i) {
      responses_ref[i] += responses->GetTexts()[i];
    }
    if (delay_on_next) {
      absl::SleepFor(absl::Milliseconds(50));
    }
  };
}

class TasksTest : public testing::Test {
 protected:
  void SetUp() override {
    auto tokenizer = SentencePieceTokenizer::CreateFromFile(
        (std::filesystem::path(::testing::SrcDir()) / kTestdataDir /
         "sentencepiece.model")
            .string());
    ASSERT_OK(tokenizer);
    tokenizer_ = std::move(*tokenizer);

    auto gemma3_tokenizer = SentencePieceTokenizer::CreateFromFile(
        (std::filesystem::path(::testing::SrcDir()) / kTestdataDir /
         "gemma3_sentencepiece.model")
            .string());
    ASSERT_OK(gemma3_tokenizer);
    gemma3_tokenizer_ = std::move(*gemma3_tokenizer);

    // The prefill tokens are the expected tokens that will be passed in at each
    // time the Tasks::Prefill function is called. The values are the token ids
    // of the input prompt "Hello World!" prepended with the bos token id (2).
    std::vector<std::vector<int>> prefill_tokens = {
        {2, 90, 547, 58, 735, 210, 466, 2294}};
    // The decode tokens are the expected tokens that will be returned by the
    // Decode function. The values are the token ids of the output response
    // "How's it going?" followed by the stop token id (2294).
    std::vector<std::vector<int>> decode_tokens = {{224}, {24}, {8},    {66},
                                                   {246}, {18}, {2295}, {2294}};
    // Vocab size needs to at least be larger than the largest token id 2295.
    executor_ = std::make_unique<FakeLlmExecutor>(
        /*vocab_size=*/2560, prefill_tokens, decode_tokens);
  }

  std::unique_ptr<Tokenizer> tokenizer_;
  std::unique_ptr<Tokenizer> gemma3_tokenizer_;
  std::unique_ptr<FakeLlmExecutor> executor_;
};

TEST_F(TasksTest, PrefillTooLong) {
  const std::string prompt = "Hello World!";
  // Set the max number of tokens to 3.
  executor_->GetMutableExecutorSettings().value()->SetMaxNumTokens(3);
  std::optional<BenchmarkInfo> benchmark_info;

  ASSERT_OK_AND_ASSIGN(std::vector<int> token_ids,
                       tokenizer_->TextToTokenIds(prompt));
  // Prepend the bos token id.
  token_ids.insert(token_ids.begin(), 2);
  ASSERT_OK_AND_ASSIGN(auto token_ids_buffer,
                       tokenizer_->TokenIdsToTensorBuffer(token_ids));
  ExecutorTextData text_data(std::move(token_ids_buffer));
  ExecutorInputs inputs(std::move(text_data), std::nullopt, std::nullopt);

  auto task_respones =
      Tasks::Prefill(*executor_, inputs,
                     /*wait_for_completion=*/true, benchmark_info);

  EXPECT_THAT(task_respones, StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(TasksTest, PrefillSucceed) {
  const std::string prompt = "Hello World!";
  std::optional<BenchmarkInfo> benchmark_info;

  ASSERT_OK_AND_ASSIGN(std::vector<int> token_ids,
                       tokenizer_->TextToTokenIds(prompt));
  // Prepend the bos token id.
  token_ids.insert(token_ids.begin(), 2);
  ASSERT_OK_AND_ASSIGN(auto token_ids_buffer,
                       tokenizer_->TokenIdsToTensorBuffer(token_ids));
  ExecutorTextData text_data(std::move(token_ids_buffer));
  ExecutorInputs inputs(std::move(text_data), std::nullopt, std::nullopt);

  auto task_response =
      Tasks::Prefill(*executor_, inputs,
                     /*wait_for_completion=*/true, benchmark_info);

  EXPECT_OK(task_response);
  EXPECT_EQ(task_response->GetTaskState(), TaskState::kDone);
}

TEST_F(TasksTest, DecodeSucceed) {
  std::optional<BenchmarkInfo> benchmark_info;

  // Run prefill first.
  std::vector<int> prefill_token_ids = {2, 90, 547, 58, 735, 210, 466, 2294};
  ASSERT_OK_AND_ASSIGN(auto token_ids_buffer,
                       tokenizer_->TokenIdsToTensorBuffer(prefill_token_ids));
  ExecutorTextData text_data(std::move(token_ids_buffer));
  ExecutorInputs inputs(std::move(text_data), std::nullopt, std::nullopt);
  auto prefill_responses = Tasks::Prefill(
      *executor_, inputs, /*wait_for_completion=*/true, benchmark_info);
  EXPECT_OK(prefill_responses);

  constexpr int kNumOutputCandidates = 1;
  StopTokenDetector stop_token_detector(kNumOutputCandidates);
  EXPECT_OK(stop_token_detector.AddStopTokenSequence({2294}));
  absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback = nullptr;

  auto task_responses = Tasks::Decode(
      *executor_, *tokenizer_, stop_token_detector, kNumOutputCandidates,
      benchmark_info, /*sampler=*/std::nullopt,
      /*constraint=*/nullptr, /*decoded_ids=*/std::nullopt,
      /*callback=*/callback, /*cancelled=*/nullptr);

  EXPECT_OK(task_responses);
  EXPECT_EQ(task_responses->GetTaskState(), TaskState::kDone);
  // The response is " How's it going?" since "!" is the stop token which is
  // not included in the response.
  EXPECT_EQ(task_responses->GetTexts().size(), 1);
  EXPECT_EQ(task_responses->GetTexts()[0], " How's it going?");
}

TEST_F(TasksTest, DecodeWithTwoStopTokens) {
  std::optional<BenchmarkInfo> benchmark_info;

  // Run prefill first.
  std::vector<int> prefill_token_ids = {2, 90, 547, 58, 735, 210, 466, 2294};
  ASSERT_OK_AND_ASSIGN(auto token_ids_buffer,
                       tokenizer_->TokenIdsToTensorBuffer(prefill_token_ids));
  ExecutorTextData text_data(std::move(token_ids_buffer));
  ExecutorInputs inputs(std::move(text_data), std::nullopt, std::nullopt);
  auto prefill_responses = Tasks::Prefill(
      *executor_, inputs, /*wait_for_completion=*/true, benchmark_info);
  EXPECT_OK(prefill_responses);

  constexpr int kNumOutputCandidates = 1;
  StopTokenDetector stop_token_detector(kNumOutputCandidates);
  EXPECT_OK(stop_token_detector.AddStopTokenSequence({2295, 2294}));
  absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback = nullptr;

  auto responses = Tasks::Decode(
      *executor_, *tokenizer_, stop_token_detector, kNumOutputCandidates,
      benchmark_info, /*sampler=*/std::nullopt,
      /*constraint=*/nullptr, /*decoded_ids=*/std::nullopt,
      /*callback=*/callback, /*cancelled=*/nullptr);
  EXPECT_OK(responses);
  // The response is " How's it going" since "?!" is the stop token which is
  // not included in the response.
  EXPECT_EQ(responses->GetTexts().size(), 1);
  EXPECT_EQ(responses->GetTexts()[0], " How's it going");
}

TEST_F(TasksTest, DecodeReachMaxNumTokens) {
  // Set the max number of tokens to 11.
  executor_->GetMutableExecutorSettings().value()->SetMaxNumTokens(11);
  std::optional<BenchmarkInfo> benchmark_info;

  // Run prefill first.
  std::vector<int> prefill_token_ids = {2, 90, 547, 58, 735, 210, 466, 2294};
  ASSERT_OK_AND_ASSIGN(auto token_ids_buffer,
                       tokenizer_->TokenIdsToTensorBuffer(prefill_token_ids));
  ExecutorTextData text_data(std::move(token_ids_buffer));
  ExecutorInputs inputs(std::move(text_data), std::nullopt, std::nullopt);
  auto prefill_responses = Tasks::Prefill(
      *executor_, inputs, /*wait_for_completion=*/true, benchmark_info);
  EXPECT_OK(prefill_responses);

  constexpr int kNumOutputCandidates = 1;
  StopTokenDetector stop_token_detector(kNumOutputCandidates);
  EXPECT_OK(stop_token_detector.AddStopTokenSequence({2294}));
  absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback = nullptr;

  auto task_responses = Tasks::Decode(
      *executor_, *tokenizer_, stop_token_detector, kNumOutputCandidates,
      benchmark_info, /*sampler=*/std::nullopt,
      /*constraint=*/nullptr, /*decoded_ids=*/std::nullopt,
      /*callback=*/callback, /*cancelled=*/nullptr);

  EXPECT_OK(task_responses);
  EXPECT_EQ(task_responses->GetTaskState(), TaskState::kMaxNumTokensReached);
  // The response is truncated at the max number of tokens.
  EXPECT_EQ(task_responses->GetTexts().size(), 1);
  EXPECT_EQ(task_responses->GetTexts()[0], " How's");
}

TEST_F(TasksTest, DecodeWithMultipleOutputCandidates) {
  constexpr int kNumOutputCandidates = 3;
  // Rebuild the executor with multiple output candidates with the same prefill
  // and decode tokens.
  std::vector<std::vector<int>> prefill_tokens = {
      {2, 90, 547, 58, 735, 210, 466, 2294}};
  // "How's it going?", "Hello World", "How's it going?"
  std::vector<std::vector<int>> decode_tokens = {
      {224, 90, 224},  {24, 547, 24}, {8, 58, 8},         {66, 735, 66},
      {246, 210, 246}, {18, 466, 18}, {2295, 2294, 2295}, {2294, 0, 2294}};
  executor_ = std::make_unique<FakeLlmExecutor>(
      /*vocab_size=*/2560, prefill_tokens, decode_tokens, kNumOutputCandidates);

  std::optional<BenchmarkInfo> benchmark_info;

  // Run prefill first.
  std::vector<int> prefill_token_ids = {2, 90, 547, 58, 735, 210, 466, 2294};
  ASSERT_OK_AND_ASSIGN(auto token_ids_buffer,
                       tokenizer_->TokenIdsToTensorBuffer(prefill_token_ids));
  ExecutorTextData text_data(std::move(token_ids_buffer));
  ExecutorInputs inputs(std::move(text_data), std::nullopt, std::nullopt);
  auto prefill_responses = Tasks::Prefill(
      *executor_, inputs, /*wait_for_completion=*/true, benchmark_info);
  EXPECT_OK(prefill_responses);

  StopTokenDetector stop_token_detector(kNumOutputCandidates);
  EXPECT_OK(stop_token_detector.AddStopTokenSequence({2294}));
  absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback = nullptr;

  auto task_responses = Tasks::Decode(
      *executor_, *tokenizer_, stop_token_detector, kNumOutputCandidates,
      benchmark_info, /*sampler=*/std::nullopt,
      /*constraint=*/nullptr, /*decoded_ids=*/std::nullopt,
      /*callback=*/callback, /*cancelled=*/nullptr);

  EXPECT_OK(task_responses);
  EXPECT_EQ(task_responses->GetTaskState(), TaskState::kDone);
  EXPECT_EQ(task_responses->GetTexts().size(), 3);
  EXPECT_EQ(task_responses->GetTexts()[0], " How's it going?");
  EXPECT_EQ(task_responses->GetTexts()[1], " Hello World");
  EXPECT_EQ(task_responses->GetTexts()[2], " How's it going?");
}

TEST_F(TasksTest, DecodeWithoutPrefillFailed) {
  std::optional<BenchmarkInfo> benchmark_info;
  constexpr int kNumOutputCandidates = 1;
  StopTokenDetector stop_token_detector(kNumOutputCandidates);
  EXPECT_OK(stop_token_detector.AddStopTokenSequence({2294}));
  absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback = nullptr;

  auto task_responses = Tasks::Decode(
      *executor_, *tokenizer_, stop_token_detector, kNumOutputCandidates,
      benchmark_info, /*sampler=*/std::nullopt,
      /*constraint=*/nullptr, /*decoded_ids=*/std::nullopt,
      /*callback=*/callback, /*cancelled=*/nullptr);

  EXPECT_THAT(task_responses, StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST_F(TasksTest, DecodeWithConstrainedDecoding) {
  // Fake constraint that expects " How's it".
  std::vector<int> expected_token_ids = {224, 24, 8, 66, 0};
  auto constraint = std::make_unique<FakeConstraint>(expected_token_ids,
                                                     /*vocabulary_size=*/2560);

  std::vector<std::vector<int>> prefill_tokens = {{2}};
  // The decode tokens are the expected tokens that will be returned by the
  // Decode function. The decoded tokens are " How's it going?!"
  std::vector<std::vector<int>> decode_tokens = {
      {224}, {24}, {8}, {66}, {246}, {18}, {2295}, {2294}, {0}};
  // Vocab size needs to at least be larger than the largest token id 2295.
  auto executor = std::make_unique<FakeLlmExecutor>(
      /*vocab_size=*/2560, prefill_tokens, decode_tokens, /*batch_size=*/1);

  std::optional<BenchmarkInfo> benchmark_info;

  // Run prefill with <bos> token.
  std::vector<int> prefill_token_ids = {2};
  ASSERT_OK_AND_ASSIGN(auto token_ids_buffer,
                       tokenizer_->TokenIdsToTensorBuffer(prefill_token_ids));
  ExecutorTextData text_data(std::move(token_ids_buffer));
  ExecutorInputs inputs(std::move(text_data), std::nullopt, std::nullopt);
  auto prefill_responses = Tasks::Prefill(
      *executor, inputs, /*wait_for_completion=*/true, benchmark_info);
  EXPECT_OK(prefill_responses);

  constexpr int kNumOutputCandidates = 1;
  StopTokenDetector stop_token_detector(kNumOutputCandidates);
  EXPECT_OK(stop_token_detector.AddStopTokenSequence({0}));
  absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback = nullptr;

  auto task_responses = Tasks::Decode(
      *executor, *tokenizer_, stop_token_detector, kNumOutputCandidates,
      benchmark_info, /*sampler=*/std::nullopt, constraint.get(),
      /*decoded_ids=*/std::nullopt, /*callback=*/callback,
      /*cancelled=*/nullptr);

  EXPECT_OK(task_responses);
  EXPECT_EQ(task_responses->GetTaskState(), TaskState::kDone);
  EXPECT_EQ(task_responses->GetTexts().size(), 1);
  EXPECT_EQ(task_responses->GetTexts()[0], " How's it");
}

TEST_F(TasksTest, DecodeStreaming) {
  std::optional<BenchmarkInfo> benchmark_info;

  // Run prefill first.
  std::vector<int> prefill_token_ids = {2, 90, 547, 58, 735, 210, 466, 2294};
  ASSERT_OK_AND_ASSIGN(auto token_ids_buffer,
                       tokenizer_->TokenIdsToTensorBuffer(prefill_token_ids));
  ExecutorTextData text_data(std::move(token_ids_buffer));
  ExecutorInputs inputs(std::move(text_data), std::nullopt, std::nullopt);
  auto prefill_responses = Tasks::Prefill(
      *executor_, inputs, /*wait_for_completion=*/true, benchmark_info);
  EXPECT_OK(prefill_responses);

  constexpr int kNumOutputCandidates = 1;
  StopTokenDetector stop_token_detector(kNumOutputCandidates);
  EXPECT_OK(stop_token_detector.AddStopTokenSequence({2294}));

  std::vector<std::string> responses(kNumOutputCandidates);
  absl::Status status;
  bool done = false;
  auto callback = CreateTestCallback(responses, status, done);

  auto task_status = Tasks::Decode(
      *executor_, *tokenizer_, stop_token_detector, kNumOutputCandidates,
      benchmark_info,
      /*sampler=*/std::nullopt, /*constraint=*/nullptr,
      /*decoded_ids=*/std::nullopt, callback, /*cancelled=*/nullptr);
  callback(task_status);

  EXPECT_OK(task_status);
  EXPECT_EQ(task_status->GetTaskState(), TaskState::kDone);

  // The response is " How's it going?" since "!" is the stop token which is
  // not included in the response.
  EXPECT_EQ(responses[0], " How's it going?");
  EXPECT_TRUE(done);
  EXPECT_OK(status);
}

TEST_F(TasksTest, DecodeStreamingReachMaxNumTokens) {
  // Set the max number of tokens to 11.
  executor_->GetMutableExecutorSettings().value()->SetMaxNumTokens(11);
  std::optional<BenchmarkInfo> benchmark_info;

  // Run prefill first.
  std::vector<int> prefill_token_ids = {2, 90, 547, 58, 735, 210, 466, 2294};
  ASSERT_OK_AND_ASSIGN(auto token_ids_buffer,
                       tokenizer_->TokenIdsToTensorBuffer(prefill_token_ids));
  ExecutorTextData text_data(std::move(token_ids_buffer));
  ExecutorInputs inputs(std::move(text_data), std::nullopt, std::nullopt);
  auto prefill_responses = Tasks::Prefill(
      *executor_, inputs, /*wait_for_completion=*/true, benchmark_info);
  EXPECT_OK(prefill_responses);

  constexpr int kNumOutputCandidates = 1;
  StopTokenDetector stop_token_detector(kNumOutputCandidates);
  EXPECT_OK(stop_token_detector.AddStopTokenSequence({2294}));

  std::vector<std::string> responses(kNumOutputCandidates);
  absl::Status status;
  bool done = false;
  auto callback = CreateTestCallback(responses, status, done);

  auto task_status = Tasks::Decode(
      *executor_, *tokenizer_, stop_token_detector, kNumOutputCandidates,
      benchmark_info,
      /*sampler=*/std::nullopt, /*constraint=*/nullptr,
      /*decoded_ids=*/std::nullopt, callback, /*cancelled=*/nullptr);
  callback(task_status);

  EXPECT_OK(task_status);
  EXPECT_EQ(task_status->GetTaskState(), TaskState::kMaxNumTokensReached);

  // The response is truncated at the max number of tokens.
  EXPECT_EQ(responses[0], " How's");
  EXPECT_TRUE(done);
}

TEST_F(TasksTest, DecodeStreamingWithConstrainedDecoding) {
  // Fake constraint that expects " How's it".
  std::vector<int> expected_token_ids = {224, 24, 8, 66, 0};
  auto constraint = std::make_unique<FakeConstraint>(expected_token_ids,
                                                     /*vocabulary_size=*/2560);

  std::vector<std::vector<int>> prefill_tokens = {{2}};
  // The decode tokens are the expected tokens that will be returned by the
  // Decode function. The decoded tokens are " How's it going?!"
  std::vector<std::vector<int>> decode_tokens = {
      {224}, {24}, {8}, {66}, {246}, {18}, {2295}, {2294}, {0}};
  // Vocab size needs to at least be larger than the largest token id 2295.
  auto executor = std::make_unique<FakeLlmExecutor>(
      /*vocab_size=*/2560, prefill_tokens, decode_tokens, /*batch_size=*/1);

  std::optional<BenchmarkInfo> benchmark_info;

  // Run prefill with <bos> token.
  std::vector<int> prefill_token_ids = {2};
  ASSERT_OK_AND_ASSIGN(auto token_ids_buffer,
                       tokenizer_->TokenIdsToTensorBuffer(prefill_token_ids));
  ExecutorTextData text_data(std::move(token_ids_buffer));
  ExecutorInputs inputs(std::move(text_data), std::nullopt, std::nullopt);
  auto prefill_responses = Tasks::Prefill(
      *executor, inputs, /*wait_for_completion=*/true, benchmark_info);
  EXPECT_OK(prefill_responses);

  constexpr int kNumOutputCandidates = 1;
  StopTokenDetector stop_token_detector(kNumOutputCandidates);
  EXPECT_OK(stop_token_detector.AddStopTokenSequence({0}));

  std::vector<std::string> responses(kNumOutputCandidates);
  absl::Status status;
  bool done = false;
  auto callback = CreateTestCallback(responses, status, done);

  auto task_status = Tasks::Decode(
      *executor, *tokenizer_, stop_token_detector, kNumOutputCandidates,
      benchmark_info,
      /*sampler=*/std::nullopt, /*constraint=*/constraint.get(),
      /*decoded_ids=*/std::nullopt, callback, /*cancelled=*/nullptr);
  callback(task_status);

  EXPECT_OK(task_status);
  EXPECT_EQ(task_status->GetTaskState(), TaskState::kDone);

  EXPECT_EQ(responses[0], " How's it");
  EXPECT_TRUE(done);
}

TEST_F(TasksTest, DecodeBytePairEncodingTokens) {
  auto tokenizer = std::make_unique<BytePairEncodingTokenizer>();
  // Pretend the first and second tokens are incomplete.
  EXPECT_CALL(*tokenizer, TokenIdsToText(std::vector<int>{224}))
      .WillOnce(
          testing::Return(absl::DataLossError("Incomplete BPE sequence")));
  EXPECT_CALL(*tokenizer, TokenIdsToText(std::vector<int>{224, 24}))
      .WillOnce(
          testing::Return(absl::DataLossError("Incomplete BPE sequence")));

  // Now  return a valid token from two tokens.
  EXPECT_CALL(*tokenizer, TokenIdsToText(std::vector<int>{224, 24, 8}))
      .WillOnce(testing::Return(" How's"));

  // Rest proceeds as normal.
  EXPECT_CALL(*tokenizer, TokenIdsToText(std::vector<int>{66}))
      .WillOnce(testing::Return(" "));
  EXPECT_CALL(*tokenizer, TokenIdsToText(std::vector<int>{246}))
      .WillOnce(testing::Return("it"));
  EXPECT_CALL(*tokenizer, TokenIdsToText(std::vector<int>{18}))
      .WillOnce(testing::Return(" "));
  EXPECT_CALL(*tokenizer, TokenIdsToText(std::vector<int>{2295}))
      .WillOnce(testing::Return("going?"));
  EXPECT_CALL(*tokenizer, TokenIdsToText(std::vector<int>{2294}))
      .WillOnce(testing::Return("!"));

  std::optional<BenchmarkInfo> benchmark_info;

  // Run prefill first.
  std::vector<int> prefill_token_ids = {2, 90, 547, 58, 735, 210, 466, 2294};
  ASSERT_OK_AND_ASSIGN(auto token_ids_buffer,
                       tokenizer_->TokenIdsToTensorBuffer(prefill_token_ids));
  ExecutorTextData text_data(std::move(token_ids_buffer));
  ExecutorInputs inputs(std::move(text_data), std::nullopt, std::nullopt);
  auto prefill_responses = Tasks::Prefill(
      *executor_, inputs, /*wait_for_completion=*/true, benchmark_info);
  EXPECT_OK(prefill_responses);

  constexpr int kNumOutputCandidates = 1;
  StopTokenDetector stop_token_detector(kNumOutputCandidates);
  EXPECT_OK(stop_token_detector.AddStopTokenSequence({2294}));
  absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback = nullptr;

  auto task_responses = Tasks::Decode(
      *executor_, *tokenizer, stop_token_detector, kNumOutputCandidates,
      benchmark_info, /*sampler=*/std::nullopt,
      /*constraint=*/nullptr, /*decoded_ids=*/std::nullopt,
      /*callback=*/callback, /*cancelled=*/nullptr);

  EXPECT_OK(task_responses);
  EXPECT_EQ(task_responses->GetTaskState(), TaskState::kDone);
  // The response is " How's it going?" since "!" is the stop token which is
  // not included in the response.
  EXPECT_EQ(task_responses->GetTexts().size(), 1);
  EXPECT_EQ(task_responses->GetTexts()[0], " How's it going?");
}

TEST_F(TasksTest, DecodeStopTokenIsPartialBytePairEncodingTokens) {
  auto tokenizer = std::make_unique<BytePairEncodingTokenizer>();
  // Pretend the first and second tokens are incomplete.
  EXPECT_CALL(*tokenizer, TokenIdsToText(std::vector<int>{224}))
      .WillOnce(
          testing::Return(absl::DataLossError("Incomplete BPE sequence")));
  EXPECT_CALL(*tokenizer, TokenIdsToText(std::vector<int>{224, 24}))
      .WillOnce(
          testing::Return(absl::DataLossError("Incomplete BPE sequence")));

  // No need to call the tokenizer again as the stop token is encoded as a
  // partial byte pair encoding token.

  std::optional<BenchmarkInfo> benchmark_info;

  // Run prefill first.
  std::vector<int> prefill_token_ids = {2, 90, 547, 58, 735, 210, 466, 2294};
  ASSERT_OK_AND_ASSIGN(auto token_ids_buffer,
                       tokenizer_->TokenIdsToTensorBuffer(prefill_token_ids));
  ExecutorTextData text_data(std::move(token_ids_buffer));
  ExecutorInputs inputs(std::move(text_data), std::nullopt, std::nullopt);
  auto prefill_responses = Tasks::Prefill(
      *executor_, inputs, /*wait_for_completion=*/true, benchmark_info);
  EXPECT_OK(prefill_responses);

  constexpr int kNumOutputCandidates = 1;
  StopTokenDetector stop_token_detector(kNumOutputCandidates);
  EXPECT_OK(stop_token_detector.AddStopTokenSequence({224, 24}));
  absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback = nullptr;

  auto task_responses = Tasks::Decode(
      *executor_, *tokenizer, stop_token_detector, kNumOutputCandidates,
      benchmark_info, /*sampler=*/std::nullopt,
      /*constraint=*/nullptr, /*decoded_ids=*/std::nullopt,
      /*callback=*/callback, /*cancelled=*/nullptr);

  EXPECT_OK(task_responses);
  EXPECT_EQ(task_responses->GetTaskState(), TaskState::kDone);
  // Empty response as the stop token is encoded as a partial byte pair encoding
  // token.
  EXPECT_EQ(task_responses->GetTexts().size(), 1);
  EXPECT_EQ(task_responses->GetTexts()[0], "");
}

TEST_F(TasksTest, DecodeConsecutiveByteTokens) {
  constexpr int kNumOutputCandidates = 1;
  constexpr int kVocabSize = 262144;
  std::vector<std::vector<int>> prefill_tokens = {{2}};
  // <0xC2> (432), <0xB0> (414) -> "°"
  std::vector<std::vector<int>> decode_tokens = {{432}, {414}, {0}};

  auto executor = std::make_unique<FakeLlmExecutor>(
      kVocabSize, prefill_tokens, decode_tokens, kNumOutputCandidates);

  std::optional<BenchmarkInfo> benchmark_info;

  // Run prefill first.
  std::vector<int> prefill_token_ids = {2};
  ASSERT_OK_AND_ASSIGN(
      auto token_ids_buffer,
      gemma3_tokenizer_->TokenIdsToTensorBuffer(prefill_token_ids));
  ExecutorTextData text_data(std::move(token_ids_buffer));
  ExecutorInputs inputs(std::move(text_data), std::nullopt, std::nullopt);
  auto prefill_responses = Tasks::Prefill(
      *executor, inputs, /*wait_for_completion=*/true, benchmark_info);
  EXPECT_OK(prefill_responses);

  StopTokenDetector stop_token_detector(kNumOutputCandidates);
  EXPECT_OK(stop_token_detector.AddStopTokenSequence({0}));

  std::vector<std::string> step_results;
  absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback =
      [&](absl::StatusOr<Responses> responses) {
        ASSERT_OK(responses);
        if (responses->GetTaskState() == TaskState::kProcessing) {
          ASSERT_EQ(responses->GetTexts().size(), 1);
          step_results.push_back(responses->GetTexts()[0]);
        }
      };

  auto task_responses = Tasks::Decode(
      *executor, *gemma3_tokenizer_, stop_token_detector, kNumOutputCandidates,
      benchmark_info, /*sampler=*/std::nullopt,
      /*constraint=*/nullptr, /*decoded_ids=*/std::nullopt,
      /*callback=*/callback, /*cancelled=*/nullptr);

  EXPECT_OK(task_responses);

  // 432 -> buffered, output ""
  // 414 -> flushed "°"
  ASSERT_EQ(step_results.size(), 1);
  EXPECT_EQ(step_results[0], "°");
}

TEST_F(TasksTest, DecodeConsecutiveByteTokensWithNonByteTokens) {
  constexpr int kNumOutputCandidates = 1;
  constexpr int kVocabSize = 262144;
  std::vector<std::vector<int>> prefill_tokens = {{2}};
  // <0x6B> (345), <0x6D> (347), <0xC2> (432), <0xB2> (416) -> km²
  std::vector<std::vector<int>> decode_tokens = {
      {345}, {347}, {432}, {416}, {0}};

  auto executor = std::make_unique<FakeLlmExecutor>(
      kVocabSize, prefill_tokens, decode_tokens, kNumOutputCandidates);

  std::optional<BenchmarkInfo> benchmark_info;

  // Run prefill first.
  std::vector<int> prefill_token_ids = {2};
  ASSERT_OK_AND_ASSIGN(
      auto token_ids_buffer,
      gemma3_tokenizer_->TokenIdsToTensorBuffer(prefill_token_ids));
  ExecutorTextData text_data(std::move(token_ids_buffer));
  ExecutorInputs inputs(std::move(text_data), std::nullopt, std::nullopt);
  auto prefill_responses = Tasks::Prefill(
      *executor, inputs, /*wait_for_completion=*/true, benchmark_info);
  EXPECT_OK(prefill_responses);

  StopTokenDetector stop_token_detector(kNumOutputCandidates);
  EXPECT_OK(stop_token_detector.AddStopTokenSequence({0}));

  std::vector<std::string> step_results;
  absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback =
      [&](absl::StatusOr<Responses> responses) {
        ASSERT_OK(responses);
        if (responses->GetTaskState() == TaskState::kProcessing) {
          ASSERT_EQ(responses->GetTexts().size(), 1);
          step_results.push_back(responses->GetTexts()[0]);
        }
      };

  auto task_responses = Tasks::Decode(
      *executor, *gemma3_tokenizer_, stop_token_detector, kNumOutputCandidates,
      benchmark_info, /*sampler=*/std::nullopt,
      /*constraint=*/nullptr, /*decoded_ids=*/std::nullopt,
      /*callback=*/callback, /*cancelled=*/nullptr);

  EXPECT_OK(task_responses);

  // 345 -> "k"
  // 347 -> "m"
  // 432 -> ""
  // 416 -> "²"
  ASSERT_EQ(step_results.size(), 3);
  EXPECT_EQ(step_results[0], "k");
  EXPECT_EQ(step_results[1], "m");
  EXPECT_EQ(step_results[2], "²");
}

TEST_F(TasksTest, DecodeConsecutiveByteTokensWithPartialBpeIgnored) {
  constexpr int kNumOutputCandidates = 1;
  constexpr int kVocabSize = 262144;
  std::vector<std::vector<int>> prefill_tokens = {{2}};
  // <0x6B> (345), <0x6D> (347), <0xC2> (432), <0xB2> (416) -> "km²"
  // Ignore 416 as it is after the stop token 0.
  std::vector<std::vector<int>> decode_tokens = {
      {345}, {347}, {432}, {0}, {416}};

  auto executor = std::make_unique<FakeLlmExecutor>(
      kVocabSize, prefill_tokens, decode_tokens, kNumOutputCandidates);

  std::optional<BenchmarkInfo> benchmark_info;

  // Run prefill first.
  std::vector<int> prefill_token_ids = {2};
  ASSERT_OK_AND_ASSIGN(
      auto token_ids_buffer,
      gemma3_tokenizer_->TokenIdsToTensorBuffer(prefill_token_ids));
  ExecutorTextData text_data(std::move(token_ids_buffer));
  ExecutorInputs inputs(std::move(text_data), std::nullopt, std::nullopt);
  auto prefill_responses = Tasks::Prefill(
      *executor, inputs, /*wait_for_completion=*/true, benchmark_info);
  EXPECT_OK(prefill_responses);

  StopTokenDetector stop_token_detector(kNumOutputCandidates);
  EXPECT_OK(stop_token_detector.AddStopTokenSequence({0}));

  std::vector<std::string> step_results;
  absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback =
      [&](absl::StatusOr<Responses> responses) {
        ASSERT_OK(responses);
        if (responses->GetTaskState() == TaskState::kProcessing) {
          ASSERT_EQ(responses->GetTexts().size(), 1);
          step_results.push_back(responses->GetTexts()[0]);
        }
      };

  auto task_responses = Tasks::Decode(
      *executor, *gemma3_tokenizer_, stop_token_detector, kNumOutputCandidates,
      benchmark_info, /*sampler=*/std::nullopt,
      /*constraint=*/nullptr, /*decoded_ids=*/std::nullopt,
      /*callback=*/callback, /*cancelled=*/nullptr);

  EXPECT_OK(task_responses);

  // 345 -> "k"
  // 347 -> "m"
  // 432 -> ""
  ASSERT_EQ(step_results.size(), 2);
  EXPECT_EQ(step_results[0], "k");
  EXPECT_EQ(step_results[1], "m");
}

class TasksCustomSamplingTest : public testing::Test {
 protected:
  void SetUp() override {
    auto tokenizer = SentencePieceTokenizer::CreateFromFile(
        (std::filesystem::path(::testing::SrcDir()) / kTestdataDir /
         "sentencepiece.model")
            .string());
    ASSERT_OK(tokenizer);
    tokenizer_ = std::move(*tokenizer);

    auto gemma3_tokenizer = SentencePieceTokenizer::CreateFromFile(
        (std::filesystem::path(::testing::SrcDir()) / kTestdataDir /
         "gemma3_sentencepiece.model")
            .string());
    ASSERT_OK(gemma3_tokenizer);
    gemma3_tokenizer_ = std::move(*gemma3_tokenizer);
  }

  FakeLlmExecutor CreateFakeLlmExecutor(
      // The expected prefill tokens that after stop tokens are found in
      // decoding with CustomSampling. That is, the last sampled tokens at
      // stop condition.
      const std::vector<std::vector<int>>& prefill_tokens = {},
      // The expected decode tokens that will be returned by the Decode
      // function.
      const std::vector<std::vector<int>>& decode_tokens = {},
      // Vocab size needs to at least be larger than the largest token id 2295.
      int vocab_size = 2560, int batch_size = 2) {
    return FakeLlmExecutor(vocab_size, prefill_tokens, decode_tokens,
                           batch_size);
  }

  absl::StatusOr<Responses> ApplyScore(
      const std::vector<std::vector<int>>& prefill_tokens,
      const std::vector<std::vector<int>>& decode_tokens, int vocab_size,
      int batch_size, const std::vector<absl::string_view>& target_texts,
      bool store_token_lengths = false) {
    auto decoded_ids = CreateTensorBuffer<int>(/*dimensions=*/{batch_size, 1});
    EXPECT_TRUE(decoded_ids.HasValue());

    StopTokenDetector stop_token_detector(batch_size);
    auto status =
        stop_token_detector.AddStopTokenSequence(/*stop_sequence=*/{0});
    RETURN_IF_ERROR(status);
    auto executor = CreateFakeLlmExecutor(prefill_tokens, decode_tokens,
                                          vocab_size, batch_size);

    std::optional<BenchmarkInfo> benchmark_info;

    // Run prefill with <bos> token.
    std::vector<int> prefill_token_ids = {2};
    ASSIGN_OR_RETURN(auto token_ids_buffer,
                     tokenizer_->TokenIdsToTensorBuffer(prefill_token_ids));
    ExecutorTextData text_data(std::move(token_ids_buffer));
    ExecutorInputs inputs(std::move(text_data), std::nullopt, std::nullopt);
    auto prefill_responses = Tasks::Prefill(
        executor, inputs, /*wait_for_completion=*/true, benchmark_info);
    EXPECT_OK(prefill_responses);

    return Tasks::Score(executor, *tokenizer_, target_texts,
                        /*temperature=*/1.0f, std::move(decoded_ids.Value()),
                        store_token_lengths);
  }

  std::unique_ptr<Tokenizer> tokenizer_;
  std::unique_ptr<Tokenizer> gemma3_tokenizer_;
};

TEST_F(TasksCustomSamplingTest, PrefillSucceed) {
  const std::string prompt = "Hello World!";
  std::optional<BenchmarkInfo> benchmark_info;
  ASSERT_OK_AND_ASSIGN(std::vector<int> token_ids,
                       tokenizer_->TextToTokenIds(prompt));
  // Prepend the bos token id.
  token_ids.insert(token_ids.begin(), 2);
  ASSERT_OK_AND_ASSIGN(auto token_ids_buffer,
                       tokenizer_->TokenIdsToTensorBuffer(token_ids));
  ExecutorTextData text_data(std::move(token_ids_buffer));
  ExecutorInputs inputs(std::move(text_data), std::nullopt, std::nullopt);

  auto executor = CreateFakeLlmExecutor(
      // "Hello World!" prepended with the bos token id (2).
      /*prefill_tokens=*/{{2, 90, 547, 58, 735, 210, 466, 2294}});
  auto task_responses =
      Tasks::Prefill(executor, inputs,
                     /*wait_for_completion=*/true, benchmark_info);
  EXPECT_OK(task_responses);
  EXPECT_EQ(task_responses->GetTaskState(), TaskState::kDone);
}

TEST_F(TasksCustomSamplingTest, PrefillTooLong) {
  auto executor = CreateFakeLlmExecutor(
      // "Hello World!" prepended with the bos token id (2).
      /*prefill_tokens=*/{{2, 90, 547, 58, 735, 210, 466, 2294}});
  // Set the max number of tokens to 3.
  executor.GetMutableExecutorSettings().value()->SetMaxNumTokens(3);
  const std::string prompt = "Hello World!";
  std::optional<BenchmarkInfo> benchmark_info;
  ASSERT_OK_AND_ASSIGN(std::vector<int> token_ids,
                       tokenizer_->TextToTokenIds(prompt));
  // Prepend the bos token id.
  token_ids.insert(token_ids.begin(), 2);
  ASSERT_OK_AND_ASSIGN(auto token_ids_buffer,
                       tokenizer_->TokenIdsToTensorBuffer(token_ids));
  ExecutorTextData text_data(std::move(token_ids_buffer));
  ExecutorInputs inputs(std::move(text_data), std::nullopt, std::nullopt);

  auto task_responses =
      Tasks::Prefill(executor, inputs,
                     /*wait_for_completion=*/true, benchmark_info);
  EXPECT_THAT(task_responses, StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(TasksCustomSamplingTest, DecodeCustomSampling) {
  auto sampler_or =
      TopPSampler::Create(/*k=*/1, /*p=*/0.5, /*temperature=*/1.0,
                          /*batch_size=*/2, /*sequence_size=*/1, /*seed=*/1);
  EXPECT_TRUE(sampler_or.ok());
  std::unique_ptr<TopPSampler> sampler = std::move(sampler_or.value());

  auto decoded_ids = CreateTensorBuffer<int>({2, 1});
  EXPECT_TRUE(decoded_ids.HasValue());
  std::optional<BenchmarkInfo> benchmark_info;
  StopTokenDetector stop_token_detector(2);
  EXPECT_OK(stop_token_detector.AddStopTokenSequence({0}));

  auto executor = CreateFakeLlmExecutor(
      // The expected prefill tokens that after stop tokens are found in
      // decoding with CustomSampling. That is, the last sampled tokens at stop
      // condition.
      /*prefill_tokens=*/{{2}, {0, 0}},
      // " How's it going?!" and " Hello World!" followed by the stop token id
      // (0).
      /*decode_tokens=*/{{224, 90},
                         {24, 547},
                         {8, 58},
                         {66, 735},
                         {246, 210},
                         {18, 466},
                         {2295, 2294},
                         {2294, 0},
                         {0, 0}});

  // Run Prefill with <bos> token.
  std::vector<int> prefill_token_ids = {2};
  ASSERT_OK_AND_ASSIGN(auto token_ids_buffer,
                       tokenizer_->TokenIdsToTensorBuffer(prefill_token_ids));
  ExecutorTextData text_data(std::move(token_ids_buffer));
  ExecutorInputs inputs(std::move(text_data), std::nullopt, std::nullopt);
  auto prefill_responses = Tasks::Prefill(
      executor, inputs, /*wait_for_completion=*/true, benchmark_info);
  EXPECT_OK(prefill_responses);

  absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback = nullptr;

  auto task_responses =
      Tasks::Decode(executor, *tokenizer_, stop_token_detector,
                    /*num_output_candidates=*/2, benchmark_info, sampler.get(),
                    /*constraint=*/nullptr, std::move(decoded_ids.Value()),
                    /*callback=*/callback,
                    /*cancelled=*/nullptr);
  EXPECT_OK(task_responses);
  EXPECT_EQ(task_responses->GetTaskState(), TaskState::kDone);
  EXPECT_EQ(task_responses->GetTexts().size(), 2);
  // First candidate: " How's it going?!".
  EXPECT_EQ(task_responses->GetTexts()[0], " How's it going?!");
  // Second candidate: " Hello World!".
  EXPECT_EQ(task_responses->GetTexts()[1], " Hello World!");

  // The scores are all equal to 0.0f (log(1.0f)).
  EXPECT_EQ(task_responses->GetScores().size(), 2);
  EXPECT_EQ(task_responses->GetScores()[0], 0.0f);
  EXPECT_EQ(task_responses->GetScores()[1], 0.0f);
}

TEST_F(TasksCustomSamplingTest, DecodeCustomSamplingWithConstrainedDecoding) {
  auto sampler_or =
      TopPSampler::Create(/*k=*/1, /*p=*/0.5, /*temperature=*/1.0,
                          /*batch_size=*/2, /*sequence_size=*/1, /*seed=*/1);
  EXPECT_TRUE(sampler_or.ok());
  std::unique_ptr<TopPSampler> sampler = std::move(sampler_or.value());

  // Fake constraint that expects " How's it".
  std::vector<int> expected_token_ids = {224, 24, 8, 66, 0};
  auto constraint = std::make_unique<FakeConstraint>(expected_token_ids,
                                                     /*vocabulary_size=*/2560);

  // Vocab size needs to at least be larger than the largest token id 2295.
  auto executor = CreateFakeLlmExecutor(
      // The expected prefill tokens that after stop tokens are found in
      // decoding with CustomSampling. That is, the last sampled tokens at stop
      // condition.
      /*prefill_tokens=*/{{2}, {0, 0}},
      // " How's it going?!" for both two batches because the constraint is
      // applied.
      /*decode_tokens=*/{{224, 224},
                         {24, 24},
                         {8, 8},
                         {66, 66},
                         {246, 246},
                         {18, 18},
                         {2295, 2295},
                         {2294, 2294},
                         {0, 0}}

  );

  std::optional<BenchmarkInfo> benchmark_info;

  // Run prefill with <bos> token.
  std::vector<int> prefill_token_ids = {2};
  ASSERT_OK_AND_ASSIGN(auto token_ids_buffer,
                       tokenizer_->TokenIdsToTensorBuffer(prefill_token_ids));
  ExecutorTextData text_data(std::move(token_ids_buffer));
  ExecutorInputs inputs(std::move(text_data), std::nullopt, std::nullopt);
  auto prefill_responses = Tasks::Prefill(
      executor, inputs, /*wait_for_completion=*/true, benchmark_info);
  EXPECT_OK(prefill_responses);

  auto decoded_ids = CreateTensorBuffer<int>({2, 1});
  EXPECT_TRUE(decoded_ids.HasValue());
  // Populate with the last pre-filled token.
  decoded_ids->Write<int>({224, 224});
  StopTokenDetector stop_token_detector(2);
  EXPECT_OK(stop_token_detector.AddStopTokenSequence({0}));
  absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback = nullptr;

  auto task_responses = Tasks::Decode(
      executor, *tokenizer_, stop_token_detector,
      /*num_output_candidates=*/2, benchmark_info, sampler.get(),
      constraint.get(), std::move(decoded_ids.Value()), /*callback=*/callback,
      /*cancelled=*/nullptr);

  EXPECT_OK(task_responses);
  EXPECT_EQ(task_responses->GetTaskState(), TaskState::kDone);
  EXPECT_EQ(task_responses->GetTexts().size(), 2);
  // First candidate: " How's it".
  EXPECT_EQ(task_responses->GetTexts()[0], " How's it");
  // Second candidate: " How's it".
  EXPECT_EQ(task_responses->GetTexts()[1], " How's it");
}

TEST_F(TasksCustomSamplingTest,
       DecodeCustomSamplingWithPartialBytePairEncodingTokens) {
  ASSERT_OK_AND_ASSIGN(
      auto sampler,
      TopPSampler::Create(/*k=*/1, /*p=*/0.5,
                          /*temperature=*/0.0,
                          /*batch_size=*/2, /*sequence_size=*/1, /*seed=*/1));

  // <432, 416> --> "km²"
  // <432, 414> --> "°"
  auto executor = CreateFakeLlmExecutor(
      // The expected prefill tokens that after stop tokens are found in
      // decoding with CustomSampling. That is, the last sampled tokens at stop
      // condition.
      /*prefill_tokens=*/{{2}, {0, 0}},
      /*decode_tokens=*/{{345, 345}, {347, 432}, {432, 414}, {416, 0}, {0, 0}});

  std::optional<BenchmarkInfo> benchmark_info;

  // Run prefill first.
  std::vector<int> prefill_token_ids = {2};
  ASSERT_OK_AND_ASSIGN(
      auto token_ids_buffer,
      gemma3_tokenizer_->TokenIdsToTensorBuffer(prefill_token_ids));
  ExecutorTextData text_data(std::move(token_ids_buffer));
  ExecutorInputs inputs(std::move(text_data), std::nullopt, std::nullopt);
  auto prefill_responses = Tasks::Prefill(
      executor, inputs, /*wait_for_completion=*/true, benchmark_info);
  EXPECT_OK(prefill_responses);

  StopTokenDetector stop_token_detector(2);
  EXPECT_OK(stop_token_detector.AddStopTokenSequence({0}));
  auto decoded_ids = CreateTensorBuffer<int>({2, 1});
  EXPECT_TRUE(decoded_ids.HasValue());
  decoded_ids->Write<int>({345, 345});

  std::vector<std::string> step_results;
  absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback = nullptr;

  ASSERT_OK_AND_ASSIGN(
      auto task_responses,
      Tasks::Decode(executor, *gemma3_tokenizer_, stop_token_detector,
                    /*num_output_candidates=*/2, benchmark_info, sampler.get(),
                    /*constraint=*/nullptr, std::move(decoded_ids.Value()),
                    /*callback=*/callback,
                    /*cancelled=*/nullptr));

  EXPECT_EQ(task_responses.GetTaskState(), TaskState::kDone);
  EXPECT_EQ(task_responses.GetTexts().size(), 2);
  EXPECT_EQ(task_responses.GetTexts()[0], "km²");
  EXPECT_EQ(task_responses.GetTexts()[1], "k°");
}

TEST_F(TasksCustomSamplingTest,
       ScoreCustomSamplingSingleBatchWithoutTokenLengths) {
  const auto responses_without_token_lengths = ApplyScore(
      /*prefill_tokens=*/{{2}},
      /*decode_tokens=*/{{90}, {547}, {58}, {735}, {210}, {466}, {2294}, {0}},
      /*vocab_size=*/2560,
      /*batch_size=*/1, /*target_texts=*/{"Hello World!"},
      /*store_token_lengths=*/false);
  ASSERT_OK(responses_without_token_lengths);
  // Expect a single output candidate.
  EXPECT_EQ(responses_without_token_lengths->GetScores().size(), 1);
  // The fake executor returns the decode tokens deterministically.
  // This corresponds to the log probability of the target text "Hello World!"
  // being generated by the model. The log probability is 0.0f because the
  // decode tokens are the same as the target text.
  EXPECT_EQ(responses_without_token_lengths->GetScores()[0], 0.0f);
  EXPECT_FALSE(responses_without_token_lengths->GetTokenLengths().has_value());
  ASSERT_TRUE(responses_without_token_lengths->GetTokenScores().has_value());
  EXPECT_EQ(responses_without_token_lengths->GetTokenScores()->size(), 1);
  EXPECT_EQ(responses_without_token_lengths->GetTokenScores()->at(0).size(), 7);
  EXPECT_THAT(responses_without_token_lengths->GetTokenScores()->at(0),
              testing::Each(0.0f));
}

TEST_F(TasksCustomSamplingTest,
       ScoreCustomSamplingSingleBatchWithTokenLengths) {
  const auto responses_with_token_lengths = ApplyScore(
      /*prefill_tokens=*/{{2}},
      /*decode_tokens=*/{{90}, {547}, {58}, {735}, {210}, {466}, {2294}, {0}},
      /*vocab_size=*/2560,
      /*batch_size=*/1, /*target_texts=*/{"Hello World!"},
      /*store_token_lengths=*/true);
  ASSERT_OK(responses_with_token_lengths);
  // Expect a single output candidate.
  EXPECT_EQ(responses_with_token_lengths->GetScores().size(), 1);
  // The fake executor returns the decode tokens deterministically.
  // This corresponds to the log probability of the target text "Hello World!"
  // being generated by the model. The log probability is 0.0f because the
  // decode tokens are the same as the target text.
  EXPECT_EQ(responses_with_token_lengths->GetScores()[0], 0.0f);
  EXPECT_TRUE(responses_with_token_lengths->GetTokenLengths().has_value());
  EXPECT_EQ(responses_with_token_lengths->GetTokenLengths()->size(), 1);
  EXPECT_EQ((*responses_with_token_lengths->GetTokenLengths())[0], 7);
  ASSERT_TRUE(responses_with_token_lengths->GetTokenScores().has_value());
  EXPECT_EQ(responses_with_token_lengths->GetTokenScores()->size(), 1);
  EXPECT_EQ(responses_with_token_lengths->GetTokenScores()->at(0).size(), 7);
  EXPECT_THAT(responses_with_token_lengths->GetTokenScores()->at(0),
              testing::Each(0.0f));
}

TEST_F(TasksCustomSamplingTest,
       ScoreCustomSamplingMultiBatchWithoutTokenLengths) {
  const auto task_responses_without_token_lengths = ApplyScore(
      /*prefill_tokens=*/{{2}},
      /*decode_tokens=*/
      {{224, 90},
       {24, 547},
       {8, 58},
       {66, 735},
       {246, 210},
       {18, 466},
       {2295, 2294},
       {2294, 0},
       {0, 0}},
      /*vocab_size=*/2560,
      /*batch_size=*/2, /*target_texts=*/{"How's it going?", "Hello World!"},
      /*store_token_lengths=*/false);
  ASSERT_OK(task_responses_without_token_lengths);
  EXPECT_EQ(task_responses_without_token_lengths->GetTaskState(),
            TaskState::kDone);
  // Expect a single output candidate.
  EXPECT_EQ(task_responses_without_token_lengths->GetScores().size(), 2);
  // The fake executor returns the decode tokens deterministically.
  // These correspond to the log probabilities of the target texts
  // "How's it going?" and "Hello World!" being generated by the model. The
  // log probabilities are 0.0f because the decode tokens are the same as the
  // target texts.
  EXPECT_EQ(task_responses_without_token_lengths->GetScores()[0], 0.0f);
  EXPECT_EQ(task_responses_without_token_lengths->GetScores()[1], 0.0f);
  EXPECT_FALSE(
      task_responses_without_token_lengths->GetTokenLengths().has_value());
  ASSERT_TRUE(
      task_responses_without_token_lengths->GetTokenScores().has_value());
  EXPECT_EQ(task_responses_without_token_lengths->GetTokenScores()->size(), 2);
  EXPECT_EQ(
      task_responses_without_token_lengths->GetTokenScores()->at(0).size(), 7);
  EXPECT_THAT(task_responses_without_token_lengths->GetTokenScores()->at(0),
              testing::Each(0.0f));
  EXPECT_EQ(
      task_responses_without_token_lengths->GetTokenScores()->at(1).size(), 7);
  EXPECT_THAT(task_responses_without_token_lengths->GetTokenScores()->at(1),
              testing::Each(0.0f));
}

TEST_F(TasksCustomSamplingTest, ScoreCustomSamplingMultiBatchWithTokenLengths) {
  const auto task_responses_with_token_lengths = ApplyScore(
      /*prefill_tokens=*/{{2}},
      /*decode_tokens=*/
      {{224, 90},
       {24, 547},
       {8, 58},
       {66, 735},
       {246, 210},
       {18, 466},
       {2295, 2294},
       {2294, 0},
       {0, 0}},
      /*vocab_size=*/2560,
      /*batch_size=*/2, /*target_texts=*/{"How's it going?", "Hello World!"},
      /*store_token_lengths=*/true);
  ASSERT_OK(task_responses_with_token_lengths);
  EXPECT_EQ(task_responses_with_token_lengths->GetTaskState(),
            TaskState::kDone);
  // Expect a single output candidate.
  EXPECT_EQ(task_responses_with_token_lengths->GetScores().size(), 2);
  // The fake executor returns the decode tokens deterministically.
  // These correspond to the log probabilities of the target texts
  // "How's it going?" and "Hello World!" being generated by the model. The
  // log probabilities are 0.0f because the decode tokens are the same as the
  // target texts.
  EXPECT_EQ(task_responses_with_token_lengths->GetScores()[0], 0.0f);
  EXPECT_EQ(task_responses_with_token_lengths->GetScores()[1], 0.0f);
  EXPECT_TRUE(task_responses_with_token_lengths->GetTokenLengths().has_value());
  EXPECT_EQ(task_responses_with_token_lengths->GetTokenLengths()->size(), 2);
  EXPECT_EQ((*task_responses_with_token_lengths->GetTokenLengths())[0], 7);
  EXPECT_EQ((*task_responses_with_token_lengths->GetTokenLengths())[1], 7);
  ASSERT_TRUE(task_responses_with_token_lengths->GetTokenScores().has_value());
  EXPECT_EQ(task_responses_with_token_lengths->GetTokenScores()->size(), 2);
  EXPECT_EQ(task_responses_with_token_lengths->GetTokenScores()->at(0).size(),
            7);
  EXPECT_THAT(task_responses_with_token_lengths->GetTokenScores()->at(0),
              testing::Each(0.0f));
  EXPECT_EQ(task_responses_with_token_lengths->GetTokenScores()->at(1).size(),
            7);
  EXPECT_THAT(task_responses_with_token_lengths->GetTokenScores()->at(1),
              testing::Each(0.0f));
}

TEST_F(TasksCustomSamplingTest, DecodeCustomSamplingReachMaxNumTokens) {
  auto executor = CreateFakeLlmExecutor(
      /*prefill_tokens=*/{{2}, {8, 58}},
      /*decode_tokens=*/{{224, 90},
                         {24, 547},
                         {8, 58},  // Stop here because of max num tokens.
                         {66, 735},
                         {246, 210},
                         {18, 466},
                         {2295, 2294},
                         {2294, 0},
                         {0, 0}});
  // Set the max number of tokens to 4.
  executor.GetMutableExecutorSettings().value()->SetMaxNumTokens(4);

  std::optional<BenchmarkInfo> benchmark_info;

  // Run prefill with <bos> token.
  std::vector<int> prefill_token_ids = {2};
  ASSERT_OK_AND_ASSIGN(auto token_ids_buffer,
                       tokenizer_->TokenIdsToTensorBuffer(prefill_token_ids));
  ExecutorTextData text_data(std::move(token_ids_buffer));
  ExecutorInputs inputs(std::move(text_data), std::nullopt, std::nullopt);
  auto prefill_responses = Tasks::Prefill(
      executor, inputs, /*wait_for_completion=*/true, benchmark_info);
  EXPECT_OK(prefill_responses);

  auto sampler_or =
      TopPSampler::Create(/*k=*/1, /*p=*/0.5, /*temperature=*/1.0,
                          /*batch_size=*/2, /*sequence_size=*/1, /*seed=*/1);
  EXPECT_TRUE(sampler_or.ok());
  std::unique_ptr<TopPSampler> sampler = std::move(sampler_or.value());

  auto decoded_ids = CreateTensorBuffer<int>({2, 1});
  EXPECT_TRUE(decoded_ids.HasValue());
  StopTokenDetector stop_token_detector(2);
  EXPECT_OK(stop_token_detector.AddStopTokenSequence({0}));
  absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback = nullptr;

  auto task_responses =
      Tasks::Decode(executor, *tokenizer_, stop_token_detector,
                    /*num_output_candidates=*/2, benchmark_info, sampler.get(),
                    /*constraint=*/nullptr, std::move(decoded_ids.Value()),
                    /*callback=*/callback,
                    /*cancelled=*/nullptr);
  EXPECT_OK(task_responses);
  EXPECT_EQ(task_responses->GetTaskState(), TaskState::kMaxNumTokensReached);
  EXPECT_EQ(task_responses->GetTexts().size(), 2);
  // First candidate truncated at max number of tokens: " How's".
  EXPECT_EQ(task_responses->GetTexts()[0], " How's");
  // Second candidate truncated at max number of tokens: " Hello".
  EXPECT_EQ(task_responses->GetTexts()[1], " Hello");
}

TEST_F(TasksCustomSamplingTest, DecodeCustomSamplingStreaming) {
  auto sampler_or =
      TopPSampler::Create(/*k=*/1, /*p=*/0.5, /*temperature=*/1.0,
                          /*batch_size=*/2, /*sequence_size=*/1, /*seed=*/1);
  EXPECT_TRUE(sampler_or.ok());
  std::unique_ptr<TopPSampler> sampler = std::move(sampler_or.value());

  auto decoded_ids = CreateTensorBuffer<int>({2, 1});
  EXPECT_TRUE(decoded_ids.HasValue());
  std::optional<BenchmarkInfo> benchmark_info;

  StopTokenDetector stop_token_detector(2);
  EXPECT_OK(stop_token_detector.AddStopTokenSequence({0}));
  EXPECT_OK(stop_token_detector.AddStopTokenSequence({2295, 2294}));

  std::vector<std::string> responses(2);
  absl::Status status;
  bool done = false;
  auto executor = CreateFakeLlmExecutor(
      // The expected prefill tokens that after stop tokens are found in
      // decoding with CustomSampling. That is, the last sampled tokens at stop
      // condition.
      /*prefill_tokens=*/{{2}, {2294, 0}},
      // " How's it going?!" and " Hello World!" followed by the stop token id
      // (0)
      /*decode_tokens=*/
      {{224, 90},
       {24, 547},
       {8, 58},
       {66, 735},
       {246, 210},
       {18, 466},
       {2295, 2294},
       {2294, 0},  // should stop decoding here
       {0, 0}},
      /*vocab_size=*/2560,
      /*batch_size=*/2);

  // Run prefill with <bos> token.
  std::vector<int> prefill_token_ids = {2};
  ASSERT_OK_AND_ASSIGN(auto token_ids_buffer,
                       tokenizer_->TokenIdsToTensorBuffer(prefill_token_ids));
  ExecutorTextData text_data(std::move(token_ids_buffer));
  ExecutorInputs inputs(std::move(text_data), std::nullopt, std::nullopt);
  auto prefill_responses = Tasks::Prefill(
      executor, inputs, /*wait_for_completion=*/true, benchmark_info);
  EXPECT_OK(prefill_responses);

  absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback =
      CreateTestCallback(responses, status, done);

  absl::StatusOr<Responses> task_responses =
      Tasks::Decode(executor, *tokenizer_, stop_token_detector,
                    /*num_output_candidates=*/2, benchmark_info, sampler.get(),
                    /*constraint=*/nullptr, std::move(decoded_ids.Value()),
                    /*callback=*/callback,
                    /*cancelled=*/nullptr);

  EXPECT_OK(task_responses);
  EXPECT_EQ(task_responses->GetTaskState(), TaskState::kDone);

  callback(task_responses);
  // First candidate: " How's it going" - ("?!") are stop tokens that is not
  // included in the output.
  EXPECT_EQ(responses[0], " How's it going");
  // Second candidate: " Hello World!"
  EXPECT_EQ(responses[1], " Hello World!");
}

TEST_F(TasksCustomSamplingTest,
       DecodeCustomSamplingStreamingReachMaxNumTokens) {
  auto executor = CreateFakeLlmExecutor(
      /*prefill_tokens=*/{{2}, {8, 58}},
      /*decode_tokens=*/{{224, 90},
                         {24, 547},
                         {8, 58},  // Stop here because of max num tokens.
                         {66, 735},
                         {246, 210},
                         {18, 466},
                         {2295, 2294},
                         {2294, 0},
                         {0, 0}});
  // Set the max number of tokens to 4.
  executor.GetMutableExecutorSettings().value()->SetMaxNumTokens(4);

  std::optional<BenchmarkInfo> benchmark_info;

  // Run prefill with <bos> token.
  std::vector<int> prefill_token_ids = {2};
  ASSERT_OK_AND_ASSIGN(auto token_ids_buffer,
                       tokenizer_->TokenIdsToTensorBuffer(prefill_token_ids));
  ExecutorTextData text_data(std::move(token_ids_buffer));
  ExecutorInputs inputs(std::move(text_data), std::nullopt, std::nullopt);
  auto prefill_responses = Tasks::Prefill(
      executor, inputs, /*wait_for_completion=*/true, benchmark_info);
  EXPECT_OK(prefill_responses);

  auto sampler_or =
      TopPSampler::Create(/*k=*/1, /*p=*/0.5, /*temperature=*/1.0,
                          /*batch_size=*/2, /*sequence_size=*/1, /*seed=*/1);
  EXPECT_TRUE(sampler_or.ok());
  std::unique_ptr<TopPSampler> sampler = std::move(sampler_or.value());

  auto decoded_ids = CreateTensorBuffer<int>({2, 1});
  EXPECT_TRUE(decoded_ids.HasValue());

  StopTokenDetector stop_token_detector(2);
  EXPECT_OK(stop_token_detector.AddStopTokenSequence({0}));

  absl::Status status;
  std::vector<std::string> responses(2);
  bool done = false;
  absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback =
      CreateTestCallback(responses, status, done);

  absl::StatusOr<Responses> task_responses =
      Tasks::Decode(executor, *tokenizer_, stop_token_detector,
                    /*num_output_candidates=*/2, benchmark_info, sampler.get(),
                    /*constraint=*/nullptr, std::move(decoded_ids.Value()),
                    /*callback=*/callback,
                    /*cancelled=*/nullptr);
  callback(task_responses);

  EXPECT_OK(task_responses);
  EXPECT_EQ(task_responses->GetTaskState(), TaskState::kMaxNumTokensReached);

  // First candidate truncated at max number of tokens: " How's".
  EXPECT_EQ(responses[0], " How's");
  // Second candidate truncated at max number of tokens: " Hello".
  EXPECT_EQ(responses[1], " Hello");
}

TEST_F(TasksCustomSamplingTest, DecodeComplexStopTokenDetector) {
  auto sampler_or =
      TopPSampler::Create(/*k=*/1, /*p=*/0.5, /*temperature=*/1.0,
                          /*batch_size=*/2, /*sequence_size=*/1, /*seed=*/1);
  EXPECT_TRUE(sampler_or.ok());
  std::unique_ptr<TopPSampler> sampler = std::move(sampler_or.value());

  auto decoded_ids = CreateTensorBuffer<int>({2, 1});
  EXPECT_TRUE(decoded_ids.HasValue());
  std::optional<BenchmarkInfo> benchmark_info;
  StopTokenDetector stop_token_detector(2);
  // This is only a partial stop token sequence matched for the first batch.
  EXPECT_OK(stop_token_detector.AddStopTokenSequence({24, 8, 9}));
  // This is a partial stop token sequence matched for the first batch,
  // overlapping with the previous stop token sequence.
  EXPECT_OK(stop_token_detector.AddStopTokenSequence({224, 24, 9}));
  // This is a full stop token sequence matched for the first batch
  EXPECT_OK(stop_token_detector.AddStopTokenSequence({0}));

  // This will be a full match for the second batch.
  EXPECT_OK(stop_token_detector.AddStopTokenSequence({90, 547, 58}));
  // This will be a partial match for the second batch, overlapping with the
  // previous stop token sequence.
  EXPECT_OK(stop_token_detector.AddStopTokenSequence({90, 548}));

  auto executor = CreateFakeLlmExecutor(
      /*prefill_tokens=*/{{2}, {0, 0}},
      /*decode_tokens=*/{{224, 90},
                         {24, 547},
                         {8, 58},
                         {66, 735},
                         {246, 210},
                         {18, 466},
                         {2295, 2294},
                         {2294, 0},
                         {0, 0}});

  // Run prefill with <bos> token.
  std::vector<int> prefill_token_ids = {2};
  ASSERT_OK_AND_ASSIGN(auto token_ids_buffer,
                       tokenizer_->TokenIdsToTensorBuffer(prefill_token_ids));
  ExecutorTextData text_data(std::move(token_ids_buffer));
  ExecutorInputs inputs(std::move(text_data), std::nullopt, std::nullopt);
  auto prefill_responses = Tasks::Prefill(
      executor, inputs, /*wait_for_completion=*/true, benchmark_info);
  EXPECT_OK(prefill_responses);

  absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback = nullptr;

  auto task_responses =
      Tasks::Decode(executor, *tokenizer_, stop_token_detector,
                    /*num_output_candidates=*/2, benchmark_info, sampler.get(),
                    /*constraint=*/nullptr, std::move(decoded_ids.Value()),
                    /*callback=*/callback,
                    /*cancelled=*/nullptr);

  EXPECT_OK(task_responses);
  // Expect two output candidates.
  EXPECT_EQ(task_responses->GetTexts().size(), 2);
  // First candidate: " How's it going?!".
  EXPECT_EQ(task_responses->GetTexts()[0], " How's it going?!");
  // Second candidate: "" since the stop token sequence is matched at
  // the beginning of the second batch.
  EXPECT_EQ(task_responses->GetTexts()[1], "");

  // The scores are equal to 0.0f (log(1.0f)).
  EXPECT_EQ(task_responses->GetScores().size(), 2);
  EXPECT_EQ(task_responses->GetScores()[0], 0.0f);
  // The second candidate doesn't have any tokens decoded so the score is set to
  // -inf.
  EXPECT_EQ(task_responses->GetScores()[1],
            -std::numeric_limits<float>::infinity());
}

TEST_F(TasksCustomSamplingTest, DecodeCustomSamplingStreamingWithCancellation) {
  std::vector<std::vector<int>> decode_tokens;
  for (int i = 0; i < 100; ++i) {
    decode_tokens.push_back({1, 1});
  }
  auto delayed_executor = CreateFakeLlmExecutor(
      /*prefill_tokens=*/{{2}, {224, 90}},
      /*decode_tokens=*/{{224, 90},
                         {24, 547},
                         {8, 58},
                         {66, 735},
                         {246, 210},
                         {18, 466},
                         {2295, 2294},
                         {2294, 0},
                         {0, 0}});

  std::optional<BenchmarkInfo> benchmark_info;

  // Run prefill with <bos> token.
  std::vector<int> prefill_token_ids = {2};
  ASSERT_OK_AND_ASSIGN(auto token_ids_buffer,
                       tokenizer_->TokenIdsToTensorBuffer(prefill_token_ids));
  ExecutorTextData text_data(std::move(token_ids_buffer));
  ExecutorInputs inputs(std::move(text_data), std::nullopt, std::nullopt);
  auto prefill_responses = Tasks::Prefill(
      delayed_executor, inputs, /*wait_for_completion=*/true, benchmark_info);
  EXPECT_OK(prefill_responses);

  // Set the delay long enough not to be flaky.
  delayed_executor.SetDecodeDelay(absl::Milliseconds(1000));

  auto sampler_or =
      TopPSampler::Create(/*k=*/1, /*p=*/0.5, /*temperature=*/1.0,
                          /*batch_size=*/2, /*sequence_size=*/1, /*seed=*/1);
  EXPECT_TRUE(sampler_or.ok());
  std::unique_ptr<TopPSampler> sampler = std::move(sampler_or.value());

  auto decoded_ids = CreateTensorBuffer<int>({2, 1});
  EXPECT_TRUE(decoded_ids.HasValue());

  StopTokenDetector stop_token_detector(2);
  EXPECT_OK(stop_token_detector.AddStopTokenSequence({0}));

  std::atomic<bool> cancelled = false;

  absl::StatusOr<Responses> task_responses;
  absl::Status callback_status;
  std::vector<std::string> responses(2);
  bool done = false;
  ThreadPool pool("test_pool", 1);
  absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback =
      CreateTestCallback(responses, callback_status, done,
                         /*delay_on_next=*/true);

  ASSERT_OK(pool.Schedule([&]() {
    task_responses = Tasks::Decode(
        delayed_executor, *tokenizer_, stop_token_detector,

        /*num_output_candidates=*/2, benchmark_info, sampler.get(),
        /*constraint=*/nullptr, std::move(decoded_ids.Value()),
        /*callback=*/callback, &cancelled);
    callback(task_responses);
  }));

  // Wait for a short time to ensure the decoding has started.
  absl::SleepFor(absl::Milliseconds(50));

  // Cancel the decoding process.
  cancelled = true;

  EXPECT_OK(pool.WaitUntilDone(absl::Seconds(5)));
  EXPECT_THAT(task_responses,
              testing::status::StatusIs(absl::StatusCode::kCancelled));
  EXPECT_THAT(callback_status,
              testing::status::StatusIs(absl::StatusCode::kCancelled));
}

TEST_F(TasksCustomSamplingTest,
       DecodeCustomSamplingStreamingWithConstrainedDecoding) {
  auto sampler_or =
      TopPSampler::Create(/*k=*/1, /*p=*/0.5, /*temperature=*/1.0,
                          /*batch_size=*/2, /*sequence_size=*/1, /*seed=*/1);
  EXPECT_TRUE(sampler_or.ok());
  std::unique_ptr<TopPSampler> sampler = std::move(sampler_or.value());

  auto decoded_ids = CreateTensorBuffer<int>({2, 1});
  EXPECT_TRUE(decoded_ids.HasValue());
  // Populate with the last pre-filled token.
  decoded_ids->Write<int>({2, 2});
  absl::Status callback_status;
  std::vector<std::string> responses(2);
  bool done = false;
  std::optional<BenchmarkInfo> benchmark_info;

  // Fake constraint that expects " Hello World".
  std::vector<int> expected_token_ids = {90, 547, 58, 735, 210, 466, 0};
  auto constraint = std::make_unique<FakeConstraint>(expected_token_ids,
                                                     /*vocabulary_size=*/2560);

  auto executor = CreateFakeLlmExecutor(
      /*prefill_tokens=*/{{2}, {0, 0}},
      // " Hello World!" for both batch because constraint is set.
      /*decode_tokens=*/{{90, 90},
                         {547, 547},
                         {58, 58},
                         {735, 735},
                         {210, 210},
                         {466, 466},
                         {2294, 2294},  // Stop here because constraint is set.
                         {0, 0},
                         {0, 0}});

  // Run prefill with <bos> token.
  std::vector<int> prefill_token_ids = {2};
  ASSERT_OK_AND_ASSIGN(auto token_ids_buffer,
                       tokenizer_->TokenIdsToTensorBuffer(prefill_token_ids));
  ExecutorTextData text_data(std::move(token_ids_buffer));
  ExecutorInputs inputs(std::move(text_data), std::nullopt, std::nullopt);
  auto prefill_responses = Tasks::Prefill(
      executor, inputs, /*wait_for_completion=*/true, benchmark_info);
  EXPECT_OK(prefill_responses);

  StopTokenDetector stop_token_detector(2);
  EXPECT_OK(stop_token_detector.AddStopTokenSequence({0}));
  absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback =
      CreateTestCallback(responses, callback_status, done);

  absl::StatusOr<Responses> task_responses = Tasks::Decode(
      executor, *tokenizer_, stop_token_detector,
      /*num_output_candidates=*/2, benchmark_info, sampler.get(),
      /*constraint=*/constraint.get(), std::move(decoded_ids.Value()),
      /*callback=*/callback,
      /*cancelled=*/nullptr);
  callback(task_responses);

  EXPECT_OK(task_responses);
  EXPECT_EQ(task_responses->GetTaskState(), TaskState::kDone);

  EXPECT_EQ(responses[0], " Hello World");
  EXPECT_EQ(responses[1], " Hello World");
}

TEST_F(TasksCustomSamplingTest, DecodeStopTokenAndBPEDetector) {
  auto sampler_or =
      TopPSampler::Create(/*k=*/1, /*p=*/0.5,
                          /*temperature=*/1.0,
                          /*batch_size=*/2, /*sequence_size=*/1, /*seed=*/1);
  EXPECT_TRUE(sampler_or.ok());
  std::unique_ptr<TopPSampler> sampler = std::move(sampler_or.value());

  auto tokenizer = std::make_unique<BytePairEncodingTokenizer>();
  // batch 1: 224, 24, 8, 66
  EXPECT_CALL(*tokenizer, TokenIdsToText(std::vector<int>{224}))
      .WillOnce(
          testing::Return(absl::DataLossError("Incomplete BPE sequence")));
  EXPECT_CALL(*tokenizer, TokenIdsToText(std::vector<int>{224, 24}))
      .WillOnce(
          testing::Return(absl::DataLossError("Incomplete BPE sequence")));
  EXPECT_CALL(*tokenizer, TokenIdsToText(std::vector<int>{224, 24, 8}))
      .WillOnce(testing::Return("BPE"));
  // Stop token: for first batch
  EXPECT_CALL(*tokenizer, TokenIdsToText(std::vector<int>{66}))
      .WillOnce(testing::Return("!"));

  // batch 2: 90, 547, 58, 735
  EXPECT_CALL(*tokenizer, TokenIdsToText(std::vector<int>{90}))
      .WillOnce(testing::Return("a"));
  EXPECT_CALL(*tokenizer, TokenIdsToText(std::vector<int>{547}))
      .WillOnce(testing::Return("b"));
  EXPECT_CALL(*tokenizer, TokenIdsToText(std::vector<int>{58}))
      .WillOnce(testing::Return("c"));
  // Already stopped, but increase the length of the matched stop sequence.
  EXPECT_CALL(*tokenizer, TokenIdsToText(std::vector<int>{735}))
      .WillOnce(testing::Return("d"));

  std::optional<BenchmarkInfo> benchmark_info;
  StopTokenDetector stop_token_detector(2);
  // Stop right after the BPE sequence.
  EXPECT_OK(stop_token_detector.AddStopTokenSequence({66}));
  // Partial stop token sequence, no 544 token - should output
  EXPECT_OK(stop_token_detector.AddStopTokenSequence({90, 544}));
  // This will stop the decoding.
  EXPECT_OK(stop_token_detector.AddStopTokenSequence({547, 58}));

  auto executor = CreateFakeLlmExecutor(
      // The expected prefill tokens that after stop tokens are found in
      // decoding with CustomSampling. That is, the last sampled tokens at
      // stop condition.
      /*prefill_tokens=*/{{2}, {66, 735}},
      /*decode_tokens=*/
      {{224, 90},
       {24, 547},
       {8, 58},
       {66, 735},  // Stop here because of BPE.
       {2294, 2294},
       {0, 0}});

  // Run prefill with <bos> token.
  std::vector<int> prefill_token_ids = {2};
  ASSERT_OK_AND_ASSIGN(auto token_ids_buffer,
                       tokenizer->TokenIdsToTensorBuffer(prefill_token_ids));
  ExecutorTextData text_data(std::move(token_ids_buffer));
  ExecutorInputs inputs(std::move(text_data), std::nullopt, std::nullopt);
  auto prefill_responses = Tasks::Prefill(
      executor, inputs, /*wait_for_completion=*/true, benchmark_info);
  EXPECT_OK(prefill_responses);

  auto decoded_ids = CreateTensorBuffer<int>({2, 1});
  EXPECT_TRUE(decoded_ids.HasValue());
  absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback = nullptr;

  absl::StatusOr<Responses> task_responses =
      Tasks::Decode(executor, *tokenizer, stop_token_detector,
                    /*num_output_candidates=*/2, benchmark_info, sampler.get(),
                    /*constraint=*/nullptr, std::move(decoded_ids.Value()),
                    /*callback=*/callback,
                    /*cancelled=*/nullptr);

  EXPECT_OK(task_responses);
  EXPECT_EQ(task_responses->GetTexts().size(), 2);
  EXPECT_EQ(task_responses->GetTexts()[0], "BPE");
  EXPECT_EQ(task_responses->GetTexts()[1], "a");
}

using TasksCallbackTest = TasksTest;

TEST_F(TasksCallbackTest, DecodeStreaming_SuccessfulCompletion) {
  std::optional<BenchmarkInfo> benchmark_info;

  // Run prefill first.
  std::vector<int> prefill_token_ids = {2, 90, 547, 58, 735, 210, 466, 2294};
  ASSERT_OK_AND_ASSIGN(auto token_ids_buffer,
                       tokenizer_->TokenIdsToTensorBuffer(prefill_token_ids));
  ExecutorTextData text_data(std::move(token_ids_buffer));
  ExecutorInputs inputs(std::move(text_data), std::nullopt, std::nullopt);
  auto prefill_responses = Tasks::Prefill(
      *executor_, inputs, /*wait_for_completion=*/true, benchmark_info);
  EXPECT_OK(prefill_responses);

  constexpr int kNumOutputCandidates = 1;
  StopTokenDetector stop_token_detector(kNumOutputCandidates);
  EXPECT_OK(stop_token_detector.AddStopTokenSequence({2294}));
  absl::Status status;
  std::vector<std::string> responses(kNumOutputCandidates);
  bool done = false;
  absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback =
      CreateTestCallback(responses, status, done);

  absl::StatusOr<Responses> task_responses = Tasks::Decode(
      *executor_, *tokenizer_, stop_token_detector, kNumOutputCandidates,
      benchmark_info, /*sampler=*/std::nullopt,
      /*constraint=*/nullptr, /*decoded_ids=*/std::nullopt,
      /*callback=*/callback,
      /*cancelled=*/nullptr);
  callback(task_responses);

  EXPECT_OK(task_responses);
  EXPECT_EQ(task_responses->GetTaskState(), TaskState::kDone);

  EXPECT_EQ(responses[0], " How's it going?");
  EXPECT_TRUE(done);
  EXPECT_OK(status);
}

TEST_F(TasksCallbackTest, DecodeStreaming_ErrorCompletion) {
  // Set the max number of tokens to 11 to trigger an error.
  executor_->GetMutableExecutorSettings().value()->SetMaxNumTokens(11);
  std::optional<BenchmarkInfo> benchmark_info;

  // Run prefill first.
  std::vector<int> prefill_token_ids = {2, 90, 547, 58, 735, 210, 466, 2294};
  ASSERT_OK_AND_ASSIGN(auto token_ids_buffer,
                       tokenizer_->TokenIdsToTensorBuffer(prefill_token_ids));
  ExecutorTextData text_data(std::move(token_ids_buffer));
  ExecutorInputs inputs(std::move(text_data), std::nullopt, std::nullopt);
  auto prefill_responses = Tasks::Prefill(
      *executor_, inputs, /*wait_for_completion=*/true, benchmark_info);
  EXPECT_OK(prefill_responses);

  constexpr int kNumOutputCandidates = 1;
  StopTokenDetector stop_token_detector(kNumOutputCandidates);
  EXPECT_OK(stop_token_detector.AddStopTokenSequence({2294}));
  absl::Status status;
  std::vector<std::string> responses(kNumOutputCandidates);
  bool done = false;
  absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback =
      CreateTestCallback(responses, status, done);

  absl::StatusOr<Responses> task_responses = Tasks::Decode(
      *executor_, *tokenizer_, stop_token_detector, kNumOutputCandidates,
      benchmark_info, /*sampler=*/std::nullopt,
      /*constraint=*/nullptr, /*decoded_ids=*/std::nullopt,
      /*callback=*/callback,
      /*cancelled=*/nullptr);
  callback(task_responses);

  EXPECT_OK(task_responses);
  EXPECT_EQ(task_responses->GetTaskState(), TaskState::kMaxNumTokensReached);

  EXPECT_EQ(responses[0], " How's");
  EXPECT_TRUE(done);
}

TEST_F(TasksCallbackTest,
       DecodeStreaming_SuccessfulCompletion_WithMultipleCandidates) {
  constexpr int kNumOutputCandidates = 3;
  // Rebuild the executor with multiple output candidates with the same prefill
  // and decode tokens.
  std::vector<std::vector<int>> prefill_tokens = {
      {2, 90, 547, 58, 735, 210, 466, 2294}};
  // "How's it going?", "Hello World", "How's it going?"
  std::vector<std::vector<int>> decode_tokens = {
      {224, 90, 224},  {24, 547, 24}, {8, 58, 8},         {66, 735, 66},
      {246, 210, 246}, {18, 466, 18}, {2295, 2294, 2295}, {2294, 0, 2294}};
  executor_ = std::make_unique<FakeLlmExecutor>(
      /*vocab_size=*/2560, prefill_tokens, decode_tokens, kNumOutputCandidates);

  std::optional<BenchmarkInfo> benchmark_info;

  // Run prefill first.
  std::vector<int> prefill_token_ids = {2, 90, 547, 58, 735, 210, 466, 2294};
  ASSERT_OK_AND_ASSIGN(auto token_ids_buffer,
                       tokenizer_->TokenIdsToTensorBuffer(prefill_token_ids));
  ExecutorTextData text_data(std::move(token_ids_buffer));
  ExecutorInputs inputs(std::move(text_data), std::nullopt, std::nullopt);
  auto prefill_responses = Tasks::Prefill(
      *executor_, inputs, /*wait_for_completion=*/true, benchmark_info);
  EXPECT_OK(prefill_responses);

  StopTokenDetector stop_token_detector(kNumOutputCandidates);
  EXPECT_OK(stop_token_detector.AddStopTokenSequence({2294}));
  absl::Status status;
  std::vector<std::string> responses(kNumOutputCandidates);
  bool done = false;
  auto callback = CreateTestCallback(responses, status, done);

  absl::StatusOr<Responses> task_responses = Tasks::Decode(
      *executor_, *tokenizer_, stop_token_detector, kNumOutputCandidates,
      benchmark_info, /*sampler=*/std::nullopt,
      /*constraint=*/nullptr, /*decoded_ids=*/std::nullopt,
      /*callback=*/callback,
      /*cancelled=*/nullptr);
  callback(task_responses);

  EXPECT_OK(task_responses);
  EXPECT_EQ(task_responses->GetTaskState(), TaskState::kDone);

  EXPECT_EQ(responses[0], " How's it going?");
  EXPECT_EQ(responses[1], " Hello World");
  EXPECT_EQ(responses[2], " How's it going?");
  EXPECT_TRUE(done);
  EXPECT_OK(status);
}

}  // namespace
}  // namespace litert::lm
