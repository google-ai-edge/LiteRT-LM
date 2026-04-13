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

#include "runtime/framework/resource_management/execution_manager.h"

#include <atomic>
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
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/constrained_decoding/fake_constraint.h"
#include "runtime/components/model_resources.h"
#include "runtime/components/tokenizer.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/audio_executor.h"
#include "runtime/executor/audio_executor_settings.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/fake_llm_executor.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/framework/resource_management/serial_execution_manager.h"
#include "runtime/framework/resource_management/threaded_execution_manager.h"
#include "runtime/proto/token.pb.h"
#include "runtime/util/status_macros.h"  // IWYU pragma: keep
#include "runtime/util/test_utils.h"  // NOLINT

namespace litert::lm {
namespace {

using ::testing::ElementsAre;
using ::testing::Return;

class MockTokenizer : public Tokenizer {
 public:
  MOCK_METHOD(absl::StatusOr<std::vector<int>>, TextToTokenIds,
              (absl::string_view text), (override));
  MOCK_METHOD(absl::StatusOr<int>, TokenToId, (absl::string_view token),
              (override));
  MOCK_METHOD(absl::StatusOr<std::string>, TokenIdsToText,
              (const std::vector<int>& token_ids), (override));
  MOCK_METHOD(TokenizerType, GetTokenizerType, (), (const, override));
  MOCK_METHOD(std::vector<std::string>, GetTokens, (), (const, override));
};

class FakeAudioExecutor : public AudioExecutor {
 public:
  absl::StatusOr<::litert::lm::ExecutorAudioData> Encode(
      const litert::TensorBuffer& spectrogram_tensor) override {
    return ::litert::lm::ExecutorAudioData();
  }
};

enum class ExecutionManagerType {
  kThreaded,
  kSerial,
};

class ExecutionManagerTest
    : public ::testing::TestWithParam<ExecutionManagerType> {
 protected:
  void SetUp() override {
    tokenizer_ = std::make_unique<MockTokenizer>();
    EXPECT_CALL(*tokenizer_, TokenIdsToText(ElementsAre(0)))
        .WillRepeatedly(Return("0"));
    EXPECT_CALL(*tokenizer_, TokenIdsToText(ElementsAre(4)))
        .WillRepeatedly(Return("4"));
    EXPECT_CALL(*tokenizer_, TokenIdsToText(ElementsAre(5)))
        .WillRepeatedly(Return("5"));
    EXPECT_CALL(*tokenizer_, TokenIdsToText(ElementsAre(6)))
        .WillRepeatedly(Return("6"));
  }

  absl::StatusOr<SessionConfig> CreateDefaultSessionConfig(
      bool use_external_sampler = false) {
    ASSIGN_OR_RETURN(auto model_assets,
                     ModelAssets::Create("test_model_path_1"));
    ASSIGN_OR_RETURN(auto settings,
                     EngineSettings::CreateDefault(model_assets));

    proto::LlmMetadata llm_metadata;
    llm_metadata.mutable_stop_tokens()
        ->Add()
        ->mutable_token_ids()
        ->mutable_ids()
        ->Add(0);
    llm_metadata.mutable_stop_tokens()
        ->Add()
        ->mutable_token_ids()
        ->mutable_ids()
        ->Add(6);
    llm_metadata.mutable_llm_model_type()->mutable_gemma3n();
    EXPECT_OK(settings.MaybeUpdateAndValidate(tokenizer_.get(), &llm_metadata));
    SessionConfig session_config = SessionConfig::CreateDefault();
    EXPECT_OK(session_config.MaybeUpdateAndValidate(settings));
    session_config.SetUseExternalSampler(use_external_sampler);
    model_resources_ = std::unique_ptr<ModelResources>();
    return session_config;
  };

  void CreateExecutionManager(
      std::unique_ptr<FakeLlmExecutor> fake_llm_executor,
      std::unique_ptr<AudioExecutorSettings> audio_executor_settings = nullptr,
      std::unique_ptr<AudioExecutor> audio_executor = nullptr) {
    // The objects are moved to execution_manager_ so we can't access them
    // after creation.
    switch (GetParam()) {
      case ExecutionManagerType::kThreaded: {
        ASSERT_OK_AND_ASSIGN(
            execution_manager_,
            ThreadedExecutionManager::Create(
                /*tokenizer=*/tokenizer_.get(),
                /*model_resources=*/model_resources_.get(),
                /*llm_executor=*/std::move(fake_llm_executor),
                /*vision_executor_settings=*/nullptr,
                std::move(audio_executor_settings),
                /*litert_env=*/nullptr, std::move(audio_executor)));
        break;
      }
      case ExecutionManagerType::kSerial: {
        ASSERT_OK_AND_ASSIGN(
            execution_manager_,
            SerialExecutionManager::Create(
                /*tokenizer=*/tokenizer_.get(),
                /*model_resources=*/model_resources_.get(),
                /*llm_executor=*/std::move(fake_llm_executor),
                /*vision_executor_settings=*/nullptr,
                std::move(audio_executor_settings),
                /*litert_env=*/nullptr, std::move(audio_executor)));
        break;
      }
    }
  }

  std::unique_ptr<FakeLlmExecutor> CreateDefaultFakeLlmExecutor(
      std::optional<std::vector<std::vector<int>>> override_prefill_tokens =
          std::nullopt) {
    auto prefill_tokens = std::vector<std::vector<int>>{{1, 2, 3}};
    if (override_prefill_tokens.has_value()) {
      prefill_tokens = *override_prefill_tokens;
    }
    auto decode_tokens = std::vector<std::vector<int>>{{4}, {5}, {6}};
    return std::make_unique<FakeLlmExecutor>(
        /*vocab_size=*/10,
        /*prefill_tokens=*/std::move(prefill_tokens),
        /*decode_tokens=*/std::move(decode_tokens));
  }

  std::unique_ptr<MockTokenizer> tokenizer_;

  std::unique_ptr<ModelResources> model_resources_;

  std::unique_ptr<ExecutionManager> execution_manager_;
};

TEST_P(ExecutionManagerTest, CanGetMutableBenchmarkInfo) {
  CreateExecutionManager(CreateDefaultFakeLlmExecutor());
  ASSERT_OK_AND_ASSIGN(auto session_config, CreateDefaultSessionConfig());
  ASSERT_OK_AND_ASSIGN(const SessionId session_id,
                       execution_manager_->RegisterNewSession(
                           session_config, std::make_optional<BenchmarkInfo>(
                                               proto::BenchmarkParams())));
  ASSERT_OK_AND_ASSIGN(BenchmarkInfo * benchmark_info,
                       execution_manager_->GetMutableBenchmarkInfo(session_id));
  EXPECT_NE(benchmark_info, nullptr);
}

TEST_P(ExecutionManagerTest, GetMutableBenchmarkInfoFailsIfNoBenchmarkInfo) {
  CreateExecutionManager(CreateDefaultFakeLlmExecutor());
  ASSERT_OK_AND_ASSIGN(auto session_config, CreateDefaultSessionConfig());
  ASSERT_OK_AND_ASSIGN(const SessionId session_id,
                       execution_manager_->RegisterNewSession(session_config));
  EXPECT_THAT(execution_manager_->GetMutableBenchmarkInfo(session_id),
              testing::status::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_P(ExecutionManagerTest, AddPrefillTask) {
  CreateExecutionManager(CreateDefaultFakeLlmExecutor({{{1, 2, 3, -4}}}));
  ASSERT_OK_AND_ASSIGN(auto session_config, CreateDefaultSessionConfig());
  ASSERT_OK_AND_ASSIGN(const SessionId session_id,
                       execution_manager_->RegisterNewSession(session_config));

  std::vector<TaskState> task_states;
  absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback =
      [&task_states](absl::StatusOr<Responses> responses) {
        ASSERT_OK(responses);
        task_states.push_back(responses->GetTaskState());
      };

  std::vector<InputData> inputs;
  ASSERT_OK_AND_ASSIGN(auto input_text,
                       tokenizer_->TokenIdsToTensorBuffer({1, 2, 3}));
  inputs.push_back(InputText(std::move(input_text)));
  inputs.push_back(InputAudioEnd());

  ASSERT_OK_AND_ASSIGN(const TaskId task_id,
                       execution_manager_->GetNewTaskId());

  ASSERT_OK(execution_manager_->AddPrefillTask(
      session_id, task_id, std::move(inputs), {},
      std::make_shared<std::atomic<bool>>(false), std::move(callback)));

  EXPECT_OK(execution_manager_->WaitUntilDone(task_id, absl::Seconds(3)));

  EXPECT_THAT(task_states,
              ElementsAre(TaskState::kCreated, TaskState::kQueued,
                          TaskState::kProcessing, TaskState::kDone));
}

TEST_P(ExecutionManagerTest, AddPrefillTaskWithAudioModality) {
  auto fake_llm_executor = CreateDefaultFakeLlmExecutor();

  ASSERT_OK_AND_ASSIGN(auto* settings,
                       fake_llm_executor->GetMutableExecutorSettings());
  EXPECT_OK(settings->SetBackend(Backend::GPU_ARTISAN));

  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create("test_model_path_2"));
  ASSERT_OK_AND_ASSIGN(auto audio_settings,
                       AudioExecutorSettings::CreateDefault(
                           model_assets, 128, Backend::GPU_ARTISAN));

  CreateExecutionManager(
      std::move(fake_llm_executor),
      std::make_unique<AudioExecutorSettings>(std::move(audio_settings)),
      std::make_unique<FakeAudioExecutor>());

  ASSERT_OK_AND_ASSIGN(auto session_config, CreateDefaultSessionConfig());
  session_config.SetAudioModalityEnabled(true);

  // Trigger RegisterNewSession which previously acquired nested locks
  ASSERT_OK_AND_ASSIGN(const SessionId session_id,
                       execution_manager_->RegisterNewSession(session_config));

  std::vector<InputData> inputs;
  ASSERT_OK_AND_ASSIGN(auto input_text,
                       tokenizer_->TokenIdsToTensorBuffer({1, 2, 3}));
  inputs.push_back(InputText(std::move(input_text)));
  inputs.push_back(InputAudioEnd());

  ASSERT_OK_AND_ASSIGN(const TaskId task_id,
                       execution_manager_->GetNewTaskId());
  ASSERT_OK(execution_manager_->AddPrefillTask(
      session_id, task_id, std::move(inputs), {},
      std::make_shared<std::atomic<bool>>(false), nullptr));

  EXPECT_OK(execution_manager_->WaitUntilDone(task_id, absl::Seconds(3)));
}

TEST_P(ExecutionManagerTest, AddPrefillTaskInvalidAudioInput) {
  CreateExecutionManager(CreateDefaultFakeLlmExecutor());
  ASSERT_OK_AND_ASSIGN(auto session_config, CreateDefaultSessionConfig());
  ASSERT_OK_AND_ASSIGN(const SessionId session_id,
                       execution_manager_->RegisterNewSession(session_config));

  std::vector<TaskState> task_states;
  absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback =
      [&task_states](absl::StatusOr<Responses> responses) {
        if (!responses.ok()) {
          ASSERT_THAT(responses, testing::status::StatusIs(
                                     absl::StatusCode::kFailedPrecondition));
          ASSERT_THAT(responses.status().message(),
                      testing::Eq("The audio is not a preprocessed tensor."));
          task_states.push_back(TaskState::kFailed);
        } else {
          ASSERT_OK(responses);
          task_states.push_back(responses->GetTaskState());
        }
      };

  std::vector<InputData> inputs;
  ASSERT_OK_AND_ASSIGN(auto input_text,
                       tokenizer_->TokenIdsToTensorBuffer({1, 2, 3}));
  inputs.push_back(InputText(std::move(input_text)));
  InputAudio input_audio("");
  inputs.push_back(std::move(input_audio));

  ASSERT_OK_AND_ASSIGN(const TaskId task_id,
                       execution_manager_->GetNewTaskId());

  ASSERT_OK(execution_manager_->AddPrefillTask(
      session_id, task_id, std::move(inputs), {},
      std::make_shared<std::atomic<bool>>(false), std::move(callback)));

  EXPECT_OK(execution_manager_->WaitUntilDone(task_id, absl::Seconds(3)));

  EXPECT_THAT(task_states,
              ElementsAre(TaskState::kCreated, TaskState::kQueued,
                          TaskState::kProcessing, TaskState::kFailed));
}

TEST_P(ExecutionManagerTest, AddPrefillTaskInvalidImageInput) {
  CreateExecutionManager(CreateDefaultFakeLlmExecutor());
  ASSERT_OK_AND_ASSIGN(auto session_config, CreateDefaultSessionConfig());
  ASSERT_OK_AND_ASSIGN(const SessionId session_id,
                       execution_manager_->RegisterNewSession(session_config));

  std::vector<TaskState> task_states;
  absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback =
      [&task_states](absl::StatusOr<Responses> responses) {
        if (!responses.ok()) {
          ASSERT_THAT(responses, testing::status::StatusIs(
                                     absl::StatusCode::kFailedPrecondition));
          ASSERT_THAT(responses.status().message(),
                      testing::Eq("Image tensor or tensor map is null in "
                                  "preprocessed_contents."));
          task_states.push_back(TaskState::kFailed);
        } else {
          ASSERT_OK(responses);
          task_states.push_back(responses->GetTaskState());
        }
      };

  std::vector<InputData> inputs;
  ASSERT_OK_AND_ASSIGN(auto input_text,
                       tokenizer_->TokenIdsToTensorBuffer({1, 2, 3}));
  inputs.push_back(InputText(std::move(input_text)));
  InputImage input_image("");
  inputs.push_back(std::move(input_image));

  ASSERT_OK_AND_ASSIGN(const TaskId task_id,
                       execution_manager_->GetNewTaskId());

  ASSERT_OK(execution_manager_->AddPrefillTask(
      session_id, task_id, std::move(inputs), {},
      std::make_shared<std::atomic<bool>>(false), std::move(callback)));

  EXPECT_OK(execution_manager_->WaitUntilDone(task_id, absl::Seconds(3)));

  EXPECT_THAT(task_states,
              ElementsAre(TaskState::kCreated, TaskState::kQueued,
                          TaskState::kProcessing, TaskState::kFailed));
}

TEST_P(ExecutionManagerTest, AddDecodeTaskWithInternalSampler) {
  // The default execution manager is using the internal sampler.
  CreateExecutionManager(CreateDefaultFakeLlmExecutor());

  ASSERT_OK_AND_ASSIGN(auto session_config, CreateDefaultSessionConfig());
  ASSERT_OK_AND_ASSIGN(const SessionId session_id,
                       execution_manager_->RegisterNewSession(session_config));

  std::vector<TaskState> task_states;
  std::vector<std::string> responses_texts;
  absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback =
      [&task_states, &responses_texts](absl::StatusOr<Responses> responses) {
        ASSERT_OK(responses);
        task_states.push_back(responses->GetTaskState());
        if (!responses->GetTexts().empty()) {
          responses_texts.push_back(responses->GetTexts()[0]);
        }
      };

  std::vector<InputData> inputs;
  ASSERT_OK_AND_ASSIGN(auto input_text,
                       tokenizer_->TokenIdsToTensorBuffer({1, 2, 3}));
  inputs.push_back(InputText(std::move(input_text)));
  std::optional<BenchmarkInfo> benchmark_info = std::nullopt;
  ASSERT_OK_AND_ASSIGN(const TaskId prefill_task_id,
                       execution_manager_->GetNewTaskId());
  ASSERT_OK(execution_manager_->AddPrefillTask(
      session_id, prefill_task_id, std::move(inputs),
      /*dependency_task_ids=*/{},
      /*cancelled=*/std::make_shared<std::atomic<bool>>(false),
      /*callback=*/[](absl::StatusOr<Responses> responses) {}));
  ASSERT_OK(
      execution_manager_->WaitUntilDone(prefill_task_id, absl::Seconds(3)));

  ASSERT_OK_AND_ASSIGN(const TaskId decode_task_id,
                       execution_manager_->GetNewTaskId());
  ASSERT_OK(execution_manager_->AddDecodeTask(
      session_id, decode_task_id,
      /*dependency_task_ids=*/{},
      /*constraint=*/nullptr,
      /*cancelled=*/std::make_shared<std::atomic<bool>>(false),
      std::move(callback)));

  EXPECT_OK(
      execution_manager_->WaitUntilDone(decode_task_id, absl::Seconds(3)));

  EXPECT_THAT(task_states,
              ElementsAre(TaskState::kCreated, TaskState::kQueued,
                          TaskState::kProcessing, TaskState::kProcessing,
                          TaskState::kProcessing, TaskState::kDone));

  EXPECT_THAT(responses_texts, ElementsAre("4", "5"));
}

TEST_P(ExecutionManagerTest, AddDecodeTaskWithExternalSampler) {
  std::vector<std::vector<int>> prefill_tokens = {{1, 2, 3}, {6}};
  std::vector<std::vector<int>> decode_tokens = {{4}, {5}, {6}};

  CreateExecutionManager(std::make_unique<FakeLlmExecutor>(
      /*vocab_size=*/10,
      /*prefill_tokens=*/std::move(prefill_tokens),
      /*decode_tokens=*/std::move(decode_tokens)));

  ASSERT_OK_AND_ASSIGN(auto session_config, CreateDefaultSessionConfig(
                                                /*use_external_sampler=*/true));
  ASSERT_OK_AND_ASSIGN(const SessionId session_id,
                       execution_manager_->RegisterNewSession(session_config));

  std::vector<TaskState> task_states;
  std::vector<std::string> responses_texts;
  absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback =
      [&task_states, &responses_texts](absl::StatusOr<Responses> responses) {
        ASSERT_OK(responses);
        task_states.push_back(responses->GetTaskState());
        if (!responses->GetTexts().empty()) {
          responses_texts.push_back(responses->GetTexts()[0]);
        }
      };

  std::vector<InputData> inputs;
  ASSERT_OK_AND_ASSIGN(auto input_text,
                       tokenizer_->TokenIdsToTensorBuffer({1, 2, 3}));
  inputs.push_back(InputText(std::move(input_text)));
  std::optional<BenchmarkInfo> benchmark_info = std::nullopt;
  ASSERT_OK_AND_ASSIGN(const TaskId prefill_task_id,
                       execution_manager_->GetNewTaskId());
  ASSERT_OK(execution_manager_->AddPrefillTask(
      session_id, prefill_task_id, std::move(inputs),
      /*dependency_task_ids=*/{},
      /*cancelled=*/std::make_shared<std::atomic<bool>>(false),
      /*callback=*/[](absl::StatusOr<Responses> responses) {}));
  ASSERT_OK(
      execution_manager_->WaitUntilDone(prefill_task_id, absl::Seconds(3)));

  ASSERT_OK_AND_ASSIGN(const TaskId decode_task_id,
                       execution_manager_->GetNewTaskId());
  ASSERT_OK(execution_manager_->AddDecodeTask(
      session_id, decode_task_id,
      /*dependency_task_ids=*/{},
      /*constraint=*/nullptr,
      /*cancelled=*/std::make_shared<std::atomic<bool>>(false),
      std::move(callback)));

  EXPECT_OK(
      execution_manager_->WaitUntilDone(decode_task_id, absl::Seconds(3)));

  EXPECT_THAT(task_states,
              ElementsAre(TaskState::kCreated, TaskState::kQueued,
                          TaskState::kProcessing, TaskState::kProcessing,
                          TaskState::kProcessing, TaskState::kDone));

  EXPECT_THAT(responses_texts, ElementsAre("4", "5"));
}

TEST_P(ExecutionManagerTest, CreateAndRunDependentTasks) {
  CreateExecutionManager(CreateDefaultFakeLlmExecutor());

  ASSERT_OK_AND_ASSIGN(auto session_config, CreateDefaultSessionConfig());
  ASSERT_OK_AND_ASSIGN(const SessionId session_id,
                       execution_manager_->RegisterNewSession(session_config));

  std::vector<InputData> inputs;
  ASSERT_OK_AND_ASSIGN(auto input_text,
                       tokenizer_->TokenIdsToTensorBuffer({1, 2, 3}));
  inputs.push_back(InputText(std::move(input_text)));
  std::optional<BenchmarkInfo> benchmark_info = std::nullopt;
  ASSERT_OK_AND_ASSIGN(const TaskId task_a_id,
                       execution_manager_->GetNewTaskId());
  ASSERT_OK(execution_manager_->AddPrefillTask(
      session_id, task_a_id, std::move(inputs),
      /*dependency_task_ids=*/{},
      /*cancelled=*/std::make_shared<std::atomic<bool>>(false),
      /*callback=*/nullptr));

  ASSERT_OK_AND_ASSIGN(const TaskId task_b_id,
                       execution_manager_->GetNewTaskId());
  ASSERT_OK(execution_manager_->AddDecodeTask(
      session_id, task_b_id,
      /*dependency_task_ids=*/{task_a_id},
      /*constraint=*/nullptr,
      /*cancelled=*/std::make_shared<std::atomic<bool>>(false),
      /*callback=*/nullptr, std::numeric_limits<int>::max()));

  EXPECT_OK(execution_manager_->WaitUntilDone(task_b_id, absl::Seconds(1)));
  EXPECT_OK(execution_manager_->WaitUntilDone(task_a_id, absl::Seconds(1)));
}

TEST_P(ExecutionManagerTest, CreateTaskWithInvalidDependency) {
  CreateExecutionManager(CreateDefaultFakeLlmExecutor());

  ASSERT_OK_AND_ASSIGN(auto session_config, CreateDefaultSessionConfig());
  ASSERT_OK_AND_ASSIGN(const SessionId session_id,
                       execution_manager_->RegisterNewSession(session_config));

  std::vector<InputData> inputs;
  inputs.push_back(InputText("test"));
  std::optional<BenchmarkInfo> benchmark_info = std::nullopt;
  ASSERT_OK_AND_ASSIGN(const TaskId task_id,
                       execution_manager_->GetNewTaskId());
  auto add_task_status = execution_manager_->AddPrefillTask(
      session_id, task_id, std::move(inputs),
      /*dependency_task_ids=*/{12345},
      /*cancelled=*/std::make_shared<std::atomic<bool>>(false),
      /*callback=*/nullptr);
  EXPECT_FALSE(add_task_status.ok());
  EXPECT_EQ(add_task_status.code(), absl::StatusCode::kInvalidArgument);
}

TEST_P(ExecutionManagerTest, CreateTaskWithInvalidDependencyId) {
  CreateExecutionManager(CreateDefaultFakeLlmExecutor());

  ASSERT_OK_AND_ASSIGN(auto session_config, CreateDefaultSessionConfig());
  ASSERT_OK_AND_ASSIGN(const SessionId session_id,
                       execution_manager_->RegisterNewSession(session_config));

  // Add a valid task.
  std::vector<InputData> inputs;
  ASSERT_OK_AND_ASSIGN(auto input_text,
                       tokenizer_->TokenIdsToTensorBuffer({1, 2, 3}));
  inputs.push_back(InputText(std::move(input_text)));
  std::optional<BenchmarkInfo> benchmark_info = std::nullopt;
  ASSERT_OK_AND_ASSIGN(const TaskId task_a_id,
                       execution_manager_->GetNewTaskId());
  ASSERT_OK(execution_manager_->AddPrefillTask(
      session_id, task_a_id, std::move(inputs),
      /*dependency_task_ids=*/{},
      /*cancelled=*/std::make_shared<std::atomic<bool>>(false),
      /*callback=*/nullptr));
  EXPECT_OK(execution_manager_->WaitUntilDone(task_a_id, absl::Seconds(1)));

  // Try to add a task with an invalid dependency.
  std::vector<InputData> inputs_b;
  ASSERT_OK_AND_ASSIGN(auto input_text_b,
                       tokenizer_->TokenIdsToTensorBuffer({4, 5, 6}));
  inputs_b.push_back(InputText(std::move(input_text_b)));
  const TaskId invalid_task_id = 99999;
  auto task_status = execution_manager_->AddPrefillTask(
      session_id, invalid_task_id, std::move(inputs_b),
      /*dependency_task_ids=*/{invalid_task_id},
      /*cancelled=*/std::make_shared<std::atomic<bool>>(false),
      /*callback=*/nullptr);
  EXPECT_FALSE(task_status.ok());
  EXPECT_EQ(task_status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(task_status.message(),
              testing::HasSubstr("Dependency task 99999 not found"));
}

TEST_P(ExecutionManagerTest, WaitUntilTaskDoneTimeout) {
  if (GetParam() == ExecutionManagerType::kSerial) {
    GTEST_SKIP() << "SerialExecutionManager executes synchronously and cannot "
                    "timeout while running.";
  }
  auto prefill_tokens = std::vector<std::vector<int>>{};
  auto decode_tokens = std::vector<std::vector<int>>{};
  decode_tokens.push_back({4});
  decode_tokens.push_back({5});
  decode_tokens.push_back({6});
  auto fake_llm_executor = std::make_unique<FakeLlmExecutor>(
      /*vocab_size=*/10,
      /*prefill_tokens=*/std::move(prefill_tokens),
      /*decode_tokens=*/std::move(decode_tokens));

  // Inject a long delay to simulate a timeout.
  fake_llm_executor->SetDecodeDelay(absl::Seconds(0.5));

  CreateExecutionManager(std::move(fake_llm_executor));

  ASSERT_OK_AND_ASSIGN(auto session_config, CreateDefaultSessionConfig());
  ASSERT_OK_AND_ASSIGN(const SessionId session_id,
                       execution_manager_->RegisterNewSession(session_config));

  std::optional<BenchmarkInfo> benchmark_info = std::nullopt;
  ASSERT_OK_AND_ASSIGN(const TaskId task_id,
                       execution_manager_->GetNewTaskId());
  ASSERT_OK(execution_manager_->AddDecodeTask(
      session_id, task_id,
      /*dependency_task_ids=*/{},

      /*constraint=*/nullptr,
      /*cancelled=*/std::make_shared<std::atomic<bool>>(false),
      /*callback=*/nullptr));

  EXPECT_EQ(
      execution_manager_->WaitUntilDone(task_id, absl::Milliseconds(100)),
      absl::DeadlineExceededError(absl::StrCat(
          "Task ", task_id, " did not complete within the timeout of 100ms.")));

  // Wait for the task to actually finish to avoid use after free.
  EXPECT_OK(execution_manager_->WaitUntilDone(task_id, absl::Seconds(3)));
}

TEST_P(ExecutionManagerTest, WaitUntilAllDoneTimeout) {
  if (GetParam() == ExecutionManagerType::kSerial) {
    GTEST_SKIP() << "SerialExecutionManager executes synchronously and cannot "
                    "timeout while running.";
  }
  auto prefill_tokens = std::vector<std::vector<int>>{};
  auto decode_tokens = std::vector<std::vector<int>>{};
  decode_tokens.push_back({4});
  decode_tokens.push_back({5});
  decode_tokens.push_back({6});
  auto fake_llm_executor = std::make_unique<FakeLlmExecutor>(
      /*vocab_size=*/10,
      /*prefill_tokens=*/std::move(prefill_tokens),
      /*decode_tokens=*/std::move(decode_tokens));

  // Inject a long delay to simulate a timeout.
  fake_llm_executor->SetDecodeDelay(absl::Seconds(0.5));

  CreateExecutionManager(std::move(fake_llm_executor));

  ASSERT_OK_AND_ASSIGN(auto session_config, CreateDefaultSessionConfig());
  ASSERT_OK_AND_ASSIGN(const SessionId session_id,
                       execution_manager_->RegisterNewSession(session_config));

  std::optional<BenchmarkInfo> benchmark_info = std::nullopt;
  ASSERT_OK_AND_ASSIGN(const TaskId task_id,
                       execution_manager_->GetNewTaskId());
  ASSERT_OK(execution_manager_->AddDecodeTask(
      session_id, task_id,
      /*dependency_task_ids=*/{},
      /*constraint=*/nullptr,
      /*cancelled=*/std::make_shared<std::atomic<bool>>(false),
      /*callback=*/nullptr));

  EXPECT_EQ(
      execution_manager_->WaitUntilAllDone(absl::Milliseconds(100)).code(),
      absl::StatusCode::kDeadlineExceeded);

  // Wait for the task to actually finish to avoid use after free.
  EXPECT_OK(execution_manager_->WaitUntilDone(task_id, absl::Seconds(3)));
}

TEST_P(ExecutionManagerTest, TaskReturnsError) {
  auto prefill_tokens = std::vector<std::vector<int>>{};
  auto decode_tokens = std::vector<std::vector<int>>{};
  prefill_tokens.push_back({1, 2, 3});
  auto fake_llm_executor = std::make_unique<FakeLlmExecutor>(
      /*vocab_size=*/10,
      /*prefill_tokens=*/std::move(prefill_tokens),
      /*decode_tokens=*/std::move(decode_tokens));

  // Inject an error.
  fake_llm_executor->SetPrefillStatus(absl::InternalError("Executor failed"));

  CreateExecutionManager(std::move(fake_llm_executor));

  ASSERT_OK_AND_ASSIGN(auto session_config, CreateDefaultSessionConfig());
  ASSERT_OK_AND_ASSIGN(const SessionId session_id,
                       execution_manager_->RegisterNewSession(session_config));

  std::vector<InputData> inputs;
  ASSERT_OK_AND_ASSIGN(auto input_text,
                       tokenizer_->TokenIdsToTensorBuffer({1, 2, 3}));
  inputs.push_back(InputText(std::move(input_text)));
  absl::Status final_status = absl::OkStatus();
  ASSERT_OK_AND_ASSIGN(const TaskId task_id,
                       execution_manager_->GetNewTaskId());
  ASSERT_OK(execution_manager_->AddPrefillTask(
      session_id, task_id, std::move(inputs), {},
      /*cancelled=*/std::make_shared<std::atomic<bool>>(false),
      [&](absl::StatusOr<Responses> responses) {
        if (!responses.ok()) {
          final_status = responses.status();
        }
      }));

  EXPECT_OK(execution_manager_->WaitUntilDone(task_id, absl::Seconds(1)));
  EXPECT_EQ(final_status, absl::InternalError("Executor failed"));
}

TEST_P(ExecutionManagerTest, CreateDependentTaskOnFailedTask) {
  auto prefill_tokens = std::vector<std::vector<int>>{};
  auto decode_tokens = std::vector<std::vector<int>>{};
  prefill_tokens.push_back({1, 2, 3});
  decode_tokens.push_back({4});
  decode_tokens.push_back({5});
  decode_tokens.push_back({6});
  auto fake_llm_executor = std::make_unique<FakeLlmExecutor>(
      /*vocab_size=*/10,
      /*prefill_tokens=*/std::move(prefill_tokens),
      /*decode_tokens=*/std::move(decode_tokens));

  // Inject an error.
  fake_llm_executor->SetPrefillStatus(absl::InternalError("Executor failed"));

  CreateExecutionManager(std::move(fake_llm_executor));

  ASSERT_OK_AND_ASSIGN(auto session_config, CreateDefaultSessionConfig());
  ASSERT_OK_AND_ASSIGN(const SessionId session_id,
                       execution_manager_->RegisterNewSession(session_config));

  std::vector<InputData> inputs;
  ASSERT_OK_AND_ASSIGN(auto input_text,
                       tokenizer_->TokenIdsToTensorBuffer({1, 2, 3}));
  inputs.push_back(InputText(std::move(input_text)));
  std::optional<BenchmarkInfo> benchmark_info = std::nullopt;
  absl::Status task_a_status = absl::OkStatus();
  ASSERT_OK_AND_ASSIGN(const TaskId task_a_id,
                       execution_manager_->GetNewTaskId());
  ASSERT_OK(execution_manager_->AddPrefillTask(
      session_id, task_a_id, std::move(inputs), {},
      /*cancelled=*/std::make_shared<std::atomic<bool>>(false),
      [&](absl::StatusOr<Responses> responses) {
        task_a_status = responses.status();
      }));

  EXPECT_OK(execution_manager_->WaitUntilDone(task_a_id, absl::Seconds(1)));
  EXPECT_EQ(task_a_status, absl::InternalError("Executor failed"));

  absl::Status task_b_status = absl::OkStatus();
  std::vector<TaskState> task_b_states;
  ASSERT_OK_AND_ASSIGN(const TaskId task_b_id,
                       execution_manager_->GetNewTaskId());
  ASSERT_OK(execution_manager_->AddDecodeTask(
      session_id, task_b_id,
      /*dependency_task_ids=*/{task_a_id},
      /*constraint=*/nullptr,
      /*cancelled=*/std::make_shared<std::atomic<bool>>(false),
      [&](absl::StatusOr<Responses> responses) {
        task_b_status = responses.status();
        if (responses.ok()) {
          task_b_states.push_back(responses->GetTaskState());
        }
      }));

  EXPECT_OK(execution_manager_->WaitUntilDone(task_b_id, absl::Seconds(1)));
  EXPECT_EQ(task_b_status, absl::OkStatus());
  EXPECT_THAT(task_b_states, ElementsAre(TaskState::kDependentTaskFailed));
}

TEST_P(ExecutionManagerTest, AddDecodeTaskWithConstraintWithInternalSampler) {
  // The default execution manager is using the internal sampler.
  CreateExecutionManager(CreateDefaultFakeLlmExecutor());

  ASSERT_OK_AND_ASSIGN(auto session_config, CreateDefaultSessionConfig());
  ASSERT_OK_AND_ASSIGN(const SessionId session_id,
                       execution_manager_->RegisterNewSession(session_config));

  std::vector<InputData> inputs;
  ASSERT_OK_AND_ASSIGN(auto input_text,
                       tokenizer_->TokenIdsToTensorBuffer({1, 2, 3}));
  inputs.push_back(InputText(std::move(input_text)));
  std::optional<BenchmarkInfo> benchmark_info = std::nullopt;
  ASSERT_OK_AND_ASSIGN(const TaskId task_a_id,
                       execution_manager_->GetNewTaskId());
  ASSERT_OK(execution_manager_->AddPrefillTask(
      session_id, task_a_id, std::move(inputs),
      /*dependency_task_ids=*/{},
      /*cancelled=*/std::make_shared<std::atomic<bool>>(false),
      /*callback=*/nullptr));

  ASSERT_OK_AND_ASSIGN(const TaskId task_b_id,
                       execution_manager_->GetNewTaskId());
  // Fake constraint that expects "45".
  std::vector<int> expected_token_ids = {4, 0};
  auto constraint = FakeConstraint(expected_token_ids, /*vocabulary_size=*/10);
  auto decode_config = DecodeConfig::CreateDefault();
  decode_config.SetConstraint(&constraint);
  std::vector<std::string> response_texts;
  absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback =
      [&response_texts](absl::StatusOr<Responses> responses) {
        ASSERT_OK(responses);
        if (!responses->GetTexts().empty()) {
          response_texts.push_back(responses->GetTexts()[0]);
        }
      };

  ASSERT_OK(execution_manager_->AddDecodeTask(
      session_id, task_b_id,
      /*dependency_task_ids=*/{task_a_id}, decode_config.GetConstraint(),
      /*cancelled=*/std::make_shared<std::atomic<bool>>(false),
      std::move(callback)));

  EXPECT_OK(execution_manager_->WaitUntilDone(task_b_id, absl::Seconds(3)));

  EXPECT_THAT(response_texts, ElementsAre("4"));
}

TEST_P(ExecutionManagerTest, AddDecodeTaskWithConstraintWithExternalSampler) {
  auto prefill_tokens = std::vector<std::vector<int>>{};
  auto decode_tokens = std::vector<std::vector<int>>{};
  prefill_tokens.push_back({1, 2, 3});
  prefill_tokens.push_back({0});
  decode_tokens.push_back({4});
  decode_tokens.push_back({5});
  decode_tokens.push_back({6});

  CreateExecutionManager(std::make_unique<FakeLlmExecutor>(
      /*vocab_size=*/10,
      /*prefill_tokens=*/std::move(prefill_tokens),
      /*decode_tokens=*/std::move(decode_tokens)));

  ASSERT_OK_AND_ASSIGN(auto session_config, CreateDefaultSessionConfig(
                                                /*use_external_sampler=*/true));
  ASSERT_OK_AND_ASSIGN(const SessionId session_id,
                       execution_manager_->RegisterNewSession(session_config));

  std::vector<InputData> inputs;
  ASSERT_OK_AND_ASSIGN(auto input_text,
                       tokenizer_->TokenIdsToTensorBuffer({1, 2, 3}));
  inputs.push_back(InputText(std::move(input_text)));
  std::optional<BenchmarkInfo> benchmark_info = std::nullopt;
  ASSERT_OK_AND_ASSIGN(const TaskId task_a_id,
                       execution_manager_->GetNewTaskId());
  ASSERT_OK(execution_manager_->AddPrefillTask(
      session_id, task_a_id, std::move(inputs),
      /*dependency_task_ids=*/{},
      /*cancelled=*/std::make_shared<std::atomic<bool>>(false),
      /*callback=*/nullptr));

  ASSERT_OK_AND_ASSIGN(const TaskId task_b_id,
                       execution_manager_->GetNewTaskId());
  // Fake constraint that expects "45".
  std::vector<int> expected_token_ids = {4, 0};
  auto constraint = FakeConstraint(expected_token_ids, /*vocabulary_size=*/10);
  auto decode_config = DecodeConfig::CreateDefault();
  decode_config.SetConstraint(&constraint);
  std::vector<std::string> response_texts;
  absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback =
      [&response_texts](absl::StatusOr<Responses> responses) {
        ASSERT_OK(responses);
        if (!responses->GetTexts().empty()) {
          response_texts.push_back(responses->GetTexts()[0]);
        }
      };

  ASSERT_OK(execution_manager_->AddDecodeTask(
      session_id, task_b_id,
      /*dependency_task_ids=*/{task_a_id}, decode_config.GetConstraint(),
      /*cancelled=*/std::make_shared<std::atomic<bool>>(false),
      std::move(callback)));

  EXPECT_OK(execution_manager_->WaitUntilDone(task_b_id, absl::Seconds(3)));

  EXPECT_THAT(response_texts, ElementsAre("4"));
}

TEST_P(ExecutionManagerTest, AddTextScoringTask) {
  CreateExecutionManager(CreateDefaultFakeLlmExecutor());
  ASSERT_OK_AND_ASSIGN(auto session_config, CreateDefaultSessionConfig());
  ASSERT_OK_AND_ASSIGN(const SessionId session_id,
                       execution_manager_->RegisterNewSession(session_config));

  std::vector<TaskState> task_states;
  std::vector<float> scores;
  absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback =
      [&task_states, &scores](absl::StatusOr<Responses> responses) {
        ASSERT_OK(responses);
        task_states.push_back(responses->GetTaskState());
        if (!responses->GetScores().empty()) {
          scores.push_back(responses->GetScores()[0]);
        }
      };

  std::vector<InputData> inputs;
  ASSERT_OK_AND_ASSIGN(auto input_text,
                       tokenizer_->TokenIdsToTensorBuffer({1, 2, 3}));
  inputs.push_back(InputText(std::move(input_text)));
  ASSERT_OK_AND_ASSIGN(const TaskId prefill_task_id,
                       execution_manager_->GetNewTaskId());
  ASSERT_OK(execution_manager_->AddPrefillTask(
      session_id, prefill_task_id, std::move(inputs),
      /*dependency_task_ids=*/{},
      /*cancelled=*/std::make_shared<std::atomic<bool>>(false),
      /*callback=*/[](absl::StatusOr<Responses> responses) {}));
  ASSERT_OK(
      execution_manager_->WaitUntilDone(prefill_task_id, absl::Seconds(3)));

  ASSERT_OK_AND_ASSIGN(const TaskId scoring_task_id,
                       execution_manager_->GetNewTaskId());
  const std::vector<absl::string_view> target_text = {"45"};
  EXPECT_CALL(*tokenizer_, TextToTokenIds("45"))
      .WillOnce(Return(std::vector<int>({4, 5})));

  ASSERT_OK(execution_manager_->AddTextScoringTask(
      session_id, scoring_task_id,
      /*dep_tasks=*/{}, target_text,
      /*store_token_lengths=*/false,
      /*cancelled=*/std::make_shared<std::atomic<bool>>(false),
      std::move(callback)));

  EXPECT_OK(
      execution_manager_->WaitUntilDone(scoring_task_id, absl::Seconds(3)));

  EXPECT_THAT(task_states,
              ElementsAre(TaskState::kCreated, TaskState::kQueued,
                          TaskState::kProcessing, TaskState::kDone));

  // The FakeLlmExecutor is set up to expect tokens 4, 5, 6.
  // The target text "45" corresponds to tokens 4, 5.
  // The fake executor will produce logits that give prob 1 to the next
  // expected token. So for the first token '4', the expected is '4', prob is 1,
  // log-prob is 0. For the second token '5', the expected is '5', prob is 1,
  // log-prob is 0. Total score is 0.
  ASSERT_EQ(scores.size(), 1);
  EXPECT_FLOAT_EQ(scores[0], 0.0f);
}

TEST_P(ExecutionManagerTest, AddCloneSessionTask) {
  CreateExecutionManager(CreateDefaultFakeLlmExecutor());
  ASSERT_OK_AND_ASSIGN(auto session_config, CreateDefaultSessionConfig());
  ASSERT_OK_AND_ASSIGN(const SessionId session_id,
                       execution_manager_->RegisterNewSession(session_config));

  ASSERT_OK_AND_ASSIGN(const TaskId prefill_task_id,
                       execution_manager_->GetNewTaskId());

  std::vector<InputData> inputs;
  ASSERT_OK_AND_ASSIGN(auto input_text,
                       tokenizer_->TokenIdsToTensorBuffer({1, 2, 3}));
  inputs.push_back(InputText(std::move(input_text)));
  ASSERT_OK(execution_manager_->AddPrefillTask(
      session_id, prefill_task_id, std::move(inputs), {},
      std::make_shared<std::atomic<bool>>(false), nullptr));
  EXPECT_OK(
      execution_manager_->WaitUntilDone(prefill_task_id, absl::Seconds(3)));

  ASSERT_OK_AND_ASSIGN(const SessionId cloned_session_id,
                       execution_manager_->RegisterNewSession(session_config));
  ASSERT_OK_AND_ASSIGN(const TaskId clone_task_id,
                       execution_manager_->GetNewTaskId());

  bool clone_task_done = false;
  ASSERT_OK(execution_manager_->AddCloneSessionTask(
      session_id, clone_task_id, {}, cloned_session_id,
      std::make_shared<std::atomic<bool>>(false),
      [&clone_task_done](absl::StatusOr<Responses> responses) {
        ASSERT_OK(responses);
        if (responses->GetTaskState() == TaskState::kDone) {
          clone_task_done = true;
        }
      }));

  EXPECT_OK(execution_manager_->WaitUntilDone(clone_task_id, absl::Seconds(3)));
  EXPECT_TRUE(clone_task_done);

  // Verify cloned session state.
  ASSERT_OK_AND_ASSIGN(auto cloned_session_info,
                       execution_manager_->GetSessionInfo(cloned_session_id));
  EXPECT_EQ(cloned_session_info->last_prefill_token_id, 3);
}

TEST_P(ExecutionManagerTest, GetCurrentStep) {
  CreateExecutionManager(CreateDefaultFakeLlmExecutor());
  ASSERT_OK_AND_ASSIGN(auto session_config, CreateDefaultSessionConfig());
  ASSERT_OK_AND_ASSIGN(const SessionId session_id,
                       execution_manager_->RegisterNewSession(session_config));
  ASSERT_OK_AND_ASSIGN(auto session_info,
                       execution_manager_->GetSessionInfo(session_id));

  // Initially step should be 0.
  ASSERT_OK_AND_ASSIGN(int step1,
                       execution_manager_->GetCurrentStep(*session_info));
  EXPECT_EQ(step1, 0);

  // Run a prefill task.
  std::vector<InputData> inputs;
  ASSERT_OK_AND_ASSIGN(auto input_text,
                       tokenizer_->TokenIdsToTensorBuffer({1, 2, 3}));
  inputs.push_back(InputText(std::move(input_text)));
  ASSERT_OK_AND_ASSIGN(const TaskId task_id,
                       execution_manager_->GetNewTaskId());
  ASSERT_OK(execution_manager_->AddPrefillTask(
      session_id, task_id, std::move(inputs), {},
      std::make_shared<std::atomic<bool>>(false), nullptr));
  EXPECT_OK(execution_manager_->WaitUntilDone(task_id, absl::Seconds(3)));

  // After prefill, step should be updated (3 tokens).
  ASSERT_OK_AND_ASSIGN(int step2,
                       execution_manager_->GetCurrentStep(*session_info));
  EXPECT_EQ(step2, 3);
}

TEST_P(ExecutionManagerTest, SetCurrentStep) {
  CreateExecutionManager(CreateDefaultFakeLlmExecutor());
  ASSERT_OK_AND_ASSIGN(auto session_config, CreateDefaultSessionConfig());
  ASSERT_OK_AND_ASSIGN(const SessionId session_id,
                       execution_manager_->RegisterNewSession(session_config));
  ASSERT_OK_AND_ASSIGN(auto session_info,
                       execution_manager_->GetSessionInfo(session_id));

  // Run a prefill task to increase step to 3.
  std::vector<InputData> inputs;
  ASSERT_OK_AND_ASSIGN(auto input_text,
                       tokenizer_->TokenIdsToTensorBuffer({1, 2, 3}));
  inputs.push_back(InputText(std::move(input_text)));
  ASSERT_OK_AND_ASSIGN(const TaskId task_id,
                       execution_manager_->GetNewTaskId());
  ASSERT_OK(execution_manager_->AddPrefillTask(
      session_id, task_id, std::move(inputs), {},
      std::make_shared<std::atomic<bool>>(false), nullptr));
  EXPECT_OK(execution_manager_->WaitUntilDone(task_id, absl::Seconds(3)));

  // Verify current step is 3.
  ASSERT_OK_AND_ASSIGN(int step1,
                       execution_manager_->GetCurrentStep(*session_info));
  EXPECT_EQ(step1, 3);

  // Set current step to 1.
  EXPECT_OK(execution_manager_->SetCurrentStep(*session_info, 1));

  // Verify current step is now 1.
  ASSERT_OK_AND_ASSIGN(int step2,
                       execution_manager_->GetCurrentStep(*session_info));
  EXPECT_EQ(step2, 1);

  // Try to set current step to 5 (greater than current step 1).
  EXPECT_THAT(execution_manager_->SetCurrentStep(*session_info, 5),
              testing::status::StatusIs(absl::StatusCode::kInvalidArgument));
}

INSTANTIATE_TEST_SUITE_P(ExecutionManagerTests, ExecutionManagerTest,
                         ::testing::Values(ExecutionManagerType::kThreaded,
                                           ExecutionManagerType::kSerial));
}  // namespace
}  // namespace litert::lm
