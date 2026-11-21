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
#include <cstdint>
#include <filesystem>  // NOLINT
#include <fstream>
#include <ios>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/constrained_decoding/fake_constraint.h"
#include "runtime/components/constrained_decoding/no_repeat_ngram_config.h"
#include "runtime/components/constrained_decoding/repetition_penalty_config.h"
#include "runtime/components/constrained_decoding/suppress_tokens_config.h"
#include "runtime/components/model_resources.h"
#include "runtime/components/random_state.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/audio/audio_executor.h"
#include "runtime/executor/audio/audio_executor_settings.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/executor_stats.h"
#include "runtime/executor/fake_llm_executor.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/framework/resource_management/serial_execution_manager.h"
#include "runtime/framework/resource_management/threaded_execution_manager.h"
#include "runtime/proto/sampler_params.pb.h"
#include "runtime/proto/token.pb.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/runtime_debugger.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/status_macros.h"  // IWYU pragma: keep
#include "runtime/util/test_utils.h"  // NOLINT
#include "support/tokenizer/tokenizer.h"

namespace litert::lm {

using Tokenizer = ::litert::support::Tokenizer;
using TokenizerType = ::litert::support::TokenizerType;

namespace {

using ::testing::Each;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::Not;
using ::testing::Return;

constexpr int kVocabSize = 10;
constexpr int kValidTokensInAudioData = 4;

class MockTokenizer : public Tokenizer {
 public:
  MOCK_METHOD(absl::StatusOr<std::vector<int>>, TextToTokenIds,
              (absl::string_view text), (override));
  MOCK_METHOD(absl::StatusOr<int>, TokenToId, (absl::string_view token),
              (override));
  MOCK_METHOD(absl::StatusOr<std::string>, TokenIdsToText,
              (absl::Span<const int> token_ids, bool skip_special_tokens),
              (override));
  MOCK_METHOD(TokenizerType, GetTokenizerType, (), (const, override));
  MOCK_METHOD(std::vector<std::string>, GetTokens, (), (const, override));
  MOCK_METHOD(int, GetVocabSize, (), (const, override));
};

class FakeAudioContext : public AudioContext {
 public:
  FakeAudioContext() = default;
  absl::StatusOr<std::unique_ptr<AudioContext>> Clone() const override {
    return std::make_unique<FakeAudioContext>(*this);
  }
};

class FakeAudioExecutor : public AudioExecutor {
 public:
  absl::StatusOr<ExecutorAudioData> Encode(
      const TensorBuffer& spectrogram_tensor) override {
    ExecutorAudioData data;
    data.SetValidTokens(kValidTokensInAudioData);
    return data;
  }

  absl::Status Reset() override {
    reset_called_ = true;
    return absl::OkStatus();
  }

  absl::StatusOr<ExecutorAudioData> Flush() override {
    flush_called_ = true;
    ExecutorAudioData data;
    data.SetValidTokens(kValidTokensInAudioData);
    return data;
  }

  absl::StatusOr<std::unique_ptr<AudioContext>> CreateNewContext() override {
    return std::make_unique<FakeAudioContext>();
  }

  absl::StatusOr<std::unique_ptr<AudioContext>> CloneContext() override {
    return std::make_unique<FakeAudioContext>();
  }

  absl::StatusOr<std::unique_ptr<AudioContext>> CloneContext(
      const AudioContext& audio_context) override {
    return audio_context.Clone();
  }

  absl::Status RestoreContext(
      std::unique_ptr<AudioContext> audio_context) override {
    return absl::OkStatus();
  }

  absl::Status LoadLoRA(uint32_t lora_id,
                        const ModelAssets& model_assets) override {
    if (!load_lora_status_.ok()) {
      return load_lora_status_;
    }
    loaded_lora_ids_.push_back(lora_id);
    return absl::OkStatus();
  }

  absl::Status UseLoRA(std::optional<uint32_t> lora_id) override {
    used_lora_ids_.push_back(lora_id);
    return absl::OkStatus();
  }

  bool reset_called_ = false;
  bool flush_called_ = false;
  absl::Status load_lora_status_ = absl::OkStatus();
  std::vector<uint32_t> loaded_lora_ids_;
  std::vector<std::optional<uint32_t>> used_lora_ids_;
};

// An audio context that records how many chunks were encoded when it was
// captured.
class StatefulFakeAudioContext : public AudioContext {
 public:
  explicit StatefulFakeAudioContext(int num_encoded_chunks)
      : num_encoded_chunks_(num_encoded_chunks) {}
  absl::StatusOr<std::unique_ptr<AudioContext>> Clone() const override {
    return std::make_unique<StatefulFakeAudioContext>(num_encoded_chunks_);
  }
  int num_encoded_chunks() const { return num_encoded_chunks_; }

 private:
  int num_encoded_chunks_;
};

// A streaming audio executor whose only state is the number of encoded chunks,
// so tests can verify that context save/restore reflects the live executor
// state.
class StatefulFakeAudioExecutor : public AudioExecutor {
 public:
  absl::StatusOr<ExecutorAudioData> Encode(
      const TensorBuffer& spectrogram_tensor) override {
    ++num_encoded_chunks_;
    ExecutorAudioData data;
    data.SetValidTokens(kValidTokensInAudioData);
    return data;
  }

  absl::Status Reset() override {
    num_encoded_chunks_ = 0;
    return absl::OkStatus();
  }

  absl::StatusOr<ExecutorAudioData> Flush() override {
    ExecutorAudioData data;
    data.SetValidTokens(kValidTokensInAudioData);
    return data;
  }

  absl::StatusOr<AudioExecutorProperties> GetAudioExecutorProperties()
      const override {
    AudioExecutorProperties properties;
    properties.is_streaming_model = true;
    return properties;
  }

  absl::StatusOr<std::unique_ptr<AudioContext>> CreateNewContext() override {
    ++num_create_new_context_calls_;
    return std::make_unique<StatefulFakeAudioContext>(0);
  }

  absl::StatusOr<std::unique_ptr<AudioContext>> CloneContext() override {
    ++num_clone_context_calls_;
    return std::make_unique<StatefulFakeAudioContext>(num_encoded_chunks_);
  }

  absl::StatusOr<std::unique_ptr<AudioContext>> CloneContext(
      const AudioContext& audio_context) override {
    return audio_context.Clone();
  }

  absl::Status RestoreContext(
      std::unique_ptr<AudioContext> audio_context) override {
    num_encoded_chunks_ =
        static_cast<const StatefulFakeAudioContext&>(*audio_context)
            .num_encoded_chunks();
    return absl::OkStatus();
  }

  int num_encoded_chunks_ = 0;
  int num_create_new_context_calls_ = 0;
  int num_clone_context_calls_ = 0;
};

// A fake LLM executor that supports the runtime config/state round-trip the
// resource manager performs when switching between or cloning context handlers
// that share the same processed context.
class ContextSwitchingFakeLlmExecutor : public FakeLlmExecutor {
 public:
  using FakeLlmExecutor::FakeLlmExecutor;

  absl::StatusOr<RuntimeConfig> GetRuntimeConfig() const override {
    return RuntimeConfig();
  }

  absl::StatusOr<RuntimeState> GetRuntimeState() const override {
    return RuntimeState();
  }

  absl::Status UpdateRuntimeState(const RuntimeState& runtime_state) override {
    return absl::OkStatus();
  }
};

// A fake LLM executor that records the random state of each prefill, decode,
// and draft verification call.
class RandomStateRecordingFakeLlmExecutor : public FakeLlmExecutor {
 public:
  using FakeLlmExecutor::FakeLlmExecutor;

  absl::Status Prefill(const ExecutorInputs& inputs,
                       const ExecutorPrefillParams& prefill_params) override {
    prefill_random_states_.push_back(prefill_params.GetRandomState());
    return FakeLlmExecutor::Prefill(inputs, prefill_params);
  }

  absl::StatusOr<std::vector<std::vector<int>>> Decode(
      const ExecutorDecodeParams& decode_params) override {
    decode_random_states_.push_back(decode_params.GetRandomState());
    return FakeLlmExecutor::Decode(decode_params);
  }

  absl::StatusOr<DraftVerificationResult> VerifyDraft(
      absl::Span<const int> draft_token_ids,
      const ExecutorDecodeParams& decode_params) override {
    verify_draft_random_states_.push_back(decode_params.GetRandomState());
    return FakeLlmExecutor::VerifyDraft(draft_token_ids, decode_params);
  }

  const std::vector<RandomState*>& prefill_random_states() const {
    return prefill_random_states_;
  }
  const std::vector<RandomState*>& decode_random_states() const {
    return decode_random_states_;
  }
  const std::vector<RandomState*>& verify_draft_random_states() const {
    return verify_draft_random_states_;
  }

 private:
  std::vector<RandomState*> prefill_random_states_;
  std::vector<RandomState*> decode_random_states_;
  std::vector<RandomState*> verify_draft_random_states_;
};

// A fake LLM executor that sleeps before every decode step, simulating a slow
// but steadily progressing decode.
class SlowDecodeStepFakeLlmExecutor : public ContextSwitchingFakeLlmExecutor {
 public:
  SlowDecodeStepFakeLlmExecutor(int vocab_size,
                                std::vector<std::vector<int>> prefill_tokens,
                                std::vector<std::vector<int>> decode_tokens,
                                absl::Duration step_delay)
      : ContextSwitchingFakeLlmExecutor(vocab_size, std::move(prefill_tokens),
                                        std::move(decode_tokens)),
        step_delay_(step_delay) {}

  absl::StatusOr<std::vector<std::vector<int>>> Decode(
      const ExecutorDecodeParams& decode_params) override {
    absl::SleepFor(step_delay_);
    return ContextSwitchingFakeLlmExecutor::Decode(decode_params);
  }

 private:
  absl::Duration step_delay_;
};

// A fake LLM executor that simulates a multi-chunk prefill, sleeping before
// each chunk and optionally reporting progress through `ExecutorPrefillParams`.
class SlowPrefillChunkFakeLlmExecutor : public FakeLlmExecutor {
 public:
  SlowPrefillChunkFakeLlmExecutor(int vocab_size,
                                  std::vector<std::vector<int>> prefill_tokens,
                                  std::vector<std::vector<int>> decode_tokens,
                                  int num_chunks, absl::Duration chunk_delay,
                                  bool record_progress)
      : FakeLlmExecutor(vocab_size, std::move(prefill_tokens),
                        std::move(decode_tokens)),
        num_chunks_(num_chunks),
        chunk_delay_(chunk_delay),
        record_progress_(record_progress) {}

  absl::Status Prefill(const ExecutorInputs& inputs,
                       const ExecutorPrefillParams& prefill_params) override {
    for (int i = 0; i < num_chunks_; ++i) {
      absl::SleepFor(chunk_delay_);
      if (record_progress_) {
        prefill_params.RecordProgress();
      }
    }
    return FakeLlmExecutor::Prefill(inputs);
  }

 private:
  int num_chunks_;
  absl::Duration chunk_delay_;
  bool record_progress_;
};

// Returns a callback that records the terminal status of a task.
absl::AnyInvocable<void(absl::StatusOr<Responses>)>
CaptureTerminalStatusCallback(absl::Status* status) {
  *status = absl::InternalError("Task did not finish.");
  return [status](absl::StatusOr<Responses> responses) {
    if (!responses.ok()) {
      *status = responses.status();
    } else if (responses->GetTaskState() == TaskState::kDone) {
      *status = absl::OkStatus();
    }
  };
}

class NonStreamingFakeAudioExecutor : public AudioExecutor {
 public:
  absl::StatusOr<ExecutorAudioData> Encode(
      const TensorBuffer& spectrogram_tensor) override {
    ExecutorAudioData data;
    data.SetValidTokens(kValidTokensInAudioData);
    return data;
  }

  absl::Status Reset() override {
    reset_called_ = true;
    return absl::OkStatus();
  }

  absl::StatusOr<ExecutorAudioData> Flush() override {
    flush_called_ = true;
    ExecutorAudioData data;
    data.SetValidTokens(0);
    return data;
  }

  // Intentionally do NOT override CreateNewContext, CloneContext,
  // RestoreContext; they return absl::UnimplementedError from
  // AudioExecutorBase.

  bool reset_called_ = false;
  bool flush_called_ = false;
};

class FailingAudioExecutor : public AudioExecutor {
 public:
  absl::StatusOr<ExecutorAudioData> Encode(
      const TensorBuffer& spectrogram_tensor) override {
    return absl::InternalError(
        "Encode() should not be called when precomputed audio embeddings are "
        "provided.");
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
    EXPECT_CALL(*tokenizer_, TokenIdsToText(testing::_, false))
        .WillRepeatedly(
            [](absl::Span<const int> ids, bool skip_special_tokens) {
              std::string result;
              for (int id : ids) {
                result += std::to_string(id);
              }
              return result;
            });
    EXPECT_CALL(*tokenizer_, GetVocabSize()).WillRepeatedly(Return(kVocabSize));
  }

  absl::StatusOr<SessionConfig> CreateDefaultSessionConfig(
      bool use_external_sampler = false) {
    ABSL_ASSIGN_OR_RETURN(auto model_assets,
                          ModelAssets::Create("test_model_path_1"));
    ABSL_ASSIGN_OR_RETURN(auto settings,
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
      std::unique_ptr<AudioExecutor> audio_executor = nullptr,
      std::shared_ptr<RuntimeDebugger> runtime_debugger = nullptr) {
    // The objects are moved to execution_manager_ so we can't access them
    // after creation.
    if (GetParam() == ExecutionManagerType::kThreaded) {
      ASSERT_OK_AND_ASSIGN(
          execution_manager_,
          ThreadedExecutionManager::Create(
              /*tokenizer=*/tokenizer_.get(),
              /*model_resources=*/model_resources_.get(),
              /*llm_executor=*/std::move(fake_llm_executor),
              /*vision_executor_settings=*/nullptr,
              std::move(audio_executor_settings),
              /*litert_env=*/nullptr, std::move(audio_executor),
              std::move(runtime_debugger)));
    } else {
      ASSERT_OK_AND_ASSIGN(
          execution_manager_,
          SerialExecutionManager::Create(
              /*tokenizer=*/tokenizer_.get(),
              /*model_resources=*/model_resources_.get(),
              /*llm_executor=*/std::move(fake_llm_executor),
              /*vision_executor_settings=*/nullptr,
              std::move(audio_executor_settings),
              /*litert_env=*/nullptr, std::move(audio_executor),
              std::move(runtime_debugger)));
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
        kVocabSize,
        /*prefill_tokens=*/std::move(prefill_tokens),
        /*decode_tokens=*/std::move(decode_tokens));
  }

  // Adds a prefill of {1, 2, 3} to `session_id`, followed by a decode if
  // `with_decode` is true.
  absl::Status AddPrefillAndDecodeTasks(SessionId session_id,
                                        bool with_decode = true) {
    std::vector<InputData> inputs;
    ABSL_ASSIGN_OR_RETURN(auto input_text,
                          tokenizer_->TokenIdsToTensorBuffer({1, 2, 3}));
    inputs.push_back(InputText(std::move(input_text)));
    ABSL_ASSIGN_OR_RETURN(const TaskId prefill_task_id,
                          execution_manager_->GetNewTaskId());
    ABSL_RETURN_IF_ERROR(execution_manager_->AddPrefillTask(
        session_id, prefill_task_id, std::move(inputs),
        /*dependency_task_ids=*/{},
        /*cancelled=*/std::make_shared<std::atomic<bool>>(false),
        /*callback=*/[](absl::StatusOr<Responses> responses) {}));
    if (!with_decode) {
      return absl::OkStatus();
    }
    ABSL_ASSIGN_OR_RETURN(const TaskId decode_task_id,
                          execution_manager_->GetNewTaskId());
    return execution_manager_->AddDecodeTask(
        session_id, decode_task_id,
        /*dependency_task_ids=*/{prefill_task_id},
        RepetitionPenaltyConfig::Default(), NoRepeatNgramConfig::Default(),
        SuppressTokensConfig::Default(),
        /*constraint=*/nullptr,
        /*cancelled=*/std::make_shared<std::atomic<bool>>(false),
        /*callback=*/[](absl::StatusOr<Responses> responses) {});
  }

  std::unique_ptr<MockTokenizer> tokenizer_;

  std::unique_ptr<ModelResources> model_resources_;

  std::unique_ptr<ExecutionManager> execution_manager_;
};

TEST_P(ExecutionManagerTest, IsDebuggerEnabledIsFalseWithoutRuntimeDebugger) {
  CreateExecutionManager(CreateDefaultFakeLlmExecutor());
  EXPECT_FALSE(execution_manager_->IsDebuggerEnabled());
}

TEST_P(ExecutionManagerTest, IsDebuggerEnabledIsTrueWithRuntimeDebugger) {
  std::shared_ptr<RuntimeDebugger> runtime_debugger =
      RuntimeDebugger::Create(::testing::TempDir());
  ASSERT_NE(runtime_debugger, nullptr);
  CreateExecutionManager(CreateDefaultFakeLlmExecutor(),
                         /*audio_executor_settings=*/nullptr,
                         /*audio_executor=*/nullptr, runtime_debugger);
  EXPECT_TRUE(execution_manager_->IsDebuggerEnabled());
}

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

  if (GetParam() == ExecutionManagerType::kThreaded) {
    EXPECT_THAT(task_states,
                ElementsAre(TaskState::kCreated, TaskState::kQueued,
                            TaskState::kProcessing, TaskState::kDone));
  } else {
    // Serial execution might skip some states or report them differently.
    // In our implementation, callback is called multiple times.
    EXPECT_THAT(task_states,
                ElementsAre(TaskState::kCreated, TaskState::kQueued,
                            TaskState::kProcessing, TaskState::kDone));
  }
}

TEST_P(ExecutionManagerTest, AddPrefillTaskWithAudioModality) {
  auto fake_llm_executor = CreateDefaultFakeLlmExecutor();

  ASSERT_OK_AND_ASSIGN(auto* settings,
                       fake_llm_executor->GetMutableExecutorSettings());
  EXPECT_OK(settings->SetBackend(Backend::GPU_ARTISAN));

  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create("test_model_path_2"));
  ASSERT_OK_AND_ASSIGN(
      auto audio_settings,
      AudioExecutorSettings::CreateDefault(
          model_assets, 128, Backend::GPU_ARTISAN, Backend::GPU_ARTISAN));

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

TEST_P(ExecutionManagerTest,
       AddPrefillTaskResetsAudioExecutorOnInputAudioEndAndCombinesMultiAudio) {
  // Two audio clips in one prefill:
  // Clip 1: Encode (2 tokens) + InputAudioEnd (Flush 0 tokens, Reset called)
  // Clip 2: Encode (2 tokens) + InputAudioEnd (Flush 0 tokens, Reset called)
  const std::vector<float> expected_combined_audio_embeddings = {
      1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  auto fake_llm_executor = std::make_unique<FakeLlmExecutor>(
      kVocabSize,
      std::vector<std::vector<int>>{
          {1, ExecutorAudioData::kSpecialToken,
           ExecutorAudioData::kSpecialToken, ExecutorAudioData::kEndToken, 2,
           ExecutorAudioData::kSpecialToken, ExecutorAudioData::kSpecialToken,
           ExecutorAudioData::kEndToken}},
      std::vector<std::vector<int>>{{4}, {5}, {6}},
      /*batch_size=*/1,
      /*projected_audio_embedding=*/expected_combined_audio_embeddings);

  ASSERT_OK_AND_ASSIGN(auto* settings,
                       fake_llm_executor->GetMutableExecutorSettings());
  EXPECT_OK(settings->SetBackend(Backend::GPU_ARTISAN));

  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create("test_model_path_2"));
  ASSERT_OK_AND_ASSIGN(
      auto audio_settings,
      AudioExecutorSettings::CreateDefault(
          model_assets, 128, Backend::GPU_ARTISAN, Backend::GPU_ARTISAN));

  class MultiClipFakeAudioExecutor : public AudioExecutor {
   public:
    absl::StatusOr<ExecutorAudioData> Encode(
        const TensorBuffer& spectrogram_tensor) override {
      ++encode_calls_;
      ++chunks_since_reset_;
      chunks_since_reset_at_encode_.push_back(chunks_since_reset_);
      // Emit a 5D padded Artisan tensor [1, 1, 5, 1, 2] where only the first 2
      // tokens are valid.
      std::vector<float> values(10, -999.0f);
      const float base = (encode_calls_ == 1) ? 1.0f : 5.0f;
      for (int i = 0; i < 4; ++i) {
        values[i] = base + static_cast<float>(i);
      }
      auto audio_buffer = CopyToTensorBuffer<float>(values, {1, 1, 5, 1, 2});
      if (!audio_buffer.HasValue()) {
        return absl::InternalError("Failed to create audio tensor buffer.");
      }
      return ExecutorAudioData(std::move(*audio_buffer), std::nullopt,
                               /*valid_tokens=*/2);
    }

    absl::StatusOr<ExecutorAudioData> Flush() override {
      ++flush_calls_;
      ExecutorAudioData data;
      data.SetValidTokens(0);
      return data;
    }

    absl::Status Reset() override {
      ++reset_calls_;
      chunks_since_reset_ = 0;
      return absl::OkStatus();
    }

    int encode_calls_ = 0;
    int flush_calls_ = 0;
    int reset_calls_ = 0;
    int chunks_since_reset_ = 0;
    std::vector<int> chunks_since_reset_at_encode_;
  };

  auto fake_audio_executor = std::make_unique<MultiClipFakeAudioExecutor>();
  auto* fake_audio_executor_ptr = fake_audio_executor.get();

  CreateExecutionManager(
      std::move(fake_llm_executor),
      std::make_unique<AudioExecutorSettings>(std::move(audio_settings)),
      std::move(fake_audio_executor));

  ASSERT_OK_AND_ASSIGN(auto session_config, CreateDefaultSessionConfig());
  session_config.SetAudioModalityEnabled(true);
  ASSERT_OK_AND_ASSIGN(const SessionId session_id,
                       execution_manager_->RegisterNewSession(session_config));

  std::vector<InputData> inputs;
  ASSERT_OK_AND_ASSIGN(auto text1, tokenizer_->TokenIdsToTensorBuffer({1}));
  inputs.push_back(InputText(std::move(text1)));
  auto spectrogram_1 = CopyToTensorBuffer<float>({1.0f, 2.0f}, {1, 2});
  ASSERT_TRUE(spectrogram_1.HasValue());
  inputs.push_back(InputAudio(std::move(*spectrogram_1)));
  inputs.push_back(InputAudioEnd());

  ASSERT_OK_AND_ASSIGN(auto text2, tokenizer_->TokenIdsToTensorBuffer({2}));
  inputs.push_back(InputText(std::move(text2)));
  auto spectrogram_2 = CopyToTensorBuffer<float>({3.0f, 4.0f}, {1, 2});
  ASSERT_TRUE(spectrogram_2.HasValue());
  inputs.push_back(InputAudio(std::move(*spectrogram_2)));
  inputs.push_back(InputAudioEnd());

  std::vector<TaskState> task_states;
  ASSERT_OK_AND_ASSIGN(const TaskId task_id,
                       execution_manager_->GetNewTaskId());
  ASSERT_OK(execution_manager_->AddPrefillTask(
      session_id, task_id, std::move(inputs), {},
      std::make_shared<std::atomic<bool>>(false),
      [&task_states](absl::StatusOr<Responses> responses) {
        ASSERT_OK(responses);
        task_states.push_back(responses->GetTaskState());
      }));

  EXPECT_OK(execution_manager_->WaitUntilDone(task_id, absl::Seconds(3)));
  EXPECT_THAT(task_states,
              ElementsAre(TaskState::kCreated, TaskState::kQueued,
                          TaskState::kProcessing, TaskState::kDone));
  EXPECT_EQ(fake_audio_executor_ptr->encode_calls_, 2);
  EXPECT_EQ(fake_audio_executor_ptr->flush_calls_, 2);
  EXPECT_EQ(fake_audio_executor_ptr->reset_calls_, 2);
  // Because Reset() was called at the first InputAudioEnd, the second clip's
  // Encode() saw chunks_since_reset_ == 1 instead of 2.
  EXPECT_THAT(fake_audio_executor_ptr->chunks_since_reset_at_encode_,
              ElementsAre(1, 1));
}

TEST_P(ExecutionManagerTest,
       AddPrefillTaskIgnoresUnimplementedResetOnInputAudioEnd) {
  auto fake_llm_executor = std::make_unique<FakeLlmExecutor>(
      kVocabSize,
      std::vector<std::vector<int>>{{1, 2, 3, ExecutorAudioData::kEndToken}},
      std::vector<std::vector<int>>{{4}, {5}, {6}});

  ASSERT_OK_AND_ASSIGN(auto* settings,
                       fake_llm_executor->GetMutableExecutorSettings());
  EXPECT_OK(settings->SetBackend(Backend::GPU_ARTISAN));

  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create("test_model_path_2"));
  ASSERT_OK_AND_ASSIGN(
      auto audio_settings,
      AudioExecutorSettings::CreateDefault(
          model_assets, 128, Backend::GPU_ARTISAN, Backend::GPU_ARTISAN));

  class UnimplementedResetFakeAudioExecutor : public AudioExecutor {
   public:
    absl::StatusOr<ExecutorAudioData> Encode(
        const TensorBuffer& spectrogram_tensor) override {
      return absl::UnimplementedError("Encode not used");
    }

    absl::StatusOr<ExecutorAudioData> Flush() override {
      return absl::UnimplementedError("Flush not implemented");
    }

    absl::Status Reset() override {
      ++reset_calls_;
      return absl::UnimplementedError("Reset not implemented");
    }

    int reset_calls_ = 0;
  };

  auto fake_audio_executor =
      std::make_unique<UnimplementedResetFakeAudioExecutor>();
  auto* fake_audio_executor_ptr = fake_audio_executor.get();

  CreateExecutionManager(
      std::move(fake_llm_executor),
      std::make_unique<AudioExecutorSettings>(std::move(audio_settings)),
      std::move(fake_audio_executor));

  ASSERT_OK_AND_ASSIGN(auto session_config, CreateDefaultSessionConfig());
  session_config.SetAudioModalityEnabled(true);
  ASSERT_OK_AND_ASSIGN(const SessionId session_id,
                       execution_manager_->RegisterNewSession(session_config));

  std::vector<InputData> inputs;
  ASSERT_OK_AND_ASSIGN(auto input_text,
                       tokenizer_->TokenIdsToTensorBuffer({1, 2, 3}));
  inputs.push_back(InputText(std::move(input_text)));
  inputs.push_back(InputAudioEnd());

  std::vector<TaskState> task_states;
  ASSERT_OK_AND_ASSIGN(const TaskId task_id,
                       execution_manager_->GetNewTaskId());
  ASSERT_OK(execution_manager_->AddPrefillTask(
      session_id, task_id, std::move(inputs), {},
      std::make_shared<std::atomic<bool>>(false),
      [&task_states](absl::StatusOr<Responses> responses) {
        ASSERT_OK(responses);
        task_states.push_back(responses->GetTaskState());
      }));

  EXPECT_OK(execution_manager_->WaitUntilDone(task_id, absl::Seconds(3)));
  EXPECT_THAT(task_states,
              ElementsAre(TaskState::kCreated, TaskState::kQueued,
                          TaskState::kProcessing, TaskState::kDone));
  EXPECT_EQ(fake_audio_executor_ptr->reset_calls_, 1);
}

TEST_P(ExecutionManagerTest,
       AddPrefillTaskPropagatesAudioExecutorResetErrorOnInputAudioEnd) {
  auto fake_llm_executor = std::make_unique<FakeLlmExecutor>(
      kVocabSize,
      std::vector<std::vector<int>>{{1, 2, 3, ExecutorAudioData::kEndToken}},
      std::vector<std::vector<int>>{{4}, {5}, {6}});

  ASSERT_OK_AND_ASSIGN(auto* settings,
                       fake_llm_executor->GetMutableExecutorSettings());
  EXPECT_OK(settings->SetBackend(Backend::GPU_ARTISAN));

  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create("test_model_path_2"));
  ASSERT_OK_AND_ASSIGN(
      auto audio_settings,
      AudioExecutorSettings::CreateDefault(
          model_assets, 128, Backend::GPU_ARTISAN, Backend::GPU_ARTISAN));

  class FailingResetFakeAudioExecutor : public AudioExecutor {
   public:
    absl::StatusOr<ExecutorAudioData> Encode(
        const TensorBuffer& spectrogram_tensor) override {
      return absl::UnimplementedError("Encode not used");
    }

    absl::StatusOr<ExecutorAudioData> Flush() override {
      ExecutorAudioData data;
      data.SetValidTokens(0);
      return data;
    }

    absl::Status Reset() override {
      ++reset_calls_;
      return absl::InternalError("Audio reset failed.");
    }

    int reset_calls_ = 0;
  };

  auto fake_audio_executor = std::make_unique<FailingResetFakeAudioExecutor>();
  auto* fake_audio_executor_ptr = fake_audio_executor.get();

  CreateExecutionManager(
      std::move(fake_llm_executor),
      std::make_unique<AudioExecutorSettings>(std::move(audio_settings)),
      std::move(fake_audio_executor));

  ASSERT_OK_AND_ASSIGN(auto session_config, CreateDefaultSessionConfig());
  session_config.SetAudioModalityEnabled(true);
  ASSERT_OK_AND_ASSIGN(const SessionId session_id,
                       execution_manager_->RegisterNewSession(session_config));

  std::vector<InputData> inputs;
  ASSERT_OK_AND_ASSIGN(auto input_text,
                       tokenizer_->TokenIdsToTensorBuffer({1, 2, 3}));
  inputs.push_back(InputText(std::move(input_text)));
  inputs.push_back(InputAudioEnd());

  std::vector<TaskState> task_states;
  ASSERT_OK_AND_ASSIGN(const TaskId task_id,
                       execution_manager_->GetNewTaskId());
  ASSERT_OK(execution_manager_->AddPrefillTask(
      session_id, task_id, std::move(inputs), {},
      std::make_shared<std::atomic<bool>>(false),
      [&task_states](absl::StatusOr<Responses> responses) {
        if (!responses.ok()) {
          ASSERT_THAT(responses,
                      testing::status::StatusIs(absl::StatusCode::kInternal,
                                                "Audio reset failed."));
          task_states.push_back(TaskState::kFailed);
        } else {
          ASSERT_OK(responses);
          task_states.push_back(responses->GetTaskState());
        }
      }));

  EXPECT_OK(execution_manager_->WaitUntilDone(task_id, absl::Seconds(3)));
  EXPECT_THAT(task_states,
              ElementsAre(TaskState::kCreated, TaskState::kQueued,
                          TaskState::kProcessing, TaskState::kFailed));
  EXPECT_EQ(fake_audio_executor_ptr->reset_calls_, 1);
}

TEST_P(ExecutionManagerTest, EncodeAudioWithSessionInfo) {
  auto fake_llm_executor = CreateDefaultFakeLlmExecutor();

  ASSERT_OK_AND_ASSIGN(auto* settings,
                       fake_llm_executor->GetMutableExecutorSettings());
  EXPECT_OK(settings->SetBackend(Backend::GPU_ARTISAN));

  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create("test_model_path_2"));
  ASSERT_OK_AND_ASSIGN(
      auto audio_settings,
      AudioExecutorSettings::CreateDefault(
          model_assets, 128, Backend::GPU_ARTISAN, Backend::GPU_ARTISAN));

  auto fake_audio_executor = std::make_unique<FakeAudioExecutor>();
  auto* fake_audio_executor_ptr = fake_audio_executor.get();

  CreateExecutionManager(
      std::move(fake_llm_executor),
      std::make_unique<AudioExecutorSettings>(std::move(audio_settings)),
      std::move(fake_audio_executor));

  ASSERT_OK_AND_ASSIGN(auto session_config, CreateDefaultSessionConfig());
  ASSERT_OK_AND_ASSIGN(const SessionId session_id,
                       execution_manager_->RegisterNewSession(session_config));
  ASSERT_OK_AND_ASSIGN(auto session_info,
                       execution_manager_->GetSessionInfo(session_id));

  const std::vector<float> kSpectrogramData = {1.0f, 2.0f, 3.0f, 4.0f};
  auto tensor_or = CopyToTensorBuffer<float>(kSpectrogramData, {1, 4});
  ASSERT_TRUE(tensor_or.HasValue());

  ASSERT_OK_AND_ASSIGN(auto audio_data, execution_manager_->EncodeAudio(
                                            *session_info, *tensor_or));
  EXPECT_EQ(audio_data.GetValidTokens(), 4);

  EXPECT_OK(execution_manager_->ResetAudio(*session_info));
  EXPECT_TRUE(fake_audio_executor_ptr->reset_called_);

  ASSERT_OK_AND_ASSIGN(auto flush_data,
                       execution_manager_->FlushAudio(*session_info));
  EXPECT_EQ(flush_data.GetValidTokens(), 4);
  EXPECT_TRUE(fake_audio_executor_ptr->flush_called_);
}

TEST_P(ExecutionManagerTest, EncodeAudioWithNonStreamingAudioExecutor) {
  auto fake_llm_executor = CreateDefaultFakeLlmExecutor();

  ASSERT_OK_AND_ASSIGN(auto* settings,
                       fake_llm_executor->GetMutableExecutorSettings());
  EXPECT_OK(settings->SetBackend(Backend::GPU_ARTISAN));

  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create("test_model_path_2"));
  ASSERT_OK_AND_ASSIGN(
      auto audio_settings,
      AudioExecutorSettings::CreateDefault(
          model_assets, 128, Backend::GPU_ARTISAN, Backend::GPU_ARTISAN));

  auto fake_audio_executor = std::make_unique<NonStreamingFakeAudioExecutor>();
  auto* fake_audio_executor_ptr = fake_audio_executor.get();

  CreateExecutionManager(
      std::move(fake_llm_executor),
      std::make_unique<AudioExecutorSettings>(std::move(audio_settings)),
      std::move(fake_audio_executor));

  ASSERT_OK_AND_ASSIGN(auto session_config, CreateDefaultSessionConfig());
  ASSERT_OK_AND_ASSIGN(const SessionId session_id,
                       execution_manager_->RegisterNewSession(session_config));
  ASSERT_OK_AND_ASSIGN(auto session_info,
                       execution_manager_->GetSessionInfo(session_id));

  const std::vector<float> kSpectrogramData = {1.0f, 2.0f, 3.0f, 4.0f};
  auto tensor_or = CopyToTensorBuffer<float>(kSpectrogramData, {1, 4});
  ASSERT_TRUE(tensor_or.HasValue());

  ASSERT_OK_AND_ASSIGN(auto audio_data, execution_manager_->EncodeAudio(
                                            *session_info, *tensor_or));
  EXPECT_EQ(audio_data.GetValidTokens(), 4);

  EXPECT_OK(execution_manager_->ResetAudio(*session_info));
  EXPECT_TRUE(fake_audio_executor_ptr->reset_called_);

  ASSERT_OK_AND_ASSIGN(auto flush_data,
                       execution_manager_->FlushAudio(*session_info));
  EXPECT_EQ(flush_data.GetValidTokens(), 0);
  EXPECT_TRUE(fake_audio_executor_ptr->flush_called_);
}

TEST_P(ExecutionManagerTest, EncodeAudioDoesNotEagerlyCloneAudioContext) {
  auto fake_llm_executor = CreateDefaultFakeLlmExecutor();
  ASSERT_OK_AND_ASSIGN(auto* settings,
                       fake_llm_executor->GetMutableExecutorSettings());
  EXPECT_OK(settings->SetBackend(Backend::GPU_ARTISAN));
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create("test_model_path_2"));
  ASSERT_OK_AND_ASSIGN(
      auto audio_settings,
      AudioExecutorSettings::CreateDefault(
          model_assets, 128, Backend::GPU_ARTISAN, Backend::GPU_ARTISAN));
  auto fake_audio_executor = std::make_unique<StatefulFakeAudioExecutor>();
  auto* fake_audio_executor_ptr = fake_audio_executor.get();
  CreateExecutionManager(
      std::move(fake_llm_executor),
      std::make_unique<AudioExecutorSettings>(std::move(audio_settings)),
      std::move(fake_audio_executor));

  ASSERT_OK_AND_ASSIGN(auto session_config, CreateDefaultSessionConfig());
  session_config.SetAudioModalityEnabled(true);
  ASSERT_OK_AND_ASSIGN(const SessionId session_id,
                       execution_manager_->RegisterNewSession(session_config));
  ASSERT_OK_AND_ASSIGN(auto session_info,
                       execution_manager_->GetSessionInfo(session_id));

  const std::vector<float> kSpectrogramData = {1.0f, 2.0f, 3.0f, 4.0f};
  auto tensor_or = CopyToTensorBuffer<float>(kSpectrogramData, {1, 4});
  ASSERT_TRUE(tensor_or.HasValue());
  EXPECT_EQ(fake_audio_executor_ptr->num_create_new_context_calls_, 1);
  ASSERT_OK(execution_manager_->EncodeAudio(*session_info, *tensor_or));
  ASSERT_OK(execution_manager_->EncodeAudio(*session_info, *tensor_or));
  ASSERT_OK(execution_manager_->FlushAudio(*session_info));

  // Encoding, flushing, and resetting must not clone or recreate the audio
  // context in the handler; the executor keeps the live state until another
  // context handler needs it.
  EXPECT_EQ(fake_audio_executor_ptr->num_clone_context_calls_, 0);
  EXPECT_EQ(fake_audio_executor_ptr->num_create_new_context_calls_, 1);
  EXPECT_EQ(fake_audio_executor_ptr->num_encoded_chunks_, 2);

  ASSERT_OK(execution_manager_->ResetAudio(*session_info));
  EXPECT_EQ(fake_audio_executor_ptr->num_clone_context_calls_, 0);
  EXPECT_EQ(fake_audio_executor_ptr->num_create_new_context_calls_, 1);
  EXPECT_EQ(fake_audio_executor_ptr->num_encoded_chunks_, 0);
}

TEST_P(ExecutionManagerTest, CloneSessionCapturesLiveAudioContext) {
  auto fake_llm_executor = std::make_unique<ContextSwitchingFakeLlmExecutor>(
      kVocabSize, std::vector<std::vector<int>>{{1, 2, 3}},
      std::vector<std::vector<int>>{{4}, {5}, {6}});
  ASSERT_OK_AND_ASSIGN(auto* settings,
                       fake_llm_executor->GetMutableExecutorSettings());
  EXPECT_OK(settings->SetBackend(Backend::GPU_ARTISAN));
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create("test_model_path_2"));
  ASSERT_OK_AND_ASSIGN(
      auto audio_settings,
      AudioExecutorSettings::CreateDefault(
          model_assets, 128, Backend::GPU_ARTISAN, Backend::GPU_ARTISAN));
  auto fake_audio_executor = std::make_unique<StatefulFakeAudioExecutor>();
  auto* fake_audio_executor_ptr = fake_audio_executor.get();
  CreateExecutionManager(
      std::move(fake_llm_executor),
      std::make_unique<AudioExecutorSettings>(std::move(audio_settings)),
      std::move(fake_audio_executor));

  ASSERT_OK_AND_ASSIGN(auto session_config, CreateDefaultSessionConfig());
  session_config.SetAudioModalityEnabled(true);
  ASSERT_OK_AND_ASSIGN(const SessionId session_id,
                       execution_manager_->RegisterNewSession(session_config));
  ASSERT_OK_AND_ASSIGN(auto session_info,
                       execution_manager_->GetSessionInfo(session_id));

  const std::vector<float> kSpectrogramData = {1.0f, 2.0f, 3.0f, 4.0f};
  auto tensor_or = CopyToTensorBuffer<float>(kSpectrogramData, {1, 4});
  ASSERT_TRUE(tensor_or.HasValue());
  ASSERT_OK(execution_manager_->EncodeAudio(*session_info, *tensor_or));
  ASSERT_OK(execution_manager_->EncodeAudio(*session_info, *tensor_or));

  // Clone the session while its audio state only lives in the executor.
  ASSERT_OK_AND_ASSIGN(const SessionId cloned_session_id,
                       execution_manager_->RegisterNewSession(session_config));
  ASSERT_OK_AND_ASSIGN(const TaskId task_id,
                       execution_manager_->GetNewTaskId());
  absl::Status clone_status;
  ASSERT_OK(execution_manager_->AddCloneSessionTask(
      session_id, task_id, {}, cloned_session_id,
      std::make_shared<std::atomic<bool>>(false),
      [&clone_status](absl::StatusOr<Responses> responses) {
        clone_status = responses.status();
      }));
  EXPECT_OK(execution_manager_->WaitUntilDone(task_id, absl::Seconds(3)));
  ASSERT_OK(clone_status);

  ASSERT_OK_AND_ASSIGN(auto cloned_session_info,
                       execution_manager_->GetSessionInfo(cloned_session_id));
  ASSERT_TRUE(cloned_session_info->context_handler->HasAudioContext());
  EXPECT_EQ(static_cast<const StatefulFakeAudioContext&>(
                cloned_session_info->context_handler->GetAudioContext())
                .num_encoded_chunks(),
            2);

  // Switching to the cloned session restores the cloned audio state, and
  // switching back to the original session restores its live state.
  ASSERT_OK(execution_manager_->EncodeAudio(*cloned_session_info, *tensor_or));
  EXPECT_EQ(fake_audio_executor_ptr->num_encoded_chunks_, 3);
  ASSERT_OK(execution_manager_->EncodeAudio(*session_info, *tensor_or));
  EXPECT_EQ(fake_audio_executor_ptr->num_encoded_chunks_, 3);
  ASSERT_OK(execution_manager_->EncodeAudio(*cloned_session_info, *tensor_or));
  EXPECT_EQ(fake_audio_executor_ptr->num_encoded_chunks_, 4);

  // Resetting a session's audio state in the executor is also lazily saved
  // when switching to another session and restored when switching back.
  const int create_calls_before_reset =
      fake_audio_executor_ptr->num_create_new_context_calls_;
  ASSERT_OK(execution_manager_->ResetAudio(*session_info));
  EXPECT_EQ(fake_audio_executor_ptr->num_create_new_context_calls_,
            create_calls_before_reset);
  EXPECT_EQ(fake_audio_executor_ptr->num_encoded_chunks_, 0);
  ASSERT_OK(execution_manager_->EncodeAudio(*cloned_session_info, *tensor_or));
  EXPECT_EQ(fake_audio_executor_ptr->num_encoded_chunks_, 5);
  ASSERT_OK(execution_manager_->EncodeAudio(*session_info, *tensor_or));
  EXPECT_EQ(fake_audio_executor_ptr->num_encoded_chunks_, 1);
}

TEST_P(ExecutionManagerTest,
       SaveAndRestoreAudioContextPreservesStreamingStateAcrossReset) {
  auto fake_llm_executor = std::make_unique<FakeLlmExecutor>(
      kVocabSize,
      std::vector<std::vector<int>>{
          {ExecutorAudioData::kSpecialToken, ExecutorAudioData::kSpecialToken,
           ExecutorAudioData::kSpecialToken, ExecutorAudioData::kSpecialToken,
           ExecutorAudioData::kEndToken}},
      std::vector<std::vector<int>>{{4}, {5}, {6}});
  ASSERT_OK_AND_ASSIGN(auto* settings,
                       fake_llm_executor->GetMutableExecutorSettings());
  EXPECT_OK(settings->SetBackend(Backend::GPU_ARTISAN));
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create("test_model_path_2"));
  ASSERT_OK_AND_ASSIGN(
      auto audio_settings,
      AudioExecutorSettings::CreateDefault(
          model_assets, 128, Backend::GPU_ARTISAN, Backend::GPU_ARTISAN));
  auto fake_audio_executor = std::make_unique<StatefulFakeAudioExecutor>();
  auto* fake_audio_executor_ptr = fake_audio_executor.get();
  CreateExecutionManager(
      std::move(fake_llm_executor),
      std::make_unique<AudioExecutorSettings>(std::move(audio_settings)),
      std::move(fake_audio_executor));

  ASSERT_OK_AND_ASSIGN(auto session_config, CreateDefaultSessionConfig());
  session_config.SetAudioModalityEnabled(true);
  ASSERT_OK_AND_ASSIGN(const SessionId session_id,
                       execution_manager_->RegisterNewSession(session_config));
  ASSERT_OK_AND_ASSIGN(auto session_info,
                       execution_manager_->GetSessionInfo(session_id));

  auto spectrogram =
      CopyToTensorBuffer<float>({1.0f, 2.0f, 3.0f, 4.0f}, {1, 4});
  ASSERT_TRUE(spectrogram.HasValue());
  ASSERT_OK(execution_manager_->EncodeAudio(*session_info, *spectrogram));
  ASSERT_OK(execution_manager_->EncodeAudio(*session_info, *spectrogram));
  EXPECT_EQ(fake_audio_executor_ptr->num_encoded_chunks_, 2);

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<AudioContext> saved_audio_context,
                       execution_manager_->SaveAudioContext(*session_info));
  ASSERT_NE(saved_audio_context, nullptr);

  // Simulate speculative InputAudioEnd prefill, which flushes and resets the
  // streaming audio executor.
  std::vector<InputData> end_inputs;
  end_inputs.push_back(InputAudioEnd());
  ASSERT_OK_AND_ASSIGN(const TaskId task_id,
                       execution_manager_->GetNewTaskId());
  ASSERT_OK(execution_manager_->AddPrefillTask(
      session_id, task_id, std::move(end_inputs), {},
      std::make_shared<std::atomic<bool>>(false), nullptr));
  EXPECT_OK(execution_manager_->WaitUntilDone(task_id, absl::Seconds(3)));
  EXPECT_EQ(fake_audio_executor_ptr->num_encoded_chunks_, 0);

  // Restore the saved audio context and verify subsequent chunks continue from
  // the checkpointed count, and that the saved context remains reusable across
  // multiple rewinds.
  ASSERT_OK(execution_manager_->RestoreAudioContext(*session_info,
                                                    *saved_audio_context));
  EXPECT_EQ(fake_audio_executor_ptr->num_encoded_chunks_, 2);
  ASSERT_OK(execution_manager_->EncodeAudio(*session_info, *spectrogram));
  EXPECT_EQ(fake_audio_executor_ptr->num_encoded_chunks_, 3);

  ASSERT_OK(execution_manager_->RestoreAudioContext(*session_info,
                                                    *saved_audio_context));
  EXPECT_EQ(fake_audio_executor_ptr->num_encoded_chunks_, 2);
}

TEST_P(ExecutionManagerTest,
       SaveAudioContextReturnsNullptrWithoutStreamingAudioContext) {
  CreateExecutionManager(CreateDefaultFakeLlmExecutor());
  ASSERT_OK_AND_ASSIGN(auto session_config, CreateDefaultSessionConfig());
  ASSERT_OK_AND_ASSIGN(const SessionId session_id,
                       execution_manager_->RegisterNewSession(session_config));
  ASSERT_OK_AND_ASSIGN(auto session_info,
                       execution_manager_->GetSessionInfo(session_id));

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<AudioContext> saved_audio_context,
                       execution_manager_->SaveAudioContext(*session_info));
  EXPECT_EQ(saved_audio_context, nullptr);
}

TEST_P(ExecutionManagerTest, AddPrefillTaskWithPrecomputedAudioEmbeddings) {
  const std::vector<float> kEmbeddingData = {1.0f, 2.0f, 3.0f, 4.0f,
                                             5.0f, 6.0f, 7.0f, 8.0f};
  auto embedding_tensor_or = CopyToTensorBuffer<float>(kEmbeddingData, {2, 4});
  ASSERT_TRUE(embedding_tensor_or.HasValue());
  auto embedding_tensor = std::move(*embedding_tensor_or);

  auto fake_llm_executor = std::make_unique<FakeLlmExecutor>(
      kVocabSize,
      std::vector<std::vector<int>>{{1, 2, 3, ExecutorAudioData::kSpecialToken,
                                     ExecutorAudioData::kSpecialToken,
                                     ExecutorAudioData::kEndToken}},
      std::vector<std::vector<int>>{{4}, {5}, {6}},
      /*batch_size=*/1,
      /*projected_audio_embedding=*/kEmbeddingData);

  ASSERT_OK_AND_ASSIGN(auto* settings,
                       fake_llm_executor->GetMutableExecutorSettings());
  EXPECT_OK(settings->SetBackend(Backend::GPU_ARTISAN));

  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create("test_model_path_2"));
  ASSERT_OK_AND_ASSIGN(
      auto audio_settings,
      AudioExecutorSettings::CreateDefault(
          model_assets, 128, Backend::GPU_ARTISAN, Backend::GPU_ARTISAN));

  CreateExecutionManager(
      std::move(fake_llm_executor),
      std::make_unique<AudioExecutorSettings>(std::move(audio_settings)),
      std::make_unique<FailingAudioExecutor>());

  ASSERT_OK_AND_ASSIGN(auto session_config, CreateDefaultSessionConfig());
  session_config.SetAudioModalityEnabled(true);

  ASSERT_OK_AND_ASSIGN(const SessionId session_id,
                       execution_manager_->RegisterNewSession(session_config));

  std::vector<InputData> inputs;
  ASSERT_OK_AND_ASSIGN(auto input_text,
                       tokenizer_->TokenIdsToTensorBuffer({1, 2, 3}));
  inputs.push_back(InputText(std::move(input_text)));
  inputs.push_back(
      InputAudio(std::move(embedding_tensor), /*is_embeddings=*/true));
  inputs.push_back(InputAudioEnd());

  std::vector<TaskState> task_states;
  absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback =
      [&task_states](absl::StatusOr<Responses> responses) {
        ASSERT_OK(responses);
        task_states.push_back(responses->GetTaskState());
      };

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
      /*dependency_task_ids=*/{}, RepetitionPenaltyConfig::Default(),
      NoRepeatNgramConfig::Default(), SuppressTokensConfig::Default(),
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

TEST_P(ExecutionManagerTest, AddDecodeTaskVerifiesTheDraft) {
  std::unique_ptr<FakeLlmExecutor> executor = CreateDefaultFakeLlmExecutor();
  executor->SetDraftVerificationSupport(absl::OkStatus());
  const FakeLlmExecutor* fake_executor = executor.get();
  CreateExecutionManager(std::move(executor));
  ASSERT_OK_AND_ASSIGN(auto session_config, CreateDefaultSessionConfig());
  ASSERT_OK_AND_ASSIGN(const SessionId session_id,
                       execution_manager_->RegisterNewSession(session_config));

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

  std::string response_text;
  absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback =
      [&response_text](absl::StatusOr<Responses> responses) {
        ASSERT_OK(responses);
        if (!responses->GetTexts().empty()) {
          response_text += responses->GetTexts()[0];
        }
      };
  ASSERT_OK_AND_ASSIGN(const TaskId decode_task_id,
                       execution_manager_->GetNewTaskId());
  ASSERT_OK(execution_manager_->AddDecodeTask(
      session_id, decode_task_id,
      /*dependency_task_ids=*/{}, RepetitionPenaltyConfig::Default(),
      NoRepeatNgramConfig::Default(), SuppressTokensConfig::Default(),
      /*constraint=*/nullptr,
      /*cancelled=*/std::make_shared<std::atomic<bool>>(false),
      std::move(callback),
      /*max_output_tokens=*/std::numeric_limits<int>::max(),
      /*thinking_token_budget=*/std::nullopt,
      /*thinking_start_token_ids=*/{}, /*thinking_end_token_ids=*/{},
      /*draft_token_ids=*/{4}));
  EXPECT_OK(
      execution_manager_->WaitUntilDone(decode_task_id, absl::Seconds(3)));

  // The same response as without the draft.
  EXPECT_EQ(response_text, "45");
  EXPECT_THAT(fake_executor->verified_drafts(), ElementsAre(ElementsAre(4)));
}

TEST_P(ExecutionManagerTest, AddDecodeTaskWithExternalSampler) {
  std::vector<std::vector<int>> prefill_tokens = {{1, 2, 3}, {6}};
  std::vector<std::vector<int>> decode_tokens = {{4}, {5}, {6}};

  CreateExecutionManager(std::make_unique<FakeLlmExecutor>(
      kVocabSize,
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
      /*dependency_task_ids=*/{}, RepetitionPenaltyConfig::Default(),
      NoRepeatNgramConfig::Default(), SuppressTokensConfig::Default(),
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

TEST_P(ExecutionManagerTest, ExternalSamplerDrawsFromTheSessionRandomState) {
  constexpr int kSeed = 42;
  CreateExecutionManager(std::make_unique<ContextSwitchingFakeLlmExecutor>(
      kVocabSize,
      /*prefill_tokens=*/std::vector<std::vector<int>>{{1, 2, 3}},
      /*decode_tokens=*/std::vector<std::vector<int>>{{4}, {5}, {6}}));

  // k > 1 makes every decode step draw uniforms from the session's random
  // state.
  ASSERT_OK_AND_ASSIGN(auto session_config, CreateDefaultSessionConfig(
                                                /*use_external_sampler=*/true));
  session_config.GetMutableSamplerParams().set_type(
      proto::SamplerParameters::TOP_P);
  session_config.GetMutableSamplerParams().set_k(2);
  session_config.GetMutableSamplerParams().set_p(1.0f);
  session_config.GetMutableSamplerParams().set_seed(kSeed);
  ASSERT_OK_AND_ASSIGN(const SessionId session_id,
                       execution_manager_->RegisterNewSession(session_config));
  ASSERT_OK_AND_ASSIGN(auto session_info,
                       execution_manager_->GetSessionInfo(session_id));
  EXPECT_EQ(session_info->random_state.sampling,
            std::default_random_engine(kSeed));

  std::vector<InputData> inputs;
  ASSERT_OK_AND_ASSIGN(auto input_text,
                       tokenizer_->TokenIdsToTensorBuffer({1, 2, 3}));
  inputs.push_back(InputText(std::move(input_text)));
  ASSERT_OK_AND_ASSIGN(const TaskId prefill_task_id,
                       execution_manager_->GetNewTaskId());
  absl::Status prefill_status;
  ASSERT_OK(execution_manager_->AddPrefillTask(
      session_id, prefill_task_id, std::move(inputs),
      /*dependency_task_ids=*/{},
      /*cancelled=*/std::make_shared<std::atomic<bool>>(false),
      CaptureTerminalStatusCallback(&prefill_status)));
  ASSERT_OK_AND_ASSIGN(const TaskId decode_task_id,
                       execution_manager_->GetNewTaskId());
  absl::Status decode_status;
  ASSERT_OK(execution_manager_->AddDecodeTask(
      session_id, decode_task_id,
      /*dependency_task_ids=*/{prefill_task_id},
      RepetitionPenaltyConfig::Default(), NoRepeatNgramConfig::Default(),
      SuppressTokensConfig::Default(), /*constraint=*/nullptr,
      /*cancelled=*/std::make_shared<std::atomic<bool>>(false),
      CaptureTerminalStatusCallback(&decode_status)));
  ASSERT_OK(
      execution_manager_->WaitUntilDone(decode_task_id, absl::Seconds(3)));
  ASSERT_OK(prefill_status);
  ASSERT_OK(decode_status);
  EXPECT_NE(session_info->random_state.sampling,
            std::default_random_engine(kSeed));

  // A clone continues from the original session's random state, not from its
  // own registration seed or the original's seed.
  session_config.GetMutableSamplerParams().set_seed(kSeed + 1);
  ASSERT_OK_AND_ASSIGN(const SessionId cloned_session_id,
                       execution_manager_->RegisterNewSession(session_config));
  ASSERT_OK_AND_ASSIGN(const TaskId clone_task_id,
                       execution_manager_->GetNewTaskId());
  absl::Status clone_status;
  ASSERT_OK(execution_manager_->AddCloneSessionTask(
      session_id, clone_task_id, /*dep_tasks=*/{}, cloned_session_id,
      /*cancelled=*/std::make_shared<std::atomic<bool>>(false),
      [&clone_status](absl::StatusOr<Responses> responses) {
        clone_status = responses.status();
      }));
  ASSERT_OK(execution_manager_->WaitUntilDone(clone_task_id, absl::Seconds(3)));
  ASSERT_OK(clone_status);
  ASSERT_OK_AND_ASSIGN(auto cloned_session_info,
                       execution_manager_->GetSessionInfo(cloned_session_id));
  EXPECT_EQ(cloned_session_info->random_state.sampling,
            session_info->random_state.sampling);
}

TEST_P(ExecutionManagerTest, TasksLendTheSessionRandomStateToTheExecutor) {
  auto executor = std::make_unique<RandomStateRecordingFakeLlmExecutor>(
      kVocabSize,
      /*prefill_tokens=*/std::vector<std::vector<int>>{{1, 2, 3}},
      /*decode_tokens=*/std::vector<std::vector<int>>{{4}, {5}, {6}});
  executor->SetDraftVerificationSupport(absl::OkStatus());
  RandomStateRecordingFakeLlmExecutor* executor_ptr = executor.get();
  CreateExecutionManager(std::move(executor));

  ASSERT_OK_AND_ASSIGN(auto session_config, CreateDefaultSessionConfig());
  ASSERT_OK_AND_ASSIGN(const SessionId session_id,
                       execution_manager_->RegisterNewSession(session_config));
  ASSERT_OK_AND_ASSIGN(auto session_info,
                       execution_manager_->GetSessionInfo(session_id));

  std::vector<InputData> inputs;
  ASSERT_OK_AND_ASSIGN(auto input_text,
                       tokenizer_->TokenIdsToTensorBuffer({1, 2, 3}));
  inputs.push_back(InputText(std::move(input_text)));
  ASSERT_OK_AND_ASSIGN(const TaskId prefill_task_id,
                       execution_manager_->GetNewTaskId());
  absl::Status prefill_status;
  ASSERT_OK(execution_manager_->AddPrefillTask(
      session_id, prefill_task_id, std::move(inputs),
      /*dependency_task_ids=*/{},
      /*cancelled=*/std::make_shared<std::atomic<bool>>(false),
      CaptureTerminalStatusCallback(&prefill_status)));
  ASSERT_OK_AND_ASSIGN(const TaskId decode_task_id,
                       execution_manager_->GetNewTaskId());
  absl::Status decode_status;
  ASSERT_OK(execution_manager_->AddDecodeTask(
      session_id, decode_task_id,
      /*dependency_task_ids=*/{prefill_task_id},
      RepetitionPenaltyConfig::Default(), NoRepeatNgramConfig::Default(),
      SuppressTokensConfig::Default(), /*constraint=*/nullptr,
      /*cancelled=*/std::make_shared<std::atomic<bool>>(false),
      CaptureTerminalStatusCallback(&decode_status),
      /*max_output_tokens=*/std::numeric_limits<int>::max(),
      /*thinking_token_budget=*/std::nullopt,
      /*thinking_start_token_ids=*/{}, /*thinking_end_token_ids=*/{},
      /*draft_token_ids=*/{4}));
  ASSERT_OK(
      execution_manager_->WaitUntilDone(decode_task_id, absl::Seconds(3)));
  ASSERT_OK(prefill_status);
  ASSERT_OK(decode_status);

  EXPECT_THAT(executor_ptr->prefill_random_states(),
              Each(&session_info->random_state));
  EXPECT_THAT(executor_ptr->prefill_random_states(), Not(IsEmpty()));
  EXPECT_THAT(executor_ptr->verify_draft_random_states(),
              Each(&session_info->random_state));
  EXPECT_THAT(executor_ptr->verify_draft_random_states(), Not(IsEmpty()));
  EXPECT_THAT(executor_ptr->decode_random_states(),
              Each(&session_info->random_state));
  EXPECT_THAT(executor_ptr->decode_random_states(), Not(IsEmpty()));
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
      /*dependency_task_ids=*/{task_a_id}, RepetitionPenaltyConfig::Default(),
      NoRepeatNgramConfig::Default(), SuppressTokensConfig::Default(),
      /*constraint=*/nullptr,
      /*cancelled=*/std::make_shared<std::atomic<bool>>(false),
      /*callback=*/nullptr));

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
              testing::HasSubstr("Dependency task 99999 is invalid."));
}

TEST_P(ExecutionManagerTest, WaitUntilTaskDoneTimeout) {
  if (GetParam() == ExecutionManagerType::kSerial) {
    // Serial execution is synchronous, so it won't timeout unless the task
    // itself takes longer than the timeout and we have some way to interrupt.
    // But currently AddDecodeTask will block until done.
    // So this test is only meaningful for Threaded.
    GTEST_SKIP() << "Skipping timeout test for SerialExecutionManager";
  }
  auto prefill_tokens = std::vector<std::vector<int>>{};
  auto decode_tokens = std::vector<std::vector<int>>{};
  decode_tokens.push_back({4});
  decode_tokens.push_back({5});
  decode_tokens.push_back({6});
  auto fake_llm_executor = std::make_unique<FakeLlmExecutor>(
      kVocabSize,
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
      /*dependency_task_ids=*/{}, RepetitionPenaltyConfig::Default(),
      NoRepeatNgramConfig::Default(), SuppressTokensConfig::Default(),
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
    GTEST_SKIP() << "Skipping timeout test for SerialExecutionManager";
  }
  auto prefill_tokens = std::vector<std::vector<int>>{};
  auto decode_tokens = std::vector<std::vector<int>>{};
  decode_tokens.push_back({4});
  decode_tokens.push_back({5});
  decode_tokens.push_back({6});
  auto fake_llm_executor = std::make_unique<FakeLlmExecutor>(
      kVocabSize,
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
      /*dependency_task_ids=*/{}, RepetitionPenaltyConfig::Default(),
      NoRepeatNgramConfig::Default(), SuppressTokensConfig::Default(),
      /*constraint=*/nullptr,
      /*cancelled=*/std::make_shared<std::atomic<bool>>(false),
      /*callback=*/nullptr));

  EXPECT_EQ(
      execution_manager_->WaitUntilAllDone(absl::Milliseconds(100)).code(),
      absl::StatusCode::kDeadlineExceeded);

  // Wait for the task to actually finish to avoid use after free.
  EXPECT_OK(execution_manager_->WaitUntilDone(task_id, absl::Seconds(3)));
}

TEST_P(ExecutionManagerTest,
       WaitUntilSessionDoneDoesNotTimeOutWhileDecodeMakesProgress) {
  if (GetParam() == ExecutionManagerType::kSerial) {
    GTEST_SKIP() << "SerialExecutionManager runs the decode task inline.";
  }
  // The decode takes about 1s in total, longer than the 500ms stall timeout,
  // but every 200ms step restarts the stall timeout.
  CreateExecutionManager(std::make_unique<SlowDecodeStepFakeLlmExecutor>(
      kVocabSize, /*prefill_tokens=*/std::vector<std::vector<int>>{{1, 2, 3}},
      /*decode_tokens=*/std::vector<std::vector<int>>{{4}, {5}, {7}, {8}, {6}},
      /*step_delay=*/absl::Milliseconds(200)));
  ASSERT_OK_AND_ASSIGN(auto session_config, CreateDefaultSessionConfig());
  ASSERT_OK_AND_ASSIGN(const SessionId session_id,
                       execution_manager_->RegisterNewSession(session_config));
  ASSERT_OK(AddPrefillAndDecodeTasks(session_id));

  EXPECT_OK(execution_manager_->WaitUntilSessionDone(session_id,
                                                     absl::Milliseconds(500)));
}

TEST_P(ExecutionManagerTest,
       WaitUntilSessionDoneDoesNotTimeOutWhileQueuedBehindAnotherSession) {
  if (GetParam() == ExecutionManagerType::kSerial) {
    GTEST_SKIP() << "SerialExecutionManager runs the tasks inline.";
  }
  // The prefill of `waiting_session_id` is queued behind the decode of
  // `decoding_session_id`, which takes about 1s but progresses every 200ms.
  CreateExecutionManager(std::make_unique<SlowDecodeStepFakeLlmExecutor>(
      kVocabSize,
      /*prefill_tokens=*/std::vector<std::vector<int>>{{1, 2, 3}, {1, 2, 3}},
      /*decode_tokens=*/std::vector<std::vector<int>>{{4}, {5}, {7}, {8}, {6}},
      /*step_delay=*/absl::Milliseconds(200)));
  ASSERT_OK_AND_ASSIGN(auto session_config, CreateDefaultSessionConfig());
  ASSERT_OK_AND_ASSIGN(const SessionId decoding_session_id,
                       execution_manager_->RegisterNewSession(session_config));
  ASSERT_OK_AND_ASSIGN(const SessionId waiting_session_id,
                       execution_manager_->RegisterNewSession(session_config));
  ASSERT_OK(AddPrefillAndDecodeTasks(decoding_session_id));
  // Let the decode start, after the prefill that the fake executor delays by
  // 100ms, so that the prefill below waits about 700ms behind it.
  absl::SleepFor(absl::Milliseconds(400));
  ASSERT_OK(
      AddPrefillAndDecodeTasks(waiting_session_id, /*with_decode=*/false));

  EXPECT_OK(execution_manager_->WaitUntilSessionDone(waiting_session_id,
                                                     absl::Milliseconds(500)));
  EXPECT_OK(execution_manager_->WaitUntilSessionDone(decoding_session_id,
                                                     absl::Seconds(3)));
}

TEST_P(ExecutionManagerTest, WaitUntilSessionDoneTimesOutWhenStalled) {
  if (GetParam() == ExecutionManagerType::kSerial) {
    GTEST_SKIP() << "SerialExecutionManager runs the decode task inline.";
  }
  auto fake_llm_executor = CreateDefaultFakeLlmExecutor();
  // A single decode step that blocks for longer than the stall timeout.
  fake_llm_executor->SetDecodeDelay(absl::Seconds(1));
  CreateExecutionManager(std::move(fake_llm_executor));
  ASSERT_OK_AND_ASSIGN(auto session_config, CreateDefaultSessionConfig());
  ASSERT_OK_AND_ASSIGN(const SessionId session_id,
                       execution_manager_->RegisterNewSession(session_config));
  ASSERT_OK(AddPrefillAndDecodeTasks(session_id));

  EXPECT_THAT(
      execution_manager_->WaitUntilSessionDone(session_id,
                                               absl::Milliseconds(200)),
      testing::status::StatusIs(absl::StatusCode::kDeadlineExceeded,
                                HasSubstr("made no progress for 200ms")));

  // Wait for the decode to actually finish to avoid use after free.
  EXPECT_OK(
      execution_manager_->WaitUntilSessionDone(session_id, absl::Seconds(3)));
}

TEST_P(ExecutionManagerTest,
       WaitUntilSessionDoneDoesNotTimeOutWhilePrefillMakesProgress) {
  if (GetParam() == ExecutionManagerType::kSerial) {
    GTEST_SKIP() << "SerialExecutionManager runs the prefill task inline.";
  }
  // 5 prefill chunks of 200ms each (~1s total) exceed the 500ms stall timeout,
  // but each chunk reports progress and restarts the stall timeout.
  CreateExecutionManager(std::make_unique<SlowPrefillChunkFakeLlmExecutor>(
      kVocabSize, /*prefill_tokens=*/std::vector<std::vector<int>>{{1, 2, 3}},
      /*decode_tokens=*/std::vector<std::vector<int>>{{6}},
      /*num_chunks=*/5, /*chunk_delay=*/absl::Milliseconds(200),
      /*record_progress=*/true));
  ASSERT_OK_AND_ASSIGN(auto session_config, CreateDefaultSessionConfig());
  ASSERT_OK_AND_ASSIGN(const SessionId session_id,
                       execution_manager_->RegisterNewSession(session_config));
  ASSERT_OK(AddPrefillAndDecodeTasks(session_id, /*with_decode=*/false));

  EXPECT_OK(execution_manager_->WaitUntilSessionDone(session_id,
                                                     absl::Milliseconds(500)));
}

TEST_P(ExecutionManagerTest, WaitUntilSessionDoneTimesOutWhenPrefillStalls) {
  if (GetParam() == ExecutionManagerType::kSerial) {
    GTEST_SKIP() << "SerialExecutionManager runs the prefill task inline.";
  }
  // A single prefill chunk that blocks for 1s without reporting progress.
  CreateExecutionManager(std::make_unique<SlowPrefillChunkFakeLlmExecutor>(
      kVocabSize, /*prefill_tokens=*/std::vector<std::vector<int>>{{1, 2, 3}},
      /*decode_tokens=*/std::vector<std::vector<int>>{{6}},
      /*num_chunks=*/1, /*chunk_delay=*/absl::Seconds(1),
      /*record_progress=*/false));
  ASSERT_OK_AND_ASSIGN(auto session_config, CreateDefaultSessionConfig());
  ASSERT_OK_AND_ASSIGN(const SessionId session_id,
                       execution_manager_->RegisterNewSession(session_config));
  ASSERT_OK(AddPrefillAndDecodeTasks(session_id, /*with_decode=*/false));

  EXPECT_THAT(
      execution_manager_->WaitUntilSessionDone(session_id,
                                               absl::Milliseconds(200)),
      testing::status::StatusIs(absl::StatusCode::kDeadlineExceeded,
                                HasSubstr("made no progress for 200ms")));

  // Wait for the prefill to actually finish to avoid use after free.
  EXPECT_OK(
      execution_manager_->WaitUntilSessionDone(session_id, absl::Seconds(3)));
}

TEST_P(ExecutionManagerTest, TaskReturnsError) {
  auto prefill_tokens = std::vector<std::vector<int>>{};
  auto decode_tokens = std::vector<std::vector<int>>{};
  prefill_tokens.push_back({1, 2, 3});
  auto fake_llm_executor = std::make_unique<FakeLlmExecutor>(
      kVocabSize,
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
      kVocabSize,
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
      /*dependency_task_ids=*/{task_a_id}, RepetitionPenaltyConfig::Default(),
      NoRepeatNgramConfig::Default(), SuppressTokensConfig::Default(),
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

TEST_P(ExecutionManagerTest,
       AddDecodeTaskWithRepetitionPenaltyConfigWithInternalSampler) {
  std::vector<std::vector<int>> prefill_tokens = {{1}, {0}};
  std::vector<std::vector<int>> decode_tokens = {{4}, {5}, {5}, {6}};

  auto fake_llm_executor = std::make_unique<FakeLlmExecutor>(
      kVocabSize,
      /*prefill_tokens=*/std::move(prefill_tokens),
      /*decode_tokens=*/std::move(decode_tokens));
  fake_llm_executor->SetDecodeLogitsOptions(
      FakeLlmExecutor::DecodeLogitsOptions{.match_value = 10.0f,
                                           .mismatch_value = -10.0f,
                                           .end_token_id = 6,
                                           .mismatch_end_token_value = 0.0f});

  CreateExecutionManager(std::move(fake_llm_executor));

  ASSERT_OK_AND_ASSIGN(auto session_config, CreateDefaultSessionConfig());
  ASSERT_OK_AND_ASSIGN(const SessionId session_id,
                       execution_manager_->RegisterNewSession(session_config));

  std::vector<InputData> inputs;
  ASSERT_OK_AND_ASSIGN(auto input_text,
                       tokenizer_->TokenIdsToTensorBuffer({1}));
  inputs.push_back(InputText(std::move(input_text)));
  ASSERT_OK_AND_ASSIGN(const TaskId task_a_id,
                       execution_manager_->GetNewTaskId());
  ASSERT_OK(execution_manager_->AddPrefillTask(
      session_id, task_a_id, std::move(inputs),
      /*dependency_task_ids=*/{},
      /*cancelled=*/std::make_shared<std::atomic<bool>>(false),
      /*callback=*/nullptr));

  // Set repetition penalty config to penalize the token "5" so that the
  // output is end token "6".
  RepetitionPenaltyConfig repetition_penalty_config(/*repetition_penalty=*/2.0f,
                                                    /*presence_penalty=*/5.0f,
                                                    /*frequency_penalty=*/1.0f,
                                                    /*window_size=*/5);
  std::vector<std::string> response_texts;
  absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback =
      [&response_texts](absl::StatusOr<Responses> responses) {
        ASSERT_OK(responses);
        if (!responses->GetTexts().empty()) {
          response_texts.push_back(responses->GetTexts()[0]);
        }
      };

  ASSERT_OK_AND_ASSIGN(const TaskId task_b_id,
                       execution_manager_->GetNewTaskId());
  ASSERT_OK(execution_manager_->AddDecodeTask(
      session_id, task_b_id,
      /*dependency_task_ids=*/{task_a_id}, repetition_penalty_config,
      NoRepeatNgramConfig::Default(), SuppressTokensConfig::Default(),
      /*constraint=*/nullptr,
      /*cancelled=*/std::make_shared<std::atomic<bool>>(false),
      std::move(callback)));

  EXPECT_OK(execution_manager_->WaitUntilDone(task_b_id, absl::Seconds(3)));

  EXPECT_THAT(response_texts, ElementsAre("4", "5"));
}

TEST_P(ExecutionManagerTest,
       AddDecodeTaskWithRepetitionPenaltyConfigWithExternalSampler) {
  std::vector<std::vector<int>> prefill_tokens = {{1}, {6}};
  std::vector<std::vector<int>> decode_tokens = {{4}, {5}, {5}, {6}};

  auto fake_llm_executor = std::make_unique<FakeLlmExecutor>(
      kVocabSize,
      /*prefill_tokens=*/std::move(prefill_tokens),
      /*decode_tokens=*/std::move(decode_tokens));
  fake_llm_executor->SetDecodeLogitsOptions(
      FakeLlmExecutor::DecodeLogitsOptions{.match_value = 10.0f,
                                           .mismatch_value = -10.0f,
                                           .end_token_id = 6,
                                           .mismatch_end_token_value = 0.0f});

  CreateExecutionManager(std::move(fake_llm_executor));

  ASSERT_OK_AND_ASSIGN(auto session_config, CreateDefaultSessionConfig(
                                                /*use_external_sampler=*/true));
  ASSERT_OK_AND_ASSIGN(const SessionId session_id,
                       execution_manager_->RegisterNewSession(session_config));

  std::vector<InputData> inputs;
  ASSERT_OK_AND_ASSIGN(auto input_text,
                       tokenizer_->TokenIdsToTensorBuffer({1}));
  inputs.push_back(InputText(std::move(input_text)));
  ASSERT_OK_AND_ASSIGN(const TaskId task_a_id,
                       execution_manager_->GetNewTaskId());
  ASSERT_OK(execution_manager_->AddPrefillTask(
      session_id, task_a_id, std::move(inputs),
      /*dependency_task_ids=*/{},
      /*cancelled=*/std::make_shared<std::atomic<bool>>(false),
      /*callback=*/nullptr));

  // Set repetition penalty config to penalize the token "5" so that the
  // output is end token "6".
  RepetitionPenaltyConfig repetition_penalty_config(/*repetition_penalty=*/2.0f,
                                                    /*presence_penalty=*/5.0f,
                                                    /*frequency_penalty=*/1.0f,
                                                    /*window_size=*/5);
  std::vector<std::string> response_texts;
  absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback =
      [&response_texts](absl::StatusOr<Responses> responses) {
        ASSERT_OK(responses);
        if (!responses->GetTexts().empty()) {
          response_texts.push_back(responses->GetTexts()[0]);
        }
      };

  ASSERT_OK_AND_ASSIGN(const TaskId task_b_id,
                       execution_manager_->GetNewTaskId());
  ASSERT_OK(execution_manager_->AddDecodeTask(
      session_id, task_b_id,
      /*dependency_task_ids=*/{task_a_id}, repetition_penalty_config,
      NoRepeatNgramConfig::Default(), SuppressTokensConfig::Default(),
      /*constraint=*/nullptr,
      /*cancelled=*/std::make_shared<std::atomic<bool>>(false),
      std::move(callback)));

  EXPECT_OK(execution_manager_->WaitUntilDone(task_b_id, absl::Seconds(3)));

  EXPECT_THAT(response_texts, ElementsAre("4", "5"));
}

TEST_P(ExecutionManagerTest,
       AddDecodeTaskWithNoRepeatNgramConfigWithInternalSampler) {
  std::vector<std::vector<int>> prefill_tokens = {{1}, {0}};
  std::vector<std::vector<int>> decode_tokens = {{4}, {5}, {4}, {5}, {6}};

  auto fake_llm_executor = std::make_unique<FakeLlmExecutor>(
      kVocabSize,
      /*prefill_tokens=*/std::move(prefill_tokens),
      /*decode_tokens=*/std::move(decode_tokens));
  fake_llm_executor->SetDecodeLogitsOptions(
      FakeLlmExecutor::DecodeLogitsOptions{.match_value = 10.0f,
                                           .mismatch_value = -10.0f,
                                           .end_token_id = 6,
                                           .mismatch_end_token_value = 0.0f});

  CreateExecutionManager(std::move(fake_llm_executor));

  ASSERT_OK_AND_ASSIGN(auto session_config, CreateDefaultSessionConfig());
  ASSERT_OK_AND_ASSIGN(const SessionId session_id,
                       execution_manager_->RegisterNewSession(session_config));

  std::vector<InputData> inputs;
  ASSERT_OK_AND_ASSIGN(auto input_text,
                       tokenizer_->TokenIdsToTensorBuffer({1}));
  inputs.push_back(InputText(std::move(input_text)));
  ASSERT_OK_AND_ASSIGN(const TaskId task_a_id,
                       execution_manager_->GetNewTaskId());
  ASSERT_OK(execution_manager_->AddPrefillTask(
      session_id, task_a_id, std::move(inputs),
      /*dependency_task_ids=*/{},
      /*cancelled=*/std::make_shared<std::atomic<bool>>(false),
      /*callback=*/nullptr));

  // Set no repeat ngram config to ban the repetition of bigrams (i.e. "4, 5"
  // and "5, 4").
  NoRepeatNgramConfig config(/*no_repeat_ngram_size=*/2, /*window_size=*/5);

  std::vector<std::string> response_texts;
  absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback =
      [&response_texts](absl::StatusOr<Responses> responses) {
        ASSERT_OK(responses);
        if (!responses->GetTexts().empty()) {
          response_texts.push_back(responses->GetTexts()[0]);
        }
      };

  ASSERT_OK_AND_ASSIGN(const TaskId task_b_id,
                       execution_manager_->GetNewTaskId());
  ASSERT_OK(execution_manager_->AddDecodeTask(
      session_id, task_b_id,
      /*dependency_task_ids=*/{task_a_id}, RepetitionPenaltyConfig::Default(),
      config, SuppressTokensConfig::Default(),
      /*constraint=*/nullptr,
      /*cancelled=*/std::make_shared<std::atomic<bool>>(false),
      std::move(callback)));

  EXPECT_OK(execution_manager_->WaitUntilDone(task_b_id, absl::Seconds(3)));

  EXPECT_THAT(response_texts, ElementsAre("4", "5", "4"));
}

TEST_P(ExecutionManagerTest,
       AddDecodeTaskWithNoRepeatNgramConfigWithExternalSampler) {
  std::vector<std::vector<int>> prefill_tokens = {{1}, {6}};
  std::vector<std::vector<int>> decode_tokens = {{4}, {5}, {4}, {5}, {6}};

  auto fake_llm_executor = std::make_unique<FakeLlmExecutor>(
      kVocabSize,
      /*prefill_tokens=*/std::move(prefill_tokens),
      /*decode_tokens=*/std::move(decode_tokens));
  fake_llm_executor->SetDecodeLogitsOptions(
      FakeLlmExecutor::DecodeLogitsOptions{.match_value = 10.0f,
                                           .mismatch_value = -10.0f,
                                           .end_token_id = 6,
                                           .mismatch_end_token_value = 0.0f});

  CreateExecutionManager(std::move(fake_llm_executor));

  ASSERT_OK_AND_ASSIGN(auto session_config, CreateDefaultSessionConfig(
                                                /*use_external_sampler=*/true));
  ASSERT_OK_AND_ASSIGN(const SessionId session_id,
                       execution_manager_->RegisterNewSession(session_config));

  std::vector<InputData> inputs;
  ASSERT_OK_AND_ASSIGN(auto input_text,
                       tokenizer_->TokenIdsToTensorBuffer({1}));
  inputs.push_back(InputText(std::move(input_text)));
  ASSERT_OK_AND_ASSIGN(const TaskId task_a_id,
                       execution_manager_->GetNewTaskId());
  ASSERT_OK(execution_manager_->AddPrefillTask(
      session_id, task_a_id, std::move(inputs),
      /*dependency_task_ids=*/{},
      /*cancelled=*/std::make_shared<std::atomic<bool>>(false),
      /*callback=*/nullptr));

  // Set no repeat ngram config to ban the repetition of bigrams (i.e. "4, 5"
  // and "5, 4").
  NoRepeatNgramConfig config(/*no_repeat_ngram_size=*/2, /*window_size=*/5);

  std::vector<std::string> response_texts;
  absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback =
      [&response_texts](absl::StatusOr<Responses> responses) {
        ASSERT_OK(responses);
        if (!responses->GetTexts().empty()) {
          response_texts.push_back(responses->GetTexts()[0]);
        }
      };

  ASSERT_OK_AND_ASSIGN(const TaskId task_b_id,
                       execution_manager_->GetNewTaskId());
  ASSERT_OK(execution_manager_->AddDecodeTask(
      session_id, task_b_id,
      /*dependency_task_ids=*/{task_a_id}, RepetitionPenaltyConfig::Default(),
      config, SuppressTokensConfig::Default(),
      /*constraint=*/nullptr,
      /*cancelled=*/std::make_shared<std::atomic<bool>>(false),
      std::move(callback)));

  EXPECT_OK(execution_manager_->WaitUntilDone(task_b_id, absl::Seconds(3)));

  EXPECT_THAT(response_texts, ElementsAre("4", "5", "4"));
}

TEST_P(ExecutionManagerTest,
       AddDecodeTaskWithSuppressTokensConfigWithInternalSampler) {
  std::vector<std::vector<int>> prefill_tokens = {{1}, {0}};
  std::vector<std::vector<int>> decode_tokens = {{4}, {4}, {5}, {5}, {6}};

  auto fake_llm_executor = std::make_unique<FakeLlmExecutor>(
      kVocabSize,
      /*prefill_tokens=*/std::move(prefill_tokens),
      /*decode_tokens=*/std::move(decode_tokens));
  fake_llm_executor->SetDecodeLogitsOptions(
      FakeLlmExecutor::DecodeLogitsOptions{.match_value = 10.0f,
                                           .mismatch_value = -10.0f,
                                           .end_token_id = 6,
                                           .mismatch_end_token_value = 0.0f});

  CreateExecutionManager(std::move(fake_llm_executor));

  ASSERT_OK_AND_ASSIGN(auto session_config, CreateDefaultSessionConfig());
  ASSERT_OK_AND_ASSIGN(const SessionId session_id,
                       execution_manager_->RegisterNewSession(session_config));

  std::vector<InputData> inputs;
  ASSERT_OK_AND_ASSIGN(auto input_text,
                       tokenizer_->TokenIdsToTensorBuffer({1}));
  inputs.push_back(InputText(std::move(input_text)));
  ASSERT_OK_AND_ASSIGN(const TaskId task_a_id,
                       execution_manager_->GetNewTaskId());
  ASSERT_OK(execution_manager_->AddPrefillTask(
      session_id, task_a_id, std::move(inputs),
      /*dependency_task_ids=*/{},
      /*cancelled=*/std::make_shared<std::atomic<bool>>(false),
      /*callback=*/nullptr));

  std::vector<std::string> response_texts;
  absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback =
      [&response_texts](absl::StatusOr<Responses> responses) {
        ASSERT_OK(responses);
        if (!responses->GetTexts().empty()) {
          response_texts.push_back(responses->GetTexts()[0]);
        }
      };

  // Suppress token "5".
  ASSERT_OK_AND_ASSIGN(const TaskId task_b_id,
                       execution_manager_->GetNewTaskId());
  ASSERT_OK(execution_manager_->AddDecodeTask(
      session_id, task_b_id,
      /*dependency_task_ids=*/{task_a_id}, RepetitionPenaltyConfig::Default(),
      NoRepeatNgramConfig::Default(),
      SuppressTokensConfig(/*suppress_tokens=*/{
          5,
      }),
      /*constraint=*/nullptr,
      /*cancelled=*/std::make_shared<std::atomic<bool>>(false),
      std::move(callback)));

  EXPECT_OK(execution_manager_->WaitUntilDone(task_b_id, absl::Seconds(3)));

  EXPECT_THAT(response_texts, ElementsAre("4", "4"));
}

TEST_P(ExecutionManagerTest,
       AddDecodeTaskWithSuppressTokensConfigWithExternalSampler) {
  std::vector<std::vector<int>> prefill_tokens = {{1}, {6}};
  std::vector<std::vector<int>> decode_tokens = {{4}, {4}, {5}, {5}, {6}};

  auto fake_llm_executor = std::make_unique<FakeLlmExecutor>(
      kVocabSize,
      /*prefill_tokens=*/std::move(prefill_tokens),
      /*decode_tokens=*/std::move(decode_tokens));
  fake_llm_executor->SetDecodeLogitsOptions(
      FakeLlmExecutor::DecodeLogitsOptions{.match_value = 10.0f,
                                           .mismatch_value = -10.0f,
                                           .end_token_id = 6,
                                           .mismatch_end_token_value = 0.0f});

  CreateExecutionManager(std::move(fake_llm_executor));

  ASSERT_OK_AND_ASSIGN(auto session_config, CreateDefaultSessionConfig(
                                                /*use_external_sampler=*/true));
  ASSERT_OK_AND_ASSIGN(const SessionId session_id,
                       execution_manager_->RegisterNewSession(session_config));

  std::vector<InputData> inputs;
  ASSERT_OK_AND_ASSIGN(auto input_text,
                       tokenizer_->TokenIdsToTensorBuffer({1}));
  inputs.push_back(InputText(std::move(input_text)));
  ASSERT_OK_AND_ASSIGN(const TaskId task_a_id,
                       execution_manager_->GetNewTaskId());
  ASSERT_OK(execution_manager_->AddPrefillTask(
      session_id, task_a_id, std::move(inputs),
      /*dependency_task_ids=*/{},
      /*cancelled=*/std::make_shared<std::atomic<bool>>(false),
      /*callback=*/nullptr));

  std::vector<std::string> response_texts;
  absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback =
      [&response_texts](absl::StatusOr<Responses> responses) {
        ASSERT_OK(responses);
        if (!responses->GetTexts().empty()) {
          response_texts.push_back(responses->GetTexts()[0]);
        }
      };

  // Suppress token "5".
  ASSERT_OK_AND_ASSIGN(const TaskId task_b_id,
                       execution_manager_->GetNewTaskId());
  ASSERT_OK(execution_manager_->AddDecodeTask(
      session_id, task_b_id,
      /*dependency_task_ids=*/{task_a_id}, RepetitionPenaltyConfig::Default(),
      NoRepeatNgramConfig::Default(),
      SuppressTokensConfig(/*suppress_tokens=*/{
          5,
      }),
      /*constraint=*/nullptr,
      /*cancelled=*/std::make_shared<std::atomic<bool>>(false),
      std::move(callback)));

  EXPECT_OK(execution_manager_->WaitUntilDone(task_b_id, absl::Seconds(3)));

  EXPECT_THAT(response_texts, ElementsAre("4", "4"));
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
  // Fake constraint that expects "4".
  std::vector<int> expected_token_ids = {4, 0};
  auto constraint = FakeConstraint(expected_token_ids, kVocabSize);
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
      /*dependency_task_ids=*/{task_a_id}, RepetitionPenaltyConfig::Default(),
      NoRepeatNgramConfig::Default(), SuppressTokensConfig::Default(),
      decode_config.GetConstraint(),
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
      kVocabSize,
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
  // Fake constraint that expects "4".
  std::vector<int> expected_token_ids = {4, 0};
  auto constraint = FakeConstraint(expected_token_ids, kVocabSize);
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
      /*dependency_task_ids=*/{task_a_id}, RepetitionPenaltyConfig::Default(),
      NoRepeatNgramConfig::Default(), SuppressTokensConfig::Default(),
      decode_config.GetConstraint(),
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

TEST_P(ExecutionManagerTest, AddTextScoringTaskStepProgression) {
  CreateExecutionManager(CreateDefaultFakeLlmExecutor());
  ASSERT_OK_AND_ASSIGN(auto session_config, CreateDefaultSessionConfig());
  ASSERT_OK_AND_ASSIGN(const SessionId session_id,
                       execution_manager_->RegisterNewSession(session_config));
  ASSERT_OK_AND_ASSIGN(auto session_info,
                       execution_manager_->GetSessionInfo(session_id));

  // Prefill 3 tokens.
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

  // After prefill, current step must be 3.
  ASSERT_OK_AND_ASSIGN(int step_after_prefill,
                       execution_manager_->GetCurrentStep(*session_info));
  EXPECT_EQ(step_after_prefill, 3);

  // Score candidate "45" (2 tokens: 4 and 5).
  ASSERT_OK_AND_ASSIGN(const TaskId scoring_task_id,
                       execution_manager_->GetNewTaskId());
  const std::vector<absl::string_view> target_text = {"45"};
  EXPECT_CALL(*tokenizer_, TextToTokenIds("45"))
      .WillOnce(Return(std::vector<int>({4, 5})));

  std::vector<float> scores;
  ASSERT_OK(execution_manager_->AddTextScoringTask(
      session_id, scoring_task_id,
      /*dep_tasks=*/{}, target_text,
      /*store_token_lengths=*/false,
      /*cancelled=*/std::make_shared<std::atomic<bool>>(false),
      [&scores](absl::StatusOr<Responses> responses) {
        ASSERT_OK(responses);
        if (!responses->GetScores().empty()) {
          scores.push_back(responses->GetScores()[0]);
        }
      }));
  EXPECT_OK(
      execution_manager_->WaitUntilDone(scoring_task_id, absl::Seconds(3)));

  ASSERT_EQ(scores.size(), 1);
  EXPECT_FLOAT_EQ(scores[0], 0.0f);

  // After scoring 2 tokens, current step must progress to 5 without
  // double-rewind.
  ASSERT_OK_AND_ASSIGN(int step_after_scoring,
                       execution_manager_->GetCurrentStep(*session_info));
  EXPECT_EQ(step_after_scoring, 5);
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

TEST_P(ExecutionManagerTest, DestructorWaitsForActiveTasks) {
  if (GetParam() == ExecutionManagerType::kSerial) {
    GTEST_SKIP() << "Skipping for SerialExecutionManager as it is synchronous";
  }
  auto fake_llm_executor = CreateDefaultFakeLlmExecutor();
  fake_llm_executor->SetDecodeDelay(absl::Milliseconds(500));
  CreateExecutionManager(std::move(fake_llm_executor));

  ASSERT_OK_AND_ASSIGN(auto session_config, CreateDefaultSessionConfig());
  ASSERT_OK_AND_ASSIGN(const SessionId session_id,
                       execution_manager_->RegisterNewSession(session_config));

  // Add prefill task and wait.
  std::vector<InputData> inputs;
  ASSERT_OK_AND_ASSIGN(auto input_text,
                       tokenizer_->TokenIdsToTensorBuffer({1, 2, 3}));
  inputs.push_back(InputText(std::move(input_text)));
  ASSERT_OK_AND_ASSIGN(const TaskId prefill_task_id,
                       execution_manager_->GetNewTaskId());
  ASSERT_OK(execution_manager_->AddPrefillTask(
      session_id, prefill_task_id, std::move(inputs), {},
      std::make_shared<std::atomic<bool>>(false), nullptr));
  ASSERT_OK(
      execution_manager_->WaitUntilDone(prefill_task_id, absl::Seconds(3)));

  // Now add decode task.
  ASSERT_OK_AND_ASSIGN(const TaskId task_id,
                       execution_manager_->GetNewTaskId());

  auto task_states = std::make_shared<std::vector<TaskState>>();
  auto mutex = std::make_shared<absl::Mutex>();

  ASSERT_OK(execution_manager_->AddDecodeTask(
      session_id, task_id, {}, RepetitionPenaltyConfig::Default(),
      NoRepeatNgramConfig::Default(), SuppressTokensConfig::Default(),
      /*constraint=*/nullptr, std::make_shared<std::atomic<bool>>(false),
      [task_states, mutex](absl::StatusOr<Responses> responses) {
        absl::MutexLock lock(*mutex);
        if (responses.ok()) {
          task_states->push_back(responses->GetTaskState());
        } else {
          task_states->push_back(TaskState::kFailed);
        }
      }));

  execution_manager_.reset();

  absl::MutexLock lock(*mutex);
  EXPECT_THAT(*task_states, testing::Contains(TaskState::kDone));
}

TEST_P(ExecutionManagerTest, ReleaseSessionCleansUpTasksAndQueue) {
  CreateExecutionManager(CreateDefaultFakeLlmExecutor());
  ASSERT_OK_AND_ASSIGN(auto session_config, CreateDefaultSessionConfig());
  ASSERT_OK_AND_ASSIGN(const SessionId session_id,
                       execution_manager_->RegisterNewSession(session_config));

  std::vector<InputData> inputs;
  ASSERT_OK_AND_ASSIGN(auto input_text,
                       tokenizer_->TokenIdsToTensorBuffer({1, 2, 3}));
  inputs.push_back(InputText(std::move(input_text)));
  ASSERT_OK_AND_ASSIGN(const TaskId task_id,
                       execution_manager_->GetNewTaskId());
  ASSERT_OK(execution_manager_->AddPrefillTask(
      session_id, task_id, std::move(inputs), {},
      std::make_shared<std::atomic<bool>>(false), nullptr));

  // Release session. This should clean up the task from task_lookup_ and
  // also ready_queue_ (for SerialExecutionManager).
  ASSERT_OK(execution_manager_->ReleaseSession(session_id));

  // WaitUntilAllDone should complete without error.
  EXPECT_OK(execution_manager_->WaitUntilAllDone(absl::Seconds(1)));
}

TEST_P(ExecutionManagerTest,
       RegisterNewSessionDeduplicatesAudioScopedLoraFiles) {
  const std::filesystem::path temp_dir(::testing::TempDir());
  const std::filesystem::path lora_path_1 = temp_dir / "audio_lora_1.bin";
  const std::filesystem::path lora_path_2 = temp_dir / "audio_lora_2.bin";
  {
    std::ofstream out1(lora_path_1, std::ios::binary);
    out1 << "lora_weights_1";
    std::ofstream out2(lora_path_2, std::ios::binary);
    out2 << "lora_weights_2_different_size";
  }

  ASSERT_OK_AND_ASSIGN(auto scoped_lora_1,
                       ScopedFile::Open(lora_path_1.string()));
  ASSERT_OK_AND_ASSIGN(auto scoped_lora_1_dup, scoped_lora_1.Duplicate());
  ASSERT_OK_AND_ASSIGN(auto scoped_lora_2,
                       ScopedFile::Open(lora_path_2.string()));
  auto shared_lora_1 = std::make_shared<ScopedFile>(std::move(scoped_lora_1));
  auto shared_lora_1_dup =
      std::make_shared<ScopedFile>(std::move(scoped_lora_1_dup));
  auto shared_lora_2 = std::make_shared<ScopedFile>(std::move(scoped_lora_2));

  auto fake_audio_executor = std::make_unique<FakeAudioExecutor>();
  auto* fake_audio_executor_ptr = fake_audio_executor.get();
  CreateExecutionManager(CreateDefaultFakeLlmExecutor(),
                         /*audio_executor_settings=*/nullptr,
                         std::move(fake_audio_executor));

  ASSERT_OK_AND_ASSIGN(auto session_config_1, CreateDefaultSessionConfig());
  session_config_1.SetAudioScopedLoraFile(shared_lora_1);
  ASSERT_OK_AND_ASSIGN(
      const SessionId session_1,
      execution_manager_->RegisterNewSession(session_config_1));

  // Second session sharing the same ScopedFile pointer should reuse lora_id 0
  // without calling LoadLoRA again.
  ASSERT_OK_AND_ASSIGN(
      const SessionId session_2,
      execution_manager_->RegisterNewSession(session_config_1));

  // Third session with a distinct ScopedFile instance pointing to the same
  // underlying file should also deduplicate to lora_id 0.
  ASSERT_OK_AND_ASSIGN(auto session_config_1_dup, CreateDefaultSessionConfig());
  session_config_1_dup.SetAudioScopedLoraFile(shared_lora_1_dup);
  ASSERT_OK_AND_ASSIGN(
      const SessionId session_3,
      execution_manager_->RegisterNewSession(session_config_1_dup));

  // Fourth session with a different underlying file should load a new LoRA
  // with lora_id 1.
  ASSERT_OK_AND_ASSIGN(auto session_config_2, CreateDefaultSessionConfig());
  session_config_2.SetAudioScopedLoraFile(shared_lora_2);
  ASSERT_OK_AND_ASSIGN(
      const SessionId session_4,
      execution_manager_->RegisterNewSession(session_config_2));

  EXPECT_THAT(fake_audio_executor_ptr->loaded_lora_ids_, ElementsAre(0, 1));
  EXPECT_THAT(
      fake_audio_executor_ptr->used_lora_ids_,
      ElementsAre(std::optional<uint32_t>(0), std::optional<uint32_t>(0),
                  std::optional<uint32_t>(0), std::optional<uint32_t>(1)));

  EXPECT_OK(execution_manager_->ReleaseSession(session_1));
  EXPECT_OK(execution_manager_->ReleaseSession(session_2));
  EXPECT_OK(execution_manager_->ReleaseSession(session_3));
  EXPECT_OK(execution_manager_->ReleaseSession(session_4));
}

TEST_P(ExecutionManagerTest,
       RegisterNewSessionRetriesAudioScopedLoraAfterLoadFailure) {
  const std::filesystem::path lora_path =
      std::filesystem::path(::testing::TempDir()) / "audio_lora_retry.bin";
  {
    std::ofstream out(lora_path, std::ios::binary);
    out << "lora_weights_retry";
  }

  ASSERT_OK_AND_ASSIGN(auto scoped_lora, ScopedFile::Open(lora_path.string()));
  auto shared_lora = std::make_shared<ScopedFile>(std::move(scoped_lora));

  auto fake_audio_executor = std::make_unique<FakeAudioExecutor>();
  auto* fake_audio_executor_ptr = fake_audio_executor.get();
  fake_audio_executor_ptr->load_lora_status_ =
      absl::InternalError("Simulated audio LoadLoRA failure");

  CreateExecutionManager(CreateDefaultFakeLlmExecutor(),
                         /*audio_executor_settings=*/nullptr,
                         std::move(fake_audio_executor));

  ASSERT_OK_AND_ASSIGN(auto session_config, CreateDefaultSessionConfig());
  session_config.SetAudioScopedLoraFile(shared_lora);

  // First session registration fails during LoadLoRA.
  EXPECT_THAT(execution_manager_->RegisterNewSession(session_config),
              testing::status::StatusIs(absl::StatusCode::kInternal));
  EXPECT_TRUE(fake_audio_executor_ptr->loaded_lora_ids_.empty());
  EXPECT_TRUE(fake_audio_executor_ptr->used_lora_ids_.empty());

  // Subsequent session with the same ScopedFile must retry LoadLoRA rather
  // than skipping LoadLoRA with a corrupted lora_hash_to_id_ entry.
  fake_audio_executor_ptr->load_lora_status_ = absl::OkStatus();
  ASSERT_OK_AND_ASSIGN(const SessionId session_id,
                       execution_manager_->RegisterNewSession(session_config));
  EXPECT_THAT(fake_audio_executor_ptr->loaded_lora_ids_, ElementsAre(0));
  EXPECT_THAT(fake_audio_executor_ptr->used_lora_ids_,
              ElementsAre(std::optional<uint32_t>(0)));

  EXPECT_OK(execution_manager_->ReleaseSession(session_id));
}

class StatsFakeLlmExecutor : public ContextSwitchingFakeLlmExecutor {
 public:
  using ContextSwitchingFakeLlmExecutor::ContextSwitchingFakeLlmExecutor;

  absl::StatusOr<std::unique_ptr<LlmContext>> CreateNewContext(
      std::optional<uint32_t> lora_id,
      RuntimeConfig runtime_config) const override {
    return std::make_unique<LlmContext>(
        std::make_unique<FakeProcessedContext>(),
        std::make_unique<RuntimeConfig>(std::move(runtime_config)),
        std::make_unique<RuntimeState>());
  }

  absl::StatusOr<std::unique_ptr<LlmContext>> CloneContext() const override {
    return std::make_unique<LlmContext>(
        std::make_unique<FakeProcessedContext>(),
        std::make_unique<RuntimeConfig>(),
        std::make_unique<RuntimeState>(
            RuntimeState{.current_step = GetCurrentStep().value_or(0)}));
  }

  absl::StatusOr<RuntimeState> GetRuntimeState() const override {
    return RuntimeState{.current_step = GetCurrentStep().value_or(0)};
  }

  absl::Status UpdateRuntimeState(const RuntimeState& runtime_state) override {
    return SetCurrentStep(runtime_state.current_step);
  }

  absl::StatusOr<std::vector<std::vector<int>>> Decode(
      const ExecutorDecodeParams& decode_params) override {
    if (std::exchange(cancel_next_step_, false)) {
      return absl::CancelledError("Step cancelled.");
    }
    ABSL_ASSIGN_OR_RETURN(
        auto tokens, ContextSwitchingFakeLlmExecutor::Decode(decode_params));
    cumulative_stats_.AccumulatePlainStep(absl::Milliseconds(3));
    if (on_decode_step_ != nullptr) {
      on_decode_step_(cumulative_stats_.plain_steps());
    }
    return tokens;
  }

  // Like the compiled-model executor, a reset restarts `GetExecutorStats()`
  // but keeps the cumulative stats.
  std::optional<ExecutorStats> GetExecutorStats() const override {
    ExecutorStats stats =
        SubtractExecutorStats(cumulative_stats_, stats_at_reset_);
    if (stats.empty()) {
      return std::nullopt;
    }
    return stats;
  }

  void ResetExecutorStats() override { stats_at_reset_ = cumulative_stats_; }

  std::optional<ExecutorStats> GetCumulativeExecutorStats() const override {
    return cumulative_stats_;
  }

  ExecutorStats cumulative_stats_{.module_name = std::string(kLlmModuleName)};
  ExecutorStats stats_at_reset_{.module_name = std::string(kLlmModuleName)};
  // Called after each decode step with the number of steps since creation.
  absl::AnyInvocable<void(int64_t)> on_decode_step_;
  // Whether the next decode step fails as cancelled before it completes.
  bool cancel_next_step_ = false;
};

TEST_P(ExecutionManagerTest, AddDecodeTaskReportsTheExecutorStatsOfItsTurn) {
  // Base session prefills {1, 2, 3}, then two cloned sessions decode from that
  // shared prefix: `clone_1` stops at `max_output_tokens = 2` (2 steps), and
  // `clone_2` decodes {4, 5, 6} until stop token 6 (3 steps).
  auto executor = std::make_unique<StatsFakeLlmExecutor>(
      kVocabSize, std::vector<std::vector<int>>{{1, 2, 3}},
      std::vector<std::vector<int>>{{4}, {5}, {6}});
  CreateExecutionManager(std::move(executor));

  ASSERT_OK_AND_ASSIGN(auto session_config, CreateDefaultSessionConfig());
  ASSERT_OK_AND_ASSIGN(const SessionId base_session,
                       execution_manager_->RegisterNewSession(session_config));
  ASSERT_OK_AND_ASSIGN(const SessionId clone_1,
                       execution_manager_->RegisterNewSession(session_config));
  ASSERT_OK_AND_ASSIGN(const SessionId clone_2,
                       execution_manager_->RegisterNewSession(session_config));

  std::vector<InputData> inputs;
  ASSERT_OK_AND_ASSIGN(auto input_text,
                       tokenizer_->TokenIdsToTensorBuffer({1, 2, 3}));
  inputs.push_back(InputText(std::move(input_text)));
  ASSERT_OK_AND_ASSIGN(const TaskId prefill_id,
                       execution_manager_->GetNewTaskId());
  ASSERT_OK(execution_manager_->AddPrefillTask(
      base_session, prefill_id, std::move(inputs), {},
      std::make_shared<std::atomic<bool>>(false), nullptr));
  ASSERT_OK(execution_manager_->WaitUntilDone(prefill_id, absl::Seconds(3)));

  for (SessionId clone_id : {clone_1, clone_2}) {
    ASSERT_OK_AND_ASSIGN(const TaskId clone_task_id,
                         execution_manager_->GetNewTaskId());
    ASSERT_OK(execution_manager_->AddCloneSessionTask(
        base_session, clone_task_id, {}, clone_id,
        std::make_shared<std::atomic<bool>>(false), nullptr));
    ASSERT_OK(
        execution_manager_->WaitUntilDone(clone_task_id, absl::Seconds(3)));
  }

  const auto run_decode =
      [&](SessionId session_id,
          int max_output_tokens) -> std::optional<ExecutorStats> {
    std::optional<ExecutorStats> terminal_stats;
    absl::StatusOr<TaskId> decode_id = execution_manager_->GetNewTaskId();
    EXPECT_OK(decode_id);
    if (!decode_id.ok()) {
      return std::nullopt;
    }
    EXPECT_OK(execution_manager_->AddDecodeTask(
        session_id, *decode_id, {}, RepetitionPenaltyConfig::Default(),
        NoRepeatNgramConfig::Default(), SuppressTokensConfig::Default(),
        /*constraint=*/nullptr, std::make_shared<std::atomic<bool>>(false),
        [&terminal_stats](absl::StatusOr<Responses> responses) {
          ASSERT_OK(responses);
          if (IsTaskEndState(responses->GetTaskState())) {
            EXPECT_EQ(responses->GetTaskState(), TaskState::kDone);
            terminal_stats = responses->GetExecutorStats();
          } else {
            EXPECT_FALSE(responses->GetExecutorStats().has_value());
          }
        },
        max_output_tokens));
    EXPECT_OK(execution_manager_->WaitUntilDone(*decode_id, absl::Seconds(3)));
    return terminal_stats;
  };

  std::optional<ExecutorStats> stats_1 =
      run_decode(clone_1, /*max_output_tokens=*/2);
  std::optional<ExecutorStats> stats_2 =
      run_decode(clone_2, std::numeric_limits<int>::max());

  ASSERT_TRUE(stats_1.has_value());
  EXPECT_EQ(stats_1->plain_steps(), 2);
  EXPECT_EQ(stats_1->plain_step_time(), absl::Milliseconds(6));

  ASSERT_TRUE(stats_2.has_value());
  EXPECT_EQ(stats_2->plain_steps(), 3);
  EXPECT_EQ(stats_2->plain_step_time(), absl::Milliseconds(9));

  EXPECT_OK(execution_manager_->ReleaseSession(base_session));
  EXPECT_OK(execution_manager_->ReleaseSession(clone_1));
  EXPECT_OK(execution_manager_->ReleaseSession(clone_2));
}

TEST_P(ExecutionManagerTest,
       AddDecodeTaskReportsTheExecutorStatsOfCancelledTurns) {
  auto executor = std::make_unique<StatsFakeLlmExecutor>(
      kVocabSize, std::vector<std::vector<int>>{{1, 2, 3}},
      // Case 3 decodes {4}, and case 4 decodes {5} and stop token {6}.
      std::vector<std::vector<int>>{{4}, {5}, {6}});
  auto* executor_ptr = executor.get();
  // Steps before the turns must not count in their stats.
  for (int i = 0; i < 7; ++i) {
    executor_ptr->cumulative_stats_.AccumulatePlainStep(absl::Milliseconds(1));
  }
  CreateExecutionManager(std::move(executor));

  ASSERT_OK_AND_ASSIGN(auto session_config, CreateDefaultSessionConfig());
  ASSERT_OK_AND_ASSIGN(const SessionId session_id,
                       execution_manager_->RegisterNewSession(session_config));

  std::vector<InputData> inputs;
  ASSERT_OK_AND_ASSIGN(auto input_text,
                       tokenizer_->TokenIdsToTensorBuffer({1, 2, 3}));
  inputs.push_back(InputText(std::move(input_text)));
  ASSERT_OK_AND_ASSIGN(const TaskId prefill_id,
                       execution_manager_->GetNewTaskId());
  ASSERT_OK(execution_manager_->AddPrefillTask(
      session_id, prefill_id, std::move(inputs), {},
      std::make_shared<std::atomic<bool>>(false), nullptr));
  ASSERT_OK(execution_manager_->WaitUntilDone(prefill_id, absl::Seconds(3)));

  // Case 1: cancelled before the decode starts, so no stats.
  {
    auto cancelled = std::make_shared<std::atomic<bool>>(true);
    TaskState end_state = TaskState::kCreated;
    std::optional<ExecutorStats> end_stats = ExecutorStats{};
    ASSERT_OK_AND_ASSIGN(const TaskId decode_id,
                         execution_manager_->GetNewTaskId());
    ASSERT_OK(execution_manager_->AddDecodeTask(
        session_id, decode_id, {}, RepetitionPenaltyConfig::Default(),
        NoRepeatNgramConfig::Default(), SuppressTokensConfig::Default(),
        /*constraint=*/nullptr, cancelled,
        [&](absl::StatusOr<Responses> responses) {
          ASSERT_OK(responses);
          if (IsTaskEndState(responses->GetTaskState())) {
            end_state = responses->GetTaskState();
            end_stats = responses->GetExecutorStats();
          }
        }));
    EXPECT_OK(execution_manager_->WaitUntilDone(decode_id, absl::Seconds(3)));
    EXPECT_EQ(end_state, TaskState::kCancelled);
    EXPECT_FALSE(end_stats.has_value());
  }

  // Case 2: the first step is cancelled before it completes, so the turn ran
  // no step and has no stats, as in case 1.
  {
    executor_ptr->cancel_next_step_ = true;
    TaskState end_state = TaskState::kCreated;
    std::optional<ExecutorStats> end_stats = ExecutorStats{};
    ASSERT_OK_AND_ASSIGN(const TaskId decode_id,
                         execution_manager_->GetNewTaskId());
    ASSERT_OK(execution_manager_->AddDecodeTask(
        session_id, decode_id, {}, RepetitionPenaltyConfig::Default(),
        NoRepeatNgramConfig::Default(), SuppressTokensConfig::Default(),
        /*constraint=*/nullptr, std::make_shared<std::atomic<bool>>(false),
        [&](absl::StatusOr<Responses> responses) {
          ASSERT_OK(responses);
          if (IsTaskEndState(responses->GetTaskState())) {
            end_state = responses->GetTaskState();
            end_stats = responses->GetExecutorStats();
          }
        }));
    EXPECT_OK(execution_manager_->WaitUntilDone(decode_id, absl::Seconds(3)));
    EXPECT_EQ(end_state, TaskState::kCancelled);
    EXPECT_FALSE(end_stats.has_value());
  }

  // Case 3: cancelled during the first step, so `Tasks::Decode` returns
  // `CancelledError` after 1 step.
  {
    auto cancelled = std::make_shared<std::atomic<bool>>(false);
    executor_ptr->on_decode_step_ = [cancelled](int64_t) {
      cancelled->store(true, std::memory_order_relaxed);
    };
    TaskState end_state = TaskState::kCreated;
    std::optional<ExecutorStats> end_stats;
    ASSERT_OK_AND_ASSIGN(const TaskId decode_id,
                         execution_manager_->GetNewTaskId());
    ASSERT_OK(execution_manager_->AddDecodeTask(
        session_id, decode_id, {}, RepetitionPenaltyConfig::Default(),
        NoRepeatNgramConfig::Default(), SuppressTokensConfig::Default(),
        /*constraint=*/nullptr, cancelled,
        [&](absl::StatusOr<Responses> responses) {
          ASSERT_OK(responses);
          if (IsTaskEndState(responses->GetTaskState())) {
            end_state = responses->GetTaskState();
            end_stats = responses->GetExecutorStats();
          }
        }));
    EXPECT_OK(execution_manager_->WaitUntilDone(decode_id, absl::Seconds(3)));
    EXPECT_EQ(end_state, TaskState::kCancelled);
    ASSERT_TRUE(end_stats.has_value());
    EXPECT_EQ(end_stats->plain_steps(), 1);
  }

  // Case 4: cancelled during the stop token's step (the 10th since creation),
  // so `Tasks::Decode` completes the turn after 2 steps and the manager reports
  // it as cancelled.
  {
    auto cancelled = std::make_shared<std::atomic<bool>>(false);
    executor_ptr->on_decode_step_ = [cancelled](int64_t cumulative_steps) {
      if (cumulative_steps == 10) {
        cancelled->store(true, std::memory_order_relaxed);
      }
    };
    TaskState end_state = TaskState::kCreated;
    std::optional<ExecutorStats> end_stats;
    ASSERT_OK_AND_ASSIGN(const TaskId decode_id,
                         execution_manager_->GetNewTaskId());
    ASSERT_OK(execution_manager_->AddDecodeTask(
        session_id, decode_id, {}, RepetitionPenaltyConfig::Default(),
        NoRepeatNgramConfig::Default(), SuppressTokensConfig::Default(),
        /*constraint=*/nullptr, cancelled,
        [&](absl::StatusOr<Responses> responses) {
          ASSERT_OK(responses);
          if (IsTaskEndState(responses->GetTaskState())) {
            end_state = responses->GetTaskState();
            end_stats = responses->GetExecutorStats();
          }
        }));
    EXPECT_OK(execution_manager_->WaitUntilDone(decode_id, absl::Seconds(3)));
    EXPECT_EQ(end_state, TaskState::kCancelled);
    ASSERT_TRUE(end_stats.has_value());
    EXPECT_EQ(end_stats->plain_steps(), 2);
  }

  EXPECT_OK(execution_manager_->ReleaseSession(session_id));
}

TEST_P(ExecutionManagerTest,
       AddDecodeTaskReportsTheExecutorStatsOfAFirstBenchmarkTurn) {
  auto executor = std::make_unique<StatsFakeLlmExecutor>(
      kVocabSize, std::vector<std::vector<int>>{{1, 2, 3}},
      std::vector<std::vector<int>>{{4}, {5}, {6}});
  auto* executor_ptr = executor.get();
  // Steps before the turn, which the benchmark reset at its start clears from
  // `GetExecutorStats()`.
  for (int i = 0; i < 7; ++i) {
    executor_ptr->cumulative_stats_.AccumulatePlainStep(absl::Milliseconds(1));
  }
  CreateExecutionManager(std::move(executor));

  ASSERT_OK_AND_ASSIGN(auto session_config, CreateDefaultSessionConfig());
  ASSERT_OK_AND_ASSIGN(const SessionId session_id,
                       execution_manager_->RegisterNewSession(
                           session_config, std::make_optional<BenchmarkInfo>(
                                               proto::BenchmarkParams())));

  std::vector<InputData> inputs;
  ASSERT_OK_AND_ASSIGN(auto input_text,
                       tokenizer_->TokenIdsToTensorBuffer({1, 2, 3}));
  inputs.push_back(InputText(std::move(input_text)));
  ASSERT_OK_AND_ASSIGN(const TaskId prefill_id,
                       execution_manager_->GetNewTaskId());
  ASSERT_OK(execution_manager_->AddPrefillTask(
      session_id, prefill_id, std::move(inputs), {},
      std::make_shared<std::atomic<bool>>(false), nullptr));
  ASSERT_OK(execution_manager_->WaitUntilDone(prefill_id, absl::Seconds(3)));

  std::optional<ExecutorStats> terminal_stats;
  ASSERT_OK_AND_ASSIGN(const TaskId decode_id,
                       execution_manager_->GetNewTaskId());
  ASSERT_OK(execution_manager_->AddDecodeTask(
      session_id, decode_id, {}, RepetitionPenaltyConfig::Default(),
      NoRepeatNgramConfig::Default(), SuppressTokensConfig::Default(),
      /*constraint=*/nullptr, std::make_shared<std::atomic<bool>>(false),
      [&terminal_stats](absl::StatusOr<Responses> responses) {
        ASSERT_OK(responses);
        if (IsTaskEndState(responses->GetTaskState())) {
          terminal_stats = responses->GetExecutorStats();
        }
      }));
  ASSERT_OK(execution_manager_->WaitUntilDone(decode_id, absl::Seconds(3)));

  ASSERT_TRUE(terminal_stats.has_value());
  EXPECT_EQ(terminal_stats->plain_steps(), 3);
  EXPECT_EQ(terminal_stats->plain_step_time(), absl::Milliseconds(9));
  // The benchmark stats of the first turn are the same.
  ASSERT_OK_AND_ASSIGN(BenchmarkInfo * benchmark_info,
                       execution_manager_->GetMutableBenchmarkInfo(session_id));
  ASSERT_TRUE(benchmark_info->GetExecutorStats().has_value());
  EXPECT_EQ(benchmark_info->GetExecutorStats()->plain_steps(), 3);

  EXPECT_OK(execution_manager_->ReleaseSession(session_id));
}

TEST_P(ExecutionManagerTest,
       TerminalCallbackCanWaitForTaskAndSessionCompletion) {
  CreateExecutionManager(CreateDefaultFakeLlmExecutor());
  ASSERT_OK_AND_ASSIGN(auto session_config, CreateDefaultSessionConfig());
  ASSERT_OK_AND_ASSIGN(const SessionId session_id,
                       execution_manager_->RegisterNewSession(session_config));

  std::vector<InputData> inputs;
  ASSERT_OK_AND_ASSIGN(auto input_text,
                       tokenizer_->TokenIdsToTensorBuffer({1, 2, 3}));
  inputs.push_back(InputText(std::move(input_text)));
  ASSERT_OK_AND_ASSIGN(const TaskId prefill_task_id,
                       execution_manager_->GetNewTaskId());
  ASSERT_OK(execution_manager_->AddPrefillTask(
      session_id, prefill_task_id, std::move(inputs), {},
      std::make_shared<std::atomic<bool>>(false), nullptr));
  ASSERT_OK(
      execution_manager_->WaitUntilDone(prefill_task_id, absl::Seconds(3)));

  ASSERT_OK_AND_ASSIGN(const TaskId decode_task_id,
                       execution_manager_->GetNewTaskId());
  absl::Status task_wait_status =
      absl::UnknownError("Terminal callback was not invoked");
  absl::Status session_wait_status =
      absl::UnknownError("Terminal callback was not invoked");
  absl::Status release_session_status =
      absl::UnknownError("Terminal callback was not invoked");

  // Simulate closing a single-use Conversation/Session inside its terminal
  // onDone callback (WaitUntilDone + WaitUntilSessionDone + ReleaseSession).
  ASSERT_OK(execution_manager_->AddDecodeTask(
      session_id, decode_task_id, {prefill_task_id},
      RepetitionPenaltyConfig::Default(), NoRepeatNgramConfig::Default(),
      SuppressTokensConfig::Default(), /*constraint=*/nullptr,
      std::make_shared<std::atomic<bool>>(false),
      [this, session_id, decode_task_id, &task_wait_status,
       &session_wait_status,
       &release_session_status](absl::StatusOr<Responses> responses) {
        if (!responses.ok() || responses->GetTaskState() != TaskState::kDone) {
          return;
        }
        task_wait_status =
            execution_manager_->WaitUntilDone(decode_task_id, absl::Seconds(1));
        session_wait_status = execution_manager_->WaitUntilSessionDone(
            session_id, absl::Seconds(1));
        release_session_status = execution_manager_->ReleaseSession(session_id);
      }));

  EXPECT_OK(execution_manager_->WaitUntilAllDone(absl::Seconds(5)));
  EXPECT_OK(task_wait_status);
  EXPECT_OK(session_wait_status);
  EXPECT_OK(release_session_status);
}

TEST_P(ExecutionManagerTest,
       ExternalWaitUntilDoneWaitsForTerminalCallbackToReturn) {
  CreateExecutionManager(CreateDefaultFakeLlmExecutor());
  ASSERT_OK_AND_ASSIGN(auto session_config, CreateDefaultSessionConfig());
  ASSERT_OK_AND_ASSIGN(const SessionId session_id,
                       execution_manager_->RegisterNewSession(session_config));

  std::vector<InputData> inputs;
  ASSERT_OK_AND_ASSIGN(auto input_text,
                       tokenizer_->TokenIdsToTensorBuffer({1, 2, 3}));
  inputs.push_back(InputText(std::move(input_text)));
  ASSERT_OK_AND_ASSIGN(const TaskId prefill_task_id,
                       execution_manager_->GetNewTaskId());
  ASSERT_OK(execution_manager_->AddPrefillTask(
      session_id, prefill_task_id, std::move(inputs), {},
      std::make_shared<std::atomic<bool>>(false), nullptr));
  ASSERT_OK(
      execution_manager_->WaitUntilDone(prefill_task_id, absl::Seconds(3)));

  ASSERT_OK_AND_ASSIGN(const TaskId decode_task_id,
                       execution_manager_->GetNewTaskId());
  std::atomic<bool> callback_finished{false};
  absl::Status reentrant_task_wait_status =
      absl::UnknownError("Terminal callback was not invoked");
  absl::Status reentrant_session_wait_status =
      absl::UnknownError("Terminal callback was not invoked");

  // Inside the terminal callback:
  // 1. Re-entrant WaitUntilDone and WaitUntilSessionDone must return
  //    immediately without self-deadlocking.
  // 2. External callers waiting via WaitUntilDone on another thread must not
  //    return until after the terminal callback has finished executing.
  ASSERT_OK(execution_manager_->AddDecodeTask(
      session_id, decode_task_id, {prefill_task_id},
      RepetitionPenaltyConfig::Default(), NoRepeatNgramConfig::Default(),
      SuppressTokensConfig::Default(), /*constraint=*/nullptr,
      std::make_shared<std::atomic<bool>>(false),
      [this, session_id, decode_task_id, &callback_finished,
       &reentrant_task_wait_status,
       &reentrant_session_wait_status](absl::StatusOr<Responses> responses) {
        if (!responses.ok() || responses->GetTaskState() != TaskState::kDone) {
          return;
        }
        reentrant_task_wait_status =
            execution_manager_->WaitUntilDone(decode_task_id, absl::Seconds(1));
        reentrant_session_wait_status =
            execution_manager_->WaitUntilSessionDone(session_id,
                                                     absl::Seconds(1));
        absl::SleepFor(absl::Milliseconds(100));
        callback_finished.store(true, std::memory_order_relaxed);
      }));

  EXPECT_OK(
      execution_manager_->WaitUntilDone(decode_task_id, absl::Seconds(3)));
  EXPECT_TRUE(callback_finished.load(std::memory_order_relaxed));
  EXPECT_OK(reentrant_task_wait_status);
  EXPECT_OK(reentrant_session_wait_status);
  EXPECT_OK(execution_manager_->ReleaseSession(session_id));
}

TEST_P(ExecutionManagerTest,
       DependentTaskTerminalCallbackCanWaitForTaskAndSessionCompletion) {
  auto fake_llm_executor = CreateDefaultFakeLlmExecutor();
  fake_llm_executor->SetDecodeDelay(absl::Milliseconds(50));
  fake_llm_executor->SetDecodeStatus(absl::InternalError("Executor failed"));
  CreateExecutionManager(std::move(fake_llm_executor));

  ASSERT_OK_AND_ASSIGN(auto session_config, CreateDefaultSessionConfig());
  ASSERT_OK_AND_ASSIGN(const SessionId session_id,
                       execution_manager_->RegisterNewSession(session_config));

  std::vector<InputData> inputs;
  ASSERT_OK_AND_ASSIGN(auto input_text,
                       tokenizer_->TokenIdsToTensorBuffer({1, 2, 3}));
  inputs.push_back(InputText(std::move(input_text)));
  ASSERT_OK_AND_ASSIGN(const TaskId prefill_task_id,
                       execution_manager_->GetNewTaskId());
  ASSERT_OK(execution_manager_->AddPrefillTask(
      session_id, prefill_task_id, std::move(inputs), {},
      std::make_shared<std::atomic<bool>>(false), nullptr));
  ASSERT_OK(
      execution_manager_->WaitUntilDone(prefill_task_id, absl::Seconds(3)));

  // Schedule a decode task that will fail after 50ms, and a dependent decode
  // task whose callback receives TaskState::kDependentTaskFailed and waits for
  // task and session completion (as happens when a user closes a Conversation
  // inside onError after an upstream task fails or is cancelled).
  ASSERT_OK_AND_ASSIGN(const TaskId failing_task_id,
                       execution_manager_->GetNewTaskId());
  std::vector<TaskId> terminal_callback_order;
  ASSERT_OK(execution_manager_->AddDecodeTask(
      session_id, failing_task_id, {prefill_task_id},
      RepetitionPenaltyConfig::Default(), NoRepeatNgramConfig::Default(),
      SuppressTokensConfig::Default(), /*constraint=*/nullptr,
      std::make_shared<std::atomic<bool>>(false),
      [failing_task_id,
       &terminal_callback_order](absl::StatusOr<Responses> responses) {
        if (!responses.ok()) {
          terminal_callback_order.push_back(failing_task_id);
        }
      }));

  ASSERT_OK_AND_ASSIGN(const TaskId dependent_task_id,
                       execution_manager_->GetNewTaskId());
  absl::Status task_wait_status =
      absl::UnknownError("Dependent terminal callback was not invoked");
  absl::Status session_wait_status =
      absl::UnknownError("Dependent terminal callback was not invoked");

  ASSERT_OK(execution_manager_->AddDecodeTask(
      session_id, dependent_task_id, {failing_task_id},
      RepetitionPenaltyConfig::Default(), NoRepeatNgramConfig::Default(),
      SuppressTokensConfig::Default(), /*constraint=*/nullptr,
      std::make_shared<std::atomic<bool>>(false),
      [this, session_id, dependent_task_id, &terminal_callback_order,
       &task_wait_status,
       &session_wait_status](absl::StatusOr<Responses> responses) {
        if (!responses.ok() ||
            responses->GetTaskState() != TaskState::kDependentTaskFailed) {
          return;
        }
        terminal_callback_order.push_back(dependent_task_id);
        task_wait_status = execution_manager_->WaitUntilDone(dependent_task_id,
                                                             absl::Seconds(1));
        session_wait_status = execution_manager_->WaitUntilSessionDone(
            session_id, absl::Seconds(1));
      }));

  EXPECT_OK(execution_manager_->WaitUntilAllDone(absl::Seconds(5)));
  EXPECT_THAT(terminal_callback_order,
              ElementsAre(failing_task_id, dependent_task_id));
  EXPECT_OK(task_wait_status);
  EXPECT_OK(session_wait_status);
  EXPECT_OK(execution_manager_->ReleaseSession(session_id));
}

TEST_P(ExecutionManagerTest,
       ReleaseSessionInUpstreamFailureCallbackSkipsDependentCallback) {
  auto fake_llm_executor = CreateDefaultFakeLlmExecutor();
  fake_llm_executor->SetDecodeDelay(absl::Milliseconds(50));
  fake_llm_executor->SetDecodeStatus(absl::InternalError("Executor failed"));
  CreateExecutionManager(std::move(fake_llm_executor));

  ASSERT_OK_AND_ASSIGN(auto session_config, CreateDefaultSessionConfig());
  ASSERT_OK_AND_ASSIGN(const SessionId session_id,
                       execution_manager_->RegisterNewSession(session_config));

  std::vector<InputData> inputs;
  ASSERT_OK_AND_ASSIGN(auto input_text,
                       tokenizer_->TokenIdsToTensorBuffer({1, 2, 3}));
  inputs.push_back(InputText(std::move(input_text)));
  ASSERT_OK_AND_ASSIGN(const TaskId prefill_task_id,
                       execution_manager_->GetNewTaskId());
  ASSERT_OK(execution_manager_->AddPrefillTask(
      session_id, prefill_task_id, std::move(inputs), {},
      std::make_shared<std::atomic<bool>>(false), nullptr));
  ASSERT_OK(
      execution_manager_->WaitUntilDone(prefill_task_id, absl::Seconds(3)));

  ASSERT_OK_AND_ASSIGN(const TaskId failing_task_id,
                       execution_manager_->GetNewTaskId());
  absl::Status session_wait_status =
      absl::UnknownError("Upstream terminal callback was not invoked");
  absl::Status release_session_status =
      absl::UnknownError("Upstream terminal callback was not invoked");

  ASSERT_OK(execution_manager_->AddDecodeTask(
      session_id, failing_task_id, {prefill_task_id},
      RepetitionPenaltyConfig::Default(), NoRepeatNgramConfig::Default(),
      SuppressTokensConfig::Default(), /*constraint=*/nullptr,
      std::make_shared<std::atomic<bool>>(false),
      [this, session_id, &session_wait_status,
       &release_session_status](absl::StatusOr<Responses> responses) {
        if (!responses.ok()) {
          session_wait_status = execution_manager_->WaitUntilSessionDone(
              session_id, absl::Seconds(1));
          release_session_status =
              execution_manager_->ReleaseSession(session_id);
        }
      }));

  ASSERT_OK_AND_ASSIGN(const TaskId dependent_task_id,
                       execution_manager_->GetNewTaskId());
  std::atomic<bool> dependent_terminal_callback_invoked{false};
  ASSERT_OK(execution_manager_->AddDecodeTask(
      session_id, dependent_task_id, {failing_task_id},
      RepetitionPenaltyConfig::Default(), NoRepeatNgramConfig::Default(),
      SuppressTokensConfig::Default(), /*constraint=*/nullptr,
      std::make_shared<std::atomic<bool>>(false),
      [&dependent_terminal_callback_invoked](
          absl::StatusOr<Responses> responses) {
        if (!responses.ok() ||
            responses->GetTaskState() == TaskState::kDependentTaskFailed) {
          dependent_terminal_callback_invoked.store(true,
                                                    std::memory_order_relaxed);
        }
      }));

  EXPECT_OK(execution_manager_->WaitUntilAllDone(absl::Seconds(5)));
  EXPECT_OK(session_wait_status);
  EXPECT_OK(release_session_status);
  EXPECT_FALSE(
      dependent_terminal_callback_invoked.load(std::memory_order_relaxed));
}

TEST_P(ExecutionManagerTest,
       DependentTaskProcessingWaitsForParentTerminalCallback) {
  CreateExecutionManager(CreateDefaultFakeLlmExecutor());
  ASSERT_OK_AND_ASSIGN(auto session_config, CreateDefaultSessionConfig());
  ASSERT_OK_AND_ASSIGN(const SessionId session_id,
                       execution_manager_->RegisterNewSession(session_config));

  std::vector<InputData> inputs;
  ASSERT_OK_AND_ASSIGN(auto input_text,
                       tokenizer_->TokenIdsToTensorBuffer({1, 2, 3}));
  inputs.push_back(InputText(std::move(input_text)));

  std::atomic<bool> prefill_done_callback_finished{false};
  std::atomic<bool> decode_processing_saw_prefill_done{false};

  ASSERT_OK_AND_ASSIGN(const TaskId prefill_task_id,
                       execution_manager_->GetNewTaskId());
  ASSERT_OK(execution_manager_->AddPrefillTask(
      session_id, prefill_task_id, std::move(inputs), {},
      std::make_shared<std::atomic<bool>>(false),
      [&prefill_done_callback_finished](absl::StatusOr<Responses> responses) {
        if (responses.ok() && responses->GetTaskState() == TaskState::kDone) {
          absl::SleepFor(absl::Milliseconds(50));
          prefill_done_callback_finished.store(true, std::memory_order_relaxed);
        }
      }));

  ASSERT_OK_AND_ASSIGN(const TaskId decode_task_id,
                       execution_manager_->GetNewTaskId());
  ASSERT_OK(execution_manager_->AddDecodeTask(
      session_id, decode_task_id, {prefill_task_id},
      RepetitionPenaltyConfig::Default(), NoRepeatNgramConfig::Default(),
      SuppressTokensConfig::Default(), /*constraint=*/nullptr,
      std::make_shared<std::atomic<bool>>(false),
      [&prefill_done_callback_finished, &decode_processing_saw_prefill_done](
          absl::StatusOr<Responses> responses) {
        if (responses.ok() &&
            responses->GetTaskState() == TaskState::kProcessing &&
            responses->GetTexts().empty()) {
          decode_processing_saw_prefill_done.store(
              prefill_done_callback_finished.load(std::memory_order_relaxed),
              std::memory_order_relaxed);
        }
      }));

  EXPECT_OK(
      execution_manager_->WaitUntilDone(decode_task_id, absl::Seconds(3)));
  EXPECT_TRUE(
      decode_processing_saw_prefill_done.load(std::memory_order_relaxed));
  EXPECT_OK(execution_manager_->ReleaseSession(session_id));
}

TEST_P(ExecutionManagerTest,
       PrefillTerminalCallbackObservesStepBeforeDependentPrefillRuns) {
  auto fake_llm_executor = std::make_unique<FakeLlmExecutor>(
      kVocabSize,
      /*prefill_tokens=*/std::vector<std::vector<int>>{{1, 2, 3}, {4, 5, 6}},
      /*decode_tokens=*/std::vector<std::vector<int>>{{0}});
  CreateExecutionManager(std::move(fake_llm_executor));

  ASSERT_OK_AND_ASSIGN(auto session_config, CreateDefaultSessionConfig());
  ASSERT_OK_AND_ASSIGN(const SessionId session_id,
                       execution_manager_->RegisterNewSession(session_config));
  ASSERT_OK_AND_ASSIGN(auto session_info,
                       execution_manager_->GetSessionInfo(session_id));

  std::vector<InputData> inputs_1;
  ASSERT_OK_AND_ASSIGN(auto input_text_1,
                       tokenizer_->TokenIdsToTensorBuffer({1, 2, 3}));
  inputs_1.push_back(InputText(std::move(input_text_1)));

  std::vector<InputData> inputs_2;
  ASSERT_OK_AND_ASSIGN(auto input_text_2,
                       tokenizer_->TokenIdsToTensorBuffer({4, 5, 6}));
  inputs_2.push_back(InputText(std::move(input_text_2)));

  absl::StatusOr<int> step_in_task_1_done_cb =
      absl::UnknownError("Task 1 done callback was not invoked");

  ASSERT_OK_AND_ASSIGN(const TaskId task_1_id,
                       execution_manager_->GetNewTaskId());
  ASSERT_OK(execution_manager_->AddPrefillTask(
      session_id, task_1_id, std::move(inputs_1), {},
      std::make_shared<std::atomic<bool>>(false),
      [this, session_info,
       &step_in_task_1_done_cb](absl::StatusOr<Responses> responses) {
        if (responses.ok() && responses->GetTaskState() == TaskState::kDone) {
          absl::SleepFor(absl::Milliseconds(50));
          step_in_task_1_done_cb =
              execution_manager_->GetCurrentStep(*session_info);
        }
      }));

  ASSERT_OK_AND_ASSIGN(const TaskId task_2_id,
                       execution_manager_->GetNewTaskId());
  ASSERT_OK(execution_manager_->AddPrefillTask(
      session_id, task_2_id, std::move(inputs_2), {task_1_id},
      std::make_shared<std::atomic<bool>>(false), nullptr));

  EXPECT_OK(execution_manager_->WaitUntilDone(task_2_id, absl::Seconds(3)));
  ASSERT_OK(step_in_task_1_done_cb);
  EXPECT_EQ(*step_in_task_1_done_cb, 3);
  ASSERT_OK_AND_ASSIGN(int final_step,
                       execution_manager_->GetCurrentStep(*session_info));
  EXPECT_EQ(final_step, 6);
  EXPECT_OK(execution_manager_->ReleaseSession(session_id));
}

TEST_P(ExecutionManagerTest,
       TerminalCallbackSchedulingFollowUpTaskDoesNotDeadlockOrClearTracking) {
  auto fake_llm_executor = std::make_unique<FakeLlmExecutor>(
      kVocabSize,
      /*prefill_tokens=*/std::vector<std::vector<int>>{{1, 2, 3}, {4, 5, 6}},
      /*decode_tokens=*/std::vector<std::vector<int>>{{0}});
  CreateExecutionManager(std::move(fake_llm_executor));

  ASSERT_OK_AND_ASSIGN(auto session_config, CreateDefaultSessionConfig());
  ASSERT_OK_AND_ASSIGN(const SessionId session_id,
                       execution_manager_->RegisterNewSession(session_config));

  std::vector<InputData> inputs_1;
  ASSERT_OK_AND_ASSIGN(auto input_text_1,
                       tokenizer_->TokenIdsToTensorBuffer({1, 2, 3}));
  inputs_1.push_back(InputText(std::move(input_text_1)));

  std::vector<InputData> inputs_2;
  ASSERT_OK_AND_ASSIGN(auto input_text_2,
                       tokenizer_->TokenIdsToTensorBuffer({4, 5, 6}));
  inputs_2.push_back(InputText(std::move(input_text_2)));

  ASSERT_OK_AND_ASSIGN(const TaskId task_1_id,
                       execution_manager_->GetNewTaskId());
  ASSERT_OK_AND_ASSIGN(const TaskId task_2_id,
                       execution_manager_->GetNewTaskId());

  std::atomic<bool> task_2_scheduled{false};
  std::atomic<bool> task_2_callback_finished{false};
  absl::Status session_wait_status =
      absl::UnknownError("Task 1 terminal callback was not invoked");

  // Simulate Conversation's channel-filtering completion flow:
  // Inside task_1's terminal callback, schedule a follow-up prefill task
  // (task_2) and then call WaitUntilSessionDone(session_id).
  // 1. WaitUntilSessionDone(session_id) must wait for task_2 to finish
  //    executing without deadlocking against execution_thread_pool_.
  // 2. When task_1's callback finishes, it must not clear task_2's pending
  //    callback tracking before task_2's own terminal callback has finished.
  ASSERT_OK(execution_manager_->AddPrefillTask(
      session_id, task_1_id, std::move(inputs_1), {},
      std::make_shared<std::atomic<bool>>(false),
      [this, session_id, task_2_id, inputs_2 = std::move(inputs_2),
       &task_2_scheduled, &task_2_callback_finished,
       &session_wait_status](absl::StatusOr<Responses> responses) mutable {
        if (!responses.ok() || responses->GetTaskState() != TaskState::kDone) {
          return;
        }
        EXPECT_OK(execution_manager_->AddPrefillTask(
            session_id, task_2_id, std::move(inputs_2), {},
            std::make_shared<std::atomic<bool>>(false),
            [&task_2_callback_finished](absl::StatusOr<Responses> resp2) {
              if (resp2.ok() && resp2->GetTaskState() == TaskState::kDone) {
                absl::SleepFor(absl::Milliseconds(100));
                task_2_callback_finished.store(true, std::memory_order_relaxed);
              }
            }));
        task_2_scheduled.store(true, std::memory_order_relaxed);
        session_wait_status = execution_manager_->WaitUntilSessionDone(
            session_id, absl::Seconds(1));
      }));

  EXPECT_OK(execution_manager_->WaitUntilDone(task_1_id, absl::Seconds(3)));
  ASSERT_TRUE(task_2_scheduled.load(std::memory_order_relaxed));
  EXPECT_OK(session_wait_status);
  EXPECT_OK(execution_manager_->WaitUntilDone(task_2_id, absl::Seconds(3)));
  EXPECT_TRUE(task_2_callback_finished.load(std::memory_order_relaxed));
  EXPECT_OK(execution_manager_->ReleaseSession(session_id));
}

INSTANTIATE_TEST_SUITE_P(
    ExecutionManagerTests, ExecutionManagerTest,
    ::testing::Values(ExecutionManagerType::kThreaded,
                      ExecutionManagerType::kSerial),
    [](const ::testing::TestParamInfo<ExecutionManagerTest::ParamType>& info) {
      switch (info.param) {
        case ExecutionManagerType::kThreaded:
          return "Threaded";
        case ExecutionManagerType::kSerial:
          return "Serial";
      }
    });

}  // namespace
}  // namespace litert::lm
