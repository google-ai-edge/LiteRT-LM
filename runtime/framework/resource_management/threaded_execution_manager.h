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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_FRAMEWORK_RESOURCE_MANAGEMENT_THREADED_EXECUTION_MANAGER_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_FRAMEWORK_RESOURCE_MANAGEMENT_THREADED_EXECUTION_MANAGER_H_

#include <atomic>
#include <memory>
#include <optional>
#include <tuple>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/base/thread_annotations.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "runtime/components/constrained_decoding/constraint.h"
#include "runtime/components/model_resources.h"
#include "runtime/components/tokenizer.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/audio_executor.h"
#include "runtime/executor/audio_executor_settings.h"
#include "runtime/executor/llm_executor.h"
#include "runtime/executor/vision_executor_settings.h"
#include "runtime/framework/resource_management/execution_manager.h"
#include "runtime/framework/resource_management/resource_manager.h"
#include "runtime/framework/threadpool.h"

namespace litert::lm {

// The execution manager is responsible for managing the execution of the tasks.
// It will handle the scheduling of the tasks and the dependencies between them.
// Note: The execution manager will create its own threadpool for executing the
// tasks, so thread safety interaction should be handled properly.
class ThreadedExecutionManager : public ExecutionManager {
 public:
  // Creates an ExecutionManager.
  // The ExecutionManager will take ownership of the executors and the sampler.
  // - tokenizer: The tokenizer used for encoding the text input. This is
  //   expected to be non-null.
  // - llm_executor: The executor used for prefill/decode the LLM. This is
  //   expected to be non-null.
  // - vision_executor_settings: The vision executor settings used for creating
  //   the vision executor. This can be null if no vision modality is used.
  // - audio_executor_settings: The audio executor settings used for creating
  //   the audio executor. This can be null if no audio modality is used.
  // - litert_env: The LIRTER environment used for creating the LLM context.
  //   This can be null if no LLM context is needed.
  static absl::StatusOr<std::unique_ptr<ThreadedExecutionManager>> Create(
      Tokenizer* absl_nonnull tokenizer,
      ModelResources* absl_nullable model_resources,
      std::unique_ptr<LlmExecutor> absl_nonnull llm_executor,
      std::unique_ptr<VisionExecutorSettings> absl_nullable
      vision_executor_settings,
      std::unique_ptr<AudioExecutorSettings> absl_nullable
      audio_executor_settings,
      ::litert::Environment* absl_nullable litert_env,
      std::unique_ptr<AudioExecutor> absl_nullable audio_executor = nullptr);

  ~ThreadedExecutionManager() override;

  // Waits until the task is done or the timeout is reached.
  // Returns:
  // - OK if the task is done.
  // - DEADLINE_EXCEEDED if the timeout is reached.
  // - Other errors if the task is failed.
  absl::Status WaitUntilDone(TaskId task_id, absl::Duration timeout) override
      ABSL_LOCKS_EXCLUDED(session_and_task_lookup_mutex_);

  // Waits until all tasks in the session are done or the timeout is reached.
  absl::Status WaitUntilSessionDone(SessionId session_id,
                                    absl::Duration timeout) override
      ABSL_LOCKS_EXCLUDED(session_and_task_lookup_mutex_);

  // Waits until all tasks are done or the timeout is reached.
  // Returns:
  // - OK if all tasks are done.
  // - DEADLINE_EXCEEDED if the timeout is reached.
  // - Other errors if any of the tasks is failed.
  absl::Status WaitUntilAllDone(absl::Duration timeout) override
      ABSL_LOCKS_EXCLUDED(session_and_task_lookup_mutex_);

  // Returns a new session ID.
  // The returned session ID is guaranteed to be unique.
  absl::StatusOr<SessionId> RegisterNewSession(
      SessionConfig session_config,
      std::optional<BenchmarkInfo> benchmark_info) override
      ABSL_LOCKS_EXCLUDED(session_and_task_lookup_mutex_);

  // Releases the session with the given session ID.
  absl::Status ReleaseSession(SessionId session_id) override
      ABSL_LOCKS_EXCLUDED(session_and_task_lookup_mutex_);

  // Cancels all tasks in the session with the given session ID.
  absl::Status CancelAllTasksInSession(SessionId session_id) override
      ABSL_LOCKS_EXCLUDED(session_and_task_lookup_mutex_);

  // Returns the session info with the given session ID.
  // Returns:
  // - The session info.
  // - INVALID_ARGUMENT if the session ID is not found.
  absl::StatusOr<std::shared_ptr<const SessionInfo>> GetSessionInfo(
      SessionId session_id) override
      ABSL_LOCKS_EXCLUDED(session_and_task_lookup_mutex_);

  // Returns the mutable benchmark info with the given session ID.
  // Note: The returned benchmark info is not thread-safe and should be used
  // with care to record appropriate metrics.
  // Returns:
  // - The mutable benchmark info.
  // - INVALID_ARGUMENT if the session ID is not found.
  absl::StatusOr<BenchmarkInfo*> GetMutableBenchmarkInfo(SessionId session_id)
      override ABSL_LOCKS_EXCLUDED(session_and_task_lookup_mutex_);

  // Returns a new task ID.
  // The returned task ID is guaranteed to be unique.
  absl::StatusOr<TaskId> GetNewTaskId() override;

  // Adds a prefill task to the execution manager.
  // - session_id: The ID of the session that created the task.
  // - task_id: The task ID of the task.
  // - inputs: The inputs of the prefill task.
  // - dep_tasks: The dependent tasks that should be done before the prefill
  //   task starts.
  // - cancelled: The cancelled flag for the prefill task.
  // - callback: The callback function.
  // Note: AddPrefillTask will acquire the task lookup mutex.
  absl::Status AddPrefillTask(
      SessionId session_id, TaskId task_id, std::vector<InputData> inputs,
      absl::flat_hash_set<TaskId> dep_tasks,
      std::shared_ptr<std::atomic<bool>> absl_nonnull cancelled,
      absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) override
      ABSL_LOCKS_EXCLUDED(session_and_task_lookup_mutex_);

  // Adds a decode task to the execution manager.
  // - session_id: The ID of the session that created the task.
  // - task_id: The task ID of the task.
  // - dep_tasks: The dependent tasks that should be done before the decode
  //   task starts.
  // - constraint: The constraint for the decode task.
  // - cancelled: The cancelled flag for the decode task.
  // - callback: The callback function.
  // - max_output_tokens: The maximum number of tokens to decode.
  // Note: AddDecodeTask will acquire the task lookup mutex.
  absl::Status AddDecodeTask(
      SessionId session_id, TaskId task_id,
      absl::flat_hash_set<TaskId> dep_tasks,
      Constraint* absl_nullable constraint,
      std::shared_ptr<std::atomic<bool>> absl_nonnull cancelled,
      absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback,
      int max_output_tokens) override
      ABSL_LOCKS_EXCLUDED(session_and_task_lookup_mutex_);

  // Adds a clone session task to the execution manager.
  // - session_id: The ID of the session that created the task.
  // - task_id: The task ID of the task.
  // - dep_tasks: The dependent tasks that should be done before the clone
  //   session task starts.
  // - cloned_session_id: The ID of the cloned session.
  // - callback: The callback function.
  // Note: AddCloneSessionTask will acquire the task lookup mutex.
  // TODO b/409401231 - Add unit tests for this function.
  absl::Status AddCloneSessionTask(
      SessionId session_id, TaskId task_id,
      absl::flat_hash_set<TaskId> dep_tasks, SessionId cloned_session_id,
      std::shared_ptr<std::atomic<bool>> absl_nonnull cancelled,
      absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) override
      ABSL_LOCKS_EXCLUDED(session_and_task_lookup_mutex_);

  // Adds a text scoring task to the execution manager.
  // - session_id: The ID of the session that created the task.
  // - task_id: The task ID of the task.
  // - dep_tasks: The dependent tasks that should be done before the text
  //   scoring task starts.
  // - target_text: The target text to be scored.
  // - store_token_lengths: Whether to store the token lengths in the
  //   responses.
  // - cancelled: The cancelled flag for the text scoring task.
  // - callback: The callback function.
  // Note: AddTextScoringTask will acquire the task lookup mutex.
  absl::Status AddTextScoringTask(
      SessionId session_id, TaskId task_id,
      absl::flat_hash_set<TaskId> dep_tasks,
      const std::vector<absl::string_view>& target_text,
      bool store_token_lengths,
      std::shared_ptr<std::atomic<bool>> absl_nonnull cancelled,
      absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) override
      ABSL_LOCKS_EXCLUDED(session_and_task_lookup_mutex_);

  // Returns the current step of the session.
  // - session_info: The session info of the session.
  // Returns:
  // - The current step of the session.
  absl::StatusOr<int> GetCurrentStep(const SessionInfo& session_info) override;

  // Sets the current step of the session to the target step.
  // - session_info: The session info of the session.
  // - target_step: The step to set the executor's current step to.
  // Returns:
  // - OK if the current step is set successfully.
  // - INVALID_ARGUMENT if the target step is greater than the current step.
  absl::Status SetCurrentStep(const SessionInfo& session_info,
                              int target_step) override;

  // Returns the audio executor properties.
  absl::StatusOr<AudioExecutorProperties> GetAudioExecutorProperties()
      const override;

  // Returns the vision executor properties.
  absl::StatusOr<VisionExecutorProperties> GetVisionExecutorProperties()
      const override;

 private:
  ThreadedExecutionManager(
      Tokenizer* absl_nonnull tokenizer,
      std::unique_ptr<ResourceManager> absl_nonnull resource_manager,
      ::litert::Environment* absl_nullable litert_env = nullptr);

  // Creates a task with the given task ID, task, dependent tasks, and callback.
  // - session_id: The ID of the session that created the task.
  // - task_id: The task ID of the task.
  // - task: The task function.
  // - dependent_tasks: The dependent tasks that should be done before the task
  //   starts.
  // - callback: The callback function.
  // Note: CreateTask will acquire the task lookup mutex.
  absl::Status CreateTask(
      SessionId session_id, TaskId task_id,
      absl::AnyInvocable<void()> absl_nonnull task,
      absl::flat_hash_set<TaskId> dependent_tasks,
      std::shared_ptr<std::atomic<bool>> absl_nonnull cancelled,
      absl::AnyInvocable<void(absl::StatusOr<Responses>)> absl_nonnull callback)
      ABSL_LOCKS_EXCLUDED(session_and_task_lookup_mutex_);

  // Queues the task with the given task ID.
  // - task_id: The task ID of the task.
  // Note: QueueTask expects the callers to acquire the task lookup mutex before
  // calling it.
  absl::Status QueueTask(TaskId task_id)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(session_and_task_lookup_mutex_);

  // Starts the task with the given task ID, and returns the session info and
  // callback function of the task.
  // - task_id: The task ID of the task.
  // Returns:
  // - The session info, cancelled flag and callback function of the task.
  // Note: StartTask will acquire the task lookup mutex.
  absl::StatusOr<std::tuple<
      std::shared_ptr<SessionInfo>, std::shared_ptr<std::atomic<bool>>,
      absl::AnyInvocable<void(absl::StatusOr<Responses>)>>>
  StartTask(TaskId task_id) ABSL_LOCKS_EXCLUDED(session_and_task_lookup_mutex_);

  // Finishes the task with the given task ID, responses, and callback.
  // - task_id: The task ID of the task.
  // - responses: The responses of the task.
  // - callback: The callback function.
  // Note: FinishTask will acquire the task lookup mutex.
  absl::Status FinishTask(
      TaskId task_id, absl::StatusOr<Responses> responses,
      absl::AnyInvocable<void(absl::StatusOr<Responses>)> absl_nonnull callback)
      ABSL_LOCKS_EXCLUDED(session_and_task_lookup_mutex_);

  // Finishes the task with the given task ID, responses, and callback. If the
  // task fails, the error will be logged.
  // - task_id: The task ID of the task.
  // - responses: The responses of the task.
  // - callback: The callback function.
  // Note: FinishTaskAndLogErrors will acquire the task lookup mutex.
  void FinishTaskAndLogErrors(
      TaskId task_id, absl::StatusOr<Responses> responses,
      absl::AnyInvocable<void(absl::StatusOr<Responses>)> absl_nonnull callback)
      ABSL_LOCKS_EXCLUDED(session_and_task_lookup_mutex_);

  // Returns all following tasks that are waiting.
  // - task_id: The task ID of the task.
  // Returns:
  // - The set of following tasks that are waiting for dependent tasks.
  // Note: AllFollowingWaitingTasks expects the callers to acquire the task
  // lookup mutex before calling it.
  absl::StatusOr<absl::flat_hash_set<TaskId>> FollowingWaitingTasks(
      TaskId task_id)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(session_and_task_lookup_mutex_);

  // Updates the task state with the given task ID and task state.
  // - task_id: The task ID of the task.
  // - task_state: The state of the task.
  // Note: UpdateTaskState expects the callers to acquire the task lookup mutex
  // before calling it.
  absl::Status UpdateTaskState(TaskId task_id, TaskState task_state)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(session_and_task_lookup_mutex_);

  // Updates all tasks to the given state.
  // - task_ids: The task IDs of the tasks.
  // - task_state: The state of the tasks.
  // Note: UpdateAllTasksToState expects the callers to acquire the task lookup
  // mutex before calling it.
  absl::Status UpdateAllTasksToState(
      const absl::flat_hash_set<TaskId>& task_ids, TaskState task_state)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(session_and_task_lookup_mutex_);

  // Processes and combines the contents of the preprocessed contents.
  // - preprocessed_contents: The preprocessed contents of the task.
  // Returns:
  // - The processed and combined contents of the preprocessed contents.
  // - benchmark_info: The benchmark info of the session.
  absl::StatusOr<ExecutorInputs> ProcessAndCombineContents(
      const std::vector<InputData>& preprocessed_contents,
      std::optional<BenchmarkInfo>& benchmark_info);

  // The session ID.
  std::atomic<SessionId> next_session_id_ = 0;

  // The next unique task ID.
  std::atomic<TaskId> next_task_id_ = 0;

  // The mutex for protecting the session and task lookup.
  absl::Mutex session_and_task_lookup_mutex_;
  // The session lookup map.
  // The key is the session ID.
  // The value is the session states.
  absl::flat_hash_map<SessionId, std::shared_ptr<SessionInfo> absl_nonnull>
      session_lookup_ ABSL_GUARDED_BY(session_and_task_lookup_mutex_) = {};
  // The task lookup map.
  // The key is the task ID.
  // The value is the task info.
  absl::flat_hash_map<TaskId, TaskInfo> task_lookup_
      ABSL_GUARDED_BY(session_and_task_lookup_mutex_) = {};
  // The last prefill token ID.
  int last_prefill_token_id_ = 0;
  // Tokenizer used for tokenization and detokenization.
  Tokenizer* absl_nonnull tokenizer_;
  // Resource manager used for managing the model resources and executors.
  std::unique_ptr<ResourceManager> absl_nonnull resource_manager_;
  // LiteRT environment used for creating the sampler.
  ::litert::Environment* absl_nullable litert_env_;
  // Thread pool used for executing the tasks.
  std::unique_ptr<ThreadPool> absl_nonnull execution_thread_pool_;
  // Thread pool used for executing the callbacks.
  std::unique_ptr<ThreadPool> absl_nonnull callback_thread_pool_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_FRAMEWORK_RESOURCE_MANAGEMENT_THREADED_EXECUTION_MANAGER_H_
