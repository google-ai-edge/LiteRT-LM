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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_FRAMEWORK_RESOURCE_MANAGEMENT_EXECUTION_MANAGER_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_FRAMEWORK_RESOURCE_MANAGEMENT_EXECUTION_MANAGER_H_

#include <atomic>
#include <memory>
#include <optional>
#include <tuple>
#include <utility>
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
#include "runtime/components/lora_manager.h"
#include "runtime/components/model_resources.h"
#include "runtime/components/sampler.h"
#include "runtime/components/stop_token_detector.h"
#include "runtime/components/tokenizer.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/audio_executor_settings.h"
#include "runtime/executor/llm_executor.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/executor/vision_executor_settings.h"
#include "runtime/framework/resource_management/context_handler/context_handler.h"
#include "runtime/framework/resource_management/resource_manager.h"
#include "runtime/framework/threadpool.h"

namespace litert::lm {

using SessionId = int;
using TaskId = int;

// All the information about a session.
// - session_config: The config of the session.
// - context_handler: The context handler of the session.
// - sampler: The sampler of the session.
// - last_prefill_token_id: The last prefill token ID of the session.
// - stop_token_detector: The stop token detector of the session.
// - benchmark_info: The benchmark info of the session.
// - active_tasks: The active tasks of the session.
struct SessionInfo {
  SessionConfig session_config;
  std::shared_ptr<ContextHandler> context_handler;
  std::unique_ptr<Sampler> sampler;
  int last_prefill_token_id = 0;
  std::unique_ptr<StopTokenDetector> stop_token_detector;
  std::optional<BenchmarkInfo> benchmark_info = std::nullopt;
  absl::flat_hash_set<TaskId> active_tasks = {};
  // The LoRA ID to activate before each forward pass for this session.
  // If nullopt, no LoRA adapter is used (base model).
  std::optional<uint32_t> lora_id = std::nullopt;
};

// All the information about a task.
// - session_id: The ID of the session that created the task.
// - task: The task function. This is the function that will be executed by the
//   execution manager. Will be retrieved and moved by the queue task function.
// - task_state: The state of the task.
// - dependent_tasks: The dependent tasks that should be done before the task
//   starts.
// - following_tasks: The following tasks that are waiting for the task to
//   finish.
// - callback: The callback function. This is the function that will be called
//   when the task is done. Will be retrieved and moved by the start task
//   function.
struct TaskInfo {
  SessionId session_id;
  absl::AnyInvocable<void()> task;
  TaskState task_state = TaskState::kUnknown;
  absl::flat_hash_set<TaskId> dependent_tasks = {};
  absl::flat_hash_set<TaskId> following_tasks = {};
  std::shared_ptr<std::atomic<bool>> cancelled = nullptr;
  absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback;
};

// The execution manager is responsible for managing the execution of the tasks.
// It will handle the scheduling of the tasks and the dependencies between them.
// Note: The execution manager will create its own threadpool for executing the
// tasks, so thread safety interaction should be handled properly.
class ExecutionManager {
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
  static absl::StatusOr<std::unique_ptr<ExecutionManager>> Create(
      Tokenizer* absl_nonnull tokenizer,
      ModelResources* absl_nullable model_resources,
      std::unique_ptr<LlmExecutor> absl_nonnull llm_executor,
      std::unique_ptr<VisionExecutorSettings> absl_nullable
      vision_executor_settings,
      std::unique_ptr<AudioExecutorSettings> absl_nullable
      audio_executor_settings,
      ::litert::Environment* absl_nullable litert_env);

  ~ExecutionManager() {
    WaitUntilAllDone(Engine::kDefaultTimeout).IgnoreError();
  };

  // Waits until the task is done or the timeout is reached.
  // Returns:
  // - OK if the task is done.
  // - DEADLINE_EXCEEDED if the timeout is reached.
  // - Other errors if the task is failed.
  absl::Status WaitUntilDone(TaskId task_id, absl::Duration timeout)
      ABSL_LOCKS_EXCLUDED(session_and_task_lookup_mutex_);

  absl::Status WaitUntilSessionDone(SessionId session_id,
                                    absl::Duration timeout)
      ABSL_LOCKS_EXCLUDED(session_and_task_lookup_mutex_);

  // Waits until all tasks are done or the timeout is reached.
  // Returns:
  // - OK if all tasks are done.
  // - DEADLINE_EXCEEDED if the timeout is reached.
  // - Other errors if any of the tasks is failed.
  absl::Status WaitUntilAllDone(absl::Duration timeout)
      ABSL_LOCKS_EXCLUDED(session_and_task_lookup_mutex_);

  // Returns a new session ID.
  // The returned session ID is guaranteed to be unique.
  absl::StatusOr<SessionId> RegisterNewSession(
      SessionConfig session_config,
      std::optional<BenchmarkInfo> benchmark_info = std::nullopt)
      ABSL_LOCKS_EXCLUDED(session_and_task_lookup_mutex_);

  // Cancels all tasks in the session with the given session ID.
  absl::Status CancelAllTasksInSession(SessionId session_id)
      ABSL_LOCKS_EXCLUDED(session_and_task_lookup_mutex_);

  // Returns the session info with the given session ID.
  // Returns:
  // - The session info.
  // - INVALID_ARGUMENT if the session ID is not found.
  absl::StatusOr<std::shared_ptr<const SessionInfo>> GetSessionInfo(
      SessionId session_id) ABSL_LOCKS_EXCLUDED(session_and_task_lookup_mutex_);

  // Returns the mutable benchmark info with the given session ID.
  // Note: The returned benchmark info is not thread-safe and should be used
  // with care to record appropriate metrics.
  // Returns:
  // - The mutable benchmark info.
  // - INVALID_ARGUMENT if the session ID is not found.
  absl::StatusOr<BenchmarkInfo*> GetMutableBenchmarkInfo(SessionId session_id)
      ABSL_LOCKS_EXCLUDED(session_and_task_lookup_mutex_);

  // Returns a new task ID.
  // The returned task ID is guaranteed to be unique.
  absl::StatusOr<TaskId> GetNewTaskId();

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
      absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback)
      ABSL_LOCKS_EXCLUDED(session_and_task_lookup_mutex_);

  // Adds a decode task to the execution manager.
  // - session_id: The ID of the session that created the task.
  // - task_id: The task ID of the task.
  // - dep_tasks: The dependent tasks that should be done before the decode
  //   task starts.
  // - constraint: The constraint for the decode task.
  // - cancelled: The cancelled flag for the decode task.
  // - callback: The callback function.
  // Note: AddDecodeTask will acquire the task lookup mutex.
  absl::Status AddDecodeTask(
      SessionId session_id, TaskId task_id,
      absl::flat_hash_set<TaskId> dep_tasks,
      Constraint* absl_nullable constraint,
      std::shared_ptr<std::atomic<bool>> absl_nonnull cancelled,
      absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback)
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
      absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback)
      ABSL_LOCKS_EXCLUDED(session_and_task_lookup_mutex_);

  // Sets the LoRA manager used for per-session adapter switching.
  // The caller retains ownership of the LoraManager.
  void SetLoraManager(LoraManager* lora_manager) {
    lora_manager_ = lora_manager;
  }

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
      absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback)
      ABSL_LOCKS_EXCLUDED(session_and_task_lookup_mutex_);

 private:
  // Private constructor. Use the Create function instead.
  ExecutionManager(
      Tokenizer* absl_nonnull tokenizer,
      std::unique_ptr<ResourceManager> absl_nonnull resource_manager,
      ::litert::Environment* absl_nullable litert_env = nullptr)
      : tokenizer_(std::move(tokenizer)),
        resource_manager_(std::move(resource_manager)),
        litert_env_(litert_env) {
    execution_thread_pool_ =
        std::make_unique<ThreadPool>(/*name_prefix=*/"execution_thread_pool",
                                     /*max_num_threads=*/1);
    callback_thread_pool_ =
        std::make_unique<ThreadPool>(/*name_prefix=*/"callback_thread_pool",
                                     /*max_num_threads=*/1);
  }

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

  // Activates the LoRA adapter for the given session, or clears LoRA if the
  // session has no adapter configured.
  absl::Status ActivateLoraForSession(const SessionInfo& session_info);

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

  // TODO b/409401231 - Use LLM Context which is will be wrapped in a session
  // state.
  int last_prefill_token_id_ = 0;

  // The tokenizer used for encoding the text input.
  Tokenizer* absl_nonnull tokenizer_;

  // The resource manager used for managing the resources.
  std::unique_ptr<ResourceManager> absl_nonnull resource_manager_;

  // The LIRTER environment used for creating the LLM context.
  ::litert::Environment* absl_nullable litert_env_;

  // The thread pool with a single worker thread used for executing the tasks.
  std::unique_ptr<ThreadPool> absl_nonnull execution_thread_pool_;

  // The thread pool used for running the callbacks without blocking the
  // execution thread pool.
  // TODO b/476205457 - Consider updating all the callback triggering to use
  // this thread pool, and remove the syncing logic.
  std::unique_ptr<ThreadPool> absl_nonnull callback_thread_pool_;

  // Optional LoRA manager for per-session adapter switching. Not owned.
  LoraManager* lora_manager_ = nullptr;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_FRAMEWORK_RESOURCE_MANAGEMENT_EXECUTION_MANAGER_H_
