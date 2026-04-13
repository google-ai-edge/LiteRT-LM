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
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/audio_executor.h"
#include "runtime/executor/audio_executor_settings.h"
#include "runtime/executor/llm_executor.h"
#include "runtime/executor/vision_executor_settings.h"
#include "runtime/framework/resource_management/execution_manager.h"
#include "runtime/framework/resource_management/execution_manager_base.h"
#include "runtime/framework/resource_management/resource_manager.h"
#include "runtime/framework/threadpool.h"

namespace litert::lm {

// The threaded execution manager is responsible for managing the execution of
// tasks in a threaded environment. It handles task scheduling and dependencies
// by utilizing internal thread pools.
class ThreadedExecutionManager : public ExecutionManagerBase {
 public:
  // Creates a ThreadedExecutionManager.
  // The ExecutionManager will take ownership of the executors and the sampler.
  // - tokenizer: The tokenizer used for encoding the text input. This is
  //   expected to be non-null.
  // - model_resources: The model resources used for the LLM.
  // - llm_executor: The executor used for prefill/decode the LLM. This is
  //   expected to be non-null.
  // - vision_executor_settings: The vision executor settings used for creating
  //   the vision executor. This can be null if no vision modality is used.
  // - audio_executor_settings: The audio executor settings used for creating
  //   the audio executor. This can be null if no audio modality is used.
  // - litert_env: The LiteRT environment used for creating the LLM context.
  //   This can be null if no LLM context is needed.
  // - audio_executor: The audio executor used for the task.
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

  ~ThreadedExecutionManager() override {
    WaitUntilAllDone(Engine::kDefaultTimeout).IgnoreError();
  }

  // Waits until the task is done or the timeout is reached.
  // Returns:
  // - OK if the task is done.
  // - DEADLINE_EXCEEDED if the timeout is reached.
  // - Other errors if the task is failed.
  absl::Status WaitUntilDone(TaskId task_id, absl::Duration timeout) override
      ABSL_LOCKS_EXCLUDED(session_and_task_lookup_mutex_);

  // Waits until all tasks in the session are done or the timeout is reached.
  // Returns:
  // - OK if all tasks in the session are done.
  // - DEADLINE_EXCEEDED if the timeout is reached.
  // - Other errors if any task in the session is failed.
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

 protected:
  absl::Status CreateTask(
      SessionId session_id, TaskId task_id, absl::AnyInvocable<void()> task,
      absl::flat_hash_set<TaskId> dependent_tasks,
      std::shared_ptr<std::atomic<bool>> absl_nonnull cancelled,
      absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) override
      ABSL_LOCKS_EXCLUDED(session_and_task_lookup_mutex_);

  absl::Status QueueTask(TaskId task_id) override
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(session_and_task_lookup_mutex_);

  absl::StatusOr<std::tuple<
      std::shared_ptr<SessionInfo>, std::shared_ptr<std::atomic<bool>>,
      absl::AnyInvocable<void(absl::StatusOr<Responses>)>>>
  StartTask(TaskId task_id) override
      ABSL_LOCKS_EXCLUDED(session_and_task_lookup_mutex_);

  absl::Status FinishTask(
      TaskId task_id, absl::StatusOr<Responses> responses,
      absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) override
      ABSL_LOCKS_EXCLUDED(session_and_task_lookup_mutex_);

  void FinishTaskAndLogErrors(
      TaskId task_id, absl::StatusOr<Responses> responses,
      absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) override
      ABSL_LOCKS_EXCLUDED(session_and_task_lookup_mutex_);

  absl::Status CloneSession(SessionId session_id,
                            SessionId cloned_session_id) override
      ABSL_LOCKS_EXCLUDED(session_and_task_lookup_mutex_);

 private:
  ThreadedExecutionManager(
      Tokenizer* absl_nonnull tokenizer,
      std::unique_ptr<ResourceManager> absl_nonnull resource_manager,
      ::litert::Environment* absl_nullable litert_env = nullptr)
      : ExecutionManagerBase(tokenizer, std::move(resource_manager),
                             litert_env) {
    execution_thread_pool_ =
        std::make_unique<ThreadPool>(/*name_prefix=*/"execution_thread_pool",
                                     /*max_num_threads=*/1);
    callback_thread_pool_ =
        std::make_unique<ThreadPool>(/*name_prefix=*/"callback_thread_pool",
                                     /*max_num_threads=*/1);
  }

  // The mutex for protecting the session and task lookup.
  absl::Mutex session_and_task_lookup_mutex_;

  // The thread pool with a single worker thread used for executing the tasks.
  std::unique_ptr<ThreadPool> absl_nonnull execution_thread_pool_;

  // The thread pool used for running the callbacks without blocking the
  // execution thread pool.
  // TODO b/476205457 - Consider updating all the callback triggering to use
  // this thread pool, and remove the syncing logic.
  std::unique_ptr<ThreadPool> absl_nonnull callback_thread_pool_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_FRAMEWORK_RESOURCE_MANAGEMENT_THREADED_EXECUTION_MANAGER_H_
