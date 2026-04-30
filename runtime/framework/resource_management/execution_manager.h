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
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "runtime/components/sampler.h"
#include "runtime/components/stop_token_detector.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/framework/resource_management/context_handler/context_handler.h"

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
// Note: The execution manager interface defines the methods for managing tasks
// and sessions.
class ExecutionManager {
 public:
  virtual ~ExecutionManager() = default;

  // Waits until the task is done or the timeout is reached.
  // Returns:
  // - OK if the task is done.
  // - DEADLINE_EXCEEDED if the timeout is reached.
  // - Other errors if the task is failed.
  virtual absl::Status WaitUntilDone(TaskId task_id,
                                     absl::Duration timeout) = 0;

  virtual absl::Status WaitUntilSessionDone(SessionId session_id,
                                            absl::Duration timeout) = 0;

  // Waits until all tasks are done or the timeout is reached.
  // Returns:
  // - OK if all tasks are done.
  // - DEADLINE_EXCEEDED if the timeout is reached.
  // - Other errors if any of the tasks is failed.
  virtual absl::Status WaitUntilAllDone(absl::Duration timeout) = 0;

  // Returns a new session ID.
  // The returned session ID is guaranteed to be unique.
  virtual absl::StatusOr<SessionId> RegisterNewSession(
      SessionConfig session_config,
      std::optional<BenchmarkInfo> benchmark_info) = 0;

  absl::StatusOr<SessionId> RegisterNewSession(SessionConfig session_config) {
    return RegisterNewSession(std::move(session_config), std::nullopt);
  }

  // Releases the session with the given session ID.
  virtual absl::Status ReleaseSession(SessionId session_id) = 0;

  // Cancels all tasks in the session with the given session ID.
  virtual absl::Status CancelAllTasksInSession(SessionId session_id) = 0;

  // Returns the session info with the given session ID.
  // Returns:
  // - The session info.
  // - INVALID_ARGUMENT if the session ID is not found.
  virtual absl::StatusOr<std::shared_ptr<const SessionInfo>> GetSessionInfo(
      SessionId session_id) = 0;

  // Returns the mutable benchmark info with the given session ID.
  // Note: The returned benchmark info is not thread-safe and should be used
  // with care to record appropriate metrics.
  // Returns:
  // - The mutable benchmark info.
  // - INVALID_ARGUMENT if the session ID is not found.
  virtual absl::StatusOr<BenchmarkInfo*> GetMutableBenchmarkInfo(
      SessionId session_id) = 0;

  // Returns a new task ID.
  // The returned task ID is guaranteed to be unique.
  virtual absl::StatusOr<TaskId> GetNewTaskId() = 0;

  // Adds a prefill task to the execution manager.
  // - session_id: The ID of the session that created the task.
  // - task_id: The task ID of the task.
  // - inputs: The inputs of the prefill task.
  // - dep_tasks: The dependent tasks that should be done before the prefill
  //   task starts.
  // - cancelled: The cancelled flag for the prefill task.
  // - callback: The callback function.
  virtual absl::Status AddPrefillTask(
      SessionId session_id, TaskId task_id, std::vector<InputData> inputs,
      absl::flat_hash_set<TaskId> dep_tasks,
      std::shared_ptr<std::atomic<bool>> absl_nonnull cancelled,
      absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) = 0;

  // Adds a decode task to the execution manager.
  // - session_id: The ID of the session that created the task.
  // - task_id: The task ID of the task.
  // - dep_tasks: The dependent tasks that should be done before the decode
  //   task starts.
  // - constraint: The constraint for the decode task.
  // - cancelled: The cancelled flag for the decode task.
  // - callback: The callback function.
  virtual absl::Status AddDecodeTask(
      SessionId session_id, TaskId task_id,
      absl::flat_hash_set<TaskId> dep_tasks,
      Constraint* absl_nullable constraint,
      std::shared_ptr<std::atomic<bool>> absl_nonnull cancelled,
      absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback,
      int max_output_tokens) = 0;

  // Adds a decode task to the execution manager with the maximum output tokens
  // set to infinity.
  absl::Status AddDecodeTask(
      SessionId session_id, TaskId task_id,
      absl::flat_hash_set<TaskId> dep_tasks,
      Constraint* absl_nullable constraint,
      std::shared_ptr<std::atomic<bool>> absl_nonnull cancelled,
      absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) {
    return AddDecodeTask(session_id, task_id, std::move(dep_tasks), constraint,
                         std::move(cancelled), std::move(callback),
                         std::numeric_limits<int>::max());
  }

  // Adds a clone session task to the execution manager.
  // - session_id: The ID of the session that created the task.
  // - task_id: The task ID of the task.
  // - dep_tasks: The dependent tasks that should be done before the clone
  //   session task starts.
  // - cloned_session_id: The ID of the cloned session.
  // - callback: The callback function.
  // TODO b/409401231 - Add unit tests for this function.
  virtual absl::Status AddCloneSessionTask(
      SessionId session_id, TaskId task_id,
      absl::flat_hash_set<TaskId> dep_tasks, SessionId cloned_session_id,
      std::shared_ptr<std::atomic<bool>> absl_nonnull cancelled,
      absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) = 0;

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
  virtual absl::Status AddTextScoringTask(
      SessionId session_id, TaskId task_id,
      absl::flat_hash_set<TaskId> dep_tasks,
      const std::vector<absl::string_view>& target_text,
      bool store_token_lengths,
      std::shared_ptr<std::atomic<bool>> absl_nonnull cancelled,
      absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) = 0;

  // Returns the current step of the session.
  // - session_info: The session info of the session.
  // Returns:
  // - The current step of the session.
  virtual absl::StatusOr<int> GetCurrentStep(
      const SessionInfo& session_info) = 0;

  // Sets the current step of the session to the target step.
  // - session_info: The session info of the session.
  // - target_step: The step to set the executor's current step to.
  // Returns:
  // - OK if the current step is set successfully.
  // - INVALID_ARGUMENT if the target step is greater than the current step.
  virtual absl::Status SetCurrentStep(const SessionInfo& session_info,
                                      int target_step) = 0;

  // Returns the audio executor properties.
  virtual absl::StatusOr<AudioExecutorProperties> GetAudioExecutorProperties()
      const = 0;

  // Returns the vision executor properties.
  virtual absl::StatusOr<VisionExecutorProperties> GetVisionExecutorProperties()
      const = 0;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_FRAMEWORK_RESOURCE_MANAGEMENT_EXECUTION_MANAGER_H_
