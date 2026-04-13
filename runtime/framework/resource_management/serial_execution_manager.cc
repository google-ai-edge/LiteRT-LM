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

#include "runtime/framework/resource_management/serial_execution_manager.h"

#include <atomic>
#include <deque>
#include <memory>
#include <optional>
#include <tuple>
#include <utility>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "runtime/components/tokenizer.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/audio_executor.h"
#include "runtime/executor/llm_executor.h"
#include "runtime/framework/resource_management/execution_manager.h"
#include "runtime/framework/resource_management/execution_manager_base.h"
#include "runtime/framework/resource_management/resource_manager.h"
#include "runtime/util/status_macros.h"

namespace litert::lm {

absl::StatusOr<std::unique_ptr<SerialExecutionManager>>
SerialExecutionManager::Create(
    Tokenizer* absl_nonnull tokenizer,
    ModelResources* absl_nullable model_resources,
    std::unique_ptr<LlmExecutor> absl_nonnull llm_executor,
    std::unique_ptr<VisionExecutorSettings> absl_nullable
    vision_executor_settings,
    std::unique_ptr<AudioExecutorSettings> absl_nullable
    audio_executor_settings,
    ::litert::Environment* absl_nullable litert_env,
    std::unique_ptr<AudioExecutor> absl_nullable audio_executor) {
  ASSIGN_OR_RETURN(
      auto resource_manager,
      ResourceManager::Create(model_resources, std::move(llm_executor),
                              std::move(vision_executor_settings),
                              std::move(audio_executor_settings), litert_env,
                              std::move(audio_executor)));
  return absl::WrapUnique(new SerialExecutionManager(
      tokenizer, std::move(resource_manager), litert_env));
}

absl::Status SerialExecutionManager::WaitUntilDone(TaskId task_id,
                                                   absl::Duration timeout) {
  absl::Time start_time = absl::Now();
  while (absl::Now() - start_time < timeout) {
    if (task_lookup_.contains(task_id) &&
        IsTaskEndState(task_lookup_.at(task_id).task_state)) {
      return absl::OkStatus();
    }
    if (task_queue_.empty()) {
      absl::SleepFor(absl::Milliseconds(1));
      continue;
    }
    TaskId next_task_id = task_queue_.front();
    task_queue_.pop_front();
    task_lookup_.at(next_task_id).task();
  }
  return absl::DeadlineExceededError("Task timed out.");
}

absl::Status SerialExecutionManager::WaitUntilSessionDone(
    SessionId session_id, absl::Duration timeout) {
  absl::Time start_time = absl::Now();
  while (absl::Now() - start_time < timeout) {
    if (session_lookup_.contains(session_id) &&
        session_lookup_.at(session_id)->active_tasks.empty()) {
      return absl::OkStatus();
    }
    if (task_queue_.empty()) {
      absl::SleepFor(absl::Milliseconds(1));
      continue;
    }
    TaskId next_task_id = task_queue_.front();
    task_queue_.pop_front();
    task_lookup_.at(next_task_id).task();
  }
  return absl::DeadlineExceededError("Session timed out.");
}

absl::Status SerialExecutionManager::WaitUntilAllDone(absl::Duration timeout) {
  absl::Time start_time = absl::Now();
  while (!task_queue_.empty()) {
    if (absl::Now() - start_time > timeout) {
      return absl::DeadlineExceededError("All tasks timed out.");
    }
    TaskId task_id = task_queue_.front();
    task_queue_.pop_front();
    task_lookup_.at(task_id).task();
  }
  return absl::OkStatus();
}

absl::StatusOr<SessionId> SerialExecutionManager::RegisterNewSession(
    SessionConfig session_config, std::optional<BenchmarkInfo> benchmark_info) {
  ASSIGN_OR_RETURN(
      auto session_info,
      CreateSessionInfo(std::move(session_config), std::move(benchmark_info)));
  return RegisterNewSessionUnlocked(std::move(session_info));
}

absl::Status SerialExecutionManager::ReleaseSession(SessionId session_id) {
  return ReleaseSessionUnlocked(session_id);
}

absl::Status SerialExecutionManager::CancelAllTasksInSession(
    SessionId session_id) {
  return CancelAllTasksInSessionUnlocked(session_id);
}

absl::StatusOr<std::shared_ptr<const SessionInfo>>
SerialExecutionManager::GetSessionInfo(SessionId session_id) {
  return GetSessionInfoUnlocked(session_id);
}

absl::StatusOr<BenchmarkInfo*> SerialExecutionManager::GetMutableBenchmarkInfo(
    SessionId session_id) {
  return GetMutableBenchmarkInfoUnlocked(session_id);
}

absl::StatusOr<TaskId> SerialExecutionManager::GetNewTaskId() {
  return next_task_id_.fetch_add(1);
}

absl::StatusOr<int> SerialExecutionManager::GetCurrentStep(
    const SessionInfo& session_info) {
  return GetCurrentStepUnlocked(session_info);
}

absl::Status SerialExecutionManager::SetCurrentStep(
    const SessionInfo& session_info, int target_step) {
  return SetCurrentStepUnlocked(session_info, target_step);
}

absl::Status SerialExecutionManager::CreateTask(
    SessionId session_id, TaskId task_id, absl::AnyInvocable<void()> task,
    absl::flat_hash_set<TaskId> dependent_tasks,
    std::shared_ptr<std::atomic<bool>> absl_nonnull cancelled,
    absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) {
  return CreateTaskUnlocked(session_id, task_id, std::move(task),
                            std::move(dependent_tasks), std::move(cancelled),
                            std::move(callback));
}

absl::Status SerialExecutionManager::QueueTask(TaskId task_id) {
  task_lookup_.at(task_id).callback(Responses(TaskState::kQueued));
  RETURN_IF_ERROR(UpdateTaskStateUnlocked(task_id, TaskState::kQueued));
  task_queue_.push_back(task_id);
  return absl::OkStatus();
}

absl::StatusOr<
    std::tuple<std::shared_ptr<SessionInfo>, std::shared_ptr<std::atomic<bool>>,
               absl::AnyInvocable<void(absl::StatusOr<Responses>)>>>
SerialExecutionManager::StartTask(TaskId task_id) {
  return StartTaskUnlocked(task_id);
}

absl::Status SerialExecutionManager::FinishTask(
    TaskId task_id, absl::StatusOr<Responses> responses,
    absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) {
  RETURN_IF_ERROR(FinishTaskUnlocked(task_id, responses, callback));
  TaskState next_task_state =
      responses.ok() ? responses->GetTaskState() : TaskState::kFailed;
  callback(std::move(responses));
  return UpdateTaskStateUnlocked(task_id, next_task_state);
}

void SerialExecutionManager::FinishTaskAndLogErrors(
    TaskId task_id, absl::StatusOr<Responses> responses,
    absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) {
  auto status = FinishTask(task_id, std::move(responses), std::move(callback));
  if (!status.ok()) {
    ABSL_LOG(ERROR) << "Failed to finish task: " << status
                    << " with task id: " << task_id;
  }
}

absl::Status SerialExecutionManager::CloneSession(
    SessionId session_id, SessionId cloned_session_id) {
  return CloneSessionUnlocked(session_id, cloned_session_id);
}

}  // namespace litert::lm
