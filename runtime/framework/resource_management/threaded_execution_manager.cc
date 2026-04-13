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

#include "runtime/framework/resource_management/threaded_execution_manager.h"

#include <atomic>
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
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/components/sampler.h"
#include "runtime/components/sampler_factory.h"
#include "runtime/components/stop_token_detector.h"
#include "runtime/components/tokenizer.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/audio_executor.h"
#include "runtime/executor/audio_executor_settings.h"
#include "runtime/executor/llm_executor.h"
#include "runtime/executor/vision_executor_settings.h"
#include "runtime/framework/resource_management/execution_manager.h"
#include "runtime/framework/resource_management/execution_manager_base.h"
#include "runtime/framework/resource_management/resource_manager.h"
#include "runtime/util/status_macros.h"

namespace litert::lm {

// Helper macro to check if the task has been cancelled.
#define RETURN_IF_CANCELLED(cancelled, task_id, callback)             \
  if (cancelled != nullptr && cancelled->load()) {                    \
    FinishTaskAndLogErrors(task_id, Responses(TaskState::kCancelled), \
                           std::move(callback));                      \
    return;                                                           \
  }

absl::StatusOr<SessionId> ThreadedExecutionManager::RegisterNewSession(
    SessionConfig session_config, std::optional<BenchmarkInfo> benchmark_info) {
  ASSIGN_OR_RETURN(
      auto session_info,
      CreateSessionInfo(std::move(session_config), std::move(benchmark_info)));
  absl::MutexLock lock(session_and_task_lookup_mutex_);
  return RegisterNewSessionUnlocked(std::move(session_info));
}

absl::Status ThreadedExecutionManager::ReleaseSession(SessionId session_id) {
  absl::MutexLock lock(session_and_task_lookup_mutex_);
  return ReleaseSessionUnlocked(session_id);
}

absl::Status ThreadedExecutionManager::CancelAllTasksInSession(
    SessionId session_id) {
  absl::MutexLock lock(session_and_task_lookup_mutex_);
  return CancelAllTasksInSessionUnlocked(session_id);
}

absl::StatusOr<std::shared_ptr<const SessionInfo>>
ThreadedExecutionManager::GetSessionInfo(SessionId session_id) {
  absl::MutexLock lock(session_and_task_lookup_mutex_);
  return GetSessionInfoUnlocked(session_id);
}

absl::StatusOr<BenchmarkInfo*>
ThreadedExecutionManager::GetMutableBenchmarkInfo(SessionId session_id) {
  absl::MutexLock lock(session_and_task_lookup_mutex_);
  return GetMutableBenchmarkInfoUnlocked(session_id);
}

absl::StatusOr<TaskId> ThreadedExecutionManager::GetNewTaskId() {
  return next_task_id_.fetch_add(1);
}

absl::Status ThreadedExecutionManager::CreateTask(
    SessionId session_id, TaskId task_id, absl::AnyInvocable<void()> task,
    absl::flat_hash_set<TaskId> dependent_tasks,
    std::shared_ptr<std::atomic<bool>> absl_nonnull cancelled,
    absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) {
  absl::MutexLock lock(session_and_task_lookup_mutex_);
  return CreateTaskUnlocked(session_id, task_id, std::move(task),
                            std::move(dependent_tasks), std::move(cancelled),
                            std::move(callback));
}

absl::Status ThreadedExecutionManager::QueueTask(TaskId task_id) {
  if (!task_lookup_.contains(task_id)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Task ", task_id, " not found in task list."));
  }
  if (task_lookup_.at(task_id).task_state != TaskState::kCreated) {
    auto error_status = absl::FailedPreconditionError(
        absl::StrCat("Task ", task_id, " is not in Created state."));
    task_lookup_.at(task_id).callback(error_status);
    return error_status;
  }
  if (!task_lookup_.at(task_id).dependent_tasks.empty()) {
    auto error_status = absl::InvalidArgumentError(
        absl::StrCat("Task ", task_id, " has dependent tasks not finished."));
    task_lookup_.at(task_id).callback(error_status);
    return error_status;
  }

  auto task = std::move(task_lookup_.at(task_id).task);

  if (execution_thread_pool_ != nullptr) {
    RETURN_IF_ERROR(execution_thread_pool_->Schedule(std::move(task)));
  } else {
    ABSL_LOG(ERROR) << "Execution thread pool is null, skipping task: "
                    << task_id;
  }

  task_lookup_.at(task_id).callback(Responses(TaskState::kQueued));
  RETURN_IF_ERROR(UpdateTaskStateUnlocked(task_id, TaskState::kQueued));

  return absl::OkStatus();
}

absl::StatusOr<
    std::tuple<std::shared_ptr<SessionInfo>, std::shared_ptr<std::atomic<bool>>,
               absl::AnyInvocable<void(absl::StatusOr<Responses>)>>>
ThreadedExecutionManager::StartTask(TaskId task_id) {
  absl::MutexLock lock(session_and_task_lookup_mutex_);
  return StartTaskUnlocked(task_id);
}

absl::Status ThreadedExecutionManager::FinishTask(
    TaskId task_id, absl::StatusOr<Responses> responses,
    absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) {
  {
    absl::MutexLock lock(session_and_task_lookup_mutex_);
    RETURN_IF_ERROR(FinishTaskUnlocked(task_id, responses, callback));

    TaskState next_task_state =
        responses.ok() ? responses->GetTaskState() : TaskState::kFailed;

    if (callback_thread_pool_ != nullptr) {
      RETURN_IF_ERROR(callback_thread_pool_->Schedule(
          [callback = std::move(callback), responses = std::move(responses),
           task_id = task_id, next_task_state = std::move(next_task_state),
           this]() mutable {
            callback(std::move(responses));
            absl::MutexLock lock(session_and_task_lookup_mutex_);
            auto status = UpdateTaskStateUnlocked(task_id, next_task_state);
            if (!status.ok()) {
              ABSL_LOG(ERROR) << "Failed to update task state: " << status
                              << " with task id: " << task_id;
            }
          }));
      RETURN_IF_ERROR(
          UpdateTaskStateUnlocked(task_id, TaskState::kLastCallbackQueued));
    } else {
      callback(
          absl::InternalError("Callback thread pool is null, skipping "
                              "callback and ignoring task state."));
    }
  }

  if (callback_thread_pool_ != nullptr) {
    // TODO b/476205457 - Consider to use a asynchronous approach to handle the
    // callback, and remove this WaitUntilDone.
    RETURN_IF_ERROR(callback_thread_pool_->WaitUntilDone(absl::Seconds(10)));
  }

  return absl::OkStatus();
}

void ThreadedExecutionManager::FinishTaskAndLogErrors(
    TaskId task_id, absl::StatusOr<Responses> responses,
    absl::AnyInvocable<void(absl::StatusOr<Responses>)> absl_nonnull callback) {
  auto status = FinishTask(task_id, std::move(responses), std::move(callback));
  if (!status.ok()) {
    ABSL_LOG(ERROR) << "Failed to finish task: " << status
                    << " with task id: " << task_id;
  }
}

absl::StatusOr<std::unique_ptr<ThreadedExecutionManager>>
ThreadedExecutionManager::Create(
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
  return absl::WrapUnique(new ThreadedExecutionManager(
      tokenizer, std::move(resource_manager), litert_env));
}

absl::Status ThreadedExecutionManager::WaitUntilDone(TaskId task_id,
                                                     absl::Duration timeout) {
  auto task_done = [this, task_id]() {
    session_and_task_lookup_mutex_.AssertReaderHeld();
    return task_lookup_.contains(task_id) &&
           IsTaskEndState(task_lookup_.at(task_id).task_state);
  };
  absl::MutexLock lock(session_and_task_lookup_mutex_);
  return session_and_task_lookup_mutex_.AwaitWithTimeout(
             absl::Condition(&task_done), timeout)
             ? absl::OkStatus()
             : absl::DeadlineExceededError(absl::StrCat(
                   "Task ", task_id, " did not complete within the timeout of ",
                   absl::FormatDuration(timeout), "."));
}

absl::Status ThreadedExecutionManager::WaitUntilSessionDone(
    SessionId session_id, absl::Duration timeout) {
  auto session_done = [this, session_id]() {
    session_and_task_lookup_mutex_.AssertReaderHeld();
    return session_lookup_.contains(session_id) &&
           session_lookup_.at(session_id)->active_tasks.empty();
  };
  absl::MutexLock lock(session_and_task_lookup_mutex_);
  return session_and_task_lookup_mutex_.AwaitWithTimeout(
             absl::Condition(&session_done), timeout)
             ? absl::OkStatus()
             : absl::DeadlineExceededError(
                   absl::StrCat("Session ", session_id,
                                " did not complete within the timeout of ",
                                absl::FormatDuration(timeout), "."));
}

absl::Status ThreadedExecutionManager::WaitUntilAllDone(
    absl::Duration timeout) {
  return execution_thread_pool_->WaitUntilDone(timeout);
}

absl::Status ThreadedExecutionManager::CloneSession(
    SessionId session_id, SessionId cloned_session_id) {
  absl::MutexLock lock(&session_and_task_lookup_mutex_);
  return CloneSessionUnlocked(session_id, cloned_session_id);
}

absl::StatusOr<int> ThreadedExecutionManager::GetCurrentStep(
    const SessionInfo& session_info) {
  return GetCurrentStepUnlocked(session_info);
}

absl::Status ThreadedExecutionManager::SetCurrentStep(
    const SessionInfo& session_info, int target_step) {
  return SetCurrentStepUnlocked(session_info, target_step);
}

}  // namespace litert::lm
