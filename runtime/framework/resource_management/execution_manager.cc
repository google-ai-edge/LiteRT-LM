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
#include <memory>
#include <optional>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/attributes.h"  // from @com_google_absl
#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/base/thread_annotations.h"  // from @com_google_absl
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
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/constrained_decoding/constraint.h"
#include "runtime/components/model_resources.h"
#include "runtime/components/sampler.h"
#include "runtime/components/sampler_factory.h"
#include "runtime/components/stop_token_detector.h"
#include "runtime/components/tokenizer.h"
#include "runtime/core/tasks.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/audio_executor_settings.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/llm_executor.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/executor/vision_executor_settings.h"
#include "runtime/framework/resource_management/resource_manager.h"
#include "runtime/proto/token.pb.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/executor_data_util.h"
#include "runtime/util/status_macros.h"  // IWYU pragma: keep
#include "runtime/util/tensor_buffer_util.h"

namespace litert::lm {

// Helper macro to check if the task has been cancelled.
#define RETURN_IF_CANCELLED(cancelled, task_id, callback)             \
  if (cancelled != nullptr && cancelled->load()) {                    \
    FinishTaskAndLogErrors(task_id, Responses(TaskState::kCancelled), \
                           std::move(callback));                      \
    return;                                                           \
  }

absl::StatusOr<SessionId> ExecutionManager::RegisterNewSession(
    SessionConfig session_config, std::optional<BenchmarkInfo> benchmark_info) {
  ASSIGN_OR_RETURN(auto context_handler,
                   resource_manager_->CreateContextHandler(session_config));
  std::unique_ptr<Sampler> sampler;
  if (session_config.UseExternalSampler()) {
    if (session_config.GetSamplerBackend() != Backend::CPU) {
      return absl::InvalidArgumentError(
          "External sampler currently only supports CPU backend.");
    }
    ASSIGN_OR_RETURN(sampler,
                     CreateSampler(session_config.GetSamplerBackend(),
                                   session_config.GetNumOutputCandidates(),
                                   session_config.GetSamplerParams(),
                                   litert_env_ ? litert_env_->Get() : nullptr));
  }
  auto stop_token_detector = std::make_unique<StopTokenDetector>(1);
  for (const auto& stop_token_sequence : session_config.GetStopTokenIds()) {
    auto status =
        stop_token_detector->AddStopTokenSequence(stop_token_sequence);
    if (!status.ok()) {
      ABSL_LOG(ERROR) << "Failed to add stop token sequence: " << status;
    }
  }
  SessionId session_id = next_session_id_.fetch_add(1);
  auto lora_id = session_config.GetLoraId();
  auto session_info = std::make_shared<SessionInfo>(SessionInfo{
      .session_config = std::move(session_config),
      .context_handler = std::move(context_handler),
      .sampler = std::move(sampler),
      .stop_token_detector = std::move(stop_token_detector),
      .benchmark_info = std::move(benchmark_info),
      .lora_id = lora_id,
  });
  {
    absl::MutexLock lock(session_and_task_lookup_mutex_);
    if (session_lookup_.contains(session_id)) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Session ", session_id, " already exists in session list."));
    }
    if (session_info->session_config.AudioModalityEnabled()) {
      RETURN_IF_ERROR(resource_manager_->TryLoadingAudioExecutor());
    }
    if (session_info->session_config.VisionModalityEnabled()) {
      RETURN_IF_ERROR(resource_manager_->TryLoadingVisionExecutor());
    }
    session_lookup_.insert({session_id, std::move(session_info)});
  }
  return session_id;
}

absl::Status ExecutionManager::CancelAllTasksInSession(SessionId session_id) {
  absl::MutexLock lock(session_and_task_lookup_mutex_);
  if (!session_lookup_.contains(session_id)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Session ", session_id, " not found in session list."));
  }
  for (TaskId task_id : session_lookup_.at(session_id)->active_tasks) {
    task_lookup_.at(task_id).cancelled->store(true);
  }
  return absl::OkStatus();
}

absl::StatusOr<std::shared_ptr<const SessionInfo>>
ExecutionManager::GetSessionInfo(SessionId session_id) {
  absl::MutexLock lock(session_and_task_lookup_mutex_);
  if (!session_lookup_.contains(session_id)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Session ", session_id, " not found in session list."));
  }
  return session_lookup_.at(session_id);
}

absl::StatusOr<BenchmarkInfo*> ExecutionManager::GetMutableBenchmarkInfo(
    SessionId session_id) {
  absl::MutexLock lock(session_and_task_lookup_mutex_);
  if (!session_lookup_.contains(session_id)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Session ", session_id, " not found in session list."));
  }
  if (!session_lookup_.at(session_id)->benchmark_info.has_value()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Session ", session_id, " does not have benchmark info."));
  }
  return &session_lookup_.at(session_id)->benchmark_info.value();
}

absl::StatusOr<TaskId> ExecutionManager::GetNewTaskId() {
  return next_task_id_.fetch_add(1);
}

absl::Status ExecutionManager::CreateTask(
    SessionId session_id, TaskId task_id,
    absl::AnyInvocable<void()> absl_nonnull task,
    absl::flat_hash_set<TaskId> dependent_tasks,
    std::shared_ptr<std::atomic<bool>> absl_nonnull cancelled,
    absl::AnyInvocable<void(absl::StatusOr<Responses>)> absl_nonnull callback) {
  absl::MutexLock lock(session_and_task_lookup_mutex_);
  if (!session_lookup_.contains(session_id)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Session ", session_id, " not found in session list. Task ", task_id,
        " cannot be created."));
  }
  if (task_lookup_.contains(task_id)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Task ", task_id, " already exists in task list."));
  }

  TaskState task_state = TaskState::kCreated;
  for (auto it = dependent_tasks.begin(); it != dependent_tasks.end();) {
    TaskId dep_task_id = *it;

    auto task_it = task_lookup_.find(dep_task_id);
    if (task_it == task_lookup_.end()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Dependency task ", dep_task_id, " not found in task list."));
    }
    TaskInfo& dep_task_info = task_it->second;

    bool erase_dependency = false;
    if (IsTaskEndState(dep_task_info.task_state)) {
      switch (dep_task_info.task_state) {
        case TaskState::kFailed:
          ABSL_FALLTHROUGH_INTENDED;
        case TaskState::kDependentTaskFailed:
          task_state = TaskState::kDependentTaskFailed;
          break;
        case TaskState::kCancelled:
          ABSL_FALLTHROUGH_INTENDED;
        case TaskState::kDependentTaskCancelled:
          if (task_state != TaskState::kDependentTaskFailed) {
            task_state = TaskState::kDependentTaskCancelled;
          }
          break;
        case TaskState::kDone:
          break;
        case TaskState::kMaxNumTokensReached:
          if (task_state == TaskState::kCreated) {
            task_state = TaskState::kMaxNumTokensReached;
          }
          break;
        default:
          return absl::InvalidArgumentError(
              absl::StrCat("Dependency task ", dep_task_id, " is in end state ",
                           dep_task_info.task_state,
                           " but not in Done or Cancelled or Failed state."));
      }
      erase_dependency = true;
    } else if (dep_task_info.task_state == TaskState::kLastCallbackQueued) {
      erase_dependency = true;
    } else {
      // Dependency task is not finished, so this new task must follow it.
      dep_task_info.following_tasks.insert(task_id);
    }

    // `erase()` will invalidate `it`, so advance `it` first.
    auto copy_it = it++;
    if (erase_dependency) {
      dependent_tasks.erase(copy_it);
    }
  }

  if (!IsTaskEndState(task_state)) {
    session_lookup_.at(session_id)->active_tasks.insert(task_id);
  }

  TaskInfo task_info;
  task_info.session_id = session_id;
  task_info.task_state = task_state;
  task_info.task = std::move(task);
  task_info.dependent_tasks = std::move(dependent_tasks);
  task_info.cancelled = cancelled;
  task_info.callback = std::move(callback);
  task_lookup_.insert({task_id, std::move(task_info)});

  task_lookup_.at(task_id).callback(Responses(task_state));

  // If there are no dependency tasks, we can queue the task immediately.
  // Otherwise, the task will be queued when all dependency tasks are done.
  if (task_state == TaskState::kCreated &&
      task_lookup_.at(task_id).dependent_tasks.empty()) {
    RETURN_IF_ERROR(QueueTask(task_id));
  }
  return absl::OkStatus();
}

absl::Status ExecutionManager::QueueTask(TaskId task_id) {
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
  RETURN_IF_ERROR(UpdateTaskState(task_id, TaskState::kQueued));

  return absl::OkStatus();
}

absl::StatusOr<
    std::tuple<std::shared_ptr<SessionInfo>, std::shared_ptr<std::atomic<bool>>,
               absl::AnyInvocable<void(absl::StatusOr<Responses>)>>>
ExecutionManager::StartTask(TaskId task_id) {
  absl::MutexLock lock(session_and_task_lookup_mutex_);
  if (!task_lookup_.contains(task_id)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Task ", task_id, " not found in task list."));
  }
  // If the task is cancelled, we don't need to start it.
  if (task_lookup_.at(task_id).task_state == TaskState::kCancelled) {
    return std::make_tuple(nullptr, nullptr, nullptr);
  }
  if (task_lookup_.at(task_id).callback == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrCat("Task ", task_id, " has no callback."));
  }
  if (task_lookup_.at(task_id).task_state != TaskState::kQueued) {
    auto error_status = absl::FailedPreconditionError(
        absl::StrCat("Task ", task_id, " is not in Queued state."));
    task_lookup_.at(task_id).callback(error_status);
    return error_status;
  }
  task_lookup_.at(task_id).callback(Responses(TaskState::kProcessing));
  RETURN_IF_ERROR(UpdateTaskState(task_id, TaskState::kProcessing));

  if (!session_lookup_.contains(task_lookup_.at(task_id).session_id)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Session ", task_lookup_.at(task_id).session_id,
                     " not found in session list."));
  }
  std::shared_ptr<SessionInfo> session_info =
      session_lookup_.at(task_lookup_.at(task_id).session_id);
  return std::make_tuple(session_info, task_lookup_.at(task_id).cancelled,
                         std::move(task_lookup_.at(task_id).callback));
}

absl::Status ExecutionManager::FinishTask(
    TaskId task_id, absl::StatusOr<Responses> responses,
    absl::AnyInvocable<void(absl::StatusOr<Responses>)> absl_nonnull callback) {
  auto invoke_callback_and_return =
      [&](absl::Status status) ABSL_EXCLUSIVE_LOCKS_REQUIRED(
          session_and_task_lookup_mutex_) -> absl::Status {
    callback(status);
    RETURN_IF_ERROR(UpdateTaskState(task_id, TaskState::kFailed));
    return status;
  };
  {
    absl::MutexLock lock(session_and_task_lookup_mutex_);
    if (!task_lookup_.contains(task_id)) {
      return absl::InvalidArgumentError(
          absl::StrCat("Task ", task_id, " not found in task list."));
    }
    if (task_lookup_.at(task_id).task_state != TaskState::kProcessing) {
      auto error_status = absl::FailedPreconditionError(
          absl::StrCat("Task ", task_id, " is not in Processing state."));
      return invoke_callback_and_return(error_status);
    }
    if (!responses.ok() || responses->GetTaskState() == TaskState::kCancelled) {
      auto following_waiting_tasks = FollowingWaitingTasks(task_id);
      if (!following_waiting_tasks.ok()) {
        return invoke_callback_and_return(following_waiting_tasks.status());
      }
      auto status = UpdateAllTasksToState(
          following_waiting_tasks.value(),
          responses.ok() ? TaskState::kDependentTaskCancelled
                         : TaskState::kDependentTaskFailed);
      if (!status.ok()) {
        return invoke_callback_and_return(status);
      }
    } else if (responses->GetTaskState() == TaskState::kDone ||
               responses->GetTaskState() == TaskState::kMaxNumTokensReached) {
      for (TaskId following_task_id :
           task_lookup_.at(task_id).following_tasks) {
        if (!task_lookup_.contains(following_task_id)) {
          auto error_status = absl::InvalidArgumentError(
              absl::StrCat("Following task ", following_task_id,
                           " not found in task list."));
          return invoke_callback_and_return(error_status);
        }
        if (IsTaskEndState(task_lookup_.at(following_task_id).task_state)) {
          continue;
        }
        if (task_lookup_.at(following_task_id).task_state !=
            TaskState::kCreated) {
          auto error_status = absl::InvalidArgumentError(
              absl::StrCat("Following task ", following_task_id,
                           " is not in Created state. Task state: ",
                           task_lookup_.at(following_task_id).task_state));
          return invoke_callback_and_return(error_status);
        }
        if (!task_lookup_.at(following_task_id)
                 .dependent_tasks.contains(task_id)) {
          auto error_status = absl::InvalidArgumentError(
              absl::StrCat("Following task ", following_task_id,
                           " does not depend on task ", task_id));
          return invoke_callback_and_return(error_status);
        }
        task_lookup_.at(following_task_id).dependent_tasks.erase(task_id);
        if (task_lookup_.at(following_task_id).dependent_tasks.empty()) {
          RETURN_IF_ERROR(QueueTask(following_task_id));
        }
      }
    } else if (!IsTaskEndState(responses->GetTaskState())) {
      return invoke_callback_and_return(absl::InvalidArgumentError(absl::StrCat(
          "Expected task state for responses to be end state, but got ",
          responses->GetTaskState())));
    }

    TaskState next_task_state =
        responses.ok() ? responses->GetTaskState() : TaskState::kFailed;
    if (callback_thread_pool_ != nullptr) {
      RETURN_IF_ERROR(callback_thread_pool_->Schedule(
          [callback = std::move(callback), responses = std::move(responses),
           task_id = task_id, next_task_state = std::move(next_task_state),
           this]() mutable {
            callback(std::move(responses));
            absl::MutexLock lock(session_and_task_lookup_mutex_);
            auto status = UpdateTaskState(task_id, next_task_state);
            if (!status.ok()) {
              ABSL_LOG(ERROR) << "Failed to update task state: " << status
                              << " with task id: " << task_id;
            }
          }));
      RETURN_IF_ERROR(UpdateTaskState(task_id, TaskState::kLastCallbackQueued));
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

void ExecutionManager::FinishTaskAndLogErrors(
    TaskId task_id, absl::StatusOr<Responses> responses,
    absl::AnyInvocable<void(absl::StatusOr<Responses>)> absl_nonnull callback) {
  auto status = FinishTask(task_id, std::move(responses), std::move(callback));
  if (!status.ok()) {
    ABSL_LOG(ERROR) << "Failed to finish task: " << status
                    << " with task id: " << task_id;
  }
}

absl::StatusOr<absl::flat_hash_set<TaskId>>
ExecutionManager::FollowingWaitingTasks(TaskId task_id) {
  absl::flat_hash_set<TaskId> following_waiting_tasks;
  for (TaskId following_task_id : task_lookup_.at(task_id).following_tasks) {
    if (!task_lookup_.contains(following_task_id)) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Following task ", following_task_id, " not found in task list."));
    }
    if (!task_lookup_.at(following_task_id).dependent_tasks.contains(task_id)) {
      return absl::InvalidArgumentError(
          absl::StrCat("Following task ", following_task_id,
                       " does not depend on task ", task_id));
    }
    if (!IsTaskEndState(task_lookup_.at(following_task_id).task_state)) {
      following_waiting_tasks.insert(following_task_id);
      ASSIGN_OR_RETURN(auto next_following_waiting_tasks,
                       FollowingWaitingTasks(following_task_id));
      following_waiting_tasks.insert(next_following_waiting_tasks.begin(),
                                     next_following_waiting_tasks.end());
    }
  }
  return following_waiting_tasks;
}

absl::Status ExecutionManager::UpdateTaskState(TaskId task_id,
                                               TaskState task_state) {
  if (!task_lookup_.contains(task_id)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Task ", task_id, " not found in task list."));
  }
  if (!IsTaskEndState(task_lookup_.at(task_id).task_state) &&
      IsTaskEndState(task_state)) {
    SessionId session_id = task_lookup_.at(task_id).session_id;
    if (session_lookup_.contains(session_id) &&
        session_lookup_.at(session_id)->active_tasks.contains(task_id)) {
      session_lookup_.at(task_lookup_.at(task_id).session_id)
          ->active_tasks.erase(task_id);
    } else {
      auto error_status = absl::InternalError(absl::StrCat(
          "Task ", task_id, " is not in active tasks of session ", session_id));
      if (task_lookup_.at(task_id).callback != nullptr) {
        task_lookup_.at(task_id).callback(error_status);
      }
      return error_status;
    }
  }
  task_lookup_.at(task_id).task_state = task_state;
  return absl::OkStatus();
}

absl::Status ExecutionManager::UpdateAllTasksToState(
    const absl::flat_hash_set<TaskId>& task_ids, TaskState task_state) {
  for (TaskId task_id : task_ids) {
    task_lookup_.at(task_id).dependent_tasks.clear();
    if (task_lookup_.at(task_id).callback) {
      task_lookup_.at(task_id).callback(Responses(task_state));
    }
    RETURN_IF_ERROR(UpdateTaskState(task_id, task_state));
  }
  return absl::OkStatus();
}

absl::Status ExecutionManager::ActivateLoraForSession(
    const SessionInfo& session_info) {
  if (lora_manager_ == nullptr) {
    return absl::OkStatus();
  }
  if (session_info.lora_id.has_value()) {
    return lora_manager_->UseLoRA(session_info.lora_id.value());
  }
  lora_manager_->ClearLoRA();
  return absl::OkStatus();
}

absl::StatusOr<ExecutorInputs> ExecutionManager::ProcessAndCombineContents(
    const std::vector<InputData>& preprocessed_contents,
    std::optional<BenchmarkInfo>& benchmark_info) {
  std::vector<int> combined_token_ids;
  std::vector<ExecutorVisionData> all_image_data;
  std::vector<ExecutorAudioData> all_audio_data;
  for (const auto& preprocessed_content : preprocessed_contents) {
    if (const auto* input_text =
            std::get_if<InputText>(&preprocessed_content)) {
      ASSIGN_OR_RETURN(const auto* token_ids,
                       input_text->GetPreprocessedTextTensor());
      if (token_ids == nullptr) {
        return absl::InvalidArgumentError(
            "Token IDs is null in preprocessed_contents.");
      }
      LITERT_ASSIGN_OR_RETURN(auto ids_buffer_span,
                              ReferTensorBufferAsSpan<int>(*token_ids));
      combined_token_ids.insert(combined_token_ids.end(),
                                ids_buffer_span.begin(), ids_buffer_span.end());
    } else if (const auto* input_image =
                   std::get_if<InputImage>(&preprocessed_content)) {
      ASSIGN_OR_RETURN(const auto* image_tensor,
                       input_image->GetPreprocessedImageTensor());
      if (image_tensor == nullptr) {
        return absl::InvalidArgumentError(
            "Image tensor is null in preprocessed_contents.");
      }
      if (benchmark_info.has_value()) {
        RETURN_IF_ERROR(benchmark_info->TimeMarkDelta("vision_executor"));
      }
      ASSIGN_OR_RETURN(auto vision_executor,
                       resource_manager_->AcquireVisionExecutor());
      ASSIGN_OR_RETURN(auto single_image_data,
                       vision_executor->Encode(*image_tensor));
      if (benchmark_info.has_value()) {
        RETURN_IF_ERROR(benchmark_info->TimeMarkDelta("vision_executor"));
      }
      ASSIGN_OR_RETURN(auto embeddings_ptr,
                       single_image_data.GetEmbeddingsPtr());
      const auto& dimensions = TensorBufferDims(*embeddings_ptr);
      // The last two dimensions are [..., image_token_num, model_dimension].
      const int image_token_num = dimensions.at(dimensions.size() - 2);
      combined_token_ids.insert(combined_token_ids.end(), image_token_num,
                                ExecutorVisionData::kSpecialToken);
      all_image_data.push_back(std::move(single_image_data));
    } else if (const auto* input_audio =
                   std::get_if<InputAudio>(&preprocessed_content)) {
      ASSIGN_OR_RETURN(const auto* spectrogram_tensor,
                       input_audio->GetPreprocessedAudioTensor());
      if (benchmark_info.has_value()) {
        RETURN_IF_ERROR(benchmark_info->TimeMarkDelta("audio_executor"));
      }
      ASSIGN_OR_RETURN(auto audio_executor,
                       resource_manager_->AcquireAudioExecutor());
      ASSIGN_OR_RETURN(auto single_audio_data,
                       audio_executor->Encode(*spectrogram_tensor));
      if (benchmark_info.has_value()) {
        RETURN_IF_ERROR(benchmark_info->TimeMarkDelta("audio_executor"));
      }
      const int num_audio_tokens = single_audio_data.GetValidTokens();
      all_audio_data.push_back(std::move(single_audio_data));
      combined_token_ids.insert(combined_token_ids.end(), num_audio_tokens,
                                ExecutorAudioData::kSpecialToken);
    } else if (const auto* input_audio_end =
                   std::get_if<InputAudioEnd>(&preprocessed_content)) {
      combined_token_ids.push_back(ExecutorAudioData::kEndToken);
    } else {
      return absl::InvalidArgumentError(
          "Unsupported input type in preprocessed_contents.");
    }
  }

  if (combined_token_ids.empty()) {
    return absl::InvalidArgumentError(
        "No token IDs found in preprocessed_contents.");
  }

  std::optional<ExecutorVisionData> combined_image_data = std::nullopt;
  if (!all_image_data.empty()) {
    ASSIGN_OR_RETURN(combined_image_data,
                     CombineExecutorVisionData(all_image_data));
  }
  std::optional<ExecutorAudioData> combined_audio_data = std::nullopt;
  if (!all_audio_data.empty()) {
    ASSIGN_OR_RETURN(combined_audio_data,
                     CombineExecutorAudioData(all_audio_data));
  }

  last_prefill_token_id_ = combined_token_ids.back();

  ASSIGN_OR_RETURN(auto token_ids_buffer,
                   tokenizer_->TokenIdsToTensorBuffer(combined_token_ids));

  ExecutorInputs inputs(ExecutorTextData(std::move(token_ids_buffer)),
                        std::move(combined_image_data),
                        std::move(combined_audio_data));
  return inputs;
}

absl::StatusOr<std::unique_ptr<ExecutionManager>> ExecutionManager::Create(
    Tokenizer* absl_nonnull tokenizer,
    ModelResources* absl_nullable model_resources,
    std::unique_ptr<LlmExecutor> absl_nonnull llm_executor,
    std::unique_ptr<VisionExecutorSettings> absl_nullable
    vision_executor_settings,
    std::unique_ptr<AudioExecutorSettings> absl_nullable
    audio_executor_settings,
    ::litert::Environment* absl_nullable litert_env) {
  std::unique_ptr<Sampler> sampler;
  ASSIGN_OR_RETURN(
      auto resource_manager,
      ResourceManager::Create(model_resources, std::move(llm_executor),
                              std::move(vision_executor_settings),
                              std::move(audio_executor_settings), litert_env));
  return absl::WrapUnique(
      new ExecutionManager(tokenizer, std::move(resource_manager), litert_env));
}

absl::Status ExecutionManager::WaitUntilDone(TaskId task_id,
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

absl::Status ExecutionManager::WaitUntilSessionDone(SessionId session_id,
                                                    absl::Duration timeout) {
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

absl::Status ExecutionManager::WaitUntilAllDone(absl::Duration timeout) {
  return execution_thread_pool_->WaitUntilDone(timeout);
}

absl::Status ExecutionManager::AddPrefillTask(
    SessionId session_id, TaskId task_id, std::vector<InputData> inputs,
    absl::flat_hash_set<TaskId> dep_tasks,
    std::shared_ptr<std::atomic<bool>> absl_nonnull cancelled,
    absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) {
  if (callback == nullptr) {
    callback = [](absl::StatusOr<Responses> responses) {};
  }

  auto task = [this, task_id, inputs = std::move(inputs)]() mutable -> void {
    auto task_info = StartTask(task_id);
    if (!task_info.ok()) {
      FinishTaskAndLogErrors(task_id, task_info.status(),
                             [](absl::StatusOr<Responses> responses) {});
      return;
    }
    auto [session_info, cancelled, callback] = std::move(task_info.value());
    // If the session info is nullptr, it means the task is cancelled before it
    // is started.
    if (session_info == nullptr) {
      return;
    }

    RETURN_IF_CANCELLED(cancelled, task_id, callback);

    auto executor_inputs =
        ProcessAndCombineContents(inputs, session_info->benchmark_info);
    if (!executor_inputs.ok()) {
      FinishTaskAndLogErrors(task_id, executor_inputs.status(),
                             std::move(callback));
      return;
    }

    RETURN_IF_CANCELLED(cancelled, task_id, callback);

    // Activate the LoRA adapter for this session before the forward pass.
    auto lora_status = ActivateLoraForSession(*session_info);
    if (!lora_status.ok()) {
      FinishTaskAndLogErrors(task_id, lora_status, std::move(callback));
      return;
    }

    auto llm_executor = resource_manager_->AcquireExecutorWithContextHandler(
        session_info->context_handler);
    if (!llm_executor.ok()) {
      FinishTaskAndLogErrors(task_id, llm_executor.status(),
                             std::move(callback));
      return;
    }

    RETURN_IF_CANCELLED(cancelled, task_id, callback);

    auto responses =
        Tasks::Prefill(*llm_executor.value(), *executor_inputs,
                       /*wait_for_completion=*/true,
                       /*benchmark_info=*/session_info->benchmark_info);
    if (!responses.ok()) {
      FinishTaskAndLogErrors(task_id, responses.status(), std::move(callback));
      return;
    }

    if (cancelled != nullptr && cancelled->load()) {
      responses = Responses(TaskState::kCancelled);
    } else {
      // Keep track of the last_prefill_token_id after prefill is done.
      auto processed_tokens = llm_executor.value()->GetProcessedTokens();
      if (!processed_tokens.ok()) {
        FinishTaskAndLogErrors(task_id, processed_tokens.status(),
                               std::move(callback));
        return;
      }
      auto current_step = llm_executor.value()->GetCurrentStep();
      if (!current_step.ok()) {
        FinishTaskAndLogErrors(task_id, current_step.status(),
                               std::move(callback));
        return;
      }
      session_info->last_prefill_token_id =
          processed_tokens.value()
              ->GetTokenAtStep(current_step.value() - 1)
              .at(0);
    }

    FinishTaskAndLogErrors(task_id, std::move(responses), std::move(callback));
    return;
  };

  return CreateTask(session_id, task_id, std::move(task), std::move(dep_tasks),
                    cancelled, std::move(callback));
}

absl::Status ExecutionManager::AddDecodeTask(
    SessionId session_id, TaskId task_id, absl::flat_hash_set<TaskId> dep_tasks,
    Constraint* absl_nullable constraint,
    std::shared_ptr<std::atomic<bool>> absl_nonnull cancelled,
    absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) {
  if (callback == nullptr) {
    callback = [](absl::StatusOr<Responses> responses) {};
  }

  auto task = [this, task_id, constraint, cancelled]() mutable -> void {
    auto task_info = StartTask(task_id);
    if (!task_info.ok()) {
      FinishTaskAndLogErrors(task_id, task_info.status(),
                             [](absl::StatusOr<Responses> responses) {});
      return;
    }
    auto [session_info, cancelled, callback] = std::move(task_info.value());
    // If the session info is nullptr, it means the task is cancelled before it
    // is started.
    if (session_info == nullptr) {
      return;
    }

    RETURN_IF_CANCELLED(cancelled, task_id, callback);

    // Activate the LoRA adapter for this session before the forward pass.
    auto lora_status = ActivateLoraForSession(*session_info);
    if (!lora_status.ok()) {
      FinishTaskAndLogErrors(task_id, lora_status, std::move(callback));
      return;
    }

    auto llm_executor = resource_manager_->AcquireExecutorWithContextHandler(
        session_info->context_handler);
    if (!llm_executor.ok()) {
      FinishTaskAndLogErrors(task_id, llm_executor.status(),
                             std::move(callback));
      return;
    }

    RETURN_IF_CANCELLED(cancelled, task_id, callback);

    auto num_output_candidates =
        session_info->session_config.GetNumOutputCandidates();
    session_info->stop_token_detector->ResetBatch(num_output_candidates);
    std::optional<Sampler*> optional_sampler = std::nullopt;
    std::optional<litert::TensorBuffer> decoded_ids_buffer = std::nullopt;
    if (session_info->sampler != nullptr) {
      optional_sampler = session_info->sampler.get();
      std::vector<int> decoded_ids(num_output_candidates,
                                   session_info->last_prefill_token_id);
      auto decoded_ids_buffer_or =
          CopyToTensorBuffer<int>(decoded_ids, {num_output_candidates, 1});
      if (!decoded_ids_buffer_or.HasValue()) {
        callback(absl::InternalError(decoded_ids_buffer_or.Error().Message()));
        return;
      }
      decoded_ids_buffer = std::move(decoded_ids_buffer_or.Value());
    }

    auto responses = Tasks::Decode(
        *llm_executor.value(), *tokenizer_, *session_info->stop_token_detector,
        num_output_candidates, session_info->benchmark_info, optional_sampler,
        constraint, std::move(decoded_ids_buffer), callback, cancelled.get());
    if (!responses.ok() && absl::IsCancelled(responses.status())) {
      responses = Responses(TaskState::kCancelled);
    }

    if (cancelled != nullptr && cancelled->load()) {
      responses = Responses(TaskState::kCancelled);
    }

    FinishTaskAndLogErrors(task_id, std::move(responses), std::move(callback));
    return;
  };

  return CreateTask(session_id, task_id, std::move(task), std::move(dep_tasks),
                    cancelled, std::move(callback));
}

absl::Status ExecutionManager::AddCloneSessionTask(
    SessionId session_id, TaskId task_id, absl::flat_hash_set<TaskId> dep_tasks,
    SessionId cloned_session_id,
    std::shared_ptr<std::atomic<bool>> absl_nonnull cancelled,
    absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) {
  if (callback == nullptr) {
    callback = [](absl::StatusOr<Responses> responses) {};
  }

  auto task = [this, task_id, session_id, cloned_session_id]() mutable -> void {
    auto task_info = StartTask(task_id);
    if (!task_info.ok()) {
      FinishTaskAndLogErrors(task_id, task_info.status(),
                             [](absl::StatusOr<Responses> responses) {});
      return;
    }
    auto [session_info, cancelled, callback] = std::move(task_info.value());
    // If the session info is nullptr, it means the task is cancelled before it
    // is started.
    if (session_info == nullptr) {
      return;
    }

    RETURN_IF_CANCELLED(cancelled, task_id, callback);

    absl::StatusOr<Responses> result = Responses(TaskState::kDone);
    [&] {
      std::shared_ptr<const SessionInfo> original_session_info;
      {
        absl::MutexLock lock(session_and_task_lookup_mutex_);
        if (!session_lookup_.contains(session_id)) {
          result = absl::InvalidArgumentError(
              absl::StrCat("Session ", session_id,
                           " not found in session list."));
          return;
        }
        original_session_info = session_lookup_.at(session_id);
      }

      auto cloned_context_handler_or = resource_manager_->CloneContextHandler(
          original_session_info->context_handler);
      if (!cloned_context_handler_or.ok()) {
        result = cloned_context_handler_or.status();
        return;
      }

      std::unique_ptr<Sampler> cloned_sampler;
      if (original_session_info->sampler != nullptr) {
        auto sampler = CreateSampler(
            original_session_info->session_config.GetSamplerBackend(),
            original_session_info->session_config.GetNumOutputCandidates(),
            original_session_info->session_config.GetSamplerParams());
        if (!sampler.ok()) {
          result = sampler.status();
          return;
        }
        cloned_sampler = std::move(*sampler);
      }

      auto cloned_stop_token_detector = std::make_unique<StopTokenDetector>(1);
      for (const auto& stop_token_sequence :
           original_session_info->session_config.GetStopTokenIds()) {
        auto status = cloned_stop_token_detector->AddStopTokenSequence(
            stop_token_sequence);
        if (!status.ok()) {
          result = status;
          return;
        }
      }

      {
        absl::MutexLock lock(session_and_task_lookup_mutex_);
        if (!session_lookup_.contains(cloned_session_id)) {
          result = absl::InvalidArgumentError(
              absl::StrCat("Cloned session ", cloned_session_id,
                           " not found in session list."));
          return;
        }
        session_lookup_.at(cloned_session_id)->session_config =
            original_session_info->session_config;
        session_lookup_.at(cloned_session_id)->context_handler =
            std::move(cloned_context_handler_or.value());
        session_lookup_.at(cloned_session_id)->sampler =
            std::move(cloned_sampler);
        session_lookup_.at(cloned_session_id)->last_prefill_token_id =
            original_session_info->last_prefill_token_id;
        session_lookup_.at(cloned_session_id)->stop_token_detector =
            std::move(cloned_stop_token_detector);
        session_lookup_.at(cloned_session_id)->benchmark_info =
            original_session_info->benchmark_info;
      }
    }();

    if (cancelled != nullptr && cancelled->load()) {
      result = Responses(TaskState::kCancelled);
    }

    FinishTaskAndLogErrors(task_id, result, std::move(callback));
    return;
  };

  return CreateTask(session_id, task_id, std::move(task), std::move(dep_tasks),
                    cancelled, std::move(callback));
}

absl::Status ExecutionManager::AddTextScoringTask(
    SessionId session_id, TaskId task_id,
    absl::flat_hash_set<TaskId> dep_tasks,
    const std::vector<absl::string_view>& target_text,
    bool store_token_lengths,
    std::shared_ptr<std::atomic<bool>> absl_nonnull cancelled,
    absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) {
  if (callback == nullptr) {
    callback = [](absl::StatusOr<Responses> responses) {};
  }

  auto task = [this, task_id, target_text,
               store_token_lengths]() mutable -> void {
    auto task_info = StartTask(task_id);
    if (!task_info.ok()) {
      FinishTaskAndLogErrors(task_id, task_info.status(),
                             [](absl::StatusOr<Responses> responses) {});
      return;
    }
    auto [session_info, cancelled, callback] = std::move(task_info.value());
    // If the session info is nullptr, it means the task is cancelled before it
    // is started.
    if (session_info == nullptr) {
      return;
    }

    RETURN_IF_CANCELLED(cancelled, task_id, callback);

    // Activate the LoRA adapter for this session before the forward pass.
    auto lora_status = ActivateLoraForSession(*session_info);
    if (!lora_status.ok()) {
      FinishTaskAndLogErrors(task_id, lora_status, std::move(callback));
      return;
    }

    auto llm_executor = resource_manager_->AcquireExecutorWithContextHandler(
        session_info->context_handler);
    if (!llm_executor.ok()) {
      FinishTaskAndLogErrors(task_id, llm_executor.status(),
                             std::move(callback));
      return;
    }

    RETURN_IF_CANCELLED(cancelled, task_id, callback);

    const int num_output_candidates =
        session_info->session_config.GetNumOutputCandidates();
    std::vector<int> decoded_ids(num_output_candidates,
                                 session_info->last_prefill_token_id);
    auto decoded_ids_buffer =
        CopyToTensorBuffer<int>(decoded_ids, {num_output_candidates, 1});
    if (!decoded_ids_buffer.HasValue()) {
      FinishTaskAndLogErrors(
          task_id, absl::InternalError(decoded_ids_buffer.Error().Message()),
          std::move(callback));
      return;
    }

    // TODO(b/435040163): Handle the temperature. Should it be calculated from
    // the sampler or the sampler parameters? For now, hardcode it to 1.0f for
    // testing.
    auto temperature = 1.0f;
    auto responses =
        Tasks::Score(*llm_executor.value(), *tokenizer_, target_text,
                     temperature, std::move(decoded_ids_buffer.Value()),
                     store_token_lengths);

    if (cancelled != nullptr && cancelled->load()) {
      responses = Responses(TaskState::kCancelled);
    }

    FinishTaskAndLogErrors(task_id, std::move(responses), std::move(callback));
    return;
  };

  return CreateTask(session_id, task_id, std::move(task), std::move(dep_tasks),
                    cancelled, std::move(callback));
}

}  // namespace litert::lm
