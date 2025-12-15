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
                                   session_config.GetSamplerParams()));
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
  auto session_info = std::make_shared<SessionInfo>(SessionInfo{
      .session_config = std::move(session_config),
      .context_handler = std::move(context_handler),
      .sampler = std::move(sampler),
      .stop_token_detector = std::move(stop_token_detector),
      .benchmark_info = std::move(benchmark_info),
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

absl::Status ExecutionManager::ReleaseSession(SessionId session_id) {
  absl::MutexLock lock(session_and_task_lookup_mutex_);
  if (!session_lookup_.contains(session_id)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Session ", session_id, " not found in session list."));
  }
  // TODO b/409401231 - Update to use cancellation of all tasks instead of
  // directly update to cancelled state.
  RETURN_IF_ERROR(UpdateAllTasksToState(
      session_lookup_.at(session_id)->active_tasks, TaskState::kCancelled));
  session_lookup_.at(session_id)->context_handler = nullptr;
  session_lookup_.at(session_id)->sampler = nullptr;
  session_lookup_.at(session_id)->stop_token_detector = nullptr;
  session_lookup_.at(session_id)->active_tasks.clear();
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

absl::StatusOr<TaskId> ExecutionManager::GetNewTaskId() {
  return next_task_id_.fetch_add(1);
}

absl::Status ExecutionManager::CreateTask(
    SessionId session_id, TaskId task_id,
    absl::AnyInvocable<void()> absl_nonnull task,
    absl::flat_hash_set<TaskId> dependent_tasks,
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
  for (TaskId dep_task_id : dependent_tasks) {
    if (!task_lookup_.contains(dep_task_id)) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Dependency task ", dep_task_id, " not found in task list."));
    }
    if (IsTaskEndState(task_lookup_.at(dep_task_id).task_state)) {
      switch (task_lookup_.at(dep_task_id).task_state) {
        case TaskState::kFailed:
          ABSL_FALLTHROUGH_INTENDED;
        case TaskState::kDependentTaskFailed:
          task_state = TaskState::kDependentTaskFailed;
          break;
        case TaskState::kCancelled:
          ABSL_FALLTHROUGH_INTENDED;
        case TaskState::kDependentTaskCancelled:
          task_state = TaskState::kDependentTaskCancelled;
          break;
        case TaskState::kDone:
          break;
        case TaskState::kMaxNumTokensReached:
          task_state = TaskState::kMaxNumTokensReached;
          break;
        default:
          return absl::InvalidArgumentError(
              absl::StrCat("Dependency task ", dep_task_id, " is in end state ",
                           task_lookup_.at(dep_task_id).task_state,
                           " but not in Done or Cancelled or Failed state."));
      }
      dependent_tasks.erase(dep_task_id);
    } else {
      task_lookup_.at(dep_task_id).following_tasks.insert(task_id);
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

  RETURN_IF_ERROR(execution_thread_pool_->Schedule(std::move(task)));

  task_lookup_.at(task_id).callback(Responses(TaskState::kQueued));
  RETURN_IF_ERROR(UpdateTaskState(task_id, TaskState::kQueued));

  return absl::OkStatus();
}

absl::StatusOr<std::pair<std::shared_ptr<SessionInfo>,
                         absl::AnyInvocable<void(absl::StatusOr<Responses>)>>>
ExecutionManager::StartTask(TaskId task_id) {
  absl::MutexLock lock(session_and_task_lookup_mutex_);
  if (!task_lookup_.contains(task_id)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Task ", task_id, " not found in task list."));
  }
  // If the task is cancelled, we don't need to start it.
  if (task_lookup_.at(task_id).task_state == TaskState::kCancelled) {
    return std::make_pair(nullptr, nullptr);
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
  return std::make_pair(session_info,
                        std::move(task_lookup_.at(task_id).callback));
}

absl::Status ExecutionManager::FinishTask(
    TaskId task_id, absl::StatusOr<Responses> responses,
    absl::AnyInvocable<void(absl::StatusOr<Responses>)> absl_nonnull callback) {
  auto invoke_callback_and_return =
      [&](absl::Status status) ABSL_EXCLUSIVE_LOCKS_REQUIRED(
          session_and_task_lookup_mutex_) -> absl::Status {
    callback(status);
    SessionId session_id = task_lookup_.at(task_id).session_id;
    if (session_lookup_.contains(session_id) &&
        session_lookup_.at(session_id)->active_tasks.contains(task_id)) {
      session_lookup_.at(task_lookup_.at(task_id).session_id)
          ->active_tasks.erase(task_id);
    }
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

    if (responses.ok()) {
      auto task_state = responses->GetTaskState();
      callback(std::move(responses));
      RETURN_IF_ERROR(UpdateTaskState(task_id, task_state));
    } else {
      callback(std::move(responses));
      RETURN_IF_ERROR(UpdateTaskState(task_id, TaskState::kFailed));
    }
  }
  return absl::OkStatus();
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

absl::StatusOr<ExecutorInputs> ExecutionManager::ProcessAndCombineContents(
    const std::vector<InputData>& preprocessed_contents) {
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
      if (benchmark_info_.has_value()) {
        RETURN_IF_ERROR(benchmark_info_->TimeMarkDelta("vision_executor"));
      }
      ASSIGN_OR_RETURN(auto vision_executor,
                       resource_manager_->AcquireVisionExecutor());
      ASSIGN_OR_RETURN(auto single_image_data,
                       vision_executor->Encode(*image_tensor));
      if (benchmark_info_.has_value()) {
        RETURN_IF_ERROR(benchmark_info_->TimeMarkDelta("vision_executor"));
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
      if (benchmark_info_.has_value()) {
        RETURN_IF_ERROR(benchmark_info_->TimeMarkDelta("audio_executor"));
      }
      ASSIGN_OR_RETURN(auto audio_executor,
                       resource_manager_->AcquireAudioExecutor());
      ASSIGN_OR_RETURN(auto single_audio_data,
                       audio_executor->Encode(*spectrogram_tensor));
      if (benchmark_info_.has_value()) {
        RETURN_IF_ERROR(benchmark_info_->TimeMarkDelta("audio_executor"));
      }
      const int num_audio_tokens = single_audio_data.GetValidTokens();
      all_audio_data.push_back(std::move(single_audio_data));
      combined_token_ids.insert(combined_token_ids.end(), num_audio_tokens,
                                ExecutorAudioData::kSpecialToken);
      combined_token_ids.push_back(ExecutorAudioData::kEndToken);
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
    std::unique_ptr<LlmExecutor> absl_nonnull llm_executor,
    std::unique_ptr<VisionExecutorSettings> absl_nullable
    vision_executor_settings,
    std::unique_ptr<AudioExecutorSettings> absl_nullable
    audio_executor_settings,
    std::unique_ptr<::litert::Environment> absl_nullable litert_env) {
  std::unique_ptr<Sampler> sampler;
  ASSIGN_OR_RETURN(auto resource_manager,
                   ResourceManager::Create(std::move(llm_executor),
                                           std::move(vision_executor_settings),
                                           std::move(audio_executor_settings),
                                           std::move(litert_env)));
  return absl::WrapUnique(
      new ExecutionManager(tokenizer, std::move(resource_manager)));
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

absl::Status ExecutionManager::WaitUntilAllDone(absl::Duration timeout) {
  return execution_thread_pool_->WaitUntilDone(timeout);
}

absl::Status ExecutionManager::AddPrefillTask(
    SessionId session_id, TaskId task_id, std::vector<InputData> inputs,
    absl::flat_hash_set<TaskId> dep_tasks,
    absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) {
  if (callback == nullptr) {
    callback = [](absl::StatusOr<Responses> responses) {};
  }

  auto task = [this, task_id, inputs = std::move(inputs)]() mutable -> void {
    auto session_info_and_callback = StartTask(task_id);
    if (!session_info_and_callback.ok()) {
      ABSL_LOG(INFO) << "Failed to start task: "
                     << session_info_and_callback.status()
                     << " with task id: " << task_id;
      return;
    }
    auto [session_info, callback] =
        std::move(session_info_and_callback.value());
    // If the session info is nullptr, it means the task is cancelled before it
    // is started.
    if (session_info == nullptr) {
      return;
    }

    auto executor_inputs = ProcessAndCombineContents(inputs);
    if (!executor_inputs.ok()) {
      callback(executor_inputs.status());
      return;
    }

    auto llm_executor = resource_manager_->AcquireExecutorWithContextHandler(
        session_info->context_handler);
    if (!llm_executor.ok()) {
      callback(llm_executor.status());
      return;
    }

    auto responses =
        Tasks::Prefill(*llm_executor.value(), *executor_inputs,
                       /*wait_for_completion=*/true,
                       /*benchmark_info=*/session_info->benchmark_info);

    // Keep track of the last_prefill_token_id after prefill is done.
    if (responses.ok()) {
      auto processed_tokens = llm_executor.value()->GetProcessedTokens();
      if (!processed_tokens.ok()) {
        responses = processed_tokens.status();
      }
      auto current_step = llm_executor.value()->GetCurrentStep();
      if (!current_step.ok()) {
        responses = current_step.status();
      }
      if (processed_tokens.ok() && current_step.ok()) {
        session_info->last_prefill_token_id =
            processed_tokens.value()
                ->GetTokenAtStep(current_step.value() - 1)
                .at(0);
      }
    }

    auto status =
        FinishTask(task_id, std::move(responses), std::move(callback));
    if (!status.ok()) {
      ABSL_LOG(ERROR) << "Failed to finish task: " << status
                      << " with task id: " << task_id;
    }
    return;
  };

  return CreateTask(session_id, task_id, std::move(task), std::move(dep_tasks),
                    std::move(callback));
}

absl::Status ExecutionManager::AddDecodeTask(
    SessionId session_id, TaskId task_id, absl::flat_hash_set<TaskId> dep_tasks,
    Constraint* absl_nullable constraint,
    std::shared_ptr<std::atomic<bool>> cancelled,
    absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) {
  if (callback == nullptr) {
    callback = [](absl::StatusOr<Responses> responses) {};
  }

  auto task = [this, task_id, constraint, cancelled]() mutable -> void {
    auto session_info_and_callback = StartTask(task_id);
    if (!session_info_and_callback.ok()) {
      ABSL_LOG(ERROR) << "Failed to start task: "
                      << session_info_and_callback.status()
                      << " with task id: " << task_id;
      return;
    }
    auto [session_info, callback] =
        std::move(session_info_and_callback.value());
    // If the session info is nullptr, it means the task is cancelled before it
    // is started.
    if (session_info == nullptr) {
      return;
    }

    auto llm_executor = resource_manager_->AcquireExecutorWithContextHandler(
        session_info->context_handler);
    if (!llm_executor.ok()) {
      callback(llm_executor.status());
      return;
    }

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

    auto status =
        FinishTask(task_id, std::move(responses), std::move(callback));
    if (!status.ok()) {
      ABSL_LOG(ERROR) << "Failed to finish task: " << status
                      << " with task id: " << task_id;
    }
    return;
  };

  return CreateTask(session_id, task_id, std::move(task), std::move(dep_tasks),
                    std::move(callback));
}

absl::Status ExecutionManager::AddCloneSessionTask(
    SessionId session_id, TaskId task_id, absl::flat_hash_set<TaskId> dep_tasks,
    SessionId cloned_session_id,
    absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) {
  if (callback == nullptr) {
    callback = [](absl::StatusOr<Responses> responses) {};
  }

  auto task = [this, task_id, cloned_session_id]() mutable -> void {
    auto session_info_and_callback = StartTask(task_id);
    if (!session_info_and_callback.ok()) {
      ABSL_LOG(ERROR) << "Failed to start task: "
                      << session_info_and_callback.status()
                      << " with task id: " << task_id;
      return;
    }
    auto [session_info, callback] =
        std::move(session_info_and_callback.value());
    // If the session info is nullptr, it means the task is cancelled before it
    // is started.
    if (session_info == nullptr) {
      return;
    }

    {
      absl::MutexLock lock(session_and_task_lookup_mutex_);
      if (!session_lookup_.contains(cloned_session_id)) {
        callback(absl::InvalidArgumentError(
            absl::StrCat("Cloned session ", cloned_session_id,
                         " not found in session list.")));
        return;
      }
      auto cloned_context_handler =
          resource_manager_->CloneContextHandler(session_info->context_handler);
      if (!cloned_context_handler.ok()) {
        callback(cloned_context_handler.status());
        return;
      }
      std::unique_ptr<Sampler> cloned_sampler;
      if (session_info->sampler != nullptr) {
        auto sampler =
            CreateSampler(session_info->session_config.GetSamplerBackend(),
                          session_info->session_config.GetNumOutputCandidates(),
                          session_info->session_config.GetSamplerParams());
        if (!sampler.ok()) {
          callback(sampler.status());
          return;
        }
        cloned_sampler = std::move(*sampler);
      }
      auto cloned_stop_token_detector = std::make_unique<StopTokenDetector>(1);
      for (const auto& stop_token_sequence :
           session_info->session_config.GetStopTokenIds()) {
        auto status = cloned_stop_token_detector->AddStopTokenSequence(
            stop_token_sequence);
        if (!status.ok()) {
          ABSL_LOG(ERROR) << "Failed to add stop token sequence: " << status;
        }
      }
      session_lookup_.at(cloned_session_id)->session_config =
          session_info->session_config;
      session_lookup_.at(cloned_session_id)->context_handler =
          std::move(cloned_context_handler.value());
      session_lookup_.at(cloned_session_id)->sampler =
          std::move(cloned_sampler);
      session_lookup_.at(cloned_session_id)->last_prefill_token_id =
          session_info->last_prefill_token_id;
      session_lookup_.at(cloned_session_id)->stop_token_detector =
          std::move(cloned_stop_token_detector);
      session_lookup_.at(cloned_session_id)->benchmark_info =
          session_info->benchmark_info;
    }

    auto status =
        FinishTask(task_id, Responses(TaskState::kDone), std::move(callback));
    if (!status.ok()) {
      ABSL_LOG(ERROR) << "Failed to finish task: " << status
                      << " with task id: " << task_id;
    }
    return;
  };

  return CreateTask(session_id, task_id, std::move(task), std::move(dep_tasks),
                    std::move(callback));
}

}  // namespace litert::lm
