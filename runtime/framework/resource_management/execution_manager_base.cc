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

#include "runtime/framework/resource_management/execution_manager_base.h"

#include <atomic>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/constrained_decoding/constraint.h"
#include "runtime/components/sampler.h"
#include "runtime/components/sampler_factory.h"
#include "runtime/components/stop_token_detector.h"
#include "runtime/components/tokenizer.h"
#include "runtime/core/tasks.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/audio_executor.h"
#include "runtime/executor/llm_executor.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/framework/resource_management/execution_manager.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/executor_data_util.h"
#include "runtime/util/status_macros.h"
#include "runtime/util/tensor_buffer_util.h"

namespace litert::lm {

#define RETURN_IF_CANCELLED(cancelled, task_id, callback)             \
  if (cancelled != nullptr && cancelled->load()) {                    \
    FinishTaskAndLogErrors(task_id, Responses(TaskState::kCancelled), \
                           std::move(callback));                      \
    return;                                                           \
  }

absl::StatusOr<std::shared_ptr<SessionInfo>>
ExecutionManagerBase::CreateSessionInfo(
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

  auto session_info = std::make_shared<SessionInfo>(SessionInfo{
      .session_config = std::move(session_config),
      .context_handler = std::move(context_handler),
      .sampler = std::move(sampler),
      .stop_token_detector = std::move(stop_token_detector),
      .benchmark_info = std::move(benchmark_info),
  });

  if (session_info->session_config.AudioModalityEnabled()) {
    RETURN_IF_ERROR(resource_manager_->TryLoadingAudioExecutor());
  }
  if (session_info->session_config.VisionModalityEnabled()) {
    RETURN_IF_ERROR(resource_manager_->TryLoadingVisionExecutor());
  }

  return session_info;
}

absl::StatusOr<SessionId> ExecutionManagerBase::RegisterNewSessionUnlocked(
    std::shared_ptr<SessionInfo> session_info) {
  SessionId session_id = next_session_id_++;
  if (session_lookup_.contains(session_id)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Session ", session_id, " already exists in session list."));
  }
  session_lookup_.insert({session_id, std::move(session_info)});

  return session_id;
}

absl::Status ExecutionManagerBase::ReleaseSessionUnlocked(
    SessionId session_id) {
  if (!session_lookup_.contains(session_id)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Session ", session_id, " not found in session list."));
  }
  // If the session is the only session and it is audio modality enabled, we
  // need to reset the audio executor, so the streaming audio executor would
  // have a clean state for the next usage.
  if (session_lookup_.at(session_id)->session_config.AudioModalityEnabled() &&
      session_lookup_.size() == 1) {
    ASSIGN_OR_RETURN(auto audio_executor,
                     resource_manager_->AcquireAudioExecutor());
    audio_executor->Reset().IgnoreError();
  }
  session_lookup_.erase(session_id);
  return absl::OkStatus();
}

absl::Status ExecutionManagerBase::CancelAllTasksInSessionUnlocked(
    SessionId session_id) {
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
ExecutionManagerBase::GetSessionInfoUnlocked(SessionId session_id) {
  if (!session_lookup_.contains(session_id)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Session ", session_id, " not found in session list."));
  }
  return session_lookup_.at(session_id);
}

absl::StatusOr<BenchmarkInfo*>
ExecutionManagerBase::GetMutableBenchmarkInfoUnlocked(SessionId session_id) {
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

absl::Status ExecutionManagerBase::CreateTaskUnlocked(
    SessionId session_id, TaskId task_id,
    absl::AnyInvocable<void()> absl_nonnull task,
    absl::flat_hash_set<TaskId> dependent_tasks,
    std::shared_ptr<std::atomic<bool>> absl_nonnull cancelled,
    absl::AnyInvocable<void(absl::StatusOr<Responses>)> absl_nonnull callback) {
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

absl::StatusOr<
    std::tuple<std::shared_ptr<SessionInfo>, std::shared_ptr<std::atomic<bool>>,
               absl::AnyInvocable<void(absl::StatusOr<Responses>)>>>
ExecutionManagerBase::StartTaskUnlocked(TaskId task_id) {
  if (!task_lookup_.contains(task_id)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Task ", task_id, " not found in task list."));
  }
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
  RETURN_IF_ERROR(UpdateTaskStateUnlocked(task_id, TaskState::kProcessing));

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

absl::Status ExecutionManagerBase::FinishTaskUnlocked(
    TaskId task_id, absl::StatusOr<Responses> responses,
    absl::AnyInvocable<void(absl::StatusOr<Responses>)>& callback) {
  auto invoke_callback_and_return = [&](absl::Status status) -> absl::Status {
    callback(status);
    RETURN_IF_ERROR(UpdateTaskStateUnlocked(task_id, TaskState::kFailed));
    return status;
  };

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
    auto following_waiting_tasks = FollowingWaitingTasksUnlocked(task_id);
    if (!following_waiting_tasks.ok()) {
      return invoke_callback_and_return(following_waiting_tasks.status());
    }
    auto status = UpdateAllTasksToStateUnlocked(
        following_waiting_tasks.value(),
        responses.ok() ? TaskState::kDependentTaskCancelled
                       : TaskState::kDependentTaskFailed);
    if (!status.ok()) {
      return invoke_callback_and_return(status);
    }
  } else if (responses->GetTaskState() == TaskState::kDone ||
             responses->GetTaskState() == TaskState::kMaxNumTokensReached) {
    for (TaskId following_task_id : task_lookup_.at(task_id).following_tasks) {
      if (!task_lookup_.contains(following_task_id)) {
        auto error_status = absl::InvalidArgumentError(absl::StrCat(
            "Following task ", following_task_id, " not found in task list."));
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

  return absl::OkStatus();
}

absl::StatusOr<absl::flat_hash_set<TaskId>>
ExecutionManagerBase::FollowingWaitingTasksUnlocked(TaskId task_id) {
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
                       FollowingWaitingTasksUnlocked(following_task_id));
      following_waiting_tasks.insert(next_following_waiting_tasks.begin(),
                                     next_following_waiting_tasks.end());
    }
  }
  return following_waiting_tasks;
}

absl::Status ExecutionManagerBase::UpdateTaskStateUnlocked(
    TaskId task_id, TaskState task_state) {
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

absl::Status ExecutionManagerBase::UpdateAllTasksToStateUnlocked(
    const absl::flat_hash_set<TaskId>& task_ids, TaskState task_state) {
  for (TaskId task_id : task_ids) {
    task_lookup_.at(task_id).dependent_tasks.clear();
    if (task_lookup_.at(task_id).callback) {
      task_lookup_.at(task_id).callback(Responses(task_state));
    }
    RETURN_IF_ERROR(UpdateTaskStateUnlocked(task_id, task_state));
  }
  return absl::OkStatus();
}

absl::StatusOr<ExecutorInputs> ExecutionManagerBase::ProcessAndCombineContents(
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
      if (benchmark_info.has_value()) {
        RETURN_IF_ERROR(benchmark_info->TimeMarkDelta("vision_executor"));
      }
      ExecutorVisionData single_image_data;
      if (input_image->IsTensorBuffer()) {
        ASSIGN_OR_RETURN(auto tensor_buffer,
                         input_image->GetPreprocessedImageTensor());
        ASSIGN_OR_RETURN(auto vision_executor,
                         resource_manager_->AcquireVisionExecutor());
        ASSIGN_OR_RETURN(single_image_data,
                         vision_executor->Encode(*tensor_buffer));
      } else if (input_image->IsTensorBufferMap()) {
        ASSIGN_OR_RETURN(auto tensor_buffer_map,
                         input_image->GetPreprocessedImageTensorMap());
        ASSIGN_OR_RETURN(auto vision_executor,
                         resource_manager_->AcquireVisionExecutor());
        ASSIGN_OR_RETURN(single_image_data,
                         vision_executor->Encode(*tensor_buffer_map));
      } else {
        return absl::FailedPreconditionError(
            "Image tensor or tensor map is null in preprocessed_contents.");
      }
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
    } else if (const auto* input_image_end =
                   std::get_if<InputImageEnd>(&preprocessed_content)) {
      combined_token_ids.push_back(ExecutorVisionData::kEndToken);
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

absl::StatusOr<AudioExecutorProperties>
ExecutionManagerBase::GetAudioExecutorProperties() const {
  return resource_manager_->GetAudioExecutorProperties();
}

absl::StatusOr<VisionExecutorProperties>
ExecutionManagerBase::GetVisionExecutorProperties() const {
  return resource_manager_->GetVisionExecutorProperties();
}

absl::Status ExecutionManagerBase::AddPrefillTask(
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

    absl::StatusOr<Responses> responses;
    {
      // Note AcquireExecutorWithContextHandler includes context switching
      // logic, so it should be called before any executor running.
      auto llm_executor = resource_manager_->AcquireExecutorWithContextHandler(
          session_info->context_handler);
      if (!llm_executor.ok()) {
        FinishTaskAndLogErrors(task_id, llm_executor.status(),
                               std::move(callback));
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

      responses =
          Tasks::Prefill(*llm_executor.value(), *executor_inputs,
                         /*wait_for_completion=*/true,
                         /*benchmark_info=*/session_info->benchmark_info);
      if (!responses.ok()) {
        FinishTaskAndLogErrors(task_id, responses.status(),
                               std::move(callback));
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
    }

    FinishTaskAndLogErrors(task_id, std::move(responses), std::move(callback));
    return;
  };

  return CreateTask(session_id, task_id, std::move(task), std::move(dep_tasks),
                    cancelled, std::move(callback));
}

absl::Status ExecutionManagerBase::AddDecodeTask(
    SessionId session_id, TaskId task_id, absl::flat_hash_set<TaskId> dep_tasks,
    Constraint* absl_nullable constraint,
    std::shared_ptr<std::atomic<bool>> absl_nonnull cancelled,
    absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback,
    int max_output_tokens) {
  if (callback == nullptr) {
    callback = [](absl::StatusOr<Responses> responses) {};
  }

  auto task = [this, task_id, constraint, cancelled,
               max_output_tokens]() mutable -> void {
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

    absl::StatusOr<Responses> responses;
    {
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
          FinishTaskAndLogErrors(
              task_id,
              absl::InternalError(decoded_ids_buffer_or.Error().Message()),
              std::move(callback));
          return;
        }
        decoded_ids_buffer = std::move(decoded_ids_buffer_or.Value());
      }

      responses = Tasks::Decode(
          *llm_executor.value(), *tokenizer_,
          *session_info->stop_token_detector, num_output_candidates,
          session_info->benchmark_info, optional_sampler, constraint,
          std::move(decoded_ids_buffer), callback, cancelled.get(),
          max_output_tokens);
      if (!responses.ok() && absl::IsCancelled(responses.status())) {
        responses = Responses(TaskState::kCancelled);
      }

      if (cancelled != nullptr && cancelled->load()) {
        responses = Responses(TaskState::kCancelled);
      }
    }

    FinishTaskAndLogErrors(task_id, std::move(responses), std::move(callback));
    return;
  };

  return CreateTask(session_id, task_id, std::move(task), std::move(dep_tasks),
                    cancelled, std::move(callback));
}

absl::Status ExecutionManagerBase::AddTextScoringTask(
    SessionId session_id, TaskId task_id, absl::flat_hash_set<TaskId> dep_tasks,
    const std::vector<absl::string_view>& target_text, bool store_token_lengths,
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

    absl::StatusOr<Responses> responses;
    {
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
      responses = Tasks::Score(
          *llm_executor.value(), *tokenizer_, target_text, temperature,
          std::move(decoded_ids_buffer.Value()), store_token_lengths);

      if (cancelled != nullptr && cancelled->load()) {
        responses = Responses(TaskState::kCancelled);
      }
    }

    FinishTaskAndLogErrors(task_id, std::move(responses), std::move(callback));
    return;
  };

  return CreateTask(session_id, task_id, std::move(task), std::move(dep_tasks),
                    cancelled, std::move(callback));
}

// TODO b/409401231 - Add unit tests for this function.
absl::Status ExecutionManagerBase::AddCloneSessionTask(
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

    auto status = CloneSession(session_id, cloned_session_id);
    absl::StatusOr<Responses> responses;
    if (!status.ok()) {
      responses = status;
    } else if (cancelled != nullptr && cancelled->load()) {
      responses = Responses(TaskState::kCancelled);
    } else {
      responses = Responses(TaskState::kDone);
    }

    FinishTaskAndLogErrors(task_id, std::move(responses), std::move(callback));
  };

  return CreateTask(session_id, task_id, std::move(task), std::move(dep_tasks),
                    cancelled, std::move(callback));
}

absl::StatusOr<int> ExecutionManagerBase::GetCurrentStepUnlocked(
    const SessionInfo& session_info) {
  ASSIGN_OR_RETURN(auto llm_executor,
                   resource_manager_->AcquireExecutorWithContextHandler(
                       session_info.context_handler));
  return llm_executor->GetCurrentStep();
}

absl::Status ExecutionManagerBase::SetCurrentStepUnlocked(
    const SessionInfo& session_info, int target_step) {
  ASSIGN_OR_RETURN(auto llm_executor,
                   resource_manager_->AcquireExecutorWithContextHandler(
                       session_info.context_handler));
  ASSIGN_OR_RETURN(int current_step, llm_executor->GetCurrentStep());
  if (target_step > current_step) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Target step is greater than the current step: ", current_step));
  }
  return llm_executor->SetCurrentStep(target_step);
}

absl::Status ExecutionManagerBase::CloneSessionUnlocked(
    SessionId session_id, SessionId cloned_session_id) {
  if (!session_lookup_.contains(session_id)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Session ", session_id, " not found in session list."));
  }
  if (!session_lookup_.contains(cloned_session_id)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Cloned session ", cloned_session_id, " not found in session list."));
  }
  const SessionInfo& source = *session_lookup_.at(session_id);
  SessionInfo& target = *session_lookup_.at(cloned_session_id);

  auto cloned_context_handler =
      resource_manager_->CloneContextHandler(source.context_handler);
  if (!cloned_context_handler.ok()) {
    return cloned_context_handler.status();
  }

  std::unique_ptr<Sampler> cloned_sampler;
  if (source.sampler != nullptr) {
    ASSIGN_OR_RETURN(
        cloned_sampler,
        CreateSampler(source.session_config.GetSamplerBackend(),
                      source.session_config.GetNumOutputCandidates(),
                      source.session_config.GetSamplerParams(),
                      litert_env_ ? litert_env_->Get() : nullptr));
  }

  auto cloned_stop_token_detector = std::make_unique<StopTokenDetector>(1);
  for (const auto& stop_token_sequence :
       source.session_config.GetStopTokenIds()) {
    auto status =
        cloned_stop_token_detector->AddStopTokenSequence(stop_token_sequence);
    if (!status.ok()) {
      return status;
    }
  }

  target.session_config = source.session_config;
  target.context_handler = std::move(cloned_context_handler.value());
  target.sampler = std::move(cloned_sampler);
  target.last_prefill_token_id = source.last_prefill_token_id;
  target.stop_token_detector = std::move(cloned_stop_token_detector);
  target.benchmark_info = source.benchmark_info;

  return absl::OkStatus();
}

}  // namespace litert::lm
