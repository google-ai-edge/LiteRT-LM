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

#include "runtime/core/session_advanced.h"

#include <algorithm>
#include <atomic>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/log/log.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "runtime/components/tokenizer.h"
#include "runtime/core/session_utils.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/framework/resource_management/execution_manager.h"
#include "runtime/proto/sampler_params.pb.h"
#include "runtime/util/status_macros.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

using TaskController = SessionInterface::TaskController;

}  // namespace

// static
absl::StatusOr<std::unique_ptr<SessionAdvanced>> SessionAdvanced::Create(
    std::weak_ptr<ExecutionManager> execution_manager,
    Tokenizer* absl_nonnull tokenizer, const SessionConfig& session_config,
    std::optional<BenchmarkInfo> benchmark_info,
    std::atomic<int>* living_sessions_count) {
  auto execution_manager_lock = execution_manager.lock();
  if (execution_manager_lock == nullptr) {
    return absl::FailedPreconditionError("Execution manager is not available.");
  }
  ASSIGN_OR_RETURN(auto session_id, execution_manager_lock->RegisterNewSession(
                                        session_config, benchmark_info));
  ASSIGN_OR_RETURN(auto session_info_,
                   execution_manager_lock->GetSessionInfo(session_id));
  return absl::WrapUnique(new SessionAdvanced(
      session_id, execution_manager, tokenizer, session_info_,
      /*session_state=*/SessionState::kFresh,
      /*last_task_ids=*/{}, living_sessions_count));
}

absl::Status SessionAdvanced::RunPrefill(
    const std::vector<InputData>& contents) {
  absl::Status status = absl::OkStatus();
  ASSIGN_OR_RETURN(
      auto task_controller,
      RunPrefillAsync(contents, [&status](absl::StatusOr<Responses> responses) {
        status = responses.status();
      }));
  RETURN_IF_ERROR(task_controller->WaitUntilDone(Engine::kDefaultTimeout));
  return status;
}

absl::StatusOr<std::unique_ptr<TaskController>>
SessionAdvanced::RunPrefillAsync(
    const std::vector<InputData>& contents,
    absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) {
  if (contents.empty()) {
    return absl::InvalidArgumentError("Input is empty.");
  }
  absl::MutexLock lock(mutex_);
  auto cancelled = std::make_shared<std::atomic<bool>>(false);

  auto execution_manager_lock = execution_manager_.lock();
  if (execution_manager_lock == nullptr) {
    return absl::FailedPreconditionError("Execution manager is not available.");
  }

  std::vector<InputData> preprocessed_contents;
  if (session_info_->benchmark_info.has_value() &&
      session_info_->benchmark_info->GetBenchmarkParams().num_prefill_tokens() >
          0) {
    ASSIGN_OR_RETURN(
        preprocessed_contents,
        PreprocessContents(contents, session_info_->session_config, *tokenizer_,
                           session_info_->benchmark_info));
  } else {
    bool is_first_turn = session_state_ == SessionState::kFresh;
    ContentType content_type;
    if (session_info_->session_config.GetApplyPromptTemplateInSession()) {
      content_type = (is_first_turn || session_state_ == SessionState::kDecoded)
                         ? ContentType::kFirst
                         : ContentType::kMiddle;
    } else {
      content_type = ContentType::kNA;
    }
    ASSIGN_OR_RETURN(std::vector<InputData> templated_contents,
                     ApplyPromptTemplates(contents, content_type,
                                          session_info_->session_config,
                                          *tokenizer_, is_first_turn));
    ASSIGN_OR_RETURN(
        preprocessed_contents,
        PreprocessContents(templated_contents, session_info_->session_config,
                           *tokenizer_, session_info_->benchmark_info));
  }
  ASSIGN_OR_RETURN(auto task_id, execution_manager_lock->GetNewTaskId());
  RETURN_IF_ERROR(execution_manager_lock->AddPrefillTask(
      session_id_, task_id, std::move(preprocessed_contents), last_task_ids_,
      cancelled, std::move(callback)));
  session_state_ = SessionState::kPrefilled;
  last_task_ids_ = {task_id};

  return std::make_unique<AdvancedTaskController>(task_id, cancelled,
                                                  execution_manager_);
}

absl::StatusOr<Responses> SessionAdvanced::RunDecode() {
  return RunDecode(DecodeConfig::CreateDefault());
}

absl::StatusOr<Responses> SessionAdvanced::RunDecode(
    const DecodeConfig& decode_config) {
  auto execution_manager_lock = execution_manager_.lock();
  if (execution_manager_lock == nullptr) {
    return absl::FailedPreconditionError("Execution manager is not available.");
  }

  absl::StatusOr<Responses> collected_responses;
  int num_candidates = session_info_->session_config.GetNumOutputCandidates();
  collected_responses =
      Responses(TaskState::kCreated,
                /*response_texts=*/std::vector<std::string>(num_candidates),
                /*scores=*/std::vector<float>(num_candidates),
                /*token_lengths=*/{},
                /*token_ids=*/std::vector<std::vector<int>>(num_candidates));
  int num_decode_tokens = 0;
  auto decode_sync_callback = [&collected_responses, &num_decode_tokens](
                                  absl::StatusOr<Responses> responses) {
    if (!collected_responses.ok()) return;
    if (!responses.ok()) {
      collected_responses = responses.status();
      return;
    }
    collected_responses->SetTaskState(responses->GetTaskState());
    // If the task is not completed and there is no text or score, we can
    // return early.
    if (!IsTaskEndState(responses->GetTaskState()) &&
        responses->GetTexts().empty() && responses->GetScores().empty()) {
      return;
    }
    // Accumulating the scores if it is provided.
    if (collected_responses->GetMutableScores().size() ==
        responses->GetScores().size()) {
      for (int i = 0; i < responses->GetScores().size(); ++i) {
        collected_responses->GetMutableScores()[i] += responses->GetScores()[i];
      }
    }
    // Accumulating the texts.
    if (collected_responses->GetMutableTexts().size() ==
        responses->GetTexts().size()) {
      num_decode_tokens += 1;
      for (int i = 0; i < responses->GetTexts().size(); ++i) {
        collected_responses->GetMutableTexts()[i] += responses->GetTexts()[i];
      }
    } else if (!responses->GetTexts().empty()) {
      collected_responses = absl::InternalError(
          absl::StrCat("Decode responses size mismatch: ",
                       collected_responses->GetTexts().size(), " vs ",
                       responses->GetTexts().size()));
    }
    // Accumulating the token IDs.
    if (collected_responses->GetMutableTokenIds().size() ==
        responses->GetTokenIds().size()) {
      for (int i = 0; i < responses->GetTokenIds().size(); ++i) {
        collected_responses->GetMutableTokenIds()[i].insert(
            collected_responses->GetMutableTokenIds()[i].end(),
            responses->GetTokenIds()[i].begin(),
            responses->GetTokenIds()[i].end());
      }
    }
    // Normalizing the scores by the number of decode tokens if the task is
    // completed.
    if (IsTaskEndState(responses->GetTaskState())) {
      for (int i = 0; i < responses->GetScores().size(); ++i) {
        collected_responses->GetMutableScores()[i] /=
            std::max(1, num_decode_tokens);
      }
    }
  };

  ASSIGN_OR_RETURN(
      auto task_controller,
      RunDecodeAsync(std::move(decode_sync_callback), decode_config));
  RETURN_IF_ERROR(task_controller->WaitUntilDone(Engine::kDefaultTimeout));
  return collected_responses;
}

absl::StatusOr<std::unique_ptr<TaskController>> SessionAdvanced::RunDecodeAsync(
    absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) {
  return RunDecodeAsync(std::move(callback), DecodeConfig::CreateDefault());
}

absl::StatusOr<std::unique_ptr<TaskController>> SessionAdvanced::RunDecodeAsync(
    absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback,
    const DecodeConfig& decode_config) {
  absl::MutexLock lock(mutex_);
  if (session_state_ != SessionState::kPrefilled) {
    return absl::InternalError("Session is not prefilled yet.");
  }

  auto cancelled = std::make_shared<std::atomic<bool>>(false);

  auto execution_manager_lock = execution_manager_.lock();
  if (execution_manager_lock == nullptr) {
    return absl::FailedPreconditionError("Execution manager is not available.");
  }

  // We need to do a last prefill before initializing the decode, to make sure
  // the prompt is correctly set up for decode.
  if (session_info_->session_config.GetApplyPromptTemplateInSession()) {
    std::vector<InputData> contents;
    contents.emplace_back(InputText(""));
    ASSIGN_OR_RETURN(
        std::vector<InputData> templated_contents,
        ApplyPromptTemplates(contents, ContentType::kLast,
                             session_info_->session_config, *tokenizer_,
                             /*is_first_turn=*/false));
    if (!templated_contents.empty()) {
      ASSIGN_OR_RETURN(
          std::vector<InputData> preprocessed_contents,
          PreprocessContents(templated_contents, session_info_->session_config,
                             *tokenizer_, session_info_->benchmark_info));
      auto noop_callback = [](absl::StatusOr<Responses> responses) {};
      ASSIGN_OR_RETURN(auto task_id, execution_manager_lock->GetNewTaskId());
      RETURN_IF_ERROR(execution_manager_lock->AddPrefillTask(
          session_id_, task_id, std::move(preprocessed_contents),
          last_task_ids_, cancelled, std::move(noop_callback)));
      last_task_ids_ = {task_id};
    }
  }
  session_state_ = SessionState::kDecoded;

  ASSIGN_OR_RETURN(auto task_id, execution_manager_lock->GetNewTaskId());

  RETURN_IF_ERROR(execution_manager_lock->AddDecodeTask(
      session_id_, task_id, last_task_ids_, decode_config.GetConstraint(),
      cancelled, std::move(callback),
      decode_config.GetMaxOutputTokens().value_or(
          session_info_->session_config.GetMaxOutputTokens()),
      decode_config.GetThinkingTokenBudget(),
      decode_config.GetThinkingStartTokenIds(),
      decode_config.GetThinkingEndTokenIds()));

  last_task_ids_ = {task_id};

  return std::make_unique<AdvancedTaskController>(task_id, cancelled,
                                                  execution_manager_);
}

absl::StatusOr<Responses> SessionAdvanced::RunTextScoring(
    const std::vector<absl::string_view>& target_text,
    bool store_token_lengths) {
  if (target_text.size() != 1) {
    // Batch scoring is not supported yet.
    return absl::InvalidArgumentError("Target text size should be 1.");
  }
  auto execution_manager_lock = execution_manager_.lock();
  if (execution_manager_lock == nullptr) {
    return absl::FailedPreconditionError("Execution manager is not available.");
  }

  absl::StatusOr<Responses> collected_responses;
  auto scoring_sync_callback =
      [&collected_responses](absl::StatusOr<Responses> responses) {
        collected_responses = std::move(responses);
      };

  ASSIGN_OR_RETURN(
      auto task_controller,
      RunTextScoringAsync(target_text, std::move(scoring_sync_callback),
                          store_token_lengths));
  RETURN_IF_ERROR(task_controller->WaitUntilDone(Engine::kDefaultTimeout));
  return collected_responses;
}

absl::StatusOr<std::unique_ptr<SessionInterface::TaskController>>
SessionAdvanced::RunTextScoringAsync(
    const std::vector<absl::string_view>& target_text,
    absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback,
    bool store_token_lengths) {
  absl::MutexLock lock(mutex_);
  if (target_text.size() != 1) {
    return absl::InvalidArgumentError("Target text size should be 1.");
  }
  auto execution_manager_lock = execution_manager_.lock();
  if (execution_manager_lock == nullptr) {
    return absl::FailedPreconditionError("Execution manager is not available.");
  }

  auto cancelled = std::make_shared<std::atomic<bool>>(false);
  ASSIGN_OR_RETURN(auto task_id, execution_manager_lock->GetNewTaskId());
  RETURN_IF_ERROR(execution_manager_lock->AddTextScoringTask(
      session_id_, task_id, last_task_ids_, target_text, store_token_lengths,
      cancelled, std::move(callback)));

  return std::make_unique<AdvancedTaskController>(task_id, cancelled,
                                                  execution_manager_);
}

absl::StatusOr<Responses> SessionAdvanced::GenerateContent(
    const std::vector<InputData>& contents) {
  RETURN_IF_ERROR(RunPrefill(contents));
  return RunDecode();
}

absl::Status SessionAdvanced::GenerateContentStream(
    const std::vector<InputData>& contents,
    absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) {
  return GenerateContentStream(contents, std::move(callback),
                               DecodeConfig::CreateDefault());
}

absl::Status SessionAdvanced::GenerateContentStream(
    const std::vector<InputData>& contents,
    absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback,
    const DecodeConfig& decode_config) {
  absl::AnyInvocable<void(absl::StatusOr<Responses>)> prefill_callback =
      [this, decode_config, stream_callback = std::move(callback)](
          absl::StatusOr<Responses> prefill_responses) mutable {
        if (!prefill_responses.ok()) {
          stream_callback(prefill_responses.status());
          return;
        }
        if (prefill_responses->GetTaskState() == TaskState::kDone) {
          auto decode_task_controller =
              RunDecodeAsync(std::move(stream_callback), decode_config);
          if (!decode_task_controller.ok()) {
            ABSL_LOG(ERROR) << "Failed to start decode task: "
                            << decode_task_controller.status();
          }
        } else if (IsTaskEndState(prefill_responses->GetTaskState())) {
          stream_callback(absl::CancelledError(
              "Prefill task finished in cancelled state."));
        }
      };

  ASSIGN_OR_RETURN(auto task_controller,
                   RunPrefillAsync(contents, std::move(prefill_callback)));

  return absl::OkStatus();
}

absl::StatusOr<BenchmarkInfo> SessionAdvanced::GetBenchmarkInfo() {
  absl::MutexLock lock(mutex_);
  if (session_info_->benchmark_info.has_value()) {
    return session_info_->benchmark_info.value();
  }
  return absl::InternalError(
      "Benchmark is not enabled. Please make sure the BenchmarkParams is set "
      "in the EngineSettings.");
}

absl::StatusOr<BenchmarkInfo*> SessionAdvanced::GetMutableBenchmarkInfo() {
  auto execution_manager_lock = execution_manager_.lock();
  if (execution_manager_lock == nullptr) {
    return absl::FailedPreconditionError("Execution manager is not available.");
  }
  return execution_manager_lock->GetMutableBenchmarkInfo(session_id_);
}

absl::StatusOr<std::unique_ptr<SessionInterface>> SessionAdvanced::Clone() {
  absl::Status status = absl::OkStatus();
  std::unique_ptr<SessionInterface> session;
  {
    absl::MutexLock lock(mutex_);
    ASSIGN_OR_RETURN(
        session,
        CloneAsyncLocked([&status](absl::StatusOr<Responses> responses) {
          status = responses.status();
        }));
  }
  RETURN_IF_ERROR(WaitUntilDone());
  RETURN_IF_ERROR(status);
  return session;
}

absl::StatusOr<std::unique_ptr<SessionInterface>> SessionAdvanced::CloneAsync(
    absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) {
  absl::MutexLock lock(mutex_);
  return CloneAsyncLocked(std::move(callback));
}

absl::StatusOr<std::unique_ptr<SessionInterface>>
SessionAdvanced::CloneAsyncLocked(
    absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) {
  auto execution_manager_lock = execution_manager_.lock();
  if (execution_manager_lock == nullptr) {
    return absl::FailedPreconditionError("Execution manager is not available.");
  }

  ASSIGN_OR_RETURN(auto task_id, execution_manager_lock->GetNewTaskId());

  ASSIGN_OR_RETURN(auto session_id, execution_manager_lock->RegisterNewSession(
                                        session_info_->session_config,
                                        session_info_->benchmark_info));

  RETURN_IF_ERROR(execution_manager_lock->AddCloneSessionTask(
      session_id_, task_id, last_task_ids_, session_id,
      std::make_shared<std::atomic<bool>>(false), std::move(callback)));

  last_task_ids_ = {task_id};

  ASSIGN_OR_RETURN(auto session_info,
                   execution_manager_lock->GetSessionInfo(session_id));

  return absl::WrapUnique(new SessionAdvanced(session_id, execution_manager_,
                                              tokenizer_, session_info,
                                              session_state_, last_task_ids_));
}

SessionAdvanced::~SessionAdvanced() {
  WaitUntilDone().IgnoreError();
  auto execution_manager_lock = execution_manager_.lock();
  if (execution_manager_lock == nullptr) {
    ABSL_LOG(ERROR) << "Execution manager is not available.";
    return;
  }
  auto status = execution_manager_lock->ReleaseSession(session_id_);
  if (!status.ok()) {
    ABSL_LOG(ERROR) << "Error occurred when releasing session: " << status;
  }
  if (living_sessions_count_) {
    (*living_sessions_count_)--;
  }
};

absl::Status SessionAdvanced::SaveCheckpoint(absl::string_view label) {
  absl::MutexLock lock(mutex_);
  auto execution_manager_lock = execution_manager_.lock();
  if (execution_manager_lock == nullptr) {
    return absl::FailedPreconditionError("Execution manager is not available.");
  }
  ASSIGN_OR_RETURN(int current_step,
                   execution_manager_lock->GetCurrentStep(*session_info_));
  checkpoint_map_[label] = {current_step, session_state_};
  return absl::OkStatus();
}

absl::Status SessionAdvanced::RewindToCheckpoint(absl::string_view label) {
  absl::MutexLock lock(mutex_);

  // Look up the checkpoint step.
  auto it = checkpoint_map_.find(label);
  if (it == checkpoint_map_.end()) {
    return absl::NotFoundError(absl::StrCat("Checkpoint not found: ", label));
  }
  int target_step = it->second.step;
  session_state_ = it->second.state;

  // Remove all checkpoints after the current step.
  absl::erase_if(checkpoint_map_, [target_step](const auto& pair) {
    return pair.second.step > target_step;
  });

  // Get the execution manager and set the current step.
  auto execution_manager_lock = execution_manager_.lock();
  if (execution_manager_lock == nullptr) {
    return absl::FailedPreconditionError("Execution manager is not available.");
  }
  return execution_manager_lock->SetCurrentStep(*session_info_, target_step);
}

absl::Status SessionAdvanced::RewindToStep(int step) {
  absl::MutexLock lock(mutex_);
  auto execution_manager_lock = execution_manager_.lock();
  if (execution_manager_lock == nullptr) {
    return absl::FailedPreconditionError("Execution manager is not available.");
  }
  if (step > 0) {
    session_state_ = SessionState::kPrefilled;
  } else {
    session_state_ = SessionState::kFresh;
  }

  // Remove all checkpoints after the current step.
  absl::erase_if(checkpoint_map_,
                 [step](const auto& pair) { return pair.second.step > step; });

  return execution_manager_lock->SetCurrentStep(*session_info_, step);
}

absl::StatusOr<int> SessionAdvanced::GetCurrentStep() const {
  auto execution_manager_lock = execution_manager_.lock();
  if (execution_manager_lock == nullptr) {
    return absl::FailedPreconditionError("Execution manager is not available.");
  }
  return execution_manager_lock->GetCurrentStep(*session_info_);
}

}  // namespace litert::lm
