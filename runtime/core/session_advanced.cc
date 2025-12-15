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
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/components/tokenizer.h"
#include "runtime/core/session_utils.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/framework/resource_management/execution_manager.h"
#include "runtime/proto/sampler_params.pb.h"
#include "runtime/util/status_macros.h"  // IWYU pragma: keep

namespace litert::lm {

// static
absl::StatusOr<std::unique_ptr<SessionAdvanced>> SessionAdvanced::Create(
    std::weak_ptr<ExecutionManager> execution_manager,
    Tokenizer* absl_nonnull tokenizer, const SessionConfig& session_config,
    std::optional<BenchmarkInfo> benchmark_info) {
  auto execution_manager_lock = execution_manager.lock();
  if (execution_manager_lock == nullptr) {
    return absl::FailedPreconditionError("Execution manager is not available.");
  }
  ASSIGN_OR_RETURN(auto session_id, execution_manager_lock->RegisterNewSession(
                                        session_config, benchmark_info));
  ASSIGN_OR_RETURN(auto session_info_,
                   execution_manager_lock->GetSessionInfo(session_id));
  return absl::WrapUnique(new SessionAdvanced(session_id, execution_manager,
                                              tokenizer, session_info_));
}

absl::Status SessionAdvanced::RunPrefill(
    const std::vector<InputData>& contents) {
  absl::Status status = absl::OkStatus();
  RETURN_IF_ERROR(
      RunPrefillAsync(contents, [&status](absl::StatusOr<Responses> responses) {
        status = responses.status();
      }));
  auto execution_manager_lock = execution_manager_.lock();
  if (execution_manager_lock == nullptr) {
    return absl::FailedPreconditionError("Execution manager is not available.");
  }
  RETURN_IF_ERROR(
      execution_manager_lock->WaitUntilAllDone(Engine::kDefaultTimeout));
  return status;
}

absl::Status SessionAdvanced::RunPrefillAsync(
    const std::vector<InputData>& contents,
    absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) {
  if (contents.empty()) {
    return absl::InvalidArgumentError("Input is empty.");
  }
  if (cancelled_->load()) {
    // Reset the cancelled flag before processing the next turn.
    cancelled_->store(false);
  }
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
    ASSIGN_OR_RETURN(
        std::vector<InputData> templated_contents,
        ApplyPromptTemplates(contents, session_info_->session_config,
                             *tokenizer_, is_first_turn_));
    ASSIGN_OR_RETURN(
        preprocessed_contents,
        PreprocessContents(templated_contents, session_info_->session_config,
                           *tokenizer_, session_info_->benchmark_info));
  }
  ASSIGN_OR_RETURN(auto task_id, execution_manager_lock->GetNewTaskId());

  RETURN_IF_ERROR(execution_manager_lock->AddPrefillTask(
      session_id_, task_id, std::move(preprocessed_contents), last_task_ids_,
      std::move(callback)));

  last_task_ids_ = {task_id};

  return absl::OkStatus();
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
  collected_responses =
      Responses(TaskState::kCreated, /*texts=*/
                std::vector<std::string>(
                    session_info_->session_config.GetNumOutputCandidates()),
                /*scores=*/
                std::vector<float>(
                    session_info_->session_config.GetNumOutputCandidates()));
  int num_decode_tokens = 0;
  auto decode_sync_callback = [&collected_responses, &num_decode_tokens](
                                  absl::StatusOr<Responses> responses) {
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
    // Normalizing the scores by the number of decode tokens if the task is
    // completed.
    if (IsTaskEndState(responses->GetTaskState())) {
      for (int i = 0; i < responses->GetScores().size(); ++i) {
        collected_responses->GetMutableScores()[i] /=
            std::max(1, num_decode_tokens);
      }
    }
  };

  RETURN_IF_ERROR(
      RunDecodeAsync(std::move(decode_sync_callback), decode_config));
  RETURN_IF_ERROR(
      execution_manager_lock->WaitUntilAllDone(Engine::kDefaultTimeout));
  return collected_responses;
}

absl::Status SessionAdvanced::RunDecodeAsync(
    absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) {
  return RunDecodeAsync(std::move(callback), DecodeConfig::CreateDefault());
}

absl::Status SessionAdvanced::RunDecodeAsync(
    absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback,
    const DecodeConfig& decode_config) {
  if (cancelled_->load()) {
    // Reset the cancelled flag before processing the next turn.
    cancelled_->store(false);
  }
  auto execution_manager_lock = execution_manager_.lock();
  if (execution_manager_lock == nullptr) {
    return absl::FailedPreconditionError("Execution manager is not available.");
  }

  ASSIGN_OR_RETURN(auto task_id, execution_manager_lock->GetNewTaskId());

  RETURN_IF_ERROR(execution_manager_lock->AddDecodeTask(
      session_id_, task_id, last_task_ids_, decode_config.GetConstraint(),
      cancelled_, std::move(callback)));

  last_task_ids_ = {task_id};

  return absl::OkStatus();
}

absl::StatusOr<Responses> SessionAdvanced::RunTextScoring(
    const std::vector<absl::string_view>& target_text,
    bool store_token_lengths) {
  return absl::UnimplementedError("RunTextScoring is not implemented.");
}

absl::StatusOr<BenchmarkInfo> SessionAdvanced::GetBenchmarkInfo() {
  if (session_info_->benchmark_info.has_value()) {
    return session_info_->benchmark_info.value();
  }
  return absl::InternalError(
      "Benchmark is not enabled. Please make sure the BenchmarkParams is set "
      "in the EngineSettings.");
}

absl::StatusOr<std::unique_ptr<Engine::Session>> SessionAdvanced::Clone(
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
      session_id_, task_id, last_task_ids_, session_id, std::move(callback)));

  last_task_ids_ = {task_id};

  ASSIGN_OR_RETURN(auto session_info,
                   execution_manager_lock->GetSessionInfo(session_id));

  return absl::WrapUnique(new SessionAdvanced(
      session_id, execution_manager_, tokenizer_, session_info, is_first_turn_,
      last_task_ids_, cancelled_));
}

}  // namespace litert::lm
