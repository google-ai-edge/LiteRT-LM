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

#include "runtime/core/audio_session_advanced.h"

#include <atomic>
#include <memory>
#include <optional>
#include <utility>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/core/session_advanced.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/framework/resource_management/execution_manager.h"
#include "runtime/util/status_macros.h"
#include "support/tokenizer/tokenizer.h"

namespace litert::lm {

// static
absl::StatusOr<std::unique_ptr<AudioSessionAdvanced>>
AudioSessionAdvanced::Create(std::weak_ptr<ExecutionManager> execution_manager,
                             support::Tokenizer* absl_nonnull tokenizer,
                             const SessionConfig& session_config,
                             std::optional<BenchmarkInfo> benchmark_info,
                             std::atomic<int>* living_sessions_count) {
  auto execution_manager_lock = execution_manager.lock();
  if (execution_manager_lock == nullptr) {
    return absl::FailedPreconditionError("Execution manager is not available.");
  }
  ABSL_ASSIGN_OR_RETURN(auto session_id,
                        execution_manager_lock->RegisterNewSession(
                            session_config, benchmark_info));
  ABSL_ASSIGN_OR_RETURN(auto session_info,
                        execution_manager_lock->GetSessionInfo(session_id));
  return absl::WrapUnique(new AudioSessionAdvanced(
      session_id, execution_manager, tokenizer, session_info,
      /*session_state=*/SessionState::kFresh,
      /*last_task_ids=*/{}, living_sessions_count));
}

// static
absl::StatusOr<std::unique_ptr<AudioSessionAdvanced>>
AudioSessionAdvanced::FromSession(std::unique_ptr<SessionInterface> session) {
  if (session == nullptr) {
    return absl::InvalidArgumentError("Session cannot be null.");
  }
  if (!session->GetSessionConfig().EnableAudioSessionAdvanced()) {
    return absl::InvalidArgumentError(
        "Session is not an AudioSessionAdvanced instance. Make sure "
        "EnableAudioSessionAdvanced is set in SessionConfig.");
  }
  return absl::WrapUnique(
      static_cast<AudioSessionAdvanced*>(session.release()));
}

absl::StatusOr<ExecutorAudioData> AudioSessionAdvanced::EncodeAudio(
    const TensorBuffer& spectrogram_tensor) {
  absl::MutexLock lock(mutex_);
  auto execution_manager_lock = execution_manager_.lock();
  if (execution_manager_lock == nullptr) {
    return absl::FailedPreconditionError("Execution manager is not available.");
  }
  return execution_manager_lock->EncodeAudio(*session_info_,
                                             spectrogram_tensor);
}

absl::Status AudioSessionAdvanced::ResetAudio() {
  absl::MutexLock lock(mutex_);
  auto execution_manager_lock = execution_manager_.lock();
  if (execution_manager_lock == nullptr) {
    return absl::FailedPreconditionError("Execution manager is not available.");
  }
  return execution_manager_lock->ResetAudio(*session_info_);
}

absl::StatusOr<ExecutorAudioData> AudioSessionAdvanced::FlushAudio() {
  absl::MutexLock lock(mutex_);
  auto execution_manager_lock = execution_manager_.lock();
  if (execution_manager_lock == nullptr) {
    return absl::FailedPreconditionError("Execution manager is not available.");
  }
  return execution_manager_lock->FlushAudio(*session_info_);
}

absl::StatusOr<std::unique_ptr<SessionInterface>>
AudioSessionAdvanced::CloneAsyncLocked(
    absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) {
  auto execution_manager_lock = execution_manager_.lock();
  if (execution_manager_lock == nullptr) {
    return absl::FailedPreconditionError("Execution manager is not available.");
  }

  ABSL_ASSIGN_OR_RETURN(auto task_id, execution_manager_lock->GetNewTaskId());

  ABSL_ASSIGN_OR_RETURN(
      auto session_id,
      execution_manager_lock->RegisterNewSession(
          session_info_->session_config, session_info_->benchmark_info));

  ABSL_RETURN_IF_ERROR(execution_manager_lock->AddCloneSessionTask(
      session_id_, task_id, last_task_ids_, session_id,
      std::make_shared<std::atomic<bool>>(false), std::move(callback)));

  last_task_ids_ = {task_id};

  ABSL_ASSIGN_OR_RETURN(auto session_info,
                        execution_manager_lock->GetSessionInfo(session_id));

  return absl::WrapUnique(new AudioSessionAdvanced(
      session_id, execution_manager_, tokenizer_, session_info, session_state_,
      last_task_ids_, living_sessions_count_));
}

}  // namespace litert::lm
