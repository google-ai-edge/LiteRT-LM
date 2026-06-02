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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CORE_SESSION_ADVANCED_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CORE_SESSION_ADVANCED_H_

#include <atomic>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/base/thread_annotations.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "runtime/components/tokenizer.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/framework/resource_management/execution_manager.h"
#include "runtime/proto/sampler_params.pb.h"

namespace litert::lm {

// SessionAdvanced is an implementation of SessionInterface. The
// underlying prefill/decode use the LLM Execution Manager's advanced resource
// management to support efficient multi-sessions and session cloning features.
class SessionAdvanced : public SessionInterface {
 public:
  class AdvancedTaskController : public SessionInterface::TaskController {
   public:
    AdvancedTaskController(TaskId task_id,
                           std::shared_ptr<std::atomic<bool>> cancelled,
                           std::weak_ptr<ExecutionManager> execution_manager)
        : task_id_(task_id),
          cancelled_(cancelled),
          execution_manager_(execution_manager) {}

    absl::Status WaitUntilDone(absl::Duration timeout) override {
      auto execution_manager_lock = execution_manager_.lock();
      if (execution_manager_lock == nullptr) {
        return absl::FailedPreconditionError(
            "Execution manager is not available.");
      }
      return execution_manager_lock->WaitUntilDone(task_id_, timeout);
    }

    absl::Status Cancel() override {
      cancelled_->store(true);
      return absl::OkStatus();
    }

   private:
    // The task ID of the async task.
    TaskId task_id_;

    // An atomic boolean to indicate whether the session is cancelled.
    std::shared_ptr<std::atomic<bool>> cancelled_;

    // The execution manager used for the session.
    std::weak_ptr<ExecutionManager> execution_manager_;
  };

  // Creates a SessionAdvanced object.
  static absl::StatusOr<std::unique_ptr<SessionAdvanced>> Create(
      std::weak_ptr<ExecutionManager> execution_manager,
      Tokenizer* absl_nonnull tokenizer, const SessionConfig& session_config,
      std::optional<BenchmarkInfo> benchmark_info,
      std::atomic<int>* living_sessions_count = nullptr);

  // Destroys the SessionAdvanced object. It will wait for all tasks to be
  // done and release the session from the execution manager.
  ~SessionAdvanced() override;

  absl::StatusOr<Responses> GenerateContent(
      const std::vector<InputData>& contents) override;
  absl::Status GenerateContentStream(
      const std::vector<InputData>& contents,
      absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) override;
  absl::Status GenerateContentStream(
      const std::vector<InputData>& contents,
      absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback,
      const DecodeConfig& decode_config) override;

  // Scores the target text after the prefill process is done. This function
  // will only run the decode process to fetch the decode output logits, which
  // is used to calculate the target text's score and update the model memory
  // using the target_text tokens.
  // This function should be called after the prefill process is done.
  // - target_text: The target text to score.
  // - store_token_lengths: Whether to store the token lengths of the target
  //   texts in `Responses`.
  // - return: This function returns the score associated with the target
  // text after the model has been prefilled. The returned score is the sum of
  // the negative log probability of seeing the target text during decode.
  absl::StatusOr<Responses> RunTextScoring(
      const std::vector<absl::string_view>& target_text,
      bool store_token_lengths) override;

  absl::StatusOr<std::unique_ptr<SessionInterface::TaskController>>
  RunTextScoringAsync(
      const std::vector<absl::string_view>& target_text,
      absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback,
      bool store_token_lengths) override ABSL_LOCKS_EXCLUDED(mutex_);

  absl::Status RunPrefill(const std::vector<InputData>& contents) override;

  absl::StatusOr<std::unique_ptr<TaskController>> RunPrefillAsync(
      const std::vector<InputData>& contents,
      absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) override
      ABSL_LOCKS_EXCLUDED(mutex_);

  absl::StatusOr<Responses> RunDecode() override;

  absl::StatusOr<Responses> RunDecode(
      const DecodeConfig& decode_config) override;

  absl::StatusOr<std::unique_ptr<TaskController>> RunDecodeAsync(
      absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) override
      ABSL_LOCKS_EXCLUDED(mutex_);

  absl::StatusOr<std::unique_ptr<TaskController>> RunDecodeAsync(
      absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback,
      const DecodeConfig& decode_config) override ABSL_LOCKS_EXCLUDED(mutex_);

  absl::StatusOr<BenchmarkInfo> GetBenchmarkInfo() override
      ABSL_LOCKS_EXCLUDED(mutex_);

  absl::StatusOr<BenchmarkInfo*> GetMutableBenchmarkInfo() override;

  // Save the current step with the name `label`. You can later rewind to this
  // checkpoint using `RewindToCheckpoint(label)`. If the checkpoint name
  // already exists, the step number will be overwritten. Returns the saved
  // step number.
  absl::Status SaveCheckpoint(absl::string_view label) override;

  // Rewinds the session to the given checkpoint and then returns the current
  // step.
  absl::Status RewindToCheckpoint(absl::string_view label) override;

  // Get the current step of the session.
  absl::StatusOr<int> GetCurrentStep() const override;

  // TODO(b/450903294): Add rollback history support for Session and
  // Conversation.
  void CancelProcess() override {
    ABSL_LOG(INFO) << "SessionAdvanced::CancelProcess";
    auto execution_manager_lock = execution_manager_.lock();
    if (execution_manager_lock == nullptr) {
      ABSL_LOG(ERROR) << "Execution manager is not available.";
      return;
    }
    auto status = execution_manager_lock->CancelAllTasksInSession(session_id_);
    if (!status.ok()) {
      ABSL_LOG(ERROR) << "Failed to cancel all tasks in session: " << status;
    }
  }

  const SessionConfig& GetSessionConfig() const override {
    return session_info_->session_config;
  }

  absl::Status WaitUntilDone() override {
    auto execution_manager_lock = execution_manager_.lock();
    if (execution_manager_lock == nullptr) {
      return absl::FailedPreconditionError(
          "Execution manager is not available.");
    }
    return execution_manager_lock->WaitUntilSessionDone(
        session_id_, Engine::kDefaultTimeout);
  }

  // TODO b/409401231 - Add unit tests for this function.
  absl::StatusOr<std::unique_ptr<SessionInterface>> Clone() override
      ABSL_LOCKS_EXCLUDED(mutex_);

  // TODO b/409401231 - Add unit tests for this function.
  absl::StatusOr<std::unique_ptr<SessionInterface>> CloneAsync(
      absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) override
      ABSL_LOCKS_EXCLUDED(mutex_);

 private:
  // The state of the session.
  // * `kFresh` means the session is just created and
  //   hasn't been prefilled yet.
  // * `kPrefilled` means the session has been prefilled
  //   but not decoded yet.
  // * `kDecoded` means the session has been decoded.
  //
  // A session is considered fresh only if it has not been prefilled or decoded
  // yet.
  // A session could transition between kPrefilled and kDecoded if
  // `RunPrefill` or `RunDecode` is called multiple times.
  enum class SessionState : int { kFresh, kPrefilled, kDecoded };

  explicit SessionAdvanced(SessionId session_id,
                           std::weak_ptr<ExecutionManager> execution_manager,
                           Tokenizer* absl_nonnull tokenizer,
                           std::shared_ptr<const SessionInfo> session_info,
                           SessionState session_state = SessionState::kFresh,
                           absl::flat_hash_set<TaskId> last_task_ids = {},
                           std::atomic<int>* living_sessions_count = nullptr)
      : session_id_(session_id),
        execution_manager_(execution_manager),
        tokenizer_(tokenizer),
        session_info_(session_info),
        session_state_(session_state),
        last_task_ids_(last_task_ids),
        living_sessions_count_(living_sessions_count) {
    if (living_sessions_count_) {
      (*living_sessions_count_)++;
    }
  }

  // The implementation of CloneAsync which assumes mutex_ is locked.
  absl::StatusOr<std::unique_ptr<SessionInterface>> CloneAsyncLocked(
      absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  // The session ID used for the session.
  SessionId session_id_;

  // The execution manager used for the session.
  std::weak_ptr<ExecutionManager> execution_manager_;

  // The tokenizer used for the session.
  Tokenizer* absl_nonnull tokenizer_;

  // The session info used for the session.
  std::shared_ptr<const SessionInfo> session_info_;

  // The state of the session.
  SessionState session_state_ ABSL_GUARDED_BY(mutex_);

  // The last task IDs that might be executing in the session.
  absl::flat_hash_set<TaskId> last_task_ids_ ABSL_GUARDED_BY(mutex_) = {};

  struct CheckpointInfo {
    int step;
    SessionState state;
  };

  // The checkpoint map for the session.
  absl::flat_hash_map<std::string, CheckpointInfo> checkpoint_map_
      ABSL_GUARDED_BY(mutex_) = {};

  // Mutex for protecting the session state and last task IDs.
  absl::Mutex mutex_;

  // Pointer to the counter of living sessions in Engine.
  std::atomic<int>* living_sessions_count_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CORE_SESSION_ADVANCED_H_
