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
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/components/tokenizer.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/framework/resource_management/execution_manager.h"
#include "runtime/proto/sampler_params.pb.h"

namespace litert::lm {

// SessionAdvanced is an advanced implementation of Engine::Session. The
// underlying prefill/decode use the LLM Execution Manager's advanced resource
// management to support efficient multi-sessions and session cloning features.
class SessionAdvanced : public Engine::Session {
 public:
  // Creates a SessionAdvanced object.
  // - executor: The initialized LLM Executor to call.
  // - tokenizer: The tokenizer to encode/decode the text into token ids.
  // - vision_executor: The vision executor to encode the image input.
  // - audio_executor: The audio executor to encode the audio input.
  // - stop_token_ids: The token ids to stop the decoding process.
  // - sampler_params: The sampler parameters used for decoding. Note that if
  //   the sampler_params.type is TYPE_UNSPECIFIED, the sampling logic will be
  //   handled by the LLM Executor.
  static absl::StatusOr<std::unique_ptr<SessionAdvanced>> Create(
      std::weak_ptr<ExecutionManager> execution_manager,
      Tokenizer* absl_nonnull tokenizer, const SessionConfig& session_config,
      std::optional<BenchmarkInfo> benchmark_info);

  // TODO b/409401231 - Call execution manager's release session instead.
  ~SessionAdvanced() override {
    CancelProcess();
    auto execution_manager_lock = execution_manager_.lock();
    if (execution_manager_lock != nullptr) {
      auto status = execution_manager_lock->ReleaseSession(session_id_);
      if (!status.ok()) {
        ABSL_LOG(ERROR) << "Failed to release session: " << status;
      }
    } else {
      ABSL_LOG(ERROR) << "Execution manager should not be deleted before "
                         "Session is deleted.";
    }
  };

  absl::StatusOr<Responses> GenerateContent(
      const std::vector<InputData>& contents) override {
    return absl::UnimplementedError("GenerateContent is not implemented.");
  };
  absl::Status GenerateContentStream(
      const std::vector<InputData>& contents,
      absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) override {
    return absl::UnimplementedError(
        "GenerateContentStream is not implemented.");
  };
  absl::Status GenerateContentStream(
      const std::vector<InputData>& contents,
      absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback,
      const DecodeConfig& decode_config) override {
    return absl::UnimplementedError(
        "GenerateContentStream is not implemented.");
  };

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

  absl::Status RunPrefill(const std::vector<InputData>& contents) override;

  absl::Status RunPrefillAsync(
      const std::vector<InputData>& contents,
      absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) override;

  absl::StatusOr<Responses> RunDecode() override;

  absl::StatusOr<Responses> RunDecode(
      const DecodeConfig& decode_config) override;

  absl::Status RunDecodeAsync(
      absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) override;

  absl::Status RunDecodeAsync(
      absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback,
      const DecodeConfig& decode_config) override;

  absl::StatusOr<BenchmarkInfo> GetBenchmarkInfo() override;

  // TODO(b/450903294): Add rollback history support for Session and
  // Conversation.
  void CancelProcess() override {
    ABSL_LOG(INFO) << "SessionAdvanced::CancelProcess";
    cancelled_->store(true);
  }

  const SessionConfig& GetSessionConfig() const override {
    return session_info_->session_config;
  }

  const Tokenizer& GetTokenizer() const override { return *tokenizer_; }

  absl::Status WaitUntilDone() override {
    auto execution_manager_lock = execution_manager_.lock();
    if (execution_manager_lock == nullptr) {
      return absl::FailedPreconditionError(
          "Execution manager is not available.");
    }
    return execution_manager_lock->WaitUntilAllDone(Engine::kDefaultTimeout);
  }

  // TODO b/409401231 - Add unit tests for this function.
  absl::StatusOr<std::unique_ptr<Session>> Clone(
      absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) override;

 private:
  explicit SessionAdvanced(SessionId session_id,
                           std::weak_ptr<ExecutionManager> execution_manager,
                           Tokenizer* absl_nonnull tokenizer,
                           std::shared_ptr<const SessionInfo> session_info,
                           bool is_first_turn = true,
                           absl::flat_hash_set<TaskId> last_task_ids = {},
                           std::shared_ptr<std::atomic<bool>> cancelled =
                               std::make_shared<std::atomic<bool>>(false))
      : session_id_(session_id),
        execution_manager_(execution_manager),
        tokenizer_(tokenizer),
        session_info_(session_info),
        is_first_turn_(is_first_turn),
        last_task_ids_(last_task_ids),
        cancelled_(cancelled) {}

  // The session ID used for the session.
  SessionId session_id_;

  // The execution manager used for the session.
  std::weak_ptr<ExecutionManager> execution_manager_;

  // The tokenizer used for the session.
  Tokenizer* absl_nonnull tokenizer_;

  // The session info used for the session.
  std::shared_ptr<const SessionInfo> session_info_;

  // Whether the current turn is the first turn.
  // TODO - b/436674053: This is a temporary solution to determine whether the
  // current turn is the first turn. Should be removed once prompt templates
  // is no longer used.
  bool is_first_turn_ = true;

  // The last task IDs that might be executing in the session.
  absl::flat_hash_set<TaskId> last_task_ids_ = {};

  // An atomic boolean to indicate whether the session is cancelled.
  std::shared_ptr<std::atomic<bool>> cancelled_ =
      std::make_shared<std::atomic<bool>>(false);
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CORE_SESSION_ADVANCED_H_
