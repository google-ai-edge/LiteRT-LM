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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CORE_AUDIO_SESSION_ADVANCED_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CORE_AUDIO_SESSION_ADVANCED_H_

#include <atomic>
#include <memory>
#include <optional>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/base/thread_annotations.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/core/session_advanced.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/framework/resource_management/execution_manager.h"
#include "support/tokenizer/tokenizer.h"

namespace litert::lm {

// AudioSessionAdvanced extends SessionAdvanced to support specialized audio
// workflows such as synchronous audio encoding, audio encoder state reset,
// and audio frame flushing.
class AudioSessionAdvanced : public SessionAdvanced {
 public:
  // Creates an AudioSessionAdvanced object.
  static absl::StatusOr<std::unique_ptr<AudioSessionAdvanced>> Create(
      std::weak_ptr<ExecutionManager> execution_manager,
      support::Tokenizer* absl_nonnull tokenizer,
      const SessionConfig& session_config,
      std::optional<BenchmarkInfo> benchmark_info,
      std::atomic<int>* living_sessions_count = nullptr);

  // Casts a SessionInterface unique_ptr to AudioSessionAdvanced if supported.
  static absl::StatusOr<std::unique_ptr<AudioSessionAdvanced>> FromSession(
      std::unique_ptr<SessionInterface> session);

  ~AudioSessionAdvanced() override = default;

  // Synchronously encodes an audio spectrogram tensor into audio soft tokens
  // within the context of this session.
  absl::StatusOr<ExecutorAudioData> EncodeAudio(
      const TensorBuffer& spectrogram_tensor) ABSL_LOCKS_EXCLUDED(mutex_);

  // Resets the streaming audio encoder state for this session.
  absl::Status ResetAudio() ABSL_LOCKS_EXCLUDED(mutex_);

  // Flushes remaining buffered audio frames from the streaming audio encoder
  // for this session.
  absl::StatusOr<ExecutorAudioData> FlushAudio() ABSL_LOCKS_EXCLUDED(mutex_);

 protected:
  absl::StatusOr<std::unique_ptr<SessionInterface>> CloneAsyncLocked(
      absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) override
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

 private:
  AudioSessionAdvanced(SessionId session_id,
                       std::weak_ptr<ExecutionManager> execution_manager,
                       support::Tokenizer* absl_nonnull tokenizer,
                       std::shared_ptr<const SessionInfo> session_info,
                       SessionState session_state = SessionState::kFresh,
                       absl::flat_hash_set<TaskId> last_task_ids = {},
                       std::atomic<int>* living_sessions_count = nullptr)
      : SessionAdvanced(session_id, execution_manager, tokenizer, session_info,
                        session_state, last_task_ids, living_sessions_count) {}
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CORE_AUDIO_SESSION_ADVANCED_H_
