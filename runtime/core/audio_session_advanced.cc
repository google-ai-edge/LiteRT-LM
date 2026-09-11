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
#include <cstddef>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/cc/litert_tensor_buffer_types.h"  // from @litert
#include "runtime/core/session_advanced.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/framework/resource_management/execution_manager.h"
#include "runtime/util/status_macros.h"
#include "support/tokenizer/tokenizer.h"

namespace litert::lm {
namespace {

// Returns the index of the sequence length dimension for audio embeddings
// tensors:
// - In 5D spatial tensors with BHWDC layout ([batch, height, width, depth,
//   channels]), width (index 2) corresponds to the sequence length (time
//   steps).
// - For 2D to 4D tensors (e.g. [batch, sequence_length, embedding_dim]), the
//   penultimate dimension (rank - 2) corresponds to the sequence length.
// Returns an error if the tensor dimensions are not understood (e.g. rank < 2).
absl::StatusOr<int> GetAudioSequenceDimensionIndex(absl::Span<const int> dims) {
  constexpr size_t kBhwdcTensorRank = 5;
  constexpr int kBhwdcSequenceDimIndex = 2;
  if (dims.size() == kBhwdcTensorRank) {
    return kBhwdcSequenceDimIndex;
  }

  constexpr size_t kMinStandardTensorRank = 2;
  constexpr int kPenultimateDimOffset = 2;
  if (dims.size() >= kMinStandardTensorRank) {
    return static_cast<int>(dims.size()) - kPenultimateDimOffset;
  }

  return absl::InvalidArgumentError(absl::StrCat(
      "Unsupported audio embeddings tensor dimensions rank: ", dims.size(),
      ". Expected rank >= 2."));
}

absl::StatusOr<::litert::TensorBuffer> SliceAudioEmbeddingTensor(
    const ::litert::TensorBuffer& source_tensor, int valid_tokens,
    const ::litert::Environment& env) {
  LITERT_ASSIGN_OR_RETURN(auto tensor_type, source_tensor.TensorType());
  const auto& dims = tensor_type.Layout().Dimensions();
  ABSL_ASSIGN_OR_RETURN(const int seq_dim_idx,
                        GetAudioSequenceDimensionIndex(dims));

  for (int i = 0; i < seq_dim_idx; ++i) {
    if (dims[i] != 1) {
      return absl::InvalidArgumentError(
          absl::StrCat("Audio embedding slicing requires unit batch/leading "
                       "dimensions, got dim ",
                       i, " = ", dims[i]));
    }
  }

  if (valid_tokens == -1 || dims[seq_dim_idx] == valid_tokens) {
    LITERT_ASSIGN_OR_RETURN(auto dup, source_tensor.Duplicate());
    return dup;
  }
  if (valid_tokens <= 0 || valid_tokens > dims[seq_dim_idx]) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Invalid valid_tokens (", valid_tokens,
        ") for sequence dimension length (", dims[seq_dim_idx], ")."));
  }

  ::litert::Dimensions sliced_dims(dims.begin(), dims.end());
  sliced_dims[seq_dim_idx] = valid_tokens;
  auto sliced_tensor_type = ::litert::RankedTensorType(
      tensor_type.ElementType(), ::litert::Layout(std::move(sliced_dims)));
  LITERT_ASSIGN_OR_RETURN(auto bytes, sliced_tensor_type.Bytes());
  LITERT_ASSIGN_OR_RETURN(auto sliced_tensor,
                          ::litert::TensorBuffer::CreateManaged(
                              env, ::litert::TensorBufferType::kHostMemory,
                              sliced_tensor_type, bytes));
  LITERT_ASSIGN_OR_RETURN(
      auto lock_and_addr,
      ::litert::TensorBufferScopedLock::Create(
          source_tensor, ::litert::TensorBuffer::LockMode::kRead));
  LITERT_RETURN_IF_ERROR(sliced_tensor.Write(absl::MakeConstSpan(
      static_cast<const char*>(lock_and_addr.second), bytes)));
  return sliced_tensor;
}

absl::StatusOr<InputAudio> ConvertExecutorAudioDataToInputAudio(
    const ExecutorAudioData& audio_data, const ::litert::Environment& env) {
  const ::litert::TensorBuffer* tensor = nullptr;
  auto proj_status = audio_data.GetProjectedAudioEmbeddingsPtr();
  if (proj_status.ok() && *proj_status != nullptr) {
    tensor = *proj_status;
  } else {
    auto audio_emb_status = audio_data.GetAudioEmbeddingsPtr();
    if (audio_emb_status.ok() && *audio_emb_status != nullptr) {
      tensor = *audio_emb_status;
    } else {
      return absl::InvalidArgumentError(
          "ExecutorAudioData does not contain audio embeddings.");
    }
  }

  ABSL_ASSIGN_OR_RETURN(
      auto sliced_tensor,
      SliceAudioEmbeddingTensor(*tensor, audio_data.GetValidTokens(), env));
  return InputAudio(std::move(sliced_tensor), /*is_embeddings=*/true);
}

}  // namespace

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

absl::StatusOr<std::vector<ExecutorAudioData>>
AudioSessionAdvanced::EncodeAudio(
    absl::Span<const TensorBuffer> spectrogram_tensors) {
  absl::MutexLock lock(mutex_);
  auto execution_manager_lock = execution_manager_.lock();
  if (execution_manager_lock == nullptr) {
    return absl::FailedPreconditionError("Execution manager is not available.");
  }
  std::vector<ExecutorAudioData> results;
  results.reserve(spectrogram_tensors.size());
  for (const auto& tensor : spectrogram_tensors) {
    ABSL_ASSIGN_OR_RETURN(
        auto data, execution_manager_lock->EncodeAudio(*session_info_, tensor));
    results.push_back(std::move(data));
  }
  return results;
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

absl::Status AudioSessionAdvanced::RunPrefill(
    const ExecutorAudioData& audio_data) {
  auto execution_manager_lock = execution_manager_.lock();
  if (execution_manager_lock == nullptr) {
    return absl::FailedPreconditionError("Execution manager is not available.");
  }
  ABSL_ASSIGN_OR_RETURN(const auto* env,
                        execution_manager_lock->GetEnvironment());
  if (env == nullptr) {
    return absl::InternalError("LiteRT environment is not available.");
  }
  ABSL_ASSIGN_OR_RETURN(auto input_audio,
                        ConvertExecutorAudioDataToInputAudio(audio_data, *env));
  std::vector<InputData> inputs;
  inputs.push_back(std::move(input_audio));
  return SessionAdvanced::RunPrefill(inputs);
}

}  // namespace litert::lm
