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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CORE_SESSION_BASIC_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CORE_SESSION_BASIC_H_

#include <atomic>
#include <memory>
#include <optional>
#include <string>
#include <utility>
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
#include "runtime/components/sampler.h"
#include "runtime/components/stop_token_detector.h"
#include "runtime/components/tokenizer.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/audio_executor.h"
#include "runtime/executor/llm_executor.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/executor/vision_executor.h"
#include "runtime/framework/threadpool.h"
#include "runtime/proto/sampler_params.pb.h"

namespace litert::lm {

// SessionBasic is a basic implementation of Engine::Session. The underlying
// prefill/decode pipelines use the LLM Executor's basic Decode function which
// does the sampling logics inside.
class SessionBasic : public Engine::Session {
 public:
  // Creates a SessionBasic object.
  // - executor: The initialized LLM Executor to call.
  // - tokenizer: The tokenizer to encode/decode the text into token ids.
  // - vision_executor: The vision executor to encode the image input.
  // - audio_executor: The audio executor to encode the audio input.
  // - stop_token_ids: The token ids to stop the decoding process.
  // - sampler_params: The sampler parameters used for decoding. Note that if
  //   the sampler_params.type is TYPE_UNSPECIFIED, the sampling logic will be
  //   handled by the LLM Executor.
  static absl::StatusOr<std::unique_ptr<SessionBasic>> Create(
      LlmExecutor* absl_nonnull executor, Tokenizer* absl_nonnull tokenizer,
      VisionExecutor* vision_executor, AudioExecutor* audio_executor,
      const SessionConfig& session_config,
      std::optional<BenchmarkInfo> benchmark_info,
      ThreadPool* absl_nonnull inference_thread_pool);

  virtual ~SessionBasic();

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

  absl::StatusOr<std::unique_ptr<Engine::Session::TaskController>>
  RunTextScoringAsync(
      const std::vector<absl::string_view>& target_text,
      absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback,
      bool store_token_lengths) override;

  absl::Status RunPrefill(const std::vector<InputData>& contents) override;

  absl::StatusOr<std::unique_ptr<Engine::Session::TaskController>>
  RunPrefillAsync(
      const std::vector<InputData>& contents,
      absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) override;

  absl::StatusOr<Responses> RunDecode() override;

  absl::StatusOr<Responses> RunDecode(
      const DecodeConfig& decode_config) override;

  absl::StatusOr<std::unique_ptr<Engine::Session::TaskController>>
  RunDecodeAsync(
      absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) override;

  absl::StatusOr<std::unique_ptr<Engine::Session::TaskController>>
  RunDecodeAsync(absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback,
                 const DecodeConfig& decode_config) override;

  absl::StatusOr<BenchmarkInfo> GetBenchmarkInfo() override;

  absl::StatusOr<BenchmarkInfo*> GetMutableBenchmarkInfo() override;

  // TODO(b/450903294): Add rollback history support for Session and
  // Conversation.
  void CancelProcess() override {
    ABSL_LOG(INFO) << "SessionBasic::CancelProcess";
    cancelled_.store(true);
  }

  absl::Status WaitUntilDone() override {
    return inference_thread_pool_.WaitUntilDone(Engine::kDefaultTimeout);
  }

  const SessionConfig& GetSessionConfig() const override {
    return session_config_;
  }

  // Util function for creating the combined ExecutorInputs from the
  // preprocessed contents.
  // TODO - b/436674053: Modularize the preprocessing logic into a separate
  // preprocessor class.
  absl::StatusOr<ExecutorInputs> ProcessAndCombineContents(
      const std::vector<InputData>& preprocessed_contents);

  // Save the current step with the name `label`. You can later rewind to this
  // checkpoint using `RewindToCheckpoint(label)`. If the checkpoint name
  // already exists, the step number will be overwritten.
  absl::Status SaveCheckpoint(absl::string_view label) override;

  // Rewinds the session to the given checkpoint. Checkpoints after the
  // restored step will be removed. Returns an error if the checkpoint name
  // does not exist.
  absl::Status RewindToCheckpoint(absl::string_view label) override;

  // Get the current step of the session.
  absl::StatusOr<int> GetCurrentStep() const override;

 private:
  explicit SessionBasic(LlmExecutor* absl_nonnull executor,
                        Tokenizer* absl_nonnull tokenizer,
                        VisionExecutor* vision_executor,
                        AudioExecutor* audio_executor,
                        std::unique_ptr<Sampler> sampler,
                        const SessionConfig& session_config,
                        std::optional<BenchmarkInfo> benchmark_info,
                        ThreadPool* absl_nonnull inference_thread_pool,
                        const StopTokenDetector& stop_token_detector)
      : executor_(*executor),
        tokenizer_(*tokenizer),
        vision_executor_(vision_executor),
        audio_executor_(audio_executor),
        sampler_(std::move(sampler)),
        session_config_(session_config),
        benchmark_info_(benchmark_info),
        inference_thread_pool_(*inference_thread_pool),
        stop_token_detector_(stop_token_detector) {}

  // The internal function to prefill the input prompt. It is for convenience to
  // wrap it with lambda function for scheduling.
  absl::Status PrefillInternal(
      const std::vector<InputData>& preprocessed_contents,
      bool wait_for_completion);

  // The internal functions to decode the input prompt. It is for convenience to
  // wrap it with lambda function for scheduling.
  absl::StatusOr<Responses> DecodeInternal(const DecodeConfig& decode_config);
  absl::Status DecodeInternalStreaming(
      absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback,
      const DecodeConfig& decode_config);

  // The executor used for run the LLM for prefill/decode.
  LlmExecutor& executor_;

  // The tokenizer used for converting between text to token ids.
  Tokenizer& tokenizer_;

  // The vision executor used for run the LLM for prefill/decode.
  VisionExecutor* vision_executor_;

  // The audio executor used for run the LLM for prefill/decode.
  AudioExecutor* audio_executor_;

  // The session config used for the session.
  std::unique_ptr<Sampler> sampler_;

  // The session config used for the session.
  SessionConfig session_config_;

  // The last token id of the prefill ids. It is used for the first decode
  // process to determine the token id to start from.
  int last_prefill_token_id_;

  // The benchmark info used for the session.
  std::optional<BenchmarkInfo> benchmark_info_;

  // The dedicated thread pool used to orchestrate the serial token prefill and
  // autoregressive inference decoding loops (effectively the inference
  // thread pool).
  ThreadPool& inference_thread_pool_;

  // The stop token detector used for the session.
  StopTokenDetector stop_token_detector_;

  // An atomic boolean to indicate whether the session is cancelled.
  std::atomic<bool> cancelled_{false};

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
  SessionState session_state_ = SessionState::kFresh;

  // The set of executors that are already existed in the system. This is used
  // to avoid creating multiple sessions for the same executor.
  static absl::flat_hash_set<LlmExecutor*>* occupied_executors_
      ABSL_GUARDED_BY(occupied_executors_mu_);
  static absl::Mutex occupied_executors_mu_;

  // The map of checkpoint name to step.
  absl::flat_hash_map<std::string, int> checkpoint_map_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CORE_SESSION_BASIC_H_
