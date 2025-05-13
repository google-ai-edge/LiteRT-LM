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

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/components/sampler.h"
#include "runtime/components/tokenizer.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/llm_executor.h"
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
  // - stop_token_ids: The token ids to stop the decoding process.
  // - sampler_params: The sampler parameters used for decoding. Note that if
  //   the sampler_params.type is TYPE_UNSPECIFIED, the sampling logic will be
  //   handled by the LLM Executor.
  static absl::StatusOr<std::unique_ptr<SessionBasic>> Create(
      std::shared_ptr<LlmExecutor> executor,
      std::shared_ptr<Tokenizer> tokenizer,
      const std::vector<int>& stop_token_ids,
      const SessionConfig& session_config,
      std::optional<BenchmarkInfo> benchmark_info,
      std::shared_ptr<ThreadPool> worker_thread_pool);

  virtual ~SessionBasic() = default;

  absl::Status RunPrefill(absl::string_view input) override;
  absl::Status RunPrefillAsync(absl::string_view input,
                               InferenceObservable* observer) override;

  absl::StatusOr<Responses> RunDecode() override;

  absl::Status RunDecodeAsync(
      InferenceObservable* observer) override;

  absl::StatusOr<BenchmarkInfo> GetBenchmarkInfo() override;

 private:
  explicit SessionBasic(std::shared_ptr<LlmExecutor> executor,
                        std::shared_ptr<Tokenizer> tokenizer,
                        const std::vector<int>& stop_token_ids,
                        std::unique_ptr<Sampler> sampler,
                        const SessionConfig& session_config,
                        std::optional<BenchmarkInfo> benchmark_info,
                        std::shared_ptr<ThreadPool> worker_thread_pool)
      : executor_(executor),
        tokenizer_(tokenizer),
        stop_token_ids_(stop_token_ids),
        sampler_(std::move(sampler)),
        session_config_(session_config),
        benchmark_info_(benchmark_info),
        worker_thread_pool_(worker_thread_pool) {}

  // The internal function to prefill the input prompt. It is for convenience to
  // wrap it with lambda function for scheduling.
  absl::Status PrefillInternal(absl::string_view input,
                               bool wait_for_completion);

  // The internal functions to decode the input prompt. It is for convenience to
  // wrap it with lambda function for scheduling.
  absl::StatusOr<Responses> DecodeInternal();
  absl::Status DecodeInternalStreaming(
      InferenceObservable* observer = nullptr);

  // The executor used for run the LLM for prefill/decode.
  std::shared_ptr<LlmExecutor> executor_;

  // The tokenizer used for converting between text to token ids.
  std::shared_ptr<Tokenizer> tokenizer_;

  // The stop token ids used for decoding.
  std::vector<int> stop_token_ids_;

  // The session config used for the session.
  std::unique_ptr<Sampler> sampler_;

  // The session config used for the session.
  SessionConfig session_config_;

  // The last token id of the prefill ids. It is used for the first decode
  // process to determine the token id to start from.
  int last_prefill_token_id_;

  // The benchmark info used for the session.
  std::optional<BenchmarkInfo> benchmark_info_;

  // The thread pool used for the session.
  std::shared_ptr<ThreadPool> worker_thread_pool_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CORE_SESSION_BASIC_H_
