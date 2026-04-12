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

#include "runtime/core/pipeline.h"

#include <atomic>
#include <optional>
#include <utility>
#include <vector>

#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/constrained_decoding/constraint.h"
#include "runtime/components/sampler.h"
#include "runtime/components/stop_token_detector.h"
#include "runtime/components/tokenizer.h"
#include "runtime/core/tasks.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/llm_executor.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/proto/sampler_params.pb.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/status_macros.h"  //NOLINT

namespace litert::lm {

absl::StatusOr<int> Prefill(LlmExecutor& executor, ExecutorInputs& inputs,
                            bool wait_for_completion,
                            std::optional<BenchmarkInfo>& benchmark_info) {
  auto task_response =
      Tasks::Prefill(executor, inputs, wait_for_completion, benchmark_info);

  if (!task_response.ok()) {
    return task_response.status();
  }

  ASSIGN_OR_RETURN(auto text_data, inputs.GetTextDataPtr());
  LITERT_ASSIGN_OR_RETURN(
      auto ids_buffer_span,
      ReferTensorBufferAsSpan<int>(text_data->GetTokenIds()));
  return ids_buffer_span.back();
}

absl::StatusOr<Responses> Decode(LlmExecutor& executor, Tokenizer& tokenizer,
                                 const StopTokenDetector& stop_token_detector,
                                 int num_output_candidates,
                                 Constraint* constraint,
                                 std::optional<BenchmarkInfo>& benchmark_info,
                                 std::atomic<bool>* cancelled,
                                 int max_output_tokens) {
  absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback = nullptr;
  return Tasks::Decode(
      executor, tokenizer, stop_token_detector, num_output_candidates,
      benchmark_info, /*sampler=*/std::nullopt, constraint,
      /*decoded_ids=*/std::nullopt, /*callback=*/callback, cancelled,
      max_output_tokens);
}

absl::Status DecodeStreaming(
    LlmExecutor& executor, Tokenizer& tokenizer,
    const StopTokenDetector& stop_token_detector, int num_output_candidates,
    Constraint* constraint, std::optional<BenchmarkInfo>& benchmark_info,
    absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback,
    std::atomic<bool>* cancelled, int max_output_tokens) {
  if (callback == nullptr) {
    return absl::InvalidArgumentError(
        "Callback must not be null for streaming.");
  }
  absl::StatusOr<Responses> task_respones =
      Tasks::Decode(executor, tokenizer, stop_token_detector,
                    num_output_candidates, benchmark_info,
                    /*sampler=*/std::nullopt, constraint,
                    /*decoded_ids=*/std::nullopt, callback, cancelled,
                    max_output_tokens);

  // Trigger the callback with the final result.
  // This can be either a error message, or a task state (e.g. kDone or
  // kMaxNumTokensReached).
  callback(task_respones);
  return task_respones.status();
}

absl::StatusOr<Responses> DecodeCustomSampling(
    LlmExecutor& executor, Tokenizer& tokenizer,
    const StopTokenDetector& stop_token_detector, int num_output_candidates,
    Sampler& sampler, litert::TensorBuffer decoded_ids, Constraint* constraint,
    std::optional<BenchmarkInfo>& benchmark_info,
    std::atomic<bool>* cancelled, int max_output_tokens) {
  absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback = nullptr;
  return Tasks::Decode(executor, tokenizer, stop_token_detector,
                       num_output_candidates, benchmark_info, &sampler,
                       constraint, std::move(decoded_ids),
                       /*callback=*/callback, cancelled, max_output_tokens);
}

absl::Status DecodeCustomSamplingStreaming(
    LlmExecutor& executor, Tokenizer& tokenizer,
    const StopTokenDetector& stop_token_detector, int num_output_candidates,
    Sampler& sampler, litert::TensorBuffer decoded_ids, Constraint* constraint,
    std::optional<BenchmarkInfo>& benchmark_info,
    absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback,
    std::atomic<bool>* cancelled, int max_output_tokens) {
  if (callback == nullptr) {
    return absl::InvalidArgumentError(
        "Callback must not be null for streaming.");
  }
  absl::StatusOr<Responses> task_respones = Tasks::Decode(
      executor, tokenizer, stop_token_detector, num_output_candidates,
      benchmark_info, &sampler, constraint, std::move(decoded_ids), callback,
      cancelled, max_output_tokens);

  // Trigger the callback with the final result.
  // This can be either a error message, or a task state (e.g. kDone or
  // kMaxNumTokensReached).
  callback(task_respones);
  return task_respones.status();
}

absl::StatusOr<ScoringResponses> ScoreCustomSampling(
    LlmExecutor& executor, Tokenizer& tokenizer,
    const std::vector<absl::string_view>& target_texts, const float temperature,
    litert::TensorBuffer decoded_ids, bool store_token_lengths) {
  return Tasks::Score(executor, tokenizer, target_texts, temperature,
                      std::move(decoded_ids), store_token_lengths);
}

}  // namespace litert::lm
