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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CORE_TASKS_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CORE_TASKS_H_

#include <atomic>
#include <limits>
#include <optional>
#include <vector>

#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/constrained_decoding/constraint.h"
#include "runtime/components/sampler.h"
#include "runtime/components/stop_token_detector.h"
#include "runtime/components/tokenizer.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/llm_executor.h"
#include "runtime/executor/llm_executor_io_types.h"

namespace litert::lm::Tasks {

absl::StatusOr<Responses> Prefill(LlmExecutor& executor, ExecutorInputs& inputs,
                                  bool wait_for_completion,
                                  std::optional<BenchmarkInfo>& benchmark_info);

absl::StatusOr<Responses> Decode(
    LlmExecutor& executor, Tokenizer& tokenizer,
    const StopTokenDetector& stop_token_detector, int num_output_candidates,
    std::optional<BenchmarkInfo>& benchmark_info,
    std::optional<Sampler*> sampler, Constraint* constraint,
    std::optional<litert::TensorBuffer> decoded_ids,
    absl::AnyInvocable<void(absl::StatusOr<Responses>)>& callback,
    std::atomic<bool>* cancelled,
    int max_output_tokens = std::numeric_limits<int>::max());

absl::StatusOr<ScoringResponses> Score(
    LlmExecutor& executor, Tokenizer& tokenizer,
    const std::vector<absl::string_view>& target_texts, float temperature,
    litert::TensorBuffer decoded_ids, bool store_token_lengths = false);

}  // namespace litert::lm::Tasks

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CORE_TASKS_H_
