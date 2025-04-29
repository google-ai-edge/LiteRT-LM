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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_PIPELINE_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_PIPELINE_H_

#include <stdbool.h>
#include <memory>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/sampler.h"
#include "runtime/components/tokenizer.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/llm_executor.h"

namespace litert::lm {

// Runs the pipeline to prefill the input prompt.
// - executor: The initialized LLM Executor to call.
// - tokenizer: The tokenizer to encode the text into token ids.
// - prompt: The input prompt to prefill.
// - bos_token_id: The token id of the start token.
// Returns the last token id of the prefill ids. It is used for
//   the next decode process to determine the token id to start from.
// - wait_for_completion: If true, wait for the prefill to complete before
//   returning.
absl::StatusOr<int> Prefill(std::shared_ptr<LlmExecutor> executor,
                            std::shared_ptr<Tokenizer> tokenizer,
                            absl::string_view prompt, int bos_token_id,
                            bool wait_for_completion = true);

// Runs the pipeline to decode the input prompt.
// - executor: The initialized LLM Executor to call.
// - tokenizer: The tokenizer to decode the token ids into text.
// - stop_token_ids: The token ids to stop the decoding process.
// TODO(b/397975034): support batched output and update the logic to avoid
// detokenizing the stop tokens.
absl::StatusOr<Responses> Decode(std::shared_ptr<LlmExecutor> executor,
                                 std::shared_ptr<Tokenizer> tokenizer,
                                 const std::vector<int>& stop_token_ids);

// Runs the pipeline to decode the input prompt.
// - executor: The initialized LLM Executor to call.
// - tokenizer: The tokenizer to decode the token ids into text.
// - stop_token_ids: The token ids to stop the decoding process.
// - num_output_candidates: The number of output candidates to generate.
// - sampler: The sampler to sample the token ids from the logits.
// - decoded_ids: The decoded token ids from the external sampling process.
absl::StatusOr<Responses> DecodeCustomSampling(
    std::shared_ptr<LlmExecutor> executor, std::shared_ptr<Tokenizer> tokenizer,
    const std::vector<int>& stop_token_ids, int num_output_candidates,
    Sampler& sampler, litert::TensorBuffer& decoded_ids);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_PIPELINE_H_
