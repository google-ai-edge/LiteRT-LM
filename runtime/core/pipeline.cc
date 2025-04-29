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

#include <algorithm>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_replace.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/sampler.h"
#include "runtime/components/token_id_util.h"
#include "runtime/components/tokenizer.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/llm_executor.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/litert_status_util.h"
#include "runtime/util/status_macros.h"

namespace litert::lm {

// TODO(b/397975034) LLM Executor should return error when reaching the
// maximum number of kv-cache steps. Remove this once it is supported.
constexpr int kMaxDecodeStop = 128;

absl::StatusOr<int> Prefill(std::shared_ptr<LlmExecutor> executor,
                            std::shared_ptr<Tokenizer> tokenizer,
                            absl::string_view prompt, int bos_token_id,
                            bool wait_for_completion) {
  ASSIGN_OR_RETURN(auto ids_buffer,
                   tokenizer->TextToTensorBuffer(
                       prompt, /*prepend_token_ids=*/{bos_token_id}));
  LITERT_ASSIGN_OR_RETURN_ABSL(auto ids_buffer_span,
                               ReferTensorBufferAsSpan<int>(ids_buffer));
  if (ids_buffer_span.empty()) {
    return absl::InternalError("Input token ids are empty.");
  }
  const int last_token_id = ids_buffer_span.back();
  ExecutorPrefillParams params;
  params.SetWaitForCompletion(wait_for_completion);
  RETURN_IF_ERROR(
      executor->Prefill(ExecutorInputs(ExecutorTextData(std::move(ids_buffer)),
                                       std::nullopt, std::nullopt),
                        params));
  return last_token_id;
}

absl::StatusOr<Responses> Decode(std::shared_ptr<LlmExecutor> executor,
                                 std::shared_ptr<Tokenizer> tokenizer,
                                 const std::vector<int>& stop_token_ids) {
  Responses responses(/*num_output_candidates=*/1);
  LITERT_ASSIGN_OR_RETURN_ABSL(auto output_tokens,
                               CreateTensorBuffer<int>({1, 1}));
  std::vector<std::string>& response_texts =
      responses.GetMutableResponseTexts();
  // TODO(b/397975034) LLM Executor should return error when reaching the
  // maximum number of kv-cache steps.
  std::vector<bool> stop_tokens_found(1, false);  // assuming batch size is 1.
  for (int i = 0; i < kMaxDecodeStop; ++i) {
    RETURN_IF_ERROR(executor->Decode(output_tokens));
    LITERT_ASSIGN_OR_RETURN_ABSL(auto output_tokens_span,
                                 ReferTensorBufferAsSpan<int>(output_tokens));
    if (output_tokens_span.size() != 1) {
      return absl::InternalError("Unexpected number of decoded tokens.");
    }

    ASSIGN_OR_RETURN(auto result_token,
                     tokenizer->TensorBufferToText(output_tokens));
    result_token[0] = absl::StrReplaceAll(result_token[0], {{"▁", " "}});
    response_texts[0] += result_token[0];

    ASSIGN_OR_RETURN(
        bool should_stop,
        StopTokenFound(output_tokens_span, stop_token_ids, stop_tokens_found));
    if (should_stop) {
      break;
    }
  }
  return responses;
}

absl::StatusOr<Responses> DecodeCustomSampling(
    std::shared_ptr<LlmExecutor> executor, std::shared_ptr<Tokenizer> tokenizer,
    const std::vector<int>& stop_token_ids, int num_output_candidates,
    Sampler& sampler, litert::TensorBuffer& decoded_ids) {
  Responses responses(num_output_candidates);
  LITERT_ASSIGN_OR_RETURN_ABSL(
      auto output_logits,
      CreateTensorBuffer<float>(
          {num_output_candidates, 1, *(executor->GetVocabSize())}));
  // TODO(b/397975034) LLM Executor should return error when reaching the
  // maximum number of kv-cache steps.
  std::vector<bool> stop_tokens_found(num_output_candidates, false);

  auto scores_tensor = CreateTensorBuffer<float>({num_output_candidates});
  std::vector<float>& scores = responses.GetMutableScores();
  std::fill(scores.begin(), scores.end(), 0.0f);
  std::vector<int> num_decoded_tokens(num_output_candidates, 0);
  for (int i = 0; i < kMaxDecodeStop; ++i) {
    LITERT_ASSIGN_OR_RETURN(auto duplicate_decoded_ids,
                            decoded_ids.Duplicate());
    ExecutorInputs inputs(ExecutorTextData(std::move(duplicate_decoded_ids)),
                          std::nullopt, std::nullopt);
    RETURN_IF_ERROR(executor->Decode(inputs, output_logits));
    RETURN_IF_ERROR(sampler.SampleToIdAndScoreBuffer(output_logits, decoded_ids,
                                                     &(*scores_tensor)));

    ASSIGN_OR_RETURN(auto result_tokens,
                     tokenizer->TensorBufferToText(decoded_ids));

    // Update the stop_tokens_found vector with the latest decoded ids.
    LITERT_ASSIGN_OR_RETURN_ABSL(auto decoded_ids_span,
                                 ReferTensorBufferAsSpan<int>(decoded_ids));
    LITERT_ASSIGN_OR_RETURN_ABSL(
        auto scores_span, ReferTensorBufferAsSpan<float>(*scores_tensor));
    ASSIGN_OR_RETURN(
        bool should_stop,
        StopTokenFound(decoded_ids_span, stop_token_ids, stop_tokens_found));

    // Append the results to the final results vector. Note that only the
    // candidates that have not reached the stop token are added to the final
    // results vector.
    std::vector<std::string>& response_texts =
        responses.GetMutableResponseTexts();
    for (int j = 0; j < num_output_candidates; ++j) {
      // Only add the result if the stop token has not been found yet.
      if (!stop_tokens_found[j]) {
        // The tokenizer may return a token with a special character "▁" that
        // should be replaced with a space.
        result_tokens[j] = absl::StrReplaceAll(result_tokens[j], {{"▁", " "}});
        response_texts[j] += result_tokens[j];
        num_decoded_tokens[j]++;
        scores[j] += scores_span[j];
      }
    }
    if (should_stop) {
      break;
    }
  }
  for (int j = 0; j < num_output_candidates; ++j) {
    if (num_decoded_tokens[j] > 0) {
      scores[j] /= num_decoded_tokens[j];
    } else {
      scores[j] = -std::numeric_limits<float>::infinity();
    }
  }
  return responses;
}

}  // namespace litert::lm
