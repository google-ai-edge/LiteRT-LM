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

#include "runtime/executor/fake_llm_executor.h"

#include <algorithm>
#include <iterator>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/status_macros.h"

namespace litert::lm {
namespace {

// Converts the given ids to logits TensorBuffer in the shape of [batch_size,
// vocab_size].
void DecodeIdsToLogits(const std::vector<int>& ids, int vocab_size,
                       ::litert::TensorBuffer& output_logits) {
  auto logits_span = ReferTensorBufferAsSpan<float>(output_logits);
  for (int i = 0; i < ids.size(); ++i) {
    for (int j = 0; j < vocab_size; ++j) {
      int index = i * vocab_size + j;
      if (ids[i] == j) {
        (*logits_span)[index] = std::numeric_limits<float>::max();
      } else {
        (*logits_span)[index] = std::numeric_limits<float>::lowest();
      }
    }
  }
}

// Converts the given logits TensorBuffer to ids TensorBuffer. If no token is
// selected, use the last token in the decode tokens set which is the EOS token.
std::vector<std::vector<int>> DecodeLogitsToIds(
    int batch_size, int vocab_size, ::litert::TensorBuffer& output_logits,
    const std::vector<std::vector<int>>& decode_tokens_set) {
  auto masked_logits_span = ReferTensorBufferAsSpan<float>(output_logits);
  std::vector<std::vector<int>> output_tokens_vector;
  output_tokens_vector.resize(batch_size);
  for (int i = 0; i < batch_size; ++i) {
    auto batch_start = masked_logits_span->begin() + i * vocab_size;
    auto batch_end = batch_start + vocab_size;

    auto max_it = std::max_element(batch_start, batch_end);

    int best_token_id;
    // Check if any logit was greater than the minimum value.
    if (max_it != batch_end && *max_it > std::numeric_limits<float>::lowest()) {
      best_token_id = std::distance(batch_start, max_it);
    } else {
      // If all logits are std::numeric_limits<float>::lowest(),
      // default to the last token in the decode tokens set (EOS token).
      best_token_id = decode_tokens_set.back().back();
    }
    output_tokens_vector[i].push_back(best_token_id);
  }
  return output_tokens_vector;
}

// Checks if the given expected and actual spans are equivalent in terms of the
// size and values.
template <typename T>
absl::Status CheckEquivalent(absl::Span<T> expected, absl::Span<T> actual) {
  if (expected.size() != actual.size()) {
    return absl::InvalidArgumentError(absl::StrCat("Expected token size is ",
                                                   expected.size(), " but got ",
                                                   actual.size()));
  }
  for (int i = 0; i < expected.size(); ++i) {
    if (expected[i] != actual[i]) {
      return absl::InvalidArgumentError(absl::StrCat("Expected token at index ",
                                                     i, " is ", expected[i],
                                                     " but got ", actual[i]));
    }
  }
  return absl::OkStatus();
}

}  // namespace

FakeLlmExecutor::FakeLlmExecutor(
    int vocab_size, const std::vector<std::vector<int>>& prefill_tokens_set,
    const std::vector<std::vector<int>>& decode_tokens_set, int batch_size,
    std::optional<std::vector<float>> audio_embedding)
    : vocab_size_(vocab_size),
      prefill_tokens_set_(prefill_tokens_set),
      decode_tokens_set_(decode_tokens_set),
      audio_embedding_set_(std::move(audio_embedding)),
      batch_size_(batch_size),
      prefill_times_(0),
      decode_times_(0),
      executor_settings_(
          LlmExecutorSettings::CreateDefault(
              ModelAssets::Create("dummy_model_path").value(), Backend::CPU)
              .value()) {
  // Set default testing max num tokens to 1024.
  executor_settings_.SetMaxNumTokens(1024);
  current_step_ = 0;
  decode_delay_ = absl::ZeroDuration();
}

absl::Status FakeLlmExecutor::Prefill(const ExecutorInputs& inputs) {
  RETURN_IF_ERROR(prefill_status_);
  if (prefill_times_ >= prefill_tokens_set_.size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Prefill function has been called more times than the number of "
        "expected prefill tokens.",
        prefill_times_));
  }
  if (inputs.GetAudioDataPtr().ok()) {
    if (!audio_embedding_set_.has_value()) {
      return absl::InvalidArgumentError(
          "Audio embedding is not set in the fake LLM executor.");
    }
    ASSIGN_OR_RETURN(auto audio_embeddings, inputs.GetAudioEmbeddingsPtr());
    LITERT_ASSIGN_OR_RETURN(auto audio_embeddings_span,
                            ReferTensorBufferAsSpan<float>(*audio_embeddings));
    RETURN_IF_ERROR(CheckEquivalent(absl::MakeSpan(*audio_embedding_set_),
                                    audio_embeddings_span));
  }
  ASSIGN_OR_RETURN(auto text_data, inputs.GetTextDataPtr());
  auto text_token_ids_span =
      ReferTensorBufferAsSpan<int>(text_data->GetTokenIds());
  RETURN_IF_ERROR(
      CheckEquivalent(absl::MakeSpan(prefill_tokens_set_[prefill_times_]),
                      *text_token_ids_span));
  last_op_ = LastOp::kPrefill;
  processed_tokens_.AddProcessedTokens(prefill_tokens_set_[prefill_times_]);
  prefill_times_++;
  current_step_ += text_token_ids_span->size();
  return absl::OkStatus();
}

absl::Status FakeLlmExecutor::Prefill(
    const ExecutorInputs& inputs, const ExecutorPrefillParams& prefill_params) {
  RETURN_IF_ERROR(prefill_status_);
  if (prefill_params.GetWaitForCompletion()) {
    // Sleep some time here to simulate a synchronous prefill.
    // We can time the function time in test to make sure the code calls prefill
    // with a correct wait_for_completion flag.
    absl::SleepFor(absl::Milliseconds(100));
  }
  return Prefill(inputs);
}

absl::StatusOr<std::vector<std::vector<int>>> FakeLlmExecutor::Decode() {
  return Decode(ExecutorDecodeParams());
}

absl::StatusOr<std::vector<std::vector<int>>> FakeLlmExecutor::Decode(
    const ExecutorDecodeParams& decode_params) {
  TryDecodeDelay();
  RETURN_IF_ERROR(decode_status_);
  if (last_op_ == LastOp::kNone) {
    return absl::FailedPreconditionError(
        "Decode called without prior prefill or decode.");
  }
  if (decode_times_ >= decode_tokens_set_.size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Decode function has been called more times than the number of "
        "expected decode tokens.",
        decode_times_));
  }
  std::vector<std::vector<int>> output_tokens;
  if (decode_params.HasConstraintDecoder()) {
    // If constraint decoder is set, we will decode logits and apply the mask
    // from the constraint decoder to generate the final output tokens.
    auto constraint_decoder = decode_params.GetConstraintDecoder();
    // Get the last token ids from the last prefill or decode call.
    LITERT_ASSIGN_OR_RETURN(auto last_token_ids,
                            CreateTensorBuffer<int>({batch_size_, 1}));
    auto last_token_ids_span = ReferTensorBufferAsSpan<int>(last_token_ids);

    if (last_op_ == LastOp::kDecode) {
      if (decode_times_ == 0) {
        return absl::InternalError("LastOp is Decode but decode_times_ is 0");
      }
      const auto& last_decode_tokens = decode_tokens_set_[decode_times_ - 1];
      for (int i = 0; i < batch_size_; ++i) {
        (*last_token_ids_span)[i] = last_decode_tokens[i];
      }
      // Update the constraint state with the last token ids.
      RETURN_IF_ERROR(
          constraint_decoder->UpdateConstraintState(last_token_ids));
    }

    LITERT_ASSIGN_OR_RETURN(
        auto output_logits,
        CreateTensorBuffer<float>({batch_size_, 1, vocab_size_}));
    DecodeIdsToLogits(decode_tokens_set_[decode_times_], vocab_size_,
                      output_logits);
    // Apply the mask from the constraint decoder to the logits.
    RETURN_IF_ERROR(constraint_decoder->MaskLogits(output_logits));
    output_tokens = DecodeLogitsToIds(batch_size_, vocab_size_, output_logits,
                                      decode_tokens_set_);
  } else {
    for (int i = 0; i < decode_tokens_set_[decode_times_].size(); ++i) {
      output_tokens.push_back({decode_tokens_set_[decode_times_][i]});
    }
  }
  last_op_ = LastOp::kDecode;
  processed_tokens_.AddProcessedTokens(decode_tokens_set_[decode_times_]);
  decode_times_++;
  current_step_++;
  return output_tokens;
}

absl::Status FakeLlmExecutor::Decode(const ExecutorInputs& inputs,
                                     ::litert::TensorBuffer& output_logits) {
  TryDecodeDelay();
  RETURN_IF_ERROR(decode_status_);
  if (last_op_ == LastOp::kNone) {
    return absl::FailedPreconditionError(
        "Decode called without prior prefill or decode.");
  }
  if (decode_times_ >= decode_tokens_set_.size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Decode function has been called more times than the number of "
        "expected decode tokens.",
        decode_times_));
  }
  if (decode_times_ > 0) {
    // Check that the input tokens match the decode tokens from the last call.
    auto input_span =
        ReferTensorBufferAsSpan<int>(*(*inputs.GetTextTokenIdsPtr()));
    RETURN_IF_ERROR(CheckEquivalent(
        absl::MakeSpan(decode_tokens_set_[decode_times_ - 1]), *input_span));
  }
  DecodeIdsToLogits(decode_tokens_set_[decode_times_], vocab_size_,
                    output_logits);
  last_op_ = LastOp::kDecode;
  processed_tokens_.AddProcessedTokens(decode_tokens_set_[decode_times_]);
  decode_times_++;
  current_step_++;
  return absl::OkStatus();
}

absl::StatusOr<::litert::TensorBuffer> FakeLlmExecutor::DecodeLogits(
    const ExecutorInputs& inputs) {
  LITERT_ASSIGN_OR_RETURN(
      auto output_logits,
      CreateTensorBuffer<float>({batch_size_, 1, vocab_size_}));
  RETURN_IF_ERROR(Decode(inputs, output_logits));
  return output_logits;
}

void FakeLlmExecutor::TryDecodeDelay() {
  if (decode_delay_ > absl::ZeroDuration()) {
    absl::SleepFor(decode_delay_);
    decode_delay_ = absl::ZeroDuration();
  }
}

absl::Status FakeLlmExecutor::Reset() {
  prefill_times_ = 0;
  decode_times_ = 0;
  current_step_ = 0;
  last_op_ = LastOp::kNone;
  return absl::OkStatus();
}

}  // namespace litert::lm
