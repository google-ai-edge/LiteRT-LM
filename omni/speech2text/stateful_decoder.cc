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

#include "omni/speech2text/stateful_decoder.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "omni/base/litert_runner.h"
#include "omni/base/stateful_litert_runner.h"
#include "omni/speech2text/speech_recognizer.h"
#include "omni/speech2text/utils.h"

namespace litert::omni::asr {

absl::StatusOr<StatefulDecoder::RunnerContext>
StatefulDecoder::InitRunnerContext(
    LiteRtRunner* absl_nonnull runner, absl::string_view signature_name,
    size_t num_non_state_inputs, size_t num_non_state_outputs) {
  if (runner == nullptr) {
    return absl::InvalidArgumentError("Runner cannot be null.");
  }
  if (num_non_state_inputs == 0) {
    return absl::InvalidArgumentError("num_non_state_inputs must be positive.");
  }
  if (num_non_state_outputs == 0) {
    return absl::InvalidArgumentError(
        "num_non_state_outputs must be positive.");
  }

  LITERT_ASSIGN_OR_RETURN(
      auto stateful_runner,
      StatefulLiteRtRunnerImpl::Create(runner, signature_name,
                                       num_non_state_inputs,
                                       num_non_state_outputs));

  auto out_buffers = stateful_runner->GetNonStateOutputBuffers();
  if (out_buffers.empty()) {
    return absl::InternalError("No non-state output buffers available.");
  }
  LITERT_ASSIGN_OR_RETURN(auto num_bytes, out_buffers[0].PackedSize());
  size_t num_logits_per_token = num_bytes / sizeof(float);
  if (num_logits_per_token == 0) {
    return absl::InternalError("Logits buffer contains 0 elements.");
  }

  // Auto-detect which non-state input corresponds to the token buffer.
  // The token buffer is a scalar or 1-element tensor (e.g. 4 bytes), which is
  // significantly smaller than encoder feature/hidden state buffers.
  size_t token_input_index = 0;
  auto in_buffers = stateful_runner->GetNonStateInputBuffers();
  if (in_buffers.size() > 1) {
    size_t min_size = std::numeric_limits<size_t>::max();
    for (size_t i = 0; i < in_buffers.size(); ++i) {
      auto packed_size = in_buffers[i].PackedSize();
      if (packed_size.HasValue() && *packed_size < min_size) {
        min_size = *packed_size;
        token_input_index = i;
      }
    }
  }

  return RunnerContext{
      .stateful_runner = std::move(stateful_runner),
      .token_input_index = token_input_index,
      .num_logits_per_token = num_logits_per_token,
  };
}

absl::StatusOr<std::unique_ptr<StatefulDecoder>> StatefulDecoder::Create(
    LiteRtRunner* absl_nonnull runner, absl::string_view signature_name,
    size_t num_non_state_inputs, size_t num_non_state_outputs,
    std::vector<int> decode_start_token_ids, int decode_stop_token_id,
    int decode_skip_until_token_id, int max_decode_steps) {
  if (max_decode_steps <= 0) {
    return absl::InvalidArgumentError("max_decode_steps must be positive.");
  }

  LITERT_ASSIGN_OR_RETURN(
      auto ctx, InitRunnerContext(runner, signature_name, num_non_state_inputs,
                                  num_non_state_outputs));

  return absl::WrapUnique(new StatefulDecoder(
      std::move(ctx.stateful_runner), ctx.token_input_index,
      ctx.num_logits_per_token, std::move(decode_start_token_ids),
      decode_stop_token_id, decode_skip_until_token_id, max_decode_steps));
}

StatefulDecoder::StatefulDecoder(
    std::unique_ptr<StatefulLiteRtRunner> stateful_runner,
    size_t token_input_index, size_t num_logits_per_token,
    std::vector<int> decode_start_token_ids, int decode_stop_token_id,
    int decode_skip_until_token_id, int max_decode_steps)
    : stateful_runner_(std::move(stateful_runner)),
      token_input_index_(token_input_index),
      num_logits_per_token_(num_logits_per_token),
      decode_start_token_ids_(std::move(decode_start_token_ids)),
      decode_stop_token_id_(decode_stop_token_id),
      decode_skip_until_token_id_(decode_skip_until_token_id),
      max_decode_steps_(max_decode_steps) {}

absl::Status StatefulDecoder::ResetAndInitStepInputs(
    absl::Span<const ::litert::TensorBuffer> encoder_outputs) {
  if (encoder_outputs.empty()) {
    return absl::InvalidArgumentError("Encoder outputs cannot be empty.");
  }
  ABSL_RETURN_IF_ERROR(stateful_runner_->Reset());
  cached_output_lock_.reset();

  auto non_state_inputs = stateful_runner_->GetNonStateInputBuffers();
  if (non_state_inputs.empty()) {
    return absl::InternalError("Non-state input buffers cannot be empty.");
  }

  // Populate step inputs: duplicate the token buffer at token_input_index_,
  // and map encoder outputs to the remaining non-state inputs.
  step_inputs_.clear();
  step_inputs_.resize(non_state_inputs.size());

  LITERT_ASSIGN_OR_RETURN(auto token_buf_dup,
                          non_state_inputs[token_input_index_].Duplicate());
  step_inputs_[token_input_index_] = std::move(token_buf_dup);

  size_t enc_idx = 0;
  for (size_t i = 0; i < non_state_inputs.size(); ++i) {
    if (i == token_input_index_) {
      continue;
    }
    if (enc_idx < encoder_outputs.size()) {
      LITERT_ASSIGN_OR_RETURN(auto enc_dup,
                              encoder_outputs[enc_idx].Duplicate());
      step_inputs_[i] = std::move(enc_dup);
      ++enc_idx;
    } else {
      LITERT_ASSIGN_OR_RETURN(auto in_dup, non_state_inputs[i].Duplicate());
      step_inputs_[i] = std::move(in_dup);
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<absl::Span<const float>> StatefulDecoder::Step(int32_t token_id) {
  if (step_inputs_.empty()) {
    return absl::FailedPreconditionError(
        "ResetAndInitStepInputs() must be called before Step().");
  }
  cached_output_lock_.reset();

  LITERT_RETURN_IF_ERROR(step_inputs_[token_input_index_].Write<int32_t>(
      absl::MakeConstSpan(&token_id, 1)));
  ABSL_ASSIGN_OR_RETURN(
      auto outputs,
      stateful_runner_->Step(step_inputs_, /*auto_commit_state=*/true));
  if (outputs.empty()) {
    return absl::InternalError("Step returned 0 output buffers.");
  }

  LITERT_ASSIGN_OR_RETURN(
      auto lock_and_addr,
      ::litert::TensorBufferScopedLock::Create<const float>(
          outputs[0], ::litert::TensorBuffer::LockMode::kRead));
  cached_output_lock_.emplace(std::move(lock_and_addr.first));
  return absl::MakeConstSpan(lock_and_addr.second, num_logits_per_token_);
}

absl::StatusOr<int32_t> StatefulDecoder::FeedPromptTokens(
    absl::Span<const int> start_token_ids) {
  int32_t current_token = -1;
  for (int tok : start_token_ids) {
    ABSL_ASSIGN_OR_RETURN(absl::Span<const float> logits, Step(tok));
    auto max_it = std::max_element(logits.begin(), logits.end());
    current_token = static_cast<int32_t>(max_it - logits.begin());
  }
  return current_token;
}

absl::StatusOr<std::vector<SpeechRecognizer::DecodedToken>>
StatefulDecoder::Decode(std::vector<::litert::TensorBuffer>& encoder_outputs,
                        std::optional<DecodeRange> /*decode_range*/) {
  ABSL_RETURN_IF_ERROR(ResetAndInitStepInputs(encoder_outputs));

  ABSL_ASSIGN_OR_RETURN(int32_t current_token,
                        FeedPromptTokens(decode_start_token_ids_));

  std::vector<SpeechRecognizer::DecodedToken> decoded_tokens;
  bool seen_skip_until_token_id = decode_skip_until_token_id_ < 0;

  for (int step = 0; step < max_decode_steps_; ++step) {
    if (current_token < 0) {
      break;
    }
    if (decode_stop_token_id_ >= 0 && current_token == decode_stop_token_id_) {
      break;
    }

    if (seen_skip_until_token_id) {
      if (TruncateOnTrailingRepetition(decoded_tokens, current_token)) {
        break;
      }
      decoded_tokens.push_back(SpeechRecognizer::DecodedToken{
          .token_id = current_token, .timestamp_ms = std::nullopt});
    } else if (current_token == decode_skip_until_token_id_) {
      seen_skip_until_token_id = true;
    }

    ABSL_ASSIGN_OR_RETURN(absl::Span<const float> logits, Step(current_token));
    auto max_it = std::max_element(logits.begin(), logits.end());
    current_token = static_cast<int32_t>(max_it - logits.begin());
  }

  return decoded_tokens;
}

}  // namespace litert::omni::asr
