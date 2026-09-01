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

#include "omni/asr/tdt_decoder.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "omni/asr/speech_recognizer.h"
#include "omni/base/litert_runner.h"
#include "omni/base/stateful_litert_runner.h"

namespace litert::omni::asr {
namespace {

constexpr absl::string_view kDecodeSignatureName = "decode";
constexpr absl::string_view kStatefulDecodeSignatureName = "decode_1";
constexpr size_t kNumDurations = 5;
// Hardcoded because CompileModel doesn't expose shape info.
constexpr size_t kNumFeatures = 1024;

template <typename T>
absl::StatusOr<size_t> GetNumOfElements(const TensorBuffer& buffer) {
  LITERT_ASSIGN_OR_RETURN(auto num_bytes, buffer.PackedSize());
  return num_bytes / sizeof(T);
}

absl::StatusOr<std::vector<TensorBuffer>> GetInputBuffersForInference(
    absl::Span<const TensorBuffer> encoder_outputs,
    const TensorBuffer& token_ids_buffer,
    absl::Span<const TensorBuffer> state_buffers) {
  std::vector<TensorBuffer> inputs;
  inputs.reserve(encoder_outputs.size() + 1 + state_buffers.size());
  for (const auto& buffer : encoder_outputs) {
    LITERT_ASSIGN_OR_RETURN(auto dup, buffer.Duplicate());
    inputs.push_back(std::move(dup));
  }
  LITERT_ASSIGN_OR_RETURN(auto dup_tok, token_ids_buffer.Duplicate());
  inputs.push_back(std::move(dup_tok));
  for (const auto& buffer : state_buffers) {
    LITERT_ASSIGN_OR_RETURN(auto dup_st, buffer.Duplicate());
    inputs.push_back(std::move(dup_st));
  }
  return inputs;
}

}  // namespace

absl::StatusOr<std::unique_ptr<TdtDecoder>> TdtDecoder::Create(
    LiteRtRunner* absl_nonnull runner, int decode_start_token_id,
    int decode_statefully_after) {
  LITERT_ASSIGN_OR_RETURN(auto inputs,
                          runner->CreateInputBuffers(kDecodeSignatureName));
  // encode output, token ids, input LSTM state 1, input LSTM state 2.
  LITERT_RETURN_IF_ERROR(inputs.size() >= 4);
  LITERT_ASSIGN_OR_RETURN(auto outputs,
                          runner->CreateOutputBuffers(kDecodeSignatureName));
  // logits, output LSTM state 1, output LSTM state 2.
  LITERT_RETURN_IF_ERROR(outputs.size() >= 3);

  ABSL_ASSIGN_OR_RETURN(auto num_input0, GetNumOfElements<float>(inputs[0]));
  size_t max_time_index = num_input0 / kNumFeatures;
  ABSL_ASSIGN_OR_RETURN(auto num_token_ids,
                        GetNumOfElements<int32_t>(inputs[1]));
  ABSL_ASSIGN_OR_RETURN(auto total_num_logits,
                        GetNumOfElements<float>(outputs[0]));
  size_t num_logits_per_token =
      total_num_logits / num_token_ids / max_time_index;

  std::unique_ptr<StatefulLiteRtRunner> stateful_runner;
  std::optional<TensorBuffer> stateful_decode_token_ids_buf;

  auto stateful_runner_res = StatefulLiteRtRunnerImpl::Create(
      runner, kStatefulDecodeSignatureName,
      /*num_non_state_inputs=*/2, /*num_non_state_outputs=*/1);
  if (stateful_runner_res.ok()) {
    stateful_runner = std::move(*stateful_runner_res);
    auto non_state_in = stateful_runner->GetNonStateInputBuffers();
    LITERT_RETURN_IF_ERROR(non_state_in.size() >= 2);
    LITERT_ASSIGN_OR_RETURN(stateful_decode_token_ids_buf,
                            non_state_in[1].Duplicate());

    // Wire up stateless decode outputs directly into stateful runner active
    // input states so transitioning from stateless to stateful requires zero
    // memory copies.
    auto active_states = stateful_runner->GetActiveInputStates();
    for (size_t s = 0; s < active_states.size() && (s + 1) < outputs.size();
         ++s) {
      LITERT_ASSIGN_OR_RETURN(outputs[s + 1], active_states[s].Duplicate());
    }
  }

  return std::unique_ptr<TdtDecoder>(new TdtDecoder(
      runner, std::move(stateful_runner), std::move(inputs), std::move(outputs),
      std::move(stateful_decode_token_ids_buf), max_time_index, num_token_ids,
      num_logits_per_token, decode_start_token_id, decode_statefully_after));
}

TdtDecoder::TdtDecoder(
    LiteRtRunner* absl_nonnull runner,
    std::unique_ptr<StatefulLiteRtRunner> stateful_runner,
    std::vector<TensorBuffer> decode_input_buffers,
    std::vector<TensorBuffer> decode_output_buffers,
    std::optional<TensorBuffer> stateful_decode_token_ids_buffer,
    size_t max_time_index, size_t num_token_ids, size_t num_logits_per_token,
    int decode_start_token_id, int decode_statefully_after)
    : runner_(runner),
      stateful_runner_(std::move(stateful_runner)),
      decode_input_buffers_(std::move(decode_input_buffers)),
      decode_output_buffers_(std::move(decode_output_buffers)),
      stateful_decode_token_ids_buffer_(
          std::move(stateful_decode_token_ids_buffer)),
      max_time_index_(max_time_index),
      num_token_ids_(num_token_ids),
      num_logits_per_token_(num_logits_per_token),
      decode_start_token_id_(decode_start_token_id),
      decode_statefully_after_(decode_statefully_after) {}

absl::StatusOr<std::vector<SpeechRecognizer::DecodedToken>> TdtDecoder::Decode(
    std::vector<TensorBuffer>& encoder_outputs) {
  const int blank_token_id = decode_start_token_id_;

  if (stateful_runner_) {
    LITERT_RETURN_IF_ERROR(stateful_runner_->Reset());
  }
  for (size_t i = 2; i < decode_input_buffers_.size(); ++i) {
    LITERT_RETURN_IF_ERROR(decode_input_buffers_[i].Clear());
  }

  std::vector<SpeechRecognizer::DecodedToken> decoded_tokens;
  // Start inference with decode signature for better quality in sequence start
  // as stateless decoding can keep context better than decode_1, stateful
  // decoding.
  TensorBuffer* current_token_ids_buffer = &decode_input_buffers_[1];
  int num_inference_token_ids = num_token_ids_;
  std::vector<int32_t> token_ids(num_token_ids_, 0);
  token_ids[0] = decode_start_token_id_;
  size_t token_index = 0;
  size_t time_index = 0;

  while (time_index < max_time_index_) {
    LITERT_RETURN_IF_ERROR(current_token_ids_buffer->Write<int32_t>(
        absl::MakeConstSpan(token_ids)));

    const TensorBuffer* current_logits_buffer = nullptr;
    bool was_stateful_step = false;

    if (num_inference_token_ids > 1 || !stateful_runner_) {
      // Stateless decode using LiteRtRunner
      LITERT_ASSIGN_OR_RETURN(
          auto step_inputs,
          GetInputBuffersForInference(
              encoder_outputs, *current_token_ids_buffer,
              absl::MakeSpan(decode_input_buffers_).subspan(2)));
      LITERT_RETURN_IF_ERROR(runner_->Run(kDecodeSignatureName, step_inputs,
                                          decode_output_buffers_));
      current_logits_buffer = &decode_output_buffers_[0];
    } else {
      // Stateful decode using StatefulLiteRtRunner
      LITERT_ASSIGN_OR_RETURN(
          auto step_inputs,
          GetInputBuffersForInference(encoder_outputs,
                                      *current_token_ids_buffer, {}));
      LITERT_ASSIGN_OR_RETURN(
          auto step_outputs,
          stateful_runner_->Step(step_inputs, /*auto_commit_state=*/false));
      if (step_outputs.empty()) {
        return absl::InternalError("No outputs returned from stateful runner.");
      }
      current_logits_buffer = &step_outputs[0];
      was_stateful_step = true;
    }

    LITERT_ASSIGN_OR_RETURN(
        auto lock, TensorBufferScopedLock::Create<const float>(
                       *current_logits_buffer, TensorBuffer::LockMode::kRead));
    const float* logits = lock.second;

    size_t start_index_in_current_time_index =
        time_index * num_inference_token_ids * num_logits_per_token_;
    size_t start_index_of_token_id =
        start_index_in_current_time_index + token_index * num_logits_per_token_;
    size_t end_index_of_duration = start_index_in_current_time_index +
                                   (token_index + 1) * num_logits_per_token_;
    size_t end_index_of_token_id = end_index_of_duration - kNumDurations;
    auto max_token_it = std::max_element(logits + start_index_of_token_id,
                                         logits + end_index_of_token_id);
    int token_id = static_cast<int>(
        std::distance(logits + start_index_of_token_id, max_token_it));
    if (token_id != blank_token_id) {
      decoded_tokens.push_back(SpeechRecognizer::DecodedToken{
          .token_id = token_id, .timestamp_ms = static_cast<int>(time_index)});
      if (num_inference_token_ids > 1) {
        ++token_index;
        if (token_index < num_inference_token_ids) {
          // Still room in the stateless token array: keep filling it, so the
          // LSTM states handed to decode_1 below are computed from real tokens
          // only (an unfilled zero slot skews them).
        } else if (stateful_decode_token_ids_buffer_.has_value()) {
          // Switch to stateful decoding.
          current_token_ids_buffer = &(*stateful_decode_token_ids_buffer_);
          num_inference_token_ids = 1;
          token_ids.resize(1, 0);
          token_index = 0;
        } else {
          break;
        }
      }
      token_ids[token_index] = token_id;
    }

    auto max_duration_it = std::max_element(logits + end_index_of_token_id,
                                            logits + end_index_of_duration);
    int duration = static_cast<int>(
        std::distance(logits + end_index_of_token_id, max_duration_it));
    time_index += (duration == 0 && token_id == blank_token_id) ? 1 : duration;

    if (was_stateful_step && token_id != blank_token_id && stateful_runner_) {
      // Stateful RNN decoder: adopt the new LSTM states by swapping the input
      // and output state buffers — but only on a non-blank emission. In RNN-T
      // greedy decoding the prediction network advances only when a token is
      // emitted; adopting the state on blank steps re-consumes the last token
      // once per blank and degrades the transcript.
      LITERT_RETURN_IF_ERROR(stateful_runner_->CommitState());
    }
  }

  decoded_tokens.push_back(SpeechRecognizer::DecodedToken{
      .token_id = SpeechRecognizer::DecodedToken::kEndOfChunkTokenId,
      .timestamp_ms = static_cast<int>(max_time_index_)});

  return decoded_tokens;
}

}  // namespace litert::omni::asr
