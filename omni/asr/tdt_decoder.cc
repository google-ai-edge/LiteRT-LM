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

namespace litert::omni::asr {
namespace {

constexpr absl::string_view kDecodeSignatureName = "decode";
constexpr absl::string_view kStatefulDecodeSignatureName = "decode_1";
constexpr size_t kNumDurations = 5;
// Hardcoded because CompileModel doesn't expose shape info.
constexpr size_t kNumFeatures = 1024;

template <typename T>
absl::StatusOr<size_t> GetNumOfElements(TensorBuffer& buffer) {
  LITERT_ASSIGN_OR_RETURN(auto num_bytes, buffer.PackedSize());
  return num_bytes / sizeof(T);
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

  std::optional<std::vector<TensorBuffer>> stateful_decode_input_buffers;
  std::optional<std::vector<TensorBuffer>> stateful_decode_output_buffers;
  auto stateful_inputs =
      runner->CreateInputBuffers(kStatefulDecodeSignatureName);
  if (stateful_inputs.ok()) {
    LITERT_RETURN_IF_ERROR(!stateful_inputs->empty());
    LITERT_ASSIGN_OR_RETURN(
        auto stateful_outputs,
        runner->CreateOutputBuffers(kStatefulDecodeSignatureName));
    LITERT_RETURN_IF_ERROR(!stateful_outputs.empty());
    stateful_decode_input_buffers = std::move(*stateful_inputs);
    stateful_decode_output_buffers = std::move(stateful_outputs);
  }

  return std::unique_ptr<TdtDecoder>(new TdtDecoder(
      runner, std::move(inputs), std::move(outputs),
      std::move(stateful_decode_input_buffers),
      std::move(stateful_decode_output_buffers), max_time_index, num_token_ids,
      num_logits_per_token, decode_start_token_id, decode_statefully_after));
}

TdtDecoder::TdtDecoder(
    LiteRtRunner* absl_nonnull runner,
    std::vector<TensorBuffer> decode_input_buffers,
    std::vector<TensorBuffer> decode_output_buffers,
    std::optional<std::vector<TensorBuffer>> stateful_decode_input_buffers,
    std::optional<std::vector<TensorBuffer>> stateful_decode_output_buffers,
    size_t max_time_index, size_t num_token_ids, size_t num_logits_per_token,
    int decode_start_token_id, int decode_statefully_after)
    : runner_(runner),
      decode_input_buffers_(std::move(decode_input_buffers)),
      decode_output_buffers_(std::move(decode_output_buffers)),
      stateful_decode_input_buffers_(std::move(stateful_decode_input_buffers)),
      stateful_decode_output_buffers_(
          std::move(stateful_decode_output_buffers)),
      input_states_buffers_(
          {&decode_input_buffers_[2], &decode_input_buffers_[3]}),
      output_states_buffers_(
          {&decode_output_buffers_[1], &decode_output_buffers_[2]}),
      max_time_index_(max_time_index),
      num_token_ids_(num_token_ids),
      num_logits_per_token_(num_logits_per_token),
      decode_start_token_id_(decode_start_token_id),
      decode_statefully_after_(decode_statefully_after) {}

absl::StatusOr<std::vector<SpeechRecognizer::DecodedToken>> TdtDecoder::Decode(
    std::vector<TensorBuffer>& encoder_outputs) {
  const int blank_token_id = decode_start_token_id_;

  // Clear the states as initial states are all zeros.
  for (auto* buffer : input_states_buffers_) {
    buffer->Clear();
  }

  std::vector<SpeechRecognizer::DecodedToken> decoded_tokens;
  // Start inference with decode signature for better quality in sequence start
  // as stateless decoding can keep context better than decode_1, stateful
  // decoding.
  absl::string_view current_signature = kDecodeSignatureName;
  TensorBuffer* current_token_ids_buffer = &decode_input_buffers_[1];
  TensorBuffer* current_logits_buffer = &decode_output_buffers_[0];
  int num_inference_token_ids = num_token_ids_;
  std::vector<int32_t> token_ids(num_token_ids_, 0);
  token_ids[0] = decode_start_token_id_;
  size_t token_index = 0;
  size_t time_index = 0;
  ABSL_ASSIGN_OR_RETURN(auto total_num_logits,
                        GetNumOfElements<float>(*current_logits_buffer));
  while (time_index < max_time_index_) {
    LITERT_RETURN_IF_ERROR(current_token_ids_buffer->Write<int32_t>(
        absl::MakeConstSpan(token_ids)));
    ABSL_ASSIGN_OR_RETURN(
        auto inputs, GetInputBuffersForInference(encoder_outputs,
                                                 *current_token_ids_buffer));
    ABSL_ASSIGN_OR_RETURN(auto outputs,
                          GetOutputBuffersForInference(*current_logits_buffer));
    LITERT_RETURN_IF_ERROR(runner_->Run(current_signature, inputs, outputs));

    std::vector<float> logits(total_num_logits);
    LITERT_RETURN_IF_ERROR(
        current_logits_buffer->Read<float>(absl::MakeSpan(logits)));

    size_t start_index_in_current_time_index =
        time_index * num_inference_token_ids * num_logits_per_token_;
    size_t start_index_of_token_id =
        start_index_in_current_time_index + token_index * num_logits_per_token_;
    size_t end_index_of_duration = start_index_in_current_time_index +
                                   (token_index + 1) * num_logits_per_token_;
    size_t end_index_of_token_id = end_index_of_duration - kNumDurations;
    auto max_token_it =
        std::max_element(logits.begin() + start_index_of_token_id,
                         logits.begin() + end_index_of_token_id);
    int token_id = static_cast<int>(
        std::distance(logits.begin() + start_index_of_token_id, max_token_it));
    if (token_id != blank_token_id) {
      decoded_tokens.push_back(SpeechRecognizer::DecodedToken{
          .token_id = token_id, .timestamp_ms = static_cast<int>(time_index)});
      if (num_inference_token_ids > 1) {
        ++token_index;
        if (token_index < num_inference_token_ids) {
          // Still room in the stateless token array: keep filling it, so the
          // LSTM states handed to decode_1 below are computed from real tokens
          // only (an unfilled zero slot skews them).
        } else if (stateful_decode_input_buffers_.has_value()) {
          // Switch to stateful decoding.
          current_signature = kStatefulDecodeSignatureName;
          current_token_ids_buffer = &(*stateful_decode_input_buffers_)[1];
          current_logits_buffer = &(*stateful_decode_output_buffers_)[0];
          num_inference_token_ids = 1;
          token_ids.resize(1, 0);
          token_index = 0;
          ABSL_ASSIGN_OR_RETURN(total_num_logits, GetNumOfElements<float>(
                                                      *current_logits_buffer));
        } else {
          break;
        }
      }
      token_ids[token_index] = token_id;
    }

    auto max_duration_it =
        std::max_element(logits.begin() + end_index_of_token_id,
                         logits.begin() + end_index_of_duration);
    int duration = static_cast<int>(
        std::distance(logits.begin() + end_index_of_token_id, max_duration_it));
    time_index += (duration == 0 && token_id == blank_token_id) ? 1 : duration;

    if (num_inference_token_ids == 1 && token_id != blank_token_id) {
      // Stateful RNN decoder: adopt the new LSTM states by swapping the input
      // and output state buffers — but only on a non-blank emission. In RNN-T
      // greedy decoding the prediction network advances only when a token is
      // emitted; adopting the state on blank steps re-consumes the last token
      // once per blank and degrades the transcript.
      std::swap(input_states_buffers_, output_states_buffers_);
    }
  }

  decoded_tokens.push_back(SpeechRecognizer::DecodedToken{
      .token_id = SpeechRecognizer::DecodedToken::kEndOfChunkTokenId,
      .timestamp_ms = static_cast<int>(max_time_index_)});

  return decoded_tokens;
}

absl::StatusOr<std::vector<TensorBuffer>>
TdtDecoder::GetInputBuffersForInference(
    std::vector<TensorBuffer>& encoder_outputs,
    TensorBuffer& token_ids_buffer) {
  std::vector<TensorBuffer> inputs;
  inputs.reserve(decode_input_buffers_.size());
  for (const auto& buffer : encoder_outputs) {
    LITERT_ASSIGN_OR_RETURN(auto dup, buffer.Duplicate());
    inputs.push_back(std::move(dup));
  }
  LITERT_ASSIGN_OR_RETURN(auto dup, token_ids_buffer.Duplicate());
  inputs.push_back(std::move(dup));
  for (auto* buffer : input_states_buffers_) {
    LITERT_ASSIGN_OR_RETURN(auto dup, buffer->Duplicate());
    inputs.push_back(std::move(dup));
  }
  return inputs;
}

absl::StatusOr<std::vector<TensorBuffer>>
TdtDecoder::GetOutputBuffersForInference(TensorBuffer& logits_buffer) {
  std::vector<TensorBuffer> outputs;
  outputs.reserve(decode_output_buffers_.size());
  LITERT_ASSIGN_OR_RETURN(auto dup, logits_buffer.Duplicate());
  outputs.push_back(std::move(dup));
  for (auto* buffer : output_states_buffers_) {
    LITERT_ASSIGN_OR_RETURN(auto dup, buffer->Duplicate());
    outputs.push_back(std::move(dup));
  }
  return outputs;
}

}  // namespace litert::omni::asr
