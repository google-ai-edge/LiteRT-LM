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

#include "omni/asr/stateless_decoder.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
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
constexpr int kTokenIdInputIndex = -2;
constexpr int kMaskInputIndex = -1;
constexpr float kMaskedInFloatValue = 0.0f;
constexpr float kMaskedOutFloatValue =
    -0.7f * std::numeric_limits<float>::max();

}  // namespace

absl::StatusOr<std::unique_ptr<StatelessDecoder>> StatelessDecoder::Create(
    LiteRtRunner* absl_nonnull runner, int decode_start_token_id,
    int decode_stop_token_id, int decode_skip_until_token_id) {
  LITERT_ASSIGN_OR_RETURN(auto inputs,
                          runner->CreateInputBuffers(kDecodeSignatureName));
  LITERT_RETURN_IF_ERROR(inputs.size() > 2);
  LITERT_ASSIGN_OR_RETURN(auto outputs,
                          runner->CreateOutputBuffers(kDecodeSignatureName));
  LITERT_RETURN_IF_ERROR(!outputs.empty());

  const size_t mask_input_index = inputs.size() + kMaskInputIndex;
  LITERT_ASSIGN_OR_RETURN(auto mask_bytes,
                          inputs[mask_input_index].PackedSize());
  const size_t num_mask_entries = mask_bytes / sizeof(float);
  const size_t n = static_cast<size_t>(std::sqrt(num_mask_entries));
  std::vector<float> causal_mask(num_mask_entries, kMaskedOutFloatValue);
  for (size_t r = 0; r < n; ++r) {
    for (size_t c = 0; c <= r; ++c) {
      causal_mask[r * n + c] = kMaskedInFloatValue;
    }
  }
  LITERT_RETURN_IF_ERROR(
      inputs[mask_input_index].Write<float>(absl::MakeConstSpan(causal_mask)));

  LITERT_ASSIGN_OR_RETURN(auto num_output_bytes, outputs[0].PackedSize());
  size_t total_logits = num_output_bytes / sizeof(float);

  size_t num_token_ids = 1;
  const size_t token_id_input_index = inputs.size() + kTokenIdInputIndex;
  if (token_id_input_index < inputs.size()) {
    LITERT_ASSIGN_OR_RETURN(auto token_bytes,
                            inputs[token_id_input_index].PackedSize());
    num_token_ids = token_bytes / sizeof(int32_t);
  }
  LITERT_RETURN_IF_ERROR(num_token_ids > 0);
  size_t num_logits_per_token = total_logits / num_token_ids;
  LITERT_RETURN_IF_ERROR(num_logits_per_token > 0);

  return std::unique_ptr<StatelessDecoder>(new StatelessDecoder(
      runner, std::move(inputs), std::move(outputs), num_logits_per_token,
      num_token_ids, decode_start_token_id, decode_stop_token_id,
      decode_skip_until_token_id));
}

StatelessDecoder::StatelessDecoder(
    LiteRtRunner* absl_nonnull runner,
    std::vector<::litert::TensorBuffer> decode_input_buffers,
    std::vector<::litert::TensorBuffer> decode_output_buffers,
    size_t num_logits_per_token, size_t num_token_ids,
    int decode_start_token_id, int decode_stop_token_id,
    int decode_skip_until_token_id)
    : runner_(runner),
      decode_input_buffers_(std::move(decode_input_buffers)),
      decode_output_buffers_(std::move(decode_output_buffers)),
      num_logits_per_token_(num_logits_per_token),
      num_token_ids_(num_token_ids),
      decode_start_token_id_(decode_start_token_id),
      decode_stop_token_id_(decode_stop_token_id),
      decode_skip_until_token_id_(decode_skip_until_token_id) {}

absl::StatusOr<std::vector<SpeechRecognizer::DecodedToken>>
StatelessDecoder::Decode(std::vector<::litert::TensorBuffer>& encoder_outputs) {
  std::vector<::litert::TensorBuffer> inputs;
  inputs.reserve(decode_input_buffers_.size());
  // Pass encoder outputs to the first set of input buffers.
  for (const auto& buffer : encoder_outputs) {
    LITERT_ASSIGN_OR_RETURN(auto dup, buffer.Duplicate());
    inputs.push_back(std::move(dup));
  }
  // Fill the rest of the input buffers with allocated ones which should be
  // token ids and mask.
  for (size_t i = inputs.size(); i < decode_input_buffers_.size(); ++i) {
    LITERT_ASSIGN_OR_RETURN(auto dup, decode_input_buffers_[i].Duplicate());
    inputs.push_back(std::move(dup));
  }

  const size_t token_id_input_index =
      decode_input_buffers_.size() + kTokenIdInputIndex;

  std::vector<SpeechRecognizer::DecodedToken> decoded_tokens;
  std::vector<int32_t> token_ids(num_token_ids_, 0);
  if (decode_start_token_id_ >= 0) {
    token_ids[0] = decode_start_token_id_;
  }
  bool seen_skip_until_token_id = decode_skip_until_token_id_ < 0;

  std::vector<float> logits(num_logits_per_token_ * num_token_ids_);
  for (size_t i = 0; i < num_token_ids_ - 1; ++i) {
    LITERT_RETURN_IF_ERROR(inputs[token_id_input_index].Write<int32_t>(
        absl::MakeConstSpan(token_ids)));
    LITERT_RETURN_IF_ERROR(
        runner_->Run(kDecodeSignatureName, inputs, decode_output_buffers_));
    LITERT_RETURN_IF_ERROR(
        decode_output_buffers_[0].Read<float>(absl::MakeSpan(logits)));

    const size_t start_index = i * num_logits_per_token_;
    const size_t end_index = (i + 1) * num_logits_per_token_;
    if (start_index >= logits.size() || end_index > logits.size()) {
      break;
    }

    auto max_it = std::max_element(logits.begin() + start_index,
                                   logits.begin() + end_index);
    const int token_id =
        static_cast<int>(std::distance(logits.begin() + start_index, max_it));
    if (decode_stop_token_id_ >= 0 && token_id == decode_stop_token_id_) {
      break;
    }

    if (seen_skip_until_token_id) {
      decoded_tokens.push_back(SpeechRecognizer::DecodedToken{
          .token_id = token_id, .timestamp_ms = std::nullopt});
    } else if (token_id == decode_skip_until_token_id_) {
      seen_skip_until_token_id = true;
    }

    token_ids[i + 1] = token_id;
  }

  return decoded_tokens;
}

}  // namespace litert::omni::asr
