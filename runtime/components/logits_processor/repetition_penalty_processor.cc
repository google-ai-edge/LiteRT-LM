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

#include "runtime/components/logits_processor/repetition_penalty_processor.h"

#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/logits_processor/repetition_penalty_config.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/status_macros.h"
#include "tflite/types/half.h"  // from @litert

namespace litert::lm {
namespace {

constexpr int kUninitializedToken = -1;

}  // namespace

RepetitionPenaltyProcessor::RepetitionPenaltyProcessor(
    int batch_size, int vocab_size, RepetitionPenaltyConfig config)
    : vocab_size_(vocab_size),
      config_(std::move(config)),
      batch_states_(batch_size) {
  if (config_.window_size() > 0) {
    for (BatchState& batch_state : batch_states_) {
      // Initialize with -1 to indicate no token has been seen yet in the
      // window.
      batch_state.token_history.resize(config_.window_size(),
                                       kUninitializedToken);
    }
  }
}

absl::Status RepetitionPenaltyProcessor::ProcessLogits(
    absl::Span<float> logits,
    absl::Span<const ::litert::Layout::Dim> logits_dims) {
  return ProcessLogitsImpl(logits, logits_dims);
}

absl::Status RepetitionPenaltyProcessor::ProcessLogits(
    absl::Span<tflite::half> logits,
    absl::Span<const ::litert::Layout::Dim> logits_dims) {
  return ProcessLogitsImpl(logits, logits_dims);
}

absl::Status RepetitionPenaltyProcessor::ProcessLogits(
    ::litert::TensorBuffer& logits) {
  LITERT_ASSIGN_OR_RETURN(auto tensor_type, logits.TensorType());
  auto element_type = tensor_type.ElementType();
  auto layout = tensor_type.Layout();

  if (element_type == ::litert::ElementType::Float32) {
    LITERT_ASSIGN_OR_RETURN(
        auto span, ::litert::lm::ReferTensorBufferAsSpan<float>(logits));
    return ProcessLogits(absl::MakeSpan(span.data(), span.size()),
                         absl::MakeConstSpan(layout.Dimensions()));
  } else if (element_type == ::litert::ElementType::Float16) {
    LITERT_ASSIGN_OR_RETURN(
        auto span, ::litert::lm::ReferTensorBufferAsSpan<tflite::half>(logits));
    return ProcessLogits(absl::MakeSpan(span.data(), span.size()),
                         absl::MakeConstSpan(layout.Dimensions()));
  }

  return absl::InvalidArgumentError(
      "Unsupported logits tensor format element type.");
}

absl::Status RepetitionPenaltyProcessor::UpdateState(
    const ::litert::TensorBuffer& next_token_ids) {
  LITERT_ASSIGN_OR_RETURN(
      auto next_token_ids_span,
      ::litert::lm::ReferTensorBufferAsSpan<int>(next_token_ids));

  return UpdateState(
      absl::MakeSpan(next_token_ids_span.data(), next_token_ids_span.size()));
}

absl::Status RepetitionPenaltyProcessor::UpdateState(
    absl::Span<const int> next_token_ids) {
  if (next_token_ids.size() != batch_states_.size()) {
    return absl::InvalidArgumentError(
        "next_token_ids size must match batch_size");
  }

  for (int token_id : next_token_ids) {
    if (token_id < 0 || token_id >= vocab_size_) {
      return absl::InvalidArgumentError(
          "next_token_ids contains invalid token id");
    }
  }

  for (int batch_idx = 0; batch_idx < batch_states_.size(); ++batch_idx) {
    int token_id = next_token_ids[batch_idx];
    BatchState& batch_state = batch_states_[batch_idx];

    ++batch_state.token_counts[token_id];

    if (config_.window_size() > 0) {
      int expired_token_id = batch_state.token_history[token_history_index_];
      // Keep tracking the most recent `window_size` tokens.
      batch_state.token_history[token_history_index_] = token_id;

      auto it = batch_state.token_counts.find(expired_token_id);
      if (it == batch_state.token_counts.end()) {
        continue;
      }
      // Decrement the count of the token that is falling out of the window.
      --it->second;
      if (it->second == 0) {
        batch_state.token_counts.erase(it);
      }
    }
  }

  // Update the token history index at the end of the call, after all states
  // have been updated.
  if (config_.window_size() > 0) {
    token_history_index_ = (token_history_index_ + 1) % config_.window_size();
  }

  return absl::OkStatus();
}

template <typename T>
absl::Status RepetitionPenaltyProcessor::ProcessLogitsImpl(
    absl::Span<T> logits, absl::Span<const ::litert::Layout::Dim> logits_dims) {
  if (logits_dims.size() != 3 || logits_dims[0] != batch_states_.size() ||
      logits_dims[1] != 1 || logits_dims[2] != vocab_size_) {
    return absl::InvalidArgumentError(
        "Logits dimensions must be [batch_size, 1, vocab_size].");
  }

  if (logits.size() != vocab_size_ * batch_states_.size()) {
    return absl::InvalidArgumentError("Logits span size incorrectly mapped.");
  }

  for (int batch_idx = 0; batch_idx < batch_states_.size(); ++batch_idx) {
    BatchState& batch_state = batch_states_[batch_idx];
    int batch_offset = batch_idx * vocab_size_;

    for (int vocab_idx = 0; vocab_idx < vocab_size_; ++vocab_idx) {
      int count = batch_state.token_counts[vocab_idx];

      if (count == 0) {
        continue;
      }

      // Cast via proxy up to full float so penalty scaling works normally
      // without truncation losses.
      float val = static_cast<float>(logits[batch_offset + vocab_idx]);

      // 1. Multiplicative Repetition Penalty
      if (config_.repetition_penalty() > 1.0f) {
        if (val > 0.0f) {
          val /= config_.repetition_penalty();
        } else {
          val *= config_.repetition_penalty();
        }
      }

      // 2 & 3. Additive Presence and Frequency Penalties
      val -= (config_.presence_penalty() + count * config_.frequency_penalty());

      logits[batch_offset + vocab_idx] = static_cast<T>(val);
    }
  }

  return absl::OkStatus();
}

absl::Span<const RepetitionPenaltyProcessor::BatchState>
RepetitionPenaltyProcessor::GetBatchStatesForTesting() {
  return batch_states_;
}

}  // namespace litert::lm
