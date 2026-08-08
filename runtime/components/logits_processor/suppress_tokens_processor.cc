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

#include "runtime/components/logits_processor/suppress_tokens_processor.h"

#include <limits>
#include <utility>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/logits_processor/suppress_tokens_config.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "tflite/types/half.h"  // from @litert

namespace litert::lm {

SuppressTokensProcessor::SuppressTokensProcessor(int batch_size, int vocab_size,
                                                 SuppressTokensConfig config)
    : batch_size_(batch_size),
      vocab_size_(vocab_size),
      config_(std::move(config)) {}

absl::Status SuppressTokensProcessor::ProcessLogits(
    absl::Span<float> logits,
    absl::Span<const ::litert::Layout::Dim> logits_dims) {
  return ProcessLogitsImpl(logits, logits_dims);
}

absl::Status SuppressTokensProcessor::ProcessLogits(
    absl::Span<tflite::half> logits,
    absl::Span<const ::litert::Layout::Dim> logits_dims) {
  return ProcessLogitsImpl(logits, logits_dims);
}

absl::Status SuppressTokensProcessor::ProcessLogits(
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

absl::Status SuppressTokensProcessor::UpdateState(
    const ::litert::TensorBuffer& next_token_ids) {
  return absl::OkStatus();
}

absl::Status SuppressTokensProcessor::UpdateState(
    absl::Span<int> next_token_ids) {
  return absl::OkStatus();
}

template <typename T>
absl::Status SuppressTokensProcessor::ProcessLogitsImpl(
    absl::Span<T> logits, absl::Span<const ::litert::Layout::Dim> logits_dims) {
  if (logits_dims.size() != 3 || logits_dims[0] != batch_size_ ||
      logits_dims[1] != 1 || logits_dims[2] != vocab_size_) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Logits dimensions must be [batch_size, 1, vocab_size], which is ",
        batch_size_, ", 1, ", vocab_size_, ". The input dimensions are: [",
        absl::StrJoin(logits_dims, ", "), "]"));
  }

  if (logits.size() != vocab_size_ * batch_size_) {
    return absl::InvalidArgumentError("Logits span size incorrectly mapped.");
  }

  if (!std::is_same_v<T, float> && !std::is_same_v<T, tflite::half>) {
    return absl::InvalidArgumentError(
        "Unsupported logits tensor format element type.");
  }

  if (!config_.enabled()) {
    return absl::OkStatus();
  }

  T min;
  if constexpr (std::is_same_v<T, float>) {
    min = -std::numeric_limits<float>::infinity();
  } else {
    min = tflite::half::min();
  }

  for (int token_id : config_.suppress_tokens()) {
    for (int batch_idx = 0; batch_idx < batch_size_; ++batch_idx) {
      if (token_id >= 0 && token_id < vocab_size_) {
        logits[batch_idx * vocab_size_ + token_id] = min;
      }
    }
  }

  return absl::OkStatus();
}

}  // namespace litert::lm
