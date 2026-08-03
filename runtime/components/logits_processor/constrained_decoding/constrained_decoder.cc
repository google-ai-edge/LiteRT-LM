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

#include "runtime/components/logits_processor/constrained_decoding/constrained_decoder.h"

#include <limits>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/status_macros.h"  //NOLINT
#include "tflite/types/half.h"  // from @litert

namespace litert::lm {

absl::Status ConstrainedDecoder::ProcessLogits(::litert::TensorBuffer& logits) {
  // Compute the allowed tokens bitmap for the current constraint state.
  LITERT_ASSIGN_OR_RETURN(auto logits_tensor_type, logits.TensorType());
  if (logits_tensor_type.ElementType() == ::litert::ElementType::Float32) {
    LITERT_ASSIGN_OR_RETURN(auto logits_span,
                            ReferTensorBufferAsSpan<float>(logits));
    return ProcessLogits(logits_span, logits_tensor_type.Layout().Dimensions());
  } else if (logits_tensor_type.ElementType() ==
             ::litert::ElementType::Float16) {
    LITERT_ASSIGN_OR_RETURN(auto logits_span,
                            ReferTensorBufferAsSpan<tflite::half>(logits));
    return ProcessLogits(logits_span, logits_tensor_type.Layout().Dimensions());
  }
  return absl::InvalidArgumentError(
      "Unsupported logits type for ConstrainedDecoder::ProcessLogits.");
}

absl::Status ConstrainedDecoder::ProcessLogits(
    absl::Span<float> logits,
    absl::Span<const ::litert::Layout::Dim> logits_dims) {
  RET_CHECK_EQ(logits_dims.size(), 3)
      << "Only support logits with dimensions [batch_size, 1, vocab_size].";
  int batch_size = logits_dims[0];
  int sequence_length = logits_dims[1];
  int vocab_size = logits_dims[2];
  RET_CHECK_EQ(sequence_length, 1) << "Only support sequence length 1.";
  // It is possible that the constraint vocabulary size is larger than the model
  // vocabulary size. The remaining tokens in the constraint vocabulary are
  // treated as unused tokens.
  RET_CHECK_LE(vocab_size, constraint_->GetVocabularySize())
      << "Vocabulary size [" << vocab_size
      << "] does not match the expected vocabulary size ["
      << constraint_->GetVocabularySize() << "].";
  RET_CHECK_EQ(batch_size, batch_size_)
      << "Batch size [" << batch_size
      << "] does not match the expected batch size [" << batch_size_ << "].";
  for (int b = 0; b < batch_size; ++b) {
    auto& constraint_state = constraint_states_[b];
    ASSIGN_OR_RETURN(auto bitmap,
                     constraint_->ComputeBitmap(*constraint_state));
    for (int i = 0; i < vocab_size; ++i) {
      if (!bitmap->Get(i)) {
        logits.data()[b * vocab_size + i] =
            std::numeric_limits<float>::lowest();
      }
    }
  }
  return absl::OkStatus();
}

absl::Status ConstrainedDecoder::ProcessLogits(
    absl::Span<tflite::half> logits,
    absl::Span<const ::litert::Layout::Dim> logits_dims) {
  RET_CHECK_EQ(logits_dims.size(), 3)
      << "Only support logits with dimensions [batch_size, 1, vocab_size].";
  int batch_size = logits_dims[0];
  int sequence_length = logits_dims[1];
  int vocab_size = logits_dims[2];
  RET_CHECK_EQ(sequence_length, 1) << "Only support sequence length 1.";
  // It is possible that the constraint vocabulary size is larger than the model
  // vocabulary size. The remaining tokens in the constraint vocabulary are
  // treated as unused tokens.
  RET_CHECK_LE(vocab_size, constraint_->GetVocabularySize())
      << "Vocabulary size [" << vocab_size
      << "] does not match the expected vocabulary size ["
      << constraint_->GetVocabularySize() << "].";
  RET_CHECK_EQ(batch_size, batch_size_)
      << "Batch size [" << batch_size
      << "] does not match the expected batch size [" << batch_size_ << "].";
  for (int b = 0; b < batch_size; ++b) {
    auto& constraint_state = constraint_states_[b];
    ASSIGN_OR_RETURN(auto bitmap,
                     constraint_->ComputeBitmap(*constraint_state));
    for (int i = 0; i < vocab_size; ++i) {
      if (!bitmap->Get(i)) {
        logits.data()[b * vocab_size + i] = tflite::half::min();
      }
    }
  }
  return absl::OkStatus();
}

absl::Status ConstrainedDecoder::UpdateState(
    const ::litert::TensorBuffer& next_token_ids) {
  LITERT_ASSIGN_OR_RETURN(auto next_token_ids_span,
                          ReferTensorBufferAsSpan<int>(next_token_ids));
  return UpdateState(next_token_ids_span);
}

absl::Status ConstrainedDecoder::UpdateState(absl::Span<int> next_token_ids) {
  RET_CHECK_EQ(next_token_ids.size(), batch_size_)
      << "Batch size [" << next_token_ids.size()
      << "] does not match the expected batch size [" << batch_size_ << "].";
  for (int i = 0; i < batch_size_; ++i) {
    auto& constraint_state = constraint_states_[i];
    ASSIGN_OR_RETURN(
        constraint_state,
        constraint_->ComputeNext(*constraint_state, next_token_ids[i]));
    if (constraint_->IsEnded(*constraint_state)) {
      constraint_state = constraint_->Start();
    }
  }
  return absl::OkStatus();
}

}  // namespace litert::lm
