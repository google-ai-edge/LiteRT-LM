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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_LOGITS_PROCESSOR_CONSTRAINED_DECODING_CONSTRAINED_DECODER_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_LOGITS_PROCESSOR_CONSTRAINED_DECODING_CONSTRAINED_DECODER_H_

#include <algorithm>
#include <iterator>
#include <memory>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/logits_processor/constrained_decoding/constraint.h"
#include "runtime/components/logits_processor/logits_processor.h"
#include "tflite/types/half.h"  // from @litert

namespace litert::lm {

// Manages the state of constrained decoding for a batch of sequences.
//
// This class uses a `Constraint` object to validate tokens during
// autoregressive decoding. It maintains the constraint state for each sequence
// in a batch and provides a method to mask logits, setting the logits of
// disallowed tokens to -inf based on the current state of each sequence.
//
// Example usage:
//   Constraint* constraint = ...;
//   ConstrainedDecoder decoder(constraint, batch_size);
//   while (!done) {
//     TensorBuffer logits = Decode(...);
//     ABSL_RETURN_IF_ERROR(decoder.ProcessLogits(logits));
//     TensorBuffer next_tokens = sampler.Sample(logits);
//     ABSL_RETURN_IF_ERROR(decoder.UpdateState(next_tokens));
//   }
class ConstrainedDecoder : public LogitsProcessor {
 public:
  // Creates a ConstrainedDecoder.
  //
  // @param constraint The constraint to apply during decoding. The caller
  // retains ownership and must ensure it outlives the decoder.
  // @param batch_size The number of sequences in the batch.
  explicit ConstrainedDecoder(Constraint* constraint, int batch_size)
      : constraint_(constraint), batch_size_(batch_size) {
    constraint_states_.reserve(batch_size_);
    std::generate_n(std::back_inserter(constraint_states_), batch_size_,
                    [&]() { return constraint_->Start(); });
  };
  virtual ~ConstrainedDecoder() = default;

  // Masks the input logits tensor based on the current constraint state of
  // each sequence in the batch.
  // For each sequence, tokens disallowed by the constraint in the current state
  // will have their corresponding logit values set to -inf.
  //
  // @param logits A tensor of shape [batch_size, sequence_length, vocab_size]
  // containing the logits for the next token prediction. This tensor is
  // modified in-place.
  // @return Ok if masking was successful, or an error if dimensionss are
  // incorrect or masking fails.
  absl::Status ProcessLogits(::litert::TensorBuffer& logits) override;

  // Same as above, but takes a span of logits instead of a tensor buffer.
  absl::Status ProcessLogits(
      absl::Span<float> logits,
      absl::Span<const ::litert::Layout::Dim> logits_dims) override;

  absl::Status ProcessLogits(
      absl::Span<tflite::half> logits,
      absl::Span<const ::litert::Layout::Dim> logits_dims) override;

  // Updates the internal constraint state for each sequence in the batch based
  // on the newly selected tokens. If a sequence reaches an end state
  // according to the constraint, its state is reset to the start state.
  //
  // @param next_token_ids A tensor of shape [batch_size, sequence_length]
  // containing the token ID selected for each sequence in the batch for the
  // current step.
  // @return Ok if the states were updated successfully, or an error if any
  // token is invalid for its corresponding state.
  absl::Status UpdateState(
      const ::litert::TensorBuffer& next_token_ids) override;

  // Same as above, but takes a span of token ids instead of a tensor buffer.
  absl::Status UpdateState(absl::Span<int> next_token_ids) override;

  // Returns a pointer to the constraint.
  Constraint* GetConstraint() const { return constraint_; }

  ConstrainedDecoder* GetConstraintDecoder() override { return this; }
  const ConstrainedDecoder* GetConstraintDecoder() const override {
    return this;
  }

 private:
  // The constraint to be applied.
  Constraint* constraint_;
  const int batch_size_;
  // The current constraint states.
  std::vector<std::unique_ptr<Constraint::State>> constraint_states_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_LOGITS_PROCESSOR_CONSTRAINED_DECODING_CONSTRAINED_DECODER_H_
