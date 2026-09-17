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

#include "runtime/components/constrained_decoding/constrained_decoder.h"

#include <algorithm>
#include <iterator>
#include <memory>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_common.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/constrained_decoding/constraint.h"
#include "runtime/components/constrained_decoding/litert_logit_mask_runner.h"
#include "runtime/components/constrained_decoding/logit_mask.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/status_macros.h"  //NOLINT

namespace litert::lm {

absl::StatusOr<std::unique_ptr<ConstrainedDecoder>> ConstrainedDecoder::Create(
    Constraint* constraint, int batch_size, ::litert::Environment& env,
    HwAccelerators accelerator, bool force_graph_on_host) {
  RET_CHECK_NE(constraint, nullptr) << "Constraint must not be null.";
  RET_CHECK_GT(batch_size, 0) << "Batch size must be positive.";
  ABSL_ASSIGN_OR_RETURN(
      auto litert_logit_mask_runner,
      LiteRtLogitMaskRunner::Create(env, accelerator, force_graph_on_host));
  return absl::WrapUnique(new ConstrainedDecoder(
      constraint, batch_size, std::move(litert_logit_mask_runner)));
}

absl::StatusOr<std::unique_ptr<ConstrainedDecoder>>
ConstrainedDecoder::CreateForHost(Constraint* constraint, int batch_size) {
  RET_CHECK_NE(constraint, nullptr) << "Constraint must not be null.";
  RET_CHECK_GT(batch_size, 0) << "Batch size must be positive.";
  ABSL_ASSIGN_OR_RETURN(auto litert_logit_mask_runner,
                        LiteRtLogitMaskRunner::CreateForHost());
  return absl::WrapUnique(new ConstrainedDecoder(
      constraint, batch_size, std::move(litert_logit_mask_runner)));
}

ConstrainedDecoder::ConstrainedDecoder(
    Constraint* constraint, int batch_size,
    std::unique_ptr<LiteRtLogitMaskRunner> litert_logit_mask_runner)
    : constraint_(constraint),
      batch_size_(batch_size),
      litert_logit_mask_runner_(std::move(litert_logit_mask_runner)) {
  constraint_states_.reserve(batch_size_);
  std::generate_n(std::back_inserter(constraint_states_), batch_size_,
                  [&]() { return constraint_->Start(); });
}

absl::Status ConstrainedDecoder::ProcessLogits(::litert::TensorBuffer& logits) {
  LITERT_ASSIGN_OR_RETURN(auto logits_tensor_type, logits.TensorType());
  const auto& dims = logits_tensor_type.Layout().Dimensions();
  RET_CHECK_EQ(dims.size(), 3)
      << "Only support logits with dimensions [batch_size, 1, vocab_size].";
  int batch_size = dims[0];
  int sequence_length = dims[1];
  RET_CHECK_EQ(sequence_length, 1) << "Only support sequence length 1.";
  RET_CHECK_EQ(batch_size, batch_size_)
      << "Batch size [" << batch_size
      << "] does not match the expected batch size [" << batch_size_ << "].";

  std::vector<std::unique_ptr<LogitMask>> masks;
  masks.reserve(batch_size_);
  std::vector<const LogitMask*> raw_masks;
  raw_masks.reserve(batch_size_);
  for (int b = 0; b < batch_size_; ++b) {
    ABSL_ASSIGN_OR_RETURN(auto mask,
                          constraint_->ComputeMask(*constraint_states_[b]));
    masks.push_back(std::move(mask));
    raw_masks.push_back(masks.back().get());
  }
  return litert_logit_mask_runner_->ApplyBatch(logits,
                                               absl::MakeConstSpan(raw_masks));
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
    ABSL_ASSIGN_OR_RETURN(
        constraint_state,
        constraint_->ComputeNext(*constraint_state, next_token_ids[i]));
    if (constraint_->IsEnded(*constraint_state)) {
      constraint_state = constraint_->Start();
    }
  }
  return absl::OkStatus();
}

}  // namespace litert::lm
