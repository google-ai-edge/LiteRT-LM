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

#include "omni/base/stateful_litert_runner.h"

#include <cstddef>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "omni/base/litert_runner.h"

namespace litert::omni {

absl::StatusOr<std::unique_ptr<StatefulLiteRtRunnerImpl>>
StatefulLiteRtRunnerImpl::Create(LiteRtRunner* absl_nonnull runner,
                                 absl::string_view signature_name,
                                 size_t num_non_state_inputs,
                                 size_t num_non_state_outputs) {
  auto impl = absl::WrapUnique(new StatefulLiteRtRunnerImpl(
      runner, std::string(signature_name), num_non_state_inputs,
      num_non_state_outputs));

  LITERT_ASSIGN_OR_RETURN(impl->input_buffers_0_,
                          runner->CreateInputBuffers(impl->signature_name_));
  LITERT_ASSIGN_OR_RETURN(impl->output_buffers_0_,
                          runner->CreateOutputBuffers(impl->signature_name_));

  if (impl->input_buffers_0_.size() < num_non_state_inputs) {
    return absl::InvalidArgumentError(
        "Model has fewer input buffers than num_non_state_inputs.");
  }
  if (impl->output_buffers_0_.size() < num_non_state_outputs) {
    return absl::InvalidArgumentError(
        "Model has fewer output buffers than num_non_state_outputs.");
  }

  size_t num_states_in = impl->input_buffers_0_.size() - num_non_state_inputs;
  size_t num_states_out =
      impl->output_buffers_0_.size() - num_non_state_outputs;
  if (num_states_in != num_states_out) {
    return absl::InvalidArgumentError(
        "Mismatch between number of state inputs and state outputs.");
  }

  // Wire up ping-pong double buffering for state tensors without duplicate
  // allocations:
  // - input_buffers_1_ uses output_buffers_0_ state outputs for its state
  // inputs.
  // - output_buffers_1_ uses input_buffers_0_ state inputs for its state
  // outputs.
  impl->input_buffers_1_.reserve(impl->input_buffers_0_.size());
  for (size_t i = 0; i < num_non_state_inputs; ++i) {
    LITERT_ASSIGN_OR_RETURN(auto dup, impl->input_buffers_0_[i].Duplicate());
    impl->input_buffers_1_.push_back(std::move(dup));
  }
  for (size_t i = 0; i < num_states_in; ++i) {
    LITERT_ASSIGN_OR_RETURN(
        auto dup,
        impl->output_buffers_0_[num_non_state_outputs + i].Duplicate());
    impl->input_buffers_1_.push_back(std::move(dup));
  }

  impl->output_buffers_1_.reserve(impl->output_buffers_0_.size());
  for (size_t i = 0; i < num_non_state_outputs; ++i) {
    LITERT_ASSIGN_OR_RETURN(auto dup, impl->output_buffers_0_[i].Duplicate());
    impl->output_buffers_1_.push_back(std::move(dup));
  }
  for (size_t i = 0; i < num_states_out; ++i) {
    LITERT_ASSIGN_OR_RETURN(
        auto dup, impl->input_buffers_0_[num_non_state_inputs + i].Duplicate());
    impl->output_buffers_1_.push_back(std::move(dup));
  }

  LITERT_RETURN_IF_ERROR(impl->Reset());
  return impl;
}

absl::Status StatefulLiteRtRunnerImpl::Reset() {
  active_set_ = 0;
  for (size_t i = num_non_state_inputs_; i < input_buffers_0_.size(); ++i) {
    LITERT_ASSIGN_OR_RETURN(
        auto lock0, TensorBufferScopedLock::Create(
                        input_buffers_0_[i], TensorBuffer::LockMode::kWrite));
    LITERT_ASSIGN_OR_RETURN(auto size0, input_buffers_0_[i].PackedSize());
    std::memset(lock0.second, 0, size0);

    LITERT_ASSIGN_OR_RETURN(
        auto lock1, TensorBufferScopedLock::Create(
                        input_buffers_1_[i], TensorBuffer::LockMode::kWrite));
    LITERT_ASSIGN_OR_RETURN(auto size1, input_buffers_1_[i].PackedSize());
    std::memset(lock1.second, 0, size1);
  }
  return absl::OkStatus();
}

absl::StatusOr<absl::Span<const TensorBuffer>> StatefulLiteRtRunnerImpl::Step(
    absl::Span<const TensorBuffer> non_state_inputs, bool auto_commit_state) {
  auto& cur_inputs = (active_set_ == 0 ? input_buffers_0_ : input_buffers_1_);
  auto& cur_outputs =
      (active_set_ == 0 ? output_buffers_0_ : output_buffers_1_);

  if (!non_state_inputs.empty()) {
    if (non_state_inputs.size() != num_non_state_inputs_) {
      return absl::InvalidArgumentError(
          "non_state_inputs size does not match num_non_state_inputs.");
    }
    for (size_t i = 0; i < num_non_state_inputs_; ++i) {
      if (cur_inputs[i].Get() != non_state_inputs[i].Get()) {
        LITERT_ASSIGN_OR_RETURN(cur_inputs[i], non_state_inputs[i].Duplicate());
      }
    }
  }

  LITERT_RETURN_IF_ERROR(runner_.Run(signature_name_, cur_inputs, cur_outputs));

  if (auto_commit_state) {
    active_set_ = 1 - active_set_;
  }

  return absl::MakeConstSpan(cur_outputs).subspan(0, num_non_state_outputs_);
}

absl::Status StatefulLiteRtRunnerImpl::CommitState() {
  active_set_ = 1 - active_set_;
  return absl::OkStatus();
}

absl::Span<const TensorBuffer> StatefulLiteRtRunnerImpl::GetActiveInputStates()
    const {
  const auto& cur_inputs =
      (active_set_ == 0 ? input_buffers_0_ : input_buffers_1_);
  return absl::MakeConstSpan(cur_inputs).subspan(num_non_state_inputs_);
}

absl::Span<TensorBuffer> StatefulLiteRtRunnerImpl::GetNonStateInputBuffers() {
  auto& cur_inputs = (active_set_ == 0 ? input_buffers_0_ : input_buffers_1_);
  return absl::MakeSpan(cur_inputs).subspan(0, num_non_state_inputs_);
}

absl::Span<const TensorBuffer>
StatefulLiteRtRunnerImpl::GetNonStateOutputBuffers() const {
  const auto& cur_outputs =
      (active_set_ == 0 ? output_buffers_1_ : output_buffers_0_);
  return absl::MakeConstSpan(cur_outputs).subspan(0, num_non_state_outputs_);
}

}  // namespace litert::omni
