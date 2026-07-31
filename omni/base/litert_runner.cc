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

#include "omni/base/litert_runner.h"

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert

namespace litert::omni {

LiteRtRunnerImpl::LiteRtRunnerImpl(
    ::litert::CompiledModel* absl_nonnull compiled_model)
    : compiled_model_(compiled_model) {}

LiteRtRunnerImpl::LiteRtRunnerImpl(
    std::unique_ptr<::litert::CompiledModel> absl_nonnull compiled_model)
    : owned_compiled_model_(std::move(compiled_model)),
      compiled_model_(owned_compiled_model_.get()) {}

absl::StatusOr<std::vector<::litert::TensorBuffer>>
LiteRtRunnerImpl::CreateInputBuffers(absl::string_view signature_name) {
  LITERT_ASSIGN_OR_RETURN(auto inputs,
                          compiled_model_->CreateInputBuffers(signature_name));
  return inputs;
}

absl::StatusOr<std::vector<::litert::TensorBuffer>>
LiteRtRunnerImpl::CreateOutputBuffers(absl::string_view signature_name) {
  LITERT_ASSIGN_OR_RETURN(auto outputs,
                          compiled_model_->CreateOutputBuffers(signature_name));
  return outputs;
}

absl::Status LiteRtRunnerImpl::Run(
    absl::string_view signature_name,
    absl::Span<const ::litert::TensorBuffer> input_buffers,
    absl::Span<const ::litert::TensorBuffer> output_buffers) {
  LITERT_ASSIGN_OR_RETURN(size_t signature_index,
                          compiled_model_->GetSignatureIndex(signature_name));
  LITERT_RETURN_IF_ERROR(
      compiled_model_->Run(signature_index, input_buffers, output_buffers));
  return absl::OkStatus();
}

}  // namespace litert::omni
