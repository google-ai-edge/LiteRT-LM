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
#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert

namespace litert::omni {

LiteRtRunnerImpl::LiteRtRunnerImpl(CompiledModel* absl_nonnull compiled_model)
    : compiled_model_(compiled_model) {}

LiteRtRunnerImpl::LiteRtRunnerImpl(
    std::unique_ptr<CompiledModel> absl_nonnull compiled_model)
    : owned_compiled_model_(std::move(compiled_model)),
      compiled_model_(owned_compiled_model_.get()) {}

absl::StatusOr<std::vector<TensorBuffer>> LiteRtRunnerImpl::CreateInputBuffers(
    absl::string_view signature_name) {
  LITERT_ASSIGN_OR_RETURN(auto inputs,
                          compiled_model_->CreateInputBuffers(signature_name));
  return inputs;
}

absl::StatusOr<std::vector<TensorBuffer>> LiteRtRunnerImpl::CreateOutputBuffers(
    absl::string_view signature_name) {
  LITERT_ASSIGN_OR_RETURN(auto outputs,
                          compiled_model_->CreateOutputBuffers(signature_name));
  return outputs;
}

absl::Status LiteRtRunnerImpl::Run(
    absl::string_view signature_name,
    absl::Span<const TensorBuffer> input_buffers,
    absl::Span<const TensorBuffer> output_buffers) {
  LITERT_ASSIGN_OR_RETURN(size_t signature_index,
                          compiled_model_->GetSignatureIndex(signature_name));
  LITERT_RETURN_IF_ERROR(
      compiled_model_->Run(signature_index, input_buffers, output_buffers));
  return absl::OkStatus();
}

PassthroughRunner::PassthroughRunner(std::vector<size_t> buffer_sizes_bytes)
    : buffer_sizes_bytes_(std::move(buffer_sizes_bytes)) {}

absl::StatusOr<std::vector<TensorBuffer>> PassthroughRunner::CreateInputBuffers(
    absl::string_view signature_name) {
  std::vector<TensorBuffer> buffers;
  buffers.reserve(buffer_sizes_bytes_.size());
  for (size_t size : buffer_sizes_bytes_) {
    RankedTensorType tensor_type(
        ElementType::UInt8, Layout(Dimensions({static_cast<int32_t>(size)})));
    LITERT_ASSIGN_OR_RETURN(
        auto buffer, TensorBuffer::CreateManagedHostMemory(tensor_type, size));
    buffers.push_back(std::move(buffer));
  }
  return buffers;
}

absl::StatusOr<std::vector<TensorBuffer>>
PassthroughRunner::CreateOutputBuffers(absl::string_view signature_name) {
  return CreateInputBuffers(signature_name);
}

absl::Status PassthroughRunner::Run(
    absl::string_view signature_name,
    absl::Span<const TensorBuffer> input_buffers,
    absl::Span<const TensorBuffer> output_buffers) {
  if (input_buffers.size() != output_buffers.size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Number of input buffers (", input_buffers.size(),
        ") and output buffers (", output_buffers.size(), ") must match."));
  }
  for (size_t i = 0; i < input_buffers.size(); ++i) {
    LITERT_ASSIGN_OR_RETURN(size_t in_size, input_buffers[i].PackedSize());
    LITERT_ASSIGN_OR_RETURN(size_t out_size, output_buffers[i].PackedSize());
    if (in_size != out_size) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Input buffer ", i, " size (", in_size,
          ") does not match output buffer size (", out_size, ")."));
    }
    LITERT_ASSIGN_OR_RETURN(
        auto in_lock_and_addr,
        TensorBufferScopedLock::Create<const char>(
            input_buffers[i], TensorBuffer::LockMode::kRead));
    LITERT_ASSIGN_OR_RETURN(auto out_lock_and_addr,
                            TensorBufferScopedLock::Create<char>(
                                const_cast<TensorBuffer&>(output_buffers[i]),
                                TensorBuffer::LockMode::kWrite));
    std::memcpy(out_lock_and_addr.second, in_lock_and_addr.second, in_size);
  }
  return absl::OkStatus();
}

}  // namespace litert::omni
