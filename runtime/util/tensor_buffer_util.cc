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

#include "runtime/util/tensor_buffer_util.h"

#include <cstring>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert

namespace litert::lm {

absl::StatusOr<int> NumSignificantDims(
    const ::litert::TensorBuffer& tensor_buffer) {
  LITERT_ASSIGN_OR_RETURN(const auto& dims, TensorBufferDims(tensor_buffer));
  int num_significant_dims = 0;
  for (int d : dims) {
    num_significant_dims += (d > 1);
  }
  return num_significant_dims;
}

absl::StatusOr<std::vector<int>> TensorBufferDims(
    const ::litert::TensorBuffer& tensor_buffer) {
  LITERT_ASSIGN_OR_RETURN(auto tensor_type, tensor_buffer.TensorType());
  auto dims = tensor_type.Layout().Dimensions();
  return std::vector<int>(dims.begin(), dims.end());
}

absl::StatusOr<::litert::TensorBuffer> CopyTensorBuffer(
    ::litert::Environment& env, const ::litert::TensorBuffer& tensor_buffer) {
  LITERT_ASSIGN_OR_RETURN(auto tensor_type, tensor_buffer.TensorType());
  LITERT_ASSIGN_OR_RETURN(auto buffer_type, tensor_buffer.BufferType());
  LITERT_ASSIGN_OR_RETURN(auto size, tensor_buffer.PackedSize());

  LITERT_ASSIGN_OR_RETURN(auto output_tensor_buffer,
                          ::litert::TensorBuffer::CreateManaged(
                              env, buffer_type, tensor_type, size));

  LITERT_ASSIGN_OR_RETURN(
      auto src_lock_and_addr,
      ::litert::TensorBufferScopedLock::Create(
          tensor_buffer, ::litert::TensorBuffer::LockMode::kRead));
  LITERT_ASSIGN_OR_RETURN(
      auto dst_lock_and_addr,
      ::litert::TensorBufferScopedLock::Create(
          output_tensor_buffer, ::litert::TensorBuffer::LockMode::kWrite));

  std::memcpy(dst_lock_and_addr.second, src_lock_and_addr.second, size);

  return std::move(output_tensor_buffer);
}

}  // namespace litert::lm
