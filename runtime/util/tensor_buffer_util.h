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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_TENSOR_BUFFER_UTIL_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_TENSOR_BUFFER_UTIL_H_

#include <utility>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert

namespace litert::lm {

// Returns the number of dimensions that are greater than 1 in the given
// tensor buffer.
absl::StatusOr<int> NumSignificantDims(
    const litert::TensorBuffer& tensor_buffer);

// Returns the dimensions of the given tensor buffer.
absl::StatusOr<std::vector<int>> TensorBufferDims(
    const ::litert::TensorBuffer& tensor_buffer);

// Creates a deep copy of the given tensor buffer.
absl::StatusOr<litert::TensorBuffer> CopyTensorBuffer(
    litert::Environment& env, const litert::TensorBuffer& tensor_buffer);

// Struct to hold a TensorBuffer and a boolean indicating if the buffer is
// wrapped around existing memory or if it owns the memory.
struct MaybeWrappedTensorBuffer {
  TensorBuffer buffer;
  bool wrapped;
};

// Wraps the given host memory with a TensorBuffer if possible, otherwise
// creates a new managed TensorBuffer.
template <typename T>
absl::StatusOr<MaybeWrappedTensorBuffer> WrapOrCreateTensorBufferFromHostMemory(
    RankedTensorType tensor_type, absl::Span<T> data) {
  size_t size = data.size() * sizeof(T);
  // First try to wrap the memory with a TensorBuffer.
  auto wrapped_buffer =
      TensorBuffer::CreateFromHostMemory(tensor_type, data.data(), size);
  if (wrapped_buffer.HasValue()) {
    return MaybeWrappedTensorBuffer{.buffer = std::move(*wrapped_buffer),
                                    .wrapped = true};
  }

  LITERT_ASSIGN_OR_RETURN(const size_t packed_size, tensor_type.Bytes());
  LITERT_ASSIGN_OR_RETURN(
      auto new_buffer,
      TensorBuffer::CreateManagedHostMemory(tensor_type, packed_size));
  return MaybeWrappedTensorBuffer{.buffer = std::move(new_buffer),
                                  .wrapped = false};
}

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_TENSOR_BUFFER_UTIL_H_
