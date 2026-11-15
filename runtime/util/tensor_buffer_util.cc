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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/cc/litert_tensor_buffer_types.h"  // from @litert

namespace litert::lm {
namespace {

// Returns true when `TensorBuffer::Lock()` maps raw linear buffer memory
// directly (where `Layout::Strides()` addresses elements across a strided byte
// span within `Size()`). Returns false for GPU/custom buffer types (`OpenCl`,
// `WebGpu`, `Metal`, `Vulkan`, `Gl`, etc.), whose `Lock()` and `Unlock()`
// convert to and from a dense, unstrided `PackedSize()`-byte host staging
// buffer regardless of device storage format (e.g. FP16 or PHWC4) or backend
// buffer-requirement stride metadata.
bool UsesRawLinearLockMemory(::litert::TensorBufferType buffer_type) {
  switch (buffer_type) {
    case ::litert::TensorBufferType::kHostMemory:
    case ::litert::TensorBufferType::kAhwb:
    case ::litert::TensorBufferType::kIon:
    case ::litert::TensorBufferType::kDmaBuf:
    case ::litert::TensorBufferType::kFastRpc:
      return true;
    default:
      return false;
  }
}

::litert::RankedTensorType StripStridesIfNeeded(
    const ::litert::RankedTensorType& tensor_type) {
  if (!tensor_type.Layout().HasStrides()) {
    return tensor_type;
  }
  const auto dims = tensor_type.Layout().Dimensions();
  return ::litert::RankedTensorType(
      tensor_type.ElementType(),
      ::litert::Layout(::litert::Dimensions(dims.begin(), dims.end())));
}

absl::StatusOr<size_t> StridedElementsToBytes(
    ::litert::ElementType element_type, size_t span_elements) {
  if (span_elements == 0) {
    return size_t{0};
  }
  size_t bytes_per_elem = 0;
  switch (element_type) {
    case ::litert::ElementType::Int2:
      return (span_elements - 1) / 4 + 1;
    case ::litert::ElementType::Int4:
    case ::litert::ElementType::UInt4:
      return (span_elements - 1) / 2 + 1;
    case ::litert::ElementType::Bool:
    case ::litert::ElementType::Int8:
    case ::litert::ElementType::UInt8:
    case ::litert::ElementType::Float8E4M3FN:
    case ::litert::ElementType::Float8E5M2:
      bytes_per_elem = 1;
      break;
    case ::litert::ElementType::Int16:
    case ::litert::ElementType::UInt16:
    case ::litert::ElementType::Float16:
    case ::litert::ElementType::BFloat16:
      bytes_per_elem = 2;
      break;
    case ::litert::ElementType::Int32:
    case ::litert::ElementType::UInt32:
    case ::litert::ElementType::Float32:
      bytes_per_elem = 4;
      break;
    case ::litert::ElementType::Int64:
    case ::litert::ElementType::UInt64:
    case ::litert::ElementType::Float64:
    case ::litert::ElementType::Complex64:
      bytes_per_elem = 8;
      break;
    case ::litert::ElementType::Complex128:
      bytes_per_elem = 16;
      break;
    default:
      return absl::InvalidArgumentError(
          "Unsupported element type for strided TensorBuffer.");
  }
  if (span_elements > std::numeric_limits<size_t>::max() / bytes_per_elem) {
    return absl::InvalidArgumentError(
        "TensorBuffer strided byte size overflows size_t.");
  }
  return span_elements * bytes_per_elem;
}

// Returns the byte span exposed by `TensorBuffer::Lock()`, accounting for
// explicit element strides on linear memory buffers and validating
// `buffer_size` when `UsesRawLinearLockMemory(buffer_type)` is true.
absl::StatusOr<size_t> LockedByteSpan(
    const ::litert::RankedTensorType& tensor_type,
    ::litert::TensorBufferType buffer_type, size_t packed_size,
    size_t buffer_size) {
  if (!UsesRawLinearLockMemory(buffer_type)) {
    return packed_size;
  }
  if (!tensor_type.Layout().HasStrides()) {
    if (packed_size > buffer_size) {
      return absl::InvalidArgumentError(
          "TensorBuffer size is smaller than its packed size.");
    }
    return packed_size;
  }

  const auto dims = tensor_type.Layout().Dimensions();
  const auto strides = tensor_type.Layout().Strides();
  if (dims.size() != strides.size()) {
    return absl::InvalidArgumentError(
        "TensorBuffer dimensions and strides rank mismatch.");
  }

  bool empty = false;
  for (size_t i = 0; i < dims.size(); ++i) {
    if (dims[i] < 0) {
      return absl::InvalidArgumentError(
          "Strided TensorBuffer cannot have negative dimensions.");
    }
    empty |= (dims[i] == 0);
  }
  if (empty) {
    absl::Status status =
        StridedElementsToBytes(tensor_type.ElementType(), 1).status();
    if (!status.ok()) {
      return status;
    }
    return size_t{0};
  }

  size_t span_elements = 1;
  for (size_t i = 0; i < dims.size(); ++i) {
    const size_t dim_less_one = static_cast<size_t>(dims[i]) - 1;
    const size_t stride = static_cast<size_t>(strides[i]);
    if (dim_less_one != 0 &&
        stride > std::numeric_limits<size_t>::max() / dim_less_one) {
      return absl::InvalidArgumentError(
          "TensorBuffer stride span overflows size_t.");
    }
    const size_t axis_span = dim_less_one * stride;
    if (axis_span > std::numeric_limits<size_t>::max() - span_elements) {
      return absl::InvalidArgumentError(
          "TensorBuffer strided element count overflows size_t.");
    }
    span_elements += axis_span;
  }

  absl::StatusOr<size_t> span_bytes =
      StridedElementsToBytes(tensor_type.ElementType(), span_elements);
  if (!span_bytes.ok()) {
    return span_bytes.status();
  }
  if (*span_bytes > buffer_size) {
    return absl::InvalidArgumentError(
        "TensorBuffer size is smaller than its strided layout span.");
  }
  return *span_bytes;
}

absl::StatusOr<::litert::TensorBuffer> CopyTensorBufferImpl(
    ::litert::Environment& env, const ::litert::TensorBuffer& tensor_buffer,
    const ::litert::RankedTensorType& dst_tensor_type,
    ::litert::TensorBufferType dst_buffer_type, size_t dst_buffer_size,
    size_t copy_bytes, bool zero_tail_padding) {
  LITERT_ASSIGN_OR_RETURN(
      auto output_tensor_buffer,
      ::litert::TensorBuffer::CreateManaged(env, dst_buffer_type,
                                            dst_tensor_type, dst_buffer_size));
  const bool need_tail_zero = zero_tail_padding && dst_buffer_size > copy_bytes;
  if (copy_bytes == 0 && !need_tail_zero) {
    return output_tensor_buffer;
  }

  LITERT_ASSIGN_OR_RETURN(
      auto dst_lock_and_addr,
      ::litert::TensorBufferScopedLock::Create(
          output_tensor_buffer, ::litert::TensorBuffer::LockMode::kWrite));
  if (copy_bytes > 0) {
    LITERT_ASSIGN_OR_RETURN(
        auto src_lock_and_addr,
        ::litert::TensorBufferScopedLock::Create(
            tensor_buffer, ::litert::TensorBuffer::LockMode::kRead));
    std::memcpy(dst_lock_and_addr.second, src_lock_and_addr.second, copy_bytes);
  }
  if (need_tail_zero) {
    std::memset(static_cast<uint8_t*>(dst_lock_and_addr.second) + copy_bytes, 0,
                dst_buffer_size - copy_bytes);
  }

  return output_tensor_buffer;
}

}  // namespace

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
  LITERT_ASSIGN_OR_RETURN(auto buffer_size, tensor_buffer.Size());
  LITERT_ASSIGN_OR_RETURN(auto packed_size, tensor_buffer.PackedSize());
  absl::StatusOr<size_t> copy_bytes =
      LockedByteSpan(tensor_type, buffer_type, packed_size, buffer_size);
  if (!copy_bytes.ok()) {
    return copy_bytes.status();
  }
  return CopyTensorBufferImpl(env, tensor_buffer, tensor_type, buffer_type,
                              buffer_size, *copy_bytes,
                              /*zero_tail_padding=*/false);
}

absl::StatusOr<::litert::TensorBuffer> CopyTensorBufferToHostMemory(
    ::litert::Environment& env, const ::litert::TensorBuffer& tensor_buffer) {
  LITERT_ASSIGN_OR_RETURN(auto tensor_type, tensor_buffer.TensorType());
  LITERT_ASSIGN_OR_RETURN(auto src_buffer_type, tensor_buffer.BufferType());
  LITERT_ASSIGN_OR_RETURN(auto buffer_size, tensor_buffer.Size());
  LITERT_ASSIGN_OR_RETURN(auto packed_size, tensor_buffer.PackedSize());
  absl::StatusOr<size_t> copy_bytes =
      LockedByteSpan(tensor_type, src_buffer_type, packed_size, buffer_size);
  if (!copy_bytes.ok()) {
    return copy_bytes.status();
  }
  const ::litert::RankedTensorType dst_tensor_type =
      UsesRawLinearLockMemory(src_buffer_type)
          ? tensor_type
          : StripStridesIfNeeded(tensor_type);
  // `TensorBuffer::CreateManaged` requires `buffer_size >= packed_size` even if
  // a strided layout uses zero/broadcast strides with a smaller active span.
  const size_t host_buffer_size = std::max(packed_size, *copy_bytes);
  return CopyTensorBufferImpl(env, tensor_buffer, dst_tensor_type,
                              ::litert::TensorBufferType::kHostMemory,
                              host_buffer_size, *copy_bytes,
                              /*zero_tail_padding=*/true);
}

absl::Status CopyTensorBufferContents(::litert::TensorBuffer& dst,
                                      const ::litert::TensorBuffer& src) {
  if (dst.Get() == src.Get()) {
    return absl::OkStatus();
  }
  LITERT_ASSIGN_OR_RETURN(const auto dst_type, dst.TensorType());
  LITERT_ASSIGN_OR_RETURN(const auto src_type, src.TensorType());
  if (dst_type.ElementType() != src_type.ElementType() ||
      dst_type.Layout().Dimensions() != src_type.Layout().Dimensions()) {
    return absl::InvalidArgumentError(
        "Cannot copy TensorBuffer contents between mismatched tensor types.");
  }
  LITERT_ASSIGN_OR_RETURN(const auto dst_buffer_type, dst.BufferType());
  LITERT_ASSIGN_OR_RETURN(const auto src_buffer_type, src.BufferType());
  const bool dst_linear = UsesRawLinearLockMemory(dst_buffer_type);
  const bool src_linear = UsesRawLinearLockMemory(src_buffer_type);
  if (dst_linear && src_linear) {
    if (dst_type != src_type) {
      return absl::InvalidArgumentError(
          "Cannot copy linear TensorBuffer contents between mismatched "
          "strides.");
    }
  } else if ((dst_linear && dst_type.Layout().HasStrides()) ||
             (src_linear && src_type.Layout().HasStrides())) {
    return absl::InvalidArgumentError(
        "Cannot copy between strided linear TensorBuffer and non-linear "
        "device TensorBuffer.");
  }
  LITERT_ASSIGN_OR_RETURN(const size_t dst_buffer_size, dst.Size());
  LITERT_ASSIGN_OR_RETURN(const size_t dst_packed_size, dst.PackedSize());
  LITERT_ASSIGN_OR_RETURN(const size_t src_buffer_size, src.Size());
  LITERT_ASSIGN_OR_RETURN(const size_t src_packed_size, src.PackedSize());
  absl::StatusOr<size_t> copy_bytes = LockedByteSpan(
      dst_type, dst_buffer_type, dst_packed_size, dst_buffer_size);
  if (!copy_bytes.ok()) {
    return copy_bytes.status();
  }
  absl::StatusOr<size_t> src_copy_bytes = LockedByteSpan(
      src_type, src_buffer_type, src_packed_size, src_buffer_size);
  if (!src_copy_bytes.ok()) {
    return src_copy_bytes.status();
  }
  if (*copy_bytes != *src_copy_bytes) {
    return absl::InvalidArgumentError(
        "TensorBuffer locked byte spans do not match.");
  }
  if (*copy_bytes == 0) {
    return absl::OkStatus();
  }
  LITERT_ASSIGN_OR_RETURN(auto src_lock_and_addr,
                          ::litert::TensorBufferScopedLock::Create(
                              src, ::litert::TensorBuffer::LockMode::kRead));
  LITERT_ASSIGN_OR_RETURN(auto dst_lock_and_addr,
                          ::litert::TensorBufferScopedLock::Create(
                              dst, ::litert::TensorBuffer::LockMode::kWrite));
  std::memcpy(dst_lock_and_addr.second, src_lock_and_addr.second, *copy_bytes);
  return absl::OkStatus();
}

}  // namespace litert::lm
