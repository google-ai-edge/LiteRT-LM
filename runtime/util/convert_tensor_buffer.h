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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_CONVERT_TENSOR_BUFFER_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_CONVERT_TENSOR_BUFFER_H_

#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_common.h"  // from @litert
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_expected.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/cc/litert_tensor_buffer_types.h"  // from @litert
#include "tflite/types/half.h"  // from @litert

namespace litert::lm {

template <typename T>
struct ElementTypeFor {
  // Don't define kType to generate a compile error for unsupported types.
};

// Here is the list of supported element types effectively. Support only minimal
// types for now to avoid compatibility issues, e.g. whether or not uint8 is
// compatible with int8.
template <>
struct ElementTypeFor<bool> {
  static constexpr ::litert::ElementType kType = ::litert::ElementType::Bool;
};

template <>
struct ElementTypeFor<int8_t> {
  static constexpr ::litert::ElementType kType = ::litert::ElementType::Int8;
};

template <>
struct ElementTypeFor<int16_t> {
  static constexpr ::litert::ElementType kType = ::litert::ElementType::Int16;
};

template <>
struct ElementTypeFor<int32_t> {
  static constexpr ::litert::ElementType kType = ::litert::ElementType::Int32;
};

template <>
struct ElementTypeFor<float> {
  static constexpr ::litert::ElementType kType = ::litert::ElementType::Float32;
};

template <>
struct ElementTypeFor<tflite::half> {
  static constexpr ::litert::ElementType kType = ::litert::ElementType::Float16;
};

template <typename T>
::litert::Expected<::litert::TensorBuffer> CreateTensorBuffer(
    ::litert::Dimensions&& dimensions,
    ::litert::TensorBufferType buffer_type =
        ::litert::TensorBufferType::kHostMemory) {
  if (buffer_type != ::litert::TensorBufferType::kHostMemory) {
    return ::litert::Unexpected(
        ::litert::Status::kErrorInvalidArgument,
        "Only host memory buffer is supported. Use CreateTensorBuffer() with "
        "Environment argument.");
  }
  int size = 1;
  for (int dim : dimensions) {
    size *= dim;
  }

  return ::litert::TensorBuffer::CreateManagedHostMemory(
      ::litert::RankedTensorType(ElementTypeFor<T>::kType,
                                 ::litert::Layout(std::move(dimensions))),
      size * sizeof(T));
}

// Creates a ::litert::TensorBuffer with the given dimensions and data.
template <typename T>
::litert::Expected<::litert::TensorBuffer> CreateTensorBuffer(
    ::litert::Dimensions&& dimensions, ::litert::TensorBufferType buffer_type,
    ::litert::Environment& env) {
  int size = 1;
  for (int dim : dimensions) {
    size *= dim;
  }

  return ::litert::TensorBuffer::CreateManaged(
      env, buffer_type,
      ::litert::RankedTensorType(ElementTypeFor<T>::kType,
                                 ::litert::Layout(std::move(dimensions))),
      size * sizeof(T));
}

// Copies a ::litert::TensorBuffer of arbitrary shape to a std::vector<T>.
template <typename T>
::litert::Expected<std::vector<T>> CopyFromTensorBuffer(
    const ::litert::TensorBuffer& tensor_buffer) {
  if (auto type = tensor_buffer.TensorType();
      !type.HasValue() || type->ElementType() != ElementTypeFor<T>::kType) {
    return ::litert::Unexpected(
        ::litert::Status::kErrorInvalidArgument,
        "Element type is not compatible to the target type.");
  }

  LITERT_ASSIGN_OR_RETURN(auto tensor_type, tensor_buffer.TensorType());
  LITERT_ASSIGN_OR_RETURN(auto num_elements,
                          tensor_type.Layout().NumElements());
  std::vector<T> copied_data(num_elements);
  LITERT_ASSIGN_OR_RETURN(
      auto lock_and_addr,
      ::litert::TensorBufferScopedLock::Create(
          *const_cast<::litert::TensorBuffer*>(&tensor_buffer),
          TensorBuffer::LockMode::kRead));
  // Note: std::vector of bool is specialized to require fewer bits per element
  // and is not compatible with a direct memcpy.
  if constexpr (std::is_same_v<T, bool>) {
    auto* src = static_cast<const bool*>(lock_and_addr.second);
    std::copy(src, src + num_elements, copied_data.begin());
  } else {
    std::memcpy(copied_data.data(), lock_and_addr.second,
                num_elements * sizeof(T));
  }
  return copied_data;
}

// Copies a 2D ::litert::TensorBuffer to a std::vector<std::vector<T>>.
template <typename T>
::litert::Expected<std::vector<std::vector<T>>> CopyFromTensorBuffer2D(
    const ::litert::TensorBuffer& tensor_buffer) {
  auto type = tensor_buffer.TensorType();
  if (!type.HasValue() || type->ElementType() != ElementTypeFor<T>::kType) {
    return ::litert::Unexpected(
        ::litert::Status::kErrorInvalidArgument,
        "Element type is not compatible to the target type.");
  }

  auto dimensions = type->Layout().Dimensions();
  if (dimensions.size() != 2) {
    return ::litert::Unexpected(::litert::Status::kErrorInvalidArgument,
                                "Tensor buffer must have 2 dimensions.");
  }

  auto lock_and_addr = ::litert::TensorBufferScopedLock::Create(
      *const_cast<::litert::TensorBuffer*>(&tensor_buffer),
      TensorBuffer::LockMode::kRead);
  ABSL_DCHECK(lock_and_addr.HasValue());
  auto data_from = absl::MakeConstSpan(static_cast<T*>(lock_and_addr->second),
                                       dimensions[0] * dimensions[1]);
  std::vector<std::vector<T>> data_to(dimensions[0]);
  for (int i = 0; i < dimensions[0]; ++i) {
    data_to[i].resize(dimensions[1]);
    std::copy(data_from.begin() + i * dimensions[1],
              data_from.begin() + (i + 1) * dimensions[1], data_to[i].begin());
  }
  return std::move(data_to);
}

// Copies an absl::Span<const T> to a ::litert::TensorBuffer with the given
// dimensions.
template <typename T>
::litert::Expected<::litert::TensorBuffer> CopyToTensorBuffer(
    absl::Span<const T> data, ::litert::Dimensions&& dimensions,
    ::litert::TensorBufferType buffer_type =
        ::litert::TensorBufferType::kHostMemory,
    ::litert::Environment* env = nullptr) {
  if (buffer_type != ::litert::TensorBufferType::kHostMemory &&
      env == nullptr) {
    return ::litert::Unexpected(
        ::litert::Status::kErrorInvalidArgument,
        "Environment is required for non-host memory buffer.");
  }
  ::litert::Expected<::litert::TensorBuffer> output_tensor_buffer =
      ::litert::Unexpected(::litert::Status::kErrorUnknown);
  if (buffer_type == ::litert::TensorBufferType::kHostMemory) {
    output_tensor_buffer = ::litert::TensorBuffer::CreateManagedHostMemory(
        ::litert::RankedTensorType(ElementTypeFor<T>::kType,
                                   ::litert::Layout(std::move(dimensions))),
        data.size() * sizeof(T));
  } else {
    output_tensor_buffer = ::litert::TensorBuffer::CreateManaged(
        *env, buffer_type,
        ::litert::RankedTensorType(ElementTypeFor<T>::kType,
                                   ::litert::Layout(std::move(dimensions))),
        data.size() * sizeof(T));
  }
  if (!output_tensor_buffer.HasValue()) {
    return output_tensor_buffer.Error();
  }
  LITERT_RETURN_IF_ERROR(output_tensor_buffer->Write(data));
  return std::move(*output_tensor_buffer);
}

// Similar to CopyToTensorBuffer(), but converts the data type before copying.
template <typename TargetType, typename SourceType>
::litert::Expected<::litert::TensorBuffer> ConvertAndCopyToTensorBuffer(
    absl::Span<const SourceType> source, ::litert::Dimensions&& dimensions,
    ::litert::TensorBufferType buffer_type =
        ::litert::TensorBufferType::kHostMemory,
    ::litert::Environment* env = nullptr) {
  if (buffer_type != ::litert::TensorBufferType::kHostMemory &&
      env == nullptr) {
    return ::litert::Unexpected(
        ::litert::Status::kErrorInvalidArgument,
        "Environment is required for non-host memory buffer.");
  }
  ::litert::Expected<::litert::TensorBuffer> tensor_buffer =
      ::litert::Unexpected(::litert::Status::kErrorUnknown);
  if (buffer_type == ::litert::TensorBufferType::kHostMemory) {
    tensor_buffer = ::litert::TensorBuffer::CreateManagedHostMemory(
        ::litert::RankedTensorType(ElementTypeFor<TargetType>::kType,
                                   ::litert::Layout(std::move(dimensions))),
        source.size() * sizeof(TargetType));
  } else {
    tensor_buffer = ::litert::TensorBuffer::CreateManaged(
        *env, buffer_type,
        ::litert::RankedTensorType(ElementTypeFor<TargetType>::kType,
                                   ::litert::Layout(std::move(dimensions))),
        source.size() * sizeof(TargetType));
  }
  if (!tensor_buffer.HasValue()) {
    return tensor_buffer.Error();
  }

  auto lock_and_addr = ::litert::TensorBufferScopedLock::Create(
      *tensor_buffer, TensorBuffer::LockMode::kWrite);
  ABSL_DCHECK(lock_and_addr.HasValue());
  auto* target = static_cast<TargetType*>(lock_and_addr->second);
  for (int i = 0; i < source.size(); ++i) {
    target[i] = static_cast<TargetType>(source[i]);
  }
  return std::move(*tensor_buffer);
}

// References (no copy) the internal buffer of a ::litert::TensorBuffer when
// it is in the host memory. It's preferable to CopyFromTensorBuffer() whenever
// possible since it's more efficient.
template <typename T>
::litert::Expected<absl::Span<T>> ReferTensorBufferAsSpan(
    const ::litert::TensorBuffer& tensor_buffer) {
  if (auto buffer_type = tensor_buffer.BufferType();
      !buffer_type.HasValue() ||
      *buffer_type != ::litert::TensorBufferType::kHostMemory) {
    return ::litert::Unexpected(::litert::Status::kErrorInvalidArgument,
                                "Tensor buffer is not in the host memory.");
  }

  auto type = tensor_buffer.TensorType();
  if (!type.HasValue() || type->ElementType() != ElementTypeFor<T>::kType) {
    return ::litert::Unexpected(
        ::litert::Status::kErrorInvalidArgument,
        "Element type is not compatible to the target type.");
  }

  auto lock_and_addr = ::litert::TensorBufferScopedLock::Create(
      *const_cast<::litert::TensorBuffer*>(&tensor_buffer),
      TensorBuffer::LockMode::kRead);
  ABSL_DCHECK(lock_and_addr.HasValue());
  LITERT_ASSIGN_OR_RETURN(auto num_elements, type->Layout().NumElements());
  return absl::MakeSpan(static_cast<T*>(lock_and_addr->second), num_elements);
}

// TODO: b/431234598 - This copies data between GPU and CPU backends which
// can be improved with a copy-and-rotate in TensorBuffer api.
// Requires a read right lock on the input buffer.
// Args:
//   tensor_buffer: The input tensor buffer to drop tokens from.
//   num_tokens_to_drop: The number of tokens to drop from the target dimension.
//     It must be non-negative and less than the size of the target dimension.
//   dimension: The target dimension to rotate. It must be a valid dimension
//     index of the tensor buffer.
//   reset_remainder_to_zero: If true, the remainder of the target dimension
//     after rotation will be reset to zero.
//     Otherwise the remainder will be left as is.
//   init_tokens_to_retain: The number of tokens to retain from the target
//     dimension before dropping the `num_tokens_to_drop` tokens.
//     It must be non-negative and less than the size of the target dimension -
//     num_tokens_to_drop.
//      If not specified, it defaults to 0, retaining all tokens.
template <typename T>
::litert::Expected<void> DropTokensfromTensorBuffer(
    ::litert::TensorBuffer& tensor_buffer, int num_tokens_to_drop = 0,
    int dimension = 0, int init_tokens_to_retain = 0,
    bool reset_remainder_to_zero = true) {
  auto type = tensor_buffer.TensorType();
  if (!type.HasValue() || type->ElementType() != ElementTypeFor<T>::kType) {
    return ::litert::Unexpected(
        ::litert::Status::kErrorInvalidArgument,
        "Element type is not compatible to the target type.");
  }
  auto dimensions = type->Layout().Dimensions();
  if (dimensions.size() <= dimension) {
    return ::litert::Unexpected(::litert::Status::kErrorInvalidArgument,
                                "Target dimension is out of range.");
  }
  if (num_tokens_to_drop < 0) {
    return ::litert::Unexpected(::litert::Status::kErrorInvalidArgument,
                                "num_tokens_to_drop is negative.");
  }
  int prev_dims_size = 1;
  for (int i = 0; i < dimension; ++i) {
    prev_dims_size *= dimensions[i];
  }
  int target_dims_size = dimensions[dimension];
  int next_dims_size = 1;
  for (int i = dimension + 1; i < dimensions.size(); ++i) {
    next_dims_size *= dimensions[i];
  }
  if (num_tokens_to_drop > target_dims_size) {
    return ::litert::Unexpected(
        ::litert::Status::kErrorInvalidArgument,
        "num_tokens_to_drop is larger than the target dimension.");
  }
  if (init_tokens_to_retain > target_dims_size) {
    return ::litert::Unexpected(
        ::litert::Status::kErrorInvalidArgument,
        "init_tokens_to_retain is larger than the target dimension.");
  }
  if (init_tokens_to_retain < 0) {
    return ::litert::Unexpected(::litert::Status::kErrorInvalidArgument,
                                "init_tokens_to_retain is negative.");
  }
  if (init_tokens_to_retain + num_tokens_to_drop > target_dims_size) {
    return ::litert::Unexpected(
        ::litert::Status::kErrorInvalidArgument,
        "the total number of tokens retained and dropped is greater than the "
        "target dimension. This will result in an out of bounds access.");
  }
  LITERT_ASSIGN_OR_RETURN(
      auto lock_and_addr,
      ::litert::TensorBufferScopedLock::Create(
          tensor_buffer, TensorBuffer::LockMode::kReadWrite));
  auto* target_ptr = static_cast<T*>(lock_and_addr.second);
  for (int i = 0; i < prev_dims_size; ++i) {
    for (int j = init_tokens_to_retain;
         j < target_dims_size - num_tokens_to_drop; ++j) {
      int dst_offset =
          i * next_dims_size * target_dims_size + j * next_dims_size;
      int src_offset = i * next_dims_size * target_dims_size +
                       (j + num_tokens_to_drop) * next_dims_size;
      std::memcpy(target_ptr + dst_offset, target_ptr + src_offset,
                  next_dims_size * sizeof(T));
    }
    if (reset_remainder_to_zero) {
      int start_j_reset_addr = target_dims_size - num_tokens_to_drop;
      int dst_offset = i * target_dims_size * next_dims_size +
                       start_j_reset_addr * next_dims_size;
      int total_elements_to_reset = next_dims_size * num_tokens_to_drop;
      // Multiply with sizeof(T) to account for data size.
      std::memset(target_ptr + dst_offset, 0,
                  total_elements_to_reset * sizeof(T));
    }
  }
  return ::litert::Expected<void>{};
}
}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_CONVERT_TENSOR_BUFFER_H_
