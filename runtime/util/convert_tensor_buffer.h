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
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"  // from @litert
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_expected.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert

namespace litert::lm {

template <typename T>
struct ElementTypeFor {
  // Don't define kType to generate a compile error for unsupported types.
};

// Here is the list of supported element types effectively. Support only minimal
// types for now to avoid compatibility issues, e.g. whether or not uint8 is
// compatible with int8.
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

// Creates a ::litert::TensorBuffer with the given dimensions and data.
template <typename T>
::litert::Expected<::litert::TensorBuffer> CreateTensorBuffer(
    ::litert::Dimensions&& dimensions,
    LiteRtTensorBufferType buffer_type = kLiteRtTensorBufferTypeHostMemory) {
  int size = 1;
  for (int dim : dimensions) {
    size *= dim;
  }

  return ::litert::TensorBuffer::CreateManaged(
      buffer_type,
      ::litert::RankedTensorType(ElementTypeFor<T>::kType,
                                 ::litert::Layout(std::move(dimensions))),
      size * sizeof(T));
}

// Copies a ::litert::TensorBuffer of arbitrary shape to a std::vector<T>.
template <typename T>
::litert::Expected<std::vector<T>> CopyFromTensorBuffer(
    ::litert::TensorBuffer& tensor_buffer) {
  if (auto type = tensor_buffer.TensorType();
      !type.HasValue() || type->ElementType() != ElementTypeFor<T>::kType) {
    return ::litert::Unexpected(
        kLiteRtStatusErrorInvalidArgument,
        "Element type is not compatible to the target type.");
  }

  auto size = tensor_buffer.Size();
  ABSL_DCHECK(size.HasValue());
  std::vector<T> copied_data(*size / sizeof(T));
  tensor_buffer.Read(absl::MakeSpan(copied_data));
  return copied_data;
}

// Const version of CopyFromTensorBuffer() above.
template <typename T>
::litert::Expected<std::vector<T>> CopyFromTensorBuffer(
    const ::litert::TensorBuffer& tensor_buffer) {
  auto mutable_tensor_buffer = tensor_buffer.Duplicate();
  if (!mutable_tensor_buffer) {
    return mutable_tensor_buffer.Error();
  }
  return CopyFromTensorBuffer<T>(*mutable_tensor_buffer);
}

// Copies a 2D ::litert::TensorBuffer to a std::vector<std::vector<T>>.
template <typename T>
::litert::Expected<std::vector<std::vector<T>>> CopyFromTensorBuffer2D(
    ::litert::TensorBuffer& tensor_buffer) {
  auto type = tensor_buffer.TensorType();
  if (!type.HasValue() || type->ElementType() != ElementTypeFor<T>::kType) {
    return ::litert::Unexpected(
        kLiteRtStatusErrorInvalidArgument,
        "Element type is not compatible to the target type.");
  }

  auto dimensions = type->Layout().Dimensions();
  if (dimensions.size() != 2) {
    return ::litert::Unexpected(kLiteRtStatusErrorInvalidArgument,
                                "Tensor buffer must have 2 dimensions.");
  }

  auto lock_and_addr = ::litert::TensorBufferScopedLock::Create(tensor_buffer);
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

// Const version of CopyFromTensorBuffer2D() above.
template <typename T>
::litert::Expected<std::vector<std::vector<T>>> CopyFromTensorBuffer2D(
    const ::litert::TensorBuffer& tensor_buffer) {
  auto mutable_tensor_buffer = tensor_buffer.Duplicate();
  if (!mutable_tensor_buffer) {
    return mutable_tensor_buffer.Error();
  }
  return CopyFromTensorBuffer2D<T>(*mutable_tensor_buffer);
}

// Copies an absl::Span<const T> to a ::litert::TensorBuffer with the given
// dimensions.
template <typename T>
::litert::Expected<::litert::TensorBuffer> CopyToTensorBuffer(
    absl::Span<const T> data, ::litert::Dimensions&& dimensions,
    LiteRtTensorBufferType buffer_type = kLiteRtTensorBufferTypeHostMemory) {
  auto output_tensor_buffer = ::litert::TensorBuffer::CreateManaged(
      buffer_type,
      ::litert::RankedTensorType(ElementTypeFor<T>::kType,
                                 ::litert::Layout(std::move(dimensions))),
      data.size() * sizeof(T));
  if (!output_tensor_buffer.HasValue()) {
    return output_tensor_buffer.Error();
  }

  output_tensor_buffer->Write(data);
  return std::move(*output_tensor_buffer);
}

// Similar to CopyToTensorBuffer(), but converts the data type before copying.
template <typename TargetType, typename SourceType>
::litert::Expected<::litert::TensorBuffer> ConvertAndCopyToTensorBuffer(
    absl::Span<const SourceType> source, ::litert::Dimensions&& dimensions,
    LiteRtTensorBufferType buffer_type = kLiteRtTensorBufferTypeHostMemory) {
  auto tensor_buffer = ::litert::TensorBuffer::CreateManaged(
      buffer_type,
      ::litert::RankedTensorType(ElementTypeFor<TargetType>::kType,
                                 ::litert::Layout(std::move(dimensions))),
      source.size() * sizeof(TargetType));
  if (!tensor_buffer.HasValue()) {
    return tensor_buffer.Error();
  }

  auto lock_and_addr = ::litert::TensorBufferScopedLock::Create(*tensor_buffer);
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
    ::litert::TensorBuffer& tensor_buffer) {
  if (auto buffer_type = tensor_buffer.BufferType();
      !buffer_type.HasValue() ||
      *buffer_type != kLiteRtTensorBufferTypeHostMemory) {
    return ::litert::Unexpected(kLiteRtStatusErrorInvalidArgument,
                                "Tensor buffer is not in the host memory.");
  }

  if (auto type = tensor_buffer.TensorType();
      !type.HasValue() || type->ElementType() != ElementTypeFor<T>::kType) {
    return ::litert::Unexpected(
        kLiteRtStatusErrorInvalidArgument,
        "Element type is not compatible to the target type.");
  }

  auto size = tensor_buffer.Size();
  ABSL_DCHECK(size.HasValue());
  auto lock_and_addr = ::litert::TensorBufferScopedLock::Create(tensor_buffer);
  ABSL_DCHECK(lock_and_addr.HasValue());
  return absl::MakeSpan(static_cast<T*>(lock_and_addr->second),
                        *size / sizeof(T));
}

// Const version of ReferTensorBufferAsSpan() above.
template <typename T>
::litert::Expected<absl::Span<T>> ReferTensorBufferAsSpan(
    const ::litert::TensorBuffer& tensor_buffer) {
  auto mutable_tensor_buffer = tensor_buffer.Duplicate();
  if (!mutable_tensor_buffer) {
    return mutable_tensor_buffer.Error();
  }
  return ReferTensorBufferAsSpan<T>(*mutable_tensor_buffer);
}

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_CONVERT_TENSOR_BUFFER_H_
