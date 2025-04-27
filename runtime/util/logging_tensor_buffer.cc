#include "runtime/util/logging_tensor_buffer.h"

#include <cstdint>
#include <iostream>

#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_tensor_buffer_types.h"  // from @litert
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert

namespace litert::lm {
namespace {

constexpr absl::string_view kTensorBufferPrefix = "TensorBuffer: ";

template <typename T>
std::ostream& LogNestedTensorBuffer(std::ostream& os, const void* data,
                                    absl::Span<const int32_t> dimensions) {
  ABSL_DCHECK_GT(dimensions.size(), 0);
  auto* typed_data = reinterpret_cast<const T*>(data);

  os << "[";
  if (dimensions.size() == 1) {
    os << absl::StrJoin(absl::MakeConstSpan(typed_data, dimensions[0]), ", ");
  } else {
    // Log nested tensor buffers.
    int num_elements_per_col = 1;
    for (int i = 1; i < dimensions.size(); ++i) {
      num_elements_per_col *= dimensions[i];
    }

    for (int i = 0; i < dimensions[0]; ++i) {
      LogNestedTensorBuffer<T>(os, typed_data + i * num_elements_per_col,
                               dimensions.subspan(1));
      if (i != dimensions[0] - 1) {
        os << ", ";
      }
    }
  }
  return os << "]";
}

template <typename T>
std::ostream& LogTensorBuffer(std::ostream& os, const void* data,
                              absl::Span<const int32_t> dimensions) {
  ABSL_DCHECK_GT(dimensions.size(), 0);
  os << kTensorBufferPrefix;
  LogNestedTensorBuffer<T>(os, data, dimensions);
  return os << " shape=(" << absl::StrJoin(dimensions, ", ") << ")";
}

}  // namespace

std::ostream& operator<<(std::ostream& os,
                         const ::litert::TensorBuffer& tensor_buffer) {
  if (auto type = tensor_buffer.BufferType();
      !type.HasValue() || *type != kLiteRtTensorBufferTypeHostMemory) {
    return os << kTensorBufferPrefix << "[tensor in non-host memory type="
              << (type.HasValue() ? *type : kLiteRtTensorBufferTypeUnknown)
              << "]";
  }

  auto tensor_type = tensor_buffer.TensorType();
  if (!tensor_type.HasValue()) {
    return os << kTensorBufferPrefix
              << "[tensor in host memory of tensor type=Unknown]";
  }

  auto lock_and_addr = ::litert::TensorBufferScopedLock::Create(
      // Though const_cast() here is not ideal, it is actually const when the
      // tensor buffer is in host memory.
      *const_cast<::litert::TensorBuffer*>(&tensor_buffer));

  switch (tensor_type->ElementType()) {
    case ::litert::ElementType::Int8:
      return LogTensorBuffer<int8_t>(os, lock_and_addr->second,
                                     tensor_type->Layout().Dimensions());
    case ::litert::ElementType::Int16:
      return LogTensorBuffer<int16_t>(os, lock_and_addr->second,
                                      tensor_type->Layout().Dimensions());
    case ::litert::ElementType::Int32:
      return LogTensorBuffer<int32_t>(os, lock_and_addr->second,
                                      tensor_type->Layout().Dimensions());
    case ::litert::ElementType::Int64:
      return LogTensorBuffer<int64_t>(os, lock_and_addr->second,
                                      tensor_type->Layout().Dimensions());
    case ::litert::ElementType::UInt8:
      return LogTensorBuffer<uint8_t>(os, lock_and_addr->second,
                                      tensor_type->Layout().Dimensions());
    case ::litert::ElementType::UInt16:
      return LogTensorBuffer<uint16_t>(os, lock_and_addr->second,
                                       tensor_type->Layout().Dimensions());
    case ::litert::ElementType::UInt32:
      return LogTensorBuffer<uint32_t>(os, lock_and_addr->second,
                                       tensor_type->Layout().Dimensions());
    case ::litert::ElementType::UInt64:
      return LogTensorBuffer<uint64_t>(os, lock_and_addr->second,
                                       tensor_type->Layout().Dimensions());
    case ::litert::ElementType::Float32:
      return LogTensorBuffer<float>(os, lock_and_addr->second,
                                    tensor_type->Layout().Dimensions());
    default:
      return os << "[tensor in host memory of type="
                << static_cast<int>(tensor_type->ElementType()) << "]";
  }
}

}  // namespace litert::lm
