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

#include "runtime/util/log_tensor_buffer.h"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_expected.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/util/convert_tensor_buffer.h"
#include "tflite/types/half.h"  // from @litert

namespace litert::lm {

namespace {

template <typename T, typename Container>
void LogValues(const Container& values, size_t num_values_to_log,
               absl::string_view prefix) {
  auto formatter = [](std::string* out, const auto& value) {
    if constexpr (std::is_same_v<T, bool>) {
      absl::StrAppend(out, value ? 1 : 0);
    } else {
      // Unary '+' promotes quantized integers to int so it prints as a number
      // instead of a char.
      absl::StrAppend(out, +value);
    }
  };

  constexpr size_t kNumExtraValuesToLog = 10;
  if (num_values_to_log * 3 + kNumExtraValuesToLog >= values.size()) {
    ABSL_LOG(INFO) << prefix << "(size=" << values.size()
                   << "): " << absl::StrJoin(values, ", ", formatter);
    return;
  }

  size_t end_offset = values.size() - num_values_to_log;
  size_t mid_offset = end_offset / 2;
  ABSL_LOG(INFO) << prefix << "(size=" << values.size() << "): "
                 << absl::StrJoin(values.begin(),
                                  values.begin() + num_values_to_log, ", ",
                                  formatter)
                 << " ... "
                 << absl::StrJoin(
                        values.begin() + mid_offset,
                        values.begin() + mid_offset + num_values_to_log, ", ",
                        formatter)
                 << " ... "
                 << absl::StrJoin(values.begin() + end_offset, values.end(),
                                  ", ", formatter);
}

template <typename T>
absl::Status TryLogTensor(TensorBuffer& tensor, size_t num_values_to_log,
                          absl::string_view prefix) {
  // Try to get the reference if tensor is in CPU memory.
  Expected<absl::Span<T>> values_span = ReferTensorBufferAsSpan<T>(tensor);
  if (values_span) {
    LogValues<T>(*values_span, num_values_to_log, prefix);
    return absl::OkStatus();
  }

  // Otherwise, copy the logits from the tensor buffer to a vector.
  LITERT_ASSIGN_OR_RETURN(std::vector<T> values_vector,
                          CopyFromTensorBuffer<T>(tensor));
  LogValues<T>(values_vector, num_values_to_log, prefix);
  return absl::OkStatus();
}

template <typename T>
absl::Status TryDumpTensorToCsv(TensorBuffer& tensor,
                                absl::string_view filename) {
  auto write_csv = [&](const auto& values) {
    std::ofstream out((std::string(filename)));
    for (size_t i = 0; i < values.size(); ++i) {
      if constexpr (std::is_same_v<T, tflite::half>) {
        out << static_cast<float>(values[i]);
      } else if constexpr (std::is_same_v<T, bool>) {
        out << (values[i] ? 1.0 : 0.0);
      } else {
        out << +values[i];
      }
      if (i + 1 < values.size()) out << ",";
    }
    out << "\n";
  };

  litert::Expected<absl::Span<T>> values_span =
      ReferTensorBufferAsSpan<T>(tensor);
  if (values_span) {
    write_csv(*values_span);
    return absl::OkStatus();
  }

  LITERT_ASSIGN_OR_RETURN(std::vector<T> values_vector,
                          CopyFromTensorBuffer<T>(tensor));
  write_csv(values_vector);
  return absl::OkStatus();
}

}  // namespace

absl::Status LogTensor(TensorBuffer& tensor, size_t num_values_to_log,
                       absl::string_view prefix) {
  LITERT_ASSIGN_OR_RETURN(auto tensor_type, tensor.TensorType());

  switch (tensor_type.ElementType()) {
    case litert::ElementType::Float32:
      LITERT_RETURN_IF_ERROR(
          TryLogTensor<float>(tensor, num_values_to_log, prefix));
      break;
    case litert::ElementType::Int8:
      LITERT_RETURN_IF_ERROR(
          TryLogTensor<int8_t>(tensor, num_values_to_log, prefix));
      break;
    case litert::ElementType::Bool:
      LITERT_RETURN_IF_ERROR(
          TryLogTensor<bool>(tensor, num_values_to_log, prefix));
      break;
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported tensor type: ", tensor_type.ElementType()));
  }

  return absl::OkStatus();
}

absl::Status DumpTensorToCsv(TensorBuffer& tensor, absl::string_view filename) {
  LITERT_ASSIGN_OR_RETURN(auto tensor_type, tensor.TensorType());

  switch (tensor_type.ElementType()) {
    case litert::ElementType::Float32:
      LITERT_RETURN_IF_ERROR(TryDumpTensorToCsv<float>(tensor, filename));
      break;
    case litert::ElementType::Int8:
      LITERT_RETURN_IF_ERROR(TryDumpTensorToCsv<int8_t>(tensor, filename));
      break;
    case litert::ElementType::Float16:
      LITERT_RETURN_IF_ERROR(
          TryDumpTensorToCsv<tflite::half>(tensor, filename));
      break;
    case litert::ElementType::Int32:
      LITERT_RETURN_IF_ERROR(TryDumpTensorToCsv<int32_t>(tensor, filename));
      break;
    case litert::ElementType::Bool:
      LITERT_RETURN_IF_ERROR(TryDumpTensorToCsv<bool>(tensor, filename));
      break;
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported tensor type: ", tensor_type.ElementType()));
  }

  return absl::OkStatus();
}

}  // namespace litert::lm
