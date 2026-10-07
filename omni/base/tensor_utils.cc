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

#include "omni/base/tensor_utils.h"

#include <cstddef>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert

namespace litert::omni {

absl::Status ValidateFloatBuffer(const TensorBuffer& buffer,
                                 size_t expected_elements,
                                 absl::string_view buffer_name) {
  LITERT_ASSIGN_OR_RETURN(const RankedTensorType tensor_type,
                          buffer.TensorType());
  if (tensor_type.ElementType() != ElementType::Float32) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s buffer element type must be Float32.", buffer_name));
  }
  LITERT_ASSIGN_OR_RETURN(const size_t num_elements,
                          tensor_type.Layout().NumElements());
  LITERT_ASSIGN_OR_RETURN(const size_t packed_size, buffer.PackedSize());
  if (num_elements != expected_elements ||
      packed_size != expected_elements * sizeof(float)) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s buffer capacity (%d elements, %d bytes) does not match expected "
        "(%d elements, %d bytes).",
        buffer_name, num_elements, packed_size, expected_elements,
        expected_elements * sizeof(float)));
  }
  return absl::OkStatus();
}

}  // namespace litert::omni
