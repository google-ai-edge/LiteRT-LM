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

#include "omni/utils/tensor_utils.h"

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
  return ValidateTensorBuffer(buffer, ElementType::Float32, expected_elements,
                              buffer_name);
}

absl::Status ValidateTensorBuffer(const TensorBuffer& buffer,
                                  ElementType expected_type,
                                  size_t expected_elements,
                                  absl::string_view buffer_name) {
  LITERT_ASSIGN_OR_RETURN(const RankedTensorType tensor_type,
                          buffer.TensorType());
  if (tensor_type.ElementType() != expected_type) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s buffer element type (%d) does not match expected (%d).",
        buffer_name, static_cast<int>(tensor_type.ElementType()),
        static_cast<int>(expected_type)));
  }
  LITERT_ASSIGN_OR_RETURN(const size_t num_elements,
                          tensor_type.Layout().NumElements());
  if (num_elements != expected_elements) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s buffer element count (%d) does not match expected (%d).",
        buffer_name, num_elements, expected_elements));
  }
  return absl::OkStatus();
}

}  // namespace litert::omni
