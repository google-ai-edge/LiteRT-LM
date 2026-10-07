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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_BASE_TENSOR_UTILS_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_BASE_TENSOR_UTILS_H_

#include <cstddef>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"  // from @litert

namespace litert::omni {

// Validates that `buffer` has `ElementType::Float32`, `expected_elements`
// elements in its layout, and a packed byte size equal to
// `expected_elements * sizeof(float)`.
//
// args
// - buffer: TensorBuffer to validate.
// - expected_elements: Expected number of float elements in the buffer.
// - buffer_name: Human-readable buffer description included in error messages.
//
// returns
// - absl::OkStatus() if valid, or absl::InvalidArgumentError otherwise.
absl::Status ValidateFloatBuffer(const TensorBuffer& buffer,
                                 size_t expected_elements,
                                 absl::string_view buffer_name);

}  // namespace litert::omni

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_BASE_TENSOR_UTILS_H_
