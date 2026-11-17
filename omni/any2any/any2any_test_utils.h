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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_ANY2ANY_TEST_UTILS_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_ANY2ANY_TEST_UTILS_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "tflite/schema/schema_generated.h"  // from @litert

namespace litert::omni::any2any {

// Specification of an input or output tensor in a minimal test TFLite model.
struct TestTensorSpec {
  std::string name;
  std::vector<int32_t> dims;
  tflite::TensorType type;
};

// Builds a minimal executable TFLite FlatBuffer model with the requested
// signature input and output tensor specifications (all outputs emit zeros).
// Output tensors must use `tflite::TensorType_FLOAT32`, matching the
// unquantized `ADD` op synthesized for each output.
std::string BuildMinimalTfliteModelBuffer(
    absl::Span<const TestTensorSpec> inputs,
    absl::Span<const TestTensorSpec> outputs);

}  // namespace litert::omni::any2any

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_ANY2ANY_TEST_UTILS_H_
