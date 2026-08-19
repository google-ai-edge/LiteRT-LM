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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_MODEL_SIGNATURE_UTILS_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_MODEL_SIGNATURE_UTILS_H_

#include <string>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/cc/litert_model.h"  // from @litert
#include "runtime/components/model_resources.h"

namespace litert::lm {

// Represents metadata about a model signature including its name, capacity,
// and tensor shapes.
struct SignatureInfo {
  std::string signature_name;
  // Sequence length (tokens) for text encoders, or visual token count for
  // vision models.
  int length = 0;
  std::vector<int> input_shape;
  std::vector<int> output_shape;
};

// Inspects the given model and extracts all available signatures with their
// corresponding capacities and dimensions for the specified ModelType.
//
// Supported ModelTypes:
// - ModelType::kTfLiteTextEncoder
// - ModelType::kTfLiteVisionEncoder
// - ModelType::kTfLiteVisionAdapter
absl::StatusOr<std::vector<SignatureInfo>> GetAvailableSignatures(
    const litert::Model& model, ModelType model_type);

// Convenience overload that retrieves the model from ModelResources.
absl::StatusOr<std::vector<SignatureInfo>> GetAvailableSignatures(
    ModelResources& resources, ModelType model_type);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_MODEL_SIGNATURE_UTILS_H_
