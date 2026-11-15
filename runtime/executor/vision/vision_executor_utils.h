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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_VISION_EXECUTOR_UTILS_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_VISION_EXECUTOR_UTILS_H_

#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/components/model_resources.h"
#include "runtime/engine/io_types.h"

namespace litert::lm {

// Utility function to get the properties of the vision executor from LiteRT
// Model. This function fetches the properties from the vision model by
// inspecting the model signature, input tensor names and tensor shapes.
//
// This function throws an error if the model is not a valid vision encoder
// model.
//
// Args:
//   - model_resources: The model resources to inspect.
// Returns:
//   A VisionExecutorProperties object containing the properties of the vision
//   executor.
absl::StatusOr<VisionExecutorProperties>
GetVisionExecutorPropertiesFromModelResources(ModelResources& model_resources);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_VISION_EXECUTOR_UTILS_H_
