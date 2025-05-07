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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_VISION_EXECUTOR_BASE_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_VISION_EXECUTOR_BASE_H_

#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"  // from @litert

namespace litert::lm {

class VisionExecutorBase {
 public:
  virtual ~VisionExecutorBase() = default;

  // ------------Encode APIs------------:
  // Basic API to trigger the "encode" process.
  // Input is image tensor with shape `[batch, height, width, channels]`
  // Output is main embeddings with shape `[batch, 1, num_vision_tokens,
  // model_dimension]` and per layer embeddings with shape `[batch, stack_size,
  // num_vision_tokens, per_layer_embedding_dimension]`.
  //
  // The output_per_layer_embeddings is optional and can be nullptr.
  virtual absl::StatusOr<std::vector<litert::TensorBuffer>> Encode(
      const litert::TensorBuffer& input_image_tensor) = 0;

  // Get the input dimension of the vision executor.
  // [batch, height, width, channels]
  virtual absl::StatusOr<std::vector<int>> GetInputDimension() const = 0;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_VISION_EXECUTOR_BASE_H_
