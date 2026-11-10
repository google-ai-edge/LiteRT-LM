// Copyright 2024 The MediaPipe Authors.
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

#include "runtime/executor/gpu_artisan/ml_drift/tensor_loaders/llm_tensor_loader.h"

#include <cstddef>
#include <memory>
#include <string>
#include <utility>

#include "ml_drift/common/data_type.h"  // from @ml_drift

namespace odml::infra::gpu {

// static
std::unique_ptr<LlmTensorLoader> LlmTensorLoader::MakeCaching(
    std::unique_ptr<LlmTensorLoader> loader) {
  return std::make_unique<CachingTensorLoader>(std::move(loader));
}

ml_drift::DataType LlmTensorLoader::GetDataTypeForWeights(
    const std::string& name, int count) {
  const size_t element_size_in_bits = GetTensorElementSizeInBits(name, count);
  if (element_size_in_bits == 32) {
    return ml_drift::DataType::kFloat32;
  } else if (element_size_in_bits == 16) {
    return ml_drift::DataType::kFloat16;
  } else if (element_size_in_bits == 8) {
    return ml_drift::DataType::kInt8;
  } else if (element_size_in_bits == 4) {
    return ml_drift::DataType::kInt4;
  } else if (element_size_in_bits == 2) {
    return ml_drift::DataType::kInt2;
  }
  return ml_drift::DataType::kUnknown;
}

}  // namespace odml::infra::gpu
