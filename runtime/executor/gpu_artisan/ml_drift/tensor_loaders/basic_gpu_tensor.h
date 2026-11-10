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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_TENSOR_LOADERS_BASIC_GPU_TENSOR_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_TENSOR_LOADERS_BASIC_GPU_TENSOR_H_

#include <cstdint>
#include <utility>
#include <vector>

#include "ml_drift/common/task/gpu_tensor.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift

namespace odml::infra::gpu {

// A stripped-down `GpuSpatialTensor` that stores a copy of the data in
// `tensor_desc`.
class BasicGpuSpatialTensor : public ::ml_drift::GpuSpatialTensor {
 public:
  explicit BasicGpuSpatialTensor(
      const ::ml_drift::TensorDescriptor& tensor_desc)
      : tensor_desc_(tensor_desc) {
    std::vector<uint8_t> data_copy(tensor_desc.GetData().begin(),
                                   tensor_desc.GetData().end());
    tensor_desc_.SetData(std::move(data_copy));
  }

  int Width() const override { return 0; }
  int Height() const override { return 0; }
  int Depth() const override { return 0; }
  int Channels() const override { return 0; }
  int Slices() const override { return 0; }
  int Batch() const override { return 0; }

  ::ml_drift::TensorDescriptor GetDescriptor() const override {
    return tensor_desc_;
  }

 private:
  ::ml_drift::TensorDescriptor tensor_desc_;
};

}  // namespace odml::infra::gpu

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_TENSOR_LOADERS_BASIC_GPU_TENSOR_H_
