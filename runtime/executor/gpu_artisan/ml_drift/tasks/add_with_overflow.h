// Copyright 2024 The ODML Authors.
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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_TASKS_ADD_WITH_OVERFLOW_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_TASKS_ADD_WITH_OVERFLOW_H_

#include "ml_drift/common/task/gpu_operation.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift

namespace ml_drift {

// value must be >= 0 (positive)
GPUOperation CreateAddWithOverflow(const TensorDescriptor& desc, int value);

// all values in second input must be >= 0 (positive)
GPUOperation CreateAddWithOverflow2Input(
    const TensorDescriptor& desc0, const TensorDescriptor& desc1);

}  // namespace ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_TASKS_ADD_WITH_OVERFLOW_H_
