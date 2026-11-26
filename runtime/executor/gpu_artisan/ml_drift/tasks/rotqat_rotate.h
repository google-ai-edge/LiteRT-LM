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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_TASKS_ROTQAT_ROTATE_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_TASKS_ROTQAT_ROTATE_H_

#include <optional>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/kernel_info.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_operation.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/task/tuning_type.h"  // from @ml_drift
#include "ml_drift/common/types.h"  // from @ml_drift

namespace ml_drift {

// Given an input tensor with 7 * 2**n channels, this operation is equivalent
// to multiplying the tensor by a specific 7x7 orthonormal matrix (see
// `kOrtho7`) to the rows of the channels, with the channels interpreted as a
// [2**n, 7] matrix, and then applying a Hadamard Transform (see
// https://en.wikipedia.org/wiki/Hadamard_transform) to the columns (or
// sub-vectors of the columns).
// The implementation uses the Fast Walsh-Hadamard Transform (see
// https://en.wikipedia.org/wiki/Fast_Walsh%E2%80%93Hadamard_transform) and
// takes advantage of the fact that the 2**(a+b) x 2**(a+b) Hadamard Transform
// is equivalent to the Kronecker Product of a 2**a x 2**a Hadamard Transform
// and a 2**b x 2**b Hadamard Transform. i.e. given a 2**(a+b) dimensional
// vector we can divide it into 2**a blocks of size 2**b and apply a
// 2**a x 2**a Hadamard Transform across the blocks, then apply a 2**b x 2**b
// Hadamard Transform to each block.
// Depending on the parameter settings, this transformation may be implemented
// as a single fused kernel or two separate kernels.
class RotqatRotateOp : public GPUOperation {
 public:
  RotqatRotateOp() = default;
  RotqatRotateOp(bool use_o7_h4, std::optional<int> hadamard_dim)
      : use_o7_h4_(use_o7_h4), hadamard_dim_(hadamard_dim) {}

  int3 GetGridSize() const override {
    if (use_o7_h4_) {
      return int3(src_[0]->Batch() * src_[0]->Width(), src_[0]->Height(),
                  src_[0]->Slices() / 7);
    } else {
      return int3(src_[0]->Batch() * src_[0]->Width(), src_[0]->Height(),
                  src_[0]->Slices());
    }
  }

  std::vector<int3> GetPossibleKernelWorkGroups(
      TuningType tuning_type, const GpuInfo& gpu_info,
      const KernelInfo& kernel_info) const override {
    return {work_group_size_};
  }

  // Move only
  RotqatRotateOp(RotqatRotateOp&& operation) = default;
  RotqatRotateOp& operator=(RotqatRotateOp&& operation) = default;
  RotqatRotateOp(const RotqatRotateOp&) = delete;
  RotqatRotateOp& operator=(const RotqatRotateOp&) = delete;

  // If both `use_o7_h4_` and `hadamard_dim_` are set, then we generate a
  // single fused kernel implementing the entire operation. Generated kernel
  // code will differ depending on work group sizes. At least one of
  // `use_o7_h4_` and `hadamard_dim_` must be set.

  // If `true` then the kernel will apply the 7x7 orthonormal matrix
  // multiplication to the rows of the channels, with the channels interpreted
  // interpreted as a [2**n, 7] matrix. It will then apply a 4x4 Hadamard
  // Transform to each block of 4 elements in the columns of the channels.
  bool use_o7_h4_ = true;
  // If set, the kernel will apply a `hadamard_dim_` x `hadamard_dim_` Hadamard
  // Transform to the columns of the slices, with the slices interpreted as
  // a [pow2, 7] matrix. If `hadamard_dim_` is less than pow2, then the Hadamard
  // Transform will be applied over each block of size `hadamard_dim_`.
  std::optional<int> hadamard_dim_ = std::nullopt;
};

absl::StatusOr<RotqatRotateOp> CreateRotqatRotateOp(
    const GpuInfo& gpu_info, const TensorDescriptor& src,
    const TensorDescriptor& dst, bool use_o7_h4,
    std::optional<int> hadamard_dim);

}  // namespace ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_TASKS_ROTQAT_ROTATE_H_
