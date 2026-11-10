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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_TASKS_FUSED_ROPE_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_TASKS_FUSED_ROPE_H_

#include <vector>

#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/kernel_info.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_operation.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/task/tuning_type.h"  // from @ml_drift
#include "ml_drift/common/types.h"  // from @ml_drift

//
// See
// runtime/executor/gpu_artisan/ml_drift/transformer_transform_qkv.cc
// for more kernel information and usage details.
//
namespace ml_drift {

GPUOperation CreateTransformQKVApplyRoPEOp(
    const GpuInfo& gpu_info, const TensorDescriptor& q_src,
    const TensorDescriptor& k_src, const TensorDescriptor& v_src,
    const TensorDescriptor& position, const TensorDescriptor& q_dst,
    const TensorDescriptor& k_dst, const TensorDescriptor& v_dst,
    int kv_heads_count, int kv_head_group_size, int sequence_size,
    float min_timescale, float max_timescale);

class TransformQKVRmsNormApplyRoPEOp : public GPUOperation {
 public:
  TransformQKVRmsNormApplyRoPEOp() = default;

  std::vector<int3> GetPossibleKernelWorkGroups(
      TuningType tuning_type, const GpuInfo& gpu_info,
      const KernelInfo& kernel_info) const override {
    return {work_group_size_};
  }
};

TransformQKVRmsNormApplyRoPEOp CreateTransformQKVRmsNormApplyRoPEOp(
    const GpuInfo& gpu_info, const TensorDescriptor& q_src,
    const TensorDescriptor& position, const TensorDescriptor& q_dst,
    int kv_heads_count, int kv_head_group_size, int sequence_size,
    float min_timescale, float max_timescale, float proportion, bool k_eq_v,
    const TensorDescriptor* k_src = nullptr,
    const TensorDescriptor* v_src = nullptr,
    const TensorDescriptor* k_dst = nullptr,
    const TensorDescriptor* v_dst = nullptr,
    const TensorDescriptor* q_gamma = nullptr,
    const TensorDescriptor* k_gamma = nullptr);

}  // namespace ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_TASKS_FUSED_ROPE_H_
