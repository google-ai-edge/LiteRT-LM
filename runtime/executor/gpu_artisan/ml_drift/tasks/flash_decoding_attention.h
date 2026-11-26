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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_TASKS_FLASH_DECODING_ATTENTION_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_TASKS_FLASH_DECODING_ATTENTION_H_

#include <string>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/kernel_info.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_operation.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/task/tuning_type.h"  // from @ml_drift
#include "ml_drift/common/types.h"  // from @ml_drift
#include "runtime/executor/gpu_artisan/ml_drift/runtime_params.h"

namespace ml_drift {

// Fused decode-time attention:
//
//   q       : (1, 1, num_q_heads, head_size)        F16/F32.
//   k_cache : (1, 1, 1, cache_size * head_size)     Layout::kLinear,
//             memory order kOSpatialIOGroupO4I4 (O = position, I = dim).
//   v_cache : (1, 1, 1, head_size * cache_size)     Layout::kLinear,
//             memory order kOSpatialIOGroupI4O4 (O = dim, I = position).
//   params  : int32 buffer, LlmRuntimeParams layout;
//             params[active_tokens_index] is L (the current token's K/V is
//             already in the cache).
//   dst     : (1, 1, num_q_heads, head_size)        same type as q.
struct FlashDecodingAttentionAttributes {
  int num_q_heads = 16;
  int num_kv_heads = 1;
  int head_size = 128;
  int cache_size = 2048;
  int active_tokens_index = LlmRuntimeParams::kActiveTokensIndex;
};

class FlashDecodingAttention : public GPUOperation {
 public:
  FlashDecodingAttention() = default;

  // Move only
  FlashDecodingAttention(FlashDecodingAttention&& kernel) = default;
  FlashDecodingAttention& operator=(FlashDecodingAttention&& kernel) = default;
  FlashDecodingAttention(const FlashDecodingAttention&) = delete;
  FlashDecodingAttention& operator=(const FlashDecodingAttention&) = delete;

  int3 GetGridSize() const override;

  std::vector<int3> GetPossibleKernelWorkGroups(
      TuningType tuning_type, const GpuInfo& gpu_info,
      const KernelInfo& kernel_info) const override {
    return {work_group_size_};
  }

  const FlashDecodingAttentionAttributes& attributes() const { return attr_; }

 private:
  FlashDecodingAttention(const GpuInfo& gpu_info, const TensorDescriptor& q,
                         const TensorDescriptor& k_cache,
                         const TensorDescriptor& v_cache,
                         const TensorDescriptor& dst,
                         const FlashDecodingAttentionAttributes& attr);

  friend absl::StatusOr<FlashDecodingAttention> CreateFlashDecodingAttention(
      const GpuInfo& gpu_info, const TensorDescriptor& q,
      const TensorDescriptor& k_cache, const TensorDescriptor& v_cache,
      const TensorDescriptor& dst,
      const FlashDecodingAttentionAttributes& attr);

  // Generates the kernel source (UCL dialect, see
  // //third_party/ml_drift/common/task/util.cc for available intrinsics).
  std::string GenerateCode(const GpuInfo& gpu_info) const;

  FlashDecodingAttentionAttributes attr_;
};

// Applicability check.
bool IsFlashDecodingAttentionSupported(
    const GpuInfo& gpu_info, const FlashDecodingAttentionAttributes& attr,
    std::string* absl_nonnull error);

// Check if tensors match with the attributes and expected data types.
absl::Status ValidateFlashDecodingAttentionTensors(
    const TensorDescriptor& q, const TensorDescriptor& k_cache,
    const TensorDescriptor& v_cache, const TensorDescriptor& dst,
    const FlashDecodingAttentionAttributes& attr);

absl::StatusOr<FlashDecodingAttention> CreateFlashDecodingAttention(
    const GpuInfo& gpu_info, const TensorDescriptor& q,
    const TensorDescriptor& k_cache, const TensorDescriptor& v_cache,
    const TensorDescriptor& dst, const FlashDecodingAttentionAttributes& attr);

}  // namespace ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_TASKS_FLASH_DECODING_ATTENTION_H_
