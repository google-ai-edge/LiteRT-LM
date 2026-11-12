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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_BYOM_PHI_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_BYOM_PHI_H_

#include <string>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "runtime/executor/gpu_artisan/ml_drift/byom/llm_base.h"

namespace odml_byom {

class PhiTransformerBuilder : public TransformerBaseBuilder {
 public:
  using TransformerBaseBuilder::TransformerBaseBuilder;

 protected:
  absl::StatusOr<GpuModelBuilder::TensorHandle> MakeLayer(
      const GpuModelBuilder::TensorHandle& inputs,
      const GpuModelBuilder::TensorHandle& attention_mask,
      const GpuModelBuilder::TensorHandle& segment_pos,
      const GpuModelBuilder::TensorHandle& key_tensor,
      const GpuModelBuilder::TensorHandle& value_tensor,
      const GpuModelBuilder::TensorHandle& params_tensor, bool is_global,
      const std::string& name) override;

  absl::StatusOr<GpuModelBuilder::TensorHandle> MakeFFLayer(
      const GpuModelBuilder::TensorHandle& src,
      const std::string& name) override;

  absl::StatusOr<GpuModelBuilder::TensorHandle> MakeSelfAttention(
      const GpuModelBuilder::TensorHandle& src,
      const GpuModelBuilder::TensorHandle& attention_mask,
      const GpuModelBuilder::TensorHandle& segment_pos,
      const GpuModelBuilder::TensorHandle& key_tensor,
      const GpuModelBuilder::TensorHandle& value_tensor,
      const GpuModelBuilder::TensorHandle& params_tensor, bool is_global,
      const std::string& name) override;
};

}  // namespace odml_byom

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_BYOM_PHI_H_
