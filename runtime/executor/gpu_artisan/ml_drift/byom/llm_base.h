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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_BYOM_LLM_BASE_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_BYOM_LLM_BASE_H_

#include <string>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_operation.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_tensor.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "runtime/executor/gpu_artisan/ml_drift/transformer.h"

namespace odml_byom {

using ml_drift::Axis;
using ml_drift::BHWC;
using ml_drift::BHWDC;
using ml_drift::CreateGpuModelInfo;
using ml_drift::GpuInfo;
using ml_drift::GpuModel;
using ml_drift::GpuModelBuilder;
using ml_drift::GPUOperation;
using ml_drift::GpuSpatialTensor;
using ml_drift::Layout;
using ml_drift::Linear;
using ml_drift::Tensor;
using ml_drift::TensorDescriptor;

class TransformerBaseBuilder : public ml_drift::TransformerBuilder {
 public:
  using TransformerBuilder::TransformerBuilder;
  virtual ~TransformerBaseBuilder() = default;

  std::string GetEmbeddingLookupName() const override;
  GpuModelBuilder::TensorHandle MakeNormalization(
      const GpuModelBuilder::TensorHandle& src,
      const std::string& name) override;

  const CreateGpuModelInfo& GetCreateInfo() { return create_info_; }

 protected:
  bool ScaleEmbeddingsBySqrtModelDim() override { return false; }

  absl::StatusOr<GpuModelBuilder::TensorHandle>
  MakeSelfAttentionWithPartialRope(
      const GpuModelBuilder::TensorHandle& src,
      const GpuModelBuilder::TensorHandle& attention_mask,
      const GpuModelBuilder::TensorHandle& segment_pos,
      const GpuModelBuilder::TensorHandle& key_tensor,
      const GpuModelBuilder::TensorHandle& value_tensor,
      const GpuModelBuilder::TensorHandle& params_tensor,
      float partial_rope_ratio, const std::string& name);

  GpuModelBuilder::TensorHandle MakeLayerNormalization(
      const GpuModelBuilder::TensorHandle& src, const std::string& name);

  // Split -> Partial RoPE -> Concat
  GpuModelBuilder::TensorHandle MakePartialRoPE(
      const GpuModelBuilder::TensorHandle& src,
      const GpuModelBuilder::TensorHandle& position, float rope_ratio);

  std::vector<GpuModelBuilder::TensorHandle> CustomSplit(
      const GpuModelBuilder::TensorHandle& src, Axis axis,
      const std::vector<int>& sizes);
};

}  // namespace odml_byom

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_BYOM_LLM_BASE_H_
