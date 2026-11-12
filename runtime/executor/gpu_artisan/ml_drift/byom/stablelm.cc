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

#include "runtime/executor/gpu_artisan/ml_drift/byom/stablelm.h"

#include <string>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "runtime/executor/gpu_artisan/ml_drift/byom/llm_base.h"
#include "runtime/executor/gpu_artisan/ml_drift/llm.h"
#include "runtime/util/status_macros.h"

namespace odml_byom {

constexpr float kPartialRopeRatio = 0.25f;

absl::StatusOr<GpuModelBuilder::TensorHandle>
StablelmTransformerBuilder::MakeFFLayer(
    const GpuModelBuilder::TensorHandle& src, const std::string& name) {
  // ffn_layer1_gate
  ASSIGN_OR_RETURN(auto gate_value,
                   MakeLinearAutoPrecision(src, name + ".ffn_layer1_gate",
                                           config_.hidden_dimension));
  gate_value = model_builder_.SiLU(gate_value);
  // ffn_layer1
  ASSIGN_OR_RETURN(auto t, MakeLinearAutoPrecision(src, name + ".ffn_layer1",
                                                   config_.hidden_dimension));
  t = model_builder_.Multiplication(t, gate_value);

  // ffn_layer2
  ASSIGN_OR_RETURN(t,
                   MakeLinearAutoPrecision(t, name + ".ffn_layer2",
                                           src.tensor_desc.GetBHWCShape().c));
  // post_layer_norm
  return t;
}

absl::StatusOr<GpuModelBuilder::TensorHandle>
StablelmTransformerBuilder::MakeLayer(
    const GpuModelBuilder::TensorHandle& inputs,
    const GpuModelBuilder::TensorHandle& attention_mask,
    const GpuModelBuilder::TensorHandle& segment_pos,
    const GpuModelBuilder::TensorHandle& key_tensor,
    const GpuModelBuilder::TensorHandle& value_tensor,
    const GpuModelBuilder::TensorHandle& params_tensor, bool is_global,
    const std::string& name) {
  GpuModelBuilder::TensorHandle residual = inputs;

  // Input layer norm
  GpuModelBuilder::TensorHandle hidden_state =
      MakeNormalization(inputs, name + ".pre_layer_norm");

  // self attention
  ASSIGN_OR_RETURN(hidden_state,
                   MakeSelfAttention(hidden_state, attention_mask, segment_pos,
                                     key_tensor, value_tensor, params_tensor,
                                     is_global, name + ".self_attention"));

  // self attention residual
  hidden_state = model_builder_.Add(hidden_state, residual);

  residual = hidden_state;
  // post layer norm
  hidden_state = MakeNormalization(hidden_state, name + ".post_layer_norm");
  // ff_layer
  ASSIGN_OR_RETURN(hidden_state, MakeFFLayer(hidden_state, name + ".ff_layer"));

  // ff residual
  GpuModelBuilder::TensorHandle output =
      model_builder_.Add(hidden_state, residual);

  return output;
}

absl::StatusOr<GpuModelBuilder::TensorHandle>
StablelmTransformerBuilder::MakeSelfAttention(
    const GpuModelBuilder::TensorHandle& src,
    const GpuModelBuilder::TensorHandle& attention_mask,
    const GpuModelBuilder::TensorHandle& segment_pos,
    const GpuModelBuilder::TensorHandle& key_tensor,
    const GpuModelBuilder::TensorHandle& value_tensor,
    const GpuModelBuilder::TensorHandle& params_tensor, bool is_global,
    const std::string& name) {
  return MakeSelfAttentionWithPartialRope(
      src, attention_mask, segment_pos, key_tensor, value_tensor, params_tensor,
      /*partial_rope_ratio=*/kPartialRopeRatio, name);
}

}  // namespace odml_byom
