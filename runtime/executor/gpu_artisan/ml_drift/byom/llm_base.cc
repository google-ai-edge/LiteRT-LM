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

#include "runtime/executor/gpu_artisan/ml_drift/byom/llm_base.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/kernels/split.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_operation.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "ml_drift/common/types.h"  // from @ml_drift
#include "runtime/executor/gpu_artisan/ml_drift/tensor_loaders/llm_tensor_loader.h"
#include "runtime/util/status_macros.h"

namespace odml_byom {

using ml_drift::Axis;
using ml_drift::BHWC;
using ml_drift::BHWDC;
using ml_drift::DataType;
using ml_drift::GpuModelBuilder;
using ml_drift::HW;
using ml_drift::int3;
using ml_drift::Linear;
using ml_drift::OHWI;
using ml_drift::OperationDef;
using ml_drift::Split;
using ml_drift::TensorFloat32;
using ml_drift::uint2;

GpuModelBuilder::TensorHandle TransformerBaseBuilder::MakeLayerNormalization(
    const GpuModelBuilder::TensorHandle& src, const std::string& name) {
  ml_drift::Tensor<Linear, DataType::kFloat32> gamma;
  ml_drift::Tensor<Linear, DataType::kFloat32> beta;
  gamma.shape = Linear(src.tensor_desc.GetBHWCShape().c);
  gamma.data = tensor_loader_->LoadFloat32(name + ".scale",
                                           src.tensor_desc.GetBHWCShape().c);
  beta.shape = Linear(src.tensor_desc.GetBHWCShape().c);
  beta.data = tensor_loader_->LoadFloat32(name + ".bias",
                                          src.tensor_desc.GetBHWCShape().c);
  return model_builder_.LayerNormalization(src, gamma, beta,
                                           /*epsilon*/ 1e-5f);
}

// Apply SRC's first (rope_ratio * head_size) dimensions with Position
// Embedding, while remaining dimensions are kept original.
GpuModelBuilder::TensorHandle TransformerBaseBuilder::MakePartialRoPE(
    const GpuModelBuilder::TensorHandle& src,
    const GpuModelBuilder::TensorHandle& position, float rope_ratio) {
  const int head_size = src.tensor_desc.GetBHWCShape().c;
  const int embed_head_size = head_size * rope_ratio;

  auto src_splits = CustomSplit(
      src, Axis::kChannels,
      {embed_head_size / 2, embed_head_size / 2, head_size - embed_head_size});
  auto embeds = model_builder_.RoPE(src_splits[0], src_splits[1], position);
  src_splits[0] = embeds[0];
  src_splits[1] = embeds[1];
  auto dst = model_builder_.Concat(src_splits, Axis::kChannels);
  return dst;
}

// Split a dimension to arbitrary number of sub-dimensions.
std::vector<GpuModelBuilder::TensorHandle> TransformerBaseBuilder::CustomSplit(
    const GpuModelBuilder::TensorHandle& src, Axis axis,
    const std::vector<int>& sizes) {
  std::vector<GpuModelBuilder::TensorHandle> dsts(sizes.size());
  for (int i = 0; i < sizes.size(); ++i) {
    BHWC new_shape = src.tensor_desc.GetBHWCShape();
    ABSL_QCHECK_GT(sizes[i], 0) << "The dimension size must be positive";
    new_shape.set(axis, sizes[i]);
    dsts[i] =
        model_builder_.AddTensor(new_shape, src.tensor_desc.GetDataType());
  }

  ml_drift::SplitAttributes split_attr;
  split_attr.axis = axis;
  OperationDef split_def;
  split_def.src_tensors.push_back(src.tensor_desc);
  for (int i = 0; i < sizes.size(); ++i) {
    split_def.dst_tensors.push_back(dsts[i].tensor_desc);
  }
  std::vector<int> channels;
  if (axis == Axis::kChannels) {
    for (auto size : sizes) {
      channels.push_back(size);
    }
  }
  Split operation = CreateSplit(gpu_info_, split_def, split_attr, channels);
  model_builder_.AddGpuOperation({src}, dsts,
                                 std::make_unique<Split>(std::move(operation)),
                                 "CustomSplit");
  return dsts;
}

absl::StatusOr<GpuModelBuilder::TensorHandle>
TransformerBaseBuilder::MakeSelfAttentionWithPartialRope(
    const GpuModelBuilder::TensorHandle& src,
    const GpuModelBuilder::TensorHandle& attention_mask,
    const GpuModelBuilder::TensorHandle& segment_pos,
    const GpuModelBuilder::TensorHandle& key_tensor,
    const GpuModelBuilder::TensorHandle& value_tensor,
    const GpuModelBuilder::TensorHandle& params_tensor,
    float partial_rope_ratio, const std::string& name) {
  // B - batch_size
  // S - sequence_size
  // HC - heads_count
  // HS - head_size
  // HGS - head_group_size (MHA: HGS = 1, MQA: HGS = HC)
  // KHC - HC / HGS
  // VHC - HC / HGS
  // KHC = VHC!
  // KVHC = KHC = VHC -> HC = KVHC * HGS
  auto src_sh = src.tensor_desc.GetBHWCShape();
  if (src_sh.c % config_.number_of_heads != 0) {
    return absl::InvalidArgumentError(
        "Can not build the model. Probably incorrect config.");
  }
  const int heads_count = config_.number_of_heads;
  const int head_size = src_sh.c / heads_count;
  const int batch_size = src_sh.b;
  const int sequence_size = src_sh.w;
  ASSIGN_OR_RETURN(auto qkv_arr, MakeGetQKV(src, name));
  auto q = qkv_arr[0];  // bhwc(B, 1, S, HC*HS)
  auto k = qkv_arr[1];  // bhwc(B, 1, S, KVHC*HS)
  auto v = qkv_arr[2];  // bhwc(B, 1, S, KVHC*HS)

  MakeTransformQKV(model_builder_, &q, &k, &v);

  q = MakePartialRoPE(q, segment_pos, partial_rope_ratio);
  k = MakePartialRoPE(k, segment_pos, partial_rope_ratio);
  ASSIGN_OR_RETURN(q, PerDimScale(model_builder_, q, name));

  AddValuesToCache(model_builder_, k, v, params_tensor, key_tensor,
                   value_tensor);
  if (kv_cache_batch_size_ != 1 && config_.batch_size == 1) {
    // extract first batch slice, this can happen when batched kv cache used
    // with non batched model, that can happen in multi output model
    k = ExtractFirstBatch(model_builder_, key_tensor);
    v = ExtractFirstBatch(model_builder_, value_tensor);
  } else {
    k = key_tensor;
    v = value_tensor;
  }

  auto qk_post_proc_graph = [this, attention_mask](auto src) {
    GpuModelBuilder::TensorHandle val = src;
    if (config_.attn_soft_cap_value > 0.0f) {  // cap_logits
      float cap_value = config_.attn_soft_cap_value;
      val = model_builder_.Multiplication(val,
                                          1.0f / config_.attn_soft_cap_value);
      val = model_builder_.Elementwise(val, ml_drift::OperationType::kTanh);
      val = model_builder_.Multiplication(val, cap_value);
    }
    return model_builder_.Add(val, attention_mask);
  };
  //  q - hwc(B*KVHC, HGS*S, HS)
  //  k - hwc(B*KVHC, CS, HS)
  //  v - hwc(B*KVHC, CS, HS)
  //  att - hwc(B*KVHC, HGS*S, HS)
  ASSIGN_OR_RETURN(
      auto att, MakeMatMulSoftmaxMatMul(model_builder_, q, k, v, params_tensor,
                                        qk_post_proc_graph,
                                        /*is_global=*/true));
  // att transform to bhwc(B, 1, S, KVHC*HGS*HS)
  // att - bhwc(B, KVHC*HGS, S, HS) = bhwc(B, HC, S, HS)
  // att - bhwc(B, S, HC, HS)
  // att - bhwc(B, 1, S, HC*HS)
  att = model_builder_.Reshape(
      att, BHWC{batch_size, heads_count, sequence_size, head_size});
  att = model_builder_.Transpose(att, BHWC(0, 2, 1, 3));
  att = model_builder_.Reshape(att, src_sh);

  return MakeLinearAutoPrecision(att, name + ".post", src_sh.c);
}

std::string TransformerBaseBuilder::GetEmbeddingLookupName() const {
  return "params.lm.token_embedding";
}

GpuModelBuilder::TensorHandle TransformerBaseBuilder::MakeNormalization(
    const GpuModelBuilder::TensorHandle& src, const std::string& name) {
  return MakeLayerNormalization(src, name);
}

}  // namespace odml_byom
