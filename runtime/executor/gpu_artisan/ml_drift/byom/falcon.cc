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

#include "runtime/executor/gpu_artisan/ml_drift/byom/falcon.h"

#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/substitute.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_operation.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/types.h"  // from @ml_drift
#include "runtime/executor/gpu_artisan/ml_drift/byom/llm_base.h"
#include "runtime/executor/gpu_artisan/ml_drift/tasks/apply_attention_mask.h"
#include "runtime/executor/gpu_artisan/ml_drift/transformer.h"

namespace odml_byom {

using ::ml_drift::DataType;
using ::ml_drift::GpuModelBuilder;

namespace {
GPUOperation CreateApplyAttentionMaskAlibiOp(const GpuInfo& gpu_info,
                                             const TensorDescriptor& src_mask,
                                             const TensorDescriptor& dst_mask,
                                             int number_of_heads) {
  ml_drift::ElementwiseDescriptor op_desc;
  float min_float = dst_mask.GetDataType() == DataType::kFloat32
                        ? std::numeric_limits<float>::max()
                        : ml_drift::kMaxHalf;
  min_float *= -0.7f;
  op_desc.args.AddFloat("fill_value", min_float);
  float alibi_slope = std::pow(2, -8 / static_cast<float>(number_of_heads));
  op_desc.args.AddFloat("alibi_slope", alibi_slope);

  std::string pow_func_name = "pow";
  if (gpu_info.IsAdreno() && gpu_info.IsApiOpenCl()) {
    pow_func_name = "native_powr";
  }

  op_desc.code = absl::Substitute(
      R"(
  float4 src_mask = ucl::Convert<float4>(in_value);
  int index = S_COORD * 4 - X_COORD;
  int4 indexes;
  indexes.x = index + 0;
  indexes.y = index + 1;
  indexes.z = index + 2;
  indexes.w = index + 3;
  float multiplier = $1(args.alibi_slope, ucl::Convert<float>(Y_COORD + 1));
  float4 mult0 = ucl::Convert<float4>(indexes) * multiplier;
  float4 mult1 = ucl::Init<float4>(args.fill_value);
  float4 out_mask = src_mask * mult1 + (src_mask - ucl::Init<float4>(1.0f)) * mult0;
  out_value = ucl::Convert<$0>(out_mask);
)",
      ToUclDataType(dst_mask.GetDataType(), 4), pow_func_name);
  ml_drift::OperationDef definition;
  definition.src_tensors.push_back(src_mask);
  definition.dst_tensors.push_back(dst_mask);
  return CreateGpuOperation(definition, std::move(op_desc));
}
}  // namespace

absl::StatusOr<GpuModelBuilder::TensorHandle>
FalconTransformerBuilder::MakeSelfAttention(
    const GpuModelBuilder::TensorHandle& src,
    const GpuModelBuilder::TensorHandle& attention_mask,
    const GpuModelBuilder::TensorHandle& segment_pos,
    const GpuModelBuilder::TensorHandle& key_tensor,
    const GpuModelBuilder::TensorHandle& value_tensor,
    const GpuModelBuilder::TensorHandle& params_tensor, bool is_global,
    const std::string& name) {
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
  ABSL_ASSIGN_OR_RETURN(auto qkv_arr, MakeGetQKV(src, name));
  auto q = qkv_arr[0];  // bhwc(B, 1, S, HC*HS)
  auto k = qkv_arr[1];  // bhwc(B, 1, S, KVHC*HS)
  auto v = qkv_arr[2];  // bhwc(B, 1, S, KVHC*HS)

  MakeTransformQKV(model_builder_, &q, &k, &v);

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
    val = model_builder_.Add(val, attention_mask);
    int head_size = config_.model_dimension / config_.number_of_heads;
    return model_builder_.Multiplication(
        val, 1.0f / sqrt(static_cast<float>(head_size)));
  };
  //  q - hwc(B*KVHC, HGS*S, HS)
  //  k - hwc(B*KVHC, CS, HS)
  //  v - hwc(B*KVHC, CS, HS)
  //  att - hwc(B*KVHC, HGS*S, HS)
  ABSL_ASSIGN_OR_RETURN(
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

absl::StatusOr<GpuModelBuilder::TensorHandle>
FalconTransformerBuilder::MakeFFLayer(const GpuModelBuilder::TensorHandle& src,
                                      const std::string& name) {
  // ffn_layer1
  ABSL_ASSIGN_OR_RETURN(auto t,
                        MakeLinearAutoPrecision(src, name + ".ffn_layer1",
                                                config_.hidden_dimension));
  t = model_builder_.MakeGelu(t);

  // ffn_layer2
  ABSL_ASSIGN_OR_RETURN(
      t, MakeLinearAutoPrecision(t, name + ".ffn_layer2",
                                 src.tensor_desc.GetBHWCShape().c));
  return t;
}

absl::StatusOr<GpuModelBuilder::TensorHandle>
FalconTransformerBuilder::MakeLayer(
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
  ABSL_ASSIGN_OR_RETURN(
      hidden_state, MakeSelfAttention(hidden_state, attention_mask, segment_pos,
                                      key_tensor, value_tensor, params_tensor,
                                      is_global, name + ".self_attention"));

  // self attention residual
  hidden_state = model_builder_.Add(hidden_state, residual);

  residual = hidden_state;
  // post layer norm
  hidden_state = MakeNormalization(hidden_state, name + ".post_layer_norm");
  // ff_layer
  ABSL_ASSIGN_OR_RETURN(hidden_state,
                        MakeFFLayer(hidden_state, name + ".ff_layer"));

  // ff residual
  GpuModelBuilder::TensorHandle output =
      model_builder_.Add(hidden_state, residual);

  return output;
}

absl::StatusOr<GpuModelBuilder::TensorHandle>
FalconTransformerBuilder::MakeFillAttention(
    const GpuModelBuilder::TensorHandle& params_i32,
    const GpuModelBuilder::TensorHandle* current_tokens) {
  ABSL_ASSIGN_OR_RETURN(const int mask_width,
                        GetMaskWidth(config_.attention_mask_type));
  const BHWC mask_shape =
      BHWC(1, config_.number_of_heads, mask_width, config_.cache_size);
  if (config_.attention_mask_type ==
      odml::infra::proto::TransformerParameters::BIDIRECTIONAL) {
    auto mask_zeroes = model_builder_.AddTensor(mask_shape, data_type_);
    model_builder_.AddGpuOperation({}, mask_zeroes,
                                   CreateFillTensorOp(mask_zeroes.tensor_desc),
                                   "fill_zeroes");
    return mask_zeroes;
  }
  auto attention_mask = model_builder_.AddTensor(mask_shape, DataType::kBool);
  model_builder_.AddGpuOperation(
      {params_i32}, attention_mask,
      std::make_unique<GPUOperation>(CreateApplyAttentionMaskOp(
          params_i32.tensor_desc, attention_mask.tensor_desc, std::nullopt)),
      "apply_attention_mask");
  auto attention_mask_alibi = model_builder_.AddTensor(mask_shape, data_type_);
  model_builder_.AddGpuOperation(
      {attention_mask}, attention_mask_alibi,
      std::make_unique<GPUOperation>(CreateApplyAttentionMaskAlibiOp(
          gpu_info_, attention_mask.tensor_desc,
          attention_mask_alibi.tensor_desc, config_.number_of_heads)),
      "fill_attention_mask_alibi");
  return attention_mask_alibi;
}

}  // namespace odml_byom
