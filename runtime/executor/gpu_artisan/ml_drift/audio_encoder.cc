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

#include "runtime/executor/gpu_artisan/ml_drift/audio_encoder.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/substitute.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/kernels/elementwise.h"  // from @ml_drift
#include "ml_drift/common/kernels/reduce.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/precision.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/buffer_desc.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_operation.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "runtime/executor/gpu_artisan/ml_drift/runtime_params.h"
#include "runtime/executor/gpu_artisan/ml_drift/transformer.h"

namespace ml_drift {
namespace {

// Extra trailing bytes padded onto host weight buffers rearranged on CPU.
constexpr size_t kWeightRearrangeExtraBytes = 16;

// The spatial block size of conformer block size in v3 models.
// kConformerBlockSize = 4 corresponds to an input of 150ms, and an output
// of 150ms. This is what v2 uses. v3 previously used 12. We use this as
// opposed to using the cfg value to avoid having to regenerate a model
// every time we want to stream different lengths of audio.
constexpr int kConformerBlockSize = 4;

std::tuple<int, int> ComputeConvPadding(int dim_size, int filter_size,
                                        int stride,
                                        SubSamplePaddingType padding_spec) {
  int pad_total;
  switch (padding_spec) {
    case SubSamplePaddingType::CAUSAL:
      return {filter_size - 1, 0};
    case SubSamplePaddingType::REVERSE_CAUSAL:
      pad_total = std::max(
          0, static_cast<int>(std::ceil(dim_size / stride) - 1) * stride +
                 filter_size - dim_size);
      return {0, pad_total};
    case SubSamplePaddingType::SAME:
      pad_total = std::max(
          0, static_cast<int>(std::ceil(dim_size / stride) - 1) * stride +
                 filter_size - dim_size);
      {
        int pad_beg = pad_total / 2;
        return {pad_beg, pad_total - pad_beg};
      }
    default:
      return {0, 0};
  }
}
}  // namespace

TensorHandle AudioEncoderBuilder::Gather(const TensorHandle& src,
                                         std::vector<int>& indices) {
  TensorDescriptor indices_desc = TensorDescriptor(
      DataType::kInt32, src.tensor_desc.GetStorageType(), Layout::kHWC);
  indices_desc.SetBHWCShape(BHWC(1, 1, 1, indices.size()));
  indices_desc.UploadData(indices.data());

  TensorHandle indices_gpu =
      model_builder_.AddConstantTensor(std::move(indices_desc));
  return model_builder_.Gather(src, indices_gpu, Axis::kWidth);
}

// Returns the first len elements along the width dimension of src.
absl::StatusOr<TensorHandle> AudioEncoderBuilder::GetFirstN(
    const TensorHandle& src, int len) {
  const BHWC src_sh = src.tensor_desc.GetBHWCShape();
  if (len == src_sh.w) {
    return src;
  } else if (len > src_sh.w) {
    return absl::InvalidArgumentError(
        "Truncate length is larger than source length.");
  }
  std::vector<int> indices(len);
  for (int i = 0; i < len; ++i) {
    indices[i] = i;
  }
  return Gather(src, indices);
}

// Returns the last len elements along the width dimension of src.
absl::StatusOr<TensorHandle> AudioEncoderBuilder::GetLastN(
    const TensorHandle& src, int len) {
  BHWC src_sh = src.tensor_desc.GetBHWCShape();
  if (len == src_sh.w) {
    return src;
  } else if (len > src_sh.w) {
    return absl::InvalidArgumentError(
        "Truncate length is larger than source length.");
  }
  std::vector<int> indices(len);
  for (int i = 0; i < len; ++i) {
    // Get last "len" indices from the array of length src_sh.w
    indices[i] = src_sh.w - len + i;
  }
  return Gather(src, indices);
}

TensorHandle AudioEncoderBuilder::ComputeOutputPadding(
    const TensorHandle& padding_tensor, const HW& filter_stride) {
  if (filter_stride.h == 1) {
    return padding_tensor;
  }
  TensorHandle padding_split = padding_tensor;
  if (IsStreaming()) {
    padding_split = model_builder_.Split(
        padding_tensor, Axis::kHeight,
        {filter_stride.h,
         padding_tensor.tensor_desc.GetBHWCShape().h - filter_stride.h})[1];
  }
  int input_length = padding_tensor.tensor_desc.GetBHWCShape().h;
  int stride = filter_stride.h;
  int pad_len = (input_length + stride - 1) / stride * stride - input_length;
  Convolution2DAttributes conv_padding_attr;
  conv_padding_attr.strides = HW(filter_stride.h, 1);
  Padding2D padding;
  padding.prepended = HW(0, 0);
  padding.appended = HW(pad_len, 0);

  const OHWI weight_shape = OHWI(1, 1, 1, 1);
  auto& attr_weights =
      conv_padding_attr.weights.emplace<Tensor<OHWI, DataType::kFloat32>>();
  attr_weights.shape = weight_shape;
  attr_weights.data.resize(weight_shape.DimensionsProduct() +
                           kWeightRearrangeExtraBytes / sizeof(float));
  attr_weights.data[0] = 1.0f;
  conv_padding_attr.dilations = HW(1, 0);
  conv_padding_attr.padding = padding;
  // No bias
  conv_padding_attr.bias.data.resize(0);
  return model_builder_.Convolution(padding_split, conv_padding_attr);
}

GPUOperation CreateNormalizeVar(
    const GpuInfo& gpu_info, const CreateGpuModelInfo& create_info,
    const TensorDescriptor& src, const TensorDescriptor& mean,
    const TensorDescriptor& variance, const TensorDescriptor& dst,
    const Tensor<Linear, DataType::kFloat32>& gamma,
    const Tensor<Linear, DataType::kFloat32>& beta, float epsilon) {
  OperationDef op_def;
  op_def.src_tensors.push_back(src);
  op_def.src_tensors.push_back(mean);
  op_def.src_tensors.push_back(variance);
  op_def.dst_tensors.push_back(dst);

  GPUOperation op;
  op.AddSrcTensor("src_tensor", src);
  op.AddSrcTensor("mean_tensor", mean);
  op.AddSrcTensor("variance", variance);
  op.AddDstTensor("dst", dst);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
  op.args_.AddFloat("variance_bias", epsilon);

  BHWC shape = BHWC(1, 1, 1, gamma.data.size());
  DataType data_type = DeduceDataTypeFromPrecision(create_info.precision);
  TensorDescriptor gamma_tensor_desc =
      CreateConstantLinearTensorDescriptor(gpu_info, data_type, gamma);
  op.args_.AddObject("gamma", std::make_unique<TensorDescriptor>(
                                  std::move(gamma_tensor_desc)));

  TensorDescriptor beta_tensor_desc =
      CreateConstantLinearTensorDescriptor(gpu_info, data_type, beta);
  op.args_.AddObject(
      "beta", std::make_unique<TensorDescriptor>(std::move(beta_tensor_desc)));

  std::string c;
  c += "MAIN_FUNCTION($0) {\n";
  if (op_def.dst_tensors[0].HasAxis(Axis::kBatch)) {
    c += "  int linear_id = ucl::GetGlobalId<0>();\n";
    c += "  int X = linear_id / args.dst.Batch();\n";
    c += "  int B = linear_id % args.dst.Batch();\n";
    c += "  args.src_tensor.SetBatchRef(B);\n";
    c += "  args.mean_tensor.SetBatchRef(B);\n";
    c += "  args.variance.SetBatchRef(B);\n";
    c += "  args.dst.SetBatchRef(B);\n";
  } else {
    c += "  int X = ucl::GetGlobalId<0>();\n";
  }
  c += R"(
      int Y = ucl::GetGlobalId<1>();
      int S = ucl::GetGlobalId<2>();
      if (X >= args.dst.Width() || Y >= args.dst.Height() || S >= args.dst.Slices()) return;
      float mean = args.mean_tensor.Read<float>(0, Y, 0).x;
      float variance = args.variance.Read<float>(0, Y, 0).x;
      float stddev_inv = rsqrt(variance + args.variance_bias);

      float4 t = args.src_tensor.Read<float>(X, Y, S);
      float4 t_normalized = (t - mean) * stddev_inv;
      t_normalized *= args.gamma.Read<float>(S);
      t_normalized += args.beta.Read<float>(S);
      args.dst::type result = ucl::Convert<args.dst::type>(t_normalized);
      args.dst.Write(result, X, Y, S);
    })";
  op.code_ = std::move(c);
  return op;
}

absl::StatusOr<std::vector<TensorHandle>>
AudioEncoderBuilder::MakeConv2DWithMask(const TensorHandle& feature,
                                        const TensorHandle& mask,
                                        const OHWI& filter_shape,
                                        const ConvConfig& conv_config,
                                        const std::string& name) {
  // feature: [B, H, W, D_in]
  // mask: [B, H]
  HW filter_stride = conv_config.stride;
  HWPadding padding = conv_config.padding;
  const BHWC& input_shape = feature.tensor_desc.GetBHWCShape();
  TensorHandle input_masked = model_builder_.Multiplication(feature, mask);

  // Next calculate padding
  Padding2D conv_padding;
  if (!feature_config_.is_causal) {
    std::tuple<int, int> h_padding = ComputeConvPadding(
        input_shape.h, filter_shape.h, filter_stride.h, padding.h);
    std::tuple<int, int> w_padding = ComputeConvPadding(
        input_shape.w, filter_shape.w, filter_stride.w, padding.w);
    conv_padding.prepended = HW(std::get<0>(h_padding), std::get<0>(w_padding));
    conv_padding.appended = HW(std::get<1>(h_padding), std::get<1>(w_padding));
  } else {
    int pad_height_total = IsStreaming() ? 0 : filter_shape.h - 1;

    int pad_width_total = filter_shape.w - 1;
    int pad_width_beg = pad_width_total / 2;
    conv_padding.prepended = HW(pad_height_total, pad_width_beg);
    conv_padding.appended = HW(0, pad_width_total - pad_width_beg);
  }
  if (IsModelv3(config_)) {  // switch padding for H dim due to v3 bug
    std::swap(conv_padding.prepended.h, conv_padding.appended.h);
  }
  Convolution2DAttributes conv_attr;
  conv_attr.strides = filter_stride;
  conv_attr.padding = conv_padding;
  conv_attr.dilations = HW(1, 1);

  if (conv_config.bias) {
    return absl::UnimplementedError("bias not supported");
  } else {  // No bias
    conv_attr.bias.data.resize(0);
  }

  auto& attr_weights =
      conv_attr.weights.emplace<Tensor<OHWI, DataType::kFloat32>>();
  attr_weights.data = tensor_loader_->LoadFloat32(
      name + ".w", filter_shape.DimensionsProduct());
  attr_weights.data.resize(filter_shape.DimensionsProduct() +
                           kWeightRearrangeExtraBytes / sizeof(float));
  attr_weights.shape = filter_shape;

  TensorHandle conv_handle =
      model_builder_.Convolution(input_masked, conv_attr);
  ElementwiseAttributes elem_attr;
  elem_attr.param = 1.0f;
  elem_attr.runtime_tensor_is_second = true;
  TensorHandle padding_tensor =  // Not operator
      model_builder_.Elementwise(mask, elem_attr, OperationType::kSub);
  TensorHandle out_padding =
      ComputeOutputPadding(padding_tensor, filter_stride);
  TensorHandle out_mask =  // Not operator
      model_builder_.Elementwise(out_padding, elem_attr, OperationType::kSub);
  TensorHandle output = model_builder_.Multiplication(conv_handle, out_mask);
  std::vector<TensorHandle> outputs = {output, out_mask};
  return outputs;
}

TensorHandle AudioEncoderBuilder::UpdateCumsum(const TensorHandle& input,
                                               int layer, int index) {
  int extract_idx =
      feature_config_.take_every_n * feature_config_.tokens_per_frame;
  for (int i = feature_config_.convs.size() - 1; i > layer; --i) {
    extract_idx *= feature_config_.convs[i].stride.h;
  }
  const int len = input.tensor_desc.GetBHWCShape().h;
  std::vector<int> sizes = {extract_idx - 1, 1};
  if (extract_idx != len) {
    sizes.push_back(len - extract_idx);
  }
  TensorHandle extracted_tensor =
      model_builder_.Split(input, Axis::kHeight, sizes)[1];
  TensorFloat32 update_cpu;
  update_cpu.shape = BHWC(1, 1, 2, 3);
  update_cpu.data = std::vector<float>(6, 0.0f);
  update_cpu.data[layer * 3 + index] = 1.0f;
  TensorHandle update_tensor = model_builder_.AddConstantTensor(
      update_cpu, input.tensor_desc.GetDataType());
  if (update_tensor.tensor_desc.GetDataType() == DataType::kFloat16) {
    update_tensor = model_builder_.Cast(update_tensor, DataType::kFloat32);
  }
  return model_builder_.Multiplication(update_tensor, extracted_tensor);
}

TensorHandle AudioEncoderBuilder::HWCMaskedGroupNorm(
    TensorHandle src, TensorHandle mask, const TensorHandle& stream_cumsum,
    int layer, float epsilon, const Tensor<Linear, DataType::kFloat32>& gamma,
    const Tensor<Linear, DataType::kFloat32>& beta) {
  const bool use_f16 =
      DeduceDataTypeFromPrecision(create_info_.precision) == DataType::kFloat16;
  if (use_f16) {  // Cast to f32 to prevent overflow; cast back at end of op
    src = model_builder_.Cast(src, DataType::kFloat32);
    mask = model_builder_.Cast(mask, DataType::kFloat32);
  }
  const auto& src_shape = src.tensor_desc.GetBHWCShape();

  auto masked_src = model_builder_.Multiplication(src, mask);
  std::set<Axis> reduce_axes = {Axis::kWidth, Axis::kChannels};
  if (src_shape.b > 1) {
    reduce_axes.insert(Axis::kBatch);
  }
  if (!feature_config_.is_causal) {
    reduce_axes.insert(Axis::kHeight);
  }
  auto sum_v =
      model_builder_.Reduce(masked_src, Reduce::Type::kSum, reduce_axes);
  auto count_v = model_builder_.Reduce(mask, Reduce::Type::kSum, reduce_axes);
  int scale = src_shape.b * src_shape.w * src_shape.c;
  std::vector<TensorHandle> stream_cumsum_vec;
  TensorHandle stream_cumsum_out;
  if (feature_config_.is_causal) {
    sum_v = model_builder_.Cumsum(sum_v, Axis::kHeight);
    count_v = model_builder_.Cumsum(count_v, Axis::kHeight);
    if (IsStreaming()) {
      stream_cumsum_vec = model_builder_.Split(
          model_builder_.Split(stream_cumsum, Axis::kWidth, 1)[layer],
          Axis::kChannels, 1);
      TensorHandle sum_v_update = UpdateCumsum(sum_v, layer, 0);
      TensorHandle count_v_update = UpdateCumsum(count_v, layer, 1);
      sum_v = model_builder_.Add(sum_v, stream_cumsum_vec[0]);
      count_v = model_builder_.Add(count_v, stream_cumsum_vec[1]);
      stream_cumsum_out = model_builder_.Add(stream_cumsum, sum_v_update);
      stream_cumsum_out = model_builder_.Add(stream_cumsum_out, count_v_update);
    }
  }
  count_v = model_builder_.Multiplication(count_v, scale);
  count_v = model_builder_.Elementwise(count_v, {1.0f},
                                       OperationType::kMaximum);  // no div by 0
  auto mean = model_builder_.Elementwise(sum_v, count_v, OperationType::kDiv);
  auto diff = model_builder_.Elementwise(src, mean, OperationType::kSub);
  diff = model_builder_.Multiplication(diff, diff);
  diff = model_builder_.Multiplication(diff, mask);
  auto sum_vv = model_builder_.Reduce(diff, Reduce::Type::kSum, reduce_axes);
  if (feature_config_.is_causal) {
    sum_vv = model_builder_.Cumsum(sum_vv, Axis::kHeight);
    if (IsStreaming()) {
      TensorHandle sum_vv_update = UpdateCumsum(sum_vv, layer, 2);
      sum_vv = model_builder_.Add(sum_vv, stream_cumsum_vec[2]);
      stream_cumsum_out = model_builder_.Add(stream_cumsum_out, sum_vv_update);
      // Copy stream_cumsum_out to stream cumsum for next iter.
      OperationDef copy_def;
      copy_def.src_tensors.push_back(stream_cumsum_out.tensor_desc);
      copy_def.dst_tensors.push_back(stream_cumsum.tensor_desc);
      auto copy_op = std::make_unique<GPUOperation>(
          CreateElementwiseOneInput(gpu_info_, copy_def, OperationType::kCopy));
      model_builder_.AddGpuOperation(std::vector<ValueId>{stream_cumsum_out.id},
                                     std::vector<ValueId>{stream_cumsum.id},
                                     (std::move(copy_op)), "copy");
    }
  }
  auto variance =
      model_builder_.Elementwise(sum_vv, count_v, OperationType::kDiv);
  if (!feature_config_.is_causal) {
    masked_src = model_builder_.Reshape(
        masked_src,
        BHWC(src_shape.b, 1, src_shape.h * src_shape.w, src_shape.c));
  }
  auto normalized =
      model_builder_.AddTensor(masked_src.tensor_desc.GetBHWCShape(),
                               masked_src.tensor_desc.GetDataType());
  auto normalized_op = CreateNormalizeVar(
      gpu_info_, create_info_, masked_src.tensor_desc, mean.tensor_desc,
      variance.tensor_desc, normalized.tensor_desc, gamma, beta, epsilon);

  model_builder_.AddGpuOperation(
      {std::move(masked_src), std::move(mean), std::move(variance)}, normalized,
      std::make_unique<GPUOperation>(std::move(normalized_op)), "normalized");
  if (!feature_config_.is_causal) {
    normalized = model_builder_.Reshape(normalized, src_shape);
  }
  GpuModelBuilder::TensorHandle out =
      model_builder_.Multiplication(normalized, mask);
  if (use_f16) {
    out = model_builder_.Cast(out, DataType::kFloat16);
  }
  return out;
}

absl::StatusOr<TensorHandle> AudioEncoderBuilder::MakeMaskedGroupNorm(
    const TensorHandle& src, const TensorHandle& mask,
    const TensorHandle& stream_cumsum, int layer, const std::string& name) {
  ml_drift::Tensor<Linear, DataType::kFloat32> gamma;
  gamma.shape = Linear(src.tensor_desc.GetBHWCShape().c);
  gamma.data = tensor_loader_->LoadFloat32(name + ".gamma",
                                           src.tensor_desc.GetBHWCShape().c);
  if (!IsModelv3(config_)) {
    for (int i = 0; i < gamma.data.size(); ++i) {
      gamma.data[i] = 1.0f + gamma.data[i];
    }
  }
  // No beta tensor (bias tensor) for audio encoder. Suffices to set to 0
  if (feature_config_.norm_bias) {
    return absl::UnimplementedError("norm_bias not supported");
  }
  auto beta = MakeZeroTensor<Linear, DataType::kFloat32>(
      Linear(src.tensor_desc.GetBHWCShape().c));
  return HWCMaskedGroupNorm(src, mask, stream_cumsum, layer, /*epsilon*/ 1e-3f,
                            gamma, beta);
}

absl::StatusOr<std::vector<TensorHandle>>
AudioEncoderBuilder::MakeSubSampleConvProjection(
    const TensorHandle& feature, const TensorHandle& mask,
    const TensorHandle& segment_pos_handle, const TensorHandle& stream_cumsum,
    const std::string& name) {
  // INPUT: feature: [b, t, d]; mask: [b, t]
  // OUTPUT: feature: [b, t // x, model_dims]; mask: [b, t // x]

  std::vector<TensorHandle> conv2d_outputs = {feature, mask};
  for (int r = 0; r < feature_config_.convs.size(); ++r) {
    int channel_size = feature_config_.convs[r].channel_size;
    int prev_channel_size =
        r > 0 ? feature_config_.convs[r - 1].channel_size : 1;
    OHWI filter_shape =
        OHWI(channel_size, feature_config_.convs[r].kernel_size.h,
             feature_config_.convs[r].kernel_size.w, prev_channel_size);
    ABSL_ASSIGN_OR_RETURN(
        conv2d_outputs,
        MakeConv2DWithMask(conv2d_outputs[0], conv2d_outputs[1], filter_shape,
                           feature_config_.convs[r],
                           name + ".subsampling_" + std::to_string(r)));
    ABSL_ASSIGN_OR_RETURN(
        conv2d_outputs[0],
        MakeMaskedGroupNorm(conv2d_outputs[0], conv2d_outputs[1], stream_cumsum,
                            r, name + ".norm_" + std::to_string(r)));
    conv2d_outputs[0] = model_builder_.ReLU(conv2d_outputs[0], {});
  }
  const BHWC feature_shape = conv2d_outputs[0].tensor_desc.GetBHWCShape();
  conv2d_outputs[0] = model_builder_.Reshape(
      conv2d_outputs[0], BHWC(feature_shape.b, 1, feature_shape.h,
                              feature_shape.w * feature_shape.c));

  // Input proj.
  ABSL_ASSIGN_OR_RETURN(
      conv2d_outputs[0],
      MakeLinearAutoPrecision(conv2d_outputs[0], name + ".input_proj",
                              feature_config_.model_dims));
  if (feature_config_.proj_bias) {
    return absl::UnimplementedError("proj_bias not supported");
  }
  if (!config_.skip_absolute_positional_embeddings) {
    conv2d_outputs[0] = model_builder_.PositionalEmbedding(conv2d_outputs[0],
                                                           segment_pos_handle);
  }
  BHWC mask_shape = conv2d_outputs[1].tensor_desc.GetBHWCShape();
  conv2d_outputs[1] = model_builder_.Reshape(
      conv2d_outputs[1], BHWC(1, mask_shape.b, mask_shape.h, 1));
  return conv2d_outputs;
}

GPUOperation AudioEncoderBuilder::CreateReducerOp(const TensorDescriptor& src,
                                                  const TensorDescriptor& dst,
                                                  int take_every_n) {
  GPUOperation op;
  op.AddSrcTensor("src", src);
  op.AddDstTensor("dst", dst);
  op.args_.AddInt("take_every_n", take_every_n);
  op.code_ = R"(
  MAIN_FUNCTION($0) {
    int linear_xb = ucl::GetGlobalId<0>();
    int X = linear_xb / args.dst.Batch();
    int B = linear_xb % args.dst.Batch();
    args.dst.SetBatchRef(B);
    int Y = ucl::GetGlobalId<1>();
    int S = ucl::GetGlobalId<2>();
    if (X >= args.dst.Width() || Y >= args.dst.Height() || S >= args.dst.Slices()) {
      return;
    }
    args.dst::type result = args.src.Read(X * args.take_every_n, Y, S);
    args.dst.Write(result, X, Y, S);
  })";
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
  return op;
}

absl::StatusOr<GpuModelBuilder::TensorHandle>
AudioEncoderBuilder::MakeAudioFFLayerv3(
    const GpuModelBuilder::TensorHandle& src, const std::string& name) {
  GpuModelBuilder::TensorHandle t = src;
  // pre_ffw_norm has no scale shift.
  t = MakeRMSNormalization(t, name + ".pre_layer_norm", /*scale_shift=*/0.0f);

  // ffn_layer1
  ABSL_ASSIGN_OR_RETURN(auto gate_value,
                        MakeLinearAutoPrecision(t, name + ".ffn_layer1",
                                                config_.hidden_dimension));

  gate_value = model_builder_.SiLU(CapInfinity(gate_value));

  // ffn_layer2
  ABSL_ASSIGN_OR_RETURN(
      t, MakeLinearAutoPrecision(gate_value, name + ".ffn_layer2",
                                 config_.model_dimension));
  // post_layer_norm
  t = MakeRMSNormalization(CapInfinity(t), name + ".post_layer_norm",
                           /*scale_shift=*/0.0f);

  // Apply out scale
  t = model_builder_.Multiplication(t, 0.5f);

  return CapInfinity(model_builder_.Add(t, src));
}

absl::StatusOr<GpuModelBuilder::TensorHandle>
AudioEncoderBuilder::MakeLightConv1d(
    const GpuModelBuilder::TensorHandle& input,
    const GpuModelBuilder::TensorHandle& stream_depthwise,
    const std::string& name) {
  GpuModelBuilder::TensorHandle working =
      MakeRMSNormalization(input, name + ".ln", /*scale_shift=*/0.0f);
  // Apply linear start dense layer
  ABSL_ASSIGN_OR_RETURN(working,
                        MakeLinearAutoPrecision(working, name + ".linear_start",
                                                config_.model_dimension * 2));
  // GLU
  std::vector<TensorHandle> glu_split = model_builder_.Split(
      working, Axis::kChannels, working.tensor_desc.GetBHWCShape().c / 2);
  glu_split[1] =
      model_builder_.Elementwise(glu_split[1], OperationType::kSigmoid);
  working = model_builder_.Multiplication(glu_split[0], glu_split[1]);

  // 1d depthwise convolution
  DepthwiseConvolution2DAttributes dw_attr;

  dw_attr.strides = HW(1, 1);
  dw_attr.padding.prepended = HW(0, config_.light_conv_kernel_size - 1);
  dw_attr.padding.appended = HW(0, 0);
  auto& dw_weights =
      dw_attr.weights.emplace<Tensor<OHWI, DataType::kFloat32>>();
  dw_weights.shape =
      OHWI(1, 1, config_.light_conv_kernel_size, config_.model_dimension);
  dw_weights.data = tensor_loader_->LoadFloat32(
      name + ".depthwise_conv1d.w", dw_weights.shape.DimensionsProduct());
  dw_weights.data.resize(dw_weights.shape.DimensionsProduct() +
                         kWeightRearrangeExtraBytes / sizeof(float));
  // No bias
  dw_attr.bias.data.resize(0);
  if (IsStreaming()) {
    // If streaming, we need to prepend src with values from the previous frame.
    dw_attr.padding.prepended = HW(0, 0);
    dw_attr.padding.appended = HW(0, 0);
    // stream depthwise has size (1, 1, light_conv_kernel_size - 1, model_dim)
    working = model_builder_.Concat({stream_depthwise, working}, Axis::kWidth);
    // Grab last 4 indices of output
    ABSL_ASSIGN_OR_RETURN(TensorHandle last_four, GetLastN(working, 4));
    // copy last four into stream_depthwise
    OperationDef copy_def;
    copy_def.src_tensors.push_back(last_four.tensor_desc);
    copy_def.dst_tensors.push_back(stream_depthwise.tensor_desc);
    auto copy_op = std::make_unique<GPUOperation>(
        CreateElementwiseOneInput(gpu_info_, copy_def, OperationType::kCopy));
    model_builder_.AddGpuOperation(std::vector<ValueId>{last_four.id},
                                   std::vector<ValueId>{stream_depthwise.id},
                                   (std::move(copy_op)), "copy");
  } else {
    // If not streaming, we prepend with 0s.
    dw_attr.padding.prepended = HW(0, config_.light_conv_kernel_size - 1);
    dw_attr.padding.appended = HW(0, 0);
  }
  TensorHandle output = model_builder_.DepthwiseConvolution(working, dw_attr);

  // RMS norm
  output =
      MakeRMSNormalization(output, name + ".conv_norm", /*scale_shift=*/0.0f);
  // conv activation
  output = model_builder_.SiLU(output);

  // Final dense layer
  ABSL_ASSIGN_OR_RETURN(output,
                        MakeLinearAutoPrecision(output, name + ".linear_end",
                                                config_.model_dimension));
  return model_builder_.Add(output, input);
}

absl::StatusOr<GpuModelBuilder::TensorHandle>
AudioEncoderBuilder::MakeSoftmaxMatMul(
    const GpuModelBuilder::TensorHandle& logits,
    const GpuModelBuilder::TensorHandle& v) {
  // att has sh: (HC, num_blocks, horizon, context_size)
  // v has sh: (1, HC, num_blocks * context_size, HS)
  // Want the output to have sh: (HC, horizon, num_blocks, HS)
  // Therefore, we do the following:
  // BMM([1, HC * num_blocks, horizon, context_size],
  // [1, HC * num_blocks, context_size, HS]) = [1, HC * num_blocks, horizon, HS]
  const BHWC att_sh = logits.tensor_desc.GetBHWCShape();
  const int hc = att_sh.b;
  const int num_blocks = att_sh.h;
  const int horizon = att_sh.w;
  const int context_size = att_sh.c;
  const int hs = v.tensor_desc.GetBHWCShape().c;
  TensorHandle att = model_builder_.Reshape(
      logits, BHWC(1, hc * num_blocks, horizon, context_size));
  TensorHandle v_prepped =
      model_builder_.Reshape(v, BHWC(1, hc * num_blocks, context_size, hs));

  GpuModelBuilder::TensorHandle att_exp;
  GpuModelBuilder::TensorHandle* att_exp_ptr = nullptr;
  if (model_builder_.UseFusedSoftmaxWithBatchedMatMul(att)) {
    att_exp_ptr = &att_exp;
    att_exp = model_builder_.SoftmaxReduce(att);
  } else {
    att = model_builder_.Softmax(att, {}, nullptr);
  }

  ABSL_ASSIGN_OR_RETURN(auto probabilities, model_builder_.BatchedMatMul(
                                                att, v_prepped, /*attr=*/{},
                                                /*src_exp=*/att_exp_ptr));
  return model_builder_.Reshape(probabilities,
                                BHWC(1, hc, num_blocks * horizon, hs));
}

TensorFloat32 AudioEncoderBuilder::GetTimingSignal1dPos(int min_timescale,
                                                        int max_timescale,
                                                        int past_horizon,
                                                        int future_horizon) {
  const int channels = config_.model_dimension;
  const int num_timescales = std::floor(channels / 2);
  const float log_timescale_increment =
      std::log(static_cast<float>(max_timescale) /
               static_cast<float>(min_timescale)) /
      std::max(num_timescales, 1);
  std::vector<float> inv_timescales(num_timescales);
  for (int i = 0; i < num_timescales; ++i) {
    inv_timescales[i] = min_timescale * std::exp(static_cast<float>(i) *
                                                 -log_timescale_increment);
  }
  const int context_size = past_horizon + future_horizon + 1;

  TensorFloat32 timing_signal;
  timing_signal.shape = BHWC(config_.batch_size, context_size, 1, channels);
  timing_signal.data =
      std::vector<float>(config_.batch_size * context_size * channels);
  for (int b = 0; b < config_.batch_size; ++b) {
    for (int c = 0; c < context_size; ++c) {
      for (int ch = 0; ch < channels; ++ch) {
        if (ch < num_timescales) {
          timing_signal.data[b * context_size * channels + c * channels + ch] =
              std::sin((context_size - c - 1) * inv_timescales[ch]);
        } else if (ch < num_timescales * 2) {
          timing_signal.data[b * context_size * channels + c * channels + ch] =
              std::cos((context_size - c - 1) *
                       inv_timescales[ch - num_timescales]);
        } else {
          timing_signal.data[b * context_size * channels + c * channels + ch] =
              0.0f;
        }
      }
    }
  }
  return timing_signal;
}

absl::StatusOr<GpuModelBuilder::TensorHandle>
AudioEncoderBuilder::MakeRelativeEmbedding(
    const GpuModelBuilder::TensorHandle& q,
    const GpuModelBuilder::TensorHandle& k,
    const GpuModelBuilder::TensorHandle& signal, const std::string& name) {
  if (config_.number_of_heads != config_.number_of_kv_heads) {
    return absl::InvalidArgumentError(
        "Number of heads must be equal to number of kv heads.");
  }
  // Q @ K
  const BHWC q_sh =
      q.tensor_desc.GetBHWCShape();  // (B * HC, 1, num_blocks * block_size, HS)
  const BHWC k_sh =
      k.tensor_desc.GetBHWCShape();  // (B * KVHC, 1, past_horizon + (num_blocks
                                     // * block_size) + future_horizon, HS)
  const int past_horizon =
      config_.relative_positional_embedding_config->past_horizon;
  const int future_horizon =
      config_.relative_positional_embedding_config->future_horizon;
  // See comment at the top of this file for why we use kConformerBlockSize.
  const int block_size = kConformerBlockSize;
  const int num_blocks = q_sh.w / block_size;
  const int context_size = past_horizon + future_horizon + block_size;
  const int hc = q_sh.b;
  const int hs = q_sh.c;
  // q first -> (1, B * HC * num_blocks, block_size, HS)
  TensorHandle q_prepped =
      model_builder_.Reshape(q, BHWC(1, hc * num_blocks, block_size, hs));

  // k next (1, B * KVHC * num_blocks, HS, context_size)
  TensorHandle k_prepped = model_builder_.Reshape(
      k, BHWC(1, k_sh.b * num_blocks, context_size, k_sh.c));
  k_prepped = model_builder_.Transpose(k_prepped, BHWC(0, 1, 3, 2));

  // ac: (1, B * HC/KVHC * num_blocks, block_size, context_size)
  ABSL_ASSIGN_OR_RETURN(TensorHandle ac,
                        model_builder_.BatchedMatMul(q_prepped, k_prepped));

  // sin_emb: (B, horizon + 1, 1, HC * HS)
  ABSL_ASSIGN_OR_RETURN(auto sin_emb,
                        MakeLinearAutoPrecision(
                            signal, name + ".pos_proj",
                            config_.number_of_heads * config_.head_dimension));

  // We need to prep for Q@sin_emb.
  // Q: (1, B * HC, num_blocks * block_size, HS)
  // sin_emb: (1, B*HC, HS, horizon + 1)
  // Q@sin_emb: (1, B*HC, num_blocks * block_size, horizon + 1)
  q_prepped = model_builder_.Reshape(q_prepped,
                                     BHWC(1, hc, num_blocks * block_size, hs));
  sin_emb = model_builder_.Reshape(
      sin_emb, BHWC(1, past_horizon + future_horizon + 1,
                    config_.number_of_heads, config_.head_dimension));
  sin_emb = model_builder_.Transpose(sin_emb, BHWC(0, 2, 3, 1));
  ABSL_ASSIGN_OR_RETURN(TensorHandle bd,
                        model_builder_.BatchedMatMul(q_prepped, sin_emb));
  // Finally, add skew to the bd term
  bd = model_builder_.Reshape(bd, BHWC(1, hc * num_blocks, block_size,
                                       past_horizon + future_horizon + 1));
  PadAttributes pad_attr;
  pad_attr.prepended = BHWC(0, 0, 0, 0);
  pad_attr.appended =
      BHWC(0, 0, 0, context_size + 1 - (past_horizon + future_horizon + 1));
  bd = model_builder_.Padding(bd, pad_attr);
  bd = model_builder_.Reshape(
      bd, BHWC(1, hc * num_blocks, block_size * (context_size + 1), 1));
  const int split_size = block_size * context_size;
  ABSL_ASSIGN_OR_RETURN(bd, GetFirstN(bd, split_size));
  bd = model_builder_.Reshape(
      bd, BHWC(1, hc * num_blocks, block_size, context_size));
  TensorHandle out = model_builder_.Add(ac, bd);
  // Finally reshape
  return model_builder_.Reshape(out,
                                BHWC(hc, num_blocks, block_size, context_size));
}

// Extract the spatial dimension into num_blocks * block_size chunks along
// the width axis.
absl::StatusOr<TensorHandle> AudioEncoderBuilder::ExtractBlockContextNoPad(
    const TensorHandle& src) {
  const int past_horizon =
      config_.relative_positional_embedding_config->past_horizon;
  const int future_horizon =
      config_.relative_positional_embedding_config->future_horizon;
  // See comment at the top of this file for why we use kConformerBlockSize.
  const int block_size = kConformerBlockSize;
  // Calculate tf.signal.frame.
  const int frame_length = block_size + past_horizon + future_horizon;
  const int frame_step = block_size;
  std::vector<int> indices;
  int index = 0;
  while (index + frame_length < src.tensor_desc.GetBHWCShape().w) {
    for (int i = 0; i < frame_length; ++i) {
      indices.push_back(index);
      ++index;
    }
    index += (frame_step - frame_length);
  }
  // Gather over the width axis on indices above.
  return Gather(src, indices);
}

// Same as ExtractBlockContextNoPad, but we prepend with zeros for the
// past_horizon. This is used for static mode when we always have zeroes in
// the past.
absl::StatusOr<TensorHandle> AudioEncoderBuilder::ExtractBlockContext(
    const TensorHandle& src) {
  const int past_horizon =
      config_.relative_positional_embedding_config->past_horizon;
  const int future_horizon =
      config_.relative_positional_embedding_config->future_horizon;
  // See comment at the top of this file for why we use kConformerBlockSize.
  const int block_size = kConformerBlockSize;
  PadAttributes pad_attr;
  pad_attr.prepended = BHWC(0, 0, past_horizon, 0);
  pad_attr.appended = BHWC(0, 0, future_horizon + block_size, 0);
  TensorHandle padded = model_builder_.Padding(src, pad_attr);
  return ExtractBlockContextNoPad(padded);
}

// src - bhwc(batch_size, 1, sequence_size, model_dimension)
absl::StatusOr<GpuModelBuilder::TensorHandle>
AudioEncoderBuilder::MakeConformerAttention(
    const GpuModelBuilder::TensorHandle& src,
    const GpuModelBuilder::TensorHandle& attention_mask,
    const GpuModelBuilder::TensorHandle& segment_pos,
    const GpuModelBuilder::TensorHandle& key_tensor,
    const GpuModelBuilder::TensorHandle& value_tensor,
    const GpuModelBuilder::TensorHandle& signal, const std::string& name) {
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
  const int head_size = config_.head_dimension;
  const int batch_size = src_sh.b;
  const int sequence_size = src_sh.w;
  const int past_horizon =
      config_.relative_positional_embedding_config->past_horizon;
  // See comment at the top of this file for why we use kConformerBlockSize.
  const int block_size = kConformerBlockSize;

  // Extract QKV from src.
  ABSL_ASSIGN_OR_RETURN(auto qkv_arr, MakeGetQKV(src, name));
  auto q = qkv_arr[0];        // bhwc(B, 1, S, HC*HS)
  auto k_block = qkv_arr[1];  // bhwc(B, 1, S, KVHC*HS)
  auto v_block = qkv_arr[2];  // bhwc(B, 1, S, KVHC*HS)
  // Transform QKV for efficient calcs
  MakeTransformQKV(model_builder_, &q, &k_block, &v_block);
  // Scale Q
  ABSL_ASSIGN_OR_RETURN(q, PerDimScale(model_builder_, q, name));

  TensorHandle k = k_block;
  TensorHandle v = v_block;
  if (IsStreaming()) {
    // kv cache has len past_horizon. k, v have len block_size.
    k = model_builder_.Concat({key_tensor, k_block}, Axis::kWidth);
    v = model_builder_.Concat({value_tensor, v_block}, Axis::kWidth);

    // move last past_horizon elements of k, v to k_block/v_block to be
    // copied into the kv cache.
    if (past_horizon != block_size) {
      // If past_horizon == block_size, then key_block already contains
      // the last past_horizon elements.
      ABSL_ASSIGN_OR_RETURN(k_block, GetLastN(k, past_horizon));
      ABSL_ASSIGN_OR_RETURN(v_block, GetLastN(v, past_horizon));
    }

    // copy to kv cache
    OperationDef k_def;
    k_def.src_tensors.push_back(k_block.tensor_desc);
    k_def.dst_tensors.push_back(key_tensor.tensor_desc);
    auto k_copy_op = std::make_unique<GPUOperation>(
        CreateElementwiseOneInput(gpu_info_, k_def, OperationType::kCopy));
    model_builder_.AddGpuOperation(std::vector<ValueId>{k_block.id},
                                   std::vector<ValueId>{key_tensor.id},
                                   (std::move(k_copy_op)), "copy");
    OperationDef v_def;
    v_def.src_tensors.push_back(v_block.tensor_desc);
    v_def.dst_tensors.push_back(value_tensor.tensor_desc);
    auto v_copy_op = std::make_unique<GPUOperation>(
        CreateElementwiseOneInput(gpu_info_, v_def, OperationType::kCopy));
    model_builder_.AddGpuOperation(std::vector<ValueId>{v_block.id},
                                   std::vector<ValueId>{value_tensor.id},
                                   (std::move(v_copy_op)), "copy");
  }

  // Combine B & (HC / KVHC) dims
  const BHWC q_sh = q.tensor_desc.GetBHWCShape();
  const BHWC k_sh = k.tensor_desc.GetBHWCShape();
  // (B, HC, S, HS) ->(B * HC, 1, S, HS)
  q = model_builder_.Reshape(q, BHWC(q_sh.b * q_sh.h, 1, q_sh.w, q_sh.c));
  // (B, KVHC, S, HS) ->(B * KVHC, 1, S, HS)
  k = model_builder_.Reshape(k, BHWC(k_sh.b * k_sh.h, 1, k_sh.w, k_sh.c));

  // Pad q, k, v for block size
  // Q first. Pad to (B * HC, 1, num_blocks * horizon, HS)
  const int t = q.tensor_desc.GetBHWCShape().w;
  const int num_blocks = std::floor((t + block_size - 1) / block_size);
  if (t < block_size) {
    return absl::InvalidArgumentError(
        "Seq length is less than block size. Check tokens_per_chunk and "
        "block_size.");
  }
  const int pad_length = num_blocks * block_size - t;
  PadAttributes pad_attr;
  pad_attr.prepended = BHWC(0, 0, 0, 0);
  pad_attr.appended = BHWC(0, 0, pad_length, 0);
  pad_attr.constant_values = 0.0f;
  q = model_builder_.Padding(q, pad_attr);
  // K, V: (B * KVHC, 1, S, HS) -> (B * KVHC, 1, num_blocks * context_size, HS)
  if (!IsStreaming()) {
    ABSL_ASSIGN_OR_RETURN(k, ExtractBlockContext(k));
    ABSL_ASSIGN_OR_RETURN(v, ExtractBlockContext(v));
  }
  // We need to ExtractBlockContext whenever we have num_blocks > 1
  if (IsStreaming() && num_blocks > 1) {
    // Unlike static mode, we actually will have values for the past_horizon,
    // and so we use real values there instead of padding with zeros.
    // We still need to make sure we pad the future_horizon and block_size.
    const int future_horizon =
        config_.relative_positional_embedding_config->future_horizon;
    PadAttributes pad_attr;
    pad_attr.prepended = BHWC(0, 0, 0, 0);
    pad_attr.appended = BHWC(0, 0, future_horizon + block_size, 0);
    TensorHandle k_padded = model_builder_.Padding(k, pad_attr);
    TensorHandle v_padded = model_builder_.Padding(v, pad_attr);
    ABSL_ASSIGN_OR_RETURN(k, ExtractBlockContextNoPad(k_padded));
    ABSL_ASSIGN_OR_RETURN(v, ExtractBlockContextNoPad(v_padded));
  }

  // Get logits with Rotary Embedding
  ABSL_ASSIGN_OR_RETURN(
      GpuModelBuilder::TensorHandle logits,
      MakeRelativeEmbedding(q, k, signal,
                            name + ".relative_position_embedding"));

  if (config_.attn_soft_cap_value > 0.0f) {  // cap_logits
    const float cap_value = config_.attn_soft_cap_value;
    logits = model_builder_.Multiplication(logits, 1.0f / cap_value);
    logits = model_builder_.Elementwise(logits, OperationType::kTanh);
    logits = model_builder_.Multiplication(logits, cap_value);
  }
  logits = model_builder_.Add(logits, attention_mask);

  ABSL_ASSIGN_OR_RETURN(auto att, MakeSoftmaxMatMul(logits, v));
  // att - (B, hc, num_blocks * horizon, hs)
  if (!IsStreaming()) {  // no split for streaming
    ABSL_ASSIGN_OR_RETURN(att, GetFirstN(att, sequence_size));
  }

  att = model_builder_.Transpose(att, BHWC(0, 2, 1, 3));
  att = model_builder_.Reshape(
      att, BHWC{batch_size, 1, sequence_size, heads_count * head_size});
  return MakeConvAutoPrecision(att, name + ".post", Layout::kOIHW,
                               config_.model_dimension);
}

absl::StatusOr<GpuModelBuilder::TensorHandle>
AudioEncoderBuilder::MakeConformerLayer(
    const GpuModelBuilder::TensorHandle& inputs,
    const GpuModelBuilder::TensorHandle& attention_mask,
    const GpuModelBuilder::TensorHandle& segment_pos,
    const GpuModelBuilder::TensorHandle& key_tensor,
    const GpuModelBuilder::TensorHandle& value_tensor,
    const GpuModelBuilder::TensorHandle& params_tensor,
    const GpuModelBuilder::TensorHandle& signal,
    const GpuModelBuilder::TensorHandle& stream_depthwise,
    const std::string& name) {
  // FF (start)
  ABSL_ASSIGN_OR_RETURN(auto ff_start,
                        MakeAudioFFLayerv3(inputs, name + ".ff_layer_start"));
  // pre norm
  auto ff_normalized =
      MakeRMSNormalization(ff_start, name + ".pre_norm", /*scale_shift=*/0.0f);
  // self attention
  ABSL_ASSIGN_OR_RETURN(
      auto atten_output,
      MakeConformerAttention(ff_normalized, attention_mask, segment_pos,
                             key_tensor, value_tensor, signal,
                             name + ".self_attention"));
  atten_output = MakeRMSNormalization(atten_output, name + ".post_norm",
                                      /*scale_shift=*/0.0f);
  atten_output = model_builder_.Add(atten_output, ff_start);

  // light conv 1d
  ABSL_ASSIGN_OR_RETURN(
      atten_output,
      MakeLightConv1d(atten_output, stream_depthwise, name + ".lconv"));

  // FF (end)
  ABSL_ASSIGN_OR_RETURN(
      atten_output, MakeAudioFFLayerv3(atten_output, name + ".ff_layer_end"));
  return MakeRMSNormalization(atten_output, name + ".final_ln",
                              /*scale_shift=*/0.0f);
}

absl::StatusOr<TensorHandle> AudioEncoderBuilder::MakeAttentionMaskv2(
    const TensorHandle& mask, const TensorHandle& params_i32_handle) {
  ABSL_ASSIGN_OR_RETURN(auto attn_mask,
                        MakeFillAttentionMask(params_i32_handle, nullptr,
                                              /*is_local=*/false));
  // attn_mask.shape = BHWC(1, 1, seq_size, cache_size)
  // For each element of the sequence, we want to ignore all elements of the
  // cache that are masked out by mask passed from the caller (conv_out[1]).
  // If we need more performance, this should be included w/i
  // MakeFillAttentionMask.
  const BHWC mask_sh = mask.tensor_desc.GetBHWCShape();
  // Pad audio mask of size (1, 1, seq_len, 1) to (1, 1, 1, cache_size)
  TensorHandle mask_tiled =
      model_builder_.Reshape(mask, BHWC(1, 1, 1, mask_sh.w));
  if (config_.cache_size > config_.sequence_size) {
    PadAttributes pad_attr;
    pad_attr.prepended = BHWC(0, 0, 0, 0);
    pad_attr.appended =
        BHWC(0, 0, 0, config_.cache_size - config_.sequence_size);
    pad_attr.constant_values = 1.0f;
    mask_tiled = model_builder_.Padding(mask_tiled, pad_attr);
  }
  mask_tiled = model_builder_.Cast(mask_tiled, DataType::kBool);
  // mask_tiled[i] == 1 means pay attn, mask_tiled[i] == 0 means ignore
  // attn_mask[i] == 0 means pay attn, attn_mask[i] == 1 means ignore
  // want mask == 0 means pay attn, mask == 1 means ignore
  mask_tiled =
      model_builder_.Elementwise(mask_tiled, OperationType::kLogicalNot);
  return model_builder_.Elementwise(attn_mask, mask_tiled,
                                    OperationType::kLogicalOr);
}

absl::StatusOr<TensorHandle> AudioEncoderBuilder::ExtractMaskStreaming(
    const TensorHandle& src, const TensorHandle& params_i32_handle) {
  const int horizon =
      config_.relative_positional_embedding_config->past_horizon +
      config_.relative_positional_embedding_config->future_horizon;
  // See comment at the top of this file for why we use kConformerBlockSize.
  const int block_size = kConformerBlockSize;
  TensorHandle dst = model_builder_.AddTensor(
      BHWC(1, 1, horizon + block_size, 1), src.tensor_desc.GetDataType());
  GPUOperation op;
  op.AddSrcTensor("src", src.tensor_desc);
  ml_drift::BufferDescriptor params_buffer;
  params_buffer.element_type = DataType::kInt32;
  params_buffer.element_size = 1;
  op.AddSrcBuffer("params", params_buffer);
  op.args_.AddInt("past_horizon",
                  config_.relative_positional_embedding_config->past_horizon);
  op.AddDstTensor("dst", dst.tensor_desc);
  op.code_ = absl::Substitute(R"(
    MAIN_FUNCTION($$0) {
      int linear_xb = ucl::GetGlobalId<0>();
      int X = linear_xb / args.dst.Batch();
      int B = linear_xb % args.dst.Batch();
      args.dst.SetBatchRef(B);
      int Y = ucl::GetGlobalId<1>();
      int S = ucl::GetGlobalId<2>();
      if (X >= args.dst.Width() || Y >= args.dst.Height() || S >= args.dst.Slices()) {
        return;
      }
      int token_index_offset = args.params.Read($0);
      args.dst::type result;
      if (X < args.past_horizon) {
        args.dst::scalar_type write_val = 1.0f;
        if (token_index_offset == 0) {
          write_val = 0.0f;
        }
        result = ucl::Init<args.dst::type>(write_val);
      } else {
        result = args.src.Read(X - args.past_horizon, Y, S);
      }
      args.dst.Write(result, X, Y, S);
    })",
                              LlmRuntimeParams::kTokenOffsetIndex);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
  model_builder_.AddGpuOperation({src, params_i32_handle}, {dst},
                                 std::make_unique<GPUOperation>(std::move(op)),
                                 "extract_mask_streaming");
  return dst;
}

// Want output to be (1, num_blocks, horizon, context_size)
absl::StatusOr<TensorHandle> AudioEncoderBuilder::MakeAttentionMaskv3(
    const TensorHandle& mask, const TensorHandle& params_i32_handle) {
  const int horizon =
      config_.relative_positional_embedding_config->past_horizon +
      config_.relative_positional_embedding_config->future_horizon;
  // See comment at the top of this file for why we use kConformerBlockSize.
  const int block_size = kConformerBlockSize;
  const int context_size = horizon + block_size;
  // (1, 1, num_blocks * context_size, 1)
  TensorHandle valid_mask_blocked;
  if (!IsStreaming()) {
    ABSL_ASSIGN_OR_RETURN(valid_mask_blocked, ExtractBlockContext(mask));
  } else {
    ABSL_ASSIGN_OR_RETURN(valid_mask_blocked,
                          ExtractMaskStreaming(mask, params_i32_handle));
  }
  const int num_blocks =
      valid_mask_blocked.tensor_desc.GetBHWCShape().w / context_size;
  valid_mask_blocked = model_builder_.Reshape(
      valid_mask_blocked, BHWC(1, num_blocks, 1, context_size));
  // Tile WIDTH dim
  valid_mask_blocked = model_builder_.Tile(
      valid_mask_blocked, BHWC(1, num_blocks, horizon, context_size));
  valid_mask_blocked = model_builder_.Cast(valid_mask_blocked, DataType::kBool);
  // Now make local causal valid mask.
  // Should be sh (1, num_blocks, horizon,context_size)
  // ex: max_past_horizon = 4, max_future_horizon = 2, block_size = 3
  // Input:
  // 0 1 2 3 a b c 4 5
  // Causal mask:
  // 1 1 1 1 a 1 1 0 0
  // 0 1 1 1 1 b 1 1 0
  // 0 0 1 1 1 1 c 1 1
  // Note that we tile num_blocks times over H so that we can add to
  // valid_mask_blocked
  // Calculate value on cpu.
  std::vector<int8_t> local_mask_data(num_blocks * horizon * context_size);
  for (int i = 0; i < horizon; ++i) {
    for (int j = 0; j < context_size; ++j) {
      if (j >= i && j <= i + horizon) {
        local_mask_data[i * context_size + j] = 1;
      } else {
        local_mask_data[i * context_size + j] = 0;
      }
    }
  }
  // Tile these values into the rest of the blocks.
  for (int i = 1; i < num_blocks; ++i) {
    std::copy(local_mask_data.begin(),
              local_mask_data.begin() + horizon * context_size,
              local_mask_data.begin() + i * horizon * context_size);
  }
  TensorDescriptor indices_desc = TensorDescriptor(
      DataType::kBool, mask.tensor_desc.GetStorageType(), Layout::kHWC);
  indices_desc.SetBHWCShape(BHWC(1, num_blocks, horizon, context_size));
  indices_desc.UploadData(local_mask_data.data());

  TensorHandle local_causal_valid_mask =
      model_builder_.AddConstantTensor(std::move(indices_desc));

  // The order of the operands here matters, the other order triggers
  // b/427757142 on Intel Macs.
  valid_mask_blocked = model_builder_.Elementwise(
      local_causal_valid_mask, valid_mask_blocked, OperationType::kLogicalAnd);
  // Finally need to prep mask by making False values -inf, and True values 0.
  // Aka, apply not & multiply
  valid_mask_blocked = model_builder_.Elementwise(valid_mask_blocked,
                                                  OperationType::kLogicalNot);
  valid_mask_blocked =
      model_builder_.Cast(valid_mask_blocked, mask.tensor_desc.GetDataType());
  float min_float = mask.tensor_desc.GetDataType() == DataType::kFloat32
                        ? std::numeric_limits<float>::max()
                        : 65504;
  min_float *= -0.7f;
  return model_builder_.Multiplication(valid_mask_blocked, min_float);
}

absl::StatusOr<TensorHandle> AudioEncoderBuilder::MakeEncoderLayer(
    const TensorHandle& feature_handle, const TensorHandle& mask_handle,
    std::vector<GpuModelBuilder::TensorHandle>* key_cached_tensors,
    std::vector<GpuModelBuilder::TensorHandle>* value_cached_tensors,
    const TensorHandle& params_i32_handle, const TensorHandle& stream_cumsum,
    std::vector<GpuModelBuilder::TensorHandle>* stream_depthwise_tensors,
    TensorHandle* signal, const std::string& name) {
  std::vector<TensorHandle> conv_out;
  TensorHandle segment_pos = MakeFillSegmentPos(params_i32_handle);
  ABSL_ASSIGN_OR_RETURN(conv_out, MakeSubSampleConvProjection(
                                      feature_handle, mask_handle, segment_pos,
                                      stream_cumsum, name + ".feature"));
  // Adjust segment_pos with audio mask
  segment_pos =
      model_builder_.Cast(segment_pos, conv_out[1].tensor_desc.GetDataType());
  segment_pos = model_builder_.Multiplication(segment_pos, conv_out[1]);
  // Make attn mask
  TensorHandle attn_mask;
  if (IsModelv3(config_)) {
    ABSL_ASSIGN_OR_RETURN(attn_mask,
                          MakeAttentionMaskv3(conv_out[1], params_i32_handle));
  } else {
    ABSL_ASSIGN_OR_RETURN(attn_mask,
                          MakeAttentionMaskv2(conv_out[1], params_i32_handle));
  }
  for (int i = 0; i < config_.stack_size; ++i) {
    if (IsModelv3(config_)) {
      ABSL_ASSIGN_OR_RETURN(
          conv_out[0],
          MakeConformerLayer(
              conv_out[0], attn_mask, segment_pos, key_cached_tensors->at(i),
              value_cached_tensors->at(i), params_i32_handle, *signal,
              stream_depthwise_tensors->at(i),
              "params.audio.transformer.x_layers_" + std::to_string(i)));
    } else {
      ABSL_ASSIGN_OR_RETURN(
          conv_out[0],
          MakeLayer(conv_out[0], attn_mask, segment_pos,
                    key_cached_tensors->at(i), value_cached_tensors->at(i),
                    params_i32_handle, /*is_global=*/true,
                    "params.audio.transformer.x_layers_" + std::to_string(i)));
    }
  }
  TensorHandle dst;
  if (feature_config_.take_every_n > 1) {
    BHWC src_sh = conv_out[0].tensor_desc.GetBHWCShape();
    src_sh.w /= feature_config_.take_every_n;
    dst =
        model_builder_.AddTensor(src_sh, conv_out[0].tensor_desc.GetDataType());
    // Can (should) be replaced by a gather op
    auto reduce_op = std::make_unique<GPUOperation>(
        CreateReducerOp(conv_out[0].tensor_desc, dst.tensor_desc,
                        feature_config_.take_every_n));
    model_builder_.AddGpuOperation(
        {conv_out[0]}, {dst}, std::move(reduce_op),
        "reduce_every_" + std::to_string(feature_config_.take_every_n));
  } else {
    dst = conv_out[0];
  }
  return dst;
}

absl::StatusOr<TensorHandle> AudioEncoderBuilder::MakeMLPAdapterLayer(
    const TensorHandle& feature_handle, const std::string& name) {
  auto working_feature = feature_handle;
  // Layer Norm
  if (adapter_config_.initial_layer_norm) {
    ml_drift::Tensor<Linear, DataType::kFloat32> gamma;
    gamma.shape = Linear(working_feature.tensor_desc.GetBHWCShape().c);
    gamma.data = tensor_loader_->LoadFloat32(
        name + ".layer_norm.scale",
        working_feature.tensor_desc.GetBHWCShape().c);
    auto beta = MakeZeroTensor<Linear, DataType::kFloat32>(
        Linear(working_feature.tensor_desc.GetBHWCShape().c));
    working_feature =
        model_builder_.LayerNormalization(working_feature, gamma, beta, 1e-6f);
  }

  // FC Layers
  for (int layer = 0; layer < adapter_config_.num_layers; ++layer) {
    ABSL_ASSIGN_OR_RETURN(
        working_feature,
        MakeLinearAutoPrecision(working_feature,
                                name + ".linear_" + std::to_string(layer),
                                adapter_config_.output_dim));
    if (layer < adapter_config_.num_layers - 1) {
      working_feature = model_builder_.ReLU(working_feature, {});
    }
  }
  return working_feature;
}

absl::StatusOr<TensorHandle> AudioEncoderBuilder::MakeOutputProjection(
    const TensorHandle& feature_handle, const std::string& name) {
  return MakeLinearAutoPrecision(feature_handle, name + ".output_projection",
                                 usm_projection_config_.output_dim);
}

absl::StatusOr<TensorHandle> AudioEncoderBuilder::MakeProjectionV3(
    const TensorHandle& feature_handle, const std::string& name) {
  TensorHandle soft_tokens;
  if (config_.audio_dual_norm_soft_and_hard_tokens) {
    soft_tokens = MakeRMSNormalization(feature_handle,
                                       name + ".audio_soft_embedding_norm",
                                       /*scale_shift=*/0.0f);
  } else {
    soft_tokens = model_builder_.RMSNormalization(feature_handle, 1e-30f);
  }
  // Streaming loading cannot easily load the same tensor twice, so as a
  // quick workaround, the web model has an extra copy of this one tensor, which
  // we load instead. TODO: b/431813047 - Fix.
#ifdef __EMSCRIPTEN__
  const std::string audio_input_proj_name = ".audio_input_projection_copy";
#else
  const std::string audio_input_proj_name = ".audio_input_projection";
#endif  // __EMSCRIPTEN__
  // Uncomment this to generate model with audio_input_projection_copy tensor.
  // A change must also be uncommented in the roc_converter
  // ABSL_ASSIGN_OR_RETURN(
  // auto tmp, MakeLinearAutoPrecision(soft_tokens,
  //                                   name + ".audio_input_projection_copy",
  //                                   projection_config_.output_dim));
  ABSL_ASSIGN_OR_RETURN(
      soft_tokens,
      MakeLinearAutoPrecision(soft_tokens, name + audio_input_proj_name,
                              projection_config_.output_dim));
  if (config_.audio_dual_norm_soft_and_hard_tokens) {
    return model_builder_.RMSNormalization(soft_tokens, 1e-6f);
  } else {
    return soft_tokens;
  }
}

bool AudioEncoderBuilder::IsStreaming() const {
  return feature_config_.pad_for_streaming > 0;
}

TensorDescriptor AudioEncoderBuilder::GetFeatureDescriptor() const {
  TensorDescriptor td =
      TensorDescriptor{DeduceDataTypeFromPrecision(create_info_.precision),
                       TensorStorageType::kBuffer, Layout::kHWC};
  td.SetBHWCShape(BHWC(config_.batch_size,
                       config_.sequence_size * feature_config_.shrink_factor +
                           feature_config_.pad_for_streaming,
                       feature_config_.input_dims, 1));
  return td;
}

TensorDescriptor AudioEncoderBuilder::GetMaskDescriptor() const {
  TensorDescriptor td =
      TensorDescriptor{DeduceDataTypeFromPrecision(create_info_.precision),
                       TensorStorageType::kBuffer, Layout::kHWC};
  td.SetBHWCShape(BHWC(config_.batch_size,
                       config_.sequence_size * feature_config_.shrink_factor +
                           feature_config_.pad_for_streaming,
                       1, 1));
  return td;
}

TensorDescriptor AudioEncoderBuilder::GetCumsumStreamDescriptor() const {
  TensorDescriptor td = TensorDescriptor{
      DataType::kFloat32, TensorStorageType::kTexture2D, Layout::kHWC};
  td.SetBHWCShape(BHWC(1, 1, feature_config_.convs.size(), 3));
  return td;
}

TensorDescriptor AudioEncoderBuilder::GetDepthwiseStreamDescriptor() const {
  TensorDescriptor td =
      TensorDescriptor{DeduceDataTypeFromPrecision(create_info_.precision),
                       TensorStorageType::kTexture2D, Layout::kHWC};
  td.SetBHWCShape(
      BHWC(1, 1, config_.light_conv_kernel_size - 1, config_.model_dimension));
  return td;
}

TensorDescriptor AudioEncoderBuilder::GetSignalDescriptor() const {
  TensorDescriptor td =
      TensorDescriptor{DeduceDataTypeFromPrecision(create_info_.precision),
                       TensorStorageType::kBuffer, Layout::kHWC};
  td.SetBHWCShape(
      BHWC(config_.batch_size,
           config_.relative_positional_embedding_config->past_horizon +
               config_.relative_positional_embedding_config->future_horizon + 1,
           1, config_.model_dimension));
  return td;
}

TensorDescriptor AudioEncoderBuilder::GetKCacheConformerTensorDescriptor()
    const {
  TensorDescriptor td =
      TensorDescriptor{DeduceDataTypeFromPrecision(create_info_.precision),
                       TensorStorageType::kTexture2D, Layout::kHWC};
  td.SetBHWCShape(
      BHWC(1, config_.number_of_kv_heads,
           config_.relative_positional_embedding_config->past_horizon,
           config_.head_dimension));
  return td;
}

TensorDescriptor AudioEncoderBuilder::GetVCacheConformerTensorDescriptor()
    const {
  TensorDescriptor td =
      TensorDescriptor{DeduceDataTypeFromPrecision(create_info_.precision),
                       TensorStorageType::kTexture2D, Layout::kHWC};
  td.SetBHWCShape(
      BHWC(1, config_.number_of_kv_heads,
           config_.relative_positional_embedding_config->past_horizon,
           config_.head_dimension));
  return td;
}

absl::StatusOr<GpuModel> AudioEncoderBuilder::CreateAudioEncoderModel(
    TensorHandle* feature_handle, TensorHandle* mask_handle,
    TensorHandle* params_i32_handle,
    std::vector<GpuModelBuilder::TensorHandle>* key_cached_tensors,
    std::vector<GpuModelBuilder::TensorHandle>* value_cached_tensors,
    TensorHandle* stream_cumsum,
    std::vector<GpuModelBuilder::TensorHandle>* stream_depthwise_tensors,
    TensorHandle* signal, TensorHandle* feature_output_handle) {
  // Setup
  model_builder_ =
      GpuModelBuilder(gpu_info_, create_info_.hints, create_info_.precision,
                      create_info_.storage_type);
  *feature_handle = model_builder_.AddTensor(GetFeatureDescriptor());
  *mask_handle = model_builder_.AddTensor(GetMaskDescriptor());
  *params_i32_handle =
      model_builder_.AddTensor(1, 1, 1, GetI32ParamsCount(),
                               TensorStorageType::kBuffer, DataType::kInt32);
  if (IsStreaming()) {
    *stream_cumsum = model_builder_.AddTensor(GetCumsumStreamDescriptor());
  }
  if (IsModelv3(config_)) {
    *signal = model_builder_.AddTensor(GetSignalDescriptor());
  }
  if (IsModelv3(config_)) {
    if (kConformerBlockSize !=
        config_.relative_positional_embedding_config->block_size) {
      ABSL_VLOG(2) << "Config block size = "
                   << config_.relative_positional_embedding_config->block_size
                   << ", but we will use block size = " << kConformerBlockSize
                   << std::endl;
    }
    for (int i = 0; i < config_.stack_size; ++i) {
      if (IsStreaming()) {
        stream_depthwise_tensors->push_back(
            model_builder_.AddTensor(GetDepthwiseStreamDescriptor()));
      } else {
        stream_depthwise_tensors->push_back(GpuModelBuilder::TensorHandle());
      }
    }
  }

  for (int i = 0; i < config_.stack_size; ++i) {
    if (IsModelv3(config_)) {
      if (IsStreaming()) {
        key_cached_tensors->push_back(
            model_builder_.AddTensor(GetKCacheConformerTensorDescriptor()));
        value_cached_tensors->push_back(
            model_builder_.AddTensor(GetVCacheConformerTensorDescriptor()));
      } else {
        key_cached_tensors->push_back(GpuModelBuilder::TensorHandle());
        value_cached_tensors->push_back(GpuModelBuilder::TensorHandle());
      }
    } else {  // v1 and v2
      key_cached_tensors->push_back(
          model_builder_.AddTensor(GetKCacheTensorDescriptor()));
      value_cached_tensors->push_back(
          model_builder_.AddTensor(GetVCacheTensorDescriptor()));
    }
  }

  // Model Creation
  ABSL_ASSIGN_OR_RETURN(
      auto audio_tokens,
      MakeEncoderLayer(*feature_handle, *mask_handle, key_cached_tensors,
                       value_cached_tensors, *params_i32_handle, *stream_cumsum,
                       stream_depthwise_tensors, signal,
                       "params.audio.transformer"));
  // only used by tiny gemma
  if (usm_projection_config_.output_dim > 0) {
    ABSL_ASSIGN_OR_RETURN(
        audio_tokens,
        MakeOutputProjection(audio_tokens, "params.audio.encoder"));
  }
  if (!IsModelv3(config_)) {
    ABSL_ASSIGN_OR_RETURN(
        audio_tokens,
        MakeMLPAdapterLayer(audio_tokens, "params.audio.adapter"));
  } else {
    ABSL_ASSIGN_OR_RETURN(
        audio_tokens,
        MakeProjectionV3(audio_tokens, "params.audio.transformer.embedder"));
  }
  *feature_output_handle = std::move(audio_tokens);
  // Post
  GpuModel gpu_model;
  std::vector<GpuModelBuilder::ValueId> input_ids = {
      feature_handle->id, mask_handle->id, params_i32_handle->id};
  std::vector<GpuModelBuilder::ValueId> output_ids = {
      feature_output_handle->id};
  if (IsStreaming() || !IsModelv3(config_)) {
    // no kv cache for v3 non-streaming
    for (int i = 0; i < key_cached_tensors->size(); ++i) {
      input_ids.push_back(key_cached_tensors->at(i).id);
      output_ids.push_back(key_cached_tensors->at(i).id);
    }
    for (int i = 0; i < value_cached_tensors->size(); ++i) {
      input_ids.push_back(value_cached_tensors->at(i).id);
      output_ids.push_back(value_cached_tensors->at(i).id);
    }
  }
  if (IsStreaming()) {
    input_ids.push_back(stream_cumsum->id);
    output_ids.push_back(stream_cumsum->id);
    if (IsModelv3(config_)) {
      for (int i = 0; i < config_.stack_size; ++i) {
        input_ids.push_back(stream_depthwise_tensors->at(i).id);
        output_ids.push_back(stream_depthwise_tensors->at(i).id);
      }
    }
  }
  if (IsModelv3(config_)) {
    input_ids.push_back(signal->id);
  }
  ABSL_RETURN_IF_ERROR(
      model_builder_.GetGpuModel(input_ids, output_ids, &gpu_model));
  return std::move(gpu_model);
}

bool IsModelv3(const TransformerConfig& config) {
  return config.relative_positional_embedding_config.has_value() &&
         config.light_conv_kernel_size > 0;
}

absl::StatusOr<GpuModel> AudioEncoderBuilder::CreateCleanCacheModelv3(
    std::vector<GpuModelBuilder::TensorHandle>* key_cached_tensors,
    std::vector<GpuModelBuilder::TensorHandle>* value_cached_tensors,
    GpuModelBuilder::TensorHandle& stream_cumsum,
    std::vector<GpuModelBuilder::TensorHandle>* stream_depthwise_tensors) {
  CleanState();
  model_builder_ =
      GpuModelBuilder(gpu_info_, create_info_.hints, create_info_.precision,
                      create_info_.storage_type);

  // Cache can stay f16 for bf16
  ChangePrecisionIfBfloat16Fix(CalculationsPrecision::kF16);

  const auto cumsum_descriptor = GetCumsumStreamDescriptor();
  const auto k_descriptor = GetKCacheConformerTensorDescriptor();
  const auto v_descriptor = GetVCacheConformerTensorDescriptor();
  const auto depthwise_descriptor = GetDepthwiseStreamDescriptor();
  for (int i = 0; i < config_.stack_size; ++i) {
    key_cached_tensors->push_back(model_builder_.AddTensor(k_descriptor));
    value_cached_tensors->push_back(model_builder_.AddTensor(v_descriptor));
    stream_depthwise_tensors->push_back(
        model_builder_.AddTensor(depthwise_descriptor));
  }
  stream_cumsum = model_builder_.AddTensor(cumsum_descriptor);

  TensorFloat32 zero_cache_cpu;
  zero_cache_cpu.shape = k_descriptor.GetBHWCShape();
  zero_cache_cpu.data =
      std::vector<float>(zero_cache_cpu.shape.DimensionsProduct(), 0.0f);
  TensorHandle zero_cache_tensor = model_builder_.AddConstantTensor(
      zero_cache_cpu, k_descriptor.GetDataType());
  for (int i = 0; i < key_cached_tensors->size(); ++i) {
    key_cached_tensors->at(i) =
        model_builder_.Elementwise(zero_cache_tensor, OperationType::kCopy);
    value_cached_tensors->at(i) =
        model_builder_.Elementwise(zero_cache_tensor, OperationType::kCopy);
  }
  // Clear stream cumsum
  TensorFloat32 zero_cumsum_cpu;
  zero_cumsum_cpu.shape = cumsum_descriptor.GetBHWCShape();
  zero_cumsum_cpu.data =
      std::vector<float>(zero_cumsum_cpu.shape.DimensionsProduct(), 0.0f);
  TensorHandle zero_cumsum_tensor = model_builder_.AddConstantTensor(
      zero_cumsum_cpu, cumsum_descriptor.GetDataType());
  zero_cumsum_tensor =
      model_builder_.Cast(zero_cumsum_tensor, DataType::kFloat32);
  stream_cumsum =
      model_builder_.Elementwise(zero_cumsum_tensor, OperationType::kCopy);

  // Clear stream depthwise
  TensorFloat32 zero_depthwise_cpu;
  zero_depthwise_cpu.shape = depthwise_descriptor.GetBHWCShape();
  zero_depthwise_cpu.data =
      std::vector<float>(zero_depthwise_cpu.shape.DimensionsProduct(), 0.0f);
  TensorHandle zero_depthwise_tensor = model_builder_.AddConstantTensor(
      zero_depthwise_cpu, depthwise_descriptor.GetDataType());
  for (int i = 0; i < stream_depthwise_tensors->size(); ++i) {
    stream_depthwise_tensors->at(i) =
        model_builder_.Elementwise(zero_depthwise_tensor, OperationType::kCopy);
  }

  std::vector<GpuModelBuilder::ValueId> input_ids, output_ids;
  for (int i = 0; i < key_cached_tensors->size(); ++i) {
    input_ids.push_back(key_cached_tensors->at(i).id);
    output_ids.push_back(key_cached_tensors->at(i).id);
  }
  for (int i = 0; i < value_cached_tensors->size(); ++i) {
    input_ids.push_back(value_cached_tensors->at(i).id);
    output_ids.push_back(value_cached_tensors->at(i).id);
  }
  input_ids.push_back(stream_cumsum.id);
  output_ids.push_back(stream_cumsum.id);
  for (int i = 0; i < stream_depthwise_tensors->size(); ++i) {
    input_ids.push_back(stream_depthwise_tensors->at(i).id);
    output_ids.push_back(stream_depthwise_tensors->at(i).id);
  }

  // Set default back to f32 for bf16 models
  ChangePrecisionIfBfloat16Fix(CalculationsPrecision::kF32);

  GpuModel gpu_model;
  ABSL_RETURN_IF_ERROR(model_builder_.GetGpuModel({}, output_ids, &gpu_model));
  return std::move(gpu_model);
}

}  // namespace ml_drift
