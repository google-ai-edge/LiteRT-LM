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

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_operation.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "runtime/executor/gpu_artisan/ml_drift/tasks/fused_rope.h"
#include "runtime/executor/gpu_artisan/ml_drift/transformer.h"

namespace ml_drift {

// From transformer.cc
int GetHeadSize(const TransformerConfig& config, bool is_global);
int GetNumKvHeads(const TransformerConfig& config, bool is_global);

// qkv input:
//  q - bhwc(B, 1, S, HC*HS)
//  k - bhwc(B, 1, S, KVHC*HS)
//  v - bhwc(B, 1, S, KVHC*HS)
//
// qkv reshape to:
//  q - bhwc(B, S,   HC, HS)
//  k - bhwc(B, S, KVHC, HS)
//  v - bhwc(B, S, KVHC, HS)
//
// qkv transpose to:
//  q - bhwc(B,   HC, S, HS) = bhwc(B, KVHC*HGS, S, HS)
//  k - bhwc(B, KVHC, S, HS)
//  v - bhwc(B, KVHC, S, HS)
//
// qkv reshape to:
//  q - hwc(B*KVHC, HGS*S, HS)
//  k - hwc(B*KVHC, S, HS)
//  v - hwc(B*KVHC, S, HS)
void TransformerBuilder::MakeTransformQKV(GpuModelBuilder& builder,
                                          GpuModelBuilder::TensorHandle* q,
                                          GpuModelBuilder::TensorHandle* k,
                                          GpuModelBuilder::TensorHandle* v,
                                          bool is_global) {
  const int batch_size = config_.batch_size;
  const int sequence_size = config_.sequence_size;
  const int head_size = GetHeadSize(config_, is_global);
  const int heads_count = config_.number_of_heads;
  const int kv_heads_count = GetNumKvHeads(config_, is_global);
  const int kv_group_size = heads_count / kv_heads_count;
  if (q != nullptr) {
    *q = builder.Reshape(
        *q, BHWC{batch_size, sequence_size, heads_count, head_size});
    *q = builder.Transpose(*q, BHWC(0, 2, 1, 3));
    *q = builder.Reshape(*q, BHWC{1, batch_size * kv_heads_count,
                                  sequence_size * kv_group_size, head_size});
  }
  if (k != nullptr) {
    *k = builder.Reshape(
        *k, BHWC{batch_size, sequence_size, kv_heads_count, head_size});
    *k = builder.Transpose(*k, BHWC(0, 2, 1, 3));
    *k = builder.Reshape(
        *k, BHWC{1, batch_size * kv_heads_count, sequence_size, head_size});
  }
  if (v != nullptr) {
    *v = builder.Reshape(
        *v, BHWC{batch_size, sequence_size, kv_heads_count, head_size});
    *v = builder.Transpose(*v, BHWC(0, 2, 1, 3));
    *v = builder.Reshape(
        *v, BHWC{1, batch_size * kv_heads_count, sequence_size, head_size});
  }
}

// Fused operation that performs QKV layout transformation (Reshape -> Transpose
// -> Reshape) combined with Rotary Position Embedding (RoPE) application across
// Q and K tensors.
//
// For Q & K: Reshape -> Transpose -> Reshape -> Split -> RoPE -> Concat
// For V:     Reshape -> Transpose -> Reshape
void TransformerBuilder::MakeTransformQKVApplyRoPE(
    GpuModelBuilder& builder, const GpuModelBuilder::TensorHandle& position,
    GpuModelBuilder::TensorHandle* q, GpuModelBuilder::TensorHandle* k,
    GpuModelBuilder::TensorHandle* v, float min_timescale,
    float max_timescale) {
  const int heads_count = config_.number_of_heads;
  const int head_size = config_.head_dimension;
  const int batch_size = config_.batch_size;
  const int sequence_size = config_.sequence_size;
  const int kv_heads_count = config_.number_of_kv_heads;
  const int kv_head_group_size = heads_count / kv_heads_count;

  auto q_dst =
      builder.AddTensor(BHWC{1, batch_size * kv_heads_count,
                             kv_head_group_size * sequence_size, head_size},
                        q->tensor_desc.GetDataType());
  auto k_dst = builder.AddTensor(
      BHWC{1, batch_size * kv_heads_count, sequence_size, head_size},
      k->tensor_desc.GetDataType());
  auto v_dst = builder.AddTensor(
      BHWC{1, batch_size * kv_heads_count, sequence_size, head_size},
      v->tensor_desc.GetDataType());

  auto custom_op = CreateTransformQKVApplyRoPEOp(
      gpu_info_, q->tensor_desc, k->tensor_desc, v->tensor_desc,
      position.tensor_desc, q_dst.tensor_desc, k_dst.tensor_desc,
      v_dst.tensor_desc, kv_heads_count, kv_head_group_size, sequence_size,
      min_timescale, max_timescale);

  builder.AddGpuOperation({*q, *k, *v, position}, {q_dst, k_dst, v_dst},
                          std::make_unique<GPUOperation>(std::move(custom_op)),
                          "RoPE (QKVReshape -> QK Split->RoPE->Concat)");
  *q = q_dst;
  *k = k_dst;
  *v = v_dst;
}

// Fused kernel details for QKVRmsNormRoPE operation:
// for Q: (Reshape+Transpose+Reshape) -> (RMSNorm) -> (Split+RoPE+Concat)
// for K: (Reshape+Transpose+Reshape) -> (RMSNorm) -> (Split+RoPE+Concat)
// for V: (Reshape+Transpose+Reshape)
// Note: K & V are skipped when kv cache is shared.
// Note: V = K-Transformed when config_.global_k_eq_v is true.
//
// Grid Size:
// Mapped via TensorToGrid::kWBToX_HDToY_SToZ to dst_[0] (q_dst).
// - Grid X: Width * Batch = (kv_head_group_size * sequence_size) * 1.
// - Grid Y: Height * Depth = (batch_size * kv_heads_count) * 1.
// - Grid Z: Slices = head_size / 4 (each slice contains 4 floats).
//
// Workgroup Size:
// Explicitly set to (1, 1, head_size / 8).
// A single workgroup processes the reduction and RoPE operations (x2) for a
// single attention head (across one sequence element, batch, and head).
//
// Thread Usage:
// - Because RoPE pairs features x_i with x_{i + d/2} (where d = head_size),
//   and d/2 elements spans half_slices_count (head_size / 8), each active
//   thread at local Z coordinate LOCAL_S processes two slices:
//   DST_S0 = DST_S_GLOBAL and DST_S1 = DST_S_GLOBAL + half_slices_count.
// - Only threads with DST_S_GLOBAL < head_size / 8 are active; others return.
// - For RMSNorm reduction, the head_size / 8 threads in the workgroup
//   cooperatively load slices, compute local sum of squares, store them in
//   local memory (__local shared_q_sum/shared_k_sum), and do a local memory
//   reduction. Thread 0 computes the inverse standard deviation, which is
//   broadcast/used by all threads in the workgroup to normalize their
//   features before applying RoPE and writing out the results.
absl::Status TransformerBuilder::MakeTransformQKVRmsNormApplyRoPE(
    GpuModelBuilder& builder, const GpuModelBuilder::TensorHandle& position,
    GpuModelBuilder::TensorHandle* q, GpuModelBuilder::TensorHandle* k,
    GpuModelBuilder::TensorHandle* v, bool is_global, float min_timescale,
    float max_timescale, const std::string& qk_norm_prefix, float scale_shift,
    float proportion) {
  const bool has_kv = (k != nullptr && v != nullptr);
  const bool k_eq_v = has_kv && (k == v || k->id == v->id);
  const int heads_count = config_.number_of_heads;
  const int head_size = GetHeadSize(config_, is_global);
  const int batch_size = config_.batch_size;
  const int sequence_size = config_.sequence_size;
  const int kv_heads_count = GetNumKvHeads(config_, is_global);
  const int kv_head_group_size = heads_count / kv_heads_count;

  auto q_dst =
      builder.AddTensor(BHWC{1, batch_size * kv_heads_count,
                             kv_head_group_size * sequence_size, head_size},
                        q->tensor_desc.GetDataType());

  GpuModelBuilder::TensorHandle k_dst, v_dst;
  if (has_kv) {
    k_dst = builder.AddTensor(
        BHWC{1, batch_size * kv_heads_count, sequence_size, head_size},
        k->tensor_desc.GetDataType());
    v_dst = builder.AddTensor(
        BHWC{1, batch_size * kv_heads_count, sequence_size, head_size},
        v->tensor_desc.GetDataType());
  }

  const bool has_lrn_norm = !qk_norm_prefix.empty();
  GpuModelBuilder::TensorHandle q_gamma_tensor, k_gamma_tensor;
  if (has_lrn_norm) {
    ml_drift::Tensor<Linear, DataType::kFloat32> q_gamma;
    q_gamma.shape = Linear(head_size);
    q_gamma.data = tensor_loader_->LoadFloat32(qk_norm_prefix + ".q_norm.scale",
                                               head_size);
    if (scale_shift != 0.0f) {
      for (int i = 0; i < q_gamma.data.size(); ++i) {
        q_gamma.data[i] = scale_shift + q_gamma.data[i];
      }
    }
    q_gamma_tensor = builder.AddConstantTensor(q_gamma, data_type_);

    if (has_kv) {
      ml_drift::Tensor<Linear, DataType::kFloat32> k_gamma;
      k_gamma.shape = Linear(head_size);
      k_gamma.data = tensor_loader_->LoadFloat32(
          qk_norm_prefix + ".k_norm.scale", head_size);
      if (scale_shift != 0.0f) {
        for (int i = 0; i < k_gamma.data.size(); ++i) {
          k_gamma.data[i] = scale_shift + k_gamma.data[i];
        }
      }
      k_gamma_tensor = builder.AddConstantTensor(k_gamma, data_type_);
    }
  }

  auto custom_op = CreateTransformQKVRmsNormApplyRoPEOp(
      gpu_info_, q->tensor_desc, position.tensor_desc, q_dst.tensor_desc,
      kv_heads_count, kv_head_group_size, sequence_size, min_timescale,
      max_timescale, proportion, k_eq_v, has_kv ? &k->tensor_desc : nullptr,
      (has_kv && !k_eq_v) ? &v->tensor_desc : nullptr,
      has_kv ? &k_dst.tensor_desc : nullptr,
      has_kv ? &v_dst.tensor_desc : nullptr,
      has_lrn_norm ? &q_gamma_tensor.tensor_desc : nullptr,
      (has_lrn_norm && has_kv) ? &k_gamma_tensor.tensor_desc : nullptr);

  std::vector<GpuModelBuilder::TensorHandle> inputs = {*q};
  if (has_kv) {
    inputs.push_back(*k);
    if (!k_eq_v) {
      inputs.push_back(*v);
    }
  }
  inputs.push_back(position);
  if (has_lrn_norm) {
    inputs.push_back(q_gamma_tensor);
    if (has_kv) {
      inputs.push_back(k_gamma_tensor);
    }
  }

  std::vector<GpuModelBuilder::TensorHandle> outputs = {q_dst};
  if (has_kv) {
    outputs.push_back(k_dst);
    outputs.push_back(v_dst);
  }

  builder.AddGpuOperation(
      inputs, outputs,
      std::make_unique<TransformQKVRmsNormApplyRoPEOp>(std::move(custom_op)),
      "QKVRmsNormRoPE (Reshape -> Transpose -> RMSNorm -> RoPE)");

  *q = q_dst;
  if (has_kv) {
    *k = k_dst;
    *v = v_dst;
  }

  return absl::OkStatus();
}

// Main dispatcher function for TransformQKV + RMS Norm + RoPE combinations.
absl::Status TransformerBuilder::QKVRoPE(
    GpuModelBuilder& builder, const GpuModelBuilder::TensorHandle& segment_pos,
    GpuModelBuilder::TensorHandle* q, GpuModelBuilder::TensorHandle* k,
    GpuModelBuilder::TensorHandle* v, bool is_global,
    const std::string& qk_norm_prefix, std::optional<float> scale_shift,
    std::optional<float> proportion_opt) {
  const int head_size = GetHeadSize(config_, is_global);

  float min_timescale = 1.0f;
  // Default rope wavelength for local attention layers.
  float max_timescale = 10000.0f;
  // Default p-RoPE (RoPE vs. NoPE) proportion (default is entirely RoPE)
  float proportion = proportion_opt.value_or(1.0f);

  // Alter global RoPE wavelength strings when specified
  if (is_global && config_.global_rope_wavelength.has_value()) {
    max_timescale = *config_.global_rope_wavelength;
  }
  if (is_global && config_.global_rope_scaling.has_value()) {
    float scale_factor = *config_.global_rope_scaling;
    min_timescale *= scale_factor;
    max_timescale *= scale_factor;
  }
  if (!proportion_opt.has_value() && is_global &&
      config_.global_rope_proportion.has_value()) {
    proportion = *config_.global_rope_proportion;
  }

  const float actual_scale_shift =
      scale_shift.value_or(config_.rms_norm_scale_shift);
  const bool use_fused_norm_rope = config_.qk_norm && (head_size % 8 == 0);
  const bool has_kv = (k != nullptr && v != nullptr);

  if (use_fused_norm_rope) {
    return MakeTransformQKVRmsNormApplyRoPE(
        builder, segment_pos, q, k, v, is_global, min_timescale, max_timescale,
        qk_norm_prefix, actual_scale_shift, proportion);
  } else if (k == nullptr && v == nullptr) {
    // Shared KV cache: transform only Q.
    MakeTransformQKV(builder, q, /*k=*/nullptr, /*v=*/nullptr, is_global);
    if (config_.qk_norm) {
      if (qk_norm_prefix.empty()) {
        *q = builder.RMSNormalization(*q, /*epsilon=*/1e-6f);
      } else {
        *q = MakeRMSNormalizationWithBuilder(
            builder, *q, qk_norm_prefix + ".q_norm", actual_scale_shift);
      }
    }
    *q = MakeRotaryEmbedding(builder, *q, segment_pos, min_timescale,
                             max_timescale, proportion);
  } else if (head_size % 8 == 0 && !config_.qk_norm && has_kv) {
    MakeTransformQKVApplyRoPE(builder, segment_pos, q, k, v, min_timescale,
                              max_timescale);
  } else {
    if (config_.global_k_eq_v && is_global && has_kv) {
      // Can save time by transforming only k and then assigning it to v.
      MakeTransformQKV(builder, q, k, /*v=*/nullptr, is_global);
      *v = *k;
    } else {
      MakeTransformQKV(builder, q, k, v, is_global);
    }
    if (config_.qk_norm) {
      if (qk_norm_prefix.empty()) {
        if (q != nullptr) *q = builder.RMSNormalization(*q, /*epsilon=*/1e-6f);
        if (k != nullptr) *k = builder.RMSNormalization(*k, /*epsilon=*/1e-6f);
      } else {
        if (q != nullptr) {
          *q = MakeRMSNormalizationWithBuilder(
              builder, *q, qk_norm_prefix + ".q_norm", actual_scale_shift);
        }
        if (k != nullptr) {
          *k = MakeRMSNormalizationWithBuilder(
              builder, *k, qk_norm_prefix + ".k_norm", actual_scale_shift);
        }
      }
    }
    if (q != nullptr) {
      *q = MakeRotaryEmbedding(builder, *q, segment_pos, min_timescale,
                               max_timescale, proportion);
    }
    if (k != nullptr) {
      *k = MakeRotaryEmbedding(builder, *k, segment_pos, min_timescale,
                               max_timescale, proportion);
    }
  }
  return absl::OkStatus();
}

}  // namespace ml_drift
