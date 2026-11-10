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

#include "runtime/executor/gpu_artisan/ml_drift/tasks/fused_rope.h"

#include <string>

#include "absl/strings/substitute.h"  // from @com_google_absl
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_operation.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift

namespace ml_drift {

GPUOperation CreateTransformQKVApplyRoPEOp(
    const GpuInfo& gpu_info, const TensorDescriptor& q_src,
    const TensorDescriptor& k_src, const TensorDescriptor& v_src,
    const TensorDescriptor& position, const TensorDescriptor& q_dst,
    const TensorDescriptor& k_dst, const TensorDescriptor& v_dst,
    int kv_heads_count, int kv_head_group_size, int sequence_size,
    float min_timescale, float max_timescale) {
  std::string pow_func_name = "pow";
  std::string sin_func_name = "sin";
  std::string cos_func_name = "cos";
  if (gpu_info.IsAdreno() && gpu_info.IsApiOpenCl()) {
    pow_func_name = "native_powr";
    sin_func_name = "native_sin";
    cos_func_name = "native_cos";
  }
  std::string code =
      absl::Substitute(R"(MAIN_FUNCTION($$0) {
  int DST_X = ucl::GetGlobalId<0>(); // kv_group_size * sequence_size
  int DST_Y = ucl::GetGlobalId<1>(); // batch_size * kv_heads_count
  int DST_S = ucl::GetGlobalId<2>(); // head_size

  int half_slices_count = args.q_dst.Slices() / 2;
  if (DST_X >= args.q_dst.Width() || DST_Y >= args.q_dst.Height() || DST_S >= half_slices_count) {
    return;
  }

  int DST_S0 = DST_S;
  int DST_S1 = DST_S + half_slices_count;

  int batch_id = DST_Y / args.kv_heads_count;
  int kv_head_id = DST_Y % args.kv_heads_count;
  int kv_head_group_id = DST_X / args.sequence_size;
  int seq_id = DST_X % args.sequence_size;

  args.q_src.SetBatchRef(batch_id);
  args.k_src.SetBatchRef(batch_id);
  args.v_src.SetBatchRef(batch_id);

  float4 fraction;
  float inv_q_dst_ch = 1.0f / ucl::Convert<float>(args.q_dst.Channels());
  fraction.x = 2.0f * ucl::Convert<float>(DST_S * 4 + 0) * inv_q_dst_ch;
  fraction.y = 2.0f * ucl::Convert<float>(DST_S * 4 + 1) * inv_q_dst_ch;
  fraction.z = 2.0f * ucl::Convert<float>(DST_S * 4 + 2) * inv_q_dst_ch;
  fraction.w = 2.0f * ucl::Convert<float>(DST_S * 4 + 3) * inv_q_dst_ch;
  float4 min_timescale = ucl::Init<float4>(args.min_timescale);
  float4 max_timescale = ucl::Init<float4>(args.max_timescale);
  float4 timescale = min_timescale * $0(max_timescale / min_timescale, fraction);
  float4 pos_val = ucl::Init<float4>(args.position.Read<float>(seq_id, 0, 0).x);
  float4 sinusoid_inp = pos_val / timescale;
  float4 sin_val = $1(sinusoid_inp);
  float4 cos_val = $2(sinusoid_inp);
  {
    int q_head_id = kv_head_id * args.kv_head_group_size + kv_head_group_id;
    int SRC_S0 = q_head_id * args.q_dst.Slices() + DST_S0;
    int SRC_S1 = q_head_id * args.q_dst.Slices() + DST_S1;
    float4 val0 = ucl::Convert<float4>(args.q_src.Read(seq_id, 0, SRC_S0));
    float4 val1 = ucl::Convert<float4>(args.q_src.Read(seq_id, 0, SRC_S1));
    float4 out0 = val0 * cos_val - val1 * sin_val;
    float4 out1 = val1 * cos_val + val0 * sin_val;
    args.q_dst::type q_out0 = ucl::Convert<args.q_dst::type>(out0);
    args.q_dst::type q_out1 = ucl::Convert<args.q_dst::type>(out1);
    args.q_dst.Write(q_out0, DST_X, DST_Y, DST_S0);
    args.q_dst.Write(q_out1, DST_X, DST_Y, DST_S1);
  }
  if (kv_head_group_id == 0) {
    int SRC_S0 = kv_head_id * args.q_dst.Slices() + DST_S0;
    int SRC_S1 = kv_head_id * args.q_dst.Slices() + DST_S1;
    {
      float4 val0 = ucl::Convert<float4>(args.k_src.Read(seq_id, 0, SRC_S0));
      float4 val1 = ucl::Convert<float4>(args.k_src.Read(seq_id, 0, SRC_S1));
      float4 out0 = val0 * cos_val - val1 * sin_val;
      float4 out1 = val1 * cos_val + val0 * sin_val;
      args.k_dst::type k_out0 = ucl::Convert<args.k_dst::type>(out0);
      args.k_dst::type k_out1 = ucl::Convert<args.k_dst::type>(out1);
      args.k_dst.Write(k_out0, DST_X, DST_Y, DST_S0);
      args.k_dst.Write(k_out1, DST_X, DST_Y, DST_S1);
    }
    {
      args.v_dst::type v_out0 = args.v_src.Read(seq_id, 0, SRC_S0);
      args.v_dst::type v_out1 = args.v_src.Read(seq_id, 0, SRC_S1);
      args.v_dst.Write(v_out0, DST_X, DST_Y, DST_S0);
      args.v_dst.Write(v_out1, DST_X, DST_Y, DST_S1);
    }
  }
})",
                       pow_func_name, sin_func_name, cos_func_name);

  GPUOperation custom_op;
  custom_op.args_.AddInt("kv_heads_count", kv_heads_count);
  custom_op.args_.AddInt("kv_head_group_size", kv_head_group_size);
  custom_op.args_.AddInt("sequence_size", sequence_size);
  custom_op.args_.AddFloat("min_timescale", min_timescale);
  custom_op.args_.AddFloat("max_timescale", max_timescale);
  custom_op.AddSrcTensor("q_src", q_src);
  custom_op.AddSrcTensor("k_src", k_src);
  custom_op.AddSrcTensor("v_src", v_src);
  custom_op.AddSrcTensor("position", position);
  custom_op.AddDstTensor("q_dst", q_dst);
  custom_op.AddDstTensor("k_dst", k_dst);
  custom_op.AddDstTensor("v_dst", v_dst);
  custom_op.code_ = code;
  custom_op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
  return custom_op;
}

TransformQKVRmsNormApplyRoPEOp CreateTransformQKVRmsNormApplyRoPEOp(
    const GpuInfo& gpu_info, const TensorDescriptor& q_src,
    const TensorDescriptor& position, const TensorDescriptor& q_dst,
    int kv_heads_count, int kv_head_group_size, int sequence_size,
    float min_timescale, float max_timescale, float proportion, bool k_eq_v,
    const TensorDescriptor* k_src, const TensorDescriptor* v_src,
    const TensorDescriptor* k_dst, const TensorDescriptor* v_dst,
    const TensorDescriptor* q_gamma, const TensorDescriptor* k_gamma) {
  const bool has_kv = (k_src != nullptr && k_dst != nullptr &&
                       v_dst != nullptr && (k_eq_v || v_src != nullptr));
  const bool has_lrn_norm = (q_gamma != nullptr);
  const int head_size = q_dst.GetBHWCShape().c;

  std::string pow_func_name = "pow";
  std::string sin_func_name = "sin";
  std::string cos_func_name = "cos";
  if (gpu_info.IsAdreno() && gpu_info.IsApiOpenCl()) {
    pow_func_name = "native_powr";
    sin_func_name = "native_sin";
    cos_func_name = "native_cos";
  }

  std::string code = R"(MAIN_FUNCTION($0) {
  int DST_X = ucl::GetGlobalId<0>();
  int DST_Y = ucl::GetGlobalId<1>();
  int DST_S_GLOBAL = ucl::GetGlobalId<2>();
  int LOCAL_S = ucl::GetLocalId<2>();

  int half_slices_count = args.q_dst.Slices() / 2;
  bool is_active = (DST_X < args.q_dst.Width() && DST_Y < args.q_dst.Height() && DST_S_GLOBAL < half_slices_count);

  int DST_S0 = DST_S_GLOBAL;
  int DST_S1 = DST_S_GLOBAL + half_slices_count;

  int batch_id = DST_Y / args.kv_heads_count;
  int kv_head_id = DST_Y % args.kv_heads_count;
  int kv_head_group_id = DST_X / args.sequence_size;
  int seq_id = DST_X % args.sequence_size;

  args.q_src.SetBatchRef(batch_id);
)";

  if (has_kv) {
    code += R"(  args.k_src.SetBatchRef(batch_id);
)";
    if (!k_eq_v) {
      code += R"(  args.v_src.SetBatchRef(batch_id);
)";
    }
  }

  code += absl::Substitute(R"(
  __local float shared_q_sum[$0];
  __local float shared_k_sum[$0];
  __local float shared_inv_std_q;
  __local float shared_inv_std_k;
)",
                           head_size / 8);

  code += R"(
  // 1. Cooperative Load & Sum of Squares for Q
  float4 q_val0 = ucl::Init<float4>(0.0f);
  float4 q_val1 = ucl::Init<float4>(0.0f);
  float local_sum_q = 0.0f;
  if (is_active) {
    int q_head_id = kv_head_id * args.kv_head_group_size + kv_head_group_id;
    int SRC_S0 = q_head_id * args.q_dst.Slices() + DST_S0;
    int SRC_S1 = q_head_id * args.q_dst.Slices() + DST_S1;
    q_val0 = ucl::Convert<float4>(args.q_src.Read(seq_id, 0, SRC_S0));
    q_val1 = ucl::Convert<float4>(args.q_src.Read(seq_id, 0, SRC_S1));
    local_sum_q = dot(q_val0, q_val0) + dot(q_val1, q_val1);
  }
  shared_q_sum[LOCAL_S] = local_sum_q;
)";

  if (has_kv) {
    code += R"(
  // Cooperative Load & Sum of Squares for K
  float4 raw_k_val0 = ucl::Init<float4>(0.0f);
  float4 raw_k_val1 = ucl::Init<float4>(0.0f);
  float4 k_val0 = ucl::Init<float4>(0.0f);
  float4 k_val1 = ucl::Init<float4>(0.0f);
  float local_sum_k = 0.0f;
  if (is_active && kv_head_group_id == 0) {
    int K_SRC_S0 = kv_head_id * args.q_dst.Slices() + DST_S0;
    int K_SRC_S1 = kv_head_id * args.q_dst.Slices() + DST_S1;
    raw_k_val0 = ucl::Convert<float4>(args.k_src.Read(seq_id, 0, K_SRC_S0));
    raw_k_val1 = ucl::Convert<float4>(args.k_src.Read(seq_id, 0, K_SRC_S1));
    k_val0 = raw_k_val0;
    k_val1 = raw_k_val1;
    local_sum_k = dot(k_val0, k_val0) + dot(k_val1, k_val1);
  }
  shared_k_sum[LOCAL_S] = local_sum_k;
)";
  }

  code += R"(
  ucl::SyncThreads<WorkGroup, Local>();
)";

  code += absl::Substitute(R"(
  int reduction_size = $0;
  while (reduction_size > 1) {
    int active_thread_limit = reduction_size / 2;
    int offset = (reduction_size + 1) / 2;
    if (LOCAL_S < active_thread_limit) {
      shared_q_sum[LOCAL_S] += shared_q_sum[LOCAL_S + offset];
      shared_k_sum[LOCAL_S] += shared_k_sum[LOCAL_S + offset];
    }
    ucl::SyncThreads<WorkGroup, Local>();
    reduction_size = offset;
  }
)",
                           head_size / 8);

  code += R"(
  if (LOCAL_S == 0) {
    shared_inv_std_q = rsqrt(shared_q_sum[0] / ucl::Convert<float>(args.q_dst.Channels()) + 1e-6f);
)";
  if (has_kv) {
    code += R"(
    shared_inv_std_k = rsqrt(shared_k_sum[0] / ucl::Convert<float>(args.k_dst.Channels()) + 1e-6f);
)";
  }
  code += R"(
  }
  ucl::SyncThreads<WorkGroup, Local>();

  if (is_active) {
    float stddev_inv_q = shared_inv_std_q;
    float4 val0 = q_val0 * stddev_inv_q;
    float4 val1 = q_val1 * stddev_inv_q;
)";

  if (has_lrn_norm) {
    code += R"(
    float4 gamma0 = args.q_gamma.Read<float>(DST_S0);
    float4 gamma1 = args.q_gamma.Read<float>(DST_S1);
    val0 = val0 * gamma0;
    val1 = val1 * gamma1;
)";
  }

  code += R"(
    // 2. RoPE Constants
    float4 fraction;
    float inv_q_dst_ch = 1.0f / ucl::Convert<float>(args.q_dst.Channels());
    fraction.x = 2.0f * ucl::Convert<float>(DST_S_GLOBAL * 4 + 0) * inv_q_dst_ch;
    fraction.y = 2.0f * ucl::Convert<float>(DST_S_GLOBAL * 4 + 1) * inv_q_dst_ch;
    fraction.z = 2.0f * ucl::Convert<float>(DST_S_GLOBAL * 4 + 2) * inv_q_dst_ch;
    fraction.w = 2.0f * ucl::Convert<float>(DST_S_GLOBAL * 4 + 3) * inv_q_dst_ch;
    float4 min_timescale = ucl::Init<float4>(args.min_timescale);
    float4 max_timescale = ucl::Init<float4>(args.max_timescale);
)";
  code += "    float4 timescale = min_timescale * " + pow_func_name +
          "(max_timescale / min_timescale, fraction);\n";
  code +=
      R"(    float4 pos_val = ucl::Init<float4>(args.position.Read<float>(seq_id, 0, 0).x);
    float4 sinusoid_inp = pos_val / timescale;
)";
  code += "    float4 sin_val = " + sin_func_name + "(sinusoid_inp);\n";
  code += "    float4 cos_val = " + cos_func_name + "(sinusoid_inp);\n";

  code += R"(
    // 3. Process Q
    {
      float4 out0;
      float4 out1;
      if (fraction.w < ucl::Init<float>(args.proportion)) {
        out0 = val0 * cos_val - val1 * sin_val;
        out1 = val1 * cos_val + val0 * sin_val;
      } else {
        out0 = val0;
        out1 = val1;
      }
      args.q_dst::type q_out0 = ucl::Convert<args.q_dst::type>(out0);
      args.q_dst::type q_out1 = ucl::Convert<args.q_dst::type>(out1);
      args.q_dst.Write(q_out0, DST_X, DST_Y, DST_S0);
      args.q_dst.Write(q_out1, DST_X, DST_Y, DST_S1);
    }
)";

  if (has_kv) {
    code += R"(
    // 4. Process K and V
    if (kv_head_group_id == 0) {
      int K_SRC_S0 = kv_head_id * args.q_dst.Slices() + DST_S0;
      int K_SRC_S1 = kv_head_id * args.q_dst.Slices() + DST_S1;

      // Process K
      {
        float stddev_inv_k = shared_inv_std_k;
        k_val0 = k_val0 * stddev_inv_k;
        k_val1 = k_val1 * stddev_inv_k;
)";

    if (has_lrn_norm) {
      code += R"(
        float4 gamma0 = args.k_gamma.Read<float>(DST_S0);
        float4 gamma1 = args.k_gamma.Read<float>(DST_S1);
        k_val0 = k_val0 * gamma0;
        k_val1 = k_val1 * gamma1;
)";
    }

    code += R"(
        float4 out0;
        float4 out1;
        if (fraction.w < ucl::Init<float>(args.proportion)) {
          out0 = k_val0 * cos_val - k_val1 * sin_val;
          out1 = k_val1 * cos_val + k_val0 * sin_val;
        } else {
          out0 = k_val0;
          out1 = k_val1;
        }
        args.k_dst::type k_out0 = ucl::Convert<args.k_dst::type>(out0);
        args.k_dst::type k_out1 = ucl::Convert<args.k_dst::type>(out1);
        args.k_dst.Write(k_out0, DST_X, DST_Y, DST_S0);
        args.k_dst.Write(k_out1, DST_X, DST_Y, DST_S1);
      }
)";

    if (k_eq_v) {
      code += R"(
      // Process V using raw K values loaded into registers
      {
        args.v_dst::type v_out0 = ucl::Convert<args.v_dst::type>(raw_k_val0);
        args.v_dst::type v_out1 = ucl::Convert<args.v_dst::type>(raw_k_val1);
        args.v_dst.Write(v_out0, DST_X, DST_Y, DST_S0);
        args.v_dst.Write(v_out1, DST_X, DST_Y, DST_S1);
      }
)";
    } else {
      code += R"(
      // Process V
      {
        args.v_dst::type v_out0 = args.v_src.Read(seq_id, 0, K_SRC_S0);
        args.v_dst::type v_out1 = args.v_src.Read(seq_id, 0, K_SRC_S1);
        args.v_dst.Write(v_out0, DST_X, DST_Y, DST_S0);
        args.v_dst.Write(v_out1, DST_X, DST_Y, DST_S1);
      }
)";
    }

    code += R"(
    }
)";
  }

  code += R"(
  }
})";

  TransformQKVRmsNormApplyRoPEOp custom_op;
  custom_op.args_.AddInt("kv_heads_count", kv_heads_count);
  custom_op.args_.AddInt("kv_head_group_size", kv_head_group_size);
  custom_op.args_.AddInt("sequence_size", sequence_size);
  custom_op.args_.AddFloat("min_timescale", min_timescale);
  custom_op.args_.AddFloat("max_timescale", max_timescale);
  custom_op.args_.AddFloat("proportion", proportion);
  custom_op.work_group_size_ = int3(1, 1, head_size / 8);

  custom_op.AddSrcTensor("q_src", q_src);
  if (has_kv) {
    custom_op.AddSrcTensor("k_src", *k_src);
    if (!k_eq_v) {
      custom_op.AddSrcTensor("v_src", *v_src);
    }
  }
  custom_op.AddSrcTensor("position", position);

  if (has_lrn_norm) {
    custom_op.AddSrcTensor("q_gamma", *q_gamma);
    if (has_kv && k_gamma != nullptr) {
      custom_op.AddSrcTensor("k_gamma", *k_gamma);
    }
  }

  custom_op.AddDstTensor("q_dst", q_dst);
  if (has_kv) {
    custom_op.AddDstTensor("k_dst", *k_dst);
    custom_op.AddDstTensor("v_dst", *v_dst);
  }

  custom_op.code_ = code;
  custom_op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
  return custom_op;
}

}  // namespace ml_drift
