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

#include "runtime/executor/gpu_artisan/ml_drift/tasks/flash_decoding_attention.h"

#include <cstdint>
#include <string>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/substitute.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/buffer_desc.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/types.h"  // from @ml_drift
#include "ml_drift/common/util.h"  // from @ml_drift
#include "runtime/executor/gpu_artisan/ml_drift/runtime_params.h"
#include "runtime/util/status_macros.h"

namespace ml_drift {
namespace {

constexpr int kSimdGroupSize = 32;
int3 ChooseWorkGroupSize(const GpuInfo& gpu_info,
                         const FlashDecodingAttentionAttributes& attr) {
  if (attr.head_size < 512) {
    return int3(1, kSimdGroupSize, 16);
  } else {
    // Smaller work-group size for larger head sizes; otherwise, the threadgroup
    // memory exceeds the limit.
    return int3(1, kSimdGroupSize, 8);
  }
}

}  // namespace

FlashDecodingAttention::FlashDecodingAttention(
    const GpuInfo& gpu_info, const TensorDescriptor& q,
    const TensorDescriptor& k_cache, const TensorDescriptor& v_cache,
    const TensorDescriptor& dst, const FlashDecodingAttentionAttributes& attr)
    : attr_(attr) {
  AddSrcTensor("q", q);
  AddSrcTensor("k_cache", k_cache);
  AddSrcTensor("v_cache", v_cache);
  BufferDescriptor params_buffer;
  params_buffer.element_type = DataType::kInt32;
  params_buffer.element_size = 1;
  AddSrcBuffer("params", params_buffer);
  AddDstTensor("dst", dst);

  args_.AddInt("num_q_heads", attr_.num_q_heads);
  args_.AddInt("head_dim_slices", DivideRoundUp(attr_.head_size, 4));
  args_.AddInt("cache_slices", DivideRoundUp(attr_.cache_size, 4));

  const bool is_tuned_head_size = attr_.head_size == 128 ||
                                  attr_.head_size == 256 ||
                                  attr_.head_size == 512;
  ABSL_LOG_IF_FIRST_N(WARNING, !is_tuned_head_size, 1)
      << "FlashDecodingAttention: head_size " << attr_.head_size
      << " is supported but not tuned, so expect suboptimal performance.";

  work_group_size_ = ChooseWorkGroupSize(gpu_info, attr_);
  grid_dimension_ = 1;

  code_ = GenerateCode(gpu_info);

  // Rough traffic estimate for profiling tools (bytes).
  const int elem_size = SizeOf(k_cache.GetDataType());
  read_size_ = static_cast<uint64_t>(attr_.cache_size) * attr_.head_size *
                   elem_size * 2 +
               static_cast<uint64_t>(attr_.num_q_heads) * attr_.head_size *
                   SizeOf(q.GetDataType());
  write_size_ = static_cast<uint64_t>(attr_.num_q_heads) * attr_.head_size *
                SizeOf(dst.GetDataType());
}

int3 FlashDecodingAttention::GetGridSize() const {
  return int3(attr_.num_q_heads, work_group_size_.y, work_group_size_.z);
}

std::string FlashDecodingAttention::GenerateCode(
    const GpuInfo& gpu_info) const {
  const std::string active_tokens_index =
      absl::StrCat(attr_.active_tokens_index);
  const int head_dim_slices = DivideRoundUp(attr_.head_size, 4);
  ABSL_CHECK(head_dim_slices % kSimdGroupSize == 0)
      << "head_size must be a multiple of " << 4 * kSimdGroupSize;
  const int head_slices_per_lane = head_dim_slices / kSimdGroupSize;
  const int num_simd_groups = work_group_size_.z;
  std::string c = absl::Substitute(R"(
MAIN_FUNCTION($$0) {
  const int head = ucl::GetGroupId<0>();
  // Lane i owns dim slices [i * $1, (i + 1) * $1).
  const int head_dim_id = ucl::GetLocalId<1>() * $1;
  const int context_window_id = ucl::GetLocalId<2>();
  const int group_size_for_context_window = $3;  // == work_group_size_.z
  const int head_slices_per_lane = $1;
  if (head >= args.num_q_heads) {
    return;
  }
  const int active_tokens = args.params.Read($0);
  const int active_token_slices = (active_tokens + 3) / 4;
  args.q::type q_val[$1];
  float4 res[$1];
  for (int slice_offset = 0; slice_offset < head_slices_per_lane;
       ++slice_offset) {
    q_val[slice_offset] = args.q.Read(head, 0, head_dim_id + slice_offset);
    res[slice_offset] = ucl::Init<float4>(0.0f);
  }
  float e_sum = 0.0f;
  float cur_max = -INFINITY;
  for (int token_index = context_window_id; token_index < active_token_slices;
       token_index += group_size_for_context_window) {
    // Q * K.
    args.q::type partial_scores = ucl::Init<args.q::type>(0.0f);
    for (int slice_offset = 0; slice_offset < head_slices_per_lane;
         ++slice_offset) {
      // K layout: kOSpatialIOGroupO4I4, O is cache size, I is head dim.
      int k_index =
          ((head_dim_id + slice_offset) * args.cache_slices + token_index) * 4;
      args.k_cache::type k_val0 = args.k_cache.Read(k_index);
      args.k_cache::type k_val1 = args.k_cache.Read(k_index + 1);
      args.k_cache::type k_val2 = args.k_cache.Read(k_index + 2);
      args.k_cache::type k_val3 = args.k_cache.Read(k_index + 3);
      partial_scores += ucl::Init<args.q::type>(
        dot(q_val[slice_offset], k_val0),
        dot(q_val[slice_offset], k_val1),
        dot(q_val[slice_offset], k_val2),
        dot(q_val[slice_offset], k_val3));
    }
    // TODO: Replace simd_sum to enable non-Metal backends.
    float4 scores = ucl::Convert<float4>(simd_sum(partial_scores));

    const int token_pos = token_index * 4;
    if (token_pos + 1 >= active_tokens) scores.y = -INFINITY;
    if (token_pos + 2 >= active_tokens) scores.z = -INFINITY;
    if (token_pos + 3 >= active_tokens) scores.w = -INFINITY;

    float prev_max = cur_max;
    cur_max = max(cur_max,
                  max(max(scores.x, scores.y), max(scores.z, scores.w)));
    float alpha = exp(prev_max - cur_max);
    float4 softmax_scores = exp(scores - cur_max);
    e_sum = e_sum * alpha + softmax_scores.x + softmax_scores.y
          + softmax_scores.z + softmax_scores.w;

    // Scores * V.
    // V layout: kOSpatialIOGroupI4O4, I is cache size, O is head dim.
    args.v_cache::type weights =
        ucl::Convert<args.v_cache::type>(softmax_scores);
    int v_index = (token_index * args.head_dim_slices + head_dim_id) * 4;
    for (int slice_offset = 0; slice_offset < head_slices_per_lane;
         ++slice_offset) {
      args.v_cache::type v_val0 = args.v_cache.Read(v_index + 4 * slice_offset);
      args.v_cache::type v_val1 =
          args.v_cache.Read(v_index + 4 * slice_offset + 1);
      args.v_cache::type v_val2 =
          args.v_cache.Read(v_index + 4 * slice_offset + 2);
      args.v_cache::type v_val3 =
          args.v_cache.Read(v_index + 4 * slice_offset + 3);
      res[slice_offset] = res[slice_offset] * alpha
          + ucl::Convert<float4>(weights.x * v_val0 + weights.y * v_val1
                                 + weights.z * v_val2 + weights.w * v_val3);
    }
  }

  // Merge the partial results of all simdgroups and write the result.
  __local float4 s_acc[$2][$3];
  __local float s_max[$3];
  __local float s_sum[$3];
  for (int slice_offset = 0; slice_offset < head_slices_per_lane;
       ++slice_offset) {
    s_acc[head_dim_id + slice_offset][context_window_id] = res[slice_offset];
  }
  s_max[context_window_id] = cur_max;
  s_sum[context_window_id] = e_sum;
  ucl::SyncThreads<WorkGroup, Local>();
  // TODO: Conduct the following computation in a separate kernel.
  if (context_window_id == 0) {
    float global_max = s_max[0];
    for (int i = 1; i < group_size_for_context_window; ++i) {
      global_max = max(global_max, s_max[i]);
    }
    float denominator = 0.0f;
    for (int i = 0; i < group_size_for_context_window; ++i) {
      denominator += s_sum[i] * exp(s_max[i] - global_max);
    }
    for (int slice_offset = 0; slice_offset < head_slices_per_lane;
         ++slice_offset) {
      float4 numerator = ucl::Init<float4>(0.0f);
      for (int i = 0; i < group_size_for_context_window; ++i) {
        numerator +=
            s_acc[head_dim_id + slice_offset][i] * exp(s_max[i] - global_max);
      }
      args.dst::type result =
          ucl::Convert<args.dst::type>(numerator / denominator);
      args.dst.Write(result, head, 0, head_dim_id + slice_offset);
    }
  }
}
)",
                                   active_tokens_index, head_slices_per_lane,
                                   head_dim_slices, num_simd_groups);
  return c;
}

bool IsFlashDecodingAttentionSupported(
    const GpuInfo& gpu_info, const FlashDecodingAttentionAttributes& attr,
    std::string* absl_nonnull error) {
  if (!gpu_info.IsApiMetal()) {
    *error = "only Metal is supported";
    return false;
  }
  if (attr.num_q_heads <= 0 || attr.num_kv_heads <= 0 || attr.head_size <= 0 ||
      attr.cache_size <= 0) {
    *error = absl::StrCat(
        "num_q_heads, num_kv_heads, head_size, and cache_size must be "
        "positive, but got ",
        attr.num_q_heads, ", ", attr.num_kv_heads, ", ", attr.head_size, ", ",
        attr.cache_size);
    return false;
  }
  // TODO: Support multi-KV-heads.
  if (attr.num_kv_heads != 1) {
    *error =
        absl::StrCat("only num_kv_heads == 1 is supported, got num_kv_heads: ",
                     attr.num_kv_heads);
    return false;
  }
  if (attr.head_size > 512 || attr.head_size % 128 != 0) {
    *error = absl::StrCat(
        "head_size must be a multiple of 128 and <= 512, but got ",
        attr.head_size);
    return false;
  }
  // The kernel reduces over head dims with simd_sum and maps exactly 32 lanes
  // to dim slices, so the subgroup width must be 32.
  if (!gpu_info.IsWaveSizeEqualTo32()) {
    *error = "kernel assumes a subgroup (simdgroup) width of exactly 32";
    return false;
  }
  if (attr.cache_size % 4 != 0) {
    *error = absl::StrCat("cache_size must be divisible by 4, but got ",
                          attr.cache_size);
    return false;
  }
  if (attr.active_tokens_index < 0 ||
      attr.active_tokens_index >= LlmRuntimeParams::kTotalParamsCount) {
    *error = absl::StrCat("active_tokens_index out of range: ",
                          attr.active_tokens_index);
    return false;
  }
  return true;
}

absl::Status ValidateFlashDecodingAttentionTensors(
    const TensorDescriptor& q, const TensorDescriptor& k_cache,
    const TensorDescriptor& v_cache, const TensorDescriptor& dst,
    const FlashDecodingAttentionAttributes& attr) {
  const BHWC q_shape = q.GetBHWCShape();
  if (q_shape.b != 1 || q_shape.h != 1 || q_shape.w != attr.num_q_heads ||
      q_shape.c != attr.head_size) {
    return absl::InvalidArgumentError(absl::StrCat(
        "q must be BHWC(1, 1, num_q_heads, head_size) = (1, 1, ",
        attr.num_q_heads, ", ", attr.head_size, "), got ", ToString(q_shape)));
  }
  const BHWC dst_shape = dst.GetBHWCShape();
  if (dst_shape != q_shape) {
    return absl::InvalidArgumentError(
        absl::StrCat("dst shape ", ToString(dst_shape), " must match q shape ",
                     ToString(q_shape)));
  }
  if (q.GetDataType() != dst.GetDataType() ||
      q.GetDataType() != k_cache.GetDataType() ||
      q.GetDataType() != v_cache.GetDataType()) {
    return absl::InvalidArgumentError(
        "q, k_cache, v_cache, and dst must have the same data type");
  }
  const int64_t kv_elements = static_cast<int64_t>(attr.num_kv_heads) *
                              attr.cache_size * attr.head_size;
  for (const auto* cache : {&k_cache, &v_cache}) {
    if (cache->GetLayout() != Layout::kLinear) {
      return absl::InvalidArgumentError(
          "k_cache / v_cache must use Layout::kLinear");
    }
    if (cache->GetBHWCShape().c != kv_elements) {
      return absl::InvalidArgumentError(absl::StrCat(
          "k / v cache has ", cache->GetBHWCShape().c,
          " elements, expected num_kv_heads * cache_size * head_size = ",
          kv_elements));
    }
    // TODO: Support quantized KV caches.
    if (cache->GetDataType() != DataType::kFloat32 &&
        cache->GetDataType() != DataType::kFloat16) {
      return absl::UnimplementedError(
          "only F32 and F16 data types are supported for kv caches");
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<FlashDecodingAttention> CreateFlashDecodingAttention(
    const GpuInfo& gpu_info, const TensorDescriptor& q,
    const TensorDescriptor& k_cache, const TensorDescriptor& v_cache,
    const TensorDescriptor& dst, const FlashDecodingAttentionAttributes& attr) {
  std::string error;
  if (!IsFlashDecodingAttentionSupported(gpu_info, attr, &error)) {
    return absl::InvalidArgumentError(
        absl::StrCat("FlashDecodingAttention is not supported: ", error));
  }
  RETURN_IF_ERROR(
      ValidateFlashDecodingAttentionTensors(q, k_cache, v_cache, dst, attr));
  return FlashDecodingAttention(gpu_info, q, k_cache, v_cache, dst, attr);
}

}  // namespace ml_drift
