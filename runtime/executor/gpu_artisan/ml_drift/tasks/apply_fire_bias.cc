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

#include "runtime/executor/gpu_artisan/ml_drift/tasks/apply_fire_bias.h"

#include <string>

#include "absl/strings/str_replace.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_operation.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift

namespace ml_drift {

GPUOperation CreateApplyFireBiasOp(const GpuInfo& gpu_info,
                                   const TensorDescriptor& position,
                                   const TensorDescriptor& dst,
                                   float log_scale_factor, float threshold) {
  GPUOperation op;
  op.args_.AddInt("mask_width", dst.GetBHWCShape().w);
  op.args_.AddInt("cache_size", dst.GetBHWCShape().c);
  op.args_.AddFloat("log_scale", log_scale_factor);
  op.args_.AddFloat("threshold", threshold);
  std::string log_func_name =
      gpu_info.IsAdreno() && gpu_info.IsApiOpenCl() ? "native_log" : "log";
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
  op.code_ = absl::StrReplaceAll(
      R"(
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
    int q_pos = args.position.Read<int>(X, 0, 0).x;
    int4 expanded_q = ucl::Init<int4>(q_pos);
    int time_stamp = args.position.Read<int>(args.mask_width - 1, 0, 0).x;
    int offset = max(0, time_stamp - args.cache_size + 1);
    int4 expanded_k =
        ucl::Init<int4>(0, 1, 2, 3) + ucl::Init<int4>(S * 4 + offset);
    float4 relative_position = ucl::Convert<float4>(expanded_k - expanded_q);

    { // fire_position_pre_process
      float4 tmp = relative_position * args.log_scale;
      tmp = LOG(fabs(tmp) + ucl::Init<float4>(1.0f));
      relative_position = tmp * sign(relative_position);
    }

    float4 processed_query = ucl::Init<float4>(0.0f);
    {  // fire_position_normalizer
      float query_pos = max(ucl::Convert<float>(q_pos), args.threshold);
      float tmp = query_pos * args.log_scale;
      tmp = LOG(fabs(tmp) + 1.0f);
      processed_query = ucl::Init<float4>(tmp);
    }

    float4 relative_bias =
        relative_position / (processed_query + ucl::Init<float4>(1.0f));
    args.dst.Write(ucl::Convert<Type>(relative_bias), X, Y, S);
  }
)",
      {{"Type", ToUclDataType(dst.GetDataType(), 4)}, {"LOG", log_func_name}});
  op.AddSrcTensor("position", position);
  op.AddDstTensor("dst", dst);
  return op;
}

}  // namespace ml_drift
