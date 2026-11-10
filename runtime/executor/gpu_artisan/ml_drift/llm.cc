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

#include "runtime/executor/gpu_artisan/ml_drift/llm.h"

#include <string>
#include <utility>

#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/task/buffer_desc.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_object_desc.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_operation.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/util.h"  // from @ml_drift

namespace ml_drift {

ContextData::ContextData() = default;
ContextData::~ContextData() = default;
ContextData::ContextData(ContextData&& other) = default;
ContextData& ContextData::operator=(ContextData&& other) = default;

GPUOperation CreateWriteParamsOp(const TensorDescriptor& dst,
                                 int params_count) {
  GPUOperation op;
  op.AddDstTensor("dst", dst);
  for (int i = 0; i < params_count; ++i) {
    const std::string param_name = absl::StrCat("param", i);
    if (dst.GetDataType() == DataType::kFloat32) {
      op.args_.AddFloat(param_name, 0);
    } else if (dst.GetDataType() == DataType::kInt32) {
      op.args_.AddInt(param_name, 0);
    }
  }
  if (dst.GetDataType() == DataType::kFloat32) {
    op.args_.AddFloat("zero_value", 0);
  } else if (dst.GetDataType() == DataType::kInt32) {
    op.args_.AddInt("zero_value", 0);
  }
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;

  std::string c;
  c += R"(
MAIN_FUNCTION($0) {
  int X = ucl::GetGlobalId<0>();
  int Y = ucl::GetGlobalId<1>();
  int S = ucl::GetGlobalId<2>();
  if (X != 0 || Y != 0 || S != 0) return;
  args.dst::type result;
)";
  std::string postfixes[4] = {".x", ".y", ".z", ".w"};
  for (int s = 0; s < DivideRoundUp(params_count, 4); ++s) {
    for (int ch = 0; ch < 4; ++ch) {
      std::string param_name = absl::StrCat("args.param", s * 4 + ch);
      if (s * 4 + ch >= params_count) {
        param_name = "args.zero_value";
      }
      c += "  result" + postfixes[ch] + " = " + param_name + ";\n";
    }
    c += "  args.dst.Write(result, 0, 0, " + std::to_string(s) + ");\n";
  }
  c += "}\n";
  op.code_ = std::move(c);
  return op;
}

GPUOperation CreateCopyNthToSingleOp(const GpuInfo& gpu_info,
                                     const TensorDescriptor& src,
                                     const TensorDescriptor& dst) {
  GPUOperation op;
  op.AddSrcTensor("src", src);
  op.AddDstTensor("dst", dst);
  op.args_.AddInt("nth", 0);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;

  std::string c;
  c += R"(
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
  int src_x_coord = args.nth;
  args.dst::type result = args.src.Read(src_x_coord, 0, S);
  args.dst.Write(result, 0, 0, S);
})";
  op.code_ = std::move(c);
  return op;
}

GPUOperation CreateCopyTokensOrPositionsOp(const TensorDescriptor& dst) {
  GPUOperation op;
  BufferDescriptor buffer_desc;
  buffer_desc.element_type = DataType::kInt32;
  buffer_desc.element_size = 1;
  buffer_desc.memory_type = MemoryType::kGlobal;
  op.AddSrcBuffer("src", buffer_desc);
  op.AddDstTensor("dst", dst);
  op.args_.AddInt("token_offset", 0);
  op.args_.AddInt("tokens_count", 0);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;

  std::string c;
  c += R"(
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
  args.dst::type res_value;
  res_value.x = 0;
  res_value.y = 0;
  res_value.z = 0;
  res_value.w = 0;
  int token_index = X + args.token_offset;
  if (token_index < args.tokens_count) {
    res_value.x = args.src.Read(token_index);
  }
  args.dst.Write(res_value, X, Y, S);
})";
  op.code_ = std::move(c);
  return op;
}

GPUOperation CreateCopyExternalEmbeddingsOp(const TensorDescriptor& dst) {
  GPUOperation op;
  BufferDescriptor buffer_desc;

  buffer_desc.element_type = DataType::kFloat32;
  buffer_desc.element_size = 4;
  buffer_desc.memory_type = MemoryType::kGlobal;

  op.AddSrcBuffer("src", buffer_desc);
  op.AddDstTensor("dst", dst);

  op.args_.AddInt("embedding_dimension", 0);
  op.args_.AddInt("embedding_offset", 0);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;

  std::string c;
  c += R"(
MAIN_FUNCTION($0) {
  int X = ucl::GetGlobalId<0>();
  int Y = ucl::GetGlobalId<1>();
  int S = ucl::GetGlobalId<2>();
  if (X >= args.dst.Width() || Y >= args.dst.Height() || S >= args.dst.Slices()) {
    return;
  }
  int index = S  + X * args.embedding_dimension / 4 + args.embedding_offset / 4;

  float4 embedding_value_f = args.src.Read(index);
  args.dst::type res_value;
  res_value.x = ucl::Convert<args.dst::scalar_type>(embedding_value_f.x);
  res_value.y = ucl::Convert<args.dst::scalar_type>(embedding_value_f.y);
  res_value.z = ucl::Convert<args.dst::scalar_type>(embedding_value_f.z);
  res_value.w = ucl::Convert<args.dst::scalar_type>(embedding_value_f.w);
  args.dst.Write(res_value, X, Y, S);
})";
  op.code_ = std::move(c);
  return op;
}

GPUOperation CreateCopyMediaEmbeddingsOp(const TensorDescriptor& src,
                                         const TensorDescriptor& dst) {
  GPUOperation op;
  op.AddSrcTensor("src", src);
  op.AddDstTensor("dst", dst);
  op.args_.AddInt("index", 0);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;

  std::string c;
  c += R"(
MAIN_FUNCTION($0) {
  int X = ucl::GetGlobalId<0>();
  int Y = ucl::GetGlobalId<1>();
  int S = ucl::GetGlobalId<2>();
  if (X >= args.src.Width() || Y >= args.src.Height() || S >= args.src.Slices()) {
    return;
  }
  args.dst::type res_value = ucl::Convert<args.dst::type>(args.src.Read(X, Y, S));
  args.dst.Write(res_value, X + args.index, Y, S);
})";
  op.code_ = std::move(c);
  return op;
}

GPUOperation CreateCopyTokensAndPositionsOp(
    const TensorDescriptor& dst_tokens, const TensorDescriptor& dst_positions) {
  GPUOperation op;
  BufferDescriptor buffer_desc;
  buffer_desc.element_type = DataType::kInt32;
  buffer_desc.element_size = 1;
  buffer_desc.memory_type = MemoryType::kGlobal;
  op.AddSrcBuffer("src", buffer_desc);
  op.AddSrcBuffer("src_positions", buffer_desc);

  op.AddDstTensor("dst", dst_tokens);
  op.AddDstTensor("dst_positions", dst_positions);

  op.args_.AddInt("token_offset", 0);
  op.args_.AddInt("tokens_count", 0);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;

  std::string c;
  c += R"(
MAIN_FUNCTION($0) {
  int X = ucl::GetGlobalId<0>();
  int Y = ucl::GetGlobalId<1>();
  int S = ucl::GetGlobalId<2>();
  if (X >= args.dst.Width() || Y >= args.dst.Height() || S >= args.dst.Slices()) {
    return;
  }
  args.dst::type res_value;
  res_value.x = 0;
  res_value.y = 0;
  res_value.z = 0;
  res_value.w = 0;

  args.dst_positions::type res_value2;
  res_value2.x = 0;
  res_value2.y = 0;
  res_value2.z = 0;
  res_value2.w = 0;
  int token_index = X + args.token_offset;
  if (token_index < args.tokens_count) {
    res_value.x = args.src.Read(token_index);
    args.dst::scalar_type v = ucl::Convert<args.dst::scalar_type>(args.src_positions.Read(token_index));
    res_value2.x = v;
  }
  args.dst.Write(res_value, X, Y, S);
  args.dst_positions.Write(res_value2, X, Y, S);
})";
  op.code_ = std::move(c);
  return op;
}

GPUOperation CreateCopyMasksOp(const TensorDescriptor& dst_local,
                               const TensorDescriptor& dst_global,
                               int mask_width) {
  GPUOperation op;
  BufferDescriptor buffer_desc;
  // Source is float
  buffer_desc.element_type = DataType::kFloat32;
  buffer_desc.element_size = 4;
  buffer_desc.memory_type = MemoryType::kGlobal;

  op.AddSrcBuffer("src_local", buffer_desc);
  op.AddSrcBuffer("src_global", buffer_desc);

  // Destination is bool (UINT8)
  op.AddDstTensor("dst_local", dst_local);
  op.AddDstTensor("dst_global", dst_global);

  op.args_.AddInt("mask_width", mask_width);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;

  std::string c;
  c += R"(
MAIN_FUNCTION($0) {
  int X = ucl::GetGlobalId<0>();
  int Y = ucl::GetGlobalId<1>();
  int S = ucl::GetGlobalId<2>();
  if (X >= args.dst_global.Width() || Y >= args.dst_global.Height() || S >= args.dst_global.Slices()) {
    return;
  }
  int index = S  + X * args.mask_width/4;

  // true - mark out, false - mark in
  float4 mask_value_f = args.src_global.Read(index);
  args.dst_global::type mask_value_uint;
  mask_value_uint.x = mask_value_f.x > -1e-5f ? false : true;
  mask_value_uint.y = mask_value_f.y > -1e-5f ? false : true;
  mask_value_uint.z = mask_value_f.z > -1e-5f ? false : true;
  mask_value_uint.w = mask_value_f.w > -1e-5f ? false : true;
  args.dst_global.Write(mask_value_uint, X, Y, S);

  mask_value_f = args.src_local.Read(index);
  args.dst_local::type mask_value_local_uint;
  mask_value_local_uint.x = mask_value_f.x > -1e-5f ? false : true;
  mask_value_local_uint.y = mask_value_f.y > -1e-5f ? false : true;
  mask_value_local_uint.z = mask_value_f.z > -1e-5f ? false : true;
  mask_value_local_uint.w = mask_value_f.w > -1e-5f ? false : true;
  args.dst_local.Write(mask_value_local_uint, X, Y, S);
})";
  op.code_ = std::move(c);
  return op;
}

}  // namespace ml_drift
