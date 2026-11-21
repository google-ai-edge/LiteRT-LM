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

#include "runtime/executor/gpu_artisan/ml_drift/tasks/apply_attention_mask.h"

#include <algorithm>
#include <optional>

#include "absl/strings/substitute.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/task/buffer_desc.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_operation.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "runtime/executor/gpu_artisan/ml_drift/runtime_params.h"

namespace ml_drift {

GPUOperation CreateApplyAttentionMaskOp(
    const TensorDescriptor& params_desc, const TensorDescriptor& mask_desc,
    std::optional<int> sliding_window, bool sliding_local_cache,
    const TensorDescriptor* block_indices_desc) {
  GPUOperation op;
  op.args_.AddInt("window_size", sliding_window.has_value()
                                     ? std::max(*sliding_window, 0)
                                     : 0);
  op.args_.AddInt("sliding_local_cache", sliding_local_cache ? 1 : 0);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;

  op.code_ = absl::Substitute(R"(
  MAIN_FUNCTION($$0) {
  int X = ucl::GetGlobalId<0>();
  int Y = ucl::GetGlobalId<1>();
  int S = ucl::GetGlobalId<2>();
  if (X >= args.mask_tensor.Width() || Y >= args.mask_tensor.Height() || S >= args.mask_tensor.Slices()) {
    return;
  }
  int token_index_offset = args.params.Read($0);
  args.mask_tensor::scalar_type temps[4];
  for (int i = 0; i < 4; ++i) {
)",
                              LlmRuntimeParams::kTokenOffsetIndex);
  if (sliding_local_cache) {
    op.code_ +=
        "    int token_index = S * 4 + i - min(token_index_offset, "
        "args.window_size - 1);\n";
  } else {
    op.code_ += "    int token_index = S * 4 + i - token_index_offset;\n";
  }
  op.code_ += "    temps[i] = token_index <= X ? false : true;\n";
  if (sliding_window.has_value()) {
    op.code_ +=
        "    temps[i] = token_index > X - args.window_size ? temps[i] : "
        "true;\n";
  }
  if (sliding_local_cache) {
    op.code_ +=
        "    temps[i] = token_index_offset + token_index < 0 ? true : "
        "temps[i];\n";
  }
  if (block_indices_desc != nullptr) {
    // block_indices holds the block of each token of the current chunk, so the
    // key is looked up by its position relative to the chunk (token_index),
    // not by its absolute position.
    op.code_ += R"(
    int q_block = args.block_indices.Read(X, Y, 0, 0).x;
    if (q_block > 0) {
      if (token_index >= 0 && token_index < args.block_indices.Width()) {
        int kv_block = args.block_indices.Read(token_index, Y, 0, 0).x;
        if (kv_block == q_block) {
          temps[i] = false;
        }
      }
    }
)";
  }
  op.code_ += R"(
  }
  args.mask_tensor::type mask_value;
  mask_value.x = temps[0];
  mask_value.y = temps[1];
  mask_value.z = temps[2];
  mask_value.w = temps[3];
  args.mask_tensor.Write(mask_value, X, Y, S);
})";
  BufferDescriptor params_buffer;
  params_buffer.element_type = DataType::kInt32;
  params_buffer.element_size = 1;
  op.AddSrcBuffer("params", params_buffer);
  if (block_indices_desc != nullptr) {
    op.AddSrcTensor("block_indices", *block_indices_desc);
  }
  op.AddDstTensor("mask_tensor", mask_desc);
  return op;
}

}  // namespace ml_drift
