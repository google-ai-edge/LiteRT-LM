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

#include "runtime/executor/gpu_artisan/ml_drift/tasks/mul_with_overflow.h"

#include <string>
#include <utility>

#include "absl/strings/substitute.h"  // from @com_google_absl
#include "ml_drift/common/task/gpu_operation.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift

namespace ml_drift {

std::string MulWithOverflowCode(const std::string& a, const std::string& b,
                                const std::string& c) {
  return absl::Substitute(R"(
  {
    uint a_uint = ucl::Reinterpret<int, uint>($0);
    uint b_uint = ucl::Reinterpret<int, uint>($1);
    uint alo = a_uint & 0x0000FFFF;
    uint ahi = (a_uint >> 16u) & 0x0000FFFF;
    uint blo = b_uint & 0x0000FFFF;
    uint bhi = (b_uint >> 16u) & 0x0000FFFF;
    uint clo = alo * blo;
    uint carry = (clo >> 16u) & 0x0000FFFF;
    clo = clo & 0x0000FFFF;
    uint chi1 = alo * bhi;
    chi1 = chi1 & 0x0000FFFF;
    uint chi2 = ahi * blo;
    chi2 = chi2 & 0x0000FFFF;
    uint chi = chi1 + chi2 + carry;
    chi = chi & 0x0000FFFF;
    uint c_uint = (chi << 16u) | clo;
    $2 = ucl::Reinterpret<uint, int>(c_uint);
  }
)",
                          a, b, c);
}

GPUOperation CreateMulWithOverflow(const TensorDescriptor& desc, int value) {
  ElementwiseDescriptor op_desc;
  op_desc.args.AddInt("second_value", value);
  op_desc.code +=
      MulWithOverflowCode("in_value.x", "args.second_value", "out_value.x");
  op_desc.code +=
      MulWithOverflowCode("in_value.y", "args.second_value", "out_value.y");
  op_desc.code +=
      MulWithOverflowCode("in_value.z", "args.second_value", "out_value.z");
  op_desc.code +=
      MulWithOverflowCode("in_value.w", "args.second_value", "out_value.w");
  OperationDef definition;
  definition.src_tensors.push_back(desc);
  definition.dst_tensors.push_back(desc);
  return CreateGpuOperation(definition, std::move(op_desc));
}

}  // namespace ml_drift
