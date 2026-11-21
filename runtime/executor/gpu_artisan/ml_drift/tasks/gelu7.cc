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

#include "runtime/executor/gpu_artisan/ml_drift/tasks/gelu7.h"

#include <array>
#include <utility>

#include "absl/strings/str_replace.h"  // from @com_google_absl
#include "absl/strings/substitute.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_operation.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift

namespace ml_drift {

GPUOperation CreateGELU7Op(const TensorDescriptor& desc,
                           const std::array<float, 7>& anchors,
                           const std::array<float, 7>& slopes) {
  ElementwiseDescriptor op_desc;
  op_desc.code = "  Type linear_approx = ucl::Init<Type>(0.0f);\n";
  for (int i = 0; i < 7; ++i) {
    op_desc.code += absl::Substitute(
        "  linear_approx += max(in_value - ucl::Init<Type>($0), "
        "ucl::Init<Type>(0.0f)) * ucl::Init<Type>($1);\n",
        anchors[i], slopes[i]);
  }
  op_desc.code += "  out_value = linear_approx;\n";
  absl::StrReplaceAll({{"Type", ToUclDataType(desc.GetDataType(), 4)}},
                      &op_desc.code);

  OperationDef defn;
  defn.src_tensors.push_back(desc);
  defn.dst_tensors.push_back(desc);
  return CreateGpuOperation(defn, std::move(op_desc));
}

}  // namespace ml_drift
