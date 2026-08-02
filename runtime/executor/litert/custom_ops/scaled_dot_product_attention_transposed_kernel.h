/* Copyright 2026 The LiteRT Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LITERT_CUSTOM_OPS_SCALED_DOT_PRODUCT_ATTENTION_TRANSPOSED_KERNEL_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LITERT_CUSTOM_OPS_SCALED_DOT_PRODUCT_ATTENTION_TRANSPOSED_KERNEL_H_

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "litert/cc/litert_custom_op_kernel.h"  // from @litert
#include "litert/cc/litert_expected.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/framework/threadpool.h"

namespace litert {
namespace lm {


class ScaledDotProductAttentionTransposedKernel : public CustomOpKernel {
 public:
  ScaledDotProductAttentionTransposedKernel();
  ~ScaledDotProductAttentionTransposedKernel() override = default;

  const std::string& OpName() const override { return op_name_; }
  int OpVersion() const override { return 1; }

  Expected<void> Init(const void* init_data, size_t init_data_size) override;

  Expected<void> GetOutputLayouts(const std::vector<Layout>& input_layouts,
                                  std::vector<Layout>& output_layouts) override;

  Expected<void> Run(const std::vector<TensorBuffer>& inputs,
                     std::vector<TensorBuffer>& outputs) override;

  Expected<void> Destroy() override;

 private:
  const std::string op_name_ = "litert_custom_op.sdpa_transposed";

  std::optional<float> softcap_;
  int k_ts_idx_ = 2;
  int v_ts_idx_ = 3;

  ThreadPool pool_;
};

}  // namespace lm
}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LITERT_CUSTOM_OPS_SCALED_DOT_PRODUCT_ATTENTION_TRANSPOSED_KERNEL_H_
