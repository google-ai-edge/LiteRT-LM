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
#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LITERT_CUSTOM_OPS_ATTENTION_UTILS_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LITERT_CUSTOM_OPS_ATTENTION_UTILS_H_

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace ruy {
class Context;
}  // namespace ruy

namespace litert {
namespace lm {

struct TensorRef {
  const float* data;
  std::vector<int32_t> shape;
};

struct MutableTensorRef {
  float* data;
  std::vector<int32_t> shape;
};

// Computes scaled dot product attention with transposed key and value.
// q_shape is [1, bk, gt, h]
// k_shape is [1, c, s, h] (k_ts_idx == 2) or [1, c, h, s] (k_ts_idx == 3)
// v_shape is [1, c, h, s] (v_ts_idx == 3) or [1, c, s, h] (v_ts_idx == 2)
// mask_shape is [1, 1, t, s] (where t divides gt)
// output_shape is [1, bk, gt, h]
bool ComputeTransposedAttention(
    const TensorRef& query, const TensorRef& key, const TensorRef& value,
    const TensorRef& mask, std::optional<float> logit_cap, int k_ts_idx,
    int v_ts_idx, const MutableTensorRef& output,
    std::optional<int32_t> active_seq_len = std::nullopt);

bool ComputeTransposedAttentionSingleHead(
    int h_idx, const TensorRef& query, const TensorRef& key,
    const TensorRef& value, const TensorRef& mask,
    std::optional<float> logit_cap, int k_ts_idx, int v_ts_idx,
    const MutableTensorRef& output, std::optional<int32_t> active_seq_len,
    std::optional<std::pair<int32_t, int32_t>> query_range = std::nullopt,
    int32_t max_ruy_threads = 1, float* logits_scratch_ptr = nullptr,
    float* out_scratch_ptr = nullptr, ruy::Context* ruy_context = nullptr);

}  // namespace lm
}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LITERT_CUSTOM_OPS_ATTENTION_UTILS_H_
