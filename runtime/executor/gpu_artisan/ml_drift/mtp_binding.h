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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_MTP_BINDING_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_MTP_BINDING_H_

#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/tensor_handle.h"  // from @ml_drift
#include "runtime/executor/gpu_artisan/ml_drift/speculative_shared_state.h"

namespace odml::infra::gpu {

struct MtpDrafterGraphHandles {
  ml_drift::TensorHandle input{};
  ml_drift::TensorHandle tokens_ids{};
  ml_drift::TensorHandle params_i32{};
  ml_drift::TensorHandle input_activations{};
  ml_drift::TensorHandle output{};
  ml_drift::TensorHandle output_activations{};
  std::vector<ml_drift::TensorHandle> key_cache;    // size 2: [local, global]
  std::vector<ml_drift::TensorHandle> value_cache;  // size 2: [local, global]
};

// {kv_shared_start_layer - 2 (local), kv_shared_start_layer - 1 (global)};
// errors if < 2.
absl::StatusOr<std::pair<int, int>> MtpSharedKvLayerIndices(
    int main_kv_shared_start_layer);

// Binds external immutable tensors for the MTP drafter model into `info` using
// graph handles `g` and shared state from `in`.
// Local and global KV cache layers are mapped using
// MtpSharedKvLayerIndices(in.kv_shared_start_layer).
absl::Status BindMtpDrafterExternals(const MtpDrafterGraphHandles& g,
                                     const MtpSharedInputs& in,
                                     ml_drift::CreateGpuModelInfo* info);

template <typename CacheT>
void PopulateMtpCacheTensors(const CacheT& cache, MtpSharedInputs* out) {
  out->key_cache.clear();
  out->key_cache.reserve(cache.key_cached_tensors.size());
  for (const auto& k : cache.key_cached_tensors) {
    out->key_cache.push_back(k.get());
  }
  out->value_cache.clear();
  out->value_cache.reserve(cache.value_cached_tensors.size());
  for (const auto& v : cache.value_cached_tensors) {
    out->value_cache.push_back(v.get());
  }
}

}  // namespace odml::infra::gpu

namespace ml_drift {
// Re-exported into ml_drift for compatibility with legacy LLM graph builders.
using ::odml::infra::gpu::BindMtpDrafterExternals;
using ::odml::infra::gpu::MtpDrafterGraphHandles;
using ::odml::infra::gpu::MtpSharedKvLayerIndices;
}  // namespace ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_MTP_BINDING_H_
