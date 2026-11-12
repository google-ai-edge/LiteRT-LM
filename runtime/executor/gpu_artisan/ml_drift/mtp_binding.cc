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

#include "runtime/executor/gpu_artisan/ml_drift/mtp_binding.h"

#include <utility>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "runtime/executor/gpu_artisan/ml_drift/speculative_shared_state.h"

namespace odml::infra::gpu {

absl::StatusOr<std::pair<int, int>> MtpSharedKvLayerIndices(
    int main_kv_shared_start_layer) {
  if (main_kv_shared_start_layer < 2) {
    return absl::InvalidArgumentError(
        absl::StrCat("main_kv_shared_start_layer must be >= 2, got: ",
                     main_kv_shared_start_layer));
  }
  return std::make_pair(main_kv_shared_start_layer - 2,
                        main_kv_shared_start_layer - 1);
}

absl::Status BindMtpDrafterExternals(const MtpDrafterGraphHandles& g,
                                     const MtpSharedInputs& in,
                                     ml_drift::CreateGpuModelInfo* info) {
  if (info == nullptr) {
    return absl::InvalidArgumentError("info cannot be null.");
  }
  if (g.key_cache.size() != 2 || g.value_cache.size() != 2) {
    return absl::InvalidArgumentError(
        "MtpDrafterGraphHandles must have 2 key_cache and 2 value_cache "
        "handles.");
  }
  const absl::StatusOr<std::pair<int, int>> indices =
      MtpSharedKvLayerIndices(in.kv_shared_start_layer);
  if (!indices.ok()) {
    return indices.status();
  }
  const auto [local_idx, global_idx] = *indices;

  if (local_idx < 0 || local_idx >= static_cast<int>(in.key_cache.size()) ||
      global_idx >= static_cast<int>(in.key_cache.size()) ||
      local_idx >= static_cast<int>(in.value_cache.size()) ||
      global_idx >= static_cast<int>(in.value_cache.size())) {
    return absl::FailedPreconditionError(absl::StrFormat(
        "Shared KV cache has %zu key layers and %zu value layers, but required "
        "indices are local=%d, global=%d.",
        in.key_cache.size(), in.value_cache.size(), local_idx, global_idx));
  }
  if (in.key_cache[local_idx] == nullptr ||
      in.key_cache[global_idx] == nullptr ||
      in.value_cache[local_idx] == nullptr ||
      in.value_cache[global_idx] == nullptr) {
    return absl::InvalidArgumentError(
        "Shared KV cache entries for local and global layers must not be "
        "null.");
  }
  if (in.params_i32 == nullptr || in.tokens_ids == nullptr ||
      in.last_accepted_hidden_state == nullptr) {
    return absl::InvalidArgumentError(
        "params_i32, tokens_ids, and last_accepted_hidden_state cannot be "
        "null.");
  }

  auto& ext = info->external_immutable_tensors;
  ext[g.key_cache[0].id] = in.key_cache[local_idx];
  ext[g.key_cache[1].id] = in.key_cache[global_idx];
  ext[g.value_cache[0].id] = in.value_cache[local_idx];
  ext[g.value_cache[1].id] = in.value_cache[global_idx];
  ext[g.params_i32.id] = in.params_i32;
  ext[g.tokens_ids.id] = in.tokens_ids;
  ext[g.input_activations.id] = in.last_accepted_hidden_state;
  return absl::OkStatus();
}

}  // namespace odml::infra::gpu
