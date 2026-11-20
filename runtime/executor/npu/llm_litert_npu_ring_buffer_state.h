// Copyright 2026 Google LLC.
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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_NPU_LLM_LITERT_NPU_RING_BUFFER_STATE_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_NPU_LLM_LITERT_NPU_RING_BUFFER_STATE_H_

#include <memory>
#include <string>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/executor/litert/legacy_map_state.h"
#include "runtime/executor/npu/llm_litert_npu_compiled_model_executor_utils.h"

namespace litert::lm {

// Saves and restores the ring buffer KV cache of the NPU executor's sliding
// window (local) layers, for `LlmLiteRtNpuCompiledModelExecutor::SaveContext` /
// `RestoreContext`.
//
// The entries of a ring buffer are overwritten by the steps one ring buffer
// later, so rolling back past a wrap-around needs a saved copy. The global KV
// cache is position indexed (rolled back steps are overwritten by the next
// steps) and is not saved. The same holds for local layers whose KV cache is as
// long as the global one (where the sliding window is only applied by the
// attention mask).
//
// Usage workflow:
// 1) `Save(env, layout, kv_cache_buffers)` returns host memory copies of the
//    ring buffers, e.g. in `SaveContext()`.
// 2) `Restore(saved, layout, kv_cache_buffers)` copies them back, e.g. in
//    `RestoreContext(ProcessedContext)` after rolling back past a wrap-around.
struct RingBufferState {
  // Returns the names of the ring buffers among `kv_cache_buffers`: the local
  // KV cache buffers that are physically shorter than the global KV cache.
  static absl::flat_hash_set<std::string> GetNames(
      const NpuGraphLayout& layout,
      const absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          kv_cache_buffers);

  // Returns host memory copies of the ring buffers among `kv_cache_buffers`
  // (none for models without a ring buffer).
  static absl::StatusOr<std::unique_ptr<LegacyMapState>> Save(
      ::litert::Environment& env, const NpuGraphLayout& layout,
      const absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          kv_cache_buffers);

  // Copies the ring buffers saved by `Save` back into `kv_cache_buffers`. The
  // saved buffers have to be exactly the ring buffers of `layout`, with the
  // same shapes (e.g. a context saved before the executor switched to a context
  // size with a ring buffer can't restore it). Checks this before copying
  // anything, so `kv_cache_buffers` are unchanged if it fails.
  static absl::Status Restore(
      const LegacyMapState& saved, const NpuGraphLayout& layout,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          kv_cache_buffers);
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_NPU_LLM_LITERT_NPU_RING_BUFFER_STATE_H_
