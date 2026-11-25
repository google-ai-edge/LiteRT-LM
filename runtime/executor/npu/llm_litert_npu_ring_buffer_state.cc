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

#include "runtime/executor/npu/llm_litert_npu_ring_buffer_state.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/executor/litert/legacy_map_state.h"
#include "runtime/executor/npu/llm_litert_npu_compiled_model_executor_utils.h"
#include "runtime/util/tensor_buffer_util.h"

namespace litert::lm {
namespace {

absl::StatusOr<std::vector<int>> GetDimensions(
    const ::litert::TensorBuffer& buffer) {
  LITERT_ASSIGN_OR_RETURN(auto type, buffer.TensorType());
  const auto dims = type.Layout().Dimensions();
  return std::vector<int>(dims.begin(), dims.end());
}

}  // namespace

absl::flat_hash_set<std::string> RingBufferState::GetNames(
    const NpuGraphLayout& layout,
    const absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        kv_cache_buffers) {
  absl::flat_hash_set<std::string> names;
  if (!layout.uses_ringbuffer) {
    return names;
  }
  for (const auto& [name, buffer] : kv_cache_buffers) {
    auto it = layout.kv_cache_buffers.find(name);
    if (it == layout.kv_cache_buffers.end() || !it->second.is_local ||
        it->second.min_sequence_length <= 0 ||
        it->second.min_sequence_length >= layout.global_cache_length) {
      continue;
    }
    names.insert(std::string(name));
  }
  return names;
}

absl::StatusOr<std::unique_ptr<LegacyMapState>> RingBufferState::Save(
    ::litert::Environment& env, const NpuGraphLayout& layout,
    const absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        kv_cache_buffers) {
  absl::flat_hash_map<std::string, ::litert::TensorBuffer> saved;
  for (const std::string& name : GetNames(layout, kv_cache_buffers)) {
    // Host memory is less scarce than the device memory some NPU backends
    // allocate KV cache buffers in.
    ABSL_ASSIGN_OR_RETURN(auto copy, CopyTensorBufferToHostMemory(
                                         env, kv_cache_buffers.at(name)));
    // Not `operator[]`, which would default-construct a `TensorBuffer` (and
    // with it LiteRT's default environment).
    saved.insert_or_assign(name, std::move(copy));
  }
  return std::make_unique<LegacyMapState>(std::move(saved));
}

absl::Status RingBufferState::Restore(
    const LegacyMapState& saved, const NpuGraphLayout& layout,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        kv_cache_buffers) {
  const absl::flat_hash_set<std::string> names =
      GetNames(layout, kv_cache_buffers);
  if (names.size() != saved.buffers().size()) {
    return absl::FailedPreconditionError(absl::StrCat(
        "The saved context has ", saved.buffers().size(),
        " ring buffers, but the KV cache has ", names.size(), "."));
  }
  for (const auto& [name, saved_buffer] : saved.buffers()) {
    if (!names.contains(name)) {
      return absl::FailedPreconditionError(
          absl::StrCat("Saved ring buffer ", name,
                       " is not a ring buffer of the KV cache."));
    }
    ABSL_ASSIGN_OR_RETURN(std::vector<int> saved_dims,
                          GetDimensions(saved_buffer));
    ABSL_ASSIGN_OR_RETURN(std::vector<int> dims,
                          GetDimensions(kv_cache_buffers.at(name)));
    if (saved_dims != dims) {
      return absl::FailedPreconditionError(absl::StrCat(
          "Saved ring buffer ", name, " has shape [",
          absl::StrJoin(saved_dims, ","), "], but the KV cache has shape [",
          absl::StrJoin(dims, ","), "]."));
    }
  }
  for (const auto& [name, saved_buffer] : saved.buffers()) {
    // Also checks the element type, strides and sizes of both buffers.
    ABSL_RETURN_IF_ERROR(
        CopyTensorBufferContents(kv_cache_buffers.at(name), saved_buffer));
  }
  return absl::OkStatus();
}

}  // namespace litert::lm
