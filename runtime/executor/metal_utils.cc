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

#if defined(__APPLE__)
#include "runtime/executor/metal_utils.h"

#include <dlfcn.h>

#include <cstddef>
#include <variant>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/cc/litert_environment_options.h"  // from @litert

namespace litert::lm {

typedef int (*CopyKvCacheMetalFn)(void*, void*, int, int, size_t, size_t,
                                  void*);
static CopyKvCacheMetalFn g_copy_kv_cache_metal = []() -> CopyKvCacheMetalFn {
  return reinterpret_cast<CopyKvCacheMetalFn>(
      dlsym(RTLD_DEFAULT, "LiteRtCopyKvCacheMetal"));
}();

void* GetMetalCommandQueue(litert::Environment& env) {
  auto env_options_or = env.GetOptions();
  if (env_options_or.HasValue()) {
    auto queue_option_or = env_options_or.Value().GetOption(
        litert::EnvironmentOptions::Tag::kMetalCommandQueue);
    if (queue_option_or.HasValue()) {
      const auto& val = queue_option_or.Value();
      if (std::holds_alternative<const void*>(val)) {
        return const_cast<void*>(std::get<const void*>(val));
      } else if (std::holds_alternative<void*>(val)) {
        return std::get<void*>(val);
      }
    }
  }
  return nullptr;
}

int CopyKvCacheMetal(void* src_buffer_ptr, void* dst_buffer_ptr,
                     int src_index_to_copy_on_prefill, int decode_batch_size,
                     size_t src_buffer_size, size_t dst_buffer_size,
                     void* command_queue) {
  if (g_copy_kv_cache_metal == nullptr) {
    return -1;
  }
  return g_copy_kv_cache_metal(src_buffer_ptr, dst_buffer_ptr,
                               src_index_to_copy_on_prefill, decode_batch_size,
                               src_buffer_size, dst_buffer_size, command_queue);
}

absl::StatusOr<bool> TryCopyKvCacheMetal(const litert::TensorBuffer& src_buffer,
                                         litert::TensorBuffer& dst_buffer,
                                         int src_index_to_copy_on_prefill,
                                         int decode_batch_size,
                                         void* command_queue) {
  if (!g_copy_kv_cache_metal || !command_queue || !src_buffer.IsMetalMemory() ||
      !dst_buffer.IsMetalMemory()) {
    return false;
  }

  auto src_metal_buffer_or = src_buffer.GetMetalBuffer();
  auto dst_metal_buffer_or = dst_buffer.GetMetalBuffer();
  if (!src_metal_buffer_or.HasValue() || !dst_metal_buffer_or.HasValue()) {
    return absl::InternalError("Failed to retrieve Metal buffer handles");
  }

  auto src_size_expected = src_buffer.PackedSize();
  auto dst_size_expected = dst_buffer.PackedSize();
  if (!src_size_expected.HasValue() || !dst_size_expected.HasValue()) {
    return absl::InternalError("Failed to retrieve buffer packed sizes");
  }

  int ret = CopyKvCacheMetal(
      src_metal_buffer_or.Value(), dst_metal_buffer_or.Value(),
      src_index_to_copy_on_prefill, decode_batch_size,
      src_size_expected.Value(), dst_size_expected.Value(), command_queue);
  if (ret != 0) {
    ABSL_LOG(WARNING) << "Metal GPU KV cache copy failed with " << ret;
    return false;
  }
  return true;
}

}  // namespace litert::lm
#endif
