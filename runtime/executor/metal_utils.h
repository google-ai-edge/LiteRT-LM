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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_METAL_UTILS_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_METAL_UTILS_H_

#include <cstddef>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert

namespace litert::lm {

#if defined(__APPLE__)
// Extract Metal command queue from LiteRT Environment options.
void* GetMetalCommandQueue(litert::Environment& env);

// Attempt to copy KV cache buffers on GPU. Checks if both buffers are Metal.
// Returns true if copy succeeded, false if skipped (not Metal or missing
// queue), or an error status if copy failed due to internal errors.
absl::StatusOr<bool> TryCopyKvCacheMetal(const litert::TensorBuffer& src_buffer,
                                         litert::TensorBuffer& dst_buffer,
                                         int src_index_to_copy_on_prefill,
                                         int decode_batch_size,
                                         void* command_queue);
#endif

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_METAL_UTILS_H_
