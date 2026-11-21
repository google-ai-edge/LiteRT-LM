// Copyright 2025 The ODML Authors.
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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_TENSOR_BUFFER_UTIL_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_TENSOR_BUFFER_UTIL_H_

#include <cstddef>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert

namespace litert::lm {

// Returns the number of dimensions that are greater than 1 in the given
// tensor buffer.
absl::StatusOr<int> NumSignificantDims(
    const litert::TensorBuffer& tensor_buffer);

// Returns the dimensions of the given tensor buffer.
absl::StatusOr<std::vector<int>> TensorBufferDims(
    const ::litert::TensorBuffer& tensor_buffer);

// Creates a deep copy of the given tensor buffer. Device buffers are copied on
// the device when supported (e.g. WebGPU, OpenCL, or Metal buffers when linked
// with the matching copy handler), otherwise through locked host memory.
//
// args:
//   env: LiteRT environment used to allocate the copy and look up device/queue
//     handles.
//   tensor_buffer: source tensor buffer to copy.
// returns: newly allocated TensorBuffer with identical buffer type, size, and
//   contents, or an error status on failure.
absl::StatusOr<litert::TensorBuffer> CopyTensorBuffer(
    litert::Environment& env, const litert::TensorBuffer& tensor_buffer);

// Creates a deep copy of the given tensor buffer backed by host memory
// (`TensorBufferType::kHostMemory`), compacting any trailing hardware buffer
// padding beyond the locked tensor payload.
absl::StatusOr<litert::TensorBuffer> CopyTensorBufferToHostMemory(
    litert::Environment& env, const litert::TensorBuffer& tensor_buffer);

// Copies the locked tensor payload from `src` into `dst` in place using host
// memory locks. `src` and `dst` must have matching element type and dimensions;
// linear memory buffers must also have matching strides. Prefer the overload
// taking `litert::Environment` when `dst` and `src` may be GPU device buffers.
//
// args:
//   dst: destination tensor buffer updated in place.
//   src: source tensor buffer to copy from.
// returns: OK on success, or an error if tensor types, strides, or byte spans
//   mismatch or if locking fails.
absl::Status CopyTensorBufferContents(litert::TensorBuffer& dst,
                                      const litert::TensorBuffer& src);

// Copies the tensor payload from `src` into `dst` in place without reallocating
// `dst`. Device buffers of the same type are copied directly on the device
// using handles from `env` when supported (e.g. WebGPU buffers when linked with
// the WebGPU copy handler), otherwise through locked host memory. `src` and
// `dst` must have matching element type and dimensions; linear memory buffers
// must also have matching strides.
//
// args:
//   env: LiteRT environment owning the device and queue handles.
//   dst: destination tensor buffer updated in place.
//   src: source tensor buffer to copy from.
// returns: OK on success, or an error if tensor types, strides, or byte spans
//   mismatch or if copying fails.
absl::Status CopyTensorBufferContents(const litert::Environment& env,
                                      litert::TensorBuffer& dst,
                                      const litert::TensorBuffer& src);

// TODO: b/582556349 - LiteRT has no public API to copy `TensorBuffer`
// contents on the device, so LiteRT-LM implements device copies itself with
// native GPU handles from `litert::Environment`. Once LiteRT provides a device
// copy API (e.g. `TensorBuffer::CopyContentsFrom()`), switch
// `CopyTensorBuffer` and `CopyTensorBufferContents` to it and delete this
// registry and the backend implementations (`:webgpu_tensor_buffer_copy`)
// instead of maintaining GPU-to-GPU copies in LiteRT-LM.
//
// Callback that copies the contents of `src` into `dst` on the device owned by
// `env` without a round trip through host memory.
//
// args:
//   env: LiteRT environment owning the device and command queue.
//   dst: destination device tensor buffer.
//   src: source device tensor buffer of the same buffer type and layout.
// returns: true when the device copy succeeded, false when the caller should
//   fall back to a locked host-memory copy, or an error status on failure.
using DeviceTensorBufferCopyFn = absl::StatusOr<bool> (*)(
    const litert::Environment& env, litert::TensorBuffer& dst,
    const litert::TensorBuffer& src);

// GPU memory backends that can register a `DeviceTensorBufferCopyFn`.
enum class DeviceTensorBufferCopyBackend {
  kWebGpu,
  kOpenCl,
  kMetal,
};

// Registers (or clears when `copy_fn` is nullptr) the callback used by
// `CopyTensorBuffer` and `CopyTensorBufferContents` to copy `TensorBuffer`s of
// the given GPU memory `backend` on the device without locking host staging
// memory. Backend implementations (e.g. `:webgpu_tensor_buffer_copy`) register
// themselves when linked.
//
// args:
//   backend: GPU memory backend whose buffers `copy_fn` copies.
//   copy_fn: device copy callback, or nullptr to disable.
// returns: the previously registered callback for `backend` (or nullptr if
//   none was set).
DeviceTensorBufferCopyFn RegisterDeviceTensorBufferCopyFn(
    DeviceTensorBufferCopyBackend backend, DeviceTensorBufferCopyFn copy_fn);

// Struct to hold a TensorBuffer and a boolean indicating if the buffer is
// wrapped around existing memory or if it owns the memory.
struct MaybeWrappedTensorBuffer {
  TensorBuffer buffer;
  bool wrapped;
};

// Wraps the given host memory with a TensorBuffer if possible, otherwise
// creates a new managed TensorBuffer.
template <typename T>
absl::StatusOr<MaybeWrappedTensorBuffer> WrapOrCreateTensorBufferFromHostMemory(
    RankedTensorType tensor_type, absl::Span<T> data) {
  size_t size = data.size() * sizeof(T);
  // First try to wrap the memory with a TensorBuffer.
  auto wrapped_buffer =
      TensorBuffer::CreateFromHostMemory(tensor_type, data.data(), size);
  if (wrapped_buffer.HasValue()) {
    return MaybeWrappedTensorBuffer{.buffer = std::move(*wrapped_buffer),
                                    .wrapped = true};
  }

  LITERT_ASSIGN_OR_RETURN(const size_t packed_size, tensor_type.Bytes());
  LITERT_ASSIGN_OR_RETURN(
      auto new_buffer,
      TensorBuffer::CreateManagedHostMemory(tensor_type, packed_size));
  return MaybeWrappedTensorBuffer{.buffer = std::move(new_buffer),
                                  .wrapped = false};
}

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_TENSOR_BUFFER_UTIL_H_
