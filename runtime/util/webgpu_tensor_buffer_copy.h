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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_WEBGPU_TENSOR_BUFFER_COPY_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_WEBGPU_TENSOR_BUFFER_COPY_H_

#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert

namespace litert::lm {

// TODO: b/582556349 - Replace this with the LiteRT device copy API once LiteRT
// provides one, instead of maintaining the WebGPU copy in LiteRT-LM (see
// `DeviceTensorBufferCopyFn` in tensor_buffer_util.h).
//
// Copies the contents of a WebGPU-backed `TensorBuffer` `src` into `dst` on the
// GPU using a `wgpu::CommandEncoder::CopyBufferToBuffer` command submitted to
// the WebGPU queue stored in `env`, avoiding a host staging round trip.
//
// Linking `:webgpu_tensor_buffer_copy` automatically registers this function
// with
// `RegisterDeviceTensorBufferCopyFn(DeviceTensorBufferCopyBackend::kWebGpu,
// ...)` so `CopyTensorBuffer` and `CopyTensorBufferContents` use it for WebGPU
// buffers.
//
// Callers must ensure `src` and `dst` were allocated on `env`'s `WGPUDevice`.
// No CPU wait is performed: ordering relies on WebGPU's single in-order queue
// per device, so later work submitted to that queue (the GPU delegate's next
// dispatch, `Lock()` read-backs) observes the copied contents.
//
// args:
//   env: LiteRT environment holding `kWebGpuDevice` and optional `kWebGpuQueue`
//     options.
//   dst: destination WebGPU tensor buffer.
//   src: source WebGPU tensor buffer with the same buffer type, element type,
//     dimensions, and packed size.
// returns: true once the device-to-device copy command has been submitted,
//   false when the buffers cannot be copied via `CopyBufferToBuffer` (for
//   example texture-backed tensors, missing WebGPU device option, unaligned
//   ranges, missing CopySrc/CopyDst usage, or non-identical sub-ranges of the
//   same underlying `wgpu::Buffer`) so the caller can fall back to a host copy,
//   or an error status if tensor types or buffer sizes mismatch or
//   synchronization fails.
absl::StatusOr<bool> CopyWebGpuTensorBuffer(const litert::Environment& env,
                                            litert::TensorBuffer& dst,
                                            const litert::TensorBuffer& src);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_WEBGPU_TENSOR_BUFFER_COPY_H_
