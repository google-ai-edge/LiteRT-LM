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

#include "runtime/util/webgpu_tensor_buffer_copy.h"

#include <cstddef>
#include <cstdint>
#include <variant>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "ml_drift/webgpu/spatial_tensor.h"  // from @ml_drift
#include "ml_drift/webgpu/webgpu_headers.h"  // from @ml_drift
#include "litert/c/litert_common.h"  // from @litert
#include "litert/cc/litert_any.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_environment_options.h"  // from @litert
#include "litert/cc/litert_event.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/cc/litert_tensor_buffer_types.h"  // from @litert
#include "runtime/util/tensor_buffer_util.h"

namespace litert::lm {
namespace {

template <typename HandleT>
HandleT ExtractHandleFromVariant(const ::litert::LiteRtVariant& variant) {
  if (const auto* int_val = std::get_if<int64_t>(&variant)) {
    return reinterpret_cast<HandleT>(static_cast<intptr_t>(*int_val));
  }
  if (const auto* ptr_val = std::get_if<void*>(&variant)) {
    return reinterpret_cast<HandleT>(*ptr_val);
  }
  if (const auto* const_ptr_val = std::get_if<const void*>(&variant)) {
    return reinterpret_cast<HandleT>(const_cast<void*>(*const_ptr_val));
  }
  return nullptr;
}

absl::Status WaitTensorBufferEventIfPresent(const ::litert::TensorBuffer& buf) {
  if (!buf.HasEvent()) {
    return absl::OkStatus();
  }
  LITERT_ASSIGN_OR_RETURN(::litert::Event event, buf.GetEvent());
  LITERT_RETURN_IF_ERROR(event.Wait(-1));
  return absl::OkStatus();
}

struct WebGpuTensorBufferCopyRegistrar {
  WebGpuTensorBufferCopyRegistrar() {
    RegisterDeviceTensorBufferCopyFn(DeviceTensorBufferCopyBackend::kWebGpu,
                                     &CopyWebGpuTensorBuffer);
  }
};

const WebGpuTensorBufferCopyRegistrar kWebGpuTensorBufferCopyRegistrar{};

}  // namespace

absl::StatusOr<bool> CopyWebGpuTensorBuffer(const ::litert::Environment& env,
                                            ::litert::TensorBuffer& dst,
                                            const ::litert::TensorBuffer& src) {
#if !LITERT_HAS_WEBGPU_SUPPORT
  return false;
#else
  if (dst.Get() == src.Get()) {
    return true;
  }
  if (!dst.IsWebGpuMemory() || !src.IsWebGpuMemory()) {
    return false;
  }

  LITERT_ASSIGN_OR_RETURN(const ::litert::TensorBufferType dst_buffer_type,
                          dst.BufferType());
  LITERT_ASSIGN_OR_RETURN(const ::litert::TensorBufferType src_buffer_type,
                          src.BufferType());
  if (dst_buffer_type != src_buffer_type) {
    return false;
  }

  LITERT_ASSIGN_OR_RETURN(const ::litert::RankedTensorType dst_type,
                          dst.TensorType());
  LITERT_ASSIGN_OR_RETURN(const ::litert::RankedTensorType src_type,
                          src.TensorType());
  if (dst_type.ElementType() != src_type.ElementType() ||
      dst_type.Layout().Dimensions() != src_type.Layout().Dimensions()) {
    return absl::InvalidArgumentError(
        "Cannot copy WebGPU TensorBuffer contents between mismatched tensor "
        "types.");
  }

  LITERT_ASSIGN_OR_RETURN(const size_t dst_packed_size, dst.PackedSize());
  LITERT_ASSIGN_OR_RETURN(const size_t src_packed_size, src.PackedSize());
  if (dst_packed_size != src_packed_size) {
    return absl::InvalidArgumentError(
        absl::StrCat("Cannot copy WebGPU TensorBuffer contents between "
                     "buffers of different packed sizes: dst=",
                     dst_packed_size, " vs src=", src_packed_size, "."));
  }

  auto env_options = env.GetOptions();
  if (!env_options) {
    return false;
  }

  auto wgpu_device_opt =
      env_options->GetOption(::litert::EnvironmentOptions::Tag::kWebGpuDevice);
  if (!wgpu_device_opt) {
    return false;
  }
  // `WGPUDevice` / `WGPUQueue` are provided by `webgpu_headers.h` (via Dawn's
  // `webgpu_cpp.h`) rather than including Dawn's C header directly.
  WGPUDevice raw_device =  // NOLINT(misc-include-cleaner)
      ExtractHandleFromVariant<WGPUDevice>(*wgpu_device_opt);
  if (raw_device == nullptr) {
    return false;
  }

  auto dst_handle = dst.GetWebGpuBuffer();
  auto src_handle = src.GetWebGpuBuffer();
  if (!dst_handle || !src_handle || *dst_handle == nullptr ||
      *src_handle == nullptr) {
    return false;
  }

  const auto* dst_tensor =
      reinterpret_cast<const ::ml_drift::webgpu::SpatialTensor*>(*dst_handle);
  const auto* src_tensor =
      reinterpret_cast<const ::ml_drift::webgpu::SpatialTensor*>(*src_handle);
  if (dst_tensor == src_tensor) {
    return true;
  }

  wgpu::Buffer dst_buffer = dst_tensor->GetBufferHandle();
  wgpu::Buffer src_buffer = src_tensor->GetBufferHandle();
  if (!dst_buffer || !src_buffer) {
    // Texture-backed WebGPU tensors fall back to locked host copy.
    return false;
  }

  const uint64_t tensor_memory_size = src_tensor->GetMemorySizeInBytes();
  if (dst_tensor->GetMemorySizeInBytes() != tensor_memory_size) {
    return absl::InvalidArgumentError(
        absl::StrCat("WebGPU SpatialTensor memory size mismatch: dst=",
                     dst_tensor->GetMemorySizeInBytes(),
                     " vs src=", tensor_memory_size, "."));
  }
  uint64_t copy_size = tensor_memory_size;
  if (src_buffer_type == ::litert::TensorBufferType::kWebGpuBufferPacked) {
    // Packed buffers hold `PackedSize()` bytes rounded up to 4, the range
    // `LiteRtLockWebGpuMemory()` / `LiteRtUnlockWebGpuMemory()` read and write.
    // `GetMemorySizeInBytes()` is the PHWC4-padded size, which over-copies and
    // can exceed the size of imported packed buffers.
    const uint64_t packed_copy_size = (src_packed_size + 3) / 4 * 4;
    if (packed_copy_size > tensor_memory_size) {
      return false;
    }
    copy_size = packed_copy_size;
  }
  if (copy_size == 0) {
    return true;
  }

  const uint64_t dst_offset = dst_tensor->offset();
  const uint64_t src_offset = src_tensor->offset();
  if (dst_buffer.Get() == src_buffer.Get()) {
    // WebGPU validation forbids CopyBufferToBuffer when source and destination
    // are the same WGPUBuffer, even for non-overlapping sub-ranges.
    return dst_offset == src_offset;
  }

  // CopyBufferToBuffer requires 4-byte aligned offsets and sizes, CopySrc /
  // CopyDst buffer usages, and ranges within the underlying WGPUBuffer bounds.
  if (copy_size % 4 != 0 || dst_offset % 4 != 0 || src_offset % 4 != 0 ||
      !(src_buffer.GetUsage() & wgpu::BufferUsage::CopySrc) ||
      !(dst_buffer.GetUsage() & wgpu::BufferUsage::CopyDst) ||
      src_offset > src_buffer.GetSize() ||
      copy_size > src_buffer.GetSize() - src_offset ||
      dst_offset > dst_buffer.GetSize() ||
      copy_size > dst_buffer.GetSize() - dst_offset) {
    return false;
  }

  ABSL_RETURN_IF_ERROR(WaitTensorBufferEventIfPresent(src));
  ABSL_RETURN_IF_ERROR(WaitTensorBufferEventIfPresent(dst));

  wgpu::Device device(raw_device);
  WGPUQueue raw_queue = nullptr;  // NOLINT(misc-include-cleaner)
  auto wgpu_queue_opt =
      env_options->GetOption(::litert::EnvironmentOptions::Tag::kWebGpuQueue);
  if (wgpu_queue_opt) {
    raw_queue = ExtractHandleFromVariant<WGPUQueue>(*wgpu_queue_opt);
  }
  wgpu::Queue queue =
      raw_queue != nullptr ? wgpu::Queue(raw_queue) : device.GetQueue();
  if (!queue) {
    return false;
  }

  wgpu::CommandEncoder command_encoder = device.CreateCommandEncoder();
  command_encoder.CopyBufferToBuffer(src_buffer, src_offset, dst_buffer,
                                     dst_offset, copy_size);
  wgpu::CommandBuffer command_buffer = command_encoder.Finish();
  // The copy is intentionally not waited on here. WebGPU has a single queue per
  // device, `kWebGpuQueue` is the queue the GPU delegate and the WebGPU buffer
  // handlers submit to, and submissions on a queue execute in order with
  // implicit hazard synchronization. So the delegate's next dispatch reading
  // `dst` and any `Lock()` read-back of `dst` (which submits a read-back to the
  // same queue and waits for it) are ordered after this copy. Buffers used by
  // the submitted copy are kept alive by Dawn until the copy completes.
  queue.Submit(1, &command_buffer);
  return true;
#endif  // !LITERT_HAS_WEBGPU_SUPPORT
}

}  // namespace litert::lm
