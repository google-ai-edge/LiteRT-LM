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

// macOS and iOS implementation of the platform contract in
// accelerator_memory.h.
//
// What is measured, from two sources. Both are per process; neither sees
// other processes' GPU use. The behaviour described here was measured with
// accelerator_memory_apple_metal_test on an Apple M5 Pro and on Forge-on-Mac.
//
//   - The kernel's ledger (accelerator_memory_apple.h,
//     task_info(TASK_VM_INFO)): the part of phys_footprint, the memory the
//     platform attributes to the process (host_memory_apple.cc), that the
//     kernel tags as graphics or neural. A Metal buffer the CPU has written
//     (StorageModeShared) or the GPU has written (StorageModePrivate) is in
//     here at its size plus some driver overhead (300 MB for a 256 MB private
//     buffer on the M5 Pro), compressed pages included and counted apart.
//   - Metal's own figure, MTLDevice.currentAllocatedSize per GPU: every
//     resource this process has created on the device, including buffers
//     created over the process's memory without a copy, which is how ML
//     Drift's Metal backend (gpu_backend_metal.mm) hands LiteRT-LM's weights,
//     mapped from the model file, to the GPU. The kernel charges those to
//     nobody: clean file-backed pages are not in phys_footprint, and once the
//     GPU has read them through a MAP_PRIVATE mapping they leave the
//     process's resident set too. Metal's figure is the only one that sees
//     them.
//
// From these, the four figures of the contract:
//
//   total_bytes (present)   ledger graphics + neural pages in RAM, plus what
//                           Metal has allocated beyond the whole ledger: the
//                           resources phys_footprint does not include, in
//                           practice the weights mapped from the model file.
//   attributed_bytes        the whole ledger, compressed pages included: the
//                           GPU part of phys_footprint. The mapped weights are
//                           not in it, because the OS charges nobody for them
//                           and the attributed figure is meant to match what
//                           Activity Monitor shows.
//   attributed_counted_by_host_bytes
//                           all of attributed_bytes: phys_footprint already
//                           contains the ledger, so a MemorySnapshot files it
//                           under the GPU and shows phys_footprint unchanged.
//   resident_counted_by_host_bytes
//                           0; see below.
//   breakdown               graphics, neural (ledger pages in RAM) and
//                           "mapped weights" (the Metal remainder); sums to
//                           total_bytes.
//   allocations             one per GPU, Metal's figure for that GPU.
//
// Approximations, in order of likely size:
//   - Mapped weights are taken to be present in full. Pages the kernel has
//     evicted (a model larger than RAM) still count, because the only query
//     that could tell (mach_vm_region_recurse / mach_vm_page_range_query over
//     the mapping) walks every page of a multi-gigabyte mapping on every
//     sample.
//   - A Metal buffer the CPU has touched (a shared buffer it wrote, GPU output
//     it read back, weights it re-reads after the GPU took them over) is in
//     the process's resident count as well as here, and the resident figure
//     of a MemorySnapshot shows it on both sides. The kernel exposes nothing
//     public that says which GPU pages are in this process's page tables: its
//     region and page queries describe the VM object, which the GPU's own
//     writes fill too (a shared buffer only the GPU has written shows all
//     its pages "resident" while the process's resident count has not
//     moved), so subtracting them would hide the CPU's own memory behind
//     GPU-written buffers instead. The attributed figure is exact either way.
//   - Driver overhead (in the ledger, not in Metal's figure) is netted against
//     the mapped weights (in Metal's figure, not in the ledger), so with both
//     present the weights read low by the overhead.
//
// What is not seen:
//   - Memory of a GPU the process uses through something other than Metal
//     (OpenCL on an Intel Mac) is in the ledger, not in Metal's figure.
//   - The ledger lags a release by some milliseconds: the driver frees
//     asynchronously. Metal's figure drops at once. Peaks are unaffected.
// Reading Metal's figure creates the process's Metal device objects if
// nothing else has; that costs well under a megabyte, once.

#import <Metal/Metal.h>
#include <TargetConditionals.h>
#include <mach/mach.h>

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "runtime/util/memory_profiling/accelerator_memory.h"
#include "runtime/util/memory_profiling/accelerator_memory_apple.h"

namespace litert::lm {

namespace {

// The process's Metal devices, enumerated once; GPUs do not come and go
// (external GPUs on Intel Macs aside, which are ignored here).
NSArray<id<MTLDevice>>* GetMetalDevices() {
  static NSArray<id<MTLDevice>>* const devices = []() -> NSArray<id<MTLDevice>>* {
#if TARGET_OS_OSX
    NSArray<id<MTLDevice>>* all = MTLCopyAllDevices();
    if (all != nil && all.count > 0) return all;
#endif
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) return @[];
    return @[ device ];
  }();
  return devices;
}

}  // namespace

std::optional<AppleAcceleratorLedger> ReadAppleAcceleratorLedger() {
  task_vm_info_data_t info = {};
  mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
  if (task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&info), &count) !=
      KERN_SUCCESS) {
    return std::nullopt;
  }
  // The per-tag ledger fields arrived with revision 3 of the struct; an older
  // kernel fills fewer words and leaves them zero, which would read as "no
  // GPU memory" rather than "unknown".
  if (count < TASK_VM_INFO_REV3_COUNT) return std::nullopt;
  const auto non_negative = [](int64_t bytes) {
    return bytes > 0 ? static_cast<uint64_t>(bytes) : 0;
  };
  AppleAcceleratorLedger ledger;
  ledger.graphics_bytes = non_negative(info.ledger_tag_graphics_footprint);
  ledger.graphics_compressed_bytes = non_negative(info.ledger_tag_graphics_footprint_compressed);
  ledger.neural_bytes = non_negative(info.ledger_tag_neural_footprint);
  ledger.neural_compressed_bytes = non_negative(info.ledger_tag_neural_footprint_compressed);
  return ledger;
}

std::vector<MetalDeviceMemory> ReadMetalAllocatedMemory() {
  std::vector<MetalDeviceMemory> devices;
  for (id<MTLDevice> device in GetMetalDevices()) {
    devices.push_back({.name = device.name.UTF8String,
                       .allocated_bytes = static_cast<uint64_t>(device.currentAllocatedSize)});
  }
  return devices;
}

AcceleratorMemoryUsage AcceleratorMemoryUsageFromAppleSources(
    const AppleAcceleratorLedger& ledger, const std::vector<MetalDeviceMemory>& devices) {
  AcceleratorMemoryUsage usage;
  uint64_t metal_bytes = 0;
  for (const MetalDeviceMemory& device : devices) {
    metal_bytes += device.allocated_bytes;
    usage.allocations.emplace_back(absl::StrCat(device.name, " (Metal)"), device.allocated_bytes);
  }
  // What phys_footprint charges to the GPU and the Neural Engine, in RAM or
  // compressed.
  const uint64_t charged_bytes = ledger.graphics_bytes + ledger.graphics_compressed_bytes +
                                 ledger.neural_bytes + ledger.neural_compressed_bytes;
  // Metal resources beyond that: the weights mapped from the model file (see
  // the file comment), which the OS charges to nobody.
  const uint64_t mapped_weights_bytes =
      metal_bytes > charged_bytes ? metal_bytes - charged_bytes : 0;

  usage.total_bytes = ledger.graphics_bytes + ledger.neural_bytes + mapped_weights_bytes;
  usage.attributed_bytes = charged_bytes;
  usage.attributed_counted_by_host_bytes = charged_bytes;
  usage.resident_counted_by_host_bytes = 0;  // Not measurable; file comment.
  if (ledger.graphics_bytes > 0) {
    usage.breakdown.emplace_back("graphics", ledger.graphics_bytes);
  }
  if (ledger.neural_bytes > 0) {
    usage.breakdown.emplace_back("neural", ledger.neural_bytes);
  }
  if (mapped_weights_bytes > 0) {
    usage.breakdown.emplace_back("mapped weights", mapped_weights_bytes);
  }
  return usage;
}

std::optional<AcceleratorMemoryUsage> GetProcessAcceleratorMemoryUsage() {
  const std::optional<AppleAcceleratorLedger> ledger = ReadAppleAcceleratorLedger();
  const std::vector<MetalDeviceMemory> devices = ReadMetalAllocatedMemory();
  // Without either source there is nothing to measure with: no Metal device
  // and a kernel without the ledger.
  if (!ledger.has_value() && devices.empty()) return std::nullopt;
  return AcceleratorMemoryUsageFromAppleSources(ledger.value_or(AppleAcceleratorLedger()), devices);
}

}  // namespace litert::lm
