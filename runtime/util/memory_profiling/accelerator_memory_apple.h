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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_MEMORY_PROFILING_ACCELERATOR_MEMORY_APPLE_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_MEMORY_PROFILING_ACCELERATOR_MEMORY_APPLE_H_

// Building blocks of the macOS and iOS implementation of the platform
// contract in accelerator_memory.h (accelerator_memory_apple.mm). Internal to
// memory_profiling: exposed only so that accelerator_memory_apple_test.mm can
// exercise the rule that combines the two sources on fixed inputs. See the top
// of accelerator_memory_apple.mm for what the sources are and why.

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "runtime/util/memory_profiling/accelerator_memory_monitor.h"

namespace litert::lm {

// The accelerator part of the kernel's memory ledger for this process: the
// bytes of phys_footprint (HostMemoryUsage::platform_attributed_bytes) that
// the kernel attributes to the GPU and to the Neural Engine. Read from
// task_info(TASK_VM_INFO), fields ledger_tag_graphics_footprint,
// ledger_tag_neural_footprint and their compressed counterparts.
struct AppleAcceleratorLedger {
  // Pages in RAM.
  uint64_t graphics_bytes = 0;
  uint64_t neural_bytes = 0;
  // Pages the kernel has compressed: still charged to the process, not in
  // RAM at their size.
  uint64_t graphics_compressed_bytes = 0;
  uint64_t neural_compressed_bytes = 0;
};

// Reads the ledger. std::nullopt if the kernel does not report it (task_info
// fails or predates the per-tag ledger fields).
std::optional<AppleAcceleratorLedger> ReadAppleAcceleratorLedger();

// Metal's own figure for this process's resources on one GPU.
struct MetalDeviceMemory {
  // Product name, e.g. "Apple M3 Pro".
  std::string name;
  // MTLDevice.currentAllocatedSize: every Metal resource this process has
  // created on the device, including buffers wrapped over the process's own
  // memory without a copy (newBufferWithBytesNoCopy), which is how ML Drift's
  // Metal backend hands the mmap()ed model file to the GPU.
  uint64_t allocated_bytes = 0;
};

// Reads Metal's figure for every GPU. Enumerates the devices on first call,
// creating the process's Metal device objects if nothing else has (well under
// a megabyte, once). Empty if there is no Metal device.
std::vector<MetalDeviceMemory> ReadMetalAllocatedMemory();

// Combines the two sources into an AcceleratorMemoryUsage: present memory is
// the ledger's pages in RAM plus whatever Metal has allocated beyond the whole
// ledger (the mapped weights); attributed memory is the whole ledger,
// compressed pages included, all of which phys_footprint already contains.
// The breakdown is {"graphics", "neural", "mapped weights"} with zero entries
// omitted; the allocations are one "<name> (Metal)" entry per GPU, largest
// first, zero or not. See the top of accelerator_memory_apple.mm for the rule
// and why.
AcceleratorMemoryUsage AcceleratorMemoryUsageFromAppleSources(
    const AppleAcceleratorLedger& ledger,
    const std::vector<MetalDeviceMemory>& devices);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_MEMORY_PROFILING_ACCELERATOR_MEMORY_APPLE_H_
