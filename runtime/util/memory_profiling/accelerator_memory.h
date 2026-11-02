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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_MEMORY_PROFILING_ACCELERATOR_MEMORY_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_MEMORY_PROFILING_ACCELERATOR_MEMORY_H_

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace litert::lm {

// Accelerator (GPU / NPU) memory used by the current process at a point in
// time, in the two senses a MemorySnapshot (memory_snapshot.h) reports memory
// in: physically present now, and charged to the process by the platform.
//
// Only memory attributed to the calling process is counted; memory used by
// other processes on the same accelerator is never included.
struct AcceleratorMemoryUsage {
  // Accelerator memory physically present right now, in bytes: in VRAM or in
  // system RAM, not evicted, swapped or merely reserved. The accelerator part
  // of "memory resident in RAM".
  uint64_t total_bytes = 0;
  // Accelerator memory the platform charges to the process, in bytes:
  // allocated / committed, present or not. The accelerator part of "memory
  // attributed to the process by the OS". Equal to `total_bytes` where the
  // platform makes no such distinction.
  uint64_t attributed_bytes = 0;
  // The part of `total_bytes` that the OS also counts in the process's own
  // resident figure (HostMemoryUsage::resident_bytes): accelerator buffers
  // mapped into the process and touched by the CPU. A snapshot takes it out of
  // the CPU side so that CPU + accelerator counts each page once. 0 where
  // there is no such overlap or it cannot be measured.
  uint64_t resident_counted_by_host_bytes = 0;
  // The part of `attributed_bytes` that is already inside the memory the OS
  // attributes to the process (HostMemoryUsage::platform_attributed_bytes):
  // all of it on Windows (the commit charge includes GPU allocations) and on
  // Apple platforms (the physical footprint includes Metal buffers), none on
  // Linux. Taken out of the CPU side the same way.
  uint64_t attributed_counted_by_host_bytes = 0;
  // Optional breakdown of `total_bytes` into platform-specific categories, as
  // (category name, bytes) pairs. Where a breakdown is available, the entries
  // sum to `total_bytes`. Empty when the platform offers no breakdown.
  std::vector<std::pair<std::string, uint64_t>> breakdown;
  // Optional per-allocation detail, as (label, bytes) pairs ordered largest
  // first, for diagnostic logging. Empty when the platform does not expose
  // individual allocations.
  std::vector<std::pair<std::string, uint64_t>> allocations;
};

// -----------------------------------------------------------------------------
// Platform contract
//
// Implemented once per operating system, in accelerator_memory_<os>.cc; the
// build picks the file for the target OS and accelerator_memory_unsupported.cc
// covers every other OS. Each platform file documents at its top what it
// measures and what it cannot see. Nothing else in this header or in
// accelerator_memory.cc is platform-specific.
// -----------------------------------------------------------------------------

// Returns the accelerator memory currently used by this process, or
// std::nullopt when accelerator memory tracking is not supported on this
// platform.
std::optional<AcceleratorMemoryUsage> GetProcessAcceleratorMemoryUsage();

// -----------------------------------------------------------------------------
// Platform-neutral helpers
// -----------------------------------------------------------------------------

// Converts bytes to mebibytes. Used for host memory as well, so that every
// megabyte figure in this directory is computed the same way.
float BytesToMb(uint64_t bytes);

// Formats `usage.breakdown` as "name=1.00 MB, other=2.00 MB". Returns an empty
// string when there is no breakdown.
std::string FormatAcceleratorMemoryBreakdown(
    const AcceleratorMemoryUsage& usage);

// Formats `usage.allocations` for diagnostic logging, one line per allocation
// of at least 1 MiB, in the order they appear in `usage.allocations`.
std::vector<std::string> FormatAcceleratorMemoryAllocations(
    const AcceleratorMemoryUsage& usage);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_MEMORY_PROFILING_ACCELERATOR_MEMORY_H_
