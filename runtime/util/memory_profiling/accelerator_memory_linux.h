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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_MEMORY_PROFILING_ACCELERATOR_MEMORY_LINUX_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_MEMORY_PROFILING_ACCELERATOR_MEMORY_LINUX_H_

// Building blocks of the Linux implementation of the platform contract in
// accelerator_memory.h. Internal to memory_profiling: exposed only so
// that accelerator_memory_linux_test.cc can exercise them without a GPU.

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/util/memory_profiling/accelerator_memory.h"
#include "runtime/util/memory_profiling/accelerator_memory_nvml.h"
#include "runtime/util/memory_profiling/fdinfo.h"

namespace litert::lm {

// One DRM client of this process: what one open handle on a GPU device node
// has allocated, as the kernel reports it in that descriptor's fdinfo.
struct DrmClientMemory {
  // Kernel driver, e.g. "amdgpu", "i915", "xe", "msm".
  std::string driver;
  // PCI address of the GPU, e.g. "0000:03:00.0"; empty for non-PCI GPUs.
  std::string pdev;
  // Identifies the client within the driver; several descriptors can refer to
  // the same client (dup(), fork()).
  std::string client_id;
  // One memory region of the GPU ("vram", "gtt", "cpu", "system0", ...).
  struct Region {
    std::string name;
    // Bytes allocated in the region: buffers evicted to make room are still
    // counted, as are buffers shared with other clients.
    uint64_t total_bytes = 0;
    // Bytes of those buffers that are in the region right now. Equal to
    // `total_bytes` when the driver does not report residency.
    uint64_t resident_bytes = 0;
  };
  // The client's memory per region, ordered by region name.
  std::vector<Region> regions;
};

// Parses the DRM part of one descriptor's fdinfo, or returns std::nullopt if
// the descriptor is not a DRM client. Allocated memory comes from the
// `drm-total-<region>` keys, falling back to the older `drm-memory-<region>`
// spelling, resident memory from `drm-resident-<region>`; a client whose
// driver reports no memory has no `regions`.
std::optional<DrmClientMemory> ParseDrmFdinfo(const FdinfoFields& fields);

// Reads this process's DRM clients from `<proc_self_dir>/fdinfo`, counting
// each client once however many descriptors refer to it. `proc_self_dir` is
// "/proc/self" in production.
std::vector<DrmClientMemory> ReadDrmClients(absl::string_view proc_self_dir);

// Aggregates DRM clients into an AcceleratorMemoryUsage: present memory is the
// resident bytes and attributed memory the allocated bytes, each summed over
// all clients and regions; the breakdown is the resident bytes per region and
// the allocations the per-client detail ("<driver> <pdev> client <id>
// (<region>)", largest first, zero entries omitted). None of it is counted by
// the host (see the file comment in accelerator_memory_linux.cc). Returns
// std::nullopt when there are clients but none reports memory, i.e. the
// kernel or driver predates per-process memory statistics; no clients at all
// means no GPU in use, reported as zero.
std::optional<AcceleratorMemoryUsage> AcceleratorMemoryUsageFromDrmClients(
    const std::vector<DrmClientMemory>& clients);

// Combines the two Linux sources: DRM clients (every GPU driver but NVIDIA's)
// and NVML devices (NVIDIA; std::nullopt when NVML is unavailable). NVIDIA
// framebuffer memory is added under the region "vram", with one allocation
// per GPU ("<name> #<index> (vram)"). Tracking is unsupported only when DRM
// reports clients without memory statistics and NVML is unavailable, i.e.
// a GPU is in use and nothing can measure it; DRM clients without statistics
// next to an available NVML are taken to be NVIDIA's, which NVML covers.
std::optional<AcceleratorMemoryUsage> AcceleratorMemoryUsageFromLinuxSources(
    const std::vector<DrmClientMemory>& drm_clients,
    const std::optional<std::vector<NvmlDeviceMemory>>& nvml_devices);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_MEMORY_PROFILING_ACCELERATOR_MEMORY_LINUX_H_
