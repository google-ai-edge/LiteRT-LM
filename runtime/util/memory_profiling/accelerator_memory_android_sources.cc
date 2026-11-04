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

// The platform-independent part of the Android implementation: how the
// sources combine. accelerator_memory_android.cc reads them on a device; the
// rule itself is here so that it can be unit tested on Linux.

#include <algorithm>
#include <cstdint>
#include <optional>
#include <tuple>
#include <vector>

#include "absl/strings/match.h"  // from @com_google_absl
#include "runtime/util/memory_profiling/accelerator_memory_android.h"
#include "runtime/util/memory_profiling/accelerator_memory_dmabuf.h"
#include "runtime/util/memory_profiling/accelerator_memory_monitor.h"
#include "runtime/util/memory_profiling/proc_mappings.h"

namespace litert::lm {

GpuDriverMappings SummarizeGpuDriverMappings(
    const std::vector<ProcMapping>& mappings) {
  GpuDriverMappings driver;
  for (const ProcMapping& mapping : mappings) {
    // KGSL backs each allocation with its own shmem file, which the kernel
    // lists as "/kgsl-3d0 (deleted)" rather than as the device node.
    if (absl::StrContains(mapping.path, "kgsl-3d0")) {
      driver.kgsl_bytes += mapping.size_bytes();
    } else if (absl::StartsWith(mapping.path, "/dev/dri/renderD")) {
      driver.drm_bytes += mapping.size_bytes();
    } else if (absl::StartsWith(mapping.path, "/dev/mali")) {
      driver.has_mali = true;
    }
  }
  return driver;
}

uint64_t UnmappedRssBytes(const ProcResidentCounters& status,
                          std::optional<uint64_t> smaps_rss_bytes) {
  if (!smaps_rss_bytes.has_value() || *smaps_rss_bytes >= status.vm_rss_bytes) {
    return 0;
  }
  return status.vm_rss_bytes - *smaps_rss_bytes;
}

AcceleratorMemoryUsage AcceleratorMemoryUsageFromAndroidSources(
    const DmaBufSummary& dmabuf, const GpuDriverMappings& driver,
    const AndroidResidentView& resident) {
  AcceleratorMemoryUsage usage = AcceleratorMemoryUsageFromDmaBuf(dmabuf);

  // The GPU driver's own pool, by the mechanism the driver uses (file
  // comment of accelerator_memory_android.cc). One breakdown entry and one
  // allocation line per driver in use, so the headline stays short.
  struct DriverPool {
    const char* label;
    const char* detail;
    uint64_t bytes;
  };
  const uint64_t mali_bytes = driver.has_mali ? resident.unmapped_rss_bytes : 0;
  const DriverPool pools[] = {
      {"gpu driver (kgsl)", "GPU driver pool, KGSL shmem mappings (/kgsl-3d0)",
       driver.kgsl_bytes},
      {"gpu driver (drm)", "GPU driver pool, mapped from /dev/dri/renderD*",
       driver.drm_bytes},
      {"gpu driver (mali)", "GPU driver pool, counted in VmRSS by kbase",
       mali_bytes},
  };
  for (const DriverPool& pool : pools) {
    if (pool.bytes == 0) continue;
    usage.breakdown.emplace_back(pool.label, pool.bytes);
    usage.allocations.emplace_back(pool.detail, pool.bytes);
    usage.total_bytes += pool.bytes;
  }
  // None of it is swappable and Linux has no commit charge: present and
  // charged are the same figure.
  usage.attributed_bytes = usage.total_bytes;

  // What the host's resident figure (VmRSS) already contains of this:
  //   - the Mali pool in full, since kbase put it there (that is how it was
  //     found);
  //   - of the KGSL pool, the pages the CPU has touched. KGSL backs each
  //     allocation with a shmem file, so those pages are RssShmem (measured:
  //     the kgsl-3d0 mappings' Rss equals RssShmem on a Galaxy S25); capped at
  //     the pool in case the process uses shmem for something else.
  //   dma-buf and DRM render-node pages are mapped by physical frame number
  //   and are in no RSS counter whether or not the CPU touched them (measured
  //   on Mali, PowerVR, Adreno and Xclipse devices), so they contribute
  //   nothing here.
  const uint64_t touched_bytes =
      std::min(resident.status.rss_shmem_bytes, driver.kgsl_bytes);
  usage.resident_counted_by_host_bytes = mali_bytes + touched_bytes;
  usage.attributed_counted_by_host_bytes = usage.resident_counted_by_host_bytes;

  // Largest first; ties broken by name so the output is deterministic.
  const auto larger_first = [](const auto& a, const auto& b) {
    return std::tie(b.second, a.first) < std::tie(a.second, b.first);
  };
  std::sort(usage.breakdown.begin(), usage.breakdown.end(), larger_first);
  std::sort(usage.allocations.begin(), usage.allocations.end(), larger_first);
  return usage;
}

}  // namespace litert::lm
