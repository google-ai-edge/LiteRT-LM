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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_MEMORY_PROFILING_ACCELERATOR_MEMORY_ANDROID_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_MEMORY_PROFILING_ACCELERATOR_MEMORY_ANDROID_H_

// Building blocks of the Android implementation of the platform contract in
// accelerator_memory.h (accelerator_memory_android.cc). Internal to
// memory_profiling: exposed, and built on Linux as well, so that
// accelerator_memory_android_sources_test.cc can exercise the rule that
// combines the sources on fixed inputs without a device. See the top of
// accelerator_memory_android.cc for what the sources are and why.

#include <cstdint>
#include <optional>
#include <vector>

#include "runtime/util/memory_profiling/accelerator_memory.h"
#include "runtime/util/memory_profiling/accelerator_memory_dmabuf.h"
#include "runtime/util/memory_profiling/proc_mappings.h"

namespace litert::lm {

// What the process's own address space says about GPU driver memory: the
// bytes mapped from each kind of GPU device node, and whether a Mali GPU is
// in use.
struct GpuDriverMappings {
  // Bytes mapped from Qualcomm Adreno's KGSL driver, which backs each
  // allocation with a shmem file listed as "/kgsl-3d0 (deleted)" and maps it
  // into the process at allocation time, so this is the process's GPU memory
  // (measured within 0.05% of Android's own memtrack figure).
  uint64_t kgsl_bytes = 0;
  // Bytes mapped from /dev/dri/renderD* (DRM render nodes: PowerVR on Pixel
  // 10/11, Samsung Xclipse on Exynos). Likewise every allocation is mapped
  // (within 0.7% of memtrack on PowerVR; Exynos has no memtrack to compare).
  uint64_t drm_bytes = 0;
  // Whether /dev/mali* is mapped, i.e. an Arm Mali GPU through the kbase
  // driver is in use. kbase reserves virtual address space well beyond what
  // it allocates, so its mapped bytes say nothing; its memory is found
  // through the resident counters instead (see AndroidResidentView).
  bool has_mali = false;
};

// Classifies the process's mappings.
GpuDriverMappings SummarizeGpuDriverMappings(
    const std::vector<ProcMapping>& mappings);

// GPU memory that Mali's kbase driver has added to the process's VmRSS
// without mapping it: VmRSS minus what smaps_rollup finds really mapped.
// Zero when smaps_rollup could not be read, or when nothing was added. Both
// figures must come from the same instant; the CPU heap moving between two
// reads would otherwise be mistaken for GPU memory.
uint64_t UnmappedRssBytes(const ProcResidentCounters& status,
                          std::optional<uint64_t> smaps_rss_bytes);

// The resident-memory inputs of the combination rule.
struct AndroidResidentView {
  // From /proc/self/status, read on every sample.
  ProcResidentCounters status;
  // UnmappedRssBytes() as of the last address-space walk: the Mali pool. 0
  // for other GPUs.
  uint64_t unmapped_rss_bytes = 0;
};

// Combines the three sources into the accelerator figure. See the top of
// accelerator_memory_android.cc for the rule and why.
AcceleratorMemoryUsage AcceleratorMemoryUsageFromAndroidSources(
    const DmaBufSummary& dmabuf, const GpuDriverMappings& driver,
    const AndroidResidentView& resident);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_MEMORY_PROFILING_ACCELERATOR_MEMORY_ANDROID_H_
