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

// Android implementation of the platform contract in
// accelerator_memory_monitor.h.
//
// Android has no per-process accelerator memory API an app may call: the
// memtrack HAL behind `dumpsys meminfo` is reserved to the system server, and
// the GPU drivers' own nodes (/sys/kernel/debug/mali0, /sys/class/kgsl/.../
// proc/<pid>) are root-only or SELinux-restricted. Everything here is read
// from the process's own /proc files, which are always readable. Three
// sources, because the GPU vendors account for their memory in three
// different ways (measured on a Pixel 9 Pro, a Galaxy S25 Ultra, a Pixel 11
// Pro and an Exynos Galaxy S24; see the comments at each):
//
//   1. dma-buf heap buffers (accelerator_memory_dmabuf.h): the
//      AHardwareBuffers through which LiteRT-LM's NPU backends, and some GPU
//      backends, allocate. Read from /proc/self/fdinfo. On the S25 Ultra's
//      Hexagon NPU this is the whole working set (915 MB for Gemma 4 E2B), and
//      the only source that sees it: memtrack reports 0 for it.
//
//   2. The GPU driver's pool, for drivers that map every allocation into the
//      process (accelerator_memory_android.h, GpuDriverMappings): Qualcomm's
//      KGSL (listed in /proc/self/maps as "/kgsl-3d0 (deleted)", one shmem
//      file per allocation) and the DRM render nodes used by PowerVR and
//      Samsung's Xclipse (/dev/dri/renderD*). The bytes mapped are the
//      process's GPU memory, within 2% of memtrack where memtrack has a
//      figure (Samsung's reports none for Xclipse). Neither driver reports
//      memory through DRM fdinfo (accelerator_memory_linux.cc's source), so
//      the Linux reader does not apply on Android.
//
//   3. The GPU driver's pool, for Arm Mali through kbase (/dev/mali*). kbase
//      maps its memory by physical frame number, invisible to /proc/self/maps
//      and smaps, but adds every page to the process's VmRSS counter so that
//      the low-memory killer sees it. The pool is therefore VmRSS (which has
//      it) minus smaps_rollup's Rss (which does not): 1241 MB on the Pixel 9
//      Pro against memtrack's 1254 MB, and 0 on every non-Mali device
//      measured.
//
// From these, the four figures of the contract
// (AcceleratorMemoryUsageFromAndroidSources):
//
//   total_bytes (present) = dma-bufs + mapped pools + Mali pool. None of it
//                           is swappable and Linux has no commit charge, so
//   attributed_bytes      = total_bytes.
//   resident_counted_by_host_bytes
//                         = the Mali pool (kbase put it in VmRSS, which is how
//                           it was found) + the KGSL pages the CPU has
//                           touched. KGSL backs each allocation with a shmem
//                           file, so those pages are RssShmem (118 MB of
//                           1238 MB on a Galaxy S25, where the kgsl-3d0
//                           mappings' Rss equals RssShmem). dma-buf and DRM
//                           render-node pages are mapped by physical frame
//                           number and are in no RSS counter, touched or not
//                           (a 256 MB AHardwareBuffer the CPU wrote moved no
//                           counter on any of the four devices), so they add
//                           nothing here.
//   attributed_counted_by_host_bytes = the same.
//   breakdown             = per dma-buf heap, plus one "gpu driver (<node>)"
//                           entry per driver in use.
//
// Approximations, documented so that they can be revisited:
//   - A mapped pool is taken to be present in full. KGSL and the DRM drivers
//     back their allocations when they make them, and memtrack agrees to
//     within 2%, but memory the driver evicts under pressure would still be
//     counted. That every allocation is mapped is the user-mode driver's
//     habit, not a guarantee; a driver that stopped mapping device-local
//     allocations would read low here.
//   - RssShmem stands for "KGSL pages the CPU touched". A process that uses
//     shmem for something else (ashmem, binder buffers) inflates it; it is
//     capped at the KGSL pool so the CPU side can never be reduced by more
//     than the accelerator side claims, and contributes nothing without a
//     KGSL pool.
//   - The Mali pool is the kernel's counter, which includes pages kbase keeps
//     in its per-context pool after a free; the figure can lag a release. Its
//     two inputs are read a few milliseconds apart and VmRSS is a batched
//     counter, so the gap carries a few MB of slack (under 3 MB measured on
//     non-Mali devices, where it is zero by construction).
//   - smaps_rollup walks the process's page tables (10-30 ms on a phone), so
//     it and /proc/self/maps are re-read at most every
//     kAddressSpaceRefreshInterval (below) and the pools are carried over in
//     between. During model load the GPU figure can therefore be up to that
//     much behind; at steady state it does not move.
//
// What is not seen: GPU memory of a driver that neither maps its allocations
// into the process nor counts them in VmRSS (none of the four measured), and
// vendor NPU pools that bypass dma-buf. None of the four drivers reports
// memory through DRM fdinfo (accelerator_memory_linux.cc's source) either;
// should an Android DRM driver grow the memory keys, preferring fdinfo over
// the mapped bytes would be the better reading.

#include "runtime/util/memory_profiling/accelerator_memory_android.h"

#include <cstdint>
#include <optional>

#include "absl/base/thread_annotations.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "runtime/util/memory_profiling/accelerator_memory_dmabuf.h"
#include "runtime/util/memory_profiling/accelerator_memory_monitor.h"
#include "runtime/util/memory_profiling/proc_mappings.h"

namespace litert::lm {

namespace {

// How often the address-space walk (/proc/self/maps and
// /proc/self/smaps_rollup) is repeated. The knob to turn if GPU memory moves
// faster than this in a workload of interest, at the cost of the walk's 10-30
// ms per refresh on the sampling thread; MemoryMonitor samples every 50 ms by
// default. /proc/self/status and /proc/self/fdinfo are cheap and are read on
// every call.
//
// What the lag costs. On Mali, nothing in the headline: the resident total is
// VmRSS (fresh) + dma-bufs + Mali pool - (Mali pool + touched KGSL pages), in
// which the stale pool cancels, so only the CPU / accelerator split of the
// figure can be up to one interval behind. On KGSL and DRM the pool is added
// to the total outright, so the total lags too, and since MemoryMonitor keeps
// the peak sample, a GPU peak shorter than the interval can be missed
// altogether rather than reported late. Steady-state benchmarks are
// unaffected either way.
constexpr absl::Duration kAddressSpaceRefreshInterval = absl::Milliseconds(500);

// The address-space walk, cached between refreshes.
struct AddressSpaceWalk {
  GpuDriverMappings driver;
  // The Mali pool, computed at walk time from the VmRSS of the same instant
  // (see UnmappedRssBytes).
  uint64_t mali_pool_bytes = 0;
};

// The most recent walk, refreshed at most every kAddressSpaceRefreshInterval
// and shared by every caller in the process (the MemoryMonitor sampler and
// anyone taking a MemorySnapshot). Returned by value, since another caller
// may refresh the cached copy as soon as the lock is released.
AddressSpaceWalk GetAddressSpaceWalk(const ProcResidentCounters& status) {
  static absl::Mutex* const mutex = new absl::Mutex;
  static AddressSpaceWalk walk ABSL_GUARDED_BY(*mutex);
  static absl::Time last_refresh ABSL_GUARDED_BY(*mutex) = absl::InfinitePast();

  absl::MutexLock lock(mutex);
  const absl::Time now = absl::Now();
  if (now - last_refresh < kAddressSpaceRefreshInterval) return walk;
  last_refresh = now;
  walk.driver = SummarizeGpuDriverMappings(ReadProcMaps("/proc/self"));
  // smaps_rollup is the expensive read; only Mali needs it.
  walk.mali_pool_bytes =
      walk.driver.has_mali
          ? UnmappedRssBytes(status, ReadSmapsRollupRss("/proc/self"))
          : 0;
  return walk;
}

}  // namespace

std::optional<AcceleratorMemoryUsage> GetProcessAcceleratorMemoryUsage() {
  const DmaBufSummary dmabuf = ReadDmaBufSummary("/proc/self");
  // /proc/self/status is always readable; should it not be, the mapped pools
  // (which need only /proc/self/maps) are still reported, and only the Mali
  // pool and the overlap with the host's figures are lost.
  const ProcResidentCounters status =
      ReadProcStatusResident("/proc/self").value_or(ProcResidentCounters());
  const AddressSpaceWalk walk = GetAddressSpaceWalk(status);
  return AcceleratorMemoryUsageFromAndroidSources(
      dmabuf, walk.driver,
      AndroidResidentView{.status = status,
                          .unmapped_rss_bytes = walk.mali_pool_bytes});
}

}  // namespace litert::lm
