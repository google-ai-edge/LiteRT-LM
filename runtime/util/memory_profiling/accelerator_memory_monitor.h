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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_MEMORY_PROFILING_ACCELERATOR_MEMORY_MONITOR_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_MEMORY_PROFILING_ACCELERATOR_MEMORY_MONITOR_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "absl/synchronization/notification.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl

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
// accelerator_memory_monitor.cc is platform-specific.
// -----------------------------------------------------------------------------

// Returns the accelerator memory currently used by this process, or
// std::nullopt when accelerator memory tracking is not supported on this
// platform.
std::optional<AcceleratorMemoryUsage> GetProcessAcceleratorMemoryUsage();

// -----------------------------------------------------------------------------
// Platform-neutral helpers
// -----------------------------------------------------------------------------

// Converts bytes to mebibytes.
float AcceleratorMemoryBytesToMb(uint64_t bytes);

// Formats `usage.breakdown` as "name=1.00 MB, other=2.00 MB". Returns an empty
// string when there is no breakdown.
std::string FormatAcceleratorMemoryBreakdown(
    const AcceleratorMemoryUsage& usage);

// Formats `usage.allocations` for diagnostic logging, one line per allocation
// of at least 1 MiB, in the order they appear in `usage.allocations`.
std::vector<std::string> FormatAcceleratorMemoryAllocations(
    const AcceleratorMemoryUsage& usage);

// Tracks the peak accelerator memory usage of the process by sampling
// GetProcessAcceleratorMemoryUsage() from a background thread at a fixed
// interval.
//
// The peak is the sample with the largest `total_bytes`; its breakdown is kept
// as-is so the reported categories always sum to the reported total.
class AcceleratorMemoryMonitor {
 public:
  // Produces the samples. Can be overridden for testing or to provide a
  // platform-specific implementation.
  class Sampler {
   public:
    virtual ~Sampler() = default;
    virtual bool IsSupported() {
      return GetProcessAcceleratorMemoryUsage().has_value();
    }
    virtual std::optional<AcceleratorMemoryUsage> Sample() {
      return GetProcessAcceleratorMemoryUsage();
    }
    virtual void SleepFor(absl::Duration duration);
  };

  explicit AcceleratorMemoryMonitor(
      absl::Duration sampling_interval = absl::Milliseconds(50))
      : AcceleratorMemoryMonitor(sampling_interval,
                                 std::make_unique<Sampler>()) {}
  AcceleratorMemoryMonitor(absl::Duration sampling_interval,
                           std::unique_ptr<Sampler> sampler);
  ~AcceleratorMemoryMonitor();

  AcceleratorMemoryMonitor(const AcceleratorMemoryMonitor&) = delete;
  AcceleratorMemoryMonitor& operator=(const AcceleratorMemoryMonitor&) = delete;

  // Whether accelerator memory tracking is supported. When false, Start() and
  // Stop() are no-ops and GetPeakUsage() always returns std::nullopt.
  bool IsSupported() const { return is_supported_; }

  // Starts sampling. Calling Start() while already running is a no-op.
  void Start() ABSL_LOCKS_EXCLUDED(mutex_);
  // Stops sampling and joins the sampling thread. Calling Stop() when not
  // running is a no-op. The peak remains available after Stop().
  void Stop() ABSL_LOCKS_EXCLUDED(mutex_);

  // Returns the peak sample observed so far, or std::nullopt if tracking is
  // unsupported or no sample has been taken yet.
  std::optional<AcceleratorMemoryUsage> GetPeakUsage() const
      ABSL_LOCKS_EXCLUDED(mutex_);

 private:
  void StopInternal();

  mutable absl::Mutex mutex_;
  std::unique_ptr<Sampler> sampler_;
  bool is_supported_ = false;
  absl::Duration sampling_interval_;
  std::unique_ptr<absl::Notification> stop_signal_;
  std::unique_ptr<std::thread> sampling_thread_;
  std::optional<AcceleratorMemoryUsage> peak_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_MEMORY_PROFILING_ACCELERATOR_MEMORY_MONITOR_H_
