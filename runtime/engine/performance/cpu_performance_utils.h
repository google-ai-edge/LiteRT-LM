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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_PERFORMANCE_CPU_PERFORMANCE_UTILS_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_PERFORMANCE_CPU_PERFORMANCE_UTILS_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl

namespace litert::lm {

enum class CpuPriorityLevel {
  kNormal = 0,
  kMedium = 1,
  kHigh = 2,
  kRealtime = 3,
};

// Sets CPU core affinity for the current thread and any child threads it
// creates. On Apple Silicon (macOS/iOS), this routes threads to P-cores.
// Automatically discovers and returns high-performance CPU cores (P-cores)
// on supported operating systems.
std::vector<int> GetDefaultPerformanceCores();

absl::Status SetCpuAffinity(const std::vector<int>& cpu_affinity_cores);

// Promotes process priority class to prevent CPU governor throttling.
absl::Status SetProcessPriority(CpuPriorityLevel priority);

// Promotes thread scheduling priority and disables power throttling.
absl::Status SetThreadPriority(CpuPriorityLevel priority);

// Applies CPU affinity and priority settings when running in benchmark mode.
absl::Status EnableCpuPerformanceBoosters(
    const std::vector<int>& cpu_affinity_cores, CpuPriorityLevel priority);

// Abstract session for reporting per-step durations to dynamic governors
// (ADPF).
class CpuPerformanceSession {
 public:
  static absl::StatusOr<std::unique_ptr<CpuPerformanceSession>> Create(
      const std::vector<int>& tids, int64_t target_duration_ns);
  virtual ~CpuPerformanceSession() = default;

  // Reports actual work step duration to scale CPU clock frequencies.
  virtual void ReportActualDuration(int64_t actual_duration_ns) = 0;
};

// Keeps CPU cores at elevated clock frequencies by spinning in a busy-wait
// loop on the current thread for the specified duration in nanoseconds.
// This prevents kernel CPU governors (schedutil / ondemand) from dropping
// frequency or entering low-power sleep states during brief lulls between
// autoregressive token decode steps in benchmark mode.
void BusyWaitPolling(int64_t duration_ns);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_PERFORMANCE_CPU_PERFORMANCE_UTILS_H_
