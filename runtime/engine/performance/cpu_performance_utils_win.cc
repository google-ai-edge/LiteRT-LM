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

#include "runtime/engine/performance/cpu_performance_utils.h"

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <processthreadsapi.h>
#include <windows.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "runtime/util/status_macros.h"

#ifndef THREAD_POWER_THROTTLING_CURRENT_VERSION
#define THREAD_POWER_THROTTLING_CURRENT_VERSION 1
#define THREAD_POWER_THROTTLING_EXECUTION_SPEED 0x1UL
typedef struct _THREAD_POWER_THROTTLING_STATE {
  ULONG Version;
  ULONG ControlMask;
  ULONG StateMask;
} THREAD_POWER_THROTTLING_STATE;
#endif

#endif  // defined(_WIN32)

namespace litert::lm {

#if defined(_WIN32)

absl::Status SetCpuAffinity(const std::vector<int>& cpu_affinity_cores) {
  if (cpu_affinity_cores.empty()) {
    ABSL_LOG(WARNING) << "CPU affinity core list is empty, skipping affinity.";
    return absl::OkStatus();
  }

  DWORD_PTR mask = 0;
  for (int cpu : cpu_affinity_cores) {
    if (cpu >= 0 && cpu < static_cast<int>(sizeof(DWORD_PTR) * 8)) {
      mask |= (1ULL << cpu);
    }
  }

  if (mask == 0) {
    ABSL_LOG(WARNING) << "CPU affinity mask is zero after processing cores.";
    return absl::OkStatus();
  }

  DWORD_PTR res = SetThreadAffinityMask(GetCurrentThread(), mask);
  if (res == 0) {
    ABSL_LOG(WARNING) << "Failed to set thread affinity mask, error: "
                      << GetLastError();
    return absl::OkStatus();
  }

  ABSL_VLOG(1) << "Successfully set CPU affinity on Windows.";
  return absl::OkStatus();
}

absl::Status SetProcessPriority(CpuPriorityLevel priority) {
  if (priority == CpuPriorityLevel::kNormal) {
    return absl::OkStatus();
  }

  DWORD priority_class = NORMAL_PRIORITY_CLASS;
  switch (priority) {
    case CpuPriorityLevel::kNormal:
      priority_class = NORMAL_PRIORITY_CLASS;
      break;
    case CpuPriorityLevel::kMedium:
      priority_class = ABOVE_NORMAL_PRIORITY_CLASS;
      break;
    case CpuPriorityLevel::kHigh:
      priority_class = HIGH_PRIORITY_CLASS;
      break;
    case CpuPriorityLevel::kRealtime:
      priority_class = REALTIME_PRIORITY_CLASS;
      break;
  }

  if (SetPriorityClass(GetCurrentProcess(), priority_class) == 0) {
    ABSL_LOG(WARNING) << "Failed to set process priority class, error: "
                      << GetLastError();
  }
  return absl::OkStatus();
}

absl::Status SetThreadPriority(CpuPriorityLevel priority) {
  if (priority == CpuPriorityLevel::kNormal) {
    return absl::OkStatus();
  }

  int thread_priority = THREAD_PRIORITY_NORMAL;
  switch (priority) {
    case CpuPriorityLevel::kNormal:
      thread_priority = THREAD_PRIORITY_NORMAL;
      break;
    case CpuPriorityLevel::kMedium:
      thread_priority = THREAD_PRIORITY_ABOVE_NORMAL;
      break;
    case CpuPriorityLevel::kHigh:
      thread_priority = THREAD_PRIORITY_HIGHEST;
      break;
    case CpuPriorityLevel::kRealtime:
      thread_priority = THREAD_PRIORITY_TIME_CRITICAL;
      break;
  }

  if (SetThreadPriority(GetCurrentThread(), thread_priority) == 0) {
    ABSL_LOG(WARNING) << "Failed to set thread priority, error: "
                      << GetLastError();
  }

  if (priority == CpuPriorityLevel::kHigh ||
      priority == CpuPriorityLevel::kRealtime) {
    THREAD_POWER_THROTTLING_STATE throttling_state;
    ZeroMemory(&throttling_state, sizeof(throttling_state));
    throttling_state.Version = THREAD_POWER_THROTTLING_CURRENT_VERSION;
    throttling_state.ControlMask = THREAD_POWER_THROTTLING_EXECUTION_SPEED;
    throttling_state.StateMask = 0;
    SetThreadInformation(GetCurrentThread(), ThreadPowerThrottling,
                         &throttling_state, sizeof(throttling_state));
  }
  return absl::OkStatus();
}

void BusyWaitPolling(int64_t duration_ns) {
  if (duration_ns <= 0) {
    return;
  }
  auto start = std::chrono::steady_clock::now();
  auto target_duration = std::chrono::nanoseconds(duration_ns);
  while (std::chrono::steady_clock::now() - start < target_duration) {
#if defined(_WIN32)
    YieldProcessor();
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || \
    defined(_M_IX86)
    __builtin_ia32_pause();
#elif defined(__aarch64__) || defined(_M_ARM64) || defined(__arm__)
    asm volatile("yield" ::: "memory");
#endif
  }
}

#endif  // defined(_WIN32)

}  // namespace litert::lm
