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

#if defined(__gnu_linux__) && !defined(__ANDROID__)
#include <pthread.h>
#include <sched.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/util/status_macros.h"
#endif  // defined(__gnu_linux__) && !defined(__ANDROID__)

namespace litert::lm {

#if defined(__gnu_linux__) && !defined(__ANDROID__)

absl::Status SetCpuAffinity(const std::vector<int>& cpu_affinity_cores) {
  if (cpu_affinity_cores.empty()) {
    ABSL_LOG(WARNING) << "CPU affinity core list is empty, skipping affinity.";
    return absl::OkStatus();
  }

  cpu_set_t mask;
  CPU_ZERO(&mask);
  for (int cpu : cpu_affinity_cores) {
    CPU_SET(cpu, &mask);
  }

  int err = pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask);
  if (err != 0) {
    ABSL_LOG(WARNING) << "Failed to set CPU affinity: " << strerror(err);
    return absl::OkStatus();
  }

  ABSL_VLOG(1) << "Successfully set CPU affinity on Linux.";
  return absl::OkStatus();
}

absl::Status SetProcessPriority(CpuPriorityLevel priority) {
  if (priority == CpuPriorityLevel::kNormal) {
    return absl::OkStatus();
  }

  int nice_value = 0;
  switch (priority) {
    case CpuPriorityLevel::kNormal:
      nice_value = 0;
      break;
    case CpuPriorityLevel::kMedium:
      nice_value = -5;
      break;
    case CpuPriorityLevel::kHigh:
      nice_value = -10;
      break;
    case CpuPriorityLevel::kRealtime:
      nice_value = -20;
      break;
  }

  if (setpriority(PRIO_PROCESS, 0, nice_value) != 0) {
    ABSL_VLOG(1) << "Failed to set process priority to nice value "
                 << nice_value << ": " << strerror(errno);
  }
  return absl::OkStatus();
}

absl::Status SetThreadPriority(CpuPriorityLevel priority) {
  if (priority == CpuPriorityLevel::kNormal) {
    return absl::OkStatus();
  }

  struct sched_param param;
  int policy = SCHED_OTHER;
  switch (priority) {
    case CpuPriorityLevel::kNormal:
      policy = SCHED_OTHER;
      param.sched_priority = 0;
      break;
    case CpuPriorityLevel::kMedium:
      policy = SCHED_FIFO;
      param.sched_priority = 40;
      break;
    case CpuPriorityLevel::kHigh:
      policy = SCHED_FIFO;
      param.sched_priority = 80;
      break;
    case CpuPriorityLevel::kRealtime:
      policy = SCHED_FIFO;
      param.sched_priority = 90;
      break;
  }

  int err = pthread_setschedparam(pthread_self(), policy, &param);
  if (err != 0) {
    ABSL_LOG(WARNING) << "Failed to set thread scheduling priority: "
                      << strerror(err);
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
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || \
    defined(_M_IX86)
#if defined(_MSC_VER)
    _mm_pause();
#else
    __builtin_ia32_pause();
#endif
#elif defined(__aarch64__) || defined(_M_ARM64) || defined(__arm__)
#if defined(_MSC_VER)
    __yield();
#else
    asm volatile("yield" ::: "memory");
#endif
#endif
  }
}

#endif  // defined(__gnu_linux__) && !defined(__ANDROID__)

}  // namespace litert::lm
