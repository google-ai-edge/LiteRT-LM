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

#if defined(__APPLE__)
#include <pthread.h>
#include <pthread/qos.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "runtime/util/status_macros.h"
#endif  // defined(__APPLE__)

namespace litert::lm {

#if defined(__APPLE__)

absl::Status SetCpuAffinity(const std::vector<int>& cpu_affinity_cores) {
  // Explicit CPU core pinning is not supported by macOS/iOS XNU kernel.
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

  qos_class_t qos_class = QOS_CLASS_DEFAULT;
  switch (priority) {
    case CpuPriorityLevel::kNormal:
      qos_class = QOS_CLASS_DEFAULT;
      break;
    case CpuPriorityLevel::kMedium:
      qos_class = QOS_CLASS_USER_INITIATED;
      break;
    case CpuPriorityLevel::kHigh:
      qos_class = QOS_CLASS_USER_INTERACTIVE;
      break;
    case CpuPriorityLevel::kRealtime:
      qos_class = QOS_CLASS_USER_INTERACTIVE;
      break;
  }

  int err = pthread_set_qos_class_self_np(qos_class, 0);
  if (err != 0) {
    ABSL_LOG(WARNING) << "Failed to set thread QoS class: " << strerror(err);
  }
  return absl::OkStatus();
}

#endif  // defined(__APPLE__)

}  // namespace litert::lm
