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

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/engine/performance/cpu_performance_utils.h"

namespace litert::lm {

std::vector<int> GetDefaultPerformanceCores() { return {}; }

#if !defined(__gnu_linux__) && !defined(__ANDROID__) && !defined(_WIN32)
absl::Status SetCpuAffinity(const std::vector<int>& cpu_affinity_cores) {
  return absl::OkStatus();
}

absl::Status SetProcessPriority(CpuPriorityLevel priority) {
  return absl::OkStatus();
}

absl::Status SetThreadPriority(CpuPriorityLevel priority) {
  return absl::OkStatus();
}

#endif  // !defined(__gnu_linux__) && !defined(__ANDROID__) && !defined(_WIN32)

}  // namespace litert::lm
