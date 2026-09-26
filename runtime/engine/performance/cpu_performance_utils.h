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

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_PERFORMANCE_CPU_PERFORMANCE_UTILS_H_
