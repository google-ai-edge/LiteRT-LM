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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_CPU_AFFINITY_UTILS_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_CPU_AFFINITY_UTILS_H_

#include <vector>

#include "absl/status/status.h"  // from @com_google_absl

namespace litert::lm {

// Checks if the device is a Google Pixel with a supported Tensor SoC.
bool IsPixelTensorDevice();

// Returns the hardcoded performance zero indexed core IDs for the detected
// Pixel Tensor SoC.
std::vector<int> GetPixelPerformanceCores();

// Sets the CPU affinity to the given cores for the current thread and any child
// threads it creates. This informs the scheduler which cores the thread
// should be run on. The scheduler will then attempt to run the thread on
// these cores most of the time.
absl::Status SetCpuAffinity(const std::vector<int>& cpu_affinity_cores);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_CPU_AFFINITY_UTILS_H_
