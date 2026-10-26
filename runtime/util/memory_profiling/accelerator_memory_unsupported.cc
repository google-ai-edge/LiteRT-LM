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

// Fallback implementation of the platform contract in
// accelerator_memory_monitor.h for operating systems that do not have their
// own accelerator_memory_<os>.cc yet. Accelerator memory tracking is reported
// as unsupported, which makes AcceleratorMemoryMonitor a no-op.

#include <optional>

#include "runtime/util/memory_profiling/accelerator_memory_monitor.h"

namespace litert::lm {

std::optional<AcceleratorMemoryUsage> GetProcessAcceleratorMemoryUsage() {
  return std::nullopt;
}

}  // namespace litert::lm
