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

// Windows implementation of the platform contract in host_memory.h, read with
// GetProcessMemoryInfo().
//
//   resident_bytes             WorkingSetSize: the pages currently in RAM.
//   platform_attributed_bytes  PrivateUsage: the process's commit charge,
//                              i.e. the private memory the system has
//                              promised backing store for, whether it is in
//                              RAM or in the pagefile. This is Task Manager's
//                              "Commit size" and what the system's commit
//                              limit is enforced against. It does not include
//                              file-backed mappings. It does include every
//                              GPU allocation the process made, which the
//                              video memory manager charges to the owning
//                              process's commit.

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
// psapi.h must come after windows.h.
#include <psapi.h>

#include "runtime/util/memory_profiling/host_memory.h"

namespace litert::lm {

HostMemoryUsage GetHostMemoryUsage() {
  HostMemoryUsage usage;
  PROCESS_MEMORY_COUNTERS_EX counters = {};
  if (GetProcessMemoryInfo(
          GetCurrentProcess(),
          reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters),
          sizeof(counters))) {
    usage.resident_bytes = counters.WorkingSetSize;
    usage.platform_attributed_bytes = counters.PrivateUsage;
  }
  return usage;
}

}  // namespace litert::lm
