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

// macOS and iOS implementation of the platform contract in host_memory.h,
// read with task_info(TASK_VM_INFO).
//
//   resident_bytes             resident_size: the pages currently in RAM.
//   platform_attributed_bytes  phys_footprint: Activity Monitor's "Memory"
//                              column and the figure iOS enforces its
//                              per-process memory limit (jetsam) against. It
//                              counts the process's anonymous memory, the
//                              compressed part of it, purgeable memory that
//                              is not currently purgeable, and IOKit mappings
//                              (which is where Metal allocations live on
//                              unified-memory hardware); it does not count
//                              clean file-backed pages such as an mmap()ed
//                              model. LiteRT-LM maps its weights from the
//                              model file and never writes to them, so on
//                              Apple platforms the weights are in
//                              resident_bytes (once touched) but not here:
//                              expect platform_attributed_bytes well below
//                              resident_bytes while a model is loaded.

#include <mach/mach.h>

#include "runtime/util/memory_profiling/host_memory.h"

namespace litert::lm {

HostMemoryUsage GetHostMemoryUsage() {
  HostMemoryUsage usage;
  task_vm_info_data_t info = {};
  mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
  if (task_info(mach_task_self(), TASK_VM_INFO,
                reinterpret_cast<task_info_t>(&info), &count) == KERN_SUCCESS) {
    usage.resident_bytes = info.resident_size;
    usage.platform_attributed_bytes = info.phys_footprint;
  }
  return usage;
}

}  // namespace litert::lm
