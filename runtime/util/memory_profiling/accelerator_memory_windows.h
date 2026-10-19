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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_MEMORY_PROFILING_ACCELERATOR_MEMORY_WINDOWS_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_MEMORY_PROFILING_ACCELERATOR_MEMORY_WINDOWS_H_

// Building blocks of the Windows implementation of the platform contract in
// accelerator_memory.h. Internal to memory_profiling: exposed only so
// that accelerator_memory_windows_test.cc can exercise them without a GPU.

#include <cstdint>
#include <string>
#include <vector>

#include "runtime/util/memory_profiling/accelerator_memory.h"

namespace litert::lm {

// Video memory used by this process on a single GPU. (Windows calls a GPU an
// "adapter"; the two terms are used interchangeably in this directory.)
struct AdapterVideoMemory {
  // Human-readable GPU name, e.g. "NVIDIA GeForce RTX 4070".
  std::string description;
  // Bytes currently resident in the GPU's local segment group: dedicated VRAM
  // on discrete GPUs. Integrated GPUs may file their main system memory
  // segment here too (Intel does), so "local" does not imply "not system
  // memory".
  uint64_t local_bytes = 0;
  // Bytes currently resident in the non-local segment group (system memory
  // shared with the GPU).
  uint64_t non_local_bytes = 0;
  // Bytes the video memory manager has committed for this process on the GPU,
  // resident or paged out: what it charges to the process's commit charge. See
  // the background comment in accelerator_memory_windows.cc.
  uint64_t committed_bytes = 0;
  // The part of `committed_bytes` that lives in the GPU's system-memory
  // segments (an integrated GPU's whole memory, a discrete GPU's shared
  // memory) rather than in dedicated VRAM. Those pages are process pages: in
  // RAM they are in the process's working set, so they are the overlap with
  // HostMemoryUsage::resident_bytes. See the background comment.
  uint64_t system_memory_committed_bytes = 0;
};

// Aggregates per-GPU video memory into an AcceleratorMemoryUsage: resident
// memory is local + non-local over all GPUs, with the breakdown {"local",
// "non-local"} and the per-GPU detail in the allocations ("<description>
// (local)" / "<description> (non-local)", largest first, zero entries
// omitted); attributed memory is the committed bytes over all GPUs, all of
// which the commit charge already contains; and the part of resident memory
// the working set already contains is the system-memory share of each GPU's
// committed bytes, capped at its resident bytes.
AcceleratorMemoryUsage AcceleratorMemoryUsageFromAdapters(
    const std::vector<AdapterVideoMemory>& adapters);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_MEMORY_PROFILING_ACCELERATOR_MEMORY_WINDOWS_H_
