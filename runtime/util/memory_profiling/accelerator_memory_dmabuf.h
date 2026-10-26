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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_MEMORY_PROFILING_ACCELERATOR_MEMORY_DMABUF_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_MEMORY_PROFILING_ACCELERATOR_MEMORY_DMABUF_H_

// The DMA-BUF buffers a process holds, read from /proc/self/fdinfo. The
// building block of the Android implementation of the platform contract in
// accelerator_memory_monitor.h; built on Linux as well so that it can be tested
// in CI without a device (accelerator_memory_dmabuf_test.cc). Internal to
// memory_profiling.

#include <cstdint>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/util/memory_profiling/accelerator_memory_monitor.h"

namespace litert::lm {

// Summary of the DMA-BUF buffers held by the current process.
struct DmaBufSummary {
  uint64_t total_bytes = 0;
  struct Entry {
    uint64_t inode = 0;
    uint64_t size_bytes = 0;
    std::string name;
    std::string exp_name;
  };
  // Ordered largest first.
  std::vector<Entry> entries;
};

// Reads the DMA-BUF buffers held by this process from the kernel's per-process
// file descriptor information: `<proc_self_dir>/fdinfo/<fd>` for every open
// file descriptor, with `<proc_self_dir>/fd/<fd>` as the fallback source of a
// buffer's inode. `proc_self_dir` is "/proc/self" in production; tests point
// it at a directory of fake fdinfo files.
//
// A file descriptor refers to a DMA-BUF when its fdinfo lists an `exp_name`
// (the exporting dma-buf heap) and a non-zero `size`. Buffers that are mapped
// through several file descriptors are counted once, keyed by inode.
DmaBufSummary ReadDmaBufSummary(absl::string_view proc_self_dir);

// Converts a DMA-BUF summary into an AcceleratorMemoryUsage: the breakdown is
// the total per exporter (largest first) and the allocations are the
// individual buffers in the order of `dmabuf.entries`.
AcceleratorMemoryUsage AcceleratorMemoryUsageFromDmaBuf(
    const DmaBufSummary& dmabuf);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_MEMORY_PROFILING_ACCELERATOR_MEMORY_DMABUF_H_
