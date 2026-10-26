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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_MEMORY_PROFILING_HOST_MEMORY_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_MEMORY_PROFILING_HOST_MEMORY_H_

#include <cstdint>
#include <optional>

namespace litert::lm {

// Host (CPU-side) memory of the current process at a point in time, as two
// numbers that answer two different questions. Both are instantaneous: they
// go down again when memory is freed. A field is std::nullopt when the
// platform cannot measure it.
struct HostMemoryUsage {
  // How much physical RAM the process occupies right now: the resident set
  // size (Linux / Android), working set (Windows) or resident size (Apple).
  // Counts every page in RAM that is mapped into the process, including pages
  // shared with other processes and file-backed ones such as an mmap()ed
  // model, and not pages that are swapped out or compressed. The number to
  // look at for "does it fit" and "how close is it to being killed"; it also
  // moves with page cache pressure, so it is a noisy regression metric.
  //
  // This is a physical figure, not a host-only one: on systems where the
  // accelerator works out of system RAM (Android, integrated GPUs, Apple
  // silicon) accelerator buffers the process has mapped are in here too,
  // because they do occupy RAM. How much of the accelerator figure that is
  // depends on the driver and is not measured, so this number is reported as
  // the OS gives it and is never added to the accelerator figure.
  std::optional<uint64_t> resident_bytes;
  // How much memory the platform attributes to the process: the figure its
  // own tools and memory limits use. This is deliberately platform-specific
  // and NOT comparable across platforms. Each host_memory_<os>.cc documents
  // its definition at the top; in short:
  //   Linux / Android  resident pages plus swapped / zram-compressed ones,
  //                    from /proc/self/status (VmRSS + VmSwap): the cheap
  //                    approximation of Android's PSS, file-backed pages
  //                    (an mmap()ed model) included
  //   Windows          the commit charge (PrivateUsage): private memory in RAM
  //                    or in the pagefile, file-backed pages excluded
  //   macOS / iOS      the physical footprint (phys_footprint): anonymous,
  //                    compressed and IOKit memory, clean file-backed pages
  //                    excluded (so an mmap()ed, never written model is not
  //                    in it, only in resident_bytes)
  // Note the platforms disagree on whether file-backed pages count; that is
  // their accounting, faithfully reported, not ours to reconcile.
  // On some platforms this figure includes accelerator (GPU / NPU) memory the
  // process allocated: Windows, for one, charges every GPU allocation to the
  // commit charge.
  std::optional<uint64_t> platform_attributed_bytes;
};

// -----------------------------------------------------------------------------
// Platform contract
//
// Implemented once per operating system in host_memory_<os>.cc; the build
// picks the file for the target OS and host_memory_unsupported.cc covers every
// other OS. Costs one system call or one small /proc read.
// -----------------------------------------------------------------------------
HostMemoryUsage GetHostMemoryUsage();

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_MEMORY_PROFILING_HOST_MEMORY_H_
