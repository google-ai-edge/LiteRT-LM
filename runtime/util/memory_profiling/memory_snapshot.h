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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_MEMORY_PROFILING_MEMORY_SNAPSHOT_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_MEMORY_PROFILING_MEMORY_SNAPSHOT_H_

#include <cstdint>
#include <optional>
#include <ostream>
#include <string>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/util/memory_profiling/accelerator_memory_monitor.h"
#include "runtime/util/memory_profiling/host_memory.h"

namespace litert::lm {

// An amount of memory split into the part the CPU side of the process uses
// and the part its accelerators (GPU, and where measured NPU) use.
// `total_bytes` is always `cpu_bytes + accelerator_bytes`; no memory is in
// both.
struct MemorySplit {
  uint64_t total_bytes = 0;
  uint64_t cpu_bytes = 0;
  uint64_t accelerator_bytes = 0;
};

// The memory a LiteRT-LM process uses at one point in time, as the two
// numbers most people mean by "memory":
//
//   resident_in_ram    How much physical memory the process occupies right
//                      now, wherever it is (system RAM or a GPU's own memory)
//                      and whoever uses it (CPU or accelerator). Memory
//                      swapped or compressed out does not count. The number
//                      for "does it fit on this machine".
//   attributed_by_os   How much memory the operating system holds the process
//                      responsible for, as its own tools show it: Activity
//                      Monitor's "Memory" column, Task Manager's commit size,
//                      RSS + swap on Linux and Android; plus accelerator
//                      memory the OS keeps no charge for at all (video memory
//                      on Linux). The number for "what the OS will kill it
//                      over" and for comparing against those tools. Its
//                      definition differs by OS, deliberately; host_memory.h
//                      has the table.
//
// Each is split into a CPU and an accelerator part that add up to it. The
// accelerator part is GPU memory; NPU memory is in it only where the platform
// does not tell the two apart (Android's dma-bufs), not on every platform. On
// systems where the OS counts accelerator memory in its own figures (Windows,
// Apple, GPU buffers the CPU has mapped on Linux), that memory appears once,
// on the accelerator side; see AcceleratorMemoryUsage for how each platform
// tells the two apart.
//
// All sizes are in bytes. A figure is std::nullopt when the platform cannot
// measure its host part; the accelerator part alone is then in `accelerator`.
struct MemorySnapshot {
  std::optional<MemorySplit> resident_in_ram;
  std::optional<MemorySplit> attributed_by_os;
  // The accelerator side in detail (regions, per-device or per-buffer lines);
  // std::nullopt where accelerator memory tracking is unsupported, in which
  // case the accelerator parts above are 0.
  std::optional<AcceleratorMemoryUsage> accelerator;

  // Single-line summary, e.g.
  //   resident in RAM=1548.02 MB (CPU 352.63 MB + accelerator 1195.39 MB),
  //   attributed by OS=783.41 MB (CPU 353.03 MB + accelerator 430.38 MB),
  //   accelerator detail=1195.39 MB (graphics=430.38 MB, mapped
  //   weights=765.02 MB)
  // Unavailable quantities are omitted. Also what `<<` and absl::StrCat()
  // print, so `std::cerr << "after load " << snapshot` just works. See
  // FormatMemorySnapshot() for the detailed, multi-line form.
  std::string ToString() const;

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const MemorySnapshot& snapshot) {
    sink.Append(snapshot.ToString());
  }
};

std::ostream& operator<<(std::ostream& os, const MemorySnapshot& snapshot);

// Formats a split as
// "<total> MB (CPU <cpu> MB + accelerator <accelerator> MB)".
std::string FormatMemorySplit(const MemorySplit& split);

// Takes a snapshot. The host and accelerator parts are read back to back so
// they describe the same instant. Costs a few system calls (a /proc read or
// two on Linux / Android, per-GPU kernel queries on Windows, a walk of the
// process's memory map on Apple platforms): fine at phase boundaries or at a
// sampling cadence of tens of milliseconds, not inside hot loops.
MemorySnapshot TakeMemorySnapshot();

// Takes a snapshot, logs FormatMemorySnapshot(snapshot, title) (see below) at
// INFO, one log line per text line, and returns the snapshot. For marking the
// phases of a run:
//
//   TakeAndLogMemorySnapshot("Memory after model load");
MemorySnapshot TakeAndLogMemorySnapshot(
    absl::string_view title = "Memory usage");

// Detailed, multi-line form: `title` on the first line, then
//   Memory resident in RAM: <total> MB (CPU <cpu> MB + accelerator <acc> MB)
//   Memory attributed to the process by the OS: <total> MB (CPU ... + ...)
//   Accelerator memory detail: <total> MB (<breakdown>)
//     - <bytes> MB  <allocation>
// for the available quantities. Note that log parsers which look for
// "<name>: <value> <unit>" lines will pick these up as metrics.
std::string FormatMemorySnapshot(const MemorySnapshot& snapshot,
                                 absl::string_view title = "Memory usage");

// FOR TESTING PURPOSES ONLY
//
// Builds a snapshot from its raw ingredients, the host memory and the
// accelerator usage. TakeMemorySnapshot() is this applied to the live values.
MemorySnapshot MakeMemorySnapshot(
    const HostMemoryUsage& host,
    std::optional<AcceleratorMemoryUsage> accelerator);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_MEMORY_PROFILING_MEMORY_SNAPSHOT_H_
