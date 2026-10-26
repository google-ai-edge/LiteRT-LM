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

// Linux and Android implementation of the platform contract in host_memory.h,
// read from /proc/self/status (one small file; a process may always read its
// own).
//
//   resident_bytes             VmRSS: the pages currently in RAM, anonymous,
//                              shared and file-backed alike.
//   platform_attributed_bytes  VmRSS + VmSwap: the resident pages plus what
//                              has been swapped out or compressed into zram.
//
// Why file-backed pages are counted. Linux has no per-process figure of its
// own beyond RSS; Android's accounting (PSS, as shown by `dumpsys meminfo`
// and used by the low-memory killer) is RSS with shared pages divided among
// the processes sharing them, and it does count file-backed pages such as an
// mmap()ed model. LiteRT-LM releases model pages it is done with
// (madvise(MADV_DONTNEED)), so the file-backed pages that are resident at any
// moment are the ones the engine is actively using, and they are as much part
// of its footprint as its heap. Leaving them out would hide that cost, and
// would hide a regression in which the releasing stopped working.
//
// VmRSS + VmSwap is therefore the closest cheap approximation of the
// platform's figure: an upper bound on PSS + SwapPss, exact except that pages
// shared with other processes (libc, a model another process also mapped)
// count in full rather than proportionally. The exact figure needs
// /proc/self/smaps_rollup, which walks every mapping and costs milliseconds in
// a large process; too much for a sampler that runs every 50 ms during a
// benchmark.
//
// A caution about accelerator memory. Whether GPU / NPU memory the process
// allocated appears in these counters at all, and in which one, depends on
// how the driver maps it into the process: shmem-backed buffers count under
// RssShmem, pages a driver inserts with vm_insert_page() (Android's system
// dma-buf heap, Adreno's KGSL) count under RssFile, and pages mapped by
// physical frame number (Mali's kbase, many vendor heaps) are not counted in
// RSS at all; buffers the process never maps on the CPU side are in no
// counter regardless. Android's own tooling (`dumpsys meminfo`'s GL and Gfx
// rows) therefore does not read /proc for this but the vendor memtrack HAL.
// So VmRSS + VmSwap may or may not already include some accelerator memory,
// varying by GPU vendor and allocation path. This file reports the host
// counters as the OS gives them; working out how much accelerator memory they
// already contain is the accelerator reader's job (accelerator_memory_*.cc).

#include "runtime/util/memory_profiling/host_memory_linux.h"

#include <cstdint>
#include <fstream>
#include <optional>
#include <sstream>

#include "absl/strings/ascii.h"  // from @com_google_absl
#include "absl/strings/numbers.h"  // from @com_google_absl
#include "absl/strings/str_split.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/strings/strip.h"  // from @com_google_absl
#include "runtime/util/memory_profiling/host_memory.h"

namespace litert::lm {

namespace {

// Returns the value of a "<key>:\t<n> kB" status line, in bytes, or
// std::nullopt if `line` is not for `key` or does not parse.
std::optional<uint64_t> StatusValueBytes(absl::string_view line,
                                         absl::string_view key) {
  if (!absl::ConsumePrefix(&line, key) || !absl::ConsumePrefix(&line, ":")) {
    return std::nullopt;
  }
  line = absl::StripAsciiWhitespace(line);
  if (!absl::ConsumeSuffix(&line, "kB")) return std::nullopt;
  uint64_t kb = 0;
  if (!absl::SimpleAtoi(absl::StripAsciiWhitespace(line), &kb)) {
    return std::nullopt;
  }
  return kb * 1024;
}

}  // namespace

HostMemoryUsage ParseProcStatus(absl::string_view status) {
  std::optional<uint64_t> vm_rss, vm_swap;
  for (absl::string_view line : absl::StrSplit(status, '\n')) {
    if (const auto v = StatusValueBytes(line, "VmRSS")) {
      vm_rss = v;
    } else if (const auto v = StatusValueBytes(line, "VmSwap")) {
      vm_swap = v;
    }
  }

  HostMemoryUsage usage;
  usage.resident_bytes = vm_rss;
  if (vm_rss.has_value()) {
    // A missing VmSwap line means no swap, not an unknown amount.
    usage.platform_attributed_bytes = *vm_rss + vm_swap.value_or(0);
  }
  return usage;
}

HostMemoryUsage GetHostMemoryUsage() {
  std::ifstream file("/proc/self/status");
  if (!file.is_open()) return HostMemoryUsage();
  std::stringstream contents;
  contents << file.rdbuf();
  return ParseProcStatus(contents.str());
}

}  // namespace litert::lm
