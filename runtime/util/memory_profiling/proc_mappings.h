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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_MEMORY_PROFILING_PROC_MAPPINGS_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_MEMORY_PROFILING_PROC_MAPPINGS_H_

// Readers for the kernel's description of a process's address space (Linux
// and Android): /proc/<pid>/maps, /proc/<pid>/smaps_rollup and the resident
// set counters of /proc/<pid>/status. Internal to memory_profiling; the
// Android accelerator memory implementation is built on them.
//
// Cost, measured on a Pixel 9 Pro with ~7000 mappings: `maps` is a plain
// listing of the mappings (no page-table walk), about 1 ms; `status` is a
// handful of counters, well under 1 ms; `smaps_rollup` walks every page
// table of the process to count what is resident, 10-30 ms. Callers that
// sample frequently should read `smaps_rollup` sparingly.

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl

namespace litert::lm {

// One line of /proc/<pid>/maps: a mapping of [start, end) and the path it is
// backed by ("/dev/kgsl-3d0", "/data/model.litertlm", "[anon:scudo:primary]",
// "" for anonymous memory without a name).
struct ProcMapping {
  uint64_t start = 0;
  uint64_t end = 0;
  std::string path;

  uint64_t size_bytes() const { return end - start; }
};

// Parses the contents of a maps file. Lines that do not parse are skipped.
std::vector<ProcMapping> ParseProcMaps(absl::string_view contents);

// Reads `<proc_self_dir>/maps`. Empty if the file cannot be read.
// `proc_self_dir` is "/proc/self" in production.
std::vector<ProcMapping> ReadProcMaps(absl::string_view proc_self_dir);

// The resident memory of a process as the kernel's two sources report it.
struct ProcResidentCounters {
  // VmRSS of /proc/<pid>/status: the kernel's running per-process counters
  // (anonymous + file + shmem pages). Drivers may add to these counters
  // without mapping pages into the process (Mali's kbase does, so that its
  // GPU memory is visible to the low-memory killer).
  uint64_t vm_rss_bytes = 0;
  // RssShmem of /proc/<pid>/status: the part of VmRSS that is shmem-backed,
  // which on Android is mostly the dma-buf and GPU driver pages the CPU has
  // touched.
  uint64_t rss_shmem_bytes = 0;
};

// Parses the counters above out of the contents of a status file, or returns
// std::nullopt if VmRSS is missing.
std::optional<ProcResidentCounters> ParseProcStatusResident(
    absl::string_view contents);

// Reads them from `<proc_self_dir>/status`.
std::optional<ProcResidentCounters> ReadProcStatusResident(
    absl::string_view proc_self_dir);

// Parses the Rss line out of the contents of a smaps_rollup file: the pages
// actually mapped into the process and resident, found by walking its page
// tables. Unlike VmRSS this cannot be inflated by a driver.
std::optional<uint64_t> ParseSmapsRollupRss(absl::string_view contents);

// Reads it from `<proc_self_dir>/smaps_rollup`. See the file comment for the
// cost.
std::optional<uint64_t> ReadSmapsRollupRss(absl::string_view proc_self_dir);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_MEMORY_PROFILING_PROC_MAPPINGS_H_
