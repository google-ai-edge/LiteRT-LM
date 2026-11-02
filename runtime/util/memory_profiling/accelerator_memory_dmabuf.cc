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

#include "runtime/util/memory_profiling/accelerator_memory_dmabuf.h"

#include <sys/stat.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/strings/numbers.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/util/memory_profiling/accelerator_memory.h"
#include "runtime/util/memory_profiling/fdinfo.h"

namespace litert::lm {

namespace {

// The value of `key` parsed as an integer, or 0 if absent or malformed.
uint64_t IntField(const FdinfoFields& fields, absl::string_view key) {
  const auto it = fields.find(key);
  uint64_t value = 0;
  if (it != fields.end()) (void)absl::SimpleAtoi(it->second, &value);
  return value;
}

std::string StringField(const FdinfoFields& fields, absl::string_view key) {
  const auto it = fields.find(key);
  return it == fields.end() ? std::string() : it->second;
}

}  // namespace

DmaBufSummary ReadDmaBufSummary(absl::string_view proc_self_dir) {
  struct DmaBufItem {
    uint64_t size = 0;
    std::string name;
    std::string exp_name;
  };
  std::unordered_map<uint64_t, DmaBufItem> buffers;

  ForEachFdinfo(proc_self_dir, [&](absl::string_view fd_name,
                                   const FdinfoFields& fields) {
    const uint64_t size = IntField(fields, "size");
    const std::string exp_name = StringField(fields, "exp_name");
    if (size == 0 || exp_name.empty()) return;  // Not a DMA-BUF.

    uint64_t inode = IntField(fields, "ino");
    if (inode == 0) {
      // Older kernels do not list the inode in fdinfo; take it from the
      // descriptor itself, and as a last resort key by descriptor number.
      struct stat st;
      if (stat(absl::StrCat(proc_self_dir, "/fd/", fd_name).c_str(), &st) ==
          0) {
        inode = st.st_ino;
      } else {
        (void)absl::SimpleAtoi(fd_name, &inode);
      }
    }
    const std::string name = StringField(fields, "name");
    buffers[inode] = DmaBufItem{
        .size = size,
        .name = name.empty() ? exp_name : name,
        .exp_name = exp_name,
    };
  });

  DmaBufSummary summary;
  summary.entries.reserve(buffers.size());
  for (const auto& [inode, item] : buffers) {
    summary.total_bytes += item.size;
    summary.entries.push_back({
        .inode = inode,
        .size_bytes = item.size,
        .name = item.name,
        .exp_name = item.exp_name,
    });
  }
  // Largest first; ties broken by inode so the output is deterministic.
  std::sort(summary.entries.begin(), summary.entries.end(),
            [](const DmaBufSummary::Entry& a, const DmaBufSummary::Entry& b) {
              return std::tie(b.size_bytes, a.inode) <
                     std::tie(a.size_bytes, b.inode);
            });
  return summary;
}

AcceleratorMemoryUsage AcceleratorMemoryUsageFromDmaBuf(
    const DmaBufSummary& dmabuf) {
  AcceleratorMemoryUsage usage;
  // A dma-buf is allocated in full when created and stays in memory (it is
  // not swappable), so present and charged are the same figure.
  usage.total_bytes = dmabuf.total_bytes;
  usage.attributed_bytes = dmabuf.total_bytes;
  // How much of this the host's own figures (VmRSS, VmRSS + VmSwap; see
  // host_memory_linux.cc) already contain depends on whether the CPU has
  // touched the buffers, which takes the process's resident counters to
  // tell. The Android implementation has them and fills these in
  // (accelerator_memory_android_sources.cc); on its own a dma-buf summary
  // claims no overlap.
  usage.resident_counted_by_host_bytes = 0;
  usage.attributed_counted_by_host_bytes = 0;

  // Break the total down by exporter, i.e. the dma-buf heap the buffer was
  // allocated from (e.g. "system-uncached", "system"), and keep every buffer
  // for the diagnostic listing. `dmabuf.entries` is already ordered largest
  // first.
  absl::flat_hash_map<std::string, uint64_t> bytes_by_exporter;
  usage.allocations.reserve(dmabuf.entries.size());
  for (const DmaBufSummary::Entry& entry : dmabuf.entries) {
    bytes_by_exporter[entry.exp_name] += entry.size_bytes;
    usage.allocations.emplace_back(
        absl::StrFormat("%s (Inode %u)", entry.name, entry.inode),
        entry.size_bytes);
  }
  usage.breakdown.assign(bytes_by_exporter.begin(), bytes_by_exporter.end());
  // Largest first; ties broken by name so the output is deterministic.
  std::sort(usage.breakdown.begin(), usage.breakdown.end(),
            [](const auto& a, const auto& b) {
              return std::tie(b.second, a.first) < std::tie(a.second, b.first);
            });
  return usage;
}

}  // namespace litert::lm
