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

#include <dirent.h>
#include <sys/stat.h>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/strings/ascii.h"  // from @com_google_absl
#include "absl/strings/numbers.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/strings/strip.h"  // from @com_google_absl
#include "runtime/util/memory_profiling/accelerator_memory_monitor.h"

namespace litert::lm {

namespace {

// Returns the value of an fdinfo line of the form "<key>:\t<value>" or
// "<key>: <value>", or std::nullopt if `line` is not for `key`.
std::optional<absl::string_view> FdinfoValue(absl::string_view line,
                                             absl::string_view key) {
  if (!absl::ConsumePrefix(&line, key) || !absl::ConsumePrefix(&line, ":")) {
    return std::nullopt;
  }
  return absl::StripAsciiWhitespace(line);
}

}  // namespace

DmaBufSummary ReadDmaBufSummary(absl::string_view proc_self_dir) {
  DmaBufSummary summary;
  const std::string fdinfo_dir = absl::StrCat(proc_self_dir, "/fdinfo");
  DIR* dir = opendir(fdinfo_dir.c_str());
  if (dir == nullptr) {
    return summary;
  }

  struct DmaBufItem {
    uint64_t size = 0;
    std::string name;
    std::string exp_name;
  };
  std::unordered_map<uint64_t, DmaBufItem> buffers;

  struct dirent* de = nullptr;
  while ((de = readdir(dir)) != nullptr) {
    if (de->d_name[0] == '.') {
      continue;
    }
    std::ifstream file(absl::StrCat(fdinfo_dir, "/", de->d_name));
    if (!file.is_open()) {
      continue;
    }

    uint64_t cur_ino = 0;
    uint64_t cur_size = 0;
    std::string cur_name;
    std::string cur_exp;
    std::string line;
    while (std::getline(file, line)) {
      if (const auto value = FdinfoValue(line, "size")) {
        (void)absl::SimpleAtoi(*value, &cur_size);
      } else if (const auto value = FdinfoValue(line, "ino")) {
        (void)absl::SimpleAtoi(*value, &cur_ino);
      } else if (const auto value = FdinfoValue(line, "exp_name")) {
        cur_exp = std::string(*value);
      } else if (const auto value = FdinfoValue(line, "name")) {
        cur_name = std::string(*value);
      }
    }

    if (cur_size == 0 || cur_exp.empty()) {
      continue;  // Not a DMA-BUF.
    }
    if (cur_ino == 0) {
      // Older kernels do not list the inode in fdinfo; take it from the
      // descriptor itself, and as a last resort key by descriptor number.
      struct stat st;
      if (stat(absl::StrCat(proc_self_dir, "/fd/", de->d_name).c_str(), &st) ==
          0) {
        cur_ino = st.st_ino;
      } else {
        (void)absl::SimpleAtoi(de->d_name, &cur_ino);
      }
    }
    buffers[cur_ino] = DmaBufItem{
        .size = cur_size,
        .name = cur_name.empty() ? cur_exp : cur_name,
        .exp_name = cur_exp,
    };
  }
  closedir(dir);

  summary.entries.reserve(buffers.size());
  for (const auto& [ino, item] : buffers) {
    summary.total_bytes += item.size;
    summary.entries.push_back({
        .inode = ino,
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
  usage.total_bytes = dmabuf.total_bytes;

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
