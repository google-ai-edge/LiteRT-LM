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

#include "runtime/engine/dmabuf_util.h"

#if defined(__ANDROID__)
#include <dirent.h>
#include <sys/stat.h>
#endif

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/strings/numbers.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/strings/strip.h"  // from @com_google_absl

namespace litert::lm {

#if defined(__ANDROID__)
DmaBufSummary GetProcessDmaBufUsage() {
  DmaBufSummary summary;
  DIR* dir = opendir("/proc/self/fdinfo");
  if (dir == nullptr) {
    return summary;
  }

  struct dirent* de = nullptr;
  struct DmaBufItem {
    uint64_t size = 0;
    std::string name;
    std::string exp_name;
  };
  std::unordered_map<uint64_t, DmaBufItem> buffers;

  while ((de = readdir(dir)) != nullptr) {
    if (de->d_name[0] == '.') {
      continue;
    }
    std::string fd_path = absl::StrCat("/proc/self/fdinfo/", de->d_name);
    std::ifstream file(fd_path);
    if (!file.is_open()) {
      continue;
    }

    uint64_t cur_ino = 0;
    uint64_t cur_size = 0;
    std::string cur_name;
    std::string cur_exp;
    std::string line;

    while (std::getline(file, line)) {
      if (absl::StartsWith(line, "size:\t") ||
          absl::StartsWith(line, "size: ")) {
        (void)absl::SimpleAtoi(line.substr(line.find_first_of(" \t") + 1),
                               &cur_size);
      } else if (absl::StartsWith(line, "ino:\t") ||
                 absl::StartsWith(line, "ino: ")) {
        (void)absl::SimpleAtoi(line.substr(line.find_first_of(" \t") + 1),
                               &cur_ino);
      } else if (absl::StartsWith(line, "exp_name:\t") ||
                 absl::StartsWith(line, "exp_name: ")) {
        cur_exp = std::string(absl::StripTrailingAsciiWhitespace(
            line.substr(line.find_first_of(" \t") + 1)));
      } else if (absl::StartsWith(line, "name:\t") ||
                 absl::StartsWith(line, "name: ")) {
        cur_name = std::string(absl::StripTrailingAsciiWhitespace(
            line.substr(line.find_first_of(" \t") + 1)));
      }
    }

    if (cur_size != 0 && !cur_exp.empty()) {
      if (cur_ino == 0) {
        struct stat st;
        if (stat(absl::StrCat("/proc/self/fd/", de->d_name).c_str(), &st) ==
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
  }
  closedir(dir);

  for (const auto& [ino, item] : buffers) {
    summary.total_bytes += item.size;
    summary.entries.push_back({
        .inode = ino,
        .size_bytes = item.size,
        .name = item.name,
        .exp_name = item.exp_name,
    });
  }

  std::sort(
      summary.entries.begin(), summary.entries.end(),
      [](const auto& a, const auto& b) { return a.size_bytes > b.size_bytes; });
  return summary;
}
#else
DmaBufSummary GetProcessDmaBufUsage() { return DmaBufSummary(); }
#endif  // defined(__ANDROID__)

void LogDmaBufUsage(const DmaBufSummary& dmabuf, std::string* report_out) {
  if (dmabuf.total_bytes == 0) {
    return;
  }
  const double total_mb =
      static_cast<double>(dmabuf.total_bytes) / (1024.0 * 1024.0);
  ABSL_LOG(INFO) << absl::StrFormat(
      "Peak DMA-BUF hardware usage: %.2f MB (%zu unique buffers)", total_mb,
      dmabuf.entries.size());
  if (report_out != nullptr) {
    absl::StrAppend(
        report_out,
        absl::StrFormat(
            "Peak DMA-BUF hardware usage: %.2f MB (%zu unique buffers)\n",
            total_mb, dmabuf.entries.size()));
  }

  for (const auto& entry : dmabuf.entries) {
    if (entry.size_bytes >= 1024 * 1024) {
      double sz_mb = static_cast<double>(entry.size_bytes) / (1024.0 * 1024.0);
      ABSL_LOG(INFO) << absl::StrFormat("  - %-28s: %6.2f MB (Inode %llu)",
                                        entry.name, sz_mb,
                                        static_cast<uint64_t>(entry.inode));
      if (report_out != nullptr) {
        absl::StrAppend(
            report_out,
            absl::StrFormat("  - %-28s: %6.2f MB (Inode %llu)\n", entry.name,
                            sz_mb, static_cast<uint64_t>(entry.inode)));
      }
    }
  }
}

}  // namespace litert::lm
