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

#include "runtime/util/memory_profiling/memory_snapshot.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/str_split.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/util/memory_profiling/accelerator_memory_monitor.h"
#include "runtime/util/memory_profiling/host_memory.h"

namespace litert::lm {

namespace {

std::string Mb(uint64_t bytes) {
  return absl::StrFormat("%.2f MB", AcceleratorMemoryBytesToMb(bytes));
}

// "<total> (<breakdown>)", or just "<total>" without a breakdown.
std::string AcceleratorSummary(const AcceleratorMemoryUsage& accelerator) {
  std::string summary = Mb(accelerator.total_bytes);
  if (!accelerator.breakdown.empty()) {
    absl::StrAppend(&summary, " (",
                    FormatAcceleratorMemoryBreakdown(accelerator), ")");
  }
  return summary;
}

// Combines a host figure with the accelerator's figure of the same kind.
// `accelerator_bytes` is the accelerator side, `counted_by_host_bytes` the
// part of it the host figure already contains: it moves from the CPU side to
// the accelerator side rather than being counted on both. The host figure is
// read a moment before the accelerator's, so the overlap can transiently
// exceed it; it is capped at what both contain.
std::optional<MemorySplit> Split(std::optional<uint64_t> host_bytes,
                                 uint64_t accelerator_bytes,
                                 uint64_t counted_by_host_bytes) {
  if (!host_bytes.has_value()) return std::nullopt;
  const uint64_t overlap =
      std::min({counted_by_host_bytes, *host_bytes, accelerator_bytes});
  MemorySplit split;
  split.cpu_bytes = *host_bytes - overlap;
  split.accelerator_bytes = accelerator_bytes;
  split.total_bytes = split.cpu_bytes + split.accelerator_bytes;
  return split;
}

}  // namespace

std::string FormatMemorySplit(const MemorySplit& split) {
  return absl::StrCat(Mb(split.total_bytes), " (CPU ", Mb(split.cpu_bytes),
                      " + accelerator ", Mb(split.accelerator_bytes), ")");
}

std::string MemorySnapshot::ToString() const {
  std::vector<std::string> parts;
  if (resident_in_ram.has_value()) {
    parts.push_back(
        absl::StrCat("resident in RAM=", FormatMemorySplit(*resident_in_ram)));
  }
  if (attributed_by_os.has_value()) {
    parts.push_back(absl::StrCat("attributed by OS=",
                                 FormatMemorySplit(*attributed_by_os)));
  }
  parts.push_back(
      absl::StrCat("accelerator detail=", accelerator.has_value()
                                              ? AcceleratorSummary(*accelerator)
                                              : std::string("unsupported")));
  return absl::StrJoin(parts, ", ");
}

std::ostream& operator<<(std::ostream& os, const MemorySnapshot& snapshot) {
  return os << snapshot.ToString();
}

MemorySnapshot MakeMemorySnapshot(
    const HostMemoryUsage& host,
    std::optional<AcceleratorMemoryUsage> accelerator) {
  MemorySnapshot snapshot;
  const AcceleratorMemoryUsage acc =
      accelerator.value_or(AcceleratorMemoryUsage());
  snapshot.resident_in_ram = Split(host.resident_bytes, acc.total_bytes,
                                   acc.resident_counted_by_host_bytes);
  snapshot.attributed_by_os =
      Split(host.platform_attributed_bytes, acc.attributed_bytes,
            acc.attributed_counted_by_host_bytes);
  snapshot.accelerator = std::move(accelerator);
  return snapshot;
}

MemorySnapshot TakeMemorySnapshot() {
  const HostMemoryUsage host = GetHostMemoryUsage();
  return MakeMemorySnapshot(host, GetProcessAcceleratorMemoryUsage());
}

std::string FormatMemorySnapshot(const MemorySnapshot& snapshot,
                                 absl::string_view title) {
  std::vector<std::string> lines;
  lines.push_back(absl::StrCat(title, ":"));
  if (snapshot.resident_in_ram.has_value()) {
    lines.push_back(absl::StrCat("  Memory resident in RAM: ",
                                 FormatMemorySplit(*snapshot.resident_in_ram)));
  }
  if (snapshot.attributed_by_os.has_value()) {
    lines.push_back(
        absl::StrCat("  Memory attributed to the process by the OS: ",
                     FormatMemorySplit(*snapshot.attributed_by_os)));
  }
  if (snapshot.accelerator.has_value()) {
    lines.push_back(absl::StrCat("  Accelerator memory detail: ",
                                 AcceleratorSummary(*snapshot.accelerator)));
    for (const std::string& line :
         FormatAcceleratorMemoryAllocations(*snapshot.accelerator)) {
      lines.push_back(absl::StrCat("  ", line));
    }
  } else {
    lines.push_back("  Accelerator memory detail: unsupported");
  }
  return absl::StrJoin(lines, "\n");
}

MemorySnapshot TakeAndLogMemorySnapshot(absl::string_view title) {
  const MemorySnapshot snapshot = TakeMemorySnapshot();
  for (absl::string_view line :
       absl::StrSplit(FormatMemorySnapshot(snapshot, title), '\n')) {
    ABSL_LOG(INFO) << line;
  }
  return snapshot;
}

}  // namespace litert::lm
