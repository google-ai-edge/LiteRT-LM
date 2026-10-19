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

#include "runtime/util/memory_profiling/accelerator_memory.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl

namespace litert::lm {

namespace {

// Allocations smaller than this are omitted from the diagnostic listing.
constexpr uint64_t kMinLoggedAllocationBytes = 1024 * 1024;

}  // namespace

float BytesToMb(uint64_t bytes) {
  return static_cast<float>(static_cast<double>(bytes) / (1024.0 * 1024.0));
}

std::string FormatAcceleratorMemoryBreakdown(
    const AcceleratorMemoryUsage& usage) {
  return absl::StrJoin(
      usage.breakdown, ", ",
      [](std::string* out, const std::pair<std::string, uint64_t>& entry) {
        absl::StrAppendFormat(out, "%s=%.2f MB", entry.first,
                              BytesToMb(entry.second));
      });
}

std::vector<std::string> FormatAcceleratorMemoryAllocations(
    const AcceleratorMemoryUsage& usage) {
  std::vector<std::string> lines;
  for (const auto& [label, bytes] : usage.allocations) {
    if (bytes < kMinLoggedAllocationBytes) continue;
    // The size comes before the label so that generic "<name>: <value> <unit>"
    // log parsers do not mistake these lines for metrics.
    lines.push_back(
        absl::StrFormat("  - %8.2f MB  %s", BytesToMb(bytes), label));
  }
  return lines;
}

}  // namespace litert::lm
