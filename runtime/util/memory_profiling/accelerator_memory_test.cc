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
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace litert::lm {
namespace {

using ::testing::ElementsAre;
using ::testing::FloatEq;
using ::testing::IsEmpty;

constexpr uint64_t kMb = 1024 * 1024;

AcceleratorMemoryUsage MakeUsage(
    uint64_t total_bytes,
    std::vector<std::pair<std::string, uint64_t>> breakdown = {}) {
  AcceleratorMemoryUsage usage;
  usage.total_bytes = total_bytes;
  usage.breakdown = std::move(breakdown);
  return usage;
}

// Whether tracking is supported depends on the platform and is checked by the
// per-platform tests; what every platform must satisfy is the contract on the
// shape of the result.
TEST(GetProcessAcceleratorMemoryUsageTest, BreakdownSumsToTotalWhenSupported) {
  const std::optional<AcceleratorMemoryUsage> usage =
      GetProcessAcceleratorMemoryUsage();
  if (!usage.has_value()) {
    GTEST_SKIP() << "Accelerator memory tracking is unsupported here.";
  }
  if (usage->breakdown.empty()) {
    GTEST_SKIP() << "This platform offers no breakdown.";
  }
  uint64_t breakdown_total = 0;
  for (const auto& [_, bytes] : usage->breakdown) breakdown_total += bytes;
  EXPECT_EQ(breakdown_total, usage->total_bytes);
}

TEST(BytesToMbTest, ConvertsMebibytes) {
  EXPECT_THAT(BytesToMb(0), FloatEq(0.0f));
  EXPECT_THAT(BytesToMb(kMb), FloatEq(1.0f));
  EXPECT_THAT(BytesToMb(kMb + kMb / 2), FloatEq(1.5f));
}

TEST(FormatAcceleratorMemoryBreakdownTest, EmptyBreakdownFormatsAsEmptyString) {
  EXPECT_THAT(FormatAcceleratorMemoryBreakdown(MakeUsage(5 * kMb)), IsEmpty());
}

TEST(FormatAcceleratorMemoryBreakdownTest, FormatsEntriesInOrder) {
  const AcceleratorMemoryUsage usage =
      MakeUsage(3 * kMb, {{"local", 2 * kMb}, {"non-local", kMb}});
  EXPECT_EQ(FormatAcceleratorMemoryBreakdown(usage),
            "local=2.00 MB, non-local=1.00 MB");
}

TEST(FormatAcceleratorMemoryAllocationsTest, NoAllocationsFormatsAsNoLines) {
  EXPECT_THAT(FormatAcceleratorMemoryAllocations(MakeUsage(5 * kMb)),
              IsEmpty());
}

TEST(FormatAcceleratorMemoryAllocationsTest, ListsAllocationsOfAtLeastOneMb) {
  AcceleratorMemoryUsage usage = MakeUsage(13 * kMb);
  usage.allocations = {
      {"weights (Inode 12345)", 10 * kMb},
      {"kv_cache (Inode 67890)", 2 * kMb},
      {"small (Inode 99999)", 512 * 1024},  // Below 1 MiB: omitted.
      {"exactly_one_mb (Inode 1)", kMb},
  };

  EXPECT_THAT(FormatAcceleratorMemoryAllocations(usage),
              ElementsAre("  -    10.00 MB  weights (Inode 12345)",
                          "  -     2.00 MB  kv_cache (Inode 67890)",
                          "  -     1.00 MB  exactly_one_mb (Inode 1)"));
}
}  // namespace
}  // namespace litert::lm
