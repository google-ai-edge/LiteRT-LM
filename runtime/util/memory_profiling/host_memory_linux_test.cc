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

#include "runtime/util/memory_profiling/host_memory_linux.h"

#include <cstdint>
#include <memory>
#include <optional>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "runtime/util/memory_profiling/host_memory.h"

namespace litert::lm {
namespace {

using ::testing::Eq;
using ::testing::Ge;
using ::testing::Gt;
using ::testing::Optional;

constexpr uint64_t kKb = 1024;

TEST(ParseProcStatusTest, ReadsResidentAndAttributedMemory) {
  // Excerpt of a real /proc/self/status, with unrelated lines in between as in
  // the real file.
  const HostMemoryUsage usage = ParseProcStatus(
      "Name:\tlitert_lm_main\n"
      "VmPeak:\t 9876543 kB\n"
      "VmSize:\t 8765432 kB\n"
      "VmHWM:\t 2000000 kB\n"
      "VmRSS:\t 1500000 kB\n"
      "RssAnon:\t 1000000 kB\n"
      "RssFile:\t  450000 kB\n"
      "RssShmem:\t   50000 kB\n"
      "VmSwap:\t  120000 kB\n"
      "Threads:\t12\n");
  EXPECT_THAT(usage.resident_bytes, Optional(Eq(1500000 * kKb)));
  // Resident (file-backed pages included) plus swapped; not the high-water
  // mark (VmHWM).
  EXPECT_THAT(usage.platform_attributed_bytes,
              Optional(Eq((1500000 + 120000) * kKb)));
}

TEST(ParseProcStatusTest, MissingSwapLineMeansNoSwap) {
  const HostMemoryUsage usage = ParseProcStatus("VmRSS:\t 100 kB\n");
  EXPECT_THAT(usage.resident_bytes, Optional(Eq(100 * kKb)));
  EXPECT_THAT(usage.platform_attributed_bytes, Optional(Eq(100 * kKb)));
}

TEST(ParseProcStatusTest, LeavesFieldsUnsetWhenLinesAreMissingOrMalformed) {
  EXPECT_EQ(ParseProcStatus("").resident_bytes, std::nullopt);
  EXPECT_EQ(ParseProcStatus("").platform_attributed_bytes, std::nullopt);
  // Swap alone is not enough to say anything.
  EXPECT_EQ(ParseProcStatus("VmSwap:\t 5 kB\n").platform_attributed_bytes,
            std::nullopt);
  // Wrong unit, no number, and a key that merely shares a prefix.
  const HostMemoryUsage usage = ParseProcStatus(
      "VmRSS:\t 1500000 MB\n"
      "VmSwap:\t kB\n"
      "VmRSSExtra:\t 5 kB\n");
  EXPECT_EQ(usage.resident_bytes, std::nullopt);
  EXPECT_EQ(usage.platform_attributed_bytes, std::nullopt);
}

TEST(GetHostMemoryUsageTest, Linux_ReportsThisProcess) {
  const HostMemoryUsage before = GetHostMemoryUsage();
  ASSERT_TRUE(before.resident_bytes.has_value());
  ASSERT_TRUE(before.platform_attributed_bytes.has_value());
  EXPECT_THAT(*before.resident_bytes, Gt(0));
  EXPECT_THAT(*before.platform_attributed_bytes, Gt(0));

  // Touch one byte of every page of a 64 MB anonymous block: both figures
  // must grow by about that much, as the pages are both resident and
  // anonymous. (Volatile so the stores cannot be optimized away.)
  constexpr uint64_t kBytes = 64 * 1024 * 1024;
  constexpr uint64_t kPageBytes = 4096;
  auto block = std::make_unique<char[]>(kBytes);
  volatile char* pages = block.get();
  for (uint64_t i = 0; i < kBytes; i += kPageBytes) pages[i] = 1;
  const HostMemoryUsage after = GetHostMemoryUsage();
  EXPECT_THAT(*after.resident_bytes, Ge(*before.resident_bytes + kBytes / 2));
  EXPECT_THAT(*after.platform_attributed_bytes,
              Ge(*before.platform_attributed_bytes + kBytes / 2));
}

}  // namespace
}  // namespace litert::lm
