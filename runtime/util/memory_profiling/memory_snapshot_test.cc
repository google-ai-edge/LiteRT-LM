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

#include <cstdint>
#include <optional>
#include <sstream>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/log_severity.h"  // from @com_google_absl
#include "absl/log/scoped_mock_log.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "runtime/util/memory_profiling/accelerator_memory_monitor.h"
#include "runtime/util/memory_profiling/host_memory.h"

namespace litert::lm {
namespace {

using ::testing::_;
using ::testing::AnyNumber;
using ::testing::HasSubstr;
using ::testing::Not;
using ::testing::StartsWith;

constexpr uint64_t kMb = 1024 * 1024;

HostMemoryUsage Host(std::optional<uint64_t> resident_mb,
                     std::optional<uint64_t> attributed_mb) {
  HostMemoryUsage usage;
  if (resident_mb.has_value()) usage.resident_bytes = *resident_mb * kMb;
  if (attributed_mb.has_value()) {
    usage.platform_attributed_bytes = *attributed_mb * kMb;
  }
  return usage;
}

// Accelerator memory as a platform reports it, all in MB: physically present,
// charged, and how much of each the host figures already contain.
AcceleratorMemoryUsage Accelerator(uint64_t resident_mb, uint64_t attributed_mb,
                                   uint64_t resident_counted_by_host_mb,
                                   uint64_t attributed_counted_by_host_mb) {
  AcceleratorMemoryUsage usage;
  usage.total_bytes = resident_mb * kMb;
  usage.attributed_bytes = attributed_mb * kMb;
  usage.resident_counted_by_host_bytes = resident_counted_by_host_mb * kMb;
  usage.attributed_counted_by_host_bytes = attributed_counted_by_host_mb * kMb;
  usage.breakdown = {{"local", resident_mb * kMb}, {"non-local", 0}};
  usage.allocations = {{"Some GPU (local)", resident_mb * kMb}};
  return usage;
}

MATCHER_P3(SplitsAs, total_mb, cpu_mb, accelerator_mb, "") {
  if (!arg.has_value()) {
    *result_listener << "is nullopt";
    return false;
  }
  *result_listener << "is " << FormatMemorySplit(*arg);
  return arg->total_bytes == total_mb * kMb && arg->cpu_bytes == cpu_mb * kMb &&
         arg->accelerator_bytes == accelerator_mb * kMb;
}

// Linux with a discrete GPU: the OS counts no GPU memory in either host
// figure, so the GPU simply adds to both.
TEST(MakeMemorySnapshotTest,
     AcceleratorMemoryTheOsDoesNotCountAddsToBothFigures) {
  const MemorySnapshot snapshot =
      MakeMemorySnapshot(Host(/*resident_mb=*/900, /*attributed_mb=*/1000),
                         Accelerator(/*resident_mb=*/256, /*attributed_mb=*/256,
                                     /*resident_counted_by_host_mb=*/0,
                                     /*attributed_counted_by_host_mb=*/0));
  EXPECT_THAT(snapshot.resident_in_ram, SplitsAs(1156, 900, 256));
  EXPECT_THAT(snapshot.attributed_by_os, SplitsAs(1256, 1000, 256));
}

// Windows: the commit charge already contains every GPU allocation, so the
// GPU's share moves from the CPU side to the accelerator side and the total
// stays Task Manager's figure. Nothing of it is in the working set.
TEST(MakeMemorySnapshotTest,
     AcceleratorMemoryTheOsCountsMovesToTheAcceleratorSide) {
  const MemorySnapshot snapshot =
      MakeMemorySnapshot(Host(900, 1000), Accelerator(256, 256, 0, 256));
  EXPECT_THAT(snapshot.resident_in_ram, SplitsAs(1156, 900, 256));
  EXPECT_THAT(snapshot.attributed_by_os, SplitsAs(1000, 744, 256));
}

// macOS with file-mapped weights: the kernel charges 300 MB of GPU memory to
// the footprint and counts 100 MB of CPU-written GPU buffers in the resident
// size; Metal holds 1300 MB in all, including 1000 MB of weights the kernel
// charges nobody for.
TEST(MakeMemorySnapshotTest, CountsEachPageOnceOnEitherSide) {
  const MemorySnapshot snapshot =
      MakeMemorySnapshot(Host(500, 800), Accelerator(1300, 300, 100, 300));
  EXPECT_THAT(snapshot.resident_in_ram, SplitsAs(1700, 400, 1300));
  // Activity Monitor's 800 MB, of which 300 MB are the GPU's.
  EXPECT_THAT(snapshot.attributed_by_os, SplitsAs(800, 500, 300));
}

TEST(MakeMemorySnapshotTest,
     UnsupportedAcceleratorLeavesTheAcceleratorSideZero) {
  const MemorySnapshot snapshot =
      MakeMemorySnapshot(Host(900, 1000), std::nullopt);
  EXPECT_FALSE(snapshot.accelerator.has_value());
  EXPECT_THAT(snapshot.resident_in_ram, SplitsAs(900, 900, 0));
  EXPECT_THAT(snapshot.attributed_by_os, SplitsAs(1000, 1000, 0));
}

TEST(MakeMemorySnapshotTest, OverlapIsCappedByBothFigures) {
  // The host figures are read before the accelerator's and a reader may
  // report an overlap larger than either figure; the CPU side never goes
  // negative and the overlap never exceeds the accelerator side.
  const MemorySnapshot snapshot =
      MakeMemorySnapshot(Host(2, 1000), Accelerator(200, 200, 3, 300));
  EXPECT_THAT(snapshot.resident_in_ram, SplitsAs(200, 0, 200));
  EXPECT_THAT(snapshot.attributed_by_os, SplitsAs(1000, 800, 200));
}

TEST(MakeMemorySnapshotTest, UnavailableHostFiguresStayNullopt) {
  const MemorySnapshot snapshot =
      MakeMemorySnapshot(HostMemoryUsage(), Accelerator(256, 256, 0, 0));
  EXPECT_FALSE(snapshot.resident_in_ram.has_value());
  EXPECT_FALSE(snapshot.attributed_by_os.has_value());
  // The accelerator side is still there in the detail.
  ASSERT_TRUE(snapshot.accelerator.has_value());
  EXPECT_EQ(snapshot.accelerator->total_bytes, 256 * kMb);
}

TEST(TakeMemorySnapshotTest, ReportsThisProcess) {
  const MemorySnapshot snapshot = TakeMemorySnapshot();
  // Every supported platform reports both figures, and this process
  // certainly has some memory.
  ASSERT_TRUE(snapshot.resident_in_ram.has_value());
  ASSERT_TRUE(snapshot.attributed_by_os.has_value());
  EXPECT_GT(snapshot.resident_in_ram->cpu_bytes, 0);
  EXPECT_GT(snapshot.attributed_by_os->cpu_bytes, 0);
  for (const MemorySplit& split :
       {*snapshot.resident_in_ram, *snapshot.attributed_by_os}) {
    EXPECT_EQ(split.total_bytes, split.cpu_bytes + split.accelerator_bytes);
  }
}

TEST(FormatMemorySplitTest, ShowsTheTotalAndItsParts) {
  EXPECT_EQ(FormatMemorySplit({.total_bytes = 1156 * kMb,
                               .cpu_bytes = 900 * kMb,
                               .accelerator_bytes = 256 * kMb}),
            "1156.00 MB (CPU 900.00 MB + accelerator 256.00 MB)");
}

TEST(MemorySnapshotToStringTest, ListsAvailableQuantitiesOnOneLine) {
  const MemorySnapshot snapshot =
      MakeMemorySnapshot(Host(900, 1000), Accelerator(256, 256, 0, 256));
  EXPECT_EQ(
      snapshot.ToString(),
      "resident in RAM=1156.00 MB (CPU 900.00 MB + accelerator 256.00 MB), "
      "attributed by OS=1000.00 MB (CPU 744.00 MB + accelerator 256.00 MB), "
      "accelerator detail=256.00 MB (local=256.00 MB, non-local=0.00 MB)");
  // The same text comes out of streams and absl string building.
  std::ostringstream stream;
  stream << "after load " << snapshot;
  EXPECT_EQ(stream.str(), absl::StrCat("after load ", snapshot.ToString()));
  EXPECT_EQ(absl::StrCat(snapshot), snapshot.ToString());
}

TEST(MemorySnapshotToStringTest, OmitsUnavailableQuantities) {
  const MemorySnapshot snapshot =
      MakeMemorySnapshot(HostMemoryUsage(), std::nullopt);
  EXPECT_EQ(snapshot.ToString(), "accelerator detail=unsupported");
}

TEST(FormatMemorySnapshotTest, FormatsOneQuantityPerLine) {
  const MemorySnapshot snapshot =
      MakeMemorySnapshot(Host(900, 1000), Accelerator(256, 256, 0, 256));
  EXPECT_EQ(FormatMemorySnapshot(snapshot, "After load"),
            "After load:\n"
            "  Memory resident in RAM: 1156.00 MB (CPU 900.00 MB + accelerator "
            "256.00 MB)\n"
            "  Memory attributed to the process by the OS: 1000.00 MB (CPU "
            "744.00 MB + accelerator 256.00 MB)\n"
            "  Accelerator memory detail: 256.00 MB (local=256.00 MB, "
            "non-local=0.00 MB)\n"
            "    -   256.00 MB  Some GPU (local)");
}

TEST(FormatMemorySnapshotTest, SaysWhenAcceleratorMemoryIsUnsupported) {
  const MemorySnapshot snapshot = MakeMemorySnapshot(Host(8, 10), std::nullopt);
  const std::string text = FormatMemorySnapshot(snapshot);
  EXPECT_THAT(text, HasSubstr("Memory usage:\n"));
  EXPECT_THAT(text, HasSubstr("  Memory resident in RAM: 8.00 MB (CPU 8.00 MB "
                              "+ accelerator 0.00 MB)\n"));
  EXPECT_THAT(text, HasSubstr("  Accelerator memory detail: unsupported"));
  EXPECT_THAT(text, Not(HasSubstr("    - ")));
}

TEST(TakeAndLogMemorySnapshotTest, LogsEachLineAndReturnsTheSnapshot) {
  absl::ScopedMockLog log(absl::MockLogDefault::kDisallowUnexpected);
  // One INFO line per text line of FormatMemorySnapshot(): the title, then an
  // indented line per quantity the platform can measure. (Later expectations
  // take precedence, so the catch-all comes first.)
  EXPECT_CALL(log, Log(absl::LogSeverity::kInfo, _, StartsWith("  ")))
      .Times(AnyNumber());
  EXPECT_CALL(log, Log(absl::LogSeverity::kInfo, _,
                       StartsWith("  Accelerator memory detail: ")));
  EXPECT_CALL(log, Log(absl::LogSeverity::kInfo, _, "After load:"));
  log.StartCapturingLogs();

  const MemorySnapshot snapshot = TakeAndLogMemorySnapshot("After load");

  // The returned snapshot is a live one: it has whatever TakeMemorySnapshot()
  // has on this platform.
  const MemorySnapshot reference = TakeMemorySnapshot();
  EXPECT_EQ(snapshot.resident_in_ram.has_value(),
            reference.resident_in_ram.has_value());
  EXPECT_EQ(snapshot.attributed_by_os.has_value(),
            reference.attributed_by_os.has_value());
  EXPECT_EQ(snapshot.accelerator.has_value(),
            reference.accelerator.has_value());
}

}  // namespace
}  // namespace litert::lm
