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

#include "runtime/util/memory_profiling/proc_mappings.h"

#include <cstdint>
#include <optional>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "runtime/util/memory_profiling/fake_proc_self.h"

namespace litert::lm {
namespace {

using ::testing::ElementsAre;
using ::testing::Field;
using ::testing::IsEmpty;

constexpr uint64_t kKb = 1024;

// Lines as a Galaxy S25 Ultra prints them: a KGSL mapping, an anonymous one,
// a named anonymous one, a file with a space in its path, and a dma-buf.
constexpr char kMaps[] =
    "7000000000-7000010000 rw-s 00000000 00:0e 1234     /dev/kgsl-3d0\n"
    "7000010000-7000020000 rw-p 00000000 00:00 0 \n"
    "7000020000-7000040000 rw-p 00000000 00:00 0        [anon:scudo:primary]\n"
    "7000040000-7000041000 r--p 00001000 fe:21 99       /data/my model.bin\n"
    "7000041000-7000081000 rw-s 00000000 00:0f 77       /dmabuf:system\n"
    "garbage line\n";

TEST(ParseProcMapsTest, ParsesRangesAndPaths) {
  const std::vector<ProcMapping> mappings = ParseProcMaps(kMaps);
  ASSERT_EQ(mappings.size(), 5);
  EXPECT_EQ(mappings[0].start, 0x7000000000);
  EXPECT_EQ(mappings[0].end, 0x7000010000);
  EXPECT_EQ(mappings[0].size_bytes(), 0x10000);
  EXPECT_EQ(mappings[0].path, "/dev/kgsl-3d0");
  EXPECT_EQ(mappings[1].path, "");
  EXPECT_EQ(mappings[2].path, "[anon:scudo:primary]");
  EXPECT_EQ(mappings[3].path, "/data/my model.bin");
  EXPECT_EQ(mappings[4].path, "/dmabuf:system");
  EXPECT_EQ(mappings[4].size_bytes(), 0x40000);
}

TEST(ParseProcMapsTest, EmptyAndMalformedInputYieldNothing) {
  EXPECT_THAT(ParseProcMaps(""), IsEmpty());
  EXPECT_THAT(ParseProcMaps("not a mapping\n"), IsEmpty());
  // end < start is rejected rather than wrapping.
  EXPECT_THAT(ParseProcMaps("7000010000-7000000000 rw-p 00000000 00:00 0\n"),
              IsEmpty());
}

TEST(ReadProcMapsTest, ReadsTheFile) {
  FakeProcSelf proc;
  proc.WriteFile("maps", kMaps);
  EXPECT_THAT(ReadProcMaps(proc.dir()),
              ElementsAre(Field(&ProcMapping::path, "/dev/kgsl-3d0"),
                          Field(&ProcMapping::path, ""),
                          Field(&ProcMapping::path, "[anon:scudo:primary]"),
                          Field(&ProcMapping::path, "/data/my model.bin"),
                          Field(&ProcMapping::path, "/dmabuf:system")));
  EXPECT_THAT(ReadProcMaps("/nonexistent/proc_self"), IsEmpty());
}

TEST(ParseProcStatusResidentTest, ReadsVmRssAndRssShmem) {
  const std::optional<ProcResidentCounters> counters = ParseProcStatusResident(
      "Name:\tlitert_lm\n"
      "VmHWM:\t 2082884 kB\n"
      "VmRSS:\t 2078700 kB\n"
      "RssAnon:\t  510776 kB\n"
      "RssFile:\t 1567560 kB\n"
      "RssShmem:\t     364 kB\n"
      "VmSwap:\t       0 kB\n");
  ASSERT_TRUE(counters.has_value());
  EXPECT_EQ(counters->vm_rss_bytes, 2078700 * kKb);
  EXPECT_EQ(counters->rss_shmem_bytes, 364 * kKb);
}

TEST(ParseProcStatusResidentTest, RequiresVmRss) {
  EXPECT_FALSE(ParseProcStatusResident("RssShmem:\t 364 kB\n").has_value());
  // RssShmem is optional (older kernels do not split RSS).
  const std::optional<ProcResidentCounters> counters =
      ParseProcStatusResident("VmRSS:\t 100 kB\n");
  ASSERT_TRUE(counters.has_value());
  EXPECT_EQ(counters->vm_rss_bytes, 100 * kKb);
  EXPECT_EQ(counters->rss_shmem_bytes, 0);
}

TEST(ParseSmapsRollupRssTest, ReadsRssNotOtherRssLines) {
  EXPECT_EQ(ParseSmapsRollupRss("00400000-7fff00000000 ---p 00000000 00:00 0 "
                                "[rollup]\n"
                                "Rss:              808372 kB\n"
                                "Pss:              738683 kB\n"
                                "Anonymous:        510880 kB\n"),
            808372 * kKb);
  EXPECT_FALSE(ParseSmapsRollupRss("Pss: 1 kB\n").has_value());
}

TEST(ReadProcStatusAndSmapsRollupTest, ReadTheFiles) {
  FakeProcSelf proc;
  proc.WriteFile("status", "VmRSS:\t 2078700 kB\nRssShmem:\t 364 kB\n");
  proc.WriteFile("smaps_rollup", "Rss:\t 808372 kB\n");
  const std::optional<ProcResidentCounters> counters =
      ReadProcStatusResident(proc.dir());
  ASSERT_TRUE(counters.has_value());
  EXPECT_EQ(counters->vm_rss_bytes, 2078700 * kKb);
  EXPECT_EQ(ReadSmapsRollupRss(proc.dir()), 808372 * kKb);
  EXPECT_FALSE(ReadProcStatusResident("/nonexistent/proc_self").has_value());
  EXPECT_FALSE(ReadSmapsRollupRss("/nonexistent/proc_self").has_value());
}

TEST(ProcMappingsLiveTest, Linux_OwnProcessIsReadable) {
  // The real files on the test machine: mappings exist, the counters are
  // consistent (what is mapped is at most what the counters say).
  const std::vector<ProcMapping> mappings = ReadProcMaps("/proc/self");
  EXPECT_FALSE(mappings.empty());
  const std::optional<ProcResidentCounters> counters =
      ReadProcStatusResident("/proc/self");
  ASSERT_TRUE(counters.has_value());
  EXPECT_GT(counters->vm_rss_bytes, 0);
  const std::optional<uint64_t> smaps_rss = ReadSmapsRollupRss("/proc/self");
  if (smaps_rss.has_value()) {
    // No driver inflates the counter in this process; the two agree to
    // within the kernel's per-CPU counter slack.
    EXPECT_NEAR(static_cast<double>(*smaps_rss),
                static_cast<double>(counters->vm_rss_bytes),
                16.0 * 1024 * 1024);
  }
}

}  // namespace
}  // namespace litert::lm
