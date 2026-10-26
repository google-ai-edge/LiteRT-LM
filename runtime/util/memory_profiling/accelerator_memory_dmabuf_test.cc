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

// The dma-buf reader on canned fdinfo files, so that it can run on Linux in CI;
// the live check against a device is in accelerator_memory_android_test.cc.

#include "runtime/util/memory_profiling/accelerator_memory_dmabuf.h"

#include <sys/stat.h>

#include <cstdint>
#include <fstream>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/util/memory_profiling/accelerator_memory_monitor.h"

namespace litert::lm {
namespace {

using ::testing::ElementsAre;
using ::testing::FieldsAre;
using ::testing::IsEmpty;
using ::testing::Pair;

constexpr uint64_t kMb = 1024 * 1024;

// A fake /proc/self directory, private to the current test: `fdinfo/<fd>`
// files with the given contents and no `fd/` directory, so inode lookups
// through `fd/` fail the way they do on a kernel that does not expose them.
class FakeProcSelf {
 public:
  FakeProcSelf()
      : dir_(absl::StrCat(
            testing::TempDir(), "/proc_self_",
            testing::UnitTest::GetInstance()->current_test_info()->name())) {
    mkdir(dir_.c_str(), 0755);
    mkdir(absl::StrCat(dir_, "/fdinfo").c_str(), 0755);
  }

  void AddFd(int fd, absl::string_view fdinfo) {
    std::ofstream file(absl::StrCat(dir_, "/fdinfo/", fd));
    file << fdinfo;
  }

  const std::string& dir() const { return dir_; }

 private:
  std::string dir_;
};

TEST(ReadDmaBufSummaryTest, MissingDirectoryIsEmpty) {
  const DmaBufSummary summary =
      ReadDmaBufSummary(absl::StrCat(testing::TempDir(), "/does_not_exist"));
  EXPECT_EQ(summary.total_bytes, 0);
  EXPECT_THAT(summary.entries, IsEmpty());
}

TEST(ReadDmaBufSummaryTest, ParsesDmaBufsAndSkipsOtherDescriptors) {
  FakeProcSelf proc;
  // A regular file: no size / exp_name, ignored.
  proc.AddFd(3, "pos:\t0\nflags:\t02100002\nmnt_id:\t28\nino:\t77\n");
  // A DMA-BUF as a current kernel reports it (tab separated, with inode).
  proc.AddFd(4,
             "pos:\t0\nflags:\t02000002\nmnt_id:\t15\nino:\t1001\n"
             "size:\t4194304\ncount:\t2\nexp_name:\tsystem-uncached\n"
             "name:\tweights\n");
  // The same buffer mapped through a second descriptor: counted once.
  proc.AddFd(5,
             "pos:\t0\nflags:\t02000002\nmnt_id:\t15\nino:\t1001\n"
             "size:\t4194304\ncount:\t2\nexp_name:\tsystem-uncached\n"
             "name:\tweights\n");
  // A buffer with a size but no exporter is not a DMA-BUF (e.g. a memfd).
  proc.AddFd(6, "pos:\t0\nflags:\t02\nsize:\t8388608\n");
  // An older kernel: space separated, no inode and no name. The inode falls
  // back to the descriptor number and the name to the exporter.
  proc.AddFd(7, "pos: 0\nsize: 2097152\nexp_name: system\n");

  const DmaBufSummary summary = ReadDmaBufSummary(proc.dir());

  EXPECT_EQ(summary.total_bytes, 6 * kMb);
  // Largest first.
  EXPECT_THAT(
      summary.entries,
      ElementsAre(FieldsAre(1001, 4 * kMb, "weights", "system-uncached"),
                  FieldsAre(7, 2 * kMb, "system", "system")));
}

TEST(ReadDmaBufSummaryTest, EqualSizesAreOrderedByInode) {
  FakeProcSelf proc;
  proc.AddFd(9, "ino:\t20\nsize:\t1048576\nexp_name:\tsystem\n");
  proc.AddFd(8, "ino:\t10\nsize:\t1048576\nexp_name:\tsystem\n");

  EXPECT_THAT(ReadDmaBufSummary(proc.dir()).entries,
              ElementsAre(FieldsAre(10, kMb, "system", "system"),
                          FieldsAre(20, kMb, "system", "system")));
}

TEST(AcceleratorMemoryUsageFromDmaBufTest, EmptySummaryIsZero) {
  const AcceleratorMemoryUsage usage =
      AcceleratorMemoryUsageFromDmaBuf(DmaBufSummary());
  EXPECT_EQ(usage.total_bytes, 0);
  EXPECT_THAT(usage.breakdown, IsEmpty());
  EXPECT_THAT(usage.allocations, IsEmpty());
}

TEST(AcceleratorMemoryUsageFromDmaBufTest,
     BreaksDownByExporterAndListsBuffers) {
  DmaBufSummary dmabuf;
  dmabuf.total_bytes = 17 * kMb;
  // Largest first, as produced by ReadDmaBufSummary().
  dmabuf.entries = {
      {.inode = 1,
       .size_bytes = 10 * kMb,
       .name = "weights",
       .exp_name = "system-uncached"},
      {.inode = 2, .size_bytes = 4 * kMb, .name = "ahwb", .exp_name = "system"},
      {.inode = 3,
       .size_bytes = 2 * kMb,
       .name = "kv_cache",
       .exp_name = "system-uncached"},
      {.inode = 4,
       .size_bytes = 1 * kMb,
       .name = "scratch",
       .exp_name = "qcom,dsp"},
  };

  const AcceleratorMemoryUsage usage = AcceleratorMemoryUsageFromDmaBuf(dmabuf);

  EXPECT_EQ(usage.total_bytes, 17 * kMb);
  // Per-exporter totals, largest first, summing to the total.
  EXPECT_THAT(usage.breakdown,
              ElementsAre(Pair("system-uncached", 12 * kMb),
                          Pair("system", 4 * kMb), Pair("qcom,dsp", 1 * kMb)));
  // One allocation per buffer, in the original order, labelled by name and
  // inode so individual buffers can be told apart.
  EXPECT_THAT(usage.allocations,
              ElementsAre(Pair("weights (Inode 1)", 10 * kMb),
                          Pair("ahwb (Inode 2)", 4 * kMb),
                          Pair("kv_cache (Inode 3)", 2 * kMb),
                          Pair("scratch (Inode 4)", 1 * kMb)));
}

TEST(AcceleratorMemoryUsageFromDmaBufTest, EqualExportersAreOrderedByName) {
  DmaBufSummary dmabuf;
  dmabuf.total_bytes = 2 * kMb;
  dmabuf.entries = {
      {.inode = 1, .size_bytes = kMb, .name = "b", .exp_name = "zeta"},
      {.inode = 2, .size_bytes = kMb, .name = "a", .exp_name = "alpha"},
  };

  EXPECT_THAT(AcceleratorMemoryUsageFromDmaBuf(dmabuf).breakdown,
              ElementsAre(Pair("alpha", kMb), Pair("zeta", kMb)));
}

}  // namespace
}  // namespace litert::lm
