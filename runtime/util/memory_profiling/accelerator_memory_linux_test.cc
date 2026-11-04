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

#include "runtime/util/memory_profiling/accelerator_memory_linux.h"

#include <cstdint>
#include <optional>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "runtime/util/memory_profiling/accelerator_memory_monitor.h"
#include "runtime/util/memory_profiling/accelerator_memory_nvml.h"
#include "runtime/util/memory_profiling/fake_proc_self.h"
#include "runtime/util/memory_profiling/fdinfo.h"

namespace litert::lm {
namespace {

using ::testing::ElementsAre;
using ::testing::Field;
using ::testing::IsEmpty;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;

constexpr uint64_t kKb = 1024;
constexpr uint64_t kMb = 1024 * 1024;

// An amdgpu client on a 6.x kernel: both the current `drm-total-*` keys and
// the older `drm-memory-*` ones, the `drm-resident-*` keys, plus the shared /
// purgeable variants that must not be counted.
constexpr char kAmdgpuFdinfo[] =
    "pos:\t0\n"
    "flags:\t02100002\n"
    "mnt_id:\t26\n"
    "ino:\t1159\n"
    "drm-driver:\tamdgpu\n"
    "drm-client-id:\t17\n"
    "drm-pdev:\t0000:03:00.0\n"
    "pasid:\t32771\n"
    "drm-engine-gfx:\t123456789 ns\n"
    "drm-memory-vram:\t1048576 KiB\n"
    "drm-memory-gtt:\t262144 KiB\n"
    "drm-memory-cpu:\t0 KiB\n"
    "drm-total-vram:\t1048576 KiB\n"
    "drm-total-gtt:\t262144 KiB\n"
    "drm-total-cpu:\t0 KiB\n"
    "drm-shared-vram:\t0 KiB\n"
    "drm-resident-vram:\t524288 KiB\n"
    "drm-resident-gtt:\t262144 KiB\n"
    "drm-resident-cpu:\t0 KiB\n"
    "drm-purgeable-vram:\t0 KiB\n";

MATCHER_P3(RegionIs, name, total_bytes, resident_bytes, "") {
  return arg.name == name && arg.total_bytes == total_bytes &&
         arg.resident_bytes == resident_bytes;
}

TEST(ParseDrmFdinfoTest, ReadsIdentityTotalsAndResidentPerRegion) {
  const std::optional<DrmClientMemory> client =
      ParseDrmFdinfo(ParseFdinfo(kAmdgpuFdinfo));
  ASSERT_TRUE(client.has_value());
  EXPECT_EQ(client->driver, "amdgpu");
  EXPECT_EQ(client->pdev, "0000:03:00.0");
  EXPECT_EQ(client->client_id, "17");
  // drm-total-* and drm-resident-*, in bytes, ordered by region; shared /
  // purgeable ignored.
  EXPECT_THAT(
      client->regions,
      ElementsAre(RegionIs("cpu", 0, 0), RegionIs("gtt", 256 * kMb, 256 * kMb),
                  RegionIs("vram", 1024 * kMb, 512 * kMb)));
}

TEST(ParseDrmFdinfoTest, FallsBackToLegacyMemoryKeys) {
  // amdgpu on a 5.19-6.1 kernel printed only drm-memory-*, with no residency:
  // everything allocated is taken to be present.
  const std::optional<DrmClientMemory> client =
      ParseDrmFdinfo(ParseFdinfo("drm-driver:\tamdgpu\n"
                                 "drm-client-id:\t3\n"
                                 "drm-memory-vram:\t2048 KiB\n"
                                 "drm-memory-gtt:\t1 MiB\n"));
  ASSERT_TRUE(client.has_value());
  EXPECT_THAT(client->regions, ElementsAre(RegionIs("gtt", kMb, kMb),
                                           RegionIs("vram", 2 * kMb, 2 * kMb)));
}

TEST(ParseDrmFdinfoTest, ResidencyReportedForSomeRegionsOnly) {
  const std::optional<DrmClientMemory> client =
      ParseDrmFdinfo(ParseFdinfo("drm-driver:\txe\n"
                                 "drm-client-id:\t5\n"
                                 "drm-total-system:\t100 MiB\n"
                                 "drm-total-vram0:\t200 MiB\n"
                                 "drm-resident-vram0:\t150 MiB\n"));
  ASSERT_TRUE(client.has_value());
  EXPECT_THAT(client->regions,
              ElementsAre(RegionIs("system", 100 * kMb, 100 * kMb),
                          RegionIs("vram0", 200 * kMb, 150 * kMb)));
}

TEST(ParseDrmFdinfoTest, HandlesEveryUnitAndNonPciGpus) {
  const std::optional<DrmClientMemory> client =
      ParseDrmFdinfo(ParseFdinfo("drm-driver:\tmsm\n"
                                 "drm-client-id:\t9\n"
                                 "drm-total-memory:\t3 GiB\n"
                                 "drm-total-bytes:\t512\n"
                                 "drm-total-kib:\t2 KiB\n"));
  ASSERT_TRUE(client.has_value());
  EXPECT_EQ(client->pdev, "");
  EXPECT_THAT(client->regions, ElementsAre(RegionIs("bytes", 512, 512),
                                           RegionIs("kib", 2 * kKb, 2 * kKb),
                                           RegionIs("memory", 3ull * 1024 * kMb,
                                                    3ull * 1024 * kMb)));
}

TEST(ParseDrmFdinfoTest, DriverWithoutMemoryStatsHasNoRegions) {
  const std::optional<DrmClientMemory> client =
      ParseDrmFdinfo(ParseFdinfo("drm-driver:\tnouveau\n"
                                 "drm-client-id:\t4\n"
                                 "drm-engine-gr:\t1000 ns\n"));
  ASSERT_TRUE(client.has_value());
  EXPECT_THAT(client->regions, IsEmpty());
}

TEST(ParseDrmFdinfoTest, NonDrmDescriptorIsNullopt) {
  EXPECT_FALSE(ParseDrmFdinfo(ParseFdinfo("pos:\t0\nino:\t77\n")).has_value());
  EXPECT_FALSE(ParseDrmFdinfo(ParseFdinfo("size:\t4096\nexp_name:\tsystem\n"))
                   .has_value());
}

TEST(ReadDrmClientsTest, CountsEachClientOnceAndSkipsOtherDescriptors) {
  FakeProcSelf proc;
  proc.AddFd(3, "pos:\t0\nino:\t77\n");  // A regular file.
  proc.AddFd(4, kAmdgpuFdinfo);
  proc.AddFd(5, kAmdgpuFdinfo);  // dup() of the same client.
  // A second GPU with the same driver and, by coincidence, the same client
  // id: distinct because the PCI address differs.
  proc.AddFd(6,
             "drm-driver:\tamdgpu\n"
             "drm-client-id:\t17\n"
             "drm-pdev:\t0000:0a:00.0\n"
             "drm-total-vram:\t4096 KiB\n");

  EXPECT_THAT(
      ReadDrmClients(proc.dir()),
      UnorderedElementsAre(Field(&DrmClientMemory::pdev, "0000:03:00.0"),
                           Field(&DrmClientMemory::pdev, "0000:0a:00.0")));
}

TEST(AcceleratorMemoryUsageFromDrmClientsTest, NoClientsIsZero) {
  const std::optional<AcceleratorMemoryUsage> usage =
      AcceleratorMemoryUsageFromDrmClients({});
  ASSERT_TRUE(usage.has_value());
  EXPECT_EQ(usage->total_bytes, 0);
  EXPECT_EQ(usage->attributed_bytes, 0);
  EXPECT_THAT(usage->breakdown, IsEmpty());
  EXPECT_THAT(usage->allocations, IsEmpty());
}

TEST(AcceleratorMemoryUsageFromDrmClientsTest,
     ClientsWithoutMemoryStatsAreUnsupported) {
  EXPECT_FALSE(AcceleratorMemoryUsageFromDrmClients(
                   {{.driver = "nouveau", .client_id = "4"}})
                   .has_value());
}

TEST(AcceleratorMemoryUsageFromDrmClientsTest,
     SumsRegionsAcrossClientsAndListsEachClient) {
  const std::vector<DrmClientMemory> clients = {
      {.driver = "amdgpu",
       .pdev = "0000:03:00.0",
       .client_id = "17",
       .regions = {{"cpu", 0, 0},
                   {"gtt", 256 * kMb, 256 * kMb},
                   // Half the VRAM allocation has been evicted.
                   {"vram", 1024 * kMb, 512 * kMb}}},
      {.driver = "amdgpu",
       .pdev = "0000:03:00.0",
       .client_id = "18",
       .regions = {{"gtt", 64 * kMb, 64 * kMb}, {"vram", 0, 0}}},
      {.driver = "i915",
       .pdev = "0000:00:02.0",
       .client_id = "2",
       .regions = {{"system0", 32 * kMb, 32 * kMb}}},
  };

  const std::optional<AcceleratorMemoryUsage> usage =
      AcceleratorMemoryUsageFromDrmClients(clients);

  ASSERT_TRUE(usage.has_value());
  // Present is the resident bytes, attributed the allocated bytes.
  EXPECT_EQ(usage->total_bytes, (512 + 256 + 64 + 32) * kMb);
  EXPECT_EQ(usage->attributed_bytes, (1024 + 256 + 64 + 32) * kMb);
  // Linux charges the process for none of it and counts none of it in RSS.
  EXPECT_EQ(usage->resident_counted_by_host_bytes, 0);
  EXPECT_EQ(usage->attributed_counted_by_host_bytes, 0);
  // Per-region resident totals over all clients, largest first; zero regions
  // kept so the shape is stable.
  EXPECT_THAT(usage->breakdown,
              ElementsAre(Pair("vram", 512 * kMb), Pair("gtt", 320 * kMb),
                          Pair("system0", 32 * kMb), Pair("cpu", 0)));
  // Per-client detail names the GPU, largest first, zero entries omitted.
  EXPECT_THAT(
      usage->allocations,
      ElementsAre(Pair("amdgpu 0000:03:00.0 client 17 (vram)", 512 * kMb),
                  Pair("amdgpu 0000:03:00.0 client 17 (gtt)", 256 * kMb),
                  Pair("amdgpu 0000:03:00.0 client 18 (gtt)", 64 * kMb),
                  Pair("i915 0000:00:02.0 client 2 (system0)", 32 * kMb)));
}

TEST(AcceleratorMemoryUsageFromLinuxSourcesTest, WithoutNvmlIsJustDrm) {
  // No NVIDIA driver: DRM's verdict stands, including "unsupported".
  EXPECT_FALSE(AcceleratorMemoryUsageFromLinuxSources(
                   {{.driver = "nouveau", .client_id = "4"}}, std::nullopt)
                   .has_value());
  const std::optional<AcceleratorMemoryUsage> usage =
      AcceleratorMemoryUsageFromLinuxSources({}, std::nullopt);
  ASSERT_TRUE(usage.has_value());
  EXPECT_EQ(usage->total_bytes, 0);
}

TEST(AcceleratorMemoryUsageFromLinuxSourcesTest,
     NvmlCoversDrmClientsWithoutStatistics) {
  // nvidia-drm clients report no memory; with NVML available that is not
  // "unsupported" but "ask NVML".
  const std::vector<DrmClientMemory> drm = {
      {.driver = "nvidia-drm", .pdev = "0000:01:00.0", .client_id = "5"}};
  const std::vector<NvmlDeviceMemory> nvml = {
      {.index = 0, .name = "NVIDIA H100 80GB HBM3", .used_bytes = 512 * kMb}};

  const std::optional<AcceleratorMemoryUsage> usage =
      AcceleratorMemoryUsageFromLinuxSources(drm, nvml);

  ASSERT_TRUE(usage.has_value());
  EXPECT_EQ(usage->total_bytes, 512 * kMb);
  // NVML's one figure is both present and allocated; none of it in the
  // host's figures.
  EXPECT_EQ(usage->attributed_bytes, 512 * kMb);
  EXPECT_EQ(usage->resident_counted_by_host_bytes, 0);
  EXPECT_EQ(usage->attributed_counted_by_host_bytes, 0);
  EXPECT_THAT(usage->breakdown, ElementsAre(Pair("vram", 512 * kMb)));
  EXPECT_THAT(usage->allocations,
              ElementsAre(Pair("NVIDIA H100 80GB HBM3 #0 (vram)", 512 * kMb)));
}

TEST(AcceleratorMemoryUsageFromLinuxSourcesTest, MergesNvidiaIntoDrmRegions) {
  // An AMD iGPU through DRM and an NVIDIA dGPU through NVML in one machine;
  // a second NVIDIA GPU the process does not use is omitted.
  const std::vector<DrmClientMemory> drm = {
      {.driver = "amdgpu",
       .pdev = "0000:06:00.0",
       .client_id = "3",
       // 32 MB allocated in VRAM, half of it evicted.
       .regions = {{"gtt", 64 * kMb, 64 * kMb}, {"vram", 32 * kMb, 16 * kMb}}}};
  const std::vector<NvmlDeviceMemory> nvml = {
      {.index = 0, .name = "NVIDIA RTX 4070", .used_bytes = 1024 * kMb},
      {.index = 1, .name = "NVIDIA RTX 4070", .used_bytes = 0}};

  const std::optional<AcceleratorMemoryUsage> usage =
      AcceleratorMemoryUsageFromLinuxSources(drm, nvml);

  ASSERT_TRUE(usage.has_value());
  EXPECT_EQ(usage->total_bytes, (1024 + 64 + 16) * kMb);
  EXPECT_EQ(usage->attributed_bytes, (1024 + 64 + 32) * kMb);
  // NVIDIA framebuffer joins the "vram" region.
  EXPECT_THAT(usage->breakdown,
              ElementsAre(Pair("vram", 1040 * kMb), Pair("gtt", 64 * kMb)));
  EXPECT_THAT(
      usage->allocations,
      ElementsAre(Pair("NVIDIA RTX 4070 #0 (vram)", 1024 * kMb),
                  Pair("amdgpu 0000:06:00.0 client 3 (gtt)", 64 * kMb),
                  Pair("amdgpu 0000:06:00.0 client 3 (vram)", 16 * kMb)));
}

TEST(GetProcessAcceleratorMemoryUsageTest, Linux_ReportsSomethingConsistent) {
  // The test machine may have no GPU (zero), a GPU with memory statistics
  // (some value), or one without (unsupported). Whatever it is, the breakdown
  // must add up and nothing may be claimed as counted by the host.
  const std::optional<AcceleratorMemoryUsage> usage =
      GetProcessAcceleratorMemoryUsage();
  if (!usage.has_value()) GTEST_SKIP() << "No per-process GPU memory stats.";
  uint64_t breakdown_total = 0;
  for (const auto& [_, bytes] : usage->breakdown) breakdown_total += bytes;
  EXPECT_EQ(breakdown_total, usage->total_bytes);
  EXPECT_EQ(usage->resident_counted_by_host_bytes, 0);
  EXPECT_EQ(usage->attributed_counted_by_host_bytes, 0);
}

}  // namespace
}  // namespace litert::lm
