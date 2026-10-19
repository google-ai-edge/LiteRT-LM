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

// Tests of the rule that combines the Android sources, on inputs modelled on
// the four devices it was measured on (see accelerator_memory_android.cc).

#include <cstdint>
#include <optional>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "runtime/util/memory_profiling/accelerator_memory.h"
#include "runtime/util/memory_profiling/accelerator_memory_android.h"
#include "runtime/util/memory_profiling/accelerator_memory_dmabuf.h"
#include "runtime/util/memory_profiling/proc_mappings.h"

namespace litert::lm {
namespace {

using ::testing::Contains;
using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::Pair;

constexpr uint64_t kMb = 1024 * 1024;

ProcMapping Mapping(uint64_t size, const char* path) {
  return {.start = 0x7000000000, .end = 0x7000000000 + size, .path = path};
}

DmaBufSummary DmaBufs(uint64_t bytes, const char* heap) {
  DmaBufSummary dmabuf;
  dmabuf.total_bytes = bytes;
  dmabuf.entries.push_back(
      {.inode = 1, .size_bytes = bytes, .name = heap, .exp_name = heap});
  return dmabuf;
}

TEST(SummarizeGpuDriverMappingsTest, ClassifiesByDeviceNode) {
  const GpuDriverMappings driver = SummarizeGpuDriverMappings({
      // KGSL lists its allocations as deleted shmem files named after the
      // device, plus the node itself.
      Mapping(100 * kMb, "/kgsl-3d0 (deleted)"),
      Mapping(20 * kMb, "/dev/kgsl-3d0"),
      Mapping(300 * kMb, "/dev/dri/renderD128"),
      Mapping(5 * kMb, "/dev/dri/renderD129"),
      Mapping(1 * kMb, "/dev/dri/card0"),  // Display node: not GPU memory.
      Mapping(900 * kMb, "/dev/mali0"),    // VA reservation: not counted.
      Mapping(50 * kMb, "/dmabuf:system"),
      Mapping(10 * kMb, "[anon:scudo:primary]"),
      Mapping(10 * kMb, ""),
  });
  EXPECT_EQ(driver.kgsl_bytes, 120 * kMb);
  EXPECT_EQ(driver.drm_bytes, 305 * kMb);
  EXPECT_TRUE(driver.has_mali);

  EXPECT_FALSE(SummarizeGpuDriverMappings({Mapping(kMb, "")}).has_mali);
}

TEST(UnmappedRssBytesTest, DifferenceOrZero) {
  const ProcResidentCounters status{.vm_rss_bytes = 2030 * kMb};
  EXPECT_EQ(UnmappedRssBytes(status, 789 * kMb), 1241 * kMb);
  EXPECT_EQ(UnmappedRssBytes(status, 2030 * kMb), 0);
  EXPECT_EQ(UnmappedRssBytes(status, 2100 * kMb), 0);  // Counter slack.
  EXPECT_EQ(UnmappedRssBytes(status, std::nullopt), 0);
}

// Pixel 9 Pro, Mali: 390 MB of dma-bufs the CPU never touched, plus 1241 MB
// that kbase counted into VmRSS without mapping.
TEST(AcceleratorMemoryUsageFromAndroidSourcesTest, Mali) {
  const AcceleratorMemoryUsage usage = AcceleratorMemoryUsageFromAndroidSources(
      DmaBufs(390 * kMb, "system-uncached"),
      {.kgsl_bytes = 0, .drm_bytes = 0, .has_mali = true},
      {.status = {.vm_rss_bytes = 2030 * kMb, .rss_shmem_bytes = 0},
       .unmapped_rss_bytes = 1241 * kMb});
  EXPECT_EQ(usage.total_bytes, (390 + 1241) * kMb);
  EXPECT_EQ(usage.attributed_bytes, usage.total_bytes);
  // The Mali pool is inside VmRSS; the dma-bufs are not.
  EXPECT_EQ(usage.resident_counted_by_host_bytes, 1241 * kMb);
  EXPECT_EQ(usage.attributed_counted_by_host_bytes, 1241 * kMb);
  EXPECT_THAT(usage.breakdown,
              ElementsAre(Pair("gpu driver (mali)", 1241 * kMb),
                          Pair("system-uncached", 390 * kMb)));
  EXPECT_THAT(
      usage.allocations,
      Contains(Pair("GPU driver pool, counted in VmRSS by kbase", 1241 * kMb)));
}

// Galaxy S25 Ultra, Adreno: no dma-bufs, 1474 MB mapped from KGSL of which
// the CPU touched 174 MB (RssShmem).
TEST(AcceleratorMemoryUsageFromAndroidSourcesTest, Adreno) {
  const AcceleratorMemoryUsage usage = AcceleratorMemoryUsageFromAndroidSources(
      DmaBufSummary(), {.kgsl_bytes = 1474 * kMb},
      {.status = {.vm_rss_bytes = 730 * kMb, .rss_shmem_bytes = 174 * kMb}});
  EXPECT_EQ(usage.total_bytes, 1474 * kMb);
  EXPECT_EQ(usage.attributed_bytes, 1474 * kMb);
  EXPECT_EQ(usage.resident_counted_by_host_bytes, 174 * kMb);
  EXPECT_THAT(usage.breakdown,
              ElementsAre(Pair("gpu driver (kgsl)", 1474 * kMb)));
  EXPECT_THAT(
      usage.allocations,
      ElementsAre(Pair("GPU driver pool, KGSL shmem mappings (/kgsl-3d0)",
                       1474 * kMb)));
}

// Pixel 11 Pro (PowerVR) and Exynos (Xclipse): mapped from a DRM render node,
// nothing resident on the CPU side.
TEST(AcceleratorMemoryUsageFromAndroidSourcesTest, DrmRenderNode) {
  const AcceleratorMemoryUsage usage = AcceleratorMemoryUsageFromAndroidSources(
      DmaBufSummary(), {.drm_bytes = 1350 * kMb},
      {.status = {.vm_rss_bytes = 1080 * kMb, .rss_shmem_bytes = 0}});
  EXPECT_EQ(usage.total_bytes, 1350 * kMb);
  EXPECT_EQ(usage.resident_counted_by_host_bytes, 0);
  EXPECT_THAT(usage.breakdown,
              ElementsAre(Pair("gpu driver (drm)", 1350 * kMb)));
}

// Galaxy S25 Ultra, Hexagon NPU: dma-bufs only, CPU never touched them.
TEST(AcceleratorMemoryUsageFromAndroidSourcesTest, NpuDmaBufsOnly) {
  const AcceleratorMemoryUsage usage = AcceleratorMemoryUsageFromAndroidSources(
      DmaBufs(915 * kMb, "system"), GpuDriverMappings(),
      {.status = {.vm_rss_bytes = 220 * kMb, .rss_shmem_bytes = 0}});
  EXPECT_EQ(usage.total_bytes, 915 * kMb);
  EXPECT_EQ(usage.resident_counted_by_host_bytes, 0);
  EXPECT_THAT(usage.breakdown, ElementsAre(Pair("system", 915 * kMb)));
}

// The CPU touched part of the dma-bufs: that part is in RSS and is taken out
// of the CPU side. Shmem the process uses for other things cannot push the
// overlap past what the accelerator side claims.
TEST(AcceleratorMemoryUsageFromAndroidSourcesTest, TouchedDmaBufsAreCapped) {
  const AcceleratorMemoryUsage touched =
      AcceleratorMemoryUsageFromAndroidSources(
          DmaBufs(100 * kMb, "system"), GpuDriverMappings(),
          {.status = {.vm_rss_bytes = 500 * kMb, .rss_shmem_bytes = 60 * kMb}});
  EXPECT_EQ(touched.resident_counted_by_host_bytes, 60 * kMb);

  const AcceleratorMemoryUsage capped =
      AcceleratorMemoryUsageFromAndroidSources(
          DmaBufs(100 * kMb, "system"), GpuDriverMappings(),
          {.status = {.vm_rss_bytes = 500 * kMb,
                      .rss_shmem_bytes = 300 * kMb}});
  EXPECT_EQ(capped.resident_counted_by_host_bytes, 100 * kMb);
}

// A Mali pool is only believed when a Mali node is mapped; the same VmRSS gap
// on another GPU (counter slack, a driver we have not seen) is left alone.
TEST(AcceleratorMemoryUsageFromAndroidSourcesTest, UnmappedRssNeedsMali) {
  const AcceleratorMemoryUsage usage = AcceleratorMemoryUsageFromAndroidSources(
      DmaBufSummary(), {.kgsl_bytes = 100 * kMb, .has_mali = false},
      {.status = {.vm_rss_bytes = 500 * kMb}, .unmapped_rss_bytes = 50 * kMb});
  EXPECT_EQ(usage.total_bytes, 100 * kMb);
  EXPECT_EQ(usage.resident_counted_by_host_bytes, 0);
}

TEST(AcceleratorMemoryUsageFromAndroidSourcesTest, NothingInUseIsZero) {
  const AcceleratorMemoryUsage usage = AcceleratorMemoryUsageFromAndroidSources(
      DmaBufSummary(), GpuDriverMappings(), AndroidResidentView());
  EXPECT_EQ(usage.total_bytes, 0);
  EXPECT_THAT(usage.breakdown, IsEmpty());
  EXPECT_THAT(usage.allocations, IsEmpty());
}

}  // namespace
}  // namespace litert::lm
