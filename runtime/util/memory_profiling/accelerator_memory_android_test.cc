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

// Device-only checks of the Android implementation. The parsers and the rule
// that combines the sources are tested on Linux in
// accelerator_memory_dmabuf_test.cc, proc_mappings_test.cc and
// accelerator_memory_android_sources_test.cc.

#include "runtime/util/memory_profiling/accelerator_memory_android.h"

#include <cstdint>
#include <optional>

#include <gtest/gtest.h>
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "runtime/util/memory_profiling/accelerator_memory_monitor.h"
#include "runtime/util/memory_profiling/proc_mappings.h"

namespace litert::lm {
namespace {

TEST(GetProcessAcceleratorMemoryUsageTest, Android_IsSupported) {
  // Backed by the process's own /proc files, which are always readable, so
  // the monitor is live on every Android device even when nothing is
  // allocated.
  ASSERT_TRUE(GetProcessAcceleratorMemoryUsage().has_value());
}

TEST(GetProcessAcceleratorMemoryUsageTest, Android_FiguresAreConsistent) {
  // A test process uses no GPU, so everything should be zero or near it; what
  // matters is that the figures hang together on this device's kernel.
  const std::optional<AcceleratorMemoryUsage> usage =
      GetProcessAcceleratorMemoryUsage();
  ASSERT_TRUE(usage.has_value());
  uint64_t breakdown_total = 0;
  for (const auto& [_, bytes] : usage->breakdown) breakdown_total += bytes;
  EXPECT_EQ(breakdown_total, usage->total_bytes);
  EXPECT_EQ(usage->attributed_bytes, usage->total_bytes);
  EXPECT_LE(usage->resident_counted_by_host_bytes, usage->total_bytes);
  ABSL_LOG(INFO) << "total=" << usage->total_bytes
                 << " counted_by_host=" << usage->resident_counted_by_host_bytes
                 << " breakdown=" << FormatAcceleratorMemoryBreakdown(*usage);
}

TEST(ProcMappingsTest, Android_OwnProcessIsReadable) {
  // The three /proc files the implementation relies on are readable by an
  // unprivileged process on this device.
  EXPECT_FALSE(ReadProcMaps("/proc/self").empty());
  EXPECT_TRUE(ReadProcStatusResident("/proc/self").has_value());
  EXPECT_TRUE(ReadSmapsRollupRss("/proc/self").has_value());
}

}  // namespace
}  // namespace litert::lm
