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

#include "runtime/engine/dmabuf_util.h"

#include <cstdint>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace litert::lm {
namespace {

using ::testing::HasSubstr;
using ::testing::IsEmpty;

TEST(DmaBufUtilTest, GetProcessDmaBufUsageDoesNotCrash) {
  DmaBufSummary summary = GetProcessDmaBufUsage();
#if defined(__ANDROID__)
  // On Android, GetProcessDmaBufUsage attempts to read /proc/self/fdinfo.
  // total_bytes will be >= 0.
  EXPECT_GE(summary.total_bytes, 0);
#else
  // On non-Android platforms, it returns an empty summary.
  EXPECT_EQ(summary.total_bytes, 0);
  EXPECT_THAT(summary.entries, IsEmpty());
#endif
}

TEST(DmaBufUtilTest, LogDmaBufUsageEmptySummaryDoesNotLogOrAppend) {
  DmaBufSummary summary;
  std::string report;
  LogDmaBufUsage(summary, &report);
  EXPECT_TRUE(report.empty());
}

TEST(DmaBufUtilTest, LogDmaBufUsageFormatsCorrectly) {
  DmaBufSummary summary;
  summary.total_bytes = 12 * 1024 * 1024;  // 12 MB
  summary.entries.push_back({
      .inode = 12345,
      .size_bytes = 10 * 1024 * 1024,  // 10 MB (>= 1MB, so listed)
      .name = "test_npu_buffer",
      .exp_name = "qcom,kgsl-3d0",
  });
  summary.entries.push_back({
      .inode = 67890,
      .size_bytes = 2 * 1024 * 1024,  // 2 MB (>= 1MB, so listed)
      .name = "test_weight_buffer",
      .exp_name = "ion",
  });

  std::string report;
  LogDmaBufUsage(summary, &report);

  EXPECT_THAT(
      report,
      HasSubstr("Peak DMA-BUF hardware usage: 12.00 MB (2 unique buffers)"));
  EXPECT_THAT(report, HasSubstr("test_npu_buffer"));
  EXPECT_THAT(report, HasSubstr("10.00 MB (Inode 12345)"));
  EXPECT_THAT(report, HasSubstr("test_weight_buffer"));
  EXPECT_THAT(report, HasSubstr("2.00 MB (Inode 67890)"));
}

TEST(DmaBufUtilTest, LogDmaBufUsageSkipsSmallBuffersBelow1MB) {
  DmaBufSummary summary;
  summary.total_bytes = 512 * 1024;  // 512 KB
  summary.entries.push_back({
      .inode = 99999,
      .size_bytes = 512 * 1024,  // < 1 MB
      .name = "small_buffer",
      .exp_name = "ion",
  });

  std::string report;
  LogDmaBufUsage(summary, &report);

  EXPECT_THAT(
      report,
      HasSubstr("Peak DMA-BUF hardware usage: 0.50 MB (1 unique buffers)"));
  // The summary line is logged, but the buffer detail line should be skipped
  EXPECT_THAT(report, testing::Not(HasSubstr("small_buffer")));
}

}  // namespace
}  // namespace litert::lm
