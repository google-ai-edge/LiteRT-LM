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

#include "runtime/engine/performance/cpu_performance_utils.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl

namespace litert::lm {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;

TEST(CpuPerformanceUtilsTest, SetCpuAffinityReturnsOk) {
  EXPECT_THAT(SetCpuAffinity({}), IsOk());
  EXPECT_THAT(SetCpuAffinity({0}), IsOk());
}

TEST(CpuPerformanceUtilsTest, SetPriorityLevelsReturnOk) {
  EXPECT_THAT(SetProcessPriority(CpuPriorityLevel::kNormal), IsOk());
  EXPECT_THAT(SetProcessPriority(CpuPriorityLevel::kMedium), IsOk());
  EXPECT_THAT(SetProcessPriority(CpuPriorityLevel::kHigh), IsOk());
  EXPECT_THAT(SetProcessPriority(CpuPriorityLevel::kRealtime), IsOk());

  EXPECT_THAT(SetThreadPriority(CpuPriorityLevel::kNormal), IsOk());
  EXPECT_THAT(SetThreadPriority(CpuPriorityLevel::kMedium), IsOk());
  EXPECT_THAT(SetThreadPriority(CpuPriorityLevel::kHigh), IsOk());
  EXPECT_THAT(SetThreadPriority(CpuPriorityLevel::kRealtime), IsOk());
}

TEST(CpuPerformanceUtilsTest, EnableCpuPerformanceBoostersReturnsOk) {
  EXPECT_THAT(EnableCpuPerformanceBoosters({0}, CpuPriorityLevel::kHigh),
              IsOk());
}

TEST(CpuPerformanceUtilsTest, BusyWaitPollingTest) {
  auto start = absl::Now();
  BusyWaitPolling(5000000);  // 5 ms
  auto duration = absl::Now() - start;
  EXPECT_GE(duration, absl::Milliseconds(4));
}

TEST(CpuPerformanceUtilsTest, BusyWaitPollingZeroOrNegative) {
  auto start = absl::Now();
  BusyWaitPolling(0);
  BusyWaitPolling(-100);
  auto duration = absl::Now() - start;
  EXPECT_LT(duration, absl::Milliseconds(5));
}

}  // namespace
}  // namespace litert::lm
