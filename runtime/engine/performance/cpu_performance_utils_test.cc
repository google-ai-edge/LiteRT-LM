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

namespace litert::lm {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;

TEST(CpuPerformanceUtilsTest, DefaultFallbacksReturnOk) {
  EXPECT_THAT(SetCpuAffinity({4, 5, 6, 7}), IsOk());
  EXPECT_THAT(SetProcessPriority(CpuPriorityLevel::kHigh), IsOk());
  EXPECT_THAT(SetThreadPriority(CpuPriorityLevel::kHigh), IsOk());
  EXPECT_THAT(
      EnableCpuPerformanceBoosters({4, 5, 6, 7}, CpuPriorityLevel::kHigh),
      IsOk());
}

}  // namespace
}  // namespace litert::lm
