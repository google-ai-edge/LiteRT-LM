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

// Live test of GetHostMemoryUsage() on the platforms whose implementation is
// a single system call (Windows, macOS / iOS). The Linux / Android one has a
// parser and its own test in host_memory_linux_test.cc.

#include "runtime/util/memory_profiling/host_memory.h"

#include <cstdint>
#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace litert::lm {
namespace {

using ::testing::Ge;
using ::testing::Gt;

TEST(GetHostMemoryUsageTest, ReportsThisProcess) {
  const HostMemoryUsage before = GetHostMemoryUsage();
  ASSERT_TRUE(before.resident_bytes.has_value());
  ASSERT_TRUE(before.platform_attributed_bytes.has_value());
  EXPECT_THAT(*before.resident_bytes, Gt(0));
  EXPECT_THAT(*before.platform_attributed_bytes, Gt(0));

  // Touch one byte of every page of a 64 MB anonymous block: both figures
  // must grow by about that much, as the pages are both resident and private.
  // (Volatile so the stores cannot be optimized away.)
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
