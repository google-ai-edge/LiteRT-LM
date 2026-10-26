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

// Device-only check of the Android implementation. The dma-buf reader it is
// built on is tested on Linux in accelerator_memory_dmabuf_test.cc.

#include <gtest/gtest.h>
#include "runtime/util/memory_profiling/accelerator_memory_monitor.h"

namespace litert::lm {
namespace {

TEST(GetProcessAcceleratorMemoryUsageTest, Android_IsSupported) {
  // Backed by the process's DMA-BUF usage, which is always readable, so the
  // monitor is live on every Android device even when nothing is allocated.
  ASSERT_TRUE(GetProcessAcceleratorMemoryUsage().has_value());
}

}  // namespace
}  // namespace litert::lm
