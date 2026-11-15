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

#include "runtime/executor/cancellation_latch.h"

#include <gtest/gtest.h>

namespace litert::lm {
namespace {

// The registration of the per-op checks with LiteRT is tested by the executor
// tests (TextEncoderCancellationScopeTest and VisionCancellationScopeTest).

TEST(CancellationLatchTest, NotCancelled) {
  int num_checks = 0;
  auto should_cancel = [&] {
    ++num_checks;
    return false;
  };
  CancellationLatch latch(should_cancel, /*compiled_models=*/{});
  EXPECT_FALSE(latch.IsCancelled());
  EXPECT_FALSE(latch.IsCancelled());
  EXPECT_EQ(num_checks, 2);
}

TEST(CancellationLatchTest, LatchesTheCancellation) {
  int num_checks = 0;
  // A one-shot predicate, which only returns true on the 2nd call.
  auto should_cancel = [&] { return ++num_checks == 2; };
  CancellationLatch latch(should_cancel, /*compiled_models=*/{});
  EXPECT_FALSE(latch.IsCancelled());
  EXPECT_TRUE(latch.IsCancelled());
  // Still cancelled, without calling the predicate again.
  EXPECT_TRUE(latch.IsCancelled());
  EXPECT_EQ(num_checks, 2);
}

}  // namespace
}  // namespace litert::lm
