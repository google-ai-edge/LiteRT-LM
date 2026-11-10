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

#include "runtime/executor/gpu_artisan/ml_drift/litert_thread_pool_executor.h"

#include <atomic>
#include <memory>

#include <gtest/gtest.h>
#include "absl/synchronization/notification.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "ml_drift/common/executor.h"  // from @ml_drift
#include "runtime/util/test_utils.h"

namespace odml::infra::gpu {
namespace {

TEST(LitertThreadPoolExecutorTest, SchedulesAndRunsTasks) {
  LitertThreadPoolExecutorFactory factory;
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<::ml_drift::Executor> executor,
                       factory.CreateExecutor(/*num_threads=*/2));

  absl::Notification done_first;
  absl::Notification done_second;
  std::atomic<int> counter{0};

  executor->Schedule([&]() {
    counter.fetch_add(1);
    done_first.Notify();
  });
  executor->Schedule([&]() {
    counter.fetch_add(1);
    done_second.Notify();
  });

  EXPECT_TRUE(done_first.WaitForNotificationWithTimeout(absl::Seconds(5)));
  EXPECT_TRUE(done_second.WaitForNotificationWithTimeout(absl::Seconds(5)));
  EXPECT_EQ(counter.load(), 2);
}

TEST(LitertThreadPoolExecutorTest,
     DefaultsNonPositiveThreadCountToSingleWorker) {
  LitertThreadPoolExecutorFactory factory;
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<::ml_drift::Executor> executor,
                       factory.CreateExecutor(/*num_threads=*/0));

  absl::Notification done;
  executor->Schedule([&]() { done.Notify(); });
  EXPECT_TRUE(done.WaitForNotificationWithTimeout(absl::Seconds(5)));
}

}  // namespace
}  // namespace odml::infra::gpu
