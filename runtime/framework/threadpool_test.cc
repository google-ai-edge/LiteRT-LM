// Copyright 2025 The ODML Authors.
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

#include "runtime/framework/threadpool.h"

#include <atomic>
#include <set>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "runtime/framework/thread_options.h"
#include "runtime/util/test_utils.h"  // NOLINT

namespace litert::lm {
namespace {

TEST(ThreadPoolTest, DestroyWithoutStart) {
  ThreadPool thread_pool("testpool", 10);
  EXPECT_EQ(thread_pool.max_num_threads(), 10);
  EXPECT_EQ(thread_pool.num_threads(), 0);
}

TEST(ThreadPoolTest, EmptyThread) {
  ThreadPool thread_pool("testpool", 0);
  EXPECT_EQ(thread_pool.max_num_threads(), 1);
  EXPECT_EQ(thread_pool.num_threads(), 0);
}

TEST(ThreadPoolTest, SingleThread) {
  std::atomic<int> n = 100;
  {
    ThreadPool thread_pool("testpool", 1);
    EXPECT_EQ(thread_pool.max_num_threads(), 1);
    EXPECT_EQ(thread_pool.num_threads(), 0);

    for (uint8_t i = 0; i < 100; ++i) {
      EXPECT_OK(thread_pool.Schedule([&n]() { --n; }));
    }
    EXPECT_EQ(thread_pool.num_threads(), 1);
  }
  EXPECT_EQ(n, 0);
}

TEST(ThreadPoolTest, MultiThreadsScheduledFast) {
  std::atomic<int> n = 100;
  {
    ThreadPool thread_pool("testpool", 10);
    EXPECT_EQ(thread_pool.max_num_threads(), 10);
    EXPECT_EQ(thread_pool.num_threads(), 0);

    // Schedule 100 tasks back to back.
    for (uint8_t i = 0; i < 100; ++i) {
      EXPECT_OK(thread_pool.Schedule([&n]() { --n; }));
    }
    // Need more workers up to max, 10.
    EXPECT_EQ(thread_pool.num_threads(), 10);
  }
  EXPECT_EQ(n, 0);
}

TEST(ThreadPoolTest, MultiThreadsScheduledSlow) {
  std::atomic<int> n = 100;
  {
    ThreadPool thread_pool("testpool", 10);
    EXPECT_EQ(thread_pool.max_num_threads(), 10);
    EXPECT_EQ(thread_pool.num_threads(), 0);

    // Schedule 100 tasks with a delay.
    for (uint8_t i = 0; i < 100; ++i) {
      EXPECT_OK(thread_pool.Schedule([&n]() { --n; }));
      absl::SleepFor(absl::Milliseconds(10));
    }
    // Not many workers are needed since workers are not busy.
    EXPECT_LT(thread_pool.num_threads(), 10);
  }
  EXPECT_EQ(n, 0);
}

TEST(ThreadPoolTest, CreateWithThreadOptions) {
  ThreadPool thread_pool("testpool", 10, ThreadOptions());
  EXPECT_EQ(thread_pool.max_num_threads(), 10);
  EXPECT_EQ(thread_pool.num_threads(), 0);
}

TEST(ThreadPoolTest, CreateWithThreadPriority) {
  ThreadOptions thread_options = ThreadOptions().set_nice_priority_level(-10);
  ThreadPool thread_pool("testpool", 10, thread_options);
  EXPECT_EQ(thread_pool.max_num_threads(), 10);
  EXPECT_EQ(thread_pool.num_threads(), 0);
  EXPECT_EQ(thread_pool.thread_options().nice_priority_level(), -10);
}

TEST(ThreadPoolTest, CreateWithCPUAffinity) {
  ThreadOptions thread_options = ThreadOptions().set_cpu_set({0});
  ThreadPool thread_pool("testpool", 10, thread_options);
  EXPECT_EQ(thread_pool.max_num_threads(), 10);
  EXPECT_EQ(thread_pool.num_threads(), 0);
  EXPECT_EQ(thread_pool.thread_options().cpu_set().size(), 1);
}

TEST(ThreadPoolTest, WaitUntilIdle) {
  ThreadPool thread_pool("testpool", 1);
  EXPECT_EQ(thread_pool.max_num_threads(), 1);
  EXPECT_EQ(thread_pool.num_threads(), 0);

  absl::Mutex mu;
  std::vector<int> v;
  for (uint8_t i = 0; i < 10; ++i) {
    EXPECT_OK(thread_pool.Schedule([&v, &mu, i]() {
      // Simulate a task that takes some time to execute.
      absl::SleepFor(absl::Milliseconds(50));
      absl::MutexLock l(&mu);
      v.push_back(i);
    }));
    EXPECT_EQ(thread_pool.num_threads(), 1);
  }
  EXPECT_OK(thread_pool.WaitUntilIdle(absl::Seconds(50)));
  // WaitUntilIdle() should wait until the task queue is empty. Note that when
  // the function returns, the last task is still being executed so the vector
  // will only have 9 elements instead of 10.
  {
    absl::MutexLock l(&mu);
    EXPECT_THAT(v, testing::ElementsAre(0, 1, 2, 3, 4, 5, 6, 7, 8));
  }
  // Wait all tasks to be done before destroying v.
  EXPECT_OK(thread_pool.WaitUntilDone(absl::Seconds(50)));
}

TEST(ThreadPoolTest, WaitUntilDone) {
  ThreadPool thread_pool("testpool", 1);
  EXPECT_EQ(thread_pool.max_num_threads(), 1);
  EXPECT_EQ(thread_pool.num_threads(), 0);

  absl::Mutex mu;
  std::vector<int> v;
  for (uint8_t i = 0; i < 10; ++i) {
    EXPECT_OK(thread_pool.Schedule([&v, &mu, i]() {
      // Simulate a task that takes some time to execute.
      absl::SleepFor(absl::Milliseconds(50));
      absl::MutexLock l(&mu);
      v.push_back(i);
    }));
    EXPECT_EQ(thread_pool.num_threads(), 1);
  }
  EXPECT_OK(thread_pool.WaitUntilDone(absl::Seconds(50)));
  // Even without destroying the thread pool and force all threads to join,
  // calling WaitUntilDone() should be enough to ensure that all of the
  // scheduled tasks are executed.
  EXPECT_THAT(v, testing::ElementsAre(0, 1, 2, 3, 4, 5, 6, 7, 8, 9));
}

}  // namespace
}  // namespace litert::lm
