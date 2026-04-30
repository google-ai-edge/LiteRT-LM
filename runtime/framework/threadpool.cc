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

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "runtime/framework/thread_options.h"
#include "runtime/framework/worker_thread.h"

namespace litert::lm {

ThreadPool::ThreadPool(const std::string& name_prefix, size_t max_num_threads,
                       ThreadOptions thread_options)
    : name_prefix_(name_prefix),
      max_num_threads_(max_num_threads == 0 ? 1 : max_num_threads),
      thread_options_(std::move(thread_options)) {
  ABSL_LOG(INFO) << "ThreadPool '" << name_prefix_ << "': Running up to "
                 << max_num_threads_ << " threads.";
}

ThreadPool::~ThreadPool() {
  ABSL_LOG(INFO) << "ThreadPool '" << name_prefix_ << "': Shutting down...";

  std::vector<std::unique_ptr<WorkerThread>> threads_to_join;
  {
    absl::MutexLock lock(mutex_);
    stopped_ = true;
    threads_to_join.swap(threads_);
  }

  for (auto& thread_ptr : threads_to_join) {
    // Wait for each worker thread to finish.
    auto status = thread_ptr->Join();
    if (!status.ok()) {
      ABSL_LOG(ERROR) << "Failed to join worker thread: " << status;
    }
  }

  {
    absl::MutexLock lock(mutex_);
    if (!threads_.empty()) {
      ABSL_LOG(ERROR) << "ThreadPool '" << name_prefix_
                      << "': threads_ is not empty during shutdown.";
    }
    if (num_active_tasks_ != 0) {
      ABSL_LOG(ERROR) << "ThreadPool '" << name_prefix_
                      << "': num_active_tasks_ is " << num_active_tasks_
                      << " during shutdown.";
    }
  }
  ABSL_LOG(INFO) << "ThreadPool '" << name_prefix_ << "': Shutdown complete.";
}

absl::Status ThreadPool::Schedule(absl::AnyInvocable<void() &&> callback) {
  absl::MutexLock lock(mutex_);
  if (stopped_) {
    ABSL_LOG(WARNING) << "ThreadPool '" << name_prefix_
                      << "': Schedule called on a stopped pool.";
    return absl::FailedPreconditionError(
        absl::StrCat("ThreadPool '", name_prefix_, "' is stopped."));
  }

  // If all worker threads are (supposed to be) busy, instantiates a new worker
  // thread to run the task.
  size_t num_threads = threads_.size();
  if (num_threads < max_num_threads_) {
    size_t num_tasks = num_active_tasks_ + tasks_.size();
    if (num_threads <= num_tasks) {
      auto thread = WorkerThread::Create(this, name_prefix_);
      if (thread.ok()) {
        threads_.push_back(std::move(*thread));
        ABSL_LOG(INFO) << "ThreadPool '" << name_prefix_
                       << "': Created a worker thread since all " << num_threads
                       << " worker threads are (supposed to be) busy.";
      } else if (num_threads == 0) {
        ABSL_LOG(ERROR) << "ThreadPool '" << name_prefix_
                        << "': Failed to create the first worker thread: "
                        << thread.status();
        // Return the error to the caller since it would be fatal.
        return thread.status();
      } else {
        ABSL_LOG(WARNING) << "ThreadPool '" << name_prefix_
                          << "': Failed to create a worker thread when all "
                          << num_threads
                          << " worker threads are (supposed to be) busy. "
                          << "Waits for some worker threads to finish: "
                          << thread.status();
        // Ignore the error since tasks can still be scheduled by existing
        // worker threads.
      }
    }
  }

  tasks_.push_back(std::move(callback));
  return absl::OkStatus();
}

absl::Status ThreadPool::WaitUntilIdle(absl::Duration timeout) {
  absl::MutexLock lock(mutex_);
  absl::Time deadline = absl::Now() + timeout;
  // Wait until tasks_ is empty OR the deadline is reached.
  auto is_tasks_empty = [this]() {
    mutex_.AssertHeld();
    return tasks_.empty();
  };
  if (mutex_.AwaitWithDeadline(absl::Condition(&is_tasks_empty), deadline)) {
    return absl::OkStatus();
  }
  return absl::DeadlineExceededError(
      absl::StrCat("Timeout waiting for task queue to become idle in pool '",
                   name_prefix_, "'. Tasks still in queue: ", tasks_.size()));
}

absl::Status ThreadPool::WaitUntilDone(absl::Duration timeout) {
  absl::MutexLock lock(mutex_);
  absl::Time deadline = absl::Now() + timeout;
  // Wait until tasks_ is empty OR the deadline is reached.
  auto is_done = [this]() {
    mutex_.AssertHeld();
    return tasks_.empty() && num_active_tasks_ == 0;
  };
  if (mutex_.AwaitWithDeadline(absl::Condition(&is_done), deadline)) {
    return absl::OkStatus();
  }
  return absl::DeadlineExceededError(
      absl::StrCat("Timeout waiting for all tasks to be done in pool '",
                   name_prefix_, "'. Tasks still in queue: ", tasks_.size(),
                   ", Active tasks: ", num_active_tasks_));
}

void ThreadPool::RunWorker() {
  absl::MutexLock lock(mutex_);
  while (true) {
    // Wait until a task is available OR the pool is stopped.
    auto is_task_available_or_stopped = [this]() {
      mutex_.AssertHeld();
      return !tasks_.empty() || stopped_;
    };
    mutex_.Await(absl::Condition(&is_task_available_or_stopped));

    if (tasks_.empty()) {
      if (!stopped_) {
        ABSL_LOG(ERROR) << "ThreadPool '" << name_prefix_
                        << "': Theoretical invariant violation: Worker "
                           "thread woke up with no tasks but not stopped.";
      }
      ABSL_LOG(INFO) << "ThreadPool '" << name_prefix_
                     << "': Worker thread stopped.";
      return;
    }

    auto task_to_run = std::move(tasks_.front());
    tasks_.pop_front();
    ++num_active_tasks_;

    // Execute the task with mutex released.
    mutex_.unlock();
    std::move(task_to_run)();
    mutex_.lock();

    --num_active_tasks_;
  }
}

}  // namespace litert::lm
