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

#ifndef THIRD_PARTY_LITERT_LM_RUNTIME_FRAMEWORK_WORKER_THREAD_H_
#define THIRD_PARTY_LITERT_LM_RUNTIME_FRAMEWORK_WORKER_THREAD_H_

#include <atomic>
#include <memory>
#include <string>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/framework/threadpool.h"

namespace litert::lm {

class WorkerThread {
 public:
  // Creates and starts a thread that runs pool->RunWorker().
  static absl::StatusOr<std::unique_ptr<WorkerThread>> Create(
      ThreadPool* absl_nonnull pool, const std::string& name_prefix);

  // REQUIRES: Join() must have been called.
  virtual ~WorkerThread();

  // Joins with the running thread.
  absl::Status Join();

 protected:
  WorkerThread(ThreadPool* absl_nonnull pool, const std::string& name_prefix);

  // The implementation of Join().
  virtual absl::Status JoinImpl() = 0;

  // For the visibility from WorkerThread subclasses.
  void RunWorker();

  ThreadPool& pool_;
  const std::string name_prefix_;

  // Track if this thread was successfully started.
  std::atomic<bool> started_;

  // Track if this thread is joined.
  std::atomic<bool> joined_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_LITERT_LM_RUNTIME_FRAMEWORK_WORKER_THREAD_H_
