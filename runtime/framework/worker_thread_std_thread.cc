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

#include <atomic>
#include <memory>
#include <string>
#include <thread>  // NOLINT(build/c++11)

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/framework/threadpool.h"
#include "runtime/framework/worker_thread.h"

namespace litert::lm {
namespace {

class WorkerThreadStdThread : public WorkerThread {
 public:
  WorkerThreadStdThread(ThreadPool* absl_nonnull pool,
                        const std::string& name_prefix);

 private:
  absl::Status JoinImpl() override;

  static void* ThreadBody(void* arg);

  std::thread thread_;
};

WorkerThreadStdThread::WorkerThreadStdThread(ThreadPool* absl_nonnull pool,
                                             const std::string& name_prefix)
    : WorkerThread(pool, name_prefix) {
  thread_ = std::thread(ThreadBody, this);
  started_ = true;
}

absl::Status WorkerThreadStdThread::JoinImpl() {
  thread_.join();
  return absl::OkStatus();
}

void* WorkerThreadStdThread::ThreadBody(void* arg) {
  auto thread = reinterpret_cast<WorkerThreadStdThread*>(arg);
  thread->RunWorker();
  return nullptr;
}

}  // namespace

absl::StatusOr<std::unique_ptr<WorkerThread>> WorkerThread::Create(
    ThreadPool* absl_nonnull pool, const std::string& name_prefix) {
  return std::make_unique<WorkerThreadStdThread>(pool, name_prefix);
}

}  // namespace litert::lm
