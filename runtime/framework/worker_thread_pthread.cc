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

#include <errno.h>
#include <pthread.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <memory>
#include <ostream>
#include <set>
#include <string>
#include <utility>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "runtime/framework/thread_options.h"
#include "runtime/framework/threadpool.h"
#include "runtime/framework/worker_thread.h"

namespace litert::lm {
namespace {

// Create a thread name from the given prefix and thread id.
// - thread_id is not portable
// - the 16-byte limit is Linux-specific
// - the std::thread implementation has a copy of this but doesn't use it
// - why do we even need the thread id in the name? any thread list should show
//   the id too.
std::string CreateThreadName(const std::string& prefix, int thread_id) {
  std::string name = absl::StrCat(prefix, "/", thread_id);
  // 16 is the limit allowed by `pthread_setname_np`, including
  // the terminating null byte ('\0')
  constexpr size_t kMaxThreadNameLength = 15;
  name.resize(std::min(name.length(), kMaxThreadNameLength));
  return name;
}

class WorkerThreadPthread : public WorkerThread {
 public:
  WorkerThreadPthread(ThreadPool* absl_nonnull pool,
                      const std::string& name_prefix);

  // Starts the thread and reports the status.
  absl::Status Start();

 private:
  absl::Status JoinImpl() override;

  static void* ThreadBody(void* arg);

  pthread_t thread_;
};

WorkerThreadPthread::WorkerThreadPthread(ThreadPool* absl_nonnull pool,
                                         const std::string& name_prefix)
    : WorkerThread(pool, name_prefix) {}

absl::Status WorkerThreadPthread::Start() {
  int res = pthread_create(&thread_, nullptr, ThreadBody, this);
  if (res == 0) {
    return absl::OkStatus();
  }

  return absl::ErrnoToStatus(
      res, absl::StrCat("pthread_create failed for pool ", name_prefix_, ": ",
                        strerror(res)));
}

absl::Status WorkerThreadPthread::JoinImpl() {
  int res = pthread_join(thread_, nullptr);
  if (res == 0) {
    return absl::OkStatus();
  }

  return absl::ErrnoToStatus(
      res, absl::StrCat("pthread_join failed for pool ", name_prefix_, ": ",
                        strerror(res)));
}

void* WorkerThreadPthread::ThreadBody(void* arg) {
  auto thread = reinterpret_cast<WorkerThreadPthread*>(arg);
  int nice_priority_level =
      thread->pool_.thread_options().nice_priority_level();
  const std::set<int> selected_cpus = thread->pool_.thread_options().cpu_set();
#if defined(__linux__)
  const std::string name =
      CreateThreadName(thread->name_prefix_, syscall(SYS_gettid));
  if (nice_priority_level != 0) {
    if (nice(nice_priority_level) != -1 || errno == 0) {
      ABSL_LOG(INFO) << "Changed the nice priority level by "
                     << nice_priority_level;
    } else {
      ABSL_LOG(ERROR) << "Error : " << strerror(errno) << std::endl
                      << "Could not change the nice priority level by "
                      << nice_priority_level;
    }
  }
  if (!selected_cpus.empty()) {
    cpu_set_t cpu_set;
    CPU_ZERO(&cpu_set);
    for (const int cpu : selected_cpus) {
      CPU_SET(cpu, &cpu_set);
    }
    if (sched_setaffinity(syscall(SYS_gettid), sizeof(cpu_set_t), &cpu_set) !=
            -1 ||
        errno == 0) {
      ABSL_LOG(INFO) << "Pinned the thread pool executor to processor "
                     << absl::StrJoin(selected_cpus, ", processor ") << ".";
    } else {
      ABSL_LOG(ERROR) << "Error : " << strerror(errno) << std::endl
                      << "Failed to set processor affinity. Ignore processor "
                         "affinity setting for now.";
    }
  }
  int error = pthread_setname_np(pthread_self(), name.c_str());
  if (error != 0) {
    ABSL_LOG(ERROR) << "Error : " << strerror(error) << std::endl
                    << "Failed to set name for thread: " << name;
  }
#else
  const std::string name = CreateThreadName(thread->name_prefix_, 0);
  if (nice_priority_level != 0 || !selected_cpus.empty()) {
    ABSL_LOG(ERROR) << "Thread priority and processor affinity feature aren't "
                       "supported on the current platform.";
  }
#if __APPLE__
  int error = pthread_setname_np(name.c_str());
  if (error != 0) {
    ABSL_LOG(ERROR) << "Error : " << strerror(error) << std::endl
                    << "Failed to set name for thread: " << name;
  }
#endif  // __APPLE__
#endif  // __linux__

  thread->RunWorker();
  return nullptr;
}

}  // namespace

absl::StatusOr<std::unique_ptr<WorkerThread>> WorkerThread::Create(
    ThreadPool* absl_nonnull pool, const std::string& name_prefix) {
  auto worker = std::make_unique<WorkerThreadPthread>(pool, name_prefix);
  auto status = worker->Start();
  if (!status.ok()) {
    return status;
  }
  return std::move(worker);
}

}  // namespace litert::lm
