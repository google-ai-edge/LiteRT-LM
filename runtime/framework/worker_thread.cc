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

#include "runtime/framework/worker_thread.h"

#include <string>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "runtime/framework/threadpool.h"

namespace litert::lm {

WorkerThread::WorkerThread(ThreadPool* absl_nonnull pool,
                           const std::string& name_prefix)
    : pool_(*pool), name_prefix_(name_prefix), joined_(false) {}

WorkerThread::~WorkerThread() {
  if (!joined_) {
    ABSL_LOG(ERROR) << "WorkerThread '" << name_prefix_
                    << "' destroyed without being joined.";
  }
}

absl::Status WorkerThread::Join() {
  if (joined_) {
    return absl::OkStatus();
  }

  joined_ = true;
  return JoinImpl();
}

void WorkerThread::RunWorker() { pool_.RunWorker(); }

}  // namespace litert::lm
