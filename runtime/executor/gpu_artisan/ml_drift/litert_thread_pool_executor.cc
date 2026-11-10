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

#include <cstddef>
#include <functional>
#include <memory>
#include <utility>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "ml_drift/common/executor.h"  // from @ml_drift
#include "runtime/framework/thread_options.h"

namespace odml::infra::gpu {

LitertThreadPoolExecutor::LitertThreadPoolExecutor(
    size_t max_num_threads, ::litert::lm::ThreadOptions thread_options)
    : pool_("gpu_artisan", max_num_threads, std::move(thread_options)) {}

void LitertThreadPoolExecutor::Schedule(std::function<void()> fn) {
  pool_.Schedule([fn = std::move(fn)]() mutable { fn(); }).IgnoreError();
}

absl::StatusOr<std::unique_ptr<::ml_drift::Executor>>
LitertThreadPoolExecutorFactory::CreateExecutor(int num_threads) {
  const size_t max_threads =
      num_threads > 0 ? static_cast<size_t>(num_threads) : 1;
  ::litert::lm::ThreadOptions thread_options;
#if defined(__linux__) && !defined(__ANDROID__)
  // LLM models need a larger stack size for compiling shaders on Linux.
  thread_options.set_stack_size(1024 * 1024);
#endif
  return std::make_unique<LitertThreadPoolExecutor>(max_threads,
                                                    std::move(thread_options));
}

}  // namespace odml::infra::gpu
