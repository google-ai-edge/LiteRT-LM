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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_LITERT_THREAD_POOL_EXECUTOR_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_LITERT_THREAD_POOL_EXECUTOR_H_

#include <cstddef>
#include <functional>
#include <memory>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "ml_drift/common/executor.h"  // from @ml_drift
#include "runtime/executor/gpu_artisan/ml_drift/executor_factory.h"
#include "runtime/framework/thread_options.h"
#include "runtime/framework/threadpool.h"

namespace odml::infra::gpu {

// `ml_drift::Executor` backed by `litert::lm::ThreadPool`.
class LitertThreadPoolExecutor : public ::ml_drift::Executor {
 public:
  LitertThreadPoolExecutor(size_t max_num_threads,
                           ::litert::lm::ThreadOptions thread_options =
                               ::litert::lm::ThreadOptions());
  ~LitertThreadPoolExecutor() override = default;

  void Schedule(std::function<void()> fn) override;

 private:
  ::litert::lm::ThreadPool pool_;
};

// Creates `LitertThreadPoolExecutor` instances configured for GPU Artisan
// shader compilation and weight uploads.
class LitertThreadPoolExecutorFactory : public MlDriftExecutorFactory {
 public:
  ~LitertThreadPoolExecutorFactory() override = default;

  absl::StatusOr<std::unique_ptr<::ml_drift::Executor>> CreateExecutor(
      int num_threads) override;
};

}  // namespace odml::infra::gpu

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_LITERT_THREAD_POOL_EXECUTOR_H_
