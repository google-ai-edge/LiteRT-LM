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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_EXECUTOR_FACTORY_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_EXECUTOR_FACTORY_H_

#include <memory>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "ml_drift/common/executor.h"  // from @ml_drift

namespace odml::infra::gpu {

// Factory interface for creating `ml_drift::Executor` background task runners.
class MlDriftExecutorFactory {
 public:
  virtual ~MlDriftExecutorFactory() = default;

  // Creates an executor with the given number of worker threads. If
  // `num_threads` is non-positive, an implementation-defined positive thread
  // count is used.
  virtual absl::StatusOr<std::unique_ptr<::ml_drift::Executor>> CreateExecutor(
      int num_threads) = 0;
};

}  // namespace odml::infra::gpu

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_EXECUTOR_FACTORY_H_
