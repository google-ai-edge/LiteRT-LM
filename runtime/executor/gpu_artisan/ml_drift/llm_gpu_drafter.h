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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_LLM_GPU_DRAFTER_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_LLM_GPU_DRAFTER_H_

#include <atomic>
#include <memory>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/executor/gpu_artisan/ml_drift/drafter.h"
#include "runtime/executor/gpu_artisan/ml_drift/llm.h"
#include "runtime/executor/gpu_artisan/ml_drift/speculative_utils.h"

namespace odml::infra::gpu {

class LlmGpuRunner;

class LlmGpuDrafter : public Drafter {
 public:
  explicit LlmGpuDrafter(std::unique_ptr<LlmGpuRunner> model);
  ~LlmGpuDrafter();

  // Drafter:
  LlmGpuRunner* GetGpuRunner() override;
  absl::Status InitInputTokens(const std::vector<int>& input_tokens,
                               int current_step, bool wait_for_completion,
                               const std::atomic_bool* cancel) override;
  absl::StatusOr<std::vector<int>> DraftTokensAfter(
      const ml_drift::speculative_utils::TokenAtTimeStep& token) override;
  using Drafter::CleanCache;
  absl::Status CleanCache(int time_step,
                          bool skip_ringbuffer_delta_check) override;
  absl::StatusOr<std::vector<ml_drift::speculative_utils::TokenProbability>>
  GetProbabilityCache(int time_step) override;
  absl::Status RejectAfterResample(
      const ml_drift::speculative_utils::TokenAtTimeStep& token) override;
  absl::StatusOr<std::unique_ptr<ml_drift::ContextData>> SaveContext() override;
  absl::StatusOr<std::unique_ptr<ml_drift::ContextData>> SaveLocalContext()
      override;
  absl::Status RestoreContext(
      std::unique_ptr<ml_drift::ContextData> drafter_context_data) override;
  absl::Status RestoreLocalContext(
      std::unique_ptr<ml_drift::ContextData> drafter_context_data) override;
  absl::Status RestoreContextWithCacheCleaning(
      const ml_drift::ContextData& context_data) override;

 private:
  std::unique_ptr<LlmGpuRunner> model_;
};

}  // namespace odml::infra::gpu

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_LLM_GPU_DRAFTER_H_
