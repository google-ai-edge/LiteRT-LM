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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_DRAFTER_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_DRAFTER_H_

#include <atomic>
#include <memory>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/executor/gpu_artisan/ml_drift/llm.h"
#include "runtime/executor/gpu_artisan/ml_drift/speculative_utils.h"

namespace odml::infra::gpu {

class LlmGpuRunner;

// An interface for an object that drafts (guesses) tokens for speculative
// decoding. This may be implemented with a smaller LLM or a heuristic
// algorithm.
class Drafter {
 public:
  // Returns the underlying model, if any.
  virtual LlmGpuRunner* GetGpuRunner() { return nullptr; }

  // Initializes the drafter with the input tokens.
  virtual absl::Status InitInputTokens(const std::vector<int>& input_tokens,
                                       int current_step,
                                       bool wait_for_completion,
                                       const std::atomic_bool* cancel) = 0;

  // Updates/sets the token at token.time_step() and drafts an additional
  // `gamma_` tokens afterwards.
  virtual absl::StatusOr<std::vector<int>> DraftTokensAfter(
      const ml_drift::speculative_utils::TokenAtTimeStep& token) = 0;

  // Clean the cache including and after the given current time step (the number
  // of tokens).
  virtual absl::Status CleanCache(int time_step,
                                  bool skip_ringbuffer_delta_check) = 0;
  absl::Status CleanCache(int time_step) {
    return CleanCache(time_step, /*skip_ringbuffer_delta_check=*/false);
  }

  // Gets tokens and probabilities for the given time step, typically `k` tokens
  // will be returned, but some algorithms may ignore `k`.
  virtual absl::StatusOr<
      std::vector<ml_drift::speculative_utils::TokenProbability>>
  GetProbabilityCache(int time_step) = 0;

  // See LlmGpuRunner::RejectAfterResample. This is called after a drafted token
  // is rejected by the main model.
  virtual absl::Status RejectAfterResample(
      const ml_drift::speculative_utils::TokenAtTimeStep& token) = 0;

  // Saves the context of the drafter.
  virtual absl::StatusOr<std::unique_ptr<ml_drift::ContextData>>
  SaveContext() = 0;

  // Saves the local context of the drafter.
  virtual absl::StatusOr<std::unique_ptr<ml_drift::ContextData>>
  SaveLocalContext() = 0;

  // Restores the context of the drafter.
  virtual absl::Status RestoreContext(
      std::unique_ptr<ml_drift::ContextData> context_data) = 0;

  // Restores the local context of the drafter.
  virtual absl::Status RestoreLocalContext(
      std::unique_ptr<ml_drift::ContextData> context_data) = 0;

  virtual absl::Status RestoreContextWithCacheCleaning(
      const ml_drift::ContextData& context_data) = 0;

  virtual ~Drafter() = default;

 protected:
  Drafter() = default;
};

}  // namespace odml::infra::gpu

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_DRAFTER_H_
