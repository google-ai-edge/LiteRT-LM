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

#include "runtime/executor/gpu_artisan/ml_drift/llm_gpu_drafter.h"

#include <atomic>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/executor/gpu_artisan/ml_drift/llm.h"
#include "runtime/executor/gpu_artisan/ml_drift/llm_gpu_runner.h"
#include "runtime/executor/gpu_artisan/ml_drift/speculative_utils.h"
#include "runtime/util/status_macros.h"

namespace odml::infra::gpu {

LlmGpuDrafter::LlmGpuDrafter(std::unique_ptr<LlmGpuRunner> model)
    : model_(std::move(model)) {}

LlmGpuDrafter::~LlmGpuDrafter() = default;

LlmGpuRunner* LlmGpuDrafter::GetGpuRunner() { return model_.get(); }

absl::Status LlmGpuDrafter::InitInputTokens(
    const std::vector<int>& input_tokens, int current_step,
    bool wait_for_completion, const std::atomic_bool* cancel) {
#ifdef __EMSCRIPTEN__
  return model_->InitInputTokens(input_tokens, current_step,
                                 wait_for_completion);
#else
  return model_->InitInputTokens(input_tokens, current_step,
                                 wait_for_completion, cancel);
#endif  // __EMSCRIPTEN__
}

absl::StatusOr<std::vector<int>> LlmGpuDrafter::DraftTokensAfter(
    const ml_drift::speculative_utils::TokenAtTimeStep& token) {
  RETURN_IF_ERROR(model_->OverwriteOneInputToken(token));
  return model_->DraftTokens();
}

absl::Status LlmGpuDrafter::CleanCache(int time_step,
                                       bool skip_ringbuffer_delta_check) {
  return model_->CleanCache(time_step, skip_ringbuffer_delta_check);
}

absl::StatusOr<std::vector<ml_drift::speculative_utils::TokenProbability>>
LlmGpuDrafter::GetProbabilityCache(int time_step) {
  return model_->GetProbabilityCache(time_step);
}

absl::Status LlmGpuDrafter::RejectAfterResample(
    const ml_drift::speculative_utils::TokenAtTimeStep& token) {
  return model_->RejectAfterResample(token);
}

absl::StatusOr<std::unique_ptr<ml_drift::ContextData>>
LlmGpuDrafter::SaveContext() {
  ASSIGN_OR_RETURN(auto context_data, model_->SaveContext());
  return std::make_unique<ml_drift::ContextData>(std::move(context_data));
}

absl::StatusOr<std::unique_ptr<ml_drift::ContextData>>
LlmGpuDrafter::SaveLocalContext() {
  ASSIGN_OR_RETURN(auto context_data, model_->SaveLocalContext());
  return std::make_unique<ml_drift::ContextData>(std::move(context_data));
}

absl::Status LlmGpuDrafter::RestoreContext(
    std::unique_ptr<ml_drift::ContextData> context_data) {
  return model_->RestoreContext(std::move(*context_data));
}

absl::Status LlmGpuDrafter::RestoreLocalContext(
    std::unique_ptr<ml_drift::ContextData> context_data) {
  return model_->RestoreLocalContext(std::move(*context_data));
}

absl::Status LlmGpuDrafter::RestoreContextWithCacheCleaning(
    const ml_drift::ContextData& context_data) {
  return model_->RestoreContextWithCacheCleaning(context_data);
}

}  // namespace odml::infra::gpu
