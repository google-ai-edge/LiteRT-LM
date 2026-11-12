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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_SPECULATIVE_SHARED_STATE_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_SPECULATIVE_SHARED_STATE_H_

#include <memory>
#include <optional>
#include <vector>

#include "absl/time/time.h"  // from @com_google_absl
#include "ml_drift/common/task/gpu_tensor.h"  // from @ml_drift

namespace odml::infra::gpu {

// Per-round MTP speculative decoding statistics.
struct MtpRoundStats {
  int drafted = 0;
  int accepted = 0;
  int emitted = 0;
  absl::Duration draft_time = absl::ZeroDuration();
  absl::Duration verify_time = absl::ZeroDuration();
  absl::Duration round_time = absl::ZeroDuration();
};

// Unified cross-runner draft chaining state across all backends (WebGPU,
// OpenCL, Metal). Unifies {Gpu,OpenCl,Metal}DraftChainingState with unified
// capacity policy kMaxGamma = 16.
struct MtpDraftChainingState {
  MtpDraftChainingState() = default;
  MtpDraftChainingState(const MtpDraftChainingState&) = delete;
  MtpDraftChainingState& operator=(const MtpDraftChainingState&) = delete;
  virtual ~MtpDraftChainingState() = default;

  static constexpr int kMaxGamma = 16;

  // Set by drafter_runner in DraftTokens().
  // Checked by main_runner in ScoreTokens() (skips host upload of
  // tokens 1..gamma). Checked and reset to false by main_runner in
  // PullScoringTopKProbabilities() after reading back drafted tokens.
  bool has_gpu_chained_drafts = false;

  // Number of drafted tokens produced in the current draft cycle.
  int pending_draft_gamma = 0;

  // Base time step of the current draft cycle (corresponding to
  // peeked_token.time_step()).
  int base_time_step = 0;

  // Drafted tokens retrieved via batched readback in
  // PullScoringTopKProbabilities(). Cached here so
  // drafter_runner->GetProbabilityCache(time_step) can return greedy
  // TokenProbability entries for the drafted tokens.
  std::vector<int> last_drafted_tokens;

  // Cross-runner MTP telemetry accumulators populated during DraftTokens.
  int draft_cb_count = 0;

  // Staging buffer holding gamma drafted token IDs on GPU in VRAM for batched
  // readback during verification scoring. Capacity defaults to kMaxGamma = 16.
  int drafted_tokens_buf_capacity = 0;

  // Non-owning backend Buffer* (cl::Buffer / metal::Buffer) for the main
  // runner's input_token_ids_; owned by the main runner and valid for its
  // lifetime. WebGPU's GpuDraftChainingState uses typed members instead.
  void* scoring_input_buf = nullptr;

  // Non-owning backend Buffer* to the staged drafted-token buffer; owned by the
  // backend subclass or drafter runner and valid while this state is alive.
  void* drafted_tokens_buf = nullptr;
};

// Compatibility alias for merged DraftChainingState.
using DraftChainingState = MtpDraftChainingState;

struct MtpSharedInputs {
  std::vector<ml_drift::GpuSpatialTensor*> key_cache;
  std::vector<ml_drift::GpuSpatialTensor*>
      value_cache;  // indexed by main layer
  ml_drift::GpuSpatialTensor* params_i32 = nullptr;
  ml_drift::GpuSpatialTensor* tokens_ids = nullptr;
  ml_drift::GpuSpatialTensor* last_accepted_hidden_state =
      nullptr;  // [1,1,1,D]
  int kv_shared_start_layer = 0;
  // Local KV ringbuffer cache size from the main model when
  // use_autosized_ringbuffers is enabled.
  std::optional<int> local_ring_cache_size = std::nullopt;
  // Shared draft chaining state across main (verifier) and drafter runner.
  std::shared_ptr<MtpDraftChainingState> draft_chaining = nullptr;
  // Non-owning backend Buffer* (cl/webgpu/metal ::Buffer) to the main runner's
  // input_token_ids_; valid for the main runner's lifetime.
  void* scoring_input_buffer = nullptr;
  // Suppressed token IDs configured on the main model runner.
  std::vector<int> suppressed_tokens;
};

}  // namespace odml::infra::gpu

namespace ml_drift {
// Re-exported into ml_drift for compatibility with legacy LLM graph builders.
using ::odml::infra::gpu::DraftChainingState;
using ::odml::infra::gpu::MtpDraftChainingState;
using ::odml::infra::gpu::MtpRoundStats;
using ::odml::infra::gpu::MtpSharedInputs;
}  // namespace ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_SPECULATIVE_SHARED_STATE_H_
