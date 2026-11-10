// Copyright 2024 The ODML Authors.
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

#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <queue>
#include <random>
#include <utility>
#include <vector>

#ifndef __EMSCRIPTEN__

#include <atomic>

#endif  // __EMSCRIPTEN__

#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "ml_drift/common/task/gpu_tensor.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/executor/gpu_artisan/ml_drift/drafter.h"
#include "runtime/executor/gpu_artisan/ml_drift/llm.h"
#include "runtime/executor/gpu_artisan/ml_drift/llm_gpu_runner.h"
#include "runtime/executor/gpu_artisan/ml_drift/speculative_utils.h"
#include "runtime/executor/gpu_artisan/ml_drift/tensor_loaders/llm_tensor_loader.h"
#include "runtime/executor/gpu_artisan/proto/sampler_params.pb.h"
#include "runtime/executor/gpu_artisan/utils/bitmap.h"
#include "runtime/util/status_macros.h"

namespace odml::infra::gpu {

using ::ml_drift::speculative_utils::GetProbFromDistribution;
using ::ml_drift::speculative_utils::ResampleToken;
using ::ml_drift::speculative_utils::TokenAtTimeStep;
using ::ml_drift::speculative_utils::TokenProbability;

// A version of the LLM runner that runs speculative decoding. This runner takes
// a main model runner and a drafter model runner.
class LlmSpeculativeRunner : public LlmGpuRunner {
 public:
  LlmSpeculativeRunner() = default;

  absl::Status Initialize(
      std::unique_ptr<LlmGpuRunner> main_runner,
      std::unique_ptr<Drafter> drafter, int gamma,
      std::shared_ptr<std::default_random_engine> rand_gen) {
    main_runner_ = std::move(main_runner);
    drafter_ = std::move(drafter);

    if (GetDrafterRunner()) {
      ABSL_CHECK_EQ(main_runner_->GetMaxTopK(),
                    GetDrafterRunner()->GetMaxTopK());
    }
    gamma_ = gamma;
    if (rand_gen != nullptr) {
      rand_gen_ = rand_gen;
    }
    return absl::OkStatus();
  }

#ifdef __EMSCRIPTEN__
  // InitInputTokens gpu asynchronous.
  absl::Status InitInputTokens(const std::vector<int>& input_tokens,
                               int current_step, bool wait_for_completion) {
    RETURN_IF_ERROR(main_runner_->InitInputTokens(input_tokens, current_step,
                                                  wait_for_completion));
    return drafter_->InitInputTokens(input_tokens, current_step,
                                     wait_for_completion, nullptr);
  }
#else
  absl::Status InitInputTokens(const std::vector<int>& input_tokens,
                               int current_step, bool wait_for_completion,
                               const std::atomic_bool* cancel) override {
    ready_tokens_ = {};
    if (enable_speculative_decoding_) {
      RETURN_IF_ERROR(drafter_->InitInputTokens(input_tokens, current_step,
                                                wait_for_completion, cancel));
    }
    return main_runner_->InitInputTokens(input_tokens, current_step,
                                         wait_for_completion, cancel);
  }
#endif

  using LlmGpuRunner::CleanCache;
  absl::Status CleanCache(int time_step,
                          bool skip_ringbuffer_delta_check) override {
    if (enable_speculative_decoding_) {
      RETURN_IF_ERROR(
          drafter_->CleanCache(time_step, skip_ringbuffer_delta_check));
    }
    return main_runner_->CleanCache(time_step, skip_ringbuffer_delta_check);
  }

  absl::Status GetNextTokens(std::vector<std::vector<int>>* output_tokens,
                             UpdateConstraintFn update_constraint_fn) override {
    RET_CHECK_EQ(output_tokens->size(), 1);

    if (!enable_speculative_decoding_) {
      return main_runner_->GetNextTokens(output_tokens, update_constraint_fn);
    }

    // TODO(b/362603136): Note that GetNextTokens() now returns a fixed number
    // of tokens as specified by output_tokens[0].size(), but that may result in
    // janky output. We might want to investigate later if that's an issue and
    // change the contract.
    int filled = 0;
    while (filled < (*output_tokens)[0].size()) {
      if (ready_tokens_.empty()) {
        RETURN_IF_ERROR(RunOneSpeedIteration());
      }
      (*output_tokens)[0][filled] = ready_tokens_.front();
      filled++;
      ready_tokens_.pop();
    }
    return absl::OkStatus();
  }

  absl::StatusOr<ml_drift::ContextData> SaveContext() override {
    ASSIGN_OR_RETURN(ml_drift::ContextData context_data,
                     main_runner_->SaveContext());
    ASSIGN_OR_RETURN(context_data.drafter_context, drafter_->SaveContext());
    context_data.ready_tokens = ready_tokens_;
    return context_data;
  }

  absl::StatusOr<ml_drift::ContextData> SaveLocalContext() override {
    ASSIGN_OR_RETURN(ml_drift::ContextData context_data,
                     main_runner_->SaveLocalContext());
    ASSIGN_OR_RETURN(context_data.drafter_context,
                     drafter_->SaveLocalContext());
    context_data.ready_tokens = ready_tokens_;
    return context_data;
  }

  absl::Status RestoreContext(ml_drift::ContextData&& context_data) override {
    ready_tokens_ = std::move(context_data.ready_tokens);
    RETURN_IF_ERROR(
        drafter_->RestoreContext(std::move(context_data.drafter_context)));
    return main_runner_->RestoreContext(std::move(context_data));
  }

  absl::Status RestoreLocalContext(
      ml_drift::ContextData&& context_data) override {
    RETURN_IF_ERROR(
        drafter_->RestoreLocalContext(std::move(context_data.drafter_context)));
    return main_runner_->RestoreLocalContext(std::move(context_data));
  }

  absl::Status RestoreContextWithCacheCleaning(
      const ml_drift::ContextData& context_data) override {
    if (enable_speculative_decoding_) {
      RETURN_IF_ERROR(drafter_->RestoreContextWithCacheCleaning(
          *context_data.drafter_context));
    }
    return main_runner_->RestoreContextWithCacheCleaning(context_data);
  }

  absl::StatusOr<ml_drift::ContextData> CloneContext(
      const ml_drift::ContextData& context_data) override {
    return main_runner_->CloneContext(context_data);
  }

  absl::StatusOr<ml_drift::ContextData> CloneLocalContext(
      const ml_drift::ContextData& context_data) override {
    return main_runner_->CloneLocalContext(context_data);
  }

  absl::Status LoadLoRA(
      uint32_t lora_id, int lora_rank,
      std::unique_ptr<LlmTensorLoader> tensor_loader) override {
    return main_runner_->LoadLoRA(lora_id, lora_rank, std::move(tensor_loader));
  }

  absl::Status UnloadLoRA(uint32_t lora_id) override {
    return main_runner_->UnloadLoRA(lora_id);
  }

  absl::Status UseLoRA(std::optional<uint32_t> lora_id) override {
    return main_runner_->UseLoRA(lora_id);
  }

  absl::Status UseLayers(const absl::flat_hash_set<int>& layers,
                         int final_ln) override {
    return main_runner_->UseLayers(layers, final_ln);
  }

  absl::Status FillVisionEmbeddings(
      const ml_drift::TensorFloat32& vision_embedding_tensor,
      const std::vector<ml_drift::TensorFloat32>& vision_ple_tensors,
      int image_index) override {
    // TODO(b/432791032): How should the drafter handle vision embeddings?
    return main_runner_->FillVisionEmbeddings(vision_embedding_tensor,
                                              vision_ple_tensors, image_index);
  }

  absl::Status FillAudioEmbeddingsWithTensorBuffer(
      const ::litert::TensorBuffer& audio_tensor_buffer, int index) override {
    // TODO(b/432791032): How should the drafter handle audio embeddings?
    return main_runner_->FillAudioEmbeddingsWithTensorBuffer(
        audio_tensor_buffer, index);
  }

  absl::Status FillAudioEmbeddings(ml_drift::GpuSpatialTensor& audio_tensor,
                                   int index) override {
    // TODO(b/432791032): How should the drafter handle audio embeddings?
    return main_runner_->FillAudioEmbeddings(audio_tensor, index);
  }

  absl::Status ClearVisionPle() override {
    return main_runner_->ClearVisionPle();
  }

  absl::Status UpdateSamplerParams(
      const proto::SamplerParameters& new_params,
      std::shared_ptr<std::default_random_engine> rand_gen) override {
    // TODO(b/432791032): Is it advantageous to update the drafter's sampler
    // params?
    if (GetDrafterRunner() && enable_speculative_decoding_) {
      RETURN_IF_ERROR(
          GetDrafterRunner()->UpdateSamplerParams(new_params, rand_gen));
    }
    return main_runner_->UpdateSamplerParams(new_params, rand_gen);
  }

  absl::Status UpdateConstraintMask(
      const llm_utils::Bitmap& constraint_bitmap) override {
    // TODO(b/432791032): How should the drafter handle constraint masks?
    return main_runner_->UpdateConstraintMask(constraint_bitmap);
  }

  int GetMaxTopK() override { return main_runner_->GetMaxTopK(); }

  absl::Status EnableSpeculativeDecoding(
      bool enable_speculative_decoding) override {
    enable_speculative_decoding_ = enable_speculative_decoding;
    return absl::OkStatus();
  }

  absl::StatusOr<RunnerEnvironment> GetRunnerEnvironment() override {
    return main_runner_->GetRunnerEnvironment();
  }

 private:
  // This attempts to run one cycle of speculative decoding, that is drafting
  // tokens from the drafter models, then score them with the main model, then
  // runs the speculative decoding algorithm to accept or reject and resample.
  // This should produce [1, gamma+1] tokens.
  absl::Status RunOneSpeedIteration() {
    std::uniform_real_distribution<float> distribution =
        std::uniform_real_distribution<float>(0.0f, 1.0f);

    ASSIGN_OR_RETURN(TokenAtTimeStep peeked_token,
                     main_runner_->PeekNextToken());
    ASSIGN_OR_RETURN(std::vector<int> drafted_tokens,
                     drafter_->DraftTokensAfter(peeked_token));
    RET_CHECK_LE(drafted_tokens.size(), gamma_ + 1);

    // Unconditionally accept the first token, this is because the first token
    // is always the token that is kept internally ready in the main model and
    // not sent to the caller yet. We're passing it through the transformer
    // stack only to fill up the corresponding KV cache.
    ready_tokens_.push(peeked_token.token());

    // The drafter may fail to come up with any tokens at all, or insufficient
    // tokens to match the expected sequence size of the main model's scorer
    // (gamma). In this case, we accept the first token and prep the main model
    // for the next round by generating one more token (which is ignored for
    // now).
    if (drafted_tokens.size() != gamma_ + 1) {
      std::vector<std::vector<int>> next_tokens(1);
      return main_runner_->GetNextTokens(&next_tokens,
                                         /*update_constraint_fn=*/{});
    }

    RET_CHECK_EQ(drafted_tokens[0], peeked_token.token())
        << "Mismatch between main model scoring output and peeked output";
    RET_CHECK_EQ(drafted_tokens.size(), gamma_ + 1)
        << "Gamma mismatch between the runner and drafter model";

    RETURN_IF_ERROR(main_runner_->ScoreTokens(drafted_tokens));

    const int base_time_step = peeked_token.time_step();
    for (int i = 1; i < gamma_ + 1; ++i) {
      const int time_step = base_time_step + i;
      ASSIGN_OR_RETURN(std::vector<TokenProbability> score_probs,
                       main_runner_->GetProbabilityCache(time_step));
      ASSIGN_OR_RETURN(std::vector<TokenProbability> draft_probs,
                       drafter_->GetProbabilityCache(time_step));

      int current_drafted_token = drafted_tokens[i];
      float score_prob =
          GetProbFromDistribution(score_probs, current_drafted_token);
      float draft_prob =
          GetProbFromDistribution(draft_probs, current_drafted_token);
      // Note that it is actually possible for drafter probability to be zero
      // but only for the first token in the drafted tokens, because the first
      // token may be the result of OverwriteOneInputToken(). This loop does not
      // operate on that token so it's safe to do RET_CHECK on draft_prob being
      // non-zero.
      RET_CHECK_NE(draft_prob, 0.0f) << "Drafted token probability is zero";
      float reject_prob = fmax(0.0f, 1.0f - score_prob / draft_prob);
      float sampled_val = distribution(*rand_gen_);
      if (sampled_val >= reject_prob) {
        // Accept the token.
        ready_tokens_.push(current_drafted_token);
        continue;
      }
      // Token rejected.
      ASSIGN_OR_RETURN(
          int resampled_token,
          ResampleToken(score_probs, draft_probs, rand_gen_.get()));
      TokenAtTimeStep resampled(time_step, resampled_token);
      RETURN_IF_ERROR(main_runner_->RejectAfterResample(resampled));
      RETURN_IF_ERROR(drafter_->RejectAfterResample(resampled));
      break;
    }
    return absl::OkStatus();
  }

  LlmGpuRunner* GetDrafterRunner() { return drafter_->GetGpuRunner(); }

  std::unique_ptr<LlmGpuRunner> main_runner_;
  std::unique_ptr<Drafter> drafter_;

  bool enable_speculative_decoding_ = false;

  // This is the Gamma specified in the speculative decoding paper.
  // The paper: https://arxiv.org/pdf/2211.17192
  // The actual number of tokens per block is Gamma+1.
  int gamma_;

  // Tokens that has been generated by the speculative decoding loop but has not
  // been fed back to the user through GetNextToken().
  std::queue<int> ready_tokens_;

  // Used for the resampling process and sampling during speculative decoding.
  // Creating member variables here to make sure the generated random numbers
  // are repeatable. This may be provided during construction.
  std::shared_ptr<std::default_random_engine> rand_gen_ =
      std::make_shared<std::default_random_engine>(0);
};

absl::StatusOr<std::unique_ptr<LlmGpuRunner>>
LlmGpuRunner::CreateSpeculativeRunner(
    std::unique_ptr<LlmGpuRunner> main_runner, std::unique_ptr<Drafter> drafter,
    int gamma, std::shared_ptr<std::default_random_engine> rand_gen) {
  auto speculative_runner = std::make_unique<LlmSpeculativeRunner>();
  RETURN_IF_ERROR(speculative_runner->Initialize(
      std::move(main_runner), std::move(drafter), gamma, rand_gen));
  return speculative_runner;
}

}  // namespace odml::infra::gpu
