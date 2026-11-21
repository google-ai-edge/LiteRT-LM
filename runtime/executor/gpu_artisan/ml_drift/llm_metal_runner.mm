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

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#import "third_party/ml_drift/metal/common.h"
#include "runtime/executor/gpu_artisan/ml_drift/drafter.h"
#include "runtime/executor/gpu_artisan/ml_drift/executor_factory.h"
#include "runtime/executor/gpu_artisan/ml_drift/llm_gpu_runner.h"
#include "runtime/executor/gpu_artisan/ml_drift/llm_metal.h"
#include "runtime/executor/gpu_artisan/ml_drift/speculative_utils.h"
#include "runtime/executor/gpu_artisan/ml_drift/tensor_loaders/llm_tensor_loader.h"
#include "runtime/executor/gpu_artisan/proto/sampler_params.pb.h"
#include "runtime/executor/gpu_artisan/utils/bitmap.h"
#include "runtime/util/status_macros.h"

namespace odml::infra::gpu {
namespace {

using ml_drift::metal::LlmMetal;
using ml_drift::metal::MetalDevice;

// Implementation of LlmGpuRunner with `ml_drift::metal::MetalDevice`.
class LlmMetalRunner : public LlmGpuRunner {
 public:
  LlmMetalRunner() = default;
  ~LlmMetalRunner() override = default;

  LlmGpuRunnerType GetLlmGpuRunnerType() override { return LlmGpuRunnerType::kMetal; }

  absl::Status InitInputTokens(const std::vector<int>& input_tokens, int current_step,
                               bool wait_for_completion, const std::atomic_bool* cancel) override {
    RETURN_IF_ERROR(model_->InitInputTokens(command_queue_, input_tokens, current_step));
    if (wait_for_completion) {
      ml_drift::metal::WaitUntilCompleted(command_queue_);
    }
    return absl::OkStatus();
  }

  using LlmGpuRunner::CleanCache;
  absl::Status CleanCache(int time_step, bool skip_ringbuffer_delta_check) override {
    return model_->CleanCache(command_queue_, time_step, skip_ringbuffer_delta_check);
  }

  absl::Status UpdateConstraintMask(const llm_utils::Bitmap& constraint_bitmap) override {
    return model_->UpdateConstraintMask(constraint_bitmap);
  }

  absl::Status GetNextTokens(std::vector<std::vector<int>>* output_tokens,
                             UpdateConstraintFn update_constraint_fn) override {
    return model_->GetNextTokens(command_queue_, output_tokens);
  }

  absl::StatusOr<ml_drift::speculative_utils::TokenAtTimeStep> PeekNextToken() override {
    return model_->PeekNextToken(command_queue_);
  }

  absl::StatusOr<ml_drift::ContextData> SaveContext() override {
    return model_->SaveContext(metal_device_, command_queue_);
  }

  absl::Status RestoreContext(ml_drift::ContextData&& context_data) override {
    return model_->RestoreContext(command_queue_, context_data);
  }

  absl::Status RestoreContextWithCacheCleaning(const ml_drift::ContextData& context_data) override {
    if (!context_data.tokens.empty() && context_data.key_tensors.empty()) {
      return absl::InvalidArgumentError(
          "ContextData contains processed token but has empty kv-cache.");
    }

    if (!context_data.tokens.empty()) {
      RETURN_IF_ERROR(model_->RestoreContext(command_queue_, context_data));
    }

    // Clean cache after the current step.
    RETURN_IF_ERROR(CleanCache(context_data.tokens.size()));
    return absl::OkStatus();
  }

  absl::StatusOr<ml_drift::ContextData> CloneContext(
      const ml_drift::ContextData& context_data) override {
    return model_->CloneContext(metal_device_, command_queue_, context_data);
  }

  absl::StatusOr<ml_drift::ContextData> SaveLocalContext() override {
    return model_->SaveLocalContext(metal_device_, command_queue_);
  }

  absl::Status RestoreLocalContext(ml_drift::ContextData&& context_data) override {
    return model_->RestoreLocalContext(command_queue_, context_data);
  }

  absl::StatusOr<ml_drift::ContextData> CloneLocalContext(
      const ml_drift::ContextData& context_data) override {
    return model_->CloneLocalContext(metal_device_, command_queue_, context_data);
  }

  absl::Status LoadLoRA(uint32_t lora_id, int lora_rank,
                        std::unique_ptr<LlmTensorLoader> tensor_loader) override {
    return model_->LoadLoRA(&metal_device_, lora_id, lora_rank, std::move(tensor_loader));
  }

  absl::Status UnloadLoRA(uint32_t lora_id) override {
    return model_->UnloadLoRA(command_queue_, lora_id);
  }

  absl::Status UseLoRA(std::optional<uint32_t> lora_id) override {
    return model_->UseLoRA(command_queue_, lora_id);
  }

  absl::Status UseLayers(const absl::flat_hash_set<int>& layers, int final_ln) override {
    return model_->UseLayers(command_queue_, layers, final_ln);
  }

  absl::Status FillVisionEmbeddings(const ml_drift::TensorFloat32& vision_embedding_tensor,
                                    const std::vector<ml_drift::TensorFloat32>& vision_ple_tensors,
                                    int image_index) override {
    return model_->FillVisionEmbeddings(metal_device_, command_queue_, vision_embedding_tensor,
                                        vision_ple_tensors, image_index);
  }

  absl::Status FillAudioEmbeddingsWithTensorBuffer(
      const ::litert::TensorBuffer& audio_tensor_buffer, int index) override {
    return model_->FillAudioEmbeddingsWithTensorBuffer(metal_device_, command_queue_,
                                                       audio_tensor_buffer, index);
  }

  absl::Status FillAudioEmbeddings(ml_drift::GpuSpatialTensor& audio_tensor, int index) override {
    ml_drift::metal::MetalSpatialTensor* metal_audio_tensor =
        dynamic_cast<ml_drift::metal::MetalSpatialTensor*>(&audio_tensor);
    if (metal_audio_tensor == nullptr) {
      return absl::InternalError("Failed to cast audio tensor to Metal tensor.");
    }
    return model_->FillAudioEmbeddings(*metal_audio_tensor, index, command_queue_);
  }

  absl::Status UpdateSamplerParams(const proto::SamplerParameters& new_params,
                                   std::shared_ptr<std::default_random_engine> rand_gen) override {
    return model_->UpdateSamplerParams(new_params, rand_gen);
  }

  int GetMaxTopK() override { return model_->GetMaxTopK(); }

  absl::StatusOr<RunnerEnvironment> GetRunnerEnvironment() override { return &metal_device_; }

  MetalDevice metal_device_;
  id<MTLCommandQueue> command_queue_;
  std::unique_ptr<LlmMetal> model_;
};

}  // namespace

absl::StatusOr<std::unique_ptr<LlmGpuRunner>> LlmGpuRunner::Create(
    const ml_drift::LlmConfig& llm_config, std::unique_ptr<LlmTensorLoader> tensor_loader,
    ml_drift::CreateGpuModelInfo create_info, int num_output_heads, bool wait_for_weight_uploads,
    std::shared_ptr<MlDriftExecutorFactory> executor_factory) {
  auto llm_runner = std::make_unique<LlmMetalRunner>();
  ml_drift::GpuInfo gpu_info = llm_runner->metal_device_.GetInfo();

  create_info.storage_type = ml_drift::metal::GetFastestStorageType(gpu_info);

  llm_runner->model_ = std::make_unique<LlmMetal>();
  RETURN_IF_ERROR(llm_runner->model_->Init(llm_config, num_output_heads, std::move(tensor_loader),
                                           create_info, &llm_runner->metal_device_));
  ABSL_LOG(INFO) << "Metal LLM tokens initialized.";

  llm_runner->command_queue_ = [llm_runner->metal_device_.device() newCommandQueue];
  RETURN_IF_ERROR(llm_runner->CleanCache(0));

  if (wait_for_weight_uploads) {
    ml_drift::metal::WaitUntilCompleted(llm_runner->command_queue_);
    // Assume compilation is sync there, so we don't need to wait for it in benchmark mode.
  }

  return llm_runner;
}

absl::StatusOr<std::unique_ptr<LlmGpuRunner>> LlmGpuRunner::CreateSpeculativeRunner(
    std::unique_ptr<LlmGpuRunner> main_runner, std::unique_ptr<Drafter> drafter, int gamma,
    std::shared_ptr<std::default_random_engine> rand_gen) {
  return absl::UnimplementedError("LlmMetalRunner doesn't support speculative decoding.");
}

}  // namespace odml::infra::gpu
