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

#include <memory>
#include <optional>
#include <random>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/model_hints.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_tensor.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "ml_drift/webgpu/execution_environment.h"  // from @ml_drift
#include "ml_drift/webgpu/spatial_tensor.h"  // from @ml_drift
#include "ml_drift/webgpu/webgpu_api_util.h"  // from @ml_drift
#include "ml_drift/webgpu/webgpu_headers.h"  // from @ml_drift  // IWYU pragma: keep
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/executor/gpu_artisan/ml_drift/executor_factory.h"
#include "runtime/executor/gpu_artisan/ml_drift/llm.h"
#include "runtime/executor/gpu_artisan/ml_drift/llm_gpu_runner.h"
#include "runtime/executor/gpu_artisan/ml_drift/llm_webgpu.h"
#include "runtime/executor/gpu_artisan/ml_drift/speculative_utils.h"
#include "runtime/executor/gpu_artisan/ml_drift/tensor_loaders/llm_tensor_loader.h"
#include "runtime/executor/gpu_artisan/proto/sampler_params.pb.h"
#include "runtime/executor/gpu_artisan/utils/bitmap.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/status_macros.h"

#ifndef __EMSCRIPTEN__
#include <atomic>
#include <cstdint>
#endif  // __EMSCRIPTEN__

namespace odml::infra::gpu {
namespace {

using ::ml_drift::webgpu::ExecutionEnvironment;
using ::ml_drift::webgpu::LlmWebGpu;

// WebGPU implementation of LlmGpuRunner.
class LlmWebGpuRunner : public LlmGpuRunner {
 public:
#ifdef __EMSCRIPTEN__
  explicit LlmWebGpuRunner(const wgpu::Device& device) {}
#elif defined(__APPLE__)
  LlmWebGpuRunner() : env_(wgpu::BackendType::Metal) {}
#elif defined(_WIN32)
  LlmWebGpuRunner() : env_(wgpu::BackendType::D3D12) {}
#else
  LlmWebGpuRunner() : env_(wgpu::BackendType::Vulkan) {}
#endif  // __EMSCRIPTEN__
  ~LlmWebGpuRunner() override = default;

#ifdef __EMSCRIPTEN__
  LlmGpuRunnerType GetLlmGpuRunnerType() override {
    return LlmGpuRunnerType::kWebGpu_Emscripten;
  }

  absl::Status Initialize(const wgpu::Device& device,
                          const wgpu::AdapterInfo& adapter_info,
                          const ml_drift::LlmConfig& llm_config,
                          std::unique_ptr<LlmTensorLoader> tensor_loader,
                          ml_drift::CreateGpuModelInfo create_info,
                          int num_output_heads) {
    if (create_info.precision == ml_drift::CalculationsPrecision::kF16 ||
        create_info.precision == ml_drift::CalculationsPrecision::kF32F16) {
      env_.RequestExtension("shader_f16");
    }
    // TODO: Create path for calling `env_.SetUseAsyncCreateCalls(true)` on
    //   user's request in order to allow for web-only async shader compilation.
    env_.RequestExtension("subgroups");
    if (create_info.precision == ml_drift::CalculationsPrecision::kF16) {
      env_.RequestExtension("subgroups_f16");
    }
    env_.Initialize(device, adapter_info).IgnoreError();
    create_info.storage_type = ml_drift::TensorStorageType::kTexture2D;
    return model_.Init(llm_config, num_output_heads, std::move(tensor_loader),
                       create_info,
                       /*executor_factory=*/nullptr, /*gamma=*/0, &env_);
  }

  absl::Status InitInputTokens(const std::vector<int>& input_tokens,
                               int current_step, bool) override {
    RETURN_IF_ERROR(model_.InitInputTokens(&env_, input_tokens, current_step));
    RETURN_IF_ERROR(ml_drift::webgpu::WaitUntilCompleted(
        env_.queue(), env_.device(), absl::Minutes(5)));
    return absl::OkStatus();
  }
#else

  LlmGpuRunnerType GetLlmGpuRunnerType() override {
    return LlmGpuRunnerType::kWebGpu;
  }

  absl::Status Initialize(
      const ml_drift::LlmConfig& llm_config,
      std::unique_ptr<LlmTensorLoader> tensor_loader,
      ml_drift::CreateGpuModelInfo create_info, int num_output_heads,
      bool wait_for_weight_uploads,
      std::shared_ptr<MlDriftExecutorFactory> executor_factory) {
    RETURN_IF_ERROR(env_.Initialize({
        .use_low_power =
            create_info.hints.Check(ml_drift::ModelHints::kLowPower),
        .enable_host_mapped_pointer = create_info.hints.Check(
            ml_drift::ModelHints::kEnableHostMappedPointer),
    }));

    auto set_executor = [&]() -> absl::Status {
      if (executor_factory) {
        // By default use one thread to compile compute pipelines.
        ASSIGN_OR_RETURN(std::unique_ptr<::ml_drift::Executor> executor,
                         executor_factory->CreateExecutor(1));
        env_.GetComputePipelineCache()->set_executor(std::move(executor));
      }
      return absl::OkStatus();
    };
    RETURN_IF_ERROR(set_executor());

    const auto& gpu_info = env_.GetInfo();
    if (gpu_info.IsApple()) {
      create_info.storage_type = ml_drift::TensorStorageType::kBuffer;
      // Weights will also be faster with buffers, so remove hint.
      create_info.hints.hints &= ~ml_drift::ModelHints::kPreferTextureWeights;
    } else {
      create_info.storage_type = ml_drift::TensorStorageType::kTexture2D;
    }
    int gamma = 0;
    if (llm_config.text_decoder->num_draft_tokens.has_value()) {
      gamma = llm_config.text_decoder->num_draft_tokens.value();
    }
    RETURN_IF_ERROR(model_.Init(llm_config, num_output_heads,
                                std::move(tensor_loader), create_info,
                                executor_factory, gamma, &env_));
    if (wait_for_weight_uploads) {
      model_.WaitForWeightUploads();
      // Reset the executor to make sure shaders have finished compiling.
      RETURN_IF_ERROR(set_executor());
    }
    return absl::OkStatus();
  }

  absl::Status InitInputTokens(const std::vector<int>& input_tokens,
                               int current_step, bool wait_for_completion,
                               const std::atomic_bool* cancel) override {
    RETURN_IF_ERROR(
        model_.InitInputTokens(&env_, input_tokens, current_step,
                               [cancel] { return cancel && *cancel; }));
    if (wait_for_completion) {
      RETURN_IF_ERROR(ml_drift::webgpu::WaitUntilCompleted(
          env_.queue(), env_.device(), absl::Minutes(5)));
    }
    return absl::OkStatus();
  }
#endif  // !__EMSCRIPTEN__

  absl::StatusOr<std::vector<float>> GetNextTokenProbabilities() override {
    return model_.GetNextTokenProbabilities(&env_);
  }

  using LlmGpuRunner::CleanCache;
  absl::Status CleanCache(int time_step,
                          bool skip_ringbuffer_delta_check) override {
    return model_.CleanCache(&env_, time_step, skip_ringbuffer_delta_check);
  }

  absl::Status GetNextTokens(std::vector<std::vector<int>>* output_tokens,
                             UpdateConstraintFn update_constraint_fn) override {
    return model_.GetNextTokens(&env_, output_tokens,
                                std::move(update_constraint_fn));
  }

  absl::StatusOr<ml_drift::speculative_utils::TokenAtTimeStep> PeekNextToken()
      override {
    return model_.PeekNextToken(&env_);
  };

  absl::Status OverwriteOneInputToken(
      const ml_drift::speculative_utils::TokenAtTimeStep& token) override {
    return model_.OverwriteOneInputToken(&env_, token);
  }

  absl::StatusOr<std::vector<ml_drift::speculative_utils::TokenProbability>>
  GetProbabilityCache(int time_step) override {
    return model_.GetProbabilityCache(time_step);
  }

  absl::StatusOr<std::vector<int>> DraftTokens() override {
    return model_.DraftTokens(&env_);
  }

  absl::Status ScoreTokens(const std::vector<int>& tokens) override {
    return model_.ScoreTokens(&env_, tokens);
  }

  absl::Status RejectAfterResample(
      const ml_drift::speculative_utils::TokenAtTimeStep& resampled) override {
    return model_.RejectAfterResample(&env_, resampled);
  }

  absl::StatusOr<ml_drift::ContextData> SaveContext() override {
    return model_.SaveContext(&env_);
  }

  absl::Status RestoreContext(ml_drift::ContextData&& context_data) override {
    return model_.RestoreContext(&env_, std::move(context_data));
  }

  absl::Status RestoreContextWithCacheCleaning(
      const ml_drift::ContextData& context_data) override {
    if (!context_data.tokens.empty() && context_data.key_tensors.empty()) {
      return absl::InvalidArgumentError(
          "ContextData contains processed token but has empty kv-cache.");
    }

    if (!context_data.tokens.empty()) {
      RETURN_IF_ERROR(model_.RestoreContext(&env_, context_data));
    }

    // Clean cache after the current step.
    RETURN_IF_ERROR(CleanCache(context_data.tokens.size()));
    return absl::OkStatus();
  }

  absl::StatusOr<ml_drift::ContextData> CloneContext(
      const ml_drift::ContextData& context_data) override {
    return model_.CloneContext(&env_, context_data);
  }

  absl::StatusOr<ml_drift::ContextData> SaveLocalContext() override {
    return model_.SaveLocalContext(&env_);
  }

  absl::Status RestoreLocalContext(
      ml_drift::ContextData&& context_data) override {
    return model_.RestoreLocalContext(&env_, std::move(context_data));
  }

  absl::StatusOr<ml_drift::ContextData> CloneLocalContext(
      const ml_drift::ContextData& context_data) override {
    return model_.CloneLocalContext(&env_, context_data);
  }

  absl::Status LoadLoRA(
      uint32_t lora_id, int lora_rank,
      std::unique_ptr<LlmTensorLoader> tensor_loader) override {
    return model_.LoadLoRA(&env_, lora_id, lora_rank, std::move(tensor_loader));
  }

  absl::Status UnloadLoRA(uint32_t lora_id) override {
    return model_.UnloadLoRA(&env_, lora_id);
  }

  absl::Status UseLoRA(std::optional<uint32_t> lora_id) override {
    return model_.UseLoRA(&env_, lora_id);
  }

  absl::Status UseLayers(const absl::flat_hash_set<int>& layers,
                         int final_ln) override {
    return model_.UseLayers(&env_, layers, final_ln);
  }

  absl::Status FillVisionEmbeddings(
      const ml_drift::TensorFloat32& vision_embedding_tensor,
      const std::vector<ml_drift::TensorFloat32>& vision_ple_tensors,
      int image_index) override {
    return model_.FillVisionEmbeddings(vision_embedding_tensor,
                                       vision_ple_tensors, image_index, &env_);
  }

  absl::Status FillAudioEmbeddingsWithTensorBuffer(
      const ::litert::TensorBuffer& audio_tensor_buffer, int index) override {
    // Prepare TensorDescriptor
    ml_drift::TensorDescriptor desc(ml_drift::DataType::kFloat16,
                                    ml_drift::TensorStorageType::kTexture2D,
                                    ml_drift::Layout::kBHWDC);

    LITERT_ASSIGN_OR_RETURN(auto tensor_type, audio_tensor_buffer.TensorType());
    if (tensor_type.Layout().Dimensions().size() != 5) {
      return absl::InvalidArgumentError(
          "Audio embeddings must have 5 dimensions [b, h, w, d, c].");
    }
    desc.SetBHWDCShape(ml_drift::BHWDC(tensor_type.Layout().Dimensions()[0],
                                       tensor_type.Layout().Dimensions()[1],
                                       tensor_type.Layout().Dimensions()[2],
                                       tensor_type.Layout().Dimensions()[3],
                                       tensor_type.Layout().Dimensions()[4]));

    // Get a span of the data
    LITERT_ASSIGN_OR_RETURN(
        auto span,
        ::litert::lm::ReferTensorBufferAsSpan<float>(audio_tensor_buffer));

    // Upload data to the descriptor
    desc.UploadData(span.data());

    // Create SpatialTensor from the descriptor
    ml_drift::webgpu::SpatialTensor webgpu_audio_tensor;
    RETURN_IF_ERROR(
        webgpu_audio_tensor.CreateFromDescriptor(env_.device(), desc));

    return FillAudioEmbeddings(webgpu_audio_tensor, index);
  }

  absl::Status FillAudioEmbeddings(ml_drift::GpuSpatialTensor& audio_tensor,
                                   int index) override {
    ml_drift::webgpu::SpatialTensor* webgpu_audio_tensor =
        dynamic_cast<ml_drift::webgpu::SpatialTensor*>(&audio_tensor);
    if (webgpu_audio_tensor == nullptr) {
      return absl::InternalError(
          "Failed to cast audio tensor to WebGPU tensor.");
    }
    return model_.FillAudioEmbeddings(*webgpu_audio_tensor, index, &env_);
  }

  absl::StatusOr<std::vector<std::pair<uint32_t, uint32_t>>> GetVariants()
      override {
    return model_.GetVariants(&env_);
  }

  absl::StatusOr<std::vector<float>> DecodeToLogits(
      const uint32_t model_size, const uint32_t cache_size,
      absl::Span<const float> embeddings,
      absl::Span<const float> ple_embeddings,
      absl::Span<const float> local_attention_mask,
      absl::Span<const float> global_attention_mask,
      absl::Span<const int> seq_position,
      absl::Span<const uint32_t> logits_indices,
      const std::atomic_bool* cancel) override {
    return model_.DecodeToLogits(&env_, model_size, cache_size, embeddings,
                                 ple_embeddings, local_attention_mask,
                                 global_attention_mask, seq_position,
                                 logits_indices, cancel);
  }

  absl::Status ComputeTokenEmbeddings(
      absl::Span<const int> input_tokens, absl::Span<float> output_embeddings,
      absl::Span<float> output_ple_embeddings) override {
    return model_.ComputeTokenEmbeddings(&env_, input_tokens, output_embeddings,
                                         output_ple_embeddings);
  }

  absl::Status ClearVisionPle() override {
    return model_.ClearVisionPle(&env_);
  }

  absl::Status UpdateSamplerParams(
      const proto::SamplerParameters& new_params,
      std::shared_ptr<std::default_random_engine> rand_gen) override {
    return model_.UpdateSamplerParams(new_params, rand_gen);
  }

  absl::Status UpdateConstraintMask(
      const llm_utils::Bitmap& constraint_bitmap) override {
    return model_.UpdateConstraintMask(constraint_bitmap, &env_);
  }

  int GetMaxTopK() override { return model_.GetMaxTopK(); }

  absl::StatusOr<RunnerEnvironment> GetRunnerEnvironment() override {
    return &env_;
  }

 private:
  ExecutionEnvironment env_;
  LlmWebGpu model_;
};

}  // namespace

#ifdef __EMSCRIPTEN__
absl::StatusOr<std::unique_ptr<LlmGpuRunner>> LlmGpuRunner::Create(
    const wgpu::Device& device, const wgpu::AdapterInfo& adapter_info,
    const ml_drift::LlmConfig& llm_config,
    std::unique_ptr<LlmTensorLoader> tensor_loader,
    ml_drift::CreateGpuModelInfo create_info, int num_output_heads) {
  auto llm_runner = std::make_unique<LlmWebGpuRunner>(device);
  RETURN_IF_ERROR(llm_runner->Initialize(device, adapter_info, llm_config,
                                         std::move(tensor_loader), create_info,
                                         num_output_heads));
  return llm_runner;
}
#else
absl::StatusOr<std::unique_ptr<LlmGpuRunner>> LlmGpuRunner::Create(
    const ml_drift::LlmConfig& llm_config,
    std::unique_ptr<LlmTensorLoader> tensor_loader,
    ml_drift::CreateGpuModelInfo create_info, int num_output_heads,
    bool wait_for_weight_uploads,
    std::shared_ptr<MlDriftExecutorFactory> executor_factory) {
  auto llm_runner = std::make_unique<LlmWebGpuRunner>();
  RETURN_IF_ERROR(llm_runner->Initialize(
      llm_config, std::move(tensor_loader), create_info, num_output_heads,
      wait_for_weight_uploads, std::move(executor_factory)));
  return llm_runner;
}
#endif  // __EMSCRIPTEN__

}  // namespace odml::infra::gpu
