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
#include <random>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift/cl/environment.h"  // from @ml_drift
#include "ml_drift/cl/opencl_wrapper.h"  // from @ml_drift
#include "ml_drift/cl/tensor.h"  // from @ml_drift
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_tensor.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/executor/gpu_artisan/ml_drift/drafter.h"
#include "runtime/executor/gpu_artisan/ml_drift/executor_factory.h"
#include "runtime/executor/gpu_artisan/ml_drift/llm.h"
#include "runtime/executor/gpu_artisan/ml_drift/llm_gpu_runner.h"
#include "runtime/executor/gpu_artisan/ml_drift/llm_opencl.h"
#include "runtime/executor/gpu_artisan/ml_drift/tensor_loaders/llm_tensor_loader.h"
#include "runtime/executor/gpu_artisan/proto/sampler_params.pb.h"
#include "runtime/executor/gpu_artisan/utils/bitmap.h"
#include "runtime/util/convert_tensor_buffer.h"

namespace odml::infra::gpu {
namespace {

using ml_drift::cl::Environment;
using ml_drift::cl::LlmOpenCl;

// Implementation of `LlmGpuRunner` with `LlmOpenCl`.
class LlmOpenClRunner : public LlmGpuRunner {
 public:
  LlmOpenClRunner() = default;
  ~LlmOpenClRunner() override = default;

  LlmGpuRunnerType GetLlmGpuRunnerType() override {
    return LlmGpuRunnerType::kOpenCl;
  }

  absl::Status InitInputTokens(const std::vector<int>& input_tokens,
                               int current_step, bool wait_for_completion,
                               const std::atomic_bool* cancel) override {
    ABSL_RETURN_IF_ERROR(
        cl_model_->InitInputTokens(env_.get(), input_tokens, current_step));
    if (wait_for_completion) {
      ABSL_RETURN_IF_ERROR(env_->queue()->WaitForCompletion());
    }
    return absl::OkStatus();
  }

  absl::StatusOr<std::vector<float>> GetNextTokenProbabilities() override {
    return cl_model_->GetNextTokenProbabilities(env_.get());
  }

  using LlmGpuRunner::CleanCache;
  absl::Status CleanCache(int time_step,
                          bool skip_ringbuffer_delta_check) override {
    return cl_model_->CleanCache(env_.get(), time_step,
                                 skip_ringbuffer_delta_check);
  }

  absl::Status GetNextTokens(std::vector<std::vector<int>>* output_tokens,
                             UpdateConstraintFn update_constraint_fn) override {
    return cl_model_->GetNextTokens(env_.get(), output_tokens);
  }

  absl::StatusOr<std::vector<float>> DecodeToLogits(
      const uint32_t model_size, const uint32_t cache_size,
      absl::Span<const int> input_tokens,
      absl::Span<const float> local_attention_mask,
      absl::Span<const float> global_attention_mask,
      absl::Span<const int> seq_position,
      absl::Span<const uint32_t> logits_indices,
      const std::atomic_bool* cancel) override {
    return cl_model_->DecodeToLogits(
        env_.get(), model_size, cache_size, input_tokens, local_attention_mask,
        global_attention_mask, seq_position, logits_indices, cancel);
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
    return cl_model_->DecodeToLogits(
        env_.get(), model_size, cache_size, embeddings, ple_embeddings,
        local_attention_mask, global_attention_mask, seq_position,
        logits_indices, cancel);
  }

  absl::Status ComputeTokenEmbeddings(
    absl::Span<const int> input_tokens,
    absl::Span<float> output_embeddings,
    absl::Span<float> output_ple_embeddings) override {
    return cl_model_->ComputeTokenEmbeddings(
      env_.get(), input_tokens, output_embeddings, output_ple_embeddings);
  }

  absl::StatusOr<std::vector<std::pair<uint32_t, uint32_t>>> GetVariants()
      override {
    return cl_model_->GetVariants(env_.get());
  }

  absl::Status CleanCacheAt(absl::Span<const int32_t> indices) override {
    return cl_model_->CleanCacheAt(env_.get(), indices);
  }

  absl::StatusOr<ml_drift::ContextData> SaveContext()
      override {
    return cl_model_->SaveContext(env_.get());
  }

  absl::Status RestoreContext(
      ml_drift::ContextData&& context_data) override {
    return cl_model_->RestoreContext(env_.get(), std::move(context_data));
  }

  absl::Status RestoreContextWithCacheCleaning(
      const ml_drift::ContextData& context_data) override {
    if (!context_data.tokens.empty() && context_data.key_tensors.empty()) {
      return absl::InvalidArgumentError(
          "ContextData contains processed token but has empty kv-cache.");
    }

    if (!context_data.tokens.empty()) {
      ABSL_RETURN_IF_ERROR(cl_model_->RestoreContext(env_.get(), context_data));
    }

    // Clean cache after the current step.
    ABSL_RETURN_IF_ERROR(CleanCache(context_data.tokens.size()));
    return absl::OkStatus();
  }

  absl::StatusOr<ml_drift::ContextData> CloneContext(
      const ml_drift::ContextData& context_data) override {
    return cl_model_->CloneContext(env_.get(), context_data);
  }

  absl::StatusOr<ml_drift::ContextData> SaveLocalContext() override {
    return cl_model_->SaveLocalContext(env_.get());
  }

  absl::Status RestoreLocalContext(
      ml_drift::ContextData&& context_data) override {
    return cl_model_->RestoreLocalContext(env_.get(), std::move(context_data));
  }

  absl::StatusOr<ml_drift::ContextData> CloneLocalContext(
      const ml_drift::ContextData& context_data) override {
    return cl_model_->CloneLocalContext(env_.get(), context_data);
  }

  absl::Status LoadLoRA(
      uint32_t lora_id, int lora_rank,
      std::unique_ptr<LlmTensorLoader> tensor_loader) override {
    return cl_model_->LoadLoRA(env_.get(), lora_id, lora_rank,
                               std::move(tensor_loader));
  }

  absl::Status UnloadLoRA(uint32_t lora_id) override {
    return cl_model_->UnloadLoRA(env_.get(), lora_id);
  }

  absl::Status UseLoRA(std::optional<uint32_t> lora_id) override {
    return cl_model_->UseLoRA(env_.get(), lora_id);
  }

  absl::Status UseLayers(const absl::flat_hash_set<int>& layers,
                         int final_ln) override {
    return cl_model_->UseLayers(env_.get(), layers, final_ln);
  }

  absl::Status FillVisionEmbeddings(
      const ml_drift::TensorFloat32& vision_embedding_tensor,
      const std::vector<ml_drift::TensorFloat32>& vision_ple_tensors,
      int image_index) override {
    return cl_model_->FillVisionEmbeddings(
        vision_embedding_tensor, vision_ple_tensors, image_index, env_.get());
  }

  absl::Status FillAudioEmbeddings(ml_drift::GpuSpatialTensor& audio_tensor,
                                   int index) override {
    ml_drift::cl::Tensor* cl_audio_tensor =
        dynamic_cast<ml_drift::cl::Tensor*>(&audio_tensor);
    if (cl_audio_tensor == nullptr) {
      return absl::InternalError("Failed to cast audio tensor to CL tensor.");
    }
    return cl_model_->FillAudioEmbeddings(*cl_audio_tensor, index, env_.get());
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
    ml_drift::cl::Tensor cl_audio_tensor;
    ABSL_RETURN_IF_ERROR(
        cl_audio_tensor.CreateFromDescriptor(desc, env_.get()->context()));

    return FillAudioEmbeddings(cl_audio_tensor, index);
  }

  absl::Status ClearVisionPle() override {
    return cl_model_->ClearVisionPle(env_.get());
  }

  absl::Status UpdateSamplerParams(
      const proto::SamplerParameters& new_params,
      std::shared_ptr<std::default_random_engine> rand_gen) override {
    return cl_model_->UpdateSamplerParams(new_params, rand_gen);
  }

  absl::Status UpdateConstraintMask(
      const llm_utils::Bitmap& constraint_bitmap) override {
    return cl_model_->UpdateConstraintMask(constraint_bitmap, env_.get());
  }

  int GetMaxTopK() override { return cl_model_->GetMaxTopK(); }

  // For debugging and profiling use
#ifdef ML_DRIFT_PROFILING
  Environment* GetEnv() const override { return env_.get(); }

  absl::StatusOr<ml_drift::cl::InferenceContext*>
  GetInferenceContextForSequenceSize(int sequence_size) const override {
    return cl_model_->GetInferenceContextForSequenceSize(sequence_size);
  }
#endif  // ML_DRIFT_PROFILING

  absl::StatusOr<RunnerEnvironment> GetRunnerEnvironment() override {
    return env_.get();
  }

  std::unique_ptr<Environment> env_;
  std::unique_ptr<LlmOpenCl> cl_model_;
};

}  // namespace

absl::StatusOr<std::unique_ptr<LlmGpuRunner>> LlmGpuRunner::Create(
    const ml_drift::LlmConfig& llm_config,
    std::unique_ptr<LlmTensorLoader> tensor_loader,
    ml_drift::CreateGpuModelInfo create_info, int num_output_heads,
    bool wait_for_weight_uploads,
    std::shared_ptr<MlDriftExecutorFactory> executor_factory) {
  auto load_status = ml_drift::cl::LoadOpenCL();
  if (!load_status.ok()) {
    return absl::FailedPreconditionError(load_status.message());
  }

  auto llm_runner = std::make_unique<LlmOpenClRunner>();
  llm_runner->cl_model_ = std::make_unique<LlmOpenCl>();

  llm_runner->env_ = std::make_unique<Environment>();
  auto* env = llm_runner->env_.get();
  ABSL_RETURN_IF_ERROR(ml_drift::cl::CreateEnvironment(env));

  const auto& gpu_info = env->GetDevicePtr()->GetInfo();
  create_info.storage_type = ml_drift::cl::GetFastestStorageType(gpu_info);

  ABSL_RETURN_IF_ERROR(llm_runner->cl_model_->Init(llm_config, num_output_heads,
                                                   std::move(tensor_loader),
                                                   create_info, env));

  if (wait_for_weight_uploads) {
    ABSL_RETURN_IF_ERROR(env->queue()->WaitForCompletion());
    // Assume compilation is sync there, so we don't need to wait for it in
    // benchmark mode.
  }

  return llm_runner;
}

absl::StatusOr<std::unique_ptr<LlmGpuRunner>>
LlmGpuRunner::CreateSpeculativeRunner(
    std::unique_ptr<LlmGpuRunner> main_runner,
    std::unique_ptr<Drafter> drafter, int gamma,
    std::shared_ptr<std::default_random_engine> rand_gen) {
  return absl::UnimplementedError(
      "LlmOpenClRunner doesn't support speculative decoding.");
}

}  // namespace odml::infra::gpu
