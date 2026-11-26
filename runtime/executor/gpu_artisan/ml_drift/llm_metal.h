// Copyright 2024 The MediaPipe Authors.
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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_LLM_METAL_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_LLM_METAL_H_

#import <Metal/Metal.h>

#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_tensor.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "ml_drift/metal/buffer.h"  // from @ml_drift
#include "ml_drift/metal/compute_task.h"  // from @ml_drift
#include "ml_drift/metal/environment.h"  // from @ml_drift
#include "ml_drift/metal/inference_context.h"  // from @ml_drift
#include "ml_drift/metal/metal_device.h"  // from @ml_drift
#include "ml_drift/metal/metal_spatial_tensor.h"  // from @ml_drift
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/executor/gpu_artisan/ml_drift/audio_encoder.h"
#include "runtime/executor/gpu_artisan/ml_drift/group_size.h"
#include "runtime/executor/gpu_artisan/ml_drift/llm.h"
#include "runtime/executor/gpu_artisan/ml_drift/runtime_params.h"
#include "runtime/executor/gpu_artisan/ml_drift/speculative_utils.h"
#include "runtime/executor/gpu_artisan/ml_drift/tensor_loaders/embedding_updater.h"
#include "runtime/executor/gpu_artisan/ml_drift/tensor_loaders/llm_tensor_loader.h"
#include "runtime/executor/gpu_artisan/ml_drift/transformer.h"
#include "runtime/executor/gpu_artisan/proto/sampler_params.pb.h"
#include "runtime/executor/gpu_artisan/utils/bitmap.h"

namespace ml_drift {
namespace metal {

// API for initializing and running inference on LLMs with MLDrift and Metal.
class LlmMetal {
 public:
  absl::Status Init(const LlmConfig& config, int outputs_count,
                    const std::string& weights_path,
                    const CreateGpuModelInfo& create_info, MetalDevice* device);
  absl::Status Init(
      const LlmConfig& config, int outputs_count,
      std::unique_ptr<::odml::infra::gpu::LlmTensorLoader> tensor_loader,
      const CreateGpuModelInfo& create_info, MetalDevice* device);

  absl::Status InitAudioEncoder(
      const AudioEncoderConfig& config,
      std::unique_ptr<::odml::infra::gpu::LlmTensorLoader> tensor_loader,
      const CreateGpuModelInfo& create_info, int output_batch_size,
      id<MTLCommandQueue> command_queue, Environment* env);

  // Zeroes key/value cached tensors
  absl::Status CleanCache(id<MTLCommandQueue> command_queue, int time_step,
                          bool skip_ringbuffer_delta_check = false);

  // Zeroes audio cache
  absl::Status CleanAudioCache(id<MTLCommandQueue> command_queue,
                               int time_step);

  // Takes a snapshot of the kv cache for restoring inference later.
  absl::StatusOr<ContextData> SaveContext(MetalDevice& device,
                                          id<MTLCommandQueue> command_queue,
                                          bool local_only = false);
  // Overwrites the kv cache with the data from the function argument
  // context_data.
  absl::Status RestoreContext(id<MTLCommandQueue> command_queue,
                              const ContextData& context_data,
                              bool local_only = false);
  // Initializes a duplicate kv cache as in the function argument context_data,
  // and make a deep GPU -> GPU copy of the context data.
  absl::StatusOr<ContextData> CloneContext(MetalDevice& device,
                                           id<MTLCommandQueue> command_queue,
                                           const ContextData& context_data,
                                           bool local_only = false);

  // Takes a snapshot of the local kv cache for restoring inference later.
  absl::StatusOr<ContextData> SaveLocalContext(
      MetalDevice& device, id<MTLCommandQueue> command_queue) {
    return SaveContext(device, command_queue, /*local_only=*/true);
  }
  // Overwrites the local kv cache with the data from the function argument
  // context_data.
  absl::Status RestoreLocalContext(id<MTLCommandQueue> command_queue,
                                   const ContextData& context_data) {
    return RestoreContext(command_queue, context_data, /*local_only=*/true);
  }
  // Initializes a duplicate local kv cache as in the function argument
  // context_data, and make a deep GPU -> GPU copy of the local context data.
  absl::StatusOr<ContextData> CloneLocalContext(
      MetalDevice& device, id<MTLCommandQueue> command_queue,
      const ContextData& context_data) {
    return CloneContext(device, command_queue, context_data,
                        /*local_only=*/true);
  }

  // Save the audio context data for the current inference.
  absl::StatusOr<AudioContextData> SaveAudioContext(
      MetalDevice& device, id<MTLCommandQueue> command_queue);
  // Restore the audio context data for the current inference.
  absl::Status RestoreAudioContext(id<MTLCommandQueue> command_queue,
                                   const AudioContextData& context_data);
  // Clone audio context data
  absl::StatusOr<AudioContextData> CloneAudioContext(
      MetalDevice& device, id<MTLCommandQueue> command_queue,
      const AudioContextData& context_data);

  // InitInputTokens and GetNextTokens must be used only together.
  // InitInputTokens gpu asynchronous, for measurements gpu sync must be used.
  absl::Status InitInputTokens(id<MTLCommandQueue> command_queue,
                               const std::vector<int>& input_tokens,
                               int time_step);
  absl::Status GetNextTokens(id<MTLCommandQueue> command_queue,
                             std::vector<int>* output_tokens);
  // for multiple output
  absl::Status GetNextTokens(id<MTLCommandQueue> command_queue,
                             std::vector<std::vector<int>>* output_tokens);
  absl::StatusOr<id<MTLCommandBuffer>> ProcessNextOutputToken(
      id<MTLCommandQueue> command_queue);
  absl::StatusOr<GpuSpatialTensor*> ExecuteAudioEncoder(
      const TensorFloat32& audio_tensor, const TensorFloat32& mask_tensor,
      int time_step, id<MTLCommandQueue> command_queue);
  // Inserts the gpu audio_tensor into media_embeddings_ tensor to be used by
  // text decoder. Index should be set to 0 unless the encoder gets ahead of
  // the decoder. If you wish to do this, you will need to add functionality to
  // manually set media_cache_size_ to be larger than the audio encoder output.
  absl::Status FillAudioEmbeddings(MetalSpatialTensor& audio_tensor, int index,
                                   id<MTLCommandQueue> command_queue);
  // Similar to FillAudioEmbeddings, but takes a LiteRT TensorBuffer as input.
  // This function will create a metal spatial tensor from the LiteRT
  // TensorBuffer and then fill the audio embeddings.
  absl::Status FillAudioEmbeddingsWithTensorBuffer(
      MetalDevice& device, id<MTLCommandQueue> command_queue,
      const ::litert::TensorBuffer& audio_tensor_buffer, int index);
  // Inserts vision_embedding tensor into media_embeddings_ and
  // vision_ple_tensors into vision_ple_tensors_ to be used by text decoder.
  // Both at index = image_index * text_config_.vision_tokens_nums
  absl::Status FillVisionEmbeddings(
      MetalDevice& device, id<MTLCommandQueue> command_queue,
      const TensorFloat32& vision_embedding_tensor,
      const std::vector<TensorFloat32>& vision_ple_tensors, int image_index);

  absl::Status LoadLoRA(
      MetalDevice* device, uint32_t lora_id, int lora_rank,
      std::unique_ptr<::odml::infra::gpu::LlmTensorLoader> tensor_loader);
  absl::Status UnloadLoRA(id<MTLCommandQueue> command_queue, uint32_t lora_id);
  absl::Status UseLoRA(id<MTLCommandQueue> command_queue,
                       std::optional<uint32_t> lora_id);
  absl::Status UseLayers(id<MTLCommandQueue> command_queue,
                         const absl::flat_hash_set<int>& layers, int final_ln);

  // For profiling.
  // If sequence_size >= 1, appropriate inference context is returned.
  // If sequence_size is -1, the post-processing context is returned.
  absl::StatusOr<InferenceContext*> GetInferenceContextForSequenceSize(
      int sequence_size) {
    if (sequence_size == -1) {
      return post_processing_.get();
    }
    auto iter = models_.find(sequence_size);
    if (iter == models_.end()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Sequence size ", sequence_size, " not found."));
    }
    return iter->second.context.get();
  }

  absl::Status UpdateSamplerParams(
      const odml::infra::proto::SamplerParameters& new_params,
      std::shared_ptr<std::default_random_engine> rand_gen);

  uint64_t GetCacheSize() const {
    uint64_t total_size = 0;
    for (const auto& tensor : cache_.key_cached_tensors) {
      total_size += tensor->GetMemorySizeInBytes();
    }
    for (const auto& tensor : cache_.value_cached_tensors) {
      total_size += tensor->GetMemorySizeInBytes();
    }
    return total_size;
  }

  uint64_t GetSharedWeightsSize() const {
    uint64_t total_size = 0;
    for (const auto& tensor : model_tensors_) {
      total_size += tensor->GetMemorySizeInBytes();
    }
    return total_size;
  }

  uint64_t GetModelsDataSize() const {
    uint64_t total_size = 0;
    for (const auto& model : models_) {
      total_size += model.second.context->GetConstantTensorsSize();
      total_size += model.second.context->GetIntermediateTensorsSize();
    }
    total_size += post_processing_->GetConstantTensorsSize();
    total_size += post_processing_->GetIntermediateTensorsSize();
    return total_size;
  }

  int GetMaxTopK() { return max_top_k_; }

  // The LlmMetal class actually holds the next token to be predicted in
  // memory, ie. after every sampled token that is return, returned token is
  // always passed through the transformer stack again so there's always an
  // available token that is hidden in the LlmMetal class and not returned.
  // This function allows the caller to peek at that token. Returned is the time
  // step and token ID being sampled by the last forward path.
  absl::StatusOr<ml_drift::speculative_utils::TokenAtTimeStep> PeekNextToken(
      id<MTLCommandQueue> command_queue);

  absl::Status UpdateConstraintMask(
      const odml::infra::llm_utils::Bitmap& constraint_bitmap);

 private:
  // For unit tests
  friend class LlmMetalSpouse;

  // Struct that holds the cache for the key and value tensors.
  struct TransformerCache {
    std::vector<std::unique_ptr<MetalSpatialTensor>> key_cached_tensors;
    std::vector<std::unique_ptr<MetalSpatialTensor>> value_cached_tensors;
    std::unique_ptr<InferenceContext> clean_cache;
  };

  // Prefill param struct
  struct TransformerParams {
    std::unique_ptr<MetalSpatialTensor> params_i32;
    std::unique_ptr<ComputeTask> write_i32_params;
  };

  // Function for initializing key/value cached tensors.
  absl::StatusOr<std::vector<std::unique_ptr<MetalSpatialTensor>>>
  InitKvTensors(MetalDevice& device, const TensorDescriptor& tensor_desc,
                int num_cache_layers);

  // Must be called before models initialization.
  absl::Status InitSharedTensors(TransformerBuilder* transformer_builder,
                                 TransformerCache& cache,
                                 TransformerParams& params,
                                 int num_cache_layers,
                                 MetalDevice* device);

  absl::Status InitModel(TransformerBuilder* transformer_builder,
                         int sequence_size, int batch_size,
                         MetalDevice* device);
  absl::Status InitPostProcessingModel(TransformerBuilder* transformer_builder,
                                       MetalDevice* device);
  absl::Status InitCleanCacheModel(TransformerBuilder* transformer_builder,
                                   TransformerCache& cache,
                                   TransformerParams& params,
                                   MetalDevice* device);
  absl::Status InitAudioEncoderModel(AudioEncoderBuilder* audio_encoder_builder,
                                     id<MTLCommandQueue> command_queue,
                                     Environment* env);
  absl::Status InitHelperOps(MetalDevice* device);

  absl::Status InitInputTokensWithModelOne(
      id<MTLCommandQueue> command_queue, const std::vector<int>& input_tokens);
  absl::Status InitInputTokensWithModelN(id<MTLCommandQueue> command_queue,
                                         const std::vector<int>& input_tokens);

  bool IsLayerGlobal(int layer_number) const;
  absl::Status ExecuteCopyCache(id<MTLComputeCommandEncoder> encoder,
                                int src_size, MetalSpatialTensor* src_k,
                                MetalSpatialTensor* src_v, int dst_size,
                                MetalSpatialTensor* dst_k,
                                MetalSpatialTensor* dst_v,
                                bool is_global = false);

  absl::Status ExecuteUpdateIntParams(id<MTLComputeCommandEncoder> encoder,
                                      TransformerConfig& config,
                                      TransformerParams& params,
                                      const LlmRuntimeParams& param_vals);
  absl::Status ExecuteUpdateParams(id<MTLComputeCommandEncoder> encoder,
                                   MetalSpatialTensor* tensor,
                                   const std::vector<float>& params);

  absl::Status ExecutePostProcessing(id<MTLComputeCommandEncoder> encoder);

  absl::StatusOr<GpuSpatialTensor*> CreateModelTensor(
      MetalDevice* device,
      std::vector<std::unique_ptr<MetalSpatialTensor>>* model_tensors,
      const TensorDescriptor& tensor_desc);

  absl::StatusOr<GpuSpatialTensor*> CreateModelTensorMmaped(
      MetalDevice* device, odml::infra::gpu::LlmTensorLoader* tensor_loader,
      std::vector<std::unique_ptr<MetalSpatialTensor>>* model_tensors,
      const TensorDescriptor& tensor_desc);

  void SetupOptionalNodes();

  struct StackedTransformerInference {
    std::unique_ptr<InferenceContext> context;
    int seq_size;
    GpuModelBuilder::TensorHandle input, output;
    std::unique_ptr<ComputeTask> copy_nth_op;
    std::unique_ptr<ComputeTask> copy_tokens_op;
    std::vector<int> default_tags;
    std::vector<odml::infra::gpu::EmbeddingWeights> ple_weights;
    std::vector<odml::infra::gpu::EmbeddingWeights> ngrammer_weights;
    // Holds the LoRA tensors created while building the LLM. The first key is
    // the LoRA rank and the second key is the name of the tensor.
    absl::flat_hash_map<int, absl::flat_hash_map<std::string, LoRATensorInfo>>
        lora_tensors;
    // For debugging, tensors to inspect will be referenced here.
    absl::flat_hash_map<std::string, GpuModelBuilder::TensorHandle>
        debug_tensors;
  };

  absl::Status UpdateEmbeddings(id<MTLCommandQueue> command_queue,
                                StackedTransformerInference& model,
                                int previous_token,
                                absl::Span<const int> tokens,
                                bool require_load_with_queue = true);

  // src has shape: [1, 1, x, emb_dim] where x is the num of media hard tokens.
  // dst has shape: [1, 1, cache_size, emb_dim]
  // we perform: dst[0, 0, index:index + x, :] = src
  // Can be used for media embeddings and vision ple embeddings.
  absl::Status UpdateMultimodalEmbeddings(id<MTLCommandQueue> command_queue,
                                          MetalSpatialTensor* src,
                                          MetalSpatialTensor* dst, int index);

  // Inserts the tensors passed into vision_ple_tensors_ to be used by text
  // decoder. Expect vision_ple_tensors to be of size (text_config_.stack_size -
  // text_config_.extra_stack_size). Index should be set to 0 unless the encoder
  // gets ahead of the decoder.
  absl::Status FillVisionPleEmbeddings(
      MetalDevice& device, id<MTLCommandQueue> command_queue,
      const std::vector<TensorFloat32>& vision_ple_tensors, int index);

  // Inserts the cpu media_tensor into media_embeddings_ tensor to be used by
  // text decoder. Index should be set to 0 unless the encoder gets ahead of
  // the decoder.
  absl::Status FillMediaEmbeddings(MetalDevice& device,
                                   id<MTLCommandQueue> command_queue,
                                   const TensorFloat32& media_tensor,
                                   int index);

  CreateGpuModelInfo create_info_;
  // Version of the InferenceContext for processing sequence length 1.
  StackedTransformerInference* model_one_ = nullptr;
  // A mapping from sequence size to model. This will always have at least a
  // single model with sequence size 1 that is referred to by |model_one_|.
  std::map<int, StackedTransformerInference> models_;
  std::vector<::odml::infra::gpu::GroupSize> group_sizes_;

  // Decode params
  std::unique_ptr<MetalSpatialTensor> params_f32_;

  struct AudioParams {
    std::unique_ptr<InferenceContext> context;
    std::unique_ptr<MetalSpatialTensor> feature;
    std::unique_ptr<MetalSpatialTensor> mask;
    GpuModelBuilder::TensorHandle output_handle;  // audio enc output
    std::unique_ptr<MetalSpatialTensor> signal;   // v3 tensor for RelPE

    TransformerCache cache;
    TransformerParams transformer_params;
    AudioEncoderConfig config;

    // Audio group norm cumsum for streaming from audio encoder -> audio encoder
    std::unique_ptr<MetalSpatialTensor>
        stream_cumsum;  // audio stream for encoder n
    // Audio light conv1d requires data from previous frame for streaming.
    // Pass it from frame to frame with this tensor.
    // Note that we have one for each layer in the transformer.
    std::vector<std::unique_ptr<MetalSpatialTensor>> stream_depthwise_tensors;
  };
  AudioParams audio_params_;

  // Vision fields
  std::vector<std::unique_ptr<MetalSpatialTensor>> vision_ple_tensors_;
  std::unique_ptr<MetalSpatialTensor> media_embeddings_;
  int media_cache_size_;
  odml::infra::gpu::TokensTransformFn tokens_transform_fn_ = [](int token) {
    return token;
  };

  // This vector holds tensors that are used across model_one_, model_n_, and
  // post_processing_. Caching the tensors here allows sharing these tensors to
  // prevent duplicating them in GPU memory. This vector is populated by
  // LlmFileTensorLoader as the model is initialized.
  std::unique_ptr<::odml::infra::gpu::LlmTensorLoader> tensor_loader_;
  std::vector<std::unique_ptr<MetalSpatialTensor>> model_tensors_;

  std::unique_ptr<InferenceContext> post_processing_;
  std::unique_ptr<ComputeTask> write_f32_params_;
  std::unique_ptr<MetalSpatialTensor> tokens_ids_;
  std::unique_ptr<Buffer> input_token_ids_;

  // Op that copies tokens to the input tensor.
  std::unique_ptr<ComputeTask> copy_cache_;
  std::unique_ptr<ComputeTask> copy_cache_global_;

  // Op that copies multimodal embeddings to the input tensor.
  std::unique_ptr<ComputeTask> multimodal_embeddings_op_;

  // Cache for the key and value tensors.
  TransformerCache cache_;

  // Prefill params
  TransformerParams params_;

  int ready_tokens_ = 0;
  int num_output_tokens_ = 0;
  bool need_sync_for_peek_token_ = false;
  std::vector<int> input_tokens_;
  int output_batch_size_ = 0;
  id<MTLDevice> device_ = nullptr;

  // Used for floating-point comparisons with 0.0f. Its value depends on the
  // calculation precision.
  float epsilon_ = std::numeric_limits<float>::epsilon();

  // Uniform random sampling between [0.0, sampler_params_.p()). Creating member
  // variables here to make sure the generated random numbers are repeatable.
  // The constructor for `std::default_random_engine` will use a random seed if
  // one is provided and 0 otherwise.
  std::shared_ptr<std::default_random_engine> rand_gen_ =
      std::make_shared<std::default_random_engine>(0);

  // Maximum allowed value for `sampler_params_.k()`. Set at initialization.
  int max_top_k_ = 1;
  odml::infra::proto::SamplerParameters sampler_params_;

  // Fields used for LoRA support.
  // Maps the LoRA ID to tensor loader to get weights for that adaptation.
  struct LoRALoader {
    std::unique_ptr<::odml::infra::gpu::LlmTensorLoader> loader;
    int lora_rank;
    std::vector<std::unique_ptr<MetalSpatialTensor>> tensors;
  };
  absl::flat_hash_map<uint32_t, std::unique_ptr<LoRALoader>> lora_loaders_;

  // The current LoRA ID that is loaded for this model.
  std::optional<uint32_t> current_lora_id_;

  // The optional model tags for InferenceContext.
  absl::flat_hash_set<int> layer_tags_;

  // The optional lora tag for InferenceContext.
  absl::flat_hash_set<int> lora_tags_;

  // The optional tag for layers with shared KV cache.
  std::optional<int> shared_kv_cache_tag_;

  std::vector<odml::infra::gpu::EmbeddingRaw> ple_embeddings_;
  std::vector<odml::infra::gpu::EmbeddingRaw> ngrammer_embeddings_;

  TransformerConfig text_config_;
  std::unique_ptr<MetalSpatialTensor> constraint_mask_;
  std::vector<bool> last_constraint_mask_;
  bool constraint_mask_changed_since_last_input_ = true;
  std::vector<int> suppressed_tokens_;

  std::vector<std::unique_ptr<Buffer>> mmaped_buffers_;
};

}  // namespace metal
}  // namespace ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_LLM_METAL_H_
