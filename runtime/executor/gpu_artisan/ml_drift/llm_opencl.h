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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_LLM_OPENCL_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_LLM_OPENCL_H_

#include <atomic>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift/cl/buffer.h"  // from @ml_drift
#include "ml_drift/cl/cl_command_queue.h"  // from @ml_drift
#include "ml_drift/cl/cl_context.h"  // from @ml_drift
#include "ml_drift/cl/cl_event.h"  // from @ml_drift
#include "ml_drift/cl/cl_operation.h"  // from @ml_drift
#include "ml_drift/cl/environment.h"  // from @ml_drift
#include "ml_drift/cl/inference_context.h"  // from @ml_drift
#include "ml_drift/cl/tensor.h"  // from @ml_drift
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_tensor.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "runtime/executor/gpu_artisan/ml_drift/audio_encoder.h"
#include "runtime/executor/gpu_artisan/ml_drift/group_size.h"
#include "runtime/executor/gpu_artisan/ml_drift/llm.h"
#include "runtime/executor/gpu_artisan/ml_drift/runtime_params.h"
#include "runtime/executor/gpu_artisan/ml_drift/tensor_loaders/embedding_updater.h"
#include "runtime/executor/gpu_artisan/ml_drift/tensor_loaders/llm_tensor_loader.h"
#include "runtime/executor/gpu_artisan/ml_drift/transformer.h"
#include "runtime/executor/gpu_artisan/proto/sampler_params.pb.h"
#include "runtime/executor/gpu_artisan/utils/bitmap.h"
#include "runtime/executor/gpu_artisan/utils/model_data.h"

namespace ml_drift {
namespace cl {

// API for initializing and running inference on LLMs with MLDrift and OpenCL.
class LlmOpenCl {
 public:
  absl::Status Init(const LlmConfig& config, int outputs_count,
                    const std::string& weights_path,
                    const CreateGpuModelInfo& create_info, Environment* env);
  absl::Status Init(
      const LlmConfig& config, int outputs_count,
      std::unique_ptr<::odml::infra::gpu::LlmTensorLoader> tensor_loader,
      const CreateGpuModelInfo& create_info, Environment* env);

  absl::Status InitAudioEncoder(
      const AudioEncoderConfig& config,
      std::unique_ptr<::odml::infra::gpu::LlmTensorLoader> tensor_loader,
      const CreateGpuModelInfo& create_info, int output_batch_size,
      Environment* env);

  // Zeroes key/value cached tensors
  absl::Status CleanCache(Environment* env, int time_step,
                          bool skip_ringbuffer_delta_check = false);
  absl::Status CleanCacheAt(Environment* env,
                            absl::Span<const int32_t> to_remove);

  absl::Status CleanAudioCache(Environment* env, int time_step);

  // Save the context data for the current inference.
  absl::StatusOr<ContextData> SaveContext(Environment* env,
                                          bool local_only = false);
  // Restore the context data for the current inference.
  absl::Status RestoreContext(Environment* env, const ContextData& context_data,
                              bool local_only = false);
  // Clone context data
  absl::StatusOr<ContextData> CloneContext(Environment* env,
                                           const ContextData& context_data,
                                           bool local_only = false);

  // Save the local context data for the current inference.
  absl::StatusOr<ContextData> SaveLocalContext(Environment* env) {
    return SaveContext(env, /*local_only=*/true);
  }
  // Restore the local context data for the current inference.
  absl::Status RestoreLocalContext(Environment* env,
                                   const ContextData& context_data) {
    return RestoreContext(env, context_data, /*local_only=*/true);
  }
  // Clone local context data
  absl::StatusOr<ContextData> CloneLocalContext(
      Environment* env, const ContextData& context_data) {
    return CloneContext(env, context_data, /*local_only=*/true);
  }

  // Save the audio context data for the current inference.
  absl::StatusOr<AudioContextData> SaveAudioContext(Environment* env);
  // Restore the audio context data for the current inference.
  absl::Status RestoreAudioContext(Environment* env,
                                   const AudioContextData& context_data);
  // Clone audio context data
  absl::StatusOr<AudioContextData> CloneAudioContext(
      Environment* env, const AudioContextData& context_data);

  // InitInputTokens and GetNextTokens must be used only together.
  // InitInputTokens gpu asynchronous, for measurements clFinish must be used.
  absl::Status InitInputTokens(Environment* env,
                               const std::vector<int>& input_tokens,
                               int time_step);

  absl::StatusOr<std::vector<float>> GetNextTokenProbabilities(
      Environment* env);

  // DecodeToLogits is used to generate logits for the given set of input
  // tokens.
  // model_size: The model size is the number of tokens that can be processed
  // in a single forward pass.
  // cache_size: The cache size is the number of tokens that can be stored in
  // the KV cache.
  // input_tokens: 1D Array of input tokens to generate logits for.
  // local_attention_mask: The local attention mask for the input tokens, in
  //   shape of [input_tokens.size(), max_sequence_length]
  // global_attention_mask: The global attention mask for the input tokens, in
  //   shape of [input_tokens.size(), max_sequence_length]
  // seq_position: 1D Array with the sequence position for the input tokens,
  //   same size as the input_tokens.
  // logits_indices: Array of indices of the tokens for which it will generate
  //   output logits.
  // cancel: A flag to cancel the operation.
  absl::StatusOr<std::vector<float>> DecodeToLogits(
      Environment* env, uint32_t model_size, uint32_t cache_size,
      absl::Span<const int> input_tokens,
      absl::Span<const float> local_attention_mask,
      absl::Span<const float> global_attention_mask,
      absl::Span<const int> seq_position,
      absl::Span<const uint32_t> logits_indices,
      const std::atomic_bool* cancel);
  absl::StatusOr<std::vector<std::pair<uint32_t, uint32_t>>> GetVariants(
      Environment* env);

  // Similar to the above DecodeToLogits, but takes embeddings and
  // ple_embeddings as inputs.
  absl::StatusOr<std::vector<float>> DecodeToLogits(
      Environment* env, uint32_t model_size, uint32_t cache_size,
      absl::Span<const float> embeddings,
      absl::Span<const float> ple_embeddings,
      absl::Span<const float> local_attention_mask,
      absl::Span<const float> global_attention_mask,
      absl::Span<const int> seq_position,
      absl::Span<const uint32_t> logits_indices,
      const std::atomic_bool* cancel);

  absl::Status ComputeTokenEmbeddings(
      Environment* env,
      absl::Span<const int> input_tokens,
      absl::Span<float> output_embeddings,
      absl::Span<float> output_ple_embeddings);

  absl::Status GetNextTokens(Environment* env, std::vector<int>* output_tokens);
  // for multiple output
  absl::Status GetNextTokens(Environment* env,
                             std::vector<std::vector<int>>* output_tokens);

  absl::Status UpdateConstraintMask(
      const odml::infra::llm_utils::Bitmap& constraint_bitmap,
      Environment* env);

  absl::StatusOr<GpuSpatialTensor*> ExecuteAudioEncoder(
      const TensorFloat32& audio_tensor, const TensorFloat32& mask_tensor,
      int time_step, Environment* env);

  // Inserts vision_embedding tensor into media_embeddings_ and
  // vision_ple_tensors into vision_ple_tensors_ to be used by text decoder.
  // Both at index = image_index * text_config_.vision_tokens_num.
  absl::Status FillVisionEmbeddings(
      const TensorFloat32& vision_embedding_tensor,
      const std::vector<TensorFloat32>& vision_ple_tensors, int image_index,
      Environment* env);

  // Inserts the gpu audio_tensor into media_embeddings_ tensor to be used by
  // text decoder. Index should be set to 0 unless the encoder gets ahead of
  // the decoder. If you wish to do this, you will need to add functionality to
  // manually set media_cache_size_ to be larger than the audio encoder output.
  absl::Status FillAudioEmbeddings(Tensor& audio_tensor, int index,
                                   Environment* env);

  // Clears the vision_ple_tensors_ to be used by text decoder. This should
  // be called if we want to run audio after filling the vision
  // ple embeddings. Ie, we should have a sequence such as:
  // 1. FillVisionEmbeddings
  // 2. InitInputTokens
  // 3. ClearVisionPle
  // 4. FillAudioEmbeddings
  absl::Status ClearVisionPle(Environment* env);

  absl::Status LoadLoRA(
      Environment* env, uint32_t lora_id, int lora_rank,
      std::unique_ptr<::odml::infra::gpu::LlmTensorLoader> tensor_loader);
  absl::Status UnloadLoRA(Environment* env, uint32_t lora_id);
  absl::Status UseLoRA(Environment* env, std::optional<uint32_t> lora_id);
  absl::Status UseLayers(Environment* env,
                         const absl::flat_hash_set<int>& layers, int final_ln);

  absl::Status UpdateSamplerParams(
      const odml::infra::proto::SamplerParameters& new_params,
      std::shared_ptr<std::default_random_engine> rand_gen);

  // For debugging and profiling only. This is used to get the inference context
  // for the prefill phase with sequence size equal to
  // mldrift_options.sequence_batch_size().
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

  uint64_t GetCacheSizeInBytes() const {
    uint64_t total_size = 0;
    for (const auto& tensor : text_cache_.key_cached_tensors) {
      total_size += tensor->GetMemorySizeInBytes();
    }
    for (const auto& tensor : text_cache_.value_cached_tensors) {
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
      total_size += model.second.context
                        ->GetSizeOfMemoryAllocatedForIntermediateTensors();
    }
    total_size += post_processing_->GetConstantTensorsSize();
    total_size +=
        post_processing_->GetSizeOfMemoryAllocatedForIntermediateTensors();
    return total_size;
  }

  int GetMaxTopK() const { return max_top_k_; }

 private:
  // For unit tests only.
  friend class LlmOpenClSpouse;

  struct TransformerParams {
    std::unique_ptr<Tensor> params_i32;
    std::unique_ptr<ClOperation> write_i32_params;
  };
  struct TransformerCache {
    std::vector<std::unique_ptr<Tensor>> key_cached_tensors;
    std::vector<std::unique_ptr<Tensor>> value_cached_tensors;
    std::unique_ptr<InferenceContext> clean_cache;

    std::unique_ptr<InferenceContext> clean_cache_at_ids;
    GpuModelBuilder::TensorHandle ids_to_remove;
    std::unique_ptr<Buffer> ids_to_remove_buffer;
  };

  struct EmbeddingLookupModel {
    GpuModelBuilder::TensorHandle input_tokens;
    GpuModelBuilder::TensorHandle output_embeddings;
    std::vector<GpuModelBuilder::TensorHandle> output_ple_embeddings;
    std::vector<odml::infra::gpu::EmbeddingWeights> ple_weights;
    std::unique_ptr<InferenceContext> context;
    std::unique_ptr<ClOperation> copy_tokens_op;
  };

  // Function for initializing key/value cached tensors.
  absl::StatusOr<std::vector<std::unique_ptr<Tensor>>> InitKvTensors(
      const Environment& env, const TensorDescriptor& tensor_desc,
      int num_cache_layers);

  // Must be called before models initialization.
  absl::Status InitSharedTensors(TransformerBuilder* transformer_builder,
                                 TransformerCache& cache,
                                 TransformerParams& params,
                                 int num_cache_layers,
                                 Environment* env);

  absl::Status InitModel(TransformerBuilder* transformer_builder,
                         int sequence_size, int batch_size, Environment* env);
  absl::Status InitPostProcessingModel(TransformerBuilder* transformer_builder,
                                       Environment* env);
  absl::Status InitCleanCacheModel(TransformerBuilder* transformer_builder,
                                   TransformerCache& cache,
                                   TransformerParams& params, Environment* env);

  absl::Status InitAudioEncoderModel(AudioEncoderBuilder* audio_encoder_builder,
                                     Environment* env);

  absl::Status InitHelperOps(Environment* env);

  absl::Status InitInputTokensWithModelN(Environment* env,
                                         const std::vector<int>& input_tokens);

  absl::Status ExecuteCopyCache(Environment* env, int src_size, Tensor* src_k,
                                Tensor* src_v, int dst_size, Tensor* dst_k,
                                Tensor* dst_v, bool is_global = false,
                                CLEvent* event = nullptr);

  absl::Status ExecuteUpdateIntParams(Environment* env,
                                      TransformerConfig& config,
                                      TransformerParams& params,
                                      const LlmRuntimeParams& param_vals);
  absl::Status ExecuteUpdateParams(Environment* env, Tensor* tensor,
                                   const std::vector<float>& params);

  absl::Status ExecutePostProcessing(Environment* env, CLEvent* event);

  absl::Status ExecutePostProcessing(Environment* env);

  absl::Status EnqueueNextToken(Environment* env, int time_step,
                                CLEvent* event = nullptr);

  absl::StatusOr<GpuSpatialTensor*> CreateModelTensor(
      CLContext* context, std::vector<std::unique_ptr<Tensor>>* model_tensors,
      const TensorDescriptor& tensor_desc);
  absl::StatusOr<GpuSpatialTensor*> CreateModelTensorMmaped(
      CLContext* context, const GpuInfo& gpu_info,
      ::odml::infra::gpu::LlmTensorLoader* tensor_loader,
      std::vector<std::unique_ptr<Tensor>>* model_tensors,
      const TensorDescriptor& tensor_desc);
  void SetupOptionalNodes();

  struct StackedTransformerInference {
    std::unique_ptr<InferenceContext> context;
    int seq_size;
    GpuModelBuilder::TensorHandle input, output;
    std::unique_ptr<ClOperation> copy_nth_op;
    std::unique_ptr<ClOperation> copy_tokens_op;
    std::vector<odml::infra::gpu::EmbeddingWeights> ple_weights;
    std::vector<odml::infra::gpu::EmbeddingWeights> ngrammer_weights;
    // Holds the LoRA tensors created while building the LLM. The first key is
    // the LoRA rank and the second key is the name of the tensor.
    absl::flat_hash_map<int, absl::flat_hash_map<std::string, LoRATensorInfo>>
        lora_tensors;

    GpuModelBuilder::TensorHandle positions, local_mask;
    GpuModelBuilder::TensorHandle global_mask;
    GpuModelBuilder::TensorHandle external_embeddings;
    std::vector<GpuModelBuilder::TensorHandle> external_ple_embeddings;
    std::unique_ptr<ClOperation> copy_tokens_and_positions_op;
    std::unique_ptr<ClOperation> copy_positions_op;
    std::unique_ptr<ClOperation> copy_masks_op;
    std::unique_ptr<ClOperation> copy_external_embeddings_op;
    EmbeddingLookupModel embedding_lookup_model;

    // For debugging, tensors to inspect will be referenced here.
    absl::flat_hash_map<std::string, GpuModelBuilder::TensorHandle>
        debug_tensors;
  };

  absl::Status InitEmbeddingLookupModel(TransformerBuilder* transformer_builder,
                                        Environment* env,
                                        StackedTransformerInference& model,
                                        int sequence_size, int batch_size);

  absl::Status InitFinalEmbeddingToLogitsModel(
      TransformerBuilder* transformer_builder, Environment* env);

  absl::Status UpdateEmbeddings(Environment* env,
                                StackedTransformerInference& model,
                                int previous_token,
                                absl::Span<const int> tokens);

  absl::Status UpdatePleEmbeddings(Environment* env,
                                   StackedTransformerInference& model,
                                   absl::Span<const int> tokens);

  // src has shape: [1, 1, x, emb_dim] where x is the num of media hard tokens.
  // dst has shape: [1, 1, cache_size, emb_dim]
  // we perform: dst[0, 0, index:index + x, :] = src
  // Can be used for media embeddings and vision ple embeddings.
  absl::Status UpdateMultimodalEmbeddings(Tensor* src, Tensor* dst, int index,
                                          Environment* env);

  // Inserts the tensors passed into vision_ple_tensors_ to be used by text
  // decoder. Expect vision_ple_tensors to be of size (text_config_.stack_size -
  // text_config_.extra_stack_size). Index should be set to 0 unless the encoder
  // gets ahead of the decoder.
  absl::Status FillVisionPleEmbeddings(
      const std::vector<TensorFloat32>& vision_ple_tensors, int index,
      Environment* env);

  // Inserts the cpu media_tensor into media_embeddings_ tensor to be used by
  // text decoder. Index should be set to 0 unless the encoder gets ahead of
  // the decoder.
  absl::Status FillMediaEmbeddings(const TensorFloat32& media_tensor, int index,
                                   Environment* env);

  bool IsLayerGlobal(int layer_index) const;

  CreateGpuModelInfo create_info_;
  // Version of the InferenceContext for processing sequence length 1.
  StackedTransformerInference* model_one_ = nullptr;
  // A mapping from sequence size to model. This will always have at least a
  // single model with sequence size 1 that is referred to by |model_one_|.
  std::map<int, StackedTransformerInference> models_;
  std::vector<::odml::infra::gpu::GroupSize> group_sizes_;
  TransformerCache text_cache_;
  TransformerParams text_transformer_params_;
  std::unique_ptr<Tensor> params_f32_;
  std::unique_ptr<ClOperation> write_f32_params_;

  // Op for copying cache tensors for saving / reloading context.
  std::unique_ptr<ClOperation> copy_cache_;
  std::unique_ptr<ClOperation> copy_cache_global_;

  struct AudioParams {
    std::unique_ptr<InferenceContext> context;
    std::unique_ptr<Tensor> feature;
    std::unique_ptr<Tensor> mask;
    GpuModelBuilder::TensorHandle output_handle;  // audio enc output
    std::unique_ptr<Tensor> signal;               // v3 tensor for RelPE

    TransformerCache cache;
    TransformerParams transformer_params;
    AudioEncoderConfig config;

    // Audio group norm cumsum for streaming from audio encoder -> audio encoder
    std::unique_ptr<Tensor> stream_cumsum;  // audio stream for encoder n
    // Audio light conv1d requires data from previous frame for streaming.
    // Pass it from frame to frame with this tensor.
    // Note that we have one for each layer in the transformer.
    std::vector<std::unique_ptr<Tensor>> stream_depthwise_tensors;
  };
  AudioParams audio_params_;

  // Vision fields
  std::vector<std::unique_ptr<Tensor>> vision_ple_tensors_;

  // Multimodal fields
  std::unique_ptr<Tensor> media_embeddings_;
  // Note, if we ever want to support the audio encoder getting ahead of the
  // text decoder, we need to allow the user to set this value themselves.
  int media_cache_size_;

  // Op for copying multimodal embeddings.
  std::unique_ptr<ClOperation> multimodal_embeddings_op_;

  // Transform the tokens before they are passed to the embedding updater for
  // multimodal support.
  odml::infra::gpu::TokensTransformFn tokens_transform_fn_ = [](int token) {
    return token;
  };

  // This vector holds tensors that are used across model_one_, model_n_, and
  // post_processing_. Caching the tensors here allows sharing these tensors to
  // prevent duplicating them in GPU memory. This vector is populated by
  // LlmFileTensorLoader as the model is initialized.
  std::vector<std::unique_ptr<Tensor>> model_tensors_;
  std::vector<std::unique_ptr<Buffer>> mmaped_buffers_;
  std::vector<std::unique_ptr<::odml::infra::llm_utils::DataHolder<uint8_t>>>
      owned_weights_;

  std::unique_ptr<InferenceContext> post_processing_;
  std::unique_ptr<InferenceContext> softmax_;
  std::unique_ptr<InferenceContext> final_embedding_to_logits_;
  std::unique_ptr<Tensor> tokens_ids_;
  std::unique_ptr<Buffer> input_token_ids_;
  std::unique_ptr<Buffer> input_embeddings_;
  std::unique_ptr<Buffer> input_ple_embeddings_;

  GpuModelBuilder::TensorHandle token_probabilities_;
  GpuModelBuilder::TensorHandle logits_;
  GpuModelBuilder::TensorHandle final_embedding_;

  bool enable_decode_logits_;
  bool enable_external_embeddings_;

  std::unique_ptr<Buffer> input_sequence_position_;
  std::unique_ptr<Buffer> input_global_mask_;
  std::unique_ptr<Buffer> input_local_mask_;

  CLCommandQueue reading_queue_;
  CLEvent last_token_enqueued_event_;

  int ready_tokens_;
  std::vector<int> input_tokens_;
  int output_batch_size_;
  int max_tokens_;
  std::vector<int32_t> output_tokens_;  // CPU side Tensor tokens_ids_

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
    std::vector<std::unique_ptr<Tensor>> tensors;
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
  std::unique_ptr<Tensor> constraint_mask_;
  std::vector<bool> last_constraint_mask_;
  bool constraint_mask_changed_since_last_input_ = true;
  std::vector<int> suppressed_tokens_;
};

}  // namespace cl
}  // namespace ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_LLM_OPENCL_H_
