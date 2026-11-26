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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_LLM_WEBGPU_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_LLM_WEBGPU_H_

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
#include "ml_drift/common/executor.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_tensor.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "ml_drift/webgpu/buffer.h"  // from @ml_drift
#include "ml_drift/webgpu/compute_task.h"  // from @ml_drift
#include "ml_drift/webgpu/environment.h"  // from @ml_drift
#include "ml_drift/webgpu/execution_environment.h"  // from @ml_drift
#include "ml_drift/webgpu/inference_context.h"  // from @ml_drift
#include "ml_drift/webgpu/memory_manager.h"  // from @ml_drift
#include "ml_drift/webgpu/spatial_tensor.h"  // from @ml_drift
#include "ml_drift/webgpu/webgpu_headers.h"  // from @ml_drift
#include "runtime/executor/gpu_artisan/ml_drift/audio_encoder.h"
#include "runtime/executor/gpu_artisan/ml_drift/executor_factory.h"
#include "runtime/executor/gpu_artisan/ml_drift/group_size.h"
#include "runtime/executor/gpu_artisan/ml_drift/llm.h"
#include "runtime/executor/gpu_artisan/ml_drift/runtime_params.h"
#include "runtime/executor/gpu_artisan/ml_drift/speculative_utils.h"
#include "runtime/executor/gpu_artisan/ml_drift/tensor_loaders/embedding_updater.h"
#include "runtime/executor/gpu_artisan/ml_drift/tensor_loaders/llm_tensor_loader.h"
#include "runtime/executor/gpu_artisan/ml_drift/tensor_loaders/placeholder_tensor_loader.h"
#include "runtime/executor/gpu_artisan/ml_drift/transformer.h"
#include "runtime/executor/gpu_artisan/proto/sampler_params.pb.h"
#include "runtime/executor/gpu_artisan/utils/bitmap.h"
#include "runtime/executor/gpu_artisan/utils/model_data.h"

#ifndef __EMSCRIPTEN__
#include <functional>
#endif  // __EMSCRIPTEN__

namespace ml_drift {
namespace webgpu {

// API for initializing and running inference on LLMs with MLDrift and WebGPU.
class LlmWebGpu {
 public:
  absl::Status Init(const LlmConfig& config, int outputs_count,
                    const std::string& weights_path,
                    const CreateGpuModelInfo& create_info, int gamma,
                    ExecutionEnvironment* env, bool init_audio_encoder = false);
  absl::Status Init(
      const LlmConfig& config, int outputs_count,
      std::unique_ptr<::odml::infra::gpu::LlmTensorLoader> tensor_loader,
      const CreateGpuModelInfo& create_info,
      std::shared_ptr<::odml::infra::gpu::MlDriftExecutorFactory>
          executor_factory,
      int gamma, ExecutionEnvironment* env, bool init_audio_encoder = false);

  absl::Status InitAudioEncoder(
      const AudioEncoderConfig& config,
      std::unique_ptr<::odml::infra::gpu::LlmTensorLoader> tensor_loader,
      CreateGpuModelInfo& create_info, int output_batch_size,
      ExecutionEnvironment* env);

  // Zeroes key/value cached tensors
  absl::Status CleanCache(ExecutionEnvironment* env, int time_step,
                          bool skip_ringbuffer_delta_check = false);
  absl::Status CleanAudioCache(ExecutionEnvironment* env, int time_step);

  // InitInputTokens and GetNextTokens must be used only together.
  // InitInputTokens gpu asynchronous, for measurements
  // ml_drift::webgpu::WaitUntilCompleted() must be used.
#ifdef __EMSCRIPTEN__
  absl::Status InitInputTokens(ExecutionEnvironment* env,
                               const std::vector<int>& input_tokens,
                               int time_step);
#else
  using CancelFn = std::function<bool()>;
  absl::Status InitInputTokens(ExecutionEnvironment* env,
                               const std::vector<int>& input_tokens,
                               int time_step, CancelFn canceled);
#endif  // __EMSCRIPTEN__

  absl::StatusOr<std::vector<float>> GetNextTokenProbabilities(
      ExecutionEnvironment* env);
  absl::Status GetNextTokens(ExecutionEnvironment* env,
                             std::vector<int>* output_tokens);

  absl::StatusOr<std::vector<float>> DecodeToLogits(
      Environment* env, uint32_t model_size, uint32_t cache_size,
      absl::Span<const float> embeddings,
      absl::Span<const float> ple_embeddings,
      absl::Span<const float> local_attention_mask,
      absl::Span<const float> global_attention_mask,
      absl::Span<const int> seq_position,
      absl::Span<const uint32_t> logits_indices,
      const std::atomic_bool* cancel);

  absl::StatusOr<std::vector<std::pair<uint32_t, uint32_t>>> GetVariants(
      Environment* env);

  absl::Status ComputeTokenEmbeddings(Environment* env,
                                      absl::Span<const int> input_tokens,
                                      absl::Span<float> output_embeddings,
                                      absl::Span<float> output_ple_embeddings);

  // for multiple output
  using UpdateConstraintFn = std::function<
      absl::StatusOr<std::unique_ptr<odml::infra::llm_utils::Bitmap>>(int)>;
  absl::Status GetNextTokens(ExecutionEnvironment* env,
                             std::vector<std::vector<int>>* output_tokens,
                             UpdateConstraintFn update_constraint_fn);

  absl::StatusOr<GpuSpatialTensor*> ExecuteAudioEncoder(
      const TensorFloat32& audio_tensor, const TensorFloat32& mask_tensor,
      int time_step, ExecutionEnvironment* env);

  // Inserts vision_embedding tensor into media_embeddings_ and
  // vision_ple_tensors into vision_ple_tensors_ to be used by text decoder.
  // Both at index = image_index * text_config_.vision_tokens_nums
  absl::Status FillVisionEmbeddings(
      const ml_drift::TensorFloat32& vision_embedding_tensor,
      const std::vector<ml_drift::TensorFloat32>& vision_ple_tensors,
      int image_index, ExecutionEnvironment* env);

  // Inserts the gpu audio_tensor into media_embeddings_ tensor to be used by
  // text decoder. Index should be set to 0 unless the encoder gets ahead of
  // the decoder. If you wish to do this, you will need to add functionality to
  // manually set media_cache_size_ to be larger than the audio encoder output.
  absl::Status FillAudioEmbeddings(SpatialTensor& audio_tensor, int index,
                                   ExecutionEnvironment* env);

  // Clears the vision_ple_tensors_ to be used by text decoder. This should
  // be called if we want to run audio after filling the vision
  // ple embeddings. Ie, we should have a sequence such as:
  // 1. FillVisionEmbeddings
  // 2. InitInputTokens
  // 3. ClearVisionPle
  // 4. FillAudioEmbeddings
  absl::Status ClearVisionPle(ExecutionEnvironment* env);

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

  // These methods save and restore the KV cache to a ContextData object. These
  // keep all the copy operations in GPU buffers instead of downloading /
  // uploading to CPU.
  absl::StatusOr<ContextData> SaveContext(ExecutionEnvironment* env,
                                          bool local_only = false);
  absl::Status RestoreContext(ExecutionEnvironment* env,
                              const ContextData& context_data,
                              bool local_only = false);
  absl::StatusOr<ContextData> CloneContext(ExecutionEnvironment* env,
                                           const ContextData& context_data,
                                           bool local_only = false);
  absl::StatusOr<ContextData> SaveLocalContext(ExecutionEnvironment* env) {
    return SaveContext(env, /*local_only=*/true);
  }
  absl::Status RestoreLocalContext(ExecutionEnvironment* env,
                                   const ContextData& context_data) {
    return RestoreContext(env, context_data, /*local_only=*/true);
  }
  absl::StatusOr<ContextData> CloneLocalContext(
      ExecutionEnvironment* env, const ContextData& context_data) {
    return CloneContext(env, context_data, /*local_only=*/true);
  }
  absl::Status LoadLoRA(
      ExecutionEnvironment* env, uint32_t lora_id, int lora_rank,
      std::unique_ptr<::odml::infra::gpu::LlmTensorLoader> tensor_loader);
  absl::Status UnloadLoRA(ExecutionEnvironment* env, uint32_t lora_id);
  absl::Status UseLoRA(ExecutionEnvironment* env,
                       std::optional<uint32_t> lora_id);
  absl::StatusOr<AudioContextData> SaveAudioContext(ExecutionEnvironment* env);
  absl::Status RestoreAudioContext(ExecutionEnvironment* env,
                                   const AudioContextData& context_data);
  absl::StatusOr<AudioContextData> CloneAudioContext(
      ExecutionEnvironment* env, const AudioContextData& context_data);
  absl::Status UseLayers(ExecutionEnvironment* env,
                         const absl::flat_hash_set<int>& layers, int final_ln);

  absl::Status UpdateSamplerParams(
      const odml::infra::proto::SamplerParameters& new_params,
      std::shared_ptr<std::default_random_engine> rand_gen);

  absl::Status UpdateConstraintMask(
      const odml::infra::llm_utils::Bitmap& constraint_bitmap,
      ExecutionEnvironment* env);

  void WaitForWeightUploads() { buffer_upload_executor_ = nullptr; }

  // Get the maximum possible Top-K value. Top-K provided to UpdateSamplerParams
  // should not be larger than max Top-K.
  int GetMaxTopK() { return max_top_k_; }

  // Speculative decoding methods used by both drafter and main model.

  // The LlmWebGpu runner class actually holds the next token to be predicted in
  // memory, ie. after every sampled token that is return, returned token is
  // always passed through the transformer stack again so there's always an
  // available token that is hidden in the LlmWebGpu class and not returned.
  // This function allows the caller to peek at that token. Returned is the time
  // step and token ID being sampled by the last forward path.
  absl::StatusOr<speculative_utils::TokenAtTimeStep> PeekNextToken(
      ExecutionEnvironment* env);

  // Retrieve the top k probabilities and tokens for the corresponding time
  // step. This will only provide the probabilities after DraftTokens()
  // finished.
  absl::StatusOr<std::vector<speculative_utils::TokenProbability>>
  GetProbabilityCache(int time_step);

  // New methods used by the drafter model.

  // The LlmWebGpu class always holds the next sampled but not yet returned
  // token, that token is used as the basis for predicting the next token in
  // both GetNextTokens() and DraftTokens(). In order to ensure correct
  // prediction, we need to override that token to whatever the result of
  // speculating decoding is so that further drafting is based on correct input.
  absl::Status OverwriteOneInputToken(
      ExecutionEnvironment* env,
      const speculative_utils::TokenAtTimeStep& token);

  // Attempts to draft gamma_ + 1 tokens. This is similar to GetNextTokens()
  // but:
  // - number of output tokens will be gamma_ + 1.
  // - The probability distribution is available from GetProbabilityCache()
  absl::StatusOr<std::vector<int>> DraftTokens(ExecutionEnvironment* env);

  // New methods used by the main model.

  // Passes these tokens through the LLM stack in 1 pass, the single batch
  // will go through the transformer stack, projection (Fully Connected
  // Linear) and sampling.
  // Note that if any token is rejected it is up to the caller to call
  // RejectAfterResample().
  absl::Status ScoreTokens(ExecutionEnvironment* env,
                           const std::vector<int>& tokens);

  // Called after the token at `resampled.time_step()` is rejected and replaced
  // with `resampled.token()`. This function will ensure that the next
  // invocation of PeekNextToken() returns the correct value. While this is
  // similar to OverwriteOneInputToken(), it differs in the sense that in
  // packages CleanCache() operation together as well. This should be used when
  // any token is rejected (and thus CleanCache() is needed) while
  // OverwriteOneInputToken() should be used when no tokens are rejected (thus
  // no CleanCache() is needed).
  absl::Status RejectAfterResample(
      ExecutionEnvironment* env,
      const speculative_utils::TokenAtTimeStep& resampled);

 private:
  // For unit tests.
  friend class LlmWebGpuSpouse;

  struct TransformerCache {
    std::vector<std::unique_ptr<SpatialTensor>> key_cached_tensors;
    std::vector<std::unique_ptr<SpatialTensor>> value_cached_tensors;
    std::unique_ptr<InferenceContext> clean_cache;
  };
  struct TransformerParams {
    std::unique_ptr<ComputeTask> write_i32_params;
    std::unique_ptr<SpatialTensor> params_i32;
  };

  struct EmbeddingLookupModel {
    GpuModelBuilder::TensorHandle input_tokens;
    GpuModelBuilder::TensorHandle output_embeddings;
    std::vector<GpuModelBuilder::TensorHandle> output_ple_embeddings;
    std::vector<odml::infra::gpu::EmbeddingWeights> ple_weights;
    std::unique_ptr<InferenceContext> context;
    std::unique_ptr<ComputeTask> copy_tokens_op;
  };

  struct StackedTransformerInference {
    std::unique_ptr<InferenceContext> context;
    int seq_size;
    GpuModelBuilder::TensorHandle input, output;
    std::unique_ptr<ComputeTask> copy_nth_op;
    std::unique_ptr<ComputeTask> copy_tokens_op;
    // Holds the layer tensors reload functions for different transformer layer.
    absl::flat_hash_map<std::string,
                        std::vector<::odml::infra::gpu::LoadTensorFn>>
        tensor_reload_fns;
    std::vector<odml::infra::gpu::EmbeddingWeights> ple_weights;
    std::vector<odml::infra::gpu::EmbeddingWeights> ngrammer_weights;
    // Holds the LoRA tensors created while building the LLM. The first key is
    // the LoRA rank and the second key is the name of the tensor.
    absl::flat_hash_map<int, absl::flat_hash_map<std::string, LoRATensorInfo>>
        lora_tensors;

    // Below are for exposing logits and embedding tensors:
    // DecodeToLogits and ComputeTokenEmbeddings APIs.
    GpuModelBuilder::TensorHandle positions;
    GpuModelBuilder::TensorHandle local_mask;
    GpuModelBuilder::TensorHandle global_mask;
    GpuModelBuilder::TensorHandle external_embeddings;
    std::vector<GpuModelBuilder::TensorHandle> external_ple_embeddings;
    std::unique_ptr<ComputeTask> copy_positions_op;
    std::unique_ptr<ComputeTask> copy_masks_op;
    std::unique_ptr<ComputeTask> copy_external_embeddings_op;
    EmbeddingLookupModel embedding_lookup_model;
    // For debugging, tensors to inspect will be referenced here.
    absl::flat_hash_map<std::string, GpuModelBuilder::TensorHandle>
        debug_tensors;
  };
  bool IsLayerGlobal(int layer_number);
  absl::Status InitEmbeddingLookupModel(TransformerBuilder* transformer_builder,
                                        ExecutionEnvironment* env,
                                        StackedTransformerInference& model,
                                        int sequence_size, int batch_size);
  absl::Status InitFinalEmbeddingToLogitsModel(
      TransformerBuilder* transformer_builder, ExecutionEnvironment* env);

  // Must be called before models initialization.
  absl::Status InitParams(TransformerBuilder* transformer_builder,
                          TransformerParams& params, ExecutionEnvironment* env);
  absl::Status InitCache(TransformerBuilder* transformer_builder,
                         TransformerCache& cache, int num_cache_layers,
                         ExecutionEnvironment* env);
  absl::Status InitCache(TransformerBuilder* transformer_builder,
                         TransformerCache& cache,
                         int num_local_layers_per_global, int num_cache_layers,
                         std::optional<int> global_layer_offset,
                         ExecutionEnvironment* env);
  absl::Status InitSharedTensors(TransformerBuilder* transformer_builder,
                                 const TransformerConfig& config,
                                 ExecutionEnvironment* env);

  absl::Status InitModel(
      TransformerBuilder* transformer_builder, int batch_size,
      odml::infra::gpu::PlaceholderTensorLoader* placeholder_loader,
      StackedTransformerInference& model_n, ExecutionEnvironment* env);
  absl::Status InitPostProcessingModel(TransformerBuilder* transformer_builder,
                                       ExecutionEnvironment* env);
  absl::Status InitCleanCacheModel(TransformerBuilder* transformer_builder,
                                   TransformerCache& cache,
                                   TransformerParams& params,
                                   ExecutionEnvironment* env);
  absl::Status InitAudioEncoderModel(AudioEncoderBuilder* audio_encoder_builder,
                                     ExecutionEnvironment* env);
#ifdef __EMSCRIPTEN__
  absl::Status InitInputTokensWithModelN(ExecutionEnvironment* env,
                                         const std::vector<int>& input_tokens);
#else
  absl::Status InitInputTokensWithModelN(ExecutionEnvironment* env,
                                         const std::vector<int>& input_tokens,
                                         CancelFn canceled);
#endif  // __EMSCRIPTEN__

  absl::Status EncodeUpdateParams(ExecutionEnvironment* env,
                                  TransformerConfig& config,
                                  TransformerParams& params,
                                  const LlmRuntimeParams& param_vals,
                                  wgpu::ComputePassEncoder encoder);
  absl::Status ExecuteUpdateIntParams(ExecutionEnvironment* env,
                                      TransformerConfig& config,
                                      TransformerParams& params,
                                      const LlmRuntimeParams& param_vals);
  absl::Status ExecuteUpdateParams(ExecutionEnvironment* env,
                                   SpatialTensor* tensor,
                                   const std::vector<float>& params);
  absl::Status ExecuteUpdateScoringParams(ExecutionEnvironment* env,
                                          SpatialTensor* tensor,
                                          const std::vector<float>& params);
  absl::Status EncodeCopyCache(ExecutionEnvironment* env, int src_size,
                               SpatialTensor* src_k, SpatialTensor* src_v,
                               int dst_size, SpatialTensor* dst_k,
                               SpatialTensor* dst_v,
                               wgpu::ComputePassEncoder encoder,
                               bool is_global = false);

  absl::Status ExecutePostProcessing(ExecutionEnvironment* env);

  absl::StatusOr<GpuSpatialTensor*> CreateModelTensor(
      ExecutionEnvironment* env,
      ::odml::infra::gpu::LlmTensorLoader* tensor_loader,
      std::vector<std::unique_ptr<SpatialTensor>>* model_tensors,
      const TensorDescriptor& tensor_desc);
#ifndef __EMSCRIPTEN__
  absl::Status CopyBufferToTensor(
      ExecutionEnvironment* env, const TensorDescriptor& desc,
      std::unique_ptr<odml::infra::llm_utils::DataHolder<uint8_t>>
          owned_weights,
      SpatialTensor* tensor);
#endif  // !__EMSCRIPTEN__

  void SetupOptionalNodes();

  absl::Status GetNextTokensWithPullTopKProbabilities(
      ExecutionEnvironment* env, std::vector<std::vector<int>>* output_tokens,
      bool pull_top_k_probabilities, UpdateConstraintFn update_constraint_fn);

  absl::Status ProcessNextOutputToken(ExecutionEnvironment* env,
                                      bool pull_top_k_probabilities);

  // Pulls out the top K probabilities from the GPU buffer.
  absl::Status PullTopKProbabilities(ExecutionEnvironment* env, int time_step);

  // Pulls out the top K probabilities from the GPU buffer for scoring post
  // processing inference context.
  absl::Status PullScoringTopKProbabilities(ExecutionEnvironment* env,
                                            int time_step);

  absl::Status UpdateEmbeddings(ExecutionEnvironment* env,
                                StackedTransformerInference& model,
                                int previous_token,
                                absl::Span<const int> tokens);

  absl::Status UpdatePleEmbeddings(Environment* env,
                                   StackedTransformerInference& model,
                                   absl::Span<const int> tokens);

  // Internal methods used for speculative decoding.
  absl::Status InitScoringPostProcessingModel(
      TransformerBuilder* transformer_builder, ExecutionEnvironment* env);

  // Downloads and returns the contents of the GPU tokens buffer (input/output
  // stream).
  absl::StatusOr<TensorInt32> DownloadTokenIds(ExecutionEnvironment* env);

  // src has shape: [1, 1, x, emb_dim] where x is the num of media hard tokens.
  // dst has shape: [1, 1, cache_size, emb_dim]
  // we perform: dst[0, 0, index:index + x, :] = src
  // Can be used for media embeddings and vision ple embeddings.
  absl::Status UpdateMultimodalEmbeddings(SpatialTensor* src,
                                          SpatialTensor* dst, int index,
                                          ExecutionEnvironment* env,
                                          const wgpu::CommandEncoder& encoder);

  // Inserts the tensors passed into vision_ple_tensors_ to be used by text
  // decoder. Expect vision_ple_tensors to be of size (text_config_.stack_size -
  // text_config_.extra_stack_size). Index should be set to 0 unless the encoder
  // gets ahead of the decoder.
  absl::Status FillVisionPleEmbeddings(
      const std::vector<TensorFloat32>& vision_ple_tensors, int index,
      ExecutionEnvironment* env, const wgpu::CommandEncoder& encoder,
      std::vector<std::unique_ptr<SpatialTensor>>& temp_tensors);

  // Inserts the cpu media_tensor into media_embeddings_ tensor to be used by
  // text decoder. Index should be set to 0 unless the encoder gets ahead of
  // the decoder.
  absl::Status FillMediaEmbeddings(
      const TensorFloat32& media_tensor, int index, ExecutionEnvironment* env,
      const wgpu::CommandEncoder& encoder,
      std::vector<std::unique_ptr<SpatialTensor>>& temp_tensors);

  // Prepares `model.copy_tokens_op` with `input_tokens`. Executes the operation
  // immediately if `compute_pass_helper` is null, otherwise adds the operation
  // to the compute pass.
  absl::Status WriteInputTokensForModel(
      const ExecutionEnvironment& env, StackedTransformerInference& model,
      absl::Span<const int> input_tokens, int token_index_offset = 0,
      const wgpu::ComputePassEncoder* compute_pass_helper = nullptr);

  CreateGpuModelInfo create_info_;
  // Version of the InferenceContext for processing sequence length 1.
  StackedTransformerInference* model_one_ = nullptr;
  // A mapping from sequence size to model. This will always have at least a
  // single model with sequence size 1 that is referred to by |model_one_|.
  MemoryManager memory_manager_;
  std::map<int, StackedTransformerInference> models_;
  std::vector<::odml::infra::gpu::GroupSize> group_sizes_;
  TransformerCache text_cache_;
  TransformerParams text_params_;
  std::unique_ptr<SpatialTensor> params_f32_;
  std::unique_ptr<ComputeTask> write_f32_params_;

  // Audio fields
  struct AudioParams {
    std::unique_ptr<InferenceContext> context;
    std::unique_ptr<SpatialTensor> feature;
    std::unique_ptr<SpatialTensor> mask;
    GpuModelBuilder::TensorHandle output_handle;  // audio enc output
    std::unique_ptr<SpatialTensor> signal;        // v3 tensor for RelPE

    TransformerCache cache;
    TransformerParams transformer_params;
    AudioEncoderConfig config;

    // Audio group norm cumsum for streaming from audio encoder -> audio encoder
    std::unique_ptr<SpatialTensor> stream_cumsum;  // audio stream for encoder n
    // Audio light conv1d requires data from previous frame for streaming.
    // Pass it from frame to frame with this tensor.
    // Note that we have one for each layer in the transformer.
    std::vector<std::unique_ptr<SpatialTensor>> stream_depthwise_tensors;
  };
  AudioParams audio_params_;

  // Vision fields
  std::vector<std::unique_ptr<SpatialTensor>> vision_ple_tensors_;

  // Multimodal fields
  std::unique_ptr<SpatialTensor> media_embeddings_;
  // Note, if we ever want to support the audio encoder getting ahead of the
  // text decoder, we need to allow the user to set this value themselves.
  int media_cache_size_;

  // Op for copying multimodal embeddings.
  std::unique_ptr<ComputeTask> multimodal_embeddings_op_;

  // Transform the tokens before they are passed to the embedding updater for
  // multimodal support.
  odml::infra::gpu::TokensTransformFn tokens_transform_fn_ = [](int token) {
    return token;
  };

  // This vector holds tensors that are used across model_one_, model_n_, and
  // post_processing_. Caching the tensors here allows sharing these tensors to
  // prevent duplicating them in GPU memory. This vector is populated by
  // LlmFileTensorLoader as the model is initialized.
  std::vector<std::unique_ptr<SpatialTensor>> model_tensors_;

  std::unique_ptr<InferenceContext> post_processing_;
  std::unique_ptr<InferenceContext> softmax_;
  std::unique_ptr<InferenceContext> final_embedding_to_logits_;
  std::unique_ptr<SpatialTensor> tokens_ids_;
  std::unique_ptr<SpatialTensor> topk_tokens_;
  std::unique_ptr<SpatialTensor> topk_probabilities_;
  std::unique_ptr<Buffer> input_token_ids_;
  std::unique_ptr<Buffer> input_embeddings_;
  std::unique_ptr<Buffer> input_ple_embeddings_;
  std::unique_ptr<SpatialTensor> temporary_output_tensor_;
  GpuModelBuilder::TensorHandle token_probabilities_;
  GpuModelBuilder::TensorHandle logits_;
  GpuModelBuilder::TensorHandle final_embedding_;

  bool enable_decode_logits_;

  std::unique_ptr<Buffer> input_sequence_position_;
  std::unique_ptr<Buffer> input_global_mask_;
  std::unique_ptr<Buffer> input_local_mask_;

  // The constraint mask used for constrained decoding, which indicates the
  // allowed tokens during sampling.
  std::unique_ptr<SpatialTensor> constraint_mask_;

  int ready_tokens_ = 0;
  int num_output_tokens_ = 0;
  std::vector<int> input_tokens_;
  int output_batch_size_ = 0;

  // True when GetNextTokens() is not ready.
  bool needs_init_input_tokens_ = true;

  std::vector<std::string> layer_names_;

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
    std::vector<std::unique_ptr<SpatialTensor>> tensors;
  };
  absl::flat_hash_map<uint32_t, std::unique_ptr<LoRALoader>> lora_loaders_;

  // The tensor loader used to load weights while building the LLM.
  // Note: It only valid when we enable `use_optional_transformer_layers`.
  std::unique_ptr<::odml::infra::gpu::LlmTensorLoader> tensor_loader_;

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
  std::vector<int> suppressed_tokens_;

  // Executors for uploading weights.
  std::shared_ptr<::odml::infra::gpu::MlDriftExecutorFactory> executor_factory_;
  std::unique_ptr<::ml_drift::Executor> buffer_upload_executor_;

  // Variables for speculative decoding.
  // See Gamma in LlmSpeculativeRunner class.
  // Set Gamma to 0 to disable speculative decoding.
  int gamma_;

  // InferenceContext, models or tensors for speculative decoding.
  StackedTransformerInference scoring_model_;
  std::unique_ptr<InferenceContext> scoring_post_processing_;
  std::unique_ptr<SpatialTensor> scoring_params_f32_;
  std::unique_ptr<SpatialTensor> scoring_topk_tokens_;
  std::unique_ptr<SpatialTensor> scoring_topk_probabilities_;
  std::unique_ptr<ComputeTask> write_scoring_params_f32_;
  std::unique_ptr<ComputeTask> copy_cache_;
  std::unique_ptr<ComputeTask> copy_cache_global_;

  // Maps from time_step to token ID. This is used to override the next input
  // token for inferencing in transformer stack.
  absl::flat_hash_map<int, int> token_override_;

  // Holds the probability distribution (top K) for recent calls to
  // DraftTokens() or ScoreTokens().
  absl::flat_hash_map<int, std::vector<speculative_utils::TokenProbability>>
      probability_cache_;

  // Tracks the latest constraint that was set to avoid re-uploading the same
  // mask.
  std::vector<bool> last_constraint_mask_;
  bool constraint_mask_changed_since_last_input_ = true;
};

}  // namespace webgpu
}  // namespace ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_LLM_WEBGPU_H_
