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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_LLM_GPU_RUNNER_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_LLM_GPU_RUNNER_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <random>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_tensor.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/executor/gpu_artisan/ml_drift/audio_encoder.h"
#include "runtime/executor/gpu_artisan/ml_drift/executor_factory.h"
#include "runtime/executor/gpu_artisan/ml_drift/llm.h"
#include "runtime/executor/gpu_artisan/ml_drift/speculative_utils.h"
#include "runtime/executor/gpu_artisan/ml_drift/tensor_loaders/llm_tensor_loader.h"
#include "runtime/executor/gpu_artisan/proto/sampler_params.pb.h"
#include "runtime/executor/gpu_artisan/utils/bitmap.h"

#ifdef __EMSCRIPTEN__
#include <webgpu/webgpu_cpp.h>
#else
#include <atomic>
#endif  // __EMSCRIPTEN__

#ifdef ML_DRIFT_PROFILING
#include "ml_drift/cl/environment.h"  // from @ml_drift
#include "ml_drift/cl/inference_context.h"  // from @ml_drift
#endif  // ML_DRIFT_PROFILING

namespace ml_drift {
namespace webgpu {
class Environment;
}  // namespace webgpu
namespace cl {
class Environment;
}  // namespace cl
namespace metal {
class Environment;
}  // namespace metal
}  // namespace ml_drift

namespace odml::infra::gpu {

using RunnerEnvironment =
    std::variant<::ml_drift::webgpu::Environment*, ::ml_drift::cl::Environment*,
                 ::ml_drift::metal::Environment*>;

class Drafter;

enum class LlmGpuRunnerType {
  kUnknown,
  kOpenCl,
  kWebGpu,
  kWebGpu_Emscripten,
  kMetal,
};

// A helper class to hide the difference between OpenCl, WebGPU, and Metal.
class LlmGpuRunner {
 public:
  virtual ~LlmGpuRunner() = default;

#ifdef __EMSCRIPTEN__
  static absl::StatusOr<std::unique_ptr<LlmGpuRunner>> Create(
      const wgpu::Device& device, const wgpu::AdapterInfo& adapter_info,
      const ml_drift::LlmConfig& llm_config,
      std::unique_ptr<LlmTensorLoader> tensor_loader,
      ml_drift::CreateGpuModelInfo create_info, int num_output_heads);

  // InitInputTokens gpu asynchronous.
  virtual absl::Status InitInputTokens(const std::vector<int>& input_tokens,
                                       int current_step,
                                       bool wait_for_completion) = 0;
#else
  static absl::StatusOr<std::unique_ptr<LlmGpuRunner>> Create(
      const ml_drift::LlmConfig& llm_config,
      std::unique_ptr<LlmTensorLoader> tensor_loader,
      ml_drift::CreateGpuModelInfo create_info, int num_output_heads,
      bool wait_for_weight_uploads,
      std::shared_ptr<MlDriftExecutorFactory> executor_factory = nullptr);

  // InitInputTokens gpu asynchronous.
  virtual absl::Status InitInputTokens(const std::vector<int>& input_tokens,
                                       int current_step,
                                       bool wait_for_completion,
                                       const std::atomic_bool* cancel) = 0;
#endif  // __EMSCRIPTEN__

  // Returns the type of the runner.
  virtual LlmGpuRunnerType GetLlmGpuRunnerType() {
    return LlmGpuRunnerType::kUnknown;
  }

  // |main_runner| is an LlmGpuRunner for the main model and should implement
  // methods required for speculative decoding main model, that is, those
  // documented as speculative decoding methods used by main model.
  // |drafter| is a drafter, which may wrap an LlmGpuRunner i.e. a drafter model
  // and should implement methods required for speculative decoding drafting.
  // |gamma| is the Gamma value for speculative decoding, see
  // LlmSpeculativeRunner::gamma_ for more info.
  // |rand_gen| is a random source, allowing the caller to ensure deterministic
  // generation.
  static absl::StatusOr<std::unique_ptr<LlmGpuRunner>> CreateSpeculativeRunner(
      std::unique_ptr<LlmGpuRunner> main_runner,
      std::unique_ptr<Drafter> drafter, int gamma,
      std::shared_ptr<std::default_random_engine> rand_gen);

  virtual absl::StatusOr<std::vector<float>> GetNextTokenProbabilities() {
    return absl::UnimplementedError(
        "GetNextTokenProbabilities not implemented for this backend.");
  }

  // This function will generate KV cache
  // entries for all the tokens in the input_tokens, and generate logits for
  // the tokens listed in the logits_indices. There's no direct support for
  // batching, the caller is responsible for flattening the batches into
  // virtual batches.
  // DecodeToLogits is used to generate logits for the given set of input
  // tokens.
  // The returned logits are packed into a 1D array with the shape
  // [logits_indices.size()][VOCAB_SIZE].
  //
  // Args:
  // model_size: The model size is the number of tokens that can be processed
  //   in a single forward pass.
  // cache_size: The cache size is the number of tokens that can be stored in
  //   the KV cache.
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

  virtual absl::StatusOr<std::vector<float>> DecodeToLogits(
      const uint32_t model_size, const uint32_t cache_size,
      absl::Span<const int> input_tokens,
      absl::Span<const float> local_attention_mask,
      absl::Span<const float> global_attention_mask,
      absl::Span<const int> seq_position,
      absl::Span<const uint32_t> logits_indices,
      const std::atomic_bool* cancel) {
    return absl::UnimplementedError(
        "DecodeToLogits not implemented for this backend.");
  }
  virtual absl::StatusOr<std::vector<float>> DecodeToLogits(
      const uint32_t model_size, const uint32_t cache_size,
      absl::Span<const float> embeddings,
      absl::Span<const float> ple_embeddings,
      absl::Span<const float> local_attention_mask,
      absl::Span<const float> global_attention_mask,
      absl::Span<const int> seq_position,
      absl::Span<const uint32_t> logits_indices,
      const std::atomic_bool* cancel) {
    return absl::UnimplementedError(
        "DecodeToLogits not implemented for this backend.");
  }
  virtual absl::Status ComputeTokenEmbeddings(
    absl::Span<const int> input_tokens,
      absl::Span<float> output_embeddings,
      absl::Span<float> output_ple_embeddings) {
    return absl::UnimplementedError(
        "ComputeTokenEmbeddings not implemented for this backend.");
  }
  virtual absl::StatusOr<std::vector<std::pair<uint32_t, uint32_t>>>
  GetVariants() {
    return absl::UnimplementedError(
        "GetVariants not implemented for this backend.");
  }
  virtual absl::Status CleanCacheAt(absl::Span<const int32_t> indices) {
    return absl::UnimplementedError(
        "CleanCacheAt not implemented for this backend.");
  }

  // Clean the cache including and after the given current time step (the number
  // of tokens).
  virtual absl::Status CleanCache(int time_step,
                                  bool skip_ringbuffer_delta_check) = 0;
  absl::Status CleanCache(int time_step) {
    return CleanCache(time_step, /*skip_ringbuffer_delta_check=*/false);
  }

  using UpdateConstraintFn =
      std::function<absl::StatusOr<std::unique_ptr<llm_utils::Bitmap>>(int)>;
  virtual absl::Status GetNextTokens(
      std::vector<std::vector<int>>* output_tokens,
      UpdateConstraintFn update_constraint_fn) = 0;

  virtual absl::StatusOr<ml_drift::ContextData> SaveContext() {
    return absl::UnimplementedError(
        "SaveContext not implemented for this backend.");
  }

  virtual absl::Status RestoreContext(ml_drift::ContextData&& context_data) {
    return absl::UnimplementedError(
        "RestoreContext not implemented for this backend.");
  }

  virtual absl::Status RestoreContextWithCacheCleaning(
      const ml_drift::ContextData& context_data) {
    return absl::UnimplementedError(
        "RestoreContextWithCacheCleaning not implemented for this backend.");
  }

  virtual absl::StatusOr<ml_drift::ContextData> CloneContext(
      const ml_drift::ContextData& context_data) {
    return absl::UnimplementedError(
        "CloneContext not implemented for this backend.");
  }

  virtual absl::StatusOr<ml_drift::ContextData> SaveLocalContext() {
    return absl::UnimplementedError(
        "SaveLocalContext not implemented for this backend.");
  }

  virtual absl::Status RestoreLocalContext(
      ml_drift::ContextData&& context_data) {
    return absl::UnimplementedError(
        "RestoreLocalContext not implemented for this backend.");
  }

  virtual absl::StatusOr<ml_drift::ContextData> CloneLocalContext(
      const ml_drift::ContextData& context_data) {
    return absl::UnimplementedError(
        "CloneLocalContext not implemented for this backend.");
  }

  virtual absl::Status LoadLoRA(
      uint32_t lora_id, int lora_rank,
      std::unique_ptr<LlmTensorLoader> tensor_loader) = 0;
  virtual absl::Status UnloadLoRA(uint32_t lora_id) = 0;

  virtual absl::Status UseLoRA(std::optional<uint32_t> lora_id) = 0;

  virtual absl::Status UseLayers(const absl::flat_hash_set<int>& layers,
                                 int final_ln) = 0;

  // Vision tensor should have shape [1, 1, x, emb_dim] where x is the maximum
  // number of vision tokens. Vision tensor will be inserted into media cache.
  // The media cache has size [1, 1, media_cache_size, emb_dim].
  // media_cache[0, 0, index:index+x, :] = vision_tensor.
  virtual absl::Status FillVisionEmbeddings(
      const ml_drift::TensorFloat32& vision_embedding_tensor,
      const std::vector<ml_drift::TensorFloat32>& vision_ple_tensors,
      int image_index) {
    return absl::UnimplementedError(
        "FillVisionEmbeddings not implemented for this backend.");
  }

  // Audio tensor should have shape [1, 1, x, emb_dim] where x is the number
  // of audio hard tokens. Audio tensor will be inserted into the media cache.
  // The media cache has size [1, 1, media_cache_size, emb_dim].
  // media_cache[0, 0, index:index+x, :] = audio_tensor.
  virtual absl::Status FillAudioEmbeddingsWithTensorBuffer(
      const ::litert::TensorBuffer& audio_tensor_buffer, int index) {
    return absl::UnimplementedError(
        "FillAudioEmbeddingsWithTensorBuffer not implemented for this "
        "backend.");
  }

  // Audio tensor should have shape [1, 1, x, emb_dim] where x is the number
  // of audio hard tokens. Audio tensor will be inserted into the media cache.
  // The media cache has size [1, 1, media_cache_size, emb_dim].
  // media_cache[0, 0, index:index+x, :] = audio_tensor
  virtual absl::Status FillAudioEmbeddings(
      ml_drift::GpuSpatialTensor& audio_tensor, int index) {
    return absl::UnimplementedError(
        "FillAudioEmbeddings not implemented for this backend.");
  }

  virtual absl::Status ClearVisionPle() {
    return absl::UnimplementedError(
        "ClearVisionPle not implemented for this backend.");
  }

  virtual absl::Status UpdateSamplerParams(
      const proto::SamplerParameters& new_params,
      std::shared_ptr<std::default_random_engine> rand_gen) = 0;

  // Updates the constraint mask for the given bitmap. The bitmap indicates the
  // allowed tokens during sampling. The function is used for constrained
  // decoding.
  virtual absl::Status UpdateConstraintMask(
      const llm_utils::Bitmap& constraint_bitmap) {
    // TODO: b/391468697 - This is called in LlmGpuCalculator even for backends
    // that don't yet support constrained decoding.
    return absl::OkStatus();
  }

  // Get the maximum possible Top-K value. Top-K provided to UpdateSamplerParams
  // should not be larger than max Top-K.
  virtual int GetMaxTopK() = 0;

  // Enables speculative decoding for the runner if supported, otherwise does
  // nothing.
  virtual absl::Status EnableSpeculativeDecoding(
      bool enable_speculative_decoding) {
    return absl::OkStatus();
  }

  // Speculative decoding methods used by both drafter and main model.

  // The Llm* runner class actually holds the next token to be predicted in
  // memory, ie. after every sampled token that is return, returned token is
  // always passed through the transformer stack again so there's always an
  // available token that is hidden in the Llm* class and not returned. This
  // function allows the caller to peek at that token. Returned is the time step
  // and token ID being sampled by the last forward path.
  virtual absl::StatusOr<ml_drift::speculative_utils::TokenAtTimeStep>
  PeekNextToken() {
    return absl::UnimplementedError(
        "PeekNextToken not implemented for this backend.");
  };

  // The Llm* class always holds the next sampled but not yet returned token,
  // that token is used as the basis for predicting the next token in both
  // GetNextTokens() and DraftTokens(). In order to ensure correct prediction,
  // we need to override that token to whatever the result of speculating
  // decoding is so that further drafting is based on correct input.
  virtual absl::Status OverwriteOneInputToken(
      const ml_drift::speculative_utils::TokenAtTimeStep& token) {
    return absl::UnimplementedError(
        "OverwriteOneInputToken not implemented for this backend.");
  }

  // Retrieve the top k probabilities and tokens for the corresponding time
  // step. This will only provide the probabilities after ScoreTokens() or
  // DraftTokens() finished.
  virtual absl::StatusOr<
      std::vector<ml_drift::speculative_utils::TokenProbability>>
  GetProbabilityCache(int time_step) {
    return absl::UnimplementedError(
        "GetProbabilityCache not implemented for this backend.");
  }

  // Speculative decoding methods used by drafter only.

  // Attempts to draft gamma tokens. This is similar to GetNextTokens() but:
  // - number of output tokens is limited to gamma+1.
  // - The probability distribution is available from GetTopKProbabilityCache()
  virtual absl::StatusOr<std::vector<int>> DraftTokens() {
    return absl::UnimplementedError(
        "DraftTokens not implemented for this backend.");
  }

  // Speculative decoding methods used by main model only.

  // Passes these tokens through the LLM stack in 1 pass, the single batch
  // will go through the transformer stack, projection (Fully Connected Linear)
  // and sampling. The tokens is a vector of Token ID.
  virtual absl::Status ScoreTokens(const std::vector<int>& tokens) {
    return absl::UnimplementedError(
        "ScoreTokens not implemented for this backend.");
  }

  // Reject the token at time_step, and the correct (resampled value) at
  // time_step is token_index. This function will ensure that the next
  // invocation of PeekNextToken() returns the correct value.
  // Note that this is pretty much OverwriteOneInputToken() bundled with
  // CleanCache() and overwriting ready_tokens_. The reason why we need both
  // this and OverwriteOneInputToken() is that OverwriteOneInputToken() is
  // useful in cases such when the whole round is speculated correctly but the
  // last (not yet output) token still needs to be overwritten without
  // adjustment to the progress of current LLM stack.
  virtual absl::Status RejectAfterResample(
      const ml_drift::speculative_utils::TokenAtTimeStep& token) {
    return absl::UnimplementedError(
        "RejectAfterResample not implemented for this backend.");
  }

  virtual absl::StatusOr<RunnerEnvironment> GetRunnerEnvironment() {
    return absl::UnimplementedError(
        "GetExecutionEnvironment not implemented for this backend.");
  }

#ifdef ML_DRIFT_PROFILING
  // Virtual classes to be overridden by the OpenCL backend, returns nullptr
  // otherwise. For profiling use.
  virtual ml_drift::cl::Environment* GetEnv() const {
    ml_drift::cl::Environment* env = nullptr;
    return env;
  };
  virtual ml_drift::cl::InferenceContext* GetInferenceContext() const {
    return nullptr;
  };
  virtual absl::StatusOr<ml_drift::cl::InferenceContext*>
  GetInferenceContextForSequenceSize(int sequence_size) const {
    return absl::UnimplementedError(
        "GetInferenceContextForSequenceSize not implemented for this backend.");
  };
#endif  // ML_DRIFT_PROFILING
};

}  // namespace odml::infra::gpu

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_LLM_GPU_RUNNER_H_
