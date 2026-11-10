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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_LLM_GPU_ARTISAN_EXECUTOR_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_LLM_GPU_ARTISAN_EXECUTOR_H_

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/model_data_provider.h"
#include "runtime/components/model_resources.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/gpu_artisan/gpu_artisan_executor_utils.h"
#include "runtime/executor/gpu_artisan/ml_drift/llm_gpu_runner.h"
#include "runtime/executor/gpu_artisan/ml_drift/transformer.h"
#include "runtime/executor/llm_executor.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/executor/llm_executor_processed_tokens.h"
#include "runtime/executor/llm_executor_settings.h"

namespace odml::infra {

// GPU hand-written path's executor.
class LlmGpuArtisanExecutor : public ::litert::lm::LlmExecutor {
 public:
  // Creates a LlmGpuArtisanExecutor from the given executor settings and
  // model resources. This is a preferred way to create a
  // LlmGpuArtisanExecutor as it's implementation doesn't involve using legacy
  // libraries.
  static absl::StatusOr<std::unique_ptr<LlmGpuArtisanExecutor>> Create(
      const ::litert::lm::LlmExecutorSettings& executor_settings,
      ::litert::lm::ModelResources& model_resources);

  // Creates a LlmGpuArtisanExecutor from the given executor settings and
  // model data. This is a preferred way to create a
  // LlmGpuArtisanExecutor as it's implementation doesn't involve using legacy
  // libraries.
  //
  // `end_of_audio_embedding`: Embedding of the end-of-audio token. If empty,
  //   the end-of-audio token is looked up in the text embedding table instead.
  // `max_num_audio_tokens`: Number of media embedding cache slots to reserve
  //   for audio embeddings (including end-of-audio tokens) produced by an
  //   audio encoder that is not part of the artisan model, e.g. a LiteRT audio
  //   encoder. These slots come after the slots reserved for images.
  static absl::StatusOr<std::unique_ptr<LlmGpuArtisanExecutor>> Create(
      const ::litert::lm::LlmExecutorSettings& executor_settings,
      std::shared_ptr<odml::infra::llm_utils::ModelData> model_data,
      std::vector<float> end_of_vision_embedding = {},
      std::vector<float> end_of_audio_embedding = {},
      int max_num_audio_tokens = 0);

  // Computes the embedding for the end-of-vision token from the given
  // end-of-vision model. `embedding_dimension` is the text decoder's embedding
  // dimension.
  static absl::StatusOr<std::vector<float>> ComputeEndOfVisionEmbedding(
      const ::litert::Model* absl_nonnull end_of_vision_model,
      int embedding_dimension);

  // Computes the embedding for the end-of-audio token from the given
  // end-of-audio model. `embedding_dimension` is the text decoder's embedding
  // dimension.
  static absl::StatusOr<std::vector<float>> ComputeEndOfAudioEmbedding(
      const ::litert::Model* absl_nonnull end_of_audio_model,
      int embedding_dimension);

  // Sets the embedding for the end-of-vision token, which is inserted after
  // each image's vision embeddings. Use this when the end-of-vision model is
  // loaded after the executor is created (e.g. when streaming the model).
  void SetEndOfVisionEmbedding(std::vector<float> end_of_vision_embedding) {
    end_of_vision_embedding_ = std::move(end_of_vision_embedding);
  }

  // Sets the embedding for the end-of-audio token. Use this when the
  // end-of-audio model is loaded after the executor is created (e.g. when
  // streaming the model).
  void SetEndOfAudioEmbedding(std::vector<float> end_of_audio_embedding) {
    end_of_audio_embedding_ = std::move(end_of_audio_embedding);
  }

  // Default number of media embedding cache slots to reserve for audio
  // embeddings produced by an audio encoder that is not part of the artisan
  // model (capped by the max number of tokens). E.g. at one audio token per
  // 40 ms audio frame, this is ~80 seconds of audio.
  static constexpr int kDefaultMaxNumExternalAudioTokens = 2048;

  absl::Status Prefill(const ::litert::lm::ExecutorInputs& inputs) override;

  absl::Status Prefill(
      const ::litert::lm::ExecutorInputs& inputs,
      const ::litert::lm::ExecutorPrefillParams& prefill_params) override;

  absl::StatusOr<std::vector<std::vector<int>>> Decode() override;

  absl::StatusOr<std::vector<std::vector<int>>> Decode(
      const ::litert::lm::ExecutorDecodeParams& decode_params) override;

  absl::string_view ExecutorBackendName() const override {
    return "GpuArtisan";
  }

  absl::StatusOr<std::unique_ptr<::litert::lm::LlmContext>> CreateNewContext(
      std::optional<uint32_t> lora_id,
      ::litert::lm::RuntimeConfig runtime_config) const override;

  absl::StatusOr<std::unique_ptr<::litert::lm::LlmContext>> CloneContext()
      const override;

  absl::Status RestoreContext(
      std::unique_ptr<::litert::lm::LlmContext> llm_context) override;

  absl::StatusOr<std::unique_ptr<::litert::lm::ProcessedContext>> SaveContext()
      const override;

  absl::Status RestoreContext(std::unique_ptr<::litert::lm::ProcessedContext>
                                  processed_context) override;

  absl::Status UpdateRuntimeConfig(
      const ::litert::lm::RuntimeConfig& runtime_config) override;

  absl::StatusOr<::litert::lm::RuntimeConfig> GetRuntimeConfig() const override;

  absl::Status UpdateRuntimeState(
      const ::litert::lm::RuntimeState& runtime_state) override;

  absl::StatusOr<::litert::lm::RuntimeState> GetRuntimeState() const override;

  absl::StatusOr<int> GetCurrentStep() const override;

  absl::Status SetCurrentStep(int new_step) override;

  absl::StatusOr<const ::litert::lm::ProcessedTokens*> GetProcessedTokens()
      const override;

  // Clears the state of the model (e.g. KVCache). Loaded and used LoRA models
  // are not affected (remain loaded and in use).
  absl::Status Reset() override;

  // TODO: b/364902068 - Add unit test for LlmGpuArtisanExecutor by mocking
  // LlmGpuRunner.
  explicit LlmGpuArtisanExecutor(
      const ::litert::lm::LlmExecutorSettings& executor_settings,
      std::unique_ptr<gpu::LlmGpuRunner> llm_gpu_runner,
      const ml_drift::TransformerConfig& text_config,
      std::vector<float> end_of_vision_embedding = {},
      std::vector<float> end_of_audio_embedding = {})
      : executor_settings_(executor_settings),
        llm_gpu_runner_(std::move(llm_gpu_runner)),
        text_config_(text_config),
        end_of_vision_embedding_(std::move(end_of_vision_embedding)),
        end_of_audio_embedding_(std::move(end_of_audio_embedding)) {
    ::litert::lm::proto::SamplerParameters sampling_params;
    sampling_params.set_type(::litert::lm::proto::SamplerParameters::GREEDY);
    sampling_params.set_k(1);
    sampling_params.set_seed(0);
    llm_context_ =
        CreateNewContext(std::nullopt, ::litert::lm::RuntimeConfig({
                                           .sampler_params = sampling_params,
                                           .output_heads = 1,
                                           .tokens_per_decode = 1,
                                       }))
            .value();

    const char* backend_name = "Unknown";
    if (llm_gpu_runner_ != nullptr) {
      switch (llm_gpu_runner_->GetLlmGpuRunnerType()) {
        case gpu::LlmGpuRunnerType::kOpenCl:
          backend_name = "OpenCL";
          break;
        case gpu::LlmGpuRunnerType::kWebGpu:
        case gpu::LlmGpuRunnerType::kWebGpu_Emscripten:
          backend_name = "WebGPU";
          break;
        case gpu::LlmGpuRunnerType::kMetal:
          backend_name = "Metal";
          break;
        default:
          break;
      }
    }
    ABSL_VLOG(1) << "Selected Artisan Backend: " << backend_name << std::endl;
  };

  // This function will populate the GPU tensors with the vision embeddings and
  // vision per layer embeddings. This should only be used before the
  // prefill/prefix stage.
  // vision_input: The vision embeddings to populate the GPU tensors with.
  // image_index: The index of the image in the batch. It must be non-negative
  // and less than `max_num_images`. This will overwrite the vision embeddings
  // for the given image index if it is already populated.
  absl::Status FillVisionEmbeddings(
      const ::litert::lm::ExecutorVisionData& vision_input,
      int image_index) override;

  // Gets the executor settings of the model.
  absl::StatusOr<::litert::lm::LlmExecutorSettings> GetExecutorSettings()
      const override;

  // Gets the runner environment of the model.
  absl::StatusOr<odml::infra::gpu::RunnerEnvironment> GetRunnerEnvironment()
      const {
    if (llm_gpu_runner_) {
      return llm_gpu_runner_->GetRunnerEnvironment();
    }
    return absl::InternalError("LlmGpuRunner is not initialized.");
  }

 private:
  // The vision inputs of a prefill chunk.
  struct PrefillChunkVisionInputs {
    // The vision embeddings of all images in the prefill input, with shape
    // [1, 1, num_vision_tokens, embedding_dim]. Null if there is no vision
    // input.
    const ml_drift::TensorFloat32* absl_nullable embeddings = nullptr;
    // The vision per-layer embeddings of all images in the prefill input, each
    // with shape [1, 1, num_vision_tokens, ple_dim]. Empty if there are none.
    absl::Span<const ml_drift::TensorFloat32> per_layer_embeddings;
    // The images in the chunk.
    absl::Span<const ImageTokenInfo> image_infos;
    // The index of the first vision token of the chunk in `embeddings` and
    // `per_layer_embeddings`.
    int token_offset = 0;
  };

  // The audio inputs of a prefill chunk.
  struct PrefillChunkAudioInputs {
    // The projected audio embeddings for the audio tokens of the chunk, in
    // order, flattened from shape [num_embeddings, embedding_dimension]. There
    // may be more embeddings than audio tokens, e.g. due to padding. Empty if
    // there is no audio input.
    absl::Span<const float> embeddings;
    // The dimension of each audio embedding.
    int embedding_dimension = 0;
  };

  // Prefills a chunk of the input tokens. The chunk's images are filled into
  // the media embedding cache starting from slot 0, followed by its audio and
  // end-of-audio embeddings, so they must fit in the media embedding cache.
  //
  // args:
  //   input_ids: The token ids of the chunk.
  //   vision_inputs: The vision inputs of the chunk.
  //   audio_inputs: The audio inputs of the chunk.
  //   wait_for_completion: Whether to wait for the prefill to complete.
  //   cancel: The cancel flag, or null if the prefill cannot be cancelled.
  // returns:
  //   OkStatus on success or cancellation, or an error status if filling the
  //   media embeddings or running the prefill fails.
  absl::Status PrefillChunk(absl::Span<const int> input_ids,
                            const PrefillChunkVisionInputs& vision_inputs,
                            const PrefillChunkAudioInputs& audio_inputs,
                            bool wait_for_completion,
                            const std::atomic_bool* absl_nullable cancel);

  ::litert::lm::LlmExecutorSettings executor_settings_;

  // Updates the constraint state using the next token and updates the
  // gpu runner's constraint mask.
  absl::Status UpdateConstraintState(int next_token);

  std::unique_ptr<::litert::lm::LlmContext> llm_context_;

  // The LlmGpuRunner to run the inference.
  std::unique_ptr<gpu::LlmGpuRunner> llm_gpu_runner_;

  ml_drift::TransformerConfig text_config_;

  // This flag is used to indicate whether the executor is ready for decode.
  // If it is false, the executor will re-prefill the last token before decode.
  // This is to align with the behavior of how the current handwritten path
  // is implemented.
  bool ready_for_decode_ = false;

  // Embedding for eoi token if available, otherwise it will be empty.
  std::vector<float> end_of_vision_embedding_;

  // Embedding for the end-of-audio token if available, otherwise it will be
  // empty, and the end-of-audio token is looked up in the text embedding table.
  std::vector<float> end_of_audio_embedding_;
};

}  // namespace odml::infra

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_LLM_GPU_ARTISAN_EXECUTOR_H_
