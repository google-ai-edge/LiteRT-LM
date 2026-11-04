// Copyright 2025 The ODML Authors.
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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_VISION_LITERT_COMPILED_MODEL_EXECUTOR_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_VISION_LITERT_COMPILED_MODEL_EXECUTOR_H_

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/functional/function_ref.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_expected.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/cancellation_latch.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/executor_stats.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/executor/vision/vision_executor.h"
#include "runtime/executor/vision/vision_executor_settings.h"

namespace litert::lm {

// The Vision Executor that uses the LiteRT CompiledModel to run the vision
// encoder and vision adapter models to encode the image tensor into vision
// soft token embeddings.
class VisionLiteRtCompiledModelExecutor : public VisionExecutor {
 public:
  // Creates a VisionLiteRtCompiledModelExecutor from the given
  // VisionExecutorSettings.
  static absl::StatusOr<std::unique_ptr<VisionLiteRtCompiledModelExecutor>>
  Create(const VisionExecutorSettings& vision_executor_settings,
         Environment& env);

  // Creates a VisionLiteRtCompiledModelExecutor from the given
  // VisionExecutorSettings and ModelResources. `resources` and `env` must
  // outlive the executor.
  static absl::StatusOr<std::unique_ptr<VisionLiteRtCompiledModelExecutor>>
  Create(const VisionExecutorSettings& vision_executor_settings,
         Environment& env, ModelResources& resources);

  // Creates a VisionLiteRtCompiledModelExecutor from the given
  // VisionExecutorSettings and ModelResources, sharing the ownership of
  // `resources`.
  //
  // The same resources can be shared with other executors (e.g. an
  // `EmbeddingLiteRtCompiledModelExecutor` created from the same model), so
  // that each executor can be created and destroyed independently: the
  // resources are released when the last executor using them is destroyed.
  // `env` must outlive the executor.
  //
  // Note: `ModelResources` is not thread-safe, and `Create()` mutates
  // `*resources` (e.g. it lazily loads the models). It must not be called
  // concurrently with other users of the same `ModelResources` instance across
  // threads without external synchronization.
  static absl::StatusOr<std::unique_ptr<VisionLiteRtCompiledModelExecutor>>
  Create(const VisionExecutorSettings& vision_executor_settings,
         Environment& env, std::shared_ptr<ModelResources> resources);

  // Encodes the input image tensor into vision embeddings.
  // Args:
  //   - input_image_tensor: The input image tensor to encode.
  // Returns:
  //   A ExecutorVisionData object containing the vision embeddings if
  //   successful, or an error status if failed.
  absl::StatusOr<ExecutorVisionData> Encode(
      const litert::TensorBuffer& input_image_tensor) override;

  absl::StatusOr<ExecutorVisionData> Encode(
      const absl::flat_hash_map<std::string, litert::TensorBuffer>& input_maps)
      override;

  // Returns the expected input dimension of the vision encoder model.
  absl::StatusOr<std::vector<int>> GetExpectedInputDimension() const override;

  // Returns the vision executor properties.
  absl::StatusOr<VisionExecutorProperties> GetVisionExecutorProperties()
      const override;

  absl::Status StartProfiling() override;
  absl::StatusOr<ExecutorStats> StopProfiling() override;

  // Same as the Encode overloads above, but can be cancelled while they run.
  //
  // `should_cancel` is checked before any input is copied, before the vision
  // encoder and the vision adapter run, and, for the duration of the call
  // only, after every op executed by the LiteRT interpreter. Once it returns
  // true, the call stops at the next check; `should_cancel` is not called again
  // for this call. An op delegated to an accelerator (e.g. the NPU dispatch op
  // running a whole compiled graph) cannot be interrupted, so cancellation
  // takes effect at op boundaries. Calls without `should_cancel` do not pay the
  // per-op check.
  //
  // Args:
  //   input_image_tensor / input_maps: The image input, as for the overloads
  //     above.
  //   should_cancel: Returns true to request cancellation. Called on the thread
  //     running this method.
  //
  // Returns:
  //   The vision embeddings, a CancelledError if `should_cancel` returned true,
  //   or the errors of the overloads above. The executor remains usable after a
  //   cancelled call.
  absl::StatusOr<ExecutorVisionData> Encode(
      const litert::TensorBuffer& input_image_tensor,
      absl::FunctionRef<bool()> should_cancel);
  absl::StatusOr<ExecutorVisionData> Encode(
      const absl::flat_hash_map<std::string, litert::TensorBuffer>& input_maps,
      absl::FunctionRef<bool()> should_cancel);

 private:
  // Caches the input and output buffers of each signature across Encode()
  // calls.
  class SignatureBufferCache {
   public:
    // Returns the cached input buffers for `signature_index`, creating them on
    // first use.
    Expected<std::vector<TensorBuffer>&> GetOrCreateInputBuffers(
        CompiledModel& compiled_model, int signature_index);

    // Returns the cached output buffers for `signature_index`, creating them on
    // first use.
    Expected<std::vector<TensorBuffer>&> GetOrCreateOutputBuffers(
        CompiledModel& compiled_model, int signature_index);

   private:
    absl::flat_hash_map<int, std::vector<TensorBuffer>> input_buffers_;
    absl::flat_hash_map<int, std::vector<TensorBuffer>> output_buffers_;
  };

  // The Vision Encoder LiteRT CompiledModel wrapper manage the input and
  // output buffers of the vision encoder model. It is not expected to be used
  // directly by the user. It is used by the VisionLiteRtCompiledModelExecutor
  // to encode the image tensor into vision embeddings. The user should use the
  // VisionLiteRtCompiledModelExecutor instead.
  class VisionEncoder {
   public:
    // Create an VisionEncoder to run vision encoder LiteRT CompiledModel.
    // Args:
    //   - model: The vision encoder model.
    //   - env: The LiteRT environment.
    // Returns:
    //   A unique pointer to the VisionEncoder if successful, or an error status
    //   if failed.
    static absl::StatusOr<std::unique_ptr<VisionEncoder>> Create(
        Environment& env, const Model* absl_nonnull model,
        const VisionExecutorSettings& vision_executor_settings,
        const VisionExecutorProperties& vision_executor_properties,
        ModelResources& resources);

    // Initialize the VisionEncoder, which will create the input and output
    // buffers for the vision encoder model.
    absl::Status Initialize(ModelResources& resources);

    // Returns the CompiledModel for the vision encoder model.
    const CompiledModel& GetCompiledModel() const { return compiled_model_; }

    // Returns the mutable CompiledModel for the vision encoder model.
    CompiledModel& GetMutableCompiledModel() { return compiled_model_; }

    // Returns the LiteRT model for the vision encoder model.
    const Model& GetModel() const { return model_; }

    // Returns the cached input buffers for `signature_index`, creating them on
    // first use.
    Expected<std::vector<TensorBuffer>&> GetOrCreateInputBuffers(
        int signature_index) {
      return buffer_cache_.GetOrCreateInputBuffers(compiled_model_,
                                                   signature_index);
    }

    // Returns the cached output buffers for `signature_index`, creating them on
    // first use.
    Expected<std::vector<TensorBuffer>&> GetOrCreateOutputBuffers(
        int signature_index) {
      return buffer_cache_.GetOrCreateOutputBuffers(compiled_model_,
                                                    signature_index);
    }

   private:
    VisionEncoder(Environment& env, const Model* absl_nonnull model,
                  const VisionExecutorSettings& vision_executor_settings,
                  const VisionExecutorProperties& vision_executor_properties)
        : env_(env),
          vision_executor_settings_(vision_executor_settings),
          model_(*model),
          vision_executor_properties_(vision_executor_properties) {
      backend_ = vision_executor_settings.GetEncoderBackend();
    }

    // The LiteRT environment.
    Environment& env_;

    // The VisionExecutorSettings for the vision encoder model.
    const VisionExecutorSettings& vision_executor_settings_;

    // The backend to use for the vision encoder model.
    Backend backend_;

    // The vision encoder model.
    const Model& model_;

    // The vision executor properties.
    const VisionExecutorProperties& vision_executor_properties_;

    // The vision encoder compiled model.
    CompiledModel compiled_model_;

    // Cached input and output buffers for the vision encoder model.
    SignatureBufferCache buffer_cache_;
  };

  // The Vision Adapter LiteRT CompiledModel wrapper manage the input and
  // output buffers of the vision adapter model. It is not expected to be used
  // directly by the user. It is used by the VisionLiteRtCompiledModelExecutor
  // to encode the vision embeddings into vision soft tokens. The user should
  // use the VisionLiteRtCompiledModelExecutor instead.
  class VisionAdapter {
   public:
    // Create an VisionAdapter to run vision adapter LiteRT CompiledModel.
    // Args:
    //   - model: The vision adapter model.
    //   - env: The LiteRT environment.
    // Returns:
    //   A unique pointer to the VisionAdapter if successful, or an error status
    //   if failed.
    static absl::StatusOr<std::unique_ptr<VisionAdapter>> Create(
        Environment& env, const Model* absl_nonnull model,
        const VisionExecutorSettings& vision_executor_settings,
        const VisionExecutorProperties& vision_executor_properties,
        ModelResources& resources);

    // Initialize the VisionAdapter.
    absl::Status Initialize(ModelResources& resources);

    // Returns the CompiledModel for the vision adapter model.
    const CompiledModel& GetCompiledModel() const { return compiled_model_; }

    // Returns the mutable CompiledModel for the vision adapter model.
    CompiledModel& GetMutableCompiledModel() { return compiled_model_; }

    // Returns the LiteRT model for the vision adapter model.
    const Model& GetModel() const { return model_; }

    // Returns the cached input buffers for `signature_index`, creating them on
    // first use.
    Expected<std::vector<TensorBuffer>&> GetOrCreateInputBuffers(
        int signature_index) {
      return buffer_cache_.GetOrCreateInputBuffers(compiled_model_,
                                                   signature_index);
    }

    // Returns the cached output buffers for `signature_index`, creating them on
    // first use.
    Expected<std::vector<TensorBuffer>&> GetOrCreateOutputBuffers(
        int signature_index) {
      return buffer_cache_.GetOrCreateOutputBuffers(compiled_model_,
                                                    signature_index);
    }

   private:
    VisionAdapter(Environment& env, const Model* absl_nonnull model,
                  const VisionExecutorSettings& vision_executor_settings,
                  const VisionExecutorProperties& vision_executor_properties)
        : env_(env),
          vision_executor_settings_(vision_executor_settings),
          model_(*model),
          vision_executor_properties_(vision_executor_properties) {
      backend_ = vision_executor_settings.GetAdapterBackend();
    }

    // The LiteRT environment.
    Environment& env_;

    // The VisionExecutorSettings for the vision adapter model.
    const VisionExecutorSettings& vision_executor_settings_;

    // The backend to use for the vision adapter model.
    Backend backend_;

    // The vision adapter model.
    const Model& model_;

    // The vision executor properties.
    const VisionExecutorProperties& vision_executor_properties_;

    // The vision adapter compiled model.
    CompiledModel compiled_model_;

    // Cached input and output buffers for the vision adapter model.
    SignatureBufferCache buffer_cache_;
  };

  explicit VisionLiteRtCompiledModelExecutor(
      const VisionExecutorSettings& vision_executor_settings, Environment& env,
      std::shared_ptr<ModelResources> resources,
      std::unique_ptr<VisionEncoder> vision_encoder,
      std::unique_ptr<VisionAdapter> vision_adapter,
      std::vector<int> expected_input_dimension,
      const VisionExecutorProperties& vision_executor_properties)
      : vision_executor_settings_(vision_executor_settings),
        env_(env),
        resources_(std::move(resources)),
        vision_encoder_(std::move(vision_encoder)),
        vision_adapter_(std::move(vision_adapter)),
        expected_input_dimension_(expected_input_dimension),
        vision_executor_properties_(vision_executor_properties) {}

  // Implement the Encode() overloads, cancellable with `latch` if not null
  // (see RunCompiledModel).
  absl::StatusOr<ExecutorVisionData> EncodeImageTensor(
      const litert::TensorBuffer& input_image_tensor, CancellationLatch* latch);
  absl::StatusOr<ExecutorVisionData> EncodeInputMaps(
      const absl::flat_hash_map<std::string, litert::TensorBuffer>& input_maps,
      CancellationLatch* latch);

  // Runs `encode` as a cancellable call, with the latch of the call.
  absl::StatusOr<ExecutorVisionData> RunCancellable(
      absl::FunctionRef<bool()> should_cancel,
      absl::FunctionRef<absl::StatusOr<ExecutorVisionData>(CancellationLatch*)>
          encode);

  // Runs the signature `signature_index` of `compiled_model` and records its
  // latency as `latency_metric`.
  //
  // Args:
  //   compiled_model: The vision encoder or vision adapter compiled model.
  //   signature_index: The signature to run.
  //   input_buffers / output_buffers: The buffers of the signature.
  //   latency_metric: The latency metric of the run.
  //   latch: The latch of a cancellable call, or nullptr. If set, the call is
  //     cancelled before the run, and an interrupted run is reported as
  //     cancelled.
  //
  // Returns:
  //   OK, a CancelledError if `latch` is cancelled, or the error of the run.
  absl::Status RunCompiledModel(
      const CompiledModel& compiled_model, size_t signature_index,
      absl::Span<const litert::TensorBuffer> input_buffers,
      absl::Span<const litert::TensorBuffer> output_buffers,
      absl::string_view latency_metric, CancellationLatch* latch);

  // The VisionExecutorSettings for the vision encoder and vision adapter
  // models.
  VisionExecutorSettings vision_executor_settings_;

  // The LiteRT environment.
  Environment& env_;

  // The model resources for the vision encoder and vision adapter models.
  std::shared_ptr<ModelResources> resources_;

  // The vision encoder and vision adapter models.
  std::unique_ptr<VisionEncoder> vision_encoder_;

  // The vision adapter model.
  std::unique_ptr<VisionAdapter> vision_adapter_;

  // The expected input dimension of the vision encoder model.
  std::vector<int> expected_input_dimension_;

  // The vision executor properties.
  VisionExecutorProperties vision_executor_properties_;

  std::optional<ExecutorStats> latency_stats_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_VISION_LITERT_COMPILED_MODEL_EXECUTOR_H_
