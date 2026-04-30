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
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/executor/vision_executor.h"
#include "runtime/executor/vision_executor_settings.h"

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

 private:
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
        const VisionExecutorProperties& vision_executor_properties);

    // Initialize the VisionEncoder, which will create the input and output
    // buffers for the vision encoder model.
    absl::Status Initialize();

    // Returns the CompiledModel for the vision encoder model.
    const CompiledModel& GetCompiledModel() const { return compiled_model_; }

    // Returns the mutable CompiledModel for the vision encoder model.
    CompiledModel& GetMutableCompiledModel() { return compiled_model_; }

    // Returns the LiteRT model for the vision encoder model.
    const Model& GetModel() const { return model_; }

    // Returns the input buffers for the vision encoder model.
    const std::vector<TensorBuffer>& GetInputBuffers() const {
      return input_buffers_;
    }

    // Returns the mutable input buffers for the vision encoder model.
    std::vector<TensorBuffer>& GetMutableInputBuffers() {
      return input_buffers_;
    }

    // Returns the output buffers for the vision encoder model.
    const std::vector<TensorBuffer>& GetOutputBuffers() const {
      return output_buffers_;
    }

    // Returns the mutable output buffers for the vision encoder model.
    std::vector<TensorBuffer>& GetMutableOutputBuffers() {
      return output_buffers_;
    }

    // Clears the input buffers for the vision encoder model.
    absl::Status ClearInputBuffers();

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

    // The input buffers for the vision encoder model.
    std::vector<TensorBuffer> input_buffers_;

    // The output buffers for the vision encoder model.
    std::vector<TensorBuffer> output_buffers_;
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
        const VisionExecutorProperties& vision_executor_properties);

    // Initialize the VisionAdapter.
    absl::Status Initialize();

    // Returns the CompiledModel for the vision adapter model.
    const CompiledModel& GetCompiledModel() const { return compiled_model_; }

    // Returns the mutable CompiledModel for the vision adapter model.
    CompiledModel& GetMutableCompiledModel() { return compiled_model_; }

    // Returns the LiteRT model for the vision adapter model.
    const Model& GetModel() const { return model_; }

    // Returns the input buffers for the vision adapter model.
    const std::vector<TensorBuffer>& GetInputBuffers() const {
      return input_buffers_;
    }

    // Returns the mutable input buffers for the vision adapter model.
    std::vector<TensorBuffer>& GetMutableInputBuffers() {
      return input_buffers_;
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

    // The input buffers for the vision adapter model.
    std::vector<TensorBuffer> input_buffers_;
  };

  explicit VisionLiteRtCompiledModelExecutor(
      const VisionExecutorSettings& vision_executor_settings, Environment& env,
      std::unique_ptr<ModelResources> resources,
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

  // The VisionExecutorSettings for the vision encoder and vision adapter
  // models.
  VisionExecutorSettings vision_executor_settings_;

  // The LiteRT environment.
  Environment& env_;

  // The model resources for the vision encoder and vision adapter models.
  std::unique_ptr<ModelResources> resources_;

  // The vision encoder and vision adapter models.
  std::unique_ptr<VisionEncoder> vision_encoder_;

  // The vision adapter model.
  std::unique_ptr<VisionAdapter> vision_adapter_;

  // The expected input dimension of the vision encoder model.
  std::vector<int> expected_input_dimension_;

  // The vision executor properties.
  VisionExecutorProperties vision_executor_properties_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_VISION_LITERT_COMPILED_MODEL_EXECUTOR_H_
