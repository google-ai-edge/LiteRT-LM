// Copyright 2026 The ODML Authors.
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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2IMAGE_TEXT2IMAGE_ENGINE_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2IMAGE_TEXT2IMAGE_ENGINE_H_

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <variant>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "omni/base/io_types.h"
#include "omni/base/model_resources.h"
#include "omni/multi_staged_session.h"
#include "omni/text2image/flux2/flux2_model_config.h"
#include "omni/text2image/prompt_source.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/framework/threadpool.h"

namespace litert::omni::text2image {

// Supported text-to-image generation model architectures.
enum class ModelType {
  UNSPECIFIED = 0,
  // 3-graph FLUX.2-klein-4B / Bonsai-Image-ternary-4B pipeline
  // (textenc + dit + vae_dec).
  BONSAI_FLUX2 = 1,
};

// Model-specific configuration variant.
using ModelConfig = std::variant<std::monostate, Flux2ModelConfig>;

// Determines the text-to-image model architecture from the `.litertlm` model
// container or `.tflite` model files found under `model_folder` (e.g., a
// `.litertlm` container with `ImageGenMetadata` or "textenc_*.tflite" /
// "dit_*.tflite" / "vae_dec_*.tflite" -> `ModelType::BONSAI_FLUX2`).
absl::StatusOr<ModelType> DetectModelType(absl::string_view model_folder);

// High-level Text-to-Image Engine owning compiled model resources and
// creating lightweight `MultiStagedSession` instances.
class Text2ImageEngine {
 public:
  // Configuration settings for `Text2ImageEngine` initialization.
  struct Settings {
    // Folder or file path containing the text-to-image model files.
    std::string model_folder;
    // Optional cache directory for model acceleration (e.g., XNNPACK weight
    // cache).
    std::string cache_dir;
    // Default backend to use for model execution.
    lm::Backend backend = lm::Backend::CPU;
    // Number of threads to use for CPU model execution.
    int num_threads = 4;

    // Model-specific configuration. If left unset (`std::monostate`), the model
    // type is detected from `model_folder` in `Text2ImageEngine::Create` and
    // the default configuration of the detected model is used.
    ModelConfig model_config;

    ModelType GetModelType() const {
      if (std::holds_alternative<Flux2ModelConfig>(model_config)) {
        return ModelType::BONSAI_FLUX2;
      }
      return ModelType::UNSPECIFIED;
    }
  };

  // Configuration settings for a text-to-image session instance.
  struct SessionSettings {
    // Target output image width and height in pixels. If <= 0, defaults to the
    // model's configured image size.
    int width = 0;
    int height = 0;
    // Number of denoising steps. If <= 0, defaults to the model's configured
    // step count.
    int num_inference_steps = 0;
    // Default random seed for initial latent noise.
    uint64_t seed = 42;
  };

  static absl::StatusOr<std::unique_ptr<Text2ImageEngine>> Create(
      const Settings& settings);

  ~Text2ImageEngine() = default;

  // Resolves the default `ImageGenInputMetadata` parameters for
  // `session_settings`, applying any model-specific defaults.
  ImageGenInputMetadata ResolveDefaultPromptParams(
      const SessionSettings& session_settings) const;

  // Creates a lightweight `MultiStagedSession` for image generation.
  // The first stage (`stages()[0]`) is guaranteed to be a `PromptSource`.
  absl::StatusOr<std::unique_ptr<MultiStagedSession>> CreateSession(
      const SessionSettings& session_settings,
      std::unique_ptr<PromptSource> absl_nullable prompt_source = nullptr);

  const Settings& settings() const { return settings_; }
  std::shared_ptr<ModelResources> model_resources() const {
    return model_resources_;
  }

 private:
  friend struct Text2ImageEngineTestingPeer;

  Text2ImageEngine(const Settings& settings,
                   std::shared_ptr<ModelResources> resources,
                   std::unique_ptr<lm::ThreadPool> thread_pool)
      : settings_(settings),
        model_resources_(std::move(resources)),
        thread_pool_(std::move(thread_pool)) {}

  Settings settings_;
  std::shared_ptr<ModelResources> model_resources_;
  std::unique_ptr<lm::ThreadPool> thread_pool_;
};

}  // namespace litert::omni::text2image

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2IMAGE_TEXT2IMAGE_ENGINE_H_
