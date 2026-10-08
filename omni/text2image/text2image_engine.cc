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

#include "omni/text2image/text2image_engine.h"

#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "omni/base/io_types.h"
#include "omni/base/model_resources.h"
#include "omni/base/model_utils.h"
#include "omni/base/stage.h"
#include "omni/multi_staged_session.h"
#include "omni/text2image/flux2/flux2_factory.h"
#include "omni/text2image/flux2/flux2_model_config.h"
#include "omni/text2image/prompt_source.h"
#include "runtime/components/model_resources.h"
#include "runtime/framework/threadpool.h"
#include "runtime/proto/image_gen_metadata.pb.h"
#include "runtime/proto/image_gen_model_type.pb.h"

namespace litert::omni::text2image {
namespace {

absl::StatusOr<ModelType> DetectModelTypeFromLitertLm(
    lm::ModelResources& lm_resources) {
  auto image_gen_metadata = lm_resources.GetImageGenMetadata();
  if (image_gen_metadata.ok() && *image_gen_metadata != nullptr) {
    if ((*image_gen_metadata)->image_gen_model_type().has_bonsai_flux2()) {
      return ModelType::BONSAI_FLUX2;
    }
  }
  if (lm_resources
          .GetTFLiteModelBuffer(
              lm::proto::ImageGenMetadata::TF_LITE_IMAGE_DENOISER)
          .ok()) {
    return ModelType::BONSAI_FLUX2;
  }
  return ModelType::UNSPECIFIED;
}

}  // namespace

absl::StatusOr<ModelType> DetectModelType(absl::string_view model_folder) {
  if (model_folder.empty()) {
    return absl::InvalidArgumentError(
        "Text2ImageEngine::Settings::model_folder must not be empty.");
  }

  const std::string litertlm_path = ResolveLitertLmPath(model_folder);
  if (!litertlm_path.empty()) {
    LITERT_ASSIGN_OR_RETURN(auto lm_resources,
                            CreateLmModelResources(litertlm_path));
    LITERT_ASSIGN_OR_RETURN(const ModelType from_litertlm,
                            DetectModelTypeFromLitertLm(*lm_resources));
    if (from_litertlm != ModelType::UNSPECIFIED) {
      return from_litertlm;
    }
  }

  return absl::InvalidArgumentError(absl::StrCat(
      "Unable to determine the text2image model type from '", model_folder,
      "': expected a .litertlm container with ImageGenMetadata."));
}

absl::StatusOr<std::unique_ptr<Text2ImageEngine>> Text2ImageEngine::Create(
    const Settings& settings) {
  Settings resolved_settings = settings;
  std::shared_ptr<lm::ModelResources> lm_resources = nullptr;
  const std::string litertlm_path =
      ResolveLitertLmPath(resolved_settings.model_folder);
  if (!litertlm_path.empty()) {
    LITERT_ASSIGN_OR_RETURN(lm_resources,
                            CreateLmModelResources(litertlm_path));
  }

  if (resolved_settings.GetModelType() == ModelType::UNSPECIFIED) {
    ModelType model_type = ModelType::UNSPECIFIED;
    if (lm_resources != nullptr) {
      LITERT_ASSIGN_OR_RETURN(model_type,
                              DetectModelTypeFromLitertLm(*lm_resources));
    }
    if (model_type == ModelType::UNSPECIFIED) {
      LITERT_ASSIGN_OR_RETURN(model_type,
                              DetectModelType(resolved_settings.model_folder));
    }
    switch (model_type) {
      case ModelType::BONSAI_FLUX2:
        resolved_settings.model_config = Flux2ModelConfig{};
        break;
      case ModelType::UNSPECIFIED:
        return absl::InvalidArgumentError(
            absl::StrCat("Unable to determine the text2image model type from ",
                         resolved_settings.model_folder));
    }
  }

  LITERT_ASSIGN_OR_RETURN(auto env, Environment::Create({}));
  auto shared_env = std::make_shared<Environment>(std::move(env));
  auto resources = std::make_shared<ModelResources>(shared_env);
  if (lm_resources != nullptr) {
    resources->SetLmModelResources(lm_resources);
  }

  if (auto* config =
          std::get_if<Flux2ModelConfig>(&resolved_settings.model_config)) {
    LITERT_RETURN_IF_ERROR(InitFlux2Resources(
        *config, resolved_settings.model_folder, resolved_settings.cache_dir,
        resolved_settings.backend, resolved_settings.num_threads, *shared_env,
        *resources));
  } else {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported model_config in Text2ImageEngine::Settings: ",
                     static_cast<int>(resolved_settings.GetModelType())));
  }

  auto thread_pool = std::make_unique<lm::ThreadPool>(
      "text2image_engine_pool", resolved_settings.num_threads);

  return std::unique_ptr<Text2ImageEngine>(new Text2ImageEngine(
      resolved_settings, resources, std::move(thread_pool)));
}

ImageGenInputMetadata Text2ImageEngine::ResolveDefaultPromptParams(
    const SessionSettings& session_settings) const {
  ImageGenInputMetadata params{
      .width = session_settings.width,
      .height = session_settings.height,
      .num_inference_steps = session_settings.num_inference_steps,
      .seed = session_settings.seed,
  };
  if (const auto* config =
          std::get_if<Flux2ModelConfig>(&settings_.model_config)) {
    if (params.width <= 0) {
      params.width = config->img_size;
    }
    if (params.height <= 0) {
      params.height = config->img_size;
    }
    if (params.num_inference_steps <= 0) {
      params.num_inference_steps = config->steps;
    }
  }
  return params;
}

absl::StatusOr<std::unique_ptr<MultiStagedSession>>
Text2ImageEngine::CreateSession(
    const SessionSettings& session_settings,
    std::unique_ptr<PromptSource> absl_nullable prompt_source) {
  if (prompt_source == nullptr) {
    prompt_source = std::make_unique<PushPromptSource>(
        ResolveDefaultPromptParams(session_settings));
  }

  std::vector<std::unique_ptr<internal::StageBase>> stages;
  Stage<Output>* output_stage = nullptr;
  if (const auto* config =
          std::get_if<Flux2ModelConfig>(&settings_.model_config)) {
    Flux2ModelConfig session_model_config = *config;
    if (session_settings.width > 0 || session_settings.height > 0) {
      if (session_settings.width != session_settings.height) {
        return absl::InvalidArgumentError(absl::StrCat(
            "FLUX.2 requires square image dimensions (width == height), got ",
            session_settings.width, "x", session_settings.height));
      }
      session_model_config.img_size = session_settings.width;
    }
    if (session_settings.num_inference_steps > 0) {
      session_model_config.steps = session_settings.num_inference_steps;
    }
    session_model_config.default_seed = session_settings.seed;

    LITERT_RETURN_IF_ERROR(CreateFlux2Components(
        session_model_config, settings_.model_folder, std::move(prompt_source),
        model_resources_, stages, &output_stage));
  } else {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported model_config in Text2ImageEngine::Settings: ",
                     static_cast<int>(settings_.GetModelType())));
  }

  if (output_stage == nullptr) {
    return absl::InternalError("Output stage must not be null.");
  }
  return MultiStagedSession::Create(std::move(stages), output_stage,
                                    thread_pool_.get());
}

}  // namespace litert::omni::text2image
