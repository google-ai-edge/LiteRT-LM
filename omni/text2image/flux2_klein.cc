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

#include <iterator>
#include <memory>
#include <string>
#include <utility>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_options.h"  // from @litert
#include "omni/base/io_types.h"
#include "omni/base/model_resources.h"
#include "omni/base/model_utils.h"
#include "omni/omni_engine.h"
#include "omni/omni_registry.h"
#include "omni/omni_session.h"
#include "omni/text2image/flux2/flux2_factory.h"
#include "omni/text2image/flux2/flux2_model_config.h"
#include "omni/text2image/text2image_session.h"
#include "runtime/components/model_resources.h"
#include "runtime/proto/image_gen_model_type.pb.h"
#include "runtime/proto/text2image_metadata.pb.h"

namespace litert::omni::text2image {
namespace {

constexpr absl::string_view kModelNames[] = {
    "flux2-klein",
    "flux2_klein",
    "flux.2-klein-4b",
};

lm::proto::Text2ImageMetadata ResolveMetadata(
    const MessageLite* absl_nullable metadata) {
  lm::proto::Text2ImageMetadata resolved;
  if (metadata != nullptr) {
    resolved.CheckTypeAndMergeFrom(*metadata);
  }
  resolved.mutable_image_gen_model_type()->mutable_flux2_klein();
  return resolved;
}

absl::StatusOr<std::unique_ptr<OmniSessionFactory>> GetFlux2KleinSessionFactory(
    absl::string_view model_name, const OmniEngine::Settings& settings,
    const MessageLite* absl_nullable session_config,
    const MessageLite* absl_nullable metadata, ::litert::Environment& env,
    ::litert::Options& /*options*/,
    std::shared_ptr<lm::ModelResources> absl_nullable lm_resources) {
  if (!absl::c_linear_search(kModelNames, model_name)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Unexpected model_name: ", model_name));
  }
  lm::proto::Text2ImageMetadata resolved_metadata = ResolveMetadata(metadata);
  Flux2ModelConfig config;
  config.is_klein = true;
  PopulateFlux2ConfigFromProto(
      resolved_metadata.image_gen_model_type().flux2_klein(), config);

  auto shared_env = std::make_shared<Environment>(std::move(env));
  auto resources = std::make_shared<ModelResources>(shared_env);
  if (lm_resources == nullptr && !settings.model_path.empty()) {
    const std::string litertlm_path = ResolveLitertLmPath(settings.model_path);
    if (!litertlm_path.empty()) {
      LITERT_ASSIGN_OR_RETURN(lm_resources,
                              CreateLmModelResources(litertlm_path));
    }
  }
  if (lm_resources != nullptr) {
    resources->SetLmModelResources(std::move(lm_resources));
  }

  const int num_threads = settings.num_threads > 0 ? settings.num_threads : 4;
  LITERT_RETURN_IF_ERROR(InitFlux2Resources(
      config, settings.model_path, settings.cache_dir,
      ToLmBackend(settings.backend), num_threads, *shared_env, *resources));

  LeasedText2ImageSessionOptions lease_options{
      .model_display_name = "FLUX.2-klein",
      .default_prompt_params =
          ImageGenInputMetadata{
              .width = config.img_size,
              .height = config.img_size,
              .num_inference_steps = config.steps,
              .seed = config.default_seed,
          },
  };
  return std::make_unique<LeasedText2ImageSessionFactory>(
      settings, session_config, resolved_metadata, std::move(resources),
      MakeFlux2ComponentBuilder(std::move(config), settings.model_path),
      std::move(lease_options));
}

[[maybe_unused]] const bool kFlux2KleinRegistered = [] {
  ABSL_CHECK_OK(OmniRegistry::Get().RegisterSessionFactory(RegistrySpec{
      .model_names = {std::begin(kModelNames), std::end(kModelNames)},
      .get_session_factory = &GetFlux2KleinSessionFactory,
      .session_config_type_id = "Text2Image",
      .model_remote_url =
          "https://huggingface.co/litert-community/FLUX.2-klein-4B-LiteRT/"
          "resolve/main/flux2-klein-4b.litertlm",
  }));
  return true;
}();

}  // namespace
}  // namespace litert::omni::text2image
