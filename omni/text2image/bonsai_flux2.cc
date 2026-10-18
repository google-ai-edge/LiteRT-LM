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

#include <cstdint>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

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
#include "omni/base/stage.h"
#include "omni/multi_staged_session.h"
#include "omni/omni_engine.h"
#include "omni/omni_registry.h"
#include "omni/omni_session.h"
#include "omni/text2image/flux2/flux2_factory.h"
#include "omni/text2image/flux2/flux2_model_config.h"
#include "omni/text2image/prompt_source.h"
#include "omni/text2image/text2image_session.h"
#include "runtime/components/model_resources.h"
#include "runtime/proto/image_gen_model_type.pb.h"
#include "runtime/proto/text2image_metadata.pb.h"
#include "runtime/proto/text2image_session_config.pb.h"

namespace litert::omni::text2image {
namespace {

constexpr absl::string_view kModelNames[] = {
    "bonsai-flux2",
    "bonsai_flux2",
    "flux2",
};

lm::proto::Text2ImageMetadata ResolveMetadata(
    const MessageLite* absl_nullable metadata) {
  lm::proto::Text2ImageMetadata resolved;
  if (metadata != nullptr) {
    resolved.CheckTypeAndMergeFrom(*metadata);
  }
  resolved.mutable_image_gen_model_type()->mutable_bonsai_flux2();
  return resolved;
}

class BonsaiFlux2SessionFactory : public Text2ImageSessionFactory {
 public:
  BonsaiFlux2SessionFactory(
      const OmniEngine::Settings& settings,
      const MessageLite* absl_nullable session_config,
      const lm::proto::Text2ImageMetadata& metadata,
      std::shared_ptr<ModelResources> absl_nullable model_resources,
      Flux2ModelConfig config)
      : Text2ImageSessionFactory(settings, session_config, metadata,
                                 std::move(model_resources)),
        config_(std::move(config)) {}

  ImageGenInputMetadata ResolveDefaultPromptParams(
      const lm::proto::Text2ImageSessionConfig& session_config) const override {
    ImageGenInputMetadata params{
        .width = session_config.has_width() ? session_config.width()
                                            : this->session_config().width(),
        .height = session_config.has_height() ? session_config.height()
                                              : this->session_config().height(),
        .num_inference_steps =
            session_config.has_num_inference_steps()
                ? session_config.num_inference_steps()
                : this->session_config().num_inference_steps(),
        .seed =
            session_config.has_seed()
                ? static_cast<uint64_t>(session_config.seed())
                : (this->session_config().has_seed()
                       ? static_cast<uint64_t>(this->session_config().seed())
                       : config_.default_seed),
    };
    if (params.width <= 0) {
      params.width = config_.img_size;
    }
    if (params.height <= 0) {
      params.height = config_.img_size;
    }
    if (params.num_inference_steps <= 0) {
      params.num_inference_steps = config_.steps;
    }
    return params;
  }

 protected:
  absl::StatusOr<std::unique_ptr<MultiStagedSession>> CreateWithPromptSource(
      const lm::proto::Text2ImageSessionConfig& session_config,
      std::unique_ptr<PromptSource> absl_nullable prompt_source) override {
    lm::proto::Text2ImageSessionConfig effective_config =
        this->session_config();
    effective_config.MergeFrom(session_config);
    if (prompt_source == nullptr) {
      prompt_source = std::make_unique<PushPromptSource>(
          ResolveDefaultPromptParams(effective_config));
    }

    Flux2ModelConfig session_model_config = config_;
    if (effective_config.width() > 0 || effective_config.height() > 0) {
      if (effective_config.width() != effective_config.height()) {
        return absl::InvalidArgumentError(absl::StrCat(
            "FLUX.2 requires square image dimensions (width == height), got ",
            effective_config.width(), "x", effective_config.height()));
      }
      session_model_config.img_size = effective_config.width();
    }
    if (effective_config.num_inference_steps() > 0) {
      session_model_config.steps = effective_config.num_inference_steps();
    }
    if (effective_config.has_seed()) {
      session_model_config.default_seed = effective_config.seed();
    }

    std::vector<std::unique_ptr<internal::StageBase>> stages;
    Stage<Output>* output_stage = nullptr;
    LITERT_RETURN_IF_ERROR(CreateFlux2Components(
        session_model_config, engine_settings().model_path,
        std::move(prompt_source), model_resources(), stages, &output_stage));
    if (output_stage == nullptr) {
      return absl::InternalError("Output stage must not be null.");
    }
    return MultiStagedSession::Create(std::move(stages), output_stage,
                                      thread_pool());
  }

 private:
  Flux2ModelConfig config_;
};

absl::StatusOr<std::unique_ptr<OmniSessionFactory>>
GetBonsaiFlux2SessionFactory(
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
  PopulateFlux2ConfigFromProto(
      resolved_metadata.image_gen_model_type().bonsai_flux2(), config);

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

  return std::make_unique<BonsaiFlux2SessionFactory>(
      settings, session_config, resolved_metadata, std::move(resources),
      std::move(config));
}

[[maybe_unused]] const bool kBonsaiFlux2Registered = [] {
  ABSL_CHECK_OK(OmniRegistry::Get().RegisterSessionFactory(RegistrySpec{
      .model_names = {std::begin(kModelNames), std::end(kModelNames)},
      .get_session_factory = &GetBonsaiFlux2SessionFactory,
      .session_config_type_id = "Text2Image",
      .model_remote_url =
          "https://huggingface.co/litert-community/Bonsai-Image-ternary-4B/"
          "resolve/main/bonsai-flux2-4b.litertlm",
  }));
  return true;
}();

}  // namespace
}  // namespace litert::omni::text2image
