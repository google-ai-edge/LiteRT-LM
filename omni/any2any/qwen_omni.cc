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
#include <vector>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_options.h"  // from @litert
#include "omni/any2any/any2any_session.h"
#include "omni/any2any/qwen_omni/qwen_omni_factory.h"
#include "omni/any2any/qwen_omni/qwen_omni_model_config.h"
#include "omni/base/io_types.h"
#include "omni/base/model_resources.h"
#include "omni/base/model_utils.h"
#include "omni/base/stage.h"
#include "omni/omni_engine.h"
#include "omni/omni_registry.h"
#include "omni/omni_session.h"
#include "runtime/components/model_resources.h"
#include "runtime/proto/any2any_metadata.pb.h"
#include "runtime/proto/any2any_model_type.pb.h"
#include "runtime/proto/any2any_session_config.pb.h"

namespace litert::omni::any2any {
namespace {

constexpr absl::string_view kModelNames[] = {
    "qwen2.5-omni-3b", "qwen2.5-omni", "qwen2_5_omni",
    "qwen-omni",       "qwen_omni",    "Qwen/Qwen2.5-Omni-3B",
};

absl::StatusOr<std::unique_ptr<OmniSessionFactory>> GetQwenOmniSessionFactory(
    absl::string_view model_name, const OmniEngine::Settings& settings,
    const MessageLite* absl_nullable session_config,
    const MessageLite* absl_nullable metadata, ::litert::Environment& env,
    ::litert::Options& /*options*/,
    std::shared_ptr<lm::ModelResources> absl_nullable lm_resources) {
  if (!absl::c_linear_search(kModelNames, model_name)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Unexpected model_name: ", model_name));
  }
  if (lm_resources == nullptr &&
      absl::EndsWithIgnoreCase(settings.model_path, ".litertlm")) {
    ABSL_ASSIGN_OR_RETURN(lm_resources,
                          CreateLmModelResources(settings.model_path));
  }

  lm::proto::Any2AnyMetadata resolved_metadata =
      ResolveAny2AnyMetadata(metadata);
  resolved_metadata.mutable_any2any_model_type()->mutable_qwen_omni();
  QwenOmniModelConfig config;
  config.force_fp32_on_gpu = settings.force_fp32_on_gpu;
  PopulateQwenOmniConfigFromProto(resolved_metadata, config);
  if (session_config != nullptr) {
    lm::proto::Any2AnySessionConfig init_session_cfg;
    init_session_cfg.CheckTypeAndMergeFrom(*session_config);
    ApplySessionConfigToQwenOmniConfig(init_session_cfg, config);
  }

  auto shared_env = std::make_shared<Environment>(std::move(env));
  const int num_threads = settings.num_threads > 0 ? settings.num_threads : 4;
  ABSL_ASSIGN_OR_RETURN(
      ModelResources model_resources,
      CreateQwenOmniResources(config, settings.model_path, settings.cache_dir,
                              ToLmBackend(settings.backend), num_threads,
                              std::move(shared_env), std::move(lm_resources)));
  auto resources = std::make_shared<ModelResources>(std::move(model_resources));

  std::vector<std::string> available_voices;
  if (!config.voice_name.empty()) {
    available_voices.push_back(config.voice_name);
  }
  for (absl::string_view supported_voice : {"Chelsie", "Ethan"}) {
    if (!absl::EqualsIgnoreCase(config.voice_name, supported_voice)) {
      available_voices.emplace_back(supported_voice);
    }
  }
  LeasedAny2AnySessionOptions lease_options{
      .model_display_name = "Qwen-Omni",
      .fallback_default_voice = config.voice_name,
  };
  LeasedAny2AnySessionFactory::ComponentBuilder component_builder =
      [base_config = std::move(config)](
          const lm::proto::Any2AnySessionConfig& effective_config,
          std::unique_ptr<OmniSession::InputSource> absl_nonnull input_source,
          std::shared_ptr<ModelResources> absl_nonnull model_resources,
          std::vector<std::unique_ptr<internal::StageBase>>& stages)
      -> absl::StatusOr<Stage<Output>* absl_nonnull> {
    QwenOmniModelConfig session_model_config = base_config;
    ApplySessionConfigToQwenOmniConfig(effective_config, session_model_config);
    return CreateQwenOmniComponents(session_model_config,
                                    std::move(input_source),
                                    std::move(model_resources), stages);
  };

  return std::make_unique<LeasedAny2AnySessionFactory>(
      settings, session_config, resolved_metadata, std::move(available_voices),
      std::move(resources), std::move(component_builder),
      std::move(lease_options));
}

[[maybe_unused]] const bool kQwenOmniRegistered = [] {
  ABSL_CHECK_OK(OmniRegistry::Get().RegisterSessionFactory(RegistrySpec{
      .model_names = {std::begin(kModelNames), std::end(kModelNames)},
      .get_session_factory = &GetQwenOmniSessionFactory,
      .session_config_type_id = "Any2Any",
      .model_remote_url =
          "https://huggingface.co/litert-community/Qwen2.5-Omni-3B/resolve/"
          "main/qwen2_5_omni_3b_int8.litertlm",
  }));
  return true;
}();

}  // namespace
}  // namespace litert::omni::any2any
